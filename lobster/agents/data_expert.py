"""
Data Expert Agent for multi-omics data acquisition, processing, and workspace management.

This agent is responsible for managing all data-related operations using the modular
DataManagerV2 system, including GEO data fetching, local file processing, workspace
restoration, and multi-omics data integration with proper modality handling and
schema validation.
"""

from datetime import date
from typing import List, Optional

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.data_expert_assistant import DataExpertAssistant
from lobster.agents.state import DataExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.download_queue import DownloadStatus
from lobster.tools.download_orchestrator import DownloadOrchestrator
from lobster.tools.geo_download_service import GEODownloadService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def data_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "data_expert_agent",
    handoff_tools: List = None,
):
    """
    Create a multi-omics data acquisition, processing, and workspace management specialist agent.

    This expert agent serves as the primary interface for all data-related operations in the
    Lobster bioinformatics platform, specializing in:

    - **GEO Data Acquisition**: Fetching, validating, and downloading datasets from NCBI GEO
    - **Local File Processing**: Loading and validating custom data files with automatic format detection
    - **Workspace Management**: Restoring previous sessions and managing dataset persistence
    - **Multi-modal Integration**: Handling transcriptomics, proteomics, and other omics data types
    - **Quality Assurance**: Ensuring data integrity through schema validation and provenance tracking

    Built on the modular DataManagerV2 architecture, this agent provides seamless integration
    with downstream analysis workflows while maintaining professional scientific standards.

    Args:
        data_manager: DataManagerV2 instance for modular data operations
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        handoff_tools: Additional tools for inter-agent communication

    Returns:
        Configured ReAct agent with comprehensive data management capabilities
    """

    settings = get_settings()
    model_params = settings.get_agent_llm_params("data_expert_agent")
    llm = create_llm("data_expert_agent", model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    # Initialize the assistant for LLM operations
    assistant = DataExpertAssistant()

    # Initialize modality management service
    from lobster.tools.modality_management_service import ModalityManagementService

    modality_service = ModalityManagementService(data_manager)

    # Define tools for data operations
    @tool
    def execute_download_from_queue(
        entry_id: str,
        concatenation_strategy: str = "auto",
    ) -> str:
        """
        Execute download from queue entry prepared by research_agent.

        This tool implements the queue consumer pattern where:
        1. research_agent validates metadata and adds to queue (Task 2.2B)
        2. Supervisor queries queue and extracts entry_id
        3. data_expert downloads using queue entry metadata

        Args:
            entry_id: Download queue entry ID (format: queue_GSE12345_abc123)
            concatenation_strategy: How to merge samples ("auto"|"union"|"intersection")
                - 'auto' (RECOMMENDED): Intelligently decides based on DUAL CRITERIA
                  * CV criterion: If coefficient of variation > 20% → UNION
                  * Range criterion: If max/min gene ratio > 1.5x → UNION
                  * BOTH criteria must pass (CV ≤ 20% AND ratio ≤ 1.5x) for INTERSECTION
                - 'intersection': Keep only genes present in ALL samples (inner join)
                - 'union': Include all genes from all samples (outer join with zero-filling)

        Returns:
            Download report with modality name, status, and statistics
        """
        try:
            # 1. RETRIEVE QUEUE ENTRY
            if entry_id not in [
                e.entry_id for e in data_manager.download_queue.list_entries()
            ]:
                available = [
                    e.entry_id for e in data_manager.download_queue.list_entries()
                ]
                return (
                    f"Error: Queue entry '{entry_id}' not found. Available: {available}"
                )

            entry = data_manager.download_queue.get_entry(entry_id)

            # Verify entry is downloadable
            if entry.status == DownloadStatus.COMPLETED:
                return (
                    f"Entry '{entry_id}' already downloaded as '{entry.modality_name}'"
                )
            elif entry.status == DownloadStatus.IN_PROGRESS:
                return (
                    f"Entry '{entry_id}' is currently being downloaded by another agent"
                )

            # 2. UPDATE STATUS TO IN_PROGRESS
            data_manager.download_queue.update_status(
                entry_id=entry_id,
                status=DownloadStatus.IN_PROGRESS,
                downloaded_by="data_expert",
            )

            logger.info(
                f"Starting download for {entry.dataset_id} from queue entry {entry_id}"
            )

            # 3. DETERMINE DOWNLOAD STRATEGY
            # Use recommended_strategy from entry (if set by data_expert_assistant)
            # Otherwise use default strategy based on available URLs
            if entry.recommended_strategy:
                download_strategy = entry.recommended_strategy.strategy_type
                logger.info(f"Using recommended strategy: {download_strategy}")
            else:
                # Auto-determine strategy from available URLs
                if entry.h5_url:
                    download_strategy = "H5_FIRST"
                elif entry.matrix_url:
                    download_strategy = "MATRIX_FIRST"
                elif entry.supplementary_urls:
                    download_strategy = "SUPPLEMENTARY_FIRST"
                else:
                    download_strategy = "RAW_FIRST"
                logger.info(f"Auto-selected strategy: {download_strategy}")

            # 4. EXECUTE DOWNLOAD USING GEO SERVICE
            # Import GEO service
            from lobster.tools.geo_service import GEOService

            geo_service = GEOService(data_manager=data_manager)

            try:
                # Map concatenation_strategy to use_intersecting_genes_only parameter
                if concatenation_strategy == "auto":
                    use_intersecting = None  # Triggers intelligent auto-detection
                elif concatenation_strategy == "intersection":
                    use_intersecting = True  # Inner join - only common genes
                elif concatenation_strategy == "union":
                    use_intersecting = False  # Outer join - all genes
                else:
                    logger.warning(
                        f"Unknown concatenation_strategy '{concatenation_strategy}', defaulting to 'auto'"
                    )
                    use_intersecting = None

                # Download using GEOService.download_dataset()
                # Note: GEOService internally stores modality in data_manager and returns success message
                result_message = geo_service.download_dataset(
                    geo_id=entry.dataset_id,
                    manual_strategy_override=download_strategy,
                    use_intersecting_genes_only=use_intersecting,
                )

                # Check if download was successful (result_message contains success indicators)
                if "Failed" in result_message or "Error" in result_message:
                    raise Exception(result_message)

                # 5. RETRIEVE STORED MODALITY FROM DATA MANAGER
                # GEOService stores modality with naming pattern: geo_{gse_id}_{adapter}
                # We need to find the modality that matches our dataset_id
                modalities = data_manager.list_modalities()
                modality_name = None

                # Look for modality matching pattern geo_{dataset_id}
                dataset_pattern = f"geo_{entry.dataset_id.lower()}"
                for mod_name in modalities:
                    if mod_name.startswith(dataset_pattern):
                        modality_name = mod_name
                        break

                if not modality_name:
                    raise Exception(
                        f"Modality not found after download. Expected pattern: {dataset_pattern}_*"
                    )

                result_adata = data_manager.get_modality(modality_name)

                # Log provenance
                data_manager.log_tool_usage(
                    tool_name="execute_download_from_queue",
                    parameters={
                        "entry_id": entry_id,
                        "dataset_id": entry.dataset_id,
                        "strategy": download_strategy,
                        "concatenation_strategy": concatenation_strategy,
                        "modality_name": modality_name,
                        "n_obs": result_adata.n_obs,
                        "n_vars": result_adata.n_vars,
                    },
                )

                # 6. UPDATE QUEUE STATUS TO COMPLETED
                data_manager.download_queue.update_status(
                    entry_id=entry_id,
                    status=DownloadStatus.COMPLETED,
                    modality_name=modality_name,
                )

                logger.info(f"Download complete: {entry.dataset_id} → {modality_name}")

                # 7. RETURN SUCCESS REPORT
                response = f"## Download Complete: {entry.dataset_id}\n\n"
                response += "✅ **Status**: Downloaded successfully\n"
                response += f"- **Modality name**: `{modality_name}`\n"
                response += f"- **Samples**: {result_adata.n_obs}\n"
                response += f"- **Features**: {result_adata.n_vars}\n"
                response += f"- **Strategy**: {download_strategy}\n"
                response += f"- **Concatenation**: {concatenation_strategy}\n"
                response += f"- **Queue entry**: `{entry_id}` (COMPLETED)\n"
                response += f"\n**Available for analysis**: Use `get_modality_overview('{modality_name}')` to inspect\n"

                return response

            except Exception as download_error:
                # 8. UPDATE QUEUE STATUS TO FAILED
                error_msg = str(download_error)
                data_manager.download_queue.update_status(
                    entry_id=entry_id,
                    status=DownloadStatus.FAILED,
                    error=error_msg,
                )

                logger.error(f"Download failed for {entry.dataset_id}: {error_msg}")

                response = f"## Download Failed: {entry.dataset_id}\n\n"
                response += "❌ **Status**: Download failed\n"
                response += f"- **Error**: {error_msg}\n"
                response += f"- **Queue entry**: `{entry_id}` (FAILED)\n"
                response += "\n**Troubleshooting**:\n"
                response += f"1. Check GEO dataset availability: {entry.dataset_id}\n"
                response += "2. Verify URLs are accessible\n"
                response += "3. Review error log in queue entry\n"

                return response

        except Exception as e:
            logger.error(f"Error in execute_download_from_queue: {e}")
            return f"Error processing queue entry '{entry_id}': {str(e)}"

    @tool
    def retry_failed_download(
        entry_id: str,
        strategy_override: Optional[str] = None,
        use_intersecting_genes_only: Optional[bool] = None,
    ) -> str:
        """
        Retry a failed download with optional strategy override.

        This tool allows retrying downloads that previously failed, optionally
        using a different download strategy or concatenation approach.

        Args:
            entry_id: Download queue entry ID (format: queue_GSE12345_abc123)
            strategy_override: Optional strategy override (e.g., "MATRIX_FIRST", "SUPPLEMENTARY_FIRST")
                              If not provided, uses original recommended strategy
            use_intersecting_genes_only: Optional concatenation override:
                                         - True: Keep only common genes (intersection)
                                         - False: Include all genes (union with zero-fill)
                                         - None: Use original recommended setting

        Returns:
            Retry report with status, modality name (if successful), or error details
        """
        try:
            # 1. VALIDATE ENTRY EXISTS
            all_entries = data_manager.download_queue.list_entries()
            if entry_id not in [e.entry_id for e in all_entries]:
                available = [e.entry_id for e in all_entries]
                return f"Error: Queue entry '{entry_id}' not found. Available entries: {available}"

            # 2. VALIDATE ENTRY STATUS
            entry = data_manager.download_queue.get_entry(entry_id)

            if entry.status == DownloadStatus.COMPLETED:
                return f"Entry '{entry_id}' already completed as '{entry.modality_name}'. Cannot retry completed downloads."
            elif entry.status == DownloadStatus.IN_PROGRESS:
                return f"Entry '{entry_id}' is currently in progress. Cannot retry while downloading."
            elif entry.status == DownloadStatus.PENDING:
                return f"Entry '{entry_id}' is pending (not failed). Use execute_download_from_queue instead."

            # 3. BUILD STRATEGY OVERRIDE DICT
            strategy_override_dict = None
            if strategy_override or use_intersecting_genes_only is not None:
                strategy_override_dict = {}

                if strategy_override:
                    strategy_override_dict["strategy_name"] = strategy_override

                if use_intersecting_genes_only is not None:
                    strategy_override_dict["strategy_params"] = {
                        "use_intersecting_genes_only": use_intersecting_genes_only
                    }

            # 4. INITIALIZE DOWNLOAD ORCHESTRATOR
            orchestrator = DownloadOrchestrator(data_manager)
            orchestrator.register_service(GEODownloadService(data_manager))

            logger.info(
                f"Retrying download for {entry.dataset_id} (entry: {entry_id})"
                + (
                    f" with strategy override: {strategy_override_dict}"
                    if strategy_override_dict
                    else ""
                )
            )

            # 5. EXECUTE RETRY
            try:
                modality_name, stats = orchestrator.execute_download(
                    entry_id, strategy_override_dict
                )

                # Success - format response
                response = f"## Retry Successful: {entry.dataset_id}\n\n"
                response += "✅ **Status**: Download completed on retry\n"
                response += f"- **Modality name**: `{modality_name}`\n"
                response += f"- **Samples**: {stats['shape']['n_obs']}\n"
                response += f"- **Features**: {stats['shape']['n_vars']}\n"
                if strategy_override:
                    response += f"- **Strategy override used**: {strategy_override}\n"
                if use_intersecting_genes_only is not None:
                    concat_mode = (
                        "intersection" if use_intersecting_genes_only else "union"
                    )
                    response += f"- **Concatenation override**: {concat_mode}\n"
                response += f"\n**Available for analysis**: Use `get_modality_overview('{modality_name}')` to inspect\n"

                # Log successful retry
                data_manager.log_tool_usage(
                    tool_name="retry_failed_download",
                    parameters={
                        "entry_id": entry_id,
                        "dataset_id": entry.dataset_id,
                        "strategy_override": strategy_override,
                        "use_intersecting_genes_only": use_intersecting_genes_only,
                        "retry_status": "success",
                        "modality_name": modality_name,
                    },
                    description=f"Successfully retried download for {entry.dataset_id}",
                )

                return response

            except Exception as download_error:
                # Failure - provide troubleshooting guidance
                error_msg = str(download_error)

                response = f"## Retry Failed: {entry.dataset_id}\n\n"
                response += "❌ **Status**: Retry attempt failed\n"
                response += f"- **Error**: {error_msg}\n"
                response += f"- **Queue entry**: `{entry_id}` (remains FAILED)\n"
                response += "\n**Troubleshooting**:\n"
                response += "1. Try different strategy:\n"
                response += "   - MATRIX_FIRST: Try matrix format instead\n"
                response += "   - SUPPLEMENTARY_FIRST: Try supplementary files\n"
                response += "   - H5_FIRST: Try H5 format if available\n"
                response += "2. Try different concatenation mode:\n"
                response += "   - use_intersecting_genes_only=False (union - preserves all genes)\n"
                response += "   - use_intersecting_genes_only=True (intersection - only common genes)\n"
                response += "3. Check GEO dataset availability\n"
                response += (
                    "4. Review error log: `get_queue_status(status_filter='FAILED')`\n"
                )

                # Log failed retry
                data_manager.log_tool_usage(
                    tool_name="retry_failed_download",
                    parameters={
                        "entry_id": entry_id,
                        "dataset_id": entry.dataset_id,
                        "strategy_override": strategy_override,
                        "use_intersecting_genes_only": use_intersecting_genes_only,
                        "retry_status": "failed",
                        "error": error_msg,
                    },
                    description=f"Failed retry for {entry.dataset_id}: {error_msg}",
                )

                return response

        except Exception as e:
            logger.error(f"Error in retry_failed_download: {e}")
            return f"Error retrying download for '{entry_id}': {str(e)}"

    @tool
    def get_modality_overview(
        modality_name: str = "",
        detail_level: str = "summary",
        include_provenance: bool = False,
    ) -> str:
        """
        Get overview of available modalities with flexible detail levels.

        Consolidates previous get_data_summary + list_available_modalities tools.

        Args:
            modality_name: Specific modality (empty string = all modalities)
            detail_level: "summary" | "detailed"
            include_provenance: Include W3C-PROV tracking info

        Returns:
            Formatted overview with modality statistics
        """
        try:
            if modality_name == "" or modality_name.lower() == "all":
                # List all modalities (summary mode)
                modalities = data_manager.list_modalities()
                if not modalities:
                    return "No modalities currently loaded. Use download_geo_dataset to load data."

                response = f"## Available Modalities ({len(modalities)})\n\n"
                for mod_name in modalities:
                    try:
                        adata = data_manager.get_modality(mod_name)
                        response += f"- **{mod_name}**: {adata.n_obs} samples × {adata.n_vars} features\n"
                    except Exception as e:
                        response += (
                            f"- **{mod_name}**: Error retrieving info - {str(e)}\n"
                        )

                # Add workspace status
                workspace_status = data_manager.get_workspace_status()
                response += f"\n**Workspace**: {workspace_status['workspace_path']}\n"
                response += f"**Available adapters**: {', '.join(workspace_status['registered_adapters'])}\n"

                return response
            else:
                # Single modality (detailed mode)
                if modality_name not in data_manager.list_modalities():
                    available = data_manager.list_modalities()
                    return f"Error: Modality '{modality_name}' not found. Available: {', '.join(available) if available else 'none'}"

                adata = data_manager.get_modality(modality_name)
                metrics = data_manager.get_quality_metrics(modality_name)

                response = f"## Modality: {modality_name}\n\n"
                response += (
                    f"**Shape**: {adata.n_obs} samples × {adata.n_vars} features\n"
                )
                response += (
                    f"**Obs Columns**: {', '.join(list(adata.obs.columns)[:5])}\n"
                )
                response += (
                    f"**Var Columns**: {', '.join(list(adata.var.columns)[:5])}\n"
                )

                if "total_counts" in metrics:
                    response += f"**Total counts**: {metrics['total_counts']:,.0f}\n"
                if "mean_counts_per_obs" in metrics:
                    response += f"**Mean counts per obs**: {metrics['mean_counts_per_obs']:.1f}\n"

                if detail_level == "detailed":
                    response += f"\n**Layers**: {list(adata.layers.keys())}\n"

                    # Add obsm/varm/uns info
                    if hasattr(adata, "obsm") and len(adata.obsm.keys()) > 0:
                        response += f"**Obsm Keys**: {list(adata.obsm.keys())}\n"
                    if hasattr(adata, "varm") and len(adata.varm.keys()) > 0:
                        response += f"**Varm Keys**: {list(adata.varm.keys())}\n"
                    if hasattr(adata, "uns") and len(adata.uns.keys()) > 0:
                        response += f"**Uns Keys**: {list(adata.uns.keys())}\n"

                if include_provenance:
                    # Add W3C-PROV info from DataManagerV2
                    try:
                        prov_info = data_manager.get_provenance_summary(modality_name)
                        if prov_info:
                            response += f"\n**Provenance**: {prov_info}\n"
                    except AttributeError:
                        # Fallback if method doesn't exist
                        response += "\n**Provenance**: Not available\n"

                return response

        except Exception as e:
            logger.error(f"Error in get_modality_overview: {e}")
            return f"Error retrieving modality overview: {str(e)}"

    @tool
    def list_available_modalities(filter_pattern: str = None) -> str:
        """
        List all available modalities with optional filtering.

        Args:
            filter_pattern: Optional glob-style pattern to filter modality names
                          (e.g., "geo_gse*", "*clustered", "bulk_*")

        Returns:
            str: Formatted list of modalities with details
        """
        try:
            modality_info, stats, ir = modality_service.list_modalities(
                filter_pattern=filter_pattern
            )

            # Log to provenance
            data_manager.log_tool_usage(
                tool_name="list_available_modalities",
                parameters={"filter_pattern": filter_pattern},
                description=stats,
                ir=ir,
            )

            # Format response
            if not modality_info:
                return "No modalities found matching the criteria."

            response = f"## Available Modalities ({stats['matched_modalities']}/{stats['total_modalities']})\n\n"
            if filter_pattern:
                response += f"**Filter**: `{filter_pattern}`\n\n"

            for info in modality_info:
                if "error" in info:
                    response += f"- **{info['name']}**: Error - {info['error']}\n"
                else:
                    response += f"- **{info['name']}**: {info['n_obs']} obs × {info['n_vars']} vars\n"
                    if info["obs_columns"]:
                        response += f"  - Obs: {', '.join(info['obs_columns'][:3])}\n"
                    if info["var_columns"]:
                        response += f"  - Var: {', '.join(info['var_columns'][:3])}\n"

            return response

        except Exception as e:
            logger.error(f"Error listing modalities: {e}")
            return f"Error listing modalities: {str(e)}"

    @tool
    def get_modality_details(modality_name: str) -> str:
        """
        Get detailed information about a specific modality.

        Args:
            modality_name: Name of the modality to inspect

        Returns:
            str: Detailed modality information
        """
        try:
            info, stats, ir = modality_service.get_modality_info(modality_name)

            # Log to provenance
            data_manager.log_tool_usage(
                tool_name="get_modality_details",
                parameters={"modality_name": modality_name},
                description=stats,
                ir=ir,
            )

            # Format response
            response = f"## Modality: {info['name']}\n\n"
            response += f"**Shape**: {info['shape']['n_obs']} obs × {info['shape']['n_vars']} vars\n"
            response += f"**Sparse**: {info['is_sparse']}\n\n"

            response += "**Observation Columns**:\n"
            response += f"  {', '.join(info['obs_columns'][:10])}\n"
            if len(info["obs_columns"]) > 10:
                response += f"  ... and {len(info['obs_columns']) - 10} more\n"

            response += "\n**Variable Columns**:\n"
            response += f"  {', '.join(info['var_columns'][:10])}\n"
            if len(info["var_columns"]) > 10:
                response += f"  ... and {len(info['var_columns']) - 10} more\n"

            if info["layers"]:
                response += f"\n**Layers**: {', '.join(info['layers'])}\n"
            if info["obsm_keys"]:
                response += f"**Obsm Keys**: {', '.join(info['obsm_keys'])}\n"
            if info["varm_keys"]:
                response += f"**Varm Keys**: {', '.join(info['varm_keys'])}\n"
            if info["uns_keys"]:
                response += f"**Uns Keys**: {', '.join(info['uns_keys'])}\n"

            if info["quality_metrics"]:
                response += f"\n**Quality Metrics**: {len(info['quality_metrics'])} metrics available\n"

            return response

        except Exception as e:
            logger.error(f"Error getting modality details: {e}")
            return f"Error getting modality details: {str(e)}"

    @tool
    def remove_modality(modality_name: str) -> str:
        """
        Remove a modality from memory using the modality management service.

        Args:
            modality_name: Name of modality to remove

        Returns:
            str: Status of removal operation
        """
        try:
            success, stats, ir = modality_service.remove_modality(modality_name)

            # Log to provenance
            data_manager.log_tool_usage(
                tool_name="remove_modality",
                parameters={"modality_name": modality_name},
                description=stats,
                ir=ir,
            )

            response = f"## Removed Modality: {stats['removed_modality']}\n\n"
            response += f"**Shape**: {stats['shape']['n_obs']} obs × {stats['shape']['n_vars']} vars\n"
            response += f"**Remaining modalities**: {stats['remaining_modalities']}\n"

            return response

        except Exception as e:
            logger.error(f"Error removing modality {modality_name}: {e}")
            return f"Error removing modality: {str(e)}"

    @tool
    def validate_modality_compatibility(modality_names: List[str]) -> str:
        """
        Validate compatibility between multiple modalities for integration.

        Checks observation/variable overlap, batch effects, and provides recommendations.

        Args:
            modality_names: List of modality names to validate for compatibility

        Returns:
            str: Compatibility validation report with recommendations
        """
        try:
            validation, stats, ir = modality_service.validate_compatibility(
                modality_names
            )

            # Log to provenance
            data_manager.log_tool_usage(
                tool_name="validate_modality_compatibility",
                parameters={"modality_names": modality_names},
                description=stats,
                ir=ir,
            )

            # Format response
            response = f"## Modality Compatibility Report\n\n"
            response += f"**Status**: {'✅ Compatible' if validation['compatible'] else '⚠️ Issues Detected'}\n"
            response += f"**Modalities**: {', '.join(validation['modalities'])}\n\n"

            response += "**Overlap Analysis**:\n"
            response += (
                f"  - Shared observations: {validation['shared_observations']}\n"
            )
            response += f"  - Observation overlap: {validation['observation_overlap_rate']:.1%}\n"
            response += f"  - Shared variables: {validation['shared_variables']}\n"
            response += (
                f"  - Variable overlap: {validation['variable_overlap_rate']:.1%}\n"
            )

            if validation["batch_columns"]:
                response += (
                    f"\n**Batch Columns**: {', '.join(validation['batch_columns'])}\n"
                )

            if validation["issues"]:
                response += "\n**Issues**:\n"
                for issue in validation["issues"]:
                    response += f"  - {issue}\n"

            response += "\n**Recommendations**:\n"
            for rec in validation["recommendations"]:
                response += f"  - {rec}\n"

            return response

        except Exception as e:
            logger.error(f"Error validating compatibility: {e}")
            return f"Error validating compatibility: {str(e)}"

    @tool
    def load_modality(
        modality_name: str,
        file_path: str,
        adapter: str,
        dataset_type: str = "custom",
    ) -> str:
        """
        Load a data file as a modality using the modular adapter system.

        This tool consolidates upload_data_file and load_modality_from_file functionality.

        Args:
            modality_name: Name for the new modality
            file_path: Path to the data file
            adapter: Adapter to use (e.g., 'transcriptomics_single_cell', 'proteomics_ms')
            dataset_type: Source type (e.g., 'custom', 'geo', 'local')

        Returns:
            str: Status of loading operation with modality details
        """
        try:
            adata, stats, ir = modality_service.load_modality(
                modality_name=modality_name,
                file_path=file_path,
                adapter=adapter,
                dataset_type=dataset_type,
                validate=True,
            )

            # Log to provenance
            data_manager.log_tool_usage(
                tool_name="load_modality",
                parameters={
                    "modality_name": modality_name,
                    "file_path": file_path,
                    "adapter": adapter,
                    "dataset_type": dataset_type,
                },
                description=stats,
                ir=ir,
            )

            # Format response
            response = f"## Loaded Modality: {stats['modality_name']}\n\n"
            response += f"**Shape**: {stats['shape']['n_obs']} obs × {stats['shape']['n_vars']} vars\n"
            response += f"**Adapter**: {stats['adapter']}\n"
            response += f"**File**: {stats['file_path']}\n"
            response += f"**Dataset Type**: {stats['dataset_type']}\n"
            response += f"**Validation**: {stats['validation_status']}\n"
            response += f"**Quality Metrics**: {stats['quality_metrics_count']} metrics calculated\n"
            response += (
                f"\nThe modality '{modality_name}' is now available for analysis.\n"
            )

            return response

        except Exception as e:
            logger.error(f"Error loading modality {modality_name}: {e}")
            return f"Error loading modality: {str(e)}"

    @tool
    def create_mudata_from_modalities(
        modality_names: List[str], output_name: str = "multimodal_analysis"
    ) -> str:
        """
        Create a MuData object from multiple loaded modalities for integrated analysis.

        Args:
            modality_names: List of modality names to combine
            output_name: Name for the output file

        Returns:
            str: Status of MuData creation
        """
        try:
            # Check that all modalities exist
            available_modalities = data_manager.list_modalities()
            missing = [
                name for name in modality_names if name not in available_modalities
            ]
            if missing:
                return f"Modalities not found: {missing}. Available: {available_modalities}"

            # Create MuData object
            mdata = data_manager.to_mudata(modalities=modality_names)

            # Save the MuData object
            mudata_path = f"{output_name}.h5mu"
            data_manager.save_mudata(mudata_path, modalities=modality_names)

            return f"""Successfully created MuData from {len(modality_names)} modalities.

Combined modalities: {', '.join(modality_names)}
Global shape: {mdata.n_obs} obs across {len(mdata.mod)} modalities
Saved to: {mudata_path}
Ready for integrated multi-omics analysis

The MuData object contains all selected modalities and is ready for cross-modal analysis."""

        except Exception as e:
            logger.error(f"Error creating MuData: {e}")
            return f"Error creating MuData: {str(e)}"

    @tool
    def get_adapter_info() -> str:
        """
        Get information about available adapters and their capabilities.

        Returns:
            str: Information about available adapters
        """
        try:
            adapter_info = data_manager.get_adapter_info()

            response = "Available Data Adapters:\n\n"

            for adapter_name, info in adapter_info.items():
                response += f"**{adapter_name}**:\n"
                response += f"  - Modality: {info['modality_name']}\n"
                response += (
                    f"  - Supported formats: {', '.join(info['supported_formats'])}\n"
                )
                response += f"  - Schema: {info['schema']['modality']}\n"
                response += "\n"

            response += "\nUse these adapter names when loading data with load_modality_from_file or upload_data_file."

            return response

        except Exception as e:
            logger.error(f"Error getting adapter info: {e}")
            return f"Error getting adapter info: {str(e)}"

    @tool
    def concatenate_samples(
        sample_modalities: List[str] = None,
        output_modality_name: str = None,
        geo_id: str = None,
        use_intersecting_genes_only: bool = True,
        save_to_file: bool = True,
    ) -> str:
        """
        Concatenate multiple sample modalities into a single combined modality.
        This is useful after downloading individual samples with SAMPLES_FIRST strategy.

        Args:
            sample_modalities: List of modality names to concatenate. If None, will auto-detect based on geo_id
            output_modality_name: Name for the output modality. If None, will generate based on geo_id
            geo_id: GEO accession ID to auto-detect samples (e.g., GSE12345)
            use_intersecting_genes_only: If True, use only common genes. If False, use all genes (fill missing with 0)
            save_to_file: Whether to save the concatenated data to a file

        Returns:
            str: Status message with concatenation results
        """
        try:
            # Import the ConcatenationService
            from lobster.tools.concatenation_service import ConcatenationService

            # Initialize the concatenation service
            concat_service = ConcatenationService(data_manager)

            # Auto-detect sample modalities if not provided
            if sample_modalities is None:
                if geo_id is None:
                    return "Either provide sample_modalities list or geo_id for auto-detection"

                clean_geo_id = geo_id.strip().upper()
                sample_modalities = concat_service.auto_detect_samples(
                    f"geo_{clean_geo_id.lower()}"
                )

                if not sample_modalities:
                    return f"No sample modalities found for {clean_geo_id}"

                logger.info(
                    f"Auto-detected {len(sample_modalities)} samples for {clean_geo_id}"
                )

            # Generate output name if not provided
            if output_modality_name is None:
                if geo_id:
                    output_modality_name = f"geo_{geo_id.lower()}_concatenated"
                else:
                    prefix = (
                        sample_modalities[0].rsplit("_sample_", 1)[0]
                        if "_sample_" in sample_modalities[0]
                        else sample_modalities[0].split("_")[0]
                    )
                    output_modality_name = f"{prefix}_concatenated"

            # Check if output modality already exists
            if output_modality_name in data_manager.list_modalities():
                return f"Modality '{output_modality_name}' already exists. Use remove_modality first or choose a different name."

            # Use ConcatenationService for the actual concatenation
            concatenated_adata, statistics, ir = (
                concat_service.concatenate_from_modalities(
                    modality_names=sample_modalities,
                    output_name=output_modality_name if save_to_file else None,
                    use_intersecting_genes_only=use_intersecting_genes_only,
                    batch_key="batch",
                )
            )

            # Add concatenation metadata for provenance tracking
            concatenated_adata.uns["concatenation_metadata"] = {
                "dataset_type": "concatenated_samples",
                "source_modalities": sample_modalities,
                "processing_date": pd.Timestamp.now().isoformat(),
                "concatenation_strategy": statistics.get(
                    "strategy_used", "smart_sparse"
                ),
                "concatenation_info": statistics,
            }

            # Store the concatenated result in DataManager (following tool pattern)
            data_manager.modalities[output_modality_name] = concatenated_adata

            # Log the concatenation operation for provenance
            data_manager.log_tool_usage(
                tool_name="concatenate_samples",
                parameters={
                    "sample_modalities": sample_modalities,
                    "output_modality_name": output_modality_name,
                    "use_intersecting_genes_only": use_intersecting_genes_only,
                    "save_to_file": save_to_file,
                },
                description=f"Concatenated {len(sample_modalities)} samples into modality '{output_modality_name}'",
                ir=ir,
            )

            # Format results for user display
            if save_to_file:
                return f"""Successfully concatenated {statistics['n_samples']} samples using ConcatenationService.

Output modality: '{output_modality_name}'
Shape: {statistics['final_shape'][0]} obs × {statistics['final_shape'][1]} vars
Join type: {statistics['join_type']}
Strategy: {statistics['strategy_used']}
Processing time: {statistics.get('processing_time_seconds', 0):.2f}s
Saved and stored as modality for analysis

The concatenated dataset is now available as modality '{output_modality_name}' for analysis."""
            else:
                return f"""Concatenation preview (not saved):

Shape: {statistics['final_shape'][0]} obs × {statistics['final_shape'][1]} vars
Join type: {statistics['join_type']}
Strategy: {statistics['strategy_used']}

To save, run again with save_to_file=True"""

        except Exception as e:
            logger.error(f"Error concatenating samples: {e}")
            return f"Error concatenating samples: {str(e)}"

    @tool
    def get_queue_status(
        status_filter: str = None,
        dataset_id_filter: str = None,
    ) -> str:
        """
        Get current status of download queue with optional filtering.

        This tool provides visibility into the download queue, showing which datasets
        are pending download, in progress, completed, or failed. Useful for tracking
        download operations and troubleshooting issues.

        Args:
            status_filter: Optional status filter ("PENDING", "IN_PROGRESS", "COMPLETED", "FAILED", "all")
                         If None, shows all entries.
            dataset_id_filter: Optional dataset ID filter (e.g., "GSE12345")
                             Shows only entries matching this dataset ID.

        Returns:
            Formatted queue status report with entry details
        """
        try:
            # Get all queue entries
            all_entries = data_manager.download_queue.list_entries()

            if not all_entries:
                return "## Download Queue Status\n\n📭 Queue is empty - no pending downloads"

            # Apply filters
            filtered_entries = all_entries

            if status_filter and status_filter.upper() != "ALL":
                try:
                    filter_status = DownloadStatus[status_filter.upper()]
                    filtered_entries = [
                        e for e in filtered_entries if e.status == filter_status
                    ]
                except KeyError:
                    return f"Invalid status filter '{status_filter}'. Valid options: PENDING, IN_PROGRESS, COMPLETED, FAILED, all"

            if dataset_id_filter:
                dataset_pattern = dataset_id_filter.upper()
                filtered_entries = [
                    e
                    for e in filtered_entries
                    if dataset_pattern in e.dataset_id.upper()
                ]

            # Group entries by status for better readability
            status_groups = {
                DownloadStatus.PENDING: [],
                DownloadStatus.IN_PROGRESS: [],
                DownloadStatus.COMPLETED: [],
                DownloadStatus.FAILED: [],
            }

            for entry in filtered_entries:
                status_groups[entry.status].append(entry)

            # Build response
            response = "## Download Queue Status\n\n"

            # Summary counts
            response += "**Summary**:\n"
            response += f"- Total entries: {len(all_entries)}\n"
            if status_filter or dataset_id_filter:
                response += f"- Filtered entries: {len(filtered_entries)}\n"
            response += f"- Pending: {len(status_groups[DownloadStatus.PENDING])}\n"
            response += (
                f"- In Progress: {len(status_groups[DownloadStatus.IN_PROGRESS])}\n"
            )
            response += f"- Completed: {len(status_groups[DownloadStatus.COMPLETED])}\n"
            response += f"- Failed: {len(status_groups[DownloadStatus.FAILED])}\n\n"

            if not filtered_entries:
                response += "No entries match the specified filters.\n"
                return response

            # Detailed entries by status
            for status in [
                DownloadStatus.PENDING,
                DownloadStatus.IN_PROGRESS,
                DownloadStatus.COMPLETED,
                DownloadStatus.FAILED,
            ]:
                entries = status_groups[status]
                if not entries:
                    continue

                # Status section header with emoji
                status_emoji = {
                    DownloadStatus.PENDING: "⏳",
                    DownloadStatus.IN_PROGRESS: "🔄",
                    DownloadStatus.COMPLETED: "✅",
                    DownloadStatus.FAILED: "❌",
                }
                response += (
                    f"### {status_emoji[status]} {status.value} ({len(entries)})\n\n"
                )

                # Table header
                response += (
                    "| Entry ID | Dataset ID | Database | Priority | Modality |\n"
                )
                response += (
                    "|----------|------------|----------|----------|----------|\n"
                )

                for entry in entries:
                    modality_display = entry.modality_name or "-"
                    response += f"| `{entry.entry_id}` | {entry.dataset_id} | {entry.database} | {entry.priority} | {modality_display} |\n"

                # Show error details for failed entries
                if status == DownloadStatus.FAILED:
                    response += "\n**Error Details**:\n"
                    for entry in entries:
                        if entry.error_log:
                            response += f"- `{entry.entry_id}`: {entry.error_log[-1]}\n"

                response += "\n"

            # Log tool usage
            data_manager.log_tool_usage(
                tool_name="get_queue_status",
                parameters={
                    "status_filter": status_filter,
                    "dataset_id_filter": dataset_id_filter,
                    "total_entries": len(all_entries),
                    "filtered_entries": len(filtered_entries),
                },
                description=f"Retrieved queue status: {len(filtered_entries)} entries",
            )

            return response

        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return f"Error getting queue status: {str(e)}"

    base_tools = [
        # CORE (4 tools)
        execute_download_from_queue,
        retry_failed_download,
        concatenate_samples,
        get_queue_status,
        # MODALITY MANAGEMENT (ModalityManagementService)
        list_available_modalities,
        get_modality_details,
        load_modality,
        remove_modality,
        validate_modality_compatibility,
        # HELPER
        get_modality_overview,
        get_adapter_info,
    ]
    # create_mudata_from_modalities: Combine modalities into MuData for integrated analysis

    # Combine base tools with handoff tools if provided
    tools = base_tools + (handoff_tools or [])

    system_prompt = """
<Identity_And_Expertise>
Data Expert: Local data operations and modality management specialist.

**Core Capabilities**: Execute downloads from pre-validated queue entries (created by research_agent), load local files, manage modalities (list/inspect/remove/validate), concatenate samples, retry failed downloads, provide data summaries, workspace management.

**ZERO ONLINE ACCESS**: NO internet, NO GEO/SRA metadata fetching, NO URL extraction, NO external API calls. ALL online operations delegated to research_agent.

**Not Responsible For**: Metadata validation (research_agent), dataset discovery (research_agent), URL extraction (research_agent), omics analysis (specialist agents), visualizations (visualization_expert).

**Communication**: Professional, structured markdown with clear sections. Include download status, modality shapes, queue summaries, troubleshooting guidance.

**Collaborators**: research_agent (provides validated queue entries), metadata_assistant (harmonization), specialist agents (analysis).
</Identity_And_Expertise>

<Critical_Rules>
1. **ZERO ONLINE ACCESS BOUNDARY**:

YOU HAVE NO INTERNET ACCESS. You CANNOT:
- ❌ Fetch GEO metadata, URLs, or supplementary files
- ❌ Query external databases (GEO, SRA, PRIDE, PubMed, etc.)
- ❌ Download files from URLs directly
- ❌ Validate dataset availability online

You CAN ONLY:
- ✅ Read from download queue (prepared by research_agent)
- ✅ Execute downloads from queue entries with all metadata provided
- ✅ Load local files from disk
- ✅ Manage modalities in DataManagerV2

**Decision Tree**:
Need dataset metadata → Is it in queue entry?
  ├─ YES → Use queue entry metadata
  └─ NO → handoff_to_research_agent("Need metadata validation for {{dataset_id}}")

Need to download dataset → Is there a queue entry?
  ├─ YES → execute_download_from_queue(entry_id)
  └─ NO → handoff_to_research_agent("No queue entry for {{dataset_id}}, need validation first")

2. **QUEUE-FIRST DOWNLOAD PATTERN**:

ALL downloads MUST go through queue:
1. research_agent validates metadata + creates queue entry (status: PENDING)
2. Supervisor extracts entry_id from research_agent response
3. You execute: execute_download_from_queue(entry_id="queue_GSE12345_abc123")
4. Update status: PENDING → IN_PROGRESS → COMPLETED/FAILED

NEVER attempt direct downloads. NEVER bypass queue.

3. **INTELLIGENT CONCATENATION STRATEGY**:

When downloading datasets with multiple samples, automatically decide merge strategy:
- `concatenation_strategy='auto'` (DEFAULT & RECOMMENDED): Analyzes gene coverage using DUAL CRITERIA:
  * CV criterion: If coefficient of variation > 20% → UNION
  * Range criterion: If max/min gene ratio > 1.5x → UNION
  * BOTH must pass for INTERSECTION
- `concatenation_strategy='union'`: Force include all genes (outer join with zero-filling)
- `concatenation_strategy='intersection'`: Force only common genes (inner join)

Decision logged to console + stored in provenance for transparency.

4. **PROFESSIONAL MODALITY NAMING**:

Follow naming conventions:
- GEO datasets: `geo_{{gse_id}}_transcriptomics_single_cell` (automatic)
- Custom uploads: Use descriptive names: `control_group_rnaseq`, `patient_liver_proteomics`
- Processed data: `{{base_name}}_{{operation}}`: `geo_gse12345_quality_assessed`, `geo_gse12345_clustered`

Never use generic names like "data", "test", "temp".

5. **MODALITY LIFECYCLE MANAGEMENT**:

Before loading: Check if already exists → avoid duplicates
After loading: Verify shape + quality metrics → log to provenance
Before analysis: Validate compatibility → use validate_modality_compatibility()
After operations: Use descriptive suffix → enable workflow tracking

6. **ERROR-FIRST DOWNLOAD HANDLING**:

ALWAYS check status before execute_download_from_queue:
- PENDING or FAILED → OK to execute
- IN_PROGRESS → Return error (concurrent execution conflict)
- COMPLETED → Return existing modality name (already done)
- INVALID entry_id → List available entries

On failure: Update queue to FAILED with full error log, suggest retry with different strategy.

7. **NEVER HALLUCINATE IDENTIFIERS**:

Never make up GEO accessions, dataset IDs, file paths, or modality names. Always verify what exists before referencing it.
</Critical_Rules>

<Your_11_Data_Tools>

You have **11 specialized tools** organized into 3 categories:

## 🔄 Download & Queue Tools (4 tools)

1. **`execute_download_from_queue`** - Your ONLY download mechanism
   - WHEN: After research_agent validates and queues a dataset
   - USE FOR: Executing downloads from PENDING queue entries
   - CRITICAL: Never attempt direct downloads - always use queue

2. **`retry_failed_download`** - Recovery for failed downloads
   - WHEN: After get_queue_status shows FAILED entries
   - USE FOR: Testing alternative download strategies (MATRIX_FIRST vs H5_FIRST vs SUPPLEMENTARY_FIRST)
   - PATTERN: Try different strategies until one succeeds

3. **`concatenate_samples`** - Merge individual samples into unified dataset
   - WHEN: After SAMPLES_FIRST strategy downloads multiple sample modalities
   - USE FOR: Combining geo_gse12345_sample_* into single geo_gse12345 dataset
   - STRATEGY: Use auto mode (default) for intelligent union/intersection decision

4. **`get_queue_status`** - Monitor queue state
   - WHEN: Before downloads (check PENDING), after downloads (verify COMPLETED), troubleshooting (inspect FAILED)
   - USE FOR: Getting entry_id values, checking download progress, viewing error logs
   - FIRST STEP: Always check queue before attempting downloads

## 📊 Modality Management Tools (5 tools)

5. **`list_available_modalities`** - Discover loaded datasets
   - WHEN: Start of workflow, before loading new data, after operations
   - USE FOR: Seeing what's available, checking for duplicates, filtering by pattern

6. **`get_modality_details`** - Deep inspection of single modality
   - WHEN: After loading data, before analysis, investigating issues
   - USE FOR: Verifying shape/quality, checking processing history, detailed metadata

7. **`load_modality`** - Load local files into workspace
   - WHEN: User provides custom data files (CSV/H5AD/TSV)
   - USE FOR: Adding non-GEO datasets, loading preprocessed data
   - REQUIRES: Choosing correct adapter (transcriptomics_single_cell/bulk, proteomics_ms/affinity)

8. **`remove_modality`** - Free memory
   - WHEN: Cleaning workspace, removing temporary data, managing resources
   - USE FOR: Deleting unwanted modalities, clearing failed loads

9. **`validate_modality_compatibility`** - Pre-integration check
   - WHEN: Before multi-omics integration, before meta-analysis, before concatenation
   - USE FOR: Checking obs/var overlap, detecting batch effects, recommending integration strategy
   - CRITICAL: Always run before attempting to combine modalities

## 🛠️ Helper Tools (2 tools)

10. **`get_modality_overview`** - Quick workspace summary
    - WHEN: User asks "what data do we have?", quick status checks
    - USE FOR: High-level overview of all modalities or specific one
    - NOTE: Prefer list_available_modalities + get_modality_details for detailed inspection

11. **`get_adapter_info`** - Show file format support
    - WHEN: User asks "what formats do you support?", before load_modality
    - USE FOR: Listing available adapters, understanding capabilities

</Your_11_Data_Tools>

<Tool_Selection_Decision_Trees>

## Tool Selection Logic

**Performance**: execute_download_from_queue (2-60s depending on size) | retry_failed_download (2-60s) | concatenate_samples (5-30s) | get_queue_status (instant) | list_available_modalities (instant) | get_modality_details (<1s) | load_modality (1-30s) | remove_modality (instant) | validate_modality_compatibility (1-5s) | get_modality_overview (instant) | get_adapter_info (instant)

**Downloads**: User requests download → ALWAYS check queue first: get_queue_status() → If PENDING entry exists → execute_download_from_queue(entry_id) | If NO entry → handoff_to_research_agent("Need to validate {{dataset_id}} and add to queue") | If FAILED entry → retry_failed_download(entry_id, strategy_override=...) | NEVER attempt direct download

**Queue Monitoring**: Before download → get_queue_status(status_filter="PENDING") to see what's ready | After download → get_queue_status(dataset_id_filter="GSE12345") to verify COMPLETED | Troubleshooting failures → get_queue_status(status_filter="FAILED") to see error logs

**Modality Operations**: List available data → list_available_modalities() or get_modality_overview() | Inspect specific modality → get_modality_details(modality_name) | Load local file → load_modality(name, path, adapter) | Remove data → remove_modality(name) | Before integration → validate_modality_compatibility([name1, name2])

**Sample Concatenation**: After SAMPLES_FIRST download → concatenate_samples(geo_id="GSE12345") auto-detects samples | Manual list → concatenate_samples(sample_modalities=[...]) | Control merge → use_intersecting_genes_only=True/False

**Handoff**: Dataset discovery/metadata validation/URL extraction → research_agent (ZERO online access) | Sample mapping/standardization → metadata_assistant | Analysis (QC/DE/clustering) → specialist agents (singlecell_expert, bulk_rnaseq_expert, etc.) | Visualizations → visualization_expert | Phrasing: "I'm connecting you to [agent] who specializes in [capability]" (never "I can't" or "not my job")

</Tool_Selection_Decision_Trees>

<Queue_Based_Download_Workflow>

**Pattern**: research_agent validates metadata + creates queue entry (PENDING) → Supervisor extracts entry_id from response → You check queue: get_queue_status() → Execute: execute_download_from_queue(entry_id) (PENDING → IN_PROGRESS → COMPLETED/FAILED) → Store modality in data_manager → Log provenance → Return modality name

**Queue Entry Contains**:
- dataset_id: GEO/SRA/PRIDE accession
- database: "geo", "sra", "pride"
- URLs: h5_url, matrix_url, supplementary_urls, raw_urls
- recommended_strategy: H5_FIRST, MATRIX_FIRST, SUPPLEMENTARY_FIRST, SAMPLES_FIRST, RAW_FIRST
- validation_result: Metadata validation from research_agent
- status: PENDING, IN_PROGRESS, COMPLETED, FAILED
- entry_id: Unique identifier (queue_GSE12345_abc123)

**Status Transitions**:
- PENDING → IN_PROGRESS: You call execute_download_from_queue
- IN_PROGRESS → COMPLETED: Download succeeds, modality stored
- IN_PROGRESS → FAILED: Download fails, error logged
- FAILED → IN_PROGRESS: You call retry_failed_download

</Queue_Based_Download_Workflow>

<Handoff_Triggers>

| Task | Triggers | Handoff To |
|------|----------|-----------|
| Dataset discovery, metadata validation, URL extraction | "find datasets for", "validate GSE", "check if available" | research_agent (YOU HAVE NO ONLINE ACCESS) |
| Sample mapping, metadata standardization | "map samples between", "standardize metadata", "harmonize fields" | metadata_assistant |
| Analysis (QC, DE, clustering, annotation) | "cluster cells", "find markers", "run DE analysis" | Specialist agents |
| Visualizations | "plot UMAP", "create heatmap", "visualize expression" | visualization_expert |
| Complex multi-agent workflows | 3+ agents, unclear requirements | supervisor |

</Handoff_Triggers>

<Example_Workflows>

## Workflow 1: Download from Queue (Standard)
User: "Download GSE180759"
You:
1. Check queue: get_queue_status(dataset_id_filter="GSE180759")
2. If PENDING entry found: execute_download_from_queue(entry_id="queue_GSE180759_abc123")
3. If NO entry: handoff_to_research_agent("Need to validate GSE180759 and add to download queue")
4. Verify success: get_modality_details("geo_gse180759_transcriptomics_single_cell")

## Workflow 2: Retry Failed Download with Strategy Override
User: "The GSE12345 download failed, can you try a different approach?"
You:
1. Check failed entry: get_queue_status(status_filter="FAILED", dataset_id_filter="GSE12345")
2. Retry with override: retry_failed_download(entry_id="queue_GSE12345_xyz", strategy_override="MATRIX_FIRST")
3. If still fails: retry_failed_download(entry_id="...", strategy_override="SUPPLEMENTARY_FIRST", use_intersecting_genes_only=False)

## Workflow 3: Load Local File
User: "I have a CSV file with RNA-seq counts"
You:
1. Check adapters: get_adapter_info()
2. Load file: load_modality(modality_name="patient_liver_rnaseq", file_path="/path/to/counts.csv", adapter="transcriptomics_bulk", dataset_type="custom")
3. Verify: get_modality_details("patient_liver_rnaseq")

## Workflow 4: Sample Concatenation After SAMPLES_FIRST
User: "Combine the individual samples from GSE12345"
You:
1. Check samples: list_available_modalities(filter_pattern="geo_gse12345_sample_*")
2. Concatenate: concatenate_samples(geo_id="GSE12345", use_intersecting_genes_only=None)  # Auto mode
3. Verify: get_modality_details("geo_gse12345_concatenated")

## Workflow 5: Modality Compatibility Check Before Integration
User: "Can I integrate geo_gse12345 and geo_gse67890?"
You:
1. Validate compatibility: validate_modality_compatibility(modality_names=["geo_gse12345", "geo_gse67890"])
2. If compatible (>90% obs overlap): "✅ Compatible - proceed with sample-level integration"
3. If partial (50-90%): "⚠️ Medium overlap - consider cohort-level or metadata matching"
4. If incompatible (<50%): "❌ Low overlap - use cross-modal integration or pathway-level analysis"

## Workflow 6: Workspace Exploration
User: "What data do we have loaded?"
You:
1. List all: list_available_modalities()
2. Detailed view of specific: get_modality_details("geo_gse12345")
3. Show adapters: get_adapter_info()

## Workflow 7: Error Recovery - No Queue Entry
User: "Download GSE99999"
You:
1. Check queue: get_queue_status(dataset_id_filter="GSE99999")
2. No entry found → handoff_to_research_agent("GSE99999 not in download queue. Please validate metadata and add to queue before download.")
3. (DO NOT attempt to fetch metadata yourself - YOU HAVE NO ONLINE ACCESS)

</Example_Workflows>

<Modality_System>
The DataManagerV2 uses a modular approach where each dataset is loaded as a **modality** with appropriate schema:

**Available Adapters:**
- `transcriptomics_single_cell`: Single-cell RNA-seq data
- `transcriptomics_bulk`: Bulk RNA-seq data
- `proteomics_ms`: Mass spectrometry proteomics
- `proteomics_affinity`: Affinity-based proteomics

**Data Flow:**
1. Load data using appropriate adapter → Creates modality with schema validation
2. Modalities stored with unique names → Accessible to other agents
3. Multiple modalities → Can be combined into MuData for integrated analysis

When working with DataManagerV2, always think in terms of **modalities** rather than single datasets.
</Modality_System>

<Critical_Reminders>
- NEVER HALLUCINATE IDENTIFIERS (GEO accessions, dataset IDs, file paths, modality names)
- YOU HAVE ZERO ONLINE ACCESS - delegate all metadata/URL operations to research_agent
- ALL downloads MUST go through queue - no exceptions
- Check queue status before executing downloads
- Use descriptive modality names for workflow tracking
- Validate compatibility before multi-omics integration
- Log all operations to provenance
</Critical_Reminders>

Today's date is {date}.
""".format(
        date=date.today()
    )

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=DataExpertState,
    )
