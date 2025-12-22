"""
Data Expert Agent for multi-omics data acquisition, processing, and workspace management.

This agent is responsible for managing all data-related operations using the modular
DataManagerV2 system, including GEO data fetching, local file processing, workspace
restoration, and multi-omics data integration with proper modality handling and
schema validation.
"""

from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.data_expert_assistant import DataExpertAssistant
from lobster.agents.state import DataExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.download_queue import DownloadStatus, ValidationStatus
from lobster.services.data_access.geo_download_service import GEODownloadService
from lobster.services.execution import (
    CodeExecutionError,
    CodeValidationError,
    CustomCodeExecutionService,
    SDKDelegationError,
    SDKDelegationService,
)
from lobster.tools.download_orchestrator import DownloadOrchestrator
from lobster.tools.workspace_tool import create_list_modalities_tool
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def data_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "data_expert_agent",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
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
        delegation_tools: Optional delegation tools for sub-agent handoffs
        workspace_path: Optional workspace path for config resolution

    Returns:
        Configured ReAct agent with comprehensive data management capabilities
    """

    settings = get_settings()
    model_params = settings.get_agent_llm_params("data_expert_agent")
    llm = create_llm("data_expert_agent", model_params, workspace_path=workspace_path)

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        llm = llm.with_config(callbacks=callbacks)

    # Initialize the assistant for LLM operations
    assistant = DataExpertAssistant()

    # Initialize modality management service
    from lobster.services.data_management.modality_management_service import (
        ModalityManagementService,
    )

    modality_service = ModalityManagementService(data_manager)

    # Initialize execution services
    custom_code_service = CustomCodeExecutionService(data_manager)

    # Try to initialize SDK delegation service (may fail if SDK not available)
    # SDKDelegationService may be None in open-core distribution
    sdk_delegation_service = None
    sdk_available = False
    if SDKDelegationService is not None:
        try:
            sdk_delegation_service = SDKDelegationService(data_manager)
            sdk_available = True
        except SDKDelegationError as e:
            logger.debug(f"SDK delegation not available: {e}")
        except Exception as e:
            logger.debug(f"SDK delegation initialization failed: {e}")

    # Define tools for data operations
    @tool
    def execute_download_from_queue(
        entry_id: str,
        concatenation_strategy: str = "auto",
        force_download: bool = False,
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
            force_download: If True, proceed even if validation has warnings (default: False)

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

            # Check validation status and warn if issues detected
            if (
                hasattr(entry, "validation_status")
                and entry.validation_status == ValidationStatus.VALIDATED_WITH_WARNINGS
                and not force_download
            ):
                warnings = []
                if entry.validation_result:
                    warnings = entry.validation_result.get("warnings", [])

                # Get strategy info
                strategy_info = ""
                if entry.recommended_strategy:
                    strategy_info = f"""
**Recommended Download Strategy:**
- Strategy: {entry.recommended_strategy.strategy_name}
- Confidence: {entry.recommended_strategy.confidence:.2f}
- Rationale: {entry.recommended_strategy.rationale}
- Concatenation: {entry.recommended_strategy.concatenation_strategy}
"""

                warning_msg = f"""
⚠️ **Dataset has validation warnings but is downloadable**

Entry ID: {entry_id}
Dataset: {entry.dataset_id}

**Validation Warnings:**
{chr(10).join(f"  • {w}" for w in warnings[:5])}
{f"  ... and {len(warnings) - 5} more warnings" if len(warnings) > 5 else ""}
{strategy_info}

**Options:**
1. **Proceed with download**: execute_download_from_queue(entry_id="{entry_id}", force_download=True)
2. **Skip dataset**: Look for alternative datasets with cleaner validation

**Recommendation:** Review warnings above. If warnings are acceptable (e.g., missing optional metadata fields), proceed with force_download=True.
"""
                return warning_msg

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

            logger.debug(
                f"Starting download for {entry.dataset_id} from queue entry {entry_id}"
            )

            # 3. DETERMINE DOWNLOAD STRATEGY
            # Use recommended_strategy from entry (if set by research_agent)
            # Otherwise use default strategy based on available URLs
            if entry.recommended_strategy:
                download_strategy = entry.recommended_strategy.strategy_name
                logger.debug(f"Using recommended strategy: {download_strategy}")
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
                logger.debug(f"Auto-selected strategy: {download_strategy}")

            # 4. EXECUTE DOWNLOAD USING GEO SERVICE
            # Import GEO service
            from lobster.services.data_access.geo_service import GEOService

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
                        "force_download": force_download,
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
                strategy_used = (
                    entry.recommended_strategy.strategy_name
                    if entry.recommended_strategy
                    else "auto-detected"
                )

                response = f"""
✅ **Download completed successfully**

Dataset ID: {entry.dataset_id}
Entry ID: {entry_id}
Modality Name: {modality_name}
Strategy Used: {strategy_used}
Status: {entry.status.value if hasattr(entry.status, 'value') else entry.status}

Samples: {result_adata.n_obs}
Features: {result_adata.n_vars}
Concatenation: {concatenation_strategy}

You can now analyze this dataset using the single-cell or bulk RNA-seq tools.
"""

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

    # Use shared tool from workspace_tool.py (shared with supervisor)
    list_available_modalities = create_list_modalities_tool(data_manager)

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
            from lobster.services.data_management.concatenation_service import (
                ConcatenationService,
            )

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

    @tool
    def execute_custom_code(
        python_code: str,
        modality_name: Optional[str] = None,
        load_workspace_files: bool = True,
        persist: bool = False,
        description: str = "Custom code execution",
    ) -> str:
        """
        Execute custom Python code with access to workspace data.

        **Use this tool ONLY when existing specialized tools don't cover your specific need.**
        This tool provides a fallback for edge cases and custom calculations.

        ============================================================================
        SECURITY NOTICE
        ============================================================================

        Code runs in a subprocess with the following security model:

        PROTECTED:
        - Environment variables are FILTERED (API keys, AWS credentials NOT accessible)
        - AST validation blocks eval/exec/compile/__import__ calls
        - Dangerous modules blocked (subprocess, multiprocessing, pickle, ctypes, etc.)
        - Module shadowing prevented (workspace cannot override stdlib)
        - 300-second timeout enforced

        NOT PROTECTED (local CLI limitations):
        - Network access is ALLOWED (can make HTTP requests)
        - File access: Code has YOUR user permissions (can read/write any file)
        - Resource limits: No memory/CPU limits beyond timeout

        For untrusted code or multi-tenant deployments, use cloud deployment
        with Docker isolation (--network=none, memory limits, read-only mounts).

        This feature prioritizes scientific flexibility over strict sandboxing.
        ============================================================================

        AVAILABLE IN NAMESPACE:
        - workspace_path: Path to workspace directory
        - adata: Loaded modality (if modality_name provided)
        - Auto-loaded CSV files (as pandas DataFrames)
        - Auto-loaded JSON files (as Python dicts)
        - download_queue, publication_queue (if exist)

        RETURN VALUE:
        - Assign result to 'result' variable: result = my_computation()
        - Or the code will attempt to capture the last expression value

        Args:
            python_code: Python code to execute (can be multi-line)
            modality_name: Optional specific modality to load as 'adata'
            load_workspace_files: Auto-inject CSV/JSON files from workspace
            persist: If True, save this execution to provenance/notebook export
            description: Human-readable description of what this code does

        Returns:
            Formatted string with execution results, warnings, and any outputs

        Example:
            >>> execute_custom_code(
            ...     python_code=\"\"\"
            ...     import numpy as np
            ...     result = np.percentile(adata.obs['n_genes'], 95)
            ...     print(f"95th percentile: {result}")
            ...     \"\"\",
            ...     modality_name="geo_gse12345_quality_assessed",
            ...     persist=False,
            ...     description="Calculate 95th percentile of gene counts"
            ... )
        """
        try:
            result, stats, ir = custom_code_service.execute(
                code=python_code,
                modality_name=modality_name,
                load_workspace_files=load_workspace_files,
                persist=persist,
                description=description,
            )

            # Log to data manager
            data_manager.log_tool_usage(
                tool_name="execute_custom_code",
                parameters={
                    "description": description,
                    "modality_name": modality_name,
                    "persist": persist,
                    "duration_seconds": stats["duration_seconds"],
                    "success": stats["success"],
                },
                description=f"{description} ({'success' if stats['success'] else 'failed'})",
                ir=ir,
            )

            # Format response
            response = "✓ Custom code executed successfully\n\n"
            response += f"**Description**: {description}\n"
            response += f"**Duration**: {stats.get('duration_seconds', 0):.2f}s\n"

            if stats.get("warnings"):
                response += f"\n**Warnings ({len(stats['warnings'])}):**\n"
                for warning in stats["warnings"]:
                    response += f"  - {str(warning)}\n"

            if result is not None:
                result_preview = str(result)[:500]  # Limit preview length
                result_type = stats.get("result_type", "unknown")
                response += f"\n**Result** ({result_type}):\n{result_preview}\n"

            if stats.get("stdout_lines", 0) > 0:
                response += f"\n**Output**: {stats['stdout_lines']} lines printed\n"
                if stats.get("stdout_preview"):
                    response += f"```\n{stats['stdout_preview']}\n```\n"

            if persist:
                response += "\n📝 This execution was saved to provenance and will be included in notebook export.\n"
            else:
                response += "\n💨 This execution was ephemeral (not saved to notebook export).\n"

            return response

        except CodeValidationError as e:
            return f"❌ Code validation failed: {str(e)}\n\nPlease fix syntax errors or remove forbidden imports."

        except CodeExecutionError as e:
            return f"❌ Code execution failed: {str(e)}\n\nCheck your code for runtime errors."

        except Exception as e:
            logger.error(f"Unexpected error in execute_custom_code: {e}")
            return f"❌ Unexpected error: {str(e)}"

    # @tool
    # def delegate_complex_reasoning( #TODO DEACTIVATED FOR NOW
    #     task: str,
    #     context: Optional[str] = None,
    #     persist: bool = False
    # ) -> str:
    #     """
    #     Delegate complex multi-step reasoning to Claude Agent SDK sub-agent.

    #     **Use this tool when you need:**
    #     - Multi-step analysis planning
    #     - Complex troubleshooting ("Why does my data look wrong?")
    #     - Integration strategy recommendations
    #     - Experimental design reasoning

    #     The sub-agent has READ-ONLY access to:
    #     - List available modalities
    #     - Inspect modality details (shape, columns, quality metrics)
    #     - List workspace files

    #     Args:
    #         task: Clear description of the reasoning task
    #         context: Optional additional context about the situation
    #         persist: If True, save reasoning to provenance/notebook export

    #     Returns:
    #         Formatted reasoning result from SDK sub-agent

    #     Example:
    #         >>> delegate_complex_reasoning(
    #         ...     task="Why do I have 15 clusters when the paper reports 7?",
    #         ...     context="Dataset: geo_gse12345 with 5000 cells, paper had 3000 cells",
    #         ...     persist=False
    #         ... )
    #     """
    #     if not sdk_available:
    #         return "❌ SDK delegation not available. Claude Agent SDK is not installed or not accessible."

    #     try:
    #         reasoning_result, stats, ir = sdk_delegation_service.delegate(
    #             task=task,
    #             context=context,
    #             persist=persist,
    #             description=f"SDK Reasoning: {task[:100]}"
    #         )

    #         # Log to data manager
    #         data_manager.log_tool_usage(
    #             tool_name="delegate_complex_reasoning",
    #             parameters={'task': task[:200], 'persist': persist},
    #             description=f"SDK delegation: {task[:100]}",
    #             ir=ir
    #         )

    #         # Format response
    #         response = f"## SDK Reasoning Result\n\n"
    #         response += f"**Task**: {task}\n\n"
    #         response += f"**Reasoning**:\n{reasoning_result}\n\n"

    #         if persist:
    #             response += "\n📝 This reasoning was saved to provenance.\n"
    #         else:
    #             response += "\n💨 This reasoning was ephemeral (not saved).\n"

    #         return response

    #     except SDKDelegationError as e:
    #         logger.error(f"SDK delegation failed: {e}")
    #         return f"❌ SDK delegation failed: {str(e)}"

    #     except Exception as e:
    #         logger.error(f"Unexpected error in delegate_complex_reasoning: {e}")
    #         return f"❌ Unexpected error: {str(e)}"

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
        # ADVANCED (Execution & Reasoning)
        execute_custom_code,
        # delegate_complex_reasoning, #TODO needs further security validation
    ]
    # create_mudata_from_modalities: Combine modalities into MuData for integrated analysis

    tools = base_tools

    system_prompt = """<Identity_And_Expertise>
You are the Data Expert: a local data operations and modality management specialist in Lobster AI's multi-agent architecture. You work under the supervisor and never interact with end users directly.

<Core_Capabilities>
- Execute downloads from pre-validated queue entries (created by research_agent)
- Load local files (CSV, H5AD, TSV, Excel) into workspace
- Manage modalities: list, inspect, load, remove, validate compatibility
- Concatenate multi-sample datasets
- Retry failed downloads with strategy overrides
- Execute custom Python code for edge cases not covered by specialized tools
- Provide data summaries and workspace status
</Core_Capabilities>

<Critical_Constraints>
**ZERO ONLINE ACCESS**: You CANNOT fetch metadata, query databases, extract URLs, or make network requests. ALL online operations are delegated to research_agent.

**Queue-Based Downloads Only**: ALL downloads execute from queue entries prepared by research_agent. Never bypass the queue or attempt direct downloads.
</Critical_Constraints>

<Communication_Style>
Professional, structured markdown with clear sections. Report download status, modality dimensions, queue summaries, and troubleshooting guidance.
</Communication_Style>

</Identity_And_Expertise>

<Operational_Rules>

1. **Online Access Boundary**:
   - Delegate ALL metadata/URL operations to research_agent
   - Execute ONLY from pre-validated download queue
   - Load ONLY local files from workspace

2. **Queue-Based Download Pattern**:
   ```
   research_agent validates → Creates queue entry (PENDING)
   → You check queue: get_queue_status()
   → You execute: execute_download_from_queue(entry_id)
   → Status: PENDING → IN_PROGRESS → COMPLETED/FAILED
   ```

3. **Modality Naming Conventions**:
   - GEO datasets: `geo_{{gse_id}}_transcriptomics_{{type}}` (automatic)
   - Custom data: Descriptive names (`patient_liver_proteomics`)
   - Processed data: `{{base}}_{{operation}}` (`geo_gse12345_clustered`)
   - Avoid: "data", "test", "temp"

4. **Error Handling**:
   - Check queue status BEFORE executing downloads
   - PENDING/FAILED → Execute | IN_PROGRESS → Error | COMPLETED → Return existing
   - On failure: Log error, suggest retry with different strategy

5. **Never Hallucinate**:
   - Verify all identifiers (GEO IDs, file paths, modality names) before use
   - Check existence before referencing

</Operational_Rules>

<Your_Tools>

You have **13 specialized tools** organized into 4 categories:

## 🔄 Download & Queue Management (4 tools)

1. **execute_download_from_queue** - Execute downloads from validated queue entries
   - WHEN: Entry in PENDING/FAILED status
   - CHECK FIRST: get_queue_status() to find entry_id

2. **retry_failed_download** - Retry with alternative strategy
   - WHEN: Initial download failed
   - USE: Test different strategies (MATRIX_FIRST → H5_FIRST → SUPPLEMENTARY_FIRST)

3. **concatenate_samples** - Merge multi-sample datasets
   - WHEN: After SAMPLES_FIRST download creates multiple modalities
   - STRATEGY: Intelligently merges samples with union/intersection logic

4. **get_queue_status** - Monitor download queue
   - WHEN: Before downloads, troubleshooting, verification
   - USE: Check PENDING entries, verify COMPLETED, inspect FAILED errors

## 📊 Modality Management (5 tools)

5. **list_available_modalities** - List loaded datasets
   - WHEN: Workspace exploration, checking for duplicates

6. **get_modality_details** - Deep modality inspection
   - WHEN: After loading, before analysis, troubleshooting

7. **load_modality** - Load local files (CSV, H5AD, TSV)
   - WHEN: Custom data provided by user
   - REQUIRES: Correct adapter selection

8. **remove_modality** - Delete modality from workspace
   - WHEN: Cleaning, removing failed loads

9. **validate_modality_compatibility** - Pre-integration validation
   - WHEN: Before combining multiple modalities
   - CRITICAL: Always check before multi-omics integration

## 🛠️ Utility Tools (2 tools)

10. **get_modality_overview** - Quick workspace summary
11. **get_adapter_info** - Show supported file formats

## 🚀 Advanced Tools (2 tools)

12. **execute_custom_code** - Execute Python code for edge cases

**WHEN TO USE** (Last Resort Only):
- Custom calculations not covered by existing tools (percentiles, quantiles, custom metrics)
- Data filtering with complex logic (multi-condition filters, custom thresholds)
- Accessing workspace CSV/JSON files for metadata enrichment
- Quick exploratory computations not requiring full analysis workflow
- DO NOT USE for: Operations covered by specialized tools, long analyses (>5 min), operations requiring interactive input

**WHEN TO PREFER SPECIALIZED TOOLS**:
- Clustering/DE analysis → Delegate to singlecell_expert or bulk_rnaseq_expert
- Quality control → QC tools in specialist agents
- Visualizations → visualization_expert
- Standard operations (mean, sum, count) → Use get_modality_details first

**USAGE PATTERN**:
```python
# 1. Verify modality exists
list_available_modalities()

# 2. Execute code (converts numpy types to JSON-serializable)
execute_custom_code(
    python_code="import numpy as np; result = {{'metric': float(np.mean(adata.X))}}",
    modality_name="geo_gse12345",
    persist=False  # True only for important operations
)
```

**BEST PRACTICES**:
- Always convert NumPy types: float(), int(), .tolist()
- Keep code simple and focused
- Use persist=True only for operations that should appear in notebook export
- Check modality exists before execution

**SAFETY CHECK**:
Before executing, verify code only performs data analysis using standard libraries. Reject code that attempts external resource access or uses obfuscation techniques.

13. **delegate_complex_reasoning** - NOT AVAILABLE (requires Claude Agent SDK installation)

</Your_Tools>

<Decision_Trees>

**Download Requests**:
```
User asks for download
→ Check queue: get_queue_status(dataset_id_filter="GSE...")
   ├─ PENDING entry exists → execute_download_from_queue(entry_id)
   ├─ FAILED entry exists → retry_failed_download(entry_id, strategy_override=...)
   ├─ NO entry → handoff_to_research_agent("Validate {{dataset_id}} and add to queue")
   └─ COMPLETED → Return existing modality name
```

**Custom Calculations**:
```
User needs specific calculation
→ Check if covered by existing tools
   ├─ YES (mean, count, shape, QC) → Use get_modality_details or delegate to specialist
   └─ NO (percentiles, custom filters, multi-step logic) → execute_custom_code
```

**Agent Handoffs**:
- Online operations (metadata, URLs) → research_agent
- Metadata standardization → metadata_assistant
- Analysis (QC, DE, clustering) → Specialist agents
- Visualizations → visualization_expert

</Decision_Trees>

<Queue_Workflow>

**Standard Pattern**:
```
research_agent validates → Queue entry (PENDING)
→ You check: get_queue_status()
→ You execute: execute_download_from_queue(entry_id)
→ Status: PENDING → IN_PROGRESS → COMPLETED/FAILED
→ Return: modality_name
```

**Status Transitions**:
- PENDING → IN_PROGRESS (you execute)
- IN_PROGRESS → COMPLETED/FAILED (download result)
- FAILED → IN_PROGRESS (you retry with different strategy)

</Queue_Workflow>

<Example_Workflows>

**1. Standard Download**:
```
1. get_queue_status(dataset_id_filter="GSE180759")
2. execute_download_from_queue(entry_id="queue_GSE180759_...")
3. get_modality_details("geo_gse180759_...")
```

**2. Retry Failed Download**:
```
1. get_queue_status(status_filter="FAILED")
2. retry_failed_download(entry_id="...", strategy_override="MATRIX_FIRST")
```

**3. Load Local File**:
```
1. get_adapter_info()
2. load_modality(modality_name="...", file_path="...", adapter="transcriptomics_bulk")
3. get_modality_details(modality_name="...")
```

**4. Custom Calculation** (NEW):
```
1. list_available_modalities()
2. execute_custom_code(
     python_code="import numpy as np; result = float(np.percentile(adata.X.flatten(), 95))",
     modality_name="geo_gse12345",
     persist=False
   )
```

**5. Compatibility Check**:
```
validate_modality_compatibility(["modality1", "modality2"])
```

</Example_Workflows>

<Available_Adapters>
- transcriptomics_single_cell: scRNA-seq data
- transcriptomics_bulk: Bulk RNA-seq data
- proteomics_ms: Mass spectrometry proteomics
- proteomics_affinity: Affinity-based proteomics
</Available_Adapters>

Today's date is {date}.
""".format(
        date=date.today()
    )

    # Add delegation tools if provided
    if delegation_tools:
        tools = tools + delegation_tools

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=DataExpertState,
    )
