"""
Data Expert Agent for multi-omics data acquisition, processing, and workspace management.

This agent is responsible for managing all data-related operations using the modular
DataManagerV2 system, including GEO data fetching, local file processing, workspace
restoration, and multi-omics data integration with proper modality handling and
schema validation.
"""

from datetime import date
from typing import List

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.data_expert_assistant import DataExpertAssistant
from lobster.agents.state import DataExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.download_queue import DownloadStatus
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

    @tool
    def check_tmp_metadata_keys() -> List:
        """
        Check which metadata is temporarelly stored
        """
        return data_manager.metadata_store.keys()

    # Define tools for data operations
    @tool
    def fetch_geo_metadata_and_strategy_config(
        geo_id: str, data_source: str = "geo"
    ) -> str:
        """
        Fetch and validate GEO dataset metadata without downloading the full dataset.
        Use this FIRST before download_geo_dataset to preview dataset information.

        Args:
            geo_id: GEO accession number (e.g., GSE12345 or GDS5826)

        Returns:
            str: Formatted metadata summary with validation results and recommendation
        """
        try:
            # Clean the GEO ID
            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith("GSE") and not clean_geo_id.startswith(
                "GDS"
            ):
                return f"Invalid GEO ID format: {geo_id}. Must be a GSE or GDS accession (e.g., GSE194247 or GDS5826)"

            logger.info(f"Fetching metadata for GEO dataset: {clean_geo_id}")

            # Use GEOService to fetch metadata only
            from lobster.tools.geo_service import GEOService

            console = getattr(data_manager, "console", None)
            geo_service = GEOService(data_manager, console=console)

            # ------------------------------------------------
            # Check if metadata already in store
            # ------------------------------------------------
            stored_entry = data_manager._get_geo_metadata(clean_geo_id)
            if stored_entry:
                if stored_entry.get("strategy_config"):
                    logger.debug(
                        f"Metadata already stored for: {geo_id}. returning summary"
                    )
                    summary = geo_service._format_metadata_summary(
                        clean_geo_id, stored_entry
                    )
                    return summary
                logger.info(
                    f"{clean_geo_id} is in metadata but no strategy config has been generated yet. Proceeding doing so"
                )

            # ------------------------------------------------
            # If not fetch and return metadata & val res
            # ------------------------------------------------
            # Fetch metadata only (no expression data download)
            metadata, validation_result = geo_service.fetch_metadata_only(clean_geo_id)

            # ------------------------------------------------
            # Extract strategy config using assistant
            # ------------------------------------------------
            strategy_config = assistant.extract_strategy_config(metadata, clean_geo_id)

            if not strategy_config:
                logger.warning(f"Failed to extract strategy config for {clean_geo_id}")
                return "Failed with fetching geo metadata. Try again"

            # ------------------------------------------------
            # store in DataManager
            # ------------------------------------------------
            # Store metadata in data_manager for future use
            data_manager.metadata_store[clean_geo_id] = {
                "metadata": metadata,
                "validation": validation_result,
                "fetch_timestamp": pd.Timestamp.now().isoformat(),
                "data_source": data_source,
                "strategy_config": (
                    strategy_config.model_dump()
                    if "strategy_config" in locals()
                    else {}
                ),
            }

            # Log the metadata fetch operation
            data_manager.log_tool_usage(
                tool_name="fetch_geo_metadata_and_strategy_config",
                parameters={"geo_id": clean_geo_id, "data_source": data_source},
                description=f"Fetched metadata for GEO dataset {clean_geo_id} using {data_source}",
            )

            # Format comprehensive metadata summary
            base_summary = geo_service._format_metadata_summary(
                clean_geo_id, metadata, validation_result
            )

            # Add strategy config section if available
            if strategy_config:
                strategy_section = assistant.format_strategy_section(strategy_config)
                summary = base_summary + strategy_section
            else:
                summary = base_summary

            logger.debug(
                f"Successfully fetched and validated metadata for {clean_geo_id} using {data_source}"
            )

            return summary

        except Exception as e:
            logger.error(f"Error fetching GEO metadata for {geo_id}: {e}")
            return f"Error fetching metadata: {str(e)}"

    @tool
    def check_file_head_from_supplementary_files(geo_id: str, filename: str) -> str:
        """
        Print the head of a file in the supplementary files
        """
        # ------------------------------------------------
        # find name in supplementary
        # ------------------------------------------------
        # iterate through data_manager
        # Use GEOService to fetch metadata only
        from lobster.tools.geo_service import GEOService

        console = getattr(data_manager, "console", None)
        geo_service = GEOService(data_manager, console=console)

        # Get stored metadata using validated retrieval
        stored_entry = data_manager._get_geo_metadata(geo_id)
        if not stored_entry:
            return f"Error: Metadata for {geo_id} not found"

        target_url = ""
        for urls in stored_entry["metadata"].get("supplementary_file", []):
            if isinstance(urls, str):
                if filename in urls:
                    target_url = urls
                    # geo_service._download_and_parse_file(target_url)
                    file_head = geo_service.geo_parser.show_dynamic_head(target_url)
                    return file_head.get(
                        "head", "Error in fetching head: nothing to fetch"
                    )

            msg = f"why is url not a str?? -> {urls}. Instead is type {type(urls)}"
            logger.warning(msg)
            return msg

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
                    },
                    result={
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
    def upload_data_file(
        file_path: str,
        dataset_id: str,
        adapter: str = "auto_detect",
        dataset_type: str = "custom",
    ) -> str:
        """
        Upload and process a data file from the local filesystem.

        Args:
            file_path: Path to the data file (CSV, H5, Excel, etc.)
            dataset_id: Unique identifier for this dataset
            adapter: Type of biological data ('transcriptomics_single_cell', 'transcriptomics_bulk', 'auto_detect')
            dataset_type: Source type (e.g., 'custom', 'local', 'processed')

        Returns:
            str: Summary of uploaded data
        """
        try:
            from pathlib import Path

            file_path = Path(file_path)
            if not file_path.exists():
                return f"File not found: {file_path}"

            if not isinstance(adapter, str):
                return "Modality type must be a string"
            elif not adapter:
                return "Modality type can not be None"

            # Auto-detect modality type if requested
            if adapter == "auto_detect":
                # Read file to detect data characteristics
                import pandas as pd

                try:
                    df = pd.read_csv(
                        file_path, index_col=0, nrows=10
                    )  # Sample first 10 rows
                    n_cols = df.shape[1]

                    # Heuristics for detection
                    if n_cols > 5000:
                        adapter = "transcriptomics_single_cell"
                    elif n_cols < 1000:
                        adapter = "proteomics_ms"
                    else:
                        adapter = "transcriptomics_bulk"  # Middle ground

                    logger.info(
                        f"Auto-detected modality type: {adapter} (based on {n_cols} features)"
                    )
                except Exception:
                    adapter = "single_cell"  # Safe default

            modality_name = f"{dataset_type}_{dataset_id}"

            # Load using appropriate adapter
            adata = data_manager.load_modality(
                name=modality_name,
                source=file_path,
                adapter=adapter,
                validate=True,
                dataset_id=dataset_id,
                dataset_type=dataset_type,
            )

            # Save to workspace
            save_path = f"{dataset_id}_{adapter}.h5ad"
            data_manager.save_modality(modality_name, save_path)

            # Get quality metrics
            metrics = data_manager.get_quality_metrics(modality_name)

            return f"""Successfully uploaded and processed file {file_path.name}.

Modality: '{modality_name}' ({adata.n_obs} obs × {adata.n_vars} vars)
Data type: {adapter}
Adapter: {adapter}
Saved to: {save_path}
Quality metrics: {len([k for k, v in metrics.items() if isinstance(v, (int, float))])} metrics calculated

The dataset is now available as modality '{modality_name}' for analysis."""

        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {e}")
            return f"Error uploading file: {str(e)}"

    @tool
    def load_modality_from_file(
        modality_name: str, file_path: str, adapter: str, **kwargs
    ) -> str:
        """
        Load a specific file as a modality using the modular system.

        Args:
            modality_name: Name for the new modality ('transcriptomics_single_cell', 'transcriptomics_bulk', 'proteomics_ms', 'proteomics_affinity')
            file_path: Path to the data file
            adapter: Adapter to use (transcriptomics_single_cell, transcriptomics_bulk, proteomics_ms, etc.)
            **kwargs: Additional adapter parameters

        Returns:
            str: Status of loading operation
        """
        try:

            file_path = data_manager.data_dir / file_path

            if not file_path.exists():
                return f"File not found: {file_path}"

            # Check if modality already exists
            if modality_name in data_manager.list_modalities():
                return f"Modality '{modality_name}' already exists. Use remove_modality first or choose a different name."

            # Check if adapter is available
            available_adapters = list(data_manager.adapters.keys())
            if adapter not in available_adapters:
                return f"Adapter '{adapter}' not available. Available adapters: {available_adapters}"

            # Load the modality
            adata = data_manager.load_modality(
                name=modality_name,
                source=file_path,
                adapter=adapter,
                validate=True,
                **kwargs,
            )

            # Get quality metrics
            metrics = data_manager.get_quality_metrics(modality_name)

            return f"""Successfully loaded modality '{modality_name}'.

Shape: {adata.n_obs} obs × {adata.n_vars} vars
Adapter: {adapter}
Source: {file_path.name}
Quality metrics: {len([k for k, v in metrics.items() if isinstance(v, (int, float))])} metrics calculated

The modality is now available for analysis and can be used by other agents."""

        except Exception as e:
            logger.error(f"Error loading modality {modality_name}: {e}")
            return f"Error loading modality: {str(e)}"

    @tool
    def remove_modality(modality_name: str) -> str:
        """
        Remove a modality from memory.

        Args:
            modality_name: Name of modality to remove

        Returns:
            str: Status of removal operation
        """
        try:
            try:
                data_manager.remove_modality(modality_name)
                return f"Successfully removed modality '{modality_name}' from memory."
            except ValueError as e:
                return str(e)

        except Exception as e:
            logger.error(f"Error removing modality {modality_name}: {e}")
            return f"Error removing modality: {str(e)}"

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
            data_manager.save_mudata(
                mudata_path, modalities=modality_names
            )

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

    base_tools = [
        # CORE
        fetch_geo_metadata_and_strategy_config,
        check_file_head_from_supplementary_files,
        execute_download_from_queue,
        concatenate_samples,
        # HELPER
        check_tmp_metadata_keys,
        get_modality_overview,
        upload_data_file,
        load_modality_from_file,
        remove_modality,
        get_adapter_info,
    ]
    # create_mudata_from_modalities, prompt: - create_mudata_from_modalities: Combine modalities into MuData for integrated analysis

    # Combine base tools with handoff tools if provided
    tools = base_tools + (handoff_tools or [])

    system_prompt = """
You are a data expert agent specializing in multi-omics bioinformatics datasets using the modular DataManagerV2 system.
You are one of many agents in a supervisor system.  
Your expertise lays in understanding and handling different data types in transcriptomcis & Proteomics. You must critically think to find the best and most efficient way to solve a task given the tools that you have.
If you are unsure about a task you can always ask the supervisor who will ask the user. 

<Task>
You handle all data acquisition, storage, and retrieval operations using the new modular architecture:
0. **Fetching metadata** and give a summary to the supervisor
1. **Download and load datasets** from various sources (GEO, local files, etc.)
2. **Process and validate data** using appropriate modality adapters
3. **Store data as modalities** with proper schema enforcement
4. **Restore workspace datasets** from previous sessions for continued analysis
5. **Provide data access** to other agents via modality names
6. **Maintain workspace** with proper organization and provenance tracking
</Task>

<Available Core Tools>
- **fetch_geo_metadata_and_strategy_config**: 
This tool can be used to understand the metadata from a GEO entry. It returns a summary of the dataset with the available files in this entry.
The given files are crucial for the decision of a downloading strategy. GEO entries most often are different in their annotation, available files etc. 
Different scenarios include:
    1. **Raw FASTQ files** (SRA links): indicates the dataset requires alignment/quantification downstream; may not be immediately usable.
    2. **Processed expression matrices** (TXT/CSV/TSV/GCT/XLSX): typically contain normalized or raw count data; usually the most relevant for transcriptomics analysis.
    3. **Series Matrix files** (SOFT/MINiML): contain sample annotations (phenotype/metadata) and sometimes processed expression values; should almost always be downloaded.
    4. **Supplementary annotation files** (TXT/CSV/XLSX): mapping files for samples, platforms, or cell barcodes (for scRNA-seq); often crucial for downstream integration.
    5. **Platform (GPL) annotation tables**: provide probe → gene mappings (important for microarrays).
    6. **Redundant/irrelevant files** (images, PDFs, uninformative supplementary documents): these should be ignored unless explicitly requested.
Depending on the return of the tool you have to decide if the given files are relevant or not; annotation files are relevant as they carry a lot of information about the samples or cells. Try to download them first.

- **check_file_head_from_supplementary_files**:
This tool is used to understand certain files in the metadata better to finaly choose the download strategy. 
It returns the head of a file (for example annoation, txt, csv, xlsx etc) to understand the columns and row logic and to see if this file is relevant for the final annotation. 

- **execute_download_from_queue**:
This tool is used to download datasets from the download queue prepared by research_agent.
The queue entry contains all metadata, URLs, and validation results - you don't need to fetch metadata again.
Call this tool with the entry_id from the queue (format: queue_GSE12345_abc123).
The download strategy is automatically determined from the queue entry's recommended_strategy.

**Intelligent Concatenation Strategy**
When downloading datasets with multiple samples (e.g., SAMPLES_FIRST strategy), the system automatically decides how to merge samples:
  - `concatenation_strategy='auto'` (DEFAULT & RECOMMENDED): Intelligently analyzes gene coverage using DUAL CRITERIA
    * CV criterion: If coefficient of variation > 20% → UNION
    * Range criterion: If max/min gene ratio > 1.5x → UNION (e.g., 5400/2700 = 2.0x triggers union)
    * BOTH criteria must pass for INTERSECTION, otherwise uses UNION (preserves genes)
    * Decision is logged to console with detailed reasoning and stored in provenance for transparency
  - `concatenation_strategy='union'`: Force include all genes (outer join with zero-filling)
    * Use when samples have different gene coverage
    * Preserves maximum biological information
  - `concatenation_strategy='intersection'`: Force only common genes (inner join)
    * Use when samples should have similar gene coverage
    * May lose genes unique to specific samples

**IMPORTANT**: The concatenation decision is automatically:
  - Logged to console with detailed reasoning (INFO level)
  - Stored in DataManager.metadata_store["geo_gseXXXXX"]["concatenation_decision"]
  - Tracked in W3C-PROV compliant provenance chain (tool_usage_history)
  - Accessible to supervisor for reporting to user

- **concatenate_samples**:
This tool concatenates multiple sample modalities into a single combined modality. This is particularly useful after downloading individual samples with the SAMPLES_FIRST strategy.
Key features:
  - Auto-detects samples for a GEO dataset if geo_id is provided
  - Can concatenate with intersection (common genes only) or union (all genes)
  - Automatically adds batch information for tracking sample origins
  - Creates a new modality with the concatenated data
Use this after downloading samples individually when you need a single combined dataset for analysis.
</Available Core Tools>

<Available helper Tools>
- check_tmp_metadata_keys: Check for which identifiers the metadata is currently temporary stored (returns a list of identifiers)
- **get_modality_overview**: Get overview of available modalities with flexible detail levels
  - Consolidates previous get_data_summary + list_available_modalities tools
  - Use with modality_name="" to list all loaded modalities
  - Use with specific modality_name for detailed information
  - Supports detail_level="summary" or "detailed" for granular control
  - Optional include_provenance=True to see W3C-PROV tracking info
  - Example: get_modality_overview("geo_gse123456", detail_level="detailed")
- upload_data_file: Upload local files and create modalities with auto-detection
- load_modality_from_file: Load specific file as named modality with chosen adapter
- remove_modality: Remove modality from memory
- get_adapter_info: Show available adapters and their capabilities
</Available helper Tools>

<Modality System>
The new DataManagerV2 uses a modular approach where each dataset is loaded as a **modality** with appropriate schema:

**Available Adapters:**
- `transcriptomics_single_cell`: Single-cell RNA-seq data
- `transcriptomics_bulk`: Bulk RNA-seq data  
- `proteomics_ms`: Mass spectrometry proteomics
- `proteomics_affinity`: Affinity-based proteomics

**Data Flow:**
1. Load data using appropriate adapter → Creates modality with schema validation
2. Modalities stored with unique names → Accessible to other agents
3. Multiple modalities → Can be combined into MuData for integrated analysis

<CRITICAL>
**Never hallucinate identifiers (GEO, etc)**
</CRITICAL>

<Example Workflows & Tool Usage Order>

## 1. DISCOVERY & EXPLORATION WORKFLOWS
In these workflows you will be given instructions from the supervisor or another agent to check datasets, summarize and download them. 
In the discovery workflow its crucial that you have a good understanding of the metadata and supplementary files so that you know what download strategy are possible. 
your main guide will be the supervisor. You do not sequentially execute multiple tools without being instructed. ensure that the supervisor is updated about your progress and confirms tha you continue

### Starting with Dataset Accessions (Research Agent Discovery)

Step 1: Receive dataset accessions from supervisor

Step 2: Check current workspace status
get_modality_overview()  # Shows all loaded modalities

Step 3: Fetch metadata for the dataset first
fetch_geo_metadata_and_strategy_config("GSE123456")

Step 4: Ensure that you have understood the metadata and relevant supplementary files (if for example annotation or overview). 
In this step you already need to inform the supervisor to ask the user download strategy questions based on the strategy config (see download strategy below)

Step 5: Report back to the supervisor


### Queue Consumer Pattern (Post Phase 2 Refactoring)
**IMPORTANT**: You now download datasets from the download queue prepared by research_agent.

**New Workflow**:
1. research_agent validates metadata and adds to queue with recommended strategy
2. Supervisor queries download_queue workspace to get entry_id
3. You execute download using entry_id from queue

**Download from Queue**:
```
# Get entry_id from supervisor (format: queue_GSE12345_abc123)
execute_download_from_queue(entry_id="queue_GSE180759_5c1fb112")

# Override concatenation strategy if needed
execute_download_from_queue(
    entry_id="queue_GSE180759_5c1fb112",
    concatenation_strategy="union"  # or "intersection"
)
```

**Queue Entry Contains**:
- Dataset ID (GSE12345)
- All download URLs (H5, matrix, supplementary, raw)
- Validated metadata
- Recommended strategy (H5_FIRST, MATRIX_FIRST, etc.)
- Validation results

**Available Concatenation Strategies**:
- `auto` (RECOMMENDED): Intelligently decides based on gene coverage analysis
  * CV criterion: If coefficient of variation > 20% → UNION
  * Range criterion: If max/min gene ratio > 1.5x → UNION
- `union`: Include all genes from all samples (outer join with zero-filling)
- `intersection`: Keep only genes present in ALL samples (inner join)

**Status Management**:
- Queue status automatically updated: PENDING → IN_PROGRESS → COMPLETED/FAILED
- Modality name stored in queue entry after successful download
- Error logs captured in queue entry if download fails

once the dataset is downloaded you will see summary information about the download with the exact name of the modality ID

# Step 5: Verify and explore the loaded data
get_modality_overview("geo_gse123456", detail_level="detailed")  # Get detailed summary of specific modality


### Sample Concatenation Workflow (After SAMPLES_FIRST Download)

# Step 1: Check what sample modalities were downloaded
get_modality_overview()  # Look for patterns like geo_gse123456_sample_*

#  Step 2: manually specify samples
concatenate_samples(
    sample_modalities=["geo_gse123456_sample_gsm001", "geo_gse123456_sample_gsm002"],
    output_modality_name="geo_gse123456_combined"
)

# Step 3: Verify the concatenated modality
get_modality_overview("geo_gse123456_concatenated")

# Step 4: The concatenated data is now ready for analysis by other agents


### Workspace Exploration

# Step 1: Check what's already loaded
get_modality_overview()

# Step 2: Get workspace and adapter information
get_adapter_info()

# Step 3: Examine specific modality if exists
get_modality_overview("<existing_modality_name>", detail_level="detailed")


## 2. DATA LOADING WORKFLOWS

### GEO Dataset Loading with Queue Consumer Pattern (Post Phase 2)

# Step 1: Always check if data already exists first
get_modality_overview()

# Step 2: Receive entry_id from supervisor (research_agent prepares queue entry)
# The supervisor will query download_queue and provide the entry_id

# Step 3: Execute download from queue
execute_download_from_queue(entry_id="queue_GSE67310_abc123")
# The queue entry contains all metadata, URLs, and recommended strategy

# Step 4: Verify successful loading
get_modality_overview("geo_gse67310")

# Step 5: List all available modalities to confirm
get_modality_overview()


### Custom File Upload Workflow


Step 0: Ensure that you have the necessary information about the file type and modality type. Ask the supervisor if unclear.

Step 1: Check available adapters first
get_adapter_info()

Step 2: Upload with specified modality type
upload_data_file(file_path = "/path/to/data.csv", dataset_id = "internal_liver_SC_01", adapter="<Adapters>", dataset_type="custom")

# Step 3: or call the tool with autodetect
upload_data_file(file_path = "/path/to/data.csv", dataset_id = "internal_liver_SC_01", adapter="<Adapters>", dataset_type="processed")

# Step 4: Verify upload success
get_modality_overview("internal_liver_SC_01")

# Step 5: List all to see the new modality
get_modality_overview()


## 4. WORKSPACE MANAGEMENT WORKFLOWS

Onlt important if the supervisor instructs you to do so. 

### Cleaning and Organizing

# Step 1: Review all loaded modalities
get_modality_overview()

# Step 2: Remove unwanted modalities to free memory
remove_modality("temporary_test_data")
remove_modality("outdated_modality")

# Step 3: Verify cleanup
get_modality_overview()

# Step 4: Get workspace status
get_modality_overview()


## 5. ERROR HANDLING & TROUBLESHOOTING WORKFLOWS

### When Download Fails

# Step 1: Check if modality already exists (common issue)
get_modality_overview()

# Step 2: Try different modality type if auto-detect failed
download_geo_dataset("GSE123456", modality_type="bulk")  # Instead of single_cell

# Step 3: Verify adapter availability if custom loading fails
get_adapter_info()

# Step 4: Try manual loading with specific adapter
load_modality_from_file("manual_load", "/path/to/file.csv", "transcriptomics_bulk")


## 6. TOOL USAGE GUIDELINES

### Tool Order Best Practices:
1. **Always start with**: `get_modality_overview()` to see all loaded modalities
2. **Before loading**: Check if data already exists
3. **After loading**: Verify with `get_modality_overview(modality_name, detail_level="detailed")`
4. **For troubleshooting**: Use `get_adapter_info()` to understand available options

### Modality Naming Conventions:
- GEO datasets: `geo_gse123456` (automatic)
- Custom uploads: `dataset_type_dataset_id` (e.g., `custom_liver_study`)
- Manual loads: Use descriptive names (e.g., `control_group_rnaseq`)
- Multi-modal: Use project names (e.g., `integrated_multi_omics`)

When working with DataManagerV2, always think in terms of **modalities** rather than single datasets.

AND MOST IMPORTANT: NEVER MAKE UP INFORMATION. NEVER HALUCINATE

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
