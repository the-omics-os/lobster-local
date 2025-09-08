"""
Data Expert Agent for handling all data fetching, downloading, and extraction.

This agent is responsible for managing all data acquisition operations using
the new modular DataManagerV2 system with proper modality handling and 
multi-omics support.
"""

from typing import List, Dict
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

import pandas as pd
from datetime import date

from lobster.agents.state import DataExpertState
from lobster.agents.data_expert_assistant import DataExpertAssistant, StrategyConfig
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def data_expert(
    data_manager: DataManagerV2, 
    callback_handler=None, 
    agent_name: str = "data_expert_agent",
    handoff_tools: List = None
):  
    """Create data expert agent for handling all data operations with DataManagerV2."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('data_expert')
    llm = ChatBedrockConverse(**model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
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
    def fetch_geo_metadata_and_strategy_config(geo_id: str, data_source: str = 'geo') -> str:
        """
        Fetch and validate GEO dataset metadata without downloading the full dataset.
        Use this FIRST before download_geo_dataset to preview dataset information.
        
        Args:
            geo_id: GEO accession number (e.g., GSE12345)
            
        Returns:
            str: Formatted metadata summary with validation results and recommendation
        """
        try:
            # Clean the GEO ID
            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith('GSE'):
                return f"Invalid GEO ID format: {geo_id}. Expected format: GSE12345"
            
            logger.info(f"Fetching metadata for GEO dataset: {clean_geo_id}")
            
            # Use GEOService to fetch metadata only
            from lobster.tools.geo_service import GEOService
            
            console = getattr(data_manager, 'console', None)
            geo_service = GEOService(data_manager, console=console)

            #------------------------------------------------
            # Check if metadata already in store
            #------------------------------------------------
            if clean_geo_id in data_manager.metadata_store:
                if data_manager.metadata_store[clean_geo_id]['strategy_config']:
                    logger.debug(f"Metadata already stored for: {geo_id}. returning summary")
                    summary = geo_service._format_metadata_summary(
                        clean_geo_id,
                        data_manager.metadata_store[clean_geo_id]
                    )
                    return summary
                logger.info(f"{clean_geo_id} is in metadata but no strategy config has been generated yet. Proceeding doing so")
                        
            #------------------------------------------------
            # If not fetch and return metadata & val res 
            #------------------------------------------------
            # Fetch metadata only (no expression data download)
            metadata, validation_result = geo_service.fetch_metadata_only(clean_geo_id)

            #------------------------------------------------
            # Extract strategy config using assistant
            #------------------------------------------------
            strategy_config = assistant.extract_strategy_config(metadata, clean_geo_id)
            
            if not strategy_config:
                logger.warning(f"Failed to extract strategy config for {clean_geo_id}")
                return 'Failed with fetching geo metadata. Try again'


            #------------------------------------------------
            # store in DataManager
            #------------------------------------------------
            # Store metadata in data_manager for future use
            data_manager.metadata_store[clean_geo_id] = {
                'metadata': metadata,
                'validation': validation_result,
                'fetch_timestamp': pd.Timestamp.now().isoformat(),
                'data_source': data_source,
                'strategy_config': strategy_config.model_dump() if 'strategy_config' in locals() else {}
            }
            
            # Log the metadata fetch operation
            data_manager.log_tool_usage(
                tool_name="fetch_geo_metadata_and_strategy_config",
                parameters={"geo_id": clean_geo_id, "data_source": data_source},
                description=f"Fetched metadata for GEO dataset {clean_geo_id} using {data_source}"
            )
            
            # Format comprehensive metadata summary
            base_summary = geo_service._format_metadata_summary(clean_geo_id, metadata, validation_result)
            
            # Add strategy config section if available
            if strategy_config:
                strategy_section = assistant.format_strategy_section(strategy_config)
                summary = base_summary + strategy_section
            else:
                summary = base_summary
            
            logger.debug(f"Successfully fetched and validated metadata for {clean_geo_id} using {data_source}")

            return summary
                
        except Exception as e:
            logger.error(f"Error fetching GEO metadata for {geo_id}: {e}")
            return f"Error fetching metadata: {str(e)}"


    @tool
    def check_file_head_from_supplementary_files(geo_id: str, filename: str) -> str:
        """
        Print the head of a file in the supplementary files
        """
        #------------------------------------------------
        # find name in supplementary 
        #------------------------------------------------
        # iterate through data_manager
            # Use GEOService to fetch metadata only
        from lobster.tools.geo_service import GEOService
        
        console = getattr(data_manager, 'console', None)
        geo_service = GEOService(data_manager, console=console)  

        target_url = ''
        for urls in data_manager.metadata_store[geo_id]['metadata']['supplementary_file']:
            if isinstance(urls, str):
                if filename in urls:
                    target_url = urls
                    # geo_service._download_and_parse_file(target_url)
                    file_head =  geo_service.geo_parser.show_dynamic_head(target_url)
                    return file_head.get('head', "Error in fetching head: nothing to fetch")

            msg = f"why is url not a str?? -> {urls}. Instead is type {type(urls)}"
            logger.warning(msg)
            return msg
                
    @tool
    def download_geo_dataset(
        geo_id: str, modality_type: str = "single_cell", manual_strategy_override = 'MATRIX_FIRST', **kwargs) -> str:
        """
        Download dataset from GEO using accession number and load as a modality.
        IMPORTANT: Use fetch_geo_metadata_and_strategy_config FIRST to preview dataset before downloading.
        
        Args:
            geo_id: GEO accession number (e.g., GSE12345)
            modality_type: Type of data modality ('single_cell', 'bulk', or 'auto_detect')
            manual_strategy_override: Optional manual override for download strategy
            
        Returns:
            str: Summary of downloaded data with modality information
        """
        try:
            # Clean the GEO ID
            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith('GSE'):
                return f"Invalid GEO ID format: {geo_id}. Expected format: GSE12345"
            
            logger.info(f"Processing GEO dataset request: {clean_geo_id} (modality: {modality_type})")
            
            # Check if modality already exists
            modality_name = f"geo_{clean_geo_id.lower()}"
            existing_modalities = data_manager.list_modalities()
            
            if modality_name in existing_modalities:
                adata = data_manager.get_modality(modality_name)
                return f"""Found existing modality '{modality_name}'!

📊 Matrix: {adata.n_obs} obs × {adata.n_vars} vars
💾 No download needed - using cached modality
⚡ Ready for immediate analysis!

Use this modality for quality control, filtering, or downstream analysis."""
            
            if clean_geo_id not in data_manager.metadata_store:
                return f"Error: You forgot to fetch metdata. First go fetch the metadata for {clean_geo_id}"
            
            # Use enhanced GEOService with modular architecture and fallbacks
            from lobster.tools.geo_service import GEOService
            
            console = getattr(data_manager, 'console', None)
            geo_service = GEOService(data_manager, console=console)
            
            # Use the enhanced download_dataset method (handles all scenarios with fallbacks)
            result = geo_service.download_dataset(
                geo_id=clean_geo_id, 
                # modality_type=modality_type,
                manual_strategy_override=manual_strategy_override,
                **kwargs) #This kwargs contains the config dict
            
            return result
                
        except Exception as e:
            logger.error(f"Error processing GEO dataset {geo_id}: {e}")
            return f"Error processing dataset: {str(e)}"


    @tool
    def get_data_summary(modality_name: str = "") -> str:
        """
        Get summary of currently loaded modalities or list available datasets.
        
        Args:
            modality_name: Optional modality name (leave empty to list all)
            
        Returns:
            str: Data summary or list of available modalities
        """
        try:
            if modality_name == "" or modality_name.lower() == "all":
                # List all loaded modalities
                modalities = data_manager.list_modalities()
                if not modalities:
                    return "No modalities currently loaded. Use download_geo_dataset to load data."
                
                response = "Currently loaded modalities:\n\n"
                for mod_name in modalities:
                    adata = data_manager.get_modality(mod_name)
                    metrics = data_manager.get_quality_metrics(mod_name)
                    response += f"- **{mod_name}**: {adata.n_obs} obs × {adata.n_vars} vars\n"
                    if 'total_counts' in metrics:
                        response += f"  Total counts: {metrics['total_counts']:,.0f}\n"
                
                # Also show workspace status
                workspace_status = data_manager.get_workspace_status()
                response += f"\nWorkspace: {workspace_status['workspace_path']}"
                response += f"\nAvailable adapters: {', '.join(workspace_status['registered_adapters'])}"
                
                return response
            
            else:
                # Get summary of specific modality
                try:
                    adata = data_manager.get_modality(modality_name)
                    metrics = data_manager.get_quality_metrics(modality_name)
                    
                    response = f"Modality: {modality_name}\n"
                    response += f"Shape: {adata.n_obs} obs × {adata.n_vars} vars\n"
                    response += f"Obs columns: {list(adata.obs.columns)[:5]}\n"
                    response += f"Var columns: {list(adata.var.columns)[:5]}\n"
                    
                    if 'total_counts' in metrics:
                        response += f"Total counts: {metrics['total_counts']:,.0f}\n"
                    if 'mean_counts_per_obs' in metrics:
                        response += f"Mean counts per obs: {metrics['mean_counts_per_obs']:.1f}\n"
                    
                    return response
                except ValueError:
                    return f"Modality '{modality_name}' not found. Available modalities: {data_manager.list_modalities()}"
                
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return f"Error getting data summary: {str(e)}"

    @tool
    def upload_data_file(
        file_path: str, 
        dataset_id: str, 
        adapter: str = "auto_detect",
        dataset_type: str = "custom"
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
                return f"Modality type must be a string"
            elif not adapter:
                return f"Modality type can not be None"            
            
            # Auto-detect modality type if requested
            if adapter == "auto_detect":
                # Read file to detect data characteristics
                import pandas as pd
                try:
                    df = pd.read_csv(file_path, index_col=0, nrows=10)  # Sample first 10 rows
                    n_cols = df.shape[1]
                    
                    # Heuristics for detection
                    if n_cols > 5000:
                        adapter = "transcriptomics_single_cell"
                    elif n_cols < 1000:
                        adapter = "proteomics_ms"
                    else:
                        adapter = "transcriptomics_bulk"  # Middle ground
                    
                    logger.info(f"Auto-detected modality type: {adapter} (based on {n_cols} features)")
                except:
                    adapter = "single_cell"  # Safe default

            modality_name = f"{dataset_type}_{dataset_id}"
            
            # Load using appropriate adapter
            adata = data_manager.load_modality(
                name=modality_name,
                source=file_path,
                adapter=adapter,
                validate=True,
                dataset_id=dataset_id,
                dataset_type=dataset_type
            )
            
            # Save to workspace
            save_path = f"{dataset_id}_{adapter}.h5ad"
            saved_path = data_manager.save_modality(modality_name, save_path)
            
            # Get quality metrics
            metrics = data_manager.get_quality_metrics(modality_name)
            
            return f"""Successfully uploaded and processed file {file_path.name}!

📊 Modality: '{modality_name}' ({adata.n_obs} obs × {adata.n_vars} vars)
🔬 Data type: {adapter}
🎯 Adapter: {adapter}
💾 Saved to: {save_path}
📈 Quality metrics: {len([k for k, v in metrics.items() if isinstance(v, (int, float))])} metrics calculated

The dataset is now available as modality '{modality_name}' for analysis."""
            
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {e}")
            return f"Error uploading file: {str(e)}"

    @tool
    def load_modality_from_file(
        modality_name: str,
        file_path: str, 
        adapter: str,
        **kwargs
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
            from pathlib import Path
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
                **kwargs
            )
            
            # Get quality metrics
            metrics = data_manager.get_quality_metrics(modality_name)
            
            return f"""Successfully loaded modality '{modality_name}'!

📊 Shape: {adata.n_obs} obs × {adata.n_vars} vars
🔬 Adapter: {adapter}
📁 Source: {file_path.name}
📈 Quality metrics: {len([k for k, v in metrics.items() if isinstance(v, (int, float))])} metrics calculated

The modality is now available for analysis and can be used by other agents."""
            
        except Exception as e:
            logger.error(f"Error loading modality {modality_name}: {e}")
            return f"Error loading modality: {str(e)}"

    @tool
    def list_available_modalities() -> str:
        """
        List all currently loaded modalities and their details.
        
        Returns:
            str: Formatted list of available modalities
        """
        try:
            modalities = data_manager.list_modalities()
            
            if not modalities:
                return "No modalities currently loaded. Use download_geo_dataset or upload_data_file to load data."
            
            response = f"Currently loaded modalities ({len(modalities)}):\n\n"
            
            for mod_name in modalities:
                adata = data_manager.get_modality(mod_name)
                response += f"**{mod_name}**:\n"
                response += f"  - Shape: {adata.n_obs} obs × {adata.n_vars} vars\n"
                response += f"  - Obs columns: {len(adata.obs.columns)} ({', '.join(list(adata.obs.columns)[:3])}...)\n"
                response += f"  - Var columns: {len(adata.var.columns)} ({', '.join(list(adata.var.columns)[:3])}...)\n"
                if adata.layers:
                    response += f"  - Layers: {', '.join(list(adata.layers.keys()))}\n"
                response += "\n"
            
            # Add workspace information
            workspace_status = data_manager.get_workspace_status()
            response += f"Workspace: {workspace_status['workspace_path']}\n"
            response += f"Available adapters: {len(workspace_status['registered_adapters'])}\n"
            response += f"Available backends: {len(workspace_status['registered_backends'])}"
            
            return response
                
        except Exception as e:
            logger.error(f"Error listing available data: {e}")
            return f"Error listing available data: {str(e)}"

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
        modality_names: List[str],
        output_name: str = "multimodal_analysis"
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
            missing = [name for name in modality_names if name not in available_modalities]
            if missing:
                return f"Modalities not found: {missing}. Available: {available_modalities}"
            
            # Create MuData object
            mdata = data_manager.to_mudata(modalities=modality_names)
            
            # Save the MuData object
            mudata_path = f"{output_name}.h5mu"
            saved_path = data_manager.save_mudata(mudata_path, modalities=modality_names)
            
            return f"""Successfully created MuData from {len(modality_names)} modalities!

🔗 Combined modalities: {', '.join(modality_names)}
📊 Global shape: {mdata.n_obs} obs across {len(mdata.mod)} modalities
💾 Saved to: {mudata_path}
🎯 Ready for integrated multi-omics analysis!

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
                response += f"  - Supported formats: {', '.join(info['supported_formats'])}\n"
                response += f"  - Schema: {info['schema']['modality']}\n"
                response += "\n"
            
            response += "\nUse these adapter names when loading data with load_modality_from_file or upload_data_file."
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting adapter info: {e}")
            return f"Error getting adapter info: {str(e)}"

    base_tools = [
        #CORE
        fetch_geo_metadata_and_strategy_config,
        check_file_head_from_supplementary_files,
        download_geo_dataset,
        #HELPER
        check_tmp_metadata_keys,
        get_data_summary,
        upload_data_file,
        list_available_modalities,
        load_modality_from_file,
        remove_modality,
        get_adapter_info
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
4. **Provide data access** to other agents via modality names
5. **Maintain workspace** with proper organization and provenance tracking
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

- **download_geo_dataset**: 
This tool is used after understanding the metadata logic of a GEO entry. This tool downloads data from GEO and load as modality. 
Before using this tool always fetch metadata first and get a good understand what the relevant files are. 
</Available Core Tools>

<Available helper Tools>
- check_tmp_metadata_keys: Check for which identifiers the metadata is currently temporary stored (returns a list of identifiers) 
- get_data_summary: Get summary of loaded modalities or specific modality
- upload_data_file: Upload local files and create modalities with auto-detection
- list_available_modalities: Show all currently loaded modalities with details
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
```bash
Step 1: Receive dataset accessions from supervisor

Step 2: Check current workspace status  
get_data_summary()  # Shows all loaded modalities

Step 3: Fetch metadata for the dataset first
fetch_geo_metadata_and_strategy_config("GSE123456")

Step 4: Ensure that you have understood the metadata and relevant supplementary files (if for example annotation or overview). 
In this step you already need to inform the supervisor to ask the user download strategy questions based on the strategy config (see download strategy below)

Step 5: Report back to the supervisor


### Download strategy
There are different ways on how to load the dataset and each way depends on the user requirement. 
Entries in GEO either have already processed matrizes that can be taken directly, but there is a lack of control about filtering and pre-processing. 
Entries can also have raw UMI count matrizes which contain the transcript counts but the sample or cell annotation is in another annotation file. 
Entries can also have only raw files as large .tar files. In this case the best option would be to go directly to the samples
And in rare cases entries only contain raw .tar files which need to be extracted first and the file structure needs to be understood

For each case there is another strategy to use: 
'MATRIX_FIRST' : pre-processed matrix available, annotation data available
'RAW_FIRST' : raw matrix available, annotation data available
'H5_FIRST' : Prepared annotated object available
'SAMPLES_FIRST' : raw-data must be available. Mostly user choice to work directly with the raw count data.
'SUPPLEMENTARY_FIRST' : compressed large supplementary file available which contains more information. Needs to be further checked. 

<Fallbacks>
Sometimes these strategies do not work, even multiple ones. In this case the downloaded folders need to be analyzed manually.
</Fallbacks>

<important: Before choosing a download strategy>
You will receive metadata information AND supplementary files information. Its crucial that you tell the supervisor to ask the user which approach they want to take or if they want you (data expert) to decide (no input)
Always report back after fetching the metadata. Once confirmed by the supervisor you can continue
</important>

Step 1: Download the dataset with appropriate modality type and strategy:
Scenario A: Let the system decide which strategy
download_geo_dataset("<GEO ID>", modality_type="<Adapters>")

Scenario B: Download the dataset with appropriate modality type, choice of strategy was MATRIX_FIRST:
download_geo_dataset("<GEO ID>", modality_type="<Adapters>", manual_strategy_override='MATRIX_FIRST')

Scenario C: Download the dataset with appropriate modality type, choice of strategy was SAMPLES_FIRST:
download_geo_dataset("<GEO ID>", modality_type="<Adapters>", manual_strategy_override='SAMPLES_FIRST')

Other strategies include: 
RAW_FIRST               # Prioritize raw UMI/count matrices
SAMPLES_FIRST           # Download individual samples
H5_FIRST                # Prioritize H5/H5AD files
ARCHIVE_FIRST           # Extract from archives first
FALLBACK                # Use fallback mechanisms

once the dataset is downloaded you will see summary information about the download with the exact name of the modality ID

# Step 5: Verify and explore the loaded data
get_data_summary("geo_gse123456")  # Get detailed summary of specific modality
```

### Workspace Exploration
```bash
# Step 1: Check what's already loaded
list_available_modalities()

# Step 2: Get workspace and adapter information
get_adapter_info()

# Step 3: Examine specific modality if exists
get_data_summary("<existing_modality_name>")
```

## 2. DATA LOADING WORKFLOWS

### GEO Dataset Loading with Metadata-First Approach (Most Common)
```bash
# Step 1: Always check if data already exists first
get_data_summary()

# Step 2: REQUIRED - Fetch metadata first to preview dataset
fetch_geo_metadata_and_strategy_config("GSE67310")

# Step 3: Review metadata summary, then download with appropriate modality type
download_geo_dataset("GSE67310", modality_type="<adapter>")
# OR with auto-detection based on metadata recommendation:
download_geo_dataset("GSE67310", modality_type="<adapter>")

# Step 4: Verify successful loading
get_data_summary("geo_gse67310")

# Step 5: List all available modalities to confirm
list_available_modalities()
```

### Custom File Upload Workflow
```bash

Step 0: Ensure that you have the necessary information about the file type and modality type. Ask the supervisor if unclear.

Step 1: Check available adapters first
get_adapter_info()

Step 2: Upload with specified modality type
upload_data_file(file_path = "/path/to/data.csv", dataset_id = "internal_liver_SC_01", adapter="<Adapters>", dataset_type="custom")

# Step 3: or call the tool with autodetect
upload_data_file(file_path = "/path/to/data.csv", dataset_id = "internal_liver_SC_01", adapter="<Adapters>", dataset_type="processed")

# Step 4: Verify upload success
get_data_summary("internal_liver_SC_01")

# Step 5: List all to see the new modality
list_available_modalities()
```

## 4. WORKSPACE MANAGEMENT WORKFLOWS

Onlt important if the supervisor instructs you to do so. 

### Cleaning and Organizing
```bash
# Step 1: Review all loaded modalities
list_available_modalities()

# Step 2: Remove unwanted modalities to free memory
remove_modality("temporary_test_data")
remove_modality("outdated_modality")

# Step 3: Verify cleanup
list_available_modalities()

# Step 4: Get workspace status
get_data_summary()
```

## 5. ERROR HANDLING & TROUBLESHOOTING WORKFLOWS

### When Download Fails
```bash
# Step 1: Check if modality already exists (common issue)
get_data_summary()

# Step 2: Try different modality type if auto-detect failed
download_geo_dataset("GSE123456", modality_type="bulk")  # Instead of single_cell

# Step 3: Verify adapter availability if custom loading fails
get_adapter_info()

# Step 4: Try manual loading with specific adapter
load_modality_from_file("manual_load", "/path/to/file.csv", "transcriptomics_bulk")
```

## 6. TOOL USAGE GUIDELINES

### Tool Order Best Practices:
1. **Always start with**: `list_available_modalities()` or `get_data_summary()`
2. **Before loading**: Check if data already exists
3. **After loading**: Verify with `get_data_summary(modality_name)`
4. **For troubleshooting**: Use `get_adapter_info()` to understand available options

### Modality Naming Conventions:
- GEO datasets: `geo_gse123456` (automatic)
- Custom uploads: `dataset_type_dataset_id` (e.g., `custom_liver_study`)
- Manual loads: Use descriptive names (e.g., `control_group_rnaseq`)
- Multi-modal: Use project names (e.g., `integrated_multi_omics`)

When working with DataManagerV2, always think in terms of **modalities** rather than single datasets.

AND MOST IMPORTANT: NEVER MAKE UP INFORMATION. NEVER HALUCINATE

Today's date is {date}.
""".format(date=date.today())

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=DataExpertState
    )
