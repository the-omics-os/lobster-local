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
from pydantic import Field, BaseModel

from datetime import date

from lobster.agents.state import DataExpertState
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Define StrategyConfig with all field descriptions
class StrategyConfig(BaseModel):
    summary_file_name : str     = Field(default="", description="name of the summary or overview file (has 'GSE' in the name). Example: 'GSE131907_Lung_Cancer_Feature_Summary'")
    summary_file_type : str     = Field(default="", description="Filetype of the Summary File. Example: 'xlsx'")
    processed_matrix_name : str     = Field(default="", description="name of the TARGET processed (log2, tpm, normalized, etc) matrix file (preference is non R objects). Example: 'GSE131907_Lung_Cancer_normalized_log2TPM_matrix'")
    processed_matrix_filetype : str = Field(default="", description="filetype of the TARGET processed file (preference is non R objects). Example: 'txt'")
    raw_UMI_like_matrix_name : str  = Field(default="", description="name of the raw TARGET UMI-like (UMI, raw, tar) matrix file (preference is non R objects). Example: 'GSE131907_Lung_Cancer_raw_UMI_matrix'")
    raw_UMI_like_matrix_filetype : str = Field(default="", description="filetype of the TARGET raw UMI-like file (preference is non R objects). Example: 'txt'")
    cell_annotation_name : str      = Field(default="", description="Filename of the file which likely is linked to the cell annotation. Example:'GSE131907_Lung_Cancer_cell_annotation'")
    cell_annotation_filetype : str  = Field(default="", description="Filetype of cell annotation file. Example: 'txt'")
    raw_data_available: bool        = Field(default=False, description="If based on the metadata raw data is available for the samples. Sometimes it stated in the study description")


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

    
    @tool
    def check_tmp_metadata_keys() -> List:
        """
        Check which metadata is temporarelly stored
        """
        return data_manager.metadata_store.keys()
    
    # Define tools for data operations
    @tool
    def fetch_geo_metadata(geo_id: str, data_source: str = 'geo') -> str:
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
                logger.debug(f"Metadata already stored for: {geo_id}. returning summary")
                summary = geo_service._format_metadata_summary(
                    clean_geo_id,
                    data_manager.metadata_store[clean_geo_id]
                )
                return summary
                        
            #------------------------------------------------
            # If not fetch and return metadata & val res 
            #------------------------------------------------
            # Fetch metadata only (no expression data download)
            metadata, validation_result = geo_service.fetch_metadata_only(clean_geo_id)

            #------------------------------------------------
            # Extract strategy config using LLM
            #------------------------------------------------
            import json
            

            # Extract key metadata fields
            title = metadata.get('title', 'N/A')
            summary = metadata.get('summary', 'N/A')
            overall_design = metadata.get('overall_design', 'N/A')
            supplementary_files = metadata.get('supplementary_file', [])
            
            # Prepare the context for LLM
            metadata_context = f"""
Title: {title}

Summary: {summary}

Overall Design: {overall_design}

Supplementary Files:
{supplementary_files if supplementary_files else 'No supplementary files listed'}
"""

            # Get the schema from StrategyConfig class
            strategy_schema = StrategyConfig.model_json_schema()
            
            # Create system prompt using the schema
            system_prompt = f"""You are a bioinformatics expert analyzing GEO dataset metadata. Your task is to extract file information and populate a StrategyConfig object.

The StrategyConfig schema is:
{json.dumps(strategy_schema, indent=2)}

Important notes:
1. For each file type, extract the filename WITHOUT the extension (e.g., "GSE131907_Lung_Cancer_Feature_Summary" not "GSE131907_Lung_Cancer_Feature_Summary.xlsx")
2. Extract the file extension separately (e.g., "xlsx", "txt", "csv")
3. Prefer non-R objects (.txt, .csv) over .rds files when multiple options exist
4. Check the summary and overall design text for mentions of raw data availability
5. If a field cannot be determined from the metadata, populate field with "null" so it becomes None in Python

Return only a valid JSON object that matches the StrategyConfig schema."""

            # Initialize LLM with settings
            llm_params = settings.get_agent_llm_params('assistant')
            llm = ChatBedrockConverse(**llm_params)
            
            try:
                # Create the prompt
                prompt = f"""Given this GEO dataset metadata, extract the file information into a StrategyConfig:

{metadata_context}

Return only a valid JSON object that conforms to the StrategyConfig schema provided in the system prompt.
"""

                # Invoke the LLM
                response = llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ])
                
                # Extract the JSON from the response
                response_text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content)
                
                # Parse the JSON response
                # Try to extract JSON from the response in case it's wrapped in markdown or other text
                import re
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    strategy_dict = json.loads(json_match.group())
                else:
                    strategy_dict = json.loads(response_text)
                
                # Create StrategyConfig object #FIXME
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                strategy_config = StrategyConfig(**strategy_dict)
                #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                
                logger.info(f"Successfully extracted strategy config for {clean_geo_id}")
                                
            except Exception as e:
                logger.warning(f"Failed to extract strategy config using LLM: {e}")
                # Create empty strategy config as fallback
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
                tool_name="fetch_geo_metadata_with_fallback",
                parameters={"geo_id": clean_geo_id, "data_source": data_source},
                description=f"Fetched metadata for GEO dataset {clean_geo_id} using {data_source}"
            )
            
            # Format comprehensive metadata summary
            base_summary = geo_service._format_metadata_summary(clean_geo_id, metadata, validation_result)
            
            # Add strategy config section if available
            if 'strategy_config' in locals() and strategy_config:
                config_dict = data_manager.metadata_store[clean_geo_id]['strategy_config']
                
                strategy_section = f"""

📁 **Extracted File Strategy Configuration:**
- **Summary File:** {config_dict.get('summary_file_name', 'Not found')}{'.'+config_dict.get('summary_file_type') if config_dict.get('summary_file_type') else ''}
- **Processed Matrix:** {config_dict.get('processed_matrix_name', 'Not found')}{'.'+config_dict.get('processed_matrix_filetype') if config_dict.get('processed_matrix_filetype') else ''})
- **Raw/UMI Matrix:** {config_dict.get('raw_UMI_like_matrix_name', 'Not found')}{'.'+config_dict.get('raw_UMI_like_matrix_filetype') if config_dict.get('raw_UMI_like_matrix_filetype') else ''})
- **Cell Annotations:** {config_dict.get('cell_annotation_name', 'Not found')}{'.'+config_dict.get('cell_annotation_filetype') if config_dict.get('cell_annotation_filetype') else ''})
- **Raw Data Available:** {'Yes' if config_dict.get('raw_data_available', True) else 'No (see study description for details)'}
"""
                
                summary = base_summary + strategy_section
            else:
                summary = base_summary
            
            logger.debug(f"Successfully fetched and validated metadata for {clean_geo_id} using {data_source}")

            return summary
                
        except Exception as e:
            logger.error(f"Error fetching GEO metadata for {geo_id}: {e}")
            return f"Error fetching metadata: {str(e)}"

    @tool
    def download_geo_dataset(
        geo_id: str, modality_type: str = "single_cell", **kwargs) -> str:
        """
        Download dataset from GEO using accession number and load as a modality.
        IMPORTANT: Use fetch_geo_metadata FIRST to preview dataset before downloading.
        
        Args:
            geo_id: GEO accession number (e.g., GSE12345)
            modality_type: Type of data modality ('single_cell', 'bulk', or 'auto_detect')
            
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
                modality_type=modality_type,
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
        modality_type: str = "auto_detect",
        dataset_type: str = "custom"
    ) -> str:
        """
        Upload and process a data file from the local filesystem.
        
        Args:
            file_path: Path to the data file (CSV, H5, Excel, etc.)
            dataset_id: Unique identifier for this dataset
            modality_type: Type of biological data ('single_cell', 'bulk', 'proteomics_ms', 'auto_detect')
            dataset_type: Source type (e.g., 'custom', 'local', 'processed')
            
        Returns:
            str: Summary of uploaded data
        """
        try:
            from pathlib import Path
            
            file_path = Path(file_path)
            if not file_path.exists():
                return f"File not found: {file_path}"
            
            # Auto-detect modality type if requested
            if modality_type == "auto_detect":
                # Read file to detect data characteristics
                import pandas as pd
                try:
                    df = pd.read_csv(file_path, index_col=0, nrows=10)  # Sample first 10 rows
                    n_cols = df.shape[1]
                    
                    # Heuristics for detection
                    if n_cols > 5000:
                        modality_type = "single_cell"
                    elif n_cols < 1000:
                        modality_type = "proteomics_ms"
                    else:
                        modality_type = "bulk"  # Middle ground
                    
                    logger.info(f"Auto-detected modality type: {modality_type} (based on {n_cols} features)")
                except:
                    modality_type = "single_cell"  # Safe default
            
            # Map modality types to adapter names
            adapter_mapping = {
                "single_cell": "transcriptomics_single_cell",
                "bulk": "transcriptomics_bulk", 
                "proteomics_ms": "proteomics_ms",
                "proteomics_affinity": "proteomics_affinity"
            }
            
            adapter_name = adapter_mapping.get(modality_type, "transcriptomics_single_cell")
            modality_name = f"{dataset_type}_{dataset_id}"
            
            # Load using appropriate adapter
            adata = data_manager.load_modality(
                name=modality_name,
                source=file_path,
                adapter=adapter_name,
                validate=True,
                dataset_id=dataset_id,
                dataset_type=dataset_type
            )
            
            # Save to workspace
            save_path = f"{dataset_id}_{modality_type}.h5ad"
            saved_path = data_manager.save_modality(modality_name, save_path)
            
            # Get quality metrics
            metrics = data_manager.get_quality_metrics(modality_name)
            
            return f"""Successfully uploaded and processed file {file_path.name}!

📊 Modality: '{modality_name}' ({adata.n_obs} obs × {adata.n_vars} vars)
🔬 Data type: {modality_type}
🎯 Adapter: {adapter_name}
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
        check_tmp_metadata_keys,
        fetch_geo_metadata,
        download_geo_dataset,
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
You are a data management expert specializing in multi-omics bioinformatics datasets using the modular DataManagerV2 system.

<Task>
You handle all data acquisition, storage, and retrieval operations using the new modular architecture:
0. **Fetching metadata** and give a summary to the supervisor
1. **Download and load datasets** from various sources (GEO, local files, etc.)
2. **Process and validate data** using appropriate modality adapters
3. **Store data as modalities** with proper schema enforcement
4. **Provide data access** to other agents via modality names
5. **Maintain workspace** with proper organization and provenance tracking
</Task>

<Available Tools>
- check_tmp_metadata_keys: Check for which identifiers the metadata is currently temporary stored (returns a list of identifiers) 
- fetch_geo_metadata: Fetch and validate GEO metadata without downloading full dataset (USE FIRST for GEO datasets)
- download_geo_dataset: Download data from GEO and load as modality (requires metadata fetch first)
- find_geo_from_doi: Find GEO datasets associated with a DOI
- find_geo_from_pmid: Find GEO datasets associated with a PMID
- get_data_summary: Get summary of loaded modalities or specific modality
- upload_data_file: Upload local files and create modalities with auto-detection
- load_modality_from_file: Load specific file as named modality with chosen adapter
- list_available_modalities: Show all currently loaded modalities with details
- remove_modality: Remove modality from memory

- get_adapter_info: Show available adapters and their capabilities

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

<Important Guidelines>
1. **Auto-detect data types** when possible to choose appropriate adapters (see available adapter above)
2. **Validate all data** before confirming successful load
3. **Never hallucinate GEO identifiers** - always ask user to provide them

<Example Workflows & Tool Usage Order>

## 1. DISCOVERY & EXPLORATION WORKFLOWS

### Starting with Dataset Accessions (Research Agent Discovery)
```bash
# Step 1: Receive dataset accessions from research_agent
# (Dataset discovery is now handled by research_agent)

# Step 2: Check current workspace status  
get_data_summary()  # Shows all loaded modalities

# Step 3: Fetch metadata for the dataset first
fetch_geo_metadata("GSE123456")

<important: download strategy>
You will receive meta information about the dataset AND information about available files. Its crucial that you tell the supervisor to ask the user which approach they want to take: 
'MATRIX_FIRST' : if the processed matrix is available they could continue with this but might have unwanted preprocessing decision in there
'H5_FIRST' : if a h5ad file is given maybe they want to directly work with this
'SAMPLES_FIRST' : or they want to work with the raw sample data where they can guide the pre-processing part

or they want the agent to decide (no input)
Always report back after fetching the metadata. Once confirmed by the supervisor you can continue
</important>

# Step 4 scenario a: Download the dataset with appropriate modality type, automatically
download_geo_dataset("GSE123456", modality_type="transcriptomics_single_cell")

# Step 4 scenario b: Download the dataset with appropriate modality type, with manual override by user
If the user has decided to manually choose which download strategy to choose (MATRIX_FIRST, H5_FIRST, SAMPLES_FIRST)
### example user wants to work with the raw sample files
download_geo_dataset("GSE123456", modality_type="transcriptomics_single_cell", manual_strategy_override='SAMPLES_FIRST')

<important> 
# You will see summary information about the download with the exact name of the modality ID
</important>

# Step 5: Verify and explore the loaded data
get_data_summary("geo_gse123456")  # Get detailed summary of specific modality
```

### Starting with Workspace Exploration
```bash
# Step 1: Check what's already loaded
list_available_modalities()

# Step 2: Get workspace and adapter information
get_adapter_info()

# Step 3: Examine specific modality if exists
get_data_summary("existing_modality_name")
```

## 2. DATA LOADING WORKFLOWS

### GEO Dataset Loading with Metadata-First Approach (Most Common)
```bash
# Step 1: Always check if data already exists first
get_data_summary()

# Step 2: REQUIRED - Fetch metadata first to preview dataset
fetch_geo_metadata("GSE67310")

# Step 3: Review metadata summary, then download with appropriate modality type
download_geo_dataset("GSE67310", modality_type="transcriptomics_single_cell")
# OR with auto-detection based on metadata recommendation:
download_geo_dataset("GSE67310", modality_type="auto_detect")

# Step 4: Verify successful loading
get_data_summary("geo_gse67310")

# Step 5: List all available modalities to confirm
list_available_modalities()
```

### Custom File Upload Workflow
```bash
# Step 1: Check available adapters first
get_adapter_info()

# Step 2: Upload with auto-detection (recommended)
upload_data_file("/path/to/data.csv", "liver_sc_study", modality_type="auto_detect", dataset_type="custom")

# Step 3: OR upload with specific type if known
upload_data_file("/path/to/proteomics.csv", "protein_expr", modality_type="proteomics_ms", dataset_type="processed")

# Step 4: Verify upload success
get_data_summary("custom_liver_sc_study")

# Step 5: List all to see the new modality
list_available_modalities()
```

### Advanced File Loading (Expert Use)
```bash
# Step 1: Check available adapters for compatibility
get_adapter_info()

# Step 2: Load with specific adapter and custom name
load_modality_from_file("custom_name", "/path/to/h5ad_file.h5ad", "transcriptomics_single_cell")

# Step 3: Verify loading with custom parameters
get_data_summary("custom_name")
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

### When to Use Each Tool:

**list_available_modalities()** - Use FIRST to check workspace state
**get_data_summary()** - Use to check specific modality details or overview
**get_adapter_info()** - Use when unsure about data types/formats
**download_geo_dataset()** - Primary tool for GEO data acquisition
**upload_data_file()** - Use for local files with auto-detection
**load_modality_from_file()** - Use for expert/manual loading with specific adapters
**create_mudata_from_modalities()** - Use for multi-modal integration
**remove_modality()** - Use for workspace cleanup

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
