"""
Data Expert Agent for handling all data fetching, downloading, and extraction.

This agent is responsible for managing all data acquisition operations,
ensuring proper storage and indexing through the DataManager.
"""

from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

from datetime import date

from lobster.agents.state import DataExpertState
from lobster.config.settings import get_settings
from lobster.core.data_manager import DataManager
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def data_expert(
    data_manager: DataManager, 
    callback_handler=None, 
    agent_name: str = "data_expert_agent",
    handoff_tools: List = None
):  
    """Create data expert agent for handling all data operations."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('data_expert')
    llm = ChatBedrockConverse(**model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Define tools for data operations
    @tool
    def download_geo_dataset(geo_id: str) -> str:
        """
        Download dataset from GEO using accession number.
        First checks if the dataset already exists in workspace before downloading.
        
        Args:
            geo_id: GEO accession number (e.g., GSE12345)
            
        Returns:
            str: Summary of downloaded data with identifier information
        """
        try:
            from lobster.tools import GEOService
            from lobster.utils.file_naming import BioinformaticsFileNaming
            import pandas as pd
            import json
            
            # Clean the GEO ID
            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith('GSE'):
                return f"Invalid GEO ID format: {geo_id}. Expected format: GSE12345"
            
            logger.info(f"Processing GEO dataset request: {clean_geo_id}")
            
            # Check if dataset already exists in workspace using professional naming
            data_dir = data_manager.data_dir
            existing_file = BioinformaticsFileNaming.find_latest_file(
                directory=data_dir,
                data_source='GEO',
                dataset_id=clean_geo_id,
                processing_step='raw_matrix'
            )
            
            if existing_file and existing_file.exists():
                logger.info(f"Found existing processed file: {existing_file.name}")
                
                try:
                    # Load the existing data
                    logger.info(f"Loading existing raw matrix: {existing_file}")
                    existing_data = pd.read_csv(existing_file, index_col=0)
                    
                    # Look for corresponding metadata file
                    metadata_file = existing_file.parent / BioinformaticsFileNaming.generate_metadata_filename(existing_file.name)
                    existing_metadata = {}
                    
                    if metadata_file.exists():
                        logger.info(f"Loading existing metadata: {metadata_file}")
                        with open(metadata_file, 'r') as f:
                            existing_metadata = json.load(f)
                    
                    # Set the data in data manager
                    data_manager.set_data(data=existing_data, metadata=existing_metadata)
                    
                    # Ensure proper metadata structure
                    data_manager.current_metadata['dataset_id'] = clean_geo_id
                    data_manager.current_metadata['dataset_type'] = 'GEO'
                    data_manager.current_metadata['source'] = f"GEO:{clean_geo_id}"
                    data_manager.current_metadata['loaded_from_existing'] = True
                    data_manager.current_metadata['existing_file'] = str(existing_file)
                    
                    # Log tool usage for loading existing data
                    data_manager.log_tool_usage(
                        tool_name="load_existing_geo_dataset",
                        parameters={"geo_id": clean_geo_id, "file_path": str(existing_file)},
                        description=f"Loaded existing processed GEO dataset {clean_geo_id} from workspace"
                    )
                    
                    processing_date = existing_metadata.get('processing_date', 'Unknown')
                    if isinstance(processing_date, str) and 'T' in processing_date:
                        # Format ISO timestamp to readable format
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(processing_date.replace('Z', '+00:00'))
                            processing_date = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            pass
                    
                    return f"""Found and loaded existing dataset {clean_geo_id}!

📁 Loaded from: {existing_file.name}
📊 Matrix: {existing_data.shape[0]} cells × {existing_data.shape[1]} genes
💾 No download needed - using cached data
🕒 Originally processed: {processing_date}
🔬 Samples: {existing_metadata.get('n_samples', 'Unknown')}
⚡ Ready for immediate analysis!

Use this dataset for quality control, filtering, or downstream analysis."""
                    
                except Exception as load_error:
                    logger.warning(f"Failed to load existing file {existing_file}: {load_error}")
                    logger.info("Proceeding with fresh download due to loading error")
            
            # No existing file found or loading failed - proceed with download
            logger.info(f"No existing file found for {clean_geo_id}, proceeding with download")
            
            # Get console from data_manager for showing download progress
            console = getattr(data_manager, 'console', None)
            
            # Pass console to GEOService for download progress tracking
            geo_service = GEOService(data_manager, console=console)
            result = geo_service.download_dataset(clean_geo_id)
            
            # Store identifier in metadata for reference
            if data_manager.has_data():
                data_manager.current_metadata['dataset_id'] = clean_geo_id
                data_manager.current_metadata['dataset_type'] = 'GEO'
                data_manager.current_metadata['source'] = f"GEO:{clean_geo_id}"
                data_manager.current_metadata['loaded_from_existing'] = False
                
                # The GEO service now uses professional naming via DataManager.save_processed_data
                # No need to manually save here as it's handled by the service
                
                # Log tool usage
                data_manager.log_tool_usage(
                    tool_name="download_geo_dataset",
                    parameters={"geo_id": clean_geo_id},
                    description=f"Downloaded GEO dataset {clean_geo_id}"
                )
            
            logger.info(f"Downloaded GEO dataset: {clean_geo_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing GEO dataset {geo_id}: {e}")
            return f"Error processing dataset: {str(e)}"

    @tool
    def find_geo_from_doi(doi: str) -> str:
        """
        Find GEO accession numbers associated with a DOI.
        
        Args:  
            doi: Digital Object Identifier (e.g., 10.1234/example)
            
        Returns:
            str: List of GEO accessions found for the DOI
        """
        try:
            if not doi.startswith("10."):
                return "Invalid DOI format. DOI should start with '10.'"
            
            from lobster.tools import PubMedService
            pubmed_service = PubMedService(parse=None, data_manager=data_manager)
            results = pubmed_service.find_geo_from_doi(doi)
            
            # Store DOI information in metadata
            data_manager.current_metadata['associated_doi'] = doi
            
            logger.info(f"GEO search completed for DOI: {doi}")
            return results
        except Exception as e:
            logger.error(f"Error finding GEO from DOI: {e}")
            return f"Error finding GEO datasets: {str(e)}"

    @tool
    def get_data_summary(identifier: str = "") -> str:
        """
        Get summary of currently loaded data or list available datasets.
        
        Args:
            identifier: Optional dataset identifier (GEO ID, DOI, or 'all' to list all)
            
        Returns:
            str: Data summary or list of available datasets
        """
        try:
            if identifier.lower() == 'all' or identifier == "":
                # List all available datasets in workspace
                files = data_manager.list_workspace_files()
                data_files = files.get('data', [])
                
                if not data_files:
                    return "No datasets available. Use download_geo_dataset to download data."
                
                response = "Available datasets:\n"
                for file in data_files:
                    if file['name'].endswith('.csv'):
                        # Extract identifier from filename if possible
                        if file['name'].startswith('GEO_'):
                            dataset_id = file['name'].replace('GEO_', '').replace('_data.csv', '')
                            response += f"- GEO:{dataset_id} ({file['name']})\n"
                        else:
                            response += f"- {file['name']}\n"
                
                return response
            
            # Check current data
            if not data_manager.has_data():
                return "No data currently loaded. Use download_geo_dataset to load data."
            
            # Get summary of current data
            summary = data_manager.get_data_summary()
            current_id = data_manager.current_metadata.get('dataset_id', 'Unknown')
            dataset_type = data_manager.current_metadata.get('dataset_type', 'Unknown')
            
            response = f"Current dataset: {dataset_type}:{current_id}\n"
            response += f"Shape: {summary['shape'][0]} cells × {summary['shape'][1]} genes\n"
            response += f"Memory usage: {summary['memory_usage']}"
            
            if 'metadata_keys' in summary and summary['metadata_keys']:
                response += f"\nMetadata available: {', '.join(summary['metadata_keys'][:5])}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return f"Error getting data summary: {str(e)}"

    @tool
    def upload_data_file(file_path: str, dataset_id: str, dataset_type: str = "custom") -> str:
        """
        Upload and process a data file from the local filesystem.
        
        Args:
            file_path: Path to the data file (CSV, H5, or Excel)
            dataset_id: Unique identifier for this dataset
            dataset_type: Type of dataset (e.g., 'custom', 'local', 'processed')
            
        Returns:
            str: Summary of uploaded data
        """
        try:
            from lobster.tools import FileUploadService
            upload_service = FileUploadService(data_manager)
            result = upload_service.upload_file(file_path)
            
            if "successfully" in result:
                # Add identifier metadata
                data_manager.current_metadata['dataset_id'] = dataset_id
                data_manager.current_metadata['dataset_type'] = dataset_type
                data_manager.current_metadata['source'] = f"{dataset_type}:{dataset_id}"
                data_manager.current_metadata['original_file'] = file_path
                
                # Save to workspace with identifier
                filename = f"{dataset_type}_{dataset_id}_data.csv"
                data_manager.save_data_to_workspace(filename)
                
                # Log tool usage
                data_manager.log_tool_usage(
                    tool_name="upload_data_file",
                    parameters={
                        "file_path": file_path,
                        "dataset_id": dataset_id,
                        "dataset_type": dataset_type
                    },
                    description=f"Uploaded {dataset_type} dataset: {dataset_id}"
                )
            
            logger.info(f"Uploaded data file: {file_path} as {dataset_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {e}")
            return f"Error uploading file: {str(e)}"

    @tool
    def load_dataset_by_id(dataset_id: str) -> str:
        """
        Load a previously downloaded dataset by its identifier.
        
        Args:
            dataset_id: Dataset identifier (e.g., GSE12345)
            
        Returns:
            str: Status of loading operation
        """
        try:
            # Look for the dataset in workspace
            files = data_manager.list_workspace_files()
            data_files = files.get('data', [])
            logger.info(f"Looking for datasets by ID - files: {data_files}")
            
            # Search for matching file
            found_file = None
            for file in data_files:
                if dataset_id in file['name']:
                    found_file = file
                    break
            
            if not found_file:
                return f"Dataset {dataset_id} not found in workspace. Available datasets: {[f['name'] for f in data_files]}"
            
            # Load the data
            import pandas as pd
            data = pd.read_csv(found_file['path'], index_col=0)
            
            # Load metadata if available
            metadata_path = found_file['path'].replace('.csv', '_metadata.json')
            metadata = {}
            try:
                import json
                from pathlib import Path
                if Path(metadata_path).exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
            except:
                pass
            
            # Set the data
            data_manager.set_data(data, metadata)
            
            # Ensure identifier is in metadata
            if 'dataset_id' not in data_manager.current_metadata:
                data_manager.current_metadata['dataset_id'] = dataset_id
            
            logger.info(f"Loaded dataset: {dataset_id}")
            return f"Successfully loaded dataset {dataset_id}: {data.shape[0]} cells × {data.shape[1]} genes"
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {e}")
            return f"Error loading dataset: {str(e)}"

    @tool  
    def list_available_datasets() -> str:
        """
        List all available datasets in the workspace with their identifiers.
        
        Returns:
            str: Formatted list of available datasets
        """
        try:
            files = data_manager.list_workspace_files()
            data_files = files.get('data', [])
            
            if not data_files:
                return "No datasets available in workspace."
            
            response = "Available datasets in workspace:\n\n"
            
            # Group by dataset type
            geo_datasets = []
            custom_datasets = []
            other_datasets = []
            
            for file in data_files:
                if file['name'].endswith('.csv'):
                    if file['name'].startswith('GEO_'):
                        geo_id = file['name'].replace('GEO_', '').replace('_data.csv', '')
                        geo_datasets.append(f"- GEO:{geo_id} (Size: {file['size']/1024:.1f} KB)")
                    elif '_' in file['name'] and file['name'].endswith('_data.csv'):
                        parts = file['name'].replace('_data.csv', '').split('_', 1)
                        if len(parts) == 2:
                            dtype, did = parts
                            custom_datasets.append(f"- {dtype}:{did} (Size: {file['size']/1024:.1f} KB)")
                        else:
                            other_datasets.append(f"- {file['name']} (Size: {file['size']/1024:.1f} KB)")
                    else:
                        other_datasets.append(f"- {file['name']} (Size: {file['size']/1024:.1f} KB)")
            
            if geo_datasets:
                response += "GEO Datasets:\n" + "\n".join(geo_datasets) + "\n\n"
            if custom_datasets:
                response += "Custom Datasets:\n" + "\n".join(custom_datasets) + "\n\n"
            if other_datasets:
                response += "Other Files:\n" + "\n".join(other_datasets) + "\n"
            
            response += f"\nTotal: {len(data_files)} data files"
            
            return response
            
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return f"Error listing datasets: {str(e)}"

    base_tools = [
        download_geo_dataset,
        find_geo_from_doi,
        get_data_summary,
        upload_data_file,
        load_dataset_by_id,
        list_available_datasets
    ]
    
    # Combine base tools with handoff tools if provided
    tools = base_tools + (handoff_tools or [])
    
    system_prompt = """
You are a data management expert specializing in bioinformatics datasets.

<Task>
You handle all data acquisition, storage, and retrieval operations:
1. Download datasets from various sources (GEO, etc.)
2. Process and validate data files
3. Store data with proper identifiers for reference
4. Provide data access to other agents
5. Maintain a catalog of available datasets
</Task>

<Available Tools>
- download_geo_dataset: Download data from GEO
- find_geo_from_doi: Find GEO datasets associated with a DOI
- get_data_summary: Get summary of current or available data
- upload_data_file: Upload local data files
- load_dataset_by_id: Load a specific dataset by identifier
- list_available_datasets: Show all available datasets

<Important Guidelines>
1. **Always store datasets with clear identifiers** (GEO ID, DOI, custom ID)
2. **Auto-save data to workspace** after downloading/uploading
3. **Track dataset metadata** including source, date, and processing info
4. **Provide clear dataset references** so other agents can request specific data
5. **Validate data** before confirming successful load

<Dataset Identifier Format>
- GEO datasets: "GEO:GSE12345"
- DOI-linked data: "DOI:10.1234/example"
- Custom uploads: "custom:dataset_name"

When other agents need data, they should specify the dataset identifier,
and you will load the appropriate dataset for them.

Today's date is {date}.
""".format(date=date.today())

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=DataExpertState
    )
