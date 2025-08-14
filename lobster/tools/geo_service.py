"""
GEO data service.

This service provides a unified interface for downloading and parsing
data from the Gene Expression Omnibus (GEO) database.
"""

import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import pandas as pd

from .geo_downloader import GEODownloadManager
from .geo_parser import GEOParser
from ..core.data_manager import DataManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class GEOService:
    """
    Service for accessing and processing GEO data.
    
    This class provides a high-level interface for working with GEO data,
    handling the downloading, parsing, and processing of datasets.
    """
    
    def __init__(self, data_manager: DataManager, cache_dir: Optional[str] = None, console=None):
        """
        Initialize the GEO service.
        
        Args:
            data_manager: DataManager instance for storing processed data
            cache_dir: Directory to cache downloaded files
            console: Rich console instance for display (creates new if None)
        """
        logger.info("Initializing GEOService")
        logger.debug(f"Cache directory: {cache_dir}")
        
        self.data_manager = data_manager
        # Pass the console to the download manager
        self.download_manager = GEODownloadManager(cache_dir, console=console)
        self.parser = GEOParser()
        
        logger.info("GEO download manager and parser initialized")
        logger.info("GEOService initialized successfully")
    
    def download_dataset(self, query: str) -> str:
        """
        Download and process a dataset from GEO.
        
        Args:
            query: Query string containing a GEO accession number
            
        Returns:
            str: Status message
        """
        try:
            logger.info(f"Processing GEO query: {query}")
            
            gse_id = self._extract_gse_id(query)
            if not gse_id:
                return "Please provide a valid GSE accession number (e.g., GSE109564)"
            
            logger.info(f"Identified GSE ID: {gse_id}")
            
            # Download SOFT and data files (supplementary or TAR)
            soft_file, data_files = self.download_manager.download_geo_data(gse_id)
            
            if not soft_file:
                return f"Could not download SOFT file for {gse_id}"
            
            # Check if we got multiple data sources as a dictionary
            data_source_info = None
            if isinstance(data_files, dict):
                # Analyze data sources to determine which one to use
                logger.info(f"Multiple data sources found: {list(data_files.keys())}")
                data_source_analysis = self.parser.analyze_data_sources(soft_file, data_files)
                recommended_source = data_source_analysis['recommended_source']
                
                if recommended_source and recommended_source in data_files:
                    data_source_info = {
                        'type': recommended_source,
                        'reason': data_source_analysis['reason']
                    }
                    data_file = data_files[recommended_source]
                    logger.info(f"Using recommended data source: {recommended_source} ({data_source_analysis['reason']})")
                else:
                    # Fall back to the first available source
                    source_key = next(iter(data_files.keys()))
                    data_file = data_files[source_key]
                    data_source_info = {
                        'type': source_key,
                        'reason': "Default selection (first available source)"
                    }
                    logger.info(f"Using default data source: {source_key}")
            else:
                # Single data file
                data_file = data_files
                data_source_info = {
                    'type': 'supplementary' if data_file else None,
                    'reason': "Only one data source available"
                }
            
            # Parse all available data sources to extract expression matrices
            parsed_data = {}
            parsed_metadata = {}
            
            # Dictionary to store all data files info
            all_data_files = {}
            
            # Get the base metadata from SOFT file first
            _, base_metadata = self.parser.parse_soft_file(soft_file)
            
            # Process all data sources if we received multiple
            if isinstance(data_files, dict):
                for source_type, source_path in data_files.items():
                    logger.info(f"Processing data source: {source_type} from {source_path}")
                    source_info = {
                        'type': source_type,
                        'reason': data_source_info['reason'] if data_source_info and source_type == data_source_info.get('type') else None
                    }
                    
                    # Parse this specific data source
                    expr_data, src_metadata = self._parse_geo_file(soft_file, source_path, source_info)
                    parsed_metadata[source_type] = src_metadata
                    
                    if expr_data is not None:
                        parsed_data[source_type] = expr_data
                        all_data_files[source_type] = str(source_path)
                        logger.info(f"Successfully parsed {source_type} data: {expr_data.shape}")
            else:
                # Single data file
                source_type = data_source_info['type'] if data_source_info else 'supplementary'
                expr_data, src_metadata = self._parse_geo_file(soft_file, data_file, data_source_info)
                parsed_metadata[source_type] = src_metadata
                
                if expr_data is not None:
                    parsed_data[source_type] = expr_data
                    all_data_files[source_type] = str(data_file) if data_file else None
            
            # Check if we have any valid expression data
            if not parsed_data:
                return f"Could not parse any expression data from {gse_id}"
            
            # Select primary data source based on recommendation or highest gene count
            primary_data_key = None
            
            if data_source_info and data_source_info.get('type') in parsed_data:
                # Use the recommended data source as primary
                primary_data_key = data_source_info.get('type')
            else:
                # Choose data source with the most genes as default
                gene_counts = {k: v.shape[1] for k, v in parsed_data.items()}
                if gene_counts:
                    primary_data_key = max(gene_counts.items(), key=lambda x: x[1])[0]
            
            # Set primary expression data
            primary_data = parsed_data[primary_data_key]
            
            # Enhanced metadata including all data sources
            enhanced_metadata = {
                'source': gse_id,
                'soft_file': str(soft_file),
                'primary_data_source': primary_data_key,
                'all_data_sources': list(parsed_data.keys()),
                'all_data_files': all_data_files,
                'data_source_info': data_source_info,
                'n_cells': primary_data.shape[0],
                'n_genes': primary_data.shape[1],
                'available_matrices': {k: {'shape': v.shape} for k, v in parsed_data.items()},
                **base_metadata
            }
            
            # Store primary data in data manager
            self.data_manager.set_data(data=primary_data, metadata=enhanced_metadata)
            
            # Store additional data sources as separate entries in the metadata
            # This allows the agent to access these alternative expression matrices if needed
            for source_type, expr_data in parsed_data.items():
                if source_type != primary_data_key:
                    # Store as a named matrix in the metadata
                    matrix_key = f"{source_type}_matrix"
                    enhanced_metadata[matrix_key] = expr_data
            
            # Prepare response message with information about all available data sources
            data_source_str = self._get_data_source_description({'type': primary_data_key})
            return self._format_download_response(gse_id, primary_data, base_metadata, data_source_str, parsed_data, all_data_files)
                
        except Exception as e:
            logger.exception(f"Error downloading dataset: {e}")
            return f"Error downloading dataset: {str(e)}"
    
    def _extract_gse_id(self, query: str) -> Optional[str]:
        """
        Extract GSE ID from query string.
        
        Args:
            query: Query string
            
        Returns:
            str: GSE ID, or None if not found
        """
        match = re.search(r'GSE\d+', query.upper())
        return match.group(0) if match else None
    
    def _get_data_source_description(self, data_source_info: Optional[Dict[str, Any]]) -> str:
        """
        Get a user-friendly description of the data source.
        
        Args:
            data_source_info: Data source information
            
        Returns:
            str: Description of the data source
        """
        if not data_source_info or not data_source_info.get('type'):
            return "unknown source"
            
        source_type = data_source_info['type']
        if source_type == 'supplementary':
            return "supplementary file"
        elif source_type == 'tar':
            return "TAR archive file"
        elif source_type == 'tar_dir':
            return "extracted TAR archive"
        else:
            return str(source_type)

    def _parse_geo_file(self, soft_path: Path, data_path: Optional[Path] = None, 
                      data_source_info: Optional[Dict[str, Any]] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Parse GEO files.
        
        Args:
            soft_path: Path to SOFT file
            data_path: Optional path to data file (supplementary or from TAR)
            data_source_info: Information about the data source
            
        Returns:
            tuple: DataFrame of expression data and metadata dictionary
        """
        # Parse SOFT file for metadata
        _, metadata = self.parser.parse_soft_file(soft_path)
        
        expression_data = None
        source_type = data_source_info.get('type') if data_source_info else None
        
        # Try to parse data file
        if data_path and data_path.exists():
            logger.info(f"Parsing data from {source_type} source: {data_path}")
            expression_data = self.parser.parse_supplementary_file(data_path)
            
            if expression_data is not None:
                logger.info(f"Successfully parsed data: {expression_data.shape}")
                # Ensure cells are rows, genes are columns
                if expression_data.shape[0] < expression_data.shape[1]:
                    expression_data = expression_data.T
                    logger.info("Transposed data so cells are rows")
        
        # If no data found, return None
        if expression_data is None:
            logger.warning("No expression data could be parsed from available files")
            # Don't generate synthetic data - this is a production app
        
        return expression_data, metadata
    
    def _format_download_response(self, gse_id: str, primary_data: pd.DataFrame, 
                                 metadata: Dict[str, Any], primary_source_str: str,
                                 all_parsed_data: Dict[str, pd.DataFrame],
                                 all_data_files: Dict[str, str]) -> str:
        """
        Format download response message with information about all available data sources.
        
        Args:
            gse_id: GEO series ID
            primary_data: Primary expression data matrix
            metadata: Metadata dictionary
            primary_source_str: Description of the primary data source
            all_parsed_data: Dictionary of all parsed data matrices
            all_data_files: Dictionary of all data file paths
            
        Returns:
            str: Formatted response message
        """
        study_title = metadata.get('series', {}).get('Series_title', 'N/A')
        protocol = 'N/A'
        
        # Try to extract protocol information
        if metadata.get('samples'):
            sample_keys = list(metadata.get('samples', {}).keys())
            if sample_keys:
                first_sample = metadata['samples'][sample_keys[0]]
                protocol = first_sample.get('Sample_characteristics_ch1', 'N/A')
        
        # Build information about all available data sources
        additional_sources = []
        for source_type, data_matrix in all_parsed_data.items():
            if data_matrix is not primary_data:  # Skip the primary data source
                source_desc = self._get_data_source_description({'type': source_type})
                additional_sources.append(f"• {source_desc}: {data_matrix.shape[0]} cells × {data_matrix.shape[1]} genes")
        
        additional_sources_text = ""
        if additional_sources:
            additional_sources_text = "\n\n📚 Additional data sources available:\n" + "\n".join(additional_sources)
            additional_sources_text += "\n\nThese alternative matrices are stored and can be accessed for analysis if needed."
        
        return f"""Successfully downloaded {gse_id}!

📊 Primary data matrix ({primary_source_str}): {primary_data.shape[0]} cells × {primary_data.shape[1]} genes
📋 Study: {study_title}
🔬 Protocol: {protocol}{additional_sources_text}"""
