"""
GEO data parser module.

This module handles parsing GEO files (SOFT, matrix, supplementary) and
extracting structured data for analysis.
"""

import pandas as pd
import gzip
import io
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class GEOParserError(Exception):
    """Custom exception for GEO parser errors."""
    pass


class GEOFormatError(Exception):
    """Custom exception for unsupported GEO file formats."""
    pass

class GEOParser:
    """
    Parser for GEO database files.
    
    This class provides methods to parse and extract meaningful data
    from GEO SOFT files, series matrix files, and supplementary files.
    """
    
    @staticmethod
    def parse_soft_file(file_path: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Parse a GEO SOFT format file.
        
        Args:
            file_path: Path to the SOFT file
            
        Returns:
            tuple: DataFrame of expression data and metadata dictionary
        """
        try:
            if file_path.suffix == '.gz':
                opener = gzip.open
            else:
                opener = open
            
            with opener(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                return GEOParser._parse_soft_content(f)
                
        except Exception as e:
            logger.error(f"Error parsing SOFT file {file_path}: {e}")
            return None, {}
    
    @staticmethod
    def _parse_soft_content(file_handle) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Parse SOFT file content from an open file handle.
        
        Args:
            file_handle: Open file handle to SOFT file
            
        Returns:
            tuple: DataFrame of expression data and metadata dictionary
        """
        metadata = {}
        samples = {}
        platforms = {}
        series_info = {}
        current_section = None
        current_sample = None
        current_platform = None
        
        # Read file line by line
        data_section = False
        data_lines = []
        
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            
            # Series metadata
            if line.startswith('^SERIES'):
                current_section = 'series'
                series_id = line.split('=')[1].strip() if '=' in line else ''
                series_info['id'] = series_id
            elif line.startswith('!Series_'):
                if current_section == 'series':
                    key = line.split('=')[0][1:].strip()  # Remove !
                    value = line.split('=')[1].strip() if '=' in line else ''
                    series_info[key] = value
            
            # Platform metadata
            elif line.startswith('^PLATFORM'):
                current_section = 'platform'
                current_platform = line.split('=')[1].strip() if '=' in line else ''
                platforms[current_platform] = {}
            elif line.startswith('!platform_') and current_section == 'platform':
                key = line.split('=')[0][1:].strip()
                value = line.split('=')[1].strip() if '=' in line else ''
                platforms[current_platform][key] = value
            
            # Sample metadata
            elif line.startswith('^SAMPLE'):
                current_section = 'sample'
                current_sample = line.split('=')[1].strip() if '=' in line else ''
                samples[current_sample] = {}
                data_section = False
            elif line.startswith('!Sample_') and current_section == 'sample':
                key = line.split('=')[0][1:].strip()
                value = line.split('=')[1].strip() if '=' in line else ''
                samples[current_sample][key] = value
            elif line.startswith('!sample_table_begin') and current_section == 'sample':
                data_section = True
                continue
            elif line.startswith('!sample_table_end'):
                data_section = False
                continue
            elif data_section and current_section == 'sample':
                data_lines.append(line)
        
        # Combine metadata
        metadata = {
            'series': series_info,
            'platforms': platforms,
            'samples': samples,
            'n_samples': len(samples)
        }
        
        # Try to construct expression matrix
        expression_data = GEOParser._construct_expression_matrix(samples, data_lines)
        
        return expression_data, metadata
    
    @staticmethod
    def parse_supplementary_file(file_path: Path) -> Optional[pd.DataFrame]:
        """
        Parse supplementary expression data file.
        
        Args:
            file_path: Path to the supplementary file
            
        Returns:
            DataFrame: Expression data, or None if parsing failed
        """
        try:
            # Handle different file types based on extension
            file_suffix = file_path.suffix.lower()
            
            # Handle compressed files
            if file_suffix == '.gz':
                opener = gzip.open
                # Get the actual file type from the inner extension
                inner_suffix = Path(str(file_path)[:-3]).suffix.lower()
            else:
                opener = open
                inner_suffix = file_suffix
            
            # Handle specific file formats
            if inner_suffix in ['.h5', '.h5ad']:
                try:
                    import anndata
                    adata = anndata.read_h5ad(file_path)
                    logger.info(f"Parsed h5ad file: {file_path}")
                    # Convert AnnData to DataFrame (X contains expression matrix)
                    return pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
                except ImportError:
                    logger.warning("anndata package not available, cannot parse h5ad")
                    return None
                except Exception as e:
                    logger.warning(f"Failed to parse h5ad file: {e}")
                    return None
            
            elif inner_suffix == '.rds':
                try:
                    import pyreadr
                    result = pyreadr.read_r(str(file_path))
                    logger.info(f"Parsed RDS file: {file_path}")
                    # Get the first dataframe in the result
                    if result and result[list(result.keys())[0]] is not None:
                        return result[list(result.keys())[0]]
                except ImportError:
                    logger.warning("pyreadr package not available, cannot parse RDS")
                    return None
                except Exception as e:
                    logger.warning(f"Failed to parse RDS file: {e}")
                    return None
                    
            elif inner_suffix in ['.txt', '.tsv', '.csv']:
                # Try to determine format and parse accordingly
                with opener(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    # Read first few lines to determine format
                    first_lines = [f.readline().strip() for _ in range(5)]
                    f.seek(0)
                    
                    # Check if it's a standard matrix format
                    if any('\t' in line for line in first_lines):
                        # Tab-delimited format
                        try:
                            df = pd.read_csv(f, sep='\t', index_col=0, low_memory=False)
                            logger.info(f"Parsed file as tab-delimited: {df.shape}")
                            return df
                        except Exception as e:
                            logger.warning(f"Failed to parse as tab-delimited: {e}")
                    
                    # Try comma-separated
                    f.seek(0)
                    try:
                        df = pd.read_csv(f, index_col=0, low_memory=False)
                        logger.info(f"Parsed file as CSV: {df.shape}")
                        return df
                    except Exception as e:
                        logger.warning(f"Failed to parse as CSV: {e}")
            
            # Handle MTX format (common in single-cell data)
            elif inner_suffix == '.mtx' or file_path.name.lower().endswith('.mtx.gz'):
                try:
                    import scipy.io as sio
                    # Load the mtx file
                    mtx = sio.mmread(file_path)
                    logger.info(f"Parsed MTX file: {file_path}")
                    
                    # Try to find barcodes and features files in the same directory
                    parent_dir = file_path.parent
                    barcodes_file = None
                    features_file = None
                    
                    for potential_file in parent_dir.glob('*'):
                        name_lower = potential_file.name.lower()
                        if 'barcode' in name_lower or 'cell' in name_lower:
                            barcodes_file = potential_file
                        elif 'feature' in name_lower or 'gene' in name_lower:
                            features_file = potential_file
                    
                    # Read barcodes and features if found
                    cell_ids = []
                    gene_ids = []
                    
                    if barcodes_file:
                        with open(barcodes_file, 'r') as f:
                            cell_ids = [line.strip() for line in f]
                    
                    if features_file:
                        with open(features_file, 'r') as f:
                            # Features might have multiple columns
                            gene_ids = [line.strip().split('\t')[0] for line in f]
                    
                    # Convert to dense array if it's sparse
                    if hasattr(mtx, 'todense'):
                        mtx_dense = mtx.todense()
                    else:
                        mtx_dense = mtx
                    
                    # Create DataFrame
                    if not cell_ids:
                        cell_ids = [f"Cell_{i}" for i in range(mtx_dense.shape[0])]
                    if not gene_ids:
                        gene_ids = [f"Gene_{i}" for i in range(mtx_dense.shape[1])]
                    
                    return pd.DataFrame(mtx_dense, index=cell_ids, columns=gene_ids)
                except ImportError:
                    logger.warning("scipy.io package not available, cannot parse MTX")
                    return None
                except Exception as e:
                    logger.warning(f"Failed to parse MTX file: {e}")
                    return None
            
            logger.warning(f"Unsupported file format: {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return None

    @staticmethod
    def _construct_expression_matrix(samples: Dict, data_lines: list) -> Optional[pd.DataFrame]:
        """
        Construct expression matrix from SOFT data.
        
        Args:
            samples: Dictionary of sample metadata
            data_lines: List of data lines from SOFT file
            
        Returns:
            DataFrame: Expression matrix, or None if construction failed
        """
        try:
            if not data_lines:
                logger.warning("No expression data found in SOFT file")
                return None
            
            # In a production environment, we don't generate synthetic data
            # We would need to extract actual data from the data_lines
            # This is complex and depends on the specific structure of the SOFT file
            # For now, we'll return None if we can't parse real data
            
            # TODO: Implement actual parsing of expression data from SOFT file data lines
            # This would involve parsing the data_lines to extract gene expression values
            
            logger.warning("Expression matrix construction from SOFT file not implemented")
            return None
            
        except Exception as e:
            logger.error(f"Error constructing expression matrix: {e}")
            return None
    
    @staticmethod
    def analyze_data_sources(soft_file_path: Path, data_sources: Dict[str, Path]) -> Dict[str, Any]:
        """
        Analyze available data sources and SOFT file to determine the best source.
        
        Args:
            soft_file_path: Path to the SOFT file
            data_sources: Dictionary of data sources (supplementary, tar)
            
        Returns:
            dict: Analysis results including recommended data source
        """
        analysis = {
            'available_sources': list(data_sources.keys()),
            'recommended_source': None,
            'reason': None
        }
        
        try:
            # Parse the SOFT file to get metadata
            _, metadata = GEOParser.parse_soft_file(soft_file_path)
            
            # Extract relevant information from metadata
            series_info = metadata.get('series', {})
            platform_info = metadata.get('platforms', {})
            sample_info = metadata.get('samples', {})
            
            # Look for clues in the metadata about data format
            data_type = series_info.get('Series_type', '').lower()
            raw_data_mentioned = False
            processed_data_mentioned = False
            
            # Check series title and summary for clues
            title = series_info.get('Series_title', '').lower()
            summary = series_info.get('Series_summary', '').lower()
            
            # Keywords that suggest raw data
            raw_keywords = ['raw', 'counts', 'fastq', 'seq', 'single-cell', 'single cell', 
                            'scrna-seq', 'scrnaseq', '10x', 'rna-seq', 'rnaseq']
                            
            # Check if raw data is mentioned
            for keyword in raw_keywords:
                if keyword in title or keyword in summary:
                    raw_data_mentioned = True
                    break
            
            # Check if processed data is mentioned
            processed_keywords = ['normalized', 'processed', 'expression matrix']
            for keyword in processed_keywords:
                if keyword in title or keyword in summary:
                    processed_data_mentioned = True
                    break
            
            # Decision logic
            if 'tar' in data_sources and 'supplementary' in data_sources:
                if raw_data_mentioned and not processed_data_mentioned:
                    analysis['recommended_source'] = 'tar'
                    analysis['reason'] = "SOFT file metadata indicates raw data, using TAR file"
                elif processed_data_mentioned and not raw_data_mentioned:
                    analysis['recommended_source'] = 'supplementary'
                    analysis['reason'] = "SOFT file metadata indicates processed data, using supplementary file"
                else:
                    # Check file sizes as a heuristic - larger files are often raw data
                    try:
                        tar_size = data_sources['tar'].stat().st_size
                        suppl_size = data_sources['supplementary'].stat().st_size
                        
                        if tar_size > suppl_size * 2:  # If TAR file is much larger
                            analysis['recommended_source'] = 'tar'
                            analysis['reason'] = "TAR file is significantly larger, likely contains more complete data"
                        else:
                            analysis['recommended_source'] = 'supplementary'
                            analysis['reason'] = "Supplementary file appears adequate based on size comparison"
                    except:
                        # Default to TAR if we can't compare sizes
                        analysis['recommended_source'] = 'tar'
                        analysis['reason'] = "Unable to compare file sizes, defaulting to TAR file"
            elif 'tar' in data_sources:
                analysis['recommended_source'] = 'tar'
                analysis['reason'] = "Only TAR file is available"
            elif 'supplementary' in data_sources:
                analysis['recommended_source'] = 'supplementary'
                analysis['reason'] = "Only supplementary file is available"
            elif 'tar_dir' in data_sources:
                analysis['recommended_source'] = 'tar_dir'
                analysis['reason'] = "Only TAR directory is available, but no suitable expression file was found"
            else:
                analysis['recommended_source'] = None
                analysis['reason'] = "No suitable data sources found"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing data sources: {e}")
            analysis['recommended_source'] = list(data_sources.keys())[0] if data_sources else None
            analysis['reason'] = f"Error during analysis, using first available source: {e}"
            return analysis
    
    @staticmethod
    def parse_matrix_file(file_path: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Parse GEO series matrix file.
        
        Args:
            file_path: Path to the matrix file
            
        Returns:
            tuple: DataFrame of expression data and metadata dictionary
        """
        try:
            if file_path.suffix == '.gz':
                opener = gzip.open
            else:
                opener = open
            
            metadata = {}
            data_start_line = 0
            
            with opener(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Find metadata and data start
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('!'):
                    # Metadata line
                    if '=' in line:
                        key = line[1:].split('=')[0].strip()
                        value = line.split('=')[1].strip().strip('"')
                        metadata[key] = value
                elif line.startswith('"ID_REF"') or line.startswith('ID_REF'):
                    data_start_line = i
                    break
            
            # Read data
            if data_start_line > 0:
                data_df = pd.read_csv(
                    io.StringIO(''.join(lines[data_start_line:])),
                    sep='\t',
                    index_col=0,
                    low_memory=False
                )
                
                # Transpose so cells are rows, genes are columns
                data_df = data_df.T
                
                return data_df, metadata
            
            return None, metadata
            
        except Exception as e:
            logger.error(f"Error parsing matrix file {file_path}: {e}")
            return None, {}
