"""
Advanced GEO data parser module with optimized performance.

This module handles parsing GEO files (SOFT, matrix, supplementary) with
modern optimization techniques including Polars integration, intelligent
delimiter detection, and memory-efficient chunked processing.

Features:
- Polars integration for 2-5x faster parsing
- Intelligent delimiter detection using csv.Sniffer
- Memory-efficient chunked reading for large files
- Optimized Matrix Market parsing with sparse DataFrame support
- Multi-format support (CSV, TSV, H5, H5AD, MTX, 10X Genomics)
"""

import pandas as pd
import numpy as np
import gzip
import csv
import io
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

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
    Advanced parser for GEO database files with optimized performance.
    
    This class provides methods to parse and extract meaningful data
    from various file formats commonly found in GEO datasets, with
    performance optimizations for large-scale genomics data.
    """
    
    def __init__(self):
        """Initialize the GEO parser."""
        logger.debug("GEOParser initialized with optimized parsing capabilities")

    def parse_expression_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Parse an expression data file with optimized performance using Polars + sniffer.
        
        Uses intelligent delimiter detection and Polars for faster parsing of large files,
        with pandas fallback for compatibility.

        Args:
            file_path: Path to expression file

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            logger.info(f"Parsing expression file: {file_path}")

            # Check if it's a Matrix Market format file (.mtx or .mtx.gz)
            if "matrix.mtx" in file_path.name.lower():
                logger.info(f"Detected Matrix Market format file: {file_path.name}")
                return self.parse_matrix_market_file_optimized(file_path)

            # Detect delimiter intelligently using sniffer
            delimiter = self.sniff_delimiter(file_path)
            logger.debug(f"Detected delimiter: '{delimiter}' for {file_path.name}")

            # Determine compression
            compression = "gzip" if file_path.name.endswith(".gz") else None

            # Try Polars first for better performance on large files
            try:
                import polars as pl
                logger.debug("Using Polars for optimized parsing")
                
                # Read with Polars - much faster for large files
                df_polars = pl.read_csv(
                    file_path,
                    has_header=True,
                    separator=delimiter,
                    try_parse_dates=False,
                    infer_schema_length=1000,  # Sample more rows for better schema detection
                    ignore_errors=True,  # Skip problematic rows
                    null_values=['', 'NA', 'N/A', 'null', 'NULL'],
                )
                
                # Convert to pandas for downstream compatibility
                df = df_polars.to_pandas()
                
                # Set first column as index if it looks like gene/feature names
                if df.shape[1] > 1 and df.iloc[:, 0].dtype == 'object':
                    df = df.set_index(df.columns[0])
                
                logger.info(f"Successfully parsed with Polars: {df.shape}")
                return df
                
            except ImportError:
                logger.debug("Polars not available, using pandas with optimized settings")
            except Exception as polars_error:
                logger.warning(f"Polars parsing failed: {polars_error}, falling back to pandas")

            # Fallback to pandas with optimized settings
            try:
                # Use chunking for very large files
                file_size = file_path.stat().st_size
                use_chunks = file_size > 100_000_000  # Files larger than 100MB
                
                if use_chunks:
                    logger.info(f"Large file detected ({file_size:,} bytes), using chunked reading")
                    return self.parse_large_file_in_chunks(file_path, delimiter, compression)
                else:
                    # Standard pandas parsing with optimizations
                    df = pd.read_csv(
                        file_path,
                        sep=delimiter,
                        index_col=0,
                        compression=compression,
                        low_memory=False,
                        engine='c',  # Use C engine for better performance
                        na_values=['', 'NA', 'N/A', 'null', 'NULL'],
                        keep_default_na=True,
                        dtype_backend='pyarrow'  # Use PyArrow backend if available
                    )
                    
                    logger.info(f"Successfully parsed with pandas: {df.shape}")
                    return df
                    
            except Exception as pandas_error:
                # Final fallback - try with basic settings
                logger.warning(f"Optimized pandas parsing failed: {pandas_error}, trying basic parsing")
                return self.parse_with_basic_pandas(file_path, delimiter, compression)

        except Exception as e:
            logger.error(f"Error parsing expression file {file_path}: {e}")
            return None

    def sniff_delimiter(self, file_path: Path, sample_size: int = 8192) -> str:
        """
        Intelligently detect delimiter from a sample of the file.
        Works with both compressed and uncompressed files.
        
        Args:
            file_path: Path to the file
            sample_size: Number of bytes to sample for delimiter detection
            
        Returns:
            str: Detected delimiter (defaults to tab if detection fails)
        """
        try:
            # Read sample from file (handle compression)
            if file_path.name.endswith(".gz"):
                with gzip.open(file_path, "rt") as f:
                    sample = f.read(sample_size)
            else:
                with open(file_path, "r") as f:
                    sample = f.read(sample_size)

            # Use csv.Sniffer to detect delimiter
            try:
                delimiter = csv.Sniffer().sniff(sample).delimiter
                logger.debug(f"Sniffed delimiter: '{delimiter}' for {file_path.name}")
                return delimiter
            except Exception:
                # Fallback: detect manually based on frequency
                delimiters = ['\t', ',', ';', '|', ' ']
                delimiter_counts = {d: sample.count(d) for d in delimiters}
                best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                
                # Only use if it appears frequently enough
                if delimiter_counts[best_delimiter] > 10:
                    logger.debug(f"Manual delimiter detection: '{best_delimiter}' for {file_path.name}")
                    return best_delimiter
                else:
                    logger.debug(f"Using default tab delimiter for {file_path.name}")
                    return '\t'
                    
        except Exception as e:
            logger.warning(f"Error sniffing delimiter: {e}, using tab")
            return '\t'

    def parse_large_file_in_chunks(
        self, file_path: Path, delimiter: str, compression: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """
        Parse very large files using chunked reading to manage memory efficiently.
        
        Args:
            file_path: Path to the file
            delimiter: Detected delimiter
            compression: Compression type
            
        Returns:
            DataFrame: Parsed expression matrix or None
        """
        try:
            logger.info(f"Using chunked reading for large file: {file_path.name}")
            
            chunk_size = 10000  # Process 10k rows at a time
            chunks = []
            
            # Read file in chunks
            chunk_reader = pd.read_csv(
                file_path,
                sep=delimiter,
                index_col=0,
                compression=compression,
                chunksize=chunk_size,
                low_memory=False,
                engine='c',
                na_values=['', 'NA', 'N/A', 'null', 'NULL'],
            )
            
            for i, chunk in enumerate(chunk_reader):
                chunks.append(chunk)
                if i % 10 == 0:  # Log progress every 100k rows
                    logger.debug(f"Processed {(i + 1) * chunk_size:,} rows")
                
                # Memory management: limit chunks to avoid OOM
                if len(chunks) > 50:  # If we have too many chunks, combine them
                    logger.debug("Combining intermediate chunks to manage memory")
                    combined_chunk = pd.concat(chunks, axis=0)
                    chunks = [combined_chunk]
            
            # Combine all chunks
            if chunks:
                df = pd.concat(chunks, axis=0)
                logger.info(f"Successfully parsed large file in chunks: {df.shape}")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing large file in chunks: {e}")
            return None

    def parse_with_basic_pandas(
        self, file_path: Path, delimiter: str, compression: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """
        Final fallback parsing with basic pandas settings.
        
        Args:
            file_path: Path to the file
            delimiter: Detected delimiter
            compression: Compression type
            
        Returns:
            DataFrame: Parsed expression matrix or None
        """
        try:
            logger.info(f"Using basic pandas parsing as final fallback: {file_path.name}")
            
            # Very basic pandas parsing
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                index_col=0,
                compression=compression,
                engine='python',  # More forgiving but slower
                error_bad_lines=False,  # Skip bad lines
                warn_bad_lines=False
            )
            
            if df.shape[0] > 0 and df.shape[1] > 0:
                logger.info(f"Successfully parsed with basic pandas: {df.shape}")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Basic pandas parsing also failed: {e}")
            return None

    def parse_matrix_market_file_optimized(self, matrix_file: Path) -> Optional[pd.DataFrame]:
        """
        Parse a Matrix Market format file with optimized performance.
        Uses sparse DataFrame when possible for memory efficiency.

        Args:
            matrix_file: Path to Matrix Market format file

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            logger.info(f"Parsing Matrix Market file with optimization: {matrix_file}")

            # Import scipy for sparse matrix handling
            try:
                import scipy.io as sio
                from scipy.sparse import csr_matrix
            except ImportError:
                logger.error("scipy is required for parsing Matrix Market files but not available")
                return None

            # Read the sparse matrix
            if matrix_file.name.endswith(".gz"):
                with gzip.open(matrix_file, "rt") as f:
                    matrix = sio.mmread(f)
            else:
                matrix = sio.mmread(matrix_file)

            # For very large matrices, consider keeping as sparse
            if matrix.shape[0] * matrix.shape[1] > 10_000_000:  # > 10M elements
                logger.info(f"Large sparse matrix detected, using sparse DataFrame: {matrix.shape}")
                
                # Transpose sparse matrix (genes x cells -> cells x genes)
                matrix_transposed = matrix.T
                
                # Create sparse DataFrame directly
                df = pd.DataFrame.sparse.from_spmatrix(
                    matrix_transposed,
                    index=[f"cell_{i}" for i in range(matrix_transposed.shape[0])],
                    columns=[f"gene_{i}" for i in range(matrix_transposed.shape[1])]
                )
                
                logger.info(f"Successfully created sparse DataFrame: {df.shape}")
                return df
            
            else:
                # Convert to dense for smaller matrices
                if hasattr(matrix, "todense"):
                    matrix_dense = matrix.todense()
                else:
                    matrix_dense = matrix

                # Transpose so that cells are rows and genes are columns
                matrix_dense = matrix_dense.T
                logger.debug(f"Matrix shape after transpose: {matrix_dense.shape}")

                # Create DataFrame with generic names (will be updated with real names if available)
                df = pd.DataFrame(
                    matrix_dense,
                    index=[f"cell_{i}" for i in range(matrix_dense.shape[0])],
                    columns=[f"gene_{i}" for i in range(matrix_dense.shape[1])]
                )

                logger.info(f"Successfully created dense DataFrame: {df.shape}")
                return df

        except Exception as e:
            logger.error(f"Error parsing Matrix Market file {matrix_file}: {e}")
            return None

    def parse_matrix_market_file_with_metadata(self, matrix_file: Path, sample_id: str = None) -> Optional[pd.DataFrame]:
        """
        Parse a Matrix Market format file (.mtx or .mtx.gz) and look for associated barcodes/features.
        This is the comprehensive version that tries to find and incorporate metadata.

        Args:
            matrix_file: Path to Matrix Market format file
            sample_id: Optional sample identifier for naming

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            logger.info(f"Parsing Matrix Market file with metadata: {matrix_file}")

            # Import scipy for sparse matrix handling
            try:
                import scipy.io as sio
                from scipy.sparse import csr_matrix
            except ImportError:
                logger.error("scipy is required for parsing Matrix Market files but not available")
                return None

            # Read the sparse matrix
            if matrix_file.name.endswith(".gz"):
                with gzip.open(matrix_file, "rt") as f:
                    matrix = sio.mmread(f)
            else:
                matrix = sio.mmread(matrix_file)

            # Convert to dense format and transpose (Matrix Market format is often genes x cells, we want cells x genes)
            if hasattr(matrix, "todense"):
                matrix_dense = matrix.todense()
            else:
                matrix_dense = matrix

            # Transpose so that cells are rows and genes are columns
            matrix_dense = matrix_dense.T
            logger.info(f"Matrix shape after transpose: {matrix_dense.shape}")

            # Look for associated barcodes and features files in the same directory
            matrix_dir = matrix_file.parent
            if not sample_id:
                sample_id = (
                    matrix_file.name.replace("_matrix.mtx.gz", "")
                    .replace("_matrix.mtx", "")
                    .replace("matrix.mtx.gz", "")
                    .replace("matrix.mtx", "")
                )
                if not sample_id:
                    sample_id = matrix_dir.name

            logger.info(f"Looking for barcodes/features files for sample: {sample_id}")

            # Find barcodes and features files
            barcodes_file = None
            features_file = None

            # Common patterns for barcodes and features files
            barcode_patterns = [
                "barcodes.tsv.gz", "barcodes.tsv",
                f"{sample_id}_barcodes.tsv.gz", f"{sample_id}_barcodes.tsv",
            ]
            feature_patterns = [
                "features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv",
                f"{sample_id}_features.tsv.gz", f"{sample_id}_features.tsv",
                f"{sample_id}_genes.tsv.gz", f"{sample_id}_genes.tsv",
            ]

            # Search in the same directory
            for file_path in matrix_dir.glob("*"):
                if file_path.is_file():
                    name_lower = file_path.name.lower()
                    if any(pattern.lower() in name_lower for pattern in barcode_patterns):
                        barcodes_file = file_path
                    elif any(pattern.lower() in name_lower for pattern in feature_patterns):
                        features_file = file_path

            logger.info(f"Found files - Barcodes: {barcodes_file.name if barcodes_file else 'None'}, "
                       f"Features: {features_file.name if features_file else 'None'}")

            # Read barcodes (cell identifiers)
            cell_ids = []
            if barcodes_file and barcodes_file.exists():
                try:
                    if barcodes_file.name.endswith(".gz"):
                        with gzip.open(barcodes_file, "rt") as f:
                            cell_ids = [line.strip() for line in f]
                    else:
                        with open(barcodes_file, "r") as f:
                            cell_ids = [line.strip() for line in f]
                    logger.info(f"Read {len(cell_ids)} cell barcodes")
                except Exception as e:
                    logger.warning(f"Error reading barcodes file: {e}")

            # Read features/genes
            gene_ids = []
            gene_names = []
            if features_file and features_file.exists():
                try:
                    if features_file.name.endswith(".gz"):
                        with gzip.open(features_file, "rt") as f:
                            lines = f.readlines()
                    else:
                        with open(features_file, "r") as f:
                            lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            gene_ids.append(parts[0])
                            gene_names.append(parts[1])
                        elif len(parts) == 1:
                            gene_ids.append(parts[0])
                            gene_names.append(parts[0])

                    logger.info(f"Read {len(gene_ids)} gene features")
                except Exception as e:
                    logger.warning(f"Error reading features file: {e}")

            # Create appropriate cell and gene names
            if not cell_ids:
                cell_ids = [f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])]
            else:
                # Add sample prefix to cell IDs
                cell_ids = [f"{sample_id}_{cell_id}" for cell_id in cell_ids]

            if not gene_names:
                if gene_ids:
                    gene_names = gene_ids
                else:
                    gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Ensure we have the right number of identifiers
            if len(cell_ids) != matrix_dense.shape[0]:
                logger.warning(f"Cell ID count mismatch: {len(cell_ids)} vs {matrix_dense.shape[0]}")
                cell_ids = [f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])]

            if len(gene_names) != matrix_dense.shape[1]:
                logger.warning(f"Gene name count mismatch: {len(gene_names)} vs {matrix_dense.shape[1]}")
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Create DataFrame
            df = pd.DataFrame(matrix_dense, index=cell_ids, columns=gene_names)
            logger.info(f"Successfully created Matrix Market DataFrame for {sample_id}: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error parsing Matrix Market file {matrix_file}: {e}")
            return None

    def parse_10x_data(self, extract_dir: Path, sample_id: str) -> Optional[pd.DataFrame]:
        """
        Parse 10X Genomics format data (matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz).

        Args:
            extract_dir: Directory containing 10X format files
            sample_id: Sample identifier for cell naming

        Returns:
            DataFrame: Expression matrix with cells as rows, genes as columns
        """
        try:
            logger.info(f"Parsing 10X data for {sample_id} in {extract_dir}")

            # Look for 10X files in the directory and subdirectories
            matrix_file = None
            barcodes_file = None
            features_file = None

            # Search recursively for 10X files
            for file_path in extract_dir.rglob("*"):
                if file_path.is_file():
                    name_lower = file_path.name.lower()
                    if "matrix.mtx" in name_lower:
                        matrix_file = file_path
                    elif "barcode" in name_lower or "barcodes" in name_lower:
                        barcodes_file = file_path
                    elif "feature" in name_lower or "genes" in name_lower:
                        features_file = file_path

            if not matrix_file:
                logger.warning(f"No matrix.mtx file found for {sample_id}")
                return None

            logger.info(f"Found 10X files - Matrix: {matrix_file.name}, "
                       f"Barcodes: {barcodes_file.name if barcodes_file else 'None'}, "
                       f"Features: {features_file.name if features_file else 'None'}")

            # Import scipy for sparse matrix handling
            try:
                import scipy.io as sio
                from scipy.sparse import csr_matrix
            except ImportError:
                logger.error("scipy is required for parsing 10X data but not available")
                return None

            # Read the sparse matrix
            logger.info(f"Reading sparse matrix from {matrix_file}")
            if matrix_file.name.endswith(".gz"):
                with gzip.open(matrix_file, "rt") as f:
                    matrix = sio.mmread(f)
            else:
                matrix = sio.mmread(matrix_file)

            # Convert to dense format and transpose (10X format is genes x cells, we want cells x genes)
            if hasattr(matrix, "todense"):
                matrix_dense = matrix.todense()
            else:
                matrix_dense = matrix

            # Transpose so that cells are rows and genes are columns
            matrix_dense = matrix_dense.T
            logger.info(f"Matrix shape after transpose: {matrix_dense.shape}")

            # Read barcodes (cell identifiers)
            cell_ids = []
            if barcodes_file and barcodes_file.exists():
                try:
                    if barcodes_file.name.endswith(".gz"):
                        with gzip.open(barcodes_file, "rt") as f:
                            cell_ids = [line.strip() for line in f]
                    else:
                        with open(barcodes_file, "r") as f:
                            cell_ids = [line.strip() for line in f]
                    logger.info(f"Read {len(cell_ids)} cell barcodes")
                except Exception as e:
                    logger.warning(f"Error reading barcodes file: {e}")

            # Read features/genes
            gene_ids = []
            gene_names = []
            if features_file and features_file.exists():
                try:
                    if features_file.name.endswith(".gz"):
                        with gzip.open(features_file, "rt") as f:
                            lines = f.readlines()
                    else:
                        with open(features_file, "r") as f:
                            lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            gene_ids.append(parts[0])
                            gene_names.append(parts[1])
                        elif len(parts) == 1:
                            gene_ids.append(parts[0])
                            gene_names.append(parts[0])

                    logger.info(f"Read {len(gene_ids)} gene features")
                except Exception as e:
                    logger.warning(f"Error reading features file: {e}")

            # Create appropriate cell and gene names
            if not cell_ids:
                cell_ids = [f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])]
            else:
                # Add sample prefix to cell IDs
                cell_ids = [f"{sample_id}_{cell_id}" for cell_id in cell_ids]

            if not gene_names:
                if gene_ids:
                    gene_names = gene_ids
                else:
                    gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Ensure we have the right number of identifiers
            if len(cell_ids) != matrix_dense.shape[0]:
                logger.warning(f"Cell ID count mismatch: {len(cell_ids)} vs {matrix_dense.shape[0]}")
                cell_ids = [f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])]

            if len(gene_names) != matrix_dense.shape[1]:
                logger.warning(f"Gene name count mismatch: {len(gene_names)} vs {matrix_dense.shape[1]}")
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Create DataFrame
            df = pd.DataFrame(matrix_dense, index=cell_ids, columns=gene_names)
            logger.info(f"Successfully created 10X DataFrame for {sample_id}: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error parsing 10X data for {sample_id}: {e}")
            return None

    def parse_supplementary_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Parse supplementary expression data file with enhanced format support.
        
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
                # Get the actual file type from the inner extension
                inner_suffix = Path(str(file_path)[:-3]).suffix.lower()
            else:
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
                # Use the optimized expression file parser for text formats
                return self.parse_expression_file(file_path)
            
            # Handle MTX format (common in single-cell data)
            elif inner_suffix == '.mtx' or file_path.name.lower().endswith('.mtx.gz'):
                return self.parse_matrix_market_file_optimized(file_path)
            
            logger.warning(f"Unsupported file format: {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing supplementary file {file_path}: {e}")
            return None

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
