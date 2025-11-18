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

import csv
import gzip
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import psutil

from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context, handle_ssl_error

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
        self._log_system_memory()

    def _log_system_memory(self):
        """Log current system memory status."""
        try:
            vm = psutil.virtual_memory()
            logger.debug(
                f"System memory: Total={self._format_bytes(vm.total)}, "
                f"Available={self._format_bytes(vm.available)}, "
                f"Used={vm.percent}%"
            )
        except Exception as e:
            logger.warning(f"Could not get system memory info: {e}")

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} PB"

    def _estimate_dataframe_memory(
        self, file_path: Path, sample_rows: int = 100
    ) -> Optional[int]:
        """
        Estimate memory required for a DataFrame based on file size and sample.

        Args:
            file_path: Path to the file
            sample_rows: Number of rows to sample for estimation

        Returns:
            Estimated memory in bytes, or None if estimation fails
        """
        try:
            file_size = file_path.stat().st_size

            # For compressed files, estimate uncompressed size (typically 5-10x)
            if file_path.name.endswith(".gz"):
                file_size *= 7  # Conservative estimate

            # Quick heuristic: CSV/TSV files typically expand 2-3x in memory as DataFrame
            # This is a rough estimate and can vary significantly
            estimated_memory = file_size * 2.5

            logger.debug(
                f"Estimated memory requirement for {file_path.name}: "
                f"{self._format_bytes(int(estimated_memory))}"
            )

            return int(estimated_memory)
        except Exception as e:
            logger.warning(f"Could not estimate memory for {file_path}: {e}")
            return None

    def _check_memory_availability(
        self, required_memory: int, safety_factor: float = 1.5
    ) -> bool:
        """
        Check if sufficient memory is available for an operation.

        Args:
            required_memory: Memory required in bytes
            safety_factor: Multiply required memory by this factor for safety margin

        Returns:
            True if sufficient memory is available, False otherwise
        """
        try:
            vm = psutil.virtual_memory()
            available_memory = vm.available
            total_required = int(required_memory * safety_factor)

            # Keep a reasonable safety buffer (500 MB)
            safety_buffer = 500 * 1024 * 1024  # 500 MB

            if available_memory < total_required + safety_buffer:
                logger.warning(
                    f"Insufficient memory: Required={self._format_bytes(total_required)}, "
                    f"Available={self._format_bytes(available_memory)}, "
                    f"Safety buffer={self._format_bytes(safety_buffer)}"
                )
                return False

            logger.debug(
                f"Memory check passed: Required={self._format_bytes(total_required)}, "
                f"Available={self._format_bytes(available_memory)}"
            )
            return True

        except Exception as e:
            logger.warning(f"Could not check memory availability: {e}")
            # Be conservative and return False if we can't check
            return False

    def _get_adaptive_chunk_size(
        self, file_path: Path, target_memory_mb: int = 500
    ) -> int:
        """
        Calculate adaptive chunk size based on available memory and file characteristics.

        Args:
            file_path: Path to the file
            target_memory_mb: Target memory usage per chunk in MB

        Returns:
            Number of rows per chunk
        """
        try:
            # Get available memory
            vm = psutil.virtual_memory()
            available_mb = vm.available / (1024 * 1024)

            # Use at most 50% of available memory per chunk
            max_chunk_memory_mb = min(available_mb * 0.5, target_memory_mb)

            # Estimate row size by sampling the file
            delimiter = self.sniff_delimiter(file_path)
            compression = "gzip" if file_path.name.endswith(".gz") else None

            # Read a small sample to estimate row size
            sample_df = pd.read_csv(
                file_path,
                sep=delimiter,
                compression=compression,
                nrows=100,
                low_memory=False,
            )

            # Estimate memory per row (in bytes)
            memory_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)

            # Calculate chunk size
            chunk_size = int((max_chunk_memory_mb * 1024 * 1024) / memory_per_row)

            # Ensure reasonable bounds
            chunk_size = max(1000, min(chunk_size, 50000))

            logger.debug(
                f"Adaptive chunk size for {file_path.name}: {chunk_size:,} rows "
                f"(~{self._format_bytes(int(chunk_size * memory_per_row))} per chunk)"
            )

            return chunk_size

        except Exception as e:
            logger.warning(
                f"Could not calculate adaptive chunk size: {e}, using default"
            )
            return 10000  # Default chunk size

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

            # Estimate memory requirement
            estimated_memory = self._estimate_dataframe_memory(file_path)
            if estimated_memory and not self._check_memory_availability(
                estimated_memory
            ):
                logger.warning(
                    f"File {file_path.name} may be too large for available memory. "
                    "Forcing chunked reading approach."
                )
                # Force chunked reading for large files
                delimiter = self.sniff_delimiter(file_path)
                compression = "gzip" if file_path.name.endswith(".gz") else None
                return self.parse_large_file_in_chunks(
                    file_path, delimiter, compression
                )

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
                    null_values=["", "NA", "N/A", "null", "NULL"],
                )

                # Convert to pandas for downstream compatibility
                df = df_polars.to_pandas()

                # Set first column as index if it looks like gene/feature names
                if df.shape[1] > 1 and df.iloc[:, 0].dtype == "object":
                    df = df.set_index(df.columns[0])

                logger.debug(f"Successfully parsed with Polars: {df.shape}")
                self._log_system_memory()  # Log memory after parsing
                return df

            except ImportError:
                logger.debug(
                    "Polars not available, using pandas with optimized settings"
                )
            except MemoryError:
                logger.error("Memory error with Polars, forcing chunked reading")
                return self.parse_large_file_in_chunks(
                    file_path, delimiter, compression
                )
            except Exception as polars_error:
                logger.warning(
                    f"Polars parsing failed: {polars_error}, falling back to pandas"
                )

            # Fallback to pandas with optimized settings
            try:
                # Use chunking for very large files
                file_size = file_path.stat().st_size
                use_chunks = file_size > 100_000_000  # Files larger than 100MB

                if use_chunks:
                    logger.info(
                        f"Large file detected ({file_size:,} bytes), using chunked reading"
                    )
                    return self.parse_large_file_in_chunks(
                        file_path, delimiter, compression
                    )
                else:
                    # Standard pandas parsing with optimizations
                    df = pd.read_csv(
                        file_path,
                        sep=delimiter,
                        index_col=0,
                        compression=compression,
                        low_memory=False,
                        engine="c",  # Use C engine for better performance
                        na_values=["", "NA", "N/A", "null", "NULL"],
                        keep_default_na=True,
                        dtype_backend="pyarrow",  # Use PyArrow backend if available
                    )

                    logger.debug(f"Successfully parsed with pandas: {df.shape}")
                    self._log_system_memory()  # Log memory after parsing
                    return df

            except MemoryError:
                logger.error("Memory error with pandas, forcing chunked reading")
                return self.parse_large_file_in_chunks(
                    file_path, delimiter, compression
                )
            except Exception as pandas_error:
                # Final fallback - try with basic settings
                logger.warning(
                    f"Optimized pandas parsing failed: {pandas_error}, trying basic parsing"
                )
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
                delimiters = ["\t", ",", ";", "|", " "]
                delimiter_counts = {d: sample.count(d) for d in delimiters}
                best_delimiter = max(delimiter_counts, key=delimiter_counts.get)

                # Only use if it appears frequently enough
                if delimiter_counts[best_delimiter] > 10:
                    logger.debug(
                        f"Manual delimiter detection: '{best_delimiter}' for {file_path.name}"
                    )
                    return best_delimiter
                else:
                    logger.debug(f"Using default tab delimiter for {file_path.name}")
                    return "\t"

        except Exception as e:
            logger.warning(f"Error sniffing delimiter: {e}, using tab")
            return "\t"

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

            # Get adaptive chunk size based on available memory
            chunk_size = self._get_adaptive_chunk_size(file_path)

            # Log memory status before starting
            self._log_system_memory()

            chunks = []
            total_rows = 0

            # Read file in chunks
            chunk_reader = pd.read_csv(
                file_path,
                sep=delimiter,
                index_col=0,
                compression=compression,
                chunksize=chunk_size,
                low_memory=False,
                engine="c",
                na_values=["", "NA", "N/A", "null", "NULL"],
            )

            for i, chunk in enumerate(chunk_reader):
                # Check memory before adding chunk
                chunk_memory = chunk.memory_usage(deep=True).sum()
                if not self._check_memory_availability(chunk_memory, safety_factor=1.2):
                    logger.warning(
                        f"Memory limit reached after {i} chunks ({total_rows:,} rows). "
                        f"Stopping early to prevent OOM."
                    )
                    break

                chunks.append(chunk)
                total_rows += len(chunk)

                if i % 10 == 0:  # Log progress
                    logger.debug(f"Processed {total_rows:,} rows in {i+1} chunks")
                    self._log_system_memory()

                # Memory management: dynamically adjust based on available memory
                vm = psutil.virtual_memory()
                if vm.percent > 80:  # If memory usage is high
                    logger.warning(
                        f"High memory usage ({vm.percent}%), combining chunks early"
                    )
                    if len(chunks) > 1:
                        combined_chunk = pd.concat(chunks, axis=0)
                        chunks = [combined_chunk]
                        # Force garbage collection
                        import gc

                        gc.collect()
                        logger.debug("Forced garbage collection after combining chunks")

            # Combine all chunks
            if chunks:
                logger.debug(
                    f"Combining {len(chunks)} chunks with {total_rows:,} total rows"
                )
                df = pd.concat(chunks, axis=0)
                logger.debug(f"Successfully parsed large file in chunks: {df.shape}")
                self._log_system_memory()  # Log final memory status
                return df

            logger.warning("No data was read from the file")
            return None

        except MemoryError:
            logger.error(
                "Memory error while parsing in chunks. File is too large for available memory."
            )
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
            logger.debug(
                f"Using basic pandas parsing as final fallback: {file_path.name}"
            )

            # Very basic pandas parsing
            df = pd.read_csv(
                file_path,
                sep=delimiter,
                index_col=0,
                compression=compression,
                engine="python",  # More forgiving but slower
                error_bad_lines=False,  # Skip bad lines
                warn_bad_lines=False,
            )

            if df.shape[0] > 0 and df.shape[1] > 0:
                logger.debug(f"Successfully parsed with basic pandas: {df.shape}")
                return df

            return None

        except Exception as e:
            logger.error(f"Basic pandas parsing also failed: {e}")
            return None

    def parse_matrix_market_file_optimized(
        self, matrix_file: Path
    ) -> Optional[pd.DataFrame]:
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
            except ImportError:
                logger.error(
                    "scipy is required for parsing Matrix Market files but not available"
                )
                return None

            # Read the sparse matrix
            if matrix_file.name.endswith(".gz"):
                with gzip.open(matrix_file, "rt") as f:
                    matrix = sio.mmread(f)
            else:
                matrix = sio.mmread(matrix_file)

            # For very large matrices, consider keeping as sparse
            if matrix.shape[0] * matrix.shape[1] > 10_000_000:  # > 10M elements
                logger.debug(
                    f"Large sparse matrix detected, using sparse DataFrame: {matrix.shape}"
                )

                # Transpose sparse matrix (genes x cells -> cells x genes)
                matrix_transposed = matrix.T

                # Create sparse DataFrame directly
                df = pd.DataFrame.sparse.from_spmatrix(
                    matrix_transposed,
                    index=[f"cell_{i}" for i in range(matrix_transposed.shape[0])],
                    columns=[f"gene_{i}" for i in range(matrix_transposed.shape[1])],
                )

                logger.debug(f"Successfully created sparse DataFrame: {df.shape}")
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
                    columns=[f"gene_{i}" for i in range(matrix_dense.shape[1])],
                )

                logger.debug(f"Successfully created dense DataFrame: {df.shape}")
                return df

        except Exception as e:
            logger.error(f"Error parsing Matrix Market file {matrix_file}: {e}")
            return None

    def parse_matrix_market_file_with_metadata(
        self, matrix_file: Path, sample_id: str = None
    ) -> Optional[pd.DataFrame]:
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
            except ImportError:
                logger.error(
                    "scipy is required for parsing Matrix Market files but not available"
                )
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

            logger.debug(f"Looking for barcodes/features files for sample: {sample_id}")

            # Find barcodes and features files
            barcodes_file = None
            features_file = None

            # Common patterns for barcodes and features files
            barcode_patterns = [
                "barcodes.tsv.gz",
                "barcodes.tsv",
                f"{sample_id}_barcodes.tsv.gz",
                f"{sample_id}_barcodes.tsv",
            ]
            feature_patterns = [
                "features.tsv.gz",
                "features.tsv",
                "genes.tsv.gz",
                "genes.tsv",
                f"{sample_id}_features.tsv.gz",
                f"{sample_id}_features.tsv",
                f"{sample_id}_genes.tsv.gz",
                f"{sample_id}_genes.tsv",
            ]

            # Search in the same directory
            for file_path in matrix_dir.glob("*"):
                if file_path.is_file():
                    name_lower = file_path.name.lower()
                    if any(
                        pattern.lower() in name_lower for pattern in barcode_patterns
                    ):
                        barcodes_file = file_path
                    elif any(
                        pattern.lower() in name_lower for pattern in feature_patterns
                    ):
                        features_file = file_path

            logger.debug(
                f"Found files - Barcodes: {barcodes_file.name if barcodes_file else 'None'}, "
                f"Features: {features_file.name if features_file else 'None'}"
            )

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
                    logger.debug(f"Read {len(cell_ids)} cell barcodes")
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

                    logger.debug(f"Read {len(gene_ids)} gene features")
                except Exception as e:
                    logger.warning(f"Error reading features file: {e}")

            # Create appropriate cell and gene names
            if not cell_ids:
                cell_ids = [
                    f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])
                ]
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
                logger.warning(
                    f"Cell ID count mismatch: {len(cell_ids)} vs {matrix_dense.shape[0]}"
                )
                cell_ids = [
                    f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])
                ]

            if len(gene_names) != matrix_dense.shape[1]:
                logger.warning(
                    f"Gene name count mismatch: {len(gene_names)} vs {matrix_dense.shape[1]}"
                )
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Create DataFrame
            df = pd.DataFrame(matrix_dense, index=cell_ids, columns=gene_names)
            logger.debug(
                f"Successfully created Matrix Market DataFrame for {sample_id}: {df.shape}"
            )
            return df

        except Exception as e:
            logger.error(f"Error parsing Matrix Market file {matrix_file}: {e}")
            return None

    def parse_10x_data(
        self, extract_dir: Path, sample_id: str
    ) -> Optional[pd.DataFrame]:
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

            logger.debug(
                f"Found 10X files - Matrix: {matrix_file.name}, "
                f"Barcodes: {barcodes_file.name if barcodes_file else 'None'}, "
                f"Features: {features_file.name if features_file else 'None'}"
            )

            # Import scipy for sparse matrix handling
            try:
                import scipy.io as sio
            except ImportError:
                logger.error("scipy is required for parsing 10X data but not available")
                return None

            # Read the sparse matrix
            logger.debug(f"Reading sparse matrix from {matrix_file}")
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
                    logger.debug(f"Read {len(cell_ids)} cell barcodes")
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

                    logger.debug(f"Read {len(gene_ids)} gene features")
                except Exception as e:
                    logger.warning(f"Error reading features file: {e}")

            # Create appropriate cell and gene names
            if not cell_ids:
                cell_ids = [
                    f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])
                ]
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
                logger.warning(
                    f"Cell ID count mismatch: {len(cell_ids)} vs {matrix_dense.shape[0]}"
                )
                cell_ids = [
                    f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])
                ]

            if len(gene_names) != matrix_dense.shape[1]:
                logger.warning(
                    f"Gene name count mismatch: {len(gene_names)} vs {matrix_dense.shape[1]}"
                )
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Create DataFrame
            df = pd.DataFrame(matrix_dense, index=cell_ids, columns=gene_names)
            logger.debug(
                f"Successfully created 10X DataFrame for {sample_id}: {df.shape}"
            )
            return df

        except Exception as e:
            logger.error(f"Error parsing 10X data for {sample_id}: {e}")
            return None

    def _parse_h5ad_with_fallback(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Parse H5AD files with multi-strategy fallback for legacy formats and corruption.

        Strategy:
        1. Modern anndata (0.8+): Try anndata.read_h5ad()
        2. Legacy anndata (0.7.x): Fallback to direct h5py reading (old 'matrix' parameter)
        3. HDF5 corruption: Detect and report unrecoverable corruption

        Args:
            file_path: Path to H5AD file (may be .gz compressed)

        Returns:
            DataFrame with expression data, or None if all strategies fail

        Raises:
            None - all exceptions are caught and logged
        """
        # Strategy 1: Try modern anndata format
        try:
            import anndata

            adata = anndata.read_h5ad(file_path)
            logger.debug(f"Parsed h5ad file (modern format): {file_path}")

            # Convert to DataFrame
            return pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

        except ImportError:
            logger.warning("anndata package not available, cannot parse h5ad")
            return None

        except TypeError as e:
            error_msg = str(e)

            # Strategy 2: Detect legacy format (anndata 0.7.x with 'matrix' parameter)
            if "'matrix'" in error_msg or "unexpected keyword argument" in error_msg:
                logger.info(
                    f"Detected legacy H5AD format in {file_path.name}. "
                    f"Attempting fallback parser..."
                )
                return self._parse_legacy_h5ad(file_path)
            else:
                logger.warning(f"TypeError parsing h5ad (not legacy format): {e}")
                return None

        except OSError as e:
            error_msg = str(e)

            # Strategy 3: Detect HDF5 corruption
            if (
                "bad object header" in error_msg
                or "Unable to synchronously open" in error_msg
            ):
                logger.error(
                    f"HDF5 file corruption detected in {file_path.name}: {error_msg}. "
                    f"File is unrecoverable without re-download."
                )
                return None
            elif "File signature not found" in error_msg or "Truncated" in error_msg:
                logger.error(
                    f"Incomplete/truncated HDF5 file: {file_path.name}. "
                    f"Download likely interrupted."
                )
                return None
            else:
                logger.warning(f"OS error parsing h5ad: {e}")
                return None

        except Exception as e:
            logger.warning(f"Failed to parse h5ad file: {e}")
            return None

    def _parse_legacy_h5ad(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Parse legacy anndata 0.7.x H5AD format using direct h5py access.

        Legacy format (pre-0.8) used 'matrix' parameter in AnnData.__init__(),
        which was renamed to 'X' in modern versions. This parser reads the raw
        HDF5 structure directly to bypass format incompatibilities.

        Args:
            file_path: Path to legacy H5AD file

        Returns:
            DataFrame with expression data, or None if parsing fails

        Technical Note:
            Legacy H5AD structure:
            - /X or /matrix: Expression matrix (cells × genes)
            - /obs_names or /obs/_index: Cell barcodes
            - /var_names or /var/_index: Gene names
        """
        try:
            import h5py
            import scipy.sparse as sp

            logger.debug(f"Opening legacy H5AD file with h5py: {file_path}")

            with h5py.File(file_path, "r") as f:
                # Explore available keys for diagnostics
                available_keys = list(f.keys())
                logger.debug(f"Legacy H5AD keys: {available_keys}")

                # Try to read expression matrix (multiple naming conventions)
                X = None
                if "X" in f:
                    X = f["X"][:]
                elif "matrix" in f:
                    X = f["matrix"][:]
                else:
                    logger.error(
                        f"No expression matrix found in {file_path.name}. "
                        f"Available keys: {available_keys}"
                    )
                    return None

                # Handle sparse matrices
                if isinstance(X, h5py.Group):
                    # Sparse CSR matrix format
                    if all(k in X for k in ["data", "indices", "indptr"]):
                        X = sp.csr_matrix(
                            (X["data"][:], X["indices"][:], X["indptr"][:])
                        ).toarray()
                    else:
                        logger.error(
                            f"Unsupported sparse matrix format in {file_path.name}"
                        )
                        return None

                # Read observation names (cells)
                obs_names = None
                for possible_key in ["obs_names", "obs/_index", "obs/index"]:
                    try:
                        if (
                            possible_key in f
                            or "/" in possible_key
                            and possible_key.split("/")[0] in f
                        ):
                            obs_data = f[possible_key][:]
                            obs_names = [
                                s.decode() if isinstance(s, bytes) else str(s)
                                for s in obs_data
                            ]
                            break
                    except:
                        continue

                if obs_names is None:
                    # Fallback: use indices
                    obs_names = [f"Cell_{i}" for i in range(X.shape[0])]
                    logger.warning(f"No cell names found, using generic indices")

                # Read variable names (genes)
                var_names = None
                for possible_key in ["var_names", "var/_index", "var/index"]:
                    try:
                        if (
                            possible_key in f
                            or "/" in possible_key
                            and possible_key.split("/")[0] in f
                        ):
                            var_data = f[possible_key][:]
                            var_names = [
                                s.decode() if isinstance(s, bytes) else str(s)
                                for s in var_data
                            ]
                            break
                    except:
                        continue

                if var_names is None:
                    # Fallback: use indices
                    var_names = [f"Gene_{i}" for i in range(X.shape[1])]
                    logger.warning(f"No gene names found, using generic indices")

                # Validate dimensions
                if len(obs_names) != X.shape[0]:
                    logger.error(
                        f"Dimension mismatch: {len(obs_names)} cells vs {X.shape[0]} rows"
                    )
                    return None

                if len(var_names) != X.shape[1]:
                    logger.error(
                        f"Dimension mismatch: {len(var_names)} genes vs {X.shape[1]} columns"
                    )
                    return None

                # Create DataFrame
                df = pd.DataFrame(X, index=obs_names, columns=var_names)
                logger.info(
                    f"Successfully parsed legacy H5AD: {df.shape[0]} cells × {df.shape[1]} genes"
                )
                return df

        except ImportError:
            logger.error("h5py package not available for legacy H5AD parsing")
            return None
        except Exception as e:
            logger.error(f"Failed to parse legacy H5AD format: {e}")
            import traceback

            logger.debug(traceback.format_exc())
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
            if file_suffix == ".gz":
                # Get the actual file type from the inner extension
                inner_suffix = Path(str(file_path)[:-3]).suffix.lower()
            else:
                inner_suffix = file_suffix

            # Handle specific file formats
            if inner_suffix in [".h5", ".h5ad"]:
                return self._parse_h5ad_with_fallback(file_path)

            elif inner_suffix == ".rds":
                try:
                    import pyreadr

                    result = pyreadr.read_r(str(file_path))
                    logger.debug(f"Parsed RDS file: {file_path}")
                    # Get the first dataframe in the result
                    if result and result[list(result.keys())[0]] is not None:
                        return result[list(result.keys())[0]]
                except ImportError:
                    logger.warning("pyreadr package not available, cannot parse RDS")
                    return None
                except Exception as e:
                    logger.warning(f"Failed to parse RDS file: {e}")
                    return None

            elif inner_suffix in [".txt", ".tsv", ".csv"]:
                # Use the optimized expression file parser for text formats
                return self.parse_expression_file(file_path)

            # Handle MTX format (common in single-cell data)
            elif inner_suffix == ".mtx" or file_path.name.lower().endswith(".mtx.gz"):
                return self.parse_matrix_market_file_optimized(file_path)

            logger.warning(f"Unsupported file format: {file_path}")
            return None

        except Exception as e:
            logger.error(f"Error parsing supplementary file {file_path}: {e}")
            return None

    @staticmethod
    def parse_soft_file(
        file_path: Path,
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Parse a GEO SOFT format file.

        Args:
            file_path: Path to the SOFT file

        Returns:
            tuple: DataFrame of expression data and metadata dictionary
        """
        try:
            if file_path.suffix == ".gz":
                opener = gzip.open
            else:
                opener = open

            with opener(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                return GEOParser._parse_soft_content(f)

        except Exception as e:
            logger.error(f"Error parsing SOFT file {file_path}: {e}")
            return None, {}

    @staticmethod
    def _parse_soft_content(
        file_handle,
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
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
            if line.startswith("^SERIES"):
                current_section = "series"
                series_id = line.split("=")[1].strip() if "=" in line else ""
                series_info["id"] = series_id
            elif line.startswith("!Series_"):
                if current_section == "series":
                    key = line.split("=")[0][1:].strip()  # Remove !
                    value = line.split("=")[1].strip() if "=" in line else ""
                    series_info[key] = value

            # Platform metadata
            elif line.startswith("^PLATFORM"):
                current_section = "platform"
                current_platform = line.split("=")[1].strip() if "=" in line else ""
                platforms[current_platform] = {}
            elif line.startswith("!platform_") and current_section == "platform":
                key = line.split("=")[0][1:].strip()
                value = line.split("=")[1].strip() if "=" in line else ""
                platforms[current_platform][key] = value

            # Sample metadata
            elif line.startswith("^SAMPLE"):
                current_section = "sample"
                current_sample = line.split("=")[1].strip() if "=" in line else ""
                samples[current_sample] = {}
                data_section = False
            elif line.startswith("!Sample_") and current_section == "sample":
                key = line.split("=")[0][1:].strip()
                value = line.split("=")[1].strip() if "=" in line else ""
                samples[current_sample][key] = value
            elif line.startswith("!sample_table_begin") and current_section == "sample":
                data_section = True
                continue
            elif line.startswith("!sample_table_end"):
                data_section = False
                continue
            elif data_section and current_section == "sample":
                data_lines.append(line)

        # Combine metadata
        metadata = {
            "series": series_info,
            "platforms": platforms,
            "samples": samples,
            "n_samples": len(samples),
        }

        # Try to construct expression matrix
        expression_data = GEOParser._construct_expression_matrix(samples, data_lines)

        return expression_data, metadata

    @staticmethod
    def _construct_expression_matrix(
        samples: Dict, data_lines: list
    ) -> Optional[pd.DataFrame]:
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

            logger.warning(
                "Expression matrix construction from SOFT file not implemented"
            )
            return None

        except Exception as e:
            logger.error(f"Error constructing expression matrix: {e}")
            return None

    @staticmethod
    def parse_matrix_file(
        file_path: Path,
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Parse GEO series matrix file.

        Args:
            file_path: Path to the matrix file

        Returns:
            tuple: DataFrame of expression data and metadata dictionary
        """
        try:
            if file_path.suffix == ".gz":
                opener = gzip.open
            else:
                opener = open

            metadata = {}
            data_start_line = 0

            with opener(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Find metadata and data start
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith("!"):
                    # Metadata line
                    if "=" in line:
                        key = line[1:].split("=")[0].strip()
                        value = line.split("=")[1].strip().strip('"')
                        metadata[key] = value
                elif line.startswith('"ID_REF"') or line.startswith("ID_REF"):
                    data_start_line = i
                    break

            # Read data
            if data_start_line > 0:
                data_df = pd.read_csv(
                    io.StringIO("".join(lines[data_start_line:])),
                    sep="\t",
                    index_col=0,
                    low_memory=False,
                )

                # Transpose so cells are rows, genes are columns
                data_df = data_df.T

                return data_df, metadata

            return None, metadata

        except Exception as e:
            logger.error(f"Error parsing matrix file {file_path}: {e}")
            return None, {}

    @staticmethod
    def analyze_data_sources(
        soft_file_path: Path, data_sources: Dict[str, Path]
    ) -> Dict[str, Any]:
        """
        Analyze available data sources and SOFT file to determine the best source.

        Args:
            soft_file_path: Path to the SOFT file
            data_sources: Dictionary of data sources (supplementary, tar)

        Returns:
            dict: Analysis results including recommended data source
        """
        analysis = {
            "available_sources": list(data_sources.keys()),
            "recommended_source": None,
            "reason": None,
        }

        try:
            # Parse the SOFT file to get metadata
            _, metadata = GEOParser.parse_soft_file(soft_file_path)

            # Extract relevant information from metadata
            series_info = metadata.get("series", {})
            metadata.get("platforms", {})
            metadata.get("samples", {})

            # Look for clues in the metadata about data format
            series_info.get("Series_type", "").lower()
            raw_data_mentioned = False
            processed_data_mentioned = False

            # Check series title and summary for clues
            title = series_info.get("Series_title", "").lower()
            summary = series_info.get("Series_summary", "").lower()

            # Keywords that suggest raw data
            raw_keywords = [
                "raw",
                "counts",
                "fastq",
                "seq",
                "single-cell",
                "single cell",
                "scrna-seq",
                "scrnaseq",
                "10x",
                "rna-seq",
                "rnaseq",
            ]

            # Check if raw data is mentioned
            for keyword in raw_keywords:
                if keyword in title or keyword in summary:
                    raw_data_mentioned = True
                    break

            # Check if processed data is mentioned
            processed_keywords = ["normalized", "processed", "expression matrix"]
            for keyword in processed_keywords:
                if keyword in title or keyword in summary:
                    processed_data_mentioned = True
                    break

            # Decision logic
            if "tar" in data_sources and "supplementary" in data_sources:
                if raw_data_mentioned and not processed_data_mentioned:
                    analysis["recommended_source"] = "tar"
                    analysis["reason"] = (
                        "SOFT file metadata indicates raw data, using TAR file"
                    )
                elif processed_data_mentioned and not raw_data_mentioned:
                    analysis["recommended_source"] = "supplementary"
                    analysis["reason"] = (
                        "SOFT file metadata indicates processed data, using supplementary file"
                    )
                else:
                    # Check file sizes as a heuristic - larger files are often raw data
                    try:
                        tar_size = data_sources["tar"].stat().st_size
                        suppl_size = data_sources["supplementary"].stat().st_size

                        if tar_size > suppl_size * 2:  # If TAR file is much larger
                            analysis["recommended_source"] = "tar"
                            analysis["reason"] = (
                                "TAR file is significantly larger, likely contains more complete data"
                            )
                        else:
                            analysis["recommended_source"] = "supplementary"
                            analysis["reason"] = (
                                "Supplementary file appears adequate based on size comparison"
                            )
                    except Exception:
                        # Default to TAR if we can't compare sizes
                        analysis["recommended_source"] = "tar"
                        analysis["reason"] = (
                            "Unable to compare file sizes, defaulting to TAR file"
                        )
            elif "tar" in data_sources:
                analysis["recommended_source"] = "tar"
                analysis["reason"] = "Only TAR file is available"
            elif "supplementary" in data_sources:
                analysis["recommended_source"] = "supplementary"
                analysis["reason"] = "Only supplementary file is available"
            elif "tar_dir" in data_sources:
                analysis["recommended_source"] = "tar_dir"
                analysis["reason"] = (
                    "Only TAR directory is available, but no suitable expression file was found"
                )
            else:
                analysis["recommended_source"] = None
                analysis["reason"] = "No suitable data sources found"

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing data sources: {e}")
            analysis["recommended_source"] = (
                list(data_sources.keys())[0] if data_sources else None
            )
            analysis["reason"] = (
                f"Error during analysis, using first available source: {e}"
            )
            return analysis

    def show_dynamic_head(
        self, target_url: str, rows: int = 5, cols: int = 5
    ) -> pd.DataFrame:
        """
        Download and preview the first rows/columns of a file from URL.

        Handles CSV, TSV, TXT, XLSX and other tabular formats intelligently.

        Args:
            target_url: URL (http/https/ftp) to download the file from
            rows: Number of rows to display (default: 5)
            cols: Number of columns to display (default: 5)

        Returns:
            DataFrame: Preview of the file (first 5 rows x 5 columns)
        """
        import tempfile
        import urllib.request
        from pathlib import Path

        temp_dir = None
        try:
            # Create temporary directory for download
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir)

            # Extract filename from URL
            filename = target_url.split("/")[-1].split("?")[0]
            if not filename:
                filename = "downloaded_file"

            local_file = temp_path / filename

            # Convert FTP to HTTPS for reliable downloads with TLS error detection
            # (FTP lacks error detection and causes silent corruption during throttling)
            if target_url.startswith("ftp://"):
                target_url = target_url.replace("ftp://", "https://", 1)
                logger.debug(f"Converted FTP to HTTPS: {target_url}")

            # Download file with proper SSL context and certificate verification
            logger.info(f"Downloading file from: {target_url}")
            ssl_context = create_ssl_context()
            try:
                with urllib.request.urlopen(
                    target_url, context=ssl_context
                ) as response:
                    with open(local_file, "wb") as out_file:
                        out_file.write(response.read())
            except Exception as e:
                error_str = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
                    handle_ssl_error(e, target_url, logger)
                    raise Exception(
                        "SSL certificate verification failed when downloading file. "
                        "See error message above for solutions."
                    )
                raise

            logger.debug(f"Downloaded to: {local_file}")

            # Determine file type and read accordingly
            suffix = local_file.suffix.lower()

            # Excel files
            if suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(local_file, nrows=rows)

            # Compressed files
            elif suffix == ".gz":
                inner_name = local_file.stem
                inner_suffix = Path(inner_name).suffix.lower()

                if inner_suffix in [".csv", ".txt", ".tsv"] or not inner_suffix:
                    delimiter = self.sniff_delimiter(local_file)
                    df = pd.read_csv(
                        local_file,
                        sep=delimiter,
                        compression="gzip",
                        nrows=rows,
                        low_memory=False,
                    )
                else:
                    raise ValueError(f"Unsupported compressed file type: {inner_name}")

            # Regular text files (CSV, TSV, TXT)
            else:
                delimiter = self.sniff_delimiter(local_file)
                df = pd.read_csv(
                    local_file, sep=delimiter, nrows=rows, low_memory=False
                )

            # Limit to first N columns
            if df.shape[1] > cols:
                df = df.iloc[:, :cols]

            logger.debug(f"Preview shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error previewing file from {target_url}: {e}")
            raise
        finally:
            # Clean up temporary files
            if temp_dir and Path(temp_dir).exists():
                import shutil

                shutil.rmtree(temp_dir)
                logger.debug("Cleaned up temporary files")
