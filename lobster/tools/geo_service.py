"""
Professional GEO data service using GEOparse for DataManagerV2.

This service provides a unified interface for downloading and processing
data from the Gene Expression Omnibus (GEO) database using GEOparse library
and the modular DataManagerV2 system.
"""

import json
import os
import re
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import GEOparse
except ImportError:
    GEOparse = None

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class GEOService:
    """
    Professional service for accessing and processing GEO data using GEOparse and DataManagerV2.

    This class provides a high-level interface for working with GEO data,
    handling the downloading, parsing, and processing of datasets using GEOparse
    and storing them as modalities in the DataManagerV2 system.
    """

    def __init__(
        self, data_manager: DataManagerV2, cache_dir: Optional[str] = None, console=None
    ):
        """
        Initialize the GEO service.

        Args:
            data_manager: DataManagerV2 instance for storing processed data as modalities
            cache_dir: Directory to cache downloaded files
            console: Rich console instance for display (creates new if None)
        """
        if GEOparse is None:
            raise ImportError(
                "GEOparse is required but not installed. Please install with: pip install GEOparse"
            )

        self.data_manager = data_manager
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_manager.cache_dir / "geo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.console = console

        logger.info("GEOService initialized with GEOparse backend and DataManagerV2")

    def fetch_metadata_only(self, geo_id: str) -> str:
        """
        Fetch and validate GEO metadata without downloading expression data.
        
        This function only downloads the SOFT file metadata and validates it against
        the transcriptomics schema, storing the metadata in data_manager for user review.
        
        Args:
            geo_id: GEO accession ID (e.g., GSE194247)
            
        Returns:
            str: Formatted metadata summary for user review
        """
        try:
            logger.info(f"Fetching metadata for GEO ID: {geo_id}")
            
            # Clean the GEO ID
            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith('GSE'):
                return f"Invalid GEO ID format: {geo_id}. Must be a GSE accession (e.g., GSE194247)."
            
            # Download SOFT data using GEOparse (metadata only)
            logger.info(f"Downloading SOFT metadata for {clean_geo_id}...")
            gse = GEOparse.get_GEO(geo=clean_geo_id, destdir=str(self.cache_dir))
            
            # Extract comprehensive metadata
            metadata = self._extract_metadata(gse)
            logger.info(f"Extracted metadata for {clean_geo_id}")
            
            # Validate metadata against transcriptomics schema
            validation_result = self._validate_geo_metadata(metadata)
            
            # Store metadata in data_manager for future use
            # metadata_modality_name = f"geo_{clean_geo_id.lower()}_metadata"
            # self.data_manager.metadata_store = getattr(self.data_manager, 'metadata_store', {})
            if clean_geo_id in self.data_manager.metadata_store:
                return f'{clean_geo_id} already stored in metadata_store.'
            
            self.data_manager.metadata_store[clean_geo_id] = {
                'metadata': metadata,
                'validation': validation_result,
                'fetch_timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Log the metadata fetch operation
            self.data_manager.log_tool_usage(
                tool_name="fetch_geo_metadata",
                parameters={"geo_id": clean_geo_id},
                description=f"Fetched metadata for GEO dataset {clean_geo_id}"
            )
            
            # Format comprehensive metadata summary
            summary = self._format_metadata_summary(clean_geo_id, metadata, validation_result)
            
            logger.info(f"Successfully fetched and validated metadata for {clean_geo_id}")
            return summary
            
        except Exception as e:
            logger.exception(f"Error fetching metadata for {geo_id}: {e}")
            return f"Error fetching metadata for {geo_id}: {str(e)}"

    def download_dataset(self, clean_geo_id: str) -> str:
        """
        Download and process a dataset from GEO using GEOparse.

        Args:
            clean_geo_id: Cleaned GEO accession ID

        Returns:
            str: Status message with detailed information
        """
        try:
            logger.info(f"Processing GEO query: {clean_geo_id}")

            # Check if metadata already exists (should be fetched first)
            if clean_geo_id not in self.data_manager.metadata_store:
                return f"Please fetch metadata first using fetch_metadata_only('{clean_geo_id}') before downloading the dataset."

            # Check if modality already exists in DataManagerV2
            modality_name = f"geo_{clean_geo_id.lower()}"
            existing_modalities = self.data_manager.list_modalities()

            # Step 1: Use existing metadata
            cached_metadata = self.data_manager.metadata_store[clean_geo_id]['metadata']
            logger.info(f"Using cached metadata for {clean_geo_id}")

            # Step 2: Download SOFT data using GEOparse (re-download for expression data processing)
            logger.info(f"Downloading SOFT data for {clean_geo_id} using GEOparse...")
            gse = GEOparse.get_GEO(geo=clean_geo_id, destdir=str(self.cache_dir))

            # Step 3: Check for supplementary files (TAR, etc.) first
            supplementary_data = self._process_supplementary_files(gse, clean_geo_id)

            if supplementary_data is not None and not supplementary_data.empty:
                logger.info(f"Successfully processed supplementary files for {clean_geo_id}")
                combined_matrix = supplementary_data
                sample_count = len(gse.gsms) if hasattr(gse, "gsms") else 0
            else:
                # Step 4: Fallback to individual sample matrices
                sample_info = self._get_sample_info(gse)
                logger.info(f"Found {len(sample_info)} samples in {clean_geo_id}")

                sample_matrices = self._download_sample_matrices(sample_info, clean_geo_id)
                validated_matrices = self._validate_matrices(sample_matrices)

                if not validated_matrices:
                    return f"No valid expression matrices found for {clean_geo_id}. The dataset may not contain downloadable expression data matrices."

                combined_matrix = self._concatenate_matrices(validated_matrices, clean_geo_id)
                sample_count = len(validated_matrices)

            if combined_matrix is None:
                return f"Failed to process expression data for {clean_geo_id}"

            # Step 5: Store as modality in DataManagerV2
            enhanced_metadata = {
                "dataset_id": clean_geo_id,
                "dataset_type": "GEO",
                "source_metadata": cached_metadata,
                "n_samples": sample_count,
                "data_source": "supplementary_files"
                if supplementary_data is not None
                else "individual_samples",
                "processing_date": pd.Timestamp.now().isoformat(),
                "geoparse_version": "GEOparse-based"
            }

            # Determine appropriate adapter based on data characteristics
            n_obs, n_vars = combined_matrix.shape
            if n_obs > 1000 and n_vars > 5000:
                adapter_name = "transcriptomics_single_cell"
            elif n_obs < 100:
                adapter_name = "transcriptomics_bulk"
            else:
                adapter_name = "transcriptomics_single_cell"  # Default for GEO

            # Load as modality in DataManagerV2
            modality_name = f"geo_{clean_geo_id.lower()}"
            adata = self.data_manager.load_modality(
                name=modality_name,
                source=combined_matrix,
                adapter=adapter_name,
                validate=True,
                **enhanced_metadata
            )

            # Save to workspace
            save_path = f"{clean_geo_id.lower()}_raw.h5ad"
            saved_file = self.data_manager.save_modality(modality_name, save_path)

            # Log successful download and save
            self.data_manager.log_tool_usage(
                tool_name="download_geo_dataset_geoparse",
                parameters={"geo_id": clean_geo_id},
                description=f"Downloaded and processed GEO dataset {clean_geo_id} using GEOparse, saved to {saved_file}",
            )

            # Auto-save current state
            self.data_manager.auto_save_state()

            return f"""Successfully downloaded and loaded GEO dataset {clean_geo_id}!

📊 Modality: '{modality_name}' ({adata.n_obs} obs × {adata.n_vars} vars)
🔬 Adapter: {adapter_name}
💾 Saved to: {save_path}
🎯 Source: GEO Database
⚡ Ready for quality control and downstream analysis!

The dataset is now available as modality '{modality_name}' for other agents to use."""

        except Exception as e:
            logger.exception(f"Error downloading dataset: {e}")
            return f"Error downloading dataset: {str(e)}"

    def _process_supplementary_files(self, gse, gse_id: str) -> Optional[pd.DataFrame]:
        """
        Process supplementary files (TAR archives, etc.) to extract expression data.

        Args:
            gse: GEOparse GSE object
            gse_id: GEO series ID

        Returns:
            DataFrame: Combined expression matrix from supplementary files or None
        """
        try:
            if not hasattr(gse, "metadata") or "supplementary_file" not in gse.metadata:
                logger.info(f"No supplementary files found for {gse_id}")
                return None

            suppl_files = gse.metadata["supplementary_file"]
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]

            logger.info(f"Found {len(suppl_files)} supplementary files for {gse_id}")

            # Look for TAR files first (most common for expression data)
            tar_files = [f for f in suppl_files if f.lower().endswith(".tar")]

            if tar_files:
                logger.info(f"Processing TAR file: {tar_files[0]}")
                return self._process_tar_file(tar_files[0], gse_id)

            # Look for other expression data files
            expression_files = [
                f
                for f in suppl_files
                if any(
                    ext in f.lower()
                    for ext in [".txt.gz", ".csv.gz", ".tsv.gz", ".h5", ".h5ad"]
                )
            ]

            if expression_files:
                logger.info(f"Processing expression file: {expression_files[0]}")
                return self._download_and_parse_file(expression_files[0], gse_id)

            logger.warning(
                f"No suitable expression files found in supplementary files for {gse_id}"
            )
            return None

        except Exception as e:
            logger.error(f"Error processing supplementary files: {e}")
            return None

    def _process_tar_file(self, tar_url: str, gse_id: str) -> Optional[pd.DataFrame]:
        """
        Download and process a TAR file containing expression data.

        Args:
            tar_url: URL to TAR file
            gse_id: GEO series ID

        Returns:
            DataFrame: Combined expression matrix or None
        """
        try:
            # Download TAR file
            tar_file_path = self.cache_dir / f"{gse_id}_RAW.tar"

            if not tar_file_path.exists():
                logger.info(f"Downloading TAR file from: {tar_url}")
                urllib.request.urlretrieve(tar_url, tar_file_path)
                logger.info(f"Downloaded TAR file: {tar_file_path}")
            else:
                logger.info(f"Using cached TAR file: {tar_file_path}")

            # Extract TAR file
            extract_dir = self.cache_dir / f"{gse_id}_extracted"
            if not extract_dir.exists():
                logger.info(f"Extracting TAR file to: {extract_dir}")
                extract_dir.mkdir(exist_ok=True)

                with tarfile.open(tar_file_path, "r") as tar:
                    # Security check for path traversal
                    def is_safe_member(member):
                        member_path = Path(member.name)
                        try:
                            target_path = (extract_dir / member_path).resolve()
                            common_path = Path(
                                os.path.commonpath([extract_dir.resolve(), target_path])
                            )
                            return common_path == extract_dir.resolve()
                        except (ValueError, RuntimeError):
                            return False

                    safe_members = [m for m in tar.getmembers() if is_safe_member(m)]
                    tar.extractall(path=extract_dir, members=safe_members)

                logger.info(f"Extracted {len(safe_members)} files from TAR")

            # Process nested archives and find expression data
            nested_extract_dir = self.cache_dir / f"{gse_id}_nested_extracted"
            nested_extract_dir.mkdir(exist_ok=True)

            # Extract any nested TAR.GZ files
            nested_archives = list(extract_dir.glob("*.tar.gz"))
            if nested_archives:
                logger.info(f"Found {len(nested_archives)} nested TAR.GZ files")

                all_matrices = []
                for archive_path in nested_archives:
                    try:
                        sample_id = archive_path.stem.split(".")[
                            0
                        ]  # Extract sample ID from filename
                        sample_extract_dir = nested_extract_dir / sample_id
                        sample_extract_dir.mkdir(exist_ok=True)

                        logger.info(f"Extracting nested archive: {archive_path.name}")
                        with tarfile.open(archive_path, "r:gz") as nested_tar:
                            nested_tar.extractall(path=sample_extract_dir)

                        # Try to parse 10X Genomics format data
                        matrix = self._parse_10x_data(sample_extract_dir, sample_id)
                        if matrix is not None:
                            all_matrices.append(matrix)
                            logger.info(
                                f"Successfully parsed 10X data for {sample_id}: {matrix.shape}"
                            )

                    except Exception as e:
                        logger.warning(
                            f"Failed to extract/parse {archive_path.name}: {e}"
                        )
                        continue

                if all_matrices:
                    # Concatenate all sample matrices
                    logger.info(f"Concatenating {len(all_matrices)} 10X matrices")
                    combined_matrix = pd.concat(all_matrices, axis=0, sort=False)
                    logger.info(f"Combined matrix shape: {combined_matrix.shape}")
                    return combined_matrix

            # Fallback: look for regular expression files
            expression_files = []
            for file_path in extract_dir.rglob("*"):
                if file_path.is_file() and any(
                    ext in file_path.name.lower()
                    for ext in [".txt", ".csv", ".tsv", ".gz"]
                ):
                    # Skip small files (likely barcodes or metadata)
                    if file_path.stat().st_size > 100000:  # > 100KB
                        expression_files.append(file_path)

            if expression_files:
                logger.info(f"Found {len(expression_files)} potential expression files")

                # Try to parse the largest file first (likely the main expression matrix)
                expression_files.sort(key=lambda x: x.stat().st_size, reverse=True)

                for file_path in expression_files[:3]:  # Try top 3 largest files
                    try:
                        logger.info(f"Attempting to parse: {file_path.name}")
                        matrix = self._parse_expression_file(file_path)
                        if (
                            matrix is not None
                            and matrix.shape[0] > 0
                            and matrix.shape[1] > 0
                        ):
                            logger.info(
                                f"Successfully parsed expression matrix: {matrix.shape}"
                            )
                            return matrix
                    except Exception as e:
                        logger.warning(f"Failed to parse {file_path.name}: {e}")
                        continue

            logger.warning(
                f"Could not parse any expression files from TAR for {gse_id}"
            )
            return None

        except Exception as e:
            logger.error(f"Error processing TAR file: {e}")
            return None

    def _download_and_parse_file(
        self, file_url: str, gse_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Download and parse a single expression file.

        Args:
            file_url: URL to expression file
            gse_id: GEO series ID

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            file_name = file_url.split("/")[-1]
            local_file = self.cache_dir / f"{gse_id}_{file_name}"

            if not local_file.exists():
                logger.info(f"Downloading file: {file_url}")
                urllib.request.urlretrieve(file_url, local_file)

            return self._parse_expression_file(local_file)

        except Exception as e:
            logger.error(f"Error downloading and parsing file: {e}")
            return None

    def _parse_10x_data(
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

            logger.info(
                f"Found 10X files - Matrix: {matrix_file.name}, Barcodes: {barcodes_file.name if barcodes_file else 'None'}, Features: {features_file.name if features_file else 'None'}"
            )

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
                import gzip

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

            logger.info(
                f"Successfully created 10X DataFrame for {sample_id}: {df.shape}"
            )
            return df

        except Exception as e:
            logger.error(f"Error parsing 10X data for {sample_id}: {e}")
            return None

    def _parse_expression_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Parse an expression data file.

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
                return self._parse_matrix_market_file(file_path)

            # Handle regular text files
            # Handle compressed files
            if file_path.name.endswith(".gz"):
                opener = lambda: pd.read_csv(
                    file_path,
                    compression="gzip",
                    sep="\t",
                    index_col=0,
                    low_memory=False,
                )
            else:
                opener = lambda: pd.read_csv(
                    file_path, sep="\t", index_col=0, low_memory=False
                )

            # Try tab-separated first
            try:
                df = opener()
                if df.shape[0] > 0 and df.shape[1] > 0:
                    logger.info(f"Successfully parsed as tab-separated: {df.shape}")
                    return df
            except:
                pass

            # Try comma-separated
            try:
                if file_path.name.endswith(".gz"):
                    df = pd.read_csv(
                        file_path,
                        compression="gzip",
                        sep=",",
                        index_col=0,
                        low_memory=False,
                    )
                else:
                    df = pd.read_csv(file_path, sep=",", index_col=0, low_memory=False)

                if df.shape[0] > 0 and df.shape[1] > 0:
                    logger.info(f"Successfully parsed as comma-separated: {df.shape}")
                    return df
            except:
                pass

            logger.warning(f"Could not parse expression file: {file_path}")
            return None

        except Exception as e:
            logger.error(f"Error parsing expression file {file_path}: {e}")
            return None

    def _parse_matrix_market_file(self, matrix_file: Path) -> Optional[pd.DataFrame]:
        """
        Parse a Matrix Market format file (.mtx or .mtx.gz) and look for associated barcodes/features.

        Args:
            matrix_file: Path to Matrix Market format file

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            logger.info(f"Parsing Matrix Market file: {matrix_file}")

            # Import scipy for sparse matrix handling
            try:
                import scipy.io as sio
                from scipy.sparse import csr_matrix
            except ImportError:
                logger.error(
                    "scipy is required for parsing Matrix Market files but not available"
                )
                return None

            # Read the sparse matrix
            if matrix_file.name.endswith(".gz"):
                import gzip

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

            logger.info(
                f"Found files - Barcodes: {barcodes_file.name if barcodes_file else 'None'}, Features: {features_file.name if features_file else 'None'}"
            )

            # Read barcodes (cell identifiers)
            cell_ids = []
            if barcodes_file and barcodes_file.exists():
                try:
                    if barcodes_file.name.endswith(".gz"):
                        import gzip

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
                        import gzip

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

            logger.info(
                f"Successfully created Matrix Market DataFrame for {sample_id}: {df.shape}"
            )
            return df

        except Exception as e:
            logger.error(f"Error parsing Matrix Market file {matrix_file}: {e}")
            return None

    def _extract_metadata(self, gse) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from GEOparse GSE object.

        Args:
            gse: GEOparse GSE object

        Returns:
            dict: Extracted metadata
        """
        try:
            metadata = {}

            # Basic metadata from GEOparse
            if hasattr(gse, "metadata"):
                for key, value in gse.metadata.items():
                    if isinstance(value, list) and len(value) == 1:
                        metadata[key] = value[0]
                    else:
                        metadata[key] = value

            # Platform information
            if hasattr(gse, "gpls"):
                platforms = {}
                for gpl_id, gpl in gse.gpls.items():
                    platforms[gpl_id] = {
                        "title": getattr(gpl, "metadata", {}).get("title", [""])[0]
                        if hasattr(gpl, "metadata")
                        else "",
                        "organism": getattr(gpl, "metadata", {}).get("organism", [""])[
                            0
                        ]
                        if hasattr(gpl, "metadata")
                        else "",
                        "technology": getattr(gpl, "metadata", {}).get(
                            "technology", [""]
                        )[0]
                        if hasattr(gpl, "metadata")
                        else "",
                    }
                metadata["platforms"] = platforms

            # Sample metadata
            if hasattr(gse, "gsms"):
                sample_metadata = {}
                for gsm_id, gsm in gse.gsms.items():
                    sample_meta = {}
                    if hasattr(gsm, "metadata"):
                        for key, value in gsm.metadata.items():
                            if isinstance(value, list) and len(value) == 1:
                                sample_meta[key] = value[0]
                            else:
                                sample_meta[key] = value
                    sample_metadata[gsm_id] = sample_meta
                metadata["samples"] = sample_metadata

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    def _validate_geo_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate GEO metadata against transcriptomics schema.
        
        Args:
            metadata: Extracted GEO metadata dictionary
            
        Returns:
            Dict containing validation results and schema alignment
        """
        try:
            from lobster.core.schemas.transcriptomics import TranscriptomicsSchema
            
            # Get the single-cell schema (covers most GEO datasets)
            schema = TranscriptomicsSchema.get_single_cell_schema()
            uns_schema = schema.get('uns', {}).get('optional', [])
            
            # Check which metadata fields align with our schema
            schema_aligned = {}
            schema_missing = []
            extra_fields = []
            
            # Check alignment for each field in metadata
            for field in metadata.keys():
                if field in uns_schema:
                    schema_aligned[field] = metadata[field]
                else:
                    extra_fields.append(field)
            
            # Check for schema fields not present in metadata
            for schema_field in uns_schema:
                if schema_field not in metadata:
                    schema_missing.append(schema_field)
            
            # Determine data type based on metadata
            data_type = self._determine_data_type_from_metadata(metadata)
            
            validation_result = {
                'schema_aligned_fields': len(schema_aligned),
                'schema_missing_fields': len(schema_missing),
                'extra_fields_count': len(extra_fields),
                'alignment_percentage': (len(schema_aligned) / len(uns_schema) * 100) if uns_schema else 0.0,
                'aligned_metadata': schema_aligned,
                'missing_fields': schema_missing[:10],  # Limit for display
                'extra_fields': extra_fields[:10],  # Limit for display
                'predicted_data_type': data_type,
                'validation_status': 'PASS' if len(schema_aligned) > len(schema_missing) else 'WARNING'
            }
            
            logger.info(f"Metadata validation: {validation_result['alignment_percentage']:.1f}% schema alignment")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating metadata: {e}")
            return {
                'validation_status': 'ERROR',
                'error_message': str(e),
                'schema_aligned_fields': 0,
                'alignment_percentage': 0.0
            }

    def _determine_data_type_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Determine likely data type (single-cell vs bulk) from metadata.
        
        Args:
            metadata: GEO metadata dictionary
            
        Returns:
            str: Predicted data type
        """
        try:
            # Check platform information
            platforms = metadata.get('platforms', {})
            platform_info = str(platforms).lower()
            
            # Check overall design
            overall_design = str(metadata.get('overall_design', '')).lower()
            
            # Check sample characteristics  
            samples = metadata.get('samples', {})
            sample_chars = []
            for sample in samples.values():
                chars = sample.get('characteristics_ch1', [])
                if isinstance(chars, list):
                    sample_chars.extend([str(c).lower() for c in chars])
            
            sample_text = ' '.join(sample_chars)
            
            # Keywords that suggest single-cell
            single_cell_keywords = [
                'single cell', 'single-cell', 'scrnaseq', 'scrna-seq', 
                '10x', 'chromium', 'droplet', 'microwell', 'smart-seq',
                'cell sorting', 'sorted cells', 'individual cells'
            ]
            
            # Keywords that suggest bulk
            bulk_keywords = [
                'bulk', 'tissue', 'whole', 'total rna', 'population'
            ]
            
            combined_text = f"{platform_info} {overall_design} {sample_text}"
            
            # Count keyword matches
            single_cell_score = sum(1 for keyword in single_cell_keywords if keyword in combined_text)
            bulk_score = sum(1 for keyword in bulk_keywords if keyword in combined_text)
            
            # Make prediction
            if single_cell_score > bulk_score:
                return 'single_cell_rna_seq'
            elif bulk_score > single_cell_score:
                return 'bulk_rna_seq'
            else:
                # Default to single-cell for GEO datasets (more common)
                return 'single_cell_rna_seq'
                
        except Exception as e:
            logger.warning(f"Error determining data type: {e}")
            return 'single_cell_rna_seq'  # Default

    def _format_metadata_summary(
        self, 
        geo_id: str, 
        metadata: Dict[str, Any], 
        validation_result: Dict[str, Any]
    ) -> str:
        """
        Format comprehensive metadata summary for user review.
        
        Args:
            geo_id: GEO accession ID
            metadata: Extracted metadata dictionary
            validation_result: Validation results
            
        Returns:
            str: Formatted metadata summary
        """
        try:
            # Extract key information
            title = metadata.get('title', 'N/A')
            summary = metadata.get('summary', 'N/A')[:500] + ('...' if len(str(metadata.get('summary', ''))) > 500 else '')
            overall_design = metadata.get('overall_design', 'N/A')[:300] + ('...' if len(str(metadata.get('overall_design', ''))) > 300 else '')
            
            # Sample information
            samples = metadata.get('samples', {})
            sample_count = len(samples)
            
            # Platform information
            platforms = metadata.get('platforms', {})
            platform_info = []
            for platform_id, platform_data in platforms.items():
                platform_info.append(f"{platform_id}: {platform_data.get('title', 'N/A')}")
            
            # Contact information
            contact_name = metadata.get('contact_name', 'N/A')
            contact_institute = metadata.get('contact_institute', 'N/A')
            
            # Publication info
            pubmed_id = metadata.get('pubmed_id', 'Not available')
            
            # Dates
            submission_date = metadata.get('submission_date', 'N/A')
            last_update = metadata.get('last_update_date', 'N/A')
            
            # Sample characteristics preview
            sample_preview = []
            for i, (sample_id, sample_data) in enumerate(samples.items()):
                if i < 3:  # Show first 3 samples
                    chars = sample_data.get('characteristics_ch1', [])
                    if isinstance(chars, list) and chars:
                        sample_preview.append(f"  - {sample_id}: {chars[0]}")
                    else:
                        sample_preview.append(f"  - {sample_id}: {sample_data.get('title', 'No title')}")
            
            if sample_count > 3:
                sample_preview.append(f"  ... and {sample_count - 3} more samples")
            
            # Validation status
            validation_status = validation_result.get('validation_status', 'UNKNOWN')
            alignment_pct = validation_result.get('alignment_percentage', 0.0)
            predicted_type = validation_result.get('predicted_data_type', 'unknown')
            
            # Format the summary
            summary_text = f"""📊 **GEO Dataset Metadata Summary: {geo_id}**

🔬 **Study Information:**
- **Title:** {title}
- **Summary:** {summary}
- **Design:** {overall_design}
- **Predicted Type:** {predicted_type.replace('_', ' ').title()}

👥 **Research Details:**
- **Contact:** {contact_name} ({contact_institute})
- **PubMed ID:** {pubmed_id}
- **Submission:** {submission_date}
- **Last Update:** {last_update}

🧪 **Platform Information:**
{chr(10).join(platform_info) if platform_info else '- No platform information available'}

🔢 **Sample Information ({sample_count} samples):**
{chr(10).join(sample_preview) if sample_preview else '- No sample information available'}

✅ **Schema Validation:**
- **Status:** {validation_status}
- **Schema Alignment:** {alignment_pct:.1f}% of expected fields present
- **Aligned Fields:** {validation_result.get('schema_aligned_fields', 0)}
- **Missing Fields:** {validation_result.get('schema_missing_fields', 0)}

📋 **Next Steps:**
1. **Review this metadata** to ensure it matches your research needs
2. **Confirm the predicted data type** is correct for your analysis
3. **Proceed to download** the full dataset if satisfied
4. **Use:** `download_geo_dataset('{geo_id}')` to download expression data

💡 **Note:** This metadata has been cached and validated against our transcriptomics schema. 
The actual expression data download will be much faster now that metadata is prepared."""

            return summary_text
            
        except Exception as e:
            logger.error(f"Error formatting metadata summary: {e}")
            return f"Error formatting metadata summary for {geo_id}: {str(e)}"

    def _get_sample_info(self, gse) -> Dict[str, Dict[str, Any]]:
        """
        Get sample information for downloading individual matrices.

        Args:
            gse: GEOparse GSE object

        Returns:
            dict: Sample information dictionary
        """
        sample_info = {}

        try:
            if hasattr(gse, "gsms"):
                for gsm_id, gsm in gse.gsms.items():
                    sample_info[gsm_id] = {
                        "title": getattr(gsm, "metadata", {}).get("title", [""])[0]
                        if hasattr(gsm, "metadata")
                        else "",
                        "platform": getattr(gsm, "metadata", {}).get(
                            "platform_id", [""]
                        )[0]
                        if hasattr(gsm, "metadata")
                        else "",
                        "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm_id}",
                        "download_url": f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm_id[:6]}nnn/{gsm_id}/suppl/",
                    }

            logger.info(f"Collected information for {len(sample_info)} samples")
            return sample_info

        except Exception as e:
            logger.error(f"Error getting sample info: {e}")
            return {}

    def _download_sample_matrices(
        self, sample_info: Dict[str, Dict[str, Any]], gse_id: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Download individual sample expression matrices.

        Args:
            sample_info: Dictionary of sample information
            gse_id: GEO series ID

        Returns:
            dict: Dictionary of sample matrices
        """
        sample_matrices = {}

        logger.info(f"Downloading matrices for {len(sample_info)} samples...")

        # Use threading for parallel downloads
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_sample = {
                executor.submit(
                    self._download_single_sample, gsm_id, info, gse_id
                ): gsm_id
                for gsm_id, info in sample_info.items()
            }

            for future in as_completed(future_to_sample):
                gsm_id = future_to_sample[future]
                try:
                    matrix = future.result()
                    sample_matrices[gsm_id] = matrix
                    if matrix is not None:
                        logger.info(
                            f"Successfully downloaded matrix for {gsm_id}: {matrix.shape}"
                        )
                    else:
                        logger.warning(f"No matrix data found for {gsm_id}")
                except Exception as e:
                    logger.error(f"Error downloading {gsm_id}: {e}")
                    sample_matrices[gsm_id] = None

        return sample_matrices

    def _download_single_sample(
        self, gsm_id: str, sample_info: Dict[str, Any], gse_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Download a single sample matrix.

        Args:
            gsm_id: GEO sample ID
            sample_info: Sample information
            gse_id: GEO series ID

        Returns:
            DataFrame: Sample expression matrix or None
        """
        try:
            # Try to get sample using GEOparse
            gsm = GEOparse.get_GEO(geo=gsm_id, destdir=str(self.cache_dir))

            # Check if the sample has expression data table
            if hasattr(gsm, "table") and gsm.table is not None:
                logger.info(f"Found expression table for {gsm_id}")
                return gsm.table

            # Try supplementary files
            if hasattr(gsm, "metadata") and "supplementary_file" in gsm.metadata:
                suppl_files = gsm.metadata["supplementary_file"]
                if isinstance(suppl_files, list):
                    suppl_files = suppl_files
                else:
                    suppl_files = [suppl_files]

                for suppl_url in suppl_files:
                    if any(
                        ext in suppl_url.lower()
                        for ext in [".txt", ".csv", ".tsv", ".gz"]
                    ):
                        try:
                            matrix = self._download_supplementary_file(
                                suppl_url, gsm_id
                            )
                            if matrix is not None:
                                return matrix
                        except Exception as e:
                            logger.warning(
                                f"Failed to download supplementary file {suppl_url}: {e}"
                            )
                            continue

            logger.warning(f"No expression data found for {gsm_id}")
            return None

        except Exception as e:
            logger.error(f"Error downloading sample {gsm_id}: {e}")
            return None

    def _download_supplementary_file(
        self, url: str, gsm_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Download and parse a supplementary file.

        Args:
            url: URL to supplementary file
            gsm_id: GEO sample ID

        Returns:
            DataFrame: Parsed matrix or None
        """
        try:
            local_file = self.cache_dir / f"{gsm_id}_suppl.txt"

            # Download file
            urllib.request.urlretrieve(url, local_file)

            # Try to parse as various formats
            for sep in ["\t", ","]:
                try:
                    if local_file.suffix == ".gz":
                        df = pd.read_csv(
                            local_file, sep=sep, compression="gzip", index_col=0
                        )
                    else:
                        df = pd.read_csv(local_file, sep=sep, index_col=0)

                    if df.shape[0] > 0 and df.shape[1] > 0:
                        logger.info(
                            f"Successfully parsed supplementary file for {gsm_id}: {df.shape}"
                        )
                        return df
                except:
                    continue

            return None

        except Exception as e:
            logger.error(f"Error downloading supplementary file: {e}")
            return None

    def _validate_matrices(
        self, sample_matrices: Dict[str, Optional[pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Validate downloaded matrices and filter out invalid ones.

        Args:
            sample_matrices: Dictionary of sample matrices

        Returns:
            dict: Dictionary of validated matrices
        """
        validated = {}

        for gsm_id, matrix in sample_matrices.items():
            if matrix is None:
                logger.warning(f"Skipping {gsm_id}: No matrix data")
                continue

            # Validate matrix format
            if not self._is_valid_expression_matrix(matrix):
                logger.warning(f"Skipping {gsm_id}: Invalid matrix format")
                continue

            # Check matrix dimensions
            if matrix.shape[0] < 10 or matrix.shape[1] < 10:
                logger.warning(f"Skipping {gsm_id}: Matrix too small ({matrix.shape})")
                continue

            validated[gsm_id] = matrix
            logger.info(f"Validated {gsm_id}: {matrix.shape}")

        logger.info(f"Validated {len(validated)}/{len(sample_matrices)} matrices")
        return validated

    def _is_valid_expression_matrix(self, matrix: pd.DataFrame) -> bool:
        """
        Check if a matrix is a valid expression matrix.

        Args:
            matrix: DataFrame to validate

        Returns:
            bool: True if valid expression matrix
        """
        try:
            # Check if it's a DataFrame
            if not isinstance(matrix, pd.DataFrame):
                return False

            # Check for numeric data
            numeric_cols = matrix.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return False

            # Check for non-negative values (typical for expression data)
            if (matrix[numeric_cols] < 0).any().any():
                logger.warning("Matrix contains negative values")

            # Check for reasonable value ranges
            max_val = matrix[numeric_cols].max().max()
            if max_val > 1e6:
                logger.info("Matrix contains very large values (possibly raw counts)")

            return True

        except Exception as e:
            logger.error(f"Error validating matrix: {e}")
            return False

    def _concatenate_matrices(
        self, validated_matrices: Dict[str, pd.DataFrame], gse_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Concatenate validated matrices into a single expression matrix.

        Args:
            validated_matrices: Dictionary of validated matrices
            gse_id: GEO series ID

        Returns:
            DataFrame: Combined expression matrix or None
        """
        try:
            if not validated_matrices:
                logger.error("No validated matrices to concatenate")
                return None

            logger.info(
                f"Concatenating {len(validated_matrices)} matrices for {gse_id}"
            )

            # Check if all matrices have the same genes (columns)
            all_genes = set()
            for matrix in validated_matrices.values():
                all_genes.update(matrix.columns)

            common_genes = set(next(iter(validated_matrices.values())).columns)
            for matrix in validated_matrices.values():
                common_genes = common_genes.intersection(set(matrix.columns))

            logger.info(f"Found {len(common_genes)} common genes across all matrices")

            if len(common_genes) == 0:
                logger.error("No common genes found across matrices")
                return None

            # Align matrices to common genes
            aligned_matrices = []
            for gsm_id, matrix in validated_matrices.items():
                # Select common genes and add sample prefix to cell/row names
                aligned_matrix = matrix[list(common_genes)]
                aligned_matrix.index = [
                    f"{gsm_id}_{idx}" for idx in aligned_matrix.index
                ]
                aligned_matrices.append(aligned_matrix)

            # Concatenate matrices
            combined_matrix = pd.concat(aligned_matrices, axis=0, sort=False)

            logger.info(f"Successfully concatenated matrices: {combined_matrix.shape}")
            return combined_matrix

        except Exception as e:
            logger.error(f"Error concatenating matrices: {e}")
            return None

    def _check_workspace_for_processed_data(
        self, gse_id: str
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Check if processed data for a GSE ID already exists in workspace.
        Uses professional naming patterns with backward compatibility.

        Args:
            gse_id: GEO series ID

        Returns:
            Tuple of (DataFrame, metadata) if found, None otherwise
        """
        try:
            from lobster.utils.file_naming import BioinformaticsFileNaming
            
            # First, try professional naming pattern
            data_dir = self.data_manager.data_dir
            existing_file = BioinformaticsFileNaming.find_latest_file(
                directory=data_dir,
                data_source='GEO',
                dataset_id=gse_id,
                processing_step='raw_matrix'
            )
            
            if existing_file and existing_file.exists():
                logger.info(f"Found existing file with professional naming: {existing_file.name}")
                
                # Load the data
                combined_matrix = pd.read_csv(existing_file, index_col=0)
                
                # Load associated metadata
                metadata_file = existing_file.parent / BioinformaticsFileNaming.generate_metadata_filename(existing_file.name)
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                logger.info(f"Successfully loaded existing data: {combined_matrix.shape}")
                return combined_matrix, metadata
            
            # Backward compatibility: check for legacy naming patterns
            data_files = self.data_manager.list_workspace_files()["data"]
            for file_info in data_files:
                if f"{gse_id}_processed" in file_info["name"] and file_info["name"].endswith(".csv"):
                    logger.info(f"Found existing legacy file: {file_info['name']}")

                    # Load the data
                    data_path = Path(file_info["path"])
                    combined_matrix = pd.read_csv(data_path, index_col=0)

                    # Load associated metadata
                    metadata_path = data_path.parent / f"{data_path.stem}_metadata.json"
                    metadata = {}
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                    logger.info(f"Successfully loaded legacy data: {combined_matrix.shape}")
                    return combined_matrix, metadata

            return None

        except Exception as e:
            logger.warning(f"Error checking workspace for existing data: {e}")
            return None


    def _format_download_response(
        self,
        gse_id: str,
        combined_matrix: pd.DataFrame,
        metadata: Dict[str, Any],
        sample_count: int,
        saved_file: str = None,
    ) -> str:
        """
        Format download response message.

        Args:
            gse_id: GEO series ID
            combined_matrix: Combined expression matrix
            metadata: Metadata dictionary
            sample_count: Number of samples processed
            saved_file: Path to saved workspace file

        Returns:
            str: Formatted response message
        """
        study_title = (
            metadata.get("title", ["N/A"])[0]
            if isinstance(metadata.get("title"), list)
            else metadata.get("title", "N/A")
        )
        organism = (
            metadata.get("organism", ["N/A"])[0]
            if isinstance(metadata.get("organism"), list)
            else metadata.get("organism", "N/A")
        )

        workspace_info = ""
        if saved_file and "Error" not in saved_file:
            workspace_info = f"\n💾 Saved to workspace: {Path(saved_file).name}\n⚡ Future loads will be instant (no re-downloading needed)"

        return f"""Successfully downloaded and processed {gse_id} using GEOparse!

📊 Combined expression matrix: {combined_matrix.shape[0]} cells × {combined_matrix.shape[1]} genes
📋 Study: {study_title}
🧬 Organism: {organism}
🔬 Successfully processed {sample_count} samples{workspace_info}
📈 Ready for downstream analysis (quality control, clustering, ML)

The data has been professionally concatenated and is ready for:
- Quality assessment and filtering
- Clustering and cell type annotation
- Machine learning model preparation
- Differential expression analysis"""
