"""
Professional GEO data service using GEOparse.

This service provides a unified interface for downloading and processing
data from the Gene Expression Omnibus (GEO) database using GEOparse library.
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

from ..core.data_manager import DataManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GEOService:
    """
    Professional service for accessing and processing GEO data using GEOparse.

    This class provides a high-level interface for working with GEO data,
    handling the downloading, parsing, and processing of datasets using GEOparse.
    """

    def __init__(
        self, data_manager: DataManager, cache_dir: Optional[str] = None, console=None
    ):
        """
        Initialize the GEO service.

        Args:
            data_manager: DataManager instance for storing processed data
            cache_dir: Directory to cache downloaded files
            console: Rich console instance for display (creates new if None)
        """
        if GEOparse is None:
            raise ImportError(
                "GEOparse is required but not installed. Please install with: pip install GEOparse"
            )

        self.data_manager = data_manager
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".geo_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.console = console

        logger.info("GEOService initialized with GEOparse backend")

    def download_dataset(self, query: str) -> str:
        """
        Download and process a dataset from GEO using GEOparse.

        Args:
            query: Query string containing a GEO accession number

        Returns:
            str: Status message with detailed information
        """
        try:
            logger.info(f"Processing GEO query: {query}")

            gse_id = self._extract_gse_id(query)
            if not gse_id:
                return "Please provide a valid GSE accession number (e.g., GSE109564)"

            logger.info(f"Identified GSE ID: {gse_id}")

            # Check if processed matrix already exists in workspace
            existing_data = self._check_workspace_for_processed_data(gse_id)
            if existing_data is not None:
                logger.info(f"Found existing processed data for {gse_id} in workspace")
                combined_matrix, saved_metadata = existing_data

                # Load into data manager
                self.data_manager.set_data(
                    data=combined_matrix, metadata=saved_metadata
                )

                # Log reuse of existing data
                self.data_manager.log_tool_usage(
                    tool_name="load_existing_geo_dataset",
                    parameters={"geo_id": gse_id},
                    description=f"Loaded existing processed GEO dataset {gse_id} from workspace",
                )

                return f"""Successfully loaded existing processed data for {gse_id}!

📊 Combined expression matrix: {combined_matrix.shape[0]} cells × {combined_matrix.shape[1]} genes
💾 Loaded from workspace (no re-downloading needed)
🕒 Originally processed: {saved_metadata.get('processing_date', 'Unknown')}
🔬 Samples: {saved_metadata.get('n_samples', 'Unknown')}
⚡ Ready for immediate analysis!"""

            # Step 1: Download SOFT data using GEOparse
            logger.info(f"Downloading SOFT data for {gse_id} using GEOparse...")
            gse = GEOparse.get_GEO(geo=gse_id, destdir=str(self.cache_dir))

            # Step 2: Extract metadata from SOFT
            metadata = self._extract_metadata(gse)
            logger.info(f"Extracted metadata for {gse_id}")

            # Step 3: Check for supplementary files (TAR, etc.) first
            supplementary_data = self._process_supplementary_files(gse, gse_id)

            if supplementary_data is not None and not supplementary_data.empty:
                logger.info(f"Successfully processed supplementary files for {gse_id}")
                combined_matrix = supplementary_data
                sample_count = len(gse.gsms) if hasattr(gse, "gsms") else 0
            else:
                # Step 4: Fallback to individual sample matrices
                sample_info = self._get_sample_info(gse)
                logger.info(f"Found {len(sample_info)} samples in {gse_id}")

                sample_matrices = self._download_sample_matrices(sample_info, gse_id)
                validated_matrices = self._validate_matrices(sample_matrices)

                if not validated_matrices:
                    return f"No valid expression matrices found for {gse_id}. The dataset may not contain downloadable expression data matrices."

                combined_matrix = self._concatenate_matrices(validated_matrices, gse_id)
                sample_count = len(validated_matrices)

            if combined_matrix is None:
                return f"Failed to process expression data for {gse_id}"

            # Step 5: Store in data manager and save to workspace
            enhanced_metadata = {
                "source": gse_id,
                "geo_object": gse,
                "n_samples": sample_count,
                "n_cells": combined_matrix.shape[0],
                "n_genes": combined_matrix.shape[1],
                "data_source": "supplementary_files"
                if supplementary_data is not None
                else "individual_samples",
                "processing_date": pd.Timestamp.now().isoformat(),
                "geoparse_version": "GEOparse-based",
                **metadata,
            }

            self.data_manager.set_data(data=combined_matrix, metadata=enhanced_metadata)

            # Auto-save the processed matrix to workspace for future use
            saved_file = self._save_processed_matrix_to_workspace(
                gse_id, combined_matrix, enhanced_metadata
            )

            # Log successful download and save
            self.data_manager.log_tool_usage(
                tool_name="download_geo_dataset_geoparse",
                parameters={"geo_id": gse_id},
                description=f"Downloaded and processed GEO dataset {gse_id} using GEOparse, saved to {saved_file}",
            )

            # Auto-save current state
            self.data_manager.auto_save_state()

            return self._format_download_response(
                gse_id, combined_matrix, metadata, sample_count, saved_file
            )

        except Exception as e:
            logger.exception(f"Error downloading dataset: {e}")
            return f"Error downloading dataset: {str(e)}"

    def _extract_gse_id(self, query: str) -> Optional[str]:
        """Extract GSE ID from query string."""
        match = re.search(r"GSE\d+", query.upper())
        return match.group(0) if match else None

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

        Args:
            gse_id: GEO series ID

        Returns:
            Tuple of (DataFrame, metadata) if found, None otherwise
        """
        try:
            # Look for processed data files in workspace
            data_files = self.data_manager.list_workspace_files()["data"]

            # Look for files matching the GSE ID pattern
            for file_info in data_files:
                if f"{gse_id}_processed" in file_info["name"] and file_info[
                    "name"
                ].endswith(".csv"):
                    logger.info(f"Found existing processed file: {file_info['name']}")

                    # Load the data
                    data_path = Path(file_info["path"])
                    combined_matrix = pd.read_csv(data_path, index_col=0)

                    # Load associated metadata
                    metadata_path = data_path.parent / f"{data_path.stem}_metadata.json"
                    metadata = {}
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                    logger.info(
                        f"Successfully loaded existing data: {combined_matrix.shape}"
                    )
                    return combined_matrix, metadata

            return None

        except Exception as e:
            logger.warning(f"Error checking workspace for existing data: {e}")
            return None

    def _save_processed_matrix_to_workspace(
        self, gse_id: str, combined_matrix: pd.DataFrame, metadata: Dict[str, Any]
    ) -> str:
        """
        Save processed matrix and metadata to workspace for future reuse.

        Args:
            gse_id: GEO series ID
            combined_matrix: Processed expression matrix
            metadata: Associated metadata

        Returns:
            str: Path to saved file
        """
        try:
            # Create filename with timestamp
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{gse_id}_processed_{timestamp}.csv"

            # Save the matrix
            data_path = self.data_manager.data_dir / filename
            combined_matrix.to_csv(data_path)

            # Save metadata
            metadata_path = (
                self.data_manager.data_dir
                / f"{gse_id}_processed_{timestamp}_metadata.json"
            )

            # Clean metadata for JSON serialization
            clean_metadata = {}
            for key, value in metadata.items():
                if key == "geo_object":
                    # Skip the GEOparse object as it's not JSON serializable
                    continue
                try:
                    json.dumps(value)  # Test if it's JSON serializable
                    clean_metadata[key] = value
                except (TypeError, ValueError):
                    clean_metadata[key] = str(value)

            with open(metadata_path, "w") as f:
                json.dump(clean_metadata, f, indent=2, default=str)

            logger.info(f"Saved processed matrix to workspace: {data_path}")
            logger.info(f"Saved metadata to workspace: {metadata_path}")

            return str(data_path)

        except Exception as e:
            logger.error(f"Error saving processed matrix to workspace: {e}")
            return "Error saving to workspace"

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
