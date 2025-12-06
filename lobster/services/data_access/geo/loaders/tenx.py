"""
10X Genomics adaptive loader for GEO datasets.

Handles loading of 10X Chromium single-cell RNA-seq series-level trio files
with adaptive format detection for non-standard GEO submissions.

Key Features:
- Detects single-column vs multi-column features/genes files
- Uses scanpy for standard 10X formats (fast, well-tested)
- Falls back to scipy manual loading for non-standard formats (robust)
- Handles GSE182227-style single-column genes.txt.gz files

Created: 2025-11-27
Purpose: Fix Bug #2 - Single-cell MTX parsing failures with non-standard feature files
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import anndata
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


class TenXGenomicsLoader:
    """
    Adaptive loader for 10X Genomics series-level trio files.

    Handles both standard and non-standard feature file formats:
    - Standard: 2+ column features files (gene_id, gene_name, feature_type)
    - Non-standard: 1-column genes files (symbols or IDs only)

    Uses format detection to choose the optimal loading strategy.
    """

    def __init__(self, geo_downloader, cache_dir: Path = None):
        """
        Initialize 10X loader with workspace-aware caching.

        Args:
            geo_downloader: GEODownloadManager instance for file downloads
            cache_dir: Cache directory for extracted archives (REQUIRED)

        Raises:
            ValueError: If cache_dir is not provided
        """
        if cache_dir is None:
            raise ValueError(
                "cache_dir is required. Pass workspace-relative path from DataManagerV2. "
                "Example: TenXGenomicsLoader(downloader, cache_dir=data_manager.cache_dir / 'extracted_archives')"
            )
        self.geo_downloader = geo_downloader
        self.cache_dir = Path(cache_dir)

        logger.debug("TenXGenomicsLoader initialized")

    def detect_features_format(self, features_path: Path) -> str:
        """
        Detect 10X features file format by inspecting column count.

        Args:
            features_path: Path to features/genes file (compressed or uncompressed)

        Returns:
            Format type: "standard_10x", "symbols_only", or "ids_only"
        """
        import gzip

        try:
            # Handle compressed and uncompressed files
            if features_path.name.endswith(".gz"):
                with gzip.open(features_path, "rt") as f:
                    first_line = f.readline().strip()
            else:
                with open(features_path, "r") as f:
                    first_line = f.readline().strip()

            # Count tabs (columns = tabs + 1)
            n_cols = first_line.count("\t") + 1

            if n_cols >= 2:
                logger.debug(
                    f"Features file has {n_cols} columns - standard 10X format"
                )
                return "standard_10x"
            elif n_cols == 1:
                # Single column - determine if symbols or IDs
                # Symbols typically have dashes, letters, mixed case
                # IDs are typically pure numeric or ENSG* format
                if first_line.startswith("ENSG") or first_line.startswith("ENS"):
                    logger.debug("Features file has 1 column - Ensembl IDs format")
                    return "ids_only"
                else:
                    logger.debug("Features file has 1 column - gene symbols format")
                    return "symbols_only"
            else:
                logger.warning(f"Unexpected features file format: {n_cols} columns")
                return "symbols_only"  # Conservative fallback

        except Exception as e:
            logger.error(f"Error detecting features format: {e}")
            return "symbols_only"  # Safe fallback

    def load_10x_manual(
        self, temp_dir: Path, features_format: str, gse_id: str
    ) -> anndata.AnnData:
        """
        Manually load 10X MTX when features file is non-standard.

        Handles single-column features files that scanpy's read_10x_mtx() cannot process.
        Uses scipy.sparse directly for robust loading.

        Args:
            temp_dir: Directory containing matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz
            features_format: Format detected by detect_features_format()
            gse_id: GEO series ID for metadata

        Returns:
            AnnData object with properly loaded 10X data
        """
        import gzip

        from scipy.io import mmread

        logger.info(
            f"{gse_id}: Using manual 10X loader for non-standard format: {features_format}"
        )

        # Find files (handle various naming conventions)
        matrix_files = list(temp_dir.glob("*matrix*.mtx*")) + list(
            temp_dir.glob("*.mtx*")
        )
        barcodes_files = list(temp_dir.glob("*barcode*")) + list(
            temp_dir.glob("barcodes.*")
        )
        features_files = list(temp_dir.glob("*features*")) + list(
            temp_dir.glob("*genes*")
        )

        if not (matrix_files and barcodes_files and features_files):
            raise FileNotFoundError(
                f"Could not find complete 10X trio in {temp_dir}. "
                f"Matrix: {len(matrix_files)}, Barcodes: {len(barcodes_files)}, Features: {len(features_files)}"
            )

        matrix_path = matrix_files[0]
        barcodes_path = barcodes_files[0]
        features_path = features_files[0]

        # Load matrix (scipy handles MTX format natively)
        logger.debug(f"Loading matrix from {matrix_path.name}")
        if matrix_path.name.endswith(".gz"):
            with gzip.open(matrix_path, "rb") as f:
                X = mmread(
                    f
                ).T.tocsr()  # Transpose: MTX is genes × cells, we need cells × genes
        else:
            X = mmread(matrix_path).T.tocsr()

        # Load barcodes (cell IDs)
        logger.debug(f"Loading barcodes from {barcodes_path.name}")
        if barcodes_path.name.endswith(".gz"):
            with gzip.open(barcodes_path, "rt") as f:
                barcodes = [line.strip().split("\t")[0] for line in f if line.strip()]
        else:
            with open(barcodes_path, "r") as f:
                barcodes = [line.strip().split("\t")[0] for line in f if line.strip()]

        # Load features (handle 1 or 2+ columns adaptively)
        logger.debug(
            f"Loading features from {features_path.name} (format: {features_format})"
        )
        if features_path.name.endswith(".gz"):
            with gzip.open(features_path, "rt") as f:
                lines = [line.strip() for line in f if line.strip()]
        else:
            with open(features_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

        # Parse features based on format
        if features_format in ["symbols_only", "ids_only"]:
            # Single-column format: use same value for both ID and name
            gene_ids = lines
            gene_names = lines
            logger.info(f"Loaded {len(gene_names)} genes from 1-column features file")
        else:  # standard_10x
            # Multi-column format: parse normally
            gene_ids = [line.split("\t")[0] for line in lines]
            gene_names = [
                line.split("\t")[1] if "\t" in line else line.split("\t")[0]
                for line in lines
            ]
            logger.info(
                f"Loaded {len(gene_names)} genes from standard 10X features file"
            )

        # Validate dimensions match
        if X.shape[0] != len(barcodes):
            raise ValueError(
                f"Dimension mismatch: Matrix has {X.shape[0]} rows but {len(barcodes)} barcodes"
            )
        if X.shape[1] != len(gene_names):
            raise ValueError(
                f"Dimension mismatch: Matrix has {X.shape[1]} columns but {len(gene_names)} genes"
            )

        # Build var DataFrame
        var_df = pd.DataFrame(
            {
                "gene_ids": gene_ids,
                "feature_types": "Gene Expression",  # Default for 10X RNA
            },
            index=gene_names,
        )

        # Build obs DataFrame
        obs_df = pd.DataFrame(index=barcodes)

        # Create AnnData
        adata = anndata.AnnData(X=X, obs=obs_df, var=var_df)

        # Ensure unique names
        adata.var_names_make_unique()
        adata.obs_names_make_unique()

        logger.info(
            f"{gse_id}: Manual loader successfully created AnnData: "
            f"{adata.n_obs} cells × {adata.n_vars} genes"
        )

        return adata

    def try_series_level_10x_trio(
        self, suppl_files: List[str], gse_id: str
    ) -> Optional[anndata.AnnData]:
        """
        Check for and process series-level 10x trio files with adaptive loading.

        Many single-cell datasets provide the combined matrix at series level:
        - GSE*_matrix.mtx.gz (or GSE*_*_matrix.mtx.gz)
        - GSE*_barcodes.tsv.gz (or GSE*_*_barcodes.tsv.gz)
        - GSE*_features.tsv.gz (or GSE*_*_features.tsv.gz, or GSE*_genes.tsv.gz)

        Uses adaptive loading:
        - Standard 10X format (2+ columns): Uses scanpy (fast)
        - Non-standard format (1 column): Uses manual scipy loader (robust)

        Args:
            suppl_files: List of supplementary file URLs
            gse_id: GEO series ID

        Returns:
            AnnData: Loaded 10x data or None if not found/failed
        """
        try:
            # Pattern matching for series-level 10x files
            # FIXED: Use extension-based matching instead of requiring "matrix.mtx" substring
            # This handles non-standard naming like "GSE182227_OPSCC.mtx.gz"
            matrix_files = [
                f
                for f in suppl_files
                if f.lower().endswith((".mtx", ".mtx.gz", ".mtx.bz2"))
                and gse_id.lower() in f.lower()
            ]
            barcodes_files = [
                f
                for f in suppl_files
                if f.lower().endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz"))
                and "barcode" in f.lower()
                and gse_id.lower() in f.lower()
            ]
            # Features can be named "features" or "genes"
            features_files = [
                f
                for f in suppl_files
                if f.lower().endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz"))
                and ("features" in f.lower() or "genes" in f.lower())
                and gse_id.lower() in f.lower()
            ]

            # Check if we have the complete trio
            if not (matrix_files and barcodes_files and features_files):
                logger.debug(
                    f"{gse_id}: No series-level 10x trio found. "
                    f"Matrix: {len(matrix_files)}, Barcodes: {len(barcodes_files)}, Features: {len(features_files)}"
                )
                return None

            logger.info(
                f"{gse_id}: Found series-level 10x trio files. "
                f"Matrix: {matrix_files[0].split('/')[-1]}, "
                f"Barcodes: {barcodes_files[0].split('/')[-1]}, "
                f"Features: {features_files[0].split('/')[-1]}"
            )

            # Create a temporary directory for 10x files (scanpy expects specific structure)
            temp_dir = Path(tempfile.mkdtemp(prefix=f"{gse_id}_10x_"))
            logger.debug(f"Created temp directory for 10x loading: {temp_dir}")

            try:
                # Download the three files with expected names
                file_mapping = {
                    "matrix.mtx.gz": matrix_files[0],
                    "barcodes.tsv.gz": barcodes_files[0],
                    "features.tsv.gz": features_files[0],
                }

                for target_name, url in file_mapping.items():
                    local_path = temp_dir / target_name

                    # Convert FTP to HTTPS
                    if url.startswith("ftp://"):
                        url = url.replace("ftp://", "https://", 1)

                    logger.debug(f"Downloading {target_name} from {url}")
                    if not self.geo_downloader.download_file(url, local_path):
                        logger.error(f"Failed to download {target_name}")
                        return None

                    # Handle uncompressed versions if download name doesn't match
                    source_name = url.split("/")[-1]
                    if not source_name.endswith(".gz") and target_name.endswith(".gz"):
                        # File was downloaded without .gz, need to handle
                        actual_path = temp_dir / source_name
                        if actual_path.exists() and not local_path.exists():
                            # Rename to expected name without .gz
                            target_uncompressed = temp_dir / target_name.replace(
                                ".gz", ""
                            )
                            shutil.move(str(actual_path), str(target_uncompressed))
                            local_path = target_uncompressed

                logger.debug(f"All 10x files downloaded to {temp_dir}")

                # ADAPTIVE LOADING: Detect features format first, then choose appropriate loader
                # Find the features file in temp directory
                features_paths = list(temp_dir.glob("*features*")) + list(
                    temp_dir.glob("*genes*")
                )
                if not features_paths:
                    logger.error(f"{gse_id}: Features file not found in {temp_dir}")
                    return None

                features_path = features_paths[0]
                features_format = self.detect_features_format(features_path)

                # Choose loader based on format
                if features_format == "standard_10x":
                    # Use scanpy for standard format (fast, well-tested)
                    logger.debug(
                        f"{gse_id}: Using scanpy loader for standard 10X format"
                    )
                    try:
                        adata = sc.read_10x_mtx(
                            temp_dir,
                            var_names="gene_symbols",  # Use gene symbols if available
                            cache=False,  # Don't cache since we're using temp dir
                        )
                    except Exception as e:
                        # Try with gene_ids if gene_symbols fails
                        logger.warning(
                            f"Failed with gene_symbols, trying gene_ids: {e}"
                        )
                        adata = sc.read_10x_mtx(
                            temp_dir,
                            var_names="gene_ids",
                            cache=False,
                        )
                    adata.var_names_make_unique()
                else:
                    # Use manual loader for non-standard formats (robust fallback)
                    logger.info(
                        f"{gse_id}: Non-standard format detected, using manual loader"
                    )
                    adata = self.load_10x_manual(temp_dir, features_format, gse_id)
                    # Manual loader already makes names unique

                logger.info(
                    f"{gse_id}: Successfully loaded series-level 10x data: "
                    f"{adata.n_obs} cells × {adata.n_vars} genes (format: {features_format})"
                )

                # Add metadata
                adata.uns["source"] = "series_level_10x"
                adata.uns["geo_id"] = gse_id
                adata.uns["features_format"] = features_format  # Track for provenance

                return adata

            finally:
                # Clean up temp directory
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")

        except Exception as e:
            logger.error(f"Error processing series-level 10x trio for {gse_id}: {e}")
            return None
