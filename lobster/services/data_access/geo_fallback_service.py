"""
GEO Fallback Service - Contains fallback methods used only when primary GEOparse download fails.

This service provides specialized fallback functionality for downloading and processing
data from the Gene Expression Omnibus (GEO) database when the standard GEOparse approach fails.
It includes TAR processing, supplementary file handling, and alternative download strategies.
"""

import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    import GEOparse
except ImportError:
    GEOparse = None


# Import helper modules for fallback functionality

# Import the main service classes and enums
from lobster.services.data_access.geo_service import GEODataSource, GEOResult
from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context, handle_ssl_error

logger = get_logger(__name__)


class GEOFallbackService:
    """
    Fallback service for GEO data downloading when primary methods fail.

    This service is only instantiated and used when the standard _try_geoparse_download
    method fails, providing alternative download strategies and file processing methods.
    """

    def __init__(self, geo_service):
        """
        Initialize the fallback service with references to the main GEO service.

        Args:
            geo_service: Main GEOService instance for accessing shared resources
        """
        self.geo_service = geo_service
        self.data_manager = geo_service.data_manager
        self.cache_dir = geo_service.cache_dir
        self.console = geo_service.console
        self.geo_downloader = geo_service.geo_downloader
        self.geo_parser = geo_service.geo_parser

    # ========================================
    # PIPELINE STEP FUNCTIONS (FALLBACK)
    # ========================================

    def try_supplementary_tar(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Pipeline step: Try TAR supplementary files."""
        try:
            logger.debug(f"Trying TAR supplementary files for {geo_id}")
            soft_file, data_sources = self.geo_downloader.download_geo_data(geo_id)

            if isinstance(data_sources, dict) and "tar" in data_sources:
                matrix = self.geo_parser.parse_supplementary_file(data_sources["tar"])
                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.TAR_ARCHIVE,
                        processing_info={
                            "method": "helper_tar",
                            "source_file": str(data_sources["tar"]),
                        },
                        success=True,
                    )

            return GEOResult(
                success=False, error_message="No TAR files found or parsable"
            )

        except Exception as e:
            logger.warning(f"TAR supplementary download failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    def try_series_matrix(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Pipeline step: Try series matrix files (for bulk data)."""
        try:
            logger.debug(f"Trying series matrix for {geo_id}")

            # PRE-DOWNLOAD SOFT FILE USING HTTPS TO BYPASS GEOparse's FTP DOWNLOADER
            # GEOparse internally uses FTP which lacks error detection and causes corruption.
            # By pre-downloading with HTTPS, GEOparse finds existing file and skips its FTP download.
            soft_file_path = Path(self.cache_dir) / f"{geo_id}_family.soft.gz"
            if not soft_file_path.exists():
                # Construct SOFT file URL components
                gse_num_str = geo_id[3:]  # Remove 'GSE' prefix
                if len(gse_num_str) >= 3:
                    series_folder = f"GSE{gse_num_str[:-3]}nnn"
                else:
                    series_folder = "GSEnnn"

                soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series_folder}/{geo_id}/soft/{geo_id}_family.soft.gz"

                logger.debug(f"Pre-downloading SOFT file using HTTPS: {soft_url}")
                try:
                    ssl_context = create_ssl_context()
                    with urllib.request.urlopen(
                        soft_url, context=ssl_context
                    ) as response:
                        with open(soft_file_path, "wb") as f:
                            f.write(response.read())
                    logger.debug(
                        f"Successfully pre-downloaded SOFT file to {soft_file_path}"
                    )
                except Exception as e:
                    error_str = str(e)
                    if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
                        handle_ssl_error(e, soft_url, logger)
                        raise Exception(
                            f"SSL certificate verification failed when downloading SOFT file. "
                            f"See error message above for solutions."
                        )
                    # If pre-download fails, let GEOparse try (will use FTP as fallback)
                    logger.warning(
                        f"Pre-download failed: {e}. GEOparse will attempt download."
                    )

            # Note: GEOparse will find our pre-downloaded SOFT file and skip its FTP download
            # Use GEOparse to get series matrix
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(self.cache_dir))

            # Check if there's a series matrix
            if hasattr(gse, "table") and gse.table is not None:
                return GEOResult(
                    data=gse.table,
                    metadata=metadata,
                    source=GEODataSource.GEOPARSE,
                    processing_info={"method": "series_matrix"},
                    success=True,
                )

            return GEOResult(success=False, error_message="No series matrix available")

        except Exception as e:
            logger.warning(f"Series matrix download failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    def try_supplementary_files(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Pipeline step: Try supplementary files (non-TAR)."""
        try:
            logger.debug(f"Trying supplementary files for {geo_id}")
            soft_file, data_sources = self.geo_downloader.download_geo_data(geo_id)

            if isinstance(data_sources, dict) and "supplementary" in data_sources:
                matrix = self.geo_parser.parse_supplementary_file(
                    data_sources["supplementary"]
                )
                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.SUPPLEMENTARY,
                        processing_info={
                            "method": "helper_supplementary",
                            "source_file": str(data_sources["supplementary"]),
                        },
                        success=True,
                    )

            return GEOResult(
                success=False, error_message="No supplementary files found or parsable"
            )

        except Exception as e:
            logger.warning(f"Supplementary files download failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    def try_sample_matrices_fallback(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Pipeline step: Try individual sample matrices as fallback."""
        try:
            logger.debug(f"Trying sample matrices fallback for {geo_id}")
            # This is already handled in _try_geoparse_download, so this is a no-op
            return GEOResult(
                success=False,
                error_message="Sample matrices already tried in GEOparse step",
            )

        except Exception as e:
            logger.warning(f"Sample matrices fallback failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    def try_helper_download_fallback(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Pipeline step: Final fallback using helper downloader."""
        try:
            logger.debug(f"Trying helper download fallback for {geo_id}")
            soft_file, data_sources = self.geo_downloader.download_geo_data(geo_id)

            if not data_sources:
                return GEOResult(success=False, error_message="No data sources found")

            # Try all available data sources
            if isinstance(data_sources, dict):
                for source_name, source_path in data_sources.items():
                    try:
                        matrix = self.geo_parser.parse_supplementary_file(source_path)
                        if matrix is not None and not matrix.empty:
                            return GEOResult(
                                data=matrix,
                                metadata=metadata,
                                source=GEODataSource.SOFT_FILE,
                                processing_info={
                                    "method": f"helper_fallback_{source_name}",
                                    "source_file": str(source_path),
                                },
                                success=True,
                            )
                    except Exception as e:
                        logger.warning(f"Failed to parse {source_name}: {e}")
                        continue
            else:
                # Single file
                matrix = self.geo_parser.parse_supplementary_file(data_sources)
                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.SOFT_FILE,
                        processing_info={
                            "method": "helper_fallback_single",
                            "source_file": str(data_sources),
                        },
                        success=True,
                    )

            return GEOResult(
                success=False, error_message="All helper download attempts failed"
            )

        except Exception as e:
            logger.warning(f"Helper download fallback failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    # ========================================
    # SCENARIO-SPECIFIC PUBLIC METHODS
    # ========================================

    def download_single_cell_sample(self, gsm_id: str) -> str:
        """
        Specialized single-cell sample downloading (Scenario 4).

        Args:
            gsm_id: GEO sample ID (e.g., GSM123456)

        Returns:
            str: Status message
        """
        try:
            logger.debug(f"Downloading single-cell sample: {gsm_id}")

            # Clean sample ID
            clean_gsm_id = gsm_id.strip().upper()
            if not clean_gsm_id.startswith("GSM"):
                return f"Invalid sample ID format: {gsm_id}. Must be a GSM accession (e.g., GSM123456)."

            # Try GEOparse first
            try:
                # PRE-DOWNLOAD SOFT FILE USING HTTPS TO BYPASS GEOparse's FTP DOWNLOADER
                # GEOparse internally uses FTP which lacks error detection and causes corruption.
                # By pre-downloading with HTTPS, GEOparse finds existing file and skips its FTP download.
                soft_file_path = Path(self.cache_dir) / f"{clean_gsm_id}.soft"
                if not soft_file_path.exists():
                    # Construct SOFT file URL components for GSM (sample)
                    gsm_num_str = clean_gsm_id[3:]  # Remove 'GSM' prefix
                    if len(gsm_num_str) >= 3:
                        sample_folder = f"GSM{gsm_num_str[:-3]}nnn"
                    else:
                        sample_folder = "GSMnnn"

                    soft_url = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{sample_folder}/{clean_gsm_id}/soft/{clean_gsm_id}.soft"

                    logger.debug(f"Pre-downloading SOFT file using HTTPS: {soft_url}")
                    try:
                        ssl_context = create_ssl_context()
                        with urllib.request.urlopen(
                            soft_url, context=ssl_context
                        ) as response:
                            with open(soft_file_path, "wb") as f:
                                f.write(response.read())
                        logger.debug(
                            f"Successfully pre-downloaded SOFT file to {soft_file_path}"
                        )
                    except Exception as e:
                        error_str = str(e)
                        if (
                            "CERTIFICATE_VERIFY_FAILED" in error_str
                            or "SSL" in error_str
                        ):
                            handle_ssl_error(e, soft_url, logger)
                            raise Exception(
                                f"SSL certificate verification failed when downloading SOFT file. "
                                f"See error message above for solutions."
                            )
                        # If pre-download fails, let GEOparse try (will use FTP as fallback)
                        logger.warning(
                            f"Pre-download failed: {e}. GEOparse will attempt download."
                        )

                # Note: GEOparse will find our pre-downloaded SOFT file and skip its FTP download
                gsm = GEOparse.get_GEO(geo=clean_gsm_id, destdir=str(self.cache_dir))

                # Check for 10X format supplementary files
                if hasattr(gsm, "metadata") and "supplementary_file" in gsm.metadata:
                    suppl_files = gsm.metadata["supplementary_file"]
                    if not isinstance(suppl_files, list):
                        suppl_files = [suppl_files]

                    # Look for 10X format files
                    for suppl_url in suppl_files:
                        if any(
                            pattern in suppl_url.lower()
                            for pattern in [
                                "matrix.mtx",
                                "barcodes.tsv",
                                "features.tsv",
                                ".h5",
                            ]
                        ):

                            # Download and parse 10X data
                            matrix = self._download_and_parse_10x_sample(
                                suppl_url, clean_gsm_id
                            )
                            if matrix is not None:
                                return (
                                    self.geo_service._store_single_sample_as_modality(
                                        clean_gsm_id, matrix, gsm
                                    )
                                )

                # Fallback to expression table
                if hasattr(gsm, "table") and gsm.table is not None:
                    matrix = gsm.table
                    return self.geo_service._store_single_sample_as_modality(
                        clean_gsm_id, matrix, gsm
                    )

            except Exception as e:
                logger.warning(f"GEOparse failed for {clean_gsm_id}: {e}")

            # Fallback to helper downloader
            try:
                return self.download_sample_with_helpers(clean_gsm_id)
            except Exception as e:
                logger.error(f"Helper download failed for {clean_gsm_id}: {e}")
                return f"Failed to download sample {clean_gsm_id}: {str(e)}"

        except Exception as e:
            logger.exception(f"Error downloading single-cell sample {gsm_id}: {e}")
            return f"Error downloading single-cell sample {gsm_id}: {str(e)}"

    def download_bulk_dataset(
        self, geo_id: str, prefer_series_matrix: bool = True
    ) -> str:
        """
        Enhanced bulk data downloading (Scenario 5).

        Args:
            geo_id: GEO accession ID
            prefer_series_matrix: Whether to prefer series matrix over supplementary files

        Returns:
            str: Status message
        """
        try:
            logger.debug(f"Downloading bulk dataset: {geo_id}")

            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith("GSE"):
                return f"Invalid GEO ID format: {geo_id}. Must be a GSE accession."

            # Ensure metadata exists
            if clean_geo_id not in self.data_manager.metadata_store:
                metadata_summary = self.geo_service.fetch_metadata_only(clean_geo_id)
                if "Error" in metadata_summary:
                    return f"Failed to fetch metadata: {metadata_summary}"

            # Strategy for bulk data
            from lobster.services.data_access.geo_service import (
                DownloadStrategy,
                GEODataType,
            )

            strategy = DownloadStrategy(
                prefer_geoparse=True,
                prefer_supplementary=not prefer_series_matrix,
                max_retries=2,
            )

            result = self.geo_service.download_with_strategy(
                clean_geo_id, strategy, GEODataType.BULK
            )

            if result.success:
                # Store as bulk RNA-seq modality
                modality_name = f"geo_{clean_geo_id.lower()}_bulk"
                adata = self.data_manager.load_modality(
                    name=modality_name,
                    source=result.data,
                    adapter="transcriptomics_bulk",
                    validate=True,
                    **result.processing_info,
                )

                # Save to workspace
                save_path = f"{clean_geo_id.lower()}_bulk_raw.h5ad"
                self.data_manager.save_modality(modality_name, save_path)

                return f"""Successfully downloaded bulk dataset {clean_geo_id}!

📊 Modality: '{modality_name}' ({adata.n_obs} samples × {adata.n_vars} genes)
🔬 Adapter: transcriptomics_bulk  
💾 Saved to: {save_path}
📈 Ready for bulk RNA-seq analysis!"""
            else:
                return f"Failed to download bulk dataset {clean_geo_id}: {result.error_message}"

        except Exception as e:
            logger.exception(f"Error downloading bulk dataset {geo_id}: {e}")
            return f"Error downloading bulk dataset {geo_id}: {str(e)}"

    def process_supplementary_tar_files(self, geo_id: str) -> str:
        """
        TAR file processing fallback for single-cell data (Scenario 6).

        Args:
            geo_id: GEO accession ID

        Returns:
            str: Status message
        """
        try:
            logger.debug(f"Processing supplementary TAR files for: {geo_id}")

            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith("GSE"):
                return f"Invalid GEO ID format: {geo_id}. Must be a GSE accession."

            # Use helper downloader for TAR processing
            soft_file, data_sources = self.geo_downloader.download_geo_data(
                clean_geo_id
            )

            if not data_sources:
                return f"No TAR files found for {clean_geo_id}"

            # Process TAR files using helper parser
            if isinstance(data_sources, dict):
                if "tar" in data_sources:
                    matrix = self.geo_parser.parse_supplementary_file(
                        data_sources["tar"]
                    )
                elif "tar_dir" in data_sources:
                    # Process directory with multiple files
                    matrix = self.process_tar_directory_with_helpers(
                        data_sources["tar_dir"]
                    )
                else:
                    return f"No TAR data found in downloaded files for {clean_geo_id}"
            else:
                # Single file
                matrix = self.geo_parser.parse_supplementary_file(data_sources)

            if matrix is None or matrix.empty:
                return f"Failed to parse TAR files for {clean_geo_id}"

            # Store as single-cell modality
            modality_name = f"geo_{clean_geo_id.lower()}_tar"

            enhanced_metadata = {
                "dataset_id": clean_geo_id,
                "dataset_type": "GEO",
                "data_source": "tar_supplementary",
                "processing_date": pd.Timestamp.now().isoformat(),
                "parser_version": "helper_parser",
            }

            adata = self.data_manager.load_modality(
                name=modality_name,
                source=matrix,
                adapter="transcriptomics_single_cell",
                validate=True,
                **enhanced_metadata,
            )

            # Save to workspace
            save_path = f"{clean_geo_id.lower()}_tar_processed.h5ad"
            self.data_manager.save_modality(modality_name, save_path)

            self.data_manager.log_tool_usage(
                tool_name="process_supplementary_tar_files",
                parameters={"geo_id": clean_geo_id},
                description=f"Processed TAR supplementary files for {clean_geo_id}",
            )

            return f"""Successfully processed TAR supplementary files for {clean_geo_id}!

📊 Modality: '{modality_name}' ({adata.n_obs} cells × {adata.n_vars} genes)
🔬 Source: TAR supplementary files
💾 Saved to: {save_path}
📈 Ready for single-cell analysis!"""

        except Exception as e:
            logger.exception(f"Error processing TAR files for {geo_id}: {e}")
            return f"Error processing TAR files for {geo_id}: {str(e)}"

    # ========================================
    # HELPER FUNCTIONS FOR FALLBACK METHODS
    # ========================================

    def download_sample_with_helpers(self, gsm_id: str) -> str:
        """Download sample using helper services."""
        try:
            # Use helper downloader for sample-level downloads
            # This would need to be implemented in the helper downloader
            # For now, return a not implemented message
            return f"Helper-based sample download for {gsm_id} not yet implemented. Please use GEOparse method."

        except Exception as e:
            logger.error(f"Error downloading sample with helpers {gsm_id}: {e}")
            return f"Error downloading sample with helpers {gsm_id}: {str(e)}"

    def process_tar_directory_with_helpers(
        self, tar_dir: Path
    ) -> Optional[pd.DataFrame]:
        """Process TAR directory using helper parser."""
        try:
            # Look for expression files in the directory
            expression_files = []
            for file_path in tar_dir.rglob("*"):
                if file_path.is_file() and any(
                    ext in file_path.name.lower()
                    for ext in [".txt", ".csv", ".tsv", ".gz", ".h5", ".mtx"]
                ):
                    if file_path.stat().st_size > 10000:  # > 10KB
                        expression_files.append(file_path)

            # Sort by size, try largest first
            expression_files.sort(key=lambda x: x.stat().st_size, reverse=True)

            for file_path in expression_files[:5]:  # Try top 5 files
                try:
                    matrix = self.geo_parser.parse_supplementary_file(file_path)
                    if matrix is not None and not matrix.empty:
                        logger.debug(f"Successfully parsed TAR file: {file_path.name}")
                        return matrix
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path.name}: {e}")
                    continue

            return None

        except Exception as e:
            logger.error(f"Error processing TAR directory: {e}")
            return None

    def _download_and_parse_10x_sample(
        self, suppl_url: str, gsm_id: str
    ) -> Optional[pd.DataFrame]:
        """Download and parse 10X format single-cell sample."""
        try:
            # Download using helper downloader for better progress tracking
            local_file = self.cache_dir / f"{gsm_id}_10x_data"
            if self.geo_downloader.download_file(suppl_url, local_file):
                return self.geo_parser.parse_supplementary_file(local_file)
            return None
        except Exception as e:
            logger.error(f"Error downloading 10X sample {gsm_id}: {e}")
            return None
