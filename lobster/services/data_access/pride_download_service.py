"""
PRIDE Download Service implementing IDownloadService interface.

This module provides PRIDE Archive dataset downloading with FTP protocol support,
integrating with the DownloadOrchestrator pattern for queue-based workflow.

Key features:
- FTP download with retry logic and progress tracking
- Auto-detection of file formats (MaxQuant, DIA-NN, mzTab)
- Integration with existing proteomics parsers
- Provenance tracking via AnalysisStep IR
- Support for multiple download strategies
"""

import ftplib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.interfaces.download_service import IDownloadService
from lobster.core.schemas.download_queue import DownloadQueueEntry
from lobster.services.data_access.proteomics_parsers import (
    get_available_parsers,
    get_parser_for_file,
)
from lobster.tools.providers.pride_provider import PRIDEProvider
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PRIDEDownloadService(IDownloadService):
    """
    PRIDE Archive download service implementing IDownloadService.

    Handles downloading proteomics datasets from PRIDE Archive via FTP protocol,
    with auto-detection of file formats and integration with proteomics parsers.

    Download strategies:
    - RESULT_FIRST: Look for processed results (MaxQuant, DIA-NN, Spectronaut)
    - MZML_FIRST: Prioritize mzML files (requires further processing)
    - RAW_FIRST: Download instrument RAW files (large, requires software)
    - SEARCH_FIRST: Download search engine outputs

    Attributes:
        data_manager: DataManagerV2 instance
        pride_provider: PRIDEProvider for metadata and URL extraction
    """

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize PRIDE download service.

        Args:
            data_manager: DataManagerV2 instance for storage and provenance
        """
        super().__init__(data_manager)
        self.pride_provider = PRIDEProvider(data_manager)
        logger.info("PRIDEDownloadService initialized")

    @classmethod
    def supports_database(cls, database: str) -> bool:
        """
        Check if this service handles PRIDE database.

        Args:
            database: Database identifier (case-insensitive)

        Returns:
            bool: True for 'pride' or 'pxd', False otherwise
        """
        return database.lower() in ["pride", "pxd"]

    def supported_databases(self) -> List[str]:
        """
        Get list of databases supported by this service.

        Returns:
            List[str]: ["pride", "pxd"]
        """
        return ["pride", "pxd"]

    @classmethod
    def get_supported_strategies(cls) -> List[str]:
        """
        Get list of download strategies supported by PRIDE service.

        Strategies in order of preference:
        1. RESULT_FIRST - Processed results (MaxQuant, DIA-NN)
        2. MZML_FIRST - Standardized mzML files
        3. SEARCH_FIRST - Search engine outputs
        4. RAW_FIRST - Vendor RAW files (large)

        Returns:
            List[str]: Supported strategy names
        """
        return ["RESULT_FIRST", "MZML_FIRST", "SEARCH_FIRST", "RAW_FIRST"]

    def validate_strategy_params(
        self, strategy_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate PRIDE-specific strategy parameters.

        Args:
            strategy_params: Strategy parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Define allowed parameters and their types
        allowed_params = {
            "file_type": str,  # "maxquant", "diann", "mzml", "raw"
            "include_raw": bool,  # Whether to include RAW files
            "max_file_size_mb": (int, float),  # Maximum file size filter
        }

        for param, value in strategy_params.items():
            if param not in allowed_params:
                logger.warning(f"Unknown strategy parameter '{param}', ignoring")
                continue

            expected_type = allowed_params[param]
            if not isinstance(value, expected_type):
                return (
                    False,
                    f"Parameter '{param}' must be {expected_type}, got {type(value)}",
                )

        return True, None

    def download_dataset(
        self,
        queue_entry: DownloadQueueEntry,
        strategy_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ad.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Execute PRIDE dataset download from queue entry.

        Workflow:
        1. Get file list from PRIDE API
        2. Select files based on strategy
        3. Download via FTP with retry
        4. Auto-detect parser (MaxQuant/DIA-NN)
        5. Parse to AnnData
        6. Store modality and return

        Args:
            queue_entry: Queue entry with PXD accession and metadata
            strategy_override: Optional strategy override

        Returns:
            Tuple of (adata, stats, ir)

        Raises:
            ValueError: If invalid parameters or dataset not found
            RuntimeError: If download or parsing fails
        """
        # Validate database type
        if not self.supports_database(queue_entry.database):
            raise ValueError(
                f"PRIDEDownloadService cannot handle database '{queue_entry.database}'"
            )

        dataset_id = queue_entry.dataset_id
        logger.info(f"Starting PRIDE download for {dataset_id}")

        try:
            # Step 1: Determine strategy
            strategy_name = (
                strategy_override.get("strategy_name")
                if strategy_override
                else (
                    queue_entry.recommended_strategy.strategy_name
                    if queue_entry.recommended_strategy
                    else "RESULT_FIRST"
                )
            )

            strategy_params = (
                strategy_override.get("strategy_params", {})
                if strategy_override
                else {}
            )

            logger.info(f"Using strategy: {strategy_name}")

            # Step 2: Get file URLs from PRIDE (returns DownloadUrlResult)
            url_result = self.pride_provider.get_download_urls(dataset_id)

            if url_result.error:
                raise RuntimeError(
                    f"Failed to get URLs for {dataset_id}: {url_result.error}"
                )

            logger.info(
                f"Retrieved URLs: {len(url_result.raw_files)} RAW, "
                f"{len(url_result.processed_files)} processed, "
                f"{len(url_result.search_files)} search files"
            )

            # Step 3: Select files based on strategy
            selected_files = self._select_files_by_strategy(
                url_result, strategy_name, strategy_params
            )

            if not selected_files:
                raise RuntimeError(
                    f"No suitable files found for strategy '{strategy_name}'"
                )

            logger.info(f"Selected {len(selected_files)} file(s) to download")

            # Step 4: Download files via FTP
            local_files = []
            download_dir = self.data_manager.workspace_dir / "downloads" / dataset_id
            download_dir.mkdir(parents=True, exist_ok=True)

            for file_info in selected_files:
                local_path = self._download_ftp_file(
                    file_info["url"], download_dir / file_info["filename"]
                )
                if local_path:
                    local_files.append(local_path)

            if not local_files:
                raise RuntimeError("All file downloads failed")

            # Step 5: Parse primary file
            primary_file = local_files[0]
            parser = get_parser_for_file(str(primary_file))

            if not parser:
                raise RuntimeError(f"No parser available for file: {primary_file.name}")

            logger.info(f"Parsing {primary_file.name} with {parser.__class__.__name__}")
            adata, parse_stats = parser.parse(str(primary_file))

            # Step 6: Store modality
            modality_name = f"pride_{dataset_id.lower()}_proteomics"
            self.data_manager.modalities[modality_name] = adata

            # Step 7: Build stats
            stats = {
                "dataset_id": dataset_id,
                "database": "pride",
                "modality_name": modality_name,
                "strategy_used": strategy_name,
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars,
                "files_downloaded": len(local_files),
                "parser_used": parser.__class__.__name__,
                **parse_stats,
            }

            # Step 8: Create IR
            ir = self._create_download_ir(dataset_id, strategy_name, strategy_params)

            logger.info(
                f"Successfully downloaded PRIDE dataset {dataset_id}: {adata.shape}"
            )
            return adata, stats, ir

        except Exception as e:
            logger.error(f"Error downloading PRIDE dataset {dataset_id}: {e}")
            raise RuntimeError(f"PRIDE download failed: {str(e)}")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _select_files_by_strategy(
        self,
        url_result,  # DownloadUrlResult from PRIDEProvider
        strategy: str,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Select files to download based on strategy.

        Args:
            url_result: DownloadUrlResult from PRIDEProvider
            strategy: Strategy name
            params: Strategy parameters

        Returns:
            List of file dicts to download (with url, filename, size keys)
        """
        selected = []

        if strategy == "RESULT_FIRST":
            # Look for processed results (MaxQuant, DIA-NN)
            result_files = url_result.processed_files

            # Prioritize known formats
            for keyword in ["proteinGroups", "report.tsv", "report.parquet"]:
                matching = [f for f in result_files if keyword in f.filename]
                if matching:
                    selected.extend(matching[:1])  # Take first match
                    break

            # Fallback to any result file
            if not selected and result_files:
                selected.append(result_files[0])

        elif strategy == "MZML_FIRST":
            # Look for mzML files
            processed = url_result.processed_files
            mzml_files = [f for f in processed if f.filename.endswith(".mzML")]
            if mzml_files:
                selected.extend(mzml_files[:5])  # Limit to 5 files

        elif strategy == "SEARCH_FIRST":
            # Search engine outputs
            search_files = url_result.search_files
            if search_files:
                selected.extend(search_files[:5])

        elif strategy == "RAW_FIRST":
            # Instrument RAW files
            raw_files = url_result.raw_files
            if raw_files:
                selected.extend(raw_files[:5])

        # Apply size filter if specified
        max_size_mb = params.get("max_file_size_mb")
        if max_size_mb:
            max_bytes = max_size_mb * 1024 * 1024
            selected = [f for f in selected if (f.size_bytes or 0) <= max_bytes]

        # Convert DownloadFile objects to dicts for FTP download method
        return [
            {"url": f.url, "filename": f.filename, "size": f.size_bytes or 0}
            for f in selected
        ]

    def _download_ftp_file(
        self, ftp_url: str, output_path: Path, max_retries: int = 3
    ) -> Optional[Path]:
        """
        Download file from PRIDE FTP with retry logic.

        Adapted from pridepy download patterns with simplified implementation.

        Args:
            ftp_url: FTP URL (e.g., ftp://ftp.pride.ebi.ac.uk/pride/data/archive/...)
            output_path: Local path to save file
            max_retries: Maximum retry attempts

        Returns:
            Path to downloaded file, or None if failed
        """
        # Parse FTP URL
        if not ftp_url.startswith("ftp://"):
            logger.error(f"Invalid FTP URL: {ftp_url}")
            return None

        # Extract host and path
        url_parts = ftp_url.replace("ftp://", "").split("/", 1)
        if len(url_parts) != 2:
            logger.error(f"Cannot parse FTP URL: {ftp_url}")
            return None

        host, remote_path = url_parts
        logger.info(f"Downloading from FTP: {host}/{remote_path}")

        for attempt in range(max_retries):
            try:
                ftp = ftplib.FTP(host, timeout=60)
                ftp.login()  # Anonymous login
                ftp.set_pasv(True)  # Passive mode

                # Get file size for progress
                file_size = ftp.size(remote_path)
                logger.info(
                    f"File size: {file_size / 1024 / 1024:.1f} MB"
                    if file_size
                    else "File size unknown"
                )

                # Download with binary mode
                with open(output_path, "wb") as f:
                    ftp.retrbinary(f"RETR {remote_path}", f.write)

                ftp.quit()
                logger.info(f"Downloaded: {output_path.name}")
                return output_path

            except ftplib.error_perm as e:
                logger.error(f"FTP permission error: {e}")
                return None

            except (ftplib.error_temp, ConnectionError, TimeoutError) as e:
                logger.warning(f"FTP error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to download {ftp_url} after {max_retries} attempts"
                    )
                    return None

            except Exception as e:
                logger.error(f"Unexpected error downloading from FTP: {e}")
                return None

        return None

    def _create_download_ir(
        self, dataset_id: str, strategy: str, params: Dict[str, Any]
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for PRIDE download provenance.

        Args:
            dataset_id: PXD accession
            strategy: Strategy name used
            params: Strategy parameters

        Returns:
            AnalysisStep for provenance tracking
        """
        return AnalysisStep(
            operation="pride.download.download_dataset",
            tool_name="PRIDEDownloadService",
            description=f"Download PRIDE dataset {dataset_id}",
            library="lobster.services.data_access.pride_download_service",
            code_template="""# PRIDE dataset download
from lobster.services.data_access.pride_download_service import PRIDEDownloadService
from lobster.tools.providers.pride_provider import PRIDEProvider
from lobster.core.data_manager_v2 import DataManagerV2

# Initialize
dm = DataManagerV2(workspace_dir="./workspace")
provider = PRIDEProvider(dm)
service = PRIDEDownloadService(dm)

# Get file URLs
file_urls = provider.get_download_urls({{ dataset_id | tojson }})

# Download files (simplified - see service for full implementation)
# Parse with appropriate proteomics parser
# Store as modality
""",
            imports=[
                "from lobster.services.data_access.pride_download_service import PRIDEDownloadService",
                "from lobster.tools.providers.pride_provider import PRIDEProvider",
            ],
            parameters={
                "dataset_id": dataset_id,
                "strategy": strategy,
                **params,
            },
            parameter_schema={
                "dataset_id": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="PRIDE accession (PXD format)",
                ),
            },
            input_entities=[dataset_id],
            output_entities=[f"pride_{dataset_id.lower()}_proteomics"],
        )
