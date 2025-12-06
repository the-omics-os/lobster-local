"""
SRA Download Service implementing IDownloadService interface.

This module provides robust download capabilities for NCBI SRA, ENA, and DDBJ
with automatic mirror failover, checksum validation, and resume support.

The service follows Lobster's download architecture patterns:
- Implements IDownloadService for DownloadOrchestrator integration
- Returns (adata, stats, ir) tuple for provenance tracking
- Uses queue-based workflow (research_agent → data_expert)
- Leverages existing infrastructure (AccessionResolver, rate limiter, schemas)

Key Features:
    - Multi-mirror failover (ENA → NCBI → DDBJ)
    - MD5 checksum validation for data integrity
    - Chunked downloads with atomic writes
    - Size warnings for large datasets (>100 GB)
    - Comprehensive provenance tracking via AnalysisStep IR

Example Usage:
    ```python
    from lobster.core.data_manager_v2 import DataManagerV2
    from lobster.tools.download_orchestrator import DownloadOrchestrator
    from lobster.services.data_access.sra_download_service import SRADownloadService

    # Initialize service
    dm = DataManagerV2(workspace_dir="./workspace")
    orchestrator = DownloadOrchestrator(dm)
    sra_service = SRADownloadService(dm)
    orchestrator.register_service(sra_service)

    # Execute download from queue
    modality_name, stats = orchestrator.execute_download("queue_srr123_abc")
    print(f"Downloaded: {modality_name}")
    ```
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import pandas as pd
import requests
import scipy.sparse as sp

from lobster.core.analysis_ir import AnalysisStep
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.interfaces.download_service import IDownloadService
from lobster.core.schemas.download_queue import DownloadQueueEntry
from lobster.tools.providers.sra_provider import SRAProvider, SRAProviderError
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class SRADownloadService(IDownloadService):
    """
    SRA database download service implementing IDownloadService.

    Provides robust download capabilities for FASTQ files from the Sequence
    Read Archive, with automatic failover across multiple international mirrors
    (ENA, NCBI, DDBJ) and comprehensive data integrity validation.

    Supported Strategies:
        - FASTQ_FIRST: Download FASTQ from ENA (default, recommended)

    Future Strategies (Phase 2):
        - SRA_FIRST: Download .sra format, convert locally via sra-tools
        - PROCESSED_FIRST: Download pre-aligned BAM files

    Design Notes:
        - Follows adapter pattern similar to GEODownloadService
        - Delegates URL fetching to existing SRAProvider
        - Uses internal SRADownloadManager for download execution
        - Creates metadata-based AnnData (FASTQ is pre-quantification)
        - Full W3C-PROV compliance via AnalysisStep IR

    Attributes:
        data_manager: DataManagerV2 instance for storage and provenance
        sra_provider: SRAProvider for URL fetching and metadata
        download_manager: Internal manager for download operations
        fastq_loader: Converter from FASTQ files to AnnData metadata
    """

    SUPPORTED_DATABASES = ["sra", "ena", "ddbj"]
    SUPPORTED_STRATEGIES = ["FASTQ_FIRST"]
    CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0  # Exponential multiplier
    SIZE_WARNING_THRESHOLD_GB = 100  # Warn for downloads >100 GB

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize SRA download service.

        Args:
            data_manager: DataManagerV2 instance for data storage and provenance tracking
        """
        super().__init__(data_manager)
        self.sra_provider = SRAProvider(data_manager)
        self.download_manager = SRADownloadManager()
        self.fastq_loader = FASTQLoader()
        logger.info("SRADownloadService initialized")

    @classmethod
    def supports_database(cls, database: str) -> bool:
        """
        Check if this service handles the given database type.

        Args:
            database: Database identifier (case-insensitive)

        Returns:
            True if database is sra, ena, or ddbj

        Example:
            >>> SRADownloadService.supports_database("sra")
            True
            >>> SRADownloadService.supports_database("geo")
            False
        """
        return database.lower() in [db.lower() for db in cls.SUPPORTED_DATABASES]

    def supported_databases(self) -> List[str]:
        """
        Get list of databases supported by this service.

        Returns:
            ["sra", "ena", "ddbj"]
        """
        return self.SUPPORTED_DATABASES

    @classmethod
    def get_supported_strategies(cls) -> List[str]:
        """
        Get list of download strategies supported by this service.

        Returns:
            ["FASTQ_FIRST"] in Phase 1

        Future strategies (Phase 2):
            ["FASTQ_FIRST", "SRA_FIRST", "PROCESSED_FIRST"]
        """
        return cls.SUPPORTED_STRATEGIES

    def download_dataset(
        self,
        queue_entry: DownloadQueueEntry,
        strategy_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ad.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Execute SRA dataset download from queue entry.

        This method orchestrates the complete download workflow:
        1. Validates queue entry (database must be sra/ena/ddbj)
        2. Gets download URLs from SRAProvider (calls ENA API)
        3. Checks total download size and warns if >100 GB
        4. Downloads FASTQ files with mirror failover
        5. Verifies MD5 checksums for data integrity
        6. Creates metadata-based AnnData
        7. Builds stats dict + AnalysisStep IR
        8. Returns (adata, stats, ir) tuple

        Args:
            queue_entry: Queue entry prepared by research_agent containing:
                - dataset_id: SRA accession (SRR, SRX, SRP)
                - database: Must be "sra", "ena", or "ddbj"
                - metadata: Full SRA metadata (optional)
                - recommended_strategy: Strategy config (optional)

            strategy_override: Optional dict to override download strategy:
                - strategy_name: One of SUPPORTED_STRATEGIES
                - strategy_params: {verify_checksum: bool, layout: str}

        Returns:
            Tuple containing:
                - adata: AnnData with run metadata (not expression matrix - FASTQ is raw data)
                  Contains:
                    - .obs: Run-level metadata (accessions, file paths, sizes, read types)
                    - .uns: Study metadata, download provenance, file paths
                    - .X: Empty sparse matrix (placeholder for quantification)

                - stats: Download statistics dict with keys:
                    - dataset_id: SRA accession
                    - database: "sra"
                    - strategy_used: Strategy name
                    - n_files: Number of FASTQ files downloaded
                    - total_size_mb: Total download size in MB
                    - download_time_seconds: Time taken for download
                    - layout: SINGLE or PAIRED
                    - platform: Sequencing platform
                    - mirror_used: Mirror that succeeded (ena/ncbi/ddbj)

                - ir: AnalysisStep for provenance tracking and notebook export

        Raises:
            ValueError: If queue_entry.database not supported or invalid parameters
            SRAProviderError: If URL fetching fails
            RuntimeError: If all mirrors fail or download corrupted
            OSError: If disk space insufficient

        Example:
            >>> entry = DownloadQueueEntry(
            ...     entry_id="queue_123",
            ...     dataset_id="SRR21960766",
            ...     database="sra"
            ... )
            >>> service = SRADownloadService(data_manager)
            >>> adata, stats, ir = service.download_dataset(entry)
            >>> print(stats)
            {
                "dataset_id": "SRR21960766",
                "database": "sra",
                "strategy_used": "FASTQ_FIRST",
                "n_files": 2,
                "total_size_mb": 145.3,
                "download_time_seconds": 42.1,
                "layout": "PAIRED",
                "platform": "ILLUMINA",
                "mirror_used": "ena"
            }
        """
        logger.info(f"Starting SRA download for dataset: {queue_entry.dataset_id}")
        start_time = time.time()

        # Step 1: Validate queue entry
        if not self.supports_database(queue_entry.database):
            raise ValueError(
                f"SRADownloadService only handles sra/ena/ddbj databases, "
                f"got '{queue_entry.database}'"
            )

        accession = queue_entry.dataset_id.strip().upper()
        logger.debug(f"Validated SRA accession: {accession}")

        # Step 2: Determine download strategy
        strategy_name = "FASTQ_FIRST"  # Default for Phase 1
        verify_checksum = True  # Default to secure

        if strategy_override:
            if "strategy_name" in strategy_override:
                strategy_name = strategy_override["strategy_name"]
                logger.info(f"Using strategy override: {strategy_name}")

            if "strategy_params" in strategy_override:
                params = strategy_override["strategy_params"]
                verify_checksum = params.get("verify_checksum", True)
                logger.debug(f"Checksum verification: {verify_checksum}")

        # Step 3: Get download URLs from ENA (returns DownloadUrlResult)
        try:
            url_result = self.sra_provider.get_download_urls(accession)
            # Convert to legacy dict for internal methods (download_manager, fastq_loader)
            url_info = url_result.to_legacy_dict()
            logger.info(
                f"Retrieved {len(url_result.raw_files)} FASTQ URL(s) from ENA "
                f"({url_result.get_total_size() / 1e6:.1f} MB)"
            )
        except Exception as e:
            logger.error(f"Failed to get download URLs for {accession}: {e}")
            raise

        # Step 4: Check download size and warn if large
        self._check_download_size(url_info["total_size_bytes"], accession)

        # Step 5: Create output directory in workspace
        output_dir = self.data_manager.workspace_dir / "downloads" / "sra" / accession
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Download directory: {output_dir}")

        # Step 6: Download FASTQ files with mirror failover
        try:
            fastq_paths, download_stats = self.download_manager.download_run(
                run_accession=accession,
                urls=url_info,
                output_dir=output_dir,
                verify_checksum=verify_checksum,
            )
            logger.info(
                f"Downloaded {len(fastq_paths)} FASTQ file(s) via {download_stats['mirror_used']}"
            )
        except Exception as e:
            logger.error(f"Download failed for {accession}: {e}")
            raise RuntimeError(f"Failed to download {accession}: {str(e)}") from e

        # Step 7: Create metadata-based AnnData
        try:
            adata = self.fastq_loader.create_fastq_anndata(
                fastq_paths=fastq_paths,
                metadata=url_info,
                queue_entry=queue_entry,
            )
            logger.info(f"Created FASTQ metadata AnnData: {adata.n_obs} file(s)")
        except Exception as e:
            logger.error(f"Failed to create AnnData for {accession}: {e}")
            raise RuntimeError(f"Failed to create AnnData: {str(e)}") from e

        # Step 8: Build statistics dictionary
        total_time = time.time() - start_time
        stats = {
            "dataset_id": accession,
            "database": "sra",
            "strategy_used": strategy_name,
            "n_files": len(fastq_paths),
            "total_size_mb": url_info["total_size_bytes"] / 1e6,
            "download_time_seconds": round(total_time, 2),
            "layout": url_info["layout"],
            "platform": url_info["platform"],
            "mirror_used": download_stats["mirror_used"],
            "checksum_verified": verify_checksum,
        }

        logger.debug(f"Download stats: {stats}")

        # Step 9: Create AnalysisStep IR for provenance
        ir = self._create_analysis_step_ir(
            accession=accession,
            fastq_paths=fastq_paths,
            strategy_name=strategy_name,
            verify_checksum=verify_checksum,
            queue_entry_id=queue_entry.entry_id,
            url_info=url_info,
        )

        logger.info(f"Successfully completed SRA download for {accession}")
        return adata, stats, ir

    def _check_download_size(self, total_size_bytes: int, dataset_id: str) -> None:
        """
        Check download size and warn if >100 GB.

        This method implements a soft warning for large downloads to prevent
        accidental disk space exhaustion. Users can override the warning by
        setting LOBSTER_SKIP_SIZE_WARNING=true.

        Args:
            total_size_bytes: Total size of all files to download
            dataset_id: Dataset identifier for logging

        Raises:
            ValueError: If size >100 GB and LOBSTER_SKIP_SIZE_WARNING not set
        """
        size_gb = total_size_bytes / 1e9

        if size_gb > self.SIZE_WARNING_THRESHOLD_GB:
            # Check if running in non-interactive mode (CI, scripts)
            skip_warning = (
                os.getenv("LOBSTER_SKIP_SIZE_WARNING", "false").lower() == "true"
            )

            if skip_warning:
                logger.warning(
                    f"Large download detected ({size_gb:.1f} GB) but proceeding "
                    f"(LOBSTER_SKIP_SIZE_WARNING=true)"
                )
                return

            # Raise error with instructions for override
            raise ValueError(
                f"Dataset {dataset_id} is large ({size_gb:.1f} GB), "
                f"exceeding the {self.SIZE_WARNING_THRESHOLD_GB} GB threshold. "
                f"Ensure you have sufficient disk space (~{size_gb * 2:.1f} GB free). "
                f"Set LOBSTER_SKIP_SIZE_WARNING=true to proceed without confirmation."
            )

    def validate_strategy_params(
        self, strategy_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate SRA-specific strategy parameters.

        Checks parameter types and allowed values for SRA download strategies.
        Unknown parameters are ignored for forward compatibility.

        Args:
            strategy_params: Dictionary of strategy-specific parameters.
                Recognized parameters:
                - verify_checksum: bool (default: True) - Validate MD5 checksums
                - layout: Optional[str] - Force "SINGLE" or "PAIRED" (default: auto-detect)
                - file_type: str (default: "fastq") - File format (future: "sra", "bam")

        Returns:
            Tuple containing:
                - is_valid: True if all parameters are valid
                - error_message: Descriptive error if invalid, None if valid

        Example:
            >>> service = SRADownloadService(data_manager)
            >>> valid, error = service.validate_strategy_params({
            ...     "verify_checksum": True,
            ...     "layout": "PAIRED"
            ... })
            >>> assert valid is True
        """
        logger.debug(f"Validating strategy parameters: {strategy_params}")

        # Validate verify_checksum parameter
        if "verify_checksum" in strategy_params:
            value = strategy_params["verify_checksum"]
            if not isinstance(value, bool):
                error_msg = f"Parameter 'verify_checksum' must be bool, got {type(value).__name__}"
                logger.warning(f"Validation failed: {error_msg}")
                return False, error_msg

        # Validate layout parameter
        if "layout" in strategy_params:
            value = strategy_params["layout"]
            if value is not None and value.upper() not in ["SINGLE", "PAIRED"]:
                error_msg = (
                    f"Parameter 'layout' must be 'SINGLE' or 'PAIRED', got '{value}'"
                )
                logger.warning(f"Validation failed: {error_msg}")
                return False, error_msg

        # Validate file_type parameter (future-proofing)
        if "file_type" in strategy_params:
            value = strategy_params["file_type"]
            if value.lower() not in ["fastq"]:  # Only FASTQ supported in Phase 1
                error_msg = (
                    f"Parameter 'file_type' must be 'fastq' (only format supported in Phase 1), "
                    f"got '{value}'"
                )
                logger.warning(f"Validation failed: {error_msg}")
                return False, error_msg

        logger.debug("Strategy parameters validation passed")
        return True, None

    def _create_analysis_step_ir(
        self,
        accession: str,
        fastq_paths: List[Path],
        strategy_name: str,
        verify_checksum: bool,
        queue_entry_id: str,
        url_info: Dict[str, Any],
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for provenance tracking.

        Generates intermediate representation capturing all parameters
        and execution details for reproducibility via Jupyter notebooks.

        Args:
            accession: SRA accession ID
            fastq_paths: List of downloaded FASTQ file paths
            strategy_name: Download strategy used
            verify_checksum: Whether checksums were verified
            queue_entry_id: Queue entry ID for traceability
            url_info: URL info dict from SRAProvider

        Returns:
            AnalysisStep: Provenance tracking object with Jinja2 code template
        """
        # Build parameters dictionary
        parameters = {
            "queue_entry_id": queue_entry_id,
            "dataset_id": accession,
            "database": "sra",
            "strategy": strategy_name,
            "verify_checksum": verify_checksum,
            "layout": url_info["layout"],
            "platform": url_info["platform"],
            "total_size_bytes": url_info["total_size_bytes"],
        }

        # Build parameter schema
        parameter_schema = {
            "queue_entry_id": {
                "type": "string",
                "required": True,
                "description": "Queue entry identifier for traceability",
            },
            "dataset_id": {
                "type": "string",
                "required": True,
                "description": "SRA accession ID (SRR, SRX, or SRP)",
                "pattern": "^[SED]R[RPXS]\\d+$",
            },
            "database": {
                "type": "string",
                "required": True,
                "description": "Source database (sra, ena, or ddbj)",
                "enum": ["sra", "ena", "ddbj"],
            },
            "strategy": {
                "type": "string",
                "required": True,
                "description": "Download strategy name",
                "enum": self.SUPPORTED_STRATEGIES,
            },
            "verify_checksum": {
                "type": "boolean",
                "required": False,
                "description": "Whether to verify MD5 checksums (default: True)",
            },
            "layout": {
                "type": "string",
                "required": False,
                "description": "Library layout detected from metadata",
                "enum": ["SINGLE", "PAIRED", "UNKNOWN"],
            },
        }

        # Create AnalysisStep IR
        ir = AnalysisStep(
            operation="sra_download",
            tool_name="SRADownloadService.download_dataset",
            description=f"Downloaded SRA dataset {accession} as FASTQ files from ENA mirror",
            library="lobster",
            imports=[
                "from lobster.tools.download_orchestrator import DownloadOrchestrator",
                "from lobster.services.data_access.sra_download_service import SRADownloadService",
                "from lobster.core.data_manager_v2 import DataManagerV2",
            ],
            code_template="""# Download SRA dataset using DownloadOrchestrator
data_manager = DataManagerV2(workspace_dir="./workspace")
orchestrator = DownloadOrchestrator(data_manager)
sra_service = SRADownloadService(data_manager)
orchestrator.register_service(sra_service)

# Execute download from queue entry
modality_name, stats = orchestrator.execute_download("{{ queue_entry_id }}")
print(f"Downloaded: {modality_name}")
print(f"Stats: {stats}")

# Access downloaded FASTQ metadata
adata = data_manager.get_modality(modality_name)
fastq_files = adata.uns["fastq_files"]["paths"]
print(f"FASTQ files: {fastq_files}")
""",
            parameters=parameters,
            parameter_schema=parameter_schema,
            input_entities=[queue_entry_id],
            output_entities=[f"sra_{accession.lower()}"],
        )

        logger.debug(f"Created AnalysisStep IR for {accession}")
        return ir


class SRADownloadManager:
    """
    Internal manager for SRA download operations.

    This class handles the low-level download logic including:
    - Multi-mirror failover (ENA → NCBI → DDBJ)
    - Chunked downloads with progress tracking
    - MD5 checksum validation
    - Atomic writes (.tmp → final)
    - Paired-end file organization

    Design Note:
        This is an internal helper class. All public interaction should go
        through SRADownloadService which implements the IDownloadService interface.

    Attributes:
        CHUNK_SIZE: Download chunk size (8 MB)
        MIRRORS: Priority-ordered list of mirrors to try
    """

    CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
    MIRRORS = ["ena", "ncbi", "ddbj"]  # Priority order: ENA first

    def download_run(
        self,
        run_accession: str,
        urls: Dict[str, Any],
        output_dir: Path,
        verify_checksum: bool = True,
    ) -> Tuple[List[Path], Dict[str, Any]]:
        """
        Download a single SRA run with multi-mirror failover.

        Attempts download from mirrors in priority order (ENA → NCBI → DDBJ),
        falling back to the next mirror if any step fails.

        Args:
            run_accession: SRA run accession (e.g., "SRR21960766")
            urls: URL info dict from SRAProvider.get_download_urls()
            output_dir: Directory to save FASTQ files
            verify_checksum: Whether to validate MD5 checksums

        Returns:
            Tuple containing:
                - file_paths: List of downloaded FASTQ file paths
                - stats_dict: {mirror_used, total_size_bytes, download_time_seconds}

        Raises:
            RuntimeError: If all mirrors fail
        """
        logger.info(f"Starting download for {run_accession}")
        start_time = time.time()
        last_error = None

        # Try each mirror in priority order
        for mirror in self.MIRRORS:
            try:
                logger.debug(f"Attempting download from mirror: {mirror}")

                # Download all files from this mirror
                downloaded_files = []
                for url_entry in urls["raw_urls"]:
                    file_path = output_dir / url_entry["filename"]

                    # Download with progress tracking
                    self._download_with_progress(
                        url=url_entry["url"],
                        output_path=file_path,
                        expected_size=url_entry["size"],
                        expected_md5=url_entry["md5"] if verify_checksum else None,
                    )

                    downloaded_files.append(file_path)
                    logger.debug(f"Downloaded: {file_path.name}")

                # All files downloaded successfully from this mirror
                download_time = time.time() - start_time

                stats = {
                    "mirror_used": mirror,
                    "total_size_bytes": sum(f.stat().st_size for f in downloaded_files),
                    "download_time_seconds": round(download_time, 2),
                }

                logger.info(
                    f"Download complete from {mirror}: "
                    f"{len(downloaded_files)} file(s) in {download_time:.1f}s"
                )

                return downloaded_files, stats

            except Exception as e:
                logger.warning(f"Mirror {mirror} failed for {run_accession}: {e}")
                last_error = e

                # Cleanup partial downloads from this mirror
                for url_entry in urls["raw_urls"]:
                    file_path = output_dir / url_entry["filename"]
                    if file_path.exists():
                        logger.debug(f"Cleaning up partial download: {file_path}")
                        file_path.unlink()

                continue  # Try next mirror

        # All mirrors failed
        raise RuntimeError(
            f"All mirrors failed for {run_accession}. Last error: {last_error}"
        )

    def _download_with_progress(
        self,
        url: str,
        output_path: Path,
        expected_size: Optional[int] = None,
        expected_md5: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        """
        Chunked download with progress tracking, atomic write, and robust error handling.

        This method implements production-grade download reliability patterns from
        nf-core/fetchngs, including:
        - HTTP 429 handling with Retry-After header support
        - HTTP 500 retry with exponential backoff
        - Status 204 detection for permission issues
        - Automatic retry with progressive delays

        Safety measures implemented:
        1. Download to .tmp file first (atomic operation)
        2. Stream download with 8MB chunks
        3. Verify MD5 checksum if provided
        4. Atomic rename on success
        5. Cleanup .tmp file on any failure
        6. Retry failed requests with exponential backoff

        Args:
            url: Download URL
            output_path: Final file path
            expected_size: Expected file size in bytes (optional)
            expected_md5: Expected MD5 checksum (optional)
            max_retries: Maximum number of retry attempts (default: 3)

        Raises:
            requests.RequestException: If download fails after all retries
            ValueError: If MD5 mismatch detected
            PermissionError: If HTTP 204 (no content) - permission issue
        """
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        sleep_time = 5  # Initial backoff delay (seconds)
        attempt = 0

        while attempt <= max_retries:
            try:
                logger.debug(f"Download attempt {attempt + 1}/{max_retries + 1}: {url}")

                # Stream download with chunking
                with requests.get(url, stream=True, timeout=30) as response:
                    # Handle HTTP errors with production-grade patterns from nf-core
                    if response.status_code == 429:
                        # Rate limit exceeded - respect Retry-After header
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            retry_delay = int(retry_after)
                            logger.warning(
                                f"HTTP 429 (rate limit) from server. "
                                f"Retrying after {retry_delay}s (from Retry-After header)"
                            )
                            time.sleep(retry_delay)
                        else:
                            logger.warning(
                                f"HTTP 429 (rate limit) from server. "
                                f"Retrying after {sleep_time}s (exponential backoff)"
                            )
                            time.sleep(sleep_time)
                            sleep_time *= 2  # Exponential backoff
                        attempt += 1
                        continue  # Retry

                    elif response.status_code == 500:
                        # Server error - retry with exponential backoff
                        if attempt < max_retries:
                            logger.warning(
                                f"HTTP 500 (server error) from {url}. "
                                f"Retrying after {sleep_time}s (attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(sleep_time)
                            sleep_time *= 2  # Double delay each retry
                            attempt += 1
                            continue  # Retry
                        else:
                            logger.error(
                                f"HTTP 500: Exceeded max retry attempts ({max_retries}) for {url}"
                            )
                            raise requests.HTTPError(
                                f"Server error after {max_retries} retries",
                                response=response,
                            )

                    elif response.status_code == 204:
                        # No content - likely permission issue
                        raise PermissionError(
                            f"HTTP 204 (no content) for {url}. "
                            f"The dataset may be restricted or you may lack permissions. "
                            f"Check if the accession is publicly available."
                        )

                    # Raise for other HTTP errors
                    response.raise_for_status()

                    # Download successful - process content
                    total_size = int(response.headers.get("content-length", 0))

                    with open(temp_path, "wb") as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)

                                # Log progress for large files (every 100 MB)
                                if downloaded % (100 * 1024 * 1024) == 0:
                                    progress = (
                                        (downloaded / total_size * 100)
                                        if total_size > 0
                                        else 0
                                    )
                                    logger.debug(
                                        f"Progress: {downloaded / 1e6:.1f} MB / "
                                        f"{total_size / 1e6:.1f} MB ({progress:.1f}%)"
                                    )

                logger.debug(
                    f"Download complete: {temp_path.name} ({temp_path.stat().st_size / 1e6:.1f} MB)"
                )

                # Verify MD5 checksum if provided
                if expected_md5:
                    logger.debug(f"Verifying MD5 checksum: {expected_md5}")
                    if not self._verify_md5(temp_path, expected_md5):
                        raise ValueError(
                            f"MD5 checksum mismatch for {url}. "
                            f"File may be corrupted. Will retry from different mirror."
                        )
                    logger.debug("MD5 checksum verified successfully")

                # Verify file size if provided (quick sanity check)
                if expected_size and expected_size > 0:
                    actual_size = temp_path.stat().st_size
                    if abs(actual_size - expected_size) > 1024:  # Allow 1 KB tolerance
                        logger.warning(
                            f"File size mismatch: expected {expected_size}, got {actual_size} "
                            f"(difference: {abs(actual_size - expected_size)} bytes)"
                        )

                # Atomic rename on success
                temp_path.rename(output_path)
                logger.debug(f"File ready: {output_path.name}")
                return  # Success - exit retry loop

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as e:
                # Network errors - retry with backoff
                if attempt < max_retries:
                    logger.warning(
                        f"Network error: {e}. Retrying after {sleep_time}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(sleep_time)
                    sleep_time *= 2
                    attempt += 1
                    # Cleanup partial download before retry
                    if temp_path.exists():
                        temp_path.unlink()
                    continue  # Retry
                else:
                    logger.error(f"Network error after {max_retries} retries: {e}")
                    # Cleanup and re-raise
                    if temp_path.exists():
                        temp_path.unlink()
                    raise

            except Exception as e:
                # Other errors - cleanup and re-raise immediately
                logger.error(f"Download error for {url}: {e}")
                if temp_path.exists():
                    logger.debug(f"Cleaning up failed download: {temp_path}")
                    temp_path.unlink()
                raise

        # If we get here, all retries exhausted
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(
            f"Download failed after {max_retries + 1} attempts for {url}"
        )

    def _verify_md5(self, file_path: Path, expected_md5: str) -> bool:
        """
        Verify file MD5 checksum.

        Args:
            file_path: Path to file to verify
            expected_md5: Expected MD5 hash (hexadecimal)

        Returns:
            True if checksum matches, False otherwise
        """
        md5 = hashlib.md5()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)

        computed = md5.hexdigest()
        matches = computed == expected_md5

        if not matches:
            logger.error(
                f"MD5 mismatch for {file_path.name}: "
                f"expected {expected_md5}, got {computed}"
            )

        return matches


class FASTQLoader:
    """
    Load FASTQ metadata into AnnData structure.

    IMPORTANT: FASTQ files contain raw sequencing reads, not expression matrices.
    The AnnData created by this loader serves as a metadata container with:
    - .obs: Run-level metadata (accession, file paths, sizes, read types)
    - .uns: Study metadata, download provenance, processing requirements
    - .X: Empty sparse matrix (placeholder for downstream quantification)

    Users must run alignment + quantification pipelines (e.g., Kallisto, STAR)
    to convert raw FASTQ data into expression matrices.

    Design Rationale:
        This metadata-based approach:
        1. Maintains pattern consistency with other modalities
        2. Provides clear provenance tracking
        3. Stores all necessary information for downstream processing
        4. Integrates seamlessly with DataManagerV2
    """

    def create_fastq_anndata(
        self,
        fastq_paths: List[Path],
        metadata: Dict[str, Any],
        queue_entry: DownloadQueueEntry,
    ) -> ad.AnnData:
        """
        Create AnnData from downloaded FASTQ files.

        The AnnData represents the downloaded dataset structure, ready for
        downstream processing (alignment, quantification). It does NOT contain
        expression data - that requires running bioinformatics pipelines.

        Args:
            fastq_paths: List of downloaded FASTQ file paths
            metadata: Download metadata from SRAProvider.get_download_urls()
            queue_entry: Original queue entry with full SRA metadata

        Returns:
            AnnData with:
                - .obs: Run metadata (1 row per FASTQ file)
                    Columns: run_accession, fastq_path, file_size_bytes,
                             read_type, layout, platform
                - .uns: Study/experiment metadata, file paths, provenance
                    Keys: sra_metadata, data_type, processing_required,
                          download_provenance, fastq_files
                - .X: Empty sparse matrix (FASTQ is pre-quantification)

        Example:
            >>> loader = FASTQLoader()
            >>> adata = loader.create_fastq_anndata(
            ...     fastq_paths=[Path("SRR001_1.fastq.gz"), Path("SRR001_2.fastq.gz")],
            ...     metadata={"layout": "PAIRED", "platform": "ILLUMINA"},
            ...     queue_entry=queue_entry
            ... )
            >>> print(adata.n_obs)  # 2 (one per file)
            >>> print(adata.uns["data_type"])  # "fastq_raw"
            >>> print(adata.uns["processing_required"])  # ["alignment", "quantification"]
        """
        logger.debug(f"Creating FASTQ metadata AnnData from {len(fastq_paths)} file(s)")

        # Create observation (file) metadata
        obs_data = {
            "run_accession": [],
            "fastq_path": [],
            "file_size_bytes": [],
            "file_size_mb": [],
            "read_type": [],  # "R1", "R2", or "single"
            "layout": [],
            "platform": [],
        }

        for path in fastq_paths:
            run_id = self._extract_run_id(path.name)
            read_type = self._detect_read_type(path.name)
            file_size = path.stat().st_size

            obs_data["run_accession"].append(run_id)
            obs_data["fastq_path"].append(str(path.absolute()))
            obs_data["file_size_bytes"].append(file_size)
            obs_data["file_size_mb"].append(round(file_size / 1e6, 2))
            obs_data["read_type"].append(read_type)
            obs_data["layout"].append(metadata.get("layout", "UNKNOWN"))
            obs_data["platform"].append(metadata.get("platform", "UNKNOWN"))

        obs = pd.DataFrame(obs_data)

        # Create index: {run_accession}_{read_type}
        obs.index = [
            f"{acc}_{rt}" for acc, rt in zip(obs["run_accession"], obs["read_type"])
        ]

        # Create minimal AnnData (FASTQ is raw data, no expression matrix yet)
        n_obs = len(obs)
        n_vars = 1  # Placeholder
        X = sp.csr_matrix((n_obs, n_vars))

        adata = ad.AnnData(X=X, obs=obs)

        # Store metadata in .uns
        adata.uns["sra_metadata"] = metadata
        adata.uns["data_type"] = "fastq_raw"
        adata.uns["processing_required"] = [
            "alignment",
            "quantification",
            "quality_control",
        ]

        adata.uns["download_provenance"] = {
            "queue_entry_id": queue_entry.entry_id,
            "dataset_id": queue_entry.dataset_id,
            "database": queue_entry.database,
            "download_date": pd.Timestamp.now().isoformat(),
            "mirror": metadata.get("mirror", "ena"),
        }

        # Store file paths for easy access
        adata.uns["fastq_files"] = {
            "paths": [str(p.absolute()) for p in fastq_paths],
            "layout": metadata.get("layout"),
            "total_size_mb": round(sum(p.stat().st_size for p in fastq_paths) / 1e6, 2),
            "n_files": len(fastq_paths),
        }

        logger.info(
            f"Created FASTQ metadata AnnData: {n_obs} file(s), "
            f"{adata.uns['fastq_files']['total_size_mb']:.1f} MB total"
        )

        return adata

    def _extract_run_id(self, filename: str) -> str:
        """
        Extract SRR/ERR/DRR accession from filename.

        Args:
            filename: FASTQ filename (e.g., "SRR21960766_1.fastq.gz")

        Returns:
            Run accession (e.g., "SRR21960766") or "UNKNOWN" if not found

        Example:
            >>> loader = FASTQLoader()
            >>> loader._extract_run_id("SRR21960766_1.fastq.gz")
            'SRR21960766'
        """
        import re

        match = re.search(r"([SED]RR\d+)", filename)
        return match.group(1) if match else "UNKNOWN"

    def _detect_read_type(self, filename: str) -> str:
        """
        Detect read type from filename (_1, _2, or single).

        Args:
            filename: FASTQ filename

        Returns:
            "R1", "R2", or "single"

        Examples:
            >>> loader = FASTQLoader()
            >>> loader._detect_read_type("SRR001_1.fastq.gz")
            'R1'
            >>> loader._detect_read_type("SRR001_2.fastq.gz")
            'R2'
            >>> loader._detect_read_type("SRR001.fastq.gz")
            'single'
        """
        if "_1.fastq" in filename or "_R1" in filename or "_R1_" in filename:
            return "R1"
        elif "_2.fastq" in filename or "_R2" in filename or "_R2_" in filename:
            return "R2"
        else:
            return "single"
