"""
MassIVE Download Service implementing IDownloadService interface.

This module provides MassIVE repository dataset downloading with FTP protocol support,
integrating with the DownloadOrchestrator pattern for queue-based workflow.

Key features:
- FTP download with directory scanning (PROXI doesn't provide file lists)
- Auto-detection of file formats (MaxQuant, DIA-NN, mzTab)
- Integration with existing proteomics parsers
- Provenance tracking via AnalysisStep IR
- Support for both proteomics and metabolomics data
"""

import ftplib
import re
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
from lobster.tools.providers.massive_provider import MassIVEProvider
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class MassIVEDownloadService(IDownloadService):
    """
    MassIVE repository download service implementing IDownloadService.

    Handles downloading proteomics/metabolomics datasets from MassIVE via FTP protocol.
    Unlike PRIDE, MassIVE's PROXI API doesn't provide file lists, so this service
    implements FTP directory scanning to discover available files.

    Download strategies:
    - RESULT_FIRST: Look for processed results (MaxQuant, DIA-NN)
    - MZML_FIRST: Prioritize mzML files
    - SEARCH_FIRST: Download search engine outputs
    - RAW_FIRST: Download instrument RAW files

    Attributes:
        data_manager: DataManagerV2 instance
        massive_provider: MassIVEProvider for metadata extraction
    """

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize MassIVE download service.

        Args:
            data_manager: DataManagerV2 instance for storage and provenance
        """
        super().__init__(data_manager)
        self.massive_provider = MassIVEProvider(data_manager)
        logger.info("MassIVEDownloadService initialized")

    @classmethod
    def supports_database(cls, database: str) -> bool:
        """
        Check if this service handles MassIVE database.

        Args:
            database: Database identifier (case-insensitive)

        Returns:
            bool: True for 'massive' or 'msv', False otherwise
        """
        return database.lower() in ["massive", "msv"]

    def supported_databases(self) -> List[str]:
        """
        Get list of databases supported by this service.

        Returns:
            List[str]: ["massive", "msv"]
        """
        return ["massive", "msv"]

    @classmethod
    def get_supported_strategies(cls) -> List[str]:
        """
        Get list of download strategies supported by MassIVE service.

        Returns:
            List[str]: Supported strategy names
        """
        return ["RESULT_FIRST", "MZML_FIRST", "SEARCH_FIRST", "RAW_FIRST"]

    def validate_strategy_params(
        self, strategy_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate MassIVE-specific strategy parameters.

        Args:
            strategy_params: Strategy parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        allowed_params = {
            "file_type": str,
            "include_raw": bool,
            "max_file_size_mb": (int, float),
            "scan_depth": int,  # How many subdirectories to scan
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
        Execute MassIVE dataset download from queue entry.

        Workflow:
        1. Scan FTP directory for available files
        2. Select files based on strategy
        3. Download via FTP with retry
        4. Auto-detect parser
        5. Parse to AnnData
        6. Store modality and return

        Args:
            queue_entry: Queue entry with MSV accession
            strategy_override: Optional strategy override

        Returns:
            Tuple of (adata, stats, ir)

        Raises:
            ValueError: If invalid parameters
            RuntimeError: If download or parsing fails
        """
        # Validate database type
        if not self.supports_database(queue_entry.database):
            raise ValueError(
                f"MassIVEDownloadService cannot handle database '{queue_entry.database}'"
            )

        dataset_id = queue_entry.dataset_id
        logger.info(f"Starting MassIVE download for {dataset_id}")

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

            # Step 2: Get FTP base URL
            ftp_base = f"ftp://massive.ucsd.edu/v06/{dataset_id}"

            # Step 3: Scan FTP directory for files
            available_files = self._scan_ftp_directory(ftp_base, max_depth=2)

            if not available_files:
                raise RuntimeError(f"No files found in MassIVE dataset {dataset_id}")

            logger.info(f"Found {len(available_files)} files in FTP directory")

            # Step 4: Select files based on strategy
            selected_files = self._select_files_by_strategy_massive(
                available_files, strategy_name, strategy_params
            )

            if not selected_files:
                raise RuntimeError(
                    f"No suitable files found for strategy '{strategy_name}'"
                )

            # Step 5: Download files
            local_files = []
            download_dir = self.data_manager.workspace_dir / "downloads" / dataset_id
            download_dir.mkdir(parents=True, exist_ok=True)

            for file_info in selected_files[:5]:  # Limit to 5 files
                local_path = self._download_ftp_file(
                    file_info["url"], download_dir / file_info["filename"]
                )
                if local_path:
                    local_files.append(local_path)

            if not local_files:
                raise RuntimeError("All file downloads failed")

            # Step 6: Parse primary file
            primary_file = local_files[0]
            parser = get_parser_for_file(str(primary_file))

            if not parser:
                raise RuntimeError(f"No parser available for file: {primary_file.name}")

            logger.info(f"Parsing {primary_file.name} with {parser.__class__.__name__}")
            adata, parse_stats = parser.parse(str(primary_file))

            # Step 7: Store modality
            modality_name = f"massive_{dataset_id.lower()}_proteomics"
            self.data_manager.modalities[modality_name] = adata

            # Step 8: Build stats
            stats = {
                "dataset_id": dataset_id,
                "database": "massive",
                "modality_name": modality_name,
                "strategy_used": strategy_name,
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars,
                "files_downloaded": len(local_files),
                "parser_used": parser.__class__.__name__,
                **parse_stats,
            }

            # Step 9: Create IR
            ir = self._create_download_ir(dataset_id, strategy_name, strategy_params)

            logger.info(
                f"Successfully downloaded MassIVE dataset {dataset_id}: {adata.shape}"
            )
            return adata, stats, ir

        except Exception as e:
            logger.error(f"Error downloading MassIVE dataset {dataset_id}: {e}")
            raise RuntimeError(f"MassIVE download failed: {str(e)}")

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _scan_ftp_directory(
        self, ftp_base: str, max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Scan MassIVE FTP directory to discover available files.

        Args:
            ftp_base: FTP base URL
            max_depth: Maximum subdirectory depth to scan

        Returns:
            List of file dicts with filename, url, size
        """
        # Parse FTP URL
        if not ftp_base.startswith("ftp://"):
            logger.error(f"Invalid FTP URL: {ftp_base}")
            return []

        url_parts = ftp_base.replace("ftp://", "").split("/", 1)
        if len(url_parts) != 2:
            logger.error(f"Cannot parse FTP URL: {ftp_base}")
            return []

        host, base_path = url_parts

        files = []

        try:
            ftp = ftplib.FTP(host, timeout=60)
            ftp.login()
            ftp.set_pasv(True)

            # Recursive directory scan
            def scan_dir(path: str, depth: int):
                if depth > max_depth:
                    return

                try:
                    items = []
                    ftp.retrlines(f"LIST {path}", items.append)

                    for item in items:
                        # Parse FTP LIST output
                        parts = item.split()
                        if len(parts) < 9:
                            continue

                        permissions = parts[0]
                        size = parts[4] if parts[4].isdigit() else "0"
                        filename = " ".join(parts[8:])

                        item_path = f"{path}/{filename}"

                        if permissions.startswith("d"):
                            # Directory - recurse
                            scan_dir(item_path, depth + 1)
                        else:
                            # File - add to list
                            files.append(
                                {
                                    "filename": filename,
                                    "url": f"ftp://{host}{item_path}",
                                    "size": int(size),
                                    "path": item_path,
                                }
                            )

                except ftplib.error_perm as e:
                    logger.warning(f"Cannot access directory {path}: {e}")

            # Start scanning
            scan_dir(f"/{base_path}", 0)

            ftp.quit()
            logger.info(f"FTP scan complete: {len(files)} files found")

        except Exception as e:
            logger.error(f"Error scanning MassIVE FTP directory: {e}")

        return files

    def _select_files_by_strategy_massive(
        self,
        available_files: List[Dict[str, Any]],
        strategy: str,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Select files to download based on strategy.

        Args:
            available_files: Files discovered via FTP scan
            strategy: Strategy name
            params: Strategy parameters

        Returns:
            List of selected files
        """
        selected = []

        if strategy == "RESULT_FIRST":
            # Look for known result formats
            patterns = [
                r"proteinGroups\.txt$",  # MaxQuant
                r"report\.(tsv|parquet)$",  # DIA-NN
                r".*\.mzTab$",  # mzTab standard
            ]

            for pattern in patterns:
                matching = [
                    f for f in available_files if re.search(pattern, f["filename"])
                ]
                if matching:
                    selected.extend(matching[:1])
                    break

        elif strategy == "MZML_FIRST":
            # mzML files
            selected = [
                f for f in available_files if f["filename"].lower().endswith(".mzml")
            ][:5]

        elif strategy == "SEARCH_FIRST":
            # Search outputs
            search_extensions = [".pepXML", ".pep.xml", ".dat", ".tsv"]
            selected = [
                f
                for f in available_files
                if any(f["filename"].endswith(ext) for ext in search_extensions)
            ][:5]

        elif strategy == "RAW_FIRST":
            # Vendor RAW files
            raw_extensions = [".raw", ".wiff", ".d"]
            selected = [
                f
                for f in available_files
                if any(f["filename"].lower().endswith(ext) for ext in raw_extensions)
            ][:5]

        # Apply size filter
        max_size_mb = params.get("max_file_size_mb")
        if max_size_mb:
            max_bytes = max_size_mb * 1024 * 1024
            selected = [f for f in selected if f.get("size", 0) <= max_bytes]

        return selected

    def _download_ftp_file(
        self, ftp_url: str, output_path: Path, max_retries: int = 3
    ) -> Optional[Path]:
        """
        Download file from MassIVE FTP with retry logic.

        Args:
            ftp_url: FTP URL
            output_path: Local path to save file
            max_retries: Maximum retry attempts

        Returns:
            Path to downloaded file, or None if failed
        """
        # Parse FTP URL
        if not ftp_url.startswith("ftp://"):
            logger.error(f"Invalid FTP URL: {ftp_url}")
            return None

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
                ftp.set_pasv(True)

                # Get file size
                try:
                    file_size = ftp.size(remote_path)
                    logger.info(
                        f"File size: {file_size / 1024 / 1024:.1f} MB"
                        if file_size
                        else "Size unknown"
                    )
                except:
                    file_size = None

                # Download
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
                    wait_time = 2**attempt
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
        Create AnalysisStep IR for MassIVE download provenance.

        Args:
            dataset_id: MSV accession
            strategy: Strategy name used
            params: Strategy parameters

        Returns:
            AnalysisStep for provenance tracking
        """
        return AnalysisStep(
            operation="massive.download.download_dataset",
            tool_name="MassIVEDownloadService",
            description=f"Download MassIVE dataset {dataset_id}",
            library="lobster.services.data_access.massive_download_service",
            code_template="""# MassIVE dataset download
from lobster.services.data_access.massive_download_service import MassIVEDownloadService
from lobster.tools.providers.massive_provider import MassIVEProvider
from lobster.core.data_manager_v2 import DataManagerV2

# Initialize
dm = DataManagerV2(workspace_dir="./workspace")
provider = MassIVEProvider(dm)
service = MassIVEDownloadService(dm)

# Scan FTP directory and download files
# Parse with appropriate proteomics parser
# Store as modality
""",
            imports=[
                "from lobster.services.data_access.massive_download_service import MassIVEDownloadService",
                "from lobster.tools.providers.massive_provider import MassIVEProvider",
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
                    description="MassIVE accession (MSV format)",
                ),
            },
            input_entities=[dataset_id],
            output_entities=[f"massive_{dataset_id.lower()}_proteomics"],
        )
