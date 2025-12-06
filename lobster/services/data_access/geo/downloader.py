"""
GEO database downloader and parser.

This module handles downloading and parsing data from the Gene Expression Omnibus (GEO)
database, providing structured access to gene expression datasets.
"""

import ftplib
import hashlib
import os
import random
import re
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class GEODownloadError(Exception):
    """Custom exception for GEO download errors."""

    pass


class GEODownloadManager:
    """
    Handles downloading files from GEO database.

    This class manages connections to the GEO FTP server and handles
    downloading and caching SOFT and supplementary files.
    """

    def __init__(
        self, cache_dir: Optional[str] = None, console: Optional[Console] = None
    ):
        """
        Initialize the download manager.

        Args:
            cache_dir: Directory to cache downloaded files (REQUIRED)
            console: Rich console instance for display (creates new if None)

        Raises:
            ValueError: If cache_dir is not provided
        """
        if cache_dir is None:
            raise ValueError(
                "cache_dir is required. Pass workspace_path/cache/geo from DataManagerV2. "
                "Example: GEODownloadManager(cache_dir=data_manager.cache_dir / 'geo')"
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; GEODownloader/1.0)"}
        )
        self.console = console or Console()

    def construct_geo_urls(self, gse_id: str) -> Dict[str, str]:
        """
        Construct GEO SOFT FTP URLs.

        Args:
            gse_id: GEO series ID (e.g., GSE109564)

        Returns:
            dict: Dictionary of URLs for different file types
        """
        # Extract number and determine series folder
        gse_str = gse_id[3:]  # Remove 'GSE'
        gse_num_base = int(gse_str[:-3])  # Remove last 3 digits
        series_folder = f"GSE{gse_num_base}nnn"

        base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series_folder}/{gse_id}"

        urls = {
            "soft_gz": f"{base_url}/soft/{gse_id}_family.soft.gz",
            "soft": f"{base_url}/soft/{gse_id}_family.soft",
            "suppl_folder": f"{base_url}/suppl/",
            "tar_file": f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse_id}&format=file",
        }

        return urls

    def get_supplementary_files(self, gse_id: str) -> List[str]:
        """
        Get list of supplementary files from the suppl folder.

        Args:
            gse_id: GEO series ID

        Returns:
            list: List of supplementary file URLs
        """
        try:
            urls = self.construct_geo_urls(gse_id)
            suppl_url = urls["suppl_folder"]

            response = self.session.get(suppl_url, timeout=30)
            response.raise_for_status()

            # Parse HTML directory listing to find files
            file_links = re.findall(
                r'href="([^"]*\.(?:txt|csv|xlsx|h5|gz|bz2))"',
                response.text,
                re.IGNORECASE,
            )

            # Full URLs for supplementary files
            suppl_files = [
                suppl_url + link for link in file_links if not link.startswith("..")
            ]

            logger.info(f"Found {len(suppl_files)} supplementary files")
            return suppl_files

        except Exception as e:
            logger.warning(f"Could not get supplementary files list: {e}")
            return []

    def download_tar_file(self, gse_id: str) -> Optional[Path]:
        """
        Download and extract TAR file with raw data.

        Args:
            gse_id: GEO series ID

        Returns:
            Path: Path to extracted data directory, or None if download failed
        """
        try:
            urls = self.construct_geo_urls(gse_id)
            tar_url = urls["tar_file"]

            # Define the local tar file path and extraction directory
            tar_file_path = self.cache_dir / f"{gse_id}_raw.tar"
            extract_dir = self.cache_dir / f"{gse_id}_extracted"

            # Check if already extracted
            if extract_dir.exists() and any(extract_dir.iterdir()):
                logger.info(f"Using cached extracted TAR data: {extract_dir}")
                return extract_dir

            # Download the TAR file
            logger.info(f"Downloading TAR file from: {tar_url}")
            if not self.download_file(tar_url, tar_file_path):
                logger.warning(f"Failed to download TAR file for {gse_id}")
                return None

            # Extract the TAR file
            try:
                if not extract_dir.exists():
                    extract_dir.mkdir(parents=True)

                logger.info(f"Extracting TAR file to: {extract_dir}")
                with tarfile.open(tar_file_path, "r") as tar:
                    # Security check: Validate tar file members before extraction
                    # to prevent path traversal attacks
                    def is_safe_member(member):
                        # Prevent absolute paths and path traversal
                        member_path = Path(member.name)
                        try:
                            # Path.resolve() will fail on path traversal attempts
                            # Check that the resolved path is within the extract_dir
                            target_path = (extract_dir / member_path).resolve()
                            common_path = Path(
                                os.path.commonpath([extract_dir.resolve(), target_path])
                            )
                            is_safe = common_path == extract_dir.resolve()
                            if not is_safe:
                                logger.warning(
                                    f"Skipping potentially unsafe member: {member.name}"
                                )
                            return is_safe
                        except (ValueError, RuntimeError):
                            # Path traversal attempt
                            logger.warning(
                                f"Skipping invalid path in TAR: {member.name}"
                            )
                            return False

                    # Extract only safe members with progress tracking
                    safe_members = [m for m in tar.getmembers() if is_safe_member(m)]
                    logger.debug(
                        f"Extracting {len(safe_members)} validated members from TAR"
                    )

                    # Setup progress tracking for extraction
                    sum(member.size for member in safe_members)
                    progress_columns = [
                        BarColumn(),
                        "•",
                        "{task.percentage:>3.0f}%",
                        "•",
                        "{task.completed}/{task.total} files",
                        "•",
                        TimeElapsedColumn(),
                        "•",
                        TimeRemainingColumn(),
                    ]

                    with Progress(*progress_columns, console=self.console) as progress:
                        extract_task = progress.add_task(
                            f"Extracting {gse_id} files", total=len(safe_members)
                        )

                        for i, member in enumerate(safe_members):
                            tar.extract(member, path=extract_dir)
                            progress.update(
                                extract_task,
                                completed=i + 1,
                                description=f"Extracting {Path(member.name).name[:30]}...",
                            )

                # Clean up the tar file after extraction
                tar_file_path.unlink()

                logger.info(f"Successfully extracted TAR file to: {extract_dir}")
                return extract_dir

            except tarfile.ReadError:
                logger.warning(
                    f"The downloaded file is not a valid TAR archive: {tar_file_path}"
                )
                return None

        except Exception as e:
            logger.error(f"Error downloading/extracting TAR file: {e}")
            return None

    def find_expression_file_in_tar(self, tar_dir: Path) -> Optional[Path]:
        """
        Find a suitable expression data file in the extracted TAR directory.

        Args:
            tar_dir: Path to the extracted TAR directory

        Returns:
            Path: Path to the best expression data file, or None if not found
        """
        try:
            if not tar_dir.exists():
                return None

            # Log the contents of the TAR directory to help with debugging
            logger.debug(f"Listing contents of TAR directory: {tar_dir}")
            for item in tar_dir.iterdir():
                if item.is_file():
                    logger.debug(f"File: {item.name} ({item.stat().st_size} bytes)")
                elif item.is_dir():
                    logger.debug(f"Directory: {item.name}")

            # Recursively find all files
            all_files = list(tar_dir.glob("**/*"))
            logger.debug(f"Found {len(all_files)} total files in TAR extraction")

            # Define priority keywords and extensions
            # Include specific filename patterns seen in datasets like GSE111672
            keyword_priority = [
                "expmat",
                "expression-matrix",
                "count",
                "expr",
                "matrix",
                "dge",
                "htseq",
                "featurecount",
                "rsem",
                "filtered",
                "pdac",
                "indrop",
            ]
            ext_priority = [
                ".txt.gz",
                ".txt",
                ".tsv",
                ".csv",
                ".mtx",
                ".h5",
                ".h5ad",
                ".rds",
            ]

            # Check for common expression matrix filenames for this specific dataset
            common_patterns = [
                "*expmat*",
                "*expression*matrix*",
                "*counts*",
                "*filtered*",
                "*_matrix*",
                "*_expmat*",
                "*_dge*",
                "*-dge*",
                "*_htseq*",
                "*_PDAC*",
                "*_indrop*",
            ]

            # Look for files matching common patterns first
            for pattern in common_patterns:
                matches = list(tar_dir.glob(f"**/{pattern}"))
                for file_path in matches:
                    if file_path.is_file() and any(
                        file_path.name.lower().endswith(ext) for ext in ext_priority
                    ):
                        logger.info(
                            f"Found expression file in TAR matching pattern '{pattern}': {file_path}"
                        )
                        return file_path

            # First look for files with priority keywords and extensions
            for keyword in keyword_priority:
                for ext in ext_priority:
                    for file_path in all_files:
                        if (
                            file_path.is_file()
                            and keyword in file_path.name.lower()
                            and file_path.name.lower().endswith(ext)
                        ):
                            logger.info(
                                f"Found expression file in TAR with keyword '{keyword}': {file_path}"
                            )
                            return file_path

            # Look for larger text files (expression matrices tend to be large)
            large_text_files = []
            for file_path in all_files:
                if (
                    file_path.is_file()
                    and any(
                        file_path.name.lower().endswith(ext) for ext in ext_priority
                    )
                    and file_path.stat().st_size > 100000
                ):  # Files larger than 100KB
                    large_text_files.append((file_path, file_path.stat().st_size))

            if large_text_files:
                # Sort by size, largest first
                large_text_files.sort(key=lambda x: x[1], reverse=True)
                logger.info(
                    f"Found large text file that may be an expression matrix: {large_text_files[0][0]}"
                )
                return large_text_files[0][0]

            # Then just look for files with priority extensions
            for ext in ext_priority:
                for file_path in all_files:
                    if file_path.is_file() and file_path.name.lower().endswith(ext):
                        logger.info(
                            f"Found potential expression file in TAR with extension '{ext}': {file_path}"
                        )
                        return file_path

            logger.warning("No suitable expression file found in TAR directory")
            return None

        except Exception as e:
            logger.error(f"Error finding expression file in TAR: {e}")
            return None

    def download_supplementary_data(self, gse_id: str) -> Optional[Path]:
        """
        Download supplementary expression data.

        Args:
            gse_id: GEO series ID

        Returns:
            Path: Path to downloaded file, or None if download failed
        """
        try:
            suppl_files = self.get_supplementary_files(gse_id)

            # Log all available supplementary files for debugging
            logger.debug(f"Available supplementary files for {gse_id}:")
            for file_url in suppl_files:
                file_name = file_url.split("/")[-1]
                logger.debug(f" - {file_name}")

            # Enhanced keywords for expression matrices
            keyword_priority = [
                "expmat",
                "filtered-expmat",
                "expression-matrix",
                "dge",
                "count",
                "matrix",
                "expression",
                "pdac-",
                "indrop-filtered",
            ]

            # File extensions in priority order
            ext_priority = [".txt.gz", ".csv.gz", ".tsv.gz", ".txt", ".csv", ".tsv"]

            # First, look for files that match specific expression matrix patterns for GSE111672
            selected_file = None
            gse111672_patterns = [
                "pdac-a-indrop-filtered-expmat",
                "pdac-b-indrop-filtered-expmat",
                "filtered-expmat",
                "expression-matrix",
            ]

            # For GSE111672 specifically, target the filtered expression matrices
            if gse_id == "GSE111672":
                for pattern in gse111672_patterns:
                    for file_url in suppl_files:
                        if pattern in file_url.lower():
                            selected_file = file_url
                            logger.info(
                                f"Found GSE111672 specific matrix file: {file_url}"
                            )
                            break
                    if selected_file:
                        break

            # If not found yet, use the general approach with enhanced keywords
            if not selected_file:
                # First look for the files with the largest size (likely to be expression matrices)
                file_sizes = {}
                for file_url in suppl_files:
                    file_name = file_url.split("/")[-1]
                    local_path = self.cache_dir / f"{gse_id}_suppl_{file_name}"

                    # Check if file is already cached and get its size
                    if local_path.exists():
                        file_sizes[file_url] = local_path.stat().st_size

                # Look for files with keywords in priority order
                for keyword in keyword_priority:
                    for ext in ext_priority:
                        for file_url in suppl_files:
                            fname = file_url.lower()
                            # Skip small files like barcodes or readme
                            if "barcode" in fname or "readme" in fname:
                                continue
                            if keyword in fname and fname.endswith(ext):
                                selected_file = file_url
                                logger.info(
                                    f"Found supplementary file with keyword '{keyword}': {file_url}"
                                )
                                break
                        if selected_file:
                            break
                    if selected_file:
                        break

                # If still not found, try to find the largest compressed text file
                if not selected_file and file_sizes:
                    # Sort by size, largest first
                    sorted_files = sorted(
                        file_sizes.items(), key=lambda x: x[1], reverse=True
                    )
                    for file_url, size in sorted_files:
                        if any(file_url.lower().endswith(ext) for ext in ext_priority):
                            selected_file = file_url
                            logger.info(
                                f"Selected largest compressed text file: {file_url} ({size} bytes)"
                            )
                            break

                # Last resort - just find any file with preferred extensions
                if not selected_file:
                    for ext in ext_priority:
                        for file_url in suppl_files:
                            fname = file_url.lower()
                            # Skip small auxiliary files
                            if (
                                "barcode" in fname
                                or "readme" in fname
                                or "gel_barcode" in fname
                            ):
                                continue
                            if fname.endswith(ext):
                                selected_file = file_url
                                logger.info(
                                    f"Found supplementary file with extension '{ext}': {file_url}"
                                )
                                break
                        if selected_file:
                            break

            if not selected_file:
                logger.warning("No suitable supplementary expression file found")
                return None

            # Download the file
            file_name = selected_file.split("/")[-1]
            local_path = self.cache_dir / f"{gse_id}_suppl_{file_name}"

            # Check if already cached
            if local_path.exists():
                logger.info(f"Using cached supplementary file: {local_path}")
                return local_path

            logger.info(f"Downloading supplementary file: {selected_file}")
            if self.download_file(selected_file, local_path):
                return local_path

            return None

        except Exception as e:
            logger.error(f"Error downloading supplementary data: {e}")
            return None

    def download_file(
        self, url: str, local_path: Path, description: str = None
    ) -> bool:
        """
        Download file from URL to local path with progress tracking.
        Supports both HTTP/HTTPS and FTP protocols.

        Args:
            url: URL to download from (HTTP/HTTPS/FTP)
            local_path: Path to save file to
            description: Optional description for the progress bar

        Returns:
            bool: True if download succeeded, False otherwise
        """
        try:
            logger.debug(f"Downloading from: {url}")

            # Parse the URL to determine protocol
            parsed_url = urlparse(url)

            # Create a meaningful description for the progress bar
            if not description:
                filename = local_path.name
                if len(filename) > 40:
                    filename = filename[:37] + "..."
                description = f"Downloading {filename}"

            # Handle FTP URLs
            if parsed_url.scheme.lower() == "ftp":
                return self._download_ftp(url, local_path, description)

            # Handle HTTP/HTTPS URLs (existing functionality)
            else:
                return self._download_http(url, local_path, description)

        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return False

    def _download_http(self, url: str, local_path: Path, description: str) -> bool:
        """
        Download file from HTTP/HTTPS URL with progress tracking.

        Args:
            url: HTTP/HTTPS URL to download from
            local_path: Path to save file to
            description: Description for the progress bar

        Returns:
            bool: True if download succeeded, False otherwise
        """
        try:
            # Get file size for progress tracking
            head_response = self.session.head(url, timeout=30)
            file_size = int(head_response.headers.get("content-length", 0))

            # Start the actual download with progress tracking
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Create progress columns for a rich display
            progress_columns = [
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
            ]

            # Download with progress tracking
            with Progress(*progress_columns, console=self.console) as progress:
                # Create the download task
                task_id = progress.add_task(description, total=file_size)

                with open(local_path, "wb") as f:
                    downloaded = 0
                    # Use a slightly larger chunk size for better performance
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress.update(task_id, completed=downloaded)

            logger.info(f"Downloaded to: {local_path}")

            # Validate gzip files before considering download successful
            if local_path.name.endswith(".gz"):
                if not self._validate_gzip_integrity(local_path):
                    logger.error(
                        f"Gzip validation failed, removing corrupted file: {local_path}"
                    )
                    local_path.unlink()
                    return False

            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to download HTTP {url}: {e}")
            return False

    def _calculate_md5(self, file_path: Path) -> str:
        """
        Calculate MD5 checksum for file integrity verification.

        Args:
            file_path: Path to file to checksum

        Returns:
            str: Hexadecimal MD5 checksum
        """
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read in 8KB chunks for memory efficiency
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _chunked_ftp_download(
        self,
        ftp: ftplib.FTP,
        remote_path: str,
        local_path: Path,
        file_size: int,
        description: str,
        chunk_size: int = 8192,
    ) -> bool:
        """
        Download file from FTP in chunks to prevent corruption during large transfers.

        Args:
            ftp: Connected FTP instance
            remote_path: Remote file path on FTP server
            local_path: Local destination path
            file_size: Total file size for progress tracking
            description: Description for progress bar
            chunk_size: Size of chunks to download (default: 8KB)

        Returns:
            bool: True if download succeeded, False otherwise
        """
        try:
            # Create progress columns for rich display
            progress_columns = [
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
            ]

            # Download with progress tracking
            with Progress(*progress_columns, console=self.console) as progress:
                task_id = progress.add_task(description, total=file_size)

                with open(local_path, "wb") as f:
                    downloaded = 0

                    def callback(data):
                        nonlocal downloaded
                        f.write(data)
                        downloaded += len(data)
                        progress.update(task_id, completed=downloaded)

                    # Set binary mode and download with specified chunk size
                    ftp.voidcmd("TYPE I")  # Set binary mode
                    ftp.retrbinary(
                        f"RETR {remote_path}", callback, blocksize=chunk_size
                    )

            return True

        except Exception as e:
            logger.error(f"Chunked FTP download failed: {e}")
            return False

    def _download_with_retry(
        self,
        url: str,
        local_path: Path,
        description: str,
        max_retries: int = 3,
        md5_checksum: Optional[str] = None,
    ) -> bool:
        """
        Download file with retry logic and integrity verification.

        Implements exponential backoff for transient network errors and validates
        downloaded files using gzip integrity checks and optional MD5 verification.

        Args:
            url: FTP URL to download from
            local_path: Local destination path
            description: Description for progress bar
            max_retries: Maximum retry attempts (default: 3)
            md5_checksum: Optional MD5 checksum for verification

        Returns:
            bool: True if download successful and validated, False otherwise
        """
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        username = parsed_url.username or "anonymous"
        password = parsed_url.password or "anonymous@"
        filepath = parsed_url.path

        for attempt in range(1, max_retries + 1):
            try:
                # Exponential backoff with jitter: 2^attempt ± 20% seconds
                # Jitter prevents synchronized retry storms (thundering herd problem)
                if attempt > 1:
                    base_delay = 2**attempt
                    jitter = base_delay * 0.2 * (2 * random.random() - 1)  # ±20%
                    wait_time = base_delay + jitter
                    logger.info(
                        f"Retry attempt {attempt}/{max_retries} after {wait_time:.1f}s delay "
                        f"(base: {base_delay}s, jitter: {jitter:+.1f}s)"
                    )
                    time.sleep(wait_time)

                # Connect to FTP server
                ftp = ftplib.FTP()
                ftp.connect(hostname, parsed_url.port or 21, timeout=30)
                ftp.login(username, password)

                try:
                    # Get file size for progress tracking
                    file_size = ftp.size(filepath)
                    if file_size is None:
                        file_size = 0

                    # Perform chunked download (8KB chunks for stability)
                    success = self._chunked_ftp_download(
                        ftp,
                        filepath,
                        local_path,
                        file_size,
                        description,
                        chunk_size=8192,
                    )

                    if not success:
                        logger.warning(
                            f"Chunked download failed on attempt {attempt}/{max_retries}"
                        )
                        if local_path.exists():
                            local_path.unlink()
                        continue

                    # Verify file size matches expected size
                    # CRITICAL: Detects truncated files from FTP throttling
                    if file_size > 0 and local_path.exists():
                        actual_size = local_path.stat().st_size
                        if actual_size != file_size:
                            logger.warning(
                                f"File size mismatch on attempt {attempt}/{max_retries}: "
                                f"expected {file_size} bytes, got {actual_size} bytes. "
                                f"File likely truncated by FTP throttling."
                            )
                            local_path.unlink()
                            continue

                    # Verify MD5 checksum if provided
                    if md5_checksum:
                        calculated_md5 = self._calculate_md5(local_path)
                        if calculated_md5 != md5_checksum:
                            logger.warning(
                                f"MD5 mismatch on attempt {attempt}: "
                                f"expected {md5_checksum}, got {calculated_md5}"
                            )
                            local_path.unlink()
                            continue

                    # Verify gzip integrity for compressed files
                    if local_path.name.endswith(".gz"):
                        if not self._validate_gzip_integrity(local_path):
                            logger.warning(
                                f"Gzip validation failed on attempt {attempt}/{max_retries}"
                            )
                            local_path.unlink()
                            continue

                    # Success - all validations passed
                    logger.info(
                        f"Download successful: {local_path.name} "
                        f"({file_size / (1024**2):.1f} MB)"
                    )
                    return True

                finally:
                    ftp.quit()

            except (*ftplib.all_errors, OSError) as e:
                logger.error(f"Download attempt {attempt}/{max_retries} failed: {e}")
                if local_path.exists():
                    local_path.unlink()

                if attempt == max_retries:
                    logger.error(
                        f"All {max_retries} download attempts failed for {url}"
                    )
                    return False

        return False

    def _download_ftp(self, url: str, local_path: Path, description: str) -> bool:
        """
        Download file from FTP URL with retry logic and progress tracking.

        This method wraps _download_with_retry() to provide automatic retry
        with exponential backoff for transient network errors and corrupted downloads.

        Args:
            url: FTP URL to download from
            local_path: Path to save file to
            description: Description for the progress bar

        Returns:
            bool: True if download succeeded, False otherwise
        """
        # Delegate to retry wrapper with exponential backoff
        return self._download_with_retry(url, local_path, description)

    def _validate_gzip_integrity(self, file_path: Path) -> bool:
        """
        Validate gzip file can be fully decompressed.

        This prevents corrupted gzip files from poisoning the cache by
        attempting to decompress the entire file in chunks.

        Args:
            file_path: Path to gzip file to validate

        Returns:
            bool: True if file can be fully decompressed, False otherwise
        """
        try:
            import gzip

            # Read entire file in chunks to verify complete decompression
            # Using 32KB chunks for memory efficiency
            with gzip.open(file_path, "rb") as f:
                while _ := f.read(32768):
                    pass

            logger.debug(f"Gzip validation passed: {file_path.name}")
            return True

        except (OSError, gzip.BadGzipFile, EOFError) as e:
            logger.warning(f"Gzip validation failed for {file_path.name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating {file_path.name}: {e}")
            return False

    # FIXME remove soonish
    def download_geo_data(
        self, gse_id: str
    ) -> Tuple[Optional[Path], Optional[Union[Path, Dict[str, Path]]]]:
        """
        Download GEO SOFT data and data files (supplementary or TAR).

        This method tries multiple data sources in order:
        1. SOFT file (always downloaded for metadata)
        2. Supplementary files from the FTP server
        3. TAR file with raw data

        Args:
            gse_id: GEO series ID

        Returns:
            tuple: Paths to SOFT file and data file/directory (or dict with data sources)
        """
        urls = self.construct_geo_urls(gse_id)
        soft_file = None
        data_sources = {}

        # Download SOFT file first (always needed for metadata)
        for file_type in ["soft_gz"]:
            url = urls[file_type]
            local_path = self.cache_dir / f"{gse_id}_{file_type}.gz"

            # Check if already cached
            if local_path.exists():
                logger.info(f"Using cached SOFT file: {local_path}")
                soft_file = local_path
                break

            if self.download_file(url, local_path):
                soft_file = local_path
                break

        if not soft_file:
            logger.error(f"Failed to download SOFT file for {gse_id}")
            return None, None

        # Try to get supplementary files first
        logger.info("Attempting to download supplementary files...")
        suppl_file = self.download_supplementary_data(gse_id)
        if suppl_file:
            data_sources["supplementary"] = suppl_file
            logger.info(f"Found suitable supplementary file: {suppl_file}")

        # If no supplementary file or we want to try TAR anyway
        logger.info("Attempting to download TAR file with raw data...")
        tar_dir = self.download_tar_file(gse_id)
        if tar_dir:
            # Find a suitable expression file in the extracted TAR
            expr_file = self.find_expression_file_in_tar(tar_dir)
            if expr_file:
                data_sources["tar"] = expr_file
                logger.info(f"Found suitable expression file in TAR: {expr_file}")
            else:
                # Just store the directory for later processing
                data_sources["tar_dir"] = tar_dir
                logger.info(f"Extracted TAR directory: {tar_dir}")

        # Return the SOFT file and data sources
        if data_sources:
            # If there's only one source, return it directly
            if len(data_sources) == 1:
                return soft_file, next(iter(data_sources.values()))
            # Otherwise return the dictionary of sources
            return soft_file, data_sources

        logger.warning(f"No data files found for {gse_id}")
        return soft_file, None
