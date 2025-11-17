"""
Professional GEO data service using GEOparse with modular fallback architecture.

This service provides a unified interface for downloading and processing
data from the Gene Expression Omnibus (GEO) database using a layered approach:
1. Primary: GEOparse library for standard operations
2. Fallback: Specialized downloader and parser for complex cases
3. Integration: Full DataManagerV2 compatibility with comprehensive error handling
"""

import ftplib
import json
import os
import re
import tarfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd

try:
    import GEOparse
except ImportError:
    GEOparse = None

from lobster.agents.data_expert_assistant import DataExpertAssistant
from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.exceptions import (
    FeatureNotImplementedError,
    UnsupportedPlatformError,
)

# Import bulk RNA-seq and adapter support for quantification files
from lobster.tools.bulk_rnaseq_service import BulkRNASeqService

# Import helper modules for fallback functionality
from lobster.tools.geo_downloader import GEODownloadManager
from lobster.tools.geo_parser import GEOParser
from lobster.tools.pipeline_strategy import (
    PipelineStrategyEngine,
    PipelineType,
    create_pipeline_context,
)
from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context, handle_ssl_error

logger = get_logger(__name__)


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                        DATA STRUCTURES AND ENUMS                          ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████


class GEODataSource(Enum):
    """Enumeration of data source types for GEO downloads."""

    GEOPARSE = "geoparse"
    SOFT_FILE = "soft_file"
    SUPPLEMENTARY = "supplementary"
    TAR_ARCHIVE = "tar_archive"
    SAMPLE_MATRICES = "sample_matrices"


class GEODataType(Enum):
    """Enumeration of data types for GEO datasets."""

    SINGLE_CELL = "single_cell"
    BULK = "bulk"
    MIXED = "mixed"


@dataclass
class DownloadStrategy:
    """Configuration for GEO download preferences and fallback options."""

    prefer_geoparse: bool = True
    allow_fallback: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
    prefer_supplementary: bool = False
    force_tar_extraction: bool = False


@dataclass
class GEOResult:
    """Result wrapper containing data, metadata, and processing information."""

    data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None
    source: GEODataSource = GEODataSource.GEOPARSE
    processing_info: Dict[str, Any] = None
    success: bool = False
    error_message: Optional[str] = None


class GEOServiceError(Exception):
    """Custom exception for GEO service errors."""

    pass


class GEOFallbackError(Exception):
    """Custom exception for fallback mechanism failures."""

    pass


class PlatformCompatibility(Enum):
    """Platform support status for early validation."""

    SUPPORTED = "supported"  # RNA-seq, fully supported
    UNSUPPORTED = "unsupported"  # Microarrays, clear rejection
    EXPERIMENTAL = "experimental"  # Partial support, warning only
    UNKNOWN = "unknown"  # Not in registry, conservative approach


# Comprehensive Platform Registry for Early Validation
# This registry enables rejecting unsupported platforms BEFORE downloading files
PLATFORM_REGISTRY: Dict[str, PlatformCompatibility] = {
    # === SUPPORTED RNA-SEQ PLATFORMS ===
    # Illumina RNA-seq platforms
    "GPL16791": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 2500
    "GPL18573": PlatformCompatibility.SUPPORTED,  # Illumina NextSeq 500
    "GPL20301": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 4000
    "GPL21290": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 3000
    "GPL24676": PlatformCompatibility.SUPPORTED,  # Illumina NovaSeq 6000
    "GPL13112": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 2000 (mouse)
    "GPL11154": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 2000 (humanb)
    "GPL10999": PlatformCompatibility.SUPPORTED,  # Illumina Genome Analyzer IIx
    "GPL9115": PlatformCompatibility.SUPPORTED,  # Illumina Genome Analyzer II
    "GPL9052": PlatformCompatibility.SUPPORTED,  # Illumina Genome Analyzer
    # Single-cell platforms
    "GPL24247": PlatformCompatibility.SUPPORTED,  # 10X Chromium (NovaSeq)
    "GPL26966": PlatformCompatibility.SUPPORTED,  # 10X Chromium (HiSeq X)
    "GPL21103": PlatformCompatibility.SUPPORTED,  # Illumina HiSeq 2500 (single-cell)
    "GPL19057": PlatformCompatibility.SUPPORTED,  # Illumina NextSeq 500 (single-cell)
    # === UNSUPPORTED MICROARRAY PLATFORMS ===
    # Affymetrix arrays
    "GPL570": PlatformCompatibility.UNSUPPORTED,  # Affymetrix U133 Plus 2.0
    "GPL96": PlatformCompatibility.UNSUPPORTED,  # Affymetrix U133A
    "GPL97": PlatformCompatibility.UNSUPPORTED,  # Affymetrix U133B
    "GPL571": PlatformCompatibility.UNSUPPORTED,  # Affymetrix U133 A 2.0
    "GPL1352": PlatformCompatibility.UNSUPPORTED,  # Affymetrix U133 A2
    "GPL6244": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Gene 1.0 ST
    "GPL6246": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Mouse Gene 1.0 ST
    "GPL6247": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Rat Gene 1.0 ST
    "GPL91": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Mu11KsubA
    "GPL92": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Mu11KsubB
    "GPL339": PlatformCompatibility.UNSUPPORTED,  # Affymetrix MOE430A
    "GPL340": PlatformCompatibility.UNSUPPORTED,  # Affymetrix MOE430B
    "GPL8321": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Mouse Genome 430A 2.0
    "GPL1261": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Mouse Genome 430 2.0
    # Illumina BeadArray (microarray, NOT RNA-seq)
    "GPL10558": PlatformCompatibility.UNSUPPORTED,  # Illumina HumanHT-12 V4.0
    "GPL6947": PlatformCompatibility.UNSUPPORTED,  # Illumina HumanHT-12 V3.0
    "GPL6883": PlatformCompatibility.UNSUPPORTED,  # Illumina HumanRef-8 V3.0
    "GPL6887": PlatformCompatibility.UNSUPPORTED,  # Illumina MouseRef-8 V2.0
    "GPL6885": PlatformCompatibility.UNSUPPORTED,  # Illumina MouseRef-8 V1.1
    "GPL6102": PlatformCompatibility.UNSUPPORTED,  # Illumina human-6 V2.0
    "GPL6104": PlatformCompatibility.UNSUPPORTED,  # Illumina mouse Ref-8 V2.0
    # Agilent microarrays
    "GPL6480": PlatformCompatibility.UNSUPPORTED,  # Agilent-014850
    "GPL13497": PlatformCompatibility.UNSUPPORTED,  # Agilent-026652
    "GPL17077": PlatformCompatibility.UNSUPPORTED,  # Agilent-039494
    "GPL4133": PlatformCompatibility.UNSUPPORTED,  # Agilent-014850 Whole Human Genome
    "GPL1708": PlatformCompatibility.UNSUPPORTED,  # Agilent-012391 Whole Human Genome
    "GPL7202": PlatformCompatibility.UNSUPPORTED,  # Agilent-014868 Whole Mouse Genome
    # Other microarray platforms
    "GPL341": PlatformCompatibility.UNSUPPORTED,  # Affymetrix RG_U34A
    "GPL85": PlatformCompatibility.UNSUPPORTED,  # Affymetrix RG_U34B
    "GPL1355": PlatformCompatibility.UNSUPPORTED,  # Affymetrix Rat Genome 230 2.0
}

# Keyword patterns for unknown platform detection
UNSUPPORTED_KEYWORDS = [
    "affymetrix",
    "agilent",
    "beadarray",
    "beadchip",
    "genechip",
    "microarray",
    "array",
    "snp chip",
    "exon array",
    "gene chip",
]

SUPPORTED_KEYWORDS = [
    "rna-seq",
    "rnaseq",
    "rna seq",
    "illumina hiseq",
    "illumina nextseq",
    "illumina novaseq",
    "10x",
    "chromium",
    "single cell",
    "single-cell",
    "sequencing",
]


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                           MAIN SERVICE CLASS                               ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████


class GEOService:
    """
    Professional service for accessing and processing GEO data using GEOparse and DataManagerV2.

    This class provides a high-level interface for working with GEO data,
    handling the downloading, parsing, and processing of datasets using GEOparse
    and storing them as modalities in the DataManagerV2 system.
    """

    def __init__(
        self,
        data_manager: DataManagerV2,
        cache_dir: Optional[str] = None,
        console=None,
        email: Optional[str] = None,
    ):
        """
        Initialize the GEO service with modular architecture.

        Args:
            data_manager: DataManagerV2 instance for storing processed data as modalities
            cache_dir: Directory to cache downloaded files
            console: Rich console instance for display (creates new if None)
            email: Optional email for NCBI Entrez (for backward compatibility, not currently used)
        """
        if GEOparse is None:
            raise ImportError(
                "GEOparse is required but not installed. Please install with: pip install GEOparse"
            )

        self.data_manager = data_manager
        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.data_manager.cache_dir / "geo"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.console = console

        # Initialize helper services for fallback functionality
        self.geo_downloader = GEODownloadManager(
            cache_dir=str(self.cache_dir), console=self.console
        )
        self.geo_parser = GEOParser()

        # Initialize the pipeline strategy engine
        self.pipeline_engine = PipelineStrategyEngine()

        # Default download strategy
        self.download_strategy = DownloadStrategy()

        logger.info(
            "GEOService initialized with modular architecture: GEOparse + dynamic pipeline strategy"
        )

    # ████████████████████████████████████████████████████████████████████████████████
    # ██                                                                            ██
    # ██                      RETRY LOGIC WITH EXPONENTIAL BACKOFF                 ██
    # ██                                                                            ██
    # ████████████████████████████████████████████████████████████████████████████████

    def _retry_with_backoff(
        self,
        operation: Callable[[], Any],
        operation_name: str,
        max_retries: int = 5,
        base_delay: float = 1.0,
        is_ftp: bool = False,
    ) -> Optional[Any]:
        """
        Retry operation with exponential backoff, jitter, and progress reporting.

        Implements production-grade retry logic for transient failures:
        - Exponential backoff: 1s, 2s, 4s, 8s, 16s
        - Jitter: 0.5-1.5x random multiplier (prevents thundering herd)
        - Progress reporting: Updates console during retry delays
        - FTP optimization: Reduced retry count for fast-failing FTP
        - Conservative return: Returns None rather than raising on final failure

        Args:
            operation: Function to retry (must be idempotent)
            operation_name: Human-readable name for logging
            max_retries: Maximum number of attempts (default: 5, FTP: 2)
            base_delay: Base delay in seconds (default: 1.0)
            is_ftp: Whether this is an FTP operation (affects retry count)

        Returns:
            Result of operation or None if all retries fail

        Example:
            result = self._retry_with_backoff(
                operation=lambda: requests.get(url),
                operation_name=f"Download {geo_id}",
                max_retries=5,
                is_ftp=False
            )
            if result is None:
                raise DownloadError(f"Failed after {max_retries} attempts")
        """
        import random

        import requests

        # FTP connections often fail permanently, not transiently
        if is_ftp:
            max_retries = min(max_retries, 2)

        retry_count = 0
        total_delay = 0.0

        while retry_count < max_retries:
            try:
                result = operation()

                if retry_count > 0:
                    logger.info(
                        f"{operation_name} succeeded after {retry_count} retries "
                        f"(total delay: {total_delay:.1f}s)"
                    )

                return result

            except requests.exceptions.HTTPError as e:
                # Special handling for rate limiting
                if e.response and e.response.status_code == 429:
                    delay = base_delay * 10  # Much longer backoff for rate limits
                    retry_count += 1
                    logger.warning(
                        f"{operation_name} rate limited (429). "
                        f"Waiting {delay:.0f}s before retry {retry_count}/{max_retries}..."
                    )
                    total_delay += delay

                    # Progress reporting (if console available)
                    if hasattr(self, "console") and self.console:
                        self.console.print(
                            f"[yellow]⚠ {operation_name} rate limited (attempt {retry_count}/{max_retries})[/yellow]"
                        )
                        self.console.print(
                            f"[yellow]  Retrying in {delay:.1f}s...[/yellow]"
                        )

                    time.sleep(delay)
                    continue
                else:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(
                            f"{operation_name} failed after {max_retries} attempts: {e}"
                        )
                        return None

                    delay = (
                        base_delay * (2 ** (retry_count - 1)) * (0.5 + random.random())
                    )
                    total_delay += delay

                    # Progress reporting (if console available)
                    if hasattr(self, "console") and self.console:
                        self.console.print(
                            f"[yellow]⚠ {operation_name} failed (attempt {retry_count}/{max_retries})[/yellow]"
                        )
                        self.console.print(f"[yellow]  Error: {str(e)[:100]}[/yellow]")
                        self.console.print(
                            f"[yellow]  Retrying in {delay:.1f}s...[/yellow]"
                        )
                    else:
                        logger.warning(
                            f"{operation_name} failed (attempt {retry_count}/{max_retries}). "
                            f"Retrying in {delay:.1f}s... Error: {e}"
                        )

                    time.sleep(delay)

            except OSError as e:
                # GEOparse wraps ftplib.error_perm (550) as OSError with specific message
                error_str = str(e)
                if "Download failed" in error_str and (
                    "No such file" in error_str or "not public yet" in error_str
                ):
                    logger.warning(
                        f"{operation_name} OSError indicates missing file: {error_str[:100]}. "
                        "Skipping retries, triggering fallback mechanism."
                    )
                    return "SOFT_FILE_MISSING"  # Sentinel value to signal fallback
                # Other OSErrors may be transient, fall through to generic handler
                logger.warning(
                    f"{operation_name} OSError (may retry): {error_str[:100]}"
                )
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(
                        f"{operation_name} failed after {max_retries} attempts: {e}"
                    )
                    return None
                # Continue with exponential backoff
                delay = base_delay * (2 ** (retry_count - 1)) * (0.5 + random.random())
                total_delay += delay
                logger.warning(
                    f"{operation_name} retrying after OSError (attempt {retry_count}/{max_retries}) in {delay:.1f}s"
                )
                time.sleep(delay)

            except ftplib.error_perm as e:
                # Permanent FTP errors (550 = File not found) should not be retried
                error_str = str(e)
                if error_str.startswith("550"):
                    logger.warning(
                        f"{operation_name} permanent FTP error: File not found (550). "
                        "Skipping retries, triggering fallback mechanism."
                    )
                    return "SOFT_FILE_MISSING"  # Sentinel value to signal fallback
                # Other FTP error codes may be transient, fall through to generic handler
                logger.warning(f"{operation_name} FTP error: {error_str}")
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(
                        f"{operation_name} failed after {max_retries} attempts: {e}"
                    )
                    return None
                # Continue with exponential backoff
                delay = base_delay * (2 ** (retry_count - 1)) * (0.5 + random.random())
                total_delay += delay
                logger.warning(
                    f"{operation_name} retrying after FTP error (attempt {retry_count}/{max_retries}) in {delay:.1f}s"
                )
                time.sleep(delay)

            except Exception as e:
                retry_count += 1

                if retry_count >= max_retries:
                    logger.error(
                        f"{operation_name} failed after {max_retries} attempts: {e}"
                    )
                    return None

                # Exponential backoff with jitter
                # Jitter range: 0.5-1.5x multiplier (standard practice)
                delay = base_delay * (2 ** (retry_count - 1)) * (0.5 + random.random())
                total_delay += delay

                # Progress reporting (if console available)
                if hasattr(self, "console") and self.console:
                    self.console.print(
                        f"[yellow]⚠ {operation_name} failed (attempt {retry_count}/{max_retries})[/yellow]"
                    )
                    self.console.print(f"[yellow]  Error: {str(e)[:100]}[/yellow]")
                    self.console.print(
                        f"[yellow]  Retrying in {delay:.1f}s...[/yellow]"
                    )
                else:
                    logger.warning(
                        f"{operation_name} failed (attempt {retry_count}/{max_retries}). "
                        f"Retrying in {delay:.1f}s... Error: {e}"
                    )

                time.sleep(delay)

        return None

    # ████████████████████████████████████████████████████████████████████████████████
    # ██                                                                            ██
    # ██                  MAIN ENTRY POINTS (USED BY DATA_EXPERT)                  ██
    # ██                                                                            ██
    # ████████████████████████████████████████████████████████████████████████████████

    def fetch_metadata_only(self, geo_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fetch and validate GEO metadata with fallback mechanisms (Scenario 1).

        This function handles both GSE and GDS identifiers, converting GDS to GSE
        when needed, and stores the metadata in data_manager for user review.

        Args:
            geo_id: GEO accession ID (e.g., GSE194247 or GDS5826)

        Returns:
            Tuple[Dict, Dict]: metadata and validation_result
        """
        try:
            logger.info(f"Fetching metadata for GEO ID: {geo_id}")

            # Clean the GEO ID
            clean_geo_id = geo_id.strip().upper()

            # Check if it's a GDS identifier
            if clean_geo_id.startswith("GDS"):
                logger.info(f"Detected GDS identifier: {clean_geo_id}")
                return self._fetch_gds_metadata_and_convert(clean_geo_id)
            elif not clean_geo_id.startswith("GSE"):
                logger.error(
                    f"Invalid GEO ID format: {geo_id}. Must be a GSE or GDS accession (e.g., GSE194247 or GDS5826)."
                )
                return (None, None)

            # Handle GSE identifiers (existing logic)
            return self._fetch_gse_metadata(clean_geo_id)

        except UnsupportedPlatformError:
            # Re-raise platform errors - they should be handled by caller
            raise
        except FeatureNotImplementedError:
            # Re-raise modality errors - they should be handled by caller
            raise
        except Exception as e:
            logger.exception(f"Error fetching metadata for {geo_id}: {e}")
            return (None, None)

    def _fetch_gse_metadata(self, gse_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fetch GSE metadata using GEOparse with retry logic.

        Args:
            gse_id: GSE accession ID

        Returns:
            Tuple[Dict, Dict]: metadata and validation_result
        """
        try:
            logger.debug(f"Downloading SOFT metadata for {gse_id} using GEOparse...")

            # Wrap GEOparse call with retry logic for transient network failures
            gse = self._retry_with_backoff(
                operation=lambda: GEOparse.get_GEO(
                    geo=gse_id, destdir=str(self.cache_dir)
                ),
                operation_name=f"Fetch metadata for {gse_id}",
                max_retries=5,
                is_ftp=False,
            )

            # Check if SOFT file was missing (sentinel value from retry logic)
            if gse == "SOFT_FILE_MISSING":
                logger.info(
                    f"SOFT file unavailable for {gse_id}, attempting Entrez fallback..."
                )
                try:
                    return self._fetch_gse_metadata_via_entrez(gse_id)
                except Exception as e:
                    logger.error(f"Entrez fallback also failed for {gse_id}: {e}")
                    logger.error(
                        f"Failed to fetch metadata for {gse_id}: "
                        f"SOFT file missing and Entrez fallback failed ({str(e)}). "
                        f"Please check GEO database status or try again later."
                    )
                    return (None, None)

            if gse is None:
                logger.error(
                    f"Failed to fetch metadata for {gse_id} after multiple retry attempts."
                )
                return (None, None)

            metadata = self._extract_metadata(gse)
            logger.debug(f"Successfully extracted metadata using GEOparse for {gse_id}")

            if not metadata:
                logger.error(f"No metadata could be extracted for {gse_id}")
                return (None, None)

            # Validate metadata against transcriptomics schema
            validation_result = self._validate_geo_metadata(metadata)

            # Check platform compatibility BEFORE downloading files (Phase 2: Early Validation)
            try:
                is_compatible, compat_message = self._check_platform_compatibility(
                    gse_id, metadata
                )
                logger.info(f"Platform validation for {gse_id}: {compat_message}")
            except UnsupportedPlatformError as e:
                # Store metadata and error for supervisor access, then re-raise
                self.data_manager.metadata_store[gse_id] = {
                    "metadata": metadata,
                    "validation_result": validation_result,
                    "platform_error": str(e),
                    "platform_details": e.details,
                }
                logger.error(
                    f"Platform validation failed for {gse_id}: {e.details['detected_platforms']}"
                )
                raise

            return metadata, validation_result

        except UnsupportedPlatformError:
            # Re-raise platform errors without catching them
            raise
        except FeatureNotImplementedError:
            # Re-raise modality errors without catching them
            raise
        except Exception as geoparse_error:
            logger.error(f"GEOparse metadata fetch failed: {geoparse_error}")
            logger.error(
                f"Failed to fetch metadata for {gse_id}. GEOparse ({geoparse_error}) failed."
            )
            return (None, None)

    def _fetch_gds_metadata_and_convert(
        self, gds_id: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fetch GDS metadata using NCBI E-utilities and convert to GSE for downstream processing.

        Args:
            gds_id: GDS accession ID (e.g., GDS5826)

        Returns:
            Tuple[Dict, Dict]: Combined metadata and validation_result
        """
        try:
            logger.info(f"Fetching GDS metadata for {gds_id} using NCBI E-utilities...")

            # Extract GDS number from ID
            gds_number = gds_id.replace("GDS", "")

            # Build NCBI E-utilities URL for GDS metadata
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {"db": "gds", "id": gds_number, "retmode": "json"}

            # Construct URL with parameters
            url_params = urllib.parse.urlencode(params)
            url = f"{base_url}?{url_params}"

            logger.debug(f"Fetching GDS metadata from: {url}")

            # Create SSL context for secure connection
            ssl_context = create_ssl_context()

            # Make the request with SSL support
            try:
                response = urllib.request.urlopen(url, context=ssl_context, timeout=30)
                response_data = response.read().decode("utf-8")
            except Exception as e:
                error_str = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
                    handle_ssl_error(e, url, logger)
                    raise Exception(
                        "SSL certificate verification failed when fetching GDS metadata. "
                        "See error message above for solutions."
                    )
                raise

            # Parse JSON response
            gds_data = json.loads(response_data)

            # Extract the GDS record
            if "result" not in gds_data or gds_number not in gds_data["result"]:
                logger.error(f"No GDS record found for {gds_id}")
                return (None, None)

            gds_record = gds_data["result"][gds_number]
            logger.debug(f"Successfully retrieved GDS metadata for {gds_id}")

            # Extract GSE ID from GDS record
            gse_id = gds_record.get("gse", "")
            if not gse_id:
                logger.error(f"No associated GSE found for GDS {gds_id}")
                return (None, None)

            # Ensure GSE has proper format
            if not gse_id.startswith("GSE"):
                gse_id = f"GSE{gse_id}"

            logger.info(f"Found associated GSE: {gse_id} for GDS {gds_id}")

            # Fetch the GSE metadata using existing method
            result = self._fetch_gse_metadata(gse_id)

            # Check if metadata fetch failed (returns (None, None) on error)
            if result[0] is None:
                logger.error(
                    f"Failed to fetch GSE metadata for {gse_id} (from GDS {gds_id})"
                )
                return (None, None)

            gse_metadata, validation_result = result

            # Enhance metadata with GDS information
            enhanced_metadata = self._combine_gds_gse_metadata(
                gds_record, gse_metadata, gds_id, gse_id
            )

            return enhanced_metadata, validation_result

        except UnsupportedPlatformError:
            # Re-raise platform errors - they should be handled by caller
            raise
        except urllib.error.URLError as e:
            logger.error(f"Network error fetching GDS metadata: {e}")
            logger.error(f"Network error fetching GDS metadata for {gds_id}: {str(e)}")
            return (None, None)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing GDS JSON response: {e}")
            logger.error(f"Error parsing GDS metadata response for {gds_id}: {str(e)}")
            return (None, None)
        except Exception as e:
            logger.error(f"Error fetching GDS metadata for {gds_id}: {e}")
            logger.error(f"Error fetching GDS metadata for {gds_id}: {str(e)}")
            return (None, None)

    def _combine_gds_gse_metadata(
        self,
        gds_record: Dict[str, Any],
        gse_metadata: Dict[str, Any],
        gds_id: str,
        gse_id: str,
    ) -> Dict[str, Any]:
        """
        Combine GDS and GSE metadata into a unified metadata structure.

        Args:
            gds_record: GDS record from NCBI E-utilities
            gse_metadata: GSE metadata from GEOparse
            gds_id: Original GDS identifier
            gse_id: Associated GSE identifier

        Returns:
            Dict: Combined metadata with both GDS and GSE information
        """
        try:
            # Start with GSE metadata as base
            combined_metadata = gse_metadata.copy()

            # Add GDS-specific information
            combined_metadata["gds_info"] = {
                "gds_id": gds_id,
                "gds_title": gds_record.get("title", ""),
                "gds_summary": gds_record.get("summary", ""),
                "gds_type": gds_record.get("gdstype", ""),
                "platform_technology": gds_record.get("ptechtype", ""),
                "value_type": gds_record.get("valtype", ""),
                "sample_info": gds_record.get("ssinfo", ""),
                "subset_info": gds_record.get("subsetinfo", ""),
                "n_samples": gds_record.get("n_samples", 0),
                "platform_taxa": gds_record.get("platformtaxa", ""),
                "samples_taxa": gds_record.get("samplestaxa", ""),
                "ftp_link": gds_record.get("ftplink", ""),
                "associated_gse": gse_id,
            }

            # Update title and summary to include GDS information if different
            gds_title = gds_record.get("title", "")
            if gds_title and gds_title != combined_metadata.get("title", ""):
                combined_metadata["title"] = (
                    f"{gds_title} (GDS: {gds_id}, GSE: {gse_id})"
                )
            else:
                combined_metadata["title"] = (
                    f"{combined_metadata.get('title', '')} (GDS: {gds_id}, GSE: {gse_id})"
                )

            # Add cross-reference information
            combined_metadata["cross_references"] = {
                "original_request": gds_id,
                "gds_accession": gds_id,
                "gse_accession": gse_id,
                "data_source": "GDS_to_GSE_conversion",
            }

            logger.debug(
                f"Successfully combined GDS and GSE metadata for {gds_id} -> {gse_id}"
            )
            return combined_metadata

        except Exception as e:
            logger.error(f"Error combining GDS and GSE metadata: {e}")
            # Return GSE metadata if combination fails
            return gse_metadata

    def _fetch_gse_metadata_via_entrez(
        self, gse_id: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Fetch GSE metadata using NCBI Entrez E-utilities (fallback for missing SOFT files).

        This method provides basic metadata (title, summary, organism, platform, sample count)
        when SOFT files are unavailable on the FTP server. Sample-level characteristics
        are NOT available via this fallback method.

        Uses the same Entrez esummary API as GDS fetching, but directly for GSE accessions.

        Args:
            gse_id: GSE accession ID (e.g., GSE233321)

        Returns:
            Tuple[Dict, Dict]: metadata and validation_result

        Raises:
            urllib.error.URLError: Network connection errors
            json.JSONDecodeError: Invalid JSON response
            Exception: Other unexpected errors

        Note:
            Entrez metadata is less complete than SOFT files. Missing:
            - Detailed sample-level characteristics (e.g., treatment groups)
            - Protocol details
            - Some contact information
        """
        try:
            logger.info(
                f"Fetching GSE metadata via Entrez fallback for {gse_id} "
                "(SOFT file unavailable)"
            )

            # Extract GSE number from ID
            gse_number = gse_id.replace("GSE", "")

            # Build NCBI E-utilities URL (same pattern as GDS fetching)
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {"db": "gds", "id": gse_number, "retmode": "json"}

            # Construct URL with parameters
            url_params = urllib.parse.urlencode(params)
            url = f"{base_url}?{url_params}"

            logger.debug(f"Fetching GSE metadata from Entrez: {url}")

            # Create SSL context for secure connection
            ssl_context = create_ssl_context()

            # Make the request with SSL support (timeout: 30s)
            try:
                response = urllib.request.urlopen(url, context=ssl_context, timeout=30)
                response_data = response.read().decode("utf-8")
            except Exception as e:
                error_str = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
                    handle_ssl_error(e, url, logger)
                    raise Exception(
                        "SSL certificate verification failed when fetching GSE metadata via Entrez. "
                        "See error message above for solutions."
                    )
                raise

            # Parse JSON response
            entrez_data = json.loads(response_data)

            # Extract the GSE record
            if "result" not in entrez_data or gse_number not in entrez_data["result"]:
                logger.error(f"No Entrez record found for {gse_id}")
                raise ValueError(f"No Entrez record found for {gse_id}")

            gse_record = entrez_data["result"][gse_number]
            logger.debug(f"Successfully retrieved Entrez metadata for {gse_id}")

            # Convert Entrez format to Lobster metadata format
            metadata = self._convert_entrez_to_lobster_metadata(gse_record, gse_id)

            # Validate metadata against transcriptomics schema
            validation_result = self._validate_geo_metadata(metadata)

            # Add fallback markers and warnings
            metadata["_entrez_fallback"] = True
            metadata["_metadata_source"] = "NCBI Entrez E-utilities (esummary)"
            metadata["_metadata_completeness"] = "partial"
            metadata["_warning"] = (
                "Metadata fetched via Entrez fallback due to missing SOFT file. "
                "Sample-level characteristics and protocol details not available. "
                "Basic information (title, summary, organism, platform, sample count) provided."
            )

            # Check platform compatibility BEFORE downloading files (Phase 2: Early Validation)
            try:
                is_compatible, compat_message = self._check_platform_compatibility(
                    gse_id, metadata
                )
                logger.info(
                    f"Platform validation for {gse_id} (Entrez): {compat_message}"
                )
            except UnsupportedPlatformError as e:
                # Store metadata and error for supervisor access, then re-raise
                self.data_manager.metadata_store[gse_id] = {
                    "metadata": metadata,
                    "validation_result": validation_result,
                    "platform_error": str(e),
                    "platform_details": e.details,
                }
                logger.error(
                    f"Platform validation failed for {gse_id}: {e.details['detected_platforms']}"
                )
                raise

            logger.info(
                f"Successfully fetched {gse_id} metadata via Entrez fallback "
                f"(~70% complete, missing sample characteristics)"
            )

            return metadata, validation_result

        except UnsupportedPlatformError:
            # Re-raise platform errors without catching them
            raise
        except urllib.error.URLError as e:
            logger.error(f"Network error in Entrez fallback for {gse_id}: {e}")
            raise Exception(
                f"Network error fetching Entrez metadata for {gse_id}: {str(e)}"
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Entrez JSON response for {gse_id}: {e}")
            raise Exception(
                f"Error parsing Entrez metadata response for {gse_id}: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error in Entrez fallback for {gse_id}: {e}")
            raise

    def _convert_entrez_to_lobster_metadata(
        self, entrez_record: Dict[str, Any], gse_id: str
    ) -> Dict[str, Any]:
        """
        Convert Entrez esummary record to Lobster metadata format.

        Maps Entrez JSON fields to GEOparse-compatible structure for downstream processing.
        Entrez provides basic dataset information but lacks detailed sample characteristics.

        Args:
            entrez_record: Entrez esummary record (JSON parsed)
            gse_id: GSE accession ID

        Returns:
            Dict: Metadata in Lobster format (compatible with _extract_metadata structure)

        Note:
            Entrez field mapping:
            - title: Dataset title
            - summary: Dataset description
            - taxon: Organism name
            - gpl: Platform ID(s)
            - n_samples: Sample count
            - pubmedids: Associated publications
            - pdat: Publication/submission date
        """
        try:
            # Extract platform IDs (can be list or single value)
            platform_ids = entrez_record.get("gpl", [])
            if isinstance(platform_ids, str):
                platform_ids = [platform_ids]
            elif not isinstance(platform_ids, list):
                platform_ids = []

            # Extract PubMed IDs
            pubmed_ids = entrez_record.get("pubmedids", [])
            if isinstance(pubmed_ids, str):
                pubmed_ids = [pubmed_ids]
            elif not isinstance(pubmed_ids, list):
                pubmed_ids = []

            # Build metadata dict compatible with GEOparse structure
            metadata = {
                # Core identifiers
                "geo_accession": gse_id,
                "accession": gse_id,
                # Basic information
                "title": entrez_record.get("title", ""),
                "summary": entrez_record.get("summary", ""),
                "type": entrez_record.get(
                    "gdstype", "Expression profiling by high throughput sequencing"
                ),
                # Organism information
                "taxon": entrez_record.get("taxon", ""),
                "organism": entrez_record.get("taxon", ""),
                # Platform information (as list for consistency with GEOparse)
                "platform_id": platform_ids,
                # Sample information
                "n_samples": entrez_record.get("n_samples", 0),
                "sample_count": entrez_record.get("n_samples", 0),
                # Publication information
                "pubmed_id": pubmed_ids if pubmed_ids else "",
                # Dates
                "submission_date": entrez_record.get("pdat", ""),
                "last_update_date": entrez_record.get("pdat", ""),
                # Platform details (limited from Entrez)
                "platforms": self._extract_platform_info_from_entrez(
                    entrez_record, platform_ids
                ),
                # Sample metadata (empty - not available via Entrez)
                "samples": {},
                "sample_id": [],
                # Supplementary files (not available via Entrez)
                "supplementary_file": [],
                # Status
                "status": "Public",  # Assume public if in GEO
                # FTP link (if available)
                "ftp_link": entrez_record.get("ftplink", ""),
            }

            logger.debug(f"Converted Entrez record to Lobster metadata for {gse_id}")
            return metadata

        except Exception as e:
            logger.error(f"Error converting Entrez metadata to Lobster format: {e}")
            # Return minimal metadata to avoid complete failure
            return {
                "geo_accession": gse_id,
                "title": entrez_record.get("title", "Unknown"),
                "summary": entrez_record.get("summary", ""),
                "organism": entrez_record.get("taxon", "Unknown"),
                "platform_id": [],
                "n_samples": 0,
                "_conversion_error": str(e),
            }

    def _extract_platform_info_from_entrez(
        self, entrez_record: Dict[str, Any], platform_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract platform information from Entrez record.

        Entrez provides limited platform information compared to SOFT files.
        This method creates a minimal platform dict compatible with downstream processing.

        Args:
            entrez_record: Entrez esummary record
            platform_ids: List of platform GPL IDs

        Returns:
            Dict: Platform information in GEOparse-compatible format
        """
        platforms = {}

        try:
            # Entrez doesn't provide detailed platform metadata in esummary
            # Create minimal platform entries for compatibility
            platform_organism = entrez_record.get("taxon", "")
            platform_tech = entrez_record.get("ptechtype", "")

            for gpl_id in platform_ids:
                platforms[gpl_id] = {
                    "title": f"Platform {gpl_id}",
                    "organism": platform_organism,
                    "technology": (
                        platform_tech if platform_tech else "high throughput sequencing"
                    ),
                    "_note": "Platform details from Entrez are limited. Full details available on GEO website.",
                }

            logger.debug(
                f"Extracted platform info for {len(platform_ids)} platform(s) from Entrez"
            )

        except Exception as e:
            logger.warning(f"Error extracting platform info from Entrez: {e}")

        return platforms

    def download_dataset(self, geo_id: str, adapter: str = None, **kwargs) -> str:
        """
        Download and process a dataset using modular strategy with fallbacks (Scenarios 2 & 3).

        Args:
            geo_id: GEO accession ID

        Returns:
            str: Status message with detailed information
        """
        try:
            logger.info(f"Processing GEO query with modular strategy: {geo_id}")

            # Clean the GEO ID
            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith("GSE"):
                return f"Invalid GEO ID format: {geo_id}. Must be a GSE accession."

            # Check if metadata already exists (should be fetched first)
            if clean_geo_id not in self.data_manager.metadata_store:
                logger.debug(f"Metadata not found, fetching first for {clean_geo_id}")
                metadata, validation_result = self.fetch_metadata_only(clean_geo_id)
                if metadata is None:
                    return f"Failed to fetch metadata for {clean_geo_id}"

            # Safety check: Verify platform compatibility (Phase 2: Early Validation)
            stored_metadata = self.data_manager.metadata_store.get(clean_geo_id)
            if stored_metadata:
                # Check if platform error was previously detected
                if "platform_error" in stored_metadata:
                    logger.error(
                        f"Cannot download {clean_geo_id} - platform validation failed previously"
                    )
                    raise UnsupportedPlatformError(
                        message=stored_metadata["platform_error"],
                        details=stored_metadata["platform_details"],
                    )

                # Validate platform compatibility if not done yet
                metadata_dict = stored_metadata.get("metadata", {})
                if metadata_dict:
                    try:
                        is_compatible, compat_message = (
                            self._check_platform_compatibility(
                                clean_geo_id, metadata_dict
                            )
                        )
                        logger.info(
                            f"Platform re-validation for {clean_geo_id}: {compat_message}"
                        )
                    except UnsupportedPlatformError:
                        # Already logged and stored by _check_platform_compatibility
                        raise

            # Check if modality already exists in DataManagerV2
            modality_name = f"geo_{clean_geo_id.lower()}_{adapter}"
            existing_modalities = self.data_manager.list_modalities()
            if modality_name in existing_modalities:
                return f"Dataset {clean_geo_id} already loaded as modality '{modality_name}'. Use data_manager.get_modality('{modality_name}') to access it."

            # Use the strategic download approach
            geo_result = self.download_with_strategy(geo_id=clean_geo_id, **kwargs)

            if not geo_result.success:
                return f"Failed to download {clean_geo_id} using all available methods. Last error: {geo_result.error_message}"

            # Store as modality in DataManagerV2
            enhanced_metadata = {
                "dataset_id": clean_geo_id,
                "dataset_type": "GEO",
                "source_metadata": geo_result.metadata,
                "processing_date": pd.Timestamp.now().isoformat(),
                "download_source": geo_result.source.value,
                "processing_method": geo_result.processing_info.get(
                    "method", "unknown"
                ),
                "data_type": geo_result.processing_info.get("data_type", "unknown"),
            }

            # Determine appropriate adapter based on data characteristics and metadata
            if not enhanced_metadata.get("data_type", None):
                cached_metadata = self.data_manager.metadata_store[clean_geo_id][
                    "metadata"
                ]
                self._determine_data_type_from_metadata(
                    cached_metadata
                )

            n_obs, n_vars = geo_result.data.shape

            # if no adapter name is given find out from data downloading step
            if not adapter:
                if enhanced_metadata.get("data_type") == "single_cell_rna_seq":
                    adapter_name = "transcriptomics_single_cell"
                elif enhanced_metadata.get("data_type") == "bulk_rna_seq":
                    adapter_name = "transcriptomics_bulk"
                else:
                    # Default to single-cell for GEO datasets (more common)
                    adapter_name = "transcriptomics_single_cell"
            else:
                adapter_name = adapter

            logger.debug(
                f"Using adapter '{adapter_name}' based on predicted type '{enhanced_metadata.get('data_type', None)}' and data shape {geo_result.data.shape}"
            )

            # Load as modality in DataManagerV2
            adata = self.data_manager.load_modality(
                name=modality_name,
                source=geo_result.data,
                adapter=adapter_name,
                validate=True,
                **enhanced_metadata,
            )

            # Save to workspace
            save_path = f"{modality_name}_raw.h5ad"
            saved_file = self.data_manager.save_modality(modality_name, save_path)

            # Check if this was a multi-modal dataset and log exclusions
            multimodal_info = None
            if clean_geo_id in self.data_manager.metadata_store:
                stored_entry = self.data_manager._get_geo_metadata(clean_geo_id)
                if stored_entry:
                    multimodal_info = stored_entry.get("multimodal_info")

            # Log successful download and save (with multi-modal info if applicable)
            log_params = {
                "geo_id": clean_geo_id,
                "download_source": geo_result.source.value,
                "processing_method": geo_result.processing_info.get(
                    "method", "unknown"
                ),
            }

            log_description = f"Downloaded GEO dataset {clean_geo_id} using strategic approach ({geo_result.source.value}), saved to {saved_file}"

            if multimodal_info and multimodal_info.get("is_multimodal"):
                # Add multi-modal info to parameters
                log_params["is_multimodal"] = True
                log_params["loaded_modalities"] = multimodal_info.get(
                    "supported_types", []
                )
                log_params["excluded_modalities"] = multimodal_info.get(
                    "unsupported_types", []
                )

                # Calculate excluded sample count
                sample_types = multimodal_info.get("sample_types", {})
                excluded_count = sum(
                    len(samples)
                    for modality, samples in sample_types.items()
                    if modality != "rna"
                )

                # Enhance description with exclusion info
                log_description += f" | Multi-modal dataset: loaded {len(sample_types.get('rna', []))} RNA samples, excluded {excluded_count} unsupported samples"

            self.data_manager.log_tool_usage(
                tool_name="download_geo_dataset_strategic",
                parameters=log_params,
                description=log_description,
            )

            # Auto-save current state
            self.data_manager.auto_save_state()

            # Generate success message (enhanced for multi-modal)
            success_msg = f"""Successfully downloaded and loaded GEO dataset {clean_geo_id}!

📊 Modality: '{modality_name}' ({adata.n_obs} obs × {adata.n_vars} vars)
🔬 Adapter: {adapter_name} (predicted: {enhanced_metadata.get('data_type', None)})
💾 Saved to: {save_path}
🎯 Source: {geo_result.source.value} ({geo_result.processing_info.get('method', 'unknown')})
⚡ Ready for quality control and downstream analysis!"""

            if multimodal_info and multimodal_info.get("is_multimodal"):
                sample_types = multimodal_info.get("sample_types", {})
                excluded_summary = ", ".join(
                    [
                        f"{modality.upper()}: {len(samples)}"
                        for modality, samples in sample_types.items()
                        if modality != "rna"
                    ]
                )
                success_msg += f"""

🧬 Multi-Modal Dataset Detected:
   ✓ Loaded: RNA ({len(sample_types.get('rna', []))} samples)
   ⏭️  Skipped: {excluded_summary} (support coming in v2.6+)

   Note: Only RNA samples were downloaded. Unsupported modalities were excluded to save bandwidth.
   When protein/VDJ support is added, you can re-download to get all modalities."""

            success_msg += f"\n\nThe dataset is now available as modality '{modality_name}' for other agents to use."

            return success_msg

        except Exception as e:
            logger.exception(f"Error downloading dataset: {e}")
            return f"Error downloading dataset: {str(e)}"

    # ████████████████████████████████████████████████████████████████████████████████
    # ██                                                                            ██
    # ██                   STRATEGIC DOWNLOAD COORDINATION                          ██
    # ██                                                                            ██
    # ████████████████████████████████████████████████████████████████████████████████

    def download_with_strategy(
        self,
        geo_id: str,
        manual_strategy_override: PipelineType = None,
        use_intersecting_genes_only: bool = None,
    ) -> GEOResult:
        """
        Master function implementing layered download approach using dynamic pipeline strategy.

        Args:
            geo_id: GEO accession ID
            manual_strategy_override: Optional manual pipeline override
            use_intersecting_genes_only: Concatenation strategy (None=auto, True=inner, False=outer)

        Returns:
            GEOResult: Comprehensive result with data and metadata
        """
        # Store concatenation strategy for use in pipeline functions
        self._use_intersecting_genes_only = use_intersecting_genes_only
        clean_geo_id = geo_id.strip().upper()

        logger.debug(f"Starting strategic download for {clean_geo_id}")

        try:
            # Step 1: Ensure metadata exists
            if clean_geo_id not in self.data_manager.metadata_store:
                metadata, validation_result = self.fetch_metadata_only(clean_geo_id)
                if metadata is None:
                    return GEOResult(
                        success=False,
                        error_message=f"Failed to fetch metadata for {clean_geo_id}",
                        source=GEODataSource.GEOPARSE,
                    )

            # Step 2: Get metadata and strategy config using validated retrieval
            stored_metadata_info = self.data_manager._get_geo_metadata(clean_geo_id)
            if not stored_metadata_info:
                raise ValueError(
                    f"Metadata for {clean_geo_id} not found or malformed in metadata_store. "
                    f"This indicates a storage/retrieval bug."
                )
            cached_metadata = stored_metadata_info["metadata"]
            strategy_config = stored_metadata_info.get("strategy_config", {})

            if not strategy_config:
                # Extract strategy config if not present (backward compatibility)
                logger.warning(
                    f"No strategy config found for {clean_geo_id}, using defaults"
                )
                strategy_config = {
                    "raw_data_available": True,
                    "summary_file_name": "",
                    "processed_matrix_name": "",
                    "raw_UMI_like_matrix_name": "",
                    "cell_annotation_name": "",
                }

            # Step 3: IF USER DECIDES WHICH APPROACH TO CHOOSE MANUALLY OVERRIDE THE AUTOMATED APPRAOCH
            if manual_strategy_override:
                pipeline = self.pipeline_engine.get_pipeline_functions(
                    manual_strategy_override, self
                )
            else:
                pipeline = self._get_processing_pipeline(
                    clean_geo_id, cached_metadata, strategy_config
                )

            logger.debug(f"Using dynamic pipeline with {len(pipeline)} steps")

            # Step 4: Execute pipeline with retries
            for i, pipeline_func in enumerate(pipeline):
                logger.debug(
                    f"Executing pipeline step {i + 1}: {pipeline_func.__name__}"
                )

                try:
                    # ===============================================================
                    result = pipeline_func(clean_geo_id, cached_metadata)
                    # ===============================================================
                    if result.success:
                        logger.debug(f"Success via {pipeline_func.__name__}")
                        return result
                    else:
                        logger.warning(f"Step failed: {result.error_message}")
                except Exception as e:
                    logger.warning(
                        f"Pipeline step {pipeline_func.__name__} failed: {e}"
                    )
                    continue

            return GEOResult(
                success=False,
                error_message="All pipeline steps failed after enough attempts",
                metadata=cached_metadata,
                source=GEODataSource.GEOPARSE,
            )

        except Exception as e:
            logger.exception(f"Error in strategic download: {e}")
            return GEOResult(
                success=False, error_message=str(e), source=GEODataSource.GEOPARSE
            )

    def _get_processing_pipeline(
        self, geo_id: str, metadata: Dict[str, Any], strategy_config: Dict[str, Any]
    ) -> List[Callable]:
        """
        Get the appropriate processing pipeline using the strategy engine.

        Args:
            geo_id: GEO accession ID
            metadata: GEO metadata
            strategy_config: Extracted strategy configuration

        Returns:
            List[Callable]: Pipeline functions to execute in order
        """
        # Determine data type
        data_type = self._determine_data_type_from_metadata(metadata)

        # Create pipeline context
        context = create_pipeline_context(
            geo_id=geo_id,
            strategy_config=strategy_config,
            metadata=metadata,
            data_type=data_type,
        )

        # Determine best pipeline type
        pipeline_type, description = self.pipeline_engine.determine_pipeline(context)

        logger.info(f"Pipeline selection for {geo_id}: {pipeline_type.name}")
        logger.info(f"Reason: {description}")

        # Get the actual processing functions
        pipeline_functions = self.pipeline_engine.get_pipeline_functions(
            pipeline_type, self
        )

        return pipeline_functions

    # ████████████████████████████████████████████████████████████████████████████████
    # ██                                                                            ██
    # ██                        PIPELINE STEP FUNCTIONS                            ██
    # ██                                                                            ██
    # ████████████████████████████████████████████████████████████████████████████████

    def _try_processed_matrix_first(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Try to directly download and use processed matrix files based on LLM strategy config."""
        try:
            logger.debug(f"Attempting to use processed matrix for {geo_id}")

            # Get strategy config from stored metadata
            stored_metadata = self.data_manager.metadata_store[geo_id]
            strategy_config = stored_metadata.get("strategy_config", {})

            matrix_name = strategy_config.get("processed_matrix_name", "")
            matrix_type = strategy_config.get("processed_matrix_filetype", "")

            if not matrix_name or not matrix_type:
                return GEOResult(
                    success=False,
                    error_message="No processed matrix information available in strategy config",
                )

            # Try to download the specific file from supplementary files
            suppl_files = metadata.get("supplementary_file", [])
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]

            # Find the matching file
            target_file = None
            for file_url in suppl_files:
                if matrix_name in file_url and matrix_type in file_url:
                    target_file = file_url
                    break

            if target_file:
                logger.debug(f"Found processed matrix file: {target_file}")
                matrix = self._download_and_parse_file(target_file, geo_id)

                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.SUPPLEMENTARY,
                        processing_info={
                            "method": "processed_matrix_direct",
                            "file": f"{matrix_name}.{matrix_type}",
                            "data_type": self._determine_data_type_from_metadata(
                                metadata
                            ),
                        },
                        success=True,
                    )

            return GEOResult(
                success=False,
                error_message=f"Could not download processed matrix: {matrix_name}.{matrix_type}",
            )

        except Exception as e:
            logger.error(f"Error in processed matrix pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_raw_matrix_first(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Try to directly download and use raw UMI/count matrix files based on LLM strategy config."""
        try:
            logger.debug(f"Attempting to use raw matrix for {geo_id}")

            # Get strategy config from stored metadata
            stored_metadata = self.data_manager.metadata_store[geo_id]
            strategy_config = stored_metadata.get("strategy_config", {})

            matrix_name = strategy_config.get("raw_UMI_like_matrix_name", "")
            matrix_type = strategy_config.get("raw_UMI_like_matrix_filetype", "")

            if not matrix_name or not matrix_type:
                return GEOResult(
                    success=False,
                    error_message="No raw matrix information available in strategy config",
                )

            # Similar logic to processed matrix
            suppl_files = metadata.get("supplementary_file", [])
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]

            target_file = None
            for file_url in suppl_files:
                if matrix_name in file_url and matrix_type in file_url:
                    target_file = file_url
                    break

            if target_file:
                logger.debug(f"Found raw matrix file: {target_file}")
                matrix = self._download_and_parse_file(target_file, geo_id)

                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.SUPPLEMENTARY,
                        processing_info={
                            "method": "raw_matrix_direct",
                            "file": f"{matrix_name}.{matrix_type}",
                            "data_type": self._determine_data_type_from_metadata(
                                metadata
                            ),
                        },
                        success=True,
                    )

            return GEOResult(
                success=False,
                error_message=f"Could not download raw matrix: {matrix_name}.{matrix_type}",
            )

        except Exception as e:
            logger.error(f"Error in raw matrix pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_h5_format_first(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Try to prioritize H5/H5AD format files for efficient loading."""
        try:
            logger.debug(f"Attempting to use H5 format files for {geo_id}")

            # Check both processed and raw for H5 formats
            stored_metadata = self.data_manager.metadata_store[geo_id]
            strategy_config = stored_metadata.get("strategy_config", {})

            # Check processed matrix first
            processed_name = strategy_config.get("processed_matrix_name", "")
            processed_type = strategy_config.get("processed_matrix_filetype", "")

            # Check raw matrix
            raw_name = strategy_config.get("raw_UMI_like_matrix_name", "")
            raw_type = strategy_config.get("raw_UMI_like_matrix_filetype", "")

            h5_files = []
            if processed_type in ["h5", "h5ad"]:
                h5_files.append((processed_name, processed_type, "processed"))
            if raw_type in ["h5", "h5ad"]:
                h5_files.append((raw_name, raw_type, "raw"))

            if not h5_files:
                return GEOResult(
                    success=False,
                    error_message="No H5 format files found in strategy config",
                )

            # Try to download H5 files
            suppl_files = metadata.get("supplementary_file", [])
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]

            for file_name, file_type, file_category in h5_files:
                target_file = None
                for file_url in suppl_files:
                    if file_name in file_url and file_type in file_url:
                        target_file = file_url
                        break

                if target_file:
                    logger.debug(f"Found H5 {file_category} file: {target_file}")
                    # For H5 files, try parsing with geo_parser directly
                    filename = target_file.split("/")[-1]
                    local_path = self.cache_dir / f"{geo_id}_h5_{filename}"

                    if not local_path.exists():
                        if not self.geo_downloader.download_file(
                            target_file, local_path
                        ):
                            continue

                    matrix = self.geo_parser.parse_supplementary_file(local_path)
                    if matrix is not None and not matrix.empty:
                        return GEOResult(
                            data=matrix,
                            metadata=metadata,
                            source=GEODataSource.SUPPLEMENTARY,
                            processing_info={
                                "method": f"h5_format_{file_category}",
                                "file": f"{file_name}.{file_type}",
                                "data_type": self._determine_data_type_from_metadata(
                                    metadata
                                ),
                            },
                            success=True,
                        )

            return GEOResult(
                success=False,
                error_message="Could not download or parse any H5 format files",
            )

        except Exception as e:
            logger.error(f"Error in H5 format pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_supplementary_first(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Try supplementary files as primary approach when no direct matrices available (with retry)."""
        try:
            logger.debug(f"Attempting supplementary files first for {geo_id}")

            # Get GEO object for supplementary file processing with retry logic
            gse = self._retry_with_backoff(
                operation=lambda: GEOparse.get_GEO(
                    geo=geo_id, destdir=str(self.cache_dir)
                ),
                operation_name=f"Download {geo_id} for supplementary files",
                max_retries=5,
                is_ftp=True,  # Supplementary files often use FTP
            )

            if gse is None:
                return GEOResult(
                    success=False,
                    error_message=f"Failed to download {geo_id} for supplementary files after multiple retry attempts",
                )

            # Use existing supplementary file processing
            data = self._process_supplementary_files(gse, geo_id)
            if data is not None and not data.empty:
                return GEOResult(
                    data=data,
                    metadata=metadata,
                    source=GEODataSource.SUPPLEMENTARY,
                    processing_info={
                        "method": "supplementary_first",
                        "data_type": self._determine_data_type_from_metadata(metadata),
                        "n_samples": len(gse.gsms) if hasattr(gse, "gsms") else 0,
                    },
                    success=True,
                )

            return GEOResult(
                success=False,
                error_message="No usable data found in supplementary files",
            )

        except Exception as e:
            logger.error(f"Error in supplementary first pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_archive_extraction_first(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Try extracting from archive files (TAR, ZIP) as primary approach."""
        try:
            logger.debug(f"Attempting archive extraction first for {geo_id}")

            suppl_files = metadata.get("supplementary_file", [])
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]

            # Look for archive files
            archive_files = [
                f
                for f in suppl_files
                if any(ext in f.lower() for ext in [".tar", ".zip", ".rar"])
            ]

            if not archive_files:
                return GEOResult(success=False, error_message="No archive files found")

            # Try processing archive files
            for archive_url in archive_files:
                if archive_url.lower().endswith(".tar"):
                    matrix = self._process_tar_file(archive_url, geo_id)

                    # Handle both DataFrame and AnnData return types
                    if matrix is not None:
                        is_valid = False
                        if isinstance(matrix, pd.DataFrame):
                            is_valid = not matrix.empty
                        elif isinstance(matrix, anndata.AnnData):
                            is_valid = matrix.n_obs > 0 and matrix.n_vars > 0

                        if is_valid:
                            return GEOResult(
                                data=matrix,
                                metadata=metadata,
                                source=GEODataSource.TAR_ARCHIVE,
                                processing_info={
                                    "method": "archive_extraction_first",
                                    "file": archive_url.split("/")[-1],
                                    "data_type": self._determine_data_type_from_metadata(
                                        metadata
                                    ),
                                },
                                success=True,
                            )

            return GEOResult(
                success=False,
                error_message="Could not extract usable data from archive files",
            )

        except Exception as e:
            logger.error(f"Error in archive extraction pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_supplementary_fallback(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Fallback method using supplementary files when primary approaches fail."""
        try:
            logger.debug(f"Trying supplementary fallback for {geo_id}")

            # This is essentially the same as _try_supplementary_first but with different logging
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(self.cache_dir))

            data = self._process_supplementary_files(gse, geo_id)
            if data is not None and not data.empty:
                return GEOResult(
                    data=data,
                    metadata=metadata,
                    source=GEODataSource.SUPPLEMENTARY,
                    processing_info={
                        "method": "supplementary_fallback",
                        "data_type": self._determine_data_type_from_metadata(metadata),
                        "note": "Used as fallback after primary methods failed",
                    },
                    success=True,
                )

            return GEOResult(
                success=False,
                error_message="Supplementary fallback found no usable data",
            )

        except Exception as e:
            logger.error(f"Error in supplementary fallback pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_emergency_fallback(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Emergency fallback when all other methods fail."""
        try:
            logger.warning(
                f"Using emergency fallback for {geo_id} - all other methods failed"
            )

            # Try to get any available data using basic GEOparse approach
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(self.cache_dir))

            # Try to get expression data from any available sample
            if hasattr(gse, "gsms") and gse.gsms:
                for gsm_id, gsm in list(gse.gsms.items())[
                    :5
                ]:  # Try first 5 samples only
                    try:
                        if hasattr(gsm, "table") and gsm.table is not None:
                            matrix = gsm.table
                            if matrix.shape[0] > 0 and matrix.shape[1] > 0:
                                # Add sample prefix to avoid conflicts
                                matrix.index = [
                                    f"{gsm_id}_{idx}" for idx in matrix.index
                                ]

                                logger.warning(
                                    f"Emergency fallback found data from sample {gsm_id}: {matrix.shape}"
                                )
                                return GEOResult(
                                    data=matrix,
                                    metadata=metadata,
                                    source=GEODataSource.GEOPARSE,
                                    processing_info={
                                        "method": "emergency_fallback_single_sample",
                                        "sample_id": gsm_id,
                                        "data_type": self._determine_data_type_from_metadata(
                                            metadata
                                        ),
                                        "note": "Emergency fallback - only partial data recovered",
                                    },
                                    success=True,
                                )
                    except Exception as e:
                        logger.debug(f"Could not get data from sample {gsm_id}: {e}")
                        continue

            return GEOResult(
                success=False,
                error_message="Emergency fallback could not recover any data",
            )

        except Exception as e:
            logger.error(f"Error in emergency fallback pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_geoparse_download(
        self,
        geo_id: str,
        metadata: Dict[str, Any],
        use_intersecting_genes_only: bool = None,
    ) -> GEOResult:
        """Pipeline step: Try standard GEOparse download with proper single-cell/bulk handling (with retry)."""
        try:
            logger.debug(f"Trying GEOparse download for {geo_id}")

            # Wrap GEOparse download with retry logic
            gse = self._retry_with_backoff(
                operation=lambda: GEOparse.get_GEO(
                    geo=geo_id, destdir=str(self.cache_dir)
                ),
                operation_name=f"Download {geo_id} data",
                max_retries=5,
                is_ftp=False,
            )

            if gse is None:
                return GEOResult(
                    success=False,
                    error_message=f"Failed to download {geo_id} after multiple retry attempts",
                )

            # Determine data type from metadata
            data_type = self._determine_data_type_from_metadata(metadata)

            # Use instance variable if set, otherwise use parameter default
            concat_strategy = getattr(
                self, "_use_intersecting_genes_only", use_intersecting_genes_only
            )

            # Try sample matrices
            sample_info = self._get_sample_info(gse)
            if sample_info:
                sample_matrices = self._download_sample_matrices(sample_info, geo_id)
                validated_matrices = self._validate_matrices(sample_matrices)

                if validated_matrices:
                    # UNIFIED PATH: Use ConcatenationService for both single-cell and bulk
                    # This provides robust duplicate handling via anndata.concat(merge="unique")
                    if len(validated_matrices) > 1:
                        # Multiple samples: store individually first, then concatenate
                        stored_samples = self._store_samples_as_anndata(
                            validated_matrices, geo_id, metadata
                        )

                        if stored_samples:
                            # Concatenate using ConcatenationService (handles duplicates)
                            concatenated_dataset = self._concatenate_stored_samples(
                                geo_id, stored_samples, concat_strategy
                            )

                            if concatenated_dataset is not None:
                                return GEOResult(
                                    data=concatenated_dataset,
                                    metadata=metadata,
                                    source=GEODataSource.GEOPARSE,
                                    processing_info={
                                        "method": "geoparse_samples_concatenated",
                                        "data_type": data_type,
                                        "n_samples": len(validated_matrices),
                                        "stored_sample_ids": stored_samples,
                                        "use_intersecting_genes_only": concat_strategy,
                                        "batch_info": {
                                            gsm_id: gsm_id
                                            for gsm_id in validated_matrices.keys()
                                        },
                                        "note": f"Samples concatenated with unified ConcatenationService ({data_type})",
                                    },
                                    success=True,
                                )
                    else:
                        # Single sample: store and return directly as AnnData
                        stored_samples = self._store_samples_as_anndata(
                            validated_matrices, geo_id, metadata
                        )

                        if stored_samples:
                            # Get the single stored sample
                            modality_name = stored_samples[0]
                            single_sample = self.data_manager.get_modality(
                                modality_name
                            )

                            return GEOResult(
                                data=single_sample,
                                metadata=metadata,
                                source=GEODataSource.SAMPLE_MATRICES,
                                processing_info={
                                    "method": "geoparse_single_sample",
                                    "n_samples": 1,
                                    "data_type": data_type,
                                    "stored_sample_id": modality_name,
                                },
                                success=True,
                            )

            # Try supplementary files as fallback
            data = self._process_supplementary_files(gse, geo_id)
            if data is not None and not data.empty:
                return GEOResult(
                    data=data,
                    metadata=metadata,
                    source=GEODataSource.GEOPARSE,
                    processing_info={
                        "method": "geoparse_supplementary",
                        "n_samples": len(gse.gsms) if hasattr(gse, "gsms") else 0,
                    },
                    success=True,
                )

            return GEOResult(
                success=False, error_message="GEOparse could not find usable data"
            )

        except Exception as e:
            logger.warning(f"GEOparse download failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    # ████████████████████████████████████████████████████████████████████████████████
    # ██                                                                            ██
    # ██                   METADATA AND VALIDATION UTILITIES                       ██
    # ██                                                                            ██
    # ████████████████████████████████████████████████████████████████████████████████

    def _check_platform_compatibility(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Check if dataset platform is supported before downloading files.

        Performs multi-level platform detection:
        1. Series-level platforms (most common)
        2. Sample-level platforms (for mixed-platform datasets)
        3. Keyword matching for unknown platforms

        Args:
            geo_id: GEO series identifier
            metadata: Parsed SOFT file metadata from _extract_metadata()

        Returns:
            Tuple of (is_compatible, message):
                - is_compatible: True if platform is supported or unknown
                - message: Human-readable explanation

        Raises:
            UnsupportedPlatformError: If platform is explicitly unsupported (microarray)
        """
        # Extract platform information at series level
        series_platforms = metadata.get("platforms", {})

        # Extract platform information at sample level
        # metadata["samples"] is a dict: {gsm_id: {metadata...}}
        sample_platforms = {}
        samples_dict = metadata.get("samples", {})
        if isinstance(samples_dict, dict):
            for gsm_id, sample_meta in samples_dict.items():
                platform_id = sample_meta.get("platform_id")
                if platform_id:
                    sample_platforms.setdefault(platform_id, []).append(gsm_id)

        # Combine both levels
        all_platforms = {}
        for platform_id, platform_data in series_platforms.items():
            all_platforms[platform_id] = {
                "title": platform_data.get("title", ""),
                "level": "series",
                "samples": sample_platforms.get(platform_id, []),
            }

        # Add sample-only platforms
        for platform_id, samples in sample_platforms.items():
            if platform_id not in all_platforms:
                all_platforms[platform_id] = {
                    "title": f"Platform {platform_id}",
                    "level": "sample",
                    "samples": samples,
                }

        if not all_platforms:
            logger.warning(f"No platform information found for {geo_id}")
            return True, "No platform information available - proceeding with caution"

        # Classify platforms
        unsupported_platforms = []
        supported_platforms = []
        experimental_platforms = []
        unknown_platforms = []

        for platform_id, platform_info in all_platforms.items():
            platform_title = platform_info.get("title", "").lower()

            # Check registry
            if platform_id in PLATFORM_REGISTRY:
                status = PLATFORM_REGISTRY[platform_id]

                if status == PlatformCompatibility.UNSUPPORTED:
                    unsupported_platforms.append((platform_id, platform_info))
                elif status == PlatformCompatibility.SUPPORTED:
                    supported_platforms.append((platform_id, platform_info))
                elif status == PlatformCompatibility.EXPERIMENTAL:
                    experimental_platforms.append((platform_id, platform_info))
            else:
                # Unknown platform - use keyword matching
                if any(kw in platform_title for kw in UNSUPPORTED_KEYWORDS):
                    unsupported_platforms.append((platform_id, platform_info))
                elif any(kw in platform_title for kw in SUPPORTED_KEYWORDS):
                    supported_platforms.append((platform_id, platform_info))
                else:
                    unknown_platforms.append((platform_id, platform_info))

        # Decision logic: Reject if ANY samples use unsupported platforms
        # (unless they also have supported platform data)
        if unsupported_platforms:
            # Check if this is a mixed dataset
            if supported_platforms:
                logger.warning(
                    f"{geo_id} has BOTH supported and unsupported platforms. "
                    f"Will attempt to load supported samples only."
                )
                return (
                    True,
                    "Mixed platform dataset - will filter to supported samples",
                )

            # Pure unsupported dataset - reject
            platform_list = "\n".join(
                [
                    f"  - {pid}: {info['title']} (level: {info['level']}, samples: {len(info.get('samples', []))})"
                    for pid, info in unsupported_platforms
                ]
            )

            raise UnsupportedPlatformError(
                message=f"Dataset {geo_id} uses unsupported platform(s)",
                details={
                    "geo_id": geo_id,
                    "unsupported_platforms": [
                        (pid, info["title"]) for pid, info in unsupported_platforms
                    ],
                    "platform_type": "microarray",
                    "explanation": (
                        "This dataset appears to use microarray platform(s), which are not "
                        "currently supported by Lobster. Lobster is designed for RNA-seq data "
                        "(bulk or single-cell) and proteomics data."
                    ),
                    "detected_platforms": platform_list,
                    "suggestions": [
                        "Search for RNA-seq version of this experiment",
                        "Use RNA-seq platforms: Illumina HiSeq, NextSeq, NovaSeq",
                        "Use single-cell platforms: 10X Chromium, Smart-seq",
                        f"Check if {geo_id} has supplementary RNA-seq files",
                    ],
                },
            )

        # Handle experimental platforms
        if experimental_platforms:
            platform_list = ", ".join([pid for pid, _ in experimental_platforms])
            logger.warning(
                f"{geo_id} uses experimental platform(s): {platform_list}. "
                f"Analysis may require manual validation."
            )
            return True, "Experimental platform detected - proceed with validation"

        # Handle unknown platforms conservatively
        if unknown_platforms and not supported_platforms:
            platform_list = ", ".join([pid for pid, _ in unknown_platforms])
            logger.warning(
                f"{geo_id} has unknown platform(s): {platform_list}. "
                f"Will attempt loading but recommend validation."
            )
            return True, "Unknown platform - will attempt loading"

        # All platforms supported (GPL registry check passed)
        platform_list = ", ".join([pid for pid, _ in supported_platforms])
        logger.info(f"GPL registry check passed for {geo_id}: {platform_list}")

        # === TIER 2: LLM MODALITY DETECTION (Phase 2.1) ===
        logger.info(f"Running LLM modality detection for {geo_id}...")

        # Initialize DataExpertAssistant (lazy initialization pattern)
        if not hasattr(self, "_data_expert_assistant"):
            self._data_expert_assistant = DataExpertAssistant()

        # Call LLM to detect modality
        modality_result = self._data_expert_assistant.detect_modality(metadata, geo_id)

        if modality_result is None:
            # LLM analysis failed - fall back to permissive mode with warning
            logger.warning(
                f"LLM modality detection failed for {geo_id}. "
                f"Proceeding with permissive mode (may cause issues with multi-omics data)."
            )
            return True, "LLM modality detection unavailable - proceeding with caution"

        # Log detection results
        logger.info(
            f"Modality detected: {modality_result.modality} "
            f"(confidence: {modality_result.confidence:.2f}, "
            f"supported: {modality_result.is_supported})"
        )

        # Store modality in metadata for downstream use
        if hasattr(self, "data_manager") and hasattr(
            self.data_manager, "metadata_store"
        ):
            # Store modality detection results with enforced nested structure
            modality_detection_info = {
                "modality": modality_result.modality,
                "confidence": modality_result.confidence,
                "detected_signals": modality_result.detected_signals,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            # Use helper method to store with consistent structure
            if geo_id not in self.data_manager.metadata_store:
                # Store new entry with metadata and modality detection
                self.data_manager._store_geo_metadata(
                    geo_id=geo_id,
                    metadata=metadata,
                    stored_by="_check_platform_compatibility",
                    modality_detection=modality_detection_info,
                )
            else:
                # Update existing entry's modality detection
                existing_entry = self.data_manager._get_geo_metadata(geo_id)
                if existing_entry:
                    existing_entry["modality_detection"] = modality_detection_info
                    self.data_manager.metadata_store[geo_id] = existing_entry

        # Decision: Handle multi-modal datasets intelligently
        if not modality_result.is_supported:
            # Check if this is a multi-modal dataset by examining sample types
            logger.info(
                f"Detected unsupported modality '{modality_result.modality}', checking for multi-modal composition..."
            )
            sample_types = self._detect_sample_types(metadata)

            # Check if we have any supported modalities
            has_rna = "rna" in sample_types and len(sample_types["rna"]) > 0
            has_unsupported = any(
                modality in sample_types and len(sample_types[modality]) > 0
                for modality in ["protein", "vdj", "atac"]
            )

            if has_rna and has_unsupported:
                # Multi-modal dataset with RNA + unsupported modalities
                logger.info(
                    f"Multi-modal dataset detected: RNA ({len(sample_types.get('rna', []))}) samples + "
                    f"unsupported modalities. Will load RNA samples only."
                )

                # Store multi-modal info in metadata for downstream use
                multimodal_info = {
                    "is_multimodal": True,
                    "sample_types": sample_types,
                    "supported_types": ["rna"],
                    "unsupported_types": [t for t in sample_types.keys() if t != "rna"],
                    "detection_timestamp": pd.Timestamp.now().isoformat(),
                }

                # Update metadata store with multi-modal info
                if geo_id in self.data_manager.metadata_store:
                    existing_entry = self.data_manager._get_geo_metadata(geo_id)
                    if existing_entry:
                        existing_entry["multimodal_info"] = multimodal_info
                        self.data_manager.metadata_store[geo_id] = existing_entry

                # Log what will be skipped
                unsupported_summary = ", ".join(
                    [
                        f"{modality}: {len(samples)} samples"
                        for modality, samples in sample_types.items()
                        if modality != "rna"
                    ]
                )
                logger.warning(
                    f"Skipping unsupported modalities in {geo_id}: {unsupported_summary}. "
                    f"Support planned for future releases (v2.6+)."
                )

                return (
                    True,
                    f"Multi-modal dataset - loading RNA samples only ({len(sample_types['rna'])} samples)",
                )

            elif has_rna and not has_unsupported:
                # RNA-only dataset that was misclassified by pre-filter
                logger.info(
                    "Dataset is RNA-only despite modality detection. Proceeding with load."
                )
                return (True, "RNA-only dataset (pre-filter was overly conservative)")

            else:
                # No RNA samples found - truly unsupported
                signals_display = "\n".join(
                    [f"  - {signal}" for signal in modality_result.detected_signals[:5]]
                )
                if len(modality_result.detected_signals) > 5:
                    signals_display += f"\n  ... and {len(modality_result.detected_signals) - 5} more signals"

                raise FeatureNotImplementedError(
                    message=f"Dataset {geo_id} uses unsupported sequencing modality: {modality_result.modality}",
                    details={
                        "geo_id": geo_id,
                        "modality": modality_result.modality,
                        "confidence": modality_result.confidence,
                        "detected_signals": modality_result.detected_signals,
                        "explanation": modality_result.compatibility_reason,
                        "current_workaround": (
                            f"Lobster v2.3 currently supports bulk RNA-seq, 10X single-cell, and Smart-seq2. "
                            f"Support for {modality_result.modality} is planned for future releases."
                        ),
                        "suggestions": modality_result.suggestions,
                        "estimated_implementation": "Planned for Lobster v2.6-v2.8 depending on modality",
                        "detected_signals_formatted": signals_display,
                        "sample_types_detected": (
                            sample_types if sample_types else "No samples classified"
                        ),
                    },
                )

        # Supported modality - return success with modality info
        return (
            True,
            f"Modality compatible: {modality_result.modality} (confidence: {modality_result.confidence:.0%})",
        )

    def _detect_sample_types(self, metadata: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Detect data types for each sample in a GEO dataset.

        This method examines sample-level metadata to classify samples by modality
        (RNA, protein, VDJ, ATAC, etc.). It uses GEO's standardized fields when
        available (library_strategy) and falls back to characteristics_ch1 patterns.

        Args:
            metadata: GEO metadata dict with 'samples' key containing per-sample metadata

        Returns:
            Dict mapping modality names to lists of sample IDs:
                {"rna": ["GSM1", "GSM2"], "protein": ["GSM3"], "vdj": ["GSM4"]}

        Note:
            This uses simple heuristics on reliable GEO fields, not complex pattern matching.
            The library_strategy field is standardized by NCBI and highly reliable.
        """
        sample_types: Dict[str, List[str]] = {}
        samples_dict = metadata.get("samples", {})

        if not samples_dict:
            logger.warning("No samples found in metadata for sample type detection")
            return sample_types

        logger.info(f"Detecting sample types for {len(samples_dict)} samples...")

        for gsm_id, sample_meta in samples_dict.items():
            detected_type = None

            # Strategy 1: Use library_strategy field (most reliable - NCBI controlled vocabulary)
            lib_strategy = sample_meta.get("library_strategy", "")
            if lib_strategy:
                lib_strategy_lower = lib_strategy.lower()
                if "rna-seq" in lib_strategy_lower or "rna seq" in lib_strategy_lower:
                    detected_type = "rna"
                elif "atac" in lib_strategy_lower:
                    detected_type = "atac"
                # Add other strategies as needed

            # Strategy 2: Check characteristics_ch1 field (flexible but less standardized)
            if not detected_type:
                chars = sample_meta.get("characteristics_ch1", [])
                if isinstance(chars, list):
                    chars_text = " ".join([str(c).lower() for c in chars])
                else:
                    chars_text = str(chars).lower()

                # RNA detection
                if any(
                    pattern in chars_text
                    for pattern in [
                        "assay: rna",
                        "assay:rna",
                        "library type: gene expression",
                        "library_type: gene expression",
                        "data type: rna-seq",
                        "datatype: rna-seq",
                        "library type: gex",
                        "library_type: gex",
                    ]
                ):
                    detected_type = "rna"

                # Protein detection (CITE-seq, antibody capture)
                elif any(
                    pattern in chars_text
                    for pattern in [
                        "assay: protein",
                        "assay:protein",
                        "library type: antibody capture",
                        "library_type: antibody capture",
                        "antibody-derived tag",
                        "adt",
                        "cite-seq protein",
                        "citeseq protein",
                    ]
                ):
                    detected_type = "protein"

                # VDJ detection (TCR/BCR sequencing)
                elif any(
                    pattern in chars_text
                    for pattern in [
                        "assay: vdj",
                        "assay:vdj",
                        "library type: vdj",
                        "library_type: vdj",
                        "tcr-seq",
                        "tcr seq",
                        "bcr-seq",
                        "bcr seq",
                        "immune repertoire",
                    ]
                ):
                    detected_type = "vdj"

                # ATAC detection
                elif any(
                    pattern in chars_text
                    for pattern in [
                        "assay: atac",
                        "assay:atac",
                        "library type: atac",
                        "library_type: atac",
                        "chromatin accessibility",
                    ]
                ):
                    detected_type = "atac"

            # Strategy 3: Check sample title for common patterns
            if not detected_type:
                title = sample_meta.get("title", "").lower()
                if any(
                    pattern in title for pattern in ["_rna", "_gex", "_gene_expression"]
                ):
                    detected_type = "rna"
                elif any(
                    pattern in title for pattern in ["_protein", "_adt", "_antibody"]
                ):
                    detected_type = "protein"
                elif any(pattern in title for pattern in ["_vdj", "_tcr", "_bcr"]):
                    detected_type = "vdj"
                elif "_atac" in title:
                    detected_type = "atac"

            # Store result
            if detected_type:
                sample_types.setdefault(detected_type, []).append(gsm_id)
                logger.debug(f"Sample {gsm_id}: detected as '{detected_type}'")
            else:
                # Unknown - default to RNA for backward compatibility
                sample_types.setdefault("rna", []).append(gsm_id)
                logger.debug(f"Sample {gsm_id}: type unclear, defaulting to 'rna'")

        # Log summary
        summary = ", ".join(
            [
                f"{modality}: {len(samples)}"
                for modality, samples in sample_types.items()
            ]
        )
        logger.info(f"Sample type detection complete: {summary}")

        return sample_types

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

            # Define which fields should remain as lists vs be joined as strings
            # Fields that need to remain as lists for downstream processing
            LIST_FIELDS = {
                "supplementary_file",
                "relation",
                "sample_id",
                "platform_id",
                "platform_taxid",
                "sample_taxid",
            }

            # Fields that should be joined as strings for display/summary
            STRING_FIELDS = {
                "title",
                "summary",
                "overall_design",
                "type",
                "contributor",
                "contact_name",
                "contact_email",
                "contact_phone",
                "contact_department",
                "contact_institute",
                "contact_address",
                "contact_city",
                "contact_zip/postal_code",
                "contact_country",
                "geo_accession",
                "status",
                "submission_date",
                "last_update_date",
                "pubmed_id",
                "web_link",
            }

            # Basic metadata from GEOparse
            if hasattr(gse, "metadata"):
                for key, value in gse.metadata.items():
                    if isinstance(value, list):
                        # Keep file-related and ID fields as lists for downstream processing
                        if key in LIST_FIELDS:
                            metadata[key] = value
                        # Join descriptive/text fields as strings for summary generation
                        elif key in STRING_FIELDS:
                            metadata[key] = ", ".join(value) if value else ""
                        else:
                            # For unknown fields, use a conservative approach:
                            # If it looks like a file/ID field, keep as list; otherwise join
                            if any(
                                keyword in key.lower()
                                for keyword in ["file", "url", "id", "accession"]
                            ):
                                metadata[key] = value
                            else:
                                metadata[key] = ", ".join(value) if value else ""
                    else:
                        metadata[key] = value

            # Platform information - keep as structured dict
            if hasattr(gse, "gpls"):
                platforms = {}
                for gpl_id, gpl in gse.gpls.items():
                    platforms[gpl_id] = {
                        "title": self._safely_extract_metadata_field(gpl, "title"),
                        "organism": self._safely_extract_metadata_field(
                            gpl, "organism"
                        ),
                        "technology": self._safely_extract_metadata_field(
                            gpl, "technology"
                        ),
                    }
                metadata["platforms"] = platforms

            # Sample metadata - keep as structured dict
            if hasattr(gse, "gsms"):
                sample_metadata = {}
                for gsm_id, gsm in gse.gsms.items():
                    sample_meta = {}
                    if hasattr(gsm, "metadata"):
                        for key, value in gsm.metadata.items():
                            if isinstance(value, list):
                                # For sample-level metadata, preserve lists for characteristics
                                # but join others for display
                                if (
                                    key in ["characteristics_ch1", "supplementary_file"]
                                    or "file" in key.lower()
                                ):
                                    sample_meta[key] = value
                                else:
                                    sample_meta[key] = ", ".join(value) if value else ""
                            else:
                                sample_meta[key] = value
                    sample_metadata[gsm_id] = sample_meta
                metadata["samples"] = sample_metadata

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    def _safely_extract_metadata_field(
        self, obj, field_name: str, default: str = ""
    ) -> str:
        """
        Safely extract a metadata field from a GEOparse object.

        Handles the common pattern of checking for metadata attribute,
        extracting the field, and joining all elements if it's a list.

        Args:
            obj: GEOparse object (GSE, GPL, or GSM)
            field_name: Name of the metadata field to extract
            default: Default value if field is not found

        Returns:
            str: Extracted metadata value or default
        """
        try:
            if not hasattr(obj, "metadata"):
                return default

            field_value = obj.metadata.get(field_name, [default])

            # Handle list values by joining all elements
            if isinstance(field_value, list):
                return ", ".join(field_value) if field_value else default

            return str(field_value) if field_value else default

        except (AttributeError, IndexError, KeyError):
            return default

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
            uns_schema = schema.get("uns", {}).get("optional", [])

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
                "schema_aligned_fields": len(schema_aligned),
                "schema_missing_fields": len(schema_missing),
                "extra_fields_count": len(extra_fields),
                "alignment_percentage": (
                    (len(schema_aligned) / len(uns_schema) * 100) if uns_schema else 0.0
                ),
                "aligned_metadata": schema_aligned,
                "missing_fields": schema_missing[:10],  # Limit for display
                "extra_fields": extra_fields[:10],  # Limit for display
                "predicted_data_type": data_type,
                "validation_status": (
                    "PASS" if len(schema_aligned) > len(schema_missing) else "WARNING"
                ),
            }

            logger.debug(
                f"Metadata validation: {validation_result['alignment_percentage']:.1f}% schema alignment"
            )
            return validation_result

        except Exception as e:
            logger.error(f"Error validating metadata: {e}")
            return {
                "validation_status": "ERROR",
                "error_message": str(e),
                "schema_aligned_fields": 0,
                "alignment_percentage": 0.0,
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
            platforms = metadata.get("platforms", {})
            platform_info = str(platforms).lower()

            # Check overall design
            overall_design = str(metadata.get("overall_design", "")).lower()

            # Check sample characteristics
            samples = metadata.get("samples", {})
            sample_chars = []
            for sample in samples.values():
                chars = sample.get("characteristics_ch1", [])
                if isinstance(chars, list):
                    sample_chars.extend([str(c).lower() for c in chars])

            sample_text = " ".join(sample_chars)

            # Keywords that suggest single-cell
            single_cell_keywords = [
                "single cell",
                "single-cell",
                "scrnaseq",
                "scrna-seq",
                "10x",
                "chromium",
                "droplet",
                "microwell",
                "smart-seq",
                "cell sorting",
                "sorted cells",
                "individual cells",
            ]

            # Keywords that suggest bulk
            bulk_keywords = ["bulk", "tissue", "whole", "total rna", "population"]

            combined_text = f"{platform_info} {overall_design} {sample_text}"

            # Count keyword matches
            single_cell_score = sum(
                1 for keyword in single_cell_keywords if keyword in combined_text
            )
            bulk_score = sum(1 for keyword in bulk_keywords if keyword in combined_text)

            # Make prediction
            if single_cell_score > bulk_score:
                return "single_cell_rna_seq"
            elif bulk_score > single_cell_score:
                return "bulk_rna_seq"
            else:
                # Default to single-cell for GEO datasets (more common)
                return "single_cell_rna_seq"

        except Exception as e:
            logger.warning(f"Error determining data type: {e}")
            return "single_cell_rna_seq"  # Default

    def _format_metadata_summary(
        self,
        geo_id: str,
        metadata: Dict[str, Any],
        validation_result: Dict[str, Any] = None,
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
            # Extract key information with safe string conversion
            title = str(metadata.get("title", "N/A")).strip()
            summary = str(metadata.get("summary", "N/A")).strip()
            overall_design = str(metadata.get("overall_design", "N/A")).strip()

            # Sample information
            samples = metadata.get("samples", {})
            if not isinstance(samples, dict):
                samples = {}
            sample_count = len(samples)

            # Platform information with safe handling
            platforms = metadata.get("platforms", {})
            if not isinstance(platforms, dict):
                platforms = {}
            platform_info = []
            for platform_id, platform_data in platforms.items():
                if isinstance(platform_data, dict):
                    title_info = platform_data.get("title", "N/A")
                    platform_info.append(f"{platform_id}: {title_info}")
                else:
                    platform_info.append(f"{platform_id}: N/A")

            # Contact information with safe string conversion
            contact_name = str(metadata.get("contact_name", "N/A")).strip()
            contact_institute = str(metadata.get("contact_institute", "N/A")).strip()

            # Publication info
            pubmed_id = str(metadata.get("pubmed_id", "Not available")).strip()

            # Dates with safe string conversion
            submission_date = str(metadata.get("submission_date", "N/A")).strip()
            last_update = str(metadata.get("last_update_date", "N/A")).strip()

            # Sample characteristics preview with safe handling
            sample_preview = []
            for i, (sample_id, sample_data) in enumerate(samples.items()):
                if i < 3:  # Show first 3 samples
                    if isinstance(sample_data, dict):
                        chars = sample_data.get("characteristics_ch1", [])
                        if isinstance(chars, list) and chars:
                            sample_preview.append(
                                f"  - {sample_id}: {str(chars[0]).strip()}"
                            )
                        else:
                            title_info = sample_data.get("title", "No title")
                            sample_preview.append(
                                f"  - {sample_id}: {str(title_info).strip()}"
                            )
                    else:
                        sample_preview.append(f"  - {sample_id}: No title")

            if sample_count > 3:
                sample_preview.append(f"  ... and {sample_count - 3} more samples")

            # Robust validation status handling with type checking
            validation_status = "UNKNOWN"
            alignment_pct_formatted = "UNKNOWN"
            predicted_type = "UNKNOWN"
            aligned_fields = "UNKNOWN"
            missing_fields = "UNKNOWN"

            if validation_result and isinstance(validation_result, dict):
                # Validation status
                validation_status = str(
                    validation_result.get("validation_status", "UNKNOWN")
                ).strip()

                # Alignment percentage with robust type handling
                alignment_raw = validation_result.get("alignment_percentage", None)
                if alignment_raw is not None:
                    try:
                        # Try to convert to float
                        alignment_float = float(alignment_raw)
                        alignment_pct_formatted = f"{alignment_float:.1f}"
                    except (ValueError, TypeError):
                        # If conversion fails, use string representation
                        str(alignment_raw)
                        alignment_pct_formatted = str(alignment_raw)

                # Predicted type
                predicted_type_raw = validation_result.get(
                    "predicted_data_type", "unknown"
                )
                if predicted_type_raw:
                    predicted_type = str(predicted_type_raw).replace("_", " ").title()

                # Schema field counts with robust type handling
                aligned_raw = validation_result.get("schema_aligned_fields", None)
                if aligned_raw is not None:
                    try:
                        aligned_fields = int(aligned_raw)
                    except (ValueError, TypeError):
                        aligned_fields = str(aligned_raw)

                missing_raw = validation_result.get("schema_missing_fields", None)
                if missing_raw is not None:
                    try:
                        missing_fields = int(missing_raw)
                    except (ValueError, TypeError):
                        missing_fields = str(missing_raw)

            # Format the summary with safe string formatting
            summary_text = f"""📊 **GEO Dataset Metadata Summary: {geo_id}**

🔬 **Study Information:**
- **Title:** {title}
- **Summary:** {summary}
- **Design:** {overall_design}
- **Predicted Type:** {predicted_type}

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
- **Schema Alignment:** {alignment_pct_formatted}% of expected fields present
- **Aligned Fields:** {aligned_fields}
- **Missing Fields:** {missing_fields}

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
            logger.exception("Full traceback for metadata formatting error:")
            return f"Error formatting metadata summary for {geo_id}: {str(e)}"

    # ████████████████████████████████████████████████████████████████████████████████
    # ██                                                                            ██
    # ██                     CORE PROCESSING UTILITIES                             ██
    # ██                                                                            ██
    # ████████████████████████████████████████████████████████████████████████████████

    def _detect_kallisto_salmon_files(
        self,
        supplementary_files: List[str],
    ) -> Tuple[bool, str, List[str], int]:
        """
        Detect if dataset contains Kallisto/Salmon per-sample quantification files.

        Args:
            supplementary_files: List of supplementary file URLs/paths

        Returns:
            Tuple of (has_quant_files, tool_type, matched_filenames, estimated_samples):
            - has_quant_files: True if quantification files detected
            - tool_type: "kallisto", "salmon", or "mixed"
            - matched_filenames: List of matching file names
            - estimated_samples: Estimated number of samples
        """
        kallisto_patterns = [
            "abundance.tsv",
            "abundance.h5",
            "abundance.txt",
        ]

        salmon_patterns = [
            "quant.sf",
            "quant.genes.sf",
        ]

        kallisto_files = []
        salmon_files = []
        abundance_files = []  # Track unique files for sample count

        for file_path in supplementary_files:
            # Extract just the filename from URL
            filename = os.path.basename(file_path).lower()

            # Check Kallisto patterns
            if any(pattern in filename for pattern in kallisto_patterns):
                kallisto_files.append(file_path)
                if "abundance.tsv" in filename or "abundance.h5" in filename:
                    abundance_files.append(filename)

            # Check Salmon patterns
            if any(pattern in filename for pattern in salmon_patterns):
                salmon_files.append(file_path)
                if "quant.sf" in filename:
                    abundance_files.append(filename)

        # Determine tool type
        has_quant = len(kallisto_files) > 0 or len(salmon_files) > 0

        if not has_quant:
            return False, "", [], 0

        if len(kallisto_files) > 0 and len(salmon_files) > 0:
            tool_type = "mixed"
            matched = kallisto_files + salmon_files
        elif len(kallisto_files) > 0:
            tool_type = "kallisto"
            matched = kallisto_files
        else:
            tool_type = "salmon"
            matched = salmon_files

        # Estimate sample count (one abundance file per sample)
        estimated_samples = len(abundance_files)

        return has_quant, tool_type, matched, estimated_samples

    def _load_quantification_files(
        self,
        quantification_dir: Path,
        tool_type: str,
        gse_id: str,
        data_type: str = "bulk",
    ) -> Optional[anndata.AnnData]:
        """
        Load Kallisto/Salmon quantification files into AnnData.

        This method follows the same pattern as other GEO loading methods:
        1. Merges per-sample quantification files using BulkRNASeqService
        2. Creates AnnData using TranscriptomicsAdapter
        3. Returns AnnData (NOT DataFrame) so download_dataset() can handle storage with correct naming

        Args:
            quantification_dir: Directory containing per-sample subdirectories
            tool_type: "kallisto" or "salmon"
            gse_id: GEO series ID
            data_type: Data type for TranscriptomicsAdapter ("bulk" or "single_cell", default: "bulk")

        Returns:
            AnnData: Processed AnnData object or None if loading fails
        """
        try:
            logger.info(
                f"Loading {tool_type} quantification files from {quantification_dir}"
            )

            # Step 1: Use bulk_rnaseq_service to merge quantification files
            bulk_service = BulkRNASeqService()

            try:
                df, metadata = bulk_service.load_from_quantification_files(
                    quantification_dir=quantification_dir,
                    tool=tool_type,
                )
                logger.info(
                    f"Successfully merged {metadata['n_samples']} {tool_type} samples "
                    f"× {metadata['n_genes']} genes"
                )
            except Exception as e:
                logger.error(f"Failed to merge {tool_type} files: {e}")
                raise

            # Step 2: Use TranscriptomicsAdapter to create AnnData
            # Note: Constructor needs "bulk" but from_quantification_dataframe() needs "bulk_rnaseq"
            adapter = TranscriptomicsAdapter(data_type=data_type)

            try:
                adata = adapter.from_quantification_dataframe(
                    df=df,
                    data_type="bulk_rnaseq",  # Quantification-specific data_type
                    metadata=metadata,
                )
                logger.info(
                    f"Created AnnData from quantification: "
                    f"{adata.n_obs} samples × {adata.n_vars} genes"
                )
            except Exception as e:
                logger.error(f"Failed to create AnnData from quantification: {e}")
                raise

            # Step 3: Add GEO metadata to AnnData.uns for provenance
            adata.uns["geo_metadata"] = {
                "geo_id": gse_id,
                "data_source": "quantification_files",
                "quantification_tool": tool_type,
                "n_files_merged": metadata["n_samples"],
            }

            logger.info(
                f"Successfully loaded {gse_id} from {tool_type} files: "
                f"{adata.n_obs} samples × {adata.n_vars} genes"
            )

            # Step 4: Return AnnData directly (not DataFrame)
            # Let download_dataset() handle storage with correct naming convention
            return adata

        except Exception as e:
            logger.error(f"Error loading quantification files: {e}")
            logger.exception("Full traceback for quantification loading error:")
            return None

    def _process_supplementary_files(self, gse, gse_id: str) -> Optional[pd.DataFrame]:
        """
        Process supplementary files (TAR archives, etc.) to extract expression data.

        This method now supports:
        - Kallisto/Salmon quantification files (new in Phase 3)
        - TAR archives
        - Direct expression files

        Args:
            gse: GEOparse GSE object
            gse_id: GEO series ID

        Returns:
            DataFrame: Combined expression matrix from supplementary files or None
        """
        try:
            if not hasattr(gse, "metadata") or "supplementary_file" not in gse.metadata:
                logger.debug(f"No supplementary files found for {gse_id}")
                return None

            suppl_files = gse.metadata["supplementary_file"]
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]

            logger.debug(f"Found {len(suppl_files)} supplementary files for {gse_id}")

            # STEP 1: Check for Kallisto/Salmon quantification files FIRST
            has_quant, tool_type, quant_filenames, estimated_samples = (
                self._detect_kallisto_salmon_files(suppl_files)
            )

            if has_quant:
                logger.info(
                    f"{gse_id}: Detected {tool_type} quantification files "
                    f"({estimated_samples} estimated samples)"
                )

                # STEP 2: Check for pre-merged matrix files as alternative
                matrix_files = [
                    f
                    for f in suppl_files
                    if any(
                        ext in f.lower()
                        for ext in [
                            "_matrix.txt",
                            "_counts.txt",
                            "_expression.txt",
                            ".h5ad",
                            "_tpm.txt",
                            "_fpkm.txt",
                        ]
                    )
                ]

                if matrix_files:
                    logger.info(
                        f"{gse_id}: Found {len(matrix_files)} pre-merged matrix files "
                        f"alongside quantification. Using matrix files (faster loading)."
                    )
                    # Process matrix files normally - fall through to existing logic below
                else:
                    # STEP 3: Route to quantification file handler
                    logger.info(
                        f"{gse_id}: No pre-merged matrix found. "
                        f"Loading {tool_type} quantification files..."
                    )

                    # This will be handled by the new _load_quantification_files method
                    # For now, we need to return a signal that quantification files were found
                    # The actual loading will happen in the download_dataset method
                    logger.info(
                        f"{gse_id}: Quantification file loading requires TAR extraction. "
                        f"Proceeding with TAR file processing."
                    )
                    # Fall through to TAR processing

            # Look for TAR files first (most common for expression data)
            tar_files = [f for f in suppl_files if f.lower().endswith(".tar")]

            if tar_files:
                logger.debug(f"Processing TAR file: {tar_files[0]}")
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
                logger.debug(f"Processing expression file: {expression_files[0]}")
                return self._download_and_parse_file(expression_files[0], gse_id)

            logger.warning(
                f"No suitable expression files found in supplementary files for {gse_id}"
            )
            return None

        except Exception as e:
            logger.error(f"Error processing supplementary files: {e}")
            return None

    def _process_tar_file(
        self, tar_url: str, gse_id: str
    ) -> Optional[Union[pd.DataFrame, anndata.AnnData]]:
        """
        Download and process a TAR file containing expression data.

        This method can return either DataFrame or AnnData depending on the data source:
        - Quantification files (Kallisto/Salmon): Returns AnnData directly
        - Other expression files: Returns DataFrame for adapter processing

        Args:
            tar_url: URL to TAR file
            gse_id: GEO series ID

        Returns:
            Union[DataFrame, AnnData]: Expression data or None if processing fails
        """
        try:
            # Download TAR file
            tar_file_path = self.cache_dir / f"{gse_id}_RAW.tar"

            if not tar_file_path.exists():
                logger.debug(f"Downloading TAR file from: {tar_url}")
                urllib.request.urlretrieve(tar_url, tar_file_path)
                logger.debug(f"Downloaded TAR file: {tar_file_path}")
            else:
                logger.debug(f"Using cached TAR file: {tar_file_path}")

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

                logger.debug(f"Extracted {len(safe_members)} files from TAR")

            # STEP 1: Check for Kallisto/Salmon quantification files in extracted directory
            try:
                logger.debug(f"Checking for quantification files in {extract_dir}")

                # Use BulkRNASeqService to detect quantification tool
                bulk_service = BulkRNASeqService()
                tool_type = bulk_service._detect_quantification_tool(extract_dir)

                logger.info(
                    f"{gse_id}: Detected {tool_type} quantification files in TAR archive"
                )

                # Load quantification files using the new loader
                # Returns AnnData (not DataFrame) for consistent handling
                adata_result = self._load_quantification_files(
                    quantification_dir=extract_dir,
                    tool_type=tool_type,
                    gse_id=gse_id,
                    data_type="bulk",
                )

                if adata_result is not None:
                    logger.info(
                        f"{gse_id}: Successfully loaded quantification files: "
                        f"{adata_result.n_obs} samples × {adata_result.n_vars} genes"
                    )
                    # Return the AnnData directly (GEOResult will handle it)
                    return adata_result
                else:
                    logger.warning(
                        f"{gse_id}: Quantification file loading returned None, "
                        f"falling back to standard processing"
                    )

            except ValueError as e:
                # No quantification files detected - this is expected for most datasets
                logger.debug(
                    f"{gse_id}: No quantification files detected in TAR: {e}. "
                    f"Continuing with standard TAR processing."
                )
            except Exception as e:
                # Unexpected error during quantification loading
                logger.warning(
                    f"{gse_id}: Error during quantification file detection/loading: {e}. "
                    f"Falling back to standard TAR processing."
                )

            # STEP 2: Process nested archives and find expression data (existing logic)
            nested_extract_dir = self.cache_dir / f"{gse_id}_nested_extracted"
            nested_extract_dir.mkdir(exist_ok=True)

            # Extract any nested TAR.GZ files
            nested_archives = list(extract_dir.glob("*.tar.gz"))
            if nested_archives:
                logger.debug(f"Found {len(nested_archives)} nested TAR.GZ files")

                all_matrices = []
                for archive_path in nested_archives:
                    try:
                        sample_id = archive_path.stem.split(".")[
                            0
                        ]  # Extract sample ID from filename
                        sample_extract_dir = nested_extract_dir / sample_id
                        sample_extract_dir.mkdir(exist_ok=True)

                        logger.debug(f"Extracting nested archive: {archive_path.name}")
                        with tarfile.open(archive_path, "r:gz") as nested_tar:
                            nested_tar.extractall(path=sample_extract_dir)

                        # Try to parse 10X Genomics format data
                        matrix = self.geo_parser.parse_10x_data(
                            sample_extract_dir, sample_id
                        )
                        if matrix is not None:
                            all_matrices.append(matrix)
                            logger.debug(
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
                logger.debug(
                    f"Found {len(expression_files)} potential expression files"
                )

                # Try to parse the largest file first (likely the main expression matrix)
                expression_files.sort(key=lambda x: x.stat().st_size, reverse=True)

                for file_path in expression_files[:3]:  # Try top 3 largest files
                    try:
                        logger.debug(f"Attempting to parse: {file_path.name}")
                        matrix = self.geo_parser.parse_expression_file(file_path)
                        if (
                            matrix is not None
                            and matrix.shape[0] > 0
                            and matrix.shape[1] > 0
                        ):
                            logger.debug(
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
        Download and parse a single expression file with progress tracking.

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
                # Use geo_downloader for progress tracking and better protocol support
                if not self.geo_downloader.download_file(file_url, local_file):
                    logger.error(f"Failed to download file: {file_url}")
                    return None
                logger.debug(f"Successfully downloaded: {local_file}")
            else:
                logger.debug(f"Using cached file: {local_file}")

            return self.geo_parser.parse_expression_file(local_file)

        except Exception as e:
            logger.error(f"Error downloading and parsing file: {e}")
            return None

    # Parsing functions have been moved to geo_parser.py for better separation of concerns
    # and reusability across different modalities

    # ████████████████████████████████████████████████████████████████████████████████
    # ██                                                                            ██
    # ██                   SAMPLE PROCESSING AND VALIDATION                        ██
    # ██                                                                            ██
    # ████████████████████████████████████████████████████████████████████████████████

    def _get_sample_info(self, gse) -> Dict[str, Dict[str, Any]]:
        """
        Get sample information for downloading individual matrices.

        For multi-modal datasets, filters samples to only include supported modalities (RNA).

        Args:
            gse: GEOparse GSE object

        Returns:
            dict: Sample information dictionary (filtered for multi-modal datasets)
        """
        sample_info = {}

        try:
            # Check if this is a multi-modal dataset
            geo_id = (
                gse.metadata.get("geo_accession", [""])[0]
                if hasattr(gse, "metadata")
                else ""
            )
            multimodal_info = None
            if geo_id and geo_id in self.data_manager.metadata_store:
                stored_entry = self.data_manager._get_geo_metadata(geo_id)
                if stored_entry:
                    multimodal_info = stored_entry.get("multimodal_info")

            # Collect sample info
            if hasattr(gse, "gsms"):
                for gsm_id, gsm in gse.gsms.items():
                    # Check if we should include this sample (multi-modal filtering)
                    if multimodal_info and multimodal_info.get("is_multimodal"):
                        # Get RNA sample IDs from multi-modal info
                        rna_sample_ids = multimodal_info.get("sample_types", {}).get(
                            "rna", []
                        )
                        if gsm_id not in rna_sample_ids:
                            logger.debug(
                                f"Skipping non-RNA sample {gsm_id} (multi-modal dataset)"
                            )
                            continue

                    sample_info[gsm_id] = {
                        "title": (
                            getattr(gsm, "metadata", {}).get("title", [""])[0]
                            if hasattr(gsm, "metadata")
                            else ""
                        ),
                        "platform": (
                            getattr(gsm, "metadata", {}).get("platform_id", [""])[0]
                            if hasattr(gsm, "metadata")
                            else ""
                        ),
                        "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm_id}",
                        "download_url": f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm_id[:6]}nnn/{gsm_id}/suppl/",
                    }

            if multimodal_info and multimodal_info.get("is_multimodal"):
                logger.info(
                    f"Multi-modal filtering: collected {len(sample_info)} RNA samples "
                    f"(excluded {len(gse.gsms) - len(sample_info)} unsupported samples)"
                )
            else:
                logger.debug(f"Collected information for {len(sample_info)} samples")

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
                        logger.debug(
                            f"Successfully downloaded matrix for {gsm_id}: {matrix.shape}"
                        )
                    else:
                        logger.warning(f"No matrix data found for {gsm_id}")
                except Exception as e:
                    logger.error(f"Error downloading {gsm_id}: {e}")
                    sample_matrices[gsm_id] = None

        return sample_matrices

    def _download_single_sample(
        self, gsm_id: str, info: Dict[str, Any], gse_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Download a single sample matrix with enhanced single-cell support.

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

            # Check for 10X format supplementary files
            if hasattr(gsm, "metadata"):
                suppl_files_mapped = self._extract_supplementary_files_from_metadata(
                    gsm.metadata, gsm_id
                )
                df = self._download_and_combine_single_cell_files(
                    suppl_files_mapped, gsm_id
                )
                return df

        except Exception:
            # Fallback to expression table
            if hasattr(gsm, "table") and gsm.table is not None:
                matrix = gsm.table
                return self._store_single_sample_as_modality(gsm_id, matrix, gsm)

            logger.warning(f"No expression data found for {gsm_id}")
            return None

        except Exception as e:
            logger.error(f"Error downloading sample {gsm_id}: {e}")
            return None

    def _extract_supplementary_files_from_metadata(
        self, metadata: Dict[str, Any], gsm_id: str
    ) -> Dict[str, str]:
        """
        Extract and classify supplementary files using robust pattern matching and scoring.

        This function uses a professional scoring system with regex patterns to classify
        supplementary files more accurately, handling edge cases and variations in naming
        conventions commonly found in GEO datasets.

        Args:
            metadata: Sample metadata dictionary
            gsm_id: GEO sample ID for logging

        Returns:
            Dict[str, str]: Dictionary mapping file types to URLs with highest confidence scores
        """
        try:
            # Initialize file type definitions with regex patterns and confidence scores
            file_type_patterns = self._initialize_file_type_patterns()

            # Find all supplementary file URLs
            file_urls = self._extract_all_supplementary_urls(metadata, gsm_id)

            if not file_urls:
                logger.warning(f"No supplementary files found for {gsm_id}")
                return {}

            # Score and classify each file
            classified_files = {}
            file_scores = {}  # Track scores for debugging

            for url in file_urls:
                filename = url.split("/")[-1]
                file_classification = self._classify_single_file(
                    filename, url, file_type_patterns
                )

                for file_type, score in file_classification.items():
                    if score > 0:  # Only consider positive scores
                        # Keep track of best score for each file type
                        if (
                            file_type not in classified_files
                            or score > file_scores.get(file_type, 0)
                        ):
                            classified_files[file_type] = url
                            file_scores[file_type] = score
                            logger.debug(
                                f"Updated {file_type}: {filename} (score: {score:.2f})"
                            )

            # Report final classifications
            logger.debug(f"Final classification for {gsm_id}:")
            for file_type, url in classified_files.items():
                filename = url.split("/")[-1]
                score = file_scores[file_type]
                logger.debug(f"  {file_type}: {filename} (confidence: {score:.2f})")

            # Validate 10X trio completeness if matrix found
            if "matrix" in classified_files:
                self._validate_10x_trio_completeness(classified_files, gsm_id)

            return classified_files

        except Exception as e:
            logger.error(
                f"Error extracting supplementary files from metadata for {gsm_id}: {e}"
            )
            return {}

    def _initialize_file_type_patterns(
        self,
    ) -> Dict[str, Dict[str, Union[List[re.Pattern], float]]]:
        """
        Initialize comprehensive file type classification patterns with confidence scoring.

        Returns:
            Dict containing regex patterns and base scores for each file type
        """
        patterns = {
            "matrix": {
                "patterns": [
                    # High confidence patterns (exact matches)
                    re.compile(r"matrix\.mtx(\.gz)?$", re.IGNORECASE),
                    re.compile(r"matrix\.txt(\.gz)?$", re.IGNORECASE),
                    re.compile(r"matrix\.csv(\.gz)?$", re.IGNORECASE),
                    re.compile(r"matrix\.tsv(\.gz)?$", re.IGNORECASE),
                    # Medium confidence patterns (with context)
                    re.compile(r".*_matrix\.(mtx|txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*-matrix\.(mtx|txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*\.matrix\.(mtx|txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    # Lower confidence patterns (broader matches)
                    re.compile(r".*matrix.*\.(mtx|txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*(count|expr|expression).*matrix.*", re.IGNORECASE),
                ],
                "base_score": 1.0,
                "boost_keywords": ["count", "expression", "sparse", "10x", "chromium"],
            },
            "barcodes": {
                "patterns": [
                    # High confidence patterns
                    re.compile(r"barcodes\.tsv(\.gz)?$", re.IGNORECASE),
                    re.compile(r"barcodes\.txt(\.gz)?$", re.IGNORECASE),
                    re.compile(r"barcodes\.csv(\.gz)?$", re.IGNORECASE),
                    # Medium confidence patterns
                    re.compile(r".*_barcode.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*-barcode.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*\.barcode.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    # Lower confidence patterns
                    re.compile(r".*barcode.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(
                        r".*(cell|bc).*id.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE
                    ),
                ],
                "base_score": 1.0,
                "boost_keywords": ["cell", "10x", "chromium", "droplet"],
            },
            "features": {
                "patterns": [
                    # High confidence patterns
                    re.compile(r"features\.tsv(\.gz)?$", re.IGNORECASE),
                    re.compile(r"genes\.tsv(\.gz)?$", re.IGNORECASE),
                    re.compile(r"features\.txt(\.gz)?$", re.IGNORECASE),
                    re.compile(r"genes\.txt(\.gz)?$", re.IGNORECASE),
                    # Medium confidence patterns
                    re.compile(r".*_feature.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*_gene.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*-feature.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*-gene.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    # Lower confidence patterns
                    re.compile(r".*feature.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*gene.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*annotation.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                ],
                "base_score": 1.0,
                "boost_keywords": ["gene", "ensembl", "symbol", "annotation", "10x"],
            },
            "h5_data": {
                "patterns": [
                    # High confidence patterns
                    re.compile(r".*\.h5$", re.IGNORECASE),
                    re.compile(r".*\.h5ad$", re.IGNORECASE),
                    re.compile(r".*\.hdf5$", re.IGNORECASE),
                    # Medium confidence patterns
                    re.compile(r".*_filtered.*\.h5$", re.IGNORECASE),
                    re.compile(r".*_raw.*\.h5$", re.IGNORECASE),
                    re.compile(r".*matrix.*\.h5$", re.IGNORECASE),
                ],
                "base_score": 1.0,
                "boost_keywords": ["filtered", "raw", "matrix", "10x", "chromium"],
            },
            "expression": {
                "patterns": [
                    # High confidence patterns for expression files
                    re.compile(
                        r".*(expr|expression|count|fpkm|tpm|rpkm).*\.(txt|csv|tsv)(\.gz)?$",
                        re.IGNORECASE,
                    ),
                    # Medium confidence patterns
                    re.compile(r".*_counts?\.(txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*_expr\.(txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*_data\.(txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    # Lower confidence patterns (generic data files)
                    re.compile(r".*\.(txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                ],
                "base_score": 0.5,  # Lower base score for generic expression
                "boost_keywords": ["normalized", "filtered", "processed", "log"],
            },
            "archive": {
                "patterns": [
                    re.compile(r".*\.tar(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*\.zip$", re.IGNORECASE),
                    re.compile(r".*\.rar$", re.IGNORECASE),
                ],
                "base_score": 0.8,
                "boost_keywords": ["raw", "supplementary", "all", "complete"],
            },
        }

        return patterns

    def _extract_all_supplementary_urls(
        self, metadata: Dict[str, Any], gsm_id: str
    ) -> List[str]:
        """
        Extract all supplementary file URLs from metadata using flexible key detection.

        Args:
            metadata: Sample metadata dictionary
            gsm_id: GEO sample ID for logging

        Returns:
            List[str]: All supplementary file URLs found
        """
        file_urls = []

        # Look for various supplementary file keys (case-insensitive)
        supplementary_key_patterns = [
            re.compile(r".*supplement.*file.*", re.IGNORECASE),
            re.compile(r".*suppl.*file.*", re.IGNORECASE),
            re.compile(r".*additional.*file.*", re.IGNORECASE),
            re.compile(r".*raw.*file.*", re.IGNORECASE),
            re.compile(r".*data.*file.*", re.IGNORECASE),
        ]

        # Find matching keys
        matching_keys = []
        for key in metadata.keys():
            for pattern in supplementary_key_patterns:
                if pattern.match(key):
                    matching_keys.append(key)
                    break

        if not matching_keys:
            # Fallback to any key containing 'supplementary'
            matching_keys = [
                key for key in metadata.keys() if "supplement" in key.lower()
            ]

        logger.debug(f"Found supplementary keys for {gsm_id}: {matching_keys}")

        # Extract URLs from matching keys
        for key in matching_keys:
            urls = metadata[key]
            if isinstance(urls, str):
                urls = [urls]
            elif not isinstance(urls, list):
                continue

            for url in urls:
                if url and isinstance(url, str) and ("http" in url or "ftp" in url):
                    file_urls.append(url)

        logger.info(f"Extracted {len(file_urls)} supplementary file URLs for {gsm_id}")
        return file_urls

    def _classify_single_file(
        self, filename: str, url: str, patterns: Dict
    ) -> Dict[str, float]:
        """
        Classify a single file using pattern matching and scoring.

        Args:
            filename: Name of the file
            url: Full URL of the file
            patterns: File type patterns dictionary

        Returns:
            Dict[str, float]: Scores for each file type
        """
        scores = {}
        filename_lower = filename.lower()
        url_lower = url.lower()

        for file_type, type_config in patterns.items():
            type_patterns = type_config["patterns"]
            base_score = type_config["base_score"]
            boost_keywords = type_config.get("boost_keywords", [])

            max_pattern_score = 0.0

            # Check each pattern for this file type
            for i, pattern in enumerate(type_patterns):
                if pattern.search(filename):
                    # Higher index patterns have lower confidence
                    pattern_confidence = 1.0 - (i * 0.1)  # Decreasing confidence
                    max_pattern_score = max(max_pattern_score, pattern_confidence)

            if max_pattern_score > 0:
                # Apply base score
                total_score = base_score * max_pattern_score

                # Apply keyword boosts
                keyword_boost = 0.0
                for keyword in boost_keywords:
                    if keyword in filename_lower or keyword in url_lower:
                        keyword_boost += 0.1

                total_score += keyword_boost
                scores[file_type] = min(total_score, 2.0)  # Cap at 2.0

        return scores

    def _validate_10x_trio_completeness(
        self, classified_files: Dict[str, str], gsm_id: str
    ) -> None:
        """
        Validate and report on 10X Genomics file trio completeness.

        Args:
            classified_files: Dictionary of classified file types and URLs
            gsm_id: GEO sample ID for logging
        """
        required_10x_files = {"matrix", "barcodes", "features"}
        found_10x_files = set(classified_files.keys()) & required_10x_files
        missing_10x_files = required_10x_files - found_10x_files

        if len(found_10x_files) == 3:
            logger.debug(f"Complete 10X trio found for {gsm_id} ✓")
        elif len(found_10x_files) >= 1:
            logger.warning(
                f"Incomplete 10X trio for {gsm_id}. Found: {list(found_10x_files)}, Missing: {list(missing_10x_files)}"
            )

            # Suggest alternatives
            if "h5_data" in classified_files:
                logger.debug(f"H5 format available as alternative for {gsm_id}")
            elif "expression" in classified_files:
                logger.debug(
                    f"Generic expression file available as fallback for {gsm_id}"
                )
        else:
            logger.debug(f"No 10X format files detected for {gsm_id}")

    def _download_and_combine_single_cell_files(
        self, supplementary_files_info: Dict[str, str], gsm_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Download and combine single-cell format files (matrix, barcodes, features).

        Args:
            supplementary_files_info: Dictionary mapping file types to URLs
            gsm_id: GEO sample ID

        Returns:
            DataFrame: Combined single-cell expression matrix or None
        """
        try:
            logger.debug(f"Downloading and combining single-cell files for {gsm_id}")

            # Check if we have the trio of 10X files (matrix, barcodes, features)
            if all(
                key in supplementary_files_info
                for key in ["matrix", "barcodes", "features"]
            ):
                logger.debug(f"Found complete 10X trio for {gsm_id}")
                return self._download_10x_trio(supplementary_files_info, gsm_id)

            # Check for H5 format (contains everything in one file)
            elif "h5_data" in supplementary_files_info:
                logger.debug(f"Found H5 data for {gsm_id}")
                return self._download_h5_file(
                    supplementary_files_info["h5_data"], gsm_id
                )

            # Check for single expression matrix file
            elif "expression" in supplementary_files_info:
                logger.debug(f"Found expression file for {gsm_id}")
                return self._download_single_expression_file(
                    supplementary_files_info["expression"], gsm_id
                )

            # Try individual files
            elif "matrix" in supplementary_files_info:
                logger.debug(f"Found matrix file only for {gsm_id}")
                return self._download_single_expression_file(
                    supplementary_files_info["matrix"], gsm_id
                )

            else:
                logger.warning(f"No suitable file combination found for {gsm_id}")
                return None

        except Exception as e:
            logger.error(f"Error downloading and combining files for {gsm_id}: {e}")
            return None

    def _download_10x_trio(
        self, files_info: Dict[str, str], gsm_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Download and combine 10X format trio (matrix, barcodes, features).

        Args:
            files_info: Dictionary with 'matrix', 'barcodes', 'features' URLs
            gsm_id: GEO sample ID

        Returns:
            DataFrame: Combined 10X expression matrix or None
        """
        try:
            logger.debug(f"Processing 10X trio for {gsm_id}")

            # Download all three files
            local_files = {}
            for file_type, url in files_info.items():
                if file_type in ["matrix", "barcodes", "features"]:
                    filename = url.split("/")[-1]
                    local_path = self.cache_dir / f"{gsm_id}_{file_type}_{filename}"

                    if not local_path.exists():
                        logger.debug(f"Downloading {file_type} file: {url}")
                        if self.geo_downloader.download_file(url, local_path):
                            local_files[file_type] = local_path
                        else:
                            logger.error(f"Failed to download {file_type} file")
                            return None
                    else:
                        logger.debug(f"Using cached {file_type} file: {local_path}")
                        local_files[file_type] = local_path

            if len(local_files) != 3:
                logger.error(f"Could not download all three 10X files for {gsm_id}")
                return None

            # Parse matrix file
            try:
                import scipy.io as sio
            except ImportError:
                logger.error(
                    "scipy is required for parsing 10X matrix files but not available"
                )
                return None

            matrix_file = local_files["matrix"]
            logger.info(f"Parsing matrix file: {matrix_file}")

            # Read the sparse matrix with enhanced error handling
            try:
                if matrix_file.name.endswith(".gz"):
                    import gzip

                    with gzip.open(matrix_file, "rt") as f:
                        matrix = sio.mmread(f)
                else:
                    matrix = sio.mmread(matrix_file)

            except (gzip.BadGzipFile, EOFError) as e:
                logger.error(f"Gzip corruption detected for {gsm_id}: {e}")
                logger.error(f"Removing corrupted cache: {matrix_file}")
                if matrix_file.exists():
                    matrix_file.unlink()
                return None

            except OSError as e:
                logger.error(f"File I/O error processing {gsm_id}: {e}")
                logger.error(f"Matrix file path: {matrix_file}")
                return None

            # Convert to dense format and transpose (10X format is genes x cells, we want cells x genes)
            if hasattr(matrix, "todense"):
                matrix_dense = matrix.todense()
            else:
                matrix_dense = matrix

            # Transpose so that cells are rows and genes are columns
            matrix_dense = matrix_dense.T
            logger.debug(f"Matrix shape after transpose: {matrix_dense.shape}")

            # Read barcodes
            barcodes_file = local_files["barcodes"]
            logger.debug(f"Reading barcodes from: {barcodes_file}")
            cell_ids = []

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
                cell_ids = [f"{gsm_id}_cell_{i}" for i in range(matrix_dense.shape[0])]

            # Read features
            features_file = local_files["features"]
            logger.info(f"Reading features from: {features_file}")
            gene_ids = []
            gene_names = []

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
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Create DataFrame with proper cell and gene names
            if not cell_ids or len(cell_ids) != matrix_dense.shape[0]:
                cell_ids = [f"{gsm_id}_cell_{i}" for i in range(matrix_dense.shape[0])]
            else:
                cell_ids = [f"{gsm_id}_{cell_id}" for cell_id in cell_ids]

            if not gene_names or len(gene_names) != matrix_dense.shape[1]:
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Create the final DataFrame
            df = pd.DataFrame(matrix_dense, index=cell_ids, columns=gene_names)

            # Deduplicate gene columns if necessary
            if df.columns.duplicated().any():
                duplicates = df.columns[df.columns.duplicated()].unique()
                logger.warning(
                    f"{gsm_id}: Found {len(duplicates)} duplicate gene IDs. "
                    f"Aggregating by sum. Examples: {list(duplicates[:5])}"
                )
                original_shape = df.shape
                # Aggregate duplicates by summing (biologically sound for counts)
                df = df.T.groupby(level=0).sum().T
                logger.info(
                    f"{gsm_id}: Aggregated duplicate genes. "
                    f"Shape: {original_shape} → {df.shape}"
                )

            logger.info(f"Successfully created 10X DataFrame for {gsm_id}: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error processing 10X trio for {gsm_id}: {e}")
            return None

    def _download_h5_file(self, url: str, gsm_id: str) -> Optional[pd.DataFrame]:
        """
        Download and parse H5 format single-cell data.

        Args:
            url: URL to H5 file
            gsm_id: GEO sample ID

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            logger.info(f"Processing H5 file for {gsm_id}")

            filename = url.split("/")[-1]
            local_path = self.cache_dir / f"{gsm_id}_h5_{filename}"

            # Download file
            if not local_path.exists():
                logger.info(f"Downloading H5 file: {url}")
                if not self.geo_downloader.download_file(url, local_path):
                    logger.error("Failed to download H5 file")
                    return None
            else:
                logger.info(f"Using cached H5 file: {local_path}")

            # Parse using geo_parser
            matrix = self.geo_parser.parse_supplementary_file(local_path)
            if matrix is not None and not matrix.empty:
                # Add sample prefix to cell names
                matrix.index = [f"{gsm_id}_{idx}" for idx in matrix.index]
                logger.info(f"Successfully parsed H5 file for {gsm_id}: {matrix.shape}")
                return matrix

            return None

        except Exception as e:
            logger.error(f"Error processing H5 file for {gsm_id}: {e}")
            return None

    def _determine_transpose_biologically(
        self,
        matrix: pd.DataFrame,
        gsm_id: str,
        geo_id: Optional[str] = None,
        data_type_hint: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Determine transpose using biological knowledge instead of naive shape comparison.

        Biological Rules:
        1. Genes: Always 10,000-60,000 in human/mouse datasets
        2. Samples: Bulk RNA-seq typically 2-200
        3. Cells: Single-cell typically 100-200,000

        Args:
            matrix: Input DataFrame
            gsm_id: Sample ID for logging
            geo_id: GEO ID for data type context
            data_type_hint: Optional explicit data type override

        Returns:
            Tuple of (should_transpose, reason)
        """
        n_rows, n_cols = matrix.shape

        logger.debug(f"Transpose decision for {gsm_id}: shape={n_rows}×{n_cols}")

        # Rule 1: If one dimension clearly genes (>50K), transpose so genes become columns
        if n_rows > 50000 and n_cols < 50000:
            reason = f"Large row count ({n_rows}) indicates genes as rows, transposing to samples/cells×genes"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return True, reason

        if n_cols > 50000 and n_rows < 50000:
            reason = f"Large column count ({n_cols}) indicates genes as columns, keeping as samples/cells×genes"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return False, reason

        # Rule 2: Both dimensions large (>10K) → likely cells × genes (single-cell)
        if n_rows > 10000 and n_cols > 10000:
            reason = f"Both dimensions large ({n_rows}×{n_cols}) → likely cells×genes format (single-cell)"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return False, reason

        # Rule 2b: Many observations (>10K) with few variables → likely gene panel (cells×genes)
        # This catches targeted sequencing panels before Rule 3 misidentifies them
        if n_rows > 10000 and n_cols >= 100 and n_cols < 10000:
            reason = f"Many observations, moderate variables ({n_rows}×{n_cols}) → likely cells×genes (gene panel or filtered data)"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return False, reason

        # Rule 3: One dimension very small (<1K), other large (>10K) → likely samples × genes or bulk RNA-seq
        if n_rows < 1000 and n_cols > 10000:
            reason = f"Few rows, many columns ({n_rows}×{n_cols}) → likely samples×genes format (bulk RNA-seq)"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return False, reason

        # Note: We removed the automatic transpose for n_cols < 1000 and n_rows > 10000
        # because it conflicts with gene panels (many cells, few genes).
        # This case will fall through to data type detection or conservative fallback.

        # Rule 4: Use data type detection for edge cases
        data_type = data_type_hint
        if not data_type and geo_id:
            try:
                # Get cached metadata if available using validated retrieval
                stored_metadata = self.data_manager._get_geo_metadata(geo_id)
                if stored_metadata:
                    data_type = self._determine_data_type_from_metadata(
                        stored_metadata["metadata"]
                    )
                    logger.debug(f"Detected data type from metadata: {data_type}")
            except Exception as e:
                logger.warning(
                    f"Could not get data type for biological transpose guidance: {e}"
                )

        if data_type:
            if data_type == "bulk_rna_seq":
                # Bulk: prefer samples×genes (fewer samples)
                if n_rows < n_cols:
                    reason = f"Bulk RNA-seq: {n_rows} samples × {n_cols} genes (likely correct orientation)"
                    logger.info(f"Transpose decision for {gsm_id}: {reason}")
                    return False, reason
                else:
                    reason = f"Bulk RNA-seq: {n_rows}×{n_cols} likely genes×samples, transposing to samples×genes"
                    logger.info(f"Transpose decision for {gsm_id}: {reason}")
                    return True, reason

            elif data_type == "single_cell_rna_seq":
                # Single-cell: prefer cells×genes (more cells than genes in small datasets)
                if n_rows >= n_cols:
                    reason = f"Single-cell: {n_rows} cells × {n_cols} genes (likely correct orientation)"
                    logger.info(f"Transpose decision for {gsm_id}: {reason}")
                    return False, reason
                else:
                    reason = f"Single-cell: {n_rows}×{n_cols} likely genes×cells, transposing to cells×genes"
                    logger.info(f"Transpose decision for {gsm_id}: {reason}")
                    return True, reason

        # Rule 5: Conservative fallback - only transpose if very confident
        # This replaces the old naive heuristic with a much more conservative approach
        # Use higher threshold (>100x) to avoid misidentifying large bulk studies as needing transpose
        if n_cols > n_rows * 100:  # >100x more columns than rows (extreme imbalance)
            reason = f"Extreme imbalance ({n_rows}×{n_cols}, {n_cols/n_rows:.1f}x) suggests genes as rows, transposing"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return True, reason
        else:
            reason = f"Ambiguous shape ({n_rows}×{n_cols}), defaulting to no transpose (safer - assume samples/cells × genes)"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return False, reason

    def _download_single_expression_file(
        self, url: str, gsm_id: str, geo_id: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Download and parse a single expression file with enhanced support.

        Args:
            url: URL to expression file
            gsm_id: GEO sample ID

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            logger.info(f"Processing single expression file for {gsm_id}")

            filename = url.split("/")[-1]
            local_path = self.cache_dir / f"{gsm_id}_expr_{filename}"

            # Download file using helper downloader (supports FTP)
            if not local_path.exists():
                logger.info(f"Downloading expression file: {url}")
                if not self.geo_downloader.download_file(url, local_path):
                    logger.error("Failed to download expression file")
                    return None
            else:
                logger.info(f"Using cached expression file: {local_path}")

            # Parse using geo_parser for better format support
            matrix = self.geo_parser.parse_supplementary_file(local_path)
            if matrix is not None and not matrix.empty:
                # Use biology-aware transpose logic instead of naive shape comparison
                should_transpose, reason = self._determine_transpose_biologically(
                    matrix=matrix, gsm_id=gsm_id, geo_id=geo_id
                )

                logger.info(f"Transpose decision for {gsm_id}: {reason}")

                if should_transpose:
                    matrix = matrix.T
                    logger.debug(f"Matrix transposed: {matrix.shape}")

                # Add sample prefix to row names
                matrix.index = [f"{gsm_id}_{idx}" for idx in matrix.index]

                logger.info(
                    f"Successfully parsed expression file for {gsm_id}: {matrix.shape}"
                )
                return matrix

            return None

        except Exception as e:
            logger.error(f"Error processing expression file for {gsm_id}: {e}")
            return None

    def _validate_matrices(
        self, sample_matrices: Dict[str, Optional[pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Validate downloaded matrices and filter out invalid ones using multithreading.

        Args:
            sample_matrices: Dictionary of sample matrices

        Returns:
            dict: Dictionary of validated matrices
        """
        validated = {}

        # Filter out None matrices first
        valid_matrices = {
            gsm_id: matrix
            for gsm_id, matrix in sample_matrices.items()
            if matrix is not None
        }

        if not valid_matrices:
            logger.warning("No matrices to validate")
            return validated

        logger.info(
            f"Validating {len(valid_matrices)} matrices using multithreading..."
        )

        # Use multithreading for validation - this is the main performance improvement
        with ThreadPoolExecutor(max_workers=min(8, len(valid_matrices))) as executor:
            future_to_sample = {
                executor.submit(
                    self._validate_single_matrix,
                    gsm_id,
                    matrix,
                    # sample_type defaults to "rna" in _validate_single_matrix
                ): gsm_id
                for gsm_id, matrix in valid_matrices.items()
            }

            for future in as_completed(future_to_sample):
                gsm_id = future_to_sample[future]
                try:
                    is_valid, validation_info = future.result()
                    if is_valid:
                        validated[gsm_id] = valid_matrices[gsm_id]
                        logger.info(f"Validated {gsm_id}: {validation_info}")
                    else:
                        logger.warning(f"Skipping {gsm_id}: {validation_info}")
                except Exception as e:
                    logger.error(f"Error validating {gsm_id}: {e}")

        logger.info(f"Validated {len(validated)}/{len(sample_matrices)} matrices")
        return validated

    def _validate_single_matrix(
        self, gsm_id: str, matrix: pd.DataFrame, sample_type: str = "rna"
    ) -> Tuple[bool, str]:
        """
        Validate a single matrix with biology-aware thresholds and type-aware duplicate checking.

        Args:
            gsm_id: Sample ID for logging
            matrix: DataFrame to validate
            sample_type: Data type ("rna", "protein", "vdj", "atac").
                        VDJ data (TCR/BCR) allows duplicate row indices (multi-chain per cell).

        Returns:
            Tuple[bool, str]: (is_valid, info_message)
        """
        try:
            n_obs, n_vars = matrix.shape

            # Rule 1: Must have some observations and variables
            if n_obs == 0 or n_vars == 0:
                return False, f"Empty matrix ({n_obs}×{n_vars})"

            # Rule 1.5: Check for duplicate indices (added for bug fix)
            # Duplicate gene IDs (columns) - WARNING only, since we auto-deduplicate
            if matrix.columns.duplicated().any():
                n_dup = matrix.columns.duplicated().sum()
                logger.warning(
                    f"{gsm_id}: Found {n_dup} duplicate gene IDs. "
                    f"These will be aggregated during loading."
                )
                # Don't fail validation - deduplication happens at loading stage

            # Duplicate cell/sample IDs (rows) - Type-aware validation
            if matrix.index.duplicated().any():
                n_dup = matrix.index.duplicated().sum()
                duplicate_rate = n_dup / len(matrix)

                # VDJ data (TCR/BCR sequencing): Duplicates are EXPECTED
                # One row per receptor chain = multiple rows per cell
                if sample_type == "vdj":
                    logger.info(
                        f"{gsm_id}: VDJ data has {n_dup} repeated cell barcodes "
                        f"({duplicate_rate:.1%} of total) - expected for multi-chain data"
                    )
                    # Continue validation - duplicates are scientifically correct
                else:
                    # Gene expression/protein data: Duplicates indicate corruption
                    return (
                        False,
                        f"Duplicate cell/sample IDs ({n_dup} duplicates, {duplicate_rate:.1%}) "
                        f"- invalid for {sample_type} data",
                    )

            # Rule 2: Biology-aware validation
            # For bulk RNA-seq: few samples (2-500), many genes (10K-60K)
            # For single-cell: many cells (100-200K), many genes (5K-30K)

            if n_vars >= 10000:  # Likely genes dimension is correct
                if n_obs >= 2:  # At least 2 observations (samples or cells)
                    # Use optimized validation for matrix format
                    if not self._is_valid_expression_matrix(matrix):
                        return (
                            False,
                            "Invalid matrix format (non-numeric or all-zero data)",
                        )

                    return True, f"Valid matrix: {n_obs} obs × {n_vars} genes"
                else:
                    return (
                        False,
                        f"Only {n_obs} observation(s) - insufficient for analysis (need at least 2)",
                    )

            elif (
                n_obs >= 10000
            ):  # Many observations, check if genes dimension reasonable
                if n_vars >= 100:  # Reasonable number of variables
                    # Use optimized validation for matrix format
                    if not self._is_valid_expression_matrix(matrix):
                        return (
                            False,
                            "Invalid matrix format (non-numeric or all-zero data)",
                        )

                    return True, f"Valid matrix: {n_obs} obs × {n_vars} vars"
                elif n_vars >= 4 and n_obs > 50000:
                    # Special case: Very high obs count (>50K) with few vars suggests genes×samples
                    # This will be caught and transposed by biology-aware transpose logic
                    # Accept with warning for GSE130036-type cases
                    if not self._is_valid_expression_matrix(matrix):
                        return (
                            False,
                            "Invalid matrix format (non-numeric or all-zero data)",
                        )

                    return (
                        True,
                        f"Valid but unusual matrix: {n_obs} obs × {n_vars} vars (likely needs transpose - genes as obs)",
                    )
                else:
                    return (
                        False,
                        f"Only {n_vars} variables - likely transpose error or corrupted data (need at least 100 vars for >10K obs)",
                    )

            else:  # Both dimensions small - use conservative thresholds
                if n_obs >= 10 and n_vars >= 10:
                    # Use optimized validation for matrix format
                    if not self._is_valid_expression_matrix(matrix):
                        return (
                            False,
                            "Invalid matrix format (non-numeric or all-zero data)",
                        )

                    return (
                        True,
                        f"Small matrix: {n_obs} obs × {n_vars} vars (may be test/subset data)",
                    )
                else:
                    return (
                        False,
                        f"Matrix too small for analysis ({n_obs}×{n_vars}) - need at least 10×10",
                    )

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _is_valid_expression_matrix(self, matrix: pd.DataFrame) -> bool:
        """
        Optimized check if a matrix is a valid expression matrix.
        Uses sampling and vectorized operations for better performance on large DataFrames.

        Args:
            matrix: DataFrame to validate

        Returns:
            bool: True if valid expression matrix
        """
        try:
            # Check if it's a DataFrame
            if not isinstance(matrix, pd.DataFrame):
                return False

            # Fast check: ensure we have numeric data by checking dtypes
            # This is much faster than select_dtypes() for large DataFrames
            numeric_dtypes = set(
                ["int16", "int32", "int64", "float16", "float32", "float64"]
            )
            has_numeric = any(str(dtype) in numeric_dtypes for dtype in matrix.dtypes)

            if not has_numeric:
                return False

            # For large matrices, use sampling to speed up validation
            if matrix.size > 1_000_000:  # > 1M cells
                # Sample 10% of the data or max 100k cells for validation
                sample_size = min(100_000, int(matrix.size * 0.1))

                # Get a random sample of the flattened matrix
                flat_sample = matrix.select_dtypes(include=[np.number]).values.flatten()
                if len(flat_sample) > sample_size:
                    indices = np.random.choice(
                        len(flat_sample), sample_size, replace=False
                    )
                    sample_data = flat_sample[indices]
                else:
                    sample_data = flat_sample

                # Check for non-negative values in sample
                if np.any(sample_data < 0):
                    logger.warning(
                        "Matrix contains negative values (detected in sample)"
                    )

                # Check for reasonable value ranges in sample
                max_val = np.max(sample_data)
                if max_val > 1e6:
                    logger.info(
                        "Matrix contains very large values (possibly raw counts)"
                    )

            else:
                # For smaller matrices, do full validation but with optimized operations
                numeric_data = matrix.select_dtypes(include=[np.number])

                # Use numpy operations which are faster than pandas
                values = numeric_data.values

                # Check for non-negative values using numpy
                if np.any(values < 0):
                    logger.warning("Matrix contains negative values")

                # Check for reasonable value ranges using numpy
                max_val = np.max(values)
                if max_val > 1e6:
                    logger.info(
                        "Matrix contains very large values (possibly raw counts)"
                    )

            return True

        except Exception as e:
            logger.error(f"Error validating matrix: {e}")
            return False

    def _store_samples_as_anndata(
        self,
        validated_matrices: Dict[str, pd.DataFrame],
        gse_id: str,
        metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Store each sample as an individual AnnData object in DataManagerV2.

        Args:
            validated_matrices: Dictionary of validated sample matrices
            gse_id: GEO series ID
            metadata: Metadata from GEO

        Returns:
            List[str]: List of modality names that were successfully stored
        """
        stored_samples = []

        try:
            logger.info(
                f"Storing {len(validated_matrices)} samples as individual AnnData objects"
            )

            for gsm_id, matrix in validated_matrices.items():
                try:
                    # Create unique modality name for this sample
                    modality_name = f"geo_{gse_id.lower()}_sample_{gsm_id.lower()}"

                    # Extract sample-specific metadata
                    sample_metadata = {}
                    if "samples" in metadata and gsm_id in metadata["samples"]:
                        sample_metadata = metadata["samples"][gsm_id]

                    # Create enhanced metadata for this sample
                    enhanced_metadata = {
                        "dataset_id": gse_id,
                        "sample_id": gsm_id,
                        "dataset_type": "GEO_Sample",
                        "parent_dataset": gse_id,
                        "sample_metadata": sample_metadata,
                        "processing_date": pd.Timestamp.now().isoformat(),
                        "data_source": "individual_sample_matrix",
                        "is_preprocessed": False,
                        "needs_concatenation": True,
                    }

                    # Determine adapter based on data characteristics
                    n_obs, n_vars = matrix.shape
                    if n_vars > 5000:  # High gene count suggests single-cell
                        adapter_name = "transcriptomics_single_cell"
                    else:
                        adapter_name = "transcriptomics_bulk"

                    # Load as modality in DataManagerV2
                    adata = self.data_manager.load_modality(
                        name=modality_name,
                        source=matrix,
                        adapter=adapter_name,
                        validate=False,  # Skip validation for individual samples
                        **enhanced_metadata,
                    )

                    # Save to workspace
                    save_path = f"{gse_id.lower()}_{gsm_id.lower()}_raw.h5ad"
                    # Create directory if needed
                    (self.data_manager.data_dir / gse_id.lower()).mkdir(exist_ok=True)
                    self.data_manager.save_modality(name=modality_name, path=save_path)

                    stored_samples.append(modality_name)
                    logger.info(
                        f"Stored sample {gsm_id} as modality '{modality_name}' ({adata.shape})"
                    )

                except Exception as e:
                    logger.error(f"Failed to store sample {gsm_id}: {e}")
                    continue

            logger.info(
                f"Successfully stored {len(stored_samples)} samples as individual AnnData objects"
            )
            return stored_samples

        except Exception as e:
            logger.error(f"Error storing samples as AnnData: {e}")
            return stored_samples

    def _analyze_gene_coverage_and_decide_join(
        self, sample_modalities: List[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze gene coverage variance across samples and decide optimal join strategy.

        This method examines the coefficient of variation (CV) in gene counts across
        samples to intelligently select between inner join (intersection) and outer
        join (union) concatenation strategies.

        Args:
            sample_modalities: List of modality names to analyze

        Returns:
            Tuple of (use_intersecting_genes_only, analysis_metadata)
                - use_intersecting_genes_only: True for inner join, False for outer join
                - analysis_metadata: Dict with statistics and reasoning
        """
        from datetime import datetime

        try:
            # Collect gene counts from all samples
            gene_counts = []
            for modality in sample_modalities:
                try:
                    adata = self.data_manager.get_modality(modality)
                    gene_counts.append(adata.n_vars)
                except Exception as e:
                    logger.warning(f"Could not get gene count for {modality}: {e}")
                    continue

            if not gene_counts:
                logger.warning("No valid gene counts found, defaulting to inner join")
                return True, {
                    "decision": "inner",
                    "reasoning": "No valid samples found",
                }

            # Calculate statistics
            min_genes = int(np.min(gene_counts))
            max_genes = int(np.max(gene_counts))
            mean_genes = float(np.mean(gene_counts))
            std_genes = float(np.std(gene_counts))
            cv = std_genes / mean_genes if mean_genes > 0 else 0.0

            # Decision logic: Use outer join if high variability
            # Check both CV and absolute range ratio for robustness
            VARIANCE_THRESHOLD = 0.20  # Lowered from 0.30 to be more conservative
            RANGE_RATIO_THRESHOLD = (
                1.5  # Max/min ratio - if > 1.5x difference, use outer join
            )

            range_ratio = max_genes / min_genes if min_genes > 0 else float("inf")
            use_inner_join = (
                cv <= VARIANCE_THRESHOLD and range_ratio <= RANGE_RATIO_THRESHOLD
            )

            # Build decision metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "n_samples_analyzed": len(gene_counts),
                "min_genes": min_genes,
                "max_genes": max_genes,
                "mean_genes": mean_genes,
                "std_genes": std_genes,
                "coefficient_variation": cv,
                "range_ratio": range_ratio,
                "variance_threshold": VARIANCE_THRESHOLD,
                "range_ratio_threshold": RANGE_RATIO_THRESHOLD,
                "decision": "inner" if use_inner_join else "outer",
                "reasoning": (
                    f"Gene coverage CV={cv:.1%} <= {VARIANCE_THRESHOLD:.1%} AND range ratio={range_ratio:.2f}x <= {RANGE_RATIO_THRESHOLD:.2f}x: Consistent coverage"
                    if use_inner_join
                    else (
                        f"Gene coverage variability detected: CV={cv:.1%} (threshold: {VARIANCE_THRESHOLD:.1%}) OR range ratio={range_ratio:.2f}x (threshold: {RANGE_RATIO_THRESHOLD:.2f}x) - using union to preserve all genes"
                    )
                ),
            }

            # LOG DECISION TO USER (INFO level)
            logger.info("=" * 70)
            logger.info("🔍 CONCATENATION STRATEGY DECISION")
            logger.info("=" * 70)
            logger.info(
                f"📊 Analyzing {len(sample_modalities)} samples for gene coverage..."
            )
            logger.info(
                f"   Gene count range: {min_genes:,} - {max_genes:,} ({range_ratio:.2f}x difference)"
            )
            logger.info(f"   Mean: {mean_genes:,.0f} ± {std_genes:,.0f}")
            logger.info(f"   Coefficient of Variation: {cv:.1%}")
            logger.info("")
            logger.info("📐 Decision Criteria:")
            logger.info(
                f"   CV threshold: {VARIANCE_THRESHOLD:.1%}, Range ratio threshold: {RANGE_RATIO_THRESHOLD:.2f}x"
            )
            logger.info("")

            if use_inner_join:
                logger.info("✓ Selected: INNER JOIN (intersection of genes)")
                logger.info(f"  📌 Reason: {metadata['reasoning']}")
                logger.info(
                    "  📍 Effect: Only genes present in ALL samples will be retained"
                )
                logger.info(
                    "  ⚠️  Warning: Genes unique to some samples will be excluded"
                )
            else:
                logger.info("⚠️  VARIABILITY DETECTED")
                logger.info("✓ Selected: OUTER JOIN (union of all genes)")
                logger.info(f"  📌 Reason: {metadata['reasoning']}")
                logger.info(
                    "  📍 Effect: ALL genes included, missing values filled with zeros"
                )
                logger.info("  ℹ️  Note: This preserves maximum biological information")

            logger.info("=" * 70)

            return use_inner_join, metadata

        except Exception as e:
            logger.error(f"Error analyzing gene coverage: {e}")
            logger.warning("Defaulting to outer join for safety")
            return False, {
                "decision": "outer",
                "reasoning": f"Error during analysis: {str(e)}",
                "error": str(e),
            }

    def _concatenate_stored_samples(
        self,
        geo_id: str,
        stored_samples: List[str],
        use_intersecting_genes_only: bool = None,
    ) -> Optional[pd.DataFrame]:
        """
        Concatenate stored AnnData samples using ConcatenationService with intelligent strategy selection.

        This function delegates to ConcatenationService to eliminate code duplication
        while adding intelligent auto-detection of the optimal join strategy based on
        gene coverage variance. When use_intersecting_genes_only is None (default),
        automatically analyzes gene coverage and selects the appropriate join type.

        Args:
            geo_id: GEO series ID
            stored_samples: List of modality names that were stored
            use_intersecting_genes_only: Join strategy selection:
                - None (default): Auto-detect based on gene coverage variance
                - True: Use inner join (intersection - only common genes)
                - False: Use outer join (union - all genes with zero-filling)

        Returns:
            AnnData: Concatenated AnnData object or None if concatenation fails
        """
        from datetime import datetime

        try:
            # Import and initialize ConcatenationService
            from lobster.tools.concatenation_service import ConcatenationService

            concat_service = ConcatenationService(self.data_manager)

            logger.info(
                f"Using ConcatenationService to concatenate {len(stored_samples)} stored samples for {geo_id}"
            )

            # AUTO-DETECT join strategy if not explicitly specified
            analysis_metadata = {}
            auto_detected = False

            if use_intersecting_genes_only is None:
                logger.info(
                    "No explicit join strategy specified - performing intelligent auto-detection..."
                )
                use_intersecting_genes_only, analysis_metadata = (
                    self._analyze_gene_coverage_and_decide_join(stored_samples)
                )
                auto_detected = True
                logger.info(
                    f"Auto-detection complete: using {'INNER' if use_intersecting_genes_only else 'OUTER'} join"
                )
            else:
                # Manual specification
                join_type = "inner" if use_intersecting_genes_only else "outer"
                logger.info(
                    f"Using explicitly specified join strategy: {join_type.upper()} join"
                )
                analysis_metadata = {
                    "decision": join_type,
                    "reasoning": f"Explicitly specified by user: use_intersecting_genes_only={use_intersecting_genes_only}",
                    "timestamp": datetime.now().isoformat(),
                }

            # Perform concatenation
            concatenated_adata, statistics = concat_service.concatenate_from_modalities(
                modality_names=stored_samples,
                output_name=None,  # Don't store, just return
                use_intersecting_genes_only=use_intersecting_genes_only,
                batch_key="batch",
            )

            # PROVENANCE TRACKING: Log to DataManager tool usage history
            provenance_info = {
                **analysis_metadata,
                **statistics,
                "samples_concatenated": len(stored_samples),
                "resulting_shape": (
                    concatenated_adata.n_obs,
                    concatenated_adata.n_vars,
                ),
                "auto_detected": auto_detected,
                "timestamp": datetime.now().isoformat(),
            }

            self.data_manager.log_tool_usage(
                tool_name="concatenate_geo_samples",
                parameters={
                    "geo_id": geo_id,
                    "n_samples": len(stored_samples),
                    "join_strategy": (
                        "inner" if use_intersecting_genes_only else "outer"
                    ),
                    "auto_detected": auto_detected,
                },
                result=provenance_info,
            )

            # METADATA STORAGE: Store concatenation decision for supervisor access
            modality_name = f"geo_{geo_id.lower()}"
            concatenation_info = {
                "join_strategy": "inner" if use_intersecting_genes_only else "outer",
                "auto_detected": auto_detected,
                "analysis": analysis_metadata,
                "statistics": statistics,
                "quality_impact": (
                    "Only genes present in all samples retained"
                    if use_intersecting_genes_only
                    else "All genes included, missing values filled with zeros"
                ),
                "provenance_tracked": True,
                "timestamp": datetime.now().isoformat(),
            }

            # Update existing entry or create new one with concatenation decision
            existing_entry = self.data_manager._get_geo_metadata(modality_name)
            if existing_entry:
                existing_entry["concatenation_decision"] = concatenation_info
                self.data_manager.metadata_store[modality_name] = existing_entry
            else:
                # No existing entry - store concatenation decision in proper structure
                # Note: This shouldn't happen if metadata was stored earlier, but handle it defensively
                self.data_manager.metadata_store[modality_name] = {
                    "concatenation_decision": concatenation_info,
                    "stored_by": "_handle_multi_sample_concatenation",
                    "fetch_timestamp": datetime.now().isoformat(),
                }

            logger.info(
                "✓ Concatenation decision stored in metadata_store for supervisor access"
            )
            logger.info("✓ Provenance tracked in tool_usage_history")
            logger.info(f"ConcatenationService completed: {statistics}")

            return concatenated_adata

        except Exception as e:
            logger.error(
                f"Error concatenating stored samples using ConcatenationService: {e}"
            )
            return None

    # ████████████████████████████████████████████████████████████████████████████████
    # ██                                                                            ██
    # ██                        LEGACY FUNCTIONS - REVIEW NEEDED                   ██
    # ██                                                                            ██
    # ████████████████████████████████████████████████████████████████████████████████

    def _store_single_sample_as_modality(
        self, gsm_id: str, matrix: pd.DataFrame, gsm
    ) -> str:  # FIXME
        """Store single sample as modality in DataManagerV2."""
        try:
            modality_name = f"geo_sample_{gsm_id.lower()}"

            # Extract sample metadata
            sample_metadata = {}
            if hasattr(gsm, "metadata"):
                for key, value in gsm.metadata.items():
                    if isinstance(value, list) and len(value) == 1:
                        sample_metadata[key] = value[0]
                    else:
                        sample_metadata[key] = value

            enhanced_metadata = {
                "sample_id": gsm_id,
                "dataset_type": "GEO_Sample",
                "sample_metadata": sample_metadata,
                "processing_date": pd.Timestamp.now().isoformat(),
                "data_source": "single_sample",
            }

            # Determine adapter based on data characteristics
            n_obs, n_vars = matrix.shape
            if n_vars > 5000:  # High gene count suggests single-cell
                adapter_name = "transcriptomics_single_cell"
            else:
                adapter_name = "transcriptomics_bulk"

            adata = self.data_manager.load_modality(
                name=modality_name,
                source=matrix,
                adapter=adapter_name,
                validate=True,
                **enhanced_metadata,
            )

            # Save to workspace
            save_path = f"{gsm_id.lower()}_sample.h5ad"
            self.data_manager.save_modality(modality_name, save_path)

            return f"""Successfully downloaded single-cell sample {gsm_id}!

📊 Modality: '{modality_name}' ({adata.n_obs} cells × {adata.n_vars} genes)
🔬 Adapter: {adapter_name}
💾 Saved to: {save_path}
📈 Ready for single-cell analysis!"""

        except Exception as e:
            logger.error(f"Error storing sample {gsm_id}: {e}")
            return f"Error storing sample {gsm_id}: {str(e)}"

    def _download_supplementary_file(  # FIXME
        self, url: str, gsm_id: str, geo_id: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Professional download and parse of supplementary files with FTP support.

        Args:
            url: URL to supplementary file (supports HTTP/HTTPS/FTP)
            gsm_id: GEO sample ID
            geo_id: Optional GEO dataset ID for transpose logic context

        Returns:
            DataFrame: Parsed matrix or None
        """
        try:
            logger.info(f"Downloading supplementary file for {gsm_id}: {url}")

            # Extract filename and create local path
            filename = url.split("/")[-1]
            local_file = self.cache_dir / f"{gsm_id}_suppl_{filename}"

            # Use helper downloader for better FTP/HTTP support and progress tracking
            if not local_file.exists():
                if not self.geo_downloader.download_file(
                    url, local_file, f"Downloading {filename}"
                ):
                    logger.error(f"Failed to download supplementary file: {url}")
                    return None
            else:
                logger.info(f"Using cached supplementary file: {local_file}")

            # Use geo_parser for enhanced format detection and parsing
            matrix = self.geo_parser.parse_supplementary_file(local_file)

            if matrix is not None and not matrix.empty:
                # Use biology-aware transpose logic instead of naive shape comparison
                should_transpose, reason = self._determine_transpose_biologically(
                    matrix=matrix, gsm_id=gsm_id, geo_id=geo_id
                )

                logger.info(f"Transpose decision for {gsm_id}: {reason}")

                if should_transpose:
                    matrix = matrix.T
                    logger.debug(f"Matrix transposed: {matrix.shape}")

                # Add sample prefix to row names for proper identification
                matrix.index = [f"{gsm_id}_{idx}" for idx in matrix.index]

                logger.info(
                    f"Successfully parsed supplementary file for {gsm_id}: {matrix.shape}"
                )
                return matrix
            else:
                logger.warning(
                    f"Could not parse supplementary file or file is empty: {local_file}"
                )
                return None

        except Exception as e:
            logger.error(
                f"Error downloading supplementary file {url} for {gsm_id}: {e}"
            )
            return None

    def _create_placeholder_matrix(  # FIXME
        self, stored_samples: List[str], gse_id: str
    ) -> pd.DataFrame:
        """
        Create a placeholder matrix for immediate compatibility.
        This will contain summary information about the stored samples.

        Args:
            stored_samples: List of modality names that were stored
            gse_id: GEO series ID

        Returns:
            DataFrame: Placeholder matrix with sample information
        """
        try:
            # Collect information about stored samples
            sample_info_list = []
            total_cells = 0
            all_genes = set()

            for modality_name in stored_samples:
                adata = self.data_manager.get_modality(modality_name)
                sample_id = modality_name.split("_")[-1].upper()

                sample_info_list.append(
                    {
                        "sample_id": sample_id,
                        "modality_name": modality_name,
                        "n_cells": adata.n_obs,
                        "n_genes": adata.n_vars,
                        "stored": True,
                    }
                )

                total_cells += adata.n_obs
                all_genes.update(adata.var_names)

            # Create a summary DataFrame
            summary_df = pd.DataFrame(sample_info_list)

            # Create a minimal placeholder matrix with metadata
            # This matrix will have one row per sample with summary statistics
            placeholder_matrix = pd.DataFrame(
                index=summary_df["sample_id"],
                columns=["n_cells", "n_genes", "modality_name", "status"],
            )

            for _, row in summary_df.iterrows():
                placeholder_matrix.loc[row["sample_id"]] = [
                    row["n_cells"],
                    row["n_genes"],
                    row["modality_name"],
                    "stored_for_preprocessing",
                ]

            # Add metadata as attributes
            placeholder_matrix.attrs = {
                "dataset_id": gse_id,
                "total_samples": len(stored_samples),
                "total_cells": total_cells,
                "total_unique_genes": len(all_genes),
                "stored_modalities": stored_samples,
                "note": "Individual samples stored as AnnData objects for preprocessing",
                "concatenation_pending": True,
            }

            logger.info(
                f"Created placeholder matrix with {len(stored_samples)} samples"
            )
            return placeholder_matrix

        except Exception as e:
            logger.error(f"Error creating placeholder matrix: {e}")
            # Return a minimal placeholder
            return pd.DataFrame({"status": ["error"]}, index=[gse_id])

    def _check_workspace_for_processed_data(  # FIXME
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
                data_source="GEO",
                dataset_id=gse_id,
                processing_step="raw_matrix",
            )

            if existing_file and existing_file.exists():
                logger.info(
                    f"Found existing file with professional naming: {existing_file.name}"
                )

                # Load the data
                combined_matrix = pd.read_csv(existing_file, index_col=0)

                # Load associated metadata
                metadata_file = (
                    existing_file.parent
                    / BioinformaticsFileNaming.generate_metadata_filename(
                        existing_file.name
                    )
                )
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)

                logger.info(
                    f"Successfully loaded existing data: {combined_matrix.shape}"
                )
                return combined_matrix, metadata

            # Backward compatibility: check for legacy naming patterns
            data_files = self.data_manager.list_workspace_files()["data"]
            for file_info in data_files:
                if f"{gse_id}_processed" in file_info["name"] and file_info[
                    "name"
                ].endswith(".csv"):
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

                    logger.info(
                        f"Successfully loaded legacy data: {combined_matrix.shape}"
                    )
                    return combined_matrix, metadata

            return None

        except Exception as e:
            logger.warning(f"Error checking workspace for existing data: {e}")
            return None

    def _format_download_response(  # FIXME
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
