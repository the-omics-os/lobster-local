"""
GEO provider implementation for direct GEO database search.

This provider implements direct search capabilities for NCBI's GEO database
using E-utilities, supporting all query patterns from the official API examples.
Searches across GSE (Series), GSM (Samples), GPL (Platforms), and GDS (Datasets).
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

try:
    import GEOparse
except ImportError:
    GEOparse = None

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.base_provider import (
    BasePublicationProvider,
    DatasetType,
    PublicationMetadata,
    PublicationSource,
)
from lobster.tools.providers.geo_utils import (
    GEOAccessionType,
    detect_geo_accession_subtype,
    get_ncbi_geo_url,
    is_geo_sample_accession,
)
from lobster.tools.providers.ncbi_query_builder import (
    GEOQueryBuilder,
)
from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context, handle_ssl_error

logger = get_logger(__name__)


class GEOEntryType(str, Enum):
    """GEO database entry types."""

    SERIES = "gse"  # Series (experiments)
    DATASET = "gds"  # Curated datasets
    PLATFORM = "gpl"  # Platforms
    SAMPLE = "gsm"  # Individual samples


class GEOSearchFilters(BaseModel):
    """Filters for GEO dataset searches."""

    entry_types: Optional[List[str]] = None  # gse, gds, gpl, gsm
    organisms: Optional[List[str]] = None
    platforms: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = (
        None  # {"start": "2023/01/01", "end": "2024/12/31"}
    )
    supplementary_file_types: Optional[List[str]] = None  # ["cel", "txt", "h5"]
    max_results: int = Field(default=20, ge=1, le=5000)
    use_history: bool = True


class GEOProviderConfig(BaseModel):
    """Configuration for GEO provider."""

    # GEO search settings
    max_results: int = Field(default=20, ge=1, le=1000)
    default_entry_types: List[str] = Field(default=["gse", "gds"])

    # NCBI API settings
    email: str = Field(default_factory=lambda: get_settings().NCBI_EMAIL)
    api_key: Optional[str] = ""
    max_retry: int = Field(default=3, ge=1, le=10)
    sleep_time: float = Field(default=0.5, ge=0.1, le=5.0)

    # Result processing settings
    include_summaries: bool = True
    include_publication_links: bool = True
    cache_results: bool = True


class GEOSearchResult(BaseModel):
    """Result from GEO search with history server support."""

    count: int
    ids: List[str]
    web_env: Optional[str] = None
    query_key: Optional[str] = None
    summaries: Optional[List[Dict[str, Any]]] = None
    query: Optional[str] = None


class GEOProvider(BasePublicationProvider):
    """
    GEO provider for direct dataset search via NCBI E-utilities.

    This provider implements all patterns from official GEO API examples:
    - Series search by date range
    - Organism and platform filtering
    - Supplementary file type search
    - History server support (WebEnv/query_key)
    - Comprehensive result metadata extraction
    """

    def __init__(
        self, data_manager: DataManagerV2, config: Optional[GEOProviderConfig] = None
    ):
        """
        Initialize GEO provider.

        Args:
            data_manager: DataManagerV2 instance for provenance tracking
            config: Optional configuration, uses defaults if not provided
        """
        self.data_manager = data_manager
        settings = get_settings()

        # Create config with API key from settings if not provided
        if config is None:
            self.config = GEOProviderConfig(api_key=settings.NCBI_API_KEY)
        else:
            # If config provided but no API key, update it with settings
            if config.api_key is None:
                self.config = config.model_copy(
                    update={"api_key": settings.NCBI_API_KEY}
                )
            else:
                self.config = config

        self.query_builder = GEOQueryBuilder()

        # Initialize cache directory for GEOparse
        self.cache_dir = Path(settings.GEO_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize XML parser
        try:
            import xmltodict

            self.parse_xml = xmltodict.parse
        except ImportError:
            raise ImportError(
                "Could not import xmltodict. Install with: pip install xmltodict"
            )

        # NCBI E-utilities base URLs
        self.base_url_esearch = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        )
        self.base_url_esummary = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        )
        self.base_url_elink = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

        # Session cache for WebEnv management
        self._session_cache: Dict[str, Tuple[str, str, datetime]] = {}

    @property
    def source(self) -> PublicationSource:
        """Return GEO as the publication source."""
        return PublicationSource.GEO

    @property
    def supported_dataset_types(self) -> List[DatasetType]:
        """Return list of dataset types supported by GEO."""
        return [DatasetType.GEO]

    @property
    def priority(self) -> int:
        """
        Return provider priority for capability-based routing.

        GEO has high priority (10) as the authoritative source for GEO datasets.
        Provides comprehensive dataset discovery, metadata extraction, and
        validation for genomics data.

        Returns:
            int: Priority 10 (high priority)
        """
        return 10

    def get_supported_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities supported by GEO provider.

        GEO excels at dataset discovery, metadata extraction and validation,
        and multi-omics integration through GEO SuperSeries. It's the
        authoritative source for GEO datasets but doesn't provide literature
        search or full-text access.

        Supported capabilities:
        - DISCOVER_DATASETS: Search GEO database (db=gds parameter)
        - FIND_LINKED_DATASETS: Reverse lookup (dataset → publication)
        - EXTRACT_METADATA: Parse SOFT files for dataset metadata
        - VALIDATE_METADATA: Check sample counts, platform info, design
        - QUERY_CAPABILITIES: Dynamic capability discovery
        - INTEGRATE_MULTI_OMICS: Map samples across GEO SuperSeries

        Not supported:
        - SEARCH_LITERATURE: No publication search (use PubMedProvider)
        - GET_ABSTRACT: No abstract retrieval
        - GET_FULL_CONTENT: No full-text access
        - EXTRACT_METHODS: No methods extraction
        - EXTRACT_PDF: No PDF processing

        Returns:
            Dict[str, bool]: Capability support mapping
        """
        from lobster.tools.providers.base_provider import ProviderCapability

        return {
            ProviderCapability.SEARCH_LITERATURE: False,
            ProviderCapability.DISCOVER_DATASETS: True,
            ProviderCapability.FIND_LINKED_DATASETS: True,
            ProviderCapability.EXTRACT_METADATA: True,
            ProviderCapability.VALIDATE_METADATA: True,
            ProviderCapability.QUERY_CAPABILITIES: True,
            ProviderCapability.GET_ABSTRACT: False,
            ProviderCapability.GET_FULL_CONTENT: False,
            ProviderCapability.EXTRACT_METHODS: False,
            ProviderCapability.EXTRACT_PDF: False,
            ProviderCapability.INTEGRATE_MULTI_OMICS: True,
        }

    def validate_identifier(self, identifier: str) -> bool:
        """
        Validate GEO identifiers.

        Args:
            identifier: Identifier to validate (GSE, GDS, GPL, etc.)

        Returns:
            bool: True if identifier is valid GEO format
        """
        identifier = identifier.strip().upper()

        # Check for GEO accession patterns
        geo_patterns = [
            r"^GSE\d+$",  # Series
            r"^GDS\d+$",  # Dataset
            r"^GPL\d+$",  # Platform
            r"^GSM\d+$",  # Sample
        ]

        return any(re.match(pattern, identifier) for pattern in geo_patterns)

    def get_supported_features(self) -> Dict[str, bool]:
        """Return features supported by GEO provider."""
        return {
            "literature_search": False,  # GEO is for datasets, not literature
            "dataset_discovery": True,
            "metadata_extraction": True,
            "full_text_access": False,
            "advanced_filtering": True,
            "date_filtering": True,
            "organism_filtering": True,
            "platform_filtering": True,
            "file_type_filtering": True,
        }

    def search_publications(
        self,
        query: str,
        max_results: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Search GEO database (includes GSE series, GSM samples, GPL platforms, and GDS datasets).

        Args:
            query: Search query string
            max_results: Maximum number of results (default: 20)
            filters: Optional filters for GEO search
            **kwargs: Additional parameters

        Returns:
            str: Formatted search results
        """
        logger.info(f"GEO database search: {query[:50]}...")

        try:
            # Convert filters to GEOSearchFilters if provided
            geo_filters = self._convert_filters(filters) if filters else None

            # Set max results
            if geo_filters:
                geo_filters.max_results = min(max_results, self.config.max_results)
            else:
                geo_filters = GEOSearchFilters(
                    max_results=min(max_results, self.config.max_results)
                )

            # Perform GEO search
            search_result = self.search_geo_datasets(query, geo_filters)

            if search_result.count == 0:
                return f"No GEO datasets found for query: {query}"

            # Get detailed summaries if enabled
            if self.config.include_summaries and search_result.ids:
                summaries = self.get_dataset_summaries(search_result)
                search_result.summaries = summaries

            # Format results
            formatted_results = self.format_geo_search_results(search_result, query)

            # Log the search
            self.data_manager.log_tool_usage(
                tool_name="geo_datasets_search",
                parameters={
                    "query": query[:100],
                    "max_results": (
                        geo_filters.max_results if geo_filters else max_results
                    ),
                    "filters": filters,
                },
                description="Direct GEO database search",
            )

            return formatted_results

        except Exception as e:
            logger.exception(f"GEO search error: {e}")
            raise TypeError(f"GEO search error: {str(e)}")

    def find_datasets_from_publication(
        self,
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        **kwargs,
    ) -> str:
        """
        Find GEO datasets by accession (not from publication).

        Args:
            identifier: GEO accession (GSE, GDS, etc.)
            dataset_types: Types of datasets to search for
            **kwargs: Additional parameters

        Returns:
            str: Formatted dataset information
        """
        logger.info(f"Finding GEO dataset: {identifier}")

        try:
            if not self.validate_identifier(identifier):
                return f"Invalid GEO identifier: {identifier}. Please provide GSE, GDS, GPL, or GSM accession."

            # Search for the specific accession
            query = f"{identifier}[ACCN]"
            filters = GEOSearchFilters(max_results=1)

            search_result = self.search_geo_datasets(query, filters)

            if search_result.count == 0:
                return f"GEO dataset not found: {identifier}"

            # Get detailed summary
            summaries = self.get_dataset_summaries(search_result)

            if not summaries:
                return f"Could not retrieve details for: {identifier}"

            summary = summaries[0]

            # Format detailed response
            response = self._format_detailed_dataset_report(summary, identifier)

            # Log the lookup
            self.data_manager.log_tool_usage(
                tool_name="geo_dataset_lookup",
                parameters={
                    "identifier": identifier,
                    "dataset_types": (
                        [dt.value for dt in dataset_types] if dataset_types else None
                    ),
                },
                description="GEO dataset lookup by accession",
            )

            return response

        except Exception as e:
            logger.error(f"Error finding GEO dataset: {e}")
            return f"Error finding GEO dataset: {str(e)}"

    def extract_publication_metadata(
        self, identifier: str, **kwargs
    ) -> PublicationMetadata:
        """
        Extract metadata from a GEO dataset (returns dataset info as publication-like metadata).

        Args:
            identifier: GEO accession
            **kwargs: Additional parameters

        Returns:
            PublicationMetadata: Dataset metadata formatted as publication metadata
        """
        try:
            if not self.validate_identifier(identifier):
                raise ValueError(f"Invalid GEO identifier: {identifier}")

            # Search for the dataset
            query = f"{identifier}[ACCN]"
            filters = GEOSearchFilters(max_results=1)

            search_result = self.search_geo_datasets(query, filters)

            if search_result.count == 0:
                raise ValueError(f"GEO dataset not found: {identifier}")

            # Get summary
            summaries = self.get_dataset_summaries(search_result)

            if not summaries:
                raise ValueError(f"Could not retrieve summary for: {identifier}")

            summary = summaries[0]

            # Convert to PublicationMetadata format
            return PublicationMetadata(
                uid=identifier,
                title=summary.get("title", f"GEO Dataset {identifier}"),
                journal="Gene Expression Omnibus (GEO)",
                published=summary.get("PDAT", summary.get("date")),
                abstract=summary.get("summary", summary.get("description", "")),
                keywords=[summary.get("GPL", ""), summary.get("taxon", "")],
                pmid=(
                    summary.get("PubMedIds", {}).get("int", "")
                    if "PubMedIds" in summary
                    else None
                ),
            )

        except Exception as e:
            logger.error(f"Error extracting GEO metadata: {e}")
            # Return minimal metadata on error
            return PublicationMetadata(
                uid=identifier,
                title=f"Error extracting metadata: {str(e)}",
                abstract=f"Could not retrieve GEO dataset metadata for: {identifier}",
            )

    # Core GEO-specific methods

    def search_geo_datasets(
        self, query: str, filters: Optional[GEOSearchFilters] = None
    ) -> GEOSearchResult:
        """
        Execute eSearch against GEO database (includes GSE series, GSM samples, GPL platforms, and GDS datasets).

        Args:
            query: Search query string
            filters: Optional search filters

        Returns:
            GEOSearchResult: Search results with IDs and WebEnv
        """
        # Build the complete query using the new query builder
        if filters:
            # Convert filters to dict format for the query builder
            filter_dict = {}
            if filters.organisms:
                filter_dict["organism"] = filters.organisms
            if filters.platforms:
                filter_dict["platform"] = filters.platforms
            if filters.entry_types:
                # Add entry types as special filters
                for et in filters.entry_types:
                    # Use the enum value (e.g., "gse") not the enum object
                    filter_dict[et.value if isinstance(et, GEOEntryType) else et] = True
            if filters.date_range:
                filter_dict["date_range"] = filters.date_range
            if filters.supplementary_file_types:
                filter_dict["supplementary"] = filters.supplementary_file_types

            complete_query = self.query_builder.build_query(query, filter_dict)
            max_results = filters.max_results
            use_history = filters.use_history
        else:
            complete_query = self.query_builder.build_query(query)
            max_results = self.config.max_results
            use_history = True

        # Sanitize query first (fixes unmatched quotes)
        complete_query = self.query_builder.sanitize_query(complete_query)

        # Validate query syntax
        if not self.query_builder.validate_query(complete_query):
            raise ValueError(f"Invalid query syntax: {complete_query}")

        logger.info(f"Executing GEO search: {complete_query}")

        # Build eSearch URL
        url_params = {
            "db": "gds",  # GEO DataSets database (includes GSE, GSM, GPL, GDS)
            "term": complete_query,
            "retmode": "json",
            "retmax": str(max_results),
            "tool": "lobster",
            "email": self.config.email,
        }

        if use_history:
            url_params["usehistory"] = "y"

        if self.config.api_key:
            url_params["api_key"] = self.config.api_key

        url = f"{self.base_url_esearch}?" + urllib.parse.urlencode(url_params)

        # Execute request with retry logic
        response_data = self._execute_request_with_retry(url)

        # Parse eSearch response
        try:
            json_data = json.loads(response_data)
            esearch_result = json_data.get("esearchresult", {})

            # Check for API errors (e.g., invalid database name)
            if "ERROR" in esearch_result:
                error_msg = esearch_result["ERROR"]
                logger.error(f"NCBI API error: {error_msg}")
                raise ValueError(f"NCBI API error: {error_msg}")

            count = int(esearch_result.get("count", "0"))
            ids = esearch_result.get("idlist", [])
            web_env = esearch_result.get("webenv")
            query_key = esearch_result.get("querykey")

            # Cache WebEnv session if available
            if web_env and query_key:
                cache_key = f"{web_env}:{query_key}"
                self._session_cache[cache_key] = (web_env, query_key, datetime.now())

            return GEOSearchResult(
                count=count,
                ids=ids,
                web_env=web_env,
                query_key=query_key,
                query=complete_query,
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing eSearch response: {e}")
            raise ValueError(f"Invalid eSearch response: {e}")

    def get_dataset_summaries(
        self, search_result: GEOSearchResult
    ) -> List[Dict[str, Any]]:
        """
        Retrieve summaries using eSummary with WebEnv.

        Args:
            search_result: Result from search_geo_datasets

        Returns:
            List[Dict[str, Any]]: List of dataset summaries
        """
        if not search_result.ids:
            return []

        # Build eSummary URL
        url_params = {
            "db": "gds",
            "retmode": "json",
            "tool": "lobster",
            "email": self.config.email,
        }

        # Use WebEnv if available for efficiency
        if search_result.web_env and search_result.query_key:
            url_params["webenv"] = search_result.web_env
            url_params["query_key"] = search_result.query_key
        else:
            # Fallback to ID list
            url_params["id"] = ",".join(search_result.ids)

        if self.config.api_key:
            url_params["api_key"] = self.config.api_key

        url = f"{self.base_url_esummary}?" + urllib.parse.urlencode(url_params)

        # Execute request
        response_data = self._execute_request_with_retry(url)

        # Parse eSummary response
        try:
            json_data = json.loads(response_data)
            result = json_data.get("result", {})

            summaries = []
            for uid in search_result.ids:
                if uid in result:
                    summary = result[uid]
                    summary["uid"] = uid
                    summaries.append(summary)

            return summaries

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing eSummary response: {e}")
            return []

    def link_to_pubmed(self, geo_ids: List[str]) -> Dict[str, List[str]]:
        """
        Find PubMed IDs linked to GEO entries using eLink.

        Args:
            geo_ids: List of GEO dataset IDs

        Returns:
            Dict[str, List[str]]: Mapping of GEO ID to PubMed IDs
        """
        if not geo_ids:
            return {}

        linked = {}

        # Build eLink URL
        url_params = {
            "dbfrom": "gds",
            "db": "pubmed",
            "id": ",".join(geo_ids),
            "retmode": "json",
            "tool": "lobster",
            "email": self.config.email,
        }

        if self.config.api_key:
            url_params["api_key"] = self.config.api_key

        url = f"{self.base_url_elink}?" + urllib.parse.urlencode(url_params)

        try:
            response_data = self._execute_request_with_retry(url)
            json_data = json.loads(response_data)

            if "linksets" in json_data:
                for linkset in json_data["linksets"]:
                    geo_id = linkset.get("ids", [None])[0]
                    if geo_id and "linksetdbs" in linkset:
                        pmids = []
                        for db in linkset["linksetdbs"]:
                            if db["dbto"] == "pubmed":
                                pmids.extend(db.get("links", []))
                        if pmids:
                            linked[geo_id] = pmids

        except Exception as e:
            logger.warning(f"Error finding linked PubMed IDs: {e}")

        return linked

    def get_download_urls(self, geo_id: str) -> Dict[str, Any]:
        """
        Extract download URLs from GEO metadata without downloading files.

        Supports GSE (series) and GDS (dataset) accessions. Extracts FTP URLs for:
        - Matrix files (series_matrix.txt.gz)
        - Raw data files (CEL, FASTQ, etc.)
        - Supplementary files (processed data)
        - H5AD files (if available)

        Args:
            geo_id: GEO accession (GSE12345, GDS5678, etc.)

        Returns:
            Dict with structure:
            {
                "geo_id": str,
                "matrix_url": Optional[str],  # Series matrix file
                "raw_urls": List[str],        # Raw data files
                "supplementary_urls": List[str],  # Supplementary files
                "h5_url": Optional[str],      # H5AD file if exists
                "ftp_base": str,              # Base FTP directory
                "file_count": int,            # Total files found
                "total_size_mb": Optional[float],  # Estimated size
            }

        Raises:
            ValueError: If geo_id format invalid
            Exception: If GEO fetch fails
        """
        # Validate GEO ID format
        if not geo_id or not isinstance(geo_id, str):
            raise ValueError(f"Invalid GEO ID: {geo_id}")

        geo_id = geo_id.upper().strip()
        if not (geo_id.startswith("GSE") or geo_id.startswith("GDS")):
            raise ValueError(f"GEO ID must start with GSE or GDS: {geo_id}")

        # Check if GEOparse is available
        if GEOparse is None:
            raise ImportError(
                "GEOparse is required for URL extraction. Install with: pip install GEOparse"
            )

        logger.info(f"Extracting download URLs for {geo_id}")

        try:
            # Construct HTTPS base URL (not FTP - see https fix below)
            ftp_base = self._construct_ftp_base_url(geo_id)  # Note: returns HTTPS despite name

            # PRE-DOWNLOAD SOFT FILE USING HTTPS TO BYPASS GEOparse's FTP DOWNLOADER
            # GEOparse internally uses FTP which lacks error detection and causes corruption.
            # By pre-downloading with HTTPS, GEOparse finds existing file and skips its FTP download.
            soft_file_path = self.cache_dir / f"{geo_id}_family.soft.gz"
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
                    with urllib.request.urlopen(soft_url, context=ssl_context) as response:
                        with open(soft_file_path, "wb") as f:
                            f.write(response.read())
                    logger.debug(f"Successfully pre-downloaded SOFT file to {soft_file_path}")
                except Exception as e:
                    error_str = str(e)
                    if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
                        handle_ssl_error(e, soft_url, logger)
                        raise Exception(
                            f"SSL certificate verification failed when downloading SOFT file. "
                            f"See error message above for solutions."
                        )
                    # If pre-download fails, let GEOparse try (will use FTP as fallback)
                    logger.warning(f"Pre-download failed: {e}. GEOparse will attempt download.")

            # Fetch GEO metadata using GEOparse
            # Note: GEOparse will find our pre-downloaded SOFT file and skip its FTP download
            logger.debug(f"Fetching GEO metadata for {geo_id} using GEOparse")
            gse = GEOparse.get_GEO(
                geo=geo_id,
                destdir=str(self.cache_dir),
                silent=True,
                # Don't use 'brief' as it skips supplementary file metadata
            )

            # Initialize result structure
            matrix_url = None
            raw_urls = []
            supplementary_urls = []
            h5_url = None
            all_urls = []

            # Extract series matrix URL
            if hasattr(gse, "metadata") and gse.metadata:
                series_matrix = gse.metadata.get("series_matrix_file")
                if series_matrix:
                    if isinstance(series_matrix, list):
                        matrix_url = series_matrix[0]
                    else:
                        matrix_url = series_matrix
                    all_urls.append(matrix_url)
                    logger.debug(f"Found matrix URL: {matrix_url}")

            # If no matrix URL from metadata, construct it
            if not matrix_url and geo_id.startswith("GSE"):
                matrix_url = f"{ftp_base}matrix/{geo_id}_series_matrix.txt.gz"
                all_urls.append(matrix_url)
                logger.debug(f"Constructed matrix URL: {matrix_url}")

            # Extract supplementary files from metadata
            supplementary_files = []

            # Check series-level supplementary files
            if hasattr(gse, "metadata") and gse.metadata:
                suppl = gse.metadata.get("supplementary_file", [])
                if isinstance(suppl, str):
                    supplementary_files.append(suppl)
                elif isinstance(suppl, list):
                    supplementary_files.extend(suppl)

            # Check sample-level supplementary files
            if hasattr(gse, "gsms") and gse.gsms:
                for gsm_id, gsm in gse.gsms.items():
                    if hasattr(gsm, "metadata") and gsm.metadata:
                        gsm_suppl = gsm.metadata.get("supplementary_file", [])
                        if isinstance(gsm_suppl, str):
                            supplementary_files.append(gsm_suppl)
                        elif isinstance(gsm_suppl, list):
                            supplementary_files.extend(gsm_suppl)

            # Categorize supplementary files
            for file_url in supplementary_files:
                if not file_url:
                    continue

                file_url = file_url.strip()
                file_lower = file_url.lower()

                # Check for H5AD files
                if file_url.endswith(".h5ad") or file_url.endswith(".h5"):
                    h5_url = file_url
                    all_urls.append(file_url)
                    logger.debug(f"Found H5AD file: {file_url}")

                # Check for raw data files
                elif any(
                    ext in file_lower
                    for ext in [
                        ".cel",
                        ".cel.gz",
                        ".fastq",
                        ".fastq.gz",
                        ".fq",
                        ".fq.gz",
                        ".bam",
                        ".sam",
                        ".sra",
                        ".srf",
                    ]
                ):
                    raw_urls.append(file_url)
                    all_urls.append(file_url)
                    logger.debug(f"Found raw data file: {file_url}")

                # All other supplementary files
                else:
                    supplementary_urls.append(file_url)
                    all_urls.append(file_url)
                    logger.debug(f"Found supplementary file: {file_url}")

            # Calculate file count
            file_count = len(all_urls)

            # Prepare result
            result = {
                "geo_id": geo_id,
                "matrix_url": matrix_url,
                "raw_urls": raw_urls,
                "supplementary_urls": supplementary_urls,
                "h5_url": h5_url,
                "ftp_base": ftp_base,
                "file_count": file_count,
                "total_size_mb": None,  # Size estimation would require FTP SIZE commands
            }

            logger.info(
                f"Extracted {file_count} URLs for {geo_id}: "
                f"matrix={1 if matrix_url else 0}, "
                f"raw={len(raw_urls)}, "
                f"supplementary={len(supplementary_urls)}, "
                f"h5={1 if h5_url else 0}"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to extract URLs for {geo_id}: {e}")
            # Return partial results on error
            return {
                "geo_id": geo_id,
                "matrix_url": None,
                "raw_urls": [],
                "supplementary_urls": [],
                "h5_url": None,
                "ftp_base": (
                    self._construct_ftp_base_url(geo_id)
                    if geo_id.startswith("GSE")
                    else ""
                ),
                "file_count": 0,
                "total_size_mb": None,
                "error": str(e),
            }

    def _construct_ftp_base_url(self, geo_id: str) -> str:
        """
        Construct FTP base URL for GEO series.

        Args:
            geo_id: GEO series ID (e.g., GSE180759)

        Returns:
            Base FTP URL (e.g., ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE180nnn/GSE180759/)
        """
        if not geo_id.startswith("GSE"):
            return ""

        # Extract numeric part
        gse_str = geo_id[3:]  # Remove 'GSE'

        # Determine series folder (replace last 3 digits with 'nnn')
        if len(gse_str) >= 3:
            gse_num_base = gse_str[:-3]  # Remove last 3 digits
            series_folder = f"GSE{gse_num_base}nnn"
        else:
            # For short IDs like GSE1, GSE12
            series_folder = "GSEnnn"

        # Construct base URL using HTTPS for reliable downloads with TLS error detection
        # (FTP lacks error detection and causes silent corruption during NCBI throttling)
        base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series_folder}/{geo_id}/"

        return base_url

    # Helper methods

    def _execute_request_with_retry(self, url: str) -> str:
        """Execute HTTP request with retry logic and SSL support."""
        retry = 0
        sleep_time = self.config.sleep_time

        # Create SSL context once for all retries
        ssl_context = create_ssl_context()

        while retry <= self.config.max_retry:
            try:
                with urllib.request.urlopen(url, context=ssl_context) as response:
                    return response.read().decode("utf-8")

            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < self.config.max_retry:
                    logger.warning(f"Rate limit hit, retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    sleep_time *= 2
                    retry += 1
                else:
                    raise e
            except urllib.error.URLError as e:
                # Check if this is an SSL certificate error
                error_str = str(e.reason)
                if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
                    handle_ssl_error(e, url, logger)
                    raise Exception(
                        f"SSL certificate verification failed. "
                        f"See error message above for solutions. "
                        f"Original error: {error_str}"
                    )
                # For other URLErrors, retry if attempts remain
                if retry < self.config.max_retry:
                    logger.warning(f"Request failed, retrying: {e}")
                    time.sleep(sleep_time)
                    retry += 1
                else:
                    raise e
            except Exception as e:
                if retry < self.config.max_retry:
                    logger.warning(f"Request failed, retrying: {e}")
                    time.sleep(sleep_time)
                    retry += 1
                else:
                    raise e

        raise Exception(f"Request failed after {self.config.max_retry} retries")

    def _convert_filters(self, filters: Dict[str, Any]) -> GEOSearchFilters:
        """Convert generic filters to GEOSearchFilters."""
        geo_filters = GEOSearchFilters()

        if "organisms" in filters:
            geo_filters.organisms = filters["organisms"]

        if "platforms" in filters:
            geo_filters.platforms = filters["platforms"]

        if "entry_types" in filters:
            # Convert strings to GEOEntryType enums
            entry_types = []
            for et in filters["entry_types"]:
                if isinstance(et, str):
                    try:
                        entry_types.append(GEOEntryType(et.lower()))
                    except ValueError:
                        continue
                elif isinstance(et, GEOEntryType):
                    entry_types.append(et)
            geo_filters.entry_types = entry_types

        if "date_range" in filters:
            geo_filters.date_range = filters["date_range"]

        if "supplementary_file_types" in filters:
            geo_filters.supplementary_file_types = filters["supplementary_file_types"]

        if "max_results" in filters:
            geo_filters.max_results = filters["max_results"]

        return geo_filters

    def format_geo_search_results(
        self, search_result: GEOSearchResult, original_query: str
    ) -> str:
        """Format GEO search results for display."""
        response = "## 🧬 GEO Database Search Results\n\n"
        response += f"**Query**: `{search_result.query or original_query}`\n"
        response += f"**Total Results**: {search_result.count:,}\n"
        response += f"**Showing**: {len(search_result.ids)} datasets\n\n"

        if not search_result.summaries:
            # Just show IDs if no summaries available
            response += "### Dataset IDs Found\n"
            for i, dataset_id in enumerate(search_result.ids[:10], 1):
                response += f"{i}. **GDS{dataset_id}** - [View on GEO](https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc=GDS{dataset_id})\n"

            if len(search_result.ids) > 10:
                response += f"\n... and {len(search_result.ids) - 10} more datasets.\n"
        else:
            # Show detailed summaries
            for i, summary in enumerate(search_result.summaries, 1):
                response += f"### Dataset {i}/{len(search_result.summaries)}\n"

                accession = summary.get("accession", f"GDS{summary.get('uid', '')}")
                response += f"**Accession**: [{accession}](https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc={accession})\n"
                response += f"**Title**: {summary.get('title', 'N/A')}\n"

                if summary.get("summary"):
                    desc = summary["summary"][:300]
                    if len(summary["summary"]) > 300:
                        desc += "..."
                    response += f"**Description**: {desc}\n"

                if summary.get("taxon"):
                    response += f"**Organism**: {summary['taxon']}\n"

                if summary.get("GPL"):
                    response += f"**Platform**: {summary['GPL']}\n"

                if summary.get("n_samples"):
                    response += f"**Samples**: {summary['n_samples']}\n"

                if summary.get("PDAT"):
                    response += f"**Date**: {summary['PDAT']}\n"

                response += "\n---\n\n"

        # Add helpful tips
        response += "### 💡 Next Steps\n"
        response += (
            "- Use `download_geo_dataset('GSE123456')` to download specific datasets\n"
        )
        response += "- Refine your search with filters like organisms, platforms, or date ranges\n"
        response += (
            "- Check the GEO website for supplementary files and detailed protocols\n"
        )

        return response

    def _format_detailed_dataset_report(
        self, summary: Dict[str, Any], accession: str
    ) -> str:
        """Format detailed information for a specific GEO dataset."""
        response = f"## 📊 GEO Dataset Details: {accession}\n\n"

        # Basic information
        response += "### Basic Information\n"
        response += f"**Accession**: [{accession}](https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc={accession})\n"
        response += f"**Title**: {summary.get('title', 'N/A')}\n"

        if summary.get("summary"):
            response += f"**Description**: {summary['summary']}\n"

        response += "\n### Dataset Metadata\n"

        if summary.get("taxon"):
            response += f"**Organism**: {summary['taxon']}\n"

        if summary.get("GPL"):
            platform_url = (
                f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={summary['GPL']}"
            )
            response += f"**Platform**: [{summary['GPL']}]({platform_url})\n"

        if summary.get("n_samples"):
            response += f"**Sample Count**: {summary['n_samples']}\n"

        if summary.get("PDAT"):
            response += f"**Publication Date**: {summary['PDAT']}\n"

        if summary.get("entryType"):
            response += f"**Entry Type**: {summary['entryType']}\n"

        # Publication links
        if summary.get("PubMedIds"):
            response += "\n### Related Publications\n"
            pmid = summary["PubMedIds"].get("int", "")
            if pmid:
                pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                response += f"**PubMed**: [{pmid}]({pmid_url})\n"

        # Download and access information
        response += "\n### 💾 Access Options\n"
        response += f"- **GEO Browser**: [View dataset]({f'https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc={accession}'})\n"
        response += (
            f"- **Download**: Use `download_geo_dataset('{accession}')` command\n"
        )
        response += "- **FTP Access**: Check GEO FTP site for raw files\n"

        return response

    def search_by_accession(
        self, accession: str, include_parent_series: bool = True, **kwargs
    ) -> str:
        """
        Search for GEO data using a direct accession (GSE, GSM, GDS, GPL).

        For GSM sample accessions, optionally includes parent series information.

        Args:
            accession: Direct GEO accession (GSE123456, GSM789012, etc.)
            include_parent_series: For GSM samples, also fetch parent GSE info
            **kwargs: Additional parameters

        Returns:
            str: Formatted search results with accession-specific information
        """
        logger.info(f"GEO accession search: {accession}")

        try:
            # Detect accession type
            geo_type = detect_geo_accession_subtype(accession)

            if geo_type is None:
                return f"Invalid GEO accession format: {accession}"

            # Handle GSM samples with parent lookup
            if geo_type == GEOAccessionType.SAMPLE and include_parent_series:
                return self._search_sample_with_parent(accession)

            # Handle other accessions (GSE, GDS, GPL) with direct search
            elif geo_type in [
                GEOAccessionType.SERIES,
                GEOAccessionType.DATASET,
                GEOAccessionType.PLATFORM,
            ]:
                return self._search_direct_accession(accession, geo_type)

            else:
                # Fallback to regular search
                return self.search_publications(accession, max_results=5)

        except Exception as e:
            logger.error(f"Error in accession search for {accession}: {e}")
            return f"Error searching for accession {accession}: {str(e)}"

    def find_parent_series_for_sample(self, gsm_accession: str) -> Optional[str]:
        """
        Find the parent GSE series for a given GSM sample using NCBI E-link.

        Args:
            gsm_accession: GSM sample accession (e.g., GSM6204600)

        Returns:
            GSE series accession if found, None otherwise
        """
        if not is_geo_sample_accession(gsm_accession):
            logger.warning(f"Not a valid GSM accession: {gsm_accession}")
            return None

        try:
            # First, get the GEO UID for this accession using esearch
            search_url = (
                f"{self.base_url_esearch}?db=gds&term={gsm_accession}&retmode=json"
            )
            if self.config.api_key:
                search_url += f"&api_key={self.config.api_key}"

            search_result = self._execute_request_with_retry(search_url)
            search_data = json.loads(search_result)

            if not search_data.get("esearchresult", {}).get("idlist"):
                logger.warning(f"No GEO ID found for {gsm_accession}")
                return None

            geo_uid = search_data["esearchresult"]["idlist"][0]

            # Use E-link to find related GSE series
            link_url = (
                f"{self.base_url_elink}?dbfrom=gds&db=gds&id={geo_uid}&retmode=json"
            )
            if self.config.api_key:
                link_url += f"&api_key={self.config.api_key}"

            link_result = self._execute_request_with_retry(link_url)
            link_data = json.loads(link_result)

            # Extract linked IDs
            linked_ids = []
            if "linksets" in link_data:
                for linkset in link_data["linksets"]:
                    if "linksetdbs" in linkset:
                        for db in linkset["linksetdbs"]:
                            if db.get("dbto") == "gds":
                                linked_ids.extend(db.get("links", []))

            # Get summaries for linked IDs to find GSE series
            if linked_ids:
                ids_str = ",".join(linked_ids[:10])  # Limit to first 10
                summary_url = (
                    f"{self.base_url_esummary}?db=gds&id={ids_str}&retmode=json"
                )
                if self.config.api_key:
                    summary_url += f"&api_key={self.config.api_key}"

                summary_result = self._execute_request_with_retry(summary_url)
                summary_data = json.loads(summary_result)

                # Look for GSE accessions in the results
                for uid in linked_ids:
                    uid_data = summary_data.get("result", {}).get(uid, {})
                    accession = uid_data.get("accession", "")
                    if accession.startswith("GSE"):
                        logger.info(
                            f"Found parent series {accession} for sample {gsm_accession}"
                        )
                        return accession

            logger.warning(f"No parent GSE series found for {gsm_accession}")
            return None

        except Exception as e:
            logger.error(f"Error finding parent series for {gsm_accession}: {e}")
            return None

    def _search_sample_with_parent(self, gsm_accession: str) -> str:
        """
        Search for GSM sample and include parent GSE series information.

        Args:
            gsm_accession: GSM sample accession

        Returns:
            Formatted response with sample and parent series info
        """
        response = "## GEO Sample Search Results\n\n"
        response += f"**Sample Accession**: [{gsm_accession}]({get_ncbi_geo_url(gsm_accession)})\n\n"

        try:
            # Get sample information
            sample_info = self._get_accession_summary(gsm_accession)

            if sample_info:
                response += "### Sample Information\n"
                response += f"**Title**: {sample_info.get('title', 'N/A')}\n"
                if sample_info.get("summary"):
                    summary_preview = sample_info["summary"][:300]
                    if len(sample_info["summary"]) > 300:
                        summary_preview += "..."
                    response += f"**Description**: {summary_preview}\n"

                if sample_info.get("taxon"):
                    response += f"**Organism**: {sample_info['taxon']}\n"

                response += "\n"

            # Find parent series
            parent_gse = self.find_parent_series_for_sample(gsm_accession)

            if parent_gse:
                response += f"### Parent Series: {parent_gse}\n"
                response += f"**Series Accession**: [{parent_gse}]({get_ncbi_geo_url(parent_gse)})\n"

                # Get parent series information
                parent_info = self._get_accession_summary(parent_gse)
                if parent_info:
                    response += f"**Series Title**: {parent_info.get('title', 'N/A')}\n"
                    if parent_info.get("n_samples"):
                        response += (
                            f"**Total Samples in Series**: {parent_info['n_samples']}\n"
                        )

                response += "\n### 💾 Recommended Next Steps\n"
                response += f"- **Download Full Series**: `download_geo_dataset('{parent_gse}')`\n"
                response += f"- **Download Sample Only**: `download_geo_dataset('{gsm_accession}')`\n"
                response += f"- **View Series**: [Browse {parent_gse} on GEO]({get_ncbi_geo_url(parent_gse)})\n"
            else:
                response += "### Parent Series\n"
                response += "⚠️ Could not find parent GSE series for this sample.\n\n"
                response += "### 💾 Next Steps\n"
                response += f"- **Download Sample**: `download_geo_dataset('{gsm_accession}')`\n"
                response += f"- **View Sample**: [Browse {gsm_accession} on GEO]({get_ncbi_geo_url(gsm_accession)})\n"

            return response

        except Exception as e:
            logger.error(f"Error in sample with parent search: {e}")
            return f"Error retrieving information for {gsm_accession}: {str(e)}"

    def _search_direct_accession(
        self, accession: str, geo_type: GEOAccessionType
    ) -> str:
        """
        Search for direct GEO accession (GSE, GDS, GPL).

        Args:
            accession: GEO accession
            geo_type: Type of GEO accession

        Returns:
            Formatted response with accession information
        """
        try:
            # Get accession summary
            summary_info = self._get_accession_summary(accession)

            if not summary_info:
                return f"Could not retrieve information for {accession}"

            response = f"## {geo_type.value.upper()} Accession Search Results\n\n"
            response += f"**Accession**: [{accession}]({get_ncbi_geo_url(accession)})\n"
            response += f"**Title**: {summary_info.get('title', 'N/A')}\n\n"

            if summary_info.get("summary"):
                response += f"**Description**: {summary_info['summary']}\n"

            if summary_info.get("taxon"):
                response += f"**Organism**: {summary_info['taxon']}\n"

            if summary_info.get("n_samples"):
                response += f"**Sample Count**: {summary_info['n_samples']}\n"

            if summary_info.get("PDAT"):
                response += f"**Publication Date**: {summary_info['PDAT']}\n"

            response += "\n### 💾 Next Steps\n"
            response += (
                f"- **Download Dataset**: `download_geo_dataset('{accession}')`\n"
            )
            response += f"- **View on GEO**: [Browse {accession}]({get_ncbi_geo_url(accession)})\n"

            return response

        except Exception as e:
            logger.error(f"Error in direct accession search: {e}")
            return f"Error retrieving information for {accession}: {str(e)}"

    def _get_accession_summary(self, accession: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information for a GEO accession using esummary.

        Args:
            accession: GEO accession (GSE, GSM, GDS, GPL)

        Returns:
            Dictionary with summary information or None if not found
        """
        try:
            # First, get the UID for this accession
            search_url = f"{self.base_url_esearch}?db=gds&term={accession}&retmode=json"
            if self.config.api_key:
                search_url += f"&api_key={self.config.api_key}"

            search_result = self._execute_request_with_retry(search_url)
            search_data = json.loads(search_result)

            if not search_data.get("esearchresult", {}).get("idlist"):
                return None

            geo_uid = search_data["esearchresult"]["idlist"][0]

            # Get summary using esummary
            summary_url = f"{self.base_url_esummary}?db=gds&id={geo_uid}&retmode=json"
            if self.config.api_key:
                summary_url += f"&api_key={self.config.api_key}"

            summary_result = self._execute_request_with_retry(summary_url)
            summary_data = json.loads(summary_result)

            return summary_data.get("result", {}).get(geo_uid, {})

        except Exception as e:
            logger.error(f"Error getting summary for {accession}: {e}")
            return None
