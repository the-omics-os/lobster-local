"""
PubMed provider implementation for publication search and dataset discovery.

This provider implements the BasePublicationProvider interface for PubMed/NCBI
and provides comprehensive literature search and omics dataset discovery capabilities.
"""

import json
import random
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.base_provider import (
    BasePublicationProvider,
    DatasetType,
    PublicationMetadata,
    PublicationSource,
)
from lobster.tools.rate_limiter import NCBIRateLimiter
from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context, handle_ssl_error

logger = get_logger(__name__)


class PubMedProviderConfig(BaseModel):
    """Configuration for PubMed provider."""

    # PubMed settings
    top_k_results: int = Field(default=5, ge=1, le=100)
    max_query_length: int = Field(default=500, ge=100, le=1000)
    doc_content_chars_max: int = Field(default=6000, ge=1000, le=20000)

    # NCBI API settings
    email: str = Field(default_factory=lambda: get_settings().NCBI_EMAIL)
    api_key: Optional[str] = None

    # Rate limiting and retry settings
    max_retry: int = Field(default=5, ge=1, le=10)
    base_sleep_time: float = Field(
        default=0.34, ge=0.1, le=2.0
    )  # Base delay between requests
    max_backoff_delay: int = Field(
        default=30, ge=5, le=120
    )  # Max backoff delay (seconds)
    requests_per_second_no_key: float = Field(
        default=3.0, ge=1.0, le=5.0
    )  # NCBI limit without key
    requests_per_second_with_key: float = Field(
        default=10.0, ge=5.0, le=15.0
    )  # NCBI limit with key
    circuit_breaker_threshold: int = Field(
        default=10, ge=3, le=20
    )  # Consecutive failures before circuit break

    # Dataset discovery settings
    include_geo_datasets: bool = True
    include_sra_datasets: bool = True
    include_supplementary: bool = True

    # Method extraction settings
    extract_parameters: bool = True
    extract_github_links: bool = True
    extract_software_tools: bool = True


class PubMedProvider(BasePublicationProvider):
    """
    PubMed provider for literature search and NCBI dataset discovery.

    This provider implements comprehensive PubMed/NCBI functionality including:
    - Advanced literature search with parameter extraction
    - Direct NCBI Entrez access for dataset discovery
    - Support for GEO, SRA, and other omics databases
    - Computational methodology extraction
    - GitHub repository and software tool identification
    """

    def __init__(
        self, data_manager: DataManagerV2, config: Optional[PubMedProviderConfig] = None
    ):
        """
        Initialize PubMed provider.

        Args:
            data_manager: DataManagerV2 instance for provenance tracking
            config: Optional configuration, uses defaults if not provided
        """
        self.data_manager = data_manager
        settings = get_settings()

        # Create config with API key from settings if not provided
        if config is None:
            self.config = PubMedProviderConfig(api_key=settings.NCBI_API_KEY)
        else:
            # If config provided but no API key, update it with settings
            if config.api_key is None:
                self.config = config.model_copy(
                    update={"api_key": settings.NCBI_API_KEY}
                )
            else:
                self.config = config

        # Initialize XML parser
        try:
            import xmltodict

            self.parse = xmltodict.parse
        except ImportError:
            raise ImportError(
                "Could not import xmltodict. Install with: pip install xmltodict"
            )

        # NCBI E-utilities base URLs
        self.base_url_esearch = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        )
        self.base_url_efetch = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        )
        self.base_url_esummary = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
        )
        self.base_url_elink = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?"
        )

        # Request tracking and rate limiting
        self._last_request_time = 0.0
        self._consecutive_failures = 0
        self._circuit_breaker_until = 0.0
        self._api_key_notified = False

        # Redis-based distributed rate limiter (with graceful degradation)
        self._rate_limiter = NCBIRateLimiter()

    @property
    def source(self) -> PublicationSource:
        """Return PubMed as the publication source."""
        return PublicationSource.PUBMED

    @property
    def supported_dataset_types(self) -> List[DatasetType]:
        """Return list of dataset types supported by NCBI."""
        return [DatasetType.BIOPROJECT, DatasetType.BIOSAMPLE, DatasetType.DBGAP]

    @property
    def priority(self) -> int:
        """
        Return provider priority for capability-based routing.

        PubMed has high priority (10) due to fast NCBI API access and
        authoritative metadata. Ideal for literature search and finding
        linked datasets via ELink.

        Returns:
            int: Priority 10 (high priority)
        """
        return 10

    def get_supported_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities supported by PubMed provider.

        PubMed excels at literature search, metadata extraction, and linking
        publications to datasets via NCBI ELink API. It provides fast abstract
        retrieval but does not offer full-text access (PMC handles that).

        Supported capabilities:
        - SEARCH_LITERATURE: ESearch API for PubMed/bioRxiv/medRxiv
        - FIND_LINKED_DATASETS: ELink API to connect PMID → GEO/SRA/BioProject
        - EXTRACT_METADATA: ESummary API for structured publication metadata
        - QUERY_CAPABILITIES: Dynamic capability discovery
        - GET_ABSTRACT: EFetch API for fast abstract retrieval (<500ms)

        Not supported:
        - DISCOVER_DATASETS: Links to datasets but doesn't search GEO directly
        - GET_FULL_CONTENT: No full-text access (use PMCProvider)
        - EXTRACT_METHODS: No methods extraction (use PMCProvider)
        - EXTRACT_PDF: No PDF processing
        - VALIDATE_METADATA: No dataset validation
        - INTEGRATE_MULTI_OMICS: No multi-omics integration

        Returns:
            Dict[str, bool]: Capability support mapping
        """
        from lobster.tools.providers.base_provider import ProviderCapability

        return {
            ProviderCapability.SEARCH_LITERATURE: True,
            ProviderCapability.DISCOVER_DATASETS: False,
            ProviderCapability.FIND_LINKED_DATASETS: True,
            ProviderCapability.EXTRACT_METADATA: True,
            ProviderCapability.VALIDATE_METADATA: False,
            ProviderCapability.QUERY_CAPABILITIES: True,
            ProviderCapability.GET_ABSTRACT: True,
            ProviderCapability.GET_FULL_CONTENT: False,
            ProviderCapability.EXTRACT_METHODS: False,
            ProviderCapability.EXTRACT_PDF: False,
            ProviderCapability.INTEGRATE_MULTI_OMICS: False,
        }

    def validate_identifier(self, identifier: str) -> bool:
        """
        Validate PubMed/DOI identifiers.

        Args:
            identifier: Identifier to validate (PMID or DOI)

        Returns:
            bool: True if identifier is valid
        """
        identifier = identifier.strip()

        # Check for DOI format
        if identifier.startswith("10."):
            return True

        # Check for PMID format (numeric)
        if identifier.isdigit():
            return True

        # Check for PMID with prefix
        if identifier.upper().startswith("PMID:"):
            return identifier[5:].strip().isdigit()

        return False

    def get_supported_features(self) -> Dict[str, bool]:
        """Return features supported by PubMed provider."""
        return {
            "literature_search": True,
            "dataset_discovery": True,
            "metadata_extraction": True,
            "full_text_access": True,  # NOW SUPPORTED via PMC XML API
            "advanced_filtering": True,
            "computational_methods": True,
            "github_extraction": True,
        }

    def search_publications(
        self,
        query: str,
        max_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Search PubMed for publications.

        Args:
            query: Search query string
            max_results: Maximum number of results (default: 5)
            filters: Optional filters (date_range, journal, etc.)
            **kwargs: Additional parameters

        Returns:
            str: Formatted search results
        """
        logger.debug(f"PubMed search: {query[:50]}...")

        try:
            # Apply configuration limits
            k_results = min(max_results, self.config.top_k_results)
            truncated_query = query[: self.config.max_query_length]

            # Apply filters if provided
            if filters:
                truncated_query = self._apply_search_filters(truncated_query, filters)

            # Load results using NCBI API
            results = list(self._load_with_params(truncated_query, k_results))

            if not results:
                return "No PubMed results found for your query."

            # Convert to PublicationMetadata objects
            pub_metadata = []
            for result in results:
                metadata = self._convert_to_metadata(result)
                pub_metadata.append(metadata)

            # Use base class formatting
            formatted_results = self.format_search_results(pub_metadata, query)

            # Add dataset discovery hints
            formatted_results += "\n\n💡 **Tip**: Use `find_datasets_from_publication()` with any PMID/DOI to discover associated datasets.\n"

            # Log the search
            self.data_manager.log_tool_usage(
                tool_name="pubmed_search",
                parameters={
                    "query": query[:100],
                    "max_results": k_results,
                    "filters": filters,
                },
                description="PubMed literature search",
            )

            return formatted_results

        except Exception as e:
            logger.exception(f"PubMed search error: {e}")
            return f"PubMed search error: {str(e)}"

    def find_datasets_from_publication(
        self,
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        **kwargs,
    ) -> str:
        """
        Find datasets associated with a PubMed publication.

        Args:
            identifier: DOI or PMID of the publication
            dataset_types: Types of datasets to search for
            **kwargs: Additional parameters (include_related, etc.)

        Returns:
            str: Formatted list of discovered datasets
        """
        logger.debug(f"Finding datasets from publication: {identifier}")
        logger.debug(
            f"[DEFENSIVE] dataset_types parameter: type={type(dataset_types)}, value={dataset_types}"
        )

        try:
            include_related = kwargs.get("include_related", True)

            # Determine identifier type and build query
            if identifier.startswith("10."):
                query = f"{identifier}[DOI]"
                is_doi = True
            elif identifier.isdigit() or identifier.upper().startswith("PMID:"):
                pmid = identifier.replace("PMID:", "").strip()
                query = f"{pmid}[PMID]"
                is_doi = False
            else:
                return f"Invalid identifier format: {identifier}. Please provide DOI or PMID."

            # Get the main publication
            results = list(self._load_with_params(query, 1))

            if not results:
                return f"Publication not found: {identifier}"

            article = results[0]
            pmid = article.get("uid")

            # Extract datasets using comprehensive approach
            datasets = self._extract_dataset_accessions(article.get("Summary", ""))
            logger.debug(
                f"[DEFENSIVE] After _extract_dataset_accessions: type={type(datasets)}, keys={list(datasets.keys()) if isinstance(datasets, dict) else 'NOT_DICT'}, datasets={datasets}"
            )

            # Use NCBI E-link to find linked datasets
            if pmid and include_related:
                linked_datasets = self._find_linked_datasets(pmid)
                logger.debug(
                    f"[DEFENSIVE] After _find_linked_datasets: type={type(linked_datasets)}, keys={list(linked_datasets.keys()) if isinstance(linked_datasets, dict) else 'NOT_DICT'}, linked_datasets={linked_datasets}"
                )
                datasets.update(linked_datasets)
                logger.debug(
                    f"[DEFENSIVE] After merge: type={type(datasets)}, keys={list(datasets.keys()) if isinstance(datasets, dict) else 'NOT_DICT'}, datasets={datasets}"
                )

            # Check for supplementary materials if enabled
            if self.config.include_supplementary:
                supp_datasets = self._check_supplementary_materials(article)
                datasets.update(supp_datasets)

            # Filter by requested dataset types if specified
            if dataset_types:
                logger.debug(
                    f"[DEFENSIVE] Filtering datasets. Input dataset_types: {dataset_types}"
                )
                filtered_datasets = {}
                type_map = {
                    DatasetType.GEO: "GEO",
                    DatasetType.SRA: "SRA",
                    DatasetType.ARRAYEXPRESS: "ArrayExpress",
                    DatasetType.ENA: "ENA",
                }
                for dtype in dataset_types:
                    logger.debug(
                        f"[DEFENSIVE] Processing dtype: type={type(dtype)}, value={dtype}, in_type_map={dtype in type_map}"
                    )
                    if dtype in type_map and type_map[dtype] in datasets:
                        filtered_datasets[type_map[dtype]] = datasets[type_map[dtype]]
                datasets = filtered_datasets
                logger.debug(
                    f"[DEFENSIVE] After filtering: type={type(datasets)}, keys={list(datasets.keys()) if isinstance(datasets, dict) else 'NOT_DICT'}, datasets={datasets}"
                )

            # Format comprehensive response
            logger.debug(
                f"[DEFENSIVE] Before _format_comprehensive_dataset_report: type={type(datasets)}, structure={datasets}"
            )
            response = self._format_comprehensive_dataset_report(
                article, datasets, identifier if is_doi else None, pmid
            )

            # Log the search
            self.data_manager.log_tool_usage(
                tool_name="find_datasets_from_publication",
                parameters={
                    "identifier": identifier,
                    "dataset_types": (
                        [dt.value for dt in dataset_types] if dataset_types else None
                    ),
                    "include_related": include_related,
                },
                description="Dataset discovery from publication",
            )

            return response

        except Exception as e:
            logger.error(f"Error finding datasets: {e}")
            logger.exception("[DEFENSIVE] Full traceback for slice bug investigation:")
            return f"Error finding datasets from publication: {str(e)}"

    def extract_publication_metadata(
        self, identifier: str, **kwargs
    ) -> PublicationMetadata:
        """
        Extract standardized metadata from a PubMed publication.

        Args:
            identifier: DOI or PMID of the publication
            **kwargs: Additional parameters

        Returns:
            PublicationMetadata: Standardized publication metadata
        """
        try:
            # Build query based on identifier type
            if identifier.startswith("10."):
                query = f"{identifier}[DOI]"
            elif identifier.isdigit() or identifier.upper().startswith("PMID:"):
                pmid = identifier.replace("PMID:", "").strip()
                query = f"{pmid}[PMID]"
            else:
                raise ValueError(f"Invalid identifier: {identifier}")

            # Get publication data
            results = list(self._load_with_params(query, 1))

            if not results:
                raise ValueError(f"Publication not found: {identifier}")

            article = results[0]
            return self._convert_to_metadata(article)

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            # Return minimal metadata on error
            return PublicationMetadata(
                uid=identifier,
                title=f"Error extracting metadata: {str(e)}",
                abstract=f"Could not retrieve publication metadata for: {identifier}",
            )

    def build_ncbi_url(self, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Build NCBI URL with proper parameter encoding and API key.

        Args:
            endpoint: The NCBI endpoint (esearch, efetch, elink, esummary)
            params: Dictionary of parameters

        Returns:
            str: Properly formatted URL
        """
        base = getattr(self, f"base_url_{endpoint}")
        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}
        query = urllib.parse.urlencode(params)

        if self.config.api_key:
            return f"{base}{query}&api_key={self.config.api_key}"
        return f"{base}{query}"

    def _make_ncbi_request(
        self, url: str, operation_name: str = "NCBI request"
    ) -> bytes:
        """
        Centralized NCBI request handler with comprehensive error handling and rate limiting.

        Args:
            url: The NCBI API URL to request
            operation_name: Human-readable name for the operation (for logging/errors)

        Returns:
            bytes: Response content

        Raises:
            Exception: For permanent failures or after exhausting retries
        """
        # Check circuit breaker
        current_time = time.time()
        if current_time < self._circuit_breaker_until:
            remaining_time = int(self._circuit_breaker_until - current_time)
            raise Exception(
                f"NCBI requests temporarily disabled due to repeated failures. "
                f"Try again in {remaining_time} seconds. "
                f"Consider using an NCBI API key for better reliability."
            )

        # Apply request throttling
        self._apply_request_throttling()

        # Create SSL context for secure connections
        ssl_context = create_ssl_context()

        attempt = 0
        last_exception = None

        while attempt < self.config.max_retry:
            try:
                # Add email to all requests (NCBI requirement)
                separator = "&" if "?" in url and "=" in url else "?"
                url_with_email = (
                    f"{url}{separator}email={urllib.parse.quote(self.config.email)}"
                )

                logger.debug(
                    f"NCBI {operation_name} attempt {attempt + 1}/{self.config.max_retry}: {url_with_email[:100]}..."
                )

                # Make the request with SSL context
                response = urllib.request.urlopen(
                    url_with_email, context=ssl_context, timeout=30
                )
                content = response.read()

                # Reset failure counter on success
                self._consecutive_failures = 0
                logger.debug(f"NCBI {operation_name} successful")

                return content

            except urllib.error.HTTPError as e:
                last_exception = e
                attempt += 1

                if e.code == 429:  # Too Many Requests
                    logger.warning(
                        f"NCBI rate limit hit for {operation_name} (attempt {attempt})"
                    )
                    self._handle_rate_limit_error(attempt, operation_name)

                elif e.code in [500, 502, 503, 504]:  # Server errors
                    logger.warning(
                        f"NCBI server error {e.code} for {operation_name} (attempt {attempt})"
                    )
                    self._handle_server_error(e.code, attempt, operation_name)

                elif e.code in [400, 404]:  # Client errors (don't retry)
                    logger.error(
                        f"NCBI client error {e.code} for {operation_name}: {str(e)}"
                    )
                    self._consecutive_failures += 1
                    raise Exception(f"NCBI request failed: {str(e)}")

                else:  # Other HTTP errors
                    logger.warning(
                        f"NCBI HTTP error {e.code} for {operation_name} (attempt {attempt}): {str(e)}"
                    )
                    if attempt >= self.config.max_retry:
                        break
                    self._apply_backoff_delay(attempt)

            except (socket.timeout, socket.error, urllib.error.URLError) as e:
                last_exception = e
                attempt += 1

                # Check for SSL certificate errors
                error_str = str(e)
                if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
                    handle_ssl_error(e, url_with_email, logger)
                    self._consecutive_failures += 1
                    raise Exception(
                        f"SSL certificate verification failed for NCBI {operation_name}. "
                        f"See error message above for solutions. "
                        f"Original error: {error_str}"
                    )

                logger.warning(
                    f"NCBI network error for {operation_name} (attempt {attempt}): {str(e)}"
                )

                if attempt < self.config.max_retry:
                    self._apply_backoff_delay(attempt)

            except Exception as e:
                last_exception = e
                logger.error(f"NCBI unexpected error for {operation_name}: {str(e)}")
                self._consecutive_failures += 1
                raise

        # All retries exhausted
        self._consecutive_failures += 1
        self._maybe_activate_circuit_breaker()

        error_msg = (
            f"NCBI {operation_name} failed after {self.config.max_retry} attempts"
        )
        if last_exception:
            error_msg += f": {str(last_exception)}"

        # Add API key suggestion for rate limiting issues
        if (
            isinstance(last_exception, urllib.error.HTTPError)
            and last_exception.code == 429
        ):
            error_msg += self._get_api_key_suggestion()

        raise Exception(error_msg)

    def _apply_request_throttling(self) -> None:
        """Apply rate limiting between requests using Redis + in-memory fallback."""
        # Layer 1: Redis-based distributed rate limiting
        # This ensures rate limits work across multiple processes/users
        if not self._rate_limiter.wait_for_slot("ncbi_esearch", max_wait=10.0):
            logger.warning(
                "Redis rate limiter timeout - falling back to in-memory throttling"
            )

        # Layer 2: In-memory rate limiting (backward compatibility + fallback)
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        # Determine appropriate delay based on API key presence
        if self.config.api_key:
            min_delay = 1.0 / self.config.requests_per_second_with_key
        else:
            min_delay = 1.0 / self.config.requests_per_second_no_key

        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def _handle_rate_limit_error(self, attempt: int, operation_name: str) -> None:
        """Handle rate limiting errors with progressive backoff."""
        if not self._api_key_notified:
            self._notify_about_api_key()

        if attempt < self.config.max_retry:
            # Progressive backoff with jitter for rate limits
            base_delay = self.config.base_sleep_time * (2**attempt)
            jitter = random.uniform(0, base_delay * 0.1)
            delay = min(base_delay + jitter, self.config.max_backoff_delay)

            logger.debug(f"Rate limited by NCBI, waiting {delay:.1f}s before retry...")
            time.sleep(delay)

    def _handle_server_error(
        self, status_code: int, attempt: int, operation_name: str
    ) -> None:
        """Handle server errors with appropriate backoff."""
        if attempt < self.config.max_retry:
            # Shorter delays for server errors (they often resolve quickly)
            delay = min(2**attempt, 10)
            logger.debug(
                f"NCBI server error {status_code}, waiting {delay}s before retry..."
            )
            time.sleep(delay)

    def _apply_backoff_delay(self, attempt: int) -> None:
        """Apply exponential backoff delay."""
        delay = min(
            self.config.base_sleep_time * (2**attempt), self.config.max_backoff_delay
        )
        jitter = random.uniform(0, delay * 0.1)
        total_delay = delay + jitter
        logger.debug(f"Applying backoff delay: {total_delay:.2f}s")
        time.sleep(total_delay)

    def _maybe_activate_circuit_breaker(self) -> None:
        """Activate circuit breaker if too many consecutive failures."""
        if self._consecutive_failures >= self.config.circuit_breaker_threshold:
            self._circuit_breaker_until = time.time() + 300  # 5 minutes
            logger.warning(
                f"NCBI circuit breaker activated after {self._consecutive_failures} failures. "
                f"Requests disabled for 5 minutes."
            )

    def _notify_about_api_key(self) -> None:
        """Notify user about API key benefits (once per session)."""
        if not self._api_key_notified:
            if not self.config.api_key:
                logger.warning(
                    "🔑 NCBI API Key Recommended: You're hitting rate limits. "
                    "Get a free API key at https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/ "
                    "to increase your rate limit from 3 to 10 requests/second."
                )
            self._api_key_notified = True

    def _get_api_key_suggestion(self) -> str:
        """Get API key suggestion message."""
        if not self.config.api_key:
            return (
                "\n\n💡 To avoid rate limits, get a free NCBI API key at: "
                "https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/ "
                "and set NCBI_API_KEY environment variable."
            )
        return "\n\nYour API key is active, but NCBI servers may be overloaded. Try again later."

    def validate_bioproject_publication_link(
        self, bioproject_id: str, expected_pmid: str
    ) -> Dict[str, Any]:
        """
        Validate that a BioProject is linked to the expected publication.

        Uses NCBI E-Link to check if bioproject → pubmed link exists.
        This enables provenance validation: identifiers that link to the source
        publication are "primary", others are "referenced" from other studies.

        Args:
            bioproject_id: BioProject accession (e.g., "PRJNA436359", "PRJEB12345")
            expected_pmid: PMID of source publication (e.g., "35042229")

        Returns:
            Dict with validation results:
            - is_linked: bool - True if bioproject links to expected_pmid
            - linked_pmids: list - All PMIDs linked to this bioproject
            - provenance: str - "primary", "referenced", or "uncertain"
            - confidence: float - 0.0-1.0 confidence score
            - error: Optional[str] - Error message if validation failed

        Example:
            >>> provider = PubMedProvider()
            >>> result = provider.validate_bioproject_publication_link(
            ...     "PRJNA436359", "35042229"
            ... )
            >>> print(result)
            {'is_linked': False, 'linked_pmids': ['29123456'],
             'provenance': 'referenced', 'confidence': 0.2}
        """
        import json
        import re

        # Extract numeric ID from PRJNA/PRJEB/PRJDB format
        match = re.match(r"PRJ[A-Z]{1,2}(\d+)", bioproject_id.upper())
        if not match:
            return {
                "is_linked": False,
                "linked_pmids": [],
                "provenance": "uncertain",
                "confidence": 0.0,
                "error": f"Invalid BioProject format: {bioproject_id}",
            }

        # E-Link: bioproject → pubmed
        # Note: E-Link uses BioProject UID, not accession. We need to search for it first.
        try:
            # Step 1: Get BioProject UID from accession
            search_url = self.build_ncbi_url(
                "esearch",
                {
                    "db": "bioproject",
                    "term": bioproject_id,
                    "retmode": "json",
                },
            )

            search_content = self._make_ncbi_request(
                search_url, f"search bioproject {bioproject_id}"
            )
            search_response = json.loads(search_content.decode("utf-8"))

            id_list = search_response.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                logger.debug(f"BioProject not found in NCBI: {bioproject_id}")
                return {
                    "is_linked": False,
                    "linked_pmids": [],
                    "provenance": "uncertain",
                    "confidence": 0.3,
                    "error": f"BioProject not found: {bioproject_id}",
                }

            bioproject_uid = id_list[0]

            # Step 2: E-Link from bioproject to pubmed
            elink_url = self.build_ncbi_url(
                "elink",
                {
                    "dbfrom": "bioproject",
                    "db": "pubmed",
                    "id": bioproject_uid,
                    "retmode": "json",
                },
            )

            elink_content = self._make_ncbi_request(
                elink_url, f"elink bioproject→pubmed {bioproject_id}"
            )
            elink_response = json.loads(elink_content.decode("utf-8"))

            linked_pmids = []
            if "linksets" in elink_response:
                for linkset in elink_response["linksets"]:
                    if "linksetdbs" in linkset:
                        for db in linkset["linksetdbs"]:
                            if db.get("dbto") == "pubmed":
                                linked_pmids.extend(str(x) for x in db.get("links", []))

            # Normalize expected_pmid (remove PMID: prefix if present)
            expected_pmid_normalized = str(expected_pmid).strip()
            if expected_pmid_normalized.upper().startswith("PMID:"):
                expected_pmid_normalized = expected_pmid_normalized[5:].strip()

            is_linked = expected_pmid_normalized in linked_pmids

            if is_linked:
                provenance = "primary"
                confidence = 1.0
            elif linked_pmids:
                provenance = "referenced"
                confidence = 0.2
            else:
                provenance = "uncertain"
                confidence = 0.5

            logger.debug(
                f"E-Link validation for {bioproject_id} → PMID:{expected_pmid}: "
                f"is_linked={is_linked}, linked_pmids={linked_pmids}, provenance={provenance}"
            )

            return {
                "is_linked": is_linked,
                "linked_pmids": linked_pmids,
                "provenance": provenance,
                "confidence": confidence,
            }

        except Exception as e:
            logger.warning(f"E-Link validation failed for {bioproject_id}: {e}")
            return {
                "is_linked": False,
                "linked_pmids": [],
                "provenance": "uncertain",
                "confidence": 0.3,
                "error": str(e),
            }

    def extract_computational_methods(
        self, identifier: str, method_type: str = "all", include_parameters: bool = True
    ) -> str:
        """
        Extract computational methods and parameters from a publication.

        This method prioritizes PMC full text extraction (95% accuracy, structured XML)
        over abstract-only extraction (70% accuracy). Falls back to abstract if PMC
        full text is not available.

        Args:
            identifier: DOI or PMID of the publication
            method_type: Type of methods to extract
            include_parameters: Whether to extract parameter values

        Returns:
            str: Extracted methods with parameters and citations
        """
        logger.debug(f"Extracting computational methods from: {identifier}")

        try:
            # PRIORITY: Try PMC full text first (95% accuracy, structured XML)
            pmc_full_text = self._get_pmc_full_text(identifier)

            if pmc_full_text:
                logger.debug(f"Using PMC full text for method extraction: {identifier}")
                # Extract methods from full text (methods section + full text)
                methods_text = (
                    pmc_full_text.methods_section
                    if pmc_full_text.methods_section
                    else pmc_full_text.full_text
                )
                methods = self._extract_methods_from_text(
                    methods_text, method_type, include_parameters
                )

                # Add PMC-specific extractions
                if pmc_full_text.software_tools:
                    methods["tools"] = [
                        {"text": tool, "value": None}
                        for tool in pmc_full_text.software_tools
                    ]

                if pmc_full_text.github_repos:
                    methods.setdefault("github_repos", [])
                    methods["github_repos"] = pmc_full_text.github_repos

                metadata_source = "PMC full text"
            else:
                # Fallback: Use abstract (70% accuracy)
                logger.debug(
                    f"PMC full text not available, using abstract for: {identifier}"
                )
                metadata = self.extract_publication_metadata(identifier)

                if not metadata.abstract:
                    return f"No abstract or PMC full text available for method extraction: {identifier}"

                # Extract methods using pattern matching from abstract
                methods = self._extract_methods_from_text(
                    metadata.abstract, method_type, include_parameters
                )
                methods_text = metadata.abstract
                metadata_source = "abstract"

            # Get metadata for response formatting
            if not pmc_full_text:
                # Already have metadata from extract_publication_metadata above
                pass
            else:
                # Build minimal metadata from PMC result
                metadata = self.extract_publication_metadata(identifier)

            # Format response
            response = f"## Computational Methods Extracted from {identifier}\n\n"
            response += f"**Title**: {metadata.title}\n"
            response += f"**PMID**: {metadata.pmid or 'N/A'}\n"
            response += f"**DOI**: {metadata.doi or 'N/A'}\n"
            response += f"**Source**: {metadata_source}\n\n"

            # Add extracted methods
            for mtype, mlist in methods.items():
                if mlist and mtype not in ["parameters", "tools"]:
                    response += f"### {mtype.replace('_', ' ').title()}\n"
                    for method in mlist:
                        response += f"- {method['text']}\n"
                        if method.get("value"):
                            response += f"  * Extracted value: {method['value']}\n"
                    response += "\n"

            # Add tools
            if methods.get("tools"):
                response += "### Software Tools\n"
                unique_tools = list(set(m["text"] for m in methods["tools"]))
                for tool in unique_tools:
                    response += f"- {tool}\n"
                response += "\n"

            # Add GitHub repositories
            github_repos = self._extract_github_repos(metadata.abstract)
            if github_repos:
                response += "### GitHub Repositories\n"
                for repo in github_repos:
                    response += f"- {repo}\n"
                response += "\n"

            # Add parameter summary
            if methods.get("parameters") and include_parameters:
                response += "### Extracted Parameters\n"
                for param, value in methods["parameters"].items():
                    response += f"- {param}: {value}\n"
                response += "\n"

            return response

        except Exception as e:
            logger.error(f"Error extracting methods: {e}")
            return f"Error extracting computational methods: {str(e)}"

    # Helper methods (refactored from original PubMedService)

    def _apply_search_filters(self, query: str, filters: Dict[str, Any]) -> str:
        """Apply search filters to PubMed query."""
        # Wrap the main query in parentheses for safety
        filtered_query = f"({query})"

        if "date_range" in filters:
            date_range = filters["date_range"]
            if isinstance(date_range, dict):
                start = date_range.get("start")
                end = date_range.get("end")
                if start and end:
                    # Use PubMed date range syntax
                    filtered_query += f" AND ({start}:{end}[PDAT])"
                elif start:
                    filtered_query += f" AND ({start}[PDAT])"
                elif end:
                    filtered_query += f" AND ({end}[PDAT])"

        if "journal" in filters:
            filtered_query += f" AND ({filters['journal']}[JOUR])"

        if "author" in filters:
            filtered_query += f" AND ({filters['author']}[AUTH])"

        if "publication_type" in filters:
            filtered_query += f" AND ({filters['publication_type']}[PTYP])"

        return filtered_query

    def _load_with_params(self, query: str, top_k_results: int) -> Iterator[dict]:
        """Load PubMed results with specific parameters."""
        # Use URL builder for search
        url = self.build_ncbi_url(
            "esearch",
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": top_k_results,
                "usehistory": "y",
            },
        )

        # Use centralized request handler
        content = self._make_ncbi_request(url, "PubMed search")
        text = content.decode("utf-8")
        json_text = json.loads(text)

        if int(json_text["esearchresult"].get("count", "0")) == 0:
            return

        webenv = json_text["esearchresult"]["webenv"]
        idlist = json_text["esearchresult"]["idlist"]

        # Batch fetch articles for efficiency
        if idlist:
            yield from self._batch_fetch_articles(idlist, webenv)

    def _batch_fetch_articles(self, idlist: List[str], webenv: str) -> Iterator[dict]:
        """
        Batch fetch multiple articles in a single request.

        Args:
            idlist: List of PMIDs to fetch
            webenv: Web environment for the search

        Yields:
            dict: Parsed article data
        """
        # Fetch all articles in one request
        ids = ",".join(idlist)
        url = self.build_ncbi_url(
            "efetch", {"db": "pubmed", "retmode": "xml", "id": ids, "webenv": webenv}
        )

        # Use centralized request handler
        xml_content = self._make_ncbi_request(
            url, f"batch fetch {len(idlist)} articles"
        )
        xml_text = xml_content.decode("utf-8")
        text_dict = self.parse(xml_text)

        # Handle multiple articles in response
        articles = []
        if "PubmedArticleSet" in text_dict:
            article_set = text_dict["PubmedArticleSet"]

            # Check if it's a single article or multiple
            if "PubmedArticle" in article_set:
                if isinstance(article_set["PubmedArticle"], list):
                    articles = article_set["PubmedArticle"]
                else:
                    articles = [article_set["PubmedArticle"]]
            elif "PubmedBookArticle" in article_set:
                if isinstance(article_set["PubmedBookArticle"], list):
                    articles = article_set["PubmedBookArticle"]
                else:
                    articles = [article_set["PubmedBookArticle"]]

        # Process each article
        for idx, article_data in enumerate(articles):
            uid = idlist[idx] if idx < len(idlist) else f"unknown_{idx}"
            yield self._parse_article_from_data(uid, article_data)

    def retrieve_article(self, uid: str, webenv: str) -> dict:
        """Retrieve article metadata."""
        url = self.build_ncbi_url(
            "efetch", {"db": "pubmed", "retmode": "xml", "id": uid, "webenv": webenv}
        )

        # Use centralized request handler
        xml_content = self._make_ncbi_request(url, f"retrieve article {uid}")
        xml_text = xml_content.decode("utf-8")
        text_dict = self.parse(xml_text)
        return self._parse_article(uid, text_dict)

    def _parse_article(self, uid: str, text_dict: dict) -> dict:
        """Parse article metadata from XML dictionary."""
        # Extract the article from the full response
        article_data = None

        if "PubmedArticleSet" in text_dict:
            article_set = text_dict["PubmedArticleSet"]
            if "PubmedArticle" in article_set:
                article_data = article_set["PubmedArticle"]
            elif "PubmedBookArticle" in article_set:
                article_data = article_set["PubmedBookArticle"]

        if article_data:
            return self._parse_article_from_data(uid, article_data)

        return {
            "uid": uid,
            "Title": "Could not parse article",
            "Published": "",
            "Summary": "Article data could not be parsed.",
        }

    def _parse_article_from_data(self, uid: str, article_data: dict) -> dict:
        """Parse article metadata from article data structure."""
        try:
            if "MedlineCitation" in article_data:
                ar = article_data["MedlineCitation"]["Article"]
            elif "BookDocument" in article_data:
                ar = article_data["BookDocument"]
            else:
                raise KeyError("No valid article structure found")
        except KeyError:
            return {
                "uid": uid,
                "Title": "Could not parse article",
                "Published": "",
                "Summary": "Article data could not be parsed.",
            }

        # Extract abstract
        abstract_text = ar.get("Abstract", {}).get("AbstractText", [])
        summaries = []

        if isinstance(abstract_text, list):
            for txt in abstract_text:
                if isinstance(txt, dict) and "#text" in txt:
                    label = txt.get("@Label", "")
                    text_content = txt["#text"]
                    summaries.append(
                        f"{label}: {text_content}" if label else text_content
                    )
                elif isinstance(txt, str):
                    summaries.append(txt)
        elif isinstance(abstract_text, dict) and "#text" in abstract_text:
            summaries.append(abstract_text["#text"])
        elif isinstance(abstract_text, str):
            summaries.append(abstract_text)

        summary = "\n".join(summaries) if summaries else "No abstract available"

        # Get dates
        pub_date = ""
        if "ArticleDate" in ar:
            a_d = ar["ArticleDate"]
            if isinstance(a_d, list):
                a_d = a_d[0]
            pub_date = (
                f"{a_d.get('Year', '')}-{a_d.get('Month', '')}-{a_d.get('Day', '')}"
            )
        elif "Journal" in ar and "JournalIssue" in ar["Journal"]:
            p_d = ar["Journal"]["JournalIssue"].get("PubDate", {})
            pub_date = (
                f"{p_d.get('Year', '')}-{p_d.get('Month', '')}-{p_d.get('Day', '')}"
            )

        # Get journal
        journal = ar.get("Journal", {}).get("Title", "")

        return {
            "uid": uid,
            "Title": ar.get("ArticleTitle", ""),
            "Published": pub_date,
            "Journal": journal,
            "Copyright Information": ar.get("Abstract", {}).get(
                "CopyrightInformation", ""
            ),
            "Summary": summary,
        }

    def _safe_extract_text(self, field_value: Union[str, dict, None]) -> str:
        """
        Safely extract text content from a field that might be a string, dict, or None.

        Args:
            field_value: The field value to extract text from

        Returns:
            str: Extracted text content or empty string
        """
        if field_value is None:
            return ""

        if isinstance(field_value, str):
            return field_value

        if isinstance(field_value, dict):
            # Handle common XML-to-dict patterns
            if "#text" in field_value:
                return field_value["#text"]
            elif "text" in field_value:
                return field_value["text"]
            elif len(field_value) == 1:
                # If dict has only one key, try to get its value
                key = list(field_value.keys())[0]
                return str(field_value[key])
            else:
                # Fallback: try to construct meaningful text from dict
                text_parts = []
                for key, value in field_value.items():
                    if isinstance(value, str) and key not in ["sup", "sub", "@Label"]:
                        text_parts.append(value)
                return " ".join(text_parts) if text_parts else str(field_value)

        # Fallback for other types
        return str(field_value)

    def _convert_to_metadata(self, article: dict) -> PublicationMetadata:
        """Convert article dict to PublicationMetadata."""
        return PublicationMetadata(
            uid=article.get("uid", ""),
            title=self._safe_extract_text(article.get("Title")),
            journal=self._safe_extract_text(article.get("Journal")),
            published=self._safe_extract_text(article.get("Published")),
            pmid=article.get("uid", ""),
            abstract=self._safe_extract_text(article.get("Summary")),
        )

    def _extract_dataset_accessions(self, text: str) -> Dict[str, List[str]]:
        """
        Extract various dataset accessions from text.

        Uses centralized AccessionResolver for pattern matching.
        """
        from lobster.core.identifiers import get_accession_resolver

        resolver = get_accession_resolver()
        extracted = resolver.extract_accessions_by_type(text)

        # Map resolver output to legacy format for backward compatibility
        accessions = {
            "GEO": [],
            "SRA": [],
            "ArrayExpress": [],
            "ENA": [],
            "Platforms": [],
        }

        # GEO: include GSE and GSM
        if "GEO" in extracted:
            accessions["GEO"].extend(extracted["GEO"])
        if "GEO_Sample" in extracted:
            accessions["GEO"].extend(extracted["GEO_Sample"])

        # SRA: include all SRA types
        if "SRA" in extracted:
            accessions["SRA"].extend(extracted["SRA"])

        # ArrayExpress
        if "ArrayExpress" in extracted:
            accessions["ArrayExpress"].extend(extracted["ArrayExpress"])

        # ENA: include ENA and BioProject (PRJEB, PRJDB)
        if "ENA" in extracted:
            accessions["ENA"].extend(extracted["ENA"])
        if "BioProject" in extracted:
            # Only include international BioProjects (PRJEB, PRJDB) in ENA
            accessions["ENA"].extend(
                [acc for acc in extracted["BioProject"] if not acc.startswith("PRJNA")]
            )

        # Platforms: GPL
        if "GEO_Platform" in extracted:
            accessions["Platforms"].extend(extracted["GEO_Platform"])

        # Deduplicate
        for key in accessions:
            accessions[key] = sorted(list(set(accessions[key])))

        return accessions

    def _find_linked_datasets(self, pmid: str) -> Dict[str, List[str]]:
        """
        Find datasets linked to a PubMed article using a single combined E-Link call.

        Optimized to issue ONE E-Link request instead of four sequential calls,
        reducing network round trips from ~44s to ~3-5s.

        The combined response contains multiple linksetdbs, one per target database.
        We parse each linksetdb and map it back to the appropriate dataset type.

        Args:
            pmid: PubMed ID to find linked datasets for

        Returns:
            Dict with keys: GEO, SRA, BioProject, BioSample
        """
        linked = {"GEO": [], "SRA": [], "BioProject": [], "BioSample": []}

        # Database mapping: NCBI db name -> our key and prefix
        # Note: "gds" is the NCBI database name for GEO DataSets/Series
        # Note: "geo" also works for E-Link and returns the same UIDs
        # Note: BioProject/BioSample need accession lookup because E-Link returns
        #       internal UIDs, not accessions. Prefixes vary by repository:
        #       - NCBI: PRJNA/SAMN, EBI: PRJEB/SAME, DDBJ: PRJDB/SAMD
        db_mapping = {
            "gds": {"key": "GEO", "prefix": "", "needs_accession_lookup": True},
            "geo": {"key": "GEO", "prefix": "", "needs_accession_lookup": True},
            "sra": {"key": "SRA", "prefix": "SRA", "needs_accession_lookup": False},
            "bioproject": {
                "key": "BioProject",
                "prefix": "",
                "needs_accession_lookup": True,
            },
            "biosample": {
                "key": "BioSample",
                "prefix": "",
                "needs_accession_lookup": True,
            },
        }

        # Combined E-Link call: fetch all linked databases in one request
        # NCBI E-Link supports comma-separated db values
        # Using gds instead of geo for GEO datasets (gds is the official db name)
        combined_dbs = "gds,sra,bioproject,biosample"
        url = self.build_ncbi_url(
            "elink",
            {"dbfrom": "pubmed", "db": combined_dbs, "id": pmid, "retmode": "json"},
        )

        geo_uids = []  # Collect GEO UIDs for batch accession lookup
        bioproject_uids = []  # Collect BioProject UIDs for batch accession lookup
        biosample_uids = []  # Collect BioSample UIDs for batch accession lookup

        try:
            # Single combined request instead of 4 sequential calls
            content = self._make_ncbi_request(
                url, "find all linked datasets (combined)"
            )
            text = content.decode("utf-8")
            json_response = json.loads(text)

            if "linksets" in json_response:
                for linkset in json_response["linksets"]:
                    if "linksetdbs" not in linkset:
                        continue

                    for linksetdb in linkset["linksetdbs"]:
                        dbto = linksetdb.get("dbto", "")
                        links = linksetdb.get("links", [])

                        if dbto not in db_mapping:
                            logger.debug(f"Unknown linked database: {dbto}")
                            continue

                        config = db_mapping[dbto]
                        key = config["key"]

                        for link_id in links:
                            if config["needs_accession_lookup"]:
                                # Collect UIDs for batch lookup by database type
                                if key == "GEO":
                                    geo_uids.append(str(link_id))
                                elif key == "BioProject":
                                    bioproject_uids.append(str(link_id))
                                elif key == "BioSample":
                                    biosample_uids.append(str(link_id))
                            else:
                                # Non-lookup: construct accession directly
                                linked[key].append(f"{config['prefix']}{link_id}")

            logger.debug(
                f"Combined E-Link found: GEO UIDs={len(geo_uids)}, "
                f"SRA={len(linked['SRA'])}, BioProject UIDs={len(bioproject_uids)}, "
                f"BioSample UIDs={len(biosample_uids)}"
            )

        except Exception as e:
            logger.warning(f"Error in combined E-Link call: {e}")
            # Fall back to empty results rather than crashing
            return linked

        # Batch fetch GEO accessions (single esummary call instead of N calls)
        if geo_uids:
            geo_accessions = self._batch_fetch_geo_accessions(geo_uids)
            linked["GEO"].extend(geo_accessions)

        # Batch fetch BioProject accessions (handles PRJNA/PRJEB/PRJDB prefixes)
        if bioproject_uids:
            bioproject_accessions = self._batch_fetch_bioproject_accessions(
                bioproject_uids
            )
            linked["BioProject"].extend(bioproject_accessions)

        # Batch fetch BioSample accessions (handles SAMN/SAME/SAMD prefixes)
        if biosample_uids:
            biosample_accessions = self._batch_fetch_biosample_accessions(
                biosample_uids
            )
            linked["BioSample"].extend(biosample_accessions)

        # Deduplicate and prioritize GSE over GSM/GDS
        if linked["GEO"]:
            gse = [x for x in linked["GEO"] if x.startswith("GSE")]
            gsm = [x for x in linked["GEO"] if x.startswith("GSM")]
            gds = [x for x in linked["GEO"] if x.startswith("GDS")]
            # Prioritize GSE, but keep GSM/GDS as fallback
            linked["GEO"] = gse + gsm + gds

        for key in linked:
            linked[key] = list(set(linked[key]))

        return linked

    def _batch_fetch_geo_accessions(self, uids: List[str]) -> List[str]:
        """
        Batch fetch GEO accessions (GSE/GSM) from UIDs using a single esummary call.

        NCBI esummary supports comma-separated IDs (up to ~100 per request).
        This replaces N sequential calls with 1 batch call.

        Args:
            uids: List of GEO UIDs to resolve

        Returns:
            List of accession strings (GSE*, GSM*, GDS*)
        """
        if not uids:
            return []

        accessions = []

        # NCBI allows up to ~200 IDs per request, but 100 is safer
        batch_size = 100

        for i in range(0, len(uids), batch_size):
            batch_uids = uids[i : i + batch_size]
            ids_param = ",".join(batch_uids)

            url = self.build_ncbi_url(
                "esummary", {"db": "gds", "id": ids_param, "retmode": "json"}
            )

            try:
                content = self._make_ncbi_request(
                    url, f"batch fetch {len(batch_uids)} GEO accessions"
                )
                text = content.decode("utf-8")
                summary = json.loads(text)

                result = summary.get("result", {})
                # result contains "uids" list and individual uid entries
                for uid in batch_uids:
                    docsum = result.get(uid, {})
                    acc = docsum.get("accession")
                    if acc:
                        accessions.append(acc)
                    else:
                        logger.debug(f"No accession found for GEO UID {uid}")

                logger.debug(
                    f"Batch resolved {len(batch_uids)} GEO UIDs → {len(accessions)} accessions"
                )

            except Exception as e:
                logger.warning(f"Batch GEO accession fetch failed: {e}")
                # Fallback: try individual lookups for this batch
                for uid in batch_uids:
                    acc = self._fetch_geo_accession(uid)
                    if acc:
                        accessions.append(acc)

        return accessions

    def _fetch_geo_accession(self, uid: str) -> Optional[str]:
        """Fetch a single GEO accession (fallback for batch failures)."""
        try:
            url = self.build_ncbi_url(
                "esummary", {"db": "gds", "id": uid, "retmode": "json"}
            )
            content = self._make_ncbi_request(url, f"fetch GEO accession {uid}")
            text = content.decode("utf-8")
            summary = json.loads(text)
            docsum = summary.get("result", {}).get(uid, {})
            return docsum.get("accession")
        except Exception as e:
            logger.warning(f"Error fetching GEO accession for {uid}: {e}")
            return None

    def _batch_fetch_bioproject_accessions(self, uids: List[str]) -> List[str]:
        """
        Batch fetch BioProject accessions from UIDs using esummary.

        NCBI E-Link returns internal UIDs, not accessions. This method resolves
        UIDs to proper accessions which may have different prefixes:
        - NCBI (USA): PRJNA
        - EBI (Europe): PRJEB
        - DDBJ (Japan): PRJDB, PRJDA
        - CNGB (China): PRJCA

        Args:
            uids: List of BioProject UIDs to resolve

        Returns:
            List of BioProject accession strings (PRJNA*, PRJEB*, etc.)
        """
        if not uids:
            return []

        accessions = []
        batch_size = 100

        for i in range(0, len(uids), batch_size):
            batch_uids = uids[i : i + batch_size]
            ids_param = ",".join(batch_uids)

            url = self.build_ncbi_url(
                "esummary", {"db": "bioproject", "id": ids_param, "retmode": "json"}
            )

            try:
                content = self._make_ncbi_request(
                    url, f"batch fetch {len(batch_uids)} BioProject accessions"
                )
                text = content.decode("utf-8")
                summary = json.loads(text)

                result = summary.get("result", {})
                for uid in batch_uids:
                    docsum = result.get(uid, {})
                    # BioProject uses "project_acc" field for the accession
                    acc = docsum.get("project_acc")
                    if acc:
                        accessions.append(acc)
                    else:
                        logger.debug(f"No accession found for BioProject UID {uid}")

                logger.debug(
                    f"Batch resolved {len(batch_uids)} BioProject UIDs → {len(accessions)} accessions"
                )

            except Exception as e:
                logger.warning(f"Batch BioProject accession fetch failed: {e}")

        return accessions

    def _batch_fetch_biosample_accessions(self, uids: List[str]) -> List[str]:
        """
        Batch fetch BioSample accessions from UIDs using esummary.

        NCBI E-Link returns internal UIDs, not accessions. This method resolves
        UIDs to proper accessions which may have different prefixes:
        - NCBI (USA): SAMN
        - EBI (Europe): SAME
        - DDBJ (Japan): SAMD

        Args:
            uids: List of BioSample UIDs to resolve

        Returns:
            List of BioSample accession strings (SAMN*, SAME*, etc.)
        """
        if not uids:
            return []

        accessions = []
        batch_size = 100

        for i in range(0, len(uids), batch_size):
            batch_uids = uids[i : i + batch_size]
            ids_param = ",".join(batch_uids)

            url = self.build_ncbi_url(
                "esummary", {"db": "biosample", "id": ids_param, "retmode": "json"}
            )

            try:
                content = self._make_ncbi_request(
                    url, f"batch fetch {len(batch_uids)} BioSample accessions"
                )
                text = content.decode("utf-8")
                summary = json.loads(text)

                result = summary.get("result", {})
                for uid in batch_uids:
                    docsum = result.get(uid, {})
                    # BioSample uses "accession" field
                    acc = docsum.get("accession")
                    if acc:
                        accessions.append(acc)
                    else:
                        logger.debug(f"No accession found for BioSample UID {uid}")

                logger.debug(
                    f"Batch resolved {len(batch_uids)} BioSample UIDs → {len(accessions)} accessions"
                )

            except Exception as e:
                logger.warning(f"Batch BioSample accession fetch failed: {e}")

        return accessions

    def _find_linked_datasets_batch(
        self, pmids: List[str]
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Find datasets linked to multiple PubMed articles using batch E-Link calls.

        **100x speedup**: NCBI E-Link supports up to 200 PMIDs per request.
        Instead of 100 sequential requests, we make 1 batch request.

        Performance:
        - Before: 100 PMIDs × 0.1s = 10 seconds
        - After:  1 batch request = 0.1 seconds (100x faster!)

        Args:
            pmids: List of PubMed IDs to find linked datasets for

        Returns:
            Dict mapping PMID → {GEO: [...], SRA: [...], BioProject: [...], BioSample: [...]}

        Example:
            >>> results = provider._find_linked_datasets_batch(["12345", "67890"])
            >>> results["12345"]["GEO"]  # ["GSE123", "GSE456"]
        """
        if not pmids:
            return {}

        # NCBI E-Link supports up to 200 IDs per request
        batch_size = 200
        all_results = {}

        # Initialize empty results for all PMIDs
        for pmid in pmids:
            all_results[pmid] = {
                "GEO": [],
                "SRA": [],
                "BioProject": [],
                "BioSample": [],
            }

        # Process in batches of 200
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i : i + batch_size]
            logger.debug(
                f"Batch E-Link: processing {len(batch_pmids)} PMIDs "
                f"(batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1})"
            )

            # Database mapping
            # Note: BioProject/BioSample need accession lookup because E-Link returns
            #       internal UIDs, not accessions. Prefixes vary by repository:
            #       - NCBI: PRJNA/SAMN, EBI: PRJEB/SAME, DDBJ: PRJDB/SAMD
            db_mapping = {
                "gds": {"key": "GEO", "prefix": "", "needs_accession_lookup": True},
                "geo": {"key": "GEO", "prefix": "", "needs_accession_lookup": True},
                "sra": {"key": "SRA", "prefix": "SRA", "needs_accession_lookup": False},
                "bioproject": {
                    "key": "BioProject",
                    "prefix": "",
                    "needs_accession_lookup": True,
                },
                "biosample": {
                    "key": "BioSample",
                    "prefix": "",
                    "needs_accession_lookup": True,
                },
            }

            # Combined E-Link call with multiple PMIDs
            combined_dbs = "gds,sra,bioproject,biosample"
            ids_param = ",".join(batch_pmids)
            url = self.build_ncbi_url(
                "elink",
                {
                    "dbfrom": "pubmed",
                    "db": combined_dbs,
                    "id": ids_param,
                    "retmode": "json",
                },
            )

            # Collect UIDs by type and PMID for batch accession lookup
            geo_uids_by_pmid = {pmid: [] for pmid in batch_pmids}
            bioproject_uids_by_pmid = {pmid: [] for pmid in batch_pmids}
            biosample_uids_by_pmid = {pmid: [] for pmid in batch_pmids}

            try:
                content = self._make_ncbi_request(
                    url, f"batch find linked datasets ({len(batch_pmids)} PMIDs)"
                )
                text = content.decode("utf-8")
                json_response = json.loads(text)

                # Parse linksets (one per PMID in batch)
                if "linksets" in json_response:
                    for linkset in json_response["linksets"]:
                        # Get the PMID this linkset belongs to
                        idlist = linkset.get("ids", [])
                        if not idlist:
                            continue
                        pmid = str(idlist[0])  # First ID is the source PMID

                        if "linksetdbs" not in linkset:
                            continue

                        for linksetdb in linkset["linksetdbs"]:
                            dbto = linksetdb.get("dbto", "")
                            links = linksetdb.get("links", [])

                            if dbto not in db_mapping:
                                logger.debug(f"Unknown linked database: {dbto}")
                                continue

                            config = db_mapping[dbto]
                            key = config["key"]

                            for link_id in links:
                                if config["needs_accession_lookup"]:
                                    # Collect UIDs by type for batch accession lookup
                                    if key == "GEO":
                                        geo_uids_by_pmid[pmid].append(str(link_id))
                                    elif key == "BioProject":
                                        bioproject_uids_by_pmid[pmid].append(
                                            str(link_id)
                                        )
                                    elif key == "BioSample":
                                        biosample_uids_by_pmid[pmid].append(
                                            str(link_id)
                                        )
                                else:
                                    # Non-lookup: construct accession directly
                                    all_results[pmid][key].append(
                                        f"{config['prefix']}{link_id}"
                                    )

                total_geo_uids = sum(len(v) for v in geo_uids_by_pmid.values())
                total_sra = sum(len(all_results[pmid]["SRA"]) for pmid in batch_pmids)
                total_bioproject_uids = sum(
                    len(v) for v in bioproject_uids_by_pmid.values()
                )
                total_biosample_uids = sum(
                    len(v) for v in biosample_uids_by_pmid.values()
                )

                logger.debug(
                    f"Batch E-Link found: GEO UIDs={total_geo_uids}, "
                    f"SRA={total_sra}, BioProject UIDs={total_bioproject_uids}, "
                    f"BioSample UIDs={total_biosample_uids}"
                )

            except Exception as e:
                logger.warning(f"Error in batch E-Link call: {e}")
                # Continue with empty results for this batch
                continue

            # Batch fetch accessions for all PMIDs in this batch
            for pmid in batch_pmids:
                # GEO accessions
                if geo_uids_by_pmid[pmid]:
                    geo_accessions = self._batch_fetch_geo_accessions(
                        geo_uids_by_pmid[pmid]
                    )
                    all_results[pmid]["GEO"].extend(geo_accessions)

                # BioProject accessions (handles PRJNA/PRJEB/PRJDB prefixes)
                if bioproject_uids_by_pmid[pmid]:
                    bioproject_accessions = self._batch_fetch_bioproject_accessions(
                        bioproject_uids_by_pmid[pmid]
                    )
                    all_results[pmid]["BioProject"].extend(bioproject_accessions)

                # BioSample accessions (handles SAMN/SAME/SAMD prefixes)
                if biosample_uids_by_pmid[pmid]:
                    biosample_accessions = self._batch_fetch_biosample_accessions(
                        biosample_uids_by_pmid[pmid]
                    )
                    all_results[pmid]["BioSample"].extend(biosample_accessions)

                # Deduplicate and prioritize GSE over GSM/GDS
                if all_results[pmid]["GEO"]:
                    gse = [x for x in all_results[pmid]["GEO"] if x.startswith("GSE")]
                    gsm = [x for x in all_results[pmid]["GEO"] if x.startswith("GSM")]
                    gds = [x for x in all_results[pmid]["GEO"] if x.startswith("GDS")]
                    all_results[pmid]["GEO"] = gse + gsm + gds

                # Deduplicate all lists
                for key in all_results[pmid]:
                    all_results[pmid][key] = list(set(all_results[pmid][key]))

        logger.debug(
            f"Batch E-Link complete: processed {len(pmids)} PMIDs across "
            f"{(len(pmids)-1)//batch_size + 1} batch(es)"
        )

        return all_results

    def _check_supplementary_materials(self, article: Dict) -> Dict[str, List[str]]:
        """Check for dataset mentions in supplementary materials."""
        # This would require additional API calls or web scraping
        # For now, return empty dict as placeholder
        return {"Supplementary": []}

    def _format_comprehensive_dataset_report(
        self,
        article: Dict,
        datasets: Dict[str, List[str]],
        doi: Optional[str],
        pmid: Optional[str],
    ) -> str:
        """Format a comprehensive dataset discovery report with GSE prioritized."""
        logger.debug(
            f"[DEFENSIVE] _format_comprehensive_dataset_report called with: datasets type={type(datasets)}, datasets={datasets}"
        )
        response = "## Dataset Discovery Report\n\n"

        # Publication info (compact)
        title = str(article.get("Title", "N/A"))  # Ensure string before slicing
        response += f"**Publication**: {title[:100]}...\n"
        response += f"**PMID**: {pmid or 'N/A'} | **DOI**: {doi or 'N/A'}\n"
        response += f"**Journal**: {article.get('Journal', 'N/A')} ({article.get('Published', 'N/A')})\n\n"

        # Datasets found
        total_datasets = sum(len(v) for v in datasets.values())

        if total_datasets == 0:
            response += "**No datasets found**. Check supplementary materials or contact authors.\n"
        else:
            response += "**Found dataset(s)**:\n\n"

            dataset_lines = []

            # --- GEO prioritization: GSE > GSM > GDS ---
            if datasets.get("GEO"):
                logger.debug(
                    f"[DEFENSIVE] Processing GEO datasets: type={type(datasets['GEO'])}, value={datasets['GEO']}"
                )
                gse = [x for x in datasets["GEO"] if x.startswith("GSE")]
                gsm = [x for x in datasets["GEO"] if x.startswith("GSM")]
                gds = [x for x in datasets["GEO"] if x.startswith("GDS")]
                logger.debug(
                    f"[DEFENSIVE] After GEO categorization: gse={gse}, gsm={gsm}, gds={gds}"
                )

                if gse:
                    for acc in gse:
                        dataset_lines.append(
                            f"- GEO (Series): [{acc}](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc})"
                        )

                if gsm:
                    for acc in gsm[:2]:
                        dataset_lines.append(
                            f"- GEO (Sample): [{acc}](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc})"
                        )
                    if len(gsm) > 2:
                        dataset_lines.append(
                            f"  ... and {len(gsm) - 2} more GEO samples"
                        )

                if gds:
                    for acc in gds[:2]:
                        dataset_lines.append(
                            f"- GEO (DataSet): [{acc}](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc})"
                        )
                    if len(gds) > 2:
                        dataset_lines.append(
                            f"  ... and {len(gds) - 2} more GEO datasets"
                        )

            # --- SRA ---
            # FIXME activate these once ready
            # if datasets.get('SRA'):
            #     for acc in datasets['SRA'][:3]:
            #         dataset_lines.append(f"- SRA: [{acc}](https://www.ncbi.nlm.nih.gov/sra/{acc})")
            #     if len(datasets['SRA']) > 3:
            #         dataset_lines.append(f"  ... and {len(datasets['SRA']) - 3} more SRA datasets")

            # # --- ArrayExpress ---
            # if datasets.get('ArrayExpress'):
            #     for acc in datasets['ArrayExpress'][:2]:
            #         dataset_lines.append(f"- ArrayExpress: [{acc}](https://www.ebi.ac.uk/arrayexpress/experiments/{acc}/)")

            # # --- BioProject ---
            # if datasets.get('BioProject'):
            #     for acc in datasets['BioProject'][:2]:
            #         dataset_lines.append(f"- BioProject: [{acc}](https://www.ncbi.nlm.nih.gov/bioproject/{acc})")

            # # --- BioSample ---
            # if datasets.get('BioSample'):
            #     for acc in datasets['BioSample'][:2]:
            #         dataset_lines.append(f"- BioSample: [{acc}](https://www.ncbi.nlm.nih.gov/biosample/{acc})")

            response += "\n".join(dataset_lines) + "\n\n"

        # Brief next steps
        response += "**Next steps**: "
        steps = []
        if datasets.get("GEO"):
            logger.debug(
                f"[DEFENSIVE] Building next steps. datasets['GEO']: type={type(datasets['GEO'])}, len={len(datasets['GEO']) if isinstance(datasets['GEO'], list) else 'NOT_LIST'}, value={datasets['GEO']}"
            )
            # Prefer GSE if present, otherwise first available accession
            preferred_geo = next(
                (x for x in datasets["GEO"] if x.startswith("GSE")), datasets["GEO"][0]
            )
            logger.debug(f"[DEFENSIVE] Selected preferred_geo: {preferred_geo}")
            steps.append(f"`download_geo_dataset('{preferred_geo}')`")
        steps.append("`extract_computational_methods()`")
        response += " → ".join(steps) + "\n"

        return response

    def _extract_methods_from_text(
        self, text: str, method_type: str, include_parameters: bool
    ) -> Dict[str, List]:
        """Extract computational methods from text using pattern matching."""
        methods = {
            "preprocessing": [],
            "quality_control": [],
            "normalization": [],
            "clustering": [],
            "differential_expression": [],
            "trajectory": [],
            "integration": [],
            "tools": [],
            "parameters": {},
        }

        # Define method patterns for extraction
        method_patterns = {
            "preprocessing": [
                r"filter(?:ed|ing)?\s+(?:cells?\s+)?(?:with\s+)?(\d+(?:,\d+)?\s+(?:genes?|UMIs?|counts?))",
                r"(?:minimum|min)\s+(?:of\s+)?(\d+)\s+(?:genes?|features?)",
                r"(?:removed?|excluded?)\s+(?:cells?\s+)?(?:with\s+)?(?:less\s+than\s+)?(\d+)",
                r"mitochondrial\s+(?:gene\s+)?(?:content|percentage|threshold)\s*[<>]?\s*(\d+(?:\.\d+)?%?)",
            ],
            "quality_control": [
                r"quality\s+control\s+(?:parameters?|thresholds?|criteria)",
                r"(?:doublet|multiplet)\s+(?:detection|removal|filtering)",
                r"(?:MAD|median\s+absolute\s+deviation)\s*=?\s*(\d+(?:\.\d+)?)",
            ],
            "normalization": [
                r"(?:normalized?|normalization)\s+(?:using|with|by)\s+(\w+)",
                r"(SCTransform|LogNormalize|TPM|CPM|RPKM|FPKM)",
                r"scale\s*factor\s*[=:]?\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            ],
            "clustering": [
                r"(?:resolution|clustering\s+resolution)\s*[=:]?\s*(\d+(?:\.\d+)?)",
                r"k\s*[=:]?\s*(\d+)\s+(?:neighbors?|nearest)",
                r"(?:Louvain|Leiden|k-means|hierarchical)\s+clustering",
            ],
            "tools": [
                r"(Seurat|Scanpy|Cell\s*Ranger|STAR|kallisto|salmon)",
                r"(?:version|v)\s*(\d+(?:\.\d+)*)",
                r"(?:github\.com/[\w-]+/[\w-]+)",
            ],
        }

        # Extract methods from text
        if method_type == "all":
            search_types = method_patterns.keys()
        else:
            search_types = [method_type] if method_type in method_patterns else []

        for mtype in search_types:
            if mtype not in method_patterns:
                continue

            for pattern in method_patterns[mtype]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    method_info = {
                        "type": mtype,
                        "pattern_matched": pattern,
                        "text": match.group(0),
                        "value": match.groups()[0] if match.groups() else None,
                    }
                    methods[mtype].append(method_info)

                    # Extract parameter values if requested
                    if include_parameters and match.groups():
                        param_key = f"{mtype}_{pattern.split('(')[0].strip()}"
                        methods["parameters"][param_key] = match.groups()[0]

        return methods

    def _extract_github_repos(self, text: str) -> List[str]:
        """Extract GitHub repositories from text."""
        github_pattern = r"github\.com/([\w-]+)/([\w-]+)"
        github_matches = re.finditer(github_pattern, text, re.IGNORECASE)
        return [f"https://github.com/{m.group(1)}/{m.group(2)}" for m in github_matches]

    def _get_pmc_full_text(self, identifier: str):
        """
        Get PMC full text for an identifier if available.

        This method uses PMCProvider to extract structured full text from PMC XML API.
        Returns None if PMC full text is not available.

        Args:
            identifier: PMID or DOI

        Returns:
            PMCFullText object if available, None otherwise
        """
        try:
            from lobster.tools.providers.pmc_provider import (
                PMCNotAvailableError,
                PMCProvider,
            )

            # Initialize PMC provider (reuses same config)
            pmc_provider = PMCProvider(
                data_manager=self.data_manager, config=self.config
            )

            # Try to extract PMC full text
            pmc_result = pmc_provider.extract_full_text(identifier)
            logger.debug(
                f"PMC full text retrieved: {len(pmc_result.methods_section)} chars methods"
            )
            return pmc_result

        except PMCNotAvailableError:
            logger.debug(f"PMC full text not available for: {identifier}")
            return None
        except Exception as e:
            logger.warning(f"Error retrieving PMC full text: {e}")
            return None
