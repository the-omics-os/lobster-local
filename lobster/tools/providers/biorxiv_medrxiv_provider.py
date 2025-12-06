"""
BioRxiv and MedRxiv provider for preprint access and full-text extraction.

This provider implements a unified interface for accessing preprints from both bioRxiv
and medRxiv via the bioRxiv REST API (https://api.biorxiv.org/). It provides:

- Literature search across preprints
- Full-text JATS XML extraction and parsing
- Preprint→published journal mapping
- Funding agency and publisher filtering
- Content and usage statistics

The provider reuses PMCProvider's JATS XML parsing logic via composition to extract
methods, tables, software tools, and GitHub repositories from preprint full text.

Architecture:
- Single provider handles both bioRxiv and medRxiv via server parameter
- Priority 10 (high priority, same as PubMed/PMC)
- No authentication required (public API)
- Rate limiting: 1 req/s (polite crawling)
- JATS XML parsing via PMCProvider composition

API Reference:
https://api.biorxiv.org/ (public REST API, no key required)
"""

import json
import random
import re
import time
import urllib.error
import urllib.parse
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.base_provider import (
    BasePublicationProvider,
    DatasetType,
    PublicationMetadata,
    PublicationSource,
)
from lobster.tools.providers.biorxiv_medrxiv_config import BioRxivMedRxivConfig
from lobster.tools.providers.pmc_provider import PMCFullText, PMCProvider
from lobster.tools.rate_limiter import CHROME_USER_AGENT, STEALTH_HEADERS
from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context, handle_ssl_error

# Cloudscraper for Cloudflare-protected content servers (www.biorxiv.org)
# The API (api.biorxiv.org) is friendly, but content delivery requires JS challenge solving
try:
    import cloudscraper

    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

logger = get_logger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class BioRxivMedRxivError(Exception):
    """Base exception for bioRxiv/medRxiv provider errors."""

    pass


class BioRxivAPIError(BioRxivMedRxivError):
    """API request failed (network error, timeout, etc.)."""

    pass


class BioRxivNotFoundError(BioRxivMedRxivError):
    """Preprint not found (404 error)."""

    pass


class BioRxivJATSError(BioRxivMedRxivError):
    """JATS XML parsing or fetching failed."""

    pass


class BioRxivRateLimitError(BioRxivMedRxivError):
    """Rate limit exceeded (429 error)."""

    pass


# ============================================================================
# Provider Implementation
# ============================================================================


class BioRxivMedRxivProvider(BasePublicationProvider):
    """
    Unified provider for bioRxiv and medRxiv preprint servers.

    This provider implements comprehensive access to bioRxiv and medRxiv via the
    bioRxiv REST API, including literature search, full-text JATS XML extraction,
    and preprint→journal mapping.

    Key Features:
    - Single provider handles both bioRxiv and medRxiv (via server parameter)
    - Full-text JATS XML extraction and parsing (reuses PMCProvider)
    - Preprint→published journal mapping (/pubs endpoint)
    - Funding agency filtering (/funder endpoint)
    - Publisher filtering (/publisher endpoint)
    - Content and usage statistics (/sum, /usage endpoints)

    Coverage:
    - bioRxiv: 200K+ preprints (biology, neuroscience, genomics, etc.)
    - medRxiv: 50K+ preprints (medicine, epidemiology, public health, etc.)

    Examples:
        >>> provider = BioRxivMedRxivProvider(data_manager)
        >>>
        >>> # Search bioRxiv
        >>> results = provider.search_publications("CRISPR", server="biorxiv")
        >>>
        >>> # Get full text
        >>> full_text = provider.get_full_text("10.1101/2024.01.01.123456")
        >>> print(full_text.methods_section)
        >>>
        >>> # Check if published
        >>> mapping = provider.get_publication_mapping("10.1101/2024.01.01.123456")
        >>> if mapping and mapping['published_doi']:
        >>>     print(f"Published in: {mapping['published_journal']}")
    """

    def __init__(
        self,
        data_manager: Optional[DataManagerV2] = None,
        config: Optional[BioRxivMedRxivConfig] = None,
    ):
        """
        Initialize bioRxiv/medRxiv provider.

        Args:
            data_manager: Optional DataManagerV2 for provenance tracking
            config: Optional configuration, uses defaults if not provided
        """
        self.data_manager = data_manager
        self.config = config or BioRxivMedRxivConfig()

        # Composition: Use PMCProvider for JATS XML parsing
        self._jats_parser = PMCProvider(data_manager=data_manager)

        # API session with polite bot headers (for api.biorxiv.org)
        # The API is bot-friendly and prefers identification
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Lobster Bioinformatics/1.0 (+https://omics-os.com; mailto:support@omics-os.com)",
                "Accept": "application/xml,text/xml,application/json,*/*;q=0.1",
            }
        )

        # Content delivery session with Cloudflare bypass (for www.biorxiv.org)
        # Content servers require JS challenge solving + TLS fingerprinting to avoid 403
        # Note: api.biorxiv.org is friendly (uses self.session above), but
        # www.biorxiv.org content delivery is Cloudflare-protected
        if CLOUDSCRAPER_AVAILABLE:
            self.content_session = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "darwin", "mobile": False},
            )
            logger.debug("Using cloudscraper for bioRxiv/medRxiv content delivery")
        else:
            # Fallback to requests.Session with stealth headers (may fail on 403)
            self.content_session = requests.Session()
            self.content_session.headers.update(
                {
                    "User-Agent": CHROME_USER_AGENT,
                    **STEALTH_HEADERS,
                }
            )
            logger.warning(
                "cloudscraper not available - bioRxiv content fetching may fail. "
                "Install with: pip install cloudscraper"
            )

        # Rate limiting state
        self._last_request_time = 0.0
        self._consecutive_failures = 0
        self._circuit_breaker_until = 0.0

        logger.debug(
            f"Initialized BioRxivMedRxivProvider with server={self.config.default_server}"
        )

    # ========================================================================
    # BasePublicationProvider Interface (Properties)
    # ========================================================================

    @property
    def source(self) -> PublicationSource:
        """
        Return publication source.

        Note: Returns BIORXIV by default, but provider handles both bioRxiv
        and medRxiv via server parameter.
        """
        return PublicationSource.BIORXIV

    @property
    def supported_dataset_types(self) -> List[DatasetType]:
        """
        Return list of dataset types supported by this provider.

        Note: Preprints may reference datasets, but bioRxiv/medRxiv don't
        host datasets directly. Dataset discovery is best-effort via regex
        extraction from JATS XML full-text.

        Returns:
            List[DatasetType]: Empty list (no direct dataset hosting)
        """
        return []

    @property
    def priority(self) -> int:
        """
        Return provider priority for capability-based routing.

        BioRxiv/MedRxiv has high priority (10) due to:
        - Fast API response (<500ms)
        - Authoritative source for preprints
        - Structured JATS XML for full-text
        - No authentication required

        Priority same as PubMed/PMC for preprint literature.

        Returns:
            int: Priority 10 (high priority)
        """
        return 10

    def get_supported_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities supported by bioRxiv/medRxiv provider.

        This provider excels at preprint search and full-text extraction via
        structured JATS XML. It provides comprehensive access to bioRxiv and
        medRxiv content with fast response times and no authentication.

        Supported capabilities:
        - SEARCH_LITERATURE: Search preprints by keyword, category, date
        - EXTRACT_METADATA: Structured metadata from API
        - QUERY_CAPABILITIES: Dynamic capability discovery
        - GET_ABSTRACT: Fast abstract retrieval from API
        - GET_FULL_CONTENT: Full-text via JATS XML (composition with PMCProvider)
        - EXTRACT_METHODS: Methods section extraction from JATS XML

        Not supported:
        - DISCOVER_DATASETS: No direct dataset hosting
        - FIND_LINKED_DATASETS: No ELink-style authoritative linking
        - VALIDATE_METADATA: No dataset validation
        - EXTRACT_PDF: Uses structured XML, not PDF
        - INTEGRATE_MULTI_OMICS: No multi-omics integration

        Coverage: 250K+ preprints across biology, medicine, neuroscience

        Returns:
            Dict[str, bool]: Capability support mapping
        """
        from lobster.tools.providers.base_provider import ProviderCapability

        return {
            ProviderCapability.SEARCH_LITERATURE: True,
            ProviderCapability.DISCOVER_DATASETS: False,
            ProviderCapability.FIND_LINKED_DATASETS: False,
            ProviderCapability.EXTRACT_METADATA: True,
            ProviderCapability.VALIDATE_METADATA: False,
            ProviderCapability.QUERY_CAPABILITIES: True,
            ProviderCapability.GET_ABSTRACT: True,
            ProviderCapability.GET_FULL_CONTENT: True,
            ProviderCapability.EXTRACT_METHODS: True,
            ProviderCapability.EXTRACT_PDF: False,
            ProviderCapability.INTEGRATE_MULTI_OMICS: False,
        }

    def validate_identifier(self, identifier: str) -> bool:
        """
        Validate bioRxiv/medRxiv DOI format.

        BioRxiv/medRxiv DOIs follow the pattern: 10.1101/YYYY.MM.DD.######

        Args:
            identifier: DOI to validate

        Returns:
            bool: True if identifier matches bioRxiv/medRxiv DOI pattern

        Examples:
            >>> provider.validate_identifier("10.1101/2024.01.01.123456")
            True
            >>> provider.validate_identifier("10.1038/s41586-024-00001-x")
            False
        """
        identifier = identifier.strip()

        # BioRxiv/medRxiv DOI pattern: 10.1101/YYYY.MM.DD.######
        pattern = r"^10\.1101/\d{4}\.\d{2}\.\d{2}\.\d+$"
        return bool(re.match(pattern, identifier))

    def get_supported_features(self) -> Dict[str, bool]:
        """Return features supported by bioRxiv/medRxiv provider."""
        return {
            "literature_search": True,
            "dataset_discovery": False,  # Best-effort regex only
            "metadata_extraction": True,
            "full_text_access": True,
            "advanced_filtering": True,  # Category, date, funder, publisher
            "computational_methods": True,
            "github_extraction": True,
            "preprint_journal_mapping": True,  # Unique to preprints
        }

    # ========================================================================
    # API Request Infrastructure
    # ========================================================================

    def _build_url(self, endpoint: str, server: Optional[str] = None, **params) -> str:
        """
        Build bioRxiv API URL with proper parameter encoding.

        Args:
            endpoint: API endpoint (e.g., "details", "pubs", "funder")
            server: Optional server override ("biorxiv" | "medrxiv")
            **params: Additional URL parameters

        Returns:
            str: Properly formatted API URL

        Examples:
            >>> provider._build_url("details", server="biorxiv", interval="2024-01-01/2024-01-31", cursor=0, format="json")
            "https://api.biorxiv.org/details/biorxiv/2024-01-01/2024-01-31/0/json"
        """
        server = server or self.config.default_server
        base = self.config.base_url.rstrip("/")

        # Filter out None values
        params = {k: v for k, v in params.items() if v is not None}

        # Construct URL based on endpoint
        if endpoint == "details":
            # /details/[server]/[interval]/[cursor]/[format]
            interval = params.get("interval", "")
            cursor = params.get("cursor", 0)
            fmt = params.get("format", "json")
            url = f"{base}/details/{server}/{interval}/{cursor}/{fmt}"

        elif endpoint == "pubs":
            # /pubs/[server]/[interval]/[cursor]
            interval = params.get("interval", "")
            cursor = params.get("cursor", 0)
            url = f"{base}/pubs/{server}/{interval}/{cursor}"

        elif endpoint == "funder":
            # /funder/[server]/[interval]/[funder_ror]/[cursor]/[format]
            interval = params.get("interval", "")
            funder_ror = params.get("funder_ror", "")
            cursor = params.get("cursor", 0)
            fmt = params.get("format", "json")
            url = f"{base}/funder/{server}/{interval}/{funder_ror}/{cursor}/{fmt}"

        elif endpoint == "publisher":
            # /publisher/[publisher_prefix]/[interval]/[cursor]
            publisher_prefix = params.get("publisher_prefix", "")
            interval = params.get("interval", "")
            cursor = params.get("cursor", 0)
            url = f"{base}/publisher/{publisher_prefix}/{interval}/{cursor}"

        elif endpoint == "sum":
            # /sum/[interval]/[format]
            interval = params.get("interval", "m")  # Monthly by default
            fmt = params.get("format", "json")
            url = f"{base}/sum/{interval}/{fmt}"

        elif endpoint == "usage":
            # /usage/[interval]/[format]
            interval = params.get("interval", "m")  # Monthly by default
            fmt = params.get("format", "json")
            url = f"{base}/usage/{interval}/{fmt}"

        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")

        # Add query parameters if specified (e.g., category filter)
        query_params = params.get("query_params")
        if query_params:
            query_string = urllib.parse.urlencode(query_params)
            url = f"{url}?{query_string}"

        return url

    def _make_api_request(self, url: str, operation_name: str = "API request") -> bytes:
        """
        Centralized API request handler with retry logic and rate limiting.

        Args:
            url: The API URL to request
            operation_name: Human-readable name for logging/errors

        Returns:
            bytes: Response content

        Raises:
            BioRxivAPIError: For permanent failures or after exhausting retries
            BioRxivNotFoundError: For 404 errors (preprint not found)
            BioRxivRateLimitError: For 429 errors (rate limit exceeded)
        """
        # Check circuit breaker
        current_time = time.time()
        if current_time < self._circuit_breaker_until:
            remaining_time = int(self._circuit_breaker_until - current_time)
            raise BioRxivAPIError(
                f"bioRxiv/medRxiv requests temporarily disabled due to repeated failures. "
                f"Try again in {remaining_time} seconds."
            )

        # Apply rate limiting
        self._apply_rate_limiting()

        attempt = 0
        last_exception = None

        while attempt < self.config.max_retry:
            try:
                logger.debug(
                    f"bioRxiv/medRxiv {operation_name} attempt {attempt + 1}/{self.config.max_retry}: {url[:100]}..."
                )

                # Make the request using session with proper headers
                response = self.session.get(url, timeout=self.config.timeout)
                response.raise_for_status()

                content = response.content

                # Reset failure counter on success
                self._consecutive_failures = 0
                logger.debug(f"bioRxiv/medRxiv {operation_name} successful")

                return content

            except Exception as e:
                last_exception = e
                attempt += 1

                # Check for HTTP errors (requests exceptions)
                if isinstance(e, requests.exceptions.HTTPError):
                    status_code = e.response.status_code if e.response else None

                    if status_code == 404:  # Not found (don't retry)
                        logger.error(f"bioRxiv/medRxiv 404: Preprint not found: {url}")
                        self._consecutive_failures += 1
                        raise BioRxivNotFoundError(
                            f"Preprint not found: {operation_name}"
                        )

                    elif status_code == 429:  # Rate limit exceeded
                        logger.warning(
                            f"bioRxiv/medRxiv rate limit hit for {operation_name} (attempt {attempt})"
                        )
                        self._handle_rate_limit_error(attempt, operation_name)

                    elif status_code in [500, 502, 503, 504]:  # Server errors (retry)
                        logger.warning(
                            f"bioRxiv/medRxiv server error {status_code} for {operation_name} (attempt {attempt})"
                        )
                        self._handle_server_error(status_code, attempt, operation_name)

                    elif status_code == 400:  # Bad request (don't retry)
                        logger.error(
                            f"bioRxiv/medRxiv 400: Bad request for {operation_name}: {str(e)}"
                        )
                        self._consecutive_failures += 1
                        raise BioRxivAPIError(
                            f"Bad request: {operation_name} - {str(e)}"
                        )

                    else:  # Other HTTP errors
                        logger.warning(
                            f"bioRxiv/medRxiv HTTP error {status_code} for {operation_name} (attempt {attempt}): {str(e)}"
                        )
                        if attempt >= self.config.max_retry:
                            break
                        self._apply_backoff_delay(attempt)

                # Check for network/SSL errors
                elif isinstance(
                    e,
                    (
                        requests.exceptions.RequestException,
                        requests.exceptions.SSLError,
                    ),
                ):
                    # Check for SSL certificate errors
                    error_str = str(e)
                    if "CERTIFICATE_VERIFY_FAILED" in error_str or "SSL" in error_str:
                        handle_ssl_error(e, url, logger)
                        self._consecutive_failures += 1
                        raise BioRxivAPIError(
                            f"SSL certificate verification failed for {operation_name}. "
                            f"See error message above for solutions. "
                            f"Original error: {error_str}"
                        )

                    logger.warning(
                        f"bioRxiv/medRxiv network error for {operation_name} (attempt {attempt}): {str(e)}"
                    )

                    if attempt < self.config.max_retry:
                        self._apply_backoff_delay(attempt)

        # All retries exhausted
        self._consecutive_failures += 1
        self._maybe_activate_circuit_breaker()

        error_msg = f"bioRxiv/medRxiv {operation_name} failed after {self.config.max_retry} attempts"
        if last_exception:
            error_msg += f": {str(last_exception)}"

        raise BioRxivAPIError(error_msg)

    def _apply_rate_limiting(self) -> None:
        """
        Apply rate limiting between requests.

        Enforces the configured requests_per_second limit using in-memory
        tracking. Defaults to 1 req/s (polite crawling).
        """
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        min_delay = 1.0 / self.config.requests_per_second

        if time_since_last < min_delay:
            sleep_time = min_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def _handle_rate_limit_error(self, attempt: int, operation_name: str) -> None:
        """Handle rate limiting errors with progressive backoff."""
        if attempt < self.config.max_retry:
            # Progressive backoff with jitter for rate limits
            base_delay = self.config.backoff_delay * (2**attempt)
            jitter = random.uniform(0, base_delay * 0.1)
            delay = base_delay + jitter

            logger.debug(
                f"Rate limited by bioRxiv/medRxiv, waiting {delay:.1f}s before retry..."
            )
            time.sleep(delay)

    def _handle_server_error(
        self, status_code: int, attempt: int, operation_name: str
    ) -> None:
        """Handle server errors with appropriate backoff."""
        if attempt < self.config.max_retry:
            # Shorter delays for server errors (they often resolve quickly)
            delay = min(2**attempt, 10)
            logger.debug(
                f"bioRxiv/medRxiv server error {status_code}, waiting {delay}s before retry..."
            )
            time.sleep(delay)

    def _apply_backoff_delay(self, attempt: int) -> None:
        """Apply exponential backoff delay."""
        delay = self.config.backoff_delay * (2**attempt)
        jitter = random.uniform(0, delay * 0.1)
        total_delay = delay + jitter
        logger.debug(f"Applying backoff delay: {total_delay:.2f}s")
        time.sleep(total_delay)

    def _maybe_activate_circuit_breaker(self) -> None:
        """Activate circuit breaker if too many consecutive failures."""
        if self._consecutive_failures >= 10:  # Threshold: 10 consecutive failures
            self._circuit_breaker_until = time.time() + 300  # 5 minutes
            logger.warning(
                f"bioRxiv/medRxiv circuit breaker activated after {self._consecutive_failures} failures. "
                f"Requests disabled for 5 minutes."
            )

    # ========================================================================
    # Helper Methods for /details Endpoint
    # ========================================================================

    def _fetch_details(
        self,
        doi: Optional[str] = None,
        server: Optional[str] = None,
        date_range: Optional[str] = None,
        category: Optional[str] = None,
        cursor: int = 0,
    ) -> List[Dict]:
        """
        Fetch preprint details from /details endpoint.

        Args:
            doi: Optional DOI for single preprint lookup
            server: Optional server override ("biorxiv" | "medrxiv")
            date_range: Optional date range ("YYYY-MM-DD/YYYY-MM-DD")
            category: Optional category filter
            cursor: Pagination cursor (default: 0)

        Returns:
            List[Dict]: List of preprint records

        Raises:
            BioRxivNotFoundError: Preprint not found
            BioRxivAPIError: API request failed
        """
        server = server or self.config.default_server

        # Build URL based on query type
        if doi:
            # Single preprint lookup: /details/[server]/[DOI]/na/json
            url = self._build_url(
                "details", server=server, interval=doi, cursor="na", format="json"
            )
        else:
            # Date range query: /details/[server]/[interval]/[cursor]/json
            if not date_range:
                raise ValueError(
                    "date_range required for search (e.g., '2024-01-01/2024-01-31')"
                )

            # Add category filter if specified
            query_params = {}
            if category:
                query_params["category"] = category

            url = self._build_url(
                "details",
                server=server,
                interval=date_range,
                cursor=cursor,
                format="json",
                query_params=query_params if query_params else None,
            )

        # Make API request
        content = self._make_api_request(url, f"fetch details ({server})")

        # Parse JSON response
        response = json.loads(content.decode("utf-8"))

        # Extract collection (list of preprints)
        collection = response.get("collection", [])

        if not collection:
            if doi:
                raise BioRxivNotFoundError(f"Preprint not found: {doi}")
            else:
                logger.debug(f"No preprints found for query: {date_range}, {category}")

        return collection

    def _parse_preprint_to_metadata(self, preprint: Dict) -> PublicationMetadata:
        """
        Parse preprint record to PublicationMetadata.

        Args:
            preprint: Preprint record from API

        Returns:
            PublicationMetadata: Standardized metadata
        """
        # Parse authors (format: "Smith J, Doe J, Lee K")
        authors_str = preprint.get("authors", "")
        authors = [a.strip() for a in authors_str.split(",") if a.strip()]

        return PublicationMetadata(
            uid=preprint.get("doi", ""),
            title=preprint.get("title", ""),
            journal=preprint.get("server", ""),  # bioRxiv or medRxiv
            published=preprint.get("date", ""),
            doi=preprint.get("doi", ""),
            pmid=None,  # Preprints don't have PMIDs
            abstract=preprint.get("abstract", ""),
            authors=authors,
            keywords=[preprint.get("category", "")] if preprint.get("category") else [],
        )

    def _filter_preprints_by_query(
        self, preprints: List[Dict], query: str
    ) -> List[Dict]:
        """
        Filter preprints by keyword match in title/abstract/authors.

        Args:
            preprints: List of preprint records
            query: Search query string

        Returns:
            List[Dict]: Filtered preprints
        """
        query_lower = query.lower()
        filtered = []

        for preprint in preprints:
            # Check title
            title = preprint.get("title", "").lower()
            if query_lower in title:
                filtered.append(preprint)
                continue

            # Check abstract
            abstract = preprint.get("abstract", "").lower()
            if query_lower in abstract:
                filtered.append(preprint)
                continue

            # Check authors
            authors = preprint.get("authors", "").lower()
            if query_lower in authors:
                filtered.append(preprint)
                continue

        return filtered

    # ========================================================================
    # BasePublicationProvider Interface (Abstract Methods)
    # ========================================================================

    def search_publications(
        self,
        query: str,
        max_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Search bioRxiv/medRxiv for preprints.

        This method searches preprints by keyword match in title/abstract/authors,
        with optional filtering by date range, category, and server.

        Args:
            query: Search query string (keyword to match)
            max_results: Maximum number of results (default: 5)
            filters: Optional filters:
                - date_range: "YYYY-MM-DD/YYYY-MM-DD" (required for search)
                - category: Category filter (e.g., "bioinformatics")
                - server: Server override ("biorxiv" | "medrxiv")
            **kwargs: Additional parameters

        Returns:
            str: Formatted search results

        Examples:
            >>> provider.search_publications(
            ...     "CRISPR",
            ...     max_results=10,
            ...     filters={"date_range": "2024-01-01/2024-01-31", "category": "bioinformatics"}
            ... )
        """
        logger.debug(f"bioRxiv/medRxiv search: {query[:50]}...")

        try:
            # Apply configuration limits
            max_results = min(max_results, self.config.max_results)

            # Extract filters
            filters = filters or {}
            date_range = filters.get("date_range")
            category = filters.get("category")
            server = filters.get("server") or kwargs.get("server")

            if not date_range:
                # Default to last 30 days
                from datetime import datetime, timedelta

                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                date_range = (
                    f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
                )
                logger.debug(f"No date_range specified, using default: {date_range}")

            # Fetch preprints with pagination
            all_preprints = []
            cursor = 0

            while len(all_preprints) < max_results:
                # Fetch page
                preprints = self._fetch_details(
                    server=server,
                    date_range=date_range,
                    category=category,
                    cursor=cursor,
                )

                if not preprints:
                    # No more results
                    break

                # Filter by query
                filtered = self._filter_preprints_by_query(preprints, query)
                all_preprints.extend(filtered)

                # Check if we have enough results
                if len(all_preprints) >= max_results:
                    break

                # Check if we got fewer than page_size (last page)
                if len(preprints) < self.config.page_size:
                    break

                # Next page
                cursor += self.config.page_size

            # Truncate to max_results
            all_preprints = all_preprints[:max_results]

            if not all_preprints:
                return f"No bioRxiv/medRxiv preprints found for query: {query}"

            # Convert to PublicationMetadata
            pub_metadata = [self._parse_preprint_to_metadata(p) for p in all_preprints]

            # Format with base class formatter
            formatted_results = self.format_search_results(pub_metadata, query)

            # Add preprint-specific tip
            formatted_results += "\n\n💡 **Tip**: Use `get_full_text()` to extract methods and software tools from preprints.\n"

            # Log the search
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="biorxiv_medrxiv_search",
                    parameters={
                        "query": query[:100],
                        "max_results": max_results,
                        "date_range": date_range,
                        "category": category,
                        "server": server or self.config.default_server,
                    },
                    description="bioRxiv/medRxiv preprint search",
                )

            return formatted_results

        except Exception as e:
            logger.exception(f"bioRxiv/medRxiv search error: {e}")
            return f"bioRxiv/medRxiv search error: {str(e)}"

    def extract_publication_metadata(
        self, identifier: str, **kwargs
    ) -> PublicationMetadata:
        """
        Extract standardized metadata from a preprint.

        Args:
            identifier: Preprint DOI (e.g., "10.1101/2024.01.01.123456")
            **kwargs: Additional parameters:
                - server: Optional server override ("biorxiv" | "medrxiv")

        Returns:
            PublicationMetadata: Standardized publication metadata

        Raises:
            BioRxivNotFoundError: Preprint not found
            ValueError: Invalid DOI format

        Examples:
            >>> metadata = provider.extract_publication_metadata("10.1101/2024.01.01.123456")
            >>> print(metadata.title)
        """
        try:
            # Validate DOI format
            if not self.validate_identifier(identifier):
                raise ValueError(
                    f"Invalid bioRxiv/medRxiv DOI format: {identifier}. "
                    f"Expected: 10.1101/YYYY.MM.DD.######"
                )

            # Extract server from kwargs
            server = kwargs.get("server")

            # Fetch preprint details
            preprints = self._fetch_details(doi=identifier, server=server)

            if not preprints:
                raise BioRxivNotFoundError(f"Preprint not found: {identifier}")

            # Parse first result
            preprint = preprints[0]
            metadata = self._parse_preprint_to_metadata(preprint)

            # Check if preprint has been published
            try:
                mapping = self.get_publication_mapping(identifier, server=server)
                if mapping and mapping.get("published_doi"):
                    # Add published journal info to keywords
                    metadata.keywords.append(
                        f"Published in {mapping['published_journal']}"
                    )
            except Exception as e:
                logger.debug(f"Could not fetch publication mapping: {e}")

            return metadata

        except BioRxivNotFoundError:
            # Re-raise not found errors
            raise
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            # Return minimal metadata on error
            return PublicationMetadata(
                uid=identifier,
                title=f"Error extracting metadata: {str(e)}",
                abstract=f"Could not retrieve preprint metadata for: {identifier}",
            )

    def find_datasets_from_publication(
        self,
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        **kwargs,
    ) -> str:
        """
        Find datasets referenced in preprint (best-effort regex extraction).

        TODO: Implement in Phase 4

        Args:
            identifier: Preprint DOI
            dataset_types: Types of datasets to search for
            **kwargs: Additional parameters

        Returns:
            str: Formatted list of discovered datasets
        """
        raise NotImplementedError("TODO: Implement in Phase 4")

    # ========================================================================
    # Provider-Specific Methods
    # ========================================================================

    def get_publication_mapping(
        self, doi: str, server: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Check if preprint has been published in a journal.

        Uses the /pubs endpoint to retrieve preprint→journal mapping.

        Args:
            doi: Preprint DOI (e.g., "10.1101/2024.01.01.123456")
            server: Optional server override ("biorxiv" | "medrxiv")

        Returns:
            Optional[Dict]: Publication mapping info with keys:
                - biorxiv_doi: Preprint DOI
                - published_doi: Published journal DOI (if published)
                - published_journal: Journal name (if published)
                - published_date: Publication date (if published)
                - preprint_title: Preprint title
                - preprint_date: Preprint date

            Returns None if no mapping found or preprint not published.

        Examples:
            >>> mapping = provider.get_publication_mapping("10.1101/2024.01.01.123456")
            >>> if mapping and mapping['published_doi']:
            >>>     print(f"Published in {mapping['published_journal']}")
        """
        try:
            server = server or self.config.default_server

            # Build URL: /pubs/[server]/[DOI]/na/json
            url = self._build_url("pubs", server=server, interval=doi, cursor="na")

            # Make API request
            content = self._make_api_request(
                url, f"fetch publication mapping ({server})"
            )

            # Parse JSON response
            response = json.loads(content.decode("utf-8"))

            # Extract collection
            collection = response.get("collection", [])

            if not collection:
                logger.debug(f"No publication mapping found for: {doi}")
                return None

            # Return first result
            mapping = collection[0]

            # Check if actually published
            if not mapping.get("published_doi"):
                logger.debug(f"Preprint not yet published: {doi}")
                return None

            return mapping

        except BioRxivNotFoundError:
            # Preprint doesn't exist or no mapping
            logger.debug(f"No publication mapping for {doi}")
            return None
        except Exception as e:
            logger.warning(f"Error fetching publication mapping for {doi}: {e}")
            return None

    def _fetch_jatsxml(self, jatsxml_url: str) -> str:
        """
        Fetch JATS XML from URL using proper HTTP headers.

        Uses requests.Session with User-Agent and Accept headers to avoid
        403 errors from bioRxiv/medRxiv servers. The API-provided jatsxml URL
        (including /early/ paths) is used as-is per the API contract.

        Args:
            jatsxml_url: URL to JATS XML file (from /details endpoint)

        Returns:
            str: Raw JATS XML content

        Raises:
            BioRxivJATSError: Failed to fetch JATS XML
        """
        try:
            logger.debug(f"Fetching JATS XML: {jatsxml_url[:100]}...")

            # Use content_session with browser-like headers for JATS XML URLs
            # (www.biorxiv.org content servers require browser headers to avoid 403)
            response = self.content_session.get(
                jatsxml_url,
                timeout=self.config.timeout,
                allow_redirects=True,
            )

            # Log redirects for debugging (bioRxiv often redirects to versioned URLs)
            if response.history:
                redirect_chain = " → ".join([r.url for r in response.history])
                logger.debug(f"Followed redirect: {redirect_chain} → {response.url}")

            response.raise_for_status()

            xml_text = response.text

            logger.debug(f"Successfully fetched JATS XML: {len(xml_text)} bytes")

            return xml_text

        except Exception as e:
            logger.error(f"Error fetching JATS XML: {e}")
            raise BioRxivJATSError(
                f"Failed to fetch JATS XML from {jatsxml_url}: {str(e)}"
            )

    def get_full_text(self, doi: str, server: Optional[str] = None) -> PMCFullText:
        """
        Fetch and parse JATS XML for full-text extraction.

        This method fetches structured JATS XML from bioRxiv/medRxiv and
        parses it using PMCProvider's JATS parsing logic (composition pattern).

        Process:
        1. Fetch preprint metadata from /details endpoint
        2. Extract JATS XML URL (jatsxml field)
        3. Fetch JATS XML from URL
        4. Parse using PMCProvider.parse_pmc_xml() (reuses 1000+ lines)
        5. Update metadata (source_type, doi, etc.)

        Args:
            doi: Preprint DOI (e.g., "10.1101/2024.01.01.123456")
            server: Optional server override ("biorxiv" | "medrxiv")

        Returns:
            PMCFullText: Parsed full-text with:
                - full_text: Complete markdown of all sections
                - methods_section: Extracted methods section
                - results_section: Extracted results section
                - tables: Parsed tables
                - software_tools: Detected software tools
                - github_repos: Extracted GitHub repositories
                - parameters: Extracted parameters

        Raises:
            BioRxivNotFoundError: Preprint not found
            BioRxivJATSError: JATS XML parsing or fetching failed

        Examples:
            >>> full_text = provider.get_full_text("10.1101/2024.01.01.123456")
            >>> print(f"Methods: {len(full_text.methods_section)} chars")
            >>> print(f"Software: {', '.join(full_text.software_tools)}")
            >>> print(f"Tables: {len(full_text.tables)}")
        """
        try:
            logger.debug(f"Extracting full text for preprint: {doi}")

            server = server or self.config.default_server

            # Step 1: Fetch preprint metadata from /details endpoint
            preprints = self._fetch_details(doi=doi, server=server)

            if not preprints:
                raise BioRxivNotFoundError(f"Preprint not found: {doi}")

            preprint = preprints[0]

            # Step 2: Extract JATS XML URL
            jatsxml_url = preprint.get("jatsxml")

            if not jatsxml_url:
                raise BioRxivJATSError(
                    f"No JATS XML available for preprint: {doi}. "
                    f"Preprint may be too old or JATS XML not generated yet."
                )

            # Step 3: Fetch JATS XML from URL
            xml_text = self._fetch_jatsxml(jatsxml_url)

            # Step 4: Parse JATS XML using PMCProvider (composition pattern)
            # This reuses PMCProvider's 1000+ lines of JATS parsing logic
            try:
                parsed = self._jats_parser.parse_pmc_xml(xml_text)
            except Exception as e:
                logger.error(f"PMCProvider JATS parsing failed: {e}")
                raise BioRxivJATSError(f"Failed to parse JATS XML for {doi}: {str(e)}")

            # Step 5: Update metadata (override PMC-specific fields)
            parsed.source_type = f"{server}_jatsxml"
            parsed.doi = doi

            # Update PMC ID and PMID (preprints don't have these)
            parsed.pmc_id = ""
            parsed.pmid = None

            # Update title and abstract from preprint metadata (more reliable)
            parsed.title = preprint.get("title", parsed.title)
            parsed.abstract = preprint.get("abstract", parsed.abstract)

            # Log provenance if DataManager available
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="extract_biorxiv_medrxiv_full_text",
                    parameters={"doi": doi, "server": server},
                    description=(
                        f"{server} full text extraction: {len(parsed.full_text)} chars, "
                        f"{len(parsed.software_tools)} tools, "
                        f"{len(parsed.tables)} tables"
                    ),
                )

            logger.debug(
                f"Successfully extracted full text: "
                f"{len(parsed.full_text)} chars, "
                f"{len(parsed.methods_section)} chars methods, "
                f"{len(parsed.software_tools)} tools, "
                f"{len(parsed.tables)} tables"
            )

            return parsed

        except (BioRxivNotFoundError, BioRxivJATSError):
            # Re-raise specific errors
            raise
        except Exception as e:
            logger.exception(f"Error extracting full text for {doi}: {e}")
            raise BioRxivJATSError(f"Full text extraction failed for {doi}: {str(e)}")

    def get_category_statistics(
        self, server: Optional[str] = None, date_range: Optional[str] = None
    ) -> Dict:
        """
        Get content statistics by category from /sum endpoint.

        Args:
            server: Optional server override ("biorxiv" | "medrxiv")
            date_range: Optional date range ("m" for monthly, "y" for yearly)

        Returns:
            Dict: Category statistics with keys:
                - messages: API response messages
                - collection: List of stats by month/year

        Examples:
            >>> stats = provider.get_category_statistics(server="biorxiv", date_range="m")
            >>> print(stats['collection'][0])  # Monthly stats
        """
        try:
            server = server or self.config.default_server
            interval = date_range or "m"  # Monthly by default

            # Build URL: /sum/[interval]/json
            url = self._build_url("sum", interval=interval, format="json")

            # Make API request
            content = self._make_api_request(url, "fetch content statistics")

            # Parse JSON response
            response = json.loads(content.decode("utf-8"))

            return response

        except Exception as e:
            logger.error(f"Error fetching category statistics: {e}")
            return {"error": str(e)}

    def get_usage_statistics(self, doi: str, server: Optional[str] = None) -> Dict:
        """
        Get download/view statistics for a preprint from /usage endpoint.

        Note: The /usage endpoint provides aggregate statistics, not per-DOI stats.
        For per-article statistics, use the preprint's webpage API.

        Args:
            doi: Preprint DOI (unused - kept for API compatibility)
            server: Optional server override ("biorxiv" | "medrxiv")

        Returns:
            Dict: Usage statistics with keys:
                - messages: API response messages
                - collection: List of usage stats by month

        Examples:
            >>> stats = provider.get_usage_statistics("10.1101/2024.01.01.123456")
            >>> print(stats['collection'][0])  # Monthly usage
        """
        try:
            server = server or self.config.default_server

            # Build URL: /usage/m/json (monthly stats)
            url = self._build_url("usage", interval="m", format="json")

            # Make API request
            content = self._make_api_request(url, "fetch usage statistics")

            # Parse JSON response
            response = json.loads(content.decode("utf-8"))

            return response

        except Exception as e:
            logger.error(f"Error fetching usage statistics: {e}")
            return {"error": str(e)}

    def search_by_funder(
        self,
        funder_ror: str,
        date_range: str,
        server: Optional[str] = None,
        category: Optional[str] = None,
        max_results: int = 100,
    ) -> List[Dict]:
        """
        Search preprints by funding agency using ROR ID.

        Args:
            funder_ror: 9-character ROR ID suffix (e.g., "02mhbdp94" for EU)
            date_range: Date range ("YYYY-MM-DD/YYYY-MM-DD")
            server: Optional server override ("biorxiv" | "medrxiv")
            category: Optional category filter
            max_results: Maximum number of results (default: 100)

        Returns:
            List[Dict]: List of preprint records

        Examples:
            >>> # Search EU-funded preprints
            >>> results = provider.search_by_funder(
            ...     funder_ror="02mhbdp94",
            ...     date_range="2024-01-01/2024-01-31"
            ... )
        """
        try:
            server = server or self.config.default_server

            # Build URL with optional category filter
            query_params = {}
            if category:
                query_params["category"] = category

            url = self._build_url(
                "funder",
                server=server,
                interval=date_range,
                funder_ror=funder_ror,
                cursor=0,
                format="json",
                query_params=query_params if query_params else None,
            )

            # Make API request
            content = self._make_api_request(url, "search by funder")

            # Parse JSON response
            response = json.loads(content.decode("utf-8"))

            # Extract collection
            collection = response.get("collection", [])

            # Truncate to max_results
            return collection[:max_results]

        except Exception as e:
            logger.error(f"Error searching by funder: {e}")
            return []

    def search_by_publisher(
        self,
        publisher_prefix: str,
        date_range: str,
        max_results: int = 100,
    ) -> List[Dict]:
        """
        Search preprints by publisher DOI prefix.

        Args:
            publisher_prefix: Publisher DOI prefix (e.g., "10.15252" for EMBO)
            date_range: Date range ("YYYY-MM-DD/YYYY-MM-DD")
            max_results: Maximum number of results (default: 100)

        Returns:
            List[Dict]: List of preprint records that were published by this publisher

        Examples:
            >>> # Search preprints published by EMBO
            >>> results = provider.search_by_publisher(
            ...     publisher_prefix="10.15252",
            ...     date_range="2024-01-01/2024-01-31"
            ... )
        """
        try:
            # Build URL
            url = self._build_url(
                "publisher",
                publisher_prefix=publisher_prefix,
                interval=date_range,
                cursor=0,
            )

            # Make API request
            content = self._make_api_request(url, "search by publisher")

            # Parse JSON response
            response = json.loads(content.decode("utf-8"))

            # Extract collection
            collection = response.get("collection", [])

            # Truncate to max_results
            return collection[:max_results]

        except Exception as e:
            logger.error(f"Error searching by publisher: {e}")
            return []

    def find_datasets_from_publication(
        self,
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        **kwargs,
    ) -> str:
        """
        Find datasets referenced in preprint (best-effort regex extraction).

        This method extracts dataset accessions from preprint full-text using
        regex patterns. Note that preprints lack ELink-style authoritative linking,
        so this is best-effort extraction only.

        Args:
            identifier: Preprint DOI
            dataset_types: Optional types of datasets to search for
            **kwargs: Additional parameters (server)

        Returns:
            str: Formatted list of discovered datasets

        Examples:
            >>> datasets = provider.find_datasets_from_publication("10.1101/2024.01.01.123456")
        """
        try:
            logger.debug(f"Finding datasets from preprint: {identifier}")

            # Extract server from kwargs
            server = kwargs.get("server")

            # Get full text (includes abstract + JATS XML if available)
            try:
                full_text_obj = self.get_full_text(identifier, server=server)
                search_text = full_text_obj.full_text
            except BioRxivJATSError:
                # Fallback to abstract only
                logger.warning(
                    f"JATS XML not available, searching abstract only for: {identifier}"
                )
                metadata = self.extract_publication_metadata(identifier, server=server)
                search_text = metadata.abstract

            # Extract dataset accessions using regex patterns
            datasets = self._extract_dataset_accessions(search_text)

            # Filter by requested dataset types if specified
            if dataset_types:
                type_map = {
                    DatasetType.GEO: "GEO",
                    DatasetType.SRA: "SRA",
                    DatasetType.ARRAYEXPRESS: "ArrayExpress",
                    DatasetType.ENA: "ENA",
                }
                filtered = {}
                for dtype in dataset_types:
                    if dtype in type_map and type_map[dtype] in datasets:
                        filtered[type_map[dtype]] = datasets[type_map[dtype]]
                datasets = filtered

            # Format response
            response = f"## Dataset Discovery from Preprint\n\n"
            response += f"**Preprint**: {identifier}\n"
            response += f"**Method**: Regex extraction from full-text\n\n"

            total_datasets = sum(len(v) for v in datasets.values())

            if total_datasets == 0:
                response += "**No datasets found**. Note: Dataset extraction from preprints is best-effort only.\n"
            else:
                response += f"**Found {total_datasets} dataset reference(s)**:\n\n"

                for db_type, accessions in datasets.items():
                    if accessions:
                        response += f"### {db_type}\n"
                        for acc in accessions:
                            response += f"- {acc}\n"
                        response += "\n"

            response += "\n💡 **Note**: Preprints lack authoritative dataset linking. These results are from regex pattern matching and may include false positives.\n"

            # Log the search
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="biorxiv_medrxiv_find_datasets",
                    parameters={
                        "identifier": identifier,
                        "dataset_types": (
                            [dt.value for dt in dataset_types]
                            if dataset_types
                            else None
                        ),
                    },
                    description="Dataset discovery from bioRxiv/medRxiv preprint",
                )

            return response

        except Exception as e:
            logger.exception(f"Error finding datasets: {e}")
            return f"Error finding datasets from preprint: {str(e)}"

    def _extract_dataset_accessions(self, text: str) -> Dict[str, List[str]]:
        """
        Extract dataset accessions from text.

        Uses centralized AccessionResolver for pattern matching.

        Args:
            text: Text to search (full-text or abstract)

        Returns:
            Dict[str, List[str]]: Dictionary mapping database type to accessions
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

        # Deduplicate
        for key in accessions:
            accessions[key] = sorted(list(set(accessions[key])))

        return accessions
