"""
Abstract Provider for fast abstract retrieval without PDF download.

Created during Phase 1 refactoring (2025-01-02) to enable two-tier content access:
- Tier 1: Quick abstract (this provider) - <500ms
- Tier 2: Full content (WebpageProvider/PDFProvider) - 2-8 seconds

This provider leverages NCBI E-utilities for fast abstract retrieval without
downloading or parsing full PDF files.
"""

from typing import Optional

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.base_provider import PublicationMetadata, PublicationSource
from lobster.tools.providers.pubmed_provider import PubMedProvider, PubMedProviderConfig
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class AbstractProviderError(Exception):
    """Base exception for AbstractProvider errors."""

    pass


class IdentifierNotFoundError(AbstractProviderError):
    """Invalid or non-existent publication identifier."""

    pass


class NCBIAPIError(AbstractProviderError):
    """NCBI E-utilities API error."""

    pass


class AbstractProvider:
    """
    Fast abstract retrieval without PDF download.

    This provider uses NCBI E-utilities to retrieve publication abstracts
    quickly without downloading or parsing full PDF files. Part of the
    two-tier access strategy introduced in Phase 1.

    Performance:
    - Cache miss: 200-500ms (NCBI API call)
    - Cache hit: <50ms (if using DataManager caching - Phase 2)

    Supported Identifiers:
    - PMID: "PMID:12345678" or "12345678"
    - DOI: "10.1038/s41586-021-12345-6"

    Examples:
        >>> provider = AbstractProvider()
        >>> metadata = provider.get_abstract("PMID:35042229")
        >>> print(metadata.title)
        >>> print(metadata.abstract[:200])

        >>> # DOI support
        >>> metadata = provider.get_abstract("10.1038/s41586-021-03852-1")
        >>> print(f"Authors: {', '.join(metadata.authors[:3])}")
    """

    def __init__(
        self,
        data_manager: Optional[DataManagerV2] = None,
        config: Optional[PubMedProviderConfig] = None,
    ):
        """
        Initialize AbstractProvider with optional configuration.

        Args:
            data_manager: Optional DataManagerV2 for provenance tracking
            config: Optional PubMedProviderConfig for NCBI settings
        """
        self.data_manager = data_manager

        # Use custom config or default
        if config is None:
            config = PubMedProviderConfig()

        # Initialize PubMedProvider (reuse existing NCBI integration)
        self.pubmed_provider = PubMedProvider(data_manager=data_manager, config=config)

        logger.debug("Initialized AbstractProvider with NCBI E-utilities")

    @property
    def source(self) -> str:
        """Return abstract provider as the source."""
        return "abstract"

    @property
    def supported_dataset_types(self) -> list:
        """
        Return list of dataset types supported by AbstractProvider.

        AbstractProvider is a single-purpose fast-path provider for abstract
        retrieval only. It doesn't handle datasets.

        Returns:
            list: Empty list (no dataset support)
        """
        return []

    @property
    def priority(self) -> int:
        """
        Return provider priority for capability-based routing.

        AbstractProvider has high priority (10) as a specialized fast-path
        provider for abstract retrieval:
        - 200-500ms cache miss (NCBI API)
        - <50ms cache hit
        - Single-purpose specialist

        Same priority tier as PubMed due to specialization and performance.

        Returns:
            int: Priority 10 (high priority)
        """
        return 10

    def get_supported_capabilities(self) -> dict:
        """
        Return capabilities supported by AbstractProvider.

        AbstractProvider is a single-purpose provider specialized for
        fast abstract retrieval without PDF download. It's part of the
        two-tier access strategy (Tier 1: quick abstract, Tier 2: full content).

        Supported capabilities:
        - QUERY_CAPABILITIES: Dynamic capability discovery
        - GET_ABSTRACT: Core capability - fast abstract retrieval (200-500ms)

        Not supported (all other capabilities):
        - SEARCH_LITERATURE: No search (use PubMedProvider)
        - DISCOVER_DATASETS: No dataset discovery (use GEOProvider)
        - FIND_LINKED_DATASETS: No dataset linking
        - EXTRACT_METADATA: No metadata extraction
        - VALIDATE_METADATA: No metadata validation
        - GET_FULL_CONTENT: No full-text (use PMCProvider/WebpageProvider)
        - EXTRACT_METHODS: No methods extraction
        - EXTRACT_PDF: No PDF processing
        - INTEGRATE_MULTI_OMICS: No multi-omics integration

        Returns:
            dict: Capability support mapping
        """
        from lobster.tools.providers.base_provider import ProviderCapability

        return {
            ProviderCapability.SEARCH_LITERATURE: False,
            ProviderCapability.DISCOVER_DATASETS: False,
            ProviderCapability.FIND_LINKED_DATASETS: False,
            ProviderCapability.EXTRACT_METADATA: False,
            ProviderCapability.VALIDATE_METADATA: False,
            ProviderCapability.QUERY_CAPABILITIES: True,
            ProviderCapability.GET_ABSTRACT: True,  # CORE capability
            ProviderCapability.GET_FULL_CONTENT: False,
            ProviderCapability.EXTRACT_METHODS: False,
            ProviderCapability.EXTRACT_PDF: False,
            ProviderCapability.INTEGRATE_MULTI_OMICS: False,
        }

    def get_abstract(self, identifier: str) -> PublicationMetadata:
        """
        Retrieve publication abstract without downloading full PDF.

        This is the primary method for two-tier access strategy (Tier 1: fast path).

        Args:
            identifier: PMID, DOI, or publication ID
                       Examples: "PMID:12345678", "12345678", "10.1038/..."

        Returns:
            PublicationMetadata with:
                - uid: Unique identifier
                - title: Publication title
                - abstract: Full abstract text
                - authors: List of authors
                - journal: Journal name
                - published: Publication date
                - doi: DOI if available
                - pmid: PMID if available
                - keywords: Keywords/MeSH terms

        Raises:
            IdentifierNotFoundError: Invalid or non-existent publication ID
            NCBIAPIError: NCBI service unavailable or error

        Performance:
            - Typical response: 200-500ms (NCBI API call)
            - No PDF download or parsing required

        Examples:
            >>> provider = AbstractProvider()
            >>>
            >>> # By PMID
            >>> metadata = provider.get_abstract("PMID:35042229")
            >>> print(metadata.title)
            'Single-cell eQTL models reveal dynamic T cell state...'
            >>>
            >>> # By DOI
            >>> metadata = provider.get_abstract("10.1038/s41586-021-03852-1")
            >>> print(metadata.abstract[:100])
            'The genetic basis of gene expression variation...'
        """
        try:
            logger.info(f"Retrieving abstract for: {identifier}")

            # Validate identifier format
            if not self._validate_identifier(identifier):
                raise IdentifierNotFoundError(
                    f"Invalid identifier format: {identifier}. "
                    "Expected PMID (e.g., 'PMID:12345678') or DOI (e.g., '10.1038/...')"
                )

            # Use PubMedProvider to extract metadata (includes abstract)
            metadata = self.pubmed_provider.extract_publication_metadata(identifier)

            # Verify abstract was retrieved
            if not metadata.abstract or metadata.abstract.startswith("Error"):
                raise IdentifierNotFoundError(
                    f"Could not retrieve abstract for: {identifier}. "
                    "Publication may not exist in PubMed or may be restricted."
                )

            # Log provenance if DataManager available
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="get_abstract",
                    parameters={"identifier": identifier},
                    description=f"Abstract retrieval: {len(metadata.abstract)} chars",
                )

            logger.info(
                f"Successfully retrieved abstract: {len(metadata.abstract)} chars, "
                f"{len(metadata.authors)} authors"
            )

            return metadata

        except IdentifierNotFoundError:
            # Re-raise our custom errors
            raise
        except ValueError as e:
            # Catch PubMedProvider validation errors
            raise IdentifierNotFoundError(str(e))
        except Exception as e:
            # Wrap other errors as NCBI API errors
            logger.exception(f"Error retrieving abstract for {identifier}: {e}")
            raise NCBIAPIError(
                f"NCBI API error for {identifier}: {str(e)}. "
                "Service may be temporarily unavailable."
            )

    def _validate_identifier(self, identifier: str) -> bool:
        """
        Validate publication identifier format.

        Supported formats:
        - PMID: "PMID:12345678", "12345678" (numeric)
        - DOI: "10.1038/...", "10.1101/..." (starts with 10.)

        Args:
            identifier: Publication identifier to validate

        Returns:
            True if valid format, False otherwise
        """
        if not identifier or not isinstance(identifier, str):
            return False

        identifier = identifier.strip()

        # Check PMID format
        if identifier.upper().startswith("PMID:"):
            pmid = identifier[5:].strip()
            return pmid.isdigit()

        # Check if it's a numeric PMID without prefix
        if identifier.isdigit():
            return True

        # Check DOI format (starts with 10.)
        if identifier.startswith("10."):
            return True

        return False

    def get_source(self) -> PublicationSource:
        """
        Get the publication source for this provider.

        Returns:
            PublicationSource.PUBMED
        """
        return PublicationSource.PUBMED

    def is_available(self) -> bool:
        """
        Check if the provider is available for use.

        Returns:
            True if NCBI E-utilities are accessible
        """
        # AbstractProvider delegates to PubMedProvider which has robust
        # retry logic and error handling, so it's always "available"
        return True

    def get_supported_features(self) -> dict:
        """
        Get supported features for this provider.

        Returns:
            Dictionary of feature flags
        """
        return {
            "quick_abstract": True,
            "full_text_access": False,
            "pdf_download": False,
            "doi_support": True,
            "pmid_support": True,
            "metadata_extraction": True,
            "author_extraction": True,
            "keyword_extraction": True,
        }
