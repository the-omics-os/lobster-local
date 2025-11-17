"""
Webpage Provider for webpage-first content extraction.

Created during Phase 1 refactoring (2025-01-02) to enable webpage-first extraction
strategy before falling back to PDF parsing.

This provider handles publisher webpages (Nature, Science, etc.) that are directly
accessible without PDF download. Examples:
- https://www.nature.com/articles/s41586-025-09686-5
- https://www.science.org/doi/10.1126/science.abcd1234
- https://www.cell.com/cell/fulltext/S0092-8674(21)00000-0
"""

from pathlib import Path
from typing import Any, Dict, Optional

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.docling_service import DoclingError, DoclingService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class WebpageProviderError(Exception):
    """Base exception for WebpageProvider errors."""

    pass


class WebpageExtractionError(WebpageProviderError):
    """Webpage extraction failed."""

    pass


class WebpageProvider:
    """
    Webpage-first content extraction provider.

    This provider extracts content from publisher webpages using Docling's
    webpage parsing capabilities. Part of the two-tier access strategy where
    webpages are tried before PDF fallback.

    Architecture:
    - Delegates to DoclingService for actual extraction
    - Detects non-PDF URLs (not ending in .pdf)
    - Returns clean markdown with tables and formulas
    - Graceful error handling with detailed logging

    Performance:
    - Typical webpage extraction: 2-5 seconds
    - Cache hit (via DoclingService): <100ms

    Supported Sites:
    - Nature journals (nature.com)
    - Science journals (science.org)
    - Cell Press (cell.com)
    - PLOS (plos.org)
    - And many other publishers via Docling

    Examples:
        >>> provider = WebpageProvider()
        >>>
        >>> # Extract from Nature article
        >>> content = provider.extract(
        ...     "https://www.nature.com/articles/s41586-025-09686-5"
        ... )
        >>> print(f"Extracted {len(content)} characters")
        >>>
        >>> # Check if URL can be handled
        >>> if provider.can_handle(url):
        ...     content = provider.extract(url)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        data_manager: Optional[DataManagerV2] = None,
    ):
        """
        Initialize WebpageProvider with optional caching.

        Args:
            cache_dir: Cache directory for parsed documents
                      (default: .lobster_workspace/literature_cache/parsed_docs)
            data_manager: Optional DataManagerV2 for provenance tracking
        """
        self.data_manager = data_manager

        # Initialize DoclingService for webpage extraction
        self.docling_service = DoclingService(
            cache_dir=cache_dir, data_manager=data_manager
        )

        logger.debug(
            f"Initialized WebpageProvider with Docling "
            f"(available: {self.docling_service.is_available()})"
        )

    @property
    def source(self) -> str:
        """Return webpage as the content source."""
        return "webpage"

    @property
    def supported_dataset_types(self) -> list:
        """
        Return list of dataset types supported by WebpageProvider.

        WebpageProvider doesn't host or discover datasets - it extracts
        content from publisher webpages.

        Returns:
            list: Empty list (no dataset support)
        """
        return []

    @property
    def priority(self) -> int:
        """
        Return provider priority for capability-based routing.

        WebpageProvider has medium priority (50) as a fallback after PMC
        but before PDF extraction:
        - Faster than PDF (2-5s vs. 3-8s)
        - Handles publisher HTML layouts
        - Falls back when PMC unavailable (60-70% of papers)

        Priority cascade: PMC (10) → Webpage (50) → PDF (100)

        Returns:
            int: Priority 50 (medium priority)
        """
        return 50

    def get_supported_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities supported by WebpageProvider.

        WebpageProvider extracts full-text content from publisher webpages
        using Docling's HTML parsing. It can also handle PDF extraction
        via delegation to DoclingService (composition pattern).

        Supported capabilities:
        - QUERY_CAPABILITIES: Dynamic capability discovery
        - GET_FULL_CONTENT: HTML webpage extraction (2-5s)
        - EXTRACT_METHODS: Heuristic methods extraction from HTML
        - EXTRACT_PDF: Delegates to DoclingService for PDF processing

        Not supported:
        - SEARCH_LITERATURE: No search (use PubMedProvider)
        - DISCOVER_DATASETS: No dataset discovery (use GEOProvider)
        - FIND_LINKED_DATASETS: No dataset linking
        - EXTRACT_METADATA: No metadata extraction
        - VALIDATE_METADATA: No metadata validation
        - GET_ABSTRACT: No abstract retrieval (use PMCProvider)
        - INTEGRATE_MULTI_OMICS: No multi-omics integration

        Note: EXTRACT_PDF uses composition pattern - WebpageProvider
        delegates to DoclingService internally (lines 91-93).

        Returns:
            Dict[str, bool]: Capability support mapping
        """
        from lobster.tools.providers.base_provider import ProviderCapability

        return {
            ProviderCapability.SEARCH_LITERATURE: False,
            ProviderCapability.DISCOVER_DATASETS: False,
            ProviderCapability.FIND_LINKED_DATASETS: False,
            ProviderCapability.EXTRACT_METADATA: False,
            ProviderCapability.VALIDATE_METADATA: False,
            ProviderCapability.QUERY_CAPABILITIES: True,
            ProviderCapability.GET_ABSTRACT: False,
            ProviderCapability.GET_FULL_CONTENT: True,
            ProviderCapability.EXTRACT_METHODS: True,
            ProviderCapability.EXTRACT_PDF: True,  # Via DoclingService delegation
            ProviderCapability.INTEGRATE_MULTI_OMICS: False,
        }

    def can_handle(self, url: str) -> bool:
        """
        Check if this provider can handle the URL.

        Strategy:
        - Accept URLs that don't end in .pdf
        - Assume they are publisher webpages
        - Let Docling handle the actual webpage detection

        Args:
            url: URL to check

        Returns:
            True if URL is not a PDF, False otherwise

        Examples:
            >>> provider = WebpageProvider()
            >>> provider.can_handle("https://www.nature.com/articles/...")
            True
            >>> provider.can_handle("https://arxiv.org/pdf/2408.09869.pdf")
            False
        """
        if not url or not isinstance(url, str):
            return False

        url_lower = url.lower().strip()

        # Don't handle direct PDF URLs
        if url_lower.endswith(".pdf"):
            return False

        # Handle everything else (assume webpage)
        return True

    def extract(
        self,
        url: str,
        keywords: Optional[list] = None,
        max_paragraphs: int = 100,
        max_retries: int = 2,
    ) -> str:
        """
        Extract content from webpage as clean markdown.

        This method uses Docling to parse publisher webpages and extract
        structured content including text, tables, and formulas.

        Args:
            url: Webpage URL to extract
            keywords: Optional section keywords (for Methods detection)
            max_paragraphs: Maximum paragraphs to extract (default: 100)
            max_retries: Maximum retry attempts (default: 2)

        Returns:
            Clean markdown string with:
                - Main text content
                - Tables (formatted as markdown)
                - Formulas (LaTeX if available)
                - Section structure preserved
                - Images filtered (base64 encodings removed)

        Raises:
            WebpageExtractionError: If extraction fails after retries
            DoclingError: If Docling is not available

        Performance:
            - First extraction: 2-5 seconds (Docling parsing)
            - Cached extraction: <100ms (via DoclingService cache)

        Examples:
            >>> provider = WebpageProvider()
            >>>
            >>> # Extract full article
            >>> markdown = provider.extract(
            ...     "https://www.nature.com/articles/s41586-025-09686-5"
            ... )
            >>> print(f"Content length: {len(markdown)} chars")
            >>>
            >>> # Extract Methods section specifically
            >>> methods = provider.extract(
            ...     url,
            ...     keywords=["method", "material", "procedure"]
            ... )
        """
        try:
            logger.info(f"Extracting webpage content from: {url}")

            # Check if Docling is available
            if not self.docling_service.is_available():
                raise DoclingError(
                    "Docling not available for webpage extraction. "
                    "Install with: pip install docling docling-core"
                )

            # Validate URL format
            if not self.can_handle(url):
                raise WebpageExtractionError(
                    f"URL appears to be a PDF, not a webpage: {url}. "
                    "Use PDFProvider for PDF extraction."
                )

            # Use DoclingService to extract (handles caching, retry logic)
            result = self.docling_service.extract_methods_section(
                source=url,
                keywords=keywords,
                max_paragraphs=max_paragraphs,
                max_retries=max_retries,
            )

            # Get markdown content (already has images filtered)
            markdown = result.get("methods_markdown", "")

            if not markdown:
                raise WebpageExtractionError(
                    f"No content extracted from webpage: {url}. "
                    "Page may be restricted or formatted unexpectedly."
                )

            # Log provenance if DataManager available
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="extract_webpage",
                    parameters={
                        "url": url[:100],
                        "max_paragraphs": max_paragraphs,
                    },
                    description=f"Webpage extraction: {len(markdown)} chars, "
                    f"{len(result.get('tables', []))} tables, "
                    f"{len(result.get('formulas', []))} formulas",
                )

            logger.info(
                f"Successfully extracted webpage: {len(markdown)} chars, "
                f"{len(result.get('tables', []))} tables"
            )

            return markdown

        except DoclingError:
            # Re-raise Docling errors (caller may want to fallback)
            raise
        except WebpageExtractionError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            # Wrap other errors as extraction errors
            logger.exception(f"Error extracting webpage {url}: {e}")
            raise WebpageExtractionError(
                f"Webpage extraction failed for {url}: {str(e)}. "
                "Page may be restricted or require authentication."
            )

    def extract_with_full_metadata(
        self,
        url: str,
        keywords: Optional[list] = None,
        max_paragraphs: int = 100,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Extract webpage content with full metadata.

        This method returns the full extraction result including tables,
        formulas, software mentions, and provenance information.

        Args:
            url: Webpage URL to extract
            keywords: Optional section keywords
            max_paragraphs: Maximum paragraphs to extract
            max_retries: Maximum retry attempts

        Returns:
            Dictionary with:
                'methods_text': str - Plain text content
                'methods_markdown': str - Markdown with tables
                'sections': List[Dict] - Section hierarchy
                'tables': List[DataFrame] - Extracted tables
                'formulas': List[str] - Mathematical formulas
                'software_mentioned': List[str] - Detected software
                'provenance': Dict - Extraction metadata

        Raises:
            WebpageExtractionError: If extraction fails

        Examples:
            >>> provider = WebpageProvider()
            >>> result = provider.extract_with_full_metadata(url)
            >>> print(f"Found {len(result['tables'])} tables")
            >>> print(f"Software: {', '.join(result['software_mentioned'])}")
        """
        try:
            logger.info(f"Extracting webpage with metadata from: {url[:80]}...")

            # Check Docling availability
            if not self.docling_service.is_available():
                raise DoclingError(
                    "Docling not available. Install with: pip install docling docling-core"
                )

            # Use DoclingService to extract full result
            result = self.docling_service.extract_methods_section(
                source=url,
                keywords=keywords,
                max_paragraphs=max_paragraphs,
                max_retries=max_retries,
            )

            # Log provenance
            if self.data_manager:
                self.data_manager.log_tool_usage(
                    tool_name="extract_webpage_with_metadata",
                    parameters={"url": url[:100]},
                    description=f"Full webpage extraction: {len(result['methods_text'])} chars",
                )

            logger.info(
                f"Extracted webpage with metadata: "
                f"{len(result['methods_text'])} chars, "
                f"{len(result['tables'])} tables, "
                f"{len(result['formulas'])} formulas"
            )

            return result

        except Exception as e:
            logger.exception(f"Error extracting webpage with metadata: {e}")
            raise WebpageExtractionError(f"Webpage extraction failed: {str(e)}")

    def is_available(self) -> bool:
        """
        Check if the provider is available for use.

        Returns:
            True if Docling is available for webpage extraction
        """
        return self.docling_service.is_available()

    def get_supported_features(self) -> dict:
        """
        Get supported features for this provider.

        Returns:
            Dictionary of feature flags
        """
        return {
            "webpage_extraction": True,
            "pdf_extraction": False,
            "table_extraction": True,
            "formula_extraction": True,
            "section_detection": True,
            "image_filtering": True,
            "caching": True,
            "retry_logic": True,
            "markdown_export": True,
        }
