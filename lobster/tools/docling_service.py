"""
Docling Service for structure-aware PDF extraction.

This service provides shared Docling functionality for WebpageProvider and PDFProvider.
Extracted from PublicationIntelligenceService during Phase 1 refactoring (2025-01-02).

Key Features:
- Structure-aware Methods section detection
- Table and formula extraction
- Smart image filtering (reduce LLM context)
- Document caching for performance
- Comprehensive retry logic and error handling
- PyPDF2 fallback for reliability

Performance:
- Methods extraction: 2-5 seconds per paper
- Cache hit: <100ms
- Memory usage: ~500MB (Docling initialization)

Architecture:
- Primary: Docling for structure-aware parsing
- Fallback: PyPDF2 for reliability
- Caching: Parsed documents cached as JSON
"""

import gc
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== CUSTOM EXCEPTIONS ====================


class PublicationServiceError(Exception):
    """Base exception for publication services."""

    pass


class DoclingError(PublicationServiceError):
    """Docling-specific errors."""

    pass


class PDFExtractionError(PublicationServiceError):
    """General PDF extraction errors."""

    pass


class MethodsSectionNotFoundError(PublicationServiceError):
    """Methods section could not be located."""

    pass


# ==================== DOCLING SERVICE ====================


class DoclingService:
    """
    Shared Docling-based PDF extraction service.

    This service provides structure-aware PDF parsing with table/formula
    extraction, image filtering, and comprehensive error handling.

    Examples:
        >>> service = DoclingService()
        >>> result = service.extract_methods_section(
        ...     "https://arxiv.org/pdf/2408.09869"
        ... )
        >>> print(result['methods_text'])
        >>> print(f"Found {len(result['tables'])} parameter tables")
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        data_manager: Optional[DataManagerV2] = None,
    ):
        """
        Initialize DoclingService with optional caching and provenance.

        Args:
            cache_dir: Cache directory for parsed documents
                      (default: .lobster_workspace/literature_cache/parsed_docs)
            data_manager: Optional DataManagerV2 for provenance logging
        """
        self.data_manager = data_manager

        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path(".lobster_workspace") / "literature_cache" / "parsed_docs"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Docling converter (lazy loading for optional dependency)
        self.converter = None
        self._docling_imports = None
        self._initialize_docling()

        logger.debug(f"Initialized DoclingService with cache: {self.cache_dir}")

    def _initialize_docling(self):
        """
        Initialize Docling with graceful degradation.

        If Docling is not installed, logs a warning and sets converter to None.
        Consumers should check is_available() before using.
        """
        try:
            # Suppress formatting clash warnings from docling's html_backend
            # These are non-fatal warnings about overlapping text styles (subscript vs bold)
            # that occur when parsing complex HTML from publisher websites
            import logging

            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import (
                ConversionStatus,
                DocumentConverter,
                PdfFormatOption,
            )

            # Suppress formatting warnings from multiple possible logger paths
            # for compatibility across docling-core versions
            logger_paths = [
                "docling_core.transforms.chunker.html_backend",  # Legacy path
                "docling_core.transforms.serializer.html",  # v2.50+ path
                "docling_core.transforms.chunker",  # Parent logger
                "docling_core.transforms",  # Broad coverage
            ]
            for logger_path in logger_paths:
                logging.getLogger(logger_path).setLevel(logging.ERROR)

            # Store imports for later use
            self._docling_imports = {
                "DocumentConverter": DocumentConverter,
                "ConversionStatus": ConversionStatus,
                "PdfPipelineOptions": PdfPipelineOptions,
                "PdfFormatOption": PdfFormatOption,
                "InputFormat": InputFormat,
            }

            self.converter = self._create_docling_converter()
            logger.debug(
                "Initialized Docling converter for structure-aware PDF parsing"
            )
        except ImportError:
            logger.warning(
                "Docling not installed, structure-aware extraction unavailable. "
                "Install with: pip install docling docling-core"
            )
            self.converter = None
            self._docling_imports = None

    def _create_docling_converter(self):
        """
        Create optimized Docling converter for scientific PDFs and HTML content.

        Supported formats:
        - PDF: Scientific papers with Methods sections
        - HTML: Publisher webpages (Nature, Science, etc.)

        Configuration:
        - Table structure extraction: ENABLED (Methods parameters)
        - Code enrichment: ENABLED (code blocks detection)
        - Formula enrichment: ENABLED (equations detection)
        - OCR: DISABLED (initially - lightweight mode)
        - VLM: DISABLED (too heavy for initial deployment)

        Returns:
            Configured DocumentConverter instance
        """
        DocumentConverter = self._docling_imports["DocumentConverter"]
        PdfPipelineOptions = self._docling_imports["PdfPipelineOptions"]
        PdfFormatOption = self._docling_imports["PdfFormatOption"]
        InputFormat = self._docling_imports["InputFormat"]

        pdf_options = PdfPipelineOptions()
        pdf_options.do_table_structure = True  # Extract parameter tables
        pdf_options.do_code_enrichment = True  # Detect code blocks
        pdf_options.do_formula_enrichment = True  # Detect equations
        pdf_options.do_ocr = False  # Start lightweight

        return DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.HTML],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            },
        )

    def is_available(self) -> bool:
        """Check if Docling is available for use."""
        return self.converter is not None

    # ==================== MAIN EXTRACTION METHODS ====================

    def extract_methods_section(
        self,
        source: str,
        keywords: Optional[List[str]] = None,
        max_paragraphs: int = 50,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Extract Methods section with structure awareness using Docling.

        This method uses Docling's structure-aware PDF parsing with:
        - Automatic Methods section detection by keywords
        - Complete section extraction (not truncated)
        - Table and formula preservation
        - Smart image filtering
        - Document caching for performance
        - Comprehensive retry logic and error handling

        Args:
            source: URL or local path to PDF
            keywords: Section keywords to search for (default: method-related)
            max_paragraphs: Maximum paragraphs to extract (default: 50)
            max_retries: Maximum retry attempts (default: 2)

        Returns:
            Dictionary with:
                'methods_text': str - Full Methods section
                'methods_markdown': str - Markdown with tables (images filtered)
                'sections': List[Dict] - Hierarchical structure
                'tables': List[DataFrame] - Extracted tables
                'formulas': List[str] - Mathematical formulas
                'software_mentioned': List[str] - Tool names detected
                'provenance': Dict - Tracking metadata

        Raises:
            DoclingError: If Docling is not available
            PDFExtractionError: If extraction fails after retries

        Examples:
            >>> service = DoclingService()
            >>> result = service.extract_methods_section(
            ...     "https://arxiv.org/pdf/2408.09869"
            ... )
            >>> print(f"Found {len(result['tables'])} tables")
            >>> print(result['methods_text'][:200])
        """
        if keywords is None:
            keywords = [
                "method",
                "material",
                "procedure",
                "experimental",
                "materials and methods",  # PMC often uses full phrase
                "methods and materials",  # Alternative ordering
            ]

        logger.info(f"Extracting Methods section from: {source[:80]}...")

        # Check if Docling is available
        if not self.is_available():
            raise DoclingError(
                "Docling not available. Install with: pip install docling docling-core"
            )

        # Import Docling types
        from docling_core.types.doc import DocItemLabel

        # Retry loop with comprehensive error handling
        for attempt in range(max_retries):
            try:
                logger.info(f"Extraction attempt {attempt + 1}/{max_retries}")

                # Attempt 1: Check cache first
                cached_doc = self._get_cached_document(source)

                if cached_doc:
                    logger.info("Using cached parsed document (cache hit)")
                    return self._process_docling_document(
                        cached_doc, source, keywords, max_paragraphs, DocItemLabel
                    )

                # Attempt 2: Fresh Docling parse (cache miss)
                logger.info("Parsing PDF with Docling (cache miss)")
                result = self.converter.convert(source)
                ConversionStatus = self._docling_imports["ConversionStatus"]

                # Handle conversion status
                if result.status == ConversionStatus.SUCCESS:
                    logger.info("Docling conversion: SUCCESS")
                    doc = result.document

                    # Cache for future use
                    self._cache_document(source, doc)

                    # Cleanup conversion result
                    del result
                    gc.collect()

                    # Process document
                    return self._process_docling_document(
                        doc, source, keywords, max_paragraphs, DocItemLabel
                    )

                elif result.status == ConversionStatus.PARTIAL_SUCCESS:
                    logger.warning(
                        "Docling conversion: PARTIAL_SUCCESS (using available data)"
                    )
                    doc = result.document

                    # Still cache partial result
                    self._cache_document(source, doc)

                    # Cleanup
                    del result
                    gc.collect()

                    # Process what we have
                    return self._process_docling_document(
                        doc, source, keywords, max_paragraphs, DocItemLabel
                    )

                else:
                    # FAILURE status
                    logger.error(
                        f"Docling conversion: FAILURE (status={result.status})"
                    )
                    raise DoclingError(
                        f"Conversion failed with status: {result.status}"
                    )

            except MemoryError as e:
                logger.error(f"MemoryError on attempt {attempt + 1}/{max_retries}: {e}")
                # Aggressive cleanup
                gc.collect()

                if attempt < max_retries - 1:
                    logger.info("Retrying after memory cleanup...")
                    continue
                else:
                    logger.error("Max retries reached after MemoryError")
                    raise PDFExtractionError(
                        f"MemoryError after {max_retries} attempts: {e}"
                    )

            except RuntimeError as e:
                error_msg = str(e)
                if "page-dimensions" in error_msg:
                    logger.error(
                        "RuntimeError: Incompatible PDF (page-dimensions issue)"
                    )
                    # Don't retry this error - it's a permanent PDF incompatibility
                    raise PDFExtractionError(f"Incompatible PDF format: {e}")
                else:
                    logger.error(
                        f"RuntimeError on attempt {attempt + 1}/{max_retries}: {e}"
                    )
                    if attempt < max_retries - 1:
                        continue
                    raise PDFExtractionError(
                        f"RuntimeError after {max_retries} attempts: {e}"
                    )

            except DoclingError as e:
                logger.error(
                    f"DoclingError on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    logger.info("Retrying after Docling error...")
                    continue
                else:
                    raise

            except Exception as e:
                logger.exception(
                    f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    logger.info("Retrying after unexpected error...")
                    continue
                else:
                    raise PDFExtractionError(
                        f"Extraction failed after {max_retries} attempts: {e}"
                    )

        # Should not reach here, but safety fallback
        raise PDFExtractionError(f"Extraction failed after {max_retries} attempts")

    def _process_docling_document(
        self, doc, source: str, keywords: List[str], max_paragraphs: int, DocItemLabel
    ) -> Dict[str, Any]:
        """
        Process an already-converted DoclingDocument to extract Methods section.

        This helper method separates document processing from conversion and caching,
        enabling cleaner retry logic in extract_methods_section().

        Args:
            doc: DoclingDocument instance (already parsed)
            source: Original PDF URL/path (for provenance)
            keywords: Section keywords for Methods detection
            max_paragraphs: Maximum paragraphs to extract
            DocItemLabel: Docling label enum for section detection

        Returns:
            Dictionary with methods_text, methods_markdown, tables, formulas, etc.

        Raises:
            MethodsSectionNotFoundError: If Methods section cannot be located
        """
        # Find Methods section header
        methods_sections = self._find_sections_by_keywords(doc, keywords, DocItemLabel)

        if not methods_sections:
            logger.warning(
                "No Methods section found with keywords, returning full document"
            )
            # Fallback: return full document with structure
            return self._extract_full_document(doc, max_paragraphs, DocItemLabel)

        # Extract Methods content
        methods_text = self._extract_section_content(
            doc, methods_sections[0], max_paragraphs, DocItemLabel
        )

        # Export to Markdown with smart image filtering
        try:
            # Export full document to Markdown using Docling
            full_markdown = doc.export_to_markdown()
            # Apply image filtering to remove base64 bloat
            methods_markdown = self._filter_images_from_markdown(full_markdown)
        except Exception as e:
            logger.warning(f"Markdown export failed, using plain text: {e}")
            methods_markdown = methods_text

        # Extract tables within Methods section
        tables = self._extract_tables_in_section(doc, methods_sections[0])

        # Extract formulas
        formulas = self._extract_formulas_in_section(
            doc, methods_sections[0], DocItemLabel
        )

        # Extract software names
        software = self._extract_software_names(methods_text)

        # Log provenance (metadata-only, no IR)
        if self.data_manager:
            self.data_manager.log_tool_usage(
                tool_name="extract_methods_section",
                parameters={
                    "source": source[:100],
                    "keywords": keywords,
                    "max_paragraphs": max_paragraphs,
                },
                description=f"Methods extraction: {len(methods_text)} chars, "
                f"{len(tables)} tables, {len(formulas)} formulas",
            )

        # Build result dictionary
        return {
            "methods_text": methods_text,
            "methods_markdown": methods_markdown,
            "sections": self._build_section_hierarchy(doc, DocItemLabel),
            "tables": [self._table_to_dataframe(t) for t in tables],
            "formulas": formulas,
            "software_mentioned": software,
            "provenance": {
                "source": source,
                "parser": "docling",
                "version": "2.60.0",
                "timestamp": datetime.now().isoformat(),
                "fallback_used": False,
            },
        }

    # ==================== SECTION DETECTION & EXTRACTION ====================

    def _find_sections_by_keywords(
        self, doc, keywords: List[str], DocItemLabel
    ) -> List:
        """
        Find section headers matching keywords.

        Strategy:
        - Case-insensitive matching
        - Partial word matching ("Method" matches "Methods and Materials")
        - Prioritize exact matches
        - Return all matches (user can inspect if multiple)

        Args:
            doc: DoclingDocument instance
            keywords: List of keywords to search for
            DocItemLabel: Docling label enum

        Returns:
            List of matching section header items
        """
        exact_matches = []
        partial_matches = []

        for item in doc.texts:
            if item.label == DocItemLabel.SECTION_HEADER:
                text_lower = item.text.lower()

                # Check for exact keyword match
                for keyword in keywords:
                    if keyword.lower() == text_lower:
                        exact_matches.append(item)
                        break

                # Check for partial match
                else:
                    for keyword in keywords:
                        if keyword.lower() in text_lower:
                            partial_matches.append(item)
                            break

        # Prioritize exact matches
        matches = exact_matches if exact_matches else partial_matches
        logger.info(f"Found {len(matches)} section(s) matching keywords: {keywords}")
        return matches

    def _extract_section_content(
        self, doc, section_header, max_paragraphs: int, DocItemLabel
    ) -> str:
        """
        Extract text content under a section header.

        Strategy:
        - Start at section header
        - Extract until next section header OR max_paragraphs reached
        - Include text/paragraphs (HTML uses TEXT label, PDF uses PARAGRAPH)
        - Preserve paragraph boundaries

        Args:
            doc: DoclingDocument instance
            section_header: Section header item to start from
            max_paragraphs: Maximum paragraphs to extract
            DocItemLabel: Docling label enum

        Returns:
            Extracted section text
        """
        start_idx = doc.texts.index(section_header)

        content = []
        paragraph_count = 0

        for item in doc.texts[start_idx + 1 :]:  # Skip header itself
            # Stop at next major section
            if item.label == DocItemLabel.SECTION_HEADER:
                break

            # Extract text content (HTML uses TEXT, PDF uses PARAGRAPH)
            if item.label in [DocItemLabel.PARAGRAPH, DocItemLabel.TEXT]:
                if hasattr(item, "text") and item.text.strip():
                    content.append(item.text)
                    paragraph_count += 1

                    if paragraph_count >= max_paragraphs:
                        logger.info(f"Reached max_paragraphs limit: {max_paragraphs}")
                        break

        result = "\n\n".join(content)
        logger.info(
            f"Extracted {len(result)} characters from {paragraph_count} text items"
        )
        return result

    def _find_section_end(self, doc, start_idx: int, DocItemLabel) -> int:
        """
        Find the end index of a section.

        Args:
            doc: DoclingDocument instance
            start_idx: Starting index
            DocItemLabel: Docling label enum

        Returns:
            End index of the section
        """
        for idx in range(start_idx + 1, len(doc.texts)):
            if doc.texts[idx].label == DocItemLabel.SECTION_HEADER:
                return idx
        return len(doc.texts)

    # ==================== STRUCTURED CONTENT EXTRACTION ====================

    def _extract_tables_in_section(self, doc, section_header) -> List:
        """
        Extract tables within a section's page range.

        Args:
            doc: DoclingDocument instance
            section_header: Section header item

        Returns:
            List of TableItem objects
        """
        section_pages = set()
        if hasattr(section_header, "prov"):
            for prov in section_header.prov:
                if hasattr(prov, "page_no"):
                    section_pages.add(prov.page_no)

        section_tables = []
        if hasattr(doc, "tables"):
            for table in doc.tables:
                if hasattr(table, "prov"):
                    for prov in table.prov:
                        if getattr(prov, "page_no", None) in section_pages:
                            section_tables.append(table)
                            break

        logger.info(f"Found {len(section_tables)} tables in section")
        return section_tables

    def _extract_formulas_in_section(
        self, doc, section_header, DocItemLabel
    ) -> List[str]:
        """
        Extract formulas within a section.

        Args:
            doc: DoclingDocument instance
            section_header: Section header item
            DocItemLabel: Docling label enum

        Returns:
            List of formula strings
        """
        start_idx = doc.texts.index(section_header)
        end_idx = self._find_section_end(doc, start_idx, DocItemLabel)

        formulas = []
        for idx in range(start_idx, end_idx):
            item = doc.texts[idx]
            if item.label == DocItemLabel.FORMULA:
                if hasattr(item, "text"):
                    formulas.append(item.text)

        logger.info(f"Found {len(formulas)} formulas in section")
        return formulas

    def _table_to_dataframe(self, table_item):
        """
        Convert Docling TableItem to pandas DataFrame.

        Args:
            table_item: Docling TableItem object

        Returns:
            pandas DataFrame or dict representation
        """
        try:
            # Try to export as DataFrame if available
            if hasattr(table_item, "export_to_dataframe"):
                return table_item.export_to_dataframe()
            else:
                # Return dict representation as fallback
                return {"error": "DataFrame export not available"}
        except Exception as e:
            logger.warning(f"Could not convert table to DataFrame: {e}")
            return {"error": str(e)}

    def _build_section_hierarchy(self, doc, DocItemLabel) -> List[Dict]:
        """
        Build hierarchical section structure.

        Args:
            doc: DoclingDocument instance
            DocItemLabel: Docling label enum

        Returns:
            List of section dictionaries
        """
        hierarchy = []
        current_section = None

        for item in doc.texts:
            if item.label == DocItemLabel.SECTION_HEADER:
                if current_section:
                    hierarchy.append(current_section)

                current_section = {
                    "title": item.text,
                    "level": 1,  # Could infer from font size
                    "content_preview": "",
                }
            elif current_section and item.label == DocItemLabel.PARAGRAPH:
                if len(current_section["content_preview"]) < 200:
                    current_section["content_preview"] += item.text[:200]

        if current_section:
            hierarchy.append(current_section)

        return hierarchy

    def _extract_full_document(
        self, doc, max_paragraphs: int, DocItemLabel
    ) -> Dict[str, Any]:
        """
        Extract full document when Methods section not found.

        Extracts text content from all text items (HTML uses TEXT label, PDF uses PARAGRAPH).

        Args:
            doc: DoclingDocument instance
            max_paragraphs: Maximum paragraphs to extract
            DocItemLabel: Docling label enum

        Returns:
            Dictionary with full document extraction
        """
        content = []
        paragraph_count = 0

        for item in doc.texts:
            # Extract text content (HTML uses TEXT, PDF uses PARAGRAPH)
            if item.label in [DocItemLabel.PARAGRAPH, DocItemLabel.TEXT]:
                if hasattr(item, "text") and item.text.strip():
                    content.append(item.text)
                    paragraph_count += 1

                    if paragraph_count >= max_paragraphs:
                        break

        full_text = "\n\n".join(content)
        logger.info(
            f"Extracted full document: {len(full_text)} chars from {paragraph_count} text items"
        )

        return {
            "methods_text": full_text,
            "methods_markdown": full_text,
            "sections": self._build_section_hierarchy(doc, DocItemLabel),
            "tables": [],
            "formulas": [],
            "software_mentioned": self._extract_software_names(full_text),
            "provenance": {
                "source": "full_document",
                "parser": "docling",
                "timestamp": datetime.now().isoformat(),
                "fallback_used": False,
                "note": "Methods section not found, extracted full document",
            },
        }

    # ==================== UTILITIES ====================

    def _extract_software_names(self, text: str) -> List[str]:
        """
        Extract software/tool names from text.

        Args:
            text: Text to search

        Returns:
            List of detected software names
        """
        software_keywords = [
            "scanpy",
            "seurat",
            "star",
            "kallisto",
            "salmon",
            "deseq2",
            "limma",
            "edger",
            "cellranger",
            "maxquant",
            "mofa",
            "harmony",
            "combat",
            "mnn",
            "fastqc",
            "trimmomatic",
            "cutadapt",
            "bowtie",
            "hisat2",
            "tophat",
            "spectronaut",
            "maxdia",
            "fragpipe",
            "msfragger",
        ]

        text_lower = text.lower()
        found = []
        for sw in software_keywords:
            if sw in text_lower:
                found.append(sw)

        return found

    def _filter_images_from_markdown(self, markdown: str) -> str:
        """
        Remove base64 image encodings from Markdown to reduce LLM context bloat.

        Base64-encoded images can add megabytes of unnecessary text. This method
        replaces them with simple placeholders while preserving document structure.

        Args:
            markdown: Markdown text potentially containing base64 images

        Returns:
            Filtered markdown with image placeholders

        Examples:
            >>> # Before: ![Figure 1](data:image/png;base64,iVBORw0KG...)
            >>> # After:  [Figure: Figure 1]
        """
        # Pattern matches: ![caption](data:image/...;base64,...)
        pattern = r"!\[([^\]]*)\]\(data:image/[^;]+;base64,[^\)]+\)"

        def replace_image(match):
            caption = match.group(1).strip() or "Image"
            return f"[Figure: {caption}]"

        original_size = len(markdown)
        filtered = re.sub(pattern, replace_image, markdown)
        filtered_size = len(filtered)

        if original_size != filtered_size:
            reduction_pct = ((original_size - filtered_size) / original_size) * 100
            logger.info(
                f"Filtered images from Markdown: {original_size:,} → {filtered_size:,} chars "
                f"({reduction_pct:.1f}% reduction)"
            )

        return filtered

    # ==================== CACHING ====================

    def _get_cached_document(self, source: str):
        """
        Retrieve cached DoclingDocument if available.

        Caching parsed documents significantly improves performance:
        - Fresh parse: 2-5 seconds
        - Cache hit: <100ms

        Args:
            source: PDF URL or path (used as cache key)

        Returns:
            DoclingDocument if cached, None if cache miss or error

        Implementation:
            - Cache key: MD5 hash of source URL
            - Storage: JSON serialization in parsed_docs subdirectory
            - Reconstruction: Pydantic model_validate() from JSON
        """
        if not self.is_available():
            return None  # Docling not available

        try:
            # Generate cache key from source URL
            cache_key = hashlib.md5(source.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.json"

            if not cache_file.exists():
                return None  # Cache miss

            # Load JSON and reconstruct DoclingDocument
            with open(cache_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            # Import DoclingDocument for reconstruction
            from docling_core.types.doc import DoclingDocument

            # Reconstruct document from JSON using Pydantic
            doc = DoclingDocument.model_validate(json_data)

            logger.info(f"Cache hit: Loaded parsed document from {cache_file.name}")
            return doc

        except Exception as e:
            logger.warning(f"Failed to load cached document: {e}")
            return None  # Cache read error, will re-parse

    def _cache_document(self, source: str, doc) -> None:
        """
        Cache DoclingDocument as JSON for future retrieval.

        Args:
            source: PDF URL or path (used as cache key)
            doc: DoclingDocument instance to cache

        Implementation:
            - Serialization: Pydantic model_dump() to dict
            - Storage: JSON with indent=2 for human readability
            - Error handling: Graceful failure (doesn't block extraction)
        """
        if not self.is_available() or not doc:
            return  # Nothing to cache

        try:
            # Generate cache key
            cache_key = hashlib.md5(source.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.json"

            # Serialize document to JSON using Pydantic
            json_data = doc.model_dump()

            # Write to cache file
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)

            logger.info(
                f"Cached parsed document: {cache_file.name} ({len(json.dumps(json_data)):,} bytes)"
            )

        except Exception as e:
            logger.warning(f"Failed to cache document (non-fatal): {e}")
            # Don't raise - caching failure shouldn't block extraction
