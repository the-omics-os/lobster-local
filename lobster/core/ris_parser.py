"""
RIS file parser for publication metadata extraction.

This module provides RISParser class for parsing .ris (Research Information Systems)
files and converting them to PublicationQueueEntry objects for queueing.

Note: This parser uses rispy (not pybtex) since rispy is the standard library
for RIS format parsing, while pybtex is designed for BibTeX format.
"""

import hashlib
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from lobster.core.schemas.publication_queue import (
    ExtractionLevel,
    PublicationQueueEntry,
    PublicationStatus,
)

logger = logging.getLogger(__name__)


class RISParseError(Exception):
    """Raised when RIS file parsing fails."""

    pass


class RISParser:
    """
    Parser for .ris (Research Information Systems) files.

    Converts RIS bibliography entries to PublicationQueueEntry objects
    for publication extraction workflow.

    RIS Field Mappings:
        TY - Type of reference (journal article, book, etc.)
        TI - Title
        AU - Authors (repeatable)
        PY - Publication year
        JO/T2 - Journal name
        AB - Abstract
        DO - DOI
        PMID - PubMed ID
        PMC - PubMed Central ID
        UR - URL (maps to 'urls' in rispy)
        L1 - Link 1 (maps to 'file_attachments1' in rispy, often PDF)
        L2 - Link 2 (maps to 'file_attachments2' in rispy, often PubMed)
        KW - Keywords (repeatable)
    """

    # Publisher URL transformation patterns for abstract → fulltext conversion
    PUBLISHER_URL_TRANSFORMS = {
        "cell.com": {
            "pattern": r"/abstract/",
            "replacement": "/fulltext/",
        },
        "wiley.com": {
            "pattern": r"/doi/abs/",
            "replacement": "/doi/full/",
        },
        "onlinelibrary.wiley.com": {
            "pattern": r"/doi/abs/",
            "replacement": "/doi/full/",
        },
        "springer.com": {
            # Springer /article/ URLs are already full text
            "pattern": None,
        },
        "nature.com": {
            # Nature /articles/ URLs are already full text
            "pattern": None,
        },
        "frontiersin.org": {
            # Frontiers often provides /full in UR, /pdf in L1
            "pattern": None,
        },
        "sciencedirect.com": {
            "pattern": r"/article/abs/",
            "replacement": "/article/",
        },
        "tandfonline.com": {
            "pattern": r"/doi/abs/",
            "replacement": "/doi/full/",
        },
    }

    def __init__(self):
        """Initialize RIS parser."""
        self.stats = {"parsed": 0, "skipped": 0, "errors": []}

    def parse_file(
        self, file_path: Path, encoding: str = "utf-8"
    ) -> List[PublicationQueueEntry]:
        """
        Parse RIS file and convert to PublicationQueueEntry objects.

        Args:
            file_path: Path to .ris file
            encoding: File encoding (default: utf-8)

        Returns:
            List[PublicationQueueEntry]: List of parsed queue entries

        Raises:
            RISParseError: If file cannot be read or parsed
        """
        if not file_path.exists():
            raise RISParseError(f"File not found: {file_path}")

        if not file_path.suffix.lower() in [".ris", ".txt"]:
            raise RISParseError(
                f"File must have .ris or .txt extension, got {file_path.suffix}"
            )

        # Reset statistics
        self.stats = {"parsed": 0, "skipped": 0, "errors": []}

        try:
            # Try to import rispy
            import rispy
        except ImportError:
            raise RISParseError(
                "rispy library not installed. Install with: pip install rispy"
            )

        try:
            with open(file_path, "r", encoding=encoding) as f:
                entries = rispy.load(f)

            queue_entries = []
            for idx, ris_entry in enumerate(entries, start=1):
                try:
                    queue_entry = self._convert_ris_to_queue_entry(ris_entry)
                    queue_entries.append(queue_entry)
                    self.stats["parsed"] += 1
                except Exception as e:
                    self.stats["skipped"] += 1
                    self.stats["errors"].append(f"Entry {idx}: {str(e)}")
                    logger.warning(f"Skipping malformed entry {idx}: {e}")
                    continue

            logger.debug(
                f"Parsed {self.stats['parsed']} entries, skipped {self.stats['skipped']}"
            )
            return queue_entries

        except Exception as e:
            logger.error(f"Failed to parse RIS file: {e}")
            raise RISParseError(f"Failed to parse RIS file: {e}") from e

    def _convert_ris_to_queue_entry(
        self,
        ris_entry: Dict[str, Any],
        priority: int = 5,
        extraction_level: ExtractionLevel = ExtractionLevel.METHODS,
        schema_type: str = "general",
    ) -> PublicationQueueEntry:
        """
        Convert RIS entry to PublicationQueueEntry.

        Args:
            ris_entry: Parsed RIS entry dictionary
            priority: Processing priority (1-10)
            extraction_level: Target extraction depth
            schema_type: Schema type for validation

        Returns:
            PublicationQueueEntry: Converted queue entry

        Raises:
            ValueError: If required fields are missing
        """
        # Extract identifiers first (needed for deterministic entry_id)
        pmid = self._extract_pmid(ris_entry)
        doi = self._extract_doi(ris_entry)
        pmc_id = self._extract_pmc_id(ris_entry)

        # Extract title early (needed for deterministic entry_id fallback)
        title = (
            ris_entry.get("title")
            or ris_entry.get("primary_title")
            or ris_entry.get("TI")
        )

        # Generate DETERMINISTIC entry ID to prevent duplicates on re-import
        # Priority: DOI > PMID > title hash > random UUID
        if doi:
            # Normalize DOI for use in ID (replace special chars)
            doi_normalized = doi.replace("/", "_").replace(".", "_").lower()
            entry_id = f"pub_queue_doi_{doi_normalized}"
        elif pmid:
            entry_id = f"pub_queue_pmid_{pmid}"
        elif title:
            # Use MD5 hash of title for deterministic ID
            title_hash = hashlib.md5(title.encode()).hexdigest()[:12]
            entry_id = f"pub_queue_title_{title_hash}"
        else:
            # Fallback to random UUID (should be rare)
            entry_id = f"pub_queue_{uuid.uuid4().hex[:10]}"

        # Extract metadata (title already extracted above for entry_id)
        authors = self._extract_authors(ris_entry)
        year = self._extract_year(ris_entry)
        journal = (
            ris_entry.get("journal_name")
            or ris_entry.get("secondary_title")
            or ris_entry.get("JO")
            or ris_entry.get("T2")
        )

        # Extract ALL URL fields from RIS (UR, L1, L2)
        primary_url = self._extract_primary_url(ris_entry)
        pdf_url = self._extract_pdf_url(ris_entry)
        pubmed_url = self._extract_pubmed_url(ris_entry)

        # Transform abstract URL to fulltext URL for known publishers
        fulltext_url = self._transform_to_fulltext(primary_url)

        # Extract supplementary info
        abstract = ris_entry.get("abstract") or ris_entry.get("AB")
        keywords = ris_entry.get("keywords") or ris_entry.get("KW", [])

        # Infer schema_type from keywords or type
        inferred_schema = self._infer_schema_type(ris_entry)
        if inferred_schema and schema_type == "general":
            schema_type = inferred_schema

        # Build metadata dict for later use
        extracted_metadata = {
            "abstract": abstract,
            "keywords": keywords if isinstance(keywords, list) else [keywords],
            "type": ris_entry.get("type_of_reference") or ris_entry.get("TY"),
        }

        # Create queue entry with all URL fields
        entry = PublicationQueueEntry(
            entry_id=entry_id,
            pmid=pmid,
            doi=doi,
            pmc_id=pmc_id,
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            priority=priority,
            status=PublicationStatus.PENDING,
            extraction_level=extraction_level,
            schema_type=schema_type,
            metadata_url=primary_url,
            pdf_url=pdf_url,
            pubmed_url=pubmed_url,
            fulltext_url=fulltext_url,
            extracted_metadata=extracted_metadata,
        )

        return entry

    def _extract_doi(self, ris_entry: Dict[str, Any]) -> Optional[str]:
        """Extract and sanitize DOI values."""
        doi = ris_entry.get("doi") or ris_entry.get("DO")

        if isinstance(doi, list):
            doi = doi[0] if doi else None

        if isinstance(doi, str):
            doi = doi.strip()
            match = re.search(r"(10\.\S+?\/\S+)", doi)
            if match:
                return match.group(1)
            return doi

        return doi

    def _extract_pmid(self, ris_entry: Dict[str, Any]) -> Optional[str]:
        """Extract PMID from various RIS field locations."""
        # Check standard fields
        pmid = ris_entry.get("pmid") or ris_entry.get("PMID")

        # Check notes field for PMID
        notes = ris_entry.get("notes") or ris_entry.get("N1") or ""
        if isinstance(notes, str):
            match = re.search(r"PMID[:\-\s]*(\d+)", notes, re.IGNORECASE)
            if match:
                pmid = match.group(1)

        # Fallback: search all string fields for embedded PMID (non-standard RIS exports)
        if not pmid:
            for value in ris_entry.values():
                if isinstance(value, str):
                    match = re.search(r"PMID[:\-\s]*(\d+)", value, re.IGNORECASE)
                    if match:
                        pmid = match.group(1)
                        break

        # Clean PMID format
        if pmid:
            pmid = re.sub(r"[^0-9]", "", str(pmid))

        return pmid

    def _extract_pmc_id(self, ris_entry: Dict[str, Any]) -> Optional[str]:
        """Extract PMC ID from various RIS field locations."""
        pmc_id = ris_entry.get("pmc") or ris_entry.get("PMC")

        # Check notes field for PMC
        notes = ris_entry.get("notes") or ris_entry.get("N1") or ""
        if isinstance(notes, str):
            match = re.search(r"PMC\s*-?\s*(\d+)", notes, re.IGNORECASE)
            if match:
                pmc_id = f"PMC{match.group(1)}"

        # Fallback: search in string fields when PMC is embedded elsewhere
        if not pmc_id:
            for value in ris_entry.values():
                if isinstance(value, str):
                    match = re.search(r"PMC\s*-?\s*(\d+)", value, re.IGNORECASE)
                    if match:
                        pmc_id = f"PMC{match.group(1)}"
                        break

        return pmc_id

    def _extract_authors(self, ris_entry: Dict[str, Any]) -> List[str]:
        """Extract and format author list."""
        authors = ris_entry.get("authors") or ris_entry.get("AU", [])

        if not authors:
            return []

        # Handle single author string
        if isinstance(authors, str):
            return [authors]

        # Handle list of authors
        if isinstance(authors, list):
            return [str(author) for author in authors]

        return []

    def _extract_year(self, ris_entry: Dict[str, Any]) -> Optional[int]:
        """Extract publication year from various fields."""
        year_raw = (
            ris_entry.get("year")
            or ris_entry.get("publication_year")
            or ris_entry.get("PY")
        )

        if not year_raw:
            return None

        try:
            # Handle formats like "2022", "2022/01/15", "2022-01-15"
            year_str = str(year_raw).split("/")[0].split("-")[0].strip()
            return int(year_str)
        except (ValueError, AttributeError):
            logger.warning(f"Could not parse year: {year_raw}")
            return None

    def _infer_schema_type(self, ris_entry: Dict[str, Any]) -> Optional[str]:
        """
        Infer schema type from keywords and content.

        Returns:
            str: Inferred schema type (microbiome, single_cell, etc.) or None
        """
        # Get text content for keyword matching
        title = str(ris_entry.get("title", "")).lower()
        abstract = str(ris_entry.get("abstract", "")).lower()
        keywords = ris_entry.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        keywords_text = " ".join([str(k).lower() for k in keywords])

        combined_text = f"{title} {abstract} {keywords_text}"

        # Keyword patterns for schema inference
        schema_patterns = {
            "microbiome": [
                "microbiome",
                "16s rrna",
                "metagenomics",
                "microbiota",
                "gut bacteria",
            ],
            "single_cell": [
                "single-cell",
                "single cell",
                "scrna-seq",
                "sc-rna",
                "10x genomics",
            ],
            "bulk_rnaseq": ["rna-seq", "rnaseq", "transcriptomics", "gene expression"],
            "proteomics": [
                "proteomics",
                "mass spectrometry",
                "protein expression",
                "lc-ms",
            ],
            "spatial": [
                "spatial transcriptomics",
                "visium",
                "spatial omics",
                "imaging mass cytometry",
            ],
            "metabolomics": ["metabolomics", "metabolome", "metabolite profiling"],
        }

        # Check each schema type
        for schema_type, patterns in schema_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                return schema_type

        return None  # Use default "general"

    def _transform_to_fulltext(self, url: Optional[str]) -> Optional[str]:
        """
        Transform abstract URL to fulltext URL for known publishers.

        Args:
            url: Primary article URL (often abstract page)

        Returns:
            Transformed fulltext URL if transformation is available, None otherwise
        """
        if not url:
            return None

        url_lower = url.lower()

        for domain, transform in self.PUBLISHER_URL_TRANSFORMS.items():
            if domain in url_lower and transform.get("pattern"):
                transformed = re.sub(
                    transform["pattern"], transform["replacement"], url
                )
                if transformed != url:
                    logger.debug(f"Transformed URL: {url} → {transformed}")
                    return transformed

        return None  # No transformation available

    def _extract_pdf_url(self, ris_entry: Dict[str, Any]) -> Optional[str]:
        """
        Extract PDF URL from RIS L1 field.

        L1 (file_attachments1 in rispy) often contains direct PDF link.

        Args:
            ris_entry: Parsed RIS entry dictionary

        Returns:
            PDF URL if available, None otherwise
        """
        # rispy maps L1 to 'file_attachments1'
        link1 = ris_entry.get("file_attachments1") or ris_entry.get("L1")

        if not link1:
            return None

        # Handle list (some RIS files have multiple L1 entries)
        if isinstance(link1, list):
            link1 = link1[0] if link1 else None

        if not link1:
            return None

        link1_lower = link1.lower()

        # Check if URL looks like a PDF link
        if ".pdf" in link1_lower or "/pdf" in link1_lower:
            return link1

        return None

    def _extract_pubmed_url(self, ris_entry: Dict[str, Any]) -> Optional[str]:
        """
        Extract PubMed URL from RIS L2 field.

        L2 (file_attachments2 in rispy) often contains PubMed link.

        Args:
            ris_entry: Parsed RIS entry dictionary

        Returns:
            PubMed URL if available, None otherwise
        """
        # rispy maps L2 to 'file_attachments2'
        link2 = ris_entry.get("file_attachments2") or ris_entry.get("L2")

        if not link2:
            return None

        # Handle list
        if isinstance(link2, list):
            link2 = link2[0] if link2 else None

        if not link2:
            return None

        link2_lower = link2.lower()

        # Check if URL is PubMed
        if (
            "ncbi.nlm.nih.gov/pubmed" in link2_lower
            or "pubmed.ncbi.nlm.nih.gov" in link2_lower
        ):
            return link2

        return None

    def _extract_primary_url(self, ris_entry: Dict[str, Any]) -> Optional[str]:
        """
        Extract primary URL from RIS UR field.

        rispy maps UR to 'urls' (list) or 'url' (string).

        Args:
            ris_entry: Parsed RIS entry dictionary

        Returns:
            Primary article URL if available, None otherwise
        """
        # rispy maps UR to 'urls' (as list) or sometimes 'url'
        urls = ris_entry.get("urls") or ris_entry.get("url") or ris_entry.get("UR")

        if not urls:
            return None

        # Handle list
        if isinstance(urls, list):
            return urls[0] if urls else None

        return urls

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get parsing statistics.

        Returns:
            Dict: Statistics including parsed count, skipped count, errors
        """
        return self.stats.copy()

    def to_publication_entry(
        self,
        ris_entry: Dict[str, Any],
        priority: int = 5,
        schema_type: str = "general",
        extraction_level: str = "methods",
    ) -> PublicationQueueEntry:
        """
        Public API for converting single RIS entry to PublicationQueueEntry.

        Args:
            ris_entry: Parsed RIS entry dictionary
            priority: Processing priority (1-10)
            schema_type: Schema type for validation
            extraction_level: Target extraction depth

        Returns:
            PublicationQueueEntry: Converted queue entry
        """
        # Convert string to enum
        if isinstance(extraction_level, str):
            extraction_level = ExtractionLevel(extraction_level.lower())

        return self._convert_ris_to_queue_entry(
            ris_entry,
            priority=priority,
            extraction_level=extraction_level,
            schema_type=schema_type,
        )
