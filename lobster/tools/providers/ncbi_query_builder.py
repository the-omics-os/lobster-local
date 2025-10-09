"""
Shared NCBI query builder for all NCBI databases (PubMed, GEO, SRA, etc.).

This module provides a unified, stable query construction system for NCBI E-utilities,
handling all the complexity of NCBI query syntax in a consistent way.
"""

import re
import urllib.parse
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class NCBIDatabase(Enum):
    """NCBI database identifiers."""

    PUBMED = "pubmed"
    GEO = "gse"  # GEO Series
    GDS = "gds"  # GEO DataSets
    SRA = "sra"
    BIOPROJECT = "bioproject"
    BIOSAMPLE = "biosample"


class NCBIFieldTag:
    """Common NCBI field tags used across databases."""

    # Common fields
    ALL_FIELDS = "ALL"
    TITLE = "TITL"
    ABSTRACT = "TIAB"
    AUTHOR = "AUTH"
    JOURNAL = "JOUR"
    PUBLICATION_DATE = "PDAT"
    PUBLICATION_TYPE = "PTYP"
    ACCESSION = "ACCN"
    UID = "UID"
    PMID = "PMID"
    DOI = "DOI"

    # GEO-specific fields
    ENTRY_TYPE = "ETYP"
    ORGANISM = "ORGN"
    SUPPLEMENTARY_FILE = "suppFile"
    PLATFORM = "GPL"

    # SRA-specific fields
    STRATEGY = "STRA"
    SOURCE = "SRC"
    SELECTION = "SEL"
    LAYOUT = "LAY"


class NCBIQueryBuilder:
    """
    Unified NCBI query builder for all databases.

    This builder provides a simple, consistent interface for building
    NCBI queries while handling all the complexity internally.
    """

    def __init__(self, database: NCBIDatabase = NCBIDatabase.PUBMED):
        """
        Initialize the query builder for a specific database.

        Args:
            database: The NCBI database to build queries for
        """
        self.database = database
        self._field_mappings = self._get_field_mappings()

    def _get_field_mappings(self) -> Dict[str, str]:
        """Get database-specific field mappings."""
        # Map common field names to database-specific tags
        if self.database == NCBIDatabase.PUBMED:
            return {
                "title": "TITL",
                "abstract": "TIAB",
                "author": "AUTH",
                "journal": "JOUR",
                "date": "PDAT",
                "pmid": "PMID",
                "doi": "DOI",
                "publication_type": "PTYP",
            }
        elif self.database == NCBIDatabase.GEO:
            return {
                "title": "TITL",
                "organism": "ORGN",
                "platform": "GPL",
                "accession": "ACCN",
                "entry_type": "ETYP",
                "date": "PDAT",
                "supplementary": "suppFile",
            }
        elif self.database == NCBIDatabase.SRA:
            return {
                "accession": "ACCN",
                "organism": "ORGN",
                "strategy": "STRA",
                "source": "SRC",
                "layout": "LAY",
            }
        else:
            return {}

    def build_query(
        self,
        terms: Optional[Union[str, List[str]]] = None,
        filters: Optional[Dict[str, Any]] = None,
        combine_with: str = "AND",
    ) -> str:
        """
        Build an NCBI query from terms and filters.

        Args:
            terms: Search terms (string or list of strings)
            filters: Dictionary of filters to apply
            combine_with: How to combine multiple terms ("AND" or "OR")

        Returns:
            str: Properly formatted NCBI query string

        Examples:
            >>> builder = NCBIQueryBuilder(NCBIDatabase.PUBMED)
            >>> builder.build_query("cancer", {"author": "Smith"})
            'cancer AND Smith[AUTH]'

            >>> builder = NCBIQueryBuilder(NCBIDatabase.GEO)
            >>> builder.build_query("RNA-seq", {"organism": "human", "entry_type": "gse"})
            'RNA-seq AND human[ORGN] AND gse[ETYP]'
        """
        query_parts = []

        # Handle search terms
        if terms:
            if isinstance(terms, str):
                # Single term - don't add quotes unless it contains spaces
                term_query = self._format_term(terms)
                if term_query:
                    query_parts.append(term_query)
            elif isinstance(terms, list):
                # Multiple terms - combine with specified operator
                term_queries = [self._format_term(t) for t in terms if t]
                if term_queries:
                    if len(term_queries) == 1:
                        query_parts.append(term_queries[0])
                    else:
                        combined = f" {combine_with} ".join(term_queries)
                        query_parts.append(f"({combined})")

        # Handle filters
        if filters:
            filter_queries = self._build_filters(filters)
            query_parts.extend(filter_queries)

        # Combine all parts with AND
        if not query_parts:
            return "*"  # Return all results if no query specified

        return " AND ".join(query_parts)

    def _format_term(self, term: str) -> str:
        """
        Format a single search term.

        Clean and escape a single query term:
        - Do NOT wrap if the input looks like a complex boolean/expression
          (contains AND/OR/NOT, parentheses, field tags, ranges, or quotes).
        - Remove accidental outer quotes if present.
        - Quote only simple multi-word phrases.

        Args:
            term: Search term to format

        Returns:
            str: Formatted term
        """
        if not term or not term.strip():
            return ""

        t = term.strip()

        # Remove one layer of accidental outer quotes around the whole term
        if t.startswith('"') and t.endswith('"'):
            t = t[1:-1].strip()

        # If complex expression, return as-is (no quoting)
        if self._is_complex_expression(t):
            return t

        # Collapse repeated internal quotes (e.g., '""lung cancer""' -> '"lung cancer"')
        t = re.sub(r'("{2,})', '"', t)

        # Quote only simple multi-word phrases
        if " " in t:
            return f'"{t}"'

        return t

    def _is_complex_expression(self, s: str) -> bool:
        """Heuristic: treat as complex if it includes boolean ops, grouping, field tags, ranges, or quotes."""
        if '"' in s:
            return True
        if re.search(r"\b(AND|OR|NOT)\b", s, flags=re.IGNORECASE):
            return True
        if any(ch in s for ch in "()[]"):
            return True
        if ":" in s:  # date ranges like 2018:2024[PDAT]
            return True
        return False

    def _build_filters(self, filters: Dict[str, Any]) -> List[str]:
        """
        Build filter queries from a filter dictionary.

        Args:
            filters: Dictionary of filters

        Returns:
            List[str]: List of filter query strings
        """
        filter_queries = []

        for key, value in filters.items():
            if value is None or value == "":
                continue

            # Handle special filters
            if key == "date_range" and isinstance(value, dict):
                date_filter = self._build_date_range_filter(value)
                if date_filter:
                    filter_queries.append(date_filter)
            elif key in self._field_mappings:
                # Standard field filter
                field_tag = self._field_mappings[key]
                if isinstance(value, list):
                    # Multiple values - use OR
                    sub_queries = [
                        f"{self._format_filter_value(v)}[{field_tag}]"
                        for v in value
                        if v
                    ]
                    if len(sub_queries) == 1:
                        filter_queries.append(sub_queries[0])
                    elif len(sub_queries) > 1:
                        filter_queries.append(f"({' OR '.join(sub_queries)})")
                else:
                    formatted_value = self._format_filter_value(value)
                    if formatted_value:
                        filter_queries.append(f"{formatted_value}[{field_tag}]")
            elif key in ["gse", "gds", "gpl", "gsm"]:
                # GEO entry type shortcuts
                if self.database == NCBIDatabase.GEO:
                    filter_queries.append(f"{key}[ETYP]")

        return filter_queries

    def _format_filter_value(self, value: Any) -> str:
        """Format a filter value for use in queries."""
        if isinstance(value, bool):
            return str(value).lower()

        value_str = str(value).strip()

        # Don't quote if it's likely an accession or simple identifier
        if re.match(r"^[A-Z]+\d+$", value_str):
            return value_str

        # Quote if contains spaces (no escaping needed)
        if " " in value_str:
            return f'"{value_str}"'

        return value_str

    def _build_date_range_filter(self, date_range: Dict[str, str]) -> Optional[str]:
        """
        Build a date range filter using proper NCBI E-utilities syntax.

        Args:
            date_range: Dictionary with 'start' and/or 'end' dates

        Returns:
            Optional[str]: Date filter string or None
        """
        start = date_range.get("start", "").strip()
        end = date_range.get("end", "").strip()

        if not start and not end:
            return None

        # Get appropriate date field for database
        date_field = self._field_mappings.get("date", "PDAT")

        # Format dates
        start = self._format_date(start) if start else None
        end = self._format_date(end) if end else None

        if start and end:
            # Use proper NCBI date range syntax with parentheses
            return f'("{start}"[{date_field}] : "{end}"[{date_field}])'
        elif start:
            return f'"{start}"[{date_field}]'
        elif end:
            # Default start date for open-ended ranges
            return f'("1900/01/01"[{date_field}] : "{end}"[{date_field}])'

        return None

    def _format_date(self, date_str: str) -> Optional[str]:
        """Format a date for NCBI queries (requires YYYY/MM/DD format)."""
        if not date_str:
            return None

        # Already in YYYY/MM/DD format
        if re.match(r"^\d{4}/\d{1,2}/\d{1,2}$", date_str):
            year, month, day = date_str.split("/")
            return f"{year}/{month.zfill(2)}/{day.zfill(2)}"

        # Already in YYYY/MM format - add day
        if re.match(r"^\d{4}/\d{1,2}$", date_str):
            year, month = date_str.split("/")
            return f"{year}/{month.zfill(2)}/01"

        # Just year - add month and day
        if re.match(r"^\d{4}$", date_str):
            return f"{date_str}/01/01"

        # Try to parse other formats
        try:
            # ISO format YYYY-MM-DD
            if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", date_str):
                parts = date_str.split("-")
                return f"{parts[0]}/{parts[1].zfill(2)}/{parts[2].zfill(2)}"
            # ISO format YYYY-MM
            elif re.match(r"^\d{4}-\d{1,2}$", date_str):
                parts = date_str.split("-")
                return f"{parts[0]}/{parts[1].zfill(2)}/01"
        except:
            pass

        # If we can't parse it, return as-is and let NCBI handle it
        return date_str

    def add_field_tag(self, term: str, field: str) -> str:
        """
        Add a field tag to a term.

        Args:
            term: The search term
            field: The field name or tag

        Returns:
            str: Term with field tag
        """
        # Get the appropriate field tag
        field_tag = self._field_mappings.get(field.lower(), field.upper())

        # Format the term
        formatted_term = self._format_filter_value(term)

        return f"{formatted_term}[{field_tag}]"

    def combine_queries(self, queries: List[str], operator: str = "AND") -> str:
        """
        Combine multiple queries with an operator.

        Args:
            queries: List of query strings
            operator: Boolean operator (AND, OR, NOT)

        Returns:
            str: Combined query with normalized formatting
        """
        # Filter out empty queries
        valid_queries = [q.strip() for q in queries if q and q.strip()]

        if not valid_queries:
            return ""

        if len(valid_queries) == 1:
            return self._normalize_query(valid_queries[0])

        # Wrap each query in parentheses if complex
        wrapped_queries = []
        for query in valid_queries:
            if " AND " in query or " OR " in query or " NOT " in query:
                wrapped_queries.append(f"({query})")
            else:
                wrapped_queries.append(query)

        result = f" {operator} ".join(wrapped_queries)
        return self._normalize_query(result)

    def _normalize_query(self, q: str) -> str:
        """
        Final cleanup to prevent top-level stray quotes:
        - Remove outer quotes around the entire expression.
        - Collapse duplicate quotes.
        - Remove a dangling quote directly before an opening '(' at start.

        Args:
            q: Query string to normalize

        Returns:
            str: Normalized query
        """
        if not q:
            return ""

        qq = q.strip()

        # Strip single layer of accidental outer quotes around the whole expression
        if qq.startswith('"') and qq.endswith('"'):
            qq = qq[1:-1].strip()

        # Remove any leading quote immediately before '(' (fixes: term="("single-...)
        qq = re.sub(r'^\s*"\s*(\()', r"\1", qq)

        # Collapse repeated quotes
        qq = re.sub(r'"{2,}', '"', qq)

        return qq

    def validate_query(self, query: str) -> bool:
        """
        Validate that a query has proper syntax.

        Args:
            query: Query string to validate

        Returns:
            bool: True if query syntax is valid
        """
        if not query or not query.strip():
            return False

        # Check for balanced parentheses
        paren_count = 0
        for char in query:
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
                if paren_count < 0:
                    return False

        if paren_count != 0:
            return False

        # Check for balanced quotes (should be even number)
        # Simple count since we don't escape quotes
        quote_count = query.count('"')
        if quote_count % 2 != 0:
            return False

        # Check for balanced square brackets
        bracket_count = 0
        for char in query:
            if char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if bracket_count < 0:
                    return False

        if bracket_count != 0:
            return False

        return True

    def url_encode(self, query: str) -> str:
        """
        URL encode a query for use in HTTP requests.

        Keep (), [], and : unencoded; encode quotes.
        This avoids breaking field tags and date ranges while keeping quotes safe.

        Args:
            query: Query string to encode

        Returns:
            str: URL-encoded query
        """
        return urllib.parse.quote(query, safe="()[]:")


class PubMedQueryBuilder(NCBIQueryBuilder):
    """Specialized query builder for PubMed with additional features."""

    def __init__(self):
        """Initialize PubMed query builder."""
        super().__init__(NCBIDatabase.PUBMED)

    def build_clinical_trial_query(
        self, condition: str, phase: Optional[str] = None
    ) -> str:
        """Build a query for clinical trials."""
        filters = {"publication_type": "Clinical Trial"}

        if phase:
            # Add phase to the search terms
            terms = [condition, f"phase {phase}"]
            return self.build_query(terms, filters)

        return self.build_query(condition, filters)

    def build_review_query(self, topic: str, systematic: bool = False) -> str:
        """Build a query for review articles."""
        if systematic:
            filters = {"publication_type": "Systematic Review"}
        else:
            filters = {"publication_type": "Review"}

        return self.build_query(topic, filters)


class GEOQueryBuilder(NCBIQueryBuilder):
    """Specialized query builder for GEO with additional features."""

    def __init__(self):
        """Initialize GEO query builder."""
        super().__init__(NCBIDatabase.GEO)

    def build_expression_query(
        self,
        keywords: Optional[str] = None,
        organism: Optional[str] = None,
        platform: Optional[str] = None,
        series_only: bool = True,
    ) -> str:
        """Build a query for gene expression datasets."""
        filters = {}

        if organism:
            filters["organism"] = organism

        if platform:
            filters["platform"] = platform

        if series_only:
            filters["gse"] = True  # This becomes gse[ETYP]

        return self.build_query(keywords, filters)

    def build_sequencing_query(
        self,
        seq_type: str,
        organism: Optional[str] = None,
        date_range: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Build a query for sequencing datasets.

        Args:
            seq_type: Type of sequencing (e.g., 'rna-seq', 'chip-seq')
            organism: Optional organism filter
            date_range: Optional date range dict with 'start' and/or 'end' dates
                       (e.g., {'start': '2020/01/01', 'end': '2024/12/31'})

        Returns:
            str: Formatted query string
        """
        # Common sequencing keywords
        keywords = {
            "rna-seq": "RNA-seq",
            "chip-seq": "ChIP-seq",
            "atac-seq": "ATAC-seq",
            "single-cell": "single-cell RNA-seq",
            "scrna-seq": "single-cell RNA-seq",
            "wgs": "whole genome sequencing",
            "wes": "whole exome sequencing",
        }

        search_term = keywords.get(seq_type.lower(), seq_type)

        filters = {}
        if organism:
            filters["organism"] = organism

        if date_range:
            filters["date_range"] = date_range

        # Default to series
        filters["gse"] = True

        return self.build_query(search_term, filters)


# Convenience functions for backward compatibility
def build_pubmed_query(
    terms: Union[str, List[str]], filters: Optional[Dict] = None
) -> str:
    """Build a PubMed query (convenience function)."""
    builder = PubMedQueryBuilder()
    return builder.build_query(terms, filters)


def build_geo_query(
    terms: Union[str, List[str]], filters: Optional[Dict] = None
) -> str:
    """Build a GEO query (convenience function)."""
    builder = GEOQueryBuilder()
    return builder.build_query(terms, filters)
