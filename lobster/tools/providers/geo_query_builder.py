"""
GEO query builder for constructing NCBI-compliant search queries.

This module provides specialized query construction for GEO DataSets searches,
implementing all field tags and filters from the official NCBI API examples.
"""

import re
import urllib.parse
from typing import Dict, List, Optional, Union
from enum import Enum
from datetime import datetime, timedelta

from pydantic import BaseModel, Field


class GEOFieldTag(Enum):
    """GEO-specific search field tags."""
    ENTRY_TYPE = "ETYP"  # GSE, GDS, GPL, GSM
    ORGANISM = "ORGN"
    ACCESSION = "ACCN"
    PUBLICATION_DATE = "PDAT"
    SUPPLEMENTARY_FILE = "suppFile"
    TITLE = "TITL"
    ALL_FIELDS = "ALL"


class GEOEntryType(Enum):
    """GEO database entry types."""
    SERIES = "gse"  # Series (experiments)
    DATASET = "gds"  # Curated datasets
    PLATFORM = "gpl"  # Platforms
    SAMPLE = "gsm"  # Individual samples


class GEOSearchFilters(BaseModel):
    """Filters for GEO dataset searches."""
    entry_types: Optional[List[GEOEntryType]] = None
    organisms: Optional[List[str]] = None
    platforms: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None  # {"start": "2023/01", "end": "2024/12"}
    supplementary_file_types: Optional[List[str]] = None  # ["cel", "txt", "h5"]
    published_last_n_months: Optional[int] = None
    max_results: int = Field(default=20, ge=1, le=5000)
    use_history: bool = True


class GEOQueryBuilder:
    """
    Constructs NCBI-compliant queries for GEO DataSets searches.
    
    This builder handles all the complexity of GEO query syntax, including:
    - Field tag application (ETYP, ORGN, PDAT, etc.)
    - Special filter handling ("published last N months"[Filter])
    - Proper escaping and encoding
    - Query combination logic
    """
    
    def __init__(self):
        """Initialize the query builder."""
        pass
    
    def build_geo_query(
        self, 
        base_query: str, 
        filters: Optional[GEOSearchFilters] = None
    ) -> str:
        """
        Build a complete GEO query string from base query and filters.
        
        Args:
            base_query: Base search terms (e.g., "single-cell RNA-seq")
            filters: Optional filters to apply
            
        Returns:
            str: Complete NCBI-compliant query string
        """
        if not filters:
            return self.escape_query_term(base_query)
        
        query_parts = []
        
        # Add base query if provided
        if base_query.strip():
            query_parts.append(self.escape_query_term(base_query))
        
        # Apply all filters
        filter_parts = []
        
        if filters.entry_types:
            filter_parts.append(self.build_entry_type_filter(filters.entry_types))
        
        if filters.organisms:
            filter_parts.append(self.build_organism_filter(filters.organisms))
        
        if filters.platforms:
            filter_parts.append(self.build_platform_filter(filters.platforms))
        
        if filters.date_range:
            filter_parts.append(self.build_date_filter(filters.date_range))
        
        if filters.published_last_n_months:
            filter_parts.append(self.build_recent_filter(filters.published_last_n_months))
        
        if filters.supplementary_file_types:
            filter_parts.append(self.build_supplementary_filter(filters.supplementary_file_types))
        
        # Combine all parts
        if filter_parts:
            query_parts.extend(filter_parts)
        
        return self.combine_filters(query_parts)
    
    def build_entry_type_filter(self, entry_types: List[GEOEntryType]) -> str:
        """
        Build entry type filter (GSE, GDS, etc.).
        
        Examples from official API:
        - GSE[ETYP] - Series only
        - (GSE[ETYP] OR GDS[ETYP]) - Series or datasets
        
        Args:
            entry_types: List of entry types to include
            
        Returns:
            str: Entry type filter string
        """
        if not entry_types:
            return ""
        
        if len(entry_types) == 1:
            return f"{entry_types[0].value}[{GEOFieldTag.ENTRY_TYPE.value}]"
        
        # Multiple entry types - use OR logic
        type_filters = [f"{et.value}[{GEOFieldTag.ENTRY_TYPE.value}]" 
                       for et in entry_types]
        return f"({' OR '.join(type_filters)})"
    
    def build_organism_filter(self, organisms: List[str]) -> str:
        """
        Build organism filter with proper escaping.
        
        Examples from official API:
        - human[ORGN]
        - yeast[ORGN]
        - (human[ORGN] OR mouse[ORGN])
        
        Args:
            organisms: List of organism names
            
        Returns:
            str: Organism filter string
        """
        if not organisms:
            return ""
        
        # Clean and escape organism names
        cleaned_organisms = []
        for org in organisms:
            cleaned = self._clean_organism_name(org)
            if cleaned:
                cleaned_organisms.append(cleaned)
        
        if not cleaned_organisms:
            return ""
        
        if len(cleaned_organisms) == 1:
            return f"{cleaned_organisms[0]}[{GEOFieldTag.ORGANISM.value}]"
        
        # Multiple organisms - use OR logic
        org_filters = [f"{org}[{GEOFieldTag.ORGANISM.value}]" 
                      for org in cleaned_organisms]
        return f"({' OR '.join(org_filters)})"
    
    def build_platform_filter(self, platforms: List[str]) -> str:
        """
        Build platform accession filter.
        
        Examples from official API:
        - GPL96[ACCN]
        - (GPL96[ACCN] OR GPL570[ACCN])
        
        Args:
            platforms: List of platform accessions
            
        Returns:
            str: Platform filter string
        """
        if not platforms:
            return ""
        
        # Validate and clean platform accessions
        valid_platforms = []
        for platform in platforms:
            cleaned = self._clean_platform_accession(platform)
            if cleaned:
                valid_platforms.append(cleaned)
        
        if not valid_platforms:
            return ""
        
        if len(valid_platforms) == 1:
            return f"{valid_platforms[0]}[{GEOFieldTag.ACCESSION.value}]"
        
        # Multiple platforms - use OR logic
        platform_filters = [f"{platform}[{GEOFieldTag.ACCESSION.value}]" 
                           for platform in valid_platforms]
        return f"({' OR '.join(platform_filters)})"
    
    def build_date_filter(self, date_range: Dict[str, str]) -> str:
        """
        Build date range filter for publication dates.
        
        Examples from official API:
        - 2007/01:2007/03[PDAT] - Range format
        - 2023[PDAT] - Single year
        
        Args:
            date_range: Dictionary with 'start' and/or 'end' dates
                       Format: "YYYY/MM" or "YYYY"
            
        Returns:
            str: Date filter string
        """
        if not date_range:
            return ""
        
        start = date_range.get('start', '').strip()
        end = date_range.get('end', '').strip()
        
        # Validate date formats
        start = self._validate_date_format(start) if start else None
        end = self._validate_date_format(end) if end else None
        
        if not start and not end:
            return ""
        
        if start and end:
            return f"{start}:{end}[{GEOFieldTag.PUBLICATION_DATE.value}]"
        elif start:
            return f"{start}[{GEOFieldTag.PUBLICATION_DATE.value}]"
        elif end:
            # End date only - assume from beginning of time
            return f"1900:{end}[{GEOFieldTag.PUBLICATION_DATE.value}]"
        
        return ""
    
    def build_recent_filter(self, months: int) -> str:
        """
        Build filter for recently published datasets.
        
        Examples from official API:
        - "published last 3 months"[Filter]
        - "published last 12 months"[Filter]
        
        Args:
            months: Number of months back from current date
            
        Returns:
            str: Recent publication filter string
        """
        if not months or months <= 0:
            return ""
        
        return f'"published last {months} months"[Filter]'
    
    def build_supplementary_filter(self, file_types: List[str]) -> str:
        """
        Build supplementary file type filter.
        
        Examples from official API:
        - cel[suppFile]
        - (cel[suppFile] OR h5[suppFile])
        
        Args:
            file_types: List of file extensions (without dots)
            
        Returns:
            str: Supplementary file filter string
        """
        if not file_types:
            return ""
        
        # Clean file extensions
        cleaned_types = []
        for ftype in file_types:
            cleaned = ftype.strip().lower().replace('.', '')
            if cleaned and re.match(r'^[a-zA-Z0-9]+$', cleaned):
                cleaned_types.append(cleaned)
        
        if not cleaned_types:
            return ""
        
        if len(cleaned_types) == 1:
            return f"{cleaned_types[0]}[{GEOFieldTag.SUPPLEMENTARY_FILE.value}]"
        
        # Multiple file types - use OR logic
        file_filters = [f"{ftype}[{GEOFieldTag.SUPPLEMENTARY_FILE.value}]" 
                       for ftype in cleaned_types]
        return f"({' OR '.join(file_filters)})"
    
    def combine_filters(self, filters: List[str]) -> str:
        """
        Combine multiple filters with AND logic.
        
        Args:
            filters: List of filter strings to combine
            
        Returns:
            str: Combined query string
        """
        if not filters:
            return ""
        
        # Remove empty filters
        non_empty_filters = [f.strip() for f in filters if f.strip()]
        
        if not non_empty_filters:
            return ""
        
        if len(non_empty_filters) == 1:
            return non_empty_filters[0]
        
        return " AND ".join(non_empty_filters)
    
    def escape_query_term(self, term: str) -> str:
        """
        Properly escape special characters for NCBI queries.
        
        Args:
            term: Query term to escape
            
        Returns:
            str: Escaped query term
        """
        if not term:
            return ""
        
        # Remove leading/trailing whitespace
        term = term.strip()
        
        # If term contains spaces, wrap in quotes
        if ' ' in term:
            # Escape internal quotes
            term = term.replace('"', '\\"')
            term = f'"{term}"'
        
        return term
    
    def url_encode_query(self, query: str) -> str:
        """
        URL encode the final query for HTTP requests.
        
        Args:
            query: Query string to encode
            
        Returns:
            str: URL-encoded query string
        """
        return urllib.parse.quote(query)
    
    def validate_query_syntax(self, query: str) -> bool:
        """
        Validate that a query has proper NCBI syntax.
        
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
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count < 0:
                    return False
        
        if paren_count != 0:
            return False
        
        # Check for balanced quotes
        quote_count = query.count('"')
        if quote_count % 2 != 0:
            return False
        
        return True
    
    # Helper methods
    
    def _clean_organism_name(self, organism: str) -> Optional[str]:
        """Clean and standardize organism names."""
        if not organism:
            return None
        
        # Basic cleaning
        cleaned = organism.strip().lower()
        
        # Common organism name mappings
        organism_map = {
            'human': 'human',
            'homo sapiens': 'human',
            'h. sapiens': 'human',
            'mouse': 'mouse',
            'mus musculus': 'mouse',
            'm. musculus': 'mouse',
            'rat': 'rat',
            'rattus norvegicus': 'rat',
            'r. norvegicus': 'rat',
            'yeast': 'yeast',
            'saccharomyces cerevisiae': 'yeast',
            's. cerevisiae': 'yeast',
            'fly': 'fly',
            'drosophila melanogaster': 'fly',
            'd. melanogaster': 'fly',
            'worm': 'worm',
            'c. elegans': 'worm',
            'caenorhabditis elegans': 'worm'
        }
        
        return organism_map.get(cleaned, cleaned)
    
    def _clean_platform_accession(self, platform: str) -> Optional[str]:
        """Clean and validate platform accessions."""
        if not platform:
            return None
        
        cleaned = platform.strip().upper()
        
        # Check if it looks like a valid GPL accession
        if re.match(r'^GPL\d+$', cleaned):
            return cleaned
        
        # If it's just a number, add GPL prefix
        if re.match(r'^\d+$', cleaned):
            return f"GPL{cleaned}"
        
        # Return as-is for other formats
        return cleaned
    
    def _validate_date_format(self, date_str: str) -> Optional[str]:
        """Validate and normalize date format for NCBI."""
        if not date_str:
            return None
        
        # Remove any extra whitespace
        date_str = date_str.strip()
        
        # Check YYYY/MM format
        if re.match(r'^\d{4}/\d{1,2}$', date_str):
            year, month = date_str.split('/')
            month = month.zfill(2)  # Ensure 2-digit month
            return f"{year}/{month}"
        
        # Check YYYY format
        if re.match(r'^\d{4}$', date_str):
            return date_str
        
        # Try to parse other common formats
        try:
            # Try parsing as ISO date
            dt = datetime.fromisoformat(date_str.replace('-', '/'))
            return f"{dt.year:04d}/{dt.month:02d}"
        except:
            pass
        
        return None
    
    def get_example_queries(self) -> Dict[str, str]:
        """
        Get example queries matching the official NCBI documentation.
        
        Returns:
            Dict[str, str]: Dictionary of example names and their queries
        """
        return {
            "series_recent": 'GSE[ETYP] AND "published last 3 months"[Filter]',
            "organism_date_range": "yeast[ORGN] AND 2007/01:2007/03[PDAT]",
            "platform_with_files": "GPL96[ACCN] AND gse[ETYP] AND cel[suppFile]",
            "multi_organism": "(human[ORGN] OR mouse[ORGN]) AND gse[ETYP]",
            "keyword_with_filters": '"single-cell RNA-seq" AND human[ORGN] AND gse[ETYP] AND h5[suppFile]'
        }
