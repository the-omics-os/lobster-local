"""
Unified accession resolution system.

Single source of truth for all biobank identifier operations.
Uses DATABASE_ACCESSION_REGISTRY from database_mappings.py.

This module consolidates identifier validation and extraction that was
previously duplicated across 7+ provider files.

Example:
    >>> from lobster.core.identifiers import get_accession_resolver
    >>> resolver = get_accession_resolver()
    >>> resolver.detect_database("GSE12345")
    'GEO'
    >>> resolver.detect_database("PRJEB83385")
    'BioProject_ENA'
    >>> resolver.extract_all_accessions("Data deposited in GSE123 and SRP456")
    {'GEO': ['GSE123'], 'SRA_Study': ['SRP456']}
"""

import re
import threading
from typing import Dict, List, Optional, Set, Tuple

from lobster.core.schemas.database_mappings import DATABASE_ACCESSION_REGISTRY
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class AccessionResolver:
    """
    Centralized identifier resolution for all biobank accessions.

    Thread-safe singleton that pre-compiles regex patterns for performance.
    Uses DATABASE_ACCESSION_REGISTRY as the single source of truth.

    Attributes:
        _compiled_patterns: Dict of field_name -> compiled regex pattern
        _database_by_field: Dict of field_name -> human-readable database name
        _search_patterns: Dict for text extraction (without anchors)
    """

    def __init__(self):
        """Initialize resolver and compile patterns."""
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._search_patterns: Dict[str, re.Pattern] = {}
        self._database_by_field: Dict[str, str] = {}
        self._url_templates: Dict[str, str] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        for field_name, accession in DATABASE_ACCESSION_REGISTRY.items():
            # Full pattern with anchors for validation
            self._compiled_patterns[field_name] = re.compile(
                accession.full_pattern, re.IGNORECASE
            )

            # Search pattern without anchors for text extraction
            # Only strip ^ at start and $ at end to preserve [^...] character classes
            search_pattern = accession.full_pattern
            if search_pattern.startswith("^"):
                search_pattern = search_pattern[1:]
            if search_pattern.endswith("$"):
                search_pattern = search_pattern[:-1]
            self._search_patterns[field_name] = re.compile(
                search_pattern, re.IGNORECASE
            )

            # Map field to database name
            self._database_by_field[field_name] = accession.database_name

            # Store URL templates
            self._url_templates[field_name] = accession.database_url_template

        logger.debug(
            f"AccessionResolver initialized with {len(self._compiled_patterns)} patterns"
        )

    def detect_database(self, identifier: str) -> Optional[str]:
        """
        Detect which database an identifier belongs to.

        Args:
            identifier: Accession string (e.g., "GSE12345", "PRJEB83385", "PXD012345")

        Returns:
            Human-readable database name or None if not recognized

        Example:
            >>> resolver.detect_database("GSE12345")
            'NCBI Gene Expression Omnibus'
            >>> resolver.detect_database("unknown123")
            None
        """
        identifier = identifier.strip()

        for field_name, pattern in self._compiled_patterns.items():
            if pattern.match(identifier):
                return self._database_by_field[field_name]

        return None

    def detect_field(self, identifier: str) -> Optional[str]:
        """
        Detect the field name for an identifier.

        Args:
            identifier: Accession string

        Returns:
            Field name (e.g., 'geo_accession', 'pride_accession') or None

        Example:
            >>> resolver.detect_field("GSE12345")
            'geo_accession'
        """
        identifier = identifier.strip()

        for field_name, pattern in self._compiled_patterns.items():
            if pattern.match(identifier):
                return field_name

        return None

    def extract_all_accessions(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all database accessions from text.

        Uses search patterns (without anchors) to find accessions embedded
        in natural language text like abstracts or methods sections.

        Args:
            text: Text to search (e.g., methods section, abstract)

        Returns:
            Dict mapping database name to list of found accessions

        Example:
            >>> text = "Data available at GSE123456 and PRIDE PXD012345"
            >>> resolver.extract_all_accessions(text)
            {'NCBI Gene Expression Omnibus': ['GSE123456'],
             'ProteomeXchange/PRIDE': ['PXD012345']}
        """
        if not text:
            return {}

        results: Dict[str, Set[str]] = {}

        for field_name, pattern in self._search_patterns.items():
            matches = pattern.findall(text)

            if matches:
                db_name = self._database_by_field[field_name]
                if db_name not in results:
                    results[db_name] = set()
                # Normalize to uppercase for consistency
                results[db_name].update(m.upper() for m in matches)

        # Convert sets to sorted lists for deterministic output
        return {db: sorted(list(accs)) for db, accs in results.items()}

    def extract_accessions_by_type(self, text: str) -> Dict[str, List[str]]:
        """
        Extract accessions grouped by simple type names.

        Similar to extract_all_accessions but uses simplified type names
        compatible with existing provider code (GEO, SRA, PRIDE, etc.).

        Args:
            text: Text to search

        Returns:
            Dict with simplified keys like 'GEO', 'SRA', 'PRIDE', etc.

        Example:
            >>> resolver.extract_accessions_by_type("GSE123 and SRP456")
            {'GEO': ['GSE123'], 'SRA': ['SRP456']}
        """
        if not text:
            return {}

        # Mapping from field names to simplified type names
        type_mapping = {
            "geo_accession": "GEO",
            "geo_sample_accession": "GEO_Sample",
            "geo_platform_accession": "GEO_Platform",
            "geo_dataset_accession": "GEO_Dataset",
            "sra_study_accession": "SRA",
            "sra_experiment_accession": "SRA",
            "sra_run_accession": "SRA",
            "sra_sample_accession": "SRA",
            "ena_study_accession": "ENA",
            "ena_experiment_accession": "ENA",
            "ena_run_accession": "ENA",
            "ena_sample_accession": "ENA",
            "ddbj_study_accession": "DDBJ",
            "ddbj_experiment_accession": "DDBJ",
            "ddbj_run_accession": "DDBJ",
            "ddbj_sample_accession": "DDBJ",
            "bioproject_accession": "BioProject",
            "bioproject_ena_accession": "BioProject",
            "bioproject_ddbj_accession": "BioProject",
            "biosample_accession": "BioSample",
            "biosample_ena_accession": "BioSample",
            "biosample_ddbj_accession": "BioSample",
            "pride_accession": "PRIDE",
            "massive_accession": "MassIVE",
            "metabolights_accession": "MetaboLights",
            "metabolomics_workbench_accession": "MetabolomicsWorkbench",
            "arrayexpress_accession": "ArrayExpress",
            "mgnify_accession": "MGnify",
            "qiita_accession": "Qiita",
            "publication_doi": "DOI",
            # EGA (European Genome-phenome Archive) - controlled access
            "ega_study_accession": "EGA",
            "ega_dataset_accession": "EGA",
            "ega_sample_accession": "EGA",
            "ega_experiment_accession": "EGA",
            "ega_run_accession": "EGA",
            "ega_analysis_accession": "EGA",
            "ega_policy_accession": "EGA",
            "ega_dac_accession": "EGA",
        }

        results: Dict[str, Set[str]] = {}

        for field_name, pattern in self._search_patterns.items():
            matches = pattern.findall(text)

            if matches:
                type_name = type_mapping.get(field_name, field_name)
                if type_name not in results:
                    results[type_name] = set()
                results[type_name].update(m.upper() for m in matches)

        return {t: sorted(list(accs)) for t, accs in results.items()}

    def validate(self, identifier: str, database: Optional[str] = None) -> bool:
        """
        Validate identifier format.

        Args:
            identifier: Accession to validate
            database: Optional database name to validate against (e.g., "GEO", "SRA", "PRIDE")

        Returns:
            True if valid format

        Example:
            >>> resolver.validate("GSE12345")
            True
            >>> resolver.validate("GSE12345", database="GEO")
            True
            >>> resolver.validate("INVALID")
            False
        """
        identifier = identifier.strip()

        if database:
            database_lower = database.lower()

            # Map simplified names to database name patterns
            simplified_mappings = {
                "geo": ["gene expression omnibus"],
                "sra": ["sequence read archive", "sra"],
                "ena": ["ena"],
                "ddbj": ["ddbj"],
                "pride": ["pride", "proteomexchange"],
                "massive": ["massive"],
                "bioproject": ["bioproject"],
                "biosample": ["biosample"],
                "metabolights": ["metabolights"],
                "arrayexpress": ["arrayexpress"],
                "mgnify": ["mgnify"],
            }

            # Get target patterns for this database
            target_patterns = simplified_mappings.get(database_lower, [database_lower])

            # Validate against matching fields
            for field_name, accession in DATABASE_ACCESSION_REGISTRY.items():
                db_name = accession.database_name.lower()
                field_base = field_name.replace("_accession", "")

                # Check if this field matches the requested database
                matches = (
                    any(p in db_name for p in target_patterns)
                    or database_lower in db_name
                    or database_lower == field_base
                )

                if matches and self._compiled_patterns[field_name].match(identifier):
                    return True
            return False
        else:
            # Validate against any database
            return self.detect_database(identifier) is not None

    def get_url(self, identifier: str) -> Optional[str]:
        """
        Generate database URL for an accession.

        Args:
            identifier: Accession value

        Returns:
            Database URL or None if invalid

        Example:
            >>> resolver.get_url("GSE12345")
            'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE12345'
        """
        identifier = identifier.strip()
        field_name = self.detect_field(identifier)

        if field_name and field_name in self._url_templates:
            return self._url_templates[field_name].format(accession=identifier)

        return None

    def normalize_identifier(self, identifier: str) -> str:
        """
        Normalize identifier to canonical format.

        - Strips whitespace
        - Uppercases prefix (GSE, PRJNA, PXD, etc.)
        - Preserves numeric suffix

        Args:
            identifier: Raw identifier string

        Returns:
            Normalized identifier

        Example:
            >>> resolver.normalize_identifier("  gse12345  ")
            'GSE12345'
        """
        identifier = identifier.strip()

        # Common prefix patterns to uppercase
        prefixes = [
            "GSE",
            "GSM",
            "GPL",
            "GDS",
            "PRJNA",
            "PRJEB",
            "PRJDB",
            "SAMN",
            "SAME",
            "SAMD",
            "SRP",
            "SRX",
            "SRR",
            "SRS",
            "ERP",
            "ERX",
            "ERR",
            "ERS",
            "DRP",
            "DRX",
            "DRR",
            "DRS",
            "PXD",
            "MSV",
            "MTBLS",
            "MGYS",
            "ST",
            "E-",
            # EGA prefixes (European Genome-phenome Archive)
            "EGAS",
            "EGAD",
            "EGAN",
            "EGAX",
            "EGAR",
            "EGAZ",
            "EGAP",
            "EGAC",
        ]

        for prefix in prefixes:
            if identifier.upper().startswith(prefix.upper()):
                return prefix.upper() + identifier[len(prefix) :]

        return identifier

    def get_supported_databases(self) -> List[str]:
        """
        Get list of all supported database names.

        Returns:
            Sorted list of unique database names
        """
        return sorted(list(set(self._database_by_field.values())))

    def get_supported_types(self) -> List[str]:
        """
        Get list of simplified type names.

        Returns:
            List of type names like 'GEO', 'SRA', 'PRIDE', 'EGA', etc.
        """
        return [
            "GEO",
            "GEO_Sample",
            "GEO_Platform",
            "GEO_Dataset",
            "SRA",
            "ENA",
            "DDBJ",
            "BioProject",
            "BioSample",
            "PRIDE",
            "MassIVE",
            "MetaboLights",
            "ArrayExpress",
            "MGnify",
            "EGA",
            "DOI",
        ]

    def is_geo_identifier(self, identifier: str) -> bool:
        """Check if identifier is any GEO type (GSE, GSM, GPL, GDS)."""
        db = self.detect_database(identifier)
        return db is not None and "Gene Expression Omnibus" in db

    def is_sra_identifier(self, identifier: str) -> bool:
        """Check if identifier is any SRA/ENA/DDBJ type."""
        db = self.detect_database(identifier)
        return db is not None and any(
            x in db for x in ["Sequence Read Archive", "ENA", "DDBJ"]
        )

    def is_proteomics_identifier(self, identifier: str) -> bool:
        """Check if identifier is PRIDE or MassIVE."""
        db = self.detect_database(identifier)
        return db is not None and any(x in db for x in ["PRIDE", "MassIVE"])

    def is_ega_identifier(self, identifier: str) -> bool:
        """Check if identifier is any EGA type (EGAS, EGAD, EGAN, etc.)."""
        db = self.detect_database(identifier)
        return db is not None and "Genome-phenome Archive" in db

    def get_access_type(self, identifier: str) -> str:
        """
        Get access type for an identifier.

        Args:
            identifier: Accession string

        Returns:
            Access type: "open", "controlled", "embargoed", or "unknown"

        Example:
            >>> resolver.get_access_type("GSE12345")
            'open'
            >>> resolver.get_access_type("EGAD50000000740")
            'controlled'
        """
        identifier = identifier.strip()
        field_name = self.detect_field(identifier)

        if field_name and field_name in DATABASE_ACCESSION_REGISTRY:
            return DATABASE_ACCESSION_REGISTRY[field_name].access_type

        return "unknown"

    def is_controlled_access(self, identifier: str) -> bool:
        """
        Check if identifier requires controlled access application.

        Args:
            identifier: Accession string

        Returns:
            True if access_type is "controlled"

        Example:
            >>> resolver.is_controlled_access("EGAD50000000740")
            True
            >>> resolver.is_controlled_access("GSE12345")
            False
        """
        return self.get_access_type(identifier) == "controlled"

    def get_access_notes(self, identifier: str) -> str:
        """
        Get access notes for an identifier (instructions for controlled access).

        Args:
            identifier: Accession string

        Returns:
            Access notes string, empty if not applicable
        """
        identifier = identifier.strip()
        field_name = self.detect_field(identifier)

        if field_name and field_name in DATABASE_ACCESSION_REGISTRY:
            return DATABASE_ACCESSION_REGISTRY[field_name].access_notes

        return ""

    def extract_accessions_with_metadata(self, text: str) -> List[Dict]:
        """
        Extract accessions from text with full metadata including access_type.

        Args:
            text: Text to search (e.g., methods section, abstract)

        Returns:
            List of dicts with accession details:
            - accession: Normalized accession string
            - field_name: Registry field name
            - database: Human-readable database name
            - access_type: "open", "controlled", "embargoed"
            - access_notes: Instructions for controlled access

        Example:
            >>> results = resolver.extract_accessions_with_metadata(
            ...     "Data available at GSE12345 and EGAD50000000740"
            ... )
            >>> for r in results:
            ...     print(f"{r['accession']}: {r['access_type']}")
            GSE12345: open
            EGAD50000000740: controlled
        """
        if not text:
            return []

        results = []
        seen = set()

        for field_name, pattern in self._search_patterns.items():
            matches = pattern.findall(text)

            if matches:
                accession_info = DATABASE_ACCESSION_REGISTRY[field_name]
                for match in matches:
                    normalized = match.upper()
                    if normalized in seen:
                        continue
                    seen.add(normalized)

                    results.append(
                        {
                            "accession": normalized,
                            "field_name": field_name,
                            "database": accession_info.database_name,
                            "access_type": accession_info.access_type,
                            "access_notes": accession_info.access_notes,
                        }
                    )

        return results


# Thread-safe singleton implementation
_resolver: Optional[AccessionResolver] = None
_resolver_lock = threading.Lock()


def get_accession_resolver() -> AccessionResolver:
    """
    Get singleton AccessionResolver instance.

    Thread-safe lazy initialization.

    Returns:
        AccessionResolver singleton instance
    """
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            # Double-checked locking
            if _resolver is None:
                _resolver = AccessionResolver()
    return _resolver


def reset_resolver() -> None:
    """
    Reset the singleton resolver (for testing).

    Warning: Only use in tests to ensure fresh state.
    """
    global _resolver
    with _resolver_lock:
        _resolver = None
