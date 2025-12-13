"""
PRIDE API Response Normalizer.

Handles inconsistent data types from PRIDE REST API v2 responses.
The PRIDE Archive API returns heterogeneous data structures (dict/string/list)
for the same fields across different datasets, particularly for legacy data.

This normalizer provides defensive type handling to ensure robust parsing
regardless of API response variations.

Key fields normalized:
- organisms: List[Dict] or List[str] → List[Dict[str, str]]
- references: List[Dict] or List[str] → List[Dict[str, Any]]
- submitters/labPIs: List[Dict] or List[str] → List[Dict[str, str]]
- publicFileLocations: List[Dict] or str → Dict[str, str]

Design inspired by wfondrie/ppx listify() utility.
"""

from typing import Any, Dict, List, Optional, Union

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PRIDENormalizer:
    """
    Normalize PRIDE API responses to consistent data structures.

    The PRIDE REST API v2 returns inconsistent data types for certain fields,
    particularly in legacy datasets (pre-2020). This normalizer ensures all
    responses conform to predictable structures, preventing crashes from
    unexpected type variations.

    Usage:
        # Normalize single project
        normalized_project = PRIDENormalizer.normalize_project(raw_project)

        # Normalize search results
        normalized_results = PRIDENormalizer.normalize_search_results(raw_results)
    """

    @staticmethod
    def normalize_organisms(field: Any) -> List[Dict[str, str]]:
        """
        Normalize organisms field to list of dicts with 'name' key.

        The PRIDE API returns organisms in multiple formats:
        - List[Dict]: [{"name": "Homo sapiens"}, ...]  (v3 format)
        - List[str]: ["Homo sapiens", ...]             (v2 format)
        - Dict: {"name": "Homo sapiens"}               (single organism)
        - str: "Homo sapiens"                          (legacy format)
        - None or []                                   (no organism data)

        Args:
            field: Raw organisms field from PRIDE API

        Returns:
            List[Dict[str, str]]: Normalized list of organism dicts
                Each dict has at minimum a 'name' key

        Example:
            >>> PRIDENormalizer.normalize_organisms("Homo sapiens")
            [{"name": "Homo sapiens"}]
            >>> PRIDENormalizer.normalize_organisms(["Human", "Mouse"])
            [{"name": "Human"}, {"name": "Mouse"}]
        """
        if field is None:
            return []

        # Case 1: Already a list
        if isinstance(field, list):
            if not field:  # Empty list
                return []

            normalized = []
            for org in field:
                if isinstance(org, dict):
                    # Already in dict format, keep as-is
                    normalized.append(org)
                elif isinstance(org, str):
                    # String in list: convert to dict
                    normalized.append({"name": org})
                else:
                    logger.warning(
                        f"Unexpected organism list item type: {type(org).__name__}"
                    )
            return normalized

        # Case 2: Single dict
        elif isinstance(field, dict):
            return [field]

        # Case 3: Single string
        elif isinstance(field, str):
            return [{"name": field}]

        # Case 4: Unknown type
        else:
            logger.warning(
                f"Unexpected organisms field type: {type(field).__name__}, "
                f"value: {str(field)[:100]}"
            )
            return []

    @staticmethod
    def normalize_references(field: Any) -> List[Dict[str, Any]]:
        """
        Normalize references field to list of dicts with publication metadata.

        The PRIDE API returns references in multiple formats:
        - List[Dict]: [{"pubmedId": "12345", "doi": "10.1234/..."}, ...]
        - List[str]: ["PMID:12345", "DOI:10.1234/..."]
        - Dict: {"pubmedId": "12345", ...}
        - str: "PMID:12345" or "10.1234/..."
        - None or []

        Args:
            field: Raw references field from PRIDE API

        Returns:
            List[Dict[str, Any]]: Normalized list of reference dicts
                Each dict may contain: pubmedId, doi, referenceLine

        Example:
            >>> PRIDENormalizer.normalize_references("PMID:12345")
            [{"id": "PMID:12345"}]
            >>> PRIDENormalizer.normalize_references([{"doi": "10.1234/abc"}])
            [{"doi": "10.1234/abc"}]
        """
        if field is None:
            return []

        # Case 1: Already a list
        if isinstance(field, list):
            if not field:
                return []

            normalized = []
            for ref in field:
                if isinstance(ref, dict):
                    # Already in dict format
                    normalized.append(ref)
                elif isinstance(ref, str):
                    # String reference: create minimal dict with 'id' field
                    normalized.append({"id": ref})
                else:
                    logger.warning(
                        f"Unexpected reference list item type: {type(ref).__name__}"
                    )
            return normalized

        # Case 2: Single dict
        elif isinstance(field, dict):
            return [field]

        # Case 3: Single string (DOI/PMID/citation)
        elif isinstance(field, str):
            return [{"id": field}]

        # Case 4: Unknown type
        else:
            logger.warning(
                f"Unexpected references field type: {type(field).__name__}, "
                f"value: {str(field)[:100]}"
            )
            return []

    @staticmethod
    def normalize_people(field: Any) -> List[Dict[str, str]]:
        """
        Normalize submitters/labPIs fields to list of dicts with name components.

        The PRIDE API returns people fields in multiple formats:
        - List[Dict]: [{"firstName": "John", "lastName": "Doe"}, ...]
        - List[str]: ["John Doe", ...]
        - Dict: {"firstName": "John", "lastName": "Doe"}
        - str: "John Doe"
        - None or []

        Args:
            field: Raw submitters or labPIs field from PRIDE API

        Returns:
            List[Dict[str, str]]: Normalized list of people dicts
                Each dict may contain: firstName, lastName, fullName

        Example:
            >>> PRIDENormalizer.normalize_people("John Doe")
            [{"fullName": "John Doe"}]
            >>> PRIDENormalizer.normalize_people([{"firstName": "John"}])
            [{"firstName": "John"}]
        """
        if field is None:
            return []

        # Case 1: Already a list
        if isinstance(field, list):
            if not field:
                return []

            normalized = []
            for person in field:
                if isinstance(person, dict):
                    # Already in dict format
                    normalized.append(person)
                elif isinstance(person, str):
                    # String name: create dict with fullName field
                    normalized.append({"fullName": person})
                else:
                    logger.warning(
                        f"Unexpected person list item type: {type(person).__name__}"
                    )
            return normalized

        # Case 2: Single dict
        elif isinstance(field, dict):
            return [field]

        # Case 3: Single string (full name)
        elif isinstance(field, str):
            return [{"fullName": field}]

        # Case 4: Unknown type
        else:
            logger.warning(
                f"Unexpected people field type: {type(field).__name__}, "
                f"value: {str(field)[:100]}"
            )
            return []

    @staticmethod
    def normalize_file_locations(field: Any) -> List[Dict[str, str]]:
        """
        Normalize publicFileLocations field to list of location dicts.

        The PRIDE API returns file locations in multiple formats:
        - List[Dict]: [{"name": "FTP Protocol", "value": "ftp://..."}, ...]
        - List[str]: ["ftp://...", ...]
        - Dict: {"name": "FTP Protocol", "value": "ftp://..."}
        - str: "ftp://..."
        - None or []

        Args:
            field: Raw publicFileLocations field from PRIDE API

        Returns:
            List[Dict[str, str]]: Normalized list with 'name' and 'value' keys
                Protocols include: "FTP Protocol", "Aspera Protocol", "S3 Protocol"

        Example:
            >>> PRIDENormalizer.normalize_file_locations("ftp://example.com")
            [{"name": "FTP Protocol", "value": "ftp://example.com"}]
            >>> PRIDENormalizer.normalize_file_locations([{"name": "FTP", "value": "..."}])
            [{"name": "FTP", "value": "..."}]
        """
        if field is None:
            return []

        # Case 1: Already a list
        if isinstance(field, list):
            if not field:
                return []

            normalized = []
            for loc in field:
                if isinstance(loc, dict):
                    # Already in dict format
                    normalized.append(loc)
                elif isinstance(loc, str):
                    # String URL: infer protocol from URL scheme
                    protocol = "Unknown Protocol"
                    if loc.startswith("ftp://"):
                        protocol = "FTP Protocol"
                    elif loc.startswith("http://") or loc.startswith("https://"):
                        protocol = "HTTP Protocol"
                    elif loc.startswith("s3://"):
                        protocol = "S3 Protocol"

                    normalized.append({"name": protocol, "value": loc})
                else:
                    logger.warning(
                        f"Unexpected location list item type: {type(loc).__name__}"
                    )
            return normalized

        # Case 2: Single dict
        elif isinstance(field, dict):
            return [field]

        # Case 3: Single string (direct URL)
        elif isinstance(field, str):
            # Infer protocol from URL scheme
            protocol = "Unknown Protocol"
            if field.startswith("ftp://"):
                protocol = "FTP Protocol"
            elif field.startswith("http://") or field.startswith("https://"):
                protocol = "HTTP Protocol"
            elif field.startswith("s3://"):
                protocol = "S3 Protocol"

            return [{"name": protocol, "value": field}]

        # Case 4: Unknown type
        else:
            logger.warning(
                f"Unexpected publicFileLocations field type: {type(field).__name__}, "
                f"value: {str(field)[:100]}"
            )
            return []

    @classmethod
    def normalize_project(cls, project: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize all fields in a PRIDE project dictionary.

        Applies type normalization to known problematic fields while preserving
        all other fields unchanged. This ensures defensive parsing without
        losing any data from the API response.

        Args:
            project: Raw PRIDE project dict from API

        Returns:
            Dict[str, Any]: Normalized project dict with consistent data types

        Example:
            >>> raw_project = {"organisms": "Human", "references": [{"doi": "..."}]}
            >>> normalized = PRIDENormalizer.normalize_project(raw_project)
            >>> normalized["organisms"]
            [{"name": "Human"}]
        """
        if not isinstance(project, dict):
            logger.error(
                f"normalize_project expects dict, got {type(project).__name__}"
            )
            return {}

        # Create shallow copy to avoid mutating input
        normalized = project.copy()

        # Apply normalizers to known problematic fields
        if "organisms" in normalized:
            normalized["organisms"] = cls.normalize_organisms(normalized["organisms"])

        if "references" in normalized:
            normalized["references"] = cls.normalize_references(
                normalized["references"]
            )

        if "submitters" in normalized:
            normalized["submitters"] = cls.normalize_people(normalized["submitters"])

        if "labPIs" in normalized:
            normalized["labPIs"] = cls.normalize_people(normalized["labPIs"])

        # Note: publicFileLocations typically appears in file metadata, not project metadata
        # But normalize it defensively if present
        if "publicFileLocations" in normalized:
            normalized["publicFileLocations"] = cls.normalize_file_locations(
                normalized["publicFileLocations"]
            )

        return normalized

    @classmethod
    def normalize_search_results(
        cls, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalize all projects in PRIDE search results.

        Applies project-level normalization to each result. Safe to call
        on empty lists or malformed results.

        Args:
            results: List of raw project dicts from PRIDE search API

        Returns:
            List[Dict[str, Any]]: List of normalized project dicts

        Example:
            >>> raw_results = [{"organisms": "Human"}, {"organisms": ["Mouse"]}]
            >>> normalized = PRIDENormalizer.normalize_search_results(raw_results)
            >>> all(isinstance(p["organisms"], list) for p in normalized)
            True
        """
        if not isinstance(results, list):
            logger.error(
                f"normalize_search_results expects list, got {type(results).__name__}"
            )
            return []

        return [cls.normalize_project(proj) for proj in results if isinstance(proj, dict)]

    @classmethod
    def normalize_file_metadata(
        cls, files: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalize file metadata list from PRIDE project files endpoint.

        Each file dict may have publicFileLocations that need normalization.

        Args:
            files: List of file metadata dicts

        Returns:
            List[Dict[str, Any]]: Files with normalized publicFileLocations

        Example:
            >>> files = [{"fileName": "data.raw", "publicFileLocations": "ftp://..."}]
            >>> normalized = PRIDENormalizer.normalize_file_metadata(files)
            >>> normalized[0]["publicFileLocations"]
            [{"name": "FTP Protocol", "value": "ftp://..."}]
        """
        if not isinstance(files, list):
            logger.error(
                f"normalize_file_metadata expects list, got {type(files).__name__}"
            )
            return []

        normalized_files = []
        for file_dict in files:
            if not isinstance(file_dict, dict):
                logger.warning(f"Skipping non-dict file entry: {type(file_dict).__name__}")
                continue

            # Normalize publicFileLocations if present
            file_copy = file_dict.copy()
            if "publicFileLocations" in file_copy:
                file_copy["publicFileLocations"] = cls.normalize_file_locations(
                    file_copy["publicFileLocations"]
                )

            normalized_files.append(file_copy)

        return normalized_files

    @staticmethod
    def safe_get_organism_name(org: Union[Dict, str]) -> str:
        """
        Safely extract organism name from normalized organism dict or string.

        Helper method for extracting display names after normalization.

        Args:
            org: Normalized organism (dict with 'name' key or string)

        Returns:
            str: Organism name or "Unknown"

        Example:
            >>> PRIDENormalizer.safe_get_organism_name({"name": "Human"})
            "Human"
            >>> PRIDENormalizer.safe_get_organism_name("Mouse")
            "Mouse"
        """
        if isinstance(org, dict):
            return org.get("name", "Unknown")
        elif isinstance(org, str):
            return org
        else:
            return "Unknown"

    @staticmethod
    def safe_get_person_name(person: Union[Dict, str]) -> str:
        """
        Safely extract person name from normalized person dict or string.

        Handles multiple name formats:
        - Dict with firstName/lastName
        - Dict with fullName
        - String with full name

        Args:
            person: Normalized person (dict or string)

        Returns:
            str: Person name or "Unknown"

        Example:
            >>> PRIDENormalizer.safe_get_person_name({"firstName": "John", "lastName": "Doe"})
            "John Doe"
            >>> PRIDENormalizer.safe_get_person_name({"fullName": "Jane Smith"})
            "Jane Smith"
        """
        if isinstance(person, dict):
            # Try fullName first (normalized format)
            if "fullName" in person:
                return person["fullName"]

            # Try firstName/lastName (API format)
            first = person.get("firstName", "")
            last = person.get("lastName", "")
            full_name = f"{first} {last}".strip()
            return full_name if full_name else "Unknown"

        elif isinstance(person, str):
            return person

        else:
            return "Unknown"

    @staticmethod
    def extract_ftp_url(locations: List[Dict[str, str]]) -> Optional[str]:
        """
        Extract FTP URL from normalized publicFileLocations.

        Searches for FTP Protocol in priority order. Returns None if not found.

        Args:
            locations: Normalized publicFileLocations list

        Returns:
            Optional[str]: FTP URL or None

        Example:
            >>> locs = [{"name": "FTP Protocol", "value": "ftp://example.com"}]
            >>> PRIDENormalizer.extract_ftp_url(locs)
            "ftp://example.com"
        """
        if not isinstance(locations, list):
            return None

        # Protocol priority: FTP > HTTP > S3
        priority = ["FTP Protocol", "HTTP Protocol", "HTTPS Protocol", "S3 Protocol"]

        for protocol in priority:
            for loc in locations:
                if isinstance(loc, dict) and loc.get("name") == protocol:
                    url = loc.get("value")
                    if url:
                        return url

        # Fallback: return first available URL
        for loc in locations:
            if isinstance(loc, dict):
                url = loc.get("value")
                if url:
                    return url

        return None
