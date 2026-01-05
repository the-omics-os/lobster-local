"""
Identifier Provenance Service - validates dataset ownership and access control.

Implements a 2-layer validation system:
1. Section-based extraction (Data Availability vs Methods/Body)
2. E-Link validation (BioProject → PMID link)

This service addresses the critical issue where extracted identifiers from
referenced studies were treated the same as the study's own data, leading to:
- Wrong datasets being queued for download
- Incorrect metadata being associated with publications

Example use case (PMC11441152):
- EGAD50000000740 in Data Availability → primary (study's own data)
- PRJNA436359 in Methods → referenced (different study's data)

Usage:
    >>> from lobster.services.metadata.identifier_provenance_service import (
    ...     IdentifierProvenanceService
    ... )
    >>> from lobster.tools.providers.pubmed_provider import PubMedProvider
    >>>
    >>> service = IdentifierProvenanceService(PubMedProvider())
    >>> results = service.extract_and_validate(
    ...     full_text="...",
    ...     data_availability_text="Data deposited at EGAD50000000740",
    ...     source_pmid="39380095",
    ... )
    >>> for r in results:
    ...     print(f"{r.accession}: {r.provenance} ({r.access_type})")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lobster.core.identifiers.accession_resolver import get_accession_resolver
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IdentifierWithProvenance:
    """
    Identifier with provenance and access metadata.

    Attributes:
        accession: Normalized accession string (e.g., "EGAD50000000740")
        database: Human-readable database name
        field_name: Registry field name (e.g., "ega_dataset_accession")
        access_type: "open", "controlled", or "embargoed"
        source_section: Where identifier was found: "data_availability", "methods_or_body"
        provenance: "primary", "referenced", or "uncertain"
        confidence: 0.0-1.0 confidence score
        is_downloadable: True if can be auto-downloaded (open + primary)
        linked_pmid: PMID that bioproject links to (if E-Link validated)
        validation_notes: Additional notes (access instructions, errors)
    """

    accession: str
    database: str
    field_name: str
    access_type: str  # "open", "controlled", "embargoed"
    source_section: str  # "data_availability", "methods_or_body"
    provenance: str  # "primary", "referenced", "uncertain"
    confidence: float  # 0.0-1.0
    is_downloadable: bool  # access_type=="open" AND provenance=="primary"
    linked_pmid: Optional[str] = None
    validation_notes: str = ""


class IdentifierProvenanceService:
    """
    Service to validate identifier provenance and access control.

    Two-layer validation:
    1. Section-based: Data Availability → primary, Methods/Body → uncertain
    2. E-Link validation: For non-DA BioProjects, check NCBI linkage

    Downloadability logic (strict):
    - is_downloadable = (access_type == "open") AND (provenance == "primary")
    - Uncertain identifiers are excluded by default (user's preference)
    """

    def __init__(self, pubmed_provider=None):
        """
        Initialize the provenance service.

        Args:
            pubmed_provider: Optional PubMedProvider instance for E-Link validation.
                            If not provided, E-Link validation will be skipped.
        """
        self.pubmed_provider = pubmed_provider
        self.resolver = get_accession_resolver()

    def extract_and_validate(
        self,
        full_text: str,
        data_availability_text: str = "",
        source_pmid: Optional[str] = None,
        validate_elink: bool = True,
    ) -> List[IdentifierWithProvenance]:
        """
        Extract identifiers with provenance validation.

        Args:
            full_text: Complete article text
            data_availability_text: Text from Data Availability section only
            source_pmid: PMID of source publication (for E-Link validation)
            validate_elink: Whether to perform E-Link validation (API calls)

        Returns:
            List of IdentifierWithProvenance objects, sorted by confidence (highest first)

        Validation Strategy:
        1. Extract all identifiers from full text
        2. Extract identifiers from Data Availability section
        3. Mark DA identifiers as "primary" (confidence=0.95)
        4. For non-DA BioProjects with source_pmid, run E-Link validation
        5. Determine downloadability based on access_type + provenance
        """
        results = []

        # Layer 1: Section-based extraction
        da_identifiers = (
            self.resolver.extract_accessions_with_metadata(data_availability_text)
            if data_availability_text
            else []
        )
        all_identifiers = self.resolver.extract_accessions_with_metadata(full_text)

        # Mark identifiers found in Data Availability
        da_accessions = {i["accession"] for i in da_identifiers}

        logger.debug(
            f"Extracted {len(all_identifiers)} total identifiers, "
            f"{len(da_accessions)} from Data Availability section"
        )

        # Process all identifiers
        seen = set()
        for identifier in all_identifiers:
            accession = identifier["accession"]
            if accession in seen:
                continue
            seen.add(accession)

            field_name = identifier["field_name"]
            access_type = identifier["access_type"]

            # Determine source section and base provenance
            if accession in da_accessions:
                source_section = "data_availability"
                base_confidence = 0.95
                provenance = "primary"  # High confidence for DA section
            else:
                source_section = "methods_or_body"
                base_confidence = 0.3
                provenance = "uncertain"

            linked_pmid = None
            validation_notes = identifier.get("access_notes", "")

            # Layer 2: E-Link validation for BioProject identifiers
            # Only validate non-DA BioProjects (DA already has high confidence)
            if (
                validate_elink
                and self.pubmed_provider
                and source_pmid
                and field_name.startswith("bioproject")
                and source_section != "data_availability"
            ):
                logger.debug(
                    f"Running E-Link validation for {accession} → PMID:{source_pmid}"
                )
                try:
                    elink_result = (
                        self.pubmed_provider.validate_bioproject_publication_link(
                            accession, source_pmid
                        )
                    )
                    provenance = elink_result["provenance"]
                    base_confidence = elink_result["confidence"]

                    if elink_result.get("linked_pmids"):
                        linked_pmid = elink_result["linked_pmids"][0]

                    if elink_result.get("error"):
                        validation_notes = elink_result["error"]

                except Exception as e:
                    logger.warning(f"E-Link validation failed for {accession}: {e}")
                    validation_notes = f"E-Link validation failed: {e}"

            # Determine downloadability (STRICT: only primary + open)
            is_downloadable = access_type == "open" and provenance == "primary"

            results.append(
                IdentifierWithProvenance(
                    accession=accession,
                    database=identifier["database"],
                    field_name=field_name,
                    access_type=access_type,
                    source_section=source_section,
                    provenance=provenance,
                    confidence=base_confidence,
                    is_downloadable=is_downloadable,
                    linked_pmid=linked_pmid,
                    validation_notes=validation_notes,
                )
            )

        # Sort by confidence (highest first)
        results.sort(key=lambda x: x.confidence, reverse=True)

        # Log summary
        primary_count = sum(1 for r in results if r.provenance == "primary")
        referenced_count = sum(1 for r in results if r.provenance == "referenced")
        controlled_count = sum(1 for r in results if r.access_type == "controlled")
        downloadable_count = sum(1 for r in results if r.is_downloadable)

        logger.debug(
            f"Identifier provenance validation: {len(results)} total, "
            f"{primary_count} primary, {referenced_count} referenced, "
            f"{controlled_count} controlled-access, {downloadable_count} downloadable"
        )

        return results

    def filter_downloadable(
        self, identifiers: List[IdentifierWithProvenance]
    ) -> List[IdentifierWithProvenance]:
        """
        Filter to only downloadable identifiers.

        Args:
            identifiers: List of validated identifiers

        Returns:
            List of identifiers where is_downloadable=True
        """
        return [i for i in identifiers if i.is_downloadable]

    def filter_controlled_access(
        self, identifiers: List[IdentifierWithProvenance]
    ) -> List[IdentifierWithProvenance]:
        """
        Filter to only controlled-access identifiers.

        Args:
            identifiers: List of validated identifiers

        Returns:
            List of identifiers where access_type="controlled"
        """
        return [i for i in identifiers if i.access_type == "controlled"]

    def filter_by_provenance(
        self, identifiers: List[IdentifierWithProvenance], provenance: str
    ) -> List[IdentifierWithProvenance]:
        """
        Filter identifiers by provenance type.

        Args:
            identifiers: List of validated identifiers
            provenance: "primary", "referenced", or "uncertain"

        Returns:
            List of identifiers matching the provenance type
        """
        return [i for i in identifiers if i.provenance == provenance]

    def get_user_notification(
        self, identifiers: List[IdentifierWithProvenance]
    ) -> Optional[str]:
        """
        Generate user notification for controlled-access and referenced identifiers.

        Args:
            identifiers: List of validated identifiers

        Returns:
            User-friendly notification string, or None if no notification needed
        """
        controlled = self.filter_controlled_access(identifiers)
        referenced = self.filter_by_provenance(identifiers, "referenced")

        parts = []

        if controlled:
            controlled_ids = ", ".join(i.accession for i in controlled)
            parts.append(
                f"Found {len(controlled)} controlled-access identifier(s) "
                f"(require DAC application): {controlled_ids}"
            )

        if referenced:
            referenced_ids = ", ".join(i.accession for i in referenced)
            parts.append(
                f"Excluded {len(referenced)} referenced identifier(s) "
                f"(from other studies): {referenced_ids}"
            )

        return "\n".join(parts) if parts else None

    def to_dict_list(self, identifiers: List[IdentifierWithProvenance]) -> List[Dict]:
        """
        Convert identifiers to list of dictionaries for serialization.

        Args:
            identifiers: List of validated identifiers

        Returns:
            List of dicts suitable for JSON serialization
        """
        return [
            {
                "accession": i.accession,
                "database": i.database,
                "field_name": i.field_name,
                "access_type": i.access_type,
                "source_section": i.source_section,
                "provenance": i.provenance,
                "confidence": i.confidence,
                "is_downloadable": i.is_downloadable,
                "linked_pmid": i.linked_pmid,
                "validation_notes": i.validation_notes,
            }
            for i in identifiers
        ]
