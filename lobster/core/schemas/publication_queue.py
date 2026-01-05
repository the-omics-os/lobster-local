"""
Publication queue schema definitions for publication extraction orchestration.

This module defines Pydantic schemas for managing publication queue entries
with full metadata, identifier extraction, and processing status prepared
by the research_agent.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PublicationStatus(str, Enum):
    """Status of a publication queue entry."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    METADATA_EXTRACTED = "metadata_extracted"
    METADATA_ENRICHED = "metadata_enriched"
    HANDOFF_READY = "handoff_ready"
    COMPLETED = "completed"
    FAILED = "failed"
    PAYWALLED = "paywalled"  # Paper behind paywall, awaiting manual input

    @property
    def display(self) -> tuple:
        """
        Get display (icon, style) tuple for this status.

        Returns:
            Tuple of (icon: str, style: str) for UI rendering.
            Style uses Rich markup format.
        """
        return PUBLICATION_STATUS_DISPLAY.get(self.value, ("?", "dim"))

    @property
    def icon(self) -> str:
        """Get the display icon for this status."""
        return self.display[0]

    @property
    def style(self) -> str:
        """Get the Rich style for this status."""
        return self.display[1]


# Display configuration for PublicationStatus (Single Source of Truth)
# Maps status value -> (icon, style) for consistent UI rendering
PUBLICATION_STATUS_DISPLAY: Dict[str, tuple] = {
    "pending": ("○", "dim"),
    "extracting": ("◐", "#CC2C18"),
    "metadata_extracted": ("◑", "dim #CC2C18"),
    "metadata_enriched": ("◕", "#CC2C18"),
    "handoff_ready": ("→", "bold #CC2C18"),
    "completed": ("✓", "green"),
    "failed": ("✗", "red"),
    "paywalled": ("⚠", "yellow"),
}


class ExtractionLevel(str, Enum):
    """Level of content extraction for publications."""

    ABSTRACT = "abstract"  # Fast, PubMed API only
    METHODS = "methods"  # PMC full-text methods section
    FULL_TEXT = "full_text"  # Complete PMC full-text
    IDENTIFIERS = "identifiers"  # Extract dataset identifiers only


class HandoffStatus(str, Enum):
    """Granular status for agent-to-agent handoffs."""

    NOT_READY = "not_ready"
    READY_FOR_METADATA = "ready_for_metadata"
    METADATA_IN_PROGRESS = "metadata_in_progress"
    METADATA_COMPLETE = "metadata_complete"
    METADATA_FAILED = (
        "metadata_failed"  # All samples failed validation or processing error
    )


class PublicationQueueEntry(BaseModel):
    """
    Queue entry for publication extraction with full metadata.

    This schema represents a complete publication extraction request prepared
    by the research_agent with identifiers, extraction parameters, and results.

    **Multi-Agent Handoff Contract:**
    1. research_agent: Extracts SRA identifiers from publication, fetches metadata,
       saves to workspace/metadata/, populates workspace_metadata_keys with basenames
    2. metadata_assistant: Reads workspace_metadata_keys, loads full paths via
       get_workspace_metadata_paths(), performs filtering/validation, populates
       harmonization_metadata
    3. research_agent: Reads harmonization_metadata for final CSV export

    **Workspace Integration Fields:**
    - workspace_metadata_keys: List of metadata file basenames (e.g., ['SRR123_metadata.json'])
      - Set by: research_agent after SRA metadata fetch
      - Read by: metadata_assistant for filtering
    - harmonization_metadata: Filtered/validated metadata ready for export
      - Set by: metadata_assistant after processing
      - Read by: research_agent for CSV generation

    Attributes:
        entry_id: Unique identifier for this queue entry
        pmid: PubMed identifier (if available)
        doi: Digital Object Identifier (if available)
        pmc_id: PubMed Central identifier (if available)
        title: Publication title
        authors: List of author names
        year: Publication year
        journal: Journal name
        priority: Processing priority (1=highest, 10=lowest)
        status: Current status (pending, extracting, completed, failed)
        extraction_level: Target extraction depth
        schema_type: Schema to use for validation (e.g., microbiome, single_cell)
        metadata_url: URL to publication metadata page
        supplementary_files: List of supplementary file URLs
        github_url: GitHub repository URL (if mentioned)
        extracted_identifiers: Dataset identifiers extracted from publication
        extracted_metadata: Full extracted metadata content
        workspace_metadata_keys: List of workspace metadata file basenames
        harmonization_metadata: Harmonized metadata from metadata_assistant
        error_log: List of error messages if extraction failed
        created_at: Timestamp when entry was created
        updated_at: Timestamp when entry was last updated
        processed_by: Agent or user who executed the extraction
        cached_content_path: Path to cached PublicationContent in workspace
    """

    # Core identification
    entry_id: str = Field(..., description="Unique identifier for this queue entry")
    pmid: Optional[str] = Field(None, description="PubMed identifier (PMID)")
    doi: Optional[str] = Field(None, description="Digital Object Identifier (DOI)")
    pmc_id: Optional[str] = Field(None, description="PubMed Central identifier")

    # Publication metadata
    title: Optional[str] = Field(None, description="Publication title")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    year: Optional[int] = Field(None, description="Publication year")
    journal: Optional[str] = Field(None, description="Journal name")

    # Priority and status
    priority: int = Field(
        default=5, ge=1, le=10, description="Processing priority (1=highest, 10=lowest)"
    )
    status: PublicationStatus = Field(
        default=PublicationStatus.PENDING, description="Current processing status"
    )

    # Extraction configuration
    extraction_level: ExtractionLevel = Field(
        default=ExtractionLevel.METHODS,
        description="Target extraction depth (abstract, methods, full_text, identifiers)",
    )
    schema_type: str = Field(
        default="general",
        description="Schema to use for validation (microbiome, single_cell, proteomics, general)",
    )

    # URLs and resources (multiple sources from RIS)
    metadata_url: Optional[str] = Field(
        None, description="Primary article URL (from RIS UR field)"
    )
    pdf_url: Optional[str] = Field(
        None, description="Direct PDF URL (from RIS L1 field)"
    )
    pubmed_url: Optional[str] = Field(
        None, description="PubMed URL (from RIS L2 field if ncbi.nlm.nih.gov)"
    )
    fulltext_url: Optional[str] = Field(
        None, description="Full text URL (transformed from abstract URL)"
    )
    supplementary_files: List[str] = Field(
        default_factory=list, description="List of supplementary file URLs"
    )
    github_url: Optional[str] = Field(
        None, description="GitHub repository URL (if mentioned in publication)"
    )

    # Extraction results
    extracted_identifiers: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Dataset identifiers extracted (GEO, SRA, BioProject, BioSample, etc.)",
    )
    extracted_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Full extracted metadata content"
    )

    # Workspace integration (multi-agent handoff)
    dataset_ids: List[str] = Field(
        default_factory=list,
        description="List of dataset accessions associated with this publication (GSE, SRP, etc.)",
    )
    workspace_metadata_keys: List[str] = Field(
        default_factory=list,
        description="List of workspace metadata file basenames (e.g., ['SRR123_metadata.json']). "
        "Populated by research_agent after SRA metadata fetch, consumed by metadata_assistant.",
    )
    filtered_workspace_key: Optional[str] = Field(
        None,
        description="Workspace key pointing to filtered/curated sample metadata produced by metadata_assistant",
    )
    harmonization_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Harmonized metadata from metadata_assistant. Contains filtered/validated data "
        "ready for final CSV export by research_agent.",
    )
    handoff_status: HandoffStatus = Field(
        default=HandoffStatus.NOT_READY,
        description="Fine-grained status tracking cross-agent handoff progress",
    )

    # Execution metadata
    created_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp when entry was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when entry was last updated",
    )
    processed_by: Optional[str] = Field(
        None, description="Agent or user who executed the extraction"
    )
    cached_content_path: Optional[str] = Field(
        None, description="Path to cached PublicationContent in workspace"
    )
    error_log: List[str] = Field(
        default_factory=list, description="List of error messages if extraction failed"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "entry_id": "pub_queue_1234567890",
                "pmid": "35042229",
                "doi": "10.1038/s41586-022-04426-0",
                "pmc_id": "PMC8891176",
                "title": "Single-cell RNA sequencing reveals novel cell types in human brain",
                "authors": ["Smith J", "Jones A", "Williams B"],
                "year": 2022,
                "journal": "Nature",
                "priority": 5,
                "status": "pending",
                "extraction_level": "methods",
                "schema_type": "single_cell",
                "metadata_url": "https://pubmed.ncbi.nlm.nih.gov/35042229/",
                "supplementary_files": [
                    "https://example.com/supplementary_table_1.xlsx"
                ],
                "github_url": "https://github.com/lab/analysis-pipeline",
                "extracted_identifiers": {
                    "GEO": ["GSE180759"],
                    "SRA": ["SRR14567890"],
                    "BioProject": ["PRJNA720345"],
                },
                "extracted_metadata": {"methods": "scRNA-seq analysis using..."},
                "dataset_ids": ["GSE180759"],
                "workspace_metadata_keys": [
                    "SRR14567890_metadata.json",
                    "SRR14567891_metadata.json",
                ],
                "filtered_workspace_key": "publication_pub_queue_1234567890_filtered_samples.json",
                "harmonization_metadata": {
                    "samples": [
                        {
                            "run_accession": "SRR14567890",
                            "tissue": "brain",
                            "cell_type": "neuron",
                        },
                        {
                            "run_accession": "SRR14567891",
                            "tissue": "brain",
                            "cell_type": "astrocyte",
                        },
                    ],
                    "validation_status": "passed",
                },
                "handoff_status": "not_ready",
                "created_at": "2024-01-15T10:30:00",
                "updated_at": "2024-01-15T10:30:00",
                "processed_by": None,
                "cached_content_path": "literature/publication_PMID35042229.json",
                "error_log": [],
            }
        }

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, v: str) -> str:
        """Validate that entry_id is not empty."""
        if not v or not v.strip():
            raise ValueError("entry_id cannot be empty")
        return v.strip()

    @field_validator("pmid", "doi", "pmc_id")
    @classmethod
    def validate_identifier_format(cls, v: Optional[str], info) -> Optional[str]:
        """Validate identifier format if provided."""
        if v is None or not v.strip():
            return None

        v = v.strip()

        # Validate format based on field name
        if info.field_name == "pmid":
            # Strip common prefixes/symbols (PMID, PMID:, PMID-)
            v_upper = v.upper().strip()
            if v_upper.startswith("PMID"):
                v = v_upper[4:]
            # Remove punctuation/whitespace commonly inserted by RIS exports
            digits = "".join(ch for ch in str(v) if ch.isdigit())
            if not digits:
                raise ValueError(f"PMID must contain digits, got '{v}'")
            return digits

        elif info.field_name == "doi":
            # DOI should contain a slash (e.g., 10.1234/journal.2024.123)
            if "/" not in v:
                raise ValueError(f"DOI must contain '/', got '{v}'")
            return v

        elif info.field_name == "pmc_id":
            # PMC ID should be numeric or prefixed with PMC
            v_upper = v.upper().strip()
            if v_upper.startswith("PMC"):
                v_upper = v_upper[3:]
            digits = "".join(ch for ch in v_upper if ch.isdigit())
            if not digits:
                raise ValueError(
                    f"PMC ID must contain digits after PMC prefix, got '{v}'"
                )
            return f"PMC{digits}"

        return v

    @field_validator("year")
    @classmethod
    def validate_year(cls, v: Optional[int]) -> Optional[int]:
        """Validate publication year is reasonable."""
        if v is None:
            return None

        if v < 1900 or v > datetime.now().year + 1:
            raise ValueError(
                f"Publication year must be between 1900 and {datetime.now().year + 1}, got {v}"
            )
        return v

    @field_validator("schema_type")
    @classmethod
    def validate_schema_type(cls, v: str) -> str:
        """Validate schema_type is one of supported types."""
        allowed = {
            "general",
            "microbiome",
            "single_cell",
            "bulk_rnaseq",
            "proteomics",
            "metabolomics",
            "spatial",
            "epigenomics",
        }
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(f"schema_type must be one of {allowed}, got '{v}'")
        return v_lower

    @field_validator("extracted_identifiers")
    @classmethod
    def validate_extracted_identifiers(
        cls, v: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """Validate extracted_identifiers structure."""
        if not isinstance(v, dict):
            raise ValueError("extracted_identifiers must be a dictionary")

        # Ensure all values are lists of strings
        for database, identifiers in v.items():
            if not isinstance(identifiers, list):
                raise ValueError(
                    f"Identifiers for '{database}' must be a list, got {type(identifiers)}"
                )
            if not all(isinstance(i, str) for i in identifiers):
                raise ValueError(f"All identifiers for '{database}' must be strings")

        return v

    @field_validator("workspace_metadata_keys")
    @classmethod
    def validate_workspace_metadata_keys(cls, v: List[str]) -> List[str]:
        """Validate workspace_metadata_keys has no duplicates and all are strings."""
        if not isinstance(v, list):
            raise ValueError("workspace_metadata_keys must be a list")

        if not all(isinstance(key, str) for key in v):
            raise ValueError("All workspace_metadata_keys must be strings")

        # Check for duplicates
        if len(v) != len(set(v)):
            duplicates = [key for key in set(v) if v.count(key) > 1]
            raise ValueError(f"Duplicate workspace_metadata_keys found: {duplicates}")

        return v

    @field_validator("updated_at", mode="before")
    @classmethod
    def ensure_updated_at_is_datetime(cls, v):
        """Ensure updated_at is always a datetime object."""
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

    @field_validator("created_at", mode="before")
    @classmethod
    def ensure_created_at_is_datetime(cls, v):
        """Ensure created_at is always a datetime object."""
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

    def update_status(
        self,
        status: PublicationStatus,
        cached_content_path: Optional[str] = None,
        error: Optional[str] = None,
        processed_by: Optional[str] = None,
        extracted_identifiers: Optional[Dict[str, List[str]]] = None,
        workspace_metadata_keys: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        filtered_workspace_key: Optional[str] = None,
        handoff_status: Optional[HandoffStatus] = None,
        harmonization_metadata: Optional[Dict[str, Any]] = None,
        pmc_id: Optional[str] = None,
        pmid: Optional[str] = None,
    ) -> None:
        """
        Update entry status and related fields.

        Args:
            status: New status
            cached_content_path: Optional path to cached content if completed
            error: Optional error message if failed
            processed_by: Optional agent/user who executed extraction
            extracted_identifiers: Optional extracted dataset identifiers
            workspace_metadata_keys: Optional list of workspace metadata file basenames
            dataset_ids: Optional list of dataset accessions
            filtered_workspace_key: Optional workspace key for filtered metadata artifacts
            handoff_status: Optional override for handoff progress
            harmonization_metadata: Optional harmonized metadata payload
            pmc_id: Optional PMC ID discovered during enrichment
            pmid: Optional PubMed ID discovered during enrichment
        """
        if isinstance(status, str):
            status = PublicationStatus(status)

        self.status = status
        self.updated_at = datetime.now()

        if cached_content_path:
            self.cached_content_path = cached_content_path

        if error:
            self.error_log.append(f"[{datetime.now().isoformat()}] {error}")

        if processed_by:
            self.processed_by = processed_by

        if extracted_identifiers:
            self.extracted_identifiers.update(extracted_identifiers)

        if workspace_metadata_keys is not None:
            self.workspace_metadata_keys = workspace_metadata_keys

        if dataset_ids is not None:
            self.dataset_ids = dataset_ids

        if filtered_workspace_key is not None:
            self.filtered_workspace_key = filtered_workspace_key

        if handoff_status is not None:
            if isinstance(handoff_status, str):
                self.handoff_status = HandoffStatus(handoff_status)
            else:
                self.handoff_status = handoff_status

        if harmonization_metadata is not None:
            self.harmonization_metadata = harmonization_metadata

        if pmc_id is not None:
            self.pmc_id = pmc_id

        if pmid is not None:
            self.pmid = pmid

    def get_primary_identifier(self) -> Optional[str]:
        """
        Get primary identifier for this publication.

        Returns:
            str: Primary identifier (PMID preferred, then DOI, then PMC)
        """
        if self.pmid:
            return f"PMID:{self.pmid}"
        elif self.doi:
            return f"DOI:{self.doi}"
        elif self.pmc_id:
            return self.pmc_id
        else:
            return None

    def has_identifier_data(self) -> bool:
        """
        Check if any dataset identifiers have been extracted.

        Returns:
            bool: True if identifiers exist, False otherwise
        """
        return bool(
            self.extracted_identifiers and any(self.extracted_identifiers.values())
        )

    def get_workspace_metadata_paths(self, workspace_dir: str) -> List[str]:
        """
        Get full paths to workspace metadata files.

        Args:
            workspace_dir: Base workspace directory path

        Returns:
            List[str]: Full paths to metadata files (e.g., ['/workspace/metadata/SRR123_metadata.json'])

        Example:
            >>> entry.workspace_metadata_keys = ['SRR123_metadata.json', 'SRR456_metadata.json']
            >>> paths = entry.get_workspace_metadata_paths('/workspace')
            >>> # Returns: ['/workspace/metadata/SRR123_metadata.json', '/workspace/metadata/SRR456_metadata.json']
        """
        from pathlib import Path

        workspace_path = Path(workspace_dir)
        metadata_dir = workspace_path / "metadata"

        return [str(metadata_dir / key) for key in self.workspace_metadata_keys]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation for JSON serialization.

        Returns:
            Dict[str, Any]: Dictionary with all fields serialized
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PublicationQueueEntry":
        """
        Create entry from dictionary.

        Args:
            data: Dictionary with entry data

        Returns:
            PublicationQueueEntry: Validated entry instance
        """
        # Handle datetime conversion if needed
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return cls(**data)
