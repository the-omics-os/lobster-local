"""
Workspace Content Service - Structured Content Caching.

This service manages structured caching of publications, datasets, and metadata
in the DataManagerV2 workspace. It provides schema validation, professional naming,
and flexible content retrieval with level-based filtering.

Phase 4a: Workspace Tools Implementation
- Structured caching for publications, datasets, metadata
- Pydantic schema validation for content types
- Professional naming convention enforcement
- Level-based content retrieval (summary/methods/samples/platform)
- Integration with DataManagerV2 workspace
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# ===============================================================================
# Content Type Enums
# ===============================================================================


class ContentType(str, Enum):
    """Content types for workspace caching."""

    PUBLICATION = "publication"
    DATASET = "dataset"
    METADATA = "metadata"
    DOWNLOAD_QUEUE = "download_queue"
    PUBLICATION_QUEUE = "publication_queue"
    EXPORTS = "exports"


class RetrievalLevel(str, Enum):
    """Content retrieval detail levels."""

    SUMMARY = "summary"  # Basic overview (title, authors, sample count, etc.)
    METHODS = "methods"  # Methods section (for publications)
    SAMPLES = "samples"  # Sample metadata (for datasets)
    PLATFORM = "platform"  # Platform/technology info
    FULL = "full"  # All available content


# ===============================================================================
# Pydantic Content Schemas
# ===============================================================================


class PublicationContent(BaseModel):
    """
    Schema for cached publication content.

    Stores publication metadata, abstract, methods, and full-text
    retrieved from PubMed, PMC, bioRxiv, or other sources.
    """

    identifier: str = Field(
        ..., description="Publication identifier (PMID, DOI, or bioRxiv ID)"
    )
    title: Optional[str] = Field(None, description="Publication title")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    journal: Optional[str] = Field(None, description="Journal name")
    year: Optional[int] = Field(None, description="Publication year")
    abstract: Optional[str] = Field(None, description="Abstract text")
    methods: Optional[str] = Field(None, description="Methods section text")
    full_text: Optional[str] = Field(None, description="Full publication text")
    keywords: List[str] = Field(
        default_factory=list, description="Publication keywords"
    )
    source: str = Field(..., description="Source provider (PMC, PubMed, bioRxiv, etc.)")
    cached_at: str = Field(
        ..., description="ISO 8601 timestamp when content was cached"
    )
    url: Optional[str] = Field(None, description="Publication URL")

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        """Validate publication identifier format."""
        if not v or not v.strip():
            raise ValueError("Publication identifier cannot be empty")
        return v.strip()

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate source is not empty."""
        if not v or not v.strip():
            raise ValueError("Source cannot be empty")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "identifier": "PMID:35042229",
                "title": "Single-cell RNA-seq reveals...",
                "authors": ["Smith J", "Jones A"],
                "journal": "Nature",
                "year": 2022,
                "abstract": "We performed single-cell RNA-seq...",
                "methods": "Cells were processed using 10X Chromium...",
                "source": "PMC",
                "cached_at": "2025-01-12T10:30:00",
            }
        }


class DatasetContent(BaseModel):
    """
    Schema for cached dataset content.

    Stores dataset metadata from GEO, SRA, or other repositories,
    including platform info, sample metadata, and experimental design.
    """

    identifier: str = Field(..., description="Dataset identifier (GSE, SRA, etc.)")
    title: Optional[str] = Field(None, description="Dataset title")
    platform: Optional[str] = Field(
        None, description="Platform/technology (e.g., Illumina NovaSeq)"
    )
    platform_id: Optional[Union[str, List[str]]] = Field(
        None, description="Platform ID(s) - can be single string or list for multi-platform datasets (e.g., 'GPL570' or ['GPL570', 'GPL96'])"
    )
    organism: Optional[str] = Field(None, description="Organism (e.g., Homo sapiens)")
    sample_count: int = Field(..., description="Number of samples in dataset")
    samples: Optional[Dict[str, Any]] = Field(
        None, description="Sample metadata (GSM IDs → metadata)"
    )
    experimental_design: Optional[str] = Field(
        None, description="Experimental design description"
    )
    summary: Optional[str] = Field(None, description="Dataset summary/abstract")
    pubmed_ids: List[str] = Field(
        default_factory=list, description="Associated PubMed IDs"
    )
    source: str = Field(..., description="Source repository (GEO, SRA, PRIDE, etc.)")
    cached_at: str = Field(
        ..., description="ISO 8601 timestamp when content was cached"
    )
    url: Optional[str] = Field(None, description="Dataset URL")

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        """Validate dataset identifier format."""
        if not v or not v.strip():
            raise ValueError("Dataset identifier cannot be empty")
        return v.strip()

    @field_validator("sample_count")
    @classmethod
    def validate_sample_count(cls, v: int) -> int:
        """Validate sample count is positive."""
        if v < 0:
            raise ValueError("Sample count must be non-negative")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "identifier": "GSE123456",
                "title": "Single-cell RNA-seq of aging brain",
                "platform": "Illumina NovaSeq 6000",
                "platform_id": "GPL24676",
                "organism": "Homo sapiens",
                "sample_count": 12,
                "samples": {"GSM1": {"age": 25, "tissue": "brain"}},
                "source": "GEO",
                "cached_at": "2025-01-12T10:30:00",
            }
        }


class MetadataContent(BaseModel):
    """
    Schema for cached metadata content.

    Stores arbitrary metadata like sample mappings, validation results,
    quality control reports, or other structured analysis outputs.
    """

    identifier: str = Field(..., description="Unique metadata identifier")
    content_type: str = Field(
        ..., description="Metadata type (sample_mapping, validation, qc_report, etc.)"
    )
    description: Optional[str] = Field(None, description="Human-readable description")
    data: Dict[str, Any] = Field(..., description="Metadata content (arbitrary JSON)")
    related_datasets: List[str] = Field(
        default_factory=list, description="Related dataset identifiers"
    )
    source: str = Field(..., description="Source of metadata (tool or service name)")
    cached_at: str = Field(
        ..., description="ISO 8601 timestamp when content was cached"
    )

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        """Validate metadata identifier format."""
        if not v or not v.strip():
            raise ValueError("Metadata identifier cannot be empty")
        return v.strip()

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """Validate content type is not empty."""
        if not v or not v.strip():
            raise ValueError("Content type cannot be empty")
        return v

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "identifier": "gse12345_to_gse67890_mapping",
                "content_type": "sample_mapping",
                "description": "Sample ID mapping between two datasets",
                "data": {"exact_matches": 10, "fuzzy_matches": 5},
                "related_datasets": ["GSE12345", "GSE67890"],
                "source": "SampleMappingService",
                "cached_at": "2025-01-12T10:30:00",
            }
        }


# ===============================================================================
# Workspace Content Service
# ===============================================================================


class WorkspaceContentService:
    """
    Workspace content management service.

    Manages structured caching of publications, datasets, and metadata in the
    DataManagerV2 workspace with schema validation, professional naming, and
    flexible content retrieval.

    **Features:**
    - Schema validation via Pydantic models
    - Professional naming conventions (lowercase, underscores, timestamps)
    - Level-based content retrieval (summary/methods/samples/platform/full)
    - Content listing and discovery
    - Integration with DataManagerV2 workspace

    **Content Types:**
    - Publications: Papers from PubMed, PMC, bioRxiv
    - Datasets: GEO, SRA, PRIDE datasets with metadata
    - Metadata: Sample mappings, validation results, QC reports
    - Exports: Analysis results and data exports

    **Storage Structure:**
    - workspace_path/literature/*.json (publications)
    - workspace_path/data/*.json (datasets)
    - workspace_path/metadata/*.json (metadata)
    - workspace_path/exports/*.* (exported results)

    Examples:
        >>> service = WorkspaceContentService(data_manager)
        >>>
        >>> # Cache publication
        >>> pub_content = PublicationContent(
        ...     identifier="PMID:35042229",
        ...     title="Single-cell analysis...",
        ...     authors=["Smith J"],
        ...     source="PMC",
        ...     cached_at=datetime.now().isoformat()
        ... )
        >>> path = service.write_content(pub_content, ContentType.PUBLICATION)
        >>>
        >>> # Retrieve summary level
        >>> summary = service.read_content("PMID:35042229",
        ...     ContentType.PUBLICATION, level=RetrievalLevel.SUMMARY)
        >>>
        >>> # List all cached publications
        >>> publications = service.list_content(ContentType.PUBLICATION)
    """

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize WorkspaceContentService.

        Args:
            data_manager: DataManagerV2 instance for workspace access
        """
        self.data_manager = data_manager
        # Use workspace_path directly for backward compatibility with existing tools
        self.workspace_base = Path(data_manager.workspace_path)

        # Create content subdirectories (aligned with existing research_agent tools)
        self.publications_dir = self.workspace_base / "literature"
        self.datasets_dir = self.workspace_base / "data"
        self.metadata_dir = self.workspace_base / "metadata"
        self.exports_dir = self.workspace_base / "exports"

        # Queue directories are managed by DataManagerV2 at .lobster/queues/
        # Actual files: download_queue.jsonl, publication_queue.jsonl
        self.queues_dir = self.workspace_base / ".lobster" / "queues"

        # Create directories if they don't exist (queues_dir is created by DataManagerV2)
        self.publications_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(
            f"WorkspaceContentService initialized with workspace at {self.workspace_base}"
        )

    def _get_content_dir(self, content_type: ContentType) -> Path:
        """
        Get directory for content type.

        Args:
            content_type: Content type enum

        Returns:
            Path: Directory path for content type
        """
        if content_type == ContentType.PUBLICATION:
            return self.publications_dir
        elif content_type == ContentType.DATASET:
            return self.datasets_dir
        elif content_type == ContentType.METADATA:
            return self.metadata_dir
        elif content_type == ContentType.EXPORTS:
            return self.exports_dir
        elif content_type in (
            ContentType.DOWNLOAD_QUEUE,
            ContentType.PUBLICATION_QUEUE,
        ):
            # Queue files are JSONL files in .lobster/queues/ managed by DataManagerV2
            return self.queues_dir
        else:
            raise ValueError(f"Unknown content type: {content_type}")

    def _sanitize_filename(self, identifier: str) -> str:
        """
        Sanitize identifier for use as filename.

        Converts identifiers to lowercase, replaces special characters with
        underscores, and enforces professional naming conventions.

        Args:
            identifier: Content identifier (PMID, GSE, etc.)

        Returns:
            str: Sanitized filename (without .json extension)

        Examples:
            >>> service._sanitize_filename("PMID:35042229")
            "pmid_35042229"
            >>> service._sanitize_filename("GSE123456")
            "gse123456"
            >>> service._sanitize_filename("gse12345_to_gse67890_mapping")
            "gse12345_to_gse67890_mapping"
        """
        # Convert to lowercase
        sanitized = identifier.lower()

        # Replace special characters with underscores
        sanitized = sanitized.replace(":", "_").replace("/", "_").replace("\\", "_")
        sanitized = sanitized.replace(" ", "_").replace("-", "_")

        # Remove consecutive underscores
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")

        # Strip leading/trailing underscores
        sanitized = sanitized.strip("_")

        return sanitized

    def write_content(
        self,
        content: Union[PublicationContent, DatasetContent, MetadataContent],
        content_type: ContentType,
        output_format: str = "json",
    ) -> str:
        """
        Write content to workspace with schema validation.

        Args:
            content: Content model (PublicationContent, DatasetContent, or MetadataContent)
            content_type: Content type enum
            output_format: Output format ("json" or "csv"). Default: "json"

        Returns:
            str: Path to cached content file

        Raises:
            ValueError: If content type doesn't match content model or format is invalid
            ValidationError: If content fails schema validation

        Examples:
            >>> pub = PublicationContent(
            ...     identifier="PMID:35042229",
            ...     title="Paper title",
            ...     source="PMC",
            ...     cached_at=datetime.now().isoformat()
            ... )
            >>> path = service.write_content(pub, ContentType.PUBLICATION)
            >>> print(path)
            "/workspace/cache/content/publications/pmid_35042229.json"
            >>>
            >>> # Export metadata as CSV
            >>> meta = MetadataContent(
            ...     identifier="sample_mapping",
            ...     content_type="mapping",
            ...     data={"sample_id": ["S1", "S2"], "condition": ["ctrl", "treated"]},
            ...     source="MappingService",
            ...     cached_at=datetime.now().isoformat()
            ... )
            >>> path = service.write_content(meta, ContentType.METADATA, output_format="csv")
            >>> print(path)
            "/workspace/metadata/sample_mapping.csv"
        """
        # Validate output format
        if output_format not in ["json", "csv"]:
            raise ValueError(
                f"Invalid output format '{output_format}'. Must be 'json' or 'csv'"
            )

        # Validate content type matches model
        if content_type == ContentType.PUBLICATION and not isinstance(
            content, PublicationContent
        ):
            raise ValueError(
                "Content type PUBLICATION requires PublicationContent model"
            )
        elif content_type == ContentType.DATASET and not isinstance(
            content, DatasetContent
        ):
            raise ValueError("Content type DATASET requires DatasetContent model")
        elif content_type == ContentType.METADATA and not isinstance(
            content, MetadataContent
        ):
            raise ValueError("Content type METADATA requires MetadataContent model")

        # Get content directory
        content_dir = self._get_content_dir(content_type)

        # Sanitize filename
        filename = self._sanitize_filename(content.identifier)

        # Convert content to dict
        content_dict = content.model_dump()

        # Handle different output formats
        if output_format == "csv":
            file_path = content_dir / f"{filename}.csv"
            self._write_csv(content, content_dict, file_path)
        else:  # json (default)
            file_path = content_dir / f"{filename}.json"
            with open(file_path, "w") as f:
                json.dump(content_dict, f, indent=2, default=str)

        logger.info(
            f"Cached {content_type.value} '{content.identifier}' to {file_path} ({output_format})"
        )

        return str(file_path)

    def _write_csv(
        self,
        content: Union[PublicationContent, DatasetContent, MetadataContent],
        content_dict: Dict[str, Any],
        file_path: Path,
    ) -> None:
        """
        Write content as CSV file.

        For MetadataContent with a 'data' attribute, exports the data field.
        For other content types, exports the entire model as a single-row DataFrame.

        Args:
            content: Content model
            content_dict: Content dictionary from model_dump()
            file_path: Output file path

        Raises:
            ValueError: If data cannot be converted to DataFrame
        """
        # Check if content has 'data' attribute (MetadataContent)
        if hasattr(content, "data") and content.data:
            data_content = content.data

            # Case 1: data is a dict with list values (column-oriented)
            # Example: {"col1": [1, 2, 3], "col2": [4, 5, 6]}
            if isinstance(data_content, dict) and all(
                isinstance(v, list) for v in data_content.values()
            ):
                df = pd.DataFrame.from_dict(data_content)

            # Case 2: data is a list of dicts (row-oriented)
            # Example: [{"col1": 1, "col2": 4}, {"col1": 2, "col2": 5}]
            elif isinstance(data_content, list) and all(
                isinstance(item, dict) for item in data_content
            ):
                df = pd.DataFrame(data_content)

            # Case 3: data is a dict but values are not lists
            # Convert to single-row DataFrame
            elif isinstance(data_content, dict):
                df = pd.DataFrame([data_content])

            else:
                raise ValueError(
                    f"Cannot convert 'data' attribute to CSV. "
                    f"Expected dict with list values or list of dicts, "
                    f"got {type(data_content).__name__}"
                )
        else:
            # No 'data' attribute - export entire model as single-row DataFrame
            # Flatten nested structures to strings for CSV compatibility
            flattened_dict = {}
            for key, value in content_dict.items():
                if isinstance(value, (list, dict)):
                    flattened_dict[key] = json.dumps(value)
                else:
                    flattened_dict[key] = value

            df = pd.DataFrame([flattened_dict])

        # Write CSV with proper formatting
        df.to_csv(file_path, index=False, encoding="utf-8")

        logger.debug(f"Wrote DataFrame with shape {df.shape} to {file_path}")

    def read_content(
        self,
        identifier: str,
        content_type: ContentType,
        level: Optional[RetrievalLevel] = None,
    ) -> Dict[str, Any]:
        """
        Read content from workspace with level-based filtering.

        Args:
            identifier: Content identifier (PMID, GSE, etc.)
            content_type: Content type enum
            level: Retrieval level (summary/methods/samples/platform/full)

        Returns:
            Dict[str, Any]: Content dictionary (filtered by level if specified)

        Raises:
            FileNotFoundError: If content not found in workspace
            ValueError: If level is invalid for content type

        Examples:
            >>> # Get publication summary
            >>> summary = service.read_content(
            ...     "PMID:35042229",
            ...     ContentType.PUBLICATION,
            ...     level=RetrievalLevel.SUMMARY
            ... )
            >>> print(summary.keys())
            dict_keys(['identifier', 'title', 'authors', 'journal', 'year'])
            >>>
            >>> # Get dataset samples
            >>> samples = service.read_content(
            ...     "GSE123456",
            ...     ContentType.DATASET,
            ...     level=RetrievalLevel.SAMPLES
            ... )
            >>> print(samples.keys())
            dict_keys(['identifier', 'sample_count', 'samples'])
        """
        # Get content directory
        content_dir = self._get_content_dir(content_type)

        # Sanitize filename
        filename = self._sanitize_filename(identifier)
        file_path = content_dir / f"{filename}.json"

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(
                f"{content_type.value.capitalize()} '{identifier}' not found in workspace"
            )

        # Read content from file
        with open(file_path, "r") as f:
            content_dict = json.load(f)

        # OPTION C FIX: Unwrap "data" field for metadata content type
        # WorkspaceContentService stores metadata as {"identifier": "...", "data": {...}}
        # But consumers expect the normalized structure directly
        if content_type == ContentType.METADATA and "data" in content_dict:
            # Unwrap: return the inner "data" dict + preserve top-level metadata
            unwrapped = content_dict["data"].copy()
            # Preserve metadata fields from wrapper (identifier, cached_at, source, etc.)
            for key in ["identifier", "cached_at", "source", "content_type", "description"]:
                if key in content_dict and key not in unwrapped:
                    unwrapped[key] = content_dict[key]
            content_dict = unwrapped

        # Apply level-based filtering if specified
        if level is None or level == RetrievalLevel.FULL:
            return content_dict

        # Filter by level
        filtered_dict = self._filter_by_level(content_dict, content_type, level)

        return filtered_dict

    def _filter_by_level(
        self, content: Dict[str, Any], content_type: ContentType, level: RetrievalLevel
    ) -> Dict[str, Any]:
        """
        Filter content dictionary by retrieval level.

        Args:
            content: Full content dictionary
            content_type: Content type
            level: Retrieval level

        Returns:
            Dict[str, Any]: Filtered content dictionary

        Raises:
            ValueError: If level is invalid for content type
        """
        if level == RetrievalLevel.FULL:
            return content

        # Define fields for each level and content type
        level_fields = {
            ContentType.PUBLICATION: {
                RetrievalLevel.SUMMARY: [
                    "identifier",
                    "title",
                    "authors",
                    "journal",
                    "year",
                    "keywords",
                    "source",
                    "cached_at",
                    "url",
                ],
                RetrievalLevel.METHODS: [
                    "identifier",
                    "title",
                    "methods",
                    "source",
                    "cached_at",
                ],
            },
            ContentType.DATASET: {
                RetrievalLevel.SUMMARY: [
                    "identifier",
                    "title",
                    "sample_count",
                    "organism",
                    "source",
                    "cached_at",
                    "url",
                ],
                RetrievalLevel.SAMPLES: [
                    "identifier",
                    "sample_count",
                    "samples",
                    "experimental_design",
                    "source",
                    "cached_at",
                ],
                RetrievalLevel.PLATFORM: [
                    "identifier",
                    "platform",
                    "platform_id",
                    "organism",
                    "source",
                    "cached_at",
                ],
            },
            ContentType.METADATA: {
                RetrievalLevel.SUMMARY: [
                    "identifier",
                    "content_type",
                    "description",
                    "related_datasets",
                    "source",
                    "cached_at",
                ],
            },
        }

        # Check if level is valid for content type
        if content_type not in level_fields:
            raise ValueError(f"Level filtering not supported for {content_type.value}")

        if level not in level_fields[content_type]:
            raise ValueError(
                f"Level '{level.value}' not valid for {content_type.value}. "
                f"Valid levels: {', '.join(k.value for k in level_fields[content_type].keys())}"
            )

        # Filter content to include only specified fields
        fields = level_fields[content_type][level]
        filtered = {k: v for k, v in content.items() if k in fields}

        return filtered

    def list_content(
        self, content_type: Optional[ContentType] = None
    ) -> List[Dict[str, Any]]:
        """
        List all cached content, optionally filtered by type.

        Args:
            content_type: Content type to filter by (None = all types)

        Returns:
            List[Dict[str, Any]]: List of content summaries

        Examples:
            >>> # List all cached content
            >>> all_content = service.list_content()
            >>>
            >>> # List only publications
            >>> publications = service.list_content(ContentType.PUBLICATION)
            >>>
            >>> for pub in publications:
            ...     print(f"{pub['identifier']}: {pub['title']}")
        """
        content_list = []

        # Determine which content types to list
        types_to_list = [content_type] if content_type else list(ContentType)

        for ctype in types_to_list:
            content_dir = self._get_content_dir(ctype)

            for json_file in content_dir.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        content_dict = json.load(f)

                    # Add content type and file path
                    content_dict["_content_type"] = ctype.value
                    content_dict["_file_path"] = str(json_file)

                    content_list.append(content_dict)
                except Exception as e:
                    logger.warning(f"Could not read {json_file}: {e}")

        # Sort by cached_at (most recent first)
        content_list.sort(key=lambda x: x.get("cached_at", ""), reverse=True)

        logger.info(
            f"Listed {len(content_list)} cached items"
            + (f" of type {content_type.value}" if content_type else "")
        )

        return content_list

    def delete_content(self, identifier: str, content_type: ContentType) -> bool:
        """
        Delete content from workspace.

        Args:
            identifier: Content identifier
            content_type: Content type

        Returns:
            bool: True if content was deleted, False if not found

        Examples:
            >>> service.delete_content("PMID:35042229", ContentType.PUBLICATION)
            True
        """
        content_dir = self._get_content_dir(content_type)
        filename = self._sanitize_filename(identifier)
        file_path = content_dir / f"{filename}.json"

        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted {content_type.value} '{identifier}' from workspace")
            return True
        else:
            logger.warning(
                f"{content_type.value.capitalize()} '{identifier}' not found in workspace"
            )
            return False

    def read_download_queue_entry(self, entry_id: str) -> Dict[str, Any]:
        """
        Read a specific download queue entry.

        Args:
            entry_id: Queue entry identifier

        Returns:
            Dict[str, Any]: Entry details

        Raises:
            FileNotFoundError: If entry not found in queue
            AttributeError: If DataManager download_queue not available

        Examples:
            >>> entry = service.read_download_queue_entry("queue_entry_123")
            >>> print(entry['dataset_id'])
            GSE180759
        """
        if not self.data_manager or not hasattr(self.data_manager, "download_queue"):
            raise AttributeError("DataManager download_queue not available")

        if self.data_manager.download_queue is None:
            raise AttributeError("DataManager download_queue not available")

        try:
            entry = self.data_manager.download_queue.get_entry(entry_id)
            return entry.model_dump(mode="json")  # Pydantic v2 method
        except Exception as e:
            raise FileNotFoundError(
                f"Download queue entry '{entry_id}' not found"
            ) from e

    def list_download_queue_entries(
        self, status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all download queue entries with optional status filtering.

        Args:
            status_filter: Optional status to filter by (PENDING, IN_PROGRESS, COMPLETED, FAILED)

        Returns:
            List[Dict[str, Any]]: List of entry dictionaries

        Examples:
            >>> # List all entries
            >>> entries = service.list_download_queue_entries()
            >>> print(len(entries))
            5
            >>>
            >>> # Filter by status
            >>> pending = service.list_download_queue_entries(status_filter="PENDING")
            >>> print(len(pending))
            2
        """
        try:
            if not self.data_manager or not hasattr(self.data_manager, "download_queue"):
                return []

            if self.data_manager.download_queue is None:
                return []

            from lobster.core.schemas.download_queue import DownloadStatus

            # Convert string to enum if provided
            status_enum = None
            if status_filter:
                try:
                    # DownloadStatus enum values are lowercase ("pending", "completed")
                    status_enum = DownloadStatus(status_filter.lower())
                except ValueError:
                    # Invalid status, return empty list
                    logger.warning(
                        f"Invalid status filter '{status_filter}', returning empty list"
                    )
                    return []

            entries = self.data_manager.download_queue.list_entries(status=status_enum)

            # BUGFIX: Handle None return from list_entries()
            if entries is None:
                logger.warning("download_queue.list_entries() returned None (expected list)")
                return []

            return [entry.model_dump(mode="json") for entry in entries]
        except Exception as e:
            logger.error(f"Failed to list download queue entries: {e}", exc_info=True)
            return []  # Graceful degradation

    def get_workspace_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached workspace content.

        Returns:
            Dict[str, Any]: Statistics including counts by type and total size

        Examples:
            >>> stats = service.get_workspace_stats()
            >>> print(stats)
            {
                'total_items': 42,
                'publications': 15,
                'datasets': 20,
                'metadata': 7,
                'total_size_mb': 12.5,
                'cache_dir': '/workspace/cache/content'
            }
        """
        stats = {
            "total_items": 0,
            "publications": 0,
            "datasets": 0,
            "metadata": 0,
            "total_size_mb": 0.0,
            "cache_dir": str(self.workspace_base),
        }

        # Count items by type and calculate total size
        for content_type in ContentType:
            content_dir = self._get_content_dir(content_type)
            json_files = list(content_dir.glob("*.json"))
            csv_files = list(content_dir.glob("*.csv"))
            all_files = json_files + csv_files

            count = len(all_files)
            stats[content_type.value + "s"] = count  # pluralize
            stats["total_items"] += count

            # Calculate size
            for file in all_files:
                stats["total_size_mb"] += file.stat().st_size / (1024 * 1024)

        # Round size to 2 decimals
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)

        return stats

    def read_publication_queue_entry(self, entry_id: str) -> Dict[str, Any]:
        """
        Read a specific publication queue entry.

        Args:
            entry_id: Queue entry identifier

        Returns:
            Dict[str, Any]: Entry details

        Raises:
            FileNotFoundError: If entry not found in queue
            AttributeError: If DataManager publication_queue not available

        Examples:
            >>> entry = service.read_publication_queue_entry("pub_queue_123")
            >>> print(entry['title'])
            Single-cell RNA-seq reveals...
        """
        if not self.data_manager or not hasattr(self.data_manager, "publication_queue"):
            raise AttributeError("DataManager publication_queue not available")

        if self.data_manager.publication_queue is None:
            raise AttributeError("DataManager publication_queue not available")

        try:
            entry = self.data_manager.publication_queue.get_entry(entry_id)
            return entry.to_dict()  # PublicationQueueEntry has to_dict method
        except Exception as e:
            raise FileNotFoundError(
                f"Publication queue entry '{entry_id}' not found"
            ) from e

    def list_publication_queue_entries(
        self, status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all publication queue entries with optional status filtering.

        Args:
            status_filter: Optional status to filter by (pending, extracting,
                          metadata_extracted, metadata_enriched, handoff_ready,
                          completed, failed)

        Returns:
            List[Dict[str, Any]]: List of entry dictionaries

        Examples:
            >>> # List all entries
            >>> entries = service.list_publication_queue_entries()
            >>> print(len(entries))
            5
            >>>
            >>> # Filter by status
            >>> pending = service.list_publication_queue_entries(status_filter="pending")
            >>> print(len(pending))
            2
        """
        try:
            if not self.data_manager or not hasattr(self.data_manager, "publication_queue"):
                return []

            if self.data_manager.publication_queue is None:
                return []

            try:
                from lobster.core.schemas.publication_queue import PublicationStatus
            except ImportError:
                logger.warning("Publication queue schema not available (premium feature)")
                return []

            # Convert string to enum if provided
            status_enum = None
            if status_filter:
                try:
                    # PublicationStatus enum values are lowercase ("pending", "completed", etc.)
                    status_enum = PublicationStatus(status_filter.lower())
                except ValueError:
                    # Invalid status, return empty list
                    logger.warning(
                        f"Invalid status filter '{status_filter}', returning empty list"
                    )
                    return []

            entries = self.data_manager.publication_queue.list_entries(status=status_enum)

            # BUGFIX: Handle None return from list_entries()
            if entries is None:
                logger.warning("publication_queue.list_entries() returned None (expected list)")
                return []

            return [entry.to_dict() for entry in entries]
        except Exception as e:
            logger.error(f"Failed to list publication queue entries: {e}", exc_info=True)
            return []  # Graceful degradation

    # =========================================================================
    # Export File Discovery Methods (v1.0+ - Centralized Exports)
    # =========================================================================

    def list_export_files(
        self,
        pattern: str = "*",
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List files in centralized exports directory.

        Args:
            pattern: Glob pattern (e.g., "*.csv", "samples_*.tsv")
            category: Optional filter - 'metadata', 'results', 'plots', 'custom'

        Returns:
            List of file metadata dicts with keys: name, path, size, modified, category
        """
        exports_dir = self.workspace_base / "exports"
        if not exports_dir.exists():
            return []

        files = []
        for file_path in exports_dir.glob(pattern):
            if file_path.is_file():
                file_category = self._categorize_export_file(file_path)
                if category and file_category != category:
                    continue

                files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': file_path.stat().st_mtime,
                    'category': file_category
                })

        return sorted(files, key=lambda x: x['modified'], reverse=True)

    def _categorize_export_file(self, file_path: Path) -> str:
        """Categorize export file by naming convention."""
        name_lower = file_path.name.lower()
        if any(k in name_lower for k in ['sample', 'metadata', 'filtered']):
            return 'metadata'
        elif any(k in name_lower for k in ['result', 'analysis', 'de_genes', 'differential']):
            return 'results'
        elif any(k in name_lower for k in ['plot', 'figure', 'visualization', 'umap', 'pca']):
            return 'plots'
        else:
            return 'custom'

    def get_all_metadata_sources(self) -> Dict[str, List[Dict]]:
        """
        Get metadata from all sources: memory, workspace files, exports.

        Returns:
            Dict with keys: 'in_memory', 'workspace_files', 'exports', 'deprecated'
        """
        sources: Dict[str, List[Dict]] = {
            'in_memory': [],
            'workspace_files': [],
            'exports': [],
            'deprecated': []  # Old metadata/exports/ location
        }

        # 1. In-memory metadata_store
        metadata_store = self.data_manager.metadata_store
        for key, value in metadata_store.items():
            sources['in_memory'].append({
                'identifier': key,
                'type': 'memory',
                'data': value
            })

        # 2. Workspace metadata files
        if self.metadata_dir.exists():
            for json_file in self.metadata_dir.glob("*.json"):
                if json_file.parent.name != "exports":  # Skip old exports subdir
                    sources['workspace_files'].append({
                        'identifier': json_file.stem,
                        'type': 'file',
                        'path': str(json_file),
                        'size': json_file.stat().st_size
                    })

        # 3. New exports directory
        exports_dir = self.workspace_base / "exports"
        if exports_dir.exists():
            for export_file in exports_dir.iterdir():
                if export_file.is_file() and export_file.suffix in {'.csv', '.tsv', '.xlsx'}:
                    sources['exports'].append({
                        'identifier': export_file.stem,
                        'type': 'export',
                        'path': str(export_file),
                        'size': export_file.stat().st_size,
                        'category': self._categorize_export_file(export_file)
                    })

        # 4. Deprecated location
        old_exports_dir = self.metadata_dir / "exports"
        if old_exports_dir.exists():
            for old_file in old_exports_dir.glob("*"):
                if old_file.is_file():
                    sources['deprecated'].append({
                        'identifier': old_file.stem,
                        'type': 'deprecated_export',
                        'path': str(old_file),
                        'size': old_file.stat().st_size
                    })

        return sources

    def get_exports_directory(self, create: bool = True) -> Path:
        """
        Get the centralized exports directory path.

        Args:
            create: Whether to create directory if it doesn't exist (default: True)

        Returns:
            Path to exports directory (workspace/exports/)
        """
        exports_dir = self.workspace_base / "exports"
        if create and not exports_dir.exists():
            exports_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created exports directory: {exports_dir}")
        return exports_dir
