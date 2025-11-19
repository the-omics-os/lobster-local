"""
Download queue schema definitions for dataset download orchestration.

This module defines Pydantic schemas for managing download queue entries
with full metadata, validation results, and strategy information prepared
by the research_agent and executed by the data_expert.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class DownloadStatus(str, Enum):
    """Status of a download queue entry."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class StrategyConfig(BaseModel):
    """
    Download strategy configuration prepared by research_agent.

    Contains information about the recommended download approach
    based on metadata analysis and dataset characteristics.

    Attributes:
        strategy_name: Download strategy (MATRIX_FIRST, RAW_FIRST, etc.)
        concatenation_strategy: Sample handling (auto, union, intersection)
        confidence: Confidence score for strategy recommendation (0.0 to 1.0)
        rationale: Human-readable explanation of strategy choice
        strategy_params: Database-specific strategy parameters
        execution_params: Execution parameters for download process

    Example:
        >>> strategy = StrategyConfig(
        ...     strategy_name="MATRIX_FIRST",
        ...     concatenation_strategy="union",
        ...     confidence=0.95,
        ...     rationale="H5 file available with optimal structure",
        ...     strategy_params={
        ...         "use_intersecting_genes_only": False,
        ...         "min_samples": 10
        ...     },
        ...     execution_params={
        ...         "timeout": 7200,  # 2 hours for large datasets
        ...         "max_retries": 5
        ...     }
        ... )
    """

    strategy_name: str = Field(
        ..., description="Download strategy (e.g., MATRIX_FIRST, RAW_FIRST)"
    )
    concatenation_strategy: str = Field(
        ..., description="Concatenation strategy (auto, union, intersection)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)"
    )
    rationale: Optional[str] = Field(
        None, description="Human-readable explanation of strategy choice"
    )

    # Backward compatible: existing queue entries without these fields will work
    strategy_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""
        Database-specific strategy parameters.

        Examples:
        - GEO: {"use_intersecting_genes_only": bool, "min_samples": int}
        - SRA: {"quality_threshold": int, "paired_end_only": bool}
        - PRIDE: {"search_engine_filter": str, "min_peptides": int}
        """,
    )
    execution_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""
        Execution parameters for download process.

        Common parameters:
        - timeout: int (seconds, default: 3600)
        - max_retries: int (default: 3)
        - retry_backoff: float (seconds, default: 2.0)
        - max_concurrent_downloads: int (default: 1)
        - verify_checksum: bool (default: True)
        - resume_enabled: bool (default: False)
        """,
    )

    @field_validator("strategy_name")
    @classmethod
    def validate_strategy_name(cls, v: str) -> str:
        """Validate strategy name is not empty."""
        if not v or not v.strip():
            raise ValueError("strategy_name cannot be empty")
        return v.strip()

    @field_validator("concatenation_strategy")
    @classmethod
    def validate_concatenation_strategy(cls, v: str) -> str:
        """Validate concatenation strategy is one of allowed values."""
        allowed = {"auto", "union", "intersection"}
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(
                f"concatenation_strategy must be one of {allowed}, got '{v}'"
            )
        return v_lower

    @field_validator("strategy_params")
    @classmethod
    def validate_strategy_params(
        cls, v: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Validate strategy_params are appropriate for the database type.

        This validator can be extended as more database services are added.
        For now, ensures it's a dictionary when provided.
        """
        if v is None:
            return v

        # Ensure it's a dictionary
        if not isinstance(v, dict):
            raise ValueError("strategy_params must be a dictionary")

        return v

    def get_execution_params_with_defaults(self) -> Dict[str, Any]:
        """
        Get execution parameters with default values filled in.

        Returns:
            Dict[str, Any]: All execution parameters with defaults for unspecified values

        Example:
            >>> strategy = StrategyConfig(strategy_name="MATRIX_FIRST", ...)
            >>> params = strategy.get_execution_params_with_defaults()
            >>> print(params["timeout"])  # 3600
            >>> print(params["max_retries"])  # 3
        """
        defaults = {
            "timeout": 3600,
            "max_retries": 3,
            "retry_backoff": 2.0,
            "max_concurrent_downloads": 1,
            "verify_checksum": True,
            "resume_enabled": False,
        }

        if self.execution_params is None:
            return defaults

        # Merge user params with defaults (user params take precedence)
        return {**defaults, **self.execution_params}


class DownloadQueueEntry(BaseModel):
    """
    Queue entry for dataset download with full metadata.

    This schema represents a complete download request prepared by the research_agent
    with metadata, validation results, recommended strategy, and URLs. The data_expert
    consumes these entries to execute downloads.

    Attributes:
        entry_id: Unique identifier for this queue entry
        dataset_id: Dataset identifier (e.g., GSE12345, SRA123456, PXD034567)
        database: Source database (geo, sra, pride, metabolights, etc.)
        priority: Download priority (1=highest, 10=lowest)
        status: Current status (pending, in_progress, completed, failed)
        metadata: Full metadata extracted from source database
        validation_result: Validation results from metadata_assistant
        recommended_strategy: Strategy configuration recommended by research_agent
        created_at: Timestamp when entry was created
        updated_at: Timestamp when entry was last updated
        downloaded_by: Agent or user who executed the download
        modality_name: Name of resulting modality in DataManagerV2
        error_log: List of error messages if download failed
        matrix_url: URL for processed matrix file (if available)
        raw_urls: List of URLs for raw data files
        supplementary_urls: List of URLs for supplementary files
        h5_url: URL for H5 format file (if available)
    """

    # Core identification
    entry_id: str = Field(..., description="Unique identifier for this queue entry")
    dataset_id: str = Field(
        ..., description="Dataset identifier (e.g., GSE12345, SRA123456)"
    )
    database: str = Field(
        ..., description="Source database (geo, sra, pride, metabolights)"
    )

    # Priority and status
    priority: int = Field(
        default=5, ge=1, le=10, description="Download priority (1=highest, 10=lowest)"
    )
    status: DownloadStatus = Field(
        default=DownloadStatus.PENDING, description="Current download status"
    )

    # Prepared by research_agent
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Full metadata from source database"
    )
    validation_result: Optional[Dict[str, Any]] = Field(
        None, description="Validation results from metadata_assistant"
    )
    recommended_strategy: Optional[StrategyConfig] = Field(
        None, description="Strategy configuration recommended by research_agent"
    )

    # Execution metadata
    created_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp when entry was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when entry was last updated",
    )
    downloaded_by: Optional[str] = Field(
        None, description="Agent or user who executed the download"
    )
    modality_name: Optional[str] = Field(
        None, description="Name of resulting modality in DataManagerV2"
    )
    error_log: List[str] = Field(
        default_factory=list, description="List of error messages if download failed"
    )

    # URLs (prepared by research_agent)
    matrix_url: Optional[str] = Field(
        None, description="URL for processed matrix file (if available)"
    )
    raw_urls: List[str] = Field(
        default_factory=list, description="List of URLs for raw data files"
    )
    supplementary_urls: List[str] = Field(
        default_factory=list, description="List of URLs for supplementary files"
    )
    h5_url: Optional[str] = Field(
        None, description="URL for H5 format file (if available)"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "entry_id": "queue_entry_1234567890",
                "dataset_id": "GSE180759",
                "database": "geo",
                "priority": 5,
                "status": "pending",
                "metadata": {
                    "title": "Single-cell RNA-seq of human brain samples",
                    "organism": "Homo sapiens",
                    "sample_count": 8,
                },
                "validation_result": {
                    "is_valid": True,
                    "warnings": [],
                    "errors": [],
                },
                "recommended_strategy": {
                    "strategy_name": "MATRIX_FIRST",
                    "concatenation_strategy": "auto",
                    "confidence": 0.95,
                    "rationale": "Processed matrix available with complete metadata",
                },
                "created_at": "2024-01-15T10:30:00",
                "updated_at": "2024-01-15T10:30:00",
                "downloaded_by": None,
                "modality_name": None,
                "error_log": [],
                "matrix_url": "https://example.com/GSE180759_counts.h5",
                "raw_urls": [],
                "supplementary_urls": [],
                "h5_url": "https://example.com/GSE180759_counts.h5",
            }
        }

    @field_validator("entry_id", "dataset_id", "database")
    @classmethod
    def validate_not_empty(cls, v: str, info) -> str:
        """Validate that string fields are not empty."""
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        return v.strip()

    @field_validator("database")
    @classmethod
    def validate_database(cls, v: str) -> str:
        """Validate database is one of supported sources."""
        allowed = {
            "geo",
            "sra",
            "pride",
            "metabolights",
            "arrayexpress",
            "ega",
            "ebi",
        }
        v_lower = v.lower().strip()
        if v_lower not in allowed:
            raise ValueError(f"database must be one of {allowed}, got '{v}'")
        return v_lower

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
        status: DownloadStatus,
        modality_name: Optional[str] = None,
        error: Optional[str] = None,
        downloaded_by: Optional[str] = None,
    ) -> None:
        """
        Update entry status and related fields.

        Args:
            status: New status
            modality_name: Optional modality name if completed
            error: Optional error message if failed
            downloaded_by: Optional agent/user who executed download
        """
        self.status = status
        self.updated_at = datetime.now()

        if modality_name:
            self.modality_name = modality_name

        if error:
            self.error_log.append(f"[{datetime.now().isoformat()}] {error}")

        if downloaded_by:
            self.downloaded_by = downloaded_by

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation for JSON serialization.

        Returns:
            Dict[str, Any]: Dictionary with all fields serialized
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DownloadQueueEntry":
        """
        Create entry from dictionary.

        Args:
            data: Dictionary with entry data

        Returns:
            DownloadQueueEntry: Validated entry instance
        """
        # Handle datetime conversion if needed
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # Handle nested StrategyConfig
        if "recommended_strategy" in data and isinstance(
            data["recommended_strategy"], dict
        ):
            data["recommended_strategy"] = StrategyConfig(
                **data["recommended_strategy"]
            )

        return cls(**data)
