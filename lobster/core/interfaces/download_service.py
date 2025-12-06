"""
Download Service Interface for DownloadOrchestrator pattern.

This module defines the abstract interface for database-specific download implementations,
enabling a clean separation between download orchestration (DownloadOrchestrator) and
database-specific download logic (GEODownloadService, SRADownloadService, etc.).

The DownloadOrchestrator uses this interface to:
1. Detect which service handles a given database type
2. Validate strategy parameters before execution
3. Execute database-specific download strategies
4. Handle errors consistently across all download services

Usage Example:
    >>> from lobster.core.interfaces.download_service import IDownloadService
    >>> from lobster.core.data_manager_v2 import DataManagerV2
    >>> from lobster.core.schemas.download_queue import DownloadQueueEntry
    >>>
    >>> # Implementation (e.g., GEODownloadService)
    >>> class GEODownloadService(IDownloadService):
    ...     @classmethod
    ...     def supports_database(cls, database: str) -> bool:
    ...         return database.lower() == "geo"
    ...
    ...     def download_dataset(self, queue_entry, strategy_override=None):
    ...         # GEO-specific download logic
    ...         ...
    ...         return adata, stats, ir
    ...
    ...     # ... implement other abstract methods
    >>>
    >>> # Usage in DownloadOrchestrator
    >>> dm = DataManagerV2(workspace_dir="./workspace")
    >>> service = GEODownloadService(dm)
    >>> if service.supports_database("geo"):
    ...     adata, stats, ir = service.download_dataset(queue_entry)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import anndata as ad

if TYPE_CHECKING:
    from lobster.core.data_manager_v2 import DataManagerV2
    from lobster.core.provenance import AnalysisStep
    from lobster.core.schemas.download_queue import DownloadQueueEntry


class IDownloadService(ABC):
    """
    Abstract base class for database-specific download services.

    This interface defines the contract that all download service implementations
    must follow to integrate with the DownloadOrchestrator pattern. Each implementation
    handles downloading and loading datasets from a specific bioinformatics database
    (GEO, SRA, PRIDE, MetaboLights, etc.) using database-specific strategies and
    best practices.

    Key Responsibilities:
        - Database detection: Identify if this service handles a given database type
        - Strategy execution: Download datasets using recommended or overridden strategies
        - Parameter validation: Ensure strategy parameters are valid before execution
        - Error handling: Provide consistent error reporting with retry support
        - Provenance tracking: Generate AnalysisStep IR for reproducibility

    Attributes:
        data_manager (DataManagerV2): Data manager for storing downloaded datasets
            and tracking provenance.

    Design Notes:
        - All implementations must be stateless beyond the data_manager reference
        - The 3-tuple return pattern (adata, stats, ir) follows service conventions
        - Strategy overrides allow runtime flexibility while maintaining type safety
        - Validation is separate from execution for early failure detection
    """

    def __init__(self, data_manager: "DataManagerV2"):
        """
        Initialize the download service with a data manager.

        Args:
            data_manager: DataManagerV2 instance for data storage and provenance tracking
        """
        self.data_manager = data_manager

    @abstractmethod
    def supported_databases(self) -> List[str]:
        """
        Get list of database identifiers this service handles.

        This method enables the DownloadOrchestrator to route download requests
        to the appropriate service implementation based on the database identifier.

        Returns:
            List[str]: List of database identifiers (lowercase). Examples:
                - ["geo"] - GEO service
                - ["sra", "ena", "ddbj"] - SRA service (handles multiple archives)
                - ["pride", "pxd"] - PRIDE service
                - ["massive", "msv"] - MassIVE service

        Example:
            >>> service = GEODownloadService(data_manager)
            >>> service.supported_databases()
            ["geo"]
            >>> service = PRIDEDownloadService(data_manager)
            >>> service.supported_databases()
            ["pride", "pxd"]
        """
        pass

    @classmethod
    def supports_database(cls, database: str) -> bool:
        """
        Check if this service class handles the given database type.

        This is a convenience class method. For instance-level checks,
        use `supported_databases()` instead.

        Args:
            database: Database identifier (case-insensitive).

        Returns:
            bool: True if this service can handle the database, False otherwise

        Note:
            Default implementation returns False. Subclasses should override
            if they want to support class-level database detection.
        """
        return False

    @abstractmethod
    def download_dataset(
        self,
        queue_entry: "DownloadQueueEntry",
        strategy_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ad.AnnData, Dict[str, Any], "AnalysisStep"]:
        """
        Download and load a dataset from the database.

        This is the main execution method that performs the actual download,
        loading, and initial processing of a dataset. It follows the standard
        service pattern of returning a 3-tuple for provenance tracking and
        notebook export.

        Args:
            queue_entry: Queue entry prepared by research_agent containing:
                - dataset_id: Database-specific identifier (e.g., GSE180759)
                - database: Database type (must be supported by this service)
                - metadata: Full metadata extracted from database
                - recommended_strategy: Strategy configuration from research_agent
                - URLs: matrix_url, raw_urls, h5_url, supplementary_urls
                - validation_result: Validation results from metadata_assistant

            strategy_override: Optional dictionary to override queue entry strategy.
                If provided, should contain:
                - strategy_params: Strategy-specific parameters (e.g.,
                  use_intersecting_genes_only=True for GEO, layout="PAIRED"
                  for SRA)
                - execution_params: Execution parameters (e.g., max_retries=3,
                  timeout_seconds=300, chunk_size=8192)

        Returns:
            Tuple containing:
                - adata (AnnData): Loaded and initially processed dataset with:
                    - .X: Primary data matrix
                    - .obs: Sample/observation metadata
                    - .var: Feature/variable metadata
                    - .uns: Unstructured metadata including provenance

                - stats (Dict[str, Any]): Human-readable download summary with:
                    - dataset_id: Dataset identifier
                    - database: Source database
                    - strategy_used: Strategy name that was executed
                    - n_obs: Number of observations (samples/cells)
                    - n_vars: Number of variables (genes/proteins/features)
                    - download_time_seconds: Time taken for download
                    - file_size_bytes: Total downloaded file size
                    - warnings: List of non-fatal issues encountered

                - ir (AnalysisStep): Intermediate representation for provenance,
                  containing:
                    - operation: Download operation identifier
                    - tool_name: Service class name
                    - library: Data access library used (e.g., "GEOparse", "pysradb")
                    - parameters: All parameters used (from queue + override)
                    - code_template: Jinja2 template for notebook reproduction
                    - imports: Required Python imports
                    - input_entities: Source URLs and identifiers
                    - output_entities: Resulting modality names

        Raises:
            ValueError: If queue_entry database doesn't match this service or
                strategy parameters are invalid
            FileNotFoundError: If download URLs are inaccessible or files missing
            TimeoutError: If download exceeds timeout threshold
            PermissionError: If authentication is required but not provided
            ConnectionError: If network connectivity issues occur
            RuntimeError: If download succeeds but data loading/parsing fails

        Example:
            >>> entry = DownloadQueueEntry(
            ...     entry_id="queue_123",
            ...     dataset_id="GSE180759",
            ...     database="geo",
            ...     recommended_strategy=StrategyConfig(
            ...         strategy_name="H5_FIRST",
            ...         concatenation_strategy="auto",
            ...         confidence=0.95
            ...     ),
            ...     h5_url="https://example.com/GSE180759.h5"
            ... )
            >>> service = GEODownloadService(data_manager)
            >>> adata, stats, ir = service.download_dataset(entry)
            >>> print(stats)
            {
                "dataset_id": "GSE180759",
                "database": "geo",
                "strategy_used": "H5_FIRST",
                "n_obs": 12000,
                "n_vars": 20000,
                "download_time_seconds": 45.2,
                "file_size_bytes": 524288000,
                "warnings": []
            }

        Design Notes:
            - Implementations should prefer queue_entry.recommended_strategy unless
              strategy_override is explicitly provided
            - All downloaded files should be cached in data_manager workspace
            - The AnalysisStep IR must be fully reproducible with only standard libraries
            - Long-running downloads should log progress for user feedback
            - Failed downloads should be retryable (preserve idempotency)
        """
        pass

    @abstractmethod
    def validate_strategy_params(
        self, strategy_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate strategy-specific parameters before execution.

        This method enables early validation of parameters without executing
        the download, allowing the DownloadOrchestrator to fail fast with
        clear error messages when invalid parameters are provided.

        Args:
            strategy_params: Dictionary of strategy-specific parameters to validate.
                Examples:
                - GEO: {"use_intersecting_genes_only": True, "include_supplementary": False}
                - SRA: {"layout": "PAIRED", "platform": "ILLUMINA", "file_type": "fastq"}
                - PRIDE: {"file_type": "mzML", "include_raw": False}
                - MetaboLights: {"file_format": "mzTab", "include_metadata": True}

        Returns:
            Tuple containing:
                - is_valid (bool): True if all parameters are valid, False otherwise
                - error_message (Optional[str]): Descriptive error message if invalid,
                  None if valid. Should specify which parameter(s) are invalid and why.

        Example:
            >>> service = GEODownloadService(data_manager)
            >>> # Valid parameters
            >>> valid, error = service.validate_strategy_params({
            ...     "use_intersecting_genes_only": True,
            ...     "include_supplementary": False
            ... })
            >>> assert valid is True and error is None
            >>>
            >>> # Invalid parameters
            >>> valid, error = service.validate_strategy_params({
            ...     "use_intersecting_genes_only": "invalid_type"  # should be bool
            ... })
            >>> assert valid is False
            >>> print(error)
            "Parameter 'use_intersecting_genes_only' must be bool, got str"

        Design Notes:
            - Validation should check parameter types, allowed values, and constraints
            - Error messages should be specific and actionable
            - Unknown parameters should be ignored (forward compatibility)
            - This method should NOT access network resources or filesystem
        """
        pass

    def validate_strategy(self, strategy_override: Dict[str, Any]) -> None:
        """
        Validate strategy override parameters, raising exception on failure.

        This is a convenience wrapper around validate_strategy_params() that
        raises an exception instead of returning a tuple. Used by DownloadOrchestrator.

        Args:
            strategy_override: Dictionary of strategy parameters to validate

        Raises:
            ValueError: If validation fails, with descriptive error message
        """
        is_valid, error_message = self.validate_strategy_params(strategy_override)
        if not is_valid:
            raise ValueError(error_message or "Invalid strategy parameters")

    @classmethod
    @abstractmethod
    def get_supported_strategies(cls) -> List[str]:
        """
        Get list of download strategy names supported by this service.

        Each database typically supports multiple download strategies optimized
        for different scenarios. This method enables the DownloadOrchestrator
        to validate that requested strategies are actually supported before
        execution.

        Returns:
            List[str]: List of strategy names (uppercase convention). Examples:
                - GEO: ["H5_FIRST", "MATRIX_FIRST", "SAMPLES_FIRST"]
                - SRA: ["RAW_FIRST", "PROCESSED_FIRST", "HYBRID"]
                - PRIDE: ["MZML_FIRST", "RAW_FIRST", "PROCESSED_FIRST"]
                - MetaboLights: ["MZTAB_FIRST", "RAW_FIRST", "STUDY_FIRST"]

        Example:
            >>> GEODownloadService.get_supported_strategies()
            ["H5_FIRST", "MATRIX_FIRST", "SAMPLES_FIRST"]
            >>> SRADownloadService.get_supported_strategies()
            ["RAW_FIRST", "PROCESSED_FIRST", "HYBRID"]

        Design Notes:
            - Strategy names should be descriptive and self-documenting
            - Use uppercase for consistency across implementations
            - Order should reflect preference (most efficient first)
            - Keep strategy list stable across versions for reproducibility
        """
        pass

    # Optional helper methods (non-abstract, can be overridden)

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about this download service.

        Returns:
            Dict[str, Any]: Service metadata including:
                - service_name: Class name
                - supported_databases: List of database identifiers
                - supported_strategies: List of strategy names
                - version: Service version (if applicable)

        Example:
            >>> service = GEODownloadService(data_manager)
            >>> info = service.get_service_info()
            >>> print(info)
            {
                "service_name": "GEODownloadService",
                "supported_databases": ["geo"],
                "supported_strategies": ["H5_FIRST", "MATRIX_FIRST", "SAMPLES_FIRST"],
                "version": "1.0.0"
            }
        """
        return {
            "service_name": self.__class__.__name__,
            "supported_databases": self.supported_databases(),
            "supported_strategies": self.get_supported_strategies(),
            "version": "1.0.0",
        }

    def estimate_download_size(self, queue_entry: "DownloadQueueEntry") -> int:
        """
        Estimate total download size in bytes.

        This method can be overridden by implementations that can efficiently
        determine file sizes without downloading. Useful for progress tracking
        and disk space validation.

        Args:
            queue_entry: Queue entry containing URLs to estimate

        Returns:
            int: Estimated size in bytes, or -1 if unknown

        Example:
            >>> entry = DownloadQueueEntry(...)
            >>> service = GEODownloadService(data_manager)
            >>> size_bytes = service.estimate_download_size(entry)
            >>> print(f"Estimated download: {size_bytes / 1e6:.1f} MB")
            Estimated download: 524.3 MB
        """
        # Default implementation - subclasses can override with HEAD requests
        return -1

    def supports_resume(self) -> bool:
        """
        Check if this service supports resumable downloads.

        Returns:
            bool: True if downloads can be resumed after interruption,
                False otherwise (default)

        Example:
            >>> service = GEODownloadService(data_manager)
            >>> if service.supports_resume():
            ...     print("Partial downloads can be resumed")
        """
        return False
