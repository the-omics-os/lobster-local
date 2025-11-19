"""
Download Orchestrator - Central routing for database-specific download services.

This module implements the orchestrator pattern for managing downloads from various
biological databases (GEO, SRA, PRIDE, etc.). The orchestrator:

1. Routes download requests to appropriate IDownloadService implementations
2. Manages DownloadQueue status transitions (PENDING → IN_PROGRESS → COMPLETED/FAILED)
3. Handles retry logic for failed downloads
4. Integrates with DataManagerV2 for storage and provenance tracking

Architecture:
    DownloadOrchestrator acts as a facade that:
    - Maintains registry of database-specific services
    - Executes queue-based downloads with proper error handling
    - Ensures consistent status tracking across all download types

Example usage:
    ```python
    # Initialize orchestrator
    orchestrator = DownloadOrchestrator(data_manager)

    # Register database-specific services
    orchestrator.register_service(GEODownloadService(data_manager))
    orchestrator.register_service(SRADownloadService(data_manager))

    # Execute download from queue
    try:
        modality_name, stats = orchestrator.execute_download(entry_id)
        print(f"Downloaded: {modality_name}")
        print(f"Stats: {stats}")
    except ServiceNotFoundError:
        print("No service registered for this database type")
    except ValueError as e:
        print(f"Invalid download request: {e}")
    ```

Thread Safety:
    This class is NOT thread-safe. If using in multi-threaded contexts,
    external synchronization is required around execute_download() calls.
"""

import traceback
from typing import Any, Dict, List, Optional, Tuple

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.interfaces.download_service import IDownloadService
from lobster.core.schemas.download_queue import DownloadStatus
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ServiceNotFoundError(Exception):
    """Raised when no download service is registered for a database type."""

    def __init__(self, database: str, available_databases: List[str]):
        self.database = database
        self.available_databases = available_databases
        message = (
            f"No download service registered for database '{database}'. "
            f"Available databases: {', '.join(available_databases) or 'none'}"
        )
        super().__init__(message)


class DownloadOrchestrator:
    """
    Central router for database-specific download services.

    Routes download requests to appropriate IDownloadService implementations
    based on database type, handles retry logic, and tracks queue status.

    The orchestrator maintains a registry of services and delegates download
    execution to the appropriate service based on the queue entry's database type.

    Attributes:
        data_manager: DataManagerV2 instance for storage and provenance
        _services: Registry mapping database names to download services
    """

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize DownloadOrchestrator with data manager.

        Args:
            data_manager: DataManagerV2 instance for data storage and provenance tracking
        """
        self.data_manager = data_manager
        self._services: Dict[str, IDownloadService] = {}
        self._register_default_services()
        logger.info("DownloadOrchestrator initialized")

    def _register_default_services(self) -> None:
        """
        Register default download services.

        Currently empty - placeholder for future automatic service registration.
        """
        # Future: auto-register GEODownloadService, SRADownloadService, etc.
        # This will allow plug-and-play service registration without manual setup
        pass

    def register_service(self, service: IDownloadService) -> None:
        """
        Register a download service for one or more databases.

        A single service can support multiple database types (e.g., a unified
        NCBI service might handle both GEO and SRA).

        Args:
            service: IDownloadService implementation to register

        Raises:
            ValueError: If service does not support any databases
        """
        supported_dbs = service.supported_databases()

        if not supported_dbs:
            raise ValueError(
                f"Service {service.__class__.__name__} does not declare "
                "any supported databases"
            )

        # Register service for each database it supports
        for db in supported_dbs:
            db_lower = db.lower()
            if db_lower in self._services:
                logger.warning(
                    f"Overwriting existing service for database '{db}' "
                    f"(old: {self._services[db_lower].__class__.__name__}, "
                    f"new: {service.__class__.__name__})"
                )
            self._services[db_lower] = service
            logger.info(f"Registered {service.__class__.__name__} for database '{db}'")

    def get_service_for_database(self, database: str) -> Optional[IDownloadService]:
        """
        Get registered service for database type.

        Args:
            database: Database identifier (case-insensitive, e.g., "GEO", "SRA")

        Returns:
            Registered IDownloadService instance, or None if not found
        """
        return self._services.get(database.lower())

    def list_supported_databases(self) -> List[str]:
        """
        List all databases supported by registered services.

        Returns:
            Sorted list of database identifiers (lowercase)
        """
        return sorted(self._services.keys())

    def execute_download(
        self, entry_id: str, strategy_override: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute download from queue entry with automatic service routing.

        This method orchestrates the complete download lifecycle:
        1. Retrieves queue entry and validates status
        2. Routes to appropriate service based on database type
        3. Updates queue status throughout execution
        4. Stores downloaded data in DataManagerV2
        5. Logs provenance information
        6. Handles errors with proper queue status updates

        Args:
            entry_id: Queue entry identifier
            strategy_override: Optional dict to override default download strategy.
                Must match the parameter schema of the target service.

        Returns:
            Tuple containing:
                - modality_name: Name of stored modality in DataManagerV2
                - stats: Download statistics dict (keys vary by service)

        Raises:
            ValueError: If entry_id invalid, entry status is not PENDING/FAILED,
                or strategy_override params are invalid
            ServiceNotFoundError: If no service registered for entry's database type
        """
        logger.info(f"Starting download execution for entry '{entry_id}'")

        # Step 1: Retrieve and validate queue entry
        entry = self.data_manager.download_queue.get_entry(entry_id)
        if entry is None:
            raise ValueError(f"Queue entry '{entry_id}' not found")

        # Only allow downloads from PENDING or FAILED states (retry logic)
        if entry.status not in [DownloadStatus.PENDING, DownloadStatus.FAILED]:
            # Handle status as either Enum or string for error message
            status_str = (
                entry.status.value if hasattr(entry.status, "value") else entry.status
            )
            raise ValueError(
                f"Cannot execute download for entry '{entry_id}' with status "
                f"'{status_str}'. Only PENDING or FAILED entries can be processed."
            )

        # Handle status as either Enum or string
        status_str = (
            entry.status.value if hasattr(entry.status, "value") else entry.status
        )
        logger.info(
            f"Queue entry '{entry_id}': database={entry.database}, "
            f"dataset_id={entry.dataset_id}, status={status_str}"
        )

        # Step 2: Update status to IN_PROGRESS
        self.data_manager.download_queue.update_status(
            entry_id, DownloadStatus.IN_PROGRESS
        )
        logger.info(f"Updated queue entry '{entry_id}' status to IN_PROGRESS")

        try:
            # Step 3: Detect and retrieve appropriate service
            service = self.get_service_for_database(entry.database)
            if service is None:
                available = self.list_supported_databases()
                raise ServiceNotFoundError(entry.database, available)

            logger.info(
                f"Routing download to {service.__class__.__name__} "
                f"for database '{entry.database}'"
            )

            # Step 4: Validate strategy override if provided
            if strategy_override:
                try:
                    service.validate_strategy(strategy_override)
                    logger.info(f"Strategy override validated: {strategy_override}")
                except Exception as e:
                    raise ValueError(
                        f"Invalid strategy_override parameters: {str(e)}"
                    ) from e

            # Step 5: Execute download via service
            adata, stats, ir = service.download_dataset(entry, strategy_override)

            # Step 6: Store result in DataManagerV2
            modality_name = (
                entry.modality_name or f"{entry.database.lower()}_{entry.dataset_id}"
            )
            self.data_manager.modalities[modality_name] = adata
            logger.info(
                f"Stored downloaded data as modality '{modality_name}' "
                f"({adata.n_obs} obs × {adata.n_vars} vars)"
            )

            # Step 7: Log provenance
            self.data_manager.log_tool_usage(
                "download_orchestrator.execute_download",
                {
                    "entry_id": entry_id,
                    "database": entry.database,
                    "dataset_id": entry.dataset_id,
                    "strategy_override": strategy_override or {},
                    "service": service.__class__.__name__,
                },
                stats,
                ir=ir,
            )
            logger.info("Provenance logged for download operation")

            # Step 8: Update queue status to COMPLETED
            self.data_manager.download_queue.update_status(
                entry_id, DownloadStatus.COMPLETED, modality_name=modality_name
            )
            logger.info(
                f"Download completed successfully for entry '{entry_id}' → '{modality_name}'"
            )

            return modality_name, stats

        except Exception as e:
            # Step 9: Error handling - update queue and re-raise
            error_log = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(
                f"Download failed for entry '{entry_id}': {str(e)}", exc_info=True
            )

            # Update queue status to FAILED with error details
            self.data_manager.download_queue.update_status(
                entry_id, DownloadStatus.FAILED, error=error_log
            )
            logger.info(f"Updated queue entry '{entry_id}' status to FAILED")

            # Re-raise exception for caller to handle
            raise
