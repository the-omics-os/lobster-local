"""
Publication queue management with JSON Lines persistence.

This module provides the PublicationQueue class for managing publication extraction
requests with atomic operations, automatic backups, and thread-safe access.
"""

import json
import logging
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from lobster.core.queue_storage import (
    atomic_write_jsonl,
    backups_enabled,
    queue_file_lock,
)
from lobster.core.schemas.publication_queue import (
    HandoffStatus,
    PublicationQueueEntry,
    PublicationStatus,
)

logger = logging.getLogger(__name__)


class PublicationQueueError(Exception):
    """Base exception for publication queue operations."""

    pass


class EntryNotFoundError(PublicationQueueError):
    """Raised when a queue entry is not found."""

    pass


class PublicationQueue:
    """
    Thread-safe publication queue with JSON Lines persistence.

    This class manages a queue of publication extraction requests with automatic
    persistence to disk, atomic operations, backup functionality, and
    thread-safe access for concurrent operations.

    Features:
        - JSON Lines (.jsonl) format for append-only durability
        - Automatic backups before modifications
        - Schema validation on read/write
        - Thread-safe operations with file locking
        - Efficient filtering by status
        - Atomic writes (temp file + rename)

    Attributes:
        queue_file: Path to JSON Lines queue file
        backup_dir: Directory for backup files
    """

    def __init__(self, queue_file: Path):
        """
        Initialize publication queue with JSON Lines file.

        Args:
            queue_file: Path to queue file (will be created if doesn't exist)
        """
        self.queue_file = Path(queue_file)
        # Use shared backup directory in queues folder
        self.backup_dir = self.queue_file.parent / "backups"
        self._lock = threading.Lock()
        self._file_lock_path = self.queue_file.with_suffix(".lock")

        # Create directories if they don't exist
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize empty queue file if it doesn't exist
        if not self.queue_file.exists():
            self.queue_file.touch()

        logger.debug(f"Initialized PublicationQueue at {self.queue_file}")

    @contextmanager
    def _locked(self):
        with queue_file_lock(self._lock, self._file_lock_path):
            yield

    def add_entry(self, entry: PublicationQueueEntry) -> str:
        """
        Add entry to queue with atomic write.

        BUG FIX #5: Use atomic write pattern (temp file + rename) instead of direct append.
        This prevents queue corruption if process is interrupted (Ctrl+C, crash, kill).

        Args:
            entry: PublicationQueueEntry to add

        Returns:
            str: Entry ID of added entry

        Raises:
            PublicationQueueError: If entry already exists or write fails
        """
        with self._locked():
            # Check if entry already exists
            existing_entries = self._load_entries()
            if any(e.entry_id == entry.entry_id for e in existing_entries):
                raise PublicationQueueError(
                    f"Entry with ID '{entry.entry_id}' already exists"
                )

            # Backup before modification
            self._backup_queue()

            # BUG FIX #5: Atomic write - add entry to list and write all atomically
            # Old code used direct append which could leave partial JSON on interruption
            try:
                existing_entries.append(entry)
                self._write_entries_atomic(existing_entries)

                logger.debug(f"Added entry {entry.entry_id} to publication queue")
                return entry.entry_id

            except Exception as e:
                logger.error(f"Failed to add entry to publication queue: {e}")
                raise PublicationQueueError(f"Failed to add entry: {e}") from e

    def import_entries(
        self,
        entries: List[PublicationQueueEntry],
        skip_duplicates: bool = True,
    ) -> dict:
        """
        Import entries from external source (e.g., exported queue file).

        This method is designed for restoring previously exported queues.
        By default, it skips entries that already exist (by entry_id).

        Args:
            entries: List of PublicationQueueEntry objects to import
            skip_duplicates: If True, skip entries with existing entry_id.
                           If False, raises PublicationQueueError on duplicates.

        Returns:
            dict: Import statistics with keys:
                - imported: Number of entries successfully imported
                - skipped: Number of duplicate entries skipped
                - errors: Number of entries that failed to import

        Raises:
            PublicationQueueError: If skip_duplicates=False and duplicates found,
                                  or if write operation fails
        """
        if not entries:
            return {"imported": 0, "skipped": 0, "errors": 0}

        with self._locked():
            # Load existing entries
            existing_entries = self._load_entries()
            existing_ids = {e.entry_id for e in existing_entries}

            # Track statistics
            imported = 0
            skipped = 0
            errors = 0

            # Process each entry
            for entry in entries:
                try:
                    if entry.entry_id in existing_ids:
                        if skip_duplicates:
                            skipped += 1
                            logger.debug(
                                f"Skipped duplicate entry: {entry.entry_id}"
                            )
                            continue
                        else:
                            raise PublicationQueueError(
                                f"Entry with ID '{entry.entry_id}' already exists"
                            )

                    # Add to list and track ID
                    existing_entries.append(entry)
                    existing_ids.add(entry.entry_id)
                    imported += 1

                except PublicationQueueError:
                    raise  # Re-raise if not skipping duplicates
                except Exception as e:
                    errors += 1
                    logger.error(f"Failed to import entry {entry.entry_id}: {e}")

            # Only write if we imported something
            if imported > 0:
                # Backup before modification
                self._backup_queue()

                # Write all entries atomically
                self._write_entries_atomic(existing_entries)

                logger.info(
                    f"Imported {imported} entries to publication queue "
                    f"(skipped {skipped} duplicates, {errors} errors)"
                )

            return {"imported": imported, "skipped": skipped, "errors": errors}

    def get_entry(self, entry_id: str) -> PublicationQueueEntry:
        """
        Retrieve specific entry by ID.

        Args:
            entry_id: Unique entry identifier

        Returns:
            PublicationQueueEntry: Retrieved entry

        Raises:
            EntryNotFoundError: If entry not found
        """
        with self._lock:
            entries = self._load_entries()
            for entry in entries:
                if entry.entry_id == entry_id:
                    return entry

            raise EntryNotFoundError(
                f"Entry '{entry_id}' not found in publication queue"
            )

    def update_status(
        self,
        entry_id: str,
        status: PublicationStatus,
        cached_content_path: Optional[str] = None,
        error: Optional[str] = None,
        processed_by: Optional[str] = None,
        extracted_identifiers: Optional[dict] = None,
        workspace_metadata_keys: Optional[List[str]] = None,
        dataset_ids: Optional[List[str]] = None,
        filtered_workspace_key: Optional[str] = None,
        handoff_status: Optional["HandoffStatus"] = None,
        harmonization_metadata: Optional[dict] = None,
        pmc_id: Optional[str] = None,
        pmid: Optional[str] = None,
    ) -> PublicationQueueEntry:
        """
        Update entry status and optional fields.

        Args:
            entry_id: Entry to update
            status: New status
            cached_content_path: Optional path to cached content
            error: Optional error message
            processed_by: Optional agent/user identifier
            extracted_identifiers: Optional extracted dataset identifiers
            workspace_metadata_keys: Optional list of workspace metadata file basenames
            pmc_id: Optional PMC ID discovered during enrichment
            pmid: Optional PubMed ID discovered during enrichment

        Returns:
            PublicationQueueEntry: Updated entry

        Raises:
            EntryNotFoundError: If entry not found
            PublicationQueueError: If update fails
        """
        if isinstance(status, str):
            status = PublicationStatus(status)

        with self._locked():
            # Load all entries
            entries = self._load_entries()
            entry_found = False
            updated_entry = None

            # Find and update entry
            for entry in entries:
                if entry.entry_id == entry_id:
                    entry_found = True
                    entry.update_status(
                        status=status,
                        cached_content_path=cached_content_path,
                        error=error,
                        processed_by=processed_by,
                        extracted_identifiers=extracted_identifiers,
                        workspace_metadata_keys=workspace_metadata_keys,
                        dataset_ids=dataset_ids,
                        filtered_workspace_key=filtered_workspace_key,
                        handoff_status=handoff_status,
                        harmonization_metadata=harmonization_metadata,
                        pmc_id=pmc_id,
                        pmid=pmid,
                    )
                    updated_entry = entry
                    break

            if not entry_found:
                raise EntryNotFoundError(
                    f"Entry '{entry_id}' not found in publication queue"
                )

            # Backup before modification
            self._backup_queue()

            # Write all entries atomically
            self._write_entries_atomic(entries)

            logger.debug(f"Updated entry {entry_id} status to {status}")
            return updated_entry

    def list_entries(
        self, status: Optional[PublicationStatus] = None
    ) -> List[PublicationQueueEntry]:
        """
        List all entries, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List[PublicationQueueEntry]: List of entries
        """
        with self._lock:
            entries = self._load_entries()

            if status is not None:
                entries = [e for e in entries if e.status == status]

            return entries

    def remove_entry(self, entry_id: str) -> None:
        """
        Remove entry from queue.

        Args:
            entry_id: Entry to remove

        Raises:
            EntryNotFoundError: If entry not found
            PublicationQueueError: If removal fails
        """
        with self._locked():
            entries = self._load_entries()
            original_count = len(entries)

            # Filter out entry
            entries = [e for e in entries if e.entry_id != entry_id]

            if len(entries) == original_count:
                raise EntryNotFoundError(
                    f"Entry '{entry_id}' not found in publication queue"
                )

            # Backup before modification
            self._backup_queue()

            # Write remaining entries atomically
            self._write_entries_atomic(entries)

            logger.debug(f"Removed entry {entry_id} from publication queue")

    def clear_queue(self) -> int:
        """
        Clear all entries from queue.

        Returns:
            int: Number of entries cleared

        Raises:
            PublicationQueueError: If clear operation fails
        """
        with self._locked():
            entries = self._load_entries()
            entry_count = len(entries)

            if entry_count > 0:
                # Backup before modification
                self._backup_queue()

                # Clear queue file
                try:
                    self.queue_file.write_text("", encoding="utf-8")
                    logger.debug(f"Cleared {entry_count} entries from publication queue")
                except Exception as e:
                    logger.error(f"Failed to clear publication queue: {e}")
                    raise PublicationQueueError(f"Failed to clear queue: {e}") from e

            return entry_count

    def _load_entries(self) -> List[PublicationQueueEntry]:
        """
        Load all entries from queue file with corruption detection.

        BUG FIX #5: Enhanced corruption detection and recovery.
        Detects partial JSON writes from interrupted operations and continues
        parsing rest of file. Provides detailed logging for debugging.

        Returns:
            List[PublicationQueueEntry]: List of valid entries (corrupted entries skipped)

        Raises:
            PublicationQueueError: If loading fails completely
        """
        entries = []
        corrupted_count = 0

        if not self.queue_file.exists():
            return entries

        try:
            with open(self.queue_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        entry = PublicationQueueEntry.from_dict(data)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        # BUG FIX #5: Detailed corruption logging for JSON parse errors
                        corrupted_count += 1
                        line_preview = line[:100] + "..." if len(line) > 100 else line
                        logger.error(
                            f"Corrupted JSON at line {line_num} (likely interrupted write): {e}. "
                            f"Content preview: {line_preview}"
                        )
                        continue
                    except Exception as e:
                        # BUG FIX #5: Catch schema validation errors separately
                        corrupted_count += 1
                        logger.error(
                            f"Invalid entry schema at line {line_num}: {e}. "
                            f"Entry may be from old version or corrupted."
                        )
                        continue

            # BUG FIX #5: Log corruption summary if any entries were skipped
            if corrupted_count > 0:
                logger.warning(
                    f"Loaded {len(entries)} valid entries, skipped {corrupted_count} corrupted entries. "
                    f"Queue may have been interrupted during previous write operation."
                )

            return entries

        except Exception as e:
            logger.error(f"Failed to load publication queue: {e}")
            raise PublicationQueueError(f"Failed to load queue: {e}") from e

    def _write_entries_atomic(self, entries: List[PublicationQueueEntry]) -> None:
        """
        Write entries atomically using temp file + rename.

        Args:
            entries: List of entries to write

        Raises:
            PublicationQueueError: If write fails
        """
        logger.debug(f"Writing {len(entries)} entries atomically to {self.queue_file}")

        try:
            atomic_write_jsonl(self.queue_file, entries, lambda e: e.to_dict())
            file_size = self.queue_file.stat().st_size
            logger.debug(
                f"Successfully wrote {len(entries)} entries ({file_size} bytes)"
            )

        except Exception as e:
            logger.error(f"Failed to write publication queue atomically: {e}")
            raise PublicationQueueError(f"Failed to write queue: {e}") from e

    def _backup_queue(self) -> Path:
        """
        Create timestamped backup of queue file.

        NOTE: Backups disabled for performance. The atomic write pattern
        (temp file + rename) provides sufficient protection against corruption.
        Re-enable by removing the early return below.

        Returns:
            Path: Path to backup file (or None if backup skipped/disabled)

        Raises:
            PublicationQueueError: If backup fails (non-fatal, logged as warning)
        """
        if not backups_enabled():
            return None

        if not self.queue_file.exists() or self.queue_file.stat().st_size == 0:
            # No need to backup empty file
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_file = self.backup_dir / f"publication_queue_backup_{timestamp}.jsonl"

        try:
            shutil.copy2(self.queue_file, backup_file)
            logger.debug(f"Created publication queue backup: {backup_file}")
            return backup_file

        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
            # Don't raise - backup failure shouldn't stop operations
            return None

    def get_statistics(self) -> dict:
        """
        Get queue statistics.

        Returns:
            dict: Statistics including counts by status and schema_type
        """
        with self._lock:
            entries = self._load_entries()

            # Initialize with ALL PublicationStatus values
            stats = {
                "total_entries": len(entries),
                "by_status": {status.value: 0 for status in PublicationStatus},
                "by_schema_type": {},
                "by_extraction_level": {},
                "identifiers_extracted": 0,
            }

            for entry in entries:
                # Count by status
                stats["by_status"][entry.status] += 1

                # Count by schema_type
                if entry.schema_type not in stats["by_schema_type"]:
                    stats["by_schema_type"][entry.schema_type] = 0
                stats["by_schema_type"][entry.schema_type] += 1

                # Count by extraction_level
                level_str = str(entry.extraction_level)
                if level_str not in stats["by_extraction_level"]:
                    stats["by_extraction_level"][level_str] = 0
                stats["by_extraction_level"][level_str] += 1

                # Count entries with extracted identifiers
                if entry.has_identifier_data():
                    stats["identifiers_extracted"] += 1

            return stats
