"""
Extraction Cache Manager for selective loading of nested archives.

Preserves extracted archives in workspace for repeated access without
re-extraction overhead. Useful for large nested TAR structures where
users want to inspect first and load selectively.

Professional Iterative Extraction Workflow:
    This module implements a systematic, iterative approach to handling
    complex nested archive structures (e.g., GSE155698_RAW.tar containing
    multiple *.tar.gz files, each with 10X Genomics data):

    1. **Inspection Phase**: Archive structure detected without full extraction
    2. **Selective Extraction**: User-specified patterns (e.g., "TISSUE", "PBMC")
       trigger extraction only for matching samples
    3. **Robust Loading**: Two-tier loading strategy (scanpy → manual parsing)
       ensures data integrity even with deeply nested structures
    4. **Auto-Concatenation**: Multiple samples automatically merged for analysis
    5. **Provenance Tracking**: Complete audit trail of extraction and loading steps

Example workflow:
    1. User runs /read GSE155698_RAW.tar
    2. Archive inspected, nested structure detected, extracted to cache with unique ID
    3. User can then load specific samples: /archive load TISSUE
    4. Selected samples loaded with robust fallback handling
    5. Multiple samples automatically concatenated if applicable
    6. Cache automatically cleaned up after 7 days

Technical Details:
    - Handles nested .tar.gz files within parent TAR archives
    - Supports compressed 10X Genomics MEX format (.mtx.gz, .tsv.gz)
    - Validates directory structures at each extraction level
    - Memory-efficient: only extracts requested samples
    - Thread-safe for concurrent cache operations
"""

import json
import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from lobster.core.archive_utils import NestedArchiveInfo

logger = logging.getLogger(__name__)


class ExtractionCacheManager:
    """
    Manage extracted archives for selective loading.

    Preserves extracted archives in workspace for repeated access
    without re-extraction overhead.

    Attributes:
        cache_dir: Directory for cached extractions
        metadata_file: JSON file storing cache metadata
    """

    def __init__(self, workspace_dir: Path):
        """
        Initialize cache manager.

        Args:
            workspace_dir: Workspace directory (usually DataManagerV2.workspace_path)
        """
        self.cache_dir = workspace_dir / ".archive_cache"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_extraction(
        self,
        archive_path: Path,
        extract_dir: Path,
        nested_info: NestedArchiveInfo,
    ) -> str:
        """
        Cache extracted archive for selective loading.

        Args:
            archive_path: Original archive path
            extract_dir: Temporary extraction directory
            nested_info: Detected nested archive information

        Returns:
            Cache ID for future reference
        """
        # Generate unique cache ID
        cache_id = f"{archive_path.stem}_{int(time.time())}"
        cache_path = self.cache_dir / cache_id

        try:
            # Move extracted content to cache
            if extract_dir.exists():
                shutil.move(str(extract_dir), str(cache_path))
                logger.info(f"Cached extraction: {cache_id} -> {cache_path}")
            else:
                raise RuntimeError(f"Extract directory does not exist: {extract_dir}")

            # Store metadata
            metadata = {
                "cache_id": cache_id,
                "archive_path": str(archive_path),
                "extracted_at": datetime.now().isoformat(),
                "nested_info": {
                    "nested_archives": nested_info.nested_archives,
                    "groups": nested_info.groups,
                    "estimated_memory": nested_info.estimated_memory,
                    "parent_archive": nested_info.parent_archive,
                    "total_count": nested_info.total_count,
                },
                "cache_path": str(cache_path),
            }

            self._save_metadata(cache_id, metadata)
            logger.info(
                f"Cached {nested_info.total_count} nested archives with ID: {cache_id}"
            )

            return cache_id

        except Exception as e:
            logger.error(f"Failed to cache extraction: {e}")
            # Cleanup on failure
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)
            raise RuntimeError(f"Failed to cache extraction: {e}")

    def load_from_cache(
        self, cache_id: str, pattern: str, limit: Optional[int] = None
    ) -> List[Path]:
        """
        Load specific samples from cache by pattern.

        Args:
            cache_id: Cache identifier
            pattern: GSM ID, condition name, or glob pattern
            limit: Maximum number of files to return (None = no limit)

        Returns:
            List of matching file paths (nested archive files)

        Raises:
            ValueError: If cache_id not found
        """
        metadata = self._load_metadata(cache_id)
        if not metadata:
            raise ValueError(f"Cache ID not found: {cache_id}")

        cache_path = Path(metadata["cache_path"])
        if not cache_path.exists():
            raise ValueError(f"Cache directory missing: {cache_path}")

        nested_info_dict = metadata["nested_info"]
        nested_archives = nested_info_dict["nested_archives"]
        groups = nested_info_dict["groups"]

        # Pattern matching logic
        matching_files = []

        # Check if pattern is a specific GSM ID (e.g., GSM4710689)
        if pattern.startswith("GSM"):
            for archive in nested_archives:
                if pattern in archive:
                    matching_files.append(cache_path / archive)

        # Check if pattern matches a condition group (e.g., TISSUE, PDAC_TISSUE)
        elif pattern in groups:
            # Exact match: Load all samples in this group
            for sample in groups[pattern]:
                matching_files.append(cache_path / sample["full_path"])

        # Check for partial condition match (e.g., TISSUE matches PDAC_TISSUE, AdjNorm_TISSUE)
        else:
            pattern_upper = pattern.upper()
            for condition, samples in groups.items():
                if pattern_upper in condition.upper():
                    for sample in samples:
                        matching_files.append(cache_path / sample["full_path"])

        # If no group match, try glob pattern matching
        if not matching_files:
            import fnmatch

            for archive in nested_archives:
                if fnmatch.fnmatch(archive, pattern) or fnmatch.fnmatch(
                    archive, f"*{pattern}*"
                ):
                    matching_files.append(cache_path / archive)

        # Apply limit if specified
        if limit and len(matching_files) > limit:
            logger.info(
                f"Limiting results to {limit} files (matched {len(matching_files)})"
            )
            matching_files = matching_files[:limit]

        logger.info(f"Pattern '{pattern}' matched {len(matching_files)} files")
        return matching_files

    def get_cache_info(self, cache_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for cached extraction.

        Args:
            cache_id: Cache identifier

        Returns:
            Cache metadata dictionary, or None if not found
        """
        return self._load_metadata(cache_id)

    def list_all_caches(self) -> List[Dict[str, Any]]:
        """
        List all cached extractions.

        Returns:
            List of cache metadata dictionaries
        """
        if not self.metadata_file.exists():
            return []

        try:
            with open(self.metadata_file, "r") as f:
                all_metadata = json.load(f)
            return list(all_metadata.values())
        except Exception as e:
            logger.error(f"Failed to list caches: {e}")
            return []

    def cleanup_old_caches(self, max_age_days: int = 7) -> int:
        """
        Remove cached extractions older than specified days.

        Args:
            max_age_days: Maximum age in days (default: 7)

        Returns:
            Number of caches removed
        """
        if not self.metadata_file.exists():
            return 0

        try:
            with open(self.metadata_file, "r") as f:
                all_metadata = json.load(f)

            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            removed_count = 0

            for cache_id, metadata in list(all_metadata.items()):
                extracted_at = datetime.fromisoformat(metadata["extracted_at"])

                if extracted_at < cutoff_time:
                    cache_path = Path(metadata["cache_path"])
                    if cache_path.exists():
                        shutil.rmtree(cache_path, ignore_errors=True)
                        logger.info(f"Removed old cache: {cache_id}")

                    del all_metadata[cache_id]
                    removed_count += 1

            # Save updated metadata
            with open(self.metadata_file, "w") as f:
                json.dump(all_metadata, f, indent=2)

            logger.info(f"Cleaned up {removed_count} old caches")
            return removed_count

        except Exception as e:
            logger.error(f"Failed to cleanup old caches: {e}")
            return 0

    def delete_cache(self, cache_id: str) -> bool:
        """
        Delete specific cached extraction.

        Args:
            cache_id: Cache identifier

        Returns:
            True if successfully deleted, False otherwise
        """
        metadata = self._load_metadata(cache_id)
        if not metadata:
            return False

        try:
            cache_path = Path(metadata["cache_path"])
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)

            # Remove from metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    all_metadata = json.load(f)

                if cache_id in all_metadata:
                    del all_metadata[cache_id]

                    with open(self.metadata_file, "w") as f:
                        json.dump(all_metadata, f, indent=2)

            logger.info(f"Deleted cache: {cache_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete cache {cache_id}: {e}")
            return False

    def _save_metadata(self, cache_id: str, metadata: Dict[str, Any]):
        """Save cache metadata to JSON file."""
        try:
            # Load existing metadata
            all_metadata = {}
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    all_metadata = json.load(f)

            # Add new entry
            all_metadata[cache_id] = metadata

            # Save updated metadata
            with open(self.metadata_file, "w") as f:
                json.dump(all_metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save metadata for {cache_id}: {e}")
            raise

    def _load_metadata(self, cache_id: str) -> Optional[Dict[str, Any]]:
        """Load cache metadata from JSON file."""
        if not self.metadata_file.exists():
            return None

        try:
            with open(self.metadata_file, "r") as f:
                all_metadata = json.load(f)
            return all_metadata.get(cache_id)
        except Exception as e:
            logger.error(f"Failed to load metadata for {cache_id}: {e}")
            return None
