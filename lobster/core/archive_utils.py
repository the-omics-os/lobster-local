"""
Shared archive handling utilities for bioinformatics data.

This module provides secure extraction, content detection, and processing
strategies for TAR/ZIP archives containing bioinformatics data formats.

Used by:
- geo_service.py: GEO dataset downloads (TAR from URL)
- client.py: Local file loading (TAR from filesystem)

Security:
- Path traversal attack prevention (CVE-2007-4559)
- Safe member validation for TAR extraction
- Automatic temporary directory cleanup
"""

import logging
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ArchiveContentType(Enum):
    """Detected content types for bioinformatics archives."""

    KALLISTO_QUANT = "kallisto_quantification"
    SALMON_QUANT = "salmon_quantification"
    TEN_X_MTX = "10x_genomics_mtx"
    GEO_RAW = "geo_raw_expression"
    GENERIC_EXPRESSION = "generic_expression_matrix"
    NESTED_ARCHIVES = "nested_archives"
    UNKNOWN = "unknown"


@dataclass
class NestedArchiveInfo:
    """
    Information about nested archives within a parent archive.

    Used for inspection-first workflow where user decides what to load
    rather than auto-loading everything.

    Attributes:
        nested_archives: List of nested archive filenames
        groups: Condition groups mapping (e.g., "PDAC_TISSUE" -> [sample_info])
        estimated_memory: Estimated memory in bytes for loading all samples
        parent_archive: Path to parent archive file
        total_count: Total number of nested archives
    """

    nested_archives: List[str] = field(default_factory=list)
    groups: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    estimated_memory: int = 0
    parent_archive: str = ""
    total_count: int = 0


class ArchiveExtractor:
    """
    Secure TAR/ZIP extraction with path traversal protection.

    Security Features:
    - Path traversal attack prevention (CVE-2007-4559)
    - Temporary directory management with cleanup
    - Support for nested archives (.tar.gz, .tar.bz2)

    Example:
        >>> extractor = ArchiveExtractor()
        >>> extract_dir = extractor.extract_safely(tar_path, target_dir)
        >>> # ... process files ...
        >>> extractor.cleanup()
    """

    def __init__(self):
        self.temp_dirs: List[Path] = []

    def _is_safe_member(self, member: tarfile.TarInfo, target_dir: Path) -> bool:
        """
        Prevent path traversal attacks during TAR extraction.

        Checks that extracted file path is within target directory.
        Protects against malicious TAR files with paths like:
        - ../../../../etc/passwd
        - /etc/shadow
        - ../../../tmp/malicious.sh

        Args:
            member: TAR member to check
            target_dir: Extraction target directory

        Returns:
            True if safe to extract, False otherwise
        """
        member_path = Path(member.name)
        try:
            target_path = (target_dir / member_path).resolve()
            common_path = Path(os.path.commonpath([target_dir.resolve(), target_path]))
            is_safe = common_path == target_dir.resolve()

            if not is_safe:
                logger.warning(f"Blocked unsafe TAR member: {member.name}")

            return is_safe
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Path safety check failed for {member.name}: {e}")
            return False

    def extract_safely(
        self, archive_path: Path, target_dir: Path, cleanup_on_error: bool = True
    ) -> Path:
        """
        Extract TAR/ZIP archive with security checks.

        Args:
            archive_path: Path to archive file
            target_dir: Directory for extraction
            cleanup_on_error: Remove target_dir if extraction fails

        Returns:
            Path to extracted directory

        Raises:
            ValueError: Unsupported archive format
            RuntimeError: Extraction failed
        """
        try:
            target_dir.mkdir(parents=True, exist_ok=True)

            # Handle TAR formats (.tar, .tar.gz, .tar.bz2, .tgz)
            if (
                archive_path.suffix in [".tar", ".gz", ".bz2", ".tgz"]
                or ".tar" in archive_path.name
            ):
                with tarfile.open(archive_path, "r:*") as tar:
                    # Filter safe members
                    safe_members = [
                        m
                        for m in tar.getmembers()
                        if self._is_safe_member(m, target_dir)
                    ]

                    if not safe_members:
                        raise RuntimeError("No safe members found in TAR archive")

                    logger.debug(
                        f"Extracting {len(safe_members)} safe members from TAR"
                    )
                    tar.extractall(path=target_dir, members=safe_members)

            # Handle ZIP format
            elif archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    # ZIP paths are safer but still validate
                    members = zip_ref.namelist()
                    logger.debug(f"Extracting {len(members)} files from ZIP")
                    zip_ref.extractall(target_dir)

            else:
                raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

            logger.info(f"Successfully extracted archive to {target_dir}")
            return target_dir

        except Exception as e:
            logger.error(f"Archive extraction failed: {e}")
            if cleanup_on_error and target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to extract {archive_path.name}: {e}")

    def extract_to_temp(
        self, archive_path: Path, prefix: str = "lobster_archive_"
    ) -> Path:
        """
        Extract archive to temporary directory with automatic cleanup tracking.

        Args:
            archive_path: Path to archive
            prefix: Prefix for temp directory name

        Returns:
            Path to temporary extraction directory
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_dirs.append(temp_dir)

        try:
            return self.extract_safely(archive_path, temp_dir)
        except Exception:
            # Remove from tracking if extraction failed
            self.temp_dirs.remove(temp_dir)
            raise

    def cleanup(self):
        """Remove all tracked temporary directories."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                logger.debug(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
        self.temp_dirs.clear()


class ContentDetector:
    """
    Detect bioinformatics data formats within archives.

    Supports:
    - Kallisto/Salmon quantification files
    - 10X Genomics MEX format
    - GEO RAW expression files
    - Generic expression matrices

    Example:
        >>> has_quant, tool, files, n_samples = ContentDetector.detect_kallisto_salmon(file_list)
        >>> content_type = ContentDetector.detect_content_type(extract_dir)
    """

    # Quantification tool signatures
    KALLISTO_PATTERNS = ["abundance.tsv", "abundance.h5", "abundance.txt"]
    SALMON_PATTERNS = ["quant.sf", "quant.genes.sf"]

    # 10X Genomics signatures (support both V2 and V3)
    # Note: Using stems (no extensions) since detection uses Path(f).stem
    TEN_X_V3_FILES = {"matrix", "features", "barcodes"}  # V3 chemistry
    TEN_X_V2_FILES = {"matrix", "genes", "barcodes"}  # V2 chemistry

    # GEO RAW pattern: GSM<digits>_*.txt or GSM<digits>_*.txt.gz
    GEO_RAW_PATTERN = re.compile(r"^GSM\d+_.*\.(txt|txt\.gz|cel|CEL)$")

    @classmethod
    def detect_kallisto_salmon(
        cls, file_paths: List[str]
    ) -> Tuple[bool, str, List[str], int]:
        """
        Detect Kallisto/Salmon quantification files.

        Args:
            file_paths: List of file paths/URLs to check

        Returns:
            Tuple of (has_quant_files, tool_type, matched_files, sample_count):
            - has_quant_files: True if quantification files detected
            - tool_type: "kallisto", "salmon", or "mixed"
            - matched_files: List of matching file paths
            - sample_count: Estimated number of samples
        """
        kallisto_files = []
        salmon_files = []
        abundance_files = []

        for file_path in file_paths:
            filename = os.path.basename(file_path).lower()

            # Check Kallisto
            if any(pattern in filename for pattern in cls.KALLISTO_PATTERNS):
                kallisto_files.append(file_path)
                if "abundance.tsv" in filename or "abundance.h5" in filename:
                    abundance_files.append(filename)

            # Check Salmon
            if any(pattern in filename for pattern in cls.SALMON_PATTERNS):
                salmon_files.append(file_path)
                if "quant.sf" in filename:
                    abundance_files.append(filename)

        # Determine tool type
        has_quant = len(kallisto_files) > 0 or len(salmon_files) > 0

        if not has_quant:
            return False, "", [], 0

        if len(kallisto_files) > 0 and len(salmon_files) > 0:
            tool_type = "mixed"
            matched = kallisto_files + salmon_files
        elif len(kallisto_files) > 0:
            tool_type = "kallisto"
            matched = kallisto_files
        else:
            tool_type = "salmon"
            matched = salmon_files

        estimated_samples = len(abundance_files)
        return has_quant, tool_type, matched, estimated_samples

    @classmethod
    def detect_content_type(cls, extract_dir: Path) -> ArchiveContentType:
        """
        Detect bioinformatics format from extracted directory.

        Args:
            extract_dir: Path to extracted archive directory

        Returns:
            Detected ArchiveContentType
        """
        all_files = [
            str(f.relative_to(extract_dir))
            for f in extract_dir.rglob("*")
            if f.is_file()
        ]

        # Check for Kallisto/Salmon quantification (requires subdirectories)
        subdirs = [d for d in extract_dir.iterdir() if d.is_dir()]
        if len(subdirs) >= 2:
            kallisto_count = sum(
                1
                for d in subdirs
                if (d / "abundance.tsv").exists() or (d / "abundance.h5").exists()
            )
            salmon_count = sum(1 for d in subdirs if (d / "quant.sf").exists())

            if kallisto_count >= 2:
                return ArchiveContentType.KALLISTO_QUANT
            if salmon_count >= 2:
                return ArchiveContentType.SALMON_QUANT

        # Check for 10X Genomics format (V2 or V3)
        # Handle both compressed (.gz) and uncompressed files
        basenames = {
            Path(f).stem.replace(".mtx", "").replace(".tsv", "") for f in all_files
        }
        basenames_with_ext = {Path(f).stem for f in all_files}

        # Check if we have required files (stem without compression)
        has_v3 = cls.TEN_X_V3_FILES.issubset(basenames) or cls.TEN_X_V3_FILES.issubset(
            basenames_with_ext
        )
        has_v2 = cls.TEN_X_V2_FILES.issubset(basenames) or cls.TEN_X_V2_FILES.issubset(
            basenames_with_ext
        )

        if has_v3 or has_v2:
            return ArchiveContentType.TEN_X_MTX

        # Check for GEO RAW pattern
        geo_matches = [f for f in all_files if cls.GEO_RAW_PATTERN.match(Path(f).name)]
        if len(geo_matches) >= 2:
            return ArchiveContentType.GEO_RAW

        # Check for generic expression files
        expression_extensions = [".csv", ".tsv", ".txt", ".h5", ".h5ad"]
        expression_files = [
            f
            for f in all_files
            if any(Path(f).suffix == ext for ext in expression_extensions)
        ]

        # Filter by file size heuristic (>100KB likely expression data)
        large_files = [
            f for f in expression_files if (extract_dir / f).stat().st_size > 100000
        ]
        if large_files:
            return ArchiveContentType.GENERIC_EXPRESSION

        return ArchiveContentType.UNKNOWN


class ArchiveInspector:
    """
    Inspect archive contents without full extraction.
    Fast metadata gathering for smart processing decisions.

    Example:
        >>> inspector = ArchiveInspector()
        >>> manifest = inspector.inspect_manifest(tar_path)
        >>> content_type = inspector.detect_content_type_from_manifest(manifest)
    """

    def inspect_manifest(self, archive_path: Path) -> Dict[str, Any]:
        """
        Fast inspection of archive without extracting.

        Args:
            archive_path: Path to archive file

        Returns:
            Manifest with file count, extensions, sizes, structure
        """
        try:
            if ".tar" in archive_path.name:
                with tarfile.open(archive_path, "r:*") as tar:
                    members = tar.getmembers()
                    files = [m for m in members if m.isfile()]

                    return {
                        "file_count": len(files),
                        "total_size": sum(m.size for m in members),
                        "extensions": Counter([Path(m.name).suffix for m in files]),
                        "filenames": [m.name for m in files],
                        "has_subdirectories": any(m.isdir() for m in members),
                    }

            elif archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    members = zip_ref.namelist()

                    return {
                        "file_count": len(members),
                        "total_size": sum(
                            zip_ref.getinfo(m).file_size for m in members
                        ),
                        "extensions": Counter([Path(m).suffix for m in members]),
                        "filenames": members,
                        "has_subdirectories": any("/" in m for m in members),
                    }

            else:
                raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

        except Exception as e:
            logger.error(f"Failed to inspect archive: {e}")
            return {
                "file_count": 0,
                "total_size": 0,
                "extensions": Counter(),
                "filenames": [],
                "has_subdirectories": False,
                "error": str(e),
            }

    def detect_content_type_from_manifest(
        self, manifest: Dict[str, Any]
    ) -> ArchiveContentType:
        """
        Detect content type from manifest (without extraction).

        Args:
            manifest: Archive manifest from inspect_manifest()

        Returns:
            Detected ArchiveContentType
        """
        filenames = manifest.get("filenames", [])
        has_subdirs = manifest.get("has_subdirectories", False)

        # Check for quantification files
        if has_subdirs:
            kallisto_count = sum(
                1 for f in filenames if "abundance.tsv" in f or "abundance.h5" in f
            )
            salmon_count = sum(1 for f in filenames if "quant.sf" in f)

            if kallisto_count >= 2:
                return ArchiveContentType.KALLISTO_QUANT
            if salmon_count >= 2:
                return ArchiveContentType.SALMON_QUANT

        # Check for 10X format (V2 or V3)
        # Handle both compressed (.gz) and uncompressed files
        basenames = {
            Path(f).stem.replace(".mtx", "").replace(".tsv", "") for f in filenames
        }
        basenames_with_ext = {Path(f).stem for f in filenames}

        # Check if we have required files
        has_v3 = ContentDetector.TEN_X_V3_FILES.issubset(
            basenames
        ) or ContentDetector.TEN_X_V3_FILES.issubset(basenames_with_ext)
        has_v2 = ContentDetector.TEN_X_V2_FILES.issubset(
            basenames
        ) or ContentDetector.TEN_X_V2_FILES.issubset(basenames_with_ext)

        if has_v3 or has_v2:
            return ArchiveContentType.TEN_X_MTX

        # Check for GEO RAW
        geo_matches = [
            f for f in filenames if ContentDetector.GEO_RAW_PATTERN.match(Path(f).name)
        ]
        if len(geo_matches) >= 2:
            return ArchiveContentType.GEO_RAW

        # Generic expression
        extensions = manifest.get("extensions", Counter())
        if (
            extensions.get(".csv", 0)
            + extensions.get(".tsv", 0)
            + extensions.get(".txt", 0)
            >= 1
        ):
            return ArchiveContentType.GENERIC_EXPRESSION

        return ArchiveContentType.UNKNOWN

    def detect_nested_archives(
        self, manifest: Dict[str, Any], parent_archive_path: str = ""
    ) -> Optional[NestedArchiveInfo]:
        """
        Detect nested .tar.gz archives within parent archive (e.g., 10X samples in GEO RAW.tar).

        This enables inspection-first workflow for complex nested structures where
        auto-loading everything would be memory-intensive or time-consuming.

        Args:
            manifest: Archive manifest from inspect_manifest()
            parent_archive_path: Path to parent archive (for metadata)

        Returns:
            NestedArchiveInfo if nested archives detected (>= 2 nested archives),
            None otherwise
        """
        filenames = manifest.get("filenames", [])

        # Check for nested archive pattern (.tar.gz, .tgz, .tar.bz2)
        nested_archives = [
            f
            for f in filenames
            if f.endswith(".tar.gz") or f.endswith(".tgz") or f.endswith(".tar.bz2")
        ]

        if len(nested_archives) < 2:
            return None  # Not a nested archive structure worth special handling

        logger.info(
            f"Detected {len(nested_archives)} nested archives in {parent_archive_path}"
        )

        info = NestedArchiveInfo()
        info.nested_archives = nested_archives
        info.total_count = len(nested_archives)
        info.parent_archive = parent_archive_path

        # Pattern analysis for grouping (GSM*_<CONDITION>_<NUMBER>.tar.gz)
        # Example: GSM4710689_PDAC_TISSUE_1.tar.gz
        pattern = re.compile(r"^(GSM\d+)_([A-Za-z0-9_]+)_(\d+[A-Z]?)\.tar\.gz$")

        for archive in nested_archives:
            basename = Path(archive).name
            match = pattern.match(basename)

            if match:
                gsm_id, condition, number = match.groups()

                if condition not in info.groups:
                    info.groups[condition] = []

                info.groups[condition].append(
                    {
                        "gsm_id": gsm_id,
                        "filename": basename,
                        "condition": condition,
                        "number": number,
                        "full_path": archive,
                    }
                )

        # Estimate memory (average 300MB per 10X sample, conservative estimate)
        info.estimated_memory = len(nested_archives) * 300 * 1024 * 1024

        # Sort groups by condition name for consistent display
        info.groups = dict(sorted(info.groups.items()))

        return info
