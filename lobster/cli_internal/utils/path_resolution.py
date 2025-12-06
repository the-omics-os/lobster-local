"""
Secure path resolution utility for CLI file operations.

BUG FIX #6: Centralized, secure path resolution to prevent:
- Directory traversal attacks (../ escaping)
- Special file access (devices, pipes, sockets)
- Inconsistent path handling across commands
- Security vulnerabilities in file operations

This module replaces duplicated path resolution logic scattered across
/read, /load, /workspace load, and /open commands with a single, audited implementation.
"""

import logging
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResolvedPath:
    """
    Result of path resolution with security validation.

    Attributes:
        path: Resolved absolute path
        source: How path was resolved ("absolute", "relative", "workspace", "home")
        exists: Whether path exists on filesystem
        is_safe: Whether path passed security checks
        error: Error message if resolution failed (None if successful)
    """

    path: Path
    source: str
    exists: bool
    is_safe: bool
    error: Optional[str] = None


class PathResolver:
    """
    Centralized, secure path resolution for CLI operations.

    BUG FIX #6: This class addresses multiple security and consistency issues:
    1. Directory traversal: Prevents ../ attacks that escape allowed directories
    2. Special files: Blocks access to devices, pipes, sockets
    3. Consistency: Single implementation for all path operations
    4. Workspace search: Automatic fallback to workspace subdirectories

    Security Model:
        - Only allows access to: current directory, home directory, workspace directory
        - Validates resolved paths stay within allowed base directories
        - Checks for special files (devices, pipes, sockets)
        - Provides detailed error messages for security violations

    Example:
        resolver = PathResolver(
            current_directory=Path.cwd(),
            workspace_path=Path(".lobster_workspace")
        )

        # Safe: within current directory
        result = resolver.resolve("data.csv")
        assert result.is_safe and result.exists

        # Blocked: directory traversal
        result = resolver.resolve("../../etc/passwd")
        assert not result.is_safe
        assert "Path traversal detected" in result.error

        # Safe: home directory access
        result = resolver.resolve("~/Documents/data.csv")
        assert result.is_safe
    """

    def __init__(
        self,
        current_directory: Path,
        workspace_path: Optional[Path] = None,
        allowed_bases: Optional[List[Path]] = None,
    ):
        """
        Initialize path resolver.

        Args:
            current_directory: Current working directory (usually from cli.py)
            workspace_path: Optional workspace directory for search fallback
            allowed_bases: Optional custom list of allowed base directories
                          (defaults to [current_dir, home, workspace])
        """
        self.current_directory = current_directory.resolve()
        self.workspace_path = workspace_path.resolve() if workspace_path else None

        # Security: Define allowed base directories
        if allowed_bases:
            self.allowed_bases = [p.resolve() for p in allowed_bases]
        else:
            self.allowed_bases = [self.current_directory, Path.home()]
            if self.workspace_path:
                self.allowed_bases.append(self.workspace_path)

    def resolve(
        self,
        path_str: str,
        search_workspace: bool = True,
        must_exist: bool = False,
        allow_special: bool = False,
    ) -> ResolvedPath:
        """
        Resolve path with security checks.

        Args:
            path_str: Path string to resolve (can be absolute, relative, or ~/...)
            search_workspace: If True and path not found, search workspace subdirectories
            must_exist: If True, return error if path doesn't exist
            allow_special: If True, allow special files (devices, pipes, sockets)

        Returns:
            ResolvedPath with validation results

        Examples:
            # Absolute path
            resolver.resolve("/tmp/data.csv")

            # Relative path
            resolver.resolve("./data/file.h5ad")

            # Home directory
            resolver.resolve("~/Documents/data.csv")

            # Workspace search
            resolver.resolve("geo_gse12345.h5ad", search_workspace=True)
        """
        path_str = path_str.strip()

        # 1. Resolve based on path type
        if path_str.startswith("/"):
            # Absolute path
            resolved = Path(path_str).resolve()
            source = "absolute"
        elif path_str.startswith("~/"):
            # Home-relative path
            resolved = (Path.home() / path_str[2:]).resolve()
            source = "home"
        else:
            # Current-directory-relative path
            resolved = (self.current_directory / path_str).resolve()
            source = "relative"

        # 2. Security check: prevent directory traversal
        is_safe = self._is_safe_path(resolved)
        if not is_safe:
            logger.warning(
                f"Path traversal detected: '{path_str}' resolves to '{resolved}' "
                f"which escapes allowed directories"
            )
            return ResolvedPath(
                path=resolved,
                source=source,
                exists=False,
                is_safe=False,
                error=f"Path traversal detected: '{path_str}' escapes allowed directories",
            )

        # 3. Check for special files (security risk)
        if not allow_special and self._is_special_file(resolved):
            logger.warning(f"Special file access blocked: '{resolved}'")
            return ResolvedPath(
                path=resolved,
                source=source,
                exists=True,
                is_safe=False,
                error=f"Special file access not allowed: {resolved.name} (device/pipe/socket)",
            )

        # 4. Check if file exists
        exists = resolved.exists()

        # 5. Workspace fallback if not found
        if not exists and search_workspace and self.workspace_path:
            workspace_result = self._search_workspace(path_str)
            if workspace_result:
                return workspace_result

        # 6. Enforce must_exist requirement
        if must_exist and not exists:
            return ResolvedPath(
                path=resolved,
                source=source,
                exists=False,
                is_safe=True,
                error=f"File not found: {path_str}",
            )

        return ResolvedPath(
            path=resolved, source=source, exists=exists, is_safe=True, error=None
        )

    def _is_safe_path(self, resolved_path: Path) -> bool:
        """
        Check if resolved path is within allowed directories.

        Security: Prevents directory traversal attacks by ensuring
        the resolved path is a child of one of the allowed base directories.

        Args:
            resolved_path: Resolved absolute path to check

        Returns:
            True if path is safe (within allowed bases), False otherwise
        """
        for base in self.allowed_bases:
            try:
                # This will raise ValueError if resolved_path is not relative to base
                resolved_path.relative_to(base)
                return True
            except ValueError:
                continue
        return False

    def _is_special_file(self, path: Path) -> bool:
        """
        Check if path is a special file (device, pipe, socket).

        Security: Special files can be dangerous to read/write:
        - Block devices (/dev/sda)
        - Character devices (/dev/urandom)
        - Named pipes (FIFOs)
        - Unix domain sockets

        Args:
            path: Path to check

        Returns:
            True if path is a special file, False if regular file/directory
        """
        if not path.exists():
            return False

        try:
            mode = path.stat().st_mode
            return any(
                [
                    stat.S_ISBLK(mode),  # Block device
                    stat.S_ISCHR(mode),  # Character device
                    stat.S_ISFIFO(mode),  # Named pipe (FIFO)
                    stat.S_ISSOCK(mode),  # Unix domain socket
                ]
            )
        except (OSError, PermissionError):
            # If we can't stat it, assume unsafe
            logger.warning(f"Could not stat path '{path}', assuming unsafe")
            return True

    def _search_workspace(self, filename: str) -> Optional[ResolvedPath]:
        """
        Search workspace subdirectories for file.

        Searches in common subdirectories: data/, plots/, exports/, cache/

        Args:
            filename: Filename to search for (not full path)

        Returns:
            ResolvedPath if found in workspace, None otherwise
        """
        if not self.workspace_path:
            return None

        search_dirs = ["data", "plots", "exports", "cache", ".lobster"]

        for subdir in search_dirs:
            candidate = self.workspace_path / subdir / filename
            if candidate.exists():
                logger.debug(f"Found '{filename}' in workspace subdirectory: {subdir}/")
                return ResolvedPath(
                    path=candidate,
                    source="workspace",
                    exists=True,
                    is_safe=True,
                    error=None,
                )

        return None
