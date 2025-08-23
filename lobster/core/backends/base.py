"""
Base backend implementation with common functionality.

This module provides the BaseBackend class that implements common
functionality shared across all data storage backends, including
path validation, logging, and error handling.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata

from lobster.core.interfaces.backend import IDataBackend

logger = logging.getLogger(__name__)


class BaseBackend(IDataBackend):
    """
    Base implementation of data backend with common functionality.
    
    This class provides shared functionality for all backend implementations
    including path validation, metadata handling, and common operations.
    Subclasses need only implement the storage-specific methods.
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the base backend.

        Args:
            base_path: Optional base path for all operations
        """
        self.base_path = Path(base_path) if base_path else None
        self.logger = logger

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """
        Resolve a path relative to base_path if set.

        Args:
            path: Path to resolve

        Returns:
            Path: Resolved absolute path
        """
        path = Path(path)
        
        if self.base_path and not path.is_absolute():
            return self.base_path / path
        
        return path.resolve()

    def _ensure_directory(self, path: Union[str, Path]) -> None:
        """
        Ensure the parent directory of a path exists.

        Args:
            path: Path whose parent directory should be created
        """
        resolved_path = self._resolve_path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

    def _calculate_checksum(self, path: Union[str, Path]) -> Optional[str]:
        """
        Calculate SHA256 checksum of a file.

        Args:
            path: Path to file

        Returns:
            Optional[str]: SHA256 checksum or None if file doesn't exist
        """
        try:
            resolved_path = self._resolve_path(path)
            if not resolved_path.exists():
                return None
            
            sha256_hash = hashlib.sha256()
            with open(resolved_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to calculate checksum for {path}: {e}")
            return None

    def _detect_format(self, path: Union[str, Path]) -> str:
        """
        Detect file format from extension.

        Args:
            path: Path to analyze

        Returns:
            str: Detected format
        """
        path = Path(path)
        extension = path.suffix.lower()
        
        format_mapping = {
            '.h5ad': 'h5ad',
            '.h5': 'h5',
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.txt': 'txt',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.h5mu': 'h5mu',
            '.zarr': 'zarr'
        }
        
        return format_mapping.get(extension, 'unknown')

    def validate_path(self, path: Union[str, Path]) -> Path:
        """
        Validate and normalize a path for this backend.

        Args:
            path: Path to validate

        Returns:
            Path: Validated and normalized path

        Raises:
            ValueError: If the path is invalid for this backend
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        try:
            resolved_path = self._resolve_path(path)
            return resolved_path
        except Exception as e:
            raise ValueError(f"Invalid path '{path}': {e}")

    def get_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get metadata about a file.

        Args:
            path: Path to the file

        Returns:
            Dict[str, Any]: Metadata dictionary

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        resolved_path = self._resolve_path(path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {resolved_path}")
        
        stat = resolved_path.stat()
        
        return {
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "checksum": self._calculate_checksum(resolved_path),
            "format": self._detect_format(resolved_path),
            "path": str(resolved_path),
            "name": resolved_path.name,
            "extension": resolved_path.suffix
        }

    def list_files(self, directory: Union[str, Path], pattern: str = "*") -> List[str]:
        """
        List files in a directory matching the given pattern.

        Args:
            directory: Directory to search
            pattern: File pattern to match (glob-style)

        Returns:
            List[str]: List of file paths matching the pattern

        Raises:
            FileNotFoundError: If the directory doesn't exist
            PermissionError: If read access is denied
        """
        resolved_dir = self._resolve_path(directory)
        
        if not resolved_dir.exists():
            raise FileNotFoundError(f"Directory not found: {resolved_dir}")
        
        if not resolved_dir.is_dir():
            raise ValueError(f"Path is not a directory: {resolved_dir}")
        
        try:
            files = list(resolved_dir.glob(pattern))
            # Return only files (not directories)
            return [str(f) for f in files if f.is_file()]
        except PermissionError as e:
            raise PermissionError(f"Permission denied accessing directory {resolved_dir}: {e}")

    def exists(self, path: Union[str, Path]) -> bool:
        """
        Check if data exists at the specified path.

        Args:
            path: Path to check

        Returns:
            bool: True if data exists, False otherwise
        """
        try:
            resolved_path = self._resolve_path(path)
            return resolved_path.exists()
        except Exception:
            return False

    def delete(self, path: Union[str, Path]) -> None:
        """
        Delete data at the specified path.

        Args:
            path: Path to delete

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If delete access is denied
        """
        resolved_path = self._resolve_path(path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {resolved_path}")
        
        try:
            resolved_path.unlink()
            self.logger.info(f"Deleted file: {resolved_path}")
        except PermissionError as e:
            raise PermissionError(f"Permission denied deleting {resolved_path}: {e}")

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the storage backend.

        Returns:
            Dict[str, Any]: Storage backend information
        """
        info = super().get_storage_info()
        info.update({
            "base_path": str(self.base_path) if self.base_path else None,
            "supports_directories": True,
            "supports_metadata": True,
            "path_style": "local"
        })
        return info

    def supports_format(self, format_name: str) -> bool:
        """
        Check if the backend supports a specific file format.

        Args:
            format_name: Format to check

        Returns:
            bool: True if format is supported, False otherwise
        """
        supported_formats = {
            'h5ad', 'h5', 'csv', 'tsv', 'txt', 'excel', 'xlsx', 'xls'
        }
        return format_name.lower() in supported_formats

    def create_backup(self, path: Union[str, Path]) -> Optional[Path]:
        """
        Create a backup of an existing file.

        Args:
            path: Path to backup

        Returns:
            Optional[Path]: Path to backup file, None if backup failed
        """
        try:
            resolved_path = self._resolve_path(path)
            if not resolved_path.exists():
                return None
            
            # Create backup with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = resolved_path.with_suffix(f".{timestamp}.backup{resolved_path.suffix}")
            
            import shutil
            shutil.copy2(resolved_path, backup_path)
            
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.warning(f"Failed to create backup for {path}: {e}")
            return None

    def _log_operation(self, operation: str, path: Union[str, Path], **kwargs) -> None:
        """
        Log a backend operation for debugging/auditing.

        Args:
            operation: Operation name
            path: Path involved in operation
            **kwargs: Additional operation details
        """
        details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"{operation}: {path} ({details})")
