"""
Data backend interface definitions.

This module defines the abstract interface for data storage backends,
enabling support for different storage systems (local files, S3, etc.)
with a consistent API.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import anndata


class IDataBackend(ABC):
    """
    Abstract interface for data storage backends.
    
    This interface defines the contract for storing and retrieving
    bioinformatics data in various formats and storage systems.
    All backends must implement these core operations to ensure
    consistent behavior across different storage solutions.
    """

    @abstractmethod
    def load(self, path: Union[str, Path], **kwargs) -> anndata.AnnData:
        """
        Load data from storage.

        Args:
            path: Path to the data file (local path or URI)
            **kwargs: Backend-specific loading parameters

        Returns:
            anndata.AnnData: Loaded data object

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is unsupported or corrupted
            PermissionError: If access is denied
        """
        pass

    @abstractmethod
    def save(self, adata: anndata.AnnData, path: Union[str, Path], **kwargs) -> None:
        """
        Save data to storage.

        Args:
            adata: AnnData object to save
            path: Destination path (local path or URI)
            **kwargs: Backend-specific saving parameters

        Raises:
            ValueError: If the data cannot be serialized
            PermissionError: If write access is denied
            OSError: If storage operation fails
        """
        pass

    @abstractmethod
    def exists(self, path: Union[str, Path]) -> bool:
        """
        Check if data exists at the specified path.

        Args:
            path: Path to check (local path or URI)

        Returns:
            bool: True if data exists, False otherwise
        """
        pass

    @abstractmethod
    def delete(self, path: Union[str, Path]) -> None:
        """
        Delete data at the specified path.

        Args:
            path: Path to delete (local path or URI)

        Raises:
            FileNotFoundError: If the file doesn't exist
            PermissionError: If delete access is denied
        """
        pass

    @abstractmethod
    def list_files(self, directory: Union[str, Path], pattern: str = "*") -> list[str]:
        """
        List files in a directory matching the given pattern.

        Args:
            directory: Directory to search (local path or URI)
            pattern: File pattern to match (glob-style)

        Returns:
            list[str]: List of file paths matching the pattern

        Raises:
            FileNotFoundError: If the directory doesn't exist
            PermissionError: If read access is denied
        """
        pass

    @abstractmethod
    def get_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get metadata about a file.

        Args:
            path: Path to the file (local path or URI)

        Returns:
            Dict[str, Any]: Metadata dictionary containing:
                - size: File size in bytes
                - modified: Last modification timestamp
                - checksum: File checksum (if available)
                - format: Detected file format

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        pass

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the storage backend.

        Returns:
            Dict[str, Any]: Storage backend information including:
                - backend_type: Type of backend (e.g., 'local', 's3')
                - capabilities: List of supported operations
                - configuration: Backend configuration details
        """
        return {
            "backend_type": self.__class__.__name__,
            "capabilities": ["load", "save", "exists", "delete", "list_files", "get_metadata"],
            "configuration": {}
        }

    def validate_path(self, path: Union[str, Path]) -> Union[str, Path]:
        """
        Validate and normalize a path for this backend.

        Args:
            path: Path to validate

        Returns:
            Union[str, Path]: Validated and normalized path

        Raises:
            ValueError: If the path is invalid for this backend
        """
        return path

    def supports_format(self, format_name: str) -> bool:
        """
        Check if the backend supports a specific file format.

        Args:
            format_name: Format to check (e.g., 'h5ad', 'csv', 'h5mu')

        Returns:
            bool: True if format is supported, False otherwise
        """
        # Default implementation - subclasses should override
        return format_name.lower() in ['h5ad', 'csv']
