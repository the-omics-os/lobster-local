"""
Comprehensive tests for the BaseBackend class.

This module tests all functionality of the BaseBackend class including
path resolution, metadata handling, file operations, and error cases.
"""

import pytest
import tempfile
import shutil
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import anndata
import numpy as np
import pandas as pd

from lobster.core.backends.base import BaseBackend
from lobster.core.interfaces.backend import IDataBackend


class ConcreteBaseBackend(BaseBackend):
    """
    Concrete implementation of BaseBackend for testing.
    Since BaseBackend is abstract, we need a concrete version for testing.
    """

    def load(self, path, **kwargs):
        """Mock implementation for testing."""
        return Mock(spec=anndata.AnnData)

    def save(self, adata, path, **kwargs):
        """Mock implementation for testing."""
        pass


class TestBaseBackendInitialization:
    """Test BaseBackend initialization and configuration."""

    def test_init_without_base_path(self):
        """Test initialization without base path."""
        backend = ConcreteBaseBackend()
        assert backend.base_path is None
        assert backend.logger is not None

    def test_init_with_string_base_path(self):
        """Test initialization with string base path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = TestableBaseBackend(base_path=temp_dir)
            assert backend.base_path == Path(temp_dir)

    def test_init_with_path_base_path(self):
        """Test initialization with Path base path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_obj = Path(temp_dir)
            backend = TestableBaseBackend(base_path=path_obj)
            assert backend.base_path == path_obj

    def test_interface_compliance(self):
        """Test that BaseBackend implements IDataBackend interface."""
        backend = ConcreteBaseBackend()
        assert isinstance(backend, IDataBackend)


class TestPathResolution:
    """Test path resolution functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = ConcreteBaseBackend()
        self.backend_with_base = ConcreteBaseBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resolve_absolute_path_without_base(self):
        """Test resolving absolute path without base path."""
        abs_path = Path(self.temp_dir) / "test.h5ad"
        result = self.backend._resolve_path(abs_path)
        assert result == abs_path.resolve()

    def test_resolve_relative_path_without_base(self):
        """Test resolving relative path without base path."""
        rel_path = "test.h5ad"
        result = self.backend._resolve_path(rel_path)
        assert result == Path(rel_path).resolve()

    def test_resolve_absolute_path_with_base(self):
        """Test resolving absolute path with base path (should ignore base)."""
        abs_path = Path(self.temp_dir) / "test.h5ad"
        result = self.backend_with_base._resolve_path(abs_path)
        assert result == abs_path.resolve()

    def test_resolve_relative_path_with_base(self):
        """Test resolving relative path with base path."""
        rel_path = "test.h5ad"
        result = self.backend_with_base._resolve_path(rel_path)
        expected = (Path(self.temp_dir) / rel_path).resolve()
        assert result == expected

    def test_resolve_string_path(self):
        """Test resolving string path."""
        path_str = "test.h5ad"
        result = self.backend_with_base._resolve_path(path_str)
        expected = (Path(self.temp_dir) / path_str).resolve()
        assert result == expected


class TestDirectoryOperations:
    """Test directory creation and management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = TestableBaseBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ensure_directory_creates_parent(self):
        """Test that _ensure_directory creates parent directories."""
        nested_path = Path(self.temp_dir) / "subdir" / "test.h5ad"
        assert not nested_path.parent.exists()

        self.backend._ensure_directory(nested_path)
        assert nested_path.parent.exists()
        assert nested_path.parent.is_dir()

    def test_ensure_directory_existing_parent(self):
        """Test _ensure_directory with existing parent directory."""
        existing_path = Path(self.temp_dir) / "test.h5ad"
        assert existing_path.parent.exists()

        # Should not raise error
        self.backend._ensure_directory(existing_path)
        assert existing_path.parent.exists()

    def test_ensure_directory_nested_creation(self):
        """Test creating multiple nested directories."""
        deep_path = Path(self.temp_dir) / "a" / "b" / "c" / "d" / "test.h5ad"
        self.backend._ensure_directory(deep_path)
        assert deep_path.parent.exists()
        assert deep_path.parent.is_dir()


class TestChecksumCalculation:
    """Test checksum calculation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = TestableBaseBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_calculate_checksum_existing_file(self):
        """Test checksum calculation for existing file."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = b"Hello, world!"
        test_file.write_bytes(test_content)

        # Calculate expected checksum
        expected_checksum = hashlib.sha256(test_content).hexdigest()

        result = self.backend._calculate_checksum(test_file)
        assert result == expected_checksum

    def test_calculate_checksum_nonexistent_file(self):
        """Test checksum calculation for non-existent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.txt"
        result = self.backend._calculate_checksum(nonexistent_file)
        assert result is None

    def test_calculate_checksum_large_file(self):
        """Test checksum calculation for large file (tests chunking)."""
        test_file = Path(self.temp_dir) / "large_test.txt"

        # Create file larger than chunk size (4096 bytes)
        test_content = b"A" * 10000
        test_file.write_bytes(test_content)

        expected_checksum = hashlib.sha256(test_content).hexdigest()
        result = self.backend._calculate_checksum(test_file)
        assert result == expected_checksum

    def test_calculate_checksum_empty_file(self):
        """Test checksum calculation for empty file."""
        test_file = Path(self.temp_dir) / "empty.txt"
        test_file.touch()

        expected_checksum = hashlib.sha256(b"").hexdigest()
        result = self.backend._calculate_checksum(test_file)
        assert result == expected_checksum

    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    def test_calculate_checksum_permission_error(self, mock_open):
        """Test checksum calculation with permission error."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.touch()

        result = self.backend._calculate_checksum(test_file)
        assert result is None


class TestFormatDetection:
    """Test file format detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = ConcreteBaseBackend()

    def test_detect_h5ad_format(self):
        """Test detection of H5AD format."""
        result = self.backend._detect_format("test.h5ad")
        assert result == "h5ad"

    def test_detect_h5_format(self):
        """Test detection of H5 format."""
        result = self.backend._detect_format("test.h5")
        assert result == "h5"

    def test_detect_csv_format(self):
        """Test detection of CSV format."""
        result = self.backend._detect_format("test.csv")
        assert result == "csv"

    def test_detect_excel_formats(self):
        """Test detection of Excel formats."""
        assert self.backend._detect_format("test.xlsx") == "excel"
        assert self.backend._detect_format("test.xls") == "excel"

    def test_detect_h5mu_format(self):
        """Test detection of H5MU format."""
        result = self.backend._detect_format("test.h5mu")
        assert result == "h5mu"

    def test_detect_unknown_format(self):
        """Test detection of unknown format."""
        result = self.backend._detect_format("test.unknown")
        assert result == "unknown"

    def test_detect_format_case_insensitive(self):
        """Test format detection is case insensitive."""
        assert self.backend._detect_format("test.H5AD") == "h5ad"
        assert self.backend._detect_format("TEST.CSV") == "csv"

    def test_detect_format_with_path_object(self):
        """Test format detection with Path object."""
        path_obj = Path("test.h5ad")
        result = self.backend._detect_format(path_obj)
        assert result == "h5ad"


class TestPathValidation:
    """Test path validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = TestableBaseBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_valid_path(self):
        """Test validation of valid path."""
        valid_path = "test.h5ad"
        result = self.backend.validate_path(valid_path)
        expected = (Path(self.temp_dir) / valid_path).resolve()
        assert result == expected

    def test_validate_empty_path(self):
        """Test validation of empty path."""
        with pytest.raises(ValueError, match="Path cannot be empty"):
            self.backend.validate_path("")

    def test_validate_none_path(self):
        """Test validation of None path."""
        with pytest.raises(ValueError, match="Path cannot be empty"):
            self.backend.validate_path(None)

    def test_validate_absolute_path(self):
        """Test validation of absolute path."""
        abs_path = Path(self.temp_dir) / "test.h5ad"
        result = self.backend.validate_path(abs_path)
        assert result == abs_path

    @patch.object(ConcreteBaseBackend, '_resolve_path', side_effect=Exception("Path error"))
    def test_validate_path_with_resolution_error(self, mock_resolve):
        """Test path validation when resolution fails."""
        with pytest.raises(ValueError, match="Invalid path"):
            self.backend.validate_path("invalid_path")


class TestMetadataOperations:
    """Test metadata retrieval functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = TestableBaseBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_metadata_existing_file(self):
        """Test getting metadata for existing file."""
        test_file = Path(self.temp_dir) / "test.h5ad"
        test_content = b"test content"
        test_file.write_bytes(test_content)

        metadata = self.backend.get_metadata(test_file)

        assert metadata["size"] == len(test_content)
        assert metadata["format"] == "h5ad"
        assert metadata["name"] == "test.h5ad"
        assert metadata["extension"] == ".h5ad"
        assert "modified" in metadata
        assert "checksum" in metadata
        assert "path" in metadata

    def test_get_metadata_nonexistent_file(self):
        """Test getting metadata for non-existent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.h5ad"

        with pytest.raises(FileNotFoundError):
            self.backend.get_metadata(nonexistent_file)

    def test_get_metadata_different_formats(self):
        """Test metadata for different file formats."""
        formats = [
            ("test.csv", "csv"),
            ("test.h5", "h5"),
            ("test.xlsx", "excel"),
            ("test.unknown", "unknown")
        ]

        for filename, expected_format in formats:
            test_file = Path(self.temp_dir) / filename
            test_file.write_bytes(b"content")

            metadata = self.backend.get_metadata(test_file)
            assert metadata["format"] == expected_format


class TestFileOperations:
    """Test file listing and management operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = TestableBaseBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_files_existing_directory(self):
        """Test listing files in existing directory."""
        # Create test files
        test_files = ["file1.h5ad", "file2.csv", "file3.txt"]
        for filename in test_files:
            (Path(self.temp_dir) / filename).touch()

        result = self.backend.list_files(self.temp_dir)

        # Convert to relative paths for comparison
        result_names = [Path(f).name for f in result]
        assert set(result_names) == set(test_files)

    def test_list_files_with_pattern(self):
        """Test listing files with glob pattern."""
        # Create test files
        files = ["data1.h5ad", "data2.h5ad", "config.txt", "results.csv"]
        for filename in files:
            (Path(self.temp_dir) / filename).touch()

        # List only .h5ad files
        result = self.backend.list_files(self.temp_dir, "*.h5ad")
        result_names = [Path(f).name for f in result]

        assert "data1.h5ad" in result_names
        assert "data2.h5ad" in result_names
        assert "config.txt" not in result_names
        assert "results.csv" not in result_names

    def test_list_files_nonexistent_directory(self):
        """Test listing files in non-existent directory."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"

        with pytest.raises(FileNotFoundError):
            self.backend.list_files(nonexistent_dir)

    def test_list_files_not_a_directory(self):
        """Test listing files on a file (not directory)."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.touch()

        with pytest.raises(ValueError, match="Path is not a directory"):
            self.backend.list_files(test_file)

    def test_list_files_excludes_directories(self):
        """Test that list_files excludes subdirectories."""
        # Create files and directories
        (Path(self.temp_dir) / "file.txt").touch()
        (Path(self.temp_dir) / "subdir").mkdir()
        (Path(self.temp_dir) / "subdir" / "nested.txt").touch()

        result = self.backend.list_files(self.temp_dir)
        result_names = [Path(f).name for f in result]

        assert "file.txt" in result_names
        assert "subdir" not in result_names  # Directories should be excluded
        assert len(result_names) == 1

    @patch('pathlib.Path.glob', side_effect=PermissionError("Access denied"))
    def test_list_files_permission_error(self, mock_glob):
        """Test listing files with permission error."""
        with pytest.raises(PermissionError, match="Permission denied"):
            self.backend.list_files(self.temp_dir)


class TestExistenceChecking:
    """Test file existence checking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = TestableBaseBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_exists_existing_file(self):
        """Test existence check for existing file."""
        test_file = Path(self.temp_dir) / "test.h5ad"
        test_file.touch()

        assert self.backend.exists(test_file) is True

    def test_exists_nonexistent_file(self):
        """Test existence check for non-existent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.h5ad"

        assert self.backend.exists(nonexistent_file) is False

    def test_exists_relative_path(self):
        """Test existence check with relative path."""
        test_file = Path(self.temp_dir) / "test.h5ad"
        test_file.touch()

        # Use relative path
        assert self.backend.exists("test.h5ad") is True

    @patch.object(ConcreteBaseBackend, '_resolve_path', side_effect=Exception("Error"))
    def test_exists_with_path_resolution_error(self, mock_resolve):
        """Test exists method handles path resolution errors gracefully."""
        # Should return False instead of raising exception
        assert self.backend.exists("any_path") is False


class TestFileDeletion:
    """Test file deletion functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = TestableBaseBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_delete_existing_file(self):
        """Test deleting existing file."""
        test_file = Path(self.temp_dir) / "test.h5ad"
        test_file.touch()
        assert test_file.exists()

        self.backend.delete(test_file)
        assert not test_file.exists()

    def test_delete_nonexistent_file(self):
        """Test deleting non-existent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.h5ad"

        with pytest.raises(FileNotFoundError):
            self.backend.delete(nonexistent_file)

    def test_delete_relative_path(self):
        """Test deleting file with relative path."""
        test_file = Path(self.temp_dir) / "test.h5ad"
        test_file.touch()

        self.backend.delete("test.h5ad")
        assert not test_file.exists()

    @patch('pathlib.Path.unlink', side_effect=PermissionError("Access denied"))
    def test_delete_permission_error(self, mock_unlink):
        """Test deletion with permission error."""
        test_file = Path(self.temp_dir) / "test.h5ad"
        test_file.touch()

        with pytest.raises(PermissionError, match="Permission denied"):
            self.backend.delete(test_file)


class TestStorageInfo:
    """Test storage information retrieval."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = TestableBaseBackend(base_path=self.temp_dir)
        self.backend_no_base = ConcreteBaseBackend()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_storage_info_with_base_path(self):
        """Test getting storage info with base path."""
        info = self.backend.get_storage_info()

        assert info["base_path"] == str(self.temp_dir)
        assert info["supports_directories"] is True
        assert info["supports_metadata"] is True
        assert info["path_style"] == "local"
        assert info["backend_type"] == "ConcreteBaseBackend"

    def test_get_storage_info_without_base_path(self):
        """Test getting storage info without base path."""
        info = self.backend_no_base.get_storage_info()

        assert info["base_path"] is None
        assert info["supports_directories"] is True
        assert info["supports_metadata"] is True
        assert info["path_style"] == "local"


class TestFormatSupport:
    """Test format support checking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = ConcreteBaseBackend()

    def test_supports_common_formats(self):
        """Test support for common formats."""
        supported_formats = ['h5ad', 'h5', 'csv', 'tsv', 'txt', 'excel', 'xlsx', 'xls']

        for fmt in supported_formats:
            assert self.backend.supports_format(fmt) is True

    def test_does_not_support_unknown_format(self):
        """Test lack of support for unknown formats."""
        unsupported_formats = ['pdf', 'doc', 'mp3', 'jpg']

        for fmt in unsupported_formats:
            assert self.backend.supports_format(fmt) is False

    def test_format_support_case_insensitive(self):
        """Test format support is case insensitive."""
        assert self.backend.supports_format('H5AD') is True
        assert self.backend.supports_format('CSV') is True
        assert self.backend.supports_format('Excel') is True


class TestBackupCreation:
    """Test backup creation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = TestableBaseBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_backup_existing_file(self):
        """Test creating backup of existing file."""
        test_file = Path(self.temp_dir) / "test.h5ad"
        test_content = b"test content"
        test_file.write_bytes(test_content)

        backup_path = self.backend.create_backup(test_file)

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.read_bytes() == test_content
        assert ".backup" in str(backup_path)
        assert test_file.suffix in str(backup_path)

    def test_create_backup_nonexistent_file(self):
        """Test creating backup of non-existent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.h5ad"

        backup_path = self.backend.create_backup(nonexistent_file)
        assert backup_path is None

    @patch('shutil.copy2', side_effect=Exception("Copy failed"))
    def test_create_backup_copy_failure(self, mock_copy):
        """Test backup creation when copy fails."""
        test_file = Path(self.temp_dir) / "test.h5ad"
        test_file.touch()

        backup_path = self.backend.create_backup(test_file)
        assert backup_path is None


class TestLoggingOperations:
    """Test logging functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = ConcreteBaseBackend()

    @patch.object(ConcreteBaseBackend, 'logger')
    def test_log_operation_basic(self, mock_logger):
        """Test basic operation logging."""
        self.backend._log_operation("test_op", "/path/to/file")
        mock_logger.debug.assert_called_once()

        call_args = mock_logger.debug.call_args[0][0]
        assert "test_op" in call_args
        assert "/path/to/file" in call_args

    @patch.object(ConcreteBaseBackend, 'logger')
    def test_log_operation_with_kwargs(self, mock_logger):
        """Test operation logging with additional parameters."""
        self.backend._log_operation(
            "save",
            "/path/to/file.h5ad",
            size_mb=10.5,
            compression="gzip"
        )

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "save" in call_args
        assert "size_mb=10.5" in call_args
        assert "compression=gzip" in call_args


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness of BaseBackend."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = TestableBaseBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_handle_unicode_paths(self):
        """Test handling of Unicode characters in paths."""
        unicode_path = "tëst_ünïcödë.h5ad"
        test_file = Path(self.temp_dir) / unicode_path
        test_file.touch()

        # Should handle Unicode paths without errors
        assert self.backend.exists(unicode_path) is True
        metadata = self.backend.get_metadata(unicode_path)
        assert metadata["name"] == unicode_path

    def test_handle_long_paths(self):
        """Test handling of very long paths."""
        # Create a reasonably long path (but not exceeding system limits)
        long_name = "a" * 100 + ".h5ad"
        test_file = Path(self.temp_dir) / long_name
        test_file.touch()

        assert self.backend.exists(long_name) is True
        metadata = self.backend.get_metadata(long_name)
        assert metadata["name"] == long_name

    def test_handle_spaces_in_paths(self):
        """Test handling of spaces in file paths."""
        spaced_path = "file with spaces.h5ad"
        test_file = Path(self.temp_dir) / spaced_path
        test_file.touch()

        assert self.backend.exists(spaced_path) is True
        metadata = self.backend.get_metadata(spaced_path)
        assert metadata["name"] == spaced_path

    def test_handle_special_characters_in_paths(self):
        """Test handling of special characters in paths."""
        special_chars = ["file-with-hyphens.h5ad", "file_with_underscores.h5ad", "file.with.dots.h5ad"]

        for filename in special_chars:
            test_file = Path(self.temp_dir) / filename
            test_file.touch()

            assert self.backend.exists(filename) is True
            metadata = self.backend.get_metadata(filename)
            assert metadata["name"] == filename

    def test_concurrent_access_simulation(self):
        """Test behavior under simulated concurrent access."""
        test_file = Path(self.temp_dir) / "concurrent_test.h5ad"
        test_file.write_bytes(b"test content")

        # Simulate multiple rapid accesses
        for i in range(10):
            assert self.backend.exists("concurrent_test.h5ad") is True
            metadata = self.backend.get_metadata("concurrent_test.h5ad")
            assert metadata["size"] == 12  # len(b"test content")

    def test_memory_efficiency_large_checksum(self):
        """Test memory efficiency when calculating checksums of large files."""
        import psutil
        import os

        # Create a moderately large file (1MB)
        large_file = Path(self.temp_dir) / "large_file.h5ad"
        large_content = b"A" * (1024 * 1024)  # 1MB
        large_file.write_bytes(large_content)

        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Calculate checksum
        checksum = self.backend._calculate_checksum(large_file)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for a 1MB file)
        assert memory_increase < 100 * 1024 * 1024
        assert checksum is not None
        assert len(checksum) == 64  # SHA256 hex length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])