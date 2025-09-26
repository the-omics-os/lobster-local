"""
Interface compliance tests for all backend implementations.

This module tests that all backend implementations properly implement
the IDataBackend interface with consistent behavior.
"""

import pytest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

import anndata

from lobster.core.backends.base import BaseBackend
from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.interfaces.backend import IDataBackend

# Try to import MuData backend
try:
    from lobster.core.backends.mudata_backend import MuDataBackend
    MUDATA_AVAILABLE = True
except ImportError:
    MUDATA_AVAILABLE = False


class ConcreteBaseBackend(BaseBackend):
    """Concrete implementation of BaseBackend for testing."""

    def load(self, path, **kwargs):
        """Mock implementation for testing."""
        # Create a simple AnnData object for testing
        return anndata.AnnData(X=np.random.randn(10, 5))

    def save(self, adata, path, **kwargs):
        """Mock implementation for testing."""
        resolved_path = self._resolve_path(path)
        self._ensure_directory(resolved_path)
        # Just create an empty file for testing
        resolved_path.touch()


def create_test_adata():
    """Create a simple test AnnData object."""
    np.random.seed(42)
    X = np.random.randn(20, 10).astype(np.float32)

    obs = pd.DataFrame({
        'cell_type': np.random.choice(['A', 'B'], 20),
        'batch': np.random.choice(['1', '2'], 20)
    }, index=[f'cell_{i}' for i in range(20)])

    var = pd.DataFrame({
        'gene_name': [f'Gene_{i}' for i in range(10)]
    }, index=[f'GENE{i:03d}' for i in range(10)])

    return anndata.AnnData(X=X, obs=obs, var=var)


class TestInterfaceCompliance:
    """Test interface compliance for all backends."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_adata = create_test_adata()

        # Initialize all backends
        self.backends = {
            'base': ConcreteBaseBackend(base_path=self.temp_dir),
            'h5ad': H5ADBackend(base_path=self.temp_dir)
        }

        if MUDATA_AVAILABLE:
            self.backends['mudata'] = MuDataBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_all_backends_implement_interface(self):
        """Test that all backends implement IDataBackend interface."""
        for name, backend in self.backends.items():
            assert isinstance(backend, IDataBackend), f"{name} backend must implement IDataBackend"

    def test_all_backends_have_required_methods(self):
        """Test that all backends have required interface methods."""
        required_methods = ['load', 'save', 'exists', 'delete', 'list_files', 'get_metadata']

        for name, backend in self.backends.items():
            for method in required_methods:
                assert hasattr(backend, method), f"{name} backend missing method: {method}"
                assert callable(getattr(backend, method)), f"{name} backend {method} is not callable"

    def test_all_backends_have_optional_methods(self):
        """Test that all backends have optional interface methods."""
        optional_methods = ['get_storage_info', 'validate_path', 'supports_format']

        for name, backend in self.backends.items():
            for method in optional_methods:
                assert hasattr(backend, method), f"{name} backend missing optional method: {method}"
                assert callable(getattr(backend, method)), f"{name} backend {method} is not callable"

    def test_exists_method_consistency(self):
        """Test that exists method behaves consistently across backends."""
        test_file = "test_exists.h5ad"

        for name, backend in self.backends.items():
            # Non-existent file should return False
            assert backend.exists("nonexistent_file.h5ad") is False, f"{name} backend exists() failed for non-existent file"

            # Skip actual file creation for base backend (it's a mock)
            if name != 'base':
                # Create file and test existence
                backend.save(self.test_adata, test_file)
                assert backend.exists(test_file) is True, f"{name} backend exists() failed for existing file"

    def test_get_storage_info_structure(self):
        """Test that get_storage_info returns consistent structure."""
        required_keys = ['backend_type', 'capabilities', 'configuration']

        for name, backend in self.backends.items():
            info = backend.get_storage_info()
            assert isinstance(info, dict), f"{name} backend get_storage_info() must return dict"

            # Check for required keys from base interface
            for key in required_keys:
                assert key in info, f"{name} backend get_storage_info() missing key: {key}"

            # Check backend_type is a string
            assert isinstance(info['backend_type'], str), f"{name} backend_type must be string"

            # Check capabilities is a list
            assert isinstance(info['capabilities'], list), f"{name} capabilities must be list"

    def test_supports_format_method(self):
        """Test that supports_format method works consistently."""
        for name, backend in self.backends.items():
            # Test with known formats
            result = backend.supports_format('h5ad')
            assert isinstance(result, bool), f"{name} backend supports_format() must return bool"

            # Test case insensitivity
            result_upper = backend.supports_format('H5AD')
            assert isinstance(result_upper, bool), f"{name} backend supports_format() case handling failed"

    def test_validate_path_method(self):
        """Test that validate_path method works consistently."""
        test_paths = ['test.h5ad', 'subdir/test.h5ad']

        for name, backend in self.backends.items():
            for test_path in test_paths:
                try:
                    result = backend.validate_path(test_path)
                    # Should return a path-like object
                    assert result is not None, f"{name} backend validate_path() returned None"
                except ValueError:
                    # ValueError is acceptable for invalid paths
                    pass

    def test_path_handling_consistency(self):
        """Test that path handling is consistent across backends."""
        test_files = ['simple.h5ad', 'subdir/nested.h5ad']

        for name, backend in self.backends.items():
            if name == 'base':  # Skip base backend for actual I/O
                continue

            for test_file in test_files:
                # Save and check existence
                backend.save(self.test_adata, test_file)
                assert backend.exists(test_file), f"{name} backend path handling failed for {test_file}"

                # Clean up
                backend.delete(test_file)
                assert not backend.exists(test_file), f"{name} backend delete failed for {test_file}"

    def test_error_handling_consistency(self):
        """Test that error handling is consistent across backends."""
        for name, backend in self.backends.items():
            if name == 'base':  # Skip base backend for actual I/O
                continue

            # Test loading non-existent file
            with pytest.raises(FileNotFoundError):
                backend.load("nonexistent_file.h5ad")

            # Test deleting non-existent file
            with pytest.raises(FileNotFoundError):
                backend.delete("nonexistent_file.h5ad")

    @pytest.mark.skipif(not MUDATA_AVAILABLE, reason="MuData not available")
    def test_mudata_specific_interface_compliance(self):
        """Test MuData-specific interface compliance."""
        mudata_backend = self.backends.get('mudata')
        if mudata_backend is None:
            pytest.skip("MuData backend not available")

        # Test that MuData backend has additional methods
        mudata_specific_methods = [
            'add_modality', 'remove_modality', 'get_modality',
            'list_modalities', 'get_modality_info'
        ]

        for method in mudata_specific_methods:
            assert hasattr(mudata_backend, method), f"MuData backend missing method: {method}"
            assert callable(getattr(mudata_backend, method)), f"MuData backend {method} is not callable"

    def test_compression_support_consistency(self):
        """Test compression support consistency."""
        for name, backend in self.backends.items():
            if name == 'base':  # Skip base backend
                continue

            # Test that compression parameters are accepted
            try:
                backend.save(self.test_adata, f"test_compression_{name}.h5ad", compression="gzip")
                assert backend.exists(f"test_compression_{name}.h5ad")
            except Exception as e:
                pytest.fail(f"{name} backend failed compression test: {e}")

    def test_metadata_structure_consistency(self):
        """Test that metadata structure is consistent across backends."""
        expected_keys = ['size', 'modified', 'format', 'path', 'name']

        for name, backend in self.backends.items():
            if name == 'base':  # Skip base backend for actual I/O
                continue

            test_file = f"test_metadata_{name}.h5ad"
            backend.save(self.test_adata, test_file)

            metadata = backend.get_metadata(test_file)
            assert isinstance(metadata, dict), f"{name} backend get_metadata() must return dict"

            for key in expected_keys:
                assert key in metadata, f"{name} backend metadata missing key: {key}"

    def test_list_files_consistency(self):
        """Test that list_files behaves consistently."""
        test_files = ['file1.h5ad', 'file2.h5ad', 'other.txt']

        for name, backend in self.backends.items():
            if name == 'base':  # Skip base backend for actual I/O
                continue

            # Create test files
            for test_file in test_files:
                if test_file.endswith('.h5ad'):
                    backend.save(self.test_adata, test_file)
                else:
                    # Create non-h5ad file
                    full_path = Path(self.temp_dir) / test_file
                    full_path.touch()

            # List all files
            all_files = backend.list_files(self.temp_dir)
            assert isinstance(all_files, list), f"{name} backend list_files() must return list"

            # List with pattern
            h5ad_files = backend.list_files(self.temp_dir, "*.h5ad")
            assert isinstance(h5ad_files, list), f"{name} backend list_files() with pattern must return list"

            # H5AD pattern should return fewer or equal files
            assert len(h5ad_files) <= len(all_files), f"{name} backend pattern filtering failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])