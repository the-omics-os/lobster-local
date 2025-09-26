"""
Comprehensive tests for the H5ADBackend class.

This module tests all functionality of the H5ADBackend class including
file I/O operations, data serialization/deserialization, error handling,
performance, and S3-ready functionality.
"""

import pytest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import collections
import time
import gc

import anndata
import scanpy as sc
from scipy import sparse as sp_sparse

from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.interfaces.backend import IDataBackend


class TestH5ADBackendInitialization:
    """Test H5ADBackend initialization and configuration."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        backend = H5ADBackend()
        assert backend.base_path is None
        assert backend.compression == "gzip"
        assert backend.compression_opts == 6
        assert backend.s3_config["bucket"] is None

    def test_init_with_base_path(self):
        """Test initialization with base path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = H5ADBackend(base_path=temp_dir)
            assert backend.base_path == Path(temp_dir)

    def test_init_with_compression_settings(self):
        """Test initialization with custom compression settings."""
        backend = H5ADBackend(compression="lzf", compression_opts=9)
        assert backend.compression == "lzf"
        assert backend.compression_opts == 9

    def test_interface_compliance(self):
        """Test that H5ADBackend implements IDataBackend interface."""
        backend = H5ADBackend()
        assert isinstance(backend, IDataBackend)

    def test_s3_config_initialization(self):
        """Test S3 configuration is properly initialized."""
        backend = H5ADBackend()
        expected_keys = {"bucket", "region", "access_key", "secret_key"}
        assert set(backend.s3_config.keys()) == expected_keys
        assert all(v is None for v in backend.s3_config.values())


class TestH5ADDataCreation:
    """Helper methods for creating test AnnData objects."""

    @staticmethod
    def create_simple_adata(n_obs=100, n_vars=50, sparse=False):
        """Create a simple AnnData object for testing."""
        np.random.seed(42)

        if sparse:
            X = sp_sparse.random(n_obs, n_vars, density=0.3, format='csr', random_state=42)
        else:
            X = np.random.randn(n_obs, n_vars).astype(np.float32)

        obs = pd.DataFrame({
            'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_obs),
            'batch': np.random.choice(['batch1', 'batch2'], n_obs),
            'n_genes': np.random.randint(100, 1000, n_obs)
        }, index=[f'cell_{i}' for i in range(n_obs)])

        var = pd.DataFrame({
            'gene_name': [f'Gene_{i}' for i in range(n_vars)],
            'highly_variable': np.random.choice([True, False], n_vars),
            'mean': np.random.uniform(0, 10, n_vars)
        }, index=[f'ENSG{i:05d}' for i in range(n_vars)])

        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Add some layers, obsm, varm, uns
        adata.layers['raw'] = X.copy()
        adata.obsm['X_pca'] = np.random.randn(n_obs, 10)
        adata.varm['PCs'] = np.random.randn(n_vars, 10)
        adata.uns['method'] = 'test_data'
        adata.uns['parameters'] = {'param1': 1.5, 'param2': 'value'}

        return adata

    @staticmethod
    def create_complex_adata_with_edge_cases():
        """Create AnnData with edge cases for robust testing."""
        n_obs, n_vars = 50, 30

        # Create data with special cases
        X = np.random.randn(n_obs, n_vars).astype(np.float32)
        X[0, 0] = np.inf  # Infinity value
        X[1, 1] = -np.inf  # Negative infinity
        X[2, 2] = np.nan  # NaN value

        obs = pd.DataFrame({
            'cell_type': ['Type/A', 'Type\\B', 'Type with spaces'] * (n_obs // 3) + ['TypeA'] * (n_obs % 3),
            'unicode_col': ['αβγ', 'δεζ', 'ηθι'] * (n_obs // 3) + ['abc'] * (n_obs % 3),
            'numeric_col': np.random.randn(n_obs)
        }, index=[f'cell_{i}' for i in range(n_obs)])

        var = pd.DataFrame({
            'gene/name': [f'Gene/_{i}' for i in range(n_vars)],  # Forward slash in key
            'gene\\name': [f'Gene\\_{i}' for i in range(n_vars)],  # Backslash in key
        }, index=[f'ENSG{i:05d}' for i in range(n_vars)])

        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Add problematic uns data
        adata.uns['ordered_dict'] = collections.OrderedDict([('a', 1), ('b', 2)])
        adata.uns['tuple_data'] = (1, 2, 3)
        adata.uns['numpy_scalar'] = np.float64(3.14)
        adata.uns['nested_dict'] = {
            'level1': {
                'level2/with/slashes': 'value',
                'level2_ordered': collections.OrderedDict([('x', 1), ('y', 2)])
            }
        }

        # Add obsm, varm with slashes in keys
        adata.obsm['X/pca'] = np.random.randn(n_obs, 5)
        adata.varm['loadings/pca'] = np.random.randn(n_vars, 5)
        adata.layers['raw/counts'] = X.copy()

        return adata


class TestH5ADFileOperations:
    """Test basic file I/O operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = H5ADBackend(base_path=self.temp_dir)
        self.test_data_creator = TestH5ADDataCreation()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_simple_data(self):
        """Test saving and loading simple AnnData."""
        adata_original = self.test_data_creator.create_simple_adata()
        file_path = "test_simple.h5ad"

        # Save data
        self.backend.save(adata_original, file_path)

        # Verify file exists
        full_path = Path(self.temp_dir) / file_path
        assert full_path.exists()
        assert full_path.stat().st_size > 0

        # Load data
        adata_loaded = self.backend.load(file_path)

        # Verify data integrity
        assert adata_loaded.shape == adata_original.shape
        assert list(adata_loaded.obs.columns) == list(adata_original.obs.columns)
        assert list(adata_loaded.var.columns) == list(adata_original.var.columns)
        np.testing.assert_array_almost_equal(adata_loaded.X, adata_original.X)

    def test_save_and_load_sparse_data(self):
        """Test saving and loading sparse AnnData."""
        adata_original = self.test_data_creator.create_simple_adata(sparse=True)
        file_path = "test_sparse.h5ad"

        # Save and load
        self.backend.save(adata_original, file_path)
        adata_loaded = self.backend.load(file_path)

        # Verify sparsity is preserved
        assert hasattr(adata_loaded.X, 'nnz')  # Is sparse
        assert adata_loaded.X.format == adata_original.X.format
        np.testing.assert_array_almost_equal(adata_loaded.X.toarray(), adata_original.X.toarray())

    def test_save_with_custom_compression(self):
        """Test saving with custom compression settings."""
        adata = self.test_data_creator.create_simple_adata()
        file_path = "test_compression.h5ad"

        # Save with high compression
        self.backend.save(adata, file_path, compression="gzip", compression_opts=9)

        # Verify file exists and has expected size
        full_path = Path(self.temp_dir) / file_path
        assert full_path.exists()

        # Load and verify
        adata_loaded = self.backend.load(file_path)
        assert adata_loaded.shape == adata.shape

    def test_save_as_dense_conversion(self):
        """Test converting sparse to dense during save."""
        adata = self.test_data_creator.create_simple_adata(sparse=True)
        file_path = "test_dense_conversion.h5ad"

        # Save as dense
        self.backend.save(adata, file_path, as_dense=True)
        adata_loaded = self.backend.load(file_path)

        # Verify density
        assert not hasattr(adata_loaded.X, 'nnz')  # Is dense
        np.testing.assert_array_almost_equal(adata_loaded.X, adata.X.toarray())

    def test_load_backed_mode(self):
        """Test loading in backed mode for large files."""
        adata = self.test_data_creator.create_simple_adata(n_obs=200, n_vars=100)
        file_path = "test_backed.h5ad"

        # Save data
        self.backend.save(adata, file_path)

        # Load in backed mode
        adata_backed = self.backend.load(file_path, backed=True)

        # Verify it's backed
        assert adata_backed.isbacked
        assert adata_backed.shape == adata.shape

    def test_save_overwrite_existing_file(self):
        """Test saving over existing file."""
        adata1 = self.test_data_creator.create_simple_adata(n_obs=50)
        adata2 = self.test_data_creator.create_simple_adata(n_obs=100)
        file_path = "test_overwrite.h5ad"

        # Save first file
        self.backend.save(adata1, file_path)
        original_size = (Path(self.temp_dir) / file_path).stat().st_size

        # Save second file (overwrite)
        self.backend.save(adata2, file_path)
        new_size = (Path(self.temp_dir) / file_path).stat().st_size

        # Verify overwrite worked
        assert new_size != original_size

        # Verify content is from second save
        adata_loaded = self.backend.load(file_path)
        assert adata_loaded.shape == adata2.shape


class TestH5ADDataSanitization:
    """Test data sanitization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = H5ADBackend(base_path=self.temp_dir)
        self.test_data_creator = TestH5ADDataCreation()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sanitize_edge_cases(self):
        """Test sanitization of edge cases."""
        adata = self.test_data_creator.create_complex_adata_with_edge_cases()
        file_path = "test_sanitization.h5ad"

        # This should not raise errors despite edge cases
        self.backend.save(adata, file_path)
        adata_loaded = self.backend.load(file_path)

        # Verify basic structure is preserved
        assert adata_loaded.shape == adata.shape

        # Verify sanitization worked
        assert 'ordered_dict' in adata_loaded.uns
        assert isinstance(adata_loaded.uns['ordered_dict'], dict)  # Converted from OrderedDict

        # Verify tuple to list conversion
        assert 'tuple_data' in adata_loaded.uns
        assert isinstance(adata_loaded.uns['tuple_data'], list)

        # Check slash replacement in keys
        obsm_keys = list(adata_loaded.obsm.keys())
        assert any('__' in key for key in obsm_keys)  # Slashes should be replaced

    def test_sanitize_anndata_method(self):
        """Test the static sanitize_anndata method directly."""
        # Create test data with problematic elements
        adata = anndata.AnnData(X=np.random.randn(10, 5))
        adata.uns['ordered'] = collections.OrderedDict([('a', 1), ('b', 2)])
        adata.uns['tuple'] = (1, 2, 3)
        adata.uns['numpy_scalar'] = np.float64(3.14)
        adata.uns['nested'] = {
            'level1/slash': 'value',
            'ordered_nested': collections.OrderedDict([('x', 1)])
        }
        adata.obsm['X/pca'] = np.random.randn(10, 2)
        adata.layers['raw/counts'] = np.random.randn(10, 5)

        # Sanitize
        adata_sanitized = H5ADBackend.sanitize_anndata(adata)

        # Check conversions
        assert isinstance(adata_sanitized.uns['ordered'], dict)
        assert isinstance(adata_sanitized.uns['tuple'], list)
        assert isinstance(adata_sanitized.uns['numpy_scalar'], float)
        assert 'level1__slash' in adata_sanitized.uns['nested']
        assert 'X__pca' in adata_sanitized.obsm
        assert 'raw__counts' in adata_sanitized.layers

    def test_variable_names_make_unique(self):
        """Test that variable names are made unique."""
        # Create data with duplicate var names
        X = np.random.randn(10, 5)
        var_names = ['Gene1', 'Gene2', 'Gene1', 'Gene3', 'Gene2']  # Duplicates

        adata = anndata.AnnData(X=X)
        adata.var_names = var_names

        file_path = "test_unique_vars.h5ad"
        self.backend.save(adata, file_path)
        adata_loaded = self.backend.load(file_path)

        # Verify all var names are unique
        var_names_loaded = list(adata_loaded.var_names)
        assert len(var_names_loaded) == len(set(var_names_loaded))

        # Verify duplicates were handled
        assert 'Gene1' in var_names_loaded
        assert 'Gene1__2' in var_names_loaded or 'Gene1-1' in var_names_loaded


class TestH5ADErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = H5ADBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.backend.load("nonexistent.h5ad")

    def test_load_invalid_h5ad_file(self):
        """Test loading invalid H5AD file."""
        # Create a file that's not a valid H5AD
        invalid_file = Path(self.temp_dir) / "invalid.h5ad"
        invalid_file.write_text("This is not an H5AD file")

        with pytest.raises(ValueError, match="Failed to load H5AD file"):
            self.backend.load("invalid.h5ad")

    def test_save_to_readonly_directory(self):
        """Test saving to read-only directory."""
        # Create a read-only subdirectory
        readonly_dir = Path(self.temp_dir) / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        try:
            adata = TestH5ADDataCreation.create_simple_adata()
            with pytest.raises(ValueError, match="Failed to save H5AD file"):
                self.backend.save(adata, readonly_dir / "test.h5ad")
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)

    def test_save_invalid_data_type(self):
        """Test saving invalid data type."""
        invalid_data = "not an AnnData object"

        with pytest.raises((AttributeError, ValueError)):
            self.backend.save(invalid_data, "test.h5ad")

    def test_save_cleanup_on_failure(self):
        """Test that failed saves are cleaned up."""
        adata = TestH5ADDataCreation.create_simple_adata()
        file_path = Path(self.temp_dir) / "test_cleanup.h5ad"

        # Mock write_h5ad to fail
        with patch.object(anndata.AnnData, 'write_h5ad', side_effect=Exception("Write failed")):
            with pytest.raises(ValueError, match="Failed to save H5AD file"):
                self.backend.save(adata, file_path)

        # Verify file was cleaned up
        assert not file_path.exists()

    def test_path_resolution_errors(self):
        """Test path resolution error handling."""
        backend_no_base = H5ADBackend()

        # This should work fine - path resolution should be robust
        adata = TestH5ADDataCreation.create_simple_adata()
        temp_file = Path(self.temp_dir) / "test_path.h5ad"

        backend_no_base.save(adata, str(temp_file))
        assert temp_file.exists()

    def test_corrupted_file_handling(self):
        """Test handling of corrupted H5AD files."""
        # Create a corrupted H5AD file (starts right but is truncated)
        adata = TestH5ADDataCreation.create_simple_adata()
        file_path = Path(self.temp_dir) / "corrupted.h5ad"

        # Save valid file first
        self.backend.save(adata, file_path)

        # Corrupt the file by truncating it
        with open(file_path, 'r+b') as f:
            f.truncate(100)  # Truncate to 100 bytes

        # Loading should fail gracefully
        with pytest.raises(ValueError, match="Failed to load H5AD file"):
            self.backend.load(file_path)


class TestH5ADFileIntegrity:
    """Test file integrity validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = H5ADBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_valid_file(self):
        """Test validation of valid H5AD file."""
        adata = TestH5ADDataCreation.create_simple_adata()
        file_path = "test_valid.h5ad"

        self.backend.save(adata, file_path)
        validation = self.backend.validate_file_integrity(file_path)

        assert validation["valid"] is True
        assert validation["readable"] is True
        assert validation["has_X"] is True
        assert validation["has_obs"] is True
        assert validation["has_var"] is True
        assert validation["shape"] == adata.shape
        assert len(validation["errors"]) == 0

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        validation = self.backend.validate_file_integrity("nonexistent.h5ad")

        assert validation["valid"] is False
        assert validation["readable"] is False
        assert "File does not exist" in validation["errors"]

    def test_validate_empty_dataset(self):
        """Test validation of dataset with no observations or variables."""
        # Create empty AnnData
        adata = anndata.AnnData(X=np.array([]).reshape(0, 10))
        file_path = "test_empty.h5ad"

        self.backend.save(adata, file_path)
        validation = self.backend.validate_file_integrity(file_path)

        assert validation["readable"] is True
        assert "No observations in dataset" in validation["warnings"]

    def test_validate_corrupted_file(self):
        """Test validation of corrupted file."""
        # Create invalid file
        invalid_file = Path(self.temp_dir) / "corrupted.h5ad"
        invalid_file.write_bytes(b"invalid data")

        validation = self.backend.validate_file_integrity("corrupted.h5ad")

        assert validation["valid"] is False
        assert len(validation["errors"]) > 0


class TestH5ADPerformanceOptimization:
    """Test performance optimization features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = H5ADBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_optimize_for_reading_sparse(self):
        """Test optimization analysis for sparse data."""
        adata = TestH5ADDataCreation.create_simple_adata(n_obs=1000, n_vars=500, sparse=True)
        file_path = "test_optimize_sparse.h5ad"

        self.backend.save(adata, file_path)
        analysis = self.backend.optimize_for_reading(file_path)

        assert "shape" in analysis
        assert analysis["is_sparse"] is True
        assert "sparsity" in analysis
        assert analysis["sparsity"] > 0  # Should be sparse
        assert "recommended_chunk_size" in analysis
        assert "estimated_memory_mb" in analysis

    def test_optimize_for_reading_dense(self):
        """Test optimization analysis for dense data."""
        adata = TestH5ADDataCreation.create_simple_adata(n_obs=100, n_vars=50, sparse=False)
        file_path = "test_optimize_dense.h5ad"

        self.backend.save(adata, file_path)
        analysis = self.backend.optimize_for_reading(file_path)

        assert analysis["is_sparse"] is False
        assert "sparsity" not in analysis or analysis["sparsity"] is None
        assert analysis["estimated_memory_mb"] > 0

    def test_optimize_nonexistent_file(self):
        """Test optimization of non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.backend.optimize_for_reading("nonexistent.h5ad")

    def test_optimize_with_error(self):
        """Test optimization error handling."""
        # Create invalid H5AD file
        invalid_file = Path(self.temp_dir) / "invalid.h5ad"
        invalid_file.write_text("invalid")

        analysis = self.backend.optimize_for_reading("invalid.h5ad")
        assert "error" in analysis


class TestH5ADS3Functionality:
    """Test S3-ready functionality (placeholder implementations)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = H5ADBackend()

    def test_s3_path_detection(self):
        """Test S3 path detection."""
        assert self.backend._is_s3_path("s3://bucket/path/file.h5ad") is True
        assert self.backend._is_s3_path("local/path/file.h5ad") is False
        assert self.backend._is_s3_path("/absolute/path/file.h5ad") is False

    def test_s3_load_not_implemented(self):
        """Test that S3 loading raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="S3 support is not yet implemented"):
            self.backend._load_from_s3("s3://bucket/file.h5ad")

    def test_s3_save_not_implemented(self):
        """Test that S3 saving raises NotImplementedError."""
        adata = TestH5ADDataCreation.create_simple_adata()
        with pytest.raises(NotImplementedError, match="S3 support is not yet implemented"):
            self.backend._save_to_s3(adata, "s3://bucket/file.h5ad")

    def test_configure_s3(self):
        """Test S3 configuration."""
        self.backend.configure_s3(
            bucket="test-bucket",
            region="us-west-2",
            access_key="access",
            secret_key="secret"
        )

        assert self.backend.s3_config["bucket"] == "test-bucket"
        assert self.backend.s3_config["region"] == "us-west-2"
        assert self.backend.s3_config["access_key"] == "access"
        assert self.backend.s3_config["secret_key"] == "secret"

    def test_s3_path_resolution(self):
        """Test S3 path resolution."""
        s3_path = "s3://bucket/path/file.h5ad"
        resolved = self.backend._resolve_path_with_s3_support(s3_path)
        # S3 paths should be returned as Path objects but maintain the S3 format
        assert "bucket/path/file.h5ad" in str(resolved)


class TestH5ADFormatSupport:
    """Test format support and detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = H5ADBackend()

    def test_supports_h5ad_format(self):
        """Test support for H5AD format."""
        assert self.backend.supports_format("h5ad") is True
        assert self.backend.supports_format("H5AD") is True

    def test_supports_h5_format(self):
        """Test support for H5 format."""
        assert self.backend.supports_format("h5") is True
        assert self.backend.supports_format("H5") is True

    def test_does_not_support_other_formats(self):
        """Test lack of support for other formats."""
        unsupported = ["csv", "xlsx", "h5mu", "zarr", "unknown"]
        for fmt in unsupported:
            assert self.backend.supports_format(fmt) is False

    def test_detect_format_h5ad(self):
        """Test H5AD format detection."""
        assert self.backend._detect_format("test.h5ad") == "h5ad"
        assert self.backend._detect_format("test.H5AD") == "h5ad"

    def test_detect_format_h5(self):
        """Test H5 format detection."""
        assert self.backend._detect_format("test.h5") == "h5"

    def test_detect_format_unknown(self):
        """Test unknown format detection."""
        assert self.backend._detect_format("test.csv") == "unknown"


class TestH5ADStorageInfo:
    """Test storage information retrieval."""

    def test_get_storage_info(self):
        """Test getting storage information."""
        backend = H5ADBackend(compression="lzf", compression_opts=3)
        info = backend.get_storage_info()

        assert info["supported_formats"] == ["h5ad"]
        assert info["compression"] == "lzf"
        assert info["compression_opts"] == 3
        assert info["s3_ready"] is True
        assert info["backed_mode_support"] is True
        assert info["backend_type"] == "H5ADBackend"


class TestH5ADPerformanceAndMemory:
    """Test performance and memory efficiency."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = H5ADBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_large_file_handling(self):
        """Test handling of reasonably large files."""
        # Create moderately large dataset
        adata = TestH5ADDataCreation.create_simple_adata(n_obs=2000, n_vars=1000, sparse=True)
        file_path = "test_large.h5ad"

        # Measure save time
        start_time = time.time()
        self.backend.save(adata, file_path)
        save_time = time.time() - start_time

        # Should complete in reasonable time (less than 30 seconds)
        assert save_time < 30

        # Measure load time
        start_time = time.time()
        adata_loaded = self.backend.load(file_path)
        load_time = time.time() - start_time

        assert load_time < 30
        assert adata_loaded.shape == adata.shape

    def test_memory_efficiency_backed_mode(self):
        """Test memory efficiency of backed mode."""
        # Create larger dataset
        adata = TestH5ADDataCreation.create_simple_adata(n_obs=1000, n_vars=500)
        file_path = "test_memory.h5ad"

        self.backend.save(adata, file_path)

        # Load in backed mode should use less memory
        adata_backed = self.backend.load(file_path, backed=True)

        # Basic verification
        assert adata_backed.isbacked
        assert adata_backed.shape == adata.shape

        # Should be able to access data
        assert adata_backed.X[0, 0] is not None

    def test_compression_efficiency(self):
        """Test compression efficiency."""
        adata = TestH5ADDataCreation.create_simple_adata(n_obs=500, n_vars=200)

        # Save with different compression levels
        file_low = "test_comp_low.h5ad"
        file_high = "test_comp_high.h5ad"

        self.backend.save(adata, file_low, compression_opts=1)
        self.backend.save(adata, file_high, compression_opts=9)

        # High compression should result in smaller file
        size_low = (Path(self.temp_dir) / file_low).stat().st_size
        size_high = (Path(self.temp_dir) / file_high).stat().st_size

        assert size_high <= size_low  # Higher compression should be smaller or equal

    def test_concurrent_read_simulation(self):
        """Test simulation of concurrent reads."""
        adata = TestH5ADDataCreation.create_simple_adata()
        file_path = "test_concurrent.h5ad"

        self.backend.save(adata, file_path)

        # Simulate multiple rapid reads
        for i in range(5):
            adata_loaded = self.backend.load(file_path)
            assert adata_loaded.shape == adata.shape

            # Small delay to simulate concurrent access patterns
            time.sleep(0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])