"""
Comprehensive tests for the MuDataBackend class.

This module tests all functionality of the MuDataBackend class including
multi-modal data handling, file I/O operations, modality management,
data serialization/deserialization, and error handling.
"""

import pytest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

import anndata
from scipy import sparse as sp_sparse

# Try to import mudata - backend will handle the fallback
try:
    import mudata
    MUDATA_AVAILABLE = True
except ImportError:
    MUDATA_AVAILABLE = False

from lobster.core.backends.mudata_backend import MuDataBackend
from lobster.core.interfaces.backend import IDataBackend


class TestMuDataAvailability:
    """Test MuData availability and import handling."""

    def test_mudata_import_available(self):
        """Test that MuData is available for testing."""
        assert MUDATA_AVAILABLE, "MuData must be available for testing"

    def test_backend_requires_mudata(self):
        """Test that backend initialization checks for MuData."""
        if not MUDATA_AVAILABLE:
            with pytest.raises(ImportError, match="MuData is not available"):
                MuDataBackend()


@pytest.mark.skipif(not MUDATA_AVAILABLE, reason="MuData not available")
class TestMuDataBackendInitialization:
    """Test MuDataBackend initialization and configuration."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        backend = MuDataBackend()
        assert backend.base_path is None
        assert backend.compression == "gzip"
        assert backend.compression_opts == 6

    def test_init_with_base_path(self):
        """Test initialization with base path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = MuDataBackend(base_path=temp_dir)
            assert backend.base_path == Path(temp_dir)

    def test_init_with_compression_settings(self):
        """Test initialization with custom compression settings."""
        backend = MuDataBackend(compression="lzf", compression_opts=9)
        assert backend.compression == "lzf"
        assert backend.compression_opts == 9

    def test_interface_compliance(self):
        """Test that MuDataBackend implements IDataBackend interface."""
        backend = MuDataBackend()
        assert isinstance(backend, IDataBackend)


class TestMuDataCreation:
    """Helper methods for creating test multi-modal data objects."""

    @staticmethod
    def create_rna_data(n_obs=100, n_vars=50, sparse=False):
        """Create RNA-seq AnnData for testing."""
        np.random.seed(42)

        if sparse:
            X = sp_sparse.random(n_obs, n_vars, density=0.3, format='csr', random_state=42)
        else:
            X = np.random.randn(n_obs, n_vars).astype(np.float32)

        obs = pd.DataFrame({
            'cell_type': np.random.choice(['T_cell', 'B_cell', 'NK_cell'], n_obs),
            'batch': np.random.choice(['batch1', 'batch2'], n_obs),
            'n_genes_rna': np.random.randint(1000, 5000, n_obs)
        }, index=[f'cell_{i}' for i in range(n_obs)])

        var = pd.DataFrame({
            'gene_name': [f'Gene_{i}' for i in range(n_vars)],
            'highly_variable': np.random.choice([True, False], n_vars),
            'gene_type': 'protein_coding'
        }, index=[f'ENSG{i:05d}' for i in range(n_vars)])

        adata = anndata.AnnData(X=X, obs=obs, var=var)
        adata.layers['raw'] = X.copy()
        adata.obsm['X_pca'] = np.random.randn(n_obs, 10)
        adata.varm['PCs'] = np.random.randn(n_vars, 10)
        adata.uns['method'] = 'rna_seq'

        return adata

    @staticmethod
    def create_protein_data(n_obs=100, n_proteins=20):
        """Create protein/CITE-seq AnnData for testing."""
        np.random.seed(43)

        # Protein data is typically less sparse
        X = np.random.exponential(scale=2.0, size=(n_obs, n_proteins)).astype(np.float32)

        obs = pd.DataFrame({
            'cell_type': np.random.choice(['T_cell', 'B_cell', 'NK_cell'], n_obs),
            'batch': np.random.choice(['batch1', 'batch2'], n_obs),
            'n_proteins': np.random.randint(15, 20, n_obs)
        }, index=[f'cell_{i}' for i in range(n_obs)])

        var = pd.DataFrame({
            'protein_name': [f'CD{i}' for i in range(n_proteins)],
            'antibody_clone': [f'clone_{i}' for i in range(n_proteins)],
            'protein_type': 'surface'
        }, index=[f'PROT{i:03d}' for i in range(n_proteins)])

        adata = anndata.AnnData(X=X, obs=obs, var=var)
        adata.layers['raw'] = X.copy()
        adata.obsm['X_pca'] = np.random.randn(n_obs, 5)
        adata.uns['method'] = 'cite_seq'

        return adata

    @staticmethod
    def create_atac_data(n_obs=100, n_peaks=200):
        """Create ATAC-seq AnnData for testing."""
        np.random.seed(44)

        # ATAC data is typically very sparse
        X = sp_sparse.random(n_obs, n_peaks, density=0.1, format='csr', random_state=44)

        obs = pd.DataFrame({
            'cell_type': np.random.choice(['T_cell', 'B_cell', 'NK_cell'], n_obs),
            'batch': np.random.choice(['batch1', 'batch2'], n_obs),
            'n_peaks': np.random.randint(100, 300, n_obs)
        }, index=[f'cell_{i}' for i in range(n_obs)])

        var = pd.DataFrame({
            'peak_name': [f'peak_{i}' for i in range(n_peaks)],
            'chromosome': np.random.choice(['chr1', 'chr2', 'chr3'], n_peaks),
            'peak_type': 'accessible'
        }, index=[f'PEAK{i:05d}' for i in range(n_peaks)])

        adata = anndata.AnnData(X=X, obs=obs, var=var)
        adata.layers['raw'] = X.copy()
        adata.obsm['X_lsi'] = np.random.randn(n_obs, 8)
        adata.uns['method'] = 'atac_seq'

        return adata

    @staticmethod
    def create_simple_mudata():
        """Create a simple MuData object with RNA and protein modalities."""
        rna_data = TestMuDataCreation.create_rna_data(n_obs=50, n_vars=30)
        protein_data = TestMuDataCreation.create_protein_data(n_obs=50, n_proteins=15)

        mdata = mudata.MuData({'rna': rna_data, 'protein': protein_data})
        return mdata

    @staticmethod
    def create_complex_mudata():
        """Create a complex MuData object with multiple modalities."""
        rna_data = TestMuDataCreation.create_rna_data(n_obs=100, n_vars=50, sparse=True)
        protein_data = TestMuDataCreation.create_protein_data(n_obs=100, n_proteins=20)
        atac_data = TestMuDataCreation.create_atac_data(n_obs=100, n_peaks=100)

        mdata = mudata.MuData({
            'rna': rna_data,
            'protein': protein_data,
            'atac': atac_data
        })

        # Add global observations
        mdata.obs['global_cell_type'] = np.random.choice(['TypeA', 'TypeB'], 100)
        mdata.obs['sample_id'] = np.random.choice(['sample1', 'sample2', 'sample3'], 100)

        return mdata


@pytest.mark.skipif(not MUDATA_AVAILABLE, reason="MuData not available")
class TestMuDataFileOperations:
    """Test basic file I/O operations for MuData."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = MuDataBackend(base_path=self.temp_dir)
        self.test_data_creator = TestMuDataCreation()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_simple_mudata(self):
        """Test saving and loading simple multi-modal data."""
        mdata_original = self.test_data_creator.create_simple_mudata()
        file_path = "test_simple.h5mu"

        # Save data
        self.backend.save(mdata_original, file_path)

        # Verify file exists
        full_path = Path(self.temp_dir) / file_path
        assert full_path.exists()
        assert full_path.stat().st_size > 0

        # Load data
        mdata_loaded = self.backend.load(file_path)

        # Verify data integrity
        assert mdata_loaded.shape == mdata_original.shape
        assert set(mdata_loaded.mod.keys()) == set(mdata_original.mod.keys())

        # Check individual modalities
        for mod_name in mdata_original.mod.keys():
            assert mdata_loaded.mod[mod_name].shape == mdata_original.mod[mod_name].shape

    def test_save_and_load_complex_mudata(self):
        """Test saving and loading complex multi-modal data."""
        mdata_original = self.test_data_creator.create_complex_mudata()
        file_path = "test_complex.h5mu"

        # Save and load
        self.backend.save(mdata_original, file_path)
        mdata_loaded = self.backend.load(file_path)

        # Verify structure
        assert len(mdata_loaded.mod) == 3
        assert 'rna' in mdata_loaded.mod
        assert 'protein' in mdata_loaded.mod
        assert 'atac' in mdata_loaded.mod

        # Verify global observations
        assert 'global_cell_type' in mdata_loaded.obs.columns
        assert 'sample_id' in mdata_loaded.obs.columns

    def test_save_anndata_as_mudata(self):
        """Test saving AnnData object as single-modality MuData."""
        adata = self.test_data_creator.create_rna_data()
        file_path = "test_anndata_to_mudata.h5mu"

        # Save AnnData as MuData (should auto-convert)
        self.backend.save(adata, file_path, modality_name='transcriptomics')
        mdata_loaded = self.backend.load(file_path)

        # Verify conversion
        assert isinstance(mdata_loaded, mudata.MuData)
        assert 'transcriptomics' in mdata_loaded.mod
        assert mdata_loaded.mod['transcriptomics'].shape == adata.shape

    def test_save_with_custom_compression(self):
        """Test saving with custom compression settings."""
        mdata = self.test_data_creator.create_simple_mudata()
        file_path = "test_compression.h5mu"

        # Save with high compression
        self.backend.save(mdata, file_path, compression="gzip", compression_opts=9)

        # Verify file exists and can be loaded
        full_path = Path(self.temp_dir) / file_path
        assert full_path.exists()

        mdata_loaded = self.backend.load(file_path)
        assert len(mdata_loaded.mod) == len(mdata.mod)

    def test_load_backed_mode(self):
        """Test loading in backed mode for large files."""
        mdata = self.test_data_creator.create_complex_mudata()
        file_path = "test_backed.h5mu"

        # Save data
        self.backend.save(mdata, file_path)

        # Load in backed mode
        mdata_backed = self.backend.load(file_path, backed=True)

        # Verify it's backed
        assert mdata_backed.isbacked
        assert mdata_backed.shape == mdata.shape

    def test_backup_creation(self):
        """Test backup creation when overwriting files."""
        mdata1 = self.test_data_creator.create_simple_mudata()
        mdata2 = self.test_data_creator.create_complex_mudata()
        file_path = "test_backup.h5mu"

        # Save first file
        self.backend.save(mdata1, file_path)
        original_size = (Path(self.temp_dir) / file_path).stat().st_size

        # Save second file (should create backup)
        self.backend.save(mdata2, file_path)
        new_size = (Path(self.temp_dir) / file_path).stat().st_size

        # Verify overwrite worked
        assert new_size != original_size

        # Check for backup file (look for .backup in filename)
        backup_files = list(Path(self.temp_dir).glob("*backup*"))
        assert len(backup_files) >= 1


@pytest.mark.skipif(not MUDATA_AVAILABLE, reason="MuData not available")
class TestMuDataModalityManagement:
    """Test modality management operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = MuDataBackend(base_path=self.temp_dir)
        self.test_data_creator = TestMuDataCreation()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_modality(self):
        """Test adding a new modality to existing MuData."""
        mdata = self.test_data_creator.create_simple_mudata()  # rna + protein
        new_modality = self.test_data_creator.create_atac_data(n_obs=50, n_peaks=30)

        # Add new modality
        updated_mdata = self.backend.add_modality(mdata, 'atac', new_modality)

        # Verify addition
        assert 'atac' in updated_mdata.mod
        assert len(updated_mdata.mod) == 3
        assert updated_mdata.mod['atac'].shape == new_modality.shape

    def test_add_modality_existing_name_fails(self):
        """Test that adding modality with existing name fails."""
        mdata = self.test_data_creator.create_simple_mudata()
        duplicate_rna = self.test_data_creator.create_rna_data()

        with pytest.raises(ValueError, match="Modality 'rna' already exists"):
            self.backend.add_modality(mdata, 'rna', duplicate_rna)

    def test_remove_modality(self):
        """Test removing a modality from MuData."""
        mdata = self.test_data_creator.create_complex_mudata()  # rna + protein + atac
        original_count = len(mdata.mod)

        # Remove protein modality
        updated_mdata = self.backend.remove_modality(mdata, 'protein')

        # Verify removal
        assert 'protein' not in updated_mdata.mod
        assert len(updated_mdata.mod) == original_count - 1
        assert 'rna' in updated_mdata.mod  # Others should remain

    def test_remove_nonexistent_modality_fails(self):
        """Test that removing non-existent modality fails."""
        mdata = self.test_data_creator.create_simple_mudata()

        with pytest.raises(ValueError, match="Modality 'nonexistent' does not exist"):
            self.backend.remove_modality(mdata, 'nonexistent')

    def test_get_modality(self):
        """Test extracting a specific modality."""
        mdata = self.test_data_creator.create_simple_mudata()

        # Get RNA modality
        rna_data = self.backend.get_modality(mdata, 'rna')

        # Verify extraction
        assert isinstance(rna_data, anndata.AnnData)
        assert rna_data.shape == mdata.mod['rna'].shape
        assert 'method' in rna_data.uns

    def test_get_nonexistent_modality_fails(self):
        """Test that getting non-existent modality fails."""
        mdata = self.test_data_creator.create_simple_mudata()

        with pytest.raises(ValueError, match="Modality 'nonexistent' does not exist"):
            self.backend.get_modality(mdata, 'nonexistent')

    def test_list_modalities(self):
        """Test listing all modalities."""
        mdata = self.test_data_creator.create_complex_mudata()

        modalities = self.backend.list_modalities(mdata)

        assert isinstance(modalities, list)
        assert set(modalities) == {'rna', 'protein', 'atac'}

    def test_get_modality_info(self):
        """Test getting information about all modalities."""
        mdata = self.test_data_creator.create_simple_mudata()

        info = self.backend.get_modality_info(mdata)

        # Verify structure
        assert isinstance(info, dict)
        assert 'rna' in info
        assert 'protein' in info

        # Verify RNA info
        rna_info = info['rna']
        assert 'shape' in rna_info
        assert 'n_obs' in rna_info
        assert 'n_vars' in rna_info
        assert 'obs_columns' in rna_info
        assert 'var_columns' in rna_info

        # Verify content
        assert rna_info['n_obs'] == mdata.mod['rna'].n_obs
        assert rna_info['n_vars'] == mdata.mod['rna'].n_vars


@pytest.mark.skipif(not MUDATA_AVAILABLE, reason="MuData not available")
class TestMuDataConversionOperations:
    """Test data conversion operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = MuDataBackend()
        self.test_data_creator = TestMuDataCreation()

    def test_convert_anndata_to_mudata(self):
        """Test converting AnnData to MuData."""
        adata = self.test_data_creator.create_rna_data()

        mdata = self.backend.convert_to_mudata(adata, modality_name='transcripts')

        # Verify conversion
        assert isinstance(mdata, mudata.MuData)
        assert 'transcripts' in mdata.mod
        assert mdata.mod['transcripts'].shape == adata.shape

    def test_convert_dict_to_mudata(self):
        """Test converting dictionary of AnnData to MuData."""
        rna_data = self.test_data_creator.create_rna_data()
        protein_data = self.test_data_creator.create_protein_data()

        modality_dict = {'rna': rna_data, 'protein': protein_data}
        mdata = self.backend.convert_to_mudata(modality_dict)

        # Verify conversion
        assert isinstance(mdata, mudata.MuData)
        assert set(mdata.mod.keys()) == {'rna', 'protein'}

    def test_convert_invalid_type_fails(self):
        """Test that converting invalid data type fails."""
        invalid_data = "not valid data"

        with pytest.raises(TypeError, match="Unsupported data type"):
            self.backend.convert_to_mudata(invalid_data)

    def test_create_mudata_from_dict(self):
        """Test creating MuData from dictionary with global obs."""
        rna_data = self.test_data_creator.create_rna_data(n_obs=30)
        protein_data = self.test_data_creator.create_protein_data(n_obs=30)

        modality_dict = {'rna': rna_data, 'protein': protein_data}
        global_obs = {'experiment': 'test_experiment', 'date': '2023-01-01'}

        mdata = self.backend.create_mudata_from_dict(modality_dict, global_obs)

        # Verify creation
        assert len(mdata.mod) == 2
        assert 'experiment' in mdata.obs.columns
        assert 'date' in mdata.obs.columns

    def test_merge_mudata_objects(self):
        """Test merging multiple MuData objects."""
        mdata1 = self.test_data_creator.create_simple_mudata()
        mdata2 = mudata.MuData({'atac': self.test_data_creator.create_atac_data(n_obs=50)})

        merged = self.backend.merge_mudata_objects([mdata1, mdata2])

        # Verify merge
        assert len(merged.mod) == 3
        assert 'rna' in merged.mod
        assert 'protein' in merged.mod
        assert 'atac' in merged.mod

    def test_merge_empty_list_fails(self):
        """Test that merging empty list fails."""
        with pytest.raises(ValueError, match="No MuData objects provided"):
            self.backend.merge_mudata_objects([])

    def test_merge_single_object_returns_copy(self):
        """Test that merging single object returns a copy."""
        mdata = self.test_data_creator.create_simple_mudata()

        merged = self.backend.merge_mudata_objects([mdata])

        # Should be a copy, not the same object
        assert merged is not mdata
        assert len(merged.mod) == len(mdata.mod)

    def test_merge_with_conflicting_modality_names(self):
        """Test merging objects with conflicting modality names."""
        mdata1 = self.test_data_creator.create_simple_mudata()
        mdata2 = mudata.MuData({'rna': self.test_data_creator.create_rna_data()})  # Conflicting 'rna'

        merged = self.backend.merge_mudata_objects([mdata1, mdata2])

        # Should have renamed the conflicting modality
        modality_names = list(merged.mod.keys())
        rna_modalities = [name for name in modality_names if name.startswith('rna')]

        assert len(rna_modalities) == 2
        assert 'rna' in rna_modalities
        assert any(name != 'rna' for name in rna_modalities)

    def test_merge_invalid_strategy_fails(self):
        """Test that invalid merge strategy fails."""
        mdata1 = self.test_data_creator.create_simple_mudata()
        mdata2 = self.test_data_creator.create_simple_mudata()

        with pytest.raises(ValueError, match="Unknown merge strategy"):
            self.backend.merge_mudata_objects([mdata1, mdata2], merge_strategy="invalid")


@pytest.mark.skipif(not MUDATA_AVAILABLE, reason="MuData not available")
class TestMuDataErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = MuDataBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.backend.load("nonexistent.h5mu")

    def test_load_invalid_h5mu_file(self):
        """Test loading invalid H5MU file."""
        # Create a file that's not a valid H5MU
        invalid_file = Path(self.temp_dir) / "invalid.h5mu"
        invalid_file.write_text("This is not an H5MU file")

        with pytest.raises(ValueError, match="Failed to load MuData file"):
            self.backend.load("invalid.h5mu")

    def test_save_invalid_data_type(self):
        """Test saving invalid data type."""
        invalid_data = {"not": "a mudata object"}

        with pytest.raises(TypeError, match="Expected MuData or AnnData object"):
            self.backend.save(invalid_data, "test.h5mu")

    def test_save_cleanup_on_failure(self):
        """Test that failed saves are cleaned up."""
        mdata = TestMuDataCreation.create_simple_mudata()
        file_path = Path(self.temp_dir) / "test_cleanup.h5mu"

        # Mock mudata.write_h5mu to fail
        with patch('mudata.write_h5mu', side_effect=Exception("Write failed")):
            with pytest.raises(ValueError, match="Failed to save MuData file"):
                self.backend.save(mdata, file_path)

        # Verify file was cleaned up
        assert not file_path.exists()

    def test_corrupted_file_handling(self):
        """Test handling of corrupted H5MU files."""
        # Create a corrupted H5MU file
        mdata = TestMuDataCreation.create_simple_mudata()
        file_path = Path(self.temp_dir) / "corrupted.h5mu"

        # Save valid file first
        self.backend.save(mdata, file_path)

        # Corrupt the file by truncating it
        with open(file_path, 'r+b') as f:
            f.truncate(100)  # Truncate to 100 bytes

        # Loading should fail gracefully
        with pytest.raises(ValueError, match="Failed to load MuData file"):
            self.backend.load(file_path)


@pytest.mark.skipif(not MUDATA_AVAILABLE, reason="MuData not available")
class TestMuDataFileIntegrity:
    """Test file integrity validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = MuDataBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_valid_file(self):
        """Test validation of valid H5MU file."""
        mdata = TestMuDataCreation.create_simple_mudata()
        file_path = "test_valid.h5mu"

        self.backend.save(mdata, file_path)
        validation = self.backend.validate_file_integrity(file_path)

        assert validation["valid"] is True
        assert validation["readable"] is True
        assert validation["n_modalities"] == 2
        assert set(validation["modalities"]) == {'rna', 'protein'}
        assert validation["global_shape"] == mdata.shape
        assert len(validation["errors"]) == 0

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        validation = self.backend.validate_file_integrity("nonexistent.h5mu")

        assert validation["valid"] is False
        assert validation["readable"] is False
        assert "File does not exist" in validation["errors"]

    def test_validate_empty_modalities(self):
        """Test validation of MuData with empty modalities."""
        # Create MuData with empty modality
        empty_adata = anndata.AnnData(X=np.array([]).reshape(0, 10))
        mdata = mudata.MuData({'empty': empty_adata})
        file_path = "test_empty_modality.h5mu"

        self.backend.save(mdata, file_path)
        validation = self.backend.validate_file_integrity(file_path)

        assert validation["readable"] is True
        assert "Modality 'empty' has no observations" in validation["warnings"]

    def test_validate_no_modalities(self):
        """Test validation of MuData with no modalities."""
        # Create empty MuData
        mdata = mudata.MuData({})
        file_path = "test_no_modalities.h5mu"

        self.backend.save(mdata, file_path)
        validation = self.backend.validate_file_integrity(file_path)

        assert validation["valid"] is False
        assert "No modalities found in MuData object" in validation["errors"]


@pytest.mark.skipif(not MUDATA_AVAILABLE, reason="MuData not available")
class TestMuDataFormatSupport:
    """Test format support and detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = MuDataBackend()

    def test_supports_h5mu_format(self):
        """Test support for H5MU format."""
        assert self.backend.supports_format("h5mu") is True
        assert self.backend.supports_format("H5MU") is True

    def test_supports_mudata_format(self):
        """Test support for mudata format."""
        assert self.backend.supports_format("mudata") is True
        assert self.backend.supports_format("MUDATA") is True

    def test_does_not_support_other_formats(self):
        """Test lack of support for other formats."""
        unsupported = ["h5ad", "csv", "xlsx", "zarr", "unknown"]
        for fmt in unsupported:
            assert self.backend.supports_format(fmt) is False


@pytest.mark.skipif(not MUDATA_AVAILABLE, reason="MuData not available")
class TestMuDataStorageInfo:
    """Test storage information retrieval."""

    def test_get_storage_info(self):
        """Test getting storage information."""
        backend = MuDataBackend(compression="lzf", compression_opts=3)
        info = backend.get_storage_info()

        assert info["supported_formats"] == ["h5mu"]
        assert info["compression"] == "lzf"
        assert info["compression_opts"] == 3
        assert info["mudata_available"] is True
        assert info["multi_modal"] is True
        assert info["backed_mode_support"] is True
        assert info["backend_type"] == "MuDataBackend"
        assert "mudata_version" in info


@pytest.mark.skipif(not MUDATA_AVAILABLE, reason="MuData not available")
class TestMuDataPerformanceAndMemory:
    """Test performance and memory efficiency."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = MuDataBackend(base_path=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_large_multimodal_file_handling(self):
        """Test handling of reasonably large multi-modal files."""
        # Create moderately large multi-modal dataset
        rna_data = TestMuDataCreation.create_rna_data(n_obs=500, n_vars=200, sparse=True)
        protein_data = TestMuDataCreation.create_protein_data(n_obs=500, n_proteins=50)

        mdata = mudata.MuData({'rna': rna_data, 'protein': protein_data})
        file_path = "test_large_multimodal.h5mu"

        # Measure save time
        start_time = time.time()
        self.backend.save(mdata, file_path)
        save_time = time.time() - start_time

        # Should complete in reasonable time
        assert save_time < 30

        # Measure load time
        start_time = time.time()
        mdata_loaded = self.backend.load(file_path)
        load_time = time.time() - start_time

        assert load_time < 30
        assert mdata_loaded.shape == mdata.shape
        assert len(mdata_loaded.mod) == 2

    def test_memory_efficiency_backed_mode(self):
        """Test memory efficiency of backed mode."""
        # Create larger multi-modal dataset
        rna_data = TestMuDataCreation.create_rna_data(n_obs=300, n_vars=100)
        protein_data = TestMuDataCreation.create_protein_data(n_obs=300, n_proteins=30)

        mdata = mudata.MuData({'rna': rna_data, 'protein': protein_data})
        file_path = "test_memory.h5mu"

        self.backend.save(mdata, file_path)

        # Load in backed mode should use less memory
        mdata_backed = self.backend.load(file_path, backed=True)

        # Basic verification
        assert mdata_backed.isbacked
        assert mdata_backed.shape == mdata.shape

        # Should be able to access modality data
        assert 'rna' in mdata_backed.mod
        assert mdata_backed.mod['rna'].X[0, 0] is not None

    def test_concurrent_multimodal_access_simulation(self):
        """Test simulation of concurrent access to multi-modal data."""
        mdata = TestMuDataCreation.create_simple_mudata()
        file_path = "test_concurrent_multimodal.h5mu"

        self.backend.save(mdata, file_path)

        # Simulate multiple rapid reads
        for i in range(3):
            mdata_loaded = self.backend.load(file_path)
            assert mdata_loaded.shape == mdata.shape
            assert len(mdata_loaded.mod) == len(mdata.mod)

            # Access different modalities
            for mod_name in mdata_loaded.mod.keys():
                mod_data = self.backend.get_modality(mdata_loaded, mod_name)
                assert mod_data.X[0, 0] is not None

            # Small delay to simulate concurrent access patterns
            time.sleep(0.01)


@pytest.mark.skipif(not MUDATA_AVAILABLE, reason="MuData not available")
class TestMuDataGlobalObservations:
    """Test global observation handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.backend = MuDataBackend()

    def test_update_global_obs(self):
        """Test updating global observations."""
        rna_data = TestMuDataCreation.create_rna_data(n_obs=30)
        protein_data = TestMuDataCreation.create_protein_data(n_obs=30)

        mdata = mudata.MuData({'rna': rna_data, 'protein': protein_data})

        # Initially empty global obs
        assert mdata.obs.empty or len(mdata.obs.columns) == 0

        # Update global observations
        self.backend._update_global_obs(mdata)

        # Should now have summary statistics
        expected_columns = {'rna_total_counts', 'rna_n_features', 'protein_total_counts', 'protein_n_features'}
        actual_columns = set(mdata.obs.columns)

        # Should have at least some of the expected columns
        assert len(expected_columns.intersection(actual_columns)) > 0

    def test_global_obs_with_different_observation_names(self):
        """Test global obs handling with different observation names."""
        # Create modalities with different obs names (should trigger warning)
        rna_data = TestMuDataCreation.create_rna_data(n_obs=20)
        protein_data = TestMuDataCreation.create_protein_data(n_obs=20)

        # Change protein obs names to be different
        protein_data.obs_names = [f'protein_cell_{i}' for i in range(20)]

        mdata = mudata.MuData({'rna': rna_data, 'protein': protein_data})

        # This should handle the mismatch gracefully
        self.backend._update_global_obs(mdata)

        # Should still have some global observations
        assert len(mdata.obs.columns) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])