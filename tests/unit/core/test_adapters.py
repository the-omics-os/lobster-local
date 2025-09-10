"""
Comprehensive unit tests for data adapters.

This module provides thorough testing of the adapter system including
TranscriptomicsAdapter and ProteomicsAdapter classes, interface compliance,
data format conversion, validation, and scientific accuracy.

Test coverage target: 95%+ with meaningful tests for biological data processing.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest
import anndata as ad
from pytest_mock import MockerFixture
from scipy import sparse

from lobster.core.adapters.base import BaseAdapter
from lobster.core.adapters.transcriptomics import TranscriptomicsAdapter
from lobster.core.adapters.proteomics import ProteomicsAdapter
from lobster.core.interfaces.adapter import IModalityAdapter
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.transcriptomics import TRANSCRIPTOMICS_SCHEMA
from lobster.core.schemas.proteomics import PROTEOMICS_SCHEMA

from tests.mock_data.factories import (
    SingleCellDataFactory, 
    BulkRNASeqDataFactory, 
    ProteomicsDataFactory
)
from tests.mock_data.base import SMALL_DATASET_CONFIG, MEDIUM_DATASET_CONFIG


# ===============================================================================
# Test Fixtures
# ===============================================================================

@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    return pd.DataFrame({
        'Gene1': [100, 200, 150, 75],
        'Gene2': [50, 120, 80, 90],
        'Gene3': [300, 180, 220, 160],
        'MT-ATP6': [25, 30, 20, 35],  # Mitochondrial gene
        'RPL18': [150, 180, 200, 170]  # Ribosomal gene
    }, index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])


@pytest.fixture
def sample_proteomics_data():
    """Create sample proteomics data for testing."""
    return pd.DataFrame({
        'Protein1': [1000.5, 2000.2, 1500.7, 750.1],
        'Protein2': [500.3, 1200.8, 800.9, 900.4],
        'CON_TRYP_HUMAN': [100.1, 120.5, 110.2, 95.8],  # Contaminant
        'REV_Protein3': [0.0, 0.0, 0.0, 0.0],  # Reverse hit
        'Protein4': [np.nan, 1800.3, 2200.1, 1600.7]  # Missing values
    }, index=['Sample1', 'Sample2', 'Sample3', 'Sample4'])


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing."""
    with patch('pandas.read_csv') as mock_csv, \
         patch('pandas.read_excel') as mock_excel, \
         patch('anndata.read_h5ad') as mock_h5ad, \
         patch('anndata.read_mtx') as mock_mtx:
        yield {
            'csv': mock_csv,
            'excel': mock_excel,
            'h5ad': mock_h5ad,
            'mtx': mock_mtx
        }


# ===============================================================================
# Base Adapter Tests
# ===============================================================================

@pytest.mark.unit
class TestBaseAdapter:
    """Test BaseAdapter functionality."""
    
    def test_initialization(self):
        """Test BaseAdapter initialization."""
        adapter = BaseAdapter()
        
        assert adapter.modality_name == "generic"
        assert adapter.supported_formats == ["csv", "tsv", "xlsx", "h5ad"]
        assert isinstance(adapter.schema, dict)
        
    def test_initialization_with_custom_config(self):
        """Test BaseAdapter with custom configuration."""
        config = {
            "modality_name": "custom",
            "supported_formats": ["csv", "h5ad"],
            "schema": {"custom": "schema"}
        }
        
        adapter = BaseAdapter(config=config)
        
        assert adapter.modality_name == "custom"
        assert adapter.supported_formats == ["csv", "h5ad"]
        assert adapter.schema == {"custom": "schema"}
    
    def test_get_modality_name(self):
        """Test get_modality_name method."""
        adapter = BaseAdapter()
        assert adapter.get_modality_name() == "generic"
    
    def test_get_supported_formats(self):
        """Test get_supported_formats method."""
        adapter = BaseAdapter()
        formats = adapter.get_supported_formats()
        assert "csv" in formats
        assert "h5ad" in formats
    
    def test_get_schema(self):
        """Test get_schema method."""
        adapter = BaseAdapter()
        schema = adapter.get_schema()
        assert isinstance(schema, dict)
    
    def test_load_csv_file(self, mock_file_operations, sample_csv_data):
        """Test loading CSV file."""
        adapter = BaseAdapter()
        mock_file_operations['csv'].return_value = sample_csv_data
        
        result = adapter._load_csv_file("test.csv")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        mock_file_operations['csv'].assert_called_once_with("test.csv", index_col=0)
    
    def test_load_excel_file(self, mock_file_operations, sample_csv_data):
        """Test loading Excel file."""
        adapter = BaseAdapter()
        mock_file_operations['excel'].return_value = sample_csv_data
        
        result = adapter._load_excel_file("test.xlsx")
        
        assert isinstance(result, pd.DataFrame)
        mock_file_operations['excel'].assert_called_once_with("test.xlsx", index_col=0)
    
    def test_load_h5ad_file(self, mock_file_operations):
        """Test loading H5AD file."""
        adapter = BaseAdapter()
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_file_operations['h5ad'].return_value = mock_adata
        
        result = adapter._load_h5ad_file("test.h5ad")
        
        assert isinstance(result, ad.AnnData)
        mock_file_operations['h5ad'].assert_called_once_with("test.h5ad")
    
    def test_load_file_unsupported_format(self):
        """Test loading unsupported file format raises error."""
        adapter = BaseAdapter()
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            adapter._load_file("test.unknown")
    
    def test_convert_dataframe_to_anndata(self, sample_csv_data):
        """Test DataFrame to AnnData conversion."""
        adapter = BaseAdapter()
        
        result = adapter._convert_dataframe_to_anndata(sample_csv_data)
        
        assert isinstance(result, ad.AnnData)
        assert result.shape == (4, 5)  # 4 cells, 5 genes
        assert list(result.obs_names) == ['Cell1', 'Cell2', 'Cell3', 'Cell4']
        assert 'Gene1' in result.var_names
    
    def test_convert_sparse_dataframe_to_anndata(self):
        """Test sparse DataFrame conversion.""" 
        adapter = BaseAdapter()
        
        # Create sparse data
        dense_data = np.random.randint(0, 10, (100, 1000))
        dense_data[dense_data < 8] = 0  # Make it sparse
        sparse_df = pd.DataFrame(
            dense_data,
            index=[f"Cell_{i}" for i in range(100)],
            columns=[f"Gene_{i}" for i in range(1000)]
        )
        
        result = adapter._convert_dataframe_to_anndata(sparse_df)
        
        assert isinstance(result, ad.AnnData)
        assert result.shape == (100, 1000)
        # Should convert to sparse matrix for memory efficiency
        assert sparse.issparse(result.X)
    
    def test_basic_validation_success(self):
        """Test basic validation with valid data."""
        adapter = BaseAdapter()
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        result = adapter._validate_basic_structure(test_data)
        
        assert isinstance(result, ValidationResult)
        assert not result.has_errors
    
    def test_basic_validation_empty_data(self):
        """Test basic validation with empty data."""
        adapter = BaseAdapter()
        
        # Create empty AnnData
        empty_data = ad.AnnData(X=np.array([]).reshape(0, 0))
        
        result = adapter._validate_basic_structure(empty_data)
        
        assert result.has_errors
        assert any("empty" in error.lower() for error in result.errors)
    
    def test_from_source_dataframe(self, sample_csv_data):
        """Test from_source with DataFrame input."""
        adapter = BaseAdapter()
        
        result = adapter.from_source(sample_csv_data)
        
        assert isinstance(result, ad.AnnData)
        assert result.shape == (4, 5)
    
    def test_from_source_anndata(self):
        """Test from_source with AnnData input."""
        adapter = BaseAdapter()
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        result = adapter.from_source(test_data)
        
        assert result is test_data  # Should return same object
    
    def test_from_source_file_path(self, mock_file_operations, sample_csv_data):
        """Test from_source with file path."""
        adapter = BaseAdapter()
        mock_file_operations['csv'].return_value = sample_csv_data
        
        result = adapter.from_source("test.csv")
        
        assert isinstance(result, ad.AnnData)
        mock_file_operations['csv'].assert_called_once()
    
    def test_from_source_invalid_input(self):
        """Test from_source with invalid input type."""
        adapter = BaseAdapter()
        
        with pytest.raises(ValueError, match="Unsupported data source type"):
            adapter.from_source(12345)  # Invalid type
    
    def test_validate_method(self):
        """Test validate method."""
        adapter = BaseAdapter()
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        result = adapter.validate(test_data)
        
        assert isinstance(result, ValidationResult)


# ===============================================================================
# Interface Compliance Tests
# ===============================================================================

@pytest.mark.unit
class TestInterfaceCompliance:
    """Test that adapters properly implement IModalityAdapter interface."""
    
    def test_transcriptomics_adapter_interface_compliance(self):
        """Test TranscriptomicsAdapter implements IModalityAdapter."""
        adapter = TranscriptomicsAdapter(modality_type="single_cell")
        
        # Test that adapter is instance of interface
        assert isinstance(adapter, IModalityAdapter)
        
        # Test all abstract methods are implemented
        assert hasattr(adapter, 'from_source')
        assert hasattr(adapter, 'validate')
        assert hasattr(adapter, 'get_schema')
        assert hasattr(adapter, 'get_supported_formats')
        assert callable(adapter.from_source)
        assert callable(adapter.validate)
        assert callable(adapter.get_schema)
        assert callable(adapter.get_supported_formats)
    
    def test_proteomics_adapter_interface_compliance(self):
        """Test ProteomicsAdapter implements IModalityAdapter."""
        adapter = ProteomicsAdapter(proteomics_type="mass_spectrometry")
        
        # Test that adapter is instance of interface
        assert isinstance(adapter, IModalityAdapter)
        
        # Test all abstract methods are implemented
        assert hasattr(adapter, 'from_source')
        assert hasattr(adapter, 'validate')
        assert hasattr(adapter, 'get_schema')
        assert hasattr(adapter, 'get_supported_formats')
        assert callable(adapter.from_source)
        assert callable(adapter.validate)
        assert callable(adapter.get_schema)
        assert callable(adapter.get_supported_formats)
    
    @pytest.mark.parametrize("adapter_class,init_kwargs", [
        (TranscriptomicsAdapter, {"modality_type": "single_cell"}),
        (TranscriptomicsAdapter, {"modality_type": "bulk"}),
        (ProteomicsAdapter, {"proteomics_type": "mass_spectrometry"}),
        (ProteomicsAdapter, {"proteomics_type": "affinity"})
    ])
    def test_adapter_method_signatures(self, adapter_class, init_kwargs):
        """Test that adapter methods have correct signatures."""
        adapter = adapter_class(**init_kwargs)
        
        # Test from_source method
        try:
            # Should accept various source types
            test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            result = adapter.from_source(test_data)
            assert isinstance(result, ad.AnnData)
        except Exception as e:
            pytest.fail(f"from_source method failed: {e}")
        
        # Test validate method
        try:
            result = adapter.validate(test_data)
            assert isinstance(result, ValidationResult)
        except Exception as e:
            pytest.fail(f"validate method failed: {e}")
        
        # Test get_schema method
        try:
            schema = adapter.get_schema()
            assert isinstance(schema, dict)
        except Exception as e:
            pytest.fail(f"get_schema method failed: {e}")
        
        # Test get_supported_formats method
        try:
            formats = adapter.get_supported_formats()
            assert isinstance(formats, list)
            assert len(formats) > 0
        except Exception as e:
            pytest.fail(f"get_supported_formats method failed: {e}")


# ===============================================================================
# Transcriptomics Adapter Tests
# ===============================================================================

@pytest.mark.unit
class TestTranscriptomicsAdapter:
    """Test TranscriptomicsAdapter functionality."""
    
    def test_initialization_single_cell(self):
        """Test TranscriptomicsAdapter initialization for single-cell."""
        adapter = TranscriptomicsAdapter(modality_type="single_cell")
        
        assert adapter.modality_type == "single_cell"
        assert adapter.modality_name == "transcriptomics_single_cell"
        assert adapter.schema == TRANSCRIPTOMICS_SCHEMA
    
    def test_initialization_bulk(self):
        """Test TranscriptomicsAdapter initialization for bulk RNA-seq."""
        adapter = TranscriptomicsAdapter(modality_type="bulk")
        
        assert adapter.modality_type == "bulk"
        assert adapter.modality_name == "transcriptomics_bulk"
        assert adapter.schema == TRANSCRIPTOMICS_SCHEMA
    
    def test_initialization_invalid_type(self):
        """Test initialization with invalid modality type."""
        with pytest.raises(ValueError, match="Invalid modality_type"):
            TranscriptomicsAdapter(modality_type="invalid")
    
    def test_detect_data_type_single_cell(self):
        """Test automatic single-cell detection."""
        adapter = TranscriptomicsAdapter()
        
        # High gene count suggests single-cell
        sc_data = pd.DataFrame(
            np.random.randint(0, 100, (100, 15000)),  # 15k genes
            index=[f"Cell_{i}" for i in range(100)],
            columns=[f"Gene_{i}" for i in range(15000)]
        )
        
        detected_type = adapter._detect_data_type(sc_data)
        assert detected_type == "single_cell"
    
    def test_detect_data_type_bulk(self):
        """Test automatic bulk RNA-seq detection."""
        adapter = TranscriptomicsAdapter()
        
        # Lower gene count suggests bulk
        bulk_data = pd.DataFrame(
            np.random.randint(0, 1000, (24, 2000)),  # 2k genes
            index=[f"Sample_{i}" for i in range(24)],
            columns=[f"Gene_{i}" for i in range(2000)]
        )
        
        detected_type = adapter._detect_data_type(bulk_data)
        assert detected_type == "bulk"
    
    def test_preprocess_single_cell_data(self, sample_csv_data):
        """Test preprocessing for single-cell data."""
        adapter = TranscriptomicsAdapter(modality_type="single_cell")
        adata = ad.AnnData(sample_csv_data.T)  # Transpose for genes as variables
        
        processed = adapter._preprocess_data(adata)
        
        # Should add basic QC metrics
        assert "total_counts" in processed.obs.columns
        assert "n_genes_by_counts" in processed.obs.columns
        assert "pct_counts_mt" in processed.obs.columns
        
        # Should flag mitochondrial and ribosomal genes
        assert "mt" in processed.var.columns
        assert "ribo" in processed.var.columns
        assert processed.var.loc["MT-ATP6", "mt"] == True
        assert processed.var.loc["RPL18", "ribo"] == True
    
    def test_preprocess_bulk_data(self, sample_csv_data):
        """Test preprocessing for bulk RNA-seq data."""
        adapter = TranscriptomicsAdapter(modality_type="bulk")
        adata = ad.AnnData(sample_csv_data.T)
        
        processed = adapter._preprocess_data(adata)
        
        # Should add basic metrics
        assert "total_counts" in processed.obs.columns
        assert "n_genes_by_counts" in processed.obs.columns
        
        # Should handle gene annotations
        assert "mt" in processed.var.columns
        assert "ribo" in processed.var.columns
    
    def test_calculate_mitochondrial_percentage(self):
        """Test mitochondrial gene percentage calculation."""
        adapter = TranscriptomicsAdapter()
        
        # Create data with known MT gene expression
        data = pd.DataFrame({
            'Gene1': [100, 200],
            'MT-ATP6': [20, 40],  # 20% and 20% of total
            'Gene2': [80, 160]
        }).T
        adata = ad.AnnData(data)
        
        # Flag MT genes
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        
        # Calculate percentages
        adapter._add_qc_metrics(adata)
        
        # Should be exactly 10% for both cells (20/200 and 40/400)
        np.testing.assert_array_almost_equal(
            adata.obs['pct_counts_mt'].values,
            [10.0, 10.0],
            decimal=1
        )
    
    def test_identify_mitochondrial_genes(self):
        """Test mitochondrial gene identification."""
        adapter = TranscriptomicsAdapter()
        
        gene_names = pd.Index([
            'Gene1', 'MT-ATP6', 'mt-CO1', 'MT_ND1', 
            'MTRNR1', 'Gene2', 'RPL18'
        ])
        
        mt_mask = adapter._identify_mitochondrial_genes(gene_names)
        
        # Should identify various MT gene naming conventions
        expected = [False, True, True, True, True, False, False]
        assert mt_mask.tolist() == expected
    
    def test_identify_ribosomal_genes(self):
        """Test ribosomal gene identification."""
        adapter = TranscriptomicsAdapter()
        
        gene_names = pd.Index([
            'Gene1', 'RPL18', 'RPS6', 'rpl19', 'rps5',
            'MRPL1', 'MRPS1', 'Gene2'
        ])
        
        ribo_mask = adapter._identify_ribosomal_genes(gene_names)
        
        # Should identify ribosomal and mitochondrial ribosomal genes
        expected = [False, True, True, True, True, True, True, False]
        assert ribo_mask.tolist() == expected
    
    def test_from_source_csv(self, mock_file_operations, sample_csv_data):
        """Test loading from CSV file."""
        adapter = TranscriptomicsAdapter(modality_type="single_cell")
        mock_file_operations['csv'].return_value = sample_csv_data
        
        result = adapter.from_source("test.csv")
        
        assert isinstance(result, ad.AnnData)
        # Should have QC metrics added
        assert "total_counts" in result.obs.columns
        assert "pct_counts_mt" in result.obs.columns
    
    def test_from_source_h5ad(self, mock_file_operations):
        """Test loading from H5AD file.""" 
        adapter = TranscriptomicsAdapter(modality_type="single_cell")
        mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_file_operations['h5ad'].return_value = mock_adata
        
        result = adapter.from_source("test.h5ad")
        
        assert isinstance(result, ad.AnnData)
    
    def test_validation_success(self):
        """Test validation with compliant data."""
        adapter = TranscriptomicsAdapter(modality_type="single_cell")
        
        # Create compliant data
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        result = adapter.validate(test_data)
        
        assert isinstance(result, ValidationResult)
        assert not result.has_errors
    
    def test_validation_missing_required_obs(self):
        """Test validation with missing required observation metadata."""
        adapter = TranscriptomicsAdapter(modality_type="single_cell")
        
        # Create data missing required obs columns
        test_data = ad.AnnData(X=np.random.randint(0, 100, (50, 100)))
        # Missing total_counts, n_genes_by_counts
        
        result = adapter.validate(test_data, strict=True)
        
        assert result.has_errors
        assert any("total_counts" in error for error in result.errors)
    
    def test_validation_warnings_mode(self):
        """Test validation in warnings mode (non-strict)."""
        adapter = TranscriptomicsAdapter(modality_type="single_cell")
        
        # Create data with some issues but not critical
        test_data = ad.AnnData(X=np.random.randint(0, 100, (50, 100)))
        
        result = adapter.validate(test_data, strict=False)
        
        # Should have warnings but not errors
        assert not result.has_errors or result.has_warnings
    
    def test_get_quality_metrics(self):
        """Test quality metrics extraction."""
        adapter = TranscriptomicsAdapter(modality_type="single_cell")
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        metrics = adapter.get_quality_metrics(test_data)
        
        assert isinstance(metrics, dict)
        assert "n_cells" in metrics
        assert "n_genes" in metrics
        assert "mean_genes_per_cell" in metrics
        assert "mean_counts_per_cell" in metrics
        assert metrics["n_cells"] == test_data.n_obs
        assert metrics["n_genes"] == test_data.n_vars


# ===============================================================================
# Proteomics Adapter Tests
# ===============================================================================

@pytest.mark.unit
class TestProteomicsAdapter:
    """Test ProteomicsAdapter functionality."""
    
    def test_initialization_mass_spectrometry(self):
        """Test ProteomicsAdapter initialization for mass spectrometry."""
        adapter = ProteomicsAdapter(proteomics_type="mass_spectrometry")
        
        assert adapter.proteomics_type == "mass_spectrometry"
        assert adapter.modality_name == "proteomics_ms"
        assert adapter.schema == PROTEOMICS_SCHEMA
    
    def test_initialization_affinity(self):
        """Test ProteomicsAdapter initialization for affinity proteomics."""
        adapter = ProteomicsAdapter(proteomics_type="affinity")
        
        assert adapter.proteomics_type == "affinity"
        assert adapter.modality_name == "proteomics_affinity"
        assert adapter.schema == PROTEOMICS_SCHEMA
    
    def test_initialization_invalid_type(self):
        """Test initialization with invalid proteomics type."""
        with pytest.raises(ValueError, match="Invalid proteomics_type"):
            ProteomicsAdapter(proteomics_type="invalid")
    
    def test_detect_proteomics_type_mass_spec(self, sample_proteomics_data):
        """Test automatic mass spectrometry detection."""
        adapter = ProteomicsAdapter()
        
        # Add MS-specific characteristics
        ms_data = sample_proteomics_data.copy()
        ms_data.columns = ['P12345', 'Q67890', 'CON_TRYP', 'REV_P11111', 'O98765']
        
        detected_type = adapter._detect_proteomics_type(ms_data)
        assert detected_type == "mass_spectrometry"
    
    def test_detect_proteomics_type_affinity(self):
        """Test automatic affinity proteomics detection."""
        adapter = ProteomicsAdapter()
        
        # Create affinity-like data (clean protein names, no contaminants)
        affinity_data = pd.DataFrame({
            'TP53': [1000, 2000, 1500],
            'BRCA1': [500, 800, 600],
            'MYC': [1200, 1800, 1400],
            'EGFR': [800, 1200, 900]
        })
        
        detected_type = adapter._detect_proteomics_type(affinity_data)
        assert detected_type == "affinity"
    
    def test_preprocess_mass_spec_data(self, sample_proteomics_data):
        """Test preprocessing for mass spectrometry data."""
        adapter = ProteomicsAdapter(proteomics_type="mass_spectrometry")
        adata = ad.AnnData(sample_proteomics_data.T)
        
        processed = adapter._preprocess_data(adata)
        
        # Should add proteomics QC metrics
        assert "total_protein_intensity" in processed.obs.columns
        assert "n_proteins_detected" in processed.obs.columns
        assert "missing_value_pct" in processed.obs.columns
        
        # Should flag contaminants and reverse hits
        assert "contaminant" in processed.var.columns
        assert "reverse" in processed.var.columns
        assert processed.var.loc["CON_TRYP_HUMAN", "contaminant"] == True
        assert processed.var.loc["REV_Protein3", "reverse"] == True
    
    def test_preprocess_affinity_data(self):
        """Test preprocessing for affinity proteomics data."""
        adapter = ProteomicsAdapter(proteomics_type="affinity")
        
        # Create clean affinity data
        affinity_df = pd.DataFrame({
            'TP53': [1000.0, 2000.0, 1500.0],
            'BRCA1': [500.0, 800.0, 600.0],
            'MYC': [1200.0, 1800.0, 1400.0]
        })
        adata = ad.AnnData(affinity_df.T)
        
        processed = adapter._preprocess_data(adata)
        
        # Should add basic metrics
        assert "total_protein_intensity" in processed.obs.columns
        assert "n_proteins_detected" in processed.obs.columns
    
    def test_handle_missing_values_knn(self, sample_proteomics_data):
        """Test KNN imputation for missing values."""
        adapter = ProteomicsAdapter()
        adata = ad.AnnData(sample_proteomics_data.T)
        
        # Add more missing values
        adata.X[0, :2] = np.nan
        
        processed = adapter._handle_missing_values(adata, strategy="knn")
        
        # Should have no NaN values after imputation
        assert not np.isnan(processed.X).any()
        # Should store imputation info
        assert "imputation_method" in processed.uns
        assert processed.uns["imputation_method"] == "knn"
    
    def test_handle_missing_values_median(self, sample_proteomics_data):
        """Test median imputation for missing values."""
        adapter = ProteomicsAdapter()
        adata = ad.AnnData(sample_proteomics_data.T)
        
        # Ensure we have missing values
        original_nan_count = np.isnan(adata.X).sum()
        
        processed = adapter._handle_missing_values(adata, strategy="median")
        
        # Should have fewer or no NaN values
        final_nan_count = np.isnan(processed.X).sum()
        assert final_nan_count <= original_nan_count
    
    def test_handle_missing_values_invalid_strategy(self):
        """Test handling missing values with invalid strategy."""
        adapter = ProteomicsAdapter()
        adata = ad.AnnData(np.array([[1, 2], [3, np.nan]]))
        
        with pytest.raises(ValueError, match="Invalid missing value strategy"):
            adapter._handle_missing_values(adata, strategy="invalid")
    
    def test_identify_contaminants(self):
        """Test contaminant protein identification."""
        adapter = ProteomicsAdapter()
        
        protein_names = pd.Index([
            'P12345', 'CON_TRYP_HUMAN', 'Q67890', 'CONT_KERATIN', 
            'REV_P11111', 'O98765', 'CON_ALBU_HUMAN'
        ])
        
        contam_mask = adapter._identify_contaminants(protein_names)
        
        # Should identify various contaminant patterns
        expected = [False, True, False, True, False, False, True]
        assert contam_mask.tolist() == expected
    
    def test_identify_reverse_hits(self):
        """Test reverse hit identification."""
        adapter = ProteomicsAdapter()
        
        protein_names = pd.Index([
            'P12345', 'REV_P67890', 'Q11111', 'REV_CONT_TRYP',
            'REVERSE_O98765', 'N22222'
        ])
        
        reverse_mask = adapter._identify_reverse_hits(protein_names)
        
        # Should identify reverse hit patterns
        expected = [False, True, False, True, True, False]
        assert reverse_mask.tolist() == expected
    
    def test_calculate_cv_values(self):
        """Test coefficient of variation calculation."""
        adapter = ProteomicsAdapter()
        
        # Create data with known CV
        # Values: [100, 200] -> mean=150, std=70.71, CV=47.14%
        data = np.array([[100.0, 200.0], [150.0, 150.0]])
        adata = ad.AnnData(data)
        
        cv_values = adapter._calculate_cv_values(adata.X.T)  # Transpose for samples as rows
        
        # First protein should have CV ≈ 47.14%
        assert abs(cv_values[0] - 47.14) < 1.0
        # Second protein should have CV = 0% (no variation)
        assert abs(cv_values[1] - 0.0) < 0.1
    
    def test_add_protein_metadata(self):
        """Test adding protein metadata."""
        adapter = ProteomicsAdapter()
        
        protein_names = pd.Index([
            'P12345_HUMAN', 'Q67890_MOUSE', 'CON_TRYP_HUMAN'
        ])
        adata = ad.AnnData(X=np.random.randn(3, 3), var=pd.DataFrame(index=protein_names))
        
        adapter._add_protein_metadata(adata)
        
        # Should extract organisms
        assert "organism" in adata.var.columns
        assert adata.var.loc['P12345_HUMAN', 'organism'] == 'HUMAN'
        assert adata.var.loc['Q67890_MOUSE', 'organism'] == 'MOUSE'
    
    def test_from_source_csv(self, mock_file_operations, sample_proteomics_data):
        """Test loading proteomics data from CSV."""
        adapter = ProteomicsAdapter(proteomics_type="mass_spectrometry")
        mock_file_operations['csv'].return_value = sample_proteomics_data
        
        result = adapter.from_source("test.csv")
        
        assert isinstance(result, ad.AnnData)
        # Should have proteomics QC metrics
        assert "total_protein_intensity" in result.obs.columns
        assert "missing_value_pct" in result.obs.columns
    
    def test_validation_success(self):
        """Test validation with compliant proteomics data."""
        adapter = ProteomicsAdapter(proteomics_type="mass_spectrometry")
        
        test_data = ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)
        
        result = adapter.validate(test_data)
        
        assert isinstance(result, ValidationResult)
        assert not result.has_errors
    
    def test_validation_missing_required_var(self):
        """Test validation with missing required variable metadata."""
        adapter = ProteomicsAdapter(proteomics_type="mass_spectrometry")
        
        # Create data missing protein_ids
        test_data = ad.AnnData(X=np.random.randn(20, 100))
        # Missing protein_ids in var
        
        result = adapter.validate(test_data, strict=True)
        
        assert result.has_errors
        assert any("protein_ids" in error for error in result.errors)
    
    def test_get_quality_metrics(self):
        """Test proteomics quality metrics extraction."""
        adapter = ProteomicsAdapter(proteomics_type="mass_spectrometry")
        test_data = ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)
        
        metrics = adapter.get_quality_metrics(test_data)
        
        assert isinstance(metrics, dict)
        assert "n_samples" in metrics
        assert "n_proteins" in metrics
        assert "mean_proteins_per_sample" in metrics
        assert "median_intensity" in metrics
        assert metrics["n_samples"] == test_data.n_obs
        assert metrics["n_proteins"] == test_data.n_vars


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestErrorHandlingEdgeCases:
    """Test error handling and edge case scenarios."""
    
    def test_file_loading_failure(self, mock_file_operations):
        """Test handling of file loading failures."""
        adapter = TranscriptomicsAdapter()
        mock_file_operations['csv'].side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            adapter.from_source("nonexistent.csv")
    
    def test_empty_dataset_handling(self):
        """Test handling of completely empty datasets."""
        adapter = TranscriptomicsAdapter()
        
        # Create empty DataFrame
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty"):
            adapter.from_source(empty_df)
    
    def test_invalid_dimensions(self):
        """Test handling of invalid data dimensions."""
        adapter = ProteomicsAdapter()
        
        # Create data with invalid dimensions (single cell, single protein)
        invalid_data = pd.DataFrame({'Protein1': [100]})  # 1x1 matrix
        
        result = adapter.from_source(invalid_data)
        validation_result = adapter.validate(result, strict=True)
        
        # Should create warnings about insufficient data
        assert validation_result.has_warnings or validation_result.has_errors
    
    def test_mtx_format_missing_files(self, mock_file_operations):
        """Test MTX format when required files are missing."""
        adapter = TranscriptomicsAdapter()
        
        # Mock MTX loading to raise specific error
        mock_file_operations['mtx'].side_effect = FileNotFoundError("Missing barcodes.tsv")
        
        with pytest.raises(FileNotFoundError):
            adapter._load_mtx_format("matrix.mtx")
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted/invalid data."""
        adapter = TranscriptomicsAdapter()
        
        # Create data with invalid values
        corrupted_df = pd.DataFrame({
            'Gene1': [np.inf, -np.inf, np.nan, 100],
            'Gene2': [50, 200, 150, np.inf]
        })
        
        # Should handle gracefully
        result = adapter.from_source(corrupted_df)
        
        # Should replace inf values
        assert not np.isinf(result.X).any()
    
    def test_memory_efficient_processing(self):
        """Test memory efficiency with large datasets."""
        adapter = TranscriptomicsAdapter()
        
        # Create large sparse dataset
        n_cells, n_genes = 1000, 5000
        sparse_data = pd.DataFrame(
            np.random.choice([0, 1, 2, 5], size=(n_cells, n_genes), p=[0.8, 0.1, 0.05, 0.05]),
            index=[f"Cell_{i}" for i in range(n_cells)],
            columns=[f"Gene_{i}" for i in range(n_genes)]
        )
        
        result = adapter.from_source(sparse_data)
        
        # Should convert to sparse format for memory efficiency
        assert sparse.issparse(result.X)
        assert result.shape == (n_cells, n_genes)
    
    def test_special_character_handling(self):
        """Test handling of special characters in gene/protein names."""
        adapter = TranscriptomicsAdapter()
        
        # Create data with special characters
        special_df = pd.DataFrame({
            'Gene-1': [100, 200],
            'Gene.2': [150, 180],
            'Gene_3': [120, 160],
            'Gene(4)': [90, 110],
            'Gene[5]': [200, 250]
        })
        
        result = adapter.from_source(special_df)
        
        # Should handle all special characters
        assert result.n_vars == 5
        assert 'Gene-1' in result.var_names
        assert 'Gene.2' in result.var_names


# ===============================================================================
# Performance and Benchmarking Tests
# ===============================================================================

@pytest.mark.benchmark
class TestPerformanceBenchmarking:
    """Test performance characteristics of adapters."""
    
def test_csv_loading_performance(self, benchmark, mock_file_operations):
        """Benchmark CSV file loading performance."""
        adapter = TranscriptomicsAdapter()
        
        # Create large dataset
        large_df = pd.DataFrame(
            np.random.randint(0, 1000, (1000, 2000)),
            index=[f"Cell_{i}" for i in range(1000)],
            columns=[f"Gene_{i}" for i in range(2000)]
        )
        mock_file_operations['csv'].return_value = large_df
        
        result = benchmark(adapter.from_source, "large_dataset.csv")
        
        assert isinstance(result, ad.AnnData)
        assert result.shape == (1000, 2000)
    
    def test_preprocessing_performance(self, benchmark):
        """Benchmark data preprocessing performance."""
        adapter = TranscriptomicsAdapter(modality_type="single_cell")
        
        # Create realistic single-cell dataset
        sc_data = SingleCellDataFactory(config=MEDIUM_DATASET_CONFIG)
        
        result = benchmark(adapter._preprocess_data, sc_data)
        
        assert "total_counts" in result.obs.columns
        assert "pct_counts_mt" in result.obs.columns
    
    def test_validation_performance(self, benchmark):
        """Benchmark data validation performance."""
        adapter = ProteomicsAdapter(proteomics_type="mass_spectrometry")
        
        # Create large proteomics dataset
        large_proteomics = ProteomicsDataFactory(config=MEDIUM_DATASET_CONFIG)
        
        result = benchmark(adapter.validate, large_proteomics)
        
        assert isinstance(result, ValidationResult)


# ===============================================================================
# Integration Test Scenarios
# ===============================================================================

@pytest.mark.unit
class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""
    
    def test_complete_single_cell_workflow(self, mock_file_operations):
        """Test complete single-cell analysis workflow."""
        adapter = TranscriptomicsAdapter(modality_type="single_cell")
        
        # Create realistic single-cell CSV data
        sc_csv = pd.DataFrame(
            np.random.negative_binomial(5, 0.3, (500, 3000)),  # Realistic count distribution
            index=[f"Cell_{i}" for i in range(500)],
            columns=[f"Gene_{i}" for i in range(2800)] + 
                   [f"MT-{i}" for i in range(100)] +  # Mitochondrial genes
                   [f"RPL{i}" for i in range(50)] +   # Ribosomal genes
                   [f"RPS{i}" for i in range(50)]     # Ribosomal genes
        )
        mock_file_operations['csv'].return_value = sc_csv
        
        # Load and process
        result = adapter.from_source("single_cell_data.csv")
        
        # Validate workflow results
        assert isinstance(result, ad.AnnData)
        assert result.shape == (500, 3000)
        
        # Check QC metrics were calculated
        assert "total_counts" in result.obs.columns
        assert "n_genes_by_counts" in result.obs.columns  
        assert "pct_counts_mt" in result.obs.columns
        
        # Check gene annotations
        assert "mt" in result.var.columns
        assert "ribo" in result.var.columns
        assert result.var["mt"].sum() == 100  # 100 MT genes
        assert result.var["ribo"].sum() == 100  # 100 ribosomal genes
        
        # Validate the processed data
        validation_result = adapter.validate(result)
        assert not validation_result.has_errors
    
    def test_complete_proteomics_workflow(self, mock_file_operations):
        """Test complete proteomics analysis workflow."""
        adapter = ProteomicsAdapter(proteomics_type="mass_spectrometry")
        
        # Create realistic MS proteomics data
        protein_names = (
            [f"P{i:05d}" for i in range(1500)] +  # Regular proteins
            ["CON_TRYP_HUMAN", "CON_KERATIN", "CON_ALBU_HUMAN"] +  # Contaminants
            [f"REV_P{i:05d}" for i in range(50)]  # Reverse hits
        )
        
        ms_data = pd.DataFrame(
            np.random.lognormal(5, 2, (48, len(protein_names))),  # Log-normal intensities
            index=[f"Sample_{i}" for i in range(48)],
            columns=protein_names
        )
        
        # Add some missing values (realistic for MS data)
        missing_mask = np.random.choice([True, False], size=ms_data.shape, p=[0.15, 0.85])
        ms_data[missing_mask] = np.nan
        
        mock_file_operations['csv'].return_value = ms_data
        
        # Load and process
        result = adapter.from_source("proteomics_data.csv", missing_value_strategy="median")
        
        # Validate workflow results
        assert isinstance(result, ad.AnnData)
        assert result.shape == (48, len(protein_names))
        
        # Check proteomics QC metrics
        assert "total_protein_intensity" in result.obs.columns
        assert "n_proteins_detected" in result.obs.columns
        assert "missing_value_pct" in result.obs.columns
        
        # Check protein annotations
        assert "contaminant" in result.var.columns
        assert "reverse" in result.var.columns
        assert result.var["contaminant"].sum() == 3  # 3 contaminants
        assert result.var["reverse"].sum() == 50     # 50 reverse hits
        
        # Check missing value handling
        assert "imputation_method" in result.uns
        
        # Validate the processed data
        validation_result = adapter.validate(result)
        assert not validation_result.has_errors
    
    def test_multi_format_compatibility(self, mock_file_operations):
        """Test that same data can be loaded from different formats."""
        adapter = TranscriptomicsAdapter()
        
        # Create test data
        test_df = pd.DataFrame(
            np.random.randint(0, 100, (20, 50)),
            index=[f"Cell_{i}" for i in range(20)],
            columns=[f"Gene_{i}" for i in range(50)]
        )
        
        # Mock different file formats to return same data
        mock_file_operations['csv'].return_value = test_df
        mock_file_operations['excel'].return_value = test_df
        
        # Load from different formats
        result_csv = adapter.from_source("test.csv")
        result_excel = adapter.from_source("test.xlsx")
        
        # Should produce equivalent results (allowing for processing differences)
        assert result_csv.shape == result_excel.shape
        assert set(result_csv.obs.columns) == set(result_excel.obs.columns)
        assert set(result_csv.var.columns) == set(result_excel.var.columns)
    
    def test_error_recovery_workflow(self):
        """Test error recovery in complex workflows."""
        adapter = TranscriptomicsAdapter()
        
        # Start with valid data
        valid_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        # Process successfully
        processed = adapter._preprocess_data(valid_data.copy())
        validation_result = adapter.validate(processed)
        assert not validation_result.has_errors
        
        # Try to process invalid data (should fail gracefully)
        invalid_data = ad.AnnData(X=np.array([]).reshape(0, 0))
        
        validation_result = adapter.validate(invalid_data)
        assert validation_result.has_errors
        
        # Original valid data should still be processable
        reprocessed = adapter._preprocess_data(valid_data.copy())
        assert isinstance(reprocessed, ad.AnnData)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])