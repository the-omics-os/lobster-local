"""
Unit tests for pseudobulk adapter.

Tests the PseudobulkAdapter class functionality including data loading,
validation, preprocessing, and quality metrics calculation.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from lobster.core.adapters.pseudobulk_adapter import PseudobulkAdapter
from lobster.core.schemas.pseudobulk import PseudobulkSchema
from lobster.core import AdapterError, ValidationError
from lobster.core.interfaces.validator import ValidationResult


@pytest.fixture
def valid_pseudobulk_adata():
    """Create valid pseudobulk AnnData for testing."""
    sample_ids = ['Sample_A', 'Sample_A', 'Sample_A', 'Sample_B', 'Sample_B', 'Sample_B']
    cell_types = ['T_cell', 'B_cell', 'Monocyte', 'T_cell', 'B_cell', 'Monocyte']
    pseudobulk_ids = [f"{s}_{c}" for s, c in zip(sample_ids, cell_types)]
    
    expression_matrix = np.random.negative_binomial(10, 0.3, (6, 100)).astype(float)
    gene_names = [f"GENE_{i:03d}" for i in range(100)]
    
    adata = ad.AnnData(
        X=expression_matrix,
        obs=pd.DataFrame({
            'sample_id': sample_ids,
            'cell_type': cell_types,
            'n_cells_aggregated': [245, 89, 156, 198, 72, 134],
            'condition': ['Control', 'Control', 'Control', 'Treatment', 'Treatment', 'Treatment']
        }, index=pseudobulk_ids),
        var=pd.DataFrame({
            'gene_symbol': [f"Gene_{i}" for i in range(100)],
            'gene_id': [f"ENSG0000{i:05d}" for i in range(100)]
        }, index=gene_names)
    )
    
    return adata


@pytest.fixture
def pseudobulk_dataframe():
    """Create DataFrame in pseudobulk format for testing."""
    sample_ids = ['Sample_A_T_cell', 'Sample_A_B_cell', 'Sample_B_T_cell', 'Sample_B_B_cell']
    gene_names = [f"GENE_{i:03d}" for i in range(50)]
    
    df = pd.DataFrame(
        np.random.negative_binomial(5, 0.3, (4, 50)),
        index=sample_ids,
        columns=gene_names
    )
    
    return df


@pytest.fixture
def pseudobulk_adapter():
    """Create PseudobulkAdapter for testing."""
    return PseudobulkAdapter(strict_validation=False)


@pytest.fixture
def strict_pseudobulk_adapter():
    """Create strict PseudobulkAdapter for testing."""
    return PseudobulkAdapter(strict_validation=True)


@pytest.mark.unit
class TestPseudobulkAdapterInit:
    """Test PseudobulkAdapter initialization."""
    
    def test_adapter_initialization_default(self):
        """Test adapter initialization with default settings."""
        adapter = PseudobulkAdapter()
        
        assert adapter.name == "PseudobulkAdapter"
        assert adapter.strict_validation is False
        assert adapter.validator is not None
        assert adapter.qc_thresholds is not None
    
    def test_adapter_initialization_strict(self):
        """Test adapter initialization with strict validation."""
        adapter = PseudobulkAdapter(strict_validation=True)
        
        assert adapter.strict_validation is True
        assert adapter.validator is not None
    
    def test_adapter_qc_thresholds(self, pseudobulk_adapter):
        """Test QC thresholds are properly loaded."""
        thresholds = pseudobulk_adapter.qc_thresholds
        
        assert 'min_cells_per_pseudobulk' in thresholds
        assert 'min_pseudobulk_samples' in thresholds
        assert thresholds['min_cells_per_pseudobulk'] == 10
        assert thresholds['min_pseudobulk_samples'] == 3


@pytest.mark.unit
class TestPseudobulkAdapterFromSource:
    """Test PseudobulkAdapter.from_source method."""
    
    def test_from_source_anndata(self, pseudobulk_adapter, valid_pseudobulk_adata):
        """Test loading from AnnData object."""
        result = pseudobulk_adapter.from_source(
            valid_pseudobulk_adata,
            validate_schema=False
        )
        
        assert isinstance(result, ad.AnnData)
        assert result.n_obs == valid_pseudobulk_adata.n_obs
        assert result.n_vars == valid_pseudobulk_adata.n_vars
    
    def test_from_source_dataframe(self, pseudobulk_adapter, pseudobulk_dataframe):
        """Test loading from DataFrame."""
        result = pseudobulk_adapter.from_source(pseudobulk_dataframe)
        
        assert isinstance(result, ad.AnnData)
        assert result.n_obs == len(pseudobulk_dataframe)
        assert result.n_vars == len(pseudobulk_dataframe.columns)
        
        # Check parsed sample identifiers
        assert 'sample_id' in result.obs.columns
        assert 'cell_type' in result.obs.columns
        assert result.obs.loc['Sample_A_T_cell', 'sample_id'] == 'Sample_A'
        assert result.obs.loc['Sample_A_T_cell', 'cell_type'] == 'T_cell'
    
    def test_from_source_with_metadata(self, pseudobulk_adapter, pseudobulk_dataframe):
        """Test loading with additional metadata."""
        aggregation_metadata = {
            'pseudobulk_params': {
                'sample_col': 'sample_id',
                'celltype_col': 'cell_type',
                'aggregation_method': 'sum',
                'min_cells': 10
            },
            'aggregation_stats': {
                'total_cells_aggregated': 500,
                'n_samples': 2,
                'n_cell_types': 2
            }
        }
        
        original_dataset_info = {
            'original_modality': 'single_cell_rna_seq',
            'n_original_cells': 1000,
            'n_original_genes': 2000
        }
        
        result = pseudobulk_adapter.from_source(
            pseudobulk_dataframe,
            aggregation_metadata=aggregation_metadata,
            original_dataset_info=original_dataset_info
        )
        
        assert 'pseudobulk_params' in result.uns
        assert 'aggregation_stats' in result.uns
        assert 'original_dataset_info' in result.uns
        assert result.uns['pseudobulk_params']['aggregation_method'] == 'sum'
    
    def test_from_source_validation_failure(self, strict_pseudobulk_adapter):
        """Test loading with validation failure in strict mode."""
        # Create invalid data
        invalid_df = pd.DataFrame(
            np.random.randn(2, 10),  # Negative values invalid for counts
            index=['Sample1', 'Sample2'],
            columns=[f"Gene_{i}" for i in range(10)]
        )
        
        with pytest.raises(ValidationError):
            strict_pseudobulk_adapter.from_source(invalid_df, validate_schema=True)
    
    def test_from_source_unsupported_type(self, pseudobulk_adapter):
        """Test loading from unsupported source type."""
        with pytest.raises(AdapterError, match="Unsupported source type"):
            pseudobulk_adapter.from_source([1, 2, 3])  # List is unsupported


@pytest.mark.unit
class TestPseudobulkAdapterSampleParsing:
    """Test pseudobulk sample ID parsing functionality."""
    
    def test_parse_pseudobulk_sample_ids_standard(self, pseudobulk_adapter):
        """Test parsing standard sample_celltype format."""
        sample_ids = pd.Index(['Sample_A_T_cell', 'Sample_B_B_cell', 'Sample_C_Monocyte'])
        
        result = pseudobulk_adapter._parse_pseudobulk_sample_ids(sample_ids)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert result.loc['Sample_A_T_cell', 'sample_id'] == 'Sample_A'
        assert result.loc['Sample_A_T_cell', 'cell_type'] == 'T_cell'
        assert result.loc['Sample_B_B_cell', 'cell_type'] == 'B_cell'
    
    def test_parse_pseudobulk_sample_ids_complex(self, pseudobulk_adapter):
        """Test parsing complex sample names with underscores."""
        sample_ids = pd.Index(['Patient_001_Day_7_T_cell', 'Control_Sample_B_cell'])
        
        result = pseudobulk_adapter._parse_pseudobulk_sample_ids(sample_ids)
        
        # Should split from right to handle underscores in sample names
        assert result.loc['Patient_001_Day_7_T_cell', 'sample_id'] == 'Patient_001_Day_7'
        assert result.loc['Patient_001_Day_7_T_cell', 'cell_type'] == 'T_cell'
        assert result.loc['Control_Sample_B_cell', 'sample_id'] == 'Control_Sample'
        assert result.loc['Control_Sample_B_cell', 'cell_type'] == 'B_cell'
    
    def test_parse_pseudobulk_sample_ids_no_underscore(self, pseudobulk_adapter):
        """Test parsing sample IDs without cell type separator."""
        sample_ids = pd.Index(['Sample1', 'Sample2'])
        
        result = pseudobulk_adapter._parse_pseudobulk_sample_ids(sample_ids)
        
        # Should handle gracefully with unknown cell type
        assert result.loc['Sample1', 'sample_id'] == 'Sample1'
        assert result.loc['Sample1', 'cell_type'] == 'unknown'
        assert result.loc['Sample2', 'sample_id'] == 'Sample2'
        assert result.loc['Sample2', 'cell_type'] == 'unknown'


@pytest.mark.unit
class TestPseudobulkAdapterValidation:
    """Test PseudobulkAdapter validation functionality."""
    
    def test_validate_valid_data(self, pseudobulk_adapter, valid_pseudobulk_adata):
        """Test validation of valid pseudobulk data."""
        result = pseudobulk_adapter.validate(valid_pseudobulk_adata)
        
        assert isinstance(result, ValidationResult)
        assert not result.has_errors()
    
    def test_validate_override_strict(self, pseudobulk_adapter, valid_pseudobulk_adata):
        """Test validation with strict mode override."""
        # Remove some data to make it slightly invalid
        adata = valid_pseudobulk_adata.copy()
        del adata.uns['pseudobulk_params']
        
        # Non-strict should be more forgiving
        result_permissive = pseudobulk_adapter.validate(adata, strict=False)
        result_strict = pseudobulk_adapter.validate(adata, strict=True)
        
        # May have different levels of strictness
        assert isinstance(result_permissive, ValidationResult)
        assert isinstance(result_strict, ValidationResult)
    
    def test_validate_small_sample_size(self, pseudobulk_adapter):
        """Test validation with small sample size."""
        # Create very small dataset
        small_adata = ad.AnnData(
            X=np.random.randn(2, 50),  # Only 2 pseudobulk samples
            obs=pd.DataFrame({
                'sample_id': ['S1', 'S2'],
                'cell_type': ['T_cell', 'T_cell'],
                'n_cells_aggregated': [50, 45]
            }),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])
        )
        
        result = pseudobulk_adapter.validate(small_adata)
        
        # Should warn about small sample size
        assert result.has_warnings()
        warning_msg = " ".join(result.warnings).lower()
        assert 'pseudobulk samples' in warning_msg
    
    def test_validate_low_cell_counts(self, pseudobulk_adapter):
        """Test validation with low cell counts."""
        adata = ad.AnnData(
            X=np.random.randn(4, 50),
            obs=pd.DataFrame({
                'sample_id': ['S1', 'S1', 'S2', 'S2'],
                'cell_type': ['T', 'B', 'T', 'B'],
                'n_cells_aggregated': [5, 3, 8, 2]  # Low cell counts
            }),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])
        )
        
        result = pseudobulk_adapter.validate(adata)
        
        # Should warn about low cell counts
        assert result.has_warnings()
        warning_msg = " ".join(result.warnings)
        assert 'pseudobulk samples have <10 cells' in warning_msg


@pytest.mark.unit
class TestPseudobulkAdapterQualityMetrics:
    """Test PseudobulkAdapter quality metrics calculation."""
    
    def test_get_quality_metrics_basic(self, pseudobulk_adapter, valid_pseudobulk_adata):
        """Test basic quality metrics calculation."""
        metrics = pseudobulk_adapter.get_quality_metrics(valid_pseudobulk_adata)
        
        assert isinstance(metrics, dict)
        assert 'n_obs' in metrics
        assert 'n_vars' in metrics
        assert 'total_counts' in metrics
    
    def test_get_quality_metrics_with_cell_counts(self, pseudobulk_adapter, valid_pseudobulk_adata):
        """Test quality metrics with cell count information."""
        metrics = pseudobulk_adapter.get_quality_metrics(valid_pseudobulk_adata)
        
        # Should include pseudobulk-specific metrics
        assert 'total_cells_aggregated' in metrics
        assert 'mean_cells_per_pseudobulk' in metrics
        assert 'min_cells_per_pseudobulk' in metrics
        assert 'max_cells_per_pseudobulk' in metrics
        
        assert metrics['total_cells_aggregated'] == sum([245, 89, 156, 198, 72, 134])
        assert metrics['min_cells_per_pseudobulk'] == 72
        assert metrics['max_cells_per_pseudobulk'] == 245
    
    def test_get_quality_metrics_diversity(self, pseudobulk_adapter, valid_pseudobulk_adata):
        """Test quality metrics for sample and cell type diversity."""
        metrics = pseudobulk_adapter.get_quality_metrics(valid_pseudobulk_adata)
        
        assert 'n_unique_samples' in metrics
        assert 'n_cell_types' in metrics
        assert 'cell_types' in metrics
        
        assert metrics['n_unique_samples'] == 2
        assert metrics['n_cell_types'] == 3
        assert set(metrics['cell_types']) == {'T_cell', 'B_cell', 'Monocyte'}
    
    def test_get_quality_metrics_conditions(self, pseudobulk_adapter, valid_pseudobulk_adata):
        """Test quality metrics with experimental conditions."""
        metrics = pseudobulk_adapter.get_quality_metrics(valid_pseudobulk_adata)
        
        assert 'n_conditions' in metrics
        assert 'conditions' in metrics
        assert metrics['n_conditions'] == 2
        assert set(metrics['conditions']) == {'Control', 'Treatment'}


@pytest.mark.unit
class TestPseudobulkAdapterPreprocessing:
    """Test PseudobulkAdapter preprocessing functionality."""
    
    def test_preprocess_data_basic(self, pseudobulk_adapter, valid_pseudobulk_adata):
        """Test basic preprocessing functionality."""
        result = pseudobulk_adapter.preprocess_data(valid_pseudobulk_adata)
        
        assert isinstance(result, ad.AnnData)
        assert 'raw_aggregated' in result.layers
        
        # Should calculate pseudobulk metrics
        assert 'n_pseudobulk_samples' in result.var.columns
        assert 'mean_aggregated_counts' in result.var.columns
        assert 'n_genes_detected' in result.obs.columns
    
    def test_calculate_pseudobulk_metrics(self, pseudobulk_adapter, valid_pseudobulk_adata):
        """Test pseudobulk-specific metrics calculation."""
        result = pseudobulk_adapter._calculate_pseudobulk_metrics(valid_pseudobulk_adata)
        
        # Gene-level metrics
        assert 'n_pseudobulk_samples' in result.var.columns
        assert 'mean_aggregated_counts' in result.var.columns
        assert 'total_aggregated_counts' in result.var.columns
        
        # Sample-level metrics
        assert 'n_genes_detected' in result.obs.columns
        assert 'total_aggregated_counts' in result.obs.columns
        
        # Verify calculations are reasonable
        assert result.var['n_pseudobulk_samples'].min() >= 0
        assert result.var['n_pseudobulk_samples'].max() <= result.n_obs
        assert result.obs['n_genes_detected'].min() >= 0
        assert result.obs['n_genes_detected'].max() <= result.n_vars


@pytest.mark.unit
class TestPseudobulkAdapterFileLoading:
    """Test PseudobulkAdapter file loading functionality."""
    
    @patch('pandas.read_csv')
    def test_load_csv_pseudobulk_data(self, mock_read_csv, pseudobulk_adapter, pseudobulk_dataframe):
        """Test loading pseudobulk data from CSV."""
        mock_read_csv.return_value = pseudobulk_dataframe
        
        result = pseudobulk_adapter._load_csv_pseudobulk_data("test.csv")
        
        assert isinstance(result, ad.AnnData)
        mock_read_csv.assert_called_once()
    
    @patch('pandas.read_excel')
    def test_load_excel_pseudobulk_data(self, mock_read_excel, pseudobulk_adapter, pseudobulk_dataframe):
        """Test loading pseudobulk data from Excel."""
        mock_read_excel.return_value = pseudobulk_dataframe
        
        result = pseudobulk_adapter._load_excel_pseudobulk_data("test.xlsx")
        
        assert isinstance(result, ad.AnnData)
        mock_read_excel.assert_called_once()
    
    @patch.object(Path, 'exists')
    def test_load_pseudobulk_from_file_not_found(self, mock_exists, pseudobulk_adapter):
        """Test loading from non-existent file."""
        mock_exists.return_value = False
        
        with pytest.raises(AdapterError, match="File not found"):
            pseudobulk_adapter._load_pseudobulk_from_file("nonexistent.csv")
    
    def test_from_source_unsupported_format(self, pseudobulk_adapter):
        """Test loading from unsupported file format."""
        with patch.object(pseudobulk_adapter, 'detect_format', return_value='unsupported'):
            with patch.object(Path, 'exists', return_value=True):
                with pytest.raises(AdapterError, match="Unsupported file format"):
                    pseudobulk_adapter.from_source("test.unsupported")


@pytest.mark.unit
class TestPseudobulkAdapterAggregation:
    """Test PseudobulkAdapter aggregation convenience methods."""
    
    def test_create_aggregated_data_validation_only(self, pseudobulk_adapter):
        """Test that create_aggregated_data validates inputs but doesn't aggregate."""
        # Create mock single-cell data
        single_cell_adata = ad.AnnData(
            X=np.random.randint(0, 100, (1000, 200)),
            obs=pd.DataFrame({
                'sample_id': np.random.choice(['S1', 'S2'], 1000),
                'cell_type': np.random.choice(['T', 'B', 'NK'], 1000)
            }),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(200)])
        )
        
        # Should validate inputs but raise informative error
        with pytest.raises(AdapterError, match="Direct aggregation not implemented"):
            pseudobulk_adapter.create_aggregated_data(
                single_cell_adata, 'sample_id', 'cell_type'
            )
    
    def test_create_aggregated_data_missing_columns(self, pseudobulk_adapter):
        """Test aggregation with missing required columns."""
        single_cell_adata = ad.AnnData(
            X=np.random.randint(0, 100, (100, 50)),
            obs=pd.DataFrame({'sample_id': ['S1'] * 100})  # Missing cell_type
        )
        
        with pytest.raises(AdapterError, match="Cell type column 'cell_type' not found"):
            pseudobulk_adapter.create_aggregated_data(
                single_cell_adata, 'sample_id', 'cell_type'
            )


@pytest.mark.unit
class TestPseudobulkAdapterSchema:
    """Test PseudobulkAdapter schema-related functionality."""
    
    def test_get_schema(self, pseudobulk_adapter):
        """Test schema retrieval."""
        schema = pseudobulk_adapter.get_schema()
        
        assert isinstance(schema, dict)
        assert schema == PseudobulkSchema.get_pseudobulk_schema()
    
    def test_get_supported_formats(self, pseudobulk_adapter):
        """Test supported formats list."""
        formats = pseudobulk_adapter.get_supported_formats()
        
        expected_formats = ['h5ad', 'csv', 'tsv', 'txt', 'xlsx', 'xls']
        assert set(formats) == set(expected_formats)


@pytest.mark.unit
class TestPseudobulkAdapterDesignValidation:
    """Test experimental design validation in adapter."""
    
    def test_check_experimental_balance_balanced(self, pseudobulk_adapter, valid_pseudobulk_adata):
        """Test experimental balance checking with balanced design."""
        result = ValidationResult()
        
        pseudobulk_adapter._check_experimental_balance(valid_pseudobulk_adata, result)
        
        # Balanced design should not generate errors
        assert not result.has_errors()
    
    def test_check_experimental_balance_missing_combinations(self, pseudobulk_adapter):
        """Test experimental balance with missing condition-celltype combinations."""
        # Create unbalanced design (missing Treatment B_cell)
        adata = ad.AnnData(
            X=np.random.randn(5, 50),
            obs=pd.DataFrame({
                'sample_id': ['S1', 'S1', 'S1', 'S2', 'S2'],
                'cell_type': ['T_cell', 'B_cell', 'Monocyte', 'T_cell', 'Monocyte'],  # Missing Treatment B_cell
                'condition': ['Control', 'Control', 'Control', 'Treatment', 'Treatment'],
                'n_cells_aggregated': [100, 80, 90, 120, 85]
            }),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])
        )
        
        result = ValidationResult()
        pseudobulk_adapter._check_experimental_balance(adata, result)
        
        assert result.has_warnings()
        warning_msg = " ".join(result.warnings)
        assert 'condition-celltype combinations have no samples' in warning_msg
    
    def test_check_experimental_balance_low_replication(self, pseudobulk_adapter):
        """Test experimental balance with insufficient replication."""
        # Create design with single samples per group
        adata = ad.AnnData(
            X=np.random.randn(4, 50),
            obs=pd.DataFrame({
                'sample_id': ['S1', 'S2', 'S3', 'S4'],
                'cell_type': ['T_cell', 'T_cell', 'B_cell', 'B_cell'],
                'condition': ['Control', 'Treatment', 'Control', 'Treatment'],
                'n_cells_aggregated': [100, 120, 80, 90]
            }),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])
        )
        
        result = ValidationResult()
        pseudobulk_adapter._check_experimental_balance(adata, result)
        
        # Should warn about low replication
        assert result.has_warnings()
        warning_msg = " ".join(result.warnings)
        assert 'Minimum samples per group: 1' in warning_msg


@pytest.mark.unit
class TestPseudobulkAdapterErrorHandling:
    """Test PseudobulkAdapter error handling."""
    
    def test_from_source_exception_handling(self, pseudobulk_adapter):
        """Test exception handling in from_source."""
        with patch.object(pseudobulk_adapter, '_create_pseudobulk_from_dataframe', 
                         side_effect=ValueError("Test error")):
            with pytest.raises(AdapterError, match="Failed to load pseudobulk data"):
                pseudobulk_adapter.from_source(pd.DataFrame())
    
    def test_validation_result_merging(self, pseudobulk_adapter, valid_pseudobulk_adata):
        """Test that validation results are properly merged."""
        # Test that all validation components contribute to final result
        result = pseudobulk_adapter.validate(valid_pseudobulk_adata)
        
        assert isinstance(result, ValidationResult)
        # The validate method should merge results from:
        # 1. validator.validate() (schema validation)
        # 2. _validate_basic_structure() (structural validation)  
        # 3. _validate_pseudobulk_specific() (pseudobulk-specific validation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
