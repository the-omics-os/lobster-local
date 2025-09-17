"""
Unit tests for pseudobulk schema validation.

Tests the PseudobulkSchema class and related validation functions
for aggregated single-cell RNA-seq data converted to pseudobulk format.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad

from lobster.core.schemas.pseudobulk import (
    PseudobulkSchema, 
    _validate_pseudobulk_structure,
    _validate_aggregation_consistency,
    _validate_cell_counts,
    _validate_aggregation_params
)
from lobster.core.schemas.validation import FlexibleValidator
from lobster.core.interfaces.validator import ValidationResult


@pytest.fixture
def valid_pseudobulk_data():
    """Create valid pseudobulk data for testing."""
    # Create 6 pseudobulk samples (2 samples × 3 cell types)
    sample_ids = ['Sample_A', 'Sample_A', 'Sample_A', 'Sample_B', 'Sample_B', 'Sample_B']
    cell_types = ['T_cell', 'B_cell', 'Monocyte', 'T_cell', 'B_cell', 'Monocyte']
    pseudobulk_ids = [f"{s}_{c}" for s, c in zip(sample_ids, cell_types)]
    
    # Expression matrix (6 pseudobulk samples × 100 genes)
    expression_matrix = np.random.negative_binomial(10, 0.3, (6, 100)).astype(float)
    gene_names = [f"GENE_{i:03d}" for i in range(100)]
    
    adata = ad.AnnData(
        X=expression_matrix,
        obs=pd.DataFrame({
            'sample_id': sample_ids,
            'cell_type': cell_types,
            'n_cells_aggregated': [245, 89, 156, 198, 72, 134],
            'condition': ['Control', 'Control', 'Control', 'Treatment', 'Treatment', 'Treatment'],
            'batch': ['Batch1', 'Batch1', 'Batch1', 'Batch2', 'Batch2', 'Batch2']
        }, index=pseudobulk_ids),
        var=pd.DataFrame({
            'gene_symbol': [f"Gene_{i}" for i in range(100)],
            'gene_id': [f"ENSG0000{i:05d}" for i in range(100)]
        }, index=gene_names)
    )
    
    # Add required metadata
    adata.uns['pseudobulk_params'] = {
        'sample_col': 'sample_id',
        'celltype_col': 'cell_type',
        'min_cells': 10,
        'aggregation_method': 'sum',
        'original_n_cells': 894
    }
    
    adata.uns['aggregation_stats'] = {
        'n_samples': 2,
        'n_cell_types': 3,
        'total_cells_aggregated': 894,
        'mean_cells_per_pseudobulk': 149.0
    }
    
    return adata


@pytest.fixture
def invalid_pseudobulk_data():
    """Create invalid pseudobulk data for testing."""
    adata = ad.AnnData(
        X=np.random.randn(4, 50),
        obs=pd.DataFrame({
            'sample_id': ['S1', 'S1', 'S2', 'S2'],
            'cell_type': ['T', 'B', 'T', 'B']
        }),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])
    )
    # Missing required columns and metadata
    return adata


@pytest.mark.unit
class TestPseudobulkSchema:
    """Test PseudobulkSchema class functionality."""
    
    def test_get_pseudobulk_schema(self):
        """Test schema structure and content."""
        schema = PseudobulkSchema.get_pseudobulk_schema()
        
        assert isinstance(schema, dict)
        assert schema['modality'] == 'pseudobulk_rna_seq'
        
        # Check required sections
        assert 'obs' in schema
        assert 'var' in schema
        assert 'uns' in schema
        
        # Check required obs fields
        assert 'sample_id' in schema['obs']['required']
        assert 'cell_type' in schema['obs']['required']
        assert 'n_cells_aggregated' in schema['obs']['required']
        
        # Check required uns fields
        assert 'pseudobulk_params' in schema['uns']['required']
        assert 'aggregation_stats' in schema['uns']['required']
    
    def test_create_validator_default(self):
        """Test validator creation with default settings."""
        validator = PseudobulkSchema.create_validator()
        
        assert isinstance(validator, FlexibleValidator)
        assert validator.name == "PseudobulkValidator"
        assert len(validator.custom_rules) == 4  # 4 custom validation rules
    
    def test_create_validator_strict(self):
        """Test validator creation with strict mode."""
        validator = PseudobulkSchema.create_validator(strict=True)
        
        assert isinstance(validator, FlexibleValidator)
        # Strict mode should still have same structure
    
    def test_create_validator_ignore_warnings(self):
        """Test validator creation with ignored warnings."""
        ignore_warnings = ["missing values", "unexpected columns"]
        validator = PseudobulkSchema.create_validator(ignore_warnings=ignore_warnings)
        
        expected_ignored = {"missing values", "unexpected columns", "Unexpected obs columns", 
                          "Unexpected var columns", "missing values in optional fields"}
        assert expected_ignored.issubset(validator.ignore_warnings)
    
    def test_get_recommended_qc_thresholds(self):
        """Test QC thresholds structure and values."""
        thresholds = PseudobulkSchema.get_recommended_qc_thresholds()
        
        assert isinstance(thresholds, dict)
        assert 'min_cells_per_pseudobulk' in thresholds
        assert 'min_pseudobulk_samples' in thresholds
        assert 'min_genes_per_pseudobulk' in thresholds
        
        # Check reasonable values
        assert thresholds['min_cells_per_pseudobulk'] >= 5
        assert thresholds['min_pseudobulk_samples'] >= 2


@pytest.mark.unit 
class TestPseudobulkValidation:
    """Test pseudobulk validation functions."""
    
    def test_validate_pseudobulk_structure_valid(self, valid_pseudobulk_data):
        """Test structure validation with valid data."""
        result = _validate_pseudobulk_structure(valid_pseudobulk_data)
        
        assert isinstance(result, ValidationResult)
        assert not result.has_errors()
        # May have warnings but should not have errors
    
    def test_validate_pseudobulk_structure_missing_columns(self):
        """Test structure validation with missing required columns."""
        # Missing required columns
        adata = ad.AnnData(
            X=np.random.randn(6, 100),
            obs=pd.DataFrame({'sample_id': ['S1'] * 6}),  # Missing cell_type, n_cells_aggregated
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(100)])
        )
        
        result = _validate_pseudobulk_structure(adata)
        
        assert result.has_errors()
        error_msg = " ".join(result.errors)
        assert 'cell_type' in error_msg
        assert 'n_cells_aggregated' in error_msg
    
    def test_validate_pseudobulk_structure_duplicate_combinations(self):
        """Test detection of duplicate sample-celltype combinations."""
        adata = ad.AnnData(
            X=np.random.randn(6, 100),
            obs=pd.DataFrame({
                'sample_id': ['S1', 'S1', 'S1', 'S1', 'S2', 'S2'],  # Duplicate S1_T_cell
                'cell_type': ['T_cell', 'T_cell', 'B_cell', 'T_cell', 'T_cell', 'B_cell'],
                'n_cells_aggregated': [100, 50, 75, 80, 90, 85]
            }),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(100)])
        )
        
        result = _validate_pseudobulk_structure(adata)
        
        assert result.has_errors()
        assert 'duplicate' in " ".join(result.errors).lower()
    
    def test_validate_aggregation_consistency_valid(self, valid_pseudobulk_data):
        """Test aggregation consistency with valid data."""
        result = _validate_aggregation_consistency(valid_pseudobulk_data)
        
        assert isinstance(result, ValidationResult)
        # Should not have errors with valid cell counts
    
    def test_validate_aggregation_consistency_invalid_counts(self):
        """Test aggregation consistency with invalid cell counts."""
        adata = ad.AnnData(
            X=np.random.randn(4, 50),
            obs=pd.DataFrame({
                'sample_id': ['S1', 'S1', 'S2', 'S2'],
                'cell_type': ['T', 'B', 'T', 'B'],
                'n_cells_aggregated': [0, -5, 2, 150]  # Invalid counts
            }),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])
        )
        
        result = _validate_aggregation_consistency(adata)
        
        assert result.has_errors()
        error_msg = " ".join(result.errors).lower()
        assert 'zero or negative' in error_msg
    
    def test_validate_cell_counts_missing_combinations(self):
        """Test cell count validation with missing condition combinations."""
        adata = ad.AnnData(
            X=np.random.randn(4, 50),
            obs=pd.DataFrame({
                'sample_id': ['S1', 'S1', 'S2', 'S2'],
                'cell_type': ['T_cell', 'B_cell', 'T_cell', 'B_cell'],
                'condition': ['Control', 'Control', 'Treatment', 'Treatment'],  # Missing Control B_cell combo
                'n_cells_aggregated': [100, 80, 90, 70]
            }),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])
        )
        
        result = _validate_cell_counts(adata)
        
        # Should not have errors but may have warnings about missing combinations
        assert isinstance(result, ValidationResult)
    
    def test_validate_aggregation_params_valid(self, valid_pseudobulk_data):
        """Test aggregation parameter validation with valid parameters."""
        result = _validate_aggregation_params(valid_pseudobulk_data)
        
        assert isinstance(result, ValidationResult)
        # Should not have errors with valid parameters
    
    def test_validate_aggregation_params_missing(self):
        """Test aggregation parameter validation with missing parameters."""
        adata = ad.AnnData(
            X=np.random.randn(4, 50),
            obs=pd.DataFrame({
                'sample_id': ['S1', 'S1', 'S2', 'S2'],
                'cell_type': ['T', 'B', 'T', 'B'],
                'n_cells_aggregated': [100, 80, 90, 70]
            }),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])
        )
        # Missing pseudobulk_params in uns
        
        result = _validate_aggregation_params(adata)
        
        assert isinstance(result, ValidationResult)
        # Should handle missing parameters gracefully
    
    def test_validate_aggregation_params_invalid_method(self):
        """Test validation with invalid aggregation method."""
        adata = ad.AnnData(
            X=np.random.randn(4, 50),
            obs=pd.DataFrame({
                'sample_id': ['S1', 'S1', 'S2', 'S2'],
                'cell_type': ['T', 'B', 'T', 'B'],
                'n_cells_aggregated': [100, 80, 90, 70]
            }),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])
        )
        
        adata.uns['pseudobulk_params'] = {
            'aggregation_method': 'invalid_method',  # Invalid method
            'min_cells': 10
        }
        
        result = _validate_aggregation_params(adata)
        
        assert result.has_warnings()
        warning_msg = " ".join(result.warnings).lower()
        assert 'unexpected aggregation method' in warning_msg


@pytest.mark.unit
class TestPseudobulkValidator:
    """Test complete pseudobulk validation workflow."""
    
    def test_validator_with_valid_data(self, valid_pseudobulk_data):
        """Test validator with completely valid pseudobulk data."""
        validator = PseudobulkSchema.create_validator(strict=False)
        
        result = validator.validate(valid_pseudobulk_data)
        
        assert isinstance(result, ValidationResult)
        assert not result.has_errors()
        # May have some warnings but should be structurally sound
    
    def test_validator_with_invalid_data(self, invalid_pseudobulk_data):
        """Test validator with invalid pseudobulk data."""
        validator = PseudobulkSchema.create_validator(strict=True)
        
        result = validator.validate(invalid_pseudobulk_data)
        
        assert isinstance(result, ValidationResult)
        assert result.has_errors()
        
        # Should detect missing required columns
        error_msg = " ".join(result.errors).lower()
        assert 'n_cells_aggregated' in error_msg
    
    def test_validator_custom_rules_execution(self, valid_pseudobulk_data):
        """Test that all custom validation rules are executed."""
        validator = PseudobulkSchema.create_validator()
        
        # Ensure we have the expected custom rules
        expected_rules = [
            'check_pseudobulk_structure',
            'check_aggregation_consistency', 
            'check_cell_counts',
            'check_aggregation_params'
        ]
        
        for rule_name in expected_rules:
            assert rule_name in validator.custom_rules
        
        # Validate data and ensure rules were applied
        result = validator.validate(valid_pseudobulk_data)
        assert isinstance(result, ValidationResult)
    
    def test_validator_ignore_warnings(self):
        """Test validator ignoring specified warning types."""
        # Create data that would normally generate warnings
        adata = ad.AnnData(
            X=np.random.randn(4, 50),
            obs=pd.DataFrame({
                'sample_id': ['S1', 'S1', 'S2', 'S2'],
                'cell_type': ['T', 'B', 'T', 'B'],
                'n_cells_aggregated': [100, 80, 90, 70],
                'unexpected_column': ['extra'] * 4  # This would generate warning
            }),
            var=pd.DataFrame(index=[f"Gene_{i}" for i in range(50)])
        )
        
        validator = PseudobulkSchema.create_validator(
            ignore_warnings=["Unexpected obs columns"]
        )
        
        result = validator.validate(adata)
        
        # Should not warn about unexpected columns
        warning_msg = " ".join(result.warnings).lower()
        assert 'unexpected' not in warning_msg
    
    def test_qc_thresholds_structure(self):
        """Test QC thresholds return expected structure."""
        thresholds = PseudobulkSchema.get_recommended_qc_thresholds()
        
        required_thresholds = [
            'min_cells_per_pseudobulk',
            'min_pseudobulk_samples', 
            'min_genes_per_pseudobulk',
            'max_zero_fraction',
            'min_total_aggregated_counts',
            'min_samples_per_celltype',
            'max_aggregation_imbalance'
        ]
        
        for threshold in required_thresholds:
            assert threshold in thresholds
            assert isinstance(thresholds[threshold], (int, float))
            assert thresholds[threshold] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
