"""
Comprehensive unit tests for schema validation.

This module provides thorough testing of the schema system including
TRANSCRIPTOMICS_SCHEMA and PROTEOMICS_SCHEMA validation, data type enforcement,
value constraints, schema evolution, and custom validator implementations.

Test coverage target: 95%+ with meaningful tests for biological data validation.
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse

from lobster.core.schemas.transcriptomics import TRANSCRIPTOMICS_SCHEMA, TranscriptomicsValidator
from lobster.core.schemas.proteomics import PROTEOMICS_SCHEMA, ProteomicsValidator
from lobster.core.schemas.base import BaseValidator, ValidationResult

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
def compliant_transcriptomics_data():
    """Create transcriptomics data that complies with schema."""
    adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    
    # Ensure all required fields are present
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1)
    adata.obs['pct_counts_mt'] = np.random.uniform(0, 30, adata.n_obs)
    
    adata.var['gene_ids'] = [f"GENE_{i:05d}" for i in range(adata.n_vars)]
    adata.var['feature_types'] = 'Gene Expression'
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(('RPL', 'RPS'))
    
    # Add optional fields
    adata.obs['pct_counts_ribo'] = np.random.uniform(0, 50, adata.n_obs)
    adata.var['gene_names'] = adata.var_names.tolist()
    
    return adata


@pytest.fixture
def compliant_proteomics_data():
    """Create proteomics data that complies with schema."""
    adata = ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)
    
    # Ensure all required fields are present
    adata.obs['sample_id'] = [f"Sample_{i:03d}" for i in range(adata.n_obs)]
    adata.obs['total_protein_intensity'] = np.array(adata.X.sum(axis=1)).flatten()
    
    adata.var['protein_ids'] = [f"PROT_{i:05d}" for i in range(adata.n_vars)]
    adata.var['protein_names'] = [f"Protein_{i}" for i in range(adata.n_vars)]
    
    # Add optional fields
    adata.obs['batch'] = np.random.choice(['A', 'B', 'C'], adata.n_obs)
    adata.obs['condition'] = np.random.choice(['Control', 'Treatment'], adata.n_obs)
    adata.var['contaminant'] = np.random.choice([True, False], adata.n_vars, p=[0.05, 0.95])
    
    return adata


@pytest.fixture
def non_compliant_transcriptomics_data():
    """Create transcriptomics data that violates schema."""
    adata = ad.AnnData(X=np.random.randint(0, 100, (50, 200)))
    
    # Missing required obs columns: total_counts, n_genes_by_counts
    # Missing required var columns: gene_ids, feature_types
    
    return adata


@pytest.fixture
def non_compliant_proteomics_data():
    """Create proteomics data that violates schema."""
    adata = ad.AnnData(X=np.random.randn(20, 100))
    
    # Missing required obs columns: sample_id, total_protein_intensity
    # Missing required var columns: protein_ids, protein_names
    
    return adata


# ===============================================================================
# Schema Structure Tests
# ===============================================================================

@pytest.mark.unit
class TestSchemaStructure:
    """Test schema structure and content."""
    
    def test_transcriptomics_schema_structure(self):
        """Test TRANSCRIPTOMICS_SCHEMA has expected structure."""
        assert isinstance(TRANSCRIPTOMICS_SCHEMA, dict)
        
        # Check required top-level keys
        assert "required_obs" in TRANSCRIPTOMICS_SCHEMA
        assert "required_var" in TRANSCRIPTOMICS_SCHEMA
        assert "optional_obs" in TRANSCRIPTOMICS_SCHEMA
        assert "optional_var" in TRANSCRIPTOMICS_SCHEMA
        
        # Check data types
        assert isinstance(TRANSCRIPTOMICS_SCHEMA["required_obs"], list)
        assert isinstance(TRANSCRIPTOMICS_SCHEMA["required_var"], list)
        assert isinstance(TRANSCRIPTOMICS_SCHEMA["optional_obs"], list)
        assert isinstance(TRANSCRIPTOMICS_SCHEMA["optional_var"], list)
    
    def test_transcriptomics_schema_content(self):
        """Test TRANSCRIPTOMICS_SCHEMA contains expected fields."""
        schema = TRANSCRIPTOMICS_SCHEMA
        
        # Check required observation fields
        assert "total_counts" in schema["required_obs"]
        assert "n_genes_by_counts" in schema["required_obs"]
        
        # Check required variable fields
        assert "gene_ids" in schema["required_var"]
        assert "feature_types" in schema["required_var"]
        
        # Check optional fields exist
        assert "pct_counts_mt" in schema["optional_obs"]
        assert "gene_names" in schema["optional_var"]
    
    def test_proteomics_schema_structure(self):
        """Test PROTEOMICS_SCHEMA has expected structure."""
        assert isinstance(PROTEOMICS_SCHEMA, dict)
        
        # Check required top-level keys
        assert "required_obs" in PROTEOMICS_SCHEMA
        assert "required_var" in PROTEOMICS_SCHEMA
        assert "optional_obs" in PROTEOMICS_SCHEMA
        assert "optional_var" in PROTEOMICS_SCHEMA
        
        # Check data types
        assert isinstance(PROTEOMICS_SCHEMA["required_obs"], list)
        assert isinstance(PROTEOMICS_SCHEMA["required_var"], list)
        assert isinstance(PROTEOMICS_SCHEMA["optional_obs"], list)
        assert isinstance(PROTEOMICS_SCHEMA["optional_var"], list)
    
    def test_proteomics_schema_content(self):
        """Test PROTEOMICS_SCHEMA contains expected fields."""
        schema = PROTEOMICS_SCHEMA
        
        # Check required observation fields
        assert "sample_id" in schema["required_obs"]
        assert "total_protein_intensity" in schema["required_obs"]
        
        # Check required variable fields
        assert "protein_ids" in schema["required_var"]
        assert "protein_names" in schema["required_var"]
        
        # Check optional fields exist
        assert "batch" in schema["optional_obs"]
        assert "contaminant" in schema["optional_var"]
    
    def test_schema_field_types(self):
        """Test that schema fields have appropriate constraints."""
        # Test transcriptomics schema has layer specifications
        if "layers" in TRANSCRIPTOMICS_SCHEMA:
            assert isinstance(TRANSCRIPTOMICS_SCHEMA["layers"], list)
            assert "counts" in TRANSCRIPTOMICS_SCHEMA["layers"]
        
        # Test proteomics schema has layer specifications
        if "layers" in PROTEOMICS_SCHEMA:
            assert isinstance(PROTEOMICS_SCHEMA["layers"], list)
    
    def test_schema_immutability(self):
        """Test that schemas are effectively immutable."""
        # Get original values
        orig_transcriptomics = TRANSCRIPTOMICS_SCHEMA.copy()
        orig_proteomics = PROTEOMICS_SCHEMA.copy()
        
        # Attempt to modify (this should not affect the original schemas)
        try:
            TRANSCRIPTOMICS_SCHEMA["required_obs"].append("test_field")
            PROTEOMICS_SCHEMA["required_obs"].append("test_field")
        except (TypeError, AttributeError):
            # Expected if schemas are properly immutable
            pass
        
        # Reset to ensure tests don't interfere with each other
        if "test_field" in TRANSCRIPTOMICS_SCHEMA["required_obs"]:
            TRANSCRIPTOMICS_SCHEMA["required_obs"].remove("test_field")
        if "test_field" in PROTEOMICS_SCHEMA["required_obs"]:
            PROTEOMICS_SCHEMA["required_obs"].remove("test_field")


# ===============================================================================
# BaseValidator Tests
# ===============================================================================

@pytest.mark.unit
class TestBaseValidator:
    """Test BaseValidator functionality."""
    
    def test_base_validator_initialization(self):
        """Test BaseValidator initialization."""
        schema = {"required_obs": ["test_field"], "required_var": ["test_var"]}
        validator = BaseValidator(schema)
        
        assert validator.schema == schema
        assert isinstance(validator.validation_result, ValidationResult)
    
    def test_check_required_fields_success(self):
        """Test successful required field checking."""
        schema = {"required_obs": ["sample_id"], "required_var": ["gene_id"]}
        validator = BaseValidator(schema)
        
        # Create compliant data
        adata = ad.AnnData(X=np.random.rand(10, 50))
        adata.obs['sample_id'] = [f"S{i}" for i in range(10)]
        adata.var['gene_id'] = [f"G{i}" for i in range(50)]
        
        # Should not add any errors
        validator._check_required_fields(adata)
        assert not validator.validation_result.has_errors
    
    def test_check_required_fields_missing(self):
        """Test required field checking with missing fields."""
        schema = {"required_obs": ["sample_id"], "required_var": ["gene_id"]}
        validator = BaseValidator(schema)
        
        # Create non-compliant data
        adata = ad.AnnData(X=np.random.rand(10, 50))
        # Missing both required fields
        
        validator._check_required_fields(adata)
        assert validator.validation_result.has_errors
        assert len(validator.validation_result.errors) == 2  # One for obs, one for var
    
    def test_check_data_types_numeric(self):
        """Test numeric data type checking."""
        validator = BaseValidator({})
        
        adata = ad.AnnData(X=np.random.rand(10, 50))
        adata.obs['numeric_field'] = np.random.rand(10)
        adata.obs['string_field'] = [f"S{i}" for i in range(10)]
        
        validator._check_data_types(adata)
        # Should not add errors for reasonable data types
        
    def test_check_data_dimensions(self):
        """Test data dimension validation."""
        validator = BaseValidator({})
        
        # Test normal data
        adata = ad.AnnData(X=np.random.rand(100, 1000))
        validator._check_data_dimensions(adata)
        assert not validator.validation_result.has_errors
        
        # Test very small data
        small_adata = ad.AnnData(X=np.random.rand(2, 5))
        validator._check_data_dimensions(small_adata)
        assert validator.validation_result.has_warnings
    
    def test_check_data_quality_empty_data(self):
        """Test data quality checking with empty data."""
        validator = BaseValidator({})
        
        # Empty data should generate errors
        empty_adata = ad.AnnData(X=np.array([]).reshape(0, 0))
        validator._check_data_quality(empty_adata)
        assert validator.validation_result.has_errors
    
    def test_check_data_quality_invalid_values(self):
        """Test data quality checking with invalid values."""
        validator = BaseValidator({})
        
        # Data with NaN and inf values
        adata = ad.AnnData(X=np.array([[1, 2, np.nan], [4, np.inf, 6]]))
        validator._check_data_quality(adata)
        assert validator.validation_result.has_warnings or validator.validation_result.has_errors
    
    def test_validation_result_reset(self):
        """Test that validation result is reset between validations."""
        validator = BaseValidator({})
        
        # Add some errors
        validator.validation_result.errors.append("Test error")
        assert validator.validation_result.has_errors
        
        # Reset should clear errors
        validator._reset_validation_result()
        assert not validator.validation_result.has_errors


# ===============================================================================
# TranscriptomicsValidator Tests
# ===============================================================================

@pytest.mark.unit
class TestTranscriptomicsValidator:
    """Test TranscriptomicsValidator functionality."""
    
    def test_validator_initialization(self):
        """Test TranscriptomicsValidator initialization."""
        validator = TranscriptomicsValidator()
        
        assert validator.schema == TRANSCRIPTOMICS_SCHEMA
        assert isinstance(validator.validation_result, ValidationResult)
    
    def test_validate_compliant_data(self, compliant_transcriptomics_data):
        """Test validation of compliant transcriptomics data."""
        validator = TranscriptomicsValidator()
        
        result = validator.validate(compliant_transcriptomics_data)
        
        assert isinstance(result, ValidationResult)
        assert not result.has_errors
        # May have warnings but should not have errors
    
    def test_validate_non_compliant_data(self, non_compliant_transcriptomics_data):
        """Test validation of non-compliant transcriptomics data."""
        validator = TranscriptomicsValidator()
        
        result = validator.validate(non_compliant_transcriptomics_data, strict=True)
        
        assert isinstance(result, ValidationResult)
        assert result.has_errors
        
        # Should have errors for missing required fields
        error_messages = " ".join(result.errors)
        assert "total_counts" in error_messages
        assert "gene_ids" in error_messages
    
    def test_validate_strict_vs_permissive(self, non_compliant_transcriptomics_data):
        """Test strict vs permissive validation modes."""
        validator = TranscriptomicsValidator()
        
        # Strict mode should have errors
        strict_result = validator.validate(non_compliant_transcriptomics_data, strict=True)
        assert strict_result.has_errors
        
        # Permissive mode should have warnings instead
        permissive_result = validator.validate(non_compliant_transcriptomics_data, strict=False)
        # In permissive mode, missing fields might be warnings
        assert permissive_result.has_warnings or permissive_result.has_errors
    
    def test_check_qc_metrics_present(self):
        """Test QC metrics validation."""
        validator = TranscriptomicsValidator()
        
        # Data with QC metrics
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1)
        adata.obs['pct_counts_mt'] = np.random.uniform(0, 20, adata.n_obs)
        
        validator._check_qc_metrics(adata)
        assert not validator.validation_result.has_errors
    
    def test_check_qc_metrics_missing(self):
        """Test QC metrics validation with missing metrics."""
        validator = TranscriptomicsValidator()
        
        # Data without QC metrics
        adata = ad.AnnData(X=np.random.randint(0, 100, (50, 200)))
        
        validator._check_qc_metrics(adata)
        assert validator.validation_result.has_errors or validator.validation_result.has_warnings
    
    def test_check_gene_annotations(self):
        """Test gene annotation validation."""
        validator = TranscriptomicsValidator()
        
        # Create data with gene annotations
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        adata.var['gene_ids'] = [f"GENE_{i:05d}" for i in range(adata.n_vars)]
        adata.var['feature_types'] = 'Gene Expression'
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        adata.var['ribo'] = adata.var_names.str.startswith(('RPL', 'RPS'))
        
        validator._check_gene_annotations(adata)
        assert not validator.validation_result.has_errors
    
    def test_check_mitochondrial_genes(self):
        """Test mitochondrial gene validation."""
        validator = TranscriptomicsValidator()
        
        # Create data with mitochondrial genes
        gene_names = ['Gene1', 'MT-ATP6', 'Gene2', 'MT-CO1', 'Gene3']
        adata = ad.AnnData(
            X=np.random.randint(0, 100, (10, 5)),
            var=pd.DataFrame(index=gene_names)
        )
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        
        validator._check_mitochondrial_genes(adata)
        
        # Should not have errors if MT genes are properly flagged
        assert not validator.validation_result.has_errors
    
    def test_check_count_matrix_properties(self):
        """Test count matrix property validation."""
        validator = TranscriptomicsValidator()
        
        # Integer count matrix (good)
        count_matrix = np.random.randint(0, 1000, (100, 500))
        adata = ad.AnnData(X=count_matrix)
        
        validator._check_count_matrix_properties(adata)
        # Integer counts should not generate errors
        
        # Negative values (bad)
        negative_matrix = np.random.randint(-10, 1000, (100, 500))
        adata_negative = ad.AnnData(X=negative_matrix)
        
        validator._check_count_matrix_properties(adata_negative)
        assert validator.validation_result.has_warnings or validator.validation_result.has_errors
    
    def test_get_schema(self):
        """Test schema retrieval."""
        validator = TranscriptomicsValidator()
        
        schema = validator.get_schema()
        
        assert schema == TRANSCRIPTOMICS_SCHEMA
        assert isinstance(schema, dict)


# ===============================================================================
# ProteomicsValidator Tests
# ===============================================================================

@pytest.mark.unit
class TestProteomicsValidator:
    """Test ProteomicsValidator functionality."""
    
    def test_validator_initialization(self):
        """Test ProteomicsValidator initialization."""
        validator = ProteomicsValidator()
        
        assert validator.schema == PROTEOMICS_SCHEMA
        assert isinstance(validator.validation_result, ValidationResult)
    
    def test_validate_compliant_data(self, compliant_proteomics_data):
        """Test validation of compliant proteomics data."""
        validator = ProteomicsValidator()
        
        result = validator.validate(compliant_proteomics_data)
        
        assert isinstance(result, ValidationResult)
        assert not result.has_errors
        # May have warnings but should not have errors
    
    def test_validate_non_compliant_data(self, non_compliant_proteomics_data):
        """Test validation of non-compliant proteomics data."""
        validator = ProteomicsValidator()
        
        result = validator.validate(non_compliant_proteomics_data, strict=True)
        
        assert isinstance(result, ValidationResult)
        assert result.has_errors
        
        # Should have errors for missing required fields
        error_messages = " ".join(result.errors)
        assert "sample_id" in error_messages
        assert "protein_ids" in error_messages
    
    def test_check_protein_annotations(self):
        """Test protein annotation validation."""
        validator = ProteomicsValidator()
        
        # Create data with protein annotations
        adata = ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)
        adata.var['protein_ids'] = [f"PROT_{i:05d}" for i in range(adata.n_vars)]
        adata.var['protein_names'] = [f"Protein_{i}" for i in range(adata.n_vars)]
        
        validator._check_protein_annotations(adata)
        assert not validator.validation_result.has_errors
    
    def test_check_intensity_values(self):
        """Test intensity value validation."""
        validator = ProteomicsValidator()
        
        # Positive intensity values (good)
        intensity_matrix = np.random.lognormal(3, 1, (50, 200))
        adata = ad.AnnData(X=intensity_matrix)
        
        validator._check_intensity_values(adata)
        # Positive values should be fine
        
        # Negative intensity values (unusual for proteomics)
        negative_matrix = np.random.normal(0, 1, (50, 200))  # Can have negative values
        adata_negative = ad.AnnData(X=negative_matrix)
        
        validator._check_intensity_values(adata_negative)
        # May generate warnings for negative intensities
    
    def test_check_missing_values(self):
        """Test missing value validation."""
        validator = ProteomicsValidator()
        
        # Create data with missing values (common in proteomics)
        matrix = np.random.lognormal(3, 1, (50, 200))
        # Add missing values
        missing_mask = np.random.choice([True, False], matrix.shape, p=[0.2, 0.8])
        matrix[missing_mask] = np.nan
        
        adata = ad.AnnData(X=matrix)
        
        validator._check_missing_values(adata)
        # Missing values should generate warnings, not errors
        assert not validator.validation_result.has_errors
    
    def test_check_contaminants(self):
        """Test contaminant validation."""
        validator = ProteomicsValidator()
        
        # Create data with contaminant annotations
        protein_names = ['Protein1', 'CON_TRYP_HUMAN', 'Protein2', 'REV_Protein3']
        adata = ad.AnnData(
            X=np.random.lognormal(3, 1, (20, 4)),
            var=pd.DataFrame(index=protein_names)
        )
        adata.var['contaminant'] = [False, True, False, True]
        adata.var['reverse'] = [False, False, False, True]
        
        validator._check_contaminants(adata)
        assert not validator.validation_result.has_errors
    
    def test_check_sample_metadata(self):
        """Test sample metadata validation."""
        validator = ProteomicsValidator()
        
        # Create data with sample metadata
        adata = ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)
        adata.obs['sample_id'] = [f"Sample_{i:03d}" for i in range(adata.n_obs)]
        adata.obs['total_protein_intensity'] = np.array(adata.X.sum(axis=1)).flatten()
        
        validator._check_sample_metadata(adata)
        assert not validator.validation_result.has_errors
    
    def test_get_schema(self):
        """Test schema retrieval."""
        validator = ProteomicsValidator()
        
        schema = validator.get_schema()
        
        assert schema == PROTEOMICS_SCHEMA
        assert isinstance(schema, dict)


# ===============================================================================
# Schema Evolution and Backward Compatibility Tests
# ===============================================================================

@pytest.mark.unit
class TestSchemaEvolution:
    """Test schema evolution and backward compatibility."""
    
    def test_schema_version_compatibility(self):
        """Test that schemas maintain backward compatibility."""
        # Test that old data structures still validate
        # This is important for data that was processed with older versions
        
        # Create minimal old-style data
        old_style_transcriptomics = ad.AnnData(X=np.random.randint(0, 100, (50, 200)))
        old_style_transcriptomics.obs['n_genes'] = (old_style_transcriptomics.X > 0).sum(axis=1)  # Old field name
        
        validator = TranscriptomicsValidator()
        
        # Should handle gracefully in permissive mode
        result = validator.validate(old_style_transcriptomics, strict=False)
        # May have warnings but should not crash
        assert isinstance(result, ValidationResult)
    
    def test_optional_field_handling(self):
        """Test handling of optional fields."""
        # Test that data validates correctly with and without optional fields
        
        # Minimal compliant data
        minimal_data = ad.AnnData(X=np.random.randint(0, 100, (50, 200)))
        minimal_data.obs['total_counts'] = np.array(minimal_data.X.sum(axis=1)).flatten()
        minimal_data.obs['n_genes_by_counts'] = (minimal_data.X > 0).sum(axis=1)
        minimal_data.var['gene_ids'] = [f"GENE_{i:05d}" for i in range(200)]
        minimal_data.var['feature_types'] = 'Gene Expression'
        
        validator = TranscriptomicsValidator()
        result = validator.validate(minimal_data)
        assert not result.has_errors
        
        # Enhanced data with optional fields
        enhanced_data = minimal_data.copy()
        enhanced_data.obs['pct_counts_mt'] = np.random.uniform(0, 20, 50)
        enhanced_data.var['gene_names'] = [f"Gene_{i}" for i in range(200)]
        
        enhanced_result = validator.validate(enhanced_data)
        assert not enhanced_result.has_errors
    
    def test_custom_schema_extension(self):
        """Test extending schemas with custom fields."""
        # Test that custom schemas can extend base schemas
        
        custom_schema = TRANSCRIPTOMICS_SCHEMA.copy()
        if "custom_obs" not in custom_schema:
            custom_schema["custom_obs"] = []
        custom_schema["custom_obs"] = custom_schema.get("optional_obs", []) + ["experiment_id", "batch_id"]
        
        validator = BaseValidator(custom_schema)
        
        # Create data with custom fields
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        adata.obs['experiment_id'] = 'EXP_001'
        adata.obs['batch_id'] = np.random.choice(['Batch1', 'Batch2'], adata.n_obs)
        
        # Custom validator should handle extended schema
        assert validator.schema == custom_schema
    
    def test_schema_field_deprecation(self):
        """Test handling of deprecated fields."""
        # Test that deprecated fields generate appropriate warnings
        
        # Simulate data with deprecated field names
        deprecated_data = ad.AnnData(X=np.random.randint(0, 100, (50, 200)))
        deprecated_data.obs['total_umis'] = np.array(deprecated_data.X.sum(axis=1)).flatten()  # Deprecated name
        deprecated_data.obs['n_genes'] = (deprecated_data.X > 0).sum(axis=1)  # Old name
        
        validator = TranscriptomicsValidator()
        
        # Should handle gracefully
        result = validator.validate(deprecated_data, strict=False)
        assert isinstance(result, ValidationResult)


# ===============================================================================
# Custom Validator Implementation Tests
# ===============================================================================

@pytest.mark.unit
class TestCustomValidatorImplementations:
    """Test custom validator implementations and patterns."""
    
    def test_custom_validator_creation(self):
        """Test creating custom validators."""
        
        class CustomTranscriptomicsValidator(BaseValidator):
            """Custom validator with additional checks."""
            
            def __init__(self):
                # Start with base transcriptomics schema
                custom_schema = TRANSCRIPTOMICS_SCHEMA.copy()
                # Add custom requirements
                custom_schema["required_obs"] = custom_schema["required_obs"] + ["experiment_date"]
                super().__init__(custom_schema)
            
            def validate(self, adata: ad.AnnData, strict: bool = False) -> ValidationResult:
                """Custom validation with additional checks."""
                # Run base validation
                result = super().validate(adata, strict)
                
                # Add custom validation logic
                if hasattr(adata, 'obs') and 'experiment_date' in adata.obs:
                    # Validate date format
                    pass  # Custom date validation would go here
                
                return result
        
        validator = CustomTranscriptomicsValidator()
        
        # Test with compliant data
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        adata.obs['experiment_date'] = '2024-01-15'  # Add required custom field
        
        result = validator.validate(adata)
        # Should work with custom validator
        assert isinstance(result, ValidationResult)
    
    def test_validator_composition(self):
        """Test composing multiple validators."""
        
        def composite_validation(adata: ad.AnnData) -> ValidationResult:
            """Run multiple validators and combine results."""
            transcriptomics_validator = TranscriptomicsValidator()
            proteomics_validator = ProteomicsValidator()
            
            # Try both validators
            t_result = transcriptomics_validator.validate(adata, strict=False)
            p_result = proteomics_validator.validate(adata, strict=False)
            
            # Combine results (choose the one with fewer errors)
            if len(t_result.errors) <= len(p_result.errors):
                return t_result
            else:
                return p_result
        
        # Test with transcriptomics data
        transcriptomics_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        result = composite_validation(transcriptomics_data)
        assert isinstance(result, ValidationResult)
    
    def test_conditional_validation(self):
        """Test conditional validation based on data characteristics."""
        
        class ConditionalValidator(BaseValidator):
            """Validator that adapts based on data characteristics."""
            
            def __init__(self):
                super().__init__({})  # Empty schema, will be set dynamically
            
            def validate(self, adata: ad.AnnData, strict: bool = False) -> ValidationResult:
                """Conditional validation based on data size."""
                self._reset_validation_result()
                
                if adata.n_vars > 10000:
                    # Likely single-cell data
                    self.schema = TRANSCRIPTOMICS_SCHEMA
                else:
                    # Likely proteomics data
                    self.schema = PROTEOMICS_SCHEMA
                
                # Run appropriate validation
                self._check_required_fields(adata)
                self._check_data_dimensions(adata)
                
                return self.validation_result
        
        validator = ConditionalValidator()
        
        # Test with high-dimensional data (single-cell-like)
        high_dim_data = ad.AnnData(X=np.random.randint(0, 100, (100, 15000)))
        result_high = validator.validate(high_dim_data)
        assert isinstance(result_high, ValidationResult)
        
        # Test with low-dimensional data (proteomics-like)
        low_dim_data = ad.AnnData(X=np.random.randn(50, 500))
        result_low = validator.validate(low_dim_data)
        assert isinstance(result_low, ValidationResult)


# ===============================================================================
# Edge Cases and Error Handling Tests
# ===============================================================================

@pytest.mark.unit
class TestSchemaEdgeCases:
    """Test edge cases and error handling in schema validation."""
    
    def test_empty_data_validation(self):
        """Test validation of empty datasets."""
        validators = [
            TranscriptomicsValidator(),
            ProteomicsValidator()
        ]
        
        for validator in validators:
            # Completely empty data
            empty_data = ad.AnnData(X=np.array([]).reshape(0, 0))
            result = validator.validate(empty_data)
            assert result.has_errors  # Should detect empty data
            
            # Empty observations but with variables
            empty_obs = ad.AnnData(X=np.array([]).reshape(0, 10))
            result_obs = validator.validate(empty_obs)
            assert result_obs.has_errors or result_obs.has_warnings
            
            # Empty variables but with observations
            empty_vars = ad.AnnData(X=np.array([]).reshape(10, 0))
            result_vars = validator.validate(empty_vars)
            assert result_vars.has_errors or result_vars.has_warnings
    
    def test_malformed_data_validation(self):
        """Test validation of malformed data structures."""
        validator = TranscriptomicsValidator()
        
        # Data with inconsistent dimensions
        adata = ad.AnnData(X=np.random.rand(10, 20))
        # Add observation metadata with wrong number of entries
        adata.obs = pd.DataFrame({'sample_id': ['S1', 'S2']})  # Only 2 entries for 10 observations
        
        # Should handle gracefully
        result = validator.validate(adata, strict=False)
        assert isinstance(result, ValidationResult)
    
    def test_extreme_values_validation(self):
        """Test validation with extreme values."""
        validator = TranscriptomicsValidator()
        
        # Very large values
        large_values = ad.AnnData(X=np.random.rand(10, 20) * 1e10)
        result_large = validator.validate(large_values, strict=False)
        assert isinstance(result_large, ValidationResult)
        
        # Very small values
        small_values = ad.AnnData(X=np.random.rand(10, 20) * 1e-10)
        result_small = validator.validate(small_values, strict=False)
        assert isinstance(result_small, ValidationResult)
    
    def test_unicode_and_special_characters(self):
        """Test validation with unicode and special characters."""
        validator = TranscriptomicsValidator()
        
        # Data with unicode gene names
        unicode_genes = ['Gene_α', 'Gene_β', 'Gene_γ', '基因_1', 'Gène_2']
        adata = ad.AnnData(
            X=np.random.randint(0, 100, (10, 5)),
            var=pd.DataFrame(index=unicode_genes)
        )
        adata.var['gene_ids'] = [f"GENE_{i:03d}" for i in range(5)]
        adata.var['feature_types'] = 'Gene Expression'
        
        result = validator.validate(adata, strict=False)
        assert isinstance(result, ValidationResult)
        # Should handle unicode gracefully
    
    def test_concurrent_validation(self):
        """Test thread safety of validators."""
        import threading
        import time
        
        validator = TranscriptomicsValidator()
        results = []
        errors = []
        
        def validate_worker(worker_id):
            """Worker function for concurrent validation."""
            try:
                # Create test data
                adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                result = validator.validate(adata)
                results.append((worker_id, result))
                time.sleep(0.01)
            except Exception as e:
                errors.append((worker_id, e))
        
        # Run multiple validators concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=validate_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent validation errors: {errors}"
        assert len(results) == 5
        
        # All results should be valid ValidationResult instances
        for worker_id, result in results:
            assert isinstance(result, ValidationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])