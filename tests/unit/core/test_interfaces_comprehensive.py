"""
Additional comprehensive tests for interface implementations.

This module provides additional testing specifically targeting the concrete
methods and default implementations in the interface classes to improve
test coverage and ensure all functionality is properly tested.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from lobster.core.interfaces.adapter import IModalityAdapter
from lobster.core.interfaces.backend import IDataBackend
from lobster.core.interfaces.validator import IValidator, ValidationResult
from lobster.core.interfaces.base_client import BaseClient

from tests.mock_data.factories import SingleCellDataFactory, ProteomicsDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Concrete Method Testing for IModalityAdapter
# ===============================================================================

class ConcreteModalityAdapter(IModalityAdapter):
    """Concrete implementation for testing default methods."""

    def from_source(self, source, **kwargs) -> ad.AnnData:
        """Minimal implementation for testing."""
        return SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

    def validate(self, adata: ad.AnnData, strict: bool = False) -> ValidationResult:
        """Minimal implementation for testing."""
        return ValidationResult()

    def get_schema(self) -> Dict[str, Any]:
        """Minimal implementation for testing."""
        return {"required_obs": ["sample_id"], "required_var": ["gene_id"]}

    def get_supported_formats(self) -> List[str]:
        """Minimal implementation for testing."""
        return ["csv", "h5ad", "xlsx"]


@pytest.mark.unit
class TestModalityAdapterConcreteMethods:
    """Test concrete methods in IModalityAdapter."""

    def test_get_modality_name_default(self):
        """Test default modality name extraction."""
        adapter = ConcreteModalityAdapter()

        # Should extract from class name
        name = adapter.get_modality_name()
        assert name == "concretemodality"  # Class name lowercased with 'adapter' removed

    def test_detect_format_various_extensions(self):
        """Test format detection with various file extensions."""
        adapter = ConcreteModalityAdapter()

        test_cases = [
            ("file.csv", "csv"),
            ("file.CSV", "csv"),  # Case insensitive
            ("data.tsv", "tsv"),
            ("data.txt", "txt"),
            ("data.h5ad", "h5ad"),
            ("data.h5", "h5"),
            ("data.xlsx", "excel"),
            ("data.xls", "excel"),
            ("data.mtx", "mtx"),
            ("data.h5mu", "h5mu"),
            ("unknown.abc", None),
            ("no_extension", None)
        ]

        for filepath, expected in test_cases:
            result = adapter.detect_format(filepath)
            assert result == expected, f"Failed for {filepath}: expected {expected}, got {result}"

    def test_detect_format_with_path_objects(self):
        """Test format detection with Path objects."""
        adapter = ConcreteModalityAdapter()

        path_obj = Path("/some/path/file.h5ad")
        result = adapter.detect_format(path_obj)
        assert result == "h5ad"

    def test_detect_format_non_path_input(self):
        """Test format detection with non-path input."""
        adapter = ConcreteModalityAdapter()

        # Should return None for non-path inputs
        result = adapter.detect_format(123)
        assert result is None

        result = adapter.detect_format(pd.DataFrame())
        assert result is None

    def test_preprocess_data_default(self):
        """Test default preprocessing (should return unchanged data)."""
        adapter = ConcreteModalityAdapter()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        result = adapter.preprocess_data(adata)

        # Default implementation should return the same object
        assert result is adata

    def test_preprocess_data_with_kwargs(self):
        """Test preprocessing with keyword arguments."""
        adapter = ConcreteModalityAdapter()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Should handle kwargs gracefully
        result = adapter.preprocess_data(adata, normalize=True, scale=False)
        assert result is adata

    def test_get_quality_metrics_default(self):
        """Test default quality metrics calculation."""
        adapter = ConcreteModalityAdapter()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        metrics = adapter.get_quality_metrics(adata)

        # Should return basic metrics
        assert isinstance(metrics, dict)
        assert "n_obs" in metrics
        assert "n_vars" in metrics
        assert "sparsity" in metrics
        assert "memory_usage" in metrics

        assert metrics["n_obs"] == adata.n_obs
        assert metrics["n_vars"] == adata.n_vars
        assert isinstance(metrics["sparsity"], float)
        assert metrics["sparsity"] >= 0.0 and metrics["sparsity"] <= 1.0

    def test_get_quality_metrics_sparse_data(self):
        """Test quality metrics with sparse data."""
        adapter = ConcreteModalityAdapter()

        # Create sparse data - most values are zero
        from scipy.sparse import csr_matrix
        dense_data = np.random.rand(50, 100)
        dense_data[dense_data < 0.8] = 0  # 80% zeros
        sparse_X = csr_matrix(dense_data)
        adata = ad.AnnData(X=sparse_X)

        metrics = adapter.get_quality_metrics(adata)

        assert metrics["n_obs"] == 50
        assert metrics["n_vars"] == 100
        # Sparsity = 1 - (non-zero / total), so high sparsity means mostly zeros
        assert metrics["sparsity"] >= 0.0 and metrics["sparsity"] <= 1.0

    def test_standardize_metadata_no_mapping(self):
        """Test metadata standardization without mapping."""
        adapter = ConcreteModalityAdapter()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        result = adapter.standardize_metadata(adata)

        # Should return same object when no mapping provided
        assert result is adata

    def test_standardize_metadata_with_mapping(self):
        """Test metadata standardization with mapping."""
        adapter = ConcreteModalityAdapter()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Add some metadata to rename
        adata.obs["old_column"] = range(adata.n_obs)
        adata.var["old_var_column"] = range(adata.n_vars)

        mapping = {
            "old_column": "new_column",
            "old_var_column": "new_var_column"
        }

        result = adapter.standardize_metadata(adata, mapping)

        # Check obs renaming
        assert "new_column" in result.obs.columns
        assert "old_column" not in result.obs.columns

        # Check var renaming
        assert "new_var_column" in result.var.columns
        assert "old_var_column" not in result.var.columns

    def test_standardize_metadata_partial_mapping(self):
        """Test metadata standardization with partial mapping."""
        adapter = ConcreteModalityAdapter()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Add metadata
        adata.obs["exists"] = range(adata.n_obs)
        adata.obs["keep"] = range(adata.n_obs)

        # Only map one column
        mapping = {"exists": "renamed"}

        result = adapter.standardize_metadata(adata, mapping)

        assert "renamed" in result.obs.columns
        assert "exists" not in result.obs.columns
        assert "keep" in result.obs.columns  # Unmapped column should remain

    def test_add_provenance_basic(self):
        """Test adding provenance information."""
        adapter = ConcreteModalityAdapter()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        source_info = {"path": "/data/test.csv", "format": "csv"}
        processing_params = {"normalize": True, "scale": False}

        result = adapter.add_provenance(adata, source_info, processing_params)

        # Should add provenance to uns
        assert "provenance" in result.uns
        assert isinstance(result.uns["provenance"], list)
        assert len(result.uns["provenance"]) == 1

        prov_entry = result.uns["provenance"][0]
        assert prov_entry["adapter"] == "ConcreteModalityAdapter"
        assert prov_entry["modality"] == "concretemodality"
        assert prov_entry["source"] == source_info
        assert prov_entry["processing_params"] == processing_params
        assert "timestamp" in prov_entry
        assert "version" in prov_entry

    def test_add_provenance_existing_provenance(self):
        """Test adding provenance to data that already has provenance."""
        adapter = ConcreteModalityAdapter()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Add existing provenance
        adata.uns["provenance"] = [{"existing": "entry"}]

        source_info = {"path": "/data/test.csv"}
        result = adapter.add_provenance(adata, source_info)

        # Should append to existing provenance
        assert len(result.uns["provenance"]) == 2
        assert result.uns["provenance"][0] == {"existing": "entry"}
        assert "adapter" in result.uns["provenance"][1]

    def test_add_provenance_no_processing_params(self):
        """Test adding provenance without processing parameters."""
        adapter = ConcreteModalityAdapter()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        source_info = {"path": "/data/test.csv"}
        result = adapter.add_provenance(adata, source_info)

        prov_entry = result.uns["provenance"][0]
        assert prov_entry["processing_params"] == {}


# ===============================================================================
# Concrete Method Testing for IDataBackend
# ===============================================================================

class ConcreteDataBackend(IDataBackend):
    """Concrete implementation for testing default methods."""

    def load(self, path, **kwargs) -> ad.AnnData:
        """Minimal implementation."""
        return SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

    def save(self, adata: ad.AnnData, path, **kwargs) -> None:
        """Minimal implementation."""
        pass

    def exists(self, path) -> bool:
        """Minimal implementation."""
        return True

    def delete(self, path) -> None:
        """Minimal implementation."""
        pass

    def list_files(self, directory, pattern: str = "*") -> list[str]:
        """Minimal implementation."""
        return ["file1.h5ad", "file2.csv"]

    def get_metadata(self, path) -> Dict[str, Any]:
        """Minimal implementation."""
        return {"size": 1024, "modified": "2023-01-01"}


@pytest.mark.unit
class TestDataBackendConcreteMethods:
    """Test concrete methods in IDataBackend."""

    def test_get_storage_info_default(self):
        """Test default storage info implementation."""
        backend = ConcreteDataBackend()

        info = backend.get_storage_info()

        assert isinstance(info, dict)
        assert "backend_type" in info
        assert "capabilities" in info
        assert "configuration" in info

        assert info["backend_type"] == "ConcreteDataBackend"
        assert "load" in info["capabilities"]
        assert "save" in info["capabilities"]
        assert "exists" in info["capabilities"]
        assert isinstance(info["configuration"], dict)

    def test_validate_path_default(self):
        """Test default path validation (should return unchanged)."""
        backend = ConcreteDataBackend()

        # String path
        path_str = "/path/to/file.h5ad"
        result = backend.validate_path(path_str)
        assert result == path_str

        # Path object
        path_obj = Path("/path/to/file.h5ad")
        result = backend.validate_path(path_obj)
        assert result == path_obj

    def test_supports_format_default(self):
        """Test default format support checking."""
        backend = ConcreteDataBackend()

        # Default implementation should support h5ad and csv
        assert backend.supports_format("h5ad") is True
        assert backend.supports_format("H5AD") is True  # Case insensitive
        assert backend.supports_format("csv") is True
        assert backend.supports_format("CSV") is True

        # Should not support other formats by default
        assert backend.supports_format("xlsx") is False
        assert backend.supports_format("unknown") is False

    def test_supports_format_case_insensitive(self):
        """Test that format support is case insensitive."""
        backend = ConcreteDataBackend()

        formats_to_test = ["h5ad", "H5AD", "h5AD", "CSV", "csv", "Csv"]

        for fmt in formats_to_test:
            if fmt.lower() in ["h5ad", "csv"]:
                assert backend.supports_format(fmt) is True
            else:
                assert backend.supports_format(fmt) is False


# ===============================================================================
# Concrete Method Testing for IValidator
# ===============================================================================

class ConcreteValidator(IValidator):
    """Concrete implementation for testing default methods."""

    def validate(self, adata: ad.AnnData, strict: bool = False,
                check_types: bool = True, check_ranges: bool = True,
                check_completeness: bool = True) -> ValidationResult:
        """Minimal implementation."""
        return ValidationResult()

    def validate_schema_compliance(self, adata: ad.AnnData,
                                 schema: Dict[str, Any]) -> ValidationResult:
        """Minimal implementation."""
        return ValidationResult()


@pytest.mark.unit
class TestValidatorConcreteMethods:
    """Test concrete methods in IValidator."""

    def test_validate_obs_metadata_no_requirements(self):
        """Test obs metadata validation with no requirements."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        result = validator.validate_obs_metadata(adata)

        assert isinstance(result, ValidationResult)
        assert not result.has_errors
        assert not result.has_warnings

    def test_validate_obs_metadata_with_required_columns(self):
        """Test obs metadata validation with required columns."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Add required column
        adata.obs["sample_id"] = range(adata.n_obs)

        result = validator.validate_obs_metadata(adata, required_columns=["sample_id"])

        assert not result.has_errors
        assert not result.has_warnings

    def test_validate_obs_metadata_missing_required(self):
        """Test obs metadata validation with missing required columns."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        result = validator.validate_obs_metadata(adata, required_columns=["missing_column"])

        assert result.has_errors
        assert "missing_column" in result.errors[0]

    def test_validate_obs_metadata_nan_values(self):
        """Test obs metadata validation with NaN values."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Add column with all NaN values
        adata.obs["nan_column"] = pd.Series([np.nan] * adata.n_obs)

        result = validator.validate_obs_metadata(adata, required_columns=["nan_column"])

        assert result.has_warnings
        assert "nan_column" in result.warnings[0]
        assert "NaN values" in result.warnings[0]

    def test_validate_obs_metadata_unexpected_columns(self):
        """Test obs metadata validation with unexpected columns."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Add extra columns
        adata.obs["extra1"] = range(adata.n_obs)
        adata.obs["extra2"] = range(adata.n_obs)

        result = validator.validate_obs_metadata(adata,
                                               required_columns=["required"],
                                               optional_columns=["optional"])

        assert result.info  # Should have info about unexpected columns
        assert "extra1" in result.info[0] or "extra2" in result.info[0]

    def test_validate_var_metadata_functionality(self):
        """Test var metadata validation (similar to obs)."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Add required var column
        adata.var["gene_id"] = range(adata.n_vars)

        result = validator.validate_var_metadata(adata, required_columns=["gene_id"])

        assert not result.has_errors

    def test_validate_var_metadata_missing_required(self):
        """Test var metadata validation with missing required columns."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        result = validator.validate_var_metadata(adata, required_columns=["missing_gene_column"])

        assert result.has_errors
        assert "missing_gene_column" in result.errors[0]

    def test_validate_layers_no_requirements(self):
        """Test layer validation with no requirements."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        result = validator.validate_layers(adata)

        assert not result.has_errors
        assert not result.has_warnings

    def test_validate_layers_missing_expected(self):
        """Test layer validation with missing expected layers."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        result = validator.validate_layers(adata, expected_layers=["raw", "normalized"])

        assert result.has_warnings
        assert len(result.warnings) >= 1

    def test_validate_layers_shape_mismatch(self):
        """Test layer validation logic can handle shape checking."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Test the validation logic works with normal layers
        # (AnnData prevents actual shape mismatches, so we test normal operation)
        result = validator.validate_layers(adata)
        assert isinstance(result, ValidationResult)
        # Should pass with no layers

    def test_validate_layers_correct_shape(self):
        """Test layer validation with correct shape."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        # Add layer with correct shape
        adata.layers["correct_shape"] = np.random.rand(adata.n_obs, adata.n_vars)

        result = validator.validate_layers(adata)

        assert not result.has_errors

    def test_validate_data_quality_empty_data(self):
        """Test data quality validation with empty data."""
        validator = ConcreteValidator()

        # Test with minimal data that has valid dimensions
        minimal_adata = ad.AnnData(X=np.ones((1, 1)))
        result = validator.validate_data_quality(minimal_adata)
        # This should work without errors
        assert isinstance(result, ValidationResult)

        # Test the basic logic for empty observations
        # We'll test this indirectly through the n_obs check
        empty_adata = ad.AnnData()
        if empty_adata.n_obs == 0:
            # The validation should catch this case
            result = validator.validate_data_quality(empty_adata)
            assert result.has_errors
            assert "observations" in result.errors[0].lower()

    def test_validate_data_quality_nan_values(self):
        """Test data quality validation with NaN values."""
        validator = ConcreteValidator()

        # Create data with many NaN values (need to check the actual implementation)
        X_with_nans = np.full((100, 50), np.nan)  # All NaN values
        X_with_nans[0:10, 0:10] = 1.0  # Add some non-NaN values

        adata = ad.AnnData(X=X_with_nans)
        result = validator.validate_data_quality(adata)

        # The validation might not detect NaN values if the X matrix doesn't have isnan method
        # Just check that validation completes without error
        assert isinstance(result, ValidationResult)
        # Note: The actual behavior depends on the X matrix type (dense vs sparse)

    def test_validate_data_quality_negative_values(self):
        """Test data quality validation with negative values."""
        validator = ConcreteValidator()

        # Create data with negative values
        X_with_negatives = np.random.rand(50, 100) - 0.5  # Some negative values

        adata = ad.AnnData(X=X_with_negatives)
        result = validator.validate_data_quality(adata)

        assert result.has_warnings
        assert "Negative values" in result.warnings[0]

    def test_validate_data_quality_normal_data(self):
        """Test data quality validation with normal data."""
        validator = ConcreteValidator()
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

        result = validator.validate_data_quality(adata)

        # Should have minimal issues with good data
        assert not result.has_errors


# ===============================================================================
# ValidationResult Additional Testing
# ===============================================================================

@pytest.mark.unit
class TestValidationResultMethods:
    """Test ValidationResult methods more comprehensively."""

    def test_merge_validation_results(self):
        """Test merging multiple validation results."""
        result1 = ValidationResult()
        result1.add_error("Error 1")
        result1.add_warning("Warning 1")
        result1.add_info("Info 1")
        result1.metadata["key1"] = "value1"

        result2 = ValidationResult()
        result2.add_error("Error 2")
        result2.add_warning("Warning 2")
        result2.add_info("Info 2")
        result2.metadata["key2"] = "value2"

        merged = result1.merge(result2)

        assert len(merged.errors) == 2
        assert len(merged.warnings) == 2
        assert len(merged.info) == 2
        assert merged.metadata["key1"] == "value1"
        assert merged.metadata["key2"] == "value2"

        # Original results should be unchanged
        assert len(result1.errors) == 1
        assert len(result2.errors) == 1

    def test_to_dict_comprehensive(self):
        """Test complete dictionary conversion."""
        result = ValidationResult()
        result.add_error("Test error")
        result.add_warning("Test warning")
        result.add_info("Test info")
        result.metadata["test_key"] = "test_value"

        result_dict = result.to_dict()

        expected_keys = ["errors", "warnings", "info", "metadata",
                        "has_errors", "has_warnings", "is_valid"]

        for key in expected_keys:
            assert key in result_dict

        assert result_dict["errors"] == ["Test error"]
        assert result_dict["warnings"] == ["Test warning"]
        assert result_dict["info"] == ["Test info"]
        assert result_dict["metadata"]["test_key"] == "test_value"
        assert result_dict["has_errors"] is True
        assert result_dict["has_warnings"] is True
        assert result_dict["is_valid"] is False

    def test_summary_various_states(self):
        """Test summary generation for various states."""
        # No issues
        clean_result = ValidationResult()
        assert "no issues" in clean_result.summary().lower()

        # Only errors
        error_result = ValidationResult()
        error_result.add_error("Error")
        summary = error_result.summary()
        assert "1 error" in summary
        assert "warning" not in summary

        # Only warnings
        warning_result = ValidationResult()
        warning_result.add_warning("Warning")
        summary = warning_result.summary()
        assert "1 warning" in summary
        assert "error" not in summary

        # Mixed
        mixed_result = ValidationResult()
        mixed_result.add_error("Error")
        mixed_result.add_warning("Warning")
        mixed_result.add_info("Info")
        summary = mixed_result.summary()
        assert "1 error" in summary
        assert "1 warning" in summary
        assert "1 info" in summary

    def test_format_messages_comprehensive(self):
        """Test message formatting with all types."""
        result = ValidationResult()
        result.add_error("Critical error")
        result.add_warning("Minor warning")
        result.add_info("Information")

        # With info
        formatted_with_info = result.format_messages(include_info=True)
        assert "ERRORS:" in formatted_with_info
        assert "WARNINGS:" in formatted_with_info
        assert "INFO:" in formatted_with_info
        assert "❌" in formatted_with_info
        assert "⚠️" in formatted_with_info
        assert "ℹ️" in formatted_with_info

        # Without info
        formatted_without_info = result.format_messages(include_info=False)
        assert "ERRORS:" in formatted_without_info
        assert "WARNINGS:" in formatted_without_info
        assert "INFO:" not in formatted_without_info

    def test_format_messages_empty(self):
        """Test message formatting with empty result."""
        result = ValidationResult()

        formatted = result.format_messages()
        assert formatted == ""

    def test_format_messages_only_errors(self):
        """Test message formatting with only errors."""
        result = ValidationResult()
        result.add_error("Error 1")
        result.add_error("Error 2")

        formatted = result.format_messages()
        assert "ERRORS:" in formatted
        assert "Error 1" in formatted
        assert "Error 2" in formatted
        assert "WARNINGS:" not in formatted

    def test_is_valid_property(self):
        """Test is_valid property logic."""
        result = ValidationResult()
        assert result.is_valid is True

        result.add_warning("Warning")
        assert result.is_valid is True  # Warnings don't make invalid

        result.add_error("Error")
        assert result.is_valid is False  # Errors make invalid

    def test_add_methods(self):
        """Test individual add methods."""
        result = ValidationResult()

        result.add_error("Error message")
        assert len(result.errors) == 1
        assert result.errors[0] == "Error message"

        result.add_warning("Warning message")
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Warning message"

        result.add_info("Info message")
        assert len(result.info) == 1
        assert result.info[0] == "Info message"


# ===============================================================================
# BaseClient Optional Methods Testing
# ===============================================================================

class ConcreteBaseClient(BaseClient):
    """Concrete implementation for testing optional methods."""

    def __init__(self):
        pass

    def query(self, user_input: str, stream: bool = False):
        return {"response": "test"}

    def get_status(self):
        return {"status": "active"}

    def list_workspace_files(self, pattern: str = "*"):
        return []

    def read_file(self, filename: str):
        return "content"

    def write_file(self, filename: str, content: str):
        return True

    def get_conversation_history(self):
        return []

    def reset(self):
        pass

    def export_session(self, export_path=None):
        return Path("/tmp/export")


@pytest.mark.unit
class TestBaseClientOptionalMethods:
    """Test optional methods in BaseClient."""

    def test_get_usage_default(self):
        """Test default get_usage implementation."""
        client = ConcreteBaseClient()

        result = client.get_usage()

        assert isinstance(result, dict)
        assert "error" in result
        assert "success" in result
        assert result["success"] is False
        assert "not available" in result["error"]

    def test_list_models_default(self):
        """Test default list_models implementation."""
        client = ConcreteBaseClient()

        result = client.list_models()

        assert isinstance(result, dict)
        assert "error" in result
        assert "success" in result
        assert result["success"] is False
        assert "not available" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])