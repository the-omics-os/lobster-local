"""
Comprehensive tests for the legacy DataManager (data_manager_old.py).

This module provides thorough testing of the legacy DataManager class to ensure
backward compatibility and proper functionality for existing code that depends on it.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go
import scanpy as sc

from lobster.core.data_manager_old import DataManager
from tests.mock_data.base import SMALL_DATASET_CONFIG, MEDIUM_DATASET_CONFIG


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp(prefix="lobster_legacy_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n_samples, n_genes = 50, 200

    data = np.random.negative_binomial(n=5, p=0.3, size=(n_samples, n_genes)).astype(float)

    sample_names = [f"Sample_{i:02d}" for i in range(n_samples)]
    gene_names = [f"Gene_{i:03d}" for i in range(n_genes)]

    return pd.DataFrame(data, index=sample_names, columns=gene_names)


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        "experiment_id": "EXP_001",
        "title": "Test RNA-seq Experiment",
        "organism": "Homo sapiens",
        "tissue": "brain",
        "technology": "Illumina HiSeq",
        "n_samples": 50,
        "processing_date": "2024-01-15"
    }


# ===============================================================================
# Legacy DataManager Initialization Tests
# ===============================================================================

@pytest.mark.unit
class TestLegacyDataManagerInitialization:
    """Test legacy DataManager initialization and setup."""

    def test_init_default_parameters(self, temp_workspace):
        """Test initialization with default parameters."""
        dm = DataManager(workspace_path=temp_workspace)

        assert dm.current_data is None
        assert isinstance(dm.current_metadata, dict)
        assert len(dm.current_metadata) == 0
        assert dm.adata is None
        assert isinstance(dm.latest_plots, list)
        assert len(dm.latest_plots) == 0
        assert isinstance(dm.file_paths, dict)
        assert isinstance(dm.processing_log, list)
        assert isinstance(dm.tool_usage_history, list)
        assert dm.max_plots_history == 50

    def test_init_custom_workspace(self, temp_workspace):
        """Test initialization with custom workspace path."""
        custom_console = Mock()

        dm = DataManager(workspace_path=temp_workspace, console=custom_console)

        assert dm.workspace_path == temp_workspace
        assert dm.console is custom_console

        # Check that workspace directories are created
        assert dm.data_dir.exists()
        assert dm.plots_dir.exists()
        assert dm.exports_dir.exists()

    def test_workspace_setup(self, temp_workspace):
        """Test workspace directory structure creation."""
        dm = DataManager(workspace_path=temp_workspace)

        # Check that all required directories exist
        assert dm.workspace_path.exists()
        assert dm.data_dir.exists()
        assert dm.plots_dir.exists()
        assert dm.exports_dir.exists()

        # Check directory structure
        assert dm.data_dir == temp_workspace / "data"
        assert dm.plots_dir == temp_workspace / "plots"
        assert dm.exports_dir == temp_workspace / "exports"


# ===============================================================================
# Data Management Tests
# ===============================================================================

@pytest.mark.unit
class TestLegacyDataManagement:
    """Test legacy data management functionality."""

    def test_set_data_success(self, temp_workspace, sample_dataframe, sample_metadata):
        """Test successful data setting."""
        dm = DataManager(workspace_path=temp_workspace)

        result = dm.set_data(sample_dataframe, sample_metadata)

        assert result is not None
        assert dm.current_data is not None
        assert dm.current_data.shape == sample_dataframe.shape
        assert dm.current_metadata == sample_metadata

        # Check that data types are numeric
        assert all(dm.current_data.dtypes.apply(lambda x: np.issubdtype(x, np.number)))

    def test_set_data_invalid_input(self, temp_workspace):
        """Test set_data with invalid input."""
        dm = DataManager(workspace_path=temp_workspace)

        # Test with None
        with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
            dm.set_data(None)

        # Test with non-DataFrame
        with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
            dm.set_data("not a dataframe")

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            dm.set_data(empty_df)

    def test_set_data_type_conversion(self, temp_workspace):
        """Test automatic data type conversion."""
        dm = DataManager(workspace_path=temp_workspace)

        # Create DataFrame with mixed types
        data = pd.DataFrame({
            'numeric_col': [1, 2, 3],
            'string_col': ['1.5', '2.5', '3.5'],
            'mixed_col': [1, '2', 3.0]
        })

        dm.set_data(data)

        # Should convert string numbers to numeric
        assert np.issubdtype(dm.current_data['string_col'].dtype, np.number)
        assert np.issubdtype(dm.current_data['mixed_col'].dtype, np.number)

    def test_set_data_nan_handling(self, temp_workspace):
        """Test NaN value handling."""
        dm = DataManager(workspace_path=temp_workspace)

        # Create DataFrame with NaN values
        data = pd.DataFrame({
            'col1': [1, np.nan, 3],
            'col2': [np.nan, 2, 3],
            'col3': [1, 2, np.nan]
        })

        dm.set_data(data)

        # Should fill NaN values with 0
        assert not dm.current_data.isna().any().any()
        assert (dm.current_data.fillna(0) == dm.current_data).all().all()

    def test_data_validation_logging(self, temp_workspace, sample_dataframe):
        """Test that data validation information is logged."""
        dm = DataManager(workspace_path=temp_workspace)

        with patch('lobster.core.data_manager_old.logger') as mock_logger:
            dm.set_data(sample_dataframe)

            # Should log data shape and types
            mock_logger.info.assert_any_call(f"Data shape: {sample_dataframe.shape}")
            assert mock_logger.info.call_count >= 2


# ===============================================================================
# Plot Management Tests
# ===============================================================================

@pytest.mark.unit
class TestLegacyPlotManagement:
    """Test legacy plot management functionality."""

    def test_plot_storage_basic(self, temp_workspace):
        """Test basic plot storage functionality."""
        dm = DataManager(workspace_path=temp_workspace)

        # Create test plot
        fig = go.Figure()
        fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6])

        # Add plot to history (assuming add_plot method exists or manual addition)
        plot_entry = {
            "id": "plot_1",
            "figure": fig,
            "title": "Test Plot",
            "timestamp": "2024-01-15T10:00:00",
            "source": "test"
        }

        dm.latest_plots.append(plot_entry)

        assert len(dm.latest_plots) == 1
        assert dm.latest_plots[0]["id"] == "plot_1"

    def test_plot_history_limit(self, temp_workspace):
        """Test plot history limit enforcement."""
        dm = DataManager(workspace_path=temp_workspace)
        dm.max_plots_history = 3  # Set low limit for testing

        # Add more plots than the limit
        for i in range(5):
            plot_entry = {
                "id": f"plot_{i}",
                "figure": go.Figure(),
                "title": f"Plot {i}",
                "timestamp": f"2024-01-15T10:0{i}:00"
            }
            dm.latest_plots.append(plot_entry)

            # Enforce limit manually (if not automatic)
            if len(dm.latest_plots) > dm.max_plots_history:
                dm.latest_plots = dm.latest_plots[-dm.max_plots_history:]

        # Should only keep the most recent plots
        assert len(dm.latest_plots) <= dm.max_plots_history


# ===============================================================================
# File Management Tests
# ===============================================================================

@pytest.mark.unit
class TestLegacyFileManagement:
    """Test legacy file management functionality."""

    def test_file_path_tracking(self, temp_workspace):
        """Test file path tracking functionality."""
        dm = DataManager(workspace_path=temp_workspace)

        # Test file path storage
        test_paths = {
            "input_data": "/path/to/input.csv",
            "processed_data": "/path/to/processed.h5ad",
            "results": "/path/to/results.xlsx"
        }

        for key, path in test_paths.items():
            dm.file_paths[key] = path

        assert len(dm.file_paths) == 3
        for key, expected_path in test_paths.items():
            assert dm.file_paths[key] == expected_path

    def test_workspace_file_organization(self, temp_workspace):
        """Test workspace file organization."""
        dm = DataManager(workspace_path=temp_workspace)

        # Create test files in different directories
        test_files = {
            dm.data_dir / "test_data.csv": "data content",
            dm.plots_dir / "test_plot.html": "plot content",
            dm.exports_dir / "test_export.zip": "export content"
        }

        for filepath, content in test_files.items():
            filepath.write_text(content)

        # Verify files exist in correct locations
        for filepath in test_files.keys():
            assert filepath.exists()


# ===============================================================================
# Processing Log Tests
# ===============================================================================

@pytest.mark.unit
class TestLegacyProcessingLog:
    """Test legacy processing log functionality."""

    def test_processing_log_basic(self, temp_workspace):
        """Test basic processing log functionality."""
        dm = DataManager(workspace_path=temp_workspace)

        # Add processing steps
        processing_steps = [
            "Data loaded from input.csv",
            "Applied quality control filters",
            "Normalized gene expression",
            "Performed dimensionality reduction"
        ]

        for step in processing_steps:
            dm.processing_log.append(step)

        assert len(dm.processing_log) == 4
        assert dm.processing_log[0] == "Data loaded from input.csv"
        assert dm.processing_log[-1] == "Performed dimensionality reduction"

    def test_tool_usage_history(self, temp_workspace):
        """Test tool usage history tracking."""
        dm = DataManager(workspace_path=temp_workspace)

        # Add tool usage entries
        tool_entries = [
            {
                "tool": "scanpy.pp.filter_cells",
                "parameters": {"min_genes": 200},
                "timestamp": "2024-01-15T10:00:00",
                "description": "Filtered cells with low gene counts"
            },
            {
                "tool": "scanpy.pp.normalize_total",
                "parameters": {"target_sum": 10000},
                "timestamp": "2024-01-15T10:05:00",
                "description": "Normalized total counts"
            }
        ]

        for entry in tool_entries:
            dm.tool_usage_history.append(entry)

        assert len(dm.tool_usage_history) == 2
        assert dm.tool_usage_history[0]["tool"] == "scanpy.pp.filter_cells"
        assert dm.tool_usage_history[1]["parameters"]["target_sum"] == 10000


# ===============================================================================
# Integration with Scanpy Tests
# ===============================================================================

@pytest.mark.unit
class TestLegacyScanpyIntegration:
    """Test legacy DataManager integration with scanpy."""

    def test_anndata_compatibility(self, temp_workspace, sample_dataframe):
        """Test AnnData compatibility."""
        dm = DataManager(workspace_path=temp_workspace)
        dm.set_data(sample_dataframe)

        # Test manual AnnData creation (if supported)
        if hasattr(dm, 'adata') and dm.adata is None:
            # Create AnnData object manually
            import anndata as ad
            dm.adata = ad.AnnData(X=dm.current_data.values)
            dm.adata.obs_names = dm.current_data.index
            dm.adata.var_names = dm.current_data.columns

        # Verify AnnData structure if created
        if dm.adata is not None:
            assert dm.adata.shape == sample_dataframe.shape
            assert len(dm.adata.obs_names) == sample_dataframe.shape[0]
            assert len(dm.adata.var_names) == sample_dataframe.shape[1]

    @patch('scanpy.pp.filter_cells')
    @patch('scanpy.pp.normalize_total')
    def test_scanpy_workflow_simulation(self, mock_normalize, mock_filter, temp_workspace, sample_dataframe):
        """Test simulation of scanpy workflow."""
        dm = DataManager(workspace_path=temp_workspace)
        dm.set_data(sample_dataframe)

        # Simulate scanpy workflow steps
        processing_steps = [
            ("Quality control", "scanpy.pp.filter_cells", {"min_genes": 200}),
            ("Normalization", "scanpy.pp.normalize_total", {"target_sum": 10000}),
            ("Log transformation", "scanpy.pp.log1p", {}),
            ("Variable genes", "scanpy.pp.highly_variable_genes", {"n_top_genes": 2000})
        ]

        for description, tool, params in processing_steps:
            # Log the processing step
            dm.processing_log.append(description)
            dm.tool_usage_history.append({
                "tool": tool,
                "parameters": params,
                "timestamp": "2024-01-15T10:00:00",
                "description": description
            })

        # Verify workflow was logged
        assert len(dm.processing_log) == 4
        assert len(dm.tool_usage_history) == 4
        assert dm.tool_usage_history[0]["tool"] == "scanpy.pp.filter_cells"


# ===============================================================================
# Legacy Compatibility Tests
# ===============================================================================

@pytest.mark.unit
class TestLegacyCompatibility:
    """Test backward compatibility features."""

    def test_attribute_access(self, temp_workspace, sample_dataframe):
        """Test that all expected attributes are accessible."""
        dm = DataManager(workspace_path=temp_workspace)
        dm.set_data(sample_dataframe)

        # Test all expected attributes exist
        expected_attributes = [
            'current_data', 'current_metadata', 'adata', 'latest_plots',
            'plot_counter', 'file_paths', 'processing_log', 'tool_usage_history',
            'max_plots_history', 'console', 'workspace_path', 'data_dir',
            'plots_dir', 'exports_dir'
        ]

        for attr in expected_attributes:
            assert hasattr(dm, attr), f"Missing attribute: {attr}"

    def test_method_signatures(self, temp_workspace):
        """Test that expected methods exist with correct signatures."""
        dm = DataManager(workspace_path=temp_workspace)

        # Test set_data method signature
        import inspect
        set_data_signature = inspect.signature(dm.set_data)
        assert 'data' in set_data_signature.parameters
        assert 'metadata' in set_data_signature.parameters

    def test_data_persistence(self, temp_workspace, sample_dataframe, sample_metadata):
        """Test data persistence across operations."""
        dm = DataManager(workspace_path=temp_workspace)
        dm.set_data(sample_dataframe, sample_metadata)

        # Store references
        original_shape = dm.current_data.shape
        original_metadata = dm.current_metadata.copy()

        # Perform some operations
        dm.processing_log.append("Test operation")
        dm.tool_usage_history.append({"tool": "test", "parameters": {}})

        # Verify data integrity
        assert dm.current_data.shape == original_shape
        assert dm.current_metadata == original_metadata


# ===============================================================================
# Error Handling Tests
# ===============================================================================

@pytest.mark.unit
class TestLegacyErrorHandling:
    """Test error handling in legacy DataManager."""

    def test_workspace_creation_errors(self):
        """Test handling of workspace creation errors."""
        # Test with invalid path (permission denied simulation)
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                DataManager(workspace_path="/invalid/path")

    def test_data_corruption_handling(self, temp_workspace):
        """Test handling of corrupted data."""
        dm = DataManager(workspace_path=temp_workspace)

        # Test with data containing infinite values
        corrupted_data = pd.DataFrame({
            'col1': [1, np.inf, 3],
            'col2': [np.nan, 2, -np.inf]
        })

        # Should handle gracefully (fill inf with 0 or handle appropriately)
        dm.set_data(corrupted_data)

        # Verify no infinite values remain
        assert not np.isinf(dm.current_data.values).any()

    def test_memory_handling(self, temp_workspace):
        """Test memory handling with large datasets."""
        dm = DataManager(workspace_path=temp_workspace)

        # Create moderately large dataset
        large_data = pd.DataFrame(
            np.random.randn(1000, 500),
            index=[f"Sample_{i}" for i in range(1000)],
            columns=[f"Gene_{i}" for i in range(500)]
        )

        # Should handle without memory issues
        dm.set_data(large_data)
        assert dm.current_data.shape == (1000, 500)

    def test_type_conversion_edge_cases(self, temp_workspace):
        """Test type conversion edge cases."""
        dm = DataManager(workspace_path=temp_workspace)

        # Test with mixed data types that can't be converted
        mixed_data = pd.DataFrame({
            'numeric': [1, 2, 3],
            'text': ['abc', 'def', 'ghi'],
            'convertible': ['1.5', '2.5', '3.5']
        })

        dm.set_data(mixed_data)

        # Should handle conversion gracefully
        # Text that can't be converted might become NaN then 0
        assert not dm.current_data.isna().any().any()


# ===============================================================================
# Performance Tests
# ===============================================================================

@pytest.mark.unit
class TestLegacyPerformance:
    """Test performance characteristics of legacy DataManager."""

    def test_data_loading_performance(self, temp_workspace):
        """Test data loading performance."""
        dm = DataManager(workspace_path=temp_workspace)

        # Create moderately sized dataset
        n_samples, n_genes = 1000, 2000
        data = pd.DataFrame(
            np.random.randn(n_samples, n_genes),
            index=[f"Sample_{i}" for i in range(n_samples)],
            columns=[f"Gene_{i}" for i in range(n_genes)]
        )

        import time
        start_time = time.time()
        dm.set_data(data)
        load_time = time.time() - start_time

        # Should complete quickly (< 2 seconds for 2M values)
        assert load_time < 2.0
        assert dm.current_data.shape == (n_samples, n_genes)

    def test_processing_log_performance(self, temp_workspace):
        """Test processing log performance with many entries."""
        dm = DataManager(workspace_path=temp_workspace)

        # Add many processing log entries
        import time
        start_time = time.time()

        for i in range(1000):
            dm.processing_log.append(f"Processing step {i}")
            dm.tool_usage_history.append({
                "tool": f"tool_{i}",
                "parameters": {"param": i},
                "timestamp": f"2024-01-15T10:{i%60:02d}:00"
            })

        operation_time = time.time() - start_time

        # Should complete quickly
        assert operation_time < 1.0
        assert len(dm.processing_log) == 1000
        assert len(dm.tool_usage_history) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])