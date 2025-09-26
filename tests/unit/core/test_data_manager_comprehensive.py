"""
Comprehensive integration and stress tests for DataManagerV2.

This module provides extensive testing for DataManagerV2 covering:
- Multi-modal data orchestration
- Named biological datasets management
- Metadata store functionality
- Tool usage history and provenance tracking
- Backend/adapter registry and schema validation
- Workspace restoration features (v2.2+)
- Session persistence and lazy loading
- Error handling and edge cases
- Memory efficiency with large datasets
- Thread safety and concurrent access
- Data integrity and performance testing
"""

import json
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch, call
import concurrent.futures

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import plotly.graph_objects as go
from pytest_mock import MockerFixture

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.interfaces.adapter import IModalityAdapter
from lobster.core.interfaces.backend import IDataBackend
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.provenance import ProvenanceTracker

from tests.mock_data.factories import (
    SingleCellDataFactory,
    BulkRNASeqDataFactory,
    ProteomicsDataFactory,
    MultiModalDataFactory
)
from tests.mock_data.base import (
    MEDIUM_DATASET_CONFIG,
    SMALL_DATASET_CONFIG,
    LARGE_DATASET_CONFIG,
    HIGH_NOISE_CONFIG,
    LOW_QUALITY_CONFIG,
    BATCH_EFFECT_CONFIG
)


# ===============================================================================
# Test Fixtures
# ===============================================================================

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp(prefix="lobster_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_adapter():
    """Create a comprehensive mock adapter."""
    adapter = Mock(spec=IModalityAdapter)
    adapter.get_modality_name.return_value = "test_modality"
    adapter.get_supported_formats.return_value = ["csv", "h5ad", "xlsx"]
    adapter.get_schema.return_value = {
        "required_obs": ["sample_id"],
        "required_var": ["gene_ids"],
        "optional_obs": ["batch", "condition", "cell_type"],
        "optional_var": ["gene_names", "gene_symbols"],
        "layers": ["counts", "logcounts"],
        "obsm": ["X_pca", "X_umap"],
        "uns": ["processing_info", "metadata"]
    }

    # Mock validation result
    validation_result = Mock(spec=ValidationResult)
    validation_result.has_errors = False
    validation_result.has_warnings = False
    validation_result.errors = []
    validation_result.warnings = []
    adapter.validate.return_value = validation_result

    # Mock quality metrics
    adapter.get_quality_metrics.return_value = {
        "n_obs": 1000,
        "n_vars": 2000,
        "mean_counts_per_cell": 5000,
        "median_genes_per_cell": 1200,
        "total_counts": 5000000
    }

    return adapter


@pytest.fixture
def mock_backend():
    """Create a comprehensive mock backend."""
    backend = Mock(spec=IDataBackend)
    backend.get_storage_info.return_value = {
        "backend_type": "test_backend",
        "supports_multimodal": True,
        "compression": ["gzip", "lzf"],
        "max_file_size": "10GB",
        "supported_formats": ["h5ad", "zarr"]
    }
    backend.save.return_value = None
    backend.load.return_value = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    backend.exists.return_value = True
    backend.delete.return_value = True
    return backend


@pytest.fixture
def sample_datasets():
    """Create sample datasets for testing."""
    return {
        "single_cell": SingleCellDataFactory(config=MEDIUM_DATASET_CONFIG),
        "bulk_rna": BulkRNASeqDataFactory(config=MEDIUM_DATASET_CONFIG),
        "proteomics": ProteomicsDataFactory(config=MEDIUM_DATASET_CONFIG),
        "multimodal": MultiModalDataFactory(config=MEDIUM_DATASET_CONFIG)
    }


# ===============================================================================
# Multi-Modal Data Orchestration Tests
# ===============================================================================

@pytest.mark.unit
class TestMultiModalOrchestration:
    """Test multi-modal data orchestration capabilities."""

    def test_multi_modal_data_loading(self, temp_workspace, mock_adapter, sample_datasets):
        """Test loading multiple data modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Register adapters for different modalities
        for modality_type in ["transcriptomics_single_cell", "transcriptomics_bulk", "proteomics_ms"]:
            dm.register_adapter(modality_type, mock_adapter)

        # Load different data types
        mock_adapter.from_source.side_effect = [
            sample_datasets["single_cell"],
            sample_datasets["bulk_rna"],
            sample_datasets["proteomics"]
        ]

        dm.load_modality("sc_rna", "path/sc.h5ad", "transcriptomics_single_cell")
        dm.load_modality("bulk_rna", "path/bulk.csv", "transcriptomics_bulk")
        dm.load_modality("proteomics", "path/prot.xlsx", "proteomics_ms")

        # Verify all modalities loaded
        assert len(dm.modalities) == 3
        assert "sc_rna" in dm.modalities
        assert "bulk_rna" in dm.modalities
        assert "proteomics" in dm.modalities

        # Verify workspace status reflects multi-modal data
        status = dm.get_workspace_status()
        assert status["modalities_loaded"] == 3
        assert len(status["modality_details"]) == 3

    def test_cross_modality_operations(self, temp_workspace, sample_datasets):
        """Test operations across multiple modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Add multi-modal datasets
        multimodal_data = sample_datasets["multimodal"]
        dm.modalities["rna"] = multimodal_data["rna"]
        dm.modalities["protein"] = multimodal_data["protein"]

        # Test quality metrics across modalities
        with patch.object(dm, '_match_modality_to_adapter') as mock_match:
            mock_match.return_value = None  # Force base adapter usage

            with patch('lobster.core.adapters.base.BaseAdapter') as mock_base:
                mock_base_instance = Mock()
                mock_base.return_value = mock_base_instance
                mock_base_instance.get_quality_metrics.return_value = {"basic": "metrics"}

                all_metrics = dm.get_quality_metrics()

                assert "rna" in all_metrics
                assert "protein" in all_metrics
                assert mock_base.call_count == 2

    def test_mudata_integration(self, temp_workspace, sample_datasets):
        """Test MuData integration for multi-modal data."""
        with patch('lobster.core.data_manager_v2.MUDATA_AVAILABLE', True), \
             patch('lobster.core.data_manager_v2.mudata') as mock_mudata:

            mock_mdata = Mock()
            mock_mudata.MuData.return_value = mock_mdata

            dm = DataManagerV2(workspace_path=temp_workspace)
            multimodal_data = sample_datasets["multimodal"]
            dm.modalities.update(multimodal_data)

            # Test MuData conversion
            result = dm.to_mudata(modalities=["rna", "protein"])

            assert result is mock_mdata
            mock_mudata.MuData.assert_called_once()

            # Verify correct modalities were passed
            call_args = mock_mudata.MuData.call_args[0][0]
            assert set(call_args.keys()) == {"rna", "protein"}

    def test_modality_cross_validation(self, temp_workspace, sample_datasets):
        """Test validation across multiple modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Add datasets with different validation requirements
        dm.modalities["good_data"] = sample_datasets["single_cell"]
        dm.modalities["bulk_data"] = sample_datasets["bulk_rna"]

        with patch.object(dm, '_match_modality_to_adapter') as mock_match:
            mock_match.return_value = None

            with patch('lobster.core.adapters.base.BaseAdapter') as mock_base:
                mock_base_instance = Mock()
                mock_base.return_value = mock_base_instance

                # Mock different validation results
                good_result = Mock(spec=ValidationResult)
                good_result.has_errors = False
                good_result.has_warnings = False

                bulk_result = Mock(spec=ValidationResult)
                bulk_result.has_errors = False
                bulk_result.has_warnings = True
                bulk_result.warnings = ["Missing optional field"]

                mock_base_instance._validate_basic_structure.side_effect = [good_result, bulk_result]

                results = dm.validate_modalities()

                assert len(results) == 2
                assert results["good_data"] is good_result
                assert results["bulk_data"] is bulk_result


# ===============================================================================
# Named Biological Datasets Management Tests
# ===============================================================================

@pytest.mark.unit
class TestNamedDatasetsManagement:
    """Test professional naming conventions and dataset management."""

    def test_professional_naming_convention(self, temp_workspace, mock_backend):
        """Test that datasets follow professional naming conventions."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_backend("h5ad", mock_backend)

        # Test GEO dataset naming
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["geo_gse123456"] = test_data

        # Test professional processing step naming
        processing_steps = [
            "quality_assessed",
            "filtered_normalized",
            "doublets_detected",
            "clustered",
            "markers_identified",
            "annotated",
            "pseudobulk_aggregated"
        ]

        for step in processing_steps:
            with patch('lobster.utils.file_naming.BioinformaticsFileNaming') as mock_naming:
                mock_naming.generate_filename.return_value = f"geo_gse123456_{step}.h5ad"
                mock_naming.generate_metadata_filename.return_value = f"geo_gse123456_{step}_metadata.json"
                mock_naming.suggest_next_step.return_value = "next_step"
                mock_naming.get_processing_step_order.return_value = 1

                with patch.object(dm, 'save_modality') as mock_save:
                    mock_save.return_value = f"/path/to/geo_gse123456_{step}.h5ad"

                    result = dm.save_processed_data(
                        processing_step=step,
                        data_source="GEO",
                        dataset_id="GSE123456"
                    )

                    assert result is not None
                    mock_naming.generate_filename.assert_called_with(
                        data_source="GEO",
                        dataset_id="GSE123456",
                        processing_step=step
                    )

    def test_dataset_lineage_tracking(self, temp_workspace, sample_datasets):
        """Test tracking of dataset processing lineage."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Start with raw data
        dm.modalities["geo_gse123456"] = sample_datasets["single_cell"]

        # Simulate processing pipeline
        processing_chain = [
            ("geo_gse123456_quality_assessed", "quality_control"),
            ("geo_gse123456_filtered_normalized", "preprocessing"),
            ("geo_gse123456_clustered", "clustering"),
            ("geo_gse123456_annotated", "annotation")
        ]

        for dataset_name, tool in processing_chain:
            # Create derived dataset
            dm.modalities[dataset_name] = sample_datasets["single_cell"].copy()

            # Log the processing step
            dm.log_tool_usage(
                tool_name=tool,
                parameters={"source_dataset": list(dm.modalities.keys())[-2]},
                description=f"Applied {tool} to create {dataset_name}"
            )

        # Verify lineage in tool usage history
        assert len(dm.tool_usage_history) == 4
        assert all("source_dataset" in entry["parameters"] for entry in dm.tool_usage_history)

        # Verify naming consistency
        for dataset_name, _ in processing_chain:
            assert dataset_name in dm.modalities
            assert dataset_name.startswith("geo_gse123456_")

    def test_dataset_metadata_association(self, temp_workspace, sample_datasets):
        """Test metadata association with named datasets."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Store metadata for different dataset types
        datasets_metadata = {
            "geo_gse123456": {
                "title": "Single-cell RNA-seq of human brain",
                "organism": "Homo sapiens",
                "tissue": "brain",
                "technology": "10x Genomics",
                "n_samples": 8
            },
            "geo_gse789012": {
                "title": "Bulk RNA-seq of liver samples",
                "organism": "Mus musculus",
                "tissue": "liver",
                "technology": "Illumina HiSeq",
                "n_samples": 24
            }
        }

        for dataset_id, metadata in datasets_metadata.items():
            dm.store_metadata(dataset_id, metadata, {"validated": True})
            dm.modalities[dataset_id] = sample_datasets["single_cell"]

        # Test metadata retrieval
        for dataset_id in datasets_metadata:
            retrieved_metadata = dm.get_stored_metadata(dataset_id)
            assert retrieved_metadata is not None
            assert retrieved_metadata["metadata"]["title"] == datasets_metadata[dataset_id]["title"]
            assert retrieved_metadata["validation"]["validated"] is True

        # Test dataset listing
        stored_datasets = dm.list_stored_datasets()
        assert sorted(stored_datasets) == ["geo_gse123456", "geo_gse789012"]


# ===============================================================================
# Workspace Restoration and Session Management Tests
# ===============================================================================

@pytest.mark.unit
class TestWorkspaceRestoration:
    """Test workspace restoration features (v2.2+)."""

    def test_workspace_scanning(self, temp_workspace, sample_datasets):
        """Test automatic workspace scanning for available datasets."""
        dm = DataManagerV2(workspace_path=temp_workspace, auto_scan=False)

        # Create some h5ad files in data directory
        test_files = {
            "dataset1.h5ad": sample_datasets["single_cell"],
            "dataset2.h5ad": sample_datasets["bulk_rna"],
            "processed_data.h5ad": sample_datasets["proteomics"]
        }

        for filename, adata in test_files.items():
            filepath = dm.data_dir / filename
            adata.write_h5ad(filepath)

        # Manually trigger workspace scanning
        dm._scan_workspace()

        # Verify datasets were discovered
        assert len(dm.available_datasets) == 3
        assert "dataset1" in dm.available_datasets
        assert "dataset2" in dm.available_datasets
        assert "processed_data" in dm.available_datasets

        # Verify metadata extraction
        for name in ["dataset1", "dataset2", "processed_data"]:
            dataset_info = dm.available_datasets[name]
            assert "path" in dataset_info
            assert "size_mb" in dataset_info
            assert "shape" in dataset_info
            assert "modified" in dataset_info
            assert dataset_info["type"] == "h5ad"

    def test_lazy_dataset_loading(self, temp_workspace, sample_datasets):
        """Test lazy loading of datasets from workspace."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Pre-populate workspace with saved datasets
        test_datasets = {
            "experiment_1": sample_datasets["single_cell"],
            "experiment_2": sample_datasets["bulk_rna"]
        }

        for name, adata in test_datasets.items():
            filepath = dm.data_dir / f"{name}.h5ad"
            adata.write_h5ad(filepath)

        # Trigger workspace scan
        dm._scan_workspace()

        # Initially no datasets loaded in memory
        assert len(dm.modalities) == 0
        assert len(dm.available_datasets) == 2

        # Test lazy loading
        success = dm.load_dataset("experiment_1")
        assert success is True
        assert "experiment_1" in dm.modalities
        assert len(dm.modalities) == 1

        # Test loading non-existent dataset
        success = dm.load_dataset("nonexistent")
        assert success is False

    def test_session_persistence(self, temp_workspace, sample_datasets):
        """Test session persistence and restoration."""
        # Create initial session
        dm1 = DataManagerV2(workspace_path=temp_workspace)
        dm1.modalities["test_data"] = sample_datasets["single_cell"]
        dm1.log_tool_usage("test_tool", {"param": "value"})

        # Update session file
        dm1._update_session_file("test_action")

        # Verify session file exists
        assert dm1.session_file.exists()

        # Create new manager instance (simulating restart)
        dm2 = DataManagerV2(workspace_path=temp_workspace)

        # Verify session data was loaded
        assert dm2.session_data is not None
        assert "session_id" in dm2.session_data
        assert "created_at" in dm2.session_data
        assert "lobster_version" in dm2.session_data

    def test_pattern_based_restoration(self, temp_workspace, sample_datasets):
        """Test pattern-based dataset restoration."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Create datasets with different patterns
        datasets = {
            "geo_gse123456_raw": sample_datasets["single_cell"],
            "geo_gse123456_processed": sample_datasets["single_cell"],
            "geo_gse789012_raw": sample_datasets["bulk_rna"],
            "experiment_control": sample_datasets["proteomics"]
        }

        # Save datasets to workspace
        for name, adata in datasets.items():
            filepath = dm.data_dir / f"{name}.h5ad"
            adata.write_h5ad(filepath)

        dm._scan_workspace()

        # Test pattern-based restoration
        result = dm.restore_session(pattern="geo_gse123456*", max_size_mb=1000)

        assert len(result["restored"]) == 2
        assert "geo_gse123456_raw" in result["restored"]
        assert "geo_gse123456_processed" in result["restored"]

        # Test "all" pattern with size limit
        dm.modalities.clear()
        result = dm.restore_session(pattern="all", max_size_mb=50)  # Small limit

        # Should respect size limit
        assert len(result["restored"]) < len(datasets)
        assert len(result["skipped"]) > 0

    def test_session_metadata_tracking(self, temp_workspace, sample_datasets):
        """Test comprehensive session metadata tracking."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Add data and perform operations
        dm.modalities["test_dataset"] = sample_datasets["single_cell"]
        dm.log_tool_usage("quality_control", {"min_genes": 200})
        dm.log_tool_usage("normalization", {"method": "log1p"})

        # Update session
        dm._update_session_file("test_operation")

        # Load session data
        with open(dm.session_file, 'r') as f:
            session_data = json.load(f)

        # Verify comprehensive tracking
        assert "session_id" in session_data
        assert "created_at" in session_data
        assert "last_modified" in session_data
        assert "lobster_version" in session_data
        assert "active_modalities" in session_data
        assert "workspace_stats" in session_data
        assert "command_history" in session_data

        # Verify modality tracking
        assert "test_dataset" in session_data["active_modalities"]

        # Verify command history
        assert len(session_data["command_history"]) == 2
        assert session_data["command_history"][0]["command"] == "quality_control"


# ===============================================================================
# Tool Usage History and Provenance Tracking Tests
# ===============================================================================

@pytest.mark.unit
class TestProvenanceTracking:
    """Test W3C-PROV compliant provenance tracking."""

    def test_provenance_initialization(self, temp_workspace):
        """Test provenance tracker initialization and configuration."""
        # With provenance enabled
        dm_with_prov = DataManagerV2(workspace_path=temp_workspace, enable_provenance=True)
        assert dm_with_prov.provenance is not None
        assert isinstance(dm_with_prov.provenance, ProvenanceTracker)

        # With provenance disabled
        dm_without_prov = DataManagerV2(workspace_path=temp_workspace, enable_provenance=False)
        assert dm_without_prov.provenance is None

    def test_comprehensive_tool_usage_logging(self, temp_workspace, sample_datasets):
        """Test comprehensive tool usage logging for reproducibility."""
        dm = DataManagerV2(workspace_path=temp_workspace, enable_provenance=True)

        # Test various tool usage scenarios
        tool_operations = [
            {
                "tool": "data_loading",
                "parameters": {"file_path": "/path/to/data.h5ad", "format": "h5ad"},
                "description": "Loaded single-cell RNA-seq data"
            },
            {
                "tool": "quality_control",
                "parameters": {"min_genes": 200, "max_genes": 5000, "mt_pct_threshold": 20},
                "description": "Applied quality control filters"
            },
            {
                "tool": "normalization",
                "parameters": {"method": "log1p", "target_sum": 10000},
                "description": "Normalized gene expression data"
            },
            {
                "tool": "clustering",
                "parameters": {"algorithm": "leiden", "resolution": 0.5, "n_neighbors": 15},
                "description": "Performed cell clustering"
            }
        ]

        # Log all operations
        for operation in tool_operations:
            dm.log_tool_usage(**operation)

        # Verify logging
        assert len(dm.tool_usage_history) == 4

        # Verify each logged operation
        for i, operation in enumerate(tool_operations):
            logged_op = dm.tool_usage_history[i]
            assert logged_op["tool"] == operation["tool"]
            assert logged_op["parameters"] == operation["parameters"]
            assert logged_op["description"] == operation["description"]
            assert "timestamp" in logged_op

    def test_provenance_entity_creation(self, temp_workspace, mock_adapter, sample_datasets):
        """Test W3C-PROV entity creation and tracking."""
        with patch('lobster.core.data_manager_v2.ProvenanceTracker') as mock_prov_class:
            mock_prov = Mock(spec=ProvenanceTracker)
            mock_prov.activities = {}
            mock_prov.entities = {}
            mock_prov.agents = {}
            mock_prov.create_entity.return_value = "entity_123"
            mock_prov.add_to_anndata.return_value = sample_datasets["single_cell"]
            mock_prov_class.return_value = mock_prov

            dm = DataManagerV2(workspace_path=temp_workspace, enable_provenance=True)
            dm.register_adapter("test_adapter", mock_adapter)

            mock_adapter.from_source.return_value = sample_datasets["single_cell"]

            # Load modality (should create provenance entities)
            dm.load_modality("test_data", "/path/to/data", "test_adapter")

            # Verify entity creation
            mock_prov.create_entity.assert_called()
            mock_prov.add_to_anndata.assert_called()

    def test_provenance_export(self, temp_workspace):
        """Test provenance information export."""
        with patch('lobster.core.data_manager_v2.ProvenanceTracker') as mock_prov_class:
            mock_prov = Mock(spec=ProvenanceTracker)
            mock_prov.to_dict.return_value = {
                "activities": {"act_1": {"type": "data_loading"}},
                "entities": {"ent_1": {"type": "dataset"}},
                "agents": {"agent_1": {"type": "software"}}
            }
            mock_prov_class.return_value = mock_prov

            dm = DataManagerV2(workspace_path=temp_workspace, enable_provenance=True)

            # Export provenance
            export_path = dm.export_provenance("provenance_export.json")

            # Verify export
            assert Path(export_path).exists()
            mock_prov.to_dict.assert_called_once()

            # Verify export content
            with open(export_path, 'r') as f:
                exported_data = json.load(f)

            assert "activities" in exported_data
            assert "entities" in exported_data
            assert "agents" in exported_data

    def test_tool_usage_history_limits(self, temp_workspace):
        """Test tool usage history management with large numbers of operations."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Generate many tool usage entries
        for i in range(200):
            dm.log_tool_usage(
                tool_name=f"tool_{i}",
                parameters={"iteration": i, "data": f"dataset_{i}"},
                description=f"Operation {i}"
            )

        # Verify all entries are stored (no automatic limiting in current implementation)
        assert len(dm.tool_usage_history) == 200

        # Test session file handling with large history
        dm._update_session_file("large_history_test")

        # Verify session file contains recent commands (limited to 50)
        with open(dm.session_file, 'r') as f:
            session_data = json.load(f)

        assert len(session_data["command_history"]) == 50
        # Should contain the most recent commands
        assert session_data["command_history"][-1]["command"] == "tool_199"


# ===============================================================================
# Backend/Adapter Registry and Schema Validation Tests
# ===============================================================================

@pytest.mark.unit
class TestBackendAdapterRegistry:
    """Test backend and adapter registry with schema validation."""

    def test_comprehensive_backend_registration(self, temp_workspace, mock_backend):
        """Test comprehensive backend registration and management."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Test multiple backend registration
        backends = {
            "h5ad_fast": mock_backend,
            "h5ad_compressed": mock_backend,
            "zarr_backend": mock_backend,
            "custom_backend": mock_backend
        }

        for name, backend in backends.items():
            dm.register_backend(name, backend)

        # Verify all backends registered
        assert len(dm.backends) >= len(backends)  # Including defaults
        for name in backends:
            assert name in dm.backends

        # Test backend info retrieval
        backend_info = dm.get_backend_info()
        for name in backends:
            assert name in backend_info
            mock_backend.get_storage_info.assert_called()

    def test_adapter_schema_validation(self, temp_workspace):
        """Test adapter schema validation integration."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Create adapters with different schemas
        adapters_configs = {
            "strict_transcriptomics": {
                "required_obs": ["cell_id", "sample_id", "batch"],
                "required_var": ["gene_id", "gene_symbol"],
                "optional_obs": ["cell_type", "condition"],
                "optional_var": ["chromosome", "gene_biotype"]
            },
            "flexible_proteomics": {
                "required_obs": ["sample_id"],
                "required_var": ["protein_id"],
                "optional_obs": ["condition", "replicate"],
                "optional_var": ["protein_name", "molecular_weight"]
            }
        }

        for adapter_name, schema in adapters_configs.items():
            mock_adapter = Mock(spec=IModalityAdapter)
            mock_adapter.get_schema.return_value = schema
            mock_adapter.get_modality_name.return_value = adapter_name
            mock_adapter.get_supported_formats.return_value = ["csv", "xlsx", "h5ad"]

            dm.register_adapter(adapter_name, mock_adapter)

        # Test adapter info includes schema information
        adapter_info = dm.get_adapter_info()

        for adapter_name, expected_schema in adapters_configs.items():
            assert adapter_name in adapter_info
            assert adapter_info[adapter_name]["schema"] == expected_schema

    def test_validation_result_handling(self, temp_workspace, mock_adapter, sample_datasets):
        """Test comprehensive validation result handling."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)

        # Test different validation scenarios
        validation_scenarios = [
            {
                "name": "perfect_data",
                "has_errors": False,
                "has_warnings": False,
                "errors": [],
                "warnings": []
            },
            {
                "name": "warnings_only",
                "has_errors": False,
                "has_warnings": True,
                "errors": [],
                "warnings": ["Missing optional field 'gene_symbols'", "Low expression variance"]
            },
            {
                "name": "errors_present",
                "has_errors": True,
                "has_warnings": False,
                "errors": ["Missing required field 'cell_id'", "Invalid gene identifiers"],
                "warnings": []
            }
        ]

        for scenario in validation_scenarios:
            validation_result = Mock(spec=ValidationResult)
            validation_result.has_errors = scenario["has_errors"]
            validation_result.has_warnings = scenario["has_warnings"]
            validation_result.errors = scenario["errors"]
            validation_result.warnings = scenario["warnings"]

            mock_adapter.validate.return_value = validation_result
            mock_adapter.from_source.return_value = sample_datasets["single_cell"]

            if scenario["has_errors"]:
                # Should raise exception for errors
                with pytest.raises(ValueError, match="Validation failed"):
                    dm.load_modality(scenario["name"], "source", "test_adapter", validate=True)
            else:
                # Should succeed for warnings or no issues
                adata = dm.load_modality(scenario["name"], "source", "test_adapter", validate=True)
                assert adata is not None
                assert scenario["name"] in dm.modalities

    def test_schema_compliance_checking(self, temp_workspace, sample_datasets):
        """Test schema compliance checking for different data types."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Add datasets
        dm.modalities["single_cell_data"] = sample_datasets["single_cell"]
        dm.modalities["bulk_rna_data"] = sample_datasets["bulk_rna"]
        dm.modalities["proteomics_data"] = sample_datasets["proteomics"]

        # Test validation with adapter matching
        with patch.object(dm, '_match_modality_to_adapter') as mock_match:
            # Mock different adapter matches
            adapter_matches = {
                "single_cell_data": "transcriptomics_single_cell",
                "bulk_rna_data": "transcriptomics_bulk",
                "proteomics_data": "proteomics_ms"
            }

            mock_match.side_effect = lambda name: adapter_matches.get(name)

            # Mock adapters with different validation results
            for adapter_name in adapter_matches.values():
                mock_adapter = Mock(spec=IModalityAdapter)
                validation_result = Mock(spec=ValidationResult)
                validation_result.has_errors = False
                validation_result.has_warnings = False
                mock_adapter.validate.return_value = validation_result
                dm.register_adapter(adapter_name, mock_adapter)

            # Validate all modalities
            results = dm.validate_modalities(strict=True)

            assert len(results) == 3
            assert all(not result.has_errors for result in results.values())


# ===============================================================================
# Memory Efficiency and Performance Tests
# ===============================================================================

@pytest.mark.unit
class TestMemoryEfficiencyPerformance:
    """Test memory efficiency and performance with large datasets."""

    def test_large_dataset_memory_handling(self, temp_workspace):
        """Test memory efficiency with large datasets."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Create large mock dataset
        large_dataset = Mock()
        large_dataset.shape = (100000, 20000)  # 100k cells, 20k genes
        large_dataset.n_obs = 100000
        large_dataset.n_vars = 20000

        # Mock X matrix with realistic memory footprint
        large_dataset.X = Mock()
        large_dataset.X.nbytes = 8 * 100000 * 20000  # 8 bytes per float64
        large_dataset.X.shape = (100000, 20000)
        large_dataset.X.dtype = np.float64

        # Mock obs and var DataFrames
        large_dataset.obs = pd.DataFrame(index=range(100000))
        large_dataset.var = pd.DataFrame(index=range(20000))
        large_dataset.layers = {}
        large_dataset.obsm = {}
        large_dataset.uns = {}

        dm.modalities["large_dataset"] = large_dataset

        # Test memory calculation methods
        memory_usage = dm._get_safe_memory_usage(large_dataset.X)
        assert "GB" in memory_usage or "MB" in memory_usage

        # Test data summary with large dataset
        summary = dm.get_data_summary()
        assert summary["modality_name"] == "large_dataset"
        assert summary["shape"] == (100000, 20000)

        # Test workspace status
        status = dm.get_workspace_status()
        assert status["modalities_loaded"] == 1
        assert status["modality_details"]["large_dataset"]["shape"] == (100000, 20000)

    def test_sparse_matrix_handling(self, temp_workspace):
        """Test efficient handling of sparse matrices."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Create mock sparse matrix
        sparse_matrix = Mock()
        sparse_matrix.nnz = 1000000  # 1M non-zero entries
        sparse_matrix.shape = (50000, 20000)
        sparse_matrix.data = Mock()
        sparse_matrix.data.nbytes = 1000000 * 4  # 4 bytes per float32
        sparse_matrix.indices = Mock()
        sparse_matrix.indices.nbytes = 1000000 * 4  # 4 bytes per int32
        sparse_matrix.indptr = Mock()
        sparse_matrix.indptr.nbytes = 50001 * 4  # (n_rows + 1) * 4 bytes

        # Test sparse matrix detection
        assert dm._is_sparse_matrix(sparse_matrix) is True

        # Test memory calculation for sparse matrices
        memory_usage = dm._get_safe_memory_usage(sparse_matrix)
        assert "sparse" in memory_usage.lower()
        assert "MB" in memory_usage

    def test_concurrent_access_safety(self, temp_workspace, sample_datasets):
        """Test thread safety for concurrent operations."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Test concurrent modality additions
        def add_modality(thread_id):
            try:
                dataset = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                dm.modalities[f"dataset_{thread_id}"] = dataset
                dm.log_tool_usage(f"tool_{thread_id}", {"thread": thread_id})
                return True
            except Exception as e:
                print(f"Thread {thread_id} error: {e}")
                return False

        # Run concurrent operations
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(add_modality, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify all operations succeeded
        assert all(results)
        assert len(dm.modalities) == 10
        assert len(dm.tool_usage_history) == 10

    def test_memory_pressure_cleanup(self, temp_workspace):
        """Test behavior under memory pressure simulation."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Add multiple datasets to simulate memory usage
        datasets = {}
        for i in range(10):
            mock_data = Mock()
            mock_data.shape = (5000, 2000)
            mock_data.X = Mock()
            mock_data.X.nbytes = 4 * 5000 * 2000  # 4 bytes per float32
            datasets[f"dataset_{i}"] = mock_data
            dm.modalities[f"dataset_{i}"] = mock_data

        # Test selective cleanup
        initial_count = len(dm.modalities)

        # Remove every other dataset
        for i in range(0, 10, 2):
            dm.remove_modality(f"dataset_{i}")

        assert len(dm.modalities) == initial_count - 5

        # Verify remaining datasets are intact
        for i in range(1, 10, 2):
            assert f"dataset_{i}" in dm.modalities

    def test_performance_benchmarking(self, temp_workspace, sample_datasets):
        """Test performance benchmarking for key operations."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Benchmark modality loading
        start_time = time.time()
        for i in range(5):
            dm.modalities[f"dataset_{i}"] = sample_datasets["single_cell"].copy()
        loading_time = time.time() - start_time

        # Should be fast (< 1 second for 5 small datasets)
        assert loading_time < 1.0

        # Benchmark quality metrics calculation
        with patch.object(dm, '_match_modality_to_adapter') as mock_match:
            mock_match.return_value = None

            with patch('lobster.core.adapters.base.BaseAdapter') as mock_base:
                mock_base_instance = Mock()
                mock_base.return_value = mock_base_instance
                mock_base_instance.get_quality_metrics.return_value = {"test": "metrics"}

                start_time = time.time()
                metrics = dm.get_quality_metrics()
                metrics_time = time.time() - start_time

                # Should be fast (< 0.5 seconds)
                assert metrics_time < 0.5
                assert len(metrics) == 5


# ===============================================================================
# Error Handling and Edge Cases Tests
# ===============================================================================

@pytest.mark.unit
class TestErrorHandlingEdgeCases:
    """Test comprehensive error handling and edge cases."""

    def test_workspace_permission_errors(self, temp_workspace):
        """Test handling of workspace permission errors."""
        # Test with read-only workspace
        temp_workspace.chmod(0o444)  # Read-only

        try:
            # Should handle permission error gracefully
            dm = DataManagerV2(workspace_path=temp_workspace)

            # Test operations that require write access
            with pytest.raises((PermissionError, OSError)):
                dm._update_session_file("test")

        finally:
            # Restore permissions for cleanup
            temp_workspace.chmod(0o755)

    def test_corrupted_data_handling(self, temp_workspace, mock_adapter):
        """Test handling of corrupted or invalid data."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)

        # Test with various corruption scenarios
        corruption_scenarios = [
            ("empty_data", Mock(shape=(0, 0))),
            ("invalid_matrix", Mock(X=None)),
            ("missing_obs", Mock(obs=None)),
            ("missing_var", Mock(var=None))
        ]

        for scenario_name, corrupted_data in corruption_scenarios:
            # Mock adapter to return corrupted data
            mock_adapter.from_source.return_value = corrupted_data

            try:
                dm.load_modality(scenario_name, "source", "test_adapter", validate=False)

                # Test that helper methods handle corrupted data gracefully
                shape = dm._get_safe_shape(corrupted_data)
                memory = dm._get_safe_memory_usage(corrupted_data.X if hasattr(corrupted_data, 'X') else None)

                # Should return safe defaults
                assert isinstance(shape, tuple)
                assert isinstance(memory, str)

            except Exception as e:
                # Should be handled gracefully
                assert isinstance(e, (ValueError, AttributeError, TypeError))

    def test_concurrent_modification_safety(self, temp_workspace, sample_datasets):
        """Test safety during concurrent modifications."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Add initial data
        dm.modalities["shared_dataset"] = sample_datasets["single_cell"]

        def modify_modalities(operation_id):
            """Concurrent modification function."""
            try:
                if operation_id % 2 == 0:
                    # Add modality
                    dm.modalities[f"dataset_{operation_id}"] = sample_datasets["single_cell"].copy()
                else:
                    # Try to remove modality (might fail if already removed)
                    try:
                        if f"dataset_{operation_id-1}" in dm.modalities:
                            dm.remove_modality(f"dataset_{operation_id-1}")
                    except ValueError:
                        pass  # Expected if already removed

                # Log operation
                dm.log_tool_usage(f"operation_{operation_id}", {"id": operation_id})
                return True

            except Exception as e:
                print(f"Operation {operation_id} error: {e}")
                return False

        # Run concurrent modifications
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(modify_modalities, i) for i in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Verify system remained stable
        assert len(dm.tool_usage_history) == 20  # All operations logged
        assert len(dm.modalities) >= 1  # At least shared_dataset remains

    def test_resource_exhaustion_simulation(self, temp_workspace):
        """Test behavior during simulated resource exhaustion."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        # Simulate memory exhaustion during large operation
        with patch('lobster.core.data_manager_v2.np.random.negative_binomial') as mock_random:
            mock_random.side_effect = MemoryError("Insufficient memory")

            # Should handle memory error gracefully
            with pytest.raises(MemoryError):
                large_dataset = SingleCellDataFactory(config=LARGE_DATASET_CONFIG)

    def test_invalid_configuration_handling(self, temp_workspace):
        """Test handling of invalid configurations."""
        # Test with invalid backend
        dm = DataManagerV2(workspace_path=temp_workspace)

        with pytest.raises(ValueError, match="Backend 'invalid_backend' not registered"):
            dm.save_modality("nonexistent", "path", backend="invalid_backend")

        # Test with invalid adapter
        with pytest.raises(ValueError, match="Adapter 'invalid_adapter' not registered"):
            dm.load_modality("test", "source", "invalid_adapter")

    def test_data_integrity_verification(self, temp_workspace, sample_datasets):
        """Test data integrity verification across operations."""
        dm = DataManagerV2(workspace_path=temp_workspace)

        original_data = sample_datasets["single_cell"]
        original_shape = original_data.shape
        original_obs_count = len(original_data.obs.columns)
        original_var_count = len(original_data.var.columns)

        # Add to manager
        dm.modalities["integrity_test"] = original_data

        # Perform various operations
        dm.log_tool_usage("test_operation", {"param": "value"})
        status = dm.get_workspace_status()
        summary = dm.get_data_summary()

        # Verify data integrity maintained
        stored_data = dm.get_modality("integrity_test")
        assert stored_data.shape == original_shape
        assert len(stored_data.obs.columns) == original_obs_count
        assert len(stored_data.var.columns) == original_var_count

        # Verify references are consistent
        assert stored_data is original_data  # Should be same object reference


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])