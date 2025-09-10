"""
Comprehensive unit tests for DataManagerV2.

This module provides thorough testing of the DataManagerV2 class, covering all major
functionality areas including initialization, modality management, backend/adapter
registration, quality assessment, workspace management, plot handling, ML integration,
legacy compatibility, and error handling patterns.

Test coverage target: 95%+ with meaningful tests, not just line coverage.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch, mock_open, call

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
from tests.mock_data.base import MEDIUM_DATASET_CONFIG, SMALL_DATASET_CONFIG


# ===============================================================================
# Test Fixtures
# ===============================================================================

@pytest.fixture
def mock_adapter():
    """Create a mock adapter that implements IModalityAdapter interface."""
    adapter = Mock(spec=IModalityAdapter)
    adapter.get_modality_name.return_value = "test_modality"
    adapter.get_supported_formats.return_value = ["csv", "h5ad"]
    adapter.get_schema.return_value = {
        "required_obs": ["sample_id"],
        "required_var": ["gene_ids"],
        "optional_obs": ["batch", "condition"],
        "optional_var": ["gene_names"],
        "layers": ["counts"],
        "obsm": ["X_pca"],
        "uns": ["processing_info"]
    }
    
    # Mock validation result
    validation_result = Mock(spec=ValidationResult)
    validation_result.has_errors = False
    validation_result.has_warnings = False
    validation_result.errors = []
    validation_result.warnings = []
    adapter.validate.return_value = validation_result
    
    return adapter


@pytest.fixture
def mock_backend():
    """Create a mock backend that implements IDataBackend interface."""
    backend = Mock(spec=IDataBackend)
    backend.get_storage_info.return_value = {
        "backend_type": "test_backend",
        "supports_multimodal": False,
        "compression": True
    }
    backend.save.return_value = None
    backend.load.return_value = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    backend.exists.return_value = True
    return backend


@pytest.fixture
def mock_provenance():
    """Create a mock provenance tracker."""
    provenance = Mock(spec=ProvenanceTracker)
    provenance.activities = {}
    provenance.entities = {}
    provenance.agents = {}
    provenance.create_entity.return_value = "test_entity_id"
    provenance.to_dict.return_value = {"test": "provenance"}
    provenance.add_to_anndata.side_effect = lambda x: x  # Return unchanged
    return provenance


# ===============================================================================
# Core Initialization & Setup Tests
# ===============================================================================

@pytest.mark.unit
class TestDataManagerV2Initialization:
    """Test DataManagerV2 initialization and setup functionality."""
    
    def test_init_default_parameters(self, temp_workspace):
        """Test initialization with default parameters."""
        # Test with minimal parameters
        dm = DataManagerV2()
        
        assert dm.default_backend == "h5ad"
        assert dm.enable_provenance is True
        assert dm.console is None
        assert isinstance(dm.workspace_path, Path)
        assert dm.workspace_path.name == ".lobster_workspace"
        
        # Check core storage structures
        assert isinstance(dm.modalities, dict)
        assert len(dm.modalities) == 0
        assert isinstance(dm.metadata_store, dict)
        assert isinstance(dm.tool_usage_history, list)
        assert isinstance(dm.latest_plots, list)
        
        # Check legacy compatibility attributes
        assert dm.current_dataset is None
        assert dm.current_data is None
        assert isinstance(dm.current_metadata, dict)
        assert dm.adata is None

    def test_init_custom_parameters(self, temp_workspace):
        """Test initialization with custom parameters."""
        custom_console = Mock()
        
        dm = DataManagerV2(
            default_backend="mudata",
            workspace_path=temp_workspace,
            enable_provenance=False,
            console=custom_console
        )
        
        assert dm.default_backend == "mudata"
        assert dm.workspace_path == temp_workspace
        assert dm.enable_provenance is False
        assert dm.console is custom_console
        assert dm.provenance is None  # Disabled

    def test_workspace_setup(self, temp_workspace):
        """Test workspace directory structure creation."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Check that workspace directories are created
        assert dm.workspace_path.exists()
        assert dm.data_dir.exists()
        assert dm.exports_dir.exists()
        assert dm.cache_dir.exists()
        
        assert dm.data_dir == temp_workspace / "data"
        assert dm.exports_dir == temp_workspace / "exports"
        assert dm.cache_dir == temp_workspace / "cache"

    @patch('lobster.core.data_manager_v2.MUDATA_BACKEND_AVAILABLE', True)
    def test_default_backends_registration(self, temp_workspace):
        """Test that default backends are registered properly."""
        with patch('lobster.core.data_manager_v2.H5ADBackend') as mock_h5ad, \
             patch('lobster.core.data_manager_v2.MuDataBackend') as mock_mudata:
            
            dm = DataManagerV2(workspace_path=temp_workspace)
            
            # Check that backends were instantiated and registered
            assert "h5ad" in dm.backends
            assert "mudata" in dm.backends
            mock_h5ad.assert_called_once()
            mock_mudata.assert_called_once()

    @patch('lobster.core.data_manager_v2.MUDATA_BACKEND_AVAILABLE', False)
    def test_default_backends_registration_without_mudata(self, temp_workspace):
        """Test backend registration when MuData is not available."""
        with patch('lobster.core.data_manager_v2.H5ADBackend') as mock_h5ad:
            dm = DataManagerV2(workspace_path=temp_workspace)
            
            assert "h5ad" in dm.backends
            assert "mudata" not in dm.backends
            mock_h5ad.assert_called_once()

    def test_default_adapters_registration(self, temp_workspace):
        """Test that default adapters are registered properly."""
        with patch('lobster.core.data_manager_v2.TranscriptomicsAdapter') as mock_transcriptomics, \
             patch('lobster.core.data_manager_v2.ProteomicsAdapter') as mock_proteomics:
            
            dm = DataManagerV2(workspace_path=temp_workspace)
            
            # Check expected adapters are registered
            expected_adapters = [
                "transcriptomics_single_cell",
                "transcriptomics_bulk", 
                "proteomics_ms",
                "proteomics_affinity"
            ]
            
            for adapter_name in expected_adapters:
                assert adapter_name in dm.adapters
            
            # Check that adapter classes were instantiated
            assert mock_transcriptomics.call_count == 2  # single_cell + bulk
            assert mock_proteomics.call_count == 2  # ms + affinity

    def test_provenance_initialization(self, temp_workspace):
        """Test provenance tracker initialization."""
        # With provenance enabled
        dm_with_prov = DataManagerV2(workspace_path=temp_workspace, enable_provenance=True)
        assert dm_with_prov.provenance is not None
        assert isinstance(dm_with_prov.provenance, ProvenanceTracker)
        
        # With provenance disabled
        dm_without_prov = DataManagerV2(workspace_path=temp_workspace, enable_provenance=False)
        assert dm_without_prov.provenance is None


# ===============================================================================
# Backend & Adapter Management Tests
# ===============================================================================

@pytest.mark.unit
class TestBackendAdapterManagement:
    """Test backend and adapter registration and retrieval."""
    
    def test_register_backend_success(self, temp_workspace, mock_backend):
        """Test successful backend registration."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        initial_count = len(dm.backends)
        
        dm.register_backend("test_backend", mock_backend)
        
        assert len(dm.backends) == initial_count + 1
        assert "test_backend" in dm.backends
        assert dm.backends["test_backend"] is mock_backend

    def test_register_backend_duplicate_name(self, temp_workspace, mock_backend):
        """Test that registering duplicate backend name raises error."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_backend("test_backend", mock_backend)
        
        with pytest.raises(ValueError, match="Backend 'test_backend' already registered"):
            dm.register_backend("test_backend", mock_backend)

    def test_register_adapter_success(self, temp_workspace, mock_adapter):
        """Test successful adapter registration."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        initial_count = len(dm.adapters)
        
        dm.register_adapter("test_adapter", mock_adapter)
        
        assert len(dm.adapters) == initial_count + 1
        assert "test_adapter" in dm.adapters
        assert dm.adapters["test_adapter"] is mock_adapter

    def test_register_adapter_duplicate_name(self, temp_workspace, mock_adapter):
        """Test that registering duplicate adapter name raises error.""" 
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)
        
        with pytest.raises(ValueError, match="Adapter 'test_adapter' already registered"):
            dm.register_adapter("test_adapter", mock_adapter)

    def test_get_backend_info(self, temp_workspace, mock_backend):
        """Test getting backend information."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_backend("test_backend", mock_backend)
        
        info = dm.get_backend_info()
        
        assert isinstance(info, dict)
        assert "test_backend" in info
        mock_backend.get_storage_info.assert_called_once()

    def test_get_adapter_info(self, temp_workspace, mock_adapter):
        """Test getting adapter information."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)
        
        info = dm.get_adapter_info()
        
        assert isinstance(info, dict)
        assert "test_adapter" in info
        assert "modality_name" in info["test_adapter"]
        assert "supported_formats" in info["test_adapter"]
        assert "schema" in info["test_adapter"]
        
        mock_adapter.get_modality_name.assert_called_once()
        mock_adapter.get_supported_formats.assert_called_once()
        mock_adapter.get_schema.assert_called_once()


# ===============================================================================
# Modality Management Lifecycle Tests
# ===============================================================================

@pytest.mark.unit
class TestModalityManagement:
    """Test modality lifecycle management (add/get/list/delete)."""
    
    def test_load_modality_success(self, temp_workspace, mock_adapter):
        """Test successful modality loading."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)
        
        # Create test data
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_adapter.from_source.return_value = test_data
        
        # Load modality
        adata = dm.load_modality(
            name="test_modality",
            source="/path/to/data.csv",
            adapter="test_adapter",
            validate=True
        )
        
        assert adata is test_data
        assert "test_modality" in dm.modalities
        assert dm.modalities["test_modality"] is test_data
        
        # Check that adapter methods were called
        mock_adapter.from_source.assert_called_once()
        mock_adapter.validate.assert_called_once()

    def test_load_modality_unregistered_adapter(self, temp_workspace):
        """Test loading modality with unregistered adapter raises error."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        with pytest.raises(ValueError, match="Adapter 'nonexistent' not registered"):
            dm.load_modality(
                name="test_modality",
                source="/path/to/data.csv",
                adapter="nonexistent"
            )

    def test_load_modality_validation_failure(self, temp_workspace, mock_adapter):
        """Test modality loading with failed validation."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)
        
        # Setup validation failure
        validation_result = Mock(spec=ValidationResult)
        validation_result.has_errors = True
        validation_result.errors = ["Missing required field"]
        mock_adapter.validate.return_value = validation_result
        mock_adapter.from_source.return_value = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        with pytest.raises(ValueError, match="Validation failed for modality 'test_modality'"):
            dm.load_modality(
                name="test_modality",
                source="/path/to/data.csv",
                adapter="test_adapter",
                validate=True
            )

    def test_load_modality_validation_warnings(self, temp_workspace, mock_adapter):
        """Test modality loading with validation warnings (should succeed)."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)
        
        # Setup validation warnings
        validation_result = Mock(spec=ValidationResult)
        validation_result.has_errors = False
        validation_result.has_warnings = True
        validation_result.warnings = ["Optional field missing"]
        mock_adapter.validate.return_value = validation_result
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_adapter.from_source.return_value = test_data
        
        # Should succeed despite warnings
        adata = dm.load_modality(
            name="test_modality",
            source="/path/to/data.csv",
            adapter="test_adapter",
            validate=True
        )
        
        assert adata is test_data
        assert "test_modality" in dm.modalities

    def test_load_modality_skip_validation(self, temp_workspace, mock_adapter):
        """Test modality loading with validation disabled."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_adapter.from_source.return_value = test_data
        
        adata = dm.load_modality(
            name="test_modality",
            source="/path/to/data.csv",
            adapter="test_adapter",
            validate=False
        )
        
        assert adata is test_data
        mock_adapter.validate.assert_not_called()

    def test_load_modality_with_anndata_source(self, temp_workspace, mock_adapter):
        """Test loading modality when source is already AnnData."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        adata = dm.load_modality(
            name="test_modality",
            source=test_data,  # Already AnnData
            adapter="test_adapter",
            validate=False
        )
        
        assert adata is test_data
        mock_adapter.from_source.assert_not_called()  # Should not process if already AnnData

    def test_get_modality_success(self, temp_workspace):
        """Test successful modality retrieval."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_modality"] = test_data
        
        retrieved = dm.get_modality("test_modality")
        assert retrieved is test_data

    def test_get_modality_not_found(self, temp_workspace):
        """Test getting non-existent modality raises error."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        with pytest.raises(ValueError, match="Modality 'nonexistent' not found"):
            dm.get_modality("nonexistent")

    def test_list_modalities(self, temp_workspace):
        """Test listing loaded modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Initially empty
        assert dm.list_modalities() == []
        
        # Add some modalities
        dm.modalities["mod1"] = Mock()
        dm.modalities["mod2"] = Mock()
        
        modalities = dm.list_modalities()
        assert sorted(modalities) == ["mod1", "mod2"]

    def test_remove_modality_success(self, temp_workspace):
        """Test successful modality removal."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.modalities["test_modality"] = Mock()
        
        dm.remove_modality("test_modality")
        
        assert "test_modality" not in dm.modalities

    def test_remove_modality_not_found(self, temp_workspace):
        """Test removing non-existent modality raises error."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        with pytest.raises(ValueError, match="Modality 'nonexistent' not found"):
            dm.remove_modality("nonexistent")

    def test_save_modality_success(self, temp_workspace, mock_backend):
        """Test successful modality saving."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_backend("test_backend", mock_backend)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_modality"] = test_data
        
        result_path = dm.save_modality(
            name="test_modality",
            path="test_file.h5ad",
            backend="test_backend"
        )
        
        assert result_path is not None
        mock_backend.save.assert_called_once()

    def test_save_modality_not_loaded(self, temp_workspace):
        """Test saving non-existent modality raises error."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        with pytest.raises(ValueError, match="Modality 'nonexistent' not loaded"):
            dm.save_modality("nonexistent", "test.h5ad")

    def test_save_modality_unregistered_backend(self, temp_workspace):
        """Test saving with unregistered backend raises error."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.modalities["test_modality"] = Mock()
        
        with pytest.raises(ValueError, match="Backend 'nonexistent' not registered"):
            dm.save_modality("test_modality", "test.h5ad", backend="nonexistent")

    def test_save_modality_relative_path(self, temp_workspace, mock_backend):
        """Test saving modality with relative path resolves to data_dir."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_backend("test_backend", mock_backend)
        dm.modalities["test_modality"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        dm.save_modality("test_modality", "relative_path.h5ad", backend="test_backend")
        
        # Check that path was resolved relative to data_dir
        call_args = mock_backend.save.call_args
        saved_path = call_args[0][1]  # Second argument to save()
        assert str(saved_path).startswith(str(dm.data_dir))


# ===============================================================================
# Quality & Validation Tests  
# ===============================================================================

@pytest.mark.unit
class TestQualityValidation:
    """Test data quality assessment and validation functionality."""
    
    def test_get_quality_metrics_single_modality(self, temp_workspace, mock_adapter):
        """Test getting quality metrics for a single modality."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("transcriptomics_single_cell", mock_adapter)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_single_cell"] = test_data
        
        mock_adapter.get_quality_metrics.return_value = {
            "n_cells": 100,
            "n_genes": 500,
            "mean_genes_per_cell": 250
        }
        
        metrics = dm.get_quality_metrics("test_single_cell")
        
        assert isinstance(metrics, dict)
        assert "n_cells" in metrics
        mock_adapter.get_quality_metrics.assert_called_once_with(test_data)

    def test_get_quality_metrics_all_modalities(self, temp_workspace, mock_adapter):
        """Test getting quality metrics for all modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("transcriptomics_single_cell", mock_adapter)
        
        dm.modalities["mod1"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["mod2"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        mock_adapter.get_quality_metrics.return_value = {"test": "metrics"}
        
        metrics = dm.get_quality_metrics()
        
        assert isinstance(metrics, dict)
        assert "mod1" in metrics
        assert "mod2" in metrics
        assert mock_adapter.get_quality_metrics.call_count == 2

    def test_get_quality_metrics_modality_not_found(self, temp_workspace):
        """Test getting quality metrics for non-existent modality."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        with pytest.raises(ValueError, match="Modality 'nonexistent' not found"):
            dm.get_quality_metrics("nonexistent")

    def test_get_quality_metrics_no_matching_adapter(self, temp_workspace):
        """Test getting quality metrics when no adapter matches."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.modalities["unknown_type"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        with patch('lobster.core.adapters.base.BaseAdapter') as mock_base_adapter:
            mock_base_instance = Mock()
            mock_base_adapter.return_value = mock_base_instance
            mock_base_instance.get_quality_metrics.return_value = {"basic": "metrics"}
            
            metrics = dm.get_quality_metrics("unknown_type")
            
            assert metrics == {"basic": "metrics"}
            mock_base_adapter.assert_called_once()

    def test_validate_modalities_success(self, temp_workspace, mock_adapter):
        """Test validating all modalities successfully."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("transcriptomics_single_cell", mock_adapter)
        
        dm.modalities["test_mod"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        validation_result = Mock(spec=ValidationResult)
        validation_result.has_errors = False
        mock_adapter.validate.return_value = validation_result
        
        results = dm.validate_modalities(strict=False)
        
        assert isinstance(results, dict)
        assert "test_mod" in results
        assert results["test_mod"] is validation_result
        mock_adapter.validate.assert_called_once()

    def test_validate_modalities_with_base_adapter_fallback(self, temp_workspace):
        """Test validation fallback to base adapter when no specific adapter found."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.modalities["unknown_type"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        with patch('lobster.core.adapters.base.BaseAdapter') as mock_base_adapter:
            mock_base_instance = Mock()
            mock_base_adapter.return_value = mock_base_instance
            mock_base_instance._validate_basic_structure.return_value = Mock(spec=ValidationResult)
            
            results = dm.validate_modalities()
            
            assert "unknown_type" in results
            mock_base_adapter.assert_called_once()

    def test_match_modality_to_adapter_single_cell(self, temp_workspace):
        """Test modality name matching for single-cell data."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Test various single-cell naming patterns
        single_cell_names = [
            "single_cell_experiment",
            "sc_rnaseq_data", 
            "10x_genomics_data",
            "scrna_seq_analysis"
        ]
        
        for name in single_cell_names:
            adapter = dm._match_modality_to_adapter(name)
            assert adapter == "transcriptomics_single_cell"

    def test_match_modality_to_adapter_bulk(self, temp_workspace):
        """Test modality name matching for bulk RNA-seq data."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        bulk_names = [
            "bulk_rnaseq",
            "bulk_transcriptomics",
            "rna_bulk_analysis"
        ]
        
        for name in bulk_names:
            adapter = dm._match_modality_to_adapter(name)
            assert adapter == "transcriptomics_bulk"

    def test_match_modality_to_adapter_proteomics(self, temp_workspace):
        """Test modality name matching for proteomics data."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Mass spec proteomics
        ms_names = ["ms_proteomics", "mass_spec_data", "mass_spectrometry"]
        for name in ms_names:
            adapter = dm._match_modality_to_adapter(name)
            assert adapter == "proteomics_ms"
        
        # Affinity proteomics
        affinity_names = ["affinity_proteomics", "antibody_array", "western_blot"]
        for name in affinity_names:
            adapter = dm._match_modality_to_adapter(name)
            assert adapter == "proteomics_affinity"

    def test_match_modality_to_adapter_geo_datasets(self, temp_workspace):
        """Test modality name matching for GEO datasets."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # GEO single-cell
        geo_sc = dm._match_modality_to_adapter("geo_gse123456_single_cell")
        assert geo_sc == "transcriptomics_single_cell"
        
        # GEO bulk (default)
        geo_bulk = dm._match_modality_to_adapter("geo_gse789012")
        assert geo_bulk == "transcriptomics_bulk"

    def test_match_modality_to_adapter_no_match(self, temp_workspace):
        """Test modality name matching returns None for unknown patterns."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        adapter = dm._match_modality_to_adapter("completely_unknown_data_type")
        assert adapter is None


# ===============================================================================
# Workspace Management Tests
# ===============================================================================

@pytest.mark.unit
class TestWorkspaceManagement:
    """Test workspace management functionality."""
    
    def test_get_workspace_status(self, temp_workspace):
        """Test getting comprehensive workspace status."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.modalities["test_mod"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        status = dm.get_workspace_status()
        
        assert isinstance(status, dict)
        assert status["workspace_path"] == str(temp_workspace)
        assert status["modalities_loaded"] == 1
        assert "test_mod" in status["modality_names"]
        assert "registered_backends" in status
        assert "registered_adapters" in status
        assert "directories" in status
        assert "modality_details" in status

    def test_get_workspace_status_with_provenance(self, temp_workspace, mock_provenance):
        """Test workspace status includes provenance information."""
        with patch('lobster.core.data_manager_v2.ProvenanceTracker', return_value=mock_provenance):
            dm = DataManagerV2(workspace_path=temp_workspace, enable_provenance=True)
            
            status = dm.get_workspace_status()
            
            assert "provenance" in status
            assert status["provenance"]["n_activities"] == 0
            assert status["provenance"]["n_entities"] == 0
            assert status["provenance"]["n_agents"] == 0

    def test_clear_workspace_without_confirmation(self, temp_workspace):
        """Test workspace clearing requires confirmation."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.modalities["test_mod"] = Mock()
        
        with pytest.raises(ValueError, match="Must set confirm=True to clear workspace"):
            dm.clear_workspace(confirm=False)

    def test_clear_workspace_with_confirmation(self, temp_workspace, mock_provenance):
        """Test successful workspace clearing with confirmation."""
        with patch('lobster.core.data_manager_v2.ProvenanceTracker', return_value=mock_provenance):
            dm = DataManagerV2(workspace_path=temp_workspace, enable_provenance=True)
            dm.modalities["test_mod"] = Mock()
            
            dm.clear_workspace(confirm=True)
            
            assert len(dm.modalities) == 0
            # Check that new provenance tracker was created
            assert dm.provenance is not None

    def test_list_workspace_files(self, temp_workspace):
        """Test listing workspace files by category."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Create some test files
        (dm.data_dir / "test_data.h5ad").write_text("test data")
        (dm.exports_dir / "test_export.zip").write_text("test export")
        (dm.cache_dir / "test_cache.tmp").write_text("test cache")
        
        files = dm.list_workspace_files()
        
        assert isinstance(files, dict)
        assert "data" in files
        assert "exports" in files
        assert "cache" in files
        
        assert len(files["data"]) == 1
        assert files["data"][0]["name"] == "test_data.h5ad"
        assert "size" in files["data"][0]
        assert "modified" in files["data"][0]

    def test_has_data(self, temp_workspace):
        """Test has_data method."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Initially no data
        assert dm.has_data() is False
        
        # Add modality
        dm.modalities["test_mod"] = Mock()
        assert dm.has_data() is True

    def test_auto_save_state(self, temp_workspace, mock_backend):
        """Test automatic state saving."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_backend("h5ad", mock_backend)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_mod"] = test_data
        dm.tool_usage_history.append({"tool": "test", "params": {}})
        
        with patch.object(dm, 'save_modality') as mock_save:
            mock_save.return_value = "/path/to/saved.h5ad"
            
            saved_items = dm.auto_save_state()
            
            assert len(saved_items) >= 1
            mock_save.assert_called()

    def test_auto_save_state_no_data(self, temp_workspace):
        """Test auto save with no data returns empty list."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        saved_items = dm.auto_save_state()
        
        assert isinstance(saved_items, list)
        # May contain processing log but no modality saves

    @patch('lobster.core.data_manager_v2.MUDATA_AVAILABLE', True)
    def test_to_mudata_success(self, temp_workspace):
        """Test successful conversion to MuData object."""
        with patch('lobster.core.data_manager_v2.mudata') as mock_mudata:
            mock_mdata = Mock()
            mock_mudata.MuData.return_value = mock_mdata
            
            dm = DataManagerV2(workspace_path=temp_workspace)
            dm.modalities["rna"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            dm.modalities["protein"] = ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)
            
            result = dm.to_mudata()
            
            assert result is mock_mdata
            mock_mudata.MuData.assert_called_once()

    @patch('lobster.core.data_manager_v2.MUDATA_AVAILABLE', False)
    def test_to_mudata_not_available(self, temp_workspace):
        """Test MuData conversion when MuData not available."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.modalities["test"] = Mock()
        
        with pytest.raises(ImportError, match="MuData is not available"):
            dm.to_mudata()

    def test_to_mudata_no_modalities(self, temp_workspace):
        """Test MuData conversion with no modalities loaded."""
        with patch('lobster.core.data_manager_v2.MUDATA_AVAILABLE', True):
            dm = DataManagerV2(workspace_path=temp_workspace)
            
            with pytest.raises(ValueError, match="No modalities loaded"):
                dm.to_mudata()

    def test_to_mudata_specific_modalities(self, temp_workspace):
        """Test MuData conversion with specific modalities."""
        with patch('lobster.core.data_manager_v2.MUDATA_AVAILABLE', True), \
             patch('lobster.core.data_manager_v2.mudata') as mock_mudata:
            
            mock_mdata = Mock()
            mock_mudata.MuData.return_value = mock_mdata
            
            dm = DataManagerV2(workspace_path=temp_workspace)
            dm.modalities["rna"] = Mock()
            dm.modalities["protein"] = Mock()
            dm.modalities["other"] = Mock()
            
            result = dm.to_mudata(modalities=["rna", "protein"])
            
            # Check that only specified modalities were included
            call_args = mock_mudata.MuData.call_args[0][0]
            assert set(call_args.keys()) == {"rna", "protein"}

    def test_to_mudata_missing_modalities(self, temp_workspace):
        """Test MuData conversion with non-existent modalities."""
        with patch('lobster.core.data_manager_v2.MUDATA_AVAILABLE', True):
            dm = DataManagerV2(workspace_path=temp_workspace)
            dm.modalities["rna"] = Mock()
            
            with pytest.raises(ValueError, match="Modalities not found: \\['nonexistent'\\]"):
                dm.to_mudata(modalities=["rna", "nonexistent"])


# ===============================================================================
# Plot Management Tests
# ===============================================================================

@pytest.mark.unit
class TestPlotManagement:
    """Test plot storage and management functionality."""
    
    def test_add_plot_success(self, temp_workspace):
        """Test successful plot addition."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Create test plot
        fig = go.Figure()
        fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6])
        
        plot_id = dm.add_plot(
            plot=fig,
            title="Test Plot",
            source="test_service",
            dataset_info={"modality": "test_data"},
            analysis_params={"param1": "value1"}
        )
        
        assert plot_id is not None
        assert len(dm.latest_plots) == 1
        assert dm.latest_plots[0]["id"] == plot_id
        assert dm.latest_plots[0]["original_title"] == "Test Plot"
        assert dm.latest_plots[0]["source"] == "test_service"

    def test_add_plot_invalid_input(self, temp_workspace):
        """Test adding invalid plot raises error."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        with pytest.raises(ValueError, match="Plot must be a plotly Figure object"):
            dm.add_plot(plot="not a figure")

    def test_add_plot_with_dataset_context(self, temp_workspace):
        """Test plot addition with dataset context."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Add modality for context
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_modality"] = test_data
        dm.current_dataset = "test_modality"
        
        fig = go.Figure()
        plot_id = dm.add_plot(plot=fig, title="Context Plot")
        
        plot_entry = dm.latest_plots[0]
        assert "test_modality" in plot_entry["title"]
        assert plot_entry["dataset_info"]["modality_name"] == "test_modality"

    def test_add_plot_max_history_limit(self, temp_workspace):
        """Test plot history respects maximum limit."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.max_plots_history = 3  # Set low limit for testing
        
        # Add more plots than the limit
        for i in range(5):
            fig = go.Figure()
            dm.add_plot(plot=fig, title=f"Plot {i}")
        
        # Should only keep the most recent plots
        assert len(dm.latest_plots) == 3
        # Check that it kept the latest ones
        titles = [plot["original_title"] for plot in dm.latest_plots]
        assert "Plot 2" in titles
        assert "Plot 3" in titles  
        assert "Plot 4" in titles
        assert "Plot 0" not in titles
        assert "Plot 1" not in titles

    def test_get_plot_by_id(self, temp_workspace):
        """Test retrieving plot by ID."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        fig = go.Figure()
        plot_id = dm.add_plot(plot=fig, title="Test Plot")
        
        retrieved_fig = dm.get_plot_by_id(plot_id)
        assert retrieved_fig is fig

    def test_get_plot_by_id_not_found(self, temp_workspace):
        """Test retrieving non-existent plot returns None."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        result = dm.get_plot_by_id("nonexistent_id")
        assert result is None

    def test_get_latest_plots(self, temp_workspace):
        """Test getting latest plots."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Add multiple plots
        for i in range(3):
            fig = go.Figure()
            dm.add_plot(plot=fig, title=f"Plot {i}")
        
        # Get all plots
        all_plots = dm.get_latest_plots()
        assert len(all_plots) == 3
        
        # Get limited number
        recent_plots = dm.get_latest_plots(n=2)
        assert len(recent_plots) == 2

    def test_get_plot_history(self, temp_workspace):
        """Test getting plot history metadata."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        fig = go.Figure()
        plot_id = dm.add_plot(plot=fig, title="Test Plot", source="test_service")
        
        history = dm.get_plot_history()
        
        assert len(history) == 1
        assert history[0]["id"] == plot_id
        assert history[0]["title"] == dm.latest_plots[0]["title"]  # Enhanced title
        assert history[0]["source"] == "test_service"
        assert "figure" not in history[0]  # Should not include actual figure

    def test_clear_plots(self, temp_workspace):
        """Test clearing all plots."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        fig = go.Figure()
        dm.add_plot(plot=fig, title="Test Plot")
        assert len(dm.latest_plots) == 1
        
        dm.clear_plots()
        assert len(dm.latest_plots) == 0

    @patch('lobster.core.data_manager_v2.pio')
    def test_save_plots_to_workspace(self, mock_pio, temp_workspace):
        """Test saving plots to workspace directory."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        fig = go.Figure()
        dm.add_plot(plot=fig, title="Test Plot")
        
        saved_files = dm.save_plots_to_workspace()
        
        assert len(saved_files) >= 1  # At least HTML file
        mock_pio.write_html.assert_called()
        
        # Check plots directory was created
        plots_dir = dm.workspace_path / "plots"
        assert plots_dir.exists()

    def test_save_plots_to_workspace_no_plots(self, temp_workspace):
        """Test saving plots when no plots exist."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        saved_files = dm.save_plots_to_workspace()
        
        assert saved_files == []


# ===============================================================================
# Legacy Compatibility Tests
# ===============================================================================

@pytest.mark.unit
class TestLegacyCompatibility:
    """Test legacy compatibility features."""
    
    def test_set_data_dataframe(self, temp_workspace, mock_adapter):
        """Test legacy set_data method with DataFrame."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("transcriptomics_single_cell", mock_adapter)
        
        # Create test DataFrame (high gene count -> single cell)
        test_df = pd.DataFrame(
            np.random.randint(0, 100, (50, 6000)),  # 6000 genes -> single cell
            index=[f"Cell_{i}" for i in range(50)],
            columns=[f"Gene_{i}" for i in range(6000)]
        )
        
        mock_adapter.from_source.return_value = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        result = dm.set_data(test_df, metadata={"experiment": "test"})
        
        assert result is test_df
        assert len(dm.modalities) > 0
        assert dm.current_dataset is not None
        mock_adapter.from_source.assert_called_once()

    def test_set_data_empty_dataframe(self, temp_workspace):
        """Test set_data with empty DataFrame raises error."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="DataFrame is empty"):
            dm.set_data(empty_df)

    def test_set_data_invalid_input(self, temp_workspace):
        """Test set_data with invalid input raises error."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
            dm.set_data("not a dataframe")

    def test_set_data_different_modality_types(self, temp_workspace, mock_adapter):
        """Test set_data detects different modality types."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Register different adapters
        dm.register_adapter("transcriptomics_single_cell", mock_adapter)
        dm.register_adapter("transcriptomics_bulk", mock_adapter)
        dm.register_adapter("proteomics_ms", mock_adapter)
        
        mock_adapter.from_source.return_value = Mock()
        
        # Test bulk RNA-seq (medium gene count)
        bulk_df = pd.DataFrame(np.random.randint(0, 1000, (24, 1500)))
        dm.set_data(bulk_df)
        assert "legacy_bulk" in dm.modalities
        
        # Clear and test proteomics (low feature count)
        dm.modalities.clear()
        proteomics_df = pd.DataFrame(np.random.randn(48, 300))
        dm.set_data(proteomics_df)
        assert "legacy_proteomics" in dm.modalities

    def test_log_tool_usage(self, temp_workspace):
        """Test logging tool usage for reproducibility."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        dm.log_tool_usage(
            tool_name="test_tool",
            parameters={"param1": "value1", "param2": 42},
            description="Test tool execution"
        )
        
        assert len(dm.tool_usage_history) == 1
        entry = dm.tool_usage_history[0]
        assert entry["tool"] == "test_tool"
        assert entry["parameters"]["param1"] == "value1"
        assert entry["description"] == "Test tool execution"
        assert "timestamp" in entry

    def test_save_processed_data_no_modalities(self, temp_workspace):
        """Test save_processed_data with no modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        result = dm.save_processed_data("test_step")
        
        assert result is None

    @patch('lobster.utils.file_naming.BioinformaticsFileNaming')
    def test_save_processed_data_success(self, mock_naming, temp_workspace, mock_backend):
        """Test successful processed data saving."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_backend("h5ad", mock_backend)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_modality"] = test_data
        
        mock_naming.generate_filename.return_value = "processed_data.h5ad"
        mock_naming.generate_metadata_filename.return_value = "processed_data_metadata.json"
        mock_naming.suggest_next_step.return_value = "clustering"
        
        with patch.object(dm, 'save_modality') as mock_save:
            mock_save.return_value = "/path/to/saved.h5ad"
            
            result = dm.save_processed_data(
                processing_step="filtered",
                data_source="GEO",
                dataset_id="GSE123456"
            )
            
            assert result is not None
            mock_save.assert_called_once()

    def test_get_data_summary_current_dataset(self, temp_workspace, mock_adapter):
        """Test data summary for current dataset."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("transcriptomics_single_cell", mock_adapter)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["current_mod"] = test_data
        dm.current_dataset = "current_mod"
        
        mock_adapter.get_quality_metrics.return_value = {"test": "metrics"}
        
        summary = dm.get_data_summary()
        
        assert summary["status"] == "Modality loaded"
        assert summary["modality_name"] == "current_mod"
        assert "shape" in summary
        assert "quality_metrics" in summary

    def test_get_data_summary_all_modalities(self, temp_workspace):
        """Test data summary for all modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        dm.modalities["mod1"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["mod2"] = BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG)
        
        summary = dm.get_data_summary()
        
        assert "modalities" in summary
        assert "mod1" in summary["modalities"]
        assert "mod2" in summary["modalities"]
        assert "total_obs" in summary
        assert "total_vars" in summary

    def test_get_data_summary_no_data(self, temp_workspace):
        """Test data summary with no data loaded."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        summary = dm.get_data_summary()
        
        assert summary["status"] == "No modalities loaded"

    def test_safe_helper_methods(self, temp_workspace):
        """Test various _get_safe_* helper methods handle edge cases."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Test with normal AnnData
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        # Test shape
        shape = dm._get_safe_shape(test_data)
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        
        # Test memory usage
        memory = dm._get_safe_memory_usage(test_data.X)
        assert isinstance(memory, str)
        assert "MB" in memory
        
        # Test data type info
        dtype_info = dm._get_safe_data_type_info(test_data.X)
        assert isinstance(dtype_info, str)
        
        # Test with None values
        assert dm._get_safe_shape(Mock(X=None)) == (0, 0)
        assert dm._get_safe_memory_usage(None) == "N/A (No data matrix)"
        assert dm._get_safe_data_type_info(None) == "None"

    def test_is_sparse_matrix(self, temp_workspace):
        """Test sparse matrix detection."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Test with dense array
        dense_array = np.array([[1, 2], [3, 4]])
        assert dm._is_sparse_matrix(dense_array) is False
        
        # Test with mock sparse matrix
        mock_sparse = Mock()
        mock_sparse.nnz = 5  # Has nnz attribute (characteristic of sparse matrices)
        assert dm._is_sparse_matrix(mock_sparse) is True


# ===============================================================================
# Machine Learning Integration Tests
# ===============================================================================

@pytest.mark.unit
class TestMachineLearningIntegration:
    """Test machine learning workflow integration."""
    
    def test_check_ml_readiness_single_modality(self, temp_workspace):
        """Test ML readiness check for single modality."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Create good quality data
        test_data = SingleCellDataFactory(config=MEDIUM_DATASET_CONFIG)
        dm.modalities["test_mod"] = test_data
        
        readiness = dm.check_ml_readiness("test_mod")
        
        assert isinstance(readiness, dict)
        assert "readiness_score" in readiness
        assert "readiness_level" in readiness
        assert "checks" in readiness
        assert "recommendations" in readiness

    def test_check_ml_readiness_all_modalities(self, temp_workspace):
        """Test ML readiness check for all modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        dm.modalities["mod1"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["mod2"] = BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG)
        
        readiness = dm.check_ml_readiness()
        
        assert readiness["status"] == "success"
        assert "modalities" in readiness
        assert "overall_readiness" in readiness
        assert len(readiness["modalities"]) == 2

    def test_check_ml_readiness_no_modalities(self, temp_workspace):
        """Test ML readiness with no modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        readiness = dm.check_ml_readiness()
        
        assert readiness["status"] == "error"
        assert "No modalities loaded" in readiness["message"]

    def test_check_ml_readiness_nonexistent_modality(self, temp_workspace):
        """Test ML readiness for non-existent modality."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        with pytest.raises(ValueError, match="Modality 'nonexistent' not found"):
            dm.check_ml_readiness("nonexistent")

    @patch('scanpy.pp.normalize_total')
    @patch('scanpy.pp.log1p')
    @patch('scanpy.pp.highly_variable_genes')
    def test_prepare_ml_features_with_scanpy(self, mock_hvg, mock_log1p, mock_normalize, temp_workspace):
        """Test ML feature preparation with scanpy available."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_mod"] = test_data
        
        # Mock scanpy functions
        test_data.var["highly_variable"] = np.random.choice([True, False], size=test_data.n_vars)
        
        result = dm.prepare_ml_features(
            modality="test_mod",
            feature_selection="variance",
            n_features=100,
            normalization="log1p",
            scaling="standard"
        )
        
        assert isinstance(result, dict)
        assert "processed_modality" in result
        assert "feature_matrix" in result
        assert "processing_steps" in result
        assert "scaler" in result
        
        # Check that processed modality was created
        processed_name = result["processed_modality"]
        assert processed_name in dm.modalities

    def test_prepare_ml_features_without_scanpy(self, temp_workspace):
        """Test ML feature preparation without scanpy (basic processing)."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_mod"] = test_data
        
        with patch('lobster.core.data_manager_v2.sc', None):
            result = dm.prepare_ml_features(
                modality="test_mod",
                feature_selection="variance",
                n_features=50
            )
            
            assert isinstance(result, dict)
            assert result["processed_shape"][1] == 50  # Selected features

    def test_prepare_ml_features_nonexistent_modality(self, temp_workspace):
        """Test ML feature preparation for non-existent modality."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        with pytest.raises(ValueError, match="Modality 'nonexistent' not found"):
            dm.prepare_ml_features("nonexistent")

    @patch('sklearn.model_selection.train_test_split')
    def test_create_ml_splits_success(self, mock_train_test_split, temp_workspace):
        """Test successful ML data splits creation."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        test_data.obs["cell_type"] = np.random.choice(["A", "B", "C"], size=test_data.n_obs)
        dm.modalities["test_mod"] = test_data
        
        # Mock train_test_split to return predictable indices
        mock_train_test_split.side_effect = [
            (np.arange(80), np.arange(80, 100)),  # Train/temp split
            (np.arange(80, 90), np.arange(90, 100))  # Val/test split
        ]
        
        splits = dm.create_ml_splits(
            modality="test_mod",
            target_column="cell_type",
            test_size=0.2,
            validation_size=0.1
        )
        
        assert isinstance(splits, dict)
        assert "splits" in splits
        assert "train" in splits["splits"]
        assert "validation" in splits["splits"]
        assert "test" in splits["splits"]
        
        # Check that splits were stored in AnnData
        assert "ml_splits" in test_data.uns

    def test_create_ml_splits_no_stratification(self, temp_workspace):
        """Test ML splits without stratification."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_mod"] = test_data
        
        with patch('sklearn.model_selection.train_test_split') as mock_split:
            mock_split.side_effect = [
                (np.arange(80), np.arange(80, 100)),
                (np.arange(80, 90), np.arange(90, 100))
            ]
            
            splits = dm.create_ml_splits(
                modality="test_mod",
                stratify=False
            )
            
            assert splits["stratified"] is False

    def test_export_for_ml_framework_sklearn(self, temp_workspace):
        """Test ML export for sklearn framework."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        test_data.obs["target"] = np.random.choice([0, 1], size=test_data.n_obs)
        dm.modalities["test_mod"] = test_data
        
        with patch('numpy.save') as mock_save:
            export_info = dm.export_for_ml_framework(
                modality="test_mod",
                framework="sklearn",
                target_column="target"
            )
            
            assert export_info["framework"] == "sklearn"
            assert export_info["has_target"] is True
            assert "files" in export_info
            mock_save.assert_called()

    @patch('torch.save')
    @patch('torch.FloatTensor')
    @patch('torch.LongTensor')
    def test_export_for_ml_framework_pytorch(self, mock_long_tensor, mock_float_tensor, mock_torch_save, temp_workspace):
        """Test ML export for PyTorch framework."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_mod"] = test_data
        
        export_info = dm.export_for_ml_framework(
            modality="test_mod",
            framework="pytorch"
        )
        
        assert export_info["framework"] == "pytorch"
        mock_torch_save.assert_called()

    def test_export_for_ml_framework_pytorch_not_available(self, temp_workspace):
        """Test PyTorch export when PyTorch not available."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_mod"] = test_data
        
        with patch.dict('sys.modules', {'torch': None}):
            with patch('numpy.save') as mock_save:  # Should fallback to sklearn format
                export_info = dm.export_for_ml_framework(
                    modality="test_mod",
                    framework="pytorch"
                )
                
                # Should fallback to sklearn format
                mock_save.assert_called()

    def test_get_ml_summary(self, temp_workspace):
        """Test comprehensive ML workflow summary."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_mod"] = test_data
        
        summary = dm.get_ml_summary("test_mod")
        
        assert isinstance(summary, dict)
        assert "modality_name" in summary
        assert "ml_readiness" in summary
        assert "feature_processing" in summary
        assert "splits" in summary
        assert "metadata" in summary

    def test_get_ml_summary_all_modalities(self, temp_workspace):
        """Test ML summary for all modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        dm.modalities["mod1"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["mod2"] = BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG)
        
        summary = dm.get_ml_summary()
        
        assert summary["status"] == "success"
        assert "modalities" in summary
        assert "overall_ml_readiness" in summary

    def test_detect_modality_type(self, temp_workspace):
        """Test modality type detection from names."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Test different modality type detection
        assert dm._detect_modality_type("transcriptomics_single_cell") == "single_cell_rna_seq"
        assert dm._detect_modality_type("bulk_rna_seq") == "bulk_rna_seq"
        assert dm._detect_modality_type("proteomics_ms") == "mass_spectrometry_proteomics"
        assert dm._detect_modality_type("protein_affinity") == "affinity_proteomics"
        assert dm._detect_modality_type("unknown_data") == "unknown"

    def test_generate_ml_recommendations(self, temp_workspace):
        """Test ML recommendation generation."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Test with failed checks
        failed_checks = {
            "sufficient_samples": False,
            "sufficient_features": False, 
            "no_missing_values": False,
            "has_metadata": False
        }
        
        recommendations = dm._generate_ml_recommendations(failed_checks, "single_cell_rna_seq")
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("samples" in rec for rec in recommendations)
        assert any("features" in rec for rec in recommendations)
        
        # Test with all checks passed
        passed_checks = {key: True for key in failed_checks}
        recommendations = dm._generate_ml_recommendations(passed_checks, "single_cell_rna_seq")
        assert "Data appears ML-ready!" in recommendations


# ===============================================================================
# Export & Documentation Tests
# ===============================================================================

@pytest.mark.unit
class TestExportDocumentation:
    """Test export and documentation functionality."""
    
    def test_get_technical_summary(self, temp_workspace):
        """Test generation of technical summary."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Add some data and history
        dm.modalities["test_mod"] = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.processing_log.append("Test processing step")
        dm.tool_usage_history.append({
            "tool": "test_tool",
            "timestamp": "2024-01-01 12:00:00",
            "parameters": {"param1": "value1"},
            "description": "Test tool usage"
        })
        
        summary = dm.get_technical_summary()
        
        assert isinstance(summary, str)
        assert "DataManagerV2 Technical Summary" in summary
        assert "Loaded Modalities" in summary
        assert "Processing Log" in summary
        assert "Tool Usage History" in summary

    def test_get_technical_summary_empty(self, temp_workspace):
        """Test technical summary with no data."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        summary = dm.get_technical_summary()
        
        assert isinstance(summary, str)
        assert "DataManagerV2 Technical Summary" in summary

    @patch('zipfile.ZipFile')
    @patch('tempfile.TemporaryDirectory')
    def test_create_data_package(self, mock_temp_dir, mock_zipfile, temp_workspace):
        """Test comprehensive data package creation."""
        # Setup mocks
        mock_temp_path = temp_workspace / "temp"
        mock_temp_path.mkdir()
        mock_temp_dir.return_value.__enter__.return_value = str(mock_temp_path)
        
        mock_zip = Mock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip
        
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Add test data and plots
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        dm.modalities["test_mod"] = test_data
        
        fig = go.Figure()
        dm.add_plot(plot=fig, title="Test Plot")
        
        with patch.object(test_data, 'write_h5ad'), \
             patch('pandas.DataFrame.to_csv'), \
             patch('lobster.core.data_manager_v2.pio.write_html'), \
             patch('lobster.core.data_manager_v2.pio.write_image'):
            
            zip_path = dm.create_data_package()
            
            assert isinstance(zip_path, str)
            assert zip_path.endswith(".zip")
            mock_zipfile.assert_called_once()

    def test_create_data_package_no_data_no_plots(self, temp_workspace):
        """Test data package creation with no data or plots."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        with pytest.raises(ValueError, match="No data or plots to export"):
            dm.create_data_package()

    def test_export_provenance_success(self, temp_workspace, mock_provenance):
        """Test successful provenance export."""
        with patch('lobster.core.data_manager_v2.ProvenanceTracker', return_value=mock_provenance):
            dm = DataManagerV2(workspace_path=temp_workspace, enable_provenance=True)
            
            export_path = dm.export_provenance("provenance.json")
            
            assert isinstance(export_path, str)
            assert export_path.endswith("provenance.json")
            mock_provenance.to_dict.assert_called_once()

    def test_export_provenance_disabled(self, temp_workspace):
        """Test provenance export when provenance is disabled."""
        dm = DataManagerV2(workspace_path=temp_workspace, enable_provenance=False)
        
        with pytest.raises(ValueError, match="Provenance tracking is disabled"):
            dm.export_provenance("provenance.json")

    def test_store_and_retrieve_metadata(self, temp_workspace):
        """Test metadata storage and retrieval."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        test_metadata = {
            "dataset_id": "GSE123456",
            "title": "Test Dataset",
            "organism": "Homo sapiens"
        }
        
        # Store metadata
        dm.store_metadata("test_dataset", test_metadata, {"validated": True})
        
        # Retrieve metadata
        retrieved = dm.get_stored_metadata("test_dataset")
        
        assert retrieved is not None
        assert retrieved["metadata"] == test_metadata
        assert retrieved["validation"]["validated"] is True
        assert "fetch_timestamp" in retrieved

    def test_get_stored_metadata_not_found(self, temp_workspace):
        """Test retrieving non-existent metadata."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        result = dm.get_stored_metadata("nonexistent")
        assert result is None

    def test_list_stored_datasets(self, temp_workspace):
        """Test listing stored datasets."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Initially empty
        assert dm.list_stored_datasets() == []
        
        # Add metadata
        dm.store_metadata("dataset1", {"test": "data1"})
        dm.store_metadata("dataset2", {"test": "data2"})
        
        datasets = dm.list_stored_datasets()
        assert sorted(datasets) == ["dataset1", "dataset2"]


# ===============================================================================
# Error Handling Tests
# ===============================================================================

@pytest.mark.unit
class TestErrorHandling:
    """Test error handling patterns and edge cases."""
    
    def test_initialization_with_invalid_workspace(self):
        """Test initialization with invalid workspace path."""
        # Test with path that cannot be created (requires permission)
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                DataManagerV2(workspace_path="/root/invalid_path")

    def test_load_modality_with_adapter_exception(self, temp_workspace, mock_adapter):
        """Test modality loading when adapter raises exception."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("failing_adapter", mock_adapter)
        
        mock_adapter.from_source.side_effect = ValueError("Adapter failed")
        
        with pytest.raises(ValueError, match="Adapter failed"):
            dm.load_modality("test_mod", "source", "failing_adapter")

    def test_save_modality_with_backend_exception(self, temp_workspace, mock_backend):
        """Test modality saving when backend raises exception."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_backend("failing_backend", mock_backend)
        dm.modalities["test_mod"] = Mock()
        
        mock_backend.save.side_effect = IOError("Save failed")
        
        with pytest.raises(IOError, match="Save failed"):
            dm.save_modality("test_mod", "test.h5ad", "failing_backend")

    def test_workspace_operations_with_permission_errors(self, temp_workspace):
        """Test workspace operations handle permission errors gracefully."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Test auto_save_state with permission error
        with patch.object(dm, 'save_modality', side_effect=PermissionError("Access denied")):
            dm.modalities["test_mod"] = Mock()
            
            # Should not raise exception, just log error
            saved_items = dm.auto_save_state()
            # May still return items if processing log was saved

    def test_plot_operations_with_invalid_data(self, temp_workspace):
        """Test plot operations handle invalid data gracefully."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Test add_plot with exception in processing
        with patch('datetime.datetime.now', side_effect=Exception("Time error")):
            fig = go.Figure()
            
            # Should handle gracefully and return None
            plot_id = dm.add_plot(plot=fig, title="Test")
            assert plot_id is None

    def test_quality_metrics_with_corrupted_data(self, temp_workspace, mock_adapter):
        """Test quality metrics calculation with corrupted data."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)
        dm.modalities["corrupted_mod"] = Mock()  # Corrupted/invalid data
        
        mock_adapter.get_quality_metrics.side_effect = Exception("Data corrupted")
        
        # Should handle gracefully by falling back to base adapter
        with patch('lobster.core.adapters.base.BaseAdapter') as mock_base:
            mock_base_instance = Mock()
            mock_base.return_value = mock_base_instance
            mock_base_instance.get_quality_metrics.return_value = {"basic": "metrics"}
            
            metrics = dm.get_quality_metrics("corrupted_mod")
            
            # Should fall back to base adapter
            mock_base.assert_called_once()

    def test_memory_handling_large_datasets(self, temp_workspace):
        """Test memory handling for large datasets."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Test memory usage calculation with very large mock matrix
        large_mock = Mock()
        large_mock.nbytes = 10**10  # 10GB
        
        memory_str = dm._get_safe_memory_usage(large_mock)
        assert "MB" in memory_str
        assert float(memory_str.split()[0]) > 1000  # Should be > 1000 MB

    def test_concurrent_access_patterns(self, temp_workspace):
        """Test handling of concurrent access to modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Simulate concurrent modification
        dm.modalities["test_mod"] = Mock()
        
        # Test that operations are atomic at the Python level
        # (Real threading tests would require more complex setup)
        modalities_before = list(dm.modalities.keys())
        dm.remove_modality("test_mod")
        modalities_after = list(dm.modalities.keys())
        
        assert len(modalities_before) == 1
        assert len(modalities_after) == 0

    def test_cleanup_on_exceptions(self, temp_workspace):
        """Test proper cleanup when exceptions occur."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Test that failed operations don't leave partial state
        with pytest.raises(ValueError):
            dm.load_modality("test", "source", "nonexistent_adapter")
        
        # Should not have created partial modality
        assert "test" not in dm.modalities

    def test_validation_edge_cases(self, temp_workspace, mock_adapter):
        """Test validation with edge case data."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)
        
        # Test with validation result that has both errors and warnings
        validation_result = Mock(spec=ValidationResult)
        validation_result.has_errors = True
        validation_result.has_warnings = True
        validation_result.errors = ["Critical error"]
        validation_result.warnings = ["Minor warning"]
        mock_adapter.validate.return_value = validation_result
        mock_adapter.from_source.return_value = Mock()
        
        with pytest.raises(ValueError, match="Validation failed.*Critical error"):
            dm.load_modality("test", "source", "test_adapter", validate=True)

    def test_file_system_edge_cases(self, temp_workspace):
        """Test file system edge cases."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Test with workspace files that are not regular files
        (dm.data_dir / "test_dir").mkdir()  # Create directory instead of file
        
        files = dm.list_workspace_files()
        
        # Should only list regular files, not directories
        assert all(
            not any(f["name"] == "test_dir" for f in category_files)
            for category_files in files.values()
        )

    def test_resource_cleanup_on_clear_workspace(self, temp_workspace, mock_provenance):
        """Test proper resource cleanup when clearing workspace.""" 
        with patch('lobster.core.data_manager_v2.ProvenanceTracker', return_value=mock_provenance):
            dm = DataManagerV2(workspace_path=temp_workspace, enable_provenance=True)
            
            # Add various resources
            dm.modalities["test_mod"] = Mock()
            dm.latest_plots.append({"id": "plot1", "figure": Mock()})
            dm.tool_usage_history.append({"tool": "test"})
            
            dm.clear_workspace(confirm=True)
            
            # Check all resources are properly cleared/reset
            assert len(dm.modalities) == 0
            assert len(dm.latest_plots) == 0  # Plots should persist through clear_workspace
            assert dm.provenance is not None  # New provenance tracker created


# ===============================================================================
# Integration and Stress Tests
# ===============================================================================

@pytest.mark.unit
class TestIntegrationScenarios:
    """Test integration scenarios and stress conditions."""
    
    def test_full_workflow_single_cell_analysis(self, temp_workspace, mock_adapter, mock_backend):
        """Test complete single-cell analysis workflow."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("transcriptomics_single_cell", mock_adapter)
        dm.register_backend("h5ad", mock_backend)
        
        # Load data
        test_data = SingleCellDataFactory(config=MEDIUM_DATASET_CONFIG)
        mock_adapter.from_source.return_value = test_data
        
        adata = dm.load_modality("sc_data", "path/to/data.h5ad", "transcriptomics_single_cell")
        
        # Check quality
        mock_adapter.get_quality_metrics.return_value = {"n_cells": 1000, "n_genes": 2000}
        metrics = dm.get_quality_metrics("sc_data")
        
        # Log processing steps
        dm.log_tool_usage("quality_control", {"min_genes": 200})
        dm.log_tool_usage("normalization", {"method": "log1p"})
        
        # Create plots
        fig = go.Figure()
        dm.add_plot(fig, "QC Plot", "quality_service")
        
        # Save processed data
        dm.save_modality("sc_data", "processed_data.h5ad")
        
        # Validate workflow completion
        assert len(dm.modalities) == 1
        assert len(dm.tool_usage_history) == 2
        assert len(dm.latest_plots) == 1
        mock_backend.save.assert_called_once()

    def test_multi_modal_data_integration(self, temp_workspace):
        """Test integration of multiple data modalities."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Load different modalities
        rna_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        protein_data = ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)
        
        dm.modalities["rna"] = rna_data
        dm.modalities["protein"] = protein_data
        
        # Test workspace status
        status = dm.get_workspace_status()
        assert status["modalities_loaded"] == 2
        assert "rna" in status["modality_names"]
        assert "protein" in status["modality_names"]
        
        # Test quality assessment across modalities
        with patch.object(dm, '_match_modality_to_adapter') as mock_match:
            mock_match.return_value = None  # Force fallback to base adapter
            
            with patch('lobster.core.adapters.base.BaseAdapter') as mock_base:
                mock_base_instance = Mock()
                mock_base.return_value = mock_base_instance
                mock_base_instance.get_quality_metrics.return_value = {"basic": "metrics"}
                
                all_metrics = dm.get_quality_metrics()
                
                assert "rna" in all_metrics
                assert "protein" in all_metrics

    def test_large_dataset_handling(self, temp_workspace):
        """Test handling of large datasets (memory efficiency)."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Create mock large dataset
        large_data = Mock()
        large_data.shape = (50000, 20000)  # 50k cells, 20k genes
        large_data.X = Mock()
        large_data.X.nbytes = 4 * 50000 * 20000  # 4 bytes per float32
        large_data.obs = pd.DataFrame(index=range(50000))
        large_data.var = pd.DataFrame(index=range(20000))
        large_data.layers = {}
        large_data.obsm = {}
        large_data.uns = {}
        
        dm.modalities["large_dataset"] = large_data
        
        # Test that memory calculations work
        summary = dm.get_data_summary()
        assert "large_dataset" in summary["modalities"]
        
        # Test workspace status with large data
        status = dm.get_workspace_status()
        assert status["modality_details"]["large_dataset"]["shape"] == (50000, 20000)

    def test_error_recovery_scenarios(self, temp_workspace, mock_adapter, mock_backend):
        """Test error recovery in complex workflows."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        dm.register_adapter("test_adapter", mock_adapter)
        dm.register_backend("test_backend", mock_backend)
        
        # Successful operation
        test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_adapter.from_source.return_value = test_data
        dm.load_modality("good_data", "source", "test_adapter")
        
        # Failed operation should not affect existing data
        mock_adapter.from_source.side_effect = ValueError("Load failed")
        
        with pytest.raises(ValueError):
            dm.load_modality("bad_data", "source", "test_adapter")
        
        # Verify existing data is unaffected
        assert "good_data" in dm.modalities
        assert "bad_data" not in dm.modalities
        assert len(dm.modalities) == 1

    def test_concurrent_operation_simulation(self, temp_workspace):
        """Test simulation of concurrent operations."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Simulate multiple operations happening in sequence
        # (Real concurrency testing would require threading)
        
        operations = [
            lambda: dm.modalities.update({"mod1": Mock()}),
            lambda: dm.log_tool_usage("tool1", {}),
            lambda: dm.add_plot(go.Figure(), "Plot 1"),
            lambda: dm.modalities.update({"mod2": Mock()}),
            lambda: dm.log_tool_usage("tool2", {}),
        ]
        
        # Execute operations
        for op in operations:
            op()
        
        # Verify state consistency
        assert len(dm.modalities) == 2
        assert len(dm.tool_usage_history) == 2
        assert len(dm.latest_plots) == 1

    def test_memory_pressure_handling(self, temp_workspace):
        """Test behavior under memory pressure simulation."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Add multiple datasets to simulate memory usage
        for i in range(5):
            mock_data = Mock()
            mock_data.shape = (1000, 2000)
            mock_data.X = Mock()
            mock_data.X.nbytes = 8 * 1000 * 2000  # 8 bytes per value
            dm.modalities[f"dataset_{i}"] = mock_data
        
        # Test that operations still work
        status = dm.get_workspace_status()
        assert status["modalities_loaded"] == 5
        
        # Test selective cleanup
        dm.remove_modality("dataset_0")
        assert "dataset_0" not in dm.modalities
        assert len(dm.modalities) == 4

    def test_long_running_session_simulation(self, temp_workspace):
        """Test behavior in long-running sessions."""
        dm = DataManagerV2(workspace_path=temp_workspace)
        
        # Simulate long session with many operations
        for i in range(100):
            dm.log_tool_usage(f"tool_{i}", {"param": i})
            
            if i % 10 == 0:  # Add plot every 10 operations
                fig = go.Figure()
                dm.add_plot(fig, f"Plot {i//10}")
        
        # Test that history is maintained appropriately
        assert len(dm.tool_usage_history) == 100
        assert len(dm.latest_plots) == 10
        
        # Test summary generation with extensive history
        summary = dm.get_technical_summary()
        assert "Tool Usage History" in summary
        assert isinstance(summary, str)
        assert len(summary) > 1000  # Should be substantial


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])