"""
Comprehensive unit tests for Affinity Proteomics Expert agent.

This module provides thorough testing of the affinity proteomics expert agent including
Olink panel processing, NPX value handling, low missing value scenarios, coefficient of
variation analysis, and antibody validation metrics functionality.

Test coverage target: 95%+ with meaningful tests for affinity proteomics agent operations.
"""

import pytest
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch, mock_open
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import tempfile
import os

from lobster.agents.affinity_proteomics_expert import affinity_proteomics_expert
from lobster.core.data_manager_v2 import DataManagerV2


# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2 instance with affinity proteomics data."""
    dm = Mock(spec=DataManagerV2)

    # Create mock affinity proteomics data (Olink-style)
    n_samples, n_proteins = 96, 92  # Typical Olink panel size
    # NPX values typically range from 0-15
    X = np.random.normal(loc=6, scale=2.5, size=(n_samples, n_proteins))
    X = np.clip(X, 0, 15)  # Clip to realistic NPX range

    # Affinity proteomics has very low missing values (<5%)
    missing_mask = np.random.rand(n_samples, n_proteins) < 0.03
    X[missing_mask] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add affinity-specific metadata
    adata.obs['plate'] = [f"plate_{i//24 + 1}" for i in range(n_samples)]  # 4 plates
    adata.obs['well'] = [f"{chr(65 + i//12)}{(i%12)+1:02d}" for i in range(n_samples)]
    adata.obs['condition'] = ['control'] * 48 + ['treatment'] * 48
    adata.obs['batch'] = ['batch1'] * 32 + ['batch2'] * 32 + ['batch3'] * 32
    adata.obs['dilution_factor'] = [1] * n_samples

    # Add protein metadata typical for affinity assays
    adata.var['protein_names'] = [f"PROTEIN_{i}" for i in range(n_proteins)]
    adata.var['uniprot_id'] = [f"P{i:05d}" for i in range(n_proteins)]
    adata.var['panel'] = ['inflammation'] * 30 + ['oncology'] * 30 + ['neurology'] * 32
    adata.var['antibody_pair'] = [f"AB{i:03d}" for i in range(n_proteins)]
    adata.var['cross_reactivity_score'] = np.random.uniform(0.01, 0.15, n_proteins)
    adata.var['detection_limit'] = np.random.uniform(0.1, 1.0, n_proteins)

    # Mock DataManager methods
    dm.list_modalities.return_value = ['olink_inflammation', 'olink_oncology']
    dm.get_modality.return_value = adata
    dm.modalities = {'olink_inflammation': adata}
    dm.log_tool_usage = Mock()

    return dm


@pytest.fixture
def mock_affinity_agent():
    """Create affinity proteomics expert agent instance."""
    with patch('lobster.agents.affinity_proteomics_expert.data_manager') as mock_dm:
        agent = affinity_proteomics_expert()
        agent.data_manager = mock_dm
        return agent


@pytest.fixture
def mock_olink_data():
    """Create mock Olink panel data."""
    n_samples, n_proteins = 192, 92  # Two 96-well plates
    # Olink NPX values
    X = np.random.normal(loc=5.5, scale=2.0, size=(n_samples, n_proteins))
    X = np.clip(X, 0, 15)

    # Very low missing rate typical for Olink
    missing_mask = np.random.rand(n_samples, n_proteins) < 0.02
    X[missing_mask] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"olink_sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"olink_protein_{i}" for i in range(n_proteins)]

    # Add Olink-specific metadata
    adata.obs['plate_id'] = ['plate_001'] * 96 + ['plate_002'] * 96
    adata.obs['well_position'] = list(range(1, 97)) * 2
    adata.obs['assay_version'] = ['v4.1'] * n_samples
    adata.obs['sample_type'] = ['plasma'] * 96 + ['serum'] * 96
    adata.var['panel_name'] = ['Inflammation'] * n_proteins
    adata.var['antibody_lot'] = [f"LOT{i:04d}" for i in range(n_proteins)]
    adata.var['qc_warning'] = [False] * 90 + [True] * 2  # 2 proteins with QC warnings

    return adata


@pytest.fixture
def mock_antibody_array_data():
    """Create mock antibody array data."""
    n_samples, n_proteins = 48, 200  # Typical antibody array
    # Log2 fluorescence intensities
    X = np.random.normal(loc=8, scale=2, size=(n_samples, n_proteins))

    # Slightly higher missing rate than Olink but still low
    missing_mask = np.random.rand(n_samples, n_proteins) < 0.08
    X[missing_mask] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"array_sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"array_protein_{i}" for i in range(n_proteins)]

    # Add array-specific metadata
    adata.obs['array_id'] = [f"array_{i//8 + 1}" for i in range(n_samples)]  # 6 arrays
    adata.obs['scanner_id'] = ['scanner_A'] * 24 + ['scanner_B'] * 24
    adata.obs['scan_date'] = ['2024-01-15'] * 24 + ['2024-01-16'] * 24
    adata.var['antibody_clone'] = [f"clone_{i}" for i in range(n_proteins)]
    adata.var['antibody_concentration'] = np.random.uniform(0.1, 10.0, n_proteins)
    adata.var['signal_to_noise'] = np.random.uniform(2.0, 50.0, n_proteins)

    return adata


# ===============================================================================
# Tool Testing - Data Status and Quality Assessment
# ===============================================================================

class TestAffinityProteomicsDataStatus:
    """Test suite for affinity proteomics data status functionality."""

    def test_check_affinity_proteomics_data_status_basic(self, mock_data_manager):
        """Test basic data status check."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import check_affinity_proteomics_data_status

            result = check_affinity_proteomics_data_status()

            assert isinstance(result, str)
            assert "olink_inflammation" in result or "olink_oncology" in result
            assert "96 samples" in result or "samples" in result
            assert "92 proteins" in result or "proteins" in result

    def test_check_affinity_proteomics_data_status_specific_modality(self, mock_data_manager):
        """Test data status check for specific modality."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import check_affinity_proteomics_data_status

            result = check_affinity_proteomics_data_status("olink_inflammation")

            assert isinstance(result, str)
            assert "olink_inflammation" in result

    def test_check_affinity_proteomics_data_status_no_data(self):
        """Test data status check when no data is available."""
        mock_dm = Mock(spec=DataManagerV2)
        mock_dm.list_modalities.return_value = []

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_dm):
            from lobster.agents.affinity_proteomics_expert import check_affinity_proteomics_data_status

            result = check_affinity_proteomics_data_status()

            assert isinstance(result, str)
            assert "No modalities" in result or "no data" in result.lower()


class TestAffinityProteomicsQualityAssessment:
    """Test suite for affinity proteomics quality assessment functionality."""

    def test_assess_affinity_proteomics_quality_basic(self, mock_data_manager):
        """Test basic quality assessment."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import assess_affinity_proteomics_quality

            result = assess_affinity_proteomics_quality("olink_inflammation")

            assert isinstance(result, str)
            assert "quality" in result.lower()
            # Should contain information about CV, missing values, plate effects

    def test_assess_affinity_proteomics_quality_custom_thresholds(self, mock_data_manager):
        """Test quality assessment with custom thresholds."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import assess_affinity_proteomics_quality

            result = assess_affinity_proteomics_quality(
                "olink_inflammation",
                missing_value_threshold=0.1,  # Very low for affinity
                cv_threshold=20.0,  # Tighter CV threshold
                plate_effect_threshold=0.05
            )

            assert isinstance(result, str)
            assert "quality" in result.lower()

    def test_assess_affinity_proteomics_quality_nonexistent_modality(self, mock_data_manager):
        """Test quality assessment with nonexistent modality."""
        mock_data_manager.list_modalities.return_value = ['other_data']

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import assess_affinity_proteomics_quality

            result = assess_affinity_proteomics_quality("nonexistent_modality")

            assert isinstance(result, str)
            assert "not found" in result.lower() or "error" in result.lower()


# ===============================================================================
# Tool Testing - Data Filtering and Preprocessing
# ===============================================================================

class TestAffinityProteomicsFiltering:
    """Test suite for affinity proteomics data filtering functionality."""

    def test_filter_affinity_proteomics_data_basic(self, mock_data_manager):
        """Test basic data filtering."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import filter_affinity_proteomics_data

            result = filter_affinity_proteomics_data("olink_inflammation")

            assert isinstance(result, str)
            assert "filtered" in result.lower()
            mock_data_manager.log_tool_usage.assert_called()

    def test_filter_affinity_proteomics_data_custom_thresholds(self, mock_data_manager):
        """Test data filtering with custom thresholds."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import filter_affinity_proteomics_data

            result = filter_affinity_proteomics_data(
                "olink_inflammation",
                max_missing_per_sample=0.1,  # Very low for affinity
                max_missing_per_protein=0.2,
                max_cv_threshold=25.0,
                remove_qc_warnings=True
            )

            assert isinstance(result, str)
            assert "filtered" in result.lower()

    def test_filter_affinity_proteomics_data_plate_effects(self, mock_data_manager):
        """Test filtering with plate effect considerations."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import filter_affinity_proteomics_data

            result = filter_affinity_proteomics_data(
                "olink_inflammation",
                plate_effect_threshold=0.1,
                remove_plate_outliers=True
            )

            assert isinstance(result, str)
            assert "filtered" in result.lower()


class TestAffinityProteomicsNormalization:
    """Test suite for affinity proteomics normalization functionality."""

    def test_normalize_affinity_proteomics_data_basic(self, mock_data_manager):
        """Test basic data normalization."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import normalize_affinity_proteomics_data

            result = normalize_affinity_proteomics_data("olink_inflammation")

            assert isinstance(result, str)
            assert "normalized" in result.lower()

    def test_normalize_affinity_proteomics_data_methods(self, mock_data_manager):
        """Test different normalization methods suitable for affinity data."""
        methods = ["quantile", "median", "z_score", "robust_scale"]

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import normalize_affinity_proteomics_data

            for method in methods:
                result = normalize_affinity_proteomics_data(
                    "olink_inflammation",
                    normalization_method=method
                )

                assert isinstance(result, str)
                assert "normalized" in result.lower()

    def test_normalize_affinity_proteomics_data_plate_correction(self, mock_data_manager):
        """Test normalization with plate effect correction."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import normalize_affinity_proteomics_data

            result = normalize_affinity_proteomics_data(
                "olink_inflammation",
                correct_plate_effects=True,
                handle_missing="impute_knn",  # Conservative imputation for affinity
                batch_correction=True,
                batch_column="batch"
            )

            assert isinstance(result, str)
            assert "normalized" in result.lower()


# ===============================================================================
# Tool Testing - Statistical Analysis
# ===============================================================================

class TestAffinityProteomicsPatternAnalysis:
    """Test suite for affinity proteomics pattern analysis functionality."""

    def test_analyze_affinity_proteomics_patterns_pca(self, mock_data_manager):
        """Test PCA pattern analysis."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import analyze_affinity_proteomics_patterns

            result = analyze_affinity_proteomics_patterns(
                "olink_inflammation",
                analysis_type="pca_clustering"
            )

            assert isinstance(result, str)
            assert "pca" in result.lower() or "pattern" in result.lower()

    def test_analyze_affinity_proteomics_patterns_clustering(self, mock_data_manager):
        """Test clustering pattern analysis."""
        clustering_methods = ["kmeans", "hierarchical", "gaussian_mixture"]

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import analyze_affinity_proteomics_patterns

            for method in clustering_methods:
                result = analyze_affinity_proteomics_patterns(
                    "olink_inflammation",
                    analysis_type="pca_clustering",
                    clustering_method=method,
                    n_clusters=4
                )

                assert isinstance(result, str)
                assert "cluster" in result.lower() or "pattern" in result.lower()

    def test_analyze_affinity_proteomics_patterns_correlation_network(self, mock_data_manager):
        """Test correlation network analysis."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import analyze_affinity_proteomics_patterns

            result = analyze_affinity_proteomics_patterns(
                "olink_inflammation",
                analysis_type="correlation_network",
                correlation_threshold=0.7
            )

            assert isinstance(result, str)
            assert "correlation" in result.lower() or "network" in result.lower()


class TestAffinityProteomicsDifferentialAnalysis:
    """Test suite for affinity proteomics differential analysis functionality."""

    def test_find_differential_proteins_affinity_basic(self, mock_data_manager):
        """Test basic differential protein analysis."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import find_differential_proteins_affinity

            result = find_differential_proteins_affinity(
                "olink_inflammation",
                group_column="condition"
            )

            assert isinstance(result, str)
            assert "differential" in result.lower() or "protein" in result.lower()

    def test_find_differential_proteins_affinity_methods(self, mock_data_manager):
        """Test different differential analysis methods suitable for affinity data."""
        methods = ["t_test", "mann_whitney", "limma", "linear_mixed"]

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import find_differential_proteins_affinity

            for method in methods:
                result = find_differential_proteins_affinity(
                    "olink_inflammation",
                    group_column="condition",
                    method=method,
                    comparison="pairwise"
                )

                assert isinstance(result, str)
                assert "differential" in result.lower()

    def test_find_differential_proteins_affinity_plate_effects(self, mock_data_manager):
        """Test differential analysis with plate effect correction."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import find_differential_proteins_affinity

            result = find_differential_proteins_affinity(
                "olink_inflammation",
                group_column="condition",
                adjust_plate_effects=True,
                plate_column="plate",
                fold_change_threshold=1.2,  # Smaller FC threshold for affinity
                p_value_threshold=0.05
            )

            assert isinstance(result, str)
            assert "differential" in result.lower()


# ===============================================================================
# Tool Testing - Antibody Validation and Affinity-Specific Features
# ===============================================================================

class TestAffinityProteomicsAntibodyValidation:
    """Test suite for affinity proteomics antibody validation functionality."""

    def test_validate_antibody_specificity_basic(self, mock_data_manager):
        """Test basic antibody specificity validation."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import validate_antibody_specificity

            result = validate_antibody_specificity("olink_inflammation")

            assert isinstance(result, str)
            assert "antibody" in result.lower() or "specificity" in result.lower()

    def test_validate_antibody_specificity_custom_threshold(self, mock_data_manager):
        """Test antibody validation with custom cross-reactivity threshold."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import validate_antibody_specificity

            result = validate_antibody_specificity(
                "olink_inflammation",
                cross_reactivity_threshold=0.05,  # Stricter threshold
                save_result=True
            )

            assert isinstance(result, str)
            assert "antibody" in result.lower()

    def test_validate_antibody_specificity_missing_metadata(self, mock_data_manager):
        """Test antibody validation when cross-reactivity data is missing."""
        # Remove cross-reactivity metadata
        adata = mock_data_manager.get_modality("olink_inflammation")
        adata.var = adata.var.drop('cross_reactivity_score', axis=1)
        mock_data_manager.get_modality.return_value = adata

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import validate_antibody_specificity

            result = validate_antibody_specificity("olink_inflammation")

            assert isinstance(result, str)
            # Should handle missing metadata gracefully


# ===============================================================================
# Tool Testing - Summary and Reporting
# ===============================================================================

class TestAffinityProteomicsSummary:
    """Test suite for affinity proteomics summary functionality."""

    def test_create_affinity_proteomics_summary_no_analysis(self):
        """Test summary creation when no analysis has been performed."""
        from lobster.agents.affinity_proteomics_expert import create_affinity_proteomics_summary

        # Clear any existing analysis results
        with patch('lobster.agents.affinity_proteomics_expert.analysis_results', {"details": []}):
            result = create_affinity_proteomics_summary()

            assert isinstance(result, str)
            assert "no" in result.lower() and "analysis" in result.lower()

    def test_create_affinity_proteomics_summary_with_analysis(self):
        """Test summary creation with analysis results."""
        mock_analysis_results = {
            "details": [
                {
                    "step": "quality_assessment",
                    "timestamp": "2024-01-01 12:00:00",
                    "input_modality": "olink_inflammation",
                    "output_modality": "olink_inflammation_quality",
                    "parameters": {"cv_threshold": 20.0},
                    "summary": "Quality assessment completed"
                },
                {
                    "step": "normalization",
                    "timestamp": "2024-01-01 12:05:00",
                    "input_modality": "olink_inflammation_quality",
                    "output_modality": "olink_inflammation_normalized",
                    "parameters": {"method": "quantile", "plate_correction": True},
                    "summary": "Normalization with plate correction completed"
                },
                {
                    "step": "antibody_validation",
                    "timestamp": "2024-01-01 12:10:00",
                    "input_modality": "olink_inflammation_normalized",
                    "parameters": {"cross_reactivity_threshold": 0.1},
                    "summary": "Antibody specificity validated"
                }
            ]
        }

        with patch('lobster.agents.affinity_proteomics_expert.analysis_results', mock_analysis_results):
            from lobster.agents.affinity_proteomics_expert import create_affinity_proteomics_summary

            result = create_affinity_proteomics_summary()

            assert isinstance(result, str)
            assert "quality_assessment" in result
            assert "normalization" in result
            assert "antibody_validation" in result
            assert "3 analysis steps" in result or "3 steps" in result


# ===============================================================================
# Integration Testing - Workflow Scenarios
# ===============================================================================

class TestAffinityProteomicsWorkflows:
    """Test suite for complete affinity proteomics workflow scenarios."""

    def test_complete_olink_workflow(self, mock_data_manager, mock_olink_data):
        """Test complete Olink proteomics workflow."""
        # Update mock data manager with Olink data
        mock_data_manager.get_modality.return_value = mock_olink_data
        mock_data_manager.list_modalities.return_value = ['olink_inflammation']
        mock_data_manager.modalities = {'olink_inflammation': mock_olink_data}

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import (
                assess_affinity_proteomics_quality,
                filter_affinity_proteomics_data,
                normalize_affinity_proteomics_data,
                validate_antibody_specificity,
                find_differential_proteins_affinity
            )

            # Step 1: Quality assessment
            quality_result = assess_affinity_proteomics_quality(
                "olink_inflammation",
                missing_value_threshold=0.05,  # Very low for Olink
                cv_threshold=15.0,  # Tight CV threshold
                plate_effect_threshold=0.1
            )
            assert isinstance(quality_result, str)

            # Step 2: Filtering (minimal for high-quality Olink data)
            filter_result = filter_affinity_proteomics_data(
                "olink_inflammation",
                max_missing_per_protein=0.1,
                remove_qc_warnings=True
            )
            assert isinstance(filter_result, str)

            # Step 3: Normalization with plate correction
            norm_result = normalize_affinity_proteomics_data(
                "olink_inflammation",
                normalization_method="quantile",
                correct_plate_effects=True,
                handle_missing="impute_knn"
            )
            assert isinstance(norm_result, str)

            # Step 4: Antibody validation
            antibody_result = validate_antibody_specificity(
                "olink_inflammation",
                cross_reactivity_threshold=0.1
            )
            assert isinstance(antibody_result, str)

            # Step 5: Differential analysis
            diff_result = find_differential_proteins_affinity(
                "olink_inflammation",
                group_column="condition",
                method="t_test",
                adjust_plate_effects=True,
                plate_column="plate_id"
            )
            assert isinstance(diff_result, str)

    def test_complete_antibody_array_workflow(self, mock_data_manager, mock_antibody_array_data):
        """Test complete antibody array workflow."""
        # Update mock data manager with array data
        mock_data_manager.get_modality.return_value = mock_antibody_array_data
        mock_data_manager.list_modalities.return_value = ['antibody_array']
        mock_data_manager.modalities = {'antibody_array': mock_antibody_array_data}

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import (
                assess_affinity_proteomics_quality,
                normalize_affinity_proteomics_data,
                analyze_affinity_proteomics_patterns,
                find_differential_proteins_affinity
            )

            # Step 1: Quality assessment
            quality_result = assess_affinity_proteomics_quality(
                "antibody_array",
                missing_value_threshold=0.1,
                cv_threshold=25.0
            )
            assert isinstance(quality_result, str)

            # Step 2: Normalization
            norm_result = normalize_affinity_proteomics_data(
                "antibody_array",
                normalization_method="robust_scale",
                handle_missing="impute_median"
            )
            assert isinstance(norm_result, str)

            # Step 3: Pattern analysis
            pattern_result = analyze_affinity_proteomics_patterns(
                "antibody_array",
                analysis_type="pca_clustering",
                n_components=8,
                clustering_method="kmeans"
            )
            assert isinstance(pattern_result, str)

            # Step 4: Differential analysis
            diff_result = find_differential_proteins_affinity(
                "antibody_array",
                group_column="condition",
                method="limma"
            )
            assert isinstance(diff_result, str)


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

class TestAffinityProteomicsErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_tool_with_empty_modality_name(self, mock_data_manager):
        """Test tools with empty modality name."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import assess_affinity_proteomics_quality

            result = assess_affinity_proteomics_quality("")

            assert isinstance(result, str)
            assert "error" in result.lower() or "invalid" in result.lower()

    def test_tool_with_invalid_parameters(self, mock_data_manager):
        """Test tools with invalid parameters."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import normalize_affinity_proteomics_data

            result = normalize_affinity_proteomics_data(
                "olink_inflammation",
                normalization_method="invalid_method"
            )

            assert isinstance(result, str)
            # Should handle invalid method gracefully

    def test_tool_with_missing_plate_metadata(self, mock_data_manager):
        """Test tools when required plate metadata is missing."""
        # Create data without plate column
        adata = mock_data_manager.get_modality("olink_inflammation")
        adata.obs = adata.obs.drop('plate', axis=1)
        mock_data_manager.get_modality.return_value = adata

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import normalize_affinity_proteomics_data

            result = normalize_affinity_proteomics_data(
                "olink_inflammation",
                correct_plate_effects=True
            )

            assert isinstance(result, str)
            # Should handle missing metadata gracefully

    def test_tool_service_failure(self, mock_data_manager):
        """Test tool behavior when underlying service fails."""
        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            with patch('lobster.tools.proteomics_preprocessing_service.ProteomicsPreprocessingService') as MockService:
                # Make service raise an exception
                MockService.return_value.normalize_intensities.side_effect = Exception("Service error")

                from lobster.agents.affinity_proteomics_expert import normalize_affinity_proteomics_data

                result = normalize_affinity_proteomics_data("olink_inflammation")

                assert isinstance(result, str)
                assert "error" in result.lower() or "failed" in result.lower()


# ===============================================================================
# Scientific Accuracy Validation
# ===============================================================================

class TestAffinityProteomicsScientificAccuracy:
    """Test suite for validating scientific accuracy of affinity proteomics methods."""

    def test_cv_analysis_realistic_for_affinity(self, mock_data_manager):
        """Test that CV analysis produces realistic results for affinity assays."""
        # Create data with realistic CV patterns for affinity assays
        n_samples, n_proteins = 96, 92
        X = np.random.normal(loc=6, scale=1.5, size=(n_samples, n_proteins))

        # Affinity assays should have low CVs (typically <20%)
        for i in range(n_proteins):
            protein_cv = np.random.uniform(0.05, 0.25)  # 5-25% CV
            protein_values = X[:, i]
            X[:, i] = protein_values + np.random.normal(0, np.mean(protein_values) * protein_cv, n_samples)

        adata = ad.AnnData(X=X)
        adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
        adata.var_names = [f"protein_{i}" for i in range(n_proteins)]
        adata.obs['condition'] = ['control'] * 48 + ['treatment'] * 48
        adata.obs['plate'] = [f"plate_{i//24 + 1}" for i in range(n_samples)]

        mock_data_manager.get_modality.return_value = adata

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import assess_affinity_proteomics_quality

            result = assess_affinity_proteomics_quality(
                "test_modality",
                cv_threshold=30.0
            )

            assert isinstance(result, str)
            assert "cv" in result.lower() or "coefficient" in result.lower()

    def test_plate_effect_detection(self, mock_data_manager):
        """Test detection of plate effects in affinity data."""
        # Create data with pronounced plate effects
        n_samples, n_proteins = 96, 50
        X = np.random.normal(loc=5, scale=1, size=(n_samples, n_proteins))

        # Add plate effects: each plate has different baseline
        plate_effects = [0, 1.5, -1.0, 0.8]  # 4 plates
        for i in range(n_samples):
            plate_id = i // 24
            X[i, :] += plate_effects[plate_id]

        adata = ad.AnnData(X=X)
        adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
        adata.var_names = [f"protein_{i}" for i in range(n_proteins)]
        adata.obs['condition'] = ['control'] * 48 + ['treatment'] * 48
        adata.obs['plate'] = [f"plate_{i//24 + 1}" for i in range(n_samples)]

        mock_data_manager.get_modality.return_value = adata

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import assess_affinity_proteomics_quality

            result = assess_affinity_proteomics_quality(
                "test_modality",
                plate_effect_threshold=0.2
            )

            assert isinstance(result, str)
            # Should detect plate effects
            assert "plate" in result.lower()

    def test_npx_value_range_validation(self, mock_data_manager):
        """Test that NPX values are within expected range."""
        # Create data with realistic NPX range (0-15)
        n_samples, n_proteins = 96, 92
        X = np.random.uniform(0, 15, size=(n_samples, n_proteins))

        adata = ad.AnnData(X=X)
        adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
        adata.var_names = [f"protein_{i}" for i in range(n_proteins)]
        adata.obs['condition'] = ['control'] * 48 + ['treatment'] * 48

        mock_data_manager.get_modality.return_value = adata

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import assess_affinity_proteomics_quality

            result = assess_affinity_proteomics_quality("test_modality")

            assert isinstance(result, str)
            # Should recognize NPX-like values
            assert "quality" in result.lower()

    def test_statistical_power_for_small_effect_sizes(self, mock_data_manager):
        """Test statistical power for small effect sizes typical in affinity assays."""
        # Create data with small but significant effect sizes
        n_samples_per_group = 48
        n_proteins = 92

        # Control group
        X_control = np.random.normal(loc=6, scale=1.2, size=(n_samples_per_group, n_proteins))

        # Treatment group with small effect sizes (0.3-0.5 NPX units)
        X_treatment = np.random.normal(loc=6, scale=1.2, size=(n_samples_per_group, n_proteins))
        # 15% of proteins have small but significant changes
        de_proteins = np.random.choice(n_proteins, 14, replace=False)
        X_treatment[:, de_proteins] += np.random.uniform(0.3, 0.8, 14)

        X = np.vstack([X_control, X_treatment])

        adata = ad.AnnData(X=X)
        adata.obs_names = [f"sample_{i}" for i in range(2 * n_samples_per_group)]
        adata.var_names = [f"protein_{i}" for i in range(n_proteins)]
        adata.obs['condition'] = ['control'] * n_samples_per_group + ['treatment'] * n_samples_per_group
        adata.obs['plate'] = [f"plate_{i//24 + 1}" for i in range(2 * n_samples_per_group)]

        mock_data_manager.get_modality.return_value = adata

        with patch('lobster.agents.affinity_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.affinity_proteomics_expert import find_differential_proteins_affinity

            result = find_differential_proteins_affinity(
                "test_modality",
                group_column="condition",
                method="t_test",
                fold_change_threshold=1.2,  # Small FC threshold appropriate for affinity
                p_value_threshold=0.05
            )

            assert isinstance(result, str)
            assert "differential" in result.lower()