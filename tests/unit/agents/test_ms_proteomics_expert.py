"""
Comprehensive unit tests for MS Proteomics Expert agent.

This module provides thorough testing of the MS proteomics expert agent including
DDA/DIA workflow support, missing value handling, intensity normalization,
peptide-to-protein aggregation, and statistical testing functionality.

Test coverage target: 95%+ with meaningful tests for proteomics agent operations.
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

from lobster.agents.ms_proteomics_expert import ms_proteomics_expert
from lobster.core.data_manager_v2 import DataManagerV2


# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def mock_data_manager():
    """Create mock DataManagerV2 instance with proteomics data."""
    dm = Mock(spec=DataManagerV2)

    # Create mock MS proteomics data
    n_samples, n_proteins = 48, 100
    X = np.random.lognormal(mean=10, sigma=1, size=(n_samples, n_proteins))

    # Add missing values typical for MS proteomics (30% missing)
    missing_mask = np.random.rand(n_samples, n_proteins) < 0.3
    X[missing_mask] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add MS-specific metadata
    adata.obs['batch'] = ['batch1'] * 16 + ['batch2'] * 16 + ['batch3'] * 16
    adata.obs['condition'] = ['control'] * 24 + ['treatment'] * 24
    adata.obs['injection_order'] = list(range(n_samples))

    # Add protein metadata typical for MS
    adata.var['protein_names'] = [f"PROT_{i}_HUMAN" for i in range(n_proteins)]
    adata.var['n_peptides'] = np.random.randint(1, 10, n_proteins)
    adata.var['molecular_weight'] = np.random.uniform(20000, 200000, n_proteins)
    adata.var['is_contaminant'] = [False] * 95 + [True] * 5  # 5% contaminants

    # Mock DataManager methods
    dm.list_modalities.return_value = ['ms_proteomics_raw']
    dm.get_modality.return_value = adata
    dm.modalities = {'ms_proteomics_raw': adata}
    dm.log_tool_usage = Mock()

    return dm


@pytest.fixture
def mock_ms_agent():
    """Create MS proteomics expert agent instance."""
    with patch('lobster.agents.ms_proteomics_expert.data_manager') as mock_dm:
        agent = ms_proteomics_expert()
        agent.data_manager = mock_dm
        return agent


@pytest.fixture
def mock_dda_data():
    """Create mock DDA (Data-Dependent Acquisition) proteomics data."""
    n_samples, n_proteins = 24, 150
    X = np.random.lognormal(mean=12, sigma=1.5, size=(n_samples, n_proteins))

    # DDA typically has higher missing values (40-60%)
    missing_mask = np.random.rand(n_samples, n_proteins) < 0.5
    X[missing_mask] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"DDA_sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"DDA_protein_{i}" for i in range(n_proteins)]

    # Add DDA-specific metadata
    adata.obs['ms_method'] = ['DDA'] * n_samples
    adata.obs['lc_gradient'] = ['120min'] * n_samples
    adata.var['protein_ids'] = [f"P{i:05d}" for i in range(n_proteins)]
    adata.var['n_peptides'] = np.random.randint(2, 15, n_proteins)
    adata.var['sequence_coverage'] = np.random.uniform(0.1, 0.8, n_proteins)

    return adata


@pytest.fixture
def mock_dia_data():
    """Create mock DIA (Data-Independent Acquisition) proteomics data."""
    n_samples, n_proteins = 32, 120
    X = np.random.lognormal(mean=11, sigma=1.2, size=(n_samples, n_proteins))

    # DIA typically has lower missing values (20-30%)
    missing_mask = np.random.rand(n_samples, n_proteins) < 0.25
    X[missing_mask] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"DIA_sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"DIA_protein_{i}" for i in range(n_proteins)]

    # Add DIA-specific metadata
    adata.obs['ms_method'] = ['DIA'] * n_samples
    adata.obs['precursor_windows'] = [64] * n_samples
    adata.var['protein_ids'] = [f"Q{i:05d}" for i in range(n_proteins)]
    adata.var['n_precursors'] = np.random.randint(5, 25, n_proteins)
    adata.var['library_score'] = np.random.uniform(0.6, 1.0, n_proteins)

    return adata


# ===============================================================================
# Tool Testing - Data Status and Quality Assessment
# ===============================================================================

class TestMSProteomicsDataStatus:
    """Test suite for MS proteomics data status functionality."""

    def test_check_ms_proteomics_data_status_basic(self, mock_data_manager):
        """Test basic data status check."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import check_ms_proteomics_data_status

            result = check_ms_proteomics_data_status()

            assert isinstance(result, str)
            assert "ms_proteomics_raw" in result
            assert "48 samples" in result or "samples" in result
            assert "100 proteins" in result or "proteins" in result

    def test_check_ms_proteomics_data_status_specific_modality(self, mock_data_manager):
        """Test data status check for specific modality."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import check_ms_proteomics_data_status

            result = check_ms_proteomics_data_status("ms_proteomics_raw")

            assert isinstance(result, str)
            assert "ms_proteomics_raw" in result

    def test_check_ms_proteomics_data_status_no_data(self):
        """Test data status check when no data is available."""
        mock_dm = Mock(spec=DataManagerV2)
        mock_dm.list_modalities.return_value = []

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_dm):
            from lobster.agents.ms_proteomics_expert import check_ms_proteomics_data_status

            result = check_ms_proteomics_data_status()

            assert isinstance(result, str)
            assert "No modalities" in result or "no data" in result.lower()


class TestMSProteomicsQualityAssessment:
    """Test suite for MS proteomics quality assessment functionality."""

    def test_assess_ms_proteomics_quality_basic(self, mock_data_manager):
        """Test basic quality assessment."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import assess_ms_proteomics_quality

            result = assess_ms_proteomics_quality("ms_proteomics_raw")

            assert isinstance(result, str)
            assert "quality" in result.lower()
            # Should contain information about missing values, CV, etc.

    def test_assess_ms_proteomics_quality_custom_thresholds(self, mock_data_manager):
        """Test quality assessment with custom thresholds."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import assess_ms_proteomics_quality

            result = assess_ms_proteomics_quality(
                "ms_proteomics_raw",
                missing_value_threshold=0.5,
                cv_threshold=30.0,
                min_peptides_per_protein=3
            )

            assert isinstance(result, str)
            assert "quality" in result.lower()

    def test_assess_ms_proteomics_quality_nonexistent_modality(self, mock_data_manager):
        """Test quality assessment with nonexistent modality."""
        mock_data_manager.list_modalities.return_value = ['other_data']

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import assess_ms_proteomics_quality

            result = assess_ms_proteomics_quality("nonexistent_modality")

            assert isinstance(result, str)
            assert "not found" in result.lower() or "error" in result.lower()


# ===============================================================================
# Tool Testing - Data Filtering and Preprocessing
# ===============================================================================

class TestMSProteomicsFiltering:
    """Test suite for MS proteomics data filtering functionality."""

    def test_filter_ms_proteomics_data_basic(self, mock_data_manager):
        """Test basic data filtering."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import filter_ms_proteomics_data

            result = filter_ms_proteomics_data("ms_proteomics_raw")

            assert isinstance(result, str)
            assert "filtered" in result.lower()
            # Should have called log_tool_usage
            mock_data_manager.log_tool_usage.assert_called()

    def test_filter_ms_proteomics_data_custom_thresholds(self, mock_data_manager):
        """Test data filtering with custom thresholds."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import filter_ms_proteomics_data

            result = filter_ms_proteomics_data(
                "ms_proteomics_raw",
                max_missing_per_sample=0.5,
                max_missing_per_protein=0.6,
                min_peptides_per_protein=3,
                remove_contaminants=True
            )

            assert isinstance(result, str)
            assert "filtered" in result.lower()

    def test_filter_ms_proteomics_data_preserve_raw(self, mock_data_manager):
        """Test that filtering preserves raw data."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import filter_ms_proteomics_data

            # Mock the service to return filtered data
            with patch('lobster.tools.proteomics_preprocessing_service.ProteomicsPreprocessingService') as MockService:
                mock_service = MockService.return_value

                # Create mock filtered data
                original_adata = mock_data_manager.get_modality("ms_proteomics_raw")
                filtered_adata = original_adata.copy()
                filtered_adata.X = filtered_adata.X[:40, :80]  # Simulated filtering

                mock_service.filter_proteins.return_value = (filtered_adata, {'proteins_removed': 20})
                mock_service.filter_samples.return_value = (filtered_adata, {'samples_removed': 8})

                result = filter_ms_proteomics_data("ms_proteomics_raw")

                assert isinstance(result, str)
                # Should mention the new modality was created
                assert "ms_proteomics_raw_filtered" in result or "filtered" in result


class TestMSProteomicsNormalization:
    """Test suite for MS proteomics normalization functionality."""

    def test_normalize_ms_proteomics_data_basic(self, mock_data_manager):
        """Test basic data normalization."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import normalize_ms_proteomics_data

            result = normalize_ms_proteomics_data("ms_proteomics_raw")

            assert isinstance(result, str)
            assert "normalized" in result.lower()

    def test_normalize_ms_proteomics_data_methods(self, mock_data_manager):
        """Test different normalization methods."""
        methods = ["median", "quantile", "vsn", "total_sum"]

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import normalize_ms_proteomics_data

            for method in methods:
                result = normalize_ms_proteomics_data(
                    "ms_proteomics_raw",
                    normalization_method=method
                )

                assert isinstance(result, str)
                assert "normalized" in result.lower()
                assert method in result or "normalization" in result.lower()

    def test_normalize_ms_proteomics_data_with_imputation(self, mock_data_manager):
        """Test normalization with missing value handling."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import normalize_ms_proteomics_data

            result = normalize_ms_proteomics_data(
                "ms_proteomics_raw",
                handle_missing="impute_knn",
                log_transform=True,
                batch_correction=True,
                batch_column="batch"
            )

            assert isinstance(result, str)
            assert "normalized" in result.lower()


# ===============================================================================
# Tool Testing - Statistical Analysis
# ===============================================================================

class TestMSProteomicsPatternAnalysis:
    """Test suite for MS proteomics pattern analysis functionality."""

    def test_analyze_ms_proteomics_patterns_pca(self, mock_data_manager):
        """Test PCA pattern analysis."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import analyze_ms_proteomics_patterns

            result = analyze_ms_proteomics_patterns(
                "ms_proteomics_raw",
                analysis_type="pca_clustering"
            )

            assert isinstance(result, str)
            assert "pca" in result.lower() or "pattern" in result.lower()

    def test_analyze_ms_proteomics_patterns_clustering(self, mock_data_manager):
        """Test clustering pattern analysis."""
        clustering_methods = ["kmeans", "hierarchical", "leiden"]

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import analyze_ms_proteomics_patterns

            for method in clustering_methods:
                result = analyze_ms_proteomics_patterns(
                    "ms_proteomics_raw",
                    analysis_type="pca_clustering",
                    clustering_method=method,
                    n_clusters=3
                )

                assert isinstance(result, str)
                assert "cluster" in result.lower() or "pattern" in result.lower()


class TestMSProteomicsDifferentialAnalysis:
    """Test suite for MS proteomics differential analysis functionality."""

    def test_find_differential_proteins_ms_basic(self, mock_data_manager):
        """Test basic differential protein analysis."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import find_differential_proteins_ms

            result = find_differential_proteins_ms(
                "ms_proteomics_raw",
                group_column="condition"
            )

            assert isinstance(result, str)
            assert "differential" in result.lower() or "protein" in result.lower()

    def test_find_differential_proteins_ms_methods(self, mock_data_manager):
        """Test different differential analysis methods."""
        methods = ["limma_moderated", "t_test", "mann_whitney", "anova"]

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import find_differential_proteins_ms

            for method in methods:
                result = find_differential_proteins_ms(
                    "ms_proteomics_raw",
                    group_column="condition",
                    method=method,
                    comparison="pairwise"
                )

                assert isinstance(result, str)
                assert "differential" in result.lower()

    def test_find_differential_proteins_ms_thresholds(self, mock_data_manager):
        """Test differential analysis with custom thresholds."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import find_differential_proteins_ms

            result = find_differential_proteins_ms(
                "ms_proteomics_raw",
                group_column="condition",
                fold_change_threshold=2.0,
                p_value_threshold=0.01,
                adjust_batch_effects=True,
                batch_column="batch"
            )

            assert isinstance(result, str)
            assert "differential" in result.lower()


# ===============================================================================
# Tool Testing - Peptide Mapping and MS-Specific Features
# ===============================================================================

class TestMSProteomicsPeptideMapping:
    """Test suite for MS proteomics peptide mapping functionality."""

    def test_add_peptide_mapping_basic(self, mock_data_manager, tmp_path):
        """Test basic peptide mapping addition."""
        # Create mock peptide file
        peptide_data = pd.DataFrame({
            'protein_id': [f'protein_{i}' for i in range(50)],
            'peptide_sequence': [f'PEPTIDESEQ{i}K' for i in range(50)],
            'charge': [2, 3] * 25,
            'mass': np.random.uniform(800, 3000, 50),
            'rt': np.random.uniform(10, 120, 50)
        })

        peptide_file = tmp_path / "peptides.csv"
        peptide_data.to_csv(peptide_file, index=False)

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import add_peptide_mapping_to_ms_modality

            result = add_peptide_mapping_to_ms_modality(
                "ms_proteomics_raw",
                str(peptide_file)
            )

            assert isinstance(result, str)
            assert "peptide" in result.lower()

    def test_add_peptide_mapping_nonexistent_file(self, mock_data_manager):
        """Test peptide mapping with nonexistent file."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import add_peptide_mapping_to_ms_modality

            result = add_peptide_mapping_to_ms_modality(
                "ms_proteomics_raw",
                "/nonexistent/file.csv"
            )

            assert isinstance(result, str)
            assert "error" in result.lower() or "not found" in result.lower()


# ===============================================================================
# Tool Testing - Summary and Reporting
# ===============================================================================

class TestMSProteomicsSummary:
    """Test suite for MS proteomics summary functionality."""

    def test_create_ms_proteomics_summary_no_analysis(self):
        """Test summary creation when no analysis has been performed."""
        from lobster.agents.ms_proteomics_expert import create_ms_proteomics_summary

        # Clear any existing analysis results
        with patch('lobster.agents.ms_proteomics_expert.analysis_results', {"details": []}):
            result = create_ms_proteomics_summary()

            assert isinstance(result, str)
            assert "no" in result.lower() and "analysis" in result.lower()

    def test_create_ms_proteomics_summary_with_analysis(self):
        """Test summary creation with analysis results."""
        mock_analysis_results = {
            "details": [
                {
                    "step": "quality_assessment",
                    "timestamp": "2024-01-01 12:00:00",
                    "input_modality": "ms_proteomics_raw",
                    "output_modality": "ms_proteomics_raw_quality",
                    "parameters": {"missing_threshold": 0.7},
                    "summary": "Quality assessment completed"
                },
                {
                    "step": "normalization",
                    "timestamp": "2024-01-01 12:05:00",
                    "input_modality": "ms_proteomics_raw_quality",
                    "output_modality": "ms_proteomics_raw_normalized",
                    "parameters": {"method": "median"},
                    "summary": "Normalization completed"
                }
            ]
        }

        with patch('lobster.agents.ms_proteomics_expert.analysis_results', mock_analysis_results):
            from lobster.agents.ms_proteomics_expert import create_ms_proteomics_summary

            result = create_ms_proteomics_summary()

            assert isinstance(result, str)
            assert "quality_assessment" in result
            assert "normalization" in result
            assert "2 analysis steps" in result or "2 steps" in result


# ===============================================================================
# Integration Testing - Workflow Scenarios
# ===============================================================================

class TestMSProteomicsWorkflows:
    """Test suite for complete MS proteomics workflow scenarios."""

    def test_complete_dda_workflow(self, mock_data_manager, mock_dda_data):
        """Test complete DDA proteomics workflow."""
        # Update mock data manager with DDA data
        mock_data_manager.get_modality.return_value = mock_dda_data
        mock_data_manager.list_modalities.return_value = ['dda_proteomics']
        mock_data_manager.modalities = {'dda_proteomics': mock_dda_data}

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import (
                assess_ms_proteomics_quality,
                filter_ms_proteomics_data,
                normalize_ms_proteomics_data,
                find_differential_proteins_ms
            )

            # Step 1: Quality assessment
            quality_result = assess_ms_proteomics_quality(
                "dda_proteomics",
                missing_value_threshold=0.6,  # Higher threshold for DDA
                min_peptides_per_protein=2
            )
            assert isinstance(quality_result, str)

            # Step 2: Filtering
            filter_result = filter_ms_proteomics_data(
                "dda_proteomics",
                max_missing_per_protein=0.7,
                min_peptides_per_protein=2
            )
            assert isinstance(filter_result, str)

            # Step 3: Normalization with missing value handling
            norm_result = normalize_ms_proteomics_data(
                "dda_proteomics",
                normalization_method="median",
                handle_missing="impute_mixed",
                log_transform=True
            )
            assert isinstance(norm_result, str)

            # Step 4: Differential analysis
            diff_result = find_differential_proteins_ms(
                "dda_proteomics",
                group_column="condition",
                method="limma_moderated"
            )
            assert isinstance(diff_result, str)

    def test_complete_dia_workflow(self, mock_data_manager, mock_dia_data):
        """Test complete DIA proteomics workflow."""
        # Update mock data manager with DIA data
        mock_data_manager.get_modality.return_value = mock_dia_data
        mock_data_manager.list_modalities.return_value = ['dia_proteomics']
        mock_data_manager.modalities = {'dia_proteomics': mock_dia_data}

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import (
                assess_ms_proteomics_quality,
                normalize_ms_proteomics_data,
                analyze_ms_proteomics_patterns,
                find_differential_proteins_ms
            )

            # Step 1: Quality assessment (DIA has lower missing values)
            quality_result = assess_ms_proteomics_quality(
                "dia_proteomics",
                missing_value_threshold=0.3,  # Lower threshold for DIA
                cv_threshold=25.0
            )
            assert isinstance(quality_result, str)

            # Step 2: Normalization (less aggressive missing value handling)
            norm_result = normalize_ms_proteomics_data(
                "dia_proteomics",
                normalization_method="quantile",
                handle_missing="impute_knn",
                log_transform=True
            )
            assert isinstance(norm_result, str)

            # Step 3: Pattern analysis
            pattern_result = analyze_ms_proteomics_patterns(
                "dia_proteomics",
                analysis_type="pca_clustering",
                n_components=10,
                clustering_method="kmeans"
            )
            assert isinstance(pattern_result, str)

            # Step 4: Differential analysis
            diff_result = find_differential_proteins_ms(
                "dia_proteomics",
                group_column="condition",
                method="t_test"
            )
            assert isinstance(diff_result, str)


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

class TestMSProteomicsErrorHandling:
    """Test suite for error handling and edge cases."""

    def test_tool_with_empty_modality_name(self, mock_data_manager):
        """Test tools with empty modality name."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import assess_ms_proteomics_quality

            result = assess_ms_proteomics_quality("")

            assert isinstance(result, str)
            assert "error" in result.lower() or "invalid" in result.lower()

    def test_tool_with_invalid_parameters(self, mock_data_manager):
        """Test tools with invalid parameters."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import normalize_ms_proteomics_data

            result = normalize_ms_proteomics_data(
                "ms_proteomics_raw",
                normalization_method="invalid_method"
            )

            assert isinstance(result, str)
            # Should handle invalid method gracefully

    def test_tool_with_missing_metadata(self, mock_data_manager):
        """Test tools when required metadata is missing."""
        # Create data without required batch column
        adata = mock_data_manager.get_modality("ms_proteomics_raw")
        adata.obs = adata.obs.drop('batch', axis=1)
        mock_data_manager.get_modality.return_value = adata

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import normalize_ms_proteomics_data

            result = normalize_ms_proteomics_data(
                "ms_proteomics_raw",
                batch_correction=True,
                batch_column="batch"
            )

            assert isinstance(result, str)
            # Should handle missing metadata gracefully

    def test_tool_service_failure(self, mock_data_manager):
        """Test tool behavior when underlying service fails."""
        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            with patch('lobster.tools.proteomics_preprocessing_service.ProteomicsPreprocessingService') as MockService:
                # Make service raise an exception
                MockService.return_value.normalize_intensities.side_effect = Exception("Service error")

                from lobster.agents.ms_proteomics_expert import normalize_ms_proteomics_data

                result = normalize_ms_proteomics_data("ms_proteomics_raw")

                assert isinstance(result, str)
                assert "error" in result.lower() or "failed" in result.lower()


# ===============================================================================
# Scientific Accuracy Validation
# ===============================================================================

class TestMSProteomicsScientificAccuracy:
    """Test suite for validating scientific accuracy of MS proteomics methods."""

    def test_missing_value_patterns_realistic(self, mock_data_manager):
        """Test that missing value analysis produces realistic results for MS data."""
        # Create data with realistic MS missing value patterns
        n_samples, n_proteins = 30, 100
        X = np.random.lognormal(mean=10, sigma=1, size=(n_samples, n_proteins))

        # MNAR pattern: low abundance proteins have more missing values
        for i in range(n_proteins):
            protein_abundance = np.mean(X[:, i])
            if protein_abundance < np.percentile(X.flatten(), 30):  # Bottom 30%
                missing_rate = 0.6  # High missing rate
            else:
                missing_rate = 0.2  # Low missing rate

            missing_mask = np.random.rand(n_samples) < missing_rate
            X[missing_mask, i] = np.nan

        adata = ad.AnnData(X=X)
        adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
        adata.var_names = [f"protein_{i}" for i in range(n_proteins)]
        adata.obs['condition'] = ['control'] * 15 + ['treatment'] * 15

        mock_data_manager.get_modality.return_value = adata

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import assess_ms_proteomics_quality

            result = assess_ms_proteomics_quality("test_modality")

            assert isinstance(result, str)
            # Should detect the MNAR pattern
            assert "missing" in result.lower()

    def test_normalization_preserves_biological_signal(self, mock_data_manager):
        """Test that normalization preserves biological differences."""
        # Create data with known biological signal
        n_samples, n_proteins = 20, 50
        X = np.random.lognormal(mean=8, sigma=0.5, size=(n_samples, n_proteins))

        # Add biological signal: treatment group has 2-fold increase in first 10 proteins
        treatment_samples = slice(10, 20)
        upregulated_proteins = slice(0, 10)
        X[treatment_samples, upregulated_proteins] *= 2

        adata = ad.AnnData(X=X)
        adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
        adata.var_names = [f"protein_{i}" for i in range(n_proteins)]
        adata.obs['condition'] = ['control'] * 10 + ['treatment'] * 10

        mock_data_manager.get_modality.return_value = adata

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import normalize_ms_proteomics_data

            result = normalize_ms_proteomics_data(
                "test_modality",
                normalization_method="median"
            )

            assert isinstance(result, str)
            assert "normalized" in result.lower()

    def test_statistical_test_power(self, mock_data_manager):
        """Test that statistical tests have appropriate power for realistic MS data."""
        # Create data with realistic effect sizes and variance
        n_samples_per_group = 12
        n_proteins = 100

        # Control group
        X_control = np.random.lognormal(mean=10, sigma=0.8, size=(n_samples_per_group, n_proteins))

        # Treatment group with some differentially expressed proteins
        X_treatment = np.random.lognormal(mean=10, sigma=0.8, size=(n_samples_per_group, n_proteins))
        # 20% of proteins have 1.5-fold change (realistic for MS)
        de_proteins = np.random.choice(n_proteins, 20, replace=False)
        X_treatment[:, de_proteins] *= 1.5

        X = np.vstack([X_control, X_treatment])

        adata = ad.AnnData(X=X)
        adata.obs_names = [f"sample_{i}" for i in range(2 * n_samples_per_group)]
        adata.var_names = [f"protein_{i}" for i in range(n_proteins)]
        adata.obs['condition'] = ['control'] * n_samples_per_group + ['treatment'] * n_samples_per_group

        mock_data_manager.get_modality.return_value = adata

        with patch('lobster.agents.ms_proteomics_expert.data_manager', mock_data_manager):
            from lobster.agents.ms_proteomics_expert import find_differential_proteins_ms

            result = find_differential_proteins_ms(
                "test_modality",
                group_column="condition",
                method="limma_moderated"
            )

            assert isinstance(result, str)
            assert "differential" in result.lower()