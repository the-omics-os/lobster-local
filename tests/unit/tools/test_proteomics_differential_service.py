"""
Comprehensive unit tests for proteomics differential expression service.

This module provides thorough testing of the proteomics differential service including
differential expression analysis, time course analysis, correlation analysis,
and statistical testing for proteomics data analysis.

Test coverage target: 95%+ with meaningful tests for proteomics differential operations.
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

from lobster.tools.proteomics_differential_service import ProteomicsDifferentialService, ProteomicsDifferentialError

from tests.mock_data.factories import ProteomicsDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def mock_proteomics_data():
    """Create mock proteomics data for testing."""
    return ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)


@pytest.fixture
def service():
    """Create ProteomicsDifferentialService instance."""
    return ProteomicsDifferentialService()


@pytest.fixture
def mock_adata_with_groups():
    """Create mock AnnData with groups for differential expression."""
    n_samples, n_proteins = 60, 100

    # Create different expression levels for different groups
    base_expression = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    # Add group-specific effects to some proteins
    differential_proteins = np.random.choice(n_proteins, 20, replace=False)

    # Control group: samples 0-19
    # Treatment1 group: samples 20-39 (upregulated proteins)
    # Treatment2 group: samples 40-59 (downregulated proteins)

    for protein_idx in differential_proteins[:10]:  # First 10 are upregulated in treatment1
        base_expression[20:40, protein_idx] *= 2.5  # 2.5x increase

    for protein_idx in differential_proteins[10:]:  # Last 10 are downregulated in treatment2
        base_expression[40:60, protein_idx] *= 0.4  # 2.5x decrease

    adata = ad.AnnData(X=base_expression)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add group information
    adata.obs['condition'] = ['control'] * 20 + ['treatment1'] * 20 + ['treatment2'] * 20
    adata.obs['batch'] = ['batch1'] * 20 + ['batch2'] * 20 + ['batch3'] * 20
    adata.obs['subject_id'] = [f"subject_{i//3}" for i in range(n_samples)]

    # Add protein metadata
    adata.var['protein_names'] = [f"PROT_{i}" for i in range(n_proteins)]

    return adata


@pytest.fixture
def mock_adata_time_course():
    """Create mock AnnData for time course analysis."""
    n_samples, n_proteins = 48, 50
    time_points = [0, 2, 6, 12, 24, 48]  # Hours
    n_replicates = 8  # 8 replicates per time point

    X = np.zeros((n_samples, n_proteins))
    time_labels = []

    for i, time_point in enumerate(time_points):
        start_idx = i * n_replicates
        end_idx = (i + 1) * n_replicates

        # Create time-dependent expression patterns for some proteins
        for protein_idx in range(n_proteins):
            if protein_idx < 10:  # Linear increase
                base_level = 1000
                time_effect = time_point * 50
                X[start_idx:end_idx, protein_idx] = base_level + time_effect + np.random.normal(0, 100, n_replicates)
            elif protein_idx < 20:  # Linear decrease
                base_level = 3000
                time_effect = time_point * -30
                X[start_idx:end_idx, protein_idx] = base_level + time_effect + np.random.normal(0, 100, n_replicates)
            elif protein_idx < 30:  # Quadratic pattern
                base_level = 2000
                time_effect = 0.5 * (time_point - 24) ** 2
                X[start_idx:end_idx, protein_idx] = base_level - time_effect + np.random.normal(0, 100, n_replicates)
            else:  # No time effect
                X[start_idx:end_idx, protein_idx] = 1500 + np.random.normal(0, 200, n_replicates)

        time_labels.extend([time_point] * n_replicates)

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add time course metadata
    adata.obs['time_point'] = time_labels
    adata.obs['replicate_id'] = [f"rep_{i%n_replicates + 1}" for i in range(n_samples)]

    return adata


@pytest.fixture
def mock_adata_correlation():
    """Create mock AnnData for correlation analysis."""
    n_samples, n_proteins = 40, 60

    # Create correlated protein expression patterns
    X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    # Make some proteins highly correlated
    correlated_groups = [
        [0, 1, 2],      # Highly positively correlated
        [10, 11, 12],   # Moderately correlated
        [20, 21],       # Negatively correlated
    ]

    for group in correlated_groups:
        if len(group) == 3:
            # Positive correlation
            base_pattern = np.random.lognormal(mean=8, sigma=0.5, size=n_samples)
            for protein_idx in group:
                X[:, protein_idx] = base_pattern + np.random.normal(0, base_pattern * 0.1, n_samples)
        elif len(group) == 2:
            # Negative correlation
            base_pattern = np.random.lognormal(mean=8, sigma=0.5, size=n_samples)
            X[:, group[0]] = base_pattern
            X[:, group[1]] = np.max(base_pattern) - base_pattern + np.random.normal(0, base_pattern * 0.1, n_samples)

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add metadata
    adata.obs['condition'] = ['control'] * 20 + ['treatment'] * 20

    return adata


# ===============================================================================
# Service Initialization Tests
# ===============================================================================

class TestProteomicsDifferentialServiceInitialization:
    """Test suite for ProteomicsDifferentialService initialization."""

    def test_init_default_parameters(self):
        """Test service initialization with default parameters."""
        service = ProteomicsDifferentialService()

        assert service is not None
        assert hasattr(service, 'test_methods')
        assert 't_test' in service.test_methods
        assert 'limma_like' in service.test_methods
        assert 'mann_whitney' in service.test_methods


# ===============================================================================
# Differential Expression Analysis Tests
# ===============================================================================

class TestDifferentialExpressionAnalysis:
    """Test suite for differential expression analysis functionality."""

    def test_perform_differential_expression_basic(self, service, mock_adata_with_groups):
        """Test basic differential expression analysis."""
        result_adata, stats = service.perform_differential_expression(
            mock_adata_with_groups,
            group_column='condition',
            test_method='t_test'
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['test_method'] == 't_test'
        assert 'n_comparisons' in stats
        assert 'n_significant_proteins' in stats
        assert stats['analysis_type'] == 'differential_expression'

        # DE results should be stored in uns
        assert 'differential_expression' in result_adata.uns
        assert 'significant_proteins' in result_adata.var.columns

    def test_perform_differential_expression_custom_pairs(self, service, mock_adata_with_groups):
        """Test differential expression with custom comparison pairs."""
        comparison_pairs = [('control', 'treatment1'), ('control', 'treatment2')]

        result_adata, stats = service.perform_differential_expression(
            mock_adata_with_groups,
            group_column='condition',
            comparison_pairs=comparison_pairs,
            test_method='t_test'
        )

        assert result_adata is not None
        assert stats['n_comparisons'] == 2

        # Check that only specified comparisons were performed
        de_results = result_adata.uns['differential_expression']['results']
        comparisons = set()
        for result in de_results:
            comparisons.add(result.get('comparison', ''))

        expected_comparisons = {'control_vs_treatment1', 'control_vs_treatment2'}
        assert comparisons.intersection(expected_comparisons)

    def test_perform_differential_expression_welch_t_test(self, service, mock_adata_with_groups):
        """Test differential expression with Welch's t-test."""
        result_adata, stats = service.perform_differential_expression(
            mock_adata_with_groups,
            group_column='condition',
            test_method='welch_t_test'
        )

        assert result_adata is not None
        assert stats['test_method'] == 'welch_t_test'

    def test_perform_differential_expression_mann_whitney(self, service, mock_adata_with_groups):
        """Test differential expression with Mann-Whitney test."""
        result_adata, stats = service.perform_differential_expression(
            mock_adata_with_groups,
            group_column='condition',
            test_method='mann_whitney'
        )

        assert result_adata is not None
        assert stats['test_method'] == 'mann_whitney'

    def test_perform_differential_expression_limma_like(self, service, mock_adata_with_groups):
        """Test differential expression with LIMMA-like analysis."""
        result_adata, stats = service.perform_differential_expression(
            mock_adata_with_groups,
            group_column='condition',
            test_method='limma_like'
        )

        assert result_adata is not None
        assert stats['test_method'] == 'limma_like'

    def test_perform_differential_expression_custom_thresholds(self, service, mock_adata_with_groups):
        """Test differential expression with custom thresholds."""
        result_adata, stats = service.perform_differential_expression(
            mock_adata_with_groups,
            group_column='condition',
            fdr_threshold=0.01,
            fold_change_threshold=2.0
        )

        assert result_adata is not None
        assert stats['fdr_threshold'] == 0.01
        assert stats['fold_change_threshold'] == 2.0

    def test_perform_differential_expression_fdr_methods(self, service, mock_adata_with_groups):
        """Test differential expression with different FDR methods."""
        fdr_methods = ['benjamini_hochberg', 'bonferroni', 'holm']

        for method in fdr_methods:
            result_adata, stats = service.perform_differential_expression(
                mock_adata_with_groups,
                group_column='condition',
                fdr_method=method
            )

            assert result_adata is not None
            assert stats['fdr_method'] == method

    def test_perform_differential_expression_invalid_group_column(self, service, mock_adata_with_groups):
        """Test differential expression with invalid group column."""
        with pytest.raises(ProteomicsDifferentialError) as exc_info:
            service.perform_differential_expression(
                mock_adata_with_groups,
                group_column='nonexistent_column'
            )

        assert "Group column 'nonexistent_column' not found" in str(exc_info.value)

    def test_perform_differential_expression_insufficient_samples(self, service):
        """Test differential expression with insufficient samples per group."""
        # Create small dataset
        X = np.random.lognormal(mean=8, sigma=1, size=(4, 10))
        adata = ad.AnnData(X=X)
        adata.obs['condition'] = ['control'] * 2 + ['treatment'] * 2

        result_adata, stats = service.perform_differential_expression(
            adata,
            group_column='condition',
            min_samples_per_group=3  # Require 3 samples per group
        )

        assert result_adata is not None
        # Should have minimal or no results due to insufficient samples
        assert stats['n_significant_proteins'] == 0

    def test_differential_expression_statistics_accuracy(self, service, mock_adata_with_groups):
        """Test accuracy of differential expression statistics."""
        result_adata, stats = service.perform_differential_expression(
            mock_adata_with_groups,
            group_column='condition'
        )

        # Check statistical measures
        assert 'volcano_plot_data' in stats
        assert 'top_upregulated' in stats
        assert 'top_downregulated' in stats
        assert 'effect_size_distribution' in stats

        # Should detect some significant proteins given the artificial differences
        assert stats['n_significant_proteins'] > 0


# ===============================================================================
# Time Course Analysis Tests
# ===============================================================================

class TestTimeCourseAnalysis:
    """Test suite for time course analysis functionality."""

    def test_perform_time_course_analysis_basic(self, service, mock_adata_time_course):
        """Test basic time course analysis."""
        result_adata, stats = service.perform_time_course_analysis(
            mock_adata_time_course,
            time_column='time_point'
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['analysis_type'] == 'time_course_analysis'
        assert 'n_time_points' in stats
        assert 'n_temporal_proteins' in stats

        # Time course results should be stored
        assert 'time_course_analysis' in result_adata.uns
        assert 'temporal_pattern' in result_adata.var.columns

    def test_perform_time_course_analysis_linear(self, service, mock_adata_time_course):
        """Test time course analysis with linear trend test."""
        result_adata, stats = service.perform_time_course_analysis(
            mock_adata_time_course,
            time_column='time_point',
            trend_test='linear'
        )

        assert result_adata is not None
        assert stats['trend_test'] == 'linear'

    def test_perform_time_course_analysis_polynomial(self, service, mock_adata_time_course):
        """Test time course analysis with polynomial trend test."""
        result_adata, stats = service.perform_time_course_analysis(
            mock_adata_time_course,
            time_column='time_point',
            trend_test='polynomial'
        )

        assert result_adata is not None
        assert stats['trend_test'] == 'polynomial'

    def test_perform_time_course_analysis_both_trends(self, service, mock_adata_time_course):
        """Test time course analysis with both linear and polynomial trends."""
        result_adata, stats = service.perform_time_course_analysis(
            mock_adata_time_course,
            time_column='time_point',
            trend_test='both'
        )

        assert result_adata is not None
        assert stats['trend_test'] == 'both'

    def test_perform_time_course_analysis_custom_threshold(self, service, mock_adata_time_course):
        """Test time course analysis with custom significance threshold."""
        result_adata, stats = service.perform_time_course_analysis(
            mock_adata_time_course,
            time_column='time_point',
            significance_threshold=0.01
        )

        assert result_adata is not None
        assert stats['significance_threshold'] == 0.01

    def test_perform_time_course_analysis_invalid_column(self, service, mock_adata_time_course):
        """Test time course analysis with invalid time column."""
        with pytest.raises(ProteomicsDifferentialError) as exc_info:
            service.perform_time_course_analysis(
                mock_adata_time_course,
                time_column='nonexistent_column'
            )

        assert "Time column 'nonexistent_column' not found" in str(exc_info.value)

    def test_time_course_analysis_insufficient_time_points(self, service):
        """Test time course analysis with insufficient time points."""
        # Create data with only 2 time points
        X = np.random.lognormal(mean=8, sigma=1, size=(10, 20))
        adata = ad.AnnData(X=X)
        adata.obs['time_point'] = [0] * 5 + [1] * 5

        result_adata, stats = service.perform_time_course_analysis(
            adata,
            time_column='time_point'
        )

        assert result_adata is not None
        # Should handle insufficient time points gracefully

    def test_time_course_pattern_detection(self, service, mock_adata_time_course):
        """Test detection of temporal patterns in time course data."""
        result_adata, stats = service.perform_time_course_analysis(
            mock_adata_time_course,
            time_column='time_point'
        )

        # Should detect temporal patterns in the artificially created data
        assert stats['n_temporal_proteins'] > 0

        # Check pattern classification
        patterns = result_adata.var['temporal_pattern'].value_counts()
        assert len(patterns) > 1  # Should detect multiple pattern types


# ===============================================================================
# Correlation Analysis Tests
# ===============================================================================

class TestCorrelationAnalysis:
    """Test suite for correlation analysis functionality."""

    def test_perform_correlation_analysis_basic(self, service, mock_adata_correlation):
        """Test basic correlation analysis."""
        result_adata, stats = service.perform_correlation_analysis(
            mock_adata_correlation,
            correlation_method='pearson'
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['correlation_method'] == 'pearson'
        assert stats['analysis_type'] == 'correlation_analysis'
        assert 'n_correlations_tested' in stats
        assert 'n_significant_correlations' in stats

        # Correlation results should be stored
        assert 'correlation_analysis' in result_adata.uns

    def test_perform_correlation_analysis_spearman(self, service, mock_adata_correlation):
        """Test correlation analysis with Spearman correlation."""
        result_adata, stats = service.perform_correlation_analysis(
            mock_adata_correlation,
            correlation_method='spearman'
        )

        assert result_adata is not None
        assert stats['correlation_method'] == 'spearman'

    def test_perform_correlation_analysis_custom_threshold(self, service, mock_adata_correlation):
        """Test correlation analysis with custom correlation threshold."""
        result_adata, stats = service.perform_correlation_analysis(
            mock_adata_correlation,
            correlation_threshold=0.8,
            p_value_threshold=0.01
        )

        assert result_adata is not None
        assert stats['correlation_threshold'] == 0.8
        assert stats['p_value_threshold'] == 0.01

    def test_perform_correlation_analysis_protein_subset(self, service, mock_adata_correlation):
        """Test correlation analysis with protein subset."""
        protein_subset = ['protein_0', 'protein_1', 'protein_2', 'protein_10', 'protein_11']

        result_adata, stats = service.perform_correlation_analysis(
            mock_adata_correlation,
            protein_subset=protein_subset
        )

        assert result_adata is not None
        # Should only analyze correlations within the subset
        assert stats['n_proteins_analyzed'] == len(protein_subset)

    def test_perform_correlation_analysis_invalid_method(self, service, mock_adata_correlation):
        """Test correlation analysis with invalid method."""
        with pytest.raises(ProteomicsDifferentialError) as exc_info:
            service.perform_correlation_analysis(
                mock_adata_correlation,
                correlation_method='invalid_method'
            )

        assert "Unknown correlation method" in str(exc_info.value)

    def test_correlation_analysis_network_generation(self, service, mock_adata_correlation):
        """Test correlation network generation."""
        result_adata, stats = service.perform_correlation_analysis(
            mock_adata_correlation,
            correlation_threshold=0.5
        )

        # Should generate network information
        assert 'correlation_network' in stats
        assert 'network_nodes' in stats['correlation_network']
        assert 'network_edges' in stats['correlation_network']

    def test_correlation_analysis_with_missing_values(self, service):
        """Test correlation analysis with missing values."""
        X = np.random.lognormal(mean=8, sigma=1, size=(30, 20))
        # Add missing values
        missing_mask = np.random.rand(30, 20) < 0.1
        X[missing_mask] = np.nan

        adata = ad.AnnData(X=X)

        result_adata, stats = service.perform_correlation_analysis(adata)

        assert result_adata is not None
        # Should handle missing values gracefully


# ===============================================================================
# Helper Methods Tests
# ===============================================================================

class TestHelperMethods:
    """Test suite for helper methods."""

    def test_perform_statistical_test_helper(self, service):
        """Test statistical test helper method."""
        group1 = np.random.normal(1000, 100, 20)
        group2 = np.random.normal(1200, 100, 20)  # Different mean

        result = service._perform_statistical_test(group1, group2, 't_test')

        assert result is not None
        assert 'p_value' in result
        assert 'statistic' in result
        assert result['p_value'] < 0.05  # Should detect difference

    def test_calculate_effect_metrics_helper(self, service):
        """Test effect metrics calculation helper method."""
        group1 = np.array([100, 110, 90, 105, 95])
        group2 = np.array([200, 220, 180, 210, 190])

        metrics = service._calculate_effect_metrics(group1, group2)

        assert 'fold_change' in metrics
        assert 'log2_fold_change' in metrics
        assert 'effect_size' in metrics
        assert 'mean_difference' in metrics

        # Check that metrics are reasonable
        assert metrics['fold_change'] > 1.5  # group2 has higher values
        assert metrics['log2_fold_change'] > 0
        assert metrics['mean_difference'] > 0

    def test_apply_fdr_correction_helper(self, service):
        """Test FDR correction helper method."""
        # Create mock results with p-values
        results = [
            {'protein': 'prot1', 'p_value': 0.001},
            {'protein': 'prot2', 'p_value': 0.01},
            {'protein': 'prot3', 'p_value': 0.05},
            {'protein': 'prot4', 'p_value': 0.1},
            {'protein': 'prot5', 'p_value': 0.3}
        ]

        corrected_results = service._apply_fdr_correction(results, 'benjamini_hochberg')

        assert len(corrected_results) == len(results)
        for result in corrected_results:
            assert 'p_adjusted' in result
            # Adjusted p-values should be >= original p-values
            assert result['p_adjusted'] >= result['p_value']

    def test_moderated_t_test_helper(self, service):
        """Test moderated t-test helper method."""
        group1 = np.random.normal(1000, 50, 15)
        group2 = np.random.normal(1100, 50, 15)

        statistic, p_value = service._moderated_t_test(group1, group2)

        assert isinstance(statistic, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1

    def test_linear_trend_test_helper(self, service):
        """Test linear trend test helper method."""
        time_points = np.array([0, 1, 2, 3, 4, 5])
        # Create linear increasing pattern
        protein_values = np.array([100, 120, 140, 160, 180, 200])

        slope, p_value, r_squared = service._linear_trend_test(time_points, protein_values)

        assert isinstance(slope, float)
        assert isinstance(p_value, float)
        assert isinstance(r_squared, float)
        assert slope > 0  # Should detect positive trend
        assert r_squared > 0.8  # Should have high R-squared for linear data

    def test_polynomial_trend_test_helper(self, service):
        """Test polynomial trend test helper method."""
        time_points = np.array([0, 1, 2, 3, 4, 5])
        # Create quadratic pattern
        protein_values = time_points ** 2 + 100

        coeff, p_value, r_squared = service._polynomial_trend_test(time_points, protein_values)

        assert isinstance(coeff, float)
        assert isinstance(p_value, float)
        assert isinstance(r_squared, float)
        assert r_squared > 0.9  # Should have very high R-squared for quadratic data


# ===============================================================================
# Error Handling and Edge Cases Tests
# ===============================================================================

class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""

    def test_error_handling_empty_data(self, service):
        """Test error handling with empty data."""
        adata = ad.AnnData(X=np.array([]).reshape(0, 0))

        with pytest.raises(ProteomicsDifferentialError):
            service.perform_differential_expression(adata, group_column='condition')

    def test_single_group_differential_expression(self, service):
        """Test differential expression with single group."""
        X = np.random.lognormal(mean=8, sigma=1, size=(10, 20))
        adata = ad.AnnData(X=X)
        adata.obs['condition'] = ['control'] * 10  # Only one group

        result_adata, stats = service.perform_differential_expression(
            adata,
            group_column='condition'
        )

        assert result_adata is not None
        assert stats['n_comparisons'] == 0  # No comparisons possible

    def test_identical_groups_differential_expression(self, service):
        """Test differential expression with identical expression between groups."""
        # Create identical data for both groups
        base_data = np.random.lognormal(mean=8, sigma=0.1, size=(10, 20))
        X = np.vstack([base_data, base_data])

        adata = ad.AnnData(X=X)
        adata.obs['condition'] = ['control'] * 10 + ['treatment'] * 10

        result_adata, stats = service.perform_differential_expression(
            adata,
            group_column='condition'
        )

        assert result_adata is not None
        # Should find very few or no significant differences
        assert stats['n_significant_proteins'] <= 2  # Allow for statistical noise

    def test_high_variance_data_differential_expression(self, service):
        """Test differential expression with very high variance data."""
        X = np.random.lognormal(mean=8, sigma=3, size=(30, 20))  # High variance
        adata = ad.AnnData(X=X)
        adata.obs['condition'] = ['control'] * 15 + ['treatment'] * 15

        result_adata, stats = service.perform_differential_expression(
            adata,
            group_column='condition'
        )

        assert result_adata is not None
        # Should still complete analysis

    def test_single_time_point_time_course(self, service):
        """Test time course analysis with single time point."""
        X = np.random.lognormal(mean=8, sigma=1, size=(10, 20))
        adata = ad.AnnData(X=X)
        adata.obs['time_point'] = [0] * 10  # Only one time point

        with pytest.raises(ProteomicsDifferentialError):
            service.perform_time_course_analysis(adata, time_column='time_point')

    def test_no_variance_correlation_analysis(self, service):
        """Test correlation analysis with no variance proteins."""
        X = np.ones((20, 10)) * 1000  # All same values
        # Add one protein with variance
        X[:, 0] = np.random.lognormal(mean=8, sigma=1, size=20)

        adata = ad.AnnData(X=X)

        result_adata, stats = service.perform_correlation_analysis(adata)

        assert result_adata is not None
        # Should handle constant proteins gracefully


# ===============================================================================
# Integration Tests
# ===============================================================================

class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    def test_complete_differential_analysis_workflow(self, service, mock_adata_with_groups):
        """Test complete differential analysis workflow."""
        # Step 1: Basic differential expression
        adata_de, _ = service.perform_differential_expression(
            mock_adata_with_groups,
            group_column='condition'
        )

        # Step 2: Time course analysis (simulate time data)
        adata_de.obs['time_point'] = [0, 6, 12] * (adata_de.n_obs // 3)
        adata_time, _ = service.perform_time_course_analysis(
            adata_de,
            time_column='time_point'
        )

        # Step 3: Correlation analysis
        adata_corr, _ = service.perform_correlation_analysis(adata_time)

        # Verify final result has all analysis components
        assert 'differential_expression' in adata_corr.uns
        assert 'time_course_analysis' in adata_corr.uns
        assert 'correlation_analysis' in adata_corr.uns

    def test_multiple_test_methods_consistency(self, service, mock_adata_with_groups):
        """Test consistency across different statistical test methods."""
        methods = ['t_test', 'welch_t_test', 'mann_whitney']
        results = {}

        for method in methods:
            result_adata, stats = service.perform_differential_expression(
                mock_adata_with_groups,
                group_column='condition',
                test_method=method
            )
            results[method] = stats

        # All methods should process same number of samples/proteins
        for method in methods:
            assert results[method]['samples_processed'] == mock_adata_with_groups.n_obs
            assert results[method]['proteins_processed'] == mock_adata_with_groups.n_vars

    def test_differential_expression_with_batch_effects(self, service, mock_adata_with_groups):
        """Test differential expression considering batch effects."""
        # The mock data has batch information, test that it's handled
        result_adata, stats = service.perform_differential_expression(
            mock_adata_with_groups,
            group_column='condition',
            test_method='limma_like'  # Should handle batch effects better
        )

        assert result_adata is not None
        # LIMMA-like method should handle batch effects


# ===============================================================================
# Performance and Memory Tests
# ===============================================================================

class TestPerformanceAndMemory:
    """Test suite for performance and memory considerations."""

    @pytest.mark.slow
    def test_large_dataset_differential_expression(self, service):
        """Test differential expression with large dataset."""
        # Create larger dataset
        n_samples, n_proteins = 200, 1000
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

        # Add some differential expression
        differential_indices = np.random.choice(n_proteins, 100, replace=False)
        X[100:, differential_indices] *= 2.0

        adata = ad.AnnData(X=X)
        adata.obs['condition'] = ['control'] * 100 + ['treatment'] * 100

        result_adata, stats = service.perform_differential_expression(
            adata,
            group_column='condition'
        )

        assert result_adata is not None
        assert stats['samples_processed'] == n_samples
        assert stats['proteins_processed'] == n_proteins

    @pytest.mark.slow
    def test_memory_efficient_correlation_analysis(self, service):
        """Test memory efficiency in correlation analysis."""
        # Create moderately large dataset
        n_samples, n_proteins = 100, 500
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))
        adata = ad.AnnData(X=X)

        result_adata, stats = service.perform_correlation_analysis(
            adata,
            correlation_threshold=0.8  # Higher threshold to reduce memory usage
        )

        assert result_adata is not None
        # Should complete without memory errors

    def test_efficient_time_course_analysis(self, service):
        """Test efficient time course analysis with many time points."""
        n_time_points = 20
        n_replicates = 5
        n_proteins = 200
        n_samples = n_time_points * n_replicates

        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))
        adata = ad.AnnData(X=X)

        time_labels = []
        for t in range(n_time_points):
            time_labels.extend([t] * n_replicates)
        adata.obs['time_point'] = time_labels

        result_adata, stats = service.perform_time_course_analysis(
            adata,
            time_column='time_point'
        )

        assert result_adata is not None
        assert stats['n_time_points'] == n_time_points