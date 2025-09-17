"""
Comprehensive unit tests for proteomics analysis service.

This module provides thorough testing of the proteomics analysis service including
statistical testing, dimensionality reduction, clustering analysis, and pathway
enrichment for proteomics data analysis.

Test coverage target: 95%+ with meaningful tests for proteomics analysis operations.
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

from lobster.tools.proteomics_analysis_service import ProteomicsAnalysisService, ProteomicsAnalysisError

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
    """Create ProteomicsAnalysisService instance."""
    return ProteomicsAnalysisService()


@pytest.fixture
def mock_adata_with_groups():
    """Create mock AnnData with group assignments."""
    n_samples, n_proteins = 48, 100
    X = np.random.randn(n_samples, n_proteins) + np.random.randn(n_samples, 1) * 0.1

    # Add some missing values
    missing_mask = np.random.rand(n_samples, n_proteins) < 0.1
    X[missing_mask] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add group assignments
    adata.obs['condition'] = ['control'] * 16 + ['treatment1'] * 16 + ['treatment2'] * 16
    adata.obs['batch'] = ['batch1'] * 24 + ['batch2'] * 24

    # Add some metadata
    adata.var['protein_names'] = [f"PROT_{i}" for i in range(n_proteins)]
    adata.var['intensity_cv'] = np.random.rand(n_proteins)

    return adata


@pytest.fixture
def mock_adata_with_stats():
    """Create mock AnnData with statistical test results."""
    adata = mock_adata_with_groups()

    # Add significance markers
    adata.var['is_significant'] = np.random.rand(adata.n_vars) < 0.2

    # Add statistical test results
    adata.uns['statistical_tests'] = {
        'results': [
            {
                'protein': 'protein_0',
                'protein_index': 0,
                'group1': 'control',
                'group2': 'treatment1',
                'p_value': 0.01,
                'p_adjusted': 0.05,
                'effect_size': 1.2
            }
        ],
        'parameters': {
            'group_column': 'condition',
            'test_method': 't_test'
        }
    }

    return adata


# ===============================================================================
# Service Initialization Tests
# ===============================================================================

class TestProteomicsAnalysisServiceInitialization:
    """Test suite for ProteomicsAnalysisService initialization."""

    def test_init_default_parameters(self):
        """Test service initialization with default parameters."""
        service = ProteomicsAnalysisService()

        assert service is not None
        assert hasattr(service, 'pathway_databases')
        assert 'go_biological_process' in service.pathway_databases
        assert 'kegg_pathway' in service.pathway_databases
        assert 'reactome' in service.pathway_databases


# ===============================================================================
# Statistical Testing Tests
# ===============================================================================

class TestStatisticalTesting:
    """Test suite for statistical testing functionality."""

    def test_perform_statistical_testing_basic(self, service, mock_adata_with_groups):
        """Test basic statistical testing with t-test."""
        result_adata, stats = service.perform_statistical_testing(
            mock_adata_with_groups,
            group_column='condition',
            test_method='t_test',
            comparison_type='all_pairs'
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['test_method'] == 't_test'
        assert stats['comparison_type'] == 'all_pairs'
        assert stats['n_groups'] == 3
        assert 'statistical_tests' in result_adata.uns
        assert 'is_significant' in result_adata.var.columns

    def test_perform_statistical_testing_mann_whitney(self, service, mock_adata_with_groups):
        """Test statistical testing with Mann-Whitney U test."""
        result_adata, stats = service.perform_statistical_testing(
            mock_adata_with_groups,
            group_column='condition',
            test_method='mann_whitney',
            comparison_type='all_pairs'
        )

        assert result_adata is not None
        assert stats['test_method'] == 'mann_whitney'
        assert 'statistical_tests' in result_adata.uns

    def test_perform_statistical_testing_multigroup_anova(self, service, mock_adata_with_groups):
        """Test multi-group statistical testing with ANOVA."""
        result_adata, stats = service.perform_statistical_testing(
            mock_adata_with_groups,
            group_column='condition',
            test_method='anova',
            comparison_type='multi_group'
        )

        assert result_adata is not None
        assert stats['test_method'] == 'anova'
        assert stats['comparison_type'] == 'multi_group'

    def test_perform_statistical_testing_kruskal(self, service, mock_adata_with_groups):
        """Test multi-group statistical testing with Kruskal-Wallis."""
        result_adata, stats = service.perform_statistical_testing(
            mock_adata_with_groups,
            group_column='condition',
            test_method='kruskal',
            comparison_type='multi_group'
        )

        assert result_adata is not None
        assert stats['test_method'] == 'kruskal'

    def test_perform_statistical_testing_missing_values_skip(self, service, mock_adata_with_groups):
        """Test statistical testing with missing values (skip method)."""
        result_adata, stats = service.perform_statistical_testing(
            mock_adata_with_groups,
            group_column='condition',
            handle_missing='skip'
        )

        assert result_adata is not None
        assert stats['handle_missing'] == 'skip'

    def test_perform_statistical_testing_missing_values_impute_mean(self, service, mock_adata_with_groups):
        """Test statistical testing with missing values (mean imputation)."""
        result_adata, stats = service.perform_statistical_testing(
            mock_adata_with_groups,
            group_column='condition',
            handle_missing='impute_mean'
        )

        assert result_adata is not None
        assert stats['handle_missing'] == 'impute_mean'

    def test_perform_statistical_testing_missing_values_impute_median(self, service, mock_adata_with_groups):
        """Test statistical testing with missing values (median imputation)."""
        result_adata, stats = service.perform_statistical_testing(
            mock_adata_with_groups,
            group_column='condition',
            handle_missing='impute_median'
        )

        assert result_adata is not None
        assert stats['handle_missing'] == 'impute_median'

    def test_perform_statistical_testing_invalid_group_column(self, service, mock_adata_with_groups):
        """Test statistical testing with invalid group column."""
        with pytest.raises(ProteomicsAnalysisError) as exc_info:
            service.perform_statistical_testing(
                mock_adata_with_groups,
                group_column='nonexistent_column'
            )

        assert "Group column 'nonexistent_column' not found" in str(exc_info.value)

    def test_perform_statistical_testing_min_observations(self, service):
        """Test statistical testing with minimum observations requirement."""
        # Create small dataset that doesn't meet minimum observations
        X = np.random.randn(4, 10)  # Only 4 samples
        adata = ad.AnnData(X=X)
        adata.obs['condition'] = ['control'] * 2 + ['treatment'] * 2

        result_adata, stats = service.perform_statistical_testing(
            adata,
            group_column='condition',
            min_observations=5  # Require 5 observations per group
        )

        assert result_adata is not None
        assert stats['n_tests_performed'] == 0  # No tests should be performed

    def test_statistical_testing_statistics_calculation(self, service, mock_adata_with_groups):
        """Test statistical testing statistics calculation."""
        result_adata, stats = service.perform_statistical_testing(
            mock_adata_with_groups,
            group_column='condition'
        )

        assert 'n_tests_performed' in stats
        assert 'n_significant_results' in stats
        assert 'significance_rate' in stats
        assert 'group_sizes' in stats
        assert 'samples_processed' in stats
        assert 'proteins_processed' in stats
        assert stats['analysis_type'] == 'statistical_testing'


# ===============================================================================
# Dimensionality Reduction Tests
# ===============================================================================

class TestDimensionalityReduction:
    """Test suite for dimensionality reduction functionality."""

    def test_perform_dimensionality_reduction_pca(self, service, mock_adata_with_groups):
        """Test PCA dimensionality reduction."""
        result_adata, stats = service.perform_dimensionality_reduction(
            mock_adata_with_groups,
            method='pca',
            n_components=10
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert 'X_pca' in result_adata.obsm
        assert result_adata.obsm['X_pca'].shape[1] == 10
        assert stats['method'] == 'pca'
        assert stats['n_components'] == 10
        assert 'variance_explained' in stats

    def test_perform_dimensionality_reduction_tsne(self, service, mock_adata_with_groups):
        """Test t-SNE dimensionality reduction."""
        result_adata, stats = service.perform_dimensionality_reduction(
            mock_adata_with_groups,
            method='tsne',
            n_components=2,
            perplexity=10.0
        )

        assert result_adata is not None
        assert 'X_tsne' in result_adata.obsm
        assert result_adata.obsm['X_tsne'].shape[1] == 2
        assert stats['method'] == 'tsne'
        assert stats['perplexity'] == 10.0

    def test_perform_dimensionality_reduction_umap_like(self, service, mock_adata_with_groups):
        """Test UMAP-like dimensionality reduction."""
        result_adata, stats = service.perform_dimensionality_reduction(
            mock_adata_with_groups,
            method='umap_like',
            n_components=2
        )

        assert result_adata is not None
        assert 'X_umap_like' in result_adata.obsm
        assert result_adata.obsm['X_umap_like'].shape[1] == 2
        assert stats['method'] == 'umap_like'

    def test_perform_dimensionality_reduction_invalid_method(self, service, mock_adata_with_groups):
        """Test dimensionality reduction with invalid method."""
        with pytest.raises(ProteomicsAnalysisError) as exc_info:
            service.perform_dimensionality_reduction(
                mock_adata_with_groups,
                method='invalid_method'
            )

        assert "Unknown dimensionality reduction method" in str(exc_info.value)

    def test_dimensionality_reduction_with_missing_values(self, service, mock_adata_with_groups):
        """Test dimensionality reduction handles missing values correctly."""
        # Add more missing values
        mock_adata_with_groups.X[0:5, 0:10] = np.nan

        result_adata, stats = service.perform_dimensionality_reduction(
            mock_adata_with_groups,
            method='pca'
        )

        assert result_adata is not None
        assert 'X_pca' in result_adata.obsm
        assert not np.any(np.isnan(result_adata.obsm['X_pca']))

    def test_dimensionality_reduction_statistics(self, service, mock_adata_with_groups):
        """Test dimensionality reduction statistics calculation."""
        result_adata, stats = service.perform_dimensionality_reduction(
            mock_adata_with_groups,
            method='pca'
        )

        assert 'input_dimensions' in stats
        assert 'output_dimensions' in stats
        assert 'samples_processed' in stats
        assert 'proteins_processed' in stats
        assert stats['analysis_type'] == 'dimensionality_reduction'


# ===============================================================================
# Clustering Analysis Tests
# ===============================================================================

class TestClusteringAnalysis:
    """Test suite for clustering analysis functionality."""

    def test_perform_clustering_analysis_kmeans(self, service, mock_adata_with_groups):
        """Test K-means clustering analysis."""
        result_adata, stats = service.perform_clustering_analysis(
            mock_adata_with_groups,
            clustering_method='kmeans',
            n_clusters=3
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert 'cluster' in result_adata.obs.columns
        assert 'cluster_numeric' in result_adata.obs.columns
        assert len(result_adata.obs['cluster'].unique()) == 3
        assert stats['clustering_method'] == 'kmeans'
        assert stats['n_clusters'] == 3
        assert 'clustering' in result_adata.uns

    def test_perform_clustering_analysis_hierarchical(self, service, mock_adata_with_groups):
        """Test hierarchical clustering analysis."""
        result_adata, stats = service.perform_clustering_analysis(
            mock_adata_with_groups,
            clustering_method='hierarchical',
            n_clusters=4
        )

        assert result_adata is not None
        assert 'cluster' in result_adata.obs.columns
        assert len(result_adata.obs['cluster'].unique()) == 4
        assert stats['clustering_method'] == 'hierarchical'

    def test_perform_clustering_analysis_gaussian_mixture(self, service, mock_adata_with_groups):
        """Test Gaussian Mixture Model clustering analysis."""
        result_adata, stats = service.perform_clustering_analysis(
            mock_adata_with_groups,
            clustering_method='gaussian_mixture',
            n_clusters=2
        )

        assert result_adata is not None
        assert 'cluster' in result_adata.obs.columns
        assert len(result_adata.obs['cluster'].unique()) == 2
        assert stats['clustering_method'] == 'gaussian_mixture'

    def test_perform_clustering_analysis_with_pca(self, service, mock_adata_with_groups):
        """Test clustering analysis with PCA preprocessing."""
        result_adata, stats = service.perform_clustering_analysis(
            mock_adata_with_groups,
            clustering_method='kmeans',
            use_pca=True,
            n_pca_components=20
        )

        assert result_adata is not None
        assert 'X_pca_clustering' in result_adata.obsm
        assert result_adata.obsm['X_pca_clustering'].shape[1] == 20
        assert stats['use_pca'] is True
        assert stats['n_pca_components'] == 20

    def test_perform_clustering_analysis_without_pca(self, service, mock_adata_with_groups):
        """Test clustering analysis without PCA preprocessing."""
        result_adata, stats = service.perform_clustering_analysis(
            mock_adata_with_groups,
            clustering_method='kmeans',
            use_pca=False
        )

        assert result_adata is not None
        assert stats['use_pca'] is False
        assert stats['n_pca_components'] is None

    def test_perform_clustering_analysis_invalid_method(self, service, mock_adata_with_groups):
        """Test clustering analysis with invalid method."""
        with pytest.raises(ProteomicsAnalysisError) as exc_info:
            service.perform_clustering_analysis(
                mock_adata_with_groups,
                clustering_method='invalid_method'
            )

        assert "Unknown clustering method" in str(exc_info.value)

    def test_clustering_analysis_statistics(self, service, mock_adata_with_groups):
        """Test clustering analysis statistics calculation."""
        result_adata, stats = service.perform_clustering_analysis(
            mock_adata_with_groups,
            clustering_method='kmeans'
        )

        assert 'cluster_sizes' in stats
        assert 'clustering_quality' in stats
        assert 'samples_processed' in stats
        assert 'proteins_processed' in stats
        assert stats['analysis_type'] == 'clustering_analysis'


# ===============================================================================
# Pathway Enrichment Tests
# ===============================================================================

class TestPathwayEnrichment:
    """Test suite for pathway enrichment functionality."""

    def test_perform_pathway_enrichment_default_proteins(self, service, mock_adata_with_stats):
        """Test pathway enrichment with default protein list (significant proteins)."""
        result_adata, stats = service.perform_pathway_enrichment(
            mock_adata_with_stats,
            database='go_biological_process'
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert 'pathway_enrichment' in result_adata.uns
        assert stats['database'] == 'go_biological_process'
        assert 'n_query_proteins' in stats
        assert 'n_pathways_tested' in stats

    def test_perform_pathway_enrichment_custom_proteins(self, service, mock_adata_with_groups):
        """Test pathway enrichment with custom protein list."""
        protein_list = ['protein_0', 'protein_1', 'protein_2', 'protein_3']

        result_adata, stats = service.perform_pathway_enrichment(
            mock_adata_with_groups,
            protein_list=protein_list,
            database='kegg_pathway'
        )

        assert result_adata is not None
        assert stats['database'] == 'kegg_pathway'
        assert stats['n_query_proteins'] == len(protein_list)

    def test_perform_pathway_enrichment_different_databases(self, service, mock_adata_with_groups):
        """Test pathway enrichment with different databases."""
        databases = ['go_biological_process', 'kegg_pathway', 'reactome']

        for db in databases:
            result_adata, stats = service.perform_pathway_enrichment(
                mock_adata_with_groups,
                protein_list=['protein_0', 'protein_1'],
                database=db
            )

            assert result_adata is not None
            assert stats['database'] == db

    def test_perform_pathway_enrichment_custom_background(self, service, mock_adata_with_groups):
        """Test pathway enrichment with custom background proteins."""
        protein_list = ['protein_0', 'protein_1']
        background_proteins = mock_adata_with_groups.var_names[:50].tolist()

        result_adata, stats = service.perform_pathway_enrichment(
            mock_adata_with_groups,
            protein_list=protein_list,
            background_proteins=background_proteins
        )

        assert result_adata is not None
        assert stats['n_background_proteins'] == len(background_proteins)

    def test_perform_pathway_enrichment_p_threshold(self, service, mock_adata_with_groups):
        """Test pathway enrichment with different p-value thresholds."""
        result_adata, stats = service.perform_pathway_enrichment(
            mock_adata_with_groups,
            protein_list=['protein_0', 'protein_1'],
            p_value_threshold=0.01
        )

        assert result_adata is not None
        assert stats['p_value_threshold'] == 0.01

    def test_pathway_enrichment_fallback_protein_selection(self, service, mock_adata_with_groups):
        """Test pathway enrichment fallback protein selection when no significant proteins."""
        # Remove significant markers
        if 'is_significant' in mock_adata_with_groups.var.columns:
            del mock_adata_with_groups.var['is_significant']

        result_adata, stats = service.perform_pathway_enrichment(
            mock_adata_with_groups
        )

        assert result_adata is not None
        assert stats['n_query_proteins'] > 0  # Should use fallback selection

    def test_pathway_enrichment_statistics(self, service, mock_adata_with_groups):
        """Test pathway enrichment statistics calculation."""
        result_adata, stats = service.perform_pathway_enrichment(
            mock_adata_with_groups,
            protein_list=['protein_0', 'protein_1']
        )

        assert 'n_significant_pathways' in stats
        assert 'enrichment_rate' in stats
        assert 'samples_processed' in stats
        assert 'proteins_processed' in stats
        assert stats['analysis_type'] == 'pathway_enrichment'


# ===============================================================================
# Helper Methods Tests
# ===============================================================================

class TestHelperMethods:
    """Test suite for helper methods."""

    def test_handle_missing_values_for_testing_skip(self, service):
        """Test missing value handling - skip method."""
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
        result = service._handle_missing_values_for_testing(X, 'skip')

        assert np.array_equal(result, X, equal_nan=True)

    def test_handle_missing_values_for_testing_impute_mean(self, service):
        """Test missing value handling - mean imputation."""
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
        result = service._handle_missing_values_for_testing(X, 'impute_mean')

        assert result is not None
        assert not np.any(np.isnan(result))

    def test_handle_missing_values_for_testing_impute_median(self, service):
        """Test missing value handling - median imputation."""
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
        result = service._handle_missing_values_for_testing(X, 'impute_median')

        assert result is not None
        assert not np.any(np.isnan(result))

    def test_perform_pca_helper(self, service):
        """Test PCA helper method."""
        X = np.random.randn(50, 20)
        result = service._perform_pca(X, n_components=10)

        assert 'embeddings' in result
        assert 'X_pca' in result['embeddings']
        assert result['embeddings']['X_pca'].shape[1] == 10
        assert 'explained_variance' in result
        assert 'total_variance_explained' in result

    def test_perform_tsne_helper(self, service):
        """Test t-SNE helper method."""
        X = np.random.randn(30, 10)
        result = service._perform_tsne(X, n_components=2, perplexity=5.0, random_state=42)

        assert 'embeddings' in result
        assert 'X_tsne' in result['embeddings']
        assert result['embeddings']['X_tsne'].shape[1] == 2

    def test_perform_kmeans_clustering_helper(self, service):
        """Test K-means clustering helper method."""
        X = np.random.randn(30, 10)
        result = service._perform_kmeans_clustering(X, n_clusters=3)

        assert 'labels' in result
        assert 'n_clusters' in result
        assert len(np.unique(result['labels'])) == 3
        assert 'quality_metric' in result

    def test_perform_hierarchical_clustering_helper(self, service):
        """Test hierarchical clustering helper method."""
        X = np.random.randn(20, 5)
        result = service._perform_hierarchical_clustering(X, n_clusters=3)

        assert 'labels' in result
        assert 'n_clusters' in result
        assert len(np.unique(result['labels'])) == 3
        assert 'metadata' in result

    def test_perform_gaussian_mixture_clustering_helper(self, service):
        """Test Gaussian mixture clustering helper method."""
        X = np.random.randn(30, 5)
        result = service._perform_gaussian_mixture_clustering(X, n_clusters=2)

        assert 'labels' in result
        assert 'n_clusters' in result
        assert len(np.unique(result['labels'])) == 2
        assert 'quality_metric' in result


# ===============================================================================
# Error Handling and Edge Cases Tests
# ===============================================================================

class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""

    def test_error_handling_empty_data(self, service):
        """Test error handling with empty data."""
        adata = ad.AnnData(X=np.array([]).reshape(0, 0))

        with pytest.raises(ProteomicsAnalysisError):
            service.perform_statistical_testing(adata, group_column='condition')

    def test_error_handling_insufficient_data_clustering(self, service):
        """Test error handling with insufficient data for clustering."""
        # Create very small dataset
        X = np.random.randn(2, 5)
        adata = ad.AnnData(X=X)

        with pytest.raises(ProteomicsAnalysisError):
            service.perform_clustering_analysis(adata, n_clusters=5)  # More clusters than samples

    def test_error_handling_invalid_parameters(self, service, mock_adata_with_groups):
        """Test error handling with invalid parameters."""
        with pytest.raises(ProteomicsAnalysisError):
            service.perform_dimensionality_reduction(
                mock_adata_with_groups,
                method='pca',
                n_components=-1  # Invalid number of components
            )

    def test_single_group_statistical_testing(self, service):
        """Test statistical testing with single group (should handle gracefully)."""
        X = np.random.randn(10, 5)
        adata = ad.AnnData(X=X)
        adata.obs['condition'] = ['control'] * 10  # Only one group

        result_adata, stats = service.perform_statistical_testing(
            adata,
            group_column='condition'
        )

        assert result_adata is not None
        assert stats['n_tests_performed'] == 0  # No tests possible with one group

    def test_all_missing_protein_statistical_testing(self, service, mock_adata_with_groups):
        """Test statistical testing with proteins that have all missing values."""
        # Make first protein all missing
        mock_adata_with_groups.X[:, 0] = np.nan

        result_adata, stats = service.perform_statistical_testing(
            mock_adata_with_groups,
            group_column='condition',
            handle_missing='skip'
        )

        assert result_adata is not None
        # Should still work with other proteins

    def test_very_small_perplexity_tsne(self, service, mock_adata_with_groups):
        """Test t-SNE with very small perplexity (edge case)."""
        result_adata, stats = service.perform_dimensionality_reduction(
            mock_adata_with_groups,
            method='tsne',
            perplexity=1.0  # Very small perplexity
        )

        assert result_adata is not None
        assert 'X_tsne' in result_adata.obsm


# ===============================================================================
# Integration Tests
# ===============================================================================

class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    def test_complete_analysis_workflow(self, service, mock_adata_with_groups):
        """Test complete analysis workflow: stats -> reduction -> clustering -> enrichment."""

        # Step 1: Statistical testing
        adata_stats, _ = service.perform_statistical_testing(
            mock_adata_with_groups,
            group_column='condition'
        )

        # Step 2: Dimensionality reduction
        adata_reduced, _ = service.perform_dimensionality_reduction(
            adata_stats,
            method='pca'
        )

        # Step 3: Clustering
        adata_clustered, _ = service.perform_clustering_analysis(
            adata_reduced,
            clustering_method='kmeans'
        )

        # Step 4: Pathway enrichment
        adata_enriched, _ = service.perform_pathway_enrichment(
            adata_clustered
        )

        # Verify final result has all analysis components
        assert 'statistical_tests' in adata_enriched.uns
        assert 'X_pca' in adata_enriched.obsm
        assert 'cluster' in adata_enriched.obs.columns
        assert 'pathway_enrichment' in adata_enriched.uns

    def test_multiple_statistical_methods_consistency(self, service, mock_adata_with_groups):
        """Test consistency across different statistical methods."""
        methods = ['t_test', 'mann_whitney']
        results = {}

        for method in methods:
            result_adata, stats = service.perform_statistical_testing(
                mock_adata_with_groups,
                group_column='condition',
                test_method=method,
                comparison_type='all_pairs'
            )
            results[method] = stats

        # Both methods should process same number of samples/proteins
        assert results['t_test']['samples_processed'] == results['mann_whitney']['samples_processed']
        assert results['t_test']['proteins_processed'] == results['mann_whitney']['proteins_processed']

    def test_multiple_reduction_methods_consistency(self, service, mock_adata_with_groups):
        """Test consistency across different dimensionality reduction methods."""
        methods = ['pca', 'tsne', 'umap_like']
        results = {}

        for method in methods:
            result_adata, stats = service.perform_dimensionality_reduction(
                mock_adata_with_groups,
                method=method,
                n_components=2
            )
            results[method] = stats

        # All methods should process same number of samples/proteins
        for method in methods:
            assert results[method]['samples_processed'] == mock_adata_with_groups.n_obs
            assert results[method]['proteins_processed'] == mock_adata_with_groups.n_vars


# ===============================================================================
# Performance and Memory Tests
# ===============================================================================

class TestPerformanceAndMemory:
    """Test suite for performance and memory considerations."""

    @pytest.mark.slow
    def test_large_dataset_statistical_testing(self, service):
        """Test statistical testing with large dataset."""
        # Create larger dataset
        n_samples, n_proteins = 200, 1000
        X = np.random.randn(n_samples, n_proteins)
        adata = ad.AnnData(X=X)
        adata.obs['condition'] = ['control'] * 100 + ['treatment'] * 100

        result_adata, stats = service.perform_statistical_testing(
            adata,
            group_column='condition'
        )

        assert result_adata is not None
        assert stats['samples_processed'] == n_samples
        assert stats['proteins_processed'] == n_proteins

    @pytest.mark.slow
    def test_memory_efficient_dimensionality_reduction(self, service):
        """Test memory efficiency in dimensionality reduction."""
        # Create moderately large dataset
        n_samples, n_proteins = 100, 500
        X = np.random.randn(n_samples, n_proteins)
        adata = ad.AnnData(X=X)

        result_adata, stats = service.perform_dimensionality_reduction(
            adata,
            method='pca',
            n_components=50
        )

        assert result_adata is not None
        # Should complete without memory errors