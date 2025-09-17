"""
Comprehensive unit tests for proteomics visualization service.

This module provides thorough testing of the proteomics visualization service including
missing value heatmaps, intensity distributions, volcano plots, protein networks,
and QC dashboards for proteomics data visualization.

Test coverage target: 95%+ with meaningful tests for proteomics visualization operations.
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

from lobster.tools.proteomics_visualization_service import ProteomicsVisualizationService, ProteomicsVisualizationError

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
    """Create ProteomicsVisualizationService instance."""
    return ProteomicsVisualizationService()


@pytest.fixture
def mock_adata_with_missing():
    """Create mock AnnData with missing values for visualization."""
    n_samples, n_proteins = 48, 80
    X = np.random.lognormal(mean=9, sigma=1, size=(n_samples, n_proteins))

    # Add structured missing values
    missing_rate_per_protein = np.random.beta(2, 5, n_proteins) * 0.6  # 0-60% missing per protein
    for protein_idx in range(n_proteins):
        n_missing = int(missing_rate_per_protein[protein_idx] * n_samples)
        missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        X[missing_indices, protein_idx] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add metadata
    adata.obs['condition'] = ['control'] * 16 + ['treatment1'] * 16 + ['treatment2'] * 16
    adata.obs['batch'] = ['batch1'] * 16 + ['batch2'] * 16 + ['batch3'] * 16
    adata.var['protein_names'] = [f"PROT_{i}" for i in range(n_proteins)]

    # Add missing value QC metrics
    adata.obs['missing_value_percentage'] = (np.isnan(X).sum(axis=1) / n_proteins) * 100
    adata.var['missing_value_percentage'] = (np.isnan(X).sum(axis=0) / n_samples) * 100

    return adata


@pytest.fixture
def mock_adata_with_de_results():
    """Create mock AnnData with differential expression results."""
    n_samples, n_proteins = 60, 100
    X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    # Add differential expression pattern
    de_proteins = np.random.choice(n_proteins, 30, replace=False)
    for protein_idx in de_proteins[:15]:  # Upregulated
        X[30:, protein_idx] *= 2.5
    for protein_idx in de_proteins[15:]:  # Downregulated
        X[30:, protein_idx] *= 0.4

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add differential expression metadata
    adata.obs['condition'] = ['control'] * 30 + ['treatment'] * 30

    # Mock DE results
    de_results = []
    for i, protein_idx in enumerate(de_proteins):
        if i < 15:  # Upregulated
            fold_change = 2.5
            p_value = 0.001
        else:  # Downregulated
            fold_change = 0.4
            p_value = 0.001

        de_results.append({
            'protein': f'protein_{protein_idx}',
            'fold_change': fold_change,
            'log2_fold_change': np.log2(fold_change),
            'p_value': p_value,
            'p_adjusted': p_value * 1.5,
            'significant': True
        })

    # Add non-significant results
    for protein_idx in range(n_proteins):
        if protein_idx not in de_proteins:
            de_results.append({
                'protein': f'protein_{protein_idx}',
                'fold_change': 1.1,
                'log2_fold_change': np.log2(1.1),
                'p_value': 0.5,
                'p_adjusted': 0.7,
                'significant': False
            })

    adata.uns['differential_expression'] = {'results': de_results}

    return adata


@pytest.fixture
def mock_adata_with_cv_data():
    """Create mock AnnData with CV analysis data."""
    n_samples, n_proteins = 40, 60
    X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add CV metrics
    cv_values = []
    for protein_idx in range(n_proteins):
        protein_data = X[:, protein_idx]
        cv = np.std(protein_data) / np.mean(protein_data)
        cv_values.append(cv)

    adata.var['cv_mean'] = cv_values
    adata.var['high_cv_protein'] = np.array(cv_values) > 0.3

    # Add replicate information
    adata.obs['replicate_group'] = [f"group_{i//4}" for i in range(n_samples)]

    return adata


@pytest.fixture
def mock_adata_with_correlations():
    """Create mock AnnData with correlation data."""
    n_samples, n_proteins = 50, 40
    X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    # Create some correlated proteins
    X[:, 1] = X[:, 0] * 1.2 + np.random.normal(0, X[:, 0] * 0.1, n_samples)  # Positive correlation
    X[:, 2] = np.max(X[:, 0]) - X[:, 0] + np.random.normal(0, X[:, 0] * 0.1, n_samples)  # Negative correlation

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Mock correlation results
    correlation_results = [
        {'protein1': 'protein_0', 'protein2': 'protein_1', 'correlation': 0.85, 'p_value': 0.001},
        {'protein1': 'protein_0', 'protein2': 'protein_2', 'correlation': -0.75, 'p_value': 0.001},
        {'protein1': 'protein_1', 'protein2': 'protein_3', 'correlation': 0.65, 'p_value': 0.01},
    ]

    adata.uns['correlation_analysis'] = {'results': correlation_results}

    return adata


@pytest.fixture
def mock_adata_with_pathway_results():
    """Create mock AnnData with pathway enrichment results."""
    n_samples, n_proteins = 30, 50
    X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Mock pathway enrichment results
    pathway_results = [
        {
            'pathway_name': 'Cell_Cycle',
            'p_value': 0.001,
            'enrichment_ratio': 3.2,
            'overlap_count': 8,
            'pathway_size': 25
        },
        {
            'pathway_name': 'Apoptosis',
            'p_value': 0.01,
            'enrichment_ratio': 2.1,
            'overlap_count': 5,
            'pathway_size': 20
        },
        {
            'pathway_name': 'Metabolism',
            'p_value': 0.03,
            'enrichment_ratio': 1.8,
            'overlap_count': 12,
            'pathway_size': 60
        }
    ]

    adata.uns['pathway_enrichment'] = {'results': pathway_results}

    return adata


# ===============================================================================
# Service Initialization Tests
# ===============================================================================

class TestProteomicsVisualizationServiceInitialization:
    """Test suite for ProteomicsVisualizationService initialization."""

    def test_init_default_parameters(self):
        """Test service initialization with default parameters."""
        service = ProteomicsVisualizationService()

        assert service is not None


# ===============================================================================
# Missing Value Heatmap Tests
# ===============================================================================

class TestMissingValueHeatmap:
    """Test suite for missing value heatmap functionality."""

    def test_create_missing_value_heatmap_basic(self, service, mock_adata_with_missing):
        """Test basic missing value heatmap creation."""
        fig, stats = service.create_missing_value_heatmap(mock_adata_with_missing)

        assert fig is not None
        assert isinstance(stats, dict)
        assert 'plot_type' in stats
        assert stats['plot_type'] == 'missing_value_heatmap'
        assert 'total_missing_percentage' in stats
        assert 'samples_plotted' in stats
        assert 'proteins_plotted' in stats

    def test_create_missing_value_heatmap_sample_subset(self, service, mock_adata_with_missing):
        """Test missing value heatmap with sample subset."""
        sample_subset = mock_adata_with_missing.obs_names[:20]

        fig, stats = service.create_missing_value_heatmap(
            mock_adata_with_missing,
            sample_subset=sample_subset
        )

        assert fig is not None
        assert stats['samples_plotted'] == len(sample_subset)

    def test_create_missing_value_heatmap_protein_subset(self, service, mock_adata_with_missing):
        """Test missing value heatmap with protein subset."""
        protein_subset = mock_adata_with_missing.var_names[:30]

        fig, stats = service.create_missing_value_heatmap(
            mock_adata_with_missing,
            protein_subset=protein_subset
        )

        assert fig is not None
        assert stats['proteins_plotted'] == len(protein_subset)

    def test_create_missing_value_heatmap_custom_colorscale(self, service, mock_adata_with_missing):
        """Test missing value heatmap with custom colorscale."""
        fig, stats = service.create_missing_value_heatmap(
            mock_adata_with_missing,
            colorscale='Viridis'
        )

        assert fig is not None

    def test_create_missing_value_heatmap_no_missing(self, service):
        """Test missing value heatmap with no missing values."""
        X = np.random.lognormal(mean=8, sigma=1, size=(20, 30))
        adata = ad.AnnData(X=X)

        fig, stats = service.create_missing_value_heatmap(adata)

        assert fig is not None
        assert stats['total_missing_percentage'] == 0.0


# ===============================================================================
# Intensity Distribution Plot Tests
# ===============================================================================

class TestIntensityDistributionPlot:
    """Test suite for intensity distribution plot functionality."""

    def test_create_intensity_distribution_plot_basic(self, service, mock_adata_with_missing):
        """Test basic intensity distribution plot creation."""
        fig, stats = service.create_intensity_distribution_plot(mock_adata_with_missing)

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats['plot_type'] == 'intensity_distribution'
        assert 'distribution_stats' in stats

    def test_create_intensity_distribution_plot_by_group(self, service, mock_adata_with_missing):
        """Test intensity distribution plot grouped by condition."""
        fig, stats = service.create_intensity_distribution_plot(
            mock_adata_with_missing,
            group_by='condition'
        )

        assert fig is not None
        assert 'group_stats' in stats

    def test_create_intensity_distribution_plot_log_scale(self, service, mock_adata_with_missing):
        """Test intensity distribution plot with log scale."""
        fig, stats = service.create_intensity_distribution_plot(
            mock_adata_with_missing,
            log_scale=True
        )

        assert fig is not None

    def test_create_intensity_distribution_plot_histogram(self, service, mock_adata_with_missing):
        """Test intensity distribution plot as histogram."""
        fig, stats = service.create_intensity_distribution_plot(
            mock_adata_with_missing,
            plot_type='histogram'
        )

        assert fig is not None

    def test_create_intensity_distribution_plot_density(self, service, mock_adata_with_missing):
        """Test intensity distribution plot as density plot."""
        fig, stats = service.create_intensity_distribution_plot(
            mock_adata_with_missing,
            plot_type='density'
        )

        assert fig is not None

    def test_create_intensity_distribution_plot_violin(self, service, mock_adata_with_missing):
        """Test intensity distribution plot as violin plot."""
        fig, stats = service.create_intensity_distribution_plot(
            mock_adata_with_missing,
            plot_type='violin',
            group_by='condition'
        )

        assert fig is not None


# ===============================================================================
# CV Analysis Plot Tests
# ===============================================================================

class TestCVAnalysisPlot:
    """Test suite for CV analysis plot functionality."""

    def test_create_cv_analysis_plot_basic(self, service, mock_adata_with_cv_data):
        """Test basic CV analysis plot creation."""
        fig, stats = service.create_cv_analysis_plot(mock_adata_with_cv_data)

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats['plot_type'] == 'cv_analysis'
        assert 'cv_statistics' in stats

    def test_create_cv_analysis_plot_by_replicate(self, service, mock_adata_with_cv_data):
        """Test CV analysis plot by replicate groups."""
        fig, stats = service.create_cv_analysis_plot(
            mock_adata_with_cv_data,
            replicate_column='replicate_group'
        )

        assert fig is not None
        assert 'replicate_cv_stats' in stats

    def test_create_cv_analysis_plot_custom_threshold(self, service, mock_adata_with_cv_data):
        """Test CV analysis plot with custom CV threshold."""
        fig, stats = service.create_cv_analysis_plot(
            mock_adata_with_cv_data,
            cv_threshold=0.25
        )

        assert fig is not None

    def test_create_cv_analysis_plot_scatter(self, service, mock_adata_with_cv_data):
        """Test CV analysis plot as scatter plot."""
        fig, stats = service.create_cv_analysis_plot(
            mock_adata_with_cv_data,
            plot_type='scatter'
        )

        assert fig is not None

    def test_create_cv_analysis_plot_histogram(self, service, mock_adata_with_cv_data):
        """Test CV analysis plot as histogram."""
        fig, stats = service.create_cv_analysis_plot(
            mock_adata_with_cv_data,
            plot_type='histogram'
        )

        assert fig is not None


# ===============================================================================
# Volcano Plot Tests
# ===============================================================================

class TestVolcanoPlot:
    """Test suite for volcano plot functionality."""

    def test_create_volcano_plot_basic(self, service, mock_adata_with_de_results):
        """Test basic volcano plot creation."""
        comparison = 'control_vs_treatment'

        fig, stats = service.create_volcano_plot(
            mock_adata_with_de_results,
            comparison=comparison
        )

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats['plot_type'] == 'volcano_plot'
        assert 'n_significant_up' in stats
        assert 'n_significant_down' in stats

    def test_create_volcano_plot_custom_thresholds(self, service, mock_adata_with_de_results):
        """Test volcano plot with custom significance thresholds."""
        fig, stats = service.create_volcano_plot(
            mock_adata_with_de_results,
            comparison='control_vs_treatment',
            p_threshold=0.01,
            fc_threshold=2.0
        )

        assert fig is not None

    def test_create_volcano_plot_labeled_proteins(self, service, mock_adata_with_de_results):
        """Test volcano plot with labeled significant proteins."""
        fig, stats = service.create_volcano_plot(
            mock_adata_with_de_results,
            comparison='control_vs_treatment',
            label_significant=True,
            max_labels=10
        )

        assert fig is not None

    def test_create_volcano_plot_custom_colors(self, service, mock_adata_with_de_results):
        """Test volcano plot with custom colors."""
        custom_colors = {
            'upregulated': 'red',
            'downregulated': 'blue',
            'non_significant': 'gray'
        }

        fig, stats = service.create_volcano_plot(
            mock_adata_with_de_results,
            comparison='control_vs_treatment',
            colors=custom_colors
        )

        assert fig is not None

    def test_create_volcano_plot_no_de_results(self, service, mock_adata_with_missing):
        """Test volcano plot with no DE results."""
        with pytest.raises(ProteomicsVisualizationError) as exc_info:
            service.create_volcano_plot(
                mock_adata_with_missing,
                comparison='control_vs_treatment'
            )

        assert "No differential expression results found" in str(exc_info.value)


# ===============================================================================
# Protein Correlation Network Tests
# ===============================================================================

class TestProteinCorrelationNetwork:
    """Test suite for protein correlation network functionality."""

    def test_create_protein_correlation_network_basic(self, service, mock_adata_with_correlations):
        """Test basic protein correlation network creation."""
        fig, stats = service.create_protein_correlation_network(mock_adata_with_correlations)

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats['plot_type'] == 'correlation_network'
        assert 'n_nodes' in stats
        assert 'n_edges' in stats

    def test_create_protein_correlation_network_custom_threshold(self, service, mock_adata_with_correlations):
        """Test protein correlation network with custom correlation threshold."""
        fig, stats = service.create_protein_correlation_network(
            mock_adata_with_correlations,
            correlation_threshold=0.8
        )

        assert fig is not None

    def test_create_protein_correlation_network_protein_subset(self, service, mock_adata_with_correlations):
        """Test protein correlation network with protein subset."""
        protein_subset = ['protein_0', 'protein_1', 'protein_2', 'protein_3']

        fig, stats = service.create_protein_correlation_network(
            mock_adata_with_correlations,
            protein_subset=protein_subset
        )

        assert fig is not None
        assert stats['n_nodes'] <= len(protein_subset)

    def test_create_protein_correlation_network_layout(self, service, mock_adata_with_correlations):
        """Test protein correlation network with different layouts."""
        layouts = ['spring', 'circular', 'kamada_kawai']

        for layout in layouts:
            fig, stats = service.create_protein_correlation_network(
                mock_adata_with_correlations,
                layout=layout
            )

            assert fig is not None

    def test_create_protein_correlation_network_no_correlations(self, service, mock_adata_with_missing):
        """Test protein correlation network with no correlation results."""
        with pytest.raises(ProteomicsVisualizationError) as exc_info:
            service.create_protein_correlation_network(mock_adata_with_missing)

        assert "No correlation analysis results found" in str(exc_info.value)


# ===============================================================================
# Pathway Enrichment Plot Tests
# ===============================================================================

class TestPathwayEnrichmentPlot:
    """Test suite for pathway enrichment plot functionality."""

    def test_create_pathway_enrichment_plot_basic(self, service, mock_adata_with_pathway_results):
        """Test basic pathway enrichment plot creation."""
        fig, stats = service.create_pathway_enrichment_plot(mock_adata_with_pathway_results)

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats['plot_type'] == 'pathway_enrichment'
        assert 'n_pathways_plotted' in stats

    def test_create_pathway_enrichment_plot_custom_threshold(self, service, mock_adata_with_pathway_results):
        """Test pathway enrichment plot with custom p-value threshold."""
        fig, stats = service.create_pathway_enrichment_plot(
            mock_adata_with_pathway_results,
            p_threshold=0.01
        )

        assert fig is not None

    def test_create_pathway_enrichment_plot_top_pathways(self, service, mock_adata_with_pathway_results):
        """Test pathway enrichment plot showing top pathways only."""
        fig, stats = service.create_pathway_enrichment_plot(
            mock_adata_with_pathway_results,
            max_pathways=5
        )

        assert fig is not None
        assert stats['n_pathways_plotted'] <= 5

    def test_create_pathway_enrichment_plot_horizontal(self, service, mock_adata_with_pathway_results):
        """Test pathway enrichment plot with horizontal layout."""
        fig, stats = service.create_pathway_enrichment_plot(
            mock_adata_with_pathway_results,
            orientation='horizontal'
        )

        assert fig is not None

    def test_create_pathway_enrichment_plot_bubble(self, service, mock_adata_with_pathway_results):
        """Test pathway enrichment plot as bubble plot."""
        fig, stats = service.create_pathway_enrichment_plot(
            mock_adata_with_pathway_results,
            plot_type='bubble'
        )

        assert fig is not None

    def test_create_pathway_enrichment_plot_no_results(self, service, mock_adata_with_missing):
        """Test pathway enrichment plot with no pathway results."""
        with pytest.raises(ProteomicsVisualizationError) as exc_info:
            service.create_pathway_enrichment_plot(mock_adata_with_missing)

        assert "No pathway enrichment results found" in str(exc_info.value)


# ===============================================================================
# QC Dashboard Tests
# ===============================================================================

class TestProteomicsQCDashboard:
    """Test suite for proteomics QC dashboard functionality."""

    def test_create_proteomics_qc_dashboard_basic(self, service, mock_adata_with_missing):
        """Test basic proteomics QC dashboard creation."""
        fig, stats = service.create_proteomics_qc_dashboard(mock_adata_with_missing)

        assert fig is not None
        assert isinstance(stats, dict)
        assert stats['plot_type'] == 'qc_dashboard'
        assert 'dashboard_components' in stats

    def test_create_proteomics_qc_dashboard_custom_components(self, service, mock_adata_with_missing):
        """Test QC dashboard with custom components."""
        components = ['missing_values', 'intensity_distribution', 'cv_analysis']

        fig, stats = service.create_proteomics_qc_dashboard(
            mock_adata_with_missing,
            components=components
        )

        assert fig is not None
        assert len(stats['dashboard_components']) == len(components)

    def test_create_proteomics_qc_dashboard_with_batch(self, service, mock_adata_with_missing):
        """Test QC dashboard grouped by batch."""
        fig, stats = service.create_proteomics_qc_dashboard(
            mock_adata_with_missing,
            group_by='batch'
        )

        assert fig is not None

    def test_create_proteomics_qc_dashboard_minimal(self, service):
        """Test QC dashboard with minimal data."""
        X = np.random.lognormal(mean=8, sigma=1, size=(10, 20))
        adata = ad.AnnData(X=X)

        fig, stats = service.create_proteomics_qc_dashboard(adata)

        assert fig is not None


# ===============================================================================
# Plot Saving Tests
# ===============================================================================

class TestPlotSaving:
    """Test suite for plot saving functionality."""

    @patch('pathlib.Path.mkdir')
    @patch('plotly.graph_objects.Figure.write_html')
    @patch('plotly.graph_objects.Figure.write_image')
    def test_save_plots_html(self, mock_write_image, mock_write_html, mock_mkdir, service, mock_adata_with_missing):
        """Test saving plots as HTML."""
        # Create a plot
        fig, _ = service.create_missing_value_heatmap(mock_adata_with_missing)

        plots_dict = {'missing_value_heatmap': fig}

        saved_files = service.save_plots(
            plots_dict,
            output_dir='test_output',
            format='html'
        )

        assert len(saved_files) == 1
        assert saved_files[0].suffix == '.html'
        mock_write_html.assert_called_once()

    @patch('pathlib.Path.mkdir')
    @patch('plotly.graph_objects.Figure.write_html')
    @patch('plotly.graph_objects.Figure.write_image')
    def test_save_plots_png(self, mock_write_image, mock_write_html, mock_mkdir, service, mock_adata_with_missing):
        """Test saving plots as PNG."""
        fig, _ = service.create_missing_value_heatmap(mock_adata_with_missing)

        plots_dict = {'missing_value_heatmap': fig}

        saved_files = service.save_plots(
            plots_dict,
            output_dir='test_output',
            format='png'
        )

        assert len(saved_files) == 1
        assert saved_files[0].suffix == '.png'
        mock_write_image.assert_called_once()

    @patch('pathlib.Path.mkdir')
    @patch('plotly.graph_objects.Figure.write_html')
    @patch('plotly.graph_objects.Figure.write_image')
    def test_save_plots_both_formats(self, mock_write_image, mock_write_html, mock_mkdir, service, mock_adata_with_missing):
        """Test saving plots in both HTML and PNG formats."""
        fig, _ = service.create_missing_value_heatmap(mock_adata_with_missing)

        plots_dict = {'missing_value_heatmap': fig}

        saved_files = service.save_plots(
            plots_dict,
            output_dir='test_output',
            format='both'
        )

        assert len(saved_files) == 2
        formats = {f.suffix for f in saved_files}
        assert formats == {'.html', '.png'}

    def test_save_plots_invalid_format(self, service, mock_adata_with_missing):
        """Test saving plots with invalid format."""
        fig, _ = service.create_missing_value_heatmap(mock_adata_with_missing)

        plots_dict = {'missing_value_heatmap': fig}

        with pytest.raises(ProteomicsVisualizationError) as exc_info:
            service.save_plots(
                plots_dict,
                output_dir='test_output',
                format='invalid_format'
            )

        assert "Unsupported format" in str(exc_info.value)


# ===============================================================================
# Error Handling and Edge Cases Tests
# ===============================================================================

class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""

    def test_error_handling_empty_data(self, service):
        """Test error handling with empty data."""
        adata = ad.AnnData(X=np.array([]).reshape(0, 0))

        with pytest.raises(ProteomicsVisualizationError):
            service.create_missing_value_heatmap(adata)

    def test_single_sample_visualization(self, service):
        """Test visualization with single sample."""
        X = np.random.lognormal(mean=8, sigma=1, size=(1, 20))
        adata = ad.AnnData(X=X)

        # Should handle single sample gracefully
        fig, stats = service.create_intensity_distribution_plot(adata)
        assert fig is not None

    def test_single_protein_visualization(self, service):
        """Test visualization with single protein."""
        X = np.random.lognormal(mean=8, sigma=1, size=(20, 1))
        adata = ad.AnnData(X=X)

        fig, stats = service.create_missing_value_heatmap(adata)
        assert fig is not None

    def test_all_missing_data_visualization(self, service):
        """Test visualization with all missing data."""
        X = np.full((10, 5), np.nan)
        adata = ad.AnnData(X=X)

        fig, stats = service.create_missing_value_heatmap(adata)
        assert fig is not None
        assert stats['total_missing_percentage'] == 100.0

    def test_no_variation_data_visualization(self, service):
        """Test visualization with no variation data."""
        X = np.ones((10, 20)) * 1000  # All same value
        adata = ad.AnnData(X=X)

        # Should handle no variation gracefully
        fig, stats = service.create_intensity_distribution_plot(adata)
        assert fig is not None

    def test_invalid_subset_parameters(self, service, mock_adata_with_missing):
        """Test visualization with invalid subset parameters."""
        with pytest.raises(ProteomicsVisualizationError):
            service.create_missing_value_heatmap(
                mock_adata_with_missing,
                sample_subset=['nonexistent_sample']
            )

        with pytest.raises(ProteomicsVisualizationError):
            service.create_missing_value_heatmap(
                mock_adata_with_missing,
                protein_subset=['nonexistent_protein']
            )


# ===============================================================================
# Integration Tests
# ===============================================================================

class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    def test_complete_visualization_workflow(self, service, mock_adata_with_missing):
        """Test complete visualization workflow with multiple plots."""
        plots = {}

        # Create multiple plots
        plots['missing_values'], _ = service.create_missing_value_heatmap(mock_adata_with_missing)
        plots['intensity_dist'], _ = service.create_intensity_distribution_plot(mock_adata_with_missing)
        plots['qc_dashboard'], _ = service.create_proteomics_qc_dashboard(mock_adata_with_missing)

        # All plots should be created successfully
        assert all(plot is not None for plot in plots.values())

    def test_visualization_with_all_analysis_results(self, service):
        """Test visualization with comprehensive analysis results."""
        # Create comprehensive mock data
        n_samples, n_proteins = 40, 60
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

        adata = ad.AnnData(X=X)
        adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
        adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

        # Add all types of results
        adata.obs['condition'] = ['control'] * 20 + ['treatment'] * 20
        adata.var['cv_mean'] = np.random.uniform(0.1, 0.8, n_proteins)

        # Mock all analysis results
        adata.uns['differential_expression'] = {'results': []}
        adata.uns['correlation_analysis'] = {'results': []}
        adata.uns['pathway_enrichment'] = {'results': []}

        # Should be able to create QC dashboard with all results
        fig, stats = service.create_proteomics_qc_dashboard(adata)
        assert fig is not None

    def test_consistent_plot_styling(self, service, mock_adata_with_missing):
        """Test consistent styling across different plot types."""
        plots_and_stats = []

        # Create multiple plot types
        plots_and_stats.append(service.create_missing_value_heatmap(mock_adata_with_missing))
        plots_and_stats.append(service.create_intensity_distribution_plot(mock_adata_with_missing))

        # All plots should have consistent basic properties
        for fig, stats in plots_and_stats:
            assert fig is not None
            assert 'plot_type' in stats


# ===============================================================================
# Performance and Memory Tests
# ===============================================================================

class TestPerformanceAndMemory:
    """Test suite for performance and memory considerations."""

    @pytest.mark.slow
    def test_large_dataset_visualization(self, service):
        """Test visualization with large dataset."""
        # Create larger dataset
        n_samples, n_proteins = 200, 500
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))
        # Add some missing values
        missing_mask = np.random.rand(n_samples, n_proteins) < 0.1
        X[missing_mask] = np.nan

        adata = ad.AnnData(X=X)

        fig, stats = service.create_missing_value_heatmap(adata)

        assert fig is not None
        assert stats['samples_plotted'] == n_samples
        assert stats['proteins_plotted'] == n_proteins

    @pytest.mark.slow
    def test_memory_efficient_qc_dashboard(self, service):
        """Test memory efficiency in QC dashboard creation."""
        # Create moderately large dataset
        n_samples, n_proteins = 100, 300
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))
        adata = ad.AnnData(X=X)

        fig, stats = service.create_proteomics_qc_dashboard(adata)

        assert fig is not None
        # Should complete without memory errors

    def test_efficient_network_visualization(self, service):
        """Test efficient network visualization with many proteins."""
        n_samples, n_proteins = 50, 100
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))
        adata = ad.AnnData(X=X)

        # Mock minimal correlation results to avoid memory issues
        correlation_results = [
            {'protein1': 'protein_0', 'protein2': 'protein_1', 'correlation': 0.85, 'p_value': 0.001},
            {'protein1': 'protein_2', 'protein2': 'protein_3', 'correlation': 0.75, 'p_value': 0.001},
        ]
        adata.uns['correlation_analysis'] = {'results': correlation_results}

        fig, stats = service.create_protein_correlation_network(
            adata,
            correlation_threshold=0.7  # Higher threshold to reduce complexity
        )

        assert fig is not None