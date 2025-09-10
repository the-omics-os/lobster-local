"""
Unit tests for ProteomicsVisualizationService.

Tests the proteomics visualization service methods including missing value heatmaps,
intensity distributions, CV analysis, volcano plots, correlation networks, pathway
enrichment plots, and comprehensive QC dashboards.
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from unittest.mock import Mock, patch

from lobster.tools.proteomics_visualization_service import (
    ProteomicsVisualizationService,
    ProteomicsVisualizationError
)
from tests.mock_data.generators import MockProteomicsDataGenerator


class TestProteomicsVisualizationService:
    """Test suite for ProteomicsVisualizationService."""

    @pytest.fixture
    def service(self):
        """Create service instance."""
        return ProteomicsVisualizationService()

    @pytest.fixture
    def mock_proteomics_data(self):
        """Create mock proteomics data."""
        generator = MockProteomicsDataGenerator()
        return generator.create_mass_spectrometry_data(
            n_samples=50,
            n_proteins=200,
            missing_rate=0.3
        )

    def test_service_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert hasattr(service, 'intensity_colors')
        assert hasattr(service, 'missing_colors')
        assert hasattr(service, 'default_width')
        assert hasattr(service, 'default_height')
        assert service.default_pvalue_threshold == 0.05
        assert service.default_fc_threshold == 1.5

    def test_create_missing_value_heatmap(self, service, mock_proteomics_data):
        """Test missing value heatmap creation."""
        fig = service.create_missing_value_heatmap(
            mock_proteomics_data,
            max_proteins=50,
            max_samples=20
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == 'heatmap'
        assert 'Missing Value Pattern' in fig.layout.title.text

    def test_create_missing_value_heatmap_with_clustering(self, service, mock_proteomics_data):
        """Test missing value heatmap with clustering."""
        fig = service.create_missing_value_heatmap(
            mock_proteomics_data,
            cluster_samples=True,
            cluster_proteins=True,
            max_proteins=30,
            max_samples=20
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_intensity_distribution_plot(self, service, mock_proteomics_data):
        """Test intensity distribution plot creation."""
        fig = service.create_intensity_distribution_plot(
            mock_proteomics_data,
            log_transform=True
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Protein Intensity Distribution' in fig.layout.title.text

    def test_create_intensity_distribution_plot_grouped(self, service, mock_proteomics_data):
        """Test intensity distribution plot with grouping."""
        # Add a grouping column
        mock_proteomics_data.obs['condition'] = ['A'] * 25 + ['B'] * 25
        
        fig = service.create_intensity_distribution_plot(
            mock_proteomics_data,
            group_by='condition',
            log_transform=True
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Should have traces for each group

    def test_create_cv_analysis_plot(self, service, mock_proteomics_data):
        """Test CV analysis plot creation."""
        fig = service.create_cv_analysis_plot(
            mock_proteomics_data,
            cv_threshold=30.0
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Protein CV Distribution' in fig.layout.title.text

    def test_create_cv_analysis_plot_grouped(self, service, mock_proteomics_data):
        """Test CV analysis plot with grouping."""
        mock_proteomics_data.obs['batch'] = ['1'] * 25 + ['2'] * 25
        
        fig = service.create_cv_analysis_plot(
            mock_proteomics_data,
            group_by='batch',
            cv_threshold=25.0
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_volcano_plot(self, service, mock_proteomics_data):
        """Test volcano plot creation."""
        # Add mock differential expression results
        n_proteins = mock_proteomics_data.n_vars
        de_results = pd.DataFrame({
            'protein': mock_proteomics_data.var_names,
            'log2_fold_change': np.random.normal(0, 1, n_proteins),
            'p_adjusted': np.random.uniform(0, 1, n_proteins)
        })
        
        fig = service.create_volcano_plot(
            mock_proteomics_data,
            comparison_results=de_results,
            fold_change_col='log2_fold_change',
            pvalue_col='p_adjusted'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Volcano Plot' in fig.layout.title.text

    def test_create_volcano_plot_with_highlights(self, service, mock_proteomics_data):
        """Test volcano plot with highlighted proteins."""
        n_proteins = mock_proteomics_data.n_vars
        de_results = pd.DataFrame({
            'protein': mock_proteomics_data.var_names,
            'log2_fold_change': np.random.normal(0, 1, n_proteins),
            'p_adjusted': np.random.uniform(0, 1, n_proteins)
        })
        
        highlight_proteins = mock_proteomics_data.var_names[:5].tolist()
        
        fig = service.create_volcano_plot(
            mock_proteomics_data,
            comparison_results=de_results,
            highlight_proteins=highlight_proteins
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Main plot + highlighted proteins

    def test_create_protein_correlation_network(self, service, mock_proteomics_data):
        """Test protein correlation network creation."""
        fig = service.create_protein_correlation_network(
            mock_proteomics_data,
            correlation_threshold=0.7,
            max_proteins=50
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Edges and nodes
        assert 'Protein Correlation Network' in fig.layout.title.text

    def test_create_protein_correlation_network_with_coloring(self, service, mock_proteomics_data):
        """Test protein correlation network with node coloring."""
        # Add protein metadata
        mock_proteomics_data.var['protein_class'] = np.random.choice(['A', 'B', 'C'], mock_proteomics_data.n_vars)
        
        fig = service.create_protein_correlation_network(
            mock_proteomics_data,
            color_by='protein_class',
            max_proteins=30
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_create_pathway_enrichment_plot(self, service, mock_proteomics_data):
        """Test pathway enrichment plot creation."""
        # Mock pathway enrichment results
        enrichment_results = [
            {
                'pathway_name': 'Pathway A',
                'p_value': 0.01,
                'enrichment_ratio': 2.5,
                'overlap_count': 10
            },
            {
                'pathway_name': 'Pathway B',
                'p_value': 0.03,
                'enrichment_ratio': 1.8,
                'overlap_count': 7
            },
            {
                'pathway_name': 'Pathway C',
                'p_value': 0.05,
                'enrichment_ratio': 1.5,
                'overlap_count': 5
            }
        ]
        
        fig = service.create_pathway_enrichment_plot(
            mock_proteomics_data,
            enrichment_results=enrichment_results,
            plot_type='bubble'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Enriched Pathways' in fig.layout.title.text

    def test_create_pathway_enrichment_plot_bar(self, service, mock_proteomics_data):
        """Test pathway enrichment bar plot."""
        enrichment_results = [
            {
                'pathway_name': 'Pathway A',
                'p_value': 0.01,
                'enrichment_ratio': 2.5,
                'overlap_count': 10
            }
        ]
        
        fig = service.create_pathway_enrichment_plot(
            mock_proteomics_data,
            enrichment_results=enrichment_results,
            plot_type='bar'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_proteomics_qc_dashboard_ms(self, service, mock_proteomics_data):
        """Test MS proteomics QC dashboard creation."""
        fig = service.create_proteomics_qc_dashboard(
            mock_proteomics_data,
            platform_type='mass_spectrometry'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # Should have multiple subplots
        assert 'Mass Spectrometry' in fig.layout.title.text

    def test_create_proteomics_qc_dashboard_affinity(self, service, mock_proteomics_data):
        """Test affinity proteomics QC dashboard creation."""
        fig = service.create_proteomics_qc_dashboard(
            mock_proteomics_data,
            platform_type='affinity'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert 'Affinity' in fig.layout.title.text

    def test_create_proteomics_qc_dashboard_with_batch(self, service, mock_proteomics_data):
        """Test QC dashboard with batch information."""
        mock_proteomics_data.obs['batch'] = ['batch1'] * 25 + ['batch2'] * 25
        
        fig = service.create_proteomics_qc_dashboard(
            mock_proteomics_data,
            platform_type='mass_spectrometry'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_proteomics_qc_dashboard_with_peptides(self, service, mock_proteomics_data):
        """Test QC dashboard with peptide information."""
        mock_proteomics_data.var['n_peptides'] = np.random.randint(1, 10, mock_proteomics_data.n_vars)
        
        fig = service.create_proteomics_qc_dashboard(
            mock_proteomics_data,
            platform_type='mass_spectrometry'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    @patch('pathlib.Path.mkdir')
    @patch('plotly.io.write_html')
    @patch('plotly.io.write_image')
    def test_save_plots(self, mock_write_image, mock_write_html, mock_mkdir, service):
        """Test plot saving functionality."""
        # Create mock figures
        fig1 = go.Figure()
        fig2 = go.Figure()
        plots = {'plot1': fig1, 'plot2': fig2}
        
        saved_files = service.save_plots(plots, '/tmp/output', format='both')
        
        # Should attempt to save both HTML and PNG for each plot
        assert len(saved_files) == 4  # 2 plots * 2 formats
        assert mock_write_html.call_count == 2
        assert mock_write_image.call_count == 2

    def test_volcano_plot_missing_data_error(self, service, mock_proteomics_data):
        """Test volcano plot error handling when no DE results found."""
        with pytest.raises(ProteomicsVisualizationError):
            service.create_volcano_plot(mock_proteomics_data)

    def test_pathway_enrichment_plot_missing_data_error(self, service, mock_proteomics_data):
        """Test pathway enrichment plot error handling."""
        with pytest.raises(ProteomicsVisualizationError):
            service.create_pathway_enrichment_plot(mock_proteomics_data)

    def test_volcano_plot_column_mapping(self, service, mock_proteomics_data):
        """Test volcano plot with different column names."""
        de_results = pd.DataFrame({
            'protein': mock_proteomics_data.var_names,
            'fold_change': np.random.normal(0, 1, mock_proteomics_data.n_vars),
            'p_value': np.random.uniform(0, 1, mock_proteomics_data.n_vars)
        })
        
        fig = service.create_volcano_plot(
            mock_proteomics_data,
            comparison_results=de_results,
            fold_change_col='fold_change',
            pvalue_col='p_value'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_empty_network_handling(self, service, mock_proteomics_data):
        """Test network plot with high correlation threshold."""
        fig = service.create_protein_correlation_network(
            mock_proteomics_data,
            correlation_threshold=0.99,  # Very high threshold
            max_proteins=50
        )
        
        assert isinstance(fig, go.Figure)
        # Should still create figure even with no edges

    def test_custom_titles(self, service, mock_proteomics_data):
        """Test custom titles in plots."""
        custom_title = "Custom Test Title"
        
        fig = service.create_missing_value_heatmap(
            mock_proteomics_data,
            title=custom_title
        )
        
        assert custom_title in fig.layout.title.text

    def test_different_layout_algorithms(self, service, mock_proteomics_data):
        """Test different network layout algorithms."""
        for algorithm in ['spring', 'circular', 'random']:
            fig = service.create_protein_correlation_network(
                mock_proteomics_data,
                layout_algorithm=algorithm,
                max_proteins=20
            )
            
            assert isinstance(fig, go.Figure)

    def test_large_dataset_subsampling(self, service):
        """Test handling of large datasets with subsampling."""
        generator = MockProteomicsDataGenerator()
        large_data = generator.create_mass_spectrometry_data(
            n_samples=150,  # More than max_samples default
            n_proteins=600,  # More than max_proteins default
            missing_rate=0.3
        )
        
        fig = service.create_missing_value_heatmap(large_data)
        assert isinstance(fig, go.Figure)
        
        fig2 = service.create_protein_correlation_network(large_data)
        assert isinstance(fig2, go.Figure)
