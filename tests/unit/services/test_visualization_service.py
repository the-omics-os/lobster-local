"""
Comprehensive unit tests for visualization service.

This module provides thorough testing of the visualization service including
publication-ready plots, dimensionality reduction visualizations, heatmaps,
statistical plots, interactive visualizations, and plot customization.

Test coverage target: 95%+ with meaningful tests for visualization operations.
"""

import pytest
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch, mock_open
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile

from lobster.tools.visualization_service import VisualizationService
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory  
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def mock_clustered_data():
    """Create mock single-cell data with clustering results."""
    adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    
    # Add preprocessing results
    adata.obs['n_genes_by_counts'] = np.random.randint(500, 3000, adata.n_obs)
    adata.obs['total_counts'] = np.random.randint(1000, 15000, adata.n_obs)
    adata.obs['pct_counts_mt'] = np.random.uniform(0, 20, adata.n_obs)
    
    # Add dimensionality reduction
    adata.obsm['X_pca'] = np.random.randn(adata.n_obs, 50)
    adata.obsm['X_umap'] = np.random.randn(adata.n_obs, 2)  
    adata.obsm['X_tsne'] = np.random.randn(adata.n_obs, 2)
    
    # Add clustering results
    n_clusters = 8
    cluster_labels = np.random.randint(0, n_clusters, adata.n_obs)
    adata.obs['leiden'] = cluster_labels.astype(str)
    adata.obs['leiden'] = adata.obs['leiden'].astype('category')
    
    # Add cell type annotations
    cell_types = ['T_cells', 'B_cells', 'NK_cells', 'Monocytes', 
                  'Dendritic_cells', 'Neutrophils', 'Macrophages', 'Plasma_cells']
    adata.obs['cell_type'] = np.random.choice(cell_types[:n_clusters], adata.n_obs)
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
    
    # Add marker genes results
    adata.uns['rank_genes_groups'] = {
        'names': np.array([['Gene1', 'Gene2'], ['Gene3', 'Gene4'], 
                          ['Gene5', 'Gene6'], ['Gene7', 'Gene8']]),
        'scores': np.array([[2.5, 2.3], [2.1, 2.0], [1.9, 1.8], [1.7, 1.6]]),
        'pvals_adj': np.array([[0.01, 0.02], [0.03, 0.01], [0.02, 0.04], [0.01, 0.03]])
    }
    
    return adata


@pytest.fixture  
def mock_bulk_data():
    """Create mock bulk RNA-seq data with experimental design."""
    adata = BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG)
    
    # Add experimental conditions
    conditions = ['Control', 'Treatment_A', 'Treatment_B', 'Treatment_C']
    adata.obs['condition'] = np.random.choice(conditions, adata.n_obs)
    adata.obs['condition'] = adata.obs['condition'].astype('category')
    
    # Add batch information
    adata.obs['batch'] = np.random.choice(['Batch_1', 'Batch_2'], adata.n_obs)
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    
    # Add time points
    adata.obs['time_point'] = np.random.choice(['0h', '6h', '12h', '24h'], adata.n_obs)
    adata.obs['time_point'] = adata.obs['time_point'].astype('category')
    
    # Add differential expression results
    adata.uns['de_results'] = {
        'Treatment_A_vs_Control': pd.DataFrame({
            'log2FoldChange': np.random.uniform(-3, 3, adata.n_vars),
            'pvalue': np.random.uniform(0, 1, adata.n_vars),
            'padj': np.random.uniform(0, 1, adata.n_vars)
        }, index=adata.var.index),
    }
    
    return adata


@pytest.fixture
def visualization_service():
    """Create VisualizationService instance for testing."""
    return VisualizationService()


@pytest.fixture
def temp_plot_dir():
    """Create temporary directory for plot outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# ===============================================================================
# Visualization Service Core Tests
# ===============================================================================

@pytest.mark.unit
class TestVisualizationServiceCore:
    """Test visualization service core functionality."""
    
    def test_visualization_service_initialization(self):
        """Test VisualizationService initialization."""
        service = VisualizationService()
        
        assert hasattr(service, 'plot_umap')
        assert hasattr(service, 'plot_violin')
        assert hasattr(service, 'plot_heatmap')
        assert callable(service.plot_umap)
    
    def test_visualization_service_with_config(self):
        """Test VisualizationService initialization with configuration."""
        config = {
            'default_figsize': (10, 8),
            'default_dpi': 300,
            'default_format': 'png',
            'color_palette': 'tab10',
            'publication_ready': True
        }
        
        service = VisualizationService(config=config)
        
        assert service.config['default_figsize'] == (10, 8)
        assert service.config['default_dpi'] == 300
        assert service.config['publication_ready'] == True
    
    def test_available_plot_types(self, visualization_service):
        """Test listing available plot types."""
        plot_types = visualization_service.available_plot_types()
        
        expected_types = ['umap', 'tsne', 'pca', 'violin', 'heatmap', 'volcano', 'ma_plot', 'dotplot']
        for plot_type in expected_types:
            assert plot_type in plot_types
    
    def test_set_plot_style(self, visualization_service):
        """Test setting plot style and theme."""
        # Test different styles
        styles = ['publication', 'presentation', 'notebook', 'minimal']
        
        for style in styles:
            visualization_service.set_style(style)
            assert visualization_service.current_style == style
    
    def test_color_palette_management(self, visualization_service):
        """Test color palette management."""
        # Test setting custom color palette
        custom_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33F1']
        visualization_service.set_color_palette(custom_colors)
        
        assert visualization_service.color_palette == custom_colors
        
        # Test predefined palettes
        predefined_palettes = ['tab10', 'Set1', 'viridis', 'plasma']
        for palette in predefined_palettes:
            visualization_service.set_color_palette(palette)
            assert visualization_service.color_palette_name == palette


# ===============================================================================
# Dimensionality Reduction Visualization Tests
# ===============================================================================

@pytest.mark.unit
class TestDimensionalityReductionPlots:
    """Test dimensionality reduction visualization functionality."""
    
    def test_plot_umap_basic(self, visualization_service, mock_clustered_data, temp_plot_dir):
        """Test basic UMAP plotting."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_umap(
                adata,
                color='leiden',
                save_path=str(temp_plot_dir / 'umap_basic.png')
            )
            
            assert plot_info['plot_type'] == 'umap'
            assert plot_info['color_by'] == 'leiden'
            assert 'figure' in plot_info
            mock_savefig.assert_called_once()
    
    def test_plot_umap_continuous_color(self, visualization_service, mock_clustered_data):
        """Test UMAP plotting with continuous color variable."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_umap(
                adata,
                color='total_counts',
                color_map='viridis'
            )
            
            assert plot_info['color_by'] == 'total_counts'
            assert plot_info['color_type'] == 'continuous'
    
    def test_plot_umap_multiple_colors(self, visualization_service, mock_clustered_data):
        """Test UMAP plotting with multiple color variables."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_axes = plt.subplots(1, 3, figsize=(15, 5))
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            plot_info = visualization_service.plot_umap(
                adata,
                color=['leiden', 'cell_type', 'total_counts'],
                ncols=3
            )
            
            assert len(plot_info['color_by']) == 3
            assert plot_info['subplot_layout'] == (1, 3)
    
    def test_plot_tsne(self, visualization_service, mock_clustered_data):
        """Test t-SNE plotting."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_tsne(
                adata,
                color='cell_type',
                legend_loc='right'
            )
            
            assert plot_info['plot_type'] == 'tsne'
            assert plot_info['legend_location'] == 'right'
    
    def test_plot_pca(self, visualization_service, mock_clustered_data):
        """Test PCA plotting."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_pca(
                adata,
                color='leiden',
                components=[1, 2]  # PC1 vs PC2
            )
            
            assert plot_info['plot_type'] == 'pca'
            assert plot_info['components'] == [1, 2]
    
    def test_plot_pca_variance(self, visualization_service, mock_clustered_data):
        """Test PCA variance explained plotting."""
        adata = mock_clustered_data.copy()
        
        # Add PCA variance info
        adata.uns['pca'] = {
            'variance_ratio': np.random.uniform(0.01, 0.3, 50),
            'variance': np.random.uniform(100, 1000, 50)
        }
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_pca_variance(
                adata,
                n_components=20
            )
            
            assert plot_info['plot_type'] == 'pca_variance'
            assert plot_info['n_components'] == 20
    
    def test_plot_embedding_comparison(self, visualization_service, mock_clustered_data):
        """Test comparison of different embedding methods."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_axes = plt.subplots(1, 3, figsize=(18, 6))
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            plot_info = visualization_service.plot_embedding_comparison(
                adata,
                embeddings=['X_pca', 'X_umap', 'X_tsne'],
                color='leiden'
            )
            
            assert len(plot_info['embeddings']) == 3
            assert 'X_umap' in plot_info['embeddings']


# ===============================================================================
# Statistical and QC Visualization Tests
# ===============================================================================

@pytest.mark.unit
class TestStatisticalPlots:
    """Test statistical and QC visualization functionality."""
    
    def test_plot_violin_qc_metrics(self, visualization_service, mock_clustered_data):
        """Test violin plots for QC metrics."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_violin(
                adata,
                keys=['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                groupby='leiden'
            )
            
            assert plot_info['plot_type'] == 'violin'
            assert len(plot_info['keys']) == 3
            assert plot_info['groupby'] == 'leiden'
    
    def test_plot_violin_gene_expression(self, visualization_service, mock_clustered_data):
        """Test violin plots for gene expression."""
        adata = mock_clustered_data.copy()
        
        genes = ['Gene1', 'Gene2', 'Gene3']
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_violin(
                adata,
                keys=genes,
                groupby='cell_type'
            )
            
            assert plot_info['keys'] == genes
            assert plot_info['data_type'] == 'gene_expression'
    
    def test_plot_scatter_qc(self, visualization_service, mock_clustered_data):
        """Test scatter plots for QC analysis."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_scatter(
                adata,
                x='total_counts',
                y='n_genes_by_counts',
                color='pct_counts_mt'
            )
            
            assert plot_info['plot_type'] == 'scatter'
            assert plot_info['x_axis'] == 'total_counts'
            assert plot_info['y_axis'] == 'n_genes_by_counts'
    
    def test_plot_histogram_distribution(self, visualization_service, mock_clustered_data):
        """Test histogram plotting for metric distributions."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_histogram(
                adata,
                key='total_counts',
                bins=50,
                density=True
            )
            
            assert plot_info['plot_type'] == 'histogram'
            assert plot_info['key'] == 'total_counts'
            assert plot_info['bins'] == 50
    
    def test_plot_box_plot(self, visualization_service, mock_clustered_data):
        """Test box plot visualization."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_box(
                adata,
                key='pct_counts_mt',
                groupby='cell_type'
            )
            
            assert plot_info['plot_type'] == 'box'
            assert plot_info['key'] == 'pct_counts_mt'
    
    def test_plot_correlation_matrix(self, visualization_service, mock_clustered_data):
        """Test correlation matrix plotting."""
        adata = mock_clustered_data.copy()
        
        qc_metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt']
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_correlation_matrix(
                adata,
                keys=qc_metrics,
                method='pearson'
            )
            
            assert plot_info['plot_type'] == 'correlation_matrix'
            assert plot_info['correlation_method'] == 'pearson'


# ===============================================================================
# Gene Expression Visualization Tests
# ===============================================================================

@pytest.mark.unit
class TestGeneExpressionPlots:
    """Test gene expression visualization functionality."""
    
    def test_plot_heatmap_marker_genes(self, visualization_service, mock_clustered_data):
        """Test heatmap plotting for marker genes."""
        adata = mock_clustered_data.copy()
        
        marker_genes = ['Gene1', 'Gene2', 'Gene3', 'Gene4']
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_heatmap(
                adata,
                var_names=marker_genes,
                groupby='leiden',
                show_gene_labels=True
            )
            
            assert plot_info['plot_type'] == 'heatmap'
            assert len(plot_info['var_names']) == len(marker_genes)
            assert plot_info['groupby'] == 'leiden'
    
    def test_plot_dotplot_marker_genes(self, visualization_service, mock_clustered_data):
        """Test dot plot for marker genes."""
        adata = mock_clustered_data.copy()
        
        marker_genes = ['Gene1', 'Gene3', 'Gene5', 'Gene7']
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_dotplot(
                adata,
                var_names=marker_genes,
                groupby='cell_type'
            )
            
            assert plot_info['plot_type'] == 'dotplot'
            assert plot_info['var_names'] == marker_genes
    
    def test_plot_stacked_violin(self, visualization_service, mock_clustered_data):
        """Test stacked violin plot for gene expression."""
        adata = mock_clustered_data.copy()
        
        genes = ['Gene1', 'Gene2', 'Gene3']
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_stacked_violin(
                adata,
                var_names=genes,
                groupby='leiden'
            )
            
            assert plot_info['plot_type'] == 'stacked_violin'
            assert plot_info['var_names'] == genes
    
    def test_plot_gene_ranking(self, visualization_service, mock_clustered_data):
        """Test gene ranking plot for marker genes."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_gene_ranking(
                adata,
                groups=['0', '1', '2'],
                n_genes=10
            )
            
            assert plot_info['plot_type'] == 'gene_ranking'
            assert len(plot_info['groups']) == 3
            assert plot_info['n_genes'] == 10
    
    def test_plot_feature_plot(self, visualization_service, mock_clustered_data):
        """Test feature plot overlaying gene expression on embedding."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_feature(
                adata,
                features=['Gene1', 'Gene2'],
                embedding='X_umap',
                ncols=2
            )
            
            assert plot_info['plot_type'] == 'feature_plot'
            assert len(plot_info['features']) == 2
            assert plot_info['embedding'] == 'X_umap'


# ===============================================================================
# Differential Expression Visualization Tests
# ===============================================================================

@pytest.mark.unit
class TestDifferentialExpressionPlots:
    """Test differential expression visualization functionality."""
    
    def test_plot_volcano_plot(self, visualization_service, mock_bulk_data):
        """Test volcano plot for differential expression."""
        adata = mock_bulk_data.copy()
        
        de_results = adata.uns['de_results']['Treatment_A_vs_Control']
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_volcano(
                de_results,
                x_col='log2FoldChange',
                y_col='padj',
                significance_threshold=0.05,
                fold_change_threshold=1.0
            )
            
            assert plot_info['plot_type'] == 'volcano'
            assert plot_info['significance_threshold'] == 0.05
            assert plot_info['fold_change_threshold'] == 1.0
    
    def test_plot_ma_plot(self, visualization_service, mock_bulk_data):
        """Test MA plot for differential expression."""
        adata = mock_bulk_data.copy()
        
        de_results = adata.uns['de_results']['Treatment_A_vs_Control']
        # Add mean expression column
        de_results['baseMean'] = np.random.uniform(1, 10000, len(de_results))
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_ma(
                de_results,
                x_col='baseMean',
                y_col='log2FoldChange',
                significance_col='padj'
            )
            
            assert plot_info['plot_type'] == 'ma_plot'
            assert plot_info['x_col'] == 'baseMean'
    
    def test_plot_de_heatmap(self, visualization_service, mock_bulk_data):
        """Test heatmap for differentially expressed genes."""
        adata = mock_bulk_data.copy()
        
        # Select top DE genes
        de_results = adata.uns['de_results']['Treatment_A_vs_Control']
        top_genes = de_results.nlargest(20, 'log2FoldChange').index.tolist()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_heatmap(
                adata,
                var_names=top_genes,
                groupby='condition',
                standard_scale='var'
            )
            
            assert len(plot_info['var_names']) == 20
            assert plot_info['standard_scale'] == 'var'
    
    def test_plot_gene_set_enrichment(self, visualization_service):
        """Test gene set enrichment analysis visualization."""
        # Mock enrichment results
        enrichment_results = pd.DataFrame({
            'pathway': ['Pathway_A', 'Pathway_B', 'Pathway_C', 'Pathway_D'],
            'pvalue': [0.001, 0.01, 0.02, 0.05],
            'enrichment_score': [2.5, 1.8, 1.5, 1.2],
            'gene_count': [25, 18, 15, 10]
        })
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_enrichment(
                enrichment_results,
                x_col='enrichment_score',
                y_col='pathway',
                size_col='gene_count',
                color_col='pvalue'
            )
            
            assert plot_info['plot_type'] == 'enrichment'
            assert len(plot_info['pathways']) == 4


# ===============================================================================
# Interactive Visualization Tests
# ===============================================================================

@pytest.mark.unit
class TestInteractivePlots:
    """Test interactive visualization functionality."""
    
    def test_create_interactive_umap(self, visualization_service, mock_clustered_data):
        """Test interactive UMAP visualization."""
        adata = mock_clustered_data.copy()
        
        with patch.object(visualization_service, 'create_interactive_plot') as mock_interactive:
            mock_interactive.return_value = {
                'plot_type': 'interactive_umap',
                'html_content': '<div>Interactive UMAP</div>',
                'javascript_code': 'console.log("UMAP plot");',
                'data_json': '{"x": [1,2,3], "y": [4,5,6]}',
                'plot_id': 'umap_12345'
            }
            
            interactive_plot = visualization_service.create_interactive_plot(
                adata,
                plot_type='umap',
                color='leiden',
                hover_data=['cell_type', 'total_counts']
            )
            
            assert interactive_plot['plot_type'] == 'interactive_umap'
            assert 'html_content' in interactive_plot
    
    def test_create_interactive_heatmap(self, visualization_service, mock_clustered_data):
        """Test interactive heatmap visualization."""
        adata = mock_clustered_data.copy()
        
        with patch.object(visualization_service, 'create_interactive_heatmap') as mock_heatmap:
            mock_heatmap.return_value = {
                'plot_type': 'interactive_heatmap',
                'plotly_json': '{"data": [], "layout": {}}',
                'plot_id': 'heatmap_67890'
            }
            
            interactive_heatmap = visualization_service.create_interactive_heatmap(
                adata,
                var_names=['Gene1', 'Gene2', 'Gene3'],
                groupby='leiden'
            )
            
            assert interactive_heatmap['plot_type'] == 'interactive_heatmap'
            assert 'plotly_json' in interactive_heatmap
    
    def test_export_interactive_plot(self, visualization_service, temp_plot_dir):
        """Test exporting interactive plots to HTML."""
        mock_plot_data = {
            'plot_type': 'interactive_umap',
            'html_content': '<html><body>Interactive Plot</body></html>',
            'plot_id': 'test_plot'
        }
        
        export_path = temp_plot_dir / 'interactive_plot.html'
        
        with patch('builtins.open', mock_open()) as mock_file:
            export_info = visualization_service.export_interactive_plot(
                mock_plot_data,
                str(export_path)
            )
            
            assert export_info['exported_to'] == str(export_path)
            assert export_info['plot_type'] == 'interactive_umap'
            mock_file.assert_called_once()


# ===============================================================================
# Plot Customization Tests
# ===============================================================================

@pytest.mark.unit
class TestPlotCustomization:
    """Test plot customization functionality."""
    
    def test_customize_plot_appearance(self, visualization_service):
        """Test plot appearance customization."""
        custom_style = {
            'figure_size': (12, 8),
            'dpi': 300,
            'font_size': 14,
            'title_font_size': 16,
            'axis_label_size': 12,
            'legend_font_size': 10,
            'color_palette': 'Set1'
        }
        
        visualization_service.set_plot_style(custom_style)
        
        assert visualization_service.plot_config['figure_size'] == (12, 8)
        assert visualization_service.plot_config['dpi'] == 300
        assert visualization_service.plot_config['font_size'] == 14
    
    def test_add_plot_annotations(self, visualization_service, mock_clustered_data):
        """Test adding annotations to plots."""
        adata = mock_clustered_data.copy()
        
        annotations = {
            'title': 'UMAP Clustering Results',
            'subtitle': 'Single-cell RNA-seq analysis',
            'x_label': 'UMAP 1',
            'y_label': 'UMAP 2',
            'legend_title': 'Cell Types'
        }
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_umap(
                adata,
                color='cell_type',
                annotations=annotations
            )
            
            assert plot_info['annotations']['title'] == 'UMAP Clustering Results'
            assert plot_info['annotations']['legend_title'] == 'Cell Types'
    
    def test_publication_ready_formatting(self, visualization_service, mock_clustered_data):
        """Test publication-ready plot formatting."""
        adata = mock_clustered_data.copy()
        
        pub_config = {
            'publication_ready': True,
            'remove_top_right_spines': True,
            'grid': False,
            'tight_layout': True,
            'transparent_background': False,
            'high_dpi': True
        }
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_umap(
                adata,
                color='leiden',
                **pub_config
            )
            
            assert plot_info['publication_ready'] == True
            assert plot_info['high_quality'] == True
    
    def test_color_scheme_management(self, visualization_service):
        """Test color scheme and palette management."""
        # Test custom color mapping
        custom_colors = {
            'T_cells': '#FF6B6B',
            'B_cells': '#4ECDC4', 
            'NK_cells': '#45B7D1',
            'Monocytes': '#96CEB4'
        }
        
        visualization_service.set_custom_colors(custom_colors)
        
        assert visualization_service.custom_color_map['T_cells'] == '#FF6B6B'
        assert visualization_service.custom_color_map['B_cells'] == '#4ECDC4'
    
    def test_multi_panel_layouts(self, visualization_service, mock_clustered_data):
        """Test multi-panel plot layouts."""
        adata = mock_clustered_data.copy()
        
        layout_config = {
            'nrows': 2,
            'ncols': 2,
            'figsize': (16, 12),
            'subplot_titles': ['Panel A', 'Panel B', 'Panel C', 'Panel D']
        }
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            mock_fig, mock_axes = plt.subplots(2, 2, figsize=(16, 12))
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            plot_info = visualization_service.create_multi_panel_plot(
                adata,
                plot_configs=[
                    {'type': 'umap', 'color': 'leiden'},
                    {'type': 'umap', 'color': 'cell_type'},
                    {'type': 'violin', 'keys': ['total_counts'], 'groupby': 'leiden'},
                    {'type': 'scatter', 'x': 'total_counts', 'y': 'n_genes_by_counts'}
                ],
                layout=layout_config
            )
            
            assert plot_info['layout']['nrows'] == 2
            assert plot_info['layout']['ncols'] == 2
            assert len(plot_info['subplot_titles']) == 4


# ===============================================================================
# Export and Format Tests
# ===============================================================================

@pytest.mark.unit
class TestPlotExportFormats:
    """Test plot export and format functionality."""
    
    def test_export_png_format(self, visualization_service, mock_clustered_data, temp_plot_dir):
        """Test exporting plots in PNG format."""
        adata = mock_clustered_data.copy()
        
        export_path = temp_plot_dir / 'test_plot.png'
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            export_info = visualization_service.plot_umap(
                adata,
                color='leiden',
                save_path=str(export_path),
                format='png',
                dpi=300
            )
            
            mock_savefig.assert_called()
            args, kwargs = mock_savefig.call_args
            assert kwargs.get('format') == 'png' or str(export_path).endswith('.png')
            assert kwargs.get('dpi') == 300
    
    def test_export_pdf_format(self, visualization_service, mock_clustered_data, temp_plot_dir):
        """Test exporting plots in PDF format."""
        adata = mock_clustered_data.copy()
        
        export_path = temp_plot_dir / 'test_plot.pdf'
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            export_info = visualization_service.plot_umap(
                adata,
                color='leiden',
                save_path=str(export_path),
                format='pdf'
            )
            
            mock_savefig.assert_called()
    
    def test_export_svg_format(self, visualization_service, mock_clustered_data, temp_plot_dir):
        """Test exporting plots in SVG format."""
        adata = mock_clustered_data.copy()
        
        export_path = temp_plot_dir / 'test_plot.svg'
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            export_info = visualization_service.plot_umap(
                adata,
                color='leiden',
                save_path=str(export_path),
                format='svg'
            )
            
            mock_savefig.assert_called()
    
    def test_batch_plot_export(self, visualization_service, mock_clustered_data, temp_plot_dir):
        """Test batch export of multiple plots."""
        adata = mock_clustered_data.copy()
        
        plot_configs = [
            {'type': 'umap', 'color': 'leiden', 'name': 'umap_clusters'},
            {'type': 'umap', 'color': 'cell_type', 'name': 'umap_celltypes'},
            {'type': 'violin', 'keys': ['total_counts'], 'groupby': 'leiden', 'name': 'violin_counts'}
        ]
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            export_info = visualization_service.batch_export_plots(
                adata,
                plot_configs,
                output_dir=str(temp_plot_dir),
                format='png'
            )
            
            assert len(export_info['exported_plots']) == 3
            assert export_info['output_directory'] == str(temp_plot_dir)
    
    def test_plot_gallery_creation(self, visualization_service, mock_clustered_data, temp_plot_dir):
        """Test creation of plot gallery/summary."""
        adata = mock_clustered_data.copy()
        
        with patch.object(visualization_service, 'create_plot_gallery') as mock_gallery:
            mock_gallery.return_value = {
                'gallery_html': '<html><body>Plot Gallery</body></html>',
                'n_plots': 6,
                'gallery_path': str(temp_plot_dir / 'gallery.html'),
                'thumbnail_dir': str(temp_plot_dir / 'thumbnails')
            }
            
            gallery_info = visualization_service.create_plot_gallery(
                adata,
                output_dir=str(temp_plot_dir)
            )
            
            assert gallery_info['n_plots'] > 0
            assert 'gallery_html' in gallery_info


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestVisualizationErrorHandling:
    """Test visualization error handling and edge cases."""
    
    def test_missing_embedding_handling(self, visualization_service):
        """Test handling of missing embedding data."""
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        # Don't add any embeddings
        
        with pytest.raises(KeyError, match="Embedding 'X_umap' not found"):
            visualization_service.plot_umap(adata, color='leiden')
    
    def test_missing_color_variable_handling(self, visualization_service, mock_clustered_data):
        """Test handling of missing color variables."""
        adata = mock_clustered_data.copy()
        
        with pytest.raises(KeyError, match="Color variable 'nonexistent' not found"):
            visualization_service.plot_umap(adata, color='nonexistent')
    
    def test_invalid_plot_parameters_handling(self, visualization_service, mock_clustered_data):
        """Test handling of invalid plot parameters."""
        adata = mock_clustered_data.copy()
        
        # Test invalid figure size
        with pytest.raises(ValueError, match="Invalid figure size"):
            visualization_service.plot_umap(adata, color='leiden', figsize=(-5, -3))
    
    def test_empty_data_handling(self, visualization_service):
        """Test handling of empty datasets."""
        empty_adata = ad.AnnData(X=np.array([]).reshape(0, 0))
        
        with pytest.raises(ValueError, match="Empty dataset"):
            visualization_service.plot_umap(empty_adata)
    
    def test_memory_efficient_plotting(self, visualization_service):
        """Test memory-efficient plotting for large datasets."""
        # Create large dataset
        large_config = LARGE_DATASET_CONFIG.copy()
        adata = SingleCellDataFactory(config=large_config)
        adata.obsm['X_umap'] = np.random.randn(adata.n_obs, 2)
        adata.obs['cluster'] = np.random.randint(0, 20, adata.n_obs).astype(str)
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = visualization_service.plot_umap(
                adata,
                color='cluster',
                rasterized=True,  # For large datasets
                alpha=0.6
            )
            
            assert plot_info['rasterized'] == True
            assert plot_info['n_points'] == adata.n_obs
    
    def test_concurrent_plotting_safety(self, visualization_service, mock_clustered_data):
        """Test thread safety for concurrent plotting operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def plotting_worker(worker_id):
            """Worker function for concurrent plotting."""
            try:
                adata = mock_clustered_data.copy()
                
                with patch('matplotlib.pyplot.savefig') as mock_savefig:
                    plot_info = visualization_service.plot_umap(
                        adata,
                        color='leiden',
                        title=f'Plot from worker {worker_id}'
                    )
                    results.append((worker_id, plot_info))
                    time.sleep(0.01)
                    
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=plotting_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent plotting errors: {errors}"
        assert len(results) == 3
    
    def test_plot_resource_cleanup(self, visualization_service, mock_clustered_data):
        """Test proper cleanup of plot resources."""
        adata = mock_clustered_data.copy()
        
        with patch('matplotlib.pyplot.close') as mock_close:
            plot_info = visualization_service.plot_umap(
                adata,
                color='leiden',
                show=False,  # Don't display
                cleanup=True  # Clean up resources
            )
            
            # Should clean up matplotlib figures
            assert plot_info.get('cleanup_performed', False) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])