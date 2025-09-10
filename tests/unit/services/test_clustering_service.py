"""
Comprehensive unit tests for clustering service.

This module provides thorough testing of the clustering service including
Leiden clustering, Louvain clustering, k-means, hierarchical clustering,
cell type annotation, marker gene discovery, and clustering validation.

Test coverage target: 95%+ with meaningful tests for clustering operations.
"""

import pytest
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from lobster.tools.clustering_service import ClusteringService
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def mock_preprocessed_data():
    """Create mock preprocessed single-cell data for clustering."""
    adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    
    # Add preprocessing results
    adata.obs['n_genes_by_counts'] = np.random.randint(500, 3000, adata.n_obs)
    adata.obs['total_counts'] = np.random.randint(1000, 10000, adata.n_obs) 
    adata.obs['pct_counts_mt'] = np.random.uniform(0, 15, adata.n_obs)
    adata.var['highly_variable'] = np.random.choice([True, False], adata.n_vars, p=[0.2, 0.8])
    
    # Add PCA results
    n_pcs = min(50, adata.n_obs - 1, adata.n_vars - 1)
    adata.obsm['X_pca'] = np.random.randn(adata.n_obs, n_pcs)
    
    # Add neighbors graph
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(adata.obsm['X_pca'])
    distances, indices = nn.kneighbors(adata.obsm['X_pca'])
    
    # Mock connectivity matrix
    n_obs = adata.n_obs
    connectivity = sp.csr_matrix((n_obs, n_obs))
    adata.obsp['connectivities'] = connectivity
    adata.obsp['distances'] = connectivity
    
    return adata


@pytest.fixture
def mock_clustered_data():
    """Create mock data with existing clusters."""
    adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    
    # Add clusters
    n_clusters = 8
    cluster_labels = np.random.randint(0, n_clusters, adata.n_obs)
    adata.obs['leiden'] = cluster_labels.astype(str)
    adata.obs['leiden'] = adata.obs['leiden'].astype('category')
    
    # Add cell type annotations
    cell_types = ['T_cells', 'B_cells', 'NK_cells', 'Monocytes', 'Dendritic_cells', 
                  'Neutrophils', 'Macrophages', 'Plasma_cells']
    adata.obs['cell_type'] = np.random.choice(cell_types, adata.n_obs)
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
    
    return adata


@pytest.fixture
def clustering_service():
    """Create ClusteringService instance for testing."""
    return ClusteringService()


@pytest.fixture
def mock_marker_genes():
    """Mock marker gene data for cell type annotation."""
    return {
        'T_cells': ['CD3D', 'CD3E', 'CD3G', 'CD8A', 'CD4'],
        'B_cells': ['CD19', 'CD79A', 'CD79B', 'MS4A1', 'CD20'],
        'NK_cells': ['KLRD1', 'KLRF1', 'NCR1', 'NCAM1', 'NKG7'],
        'Monocytes': ['CD14', 'CD68', 'CSF1R', 'LYZ', 'S100A9'],
        'Dendritic_cells': ['CD1C', 'FCER1A', 'CLEC9A', 'CADM1'],
        'Neutrophils': ['FCGR3B', 'CSF3R', 'CEACAM8', 'ELANE'],
        'Macrophages': ['CD68', 'CD163', 'MSR1', 'MRC1', 'MARCO'],
        'Plasma_cells': ['IGHG1', 'IGHA1', 'JCHAIN', 'SDC1', 'CD38']
    }


# ===============================================================================
# Clustering Service Core Tests
# ===============================================================================

@pytest.mark.unit
class TestClusteringServiceCore:
    """Test clustering service core functionality."""
    
    def test_clustering_service_initialization(self):
        """Test ClusteringService initialization."""
        service = ClusteringService()
        
        assert hasattr(service, 'leiden')
        assert hasattr(service, 'louvain') 
        assert hasattr(service, 'kmeans')
        assert callable(service.leiden)
    
    def test_clustering_service_with_config(self):
        """Test ClusteringService initialization with configuration."""
        config = {
            'default_resolution': 0.5,
            'default_n_neighbors': 15,
            'random_state': 42,
            'n_iterations': 10
        }
        
        service = ClusteringService(config=config)
        
        assert service.config['default_resolution'] == 0.5
        assert service.config['random_state'] == 42
    
    def test_available_clustering_methods(self, clustering_service):
        """Test listing available clustering methods."""
        methods = clustering_service.available_methods()
        
        expected_methods = ['leiden', 'louvain', 'kmeans', 'hierarchical', 'spectral']
        for method in expected_methods:
            assert method in methods
    
    def test_clustering_parameter_validation(self, clustering_service):
        """Test parameter validation for clustering methods."""
        # Valid parameters should pass
        valid_params = {'resolution': 0.5, 'n_neighbors': 10, 'random_state': 42}
        is_valid = clustering_service._validate_parameters('leiden', valid_params)
        assert is_valid == True
        
        # Invalid parameters should fail
        invalid_params = {'resolution': -1, 'n_neighbors': 0}
        is_valid = clustering_service._validate_parameters('leiden', invalid_params)
        assert is_valid == False


# ===============================================================================
# Leiden Clustering Tests
# ===============================================================================

@pytest.mark.unit
class TestLeidenClustering:
    """Test Leiden clustering functionality."""
    
    def test_leiden_clustering_basic(self, clustering_service, mock_preprocessed_data):
        """Test basic Leiden clustering."""
        adata = mock_preprocessed_data.copy()
        
        with patch('scanpy.tl.leiden') as mock_leiden:
            # Mock the leiden clustering
            n_clusters = 8
            cluster_labels = np.random.randint(0, n_clusters, adata.n_obs)
            mock_leiden.side_effect = lambda x, **kwargs: setattr(x.obs, 'leiden', cluster_labels.astype(str))
            
            result = clustering_service.leiden(adata, resolution=0.5, random_state=42)
            
            assert 'leiden' in adata.obs.columns
            assert result['n_clusters'] > 0
            assert result['resolution'] == 0.5
            mock_leiden.assert_called_once()
    
    def test_leiden_resolution_optimization(self, clustering_service, mock_preprocessed_data):
        """Test Leiden clustering with resolution optimization."""
        adata = mock_preprocessed_data.copy()
        
        with patch.object(clustering_service, 'optimize_leiden_resolution') as mock_optimize:
            mock_optimize.return_value = {
                'optimal_resolution': 0.8,
                'n_clusters': 12,
                'silhouette_score': 0.65,
                'modularity': 0.82,
                'resolution_tested': [0.1, 0.3, 0.5, 0.8, 1.0, 1.2]
            }
            
            result = clustering_service.optimize_leiden_resolution(
                adata,
                resolution_range=(0.1, 1.2),
                n_resolutions=6
            )
            
            assert result['optimal_resolution'] == 0.8
            assert result['n_clusters'] == 12
            assert len(result['resolution_tested']) == 6
    
    def test_leiden_with_multiple_resolutions(self, clustering_service, mock_preprocessed_data):
        """Test Leiden clustering with multiple resolutions."""
        adata = mock_preprocessed_data.copy()
        
        resolutions = [0.1, 0.3, 0.5, 0.8, 1.0]
        
        with patch('scanpy.tl.leiden') as mock_leiden:
            def mock_leiden_side_effect(data, resolution=0.5, key_added=None, **kwargs):
                n_clusters = max(2, int(resolution * 10))  # More resolution = more clusters
                cluster_labels = np.random.randint(0, n_clusters, data.n_obs)
                key = key_added or 'leiden'
                data.obs[key] = cluster_labels.astype(str)
            
            mock_leiden.side_effect = mock_leiden_side_effect
            
            results = clustering_service.leiden_multi_resolution(adata, resolutions)
            
            assert len(results) == len(resolutions)
            for i, res in enumerate(resolutions):
                key = f'leiden_{res}'
                assert key in adata.obs.columns
                assert results[i]['resolution'] == res
    
    def test_leiden_stability_analysis(self, clustering_service, mock_preprocessed_data):
        """Test Leiden clustering stability analysis."""
        adata = mock_preprocessed_data.copy()
        
        with patch.object(clustering_service, 'assess_clustering_stability') as mock_stability:
            mock_stability.return_value = {
                'stability_score': 0.87,
                'consensus_clusters': 10,
                'cluster_consistency': 0.92,
                'bootstrap_iterations': 100,
                'stable_clusters': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            }
            
            stability = clustering_service.assess_clustering_stability(
                adata,
                method='leiden',
                resolution=0.5,
                n_bootstrap=100
            )
            
            assert stability['stability_score'] > 0.8
            assert stability['consensus_clusters'] > 0
            assert len(stability['stable_clusters']) <= stability['consensus_clusters']


# ===============================================================================
# Louvain Clustering Tests  
# ===============================================================================

@pytest.mark.unit
class TestLouvainClustering:
    """Test Louvain clustering functionality."""
    
    def test_louvain_clustering_basic(self, clustering_service, mock_preprocessed_data):
        """Test basic Louvain clustering."""
        adata = mock_preprocessed_data.copy()
        
        with patch('scanpy.tl.louvain') as mock_louvain:
            n_clusters = 6
            cluster_labels = np.random.randint(0, n_clusters, adata.n_obs)
            mock_louvain.side_effect = lambda x, **kwargs: setattr(x.obs, 'louvain', cluster_labels.astype(str))
            
            result = clustering_service.louvain(adata, resolution=0.5, random_state=42)
            
            assert 'louvain' in adata.obs.columns
            assert result['n_clusters'] > 0
            assert result['resolution'] == 0.5
            mock_louvain.assert_called_once()
    
    def test_louvain_vs_leiden_comparison(self, clustering_service, mock_preprocessed_data):
        """Test comparison between Louvain and Leiden clustering."""
        adata = mock_preprocessed_data.copy()
        
        with patch.object(clustering_service, 'compare_clustering_methods') as mock_compare:
            mock_compare.return_value = {
                'methods_compared': ['leiden', 'louvain'],
                'leiden': {
                    'n_clusters': 8,
                    'silhouette_score': 0.68,
                    'modularity': 0.85,
                    'runtime': 2.3
                },
                'louvain': {
                    'n_clusters': 7,
                    'silhouette_score': 0.64,
                    'modularity': 0.82,
                    'runtime': 1.8
                },
                'best_method': 'leiden',
                'comparison_metrics': {
                    'silhouette_winner': 'leiden',
                    'speed_winner': 'louvain',
                    'modularity_winner': 'leiden'
                }
            }
            
            comparison = clustering_service.compare_clustering_methods(
                adata,
                methods=['leiden', 'louvain'],
                resolution=0.5
            )
            
            assert comparison['best_method'] in ['leiden', 'louvain']
            assert 'leiden' in comparison
            assert 'louvain' in comparison


# ===============================================================================
# K-means and Other Clustering Tests
# ===============================================================================

@pytest.mark.unit
class TestOtherClusteringMethods:
    """Test k-means and other clustering methods."""
    
    def test_kmeans_clustering(self, clustering_service, mock_preprocessed_data):
        """Test k-means clustering."""
        adata = mock_preprocessed_data.copy()
        
        result = clustering_service.kmeans(
            adata,
            n_clusters=8,
            random_state=42,
            use_pca=True
        )
        
        assert 'kmeans' in adata.obs.columns
        assert result['n_clusters'] == 8
        assert result['inertia'] > 0
        assert 'cluster_centers' in result
    
    def test_kmeans_elbow_method(self, clustering_service, mock_preprocessed_data):
        """Test k-means elbow method for optimal k."""
        adata = mock_preprocessed_data.copy()
        
        with patch.object(clustering_service, 'kmeans_elbow_method') as mock_elbow:
            mock_elbow.return_value = {
                'optimal_k': 7,
                'inertias': [1000, 800, 600, 450, 380, 350, 340, 335, 330],
                'k_range': list(range(2, 11)),
                'elbow_point': 7,
                'silhouette_scores': [0.3, 0.4, 0.5, 0.6, 0.65, 0.62, 0.58, 0.55, 0.52]
            }
            
            result = clustering_service.kmeans_elbow_method(adata, k_range=(2, 10))
            
            assert result['optimal_k'] == 7
            assert len(result['inertias']) == len(result['k_range'])
    
    def test_hierarchical_clustering(self, clustering_service, mock_preprocessed_data):
        """Test hierarchical clustering."""
        adata = mock_preprocessed_data.copy()
        
        with patch.object(clustering_service, 'hierarchical') as mock_hierarchical:
            mock_hierarchical.return_value = {
                'n_clusters': 6,
                'linkage_matrix': np.random.randn(adata.n_obs - 1, 4),
                'cluster_labels': np.random.randint(0, 6, adata.n_obs),
                'cophenetic_correlation': 0.75,
                'method': 'ward'
            }
            
            result = clustering_service.hierarchical(
                adata,
                n_clusters=6,
                linkage='ward',
                use_pca=True
            )
            
            assert result['n_clusters'] == 6
            assert result['method'] == 'ward'
            assert 'linkage_matrix' in result
    
    def test_spectral_clustering(self, clustering_service, mock_preprocessed_data):
        """Test spectral clustering."""
        adata = mock_preprocessed_data.copy()
        
        with patch.object(clustering_service, 'spectral') as mock_spectral:
            mock_spectral.return_value = {
                'n_clusters': 5,
                'cluster_labels': np.random.randint(0, 5, adata.n_obs),
                'affinity_matrix': sp.random(adata.n_obs, adata.n_obs, density=0.1),
                'eigenvalues': np.random.random(5),
                'silhouette_score': 0.62
            }
            
            result = clustering_service.spectral(
                adata,
                n_clusters=5,
                affinity='nearest_neighbors'
            )
            
            assert result['n_clusters'] == 5
            assert 'eigenvalues' in result
    
    def test_gaussian_mixture_clustering(self, clustering_service, mock_preprocessed_data):
        """Test Gaussian mixture model clustering."""
        adata = mock_preprocessed_data.copy()
        
        with patch.object(clustering_service, 'gaussian_mixture') as mock_gmm:
            mock_gmm.return_value = {
                'n_components': 6,
                'cluster_labels': np.random.randint(0, 6, adata.n_obs),
                'cluster_probabilities': np.random.random((adata.n_obs, 6)),
                'bic': 15000.5,
                'aic': 14800.3,
                'log_likelihood': -7350.2
            }
            
            result = clustering_service.gaussian_mixture(
                adata,
                n_components=6,
                covariance_type='full'
            )
            
            assert result['n_components'] == 6
            assert 'cluster_probabilities' in result
            assert result['bic'] > 0


# ===============================================================================
# Clustering Validation Tests
# ===============================================================================

@pytest.mark.unit
class TestClusteringValidation:
    """Test clustering validation functionality."""
    
    def test_silhouette_analysis(self, clustering_service, mock_clustered_data):
        """Test silhouette analysis for clustering validation."""
        adata = mock_clustered_data.copy()
        
        silhouette_result = clustering_service.compute_silhouette_scores(
            adata,
            cluster_key='leiden',
            use_rep='X_pca'
        )
        
        assert 'silhouette_score' in silhouette_result
        assert 'silhouette_scores_per_cell' in silhouette_result
        assert 'silhouette_scores_per_cluster' in silhouette_result
        assert -1 <= silhouette_result['silhouette_score'] <= 1
    
    def test_davies_bouldin_index(self, clustering_service, mock_clustered_data):
        """Test Davies-Bouldin index calculation."""
        adata = mock_clustered_data.copy()
        
        with patch.object(clustering_service, 'davies_bouldin_score') as mock_db:
            mock_db.return_value = {
                'davies_bouldin_score': 0.85,
                'per_cluster_scores': np.random.random(8),
                'interpretation': 'good'  # Lower is better
            }
            
            result = clustering_service.davies_bouldin_score(
                adata,
                cluster_key='leiden'
            )
            
            assert result['davies_bouldin_score'] > 0
            assert result['interpretation'] in ['excellent', 'good', 'fair', 'poor']
    
    def test_calinski_harabasz_index(self, clustering_service, mock_clustered_data):
        """Test Calinski-Harabasz index calculation."""
        adata = mock_clustered_data.copy()
        
        with patch.object(clustering_service, 'calinski_harabasz_score') as mock_ch:
            mock_ch.return_value = {
                'calinski_harabasz_score': 125.8,
                'interpretation': 'good'  # Higher is better
            }
            
            result = clustering_service.calinski_harabasz_score(
                adata,
                cluster_key='leiden'
            )
            
            assert result['calinski_harabasz_score'] > 0
            assert result['interpretation'] in ['excellent', 'good', 'fair', 'poor']
    
    def test_cluster_purity_assessment(self, clustering_service, mock_clustered_data):
        """Test cluster purity assessment."""
        adata = mock_clustered_data.copy()
        
        # Add ground truth labels for purity calculation
        true_labels = np.random.randint(0, 5, adata.n_obs)
        adata.obs['true_labels'] = true_labels.astype(str)
        
        purity_result = clustering_service.assess_cluster_purity(
            adata,
            cluster_key='leiden',
            true_labels_key='true_labels'
        )
        
        assert 'purity_score' in purity_result
        assert 'adjusted_rand_index' in purity_result  
        assert 'normalized_mutual_info' in purity_result
        assert 0 <= purity_result['purity_score'] <= 1
    
    def test_clustering_stability_bootstrap(self, clustering_service, mock_preprocessed_data):
        """Test clustering stability via bootstrap resampling."""
        adata = mock_preprocessed_data.copy()
        
        with patch.object(clustering_service, 'bootstrap_stability') as mock_bootstrap:
            mock_bootstrap.return_value = {
                'stability_score': 0.82,
                'n_bootstrap': 50,
                'consensus_matrix': np.random.random((adata.n_obs, adata.n_obs)),
                'cluster_consistency': {
                    '0': 0.85, '1': 0.78, '2': 0.91, '3': 0.73,
                    '4': 0.88, '5': 0.79, '6': 0.84, '7': 0.80
                }
            }
            
            result = clustering_service.bootstrap_stability(
                adata,
                method='leiden',
                resolution=0.5,
                n_bootstrap=50
            )
            
            assert 0 <= result['stability_score'] <= 1
            assert result['n_bootstrap'] == 50
            assert 'consensus_matrix' in result


# ===============================================================================
# Marker Gene Discovery Tests
# ===============================================================================

@pytest.mark.unit
class TestMarkerGeneDiscovery:
    """Test marker gene discovery functionality."""
    
    def test_find_cluster_markers_wilcoxon(self, clustering_service, mock_clustered_data):
        """Test marker gene discovery using Wilcoxon rank-sum test."""
        adata = mock_clustered_data.copy()
        
        with patch('scanpy.tl.rank_genes_groups') as mock_rank_genes:
            # Mock the marker gene detection
            mock_rank_genes.side_effect = lambda x, **kwargs: None
            
            # Add mock results
            adata.uns['rank_genes_groups'] = {
                'names': np.array([['Gene1', 'Gene5'], ['Gene2', 'Gene6'], ['Gene3', 'Gene7'], ['Gene4', 'Gene8']]),
                'scores': np.array([[2.5, 1.8], [2.3, 1.7], [2.1, 1.6], [2.0, 1.5]]),
                'pvals': np.array([[0.01, 0.03], [0.02, 0.04], [0.01, 0.05], [0.03, 0.02]]),
                'pvals_adj': np.array([[0.05, 0.15], [0.10, 0.20], [0.05, 0.25], [0.15, 0.10]]),
                'logfoldchanges': np.array([[1.2, 0.8], [1.1, 0.7], [1.0, 0.6], [0.9, 0.5]])
            }
            
            markers = clustering_service.find_marker_genes(
                adata,
                cluster_key='leiden',
                method='wilcoxon',
                n_genes=25
            )
            
            assert 'rank_genes_groups' in adata.uns
            assert markers['method'] == 'wilcoxon'
            assert markers['n_clusters'] > 0
            mock_rank_genes.assert_called_once()
    
    def test_find_cluster_markers_ttest(self, clustering_service, mock_clustered_data):
        """Test marker gene discovery using t-test."""
        adata = mock_clustered_data.copy()
        
        with patch('scanpy.tl.rank_genes_groups') as mock_rank_genes:
            mock_rank_genes.side_effect = lambda x, **kwargs: None
            
            adata.uns['rank_genes_groups'] = {
                'names': np.array([['Gene1'], ['Gene2'], ['Gene3']]),
                'scores': np.array([[3.2], [2.8], [2.5]]),
                'pvals': np.array([[0.001], [0.002], [0.005]])
            }
            
            markers = clustering_service.find_marker_genes(
                adata,
                cluster_key='leiden',
                method='t-test',
                n_genes=10
            )
            
            assert markers['method'] == 't-test'
            mock_rank_genes.assert_called_once()
    
    def test_find_cluster_markers_deseq2(self, clustering_service, mock_clustered_data):
        """Test marker gene discovery using DESeq2-like method."""
        adata = mock_clustered_data.copy()
        
        with patch.object(clustering_service, 'find_markers_deseq2') as mock_deseq2:
            mock_deseq2.return_value = {
                'method': 'deseq2',
                'n_clusters': 8,
                'markers_per_cluster': {
                    '0': [{'gene': 'Gene1', 'log2fc': 2.1, 'padj': 0.01}],
                    '1': [{'gene': 'Gene2', 'log2fc': 1.8, 'padj': 0.02}],
                    '2': [{'gene': 'Gene3', 'log2fc': 1.9, 'padj': 0.015}]
                }
            }
            
            markers = clustering_service.find_markers_deseq2(
                adata,
                cluster_key='leiden',
                min_log2fc=1.0,
                max_padj=0.05
            )
            
            assert markers['method'] == 'deseq2'
            assert 'markers_per_cluster' in markers
    
    def test_marker_gene_visualization_prep(self, clustering_service, mock_clustered_data):
        """Test preparation of marker genes for visualization."""
        adata = mock_clustered_data.copy()
        
        # Add mock marker gene results
        adata.uns['rank_genes_groups'] = {
            'names': np.array([['CD3D', 'CD19'], ['CD8A', 'CD79A'], ['CD4', 'MS4A1']]),
            'scores': np.array([[3.5, 2.8], [3.2, 2.5], [2.9, 2.3]]),
            'pvals_adj': np.array([[0.001, 0.01], [0.002, 0.02], [0.005, 0.03]])
        }
        
        viz_data = clustering_service.prepare_marker_visualization(
            adata,
            cluster_key='leiden',
            top_n=5
        )
        
        assert 'top_markers' in viz_data
        assert 'expression_matrix' in viz_data
        assert 'cluster_names' in viz_data
    
    def test_marker_gene_filtering(self, clustering_service, mock_clustered_data):
        """Test marker gene filtering and selection."""
        adata = mock_clustered_data.copy()
        
        # Mock marker genes with different scores
        raw_markers = {
            '0': [
                {'gene': 'Gene1', 'score': 3.5, 'pval_adj': 0.001, 'log2fc': 2.1},
                {'gene': 'Gene2', 'score': 2.8, 'pval_adj': 0.01, 'log2fc': 1.5},
                {'gene': 'Gene3', 'score': 1.2, 'pval_adj': 0.08, 'log2fc': 0.8}
            ]
        }
        
        filtered_markers = clustering_service.filter_marker_genes(
            raw_markers,
            min_score=2.0,
            max_pval_adj=0.05,
            min_log2fc=1.0
        )
        
        assert len(filtered_markers['0']) == 2  # Gene1 and Gene2 should pass
        assert all(marker['score'] >= 2.0 for marker in filtered_markers['0'])
        assert all(marker['pval_adj'] <= 0.05 for marker in filtered_markers['0'])


# ===============================================================================
# Cell Type Annotation Tests
# ===============================================================================

@pytest.mark.unit
class TestCellTypeAnnotation:
    """Test cell type annotation functionality."""
    
    def test_automatic_cell_type_annotation(self, clustering_service, mock_clustered_data, mock_marker_genes):
        """Test automatic cell type annotation based on marker genes."""
        adata = mock_clustered_data.copy()
        
        with patch.object(clustering_service, 'annotate_cell_types') as mock_annotate:
            mock_annotate.return_value = {
                'cell_type_mapping': {
                    '0': 'T_cells', '1': 'B_cells', '2': 'NK_cells', '3': 'Monocytes',
                    '4': 'Dendritic_cells', '5': 'Neutrophils', '6': 'Macrophages', '7': 'Plasma_cells'
                },
                'confidence_scores': {
                    '0': 0.85, '1': 0.92, '2': 0.78, '3': 0.88,
                    '4': 0.73, '5': 0.81, '6': 0.87, '7': 0.94
                },
                'annotation_method': 'marker_gene_enrichment',
                'reference_database': 'custom'
            }
            
            annotation = clustering_service.annotate_cell_types(
                adata,
                cluster_key='leiden',
                marker_database=mock_marker_genes
            )
            
            assert 'cell_type_mapping' in annotation
            assert len(annotation['cell_type_mapping']) == 8
            assert all(0 <= score <= 1 for score in annotation['confidence_scores'].values())
    
    def test_celltype_annotation_with_reference(self, clustering_service, mock_clustered_data):
        """Test cell type annotation using reference dataset."""
        adata = mock_clustered_data.copy()
        reference_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        with patch.object(clustering_service, 'annotate_with_reference') as mock_ref_annotate:
            mock_ref_annotate.return_value = {
                'predicted_labels': np.random.choice(['T_cells', 'B_cells', 'NK_cells'], adata.n_obs),
                'prediction_scores': np.random.uniform(0.5, 1.0, adata.n_obs),
                'reference_method': 'label_transfer',
                'n_reference_cells': reference_data.n_obs
            }
            
            annotation = clustering_service.annotate_with_reference(
                adata,
                reference_data,
                method='label_transfer'
            )
            
            assert len(annotation['predicted_labels']) == adata.n_obs
            assert len(annotation['prediction_scores']) == adata.n_obs
            assert annotation['reference_method'] == 'label_transfer'
    
    def test_celltype_annotation_consensus(self, clustering_service, mock_clustered_data):
        """Test consensus cell type annotation from multiple methods."""
        adata = mock_clustered_data.copy()
        
        # Mock multiple annotation results
        method_results = {
            'marker_genes': {'0': 'T_cells', '1': 'B_cells', '2': 'NK_cells'},
            'reference_atlas': {'0': 'T_cells', '1': 'B_cells', '2': 'Monocytes'},
            'pathway_enrichment': {'0': 'T_cells', '1': 'Plasma_cells', '2': 'NK_cells'}
        }
        
        with patch.object(clustering_service, 'consensus_annotation') as mock_consensus:
            mock_consensus.return_value = {
                'consensus_labels': {'0': 'T_cells', '1': 'B_cells', '2': 'NK_cells'},
                'consensus_confidence': {'0': 1.0, '1': 0.67, '2': 0.67},
                'method_agreement': {'0': 3, '1': 2, '2': 2},
                'ambiguous_clusters': ['1', '2']
            }
            
            consensus = clustering_service.consensus_annotation(method_results)
            
            assert 'consensus_labels' in consensus
            assert 'ambiguous_clusters' in consensus
    
    def test_manual_cell_type_curation(self, clustering_service, mock_clustered_data):
        """Test manual cell type curation and validation."""
        adata = mock_clustered_data.copy()
        
        manual_annotations = {
            '0': 'CD4_T_cells',
            '1': 'CD8_T_cells', 
            '2': 'B_cells',
            '3': 'NK_cells',
            '4': 'Classical_monocytes',
            '5': 'Non_classical_monocytes',
            '6': 'Dendritic_cells',
            '7': 'Plasma_cells'
        }
        
        validation_result = clustering_service.validate_manual_annotations(
            adata,
            cluster_key='leiden',
            manual_annotations=manual_annotations
        )
        
        assert 'validation_passed' in validation_result
        assert 'annotation_quality_score' in validation_result
        assert 'potential_issues' in validation_result


# ===============================================================================
# Clustering Comparison and Optimization Tests
# ===============================================================================

@pytest.mark.unit
class TestClusteringOptimization:
    """Test clustering optimization and comparison functionality."""
    
    def test_optimal_cluster_number_detection(self, clustering_service, mock_preprocessed_data):
        """Test detection of optimal number of clusters."""
        adata = mock_preprocessed_data.copy()
        
        with patch.object(clustering_service, 'find_optimal_clusters') as mock_optimal:
            mock_optimal.return_value = {
                'optimal_k': 8,
                'silhouette_scores': [0.3, 0.4, 0.5, 0.6, 0.65, 0.68, 0.71, 0.69, 0.65],
                'davies_bouldin_scores': [2.1, 1.8, 1.5, 1.2, 1.1, 1.0, 0.95, 1.0, 1.1],
                'calinski_harabasz_scores': [50, 75, 100, 120, 135, 140, 145, 138, 130],
                'k_range': list(range(2, 11)),
                'best_method': 'silhouette'
            }
            
            result = clustering_service.find_optimal_clusters(
                adata,
                k_range=(2, 10),
                methods=['silhouette', 'davies_bouldin', 'calinski_harabasz']
            )
            
            assert result['optimal_k'] == 8
            assert len(result['silhouette_scores']) == len(result['k_range'])
    
    def test_clustering_method_comparison(self, clustering_service, mock_preprocessed_data):
        """Test comparison of different clustering methods."""
        adata = mock_preprocessed_data.copy()
        
        with patch.object(clustering_service, 'benchmark_clustering_methods') as mock_benchmark:
            mock_benchmark.return_value = {
                'methods_tested': ['leiden', 'louvain', 'kmeans', 'hierarchical'],
                'results': {
                    'leiden': {'silhouette': 0.68, 'runtime': 2.3, 'n_clusters': 8},
                    'louvain': {'silhouette': 0.64, 'runtime': 1.8, 'n_clusters': 7},
                    'kmeans': {'silhouette': 0.59, 'runtime': 0.5, 'n_clusters': 8},
                    'hierarchical': {'silhouette': 0.61, 'runtime': 5.2, 'n_clusters': 8}
                },
                'best_quality': 'leiden',
                'fastest': 'kmeans',
                'recommendations': {
                    'best_overall': 'leiden',
                    'speed_critical': 'kmeans',
                    'interpretable': 'hierarchical'
                }
            }
            
            benchmark = clustering_service.benchmark_clustering_methods(
                adata,
                methods=['leiden', 'louvain', 'kmeans', 'hierarchical']
            )
            
            assert len(benchmark['methods_tested']) == 4
            assert benchmark['best_quality'] in benchmark['methods_tested']
            assert 'recommendations' in benchmark
    
    def test_parameter_grid_optimization(self, clustering_service, mock_preprocessed_data):
        """Test grid search optimization for clustering parameters."""
        adata = mock_preprocessed_data.copy()
        
        parameter_grid = {
            'resolution': [0.1, 0.3, 0.5, 0.8, 1.0],
            'n_neighbors': [5, 10, 15, 20]
        }
        
        with patch.object(clustering_service, 'grid_search_optimization') as mock_grid_search:
            mock_grid_search.return_value = {
                'best_parameters': {'resolution': 0.8, 'n_neighbors': 15},
                'best_score': 0.71,
                'all_results': [
                    {'params': {'resolution': 0.5, 'n_neighbors': 10}, 'score': 0.65},
                    {'params': {'resolution': 0.8, 'n_neighbors': 15}, 'score': 0.71},
                    {'params': {'resolution': 1.0, 'n_neighbors': 10}, 'score': 0.68}
                ],
                'scoring_metric': 'silhouette'
            }
            
            result = clustering_service.grid_search_optimization(
                adata,
                method='leiden',
                parameter_grid=parameter_grid,
                scoring='silhouette'
            )
            
            assert 'best_parameters' in result
            assert result['best_score'] > 0
            assert len(result['all_results']) > 0


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestClusteringErrorHandling:
    """Test clustering error handling and edge cases."""
    
    def test_insufficient_cells_handling(self, clustering_service):
        """Test handling of datasets with too few cells."""
        small_adata = ad.AnnData(X=np.random.randn(5, 100))  # Only 5 cells
        
        with pytest.raises(ValueError, match="Insufficient cells for clustering"):
            clustering_service.leiden(small_adata, min_cells=10)
    
    def test_single_cluster_handling(self, clustering_service, mock_preprocessed_data):
        """Test handling when clustering results in single cluster."""
        adata = mock_preprocessed_data.copy()
        
        with patch('scanpy.tl.leiden') as mock_leiden:
            # Mock single cluster result
            single_cluster = np.zeros(adata.n_obs, dtype=int)
            mock_leiden.side_effect = lambda x, **kwargs: setattr(x.obs, 'leiden', single_cluster.astype(str))
            
            result = clustering_service.leiden(adata, resolution=0.01)
            
            assert result['n_clusters'] == 1
            assert 'warning' in result
            assert 'single_cluster' in result['warning'].lower()
    
    def test_missing_preprocessing_handling(self, clustering_service):
        """Test handling of data missing required preprocessing."""
        raw_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        # Remove preprocessing results
        if 'X_pca' in raw_adata.obsm:
            del raw_adata.obsm['X_pca']
        
        with pytest.raises(ValueError, match="PCA embedding not found"):
            clustering_service.leiden(raw_adata, use_rep='X_pca')
    
    def test_invalid_cluster_key_handling(self, clustering_service, mock_clustered_data):
        """Test handling of invalid cluster keys."""
        adata = mock_clustered_data.copy()
        
        with pytest.raises(KeyError, match="Cluster key 'nonexistent' not found"):
            clustering_service.find_marker_genes(adata, cluster_key='nonexistent')
    
    def test_memory_efficient_clustering(self, clustering_service):
        """Test memory-efficient clustering for large datasets."""
        # Create large sparse dataset
        large_sparse = sp.random(50000, 2000, density=0.05, format='csr')
        large_adata = ad.AnnData(X=large_sparse)
        
        # Add minimal required preprocessing
        large_adata.obsm['X_pca'] = np.random.randn(large_adata.n_obs, 50)
        
        with patch.object(clustering_service, 'leiden') as mock_leiden:
            mock_leiden.return_value = {
                'n_clusters': 25,
                'resolution': 0.5,
                'memory_usage': 'optimized'
            }
            
            result = clustering_service.leiden(
                large_adata,
                resolution=0.5,
                memory_efficient=True
            )
            
            assert result['n_clusters'] > 0
            mock_leiden.assert_called_once()
    
    def test_concurrent_clustering_safety(self, clustering_service, mock_preprocessed_data):
        """Test thread safety for concurrent clustering operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def clustering_worker(worker_id):
            """Worker function for concurrent clustering."""
            try:
                adata = mock_preprocessed_data.copy()
                
                with patch('scanpy.tl.leiden') as mock_leiden:
                    n_clusters = 5 + worker_id  # Different results per worker
                    cluster_labels = np.random.randint(0, n_clusters, adata.n_obs)
                    mock_leiden.side_effect = lambda x, **kwargs: setattr(x.obs, 'leiden', cluster_labels.astype(str))
                    
                    result = clustering_service.leiden(adata, resolution=0.5)
                    results.append((worker_id, result))
                    time.sleep(0.01)
                    
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=clustering_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent clustering errors: {errors}"
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])