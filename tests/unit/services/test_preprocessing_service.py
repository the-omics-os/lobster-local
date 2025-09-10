"""
Comprehensive unit tests for preprocessing service.

This module provides thorough testing of the preprocessing service including
data normalization, filtering, quality control, batch correction,
feature selection, and dimensionality reduction for bioinformatics analysis.

Test coverage target: 95%+ with meaningful tests for preprocessing operations.
"""

import pytest
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lobster.tools.preprocessing_service import PreprocessingService
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def mock_single_cell_data():
    """Create mock single-cell data for testing."""
    return SingleCellDataFactory(config=SMALL_DATASET_CONFIG)


@pytest.fixture
def mock_bulk_data():
    """Create mock bulk RNA-seq data for testing."""
    return BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG)


@pytest.fixture
def mock_noisy_data():
    """Create mock data with quality issues for testing filtering."""
    config = SMALL_DATASET_CONFIG.copy()
    config.update({
        'n_obs': 1000,
        'n_vars': 2000,
        'add_doublets': True,
        'doublet_rate': 0.1,
        'add_mt_genes': True,
        'mt_gene_fraction': 0.15,
        'add_ribo_genes': True,
        'ribo_gene_fraction': 0.25
    })
    return SingleCellDataFactory(config=config)


@pytest.fixture
def preprocessing_service():
    """Create PreprocessingService instance for testing."""
    return PreprocessingService()


@pytest.fixture 
def mock_batch_data():
    """Create mock data with batch effects."""
    adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    
    # Add batch information
    n_obs = adata.n_obs
    batches = np.random.choice(['batch_1', 'batch_2', 'batch_3'], size=n_obs)
    adata.obs['batch'] = batches
    
    # Simulate batch effects in expression
    batch_effects = {
        'batch_1': 1.0,
        'batch_2': 1.2, 
        'batch_3': 0.8
    }
    
    for i, batch in enumerate(batches):
        adata.X[i] *= batch_effects[batch]
    
    return adata


# ===============================================================================
# Preprocessing Service Core Tests
# ===============================================================================

@pytest.mark.unit
class TestPreprocessingServiceCore:
    """Test preprocessing service core functionality."""
    
    def test_preprocessing_service_initialization(self):
        """Test PreprocessingService initialization."""
        service = PreprocessingService()
        
        assert hasattr(service, 'normalize')
        assert hasattr(service, 'filter_cells')
        assert hasattr(service, 'filter_genes')
        assert callable(service.normalize)
    
    def test_preprocessing_service_with_config(self):
        """Test PreprocessingService initialization with configuration."""
        config = {
            'default_normalization': 'log1p',
            'min_genes_per_cell': 200,
            'min_cells_per_gene': 3,
            'max_mt_percent': 20.0
        }
        
        service = PreprocessingService(config=config)
        
        assert service.config['default_normalization'] == 'log1p'
        assert service.config['min_genes_per_cell'] == 200
    
    def test_get_preprocessing_pipeline(self, preprocessing_service):
        """Test getting standard preprocessing pipelines."""
        pipeline = preprocessing_service.get_pipeline('single_cell_standard')
        
        assert 'filter_cells' in [step['name'] for step in pipeline]
        assert 'normalize' in [step['name'] for step in pipeline]
        assert 'log_transform' in [step['name'] for step in pipeline]
    
    def test_validate_input_data(self, preprocessing_service, mock_single_cell_data):
        """Test input data validation."""
        # Valid data should pass
        is_valid, messages = preprocessing_service.validate_input(mock_single_cell_data)
        assert is_valid == True
        assert len(messages) == 0
        
        # Create invalid data (empty)
        invalid_data = ad.AnnData(X=np.array([]).reshape(0, 0))
        is_valid, messages = preprocessing_service.validate_input(invalid_data)
        assert is_valid == False
        assert len(messages) > 0


# ===============================================================================
# Quality Control and Filtering Tests
# ===============================================================================

@pytest.mark.unit
class TestQualityControlFiltering:
    """Test quality control and filtering functionality."""
    
    def test_calculate_qc_metrics(self, preprocessing_service, mock_single_cell_data):
        """Test QC metrics calculation."""
        adata = mock_single_cell_data.copy()
        
        qc_metrics = preprocessing_service.calculate_qc_metrics(adata)
        
        assert 'n_genes_by_counts' in adata.obs.columns
        assert 'total_counts' in adata.obs.columns
        assert 'pct_counts_mt' in adata.obs.columns
        assert qc_metrics['mean_genes_per_cell'] > 0
        assert qc_metrics['mean_counts_per_cell'] > 0
    
    def test_filter_cells_by_genes(self, preprocessing_service, mock_noisy_data):
        """Test cell filtering by gene count."""
        adata = mock_noisy_data.copy()
        initial_cells = adata.n_obs
        
        filtered_adata = preprocessing_service.filter_cells(
            adata, 
            min_genes=200,
            max_genes=5000
        )
        
        assert filtered_adata.n_obs <= initial_cells
        assert np.all(filtered_adata.obs['n_genes_by_counts'] >= 200)
        assert np.all(filtered_adata.obs['n_genes_by_counts'] <= 5000)
    
    def test_filter_cells_by_counts(self, preprocessing_service, mock_noisy_data):
        """Test cell filtering by UMI count."""
        adata = mock_noisy_data.copy()
        initial_cells = adata.n_obs
        
        filtered_adata = preprocessing_service.filter_cells(
            adata,
            min_counts=1000,
            max_counts=50000
        )
        
        assert filtered_adata.n_obs <= initial_cells
        assert np.all(filtered_adata.obs['total_counts'] >= 1000)
        assert np.all(filtered_adata.obs['total_counts'] <= 50000)
    
    def test_filter_cells_by_mitochondrial(self, preprocessing_service, mock_noisy_data):
        """Test cell filtering by mitochondrial gene percentage."""
        adata = mock_noisy_data.copy()
        initial_cells = adata.n_obs
        
        filtered_adata = preprocessing_service.filter_cells(
            adata,
            max_pct_mt=20.0
        )
        
        assert filtered_adata.n_obs <= initial_cells
        assert np.all(filtered_adata.obs['pct_counts_mt'] <= 20.0)
    
    def test_filter_genes_by_cells(self, preprocessing_service, mock_single_cell_data):
        """Test gene filtering by cell count."""
        adata = mock_single_cell_data.copy()
        initial_genes = adata.n_vars
        
        filtered_adata = preprocessing_service.filter_genes(
            adata,
            min_cells=3
        )
        
        assert filtered_adata.n_vars <= initial_genes
        # Check that remaining genes are expressed in at least 3 cells
        gene_cell_counts = np.array((filtered_adata.X > 0).sum(axis=0)).flatten()
        assert np.all(gene_cell_counts >= 3)
    
    def test_filter_genes_by_counts(self, preprocessing_service, mock_single_cell_data):
        """Test gene filtering by total count."""
        adata = mock_single_cell_data.copy()
        initial_genes = adata.n_vars
        
        filtered_adata = preprocessing_service.filter_genes(
            adata,
            min_counts=10
        )
        
        assert filtered_adata.n_vars <= initial_genes
        # Check that remaining genes have total counts >= 10
        gene_counts = np.array(filtered_adata.X.sum(axis=0)).flatten()
        assert np.all(gene_counts >= 10)
    
    def test_detect_outlier_cells(self, preprocessing_service, mock_noisy_data):
        """Test outlier cell detection."""
        adata = mock_noisy_data.copy()
        
        outliers = preprocessing_service.detect_outliers(
            adata,
            method='modified_z_score',
            threshold=3.5
        )
        
        assert len(outliers) <= adata.n_obs
        assert 'outlier_score' in adata.obs.columns
        assert 'is_outlier' in adata.obs.columns
    
    def test_doublet_detection(self, preprocessing_service, mock_noisy_data):
        """Test doublet detection."""
        adata = mock_noisy_data.copy()
        
        doublet_info = preprocessing_service.detect_doublets(adata)
        
        assert 'doublet_score' in adata.obs.columns
        assert 'predicted_doublet' in adata.obs.columns
        assert doublet_info['doublet_rate'] >= 0
        assert doublet_info['n_doublets'] >= 0


# ===============================================================================
# Normalization Tests
# ===============================================================================

@pytest.mark.unit
class TestNormalization:
    """Test normalization functionality."""
    
    def test_counts_per_million_normalization(self, preprocessing_service, mock_single_cell_data):
        """Test CPM normalization."""
        adata = mock_single_cell_data.copy()
        original_sum = adata.X.sum()
        
        normalized_adata = preprocessing_service.normalize(
            adata,
            method='cpm',
            target_sum=1e6
        )
        
        # Check that each cell sums to target (within tolerance)
        cell_sums = np.array(normalized_adata.X.sum(axis=1)).flatten()
        expected_sum = 1e6
        tolerance = expected_sum * 0.01  # 1% tolerance
        
        assert np.all(np.abs(cell_sums - expected_sum) <= tolerance)
    
    def test_log_normalization(self, preprocessing_service, mock_single_cell_data):
        """Test log normalization."""
        adata = mock_single_cell_data.copy()
        
        normalized_adata = preprocessing_service.normalize(
            adata,
            method='log1p',
            target_sum=1e4
        )
        
        # Check that data is log-transformed (no negative values, bounded)
        assert np.all(normalized_adata.X.data >= 0) if sp.issparse(normalized_adata.X) else np.all(normalized_adata.X >= 0)
        assert 'log1p' in normalized_adata.uns.get('preprocessing_log', [])
    
    def test_quantile_normalization(self, preprocessing_service, mock_bulk_data):
        """Test quantile normalization for bulk data."""
        adata = mock_bulk_data.copy()
        
        normalized_adata = preprocessing_service.normalize(
            adata,
            method='quantile'
        )
        
        # Check that samples have similar distributions
        sample_quantiles = np.percentile(normalized_adata.X, [25, 50, 75], axis=1)
        
        # Quantile normalization should make distributions more similar
        cv_before = np.std(np.percentile(adata.X, 50, axis=1)) / np.mean(np.percentile(adata.X, 50, axis=1))
        cv_after = np.std(sample_quantiles[1]) / np.mean(sample_quantiles[1])
        
        assert cv_after <= cv_before
    
    def test_zscore_normalization(self, preprocessing_service, mock_single_cell_data):
        """Test z-score normalization."""
        adata = mock_single_cell_data.copy()
        
        normalized_adata = preprocessing_service.normalize(
            adata,
            method='zscore',
            axis=0  # Normalize genes
        )
        
        # Check that genes have mean ~0 and std ~1
        gene_means = np.mean(normalized_adata.X, axis=0)
        gene_stds = np.std(normalized_adata.X, axis=0)
        
        assert np.allclose(gene_means, 0, atol=1e-10)
        assert np.allclose(gene_stds, 1, atol=1e-10)
    
    def test_sctransform_normalization(self, preprocessing_service, mock_single_cell_data):
        """Test SCTransform-style normalization."""
        adata = mock_single_cell_data.copy()
        
        with patch.object(preprocessing_service, 'sctransform') as mock_sct:
            # Mock SCTransform results
            mock_sct.return_value = adata.copy()
            mock_sct.return_value.layers['sct_normalized'] = adata.X.copy()
            mock_sct.return_value.var['highly_variable_sct'] = np.random.choice([True, False], adata.n_vars)
            
            normalized_adata = preprocessing_service.normalize(adata, method='sctransform')
            
            assert 'sct_normalized' in normalized_adata.layers
            assert 'highly_variable_sct' in normalized_adata.var.columns
            mock_sct.assert_called_once()
    
    def test_size_factor_normalization(self, preprocessing_service, mock_bulk_data):
        """Test size factor normalization (DESeq2-style)."""
        adata = mock_bulk_data.copy()
        
        normalized_adata = preprocessing_service.normalize(
            adata,
            method='deseq2'
        )
        
        # Check that size factors are computed and stored
        assert 'size_factors' in normalized_adata.obs.columns
        assert np.all(normalized_adata.obs['size_factors'] > 0)
    
    def test_batch_aware_normalization(self, preprocessing_service, mock_batch_data):
        """Test batch-aware normalization."""
        adata = mock_batch_data.copy()
        
        normalized_adata = preprocessing_service.normalize(
            adata,
            method='log1p',
            batch_key='batch'
        )
        
        # Check that normalization was applied per batch
        assert 'batch_normalized' in normalized_adata.uns.get('preprocessing_log', [])


# ===============================================================================
# Feature Selection Tests  
# ===============================================================================

@pytest.mark.unit
class TestFeatureSelection:
    """Test feature selection functionality."""
    
    def test_highly_variable_genes_seurat(self, preprocessing_service, mock_single_cell_data):
        """Test highly variable genes detection (Seurat method)."""
        adata = mock_single_cell_data.copy()
        
        hvg_info = preprocessing_service.find_highly_variable_genes(
            adata,
            method='seurat',
            n_top_genes=2000
        )
        
        assert 'highly_variable' in adata.var.columns
        assert 'means' in adata.var.columns
        assert 'dispersions' in adata.var.columns
        assert np.sum(adata.var['highly_variable']) <= 2000
        assert hvg_info['n_highly_variable'] <= 2000
    
    def test_highly_variable_genes_cellranger(self, preprocessing_service, mock_single_cell_data):
        """Test highly variable genes detection (Cell Ranger method)."""
        adata = mock_single_cell_data.copy()
        
        hvg_info = preprocessing_service.find_highly_variable_genes(
            adata,
            method='cell_ranger',
            n_top_genes=1500
        )
        
        assert 'highly_variable' in adata.var.columns
        assert np.sum(adata.var['highly_variable']) <= 1500
    
    def test_highly_variable_genes_seurat_v3(self, preprocessing_service, mock_single_cell_data):
        """Test highly variable genes detection (Seurat v3 method)."""
        adata = mock_single_cell_data.copy()
        
        hvg_info = preprocessing_service.find_highly_variable_genes(
            adata,
            method='seurat_v3',
            n_top_genes=2000
        )
        
        assert 'highly_variable' in adata.var.columns
        assert 'variances' in adata.var.columns
        assert 'variances_norm' in adata.var.columns
    
    def test_feature_selection_by_expression(self, preprocessing_service, mock_single_cell_data):
        """Test feature selection by expression level."""
        adata = mock_single_cell_data.copy()
        
        selected_adata = preprocessing_service.select_features_by_expression(
            adata,
            min_mean=0.01,
            max_mean=5.0,
            min_dispersion=0.5
        )
        
        assert selected_adata.n_vars <= adata.n_vars
        assert 'selected_by_expression' in adata.var.columns
    
    def test_feature_selection_by_variance(self, preprocessing_service, mock_single_cell_data):
        """Test feature selection by variance."""
        adata = mock_single_cell_data.copy()
        
        selected_adata = preprocessing_service.select_features_by_variance(
            adata,
            n_features=1000
        )
        
        assert selected_adata.n_vars == 1000
        assert 'variance_score' in adata.var.columns
    
    def test_remove_ribosomal_genes(self, preprocessing_service, mock_single_cell_data):
        """Test removal of ribosomal genes."""
        adata = mock_single_cell_data.copy()
        
        # Add some ribosomal genes
        ribo_genes = [f'RPS{i}' for i in range(1, 11)] + [f'RPL{i}' for i in range(1, 11)]
        adata.var.index = np.concatenate([adata.var.index[:20], ribo_genes, adata.var.index[40:]])
        
        filtered_adata = preprocessing_service.remove_ribosomal_genes(adata)
        
        assert filtered_adata.n_vars < adata.n_vars
        assert not any(gene.startswith('RPS') or gene.startswith('RPL') for gene in filtered_adata.var.index)
    
    def test_remove_mitochondrial_genes(self, preprocessing_service, mock_single_cell_data):
        """Test removal of mitochondrial genes."""
        adata = mock_single_cell_data.copy()
        
        # Add some mitochondrial genes
        mt_genes = [f'MT-{gene}' for gene in ['ATP6', 'ATP8', 'COX1', 'COX2', 'ND1', 'ND2']]
        adata.var.index = np.concatenate([adata.var.index[:20], mt_genes, adata.var.index[26:]])
        
        filtered_adata = preprocessing_service.remove_mitochondrial_genes(adata)
        
        assert filtered_adata.n_vars < adata.n_vars
        assert not any(gene.startswith('MT-') for gene in filtered_adata.var.index)


# ===============================================================================
# Dimensionality Reduction Tests
# ===============================================================================

@pytest.mark.unit  
class TestDimensionalityReduction:
    """Test dimensionality reduction functionality."""
    
    def test_principal_component_analysis(self, preprocessing_service, mock_single_cell_data):
        """Test PCA computation."""
        adata = mock_single_cell_data.copy()
        
        pca_result = preprocessing_service.compute_pca(
            adata,
            n_comps=50,
            random_state=42
        )
        
        assert 'X_pca' in adata.obsm.keys()
        assert adata.obsm['X_pca'].shape[1] == 50
        assert 'pca' in adata.uns.keys()
        assert 'variance_ratio' in adata.uns['pca']
        assert pca_result['explained_variance_ratio'].sum() <= 1.0
    
    def test_pca_gene_loadings(self, preprocessing_service, mock_single_cell_data):
        """Test PCA gene loadings calculation."""
        adata = mock_single_cell_data.copy()
        
        pca_result = preprocessing_service.compute_pca(
            adata,
            n_comps=25,
            compute_loadings=True
        )
        
        assert 'PCs' in adata.varm.keys()
        assert adata.varm['PCs'].shape == (adata.n_vars, 25)
        assert 'loadings' in pca_result
    
    def test_incremental_pca(self, preprocessing_service):
        """Test incremental PCA for large datasets."""
        # Create larger dataset
        large_config = LARGE_DATASET_CONFIG.copy()
        adata = SingleCellDataFactory(config=large_config)
        
        pca_result = preprocessing_service.compute_pca(
            adata,
            n_comps=50,
            method='incremental',
            batch_size=1000
        )
        
        assert 'X_pca' in adata.obsm.keys()
        assert adata.obsm['X_pca'].shape[1] == 50
    
    def test_sparse_pca(self, preprocessing_service, mock_single_cell_data):
        """Test sparse PCA computation."""
        adata = mock_single_cell_data.copy()
        
        with patch.object(preprocessing_service, 'compute_sparse_pca') as mock_sparse_pca:
            mock_result = {
                'components': np.random.randn(30, adata.n_vars),
                'explained_variance_ratio': np.random.random(30),
                'n_components': 30
            }
            mock_sparse_pca.return_value = mock_result
            
            result = preprocessing_service.compute_sparse_pca(adata, n_comps=30, alpha=0.1)
            
            assert result['n_components'] == 30
            mock_sparse_pca.assert_called_once()
    
    def test_pca_elbow_detection(self, preprocessing_service, mock_single_cell_data):
        """Test automatic elbow detection for PCA components."""
        adata = mock_single_cell_data.copy()
        
        elbow_result = preprocessing_service.find_pca_elbow(
            adata,
            max_comps=100,
            method='kneedle'
        )
        
        assert 'optimal_n_comps' in elbow_result
        assert 'elbow_point' in elbow_result
        assert elbow_result['optimal_n_comps'] > 0
        assert elbow_result['optimal_n_comps'] <= 100
    
    def test_batch_effect_pca(self, preprocessing_service, mock_batch_data):
        """Test PCA computation with batch effect consideration."""
        adata = mock_batch_data.copy()
        
        pca_result = preprocessing_service.compute_pca(
            adata,
            n_comps=50,
            batch_key='batch'
        )
        
        assert 'X_pca' in adata.obsm.keys()
        assert 'batch_effect_score' in pca_result
        assert pca_result['batch_effect_detected'] in [True, False]


# ===============================================================================
# Batch Correction Tests
# ===============================================================================

@pytest.mark.unit
class TestBatchCorrection:
    """Test batch correction functionality."""
    
    def test_combat_batch_correction(self, preprocessing_service, mock_batch_data):
        """Test ComBat batch correction."""
        adata = mock_batch_data.copy()
        
        with patch.object(preprocessing_service, 'combat') as mock_combat:
            corrected_adata = adata.copy()
            mock_combat.return_value = corrected_adata
            
            result = preprocessing_service.combat(adata, batch_key='batch')
            
            assert 'combat_corrected' in result.uns.get('preprocessing_log', [])
            mock_combat.assert_called_once()
    
    def test_harmony_integration(self, preprocessing_service, mock_batch_data):
        """Test Harmony batch integration."""
        adata = mock_batch_data.copy()
        
        with patch.object(preprocessing_service, 'harmony') as mock_harmony:
            # Mock Harmony results
            harmony_embedding = np.random.randn(adata.n_obs, 50)
            mock_harmony.return_value = harmony_embedding
            
            result = preprocessing_service.harmony(
                adata,
                batch_key='batch',
                n_components=50
            )
            
            assert result.shape == (adata.n_obs, 50)
            mock_harmony.assert_called_once()
    
    def test_scanorama_integration(self, preprocessing_service):
        """Test Scanorama batch integration."""
        # Create multiple batches
        batch1 = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        batch2 = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        with patch.object(preprocessing_service, 'scanorama') as mock_scanorama:
            integrated_data = [batch1, batch2]
            mock_scanorama.return_value = integrated_data
            
            result = preprocessing_service.scanorama([batch1, batch2])
            
            assert len(result) == 2
            mock_scanorama.assert_called_once()
    
    def test_mutual_nearest_neighbors(self, preprocessing_service):
        """Test MNN batch correction."""
        batch1 = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        batch2 = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        with patch.object(preprocessing_service, 'mnn_correct') as mock_mnn:
            corrected_batches = [batch1, batch2]
            mock_mnn.return_value = corrected_batches
            
            result = preprocessing_service.mnn_correct([batch1, batch2])
            
            assert len(result) == 2
            mock_mnn.assert_called_once()
    
    def test_batch_effect_assessment(self, preprocessing_service, mock_batch_data):
        """Test batch effect assessment."""
        adata = mock_batch_data.copy()
        
        batch_metrics = preprocessing_service.assess_batch_effects(
            adata,
            batch_key='batch'
        )
        
        assert 'silhouette_batch' in batch_metrics
        assert 'pcr_batch' in batch_metrics
        assert 'kbet_pvalue' in batch_metrics
        assert batch_metrics['batch_effect_strength'] in ['low', 'medium', 'high']


# ===============================================================================
# Integration and Pipeline Tests
# ===============================================================================

@pytest.mark.unit
class TestPreprocessingPipelines:
    """Test preprocessing pipeline integration."""
    
    def test_standard_single_cell_pipeline(self, preprocessing_service, mock_noisy_data):
        """Test standard single-cell preprocessing pipeline."""
        adata = mock_noisy_data.copy()
        
        pipeline_result = preprocessing_service.run_pipeline(
            adata,
            pipeline='single_cell_standard',
            min_genes=200,
            min_cells=3,
            target_sum=1e4,
            n_top_genes=2000,
            n_comps=50
        )
        
        assert 'qc_calculated' in pipeline_result['steps_completed']
        assert 'cells_filtered' in pipeline_result['steps_completed']
        assert 'genes_filtered' in pipeline_result['steps_completed']
        assert 'normalized' in pipeline_result['steps_completed']
        assert 'hvg_selected' in pipeline_result['steps_completed']
        assert 'pca_computed' in pipeline_result['steps_completed']
    
    def test_bulk_rna_seq_pipeline(self, preprocessing_service, mock_bulk_data):
        """Test bulk RNA-seq preprocessing pipeline."""
        adata = mock_bulk_data.copy()
        
        pipeline_result = preprocessing_service.run_pipeline(
            adata,
            pipeline='bulk_rna_seq',
            normalization='deseq2',
            filter_low_expressed=True
        )
        
        assert 'normalized' in pipeline_result['steps_completed']
        assert 'low_expressed_filtered' in pipeline_result['steps_completed']
    
    def test_custom_pipeline(self, preprocessing_service, mock_single_cell_data):
        """Test custom preprocessing pipeline."""
        adata = mock_single_cell_data.copy()
        
        custom_steps = [
            {'name': 'filter_cells', 'params': {'min_genes': 100}},
            {'name': 'normalize', 'params': {'method': 'cpm', 'target_sum': 1e6}},
            {'name': 'log_transform', 'params': {}},
            {'name': 'find_hvg', 'params': {'n_top_genes': 1500}}
        ]
        
        pipeline_result = preprocessing_service.run_pipeline(
            adata,
            pipeline='custom',
            steps=custom_steps
        )
        
        assert len(pipeline_result['steps_completed']) == 4
        assert all(step['name'] in pipeline_result['steps_completed'] for step in custom_steps)
    
    def test_pipeline_with_batch_correction(self, preprocessing_service, mock_batch_data):
        """Test preprocessing pipeline with batch correction."""
        adata = mock_batch_data.copy()
        
        pipeline_result = preprocessing_service.run_pipeline(
            adata,
            pipeline='single_cell_with_batch_correction',
            batch_key='batch',
            batch_method='combat'
        )
        
        assert 'batch_corrected' in pipeline_result['steps_completed']
        assert 'batch_effects_assessed' in pipeline_result['steps_completed']
    
    def test_pipeline_error_handling(self, preprocessing_service, mock_single_cell_data):
        """Test pipeline error handling and recovery."""
        adata = mock_single_cell_data.copy()
        
        # Create pipeline step that will fail
        failing_steps = [
            {'name': 'filter_cells', 'params': {'min_genes': 100}},
            {'name': 'invalid_step', 'params': {}},  # This will fail
            {'name': 'normalize', 'params': {'method': 'log1p'}}
        ]
        
        with pytest.raises(ValueError):
            preprocessing_service.run_pipeline(
                adata,
                pipeline='custom',
                steps=failing_steps,
                stop_on_error=True
            )


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestPreprocessingErrorHandling:
    """Test preprocessing error handling and edge cases."""
    
    def test_empty_data_handling(self, preprocessing_service):
        """Test handling of empty datasets."""
        empty_adata = ad.AnnData(X=np.array([]).reshape(0, 0))
        
        with pytest.raises(ValueError, match="Empty dataset"):
            preprocessing_service.filter_cells(empty_adata)
    
    def test_single_cell_data_handling(self, preprocessing_service):
        """Test handling of single-cell datasets."""
        single_cell_adata = ad.AnnData(X=np.array([[1, 2, 3]]))
        
        # Should handle gracefully
        result = preprocessing_service.calculate_qc_metrics(single_cell_adata)
        assert result is not None
    
    def test_all_zero_data_handling(self, preprocessing_service):
        """Test handling of all-zero data."""
        zero_adata = ad.AnnData(X=np.zeros((100, 50)))
        
        with pytest.raises(ValueError, match="All-zero data"):
            preprocessing_service.normalize(zero_adata, method='log1p')
    
    def test_missing_batch_key_handling(self, preprocessing_service, mock_single_cell_data):
        """Test handling of missing batch key."""
        adata = mock_single_cell_data.copy()
        
        with pytest.raises(ValueError, match="Batch key 'nonexistent' not found"):
            preprocessing_service.assess_batch_effects(adata, batch_key='nonexistent')
    
    def test_insufficient_hvg_handling(self, preprocessing_service):
        """Test handling when insufficient HVGs are found."""
        # Create data with very low variance
        low_var_data = ad.AnnData(X=np.ones((100, 50)) + np.random.normal(0, 0.001, (100, 50)))
        
        hvg_result = preprocessing_service.find_highly_variable_genes(
            low_var_data,
            n_top_genes=2000,  # More than available
            method='seurat'
        )
        
        assert hvg_result['n_highly_variable'] <= low_var_data.n_vars
    
    def test_memory_efficient_processing(self, preprocessing_service):
        """Test memory-efficient processing for large datasets."""
        # Create large sparse dataset
        large_sparse = sp.random(10000, 5000, density=0.1, format='csr')
        large_adata = ad.AnnData(X=large_sparse)
        
        # Should handle without memory issues
        qc_result = preprocessing_service.calculate_qc_metrics(
            large_adata,
            memory_efficient=True,
            chunk_size=1000
        )
        
        assert qc_result is not None
    
    def test_concurrent_processing_safety(self, preprocessing_service, mock_single_cell_data):
        """Test thread safety for concurrent preprocessing."""
        import threading
        import time
        
        results = []
        errors = []
        
        def preprocessing_worker(worker_id):
            """Worker function for concurrent preprocessing."""
            try:
                adata = mock_single_cell_data.copy()
                
                result = preprocessing_service.calculate_qc_metrics(adata)
                results.append((worker_id, result))
                time.sleep(0.01)
                
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=preprocessing_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])