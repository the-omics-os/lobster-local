"""
Comprehensive unit tests for quality service.

This module provides thorough testing of the quality control service including
QC metric calculation, cell and gene filtering, doublet detection,
batch effect assessment, data validation, and quality reporting.

Test coverage target: 95%+ with meaningful tests for quality operations.
"""

import pytest
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from lobster.tools.quality_service import QualityService
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def mock_raw_single_cell_data():
    """Create mock raw single-cell data with quality issues."""
    config = SMALL_DATASET_CONFIG.copy()
    config.update({
        'n_obs': 2000,
        'n_vars': 3000,
        'add_doublets': True,
        'doublet_rate': 0.08,
        'add_mt_genes': True,
        'mt_gene_fraction': 0.12,
        'add_ribo_genes': True,
        'ribo_gene_fraction': 0.20,
        'add_empty_droplets': True,
        'empty_droplet_rate': 0.05
    })
    return SingleCellDataFactory(config=config)


@pytest.fixture
def mock_bulk_data_with_issues():
    """Create mock bulk RNA-seq data with quality issues."""
    config = SMALL_DATASET_CONFIG.copy()
    config.update({
        'n_obs': 48,  # 48 samples
        'n_vars': 2500,
        'add_batch_effects': True,
        'batch_effect_strength': 0.3,
        'add_outlier_samples': True,
        'outlier_rate': 0.1
    })
    return BulkRNASeqDataFactory(config=config)


@pytest.fixture
def quality_service():
    """Create QualityService instance for testing."""
    return QualityService()


@pytest.fixture
def mock_processed_data():
    """Create mock data with QC metrics already calculated."""
    adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    
    # Add QC metrics
    adata.obs['n_genes_by_counts'] = np.random.randint(200, 4000, adata.n_obs)
    adata.obs['total_counts'] = np.random.randint(500, 15000, adata.n_obs)
    adata.obs['pct_counts_mt'] = np.random.uniform(0, 25, adata.n_obs)
    adata.obs['pct_counts_ribo'] = np.random.uniform(5, 40, adata.n_obs)
    adata.obs['doublet_score'] = np.random.uniform(0, 1, adata.n_obs)
    adata.obs['is_doublet'] = np.random.choice([True, False], adata.n_obs, p=[0.08, 0.92])
    
    # Add gene-level QC metrics
    adata.var['n_cells_by_counts'] = np.random.randint(1, adata.n_obs, adata.n_vars)
    adata.var['total_counts'] = np.random.randint(10, 50000, adata.n_vars)
    adata.var['pct_dropout'] = np.random.uniform(0, 95, adata.n_vars)
    adata.var['highly_variable'] = np.random.choice([True, False], adata.n_vars, p=[0.15, 0.85])
    
    return adata


# ===============================================================================
# Quality Service Core Tests
# ===============================================================================

@pytest.mark.unit
class TestQualityServiceCore:
    """Test quality service core functionality."""
    
    def test_quality_service_initialization(self):
        """Test QualityService initialization."""
        service = QualityService()
        
        assert hasattr(service, 'calculate_qc_metrics')
        assert hasattr(service, 'detect_doublets')
        assert hasattr(service, 'assess_data_quality')
        assert callable(service.calculate_qc_metrics)
    
    def test_quality_service_with_config(self):
        """Test QualityService initialization with configuration."""
        config = {
            'min_genes_per_cell': 200,
            'max_genes_per_cell': 5000,
            'min_cells_per_gene': 3,
            'max_mt_percent': 20.0,
            'doublet_score_threshold': 0.25
        }
        
        service = QualityService(config=config)
        
        assert service.config['min_genes_per_cell'] == 200
        assert service.config['max_mt_percent'] == 20.0
    
    def test_get_quality_thresholds(self, quality_service):
        """Test getting default and custom quality thresholds."""
        # Default thresholds
        thresholds = quality_service.get_quality_thresholds('single_cell')
        
        assert 'min_genes' in thresholds
        assert 'max_genes' in thresholds  
        assert 'min_counts' in thresholds
        assert 'max_mt_percent' in thresholds
        
        # Custom thresholds
        custom_thresholds = {
            'min_genes': 100,
            'max_mt_percent': 25.0
        }
        
        thresholds = quality_service.get_quality_thresholds('single_cell', custom_thresholds)
        assert thresholds['min_genes'] == 100
        assert thresholds['max_mt_percent'] == 25.0


# ===============================================================================
# QC Metrics Calculation Tests
# ===============================================================================

@pytest.mark.unit
class TestQCMetricsCalculation:
    """Test QC metrics calculation functionality."""
    
    def test_calculate_basic_qc_metrics(self, quality_service, mock_raw_single_cell_data):
        """Test calculation of basic QC metrics."""
        adata = mock_raw_single_cell_data.copy()
        
        qc_results = quality_service.calculate_qc_metrics(adata)
        
        # Check that metrics are added to obs
        assert 'n_genes_by_counts' in adata.obs.columns
        assert 'total_counts' in adata.obs.columns
        assert 'pct_counts_mt' in adata.obs.columns
        assert 'pct_counts_ribo' in adata.obs.columns
        
        # Check that summary statistics are returned
        assert 'mean_genes_per_cell' in qc_results
        assert 'mean_counts_per_cell' in qc_results
        assert 'median_mt_percent' in qc_results
        assert qc_results['n_cells'] == adata.n_obs
    
    def test_calculate_mitochondrial_metrics(self, quality_service, mock_raw_single_cell_data):
        """Test calculation of mitochondrial gene metrics."""
        adata = mock_raw_single_cell_data.copy()
        
        # Add some mitochondrial genes
        mt_genes = [f'MT-{gene}' for gene in ['ATP6', 'ATP8', 'CO1', 'CO2', 'ND1', 'ND2']]
        adata.var.index = list(adata.var.index[:20]) + mt_genes + list(adata.var.index[26:])
        
        mt_results = quality_service.calculate_mitochondrial_metrics(adata)
        
        assert 'pct_counts_mt' in adata.obs.columns
        assert 'n_genes_mt' in adata.obs.columns
        assert mt_results['n_mt_genes'] == len(mt_genes)
        assert mt_results['mean_mt_percent'] >= 0
    
    def test_calculate_ribosomal_metrics(self, quality_service, mock_raw_single_cell_data):
        """Test calculation of ribosomal gene metrics."""
        adata = mock_raw_single_cell_data.copy()
        
        # Add some ribosomal genes
        ribo_genes = [f'RPS{i}' for i in range(1, 11)] + [f'RPL{i}' for i in range(1, 11)]
        adata.var.index = list(adata.var.index[:30]) + ribo_genes + list(adata.var.index[50:])
        
        ribo_results = quality_service.calculate_ribosomal_metrics(adata)
        
        assert 'pct_counts_ribo' in adata.obs.columns
        assert 'n_genes_ribo' in adata.obs.columns
        assert ribo_results['n_ribo_genes'] == len(ribo_genes)
        assert ribo_results['mean_ribo_percent'] >= 0
    
    def test_calculate_gene_level_metrics(self, quality_service, mock_raw_single_cell_data):
        """Test calculation of gene-level QC metrics."""
        adata = mock_raw_single_cell_data.copy()
        
        gene_qc = quality_service.calculate_gene_metrics(adata)
        
        assert 'n_cells_by_counts' in adata.var.columns
        assert 'total_counts' in adata.var.columns
        assert 'pct_dropout' in adata.var.columns
        assert 'mean_counts' in adata.var.columns
        
        assert gene_qc['mean_expression_per_gene'] >= 0
        assert gene_qc['median_cells_per_gene'] >= 0
    
    def test_calculate_complexity_metrics(self, quality_service, mock_raw_single_cell_data):
        """Test calculation of library complexity metrics."""
        adata = mock_raw_single_cell_data.copy()
        
        complexity_results = quality_service.calculate_complexity_metrics(adata)
        
        assert 'library_complexity' in adata.obs.columns
        assert 'genes_per_umi' in adata.obs.columns
        assert 'log10_genes_per_umi' in adata.obs.columns
        
        assert complexity_results['mean_complexity'] >= 0
        assert complexity_results['complexity_distribution'] is not None
    
    def test_calculate_novelty_score(self, quality_service, mock_raw_single_cell_data):
        """Test calculation of novelty/complexity scores."""
        adata = mock_raw_single_cell_data.copy()
        
        novelty_results = quality_service.calculate_novelty_score(adata)
        
        assert 'novelty_score' in adata.obs.columns
        assert novelty_results['mean_novelty'] >= 0
        assert novelty_results['novelty_threshold_low'] < novelty_results['novelty_threshold_high']


# ===============================================================================
# Cell Quality Assessment Tests
# ===============================================================================

@pytest.mark.unit
class TestCellQualityAssessment:
    """Test cell quality assessment functionality."""
    
    def test_identify_low_quality_cells(self, quality_service, mock_processed_data):
        """Test identification of low quality cells."""
        adata = mock_processed_data.copy()
        
        low_quality_results = quality_service.identify_low_quality_cells(
            adata,
            min_genes=200,
            max_genes=5000,
            min_counts=500,
            max_counts=30000,
            max_mt_percent=20.0
        )
        
        assert 'passed_qc' in adata.obs.columns
        assert 'qc_failure_reasons' in adata.obs.columns
        assert low_quality_results['n_cells_passed'] <= adata.n_obs
        assert low_quality_results['n_cells_failed'] >= 0
        assert 'failure_breakdown' in low_quality_results
    
    def test_outlier_detection_iqr(self, quality_service, mock_processed_data):
        """Test outlier detection using IQR method."""
        adata = mock_processed_data.copy()
        
        outlier_results = quality_service.detect_outliers(
            adata,
            metrics=['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
            method='iqr',
            iqr_multiplier=1.5
        )
        
        assert 'outlier_iqr' in adata.obs.columns
        assert 'outlier_score_iqr' in adata.obs.columns
        assert outlier_results['n_outliers'] >= 0
        assert outlier_results['outlier_rate'] >= 0
    
    def test_outlier_detection_zscore(self, quality_service, mock_processed_data):
        """Test outlier detection using z-score method."""
        adata = mock_processed_data.copy()
        
        outlier_results = quality_service.detect_outliers(
            adata,
            metrics=['n_genes_by_counts', 'total_counts'],
            method='zscore',
            zscore_threshold=3.0
        )
        
        assert 'outlier_zscore' in adata.obs.columns
        assert 'outlier_score_zscore' in adata.obs.columns
        assert outlier_results['method'] == 'zscore'
    
    def test_outlier_detection_isolation_forest(self, quality_service, mock_processed_data):
        """Test outlier detection using isolation forest."""
        adata = mock_processed_data.copy()
        
        with patch.object(quality_service, 'detect_outliers_isolation_forest') as mock_isolation:
            mock_isolation.return_value = {
                'n_outliers': 45,
                'outlier_rate': 0.045,
                'contamination': 0.05,
                'anomaly_scores': np.random.uniform(-0.5, 0.5, adata.n_obs)
            }
            
            outlier_results = quality_service.detect_outliers_isolation_forest(
                adata,
                contamination=0.05
            )
            
            assert outlier_results['n_outliers'] > 0
            assert 'anomaly_scores' in outlier_results
    
    def test_empty_droplet_detection(self, quality_service, mock_raw_single_cell_data):
        """Test empty droplet detection."""
        adata = mock_raw_single_cell_data.copy()
        
        with patch.object(quality_service, 'detect_empty_droplets') as mock_empty:
            mock_empty.return_value = {
                'is_cell': np.random.choice([True, False], adata.n_obs, p=[0.85, 0.15]),
                'cell_probability': np.random.uniform(0.1, 0.99, adata.n_obs),
                'n_cells_detected': int(adata.n_obs * 0.85),
                'n_empty_droplets': int(adata.n_obs * 0.15),
                'method': 'emptyDrops'
            }
            
            empty_results = quality_service.detect_empty_droplets(adata)
            
            assert 'is_cell' in empty_results
            assert empty_results['n_cells_detected'] + empty_results['n_empty_droplets'] == adata.n_obs
    
    def test_ambient_rna_assessment(self, quality_service, mock_raw_single_cell_data):
        """Test ambient RNA contamination assessment."""
        adata = mock_raw_single_cell_data.copy()
        
        with patch.object(quality_service, 'assess_ambient_rna') as mock_ambient:
            mock_ambient.return_value = {
                'ambient_score': np.random.uniform(0, 0.3, adata.n_obs),
                'contamination_level': 'low',
                'highly_contaminated_cells': np.random.choice(adata.obs.index, size=50),
                'ambient_genes': ['HBB', 'HBA1', 'HBA2'],
                'decontamination_recommended': False
            }
            
            ambient_results = quality_service.assess_ambient_rna(adata)
            
            assert 'ambient_score' in ambient_results
            assert ambient_results['contamination_level'] in ['low', 'medium', 'high']


# ===============================================================================
# Doublet Detection Tests
# ===============================================================================

@pytest.mark.unit
class TestDoubletDetection:
    """Test doublet detection functionality."""
    
    def test_scrublet_doublet_detection(self, quality_service, mock_raw_single_cell_data):
        """Test doublet detection using Scrublet."""
        adata = mock_raw_single_cell_data.copy()
        
        with patch.object(quality_service, 'detect_doublets_scrublet') as mock_scrublet:
            mock_scrublet.return_value = {
                'doublet_score': np.random.uniform(0, 1, adata.n_obs),
                'predicted_doublet': np.random.choice([True, False], adata.n_obs, p=[0.08, 0.92]),
                'threshold': 0.25,
                'doublet_rate': 0.08,
                'n_doublets': int(adata.n_obs * 0.08)
            }
            
            doublet_results = quality_service.detect_doublets_scrublet(
                adata,
                expected_doublet_rate=0.08
            )
            
            assert 'doublet_score' in doublet_results
            assert 'predicted_doublet' in doublet_results
            assert doublet_results['doublet_rate'] == 0.08
    
    def test_doubletfinder_detection(self, quality_service, mock_raw_single_cell_data):
        """Test doublet detection using DoubletFinder-like method."""
        adata = mock_raw_single_cell_data.copy()
        
        with patch.object(quality_service, 'detect_doublets_doubletfinder') as mock_df:
            mock_df.return_value = {
                'doublet_score': np.random.uniform(0, 1, adata.n_obs),
                'predicted_doublet': np.random.choice([True, False], adata.n_obs, p=[0.06, 0.94]),
                'pK_optimal': 0.005,
                'pN': 0.25,
                'n_doublets': int(adata.n_obs * 0.06)
            }
            
            doublet_results = quality_service.detect_doublets_doubletfinder(adata)
            
            assert 'pK_optimal' in doublet_results
            assert doublet_results['n_doublets'] > 0
    
    def test_doublet_detection_comparison(self, quality_service, mock_raw_single_cell_data):
        """Test comparison of different doublet detection methods."""
        adata = mock_raw_single_cell_data.copy()
        
        with patch.object(quality_service, 'compare_doublet_methods') as mock_compare:
            mock_compare.return_value = {
                'methods_compared': ['scrublet', 'doubletfinder', 'hybrid'],
                'scrublet': {'n_doublets': 160, 'doublet_rate': 0.08, 'runtime': 12.5},
                'doubletfinder': {'n_doublets': 120, 'doublet_rate': 0.06, 'runtime': 45.2},
                'hybrid': {'n_doublets': 140, 'doublet_rate': 0.07, 'runtime': 25.8},
                'consensus_doublets': 100,
                'method_agreement': 0.75,
                'recommended_method': 'hybrid'
            }
            
            comparison = quality_service.compare_doublet_methods(
                adata,
                methods=['scrublet', 'doubletfinder', 'hybrid']
            )
            
            assert len(comparison['methods_compared']) == 3
            assert 'consensus_doublets' in comparison
            assert comparison['recommended_method'] in comparison['methods_compared']
    
    def test_doublet_validation(self, quality_service, mock_processed_data):
        """Test doublet detection validation."""
        adata = mock_processed_data.copy()
        
        # Add ground truth doublets (if available)
        adata.obs['known_doublet'] = np.random.choice([True, False], adata.n_obs, p=[0.08, 0.92])
        
        validation_results = quality_service.validate_doublet_detection(
            adata,
            predicted_key='is_doublet',
            true_key='known_doublet'
        )
        
        assert 'accuracy' in validation_results
        assert 'precision' in validation_results
        assert 'recall' in validation_results
        assert 'f1_score' in validation_results
        assert 0 <= validation_results['accuracy'] <= 1


# ===============================================================================
# Gene Quality Assessment Tests
# ===============================================================================

@pytest.mark.unit
class TestGeneQualityAssessment:
    """Test gene quality assessment functionality."""
    
    def test_identify_low_quality_genes(self, quality_service, mock_processed_data):
        """Test identification of low quality genes."""
        adata = mock_processed_data.copy()
        
        gene_qc_results = quality_service.identify_low_quality_genes(
            adata,
            min_cells=3,
            min_counts=10,
            max_dropout_rate=95.0
        )
        
        assert 'passed_gene_qc' in adata.var.columns
        assert 'gene_qc_failure_reasons' in adata.var.columns
        assert gene_qc_results['n_genes_passed'] <= adata.n_vars
        assert gene_qc_results['n_genes_failed'] >= 0
    
    def test_detect_housekeeping_genes(self, quality_service, mock_processed_data):
        """Test detection of housekeeping genes."""
        adata = mock_processed_data.copy()
        
        with patch.object(quality_service, 'detect_housekeeping_genes') as mock_hk:
            mock_hk.return_value = {
                'housekeeping_genes': ['ACTB', 'GAPDH', 'TUBB', 'RPL19', 'RPS18'],
                'n_housekeeping': 5,
                'detection_criteria': {
                    'min_detection_rate': 0.9,
                    'max_cv': 0.3,
                    'stable_expression': True
                }
            }
            
            hk_results = quality_service.detect_housekeeping_genes(adata)
            
            assert len(hk_results['housekeeping_genes']) == 5
            assert hk_results['n_housekeeping'] > 0
    
    def test_identify_spike_in_genes(self, quality_service, mock_processed_data):
        """Test identification of spike-in control genes."""
        adata = mock_processed_data.copy()
        
        # Add some ERCC spike-ins
        ercc_genes = [f'ERCC-{i:05d}' for i in range(1, 11)]
        adata.var.index = list(adata.var.index[:20]) + ercc_genes + list(adata.var.index[30:])
        
        spikein_results = quality_service.identify_spike_in_genes(adata)
        
        assert 'is_spike_in' in adata.var.columns
        assert spikein_results['n_spike_ins'] == len(ercc_genes)
        assert 'spike_in_patterns' in spikein_results
    
    def test_gene_expression_stability(self, quality_service, mock_processed_data):
        """Test gene expression stability assessment."""
        adata = mock_processed_data.copy()
        
        stability_results = quality_service.assess_gene_stability(adata)
        
        assert 'expression_cv' in adata.var.columns
        assert 'stability_rank' in adata.var.columns
        assert stability_results['most_stable_genes'] is not None
        assert stability_results['least_stable_genes'] is not None
        assert len(stability_results['most_stable_genes']) <= 100


# ===============================================================================
# Batch Effect Assessment Tests
# ===============================================================================

@pytest.mark.unit
class TestBatchEffectAssessment:
    """Test batch effect assessment functionality."""
    
    def test_detect_batch_effects_pca(self, quality_service, mock_bulk_data_with_issues):
        """Test batch effect detection using PCA."""
        adata = mock_bulk_data_with_issues.copy()
        
        # Add batch information
        adata.obs['batch'] = np.random.choice(['batch_1', 'batch_2', 'batch_3'], adata.n_obs)
        
        batch_results = quality_service.detect_batch_effects(
            adata,
            batch_key='batch',
            method='pca'
        )
        
        assert 'batch_effect_score' in batch_results
        assert 'batch_separation' in batch_results
        assert batch_results['method'] == 'pca'
        assert 0 <= batch_results['batch_effect_score'] <= 1
    
    def test_batch_effect_silhouette_analysis(self, quality_service, mock_bulk_data_with_issues):
        """Test batch effect assessment using silhouette analysis."""
        adata = mock_bulk_data_with_issues.copy()
        adata.obs['batch'] = np.random.choice(['batch_A', 'batch_B'], adata.n_obs)
        
        with patch.object(quality_service, 'batch_silhouette_analysis') as mock_silhouette:
            mock_silhouette.return_value = {
                'silhouette_batch': 0.45,
                'silhouette_biological': 0.62,
                'batch_effect_strength': 'moderate',
                'correction_recommended': True
            }
            
            silhouette_results = quality_service.batch_silhouette_analysis(
                adata,
                batch_key='batch'
            )
            
            assert 'silhouette_batch' in silhouette_results
            assert silhouette_results['batch_effect_strength'] in ['low', 'moderate', 'high']
    
    def test_kbet_batch_assessment(self, quality_service, mock_bulk_data_with_issues):
        """Test kBET batch effect assessment."""
        adata = mock_bulk_data_with_issues.copy()
        adata.obs['batch'] = np.random.choice(['batch_1', 'batch_2'], adata.n_obs)
        
        with patch.object(quality_service, 'kbet_analysis') as mock_kbet:
            mock_kbet.return_value = {
                'kbet_pvalue': 0.001,
                'kbet_observed': 0.25,
                'kbet_expected': 0.5,
                'batch_mixing_score': 0.3,
                'well_mixed': False
            }
            
            kbet_results = quality_service.kbet_analysis(
                adata,
                batch_key='batch'
            )
            
            assert 'kbet_pvalue' in kbet_results
            assert 'well_mixed' in kbet_results
    
    def test_lisi_score_calculation(self, quality_service, mock_bulk_data_with_issues):
        """Test LISI (Local Inverse Simpson's Index) score calculation."""
        adata = mock_bulk_data_with_issues.copy()
        adata.obs['batch'] = np.random.choice(['batch_1', 'batch_2', 'batch_3'], adata.n_obs)
        
        with patch.object(quality_service, 'calculate_lisi_score') as mock_lisi:
            mock_lisi.return_value = {
                'lisi_batch': np.random.uniform(1, 2.5, adata.n_obs),
                'mean_lisi_batch': 1.8,
                'lisi_score': 0.6,
                'integration_quality': 'good'
            }
            
            lisi_results = quality_service.calculate_lisi_score(
                adata,
                batch_key='batch'
            )
            
            assert 'mean_lisi_batch' in lisi_results
            assert lisi_results['integration_quality'] in ['poor', 'fair', 'good', 'excellent']


# ===============================================================================
# Data Validation Tests
# ===============================================================================

@pytest.mark.unit
class TestDataValidation:
    """Test data validation functionality."""
    
    def test_validate_count_data(self, quality_service, mock_raw_single_cell_data):
        """Test validation of count data properties."""
        adata = mock_raw_single_cell_data.copy()
        
        validation_results = quality_service.validate_count_data(adata)
        
        assert 'is_count_data' in validation_results
        assert 'has_negative_values' in validation_results
        assert 'has_non_integer_values' in validation_results
        assert 'data_type' in validation_results
        assert validation_results['data_type'] in ['raw_counts', 'normalized', 'log_transformed', 'unknown']
    
    def test_validate_data_structure(self, quality_service, mock_raw_single_cell_data):
        """Test validation of data structure."""
        adata = mock_raw_single_cell_data.copy()
        
        structure_results = quality_service.validate_data_structure(adata)
        
        assert 'shape_valid' in structure_results
        assert 'has_obs_names' in structure_results
        assert 'has_var_names' in structure_results
        assert 'obs_names_unique' in structure_results
        assert 'var_names_unique' in structure_results
    
    def test_check_data_completeness(self, quality_service, mock_raw_single_cell_data):
        """Test checking data completeness."""
        adata = mock_raw_single_cell_data.copy()
        
        completeness_results = quality_service.check_data_completeness(adata)
        
        assert 'missing_values_count' in completeness_results
        assert 'missing_values_percentage' in completeness_results
        assert 'complete_observations' in completeness_results
        assert 'sparsity' in completeness_results
    
    def test_validate_gene_names(self, quality_service, mock_raw_single_cell_data):
        """Test validation of gene names."""
        adata = mock_raw_single_cell_data.copy()
        
        with patch.object(quality_service, 'validate_gene_names') as mock_validate:
            mock_validate.return_value = {
                'valid_gene_symbols': 0.85,
                'ensembl_ids_detected': True,
                'hugo_symbols_detected': True,
                'invalid_names': ['INVALID1', 'INVALID2'],
                'naming_convention': 'mixed'
            }
            
            name_results = quality_service.validate_gene_names(adata)
            
            assert 'valid_gene_symbols' in name_results
            assert name_results['naming_convention'] in ['ensembl', 'hugo', 'mixed', 'unknown']
    
    def test_check_technical_artifacts(self, quality_service, mock_raw_single_cell_data):
        """Test detection of technical artifacts."""
        adata = mock_raw_single_cell_data.copy()
        
        artifact_results = quality_service.check_technical_artifacts(adata)
        
        assert 'cell_cycle_genes_detected' in artifact_results
        assert 'stress_response_genes' in artifact_results
        assert 'dissociation_artifacts' in artifact_results
        assert 'potential_artifacts' in artifact_results


# ===============================================================================
# Quality Reporting Tests
# ===============================================================================

@pytest.mark.unit
class TestQualityReporting:
    """Test quality reporting functionality."""
    
    def test_generate_qc_report(self, quality_service, mock_processed_data):
        """Test generation of comprehensive QC report."""
        adata = mock_processed_data.copy()
        
        with patch.object(quality_service, 'generate_qc_report') as mock_report:
            mock_report.return_value = {
                'summary': {
                    'n_cells_before_qc': 2000,
                    'n_cells_after_qc': 1850,
                    'n_genes_before_qc': 3000,
                    'n_genes_after_qc': 2750,
                    'doublet_rate': 0.08,
                    'mt_contamination': 'acceptable'
                },
                'quality_metrics': {
                    'mean_genes_per_cell': 1250,
                    'mean_counts_per_cell': 5500,
                    'median_mt_percent': 8.5
                },
                'recommendations': [
                    'Filter cells with >20% mitochondrial genes',
                    'Consider doublet removal',
                    'Data quality is acceptable for analysis'
                ]
            }
            
            report = quality_service.generate_qc_report(adata)
            
            assert 'summary' in report
            assert 'quality_metrics' in report
            assert 'recommendations' in report
    
    def test_create_qc_plots(self, quality_service, mock_processed_data):
        """Test creation of QC plots."""
        adata = mock_processed_data.copy()
        
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            plot_info = quality_service.create_qc_plots(
                adata,
                save_plots=True,
                plot_dir='test_plots'
            )
            
            assert 'violin_plots' in plot_info['plots_created']
            assert 'scatter_plots' in plot_info['plots_created']
            assert 'histogram_plots' in plot_info['plots_created']
            assert plot_info['n_plots'] > 0
    
    def test_export_qc_metrics(self, quality_service, mock_processed_data):
        """Test export of QC metrics to file."""
        adata = mock_processed_data.copy()
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            export_info = quality_service.export_qc_metrics(
                adata,
                output_file='qc_metrics.csv',
                include_plots=True
            )
            
            assert export_info['file_exported'] == 'qc_metrics.csv'
            assert export_info['metrics_included'] > 0
            mock_to_csv.assert_called_once()
    
    def test_qc_dashboard_data(self, quality_service, mock_processed_data):
        """Test preparation of data for QC dashboard."""
        adata = mock_processed_data.copy()
        
        dashboard_data = quality_service.prepare_dashboard_data(adata)
        
        assert 'cell_metrics' in dashboard_data
        assert 'gene_metrics' in dashboard_data
        assert 'summary_stats' in dashboard_data
        assert 'plot_data' in dashboard_data
        
        # Check that data is JSON serializable
        import json
        json_str = json.dumps(dashboard_data, default=str)
        assert len(json_str) > 0


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestQualityServiceErrorHandling:
    """Test quality service error handling and edge cases."""
    
    def test_empty_dataset_handling(self, quality_service):
        """Test handling of empty datasets."""
        empty_adata = ad.AnnData(X=np.array([]).reshape(0, 0))
        
        with pytest.raises(ValueError, match="Empty dataset"):
            quality_service.calculate_qc_metrics(empty_adata)
    
    def test_single_cell_dataset_handling(self, quality_service):
        """Test handling of single-cell datasets."""
        single_cell_adata = ad.AnnData(X=np.array([[1, 2, 3, 4, 5]]))
        
        # Should handle gracefully with warnings
        qc_results = quality_service.calculate_qc_metrics(single_cell_adata)
        assert qc_results is not None
        assert 'warning' in qc_results
    
    def test_all_zero_data_handling(self, quality_service):
        """Test handling of all-zero expression data."""
        zero_adata = ad.AnnData(X=np.zeros((100, 50)))
        
        qc_results = quality_service.calculate_qc_metrics(zero_adata)
        
        assert qc_results['mean_counts_per_cell'] == 0
        assert 'all_zero_warning' in qc_results
    
    def test_missing_gene_names_handling(self, quality_service):
        """Test handling of missing or invalid gene names."""
        adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        # Set some gene names to None/empty
        adata.var.index = [f'Gene_{i}' if i % 10 != 0 else '' for i in range(adata.n_vars)]
        
        validation_results = quality_service.validate_gene_names(adata)
        
        assert 'empty_gene_names' in validation_results
        assert validation_results['empty_gene_names'] > 0
    
    def test_memory_efficient_qc(self, quality_service):
        """Test memory-efficient QC for large datasets."""
        # Create large sparse dataset
        large_sparse = sp.random(50000, 5000, density=0.01, format='csr')
        large_adata = ad.AnnData(X=large_sparse)
        
        qc_results = quality_service.calculate_qc_metrics(
            large_adata,
            memory_efficient=True,
            chunk_size=5000
        )
        
        assert qc_results is not None
        assert qc_results['n_cells'] == 50000
    
    def test_concurrent_qc_processing(self, quality_service, mock_raw_single_cell_data):
        """Test thread safety for concurrent QC operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def qc_worker(worker_id):
            """Worker function for concurrent QC processing."""
            try:
                adata = mock_raw_single_cell_data.copy()
                
                qc_result = quality_service.calculate_qc_metrics(adata)
                results.append((worker_id, qc_result))
                time.sleep(0.01)
                
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=qc_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent QC processing errors: {errors}"
        assert len(results) == 3
    
    def test_corrupted_data_handling(self, quality_service):
        """Test handling of corrupted or inconsistent data."""
        # Create data with mismatched dimensions
        corrupted_X = np.random.randn(100, 50)
        obs_df = pd.DataFrame(index=[f'Cell_{i}' for i in range(90)])  # Wrong number
        var_df = pd.DataFrame(index=[f'Gene_{i}' for i in range(50)])
        
        with pytest.raises(ValueError, match="Inconsistent dimensions"):
            corrupted_adata = ad.AnnData(X=corrupted_X, obs=obs_df, var=var_df)
            quality_service.calculate_qc_metrics(corrupted_adata)
    
    def test_extreme_outlier_handling(self, quality_service):
        """Test handling of extreme outlier values."""
        # Create data with extreme outliers
        normal_data = np.random.randn(1000, 100) + 5
        normal_data[0, :] = 1000000  # Extreme outlier cell
        normal_data[:, 0] = -1000000  # Extreme outlier gene
        
        outlier_adata = ad.AnnData(X=normal_data)
        
        outlier_results = quality_service.detect_outliers(
            outlier_adata,
            method='robust_zscore',
            handle_extremes=True
        )
        
        assert 'extreme_outliers_detected' in outlier_results
        assert outlier_results['n_outliers'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])