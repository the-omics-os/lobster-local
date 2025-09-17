"""
Comprehensive unit tests for proteomics preprocessing service.

This module provides thorough testing of the proteomics preprocessing service including
missing value imputation, intensity normalization, and batch correction
for proteomics data analysis.

Test coverage target: 95%+ with meaningful tests for proteomics preprocessing operations.
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

from lobster.tools.proteomics_preprocessing_service import ProteomicsPreprocessingService, ProteomicsPreprocessingError

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
    """Create ProteomicsPreprocessingService instance."""
    return ProteomicsPreprocessingService()


@pytest.fixture
def mock_adata_with_missing():
    """Create mock AnnData with missing values."""
    n_samples, n_proteins = 48, 100
    X = np.random.randn(n_samples, n_proteins) * 1000 + 5000  # Positive values

    # Add missing values (20% missing rate)
    missing_mask = np.random.rand(n_samples, n_proteins) < 0.2
    X[missing_mask] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add metadata
    adata.obs['batch'] = ['batch1'] * 16 + ['batch2'] * 16 + ['batch3'] * 16
    adata.obs['condition'] = ['control'] * 24 + ['treatment'] * 24
    adata.var['protein_names'] = [f"PROT_{i}" for i in range(n_proteins)]

    return adata


@pytest.fixture
def mock_adata_no_missing():
    """Create mock AnnData without missing values."""
    n_samples, n_proteins = 48, 100
    X = np.random.randn(n_samples, n_proteins) * 1000 + 5000  # Positive values

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add metadata
    adata.obs['batch'] = ['batch1'] * 16 + ['batch2'] * 16 + ['batch3'] * 16
    adata.obs['condition'] = ['control'] * 24 + ['treatment'] * 24

    return adata


@pytest.fixture
def mock_adata_with_batches():
    """Create mock AnnData with pronounced batch effects."""
    n_samples, n_proteins = 60, 100
    X = np.random.randn(n_samples, n_proteins) * 500 + 3000

    # Add batch effects
    batch_effects = [0, 1000, 2000]  # Different offset for each batch
    for i, batch_effect in enumerate(batch_effects):
        start_idx = i * 20
        end_idx = (i + 1) * 20
        X[start_idx:end_idx, :] += batch_effect

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add batch labels
    adata.obs['batch'] = ['batch1'] * 20 + ['batch2'] * 20 + ['batch3'] * 20
    adata.obs['condition'] = ['control'] * 30 + ['treatment'] * 30

    return adata


# ===============================================================================
# Service Initialization Tests
# ===============================================================================

class TestProteomicsPreprocessingServiceInitialization:
    """Test suite for ProteomicsPreprocessingService initialization."""

    def test_init_default_parameters(self):
        """Test service initialization with default parameters."""
        service = ProteomicsPreprocessingService()

        assert service is not None


# ===============================================================================
# Missing Value Imputation Tests
# ===============================================================================

class TestMissingValueImputation:
    """Test suite for missing value imputation functionality."""

    def test_impute_missing_values_knn(self, service, mock_adata_with_missing):
        """Test KNN imputation."""
        original_missing = np.isnan(mock_adata_with_missing.X).sum()

        result_adata, stats = service.impute_missing_values(
            mock_adata_with_missing,
            method='knn',
            knn_neighbors=5
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['method'] == 'knn'
        assert stats['missing_values_found'] is True
        assert stats['imputation_performed'] is True
        assert stats['original_missing_count'] == original_missing
        assert stats['remaining_missing_count'] == 0  # All should be imputed
        assert not np.any(np.isnan(result_adata.X))

    def test_impute_missing_values_min_prob(self, service, mock_adata_with_missing):
        """Test minimum probability imputation."""
        result_adata, stats = service.impute_missing_values(
            mock_adata_with_missing,
            method='min_prob',
            min_prob_percentile=2.5
        )

        assert result_adata is not None
        assert stats['method'] == 'min_prob'
        assert not np.any(np.isnan(result_adata.X))

    def test_impute_missing_values_mnar(self, service, mock_adata_with_missing):
        """Test MNAR imputation."""
        result_adata, stats = service.impute_missing_values(
            mock_adata_with_missing,
            method='mnar',
            mnar_width=0.3,
            mnar_downshift=1.8
        )

        assert result_adata is not None
        assert stats['method'] == 'mnar'
        assert not np.any(np.isnan(result_adata.X))

    def test_impute_missing_values_mixed(self, service, mock_adata_with_missing):
        """Test mixed imputation strategy."""
        result_adata, stats = service.impute_missing_values(
            mock_adata_with_missing,
            method='mixed',
            knn_neighbors=5,
            min_prob_percentile=2.5
        )

        assert result_adata is not None
        assert stats['method'] == 'mixed'
        assert not np.any(np.isnan(result_adata.X))

    def test_impute_missing_values_no_missing(self, service, mock_adata_no_missing):
        """Test imputation when no missing values are present."""
        result_adata, stats = service.impute_missing_values(
            mock_adata_no_missing,
            method='knn'
        )

        assert result_adata is not None
        assert stats['missing_values_found'] is False
        assert stats['imputation_performed'] is False

    def test_impute_missing_values_invalid_method(self, service, mock_adata_with_missing):
        """Test imputation with invalid method."""
        with pytest.raises(ProteomicsPreprocessingError) as exc_info:
            service.impute_missing_values(
                mock_adata_with_missing,
                method='invalid_method'
            )

        assert "Unknown imputation method" in str(exc_info.value)

    def test_impute_missing_values_preserves_raw(self, service, mock_adata_with_missing):
        """Test that imputation preserves raw data."""
        # Remove raw if it exists to test creation
        mock_adata_with_missing.raw = None

        result_adata, stats = service.impute_missing_values(
            mock_adata_with_missing,
            method='knn'
        )

        assert result_adata.raw is not None
        # Raw should contain original data with missing values
        assert np.isnan(result_adata.raw.X).any()
        # Processed data should have no missing values
        assert not np.any(np.isnan(result_adata.X))

    def test_imputation_statistics_accuracy(self, service):
        """Test accuracy of imputation statistics."""
        # Create data with known missing value pattern
        X = np.ones((10, 5)) * 1000
        X[0:2, 0:2] = np.nan  # 4 missing values

        adata = ad.AnnData(X=X)

        result_adata, stats = service.impute_missing_values(adata, method='knn')

        assert stats['original_missing_count'] == 4
        assert stats['original_missing_percentage'] == 8.0  # 4/50 * 100
        assert stats['remaining_missing_count'] == 0


# ===============================================================================
# Helper Methods Tests for Imputation
# ===============================================================================

class TestImputationHelperMethods:
    """Test suite for imputation helper methods."""

    def test_knn_imputation_helper(self, service):
        """Test KNN imputation helper method."""
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        result = service._knn_imputation(X, n_neighbors=2)

        assert result is not None
        assert not np.any(np.isnan(result))
        assert result.shape == X.shape

    def test_min_prob_imputation_helper(self, service):
        """Test minimum probability imputation helper method."""
        X = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        result = service._min_prob_imputation(X, percentile=10.0)

        assert result is not None
        assert not np.any(np.isnan(result))
        assert result.shape == X.shape

    def test_mnar_imputation_helper(self, service):
        """Test MNAR imputation helper method."""
        X = np.array([[10.0, 20.0, np.nan], [40.0, np.nan, 60.0], [70.0, 80.0, 90.0]])
        result = service._mnar_imputation(X, width=0.3, downshift=1.8)

        assert result is not None
        assert not np.any(np.isnan(result))
        assert result.shape == X.shape

        # MNAR imputed values should be lower than observed values
        for col in range(X.shape[1]):
            observed = X[:, col][~np.isnan(X[:, col])]
            if len(observed) > 0:
                min_observed = np.min(observed)
                imputed_positions = np.isnan(X[:, col])
                if imputed_positions.any():
                    imputed_values = result[:, col][imputed_positions]
                    assert all(val < min_observed for val in imputed_values)

    def test_mixed_imputation_helper(self, service):
        """Test mixed imputation helper method."""
        # Create data with different missing patterns
        X = np.random.randn(20, 4) * 100 + 1000

        # Low missing rate column (should use KNN)
        X[0:2, 0] = np.nan  # 10% missing

        # High missing rate column (should use MNAR)
        X[0:12, 1] = np.nan  # 60% missing

        result = service._mixed_imputation(
            X, knn_neighbors=5, min_prob_percentile=2.5,
            mnar_width=0.3, mnar_downshift=1.8, mcar_threshold=0.4
        )

        assert result is not None
        assert not np.any(np.isnan(result))
        assert result.shape == X.shape


# ===============================================================================
# Intensity Normalization Tests
# ===============================================================================

class TestIntensityNormalization:
    """Test suite for intensity normalization functionality."""

    def test_normalize_intensities_median(self, service, mock_adata_no_missing):
        """Test median normalization."""
        result_adata, stats = service.normalize_intensities(
            mock_adata_no_missing,
            method='median',
            log_transform=False
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['method'] == 'median'
        assert stats['log_transform'] is False
        assert 'normalized' in result_adata.layers
        assert result_adata.raw is not None

    def test_normalize_intensities_with_log(self, service, mock_adata_no_missing):
        """Test normalization with log transformation."""
        result_adata, stats = service.normalize_intensities(
            mock_adata_no_missing,
            method='median',
            log_transform=True,
            pseudocount_strategy='adaptive'
        )

        assert result_adata is not None
        assert stats['log_transform'] is True
        assert 'log2_normalized' in result_adata.layers
        assert 'pseudocount' in stats
        assert 'pseudocount_strategy' in stats

    def test_normalize_intensities_quantile(self, service, mock_adata_no_missing):
        """Test quantile normalization."""
        result_adata, stats = service.normalize_intensities(
            mock_adata_no_missing,
            method='quantile'
        )

        assert result_adata is not None
        assert stats['method'] == 'quantile'

    def test_normalize_intensities_vsn(self, service, mock_adata_no_missing):
        """Test VSN normalization."""
        result_adata, stats = service.normalize_intensities(
            mock_adata_no_missing,
            method='vsn'
        )

        assert result_adata is not None
        assert stats['method'] == 'vsn'

    def test_normalize_intensities_total_sum(self, service, mock_adata_no_missing):
        """Test total sum normalization."""
        result_adata, stats = service.normalize_intensities(
            mock_adata_no_missing,
            method='total_sum'
        )

        assert result_adata is not None
        assert stats['method'] == 'total_sum'

    def test_normalize_intensities_invalid_method(self, service, mock_adata_no_missing):
        """Test normalization with invalid method."""
        with pytest.raises(ProteomicsPreprocessingError) as exc_info:
            service.normalize_intensities(
                mock_adata_no_missing,
                method='invalid_method'
            )

        assert "Unknown normalization method" in str(exc_info.value)

    def test_different_pseudocount_strategies(self, service, mock_adata_no_missing):
        """Test different pseudocount strategies."""
        strategies = ['adaptive', 'fixed', 'min_observed']

        for strategy in strategies:
            result_adata, stats = service.normalize_intensities(
                mock_adata_no_missing,
                method='median',
                log_transform=True,
                pseudocount_strategy=strategy
            )

            assert result_adata is not None
            assert stats['pseudocount_strategy'] == strategy
            assert 'pseudocount' in stats

    def test_normalization_with_negative_values(self, service):
        """Test normalization handling negative values."""
        X = np.random.randn(10, 5)  # Contains negative values
        adata = ad.AnnData(X=X)

        # Should still work but may log warning
        result_adata, stats = service.normalize_intensities(adata, method='median')

        assert result_adata is not None

    def test_normalization_statistics(self, service, mock_adata_no_missing):
        """Test normalization statistics calculation."""
        result_adata, stats = service.normalize_intensities(
            mock_adata_no_missing,
            method='median'
        )

        assert 'median_intensity_before' in stats
        assert 'median_intensity_after' in stats
        assert 'samples_processed' in stats
        assert 'proteins_processed' in stats
        assert stats['analysis_type'] == 'intensity_normalization'


# ===============================================================================
# Helper Methods Tests for Normalization
# ===============================================================================

class TestNormalizationHelperMethods:
    """Test suite for normalization helper methods."""

    def test_median_normalization_helper(self, service):
        """Test median normalization helper method."""
        X = np.array([[100, 200, 300], [200, 400, 600], [50, 100, 150]])
        result = service._median_normalization(X)

        assert result is not None
        assert result.shape == X.shape
        # After median normalization, sample medians should be similar
        sample_medians = np.median(result, axis=1)
        assert np.allclose(sample_medians, sample_medians[0], rtol=0.1)

    def test_quantile_normalization_helper(self, service):
        """Test quantile normalization helper method."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = service._quantile_normalization(X)

        assert result is not None
        assert result.shape == X.shape

    def test_vsn_normalization_helper(self, service):
        """Test VSN normalization helper method."""
        X = np.array([[100, 200, 300], [400, 500, 600]])
        result = service._vsn_normalization(X)

        assert result is not None
        assert result.shape == X.shape
        # VSN should produce values in a different scale
        assert not np.allclose(result, X)

    def test_total_sum_normalization_helper(self, service):
        """Test total sum normalization helper method."""
        X = np.array([[10, 20, 30], [20, 40, 60]])
        result = service._total_sum_normalization(X)

        assert result is not None
        assert result.shape == X.shape
        # Row sums should be equal after normalization (scaled to 1M)
        row_sums = np.sum(result, axis=1)
        assert np.allclose(row_sums, 1e6)

    def test_apply_log_transformation_helper(self, service):
        """Test log transformation helper method."""
        X = np.array([[100, 200, 300], [400, 500, 600]])

        strategies = ['adaptive', 'fixed', 'min_observed']

        for strategy in strategies:
            result, log_stats = service._apply_log_transformation(X, strategy)

            assert result is not None
            assert result.shape == X.shape
            assert 'pseudocount' in log_stats
            assert 'pseudocount_strategy' in log_stats
            assert log_stats['pseudocount_strategy'] == strategy


# ===============================================================================
# Batch Correction Tests
# ===============================================================================

class TestBatchCorrection:
    """Test suite for batch correction functionality."""

    def test_correct_batch_effects_combat(self, service, mock_adata_with_batches):
        """Test ComBat batch correction."""
        result_adata, stats = service.correct_batch_effects(
            mock_adata_with_batches,
            batch_key='batch',
            method='combat'
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['method'] == 'combat'
        assert stats['batch_correction_performed'] is True
        assert stats['n_batches'] == 3
        assert 'batch_corrected' in result_adata.layers
        assert result_adata.raw is not None

    def test_correct_batch_effects_median_centering(self, service, mock_adata_with_batches):
        """Test median centering batch correction."""
        result_adata, stats = service.correct_batch_effects(
            mock_adata_with_batches,
            batch_key='batch',
            method='median_centering'
        )

        assert result_adata is not None
        assert stats['method'] == 'median_centering'

    def test_correct_batch_effects_reference_based(self, service, mock_adata_with_batches):
        """Test reference-based batch correction."""
        result_adata, stats = service.correct_batch_effects(
            mock_adata_with_batches,
            batch_key='batch',
            method='reference_based',
            reference_batch='batch1'
        )

        assert result_adata is not None
        assert stats['method'] == 'reference_based'
        assert stats['reference_batch'] == 'batch1'

    def test_correct_batch_effects_auto_reference(self, service, mock_adata_with_batches):
        """Test reference-based correction with automatic reference selection."""
        result_adata, stats = service.correct_batch_effects(
            mock_adata_with_batches,
            batch_key='batch',
            method='reference_based'
        )

        assert result_adata is not None
        assert stats['reference_batch'] is not None  # Should auto-select

    def test_correct_batch_effects_invalid_batch_key(self, service, mock_adata_with_batches):
        """Test batch correction with invalid batch key."""
        with pytest.raises(ProteomicsPreprocessingError) as exc_info:
            service.correct_batch_effects(
                mock_adata_with_batches,
                batch_key='nonexistent_batch'
            )

        assert "Batch key 'nonexistent_batch' not found" in str(exc_info.value)

    def test_correct_batch_effects_single_batch(self, service):
        """Test batch correction with single batch (should skip)."""
        X = np.random.randn(10, 5)
        adata = ad.AnnData(X=X)
        adata.obs['batch'] = ['batch1'] * 10  # Only one batch

        result_adata, stats = service.correct_batch_effects(
            adata,
            batch_key='batch'
        )

        assert result_adata is not None
        assert stats['batch_correction_performed'] is False
        assert stats['n_batches'] == 1

    def test_correct_batch_effects_invalid_method(self, service, mock_adata_with_batches):
        """Test batch correction with invalid method."""
        with pytest.raises(ProteomicsPreprocessingError) as exc_info:
            service.correct_batch_effects(
                mock_adata_with_batches,
                batch_key='batch',
                method='invalid_method'
            )

        assert "Unknown batch correction method" in str(exc_info.value)

    def test_batch_correction_statistics(self, service, mock_adata_with_batches):
        """Test batch correction statistics calculation."""
        result_adata, stats = service.correct_batch_effects(
            mock_adata_with_batches,
            batch_key='batch'
        )

        assert 'batch_counts' in stats
        assert 'samples_processed' in stats
        assert 'proteins_processed' in stats
        assert stats['analysis_type'] == 'batch_correction'


# ===============================================================================
# Helper Methods Tests for Batch Correction
# ===============================================================================

class TestBatchCorrectionHelperMethods:
    """Test suite for batch correction helper methods."""

    def test_combat_correction_helper(self, service):
        """Test ComBat correction helper method."""
        X = np.array([[100, 200], [300, 400], [150, 250], [350, 450]])
        batch_labels = pd.Series(['batch1', 'batch1', 'batch2', 'batch2'])

        result = service._combat_correction(X, batch_labels)

        assert result is not None
        assert result.shape == X.shape

    def test_median_centering_correction_helper(self, service):
        """Test median centering correction helper method."""
        X = np.array([[100, 200], [300, 400], [150, 250], [350, 450]])
        batch_labels = pd.Series(['batch1', 'batch1', 'batch2', 'batch2'])

        result = service._median_centering_correction(X, batch_labels)

        assert result is not None
        assert result.shape == X.shape

    def test_reference_based_correction_helper(self, service):
        """Test reference-based correction helper method."""
        X = np.array([[100, 200], [300, 400], [150, 250], [350, 450]])
        batch_labels = pd.Series(['batch1', 'batch1', 'batch2', 'batch2'])

        result = service._reference_based_correction(X, batch_labels, 'batch1')

        assert result is not None
        assert result.shape == X.shape

    def test_reference_based_correction_auto_reference(self, service):
        """Test reference-based correction with auto reference selection."""
        X = np.array([[100, 200], [300, 400], [150, 250]])
        # batch1 has more samples, should be auto-selected as reference
        batch_labels = pd.Series(['batch1', 'batch1', 'batch2'])

        result = service._reference_based_correction(X, batch_labels, None)

        assert result is not None
        assert result.shape == X.shape

    def test_calculate_batch_correction_stats_helper(self, service):
        """Test batch correction statistics calculation helper."""
        X_before = np.random.randn(20, 10)
        X_after = np.random.randn(20, 10)
        batch_labels = pd.Series(['batch1'] * 10 + ['batch2'] * 10)

        stats = service._calculate_batch_correction_stats(
            X_before, X_after, batch_labels, n_pcs=5
        )

        assert isinstance(stats, dict)
        # Should contain PCA statistics or gracefully handle errors


# ===============================================================================
# Error Handling and Edge Cases Tests
# ===============================================================================

class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""

    def test_error_handling_empty_data(self, service):
        """Test error handling with empty data."""
        adata = ad.AnnData(X=np.array([]).reshape(0, 0))

        with pytest.raises(ProteomicsPreprocessingError):
            service.impute_missing_values(adata, method='knn')

    def test_imputation_with_all_missing_protein(self, service):
        """Test imputation when a protein has all missing values."""
        X = np.ones((10, 3)) * 1000
        X[:, 0] = np.nan  # First protein all missing

        adata = ad.AnnData(X=X)

        result_adata, stats = service.impute_missing_values(adata, method='knn')

        assert result_adata is not None
        # Should handle gracefully

    def test_normalization_with_zero_values(self, service):
        """Test normalization with zero values."""
        X = np.array([[0, 100, 200], [0, 200, 400]])
        adata = ad.AnnData(X=X)

        result_adata, stats = service.normalize_intensities(adata, method='median')

        assert result_adata is not None

    def test_batch_correction_with_missing_values(self, service):
        """Test batch correction with missing values."""
        X = np.random.randn(20, 10) * 1000 + 5000
        X[0:5, 0:3] = np.nan  # Add some missing values

        adata = ad.AnnData(X=X)
        adata.obs['batch'] = ['batch1'] * 10 + ['batch2'] * 10

        result_adata, stats = service.correct_batch_effects(adata, batch_key='batch')

        assert result_adata is not None

    def test_very_small_dataset_imputation(self, service):
        """Test imputation with very small dataset."""
        X = np.array([[1.0, np.nan], [np.nan, 2.0]])
        adata = ad.AnnData(X=X)

        result_adata, stats = service.impute_missing_values(adata, method='knn', knn_neighbors=1)

        assert result_adata is not None

    def test_single_sample_normalization(self, service):
        """Test normalization with single sample."""
        X = np.array([[100, 200, 300]])  # Single sample
        adata = ad.AnnData(X=X)

        result_adata, stats = service.normalize_intensities(adata, method='median')

        assert result_adata is not None


# ===============================================================================
# Integration Tests
# ===============================================================================

class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    def test_complete_preprocessing_workflow(self, service, mock_adata_with_missing):
        """Test complete preprocessing workflow: imputation -> normalization -> batch correction."""
        # Add batch information
        mock_adata_with_missing.obs['batch'] = ['batch1'] * 16 + ['batch2'] * 16 + ['batch3'] * 16

        # Step 1: Imputation
        adata_imputed, _ = service.impute_missing_values(
            mock_adata_with_missing,
            method='mixed'
        )

        # Step 2: Normalization
        adata_normalized, _ = service.normalize_intensities(
            adata_imputed,
            method='median',
            log_transform=True
        )

        # Step 3: Batch correction
        adata_corrected, _ = service.correct_batch_effects(
            adata_normalized,
            batch_key='batch',
            method='combat'
        )

        # Verify final result has all preprocessing components
        assert not np.any(np.isnan(adata_corrected.X))  # No missing values
        assert 'normalized' in adata_corrected.layers
        assert 'log2_normalized' in adata_corrected.layers
        assert 'batch_corrected' in adata_corrected.layers
        assert adata_corrected.raw is not None

    def test_preprocessing_preserves_metadata(self, service, mock_adata_with_missing):
        """Test that preprocessing preserves sample and protein metadata."""
        original_obs_columns = mock_adata_with_missing.obs.columns.tolist()
        original_var_columns = mock_adata_with_missing.var.columns.tolist()
        original_obs_names = mock_adata_with_missing.obs_names.tolist()
        original_var_names = mock_adata_with_missing.var_names.tolist()

        result_adata, _ = service.impute_missing_values(
            mock_adata_with_missing,
            method='knn'
        )

        # Metadata should be preserved
        assert result_adata.obs.columns.tolist() == original_obs_columns
        assert result_adata.var.columns.tolist() == original_var_columns
        assert result_adata.obs_names.tolist() == original_obs_names
        assert result_adata.var_names.tolist() == original_var_names

    def test_multiple_normalization_methods_consistency(self, service, mock_adata_no_missing):
        """Test consistency across different normalization methods."""
        methods = ['median', 'quantile', 'total_sum']
        results = {}

        for method in methods:
            result_adata, stats = service.normalize_intensities(
                mock_adata_no_missing,
                method=method,
                log_transform=False
            )
            results[method] = stats

        # All methods should process same number of samples/proteins
        for method in methods:
            assert results[method]['samples_processed'] == mock_adata_no_missing.n_obs
            assert results[method]['proteins_processed'] == mock_adata_no_missing.n_vars


# ===============================================================================
# Performance and Memory Tests
# ===============================================================================

class TestPerformanceAndMemory:
    """Test suite for performance and memory considerations."""

    @pytest.mark.slow
    def test_large_dataset_imputation(self, service):
        """Test imputation with large dataset."""
        # Create larger dataset
        n_samples, n_proteins = 200, 500
        X = np.random.randn(n_samples, n_proteins) * 1000 + 5000
        # Add missing values
        missing_mask = np.random.rand(n_samples, n_proteins) < 0.15
        X[missing_mask] = np.nan

        adata = ad.AnnData(X=X)

        result_adata, stats = service.impute_missing_values(adata, method='knn')

        assert result_adata is not None
        assert stats['samples_processed'] == n_samples
        assert stats['proteins_processed'] == n_proteins

    @pytest.mark.slow
    def test_memory_efficient_normalization(self, service):
        """Test memory efficiency in normalization."""
        # Create moderately large dataset
        n_samples, n_proteins = 100, 1000
        X = np.random.randn(n_samples, n_proteins) * 1000 + 5000
        adata = ad.AnnData(X=X)

        result_adata, stats = service.normalize_intensities(
            adata,
            method='quantile',
            log_transform=True
        )

        assert result_adata is not None
        # Should complete without memory errors

    def test_batch_correction_performance(self, service):
        """Test batch correction performance with multiple batches."""
        n_samples, n_proteins = 100, 200
        X = np.random.randn(n_samples, n_proteins) * 1000 + 5000
        adata = ad.AnnData(X=X)

        # Create 5 batches
        adata.obs['batch'] = [f'batch{i//20 + 1}' for i in range(n_samples)]

        result_adata, stats = service.correct_batch_effects(
            adata,
            batch_key='batch',
            method='combat'
        )

        assert result_adata is not None
        assert stats['n_batches'] == 5