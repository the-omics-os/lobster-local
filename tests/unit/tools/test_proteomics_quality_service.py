"""
Comprehensive unit tests for proteomics quality service.

This module provides thorough testing of the proteomics quality service including
missing value pattern analysis, coefficient of variation assessment, contaminant detection,
and technical replicate validation for proteomics data analysis.

Test coverage target: 95%+ with meaningful tests for proteomics quality operations.
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

from lobster.tools.proteomics_quality_service import ProteomicsQualityService, ProteomicsQualityError

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
    """Create ProteomicsQualityService instance."""
    return ProteomicsQualityService()


@pytest.fixture
def mock_adata_with_missing():
    """Create mock AnnData with missing values and realistic patterns."""
    n_samples, n_proteins = 48, 100
    X = np.random.lognormal(mean=10, sigma=1, size=(n_samples, n_proteins))

    # Add structured missing values
    # Some proteins have high missing rates (MNAR pattern)
    high_missing_proteins = np.random.choice(n_proteins, 20, replace=False)
    for protein_idx in high_missing_proteins:
        missing_mask = np.random.rand(n_samples) < 0.6  # 60% missing
        X[missing_mask, protein_idx] = np.nan

    # Some samples have slightly higher missing rates
    high_missing_samples = np.random.choice(n_samples, 10, replace=False)
    for sample_idx in high_missing_samples:
        missing_mask = np.random.rand(n_proteins) < 0.3  # 30% missing
        X[sample_idx, missing_mask] = np.nan

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add metadata
    adata.obs['replicate_group'] = [f"group_{i//3}" for i in range(n_samples)]
    adata.obs['batch'] = ['batch1'] * 16 + ['batch2'] * 16 + ['batch3'] * 16
    adata.obs['condition'] = ['control'] * 24 + ['treatment'] * 24

    return adata


@pytest.fixture
def mock_adata_with_contaminants():
    """Create mock AnnData with contaminant proteins."""
    n_samples, n_proteins = 30, 50
    X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]

    # Create protein names with some contaminants
    protein_names = []
    for i in range(n_proteins):
        if i < 5:
            protein_names.append(f"KRT{i}_HUMAN")  # Keratin contaminants
        elif i < 8:
            protein_names.append(f"CON_{i}_BOVIN")  # Common contaminants
        elif i < 10:
            protein_names.append(f"REV_{i}_HUMAN")  # Reverse hits
        else:
            protein_names.append(f"PROT{i}_HUMAN")  # Normal proteins

    adata.var_names = protein_names
    adata.var['protein_names'] = protein_names

    return adata


@pytest.fixture
def mock_adata_with_replicates():
    """Create mock AnnData with technical replicates."""
    n_replicates, n_proteins = 12, 50  # 4 groups × 3 replicates each
    X = np.random.lognormal(mean=9, sigma=0.5, size=(n_replicates, n_proteins))

    # Add small variations between replicates in same group
    for group in range(4):
        group_start = group * 3
        group_end = (group + 1) * 3
        # Make replicates similar within each group
        group_mean = np.mean(X[group_start:group_end, :], axis=0)
        for rep in range(3):
            rep_idx = group_start + rep
            # Add small noise around group mean
            X[rep_idx, :] = group_mean + np.random.normal(0, group_mean * 0.1, n_proteins)

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_replicates)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    # Add replicate information
    adata.obs['replicate_group'] = ['group_A'] * 3 + ['group_B'] * 3 + ['group_C'] * 3 + ['group_D'] * 3
    adata.obs['replicate_id'] = ['rep1', 'rep2', 'rep3'] * 4

    return adata


@pytest.fixture
def mock_adata_with_outliers():
    """Create mock AnnData with outlier samples."""
    n_samples, n_proteins = 24, 100
    X = np.random.lognormal(mean=8, sigma=0.8, size=(n_samples, n_proteins))

    # Create outlier samples
    outlier_indices = [5, 15, 20]
    for idx in outlier_indices:
        # Make these samples very different (10x higher intensity)
        X[idx, :] = X[idx, :] * 10

    adata = ad.AnnData(X=X)
    adata.obs_names = [f"sample_{i}" for i in range(n_samples)]
    adata.var_names = [f"protein_{i}" for i in range(n_proteins)]

    return adata


# ===============================================================================
# Service Initialization Tests
# ===============================================================================

class TestProteomicsQualityServiceInitialization:
    """Test suite for ProteomicsQualityService initialization."""

    def test_init_default_parameters(self):
        """Test service initialization with default parameters."""
        service = ProteomicsQualityService()

        assert service is not None
        assert hasattr(service, 'contaminant_patterns')
        assert 'keratin' in service.contaminant_patterns
        assert 'common_contaminants' in service.contaminant_patterns
        assert 'reverse_hits' in service.contaminant_patterns


# ===============================================================================
# Missing Value Pattern Analysis Tests
# ===============================================================================

class TestMissingValuePatterns:
    """Test suite for missing value pattern analysis functionality."""

    def test_assess_missing_value_patterns_basic(self, service, mock_adata_with_missing):
        """Test basic missing value pattern assessment."""
        result_adata, stats = service.assess_missing_value_patterns(
            mock_adata_with_missing,
            sample_threshold=0.7,
            protein_threshold=0.8
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert 'total_missing_percentage' in stats
        assert 'n_high_missing_samples' in stats
        assert 'n_high_missing_proteins' in stats
        assert 'missing_value_patterns' in stats
        assert stats['analysis_type'] == 'missing_value_assessment'

        # QC metrics should be added to observations and variables
        assert 'missing_value_percentage' in result_adata.obs.columns
        assert 'missing_value_percentage' in result_adata.var.columns
        assert 'high_missing_sample' in result_adata.obs.columns
        assert 'high_missing_protein' in result_adata.var.columns

    def test_assess_missing_value_patterns_custom_thresholds(self, service, mock_adata_with_missing):
        """Test missing value assessment with custom thresholds."""
        result_adata, stats = service.assess_missing_value_patterns(
            mock_adata_with_missing,
            sample_threshold=0.5,
            protein_threshold=0.6
        )

        assert result_adata is not None
        assert stats['sample_threshold'] == 0.5
        assert stats['protein_threshold'] == 0.6

    def test_assess_missing_value_patterns_no_missing(self, service):
        """Test missing value assessment with no missing values."""
        X = np.random.lognormal(mean=8, sigma=1, size=(20, 30))
        adata = ad.AnnData(X=X)

        result_adata, stats = service.assess_missing_value_patterns(adata)

        assert result_adata is not None
        assert stats['total_missing_percentage'] == 0.0
        assert stats['n_high_missing_samples'] == 0
        assert stats['n_high_missing_proteins'] == 0

    def test_assess_missing_value_patterns_all_missing_protein(self, service):
        """Test missing value assessment with proteins that are all missing."""
        X = np.random.lognormal(mean=8, sigma=1, size=(20, 10))
        X[:, 0] = np.nan  # First protein all missing
        X[:, 1] = np.nan  # Second protein all missing

        adata = ad.AnnData(X=X)

        result_adata, stats = service.assess_missing_value_patterns(adata)

        assert result_adata is not None
        assert stats['n_high_missing_proteins'] >= 2

    def test_missing_value_pattern_identification(self, service, mock_adata_with_missing):
        """Test identification of specific missing value patterns."""
        result_adata, stats = service.assess_missing_value_patterns(mock_adata_with_missing)

        patterns = stats['missing_value_patterns']
        assert 'sample_wise' in patterns
        assert 'protein_wise' in patterns
        assert 'total_patterns' in patterns


# ===============================================================================
# Coefficient of Variation Assessment Tests
# ===============================================================================

class TestCoefficientVariation:
    """Test suite for coefficient of variation assessment functionality."""

    def test_assess_coefficient_variation_basic(self, service, mock_adata_with_replicates):
        """Test basic coefficient of variation assessment."""
        result_adata, stats = service.assess_coefficient_variation(
            mock_adata_with_replicates,
            replicate_column='replicate_group'
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['analysis_type'] == 'coefficient_variation_assessment'
        assert 'mean_cv_across_proteins' in stats
        assert 'median_cv_across_proteins' in stats
        assert 'n_high_cv_proteins' in stats

        # CV metrics should be added to variables
        assert 'cv_mean' in result_adata.var.columns
        assert 'cv_median' in result_adata.var.columns
        assert 'high_cv_protein' in result_adata.var.columns

    def test_assess_coefficient_variation_custom_threshold(self, service, mock_adata_with_replicates):
        """Test CV assessment with custom CV threshold."""
        result_adata, stats = service.assess_coefficient_variation(
            mock_adata_with_replicates,
            replicate_column='replicate_group',
            cv_threshold=0.15
        )

        assert result_adata is not None
        assert stats['cv_threshold'] == 0.15

    def test_assess_coefficient_variation_no_replicate_column(self, service, mock_adata_with_replicates):
        """Test CV assessment without replicate column (overall CV)."""
        result_adata, stats = service.assess_coefficient_variation(
            mock_adata_with_replicates,
            replicate_column=None
        )

        assert result_adata is not None
        assert 'cv_overall' in result_adata.var.columns

    def test_assess_coefficient_variation_invalid_column(self, service, mock_adata_with_replicates):
        """Test CV assessment with invalid replicate column."""
        with pytest.raises(ProteomicsQualityError) as exc_info:
            service.assess_coefficient_variation(
                mock_adata_with_replicates,
                replicate_column='nonexistent_column'
            )

        assert "Replicate column 'nonexistent_column' not found" in str(exc_info.value)

    def test_coefficient_variation_calculation_accuracy(self, service):
        """Test accuracy of CV calculation."""
        # Create data with known CV values
        X = np.array([
            [100, 200],  # CV = 0.5, 0.5
            [150, 300],  # for each protein
            [50, 100]
        ])
        adata = ad.AnnData(X=X)
        adata.obs['replicate_group'] = ['group_A'] * 3

        result_adata, stats = service.assess_coefficient_variation(
            adata,
            replicate_column='replicate_group'
        )

        # Check that CV calculations are reasonable
        assert 'cv_mean' in result_adata.var.columns
        assert all(cv > 0 for cv in result_adata.var['cv_mean'])


# ===============================================================================
# Contaminant Detection Tests
# ===============================================================================

class TestContaminantDetection:
    """Test suite for contaminant detection functionality."""

    def test_detect_contaminants_basic(self, service, mock_adata_with_contaminants):
        """Test basic contaminant detection."""
        result_adata, stats = service.detect_contaminants(mock_adata_with_contaminants)

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['analysis_type'] == 'contaminant_detection'
        assert 'total_contaminants_detected' in stats
        assert 'contaminant_types' in stats

        # Contaminant flags should be added to variables
        assert 'is_contaminant' in result_adata.var.columns
        assert 'contaminant_type' in result_adata.var.columns

    def test_detect_contaminants_custom_patterns(self, service, mock_adata_with_contaminants):
        """Test contaminant detection with custom patterns."""
        custom_patterns = {
            'custom_contam': ['CUSTOM_', 'TEST_']
        }

        result_adata, stats = service.detect_contaminants(
            mock_adata_with_contaminants,
            custom_patterns=custom_patterns
        )

        assert result_adata is not None
        assert 'custom_contam' in stats['contaminant_types']

    def test_detect_contaminants_protein_name_column(self, service):
        """Test contaminant detection using protein_name column."""
        X = np.random.lognormal(mean=8, sigma=1, size=(10, 5))
        adata = ad.AnnData(X=X)
        adata.var_names = ['PROT1', 'PROT2', 'PROT3', 'PROT4', 'PROT5']
        adata.var['protein_names'] = ['KRT1_HUMAN', 'CON_ALB_BOVIN', 'NORMAL_PROT', 'REV_TRYP', 'ANOTHER_PROT']

        result_adata, stats = service.detect_contaminants(
            adata,
            protein_name_column='protein_names'
        )

        assert result_adata is not None
        # Should detect keratin, common contaminant, and reverse hit
        assert stats['total_contaminants_detected'] >= 3

    def test_detect_contaminants_no_contaminants(self, service):
        """Test contaminant detection when no contaminants are present."""
        X = np.random.lognormal(mean=8, sigma=1, size=(10, 5))
        adata = ad.AnnData(X=X)
        adata.var_names = ['NORMAL_PROT1', 'NORMAL_PROT2', 'CLEAN_PROT3', 'GOOD_PROT4', 'REGULAR_PROT5']

        result_adata, stats = service.detect_contaminants(adata)

        assert result_adata is not None
        assert stats['total_contaminants_detected'] == 0

    def test_contaminant_detection_statistics(self, service, mock_adata_with_contaminants):
        """Test contaminant detection statistics accuracy."""
        result_adata, stats = service.detect_contaminants(mock_adata_with_contaminants)

        # Check that detected contaminants match expected patterns
        contaminant_mask = result_adata.var['is_contaminant']
        detected_count = contaminant_mask.sum()

        assert stats['total_contaminants_detected'] == detected_count
        assert stats['contaminant_percentage'] == (detected_count / len(contaminant_mask)) * 100


# ===============================================================================
# Dynamic Range Evaluation Tests
# ===============================================================================

class TestDynamicRangeEvaluation:
    """Test suite for dynamic range evaluation functionality."""

    def test_evaluate_dynamic_range_basic(self, service, mock_adata_with_missing):
        """Test basic dynamic range evaluation."""
        result_adata, stats = service.evaluate_dynamic_range(mock_adata_with_missing)

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['analysis_type'] == 'dynamic_range_evaluation'
        assert 'overall_dynamic_range_log10' in stats
        assert 'median_dynamic_range_log10' in stats
        assert 'intensity_distribution' in stats

        # Dynamic range metrics should be added to variables
        assert 'dynamic_range_log10' in result_adata.var.columns
        assert 'min_intensity' in result_adata.var.columns
        assert 'max_intensity' in result_adata.var.columns

    def test_evaluate_dynamic_range_sample_wise(self, service, mock_adata_with_missing):
        """Test sample-wise dynamic range evaluation."""
        result_adata, stats = service.evaluate_dynamic_range(
            mock_adata_with_missing,
            calculate_sample_wise=True
        )

        assert result_adata is not None
        # Sample-wise metrics should be added
        assert 'dynamic_range_log10' in result_adata.obs.columns
        assert 'sample_wise_ranges' in stats

    def test_evaluate_dynamic_range_with_zeros(self, service):
        """Test dynamic range evaluation with zero values."""
        X = np.random.lognormal(mean=5, sigma=1, size=(20, 10))
        X[0:5, 0:3] = 0  # Add some zero values

        adata = ad.AnnData(X=X)

        result_adata, stats = service.evaluate_dynamic_range(adata)

        assert result_adata is not None
        # Should handle zero values gracefully

    def test_dynamic_range_calculation_accuracy(self, service):
        """Test accuracy of dynamic range calculation."""
        # Create data with known dynamic range
        X = np.array([
            [1, 10, 100],      # Dynamic ranges: 2, 1, 1 (log10)
            [10, 100, 1000],   # Dynamic ranges: 1, 1, 1
            [100, 10, 100]     # Dynamic ranges: 0, 0, 0
        ])
        adata = ad.AnnData(X=X)

        result_adata, stats = service.evaluate_dynamic_range(adata)

        # Check dynamic range calculations
        expected_ranges = [2.0, 1.0, 2.0]  # log10(100/1), log10(100/10), log10(1000/100)
        calculated_ranges = result_adata.var['dynamic_range_log10'].values

        assert len(calculated_ranges) == 3
        for calc, exp in zip(calculated_ranges, expected_ranges):
            assert abs(calc - exp) < 0.1  # Allow small floating point differences


# ===============================================================================
# PCA Outlier Detection Tests
# ===============================================================================

class TestPCAOutlierDetection:
    """Test suite for PCA outlier detection functionality."""

    def test_detect_pca_outliers_basic(self, service, mock_adata_with_outliers):
        """Test basic PCA outlier detection."""
        result_adata, stats = service.detect_pca_outliers(mock_adata_with_outliers)

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['analysis_type'] == 'pca_outlier_detection'
        assert 'n_outliers_detected' in stats
        assert 'outlier_threshold' in stats

        # PCA and outlier information should be added
        assert 'X_pca' in result_adata.obsm
        assert 'is_outlier' in result_adata.obs.columns
        assert 'outlier_score' in result_adata.obs.columns

    def test_detect_pca_outliers_custom_components(self, service, mock_adata_with_outliers):
        """Test PCA outlier detection with custom number of components."""
        result_adata, stats = service.detect_pca_outliers(
            mock_adata_with_outliers,
            n_components=5
        )

        assert result_adata is not None
        assert result_adata.obsm['X_pca'].shape[1] == 5
        assert stats['n_components'] == 5

    def test_detect_pca_outliers_custom_threshold(self, service, mock_adata_with_outliers):
        """Test PCA outlier detection with custom threshold."""
        result_adata, stats = service.detect_pca_outliers(
            mock_adata_with_outliers,
            outlier_threshold=2.0
        )

        assert result_adata is not None
        assert stats['outlier_threshold'] == 2.0

    def test_detect_pca_outliers_no_outliers(self, service):
        """Test PCA outlier detection when no outliers are present."""
        # Create very uniform data
        X = np.random.normal(loc=1000, scale=50, size=(20, 50))
        adata = ad.AnnData(X=X)

        result_adata, stats = service.detect_pca_outliers(adata)

        assert result_adata is not None
        # Should detect few or no outliers
        assert stats['n_outliers_detected'] <= 2  # Allow for some statistical variation

    def test_pca_outlier_detection_with_missing_values(self, service, mock_adata_with_missing):
        """Test PCA outlier detection with missing values (should handle via imputation)."""
        result_adata, stats = service.detect_pca_outliers(mock_adata_with_missing)

        assert result_adata is not None
        assert 'X_pca' in result_adata.obsm
        # Should complete without errors despite missing values

    def test_pca_variance_calculation(self, service, mock_adata_with_outliers):
        """Test PCA variance explanation calculation."""
        result_adata, stats = service.detect_pca_outliers(mock_adata_with_outliers)

        assert 'pca_variance_explained' in stats
        assert 'cumulative_variance_explained' in stats
        # Variance explained should be reasonable
        assert 0 < stats['pca_variance_explained'] <= 1
        assert 0 < stats['cumulative_variance_explained'] <= 1


# ===============================================================================
# Technical Replicate Assessment Tests
# ===============================================================================

class TestTechnicalReplicateAssessment:
    """Test suite for technical replicate assessment functionality."""

    def test_assess_technical_replicates_basic(self, service, mock_adata_with_replicates):
        """Test basic technical replicate assessment."""
        result_adata, stats = service.assess_technical_replicates(
            mock_adata_with_replicates,
            replicate_column='replicate_group'
        )

        assert result_adata is not None
        assert isinstance(stats, dict)
        assert stats['analysis_type'] == 'technical_replicate_assessment'
        assert 'n_replicate_groups' in stats
        assert 'mean_replicate_correlation' in stats
        assert 'replicate_group_stats' in stats

        # Replicate metrics should be added to observations
        assert 'replicate_correlation' in result_adata.obs.columns
        assert 'replicate_quality' in result_adata.obs.columns

    def test_assess_technical_replicates_custom_min_replicates(self, service, mock_adata_with_replicates):
        """Test replicate assessment with custom minimum replicates requirement."""
        result_adata, stats = service.assess_technical_replicates(
            mock_adata_with_replicates,
            replicate_column='replicate_group',
            min_replicates=2
        )

        assert result_adata is not None
        assert stats['min_replicates'] == 2

    def test_assess_technical_replicates_custom_correlation_threshold(self, service, mock_adata_with_replicates):
        """Test replicate assessment with custom correlation threshold."""
        result_adata, stats = service.assess_technical_replicates(
            mock_adata_with_replicates,
            replicate_column='replicate_group',
            correlation_threshold=0.9
        )

        assert result_adata is not None
        assert stats['correlation_threshold'] == 0.9

    def test_assess_technical_replicates_invalid_column(self, service, mock_adata_with_replicates):
        """Test replicate assessment with invalid replicate column."""
        with pytest.raises(ProteomicsQualityError) as exc_info:
            service.assess_technical_replicates(
                mock_adata_with_replicates,
                replicate_column='nonexistent_column'
            )

        assert "Replicate column 'nonexistent_column' not found" in str(exc_info.value)

    def test_assess_technical_replicates_insufficient_replicates(self, service):
        """Test replicate assessment with insufficient replicates."""
        # Create data with single samples (no replicates)
        X = np.random.lognormal(mean=8, sigma=1, size=(5, 20))
        adata = ad.AnnData(X=X)
        adata.obs['replicate_group'] = ['group_A', 'group_B', 'group_C', 'group_D', 'group_E']

        result_adata, stats = service.assess_technical_replicates(
            adata,
            replicate_column='replicate_group',
            min_replicates=2
        )

        assert result_adata is not None
        assert stats['n_valid_replicate_groups'] == 0  # No groups meet minimum requirement

    def test_replicate_correlation_calculation(self, service):
        """Test accuracy of replicate correlation calculation."""
        # Create highly correlated replicates
        base_profile = np.random.lognormal(mean=8, sigma=1, size=20)
        X = np.array([
            base_profile + np.random.normal(0, base_profile * 0.05, 20),  # rep1
            base_profile + np.random.normal(0, base_profile * 0.05, 20),  # rep2
            base_profile + np.random.normal(0, base_profile * 0.05, 20),  # rep3
        ])

        adata = ad.AnnData(X=X)
        adata.obs['replicate_group'] = ['group_A'] * 3

        result_adata, stats = service.assess_technical_replicates(
            adata,
            replicate_column='replicate_group'
        )

        # Should detect high correlation
        assert stats['mean_replicate_correlation'] > 0.8


# ===============================================================================
# Helper Methods Tests
# ===============================================================================

class TestHelperMethods:
    """Test suite for helper methods."""

    def test_identify_missing_patterns_helper(self, service):
        """Test missing pattern identification helper method."""
        # Create structured missing data
        is_missing = np.array([
            [True, False, False],
            [True, True, False],
            [False, True, True],
            [False, False, True]
        ])

        patterns = service._identify_missing_patterns(is_missing)

        assert isinstance(patterns, dict)
        assert 'sample_wise' in patterns
        assert 'protein_wise' in patterns
        assert 'total_patterns' in patterns


# ===============================================================================
# Error Handling and Edge Cases Tests
# ===============================================================================

class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""

    def test_error_handling_empty_data(self, service):
        """Test error handling with empty data."""
        adata = ad.AnnData(X=np.array([]).reshape(0, 0))

        with pytest.raises(ProteomicsQualityError):
            service.assess_missing_value_patterns(adata)

    def test_single_sample_quality_assessment(self, service):
        """Test quality assessment with single sample."""
        X = np.random.lognormal(mean=8, sigma=1, size=(1, 20))
        adata = ad.AnnData(X=X)

        # Should handle single sample gracefully
        result_adata, stats = service.assess_missing_value_patterns(adata)
        assert result_adata is not None

        result_adata, stats = service.evaluate_dynamic_range(adata)
        assert result_adata is not None

    def test_single_protein_quality_assessment(self, service):
        """Test quality assessment with single protein."""
        X = np.random.lognormal(mean=8, sigma=1, size=(20, 1))
        adata = ad.AnnData(X=X)

        result_adata, stats = service.assess_missing_value_patterns(adata)
        assert result_adata is not None

    def test_all_missing_data(self, service):
        """Test quality assessment with all missing data."""
        X = np.full((10, 5), np.nan)
        adata = ad.AnnData(X=X)

        result_adata, stats = service.assess_missing_value_patterns(adata)
        assert result_adata is not None
        assert stats['total_missing_percentage'] == 100.0

    def test_no_variation_data_pca(self, service):
        """Test PCA outlier detection with no variation data."""
        X = np.ones((10, 20)) * 1000  # All same value
        adata = ad.AnnData(X=X)

        # Should handle gracefully
        result_adata, stats = service.detect_pca_outliers(adata)
        assert result_adata is not None

    def test_insufficient_data_for_pca(self, service):
        """Test PCA with insufficient data dimensions."""
        X = np.random.lognormal(mean=8, sigma=1, size=(3, 50))  # Fewer samples than features
        adata = ad.AnnData(X=X)

        result_adata, stats = service.detect_pca_outliers(adata, n_components=2)
        assert result_adata is not None


# ===============================================================================
# Integration Tests
# ===============================================================================

class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    def test_complete_quality_assessment_workflow(self, service, mock_adata_with_missing):
        """Test complete quality assessment workflow."""
        # Add additional metadata for complete testing
        mock_adata_with_missing.obs['replicate_group'] = [f"group_{i//3}" for i in range(mock_adata_with_missing.n_obs)]

        # Step 1: Missing value assessment
        adata_mv, _ = service.assess_missing_value_patterns(mock_adata_with_missing)

        # Step 2: CV assessment
        adata_cv, _ = service.assess_coefficient_variation(
            adata_mv,
            replicate_column='replicate_group'
        )

        # Step 3: Contaminant detection
        adata_cont, _ = service.detect_contaminants(adata_cv)

        # Step 4: Dynamic range evaluation
        adata_dr, _ = service.evaluate_dynamic_range(adata_cont)

        # Step 5: PCA outlier detection
        adata_pca, _ = service.detect_pca_outliers(adata_dr)

        # Step 6: Replicate assessment
        adata_final, _ = service.assess_technical_replicates(
            adata_pca,
            replicate_column='replicate_group'
        )

        # Verify final result has all QC components
        assert 'missing_value_percentage' in adata_final.obs.columns
        assert 'cv_mean' in adata_final.var.columns
        assert 'is_contaminant' in adata_final.var.columns
        assert 'dynamic_range_log10' in adata_final.var.columns
        assert 'is_outlier' in adata_final.obs.columns
        assert 'replicate_correlation' in adata_final.obs.columns

    def test_quality_metrics_preservation(self, service, mock_adata_with_missing):
        """Test that quality metrics are preserved across multiple assessments."""
        # Run multiple quality assessments
        adata_step1, _ = service.assess_missing_value_patterns(mock_adata_with_missing)
        adata_step2, _ = service.evaluate_dynamic_range(adata_step1)

        # Metrics from step 1 should be preserved in step 2
        assert 'missing_value_percentage' in adata_step2.obs.columns
        assert 'missing_value_percentage' in adata_step2.var.columns
        assert 'dynamic_range_log10' in adata_step2.var.columns

    def test_consistent_quality_thresholds(self, service, mock_adata_with_missing):
        """Test consistency of quality thresholds across different datasets."""
        # Test with different thresholds
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = {}

        for threshold in thresholds:
            result_adata, stats = service.assess_missing_value_patterns(
                mock_adata_with_missing,
                sample_threshold=threshold
            )
            results[threshold] = stats['n_high_missing_samples']

        # Higher thresholds should generally result in fewer flagged samples
        assert results[0.9] <= results[0.7] <= results[0.5]


# ===============================================================================
# Performance and Memory Tests
# ===============================================================================

class TestPerformanceAndMemory:
    """Test suite for performance and memory considerations."""

    @pytest.mark.slow
    def test_large_dataset_quality_assessment(self, service):
        """Test quality assessment with large dataset."""
        # Create larger dataset
        n_samples, n_proteins = 200, 1000
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))
        # Add some missing values
        missing_mask = np.random.rand(n_samples, n_proteins) < 0.1
        X[missing_mask] = np.nan

        adata = ad.AnnData(X=X)

        result_adata, stats = service.assess_missing_value_patterns(adata)

        assert result_adata is not None
        assert stats['samples_processed'] == n_samples
        assert stats['proteins_processed'] == n_proteins

    @pytest.mark.slow
    def test_memory_efficient_pca_outlier_detection(self, service):
        """Test memory efficiency in PCA outlier detection."""
        # Create moderately large dataset
        n_samples, n_proteins = 100, 500
        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))
        adata = ad.AnnData(X=X)

        result_adata, stats = service.detect_pca_outliers(adata, n_components=20)

        assert result_adata is not None
        # Should complete without memory errors

    def test_efficient_contaminant_detection(self, service):
        """Test efficient contaminant detection with many proteins."""
        n_samples, n_proteins = 50, 5000
        protein_names = [f"PROT{i}_HUMAN" for i in range(n_proteins)]
        # Add some contaminants
        for i in range(0, 100, 10):
            protein_names[i] = f"KRT{i}_HUMAN"

        X = np.random.lognormal(mean=8, sigma=1, size=(n_samples, n_proteins))
        adata = ad.AnnData(X=X)
        adata.var_names = protein_names

        result_adata, stats = service.detect_contaminants(adata)

        assert result_adata is not None
        assert stats['total_contaminants_detected'] >= 10