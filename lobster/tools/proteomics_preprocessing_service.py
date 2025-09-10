"""
Proteomics preprocessing service for missing value imputation, normalization, and batch correction.

This service implements professional-grade preprocessing methods specifically designed for 
proteomics data including MNAR imputation, proteomics-specific normalization methods,
and batch correction techniques suitable for mass spectrometry data.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from scipy import stats
from scipy.stats import rankdata

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsPreprocessingError(Exception):
    """Base exception for proteomics preprocessing operations."""
    pass


class ProteomicsPreprocessingService:
    """
    Advanced preprocessing service for proteomics data.
    
    This stateless service provides methods for missing value imputation, normalization,
    and batch correction following best practices from proteomics analysis pipelines.
    Handles the unique challenges of proteomics data including high missing value rates,
    intensity-dependent noise, and batch effects.
    """

    def __init__(self):
        """
        Initialize the proteomics preprocessing service.
        
        This service is stateless and doesn't require a data manager instance.
        """
        logger.info("Initializing stateless ProteomicsPreprocessingService")
        logger.info("ProteomicsPreprocessingService initialized successfully")

    def impute_missing_values(
        self,
        adata: anndata.AnnData,
        method: str = "mixed",
        knn_neighbors: int = 5,
        min_prob_percentile: float = 2.5,
        mnar_width: float = 0.3,
        mnar_downshift: float = 1.8
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Impute missing values using proteomics-appropriate methods.
        
        Args:
            adata: AnnData object with proteomics data
            method: Method ('knn', 'min_prob', 'mnar', 'mixed')
            knn_neighbors: Number of neighbors for KNN imputation
            min_prob_percentile: Percentile for minimum probability imputation
            mnar_width: Width parameter for MNAR distribution
            mnar_downshift: Downshift parameter for MNAR distribution
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: Imputed AnnData and processing stats
            
        Raises:
            ProteomicsPreprocessingError: If imputation fails
        """
        try:
            logger.info(f"Starting missing value imputation with method: {method}")
            
            # Create working copy
            adata_imputed = adata.copy()
            original_shape = adata_imputed.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            # Store original data for comparison
            if adata_imputed.raw is None:
                adata_imputed.raw = adata_imputed.copy()

            # Check for missing values
            X = adata_imputed.X.copy()
            if not hasattr(X, 'isnan') or not np.isnan(X).any():
                logger.info("No missing values detected, skipping imputation")
                return adata_imputed, {
                    "method": method,
                    "missing_values_found": False,
                    "imputation_performed": False,
                    "analysis_type": "missing_value_imputation"
                }

            # Calculate missing value statistics
            total_missing = np.isnan(X).sum()
            total_values = X.size
            missing_percentage = (total_missing / total_values) * 100
            
            logger.info(f"Missing values: {total_missing:,} ({missing_percentage:.1f}%)")

            # Apply imputation method
            if method == "knn":
                X_imputed = self._knn_imputation(X, knn_neighbors)
            elif method == "min_prob":
                X_imputed = self._min_prob_imputation(X, min_prob_percentile)
            elif method == "mnar":
                X_imputed = self._mnar_imputation(X, mnar_width, mnar_downshift)
            elif method == "mixed":
                X_imputed = self._mixed_imputation(
                    X, knn_neighbors, min_prob_percentile, mnar_width, mnar_downshift
                )
            else:
                raise ProteomicsPreprocessingError(f"Unknown imputation method: {method}")

            # Update the data
            adata_imputed.X = X_imputed

            # Calculate imputation statistics
            imputation_stats = {
                "method": method,
                "missing_values_found": True,
                "imputation_performed": True,
                "original_missing_count": int(total_missing),
                "original_missing_percentage": float(missing_percentage),
                "remaining_missing_count": int(np.isnan(X_imputed).sum()),
                "proteins_processed": adata_imputed.n_vars,
                "samples_processed": adata_imputed.n_obs,
                "knn_neighbors": knn_neighbors if method in ["knn", "mixed"] else None,
                "min_prob_percentile": min_prob_percentile if method in ["min_prob", "mixed"] else None,
                "analysis_type": "missing_value_imputation"
            }

            logger.info(f"Imputation completed: {total_missing:,} → {np.isnan(X_imputed).sum():,} missing values")
            return adata_imputed, imputation_stats

        except Exception as e:
            logger.exception(f"Error in missing value imputation: {e}")
            raise ProteomicsPreprocessingError(f"Missing value imputation failed: {str(e)}")

    def normalize_intensities(
        self,
        adata: anndata.AnnData,
        method: str = "median",
        log_transform: bool = True,
        pseudocount_strategy: str = "adaptive",
        reference_sample: Optional[str] = None
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Normalize proteomics intensity data using appropriate methods.
        
        Args:
            adata: AnnData object with proteomics data
            method: Normalization method ('median', 'quantile', 'vsn', 'total_sum')
            log_transform: Whether to apply log2 transformation
            pseudocount_strategy: Strategy for pseudocount ('adaptive', 'fixed', 'min_observed')
            reference_sample: Reference sample for normalization (if applicable)
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: Normalized AnnData and processing stats
            
        Raises:
            ProteomicsPreprocessingError: If normalization fails
        """
        try:
            logger.info(f"Starting intensity normalization with method: {method}")
            
            # Create working copy
            adata_norm = adata.copy()
            original_shape = adata_norm.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            # Store raw data if not already stored
            if adata_norm.raw is None:
                adata_norm.raw = adata_norm.copy()

            X = adata_norm.X.copy()
            
            # Check for negative values
            if np.any(X < 0):
                logger.warning("Negative values detected in data - may indicate pre-processed data")

            # Apply normalization
            if method == "median":
                X_norm = self._median_normalization(X)
            elif method == "quantile":
                X_norm = self._quantile_normalization(X)
            elif method == "vsn":
                X_norm = self._vsn_normalization(X)
            elif method == "total_sum":
                X_norm = self._total_sum_normalization(X)
            else:
                raise ProteomicsPreprocessingError(f"Unknown normalization method: {method}")

            # Apply log transformation if requested
            log_stats = {}
            if log_transform:
                X_norm, log_stats = self._apply_log_transformation(X_norm, pseudocount_strategy)

            # Update the data
            adata_norm.X = X_norm
            
            # Store in layers
            adata_norm.layers['normalized'] = X_norm
            if log_transform:
                adata_norm.layers['log2_normalized'] = X_norm

            # Calculate normalization statistics
            normalization_stats = {
                "method": method,
                "log_transform": log_transform,
                "pseudocount_strategy": pseudocount_strategy if log_transform else None,
                "samples_processed": adata_norm.n_obs,
                "proteins_processed": adata_norm.n_vars,
                "median_intensity_before": float(np.nanmedian(adata_norm.raw.X)),
                "median_intensity_after": float(np.nanmedian(X_norm)),
                "analysis_type": "intensity_normalization",
                **log_stats
            }

            logger.info(f"Normalization completed: {method} method applied")
            return adata_norm, normalization_stats

        except Exception as e:
            logger.exception(f"Error in intensity normalization: {e}")
            raise ProteomicsPreprocessingError(f"Intensity normalization failed: {str(e)}")

    def correct_batch_effects(
        self,
        adata: anndata.AnnData,
        batch_key: str,
        method: str = "combat",
        n_pcs: int = 50,
        reference_batch: Optional[str] = None
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Correct for batch effects in proteomics data.
        
        Args:
            adata: AnnData object with proteomics data
            batch_key: Column in obs containing batch information
            method: Batch correction method ('combat', 'median_centering', 'reference_based')
            n_pcs: Number of principal components for analysis
            reference_batch: Reference batch for correction (if applicable)
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: Batch-corrected AnnData and processing stats
            
        Raises:
            ProteomicsPreprocessingError: If batch correction fails
        """
        try:
            logger.info(f"Starting batch correction with method: {method}")
            
            # Create working copy
            adata_corrected = adata.copy()
            original_shape = adata_corrected.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            # Check batch information
            if batch_key not in adata_corrected.obs.columns:
                raise ProteomicsPreprocessingError(f"Batch key '{batch_key}' not found in obs")

            batch_counts = adata_corrected.obs[batch_key].value_counts().to_dict()
            n_batches = len(batch_counts)
            logger.info(f"Found {n_batches} batches: {batch_counts}")

            if n_batches < 2:
                logger.warning("Less than 2 batches found, skipping batch correction")
                return adata_corrected, {
                    "method": method,
                    "batch_correction_performed": False,
                    "n_batches": n_batches,
                    "analysis_type": "batch_correction"
                }

            # Store original data
            if adata_corrected.raw is None:
                adata_corrected.raw = adata_corrected.copy()

            X = adata_corrected.X.copy()

            # Apply batch correction method
            if method == "combat":
                X_corrected = self._combat_correction(X, adata_corrected.obs[batch_key])
            elif method == "median_centering":
                X_corrected = self._median_centering_correction(X, adata_corrected.obs[batch_key])
            elif method == "reference_based":
                X_corrected = self._reference_based_correction(
                    X, adata_corrected.obs[batch_key], reference_batch
                )
            else:
                raise ProteomicsPreprocessingError(f"Unknown batch correction method: {method}")

            # Update the data
            adata_corrected.X = X_corrected
            adata_corrected.layers['batch_corrected'] = X_corrected

            # Calculate batch correction statistics
            batch_stats = self._calculate_batch_correction_stats(
                adata_corrected.raw.X, X_corrected, adata_corrected.obs[batch_key], n_pcs
            )

            correction_stats = {
                "method": method,
                "batch_correction_performed": True,
                "batch_key": batch_key,
                "n_batches": n_batches,
                "batch_counts": batch_counts,
                "reference_batch": reference_batch,
                "samples_processed": adata_corrected.n_obs,
                "proteins_processed": adata_corrected.n_vars,
                "analysis_type": "batch_correction",
                **batch_stats
            }

            logger.info(f"Batch correction completed: {method} method applied")
            return adata_corrected, correction_stats

        except Exception as e:
            logger.exception(f"Error in batch correction: {e}")
            raise ProteomicsPreprocessingError(f"Batch correction failed: {str(e)}")

    # Helper methods for missing value imputation
    def _knn_imputation(self, X: np.ndarray, n_neighbors: int) -> np.ndarray:
        """Apply KNN imputation."""
        logger.info(f"Applying KNN imputation with {n_neighbors} neighbors")
        imputer = KNNImputer(n_neighbors=n_neighbors)
        return imputer.fit_transform(X)

    def _min_prob_imputation(self, X: np.ndarray, percentile: float) -> np.ndarray:
        """Apply minimum probability imputation."""
        logger.info(f"Applying minimum probability imputation at {percentile}th percentile")
        X_imputed = X.copy()
        
        for i in range(X.shape[1]):  # For each protein
            protein_values = X[:, i]
            observed_values = protein_values[~np.isnan(protein_values)]
            
            if len(observed_values) > 0:
                # Use percentile of observed values as imputation value
                impute_value = np.percentile(observed_values, percentile)
                X_imputed[np.isnan(protein_values), i] = impute_value
        
        return X_imputed

    def _mnar_imputation(self, X: np.ndarray, width: float, downshift: float) -> np.ndarray:
        """Apply MNAR (Missing Not At Random) imputation."""
        logger.info(f"Applying MNAR imputation (width={width}, downshift={downshift})")
        X_imputed = X.copy()
        
        for i in range(X.shape[1]):  # For each protein
            protein_values = X[:, i]
            observed_values = protein_values[~np.isnan(protein_values)]
            
            if len(observed_values) > 0:
                # Create left-truncated normal distribution
                mean_obs = np.mean(observed_values)
                std_obs = np.std(observed_values)
                
                # Parameters for MNAR distribution
                mnar_mean = mean_obs - downshift * std_obs
                mnar_std = width * std_obs
                
                # Generate random values from truncated normal
                n_missing = np.isnan(protein_values).sum()
                if n_missing > 0:
                    impute_values = np.random.normal(mnar_mean, mnar_std, n_missing)
                    # Ensure values are below the minimum observed value
                    min_obs = np.min(observed_values)
                    impute_values = np.minimum(impute_values, min_obs - 0.1 * std_obs)
                    X_imputed[np.isnan(protein_values), i] = impute_values
        
        return X_imputed

    def _mixed_imputation(
        self, X: np.ndarray, knn_neighbors: int, min_prob_percentile: float,
        mnar_width: float, mnar_downshift: float, mcar_threshold: float = 0.4
    ) -> np.ndarray:
        """Apply mixed imputation strategy based on missing value patterns."""
        logger.info("Applying mixed imputation strategy")
        X_imputed = X.copy()
        
        # Calculate missing percentages per protein
        missing_per_protein = np.isnan(X).sum(axis=0) / X.shape[0]
        
        for i in range(X.shape[1]):
            protein_values = X[:, i]
            missing_pct = missing_per_protein[i]
            
            if np.isnan(protein_values).any():
                if missing_pct < mcar_threshold:
                    # Low missing rate: use KNN (assume MCAR)
                    protein_data = X[:, [i]]
                    imputer = KNNImputer(n_neighbors=min(knn_neighbors, X.shape[0]-1))
                    X_imputed[:, [i]] = imputer.fit_transform(protein_data)
                else:
                    # High missing rate: use MNAR approach
                    observed_values = protein_values[~np.isnan(protein_values)]
                    if len(observed_values) > 0:
                        mean_obs = np.mean(observed_values)
                        std_obs = np.std(observed_values)
                        mnar_mean = mean_obs - mnar_downshift * std_obs
                        mnar_std = mnar_width * std_obs
                        
                        n_missing = np.isnan(protein_values).sum()
                        impute_values = np.random.normal(mnar_mean, mnar_std, n_missing)
                        X_imputed[np.isnan(protein_values), i] = impute_values
        
        return X_imputed

    # Helper methods for normalization
    def _median_normalization(self, X: np.ndarray) -> np.ndarray:
        """Apply median normalization."""
        logger.info("Applying median normalization")
        sample_medians = np.nanmedian(X, axis=1)
        global_median = np.nanmedian(sample_medians)
        
        # Avoid division by zero
        sample_medians[sample_medians == 0] = 1
        normalization_factors = global_median / sample_medians
        
        return X * normalization_factors[:, np.newaxis]

    def _quantile_normalization(self, X: np.ndarray) -> np.ndarray:
        """Apply quantile normalization."""
        logger.info("Applying quantile normalization")
        
        # Handle missing values by working with ranks
        X_norm = X.copy()
        
        # Get ranks for each sample (handling NaN)
        ranks = np.zeros_like(X)
        for i in range(X.shape[0]):
            sample_data = X[i, :]
            valid_mask = ~np.isnan(sample_data)
            if valid_mask.sum() > 0:
                ranks[i, valid_mask] = rankdata(sample_data[valid_mask])
        
        # Calculate mean values at each rank
        max_rank = int(np.nanmax(ranks))
        mean_values = np.zeros(max_rank + 1)
        
        for rank in range(1, max_rank + 1):
            rank_values = []
            for i in range(X.shape[0]):
                rank_positions = ranks[i, :] == rank
                if rank_positions.any():
                    rank_values.extend(X[i, rank_positions])
            if rank_values:
                mean_values[rank] = np.mean(rank_values)
        
        # Apply quantile normalization
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if not np.isnan(X[i, j]):
                    rank = int(ranks[i, j])
                    X_norm[i, j] = mean_values[rank]
        
        return X_norm

    def _vsn_normalization(self, X: np.ndarray) -> np.ndarray:
        """Apply variance stabilizing normalization (VSN-like)."""
        logger.info("Applying VSN-like normalization")
        
        # Simple VSN approximation: asinh transformation with scaling
        X_positive = np.maximum(X, 1e-8)  # Avoid log of zero
        return np.arcsinh(X_positive / 2)

    def _total_sum_normalization(self, X: np.ndarray) -> np.ndarray:
        """Apply total sum normalization."""
        logger.info("Applying total sum normalization")
        sample_sums = np.nansum(X, axis=1)
        sample_sums[sample_sums == 0] = 1  # Avoid division by zero
        return (X / sample_sums[:, np.newaxis]) * 1e6  # Scale to 1M

    def _apply_log_transformation(self, X: np.ndarray, strategy: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply log2 transformation with appropriate pseudocount."""
        if strategy == "adaptive":
            # Use fraction of minimum positive value
            min_positive = np.nanmin(X[X > 0]) if np.any(X > 0) else 1.0
            pseudocount = min_positive * 0.1
        elif strategy == "fixed":
            pseudocount = 1.0
        elif strategy == "min_observed":
            pseudocount = np.nanmin(X[X > 0]) if np.any(X > 0) else 1.0
        else:
            pseudocount = 1.0

        X_log = np.log2(X + pseudocount)
        
        log_stats = {
            "pseudocount": float(pseudocount),
            "pseudocount_strategy": strategy,
            "median_after_log": float(np.nanmedian(X_log))
        }
        
        return X_log, log_stats

    # Helper methods for batch correction
    def _combat_correction(self, X: np.ndarray, batch_labels: pd.Series) -> np.ndarray:
        """Apply ComBat-like batch correction."""
        logger.info("Applying ComBat-like batch correction")
        
        X_corrected = X.copy()
        unique_batches = batch_labels.unique()
        
        # Calculate overall mean for each protein
        overall_means = np.nanmean(X, axis=0)
        
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_data = X[batch_mask, :]
            
            # Calculate batch-specific means and standard deviations
            batch_means = np.nanmean(batch_data, axis=0)
            batch_stds = np.nanstd(batch_data, axis=0)
            batch_stds[batch_stds == 0] = 1  # Avoid division by zero
            
            # Apply correction: standardize within batch, then scale to overall distribution
            for i in np.where(batch_mask)[0]:
                X_corrected[i, :] = ((X[i, :] - batch_means) / batch_stds) * np.nanstd(X, axis=0) + overall_means
        
        return X_corrected

    def _median_centering_correction(self, X: np.ndarray, batch_labels: pd.Series) -> np.ndarray:
        """Apply median centering batch correction."""
        logger.info("Applying median centering batch correction")
        
        X_corrected = X.copy()
        unique_batches = batch_labels.unique()
        overall_median = np.nanmedian(X, axis=0)
        
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            batch_data = X[batch_mask, :]
            batch_median = np.nanmedian(batch_data, axis=0)
            
            # Apply median centering
            correction = overall_median - batch_median
            X_corrected[batch_mask, :] = batch_data + correction
        
        return X_corrected

    def _reference_based_correction(
        self, X: np.ndarray, batch_labels: pd.Series, reference_batch: Optional[str]
    ) -> np.ndarray:
        """Apply reference-based batch correction."""
        logger.info(f"Applying reference-based batch correction (reference: {reference_batch})")
        
        X_corrected = X.copy()
        unique_batches = batch_labels.unique()
        
        # Determine reference batch
        if reference_batch is None or reference_batch not in unique_batches:
            # Use batch with most samples as reference
            batch_counts = batch_labels.value_counts()
            reference_batch = batch_counts.index[0]
            logger.info(f"Using batch with most samples as reference: {reference_batch}")
        
        # Get reference batch statistics
        ref_mask = batch_labels == reference_batch
        ref_data = X[ref_mask, :]
        ref_median = np.nanmedian(ref_data, axis=0)
        
        # Correct other batches to match reference
        for batch in unique_batches:
            if batch != reference_batch:
                batch_mask = batch_labels == batch
                batch_data = X[batch_mask, :]
                batch_median = np.nanmedian(batch_data, axis=0)
                
                correction = ref_median - batch_median
                X_corrected[batch_mask, :] = batch_data + correction
        
        return X_corrected

    def _calculate_batch_correction_stats(
        self, X_before: np.ndarray, X_after: np.ndarray, batch_labels: pd.Series, n_pcs: int
    ) -> Dict[str, Any]:
        """Calculate statistics to assess batch correction effectiveness."""
        
        # PCA before and after correction
        try:
            # Handle missing values for PCA
            imputer = SimpleImputer(strategy='mean')
            X_before_pca = imputer.fit_transform(X_before)
            X_after_pca = imputer.fit_transform(X_after)
            
            pca = PCA(n_components=min(n_pcs, X_before_pca.shape[1]))
            
            # PCA before correction
            pca_before = pca.fit_transform(X_before_pca)
            var_explained_before = pca.explained_variance_ratio_[:3].sum()
            
            # PCA after correction  
            pca_after = pca.fit_transform(X_after_pca)
            var_explained_after = pca.explained_variance_ratio_[:3].sum()
            
            stats = {
                "pca_variance_before": float(var_explained_before),
                "pca_variance_after": float(var_explained_after),
                "correction_effectiveness": "improved" if var_explained_after > var_explained_before else "mixed"
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate PCA statistics: {e}")
            stats = {
                "pca_variance_before": None,
                "pca_variance_after": None,
                "correction_effectiveness": "unknown"
            }
        
        return stats
