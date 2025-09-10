"""
Proteomics quality control service for comprehensive quality assessment and validation.

This service implements professional-grade quality control methods specifically designed for 
proteomics data including missing value pattern analysis, contaminant detection, CV assessment,
dynamic range evaluation, and technical replicate validation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Set

import anndata
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsQualityError(Exception):
    """Base exception for proteomics quality control operations."""
    pass


class ProteomicsQualityService:
    """
    Advanced quality control service for proteomics data.
    
    This stateless service provides comprehensive quality assessment methods
    following best practices from proteomics analysis pipelines. Handles missing value
    pattern analysis, contaminant detection, coefficient of variation assessment,
    and technical replicate validation.
    """

    def __init__(self):
        """
        Initialize the proteomics quality service.
        
        This service is stateless and doesn't require a data manager instance.
        """
        logger.info("Initializing stateless ProteomicsQualityService")
        
        # Define common contaminant patterns
        self.contaminant_patterns = {
            'keratin': ['KRT', 'KERATIN', 'KER_'],
            'common_contaminants': ['CON_', 'CONTAM_', 'CONT_'],
            'reverse_hits': ['REV_', 'REVERSE_', 'rev_'],
            'trypsin': ['TRYP_', 'TRYPSIN'],
            'albumin': ['ALB_', 'ALBUMIN', 'BSA'],
            'immunoglobulin': ['IGG_', 'IGH_', 'IGL_', 'IGK_']
        }
        
        logger.info("ProteomicsQualityService initialized successfully")

    def assess_missing_value_patterns(
        self,
        adata: anndata.AnnData,
        sample_threshold: float = 0.7,
        protein_threshold: float = 0.8
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Analyze missing value patterns in proteomics data.
        
        Args:
            adata: AnnData object with proteomics data
            sample_threshold: Threshold for high missing value samples
            protein_threshold: Threshold for high missing value proteins
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with QC metrics and analysis stats
            
        Raises:
            ProteomicsQualityError: If assessment fails
        """
        try:
            logger.info("Starting missing value pattern analysis")
            
            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            X = adata_qc.X.copy()
            
            # Calculate missing value statistics
            if hasattr(X, 'isnan'):
                is_missing = np.isnan(X)
                total_missing = is_missing.sum()
                total_values = X.size
            else:
                # Assume missing values are represented as zeros or very small values
                is_missing = (X == 0) | (X < 1e-10)
                total_missing = is_missing.sum()
                total_values = X.size

            # Sample-level missing value analysis
            sample_missing_counts = is_missing.sum(axis=1)
            sample_missing_rates = sample_missing_counts / adata_qc.n_vars
            
            # Protein-level missing value analysis
            protein_missing_counts = is_missing.sum(axis=0)
            protein_missing_rates = protein_missing_counts / adata_qc.n_obs
            
            # Add QC metrics to observations (samples)
            adata_qc.obs['missing_protein_count'] = sample_missing_counts
            adata_qc.obs['missing_protein_rate'] = sample_missing_rates
            adata_qc.obs['high_missing_sample'] = sample_missing_rates > sample_threshold
            adata_qc.obs['detected_protein_count'] = adata_qc.n_vars - sample_missing_counts
            
            # Add QC metrics to variables (proteins)  
            adata_qc.var['missing_sample_count'] = protein_missing_counts
            adata_qc.var['missing_sample_rate'] = protein_missing_rates
            adata_qc.var['high_missing_protein'] = protein_missing_rates > protein_threshold
            adata_qc.var['detected_sample_count'] = adata_qc.n_obs - protein_missing_counts

            # Identify missing value patterns
            missing_patterns = self._identify_missing_patterns(is_missing)
            
            # Calculate missing value statistics
            missing_stats = {
                "total_missing_values": int(total_missing),
                "total_possible_values": int(total_values),
                "overall_missing_rate": float(total_missing / total_values),
                "high_missing_samples": int((sample_missing_rates > sample_threshold).sum()),
                "high_missing_proteins": int((protein_missing_rates > protein_threshold).sum()),
                "median_missing_rate_samples": float(np.median(sample_missing_rates)),
                "median_missing_rate_proteins": float(np.median(protein_missing_rates)),
                "samples_processed": adata_qc.n_obs,
                "proteins_processed": adata_qc.n_vars,
                "analysis_type": "missing_value_patterns",
                **missing_patterns
            }

            logger.info(f"Missing value analysis completed: {total_missing:,} missing values ({(total_missing/total_values)*100:.1f}%)")
            return adata_qc, missing_stats

        except Exception as e:
            logger.exception(f"Error in missing value pattern analysis: {e}")
            raise ProteomicsQualityError(f"Missing value pattern analysis failed: {str(e)}")

    def assess_coefficient_variation(
        self,
        adata: anndata.AnnData,
        cv_threshold: float = 50.0,
        min_observations: int = 3
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Assess coefficient of variation (CV) for proteins and samples.
        
        Args:
            adata: AnnData object with proteomics data
            cv_threshold: Threshold for high CV proteins (%)
            min_observations: Minimum observations required for CV calculation
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with CV metrics and analysis stats
            
        Raises:
            ProteomicsQualityError: If assessment fails
        """
        try:
            logger.info("Starting coefficient of variation assessment")
            
            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            X = adata_qc.X.copy()
            
            # Calculate protein-level CVs
            protein_cvs = []
            protein_means = []
            protein_stds = []
            
            for i in range(adata_qc.n_vars):
                protein_values = X[:, i]
                
                # Remove missing values for CV calculation
                if hasattr(protein_values, 'isnan'):
                    valid_values = protein_values[~np.isnan(protein_values)]
                else:
                    valid_values = protein_values[protein_values > 0]
                
                if len(valid_values) >= min_observations:
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)
                    cv_val = (std_val / mean_val) * 100 if mean_val > 0 else np.inf
                else:
                    mean_val = np.nan
                    std_val = np.nan
                    cv_val = np.nan
                
                protein_means.append(mean_val)
                protein_stds.append(std_val)
                protein_cvs.append(cv_val)
            
            # Calculate sample-level CVs (across proteins)
            sample_cvs = []
            sample_means = []
            sample_stds = []
            
            for i in range(adata_qc.n_obs):
                sample_values = X[i, :]
                
                # Remove missing values for CV calculation
                if hasattr(sample_values, 'isnan'):
                    valid_values = sample_values[~np.isnan(sample_values)]
                else:
                    valid_values = sample_values[sample_values > 0]
                
                if len(valid_values) >= min_observations:
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)
                    cv_val = (std_val / mean_val) * 100 if mean_val > 0 else np.inf
                else:
                    mean_val = np.nan
                    std_val = np.nan
                    cv_val = np.nan
                
                sample_means.append(mean_val)
                sample_stds.append(std_val)
                sample_cvs.append(cv_val)
            
            # Add CV metrics to observations (samples)
            adata_qc.obs['intensity_mean'] = sample_means
            adata_qc.obs['intensity_std'] = sample_stds
            adata_qc.obs['intensity_cv'] = sample_cvs
            adata_qc.obs['high_cv_sample'] = np.array(sample_cvs) > cv_threshold
            
            # Add CV metrics to variables (proteins)
            adata_qc.var['intensity_mean'] = protein_means
            adata_qc.var['intensity_std'] = protein_stds
            adata_qc.var['intensity_cv'] = protein_cvs
            adata_qc.var['high_cv_protein'] = np.array(protein_cvs) > cv_threshold

            # Calculate CV statistics
            valid_protein_cvs = [cv for cv in protein_cvs if not np.isnan(cv) and not np.isinf(cv)]
            valid_sample_cvs = [cv for cv in sample_cvs if not np.isnan(cv) and not np.isinf(cv)]
            
            cv_stats = {
                "median_protein_cv": float(np.median(valid_protein_cvs)) if valid_protein_cvs else np.nan,
                "mean_protein_cv": float(np.mean(valid_protein_cvs)) if valid_protein_cvs else np.nan,
                "high_cv_proteins": int(np.sum(np.array(protein_cvs) > cv_threshold)),
                "median_sample_cv": float(np.median(valid_sample_cvs)) if valid_sample_cvs else np.nan,
                "mean_sample_cv": float(np.mean(valid_sample_cvs)) if valid_sample_cvs else np.nan,
                "high_cv_samples": int(np.sum(np.array(sample_cvs) > cv_threshold)),
                "cv_threshold": cv_threshold,
                "min_observations": min_observations,
                "samples_processed": adata_qc.n_obs,
                "proteins_processed": adata_qc.n_vars,
                "analysis_type": "coefficient_variation"
            }

            logger.info(f"CV assessment completed: median protein CV = {cv_stats['median_protein_cv']:.1f}%")
            return adata_qc, cv_stats

        except Exception as e:
            logger.exception(f"Error in coefficient of variation assessment: {e}")
            raise ProteomicsQualityError(f"Coefficient of variation assessment failed: {str(e)}")

    def detect_contaminants(
        self,
        adata: anndata.AnnData,
        protein_id_column: str = None,
        custom_patterns: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Detect contaminant proteins based on naming patterns.
        
        Args:
            adata: AnnData object with proteomics data
            protein_id_column: Column in var containing protein IDs (uses index if None)
            custom_patterns: Custom contaminant patterns to add to defaults
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with contaminant flags and analysis stats
            
        Raises:
            ProteomicsQualityError: If detection fails
        """
        try:
            logger.info("Starting contaminant detection")
            
            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            # Get protein identifiers
            if protein_id_column and protein_id_column in adata_qc.var.columns:
                protein_ids = adata_qc.var[protein_id_column].astype(str)
            else:
                protein_ids = adata_qc.var_names.astype(str)

            # Combine default and custom patterns
            contaminant_patterns = self.contaminant_patterns.copy()
            if custom_patterns:
                contaminant_patterns.update(custom_patterns)

            # Initialize contaminant flags
            contaminant_flags = {}
            for contaminant_type in contaminant_patterns.keys():
                contaminant_flags[f'is_{contaminant_type}'] = np.zeros(adata_qc.n_vars, dtype=bool)

            # Check each protein against contaminant patterns
            contaminant_counts = {}
            for contaminant_type, patterns in contaminant_patterns.items():
                count = 0
                for i, protein_id in enumerate(protein_ids):
                    protein_id_upper = protein_id.upper()
                    for pattern in patterns:
                        if pattern.upper() in protein_id_upper:
                            contaminant_flags[f'is_{contaminant_type}'][i] = True
                            count += 1
                            break
                contaminant_counts[contaminant_type] = count

            # Add contaminant flags to var
            for flag_name, flag_values in contaminant_flags.items():
                adata_qc.var[flag_name] = flag_values

            # Create overall contaminant flag
            overall_contaminant = np.zeros(adata_qc.n_vars, dtype=bool)
            for flag_values in contaminant_flags.values():
                overall_contaminant |= flag_values
            adata_qc.var['is_contaminant'] = overall_contaminant

            # Calculate contaminant statistics
            total_contaminants = overall_contaminant.sum()
            contaminant_percentage = (total_contaminants / adata_qc.n_vars) * 100

            contaminant_stats = {
                "total_contaminants": int(total_contaminants),
                "contaminant_percentage": float(contaminant_percentage),
                "contaminant_counts_by_type": contaminant_counts,
                "patterns_used": {k: len(v) for k, v in contaminant_patterns.items()},
                "samples_processed": adata_qc.n_obs,
                "proteins_processed": adata_qc.n_vars,
                "analysis_type": "contaminant_detection"
            }

            logger.info(f"Contaminant detection completed: {total_contaminants} contaminants ({contaminant_percentage:.1f}%)")
            return adata_qc, contaminant_stats

        except Exception as e:
            logger.exception(f"Error in contaminant detection: {e}")
            raise ProteomicsQualityError(f"Contaminant detection failed: {str(e)}")

    def evaluate_dynamic_range(
        self,
        adata: anndata.AnnData,
        percentile_low: float = 5.0,
        percentile_high: float = 95.0
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Evaluate dynamic range of proteomics measurements.
        
        Args:
            adata: AnnData object with proteomics data
            percentile_low: Lower percentile for dynamic range calculation
            percentile_high: Higher percentile for dynamic range calculation
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with dynamic range metrics and analysis stats
            
        Raises:
            ProteomicsQualityError: If evaluation fails
        """
        try:
            logger.info("Starting dynamic range evaluation")
            
            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            X = adata_qc.X.copy()
            
            # Calculate dynamic range metrics per sample
            sample_dynamic_ranges = []
            sample_intensity_ranges = []
            sample_percentiles_low = []
            sample_percentiles_high = []
            
            for i in range(adata_qc.n_obs):
                sample_values = X[i, :]
                
                # Remove missing/zero values
                if hasattr(sample_values, 'isnan'):
                    valid_values = sample_values[~np.isnan(sample_values) & (sample_values > 0)]
                else:
                    valid_values = sample_values[sample_values > 0]
                
                if len(valid_values) > 0:
                    p_low = np.percentile(valid_values, percentile_low)
                    p_high = np.percentile(valid_values, percentile_high)
                    dynamic_range = np.log10(p_high / p_low) if p_low > 0 else np.nan
                    intensity_range = p_high - p_low
                else:
                    p_low = np.nan
                    p_high = np.nan
                    dynamic_range = np.nan
                    intensity_range = np.nan
                
                sample_dynamic_ranges.append(dynamic_range)
                sample_intensity_ranges.append(intensity_range)
                sample_percentiles_low.append(p_low)
                sample_percentiles_high.append(p_high)
            
            # Calculate dynamic range metrics per protein
            protein_dynamic_ranges = []
            protein_intensity_ranges = []
            protein_percentiles_low = []
            protein_percentiles_high = []
            
            for j in range(adata_qc.n_vars):
                protein_values = X[:, j]
                
                # Remove missing/zero values
                if hasattr(protein_values, 'isnan'):
                    valid_values = protein_values[~np.isnan(protein_values) & (protein_values > 0)]
                else:
                    valid_values = protein_values[protein_values > 0]
                
                if len(valid_values) > 0:
                    p_low = np.percentile(valid_values, percentile_low)
                    p_high = np.percentile(valid_values, percentile_high)
                    dynamic_range = np.log10(p_high / p_low) if p_low > 0 else np.nan
                    intensity_range = p_high - p_low
                else:
                    p_low = np.nan
                    p_high = np.nan
                    dynamic_range = np.nan
                    intensity_range = np.nan
                
                protein_dynamic_ranges.append(dynamic_range)
                protein_intensity_ranges.append(intensity_range)
                protein_percentiles_low.append(p_low)
                protein_percentiles_high.append(p_high)

            # Add dynamic range metrics to observations (samples)
            adata_qc.obs['dynamic_range_log10'] = sample_dynamic_ranges
            adata_qc.obs['intensity_range'] = sample_intensity_ranges
            adata_qc.obs[f'percentile_{percentile_low}'] = sample_percentiles_low
            adata_qc.obs[f'percentile_{percentile_high}'] = sample_percentiles_high
            
            # Add dynamic range metrics to variables (proteins)
            adata_qc.var['dynamic_range_log10'] = protein_dynamic_ranges
            adata_qc.var['intensity_range'] = protein_intensity_ranges
            adata_qc.var[f'percentile_{percentile_low}'] = protein_percentiles_low
            adata_qc.var[f'percentile_{percentile_high}'] = protein_percentiles_high

            # Calculate overall dynamic range statistics
            valid_sample_ranges = [dr for dr in sample_dynamic_ranges if not np.isnan(dr)]
            valid_protein_ranges = [dr for dr in protein_dynamic_ranges if not np.isnan(dr)]
            
            dynamic_range_stats = {
                "median_sample_dynamic_range": float(np.median(valid_sample_ranges)) if valid_sample_ranges else np.nan,
                "mean_sample_dynamic_range": float(np.mean(valid_sample_ranges)) if valid_sample_ranges else np.nan,
                "median_protein_dynamic_range": float(np.median(valid_protein_ranges)) if valid_protein_ranges else np.nan,
                "mean_protein_dynamic_range": float(np.mean(valid_protein_ranges)) if valid_protein_ranges else np.nan,
                "percentile_low": percentile_low,
                "percentile_high": percentile_high,
                "samples_processed": adata_qc.n_obs,
                "proteins_processed": adata_qc.n_vars,
                "analysis_type": "dynamic_range_evaluation"
            }

            logger.info(f"Dynamic range evaluation completed: median sample range = {dynamic_range_stats['median_sample_dynamic_range']:.2f} log10")
            return adata_qc, dynamic_range_stats

        except Exception as e:
            logger.exception(f"Error in dynamic range evaluation: {e}")
            raise ProteomicsQualityError(f"Dynamic range evaluation failed: {str(e)}")

    def detect_pca_outliers(
        self,
        adata: anndata.AnnData,
        n_components: int = 50,
        outlier_threshold: float = 3.0
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Detect outlier samples using PCA analysis.
        
        Args:
            adata: AnnData object with proteomics data
            n_components: Number of PCA components to compute
            outlier_threshold: Threshold in standard deviations for outlier detection
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with outlier flags and analysis stats
            
        Raises:
            ProteomicsQualityError: If detection fails
        """
        try:
            logger.info("Starting PCA outlier detection")
            
            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            X = adata_qc.X.copy()
            
            # Prepare data for PCA (handle missing values)
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # Perform PCA
            n_components_actual = min(n_components, min(X_scaled.shape) - 1)
            pca = PCA(n_components=n_components_actual)
            X_pca = pca.fit_transform(X_scaled)
            
            # Store PCA results
            adata_qc.obsm['X_pca'] = X_pca
            adata_qc.uns['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'explained_variance': pca.explained_variance_,
                'components': pca.components_
            }
            
            # Detect outliers using Mahalanobis distance approximation
            # Calculate distance from center for each sample
            pc_distances = []
            for i in range(adata_qc.n_obs):
                # Calculate distance in PC space (first few components)
                pc_coords = X_pca[i, :min(10, n_components_actual)]  # Use first 10 PCs
                distance = np.sqrt(np.sum(pc_coords**2))
                pc_distances.append(distance)
            
            pc_distances = np.array(pc_distances)
            
            # Identify outliers based on distance threshold
            mean_distance = np.mean(pc_distances)
            std_distance = np.std(pc_distances)
            outlier_cutoff = mean_distance + outlier_threshold * std_distance
            
            is_outlier = pc_distances > outlier_cutoff
            
            # Add outlier information to observations
            adata_qc.obs['pca_distance'] = pc_distances
            adata_qc.obs['is_pca_outlier'] = is_outlier
            adata_qc.obs['outlier_score'] = (pc_distances - mean_distance) / std_distance

            # Calculate PCA statistics
            variance_explained_top10 = pca.explained_variance_ratio_[:10].sum()
            n_outliers = is_outlier.sum()
            
            pca_stats = {
                "n_components_computed": n_components_actual,
                "variance_explained_top10": float(variance_explained_top10),
                "n_outliers_detected": int(n_outliers),
                "outlier_percentage": float((n_outliers / adata_qc.n_obs) * 100),
                "outlier_threshold": outlier_threshold,
                "mean_pca_distance": float(mean_distance),
                "std_pca_distance": float(std_distance),
                "samples_processed": adata_qc.n_obs,
                "proteins_processed": adata_qc.n_vars,
                "analysis_type": "pca_outlier_detection"
            }

            logger.info(f"PCA outlier detection completed: {n_outliers} outliers ({(n_outliers/adata_qc.n_obs)*100:.1f}%)")
            return adata_qc, pca_stats

        except Exception as e:
            logger.exception(f"Error in PCA outlier detection: {e}")
            raise ProteomicsQualityError(f"PCA outlier detection failed: {str(e)}")

    def assess_technical_replicates(
        self,
        adata: anndata.AnnData,
        replicate_column: str,
        correlation_method: str = "pearson"
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Assess technical replicate reproducibility and variation.
        
        Args:
            adata: AnnData object with proteomics data
            replicate_column: Column in obs identifying technical replicates
            correlation_method: Method for correlation analysis ('pearson', 'spearman')
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with replicate metrics and analysis stats
            
        Raises:
            ProteomicsQualityError: If assessment fails
        """
        try:
            logger.info("Starting technical replicate assessment")
            
            # Create working copy
            adata_qc = adata.copy()
            original_shape = adata_qc.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            if replicate_column not in adata_qc.obs.columns:
                raise ProteomicsQualityError(f"Replicate column '{replicate_column}' not found in obs")

            X = adata_qc.X.copy()
            replicate_groups = adata_qc.obs[replicate_column]
            
            # Find replicate groups
            unique_groups = replicate_groups.unique()
            replicate_correlations = []
            replicate_cvs = []
            group_sizes = []
            
            for group in unique_groups:
                group_mask = replicate_groups == group
                group_samples = X[group_mask, :]
                group_size = group_samples.shape[0]
                group_sizes.append(group_size)
                
                if group_size < 2:
                    continue  # Need at least 2 replicates for correlation
                
                # Calculate pairwise correlations within group
                group_correlations = []
                for i in range(group_size):
                    for j in range(i + 1, group_size):
                        sample1 = group_samples[i, :]
                        sample2 = group_samples[j, :]
                        
                        # Remove missing values for correlation
                        if hasattr(sample1, 'isnan'):
                            valid_mask = ~(np.isnan(sample1) | np.isnan(sample2))
                        else:
                            valid_mask = (sample1 > 0) & (sample2 > 0)
                        
                        if valid_mask.sum() > 10:  # Need at least 10 points for correlation
                            if correlation_method == "pearson":
                                corr, _ = stats.pearsonr(sample1[valid_mask], sample2[valid_mask])
                            else:  # spearman
                                corr, _ = stats.spearmanr(sample1[valid_mask], sample2[valid_mask])
                            group_correlations.append(corr)
                
                replicate_correlations.extend(group_correlations)
                
                # Calculate CV within replicate group for each protein
                group_cvs_per_protein = []
                for j in range(adata_qc.n_vars):
                    protein_values = group_samples[:, j]
                    if hasattr(protein_values, 'isnan'):
                        valid_values = protein_values[~np.isnan(protein_values)]
                    else:
                        valid_values = protein_values[protein_values > 0]
                    
                    if len(valid_values) >= 2:
                        mean_val = np.mean(valid_values)
                        std_val = np.std(valid_values)
                        cv_val = (std_val / mean_val) * 100 if mean_val > 0 else np.nan
                        group_cvs_per_protein.append(cv_val)
                
                replicate_cvs.extend(group_cvs_per_protein)

            # Calculate replicate statistics
            valid_correlations = [corr for corr in replicate_correlations if not np.isnan(corr)]
            valid_cvs = [cv for cv in replicate_cvs if not np.isnan(cv)]
            
            replicate_stats = {
                "n_replicate_groups": len(unique_groups),
                "group_sizes": group_sizes,
                "median_replicate_correlation": float(np.median(valid_correlations)) if valid_correlations else np.nan,
                "mean_replicate_correlation": float(np.mean(valid_correlations)) if valid_correlations else np.nan,
                "median_replicate_cv": float(np.median(valid_cvs)) if valid_cvs else np.nan,
                "mean_replicate_cv": float(np.mean(valid_cvs)) if valid_cvs else np.nan,
                "correlation_method": correlation_method,
                "samples_processed": adata_qc.n_obs,
                "proteins_processed": adata_qc.n_vars,
                "analysis_type": "technical_replicate_assessment"
            }

            # Add replicate quality flags
            adata_qc.obs['replicate_group'] = replicate_groups
            adata_qc.obs['group_size'] = adata_qc.obs[replicate_column].map(
                replicate_groups.value_counts().to_dict()
            )

            logger.info(f"Technical replicate assessment completed: {len(unique_groups)} replicate groups")
            return adata_qc, replicate_stats

        except Exception as e:
            logger.exception(f"Error in technical replicate assessment: {e}")
            raise ProteomicsQualityError(f"Technical replicate assessment failed: {str(e)}")

    # Helper methods
    def _identify_missing_patterns(self, is_missing: np.ndarray) -> Dict[str, Any]:
        """Identify common missing value patterns."""
        n_samples, n_proteins = is_missing.shape
        
        # Pattern 1: Completely missing proteins
        completely_missing_proteins = np.sum(is_missing, axis=0) == n_samples
        
        # Pattern 2: Completely missing samples
        completely_missing_samples = np.sum(is_missing, axis=1) == n_proteins
        
        # Pattern 3: Block missing patterns (simple heuristic)
        missing_blocks = 0
        for i in range(n_samples):
            # Count consecutive missing values
            missing_runs = []
            current_run = 0
            for j in range(n_proteins):
                if is_missing[i, j]:
                    current_run += 1
                else:
                    if current_run > 0:
                        missing_runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                missing_runs.append(current_run)
            
            # Count blocks of 10+ consecutive missing values
            missing_blocks += sum(1 for run in missing_runs if run >= 10)
        
        return {
            "completely_missing_proteins": int(completely_missing_proteins.sum()),
            "completely_missing_samples": int(completely_missing_samples.sum()),
            "estimated_missing_blocks": missing_blocks,
            "random_missing_pattern": missing_blocks < n_samples * 0.1  # Heuristic
        }
