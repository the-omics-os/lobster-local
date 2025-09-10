"""
Proteomics differential expression service for protein expression analysis between groups.

This service implements professional-grade differential expression methods specifically designed for 
proteomics data including statistical testing with FDR control, volcano plot generation, 
effect size calculations, and MSstats-like workflows.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Set

import anndata
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from itertools import combinations

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsDifferentialError(Exception):
    """Base exception for proteomics differential expression operations."""
    pass


class ProteomicsDifferentialService:
    """
    Advanced differential expression service for proteomics data.
    
    This stateless service provides comprehensive differential expression analysis methods
    including statistical testing with proper FDR control, effect size calculations,
    volcano plot generation, and MSstats-like workflows for proteomics data.
    """

    def __init__(self):
        """
        Initialize the proteomics differential service.
        
        This service is stateless and doesn't require a data manager instance.
        """
        logger.info("Initializing stateless ProteomicsDifferentialService")
        
        # Define statistical test methods
        self.test_methods = {
            't_test': 'Student\'s t-test',
            'welch_t_test': 'Welch\'s t-test (unequal variances)',
            'mann_whitney': 'Mann-Whitney U test',
            'limma_like': 'Linear model for proteomics',
            'mixed_effects': 'Mixed effects model',
            'anova': 'One-way ANOVA',
            'kruskal_wallis': 'Kruskal-Wallis test'
        }
        
        logger.info("ProteomicsDifferentialService initialized successfully")

    def perform_differential_expression(
        self,
        adata: anndata.AnnData,
        group_column: str,
        comparison_pairs: Optional[List[Tuple[str, str]]] = None,
        test_method: str = "t_test",
        fdr_method: str = "benjamini_hochberg",
        fdr_threshold: float = 0.05,
        fold_change_threshold: float = 1.5,
        min_samples_per_group: int = 3
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Perform comprehensive differential expression analysis between groups.
        
        Args:
            adata: AnnData object with proteomics data
            group_column: Column in obs containing group labels
            comparison_pairs: Specific pairs to compare (if None, does all pairwise)
            test_method: Statistical test method to use
            fdr_method: FDR correction method ('benjamini_hochberg', 'bonferroni', 'holm')
            fdr_threshold: FDR threshold for significance
            fold_change_threshold: Minimum fold change threshold
            min_samples_per_group: Minimum samples required per group
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with DE results and analysis stats
            
        Raises:
            ProteomicsDifferentialError: If analysis fails
        """
        try:
            logger.info(f"Starting differential expression analysis with method: {test_method}")
            
            # Create working copy
            adata_de = adata.copy()
            original_shape = adata_de.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            if group_column not in adata_de.obs.columns:
                raise ProteomicsDifferentialError(f"Group column '{group_column}' not found in obs")

            groups = adata_de.obs[group_column]
            unique_groups = groups.unique()
            
            logger.info(f"Found {len(unique_groups)} groups: {list(unique_groups)}")

            # Determine comparison pairs
            if comparison_pairs is None:
                comparison_pairs = list(combinations(unique_groups, 2))
            
            logger.info(f"Performing {len(comparison_pairs)} comparisons")

            # Perform differential expression analysis
            de_results = self._perform_de_analysis(
                adata_de, groups, comparison_pairs, test_method, 
                fdr_method, min_samples_per_group
            )

            # Filter significant results
            significant_results = self._filter_significant_results(
                de_results, fdr_threshold, fold_change_threshold
            )

            # Store results in AnnData
            adata_de.uns['differential_expression'] = {
                'all_results': de_results,
                'significant_results': significant_results,
                'parameters': {
                    'group_column': group_column,
                    'comparison_pairs': [f"{p[0]}_vs_{p[1]}" for p in comparison_pairs],
                    'test_method': test_method,
                    'fdr_method': fdr_method,
                    'fdr_threshold': fdr_threshold,
                    'fold_change_threshold': fold_change_threshold,
                    'min_samples_per_group': min_samples_per_group
                }
            }

            # Add significance flags to var
            self._add_significance_flags(adata_de, significant_results)

            # Generate volcano plot data
            volcano_data = self._generate_volcano_plot_data(de_results, fdr_threshold, fold_change_threshold)
            adata_de.uns['volcano_plot_data'] = volcano_data

            # Calculate analysis statistics
            de_stats = self._calculate_de_statistics(
                de_results, significant_results, comparison_pairs, adata_de.n_vars
            )

            de_stats.update({
                "test_method": test_method,
                "fdr_method": fdr_method,
                "fdr_threshold": fdr_threshold,
                "fold_change_threshold": fold_change_threshold,
                "group_column": group_column,
                "samples_processed": adata_de.n_obs,
                "proteins_processed": adata_de.n_vars,
                "analysis_type": "differential_expression"
            })

            logger.info(f"Differential expression completed: {len(significant_results)} significant proteins")
            return adata_de, de_stats

        except Exception as e:
            logger.exception(f"Error in differential expression analysis: {e}")
            raise ProteomicsDifferentialError(f"Differential expression analysis failed: {str(e)}")

    def perform_time_course_analysis(
        self,
        adata: anndata.AnnData,
        time_column: str,
        group_column: Optional[str] = None,
        test_method: str = "linear_trend",
        fdr_threshold: float = 0.05
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Perform time course differential expression analysis.
        
        Args:
            adata: AnnData object with proteomics data
            time_column: Column in obs containing time points
            group_column: Optional grouping column for separate time course analysis
            test_method: Method for time course analysis ('linear_trend', 'polynomial', 'spline')
            fdr_threshold: FDR threshold for significance
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with time course results and analysis stats
            
        Raises:
            ProteomicsDifferentialError: If analysis fails
        """
        try:
            logger.info(f"Starting time course analysis with method: {test_method}")
            
            # Create working copy
            adata_tc = adata.copy()
            original_shape = adata_tc.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            if time_column not in adata_tc.obs.columns:
                raise ProteomicsDifferentialError(f"Time column '{time_column}' not found in obs")

            time_points = adata_tc.obs[time_column].astype(float)
            
            # Perform time course analysis
            tc_results = self._perform_time_course_analysis(
                adata_tc, time_points, group_column, test_method
            )

            # Apply FDR correction
            tc_results = self._apply_fdr_correction(tc_results, "benjamini_hochberg")

            # Filter significant results
            significant_tc = [r for r in tc_results if r.get('p_adjusted', 1.0) < fdr_threshold]

            # Store results
            adata_tc.uns['time_course_analysis'] = {
                'results': tc_results,
                'significant_results': significant_tc,
                'parameters': {
                    'time_column': time_column,
                    'group_column': group_column,
                    'test_method': test_method,
                    'fdr_threshold': fdr_threshold
                }
            }

            # Calculate statistics
            tc_stats = {
                "test_method": test_method,
                "time_column": time_column,
                "group_column": group_column,
                "n_time_points": len(time_points.unique()),
                "time_range": (float(time_points.min()), float(time_points.max())),
                "n_tests_performed": len(tc_results),
                "n_significant_results": len(significant_tc),
                "significance_rate": len(significant_tc) / len(tc_results) if tc_results else 0.0,
                "fdr_threshold": fdr_threshold,
                "samples_processed": adata_tc.n_obs,
                "proteins_processed": adata_tc.n_vars,
                "analysis_type": "time_course_analysis"
            }

            logger.info(f"Time course analysis completed: {len(significant_tc)} significant proteins")
            return adata_tc, tc_stats

        except Exception as e:
            logger.exception(f"Error in time course analysis: {e}")
            raise ProteomicsDifferentialError(f"Time course analysis failed: {str(e)}")

    def perform_correlation_analysis(
        self,
        adata: anndata.AnnData,
        target_column: str,
        correlation_method: str = "pearson",
        fdr_threshold: float = 0.05,
        min_correlation: float = 0.3
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Perform correlation analysis between proteins and a continuous variable.
        
        Args:
            adata: AnnData object with proteomics data
            target_column: Column in obs containing continuous target variable
            correlation_method: Correlation method ('pearson', 'spearman', 'kendall')
            fdr_threshold: FDR threshold for significance
            min_correlation: Minimum absolute correlation threshold
            
        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with correlation results and analysis stats
            
        Raises:
            ProteomicsDifferentialError: If analysis fails
        """
        try:
            logger.info(f"Starting correlation analysis with method: {correlation_method}")
            
            # Create working copy
            adata_corr = adata.copy()
            original_shape = adata_corr.shape
            logger.info(f"Input data shape: {original_shape[0]} samples × {original_shape[1]} proteins")

            if target_column not in adata_corr.obs.columns:
                raise ProteomicsDifferentialError(f"Target column '{target_column}' not found in obs")

            target_values = adata_corr.obs[target_column].astype(float)
            
            # Perform correlation analysis
            correlation_results = self._perform_correlation_analysis(
                adata_corr, target_values, correlation_method
            )

            # Apply FDR correction
            correlation_results = self._apply_fdr_correction(correlation_results, "benjamini_hochberg")

            # Filter significant results
            significant_corr = [
                r for r in correlation_results 
                if r.get('p_adjusted', 1.0) < fdr_threshold and abs(r.get('correlation', 0)) >= min_correlation
            ]

            # Store results
            adata_corr.uns['correlation_analysis'] = {
                'results': correlation_results,
                'significant_results': significant_corr,
                'parameters': {
                    'target_column': target_column,
                    'correlation_method': correlation_method,
                    'fdr_threshold': fdr_threshold,
                    'min_correlation': min_correlation
                }
            }

            # Add correlation values to var
            corr_dict = {r['protein']: r['correlation'] for r in correlation_results}
            adata_corr.var['correlation_with_target'] = [
                corr_dict.get(protein, 0.0) for protein in adata_corr.var_names
            ]

            # Calculate statistics
            corr_stats = {
                "correlation_method": correlation_method,
                "target_column": target_column,
                "target_range": (float(target_values.min()), float(target_values.max())),
                "n_tests_performed": len(correlation_results),
                "n_significant_results": len(significant_corr),
                "significance_rate": len(significant_corr) / len(correlation_results) if correlation_results else 0.0,
                "median_abs_correlation": float(np.median([abs(r['correlation']) for r in correlation_results])),
                "max_abs_correlation": float(max([abs(r['correlation']) for r in correlation_results])),
                "fdr_threshold": fdr_threshold,
                "min_correlation": min_correlation,
                "samples_processed": adata_corr.n_obs,
                "proteins_processed": adata_corr.n_vars,
                "analysis_type": "correlation_analysis"
            }

            logger.info(f"Correlation analysis completed: {len(significant_corr)} significant correlations")
            return adata_corr, corr_stats

        except Exception as e:
            logger.exception(f"Error in correlation analysis: {e}")
            raise ProteomicsDifferentialError(f"Correlation analysis failed: {str(e)}")

    # Helper methods for differential expression
    def _perform_de_analysis(
        self, adata: anndata.AnnData, groups: pd.Series, comparison_pairs: List[Tuple[str, str]],
        test_method: str, fdr_method: str, min_samples_per_group: int
    ) -> List[Dict[str, Any]]:
        """Perform differential expression analysis for all comparison pairs."""
        all_results = []
        X = adata.X.copy()
        protein_names = adata.var_names.tolist()

        for group1, group2 in comparison_pairs:
            logger.info(f"Analyzing comparison: {group1} vs {group2}")
            
            # Get group data
            group1_mask = groups == group1
            group2_mask = groups == group2
            
            group1_data = X[group1_mask, :]
            group2_data = X[group2_mask, :]
            
            # Check minimum sample requirements
            if group1_data.shape[0] < min_samples_per_group or group2_data.shape[0] < min_samples_per_group:
                logger.warning(f"Skipping {group1} vs {group2}: insufficient samples")
                continue
            
            # Perform statistical tests for each protein
            comparison_results = self._test_proteins_pairwise(
                group1_data, group2_data, protein_names, group1, group2, test_method
            )
            
            all_results.extend(comparison_results)

        # Apply FDR correction
        if all_results:
            all_results = self._apply_fdr_correction(all_results, fdr_method)

        return all_results

    def _test_proteins_pairwise(
        self, group1_data: np.ndarray, group2_data: np.ndarray, 
        protein_names: List[str], group1: str, group2: str, test_method: str
    ) -> List[Dict[str, Any]]:
        """Test each protein for differential expression between two groups."""
        results = []
        
        for i, protein_name in enumerate(protein_names):
            g1_values = group1_data[:, i]
            g2_values = group2_data[:, i]
            
            # Remove missing values
            if hasattr(g1_values, 'isnan'):
                g1_clean = g1_values[~np.isnan(g1_values)]
                g2_clean = g2_values[~np.isnan(g2_values)]
            else:
                g1_clean = g1_values[g1_values > 0] if test_method != 'limma_like' else g1_values
                g2_clean = g2_values[g2_values > 0] if test_method != 'limma_like' else g2_values
            
            if len(g1_clean) < 2 or len(g2_clean) < 2:
                continue
            
            # Perform statistical test
            test_result = self._perform_statistical_test(g1_clean, g2_clean, test_method)
            
            if test_result is not None:
                # Calculate effect sizes and additional metrics
                effect_metrics = self._calculate_effect_metrics(g1_clean, g2_clean)
                
                result = {
                    'protein': protein_name,
                    'protein_index': i,
                    'group1': group1,
                    'group2': group2,
                    'comparison': f"{group1}_vs_{group2}",
                    'n_group1': len(g1_clean),
                    'n_group2': len(g2_clean),
                    'mean_group1': float(np.mean(g1_clean)),
                    'mean_group2': float(np.mean(g2_clean)),
                    'std_group1': float(np.std(g1_clean)),
                    'std_group2': float(np.std(g2_clean)),
                    **test_result,
                    **effect_metrics
                }
                
                results.append(result)
        
        return results

    def _perform_statistical_test(self, group1: np.ndarray, group2: np.ndarray, method: str) -> Optional[Dict[str, Any]]:
        """Perform statistical test between two groups."""
        try:
            if method == "t_test":
                statistic, p_value = stats.ttest_ind(group1, group2)
            elif method == "welch_t_test":
                statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            elif method == "mann_whitney":
                statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            elif method == "limma_like":
                # Simplified limma-like approach using moderated t-test
                statistic, p_value = self._moderated_t_test(group1, group2)
            else:
                logger.warning(f"Unknown test method: {method}, falling back to t-test")
                statistic, p_value = stats.ttest_ind(group1, group2)
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'test_method': method
            }
            
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            return None

    def _moderated_t_test(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
        """Simplified moderated t-test (limma-like approach)."""
        # This is a simplified version - full limma uses empirical Bayes moderation
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled variance with simple moderation
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        
        # Add small regularization term (simplified empirical Bayes)
        moderated_var = 0.9 * pooled_var + 0.1 * (var1 + var2) / 2
        
        # Calculate t-statistic
        se = np.sqrt(moderated_var * (1/n1 + 1/n2))
        t_stat = (mean1 - mean2) / se
        
        # Degrees of freedom (simplified)
        df = n1 + n2 - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        return t_stat, p_value

    def _calculate_effect_metrics(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
        """Calculate effect size metrics."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Fold change
        fold_change = (mean1 + 1e-8) / (mean2 + 1e-8)  # Add small pseudocount
        log2_fold_change = np.log2(fold_change)
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Hedges' g (corrected effect size)
        correction_factor = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
        hedges_g = cohens_d * correction_factor
        
        return {
            'fold_change': float(fold_change),
            'log2_fold_change': float(log2_fold_change),
            'cohens_d': float(cohens_d),
            'hedges_g': float(hedges_g)
        }

    def _apply_fdr_correction(self, results: List[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
        """Apply FDR correction to p-values."""
        if not results:
            return results
        
        p_values = [r['p_value'] for r in results]
        
        if method == "benjamini_hochberg":
            from statsmodels.stats.multitest import fdrcorrection
            _, p_adjusted = fdrcorrection(p_values, alpha=0.05, method='indep')
        elif method == "bonferroni":
            p_adjusted = [min(p * len(p_values), 1.0) for p in p_values]
        elif method == "holm":
            from statsmodels.stats.multitest import multipletests
            _, p_adjusted, _, _ = multipletests(p_values, method='holm')
        else:
            logger.warning(f"Unknown FDR method: {method}, using Benjamini-Hochberg")
            from statsmodels.stats.multitest import fdrcorrection
            _, p_adjusted = fdrcorrection(p_values, alpha=0.05, method='indep')
        
        # Add adjusted p-values to results
        for i, result in enumerate(results):
            result['p_adjusted'] = float(p_adjusted[i])
        
        return results

    def _filter_significant_results(
        self, results: List[Dict[str, Any]], fdr_threshold: float, fold_change_threshold: float
    ) -> List[Dict[str, Any]]:
        """Filter results for significance."""
        significant = []
        
        for result in results:
            is_significant = (
                result.get('p_adjusted', 1.0) < fdr_threshold and
                abs(result.get('log2_fold_change', 0)) > np.log2(fold_change_threshold)
            )
            
            if is_significant:
                result['is_significant'] = True
                significant.append(result)
        
        return significant

    def _add_significance_flags(self, adata: anndata.AnnData, significant_results: List[Dict[str, Any]]):
        """Add significance flags to var annotations."""
        significant_proteins = set(r['protein'] for r in significant_results)
        
        adata.var['is_de_significant'] = [
            protein in significant_proteins for protein in adata.var_names
        ]
        
        # Add max fold change across comparisons
        max_log2fc_dict = {}
        for result in significant_results:
            protein = result['protein']
            log2fc = abs(result.get('log2_fold_change', 0))
            if protein not in max_log2fc_dict or log2fc > max_log2fc_dict[protein]:
                max_log2fc_dict[protein] = log2fc
        
        adata.var['max_abs_log2_fold_change'] = [
            max_log2fc_dict.get(protein, 0.0) for protein in adata.var_names
        ]

    def _generate_volcano_plot_data(
        self, results: List[Dict[str, Any]], fdr_threshold: float, fold_change_threshold: float
    ) -> Dict[str, Any]:
        """Generate data for volcano plot visualization."""
        volcano_data = {
            'proteins': [],
            'log2_fold_change': [],
            'neg_log10_p_adjusted': [],
            'significance_category': [],
            'comparison': []
        }
        
        log2fc_threshold = np.log2(fold_change_threshold)
        
        for result in results:
            log2fc = result.get('log2_fold_change', 0)
            p_adj = result.get('p_adjusted', 1.0)
            neg_log10_p = -np.log10(max(p_adj, 1e-300))  # Avoid log(0)
            
            # Determine significance category
            if p_adj < fdr_threshold and abs(log2fc) > log2fc_threshold:
                if log2fc > 0:
                    category = 'Upregulated'
                else:
                    category = 'Downregulated'
            elif p_adj < fdr_threshold:
                category = 'Significant (low FC)'
            elif abs(log2fc) > log2fc_threshold:
                category = 'High FC (not significant)'
            else:
                category = 'Not significant'
            
            volcano_data['proteins'].append(result['protein'])
            volcano_data['log2_fold_change'].append(float(log2fc))
            volcano_data['neg_log10_p_adjusted'].append(float(neg_log10_p))
            volcano_data['significance_category'].append(category)
            volcano_data['comparison'].append(result.get('comparison', 'Unknown'))
        
        return volcano_data

    def _calculate_de_statistics(
        self, all_results: List[Dict[str, Any]], significant_results: List[Dict[str, Any]],
        comparison_pairs: List[Tuple[str, str]], total_proteins: int
    ) -> Dict[str, Any]:
        """Calculate differential expression statistics."""
        # Count results by comparison
        comparison_counts = {}
        significant_counts = {}
        
        for result in all_results:
            comp = result.get('comparison', 'Unknown')
            comparison_counts[comp] = comparison_counts.get(comp, 0) + 1
        
        for result in significant_results:
            comp = result.get('comparison', 'Unknown')
            significant_counts[comp] = significant_counts.get(comp, 0) + 1
        
        # Overall statistics
        total_tests = len(all_results)
        total_significant = len(significant_results)
        
        return {
            "n_comparisons": len(comparison_pairs),
            "comparison_pairs": [f"{p[0]}_vs_{p[1]}" for p in comparison_pairs],
            "total_tests_performed": total_tests,
            "total_significant_proteins": total_significant,
            "overall_significance_rate": (total_significant / total_tests) if total_tests > 0 else 0.0,
            "tests_per_comparison": comparison_counts,
            "significant_per_comparison": significant_counts,
            "proteins_tested": total_proteins
        }

    # Helper methods for time course analysis
    def _perform_time_course_analysis(
        self, adata: anndata.AnnData, time_points: pd.Series, 
        group_column: Optional[str], test_method: str
    ) -> List[Dict[str, Any]]:
        """Perform time course analysis."""
        results = []
        X = adata.X.copy()
        protein_names = adata.var_names.tolist()
        
        if group_column is not None:
            groups = adata.obs[group_column].unique()
        else:
            groups = ['all']
        
        for group in groups:
            if group_column is not None:
                group_mask = adata.obs[group_column] == group
                group_X = X[group_mask, :]
                group_time = time_points[group_mask]
            else:
                group_X = X
                group_time = time_points
            
            for i, protein_name in enumerate(protein_names):
                protein_values = group_X[:, i]
                
                # Remove missing values
                if hasattr(protein_values, 'isnan'):
                    valid_mask = ~np.isnan(protein_values)
                else:
                    valid_mask = protein_values > 0
                
                if valid_mask.sum() < 4:  # Need at least 4 points for trend analysis
                    continue
                
                valid_protein = protein_values[valid_mask]
                valid_time = group_time[valid_mask]
                
                # Perform time course test
                if test_method == "linear_trend":
                    slope, p_value, r_squared = self._linear_trend_test(valid_time, valid_protein)
                    test_statistic = slope
                elif test_method == "polynomial":
                    test_statistic, p_value, r_squared = self._polynomial_trend_test(valid_time, valid_protein)
                else:
                    # Default to linear trend
                    slope, p_value, r_squared = self._linear_trend_test(valid_time, valid_protein)
                    test_statistic = slope
                
                results.append({
                    'protein': protein_name,
                    'protein_index': i,
                    'group': group if group_column else 'all',
                    'test_method': test_method,
                    'test_statistic': float(test_statistic),
                    'p_value': float(p_value),
                    'r_squared': float(r_squared),
                    'n_samples': len(valid_protein)
                })
        
        return results

    def _linear_trend_test(self, time_points: np.ndarray, protein_values: np.ndarray) -> Tuple[float, float, float]:
        """Perform linear trend test for time course analysis."""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Reshape for sklearn
        X = time_points.reshape(-1, 1)
        y = protein_values
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Calculate p-value for slope
        y_pred = reg.predict(X)
        n = len(y)
        
        if n > 2:
            # Calculate standard error of slope
            residuals = y - y_pred
            mse = np.sum(residuals**2) / (n - 2)
            x_centered = time_points - np.mean(time_points)
            se_slope = np.sqrt(mse / np.sum(x_centered**2))
            
            # t-statistic and p-value
            t_stat = reg.coef_[0] / se_slope
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            p_value = 1.0
        
        r_squared = r2_score(y, y_pred)
        
        return reg.coef_[0], p_value, r_squared

    def _polynomial_trend_test(self, time_points: np.ndarray, protein_values: np.ndarray) -> Tuple[float, float, float]:
        """Perform polynomial trend test for time course analysis."""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Fit polynomial regression (degree 2)
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(time_points.reshape(-1, 1))
        
        reg = LinearRegression()
        reg.fit(X_poly, protein_values)
        
        # F-test for overall model significance
        y_pred = reg.predict(X_poly)
        n = len(protein_values)
        k = X_poly.shape[1] - 1  # Number of predictors excluding intercept
        
        if n > k + 1:
            ssr = np.sum((y_pred - np.mean(protein_values))**2)
            sse = np.sum((protein_values - y_pred)**2)
            msr = ssr / k
            mse = sse / (n - k - 1)
            
            f_stat = msr / mse if mse > 0 else 0
            p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
        else:
            f_stat = 0
            p_value = 1.0
        
        r_squared = r2_score(protein_values, y_pred)
        
        return f_stat, p_value, r_squared

    # Helper methods for correlation analysis
    def _perform_correlation_analysis(
        self, adata: anndata.AnnData, target_values: pd.Series, correlation_method: str
    ) -> List[Dict[str, Any]]:
        """Perform correlation analysis between proteins and target variable."""
        results = []
        X = adata.X.copy()
        protein_names = adata.var_names.tolist()
        
        for i, protein_name in enumerate(protein_names):
            protein_values = X[:, i]
            
            # Remove missing values
            if hasattr(protein_values, 'isnan'):
                valid_mask = ~(np.isnan(protein_values) | np.isnan(target_values))
            else:
                valid_mask = (protein_values > 0) & ~np.isnan(target_values)
            
            if valid_mask.sum() < 4:  # Need at least 4 points for correlation
                continue
            
            valid_protein = protein_values[valid_mask]
            valid_target = target_values[valid_mask]
            
            # Calculate correlation
            if correlation_method == "pearson":
                correlation, p_value = pearsonr(valid_protein, valid_target)
            elif correlation_method == "spearman":
                correlation, p_value = spearmanr(valid_protein, valid_target)
            elif correlation_method == "kendall":
                from scipy.stats import kendalltau
                correlation, p_value = kendalltau(valid_protein, valid_target)
            else:
                # Default to Pearson
                correlation, p_value = pearsonr(valid_protein, valid_target)
            
            results.append({
                'protein': protein_name,
                'protein_index': i,
                'correlation': float(correlation),
                'p_value': float(p_value),
                'correlation_method': correlation_method,
                'n_samples': len(valid_protein)
            })
        
        return results
