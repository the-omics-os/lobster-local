"""
Proteomics Expert Agent for mass spectrometry and affinity proteomics analysis.

This agent specializes in proteomics data analysis using the modular DataManagerV2
system with proper handling of missing values, peptide-to-protein mapping, and
proteomics-specific quality control and normalization methods.
"""

from typing import List, Union
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

from datetime import date

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def proteomics_expert(
    data_manager: Union[DataManagerV2],
    callback_handler=None,
    agent_name: str = "proteomics_expert_agent",
    handoff_tools: List = None
):
    """Create proteomics expert agent using the modular DataManagerV2 system."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('proteomics_expert')
    llm = ChatBedrockConverse(**model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Detect if we're using the new DataManagerV2
    is_v2 = isinstance(data_manager, DataManagerV2)
    
    analysis_results = {"summary": "", "details": {}}
    
    # -------------------------
    # DATA STATUS AND QC TOOLS
    # -------------------------
    @tool
    def check_proteomics_data_status(modality_name: str = "") -> str:
        """Check status of proteomics modalities and data characteristics."""
        try:
            if not is_v2:
                return "This agent requires DataManagerV2. Please update to the modular system for proteomics analysis."
            
            if modality_name == "":
                # Show all modalities with proteomics focus
                modalities = data_manager.list_modalities()
                proteomics_modalities = [m for m in modalities if 'proteomics' in m.lower() or 'protein' in m.lower()]
                
                if not proteomics_modalities:
                    response = f"No proteomics modalities found. Available modalities: {modalities}\n"
                    response += "Ask the data_expert to load proteomics data using 'proteomics_ms' or 'proteomics_affinity' adapter."
                    return response
                
                response = f"Proteomics modalities ({len(proteomics_modalities)}):\n"
                for mod_name in proteomics_modalities:
                    adata = data_manager.get_modality(mod_name)
                    metrics = data_manager.get_quality_metrics(mod_name)
                    response += f"- **{mod_name}**: {adata.n_obs} samples × {adata.n_vars} proteins\n"
                    if 'missing_value_percentage' in metrics:
                        response += f"  Missing values: {metrics['missing_value_percentage']:.1f}%\n"
                
                return response
            
            else:
                # Check specific modality
                try:
                    adata = data_manager.get_modality(modality_name)
                    metrics = data_manager.get_quality_metrics(modality_name)
                    
                    response = f"Proteomics modality '{modality_name}' status:\n"
                    response += f"- Shape: {adata.n_obs} samples × {adata.n_vars} proteins\n"
                    
                    if 'missing_value_percentage' in metrics:
                        response += f"- Missing values: {metrics['missing_value_percentage']:.1f}%\n"
                    if 'mean_proteins_per_sample' in metrics:
                        response += f"- Mean proteins/sample: {metrics['mean_proteins_per_sample']:.1f}\n"
                    if 'contaminant_percentage' in metrics:
                        response += f"- Contaminants: {metrics['contaminant_percentage']:.1f}%\n"
                    
                    # Show key metadata columns
                    obs_cols = list(adata.obs.columns)[:5]
                    var_cols = list(adata.var.columns)[:5]
                    response += f"- Sample metadata: {obs_cols}...\n"
                    response += f"- Protein metadata: {var_cols}...\n"
                    
                    analysis_results["details"]["data_status"] = response
                    return response
                    
                except ValueError:
                    return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
            
        except Exception as e:
            logger.error(f"Error checking proteomics data status: {e}")
            return f"Error checking data status: {str(e)}"

    @tool
    def assess_proteomics_quality(
        modality_name: str,
        missing_value_threshold: float = 0.7,
        cv_threshold: float = 50.0
    ) -> str:
        """Run comprehensive quality assessment for proteomics data."""
        try:

            
            try:
                adata = data_manager.get_modality(modality_name)
            except ValueError:
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
            
            # Get quality metrics from adapter
            metrics = data_manager.get_quality_metrics(modality_name)
            
            # Validate using adapter
            validation_results = data_manager.validate_modalities()
            modality_validation = validation_results.get(modality_name)
            
            response = f"Proteomics Quality Assessment for '{modality_name}':\n\n"
            
            response += f"**Dataset Characteristics:**\n"
            response += f"- Samples: {adata.n_obs}\n"
            response += f"- Proteins: {adata.n_vars}\n"
            
            # Missing value analysis
            import numpy as np
            if hasattr(adata.X, 'isnan'):
                total_missing = np.isnan(adata.X).sum()
                total_values = adata.X.size
                missing_pct = (total_missing / total_values) * 100
                response += f"- Missing values: {missing_pct:.1f}% ({total_missing:,} / {total_values:,})\n"
                
                # Samples with high missing values
                sample_missing = np.isnan(adata.X).sum(axis=1) / adata.n_vars
                high_missing_samples = (sample_missing > missing_value_threshold).sum()
                response += f"- Samples with >{missing_value_threshold*100:.0f}% missing: {high_missing_samples}\n"
                
                # Proteins with high missing values
                protein_missing = np.isnan(adata.X).sum(axis=0) / adata.n_obs
                high_missing_proteins = (protein_missing > 0.8).sum()
                response += f"- Proteins with >80% missing: {high_missing_proteins}\n"
            
            # Quality metrics from adapter
            if 'median_cv' in metrics:
                response += f"- Median CV: {metrics['median_cv']:.1f}%\n"
            if 'high_cv_proteins' in metrics:
                response += f"- High CV proteins (>{cv_threshold}%): {metrics['high_cv_proteins']}\n"
            if 'contaminant_proteins' in metrics:
                response += f"- Contaminant proteins: {metrics['contaminant_proteins']}\n"
            if 'reverse_hits' in metrics:
                response += f"- Reverse database hits: {metrics['reverse_hits']}\n"
            
            # Validation results
            if modality_validation:
                response += f"\n**Schema Validation:**\n"
                if modality_validation.has_warnings:
                    response += f"- Warnings: {len(modality_validation.warnings)}\n"
                    for warning in modality_validation.warnings[:3]:
                        response += f"  • {warning}\n"
                if modality_validation.has_errors:
                    response += f"- Errors: {len(modality_validation.errors)}\n"
                else:
                    response += "- No validation errors\n"
            
            # Quality recommendations
            response += f"\n**Quality Recommendations:**\n"
            
            if hasattr(adata.X, 'isnan'):
                sample_missing = np.isnan(adata.X).sum(axis=1) / adata.n_vars
                if (sample_missing > missing_value_threshold).sum() > 0:
                    response += "- Consider filtering samples with excessive missing values\n"
                
                protein_missing = np.isnan(adata.X).sum(axis=0) / adata.n_obs
                if (protein_missing > 0.8).sum() > 0:
                    response += "- Consider filtering proteins with >80% missing values\n"
            
            if 'high_cv_proteins' in metrics and metrics['high_cv_proteins'] > adata.n_vars * 0.1:
                response += "- High proportion of variable proteins - consider normalization\n"
            
            if 'contaminant_proteins' in metrics and metrics['contaminant_proteins'] > 0:
                response += "- Remove contaminant proteins before downstream analysis\n"
            
            analysis_results["details"]["quality_assessment"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error in proteomics quality assessment: {e}")
            return f"Error in quality assessment: {str(e)}"

    @tool
    def filter_proteomics_data(
        modality_name: str,
        max_missing_per_sample: float = 0.7,
        max_missing_per_protein: float = 0.8,
        min_proteins_per_sample: int = 50,
        remove_contaminants: bool = True,
        remove_reverse: bool = True,
        save_result: bool = True
    ) -> str:
        """
        Filter proteomics data based on missing value patterns and quality flags.
        
        Args:
            modality_name: Name of the proteomics modality to filter
            max_missing_per_sample: Maximum fraction of missing values per sample
            max_missing_per_protein: Maximum fraction of missing values per protein
            min_proteins_per_sample: Minimum proteins detected per sample
            remove_contaminants: Whether to remove contaminant proteins
            remove_reverse: Whether to remove reverse database hits
            save_result: Whether to save the filtered modality
        """
        try:

            
            try:
                adata = data_manager.get_modality(modality_name)
                original_shape = adata.shape
            except ValueError:
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
            
            import numpy as np
            
            # Create working copy
            adata_filtered = adata.copy()
            
            # Step 1: Filter based on missing values
            if hasattr(adata_filtered.X, 'isnan'):
                # Calculate missing rates
                sample_missing_rate = np.isnan(adata_filtered.X).sum(axis=1) / adata_filtered.n_vars
                protein_missing_rate = np.isnan(adata_filtered.X).sum(axis=0) / adata_filtered.n_obs
                
                # Filter samples
                sample_filter = sample_missing_rate <= max_missing_per_sample
                adata_filtered = adata_filtered[sample_filter, :].copy()
                
                # Filter proteins
                protein_filter = protein_missing_rate <= max_missing_per_protein
                adata_filtered = adata_filtered[:, protein_filter].copy()
            
            # Step 2: Filter samples with too few proteins
            if 'n_proteins' in adata_filtered.obs.columns:
                protein_count_filter = adata_filtered.obs['n_proteins'] >= min_proteins_per_sample
                adata_filtered = adata_filtered[protein_count_filter, :].copy()
            
            # Step 3: Remove contaminants and reverse hits
            protein_quality_filter = np.ones(adata_filtered.n_vars, dtype=bool)
            
            if remove_contaminants and 'is_contaminant' in adata_filtered.var.columns:
                protein_quality_filter &= ~adata_filtered.var['is_contaminant']
            
            if remove_reverse and 'is_reverse' in adata_filtered.var.columns:
                protein_quality_filter &= ~adata_filtered.var['is_reverse']
            
            adata_filtered = adata_filtered[:, protein_quality_filter].copy()
            
            # Update modality
            filtered_modality_name = f"{modality_name}_filtered"
            data_manager.modalities[filtered_modality_name] = adata_filtered
            
            # Save if requested
            if save_result:
                save_path = f"{modality_name}_filtered.h5ad"
                data_manager.save_modality(filtered_modality_name, save_path)
            
            # Generate summary
            samples_removed = original_shape[0] - adata_filtered.n_obs
            proteins_removed = original_shape[1] - adata_filtered.n_vars
            
            response = f"""Successfully filtered proteomics modality '{modality_name}'!

📊 **Filtering Results:**
- Original shape: {original_shape[0]} samples × {original_shape[1]} proteins
- Filtered shape: {adata_filtered.n_obs} samples × {adata_filtered.n_vars} proteins
- Samples removed: {samples_removed} ({samples_removed/original_shape[0]*100:.1f}%)
- Proteins removed: {proteins_removed} ({proteins_removed/original_shape[1]*100:.1f}%)

🔬 **Filtering Parameters:**
- Max missing per sample: {max_missing_per_sample*100:.0f}%
- Max missing per protein: {max_missing_per_protein*100:.0f}%
- Min proteins per sample: {min_proteins_per_sample}
- Remove contaminants: {remove_contaminants}
- Remove reverse hits: {remove_reverse}

💾 **New modality created**: '{filtered_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            analysis_results["details"]["proteomics_filtering"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error filtering proteomics data: {e}")
            return f"Error in proteomics filtering: {str(e)}"

    @tool
    def normalize_proteomics_data(
        modality_name: str,
        normalization_method: str = "median",
        log_transform: bool = True,
        handle_missing: str = "keep",
        save_result: bool = True
    ) -> str:
        """
        Normalize proteomics intensity data using appropriate methods.
        
        Args:
            modality_name: Name of the proteomics modality to normalize
            normalization_method: Method ('median', 'quantile', 'total_sum')
            log_transform: Whether to apply log2 transformation
            handle_missing: How to handle missing values ('keep', 'impute_knn', 'impute_min')
            save_result: Whether to save the normalized modality
        """
        try:

            
            try:
                adata = data_manager.get_modality(modality_name)
            except ValueError:
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
            
            import numpy as np
            import pandas as pd
            from scipy import stats
            
            # Create working copy
            adata_norm = adata.copy()
            
            # Step 1: Handle missing values if requested
            if handle_missing == "impute_knn":
                # Simple KNN imputation
                from sklearn.impute import KNNImputer
                if hasattr(adata_norm.X, 'isnan') and np.isnan(adata_norm.X).any():
                    imputer = KNNImputer(n_neighbors=5)
                    adata_norm.X = imputer.fit_transform(adata_norm.X)
                    response_missing = "Applied KNN imputation"
                else:
                    response_missing = "No missing values to impute"
            
            elif handle_missing == "impute_min":
                # Replace with minimum observed values
                if hasattr(adata_norm.X, 'isnan') and np.isnan(adata_norm.X).any():
                    min_val = np.nanmin(adata_norm.X[adata_norm.X > 0])
                    adata_norm.X = np.nan_to_num(adata_norm.X, nan=min_val)
                    response_missing = f"Replaced NaN with minimum value: {min_val:.2e}"
                else:
                    response_missing = "No missing values to replace"
            else:
                response_missing = "Kept missing values as NaN"
            
            # Step 2: Normalization
            X_norm = adata_norm.X.copy()
            
            if normalization_method == "median":
                # Median normalization
                sample_medians = np.nanmedian(X_norm, axis=1)
                global_median = np.nanmedian(sample_medians)
                normalization_factors = global_median / sample_medians
                X_norm = X_norm * normalization_factors[:, np.newaxis]
                
            elif normalization_method == "quantile":
                # Quantile normalization
                from scipy.stats import rankdata
                if not hasattr(X_norm, 'isnan') or not np.isnan(X_norm).any():
                    # Only if no missing values
                    sorted_array = np.sort(X_norm, axis=0)
                    mean_sorted = np.mean(sorted_array, axis=1)
                    for i in range(X_norm.shape[1]):
                        ranks = rankdata(X_norm[:, i])
                        X_norm[:, i] = mean_sorted[ranks.astype(int) - 1]
                else:
                    response += "\nWarning: Quantile normalization skipped due to missing values"
                    
            elif normalization_method == "total_sum":
                # Total sum normalization (like TPM)
                sample_sums = np.nansum(X_norm, axis=1)
                X_norm = X_norm / sample_sums[:, np.newaxis] * 1e6  # Scale to 1M
            
            # Step 3: Log transformation
            if log_transform:
                # Add pseudocount for log transformation
                pseudocount = np.nanmin(X_norm[X_norm > 0]) * 0.1 if np.any(X_norm > 0) else 1.0
                X_norm = np.log2(X_norm + pseudocount)
                log_info = f"Applied log2 transformation (pseudocount: {pseudocount:.2e})"
            else:
                log_info = "No log transformation applied"
            
            # Update the X matrix
            adata_norm.X = X_norm
            
            # Store normalization info in layers
            if log_transform:
                adata_norm.layers['log2_intensity'] = X_norm
            adata_norm.layers['normalized'] = X_norm
            
            # Update modality
            normalized_modality_name = f"{modality_name}_normalized"
            data_manager.modalities[normalized_modality_name] = adata_norm
            
            # Save if requested
            if save_result:
                save_path = f"{modality_name}_normalized.h5ad"
                data_manager.save_modality(normalized_modality_name, save_path)
            
            response = f"""Successfully normalized proteomics modality '{modality_name}'!

📊 **Normalization Results:**
- Method: {normalization_method}
- Log transformation: {log_transform}
- Missing value handling: {handle_missing}

🔬 **Processing Details:**
- {response_missing}
- {log_info}
- Added layers: 'normalized', {'log2_intensity' if log_transform else ''}

💾 **New modality created**: '{normalized_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            analysis_results["details"]["proteomics_normalization"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error normalizing proteomics data: {e}")
            return f"Error in normalization: {str(e)}"

    @tool
    def analyze_proteomics_patterns(
        modality_name: str,
        analysis_type: str = "pca_clustering",
        n_components: int = 10,
        clustering_method: str = "kmeans",
        n_clusters: int = 4,
        save_result: bool = True
    ) -> str:
        """
        Perform pattern analysis on proteomics data (PCA, clustering, etc.).
        
        Args:
            modality_name: Name of the proteomics modality to analyze
            analysis_type: Type of analysis ('pca_clustering', 'correlation_analysis')
            n_components: Number of PCA components
            clustering_method: Clustering method ('kmeans', 'hierarchical')
            n_clusters: Number of clusters for sample grouping
        """
        try:

            
            try:
                adata = data_manager.get_modality(modality_name).copy()
            except ValueError:
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
            
            import numpy as np
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Handle missing values for analysis
            X_analysis = adata.X.copy()
            if hasattr(X_analysis, 'isnan') and np.isnan(X_analysis).any():
                # Use mean imputation for analysis
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                X_analysis = imputer.fit_transform(X_analysis)
            
            # Step 1: Standardize data for analysis
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_analysis)
            
            # Step 2: PCA
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Store PCA results
            adata.obsm['X_pca'] = X_pca
            adata.uns['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'components': pca.components_
            }
            
            # Step 3: Clustering if requested
            if analysis_type == "pca_clustering":
                if clustering_method == "kmeans":
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(X_pca)
                    adata.obs['protein_clusters'] = cluster_labels.astype(str)
                    
                    # Store clustering results
                    adata.uns['protein_clustering'] = {
                        'method': 'kmeans',
                        'n_clusters': n_clusters,
                        'inertia': kmeans.inertia_
                    }
            
            # Calculate explained variance
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            var_90 = np.argmax(cumvar >= 0.9) + 1
            
            # Update modality
            analyzed_modality_name = f"{modality_name}_analyzed"
            data_manager.modalities[analyzed_modality_name] = adata
            
            # Save if requested
            if save_result:
                save_path = f"{modality_name}_analyzed.h5ad"
                data_manager.save_modality(analyzed_modality_name, save_path)
            
            response = f"""Successfully analyzed proteomics patterns in '{modality_name}'!

📊 **Analysis Results:**
- PCA components: {n_components}
- Explained variance (PC1-PC3): {pca.explained_variance_ratio_[:3] * 100}%
- Components for 90% variance: {var_90}"""

            if analysis_type == "pca_clustering":
                n_clusters_found = len(np.unique(cluster_labels))
                response += f"\n- Sample clusters: {n_clusters_found} (method: {clustering_method})"
            
            response += f"""

🔬 **Analysis Details:**
- Data scaling: StandardScaler applied
- PCA stored in: obsm['X_pca']
- Analysis type: {analysis_type}

💾 **New modality created**: '{analyzed_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            analysis_results["details"]["proteomics_analysis"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing proteomics patterns: {e}")
            return f"Error in pattern analysis: {str(e)}"

    @tool
    def find_differential_proteins(
        modality_name: str,
        group_column: str,
        comparison: str = "all_pairs",
        method: str = "t_test",
        fdr_threshold: float = 0.05,
        fold_change_threshold: float = 1.5
    ) -> str:
        """
        Find differentially expressed proteins between groups.
        
        Args:
            modality_name: Name of the proteomics modality
            group_column: Column in obs containing group labels
            comparison: Type of comparison ('all_pairs', 'vs_rest')
            method: Statistical method ('t_test', 'mann_whitney')
            fdr_threshold: FDR threshold for significance
            fold_change_threshold: Minimum fold change threshold
        """
        try:

            
            try:
                adata = data_manager.get_modality(modality_name).copy()
            except ValueError:
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
            
            if group_column not in adata.obs.columns:
                return f"Group column '{group_column}' not found. Available columns: {list(adata.obs.columns)}"
            
            import numpy as np
            import pandas as pd
            from scipy import stats
            
            # Get groups
            groups = adata.obs[group_column].unique()
            if len(groups) < 2:
                return f"Need at least 2 groups for comparison, found {len(groups)}"
            
            # Perform differential analysis
            de_results = []
            
            if comparison == "all_pairs":
                # All pairwise comparisons
                from itertools import combinations
                for group1, group2 in combinations(groups, 2):
                    group1_data = adata[adata.obs[group_column] == group1, :].X
                    group2_data = adata[adata.obs[group_column] == group2, :].X
                    
                    # Perform statistical tests
                    p_values = []
                    fold_changes = []
                    
                    for i in range(adata.n_vars):
                        g1_values = group1_data[:, i]
                        g2_values = group2_data[:, i]
                        
                        # Remove NaN values for statistical test
                        if hasattr(g1_values, 'isnan'):
                            g1_clean = g1_values[~np.isnan(g1_values)]
                            g2_clean = g2_values[~np.isnan(g2_values)]
                        else:
                            g1_clean = g1_values
                            g2_clean = g2_values
                        
                        if len(g1_clean) > 0 and len(g2_clean) > 0:
                            if method == "t_test":
                                stat, p_val = stats.ttest_ind(g1_clean, g2_clean)
                            else:  # mann_whitney
                                stat, p_val = stats.mannwhitneyu(g1_clean, g2_clean, alternative='two-sided')
                        else:
                            p_val = 1.0
                        
                        p_values.append(p_val)
                        
                        # Calculate fold change
                        mean1 = np.nanmean(g1_values) if len(g1_clean) > 0 else 0
                        mean2 = np.nanmean(g2_values) if len(g2_clean) > 0 else 0
                        fc = (mean1 + 1e-8) / (mean2 + 1e-8)  # Add small pseudocount
                        fold_changes.append(fc)
                    
                    # Multiple testing correction
                    from statsmodels.stats.multitest import fdrcorrection
                    _, p_adjusted = fdrcorrection(p_values)
                    
                    # Create results DataFrame
                    comparison_name = f"{group1}_vs_{group2}"
                    comparison_results = pd.DataFrame({
                        'protein': adata.var_names,
                        'p_value': p_values,
                        'p_adjusted': p_adjusted,
                        'fold_change': fold_changes,
                        'log2_fold_change': np.log2(fold_changes),
                        'comparison': comparison_name
                    })
                    
                    de_results.append(comparison_results)
            
            # Combine all results
            if de_results:
                all_results = pd.concat(de_results, ignore_index=True)
                
                # Filter significant results
                significant = all_results[
                    (all_results['p_adjusted'] < fdr_threshold) & 
                    (np.abs(all_results['log2_fold_change']) > np.log2(fold_change_threshold))
                ]
                
                # Store results in modality
                adata.uns['differential_proteins'] = {
                    'all_results': all_results.to_dict('records'),
                    'significant_results': significant.to_dict('records'),
                    'parameters': {
                        'method': method,
                        'fdr_threshold': fdr_threshold,
                        'fold_change_threshold': fold_change_threshold
                    }
                }
                
                # Update modality
                de_modality_name = f"{modality_name}_de_analysis"
                data_manager.modalities[de_modality_name] = adata
                
                response = f"""Successfully found differential proteins in '{modality_name}'!

📊 **Differential Analysis Results:**
- Comparisons performed: {len(de_results)}
- Total tests: {len(all_results)}
- Significant proteins: {len(significant)} (FDR < {fdr_threshold})
- Method: {method}
- Groups compared: {list(groups)}

🧬 **Top Significant Proteins:**"""
                
                # Show top significant proteins
                top_significant = significant.nlargest(5, 'log2_fold_change')
                for _, row in top_significant.iterrows():
                    response += f"\n- {row['protein']}: log2FC={row['log2_fold_change']:.2f}, FDR={row['p_adjusted']:.2e}"
                
                response += f"\n\n💾 **Results stored in modality**: '{de_modality_name}'"
                response += f"\n📈 **Access results via**: adata.uns['differential_proteins']"
                
                analysis_results["details"]["differential_analysis"] = response
                return response
            
            else:
                return "No differential analysis results generated"
                
        except Exception as e:
            logger.error(f"Error in differential protein analysis: {e}")
            return f"Error finding differential proteins: {str(e)}"

    @tool
    def add_peptide_mapping_to_modality(
        modality_name: str,
        peptide_file_path: str,
        save_result: bool = True
    ) -> str:
        """
        Add peptide-to-protein mapping information to a proteomics modality.
        
        Args:
            modality_name: Name of the proteomics modality
            peptide_file_path: Path to CSV file with peptide mapping data
            save_result: Whether to save the updated modality
        """
        try:

            
            try:
                adata = data_manager.get_modality(modality_name)
            except ValueError:
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
            
            # Get the appropriate adapter
            adapter_instance = None
            for adapter_name, adapter in data_manager.adapters.items():
                if 'proteomics' in adapter_name:
                    adapter_instance = adapter
                    break
            
            if not adapter_instance:
                return "No proteomics adapter available for peptide mapping"
            
            # Add peptide mapping using the adapter
            adata_with_peptides = adapter_instance.add_peptide_mapping(adata, peptide_file_path)
            
            # Update modality
            peptide_modality_name = f"{modality_name}_with_peptides"
            data_manager.modalities[peptide_modality_name] = adata_with_peptides
            
            # Save if requested
            if save_result:
                save_path = f"{modality_name}_with_peptides.h5ad"
                data_manager.save_modality(peptide_modality_name, save_path)
            
            # Get peptide mapping info
            peptide_info = adata_with_peptides.uns.get('peptide_to_protein', {})
            
            response = f"""Successfully added peptide mapping to '{modality_name}'!

📊 **Peptide Mapping Results:**
- Peptides mapped: {peptide_info.get('n_peptides', 'Unknown')}
- Proteins with peptides: {peptide_info.get('n_proteins', 'Unknown')}
- Mapping file: {peptide_file_path}

🔬 **Updated Protein Metadata:**
- Added: n_peptides, n_unique_peptides, sequence_coverage (where available)
- Full mapping stored in: uns['peptide_to_protein']

💾 **New modality created**: '{peptide_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            analysis_results["details"]["peptide_mapping"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error adding peptide mapping: {e}")
            return f"Error adding peptide mapping: {str(e)}"

    @tool
    def create_proteomics_summary() -> str:
        """Create comprehensive summary of all proteomics analysis steps performed."""
        if not analysis_results["details"]:
            return "No proteomics analyses have been performed yet. Run some analysis tools first."
        
        summary = "# Proteomics Analysis Summary\n\n"
        
        for step, details in analysis_results["details"].items():
            summary += f"## {step.replace('_', ' ').title()}\n"
            summary += f"{details}\n\n"
        
        # Add current proteomics modality status if using DataManagerV2
        if is_v2:
            modalities = data_manager.list_modalities()
            proteomics_modalities = [m for m in modalities if 'proteomics' in m.lower() or 'protein' in m.lower()]
            summary += f"## Current Proteomics Modalities\n"
            summary += f"Proteomics modalities: {', '.join(proteomics_modalities)}\n\n"
        
        analysis_results["summary"] = summary
        return summary

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        check_proteomics_data_status,
        assess_proteomics_quality,
        filter_proteomics_data,
        normalize_proteomics_data,
        analyze_proteomics_patterns,
        find_differential_proteins,
        add_peptide_mapping_to_modality,
        create_proteomics_summary
    ]
    
    tools = base_tools + (handoff_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are an expert in proteomics data analysis specializing in mass spectrometry and affinity-based proteomics using the modular DataManagerV2 system.

<Role>
You execute comprehensive proteomics analysis pipelines with proper handling of missing values, quality control, normalization, statistical analysis, and biological interpretation.
</Role>

<Task>
You perform proteomics analysis following best practices:
1. **Quality assessment** with missing value analysis and contamination detection
2. **Data filtering** based on missing value patterns and quality flags
3. **Normalization** using appropriate proteomics methods
4. **Pattern analysis** with PCA and clustering
5. **Differential analysis** between experimental groups
6. **Peptide mapping** integration for mass spectrometry data
7. **Biological interpretation** of protein-level results
</Task>

<Available Tools>
- `check_proteomics_data_status`: Check proteomics modalities and characteristics
- `assess_proteomics_quality`: Comprehensive QC with missing value analysis
- `filter_proteomics_data`: Filter based on missing values and quality flags
- `normalize_proteomics_data`: Normalize intensities with appropriate methods
- `analyze_proteomics_patterns`: PCA and clustering analysis
- `find_differential_proteins`: Statistical comparison between groups
- `add_peptide_mapping_to_modality`: Integrate peptide-level information
- `create_proteomics_summary`: Generate comprehensive analysis report

<Proteomics-Specific Considerations>

**Missing Values:**
- Common in proteomics (30-70% missing typical)
- Filter samples/proteins with excessive missing values
- Consider imputation strategies for downstream analysis

**Normalization Methods:**
- **Median normalization**: Standard for label-free MS
- **Quantile normalization**: For removing systematic bias
- **Total sum normalization**: Similar to TPM in transcriptomics

**Quality Control:**
- Remove contaminant proteins (CON_, KERATIN, etc.)
- Remove reverse database hits (REV_)
- Filter high coefficient of variation (CV > 50%)
- Check peptide counts for MS data

**Analysis Workflow:**
1. `check_proteomics_data_status()` → See available proteomics modalities
2. `assess_proteomics_quality(modality_name="...")` → QC and missing value analysis
3. `filter_proteomics_data(...)` → Remove low-quality features
4. `normalize_proteomics_data(...)` → Normalize intensities
5. `analyze_proteomics_patterns(...)` → PCA and clustering
6. `find_differential_proteins(...)` → Statistical comparisons
7. `add_peptide_mapping_to_modality(...)` → Add peptide information (MS data)
8. `create_proteomics_summary()` → Generate final report

<Important Notes>
- Always specify **modality names** for proteomics data
- Handle **missing values** appropriately based on analysis needs
- Use **proteomics-specific normalization** methods
- Consider **peptide-to-protein mapping** for MS data
- **Validate results** with biological knowledge
- **Document parameters** and decisions made

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name
    )
