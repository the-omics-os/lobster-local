"""
Affinity Proteomics Expert Agent for targeted proteomics analysis.

This agent specializes in affinity-based proteomics data analysis (Olink, SomaScan, etc.)
using the modular DataManagerV2 system with proper handling of targeted protein panels,
antibody-specific quality control, and platform-specific normalization methods.
"""

from typing import List, Union
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from lobster.config.llm_factory import create_llm

from datetime import date

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger
from lobster.tools.proteomics_preprocessing_service import ProteomicsPreprocessingService
from lobster.tools.proteomics_quality_service import ProteomicsQualityService
from lobster.tools.proteomics_analysis_service import ProteomicsAnalysisService
from lobster.tools.proteomics_differential_service import ProteomicsDifferentialService

logger = get_logger(__name__)


def affinity_proteomics_expert(
    data_manager: Union[DataManagerV2],
    callback_handler=None,
    agent_name: str = "affinity_proteomics_expert_agent",
    handoff_tools: List = None
):
    """Create affinity proteomics expert agent using the modular DataManagerV2 system."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('affinity_proteomics_expert_agent')
    llm = create_llm('affinity_proteomics_expert_agent', model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Always use DataManagerV2 for modular affinity proteomics analysis
    if not isinstance(data_manager, DataManagerV2):
        raise ValueError("AffinityProteomicsExpert requires DataManagerV2 for modular analysis")
    
    # Initialize stateless services
    preprocessing_service = ProteomicsPreprocessingService()
    quality_service = ProteomicsQualityService()
    analysis_service = ProteomicsAnalysisService()
    differential_service = ProteomicsDifferentialService()
    
    analysis_results = {"summary": "", "details": {}}
    
    # -------------------------
    # AFFINITY-SPECIFIC DATA TOOLS
    # -------------------------
    @tool
    def check_affinity_proteomics_data_status(modality_name: str = "") -> str:
        """Check status of affinity proteomics modalities and data characteristics."""
        try:
            if modality_name == "":
                # Show all modalities with affinity proteomics focus
                modalities = data_manager.list_modalities()
                affinity_modalities = [m for m in modalities if any(term in m.lower() for term in ['proteomics', 'protein', 'olink', 'soma', 'affinity', 'panel'])]
                
                if not affinity_modalities:
                    response = f"No affinity proteomics modalities found. Available modalities: {modalities}\n"
                    response += "Ask the data_expert to load affinity proteomics data using 'proteomics_affinity' adapter."
                    return response
                
                response = f"Affinity Proteomics modalities ({len(affinity_modalities)}):\n"
                for mod_name in affinity_modalities:
                    adata = data_manager.get_modality(mod_name)
                    metrics = data_manager.get_quality_metrics(mod_name)
                    response += f"- **{mod_name}**: {adata.n_obs} samples × {adata.n_vars} proteins\n"
                    if 'missing_value_percentage' in metrics:
                        response += f"  Missing values: {metrics['missing_value_percentage']:.1f}%\n"
                    if 'median_cv' in metrics:
                        response += f"  Median CV: {metrics['median_cv']:.1f}%\n"
                
                return response
            
            else:
                # Check specific modality
                try:
                    adata = data_manager.get_modality(modality_name)
                    metrics = data_manager.get_quality_metrics(modality_name)
                    
                    response = f"Affinity Proteomics modality '{modality_name}' status:\n"
                    response += f"- Shape: {adata.n_obs} samples × {adata.n_vars} proteins\n"
                    
                    if 'missing_value_percentage' in metrics:
                        response += f"- Missing values: {metrics['missing_value_percentage']:.1f}% (expected <30% for affinity)\n"
                    if 'mean_proteins_per_sample' in metrics:
                        response += f"- Mean proteins/sample: {metrics['mean_proteins_per_sample']:.1f}\n"
                    if 'median_cv' in metrics:
                        response += f"- Median CV: {metrics['median_cv']:.1f}%\n"
                    
                    # Affinity-specific metadata
                    affinity_cols = ['antibody_id', 'antibody_clone', 'panel_type', 'lot_number', 'npx_value']
                    present_cols = [col for col in affinity_cols if col in adata.var.columns]
                    if present_cols:
                        response += f"- Affinity metadata available: {present_cols}\n"
                    
                    # Panel information
                    if adata.n_vars < 200:
                        response += f"- Targeted panel detected ({adata.n_vars} proteins)\n"
                    
                    # Show key metadata columns
                    obs_cols = list(adata.obs.columns)[:5]
                    var_cols = list(adata.var.columns)[:5]
                    response += f"- Sample metadata: {obs_cols}...\n"
                    response += f"- Protein metadata: {var_cols}...\n"
                    
                    analysis_results["details"]["affinity_data_status"] = response
                    return response
                    
                except ValueError:
                    return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
            
        except Exception as e:
            logger.error(f"Error checking affinity proteomics data status: {e}")
            return f"Error checking data status: {str(e)}"

    @tool
    def assess_affinity_proteomics_quality(
        modality_name: str,
        missing_value_threshold: float = 0.3,
        cv_threshold: float = 30.0,
        plate_effect_threshold: float = 0.1
    ) -> str:
        """Run comprehensive quality assessment for affinity proteomics data with platform-specific metrics."""
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
        
        try:
            # Use the quality service for comprehensive assessment
            processed_adata, stats = quality_service.assess_missing_value_patterns(adata)
            cv_adata, cv_stats = quality_service.assess_coefficient_variation(processed_adata, cv_threshold)
            contam_adata, contam_stats = quality_service.detect_contaminants(cv_adata)
            final_adata, range_stats = quality_service.evaluate_dynamic_range(contam_adata)
            
            # Affinity-specific quality checks
            if 'plate_id' in final_adata.obs.columns:
                # Assess plate effects
                plate_adata, plate_stats = quality_service.assess_technical_replicates(final_adata, replicate_column='plate_id')
                final_adata = plate_adata
            else:
                plate_stats = {}
            
            # Update the modality with quality assessment results
            data_manager.modalities[modality_name] = final_adata
            
            # Combine all statistics
            combined_stats = {**stats, **cv_stats, **contam_stats, **range_stats, **plate_stats}
            
            # Generate comprehensive response
            response = f"Affinity Proteomics Quality Assessment for '{modality_name}':\n\n"
            response += "**Dataset Characteristics:**\n"
            response += f"- Samples: {final_adata.n_obs}\n"
            response += f"- Proteins: {final_adata.n_vars} (targeted panel)\n"
            
            # Missing value patterns (should be low for affinity)
            if 'missing_value_percentage' in combined_stats:
                response += f"- Missing values: {combined_stats['missing_value_percentage']:.1f}% (expected <30% for affinity)\n"
            if 'samples_high_missing' in combined_stats:
                response += f"- Samples with >{missing_value_threshold*100:.0f}% missing: {combined_stats['samples_high_missing']}\n"
            if 'proteins_high_missing' in combined_stats:
                response += f"- Proteins with >50% missing: {combined_stats['proteins_high_missing']}\n"
            
            # CV assessment (should be lower for affinity platforms)
            if 'median_cv' in combined_stats:
                response += f"- Median CV: {combined_stats['median_cv']:.1f}% (expected <30% for affinity)\n"
            if 'high_cv_proteins' in combined_stats:
                response += f"- High CV proteins (>{cv_threshold}%): {combined_stats['high_cv_proteins']}\n"
            
            # Platform-specific quality metrics
            if 'antibody_specificity' in combined_stats:
                response += f"- Antibody cross-reactivity flagged: {combined_stats['antibody_specificity']}\n"
            
            # Plate effects (important for affinity platforms)
            if 'plate_correlation' in combined_stats:
                response += f"- Inter-plate correlation: {combined_stats['plate_correlation']:.3f}\n"
            if 'plate_effect_detected' in combined_stats and combined_stats['plate_effect_detected']:
                response += "- Plate effects detected: requires correction\n"
            
            # Dynamic range
            if 'dynamic_range_log10' in combined_stats:
                response += f"- Dynamic range: {combined_stats['dynamic_range_log10']:.1f} log10 units\n"
            
            # Affinity-specific quality recommendations
            response += "\n**Affinity Platform Quality Recommendations:**\n"
            
            if combined_stats.get('missing_value_percentage', 0) > 30:
                response += "- High missing values unusual for affinity platforms - check assay quality\n"
            if combined_stats.get('samples_high_missing', 0) > 0:
                response += "- Consider re-running samples with excessive missing values\n"
            if combined_stats.get('median_cv', 0) > 30:
                response += "- High CVs suggest technical issues - check sample handling and pipetting\n"
            if combined_stats.get('plate_effect_detected', False):
                response += "- Apply plate effect correction before analysis\n"
            if combined_stats.get('high_cv_proteins', 0) > final_adata.n_vars * 0.1:
                response += "- Multiple high-CV proteins suggest systematic technical issues\n"
            
            # Platform-specific notes
            if final_adata.n_vars < 100:
                response += "- Small targeted panel - consider protein selection bias in interpretation\n"
            
            analysis_results["details"]["affinity_quality_assessment"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error in affinity proteomics quality assessment: {e}")
            return f"Error in quality assessment: {str(e)}"

    @tool
    def filter_affinity_proteomics_data(
        modality_name: str,
        max_missing_per_sample: float = 0.3,
        max_missing_per_protein: float = 0.5,
        max_cv_threshold: float = 50.0,
        min_proteins_per_sample: int = 20,
        remove_failed_antibodies: bool = True,
        save_result: bool = True
    ) -> str:
        """
        Filter affinity proteomics data with platform-specific quality criteria.
        
        Args:
            modality_name: Name of the affinity proteomics modality to filter
            max_missing_per_sample: Maximum fraction of missing values per sample
            max_missing_per_protein: Maximum fraction of missing values per protein
            max_cv_threshold: Maximum CV threshold for protein filtering
            min_proteins_per_sample: Minimum proteins detected per sample
            remove_failed_antibodies: Whether to remove antibodies with quality issues
            save_result: Whether to save the filtered modality
        """
        try:
            adata = data_manager.get_modality(modality_name)
            original_shape = adata.shape
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
        
        try:
            import numpy as np
            
            # Create working copy
            adata_filtered = adata.copy()
            
            # Step 1: Filter based on missing values (more stringent for affinity)
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
            
            # Step 2: Filter based on coefficient of variation
            if 'cv' in adata_filtered.var.columns:
                cv_filter = adata_filtered.var['cv'] <= max_cv_threshold
                adata_filtered = adata_filtered[:, cv_filter].copy()
            
            # Step 3: Filter samples with too few proteins detected
            if 'n_proteins' in adata_filtered.obs.columns:
                protein_count_filter = adata_filtered.obs['n_proteins'] >= min_proteins_per_sample
                adata_filtered = adata_filtered[protein_count_filter, :].copy()
            
            # Step 4: Remove failed antibodies (affinity-specific)
            protein_quality_filter = np.ones(adata_filtered.n_vars, dtype=bool)
            
            if remove_failed_antibodies:
                # Check for antibody quality flags
                if 'antibody_quality' in adata_filtered.var.columns:
                    protein_quality_filter &= adata_filtered.var['antibody_quality'] != 'failed'
                
                if 'cross_reactive' in adata_filtered.var.columns:
                    protein_quality_filter &= ~adata_filtered.var['cross_reactive']
                
                if 'below_detection' in adata_filtered.var.columns:
                    protein_quality_filter &= ~adata_filtered.var['below_detection']
            
            adata_filtered = adata_filtered[:, protein_quality_filter].copy()
            
            # Update modality
            filtered_modality_name = f"{modality_name}_affinity_filtered"
            data_manager.modalities[filtered_modality_name] = adata_filtered
            
            # Save if requested
            if save_result:
                save_path = f"{modality_name}_affinity_filtered.h5ad"
                data_manager.save_modality(filtered_modality_name, save_path)
            
            # Generate summary
            samples_removed = original_shape[0] - adata_filtered.n_obs
            proteins_removed = original_shape[1] - adata_filtered.n_vars
            
            response = f"""Successfully filtered affinity proteomics modality '{modality_name}'!

📊 **Affinity Proteomics Filtering Results:**
- Original shape: {original_shape[0]} samples × {original_shape[1]} proteins
- Filtered shape: {adata_filtered.n_obs} samples × {adata_filtered.n_vars} proteins
- Samples removed: {samples_removed} ({samples_removed/original_shape[0]*100:.1f}%)
- Proteins removed: {proteins_removed} ({proteins_removed/original_shape[1]*100:.1f}%)

🔬 **Affinity-Specific Filtering Parameters:**
- Max missing per sample: {max_missing_per_sample*100:.0f}% (stringent for affinity)
- Max missing per protein: {max_missing_per_protein*100:.0f}%
- Max CV threshold: {max_cv_threshold:.0f}%
- Min proteins per sample: {min_proteins_per_sample}
- Remove failed antibodies: {remove_failed_antibodies}

💾 **New modality created**: '{filtered_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            analysis_results["details"]["affinity_proteomics_filtering"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error filtering affinity proteomics data: {e}")
            return f"Error in affinity proteomics filtering: {str(e)}"

    @tool
    def normalize_affinity_proteomics_data(
        modality_name: str,
        normalization_method: str = "quantile",
        correct_plate_effects: bool = True,
        handle_missing: str = "impute_knn",
        save_result: bool = True
    ) -> str:
        """
        Normalize affinity proteomics data using platform-appropriate methods.
        
        Args:
            modality_name: Name of the affinity proteomics modality to normalize
            normalization_method: Method ('quantile', 'median', 'robust_z_score')
            correct_plate_effects: Whether to correct for plate effects
            handle_missing: How to handle missing values ('impute_knn', 'impute_median', 'keep')
            save_result: Whether to save the normalized modality
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
        
        try:
            # Step 1: Handle missing values (more aggressive for affinity data)
            if handle_missing == "impute_knn":
                processed_adata, impute_stats = preprocessing_service.impute_missing_values(adata, method="knn")
            elif handle_missing == "impute_median":
                processed_adata, impute_stats = preprocessing_service.impute_missing_values(adata, method="median")
            else:
                processed_adata = adata.copy()
                impute_stats = {"imputation_method": "none", "imputation_applied": False}
            
            # Step 2: Correct plate effects if requested
            if correct_plate_effects and 'plate_id' in processed_adata.obs.columns:
                corrected_adata, batch_stats = preprocessing_service.correct_batch_effects(
                    processed_adata, batch_column='plate_id'
                )
                processed_adata = corrected_adata
            else:
                batch_stats = {"batch_correction_applied": False}
            
            # Step 3: Affinity-specific normalization (no log transform for NPX data)
            normalized_adata, norm_stats = preprocessing_service.normalize_intensities(
                processed_adata, 
                method=normalization_method, 
                log_transform=False  # NPX values already log-transformed
            )
            
            # Update modality
            normalized_modality_name = f"{modality_name}_affinity_normalized"
            data_manager.modalities[normalized_modality_name] = normalized_adata
            
            # Save if requested
            if save_result:
                save_path = f"{modality_name}_affinity_normalized.h5ad"
                data_manager.save_modality(normalized_modality_name, save_path)
            
            # Combine statistics
            combined_stats = {**impute_stats, **batch_stats, **norm_stats}
            
            response = f"""Successfully normalized affinity proteomics modality '{modality_name}'!

📊 **Affinity Proteomics Normalization Results:**
- Method: {normalization_method} (optimized for targeted panels)
- Plate effect correction: {correct_plate_effects}
- Missing value handling: {handle_missing}

🔬 **Affinity-Specific Processing Details:**"""
            
            if combined_stats.get('imputation_applied', False):
                response += f"\n- Applied {combined_stats.get('imputation_method', 'unknown')} imputation"
                if 'n_imputed_values' in combined_stats:
                    response += f" ({combined_stats['n_imputed_values']} values)"
                response += "\n  Note: Imputation appropriate for affinity data with low missing rates"
            else:
                response += "\n- No imputation applied"
            
            if combined_stats.get('batch_correction_applied', False):
                response += "\n- Plate effect correction applied"
                if 'n_batches_corrected' in combined_stats:
                    response += f" ({combined_stats['n_batches_corrected']} plates)"
            
            if 'normalization_factor_median' in combined_stats:
                response += f"\n- Median normalization factor: {combined_stats['normalization_factor_median']:.2f}"
            
            response += "\n- No log transformation (NPX values already log-scale)"
            
            response += f"\n\n💾 **New modality created**: '{normalized_modality_name}'"

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            analysis_results["details"]["affinity_proteomics_normalization"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error normalizing affinity proteomics data: {e}")
            return f"Error in affinity normalization: {str(e)}"

    @tool
    def analyze_affinity_proteomics_patterns(
        modality_name: str,
        analysis_type: str = "pca_clustering",
        n_components: int = 10,
        clustering_method: str = "kmeans",
        n_clusters: int = 4,
        save_result: bool = True
    ) -> str:
        """
        Perform pattern analysis on affinity proteomics data optimized for targeted panels.
        
        Args:
            modality_name: Name of the affinity proteomics modality to analyze
            analysis_type: Type of analysis ('pca_clustering', 'correlation_analysis')
            n_components: Number of PCA components (lower for targeted panels)
            clustering_method: Clustering method ('kmeans', 'hierarchical')
            n_clusters: Number of clusters for sample grouping
            save_result: Whether to save results
        """
        try:
            adata = data_manager.get_modality(modality_name).copy()
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
        
        try:
            # Use analysis service for dimensionality reduction
            pca_adata, pca_stats = analysis_service.perform_dimensionality_reduction(
                adata, method="pca", n_components=n_components
            )
            
            # Perform clustering if requested
            if analysis_type == "pca_clustering":
                clustered_adata, cluster_stats = analysis_service.perform_clustering_analysis(
                    pca_adata, method=clustering_method, n_clusters=n_clusters
                )
                final_adata = clustered_adata
                combined_stats = {**pca_stats, **cluster_stats}
            else:
                final_adata = pca_adata
                combined_stats = pca_stats
            
            # Update modality
            analyzed_modality_name = f"{modality_name}_affinity_analyzed"
            data_manager.modalities[analyzed_modality_name] = final_adata
            
            # Save if requested
            if save_result:
                save_path = f"{modality_name}_affinity_analyzed.h5ad"
                data_manager.save_modality(analyzed_modality_name, save_path)
            
            response = f"""Successfully analyzed affinity proteomics patterns in '{modality_name}'!

📊 **Affinity Proteomics Analysis Results:**
- PCA components: {n_components} (optimized for targeted panels)"""
            
            if 'explained_variance_ratio' in combined_stats:
                ev_ratio = combined_stats['explained_variance_ratio'][:3]
                response += f"\n- Explained variance (PC1-PC3): {[f'{x*100:.1f}%' for x in ev_ratio]}"
            
            if 'components_for_90_variance' in combined_stats:
                response += f"\n- Components for 90% variance: {combined_stats['components_for_90_variance']}"
            
            if analysis_type == "pca_clustering":
                if 'n_clusters_found' in combined_stats:
                    response += f"\n- Sample clusters: {combined_stats['n_clusters_found']} (method: {clustering_method})"
                if 'silhouette_score' in combined_stats:
                    response += f"\n- Clustering quality (silhouette): {combined_stats['silhouette_score']:.3f}"
            
            response += f"""

🔬 **Affinity-Specific Analysis Details:**
- Analysis optimized for targeted protein panels
- PCA stored in: obsm['{combined_stats.get('pca_key', 'X_pca')}']
- Data characteristics: low missing values, controlled measurement conditions"""
            
            if analysis_type == "pca_clustering":
                response += f"\n- Cluster labels stored in: obs['{combined_stats.get('cluster_key', 'clusters')}']"
            
            # Platform-specific notes
            if final_adata.n_vars < 100:
                response += f"\n- Small targeted panel ({final_adata.n_vars} proteins) - interpret patterns cautiously"
            
            response += f"\n\n💾 **New modality created**: '{analyzed_modality_name}'"

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            analysis_results["details"]["affinity_proteomics_analysis"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing affinity proteomics patterns: {e}")
            return f"Error in affinity pattern analysis: {str(e)}"

    @tool
    def find_differential_proteins_affinity(
        modality_name: str,
        group_column: str,
        comparison: str = "all_pairs",
        method: str = "t_test",
        fdr_threshold: float = 0.05,
        fold_change_threshold: float = 1.2
    ) -> str:
        """
        Find differentially expressed proteins between groups using affinity-optimized methods.
        
        Args:
            modality_name: Name of the affinity proteomics modality
            group_column: Column in obs containing group labels
            comparison: Type of comparison ('all_pairs', 'vs_rest')
            method: Statistical method ('t_test', 'mann_whitney', 'limma_moderated')
            fdr_threshold: FDR threshold for significance
            fold_change_threshold: Minimum fold change threshold (lower for targeted panels)
        """
        try:
            adata = data_manager.get_modality(modality_name).copy()
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
        
        try:
            if group_column not in adata.obs.columns:
                return f"Group column '{group_column}' not found. Available columns: {list(adata.obs.columns)}"
            
            # Use differential service for analysis
            differential_adata, de_stats = differential_service.perform_differential_expression(
                adata, 
                group_column=group_column, 
                method=method,
                fdr_threshold=fdr_threshold,
                fold_change_threshold=fold_change_threshold
            )
            
            # Update modality
            de_modality_name = f"{modality_name}_affinity_de_analysis"
            data_manager.modalities[de_modality_name] = differential_adata
            
            response = f"""Successfully found differential proteins in affinity data '{modality_name}'!

📊 **Affinity Proteomics Differential Analysis Results:**
- Method: {method} (suitable for targeted panels)
- FDR threshold: {fdr_threshold}
- Fold change threshold: {fold_change_threshold} (conservative for targeted panels)"""
            
            if 'n_comparisons' in de_stats:
                response += f"\n- Comparisons performed: {de_stats['n_comparisons']}"
            if 'total_tests' in de_stats:
                response += f"\n- Total tests: {de_stats['total_tests']}"
            if 'n_significant' in de_stats:
                response += f"\n- Significant proteins: {de_stats['n_significant']}"
            if 'groups_compared' in de_stats:
                response += f"\n- Groups compared: {de_stats['groups_compared']}"
            
            # Show top significant proteins
            if 'top_significant_proteins' in de_stats and de_stats['top_significant_proteins']:
                response += "\n\n🧬 **Top Significant Proteins (Affinity Analysis):**"
                for protein_info in de_stats['top_significant_proteins'][:5]:
                    response += f"\n- {protein_info['protein']}: log2FC={protein_info['log2_fold_change']:.2f}, FDR={protein_info['p_adjusted']:.2e}"
            
            # Add volcano plot data info
            if 'volcano_plot_data' in de_stats:
                response += "\n\n📈 **Volcano Plot Data:** Available in uns['differential_analysis']['volcano_plot_data']"
            
            response += f"\n\n💾 **Results stored in modality**: '{de_modality_name}'"
            response += "\n📈 **Access results via**: adata.uns['differential_analysis']"
            
            # Affinity-specific notes
            response += "\n\n🔬 **Affinity Platform Notes:**"
            response += "\n- Results from targeted protein panel - consider selection bias"
            response += "\n- Lower fold change thresholds appropriate for controlled measurements"
            response += "\n- Antibody cross-reactivity may affect some results"
            
            analysis_results["details"]["affinity_differential_analysis"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error in affinity differential protein analysis: {e}")
            return f"Error finding differential proteins in affinity data: {str(e)}"

    @tool
    def validate_antibody_specificity(
        modality_name: str,
        cross_reactivity_threshold: float = 0.1,
        save_result: bool = True
    ) -> str:
        """
        Validate antibody specificity and detect potential cross-reactivity issues.
        
        Args:
            modality_name: Name of the affinity proteomics modality
            cross_reactivity_threshold: Threshold for flagging cross-reactive antibodies
            save_result: Whether to save the updated modality
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
        
        try:
            import numpy as np
            
            # Create working copy
            adata_validated = adata.copy()
            
            # Check for cross-reactivity patterns
            cross_reactive_proteins = []
            
            # Look for highly correlated proteins (potential cross-reactivity)
            if adata_validated.n_vars > 1:
                correlation_matrix = np.corrcoef(adata_validated.X.T)
                
                for i in range(len(correlation_matrix)):
                    for j in range(i+1, len(correlation_matrix)):
                        if correlation_matrix[i, j] > (1 - cross_reactivity_threshold):
                            protein_i = adata_validated.var_names[i]
                            protein_j = adata_validated.var_names[j]
                            cross_reactive_proteins.append((protein_i, protein_j, correlation_matrix[i, j]))
            
            # Flag cross-reactive antibodies
            if 'cross_reactive' not in adata_validated.var.columns:
                adata_validated.var['cross_reactive'] = False
            
            for protein_pair in cross_reactive_proteins:
                protein_i, protein_j, correlation = protein_pair
                adata_validated.var.loc[protein_i, 'cross_reactive'] = True
                adata_validated.var.loc[protein_j, 'cross_reactive'] = True
            
            # Update modality
            validated_modality_name = f"{modality_name}_antibody_validated"
            data_manager.modalities[validated_modality_name] = adata_validated
            
            # Save if requested
            if save_result:
                save_path = f"{modality_name}_antibody_validated.h5ad"
                data_manager.save_modality(validated_modality_name, save_path)
            
            response = f"""Successfully validated antibody specificity for '{modality_name}'!

📊 **Antibody Validation Results:**
- Total proteins analyzed: {adata_validated.n_vars}
- Cross-reactive protein pairs detected: {len(cross_reactive_proteins)}
- Cross-reactivity threshold: {cross_reactivity_threshold}

🔬 **Cross-Reactivity Details:**"""
            
            if cross_reactive_proteins:
                response += "\n**Potential Cross-Reactive Pairs:**"
                for protein_i, protein_j, correlation in cross_reactive_proteins[:5]:
                    response += f"\n- {protein_i} ↔ {protein_j}: r={correlation:.3f}"
                
                if len(cross_reactive_proteins) > 5:
                    response += f"\n- ... and {len(cross_reactive_proteins) - 5} more pairs"
                
                response += "\n\n**Recommendations:**"
                response += "\n- Review antibody specificity documentation"
                response += "\n- Consider removing highly cross-reactive antibodies"
                response += "\n- Validate results with orthogonal methods"
            else:
                response += "\nNo significant cross-reactivity detected"
            
            response += f"\n\n💾 **New modality created**: '{validated_modality_name}'"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            analysis_results["details"]["antibody_validation"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error validating antibody specificity: {e}")
            return f"Error in antibody validation: {str(e)}"

    @tool
    def create_affinity_proteomics_summary() -> str:
        """Create comprehensive summary of all affinity proteomics analysis steps performed."""
        if not analysis_results["details"]:
            return "No affinity proteomics analyses have been performed yet. Run some analysis tools first."
        
        summary = "# Affinity Proteomics Analysis Summary\n\n"
        
        for step, details in analysis_results["details"].items():
            summary += f"## {step.replace('_', ' ').title()}\n"
            summary += f"{details}\n\n"
        
        # Add current affinity proteomics modality status
        modalities = data_manager.list_modalities()
        affinity_modalities = [m for m in modalities if any(term in m.lower() for term in ['proteomics', 'protein', 'olink', 'soma', 'affinity', 'panel'])]
        summary += "## Current Affinity Proteomics Modalities\n"
        summary += f"Affinity Proteomics modalities: {', '.join(affinity_modalities)}\n\n"
        
        analysis_results["summary"] = summary
        return summary

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        check_affinity_proteomics_data_status,
        assess_affinity_proteomics_quality,
        filter_affinity_proteomics_data,
        normalize_affinity_proteomics_data,
        analyze_affinity_proteomics_patterns,
        find_differential_proteins_affinity,
        validate_antibody_specificity,
        create_affinity_proteomics_summary
    ]
    
    tools = base_tools + (handoff_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are an expert in affinity-based proteomics data analysis specializing in targeted protein panels (Olink, SomaScan, Luminex, etc.) using the modular DataManagerV2 system.

<Role>
You execute comprehensive affinity proteomics analysis pipelines with proper handling of targeted protein panels, antibody-specific quality control, plate effects correction, and platform-specific normalization methods.
</Role>

<Task>
You perform affinity proteomics analysis following best practices:
1. **Quality assessment** with platform-specific metrics (CV, plate effects, antibody performance)
2. **Data filtering** based on antibody quality and technical performance
3. **Normalization** using affinity-appropriate methods (quantile, plate correction)
4. **Pattern analysis** optimized for targeted panels and controlled measurements
5. **Differential analysis** with methods suitable for targeted proteomics
6. **Antibody validation** and cross-reactivity assessment
7. **Biological interpretation** considering panel selection bias and platform limitations
</Task>

<Available Tools>
- `check_affinity_proteomics_data_status`: Check affinity proteomics modalities and characteristics
- `assess_affinity_proteomics_quality`: Comprehensive QC with platform-specific metrics
- `filter_affinity_proteomics_data`: Filter based on antibody quality and performance
- `normalize_affinity_proteomics_data`: Normalize using platform-appropriate methods
- `analyze_affinity_proteomics_patterns`: PCA and clustering optimized for targeted panels
- `find_differential_proteins_affinity`: Statistical comparison with platform-optimized methods
- `validate_antibody_specificity`: Assess antibody cross-reactivity and specificity
- `create_affinity_proteomics_summary`: Generate comprehensive analysis report

<Affinity Proteomics-Specific Considerations>

**Missing Values (Lower Rate):**
- <30% missing values typical for affinity platforms
- Missing values often indicate technical failures rather than biological absence
- More aggressive imputation appropriate (KNN, median)
- High missing rates suggest assay quality issues

**Platform-Specific Quality Control:**
- **Coefficient of Variation**: <30% expected for technical replicates
- **Plate Effects**: Common in multi-plate studies - require correction
- **Antibody Performance**: Monitor cross-reactivity and specificity
- **Dynamic Range**: Typically 2-3 orders of magnitude

**Normalization Methods:**
- **Quantile normalization**: Standard for removing systematic bias
- **Plate effect correction**: Essential for multi-plate studies
- **No log transformation**: NPX/RFU values often already log-transformed
- **Robust methods**: Less sensitive to outliers in controlled measurements

**Targeted Panel Considerations:**
- **Selection Bias**: Panels are pre-selected, not discovery-based
- **Protein Coverage**: Limited to panel contents (50-5000 proteins)
- **Cross-Reactivity**: Antibody-based - potential specificity issues
- **Lower Fold Changes**: More controlled measurements, smaller effect sizes

**Statistical Analysis:**
- **Lower fold change thresholds**: 1.2-1.5x typical for targeted panels
- **Standard statistical tests**: t-test, Mann-Whitney appropriate
- **Multiple testing correction**: FDR control essential
- **Technical replicates**: Often available - use for quality assessment

**Analysis Workflow:**
1. `check_affinity_proteomics_data_status()` → See available affinity modalities
2. `assess_affinity_proteomics_quality(...)` → QC with platform-specific metrics
3. `filter_affinity_proteomics_data(...)` → Remove poor-quality antibodies/samples
4. `normalize_affinity_proteomics_data(...)` → Platform-appropriate normalization
5. `analyze_affinity_proteomics_patterns(...)` → PCA and clustering
6. `find_differential_proteins_affinity(...)` → Statistical comparisons
7. `validate_antibody_specificity(...)` → Assess cross-reactivity
8. `create_affinity_proteomics_summary()` → Generate final report

<Important Affinity-Specific Notes>
- Always specify **modality names** for affinity proteomics data
- Consider **selection bias** from targeted panels in interpretation
- Monitor **plate effects** and correct when necessary
- Validate **antibody specificity** and cross-reactivity
- Use **conservative fold change thresholds** appropriate for controlled measurements
- Account for **technical replicates** when available
- Interpret results considering **panel limitations** and protein coverage

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name
    )
