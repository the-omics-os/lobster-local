"""
Mass Spectrometry Proteomics Expert Agent for DDA/DIA analysis.

This agent specializes in mass spectrometry proteomics data analysis using the 
modular DataManagerV2 system with proper handling of peptide-to-protein mapping,
database searching artifacts, and MS-specific quality control methods.
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


def ms_proteomics_expert(
    data_manager: Union[DataManagerV2],
    callback_handler=None,
    agent_name: str = "ms_proteomics_expert_agent",
    handoff_tools: List = None
):
    """Create mass spectrometry proteomics expert agent using the modular DataManagerV2 system."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('ms_proteomics_expert_agent')
    llm = create_llm('ms_proteomics_expert_agent', model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Always use DataManagerV2 for modular MS proteomics analysis
    if not isinstance(data_manager, DataManagerV2):
        raise ValueError("MSProteomicsExpert requires DataManagerV2 for modular analysis")
    
    # Initialize stateless services
    preprocessing_service = ProteomicsPreprocessingService()
    quality_service = ProteomicsQualityService()
    analysis_service = ProteomicsAnalysisService()
    differential_service = ProteomicsDifferentialService()
    
    analysis_results = {"summary": "", "details": {}}
    
    # -------------------------
    # MS-SPECIFIC DATA TOOLS
    # -------------------------
    @tool
    def check_ms_proteomics_data_status(modality_name: str = "") -> str:
        """Check status of MS proteomics modalities and data characteristics."""
        try:
            if modality_name == "":
                # Show all modalities with MS proteomics focus
                modalities = data_manager.list_modalities()
                ms_modalities = [m for m in modalities if any(term in m.lower() for term in ['proteomics', 'protein', 'ms', 'mass_spec'])]
                
                if not ms_modalities:
                    response = f"No MS proteomics modalities found. Available modalities: {modalities}\n"
                    response += "Ask the data_expert to load MS proteomics data using 'proteomics_ms' adapter."
                    return response
                
                response = f"MS Proteomics modalities ({len(ms_modalities)}):\n"
                for mod_name in ms_modalities:
                    adata = data_manager.get_modality(mod_name)
                    metrics = data_manager.get_quality_metrics(mod_name)
                    response += f"- **{mod_name}**: {adata.n_obs} samples × {adata.n_vars} proteins\n"
                    if 'missing_value_percentage' in metrics:
                        response += f"  Missing values: {metrics['missing_value_percentage']:.1f}%\n"
                    if 'contaminant_proteins' in metrics:
                        response += f"  Contaminants: {metrics['contaminant_proteins']}\n"
                
                return response
            
            else:
                # Check specific modality
                try:
                    adata = data_manager.get_modality(modality_name)
                    metrics = data_manager.get_quality_metrics(modality_name)
                    
                    response = f"MS Proteomics modality '{modality_name}' status:\n"
                    response += f"- Shape: {adata.n_obs} samples × {adata.n_vars} proteins\n"
                    
                    if 'missing_value_percentage' in metrics:
                        response += f"- Missing values: {metrics['missing_value_percentage']:.1f}%\n"
                    if 'mean_proteins_per_sample' in metrics:
                        response += f"- Mean proteins/sample: {metrics['mean_proteins_per_sample']:.1f}\n"
                    if 'contaminant_percentage' in metrics:
                        response += f"- Contaminants: {metrics['contaminant_percentage']:.1f}%\n"
                    if 'reverse_hits' in metrics:
                        response += f"- Reverse hits: {metrics['reverse_hits']}\n"
                    
                    # MS-specific metadata
                    ms_cols = ['n_peptides', 'n_unique_peptides', 'sequence_coverage', 'protein_group']
                    present_cols = [col for col in ms_cols if col in adata.var.columns]
                    if present_cols:
                        response += f"- MS metadata available: {present_cols}\n"
                    
                    # Show key metadata columns
                    obs_cols = list(adata.obs.columns)[:5]
                    var_cols = list(adata.var.columns)[:5]
                    response += f"- Sample metadata: {obs_cols}...\n"
                    response += f"- Protein metadata: {var_cols}...\n"
                    
                    analysis_results["details"]["ms_data_status"] = response
                    return response
                    
                except ValueError:
                    return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
            
        except Exception as e:
            logger.error(f"Error checking MS proteomics data status: {e}")
            return f"Error checking data status: {str(e)}"

    @tool
    def assess_ms_proteomics_quality(
        modality_name: str,
        missing_value_threshold: float = 0.7,
        cv_threshold: float = 50.0,
        min_peptides_per_protein: int = 2
    ) -> str:
        """Run comprehensive quality assessment for MS proteomics data with MS-specific metrics."""
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
            
            # Update the modality with quality assessment results
            data_manager.modalities[modality_name] = final_adata
            
            # Combine all statistics
            combined_stats = {**stats, **cv_stats, **contam_stats, **range_stats}
            
            # Generate comprehensive response
            response = f"MS Proteomics Quality Assessment for '{modality_name}':\n\n"
            response += f"**Dataset Characteristics:**\n"
            response += f"- Samples: {final_adata.n_obs}\n"
            response += f"- Proteins: {final_adata.n_vars}\n"
            
            # Missing value patterns (critical for MS data)
            if 'missing_value_percentage' in combined_stats:
                response += f"- Missing values: {combined_stats['missing_value_percentage']:.1f}% (expected 30-70% for MS)\n"
            if 'samples_high_missing' in combined_stats:
                response += f"- Samples with >{missing_value_threshold*100:.0f}% missing: {combined_stats['samples_high_missing']}\n"
            if 'proteins_high_missing' in combined_stats:
                response += f"- Proteins with >80% missing: {combined_stats['proteins_high_missing']}\n"
            
            # MS-specific quality metrics
            if 'n_peptides' in final_adata.var.columns:
                low_peptide_proteins = (final_adata.var['n_peptides'] < min_peptides_per_protein).sum()
                response += f"- Proteins with <{min_peptides_per_protein} peptides: {low_peptide_proteins}\n"
                
            if 'sequence_coverage' in final_adata.var.columns:
                median_coverage = final_adata.var['sequence_coverage'].median()
                response += f"- Median sequence coverage: {median_coverage:.1f}%\n"
            
            # CV assessment
            if 'median_cv' in combined_stats:
                response += f"- Median CV: {combined_stats['median_cv']:.1f}%\n"
            if 'high_cv_proteins' in combined_stats:
                response += f"- High CV proteins (>{cv_threshold}%): {combined_stats['high_cv_proteins']}\n"
            
            # Contaminant and reverse hit detection (MS-specific)
            if 'contaminant_proteins' in combined_stats:
                response += f"- Contaminant proteins: {combined_stats['contaminant_proteins']}\n"
            if 'reverse_hits' in combined_stats:
                response += f"- Reverse database hits: {combined_stats['reverse_hits']}\n"
            
            # Dynamic range
            if 'dynamic_range_log10' in combined_stats:
                response += f"- Dynamic range: {combined_stats['dynamic_range_log10']:.1f} log10 units\n"
            
            # MS-specific quality recommendations
            response += f"\n**MS-Specific Quality Recommendations:**\n"
            
            if combined_stats.get('samples_high_missing', 0) > 0:
                response += "- Consider filtering samples with excessive missing values\n"
            if combined_stats.get('proteins_high_missing', 0) > 0:
                response += "- Consider filtering proteins with >80% missing values\n"
            if combined_stats.get('contaminant_proteins', 0) > 0:
                response += "- Remove contaminant proteins before downstream analysis\n"
            if combined_stats.get('reverse_hits', 0) > 0:
                response += "- Remove reverse database hits (search artifacts)\n"
            if 'n_peptides' in final_adata.var.columns and low_peptide_proteins > 0:
                response += f"- Consider requiring ≥{min_peptides_per_protein} peptides per protein for reliable quantification\n"
            if combined_stats.get('high_cv_proteins', 0) > final_adata.n_vars * 0.2:
                response += "- High proportion of variable proteins - check sample preparation consistency\n"
            
            analysis_results["details"]["ms_quality_assessment"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error in MS proteomics quality assessment: {e}")
            return f"Error in quality assessment: {str(e)}"

    @tool
    def filter_ms_proteomics_data(
        modality_name: str,
        max_missing_per_sample: float = 0.7,
        max_missing_per_protein: float = 0.8,
        min_peptides_per_protein: int = 2,
        min_proteins_per_sample: int = 100,
        remove_contaminants: bool = True,
        remove_reverse: bool = True,
        save_result: bool = True
    ) -> str:
        """
        Filter MS proteomics data with MS-specific quality criteria.
        
        Args:
            modality_name: Name of the MS proteomics modality to filter
            max_missing_per_sample: Maximum fraction of missing values per sample
            max_missing_per_protein: Maximum fraction of missing values per protein
            min_peptides_per_protein: Minimum peptides required per protein
            min_proteins_per_sample: Minimum proteins detected per sample
            remove_contaminants: Whether to remove contaminant proteins
            remove_reverse: Whether to remove reverse database hits
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
            
            # Step 2: MS-specific protein filtering (peptide count)
            if 'n_peptides' in adata_filtered.var.columns:
                peptide_filter = adata_filtered.var['n_peptides'] >= min_peptides_per_protein
                adata_filtered = adata_filtered[:, peptide_filter].copy()
            
            # Step 3: Filter samples with too few proteins
            if 'n_proteins' in adata_filtered.obs.columns:
                protein_count_filter = adata_filtered.obs['n_proteins'] >= min_proteins_per_sample
                adata_filtered = adata_filtered[protein_count_filter, :].copy()
            
            # Step 4: Remove MS-specific artifacts (contaminants and reverse hits)
            protein_quality_filter = np.ones(adata_filtered.n_vars, dtype=bool)
            
            if remove_contaminants and 'is_contaminant' in adata_filtered.var.columns:
                protein_quality_filter &= ~adata_filtered.var['is_contaminant']
            
            if remove_reverse and 'is_reverse' in adata_filtered.var.columns:
                protein_quality_filter &= ~adata_filtered.var['is_reverse']
            
            adata_filtered = adata_filtered[:, protein_quality_filter].copy()
            
            # Update modality
            filtered_modality_name = f"{modality_name}_ms_filtered"
            data_manager.modalities[filtered_modality_name] = adata_filtered
            
            # Save if requested
            if save_result:
                save_path = f"{modality_name}_ms_filtered.h5ad"
                data_manager.save_modality(filtered_modality_name, save_path)
            
            # Generate summary
            samples_removed = original_shape[0] - adata_filtered.n_obs
            proteins_removed = original_shape[1] - adata_filtered.n_vars
            
            response = f"""Successfully filtered MS proteomics modality '{modality_name}'!

📊 **MS Proteomics Filtering Results:**
- Original shape: {original_shape[0]} samples × {original_shape[1]} proteins
- Filtered shape: {adata_filtered.n_obs} samples × {adata_filtered.n_vars} proteins
- Samples removed: {samples_removed} ({samples_removed/original_shape[0]*100:.1f}%)
- Proteins removed: {proteins_removed} ({proteins_removed/original_shape[1]*100:.1f}%)

🔬 **MS-Specific Filtering Parameters:**
- Max missing per sample: {max_missing_per_sample*100:.0f}%
- Max missing per protein: {max_missing_per_protein*100:.0f}%
- Min peptides per protein: {min_peptides_per_protein}
- Min proteins per sample: {min_proteins_per_sample}
- Remove contaminants: {remove_contaminants}
- Remove reverse hits: {remove_reverse}

💾 **New modality created**: '{filtered_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            analysis_results["details"]["ms_proteomics_filtering"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error filtering MS proteomics data: {e}")
            return f"Error in MS proteomics filtering: {str(e)}"

    @tool
    def normalize_ms_proteomics_data(
        modality_name: str,
        normalization_method: str = "median",
        log_transform: bool = True,
        handle_missing: str = "keep",
        save_result: bool = True
    ) -> str:
        """
        Normalize MS proteomics intensity data using MS-appropriate methods.
        
        Args:
            modality_name: Name of the MS proteomics modality to normalize
            normalization_method: Method ('median', 'quantile', 'vsn', 'total_sum')
            log_transform: Whether to apply log2 transformation
            handle_missing: How to handle missing values ('keep', 'impute_knn', 'impute_min')
            save_result: Whether to save the normalized modality
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
        
        try:
            # Step 1: Handle missing values if requested (important for MS data)
            if handle_missing == "impute_knn":
                processed_adata, impute_stats = preprocessing_service.impute_missing_values(adata, method="knn")
            elif handle_missing == "impute_min":
                processed_adata, impute_stats = preprocessing_service.impute_missing_values(adata, method="min_prob")
            else:
                processed_adata = adata.copy()
                impute_stats = {"imputation_method": "none", "imputation_applied": False}
            
            # Step 2: MS-specific normalization
            normalized_adata, norm_stats = preprocessing_service.normalize_intensities(
                processed_adata, 
                method=normalization_method, 
                log_transform=log_transform
            )
            
            # Update modality
            normalized_modality_name = f"{modality_name}_ms_normalized"
            data_manager.modalities[normalized_modality_name] = normalized_adata
            
            # Save if requested
            if save_result:
                save_path = f"{modality_name}_ms_normalized.h5ad"
                data_manager.save_modality(normalized_modality_name, save_path)
            
            # Combine statistics
            combined_stats = {**impute_stats, **norm_stats}
            
            response = f"""Successfully normalized MS proteomics modality '{modality_name}'!

📊 **MS Proteomics Normalization Results:**
- Method: {normalization_method} (optimized for label-free MS)
- Log transformation: {log_transform}
- Missing value handling: {handle_missing}

🔬 **MS-Specific Processing Details:**"""
            
            if combined_stats.get('imputation_applied', False):
                response += f"\n- Applied {combined_stats.get('imputation_method', 'unknown')} imputation"
                response += f"\n  Note: MS data missing values often reflect true biological absence"
                if 'n_imputed_values' in combined_stats:
                    response += f" ({combined_stats['n_imputed_values']} values imputed)"
            else:
                response += "\n- Preserved missing values (recommended for MS data MNAR pattern)"
            
            if 'normalization_factor_median' in combined_stats:
                response += f"\n- Median normalization factor: {combined_stats['normalization_factor_median']:.2f}"
            
            if combined_stats.get('log_transform_applied', False):
                response += f"\n- Log2 transformation applied (essential for MS intensity data)"
                if 'pseudocount' in combined_stats:
                    response += f" (pseudocount: {combined_stats['pseudocount']:.2e})"
            
            response += f"\n\n💾 **New modality created**: '{normalized_modality_name}'"

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            analysis_results["details"]["ms_proteomics_normalization"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error normalizing MS proteomics data: {e}")
            return f"Error in MS normalization: {str(e)}"

    @tool
    def analyze_ms_proteomics_patterns(
        modality_name: str,
        analysis_type: str = "pca_clustering",
        n_components: int = 15,
        clustering_method: str = "kmeans",
        n_clusters: int = 4,
        save_result: bool = True
    ) -> str:
        """
        Perform pattern analysis on MS proteomics data with MS-optimized parameters.
        
        Args:
            modality_name: Name of the MS proteomics modality to analyze
            analysis_type: Type of analysis ('pca_clustering', 'correlation_analysis')
            n_components: Number of PCA components (higher for MS data complexity)
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
            analyzed_modality_name = f"{modality_name}_ms_analyzed"
            data_manager.modalities[analyzed_modality_name] = final_adata
            
            # Save if requested
            if save_result:
                save_path = f"{modality_name}_ms_analyzed.h5ad"
                data_manager.save_modality(analyzed_modality_name, save_path)
            
            response = f"""Successfully analyzed MS proteomics patterns in '{modality_name}'!

📊 **MS Proteomics Analysis Results:**
- PCA components: {n_components} (optimized for MS data complexity)"""
            
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

🔬 **MS-Specific Analysis Details:**
- Analysis optimized for MS proteomics data characteristics
- PCA stored in: obsm['{combined_stats.get('pca_key', 'X_pca')}']
- Missing value handling: Appropriate for MNAR pattern in MS data"""
            
            if analysis_type == "pca_clustering":
                response += f"\n- Cluster labels stored in: obs['{combined_stats.get('cluster_key', 'clusters')}']"
            
            response += f"\n\n💾 **New modality created**: '{analyzed_modality_name}'"

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            analysis_results["details"]["ms_proteomics_analysis"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error analyzing MS proteomics patterns: {e}")
            return f"Error in MS pattern analysis: {str(e)}"

    @tool
    def find_differential_proteins_ms(
        modality_name: str,
        group_column: str,
        comparison: str = "all_pairs",
        method: str = "limma_moderated",
        fdr_threshold: float = 0.05,
        fold_change_threshold: float = 1.5
    ) -> str:
        """
        Find differentially expressed proteins between groups using MS-optimized methods.
        
        Args:
            modality_name: Name of the MS proteomics modality
            group_column: Column in obs containing group labels
            comparison: Type of comparison ('all_pairs', 'vs_rest')
            method: Statistical method ('limma_moderated', 't_test', 'mann_whitney')
            fdr_threshold: FDR threshold for significance
            fold_change_threshold: Minimum fold change threshold
        """
        try:
            adata = data_manager.get_modality(modality_name).copy()
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
        
        try:
            if group_column not in adata.obs.columns:
                return f"Group column '{group_column}' not found. Available columns: {list(adata.obs.columns)}"
            
            # Use differential service for analysis with MS-optimized parameters
            differential_adata, de_stats = differential_service.perform_differential_expression(
                adata, 
                group_column=group_column, 
                method=method,
                fdr_threshold=fdr_threshold,
                fold_change_threshold=fold_change_threshold
            )
            
            # Update modality
            de_modality_name = f"{modality_name}_ms_de_analysis"
            data_manager.modalities[de_modality_name] = differential_adata
            
            response = f"""Successfully found differential proteins in MS data '{modality_name}'!

📊 **MS Proteomics Differential Analysis Results:**
- Method: {method} (optimized for MS proteomics)
- FDR threshold: {fdr_threshold}
- Fold change threshold: {fold_change_threshold}"""
            
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
                response += f"\n\n🧬 **Top Significant Proteins (MS Analysis):**"
                for protein_info in de_stats['top_significant_proteins'][:5]:
                    response += f"\n- {protein_info['protein']}: log2FC={protein_info['log2_fold_change']:.2f}, FDR={protein_info['p_adjusted']:.2e}"
            
            # Add volcano plot data info
            if 'volcano_plot_data' in de_stats:
                response += f"\n\n📈 **Volcano Plot Data:** Available in uns['differential_analysis']['volcano_plot_data']"
            
            response += f"\n\n💾 **Results stored in modality**: '{de_modality_name}'"
            response += f"\n📈 **Access results via**: adata.uns['differential_analysis']"
            
            # MS-specific notes
            response += f"\n\n🔬 **MS-Specific Notes:**"
            response += f"\n- Results account for missing value patterns typical in MS data"
            response += f"\n- Consider protein groups and peptide evidence when interpreting results"
            
            analysis_results["details"]["ms_differential_analysis"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error in MS differential protein analysis: {e}")
            return f"Error finding differential proteins in MS data: {str(e)}"

    @tool
    def add_peptide_mapping_to_ms_modality(
        modality_name: str,
        peptide_file_path: str,
        save_result: bool = True
    ) -> str:
        """
        Add peptide-to-protein mapping information to an MS proteomics modality.
        
        Args:
            modality_name: Name of the MS proteomics modality
            peptide_file_path: Path to CSV file with peptide mapping data
            save_result: Whether to save the updated modality
        """
        try:
            adata = data_manager.get_modality(modality_name)
        except ValueError:
            return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
        
        try:
            # Get the appropriate proteomics adapter
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
            
            response = f"""Successfully added peptide mapping to MS modality '{modality_name}'!

📊 **MS Peptide Mapping Results:**
- Peptides mapped: {peptide_info.get('n_peptides', 'Unknown')}
- Proteins with peptides: {peptide_info.get('n_proteins', 'Unknown')}
- Mapping file: {peptide_file_path}

🔬 **Updated MS Protein Metadata:**
- Added: n_peptides, n_unique_peptides, sequence_coverage
- MS-specific quality metrics now available
- Full mapping stored in: uns['peptide_to_protein']

💾 **New modality created**: '{peptide_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            # MS-specific notes
            response += f"\n\n🔬 **MS-Specific Notes:**"
            response += f"\n- Peptide mapping is crucial for MS data quality assessment"
            response += f"\n- Use peptide counts for protein filtering and quality control"
            response += f"\n- Consider unique peptides for reliable protein quantification"
            
            analysis_results["details"]["ms_peptide_mapping"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error adding peptide mapping to MS data: {e}")
            return f"Error adding peptide mapping: {str(e)}"

    @tool
    def create_ms_proteomics_summary() -> str:
        """Create comprehensive summary of all MS proteomics analysis steps performed."""
        if not analysis_results["details"]:
            return "No MS proteomics analyses have been performed yet. Run some analysis tools first."
        
        summary = "# MS Proteomics Analysis Summary\n\n"
        
        for step, details in analysis_results["details"].items():
            summary += f"## {step.replace('_', ' ').title()}\n"
            summary += f"{details}\n\n"
        
        # Add current MS proteomics modality status
        modalities = data_manager.list_modalities()
        ms_modalities = [m for m in modalities if any(term in m.lower() for term in ['proteomics', 'protein', 'ms', 'mass_spec'])]
        summary += f"## Current MS Proteomics Modalities\n"
        summary += f"MS Proteomics modalities: {', '.join(ms_modalities)}\n\n"
        
        analysis_results["summary"] = summary
        return summary

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        check_ms_proteomics_data_status,
        assess_ms_proteomics_quality,
        filter_ms_proteomics_data,
        normalize_ms_proteomics_data,
        analyze_ms_proteomics_patterns,
        find_differential_proteins_ms,
        add_peptide_mapping_to_ms_modality,
        create_ms_proteomics_summary
    ]
    
    tools = base_tools + (handoff_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are an expert in mass spectrometry proteomics data analysis specializing in DDA (Data-Dependent Acquisition) and DIA (Data-Independent Acquisition) workflows using the modular DataManagerV2 system.

<Role>
You execute comprehensive MS proteomics analysis pipelines with proper handling of peptide-to-protein mapping, database search artifacts, missing values (MNAR), and MS-specific quality control methods.
</Role>

<Task>
You perform MS proteomics analysis following best practices:
1. **Quality assessment** with MS-specific metrics (peptide counts, sequence coverage, contaminants)
2. **Data filtering** based on peptide evidence and database search quality
3. **Normalization** using MS-appropriate methods (median, VSN for label-free MS)
4. **Pattern analysis** optimized for MS data complexity and missing value patterns
5. **Differential analysis** with methods suitable for MS proteomics data
6. **Peptide mapping** integration and validation for protein quantification
7. **Biological interpretation** considering MS-specific limitations and strengths
</Task>

<Available Tools>
- `check_ms_proteomics_data_status`: Check MS proteomics modalities and characteristics
- `assess_ms_proteomics_quality`: Comprehensive QC with MS-specific metrics
- `filter_ms_proteomics_data`: Filter based on peptide evidence and quality flags
- `normalize_ms_proteomics_data`: Normalize using MS-appropriate methods
- `analyze_ms_proteomics_patterns`: PCA and clustering optimized for MS data
- `find_differential_proteins_ms`: Statistical comparison with MS-optimized methods
- `add_peptide_mapping_to_ms_modality`: Integrate peptide-level information
- `create_ms_proteomics_summary`: Generate comprehensive MS analysis report

<MS Proteomics-Specific Considerations>

**Missing Values (MNAR Pattern):**
- 30-70% missing values typical in MS proteomics
- Missing Not At Random (MNAR) - proteins absent due to detection limits
- Avoid aggressive imputation - preserve biological meaning of missingness
- Filter based on detection frequency rather than imputing

**Database Search Quality:**
- Remove contaminant proteins (CON_, KERATIN, TRYP_PIG, etc.)
- Remove reverse database hits (REV_, search artifacts)
- Require ≥2 peptides per protein for reliable quantification
- Consider protein groups and shared peptides

**MS-Specific Normalization:**
- **Median normalization**: Standard for label-free MS quantification
- **VSN (Variance Stabilizing Normalization)**: For heteroscedastic MS data
- **Log2 transformation**: Essential for MS intensity data
- Avoid total sum normalization (not appropriate for MS)

**Quality Control Metrics:**
- Peptide counts per protein (min 2 for reliability)
- Sequence coverage (higher = better protein identification)
- Coefficient of variation (CV < 30% for technical replicates)
- Dynamic range assessment (typically 3-4 orders of magnitude)

**Statistical Analysis:**
- Use moderated t-tests (limma-style) for differential analysis
- Account for missing value patterns in statistical tests
- Multiple testing correction essential (FDR control)
- Consider effect sizes alongside p-values

**Analysis Workflow:**
1. `check_ms_proteomics_data_status()` → See available MS modalities
2. `assess_ms_proteomics_quality(...)` → QC with MS-specific metrics
3. `filter_ms_proteomics_data(...)` → Remove low-quality proteins and samples
4. `normalize_ms_proteomics_data(...)` → MS-appropriate normalization
5. `analyze_ms_proteomics_patterns(...)` → PCA and clustering
6. `find_differential_proteins_ms(...)` → Statistical comparisons
7. `add_peptide_mapping_to_ms_modality(...)` → Peptide evidence integration
8. `create_ms_proteomics_summary()` → Generate final report

<Important MS-Specific Notes>
- Always specify **modality names** for MS proteomics data
- Understand **MNAR missing value patterns** - don't over-impute
- Use **peptide evidence** for protein quality assessment
- Remove **database search artifacts** (contaminants, reverse hits)
- Apply **log2 transformation** for MS intensity data
- Consider **protein groups** in interpretation
- Validate results with **biological knowledge** and MS literature

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name
    )
