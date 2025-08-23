"""
Transcriptomics Expert Agent for single-cell and bulk RNA-seq analysis.

This agent specializes in transcriptomics analysis using the modular DataManagerV2 
system with proper modality handling and schema enforcement.
"""

from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

from datetime import date

import pandas as pd

from lobster.agents.state import TranscriptomicsExpertState
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.preprocessing_service import PreprocessingService, PreprocessingError
from lobster.tools.quality_service import QualityService, QualityError
from lobster.tools.clustering_service import ClusteringService, ClusteringError
from lobster.tools.enhanced_singlecell_service import EnhancedSingleCellService, SingleCellError
from lobster.tools.bulk_rnaseq_service import BulkRNASeqService, BulkRNASeqError
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class TranscriptomicsError(Exception):
    """Base exception for transcriptomics operations."""
    pass


class ModalityNotFoundError(TranscriptomicsError):
    """Raised when requested modality doesn't exist."""
    pass


def transcriptomics_expert(
    data_manager: DataManagerV2, 
    callback_handler=None, 
    agent_name: str = "transcriptomics_expert_agent",
    handoff_tools: List = None
):  
    """Create transcriptomics expert agent using DataManagerV2 and modular services."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('transcriptomics_expert')
    llm = ChatBedrockConverse(**model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Initialize stateless services
    preprocessing_service = PreprocessingService()
    quality_service = QualityService()
    clustering_service = ClusteringService()
    singlecell_service = EnhancedSingleCellService()
    bulk_service = BulkRNASeqService()
    
    analysis_results = {"summary": "", "details": {}}
    
    # -------------------------
    # DATA STATUS TOOLS
    # -------------------------
    @tool
    def check_data_status(modality_name: str = "") -> str:
        """Check if data is loaded and ready for analysis."""
        try:
            if modality_name == "":
                modalities = data_manager.list_modalities()
                if not modalities:
                    return "No modalities loaded. Please ask the data expert to load a dataset first."
                
                response = f"Available modalities ({len(modalities)}):\n"
                for mod_name in modalities:
                    adata = data_manager.get_modality(mod_name)
                    response += f"- **{mod_name}**: {adata.n_obs} obs × {adata.n_vars} vars\n"
                
                response += "\nSpecify a modality_name to check specific modality status."
                return response
            
            else:
                # Check specific modality
                if modality_name not in data_manager.list_modalities():
                    return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                
                adata = data_manager.get_modality(modality_name)
                metrics = data_manager.get_quality_metrics(modality_name)
                
                response = f"Modality '{modality_name}' ready for analysis:\n"
                response += f"- Shape: {adata.n_obs} obs × {adata.n_vars} vars\n"
                response += f"- Obs columns: {list(adata.obs.columns)[:5]}...\n"
                response += f"- Var columns: {list(adata.var.columns)[:5]}...\n"
                
                if 'total_counts' in metrics:
                    response += f"- Total counts: {metrics['total_counts']:,.0f}\n"
                if 'mean_counts_per_obs' in metrics:
                    response += f"- Mean counts/obs: {metrics['mean_counts_per_obs']:.1f}\n"
                
                analysis_results["details"]["data_status"] = response
                return response
                
        except Exception as e:
            logger.error(f"Error checking data status: {e}")
            return f"Error checking data status: {str(e)}"

    @tool
    def assess_data_quality(
        modality_name: str,
        min_genes: int = 500,
        max_mt_pct: float = 20.0,
        max_ribo_pct: float = 50.0,
        min_housekeeping_score: float = 1.0
    ) -> str:
        """Run comprehensive quality control assessment on a transcriptomics modality."""
        try:
            if modality_name == "":
                return "Please specify modality_name for quality assessment. Use check_data_status() to see available modalities."
            
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Run quality assessment using service
            adata_qc, assessment_stats = quality_service.assess_quality(
                adata=adata,
                min_genes=min_genes,
                max_mt_pct=max_mt_pct,
                max_ribo_pct=max_ribo_pct,
                min_housekeeping_score=min_housekeeping_score
            )
            
            # Create new modality with QC annotations
            qc_modality_name = f"{modality_name}_quality_assessed"
            data_manager.modalities[qc_modality_name] = adata_qc
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="assess_data_quality",
                parameters={
                    "modality_name": modality_name,
                    "min_genes": min_genes,
                    "max_mt_pct": max_mt_pct,
                    "max_ribo_pct": max_ribo_pct
                },
                description=f"Quality assessment for {modality_name}"
            )
            
            # Format professional response
            response = f"""Quality Assessment Complete for '{modality_name}'!

📊 **Assessment Results:**
- Cells analyzed: {assessment_stats['cells_before_qc']:,}
- Cells passing QC: {assessment_stats['cells_after_qc']:,} ({assessment_stats['cells_retained_pct']:.1f}%)
- Quality status: {assessment_stats['quality_status']}

📈 **Quality Metrics:**
- Mean genes per cell: {assessment_stats['mean_genes_per_cell']:.0f}
- Mean mitochondrial %: {assessment_stats['mean_mt_pct']:.1f}%
- Mean ribosomal %: {assessment_stats['mean_ribo_pct']:.1f}%
- Mean total counts: {assessment_stats['mean_total_counts']:.0f}

💡 **Quality Summary:**
{assessment_stats['qc_summary']}

💾 **New modality created**: '{qc_modality_name}' (with QC annotations)

Use the QC annotations for filtering or proceed with preprocessing."""
            
            analysis_results["details"]["quality_assessment"] = response
            return response
            
        except QualityError as e:
            logger.error(f"Quality assessment error: {e}")
            return f"Quality assessment failed: {str(e)}"
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return f"Error in quality assessment: {str(e)}"

    # -------------------------
    # PREPROCESSING TOOLS FOR MODULAR SYSTEM
    # -------------------------
    @tool
    def filter_and_normalize_modality(
        modality_name: str,
        min_genes_per_cell: int = 200,
        max_genes_per_cell: int = 5000,
        min_cells_per_gene: int = 3,
        max_mito_percent: float = 20.0,
        normalization_method: str = "log1p",
        target_sum: int = 10000,
        save_result: bool = True
    ) -> str:
        """
        Filter and normalize a transcriptomics modality using professional QC standards.
        
        Args:
            modality_name: Name of the modality to process
            min_genes_per_cell: Minimum number of genes expressed per cell
            max_genes_per_cell: Maximum number of genes expressed per cell
            min_cells_per_gene: Minimum number of cells expressing each gene
            max_mito_percent: Maximum mitochondrial gene percentage
            normalization_method: Normalization method ('log1p', 'sctransform_like')
            target_sum: Target sum for normalization
            save_result: Whether to save the filtered modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Processing modality '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
            # Use preprocessing service
            adata_processed, processing_stats = preprocessing_service.filter_and_normalize_cells(
                adata=adata,
                min_genes_per_cell=min_genes_per_cell,
                max_genes_per_cell=max_genes_per_cell,
                min_cells_per_gene=min_cells_per_gene,
                max_mito_percent=max_mito_percent,
                normalization_method=normalization_method,
                target_sum=target_sum
            )
            
            # Save as new modality
            filtered_modality_name = f"{modality_name}_filtered_normalized"
            data_manager.modalities[filtered_modality_name] = adata_processed
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_filtered_normalized.h5ad"
                data_manager.save_modality(filtered_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="filter_and_normalize_modality",
                parameters={
                    "modality_name": modality_name,
                    "min_genes_per_cell": min_genes_per_cell,
                    "max_genes_per_cell": max_genes_per_cell,
                    "max_mito_percent": max_mito_percent,
                    "normalization_method": normalization_method,
                    "target_sum": target_sum
                },
                description=f"Filtered and normalized {modality_name}"
            )
            
            # Format professional response
            original_shape = processing_stats['original_shape']
            final_shape = processing_stats['final_shape']
            cells_removed_pct = processing_stats['cells_retained_pct']
            genes_removed_pct = processing_stats['genes_retained_pct']
            
            response = f"""Successfully filtered and normalized modality '{modality_name}'!

📊 **Filtering Results:**
- Original: {original_shape[0]:,} cells × {original_shape[1]:,} genes
- Filtered: {final_shape[0]:,} cells × {final_shape[1]:,} genes  
- Cells retained: {cells_removed_pct:.1f}%
- Genes retained: {genes_removed_pct:.1f}%

🔬 **Processing Parameters:**
- Min genes/cell: {min_genes_per_cell}
- Max genes/cell: {max_genes_per_cell}
- Min cells/gene: {min_cells_per_gene}
- Max mitochondrial %: {max_mito_percent}%
- Normalization: {normalization_method} (target_sum={target_sum:,})

💾 **New modality created**: '{filtered_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            response += f"\n\nUse this modality for downstream analysis like clustering or differential expression."
            
            analysis_results["details"]["filter_normalize"] = response
            return response
            
        except (PreprocessingError, ModalityNotFoundError) as e:
            logger.error(f"Error in filtering/normalization: {e}")
            return f"Error filtering and normalizing modality: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in filtering/normalization: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def cluster_modality(
        modality_name: str,
        resolution: float = 0.5,
        batch_correction: bool = False,
        batch_key: str = None,
        demo_mode: bool = False,
        save_result: bool = True
    ) -> str:
        """
        Perform clustering and UMAP visualization on a transcriptomics modality.
        
        Args:
            modality_name: Name of the modality to cluster
            resolution: Clustering resolution (0.1-2.0, higher = more clusters)
            batch_correction: Whether to perform batch correction
            batch_key: Column name for batch information (auto-detected if None)
            demo_mode: Use faster processing for large datasets
            save_result: Whether to save the clustered modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Clustering modality '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
            # Use clustering service
            adata_clustered, clustering_stats = clustering_service.cluster_and_visualize(
                adata=adata,
                resolution=resolution,
                batch_correction=batch_correction,
                batch_key=batch_key,
                demo_mode=demo_mode
            )
            
            # Save as new modality
            clustered_modality_name = f"{modality_name}_clustered"
            data_manager.modalities[clustered_modality_name] = adata_clustered
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_clustered.h5ad"
                data_manager.save_modality(clustered_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="cluster_modality",
                parameters={
                    "modality_name": modality_name,
                    "resolution": resolution,
                    "batch_correction": batch_correction,
                    "demo_mode": demo_mode
                },
                description=f"Clustered {modality_name} into {clustering_stats['n_clusters']} clusters"
            )
            
            # Format professional response
            response = f"""Successfully clustered modality '{modality_name}'!

📊 **Clustering Results:**
- Number of clusters: {clustering_stats['n_clusters']}
- Resolution used: {clustering_stats['resolution']}
- UMAP coordinates: {'✓' if clustering_stats['has_umap'] else '✗'}
- Marker genes: {'✓' if clustering_stats['has_marker_genes'] else '✗'}

🔬 **Processing Details:**
- Original shape: {clustering_stats['original_shape'][0]} × {clustering_stats['original_shape'][1]}
- Final shape: {clustering_stats['final_shape'][0]} × {clustering_stats['final_shape'][1]}
- Batch correction: {'✓' if clustering_stats['batch_correction'] else '✗'}
- Demo mode: {'✓' if clustering_stats['demo_mode'] else '✗'}

📈 **Cluster Distribution:**"""
            
            # Add cluster size information
            for cluster_id, size in list(clustering_stats['cluster_sizes'].items())[:5]:
                percentage = (size / clustering_stats['final_shape'][0]) * 100
                response += f"\n- Cluster {cluster_id}: {size} cells ({percentage:.1f}%)"
            
            if len(clustering_stats['cluster_sizes']) > 5:
                response += f"\n... and {len(clustering_stats['cluster_sizes']) - 5} more clusters"
            
            response += f"\n\n💾 **New modality created**: '{clustered_modality_name}'"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
                
            response += f"\n\nUse this modality for marker gene identification or cell type annotation."
            
            analysis_results["details"]["clustering"] = response
            return response
            
        except (ClusteringError, ModalityNotFoundError) as e:
            logger.error(f"Error in clustering: {e}")
            return f"Error clustering modality: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in clustering: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def detect_doublets_in_modality(
        modality_name: str,
        expected_doublet_rate: float = 0.06,
        threshold: float = None,
        save_result: bool = True
    ) -> str:
        """
        Detect doublets in single-cell data using Scrublet or fallback methods.
        
        Args:
            modality_name: Name of the modality to process
            expected_doublet_rate: Expected doublet rate (typically 0.05-0.1)
            threshold: Custom threshold for doublet calling (None for automatic)
            save_result: Whether to save results
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Detecting doublets in modality '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
            # Use singlecell service for doublet detection
            adata_doublets, detection_stats = singlecell_service.detect_doublets(
                adata=adata,
                expected_doublet_rate=expected_doublet_rate,
                threshold=threshold
            )
            
            # Save as new modality
            doublet_modality_name = f"{modality_name}_doublets_detected"
            data_manager.modalities[doublet_modality_name] = adata_doublets
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_doublets_detected.h5ad"
                data_manager.save_modality(doublet_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="detect_doublets_in_modality",
                parameters={
                    "modality_name": modality_name,
                    "expected_doublet_rate": expected_doublet_rate,
                    "threshold": threshold
                },
                description=f"Detected {detection_stats['n_doublets_detected']} doublets in {modality_name}"
            )
            
            # Format professional response
            response = f"""Successfully detected doublets in modality '{modality_name}'!

📊 **Doublet Detection Results:**
- Method: {detection_stats['detection_method']}
- Cells analyzed: {detection_stats['n_cells_analyzed']:,}
- Doublets detected: {detection_stats['n_doublets_detected']} ({detection_stats['actual_doublet_rate']:.1%})
- Expected rate: {detection_stats['expected_doublet_rate']:.1%}

📈 **Doublet Score Statistics:**
- Min score: {detection_stats['doublet_score_stats']['min']:.3f}
- Max score: {detection_stats['doublet_score_stats']['max']:.3f}
- Mean score: {detection_stats['doublet_score_stats']['mean']:.3f}

🔬 **Added to observations:**
- 'doublet_score': Doublet likelihood score
- 'predicted_doublet': Boolean doublet classification

💾 **New modality created**: '{doublet_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            response += f"\n\nUse the doublet annotations to filter cells or proceed with clustering."
            
            analysis_results["details"]["doublet_detection"] = response
            return response
            
        except (SingleCellError, ModalityNotFoundError) as e:
            logger.error(f"Error in doublet detection: {e}")
            return f"Error detecting doublets: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in doublet detection: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def find_marker_genes_for_clusters(
        modality_name: str,
        groupby: str = "leiden",
        groups: List[str] = None,
        method: str = "wilcoxon",
        n_genes: int = 25,
        save_result: bool = True
    ) -> str:
        """
        Find marker genes for clusters or groups in a modality using professional differential expression analysis.
        
        Args:
            modality_name: Name of the modality to analyze
            groupby: Column name to group by (e.g., 'leiden', 'cell_type')
            groups: Specific groups to analyze (None for all)
            method: Statistical method ('wilcoxon', 't-test', 'logreg')
            n_genes: Number of top marker genes per group
            save_result: Whether to save the results
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Finding marker genes in modality '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
            # Validate groupby column exists
            if groupby not in adata.obs.columns:
                available_columns = [col for col in adata.obs.columns if col in ['leiden', 'cell_type', 'batch', 'sample']]
                return f"Group column '{groupby}' not found. Available grouping columns: {available_columns}"
            
            # Use singlecell service for marker gene detection
            adata_markers, marker_stats = singlecell_service.find_marker_genes(
                adata=adata,
                groupby=groupby,
                groups=groups,
                method=method,
                n_genes=n_genes
            )
            
            # Save as new modality
            marker_modality_name = f"{modality_name}_markers"
            data_manager.modalities[marker_modality_name] = adata_markers
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_markers.h5ad"
                data_manager.save_modality(marker_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="find_marker_genes_for_clusters",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "method": method,
                    "n_genes": n_genes
                },
                description=f"Found marker genes for {marker_stats['n_groups']} groups in {modality_name}"
            )
            
            # Format professional response
            response = f"""Successfully found marker genes for modality '{modality_name}'!

📊 **Marker Gene Analysis:**
- Grouping by: {marker_stats['groupby']}
- Number of groups: {marker_stats['n_groups']}
- Method: {marker_stats['method']}
- Top genes per group: {marker_stats['n_genes']}

📈 **Groups Analyzed:**
{', '.join(marker_stats['groups_analyzed'][:10])}{'...' if len(marker_stats['groups_analyzed']) > 10 else ''}

🧬 **Top Marker Genes by Group:**"""
            
            # Show top marker genes for each group (first 5 groups)
            if 'top_markers_per_group' in marker_stats:
                for group_id in list(marker_stats['top_markers_per_group'].keys())[:5]:
                    top_genes = marker_stats['top_markers_per_group'][group_id][:5]
                    gene_names = [gene['gene'] for gene in top_genes]
                    response += f"\n- **{group_id}**: {', '.join(gene_names)}"
                
                if len(marker_stats['top_markers_per_group']) > 5:
                    remaining = len(marker_stats['top_markers_per_group']) - 5
                    response += f"\n... and {remaining} more groups"
            
            response += f"\n\n💾 **New modality created**: '{marker_modality_name}'"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            response += f"\n📈 **Access detailed results**: adata.uns['rank_genes_groups']"
            response += f"\n\nUse this modality for cell type annotation or pathway analysis."
            
            analysis_results["details"]["marker_genes"] = response
            return response
            
        except (SingleCellError, ModalityNotFoundError) as e:
            logger.error(f"Error finding marker genes: {e}")
            return f"Error finding marker genes: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error finding marker genes: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def create_analysis_summary() -> str:
        """Create a comprehensive summary of all analysis steps performed."""
        try:
            if not analysis_results["details"]:
                return "No analyses have been performed yet. Run some analysis tools first."
            
            summary = "# Transcriptomics Analysis Summary\n\n"
            
            for step, details in analysis_results["details"].items():
                summary += f"## {step.replace('_', ' ').title()}\n"
                summary += f"{details}\n\n"
            
            # Add current modality status
            modalities = data_manager.list_modalities()
            if modalities:
                summary += f"## Current Modalities\n"
                summary += f"Loaded modalities ({len(modalities)}): {', '.join(modalities)}\n\n"
                
                # Add modality details
                summary += "### Modality Details:\n"
                for mod_name in modalities:
                    try:
                        adata = data_manager.get_modality(mod_name)
                        summary += f"- **{mod_name}**: {adata.n_obs} obs × {adata.n_vars} vars\n"
                        
                        # Add key observation columns if present
                        key_cols = [col for col in adata.obs.columns if col in ['leiden', 'cell_type', 'doublet_score', 'qc_pass']]
                        if key_cols:
                            summary += f"  - Key annotations: {', '.join(key_cols)}\n"
                    except Exception as e:
                        summary += f"- **{mod_name}**: Error accessing modality\n"
            
            analysis_results["summary"] = summary
            logger.info(f"Created analysis summary with {len(analysis_results['details'])} analysis steps")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating analysis summary: {e}")
            return f"Error creating summary: {str(e)}"

    @tool
    def annotate_cell_types(
        modality_name: str,
        reference_markers: dict = None,
        save_result: bool = True
    ) -> str:
        """
        Annotate cell types based on marker gene expression patterns.
        
        Args:
            modality_name: Name of the modality with clustering results
            reference_markers: Custom marker genes dict (None to use defaults)
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Annotating cell types in modality '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
            # Use singlecell service for cell type annotation
            adata_annotated, annotation_stats = singlecell_service.annotate_cell_types(
                adata=adata,
                reference_markers=reference_markers
            )
            
            # Save as new modality
            annotated_modality_name = f"{modality_name}_annotated"
            data_manager.modalities[annotated_modality_name] = adata_annotated
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_annotated.h5ad"
                data_manager.save_modality(annotated_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="annotate_cell_types",
                parameters={
                    "modality_name": modality_name,
                    "reference_markers": "custom" if reference_markers else "default"
                },
                description=f"Annotated {annotation_stats['n_cell_types_identified']} cell types in {modality_name}"
            )
            
            # Format professional response
            response = f"""Successfully annotated cell types in modality '{modality_name}'!

📊 **Annotation Results:**
- Cell types identified: {annotation_stats['n_cell_types_identified']}
- Clusters annotated: {annotation_stats['n_clusters']}
- Marker sets used: {annotation_stats['n_marker_sets']}

📈 **Cell Type Distribution:**"""
            
            for cell_type, count in list(annotation_stats['cell_type_counts'].items())[:8]:
                response += f"\n- {cell_type}: {count} cells"
            
            if len(annotation_stats['cell_type_counts']) > 8:
                remaining = len(annotation_stats['cell_type_counts']) - 8
                response += f"\n... and {remaining} more types"
            
            response += f"\n\n💾 **New modality created**: '{annotated_modality_name}'"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
                
            response += f"\n🔬 **Cell type annotations added to**: adata.obs['cell_type']"
            response += f"\n\nUse this modality for cell type-specific downstream analysis."
            
            analysis_results["details"]["cell_type_annotation"] = response
            return response
            
        except (SingleCellError, ModalityNotFoundError) as e:
            logger.error(f"Error in cell type annotation: {e}")
            return f"Error annotating cell types: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in cell type annotation: {e}")
            return f"Unexpected error: {str(e)}"

    # -------------------------
    # BULK RNA-SEQ ANALYSIS TOOLS
    # -------------------------
    @tool
    def run_differential_expression_analysis(
        modality_name: str,
        groupby: str,
        group1: str,
        group2: str,
        method: str = "deseq2_like",
        min_expression_threshold: float = 1.0,
        save_result: bool = True
    ) -> str:
        """
        Run differential expression analysis between two groups in a bulk RNA-seq modality.
        
        Args:
            modality_name: Name of the modality to analyze
            groupby: Column name for grouping (e.g., 'condition', 'treatment')
            group1: First group for comparison (e.g., 'control')
            group2: Second group for comparison (e.g., 'treatment') 
            method: Analysis method ('deseq2_like', 'wilcoxon', 't_test')
            min_expression_threshold: Minimum expression threshold for gene filtering
            save_result: Whether to save the results
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Running DE analysis on modality '{modality_name}': {adata.shape[0]} samples × {adata.shape[1]} genes")
            
            # Use bulk service for differential expression
            adata_de, de_stats = bulk_service.run_differential_expression_analysis(
                adata=adata,
                groupby=groupby,
                group1=group1,
                group2=group2,
                method=method,
                min_expression_threshold=min_expression_threshold
            )
            
            # Save as new modality
            de_modality_name = f"{modality_name}_de_{group1}_vs_{group2}"
            data_manager.modalities[de_modality_name] = adata_de
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_de_{group1}_vs_{group2}.h5ad"
                data_manager.save_modality(de_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="run_differential_expression_analysis",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "group1": group1,
                    "group2": group2,
                    "method": method
                },
                description=f"DE analysis: {de_stats['n_significant_genes']} significant genes found"
            )
            
            # Format professional response
            response = f"""Differential Expression Analysis Complete for '{modality_name}'!

📊 **Analysis Results:**
- Comparison: {de_stats['group1']} ({de_stats['n_samples_group1']} samples) vs {de_stats['group2']} ({de_stats['n_samples_group2']} samples)
- Method: {de_stats['method']}
- Genes tested: {de_stats['n_genes_tested']:,}
- Significant genes (padj < 0.05): {de_stats['n_significant_genes']:,}

📈 **Differential Expression Summary:**
- Upregulated in {group2}: {de_stats['n_upregulated']} genes
- Downregulated in {group2}: {de_stats['n_downregulated']} genes

🧬 **Top Upregulated Genes:**"""
            
            for gene in de_stats['top_upregulated'][:5]:
                response += f"\n- {gene}"
            
            response += f"\n\n🧬 **Top Downregulated Genes:**"
            for gene in de_stats['top_downregulated'][:5]:
                response += f"\n- {gene}"
            
            response += f"\n\n💾 **New modality created**: '{de_modality_name}'"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            response += f"\n📈 **Access detailed results**: adata.uns['{de_stats['de_results_key']}']"
            response += f"\n\nUse the significant genes for pathway enrichment analysis."
            
            analysis_results["details"]["differential_expression"] = response
            return response
            
        except (BulkRNASeqError, ModalityNotFoundError) as e:
            logger.error(f"Error in differential expression analysis: {e}")
            return f"Error running differential expression analysis: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in differential expression: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def run_pathway_enrichment_analysis(
        gene_list: List[str],
        analysis_type: str = "GO",
        modality_name: str = None,
        save_result: bool = True
    ) -> str:
        """
        Run pathway enrichment analysis on a gene list from differential expression results.
        
        Args:
            gene_list: List of genes for enrichment analysis
            analysis_type: Type of analysis ("GO" or "KEGG")
            modality_name: Optional modality name to extract genes from DE results
            save_result: Whether to save enrichment results
        """
        try:
            # If modality name provided, extract significant genes from it
            if modality_name and modality_name in data_manager.list_modalities():
                adata = data_manager.get_modality(modality_name)
                
                # Look for DE results in uns
                de_keys = [key for key in adata.uns.keys() if key.startswith('de_results_')]
                if de_keys:
                    de_results = adata.uns[de_keys[0]]  # Use first DE result
                    if isinstance(de_results, dict):
                        # Extract significant genes
                        de_df = pd.DataFrame(de_results)
                        if 'padj' in de_df.columns:
                            significant_genes = de_df[de_df['padj'] < 0.05].index.tolist()
                            if significant_genes:
                                gene_list = significant_genes[:500]  # Top 500 genes
                                logger.info(f"Extracted {len(gene_list)} significant genes from {modality_name}")
            
            if not gene_list or len(gene_list) == 0:
                return "No genes provided for enrichment analysis. Please provide a gene list or run differential expression analysis first."
            
            logger.info(f"Running pathway enrichment on {len(gene_list)} genes")
            
            # Use bulk service for pathway enrichment
            enrichment_df, enrichment_stats = bulk_service.run_pathway_enrichment(
                gene_list=gene_list,
                analysis_type=analysis_type
            )
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="run_pathway_enrichment_analysis",
                parameters={
                    "gene_list_size": len(gene_list),
                    "analysis_type": analysis_type,
                    "modality_name": modality_name
                },
                description=f"{analysis_type} enrichment: {enrichment_stats['n_significant_terms']} significant terms"
            )
            
            # Format professional response
            response = f"""{analysis_type} Pathway Enrichment Analysis Complete!

📊 **Enrichment Results:**
- Genes analyzed: {enrichment_stats['n_genes_input']:,}
- Database: {enrichment_stats['enrichment_database']}
- Terms found: {enrichment_stats['n_terms_total']}
- Significant terms (p.adj < 0.05): {enrichment_stats['n_significant_terms']}

🧬 **Top Enriched Pathways:**"""
            
            for term in enrichment_stats['top_terms'][:8]:
                response += f"\n- {term}"
            
            if len(enrichment_stats['top_terms']) > 8:
                remaining = len(enrichment_stats['top_terms']) - 8
                response += f"\n... and {remaining} more pathways"
            
            response += f"\n\nPathway enrichment reveals biological processes and pathways associated with your gene set."
            
            analysis_results["details"]["pathway_enrichment"] = response
            return response
            
        except BulkRNASeqError as e:
            logger.error(f"Error in pathway enrichment: {e}")
            return f"Error running pathway enrichment: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pathway enrichment: {e}")
            return f"Unexpected error: {str(e)}"

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        check_data_status,
        assess_data_quality,
        filter_and_normalize_modality,
        cluster_modality,
        detect_doublets_in_modality,
        find_marker_genes_for_clusters,
        annotate_cell_types,
        run_differential_expression_analysis,
        run_pathway_enrichment_analysis,
        create_analysis_summary
    ]
    
    tools = base_tools + (handoff_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are an expert bioinformatician specializing in transcriptomics analysis (single-cell and bulk RNA-seq) using the professional, modular DataManagerV2 system.

<Role>
You execute comprehensive transcriptomics analysis pipelines with proper quality control, preprocessing, clustering, and biological interpretation. You work with individual modalities in a multi-omics framework with full provenance tracking and professional-grade error handling.
</Role>

<Task>
You perform transcriptomics analysis following current best practices:
1. **Data quality assessment** with comprehensive QC metrics and validation
2. **Professional preprocessing** with filtering, normalization, and batch correction
3. **Advanced clustering** with dimensionality reduction and visualization
4. **Doublet detection** using Scrublet or statistical methods
5. **Marker gene identification** with differential expression analysis
6. **Cell type annotation** using marker gene databases
7. **Comprehensive reporting** with analysis summaries and provenance
</Task>

<Available Professional Tools>

**Core Analysis Tools:**
- `check_data_status`: Check loaded modalities and comprehensive status information
- `assess_data_quality`: Professional QC assessment with statistical summaries
- `filter_and_normalize_modality`: Advanced filtering with professional QC standards
- `create_analysis_summary`: Comprehensive analysis report with modality tracking

**Single-cell RNA-seq Tools:**
- `cluster_modality`: Leiden clustering with PCA, neighborhood graphs, and UMAP
- `detect_doublets_in_modality`: Scrublet-based doublet detection with statistical scoring
- `find_marker_genes_for_clusters`: Professional differential expression analysis for clusters
- `annotate_cell_types`: Automated cell type annotation using marker databases

**Bulk RNA-seq Tools:**
- `run_differential_expression_analysis`: DESeq2-like differential expression between groups
- `run_pathway_enrichment_analysis`: GO/KEGG pathway enrichment analysis

<Professional Modality-Aware Workflows>
The DataManagerV2 system uses **modalities** as the core data unit. Always specify modality names:

**Single-cell RNA-seq Pipeline:**
1. `check_data_status()` → Review available modalities and their characteristics
2. `assess_data_quality(modality_name="geo_gse12345")` → Professional QC assessment
3. `filter_and_normalize_modality(modality_name="geo_gse12345", ...)` → Clean and normalize data
4. `detect_doublets_in_modality(modality_name="geo_gse12345_filtered_normalized", ...)` → Remove doublets
5. `cluster_modality(modality_name="geo_gse12345_filtered_normalized", ...)` → Cluster and embed
6. `find_marker_genes_for_clusters(modality_name="geo_gse12345_clustered", ...)` → Identify markers
7. `annotate_cell_types(modality_name="geo_gse12345_markers", ...)` → Annotate cell types
8. `create_analysis_summary()` → Generate comprehensive report

**Bulk RNA-seq Pipeline:**
1. `check_data_status()` → Review available bulk RNA-seq modalities
2. `assess_data_quality(modality_name="bulk_gse12345")` → QC assessment for bulk data
3. `filter_and_normalize_modality(modality_name="bulk_gse12345", ...)` → Normalize bulk data
4. `run_differential_expression_analysis(modality_name="bulk_gse12345_filtered_normalized", groupby="condition", group1="control", group2="treatment")` → DE analysis
5. `run_pathway_enrichment_analysis(gene_list=[], analysis_type="GO")` → Pathway enrichment
6. `create_analysis_summary()` → Generate comprehensive report

**Modality Naming Convention:**
Each analysis step creates new modalities with descriptive names:
- Raw data: `geo_gse12345`
- Quality assessed: `geo_gse12345_quality_assessed`
- Filtered/normalized: `geo_gse12345_filtered_normalized`
- Doublets detected: `geo_gse12345_doublets_detected`
- Clustered: `geo_gse12345_clustered`
- With markers: `geo_gse12345_markers`
- Annotated: `geo_gse12345_annotated`

<Professional Parameter Guidelines>

**Quality Control (Single-cell RNA-seq):**
- min_genes: 200-500 (filter low-quality cells)
- max_genes_per_cell: 5000-8000 (filter potential doublets)
- min_cells_per_gene: 3-10 (remove rarely expressed genes)
- max_mt_pct: 15-25% (remove dying/stressed cells)
- max_ribo_pct: 40-60% (control for ribosomal contamination)

**Preprocessing & Normalization:**
- target_sum: 10,000 (standard CPM normalization for single-cell)
- normalization_method: 'log1p' (log(x+1) transformation, standard)
- max_genes_per_cell: 5000 (doublet filtering threshold)

**Clustering & Visualization:**
- resolution: 0.4-1.2 (start with 0.5, adjust based on biological expectations)
- batch_correction: Enable for multi-sample datasets
- demo_mode: Use for datasets >50,000 cells for faster processing

**Doublet Detection:**
- expected_doublet_rate: 0.05-0.1 (typically 6% for 10X Genomics data)
- threshold: Auto-detected or custom based on dataset characteristics

<Error Handling & Quality Assurance>
- All tools include professional error handling with specific exception types
- Comprehensive logging tracks all analysis steps with parameters
- Automatic validation ensures data integrity throughout pipeline
- Provenance tracking maintains complete analysis history
- Professional reporting with statistical summaries and visualizations

<Best Practices>
- **Always validate modality existence** before processing
- **Use descriptive modality naming** for downstream traceability
- **Save intermediate results** for reproducibility and checkpointing
- **Monitor data quality** at each processing step
- **Document analysis decisions** with parameter justifications
- **Leverage batch correction** for multi-sample integration
- **Use demo mode** for exploratory analysis of large datasets

Today's date: {date.today()}
""".strip()
    system_prompt = f"""
You are an expert bioinformatician specializing in transcriptomics analysis (single-cell and bulk RNA-seq) using the professional, modular DataManagerV2 system.

<Role>
You execute comprehensive transcriptomics analysis pipelines with proper quality control, preprocessing, clustering, and biological interpretation. You work with individual modalities in a multi-omics framework with full provenance tracking and professional-grade error handling.
</Role>

<Task>
You perform transcriptomics analysis following current best practices:
1. **Data quality assessment** with comprehensive QC metrics and validation
2. **Professional preprocessing** with filtering, normalization, and batch correction
3. **Advanced clustering** with dimensionality reduction and visualization
4. **Doublet detection** using Scrublet or statistical methods
5. **Marker gene identification** with differential expression analysis
6. **Cell type annotation** using marker gene databases
7. **Comprehensive reporting** with analysis summaries and provenance
</Task>

<Available Professional Tools>
- `check_data_status`: Check loaded modalities and comprehensive status information
- `assess_data_quality`: Professional QC assessment with statistical summaries
- `filter_and_normalize_modality`: Advanced filtering with professional QC standards
- `cluster_modality`: Leiden clustering with PCA, neighborhood graphs, and UMAP
- `detect_doublets_in_modality`: Scrublet-based doublet detection with statistical scoring
- `find_marker_genes_for_clusters`: Professional differential expression analysis
- `annotate_cell_types`: Automated cell type annotation using marker databases
- `create_analysis_summary`: Comprehensive analysis report with modality tracking

<Professional Modality-Aware Workflow>
The DataManagerV2 system uses **modalities** as the core data unit. Always specify modality names:

**Standard Analysis Pipeline:**
1. `check_data_status()` → Review available modalities and their characteristics
2. `assess_data_quality(modality_name="geo_gse12345")` → Professional QC assessment
3. `filter_and_normalize_modality(modality_name="geo_gse12345", ...)` → Clean and normalize data
4. `detect_doublets_in_modality(modality_name="geo_gse12345_filtered_normalized", ...)` → Remove doublets
5. `cluster_modality(modality_name="geo_gse12345_filtered_normalized", ...)` → Cluster and embed
6. `find_marker_genes_for_clusters(modality_name="geo_gse12345_clustered", ...)` → Identify markers
7. `annotate_cell_types(modality_name="geo_gse12345_markers", ...)` → Annotate cell types
8. `create_analysis_summary()` → Generate comprehensive report

**Modality Naming Convention:**
Each analysis step creates new modalities with descriptive names:
- Raw data: `geo_gse12345`
- Quality assessed: `geo_gse12345_quality_assessed`
- Filtered/normalized: `geo_gse12345_filtered_normalized`
- Doublets detected: `geo_gse12345_doublets_detected`
- Clustered: `geo_gse12345_clustered`
- With markers: `geo_gse12345_markers`
- Annotated: `geo_gse12345_annotated`

<Professional Parameter Guidelines>

**Quality Control (Single-cell RNA-seq):**
- min_genes: 200-500 (filter low-quality cells)
- max_genes_per_cell: 5000-8000 (filter potential doublets)
- min_cells_per_gene: 3-10 (remove rarely expressed genes)
- max_mt_pct: 15-25% (remove dying/stressed cells)
- max_ribo_pct: 40-60% (control for ribosomal contamination)

**Preprocessing & Normalization:**
- target_sum: 10,000 (standard CPM normalization for single-cell)
- normalization_method: 'log1p' (log(x+1) transformation, standard)
- max_genes_per_cell: 5000 (doublet filtering threshold)

**Clustering & Visualization:**
- resolution: 0.4-1.2 (start with 0.5, adjust based on biological expectations)
- batch_correction: Enable for multi-sample datasets
- demo_mode: Use for datasets >50,000 cells for faster processing

**Doublet Detection:**
- expected_doublet_rate: 0.05-0.1 (typically 6% for 10X Genomics data)
- threshold: Auto-detected or custom based on dataset characteristics

<Error Handling & Quality Assurance>
- All tools include professional error handling with specific exception types
- Comprehensive logging tracks all analysis steps with parameters
- Automatic validation ensures data integrity throughout pipeline
- Provenance tracking maintains complete analysis history
- Professional reporting with statistical summaries and visualizations

<Best Practices>
- **Always validate modality existence** before processing
- **Use descriptive modality naming** for downstream traceability
- **Save intermediate results** for reproducibility and checkpointing
- **Monitor data quality** at each processing step
- **Document analysis decisions** with parameter justifications
- **Leverage batch correction** for multi-sample integration
- **Use demo mode** for exploratory analysis of large datasets

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=TranscriptomicsExpertState
    )
