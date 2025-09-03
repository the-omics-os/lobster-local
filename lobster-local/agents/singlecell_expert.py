"""
Single-cell RNA-seq Expert Agent for specialized single-cell analysis.

This agent focuses exclusively on single-cell RNA-seq analysis using the modular DataManagerV2 
system with proper modality handling and schema enforcement.
"""

from typing import List, Optional, Union
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

from datetime import date

import pandas as pd

from lobster.agents.state import SingleCellExpertState
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.preprocessing_service import PreprocessingService, PreprocessingError
from lobster.tools.quality_service import QualityService, QualityError
from lobster.tools.clustering_service import ClusteringService, ClusteringError
from lobster.tools.enhanced_singlecell_service import EnhancedSingleCellService, SingleCellError as ServiceSingleCellError
from lobster.tools.visualization_service import SingleCellVisualizationService, VisualizationError
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class SingleCellAgentError(Exception):
    """Base exception for single-cell agent operations."""
    pass


class ModalityNotFoundError(SingleCellAgentError):
    """Raised when requested modality doesn't exist."""
    pass


def singlecell_expert(
    data_manager: DataManagerV2, 
    callback_handler=None, 
    agent_name: str = "singlecell_expert_agent",
    handoff_tools: List = None
):  
    """Create single-cell expert agent using DataManagerV2 and modular services."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('singlecell_expert')
    llm = ChatBedrockConverse(**model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Initialize stateless services for single-cell analysis
    preprocessing_service = PreprocessingService()
    quality_service = QualityService()
    clustering_service = ClusteringService()
    singlecell_service = EnhancedSingleCellService()
    visualization_service = SingleCellVisualizationService()
    
    analysis_results = {"summary": "", "details": {}}
    
    # -------------------------
    # DATA STATUS TOOLS
    # -------------------------
    @tool
    def check_data_status(modality_name: str = "") -> str:
        """Check if single-cell data is loaded and ready for analysis."""
        try:
            if modality_name == "":
                modalities = data_manager.list_modalities()
                if not modalities:
                    return "No modalities loaded. Please ask the data expert to load a single-cell dataset first."
                
                # Filter for likely single-cell modalities
                sc_modalities = [mod for mod in modalities if 
                               'single_cell' in mod.lower() or 'sc' in mod.lower() or
                               data_manager._detect_modality_type(mod) == 'single_cell_rna_seq']
                
                if not sc_modalities:
                    response = f"Available modalities ({len(modalities)}) but none appear to be single-cell:\n"
                    for mod_name in modalities:
                        adata = data_manager.get_modality(mod_name)
                        response += f"- **{mod_name}**: {adata.n_obs} obs × {adata.n_vars} vars\n"
                    response += "\nPlease specify a modality name if it contains single-cell data."
                else:
                    response = f"Single-cell modalities found ({len(sc_modalities)}):\n"
                    for mod_name in sc_modalities:
                        adata = data_manager.get_modality(mod_name)
                        response += f"- **{mod_name}**: {adata.n_obs} cells × {adata.n_vars} genes\n"
                
                return response
            
            else:
                # Check specific modality
                if modality_name not in data_manager.list_modalities():
                    return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                
                adata = data_manager.get_modality(modality_name)
                metrics = data_manager.get_quality_metrics(modality_name)
                
                response = f"Single-cell modality '{modality_name}' ready for analysis:\n"
                response += f"- Shape: {adata.n_obs} cells × {adata.n_vars} genes\n"
                response += f"- Obs columns: {list(adata.obs.columns)[:5]}...\n"
                response += f"- Var columns: {list(adata.var.columns)[:5]}...\n"
                
                if 'total_counts' in metrics:
                    response += f"- Total counts: {metrics['total_counts']:,.0f}\n"
                if 'mean_counts_per_obs' in metrics:
                    response += f"- Mean counts/cell: {metrics['mean_counts_per_obs']:.1f}\n"
                
                # Add single-cell specific checks
                if adata.n_obs > 1000:
                    response += f"- Dataset size: Large ({adata.n_obs:,} cells) - suitable for clustering\n"
                elif adata.n_obs > 100:
                    response += f"- Dataset size: Medium ({adata.n_obs:,} cells) - good for analysis\n"
                else:
                    response += f"- Dataset size: Small ({adata.n_obs:,} cells) - may limit analysis\n"
                
                analysis_results["details"]["data_status"] = response
                return response
                
        except Exception as e:
            logger.error(f"Error checking single-cell data status: {e}")
            return f"Error checking single-cell data status: {str(e)}"

    @tool
    def assess_data_quality(
        modality_name: str,
        min_genes: int = 200,
        max_mt_pct: float = 20.0,
        max_ribo_pct: float = 50.0,
        min_housekeeping_score: float = 1.0
    ) -> str:
        """Run comprehensive quality control assessment on single-cell RNA-seq data."""
        try:
            if modality_name == "":
                return "Please specify modality_name for single-cell quality assessment. Use check_data_status() to see available modalities."
            
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Run quality assessment using service with single-cell specific parameters
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
                description=f"Single-cell quality assessment for {modality_name}"
            )
            
            # Format professional response with single-cell context
            response = f"""Single-cell Quality Assessment Complete for '{modality_name}'!

📊 **Assessment Results:**
- Cells analyzed: {assessment_stats['cells_before_qc']:,}
- Cells passing QC: {assessment_stats['cells_after_qc']:,} ({assessment_stats['cells_retained_pct']:.1f}%)
- Quality status: {assessment_stats['quality_status']}

📈 **Single-cell Quality Metrics:**
- Mean genes per cell: {assessment_stats['mean_genes_per_cell']:.0f}
- Mean mitochondrial %: {assessment_stats['mean_mt_pct']:.1f}%
- Mean ribosomal %: {assessment_stats['mean_ribo_pct']:.1f}%
- Mean UMI counts: {assessment_stats['mean_total_counts']:.0f}

💡 **Single-cell QC Summary:**
{assessment_stats['qc_summary']}

💾 **New modality created**: '{qc_modality_name}' (with single-cell QC annotations)

Proceed with filtering and normalization, then doublet detection before clustering."""
            
            analysis_results["details"]["quality_assessment"] = response
            return response
            
        except QualityError as e:
            logger.error(f"Single-cell quality assessment error: {e}")
            return f"Single-cell quality assessment failed: {str(e)}"
        except Exception as e:
            logger.error(f"Error in single-cell quality assessment: {e}")
            return f"Error in single-cell quality assessment: {str(e)}"

    # -------------------------
    # SINGLE-CELL PREPROCESSING TOOLS
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
        Filter and normalize single-cell RNA-seq data using professional QC standards.
        
        Args:
            modality_name: Name of the modality to process
            min_genes_per_cell: Minimum number of genes expressed per cell (single-cell specific)
            max_genes_per_cell: Maximum number of genes expressed per cell (doublet filtering)
            min_cells_per_gene: Minimum number of cells expressing each gene
            max_mito_percent: Maximum mitochondrial gene percentage (single-cell specific)
            normalization_method: Normalization method ('log1p' recommended for single-cell)
            target_sum: Target sum for normalization (10,000 standard for single-cell)
            save_result: Whether to save the filtered modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Processing single-cell modality '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
            # Use preprocessing service with single-cell optimized parameters
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
                description=f"Single-cell filtered and normalized {modality_name}"
            )
            
            # Format professional response
            original_shape = processing_stats['original_shape']
            final_shape = processing_stats['final_shape']
            cells_retained_pct = processing_stats['cells_retained_pct']
            genes_retained_pct = processing_stats['genes_retained_pct']
            
            response = f"""Successfully filtered and normalized single-cell modality '{modality_name}'!

📊 **Single-cell Filtering Results:**
- Original: {original_shape[0]:,} cells × {original_shape[1]:,} genes
- Filtered: {final_shape[0]:,} cells × {final_shape[1]:,} genes  
- Cells retained: {cells_retained_pct:.1f}%
- Genes retained: {genes_retained_pct:.1f}%

🔬 **Single-cell Processing Parameters:**
- Min genes/cell: {min_genes_per_cell} (removes low-quality cells)
- Max genes/cell: {max_genes_per_cell} (filters potential doublets)
- Min cells/gene: {min_cells_per_gene} (removes rarely expressed genes)
- Max mitochondrial %: {max_mito_percent}% (removes dying cells)
- Normalization: {normalization_method} (target_sum={target_sum:,} UMIs/cell)

💾 **New modality created**: '{filtered_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            response += f"\n\nNext recommended steps: doublet detection, then clustering and cell type annotation."
            
            analysis_results["details"]["filter_normalize"] = response
            return response
            
        except (PreprocessingError, ModalityNotFoundError) as e:
            logger.error(f"Error in single-cell filtering/normalization: {e}")
            return f"Error filtering and normalizing single-cell modality: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in single-cell filtering/normalization: {e}")
            return f"Unexpected error: {str(e)}"

    # -------------------------
    # SINGLE-CELL SPECIFIC ANALYSIS TOOLS
    # -------------------------
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
            modality_name: Name of the single-cell modality to process
            expected_doublet_rate: Expected doublet rate (typically 0.05-0.1 for 10X data)
            threshold: Custom threshold for doublet calling (None for automatic)
            save_result: Whether to save results
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Detecting doublets in single-cell modality '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
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
                description=f"Detected {detection_stats['n_doublets_detected']} doublets in single-cell data {modality_name}"
            )
            
            # Format professional response
            response = f"""Successfully detected doublets in single-cell modality '{modality_name}'!

📊 **Single-cell Doublet Detection Results:**
- Method: {detection_stats['detection_method']}
- Cells analyzed: {detection_stats['n_cells_analyzed']:,}
- Doublets detected: {detection_stats['n_doublets_detected']} ({detection_stats['actual_doublet_rate']:.1%})
- Expected rate: {detection_stats['expected_doublet_rate']:.1%}

📈 **Doublet Score Statistics:**
- Min score: {detection_stats['doublet_score_stats']['min']:.3f}
- Max score: {detection_stats['doublet_score_stats']['max']:.3f}
- Mean score: {detection_stats['doublet_score_stats']['mean']:.3f}

🔬 **Added to single-cell observations:**
- 'doublet_score': Doublet likelihood score
- 'predicted_doublet': Boolean doublet classification

💾 **New modality created**: '{doublet_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            response += f"\n\nFilter doublets before clustering, or proceed with clustering and filter later based on results."
            
            analysis_results["details"]["doublet_detection"] = response
            return response
            
        except (ServiceSingleCellError, ModalityNotFoundError) as e:
            logger.error(f"Error in single-cell doublet detection: {e}")
            return f"Error detecting doublets in single-cell data: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in single-cell doublet detection: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def cluster_modality(
        modality_name: str,
        resolution: float = 0.5,
        batch_correction: bool = True,
        batch_key: str = None,
        demo_mode: bool = False,
        save_result: bool = True
    ) -> str:
        """
        Perform single-cell clustering and UMAP visualization.
        
        Args:
            modality_name: Name of the single-cell modality to cluster
            resolution: Leiden clustering resolution (0.1-2.0, higher = more clusters)
            batch_correction: Whether to perform batch correction for multi-sample data
            batch_key: Column name for batch information (auto-detected if None)
            demo_mode: Use faster processing for large single-cell datasets (>50k cells)
            save_result: Whether to save the clustered modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Clustering single-cell modality '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
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
                description=f"Single-cell clustered {modality_name} into {clustering_stats['n_clusters']} clusters"
            )
            
            # Format professional response
            response = f"""Successfully clustered single-cell modality '{modality_name}'!

📊 **Single-cell Clustering Results:**
- Number of clusters: {clustering_stats['n_clusters']}
- Leiden resolution: {clustering_stats['resolution']}
- UMAP coordinates: {'✓' if clustering_stats['has_umap'] else '✗'}
- Marker genes: {'✓' if clustering_stats['has_marker_genes'] else '✗'}

🔬 **Processing Details:**
- Original shape: {clustering_stats['original_shape'][0]} × {clustering_stats['original_shape'][1]}
- Final shape: {clustering_stats['final_shape'][0]} × {clustering_stats['final_shape'][1]}
- Batch correction: {'✓' if clustering_stats['batch_correction'] else '✗'}
- Demo mode: {'✓' if clustering_stats['demo_mode'] else '✗'}

📈 **Single-cell Cluster Distribution:**"""
            
            # Add cluster size information
            for cluster_id, size in list(clustering_stats['cluster_sizes'].items())[:8]:
                percentage = (size / clustering_stats['final_shape'][0]) * 100
                response += f"\n- Cluster {cluster_id}: {size} cells ({percentage:.1f}%)"
            
            if len(clustering_stats['cluster_sizes']) > 8:
                response += f"\n... and {len(clustering_stats['cluster_sizes']) - 8} more clusters"
            
            response += f"\n\n💾 **New modality created**: '{clustered_modality_name}'"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
                
            response += f"\n\nNext steps: find marker genes for clusters and annotate cell types."
            
            analysis_results["details"]["clustering"] = response
            return response
            
        except (ClusteringError, ModalityNotFoundError) as e:
            logger.error(f"Error in single-cell clustering: {e}")
            return f"Error clustering single-cell modality: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in single-cell clustering: {e}")
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
        Find marker genes for single-cell clusters using differential expression analysis.
        
        Args:
            modality_name: Name of the single-cell modality to analyze
            groupby: Column name to group by (e.g., 'leiden', 'cell_type')
            groups: Specific clusters to analyze (None for all)
            method: Statistical method ('wilcoxon', 't-test', 'logreg')
            n_genes: Number of top marker genes per cluster
            save_result: Whether to save the results
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Finding marker genes in single-cell modality '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
            # Validate groupby column exists
            if groupby not in adata.obs.columns:
                available_columns = [col for col in adata.obs.columns if col in ['leiden', 'cell_type', 'louvain', 'cluster']]
                return f"Cluster column '{groupby}' not found. Available clustering columns: {available_columns}"
            
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
                description=f"Found marker genes for {marker_stats['n_groups']} single-cell clusters in {modality_name}"
            )
            
            # Format professional response
            response = f"""Successfully found marker genes for single-cell clusters in '{modality_name}'!

📊 **Single-cell Marker Gene Analysis:**
- Grouping by: {marker_stats['groupby']}
- Number of clusters: {marker_stats['n_groups']}
- Method: {marker_stats['method']}
- Top genes per cluster: {marker_stats['n_genes']}

📈 **Clusters Analyzed:**
{', '.join(marker_stats['groups_analyzed'][:10])}{'...' if len(marker_stats['groups_analyzed']) > 10 else ''}

🧬 **Top Marker Genes by Single-cell Cluster:**"""
            
            # Show top marker genes for each cluster (first 5 clusters)
            if 'top_markers_per_group' in marker_stats:
                for cluster_id in list(marker_stats['top_markers_per_group'].keys())[:5]:
                    top_genes = marker_stats['top_markers_per_group'][cluster_id][:5]
                    gene_names = [gene['gene'] for gene in top_genes]
                    response += f"\n- **Cluster {cluster_id}**: {', '.join(gene_names)}"
                
                if len(marker_stats['top_markers_per_group']) > 5:
                    remaining = len(marker_stats['top_markers_per_group']) - 5
                    response += f"\n... and {remaining} more clusters"
            
            response += f"\n\n💾 **New modality created**: '{marker_modality_name}'"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            response += f"\n📈 **Access detailed results**: adata.uns['rank_genes_groups']"
            response += f"\n\nNext step: use marker genes to annotate cell types in each cluster."
            
            analysis_results["details"]["marker_genes"] = response
            return response
            
        except (ServiceSingleCellError, ModalityNotFoundError) as e:
            logger.error(f"Error finding single-cell marker genes: {e}")
            return f"Error finding marker genes for single-cell clusters: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error finding single-cell marker genes: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def annotate_cell_types(
        modality_name: str,
        reference_markers: dict = None,
        save_result: bool = True
    ) -> str:
        """
        Annotate single-cell clusters with cell types based on marker gene expression patterns.
        
        Args:
            modality_name: Name of the single-cell modality with clustering results
            reference_markers: Custom marker genes dict (None to use defaults)
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Annotating cell types in single-cell modality '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
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
                description=f"Annotated {annotation_stats['n_cell_types_identified']} cell types in single-cell data {modality_name}"
            )
            
            # Format professional response
            response = f"""Successfully annotated cell types in single-cell modality '{modality_name}'!

📊 **Single-cell Annotation Results:**
- Cell types identified: {annotation_stats['n_cell_types_identified']}
- Clusters annotated: {annotation_stats['n_clusters']}
- Marker sets used: {annotation_stats['n_marker_sets']}

📈 **Single-cell Type Distribution:**"""
            
            for cell_type, count in list(annotation_stats['cell_type_counts'].items())[:8]:
                response += f"\n- {cell_type}: {count} cells"
            
            if len(annotation_stats['cell_type_counts']) > 8:
                remaining = len(annotation_stats['cell_type_counts']) - 8
                response += f"\n... and {remaining} more types"
            
            response += f"\n\n💾 **New modality created**: '{annotated_modality_name}'"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
                
            response += f"\n🔬 **Cell type annotations added to**: adata.obs['cell_type']"
            response += f"\n\nProceed with cell type-specific downstream analysis or comparative studies."
            
            analysis_results["details"]["cell_type_annotation"] = response
            return response
            
        except (ServiceSingleCellError, ModalityNotFoundError) as e:
            logger.error(f"Error in single-cell cell type annotation: {e}")
            return f"Error annotating cell types in single-cell data: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in single-cell cell type annotation: {e}")
            return f"Unexpected error: {str(e)}"

    # -------------------------
    # VISUALIZATION TOOLS
    # -------------------------
    @tool
    def create_umap_plot(
        modality_name: str,
        color_by: str = "leiden",
        point_size: Optional[int] = None,
        title: Optional[str] = None,
        save_plot: bool = True
    ) -> str:
        """
        Create an interactive UMAP plot for single-cell data.
        
        Args:
            modality_name: Name of the modality with UMAP coordinates
            color_by: Column to color by (e.g., 'leiden', 'cell_type', or gene name)
            point_size: Size of points (auto-scaled if None)
            title: Plot title
            save_plot: Whether to save the plot
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Create UMAP plot
            fig = visualization_service.create_umap_plot(
                adata=adata,
                color_by=color_by,
                point_size=point_size,
                title=title
            )
            
            # Add to data manager plots
            plot_id = data_manager.add_plot(
                plot=fig,
                title=f"UMAP - {color_by}",
                source="singlecell_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "n_cells": adata.n_obs,
                    "color_by": color_by
                }
            )
            
            # Save plot if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_umap_plot",
                parameters={
                    "modality_name": modality_name,
                    "color_by": color_by
                },
                description=f"Created UMAP plot colored by {color_by}"
            )
            
            response = f"""Successfully created UMAP plot for '{modality_name}'!

📊 **Plot Details:**
- Colored by: {color_by}
- Number of cells: {adata.n_obs:,}
- Plot ID: {plot_id}
- Interactive plot added to workspace

💾 **Saved files:** {len(saved_files)} files"""
            
            if saved_files:
                response += f"\n- Files: {', '.join([f.split('/')[-1] for f in saved_files[:3]])}"
            
            response += f"\n\nThe UMAP plot shows the distribution of cells in 2D space, colored by {color_by}."
            
            return response
            
        except (VisualizationError, ModalityNotFoundError) as e:
            logger.error(f"Error creating UMAP plot: {e}")
            return f"Error creating UMAP plot: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in UMAP plot creation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def create_qc_plots(
        modality_name: str,
        title: Optional[str] = None,
        save_plot: bool = True
    ) -> str:
        """
        Create comprehensive QC plots for single-cell data.
        
        Args:
            modality_name: Name of the modality to visualize
            title: Overall title for the QC plots
            save_plot: Whether to save the plots
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Create QC plots
            fig = visualization_service.create_qc_plots(
                adata=adata,
                title=title
            )
            
            # Add to data manager plots
            plot_id = data_manager.add_plot(
                plot=fig,
                title="QC Plots",
                source="singlecell_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "n_cells": adata.n_obs,
                    "plot_type": "qc_multi_panel"
                }
            )
            
            # Save plot if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_qc_plots",
                parameters={
                    "modality_name": modality_name
                },
                description="Created comprehensive QC plots"
            )
            
            response = f"""Successfully created QC plots for '{modality_name}'!

📊 **QC Plots Created:**
- nGenes vs nUMIs scatter plot
- Mitochondrial % vs nUMIs
- nGenes distribution histogram
- nUMIs distribution histogram
- Mitochondrial % distribution
- Cells per sample (if batch info available)

📈 **Data Overview:**
- Total cells: {adata.n_obs:,}
- Total genes: {adata.n_vars:,}
- Plot ID: {plot_id}

💾 **Saved files:** {len(saved_files)} files"""
            
            if saved_files:
                response += f"\n- Files: {', '.join([f.split('/')[-1] for f in saved_files[:3]])}"
            
            response += "\n\nUse these plots to identify QC thresholds for filtering."
            
            return response
            
        except (VisualizationError, ModalityNotFoundError) as e:
            logger.error(f"Error creating QC plots: {e}")
            return f"Error creating QC plots: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in QC plot creation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def create_violin_plot(
        modality_name: str,
        genes: Union[str, List[str]],
        groupby: str = "leiden",
        use_raw: bool = True,
        log_scale: bool = False,
        save_plot: bool = True
    ) -> str:
        """
        Create violin plots for gene expression across groups.
        
        Args:
            modality_name: Name of the modality
            genes: Gene or list of genes to plot
            groupby: Column to group by (e.g., 'leiden', 'cell_type')
            use_raw: Whether to use raw data
            log_scale: Whether to use log scale
            save_plot: Whether to save the plot
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Ensure genes is a list
            if isinstance(genes, str):
                genes = [genes]
            
            # Create violin plot
            fig = visualization_service.create_violin_plot(
                adata=adata,
                genes=genes,
                groupby=groupby,
                use_raw=use_raw,
                log_scale=log_scale
            )
            
            # Add to data manager plots
            plot_id = data_manager.add_plot(
                plot=fig,
                title=f"Violin Plot - {', '.join(genes)}",
                source="singlecell_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "genes": genes,
                    "groupby": groupby
                }
            )
            
            # Save plot if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_violin_plot",
                parameters={
                    "modality_name": modality_name,
                    "genes": genes,
                    "groupby": groupby,
                    "use_raw": use_raw,
                    "log_scale": log_scale
                },
                description=f"Created violin plot for {len(genes)} genes"
            )
            
            response = f"""Successfully created violin plot for '{modality_name}'!

📊 **Plot Details:**
- Genes plotted: {', '.join(genes)}
- Grouped by: {groupby}
- Using {'raw' if use_raw else 'normalized'} data
- Log scale: {'Yes' if log_scale else 'No'}
- Plot ID: {plot_id}

💾 **Saved files:** {len(saved_files)} files"""
            
            if saved_files:
                response += f"\n- Files: {', '.join([f.split('/')[-1] for f in saved_files[:3]])}"
            
            response += f"\n\nViolin plots show the distribution of gene expression across {groupby} groups."
            
            return response
            
        except (VisualizationError, ModalityNotFoundError) as e:
            logger.error(f"Error creating violin plot: {e}")
            return f"Error creating violin plot: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in violin plot creation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def create_feature_plot(
        modality_name: str,
        genes: Union[str, List[str]],
        use_raw: bool = True,
        ncols: int = 2,
        point_size: Optional[int] = None,
        save_plot: bool = True
    ) -> str:
        """
        Create feature plots showing gene expression on UMAP.
        
        Args:
            modality_name: Name of the modality with UMAP coordinates
            genes: Gene or list of genes to plot
            use_raw: Whether to use raw data
            ncols: Number of columns in subplot grid
            point_size: Size of points
            save_plot: Whether to save the plot
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Ensure genes is a list
            if isinstance(genes, str):
                genes = [genes]
            
            # Create feature plot
            fig = visualization_service.create_feature_plot(
                adata=adata,
                genes=genes,
                use_raw=use_raw,
                ncols=ncols,
                point_size=point_size
            )
            
            # Add to data manager plots
            plot_id = data_manager.add_plot(
                plot=fig,
                title=f"Feature Plot - {', '.join(genes[:3])}{'...' if len(genes) > 3 else ''}",
                source="singlecell_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "genes": genes,
                    "n_genes": len(genes)
                }
            )
            
            # Save plot if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_feature_plot",
                parameters={
                    "modality_name": modality_name,
                    "genes": genes,
                    "use_raw": use_raw,
                    "ncols": ncols
                },
                description=f"Created feature plot for {len(genes)} genes"
            )
            
            response = f"""Successfully created feature plot for '{modality_name}'!

📊 **Plot Details:**
- Genes plotted: {', '.join(genes[:5])}{'...' if len(genes) > 5 else ''}
- Total genes: {len(genes)}
- Using {'raw' if use_raw else 'normalized'} data
- Grid layout: {ncols} columns
- Plot ID: {plot_id}

💾 **Saved files:** {len(saved_files)} files"""
            
            if saved_files:
                response += f"\n- Files: {', '.join([f.split('/')[-1] for f in saved_files[:3]])}"
            
            response += "\n\nFeature plots show gene expression levels projected onto UMAP coordinates."
            
            return response
            
        except (VisualizationError, ModalityNotFoundError) as e:
            logger.error(f"Error creating feature plot: {e}")
            return f"Error creating feature plot: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in feature plot creation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def create_dot_plot(
        modality_name: str,
        genes: List[str],
        groupby: str = "leiden",
        use_raw: bool = True,
        standard_scale: str = "var",
        save_plot: bool = True
    ) -> str:
        """
        Create a dot plot for marker gene expression.
        
        Args:
            modality_name: Name of the modality
            genes: List of genes to plot
            groupby: Column to group by
            use_raw: Whether to use raw data
            standard_scale: How to scale ('var', 'group', or None)
            save_plot: Whether to save the plot
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Create dot plot
            fig = visualization_service.create_dot_plot(
                adata=adata,
                genes=genes,
                groupby=groupby,
                use_raw=use_raw,
                standard_scale=standard_scale
            )
            
            # Add to data manager plots
            plot_id = data_manager.add_plot(
                plot=fig,
                title=f"Dot Plot - {groupby}",
                source="singlecell_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "n_genes": len(genes),
                    "groupby": groupby
                }
            )
            
            # Save plot if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_dot_plot",
                parameters={
                    "modality_name": modality_name,
                    "n_genes": len(genes),
                    "groupby": groupby,
                    "standard_scale": standard_scale
                },
                description=f"Created dot plot for {len(genes)} genes"
            )
            
            response = f"""Successfully created dot plot for '{modality_name}'!

📊 **Plot Details:**
- Genes plotted: {len(genes)}
- Grouped by: {groupby}
- Using {'raw' if use_raw else 'normalized'} data
- Scaling: {standard_scale if standard_scale else 'None'}
- Plot ID: {plot_id}

📈 **Dot Plot Legend:**
- Dot size: Percentage of cells expressing the gene
- Dot color: Mean expression level

💾 **Saved files:** {len(saved_files)} files"""
            
            if saved_files:
                response += f"\n- Files: {', '.join([f.split('/')[-1] for f in saved_files[:3]])}"
            
            return response
            
        except (VisualizationError, ModalityNotFoundError) as e:
            logger.error(f"Error creating dot plot: {e}")
            return f"Error creating dot plot: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in dot plot creation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def create_heatmap(
        modality_name: str,
        genes: Optional[List[str]] = None,
        groupby: str = "leiden",
        use_raw: bool = True,
        n_top_genes: int = 5,
        standard_scale: bool = True,
        save_plot: bool = True
    ) -> str:
        """
        Create a heatmap of gene expression.
        
        Args:
            modality_name: Name of the modality
            genes: List of genes (if None, use top marker genes)
            groupby: Column to group by
            use_raw: Whether to use raw data
            n_top_genes: Number of top genes per group if genes not specified
            standard_scale: Whether to z-score normalize
            save_plot: Whether to save the plot
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Create heatmap
            fig = visualization_service.create_heatmap(
                adata=adata,
                genes=genes,
                groupby=groupby,
                use_raw=use_raw,
                n_top_genes=n_top_genes,
                standard_scale=standard_scale
            )
            
            # Add to data manager plots
            plot_id = data_manager.add_plot(
                plot=fig,
                title=f"Heatmap - {groupby}",
                source="singlecell_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "n_genes": len(genes) if genes else "auto"
                }
            )
            
            # Save plot if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_heatmap",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "n_genes": len(genes) if genes else f"top {n_top_genes} per group",
                    "standard_scale": standard_scale
                },
                description="Created gene expression heatmap"
            )
            
            response = f"""Successfully created heatmap for '{modality_name}'!

📊 **Plot Details:**
- Grouped by: {groupby}
- Genes: {len(genes) if genes else f'Top {n_top_genes} marker genes per group'}
- Using {'raw' if use_raw else 'normalized'} data
- Z-score normalized: {'Yes' if standard_scale else 'No'}
- Plot ID: {plot_id}

💾 **Saved files:** {len(saved_files)} files"""
            
            if saved_files:
                response += f"\n- Files: {', '.join([f.split('/')[-1] for f in saved_files[:3]])}"
            
            response += "\n\nHeatmap shows mean expression levels across groups."
            
            return response
            
        except (VisualizationError, ModalityNotFoundError) as e:
            logger.error(f"Error creating heatmap: {e}")
            return f"Error creating heatmap: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in heatmap creation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def create_elbow_plot(
        modality_name: str,
        n_pcs: int = 50,
        save_plot: bool = True
    ) -> str:
        """
        Create an elbow plot for PCA variance explained.
        
        Args:
            modality_name: Name of the modality with PCA results
            n_pcs: Number of PCs to show
            save_plot: Whether to save the plot
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Create elbow plot
            fig = visualization_service.create_elbow_plot(
                adata=adata,
                n_pcs=n_pcs
            )
            
            # Add to data manager plots
            plot_id = data_manager.add_plot(
                plot=fig,
                title="PCA Elbow Plot",
                source="singlecell_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "n_pcs": n_pcs
                }
            )
            
            # Save plot if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_elbow_plot",
                parameters={
                    "modality_name": modality_name,
                    "n_pcs": n_pcs
                },
                description="Created PCA elbow plot"
            )
            
            response = f"""Successfully created elbow plot for '{modality_name}'!

📊 **Plot Details:**
- PCs shown: {n_pcs}
- Shows individual and cumulative variance explained
- Plot ID: {plot_id}

💡 **How to interpret:**
- Look for the "elbow" where variance explained plateaus
- This indicates the optimal number of PCs to use for clustering

💾 **Saved files:** {len(saved_files)} files"""
            
            if saved_files:
                response += f"\n- Files: {', '.join([f.split('/')[-1] for f in saved_files[:3]])}"
            
            return response
            
        except (VisualizationError, ModalityNotFoundError) as e:
            logger.error(f"Error creating elbow plot: {e}")
            return f"Error creating elbow plot: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in elbow plot creation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def create_cluster_composition_plot(
        modality_name: str,
        cluster_col: str = "leiden",
        sample_col: Optional[str] = None,
        normalize: bool = True,
        save_plot: bool = True
    ) -> str:
        """
        Create a stacked bar plot showing cluster composition.
        
        Args:
            modality_name: Name of the modality
            cluster_col: Column with cluster assignments
            sample_col: Column with sample/batch info (auto-detect if None)
            normalize: Whether to normalize to percentages
            save_plot: Whether to save the plot
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Create composition plot
            fig = visualization_service.create_cluster_composition_plot(
                adata=adata,
                cluster_col=cluster_col,
                sample_col=sample_col,
                normalize=normalize
            )
            
            # Add to data manager plots
            plot_id = data_manager.add_plot(
                plot=fig,
                title=f"Cluster Composition - {cluster_col}",
                source="singlecell_expert",
                dataset_info={
                    "modality_name": modality_name,
                    "cluster_col": cluster_col,
                    "sample_col": sample_col
                }
            )
            
            # Save plot if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_cluster_composition_plot",
                parameters={
                    "modality_name": modality_name,
                    "cluster_col": cluster_col,
                    "sample_col": sample_col,
                    "normalize": normalize
                },
                description="Created cluster composition plot"
            )
            
            response = f"""Successfully created cluster composition plot for '{modality_name}'!

📊 **Plot Details:**
- Cluster column: {cluster_col}
- Sample column: {sample_col if sample_col else 'Auto-detected or none'}
- Normalized: {'Yes (percentages)' if normalize else 'No (counts)'}
- Plot ID: {plot_id}

💾 **Saved files:** {len(saved_files)} files"""
            
            if saved_files:
                response += f"\n- Files: {', '.join([f.split('/')[-1] for f in saved_files[:3]])}"
            
            response += "\n\nComposition plot shows cluster distribution across samples/batches."
            
            return response
            
        except (VisualizationError, ModalityNotFoundError) as e:
            logger.error(f"Error creating composition plot: {e}")
            return f"Error creating cluster composition plot: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in composition plot creation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def create_analysis_summary() -> str:
        """Create a comprehensive summary of all single-cell analysis steps performed."""
        try:
            if not analysis_results["details"]:
                return "No single-cell analyses have been performed yet. Run some analysis tools first."
            
            summary = "# Single-cell RNA-seq Analysis Summary\n\n"
            
            for step, details in analysis_results["details"].items():
                summary += f"## {step.replace('_', ' ').title()}\n"
                summary += f"{details}\n\n"
            
            # Add current modality status
            modalities = data_manager.list_modalities()
            if modalities:
                # Filter for single-cell modalities
                sc_modalities = [mod for mod in modalities if 
                               'single_cell' in mod.lower() or 'sc' in mod.lower() or
                               data_manager._detect_modality_type(mod) == 'single_cell_rna_seq']
                
                summary += f"## Current Single-cell Modalities\n"
                summary += f"Single-cell modalities ({len(sc_modalities)}): {', '.join(sc_modalities)}\n\n"
                
                # Add modality details
                summary += "### Single-cell Modality Details:\n"
                for mod_name in sc_modalities:
                    try:
                        adata = data_manager.get_modality(mod_name)
                        summary += f"- **{mod_name}**: {adata.n_obs} cells × {adata.n_vars} genes\n"
                        
                        # Add key single-cell observation columns if present
                        key_cols = [col for col in adata.obs.columns if col in ['leiden', 'cell_type', 'doublet_score', 'qc_pass', 'louvain']]
                        if key_cols:
                            summary += f"  - Single-cell annotations: {', '.join(key_cols)}\n"
                    except Exception as e:
                        summary += f"- **{mod_name}**: Error accessing modality\n"
            
            analysis_results["summary"] = summary
            logger.info(f"Created single-cell analysis summary with {len(analysis_results['details'])} analysis steps")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating single-cell analysis summary: {e}")
            return f"Error creating single-cell summary: {str(e)}"

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        check_data_status,
        assess_data_quality,
        filter_and_normalize_modality,
        detect_doublets_in_modality,
        cluster_modality,
        find_marker_genes_for_clusters,
        annotate_cell_types,
        # Visualization tools
        create_umap_plot,
        create_qc_plots,
        create_violin_plot,
        create_feature_plot,
        create_dot_plot,
        create_heatmap,
        create_elbow_plot,
        create_cluster_composition_plot,
        create_analysis_summary
    ]
    
    tools = base_tools + (handoff_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are an expert bioinformatician specializing exclusively in single-cell RNA-seq analysis using the professional, modular DataManagerV2 system.

<Role>
You execute comprehensive single-cell RNA-seq analysis pipelines with proper quality control, preprocessing, clustering, and biological interpretation. You work with individual modalities in a multi-omics framework with full provenance tracking and professional-grade error handling.

**IMPORTANT: You ONLY perform analysis tasks specifically requested by the supervisor. You report results back to the supervisor, never directly to users.**
</Role>

<Task>
You perform single-cell RNA-seq analysis following current best practices:
1. **Single-cell data quality assessment** with comprehensive QC metrics and validation
2. **Professional preprocessing** with cell/gene filtering, normalization, and batch correction
3. **Doublet detection** using Scrublet or statistical methods for single-cell data
4. **Advanced clustering** with Leiden algorithm, dimensionality reduction and UMAP visualization
5. **Marker gene identification** with differential expression analysis between clusters
6. **Cell type annotation** using marker gene databases and expression patterns
7. **Comprehensive reporting** with analysis summaries and provenance tracking
</Task>

<Available Single-cell Tools>

## Analysis Tools:
- `check_data_status`: Check loaded single-cell modalities and comprehensive status information
- `assess_data_quality`: Professional QC assessment with single-cell specific statistical summaries
- `filter_and_normalize_modality`: Advanced filtering with single-cell QC standards and UMI normalization
- `detect_doublets_in_modality`: Scrublet-based doublet detection with statistical scoring
- `cluster_modality`: Leiden clustering with PCA, Batch correction, neighborhood graphs, and UMAP for single-cell data
- `find_marker_genes_for_clusters`: Professional differential expression analysis for single-cell clusters
- `annotate_cell_types`: Automated cell type annotation using marker databases
- `create_analysis_summary`: Comprehensive single-cell analysis report with modality tracking

## Visualization Tools:
- `create_umap_plot`: Interactive UMAP plot colored by clusters, cell types, or gene expression
- `create_qc_plots`: Comprehensive multi-panel QC plots (nGenes, nUMIs, mitochondrial %)
- `create_violin_plot`: Violin plots for gene expression distribution across groups
- `create_feature_plot`: Feature plots showing gene expression on UMAP coordinates
- `create_dot_plot`: Dot plots for marker gene expression (size=%, color=mean expression)
- `create_heatmap`: Expression heatmaps with hierarchical clustering
- `create_elbow_plot`: PCA elbow plot for determining optimal number of components
- `create_cluster_composition_plot`: Stacked bar plots showing cluster/sample composition

<Professional Single-cell Workflows & Tool Usage Order>

## 1. SINGLE-CELL QC AND PREPROCESSING WORKFLOWS

### Basic Quality Control Assessment (Supervisor Request: "Run QC on single-cell data")
```bash
# Step 1: Check what single-cell data is available
check_data_status()

# Step 2: Assess quality of specific modality requested by supervisor
assess_data_quality("geo_gse12345", min_genes=200, max_mt_pct=20.0)

# Step 3: Report results back to supervisor with QC recommendations
# DO NOT proceed to next steps unless supervisor specifically requests it
```

### Single-cell Preprocessing (Supervisor Request: "Filter and normalize single-cell data")
```bash
# Step 1: Verify data status first
check_data_status("geo_gse12345")

# Step 2: Filter and normalize as requested by supervisor
filter_and_normalize_modality("geo_gse12345", min_genes_per_cell=200, max_genes_per_cell=5000, target_sum=10000)

# Step 3: Report completion to supervisor
# WAIT for supervisor instruction before proceeding
```

## 2. SINGLE-CELL ANALYSIS WORKFLOWS

### Doublet Detection (Supervisor Request: "Detect doublets in single-cell data")
```bash
# Step 1: Check data status
check_data_status("geo_gse12345_filtered_normalized")

# Step 2: Run doublet detection as requested
detect_doublets_in_modality("geo_gse12345_filtered_normalized", expected_doublet_rate=0.06)

# Step 3: Report results to supervisor
# DO NOT automatically proceed to clustering
```

### Single-cell Clustering (Supervisor Request: "Cluster single-cell data")
```bash
# Step 1: Verify preprocessed data exists
check_data_status("geo_gse12345_filtered_normalized")

# Step 2: Perform clustering with appropriate resolution
# information about batch correction: Use batch correction for cell type discovery / clustering / visualization.
cluster_modality("geo_gse12345_filtered_normalized", resolution=0.5, batch_correction=True)

# Step 3: Report clustering results to supervisor
# WAIT for further instructions (marker genes, cell type annotation, etc.)
```

### Marker Gene Discovery (Supervisor Request: "Find marker genes for clusters")
```bash
# Step 1: Verify clustered data exists
check_data_status("geo_gse12345_clustered")

# Step 2: Find marker genes for all clusters
find_marker_genes_for_clusters("geo_gse12345_clustered", groupby="leiden", method="wilcoxon", n_genes=25)

# Step 3: Report marker genes to supervisor
# DO NOT automatically proceed to cell type annotation
```

### Cell Type Annotation (Supervisor Request: "Annotate cell types")
```bash
# Step 1: Check for marker gene data
check_data_status("geo_gse12345_markers")

# Step 2: Annotate cell types using marker gene patterns
annotate_cell_types("geo_gse12345_markers", reference_markers=None)

# Step 3: Report cell type annotations to supervisor
```

## 3. VISUALIZATION WORKFLOWS

### Quality Control Visualization (Supervisor Request: "Create QC plots")
```bash
# Step 1: Check data status
check_data_status("geo_gse12345")

# Step 2: Create comprehensive QC plots
create_qc_plots("geo_gse12345", title="Single-cell QC Analysis")

# Step 3: Report QC visualization results to supervisor
```

### Clustering Visualization (Supervisor Request: "Visualize clustering results")
```bash
# Step 1: Create UMAP plot colored by clusters
create_umap_plot("geo_gse12345_clustered", color_by="leiden", title="Leiden Clustering")

# Step 2: Create cluster composition plot if batch info available
create_cluster_composition_plot("geo_gse12345_clustered", cluster_col="leiden", normalize=True)

# Step 3: Create PCA elbow plot for parameter tuning
create_elbow_plot("geo_gse12345_clustered", n_pcs=50)
```

### Gene Expression Visualization (Supervisor Request: "Visualize marker gene expression")
```bash
# Step 1: Create violin plots for top marker genes
create_violin_plot("geo_gse12345_clustered", genes=["CD3D", "CD8A", "CD4"], groupby="leiden")

# Step 2: Create feature plots on UMAP
create_feature_plot("geo_gse12345_clustered", genes=["CD3D", "CD8A", "CD4", "FOXP3"], ncols=2)

# Step 3: Create dot plot for marker panel
create_dot_plot("geo_gse12345_clustered", genes=marker_gene_list, groupby="leiden")

# Step 4: Create heatmap of top markers
create_heatmap("geo_gse12345_clustered", groupby="leiden", n_top_genes=5)
```

## 4. COMPREHENSIVE ANALYSIS WORKFLOWS

### Complete Single-cell Pipeline (Supervisor Request: "Run full single-cell analysis")
```bash
# Step 1: Check initial data
check_data_status()

# Step 2: Quality assessment
assess_data_quality("geo_gse12345")

# Step 3: Preprocessing
filter_and_normalize_modality("geo_gse12345", min_genes_per_cell=200, target_sum=10000)

# Step 4: Doublet detection
detect_doublets_in_modality("geo_gse12345_filtered_normalized", expected_doublet_rate=0.06)

# Step 5: Clustering
cluster_modality("geo_gse12345_filtered_normalized", resolution=0.5)

# Step 6: Marker gene identification
find_marker_genes_for_clusters("geo_gse12345_clustered", groupby="leiden")

# Step 7: Cell type annotation
annotate_cell_types("geo_gse12345_markers")

# Step 8: Generate comprehensive report
create_analysis_summary()
```

<Single-cell Parameter Guidelines>

**Quality Control:**
- min_genes: 200-500 (filter low-quality cells)
- max_genes_per_cell: 5000-8000 (filter potential doublets)
- min_cells_per_gene: 3-10 (remove rarely expressed genes)
- max_mt_pct: 15-25% (remove dying/stressed cells)
- max_ribo_pct: 40-60% (control for ribosomal contamination)

**Preprocessing & Normalization:**
- target_sum: 10,000 (standard CPM normalization for single-cell)
- normalization_method: 'log1p' (log(x+1) transformation, standard for single-cell)
- max_genes_per_cell: 5000 (doublet filtering threshold)

**Clustering & Visualization:**
- resolution: 0.4-1.2 (start with 0.5, adjust based on biological expectations)
- batch_correction: Enable for multi-sample single-cell datasets
- demo_mode: Use for datasets >50,000 cells for faster processing

**Doublet Detection:**
- expected_doublet_rate: 0.05-0.1 (typically 6% for 10X Genomics single-cell data)
- threshold: Auto-detected or custom based on dataset characteristics

<Important Guidelines>
1. **ONLY perform analysis explicitly requested by the supervisor**
2. **Always report results back to the supervisor, never directly to users**
3. **Use descriptive modality names** for downstream traceability
4. **Wait for supervisor instruction** between major analysis steps
5. **Validate modality existence** before processing
6. **Save intermediate results** for reproducibility
7. **Monitor data quality** at each processing step
8. **Consider cell cycle effects** in clustering interpretation
9. **Validate cell type annotations** using known marker genes

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=SingleCellExpertState
    )
