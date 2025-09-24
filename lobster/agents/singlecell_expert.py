"""
Single-cell RNA-seq Expert Agent for specialized single-cell analysis.

This agent focuses exclusively on single-cell RNA-seq analysis using the modular DataManagerV2 
system with proper modality handling and schema enforcement.
"""

from typing import List, Optional, Union
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

import datetime
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
from lobster.tools.pseudobulk_service import PseudobulkService
from lobster.tools.bulk_rnaseq_service import BulkRNASeqService
from lobster.tools.manual_annotation_service import ManualAnnotationService
from lobster.tools.annotation_templates import AnnotationTemplateService, TissueType
from lobster.tools.differential_formula_service import DifferentialFormulaService
from lobster.core import PseudobulkError, AggregationError, InsufficientCellsError, FormulaError, DesignMatrixError
# COMMENTED OUT FOR SUPERVISOR-MEDIATED FLOW:
# from lobster.tools.enhanced_handoff_tool import create_expert_handoff_tool, SCVI_CONTEXT_SCHEMA
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
    pseudobulk_service = PseudobulkService()
    bulk_rnaseq_service = BulkRNASeqService()
    
    # Initialize manual annotation services
    manual_annotation_service = ManualAnnotationService()
    template_service = AnnotationTemplateService()
    
    # Initialize formula service for agent-guided DE analysis
    formula_service = DifferentialFormulaService()
    
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
        use_rep: Optional[str] = None,
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
            use_rep: Representation to use for clustering (e.g., 'X_scvi' for deep learning embeddings, 'X_pca' for PCA). 
                    If None, uses standard PCA workflow. Custom embeddings like scVI often provide better results.
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
                use_rep=use_rep,
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

    # -------------------------
    # PSEUDOBULK ANALYSIS TOOLS
    # -------------------------
    @tool
    def create_pseudobulk_matrix(
        modality_name: str,
        sample_col: str,
        celltype_col: str,
        layer: str = None,
        min_cells: int = 10,
        aggregation_method: str = 'sum',
        min_genes: int = 200,
        filter_zeros: bool = True,
        save_result: bool = True
    ) -> str:
        """
        Aggregate single-cell data to pseudobulk matrix for differential expression analysis.
        
        Args:
            modality_name: Name of single-cell modality to aggregate
            sample_col: Column containing sample identifiers
            celltype_col: Column containing cell type identifiers
            layer: Layer to use for aggregation (default: X)
            min_cells: Minimum cells per sample-celltype combination
            aggregation_method: Aggregation method ('sum', 'mean', 'median')
            min_genes: Minimum genes detected per pseudobulk sample
            filter_zeros: Remove genes with all zeros after aggregation
            save_result: Whether to save the pseudobulk modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the single-cell modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Creating pseudobulk matrix from single-cell modality '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
            # Validate required columns exist
            if sample_col not in adata.obs.columns:
                available_cols = list(adata.obs.columns)[:10]
                raise PseudobulkError(f"Sample column '{sample_col}' not found. Available columns: {available_cols}...")
            
            if celltype_col not in adata.obs.columns:
                available_cols = list(adata.obs.columns)[:10]
                raise PseudobulkError(f"Cell type column '{celltype_col}' not found. Available columns: {available_cols}...")
            
            # Use pseudobulk service with provenance tracking
            pseudobulk_adata = pseudobulk_service.aggregate_to_pseudobulk(
                adata=adata,
                sample_col=sample_col,
                celltype_col=celltype_col,
                layer=layer,
                min_cells=min_cells,
                aggregation_method=aggregation_method,
                min_genes=min_genes,
                filter_zeros=filter_zeros
            )
            
            # Save as new modality
            pseudobulk_modality_name = f"{modality_name}_pseudobulk"
            data_manager.modalities[pseudobulk_modality_name] = pseudobulk_adata
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_pseudobulk.h5ad"
                data_manager.save_modality(pseudobulk_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_pseudobulk_matrix",
                parameters={
                    "modality_name": modality_name,
                    "sample_col": sample_col,
                    "celltype_col": celltype_col,
                    "aggregation_method": aggregation_method,
                    "min_cells": min_cells
                },
                description=f"Created pseudobulk matrix: {pseudobulk_adata.n_obs} samples × {pseudobulk_adata.n_vars} genes"
            )
            
            # Get aggregation statistics
            agg_stats = pseudobulk_adata.uns.get('aggregation_stats', {})
            
            # Format professional response
            response = f"""Successfully created pseudobulk matrix from single-cell data '{modality_name}'!

📊 **Pseudobulk Aggregation Results:**
- Original: {adata.n_obs:,} single cells → {pseudobulk_adata.n_obs} pseudobulk samples
- Genes retained: {pseudobulk_adata.n_vars:,} / {adata.n_vars:,} ({pseudobulk_adata.n_vars/adata.n_vars*100:.1f}%)
- Aggregation method: {aggregation_method}
- Min cells threshold: {min_cells}

📈 **Sample & Cell Type Distribution:**
- Unique samples: {agg_stats.get('n_samples', 'N/A')}
- Cell types: {agg_stats.get('n_cell_types', 'N/A')}
- Total cells aggregated: {agg_stats.get('total_cells_aggregated', 'N/A'):,}
- Mean cells per pseudobulk: {agg_stats.get('mean_cells_per_pseudobulk', 0):.1f}

💾 **New modality created**: '{pseudobulk_modality_name}'"""

            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            response += f"\n\nNext step: Use 'prepare_differential_expression_design' to set up statistical design for DE analysis."
            
            analysis_results["details"]["pseudobulk_aggregation"] = response
            return response
            
        except (PseudobulkError, AggregationError, InsufficientCellsError, ModalityNotFoundError) as e:
            logger.error(f"Error creating pseudobulk matrix: {e}")
            return f"Error creating pseudobulk matrix: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pseudobulk creation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def prepare_differential_expression_design(
        modality_name: str,
        formula: str,
        contrast: List[str],
        reference_condition: str = None
    ) -> str:
        """
        Prepare design matrix and validate experimental design for differential expression analysis.
        
        Args:
            modality_name: Name of pseudobulk modality
            formula: R-style formula (e.g., "~condition + batch")
            contrast: Contrast specification [factor, level1, level2]
            reference_condition: Reference level for main condition
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the pseudobulk modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Preparing DE design for pseudobulk modality '{modality_name}': {adata.shape[0]} samples × {adata.shape[1]} genes")
            
            # Validate design using bulk RNA-seq service
            design_validation = bulk_rnaseq_service.validate_experimental_design(
                metadata=adata.obs,
                formula=formula,
                min_replicates=2
            )
            
            if not design_validation['valid']:
                error_msg = "; ".join(design_validation['errors'])
                raise DesignMatrixError(f"Invalid experimental design: {error_msg}")
            
            # Create design matrix
            design_result = bulk_rnaseq_service.create_formula_design(
                metadata=adata.obs,
                condition_col=contrast[0],
                reference_condition=reference_condition
            )
            
            # Store design information in modality
            adata.uns['formula_design'] = {
                'formula': formula,
                'contrast': contrast,
                'design_matrix_info': design_result,
                'validation_results': design_validation
            }
            
            # Update modality with design info
            data_manager.modalities[modality_name] = adata
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="prepare_differential_expression_design",
                parameters={
                    "modality_name": modality_name,
                    "formula": formula,
                    "contrast": contrast
                },
                description=f"Prepared DE design for {adata.n_obs} pseudobulk samples"
            )
            
            # Format response
            response = f"""Successfully prepared differential expression design for '{modality_name}'!

📊 **Experimental Design:**
- Formula: {formula}
- Contrast: {contrast[1]} vs {contrast[2]} in {contrast[0]}
- Design matrix: {design_result['design_matrix'].shape[0]} samples × {design_result['design_matrix'].shape[1]} coefficients
- Matrix rank: {design_result['rank']} (full rank: {'✓' if design_result['rank'] == design_result['n_coefficients'] else '✗'})

📈 **Design Validation:**
- Valid: {'✓' if design_validation['valid'] else '✗'}
- Warnings: {len(design_validation['warnings'])} ({', '.join(design_validation['warnings'][:2]) if design_validation['warnings'] else 'None'})

🔬 **Sample Distribution:**"""
            
            for factor, counts in design_validation.get('design_summary', {}).items():
                response += f"\n- {factor}: {dict(list(counts.items())[:5])}"
            
            response += f"\n\n💾 **Design information stored in**: adata.uns['formula_design']"
            response += f"\n\nNext step: Run 'run_pseudobulk_differential_expression' to perform pyDESeq2 analysis."
            
            analysis_results["details"]["de_design"] = response
            return response
            
        except (DesignMatrixError, FormulaError, ModalityNotFoundError) as e:
            logger.error(f"Error preparing DE design: {e}")
            return f"Error preparing differential expression design: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in DE design preparation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def run_pseudobulk_differential_expression(
        modality_name: str,
        alpha: float = 0.05,
        shrink_lfc: bool = True,
        n_cpus: int = 1,
        save_result: bool = True
    ) -> str:
        """
        Run pyDESeq2 differential expression analysis on pseudobulk data.
        
        Args:
            modality_name: Name of pseudobulk modality with design prepared
            alpha: Significance threshold for multiple testing
            shrink_lfc: Whether to apply log fold change shrinkage
            n_cpus: Number of CPUs for parallel processing
            save_result: Whether to save results
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the pseudobulk modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Running pyDESeq2 DE analysis on pseudobulk modality '{modality_name}': {adata.shape[0]} samples × {adata.shape[1]} genes")
            
            # Validate design exists
            if 'formula_design' not in adata.uns:
                raise PseudobulkError("No design matrix prepared. Run 'prepare_differential_expression_design' first.")
            
            design_info = adata.uns['formula_design']
            formula = design_info['formula']
            contrast = design_info['contrast']
            
            # Run pyDESeq2 analysis using bulk RNA-seq service
            results_df, analysis_stats = bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                pseudobulk_adata=adata,
                formula=formula,
                contrast=contrast,
                alpha=alpha,
                shrink_lfc=shrink_lfc,
                n_cpus=n_cpus
            )
            
            # Store results in modality
            contrast_name = f"{contrast[0]}_{contrast[1]}_vs_{contrast[2]}"
            adata.uns[f'de_results_{contrast_name}'] = {
                'results_df': results_df,
                'analysis_stats': analysis_stats,
                'parameters': {
                    'alpha': alpha,
                    'shrink_lfc': shrink_lfc,
                    'formula': formula,
                    'contrast': contrast
                }
            }
            
            # Update modality with results
            data_manager.modalities[modality_name] = adata
            
            # Save results if requested
            if save_result:
                results_path = f"{modality_name}_de_results.csv"
                results_df.to_csv(results_path)
                
                save_path = f"{modality_name}_with_de_results.h5ad"
                data_manager.save_modality(modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="run_pseudobulk_differential_expression",
                parameters={
                    "modality_name": modality_name,
                    "formula": formula,
                    "contrast": contrast,
                    "alpha": alpha,
                    "shrink_lfc": shrink_lfc
                },
                description=f"pyDESeq2 analysis: {analysis_stats['n_significant_genes']} significant genes found"
            )
            
            # Format response
            response = f"""Successfully completed pyDESeq2 differential expression analysis on '{modality_name}'!

📊 **pyDESeq2 Analysis Results:**
- Contrast: {contrast[1]} vs {contrast[2]} in {contrast[0]}
- Genes tested: {analysis_stats['n_genes_tested']:,}
- Significant genes: {analysis_stats['n_significant_genes']:,} (α={alpha})
- Upregulated: {analysis_stats['n_upregulated']:,}
- Downregulated: {analysis_stats['n_downregulated']:,}

🧬 **Top Differentially Expressed Genes:**
**Upregulated ({contrast[1]} > {contrast[2]}):**
{chr(10).join([f"- {gene}" for gene in analysis_stats['top_upregulated'][:5]])}

**Downregulated ({contrast[1]} < {contrast[2]}):**
{chr(10).join([f"- {gene}" for gene in analysis_stats['top_downregulated'][:5]])}

📈 **Analysis Parameters:**
- Formula: {formula}
- LFC shrinkage: {'✓' if shrink_lfc else '✗'}
- Parallel CPUs: {n_cpus}
- Significance threshold: {alpha}

💾 **Results stored in**: adata.uns['de_results_{contrast_name}']"""

            if save_result:
                response += f"\n💾 **Saved to**: {results_path} & {save_path}"
            
            response += f"\n\nNext steps: Visualize results with volcano plots or run pathway enrichment analysis."
            
            analysis_results["details"]["differential_expression"] = response
            return response
            
        except (PseudobulkError, ModalityNotFoundError) as e:
            logger.error(f"Error in pseudobulk DE analysis: {e}")
            return f"Error in pseudobulk differential expression analysis: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pseudobulk DE analysis: {e}")
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
                
                # Filter for pseudobulk modalities
                pb_modalities = [mod for mod in modalities if 'pseudobulk' in mod.lower()]
                
                summary += f"## Current Single-cell Modalities\n"
                summary += f"Single-cell modalities ({len(sc_modalities)}): {', '.join(sc_modalities)}\n"
                if pb_modalities:
                    summary += f"Pseudobulk modalities ({len(pb_modalities)}): {', '.join(pb_modalities)}\n"
                summary += "\n"
                
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
                
                # Add pseudobulk modality details
                if pb_modalities:
                    summary += "\n### Pseudobulk Modality Details:\n"
                    for mod_name in pb_modalities:
                        try:
                            adata = data_manager.get_modality(mod_name)
                            summary += f"- **{mod_name}**: {adata.n_obs} pseudobulk samples × {adata.n_vars} genes\n"
                            
                            # Add DE results if available
                            de_keys = [key for key in adata.uns.keys() if key.startswith('de_results_')]
                            if de_keys:
                                summary += f"  - DE analyses: {', '.join([key.replace('de_results_', '') for key in de_keys])}\n"
                        except Exception as e:
                            summary += f"- **{mod_name}**: Error accessing modality\n"
            
            analysis_results["summary"] = summary
            logger.info(f"Created single-cell analysis summary with {len(analysis_results['details'])} analysis steps")
            return summary
            
        except Exception as e:
            logger.error(f"Error creating single-cell analysis summary: {e}")
            return f"Error creating single-cell summary: {str(e)}"

    # -------------------------
    # MANUAL ANNOTATION TOOLS
    # -------------------------
    @tool
    def manually_annotate_clusters_interactive(
        modality_name: str,
        cluster_col: str = "leiden",
        save_result: bool = True
    ) -> str:
        """
        Launch Rich terminal interface for manual cluster annotation with color synchronization.
        
        Args:
            modality_name: Name of clustered single-cell modality
            cluster_col: Column containing cluster assignments
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(f"Launching interactive annotation for '{modality_name}': {adata.shape[0]} cells × {adata.shape[1]} genes")
            
            # Validate cluster column exists
            if cluster_col not in adata.obs.columns:
                available_cols = [col for col in adata.obs.columns if col in ['leiden', 'cell_type', 'louvain']]
                return f"Cluster column '{cluster_col}' not found. Available: {available_cols}"
            
            # Initialize annotation session
            annotation_state = manual_annotation_service.initialize_annotation_session(
                adata=adata,
                cluster_key=cluster_col
            )
            
            # Launch Rich terminal interface
            cell_type_mapping = manual_annotation_service.rich_annotation_interface()
            
            # Apply annotations to data
            adata_annotated = manual_annotation_service.apply_annotations_to_adata(
                adata=adata,
                cluster_key=cluster_col,
                cell_type_column='cell_type_manual'
            )
            
            # Save as new modality
            annotated_modality_name = f"{modality_name}_manually_annotated"
            data_manager.modalities[annotated_modality_name] = adata_annotated
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_manually_annotated.h5ad"
                data_manager.save_modality(annotated_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="manually_annotate_clusters_interactive",
                parameters={
                    "modality_name": modality_name,
                    "cluster_col": cluster_col,
                    "n_annotations": len(cell_type_mapping)
                },
                description=f"Manual annotation completed for {len(cell_type_mapping)} clusters"
            )
            
            # Validate results
            validation = manual_annotation_service.validate_annotation_coverage(
                adata_annotated, 'cell_type_manual'
            )
            
            # Format response
            response = f"""Manual cluster annotation completed for '{modality_name}'!

📊 **Interactive Annotation Results:**
- Total clusters: {len(annotation_state.clusters)}
- Manually annotated: {len(cell_type_mapping)}
- Marked as debris: {len(annotation_state.debris_clusters)}
- Coverage: {validation['coverage_percentage']:.1f}%

🎨 **Color-Synchronized Interface:**
- Rich terminal colors matched UMAP plot colors
- Visual cluster identification completed
- Expert-guided annotation workflow

📈 **Cell Type Distribution:**"""
            
            for cell_type, count in list(validation['cell_type_counts'].items())[:8]:
                response += f"\n- {cell_type}: {count} cells"
            
            response += f"\n\n💾 **New modality created**: '{annotated_modality_name}'"
            response += f"\n🔬 **Manual annotations in**: adata.obs['cell_type_manual']"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            response += f"\n\nManual annotation complete! Use for downstream analysis or pseudobulk aggregation."
            
            analysis_results["details"]["manual_annotation"] = response
            return response
            
        except Exception as e:
            logger.error(f"Error in interactive manual annotation: {e}")
            return f"Error in manual cluster annotation: {str(e)}"

    @tool
    def manually_annotate_clusters(
        modality_name: str,
        annotations: dict,
        cluster_col: str = "leiden",
        save_result: bool = True
    ) -> str:
        """
        Directly assign cell types to clusters without interactive interface.
        
        Args:
            modality_name: Name of clustered single-cell modality
            annotations: Dictionary mapping cluster IDs to cell type names
            cluster_col: Column containing cluster assignments
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Validate cluster column exists
            if cluster_col not in adata.obs.columns:
                return f"Cluster column '{cluster_col}' not found."
            
            # Apply annotations directly
            adata_copy = adata.copy()
            cell_type_mapping = {}
            
            for cluster_id, cell_type in annotations.items():
                cell_type_mapping[str(cluster_id)] = cell_type
            
            # Create cell type column
            adata_copy.obs['cell_type_manual'] = adata_copy.obs[cluster_col].astype(str).map(
                cell_type_mapping
            ).fillna('Unassigned')
            
            # Save as new modality
            annotated_modality_name = f"{modality_name}_manually_annotated"
            data_manager.modalities[annotated_modality_name] = adata_copy
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_manually_annotated.h5ad"
                data_manager.save_modality(annotated_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="manually_annotate_clusters",
                parameters={
                    "modality_name": modality_name,
                    "cluster_col": cluster_col,
                    "annotations": annotations
                },
                description=f"Direct manual annotation of {len(annotations)} clusters"
            )
            
            response = f"""Manual cluster annotation applied to '{modality_name}'!

📊 **Annotation Results:**
- Clusters annotated: {len(annotations)}
- Cell types assigned: {len(set(annotations.values()))}

📈 **Annotations Applied:**"""
            
            for cluster_id, cell_type in list(annotations.items())[:10]:
                response += f"\n- Cluster {cluster_id}: {cell_type}"
            
            if len(annotations) > 10:
                response += f"\n... and {len(annotations) - 10} more clusters"
            
            response += f"\n\n💾 **New modality created**: '{annotated_modality_name}'"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in manual annotation: {e}")
            return f"Error applying manual annotations: {str(e)}"

    @tool
    def collapse_clusters_to_celltype(
        modality_name: str,
        cluster_list: List[str],
        cell_type_name: str,
        cluster_col: str = "leiden",
        save_result: bool = True
    ) -> str:
        """
        Collapse multiple clusters into a single cell type.
        
        Args:
            modality_name: Name of single-cell modality
            cluster_list: List of cluster IDs to collapse
            cell_type_name: New cell type name for collapsed clusters
            cluster_col: Column containing cluster assignments
            save_result: Whether to save result
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Validate clusters exist
            unique_clusters = set(adata.obs[cluster_col].astype(str).unique())
            invalid_clusters = [c for c in cluster_list if str(c) not in unique_clusters]
            if invalid_clusters:
                return f"Invalid cluster IDs: {invalid_clusters}. Available: {sorted(unique_clusters)}"
            
            # Create collapsed annotation
            adata_copy = adata.copy()
            
            # Create or update manual cell type column
            if 'cell_type_manual' not in adata_copy.obs:
                adata_copy.obs['cell_type_manual'] = 'Unassigned'
            
            # Apply collapse
            for cluster_id in cluster_list:
                mask = adata_copy.obs[cluster_col].astype(str) == str(cluster_id)
                adata_copy.obs.loc[mask, 'cell_type_manual'] = cell_type_name
            
            # Calculate statistics
            total_cells_collapsed = sum(
                (adata_copy.obs[cluster_col].astype(str) == str(c)).sum() 
                for c in cluster_list
            )
            
            # Save as new modality
            collapsed_modality_name = f"{modality_name}_collapsed"
            data_manager.modalities[collapsed_modality_name] = adata_copy
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_collapsed.h5ad"
                data_manager.save_modality(collapsed_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="collapse_clusters_to_celltype",
                parameters={
                    "modality_name": modality_name,
                    "cluster_list": cluster_list,
                    "cell_type_name": cell_type_name,
                    "cluster_col": cluster_col
                },
                description=f"Collapsed {len(cluster_list)} clusters into '{cell_type_name}'"
            )
            
            response = f"""Successfully collapsed clusters in '{modality_name}'!

📊 **Collapse Results:**
- Clusters collapsed: {', '.join(cluster_list)}
- New cell type: {cell_type_name}
- Total cells affected: {total_cells_collapsed:,}

💾 **New modality created**: '{collapsed_modality_name}'"""
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            response += f"\n\nClusters {', '.join(cluster_list)} are now annotated as '{cell_type_name}'."
            
            return response
            
        except Exception as e:
            logger.error(f"Error collapsing clusters: {e}")
            return f"Error collapsing clusters: {str(e)}"

    @tool
    def mark_clusters_as_debris(
        modality_name: str,
        debris_clusters: List[str],
        remove_debris: bool = False,
        cluster_col: str = "leiden",
        save_result: bool = True
    ) -> str:
        """
        Mark specified clusters as debris for quality control.
        
        Args:
            modality_name: Name of single-cell modality
            debris_clusters: List of cluster IDs to mark as debris
            remove_debris: Whether to remove debris clusters from data
            cluster_col: Column containing cluster assignments
            save_result: Whether to save result
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Validate clusters exist
            unique_clusters = set(adata.obs[cluster_col].astype(str).unique())
            invalid_clusters = [c for c in debris_clusters if str(c) not in unique_clusters]
            if invalid_clusters:
                return f"Invalid cluster IDs: {invalid_clusters}. Available: {sorted(unique_clusters)}"
            
            adata_copy = adata.copy()
            
            # Mark debris clusters
            if 'cell_type_manual' not in adata_copy.obs:
                adata_copy.obs['cell_type_manual'] = 'Unassigned'
            
            debris_mask = adata_copy.obs[cluster_col].astype(str).isin([str(c) for c in debris_clusters])
            adata_copy.obs.loc[debris_mask, 'cell_type_manual'] = 'Debris'
            
            # Add debris flag
            adata_copy.obs['is_debris'] = False
            adata_copy.obs.loc[debris_mask, 'is_debris'] = True
            
            # Optionally remove debris
            if remove_debris:
                adata_copy = adata_copy[~debris_mask].copy()
                
            total_debris_cells = debris_mask.sum()
            
            # Save as new modality
            debris_modality_name = f"{modality_name}_debris_marked"
            if remove_debris:
                debris_modality_name = f"{modality_name}_debris_removed"
                
            data_manager.modalities[debris_modality_name] = adata_copy
            
            # Save to file if requested
            if save_result:
                save_path = f"{debris_modality_name}.h5ad"
                data_manager.save_modality(debris_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="mark_clusters_as_debris",
                parameters={
                    "modality_name": modality_name,
                    "debris_clusters": debris_clusters,
                    "remove_debris": remove_debris,
                    "cluster_col": cluster_col
                },
                description=f"Marked {len(debris_clusters)} clusters as debris ({total_debris_cells} cells)"
            )
            
            response = f"""Successfully marked debris clusters in '{modality_name}'!

📊 **Debris Marking Results:**
- Clusters marked: {', '.join(debris_clusters)}
- Total debris cells: {total_debris_cells:,}
- Action: {'Removed' if remove_debris else 'Marked only'}

💾 **New modality created**: '{debris_modality_name}'"""
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            if remove_debris:
                remaining_cells = adata_copy.n_obs
                response += f"\n🔬 **Remaining cells**: {remaining_cells:,} ({remaining_cells/adata.n_obs*100:.1f}%)"
            else:
                response += f"\n🔬 **Debris flag added**: adata.obs['is_debris']"
            
            return response
            
        except Exception as e:
            logger.error(f"Error marking debris clusters: {e}")
            return f"Error marking clusters as debris: {str(e)}"

    @tool
    def suggest_debris_clusters(
        modality_name: str,
        min_genes: int = 200,
        max_mt_percent: float = 50,
        min_umi: int = 500,
        cluster_col: str = "leiden"
    ) -> str:
        """
        Get smart suggestions for potential debris clusters based on QC metrics.
        
        Args:
            modality_name: Name of single-cell modality
            min_genes: Minimum genes per cell threshold
            max_mt_percent: Maximum mitochondrial percentage
            min_umi: Minimum UMI count threshold  
            cluster_col: Column containing cluster assignments
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Get suggestions using manual annotation service
            suggested_debris = manual_annotation_service.suggest_debris_clusters(
                adata=adata,
                min_genes=min_genes,
                max_mt_percent=max_mt_percent,
                min_umi=min_umi
            )
            
            if not suggested_debris:
                return f"No debris clusters suggested based on QC thresholds (min_genes={min_genes}, max_mt%={max_mt_percent}, min_umi={min_umi})"
            
            # Get cluster statistics for suggestions
            response = f"""Smart debris cluster suggestions for '{modality_name}':

📊 **QC-Based Suggestions:**
- Clusters flagged: {len(suggested_debris)}
- Thresholds used: min_genes={min_genes}, max_mt%={max_mt_percent}, min_umi={min_umi}

🗑️ **Suggested Debris Clusters:**"""
            
            for cluster_id in suggested_debris[:10]:
                cluster_mask = adata.obs[cluster_col].astype(str) == cluster_id
                n_cells = cluster_mask.sum()
                
                # Get QC stats for cluster
                if cluster_mask.sum() > 0:
                    mean_genes = adata.obs.loc[cluster_mask, 'n_genes'].mean() if 'n_genes' in adata.obs else 0
                    mean_mt = adata.obs.loc[cluster_mask, 'percent_mito'].mean() if 'percent_mito' in adata.obs else 0
                    mean_umi = adata.obs.loc[cluster_mask, 'n_counts'].mean() if 'n_counts' in adata.obs else 0
                    
                    response += f"\n- Cluster {cluster_id}: {n_cells} cells (genes: {mean_genes:.0f}, MT: {mean_mt:.1f}%, UMI: {mean_umi:.0f})"
            
            if len(suggested_debris) > 10:
                response += f"\n... and {len(suggested_debris) - 10} more clusters"
            
            response += f"\n\n💡 **Recommendation:**"
            response += f"\nUse 'mark_clusters_as_debris' to apply these suggestions."
            response += f"\nExample: mark_clusters_as_debris('{modality_name}', {suggested_debris[:5]})"
            
            return response
            
        except Exception as e:
            logger.error(f"Error suggesting debris clusters: {e}")
            return f"Error suggesting debris clusters: {str(e)}"

    @tool
    def review_annotation_assignments(
        modality_name: str,
        annotation_col: str = "cell_type_manual",
        show_unassigned: bool = True,
        show_debris: bool = True
    ) -> str:
        """
        Review current manual annotation assignments.
        
        Args:
            modality_name: Name of modality with annotations
            annotation_col: Column containing cell type annotations
            show_unassigned: Whether to show unassigned clusters
            show_debris: Whether to show debris clusters
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            if annotation_col not in adata.obs.columns:
                return f"Annotation column '{annotation_col}' not found. Available columns: {list(adata.obs.columns)[:10]}"
            
            # Validate annotation coverage
            validation = manual_annotation_service.validate_annotation_coverage(
                adata, annotation_col
            )
            
            response = f"""Annotation review for '{modality_name}':

📊 **Coverage Summary:**
- Total cells: {validation['total_cells']:,}
- Annotated cells: {validation['annotated_cells']:,} ({validation['coverage_percentage']:.1f}%)
- Unassigned cells: {validation['unassigned_cells']:,}
- Debris cells: {validation['debris_cells']:,}
- Unique cell types: {validation['unique_cell_types']}

📈 **Cell Type Distribution:**"""
            
            # Show all cell types
            for cell_type, count in validation['cell_type_counts'].items():
                if cell_type == 'Unassigned' and not show_unassigned:
                    continue
                if cell_type == 'Debris' and not show_debris:
                    continue
                
                percentage = (count / validation['total_cells']) * 100
                response += f"\n- {cell_type}: {count:,} cells ({percentage:.1f}%)"
            
            # Add quality assessment
            if validation['coverage_percentage'] >= 90:
                response += f"\n\n✅ **Quality**: Excellent annotation coverage"
            elif validation['coverage_percentage'] >= 70:
                response += f"\n\n⚠️ **Quality**: Good annotation coverage, consider annotating remaining clusters"
            else:
                response += f"\n\n❌ **Quality**: Low annotation coverage, more annotation needed"
            
            return response
            
        except Exception as e:
            logger.error(f"Error reviewing annotations: {e}")
            return f"Error reviewing annotation assignments: {str(e)}"

    @tool
    def apply_annotation_template(
        modality_name: str,
        tissue_type: str,
        cluster_col: str = "leiden",
        expression_threshold: float = 0.5,
        save_result: bool = True
    ) -> str:
        """
        Apply predefined tissue-specific annotation template to suggest cell types.
        
        Args:
            modality_name: Name of single-cell modality
            tissue_type: Type of tissue (pbmc, brain, lung, heart, kidney, liver, intestine, skin, tumor)
            cluster_col: Column containing cluster assignments
            expression_threshold: Minimum expression for marker detection
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Validate tissue type
            try:
                tissue_enum = TissueType(tissue_type.lower())
            except ValueError:
                available_tissues = [t.value for t in TissueType]
                return f"Invalid tissue type '{tissue_type}'. Available: {available_tissues}"
            
            # Apply template
            cluster_suggestions = template_service.apply_template_to_clusters(
                adata=adata,
                tissue_type=tissue_enum,
                cluster_col=cluster_col,
                expression_threshold=expression_threshold
            )
            
            if not cluster_suggestions:
                return f"No template suggestions generated for tissue type '{tissue_type}'"
            
            # Apply suggestions to data
            adata_copy = adata.copy()
            adata_copy.obs['cell_type_template'] = adata_copy.obs[cluster_col].astype(str).map(
                cluster_suggestions
            ).fillna('Unknown')
            
            # Save as new modality
            template_modality_name = f"{modality_name}_template_{tissue_type}"
            data_manager.modalities[template_modality_name] = adata_copy
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_template_{tissue_type}.h5ad"
                data_manager.save_modality(template_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="apply_annotation_template",
                parameters={
                    "modality_name": modality_name,
                    "tissue_type": tissue_type,
                    "cluster_col": cluster_col,
                    "expression_threshold": expression_threshold
                },
                description=f"Applied {tissue_type} template: {len(cluster_suggestions)} clusters annotated"
            )
            
            # Get template cell types
            template = template_service.get_template(tissue_enum)
            available_types = list(template.keys()) if template else []
            
            response = f"""Applied {tissue_type.upper()} annotation template to '{modality_name}'!

📊 **Template Application Results:**
- Tissue type: {tissue_type.upper()}
- Clusters analyzed: {len(cluster_suggestions)}
- Expression threshold: {expression_threshold}

📈 **Suggested Annotations:**"""
            
            # Show suggestions
            suggestion_counts = {}
            for cluster_id, cell_type in cluster_suggestions.items():
                if cell_type not in suggestion_counts:
                    suggestion_counts[cell_type] = []
                suggestion_counts[cell_type].append(cluster_id)
            
            for cell_type, clusters in suggestion_counts.items():
                response += f"\n- {cell_type}: clusters {', '.join(sorted(clusters))}"
            
            response += f"\n\n🧬 **Available cell types in {tissue_type} template:**"
            response += f"\n{', '.join(available_types[:8])}"
            if len(available_types) > 8:
                response += f"... and {len(available_types) - 8} more"
            
            response += f"\n\n💾 **New modality created**: '{template_modality_name}'"
            response += f"\n🔬 **Template suggestions in**: adata.obs['cell_type_template']"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            response += f"\n\n💡 **Next steps:** Review suggestions and refine with manual annotation if needed."
            
            return response
            
        except Exception as e:
            logger.error(f"Error applying annotation template: {e}")
            return f"Error applying annotation template: {str(e)}"

    @tool
    def export_annotation_mapping(
        modality_name: str,
        annotation_col: str = "cell_type_manual",
        output_filename: str = "annotation_mapping.json",
        format: str = "json"
    ) -> str:
        """
        Export annotation mapping for reuse in other analyses.
        
        Args:
            modality_name: Name of annotated modality
            annotation_col: Column containing cell type annotations
            output_filename: Output filename
            format: Export format ('json' or 'csv')
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            if annotation_col not in adata.obs.columns:
                return f"Annotation column '{annotation_col}' not found."
            
            # Create export data
            export_data = {
                'modality_name': modality_name,
                'annotation_column': annotation_col,
                'export_timestamp': datetime.now().isoformat(),
                'total_cells': adata.n_obs,
                'cell_type_mapping': {},
                'cell_type_counts': adata.obs[annotation_col].value_counts().to_dict()
            }
            
            # Create cluster-to-celltype mapping if cluster info available
            cluster_cols = [col for col in adata.obs.columns if col in ['leiden', 'louvain']]
            if cluster_cols:
                cluster_col = cluster_cols[0]
                cluster_mapping = {}
                for cluster_id in adata.obs[cluster_col].unique():
                    cluster_mask = adata.obs[cluster_col] == cluster_id
                    most_common_type = adata.obs.loc[cluster_mask, annotation_col].mode().iloc[0]
                    cluster_mapping[str(cluster_id)] = most_common_type
                
                export_data['cluster_to_celltype'] = cluster_mapping
                export_data['cluster_column'] = cluster_col
            
            # Export based on format
            if format.lower() == 'json':
                import json
                with open(output_filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif format.lower() == 'csv':
                # Export as CSV
                df_data = []
                for cell_type, count in export_data['cell_type_counts'].items():
                    df_data.append({
                        'cell_type': cell_type,
                        'cell_count': count,
                        'percentage': (count / export_data['total_cells']) * 100
                    })
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_filename, index=False)
            else:
                return f"Unsupported export format: {format}. Use 'json' or 'csv'."
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="export_annotation_mapping",
                parameters={
                    "modality_name": modality_name,
                    "annotation_col": annotation_col,
                    "output_filename": output_filename,
                    "format": format
                },
                description=f"Exported annotation mapping with {len(export_data['cell_type_counts'])} cell types"
            )
            
            response = f"""Successfully exported annotation mapping for '{modality_name}'!

📊 **Export Details:**
- Annotation column: {annotation_col}
- Output file: {output_filename}
- Format: {format.upper()}
- Cell types: {len(export_data['cell_type_counts'])}

📈 **Exported Data:**
- Total cells: {export_data['total_cells']:,}
- Cell type counts included
- Cluster mapping included (if available)
- Export timestamp: {export_data['export_timestamp']}

💾 **File created**: {output_filename}

Use this mapping to apply consistent annotations to similar datasets."""
            
            return response
            
        except Exception as e:
            logger.error(f"Error exporting annotation mapping: {e}")
            return f"Error exporting annotation mapping: {str(e)}"

    @tool
    def import_annotation_mapping(
        modality_name: str,
        mapping_file: str,
        preview_only: bool = False,
        save_result: bool = True
    ) -> str:
        """
        Import and apply annotation mapping from previous analysis.
        
        Args:
            modality_name: Name of modality to annotate
            mapping_file: Path to mapping file (JSON format)
            preview_only: If True, only show what would be applied
            save_result: Whether to save annotated modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}")
            
            # Get the modality
            adata = data_manager.get_modality(modality_name)
            
            # Load mapping file
            import json
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
            
            if preview_only:
                response = f"""Preview of annotation mapping from '{mapping_file}':

📊 **Mapping File Details:**
- Source modality: {mapping_data.get('modality_name', 'N/A')}
- Annotation column: {mapping_data.get('annotation_column', 'N/A')}
- Export timestamp: {mapping_data.get('export_timestamp', 'N/A')}

📈 **Cell Types in Mapping:**"""
                
                for cell_type, count in mapping_data.get('cell_type_counts', {}).items():
                    response += f"\n- {cell_type}: {count} cells"
                
                if 'cluster_to_celltype' in mapping_data:
                    response += f"\n\n🔗 **Cluster Mappings:**"
                    cluster_mapping = mapping_data['cluster_to_celltype']
                    for cluster_id, cell_type in list(cluster_mapping.items())[:10]:
                        response += f"\n- Cluster {cluster_id}: {cell_type}"
                    
                    if len(cluster_mapping) > 10:
                        response += f"\n... and {len(cluster_mapping) - 10} more clusters"
                
                response += f"\n\nUse preview_only=False to apply this mapping to '{modality_name}'."
                return response
            
            # Apply mapping
            adata_copy = adata.copy()
            
            if 'cluster_to_celltype' in mapping_data and 'cluster_column' in mapping_data:
                cluster_col = mapping_data['cluster_column']
                cluster_mapping = mapping_data['cluster_to_celltype']
                
                if cluster_col in adata_copy.obs.columns:
                    adata_copy.obs['cell_type_imported'] = adata_copy.obs[cluster_col].astype(str).map(
                        cluster_mapping
                    ).fillna('Unassigned')
                else:
                    return f"Cluster column '{cluster_col}' from mapping not found in modality."
            else:
                return "Mapping file does not contain cluster-to-celltype information."
            
            # Save as new modality
            imported_modality_name = f"{modality_name}_imported_annotations"
            data_manager.modalities[imported_modality_name] = adata_copy
            
            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_imported_annotations.h5ad"
                data_manager.save_modality(imported_modality_name, save_path)
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="import_annotation_mapping",
                parameters={
                    "modality_name": modality_name,
                    "mapping_file": mapping_file,
                    "preview_only": preview_only
                },
                description=f"Imported annotation mapping from {mapping_file}"
            )
            
            # Validate imported annotations
            validation = manual_annotation_service.validate_annotation_coverage(
                adata_copy, 'cell_type_imported'
            )
            
            response = f"""Successfully imported annotation mapping to '{modality_name}'!

📊 **Import Results:**
- Mapping file: {mapping_file}
- Clusters mapped: {len(cluster_mapping)}
- Coverage: {validation['coverage_percentage']:.1f}%

📈 **Imported Cell Types:**"""
            
            for cell_type, count in list(validation['cell_type_counts'].items())[:8]:
                response += f"\n- {cell_type}: {count:,} cells"
            
            response += f"\n\n💾 **New modality created**: '{imported_modality_name}'"
            response += f"\n🔬 **Imported annotations in**: adata.obs['cell_type_imported']"
            
            if save_result:
                response += f"\n💾 **Saved to**: {save_path}"
            
            return response
            
        except FileNotFoundError:
            return f"Mapping file not found: {mapping_file}"
        except Exception as e:
            logger.error(f"Error importing annotation mapping: {e}")
            return f"Error importing annotation mapping: {str(e)}"

    # -------------------------
    # AGENT-GUIDED FORMULA CONSTRUCTION TOOLS
    # -------------------------
    @tool
    def suggest_formula_for_design(
        pseudobulk_modality: str,
        analysis_goal: Optional[str] = None,
        show_metadata_summary: bool = True
    ) -> str:
        """
        Analyze metadata and suggest appropriate formulas for differential expression analysis.
        
        The agent examines the pseudobulk metadata structure and suggests 2-3 formula options
        based on available variables, explaining each in plain language with pros/cons.
        
        Args:
            pseudobulk_modality: Name of pseudobulk modality to analyze
            analysis_goal: Optional description of analysis goals
            show_metadata_summary: Whether to show detailed metadata summary
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"
            
            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)
            logger.info(f"Analyzing formula design for pseudobulk modality '{pseudobulk_modality}': {adata.shape[0]} samples × {adata.shape[1]} genes")
            
            # Analyze metadata structure
            metadata = adata.obs
            n_samples = len(metadata)
            
            # Identify variable types and characteristics
            variable_analysis = {}
            for col in metadata.columns:
                if col.startswith('_') or col in ['n_cells', 'total_counts']:
                    continue  # Skip internal columns
                    
                series = metadata[col]
                if pd.api.types.is_numeric_dtype(series):
                    var_type = 'continuous'
                    unique_vals = len(series.unique())
                    missing = series.isna().sum()
                else:
                    var_type = 'categorical'
                    unique_vals = len(series.unique())
                    missing = series.isna().sum()
                
                variable_analysis[col] = {
                    'type': var_type,
                    'unique_values': unique_vals,
                    'missing_count': missing,
                    'sample_values': list(series.unique())[:5]
                }
            
            # Generate formula suggestions
            suggestions = []
            categorical_vars = [col for col, info in variable_analysis.items() 
                              if info['type'] == 'categorical' and info['unique_values'] > 1 and info['unique_values'] < n_samples/2]
            continuous_vars = [col for col, info in variable_analysis.items() 
                             if info['type'] == 'continuous' and info['missing_count'] < n_samples/2]
            
            # Identify potential main condition and batch variables
            main_condition = None
            batch_vars = []
            
            for col in categorical_vars:
                unique_count = variable_analysis[col]['unique_values']
                if unique_count == 2 and not main_condition:
                    main_condition = col
                elif col.lower() in ['batch', 'sample', 'donor', 'patient', 'subject']:
                    batch_vars.append(col)
                elif unique_count > 2 and unique_count <= 6:
                    batch_vars.append(col)
            
            if main_condition:
                # Simple comparison
                suggestions.append({
                    'formula': f'~{main_condition}',
                    'complexity': 'Simple',
                    'description': f'Compare {main_condition} groups directly',
                    'pros': ['Maximum statistical power', 'Straightforward interpretation', 'Robust with small sample sizes'],
                    'cons': ['Ignores potential confounders', 'May miss batch effects'],
                    'recommended_for': 'Initial exploratory analysis or when confounders are minimal',
                    'min_samples': 6
                })
                
                # Batch-corrected if batch variables available
                if batch_vars:
                    primary_batch = batch_vars[0]
                    suggestions.append({
                        'formula': f'~{main_condition} + {primary_batch}',
                        'complexity': 'Batch-corrected',
                        'description': f'Compare {main_condition} while accounting for {primary_batch} effects',
                        'pros': ['Controls for technical/batch variation', 'More reliable effect estimates'],
                        'cons': ['Reduces degrees of freedom', 'Requires balanced design'],
                        'recommended_for': 'Multi-batch experiments or when batch effects are suspected',
                        'min_samples': 8
                    })
                
                # Full model with multiple covariates
                if len(batch_vars) > 1 or continuous_vars:
                    covariates = batch_vars[:2] + continuous_vars[:1]  # Limit to avoid overfitting
                    formula_terms = [main_condition] + covariates
                    suggestions.append({
                        'formula': f'~{" + ".join(formula_terms)}',
                        'complexity': 'Multi-factor',
                        'description': f'Comprehensive model accounting for {main_condition} and {len(covariates)} covariates',
                        'pros': ['Controls for multiple confounders', 'Publication-ready analysis', 'Robust effect estimates'],
                        'cons': ['Requires larger sample size', 'More complex interpretation', 'Risk of overfitting'],
                        'recommended_for': 'Final analysis with adequate sample size and multiple known confounders',
                        'min_samples': max(12, len(formula_terms) * 3)
                    })
            
            # Build response
            response = f"📊 **Formula Design Analysis for '{pseudobulk_modality}'**\n\n"
            
            if show_metadata_summary:
                response += f"**Metadata Summary:**\n"
                response += f"• Samples: {n_samples}\n"
                response += f"• Variables analyzed: {len(variable_analysis)}\n"
                response += f"• Categorical variables: {len(categorical_vars)}\n"
                response += f"• Continuous variables: {len(continuous_vars)}\n\n"
                
                response += f"**Key Variables:**\n"
                for col, info in list(variable_analysis.items())[:6]:
                    if col in categorical_vars + continuous_vars:
                        response += f"• **{col}**: {info['type']}, {info['unique_values']} levels"
                        if info['type'] == 'categorical':
                            response += f" ({', '.join(map(str, info['sample_values']))})"
                        response += f"\n"
                response += "\n"
            
            if analysis_goal:
                response += f"**Analysis Goal**: {analysis_goal}\n\n"
            
            if suggestions:
                response += f"📝 **Recommended Formula Options:**\n\n"
                for i, suggestion in enumerate(suggestions, 1):
                    response += f"**{i}. {suggestion['complexity']} Model** *(recommended for {suggestion['recommended_for']})*\n"
                    response += f"   Formula: `{suggestion['formula']}`\n"
                    response += f"   Description: {suggestion['description']}\n"
                    response += f"   ✅ Pros: {', '.join(suggestion['pros'][:2])}\n"
                    response += f"   ⚠️ Cons: {', '.join(suggestion['cons'][:2])}\n"
                    response += f"   Min samples needed: {suggestion['min_samples']}\n\n"
                
                response += f"💡 **Recommendation**: Start with the simple model for exploration, then use the batch-corrected model if you see batch effects.\n\n"
                response += f"**Next step**: Use `construct_de_formula_interactive` to build and validate your chosen formula."
                
            else:
                response += f"⚠️ **No suitable variables found for standard DE analysis.**\n"
                response += f"Please ensure your pseudobulk data has:\n"
                response += f"• At least one categorical variable with 2+ levels (main condition)\n"
                response += f"• Sufficient samples per group (minimum 3-4 replicates)\n"
                response += f"• Proper metadata annotation\n\n"
                response += f"Available variables: {list(variable_analysis.keys())}"
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="suggest_formula_for_design",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "analysis_goal": analysis_goal,
                    "n_suggestions": len(suggestions)
                },
                description=f"Generated {len(suggestions)} formula suggestions for {pseudobulk_modality}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error suggesting formulas: {e}")
            return f"Error analyzing design for formula suggestions: {str(e)}"

    @tool
    def construct_de_formula_interactive(
        pseudobulk_modality: str,
        main_variable: str,
        covariates: Optional[List[str]] = None,
        include_interactions: bool = False,
        validate_design: bool = True
    ) -> str:
        """
        Build DE formula step-by-step with validation and preview.
        
        Constructs R-style formula, validates against metadata, shows design matrix preview,
        and provides warnings about potential statistical issues.
        
        Args:
            pseudobulk_modality: Name of pseudobulk modality
            main_variable: Primary variable of interest (main comparison)
            covariates: List of covariate variables to include
            include_interactions: Whether to include interaction terms
            validate_design: Whether to validate the experimental design
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"
            
            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)
            metadata = adata.obs
            
            # Validate main variable
            if main_variable not in metadata.columns:
                available_vars = [col for col in metadata.columns if not col.startswith('_')]
                return f"Main variable '{main_variable}' not found. Available: {available_vars}"
            
            # Build formula
            formula_terms = [main_variable]
            if covariates:
                # Validate covariates
                missing_covariates = [c for c in covariates if c not in metadata.columns]
                if missing_covariates:
                    return f"Covariates not found: {missing_covariates}"
                formula_terms.extend(covariates)
            
            # Construct basic formula
            if include_interactions and covariates:
                # Add interaction between main variable and first covariate
                interaction_term = f"{main_variable}*{covariates[0]}"
                formula = f"~{interaction_term}"
                if len(covariates) > 1:
                    formula += f" + {' + '.join(covariates[1:])}"
            else:
                formula = f"~{' + '.join(formula_terms)}"
            
            # Parse and validate formula using formula service
            try:
                formula_components = formula_service.parse_formula(formula, metadata)
                design_result = formula_service.construct_design_matrix(formula_components, metadata)
                
                # Format response
                response = f"📊 **Formula Construction Complete for '{pseudobulk_modality}'**\n\n"
                response += f"🔧 **Constructed Formula**: `{formula}`\n\n"
                
                response += f"**Formula Components:**\n"
                response += f"• Main variable: {main_variable} ({formula_components['variable_info'][main_variable]['type']})\n"
                if covariates:
                    response += f"• Covariates: {', '.join(covariates)}\n"
                if include_interactions:
                    response += f"• Interactions: Yes (between {main_variable} and {covariates[0] if covariates else 'none'})\n"
                response += f"• Total terms: {len(formula_components['predictor_terms'])}\n\n"
                
                # Design matrix preview
                response += f"📈 **Design Matrix Preview**:\n"
                response += f"• Dimensions: {design_result['design_matrix'].shape[0]} samples × {design_result['design_matrix'].shape[1]} coefficients\n"
                response += f"• Matrix rank: {design_result['rank']} (full rank: {'✓' if design_result['rank'] == design_result['n_coefficients'] else '⚠️'})\n"
                response += f"• Coefficient names: {', '.join(design_result['coefficient_names'][:5])}{'...' if len(design_result['coefficient_names']) > 5 else ''}\n\n"
                
                # Variable information
                response += f"**Variable Details:**\n"
                for var, info in formula_components['variable_info'].items():
                    if info['type'] == 'categorical':
                        response += f"• **{var}**: {info['n_levels']} levels, reference = '{info['reference_level']}'\n"
                    else:
                        response += f"• **{var}**: continuous variable\n"
                response += "\n"
                
                if validate_design:
                    # Validate experimental design
                    validation = formula_service.validate_experimental_design(
                        metadata, formula, min_replicates=2
                    )
                    
                    response += f"✅ **Design Validation**:\n"
                    response += f"• Valid design: {'✓' if validation['valid'] else '✗'}\n"
                    
                    if validation['warnings']:
                        response += f"• Warnings ({len(validation['warnings'])}):\n"
                        for warning in validation['warnings'][:3]:
                            response += f"  - {warning}\n"
                    
                    if validation['errors']:
                        response += f"• Errors ({len(validation['errors'])}):\n"
                        for error in validation['errors']:
                            response += f"  - {error}\n"
                    
                    response += f"\n**Sample Distribution:**\n"
                    for var, counts in validation.get('design_summary', {}).items():
                        response += f"• **{var}**: {dict(list(counts.items())[:4])}\n"
                
                response += f"\n💡 **Recommendations**:\n"
                if design_result['rank'] < design_result['n_coefficients']:
                    response += f"⚠️ Design matrix is rank deficient - consider removing correlated variables\n"
                if validation.get('warnings'):
                    response += f"⚠️ Review warnings above before proceeding\n"
                else:
                    response += f"✅ Design looks good! Ready for differential expression analysis\n"
                
                response += f"\n**Next step**: Use `run_differential_expression_with_formula` to execute the analysis."
                
                # Store formula in modality for later use
                adata.uns['constructed_formula'] = {
                    'formula': formula,
                    'main_variable': main_variable,
                    'covariates': covariates,
                    'include_interactions': include_interactions,
                    'formula_components': formula_components,
                    'design_result': design_result,
                    'validation': validation if validate_design else None
                }
                data_manager.modalities[pseudobulk_modality] = adata
                
            except (FormulaError, DesignMatrixError) as e:
                response = f"❌ **Formula Construction Failed**\n\n"
                response += f"**Formula**: `{formula}`\n"
                response += f"**Error**: {str(e)}\n\n"
                response += f"💡 **Suggestions**:\n"
                response += f"• Check variable names are spelled correctly\n"
                response += f"• Ensure variables have multiple levels (for categorical) or variation (for continuous)\n"
                response += f"• Reduce model complexity if you have limited samples\n"
                response += f"• Available variables: {list(metadata.columns)[:10]}"
                return response
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="construct_de_formula_interactive",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "formula": formula,
                    "main_variable": main_variable,
                    "covariates": covariates,
                    "include_interactions": include_interactions
                },
                description=f"Constructed and validated formula: {formula}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error constructing formula: {e}")
            return f"Error in formula construction: {str(e)}"

    @tool
    def run_differential_expression_with_formula(
        pseudobulk_modality: str,
        formula: Optional[str] = None,
        contrast: Optional[List[str]] = None,
        reference_levels: Optional[dict] = None,
        alpha: float = 0.05,
        lfc_threshold: float = 0.0,
        save_results: bool = True
    ) -> str:
        """
        Execute differential expression analysis with agent-guided formula.
        
        Uses pyDESeq2 for analysis, returns formatted results summary, and stores
        results as new modality for downstream analysis.
        
        Args:
            pseudobulk_modality: Name of pseudobulk modality
            formula: R-style formula (uses stored formula if None)
            contrast: Contrast specification [factor, level1, level2]
            reference_levels: Reference levels for categorical variables
            alpha: Significance threshold for adjusted p-values
            lfc_threshold: Log fold change threshold
            save_results: Whether to save results to files
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"
            
            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)
            
            # Use stored formula if none provided
            if formula is None:
                if 'constructed_formula' in adata.uns:
                    formula = adata.uns['constructed_formula']['formula']
                    stored_info = adata.uns['constructed_formula']
                    response_prefix = f"Using stored formula from interactive construction:\n"
                else:
                    return "No formula provided and no stored formula found. Use `construct_de_formula_interactive` first or provide a formula."
            else:
                response_prefix = f"Using provided formula:\n"
                stored_info = None
            
            # Auto-detect contrast if not provided
            if contrast is None and stored_info:
                main_var = stored_info['main_variable']
                levels = list(adata.obs[main_var].unique())
                if len(levels) == 2:
                    contrast = [main_var, str(levels[1]), str(levels[0])]  # Compare second vs first
                    response_prefix += f"Auto-detected contrast: {contrast[1]} vs {contrast[2]}\n"
                else:
                    return f"Multiple levels found for {main_var}: {levels}. Please specify contrast as [factor, level1, level2]."
            elif contrast is None:
                return "No contrast specified. Please provide contrast as [factor, level1, level2]."
            
            logger.info(f"Running DE analysis on '{pseudobulk_modality}' with formula: {formula}")
            
            # Prepare design matrix using bulk RNA-seq service
            design_validation = bulk_rnaseq_service.validate_experimental_design(
                metadata=adata.obs,
                formula=formula,
                min_replicates=2
            )
            
            if not design_validation['valid']:
                error_msgs = "; ".join(design_validation['errors'])
                return f"❌ **Invalid experimental design**: {error_msgs}\n\nUse `construct_de_formula_interactive` to debug the design."
            
            # Create design matrix
            condition_col = contrast[0]
            reference_condition = reference_levels.get(condition_col) if reference_levels else None
            
            design_result = bulk_rnaseq_service.create_formula_design(
                metadata=adata.obs,
                condition_col=condition_col,
                reference_condition=reference_condition
            )
            
            # Store design information
            adata.uns['de_formula_design'] = {
                'formula': formula,
                'contrast': contrast,
                'design_matrix_info': design_result,
                'validation_results': design_validation,
                'reference_levels': reference_levels
            }
            
            # Run pyDESeq2 analysis
            results_df, analysis_stats = bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                pseudobulk_adata=adata,
                formula=formula,
                contrast=contrast,
                alpha=alpha,
                shrink_lfc=True,
                n_cpus=1
            )
            
            # Filter by LFC threshold if specified
            if lfc_threshold > 0:
                significant_mask = (results_df['padj'] < alpha) & (abs(results_df['log2FoldChange']) >= lfc_threshold)
                n_lfc_filtered = significant_mask.sum()
            else:
                n_lfc_filtered = analysis_stats['n_significant_genes']
            
            # Store results in modality
            contrast_name = f"{contrast[0]}_{contrast[1]}_vs_{contrast[2]}"
            adata.uns[f'de_results_formula_{contrast_name}'] = {
                'results_df': results_df,
                'analysis_stats': analysis_stats,
                'parameters': {
                    'formula': formula,
                    'contrast': contrast,
                    'alpha': alpha,
                    'lfc_threshold': lfc_threshold,
                    'reference_levels': reference_levels
                }
            }
            
            # Update modality
            data_manager.modalities[pseudobulk_modality] = adata
            
            # Save results if requested
            if save_results:
                results_path = f"{pseudobulk_modality}_formula_de_results.csv"
                results_df.to_csv(results_path)
                
                modality_path = f"{pseudobulk_modality}_with_formula_results.h5ad"
                data_manager.save_modality(pseudobulk_modality, modality_path)
            
            # Format response
            response = f"🧬 **Differential Expression Analysis Complete**\n\n"
            response += response_prefix
            response += f"**Formula**: `{formula}`\n"
            response += f"**Contrast**: {contrast[1]} vs {contrast[2]} (in {contrast[0]})\n\n"
            
            response += f"📊 **Results Summary**:\n"
            response += f"• Genes tested: {analysis_stats['n_genes_tested']:,}\n"
            response += f"• Significant genes (FDR < {alpha}): {analysis_stats['n_significant_genes']:,}\n"
            if lfc_threshold > 0:
                response += f"• Significant + |LFC| ≥ {lfc_threshold}: {n_lfc_filtered:,}\n"
            response += f"• Upregulated ({contrast[1]} > {contrast[2]}): {analysis_stats['n_upregulated']:,}\n"
            response += f"• Downregulated ({contrast[1]} < {contrast[2]}): {analysis_stats['n_downregulated']:,}\n\n"
            
            response += f"🏆 **Top Differentially Expressed Genes**:\n"
            response += f"**Most Upregulated**:\n"
            for gene in analysis_stats['top_upregulated'][:5]:
                gene_data = results_df.loc[gene]
                response += f"• {gene}: LFC = {gene_data['log2FoldChange']:.2f}, FDR = {gene_data['padj']:.2e}\n"
            
            response += f"\n**Most Downregulated**:\n"
            for gene in analysis_stats['top_downregulated'][:5]:
                gene_data = results_df.loc[gene]
                response += f"• {gene}: LFC = {gene_data['log2FoldChange']:.2f}, FDR = {gene_data['padj']:.2e}\n"
            
            response += f"\n📈 **Experimental Design**:\n"
            response += f"• Samples: {design_result['design_matrix'].shape[0]}\n"
            response += f"• Coefficients: {design_result['design_matrix'].shape[1]}\n"
            response += f"• Design rank: {design_result['rank']} (full rank: {'✓' if design_result['rank'] == design_result['n_coefficients'] else '⚠️'})\n"
            
            if design_validation['warnings']:
                response += f"\n⚠️ **Design Warnings**: {'; '.join(design_validation['warnings'][:2])}\n"
            
            response += f"\n💾 **Results Storage**:\n"
            response += f"• Stored in: adata.uns['de_results_formula_{contrast_name}']\n"
            if save_results:
                response += f"• CSV file: {results_path}\n"
                response += f"• H5AD file: {modality_path}\n"
            
            response += f"\n**Next steps**: Use `iterate_de_analysis` to try different formulas or `compare_de_iterations` to compare results."
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="run_differential_expression_with_formula",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "formula": formula,
                    "contrast": contrast,
                    "alpha": alpha,
                    "lfc_threshold": lfc_threshold
                },
                description=f"Formula-based DE analysis: {analysis_stats['n_significant_genes']} significant genes"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in formula-based DE analysis: {e}")
            return f"Error running differential expression with formula: {str(e)}"

    @tool
    def iterate_de_analysis(
        pseudobulk_modality: str,
        new_formula: Optional[str] = None,
        new_contrast: Optional[List[str]] = None,
        filter_criteria: Optional[dict] = None,
        compare_to_previous: bool = True,
        iteration_name: Optional[str] = None
    ) -> str:
        """
        Support iterative analysis with formula/filter changes.
        
        Enables Step 12 of workflow - trying different formulas or filters, tracking
        iterations, and comparing results between analyses.
        
        Args:
            pseudobulk_modality: Name of pseudobulk modality
            new_formula: New formula to try (if None, modify existing)
            new_contrast: New contrast to test
            filter_criteria: Additional filtering criteria (e.g., {'min_lfc': 0.5})
            compare_to_previous: Whether to compare with previous iteration
            iteration_name: Custom name for this iteration
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"
            
            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)
            
            # Initialize iteration tracking if not exists
            if 'de_iterations' not in adata.uns:
                adata.uns['de_iterations'] = {
                    'iterations': [],
                    'current_iteration': 0
                }
            
            iteration_tracker = adata.uns['de_iterations']
            current_iter = iteration_tracker['current_iteration'] + 1
            
            # Determine iteration name
            if iteration_name is None:
                iteration_name = f"iteration_{current_iter}"
            
            # Get previous results for comparison
            previous_results = None
            previous_iteration = None
            if compare_to_previous and iteration_tracker['iterations']:
                previous_iteration = iteration_tracker['iterations'][-1]
                prev_key = f"de_results_formula_{previous_iteration['contrast_name']}"
                if prev_key in adata.uns:
                    previous_results = adata.uns[prev_key]['results_df']
            
            # Use existing formula/contrast if not provided
            if new_formula is None or new_contrast is None:
                if 'de_formula_design' in adata.uns:
                    if new_formula is None:
                        new_formula = adata.uns['de_formula_design']['formula']
                    if new_contrast is None:
                        new_contrast = adata.uns['de_formula_design']['contrast']
                else:
                    return "No previous formula/contrast found and none provided. Run a DE analysis first."
            
            logger.info(f"Starting DE iteration '{iteration_name}' on '{pseudobulk_modality}'")
            
            # Run the analysis with new parameters
            run_result = run_differential_expression_with_formula(
                pseudobulk_modality=pseudobulk_modality,
                formula=new_formula,
                contrast=new_contrast,
                alpha=0.05,
                lfc_threshold=filter_criteria.get('min_lfc', 0.0) if filter_criteria else 0.0,
                save_results=False  # Don't save individual iterations
            )
            
            if "Error" in run_result:
                return f"Error in iteration '{iteration_name}': {run_result}"
            
            # Get current results
            contrast_name = f"{new_contrast[0]}_{new_contrast[1]}_vs_{new_contrast[2]}"
            current_key = f"de_results_formula_{contrast_name}"
            
            if current_key not in adata.uns:
                return f"Results not found after analysis. Analysis may have failed."
            
            current_results = adata.uns[current_key]['results_df']
            current_stats = adata.uns[current_key]['analysis_stats']
            
            # Store iteration information
            iteration_info = {
                'name': iteration_name,
                'formula': new_formula,
                'contrast': new_contrast,
                'contrast_name': contrast_name,
                'n_significant': current_stats['n_significant_genes'],
                'timestamp': pd.Timestamp.now().isoformat(),
                'filter_criteria': filter_criteria or {}
            }
            
            # Compare with previous if requested
            comparison_results = None
            if compare_to_previous and previous_results is not None:
                # Calculate overlap
                current_sig = set(current_results[current_results['padj'] < 0.05].index)
                previous_sig = set(previous_results[previous_results['padj'] < 0.05].index)
                
                overlap = len(current_sig & previous_sig)
                current_only = len(current_sig - previous_sig)
                previous_only = len(previous_sig - current_sig)
                
                # Calculate correlation of fold changes for overlapping genes
                common_genes = list(current_sig & previous_sig)
                if len(common_genes) > 3:
                    current_lfc = current_results.loc[common_genes, 'log2FoldChange']
                    previous_lfc = previous_results.loc[common_genes, 'log2FoldChange']
                    correlation = current_lfc.corr(previous_lfc)
                else:
                    correlation = None
                
                comparison_results = {
                    'overlap_genes': overlap,
                    'current_only': current_only,
                    'previous_only': previous_only,
                    'correlation': correlation
                }
                
                iteration_info['comparison'] = comparison_results
            
            # Update iteration tracking
            iteration_tracker['iterations'].append(iteration_info)
            iteration_tracker['current_iteration'] = current_iter
            adata.uns['de_iterations'] = iteration_tracker
            
            # Update modality
            data_manager.modalities[pseudobulk_modality] = adata
            
            # Format response
            response = f"🔄 **DE Analysis Iteration '{iteration_name}' Complete**\n\n"
            response += f"**Formula**: `{new_formula}`\n"
            response += f"**Contrast**: {new_contrast[1]} vs {new_contrast[2]} (in {new_contrast[0]})\n\n"
            
            response += f"📊 **Current Results**:\n"
            response += f"• Significant genes: {current_stats['n_significant_genes']:,}\n"
            response += f"• Upregulated: {current_stats['n_upregulated']:,}\n"
            response += f"• Downregulated: {current_stats['n_downregulated']:,}\n"
            
            if comparison_results:
                response += f"\n🔄 **Comparison with Previous Iteration**:\n"
                response += f"• Overlapping significant genes: {comparison_results['overlap_genes']:,}\n"
                response += f"• New in current: {comparison_results['current_only']:,}\n"
                response += f"• Lost from previous: {comparison_results['previous_only']:,}\n"
                if comparison_results['correlation'] is not None:
                    response += f"• Fold change correlation: {comparison_results['correlation']:.3f}\n"
            
            response += f"\n📈 **Iteration Summary**:\n"
            response += f"• Total iterations: {len(iteration_tracker['iterations'])}\n"
            response += f"• Current iteration: {current_iter}\n"
            
            response += f"\n💾 **Results stored in**: adata.uns['de_results_formula_{contrast_name}']\n"
            response += f"💾 **Iteration tracking**: adata.uns['de_iterations']\n"
            
            response += f"\n**Next steps**: Use `compare_de_iterations` to compare all iterations or continue iterating with different parameters."
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="iterate_de_analysis",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "iteration_name": iteration_name,
                    "formula": new_formula,
                    "contrast": new_contrast,
                    "compare_to_previous": compare_to_previous
                },
                description=f"DE iteration {current_iter}: {current_stats['n_significant_genes']} significant genes"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in DE iteration: {e}")
            return f"Error in DE analysis iteration: {str(e)}"

    @tool
    def compare_de_iterations(
        pseudobulk_modality: str,
        iteration_1: Optional[str] = None,
        iteration_2: Optional[str] = None,
        show_overlap: bool = True,
        show_unique: bool = True,
        save_comparison: bool = True
    ) -> str:
        """
        Compare results between different DE analysis iterations.
        
        Shows overlapping and unique DEGs, correlation of fold changes, and helps
        users understand impact of formula changes.
        
        Args:
            pseudobulk_modality: Name of pseudobulk modality
            iteration_1: Name of first iteration (latest if None)
            iteration_2: Name of second iteration (second latest if None) 
            show_overlap: Whether to show overlapping genes
            show_unique: Whether to show unique genes per iteration
            save_comparison: Whether to save comparison results
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"
            
            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)
            
            # Check iteration tracking exists
            if 'de_iterations' not in adata.uns:
                return "No iteration tracking found. Run `iterate_de_analysis` first to create iterations."
            
            iteration_tracker = adata.uns['de_iterations']
            iterations = iteration_tracker['iterations']
            
            if len(iterations) < 2:
                return f"Only {len(iterations)} iteration(s) available. Need at least 2 for comparison."
            
            # Select iterations to compare
            if iteration_1 is None:
                iter1_info = iterations[-1]  # Latest
            else:
                iter1_info = next((i for i in iterations if i['name'] == iteration_1), None)
                if not iter1_info:
                    available = [i['name'] for i in iterations]
                    return f"Iteration '{iteration_1}' not found. Available: {available}"
            
            if iteration_2 is None:
                iter2_info = iterations[-2] if len(iterations) >= 2 else iterations[0]  # Second latest
            else:
                iter2_info = next((i for i in iterations if i['name'] == iteration_2), None)
                if not iter2_info:
                    available = [i['name'] for i in iterations]
                    return f"Iteration '{iteration_2}' not found. Available: {available}"
            
            # Get results DataFrames
            iter1_key = f"de_results_formula_{iter1_info['contrast_name']}"
            iter2_key = f"de_results_formula_{iter2_info['contrast_name']}"
            
            if iter1_key not in adata.uns or iter2_key not in adata.uns:
                return f"Results not found for one or both iterations. Missing keys: {[k for k in [iter1_key, iter2_key] if k not in adata.uns]}"
            
            results1 = adata.uns[iter1_key]['results_df']
            results2 = adata.uns[iter2_key]['results_df']
            
            # Get significant genes (FDR < 0.05)
            sig1 = set(results1[results1['padj'] < 0.05].index)
            sig2 = set(results2[results2['padj'] < 0.05].index)
            
            # Calculate overlaps
            overlap = sig1 & sig2
            unique1 = sig1 - sig2
            unique2 = sig2 - sig1
            
            # Calculate fold change correlation for overlapping genes
            if len(overlap) > 3:
                overlap_genes = list(overlap)
                lfc1 = results1.loc[overlap_genes, 'log2FoldChange']
                lfc2 = results2.loc[overlap_genes, 'log2FoldChange']
                correlation = lfc1.corr(lfc2)
            else:
                correlation = None
            
            # Format response
            response = f"📊 **DE Iteration Comparison**\n\n"
            response += f"**Comparing:**\n"
            response += f"• Iteration 1: '{iter1_info['name']}' - {iter1_info['formula']}\n"
            response += f"• Iteration 2: '{iter2_info['name']}' - {iter2_info['formula']}\n\n"
            
            response += f"📈 **Results Summary:**\n"
            response += f"• Iteration 1 significant genes: {len(sig1):,}\n"
            response += f"• Iteration 2 significant genes: {len(sig2):,}\n"
            response += f"• Overlapping genes: {len(overlap):,} ({len(overlap)/max(len(sig1), len(sig2))*100:.1f}%)\n"
            response += f"• Unique to iteration 1: {len(unique1):,}\n"
            response += f"• Unique to iteration 2: {len(unique2):,}\n"
            
            if correlation is not None:
                response += f"• Fold change correlation: {correlation:.3f}\n"
            
            if show_overlap and len(overlap) > 0:
                response += f"\n🔗 **Top Overlapping Genes:**\n"
                # Get top overlapping genes by average absolute fold change
                overlap_df = results1.loc[list(overlap)]
                overlap_df = overlap_df.reindex(overlap_df['padj'].sort_values().index)
                
                for gene in list(overlap_df.index)[:10]:
                    lfc1 = results1.loc[gene, 'log2FoldChange']
                    lfc2 = results2.loc[gene, 'log2FoldChange']
                    response += f"• {gene}: LFC1={lfc1:.2f}, LFC2={lfc2:.2f}\n"
            
            if show_unique and (len(unique1) > 0 or len(unique2) > 0):
                response += f"\n🎯 **Unique Significant Genes:**\n"
                
                if len(unique1) > 0:
                    response += f"**Only in '{iter1_info['name']}'** ({len(unique1)} genes):\n"
                    unique1_sorted = results1.loc[list(unique1)].sort_values('padj')
                    for gene in unique1_sorted.index[:8]:
                        lfc = results1.loc[gene, 'log2FoldChange']
                        fdr = results1.loc[gene, 'padj']
                        response += f"• {gene}: LFC={lfc:.2f}, FDR={fdr:.2e}\n"
                
                if len(unique2) > 0:
                    response += f"\n**Only in '{iter2_info['name']}'** ({len(unique2)} genes):\n"
                    unique2_sorted = results2.loc[list(unique2)].sort_values('padj')
                    for gene in unique2_sorted.index[:8]:
                        lfc = results2.loc[gene, 'log2FoldChange']
                        fdr = results2.loc[gene, 'padj']
                        response += f"• {gene}: LFC={lfc:.2f}, FDR={fdr:.2e}\n"
            
            # Analysis interpretation
            response += f"\n💡 **Interpretation:**\n"
            if correlation is not None:
                if correlation > 0.8:
                    response += f"• High correlation ({correlation:.3f}) suggests similar biological effects\n"
                elif correlation > 0.5:
                    response += f"• Moderate correlation ({correlation:.3f}) - some consistency but notable differences\n"
                else:
                    response += f"• Low correlation ({correlation:.3f}) - formulas capture different effects\n"
            
            overlap_percent = len(overlap) / max(len(sig1), len(sig2)) * 100
            if overlap_percent > 70:
                response += f"• High overlap ({overlap_percent:.1f}%) - formulas yield similar gene sets\n"
            elif overlap_percent > 40:
                response += f"• Moderate overlap ({overlap_percent:.1f}%) - some formula-specific effects\n"
            else:
                response += f"• Low overlap ({overlap_percent:.1f}%) - formulas capture different biology\n"
            
            # Save comparison if requested
            if save_comparison:
                comparison_data = {
                    'iteration_1': iter1_info,
                    'iteration_2': iter2_info,
                    'overlap_genes': list(overlap),
                    'unique_to_1': list(unique1),
                    'unique_to_2': list(unique2),
                    'correlation': correlation,
                    'summary_stats': {
                        'n_sig_1': len(sig1),
                        'n_sig_2': len(sig2),
                        'n_overlap': len(overlap),
                        'overlap_percent': overlap_percent
                    }
                }
                
                # Store in modality
                comparison_key = f"iteration_comparison_{iter1_info['name']}_vs_{iter2_info['name']}"
                if 'iteration_comparisons' not in adata.uns:
                    adata.uns['iteration_comparisons'] = {}
                adata.uns['iteration_comparisons'][comparison_key] = comparison_data
                
                data_manager.modalities[pseudobulk_modality] = adata
                response += f"\n💾 **Comparison saved**: adata.uns['iteration_comparisons']['{comparison_key}']\n"
            
            response += f"\n**Next steps**: Choose the most appropriate formula based on biological interpretation and statistical robustness."
            
            # Log the operation
            data_manager.log_tool_usage(
                tool_name="compare_de_iterations",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "iteration_1": iter1_info['name'],
                    "iteration_2": iter2_info['name'],
                    "show_overlap": show_overlap,
                    "show_unique": show_unique
                },
                description=f"Compared iterations: {len(overlap)} overlapping, {len(unique1)}+{len(unique2)} unique genes"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error comparing DE iterations: {e}")
            return f"Error comparing DE iterations: {str(e)}"

    # -------------------------
    # DEEP LEARNING EMBEDDING TOOLS (scVI Integration)
    # -------------------------
    # -------------------------
    # FUTURE FEATURE: Direct Sub-agent Handoff (Currently Disabled)
    # -------------------------
    # The code below is commented out to maintain supervisor-mediated flow.
    # In the future, this could enable direct SingleCell → ML Expert handoffs.
    # For now, we return a message to the supervisor for delegation.
    
    # @tool
    # def request_scvi_embedding_direct_handoff(
    #     modality_name: str,
    #     state: Annotated[dict, InjectedState],
    #     tool_call_id: Annotated[str, InjectedToolCallId],
    #     n_latent: int = 10,
    #     batch_key: Optional[str] = None,
    #     max_epochs: int = 400,
    #     use_gpu: bool = False,
    # ) -> Command:
    #     """
    #     FUTURE: Direct handoff to ML Expert for scVI embeddings.
    #     Currently disabled - use request_scvi_embedding() instead.
    #     """
    #     # [Original direct handoff code preserved for future use]
    #     pass
    
#     @tool
#     def request_scvi_embedding(
#         modality_name: str,
#         n_latent: int = 10,
#         batch_key: Optional[str] = None,
#         max_epochs: int = 400,
#         use_gpu: bool = False,
#     ) -> str:
#         """
#         Request ML Expert to train scVI embeddings for deep learning-based dimensionality reduction.

#         This generates a request message for the supervisor to delegate scVI training to the ML Expert.
#         The supervisor will coordinate the handoff and ensure proper workflow management.

#         Args:
#             modality_name: Name of the single-cell modality to process
#             n_latent: Number of latent dimensions for the embedding (default: 10)
#             batch_key: Column name for batch correction (optional, auto-detect if None)
#             max_epochs: Maximum training epochs for the neural network (default: 400)
#             use_gpu: Whether to use GPU acceleration if available (default: False for stability)

#         Returns:
#             str: Request message for supervisor to delegate to ML Expert
#         """
#         try:
#             # Validate modality exists
#             if modality_name not in data_manager.list_modalities():
#                 available = data_manager.list_modalities()
#                 return f"❌ Modality '{modality_name}' not found.\n\n📊 Available modalities: {', '.join(available)}\n\nPlease specify a valid single-cell modality name."
            
#             # Get the modality for validation
#             adata = data_manager.get_modality(modality_name)
            
#             # Validate it's single-cell data (reasonable size and structure)
#             if adata.n_obs < 100:
#                 return f"❌ Modality '{modality_name}' has only {adata.n_obs} observations. scVI requires at least 100 cells for meaningful training."

#             if adata.n_vars < 500:
#                 return f"❌ Modality '{modality_name}' has only {adata.n_vars} features. scVI works best with at least 500 genes."
            
#             # Auto-detect batch key if not provided
#             detected_batch_key = None
#             if batch_key is None:
#                 # Look for common batch column names
#                 batch_candidates = [col for col in adata.obs.columns 
#                                   if any(keyword in col.lower() for keyword in ['batch', 'sample', 'donor', 'replicate', 'patient'])]
#                 if batch_candidates:
#                     # Choose the one with reasonable number of categories
#                     for candidate in batch_candidates:
#                         n_categories = adata.obs[candidate].nunique()
#                         if 2 <= n_categories <= adata.n_obs // 10:  # Between 2 categories and 10% of cells
#                             detected_batch_key = candidate
#                             break
            
#             # Validate provided batch key
#             if batch_key and batch_key not in adata.obs.columns:
#                 available_batch = [col for col in adata.obs.columns
#                                  if any(keyword in col.lower() for keyword in ['batch', 'sample', 'donor', 'replicate'])]
#                 return f"❌ Batch key '{batch_key}' not found in modality observations.\n\n📋 Available batch-related columns: {available_batch}"
            
#             # Use provided batch_key or detected one
#             final_batch_key = batch_key or detected_batch_key
            
#             # Log the operation
#             data_manager.log_tool_usage(
#                 tool_name="request_scvi_embedding",
#                 parameters={
#                     "modality_name": modality_name,
#                     "n_latent": n_latent,
#                     "batch_key": final_batch_key,
#                     "max_epochs": max_epochs,
#                     "use_gpu": use_gpu
#                 },
#                 description=f"Requested scVI embedding training for {modality_name} with {n_latent} latent dimensions"
#             )

#             # Store analysis context for tracking
#             analysis_results["details"]["scvi_embedding_request"] = {
#                 "modality_name": modality_name,
#                 "n_latent": n_latent,
#                 "batch_key": final_batch_key,
#                 "use_gpu": use_gpu,
#                 "requested_at": pd.Timestamp.now().isoformat(),
#                 "data_shape": (adata.n_obs, adata.n_vars)
#             }

#             # Create request message for supervisor to delegate to ML Expert
#             request_message = f"""🧠 **scVI Embedding Request for Single-Cell Data**

# I need the Machine Learning Expert to train scVI embeddings for deep learning-based dimensionality reduction.

# 📊 **Dataset Details:**
# - Modality: '{modality_name}'
# - Data shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes
# - Suitable for scVI: ✓ (meets size requirements)

# 🔧 **Training Parameters:**
# - Latent dimensions: {n_latent}
# - Batch correction: {'✓ (using ' + final_batch_key + ')' if final_batch_key else '✗ (not needed)'}
# - Max epochs: {max_epochs}
# - GPU acceleration: {'✓ Requested' if use_gpu else '✗ CPU only'}

# 📋 **Expected Outcome:**
# The ML Expert should:
# 1. Check scVI availability and install if needed
# 2. Train scVI model on modality '{modality_name}'
# 3. Store embeddings in adata.obsm['X_scvi']
# 4. Report back when training is complete

# 🎯 **Purpose:**
# This will enable:
# - State-of-the-art dimensionality reduction
# - Better batch effect correction
# - Improved clustering and cell type identification
# - Handling of technical variation in multi-sample data

# **Supervisor Action Required:** Please delegate this scVI training task to the Machine Learning Expert with the parameters specified above."""

#             return request_message
            
#         except Exception as e:
#             logger.error(f"Error in scVI embedding request: {e}")
#             return f"❌ Error requesting scVI embedding: {str(e)}\n\nPlease ensure the modality exists and contains valid single-cell RNA-seq data."

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
        # Deep learning embedding tools
        # request_scvi_embedding,
        # Manual annotation tools
        manually_annotate_clusters_interactive,
        manually_annotate_clusters,
        collapse_clusters_to_celltype,
        mark_clusters_as_debris,
        suggest_debris_clusters,
        review_annotation_assignments,
        apply_annotation_template,
        export_annotation_mapping,
        import_annotation_mapping,
        # Pseudobulk analysis tools
        create_pseudobulk_matrix,
        prepare_differential_expression_design,
        run_pseudobulk_differential_expression,
        # Agent-guided formula construction tools
        suggest_formula_for_design,
        construct_de_formula_interactive,
        run_differential_expression_with_formula,
        iterate_de_analysis,
        compare_de_iterations,
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

## 4. DEEP LEARNING EMBEDDING WORKFLOWS (scVI Integration)

### Deep Learning Embedding Request (Supervisor Request: "Train scVI embedding" or "Use deep learning")
```bash
# Step 1: Verify data is suitable for scVI training
check_data_status("geo_gse12345_filtered_normalized")

# Step 2: Request scVI embedding via supervisor message (NO direct handoff)
Tell the supervisor that this is a task for the ML Expert to handle.

# Step 3: Report request message to supervisor
# The supervisor will then delegate the scVI training task to the ML Expert

# Step 4: Wait for supervisor to coordinate scVI training completion
# After ML Expert completes training, supervisor will direct next steps

# Step 5: Use scVI embeddings for clustering (when supervisor directs)
cluster_modality("geo_gse12345_filtered_normalized", use_rep="X_scvi", resolution=0.5)

# Step 6: Continue with standard workflow (when supervisor directs)
find_marker_genes_for_clusters("geo_gse12345_filtered_normalized_clustered", groupby="leiden")
```

### When to Request scVI Embeddings (Deep Learning Decision Points)
- **Large datasets (>5,000 cells)**: scVI provides superior dimensionality reduction
- **Multi-batch data**: scVI offers state-of-the-art batch correction
- **Complex cell type discovery**: Better separation of rare cell populations
- **Publication-quality analysis**: scVI is the current gold standard for single-cell embeddings

**Important:** Always request scVI through supervisor - never directly handoff to ML Expert.

## 5. COMPREHENSIVE ANALYSIS WORKFLOWS

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

# Step 5: Clustering (Traditional approach)
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
