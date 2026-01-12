"""
Transcriptomics Expert Parent Agent for orchestrating single-cell and bulk RNA-seq analysis.

This agent serves as the main orchestrator for transcriptomics analysis, with:
- Shared QC tools (from shared_tools.py) available directly
- Clustering tools (SC-specific) available directly
- Delegation to annotation_expert for cell type annotation
- Delegation to de_analysis_expert for differential expression analysis

The agent auto-detects single-cell vs bulk data and adapts its behavior accordingly.
"""

from datetime import date
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.transcriptomics.prompts import create_transcriptomics_expert_prompt
from lobster.agents.transcriptomics.shared_tools import create_shared_tools
from lobster.agents.transcriptomics.state import TranscriptomicsExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.analysis.clustering_service import (
    ClusteringError,
    ClusteringService,
)
from lobster.services.analysis.enhanced_singlecell_service import (
    EnhancedSingleCellService,
)
from lobster.services.analysis.enhanced_singlecell_service import (
    SingleCellError as ServiceSingleCellError,
)
from lobster.services.quality.preprocessing_service import PreprocessingService
from lobster.services.quality.quality_service import QualityService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class TranscriptomicsAgentError(Exception):
    """Base exception for transcriptomics agent operations."""

    pass


class ModalityNotFoundError(TranscriptomicsAgentError):
    """Raised when requested modality doesn't exist."""

    pass


def transcriptomics_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "transcriptomics_expert",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
):
    """
    Factory function for transcriptomics expert parent agent.

    This agent orchestrates single-cell and bulk RNA-seq analysis.
    It has QC and clustering tools directly, and delegates annotation
    and DE analysis to specialized sub-agents.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback handler for LLM interactions
        agent_name: Name identifier for the agent instance
        delegation_tools: List of delegation tools for sub-agents (annotation_expert, de_analysis_expert)

    Returns:
        Configured ReAct agent with transcriptomics analysis capabilities
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("transcriptomics_expert")
    llm = create_llm("transcriptomics_expert", model_params, workspace_path=workspace_path)

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        llm = llm.with_config(callbacks=callbacks)

    # Initialize services
    quality_service = QualityService()
    preprocessing_service = PreprocessingService()
    clustering_service = ClusteringService()
    enhanced_service = EnhancedSingleCellService()

    # Get shared tools (QC, preprocessing, analysis summary)
    shared_tools = create_shared_tools(
        data_manager, quality_service, preprocessing_service
    )

    # Analysis results storage (for clustering tools)
    analysis_results = {"summary": "", "details": {}}

    # =========================================================================
    # CLUSTERING TOOLS (copied from singlecell_expert.py)
    # =========================================================================

    @tool
    def cluster_modality(
        modality_name: str,
        resolution: float = None,
        resolutions: Optional[List[float]] = None,
        use_rep: Optional[str] = None,
        batch_correction: bool = True,
        batch_key: str = None,
        demo_mode: bool = False,
        save_result: bool = True,
        feature_selection_method: str = "deviance",
        n_features: int = 4000,
    ) -> str:
        """
        Perform single-cell clustering and UMAP visualization.

        Args:
            modality_name: Name of the single-cell modality to cluster
            resolution: Single Leiden clustering resolution (0.1-2.0, higher = more clusters).
                       Use this for single-resolution clustering. Default: 1.0 if neither resolution nor resolutions specified.
            resolutions: List of resolutions for multi-resolution testing (e.g., [0.25, 0.5, 1.0]).
                        Creates multiple clustering results with descriptive keys (leiden_res0_25, leiden_res0_5, leiden_res1_0).
                        Use this to explore clustering granularity. If specified, overrides 'resolution' parameter.
            use_rep: Representation to use for clustering (e.g., 'X_scvi' for deep learning embeddings, 'X_pca' for PCA).
                    If None, uses standard PCA workflow. Custom embeddings like scVI often provide better results.
            batch_correction: Whether to perform batch correction for multi-sample data
            batch_key: Column name for batch information (auto-detected if None)
            demo_mode: Use faster processing for large single-cell datasets (>50k cells)
            save_result: Whether to save the clustered modality
            feature_selection_method: Method for feature selection ('deviance' or 'hvg').
                                     'deviance' (default, recommended): Binomial deviance from multinomial null, works on raw counts, no normalization bias.
                                     'hvg': Traditional highly variable genes (Seurat method), works on normalized data.
            n_features: Number of features to select (default: 4000)

        Returns:
            str: Formatted report with clustering results and cluster distribution

        Examples:
            # Single resolution clustering (traditional)
            cluster_modality("geo_gse12345_filtered", resolution=0.5)

            # Multi-resolution testing (recommended for exploration)
            cluster_modality("geo_gse12345_filtered", resolutions=[0.25, 0.5, 1.0])

            # Using scVI embeddings
            cluster_modality("geo_gse12345_filtered", resolutions=[0.5, 1.0], use_rep="X_scvi")
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Clustering single-cell modality '{modality_name}': {adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Use clustering service
            adata_clustered, clustering_stats, ir = (
                clustering_service.cluster_and_visualize(
                    adata=adata,
                    resolution=resolution,
                    resolutions=resolutions,
                    use_rep=use_rep,
                    batch_correction=batch_correction,
                    batch_key=batch_key,
                    demo_mode=demo_mode,
                    feature_selection_method=feature_selection_method,
                    n_features=n_features,
                )
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
                    "resolutions": resolutions,
                    "batch_correction": batch_correction,
                    "demo_mode": demo_mode,
                    "feature_selection_method": feature_selection_method,
                    "n_features": n_features,
                },
                description=f"Single-cell clustered {modality_name} into {clustering_stats['n_clusters']} clusters using {feature_selection_method} feature selection",
                ir=ir,
            )

            # Format professional response
            response = f"""Successfully clustered single-cell modality '{modality_name}'!

**Single-cell Clustering Results:**"""

            # Check if multi-resolution testing was performed
            if clustering_stats.get("n_resolutions", 1) > 1:
                response += (
                    f"\n- Resolutions tested: {clustering_stats['resolutions_tested']}"
                )
                response += f"\n- Cluster columns (use these exact names for visualization):"
                for res, n_clusters in clustering_stats.get(
                    "multi_resolution_summary", {}
                ).items():
                    key_name = f"leiden_res{res}".replace(".", "_")
                    response += f"\n  - `{key_name}` (resolution={res}): {n_clusters} clusters"
            else:
                # Single resolution mode (existing behavior)
                response += f"\n- Number of clusters: {clustering_stats['n_clusters']}"
                response += f"\n- Leiden resolution: {clustering_stats.get('resolution', 'N/A')}"
                response += f"\n- Cluster column name: `leiden` (use this exact name for visualization)"

            # Continue with common details
            response += f"\n- UMAP coordinates: {'Yes' if clustering_stats['has_umap'] else 'No'}"
            response += f"\n- Marker genes: {'Yes' if clustering_stats['has_marker_genes'] else 'No'}"

            response += f"""

**Processing Details:**
- Original shape: {clustering_stats['original_shape'][0]} x {clustering_stats['original_shape'][1]}
- Final shape: {clustering_stats['final_shape'][0]} x {clustering_stats['final_shape'][1]}
- Feature selection: {feature_selection_method} ({n_features} features)
- Batch correction: {'Yes' if clustering_stats['batch_correction'] else 'No'}
- Demo mode: {'Yes' if clustering_stats['demo_mode'] else 'No'}

**Cluster Distribution:**"""

            # Add cluster size information
            for cluster_id, size in list(clustering_stats["cluster_sizes"].items())[:8]:
                percentage = (size / clustering_stats["final_shape"][0]) * 100
                response += (
                    f"\n- Cluster {cluster_id}: {size} cells ({percentage:.1f}%)"
                )

            if len(clustering_stats["cluster_sizes"]) > 8:
                response += f"\n... and {len(clustering_stats['cluster_sizes']) - 8} more clusters"

            response += f"\n\n**New modality created**: '{clustered_modality_name}'"

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            # Add multi-resolution guidance if applicable
            if clustering_stats.get("n_resolutions", 1) > 1:
                response += "\n\n**Multi-Resolution Analysis:**"
                response += "\n- Compare clustering results across resolutions using visualization"
                response += (
                    "\n- Lower resolutions (0.25-0.5) identify major cell populations"
                )
                response += "\n- Higher resolutions (1.0-2.0) reveal finer cell states"
                response += "\n- Choose resolution based on biological expectations and marker gene validation"

            response += "\n\n**Next steps**: find_marker_genes_for_clusters(), then INVOKE handoff_to_annotation_expert if annotation requested."

            analysis_results["details"]["clustering"] = response
            return response

        except (ClusteringError, ModalityNotFoundError) as e:
            logger.error(f"Error in single-cell clustering: {e}")
            return f"Error clustering single-cell modality: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in single-cell clustering: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def subcluster_cells(
        modality_name: str,
        cluster_key: str = "leiden",
        clusters_to_refine: Optional[List[str]] = None,
        resolution: float = 0.5,
        resolutions: Optional[List[float]] = None,
        n_pcs: int = 20,
        n_neighbors: int = 15,
        demo_mode: bool = False,
    ) -> str:
        """
        Re-cluster specific cell subsets for finer-grained population identification.

        Useful when initial clustering groups heterogeneous populations and you want
        to refine specific clusters without affecting others. Common scenarios:
        - "Split cluster 0 into subtypes"
        - "Refine the T cell clusters"
        - "Sub-cluster the heterogeneous populations"

        IMPORTANT: This tool requires that initial clustering has already been performed
        (i.e., the modality has a leiden column or specified cluster_key).

        Args:
            modality_name: Name of the modality to sub-cluster
            cluster_key: Key in adata.obs containing cluster assignments (default: "leiden")
            clusters_to_refine: List of cluster IDs to re-cluster (e.g., ["0", "3", "5"])
                               If None, re-clusters ALL cells (full re-clustering)
            resolution: Single resolution for sub-clustering (default: 0.5)
                       Typical range: 0.1-2.0 (higher = more clusters)
            resolutions: List of resolutions for multi-resolution testing (e.g., [0.25, 0.5, 1.0])
                        Creates multiple sub-clustering results with descriptive keys
                        Use for exploring different granularities
            n_pcs: Number of PCs for sub-clustering (default: 20, fewer than full clustering)
                   Typical range: 15-30
            n_neighbors: Number of neighbors for KNN graph (default: 15)
                        Typical range: 10-30
            demo_mode: Use faster parameters for testing (default: False)

        Returns:
            str: Summary of sub-clustering results including cluster sizes and new column names

        Examples:
            # Sub-cluster a single cluster
            subcluster_cells("geo_gse12345_clustered", clusters_to_refine=["0"])

            # Sub-cluster multiple clusters
            subcluster_cells("geo_gse12345_clustered", clusters_to_refine=["0", "3", "7"])

            # Multi-resolution sub-clustering
            subcluster_cells("geo_gse12345_clustered", clusters_to_refine=["0"],
                            resolutions=[0.25, 0.5, 1.0])

            # Full re-clustering (all cells)
            subcluster_cells("geo_gse12345_clustered", clusters_to_refine=None)
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Sub-clustering modality '{modality_name}': {adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Perform sub-clustering using service
            result, stats, ir = clustering_service.subcluster_cells(
                adata,
                cluster_key=cluster_key,
                clusters_to_refine=clusters_to_refine,
                resolution=resolution,
                resolutions=resolutions,
                n_pcs=n_pcs,
                n_neighbors=n_neighbors,
                demo_mode=demo_mode,
            )

            # Store result with descriptive suffix
            new_name = f"{modality_name}_subclustered"
            data_manager.modalities[new_name] = result

            # Log with IR (mandatory for reproducibility)
            data_manager.log_tool_usage(
                "subcluster_cells",
                {
                    "cluster_key": cluster_key,
                    "clusters_to_refine": clusters_to_refine,
                    "resolution": resolution,
                    "resolutions": resolutions,
                    "n_pcs": n_pcs,
                    "n_neighbors": n_neighbors,
                    "demo_mode": demo_mode,
                },
                description=f"Subclustered {len(clusters_to_refine)} clusters from {cluster_key}",
                ir=ir,
            )

            # Format response based on single vs multi-resolution
            n_resolutions_tested = len(stats.get("resolutions_tested", [resolution]))

            if n_resolutions_tested > 1:
                # Multi-resolution formatting
                response = f"""Sub-clustering complete! Created '{new_name}' modality.

**Results:**
- Processed {stats['n_cells_subclustered']:,} cells from {len(stats['parent_clusters'])} parent cluster(s): {stats['parent_clusters']}
- Tested {n_resolutions_tested} resolutions: {stats['resolutions_tested']}
- New columns in adata.obs:"""

                # Show all resolution results
                for res_key, n_clusters in stats["multi_resolution_summary"].items():
                    primary_marker = (
                        " (primary)" if res_key == stats["primary_column"] else ""
                    )
                    response += (
                        f"\n  * {res_key}: {n_clusters} sub-clusters{primary_marker}"
                    )

                response += (
                    f"\n- Execution time: {stats['execution_time']:.2f} seconds\n"
                )

                # Show primary sub-clustering results
                response += (
                    f"\n**Primary sub-clustering ({stats['primary_column']}):**\n"
                )
                for cluster_id, size in list(stats["subcluster_sizes"].items())[:10]:
                    response += f"  - {cluster_id}: {size} cells\n"

                if len(stats["subcluster_sizes"]) > 10:
                    remaining = len(stats["subcluster_sizes"]) - 10
                    response += f"  ... and {remaining} more sub-clusters\n"

                response += """
**Interpretation:**
- Lower resolutions (0.25) = broader populations
- Higher resolutions (1.0) = finer-grained clusters
- Compare results across resolutions to determine optimal granularity"""

            else:
                # Single-resolution formatting
                response = f"""Sub-clustering complete! Created '{new_name}' modality.

**Results:**
- Processed {stats['n_cells_subclustered']:,} cells from {len(stats['parent_clusters'])} parent cluster(s): {stats['parent_clusters']}
- Generated {stats['n_subclusters']} sub-clusters at resolution {stats.get('resolution', resolution)}
- New column: '{stats['primary_column']}' in adata.obs
- Execution time: {stats['execution_time']:.2f} seconds

**Sub-cluster sizes:**"""

                for cluster_id, size in list(stats["subcluster_sizes"].items())[:10]:
                    response += f"\n  - {cluster_id}: {size} cells"

                if len(stats["subcluster_sizes"]) > 10:
                    remaining = len(stats["subcluster_sizes"]) - 10
                    response += f"\n  ... and {remaining} more sub-clusters"

                response += """

**Next steps:**
- Use visualization to display sub-clusters on UMAP
- Use find_marker_genes_for_clusters() to characterize each sub-cluster
- INVOKE handoff_to_annotation_expert immediately if annotation requested (do NOT just suggest it)"""

            analysis_results["details"]["sub_clustering"] = response
            return response

        except ValueError as e:
            # User-friendly error messages for validation failures
            return f"""Error: {str(e)}

Please check:
- Cluster key '{cluster_key}' exists in adata.obs
- Cluster IDs in clusters_to_refine are valid
- Initial clustering has been performed"""
        except (ClusteringError, ModalityNotFoundError) as e:
            logger.error(f"Error in sub-clustering: {e}")
            return f"Error sub-clustering modality: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in sub-clustering: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def evaluate_clustering_quality(
        modality_name: str,
        cluster_key: str = "leiden",
        use_rep: str = "X_pca",
        n_pcs: Optional[int] = None,
        metrics: Optional[List[str]] = None,
    ) -> str:
        """
        Evaluate clustering quality using multiple quantitative metrics.

        Computes 3 scientifically-validated metrics to assess clustering results:
        - Silhouette score: How well-separated clusters are (-1 to 1, higher better)
        - Davies-Bouldin index: Ratio of intra/inter-cluster distances (lower better)
        - Calinski-Harabasz score: Ratio of between/within variance (higher better)

        These metrics help answer:
        - "Is my clustering good?"
        - "Which resolution gives the best separation?"
        - "Am I over-clustering or under-clustering?"

        Common use cases:
        - Validate clustering quality after initial clustering
        - Compare multiple resolutions to select optimal one
        - Identify problematic clusters
        - Objectively evaluate clustering before marker gene analysis

        Args:
            modality_name: Name of the modality to evaluate
            cluster_key: Key in adata.obs containing cluster labels (default: "leiden")
                        For multi-resolution results, use keys like "leiden_res0_5"
            use_rep: Representation to use for distance calculations (default: "X_pca")
                    Options: "X_pca", "X_umap", or any key in adata.obsm
            n_pcs: Number of PCs to use (default: None = use all available)
                   Recommended: 30 for full clustering, 20 for sub-clustering
            metrics: List of specific metrics to compute (default: None = all 3)
                    Options: ["silhouette", "davies_bouldin", "calinski_harabasz"]

        Returns:
            str: Detailed quality report with scores, interpretations, and recommendations

        Examples:
            # Evaluate single clustering result
            evaluate_clustering_quality("geo_gse12345_clustered")

            # Compare multiple resolutions
            evaluate_clustering_quality("geo_gse12345_clustered", cluster_key="leiden_res0_25")
            evaluate_clustering_quality("geo_gse12345_clustered", cluster_key="leiden_res0_5")
            evaluate_clustering_quality("geo_gse12345_clustered", cluster_key="leiden_res1_0")

            # Evaluate using UMAP representation
            evaluate_clustering_quality("geo_gse12345_clustered", use_rep="X_umap")

            # Compute only silhouette score
            evaluate_clustering_quality("geo_gse12345_clustered", metrics=["silhouette"])
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                available = ", ".join(data_manager.list_modalities()[:5])
                return (
                    f"Error: Modality '{modality_name}' not found.\n\n"
                    f"Available modalities: {available}...\n\n"
                    f"Use check_data_status() to see all available modalities."
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)

            # Validate cluster_key exists
            if cluster_key not in adata.obs.columns:
                available_keys = [
                    col for col in adata.obs.columns if "leiden" in col.lower()
                ]
                return (
                    f"Error: Cluster key '{cluster_key}' not found in adata.obs.\n\n"
                    f"Available clustering columns: {', '.join(available_keys)}\n\n"
                    f"Did you run cluster_modality() first?"
                )

            # Call service
            try:
                result, stats, ir = clustering_service.compute_clustering_quality(
                    adata,
                    cluster_key=cluster_key,
                    use_rep=use_rep,
                    n_pcs=n_pcs,
                    metrics=metrics,
                )
            except ValueError as e:
                return (
                    f"Error: {str(e)}\n\n"
                    f"Please check:\n"
                    f"- Cluster key '{cluster_key}' exists in adata.obs\n"
                    f"- Representation '{use_rep}' exists in adata.obsm\n"
                    f"- At least 2 clusters are present\n"
                    f"- PCA has been computed (if use_rep='X_pca')"
                )
            except Exception as e:
                raise ClusteringError(f"Clustering quality evaluation failed: {str(e)}")

            # Store result with quality evaluation suffix
            new_name = f"{modality_name}_quality_evaluated"
            data_manager.modalities[new_name] = result

            # Log with IR (mandatory for reproducibility)
            data_manager.log_tool_usage(
                "evaluate_clustering_quality",
                {
                    "cluster_key": cluster_key,
                    "use_rep": use_rep,
                    "n_pcs": n_pcs,
                    "metrics": (
                        metrics
                        if metrics
                        else ["silhouette", "davies_bouldin", "calinski_harabasz"]
                    ),
                },
                description=f"Evaluated clustering quality with {len(stats['metrics'])} metrics",
                ir=ir,
            )

            # Build response
            response_lines = []

            # Header
            response_lines.append("=" * 70)
            response_lines.append(f"CLUSTERING QUALITY EVALUATION: {cluster_key}")
            response_lines.append("=" * 70)
            response_lines.append("")

            # Basic info
            response_lines.append(f"**Modality**: {modality_name} -> {new_name}")
            response_lines.append(f"**Cells**: {stats['n_cells']:,}")
            response_lines.append(f"**Clusters**: {stats['n_clusters']}")
            response_lines.append(
                f"**Representation**: {stats['use_rep']} ({stats['n_pcs_used']} components)"
            )
            response_lines.append("")

            # Quality metrics section
            response_lines.append("**Quality Metrics:**")
            response_lines.append("-" * 70)

            # Silhouette score
            if "silhouette_score" in stats:
                score = stats["silhouette_score"]
                emoji = "[GOOD]" if score > 0.5 else "[OK]" if score > 0.25 else "[LOW]"
                response_lines.append(
                    f"{emoji} **Silhouette Score**: {score:.4f} "
                    f"(range: -1 to 1, higher = better separation)"
                )

            # Davies-Bouldin index
            if "davies_bouldin_index" in stats:
                score = stats["davies_bouldin_index"]
                emoji = "[GOOD]" if score < 1.0 else "[OK]" if score < 2.0 else "[HIGH]"
                response_lines.append(
                    f"{emoji} **Davies-Bouldin Index**: {score:.4f} "
                    f"(range: 0 to inf, lower = better compactness)"
                )

            # Calinski-Harabasz score
            if "calinski_harabasz_score" in stats:
                score = stats["calinski_harabasz_score"]
                emoji = "[GOOD]" if score > 1000 else "[OK]" if score > 100 else "[LOW]"
                response_lines.append(
                    f"{emoji} **Calinski-Harabasz Score**: {score:.1f} "
                    f"(range: 0 to inf, higher = better variance ratio)"
                )

            response_lines.append("")

            # Per-cluster silhouette scores (if available)
            if "per_cluster_silhouette" in stats:
                response_lines.append("**Per-Cluster Silhouette Scores:**")
                per_cluster = stats["per_cluster_silhouette"]

                # Sort by score (lowest first to highlight problems)
                sorted_clusters = sorted(per_cluster.items(), key=lambda x: x[1])

                for cluster_id, score in sorted_clusters[:10]:
                    size = stats["cluster_sizes"].get(cluster_id, 0)
                    emoji = (
                        "[GOOD]" if score > 0.5 else "[OK]" if score > 0.25 else "[LOW]"
                    )
                    response_lines.append(
                        f"  {emoji} Cluster {cluster_id}: {score:.4f} ({size} cells)"
                    )

                if len(sorted_clusters) > 10:
                    response_lines.append(
                        f"  ... and {len(sorted_clusters) - 10} more clusters"
                    )
                response_lines.append("")

            # Interpretation section
            response_lines.append("**Interpretation:**")
            response_lines.append("-" * 70)
            interpretation = stats.get("interpretation", "")
            for line in interpretation.split("\n"):
                if line.strip():
                    response_lines.append(f"* {line}")
            response_lines.append("")

            # Recommendations section
            recommendations = stats.get("recommendations", [])
            if recommendations:
                response_lines.append("**Recommendations:**")
                response_lines.append("-" * 70)
                for rec in recommendations:
                    response_lines.append(f"* {rec}")
                response_lines.append("")

            # Next steps
            response_lines.append("**Next Steps:**")
            response_lines.append("-" * 70)
            response_lines.append(
                "* If comparing resolutions: Run this on each resolution key (leiden_res0_25, leiden_res0_5, etc.)"
            )
            response_lines.append(
                "* If silhouette < 0.25: Try lower resolution or different preprocessing"
            )
            response_lines.append(
                "* If clusters look good: Proceed with find_marker_genes_for_clusters()"
            )
            response_lines.append(
                "* To visualize: Request visualization through supervisor"
            )
            response_lines.append("")

            # Footer
            response_lines.append("=" * 70)
            response_lines.append(
                f"Evaluation completed in {stats['execution_time_seconds']:.2f}s"
            )
            response_lines.append("=" * 70)

            return "\n".join(response_lines)

        except ModalityNotFoundError as e:
            logger.error(f"Error in clustering quality evaluation: {e}")
            return f"Error evaluating clustering quality: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in clustering quality evaluation: {e}")
            return f"Unexpected error: {str(e)}"

    @tool
    def find_marker_genes_for_clusters(
        modality_name: str,
        groupby: str = "leiden",
        groups: List[str] = None,
        method: str = "wilcoxon",
        n_genes: int = 25,
        min_fold_change: float = 1.5,
        min_pct: float = 0.25,
        max_out_pct: float = 0.5,
        save_result: bool = True,
    ) -> str:
        """
        Find marker genes for single-cell clusters using differential expression analysis.

        Args:
            modality_name: Name of the single-cell modality to analyze
            groupby: Column name to group by (e.g., 'leiden', 'cell_type')
            groups: Specific clusters to analyze (None for all)
            method: Statistical method ('wilcoxon', 't-test', 'logreg')
            n_genes: Number of top marker genes per cluster
            min_fold_change: Minimum fold-change threshold (default: 1.5).
                Filters genes with fold-change below this value.
            min_pct: Minimum in-group expression fraction (default: 0.25).
                Filters genes expressed in <25% of in-group cells.
            max_out_pct: Maximum out-group expression fraction (default: 0.5).
                Filters genes expressed in >50% of out-group cells (less specific).
            save_result: Whether to save the results
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Finding marker genes in single-cell modality '{modality_name}': {adata.shape[0]} cells x {adata.shape[1]} genes"
            )

            # Validate groupby column exists
            if groupby not in adata.obs.columns:
                available_columns = [
                    col
                    for col in adata.obs.columns
                    if col in ["leiden", "cell_type", "louvain", "cluster"]
                ]
                return f"Cluster column '{groupby}' not found. Available clustering columns: {available_columns}"

            # Use singlecell service for marker gene detection
            adata_markers, marker_stats, ir = enhanced_service.find_marker_genes(
                adata=adata,
                groupby=groupby,
                groups=groups,
                method=method,
                n_genes=n_genes,
                min_fold_change=min_fold_change,
                min_pct=min_pct,
                max_out_pct=max_out_pct,
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
                    "n_genes": n_genes,
                    "min_fold_change": min_fold_change,
                    "min_pct": min_pct,
                    "max_out_pct": max_out_pct,
                },
                description=f"Found marker genes for {marker_stats['n_groups']} clusters (method: {marker_stats['method']}, pre-filter: {sum(marker_stats['pre_filter_counts'].values())}, post-filter: {sum(marker_stats['post_filter_counts'].values())}, filtered: {marker_stats['total_genes_filtered']})",
                ir=ir,
            )

            # Format professional response with filtering statistics
            response_parts = [
                f"Successfully found marker genes for single-cell clusters in '{modality_name}'!",
                "\n**Single-cell Marker Gene Analysis:**",
                f"- Grouping by: {marker_stats['groupby']}",
                f"- Number of clusters: {marker_stats['n_groups']}",
                f"- Method: {marker_stats['method']}",
                f"- Top genes per cluster: {marker_stats['n_genes']}",
                "\n**Filtering Parameters:**",
                f"  - Min fold-change: {marker_stats['filtering_params']['min_fold_change']}",
                f"  - Min in-group %: {marker_stats['filtering_params']['min_pct']*100:.1f}%",
                f"  - Max out-group %: {marker_stats['filtering_params']['max_out_pct']*100:.1f}%",
            ]

            # Add filtering summary
            if "filtered_counts" in marker_stats:
                response_parts.append(
                    f"\n**Filtering Summary:** {marker_stats['total_genes_filtered']} genes removed"
                )
                response_parts.append("\n**Genes per Cluster (after filtering):**")
                for group in marker_stats["groups_analyzed"][:10]:
                    if group in marker_stats["post_filter_counts"]:
                        post = marker_stats["post_filter_counts"][group]
                        filtered = marker_stats["filtered_counts"][group]
                        pre = marker_stats["pre_filter_counts"][group]
                        response_parts.append(
                            f"  - Cluster {group}: {post} genes (filtered {filtered}/{pre})"
                        )

                if len(marker_stats["groups_analyzed"]) > 10:
                    remaining = len(marker_stats["groups_analyzed"]) - 10
                    response_parts.append(f"  ... and {remaining} more clusters")

            response_parts.append("\n**Top Marker Genes by Cluster:**")

            # Show top marker genes for each cluster (first 5 clusters)
            if "top_markers_per_group" in marker_stats:
                for cluster_id in list(marker_stats["top_markers_per_group"].keys())[
                    :5
                ]:
                    top_genes = marker_stats["top_markers_per_group"][cluster_id][:5]
                    gene_names = [gene["gene"] for gene in top_genes]
                    response_parts.append(
                        f"  - **Cluster {cluster_id}**: {', '.join(gene_names)}"
                    )

                if len(marker_stats["top_markers_per_group"]) > 5:
                    remaining = len(marker_stats["top_markers_per_group"]) - 5
                    response_parts.append(f"  ... and {remaining} more clusters")

            response_parts.append(
                f"\n**New modality created**: '{marker_modality_name}'"
            )

            if save_result:
                response_parts.append(f"**Saved to**: {save_path}")

            response_parts.append(
                "**Access detailed results**: adata.uns['rank_genes_groups']"
            )
            response_parts.append(
                "\n**CRITICAL**: If annotation requested, INVOKE handoff_to_annotation_expert immediately (do NOT just suggest it)."
            )

            response = "\n".join(response_parts)

            analysis_results["details"]["marker_genes"] = response
            return response

        except (ServiceSingleCellError, ModalityNotFoundError) as e:
            logger.error(f"Error finding single-cell marker genes: {e}")
            return f"Error finding marker genes for single-cell clusters: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error finding single-cell marker genes: {e}")
            return f"Unexpected error: {str(e)}"

    # =========================================================================
    # COLLECT ALL TOOLS
    # =========================================================================

    # Clustering tools (single-cell specific)
    clustering_tools = [
        cluster_modality,
        subcluster_cells,
        evaluate_clustering_quality,
        find_marker_genes_for_clusters,
    ]

    # Combine all direct tools
    direct_tools = shared_tools + clustering_tools

    # Add delegation tools if provided (annotation_expert, de_analysis_expert)
    tools = direct_tools
    if delegation_tools:
        tools = tools + delegation_tools

    # Create system prompt
    system_prompt = create_transcriptomics_expert_prompt()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=TranscriptomicsExpertState,
    )
