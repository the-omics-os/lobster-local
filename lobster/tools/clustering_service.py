"""
Clustering service for single-cell RNA-seq data.

This service provides methods for clustering single-cell RNA-seq data
and generating visualizations of the results.
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple

import anndata
import plotly.express as px
import plotly.graph_objects as go
import scanpy as sc

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger
from lobster.utils.progress_wrapper import with_periodic_progress

logger = get_logger(__name__)


class ClusteringError(Exception):
    """Base exception for clustering operations."""

    pass


class ClusteringService:
    """
    Stateless service for clustering single-cell RNA-seq data.

    This class provides methods to perform clustering and dimensionality
    reduction on single-cell RNA-seq data and generate visualizations.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the clustering service.

        Args:
            config: Optional configuration dict (ignored, for backward compatibility)
            **kwargs: Additional arguments (ignored, for backward compatibility)

        This service is stateless and doesn't require a data manager instance.
        """
        logger.debug("Initializing stateless ClusteringService")
        self.config = config or {}
        self.default_cluster_resolution = 0.7
        self.progress_callback = None
        self.current_progress = 0
        self.total_steps = 0
        logger.debug("ClusteringService initialized successfully")

    def set_progress_callback(self, callback: Callable[[int, str], None]) -> None:
        """
        Set a callback function to report progress.

        The callback function should accept two parameters:
        - progress: int (0-100 percentage)
        - message: str (description of current operation)

        Args:
            callback: Callable function to receive progress updates
        """
        self.progress_callback = callback
        logger.info("Progress callback set")

    def _update_progress(self, step_name: str) -> None:
        """
        Update progress and call the progress callback if set.

        Args:
            step_name: Name of the current processing step
        """
        self.current_progress += 1
        if self.progress_callback is not None:
            progress_percent = int((self.current_progress / self.total_steps) * 100)
            self.progress_callback(progress_percent, step_name)
            logger.info(f"Progress updated: {progress_percent}% - {step_name}")

    def _create_progress_callback(self):
        """
        Create a progress callback wrapper for use with with_periodic_progress.

        Returns:
            Callable that adapts with_periodic_progress messages to the existing callback system
        """

        def progress_callback_wrapper(message: str):
            if self.progress_callback:
                self.progress_callback(None, message)

        return progress_callback_wrapper

    def _create_clustering_ir(
        self,
        resolution: float,
        n_neighbors: int = 15,
        n_pcs: int = 20,
        min_mean: float = 0.0125,
        max_mean: float = 3.0,
        min_disp: float = 0.5,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for full clustering pipeline.

        Args:
            resolution: Leiden clustering resolution parameter
            n_neighbors: Number of neighbors for graph construction
            n_pcs: Number of principal components to use
            min_mean: Minimum mean for HVG selection
            max_mean: Maximum mean for HVG selection
            min_disp: Minimum dispersion for HVG selection

        Returns:
            AnalysisStep with full clustering pipeline code template
        """
        # Parameter schema
        parameter_schema = {
            "resolution": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=resolution,
                required=False,
                validation_rule="resolution > 0",
                description="Resolution parameter for Leiden clustering",
            ),
            "n_neighbors": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_neighbors,
                required=False,
                validation_rule="n_neighbors > 0",
                description="Number of neighbors for graph construction",
            ),
            "n_pcs": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_pcs,
                required=False,
                validation_rule="n_pcs > 0",
                description="Number of principal components to use",
            ),
        }

        # Jinja2 template with full clustering pipeline
        code_template = """# Full single-cell clustering pipeline
# Following best practices from scanpy workflows

# 1. Normalize and log-transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("Normalization complete")

# 2. Find highly variable genes
sc.pp.highly_variable_genes(
    adata,
    min_mean={{ min_mean }},
    max_mean={{ max_mean }},
    min_disp={{ min_disp }},
    flavor='seurat'
)
n_hvg = sum(adata.var.highly_variable)
print(f"Identified {n_hvg} highly variable genes")

# 3. Store raw data and subset to HVG
adata.raw = adata.copy()
adata = adata[:, adata.var.highly_variable]

# 4. Scale data
sc.pp.scale(adata, max_value=10)
print("Scaling complete")

# 5. PCA dimensionality reduction
sc.tl.pca(adata, svd_solver='arpack')
print(f"PCA complete (using {{ n_pcs }} components)")

# 6. Compute neighborhood graph
sc.pp.neighbors(adata, n_neighbors={{ n_neighbors }}, n_pcs={{ n_pcs }})
print("Neighborhood graph computed")

# 7. Leiden clustering
sc.tl.leiden(adata, resolution={{ resolution }}, key_added='leiden')
n_clusters = len(adata.obs['leiden'].unique())
print(f"Leiden clustering complete: {n_clusters} clusters (resolution={{ resolution }})")

# 8. UMAP visualization
sc.tl.umap(adata)
print("UMAP coordinates computed")

print(f"Clustering pipeline complete: {adata.n_obs} cells in {n_clusters} clusters")
"""

        return AnalysisStep(
            operation="scanpy.tl.cluster_pipeline",
            tool_name="cluster_and_visualize",
            description=f"Full clustering pipeline: HVG + PCA + neighbors + Leiden (res={resolution}) + UMAP",
            library="scanpy",
            code_template=code_template,
            imports=["import scanpy as sc", "import numpy as np"],
            parameters={
                "resolution": resolution,
                "n_neighbors": n_neighbors,
                "n_pcs": n_pcs,
                "min_mean": min_mean,
                "max_mean": max_mean,
                "min_disp": min_disp,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "clustering",
                "pipeline_steps": [
                    "normalize",
                    "hvg",
                    "scale",
                    "pca",
                    "neighbors",
                    "leiden",
                    "umap",
                ],
                "resolution": resolution,
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def cluster_and_visualize(
        self,
        adata: anndata.AnnData,
        resolution: Optional[float] = None,
        use_rep: Optional[str] = None,
        batch_correction: bool = False,
        batch_key: Optional[str] = None,
        demo_mode: bool = False,
        subsample_size: Optional[int] = None,
        skip_steps: Optional[list] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform clustering and UMAP visualization on single-cell RNA-seq data.

        Args:
            adata: AnnData object to cluster
            resolution: Resolution parameter for Leiden clustering
            use_rep: Representation to use for clustering (e.g., 'X_scvi', 'X_pca').
                    If None, uses standard PCA workflow. If specified, uses the
                    custom embedding from adata.obsm[use_rep] for neighbor calculation.
            batch_correction: Whether to perform batch correction
            batch_key: Column name for batch information
            demo_mode: Whether to run in demo mode (faster processing with reduced quality)
            subsample_size: Maximum number of cells to include (subsamples if larger)
            skip_steps: List of steps to skip in demo mode (e.g. ['marker_genes'])

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: Clustered AnnData, clustering stats, and IR

        Raises:
            ClusteringError: If clustering fails
        """
        try:
            logger.info("Starting clustering and visualization pipeline")

            # Initialize progress tracking
            self.current_progress = 0
            skip_steps = skip_steps or []
            # Calculate total steps based on what will be performed
            self.total_steps = 7  # Base steps: subsample, batch check, normalize, hvg, scale, pca, neighbors, leiden, umap
            if batch_correction:
                self.total_steps += 1
            if "marker_genes" not in skip_steps:
                self.total_steps += 1

            # Set resolution
            if resolution is None:
                resolution = self.default_cluster_resolution

            logger.info(f"Performing clustering with resolution {resolution}")

            # Create working copy
            adata_clustered = adata.copy()
            original_shape = adata_clustered.shape

            logger.info(
                f"Input data dimensions: {original_shape[0]} cells × {original_shape[1]} genes"
            )

            # Handle demo mode settings
            if demo_mode:
                logger.info("Running in demo mode (faster processing)")
                if "marker_genes" not in skip_steps:
                    skip_steps.append("marker_genes")
                if not subsample_size:
                    subsample_size = min(
                        1000, original_shape[0]
                    )  # Default to 1000 cells in demo mode

            # Subsample if needed
            if subsample_size and adata_clustered.n_obs > subsample_size:
                logger.info(
                    f"Subsampling data to {subsample_size} cells (from {adata_clustered.n_obs})"
                )
                sc.pp.subsample(adata_clustered, n_obs=subsample_size, random_state=42)
                logger.info(f"Data subsampled: {adata_clustered.n_obs} cells remaining")

            self._update_progress("Data preparation completed")

            # Check for batch information if batch correction requested
            if batch_correction:
                if batch_key is None:
                    # Auto-detect batch key
                    for potential_key in ["Patient_ID", "patient", "batch", "sample"]:
                        if potential_key in adata_clustered.obs.columns:
                            batch_key = potential_key
                            logger.info(f"Auto-detected batch key: {batch_key}")
                            break

                if batch_key and batch_key in adata_clustered.obs.columns:
                    unique_batches = adata_clustered.obs[batch_key].unique()
                    if len(unique_batches) > 1:
                        logger.info(
                            f"Found {len(unique_batches)} batches for correction: {list(unique_batches)}"
                        )
                        adata_clustered = self._perform_batch_correction(
                            adata_clustered, batch_key
                        )
                    else:
                        logger.info(
                            "Only one batch found - proceeding without batch correction"
                        )
                        batch_correction = False
                else:
                    logger.warning(
                        "Batch correction requested but no valid batch key found"
                    )
                    batch_correction = False

            self._update_progress(
                "Batch correction completed"
                if batch_correction
                else "Batch check completed"
            )

            # Perform clustering
            adata_clustered = self._perform_clustering(
                adata_clustered, resolution, use_rep, demo_mode, skip_steps
            )

            # Compile clustering statistics
            n_clusters = len(adata_clustered.obs["leiden"].unique())
            cluster_counts = adata_clustered.obs["leiden"].value_counts().to_dict()

            clustering_stats = {
                "analysis_type": "clustering",
                "resolution": resolution,
                "n_clusters": n_clusters,
                "batch_correction": batch_correction,
                "batch_key": batch_key,
                "demo_mode": demo_mode,
                "subsample_size": subsample_size,
                "original_shape": original_shape,
                "final_shape": adata_clustered.shape,
                "cluster_counts": cluster_counts,
                "cluster_sizes": {
                    str(cluster): int(count)
                    for cluster, count in cluster_counts.items()
                },
                "has_umap": "X_umap" in adata_clustered.obsm,
                "has_marker_genes": "rank_genes_groups" in adata_clustered.uns,
            }

            # Add batch information if available
            if batch_correction and batch_key:
                batch_counts = adata_clustered.obs[batch_key].value_counts().to_dict()
                clustering_stats["batch_counts"] = batch_counts
                clustering_stats["n_batches"] = len(batch_counts)

            # Add marker gene information if available
            if (
                "rank_genes_groups" in adata_clustered.uns
                and "marker_genes" not in skip_steps
            ):
                marker_genes = {}
                for cluster in adata_clustered.obs["leiden"].unique():
                    genes = adata_clustered.uns["rank_genes_groups"]["names"][cluster]
                    scores = adata_clustered.uns["rank_genes_groups"]["scores"][cluster]
                    marker_genes[str(cluster)] = [
                        {"gene": str(gene), "score": float(score)}
                        for gene, score in zip(genes[:10], scores[:10])
                    ]
                clustering_stats["top_marker_genes"] = marker_genes

            logger.info(f"Clustering completed: {n_clusters} clusters identified")

            # Create IR for notebook export
            ir = self._create_clustering_ir(
                resolution=resolution,
                n_neighbors=15,  # Default from _perform_clustering
                n_pcs=20,  # Default from _perform_clustering
            )

            return adata_clustered, clustering_stats, ir

        except Exception as e:
            logger.exception(f"Error during clustering: {e}")
            raise ClusteringError(f"Clustering failed: {str(e)}")

    def _perform_batch_correction(
        self, adata: anndata.AnnData, batch_key: str
    ) -> anndata.AnnData:
        """
        Perform simple batch correction on the data.

        Args:
            adata: AnnData object with batch information
            batch_key: Column name containing batch labels

        Returns:
            anndata.AnnData: Batch-corrected AnnData object
        """
        logger.info(f"Performing batch correction using batch key: {batch_key}")

        try:
            # Simple batch correction by normalizing each batch separately
            unique_batches = adata.obs[batch_key].unique()
            batch_list = []

            for batch in unique_batches:
                batch_mask = adata.obs[batch_key] == batch
                batch_adata = adata[batch_mask].copy()

                # Normalize each batch separately
                sc.pp.normalize_total(batch_adata, target_sum=1e4)
                sc.pp.log1p(batch_adata)

                batch_list.append(batch_adata)

            # Concatenate corrected batches
            adata_corrected = anndata.concat(
                batch_list, label=batch_key, keys=unique_batches
            )

            logger.info(f"Batch correction completed for {len(unique_batches)} batches")
            return adata_corrected

        except Exception as e:
            logger.warning(
                f"Batch correction failed, proceeding without correction: {e}"
            )
            return adata

    def _perform_clustering(
        self,
        adata: sc.AnnData,
        resolution: float,
        use_rep: Optional[str] = None,
        demo_mode: bool = False,
        skip_steps: Optional[list] = None,
    ) -> sc.AnnData:
        """
        Perform clustering on the AnnData object based on the publication workflow.

        Args:
            adata: AnnData object
            resolution: Clustering resolution parameter
            use_rep: Representation to use for clustering (e.g., 'X_scvi', 'X_pca').
                    If None, uses standard PCA workflow. If specified, uses the
                    custom embedding from adata.obsm[use_rep] for neighbor calculation.
            demo_mode: Whether to run in demo mode (faster with reduced quality)
            skip_steps: List of steps to skip (e.g., 'marker_genes')

        Returns:
            sc.AnnData: AnnData object with clustering results
        """
        logger.info("Performing clustering pipeline based on publication workflow")
        skip_steps = skip_steps or []
        start_time = time.time()

        # Check if using custom embeddings (e.g., scVI)
        if use_rep and use_rep in adata.obsm:
            logger.info(f"Using custom embedding '{use_rep}' for clustering")

            # Validate custom embedding
            embedding_shape = adata.obsm[use_rep].shape
            logger.info(
                f"Custom embedding shape: {embedding_shape[0]} cells × {embedding_shape[1]} dimensions"
            )

            # Store raw data for marker gene analysis
            adata.raw = adata.copy()
            self._update_progress(
                "Using custom embeddings (skipping normalization/PCA)"
            )

            # Compute neighborhood graph using custom embedding
            logger.info("Computing neighborhood graph from custom embedding")
            n_neighbors = 10 if demo_mode else 15

            with with_periodic_progress(
                "Computing neighborhood graph from custom embedding",
                self._create_progress_callback(),
                update_interval=10,
                show_elapsed=True,
            ):
                sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)

            self._update_progress("Neighborhood graph computed from custom embedding")

            # Run Leiden clustering
            logger.info(f"Running Leiden clustering with resolution {resolution}")

            with with_periodic_progress(
                f"Running Leiden clustering (resolution={resolution})",
                self._create_progress_callback(),
                update_interval=10,
                show_elapsed=True,
            ):
                sc.tl.leiden(adata, resolution=resolution, key_added="leiden")

            self._update_progress("Leiden clustering completed")

            # UMAP for visualization using custom embedding
            logger.info("Computing UMAP coordinates from custom embedding")

            with with_periodic_progress(
                "Computing UMAP coordinates from custom embedding",
                self._create_progress_callback(),
                update_interval=15,
                show_elapsed=True,
            ):
                if demo_mode:
                    sc.tl.umap(adata, min_dist=0.5, spread=1.5)
                else:
                    sc.tl.umap(adata)

            self._update_progress("UMAP coordinates computed from custom embedding")

        else:
            # Standard workflow when no custom embedding specified
            if use_rep:
                logger.warning(
                    f"Custom embedding '{use_rep}' not found in adata.obsm, using standard PCA workflow"
                )

            # Basic preprocessing (follows publication workflow)
            logger.info("Normalizing data")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            self._update_progress("Normalization completed")

            # Find highly variable genes (follows the parameters from 02_All_cell_clustering.R)
            logger.info("Finding highly variable genes")
            sc.pp.highly_variable_genes(
                adata, min_mean=0.0125, max_mean=3, min_disp=0.5, flavor="seurat"
            )
            self._update_progress("Highly variable genes identified")

            # Store raw data before scaling
            adata.raw = adata.copy()

            # Use only highly variable genes for dimensionality reduction
            n_hvg = sum(adata.var.highly_variable)
            logger.info(f"Using {n_hvg} highly variable genes")

            # ISSUE #7 FIX: Handle case when no HVG detected
            if n_hvg == 0:
                logger.warning(
                    "No highly variable genes detected with current parameters. "
                    "Using all genes for clustering instead."
                )
                raise ClusteringError(
                    "No highly variable genes detected. This typically indicates:\n"
                    "1. Data has very low variance (all genes have similar expression)\n"
                    "2. Dataset is too small or uniform for HVG detection\n"
                    "3. Data may need quality filtering before clustering\n\n"
                    "Suggestions:\n"
                    "- Check data quality (use quality_service for QC metrics)\n"
                    "- Ensure data is raw counts (not already normalized/log-transformed)\n"
                    "- Try filtering out low-quality cells/genes first"
                )

            # In demo mode, further restrict the number of HVG for faster processing
            if demo_mode and n_hvg > 1000:
                logger.info("Demo mode: Restricting to top 1000 variable genes")
                # Get the top 1000 most variable genes
                most_variable_genes = (
                    adata.var.sort_values("dispersions_norm", ascending=False)
                    .head(1000)
                    .index
                )
                adata_hvg = adata[:, most_variable_genes]
            else:
                adata_hvg = adata[:, adata.var.highly_variable]

            # Scale data (with max value cap to avoid influence of outliers)
            logger.info("Scaling data")
            sc.pp.scale(adata_hvg, max_value=10)
            self._update_progress("Data scaling completed")

            # Determine optimal number of PCs (following publication's approach using 20 PCs)
            # In demo mode, use fewer PCs for faster processing
            n_pcs = 10 if demo_mode else 20

            # ISSUE #7 FIX: Validate n_pcs against available features
            n_features = adata_hvg.n_vars
            if n_features < n_pcs:
                old_n_pcs = n_pcs
                n_pcs = min(
                    n_features - 1, n_features // 2
                )  # Use at most half of features
                if n_pcs < 1:
                    raise ClusteringError(
                        f"Insufficient features for PCA: only {n_features} highly variable genes detected, "
                        f"but need at least 2 for PCA. Try using all genes or check data quality."
                    )
                logger.warning(
                    f"Reduced n_pcs from {old_n_pcs} to {n_pcs} due to limited features ({n_features} HVG)"
                )

            logger.info(f"Using {n_pcs} principal components for neighborhood graph")

            # PCA (using 'arpack' SVD solver for better performance with sparse matrices)
            logger.info("Running PCA")
            sc.tl.pca(adata_hvg, svd_solver="arpack", n_comps=n_pcs)
            self._update_progress("PCA completed")

            # Compute neighborhood graph
            logger.info("Computing neighborhood graph")
            n_neighbors = (
                10 if demo_mode else 15
            )  # Use fewer neighbors in demo mode for speed

            with with_periodic_progress(
                "Computing neighborhood graph",
                self._create_progress_callback(),
                update_interval=10,
                show_elapsed=True,
            ):
                sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=n_pcs)

            self._update_progress("Neighborhood graph computed")

            # Run Leiden clustering at specified resolution (similar to publication's approach)
            logger.info(f"Running Leiden clustering with resolution {resolution}")

            with with_periodic_progress(
                f"Running Leiden clustering (resolution={resolution})",
                self._create_progress_callback(),
                update_interval=10,
                show_elapsed=True,
            ):
                sc.tl.leiden(adata_hvg, resolution=resolution, key_added="leiden")

            self._update_progress("Leiden clustering completed")

            # UMAP for visualization
            logger.info("Computing UMAP coordinates")

            with with_periodic_progress(
                "Computing UMAP coordinates",
                self._create_progress_callback(),
                update_interval=15,
                show_elapsed=True,
            ):
                if demo_mode:
                    # Use faster UMAP settings in demo mode
                    sc.tl.umap(adata_hvg, min_dist=0.5, spread=1.5)
                else:
                    sc.tl.umap(adata_hvg)

            self._update_progress("UMAP coordinates computed")

            # Transfer clustering results and UMAP coordinates back to the original object
            adata.obs["leiden"] = adata_hvg.obs["leiden"]
            adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]

        # Find marker genes for each cluster, unless skipped
        if "marker_genes" not in skip_steps:
            logger.info("Finding marker genes for clusters")
            method = (
                "t-test" if demo_mode else "wilcoxon"
            )  # t-test is faster than wilcoxon

            # Wrap the slow marker gene operation with progress updates
            operation_name = f"Finding marker genes using {method}"

            with with_periodic_progress(
                operation_name,
                self._create_progress_callback(),
                update_interval=15,
                show_elapsed=True,
            ):
                sc.tl.rank_genes_groups(adata, "leiden", method=method)

            self._update_progress("Marker genes identified")
        else:
            logger.info("Skipping marker gene identification (demo mode)")

        n_clusters = len(adata.obs["leiden"].unique())
        logger.info(f"Identified {n_clusters} clusters")

        elapsed = time.time() - start_time
        logger.info(f"Clustering completed in {elapsed:.2f} seconds")

        return adata

    def _create_umap_plot(self, adata: sc.AnnData) -> go.Figure:
        """
        Create high-quality UMAP plot from clustering results.

        Args:
            adata: AnnData object with clustering results

        Returns:
            go.Figure: Plotly figure with UMAP plot
        """
        logger.info("Creating high-quality UMAP visualization")

        umap_coords = adata.obsm["X_umap"]
        clusters = adata.obs["leiden"].astype(str)
        n_cells = len(clusters)

        # Create a colormap similar to those in the publication
        n_clusters = len(adata.obs["leiden"].unique())
        if n_clusters <= 10:
            color_map = px.colors.qualitative.Set1
        elif n_clusters <= 20:
            color_map = px.colors.qualitative.Dark24
        else:
            color_map = px.colors.qualitative.Alphabet

        # Dynamic marker sizing based on number of cells for optimal visibility
        if n_cells < 1000:
            marker_size = 8
        elif n_cells < 5000:
            marker_size = 6
        elif n_cells < 10000:
            marker_size = 5
        else:
            marker_size = 4

        fig = px.scatter(
            x=umap_coords[:, 0],
            y=umap_coords[:, 1],
            color=clusters,
            title="UMAP Visualization with Leiden Clusters",
            labels={"x": "UMAP_1", "y": "UMAP_2", "color": "Cluster"},
            width=1200,  # Increased from 800
            height=1000,  # Increased from 700
            color_discrete_sequence=color_map,
        )

        fig.update_traces(
            marker=dict(
                size=marker_size,
                opacity=0.7,
                line=dict(width=0.5, color="rgba(50,50,50,0.4)"),  # Add subtle borders
            )
        )

        fig.update_layout(
            title=dict(
                text="UMAP Visualization with Leiden Clusters",
                font=dict(size=20, family="Arial, sans-serif"),
                x=0.5,  # Center the title
                xanchor="center",
            ),
            legend_title=dict(
                text="Cluster", font=dict(size=16, family="Arial, sans-serif")
            ),
            font=dict(size=14, family="Arial, sans-serif"),
            margin=dict(l=80, r=150, t=80, b=80),  # Increased margins
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                itemsizing="constant",
                font=dict(size=14),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
            xaxis=dict(
                title=dict(text="UMAP_1", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                title=dict(text="UMAP_2", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return fig

    def _create_batch_umap(self, adata: sc.AnnData, batch_key: str) -> go.Figure:
        """
        Create high-quality UMAP plot colored by batch.

        Args:
            adata: AnnData object with clustering results
            batch_key: Key in adata.obs containing batch information

        Returns:
            go.Figure: Plotly figure with batch-colored UMAP plot
        """
        logger.info(
            f"Creating high-quality batch-colored UMAP visualization using {batch_key}"
        )

        umap_coords = adata.obsm["X_umap"]
        batches = adata.obs[batch_key].astype(str)
        n_cells = len(batches)

        # Create color map
        n_batches = len(adata.obs[batch_key].unique())
        if n_batches <= 10:
            color_map = px.colors.qualitative.Set2
        else:
            color_map = px.colors.qualitative.Alphabet

        # Dynamic marker sizing based on number of cells for optimal visibility
        if n_cells < 1000:
            marker_size = 8
        elif n_cells < 5000:
            marker_size = 6
        elif n_cells < 10000:
            marker_size = 5
        else:
            marker_size = 4

        fig = px.scatter(
            x=umap_coords[:, 0],
            y=umap_coords[:, 1],
            color=batches,
            title=f"UMAP Visualization by {batch_key}",
            labels={"x": "UMAP_1", "y": "UMAP_2", "color": batch_key},
            width=1200,  # Increased from 800
            height=1000,  # Increased from 700
            color_discrete_sequence=color_map,
        )

        fig.update_traces(
            marker=dict(
                size=marker_size,
                opacity=0.7,
                line=dict(width=0.5, color="rgba(50,50,50,0.4)"),  # Add subtle borders
            )
        )

        fig.update_layout(
            title=dict(
                text=f"UMAP Visualization by {batch_key}",
                font=dict(size=20, family="Arial, sans-serif"),
                x=0.5,  # Center the title
                xanchor="center",
            ),
            legend_title=dict(
                text=batch_key, font=dict(size=16, family="Arial, sans-serif")
            ),
            font=dict(size=14, family="Arial, sans-serif"),
            margin=dict(l=80, r=150, t=80, b=80),  # Increased margins
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                itemsizing="constant",
                font=dict(size=14),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
            ),
            xaxis=dict(
                title=dict(text="UMAP_1", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                title=dict(text="UMAP_2", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return fig

    def _create_cluster_distribution_plot(self, adata: sc.AnnData) -> go.Figure:
        """
        Create a high-quality cluster size distribution plot.

        Args:
            adata: AnnData object with clustering results

        Returns:
            go.Figure: Plotly figure with cluster distribution plot
        """
        logger.info("Creating high-quality cluster size distribution plot")

        # Get cluster counts
        cluster_counts = adata.obs["leiden"].value_counts().sort_index()

        # Create gradient colors for better visual appeal
        n_clusters = len(cluster_counts)
        colors = px.colors.sample_colorscale(
            "viridis",
            (
                [i / (n_clusters - 1) for i in range(n_clusters)]
                if n_clusters > 1
                else [0]
            ),
        )

        # Create bar plot
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                marker=dict(
                    color=colors, line=dict(color="rgba(50,50,50,0.8)", width=1)
                ),
                text=cluster_counts.values,
                textposition="outside",
                textfont=dict(size=12, color="black"),
                hovertemplate="<b>Cluster %{x}</b><br>"
                + "Cells: %{y}<br>"
                + "Percentage: %{customdata:.1f}%<extra></extra>",
                customdata=[
                    (count / cluster_counts.sum() * 100)
                    for count in cluster_counts.values
                ],
            )
        )

        fig.update_layout(
            title=dict(
                text="Cluster Size Distribution",
                font=dict(size=20, family="Arial, sans-serif"),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title=dict(text="Cluster", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                title=dict(text="Number of Cells", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
                type=(
                    "log"
                    if max(cluster_counts) / min(cluster_counts) > 100
                    else "linear"
                ),
            ),
            width=1000,  # Increased from 800
            height=700,  # Increased from 500
            font=dict(size=14, family="Arial, sans-serif"),
            margin=dict(l=80, r=80, t=80, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )

        return fig

    def estimate_processing_time(self, n_cells: int, n_genes: int) -> Dict[str, float]:
        """
        Estimate processing time for clustering based on data dimensions.

        This can help users decide whether to use demo mode for large datasets.

        Args:
            n_cells: Number of cells in the dataset
            n_genes: Number of genes in the dataset

        Returns:
            Dict containing estimated processing times in seconds:
            - 'standard': For standard processing
            - 'demo': For demo mode processing
        """
        # These are rough approximations based on empirical testing
        # Actual performance will vary based on hardware

        # Base time (overhead)
        base_time = 5.0

        # Standard mode estimates
        # Normalization & HVG: ~0.5s per 1000 cells
        normalization_time = (n_cells / 1000) * 0.5
        # PCA: ~1s per 1000 cells with 2000 genes
        pca_time = (n_cells / 1000) * (n_genes / 2000) * 1.0
        # Neighbors: ~2s per 1000 cells
        neighbors_time = (n_cells / 1000) * 2.0
        # Clustering: ~1s per 1000 cells
        clustering_time = (n_cells / 1000) * 1.0
        # UMAP: ~3s per 1000 cells
        umap_time = (n_cells / 1000) * 3.0
        # Marker genes: ~5s per 1000 cells
        marker_time = (n_cells / 1000) * 5.0

        standard_time = (
            base_time
            + normalization_time
            + pca_time
            + neighbors_time
            + clustering_time
            + umap_time
            + marker_time
        )

        # Demo mode is approximately 5-10x faster due to:
        # - Subsampling to 1000 cells
        # - Using fewer HVG, PCs, and neighbors
        # - Using faster algorithms
        # - Skipping marker gene identification
        demo_time = base_time + min(10.0, standard_time / 5.0)

        return {"standard": standard_time, "demo": demo_time}

    def _format_clustering_report(
        self,
        adata: sc.AnnData,
        resolution: float,
        batch_correction: bool = False,
        batch_key: Optional[str] = None,
        demo_mode: bool = False,
        original_cell_count: Optional[int] = None,
    ) -> str:
        """
        Format clustering results report.

        Args:
            adata: AnnData object with clustering results
            resolution: Clustering resolution parameter
            batch_correction: Whether batch correction was performed
            batch_key: Batch key used for correction
            demo_mode: Whether demo mode was used
            original_cell_count: Original number of cells before subsampling

        Returns:
            str: Formatted report
        """
        n_clusters = len(adata.obs["leiden"].unique())
        cluster_counts = adata.obs["leiden"].value_counts().to_dict()

        # Format cluster counts for display
        cluster_summary = "\n".join(
            [
                f"- Cluster {cluster}: {count} cells ({count/len(adata)*100:.1f}%)"
                for cluster, count in sorted(
                    cluster_counts.items(), key=lambda x: int(x[0])
                )
            ]
        )

        # Get top marker genes for each cluster if available
        marker_summary = ""
        if "rank_genes_groups" in adata.uns:
            marker_summary = "\n\n**Top Marker Genes by Cluster:**\n"
            for cluster in sorted(adata.obs["leiden"].unique(), key=lambda x: int(x)):
                genes = adata.uns["rank_genes_groups"]["names"][cluster][:5]
                adata.uns["rank_genes_groups"]["scores"][cluster][:5]
                marker_summary += f"- Cluster {cluster}: {', '.join(genes)}\n"

        # Batch correction information
        batch_info = ""
        if batch_correction and batch_key:
            batch_info = f"\n- Batch correction performed using '{batch_key}'\n"
            batch_counts = adata.obs[batch_key].value_counts().to_dict()
            batch_info += f"- Number of batches: {len(batch_counts)}\n"

        # Demo mode information
        demo_info = ""
        if demo_mode:
            demo_info = "\n- Analysis performed in DEMO MODE (faster processing with reduced quality)\n"
            if original_cell_count and original_cell_count > adata.n_obs:
                demo_info += f"- Data subsampled from {original_cell_count} to {adata.n_obs} cells\n"

        return f"""Clustering Completed!

**Results Summary:**
- Resolution: {resolution}
- Number of clusters: {n_clusters}
- UMAP coordinates calculated{batch_info}{demo_info}
- Clusters stored for further analysis

**Cluster Distribution:**
{cluster_summary}
{marker_summary}

**Visualization:**
The UMAP plot shows the clustering results. Each point represents a cell, colored by cluster assignment.
You can use these clusters for downstream analysis such as finding marker genes or annotating cell types.

**Next Steps:**
- Find marker genes for specific clusters
- Annotate cell types based on marker genes
- Perform cell type-specific analyses
- Explore differential gene expression between conditions
"""
