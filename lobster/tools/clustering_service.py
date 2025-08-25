"""
Clustering service for single-cell RNA-seq data.

This service provides methods for clustering single-cell RNA-seq data
and generating visualizations of the results.
"""

import time
from typing import Callable, Dict, Optional, Tuple, Any

import anndata
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scanpy as sc

from lobster.utils.logger import get_logger

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

    def __init__(self):
        """
        Initialize the clustering service.
        
        This service is stateless and doesn't require a data manager instance.
        """
        logger.info("Initializing stateless ClusteringService")
        self.default_cluster_resolution = 0.7
        self.progress_callback = None
        self.current_progress = 0
        self.total_steps = 0
        logger.info("ClusteringService initialized successfully")

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

    def cluster_and_visualize(
        self,
        adata: anndata.AnnData,
        resolution: Optional[float] = None,
        batch_correction: bool = False,
        batch_key: Optional[str] = None,
        demo_mode: bool = False,
        subsample_size: Optional[int] = None,
        skip_steps: Optional[list] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Perform clustering and UMAP visualization on single-cell RNA-seq data.

        Args:
            adata: AnnData object to cluster
            resolution: Resolution parameter for Leiden clustering
            batch_correction: Whether to perform batch correction
            batch_key: Column name for batch information
            demo_mode: Whether to run in demo mode (faster processing with reduced quality)
            subsample_size: Maximum number of cells to include (subsamples if larger)
            skip_steps: List of steps to skip in demo mode (e.g. ['marker_genes'])

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: Clustered AnnData and clustering stats
            
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
            
            logger.info(f"Input data dimensions: {original_shape[0]} cells × {original_shape[1]} genes")

            # Handle demo mode settings
            if demo_mode:
                logger.info("Running in demo mode (faster processing)")
                if "marker_genes" not in skip_steps:
                    skip_steps.append("marker_genes")
                if not subsample_size:
                    subsample_size = min(1000, original_shape[0])  # Default to 1000 cells in demo mode

            # Subsample if needed
            if subsample_size and adata_clustered.n_obs > subsample_size:
                logger.info(f"Subsampling data to {subsample_size} cells (from {adata_clustered.n_obs})")
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
                        logger.info(f"Found {len(unique_batches)} batches for correction: {list(unique_batches)}")
                        adata_clustered = self._perform_batch_correction(adata_clustered, batch_key)
                    else:
                        logger.info("Only one batch found - proceeding without batch correction")
                        batch_correction = False
                else:
                    logger.warning("Batch correction requested but no valid batch key found")
                    batch_correction = False
            
            self._update_progress("Batch correction completed" if batch_correction else "Batch check completed")

            # Perform clustering
            adata_clustered = self._perform_clustering(
                adata_clustered, resolution, demo_mode, skip_steps
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
                    str(cluster): int(count) for cluster, count in cluster_counts.items()
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
            if "rank_genes_groups" in adata_clustered.uns and "marker_genes" not in skip_steps:
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
            
            return adata_clustered, clustering_stats

        except Exception as e:
            logger.exception(f"Error during clustering: {e}")
            raise ClusteringError(f"Clustering failed: {str(e)}")

    def _perform_batch_correction(self, adata: anndata.AnnData, batch_key: str) -> anndata.AnnData:
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
            adata_corrected = anndata.concat(batch_list, label=batch_key, keys=unique_batches)
            
            logger.info(f"Batch correction completed for {len(unique_batches)} batches")
            return adata_corrected
            
        except Exception as e:
            logger.warning(f"Batch correction failed, proceeding without correction: {e}")
            return adata


    def _perform_clustering(
        self,
        adata: sc.AnnData,
        resolution: float,
        demo_mode: bool = False,
        skip_steps: Optional[list] = None,
    ) -> sc.AnnData:
        """
        Perform clustering on the AnnData object based on the publication workflow.

        Args:
            adata: AnnData object
            resolution: Clustering resolution parameter
            demo_mode: Whether to run in demo mode (faster with reduced quality)
            skip_steps: List of steps to skip (e.g., 'marker_genes')

        Returns:
            sc.AnnData: AnnData object with clustering results
        """
        logger.info("Performing clustering pipeline based on publication workflow")
        skip_steps = skip_steps or []
        start_time = time.time()

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

        # PCA (using 'arpack' SVD solver for better performance with sparse matrices)
        logger.info("Running PCA")
        sc.tl.pca(adata_hvg, svd_solver="arpack")
        self._update_progress("PCA completed")

        # Determine optimal number of PCs (following publication's approach using 20 PCs)
        # In demo mode, use fewer PCs for faster processing
        n_pcs = 10 if demo_mode else 20
        logger.info(f"Using {n_pcs} principal components for neighborhood graph")

        # Compute neighborhood graph
        logger.info("Computing neighborhood graph")
        n_neighbors = (
            10 if demo_mode else 15
        )  # Use fewer neighbors in demo mode for speed
        sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=n_pcs)
        self._update_progress("Neighborhood graph computed")

        # Run Leiden clustering at specified resolution (similar to publication's approach)
        logger.info(f"Running Leiden clustering with resolution {resolution}")
        sc.tl.leiden(adata_hvg, resolution=resolution, key_added="leiden")
        self._update_progress("Leiden clustering completed")

        # UMAP for visualization
        logger.info("Computing UMAP coordinates")
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
        Create UMAP plot from clustering results.

        Args:
            adata: AnnData object with clustering results

        Returns:
            go.Figure: Plotly figure with UMAP plot
        """
        logger.info("Creating UMAP visualization")

        umap_coords = adata.obsm["X_umap"]
        clusters = adata.obs["leiden"].astype(str)

        # Create a colormap similar to those in the publication
        n_clusters = len(adata.obs["leiden"].unique())
        if n_clusters <= 10:
            color_map = px.colors.qualitative.Set1
        elif n_clusters <= 20:
            color_map = px.colors.qualitative.Dark24
        else:
            color_map = px.colors.qualitative.Alphabet

        fig = px.scatter(
            x=umap_coords[:, 0],
            y=umap_coords[:, 1],
            color=clusters,
            title="UMAP Visualization with Leiden Clusters",
            labels={"x": "UMAP_1", "y": "UMAP_2", "color": "Cluster"},
            width=800,
            height=700,
            color_discrete_sequence=color_map,
        )

        fig.update_traces(marker=dict(size=4, opacity=0.8))
        fig.update_layout(
            legend_title="Cluster",
            font=dict(size=12),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=1.15,
                itemsizing="constant",
            ),
        )

        return fig

    def _create_batch_umap(self, adata: sc.AnnData, batch_key: str) -> go.Figure:
        """
        Create UMAP plot colored by batch.

        Args:
            adata: AnnData object with clustering results
            batch_key: Key in adata.obs containing batch information

        Returns:
            go.Figure: Plotly figure with batch-colored UMAP plot
        """
        logger.info(f"Creating batch-colored UMAP visualization using {batch_key}")

        umap_coords = adata.obsm["X_umap"]
        batches = adata.obs[batch_key].astype(str)

        # Create color map
        n_batches = len(adata.obs[batch_key].unique())
        if n_batches <= 10:
            color_map = px.colors.qualitative.Set2
        else:
            color_map = px.colors.qualitative.Alphabet

        fig = px.scatter(
            x=umap_coords[:, 0],
            y=umap_coords[:, 1],
            color=batches,
            title=f"UMAP Visualization by {batch_key}",
            labels={"x": "UMAP_1", "y": "UMAP_2", "color": batch_key},
            width=800,
            height=700,
            color_discrete_sequence=color_map,
        )

        fig.update_traces(marker=dict(size=4, opacity=0.8))
        fig.update_layout(
            legend_title=batch_key,
            font=dict(size=12),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=1.15,
                itemsizing="constant",
            ),
        )

        return fig

    def _create_cluster_distribution_plot(self, adata: sc.AnnData) -> go.Figure:
        """
        Create a cluster size distribution plot.

        Args:
            adata: AnnData object with clustering results

        Returns:
            go.Figure: Plotly figure with cluster distribution plot
        """
        logger.info("Creating cluster size distribution plot")

        # Get cluster counts
        cluster_counts = adata.obs["leiden"].value_counts().sort_index()

        # Create bar plot
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                marker_color="steelblue",
            )
        )

        fig.update_layout(
            title="Cluster Size Distribution",
            xaxis_title="Cluster",
            yaxis_title="Number of Cells",
            width=800,
            height=500,
            yaxis=dict(
                type="log"
                if max(cluster_counts) / min(cluster_counts) > 100
                else "linear"
            ),
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
                scores = adata.uns["rank_genes_groups"]["scores"][cluster][:5]
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
