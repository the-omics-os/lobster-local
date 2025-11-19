"""
Clustering service for single-cell RNA-seq data.

This service provides methods for clustering single-cell RNA-seq data
and generating visualizations of the results.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import anndata
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scanpy as sc
import scipy.sparse as spr
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.deviance import calculate_deviance
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
        self.default_cluster_resolution = 1.0  # Scanpy standard, balances granularity
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
        feature_selection_method: str = "deviance",
        n_features: int = 4000,
        min_mean: float = 0.0125,
        max_mean: float = 3.0,
        min_disp: float = 0.5,
        resolutions: Optional[list] = None,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for full clustering pipeline.

        Args:
            resolution: Leiden clustering resolution parameter
            n_neighbors: Number of neighbors for graph construction
            n_pcs: Number of principal components to use
            feature_selection_method: 'deviance' or 'hvg'
            n_features: Number of features to select (for deviance method)
            min_mean: Minimum mean for HVG selection (for hvg method)
            max_mean: Maximum mean for HVG selection (for hvg method)
            min_disp: Minimum dispersion for HVG selection (for hvg method)

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
            "resolutions": ParameterSpec(
                param_type="List[float]",
                papermill_injectable=True,
                default_value=resolutions if resolutions else [],
                required=False,
                validation_rule="all(r > 0 for r in resolutions)" if resolutions else None,
                description="List of resolution parameters for multi-resolution testing",
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

        # Jinja2 template with branching logic for feature selection
        if feature_selection_method == "deviance":
            code_template = """# Single-cell clustering pipeline with deviance-based feature selection
# Using deviance (binomial from multinomial null) - works on raw counts, no normalization bias
from lobster.utils.deviance import calculate_deviance

# 1. Feature selection BEFORE normalization (deviance method)
deviance_scores = calculate_deviance(adata.X)
n_features_to_select = min({{ n_features }}, len(deviance_scores))
top_deviance_idx = np.argsort(deviance_scores)[::-1][:n_features_to_select]

adata.var['highly_deviant'] = False
adata.var.iloc[top_deviance_idx, adata.var.columns.get_loc('highly_deviant')] = True
adata.var['deviance_score'] = deviance_scores

n_selected = sum(adata.var['highly_deviant'])
print(f"Selected {n_selected} highly deviant genes (deviance method)")

# 2. Normalize AFTER feature selection
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("Normalization complete")

# 3. Store raw data and subset to selected features
adata.raw = adata.copy()
adata = adata[:, adata.var['highly_deviant']]

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
            pipeline_steps = [
                "deviance_selection",
                "normalize",
                "scale",
                "pca",
                "neighbors",
                "leiden",
                "umap",
            ]
            description_method = "deviance"
            imports = ["import scanpy as sc", "import numpy as np"]
            parameters = {
                "resolution": resolution,
                "n_neighbors": n_neighbors,
                "n_pcs": n_pcs,
                "n_features": n_features,
                "feature_selection_method": feature_selection_method,
            }
        else:  # hvg method
            code_template = """# Single-cell clustering pipeline with HVG feature selection
# Using HVG (Seurat method) - works on normalized data

# 1. Normalize BEFORE feature selection (HVG method requires normalized data)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("Normalization complete")

# 2. Find highly variable genes AFTER normalization
sc.pp.highly_variable_genes(
    adata,
    min_mean={{ min_mean }},
    max_mean={{ max_mean }},
    min_disp={{ min_disp }},
    flavor='seurat'
)
n_hvg = sum(adata.var.highly_variable)
print(f"Identified {n_hvg} highly variable genes (HVG method)")

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
            pipeline_steps = [
                "normalize",
                "hvg",
                "scale",
                "pca",
                "neighbors",
                "leiden",
                "umap",
            ]
            description_method = "HVG"
            imports = ["import scanpy as sc", "import numpy as np"]
            parameters = {
                "resolution": resolution,
                "n_neighbors": n_neighbors,
                "n_pcs": n_pcs,
                "min_mean": min_mean,
                "max_mean": max_mean,
                "min_disp": min_disp,
                "feature_selection_method": feature_selection_method,
            }

        return AnalysisStep(
            operation="scanpy.tl.cluster_pipeline",
            tool_name="cluster_and_visualize",
            description=f"Full clustering pipeline: {description_method} + PCA + neighbors + Leiden (res={resolution}) + UMAP",
            library="scanpy",
            code_template=code_template,
            imports=imports,
            parameters=parameters,
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "clustering",
                "pipeline_steps": pipeline_steps,
                "feature_selection_method": feature_selection_method,
                "resolution": resolution,
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def cluster_and_visualize(
        self,
        adata: anndata.AnnData,
        resolution: Optional[float] = None,
        resolutions: Optional[list] = None,
        use_rep: Optional[str] = None,
        batch_correction: bool = False,
        batch_key: Optional[str] = None,
        demo_mode: bool = False,
        subsample_size: Optional[int] = None,
        skip_steps: Optional[list] = None,
        feature_selection_method: str = "deviance",
        n_features: int = 4000,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform clustering and UMAP visualization on single-cell RNA-seq data.

        Args:
            adata: AnnData object to cluster
            resolution: Resolution parameter for Leiden clustering (single value)
            resolutions: List of resolution parameters for multi-resolution testing
            use_rep: Representation to use for clustering (e.g., 'X_scvi', 'X_pca').
                    If None, uses standard PCA workflow. If specified, uses the
                    custom embedding from adata.obsm[use_rep] for neighbor calculation.
            batch_correction: Whether to perform batch correction
            batch_key: Column name for batch information
            demo_mode: Whether to run in demo mode (faster processing with reduced quality)
            subsample_size: Maximum number of cells to include (subsamples if larger)
            skip_steps: List of steps to skip in demo mode (e.g. ['marker_genes'])
            feature_selection_method: Method for feature selection ('deviance' or 'hvg').
                                     'deviance': Binomial deviance from multinomial null (default, recommended)
                                     'hvg': Traditional highly variable genes (Seurat method)
            n_features: Number of features to select (default: 4000)

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
                adata_clustered,
                resolution,
                use_rep,
                demo_mode,
                skip_steps,
                feature_selection_method,
                n_features,
                resolutions,
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

            # Add multi-resolution info if available
            if "clustering_results" in adata_clustered.uns:
                clustering_stats["clustering_results"] = adata_clustered.uns["clustering_results"]
                clustering_stats["resolutions_tested"] = adata_clustered.uns["resolutions_tested"]
                clustering_stats["n_resolutions"] = len(adata_clustered.uns["resolutions_tested"])

                # Add summary of clusters per resolution
                if len(adata_clustered.uns["resolutions_tested"]) > 1:
                    clustering_stats["multi_resolution_summary"] = {
                        res: adata_clustered.uns["clustering_results"][res]["n_clusters"]
                        for res in adata_clustered.uns["resolutions_tested"]
                    }

            # Add UMAP distance warning to stats
            if "umap_distance_warning" in adata_clustered.uns:
                clustering_stats["umap_distance_warning"] = adata_clustered.uns["umap_distance_warning"]

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
                resolution=resolution if resolution else self.default_cluster_resolution,
                n_neighbors=15,  # Default from _perform_clustering
                n_pcs=30,  # Updated default (was 20)
                feature_selection_method=feature_selection_method,
                n_features=n_features,
                resolutions=resolutions,
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
        feature_selection_method: str = "deviance",
        n_features: int = 4000,
        resolutions: Optional[list] = None,
    ) -> sc.AnnData:
        """
        Perform clustering on the AnnData object with configurable feature selection.

        Feature selection methods:
        - 'deviance': Selects features BEFORE normalization using binomial deviance (recommended)
        - 'hvg': Selects features AFTER normalization using Seurat HVG method (traditional)

        Args:
            adata: AnnData object
            resolution: Clustering resolution parameter
            use_rep: Representation to use for clustering (e.g., 'X_scvi', 'X_pca').
                    If None, uses standard PCA workflow. If specified, uses the
                    custom embedding from adata.obsm[use_rep] for neighbor calculation.
            demo_mode: Whether to run in demo mode (faster with reduced quality)
            skip_steps: List of steps to skip (e.g., 'marker_genes')
            feature_selection_method: 'deviance' (default) or 'hvg'
            n_features: Number of features to select (default: 4000)

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

            # Feature selection: deviance (recommended) or HVG (traditional)
            logger.info(
                f"Using '{feature_selection_method}' feature selection method with n_features={n_features}"
            )

            if feature_selection_method == "deviance":
                # DEVIANCE-BASED FEATURE SELECTION (BEFORE NORMALIZATION)
                # Step 1: Select features on raw counts using deviance
                logger.info(
                    "Selecting features using deviance (on raw counts, before normalization)"
                )
                deviance_scores = calculate_deviance(adata.X)

                # Select top n_features by deviance
                n_features_to_select = min(n_features, len(deviance_scores))
                top_deviance_idx = np.argsort(deviance_scores)[::-1][
                    :n_features_to_select
                ]

                # Mark selected features
                adata.var["highly_deviant"] = False
                adata.var.iloc[top_deviance_idx, adata.var.columns.get_loc("highly_deviant")] = True
                adata.var["deviance_score"] = deviance_scores

                n_selected = sum(adata.var["highly_deviant"])
                logger.info(f"Selected {n_selected} highly deviant genes")
                self._update_progress(
                    f"Deviance-based feature selection completed ({n_selected} genes)"
                )

                # Step 2: Normalize data (AFTER feature selection)
                logger.info("Normalizing data (after feature selection)")
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                self._update_progress("Normalization completed")

                # Store raw data before scaling
                adata.raw = adata.copy()

                # Step 3: Subset to selected features
                if n_selected == 0:
                    raise ClusteringError(
                        "No highly deviant genes detected. This typically indicates:\n"
                        "1. Data has very low variance (all genes have similar expression)\n"
                        "2. Dataset is too small or uniform for feature selection\n"
                        "3. Data may need quality filtering before clustering\n\n"
                        "Suggestions:\n"
                        "- Check data quality (use quality_service for QC metrics)\n"
                        "- Ensure data is raw counts (not already normalized/log-transformed)\n"
                        "- Try filtering out low-quality cells/genes first"
                    )

                # In demo mode, further restrict features
                if demo_mode and n_selected > 1000:
                    logger.info("Demo mode: Restricting to top 1000 deviant genes")
                    most_deviant_genes = (
                        adata.var.sort_values("deviance_score", ascending=False)
                        .head(1000)
                        .index
                    )
                    adata_selected = adata[:, most_deviant_genes]
                else:
                    adata_selected = adata[:, adata.var["highly_deviant"]]

            elif feature_selection_method == "hvg":
                # TRADITIONAL HVG FEATURE SELECTION (AFTER NORMALIZATION)
                # Step 1: Normalize FIRST (required for HVG method)
                logger.info("Normalizing data (before HVG selection)")
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                self._update_progress("Normalization completed")

                # Step 2: Find highly variable genes on normalized data
                logger.info("Finding highly variable genes (after normalization)")
                sc.pp.highly_variable_genes(
                    adata, min_mean=0.0125, max_mean=3, min_disp=0.5, flavor="seurat"
                )
                self._update_progress("Highly variable genes identified")

                # Store raw data before scaling
                adata.raw = adata.copy()

                # Step 3: Subset to selected features
                n_hvg = sum(adata.var.highly_variable)
                logger.info(f"Using {n_hvg} highly variable genes")

                if n_hvg == 0:
                    raise ClusteringError(
                        "No highly variable genes detected. This typically indicates:\n"
                        "1. Data has very low variance (all genes have similar expression)\n"
                        "2. Dataset is too small or uniform for HVG detection\n"
                        "3. Data may need quality filtering before clustering\n\n"
                        "Suggestions:\n"
                        "- Check data quality (use quality_service for QC metrics)\n"
                        "- Ensure data is raw counts (not already normalized/log-transformed)\n"
                        "- Try filtering out low-quality cells/genes first\n"
                        "- Consider using 'deviance' method instead (works on raw counts, no normalization bias)"
                    )

                # In demo mode, further restrict features
                if demo_mode and n_hvg > 1000:
                    logger.info("Demo mode: Restricting to top 1000 variable genes")
                    most_variable_genes = (
                        adata.var.sort_values("dispersions_norm", ascending=False)
                        .head(1000)
                        .index
                    )
                    adata_selected = adata[:, most_variable_genes]
                else:
                    adata_selected = adata[:, adata.var.highly_variable]

            else:
                raise ClusteringError(
                    f"Unknown feature_selection_method: '{feature_selection_method}'. "
                    f"Must be 'deviance' or 'hvg'."
                )

            # Scale data (with max value cap to avoid influence of outliers)
            logger.info("Scaling data")
            sc.pp.scale(adata_selected, max_value=10)
            self._update_progress("Data scaling completed")

            # Determine optimal number of PCs (best practice: 30 PCs capture more variance)
            # In demo mode, use fewer PCs for faster processing
            n_pcs = 15 if demo_mode else 30  # Best practice: 30 PCs capture more variance

            # ISSUE #7 FIX: Validate n_pcs against available features
            n_features = adata_selected.n_vars
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
            sc.tl.pca(adata_selected, svd_solver="arpack", n_comps=n_pcs)
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
                sc.pp.neighbors(adata_selected, n_neighbors=n_neighbors, n_pcs=n_pcs)

            self._update_progress("Neighborhood graph computed")

            # Run Leiden clustering - support multi-resolution testing
            # Determine which resolutions to test
            if resolutions is not None:
                # Multi-resolution testing mode
                resolutions_to_test = resolutions
                logger.info(f"Testing {len(resolutions)} resolutions: {resolutions}")
            elif resolution is not None:
                # Single resolution mode (backward compatible)
                resolutions_to_test = [resolution]
            else:
                # Default resolution
                resolutions_to_test = [self.default_cluster_resolution]

            # Run Leiden clustering for each resolution
            clustering_results = {}
            for res in resolutions_to_test:
                # Create descriptive key name: leiden_res0_25, leiden_res0_5, etc.
                key_name = f"leiden_res{res}".replace(".", "_")

                logger.info(f"Running Leiden clustering at resolution {res} (key: {key_name})")

                with with_periodic_progress(
                    f"Running Leiden clustering (resolution={res})",
                    self._create_progress_callback(),
                    update_interval=10,
                    show_elapsed=True,
                ):
                    sc.tl.leiden(adata_selected, resolution=res, key_added=key_name)

                # Track cluster counts for this resolution
                n_clusters = adata_selected.obs[key_name].nunique()
                clustering_results[res] = {
                    "resolution": res,
                    "n_clusters": n_clusters,
                    "key_name": key_name,
                }
                logger.info(f"  → {n_clusters} clusters identified")

            # Store clustering results for stats
            adata_selected.uns["clustering_results"] = clustering_results
            adata_selected.uns["resolutions_tested"] = resolutions_to_test

            # For backward compatibility, also create 'leiden' column pointing to primary resolution
            primary_resolution = resolutions_to_test[0]
            primary_key = f"leiden_res{primary_resolution}".replace(".", "_")
            adata_selected.obs["leiden"] = adata_selected.obs[primary_key]

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
                    sc.tl.umap(adata_selected, min_dist=0.5, spread=1.5)
                else:
                    sc.tl.umap(adata_selected)

            # IMPORTANT WARNING about UMAP distances
            logger.warning(
                "UMAP visualization computed. IMPORTANT: Distances between clusters in UMAP "
                "space are NOT biologically meaningful. UMAP preserves local structure but "
                "distorts global distances. Use for visual exploration only, not quantitative analysis."
            )

            # Store warning in adata for downstream tools
            adata_selected.uns["umap_distance_warning"] = (
                "Distances between clusters in UMAP are not biologically meaningful. "
                "UMAP is optimized for local neighborhood preservation, not global distance relationships."
            )

            self._update_progress("UMAP coordinates computed")

            # Transfer clustering results and UMAP coordinates back to the original object
            adata.obs["leiden"] = adata_selected.obs["leiden"]
            adata.obsm["X_umap"] = adata_selected.obsm["X_umap"]

            # Transfer multi-resolution results if available
            if "clustering_results" in adata_selected.uns:
                adata.uns["clustering_results"] = adata_selected.uns["clustering_results"]
                adata.uns["resolutions_tested"] = adata_selected.uns["resolutions_tested"]
                # Transfer all resolution columns to main adata
                for res in adata_selected.uns["resolutions_tested"]:
                    key_name = f"leiden_res{res}".replace(".", "_")
                    if key_name in adata_selected.obs.columns:
                        adata.obs[key_name] = adata_selected.obs[key_name]

            # Transfer UMAP warning
            if "umap_distance_warning" in adata_selected.uns:
                adata.uns["umap_distance_warning"] = adata_selected.uns["umap_distance_warning"]

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

    def compute_clustering_quality(
        self,
        adata: anndata.AnnData,
        cluster_key: str = "leiden",
        use_rep: str = "X_pca",
        n_pcs: Optional[int] = None,
        metrics: Optional[List[str]] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Compute clustering quality metrics for evaluation and resolution selection.

        Calculates multiple metrics to assess clustering quality:
        - Silhouette score: How well-separated clusters are (-1 to 1, higher better)
        - Davies-Bouldin index: Ratio of intra/inter-cluster distances (lower better)
        - Calinski-Harabasz score: Ratio of between/within variance (higher better)

        These metrics help answer:
        - "Is my clustering good?"
        - "Which resolution gives the best separation?"
        - "Am I over-clustering or under-clustering?"

        Args:
            adata: AnnData object with clustering results
            cluster_key: Key in adata.obs containing cluster labels (default: "leiden")
            use_rep: Representation to use for distance calculations (default: "X_pca")
            n_pcs: Number of PCs to use (default: None = use all available)
            metrics: List of metrics to compute (default: None = all metrics)
                    Options: ["silhouette", "davies_bouldin", "calinski_harabasz"]

        Returns:
            Tuple of (adata, stats_dict, analysis_step_ir)

            stats_dict contains:
            - silhouette_score: Overall score (-1 to 1)
            - davies_bouldin_index: Overall index (lower better)
            - calinski_harabasz_score: Overall score (higher better)
            - n_clusters: Number of clusters evaluated
            - cluster_sizes: Dict mapping cluster ID to cell count
            - interpretation: Human-readable quality assessment
            - recommendations: Suggested next steps based on metrics

        Examples:
            # Evaluate single clustering result
            result, stats, ir = service.compute_clustering_quality(adata, cluster_key="leiden")

            # Compare multiple resolutions
            for res in [0.25, 0.5, 1.0]:
                result, stats, ir = service.compute_clustering_quality(
                    adata, cluster_key=f"leiden_res{str(res).replace('.', '_')}"
                )
                print(f"Resolution {res}: Silhouette={stats['silhouette_score']:.3f}")
        """
        try:
            logger.info(f"Computing clustering quality metrics for '{cluster_key}'")
            start_time = time.time()

            # Step 1: Copy input
            adata = adata.copy()

            # Step 2: Validation
            if cluster_key not in adata.obs.columns:
                raise ValueError(
                    f"Cluster key '{cluster_key}' not found in adata.obs. "
                    f"Available keys: {list(adata.obs.columns)}"
                )

            # Validate representation exists
            if use_rep == "X_pca" and "X_pca" not in adata.obsm:
                raise ValueError("X_pca not found in adata.obsm. Run PCA first.")
            elif use_rep != "X_pca" and use_rep not in adata.obsm:
                raise ValueError(
                    f"Representation '{use_rep}' not found in adata.obsm. "
                    f"Available representations: {list(adata.obsm.keys())}"
                )

            # Get cluster labels
            labels = adata.obs[cluster_key].values

            # Validate we have at least 2 clusters
            n_clusters = len(np.unique(labels))
            if n_clusters < 2:
                raise ValueError(
                    f"Need at least 2 clusters for quality metrics, found {n_clusters}. "
                    f"Consider using a higher resolution parameter."
                )

            logger.info(f"Evaluating quality for {n_clusters} clusters using '{use_rep}' representation")

            # Get data representation
            if n_pcs is not None:
                X = adata.obsm[use_rep][:, :n_pcs]
                logger.info(f"Using first {n_pcs} components from '{use_rep}'")
            else:
                X = adata.obsm[use_rep]
                logger.info(f"Using all {X.shape[1]} components from '{use_rep}'")

            # Step 3: Determine which metrics to compute
            if metrics is None:
                metrics_to_compute = ["silhouette", "davies_bouldin", "calinski_harabasz"]
            else:
                metrics_to_compute = metrics
                # Validate metric names
                valid_metrics = ["silhouette", "davies_bouldin", "calinski_harabasz"]
                invalid = set(metrics_to_compute) - set(valid_metrics)
                if invalid:
                    raise ValueError(
                        f"Invalid metrics: {invalid}. "
                        f"Valid options: {valid_metrics}"
                    )

            logger.info(f"Computing metrics: {metrics_to_compute}")

            # Step 4: Compute Metrics
            results = {}

            # Silhouette score (-1 to 1, higher better)
            if "silhouette" in metrics_to_compute:
                logger.info("Computing silhouette score...")
                results["silhouette_score"] = float(
                    silhouette_score(X, labels, metric="euclidean")
                )
                logger.info(f"  → Silhouette score: {results['silhouette_score']:.4f}")

            # Davies-Bouldin index (0 to ∞, lower better)
            if "davies_bouldin" in metrics_to_compute:
                logger.info("Computing Davies-Bouldin index...")
                results["davies_bouldin_index"] = float(
                    davies_bouldin_score(X, labels)
                )
                logger.info(f"  → Davies-Bouldin index: {results['davies_bouldin_index']:.4f}")

            # Calinski-Harabasz score (0 to ∞, higher better)
            if "calinski_harabasz" in metrics_to_compute:
                logger.info("Computing Calinski-Harabasz score...")
                results["calinski_harabasz_score"] = float(
                    calinski_harabasz_score(X, labels)
                )
                logger.info(f"  → Calinski-Harabasz score: {results['calinski_harabasz_score']:.2f}")

            # Step 5: Compute Per-Cluster Statistics
            cluster_sizes = dict(adata.obs[cluster_key].value_counts().sort_index())
            cluster_sizes = {str(k): int(v) for k, v in cluster_sizes.items()}

            # Per-cluster silhouette scores (for detailed analysis)
            per_cluster_silhouette = {}
            if "silhouette" in metrics_to_compute:
                logger.info("Computing per-cluster silhouette scores...")
                silhouette_vals = silhouette_samples(X, labels, metric="euclidean")
                for cluster_id in np.unique(labels):
                    mask = labels == cluster_id
                    per_cluster_silhouette[str(cluster_id)] = float(
                        np.mean(silhouette_vals[mask])
                    )

            # Step 6: Generate Interpretation
            interpretation = []

            # Silhouette interpretation
            if "silhouette" in metrics_to_compute:
                sil = results["silhouette_score"]
                if sil > 0.7:
                    interpretation.append(
                        f"Silhouette score {sil:.3f}: EXCELLENT cluster separation"
                    )
                elif sil > 0.5:
                    interpretation.append(
                        f"Silhouette score {sil:.3f}: GOOD cluster separation"
                    )
                elif sil > 0.25:
                    interpretation.append(
                        f"Silhouette score {sil:.3f}: MODERATE cluster separation"
                    )
                else:
                    interpretation.append(
                        f"Silhouette score {sil:.3f}: POOR cluster separation (consider different resolution)"
                    )

            # Davies-Bouldin interpretation
            if "davies_bouldin" in metrics_to_compute:
                db = results["davies_bouldin_index"]
                if db < 1.0:
                    interpretation.append(
                        f"Davies-Bouldin index {db:.3f}: GOOD cluster compactness"
                    )
                elif db < 2.0:
                    interpretation.append(
                        f"Davies-Bouldin index {db:.3f}: MODERATE cluster compactness"
                    )
                else:
                    interpretation.append(
                        f"Davies-Bouldin index {db:.3f}: POOR cluster compactness"
                    )

            # Calinski-Harabasz interpretation
            if "calinski_harabasz" in metrics_to_compute:
                ch = results["calinski_harabasz_score"]
                if ch > 1000:
                    interpretation.append(
                        f"Calinski-Harabasz score {ch:.1f}: HIGH variance ratio"
                    )
                elif ch > 100:
                    interpretation.append(
                        f"Calinski-Harabasz score {ch:.1f}: MODERATE variance ratio"
                    )
                else:
                    interpretation.append(
                        f"Calinski-Harabasz score {ch:.1f}: LOW variance ratio"
                    )

            # Step 7: Generate Recommendations
            recommendations = []

            # Overall quality assessment
            if "silhouette" in metrics_to_compute:
                sil = results["silhouette_score"]
                if sil < 0.25:
                    recommendations.append(
                        "⚠️ Low silhouette score suggests over-clustering. Try lower resolution."
                    )
                elif sil > 0.7:
                    recommendations.append(
                        "✓ Excellent separation. This resolution works well."
                    )

            # Cluster count check
            if n_clusters > 50:
                recommendations.append(
                    "⚠️ Very high cluster count. Consider using lower resolution or sub-clustering specific populations."
                )
            elif n_clusters < 5:
                recommendations.append(
                    "ℹ️ Low cluster count. Consider higher resolution for finer-grained analysis."
                )

            # Cluster size balance
            min_size = min(cluster_sizes.values())
            max_size = max(cluster_sizes.values())
            if max_size / min_size > 100:
                recommendations.append(
                    "⚠️ Highly imbalanced cluster sizes. Smallest clusters may be artifacts or rare populations."
                )

            # If no recommendations, provide positive feedback
            if not recommendations:
                recommendations.append(
                    "✓ Clustering quality looks reasonable. Proceed with downstream analysis."
                )

            # Step 8: Store Metrics in AnnData
            quality_key = f"{cluster_key}_quality"
            adata.uns[quality_key] = {
                "silhouette_score": results.get("silhouette_score"),
                "davies_bouldin_index": results.get("davies_bouldin_index"),
                "calinski_harabasz_score": results.get("calinski_harabasz_score"),
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "per_cluster_silhouette": per_cluster_silhouette,
                "interpretation": interpretation,
                "recommendations": recommendations,
                "use_rep": use_rep,
                "n_pcs_used": n_pcs,
            }

            logger.info(f"Quality metrics stored in adata.uns['{quality_key}']")

            # Step 9: Build Stats Dict
            execution_time = time.time() - start_time

            stats = {
                "cluster_key": cluster_key,
                "n_clusters": n_clusters,
                "n_cells": adata.n_obs,
                "use_rep": use_rep,
                "n_pcs_used": n_pcs if n_pcs else adata.obsm[use_rep].shape[1],
                "cluster_sizes": cluster_sizes,
                "interpretation": "\n".join(interpretation),
                "recommendations": recommendations,
                "execution_time_seconds": round(execution_time, 2),
            }

            # Add metrics
            for metric_name, value in results.items():
                stats[metric_name] = value

            # Add per-cluster silhouette
            if per_cluster_silhouette:
                stats["per_cluster_silhouette"] = per_cluster_silhouette

            logger.info(f"Clustering quality metrics computed successfully in {execution_time:.2f}s")

            # Step 10: Create IR
            ir = self._create_quality_metrics_ir(
                cluster_key=cluster_key,
                use_rep=use_rep,
                n_pcs=n_pcs,
                metrics=metrics_to_compute,
            )

            return adata, stats, ir

        except Exception as e:
            logger.exception(f"Error computing clustering quality metrics: {e}")
            raise ClusteringError(f"Quality metrics computation failed: {str(e)}")

    def _create_quality_metrics_ir(
        self,
        cluster_key: str,
        use_rep: str,
        n_pcs: Optional[int],
        metrics: List[str],
    ) -> AnalysisStep:
        """
        Create IR for clustering quality metrics.

        Args:
            cluster_key: Key in adata.obs containing cluster labels
            use_rep: Representation used for distance calculations
            n_pcs: Number of PCs used (None = all)
            metrics: List of metrics computed

        Returns:
            AnalysisStep with reproducible code for quality metrics
        """
        parameter_schema = {
            "cluster_key": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=cluster_key,
                required=True,
                description="Key in adata.obs containing cluster labels",
            ),
            "use_rep": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=use_rep,
                required=False,
                validation_rule=f"use_rep in adata.obsm.keys()",
                description="Representation to use for distance calculations",
            ),
            "n_pcs": ParameterSpec(
                param_type="Optional[int]",
                papermill_injectable=True,
                default_value=n_pcs,
                required=False,
                validation_rule="n_pcs > 0 if n_pcs else True",
                description="Number of PCs to use for calculations",
            ),
            "metrics": ParameterSpec(
                param_type="List[str]",
                papermill_injectable=True,
                default_value=metrics,
                required=False,
                description="List of metrics to compute",
            ),
        }

        code_template = '''# Compute clustering quality metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np

# Get cluster labels
labels = adata.obs["{{ cluster_key }}"].values

# Get data representation
{% if n_pcs %}
X = adata.obsm["{{ use_rep }}"][:, :{{ n_pcs }}]
{% else %}
X = adata.obsm["{{ use_rep }}"]
{% endif %}

# Compute metrics
quality_results = {}

{% if "silhouette" in metrics %}
quality_results["silhouette_score"] = silhouette_score(X, labels, metric="euclidean")
print(f"Silhouette score: {quality_results['silhouette_score']:.4f}")
{% endif %}

{% if "davies_bouldin" in metrics %}
quality_results["davies_bouldin_index"] = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin index: {quality_results['davies_bouldin_index']:.4f}")
{% endif %}

{% if "calinski_harabasz" in metrics %}
quality_results["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz score: {quality_results['calinski_harabasz_score']:.2f}")
{% endif %}

# Store results
adata.uns["{{ cluster_key }}_quality"] = quality_results
print(f"Quality metrics computed for {len(np.unique(labels))} clusters")
'''

        return AnalysisStep(
            operation="sklearn.metrics.clustering_quality",
            tool_name="compute_clustering_quality",
            description=f"Compute clustering quality metrics for {cluster_key}",
            library="sklearn.metrics",
            code_template=code_template,
            imports=[
                "from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score",
                "import numpy as np",
            ],
            parameters={
                "cluster_key": cluster_key,
                "use_rep": use_rep,
                "n_pcs": n_pcs,
                "metrics": metrics,
            },
            parameter_schema=parameter_schema,
            input_entities=["clustered_data"],
            output_entities=["quality_metrics"],
            execution_context={"method": "clustering_quality_assessment"},
            validates_on_export=True,
            requires_validation=False,
        )

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

    def subcluster_cells(
        self,
        adata: anndata.AnnData,
        cluster_key: str = "leiden",
        clusters_to_refine: Optional[List[str]] = None,
        resolution: float = 0.5,
        resolutions: Optional[List[float]] = None,
        n_pcs: int = 20,
        n_neighbors: int = 15,
        batch_key: Optional[str] = None,
        demo_mode: bool = False,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Re-cluster specific cell subsets for finer-grained population identification.

        Sub-clustering is useful when:
        - Initial clustering groups heterogeneous populations
        - You want to refine specific clusters without affecting others
        - You need higher resolution for specific cell types

        Args:
            adata: AnnData object with existing clustering results
            cluster_key: Key in adata.obs containing cluster assignments (default: "leiden")
            clusters_to_refine: List of cluster IDs to re-cluster (e.g., ["0", "3", "5"])
                               If None, re-clusters ALL cells (full re-clustering)
            resolution: Single resolution for sub-clustering (default: 0.5)
            resolutions: List of resolutions for multi-resolution sub-clustering
            n_pcs: Number of PCs for sub-clustering (default: 20, fewer than full clustering)
            n_neighbors: Number of neighbors for KNN graph (default: 15)
            batch_key: Optional batch correction key for sub-clustering
            demo_mode: If True, uses faster parameters for testing

        Returns:
            Tuple of (processed_adata, stats_dict, analysis_step_ir)

        Raises:
            ValueError: If cluster_key not found or invalid cluster IDs provided
            ClusteringError: If sub-clustering fails
        """
        try:
            logger.info("Starting sub-clustering pipeline")
            start_time = time.time()

            # Create working copy
            adata = adata.copy()

            # Step 1: Validation
            if cluster_key not in adata.obs.columns:
                raise ValueError(
                    f"Cluster key '{cluster_key}' not found in adata.obs. "
                    f"Available keys: {list(adata.obs.columns)}"
                )

            # Get unique clusters
            all_clusters = set(adata.obs[cluster_key].astype(str).unique())
            logger.info(f"Found {len(all_clusters)} clusters in '{cluster_key}': {sorted(all_clusters)}")

            # Validate clusters_to_refine if provided
            if clusters_to_refine is not None and len(clusters_to_refine) > 0:
                clusters_to_refine = [str(c) for c in clusters_to_refine]  # Ensure string type
                invalid_clusters = set(clusters_to_refine) - all_clusters
                if invalid_clusters:
                    raise ValueError(
                        f"Invalid cluster IDs: {sorted(invalid_clusters)}. "
                        f"Valid clusters: {sorted(all_clusters)}"
                    )
                logger.info(f"Sub-clustering {len(clusters_to_refine)} parent clusters: {clusters_to_refine}")
            else:
                # Re-cluster all cells (None or empty list)
                clusters_to_refine = sorted(all_clusters)
                logger.info("No specific clusters provided - re-clustering ALL cells")

            # Check for PCA results
            if "X_pca" not in adata.obsm:
                raise ValueError(
                    "PCA results not found in adata.obsm['X_pca']. "
                    "Please run clustering first to generate PCA coordinates."
                )

            # Demo mode adjustments
            if demo_mode:
                n_pcs = min(n_pcs, 10)
                n_neighbors = min(n_neighbors, 10)
                logger.info(f"Demo mode: Using n_pcs={n_pcs}, n_neighbors={n_neighbors}")

            # Step 2: Subset Selection
            subset_mask = adata.obs[cluster_key].astype(str).isin(clusters_to_refine)
            n_cells_to_subcluster = subset_mask.sum()
            logger.info(f"Selected {n_cells_to_subcluster} cells from {len(clusters_to_refine)} clusters for sub-clustering")

            if n_cells_to_subcluster == 0:
                raise ValueError("No cells selected for sub-clustering - check cluster IDs")

            # Create subset
            adata_subset = adata[subset_mask].copy()

            # Compute PCA on subset (required for sc.pp.neighbors)
            logger.info(f"Computing PCA on subset (n_comps={n_pcs})")
            sc.tl.pca(adata_subset, n_comps=n_pcs)

            # Step 3: Sub-clustering Execution
            logger.info(f"Computing neighbors on subset (n_neighbors={n_neighbors}, n_pcs={n_pcs})")
            sc.pp.neighbors(adata_subset, n_neighbors=n_neighbors, n_pcs=n_pcs)

            # Determine resolutions to test
            if resolutions is not None:
                resolutions_to_test = resolutions
                logger.info(f"Testing {len(resolutions)} resolutions: {resolutions}")
            elif resolution is not None:
                resolutions_to_test = [resolution]
                logger.info(f"Using single resolution: {resolution}")
            else:
                resolutions_to_test = [0.5]  # Default
                logger.info("Using default resolution: 0.5")

            # Run Leiden clustering for each resolution
            subclustering_results = {}
            for res in resolutions_to_test:
                # Create key name
                if len(resolutions_to_test) == 1:
                    key_name = "leiden_subcluster"
                else:
                    key_name = f"leiden_sub_res{res}".replace(".", "_")

                logger.info(f"Running Leiden sub-clustering at resolution {res} (key: {key_name})")
                sc.tl.leiden(adata_subset, resolution=res, key_added=key_name)

                # Convert categorical to string to allow new categories
                adata_subset.obs[key_name] = adata_subset.obs[key_name].astype(str)

                # Add parent cluster prefix to sub-cluster labels
                for parent_cluster in clusters_to_refine:
                    parent_mask = adata_subset.obs[cluster_key].astype(str) == parent_cluster
                    if parent_mask.sum() > 0:
                        # Get sub-cluster assignments for this parent
                        sub_labels = adata_subset.obs.loc[parent_mask, key_name]
                        # Add prefix: parent_cluster + "." + sub_label
                        prefixed_labels = sub_labels.apply(lambda x: f"{parent_cluster}.{x}")
                        adata_subset.obs.loc[parent_mask, key_name] = prefixed_labels

                # Count sub-clusters per parent
                n_subclusters_per_parent = {}
                for parent_cluster in clusters_to_refine:
                    parent_mask = adata_subset.obs[cluster_key].astype(str) == parent_cluster
                    if parent_mask.sum() > 0:
                        subclusters = adata_subset.obs.loc[parent_mask, key_name].nunique()
                        n_subclusters_per_parent[parent_cluster] = subclusters

                total_subclusters = adata_subset.obs[key_name].nunique()
                subclustering_results[res] = {
                    "resolution": res,
                    "key_name": key_name,
                    "n_total_subclusters": total_subclusters,
                    "n_subclusters_per_parent": n_subclusters_per_parent,
                }
                logger.info(f"  → {total_subclusters} total sub-clusters identified")

            # Step 4: Merge Back to Original AnnData
            logger.info("Merging sub-cluster results back to original AnnData")

            for res_key, res_data in subclustering_results.items():
                key_name = res_data["key_name"]
                # Create new column in original adata
                adata.obs[key_name] = adata.obs[cluster_key].astype(str)

                # Update cells that were sub-clustered
                adata.obs.loc[subset_mask, key_name] = adata_subset.obs[key_name]

            # Step 5: Statistics and IR
            execution_time = time.time() - start_time

            # Primary key (for backward compatibility)
            primary_key = subclustering_results[resolutions_to_test[0]]["key_name"]

            stats = {
                "analysis_type": "sub-clustering",
                "cluster_key": cluster_key,
                "parent_clusters": clusters_to_refine,
                "n_cells_subclustered": n_cells_to_subcluster,
                "n_total_cells": adata.n_obs,
                "resolution_used": resolution if resolutions is None else None,
                "resolutions_tested": resolutions_to_test,
                "n_pcs_used": n_pcs,
                "n_neighbors_used": n_neighbors,
                "execution_time_seconds": round(execution_time, 2),
                "subclustering_results": subclustering_results,
                "primary_subcluster_key": primary_key,
            }

            # Add cluster size summary
            cluster_sizes = {}
            for res_key, res_data in subclustering_results.items():
                key_name = res_data["key_name"]
                cluster_counts = adata.obs[key_name].value_counts().to_dict()
                cluster_sizes[key_name] = {
                    str(cluster): int(count) for cluster, count in cluster_counts.items()
                }
            stats["cluster_sizes"] = cluster_sizes

            logger.info(f"Sub-clustering completed in {execution_time:.2f} seconds")

            # Create IR
            ir = self._create_subcluster_ir(
                cluster_key=cluster_key,
                clusters_to_refine=clusters_to_refine,
                resolution=resolution if resolutions is None else None,
                resolutions=resolutions,
                n_pcs=n_pcs,
                n_neighbors=n_neighbors,
            )

            return adata, stats, ir

        except Exception as e:
            logger.exception(f"Error during sub-clustering: {e}")
            raise ClusteringError(f"Sub-clustering failed: {str(e)}")

    def _create_subcluster_ir(
        self,
        cluster_key: str = "leiden",
        clusters_to_refine: Optional[List[str]] = None,
        resolution: Optional[float] = None,
        resolutions: Optional[List[float]] = None,
        n_pcs: int = 20,
        n_neighbors: int = 15,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for sub-clustering operation.

        Args:
            cluster_key: Key in adata.obs containing cluster assignments
            clusters_to_refine: List of cluster IDs to re-cluster
            resolution: Single resolution parameter
            resolutions: List of resolution parameters
            n_pcs: Number of principal components to use
            n_neighbors: Number of neighbors for graph construction

        Returns:
            AnalysisStep with sub-clustering code template
        """
        # Parameter schema
        parameter_schema = {
            "cluster_key": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=cluster_key,
                required=True,
                description="Key in adata.obs containing cluster assignments",
            ),
            "clusters_to_refine": ParameterSpec(
                param_type="List[str]",
                papermill_injectable=True,
                default_value=clusters_to_refine if clusters_to_refine else [],
                required=False,
                description="List of cluster IDs to re-cluster (empty list = all clusters)",
            ),
            "resolution": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=resolution if resolution else 0.5,
                required=False,
                validation_rule="resolution > 0" if resolution else None,
                description="Resolution parameter for sub-clustering",
            ),
            "resolutions": ParameterSpec(
                param_type="List[float]",
                papermill_injectable=True,
                default_value=resolutions if resolutions else [],
                required=False,
                validation_rule="all(r > 0 for r in resolutions)" if resolutions else None,
                description="List of resolution parameters for multi-resolution sub-clustering",
            ),
            "n_pcs": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_pcs,
                required=False,
                validation_rule="n_pcs > 0",
                description="Number of principal components to use",
            ),
            "n_neighbors": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=n_neighbors,
                required=False,
                validation_rule="n_neighbors > 0",
                description="Number of neighbors for graph construction",
            ),
        }

        # Jinja2 code template
        code_template = """# Sub-clustering pipeline: Re-cluster specific cell populations
import scanpy as sc
import numpy as np

# Configuration
cluster_key = {{ cluster_key|repr }}
clusters_to_refine = {{ clusters_to_refine|repr }}
resolution = {{ resolution }}
n_pcs = {{ n_pcs }}
n_neighbors = {{ n_neighbors }}

# Validate cluster key exists
if cluster_key not in adata.obs.columns:
    raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")

# Select cells to sub-cluster
if clusters_to_refine:
    subset_mask = adata.obs[cluster_key].astype(str).isin([str(c) for c in clusters_to_refine])
    print(f"Sub-clustering {len(clusters_to_refine)} parent clusters: {clusters_to_refine}")
else:
    subset_mask = adata.obs.index.isin(adata.obs.index)  # All cells
    print("Sub-clustering ALL cells")

n_cells_to_subcluster = subset_mask.sum()
print(f"Selected {n_cells_to_subcluster} cells for sub-clustering")

# Create subset
adata_subset = adata[subset_mask].copy()

# Compute neighbors on subset (using existing PCA from adata.obsm['X_pca'])
print(f"Computing neighbors (n_neighbors={n_neighbors}, n_pcs={n_pcs})")
sc.pp.neighbors(adata_subset, n_neighbors=n_neighbors, n_pcs=n_pcs)

# Run Leiden clustering
print(f"Running Leiden sub-clustering (resolution={resolution})")
sc.tl.leiden(adata_subset, resolution=resolution, key_added='leiden_subcluster')

# Add parent cluster prefix to sub-cluster labels
if clusters_to_refine:
    for parent_cluster in clusters_to_refine:
        parent_mask = adata_subset.obs[cluster_key].astype(str) == str(parent_cluster)
        if parent_mask.sum() > 0:
            sub_labels = adata_subset.obs.loc[parent_mask, 'leiden_subcluster'].astype(str)
            prefixed_labels = sub_labels.apply(lambda x: f"{parent_cluster}.{x}")
            adata_subset.obs.loc[parent_mask, 'leiden_subcluster'] = prefixed_labels

# Merge back to original AnnData
adata.obs['leiden_subcluster'] = adata.obs[cluster_key].astype(str)
adata.obs.loc[subset_mask, 'leiden_subcluster'] = adata_subset.obs['leiden_subcluster']

n_subclusters = adata.obs['leiden_subcluster'].nunique()
print(f"Sub-clustering complete: {n_subclusters} sub-clusters identified")
"""

        return AnalysisStep(
            operation="scanpy.tl.leiden",
            tool_name="subcluster_cells",
            description=f"Sub-cluster cells from {len(clusters_to_refine) if clusters_to_refine else 'all'} parent cluster(s) at resolution {resolution if resolution else resolutions}",
            library="scanpy",
            code_template=code_template,
            imports=["import scanpy as sc", "import numpy as np"],
            parameters={
                "cluster_key": cluster_key,
                "clusters_to_refine": clusters_to_refine if clusters_to_refine else [],
                "resolution": resolution if resolution else 0.5,
                "resolutions": resolutions if resolutions else [],
                "n_pcs": n_pcs,
                "n_neighbors": n_neighbors,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "sub-clustering",
                "cluster_key": cluster_key,
                "n_parent_clusters": len(clusters_to_refine) if clusters_to_refine else "all",
            },
            validates_on_export=True,
            requires_validation=False,
        )

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
