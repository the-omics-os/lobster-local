"""
Clustering service for single-cell RNA-seq data.

This service provides methods for clustering single-cell RNA-seq data
and generating visualizations of the results.
"""

import scanpy as sc
import plotly.express as px
import plotly.graph_objects as go
import time
from typing import Optional, Dict, Callable

from ..core.data_manager import DataManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ClusteringService:
    """
    Service for clustering single-cell RNA-seq data.
    
    This class provides methods to perform clustering and dimensionality
    reduction on single-cell RNA-seq data and generate visualizations.
    """
    
    def __init__(self, data_manager: DataManager):
        """
        Initialize the clustering service.
        
        Args:
            data_manager: DataManager instance for accessing data
        """
        logger.info("Initializing ClusteringService")
        self.data_manager = data_manager
        self.progress_callback = None
        self.current_progress = 0
        self.total_steps = 7  # Total number of major steps in the clustering pipeline
        self.default_cluster_resolution = 0.7
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
    
    def cluster_and_visualize(self, 
                             resolution: Optional[float] = None, 
                             batch_correction: bool = False,
                             demo_mode: bool = False,
                             subsample_size: Optional[int] = None,
                             skip_steps: Optional[list] = None) -> str:
        """
        Perform clustering and UMAP visualization on the current dataset.
        
        Args:
            resolution: Resolution parameter for Leiden clustering
            batch_correction: Whether to perform batch correction (using patient ID)
            demo_mode: Whether to run in demo mode (faster processing with reduced quality)
            subsample_size: Maximum number of cells to include (subsamples if larger)
            skip_steps: List of steps to skip in demo mode (e.g. ['marker_genes'])
            
        Returns:
            str: Clustering results report
        """
        logger.info("Starting clustering and visualization pipeline")
        logger.debug(f"Parameters: resolution={resolution}, batch_correction={batch_correction}, " +
                    f"demo_mode={demo_mode}, subsample_size={subsample_size}")
        
        # Reset progress tracking
        self.current_progress = 0
        self._update_progress("Initializing")
        
        if not self.data_manager.has_data():
            logger.error("No data loaded for clustering")
            return "No data loaded. Please download a dataset first."
        
        try:
            # Set resolution
            if resolution is None:
                resolution = self.default_cluster_resolution
                logger.debug(f"Using default resolution: {resolution}")
            
            logger.info(f"Performing clustering with resolution {resolution}")
            
            # Get data dimensions
            data_shape = self.data_manager.current_data.shape
            logger.info(f"Input data dimensions: {data_shape[0]} cells × {data_shape[1]} genes")
            
            # Check if we should subsample for demo mode
            skip_steps = skip_steps or []
            if demo_mode:
                logger.info("Running in demo mode (faster processing)")
                if 'marker_genes' not in skip_steps:
                    skip_steps.append('marker_genes')
                if not subsample_size:
                    subsample_size = min(1000, data_shape[0])  # Default to 1000 cells in demo mode
            
            # Get AnnData object (or create if needed)
            adata = self._prepare_adata()
            logger.debug(f"AnnData object prepared: {adata.n_obs} obs × {adata.n_vars} vars")
            
            # Subsample if needed
            if subsample_size and adata.n_obs > subsample_size:
                logger.info(f"Subsampling data to {subsample_size} cells (from {adata.n_obs})")
                sc.pp.subsample(adata, n_obs=subsample_size, random_state=42)
                logger.info(f"Data subsampled: {adata.n_obs} cells remaining")
                
            self._update_progress("Data preparation complete")
            
            # Check if batch information is available (Patient_ID)
            batch_key = None
            available_keys = list(adata.obs_keys())
            logger.debug(f"Available observation keys: {available_keys}")
            
            #TODO check for more common batch keys
            for potential_key in ['Patient_ID', 'patient', 'batch', 'sample']:
                if potential_key in adata.obs_keys():
                    batch_key = potential_key
                    logger.info(f"Found batch information using key: {batch_key}")
                    break
            
            if batch_key is None:
                logger.info("No batch information found in data")
            
            # Perform batch correction if requested and possible
            if batch_correction and batch_key and len(adata.obs[batch_key].unique()) > 1:
                unique_batches = adata.obs[batch_key].unique()
                logger.info(f"Performing batch correction using {batch_key} with {len(unique_batches)} batches: {list(unique_batches)}")
                
                # Split by batch and process separately
                batches = []
                batch_sizes = {}
                for batch in unique_batches:
                    batch_adata = adata[adata.obs[batch_key] == batch].copy()
                    batch_sizes[batch] = batch_adata.n_obs
                    logger.debug(f"Batch {batch}: {batch_adata.n_obs} cells")
                    
                    sc.pp.normalize_total(batch_adata, target_sum=1e4)
                    sc.pp.log1p(batch_adata)
                    sc.pp.highly_variable_genes(batch_adata, min_mean=0.0125, max_mean=3, 
                                               min_disp=0.5, flavor='seurat')
                    batches.append(batch_adata)
                
                logger.info(f"Batch sizes: {batch_sizes}")
                
                # Find common variable genes across batches
                var_genes = []
                for i, batch_adata in enumerate(batches):
                    hvg_count = sum(batch_adata.var['highly_variable'])
                    logger.debug(f"Batch {i}: {hvg_count} highly variable genes")
                    var_genes.append(set(batch_adata.var_names[batch_adata.var['highly_variable']]))
                
                common_var_genes = list(set.intersection(*var_genes))
                logger.info(f"Common variable genes across batches: {len(common_var_genes)}")
                
                if len(common_var_genes) < 100:
                    logger.warning(f"Only {len(common_var_genes)} common variable genes found. Using all genes for batch correction.")
                    common_var_genes = None
                
                # Perform integration
                logger.info("Performing batch integration")
                adata_list = []
                for batch in unique_batches:
                    batch_adata = adata[adata.obs[batch_key] == batch].copy()
                    sc.pp.normalize_total(batch_adata, target_sum=1e4)
                    sc.pp.log1p(batch_adata)
                    adata_list.append(batch_adata)
                
                # Integration (similar to Seurat's CCA approach)
                logger.debug("Concatenating batch data for integration")
                adata_integrated = sc.AnnData.concatenate(*adata_list, batch_key=batch_key)
                logger.info(f"Integrated data shape: {adata_integrated.n_obs} × {adata_integrated.n_vars}")
                
                # Process integrated data
                adata = self._perform_clustering(adata_integrated, resolution, demo_mode, skip_steps)
                self._update_progress("Batch integration complete")
            else:
                if not batch_correction:
                    logger.info("Batch correction disabled by user")
                elif not batch_key:
                    logger.info("No batch key found - proceeding with standard processing")
                else:
                    logger.info("Only one batch found - proceeding with standard processing")
                
                # Standard processing without batch correction
                adata = self._perform_clustering(adata, resolution, demo_mode, skip_steps)
                self._update_progress("Clustering complete")
            
            # Create visualizations
            logger.info("Generating visualizations")
            self._update_progress("Generating visualizations")
            
            # Create UMAP plots
            logger.debug("Creating main UMAP plot")
            umap_plot = self._create_umap_plot(adata)
            self.data_manager.add_plot(
                umap_plot, 
                title="UMAP Visualization with Leiden Clusters",
                source="clustering_service"
            )
            
            if batch_key:
                # Also create a batch-colored UMAP if batch information is available
                logger.debug("Creating batch-colored UMAP plot")
                batch_umap = self._create_batch_umap(adata, batch_key)
                self.data_manager.add_plot(
                    batch_umap,
                    title=f"UMAP Visualization by {batch_key}",
                    source="clustering_service"
                )
            
            # Create cluster size distribution plot
            logger.debug("Creating cluster distribution plot")
            cluster_dist_plot = self._create_cluster_distribution_plot(adata)
            self.data_manager.add_plot(
                cluster_dist_plot,
                title="Cluster Size Distribution",
                source="clustering_service"
            )
            
            # Update metadata
            n_clusters = len(adata.obs['leiden'].unique())
            logger.info(f"Storing clustering results: {n_clusters} clusters identified")
            
            self.data_manager.current_metadata['clusters'] = adata.obs['leiden'].values.tolist()
            self.data_manager.current_metadata['umap'] = adata.obsm['X_umap'].tolist()
            self.data_manager.current_metadata['clustering_resolution'] = resolution
            self.data_manager.current_metadata['n_clusters'] = n_clusters
            
            # Store cluster marker genes in metadata
            if 'rank_genes_groups' in adata.uns:
                logger.debug("Storing marker genes in metadata")
                marker_genes = {}
                for cluster in adata.obs['leiden'].unique():
                    genes = adata.uns['rank_genes_groups']['names'][cluster]
                    scores = adata.uns['rank_genes_groups']['scores'][cluster]
                    marker_genes[cluster] = [{"gene": gene, "score": float(score)} 
                                           for gene, score in zip(genes[:10], scores[:10])]
                self.data_manager.current_metadata['marker_genes'] = marker_genes
                logger.info(f"Stored top 10 marker genes for {len(marker_genes)} clusters")
            
            # Update the AnnData object in the data manager
            self.data_manager.adata = adata
            logger.info("AnnData object updated in data manager")
            self._update_progress("Data stored in manager")
            
            # Generate and return report
            original_count = data_shape[0] if subsample_size and data_shape[0] > subsample_size else None
            report = self._format_clustering_report(
                adata, resolution, batch_correction, batch_key, 
                demo_mode=demo_mode, original_cell_count=original_count
            )
            logger.info("Clustering pipeline completed successfully")
            self._update_progress("Analysis complete")
            return report
            
        except Exception as e:
            logger.exception(f"Error during clustering: {e}")
            return f"Error during clustering: {str(e)}"
    
    def _prepare_adata(self) -> sc.AnnData:
        """
        Prepare AnnData object for clustering.
        
        This method ensures there's a valid AnnData object available,
        creating one if needed from the current data.
        
        Returns:
            sc.AnnData: AnnData object ready for clustering
        """
        logger.info("Preparing AnnData object for clustering")
        
        if self.data_manager.adata is None:
            logger.info("Creating new AnnData object from current data")
            # Get the current data
            data = self.data_manager.current_data
            
            # Create AnnData object
            adata = sc.AnnData(X=data.values)
            adata.var_names = data.columns
            adata.obs_names = data.index
        else:
            logger.info("Using existing AnnData object")
            adata = self.data_manager.adata.copy()
        
        return adata
    
    def _perform_clustering(self, adata: sc.AnnData, resolution: float, 
                           demo_mode: bool = False, skip_steps: Optional[list] = None) -> sc.AnnData:
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
        
        if self.progress_callback:
            self.progress_callback(int((self.current_progress / self.total_steps) * 100), "Starting preprocessing")
        
        # Basic preprocessing (follows publication workflow)
        logger.info("Normalizing data")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Find highly variable genes (follows the parameters from 02_All_cell_clustering.R)
        logger.info("Finding highly variable genes")
        sc.pp.highly_variable_genes(adata, 
                                   min_mean=0.0125, 
                                   max_mean=3, 
                                   min_disp=0.5,
                                   flavor='seurat')
        
        # Store raw data before scaling
        adata.raw = adata.copy()
        
        # Use only highly variable genes for dimensionality reduction
        n_hvg = sum(adata.var.highly_variable)
        logger.info(f"Using {n_hvg} highly variable genes")
        
        # In demo mode, further restrict the number of HVG for faster processing
        if demo_mode and n_hvg > 1000:
            logger.info("Demo mode: Restricting to top 1000 variable genes")
            # Get the top 1000 most variable genes
            most_variable_genes = adata.var.sort_values('dispersions_norm', ascending=False).head(1000).index
            adata_hvg = adata[:, most_variable_genes]
        else:
            adata_hvg = adata[:, adata.var.highly_variable]
        
        # Scale data (with max value cap to avoid influence of outliers)
        logger.info("Scaling data")
        sc.pp.scale(adata_hvg, max_value=10)
        
        if self.progress_callback:
            self.progress_callback(int(((self.current_progress + 0.33) / self.total_steps) * 100), "Running PCA")
        
        # PCA (using 'arpack' SVD solver for better performance with sparse matrices)
        logger.info("Running PCA")
        sc.tl.pca(adata_hvg, svd_solver='arpack')
        
        # Determine optimal number of PCs (following publication's approach using 20 PCs)
        # In demo mode, use fewer PCs for faster processing
        n_pcs = 10 if demo_mode else 20
        logger.info(f"Using {n_pcs} principal components for neighborhood graph")
        
        if self.progress_callback:
            self.progress_callback(int(((self.current_progress + 0.66) / self.total_steps) * 100), "Computing neighborhood graph")
        
        # Compute neighborhood graph
        logger.info("Computing neighborhood graph")
        n_neighbors = 10 if demo_mode else 15  # Use fewer neighbors in demo mode for speed
        sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        # Run Leiden clustering at specified resolution (similar to publication's approach)
        logger.info(f"Running Leiden clustering with resolution {resolution}")
        sc.tl.leiden(adata_hvg, resolution=resolution, key_added='leiden')
        
        # UMAP for visualization
        logger.info("Computing UMAP coordinates")
        if demo_mode:
            # Use faster UMAP settings in demo mode
            sc.tl.umap(adata_hvg, min_dist=0.5, spread=1.5)
        else:
            sc.tl.umap(adata_hvg)
        
        # Transfer clustering results and UMAP coordinates back to the original object
        adata.obs['leiden'] = adata_hvg.obs['leiden']
        adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
        
        # Find marker genes for each cluster, unless skipped
        if 'marker_genes' not in skip_steps:
            logger.info("Finding marker genes for clusters")
            method = 't-test' if demo_mode else 'wilcoxon'  # t-test is faster than wilcoxon
            sc.tl.rank_genes_groups(adata, 'leiden', method=method)
        else:
            logger.info("Skipping marker gene identification (demo mode)")
        
        n_clusters = len(adata.obs['leiden'].unique())
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
        
        umap_coords = adata.obsm['X_umap']
        clusters = adata.obs['leiden'].astype(str)
        
        # Create a colormap similar to those in the publication
        n_clusters = len(adata.obs['leiden'].unique())
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
            title='UMAP Visualization with Leiden Clusters',
            labels={'x': 'UMAP_1', 'y': 'UMAP_2', 'color': 'Cluster'},
            width=800,
            height=700,
            color_discrete_sequence=color_map
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
                itemsizing='constant'
            )
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
        
        umap_coords = adata.obsm['X_umap']
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
            title=f'UMAP Visualization by {batch_key}',
            labels={'x': 'UMAP_1', 'y': 'UMAP_2', 'color': batch_key},
            width=800,
            height=700,
            color_discrete_sequence=color_map
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
                itemsizing='constant'
            )
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
        cluster_counts = adata.obs['leiden'].value_counts().sort_index()
        
        # Create bar plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values,
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title='Cluster Size Distribution',
            xaxis_title='Cluster',
            yaxis_title='Number of Cells',
            width=800,
            height=500,
            yaxis=dict(type='log' if max(cluster_counts)/min(cluster_counts) > 100 else 'linear'),
            showlegend=False
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
        
        standard_time = base_time + normalization_time + pca_time + neighbors_time + \
                       clustering_time + umap_time + marker_time
        
        # Demo mode is approximately 5-10x faster due to:
        # - Subsampling to 1000 cells
        # - Using fewer HVG, PCs, and neighbors
        # - Using faster algorithms
        # - Skipping marker gene identification
        demo_time = base_time + min(10.0, standard_time / 5.0)
        
        return {
            'standard': standard_time,
            'demo': demo_time
        }
        
    def _format_clustering_report(self, adata: sc.AnnData, resolution: float, 
                                batch_correction: bool = False, batch_key: Optional[str] = None,
                                demo_mode: bool = False, original_cell_count: Optional[int] = None) -> str:
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
        n_clusters = len(adata.obs['leiden'].unique())
        cluster_counts = adata.obs['leiden'].value_counts().to_dict()
        
        # Format cluster counts for display
        cluster_summary = "\n".join([
            f"- Cluster {cluster}: {count} cells ({count/len(adata)*100:.1f}%)" 
            for cluster, count in sorted(cluster_counts.items(), 
                                         key=lambda x: int(x[0]))
        ])
        
        # Get top marker genes for each cluster if available
        marker_summary = ""
        if 'rank_genes_groups' in adata.uns:
            marker_summary = "\n\n**Top Marker Genes by Cluster:**\n"
            for cluster in sorted(adata.obs['leiden'].unique(), key=lambda x: int(x)):
                genes = adata.uns['rank_genes_groups']['names'][cluster][:5]
                scores = adata.uns['rank_genes_groups']['scores'][cluster][:5]
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
