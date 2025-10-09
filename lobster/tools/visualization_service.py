"""
Visualization service for single-cell RNA-seq data.

This service provides comprehensive visualization methods for single-cell data analysis,
generating interactive and publication-quality plots using Plotly.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scanpy as sc
from plotly.subplots import make_subplots
from scipy import stats
from scipy.sparse import issparse

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class VisualizationError(Exception):
    """Base exception for visualization operations."""

    pass


class SingleCellVisualizationService:
    """
    Professional visualization service for single-cell RNA-seq data.

    This class provides comprehensive visualization methods including
    UMAP, PCA, violin plots, feature plots, dot plots, heatmaps, and QC plots.
    All plots are interactive using Plotly for publication-quality figures.
    """

    def __init__(self):
        """Initialize the visualization service."""
        logger.debug("Initializing SingleCellVisualizationService")

        # Color palettes for consistency
        self.cluster_colors = px.colors.qualitative.Set1
        self.continuous_colors = px.colors.sequential.Viridis
        self.diverging_colors = px.colors.diverging.RdBu_r

        # Default plot settings
        self.default_width = 800
        self.default_height = 600
        self.default_marker_size = 3
        self.default_opacity = 0.8

        # Scientific color scales for gene expression
        self.expression_colorscale = [
            [0, "lightgray"],
            [0.01, "lightblue"],
            [0.1, "blue"],
            [0.5, "red"],
            [0.8, "darkred"],
            [1.0, "black"],
        ]

        logger.debug("SingleCellVisualizationService initialized successfully")

    def create_umap_plot(
        self,
        adata: anndata.AnnData,
        color_by: str = "leiden",
        point_size: Optional[int] = None,
        title: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        show_legend: bool = True,
        alpha: float = 0.8,
    ) -> go.Figure:
        """
        Create an interactive UMAP plot.

        Args:
            adata: AnnData object with UMAP coordinates
            color_by: Column in adata.obs to color by (default: 'leiden')
            point_size: Size of points (default: auto-scaled)
            title: Plot title
            width: Plot width
            height: Plot height
            show_legend: Whether to show legend
            alpha: Point transparency (0-1)

        Returns:
            go.Figure: Interactive UMAP plot

        Raises:
            VisualizationError: If UMAP coordinates not found
        """
        try:
            if "X_umap" not in adata.obsm:
                raise VisualizationError(
                    "UMAP coordinates not found. Run clustering first."
                )

            umap_coords = adata.obsm["X_umap"]

            # Auto-scale point size based on number of cells
            if point_size is None:
                n_cells = adata.n_obs
                if n_cells < 1000:
                    point_size = 8
                elif n_cells < 10000:
                    point_size = 5
                elif n_cells < 50000:
                    point_size = 3
                else:
                    point_size = 2

            # Prepare color data
            if color_by in adata.obs.columns:
                color_data = adata.obs[color_by]
                # Check if categorical or continuous
                is_categorical = (
                    pd.api.types.is_categorical_dtype(color_data)
                    or pd.api.types.is_object_dtype(color_data)
                    or color_data.nunique() < 50
                )
            elif color_by in adata.var_names:
                # Color by gene expression
                gene_idx = adata.var_names.get_loc(color_by)
                if issparse(adata.X):
                    color_data = adata.X[:, gene_idx].toarray().flatten()
                else:
                    color_data = adata.X[:, gene_idx]
                is_categorical = False
                title = title or f"UMAP - {color_by} Expression"
            else:
                raise VisualizationError(
                    f"'{color_by}' not found in obs columns or gene names"
                )

            # Create the plot
            if is_categorical:
                # Categorical coloring
                fig = px.scatter(
                    x=umap_coords[:, 0],
                    y=umap_coords[:, 1],
                    color=color_data.astype(str),
                    title=title or f"UMAP colored by {color_by}",
                    labels={"x": "UMAP 1", "y": "UMAP 2", "color": color_by},
                    width=width or self.default_width,
                    height=height or self.default_height,
                    color_discrete_sequence=self.cluster_colors,
                )
            else:
                # Continuous coloring
                fig = px.scatter(
                    x=umap_coords[:, 0],
                    y=umap_coords[:, 1],
                    color=color_data,
                    title=title or f"UMAP colored by {color_by}",
                    labels={"x": "UMAP 1", "y": "UMAP 2", "color": color_by},
                    width=width or self.default_width,
                    height=height or self.default_height,
                    color_continuous_scale=self.continuous_colors,
                )

            # Update traces
            fig.update_traces(
                marker=dict(size=point_size, opacity=alpha),
                hovertemplate="UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>%{customdata}<extra></extra>",
            )

            # Add cell IDs as hover data
            hover_data = []
            for idx in range(adata.n_obs):
                hover_text = f"Cell: {adata.obs_names[idx]}<br>{color_by}: {color_data.iloc[idx] if hasattr(color_data, 'iloc') else color_data[idx]}"
                hover_data.append(hover_text)

            fig.update_traces(customdata=hover_data)

            # Update layout
            fig.update_layout(
                showlegend=show_legend,
                legend=(
                    dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.01)
                    if show_legend
                    else None
                ),
                hovermode="closest",
                plot_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=True),
                yaxis=dict(showgrid=True, gridcolor="lightgray", zeroline=True),
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating UMAP plot: {e}")
            raise VisualizationError(f"Failed to create UMAP plot: {str(e)}")

    def create_pca_plot(
        self,
        adata: anndata.AnnData,
        color_by: str = "leiden",
        components: Tuple[int, int] = (0, 1),
        point_size: Optional[int] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create an interactive PCA plot.

        Args:
            adata: AnnData object with PCA results
            color_by: Column in adata.obs to color by
            components: Which PC components to plot (0-indexed)
            point_size: Size of points
            title: Plot title

        Returns:
            go.Figure: Interactive PCA plot
        """
        try:
            if "X_pca" not in adata.obsm:
                raise VisualizationError("PCA coordinates not found. Run PCA first.")

            pca_coords = adata.obsm["X_pca"]
            pc1, pc2 = components

            # Get variance explained if available
            var_explained = ""
            if "pca" in adata.uns and "variance_ratio" in adata.uns["pca"]:
                var_ratio = adata.uns["pca"]["variance_ratio"]
                var_explained = (
                    f" ({var_ratio[pc1]*100:.1f}% / {var_ratio[pc2]*100:.1f}% var)"
                )

            # Prepare color data
            if color_by in adata.obs.columns:
                color_data = adata.obs[color_by].astype(str)
            else:
                color_data = pd.Series(["All"] * adata.n_obs, index=adata.obs_names)

            # Create plot
            fig = px.scatter(
                x=pca_coords[:, pc1],
                y=pca_coords[:, pc2],
                color=color_data,
                title=title or f"PCA Plot - PC{pc1+1} vs PC{pc2+1}{var_explained}",
                labels={
                    "x": f"PC{pc1+1}{var_explained.split('/')[0] if var_explained else ''}",
                    "y": f"PC{pc2+1}{var_explained.split('/')[1] if var_explained else ''}",
                    "color": color_by,
                },
                width=self.default_width,
                height=self.default_height,
                color_discrete_sequence=self.cluster_colors,
            )

            # Update marker size
            if point_size is None:
                point_size = 5 if adata.n_obs < 5000 else 3

            fig.update_traces(marker=dict(size=point_size, opacity=0.8))

            return fig

        except Exception as e:
            logger.error(f"Error creating PCA plot: {e}")
            raise VisualizationError(f"Failed to create PCA plot: {str(e)}")

    def create_elbow_plot(
        self, adata: anndata.AnnData, n_pcs: int = 50, title: Optional[str] = None
    ) -> go.Figure:
        """
        Create an elbow plot for PCA variance explained.

        Args:
            adata: AnnData object with PCA results
            n_pcs: Number of PCs to show
            title: Plot title

        Returns:
            go.Figure: Elbow plot
        """
        try:
            if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
                raise VisualizationError("PCA variance information not found")

            var_ratio = adata.uns["pca"]["variance_ratio"][:n_pcs]
            cumsum_var = np.cumsum(var_ratio)

            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add variance explained per PC
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(var_ratio) + 1)),
                    y=var_ratio * 100,
                    mode="lines+markers",
                    name="Individual variance",
                    marker=dict(size=6),
                    line=dict(width=2),
                ),
                secondary_y=False,
            )

            # Add cumulative variance
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cumsum_var) + 1)),
                    y=cumsum_var * 100,
                    mode="lines+markers",
                    name="Cumulative variance",
                    marker=dict(size=6),
                    line=dict(width=2, dash="dash"),
                ),
                secondary_y=True,
            )

            # Update layout
            fig.update_xaxes(title_text="Principal Component")
            fig.update_yaxes(title_text="Variance Explained (%)", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative Variance (%)", secondary_y=True)

            fig.update_layout(
                title=title or "PCA Elbow Plot - Variance Explained",
                width=self.default_width,
                height=self.default_height,
                hovermode="x unified",
                plot_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="lightgray"),
                yaxis=dict(showgrid=True, gridcolor="lightgray"),
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating elbow plot: {e}")
            raise VisualizationError(f"Failed to create elbow plot: {str(e)}")

    def create_violin_plot(
        self,
        adata: anndata.AnnData,
        genes: Union[str, List[str]],
        groupby: str = "leiden",
        use_raw: bool = True,
        log_scale: bool = False,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create violin plots for gene expression across groups.

        Args:
            adata: AnnData object
            genes: Gene or list of genes to plot
            groupby: Column in adata.obs to group by
            use_raw: Whether to use raw data
            log_scale: Whether to use log scale
            title: Plot title

        Returns:
            go.Figure: Violin plot
        """
        try:
            # Ensure genes is a list
            if isinstance(genes, str):
                genes = [genes]

            # Validate genes exist
            data_source = adata.raw if use_raw and adata.raw else adata
            missing_genes = [g for g in genes if g not in data_source.var_names]
            if missing_genes:
                raise VisualizationError(f"Genes not found: {missing_genes}")

            # Validate groupby
            if groupby not in adata.obs.columns:
                raise VisualizationError(f"'{groupby}' not found in obs columns")

            # Create subplots for multiple genes
            n_genes = len(genes)
            fig = make_subplots(
                rows=1, cols=n_genes, subplot_titles=genes, horizontal_spacing=0.1
            )

            # Get groups
            groups = adata.obs[groupby].astype(str).unique()
            groups = sorted(groups, key=lambda x: int(x) if x.isdigit() else x)

            # Process each gene
            for gene_idx, gene in enumerate(genes):
                # Extract expression data
                gene_loc = data_source.var_names.get_loc(gene)
                if issparse(data_source.X):
                    expr_data = data_source.X[:, gene_loc].toarray().flatten()
                else:
                    expr_data = data_source.X[:, gene_loc]

                # Create violin for each group
                for group in groups:
                    mask = adata.obs[groupby].astype(str) == group
                    group_expr = expr_data[mask]

                    # Apply log transformation if requested
                    if log_scale:
                        group_expr = np.log1p(group_expr)

                    fig.add_trace(
                        go.Violin(
                            y=group_expr,
                            name=f"{groupby} {group}",
                            x=[group] * len(group_expr),
                            box_visible=True,
                            meanline_visible=True,
                            showlegend=(gene_idx == 0),
                            scalemode="width",
                            width=0.7,
                        ),
                        row=1,
                        col=gene_idx + 1,
                    )

            # Update layout
            y_title = "Expression (log)" if log_scale else "Expression"
            fig.update_yaxes(title_text=y_title, row=1, col=1)
            fig.update_xaxes(title_text=groupby)

            fig.update_layout(
                title=title or f"Violin Plot - {', '.join(genes)}",
                width=max(800, 400 * n_genes),
                height=600,
                violinmode="group",
                plot_bgcolor="white",
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating violin plot: {e}")
            raise VisualizationError(f"Failed to create violin plot: {str(e)}")

    def create_feature_plot(
        self,
        adata: anndata.AnnData,
        genes: Union[str, List[str]],
        use_raw: bool = True,
        ncols: int = 2,
        point_size: Optional[int] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> go.Figure:
        """
        Create feature plots showing gene expression on UMAP.

        Args:
            adata: AnnData object with UMAP coordinates
            genes: Gene or list of genes to plot
            use_raw: Whether to use raw data
            ncols: Number of columns in subplot grid
            point_size: Size of points
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale

        Returns:
            go.Figure: Feature plot
        """
        try:
            if "X_umap" not in adata.obsm:
                raise VisualizationError("UMAP coordinates not found")

            # Ensure genes is a list
            if isinstance(genes, str):
                genes = [genes]

            # Validate genes
            data_source = adata.raw if use_raw and adata.raw else adata
            missing_genes = [g for g in genes if g not in data_source.var_names]
            if missing_genes:
                raise VisualizationError(f"Genes not found: {missing_genes}")

            # Calculate subplot layout
            n_genes = len(genes)
            nrows = (n_genes + ncols - 1) // ncols

            # Create subplots
            fig = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=genes,
                horizontal_spacing=0.08,
                vertical_spacing=0.12,
            )

            # Auto-scale point size
            if point_size is None:
                point_size = 3 if adata.n_obs < 10000 else 2

            umap_coords = adata.obsm["X_umap"]

            # Process each gene
            for idx, gene in enumerate(genes):
                row = idx // ncols + 1
                col = idx % ncols + 1

                # Extract expression data
                gene_loc = data_source.var_names.get_loc(gene)
                if issparse(data_source.X):
                    expr_data = data_source.X[:, gene_loc].toarray().flatten()
                else:
                    expr_data = data_source.X[:, gene_loc]

                # Apply vmin/vmax if specified
                if vmin is not None:
                    expr_data = np.maximum(expr_data, vmin)
                if vmax is not None:
                    expr_data = np.minimum(expr_data, vmax)

                # Create scatter plot
                scatter = go.Scatter(
                    x=umap_coords[:, 0],
                    y=umap_coords[:, 1],
                    mode="markers",
                    marker=dict(
                        size=point_size,
                        color=expr_data,
                        colorscale=self.expression_colorscale,
                        showscale=(idx == 0),  # Show colorbar only for first plot
                        colorbar=dict(
                            title="Expression", x=1.15 if ncols == 1 else 1.05
                        ),
                        cmin=vmin if vmin is not None else expr_data.min(),
                        cmax=vmax if vmax is not None else expr_data.max(),
                    ),
                    text=[
                        f"Cell: {cell}<br>Expression: {expr:.2f}"
                        for cell, expr in zip(adata.obs_names, expr_data)
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )

                fig.add_trace(scatter, row=row, col=col)

            # Update axes
            for i in range(1, n_genes + 1):
                row = (i - 1) // ncols + 1
                col = (i - 1) % ncols + 1
                fig.update_xaxes(title_text="UMAP 1", row=row, col=col)
                fig.update_yaxes(title_text="UMAP 2", row=row, col=col)

            # Update layout
            fig.update_layout(
                title=f"Feature Plot - Gene Expression",
                width=400 * min(ncols, n_genes),
                height=400 * nrows,
                plot_bgcolor="white",
                showlegend=False,
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating feature plot: {e}")
            raise VisualizationError(f"Failed to create feature plot: {str(e)}")

    def create_dot_plot(
        self,
        adata: anndata.AnnData,
        genes: List[str],
        groupby: str = "leiden",
        use_raw: bool = True,
        standard_scale: str = "var",
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create a dot plot for marker gene expression.

        Args:
            adata: AnnData object
            genes: List of genes to plot
            groupby: Column in adata.obs to group by
            use_raw: Whether to use raw data
            standard_scale: How to scale ('var', 'group', or None)
            title: Plot title

        Returns:
            go.Figure: Dot plot
        """
        try:
            # Validate inputs
            data_source = adata.raw if use_raw and adata.raw else adata
            missing_genes = [g for g in genes if g not in data_source.var_names]
            if missing_genes:
                raise VisualizationError(f"Genes not found: {missing_genes}")

            if groupby not in adata.obs.columns:
                raise VisualizationError(f"'{groupby}' not found in obs columns")

            # Get groups
            groups = adata.obs[groupby].astype(str).unique()
            groups = sorted(groups, key=lambda x: int(x) if x.isdigit() else x)

            # Calculate statistics for each gene-group combination
            mean_expr = []
            pct_expr = []

            for gene in genes:
                gene_means = []
                gene_pcts = []

                # Get gene expression
                gene_loc = data_source.var_names.get_loc(gene)
                if issparse(data_source.X):
                    expr_data = data_source.X[:, gene_loc].toarray().flatten()
                else:
                    expr_data = data_source.X[:, gene_loc]

                for group in groups:
                    mask = adata.obs[groupby].astype(str) == group
                    group_expr = expr_data[mask]

                    # Calculate mean expression
                    gene_means.append(np.mean(group_expr))

                    # Calculate percentage of cells expressing
                    gene_pcts.append(np.sum(group_expr > 0) / len(group_expr) * 100)

                mean_expr.append(gene_means)
                pct_expr.append(gene_pcts)

            mean_expr = np.array(mean_expr)
            pct_expr = np.array(pct_expr)

            # Standard scaling if requested
            if standard_scale == "var":
                # Scale across all groups for each gene
                for i in range(len(genes)):
                    mean_expr[i] = (mean_expr[i] - mean_expr[i].mean()) / (
                        mean_expr[i].std() + 1e-8
                    )
            elif standard_scale == "group":
                # Scale across all genes for each group
                for j in range(len(groups)):
                    mean_expr[:, j] = (mean_expr[:, j] - mean_expr[:, j].mean()) / (
                        mean_expr[:, j].std() + 1e-8
                    )

            # Create dot plot
            fig = go.Figure()

            # Add dots for each gene-group combination
            for i, gene in enumerate(genes):
                for j, group in enumerate(groups):
                    # Size based on percentage expressing
                    size = pct_expr[i, j] / 100 * 30  # Scale to max size of 30

                    # Color based on mean expression
                    color = mean_expr[i, j]

                    fig.add_trace(
                        go.Scatter(
                            x=[j],
                            y=[i],
                            mode="markers",
                            marker=dict(
                                size=size,
                                color=color,
                                colorscale=self.expression_colorscale,
                                showscale=(i == 0 and j == 0),  # Show colorbar once
                                colorbar=dict(title="Mean<br>Expression", x=1.15),
                                line=dict(width=0.5, color="black"),
                            ),
                            text=f"Gene: {gene}<br>Group: {group}<br>"
                            f"Mean Expr: {mean_expr[i, j]:.2f}<br>"
                            f"% Expressing: {pct_expr[i, j]:.1f}%",
                            hovertemplate="%{text}<extra></extra>",
                            showlegend=False,
                        )
                    )

            # Add size legend
            for size_pct in [25, 50, 75, 100]:
                fig.add_trace(
                    go.Scatter(
                        x=[len(groups) + 0.5],
                        y=[len(genes) - 4 + size_pct / 25],
                        mode="markers",
                        marker=dict(
                            size=size_pct / 100 * 30,
                            color="gray",
                            line=dict(width=0.5, color="black"),
                        ),
                        text=f"{size_pct}%",
                        showlegend=False,
                    )
                )

            # Update layout
            fig.update_layout(
                title=title or f"Dot Plot - {groupby}",
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(groups))),
                    ticktext=groups,
                    title=groupby,
                ),
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(genes))),
                    ticktext=genes,
                    title="Genes",
                ),
                width=max(600, 50 * len(groups)),
                height=max(400, 30 * len(genes)),
                plot_bgcolor="white",
                hovermode="closest",
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating dot plot: {e}")
            raise VisualizationError(f"Failed to create dot plot: {str(e)}")

    def create_heatmap(
        self,
        adata: anndata.AnnData,
        genes: Optional[List[str]] = None,
        groupby: str = "leiden",
        use_raw: bool = True,
        n_top_genes: int = 5,
        standard_scale: bool = True,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create a heatmap of gene expression.

        Args:
            adata: AnnData object
            genes: List of genes (if None, use top marker genes)
            groupby: Column to group by
            use_raw: Whether to use raw data
            n_top_genes: Number of top genes per group if genes not specified
            standard_scale: Whether to z-score normalize
            title: Plot title

        Returns:
            go.Figure: Heatmap
        """
        try:
            # Get genes if not specified
            if genes is None:
                if "rank_genes_groups" not in adata.uns:
                    raise VisualizationError(
                        "No marker genes found. Run rank_genes_groups first or specify genes."
                    )

                # Get top genes from each group
                genes = []
                for group in adata.obs[groupby].unique():
                    group_genes = adata.uns["rank_genes_groups"]["names"][str(group)][
                        :n_top_genes
                    ]
                    genes.extend(group_genes)
                genes = list(
                    dict.fromkeys(genes)
                )  # Remove duplicates while preserving order

            # Validate genes
            data_source = adata.raw if use_raw and adata.raw else adata
            missing_genes = [g for g in genes if g not in data_source.var_names]
            if missing_genes:
                logger.warning(f"Genes not found: {missing_genes}")
                genes = [g for g in genes if g in data_source.var_names]

            if not genes:
                raise VisualizationError("No valid genes to plot")

            # Extract expression matrix
            gene_indices = [data_source.var_names.get_loc(g) for g in genes]
            if issparse(data_source.X):
                expr_matrix = data_source.X[:, gene_indices].toarray()
            else:
                expr_matrix = data_source.X[:, gene_indices]

            # Group by clusters
            groups = adata.obs[groupby].astype(str).unique()
            groups = sorted(groups, key=lambda x: int(x) if x.isdigit() else x)

            # Calculate mean expression per group
            group_means = []
            for group in groups:
                mask = adata.obs[groupby].astype(str) == group
                group_expr = expr_matrix[mask, :].mean(axis=0)
                group_means.append(group_expr)

            group_means = np.array(group_means).T  # Genes x Groups

            # Standard scale if requested
            if standard_scale:
                # Z-score normalize each gene across groups
                for i in range(group_means.shape[0]):
                    mean_val = group_means[i].mean()
                    std_val = group_means[i].std()
                    if std_val > 0:
                        group_means[i] = (group_means[i] - mean_val) / std_val

            # Create heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=group_means,
                    x=groups,
                    y=genes,
                    colorscale=(
                        self.diverging_colors
                        if standard_scale
                        else self.continuous_colors
                    ),
                    colorbar=dict(
                        title=(
                            "Expression<br>(z-score)"
                            if standard_scale
                            else "Expression"
                        )
                    ),
                    hovertemplate="Gene: %{y}<br>Group: %{x}<br>Expression: %{z:.2f}<extra></extra>",
                )
            )

            # Update layout
            fig.update_layout(
                title=title or f"Gene Expression Heatmap - {groupby}",
                xaxis=dict(title=groupby, tickmode="linear"),
                yaxis=dict(title="Genes", tickmode="linear"),
                width=max(600, 40 * len(groups)),
                height=max(400, 20 * len(genes)),
                plot_bgcolor="white",
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            raise VisualizationError(f"Failed to create heatmap: {str(e)}")

    def create_qc_plots(
        self, adata: anndata.AnnData, title: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive, publication-quality QC plots for single-cell transcriptomics data.

        This method generates a sophisticated multi-panel figure with all essential QC metrics
        for single-cell RNA-seq analysis, including advanced visualizations and statistical summaries.

        Args:
            adata: AnnData object with single-cell data
            title: Optional overall title for the QC plots

        Returns:
            go.Figure: Professional multi-panel QC figure
        """
        try:
            # Calculate all QC metrics upfront
            logger.info("Computing comprehensive QC metrics for single-cell data")

            # Basic metrics
            if "n_genes" not in adata.obs.columns:
                adata.obs["n_genes"] = (adata.X > 0).sum(axis=1)
            if "n_counts" not in adata.obs.columns:
                if issparse(adata.X):
                    adata.obs["n_counts"] = adata.X.sum(axis=1).A1
                else:
                    adata.obs["n_counts"] = adata.X.sum(axis=1)

            # Calculate gene detection rate (complexity)
            adata.obs["gene_detection_rate"] = adata.obs["n_genes"] / adata.n_vars * 100

            # Calculate log10 counts for better visualization
            adata.obs["log10_counts"] = np.log10(adata.obs["n_counts"] + 1)
            adata.obs["log10_genes"] = np.log10(adata.obs["n_genes"] + 1)

            # Mitochondrial percentage
            if "percent_mito" not in adata.obs.columns:
                mito_genes = adata.var_names.str.startswith(
                    "MT-"
                ) | adata.var_names.str.startswith("mt-")
                if mito_genes.sum() > 0:
                    if issparse(adata.X):
                        adata.obs["percent_mito"] = (
                            np.sum(adata[:, mito_genes].X, axis=1).A1
                            / adata.obs["n_counts"]
                            * 100
                        )
                    else:
                        adata.obs["percent_mito"] = (
                            np.sum(adata[:, mito_genes].X, axis=1)
                            / adata.obs["n_counts"]
                            * 100
                        )
                else:
                    adata.obs["percent_mito"] = 0

            # Ribosomal percentage
            if "percent_ribo" not in adata.obs.columns:
                ribo_genes = adata.var_names.str.startswith(
                    ("RPS", "RPL", "rps", "rpl")
                )
                if ribo_genes.sum() > 0:
                    if issparse(adata.X):
                        adata.obs["percent_ribo"] = (
                            np.sum(adata[:, ribo_genes].X, axis=1).A1
                            / adata.obs["n_counts"]
                            * 100
                        )
                    else:
                        adata.obs["percent_ribo"] = (
                            np.sum(adata[:, ribo_genes].X, axis=1)
                            / adata.obs["n_counts"]
                            * 100
                        )
                else:
                    adata.obs["percent_ribo"] = 0

            # Hemoglobin percentage (important for blood samples)
            hb_genes = adata.var_names.str.startswith(("HBA", "HBB", "hba", "hbb"))
            if hb_genes.sum() > 0:
                if issparse(adata.X):
                    adata.obs["percent_hb"] = (
                        np.sum(adata[:, hb_genes].X, axis=1).A1
                        / adata.obs["n_counts"]
                        * 100
                    )
                else:
                    adata.obs["percent_hb"] = (
                        np.sum(adata[:, hb_genes].X, axis=1)
                        / adata.obs["n_counts"]
                        * 100
                    )
            else:
                adata.obs["percent_hb"] = 0

            # Calculate doublet scores if available
            has_doublet = "doublet_score" in adata.obs.columns

            # Detect batch column
            batch_cols = [
                "batch",
                "sample",
                "patient",
                "Patient_ID",
                "donor",
                "replicate",
            ]
            batch_col = None
            for col in batch_cols:
                if col in adata.obs.columns:
                    batch_col = col
                    break

            # Create sophisticated subplot layout
            # Using a 4x4 grid for comprehensive visualization
            fig = make_subplots(
                rows=4,
                cols=4,
                subplot_titles=[
                    "A. Transcriptional Complexity",  # Main scatter
                    "B. Mitochondrial QC",  # Mito vs counts
                    "C. Ribosomal Content",  # Ribo vs counts
                    "D. nUMI Distribution",  # Violin + box
                    "E. nGene Distribution",  # Violin + box
                    "F. MT% Distribution",  # Violin + box
                    "G. Library Saturation",  # Complexity curve
                    "H. Cell Quality Categories",  # Pie chart
                    "I. Gene Detection Rate",  # Histogram
                    "J. Sample Composition" if batch_col else "J. Top Genes",
                    "K. Correlation Matrix",  # QC metrics correlation
                    "L. Statistical Summary",  # Table
                    "M. Doublet Detection" if has_doublet else "M. Ribo% Distribution",
                    "N. QC Thresholds",  # Threshold visualization
                    "O. Cell Filtering Impact",  # Before/after
                    "P. Batch Effects" if batch_col else "P. Overall Quality",
                ],
                specs=[
                    [
                        {"type": "scatter"},
                        {"type": "scatter"},
                        {"type": "scatter"},
                        {"type": "violin"},
                    ],
                    [
                        {"type": "violin"},
                        {"type": "violin"},
                        {"type": "scatter"},
                        {"type": "pie"},
                    ],
                    [
                        {"type": "histogram"},
                        {"type": "bar"},
                        {"type": "heatmap"},
                        {"type": "table"},
                    ],
                    [
                        {"type": "scatter" if has_doublet else "histogram"},
                        {"type": "scatter"},
                        {"type": "bar"},
                        {"type": "scatter" if batch_col else "indicator"},
                    ],
                ],
                horizontal_spacing=0.12,
                vertical_spacing=0.12,
                column_widths=[0.25, 0.25, 0.25, 0.25],
                row_heights=[0.25, 0.25, 0.25, 0.25],
            )

            # Define professional color scheme
            color_quality = px.colors.sequential.Viridis
            color_mito = px.colors.sequential.Reds
            color_ribo = px.colors.sequential.Blues
            color_discrete = px.colors.qualitative.Set2

            # Calculate QC thresholds using MAD-based approach
            def calculate_outlier_thresholds(values, nmads=3):
                median = np.median(values)
                mad = np.median(np.abs(values - median))
                lower = median - nmads * mad
                upper = median + nmads * mad
                return lower, upper

            genes_lower, genes_upper = calculate_outlier_thresholds(
                adata.obs["n_genes"]
            )
            mito_lower, mito_upper = calculate_outlier_thresholds(
                adata.obs["percent_mito"]
            )

            # A. Main complexity scatter (row 1, col 1)
            fig.add_trace(
                go.Scattergl(
                    x=adata.obs["n_counts"],
                    y=adata.obs["n_genes"],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=adata.obs["percent_mito"],
                        colorscale=color_mito,
                        showscale=False,
                        cmin=0,
                        cmax=np.percentile(adata.obs["percent_mito"], 95),
                        opacity=0.6,
                    ),
                    text=[
                        f"Cell: {cell}<br>UMIs: {umi:,}<br>Genes: {gene:,}<br>MT%: {mito:.1f}%"
                        for cell, umi, gene, mito in zip(
                            (
                                adata.obs_names[:1000]
                                if len(adata.obs_names) > 1000
                                else adata.obs_names
                            ),
                            (
                                adata.obs["n_counts"][:1000]
                                if len(adata.obs["n_counts"]) > 1000
                                else adata.obs["n_counts"]
                            ),
                            (
                                adata.obs["n_genes"][:1000]
                                if len(adata.obs["n_genes"]) > 1000
                                else adata.obs["n_genes"]
                            ),
                            (
                                adata.obs["percent_mito"][:1000]
                                if len(adata.obs["percent_mito"]) > 1000
                                else adata.obs["percent_mito"]
                            ),
                        )
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Add threshold lines
            fig.add_hline(
                y=genes_lower,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=1,
                col=1,
            )
            fig.add_hline(
                y=genes_upper,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=1,
                col=1,
            )

            # B. Mitochondrial QC (row 1, col 2)
            fig.add_trace(
                go.Scattergl(
                    x=adata.obs["n_counts"],
                    y=adata.obs["percent_mito"],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=adata.obs["n_genes"],
                        colorscale=color_quality,
                        showscale=False,
                        opacity=0.6,
                    ),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            # Add threshold line
            fig.add_hline(
                y=mito_upper,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=1,
                col=2,
            )

            # C. Ribosomal content (row 1, col 3)
            fig.add_trace(
                go.Scattergl(
                    x=adata.obs["n_counts"],
                    y=adata.obs["percent_ribo"],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=adata.obs["percent_mito"],
                        colorscale=color_mito,
                        showscale=False,
                        opacity=0.6,
                    ),
                    showlegend=False,
                ),
                row=1,
                col=3,
            )

            # D-F. Distribution plots with violin + box (rows 1-2, col 4 and row 2, cols 1-2)
            for idx, (metric, metric_name, row_pos, col_pos, color) in enumerate(
                [
                    ("n_counts", "nUMIs", 1, 4, color_discrete[0]),
                    ("n_genes", "nGenes", 2, 1, color_discrete[1]),
                    ("percent_mito", "MT%", 2, 2, color_discrete[2]),
                ]
            ):
                # Add violin plot
                fig.add_trace(
                    go.Violin(
                        y=adata.obs[metric],
                        name=metric_name,
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=color,
                        opacity=0.6,
                        showlegend=False,
                        side="positive",
                        points="outliers",
                    ),
                    row=row_pos,
                    col=col_pos,
                )

            # G. Library saturation curve (row 2, col 3)
            # Sample cells for performance
            n_sample = min(1000, adata.n_obs)
            sample_idx = np.random.choice(adata.n_obs, n_sample, replace=False)

            saturation_x = adata.obs["n_counts"].iloc[sample_idx]
            saturation_y = adata.obs["gene_detection_rate"].iloc[sample_idx]

            fig.add_trace(
                go.Scattergl(
                    x=saturation_x,
                    y=saturation_y,
                    mode="markers",
                    marker=dict(size=2, color="steelblue", opacity=0.5),
                    showlegend=False,
                ),
                row=2,
                col=3,
            )

            # H. Cell quality categories pie chart (row 2, col 4)
            n_high_quality = np.sum(
                (adata.obs["n_genes"] >= genes_lower)
                & (adata.obs["n_genes"] <= genes_upper)
                & (adata.obs["percent_mito"] <= mito_upper)
            )
            n_low_genes = np.sum(adata.obs["n_genes"] < genes_lower)
            n_high_mito = np.sum(adata.obs["percent_mito"] > mito_upper)
            n_other = adata.n_obs - n_high_quality - n_low_genes - n_high_mito

            fig.add_trace(
                go.Pie(
                    labels=["High Quality", "Low Genes", "High MT%", "Other"],
                    values=[n_high_quality, n_low_genes, n_high_mito, n_other],
                    hole=0.3,
                    marker=dict(colors=color_discrete[:4]),
                    textinfo="label+percent",
                    showlegend=False,
                ),
                row=2,
                col=4,
            )

            # I. Gene detection rate histogram (row 3, col 1)
            fig.add_trace(
                go.Histogram(
                    x=adata.obs["gene_detection_rate"],
                    nbinsx=50,
                    marker_color="teal",
                    opacity=0.7,
                    showlegend=False,
                ),
                row=3,
                col=1,
            )

            # J. Sample composition or top genes (row 3, col 2)
            if batch_col:
                sample_counts = adata.obs[batch_col].value_counts().head(10)
                fig.add_trace(
                    go.Bar(
                        x=sample_counts.index.astype(str),
                        y=sample_counts.values,
                        marker_color=color_discrete[4],
                        showlegend=False,
                        text=sample_counts.values,
                        textposition="outside",
                    ),
                    row=3,
                    col=2,
                )
            else:
                # Show top expressed genes
                if issparse(adata.X):
                    gene_counts = np.array(adata.X.sum(axis=0)).flatten()
                else:
                    gene_counts = adata.X.sum(axis=0)
                top_genes_idx = np.argsort(gene_counts)[-10:][::-1]
                top_genes = adata.var_names[top_genes_idx]
                top_counts = gene_counts[top_genes_idx]

                fig.add_trace(
                    go.Bar(
                        x=top_genes[:10],
                        y=top_counts[:10],
                        marker_color=color_discrete[5],
                        showlegend=False,
                    ),
                    row=3,
                    col=2,
                )

            # K. Correlation heatmap of QC metrics (row 3, col 3)
            qc_metrics = [
                "n_counts",
                "n_genes",
                "percent_mito",
                "percent_ribo",
                "gene_detection_rate",
            ]
            corr_data = adata.obs[qc_metrics].corr()

            fig.add_trace(
                go.Heatmap(
                    z=corr_data.values,
                    x=[m.replace("_", " ").replace("percent", "%") for m in qc_metrics],
                    y=[m.replace("_", " ").replace("percent", "%") for m in qc_metrics],
                    colorscale="RdBu_r",
                    zmid=0,
                    showscale=False,
                    text=np.round(corr_data.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    showlegend=False,
                ),
                row=3,
                col=3,
            )

            # L. Statistical summary table (row 3, col 4)
            summary_stats = pd.DataFrame(
                {
                    "Metric": [
                        "Cells",
                        "Genes",
                        "Mean UMIs",
                        "Mean Genes",
                        "Mean MT%",
                        "Mean Ribo%",
                    ],
                    "Value": [
                        f"{adata.n_obs:,}",
                        f"{adata.n_vars:,}",
                        f'{adata.obs["n_counts"].mean():.0f}',
                        f'{adata.obs["n_genes"].mean():.0f}',
                        f'{adata.obs["percent_mito"].mean():.1f}%',
                        f'{adata.obs["percent_ribo"].mean():.1f}%',
                    ],
                }
            )

            fig.add_trace(
                go.Table(
                    cells=dict(
                        values=[summary_stats["Metric"], summary_stats["Value"]],
                        align="left",
                        font=dict(size=11),
                        height=25,
                        fill_color=["lightgray", "white"],
                    )
                ),
                row=3,
                col=4,
            )

            # M. Doublet detection or Ribo distribution (row 4, col 1)
            if has_doublet:
                fig.add_trace(
                    go.Scattergl(
                        x=adata.obs["n_counts"],
                        y=adata.obs["doublet_score"],
                        mode="markers",
                        marker=dict(
                            size=3,
                            color=adata.obs["doublet_score"],
                            colorscale="Hot",
                            showscale=False,
                            opacity=0.6,
                        ),
                        showlegend=False,
                    ),
                    row=4,
                    col=1,
                )
            else:
                fig.add_trace(
                    go.Histogram(
                        x=adata.obs["percent_ribo"],
                        nbinsx=50,
                        marker_color="royalblue",
                        opacity=0.7,
                        showlegend=False,
                    ),
                    row=4,
                    col=1,
                )

            # N. QC thresholds visualization (row 4, col 2)
            # Create a 2D density plot showing QC thresholds
            fig.add_trace(
                go.Histogram2d(
                    x=adata.obs["n_genes"],
                    y=adata.obs["percent_mito"],
                    colorscale="Blues",
                    showscale=False,
                    opacity=0.8,
                ),
                row=4,
                col=2,
            )

            # Add threshold rectangles
            fig.add_shape(
                type="rect",
                x0=genes_lower,
                x1=genes_upper,
                y0=0,
                y1=mito_upper,
                line=dict(color="green", width=2),
                fillcolor="green",
                opacity=0.1,
                row=4,
                col=2,
            )

            # O. Cell filtering impact (row 4, col 3)
            filtering_data = pd.DataFrame(
                {"Stage": ["Raw", "After QC"], "Cells": [adata.n_obs, n_high_quality]}
            )

            fig.add_trace(
                go.Bar(
                    x=filtering_data["Stage"],
                    y=filtering_data["Cells"],
                    marker_color=["gray", "green"],
                    text=filtering_data["Cells"],
                    textposition="outside",
                    showlegend=False,
                ),
                row=4,
                col=3,
            )

            # P. Batch effects or overall quality indicator (row 4, col 4)
            if batch_col:
                # Show batch variation in key metrics
                batch_data = (
                    adata.obs.groupby(batch_col)
                    .agg(
                        {"n_counts": "mean", "n_genes": "mean", "percent_mito": "mean"}
                    )
                    .reset_index()
                )

                # Normalize for visualization
                for col in ["n_counts", "n_genes", "percent_mito"]:
                    batch_data[col] = (
                        batch_data[col] - batch_data[col].mean()
                    ) / batch_data[col].std()

                fig.add_trace(
                    go.Scattergl(
                        x=batch_data[batch_col].astype(str),
                        y=batch_data["n_counts"],
                        mode="lines+markers",
                        name="nUMIs",
                        marker=dict(size=8),
                        showlegend=False,
                    ),
                    row=4,
                    col=4,
                )
            else:
                # Overall quality indicator
                quality_score = (n_high_quality / adata.n_obs) * 100
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=quality_score,
                        title={"text": "Overall Quality"},
                        delta={"reference": 80},
                        gauge={
                            "axis": {"range": [None, 100]},
                            "bar": {
                                "color": "darkgreen" if quality_score > 70 else "orange"
                            },
                            "steps": [
                                {"range": [0, 50], "color": "lightgray"},
                                {"range": [50, 80], "color": "gray"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 90,
                            },
                        },
                    ),
                    row=4,
                    col=4,
                )

            # Calculate data-driven axis ranges
            counts_min, counts_max = (
                adata.obs["n_counts"].min(),
                adata.obs["n_counts"].max(),
            )
            genes_min, genes_max = (
                adata.obs["n_genes"].min(),
                adata.obs["n_genes"].max(),
            )
            mito_min, mito_max = (
                adata.obs["percent_mito"].min(),
                adata.obs["percent_mito"].max(),
            )
            ribo_min, ribo_max = (
                adata.obs["percent_ribo"].min(),
                adata.obs["percent_ribo"].max(),
            )

            # Add small padding to ranges
            counts_padding = (counts_max - counts_min) * 0.05
            genes_padding = (genes_max - genes_min) * 0.05
            mito_padding = max((mito_max - mito_min) * 0.05, 1)
            ribo_padding = max((ribo_max - ribo_min) * 0.05, 1)

            # Update all axes labels with data-driven ranges
            # Row 1
            fig.update_xaxes(
                title_text="nUMIs",
                row=1,
                col=1,
                type="log",
                range=[
                    np.log10(max(1, counts_min - counts_padding)),
                    np.log10(counts_max + counts_padding),
                ],
            )
            fig.update_yaxes(
                title_text="nGenes",
                row=1,
                col=1,
                type="log",
                range=[
                    np.log10(max(1, genes_min - genes_padding)),
                    np.log10(genes_max + genes_padding),
                ],
            )
            fig.update_xaxes(
                title_text="nUMIs",
                row=1,
                col=2,
                type="log",
                range=[
                    np.log10(max(1, counts_min - counts_padding)),
                    np.log10(counts_max + counts_padding),
                ],
            )
            fig.update_yaxes(
                title_text="MT%",
                row=1,
                col=2,
                range=[max(0, mito_min - mito_padding), mito_max + mito_padding],
            )
            fig.update_xaxes(
                title_text="nUMIs",
                row=1,
                col=3,
                type="log",
                range=[
                    np.log10(max(1, counts_min - counts_padding)),
                    np.log10(counts_max + counts_padding),
                ],
            )
            fig.update_yaxes(
                title_text="Ribo%",
                row=1,
                col=3,
                range=[max(0, ribo_min - ribo_padding), ribo_max + ribo_padding],
            )
            fig.update_yaxes(
                title_text="nUMIs",
                row=1,
                col=4,
                range=[counts_min - counts_padding, counts_max + counts_padding],
            )

            # Row 2
            fig.update_yaxes(
                title_text="nGenes",
                row=2,
                col=1,
                range=[genes_min - genes_padding, genes_max + genes_padding],
            )
            fig.update_yaxes(
                title_text="MT%",
                row=2,
                col=2,
                range=[max(0, mito_min - mito_padding), mito_max + mito_padding],
            )
            fig.update_xaxes(
                title_text="nUMIs",
                row=2,
                col=3,
                type="log",
                range=[
                    np.log10(max(1, counts_min - counts_padding)),
                    np.log10(counts_max + counts_padding),
                ],
            )
            fig.update_yaxes(
                title_text="Gene Detection %", row=2, col=3, range=[0, 100]
            )

            # Row 3
            fig.update_xaxes(title_text="Gene Detection %", row=3, col=1)
            fig.update_yaxes(title_text="Frequency", row=3, col=1)
            if batch_col:
                fig.update_xaxes(title_text=batch_col, row=3, col=2, tickangle=45)
                fig.update_yaxes(title_text="Cell Count", row=3, col=2)
            else:
                fig.update_xaxes(title_text="Gene", row=3, col=2, tickangle=45)
                fig.update_yaxes(title_text="Total UMIs", row=3, col=2, type="log")

            # Row 4
            if has_doublet:
                doublet_min, doublet_max = (
                    adata.obs["doublet_score"].min(),
                    adata.obs["doublet_score"].max(),
                )
                doublet_padding = (doublet_max - doublet_min) * 0.05
                fig.update_xaxes(
                    title_text="nUMIs",
                    row=4,
                    col=1,
                    type="log",
                    range=[
                        np.log10(max(1, counts_min - counts_padding)),
                        np.log10(counts_max + counts_padding),
                    ],
                )
                fig.update_yaxes(
                    title_text="Doublet Score",
                    row=4,
                    col=1,
                    range=[
                        max(0, doublet_min - doublet_padding),
                        doublet_max + doublet_padding,
                    ],
                )
            else:
                fig.update_xaxes(
                    title_text="Ribo%",
                    row=4,
                    col=1,
                    range=[max(0, ribo_min - ribo_padding), ribo_max + ribo_padding],
                )
                fig.update_yaxes(title_text="Frequency", row=4, col=1)
            fig.update_xaxes(
                title_text="nGenes",
                row=4,
                col=2,
                range=[genes_min - genes_padding, genes_max + genes_padding],
            )
            fig.update_yaxes(
                title_text="MT%",
                row=4,
                col=2,
                range=[max(0, mito_min - mito_padding), mito_max + mito_padding],
            )
            fig.update_xaxes(title_text="Stage", row=4, col=3)
            fig.update_yaxes(
                title_text="Cell Count",
                row=4,
                col=3,
                range=[0, max(adata.n_obs, n_high_quality) * 1.1],
            )
            if batch_col:
                fig.update_xaxes(title_text=batch_col, row=4, col=4, tickangle=45)
                fig.update_yaxes(title_text="Normalized Metrics", row=4, col=4)

            # Update overall layout with better dimensions for web display
            fig.update_layout(
                title={
                    "text": title
                    or "Comprehensive Single-Cell RNA-seq Quality Control Report",
                    "font": {"size": 18, "family": "Arial, sans-serif"},
                    "x": 0.5,
                    "xanchor": "center",
                },
                width=1400,
                height=1200,
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=9),
                margin=dict(l=50, r=50, t=80, b=60),
            )

            # Add annotation with dataset info
            dataset_info = (
                f"Dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes | "
                f"High Quality Cells: {n_high_quality:,} ({n_high_quality/adata.n_obs*100:.1f}%) | "
                f"Mean MT%: {adata.obs['percent_mito'].mean():.1f}% | "
                f"Mean UMIs/cell: {adata.obs['n_counts'].mean():.0f}"
            )

            fig.add_annotation(
                text=dataset_info,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.02,
                xanchor="center",
                showarrow=False,
                font=dict(size=11, color="gray"),
            )

            logger.info(f"Created comprehensive QC plot with {adata.n_obs:,} cells")
            return fig

        except Exception as e:
            logger.error(f"Error creating QC plots: {e}")
            raise VisualizationError(f"Failed to create QC plots: {str(e)}")

    def create_cluster_composition_plot(
        self,
        adata: anndata.AnnData,
        cluster_col: str = "leiden",
        sample_col: Optional[str] = None,
        normalize: bool = True,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Create a stacked bar plot showing cluster composition.

        Args:
            adata: AnnData object
            cluster_col: Column with cluster assignments
            sample_col: Column with sample/batch info (auto-detect if None)
            normalize: Whether to normalize to percentages
            title: Plot title

        Returns:
            go.Figure: Stacked bar plot
        """
        try:
            # Validate cluster column
            if cluster_col not in adata.obs.columns:
                raise VisualizationError(f"'{cluster_col}' not found in obs columns")

            # Auto-detect sample column if not specified
            if sample_col is None:
                for col in ["batch", "sample", "patient", "Patient_ID"]:
                    if col in adata.obs.columns:
                        sample_col = col
                        break

            if sample_col is None:
                # If no sample column, just show cluster sizes
                cluster_counts = adata.obs[cluster_col].value_counts()

                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=cluster_counts.index.astype(str),
                            y=cluster_counts.values,
                            marker_color="steelblue",
                        )
                    ]
                )

                fig.update_layout(
                    title=title or f"Cluster Sizes - {cluster_col}",
                    xaxis_title=cluster_col,
                    yaxis_title="Number of Cells",
                    width=800,
                    height=500,
                )

            else:
                # Create composition matrix
                composition = pd.crosstab(
                    adata.obs[sample_col],
                    adata.obs[cluster_col],
                    normalize="index" if normalize else False,
                )

                if normalize:
                    composition = composition * 100

                # Create stacked bar plot
                fig = go.Figure()

                for cluster in composition.columns:
                    fig.add_trace(
                        go.Bar(
                            name=f"Cluster {cluster}",
                            x=composition.index,
                            y=composition[cluster],
                        )
                    )

                fig.update_layout(
                    title=title or f"Cluster Composition by {sample_col}",
                    xaxis_title=sample_col,
                    yaxis_title="Percentage" if normalize else "Number of Cells",
                    barmode="stack",
                    width=max(800, 50 * len(composition.index)),
                    height=600,
                    legend=dict(
                        orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.01
                    ),
                )

            return fig

        except Exception as e:
            logger.error(f"Error creating cluster composition plot: {e}")
            raise VisualizationError(
                f"Failed to create cluster composition plot: {str(e)}"
            )

    def save_all_plots(
        self, plots: Dict[str, go.Figure], output_dir: str, format: str = "both"
    ) -> List[str]:
        """
        Save multiple plots to files.

        Args:
            plots: Dictionary of plot_name: figure pairs
            output_dir: Directory to save plots
            format: 'html', 'png', or 'both'

        Returns:
            List[str]: Paths to saved files
        """
        import os
        from pathlib import Path

        saved_files = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, fig in plots.items():
            try:
                if format in ["html", "both"]:
                    html_path = output_path / f"{name}.html"
                    pio.write_html(fig, html_path)
                    saved_files.append(str(html_path))
                    logger.info(f"Saved HTML: {html_path}")

                if format in ["png", "both"]:
                    png_path = output_path / f"{name}.png"
                    pio.write_image(fig, png_path, width=3200, height=2400, scale=2)
                    saved_files.append(str(png_path))
                    logger.info(f"Saved PNG: {png_path}")

            except Exception as e:
                logger.error(f"Failed to save plot '{name}': {e}")

        return saved_files

    def extract_plotly_color_palette(self, fig: go.Figure) -> Dict[str, str]:
        """
        Extract cluster colors from Plotly figure for Rich terminal sync.

        Args:
            fig: Plotly figure with colored clusters

        Returns:
            Dict mapping cluster IDs to hex color codes
        """
        try:
            color_palette = {}

            # Extract colors from scatter plot traces
            for trace in fig.data:
                if hasattr(trace, "marker") and hasattr(trace.marker, "color"):
                    # Handle discrete colors (categorical data)
                    if hasattr(trace, "name") and trace.name:
                        # Get color from marker
                        if hasattr(trace.marker, "color") and isinstance(
                            trace.marker.color, str
                        ):
                            color_palette[trace.name] = trace.marker.color
                        elif hasattr(trace.marker, "colorscale"):
                            # For continuous color scales, use a representative color
                            color_palette[trace.name] = "#1f77b4"  # Default blue

                # Handle text annotations or hover data that might contain cluster IDs
                if hasattr(trace, "customdata") and trace.customdata:
                    # Extract cluster information from custom data
                    # This would need to be customized based on how the plot was created
                    pass

            # If no colors found, generate default colors
            if not color_palette:
                import plotly.express as px

                default_colors = px.colors.qualitative.Set1
                color_palette = {
                    f"cluster_{i}": default_colors[i % len(default_colors)]
                    for i in range(10)
                }

            return color_palette

        except Exception as e:
            logger.error(f"Error extracting color palette: {e}")
            return {}

    def create_annotation_umap_with_palette(
        self, adata: anndata.AnnData, cluster_col: str = "leiden"
    ) -> Tuple[go.Figure, Dict[str, str]]:
        """
        Generate UMAP plot and extract color palette for Rich terminal sync.

        Args:
            adata: AnnData object with UMAP coordinates
            cluster_col: Column name for clustering

        Returns:
            Tuple of (plotly figure, color palette dict)
        """
        try:
            # Create standard UMAP plot
            fig = self.create_umap_plot(
                adata=adata,
                color_by=cluster_col,
                title=f"UMAP - {cluster_col} (Color Synchronized)",
            )

            # Extract color mapping from the plot
            color_palette = {}
            unique_clusters = adata.obs[cluster_col].unique()

            # Get colors from plot traces
            for i, trace in enumerate(fig.data):
                if hasattr(trace, "marker") and hasattr(trace.marker, "color"):
                    if isinstance(trace.marker.color, str):
                        # Single color for this trace
                        cluster_id = (
                            str(unique_clusters[i])
                            if i < len(unique_clusters)
                            else f"cluster_{i}"
                        )
                        color_palette[cluster_id] = trace.marker.color
                    elif hasattr(trace.marker, "colorscale"):
                        # Continuous color scale - extract discrete colors
                        colors = self._extract_discrete_colors_from_continuous(
                            trace.marker.color, len(unique_clusters)
                        )
                        for j, cluster_id in enumerate(unique_clusters):
                            if j < len(colors):
                                color_palette[str(cluster_id)] = colors[j]

            # Fallback: generate colors if extraction failed
            if not color_palette:
                color_palette = self._generate_cluster_colors(unique_clusters)

            # Add color information to figure metadata
            fig.update_layout(
                annotations=[
                    dict(
                        text=f"Colors synchronized with Rich terminal interface",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.02,
                        y=0.98,
                        xanchor="left",
                        yanchor="top",
                        font=dict(size=10, color="gray"),
                    )
                ]
            )

            return fig, color_palette

        except Exception as e:
            logger.error(f"Error creating annotation UMAP with palette: {e}")
            # Return basic plot without color sync
            fig = self.create_umap_plot(adata, color_by=cluster_col)
            fallback_colors = self._generate_cluster_colors(
                adata.obs[cluster_col].unique()
            )
            return fig, fallback_colors

    def _extract_discrete_colors_from_continuous(
        self, color_data: List, n_colors: int
    ) -> List[str]:
        """
        Extract discrete colors from continuous color data.

        Args:
            color_data: Continuous color data from plotly trace
            n_colors: Number of discrete colors needed

        Returns:
            List of hex color codes
        """
        try:
            import numpy as np

            # Convert color data to numpy array
            color_array = np.array(color_data)

            # Get unique values and map to discrete colors
            unique_values = np.unique(color_array)

            # Use plotly's default color scale
            import plotly.express as px

            colors = px.colors.sample_colorscale(
                "viridis",
                [i / (len(unique_values) - 1) for i in range(len(unique_values))],
            )

            return colors[:n_colors]

        except Exception as e:
            logger.error(f"Error extracting discrete colors: {e}")
            return self._generate_cluster_colors(list(range(n_colors)))

    def _generate_cluster_colors(self, cluster_ids: List) -> Dict[str, str]:
        """
        Generate colors for clusters using plotly's default palette.

        Args:
            cluster_ids: List of cluster identifiers

        Returns:
            Dict mapping cluster IDs to hex colors
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Use matplotlib's tab20 colormap for consistency with scanpy
            n_clusters = len(cluster_ids)
            colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

            color_palette = {}
            for i, cluster_id in enumerate(cluster_ids):
                hex_color = f"#{int(colors[i][0]*255):02x}{int(colors[i][1]*255):02x}{int(colors[i][2]*255):02x}"
                color_palette[str(cluster_id)] = hex_color

            return color_palette

        except Exception as e:
            logger.error(f"Error generating cluster colors: {e}")
            # Fallback to simple colors
            simple_colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
            return {
                str(cluster_id): simple_colors[i % len(simple_colors)]
                for i, cluster_id in enumerate(cluster_ids)
            }

    def rich_color_from_hex(self, hex_color: str) -> str:
        """
        Convert hex color to Rich terminal color format.

        Args:
            hex_color: Hex color code (e.g., "#1f77b4")

        Returns:
            Rich-compatible color string
        """
        try:
            # Remove # if present
            hex_color = hex_color.lstrip("#")

            # Rich supports hex colors directly
            return f"#{hex_color}"

        except Exception as e:
            logger.error(f"Error converting hex color {hex_color}: {e}")
            return "white"  # Fallback color

    def update_plot_with_annotations(
        self,
        fig: go.Figure,
        adata: anndata.AnnData,
        annotation_col: str = "cell_type_manual",
    ) -> go.Figure:
        """
        Update existing plot with new annotation colors.

        Args:
            fig: Existing plotly figure
            adata: AnnData object with annotations
            annotation_col: Column containing cell type annotations

        Returns:
            Updated plotly figure
        """
        try:
            if annotation_col not in adata.obs.columns:
                logger.warning(f"Annotation column {annotation_col} not found")
                return fig

            # Create new UMAP plot with annotations
            updated_fig = self.create_umap_plot(
                adata=adata,
                color_by=annotation_col,
                title=f"UMAP - Manual Annotations ({annotation_col})",
            )

            # Preserve original layout settings
            updated_fig.update_layout(
                width=fig.layout.width, height=fig.layout.height, showlegend=True
            )

            return updated_fig

        except Exception as e:
            logger.error(f"Error updating plot with annotations: {e}")
            return fig
