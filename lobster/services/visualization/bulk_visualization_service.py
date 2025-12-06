"""
Bulk RNA-seq visualization service for differential expression results.

This service provides comprehensive visualization methods specifically designed for
bulk RNA-seq differential expression analysis, generating interactive and
publication-quality plots using Plotly.
"""

from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import issparse

from lobster.core.analysis_ir import AnalysisStep
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class BulkVisualizationError(Exception):
    """Base exception for bulk RNA-seq visualization operations."""

    pass


class BulkVisualizationService:
    """
    Professional visualization service for bulk RNA-seq differential expression.

    This class provides comprehensive visualization methods including volcano plots,
    MA plots, and expression heatmaps. All plots are interactive using Plotly for
    publication-quality figures.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the bulk visualization service.

        Args:
            config: Optional configuration dict (ignored, for backward compatibility)
            **kwargs: Additional arguments (ignored, for backward compatibility)
        """
        logger.debug("Initializing BulkVisualizationService")

        # Color palettes for bulk RNA-seq visualizations
        self.significance_colors = {
            "up": "red",
            "down": "blue",
            "not_significant": "lightgray",
        }
        self.diverging_colors = px.colors.diverging.RdBu_r

        # Default plot settings
        self.default_width = 900
        self.default_height = 700
        self.default_marker_size = 5
        self.default_opacity = 0.7

        logger.debug("BulkVisualizationService initialized successfully")

    def create_volcano_plot(
        self,
        adata: anndata.AnnData,
        fdr_threshold: float = 0.05,
        fc_threshold: float = 1.0,
        top_n_genes: int = 10,
        title: Optional[str] = None,
    ) -> Tuple[go.Figure, Dict[str, Any], AnalysisStep]:
        """
        Create a volcano plot for differential expression results.

        Args:
            adata: AnnData object with DE results in .var
            fdr_threshold: FDR significance threshold (default: 0.05)
            fc_threshold: Log2 fold-change threshold (default: 1.0)
            top_n_genes: Number of top genes to label (default: 10)
            title: Plot title

        Returns:
            Tuple[go.Figure, Dict[str, Any], AnalysisStep]: Interactive volcano plot,
                statistics, and IR for provenance tracking

        Raises:
            BulkVisualizationError: If required columns are missing
        """
        try:
            logger.info("Creating volcano plot for bulk RNA-seq DE results")

            # Validate required columns
            required_cols = ["log2FoldChange", "padj"]
            missing_cols = [
                col for col in required_cols if col not in adata.var.columns
            ]
            if missing_cols:
                raise BulkVisualizationError(
                    f"Missing required columns in adata.var: {missing_cols}. "
                    f"Available columns: {list(adata.var.columns)}"
                )

            # Extract data
            log2fc = adata.var["log2FoldChange"].values
            padj = adata.var["padj"].fillna(1.0).values  # Fill NaN with 1.0
            gene_names = adata.var_names.values

            # Calculate -log10(padj)
            neg_log_padj = -np.log10(padj + 1e-300)  # Add small value to avoid log(0)

            # Determine significance categories
            significant_up = (log2fc > fc_threshold) & (padj < fdr_threshold)
            significant_down = (log2fc < -fc_threshold) & (padj < fdr_threshold)
            not_significant = ~(significant_up | significant_down)

            # Count significant genes
            n_up = int(np.sum(significant_up))
            n_down = int(np.sum(significant_down))
            n_total = len(gene_names)

            # Create figure
            fig = go.Figure()

            # Add non-significant points
            fig.add_trace(
                go.Scatter(
                    x=log2fc[not_significant],
                    y=neg_log_padj[not_significant],
                    mode="markers",
                    name="Not significant",
                    marker=dict(
                        color=self.significance_colors["not_significant"],
                        size=self.default_marker_size,
                        opacity=0.4,
                    ),
                    text=gene_names[not_significant],
                    hovertemplate="Gene: %{text}<br>log2FC: %{x:.2f}<br>-log10(FDR): %{y:.2f}<extra></extra>",
                    showlegend=True,
                )
            )

            # Add upregulated points
            if n_up > 0:
                fig.add_trace(
                    go.Scatter(
                        x=log2fc[significant_up],
                        y=neg_log_padj[significant_up],
                        mode="markers",
                        name=f"Upregulated ({n_up})",
                        marker=dict(
                            color=self.significance_colors["up"],
                            size=self.default_marker_size + 1,
                            opacity=self.default_opacity,
                        ),
                        text=gene_names[significant_up],
                        hovertemplate="Gene: %{text}<br>log2FC: %{x:.2f}<br>-log10(FDR): %{y:.2f}<extra></extra>",
                        showlegend=True,
                    )
                )

            # Add downregulated points
            if n_down > 0:
                fig.add_trace(
                    go.Scatter(
                        x=log2fc[significant_down],
                        y=neg_log_padj[significant_down],
                        mode="markers",
                        name=f"Downregulated ({n_down})",
                        marker=dict(
                            color=self.significance_colors["down"],
                            size=self.default_marker_size + 1,
                            opacity=self.default_opacity,
                        ),
                        text=gene_names[significant_down],
                        hovertemplate="Gene: %{text}<br>log2FC: %{x:.2f}<br>-log10(FDR): %{y:.2f}<extra></extra>",
                        showlegend=True,
                    )
                )

            # Label top N genes by combined score
            if top_n_genes > 0:
                # Calculate combined score (absolute log2FC * -log10(padj))
                combined_score = np.abs(log2fc) * neg_log_padj
                top_indices = np.argsort(combined_score)[-top_n_genes:]

                for idx in top_indices:
                    if padj[idx] < fdr_threshold and np.abs(log2fc[idx]) > fc_threshold:
                        fig.add_annotation(
                            x=log2fc[idx],
                            y=neg_log_padj[idx],
                            text=gene_names[idx],
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor="black",
                            ax=20 if log2fc[idx] > 0 else -20,
                            ay=-20,
                            font=dict(size=9, color="black"),
                            bgcolor="rgba(255,255,255,0.8)",
                            borderpad=2,
                        )

            # Add threshold lines
            fig.add_hline(
                y=-np.log10(fdr_threshold),
                line_dash="dash",
                line_color="darkgray",
                annotation_text=f"FDR = {fdr_threshold}",
                annotation_position="right",
            )
            fig.add_vline(
                x=fc_threshold,
                line_dash="dash",
                line_color="darkgray",
            )
            fig.add_vline(
                x=-fc_threshold,
                line_dash="dash",
                line_color="darkgray",
            )

            # Update layout
            fig.update_layout(
                title=title or f"Volcano Plot ({n_up} up, {n_down} down)",
                xaxis_title="log2 Fold Change",
                yaxis_title="-log10(FDR)",
                width=self.default_width,
                height=self.default_height,
                plot_bgcolor="white",
                hovermode="closest",
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                ),
            )

            # Update axes with proper padding (25% buffer on x-axis)
            x_min, x_max = float(np.min(log2fc)), float(np.max(log2fc))
            x_range = x_max - x_min
            x_padding = max(0.25, x_range * 0.25)  # At least 0.25 or 25% of range
            fig.update_xaxes(
                showgrid=True,
                gridcolor="lightgray",
                zeroline=True,
                range=[x_min - x_padding, x_max + x_padding],
            )
            fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=True)

            # Generate statistics
            stats = {
                "plot_type": "volcano_plot",
                "n_genes_total": n_total,
                "n_genes_up": n_up,
                "n_genes_down": n_down,
                "n_genes_not_significant": int(np.sum(not_significant)),
                "fdr_threshold": fdr_threshold,
                "fc_threshold": fc_threshold,
                "top_n_genes_labeled": min(top_n_genes, n_up + n_down),
            }

            # Create IR for provenance tracking
            ir = self._create_volcano_ir(
                fdr_threshold=fdr_threshold,
                fc_threshold=fc_threshold,
                top_n_genes=top_n_genes,
            )

            logger.info(f"Volcano plot created: {n_up} up, {n_down} down genes")
            return fig, stats, ir

        except Exception as e:
            logger.error(f"Error creating volcano plot: {e}")
            raise BulkVisualizationError(f"Failed to create volcano plot: {str(e)}")

    def create_ma_plot(
        self,
        adata: anndata.AnnData,
        fdr_threshold: float = 0.05,
        title: Optional[str] = None,
    ) -> Tuple[go.Figure, Dict[str, Any], AnalysisStep]:
        """
        Create an MA plot for differential expression results.

        Args:
            adata: AnnData object with DE results in .var
            fdr_threshold: FDR significance threshold (default: 0.05)
            title: Plot title

        Returns:
            Tuple[go.Figure, Dict[str, Any], AnalysisStep]: Interactive MA plot,
                statistics, and IR for provenance tracking

        Raises:
            BulkVisualizationError: If required columns are missing
        """
        try:
            logger.info("Creating MA plot for bulk RNA-seq DE results")

            # Validate required columns
            required_cols = ["log2FoldChange", "padj", "baseMean"]
            missing_cols = [
                col for col in required_cols if col not in adata.var.columns
            ]
            if missing_cols:
                raise BulkVisualizationError(
                    f"Missing required columns in adata.var: {missing_cols}. "
                    f"Available columns: {list(adata.var.columns)}"
                )

            # Extract data
            log2fc = adata.var["log2FoldChange"].values
            padj = adata.var["padj"].fillna(1.0).values
            base_mean = adata.var["baseMean"].values
            gene_names = adata.var_names.values

            # Calculate log10(baseMean)
            log_base_mean = np.log10(base_mean + 1)  # Add 1 to avoid log(0)

            # Determine significance
            significant = padj < fdr_threshold
            n_significant = int(np.sum(significant))
            n_total = len(gene_names)

            # Create figure
            fig = go.Figure()

            # Add non-significant points
            fig.add_trace(
                go.Scatter(
                    x=log_base_mean[~significant],
                    y=log2fc[~significant],
                    mode="markers",
                    name="Not significant",
                    marker=dict(
                        color=self.significance_colors["not_significant"],
                        size=self.default_marker_size,
                        opacity=0.4,
                    ),
                    text=gene_names[~significant],
                    hovertemplate="Gene: %{text}<br>log10(baseMean): %{x:.2f}<br>log2FC: %{y:.2f}<extra></extra>",
                    showlegend=True,
                )
            )

            # Add significant points
            if n_significant > 0:
                # Color by up/down regulation
                colors = [
                    (
                        self.significance_colors["up"]
                        if fc > 0
                        else self.significance_colors["down"]
                    )
                    for fc in log2fc[significant]
                ]

                fig.add_trace(
                    go.Scatter(
                        x=log_base_mean[significant],
                        y=log2fc[significant],
                        mode="markers",
                        name=f"Significant ({n_significant})",
                        marker=dict(
                            color=colors,
                            size=self.default_marker_size + 1,
                            opacity=self.default_opacity,
                        ),
                        text=gene_names[significant],
                        hovertemplate="Gene: %{text}<br>log10(baseMean): %{x:.2f}<br>log2FC: %{y:.2f}<extra></extra>",
                        showlegend=True,
                    )
                )

            # Add horizontal line at y=0
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="darkgray",
                annotation_text="No change",
                annotation_position="right",
            )

            # Update layout
            fig.update_layout(
                title=title or f"MA Plot ({n_significant} significant genes)",
                xaxis_title="log10(Mean Expression)",
                yaxis_title="log2 Fold Change",
                width=self.default_width,
                height=self.default_height,
                plot_bgcolor="white",
                hovermode="closest",
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                ),
            )

            # Update axes
            fig.update_xaxes(showgrid=True, gridcolor="lightgray", zeroline=True)
            fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=True)

            # Generate statistics
            stats = {
                "plot_type": "ma_plot",
                "n_genes_total": n_total,
                "n_genes_significant": n_significant,
                "fdr_threshold": fdr_threshold,
                "mean_base_mean": float(np.mean(base_mean)),
                "median_base_mean": float(np.median(base_mean)),
            }

            # Create IR for provenance tracking
            ir = self._create_ma_ir(fdr_threshold=fdr_threshold)

            logger.info(f"MA plot created: {n_significant} significant genes")
            return fig, stats, ir

        except Exception as e:
            logger.error(f"Error creating MA plot: {e}")
            raise BulkVisualizationError(f"Failed to create MA plot: {str(e)}")

    def create_expression_heatmap(
        self,
        adata: anndata.AnnData,
        gene_list: Optional[List[str]] = None,
        cluster_samples: bool = True,
        cluster_genes: bool = True,
        z_score: bool = True,
        title: Optional[str] = None,
    ) -> Tuple[go.Figure, Dict[str, Any], AnalysisStep]:
        """
        Create a hierarchical clustered heatmap of gene expression.

        Args:
            adata: AnnData object with expression data
            gene_list: List of genes to include (uses all if None)
            cluster_samples: Whether to cluster samples hierarchically
            cluster_genes: Whether to cluster genes hierarchically
            z_score: Whether to z-score normalize expression
            title: Plot title

        Returns:
            Tuple[go.Figure, Dict[str, Any], AnalysisStep]: Interactive heatmap,
                statistics, and IR for provenance tracking

        Raises:
            BulkVisualizationError: If gene list is invalid
        """
        try:
            logger.info("Creating expression heatmap for bulk RNA-seq data")

            # Filter genes if gene_list provided
            if gene_list is not None:
                # Validate gene list
                missing_genes = [g for g in gene_list if g not in adata.var_names]
                if missing_genes:
                    logger.warning(f"Genes not found: {missing_genes[:10]}...")
                    gene_list = [g for g in gene_list if g in adata.var_names]

                if len(gene_list) == 0:
                    raise BulkVisualizationError("No valid genes found in gene_list")

                # Subset to gene list
                adata_subset = adata[:, gene_list].copy()
            else:
                # Use top 50 most variable genes if no list provided
                if adata.n_vars > 50:
                    # Calculate variance
                    X = adata.X
                    if issparse(X):
                        X = X.toarray()
                    gene_var = np.var(X, axis=0)
                    top_indices = np.argsort(gene_var)[-50:]
                    adata_subset = adata[:, top_indices].copy()
                else:
                    adata_subset = adata.copy()

            # Extract expression matrix
            X = adata_subset.X
            if issparse(X):
                X = X.toarray()

            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)

            # Z-score normalization if requested
            if z_score:
                # Normalize by gene (row-wise)
                X_mean = X.mean(axis=0, keepdims=True)
                X_std = X.std(axis=0, keepdims=True)
                X_std[X_std == 0] = 1  # Avoid division by zero
                X = (X - X_mean) / X_std

            # Get sample and gene names
            sample_names = adata_subset.obs_names.values
            gene_names = adata_subset.var_names.values

            # Cluster samples if requested
            if cluster_samples and adata_subset.n_obs > 1:
                try:
                    sample_linkage = linkage(X, method="ward")
                    sample_dendro = dendrogram(sample_linkage, no_plot=True)
                    sample_order = sample_dendro["leaves"]
                    X = X[sample_order, :]
                    sample_names = sample_names[sample_order]
                except Exception as e:
                    logger.warning(f"Sample clustering failed: {e}")

            # Cluster genes if requested
            if cluster_genes and adata_subset.n_vars > 1:
                try:
                    gene_linkage = linkage(X.T, method="ward")
                    gene_dendro = dendrogram(gene_linkage, no_plot=True)
                    gene_order = gene_dendro["leaves"]
                    X = X[:, gene_order]
                    gene_names = gene_names[gene_order]
                except Exception as e:
                    logger.warning(f"Gene clustering failed: {e}")

            # Create heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=X.T,  # Transpose so genes are rows
                    x=sample_names,
                    y=gene_names,
                    colorscale=self.diverging_colors if z_score else "Viridis",
                    colorbar=dict(
                        title="Expression<br>(z-score)" if z_score else "Expression"
                    ),
                    hovertemplate="Sample: %{x}<br>Gene: %{y}<br>Expression: %{z:.2f}<extra></extra>",
                )
            )

            # Update layout
            fig.update_layout(
                title=title
                or f"Gene Expression Heatmap ({len(gene_names)} genes, {len(sample_names)} samples)",
                xaxis_title="Samples",
                yaxis_title="Genes",
                width=max(self.default_width, 40 * len(sample_names)),
                height=max(self.default_height, 15 * len(gene_names)),
                plot_bgcolor="white",
            )

            # Update axes
            fig.update_xaxes(tickangle=45, tickfont=dict(size=9))
            fig.update_yaxes(tickfont=dict(size=8))

            # Generate statistics
            stats = {
                "plot_type": "expression_heatmap",
                "n_samples": len(sample_names),
                "n_genes": len(gene_names),
                "clustered_samples": cluster_samples,
                "clustered_genes": cluster_genes,
                "z_score_normalized": z_score,
            }

            # Create IR for provenance tracking
            ir = self._create_heatmap_ir(
                n_genes=len(gene_names),
                cluster_samples=cluster_samples,
                cluster_genes=cluster_genes,
                z_score=z_score,
            )

            logger.info(
                f"Expression heatmap created: {len(gene_names)} genes × {len(sample_names)} samples"
            )
            return fig, stats, ir

        except Exception as e:
            logger.error(f"Error creating expression heatmap: {e}")
            raise BulkVisualizationError(
                f"Failed to create expression heatmap: {str(e)}"
            )

    # -------------------------
    # IR HELPER METHODS
    # -------------------------

    def _create_volcano_ir(
        self,
        fdr_threshold: float,
        fc_threshold: float,
        top_n_genes: int,
    ) -> AnalysisStep:
        """Create IR for volcano plot."""
        return AnalysisStep(
            operation="visualization.volcano_plot",
            tool_name="create_volcano_plot",
            description="Create volcano plot of differential expression results",
            library="plotly",
            code_template="""
import plotly.graph_objects as go
import numpy as np

# Extract DE results
log2fc = adata.var['log2FoldChange'].values
padj = adata.var['padj'].fillna(1.0).values
gene_names = adata.var_names.values

# Calculate -log10(padj)
neg_log_padj = -np.log10(padj + 1e-300)

# Determine significance
significant_up = (log2fc > {{ fc_threshold }}) & (padj < {{ fdr_threshold }})
significant_down = (log2fc < -{{ fc_threshold }}) & (padj < {{ fdr_threshold }})
not_significant = ~(significant_up | significant_down)

# Create plot
fig = go.Figure()

# Add points by significance
fig.add_trace(go.Scatter(
    x=log2fc[not_significant],
    y=neg_log_padj[not_significant],
    mode='markers',
    name='Not significant',
    marker=dict(color='lightgray', size=5, opacity=0.4)
))

fig.add_trace(go.Scatter(
    x=log2fc[significant_up],
    y=neg_log_padj[significant_up],
    mode='markers',
    name='Upregulated',
    marker=dict(color='red', size=6, opacity=0.7)
))

fig.add_trace(go.Scatter(
    x=log2fc[significant_down],
    y=neg_log_padj[significant_down],
    mode='markers',
    name='Downregulated',
    marker=dict(color='blue', size=6, opacity=0.7)
))

# Add threshold lines
fig.add_hline(y=-np.log10({{ fdr_threshold }}), line_dash="dash", line_color="darkgray")
fig.add_vline(x={{ fc_threshold }}, line_dash="dash", line_color="darkgray")
fig.add_vline(x=-{{ fc_threshold }}, line_dash="dash", line_color="darkgray")

# Update layout
fig.update_layout(
    title="Volcano Plot",
    xaxis_title="log2 Fold Change",
    yaxis_title="-log10(FDR)",
    template="plotly_white"
)

fig.show()
""",
            imports=[
                "import plotly.graph_objects as go",
                "import numpy as np",
            ],
            parameters={
                "fdr_threshold": fdr_threshold,
                "fc_threshold": fc_threshold,
                "top_n_genes": top_n_genes,
            },
            parameter_schema={
                "fdr_threshold": {
                    "type": "float",
                    "default": 0.05,
                    "description": "FDR significance threshold",
                },
                "fc_threshold": {
                    "type": "float",
                    "default": 1.0,
                    "description": "Log2 fold-change threshold",
                },
                "top_n_genes": {
                    "type": "int",
                    "default": 10,
                    "description": "Number of top genes to label",
                },
            },
            input_entities=[{"name": "de_results", "type": "AnnData"}],
            output_entities=[{"name": "volcano_plot", "type": "plotly.Figure"}],
        )

    def _create_ma_ir(self, fdr_threshold: float) -> AnalysisStep:
        """Create IR for MA plot."""
        return AnalysisStep(
            operation="visualization.ma_plot",
            tool_name="create_ma_plot",
            description="Create MA plot of differential expression results",
            library="plotly",
            code_template="""
import plotly.graph_objects as go
import numpy as np

# Extract DE results
log2fc = adata.var['log2FoldChange'].values
padj = adata.var['padj'].fillna(1.0).values
base_mean = adata.var['baseMean'].values
gene_names = adata.var_names.values

# Calculate log10(baseMean)
log_base_mean = np.log10(base_mean + 1)

# Determine significance
significant = padj < {{ fdr_threshold }}

# Create plot
fig = go.Figure()

# Add non-significant points
fig.add_trace(go.Scatter(
    x=log_base_mean[~significant],
    y=log2fc[~significant],
    mode='markers',
    name='Not significant',
    marker=dict(color='lightgray', size=5, opacity=0.4)
))

# Add significant points
if significant.any():
    colors = ['red' if fc > 0 else 'blue' for fc in log2fc[significant]]
    fig.add_trace(go.Scatter(
        x=log_base_mean[significant],
        y=log2fc[significant],
        mode='markers',
        name='Significant',
        marker=dict(color=colors, size=6, opacity=0.7)
    ))

# Add horizontal line at y=0
fig.add_hline(y=0, line_dash="dash", line_color="darkgray")

# Update layout
fig.update_layout(
    title="MA Plot",
    xaxis_title="log10(Mean Expression)",
    yaxis_title="log2 Fold Change",
    template="plotly_white"
)

fig.show()
""",
            imports=[
                "import plotly.graph_objects as go",
                "import numpy as np",
            ],
            parameters={
                "fdr_threshold": fdr_threshold,
            },
            parameter_schema={
                "fdr_threshold": {
                    "type": "float",
                    "default": 0.05,
                    "description": "FDR significance threshold",
                },
            },
            input_entities=[{"name": "de_results", "type": "AnnData"}],
            output_entities=[{"name": "ma_plot", "type": "plotly.Figure"}],
        )

    def _create_heatmap_ir(
        self,
        n_genes: int,
        cluster_samples: bool,
        cluster_genes: bool,
        z_score: bool,
    ) -> AnalysisStep:
        """Create IR for expression heatmap."""
        return AnalysisStep(
            operation="visualization.expression_heatmap",
            tool_name="create_expression_heatmap",
            description="Create hierarchical clustered heatmap of gene expression",
            library="plotly",
            code_template="""
import plotly.graph_objects as go
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.sparse import issparse

# Extract expression matrix
X = adata.X
if issparse(X):
    X = X.toarray()

# Handle missing values
X = np.nan_to_num(X, nan=0.0)

# Z-score normalization if requested
if {{ z_score }}:
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True)
    X_std[X_std == 0] = 1
    X = (X - X_mean) / X_std

sample_names = adata.obs_names.values
gene_names = adata.var_names.values

# Cluster samples
if {{ cluster_samples }} and len(sample_names) > 1:
    sample_linkage = linkage(X, method="ward")
    sample_dendro = dendrogram(sample_linkage, no_plot=True)
    sample_order = sample_dendro["leaves"]
    X = X[sample_order, :]
    sample_names = sample_names[sample_order]

# Cluster genes
if {{ cluster_genes }} and len(gene_names) > 1:
    gene_linkage = linkage(X.T, method="ward")
    gene_dendro = dendrogram(gene_linkage, no_plot=True)
    gene_order = gene_dendro["leaves"]
    X = X[:, gene_order]
    gene_names = gene_names[gene_order]

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=X.T,
    x=sample_names,
    y=gene_names,
    colorscale="RdBu_r" if {{ z_score }} else "Viridis",
    colorbar=dict(title="Expression (z-score)" if {{ z_score }} else "Expression")
))

fig.update_layout(
    title="Gene Expression Heatmap",
    xaxis_title="Samples",
    yaxis_title="Genes",
    template="plotly_white"
)

fig.show()
""",
            imports=[
                "import plotly.graph_objects as go",
                "import numpy as np",
                "from scipy.cluster.hierarchy import linkage, dendrogram",
                "from scipy.sparse import issparse",
            ],
            parameters={
                "n_genes": n_genes,
                "cluster_samples": cluster_samples,
                "cluster_genes": cluster_genes,
                "z_score": z_score,
            },
            parameter_schema={
                "n_genes": {
                    "type": "int",
                    "description": "Number of genes in heatmap",
                },
                "cluster_samples": {
                    "type": "bool",
                    "default": True,
                    "description": "Whether to cluster samples hierarchically",
                },
                "cluster_genes": {
                    "type": "bool",
                    "default": True,
                    "description": "Whether to cluster genes hierarchically",
                },
                "z_score": {
                    "type": "bool",
                    "default": True,
                    "description": "Whether to z-score normalize expression",
                },
            },
            input_entities=[{"name": "expression_data", "type": "AnnData"}],
            output_entities=[{"name": "expression_heatmap", "type": "plotly.Figure"}],
        )
