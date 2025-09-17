"""
Proteomics visualization service for mass spectrometry and affinity proteomics data.

This service provides comprehensive visualization methods specifically designed for 
proteomics data analysis, generating interactive and publication-quality plots using Plotly.
Handles missing value patterns, intensity distributions, and platform-specific requirements.
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import warnings

import anndata
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats
from scipy.sparse import issparse
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteomicsVisualizationError(Exception):
    """Base exception for proteomics visualization operations."""
    pass


class ProteomicsVisualizationService:
    """
    Professional visualization service for proteomics data.
    
    This class provides comprehensive visualization methods specifically designed for
    mass spectrometry and affinity proteomics data, including missing value analysis,
    intensity distributions, differential expression, pathway enrichment, and QC plots.
    All plots are interactive using Plotly for publication-quality figures.
    """

    def __init__(self):
        """Initialize the proteomics visualization service."""
        logger.debug("Initializing ProteomicsVisualizationService")
        
        # Color palettes optimized for proteomics data
        self.intensity_colors = px.colors.sequential.Viridis
        self.missing_colors = ['lightgray', 'darkred']  # Missing vs Present
        self.significance_colors = ['gray', 'red', 'blue']  # Non-sig, Up, Down
        self.platform_colors = px.colors.qualitative.Set2
        self.cv_colors = px.colors.sequential.Reds
        
        # Default plot settings
        self.default_width = 900
        self.default_height = 700
        self.default_marker_size = 4
        self.default_opacity = 0.7
        
        # Proteomics-specific color scales
        self.intensity_colorscale = [
            [0, 'lightgray'],      # Zero/missing values
            [0.001, 'lightblue'],  # Very low intensity
            [0.1, 'blue'],         # Low intensity
            [0.3, 'green'],        # Medium intensity
            [0.6, 'yellow'],       # High intensity
            [0.8, 'orange'],       # Very high intensity
            [1.0, 'red']           # Maximum intensity
        ]
        
        # Significance thresholds
        self.default_pvalue_threshold = 0.05
        self.default_fc_threshold = 1.5  # Fold change threshold
        
        logger.debug("ProteomicsVisualizationService initialized successfully")

    def create_missing_value_heatmap(
        self,
        adata: anndata.AnnData,
        max_proteins: int = 500,
        max_samples: int = 100,
        cluster_samples: bool = True,
        cluster_proteins: bool = True,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create a heatmap showing missing value patterns across samples and proteins.
        
        Args:
            adata: AnnData object with proteomics data
            max_proteins: Maximum number of proteins to display
            max_samples: Maximum number of samples to display
            cluster_samples: Whether to cluster samples by missing pattern
            cluster_proteins: Whether to cluster proteins by missing pattern
            title: Plot title
            
        Returns:
            go.Figure: Interactive missing value heatmap
            
        Raises:
            ProteomicsVisualizationError: If visualization fails
        """
        try:
            logger.info("Creating missing value heatmap")
            
            X = adata.X.copy()
            if issparse(X):
                X = X.toarray()
            
            # Create missing value matrix (1 = missing, 0 = present)
            missing_matrix = np.isnan(X).astype(int)
            
            # Subsample if too large
            if adata.n_vars > max_proteins:
                # Select proteins with most variable missing patterns
                missing_var = np.var(missing_matrix, axis=0)
                top_indices = np.argsort(missing_var)[-max_proteins:]
                missing_matrix = missing_matrix[:, top_indices]
                protein_names = adata.var_names[top_indices]
            else:
                protein_names = adata.var_names
            
            if adata.n_obs > max_samples:
                # Random sampling of samples
                sample_indices = np.random.choice(adata.n_obs, max_samples, replace=False)
                missing_matrix = missing_matrix[sample_indices, :]
                sample_names = adata.obs_names[sample_indices]
            else:
                sample_names = adata.obs_names
            
            # Clustering if requested
            if cluster_samples and missing_matrix.shape[0] > 1:
                sample_dist = pdist(missing_matrix, metric='hamming')
                if len(sample_dist) > 0 and not np.all(sample_dist == 0):
                    sample_linkage = linkage(sample_dist, method='ward')
                    sample_order = dendrogram(sample_linkage, no_plot=True)['leaves']
                    missing_matrix = missing_matrix[sample_order, :]
                    sample_names = sample_names[sample_order]
            
            if cluster_proteins and missing_matrix.shape[1] > 1:
                protein_dist = pdist(missing_matrix.T, metric='hamming')
                if len(protein_dist) > 0 and not np.all(protein_dist == 0):
                    protein_linkage = linkage(protein_dist, method='ward')
                    protein_order = dendrogram(protein_linkage, no_plot=True)['leaves']
                    missing_matrix = missing_matrix[:, protein_order]
                    protein_names = protein_names[protein_order]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=missing_matrix,
                x=protein_names,
                y=sample_names,
                colorscale=[[0, 'lightblue'], [1, 'darkred']],
                colorbar=dict(
                    title="Missing Values",
                    tickvals=[0, 1],
                    ticktext=['Present', 'Missing']
                ),
                hovertemplate="Sample: %{y}<br>Protein: %{x}<br>Status: %{customdata}<extra></extra>",
                customdata=[['Present' if val == 0 else 'Missing' for val in row] for row in missing_matrix]
            ))
            
            # Calculate statistics
            total_missing = np.sum(missing_matrix)
            total_values = missing_matrix.size
            missing_percentage = (total_missing / total_values) * 100
            
            fig.update_layout(
                title=title or f"Missing Value Pattern ({missing_percentage:.1f}% missing)",
                xaxis_title="Proteins",
                yaxis_title="Samples",
                width=max(self.default_width, min(2000, 20 * len(protein_names))),
                height=max(self.default_height, min(1500, 15 * len(sample_names))),
                xaxis=dict(tickangle=45, tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8))
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating missing value heatmap: {e}")
            raise ProteomicsVisualizationError(f"Failed to create missing value heatmap: {str(e)}")

    def create_intensity_distribution_plot(
        self,
        adata: anndata.AnnData,
        group_by: Optional[str] = None,
        log_transform: bool = True,
        show_outliers: bool = True,
        max_samples: int = 50,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create intensity distribution plots across samples.
        
        Args:
            adata: AnnData object with proteomics data
            group_by: Column in obs to group samples by
            log_transform: Whether to log-transform intensities
            show_outliers: Whether to show outlier points
            max_samples: Maximum number of samples to show individually
            title: Plot title
            
        Returns:
            go.Figure: Interactive intensity distribution plot
        """
        try:
            logger.info("Creating intensity distribution plot")
            
            X = adata.X.copy()
            if issparse(X):
                X = X.toarray()
            
            # Remove missing values for distribution analysis
            X_clean = X[~np.isnan(X)]
            
            # Log transform if requested
            if log_transform:
                X_clean = np.log10(X_clean + 1)  # Add pseudocount
                y_title = "Log10(Intensity + 1)"
            else:
                y_title = "Intensity"
            
            # Create figure
            fig = go.Figure()
            
            if group_by and group_by in adata.obs.columns:
                # Group-wise distributions
                groups = adata.obs[group_by].unique()
                colors = px.colors.qualitative.Set1[:len(groups)]
                
                for i, group in enumerate(groups):
                    group_mask = adata.obs[group_by] == group
                    group_data = X[group_mask, :]
                    if log_transform:
                        group_data = np.log10(group_data + 1)
                    
                    group_clean = group_data[~np.isnan(group_data)]
                    
                    fig.add_trace(go.Violin(
                        y=group_clean,
                        name=str(group),
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=colors[i % len(colors)],
                        opacity=0.6,
                        points='outliers' if show_outliers else False
                    ))
                
                fig.update_layout(
                    xaxis_title=group_by,
                    yaxis_title=y_title
                )
                
            else:
                # Overall distribution
                fig.add_trace(go.Histogram(
                    x=X_clean,
                    nbinsx=100,
                    opacity=0.7,
                    marker_color='steelblue',
                    name='Intensity Distribution'
                ))
                
                # Add statistics overlay
                mean_val = np.mean(X_clean)
                median_val = np.median(X_clean)
                
                fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                             annotation_text=f"Mean: {mean_val:.2f}")
                fig.add_vline(x=median_val, line_dash="dash", line_color="blue",
                             annotation_text=f"Median: {median_val:.2f}")
                
                fig.update_layout(
                    xaxis_title=y_title,
                    yaxis_title="Frequency"
                )
            
            # Calculate statistics
            stats_dict = {
                "mean": float(np.mean(X_clean)),
                "median": float(np.median(X_clean)),
                "std": float(np.std(X_clean)),
                "min": float(np.min(X_clean)),
                "max": float(np.max(X_clean)),
                "n_values": len(X_clean),
                "missing_percentage": float(np.sum(np.isnan(X)) / X.size * 100)
            }
            
            fig.update_layout(
                title=title or f"Protein Intensity Distribution (n={stats_dict['n_values']:,})",
                width=self.default_width,
                height=self.default_height,
                showlegend=True,
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating intensity distribution plot: {e}")
            raise ProteomicsVisualizationError(f"Failed to create intensity distribution plot: {str(e)}")

    def create_cv_analysis_plot(
        self,
        adata: anndata.AnnData,
        group_by: Optional[str] = None,
        cv_threshold: float = 30.0,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create coefficient of variation (CV) analysis plot.
        
        Args:
            adata: AnnData object with proteomics data
            group_by: Column in obs to calculate group-wise CVs
            cv_threshold: CV threshold for highlighting (%)
            title: Plot title
            
        Returns:
            go.Figure: Interactive CV analysis plot
        """
        try:
            logger.info("Creating CV analysis plot")
            
            X = adata.X.copy()
            if issparse(X):
                X = X.toarray()
            
            if group_by and group_by in adata.obs.columns:
                # Group-wise CV calculation
                groups = adata.obs[group_by].unique()
                cv_data = []
                
                for group in groups:
                    group_mask = adata.obs[group_by] == group
                    group_data = X[group_mask, :]
                    
                    # Calculate CV for each protein in this group
                    group_means = np.nanmean(group_data, axis=0)
                    group_stds = np.nanstd(group_data, axis=0)
                    group_cvs = (group_stds / group_means) * 100
                    
                    # Filter out invalid CVs
                    valid_cvs = group_cvs[~np.isnan(group_cvs) & ~np.isinf(group_cvs)]
                    
                    cv_data.extend([{
                        'group': group,
                        'protein': protein,
                        'cv': cv,
                        'is_high_cv': cv > cv_threshold
                    } for protein, cv in zip(adata.var_names, group_cvs) if not (np.isnan(cv) or np.isinf(cv))])
                
                cv_df = pd.DataFrame(cv_data)
                
                # Create violin plot
                fig = px.violin(
                    cv_df, 
                    x='group', 
                    y='cv',
                    box=True,
                    title=title or f"Coefficient of Variation by {group_by}",
                    labels={'cv': 'CV (%)', 'group': group_by}
                )
                
                # Add threshold line
                fig.add_hline(y=cv_threshold, line_dash="dash", line_color="red",
                             annotation_text=f"CV threshold: {cv_threshold}%")
            
            else:
                # Overall CV distribution
                means = np.nanmean(X, axis=0)
                stds = np.nanstd(X, axis=0)
                cvs = (stds / means) * 100
                
                # Filter out invalid CVs
                valid_cvs = cvs[~np.isnan(cvs) & ~np.isinf(cvs)]
                high_cv_proteins = np.sum(cvs > cv_threshold)
                
                fig = go.Figure()
                
                # Histogram of CVs
                fig.add_trace(go.Histogram(
                    x=valid_cvs,
                    nbinsx=50,
                    opacity=0.7,
                    marker_color='lightblue',
                    name='CV Distribution'
                ))
                
                # Add threshold line
                fig.add_vline(x=cv_threshold, line_dash="dash", line_color="red",
                             annotation_text=f"Threshold: {cv_threshold}%")
                
                # Add statistics
                median_cv = np.median(valid_cvs)
                fig.add_vline(x=median_cv, line_dash="dot", line_color="blue",
                             annotation_text=f"Median: {median_cv:.1f}%")
                
                fig.update_layout(
                    xaxis_title='CV (%)',
                    yaxis_title='Number of Proteins',
                    title=title or f"Protein CV Distribution ({high_cv_proteins} proteins >{cv_threshold}% CV)"
                )
            
            fig.update_layout(
                width=self.default_width,
                height=self.default_height,
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating CV analysis plot: {e}")
            raise ProteomicsVisualizationError(f"Failed to create CV analysis plot: {str(e)}")

    def create_volcano_plot(
        self,
        adata: anndata.AnnData,
        comparison_results: Optional[Dict[str, Any]] = None,
        fold_change_col: str = "log2_fold_change",
        pvalue_col: str = "p_adjusted",
        protein_names_col: Optional[str] = None,
        fc_threshold: float = 1.0,
        pvalue_threshold: float = 0.05,
        highlight_proteins: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create a volcano plot for differential protein expression.
        
        Args:
            adata: AnnData object with proteomics data
            comparison_results: Dictionary with comparison results (if None, uses adata.uns)
            fold_change_col: Column name for fold changes
            pvalue_col: Column name for p-values
            protein_names_col: Column name for protein names (uses index if None)
            fc_threshold: Fold change threshold for significance
            pvalue_threshold: P-value threshold for significance
            highlight_proteins: List of proteins to highlight
            title: Plot title
            
        Returns:
            go.Figure: Interactive volcano plot
        """
        try:
            logger.info("Creating volcano plot")
            
            # Get comparison results
            if comparison_results is None:
                if 'differential_expression' in adata.uns:
                    results_data = adata.uns['differential_expression']
                elif 'statistical_tests' in adata.uns:
                    results_data = adata.uns['statistical_tests']['results']
                else:
                    raise ProteomicsVisualizationError("No differential expression results found")
            else:
                results_data = comparison_results
            
            # Convert to DataFrame if needed
            if isinstance(results_data, dict):
                if 'results' in results_data:
                    df = pd.DataFrame(results_data['results'])
                else:
                    df = pd.DataFrame([results_data])
            elif isinstance(results_data, list):
                df = pd.DataFrame(results_data)
            else:
                df = results_data.copy()
            
            # Map column names
            if fold_change_col not in df.columns:
                # Try to find fold change column
                fc_candidates = ['fold_change', 'log2fc', 'logfc', 'effect_size']
                for candidate in fc_candidates:
                    if candidate in df.columns:
                        fold_change_col = candidate
                        break
                else:
                    raise ProteomicsVisualizationError(f"Fold change column not found. Available: {list(df.columns)}")
            
            if pvalue_col not in df.columns:
                # Try to find p-value column
                pval_candidates = ['p_value', 'pval', 'p_adj', 'padj']
                for candidate in pval_candidates:
                    if candidate in df.columns:
                        pvalue_col = candidate
                        break
                else:
                    raise ProteomicsVisualizationError(f"P-value column not found. Available: {list(df.columns)}")
            
            # Get protein names
            if protein_names_col and protein_names_col in df.columns:
                protein_names = df[protein_names_col]
            elif 'protein' in df.columns:
                protein_names = df['protein']
            elif 'protein_id' in df.columns:
                protein_names = df['protein_id']
            else:
                protein_names = df.index
            
            # Calculate -log10(p-value)
            neg_log_pval = -np.log10(df[pvalue_col] + 1e-300)  # Add small value to avoid log(0)
            
            # Determine significance categories
            significant_up = (df[fold_change_col] > fc_threshold) & (df[pvalue_col] < pvalue_threshold)
            significant_down = (df[fold_change_col] < -fc_threshold) & (df[pvalue_col] < pvalue_threshold)
            not_significant = ~(significant_up | significant_down)
            
            # Create colors
            colors = []
            for i in range(len(df)):
                if significant_up.iloc[i]:
                    colors.append('red')
                elif significant_down.iloc[i]:
                    colors.append('blue')
                else:
                    colors.append('gray')
            
            # Create volcano plot
            fig = go.Figure()
            
            # Add points
            fig.add_trace(go.Scatter(
                x=df[fold_change_col],
                y=neg_log_pval,
                mode='markers',
                marker=dict(
                    color=colors,
                    size=6,
                    opacity=0.7,
                    line=dict(width=0.5, color='black')
                ),
                text=protein_names,
                hovertemplate=(
                    "Protein: %{text}<br>"
                    f"{fold_change_col}: %{{x:.2f}}<br>"
                    f"-log10({pvalue_col}): %{{y:.2f}}<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ))
            
            # Highlight specific proteins if requested
            if highlight_proteins:
                highlight_mask = protein_names.isin(highlight_proteins)
                if highlight_mask.any():
                    fig.add_trace(go.Scatter(
                        x=df.loc[highlight_mask, fold_change_col],
                        y=neg_log_pval[highlight_mask],
                        mode='markers+text',
                        marker=dict(
                            color='yellow',
                            size=10,
                            line=dict(width=2, color='black')
                        ),
                        text=protein_names[highlight_mask],
                        textposition='top center',
                        name='Highlighted Proteins',
                        showlegend=True
                    ))
            
            # Add threshold lines
            fig.add_hline(y=-np.log10(pvalue_threshold), line_dash="dash", line_color="red",
                         annotation_text=f"p = {pvalue_threshold}")
            fig.add_vline(x=fc_threshold, line_dash="dash", line_color="red")
            fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="red")
            
            # Count significant proteins
            n_up = significant_up.sum()
            n_down = significant_down.sum()
            n_total = len(df)
            
            fig.update_layout(
                title=title or f"Volcano Plot ({n_up} up, {n_down} down, {n_total} total)",
                xaxis_title=f"{fold_change_col}",
                yaxis_title=f"-log10({pvalue_col})",
                width=self.default_width,
                height=self.default_height,
                plot_bgcolor='white',
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating volcano plot: {e}")
            raise ProteomicsVisualizationError(f"Failed to create volcano plot: {str(e)}")

    def create_protein_correlation_network(
        self,
        adata: anndata.AnnData,
        correlation_threshold: float = 0.7,
        max_proteins: int = 100,
        layout_algorithm: str = "spring",
        color_by: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create a protein correlation network visualization.
        
        Args:
            adata: AnnData object with proteomics data
            correlation_threshold: Minimum correlation for edge creation
            max_proteins: Maximum number of proteins to include
            layout_algorithm: Network layout algorithm ('spring', 'circular', 'random')
            color_by: Protein metadata column for node coloring
            title: Plot title
            
        Returns:
            go.Figure: Interactive network plot
        """
        try:
            logger.info("Creating protein correlation network")
            
            X = adata.X.copy()
            if issparse(X):
                X = X.toarray()
            
            # Handle missing values by protein-wise mean imputation
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            
            # Subsample proteins if too many
            if adata.n_vars > max_proteins:
                # Select most variable proteins
                var_proteins = np.var(X_imputed, axis=0)
                top_indices = np.argsort(var_proteins)[-max_proteins:]
                X_subset = X_imputed[:, top_indices]
                protein_names = adata.var_names[top_indices]
                if color_by and color_by in adata.var.columns:
                    color_values = adata.var[color_by].iloc[top_indices]
                else:
                    color_values = None
            else:
                X_subset = X_imputed
                protein_names = adata.var_names
                if color_by and color_by in adata.var.columns:
                    color_values = adata.var[color_by]
                else:
                    color_values = None
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(X_subset.T)
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes
            for i, protein in enumerate(protein_names):
                node_attrs = {'name': protein}
                if color_values is not None:
                    node_attrs['color_value'] = color_values.iloc[i] if hasattr(color_values, 'iloc') else color_values[i]
                G.add_node(i, **node_attrs)
            
            # Add edges based on correlation threshold
            for i in range(len(protein_names)):
                for j in range(i + 1, len(protein_names)):
                    corr = corr_matrix[i, j]
                    if abs(corr) >= correlation_threshold:
                        G.add_edge(i, j, weight=abs(corr), correlation=corr)
            
            logger.info(f"Network created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Calculate layout
            if layout_algorithm == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout_algorithm == "circular":
                pos = nx.circular_layout(G)
            elif layout_algorithm == "random":
                pos = nx.random_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # Prepare edge traces
            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                corr = edge[2]['correlation']
                edge_info.append(f"Correlation: {corr:.3f}")
            
            # Prepare node traces
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            
            for node in G.nodes(data=True):
                x, y = pos[node[0]]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node[1]['name'])
                
                if color_values is not None and 'color_value' in node[1]:
                    node_colors.append(node[1]['color_value'])
                else:
                    node_colors.append('steelblue')
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='lightgray'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            ))
            
            # Add nodes
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition='middle center',
                hovertemplate="Protein: %{text}<br>Connections: %{customdata}<extra></extra>",
                customdata=[G.degree(node) for node in G.nodes()],
                marker=dict(
                    size=10,
                    color=node_colors,
                    colorscale='Viridis' if color_values is not None else None,
                    showscale=color_values is not None,
                    colorbar=dict(title=color_by) if color_by else None,
                    line=dict(width=1, color='black')
                ),
                showlegend=False
            )
            
            fig.add_trace(node_trace)
            
            # Update layout
            fig.update_layout(
                title=title or f"Protein Correlation Network (r ≥ {correlation_threshold})",
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[dict(
                    text=f"Network: {G.number_of_nodes()} proteins, {G.number_of_edges()} correlations",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=self.default_width,
                height=self.default_height,
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating protein correlation network: {e}")
            raise ProteomicsVisualizationError(f"Failed to create protein correlation network: {str(e)}")

    def create_pathway_enrichment_plot(
        self,
        adata: anndata.AnnData,
        enrichment_results: Optional[Dict[str, Any]] = None,
        top_n: int = 20,
        plot_type: str = "bubble",
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create pathway enrichment visualization.
        
        Args:
            adata: AnnData object with proteomics data
            enrichment_results: Dictionary with enrichment results (if None, uses adata.uns)
            top_n: Number of top pathways to show
            plot_type: Type of plot ('bubble', 'bar', 'dot')
            title: Plot title
            
        Returns:
            go.Figure: Interactive pathway enrichment plot
        """
        try:
            logger.info("Creating pathway enrichment plot")
            
            # Get enrichment results
            if enrichment_results is None:
                if 'pathway_enrichment' in adata.uns:
                    results_data = adata.uns['pathway_enrichment']['results']
                else:
                    raise ProteomicsVisualizationError("No pathway enrichment results found")
            else:
                results_data = enrichment_results
            
            # Convert to DataFrame
            if isinstance(results_data, list):
                df = pd.DataFrame(results_data)
            else:
                df = results_data.copy()
            
            # Sort by p-value and take top pathways
            df = df.sort_values('p_value').head(top_n)
            
            # Calculate -log10(p-value)
            df['neg_log_pval'] = -np.log10(df['p_value'] + 1e-300)
            
            fig = go.Figure()
            
            if plot_type == "bubble":
                # Bubble plot
                fig.add_trace(go.Scatter(
                    x=df['enrichment_ratio'],
                    y=df['pathway_name'],
                    mode='markers',
                    marker=dict(
                        size=df['neg_log_pval'] * 3,
                        color=df['neg_log_pval'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="-log10(p-value)"),
                        sizemode='diameter',
                        sizeref=0.1,
                        line=dict(width=1, color='black')
                    ),
                    text=df['overlap_count'],
                    textposition='middle center',
                    hovertemplate=(
                        "Pathway: %{y}<br>"
                        "Enrichment Ratio: %{x:.2f}<br>"
                        "P-value: %{customdata:.2e}<br>"
                        "Overlapping Proteins: %{text}<br>"
                        "<extra></extra>"
                    ),
                    customdata=df['p_value']
                ))
                
                fig.update_layout(
                    xaxis_title="Enrichment Ratio",
                    yaxis_title="Pathway"
                )
                
            elif plot_type == "bar":
                # Horizontal bar plot
                fig.add_trace(go.Bar(
                    x=df['neg_log_pval'],
                    y=df['pathway_name'],
                    orientation='h',
                    marker=dict(
                        color=df['enrichment_ratio'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Enrichment Ratio")
                    ),
                    hovertemplate=(
                        "Pathway: %{y}<br>"
                        "-log10(p-value): %{x:.2f}<br>"
                        "Enrichment Ratio: %{customdata:.2f}<br>"
                        "<extra></extra>"
                    ),
                    customdata=df['enrichment_ratio']
                ))
                
                fig.update_layout(
                    xaxis_title="-log10(p-value)",
                    yaxis_title="Pathway"
                )
            
            fig.update_layout(
                title=title or f"Top {len(df)} Enriched Pathways",
                width=self.default_width,
                height=max(400, 30 * len(df)),
                plot_bgcolor='white',
                yaxis=dict(tickfont=dict(size=10))
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pathway enrichment plot: {e}")
            raise ProteomicsVisualizationError(f"Failed to create pathway enrichment plot: {str(e)}")

    def create_proteomics_qc_dashboard(
        self,
        adata: anndata.AnnData,
        platform_type: str = "mass_spectrometry",
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive quality control dashboard for proteomics data.
        
        Args:
            adata: AnnData object with proteomics data
            platform_type: Platform type ('mass_spectrometry' or 'affinity')
            title: Plot title
            
        Returns:
            go.Figure: Comprehensive QC dashboard
        """
        try:
            logger.info(f"Creating proteomics QC dashboard for {platform_type}")
            
            X = adata.X.copy()
            if issparse(X):
                X = X.toarray()
            
            # Create subplot structure
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=[
                    "A. Missing Value Pattern",
                    "B. Intensity Distribution", 
                    "C. CV Distribution",
                    "D. Sample Correlation",
                    "E. Protein Detection",
                    "F. Batch Effects" if 'batch' in adata.obs.columns else "F. Sample Stats",
                    "G. Data Quality Metrics",
                    "H. Platform-Specific QC",
                    "I. Summary Statistics"
                ],
                specs=[
                    [{"type": "heatmap"}, {"type": "histogram"}, {"type": "histogram"}],
                    [{"type": "heatmap"}, {"type": "bar"}, {"type": "violin"}],
                    [{"type": "bar"}, {"type": "scatter"}, {"type": "table"}]
                ],
                horizontal_spacing=0.1,
                vertical_spacing=0.12
            )
            
            # A. Missing value pattern (sample)
            missing_per_sample = np.isnan(X).sum(axis=1) / X.shape[1] * 100
            missing_per_protein = np.isnan(X).sum(axis=0) / X.shape[0] * 100
            
            fig.add_trace(go.Histogram(
                x=missing_per_sample,
                nbinsx=30,
                name="Missing % per Sample",
                marker_color='red',
                opacity=0.7
            ), row=1, col=1)
            
            # B. Intensity distribution
            X_clean = X[~np.isnan(X)]
            log_intensities = np.log10(X_clean + 1)
            
            fig.add_trace(go.Histogram(
                x=log_intensities,
                nbinsx=50,
                name="Log10 Intensities",
                marker_color='blue',
                opacity=0.7
            ), row=1, col=2)
            
            # C. CV distribution
            means = np.nanmean(X, axis=0)
            stds = np.nanstd(X, axis=0)
            cvs = (stds / means) * 100
            valid_cvs = cvs[~np.isnan(cvs) & ~np.isinf(cvs)]
            
            fig.add_trace(go.Histogram(
                x=valid_cvs,
                nbinsx=30,
                name="CV Distribution",
                marker_color='green',
                opacity=0.7
            ), row=1, col=3)
            
            # D. Sample correlation heatmap
            sample_corr = np.corrcoef(X)
            n_samples_show = min(50, adata.n_obs)
            
            fig.add_trace(go.Heatmap(
                z=sample_corr[:n_samples_show, :n_samples_show],
                colorscale='RdBu_r',
                zmid=0,
                showscale=False
            ), row=2, col=1)
            
            # E. Protein detection rates
            detection_rates = (1 - missing_per_protein / 100) * 100
            
            fig.add_trace(go.Bar(
                x=list(range(min(100, adata.n_vars))),
                y=detection_rates[:min(100, adata.n_vars)],
                name="Detection Rate %",
                marker_color='orange'
            ), row=2, col=2)
            
            # F. Batch effects or sample stats
            if 'batch' in adata.obs.columns:
                batch_data = []
                for batch in adata.obs['batch'].unique():
                    batch_mask = adata.obs['batch'] == batch
                    batch_intensities = X[batch_mask, :].flatten()
                    batch_clean = batch_intensities[~np.isnan(batch_intensities)]
                    batch_data.extend([{
                        'batch': batch,
                        'intensity': np.log10(val + 1)
                    } for val in batch_clean[:1000]])  # Sample for performance
                
                batch_df = pd.DataFrame(batch_data)
                for i, batch in enumerate(adata.obs['batch'].unique()):
                    batch_subset = batch_df[batch_df['batch'] == batch]['intensity']
                    fig.add_trace(go.Violin(
                        y=batch_subset,
                        name=f"Batch {batch}",
                        box_visible=True,
                        meanline_visible=True
                    ), row=2, col=3)
            else:
                # Sample statistics
                sample_stats = pd.DataFrame({
                    'n_proteins': np.sum(~np.isnan(X), axis=1),
                    'total_intensity': np.nansum(X, axis=1),
                    'median_intensity': np.nanmedian(X, axis=1)
                })
                
                fig.add_trace(go.Violin(
                    y=sample_stats['n_proteins'],
                    name="Proteins per Sample",
                    box_visible=True
                ), row=2, col=3)
            
            # G. Data quality metrics
            quality_metrics = {
                'High Quality Proteins (CV<30%)': np.sum(valid_cvs < 30),
                'Low Missing Proteins (<50%)': np.sum(missing_per_protein < 50),
                'Complete Proteins (0% missing)': np.sum(missing_per_protein == 0),
                'High Missing Proteins (>80%)': np.sum(missing_per_protein > 80)
            }
            
            fig.add_trace(go.Bar(
                x=list(quality_metrics.keys()),
                y=list(quality_metrics.values()),
                name="Quality Metrics",
                marker_color='purple'
            ), row=3, col=1)
            
            # H. Platform-specific QC
            if platform_type == "mass_spectrometry":
                # MS-specific metrics
                if 'n_peptides' in adata.var.columns:
                    peptide_counts = adata.var['n_peptides'].fillna(0)
                    fig.add_trace(go.Scatter(
                        x=peptide_counts,
                        y=detection_rates,
                        mode='markers',
                        name="Peptides vs Detection",
                        marker_color='red'
                    ), row=3, col=2)
                else:
                    # Fallback: intensity vs detection
                    mean_intensities = np.nanmean(X, axis=0)
                    fig.add_trace(go.Scatter(
                        x=np.log10(mean_intensities + 1),
                        y=detection_rates,
                        mode='markers',
                        name="Intensity vs Detection",
                        marker_color='red'
                    ), row=3, col=2)
            else:
                # Affinity-specific metrics
                fig.add_trace(go.Scatter(
                    x=valid_cvs[:min(len(valid_cvs), len(detection_rates))],
                    y=detection_rates[:min(len(valid_cvs), len(detection_rates))],
                    mode='markers',
                    name="CV vs Detection",
                    marker_color='blue'
                ), row=3, col=2)
            
            # I. Summary statistics table
            summary_stats = pd.DataFrame({
                'Metric': [
                    'Total Samples',
                    'Total Proteins', 
                    'Mean Proteins/Sample',
                    'Overall Missing %',
                    'Median CV %',
                    'High Quality Proteins'
                ],
                'Value': [
                    f"{adata.n_obs:,}",
                    f"{adata.n_vars:,}",
                    f"{np.mean(np.sum(~np.isnan(X), axis=1)):.0f}",
                    f"{np.mean(missing_per_sample):.1f}%",
                    f"{np.median(valid_cvs):.1f}%",
                    f"{np.sum(valid_cvs < 30):,}"
                ]
            })
            
            fig.add_trace(go.Table(
                cells=dict(
                    values=[summary_stats['Metric'], summary_stats['Value']],
                    align='left',
                    font=dict(size=11)
                )
            ), row=3, col=3)
            
            # Update layout
            fig.update_layout(
                title=title or f"Proteomics QC Dashboard - {platform_type.replace('_', ' ').title()}",
                width=1400,
                height=1000,
                showlegend=False,
                plot_bgcolor='white'
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Missing % per Sample", row=1, col=1)
            fig.update_xaxes(title_text="Log10(Intensity + 1)", row=1, col=2)
            fig.update_xaxes(title_text="CV (%)", row=1, col=3)
            fig.update_xaxes(title_text="Protein Index", row=2, col=2)
            fig.update_yaxes(title_text="Detection Rate (%)", row=2, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating proteomics QC dashboard: {e}")
            raise ProteomicsVisualizationError(f"Failed to create proteomics QC dashboard: {str(e)}")

    def save_plots(
        self,
        plots: Dict[str, go.Figure],
        output_dir: str,
        format: str = "both"
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
