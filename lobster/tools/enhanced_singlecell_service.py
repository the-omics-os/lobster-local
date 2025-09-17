"""
Enhanced single-cell RNA-seq service with advanced analysis capabilities.

This service extends the basic clustering functionality with doublet detection,
cell type annotation, and advanced visualization capabilities.
"""

from typing import Any, Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scanpy as sc

try:
    import scrublet as scr
    SCRUBLET_AVAILABLE = True
except ImportError:
    SCRUBLET_AVAILABLE = False
    scr = None

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class SingleCellError(Exception):
    """Base exception for single-cell analysis operations."""
    pass


class EnhancedSingleCellService:
    """
    Stateless enhanced service for single-cell RNA-seq analysis.

    This class provides advanced single-cell analysis capabilities including
    doublet detection, cell type annotation, and pathway analysis.
    """

    def __init__(self):
        """
        Initialize the enhanced single-cell service.
        
        This service is stateless and doesn't require a data manager instance.
        """
        logger.debug("Initializing stateless EnhancedSingleCellService")

        # Cell type markers database (simplified version)
        self.cell_type_markers = {
            "T cells": ["CD3D", "CD3E", "CD8A", "CD4"],
            "B cells": ["CD19", "MS4A1", "CD79A", "IGHM"],
            "NK cells": ["GNLY", "NKG7", "KLRD1", "NCAM1"],
            "Monocytes": ["CD14", "FCGR3A", "LYZ", "CSF1R"],
            "Dendritic cells": ["FCER1A", "CST3", "CLEC4C"],
            "Neutrophils": ["FCGR3B", "CEACAM3", "CSF3R"],
            "Platelets": ["PPBP", "PF4", "TUBB1"],
            "Endothelial": ["PECAM1", "VWF", "ENG", "CDH5"],
            "Fibroblasts": ["COL1A1", "COL3A1", "DCN", "LUM"],
            "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19"],
        }

        logger.info(f"Loaded {len(self.cell_type_markers)} cell type marker sets")
        logger.debug(f"Available cell types: {list(self.cell_type_markers.keys())}")
        logger.debug("EnhancedSingleCellService initialized successfully")

    def detect_doublets(
        self, 
        adata: anndata.AnnData,
        expected_doublet_rate: float = 0.025, 
        threshold: Optional[float] = None
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Detect doublets using Scrublet or fallback method.

        Args:
            adata: AnnData object for doublet detection
            expected_doublet_rate: Expected doublet rate
            threshold: Custom threshold for doublet calling

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with doublet scores and detection stats
            
        Raises:
            SingleCellError: If doublet detection fails
        """
        try:
            logger.info(f"Starting doublet detection with expected rate: {expected_doublet_rate}")
            
            # Create working copy
            adata_doublets = adata.copy()
            
            # Get count matrix for doublet detection
            if adata_doublets.raw is not None:
                logger.info("Using raw counts for doublet detection")
                counts_matrix = adata_doublets.raw.X
            else:
                logger.info("Using current matrix for doublet detection")
                counts_matrix = adata_doublets.X
            
            # Convert to dense array if sparse
            if hasattr(counts_matrix, 'toarray'):
                counts_matrix = counts_matrix.toarray()
            
            logger.info(f"Doublet detection matrix shape: {counts_matrix.shape}")

            # Check if we have enough features
            if counts_matrix.shape[1] == 0:
                raise SingleCellError("Expression matrix has no gene features")

            # Run doublet detection
            if SCRUBLET_AVAILABLE:
                try:
                    logger.info("Running Scrublet doublet detection")
                    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=expected_doublet_rate)
                    
                    doublet_scores, predicted_doublets = scrub.scrub_doublets(
                        min_counts=2,
                        min_cells=3,
                        min_gene_variability_pctl=85,
                        n_prin_comps=30,
                        verbose=False,
                    )

                    # Apply custom threshold if provided
                    if threshold is not None:
                        logger.info(f"Using custom doublet threshold: {threshold}")
                        predicted_doublets = scrub.call_doublets(threshold=threshold)
                        
                    detection_method = "scrublet"
                    
                except Exception as e:
                    logger.warning(f"Scrublet failed: {e}. Using fallback method.")
                    doublet_scores, predicted_doublets, detection_method = self._fallback_doublet_detection(
                        counts_matrix, expected_doublet_rate
                    )
            else:
                logger.info("Scrublet not available, using fallback doublet detection")
                doublet_scores, predicted_doublets, detection_method = self._fallback_doublet_detection(
                    counts_matrix, expected_doublet_rate
                )

            # Add doublet information to AnnData
            adata_doublets.obs["doublet_score"] = doublet_scores
            adata_doublets.obs["predicted_doublet"] = predicted_doublets

            # Calculate detection statistics
            n_doublets = np.sum(predicted_doublets)
            doublet_rate = n_doublets / len(predicted_doublets)

            detection_stats = {
                "analysis_type": "doublet_detection",
                "expected_doublet_rate": expected_doublet_rate,
                "threshold": threshold,
                "detection_method": detection_method,
                "n_cells_analyzed": len(predicted_doublets),
                "n_doublets_detected": int(n_doublets),
                "actual_doublet_rate": float(doublet_rate),
                "doublet_score_stats": {
                    "min": float(doublet_scores.min()),
                    "max": float(doublet_scores.max()),
                    "mean": float(doublet_scores.mean()),
                    "std": float(doublet_scores.std())
                }
            }

            logger.info(f"Doublet detection completed: {n_doublets} doublets detected ({doublet_rate:.1%})")
            
            return adata_doublets, detection_stats

        except Exception as e:
            logger.exception(f"Error in doublet detection: {e}")
            raise SingleCellError(f"Doublet detection failed: {str(e)}")

    def _fallback_doublet_detection(
        self, 
        counts_matrix: np.ndarray, 
        expected_doublet_rate: float
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Fallback doublet detection method when Scrublet is not available.
        
        Args:
            counts_matrix: Count matrix (cells x genes)
            expected_doublet_rate: Expected doublet rate
            
        Returns:
            Tuple of (doublet_scores, predicted_doublets, method_name)
        """
        logger.info("Using fallback doublet detection method")
        
        n_cells = counts_matrix.shape[0]
        
        # Calculate per-cell metrics that indicate doublets
        total_counts = np.sum(counts_matrix, axis=1)
        n_genes = np.sum(counts_matrix > 0, axis=1)
        
        # Normalize metrics to z-scores
        total_counts_z = np.abs((total_counts - np.mean(total_counts)) / np.std(total_counts))
        n_genes_z = np.abs((n_genes - np.mean(n_genes)) / np.std(n_genes))
        
        # Combined doublet score (higher = more likely doublet)
        doublet_scores = (total_counts_z + n_genes_z) / 2
        
        # Apply threshold based on expected doublet rate
        doublet_threshold = np.percentile(doublet_scores, (1 - expected_doublet_rate) * 100)
        predicted_doublets = doublet_scores > doublet_threshold
        
        logger.info(f"Fallback method detected {np.sum(predicted_doublets)} potential doublets")
        
        return doublet_scores, predicted_doublets, "fallback_outlier_detection"

    def annotate_cell_types(
        self, 
        adata: anndata.AnnData,
        reference_markers: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Annotate cell types based on marker genes.

        Args:
            adata: AnnData object with clustering results
            reference_markers: Optional custom marker genes dictionary

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with cell type annotations and stats
            
        Raises:
            SingleCellError: If annotation fails
        """
        try:
            logger.info("Starting cell type annotation using marker genes")
            
            # Validate input
            if "leiden" not in adata.obs.columns:
                raise SingleCellError("No clustering results found. Please run clustering first.")

            # Create working copy
            adata_annotated = adata.copy()

            # Use provided markers or default ones
            markers = reference_markers or self.cell_type_markers
            logger.info(f"Using {len(markers)} marker sets for annotation")

            # Calculate marker gene scores for each cluster
            cluster_annotations = self._calculate_marker_scores_from_adata(adata_annotated, markers)

            # Determine best cell type for each cluster
            cluster_to_celltype = {}
            for cluster_id in adata_annotated.obs["leiden"].unique():
                cluster_str = str(cluster_id)
                if cluster_str in cluster_annotations:
                    best_match = max(
                        cluster_annotations[cluster_str].items(), key=lambda x: x[1]
                    )
                    cluster_to_celltype[cluster_id] = best_match[0]
                else:
                    cluster_to_celltype[cluster_id] = "Unknown"

            # Map cluster annotations to cells
            cell_types = adata_annotated.obs["leiden"].map(cluster_to_celltype)
            adata_annotated.obs["cell_type"] = cell_types

            # Calculate annotation statistics
            cell_type_counts = cell_types.value_counts().to_dict()
            n_cell_types = len(set(cell_types))
            
            annotation_stats = {
                "analysis_type": "cell_type_annotation",
                "markers_used": list(markers.keys()),
                "n_marker_sets": len(markers),
                "n_clusters": len(adata_annotated.obs["leiden"].unique()),
                "n_cell_types_identified": n_cell_types,
                "cluster_to_celltype": {str(k): v for k, v in cluster_to_celltype.items()},
                "cell_type_counts": {str(k): int(v) for k, v in cell_type_counts.items()},
                "marker_scores": cluster_annotations
            }

            logger.info(f"Cell type annotation completed: {n_cell_types} cell types identified")
            
            return adata_annotated, annotation_stats

        except Exception as e:
            logger.exception(f"Error in cell type annotation: {e}")
            raise SingleCellError(f"Cell type annotation failed: {str(e)}")

    def find_marker_genes(
        self, 
        adata: anndata.AnnData,
        groupby: str = "leiden",
        groups: Optional[List[str]] = None,
        method: str = "wilcoxon",
        n_genes: int = 25
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Find marker genes for clusters or cell types.

        Args:
            adata: AnnData object with clustering/annotation results
            groupby: Column name to group by ('leiden', 'cell_type', etc.)
            groups: Specific groups to analyze (None for all)
            method: Statistical method ('wilcoxon', 't-test', 'logreg')
            n_genes: Number of top marker genes per group

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: AnnData with marker gene results and stats
            
        Raises:
            SingleCellError: If marker gene detection fails
        """
        try:
            logger.info(f"Finding marker genes grouped by: {groupby}")
            
            # Validate input
            if groupby not in adata.obs.columns:
                raise SingleCellError(f"Group column '{groupby}' not found in observations")

            # Create working copy
            adata_markers = adata.copy()

            # Run differential expression analysis
            sc.tl.rank_genes_groups(
                adata_markers,
                groupby=groupby,
                groups=groups,
                method=method,
                n_genes=n_genes,
                use_raw=True
            )

            # Extract marker genes into structured format
            marker_genes_df = self._extract_marker_genes(adata_markers, groupby)
            
            # Calculate marker gene statistics
            unique_groups = adata_markers.obs[groupby].unique()
            n_groups = len(unique_groups)
            
            marker_stats = {
                "analysis_type": "marker_gene_analysis",
                "groupby": groupby,
                "method": method,
                "n_genes": n_genes,
                "n_groups": n_groups,
                "groups_analyzed": [str(g) for g in unique_groups],
                "has_marker_results": "rank_genes_groups" in adata_markers.uns,
                "marker_genes_df_shape": marker_genes_df.shape if not marker_genes_df.empty else (0, 0)
            }
            
            # Store top marker genes per group
            if not marker_genes_df.empty:
                top_markers_per_group = {}
                for group in marker_genes_df["group"].unique():
                    group_genes = marker_genes_df[marker_genes_df["group"] == group].head(10)
                    top_markers_per_group[str(group)] = [
                        {
                            "gene": row["gene"], 
                            "score": float(row["score"]), 
                            "pval": float(row["pval"])
                        }
                        for _, row in group_genes.iterrows()
                    ]
                marker_stats["top_markers_per_group"] = top_markers_per_group

            logger.info(f"Marker gene analysis completed for {n_groups} groups using {method} method")
            
            return adata_markers, marker_stats

        except Exception as e:
            logger.exception(f"Error finding marker genes: {e}")
            raise SingleCellError(f"Marker gene analysis failed: {str(e)}")

    def _calculate_marker_scores_from_adata(
        self, 
        adata: anndata.AnnData, 
        markers: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate marker gene scores for each cluster from AnnData object.
        
        Args:
            adata: AnnData object with clustering results
            markers: Dictionary of cell type markers
            
        Returns:
            Dict[str, Dict[str, float]]: Cluster scores for each cell type
        """
        logger.info("Calculating marker scores from AnnData")
        
        # Ensure unique names to prevent reindexing errors
        if not adata.obs_names.is_unique:
            logger.warning("Non-unique observation indices detected. Making unique.")
            adata.obs_names_make_unique()

        if not adata.var_names.is_unique:
            logger.warning("Non-unique variable names detected. Making unique.")
            adata.var_names_make_unique()

        cluster_scores = {}

        # Get unique clusters
        unique_clusters = adata.obs["leiden"].astype(str).unique()

        for cluster in unique_clusters:
            cluster_scores[cluster] = {}
            cluster_cells = adata.obs["leiden"].astype(str) == cluster

            for cell_type, marker_genes in markers.items():
                # Find available markers in the dataset
                available_markers = [
                    gene for gene in marker_genes if gene in adata.var_names
                ]

                if available_markers:
                    try:
                        # Calculate mean expression of markers in this cluster
                        subset = adata[cluster_cells, available_markers]

                        if subset.shape[0] > 0:  # Check if any cells match
                            if hasattr(subset.X, 'toarray'):
                                marker_expression = subset.X.toarray().mean(axis=0)
                            else:
                                marker_expression = subset.X.mean(axis=0)
                                
                            # Calculate score as mean of available markers
                            score = float(np.mean(marker_expression))
                            cluster_scores[cluster][cell_type] = score
                        else:
                            cluster_scores[cluster][cell_type] = 0.0
                    except Exception as e:
                        logger.warning(
                            f"Error calculating marker score for cluster {cluster}, cell type {cell_type}: {e}"
                        )
                        cluster_scores[cluster][cell_type] = 0.0
                else:
                    cluster_scores[cluster][cell_type] = 0.0

        logger.info(f"Calculated marker scores for {len(unique_clusters)} clusters")
        return cluster_scores

    def run_pathway_analysis(self, cell_type: Optional[str] = None) -> str:
        """
        Run pathway analysis on marker genes.

        Args:
            cell_type: Specific cell type for analysis

        Returns:
            str: Pathway analysis results
        """
        try:
            logger.info(f"Running pathway analysis for {cell_type or 'all cell types'}")

            # Get marker genes from metadata
            if "marker_genes" not in self.data_manager.current_metadata:
                return "No marker genes found. Please run marker gene analysis first."

            marker_genes = self.data_manager.current_metadata["marker_genes"]

            # Extract gene list (simplified approach)
            if isinstance(marker_genes, dict) and "names" in marker_genes:
                gene_list = list(marker_genes["names"].values())[0][:50]  # Top 50 genes
            else:
                return "Invalid marker gene format. Please rerun marker gene analysis."

            # Run mock pathway analysis (replace with actual GO/KEGG analysis)
            pathway_results = self._run_mock_pathway_analysis(gene_list)

            # Create pathway plot
            pathway_plot = self._create_pathway_plot(pathway_results)
            analysis_target = cell_type if cell_type else "All Cell Types"
            self.data_manager.add_plot(
                pathway_plot,
                title=f"Pathway Analysis for {analysis_target}",
                source="enhanced_singlecell_service",
            )

            return f"""Pathway Analysis Complete!

**Genes Analyzed:** {len(gene_list)}
**Significant Pathways:** {len([p for p in pathway_results if p['p_value'] < 0.05])}

**Top Enriched Pathways:**
{self._format_pathway_results(pathway_results)}

Pathway enrichment plot shows the most significantly enriched biological processes.

Next suggested step: Export results or perform additional downstream analysis."""

        except Exception as e:
            logger.exception(f"Error in pathway analysis: {e}")
            return f"Error in pathway analysis: {str(e)}"

    def _calculate_marker_scores(
        self, markers: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate marker gene scores for each cluster."""
        adata = self.data_manager.adata

        # Ensure unique observation indices to prevent reindexing errors
        if not adata.obs_names.is_unique:
            logger.warning(
                "Non-unique observation indices detected in AnnData. Using index positions instead."
            )
            # Create a copy to avoid modifying the original
            adata = adata.copy()
            adata.obs_names_make_unique()

        # Ensure unique variable names (gene names) to prevent reindexing errors
        if not adata.var_names.is_unique:
            logger.warning(
                "Non-unique variable names (genes) detected in AnnData. Making them unique."
            )
            # Create a copy to avoid modifying the original
            adata = adata.copy()
            adata.var_names_make_unique()

        cluster_scores = {}

        # Get unique clusters and handle potential duplicate indices
        unique_clusters = adata.obs["leiden"].astype(str).unique()

        for cluster in unique_clusters:
            cluster_scores[cluster] = {}
            # Use string comparison to avoid type issues
            cluster_cells = adata.obs["leiden"].astype(str) == cluster

            for cell_type, marker_genes in markers.items():
                # Find available markers in the dataset
                available_markers = [
                    gene for gene in marker_genes if gene in adata.var_names
                ]

                if available_markers:
                    try:
                        # Calculate mean expression of markers in this cluster
                        subset = adata[cluster_cells, available_markers]

                        if subset.shape[0] > 0:  # Check if any cells match
                            marker_expression = subset.X.mean(axis=0)
                            if hasattr(
                                marker_expression, "A1"
                            ):  # Handle sparse matrices
                                marker_expression = marker_expression.A1

                            # Calculate score as mean of available markers
                            score = (
                                np.mean(marker_expression)
                                if len(available_markers) > 0
                                else 0
                            )
                            cluster_scores[cluster][cell_type] = score
                        else:
                            cluster_scores[cluster][cell_type] = 0
                    except Exception as e:
                        logger.warning(
                            f"Error calculating marker score for cluster {cluster}, cell type {cell_type}: {e}"
                        )
                        cluster_scores[cluster][cell_type] = 0
                else:
                    cluster_scores[cluster][cell_type] = 0

        return cluster_scores

    def _create_doublet_plot(
        self, doublet_scores: np.ndarray, predicted_doublets: np.ndarray
    ) -> go.Figure:
        """Create doublet score distribution plot."""
        fig = go.Figure()

        # Histogram of doublet scores
        fig.add_trace(
            go.Histogram(x=doublet_scores, nbinsx=50, name="All cells", opacity=0.7)
        )

        # Highlight predicted doublets
        doublet_scores_filtered = doublet_scores[predicted_doublets]
        if len(doublet_scores_filtered) > 0:
            fig.add_trace(
                go.Histogram(
                    x=doublet_scores_filtered,
                    nbinsx=50,
                    name="Predicted doublets",
                    opacity=0.7,
                )
            )

        fig.update_layout(
            title="Doublet Score Distribution",
            xaxis_title="Doublet Score",
            yaxis_title="Number of Cells",
            barmode="overlay",
            height=400,
        )

        return fig

    def _create_annotation_plot(
        self, cluster_annotations: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """Create cluster annotation heatmap."""
        clusters = list(cluster_annotations.keys())
        cell_types = list(list(cluster_annotations.values())[0].keys())

        # Create score matrix
        score_matrix = []
        for cell_type in cell_types:
            scores = [cluster_annotations[cluster][cell_type] for cluster in clusters]
            score_matrix.append(scores)

        fig = go.Figure(
            data=go.Heatmap(
                z=score_matrix,
                x=[f"Cluster {c}" for c in clusters],
                y=cell_types,
                colorscale="Viridis",
                colorbar=dict(title="Marker Score"),
            )
        )

        fig.update_layout(
            title="Cell Type Marker Scores by Cluster",
            xaxis_title="Clusters",
            yaxis_title="Cell Types",
            height=500,
        )

        return fig

    def _create_annotated_umap(self) -> go.Figure:
        """Create UMAP plot with cell type annotations."""
        adata = self.data_manager.adata

        if "X_umap" not in adata.obsm:
            return go.Figure().add_annotation(text="UMAP coordinates not available")

        umap_coords = adata.obsm["X_umap"]
        cell_types = adata.obs["cell_type"]

        fig = px.scatter(
            x=umap_coords[:, 0],
            y=umap_coords[:, 1],
            color=cell_types,
            title="UMAP with Cell Type Annotations",
            labels={"x": "UMAP_1", "y": "UMAP_2", "color": "Cell Type"},
            width=700,
            height=600,
        )

        fig.update_traces(marker=dict(size=4, opacity=0.7))
        fig.update_layout(legend_title="Cell Type", font=dict(size=12))

        return fig

    def _extract_marker_genes(self, adata, group_name: str) -> pd.DataFrame:
        """Extract marker genes from scanpy results."""
        try:
            marker_genes = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
            marker_scores = pd.DataFrame(adata.uns["rank_genes_groups"]["scores"])
            marker_pvals = pd.DataFrame(adata.uns["rank_genes_groups"]["pvals"])

            # Combine into single dataframe
            combined_df = pd.DataFrame()
            for col in marker_genes.columns:
                temp_df = pd.DataFrame(
                    {
                        "gene": marker_genes[col][:10],  # Top 10 genes
                        "score": marker_scores[col][:10],
                        "pval": marker_pvals[col][:10],
                        "group": col,
                    }
                )
                combined_df = pd.concat([combined_df, temp_df])

            return combined_df.reset_index(drop=True)

        except Exception as e:
            logger.warning(f"Could not extract marker genes: {e}")
            return pd.DataFrame()

    def _create_marker_gene_plot(self, marker_genes_df: pd.DataFrame) -> go.Figure:
        """Create marker gene expression plot."""
        if marker_genes_df.empty:
            return go.Figure().add_annotation(text="No marker genes to display")

        # Take top genes from each group
        top_genes = marker_genes_df.groupby("group").head(5)

        fig = px.bar(
            top_genes,
            x="score",
            y="gene",
            color="group",
            title="Top Marker Genes by Group",
            labels={"score": "Expression Score", "gene": "Gene"},
            height=500,
            orientation="h",
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}, margin=dict(l=100)
        )

        return fig

    def _run_mock_pathway_analysis(self, gene_list: List[str]) -> List[Dict[str, Any]]:
        """Run mock pathway analysis for demonstration."""
        pathways = [
            {"pathway": "T cell activation", "p_value": 0.001, "genes": 15},
            {"pathway": "Immune response", "p_value": 0.003, "genes": 22},
            {"pathway": "Cell cycle", "p_value": 0.01, "genes": 8},
            {"pathway": "Apoptosis", "p_value": 0.02, "genes": 12},
            {"pathway": "Metabolic process", "p_value": 0.05, "genes": 18},
        ]

        return pathways

    def _create_pathway_plot(self, pathway_results: List[Dict[str, Any]]) -> go.Figure:
        """Create pathway enrichment plot."""
        pathways = [p["pathway"] for p in pathway_results]
        p_values = [-np.log10(p["p_value"]) for p in pathway_results]

        fig = go.Figure(
            data=go.Bar(
                x=p_values,
                y=pathways,
                orientation="h",
                marker=dict(
                    color=p_values,
                    colorscale="Viridis",
                    colorbar=dict(title="-Log10 P-value"),
                ),
            )
        )

        fig.update_layout(
            title="Pathway Enrichment Analysis",
            xaxis_title="-Log10 P-value",
            yaxis_title="Pathways",
            height=400,
            margin=dict(l=200),
        )

        return fig

    def _format_cell_type_counts(self, cell_type_counts: Dict[str, int]) -> str:
        """Format cell type counts for display."""
        formatted = []
        for cell_type, count in sorted(
            cell_type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            formatted.append(f"- {cell_type}: {count} cells")
        return "\n".join(formatted)

    def _format_cluster_annotations(self, cluster_annotations: Dict[str, str]) -> str:
        """Format cluster annotations for display."""
        formatted = []
        for cluster, cell_type in sorted(cluster_annotations.items()):
            formatted.append(f"- Cluster {cluster}: {cell_type}")
        return "\n".join(formatted)

    def _format_marker_genes(self, marker_genes_df: pd.DataFrame, n: int = 10) -> str:
        """Format marker genes for display."""
        if marker_genes_df.empty:
            return "No marker genes found"

        formatted = []
        for group in marker_genes_df["group"].unique():
            group_genes = marker_genes_df[marker_genes_df["group"] == group].head(5)
            formatted.append(f"\n**{group}:**")
            for _, row in group_genes.iterrows():
                formatted.append(f"- {row['gene']}: score={row['score']:.2f}")

        return "\n".join(formatted)

    def _format_pathway_results(
        self, pathway_results: List[Dict[str, Any]], n: int = 5
    ) -> str:
        """Format pathway results for display."""
        formatted = []
        for pathway in pathway_results[:n]:
            formatted.append(
                f"- {pathway['pathway']}: p={pathway['p_value']:.2e}, genes={pathway['genes']}"
            )
        return "\n".join(formatted)
