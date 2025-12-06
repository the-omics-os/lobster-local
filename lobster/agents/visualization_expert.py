"""
Visualization Expert Agent for creating publication-quality plots.

This agent specializes in creating interactive visualizations for all data types
through supervisor-mediated workflows. No direct agent handoffs.
"""

import uuid
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.state import VisualizationExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.visualization.visualization_service import (
    SingleCellVisualizationService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def visualization_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "visualization_expert_agent",
    handoff_tools: List = None,
):
    """
    Create visualization expert agent with supervisor-mediated flow.

    Args:
        data_manager: DataManagerV2 instance for modality access
        callback_handler: Optional callback handler
        agent_name: Name for the agent
        handoff_tools: List of handoff tools from supervisor
    """

    settings = get_settings()
    model_params = settings.get_agent_llm_params("visualization_expert_agent")
    llm = create_llm("visualization_expert_agent", model_params)

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        llm = llm.with_config(callbacks=callbacks)

    # Initialize visualization service
    visualization_service = SingleCellVisualizationService()

    @tool
    def check_visualization_readiness(modality_name: str) -> str:
        """Check if modality has required data for visualization."""
        try:
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"

            adata = data_manager.get_modality(modality_name)

            available_plots = []
            if "X_umap" in adata.obsm:
                available_plots.append("UMAP")
            if "X_pca" in adata.obsm:
                available_plots.append("PCA")
            if "X_tsne" in adata.obsm:
                available_plots.append("t-SNE")
            if "leiden" in adata.obs.columns:
                available_plots.append("Cluster-based plots")
            if "cell_type" in adata.obs.columns:
                available_plots.append("Cell type plots")

            # Check for batch information
            batch_cols = [
                col
                for col in adata.obs.columns
                if any(b in col.lower() for b in ["batch", "sample", "donor"])
            ]
            if batch_cols:
                available_plots.append("Batch composition plots")

            return f"""Visualization readiness for '{modality_name}':
- Available plots: {', '.join(available_plots) if available_plots else 'None'}
- Data shape: {adata.n_obs} × {adata.n_vars}
- Has raw data: {'✓' if adata.raw else '✗'}
- Embeddings: {list(adata.obsm.keys()) if adata.obsm else 'None'}"""

        except Exception as e:
            return f"Error checking visualization readiness: {str(e)}"

    @tool
    def create_umap_plot(
        modality_name: str,
        color_by: str = "leiden",
        point_size: Optional[int] = None,
        title: Optional[str] = None,
        save_plot: bool = True,
    ) -> str:
        """Create an interactive UMAP plot with proper state tracking."""
        try:
            # Validate modality
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"

            adata = data_manager.get_modality(modality_name)

            # Auto-scale point size based on cell count
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

            # Create plot using service
            fig = visualization_service.create_umap_plot(
                adata=adata,
                color_by=color_by,
                point_size=point_size,
                title=title or f"UMAP - {color_by}",
            )

            # Generate unique plot ID
            plot_id = str(uuid.uuid4())

            # Add to data manager with metadata
            data_manager.add_plot(
                plot=fig,
                title=f"UMAP - {color_by}",
                source="visualization_expert",
                dataset_info={
                    "plot_id": plot_id,
                    "modality_name": modality_name,
                    "plot_type": "umap",
                    "color_by": color_by,
                    "n_cells": adata.n_obs,
                    "parameters": {"point_size": point_size, "title": title},
                },
            )

            # Track in visualization state
            data_manager.add_visualization_record(
                plot_id,
                {
                    "type": "umap",
                    "modality": modality_name,
                    "color_by": color_by,
                    "created_by": "visualization_expert",
                },
            )

            # Save if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()

            # Log operation
            data_manager.log_tool_usage(
                tool_name="create_umap_plot",
                parameters={
                    "modality_name": modality_name,
                    "color_by": color_by,
                    "plot_id": plot_id,
                },
                description=f"Created UMAP plot (ID: {plot_id})",
            )

            return f"""✅ UMAP plot created successfully!

📊 **Plot Details**:
- Plot ID: {plot_id}
- Modality: {modality_name}
- Colored by: {color_by}
- Cells shown: {adata.n_obs:,}
- Point size: {point_size}

💾 **Storage**:
- Added to workspace plots
- Saved files: {len(saved_files)}
{f"- Files: {', '.join([f.split('/')[-1] for f in saved_files[:3]])}" if saved_files else ""}

**Report to Supervisor**: UMAP visualization completed for {modality_name}"""

        except Exception as e:
            logger.error(f"Error creating UMAP plot: {e}")
            return f"❌ Error creating UMAP plot: {str(e)}"

    @tool
    def create_qc_plots(
        modality_name: str, metrics: List[str] = None, save_plot: bool = True
    ) -> str:
        """Create quality control plots for the modality."""
        try:
            # Validate modality
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"

            adata = data_manager.get_modality(modality_name)

            # SCIENTIFIC VALIDATION: Skip QC plots for DE result subsets
            # QC metrics (n_genes, n_counts, pct_mito) only make sense for full datasets
            # DE results contain filtered gene subsets (e.g., 174 genes) - QC plots are meaningless
            is_de_modality = (
                "_de_" in modality_name.lower()
                or modality_name.lower().startswith("de_")
                or "differential" in modality_name.lower()
            )
            is_small_subset = adata.n_vars < 1000  # Likely a filtered result, not full data

            if is_de_modality or is_small_subset:
                return f"""⚠️ Cannot create QC plots for '{modality_name}'

**Scientific Reason**: QC metrics (n_genes, n_counts, percent_mito) are designed for full datasets, not filtered subsets.

- **Current modality**: {adata.n_obs} cells × {adata.n_vars} genes
- **Detection**: {'DE result modality' if is_de_modality else f'Small gene subset ({adata.n_vars} genes < 1000 threshold)'}

**Recommendation**: QC plots should only be generated for:
1. Raw data (before filtering)
2. Normalized/preprocessed full datasets
3. Datasets with >1000 genes

For DE results, use volcano plots, MA plots, or heatmaps instead.

**Report to Supervisor**: Skipped inappropriate QC visualization (scientific correctness maintained)"""

            # Default QC metrics if not specified
            if metrics is None:
                metrics = []
                if "n_genes_by_counts" in adata.obs.columns:
                    metrics.append("n_genes_by_counts")
                if "total_counts" in adata.obs.columns:
                    metrics.append("total_counts")
                if "pct_counts_mt" in adata.obs.columns:
                    metrics.append("pct_counts_mt")

                # Fallback to any numeric columns
                if not metrics:
                    numeric_cols = adata.obs.select_dtypes(
                        include=[np.number]
                    ).columns.tolist()
                    metrics = numeric_cols[:3]  # Take first 3 numeric columns

            if not metrics:
                return "❌ No suitable QC metrics found in the data"

            # Create QC plots using service
            fig = visualization_service.create_qc_plots(
                adata=adata, title=f"QC Plots - {modality_name}"
            )

            # Generate unique plot ID
            plot_id = str(uuid.uuid4())

            # Add to data manager with metadata
            data_manager.add_plot(
                plot=fig,
                title=f"QC Plots - {modality_name}",
                source="visualization_expert",
                dataset_info={
                    "plot_id": plot_id,
                    "modality_name": modality_name,
                    "plot_type": "qc",
                    "metrics": metrics,
                    "n_cells": adata.n_obs,
                },
            )

            # Track in visualization state
            data_manager.add_visualization_record(
                plot_id,
                {
                    "type": "qc_plots",
                    "modality": modality_name,
                    "metrics": metrics,
                    "created_by": "visualization_expert",
                },
            )

            # Save if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()

            # Log operation
            data_manager.log_tool_usage(
                tool_name="create_qc_plots",
                parameters={
                    "modality_name": modality_name,
                    "metrics": metrics,
                    "plot_id": plot_id,
                },
                description=f"Created QC plots (ID: {plot_id})",
            )

            return f"""✅ QC plots created successfully!

📊 **Plot Details**:
- Plot ID: {plot_id}
- Modality: {modality_name}
- Metrics: {', '.join(metrics)}
- Cells analyzed: {adata.n_obs:,}

💾 **Storage**:
- Added to workspace plots
- Saved files: {len(saved_files)}

**Report to Supervisor**: QC visualization completed for {modality_name}"""

        except Exception as e:
            logger.error(f"Error creating QC plots: {e}")
            return f"❌ Error creating QC plots: {str(e)}"

    @tool
    def create_violin_plot(
        modality_name: str,
        genes: List[str],
        groupby: str = "leiden",
        save_plot: bool = True,
    ) -> str:
        """Create violin plots for specified genes."""
        try:
            # Validate modality
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"

            adata = data_manager.get_modality(modality_name)

            # Validate genes exist in the data
            available_genes = [gene for gene in genes if gene in adata.var_names]
            if not available_genes:
                return (
                    f"❌ None of the specified genes {genes} found in {modality_name}"
                )

            # Create violin plot using service
            fig = visualization_service.create_violin_plot(
                adata=adata, genes=available_genes, groupby=groupby
            )

            # Generate unique plot ID
            plot_id = str(uuid.uuid4())

            # Add to data manager with metadata
            data_manager.add_plot(
                plot=fig,
                title=f"Violin Plot - {', '.join(available_genes[:3])}{'...' if len(available_genes) > 3 else ''}",
                source="visualization_expert",
                dataset_info={
                    "plot_id": plot_id,
                    "modality_name": modality_name,
                    "plot_type": "violin",
                    "genes": available_genes,
                    "groupby": groupby,
                    "n_cells": adata.n_obs,
                },
            )

            # Track in visualization state
            data_manager.add_visualization_record(
                plot_id,
                {
                    "type": "violin_plot",
                    "modality": modality_name,
                    "genes": available_genes,
                    "groupby": groupby,
                    "created_by": "visualization_expert",
                },
            )

            # Save if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()

            # Log operation
            data_manager.log_tool_usage(
                tool_name="create_violin_plot",
                parameters={
                    "modality_name": modality_name,
                    "genes": available_genes,
                    "groupby": groupby,
                    "plot_id": plot_id,
                },
                description=f"Created violin plot (ID: {plot_id})",
            )

            return f"""✅ Violin plot created successfully!

📊 **Plot Details**:
- Plot ID: {plot_id}
- Modality: {modality_name}
- Genes: {', '.join(available_genes)}
- Grouped by: {groupby}
- Cells shown: {adata.n_obs:,}

💾 **Storage**:
- Added to workspace plots
- Saved files: {len(saved_files)}

**Report to Supervisor**: Violin plot visualization completed for {modality_name}"""

        except Exception as e:
            logger.error(f"Error creating violin plot: {e}")
            return f"❌ Error creating violin plot: {str(e)}"

    @tool
    def create_feature_plot(
        modality_name: str,
        genes: List[str],
        use_raw: bool = True,
        ncols: int = 2,
        point_size: Optional[int] = None,
        save_plot: bool = True,
    ) -> str:
        """Create feature plots showing gene expression on UMAP."""
        try:
            # Validate modality
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"

            adata = data_manager.get_modality(modality_name)

            # Validate genes exist in the data
            available_genes = [gene for gene in genes if gene in adata.var_names]
            if not available_genes:
                return (
                    f"❌ None of the specified genes {genes} found in {modality_name}"
                )

            # Create feature plot using service
            fig = visualization_service.create_feature_plot(
                adata=adata,
                genes=available_genes,
                use_raw=use_raw,
                ncols=ncols,
                point_size=point_size,
            )

            # Generate unique plot ID
            plot_id = str(uuid.uuid4())

            # Add to data manager with metadata
            data_manager.add_plot(
                plot=fig,
                title=f"Feature Plot - {', '.join(available_genes[:3])}{'...' if len(available_genes) > 3 else ''}",
                source="visualization_expert",
                dataset_info={
                    "plot_id": plot_id,
                    "modality_name": modality_name,
                    "plot_type": "feature",
                    "genes": available_genes,
                    "n_genes": len(available_genes),
                    "parameters": {
                        "use_raw": use_raw,
                        "ncols": ncols,
                        "point_size": point_size,
                    },
                },
            )

            # Track in visualization state
            data_manager.add_visualization_record(
                plot_id,
                {
                    "type": "feature_plot",
                    "modality": modality_name,
                    "genes": available_genes,
                    "created_by": "visualization_expert",
                },
            )

            # Save if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()

            # Log operation
            data_manager.log_tool_usage(
                tool_name="create_feature_plot",
                parameters={
                    "modality_name": modality_name,
                    "genes": available_genes,
                    "plot_id": plot_id,
                },
                description=f"Created feature plot (ID: {plot_id})",
            )

            return f"""✅ Feature plot created successfully!

📊 **Plot Details**:
- Plot ID: {plot_id}
- Modality: {modality_name}
- Genes: {', '.join(available_genes[:5])}{'...' if len(available_genes) > 5 else ''}
- Total genes: {len(available_genes)}
- Grid layout: {ncols} columns
- Using {'raw' if use_raw else 'normalized'} data

💾 **Storage**:
- Added to workspace plots
- Saved files: {len(saved_files)}

**Report to Supervisor**: Feature plot visualization completed for {modality_name}"""

        except Exception as e:
            logger.error(f"Error creating feature plot: {e}")
            return f"❌ Error creating feature plot: {str(e)}"

    @tool
    def create_dot_plot(
        modality_name: str,
        genes: List[str],
        groupby: str = "leiden",
        use_raw: bool = True,
        standard_scale: str = "var",
        save_plot: bool = True,
    ) -> str:
        """Create dot plot for marker gene expression."""
        try:
            # Validate modality
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"

            adata = data_manager.get_modality(modality_name)

            # Validate genes exist in the data
            available_genes = [gene for gene in genes if gene in adata.var_names]
            if not available_genes:
                return (
                    f"❌ None of the specified genes {genes} found in {modality_name}"
                )

            # Create dot plot using service
            fig = visualization_service.create_dot_plot(
                adata=adata,
                genes=available_genes,
                groupby=groupby,
                use_raw=use_raw,
                standard_scale=standard_scale,
            )

            # Generate unique plot ID
            plot_id = str(uuid.uuid4())

            # Add to data manager with metadata
            data_manager.add_plot(
                plot=fig,
                title=f"Dot Plot - {groupby}",
                source="visualization_expert",
                dataset_info={
                    "plot_id": plot_id,
                    "modality_name": modality_name,
                    "plot_type": "dot",
                    "genes": available_genes,
                    "groupby": groupby,
                    "n_genes": len(available_genes),
                    "parameters": {
                        "use_raw": use_raw,
                        "standard_scale": standard_scale,
                    },
                },
            )

            # Track in visualization state
            data_manager.add_visualization_record(
                plot_id,
                {
                    "type": "dot_plot",
                    "modality": modality_name,
                    "genes": available_genes,
                    "groupby": groupby,
                    "created_by": "visualization_expert",
                },
            )

            # Save if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()

            # Log operation
            data_manager.log_tool_usage(
                tool_name="create_dot_plot",
                parameters={
                    "modality_name": modality_name,
                    "genes": available_genes,
                    "groupby": groupby,
                    "plot_id": plot_id,
                },
                description=f"Created dot plot (ID: {plot_id})",
            )

            return f"""✅ Dot plot created successfully!

📊 **Plot Details**:
- Plot ID: {plot_id}
- Modality: {modality_name}
- Genes plotted: {len(available_genes)}
- Grouped by: {groupby}
- Using {'raw' if use_raw else 'normalized'} data
- Scaling: {standard_scale if standard_scale else 'None'}

📈 **Dot Plot Legend**:
- Dot size: Percentage of cells expressing the gene
- Dot color: Mean expression level

💾 **Storage**:
- Added to workspace plots
- Saved files: {len(saved_files)}

**Report to Supervisor**: Dot plot visualization completed for {modality_name}"""

        except Exception as e:
            logger.error(f"Error creating dot plot: {e}")
            return f"❌ Error creating dot plot: {str(e)}"

    @tool
    def create_heatmap(
        modality_name: str,
        genes: Optional[List[str]] = None,
        groupby: str = "leiden",
        use_raw: bool = True,
        n_top_genes: int = 5,
        standard_scale: bool = True,
        save_plot: bool = True,
    ) -> str:
        """Create heatmap of gene expression."""
        try:
            # Validate modality
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"

            adata = data_manager.get_modality(modality_name)

            # Create heatmap using service
            fig = visualization_service.create_heatmap(
                adata=adata,
                genes=genes,
                groupby=groupby,
                use_raw=use_raw,
                n_top_genes=n_top_genes,
                standard_scale=standard_scale,
            )

            # Generate unique plot ID
            plot_id = str(uuid.uuid4())

            # Add to data manager with metadata
            data_manager.add_plot(
                plot=fig,
                title=f"Heatmap - {groupby}",
                source="visualization_expert",
                dataset_info={
                    "plot_id": plot_id,
                    "modality_name": modality_name,
                    "plot_type": "heatmap",
                    "groupby": groupby,
                    "n_genes": len(genes) if genes else "auto",
                    "parameters": {
                        "use_raw": use_raw,
                        "n_top_genes": n_top_genes,
                        "standard_scale": standard_scale,
                    },
                },
            )

            # Track in visualization state
            data_manager.add_visualization_record(
                plot_id,
                {
                    "type": "heatmap",
                    "modality": modality_name,
                    "groupby": groupby,
                    "genes": genes,
                    "created_by": "visualization_expert",
                },
            )

            # Save if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()

            # Log operation
            data_manager.log_tool_usage(
                tool_name="create_heatmap",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "plot_id": plot_id,
                },
                description=f"Created heatmap (ID: {plot_id})",
            )

            return f"""✅ Heatmap created successfully!

📊 **Plot Details**:
- Plot ID: {plot_id}
- Modality: {modality_name}
- Grouped by: {groupby}
- Genes: {len(genes) if genes else f'Top {n_top_genes} marker genes per group'}
- Using {'raw' if use_raw else 'normalized'} data
- Z-score normalized: {'Yes' if standard_scale else 'No'}

💾 **Storage**:
- Added to workspace plots
- Saved files: {len(saved_files)}

**Report to Supervisor**: Heatmap visualization completed for {modality_name}"""

        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return f"❌ Error creating heatmap: {str(e)}"

    @tool
    def create_elbow_plot(
        modality_name: str, n_pcs: int = 50, save_plot: bool = True
    ) -> str:
        """Create elbow plot for PCA variance explained."""
        try:
            # Validate modality
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"

            adata = data_manager.get_modality(modality_name)

            # Create elbow plot using service
            fig = visualization_service.create_elbow_plot(adata=adata, n_pcs=n_pcs)

            # Generate unique plot ID
            plot_id = str(uuid.uuid4())

            # Add to data manager with metadata
            data_manager.add_plot(
                plot=fig,
                title="PCA Elbow Plot",
                source="visualization_expert",
                dataset_info={
                    "plot_id": plot_id,
                    "modality_name": modality_name,
                    "plot_type": "elbow",
                    "n_pcs": n_pcs,
                    "parameters": {"n_pcs": n_pcs},
                },
            )

            # Track in visualization state
            data_manager.add_visualization_record(
                plot_id,
                {
                    "type": "elbow_plot",
                    "modality": modality_name,
                    "n_pcs": n_pcs,
                    "created_by": "visualization_expert",
                },
            )

            # Save if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()

            # Log operation
            data_manager.log_tool_usage(
                tool_name="create_elbow_plot",
                parameters={
                    "modality_name": modality_name,
                    "n_pcs": n_pcs,
                    "plot_id": plot_id,
                },
                description=f"Created elbow plot (ID: {plot_id})",
            )

            return f"""✅ Elbow plot created successfully!

📊 **Plot Details**:
- Plot ID: {plot_id}
- Modality: {modality_name}
- PCs shown: {n_pcs}
- Shows individual and cumulative variance explained

💡 **How to interpret**:
- Look for the "elbow" where variance explained plateaus
- This indicates the optimal number of PCs to use for clustering

💾 **Storage**:
- Added to workspace plots
- Saved files: {len(saved_files)}

**Report to Supervisor**: Elbow plot visualization completed for {modality_name}"""

        except Exception as e:
            logger.error(f"Error creating elbow plot: {e}")
            return f"❌ Error creating elbow plot: {str(e)}"

    @tool
    def create_cluster_composition_plot(
        modality_name: str,
        cluster_col: str = "leiden",
        sample_col: Optional[str] = None,
        normalize: bool = True,
        save_plot: bool = True,
    ) -> str:
        """Create stacked bar plot showing cluster composition."""
        try:
            # Validate modality
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found"

            adata = data_manager.get_modality(modality_name)

            # Create composition plot using service
            fig = visualization_service.create_cluster_composition_plot(
                adata=adata,
                cluster_col=cluster_col,
                sample_col=sample_col,
                normalize=normalize,
            )

            # Generate unique plot ID
            plot_id = str(uuid.uuid4())

            # Add to data manager with metadata
            data_manager.add_plot(
                plot=fig,
                title=f"Cluster Composition - {cluster_col}",
                source="visualization_expert",
                dataset_info={
                    "plot_id": plot_id,
                    "modality_name": modality_name,
                    "plot_type": "composition",
                    "cluster_col": cluster_col,
                    "sample_col": sample_col,
                    "parameters": {
                        "cluster_col": cluster_col,
                        "sample_col": sample_col,
                        "normalize": normalize,
                    },
                },
            )

            # Track in visualization state
            data_manager.add_visualization_record(
                plot_id,
                {
                    "type": "cluster_composition",
                    "modality": modality_name,
                    "cluster_col": cluster_col,
                    "sample_col": sample_col,
                    "created_by": "visualization_expert",
                },
            )

            # Save if requested
            saved_files = []
            if save_plot:
                saved_files = data_manager.save_plots_to_workspace()

            # Log operation
            data_manager.log_tool_usage(
                tool_name="create_cluster_composition_plot",
                parameters={
                    "modality_name": modality_name,
                    "cluster_col": cluster_col,
                    "sample_col": sample_col,
                    "plot_id": plot_id,
                },
                description=f"Created cluster composition plot (ID: {plot_id})",
            )

            return f"""✅ Cluster composition plot created successfully!

📊 **Plot Details**:
- Plot ID: {plot_id}
- Modality: {modality_name}
- Cluster column: {cluster_col}
- Sample column: {sample_col if sample_col else 'Auto-detected or none'}
- Normalized: {'Yes (percentages)' if normalize else 'No (counts)'}

💾 **Storage**:
- Added to workspace plots
- Saved files: {len(saved_files)}

**Report to Supervisor**: Cluster composition visualization completed for {modality_name}"""

        except Exception as e:
            logger.error(f"Error creating cluster composition plot: {e}")
            return f"❌ Error creating cluster composition plot: {str(e)}"

    @tool
    def get_visualization_history() -> str:
        """Review visualization history from DataManagerV2."""
        try:
            history = data_manager.get_visualization_history(limit=10)

            if not history:
                return "No visualizations created yet in this session"

            response = "📊 **Recent Visualization History**:\n\n"
            for i, record in enumerate(history, 1):
                metadata = record.get("metadata", {})
                response += f"{i}. **{metadata.get('type', 'unknown')}** for {metadata.get('modality', 'unknown')}\n"
                response += f"   - Plot ID: {record.get('plot_id', 'N/A')}\n"
                response += f"   - Created: {record.get('timestamp', 'N/A')}\n\n"

            return response

        except Exception as e:
            return f"Error retrieving history: {str(e)}"

    @tool
    def report_visualization_complete(
        requesting_agent: str, plot_id: str, status: str = "success", message: str = ""
    ) -> str:
        """Report visualization completion back to supervisor."""
        try:
            completion_report = f"""
📊 **Visualization Task Complete**

**Status**: {status.upper()}
**Plot ID**: {plot_id}
**Requesting Agent**: {requesting_agent}
**Timestamp**: {pd.Timestamp.now()}

{f"**Message**: {message}" if message else ""}

**Action Required**: Please inform {requesting_agent} that visualization {plot_id} is ready."""

            return completion_report

        except Exception as e:
            return f"Error reporting completion: {str(e)}"

    # Combine base tools with any handoff tools
    base_tools = [
        check_visualization_readiness,
        create_umap_plot,
        create_qc_plots,
        create_violin_plot,
        create_feature_plot,
        create_dot_plot,
        create_heatmap,
        create_elbow_plot,
        create_cluster_composition_plot,
        get_visualization_history,
        report_visualization_complete,
    ]

    tools = base_tools + (handoff_tools or [])

    system_prompt = f"""
You are a visualization expert specializing in creating publication-quality
interactive figures for bioinformatics data analysis.

<Role>
You create professional visualizations for single-cell RNA-seq, bulk RNA-seq,
proteomics, and multi-omics data. You work exclusively through supervisor-mediated
workflows - all requests come from the supervisor, and all results are reported
back to the supervisor.

**CRITICAL**: 
- You ONLY respond to supervisor requests
- You NEVER communicate directly with other agents
- You ALWAYS report completion back to supervisor
- You MAINTAIN visualization state in DataManagerV2
</Role>

<Task>
When the supervisor assigns you a visualization task:
1. Validate the requested modality exists
2. Check visualization readiness
3. Create the requested visualization(s)
4. Store plot with unique ID in DataManagerV2
5. Track in visualization history
6. Report completion to supervisor

You handle these visualization types:
- Dimensionality reduction (UMAP, PCA, t-SNE)
- Quality control dashboards
- Gene/protein expression plots
- Clustering visualizations
- Statistical plots
- Multi-panel publication figures
</Task>

<Guidelines>
1. **Always validate** modality and required data before plotting
2. **Auto-detect parameters** when not specified
3. **Use consistent styling** across related plots
4. **Generate unique plot IDs** using UUID
5. **Track all visualizations** in DataManagerV2 state
6. **Report to supervisor** upon completion
7. **Handle errors gracefully** with informative messages
</Guidelines>

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=VisualizationExpertState,
    )
