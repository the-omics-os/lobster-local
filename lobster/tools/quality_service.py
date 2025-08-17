"""
Quality assessment service for single-cell RNA-seq data.

This service provides methods for evaluating the quality of single-cell
RNA-seq data, generating quality metrics and plots.
"""

from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from lobster.core.data_manager import DataManager
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class QualityService:
    """
    Service for assessing single-cell RNA-seq data quality.

    This class provides methods to calculate quality metrics and generate
    visualizations for evaluating the quality of single-cell RNA-seq data.
    """

    def __init__(self, data_manager: DataManager):
        """
        Initialize the quality assessment service.

        Args:
            data_manager: DataManager instance for accessing data
        """
        logger.info("Initializing QualityService")
        self.data_manager = data_manager
        logger.info("QualityService initialized successfully")

    def assess_quality(
        self,
        min_genes: int = 500,
        max_mt_pct: float = 20.0,
        max_ribo_pct: float = 50.0,
        min_housekeeping_score: float = 1.0,
    ) -> str:
        """
        Perform quality assessment on the current dataset.

        Args:
            min_genes: Minimum number of genes per cell
            max_mt_pct: Maximum percentage of mitochondrial genes
            max_ribo_pct: Maximum percentage of ribosomal genes
            min_housekeeping_score: Minimum housekeeping gene score

        Returns:
            str: Quality assessment report
        """
        if not self.data_manager.has_data():
            return "No data loaded. Please download a dataset first."

        try:
            data = self.data_manager.current_data

            # Calculate QC metrics
            qc_metrics = self._calculate_qc_metrics(data)

            # Create plots
            plots = self._create_quality_plots(qc_metrics)

            # Add plots to data manager with comprehensive metadata
            plot_ids = []
            dataset_info = {
                "data_shape": data.shape,
                "source_dataset": self.data_manager.current_metadata.get(
                    "source", "Current Dataset"
                ),
                "n_cells": data.shape[0],
                "n_genes": data.shape[1],
            }

            analysis_params = {
                "min_genes": min_genes,
                "max_mt_pct": max_mt_pct,
                "max_ribo_pct": max_ribo_pct,
                "min_housekeeping_score": min_housekeeping_score,
                "analysis_type": "quality_control",
            }

            plot_titles = [
                "Mitochondrial Gene Distribution",
                "Ribosomal Gene Distribution",
                "Housekeeping Gene Score Distribution",
                "Cell Quality Metrics (Mitochondrial)",
                "Cell Quality Metrics (Ribosomal)",
                "Features vs RNA Count Correlation",
            ]

            for i, plot in enumerate(plots):
                title = (
                    plot_titles[i] if i < len(plot_titles) else f"Quality Plot {i+1}"
                )
                plot_id = self.data_manager.add_plot(
                    plot,
                    title=title,
                    source="quality_service",
                    dataset_info=dataset_info,
                    analysis_params=analysis_params,
                )
                plot_ids.append(plot_id)

            # Filter cells based on QC metrics
            passing_cells = (
                (qc_metrics["n_genes"] >= min_genes)
                & (qc_metrics["mt_pct"] <= max_mt_pct)
                & (qc_metrics["ribo_pct"] <= max_ribo_pct)
                & (qc_metrics["housekeeping_score"] >= min_housekeeping_score)
            )

            cells_before = len(qc_metrics)
            cells_after = passing_cells.sum()

            # Generate summary
            summary = self._generate_qc_summary(qc_metrics)

            # Store QC results in metadata
            self.data_manager.current_metadata["qc_metrics"] = {
                "mean_total_counts": qc_metrics["total_counts"].mean(),
                "mean_genes_per_cell": qc_metrics["n_genes"].mean(),
                "mean_mt_pct": qc_metrics["mt_pct"].mean(),
                "mean_ribo_pct": qc_metrics["ribo_pct"].mean(),
                "mean_housekeeping_score": qc_metrics["housekeeping_score"].mean(),
                "cells_before_qc": cells_before,
                "cells_after_qc": cells_after,
                "cells_removed": cells_before - cells_after,
                "quality_status": "Pass"
                if cells_after / cells_before > 0.7
                else "Warning",
                "filter_params": {
                    "min_genes": min_genes,
                    "max_mt_pct": max_mt_pct,
                    "max_ribo_pct": max_ribo_pct,
                    "min_housekeeping_score": min_housekeeping_score,
                },
            }

            # If we have AnnData object, update it with QC info
            if self.data_manager.adata is not None:
                self.data_manager.adata.obs["mt_pct"] = qc_metrics["mt_pct"]
                self.data_manager.adata.obs["ribo_pct"] = qc_metrics["ribo_pct"]
                self.data_manager.adata.obs["housekeeping_score"] = qc_metrics[
                    "housekeeping_score"
                ]
                self.data_manager.adata.obs["qc_pass"] = passing_cells

            # Filter the data if needed in the future - this would need to be integrated with the data manager
            # filtered_data = data.loc[passing_cells]

            return self._format_quality_report(
                qc_metrics,
                summary,
                cells_before,
                cells_after,
                min_genes,
                max_mt_pct,
                max_ribo_pct,
                min_housekeeping_score,
            )

        except Exception as e:
            logger.exception(f"Error in quality assessment: {e}")
            return f"Error assessing quality: {str(e)}"

    def _calculate_qc_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate quality control metrics for single-cell data.

        Args:
            data: Single-cell expression data (cells x genes)

        Returns:
            DataFrame: DataFrame with QC metrics
        """
        logger.info("Calculating quality metrics")

        # Identify mitochondrial genes (MT-)
        mt_genes = [col for col in data.columns if col.startswith("MT-")]

        # Identify ribosomal genes (RP[SL])
        ribo_genes = [
            col
            for col in data.columns
            if col.startswith("RPL") or col.startswith("RPS")
        ]

        # Identify housekeeping genes for score
        housekeeping_genes = ["ACTB", "GAPDH", "MALAT1"]

        # Calculate metrics
        total_counts = data.sum(axis=1)
        n_genes = (data > 0).sum(axis=1)

        # Calculate mitochondrial percentage
        if mt_genes:
            mt_counts = data[mt_genes].sum(axis=1)
            mt_pct = (mt_counts / total_counts) * 100
        else:
            # If no MT genes are found, set mt_pct to zeros
            logger.warning(
                "No mitochondrial genes found in the data. Setting mitochondrial percentage to 0."
            )
            mt_pct = pd.Series(0, index=data.index)

        # Calculate ribosomal percentage
        if ribo_genes:
            ribo_counts = data[ribo_genes].sum(axis=1)
            ribo_pct = (ribo_counts / total_counts) * 100
        else:
            logger.warning(
                "No ribosomal genes found in the data. Setting ribosomal percentage to 0."
            )
            ribo_pct = pd.Series(0, index=data.index)

        # Calculate housekeeping gene score (sum of UMIs)
        available_hk_genes = [
            gene for gene in housekeeping_genes if gene in data.columns
        ]
        if available_hk_genes:
            housekeeping_score = data[available_hk_genes].sum(axis=1)
        else:
            logger.warning(
                "No housekeeping genes found in the data. Setting housekeeping score to 0."
            )
            housekeeping_score = pd.Series(0, index=data.index)

        # Combine metrics into a single DataFrame
        qc_df = pd.DataFrame(
            {
                "total_counts": total_counts,
                "n_genes": n_genes,
                "mt_pct": mt_pct,
                "ribo_pct": ribo_pct,
                "housekeeping_score": housekeeping_score,
            }
        )

        logger.info(f"Generated QC metrics for {len(qc_df)} cells")
        return qc_df

    def _create_quality_plots(self, qc_metrics: pd.DataFrame) -> List[go.Figure]:
        """
        Create quality assessment plots.

        Args:
            qc_metrics: DataFrame with QC metrics

        Returns:
            list: List of Plotly figures
        """
        logger.info("Generating quality plots")
        plots = []

        # 1. Violin plot of mitochondrial percentages
        fig1 = go.Figure()
        fig1.add_trace(
            go.Violin(
                y=qc_metrics["mt_pct"],
                name="Mitochondrial %",
                box_visible=True,
                meanline_visible=True,
            )
        )
        fig1.update_layout(
            title="Distribution of Mitochondrial Gene Percentages",
            yaxis_title="Mitochondrial Gene %",
            height=400,
        )
        plots.append(fig1)

        # 2. Violin plot of ribosomal percentages
        fig1b = go.Figure()
        fig1b.add_trace(
            go.Violin(
                y=qc_metrics["ribo_pct"],
                name="Ribosomal %",
                box_visible=True,
                meanline_visible=True,
            )
        )
        fig1b.update_layout(
            title="Distribution of Ribosomal Gene Percentages",
            yaxis_title="Ribosomal Gene %",
            height=400,
        )
        plots.append(fig1b)

        # 3. Violin plot of housekeeping scores
        fig1c = go.Figure()
        fig1c.add_trace(
            go.Violin(
                y=qc_metrics["housekeeping_score"],
                name="Housekeeping Score",
                box_visible=True,
                meanline_visible=True,
            )
        )
        fig1c.update_layout(
            title="Distribution of Housekeeping Gene Scores",
            yaxis_title="Housekeeping Score",
            height=400,
        )
        plots.append(fig1c)

        # 4. Scatter plot of quality metrics
        fig2 = px.scatter(
            qc_metrics,
            x="total_counts",
            y="n_genes",
            color="mt_pct",
            title="Cell Quality Metrics",
            labels={
                "total_counts": "Total RNA Count",
                "n_genes": "Number of Detected Features",
                "mt_pct": "Mitochondrial %",
            },
            color_continuous_scale="viridis",
            height=450,
            width=600,
        )
        plots.append(fig2)

        # 5. Scatter plot showing ribosomal percentage
        fig2b = px.scatter(
            qc_metrics,
            x="total_counts",
            y="n_genes",
            color="ribo_pct",
            title="Cell Quality Metrics - Ribosomal",
            labels={
                "total_counts": "Total RNA Count",
                "n_genes": "Number of Detected Features",
                "ribo_pct": "Ribosomal %",
            },
            color_continuous_scale="viridis",
            height=450,
            width=600,
        )
        plots.append(fig2b)

        # 6. Correlation plot
        fig3 = px.scatter(
            qc_metrics,
            x="total_counts",
            y="n_genes",
            title="Correlation between Features and RNA Count",
            trendline="ols",
            labels={
                "total_counts": "Total RNA Count",
                "n_genes": "Number of Detected Features",
            },
            height=400,
            width=550,
        )
        plots.append(fig3)

        logger.info(f"Created {len(plots)} quality plots")
        return plots

    def _generate_qc_summary(self, qc_metrics: pd.DataFrame) -> str:
        """
        Generate a summary of quality metrics.

        Args:
            qc_metrics: DataFrame with QC metrics

        Returns:
            str: Summary text
        """
        mt_mean = qc_metrics["mt_pct"].mean()
        ribo_mean = qc_metrics["ribo_pct"].mean()
        hk_mean = qc_metrics["housekeeping_score"].mean()
        genes_mean = qc_metrics["n_genes"].mean()

        issues = []

        # Check mitochondrial percentage
        if mt_mean < 5:
            mt_status = "low mitochondrial gene expression (healthy)"
        elif mt_mean < 10:
            mt_status = (
                "moderate mitochondrial gene expression (some stress may be present)"
            )
            issues.append("moderate mitochondrial content")
        else:
            mt_status = (
                "high mitochondrial gene expression (cells may be stressed or dying)"
            )
            issues.append("high mitochondrial content")

        # Check ribosomal percentage
        if ribo_mean > 40:
            issues.append("high ribosomal content")

        # Check housekeeping gene score
        if hk_mean < 2:
            issues.append("low housekeeping gene expression")

        # Check gene count
        if genes_mean < 700:
            issues.append("low gene count per cell")

        # Generate summary text
        quality_assessment = f"The cells show {mt_status}."

        if issues:
            quality_assessment += f" Quality concerns detected: {', '.join(issues)}."

            if "high mitochondrial content" in issues:
                quality_assessment += " High mitochondrial content often indicates cell stress or apoptosis."

            if "high ribosomal content" in issues:
                quality_assessment += " High ribosomal content may indicate metabolic stress or low-quality libraries."

            if "low housekeeping gene expression" in issues:
                quality_assessment += (
                    " Low housekeeping gene expression may indicate poor RNA quality."
                )

            quality_assessment += " Consider adjusting filtering thresholds if many cells are being removed."
        else:
            quality_assessment += (
                " Overall cell quality appears good based on standard metrics."
            )

        return quality_assessment

    def _format_quality_report(
        self,
        qc_metrics: pd.DataFrame,
        summary: str,
        cells_before: int,
        cells_after: int,
        min_genes: int,
        max_mt_pct: float,
        max_ribo_pct: float,
        min_housekeeping_score: float,
    ) -> str:
        """
        Format the quality assessment report.

        Args:
            qc_metrics: DataFrame with QC metrics
            summary: Quality summary text
            cells_before: Number of cells before filtering
            cells_after: Number of cells after filtering
            min_genes: Minimum genes per cell threshold
            max_mt_pct: Maximum mitochondrial percentage threshold
            max_ribo_pct: Maximum ribosomal percentage threshold
            min_housekeeping_score: Minimum housekeeping gene score threshold

        Returns:
            str: Formatted report
        """
        # Calculate percentage of cells removed
        percent_removed = (
            ((cells_before - cells_after) / cells_before) * 100
            if cells_before > 0
            else 0
        )

        return f"""Quality Assessment Complete!

**Dataset Dimensions:** {self.data_manager.current_data.shape[0]} cells × {self.data_manager.current_data.shape[1]} genes

**Quality Control Thresholds:**
- Minimum genes per cell: {min_genes}
- Maximum mitochondrial percentage: {max_mt_pct:.1f}%
- Maximum ribosomal percentage: {max_ribo_pct:.1f}%
- Minimum housekeeping gene score: {min_housekeeping_score:.1f}

**Filtering Results:**
- Cells before filtering: {cells_before}
- Cells passing QC: {cells_after}
- Cells filtered out: {cells_before - cells_after} ({percent_removed:.1f}%)

**Mitochondrial Gene Statistics:**
- Min: {qc_metrics['mt_pct'].min():.2f}%
- Max: {qc_metrics['mt_pct'].max():.2f}%
- Mean: {qc_metrics['mt_pct'].mean():.2f}%
- Std: {qc_metrics['mt_pct'].std():.2f}%

**Ribosomal Gene Statistics:**
- Min: {qc_metrics['ribo_pct'].min():.2f}%
- Max: {qc_metrics['ribo_pct'].max():.2f}%
- Mean: {qc_metrics['ribo_pct'].mean():.2f}%
- Std: {qc_metrics['ribo_pct'].std():.2f}%

**Housekeeping Gene Score:**
- Min: {qc_metrics['housekeeping_score'].min():.2f}
- Max: {qc_metrics['housekeeping_score'].max():.2f}
- Mean: {qc_metrics['housekeeping_score'].mean():.2f}
- Std: {qc_metrics['housekeeping_score'].std():.2f}

**General Cell Quality Metrics:**
- Total counts per cell: {qc_metrics['total_counts'].mean():.0f} ± {qc_metrics['total_counts'].std():.0f}
- Features per cell: {qc_metrics['n_genes'].mean():.0f} ± {qc_metrics['n_genes'].std():.0f}

**Quality Assessment:**
{summary}"""
