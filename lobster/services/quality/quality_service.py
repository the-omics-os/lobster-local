"""
Quality assessment service for single-cell RNA-seq data.

This service provides methods for evaluating the quality of single-cell
RNA-seq data, generating quality metrics and plots.

This service emits Intermediate Representation (IR) for automatic notebook export.
"""

from typing import Any, Dict, List, Tuple

import anndata
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class QualityError(Exception):
    """Base exception for quality assessment operations."""

    pass


class QualityService:
    """
    Stateless service for assessing single-cell RNA-seq data quality.

    This class provides methods to calculate quality metrics and generate
    visualizations for evaluating the quality of single-cell RNA-seq data.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the quality assessment service.

        Args:
            config: Optional configuration dict (ignored, for backward compatibility)
            **kwargs: Additional arguments (ignored, for backward compatibility)

        This service is stateless and doesn't require a data manager instance.
        """
        logger.debug("Initializing stateless QualityService")
        self.config = config or {}
        logger.debug("QualityService initialized successfully")

    def assess_quality(
        self,
        adata: anndata.AnnData,
        min_genes: int = 500,
        max_genes: int = 5000,
        max_mt_pct: float = 20.0,
        max_ribo_pct: float = 50.0,
        min_housekeeping_score: float = 1.0,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Perform quality assessment on single-cell RNA-seq data.

        Args:
            adata: AnnData object to assess
            min_genes: Minimum number of genes per cell
            max_genes: Maximum number of genes per cell (filters potential doublets)
            max_mt_pct: Maximum percentage of mitochondrial genes
            max_ribo_pct: Maximum percentage of ribosomal genes
            min_housekeeping_score: Minimum housekeeping gene score

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - AnnData with QC metrics
                - Assessment statistics dictionary
                - AnalysisStep IR for notebook export

        Raises:
            QualityError: If quality assessment fails
        """
        try:
            logger.info("Starting quality assessment")

            # Create working copy
            adata_qc = adata.copy()

            # Ensure unique variable names for gene subsetting operations
            if not adata_qc.var_names.is_unique:
                logger.warning("Non-unique variable names detected. Making unique.")
                adata_qc.var_names_make_unique()

            # Calculate QC metrics from expression matrix
            qc_metrics = self._calculate_qc_metrics_from_adata(adata_qc)

            # Add QC metrics to observations
            adata_qc.obs["mt_pct"] = qc_metrics["mt_pct"]
            adata_qc.obs["ribo_pct"] = qc_metrics["ribo_pct"]
            adata_qc.obs["housekeeping_score"] = qc_metrics["housekeeping_score"]

            # Filter cells based on QC metrics (with upper bound for doublets)
            passing_cells = (
                (qc_metrics["n_genes"] >= min_genes)
                & (qc_metrics["n_genes"] <= max_genes)
                & (qc_metrics["mt_pct"] <= max_mt_pct)
                & (qc_metrics["ribo_pct"] <= max_ribo_pct)
                & (qc_metrics["housekeeping_score"] >= min_housekeeping_score)
            )

            # Add QC pass/fail flag to observations
            adata_qc.obs["qc_pass"] = passing_cells

            cells_before = len(qc_metrics)
            cells_after = passing_cells.sum()

            # Generate summary
            summary = self._generate_qc_summary(qc_metrics)

            # Compile assessment statistics
            assessment_stats = {
                "analysis_type": "quality_assessment",
                "min_genes": min_genes,
                "max_genes": max_genes,
                "max_mt_pct": max_mt_pct,
                "max_ribo_pct": max_ribo_pct,
                "min_housekeeping_score": min_housekeeping_score,
                "cells_before_qc": cells_before,
                "cells_after_qc": cells_after,
                "cells_removed": cells_before - cells_after,
                "cells_retained_pct": (cells_after / cells_before) * 100,
                "quality_status": (
                    "Pass" if cells_after / cells_before > 0.7 else "Warning"
                ),
                "mean_total_counts": float(qc_metrics["total_counts"].mean()),
                "mean_genes_per_cell": float(qc_metrics["n_genes"].mean()),
                "mean_mt_pct": float(qc_metrics["mt_pct"].mean()),
                "mean_ribo_pct": float(qc_metrics["ribo_pct"].mean()),
                "mean_housekeeping_score": float(
                    qc_metrics["housekeeping_score"].mean()
                ),
                "qc_summary": summary,
                "mt_stats": {
                    "min": float(qc_metrics["mt_pct"].min()),
                    "max": float(qc_metrics["mt_pct"].max()),
                    "mean": float(qc_metrics["mt_pct"].mean()),
                    "std": float(qc_metrics["mt_pct"].std()),
                },
                "ribo_stats": {
                    "min": float(qc_metrics["ribo_pct"].min()),
                    "max": float(qc_metrics["ribo_pct"].max()),
                    "mean": float(qc_metrics["ribo_pct"].mean()),
                    "std": float(qc_metrics["ribo_pct"].std()),
                },
                "housekeeping_stats": {
                    "min": float(qc_metrics["housekeeping_score"].min()),
                    "max": float(qc_metrics["housekeeping_score"].max()),
                    "mean": float(qc_metrics["housekeeping_score"].mean()),
                    "std": float(qc_metrics["housekeeping_score"].std()),
                },
            }

            logger.info(
                f"Quality assessment completed: {cells_after}/{cells_before} cells pass QC ({assessment_stats['cells_retained_pct']:.1f}%)"
            )

            # Create IR for notebook export
            ir = self._create_quality_ir(
                min_genes=min_genes,
                max_genes=max_genes,
                max_mt_pct=max_mt_pct,
                max_ribo_pct=max_ribo_pct,
                min_housekeeping_score=min_housekeeping_score,
            )

            return adata_qc, assessment_stats, ir

        except Exception as e:
            logger.exception(f"Error in quality assessment: {e}")
            raise QualityError(f"Quality assessment failed: {str(e)}")

    def suggest_adaptive_thresholds(
        self,
        adata: anndata.AnnData,
        n_mads: float = 3.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Suggest adaptive quality control thresholds using MAD-based outlier detection.

        Uses Median Absolute Deviation (MAD) to identify outliers:
        threshold = median ± (n_mads * MAD)

        This approach is more robust than fixed thresholds for heterogeneous datasets
        with varying cell types, sequencing depths, or tissue types.

        Args:
            adata: AnnData object to analyze
            n_mads: Number of MADs from median for outlier detection (default: 3.0)
                   Higher values = more permissive, lower values = more stringent

        Returns:
            Dict with suggested thresholds:
            {
                "n_genes": {"lower": float, "upper": float, "median": float, "mad": float},
                "total_counts": {"lower": float, "upper": float, "median": float, "mad": float},
                "mt_pct": {"upper": float, "median": float, "mad": float}
            }

        Raises:
            QualityError: If threshold calculation fails
        """
        try:
            logger.info(f"Calculating adaptive thresholds with {n_mads} MADs")

            # Calculate QC metrics first if not already present
            if "n_genes" not in adata.obs.columns:
                qc_metrics = self._calculate_qc_metrics_from_adata(adata)
            else:
                qc_metrics = adata.obs[["n_genes", "total_counts", "mt_pct"]].copy()
                # If columns don't exist, calculate them
                if "n_genes" not in qc_metrics.columns:
                    qc_metrics = self._calculate_qc_metrics_from_adata(adata)

            def calculate_mad_bounds(values: pd.Series) -> Dict[str, float]:
                """Calculate median, MAD, and bounds for a metric."""
                median = values.median()
                mad = np.median(np.abs(values - median))
                return {
                    "median": float(median),
                    "mad": float(mad),
                    "lower": float(median - n_mads * mad),
                    "upper": float(median + n_mads * mad),
                }

            # Calculate adaptive thresholds
            suggestions = {
                "n_genes": calculate_mad_bounds(qc_metrics["n_genes"]),
                "total_counts": calculate_mad_bounds(qc_metrics["total_counts"]),
                "mt_pct": {
                    "median": float(qc_metrics["mt_pct"].median()),
                    "mad": float(
                        np.median(
                            np.abs(qc_metrics["mt_pct"] - qc_metrics["mt_pct"].median())
                        )
                    ),
                    "upper": float(
                        qc_metrics["mt_pct"].median()
                        + n_mads
                        * np.median(
                            np.abs(qc_metrics["mt_pct"] - qc_metrics["mt_pct"].median())
                        )
                    ),
                },
            }

            # Ensure sensible bounds (no negative values)
            for metric in ["n_genes", "total_counts"]:
                suggestions[metric]["lower"] = max(0, suggestions[metric]["lower"])

            suggestions["mt_pct"]["upper"] = min(100, suggestions["mt_pct"]["upper"])

            logger.info(
                f"Adaptive thresholds calculated: "
                f"n_genes={suggestions['n_genes']['lower']:.0f}-{suggestions['n_genes']['upper']:.0f}, "
                f"mt_pct<={suggestions['mt_pct']['upper']:.1f}%"
            )

            return suggestions

        except Exception as e:
            logger.exception(f"Error calculating adaptive thresholds: {e}")
            raise QualityError(f"Adaptive threshold calculation failed: {str(e)}")

    def _detect_mitochondrial_genes(self, adata: anndata.AnnData) -> np.ndarray:
        """
        Detect mitochondrial genes using multiple nomenclature patterns.

        Tries patterns in order of specificity:
        1. Human HGNC: MT-* (dash)
        2. Mouse MGI: mt-* (lowercase)
        3. Alternative: MT.* (dot delimiter)
        4. Ensembl: ENSG00000198*, ENSG00000210* (known MT gene ID ranges)
        5. Generic: contains "mito" or "mitochondr"

        Scientific Note:
        ----------------
        Mitochondrial percentage is a critical QC metric for single-cell RNA-seq.
        Dying cells leak cytoplasmic RNA, increasing relative MT content.
        Standard thresholds: 15-20% MT for healthy cells, >50% indicates dying cells.

        Args:
            adata: AnnData object

        Returns:
            Boolean array indicating mitochondrial genes
        """
        var_names = adata.var_names.str

        # Pattern 1: HGNC (MT-)
        mt_mask = var_names.startswith("MT-")
        n_found = mt_mask.sum()
        if n_found > 0:
            logger.info(
                f"Detected {n_found} mitochondrial genes using HGNC pattern (MT-)"
            )
            return mt_mask

        # Pattern 2: Mouse (mt- lowercase)
        mt_mask = var_names.lower().str.startswith("mt-")
        n_found = mt_mask.sum()
        if n_found > 0:
            logger.info(
                f"Detected {n_found} mitochondrial genes using mouse pattern (mt-)"
            )
            return mt_mask

        # Pattern 3: Alternative delimiter (MT.)
        mt_mask = var_names.startswith("MT.")
        n_found = mt_mask.sum()
        if n_found > 0:
            logger.info(
                f"Detected {n_found} mitochondrial genes using alternative pattern (MT.)"
            )
            return mt_mask

        # Pattern 4: Ensembl IDs (known MT genome ranges)
        # Human: ENSG00000198* (ENSG00000198888-ENSG00000198938)
        # Also check ENSG00000210* range
        ensembl_mt_prefixes = ["ENSG00000198", "ENSG00000210"]
        mt_mask = var_names.startswith(tuple(ensembl_mt_prefixes))
        n_found = mt_mask.sum()
        if n_found > 0:
            logger.info(
                f"Detected {n_found} mitochondrial genes using Ensembl ID pattern"
            )
            return mt_mask

        # Pattern 5: Generic fallback (contains "mito")
        mt_mask = var_names.lower().str.contains(
            "mito|mitochondr", regex=True, na=False
        )
        n_found = mt_mask.sum()
        if n_found > 0:
            logger.warning(
                f"Detected {n_found} mitochondrial genes using generic fallback pattern "
                "(may include false positives)"
            )
            return mt_mask

        # No MT genes found
        logger.warning(
            "No mitochondrial genes detected using any pattern. "
            "Mitochondrial QC metrics will be 0.0% (may be inaccurate)."
        )
        return np.zeros(len(adata.var_names), dtype=bool)

    def _detect_ribosomal_genes(self, adata: anndata.AnnData) -> np.ndarray:
        """
        Detect ribosomal genes using multiple nomenclature patterns.

        Tries patterns in order of specificity:
        1. Human HGNC: RPS*, RPL* (uppercase)
        2. Mouse MGI: Rps*, Rpl* (capitalized)
        3. Generic: rps*, rpl* (lowercase)
        4. Alternative: RP[SL]* (compact notation)
        5. Fallback: contains "ribosom"

        Scientific Note:
        ----------------
        High ribosomal content (>50%) may indicate:
        - Metabolic stress
        - Low-quality libraries
        - Actively proliferating cells (context-dependent)

        Args:
            adata: AnnData object

        Returns:
            Boolean array indicating ribosomal genes
        """
        var_names = adata.var_names.str

        # Pattern 1: HGNC (RPS*, RPL*)
        ribo_mask = var_names.startswith("RPS") | var_names.startswith("RPL")
        n_found = ribo_mask.sum()
        if n_found > 0:
            logger.info(
                f"Detected {n_found} ribosomal genes using HGNC pattern (RPS/RPL)"
            )
            return ribo_mask

        # Pattern 2: Mouse (Rps*, Rpl* - capitalized)
        ribo_mask = var_names.startswith("Rps") | var_names.startswith("Rpl")
        n_found = ribo_mask.sum()
        if n_found > 0:
            logger.info(
                f"Detected {n_found} ribosomal genes using mouse pattern (Rps/Rpl)"
            )
            return ribo_mask

        # Pattern 3: Generic lowercase (rps*, rpl*)
        ribo_mask = var_names.lower().str.startswith(
            "rps"
        ) | var_names.lower().str.startswith("rpl")
        n_found = ribo_mask.sum()
        if n_found > 0:
            logger.info(
                f"Detected {n_found} ribosomal genes using lowercase pattern (rps/rpl)"
            )
            return ribo_mask

        # Pattern 4: Compact notation (RP[SL]*)
        ribo_mask = var_names.match(r"^RP[SL]\d+", na=False)
        n_found = ribo_mask.sum()
        if n_found > 0:
            logger.info(
                f"Detected {n_found} ribosomal genes using compact pattern (RP[SL])"
            )
            return ribo_mask

        # Pattern 5: Generic fallback (contains "ribosom")
        ribo_mask = var_names.lower().str.contains("ribosom", regex=False, na=False)
        n_found = ribo_mask.sum()
        if n_found > 0:
            logger.warning(
                f"Detected {n_found} ribosomal genes using generic fallback pattern "
                "(may include false positives)"
            )
            return ribo_mask

        # No ribosomal genes found
        logger.warning(
            "No ribosomal genes detected using any pattern. "
            "Ribosomal QC metrics will be 0.0% (may be inaccurate)."
        )
        return np.zeros(len(adata.var_names), dtype=bool)

    def _calculate_qc_metrics_from_adata(self, adata: anndata.AnnData) -> pd.DataFrame:
        """
        Calculate quality control metrics from AnnData object.

        Args:
            adata: AnnData object

        Returns:
            DataFrame: DataFrame with QC metrics
        """
        logger.info("Calculating quality metrics from AnnData")

        # Identify mitochondrial genes using multi-pattern cascade
        mt_genes = self._detect_mitochondrial_genes(adata)

        # Identify ribosomal genes using multi-pattern cascade
        ribo_genes = self._detect_ribosomal_genes(adata)

        # Identify housekeeping genes for score
        housekeeping_genes = ["ACTB", "GAPDH", "MALAT1"]

        # Calculate basic metrics
        if hasattr(adata.X, "toarray"):
            # Sparse matrix
            total_counts = np.array(adata.X.sum(axis=1)).flatten()
            n_genes = np.array((adata.X > 0).sum(axis=1)).flatten()
        else:
            # Dense matrix
            total_counts = adata.X.sum(axis=1)
            n_genes = (adata.X > 0).sum(axis=1)

        # Calculate mitochondrial percentage
        if mt_genes.sum() > 0:
            if hasattr(adata.X, "toarray"):
                mt_counts = np.array(adata[:, mt_genes].X.sum(axis=1)).flatten()
            else:
                mt_counts = adata[:, mt_genes].X.sum(axis=1)
            mt_pct = (
                mt_counts / (total_counts + 1e-8)
            ) * 100  # Add small epsilon to avoid division by zero
        else:
            logger.warning(
                "No mitochondrial genes found. Setting mitochondrial percentage to 0."
            )
            mt_pct = np.zeros(adata.n_obs)

        # Calculate ribosomal percentage
        if ribo_genes.sum() > 0:
            if hasattr(adata.X, "toarray"):
                ribo_counts = np.array(adata[:, ribo_genes].X.sum(axis=1)).flatten()
            else:
                ribo_counts = adata[:, ribo_genes].X.sum(axis=1)
            ribo_pct = (ribo_counts / (total_counts + 1e-8)) * 100
        else:
            logger.warning(
                "No ribosomal genes found. Setting ribosomal percentage to 0."
            )
            ribo_pct = np.zeros(adata.n_obs)

        # Calculate housekeeping gene score
        available_hk_genes = [
            gene for gene in housekeeping_genes if gene in adata.var_names
        ]
        if available_hk_genes:
            if hasattr(adata.X, "toarray"):
                housekeeping_score = np.array(
                    adata[:, available_hk_genes].X.sum(axis=1)
                ).flatten()
            else:
                housekeeping_score = adata[:, available_hk_genes].X.sum(axis=1)
        else:
            logger.warning(
                "No housekeeping genes found. Setting housekeeping score to 0."
            )
            housekeeping_score = np.zeros(adata.n_obs)

        # Combine metrics into a DataFrame
        qc_df = pd.DataFrame(
            {
                "total_counts": total_counts,
                "n_genes": n_genes,
                "mt_pct": mt_pct,
                "ribo_pct": ribo_pct,
                "housekeeping_score": housekeeping_score,
            },
            index=adata.obs_names,
        )

        logger.info(f"Generated QC metrics for {len(qc_df)} cells")
        return qc_df

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
        Create high-quality assessment plots.

        Args:
            qc_metrics: DataFrame with QC metrics

        Returns:
            list: List of Plotly figures
        """
        logger.info("Generating high-quality assessment plots")
        plots = []

        # 1. Enhanced violin plot of mitochondrial percentages
        fig1 = go.Figure()
        fig1.add_trace(
            go.Violin(
                y=qc_metrics["mt_pct"],
                name="Mitochondrial %",
                box_visible=True,
                meanline_visible=True,
                fillcolor="rgba(56, 108, 176, 0.7)",
                line_color="rgba(56, 108, 176, 1.0)",
                points="outliers",
                pointpos=0,
                jitter=0.05,
                side="positive",
            )
        )
        fig1.update_layout(
            title=dict(
                text="Distribution of Mitochondrial Gene Percentages",
                font=dict(size=20, family="Arial, sans-serif"),
                x=0.5,
                xanchor="center",
            ),
            yaxis=dict(
                title=dict(text="Mitochondrial Gene %", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            xaxis=dict(tickfont=dict(size=14), showgrid=False),
            font=dict(size=14, family="Arial, sans-serif"),
            height=600,  # Increased from 400
            width=1000,  # Added width
            margin=dict(l=80, r=80, t=80, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )
        plots.append(fig1)

        # 2. Enhanced violin plot of ribosomal percentages
        fig1b = go.Figure()
        fig1b.add_trace(
            go.Violin(
                y=qc_metrics["ribo_pct"],
                name="Ribosomal %",
                box_visible=True,
                meanline_visible=True,
                fillcolor="rgba(127, 201, 127, 0.7)",
                line_color="rgba(127, 201, 127, 1.0)",
                points="outliers",
                pointpos=0,
                jitter=0.05,
                side="positive",
            )
        )
        fig1b.update_layout(
            title=dict(
                text="Distribution of Ribosomal Gene Percentages",
                font=dict(size=20, family="Arial, sans-serif"),
                x=0.5,
                xanchor="center",
            ),
            yaxis=dict(
                title=dict(text="Ribosomal Gene %", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            xaxis=dict(tickfont=dict(size=14), showgrid=False),
            font=dict(size=14, family="Arial, sans-serif"),
            height=600,  # Increased from 400
            width=1000,  # Added width
            margin=dict(l=80, r=80, t=80, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )
        plots.append(fig1b)

        # 3. Enhanced violin plot of housekeeping scores
        fig1c = go.Figure()
        fig1c.add_trace(
            go.Violin(
                y=qc_metrics["housekeeping_score"],
                name="Housekeeping Score",
                box_visible=True,
                meanline_visible=True,
                fillcolor="rgba(190, 174, 212, 0.7)",
                line_color="rgba(190, 174, 212, 1.0)",
                points="outliers",
                pointpos=0,
                jitter=0.05,
                side="positive",
            )
        )
        fig1c.update_layout(
            title=dict(
                text="Distribution of Housekeeping Gene Scores",
                font=dict(size=20, family="Arial, sans-serif"),
                x=0.5,
                xanchor="center",
            ),
            yaxis=dict(
                title=dict(text="Housekeeping Score", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            xaxis=dict(tickfont=dict(size=14), showgrid=False),
            font=dict(size=14, family="Arial, sans-serif"),
            height=600,  # Increased from 400
            width=1000,  # Added width
            margin=dict(l=80, r=80, t=80, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )
        plots.append(fig1c)

        # 4. Enhanced scatter plot of quality metrics
        fig2 = px.scatter(
            qc_metrics,
            x="total_counts",
            y="n_genes",
            color="mt_pct",
            title="Cell Quality Metrics - Mitochondrial Content",
            labels={
                "total_counts": "Total RNA Count",
                "n_genes": "Number of Detected Features",
                "mt_pct": "Mitochondrial %",
            },
            color_continuous_scale="plasma",
            height=700,  # Increased from 450
            width=1200,  # Increased from 600
        )

        fig2.update_traces(
            marker=dict(
                size=6, opacity=0.7, line=dict(width=0.5, color="rgba(50,50,50,0.4)")
            )
        )

        fig2.update_layout(
            title=dict(
                text="Cell Quality Metrics - Mitochondrial Content",
                font=dict(size=20, family="Arial, sans-serif"),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title=dict(text="Total RNA Count", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                title=dict(text="Number of Detected Features", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            font=dict(size=14, family="Arial, sans-serif"),
            margin=dict(l=80, r=150, t=80, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
            coloraxis_colorbar=dict(
                title=dict(text="Mitochondrial %", font=dict(size=14)),
                tickfont=dict(size=12),
            ),
        )
        plots.append(fig2)

        # 5. Enhanced scatter plot showing ribosomal percentage
        fig2b = px.scatter(
            qc_metrics,
            x="total_counts",
            y="n_genes",
            color="ribo_pct",
            title="Cell Quality Metrics - Ribosomal Content",
            labels={
                "total_counts": "Total RNA Count",
                "n_genes": "Number of Detected Features",
                "ribo_pct": "Ribosomal %",
            },
            color_continuous_scale="viridis",
            height=700,  # Increased from 450
            width=1200,  # Increased from 600
        )

        fig2b.update_traces(
            marker=dict(
                size=6, opacity=0.7, line=dict(width=0.5, color="rgba(50,50,50,0.4)")
            )
        )

        fig2b.update_layout(
            title=dict(
                text="Cell Quality Metrics - Ribosomal Content",
                font=dict(size=20, family="Arial, sans-serif"),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title=dict(text="Total RNA Count", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                title=dict(text="Number of Detected Features", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            font=dict(size=14, family="Arial, sans-serif"),
            margin=dict(l=80, r=150, t=80, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
            coloraxis_colorbar=dict(
                title=dict(text="Ribosomal %", font=dict(size=14)),
                tickfont=dict(size=12),
            ),
        )
        plots.append(fig2b)

        # 6. Enhanced correlation plot
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
            height=700,  # Increased from 400
            width=1100,  # Increased from 550
        )

        fig3.update_traces(
            marker=dict(
                size=6,
                opacity=0.6,
                color="rgba(31, 119, 180, 0.7)",
                line=dict(width=0.5, color="rgba(50,50,50,0.4)"),
            )
        )

        fig3.update_layout(
            title=dict(
                text="Correlation between Features and RNA Count",
                font=dict(size=20, family="Arial, sans-serif"),
                x=0.5,
                xanchor="center",
            ),
            xaxis=dict(
                title=dict(text="Total RNA Count", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            yaxis=dict(
                title=dict(text="Number of Detected Features", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor="rgba(200,200,200,0.3)",
            ),
            font=dict(size=14, family="Arial, sans-serif"),
            margin=dict(l=80, r=80, t=80, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        plots.append(fig3)

        logger.info(f"Created {len(plots)} high-quality assessment plots")
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

    def _create_quality_ir(
        self,
        min_genes: int,
        max_genes: int,
        max_mt_pct: float,
        max_ribo_pct: float,
        min_housekeeping_score: float,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for quality assessment.

        This IR enables automatic notebook generation without manual mapping.

        Args:
            min_genes: Minimum genes per cell threshold
            max_genes: Maximum genes per cell threshold (doublet filtering)
            max_mt_pct: Maximum mitochondrial percentage
            max_ribo_pct: Maximum ribosomal percentage
            min_housekeeping_score: Minimum housekeeping gene score

        Returns:
            AnalysisStep with complete code generation instructions
        """
        # Create parameter schema with Papermill flags
        parameter_schema = {
            "min_genes": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=500,
                required=False,
                validation_rule="min_genes > 0",
                description="Minimum genes per cell for QC pass",
            ),
            "max_genes": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=5000,
                required=False,
                validation_rule="max_genes > 0 and max_genes > min_genes",
                description="Maximum genes per cell (filters potential doublets)",
            ),
            "max_mt_pct": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=20.0,
                required=False,
                validation_rule="max_mt_pct > 0 and max_mt_pct <= 100",
                description="Maximum mitochondrial percentage threshold",
            ),
            "max_ribo_pct": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=50.0,
                required=False,
                validation_rule="max_ribo_pct > 0 and max_ribo_pct <= 100",
                description="Maximum ribosomal percentage threshold",
            ),
            "min_housekeeping_score": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=1.0,
                required=False,
                validation_rule="min_housekeeping_score >= 0",
                description="Minimum housekeeping gene expression score",
            ),
        }

        # Jinja2 template with parameter placeholders
        code_template = """# Calculate QC metrics
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=['mt', 'ribo'],
    percent_top=None,
    log1p=False,
    inplace=True
)

# Add QC pass/fail flags (with upper bound for doublet filtering)
adata.obs['qc_pass'] = (
    (adata.obs['n_genes_by_counts'] >= {{ min_genes }}) &
    (adata.obs['n_genes_by_counts'] <= {{ max_genes }}) &
    (adata.obs['pct_counts_mt'] <= {{ max_mt_pct }}) &
    (adata.obs['pct_counts_ribo'] <= {{ max_ribo_pct }})
)

# Display QC summary
print(f"Cells before QC: {adata.n_obs}")
print(f"Cells passing QC: {adata.obs['qc_pass'].sum()}")
print(f"Cells removed: {(~adata.obs['qc_pass']).sum()}")
print(f"Mean genes per cell: {adata.obs['n_genes_by_counts'].mean():.0f}")
print(f"Mean mitochondrial %: {adata.obs['pct_counts_mt'].mean():.2f}")
"""

        # Create AnalysisStep
        ir = AnalysisStep(
            operation="scanpy.pp.calculate_qc_metrics",
            tool_name="assess_quality",
            description="Calculate quality control metrics and filter cells with doublet detection",
            library="scanpy",
            code_template=code_template,
            imports=["import scanpy as sc", "import numpy as np"],
            parameters={
                "min_genes": min_genes,
                "max_genes": max_genes,
                "max_mt_pct": max_mt_pct,
                "max_ribo_pct": max_ribo_pct,
                "min_housekeeping_score": min_housekeeping_score,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "qc_vars": ["mt", "ribo"],
                "method": "scanpy",
            },
            validates_on_export=True,
            requires_validation=False,
        )

        logger.debug(f"Created IR for quality assessment: {ir.operation}")
        return ir
