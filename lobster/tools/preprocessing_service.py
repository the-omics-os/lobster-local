"""
Single-cell RNA-seq preprocessing service for advanced data cleaning and normalization.

This service implements professional-grade preprocessing methods including ambient RNA
correction, cell filtering, normalization, and batch correction/integration.
"""

from typing import Any, Dict, Tuple, Union

import anndata
import numpy as np
import plotly.graph_objects as go
import scanpy as sc
import scipy.sparse as spr
from plotly.subplots import make_subplots

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.deviance import calculate_deviance
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingError(Exception):
    """Base exception for preprocessing operations."""

    pass


class PreprocessingService:
    """
    Advanced preprocessing service for single-cell RNA-seq data.

    This stateless service provides methods for ambient RNA correction, quality control filtering,
    normalization, and batch correction/integration following best practices from
    current single-cell analysis pipelines.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the preprocessing service.

        Args:
            config: Optional configuration dict (ignored, for backward compatibility)
            **kwargs: Additional arguments (ignored, for backward compatibility)

        This service is stateless and doesn't require a data manager instance.
        """
        logger.debug("Initializing stateless PreprocessingService")
        self.config = config or {}
        logger.debug("PreprocessingService initialized successfully")

    def correct_ambient_rna(
        self,
        adata: anndata.AnnData,
        contamination_fraction: float = 0.1,
        empty_droplet_threshold: int = 100,
        method: str = "simple_decontamination",
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Correct for ambient RNA contamination using simplified decontamination methods.

        This implements a simplified version of methods like SoupX, estimating and removing
        ambient RNA signal that can contaminate cell-containing droplets.

        Args:
            adata: AnnData object with raw UMI counts
            contamination_fraction: Expected fraction of ambient RNA (0.05-0.2 typical)
            empty_droplet_threshold: Minimum UMI count to consider droplet as cell-containing
            method: Method to use ('simple_decontamination', 'quantile_based')

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: Corrected AnnData and processing stats

        Raises:
            PreprocessingError: If correction fails
        """
        try:
            logger.info(f"Starting ambient RNA correction with method: {method}")

            # Create working copy
            adata_corrected = adata.copy()
            original_shape = adata_corrected.shape
            logger.info(
                f"Input data shape: {original_shape[0]} cells × {original_shape[1]} genes"
            )

            # Store original data for comparison
            if adata_corrected.raw is None:
                adata_corrected.raw = adata_corrected.copy()

            # Implement ambient RNA correction
            if method == "simple_decontamination":
                corrected_matrix = self._simple_decontamination(
                    adata_corrected.X, contamination_fraction, empty_droplet_threshold
                )
            elif method == "quantile_based":
                corrected_matrix = self._quantile_based_correction(
                    adata_corrected.X, contamination_fraction
                )
            else:
                raise PreprocessingError(
                    f"Unknown ambient RNA correction method: {method}"
                )

            # Update the data
            adata_corrected.X = corrected_matrix

            # Calculate correction statistics
            original_total = np.sum(adata_corrected.raw.X)
            corrected_total = np.sum(adata_corrected.X)
            reduction_fraction = (original_total - corrected_total) / original_total

            # Store processing metadata
            processing_stats = {
                "method": method,
                "contamination_fraction": contamination_fraction,
                "empty_droplet_threshold": empty_droplet_threshold,
                "original_total_umis": float(original_total),
                "corrected_total_umis": float(corrected_total),
                "umi_reduction_fraction": float(reduction_fraction),
                "cells_processed": adata_corrected.n_obs,
                "genes_processed": adata_corrected.n_vars,
                "analysis_type": "ambient_rna_correction",
            }

            logger.info(
                f"Ambient RNA correction completed: {reduction_fraction:.1%} UMI reduction"
            )
            return adata_corrected, processing_stats

        except Exception as e:
            logger.exception(f"Error in ambient RNA correction: {e}")
            raise PreprocessingError(f"Ambient RNA correction failed: {str(e)}")

    def _create_filter_normalize_ir(
        self,
        min_genes_per_cell: int,
        max_genes_per_cell: int,
        min_cells_per_gene: int,
        max_mito_percent: float,
        max_ribo_percent: float,
        normalization_method: str,
        target_sum: int,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for filter and normalization operation.

        Args:
            min_genes_per_cell: Minimum genes per cell threshold
            max_genes_per_cell: Maximum genes per cell threshold
            min_cells_per_gene: Minimum cells per gene threshold
            max_mito_percent: Maximum mitochondrial percentage
            max_ribo_percent: Maximum ribosomal percentage
            normalization_method: Normalization method to use
            target_sum: Target sum for normalization

        Returns:
            AnalysisStep with filter and normalize code template
        """
        # Parameter schema
        parameter_schema = {
            "min_genes_per_cell": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=min_genes_per_cell,
                required=False,
                validation_rule="min_genes_per_cell > 0",
                description="Minimum number of genes per cell",
            ),
            "max_genes_per_cell": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=max_genes_per_cell,
                required=False,
                validation_rule="max_genes_per_cell > min_genes_per_cell",
                description="Maximum number of genes per cell",
            ),
            "min_cells_per_gene": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=min_cells_per_gene,
                required=False,
                validation_rule="min_cells_per_gene > 0",
                description="Minimum number of cells per gene",
            ),
            "max_mito_percent": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=max_mito_percent,
                required=False,
                validation_rule="0 <= max_mito_percent <= 100",
                description="Maximum percentage of mitochondrial genes",
            ),
            "max_ribo_percent": ParameterSpec(
                param_type="float",
                papermill_injectable=True,
                default_value=max_ribo_percent,
                required=False,
                validation_rule="0 <= max_ribo_percent <= 100",
                description="Maximum percentage of ribosomal genes",
            ),
            "target_sum": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=target_sum,
                required=False,
                validation_rule="target_sum > 0",
                description="Target sum for normalization",
            ),
        }

        # Jinja2 template with parameters
        code_template = """# Filter cells and genes based on quality metrics
# Filtering thresholds:
# - Genes per cell: {{ min_genes_per_cell }} to {{ max_genes_per_cell }}
# - Cells per gene: minimum {{ min_cells_per_gene }}
# - Mitochondrial content: maximum {{ max_mito_percent }}%
# - Ribosomal content: maximum {{ max_ribo_percent }}%

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)

# Filter cells
adata = adata[adata.obs['n_genes_by_counts'] >= {{ min_genes_per_cell }}, :].copy()
adata = adata[adata.obs['n_genes_by_counts'] <= {{ max_genes_per_cell }}, :].copy()
adata = adata[adata.obs['pct_counts_mt'] <= {{ max_mito_percent }}, :].copy()
adata = adata[adata.obs['pct_counts_ribo'] <= {{ max_ribo_percent }}, :].copy()

# Filter genes
sc.pp.filter_genes(adata, min_cells={{ min_cells_per_gene }})

print(f"After filtering: {adata.n_obs} cells × {adata.n_vars} genes")

# Normalize expression data
sc.pp.normalize_total(adata, target_sum={{ target_sum }})
sc.pp.log1p(adata)

print(f"Normalization complete (target_sum={{ target_sum }}, log1p transformed)")
"""

        return AnalysisStep(
            operation="scanpy.pp.filter_normalize",
            tool_name="filter_and_normalize_cells",
            description=f"Filter cells/genes and normalize expression data (target_sum={target_sum})",
            library="scanpy",
            code_template=code_template,
            imports=["import scanpy as sc", "import numpy as np"],
            parameters={
                "min_genes_per_cell": min_genes_per_cell,
                "max_genes_per_cell": max_genes_per_cell,
                "min_cells_per_gene": min_cells_per_gene,
                "max_mito_percent": max_mito_percent,
                "max_ribo_percent": max_ribo_percent,
                "target_sum": target_sum,
                "normalization_method": normalization_method,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "preprocessing",
                "normalization_method": normalization_method,
                "filtering_strategy": "qc_based",
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def filter_and_normalize_cells(
        self,
        adata: anndata.AnnData,
        min_genes_per_cell: int = 200,
        max_genes_per_cell: int = 5000,
        min_cells_per_gene: int = 3,
        max_mito_percent: float = 20.0,
        max_ribo_percent: float = 50.0,
        normalization_method: str = "log1p",
        target_sum: int = 10000,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Filter cells and genes based on quality metrics and normalize expression data.

        Args:
            adata: AnnData object to filter and normalize
            min_genes_per_cell: Minimum number of genes expressed per cell
            max_genes_per_cell: Maximum number of genes expressed per cell
            min_cells_per_gene: Minimum number of cells expressing each gene
            max_mito_percent: Maximum mitochondrial gene percentage
            max_ribo_percent: Maximum ribosomal gene percentage
            normalization_method: Normalization method ('log1p', 'sctransform_like')
            target_sum: Target sum for normalization (e.g., 10,000)

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: Filtered/normalized AnnData, processing stats, and IR

        Raises:
            PreprocessingError: If filtering or normalization fails
        """
        try:
            logger.info("Starting cell filtering and normalization")

            # Create working copy
            adata_processed = adata.copy()
            original_shape = adata_processed.shape
            logger.info(
                f"Input data shape: {original_shape[0]} cells × {original_shape[1]} genes"
            )

            # Calculate QC metrics
            self._calculate_qc_metrics(adata_processed)

            # Store pre-filtering metrics
            pre_filter_metrics = {
                "n_cells": adata_processed.n_obs,
                "n_genes": adata_processed.n_vars,
                "median_genes_per_cell": float(
                    np.median(adata_processed.obs["n_genes_by_counts"])
                ),
                "median_counts_per_cell": float(
                    np.median(adata_processed.obs["total_counts"])
                ),
            }

            # Apply quality filters
            filtering_stats = self._apply_quality_filters(
                adata_processed,
                min_genes_per_cell,
                max_genes_per_cell,
                min_cells_per_gene,
                max_mito_percent,
                max_ribo_percent,
            )

            # Normalize the data
            normalization_stats = self._normalize_expression_data(
                adata_processed, normalization_method, target_sum
            )

            # Combine all processing statistics
            processing_stats = {
                "min_genes_per_cell": min_genes_per_cell,
                "max_genes_per_cell": max_genes_per_cell,
                "min_cells_per_gene": min_cells_per_gene,
                "max_mito_percent": max_mito_percent,
                "max_ribo_percent": max_ribo_percent,
                "normalization_method": normalization_method,
                "target_sum": target_sum,
                "analysis_type": "filter_and_normalize",
                "original_shape": original_shape,
                "final_shape": adata_processed.shape,
                **pre_filter_metrics,
                **filtering_stats,
                **normalization_stats,
            }

            logger.info(
                f"Filtering and normalization completed: {original_shape[0]} → {adata_processed.n_obs} cells, "
                f"{original_shape[1]} → {adata_processed.n_vars} genes"
            )

            # Create IR for notebook export
            ir = self._create_filter_normalize_ir(
                min_genes_per_cell=min_genes_per_cell,
                max_genes_per_cell=max_genes_per_cell,
                min_cells_per_gene=min_cells_per_gene,
                max_mito_percent=max_mito_percent,
                max_ribo_percent=max_ribo_percent,
                normalization_method=normalization_method,
                target_sum=target_sum,
            )

            return adata_processed, processing_stats, ir

        except Exception as e:
            logger.exception(f"Error in filtering and normalization: {e}")
            raise PreprocessingError(f"Filtering and normalization failed: {str(e)}")

    def select_features_deviance(
        self,
        adata: anndata.AnnData,
        n_top_genes: int = 4000,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Select highly deviant genes using binomial deviance from multinomial null model.

        This method implements deviance-based feature selection following scry best practices.
        Unlike traditional highly variable genes (HVG) methods that use log1p transformation
        with arbitrary pseudo-counts, this approach:
        - Works directly on raw counts (no normalization bias)
        - Uses mathematically principled binomial deviance from multinomial null model
        - Requires no arbitrary pseudo-count choices
        - Provides closed-form computation

        Recommended by:
        - Townes et al. (2019): Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model
        - scry package documentation (Bioconductor)

        Args:
            adata: AnnData object (preferably with raw counts in adata.raw or adata.X)
            n_top_genes: Number of top deviant genes to select (default: 4000)

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - AnnData with 'highly_deviant' column in .var
                - Statistics dictionary with feature selection metrics
                - AnalysisStep IR for notebook export

        Raises:
            PreprocessingError: If feature selection fails

        Example:
            >>> service = PreprocessingService()
            >>> adata_selected, stats, ir = service.select_features_deviance(adata, n_top_genes=4000)
            >>> print(f"Selected {stats['n_features_selected']} highly deviant genes")
        """
        try:
            logger.info(
                f"Starting deviance-based feature selection (n_top_genes={n_top_genes})"
            )

            # Create working copy
            adata_processed = adata.copy()
            original_n_genes = adata_processed.n_vars

            # Use raw counts if available (preferred), otherwise use current X
            if adata_processed.raw is not None:
                logger.info("Using raw counts from adata.raw for deviance calculation")
                count_data = adata_processed.raw.X
                gene_names = adata_processed.raw.var_names
            else:
                logger.info("Using adata.X for deviance calculation (no adata.raw found)")
                count_data = adata_processed.X
                gene_names = adata_processed.var_names

            # Ensure we have count data (not log-transformed)
            # Check if data looks like log-transformed (all values < 15 is suspicious)
            if hasattr(count_data, "toarray"):
                max_val = count_data.max()
            else:
                max_val = count_data.max()

            if max_val < 15:
                logger.warning(
                    f"Data appears to be log-transformed (max value: {max_val:.2f}). "
                    "Deviance-based feature selection works best on raw counts. "
                    "Consider using adata.raw if available."
                )

            # Calculate deviance using shared utility
            logger.info("Calculating binomial deviance from multinomial null model")
            deviance_scores = calculate_deviance(count_data)

            # Select top n_top_genes by deviance
            n_features_to_select = min(n_top_genes, len(deviance_scores))
            top_deviance_idx = np.argsort(deviance_scores)[::-1][:n_features_to_select]

            # Mark selected features in adata.var
            adata_processed.var["highly_deviant"] = False
            adata_processed.var.loc[
                gene_names[top_deviance_idx], "highly_deviant"
            ] = True

            # Store deviance scores
            adata_processed.var["deviance_score"] = 0.0
            adata_processed.var.loc[gene_names, "deviance_score"] = deviance_scores

            # Calculate statistics
            selected_genes = adata_processed.var[
                adata_processed.var["highly_deviant"]
            ].index.tolist()
            n_selected = len(selected_genes)

            # Compute percentile thresholds for context
            deviance_threshold = (
                deviance_scores[top_deviance_idx[-1]] if n_selected > 0 else 0.0
            )
            deviance_median = float(np.median(deviance_scores))
            deviance_mean = float(np.mean(deviance_scores))

            processing_stats = {
                "analysis_type": "deviance_feature_selection",
                "method": "binomial_deviance_multinomial_null",
                "n_top_genes_requested": n_top_genes,
                "n_features_selected": n_selected,
                "original_n_genes": original_n_genes,
                "selection_rate": (n_selected / original_n_genes) * 100,
                "deviance_threshold": float(deviance_threshold),
                "deviance_median": deviance_median,
                "deviance_mean": deviance_mean,
                "used_raw_counts": adata.raw is not None,
                "top_10_genes": selected_genes[:10],
            }

            logger.info(
                f"Deviance-based feature selection completed: {n_selected}/{original_n_genes} genes selected "
                f"({processing_stats['selection_rate']:.1f}%)"
            )

            # Create IR for notebook export
            ir = self._create_deviance_selection_ir(
                n_top_genes=n_top_genes,
            )

            return adata_processed, processing_stats, ir

        except Exception as e:
            logger.exception(f"Error in deviance-based feature selection: {e}")
            raise PreprocessingError(
                f"Deviance-based feature selection failed: {str(e)}"
            )

    def _create_deviance_selection_ir(
        self,
        n_top_genes: int,
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for deviance-based feature selection.

        Args:
            n_top_genes: Number of top deviant genes to select

        Returns:
            AnalysisStep with complete code generation instructions
        """
        # Create parameter schema with Papermill flags
        parameter_schema = {
            "n_top_genes": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=4000,
                required=False,
                validation_rule="n_top_genes > 0",
                description="Number of top deviant genes to select",
            ),
        }

        # Jinja2 template with parameter placeholders
        code_template = """# Deviance-based feature selection
# Works on raw counts without normalization bias
# Implementation based on Townes et al. (2019)

import numpy as np
import scipy.sparse as spr

# Helper function to calculate deviance
def calculate_deviance(count_matrix):
    if spr.issparse(count_matrix):
        X = count_matrix.toarray()
    else:
        X = count_matrix.copy()

    X = np.maximum(X, 1e-10)
    cell_totals = X.sum(axis=1, keepdims=True)
    gene_totals = X.sum(axis=0)
    total_counts = X.sum()

    p_null = gene_totals / total_counts
    p_null = np.maximum(p_null, 1e-10)

    expected = cell_totals @ p_null.reshape(1, -1)
    expected = np.maximum(expected, 1e-10)

    mask = X > 0
    deviance_terms = np.zeros_like(X)
    deviance_terms[mask] = 2 * X[mask] * np.log(X[mask] / expected[mask])

    return deviance_terms.sum(axis=0)

# Calculate binomial deviance from multinomial null model
count_data = adata.raw.X if adata.raw is not None else adata.X
deviance_scores = calculate_deviance(count_data)

# Select top {{ n_top_genes }} genes by deviance
n_features_to_select = min({{ n_top_genes }}, len(deviance_scores))
top_deviance_idx = np.argsort(deviance_scores)[::-1][:n_features_to_select]

# Mark selected features
adata.var['highly_deviant'] = False
gene_names = adata.raw.var_names if adata.raw is not None else adata.var_names
adata.var.loc[gene_names[top_deviance_idx], 'highly_deviant'] = True

# Store deviance scores
adata.var['deviance_score'] = 0.0
adata.var.loc[gene_names, 'deviance_score'] = deviance_scores

# Display selection summary
n_selected = adata.var['highly_deviant'].sum()
print(f"Selected {n_selected} highly deviant genes out of {adata.n_vars} total genes")
print(f"Top 10 genes: {adata.var_names[adata.var['highly_deviant']].tolist()[:10]}")
"""

        # Create AnalysisStep
        ir = AnalysisStep(
            operation="deviance_feature_selection",
            tool_name="select_features_deviance",
            description=f"Select top {n_top_genes} highly deviant genes using binomial deviance from multinomial null model",
            library="numpy",
            code_template=code_template,
            imports=["import numpy as np", "import scipy.sparse as spr"],
            parameters={
                "n_top_genes": n_top_genes,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "method": "binomial_deviance_multinomial_null",
                "reference": "Townes et al. (2019)",
            },
            validates_on_export=True,
            requires_validation=False,
        )

        logger.debug(f"Created IR for deviance feature selection: {ir.operation}")
        return ir

    def integrate_and_batch_correct(self, *args, **kwargs):
        """
        DEPRECATED: This method is non-functional and has been removed.

        Batch correction was designed for the old stateful service pattern and
        references self.data_manager which no longer exists in stateless services.

        For batch correction, use scanpy's built-in methods directly on your
        AnnData object:
        - scanpy.pp.combat() for ComBat batch correction
        - scanpy.external.pp.harmony_integrate() for Harmony integration
        - scanpy.external.pp.scanorama_integrate() for Scanorama integration

        Alternatively, request this feature to be reimplemented following the
        new stateless service pattern (returning Tuple[AnnData, Dict, AnalysisStep]).

        Raises:
            NotImplementedError: This method is no longer supported

        Example:
            >>> import scanpy as sc
            >>> # ComBat batch correction
            >>> sc.pp.combat(adata, key='batch')
            >>> # Harmony integration
            >>> sc.external.pp.harmony_integrate(adata, key='batch')
        """
        raise NotImplementedError(
            "integrate_and_batch_correct() has been removed due to referencing "
            "non-existent self.data_manager. Use scanpy.pp.combat() or "
            "scanpy.external.pp.harmony_integrate() directly on your AnnData object."
        )

    # Helper methods for ambient RNA correction
    def _simple_decontamination(
        self,
        count_matrix: Union[np.ndarray, spr.spmatrix],
        contamination_fraction: float,
        empty_threshold: int,
    ) -> Union[np.ndarray, spr.spmatrix]:
        """Apply simple ambient RNA decontamination."""
        logger.info("Applying simple decontamination method")

        # Convert to dense if sparse for easier computation
        if spr.issparse(count_matrix):
            dense_matrix = count_matrix.toarray()
            was_sparse = True
        else:
            dense_matrix = count_matrix.copy()
            was_sparse = False

        # Estimate ambient profile from cells with low UMI counts
        cell_total_umis = np.sum(dense_matrix, axis=1)
        empty_cell_mask = cell_total_umis < empty_threshold

        if np.sum(empty_cell_mask) > 10:  # Need at least 10 empty droplets
            ambient_profile = np.mean(dense_matrix[empty_cell_mask, :], axis=0)
        else:
            # Fallback: use lowest expressed genes as ambient
            gene_means = np.mean(dense_matrix, axis=0)
            ambient_profile = gene_means * 0.1  # Assume ambient is 10% of average

        # Remove estimated ambient contribution
        corrected_matrix = dense_matrix.copy()
        for i in range(dense_matrix.shape[0]):
            cell_total = np.sum(dense_matrix[i, :])
            ambient_contribution = ambient_profile * cell_total * contamination_fraction
            corrected_matrix[i, :] = np.maximum(
                dense_matrix[i, :] - ambient_contribution,
                dense_matrix[i, :] * 0.1,  # Keep at least 10% of original counts
            )

        # Convert back to sparse if original was sparse
        if was_sparse:
            return spr.csr_matrix(corrected_matrix)
        else:
            return corrected_matrix

    def _quantile_based_correction(
        self,
        count_matrix: Union[np.ndarray, spr.spmatrix],
        contamination_fraction: float,
    ) -> Union[np.ndarray, spr.spmatrix]:
        """Apply quantile-based ambient RNA correction."""
        logger.info("Applying quantile-based correction method")

        if spr.issparse(count_matrix):
            dense_matrix = count_matrix.toarray()
            was_sparse = True
        else:
            dense_matrix = count_matrix.copy()
            was_sparse = False

        # For each gene, estimate ambient level as low quantile
        corrected_matrix = dense_matrix.copy()
        for j in range(dense_matrix.shape[1]):
            gene_counts = dense_matrix[:, j]
            gene_counts_nonzero = gene_counts[gene_counts > 0]

            if len(gene_counts_nonzero) > 10:
                # Estimate ambient as 5th percentile of non-zero values
                ambient_level = np.percentile(gene_counts_nonzero, 5)
                # Remove contamination fraction of ambient from all cells
                correction = ambient_level * contamination_fraction
                corrected_matrix[:, j] = np.maximum(
                    gene_counts - correction, gene_counts * 0.1
                )

        if was_sparse:
            return spr.csr_matrix(corrected_matrix)
        else:
            return corrected_matrix

    def _create_ambient_correction_plot(self, adata) -> go.Figure:
        """Create before/after ambient RNA correction comparison plot."""
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Before Correction", "After Correction"],
            horizontal_spacing=0.1,
        )

        # Before correction (raw data)
        raw_total_counts = np.array(adata.raw.X.sum(axis=1)).flatten()
        fig.add_trace(
            go.Histogram(
                x=raw_total_counts,
                nbinsx=50,
                name="Before",
                opacity=0.7,
                marker_color="red",
            ),
            row=1,
            col=1,
        )

        # After correction (current data)
        corrected_total_counts = np.array(adata.X.sum(axis=1)).flatten()
        fig.add_trace(
            go.Histogram(
                x=corrected_total_counts,
                nbinsx=50,
                name="After",
                opacity=0.7,
                marker_color="blue",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Total UMI Counts per Cell")
        fig.update_yaxes(title_text="Number of Cells")
        fig.update_layout(
            title="Ambient RNA Correction: UMI Distribution Before/After",
            height=400,
            showlegend=False,
        )

        return fig

    # Helper methods for filtering and normalization
    def _calculate_qc_metrics(self, adata):
        """Calculate quality control metrics."""
        logger.info("Calculating QC metrics")

        # Basic QC metrics
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

        # Mitochondrial genes
        adata.var["mt"] = adata.var_names.str.startswith(
            "MT-"
        ) | adata.var_names.str.startswith("mt-")
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )

        # Ribosomal genes
        adata.var["ribo"] = (
            adata.var_names.str.startswith("RPS")
            | adata.var_names.str.startswith("RPL")
            | adata.var_names.str.startswith("Rps")
            | adata.var_names.str.startswith("Rpl")
        )
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["ribo"], percent_top=None, log1p=False, inplace=True
        )

    def _apply_quality_filters(
        self,
        adata,
        min_genes_per_cell,
        max_genes_per_cell,
        min_cells_per_gene,
        max_mito_percent,
        max_ribo_percent,
    ) -> Dict[str, Any]:
        """Apply quality control filters."""
        logger.info("Applying quality control filters")

        initial_cells = adata.n_obs
        initial_genes = adata.n_vars

        # Filter cells
        sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

        # Additional cell filtering based on QC metrics
        cell_filter = (
            (adata.obs["n_genes_by_counts"] <= max_genes_per_cell)
            & (adata.obs["pct_counts_mt"] <= max_mito_percent)
            & (adata.obs["pct_counts_ribo"] <= max_ribo_percent)
        )

        adata._inplace_subset_obs(cell_filter)

        final_cells = adata.n_obs
        final_genes = adata.n_vars

        filtering_stats = {
            "initial_cells": initial_cells,
            "final_cells": final_cells,
            "cells_removed": initial_cells - final_cells,
            "cells_retained_pct": (final_cells / initial_cells) * 100,
            "initial_genes": initial_genes,
            "final_genes": final_genes,
            "genes_removed": initial_genes - final_genes,
            "genes_retained_pct": (final_genes / initial_genes) * 100,
        }

        logger.info(
            f"Filtering complete: {initial_cells} → {final_cells} cells, {initial_genes} → {final_genes} genes"
        )
        return filtering_stats

    def _normalize_expression_data(
        self, adata, method: str, target_sum: int
    ) -> Dict[str, Any]:
        """Normalize expression data."""
        logger.info(f"Normalizing expression data using method: {method}")

        # Store raw data
        adata.raw = adata.copy()

        if method == "log1p":
            # Standard log1p normalization
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)

        elif method == "sctransform_like":
            # SCTransform-like normalization (simplified)
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, max_value=10)  # Clip values for stability

        else:
            logger.warning(f"Unknown normalization method {method}, using log1p")
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)

        # Calculate normalization statistics
        final_median_counts = np.median(np.sum(adata.X, axis=1))
        final_mean_counts = np.mean(np.sum(adata.X, axis=1))

        normalization_stats = {
            "normalization_method": method,
            "target_sum": target_sum,
            "final_median_counts": final_median_counts,
            "final_mean_counts": final_mean_counts,
        }

        return normalization_stats

    def _create_filtering_plots(self, adata, filtering_stats) -> go.Figure:
        """Create quality control filtering plots."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Genes per Cell",
                "UMI Counts per Cell",
                "Mitochondrial %",
                "Ribosomal %",
            ],
            vertical_spacing=0.1,
        )

        # Genes per cell
        fig.add_trace(
            go.Histogram(
                x=adata.obs["n_genes_by_counts"],
                nbinsx=50,
                opacity=0.7,
                marker_color="blue",
            ),
            row=1,
            col=1,
        )

        # UMI counts per cell
        fig.add_trace(
            go.Histogram(
                x=adata.obs["total_counts"],
                nbinsx=50,
                opacity=0.7,
                marker_color="green",
            ),
            row=1,
            col=2,
        )

        # Mitochondrial percentage
        fig.add_trace(
            go.Histogram(
                x=adata.obs["pct_counts_mt"], nbinsx=50, opacity=0.7, marker_color="red"
            ),
            row=2,
            col=1,
        )

        # Ribosomal percentage
        fig.add_trace(
            go.Histogram(
                x=adata.obs["pct_counts_ribo"],
                nbinsx=50,
                opacity=0.7,
                marker_color="orange",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Quality Control Metrics After Filtering",
            height=600,
            showlegend=False,
        )

        return fig

    def _create_normalization_plots(self, adata, normalization_stats) -> go.Figure:
        """Create normalization comparison plots."""
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Raw Counts", "Normalized Counts"],
            horizontal_spacing=0.1,
        )

        # Raw counts (if available)
        if adata.raw is not None:
            raw_total_counts = np.array(adata.raw.X.sum(axis=1)).flatten()
            fig.add_trace(
                go.Histogram(
                    x=raw_total_counts,
                    nbinsx=50,
                    name="Raw",
                    opacity=0.7,
                    marker_color="red",
                ),
                row=1,
                col=1,
            )

        # Normalized counts
        norm_total_counts = np.array(adata.X.sum(axis=1)).flatten()
        fig.add_trace(
            go.Histogram(
                x=norm_total_counts,
                nbinsx=50,
                name="Normalized",
                opacity=0.7,
                marker_color="blue",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Total Counts per Cell")
        fig.update_yaxes(title_text="Number of Cells")
        fig.update_layout(
            title="Expression Normalization Results", height=400, showlegend=False
        )

        return fig

    def _combine_preprocessing_plots(self, qc_plot, norm_plot) -> go.Figure:
        """Combine QC and normalization plots."""
        # For simplicity, return the QC plot as the main diagnostic
        return qc_plot

    # Helper methods for batch correction and integration
    def _find_integration_features(self, adata, n_features: int) -> Dict[str, Any]:
        """Find highly variable genes for integration."""
        logger.info(f"Finding {n_features} highly variable genes for integration")

        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=n_features, flavor="seurat_v3")

        # Count selected features
        n_hvg_selected = np.sum(adata.var["highly_variable"])

        # Store HVG information
        adata.raw = adata.copy()  # Store full data
        adata = adata[:, adata.var["highly_variable"]].copy()  # Subset to HVGs

        integration_stats = {
            "n_hvg_selected": n_hvg_selected,
            "hvg_selection_method": "seurat_v3",
            "target_n_features": n_features,
        }

        logger.info(f"Selected {n_hvg_selected} highly variable genes")
        return integration_stats

    def _compute_integration_pca(self, adata, n_components: int) -> Dict[str, Any]:
        """Compute PCA for integration."""
        logger.info(f"Computing PCA with {n_components} components")

        # Compute PCA
        sc.tl.pca(adata, n_comps=n_components)

        # Calculate variance explained
        variance_ratio = adata.uns["pca"]["variance_ratio"]
        total_variance_explained = np.sum(variance_ratio) * 100

        pca_stats = {
            "n_pca_components": n_components,
            "variance_explained_pct": total_variance_explained,
            "top_10_components_variance": variance_ratio[:10].tolist(),
        }

        logger.info(f"PCA computed: {total_variance_explained:.1f}% variance explained")
        return pca_stats

    def _apply_harmony_like_correction(
        self, adata, batch_key: str, theta: float, lambda_param: float
    ) -> Dict[str, Any]:
        """Apply Harmony-like batch correction."""
        logger.info("Applying Harmony-like batch correction")

        # Get PCA coordinates
        pca_coords = adata.obsm["X_pca"].copy()

        # Simple batch correction: center each batch separately
        corrected_coords = pca_coords.copy()

        for batch in adata.obs[batch_key].unique():
            batch_mask = adata.obs[batch_key] == batch
            batch_coords = pca_coords[batch_mask, :]

            # Center batch coordinates
            batch_mean = np.mean(batch_coords, axis=0)
            global_mean = np.mean(pca_coords, axis=0)

            # Apply correction with theta parameter as scaling factor
            correction = (batch_mean - global_mean) * (theta / (theta + 1))
            corrected_coords[batch_mask, :] = batch_coords - correction

        # Store corrected coordinates
        adata.obsm["X_pca_corrected"] = corrected_coords

        # Calculate correction strength
        original_var = np.var(pca_coords, axis=0).mean()
        corrected_var = np.var(corrected_coords, axis=0).mean()
        correction_strength = f"{(1 - corrected_var/original_var)*100:.1f}%"

        correction_stats = {
            "correction_method": "harmony_like",
            "theta": theta,
            "lambda_param": lambda_param,
            "correction_strength": correction_strength,
        }

        logger.info(
            f"Harmony-like correction applied with strength: {correction_strength}"
        )
        return correction_stats

    def _apply_scanorama_like_correction(self, adata, batch_key: str) -> Dict[str, Any]:
        """Apply Scanorama-like batch correction."""
        logger.info("Applying Scanorama-like batch correction")

        # Simple implementation: scale PCA coordinates by batch
        pca_coords = adata.obsm["X_pca"].copy()
        corrected_coords = pca_coords.copy()

        # Scale each batch to have similar variance
        global_std = np.std(pca_coords, axis=0)

        for batch in adata.obs[batch_key].unique():
            batch_mask = adata.obs[batch_key] == batch
            batch_coords = pca_coords[batch_mask, :]

            if np.sum(batch_mask) > 1:  # Need at least 2 cells
                batch_std = np.std(batch_coords, axis=0)
                # Avoid division by zero
                batch_std[batch_std == 0] = 1
                scaling_factor = global_std / batch_std
                corrected_coords[batch_mask, :] = batch_coords * scaling_factor

        adata.obsm["X_pca_corrected"] = corrected_coords

        correction_stats = {
            "correction_method": "scanorama_like",
            "correction_strength": "moderate",
        }

        logger.info("Scanorama-like correction applied")
        return correction_stats

    def _apply_simple_batch_scaling(self, adata, batch_key: str) -> Dict[str, Any]:
        """Apply simple batch scaling correction."""
        logger.info("Applying simple batch scaling correction")

        # Simple z-score normalization per batch
        pca_coords = adata.obsm["X_pca"].copy()
        corrected_coords = pca_coords.copy()

        for batch in adata.obs[batch_key].unique():
            batch_mask = adata.obs[batch_key] == batch
            batch_coords = pca_coords[batch_mask, :]

            if np.sum(batch_mask) > 1:
                # Z-score normalize within batch
                batch_mean = np.mean(batch_coords, axis=0)
                batch_std = np.std(batch_coords, axis=0)
                batch_std[batch_std == 0] = 1  # Avoid division by zero

                corrected_coords[batch_mask, :] = (
                    batch_coords - batch_mean
                ) / batch_std

        adata.obsm["X_pca_corrected"] = corrected_coords

        correction_stats = {
            "correction_method": "simple_scaling",
            "correction_strength": "strong",
        }

        logger.info("Simple batch scaling correction applied")
        return correction_stats

    def _compute_integrated_umap(self, adata):
        """Compute UMAP on integrated data."""
        logger.info("Computing UMAP on integrated data")

        # Use corrected PCA coordinates for UMAP
        if "X_pca_corrected" in adata.obsm:
            # Temporarily replace PCA with corrected version
            original_pca = adata.obsm["X_pca"].copy()
            adata.obsm["X_pca"] = adata.obsm["X_pca_corrected"]

        # Compute neighborhood graph and UMAP
        sc.pp.neighbors(adata, n_pcs=min(50, adata.obsm["X_pca"].shape[1]))
        sc.tl.umap(adata)

        # Restore original PCA if we replaced it
        if "X_pca_corrected" in adata.obsm:
            adata.obsm["X_pca"] = original_pca

    def _create_integration_plots(
        self, adata, batch_key: str, integration_stats
    ) -> go.Figure:
        """Create integration diagnostic plots."""
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Before Integration", "After Integration"],
            horizontal_spacing=0.1,
        )

        # Before integration (original PCA)
        if "X_pca_corrected" in adata.obsm:
            umap_coords = adata.obsm["X_umap"]
            batches = adata.obs[batch_key]

            # Plot original UMAP colored by batch (simplified)
            colors = [
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
            ]
            for i, batch in enumerate(batches.unique()):
                batch_mask = batches == batch
                fig.add_trace(
                    go.Scatter(
                        x=umap_coords[batch_mask, 0],
                        y=umap_coords[batch_mask, 1],
                        mode="markers",
                        name=f"Before - {batch}",
                        marker=dict(color=colors[i % len(colors)], size=3, opacity=0.6),
                    ),
                    row=1,
                    col=1,
                )

                # After integration (same UMAP but conceptually "after")
                fig.add_trace(
                    go.Scatter(
                        x=umap_coords[batch_mask, 0],
                        y=umap_coords[batch_mask, 1],
                        mode="markers",
                        name=f"After - {batch}",
                        marker=dict(color=colors[i % len(colors)], size=3, opacity=0.8),
                    ),
                    row=1,
                    col=2,
                )

        fig.update_xaxes(title_text="UMAP_1")
        fig.update_yaxes(title_text="UMAP_2")
        fig.update_layout(
            title="Batch Integration Results", height=500, showlegend=True
        )

        return fig

    def _format_batch_counts(self, batch_counts: Dict[str, int]) -> str:
        """Format batch counts for display."""
        formatted = []
        for batch, count in sorted(
            batch_counts.items(), key=lambda x: x[1], reverse=True
        ):
            formatted.append(f"- {batch}: {count:,} cells")
        return "\n".join(formatted)
