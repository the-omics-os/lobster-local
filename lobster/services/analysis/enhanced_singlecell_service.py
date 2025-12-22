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
from scipy.stats import pearsonr

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.services.analysis.pathway_enrichment_service import (
    PathwayEnrichmentService,
    PathwayEnrichmentError,
)

try:
    import scrublet as scr

    SCRUBLET_AVAILABLE = True
except ImportError:
    SCRUBLET_AVAILABLE = False
    scr = None

# Future CellTypist integration (scheduled Q2 2025)
# import celltypist
# from celltypist import models

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

        logger.debug(f"Loaded {len(self.cell_type_markers)} cell type marker sets")
        logger.debug(f"Available cell types: {list(self.cell_type_markers.keys())}")
        logger.debug("EnhancedSingleCellService initialized successfully")

    def detect_doublets(
        self,
        adata: anndata.AnnData,
        expected_doublet_rate: float = 0.025,
        threshold: Optional[float] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Detect doublets using Scrublet or fallback method.

        Args:
            adata: AnnData object for doublet detection
            expected_doublet_rate: Expected doublet rate
            threshold: Custom threshold for doublet calling

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with doublet scores, detection stats, and IR

        Raises:
            SingleCellError: If doublet detection fails
        """
        try:
            logger.info(
                f"Starting doublet detection with expected rate: {expected_doublet_rate}"
            )

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
            if hasattr(counts_matrix, "toarray"):
                counts_matrix = counts_matrix.toarray()

            logger.info(f"Doublet detection matrix shape: {counts_matrix.shape}")

            # Check if we have enough features
            if counts_matrix.shape[1] == 0:
                raise SingleCellError("Expression matrix has no gene features")

            # Run doublet detection
            if SCRUBLET_AVAILABLE:
                try:
                    logger.info("Running Scrublet doublet detection")
                    scrub = scr.Scrublet(
                        counts_matrix, expected_doublet_rate=expected_doublet_rate
                    )

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
                    doublet_scores, predicted_doublets, detection_method = (
                        self._fallback_doublet_detection(
                            counts_matrix, expected_doublet_rate
                        )
                    )
            else:
                logger.info("Scrublet not available, using fallback doublet detection")
                doublet_scores, predicted_doublets, detection_method = (
                    self._fallback_doublet_detection(
                        counts_matrix, expected_doublet_rate
                    )
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
                    "std": float(doublet_scores.std()),
                },
            }

            logger.info(
                f"Doublet detection completed: {n_doublets} doublets detected ({doublet_rate:.1%})"
            )

            # Create IR for provenance tracking
            ir = self._create_doublet_detection_ir(
                expected_doublet_rate=expected_doublet_rate,
                threshold=threshold,
                detection_method=detection_method,
            )

            return adata_doublets, detection_stats, ir

        except Exception as e:
            logger.exception(f"Error in doublet detection: {e}")
            raise SingleCellError(f"Doublet detection failed: {str(e)}")

    def _fallback_doublet_detection(
        self, counts_matrix: np.ndarray, expected_doublet_rate: float
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

        counts_matrix.shape[0]

        # Calculate per-cell metrics that indicate doublets
        total_counts = np.sum(counts_matrix, axis=1)
        n_genes = np.sum(counts_matrix > 0, axis=1)

        # Normalize metrics to z-scores
        total_counts_z = np.abs(
            (total_counts - np.mean(total_counts)) / np.std(total_counts)
        )
        n_genes_z = np.abs((n_genes - np.mean(n_genes)) / np.std(n_genes))

        # Combined doublet score (higher = more likely doublet)
        doublet_scores = (total_counts_z + n_genes_z) / 2

        # Apply threshold based on expected doublet rate
        doublet_threshold = np.percentile(
            doublet_scores, (1 - expected_doublet_rate) * 100
        )
        predicted_doublets = doublet_scores > doublet_threshold

        logger.info(
            f"Fallback method detected {np.sum(predicted_doublets)} potential doublets"
        )

        return doublet_scores, predicted_doublets, "fallback_outlier_detection"

    def _create_doublet_detection_ir(
        self,
        expected_doublet_rate: float,
        threshold: Optional[float],
        detection_method: str,
    ) -> AnalysisStep:
        """Create AnalysisStep IR for doublet detection."""

        code_template = """
# Doublet detection using Scrublet (or fallback)
import scrublet as scr
import numpy as np

# Get count matrix (use raw if available)
counts_matrix = adata.raw.X if adata.raw is not None else adata.X
if hasattr(counts_matrix, 'toarray'):
    counts_matrix = counts_matrix.toarray()

# Run Scrublet
scrub = scr.Scrublet(counts_matrix, expected_doublet_rate={{ expected_doublet_rate }})
doublet_scores, predicted_doublets = scrub.scrub_doublets(
    min_counts=2,
    min_cells=3,
    min_gene_variability_pctl=85,
    n_prin_comps=30,
)
{% if threshold %}
# Apply custom threshold
predicted_doublets = scrub.call_doublets(threshold={{ threshold }})
{% endif %}

# Add results to AnnData
adata.obs['doublet_score'] = doublet_scores
adata.obs['predicted_doublet'] = predicted_doublets

# Results stored in:
# - adata.obs['doublet_score']: Continuous doublet likelihood (0-1)
# - adata.obs['predicted_doublet']: Boolean classification
"""

        return AnalysisStep(
            operation="detect_doublets",
            tool_name="EnhancedSingleCellService.detect_doublets",
            description=f"Doublet detection using {detection_method}",
            library="scrublet" if detection_method == "scrublet" else "numpy",
            code_template=code_template,
            imports=[
                "import scrublet as scr",
                "import numpy as np",
            ],
            parameters={
                "expected_doublet_rate": expected_doublet_rate,
                "threshold": threshold,
                "detection_method": detection_method,
            },
            parameter_schema={
                "expected_doublet_rate": {
                    "type": "number",
                    "description": "Expected doublet rate (typically 0.025-0.1)",
                    "default": 0.025,
                },
                "threshold": {
                    "type": "number",
                    "description": "Custom threshold for doublet calling (None for automatic)",
                    "required": False,
                },
            },
            input_entities=["adata"],
            output_entities=["adata_doublets"],
        )

    def annotate_cell_types(
        self,
        adata: anndata.AnnData,
        reference_markers: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Annotate cell types based on marker genes with per-cell confidence scoring.

        Args:
            adata: AnnData object with clustering results
            reference_markers: Optional custom marker genes dictionary

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]: AnnData with cell type annotations, stats, and IR

        Raises:
            SingleCellError: If annotation fails
        """
        try:
            logger.info("Starting cell type annotation using marker genes")

            # Validate input
            if "leiden" not in adata.obs.columns:
                raise SingleCellError(
                    "No clustering results found. Please run clustering first."
                )

            # Create working copy
            adata_annotated = adata.copy()

            # Use provided markers or default ones
            markers = reference_markers or self.cell_type_markers
            logger.info(f"Using {len(markers)} marker sets for annotation")

            # Calculate marker gene scores for each cluster
            cluster_annotations = self._calculate_marker_scores_from_adata(
                adata_annotated, markers
            )

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

            # Calculate per-cell confidence scores
            if markers:
                logger.info("Calculating per-cell confidence scores...")
                confidence, top3, entropy = self._calculate_per_cell_confidence(
                    adata_annotated, markers, cell_type_col="cell_type"
                )

                adata_annotated.obs["cell_type_confidence"] = confidence
                adata_annotated.obs["cell_type_top3"] = top3
                adata_annotated.obs["annotation_entropy"] = entropy

                # Quality flag: high confidence if score > 0.5 and entropy < 0.8
                adata_annotated.obs["annotation_quality"] = [
                    (
                        "high"
                        if (c > 0.5 and e < 0.8)
                        else "medium" if (c > 0.3 and e < 1.0) else "low"
                    )
                    for c, e in zip(confidence, entropy)
                ]

                logger.info(
                    f"Confidence scores: mean={float(np.mean(confidence)):.3f}, "
                    f"median={float(np.median(confidence)):.3f}"
                )
            else:
                logger.warning(
                    "No reference markers provided - skipping confidence scoring"
                )

            # Calculate annotation statistics
            cell_type_counts = cell_types.value_counts().to_dict()
            n_cell_types = len(set(cell_types))

            annotation_stats = {
                "analysis_type": "cell_type_annotation",
                "markers_used": list(markers.keys()),
                "n_marker_sets": len(markers),
                "n_clusters": len(adata_annotated.obs["leiden"].unique()),
                "n_cell_types_identified": n_cell_types,
                "cluster_to_celltype": {
                    str(k): v for k, v in cluster_to_celltype.items()
                },
                "cell_type_counts": {
                    str(k): int(v) for k, v in cell_type_counts.items()
                },
                "marker_scores": cluster_annotations,
            }

            # Update stats with confidence distribution if available
            if markers:
                annotation_stats["confidence_mean"] = float(np.mean(confidence))
                annotation_stats["confidence_median"] = float(np.median(confidence))
                annotation_stats["confidence_std"] = float(np.std(confidence))
                annotation_stats["quality_distribution"] = {
                    "high": int(
                        np.sum(adata_annotated.obs["annotation_quality"] == "high")
                    ),
                    "medium": int(
                        np.sum(adata_annotated.obs["annotation_quality"] == "medium")
                    ),
                    "low": int(
                        np.sum(adata_annotated.obs["annotation_quality"] == "low")
                    ),
                }

            logger.info(
                f"Cell type annotation completed: {n_cell_types} cell types identified"
            )

            # Create IR for provenance tracking
            ir = self._create_annotation_ir(reference_markers=markers)

            return adata_annotated, annotation_stats, ir

        except Exception as e:
            logger.exception(f"Error in cell type annotation: {e}")
            raise SingleCellError(f"Cell type annotation failed: {str(e)}")

    def _calculate_per_cell_confidence(
        self,
        adata: anndata.AnnData,
        reference_markers: Dict[str, List[str]],
        cell_type_col: str = "cell_type",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate per-cell confidence scores for cell type annotations.

        Uses Pearson correlation between cell expression profiles and marker gene
        signatures, plus Shannon entropy for top-3 predictions.

        Args:
            adata: AnnData with cell type annotations
            reference_markers: Marker genes per cell type
            cell_type_col: Column in .obs with cell type assignments

        Returns:
            Tuple of:
            - confidence_scores: Array of max correlation scores (0-1)
            - top3_predictions: Array of top-3 cell type predictions (str)
            - entropy_scores: Array of Shannon entropy values (lower = more confident)
        """
        from scipy.stats import entropy as shannon_entropy
        from scipy.stats import pearsonr

        n_cells = adata.n_obs
        confidence_scores = np.zeros(n_cells)
        top3_predictions = np.empty(n_cells, dtype=object)
        entropy_scores = np.zeros(n_cells)

        # Build mean expression signatures for each cell type
        cell_types = list(reference_markers.keys())
        signatures = {}

        for ct, markers in reference_markers.items():
            # Filter markers present in dataset
            valid_markers = [m for m in markers if m in adata.var_names]
            if len(valid_markers) > 0:
                # Mean expression of marker genes
                marker_expr = adata[:, valid_markers].X
                if hasattr(marker_expr, "toarray"):
                    marker_expr = marker_expr.toarray()
                signatures[ct] = np.mean(marker_expr, axis=1)
            else:
                signatures[ct] = np.zeros(n_cells)

        # Calculate correlation for each cell against all signatures
        for i in range(n_cells):
            correlations = {}
            for ct, sig in signatures.items():
                if np.std(sig) > 0:  # Avoid division by zero
                    # Pearson correlation between cell i and signature
                    cell_expr = adata[i, :].X
                    if hasattr(cell_expr, "toarray"):
                        cell_expr = cell_expr.toarray().flatten()
                    else:
                        cell_expr = np.array(cell_expr).flatten()

                    # Use only marker genes for correlation
                    markers = [m for m in reference_markers[ct] if m in adata.var_names]
                    if len(markers) > 0:
                        marker_indices = [
                            list(adata.var_names).index(m) for m in markers
                        ]
                        cell_marker_expr = cell_expr[marker_indices]
                        sig_marker_expr = (
                            sig[marker_indices] if len(sig.shape) > 0 else sig
                        )

                        if np.std(cell_marker_expr) > 0:
                            corr, _ = pearsonr(cell_marker_expr, sig_marker_expr)
                            correlations[ct] = max(
                                0, corr
                            )  # Clip negative correlations
                        else:
                            correlations[ct] = 0.0
                    else:
                        correlations[ct] = 0.0
                else:
                    correlations[ct] = 0.0

            # Get top 3 predictions
            sorted_cts = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            top3 = [ct for ct, _ in sorted_cts[:3]]
            top3_scores = np.array([correlations[ct] for ct in top3])

            # Confidence = max correlation
            confidence_scores[i] = (
                correlations[sorted_cts[0][0]] if len(sorted_cts) > 0 else 0.0
            )

            # Top 3 as comma-separated string
            top3_predictions[i] = ",".join(top3)

            # Shannon entropy of top-3 probabilities (normalized)
            if np.sum(top3_scores) > 0:
                top3_probs = top3_scores / np.sum(top3_scores)
                entropy_scores[i] = shannon_entropy(top3_probs)
            else:
                entropy_scores[i] = np.log(3)  # Max entropy for uniform distribution

        return confidence_scores, top3_predictions, entropy_scores

    def _create_annotation_ir(
        self, reference_markers: Optional[Dict[str, List[str]]] = None
    ) -> AnalysisStep:
        """Create AnalysisStep IR for cell type annotation."""

        code_template = """
# Cell type annotation with confidence scoring
import scanpy as sc
import numpy as np
from scipy.stats import pearsonr, entropy

# Define marker genes (user-provided or default)
reference_markers = {{ reference_markers }}

# Calculate mean expression per cluster
cluster_col = 'leiden'  # or user-specified
for cluster in adata.obs[cluster_col].unique():
    cluster_mask = adata.obs[cluster_col] == cluster
    cluster_cells = adata[cluster_mask]

    # Score against each cell type
    scores = {}
    for cell_type, markers in reference_markers.items():
        valid_markers = [m for m in markers if m in adata.var_names]
        if valid_markers:
            expr = cluster_cells[:, valid_markers].X.mean(axis=0)
            scores[cell_type] = float(np.mean(expr))

    # Assign cell type with highest score
    best_type = max(scores, key=scores.get)
    adata.obs.loc[cluster_mask, 'cell_type'] = best_type

# Calculate per-cell confidence (Pearson correlation with signatures)
# ... (confidence calculation logic) ...

# Results stored in:
# - adata.obs['cell_type']: Assigned cell type
# - adata.obs['cell_type_confidence']: Correlation score (0-1)
# - adata.obs['cell_type_top3']: Top 3 predictions
# - adata.obs['annotation_entropy']: Shannon entropy
# - adata.obs['annotation_quality']: 'high', 'medium', 'low'
"""

        return AnalysisStep(
            operation="annotate_cell_types_with_confidence",
            tool_name="EnhancedSingleCellService.annotate_cell_types",
            description="Manual cell type annotation using marker genes with per-cell confidence scoring",
            library="scanpy + scipy",
            code_template=code_template,
            imports=[
                "import scanpy as sc",
                "import numpy as np",
                "from scipy.stats import pearsonr, entropy",
            ],
            parameters={"reference_markers": reference_markers},
            parameter_schema={
                "reference_markers": {
                    "type": "dict",
                    "description": "Marker genes per cell type",
                    "required": False,
                }
            },
            input_entities=["adata"],
            output_entities=["adata_annotated"],
        )

    # =========================================================================
    # TODO: CellTypist Integration (Scheduled Q2 2025)
    # =========================================================================
    #
    # CellTypist is an industry-standard automated cell type annotation tool
    # offering 45+ pretrained models across tissues, species, and disease states.
    #
    # INSTALLATION:
    #   pip install celltypist
    #
    # MODEL MANAGEMENT:
    #   import celltypist
    #   celltypist.models.download_models()  # Download all models (~2GB)
    #   celltypist.models.models_description()  # List available models
    #
    # CLASSIFICATION WORKFLOW:
    #   1. Preprocess: Log-normalize counts (scanpy.pp.normalize_total + log1p)
    #   2. Run: celltypist.annotate(adata, model='Immune_All_Low.pkl')
    #   3. Post-process: Majority voting and over-clustering refinement
    #
    # INTEGRATION POINTS:
    #   - Add automated annotation as alternative to marker-based approach
    #   - Support custom model training from user-provided references
    #   - Merge CellTypist predictions with manual annotations
    #   - Compare confidence: manual marker scores vs CellTypist probabilities
    #   - Enable ensemble strategies (marker + classifier consensus)
    #
    # PERFORMANCE CONSIDERATIONS:
    #   - CellTypist requires log-normalized data (not raw counts)
    #   - Models are tissue/dataset-specific (validate applicability)
    #   - Majority voting requires over-clustering parameter tuning
    #   - Large datasets (>100K cells) may need batch processing
    #
    # RECOMMENDED MODELS BY TISSUE:
    #   - PBMC/Immune: Immune_All_Low.pkl, Immune_All_High.pkl
    #   - Pancreas: Pancreas.pkl
    #   - Lung: Healthy_COVID19_PBMC.pkl (if immune cells present)
    #   - Custom: Train with celltypist.train(...)
    #
    # REFERENCES:
    #   - Paper: Domínguez Conde et al., Science 2022
    #   - Docs: https://www.celltypist.org/
    #   - GitHub: https://github.com/Teichlab/celltypist
    #
    # SCHEDULED TIMELINE: Q2 2025
    # =========================================================================

    def annotate_with_celltypist(
        self,
        adata: anndata.AnnData,
        model: str = "Immune_All_Low.pkl",
        majority_voting: bool = False,
        over_clustering: bool = False,
        confidence_threshold: float = 0.5,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], Any]:
        """
        Annotate cell types using CellTypist automated classification.

        **Status: Not Yet Implemented - Scheduled for Q2 2025**

        CellTypist uses pretrained logistic regression models to classify cells
        based on their transcriptomic profiles. This method will provide automated
        annotation as an alternative or complement to marker-based approaches.

        Args:
            adata: AnnData object with log-normalized expression data
            model: Name of pretrained CellTypist model or path to custom model
            majority_voting: Apply majority voting to refine predictions
            over_clustering: Perform over-clustering before majority voting
            confidence_threshold: Minimum confidence score for high-quality predictions

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - AnnData with CellTypist predictions in .obs['celltypist_cell_type']
                - Statistics dictionary with prediction confidence and quality metrics
                - AnalysisStep for provenance tracking and notebook export

        Raises:
            NotImplementedError: This feature is scheduled for Q2 2025

        Future Implementation Notes:
            - Will add .obs['celltypist_cell_type'] (predicted cell types)
            - Will add .obs['celltypist_conf_score'] (per-cell confidence)
            - Will add .obs['celltypist_over_clustering'] (if enabled)
            - Will support both pretrained and custom models
            - Will integrate with existing marker-based annotation workflow
            - Will provide comparison metrics vs manual annotations

        Example Usage (Post-Implementation):
            >>> service = EnhancedSingleCellService()
            >>> adata_ann, stats, ir = service.annotate_with_celltypist(
            ...     adata,
            ...     model="Immune_All_Low.pkl",
            ...     majority_voting=True,
            ...     confidence_threshold=0.7
            ... )
            >>> print(stats['n_high_confidence'])  # cells above threshold
        """
        raise NotImplementedError(
            "CellTypist integration is scheduled for Q2 2025. "
            "This feature will enable automated cell type annotation using "
            "pretrained classifier models. Use annotate_cell_types() for "
            "marker-based annotation in the meantime.\n\n"
            "Required dependencies (not yet installed):\n"
            "  - celltypist>=1.6.0\n"
            "  - Compatible pretrained models\n\n"
            "For more information, see: https://www.celltypist.org/"
        )

    def find_marker_genes(
        self,
        adata: anndata.AnnData,
        groupby: str = "leiden",
        groups: Optional[List[str]] = None,
        method: str = "wilcoxon",
        n_genes: int = 25,
        min_fold_change: float = 1.5,
        min_pct: float = 0.25,
        max_out_pct: float = 0.5,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Find marker genes for clusters or cell types with specificity filtering.

        Args:
            adata: AnnData object with clustering/annotation results
            groupby: Column name to group by ('leiden', 'cell_type', etc.)
            groups: Specific groups to analyze (None for all)
            method: Statistical method ('wilcoxon', 't-test', 'logreg')
            n_genes: Number of top marker genes per group
            min_fold_change: Minimum fold-change threshold for filtering (default: 1.5).
                Genes with fold-change < this value will be filtered out.
            min_pct: Minimum percentage of cells expressing the gene in the group (default: 0.25).
                Genes expressed in < 25% of in-group cells will be filtered out.
            max_out_pct: Maximum percentage of cells expressing the gene in other groups (default: 0.5).
                Genes expressed in > 50% of out-group cells will be filtered out (less specific).

        Returns:
            Tuple containing:
            - AnnData: Object with .uns['rank_genes_groups'] containing filtered marker genes
            - Dict: Statistics including filtering summary and genes per group
            - AnalysisStep: Provenance IR for reproducibility

        Raises:
            SingleCellError: If marker gene detection fails
        """
        try:
            logger.info(f"Finding marker genes grouped by: {groupby}")

            # Validate input
            if groupby not in adata.obs.columns:
                raise SingleCellError(
                    f"Group column '{groupby}' not found in observations"
                )

            # Create working copy
            adata_markers = adata.copy()

            # Run differential expression analysis
            # Note: Only pass 'groups' parameter if explicitly set (not None)
            # Scanpy distinguishes between "parameter not provided" vs "parameter=None"
            # When groups=None is explicitly passed, scanpy's legacy_api_wrap fails
            scanpy_kwargs = {
                "groupby": groupby,
                "method": method,
                "n_genes": n_genes,
                "use_raw": True,
            }

            if groups is not None:
                scanpy_kwargs["groups"] = groups

            sc.tl.rank_genes_groups(adata_markers, **scanpy_kwargs)

            # Apply post-hoc filtering for marker specificity
            logger.info(
                f"Applying DEG filtering: min_fold_change={min_fold_change}, "
                f"min_pct={min_pct}, max_out_pct={max_out_pct}"
            )

            # Store pre-filter counts
            pre_filter_counts = {}
            if "names" in adata_markers.uns["rank_genes_groups"]:
                for group in adata_markers.uns["rank_genes_groups"][
                    "names"
                ].dtype.names:
                    pre_filter_counts[group] = len(
                        adata_markers.uns["rank_genes_groups"]["names"][group]
                    )

            # Apply scanpy's built-in filtering
            sc.tl.filter_rank_genes_groups(
                adata_markers,
                min_fold_change=min_fold_change,
                min_in_group_fraction=min_pct,
                max_out_group_fraction=max_out_pct,
            )

            # Store post-filter counts
            post_filter_counts = {}
            filtered_counts = {}
            if "names" in adata_markers.uns["rank_genes_groups"]:
                for group in adata_markers.uns["rank_genes_groups"][
                    "names"
                ].dtype.names:
                    # Count non-NaN entries (filtered genes are set to NaN)
                    valid_genes = ~pd.isna(
                        adata_markers.uns["rank_genes_groups"]["names"][group]
                    )
                    post_filter_counts[group] = int(np.sum(valid_genes))
                    filtered_counts[group] = (
                        pre_filter_counts[group] - post_filter_counts[group]
                    )

            logger.info(
                f"Filtering complete: "
                f"{sum(filtered_counts.values())} genes removed across all groups"
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
                "filtering_params": {
                    "min_fold_change": min_fold_change,
                    "min_pct": min_pct,
                    "max_out_pct": max_out_pct,
                },
                "pre_filter_counts": pre_filter_counts,
                "post_filter_counts": post_filter_counts,
                "filtered_counts": filtered_counts,
                "total_genes_filtered": sum(filtered_counts.values()),
                "n_groups": n_groups,
                "groups_analyzed": [str(g) for g in unique_groups],
                "has_marker_results": "rank_genes_groups" in adata_markers.uns,
                "marker_genes_df_shape": (
                    marker_genes_df.shape if not marker_genes_df.empty else (0, 0)
                ),
            }

            # Store top marker genes per group
            if not marker_genes_df.empty:
                top_markers_per_group = {}
                for group in marker_genes_df["group"].unique():
                    group_genes = marker_genes_df[
                        marker_genes_df["group"] == group
                    ].head(10)
                    top_markers_per_group[str(group)] = [
                        {
                            "gene": row["gene"],
                            "score": float(row["score"]),
                            "pval": float(row["pval"]),
                        }
                        for _, row in group_genes.iterrows()
                    ]
                marker_stats["top_markers_per_group"] = top_markers_per_group

            logger.info(
                f"Marker gene analysis completed for {n_groups} groups using {method} method"
            )

            # Create IR for provenance tracking
            ir = self._create_marker_genes_ir(
                groupby=groupby,
                method=method,
                n_genes=n_genes,
                min_fold_change=min_fold_change,
                min_pct=min_pct,
                max_out_pct=max_out_pct,
            )

            return adata_markers, marker_stats, ir

        except Exception as e:
            logger.exception(f"Error finding marker genes: {e}")
            raise SingleCellError(f"Marker gene analysis failed: {str(e)}")

    def _create_marker_genes_ir(
        self,
        groupby: str,
        method: str,
        n_genes: int,
        min_fold_change: float,
        min_pct: float,
        max_out_pct: float,
    ) -> AnalysisStep:
        """Create AnalysisStep IR for marker gene detection with filtering."""

        code_template = """
# Marker gene detection with specificity filtering
import scanpy as sc

# Step 1: Rank genes by differential expression
sc.tl.rank_genes_groups(
    adata,
    groupby='{{ groupby }}',
    method='{{ method }}',
    n_genes={{ n_genes }},
)

# Step 2: Filter for marker specificity
sc.tl.filter_rank_genes_groups(
    adata,
    min_fold_change={{ min_fold_change }},     # Upregulation threshold
    min_in_group_fraction={{ min_pct }},       # In-group expression
    max_out_group_fraction={{ max_out_pct }},  # Out-group expression
)

# Results stored in adata.uns['rank_genes_groups']:
# - 'names': Filtered gene names per group
# - 'scores': Test statistics
# - 'pvals': P-values
# - 'pvals_adj': Adjusted p-values
# - 'logfoldchanges': Log2 fold-changes
# Note: Filtered-out genes are set to NaN
"""

        return AnalysisStep(
            operation="find_marker_genes_with_filtering",
            tool_name="EnhancedSingleCellService.find_marker_genes",
            description=f"Differential expression analysis with post-hoc filtering (method: {method})",
            library="scanpy",
            code_template=code_template,
            imports=["import scanpy as sc"],
            parameters={
                "groupby": groupby,
                "method": method,
                "n_genes": n_genes,
                "min_fold_change": min_fold_change,
                "min_pct": min_pct,
                "max_out_pct": max_out_pct,
            },
            parameter_schema={
                "groupby": {
                    "type": "string",
                    "description": "Column in .obs for grouping",
                    "default": "leiden",
                },
                "method": {
                    "type": "string",
                    "description": "DE test method",
                    "default": "wilcoxon",
                    "enum": ["wilcoxon", "t-test", "logreg"],
                },
                "n_genes": {
                    "type": "integer",
                    "description": "Number of top genes to rank",
                    "default": 25,
                },
                "min_fold_change": {
                    "type": "number",
                    "description": "Minimum fold-change threshold",
                    "default": 1.5,
                },
                "min_pct": {
                    "type": "number",
                    "description": "Minimum in-group expression fraction",
                    "default": 0.25,
                },
                "max_out_pct": {
                    "type": "number",
                    "description": "Maximum out-group expression fraction",
                    "default": 0.5,
                },
            },
            input_entities=["adata"],
            output_entities=["adata_with_markers"],
        )

    def _calculate_marker_scores_from_adata(
        self, adata: anndata.AnnData, markers: Dict[str, List[str]]
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
                            if hasattr(subset.X, "toarray"):
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

    def run_pathway_enrichment(
        self,
        adata: anndata.AnnData,
        marker_genes: Optional[List[str]] = None,
        cluster_key: str = "leiden",
        databases: Optional[List[str]] = None,
        p_value_threshold: float = 0.05,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Run pathway enrichment on marker genes or DE results using gseapy.

        Uses real GO, KEGG, and Reactome databases via PathwayEnrichmentService.

        Args:
            adata: AnnData object with single-cell data
            marker_genes: List of marker genes (auto-extracts from adata if None)
            cluster_key: Cluster label key in adata.obs (for auto-extraction)
            databases: List of databases to query (defaults to GO + KEGG)
            p_value_threshold: Significance threshold for adjusted p-value

        Returns:
            Tuple of (adata_with_results, stats_dict, analysis_ir)

        Raises:
            SingleCellError: If pathway enrichment fails
        """
        try:
            logger.info("Running pathway enrichment on marker genes")

            # Auto-extract marker genes if not provided
            if marker_genes is None:
                marker_genes = self._extract_marker_genes(adata, cluster_key)
                logger.info(f"Auto-extracted {len(marker_genes)} marker genes")

            # Delegate to PathwayEnrichmentService
            pathway_service = PathwayEnrichmentService()

            # Default to GO + KEGG if no databases specified
            if databases is None:
                databases = ["GO_Biological_Process_2023", "KEGG_2021_Human"]

            # Perform enrichment
            adata_enriched, enrichment_stats, ir = pathway_service.over_representation_analysis(
                adata=adata,
                gene_list=marker_genes,
                databases=databases,
                organism="human",
                p_value_threshold=p_value_threshold,
                store_in_uns=True,
            )

            logger.info(
                f"Pathway enrichment completed: {enrichment_stats['n_significant_pathways']} significant pathways"
            )

            return adata_enriched, enrichment_stats, ir

        except PathwayEnrichmentError as e:
            logger.exception(f"Pathway enrichment error: {e}")
            raise SingleCellError(f"Pathway enrichment failed: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error in pathway enrichment: {e}")
            raise SingleCellError(f"Pathway enrichment failed: {str(e)}")

    def _extract_marker_genes(
        self, adata: anndata.AnnData, cluster_key: str, top_n: int = 50
    ) -> List[str]:
        """
        Extract top marker genes from Scanpy's rank_genes_groups results.

        Args:
            adata: AnnData with rank_genes_groups results
            cluster_key: Cluster label key
            top_n: Number of top genes per cluster

        Returns:
            List of unique marker gene symbols
        """
        marker_genes = []

        # Check if rank_genes_groups was run
        if "rank_genes_groups" in adata.uns:
            # Extract top genes per cluster
            result = adata.uns["rank_genes_groups"]
            groups = result["names"].dtype.names  # Cluster names

            for group in groups:
                # Get top N genes for this cluster
                genes = result["names"][group][:top_n]
                marker_genes.extend(genes)

            # Deduplicate
            marker_genes = list(set(marker_genes))
            logger.info(f"Extracted {len(marker_genes)} unique markers from {len(groups)} clusters")

        else:
            # Fallback: use highly variable genes
            logger.warning(
                "No rank_genes_groups found. Using highly_variable_genes as fallback."
            )
            if "highly_variable" in adata.var.columns:
                marker_genes = adata.var_names[adata.var["highly_variable"]].tolist()[:500]
            else:
                # Last resort: use all genes (not recommended)
                logger.warning("No highly variable genes found. Using all genes (not optimal).")
                marker_genes = adata.var_names.tolist()[:500]

        return marker_genes

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
