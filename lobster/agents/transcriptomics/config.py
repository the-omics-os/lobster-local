"""
Configuration for transcriptomics analysis.

This module defines data type detection and QC defaults for single-cell
and bulk RNA-seq analysis. Extracted from shared_tools.py for modularity.
"""

from typing import Any, Dict, Literal

import numpy as np
from anndata import AnnData

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["detect_data_type", "get_qc_defaults"]


def detect_data_type(adata: AnnData) -> Literal["single_cell", "bulk"]:
    """
    Auto-detect whether the data is single-cell or bulk RNA-seq.

    Detection heuristics (in order of priority):
    1. Observation count: >500 observations likely single-cell, <100 likely bulk
    2. Single-cell-specific obs columns: n_counts, n_genes, leiden, louvain
    3. Matrix sparsity: Single-cell typically >70% sparse

    Args:
        adata: AnnData object to analyze

    Returns:
        "single_cell" or "bulk" based on data characteristics

    Note:
        This is a heuristic-based detection. For ambiguous cases, defaults to
        "single_cell" as the more conservative preprocessing path.
    """
    # Heuristic 1: Observation count
    n_obs = adata.n_obs
    if n_obs < 100:
        logger.debug(f"Detected bulk RNA-seq based on low n_obs ({n_obs})")
        return "bulk"
    if n_obs > 500:
        # Continue to other checks for confirmation
        obs_suggests_sc = True
    else:
        # Ambiguous range (100-500), rely on other heuristics
        obs_suggests_sc = None

    # Heuristic 2: Single-cell-specific observation columns
    sc_indicator_columns = {"n_counts", "n_genes", "leiden", "louvain", "total_counts"}
    obs_columns = set(adata.obs.columns.str.lower())
    sc_columns_present = len(sc_indicator_columns.intersection(obs_columns))
    if sc_columns_present >= 2:
        logger.debug(
            f"Detected single-cell based on SC-specific columns: "
            f"{sc_indicator_columns.intersection(obs_columns)}"
        )
        return "single_cell"

    # Heuristic 3: Matrix sparsity
    if hasattr(adata.X, "toarray"):
        # Sparse matrix - calculate sparsity
        total_elements = adata.X.shape[0] * adata.X.shape[1]
        nonzero_elements = adata.X.nnz
        sparsity = 1 - (nonzero_elements / total_elements)
    else:
        # Dense matrix - calculate sparsity from zeros
        total_elements = adata.X.size
        nonzero_elements = np.count_nonzero(adata.X)
        sparsity = 1 - (nonzero_elements / total_elements)

    if sparsity > 0.70:
        logger.debug(f"Detected single-cell based on high sparsity ({sparsity:.2%})")
        return "single_cell"

    # Final decision based on observation count suggestion
    if obs_suggests_sc is True:
        logger.debug(
            f"Detected single-cell based on high n_obs ({n_obs}) "
            f"with moderate sparsity ({sparsity:.2%})"
        )
        return "single_cell"

    # Default to single-cell if uncertain (more conservative preprocessing)
    logger.debug(
        f"Uncertain data type (n_obs={n_obs}, sparsity={sparsity:.2%}). "
        "Defaulting to single_cell."
    )
    return "single_cell"


def get_qc_defaults(data_type: Literal["single_cell", "bulk"]) -> Dict[str, Any]:
    """
    Get QC parameter defaults based on data type.

    Single-cell defaults follow Scanpy/Seurat conventions:
    - min_genes=200: Standard Scanpy threshold for low-quality cells
    - max_genes=5000: Filter potential doublets (cells with abnormally high gene count)
    - max_mt_pct=20.0: Standard mitochondrial cutoff for dying cells
    - target_sum=10000: Standard CPM normalization for single-cell

    Bulk RNA-seq defaults are more permissive:
    - min_genes=1000: Bulk samples typically express more genes
    - max_genes=None: No upper limit (doublets not a concern)
    - max_mt_pct=30.0: More permissive for bulk samples
    - target_sum=1000000: TPM/CPM normalization for bulk

    Args:
        data_type: Either "single_cell" or "bulk"

    Returns:
        Dictionary with QC parameter defaults

    Note on max_mt_pct:
        For cardiac/muscle tissue or metabolically active cells (neurons, hepatocytes),
        consider using max_mt_pct=30-50% as these cells naturally have higher
        mitochondrial content.

    Note on max_genes:
        For metabolically active cells or highly proliferative populations,
        consider relaxing max_genes to 8000-10000 to avoid filtering
        legitimate high-complexity cells.
    """
    if data_type == "single_cell":
        return {
            "min_genes": 200,  # Scanpy standard
            "max_genes": 5000,  # Doublet filtering
            "min_cells_per_gene": 3,  # Standard filter for rare genes
            "max_mt_pct": 20.0,  # Standard mitochondrial cutoff
            "max_ribo_pct": 50.0,  # Ribosomal cutoff
            "target_sum": 10000,  # Standard SC normalization
            "normalization_method": "log1p",
        }
    else:  # bulk
        return {
            "min_genes": 1000,  # Higher threshold for bulk
            "max_genes": None,  # No upper limit for bulk
            "min_cells_per_gene": 2,  # Min samples expressing gene
            "max_mt_pct": 30.0,  # More permissive for bulk
            "max_ribo_pct": 100.0,  # No ribosomal filter for bulk
            "target_sum": 1000000,  # TPM/CPM for bulk
            "normalization_method": "log1p",
        }
