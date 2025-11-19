"""
Deviance-based feature selection utilities for single-cell RNA-seq.

Implementation based on Townes et al. (2019):
"Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model"
"""

from typing import Union

import numpy as np
import scipy.sparse as spr


def calculate_deviance(count_matrix: Union[np.ndarray, spr.spmatrix]) -> np.ndarray:
    """
    Calculate binomial deviance from multinomial null model for feature selection.

    This method works on raw counts without normalization bias, providing a mathematically
    principled alternative to highly variable genes (HVG) methods.

    The deviance measures how much each gene deviates from the expected expression under
    a simple multinomial null model where all cells have the same gene expression proportions.

    Mathematical formula:
        D(gene) = 2 × Σ_cells [x_ij × log(x_ij / μ_ij)]

    Where:
        - x_ij = observed count for gene j in cell i
        - μ_ij = expected count under multinomial null = n_i × p_j
        - n_i = total UMI count for cell i
        - p_j = gene j's proportion of total counts across all cells

    Args:
        count_matrix: Cell × gene count matrix (raw counts, sparse or dense)
                     Shape: (n_cells, n_genes)

    Returns:
        np.ndarray: Deviance score for each gene (higher = more variable)
                   Shape: (n_genes,)

    Example:
        >>> import scanpy as sc
        >>> adata = sc.datasets.pbmc3k()
        >>> deviance_scores = calculate_deviance(adata.X)
        >>> # Select top 2000 genes
        >>> top_genes_idx = np.argsort(deviance_scores)[::-1][:2000]
        >>> adata.var['highly_deviant'] = False
        >>> adata.var.iloc[top_genes_idx, adata.var.columns.get_loc('highly_deviant')] = True

    Reference:
        Townes, F. W., Hicks, S. C., Aryee, M. J., & Irizarry, R. A. (2019).
        Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model.
        Genome Biology, 20(1), 295. https://doi.org/10.1186/s13059-019-1861-6
    """
    # Convert sparse to dense if needed (for easier computation)
    if spr.issparse(count_matrix):
        X = count_matrix.toarray()
    else:
        X = count_matrix.copy()

    # Avoid log(0) and division by zero
    X = np.maximum(X, 1e-10)

    # Calculate parameters
    n_cells, n_genes = X.shape
    cell_totals = X.sum(axis=1, keepdims=True)  # n_i: total counts per cell
    gene_totals = X.sum(axis=0)  # sum of counts per gene across cells
    total_counts = X.sum()  # total counts in dataset

    # Multinomial null probabilities: p_g = (sum of gene g) / (total counts)
    p_null = gene_totals / total_counts
    p_null = np.maximum(p_null, 1e-10)  # Avoid division by zero

    # Expected counts under null: E[x_ig] = n_i * p_g
    expected = cell_totals @ p_null.reshape(1, -1)
    expected = np.maximum(expected, 1e-10)

    # Binomial deviance: 2 * x * log(x / E[x])
    # Only compute for non-zero observed counts to avoid numerical issues
    mask = X > 0
    deviance_terms = np.zeros_like(X)
    deviance_terms[mask] = 2 * X[mask] * np.log(X[mask] / expected[mask])

    # Sum deviance across cells for each gene
    deviance_scores = deviance_terms.sum(axis=0)

    return deviance_scores
