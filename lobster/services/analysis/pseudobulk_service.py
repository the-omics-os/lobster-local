"""
Pseudobulk aggregation service for single-cell RNA-seq data.

This module provides the PseudobulkService that converts single-cell RNA-seq
data into bulk-like expression matrices by aggregating counts per sample and
cell type, enabling proper differential expression analysis at the sample level.
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

from lobster.core import (
    AggregationError,
    InsufficientCellsError,
    ProvenanceError,
    PseudobulkError,
)
from lobster.core.adapters.pseudobulk_adapter import PseudobulkAdapter
from lobster.core.analysis_ir import AnalysisStep
from lobster.core.provenance import ProvenanceTracker
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PseudobulkService:
    """
    Service for pseudobulk aggregation of single-cell data.

    This service converts single-cell RNA-seq data to pseudobulk matrices by
    aggregating expression values at the sample-celltype level, following
    Lobster's modular architecture with proper provenance tracking.
    """

    def __init__(self, provenance_tracker: Optional[ProvenanceTracker] = None):
        """
        Initialize the pseudobulk service.

        Args:
            provenance_tracker: Optional provenance tracker for operation logging
        """
        self.logger = logger
        self.provenance_tracker = provenance_tracker or ProvenanceTracker(
            namespace="lobster"
        )
        self.adapter = PseudobulkAdapter(strict_validation=False)

        # Supported aggregation methods
        self.aggregation_methods = {
            "sum": self._aggregate_sum,
            "mean": self._aggregate_mean,
            "median": self._aggregate_median,
        }

    def aggregate_to_pseudobulk(
        self,
        adata: anndata.AnnData,
        sample_col: str,
        celltype_col: str,
        layer: Optional[str] = None,
        min_cells: int = 10,
        aggregation_method: str = "sum",
        min_genes: int = 200,
        filter_zeros: bool = True,
        gene_subset: Optional[List[str]] = None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Aggregate single-cell counts to pseudobulk matrix.

        Args:
            adata: Single-cell AnnData object
            sample_col: Column in obs containing sample identifiers
            celltype_col: Column in obs containing cell type identifiers
            layer: Layer to use for aggregation (default: X)
            min_cells: Minimum cells per sample-celltype combination
            aggregation_method: Method for aggregation ('sum', 'mean', 'median')
            min_genes: Minimum genes detected per pseudobulk sample
            filter_zeros: Remove genes with all zeros after aggregation
            gene_subset: Optional subset of genes to include

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
                - Pseudobulk aggregated data
                - Statistics dictionary with aggregation summary
                - AnalysisStep IR for notebook export

        Raises:
            PseudobulkError: If aggregation fails
            AggregationError: If aggregation method is invalid
            InsufficientCellsError: If insufficient cells for aggregation
        """
        # Log operation start
        self.logger.info(
            f"Starting pseudobulk aggregation: {adata.n_obs} cells → pseudobulk"
        )

        try:
            # Validate inputs
            self._validate_aggregation_inputs(
                adata, sample_col, celltype_col, aggregation_method
            )

            # Create activity for provenance tracking
            activity_id = self._start_aggregation_activity(
                adata, sample_col, celltype_col, aggregation_method, min_cells
            )

            # Create input entity
            input_entity_id = self.provenance_tracker.create_entity(
                entity_type="single_cell_data",
                metadata={
                    "n_cells": adata.n_obs,
                    "n_genes": adata.n_vars,
                    "sample_col": sample_col,
                    "celltype_col": celltype_col,
                },
            )

            # Get expression matrix
            X = self._get_expression_matrix(adata, layer)

            # Filter genes if subset provided
            if gene_subset is not None:
                adata, X = self._filter_genes(adata, X, gene_subset)

            # Group cells by sample and cell type
            grouping_df = self._create_grouping_dataframe(
                adata, sample_col, celltype_col
            )

            # Filter groups by minimum cell count
            valid_groups, filtered_stats = self._filter_groups_by_cell_count(
                grouping_df, min_cells
            )

            if len(valid_groups) == 0:
                raise InsufficientCellsError(
                    f"No sample-celltype combinations have ≥{min_cells} cells"
                )

            # Perform aggregation
            pseudobulk_matrix, group_metadata = self._perform_aggregation(
                X, adata.obs, valid_groups, aggregation_method
            )

            # Create pseudobulk AnnData
            pseudobulk_adata = self._create_pseudobulk_anndata(
                pseudobulk_matrix,
                group_metadata,
                adata.var.copy(),
                sample_col,
                celltype_col,
                aggregation_method,
                filtered_stats,
            )

            # Add aggregation parameters and statistics
            pseudobulk_adata = self._add_aggregation_metadata(
                pseudobulk_adata,
                adata,
                sample_col,
                celltype_col,
                layer,
                min_cells,
                aggregation_method,
                min_genes,
                filter_zeros,
                filtered_stats,
            )

            # Apply post-aggregation filtering
            if filter_zeros:
                pseudobulk_adata = self._filter_zero_genes(pseudobulk_adata)

            if min_genes > 0:
                pseudobulk_adata = self._filter_low_gene_samples(
                    pseudobulk_adata, min_genes
                )

            # Validate output using adapter
            pseudobulk_adata = self.adapter.from_source(
                pseudobulk_adata,
                validate_schema=True,
                aggregation_metadata={
                    "pseudobulk_params": pseudobulk_adata.uns.get(
                        "pseudobulk_params", {}
                    ),
                    "aggregation_stats": pseudobulk_adata.uns.get(
                        "aggregation_stats", {}
                    ),
                },
                original_dataset_info={
                    "n_original_cells": adata.n_obs,
                    "n_original_genes": adata.n_vars,
                    "original_modality": "single_cell_rna_seq",
                },
            )

            # Create output entity and complete provenance
            output_entity_id = self.provenance_tracker.create_entity(
                entity_type="pseudobulk_matrix",
                metadata={
                    "n_pseudobulk_samples": pseudobulk_adata.n_obs,
                    "n_genes": pseudobulk_adata.n_vars,
                    "aggregation_method": aggregation_method,
                    "total_cells_aggregated": pseudobulk_adata.uns.get(
                        "aggregation_stats", {}
                    ).get("total_cells_aggregated", 0),
                },
            )

            # Update activity with results
            self._complete_aggregation_activity(
                activity_id, input_entity_id, output_entity_id, pseudobulk_adata
            )

            # Add provenance to AnnData
            pseudobulk_adata = self.provenance_tracker.add_to_anndata(pseudobulk_adata)

            # Create IR for notebook export
            ir = self._create_pseudobulk_ir(
                sample_col=sample_col,
                celltype_col=celltype_col,
                aggregation_method=aggregation_method,
                min_cells=min_cells,
            )

            # Create statistics dictionary
            stats = {
                "n_pseudobulk_samples": pseudobulk_adata.n_obs,
                "n_genes": pseudobulk_adata.n_vars,
                "n_unique_samples": pseudobulk_adata.obs["sample_id"].nunique(),
                "n_cell_types": pseudobulk_adata.obs["cell_type"].nunique(),
                "total_cells_processed": adata.n_obs,
                "total_cells_aggregated": int(
                    pseudobulk_adata.uns.get("aggregation_stats", {}).get(
                        "total_cells_aggregated", 0
                    )
                ),
                "aggregation_method": aggregation_method,
                "cell_type_summary": pseudobulk_adata.obs["cell_type"]
                .value_counts()
                .to_dict(),
            }

            self.logger.info(
                f"Pseudobulk aggregation completed: "
                f"{pseudobulk_adata.n_obs} pseudobulk samples × {pseudobulk_adata.n_vars} genes"
            )

            return pseudobulk_adata, stats, ir

        except Exception as e:
            if isinstance(
                e, (PseudobulkError, AggregationError, InsufficientCellsError)
            ):
                self.logger.error(f"Pseudobulk aggregation failed: {e}")
                raise
            else:
                self.logger.error(f"Unexpected error in pseudobulk aggregation: {e}")
                raise PseudobulkError(f"Aggregation failed: {e}")

    def _validate_aggregation_inputs(
        self,
        adata: anndata.AnnData,
        sample_col: str,
        celltype_col: str,
        aggregation_method: str,
    ) -> None:
        """Validate inputs for aggregation."""

        if sample_col not in adata.obs.columns:
            raise AggregationError(f"Sample column '{sample_col}' not found in obs")

        if celltype_col not in adata.obs.columns:
            raise AggregationError(
                f"Cell type column '{celltype_col}' not found in obs"
            )

        if aggregation_method not in self.aggregation_methods:
            raise AggregationError(
                f"Unsupported aggregation method '{aggregation_method}'. "
                f"Supported methods: {list(self.aggregation_methods.keys())}"
            )

        # Check for missing values in grouping columns
        sample_missing = adata.obs[sample_col].isna().sum()
        celltype_missing = adata.obs[celltype_col].isna().sum()

        if sample_missing > 0:
            raise AggregationError(
                f"{sample_missing} cells have missing sample identifiers"
            )

        if celltype_missing > 0:
            raise AggregationError(
                f"{celltype_missing} cells have missing cell type identifiers"
            )

    def _start_aggregation_activity(
        self,
        adata: anndata.AnnData,
        sample_col: str,
        celltype_col: str,
        aggregation_method: str,
        min_cells: int,
    ) -> str:
        """Start provenance activity for aggregation."""

        try:
            activity_id = self.provenance_tracker.create_activity(
                activity_type="pseudobulk_aggregation",
                agent="PseudobulkService",
                parameters={
                    "sample_col": sample_col,
                    "celltype_col": celltype_col,
                    "aggregation_method": aggregation_method,
                    "min_cells": min_cells,
                    "input_n_cells": adata.n_obs,
                    "input_n_genes": adata.n_vars,
                },
                description=f"Aggregating {adata.n_obs} cells to pseudobulk using {aggregation_method} method",
            )
            return activity_id
        except Exception as e:
            raise ProvenanceError(f"Failed to create aggregation activity: {e}")

    def _get_expression_matrix(
        self, adata: anndata.AnnData, layer: Optional[str]
    ) -> Union[np.ndarray, sparse.spmatrix]:
        """Get expression matrix from AnnData."""

        if layer is None:
            return adata.X
        elif layer in adata.layers:
            return adata.layers[layer]
        else:
            raise AggregationError(f"Layer '{layer}' not found in AnnData")

    def _filter_genes(
        self,
        adata: anndata.AnnData,
        X: Union[np.ndarray, sparse.spmatrix],
        gene_subset: List[str],
    ) -> Tuple[anndata.AnnData, Union[np.ndarray, sparse.spmatrix]]:
        """Filter genes to subset."""

        # Find genes present in both subset and data
        available_genes = set(adata.var_names)
        requested_genes = set(gene_subset)
        valid_genes = list(requested_genes.intersection(available_genes))
        missing_genes = list(requested_genes - available_genes)

        if missing_genes:
            self.logger.warning(f"Requested genes not found: {missing_genes[:10]}...")

        if not valid_genes:
            raise AggregationError("No requested genes found in dataset")

        # Filter AnnData and matrix
        gene_mask = adata.var_names.isin(valid_genes)
        adata_filtered = adata[:, gene_mask].copy()
        X_filtered = X[:, gene_mask]

        self.logger.info(
            f"Filtered to {len(valid_genes)} genes from {len(gene_subset)} requested"
        )

        return adata_filtered, X_filtered

    def _create_grouping_dataframe(
        self, adata: anndata.AnnData, sample_col: str, celltype_col: str
    ) -> pd.DataFrame:
        """Create DataFrame for grouping cells by sample and cell type."""

        grouping_df = pd.DataFrame(
            {
                "cell_idx": range(adata.n_obs),
                "sample_id": adata.obs[sample_col].values,
                "cell_type": adata.obs[celltype_col].values,
            }
        )

        # Create composite group identifier
        grouping_df["group_id"] = (
            grouping_df["sample_id"].astype(str)
            + "_"
            + grouping_df["cell_type"].astype(str)
        )

        return grouping_df

    def _filter_groups_by_cell_count(
        self, grouping_df: pd.DataFrame, min_cells: int
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Filter groups by minimum cell count."""

        # Count cells per group
        group_counts = grouping_df.groupby("group_id").size()

        # Filter groups
        valid_groups = group_counts[group_counts >= min_cells]
        excluded_groups = group_counts[group_counts < min_cells]

        # Create statistics
        stats = {
            "total_groups": len(group_counts),
            "valid_groups": len(valid_groups),
            "excluded_groups": len(excluded_groups),
            "excluded_group_names": list(excluded_groups.index),
            "min_cells_threshold": min_cells,
            "cells_per_group": group_counts.to_dict(),
        }

        if excluded_groups.empty:
            self.logger.info(f"All {len(valid_groups)} groups have ≥{min_cells} cells")
        else:
            self.logger.warning(
                f"Excluded {len(excluded_groups)} groups with <{min_cells} cells: "
                f"{list(excluded_groups.index)[:5]}..."
            )

        # Filter grouping dataframe to valid groups
        valid_grouping_df = grouping_df[
            grouping_df["group_id"].isin(valid_groups.index)
        ]

        return valid_grouping_df, stats

    def _perform_aggregation(
        self,
        X: Union[np.ndarray, sparse.spmatrix],
        obs: pd.DataFrame,
        grouping_df: pd.DataFrame,
        aggregation_method: str,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Perform the actual aggregation."""

        self.logger.info(f"Performing {aggregation_method} aggregation...")

        # Get unique groups and their cell indices
        groups = grouping_df.groupby("group_id")["cell_idx"].apply(list)

        # Initialize output matrix
        n_groups = len(groups)
        n_genes = X.shape[1]
        aggregated_matrix = np.zeros((n_groups, n_genes), dtype=np.float32)

        # Initialize metadata for each group
        group_metadata = []

        # Perform aggregation for each group
        aggregation_func = self.aggregation_methods[aggregation_method]

        for i, (group_id, cell_indices) in enumerate(groups.items()):
            # Extract cells for this group
            group_cells = np.array(cell_indices)

            if sparse.issparse(X):
                group_X = X[group_cells, :].toarray()
            else:
                group_X = X[group_cells, :]

            # Aggregate expression values
            aggregated_matrix[i, :] = aggregation_func(group_X)

            # Collect metadata for this group
            sample_id, cell_type = group_id.rsplit("_", 1)
            first_cell_obs = obs.iloc[cell_indices[0]]

            metadata = {
                "sample_id": sample_id,
                "cell_type": cell_type,
                "n_cells_aggregated": len(cell_indices),
                "pseudobulk_sample_id": group_id,
            }

            # Add additional metadata from original obs (take from first cell)
            for col in obs.columns:
                if col not in ["sample_id", "cell_type"] and col not in metadata:
                    metadata[col] = first_cell_obs[col]

            group_metadata.append(metadata)

        # Create metadata DataFrame
        group_metadata_df = pd.DataFrame(group_metadata)
        group_metadata_df.index = [
            f"{row['sample_id']}_{row['cell_type']}"
            for _, row in group_metadata_df.iterrows()
        ]

        self.logger.info(
            f"Aggregated {len(cell_indices)} cells into {n_groups} pseudobulk samples"
        )

        return aggregated_matrix, group_metadata_df

    def _aggregate_sum(self, X: np.ndarray) -> np.ndarray:
        """Sum aggregation method."""
        return np.sum(X, axis=0)

    def _aggregate_mean(self, X: np.ndarray) -> np.ndarray:
        """Mean aggregation method."""
        return np.mean(X, axis=0)

    def _aggregate_median(self, X: np.ndarray) -> np.ndarray:
        """Median aggregation method."""
        return np.median(X, axis=0)

    def _create_pseudobulk_anndata(
        self,
        pseudobulk_matrix: np.ndarray,
        group_metadata: pd.DataFrame,
        var_metadata: pd.DataFrame,
        sample_col: str,
        celltype_col: str,
        aggregation_method: str,
        filtered_stats: Dict[str, Any],
    ) -> anndata.AnnData:
        """Create AnnData object for pseudobulk data."""

        # Create AnnData object
        pseudobulk_adata = anndata.AnnData(
            X=pseudobulk_matrix, obs=group_metadata, var=var_metadata
        )

        # Ensure required columns are present
        if "sample_id" not in pseudobulk_adata.obs.columns:
            pseudobulk_adata.obs["sample_id"] = [
                idx.rsplit("_", 1)[0] for idx in pseudobulk_adata.obs.index
            ]

        if "cell_type" not in pseudobulk_adata.obs.columns:
            pseudobulk_adata.obs["cell_type"] = [
                idx.rsplit("_", 1)[1] for idx in pseudobulk_adata.obs.index
            ]

        # Add aggregation method to each observation
        pseudobulk_adata.obs["aggregation_method"] = aggregation_method

        return pseudobulk_adata

    def _add_aggregation_metadata(
        self,
        pseudobulk_adata: anndata.AnnData,
        original_adata: anndata.AnnData,
        sample_col: str,
        celltype_col: str,
        layer: Optional[str],
        min_cells: int,
        aggregation_method: str,
        min_genes: int,
        filter_zeros: bool,
        filtered_stats: Dict[str, Any],
    ) -> anndata.AnnData:
        """Add aggregation parameters and statistics to uns."""

        # Aggregation parameters
        pseudobulk_adata.uns["pseudobulk_params"] = {
            "sample_col": sample_col,
            "celltype_col": celltype_col,
            "layer": layer,
            "min_cells": min_cells,
            "aggregation_method": aggregation_method,
            "min_genes": min_genes,
            "filter_zeros": filter_zeros,
            "original_n_cells": original_adata.n_obs,
            "original_n_genes": original_adata.n_vars,
            "aggregation_timestamp": datetime.datetime.now().isoformat(),
        }

        # Aggregation statistics
        sample_counts = pseudobulk_adata.obs["sample_id"].value_counts().to_dict()
        celltype_counts = pseudobulk_adata.obs["cell_type"].value_counts().to_dict()

        # Handle missing n_cells_aggregated column (safety check)
        if "n_cells_aggregated" in pseudobulk_adata.obs.columns:
            total_cells = pseudobulk_adata.obs["n_cells_aggregated"].sum()
            mean_cells_per_pseudobulk = float(
                pseudobulk_adata.obs["n_cells_aggregated"].mean()
            )
        else:
            # Fallback: estimate from filtered_stats or use default
            total_cells = filtered_stats.get(
                "total_cells_processed", pseudobulk_adata.n_obs * min_cells
            )
            mean_cells_per_pseudobulk = float(
                total_cells / max(1, pseudobulk_adata.n_obs)
            )

        pseudobulk_adata.uns["aggregation_stats"] = {
            "n_samples": pseudobulk_adata.obs["sample_id"].nunique(),
            "n_cell_types": pseudobulk_adata.obs["cell_type"].nunique(),
            "n_pseudobulk_samples": pseudobulk_adata.n_obs,
            "total_cells_aggregated": int(total_cells),
            "cells_per_sample": sample_counts,
            "cells_per_celltype": celltype_counts,
            "mean_cells_per_pseudobulk": mean_cells_per_pseudobulk,
            "min_cells_threshold": min_cells,
            **filtered_stats,
        }

        return pseudobulk_adata

    def _filter_zero_genes(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Remove genes with all zeros."""

        gene_sums = np.array(adata.X.sum(axis=0)).flatten()
        non_zero_genes = gene_sums > 0
        n_zero_genes = (~non_zero_genes).sum()

        if n_zero_genes > 0:
            self.logger.info(f"Filtering {n_zero_genes} genes with zero expression")
            adata = adata[:, non_zero_genes].copy()

        return adata

    def _filter_low_gene_samples(
        self, adata: anndata.AnnData, min_genes: int
    ) -> anndata.AnnData:
        """Remove pseudobulk samples with too few genes."""

        genes_per_sample = (adata.X > 0).sum(axis=1)
        valid_samples = genes_per_sample >= min_genes
        n_filtered = (~valid_samples).sum()

        if n_filtered > 0:
            self.logger.warning(
                f"Filtering {n_filtered} pseudobulk samples with <{min_genes} genes"
            )
            adata = adata[valid_samples, :].copy()

        return adata

    def _complete_aggregation_activity(
        self,
        activity_id: str,
        input_entity_id: str,
        output_entity_id: str,
        result_adata: anndata.AnnData,
    ) -> None:
        """Complete provenance activity with results."""

        try:
            # Update activity with input/output entities
            for activity in self.provenance_tracker.activities:
                if activity["id"] == activity_id:
                    activity["inputs"] = [
                        {"entity": input_entity_id, "role": "single_cell_data"}
                    ]
                    activity["outputs"] = [
                        {"entity": output_entity_id, "role": "pseudobulk_matrix"}
                    ]
                    activity["result_summary"] = {
                        "n_pseudobulk_samples": result_adata.n_obs,
                        "n_genes": result_adata.n_vars,
                        "total_cells_aggregated": result_adata.uns.get(
                            "aggregation_stats", {}
                        ).get("total_cells_aggregated", 0),
                    }
                    break
        except Exception as e:
            self.logger.warning(f"Failed to complete provenance activity: {e}")

    def _create_pseudobulk_ir(
        self,
        sample_col: str,
        celltype_col: str,
        aggregation_method: str,
        min_cells: int,
        **kwargs,
    ) -> AnalysisStep:
        """Create AnalysisStep IR for pseudobulk aggregation."""

        code_template = """
# Pseudobulk aggregation from single-cell data
import scanpy as sc
import pandas as pd
import numpy as np

# Group cells by sample and cell type
groups = adata.obs.groupby(['{{ sample_col }}', '{{ celltype_col }}']).groups

# Aggregate expression for each group
pseudobulk_data = []
pseudobulk_obs = []

for (sample, celltype), indices in groups.items():
    if len(indices) >= {{ min_cells }}:
        # Get cells for this group
        group_adata = adata[indices]

        # Aggregate expression
        {% if aggregation_method == 'sum' %}
        aggregated = group_adata.X.sum(axis=0)
        {% elif aggregation_method == 'mean' %}
        aggregated = group_adata.X.mean(axis=0)
        {% elif aggregation_method == 'median' %}
        aggregated = np.median(group_adata.X, axis=0)
        {% endif %}

        pseudobulk_data.append(aggregated)
        pseudobulk_obs.append({
            'sample_id': sample,
            'cell_type': celltype,
            'n_cells_aggregated': len(indices)
        })

# Create pseudobulk AnnData
pseudobulk_adata = sc.AnnData(
    X=np.array(pseudobulk_data),
    obs=pd.DataFrame(pseudobulk_obs),
    var=adata.var.copy()
)
"""

        return AnalysisStep(
            operation="pseudobulk_aggregation",
            tool_name="PseudobulkService.aggregate_to_pseudobulk",
            description=f"Aggregate single-cell data to pseudobulk using {aggregation_method} method",
            library="scanpy + numpy",
            code_template=code_template,
            imports=[
                "import scanpy as sc",
                "import pandas as pd",
                "import numpy as np",
            ],
            parameters={
                "sample_col": sample_col,
                "celltype_col": celltype_col,
                "aggregation_method": aggregation_method,
                "min_cells": min_cells,
            },
            parameter_schema={
                "sample_col": {
                    "type": "string",
                    "description": "Column in obs containing sample IDs",
                },
                "celltype_col": {
                    "type": "string",
                    "description": "Column in obs containing cell type annotations",
                },
                "aggregation_method": {
                    "type": "string",
                    "description": "Aggregation method",
                    "default": "sum",
                    "enum": ["sum", "mean", "median"],
                },
                "min_cells": {
                    "type": "integer",
                    "description": "Minimum cells required per pseudobulk sample",
                    "default": 10,
                },
            },
            input_entities=["singlecell_adata"],
            output_entities=["pseudobulk_adata"],
        )

    def get_aggregation_summary(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """
        Get summary of pseudobulk aggregation.

        Args:
            adata: Pseudobulk AnnData object

        Returns:
            Dict[str, Any]: Aggregation summary
        """
        if "aggregation_stats" not in adata.uns:
            return {"error": "No aggregation statistics found in data"}

        stats = adata.uns["aggregation_stats"].copy()

        # Add quality metrics
        quality_metrics = self.adapter.get_quality_metrics(adata)
        stats["quality_metrics"] = quality_metrics

        return stats

    def export_for_deseq2(
        self, adata: anndata.AnnData, output_dir: str, count_layer: str = None
    ) -> Dict[str, str]:
        """
        Export pseudobulk data for DESeq2 analysis.

        Args:
            adata: Pseudobulk AnnData object
            output_dir: Output directory for files
            count_layer: Layer to use for counts (default: X)

        Returns:
            Dict[str, str]: Paths to exported files
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get count matrix
        if count_layer and count_layer in adata.layers:
            count_matrix = adata.layers[count_layer]
        else:
            count_matrix = adata.X

        # Export count matrix (genes as rows, samples as columns - DESeq2 format)
        count_df = pd.DataFrame(
            count_matrix.T, index=adata.var_names, columns=adata.obs.index
        )
        count_file = output_path / "pseudobulk_counts.csv"
        count_df.to_csv(count_file)

        # Export sample metadata
        metadata_file = output_path / "sample_metadata.csv"
        adata.obs.to_csv(metadata_file)

        # Export gene metadata
        gene_file = output_path / "gene_metadata.csv"
        adata.var.to_csv(gene_file)

        self.logger.info(f"Exported pseudobulk data to {output_dir}")

        return {
            "counts": str(count_file),
            "sample_metadata": str(metadata_file),
            "gene_metadata": str(gene_file),
        }
