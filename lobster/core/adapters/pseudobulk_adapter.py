"""
Pseudobulk data adapter with schema enforcement.

This module provides the PseudobulkAdapter that handles loading,
validation, and preprocessing of pseudobulk aggregated data with
appropriate schema enforcement.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata
import numpy as np
import pandas as pd

from lobster.core import AdapterError, ValidationError
from lobster.core.adapters.base import BaseAdapter
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.pseudobulk import PseudobulkSchema

logger = logging.getLogger(__name__)


class PseudobulkAdapter(BaseAdapter):
    """
    Adapter for pseudobulk aggregated data with schema enforcement.

    This adapter handles loading and validation of pseudobulk matrices created
    by aggregating single-cell RNA-seq data at the sample and cell type level,
    enabling proper differential expression analysis.
    """

    def __init__(self, strict_validation: bool = False):
        """
        Initialize the pseudobulk adapter.

        Args:
            strict_validation: Whether to use strict validation
        """
        super().__init__(name="PseudobulkAdapter")

        self.strict_validation = strict_validation

        # Create validator
        self.validator = PseudobulkSchema.create_validator(strict=strict_validation)

        # Get QC thresholds
        self.qc_thresholds = PseudobulkSchema.get_recommended_qc_thresholds()

    def from_source(
        self, source: Union[str, Path, pd.DataFrame, anndata.AnnData], **kwargs
    ) -> anndata.AnnData:
        """
        Convert source data to AnnData with pseudobulk schema.

        Args:
            source: Data source (AnnData object, file path, or DataFrame)
            **kwargs: Additional parameters:
                - validate_schema: Whether to validate pseudobulk schema (default: True)
                - aggregation_metadata: Dict with aggregation parameters and stats
                - original_dataset_info: Dict with original single-cell dataset info

        Returns:
            anndata.AnnData: Loaded and validated pseudobulk data

        Raises:
            AdapterError: If source data is invalid
            ValidationError: If schema validation fails
        """
        self._log_operation("loading pseudobulk data", source=str(source))

        try:
            # Extract metadata fields that should be stored in uns
            validate_schema = kwargs.get("validate_schema", True)
            aggregation_metadata = kwargs.get("aggregation_metadata", {})
            original_dataset_info = kwargs.get("original_dataset_info", {})

            # Handle different source types
            if isinstance(source, anndata.AnnData):
                adata = source.copy()
            elif isinstance(source, pd.DataFrame):
                # Create AnnData from DataFrame (assuming it's already pseudobulk format)
                adata = self._create_pseudobulk_from_dataframe(source, **kwargs)
            elif isinstance(source, (str, Path)):
                adata = self._load_pseudobulk_from_file(source, **kwargs)
            else:
                raise AdapterError(f"Unsupported source type: {type(source)}")

            # Add pseudobulk-specific metadata
            adata = self._add_pseudobulk_metadata(
                adata,
                aggregation_metadata=aggregation_metadata,
                original_dataset_info=original_dataset_info,
            )

            # Apply pseudobulk-specific preprocessing
            adata = self.preprocess_data(adata, **kwargs)

            # Validate schema if requested
            if validate_schema:
                validation_result = self.validate(adata, strict=self.strict_validation)
                if validation_result.has_errors():
                    if self.strict_validation:
                        raise ValidationError(
                            f"Pseudobulk schema validation failed: {validation_result.get_error_summary()}",
                            details={"validation_result": validation_result.to_dict()},
                        )
                    else:
                        self.logger.warning(
                            f"Pseudobulk validation warnings: {validation_result.get_warning_summary()}"
                        )

            self.logger.info(
                f"Loaded pseudobulk data: {adata.n_obs} pseudobulk samples × {adata.n_vars} genes"
            )
            return adata

        except Exception as e:
            if isinstance(e, (AdapterError, ValidationError)):
                raise
            else:
                raise AdapterError(f"Failed to load pseudobulk data from {source}: {e}")

    def _load_pseudobulk_from_file(
        self, path: Union[str, Path], **kwargs
    ) -> anndata.AnnData:
        """Load pseudobulk data from file with format detection."""
        path = Path(path)

        if not path.exists():
            raise AdapterError(f"File not found: {path}")

        format_type = self.detect_format(path)

        try:
            if format_type == "h5ad":
                return self._load_h5ad_data(path)
            elif format_type in ["csv", "tsv", "txt"]:
                return self._load_csv_pseudobulk_data(path, **kwargs)
            elif format_type in ["xlsx", "xls"]:
                return self._load_excel_pseudobulk_data(path, **kwargs)
            else:
                raise AdapterError(
                    f"Unsupported file format for pseudobulk data: {format_type}"
                )
        except Exception as e:
            raise AdapterError(f"Failed to load pseudobulk data from {path}: {e}")

    def _load_csv_pseudobulk_data(
        self, path: Union[str, Path], **kwargs
    ) -> anndata.AnnData:
        """Load pseudobulk data from CSV/TSV."""
        # Load the expression matrix
        df = self._load_csv_data(path, index_col=0, **kwargs)

        # Create pseudobulk AnnData from DataFrame
        return self._create_pseudobulk_from_dataframe(df, **kwargs)

    def _load_excel_pseudobulk_data(
        self, path: Union[str, Path], **kwargs
    ) -> anndata.AnnData:
        """Load pseudobulk data from Excel file."""
        sheet_name = kwargs.get("sheet_name", 0)
        df = self._load_excel_data(path, sheet_name=sheet_name, index_col=0)

        return self._create_pseudobulk_from_dataframe(df, **kwargs)

    def _create_pseudobulk_from_dataframe(
        self, df: pd.DataFrame, **kwargs
    ) -> anndata.AnnData:
        """Create pseudobulk AnnData from DataFrame."""

        # Parse pseudobulk sample identifiers if they follow the pattern: sample_id_cell_type
        obs_metadata = self._parse_pseudobulk_sample_ids(df.index)

        # Add any additional obs metadata provided
        additional_obs = kwargs.get("obs_metadata")
        if additional_obs is not None:
            obs_metadata = pd.concat([obs_metadata, additional_obs], axis=1)

        # Create AnnData
        adata = self._create_anndata_from_dataframe(
            df,
            obs_metadata=obs_metadata,
            var_metadata=kwargs.get("var_metadata"),
            transpose=kwargs.get("transpose", False),
        )

        return adata

    def _parse_pseudobulk_sample_ids(self, sample_ids: pd.Index) -> pd.DataFrame:
        """
        Parse pseudobulk sample identifiers to extract sample_id and cell_type.

        Expects format: {sample_id}_{cell_type}

        Args:
            sample_ids: Index of sample identifiers

        Returns:
            pd.DataFrame: Parsed metadata with sample_id and cell_type columns
        """
        obs_data = []

        for sample_id in sample_ids:
            # Try to parse sample_id_cell_type format
            parts = str(sample_id).rsplit(
                "_", 1
            )  # Split from right to handle underscores in sample names

            if len(parts) == 2:
                sample, cell_type = parts
                obs_data.append(
                    {
                        "sample_id": sample,
                        "cell_type": cell_type,
                        "pseudobulk_sample_id": sample_id,
                    }
                )
            else:
                # If parsing fails, use the full identifier as both sample and create generic cell type
                obs_data.append(
                    {
                        "sample_id": sample_id,
                        "cell_type": "unknown",
                        "pseudobulk_sample_id": sample_id,
                    }
                )

        return pd.DataFrame(obs_data, index=sample_ids)

    def _add_pseudobulk_metadata(
        self,
        adata: anndata.AnnData,
        aggregation_metadata: Dict[str, Any] = None,
        original_dataset_info: Dict[str, Any] = None,
    ) -> anndata.AnnData:
        """Add pseudobulk-specific metadata to AnnData object."""

        # Add aggregation parameters if provided
        if aggregation_metadata:
            if "pseudobulk_params" in aggregation_metadata:
                adata.uns["pseudobulk_params"] = aggregation_metadata[
                    "pseudobulk_params"
                ]

            if "aggregation_stats" in aggregation_metadata:
                adata.uns["aggregation_stats"] = aggregation_metadata[
                    "aggregation_stats"
                ]

        # Add original dataset info if provided
        if original_dataset_info:
            adata.uns["original_dataset_info"] = original_dataset_info

        # Calculate pseudobulk-specific metrics if not already present
        adata = self._calculate_pseudobulk_metrics(adata)

        return adata

    def _calculate_pseudobulk_metrics(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Calculate pseudobulk-specific metrics."""

        # Add per-gene metrics for pseudobulk samples
        if "n_pseudobulk_samples" not in adata.var.columns:
            adata.var["n_pseudobulk_samples"] = np.array((adata.X > 0).sum(axis=0)).flatten()

        if "mean_aggregated_counts" not in adata.var.columns:
            adata.var["mean_aggregated_counts"] = np.array(
                adata.X.mean(axis=0)
            ).flatten()

        if "total_aggregated_counts" not in adata.var.columns:
            adata.var["total_aggregated_counts"] = np.array(
                adata.X.sum(axis=0)
            ).flatten()

        # Add per-pseudobulk-sample metrics
        if "n_genes_detected" not in adata.obs.columns:
            adata.obs["n_genes_detected"] = np.array((adata.X > 0).sum(axis=1)).flatten()

        if "total_aggregated_counts" not in adata.obs.columns:
            adata.obs["total_aggregated_counts"] = np.array(
                adata.X.sum(axis=1)
            ).flatten()

        return adata

    def validate(self, adata: anndata.AnnData, strict: bool = None) -> ValidationResult:
        """
        Validate AnnData against pseudobulk schema.

        Args:
            adata: AnnData object to validate
            strict: Override default strict setting

        Returns:
            ValidationResult: Validation results
        """
        if strict is None:
            strict = self.strict_validation

        # Use the configured validator
        result = self.validator.validate(adata, strict=strict)

        # Add basic structural validation
        basic_result = self._validate_basic_structure(adata)
        result = result.merge(basic_result)

        # Add pseudobulk-specific validation
        pseudobulk_result = self._validate_pseudobulk_specific(adata)
        result = result.merge(pseudobulk_result)

        return result

    def _validate_pseudobulk_specific(self, adata: anndata.AnnData) -> ValidationResult:
        """Perform pseudobulk-specific validation."""
        result = ValidationResult()

        # Check for reasonable number of pseudobulk samples
        if adata.n_obs < self.qc_thresholds["min_pseudobulk_samples"]:
            result.add_warning(
                f"Only {adata.n_obs} pseudobulk samples "
                f"(recommended minimum: {self.qc_thresholds['min_pseudobulk_samples']})"
            )

        # Check cell counts if available
        if "n_cells_aggregated" in adata.obs.columns:
            cell_counts = adata.obs["n_cells_aggregated"]
            low_cell_samples = (
                cell_counts < self.qc_thresholds["min_cells_per_pseudobulk"]
            ).sum()

            if low_cell_samples > 0:
                result.add_warning(
                    f"{low_cell_samples} pseudobulk samples have <{self.qc_thresholds['min_cells_per_pseudobulk']} cells"
                )

        # Check for balanced design if condition information is available
        if all(
            col in adata.obs.columns for col in ["sample_id", "cell_type", "condition"]
        ):
            self._check_experimental_balance(adata, result)

        return result

    def _check_experimental_balance(
        self, adata: anndata.AnnData, result: ValidationResult
    ):
        """Check experimental design balance."""
        design_counts = (
            adata.obs.groupby(["condition", "cell_type"]).size().unstack(fill_value=0)
        )

        # Check if all condition-celltype combinations have samples
        zero_combinations = (design_counts == 0).sum().sum()
        if zero_combinations > 0:
            result.add_warning(
                f"{zero_combinations} condition-celltype combinations have no samples"
            )

        # Check for minimum replication
        min_samples_per_group = design_counts[design_counts > 0].min().min()
        if min_samples_per_group < self.qc_thresholds["min_samples_per_celltype"]:
            result.add_warning(
                f"Minimum samples per group: {min_samples_per_group} "
                f"(recommended: {self.qc_thresholds['min_samples_per_celltype']})"
            )

    def get_schema(self) -> Dict[str, Any]:
        """
        Return the expected schema for pseudobulk data.

        Returns:
            Dict[str, Any]: Schema definition
        """
        return PseudobulkSchema.get_pseudobulk_schema()

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.

        Returns:
            List[str]: List of supported file extensions
        """
        return ["h5ad", "csv", "tsv", "txt", "xlsx", "xls"]

    def preprocess_data(self, adata: anndata.AnnData, **kwargs) -> anndata.AnnData:
        """
        Apply pseudobulk-specific preprocessing steps.

        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters

        Returns:
            anndata.AnnData: Preprocessed data object
        """
        # Apply base preprocessing
        adata = super().preprocess_data(adata, **kwargs)

        # Add pseudobulk-specific metadata if not already present
        adata = self._calculate_pseudobulk_metrics(adata)

        # Store raw aggregated counts in layers if not present
        if "raw_aggregated" not in adata.layers:
            adata.layers["raw_aggregated"] = adata.X.copy()

        return adata

    def get_quality_metrics(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """
        Calculate pseudobulk-specific quality metrics.

        Args:
            adata: AnnData object to analyze

        Returns:
            Dict[str, Any]: Quality metrics dictionary
        """
        metrics = super().get_quality_metrics(adata)

        # Add pseudobulk-specific metrics
        if "n_cells_aggregated" in adata.obs.columns:
            cell_counts = adata.obs["n_cells_aggregated"]
            metrics.update(
                {
                    "total_cells_aggregated": int(cell_counts.sum()),
                    "mean_cells_per_pseudobulk": float(cell_counts.mean()),
                    "min_cells_per_pseudobulk": int(cell_counts.min()),
                    "max_cells_per_pseudobulk": int(cell_counts.max()),
                    "low_cell_pseudobulk_samples": int((cell_counts < 10).sum()),
                }
            )

        # Sample and cell type diversity
        if "sample_id" in adata.obs.columns:
            metrics["n_unique_samples"] = adata.obs["sample_id"].nunique()

        if "cell_type" in adata.obs.columns:
            metrics["n_cell_types"] = adata.obs["cell_type"].nunique()
            metrics["cell_types"] = list(adata.obs["cell_type"].unique())

        # Experimental design metrics
        if "condition" in adata.obs.columns:
            metrics["n_conditions"] = adata.obs["condition"].nunique()
            metrics["conditions"] = list(adata.obs["condition"].unique())

        # Gene expression metrics
        if "n_pseudobulk_samples" in adata.var.columns:
            n_samples_expressing = adata.var["n_pseudobulk_samples"]
            metrics.update(
                {
                    "genes_expressed_all_samples": int(
                        (n_samples_expressing == adata.n_obs).sum()
                    ),
                    "genes_expressed_half_samples": int(
                        (n_samples_expressing >= adata.n_obs / 2).sum()
                    ),
                    "mean_samples_per_gene": float(n_samples_expressing.mean()),
                }
            )

        return metrics

    def create_aggregated_data(
        self,
        single_cell_adata: anndata.AnnData,
        sample_col: str,
        celltype_col: str,
        min_cells: int = 10,
        aggregation_method: str = "sum",
    ) -> anndata.AnnData:
        """
        Create pseudobulk data from single-cell AnnData (convenience method).

        This is a convenience method that can be called directly on the adapter.
        The main aggregation logic should be in the PseudobulkService.

        Args:
            single_cell_adata: Single-cell AnnData object
            sample_col: Column name for sample identifiers
            celltype_col: Column name for cell type identifiers
            min_cells: Minimum cells per group
            aggregation_method: Aggregation method ('sum', 'mean', 'median')

        Returns:
            anndata.AnnData: Pseudobulk aggregated data

        Raises:
            AdapterError: If aggregation fails
        """
        try:
            # This would typically call the PseudobulkService
            # For now, we'll just validate inputs and raise an informative error

            if sample_col not in single_cell_adata.obs.columns:
                raise AdapterError(f"Sample column '{sample_col}' not found in obs")

            if celltype_col not in single_cell_adata.obs.columns:
                raise AdapterError(
                    f"Cell type column '{celltype_col}' not found in obs"
                )

            raise AdapterError(
                "Direct aggregation not implemented in adapter. "
                "Please use PseudobulkService.aggregate_to_pseudobulk() method."
            )

        except Exception as e:
            if isinstance(e, AdapterError):
                raise
            else:
                raise AdapterError(f"Failed to create pseudobulk data: {e}")
