"""
Base adapter implementation with common functionality.

This module provides the BaseAdapter class that implements common
functionality shared across all modality adapters, including
data validation, format detection, and provenance tracking.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from lobster.core.interfaces.adapter import IModalityAdapter
from lobster.core.interfaces.validator import ValidationResult

logger = logging.getLogger(__name__)


class BaseAdapter(IModalityAdapter):
    """
    Base implementation of modality adapter with common functionality.

    This class provides shared functionality for all adapter implementations
    including data loading, format detection, and basic validation.
    Subclasses need only implement the modality-specific methods.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the base adapter.

        Args:
            name: Optional name for this adapter instance
        """
        self.name = name or self.__class__.__name__
        self.logger = logger

    def _load_csv_data(
        self, path: Union[str, Path], sep: str = None, index_col: int = 0, **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file with automatic delimiter detection.

        Args:
            path: Path to CSV file
            sep: Column separator (auto-detected if None)
            index_col: Column to use as row index
            **kwargs: Additional pandas.read_csv parameters

        Returns:
            pd.DataFrame: Loaded data
        """
        path = Path(path)

        # Auto-detect separator if not provided
        if sep is None:
            with open(path, "r") as f:
                first_line = f.readline()
                if "\t" in first_line:
                    sep = "\t"
                elif "," in first_line:
                    sep = ","
                else:
                    sep = ","  # Default fallback

        try:
            df = pd.read_csv(path, sep=sep, index_col=index_col, **kwargs)
            self.logger.info(f"Loaded CSV data from {path}: shape {df.shape}")
            return df
        except Exception as e:
            raise ValueError(f"Failed to load CSV data from {path}: {e}")

    def _load_excel_data(
        self,
        path: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        index_col: int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from Excel file.

        Args:
            path: Path to Excel file
            sheet_name: Sheet name or index
            index_col: Column to use as row index
            **kwargs: Additional pandas.read_excel parameters

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            df = pd.read_excel(
                path, sheet_name=sheet_name, index_col=index_col, **kwargs
            )
            self.logger.info(f"Loaded Excel data from {path}: shape {df.shape}")
            return df
        except Exception as e:
            raise ValueError(f"Failed to load Excel data from {path}: {e}")

    def _load_h5ad_data(self, path: Union[str, Path]) -> anndata.AnnData:
        """
        Load data from H5AD file.

        Args:
            path: Path to H5AD file

        Returns:
            anndata.AnnData: Loaded AnnData object
        """
        try:
            adata = anndata.read_h5ad(path)
            # adata = sc.read_h5ad(path)
            self.logger.info(
                f"Loaded H5AD data from {path}: {adata.n_obs} obs × {adata.n_vars} vars"
            )
            return adata
        except Exception as e:
            raise ValueError(f"Failed to load H5AD data from {path}: {e}")

    def _create_anndata_from_dataframe(
        self,
        df: pd.DataFrame,
        obs_metadata: Optional[pd.DataFrame] = None,
        var_metadata: Optional[pd.DataFrame] = None,
        transpose: bool = False,
    ) -> anndata.AnnData:
        """
        Create AnnData object from pandas DataFrame.

        Args:
            df: Expression matrix DataFrame
            obs_metadata: Optional observation metadata
            var_metadata: Optional variable metadata
            transpose: Whether to transpose the matrix (genes as rows)

        Returns:
            anndata.AnnData: Created AnnData object
        """
        try:
            # Transpose if needed (common for genomics data where genes are rows)
            if transpose:
                df = df.T

            # Ensure numeric data
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] != df.shape[1]:
                self.logger.warning(
                    f"Dropping {df.shape[1] - numeric_df.shape[1]} non-numeric columns"
                )
                df = numeric_df

            # Create basic AnnData object
            adata = anndata.AnnData(
                X=df.values.astype(np.float32),
                obs=(
                    obs_metadata
                    if obs_metadata is not None
                    else pd.DataFrame(index=df.index)
                ),
                var=(
                    var_metadata
                    if var_metadata is not None
                    else pd.DataFrame(index=df.columns)
                ),
            )

            self.logger.info(
                f"Created AnnData: {adata.n_obs} obs × {adata.n_vars} vars"
            )
            return adata

        except Exception as e:
            raise ValueError(f"Failed to create AnnData from DataFrame: {e}")

    def _ensure_numeric_matrix(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Ensure the expression matrix contains only numeric values.

        Args:
            adata: AnnData object to process

        Returns:
            anndata.AnnData: AnnData with numeric matrix
        """
        try:
            # Convert to float32 if needed
            if not np.issubdtype(adata.X.dtype, np.floating):
                adata.X = adata.X.astype(np.float32)

            # Handle NaN values
            if hasattr(adata.X, "isnan"):
                nan_count = np.isnan(adata.X).sum()
                if nan_count > 0:
                    self.logger.warning(f"Found {nan_count} NaN values, filling with 0")
                    adata.X = np.nan_to_num(adata.X, nan=0.0)

            return adata
        except Exception as e:
            raise ValueError(f"Failed to ensure numeric matrix: {e}")

    def _add_basic_metadata(
        self, adata: anndata.AnnData, source_path: Optional[Union[str, Path]] = None
    ) -> anndata.AnnData:
        """
        Add basic metadata to AnnData object.

        Args:
            adata: AnnData object to annotate
            source_path: Optional path to source file

        Returns:
            anndata.AnnData: AnnData with basic metadata
        """
        # Add basic observation metadata if missing
        if "n_genes" not in adata.obs.columns:
            adata.obs["n_genes"] = (adata.X > 0).sum(axis=1)

        if "total_counts" not in adata.obs.columns:
            adata.obs["total_counts"] = adata.X.sum(axis=1)

        # Add basic variable metadata if missing
        if "n_cells" not in adata.var.columns:
            adata.var["n_cells"] = (adata.X > 0).sum(axis=0)

        if "mean_counts" not in adata.var.columns:
            adata.var["mean_counts"] = adata.X.mean(axis=0)

        # Add source information to uns
        if source_path:
            adata.uns["source_file"] = str(source_path)

        return adata

    def _validate_basic_structure(self, adata: anndata.AnnData) -> ValidationResult:
        """
        Perform basic structural validation of AnnData object.

        Args:
            adata: AnnData object to validate

        Returns:
            ValidationResult: Basic validation results
        """
        result = ValidationResult()

        # Check basic structure
        if adata.n_obs == 0:
            result.add_error("No observations in dataset")
        if adata.n_vars == 0:
            result.add_error("No variables in dataset")

        # Check matrix dimensions
        if hasattr(adata.X, "shape"):
            if adata.X.shape != (adata.n_obs, adata.n_vars):
                result.add_error(
                    f"Matrix shape {adata.X.shape} doesn't match obs/var dimensions ({adata.n_obs}, {adata.n_vars})"
                )

        # Check for completely empty observations or variables
        if hasattr(adata.X, "sum"):
            obs_sums = np.array(adata.X.sum(axis=1)).flatten()
            var_sums = np.array(adata.X.sum(axis=0)).flatten()

            empty_obs = (obs_sums == 0).sum()
            empty_vars = (var_sums == 0).sum()

            if empty_obs > 0:
                result.add_warning(f"{empty_obs} observations have zero total counts")
            if empty_vars > 0:
                result.add_warning(f"{empty_vars} variables have zero total counts")

        return result

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.

        Returns:
            List[str]: List of supported file extensions
        """
        return ["csv", "tsv", "txt", "xlsx", "xls", "h5ad"]

    def preprocess_data(self, adata: anndata.AnnData, **kwargs) -> anndata.AnnData:
        """
        Apply basic preprocessing steps.

        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters

        Returns:
            anndata.AnnData: Preprocessed data object
        """
        # Ensure numeric matrix
        adata = self._ensure_numeric_matrix(adata)

        # Add basic metadata
        adata = self._add_basic_metadata(adata)

        return adata

    def get_quality_metrics(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """
        Calculate basic quality metrics.

        Args:
            adata: AnnData object to analyze

        Returns:
            Dict[str, Any]: Quality metrics dictionary
        """
        metrics = super().get_quality_metrics(adata)

        # Add modality-agnostic metrics
        if hasattr(adata.X, "sum"):
            obs_sums = np.array(adata.X.sum(axis=1)).flatten()
            var_sums = np.array(adata.X.sum(axis=0)).flatten()

            metrics.update(
                {
                    "total_counts": float(adata.X.sum()),
                    "mean_counts_per_obs": float(obs_sums.mean()),
                    "mean_counts_per_var": float(var_sums.mean()),
                    "zero_obs": int((obs_sums == 0).sum()),
                    "zero_vars": int((var_sums == 0).sum()),
                    "density": (
                        float(1.0 - (adata.X == 0).sum() / adata.X.size)
                        if hasattr(adata.X, "size")
                        else 0.0
                    ),
                }
            )

        return metrics

    # FIXME no harcoded solutions
    def detect_data_type(self, adata: anndata.AnnData) -> str:
        """
        Attempt to detect the type of biological data.

        Args:
            adata: AnnData object to analyze

        Returns:
            str: Detected data type hint
        """
        # This is a basic implementation - subclasses should override
        # with modality-specific detection logic

        if adata.n_vars > 10000:
            return "likely_genomics"  # High feature count suggests genomics
        elif adata.n_vars < 1000:
            return "likely_proteomics"  # Lower feature count suggests proteins
        else:
            return "unknown"

    def _log_operation(self, operation: str, **kwargs) -> None:
        """
        Log an adapter operation for debugging.

        Args:
            operation: Operation name
            **kwargs: Additional operation details
        """
        details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.logger.debug(f"{self.name} {operation}: {details}")
