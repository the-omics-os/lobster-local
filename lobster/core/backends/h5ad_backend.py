"""
H5AD backend implementation with S3-ready path handling.

This module provides the H5ADBackend for storing AnnData objects
in the H5AD format with support for local storage and future
S3 integration without API changes.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from lobster.core.backends.base import BaseBackend
from lobster.core.utils.h5ad_utils import sanitize_key as util_sanitize_key
from lobster.core.utils.h5ad_utils import sanitize_value as util_sanitize_value

logger = logging.getLogger(__name__)


class H5ADBackend(BaseBackend):
    """
    Backend for H5AD file storage with S3-ready path handling.

    This backend handles AnnData objects stored in H5AD format,
    with path parsing that's ready for future S3 integration
    without requiring API changes.
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        compression: str = "gzip",
        compression_opts: Optional[int] = None,
    ):
        """
        Initialize the H5AD backend.

        Args:
            base_path: Optional base path for all operations
            compression: Compression method for H5AD files
            compression_opts: Compression level (1-9 for gzip)
        """
        super().__init__(base_path=base_path)
        self.compression = compression
        self.compression_opts = compression_opts or 6  # Default compression level

        # S3-ready configuration (for future use)
        self.s3_config = {
            "bucket": None,
            "region": None,
            "access_key": None,
            "secret_key": None,
        }

    def load(self, path: Union[str, Path], **kwargs) -> anndata.AnnData:
        """
        Load AnnData from H5AD file.

        Args:
            path: Path to H5AD file (local path or future S3 URI)
            **kwargs: Additional loading parameters:
                - backed: Load in backed mode (default: False)
                - chunk_size: Chunk size for backed mode

        Returns:
            anndata.AnnData: Loaded AnnData object

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        resolved_path = self._resolve_path_with_s3_support(path)

        # Check if this is an S3 path (for future implementation)
        if self._is_s3_path(path):
            return self._load_from_s3(path, **kwargs)

        # Local file loading
        if not resolved_path.exists():
            raise FileNotFoundError(f"H5AD file not found: {resolved_path}")

        try:
            # Extract loading parameters
            backed = kwargs.get("backed", False)

            if backed:
                # Load in backed mode for large files
                adata = sc.read_h5ad(resolved_path, backed="r")
            else:
                # Load fully into memory
                adata = sc.read_h5ad(resolved_path)

            self._log_operation(
                "load",
                resolved_path,
                backed=backed,
                size_mb=resolved_path.stat().st_size / 1024**2,
            )

            return adata

        except Exception as e:
            raise ValueError(f"Failed to load H5AD file {resolved_path}: {e}")

    @staticmethod
    def sanitize_anndata(adata, slash_replacement="__"):
        """
        Sanitize AnnData object so it can be safely written to H5AD.
        - Converts OrderedDict → dict
        - Converts tuple → list
        - Converts numpy scalars → Python scalars
        - Replaces '/' in keys with '__' (HDF5 safe)
        - Converts boolean columns to strings (HDF5 requirement)
        - Handles mixed-type columns (converts to strings)
        - Converts None/NaN to "NA" for HDF5 compatibility
        - Recursively applies to .uns, .obsm, .varm, .layers

        Note: Uses centralized sanitization utilities from lobster.core.utils.h5ad_utils
        """

        # Sanitize uns (unstructured metadata) using centralized utility
        adata.uns = {
            util_sanitize_key(k, slash_replacement): util_sanitize_value(v, slash_replacement)
            for k, v in adata.uns.items()
        }

        # Sanitize obsm, varm, layers (keys must be safe)
        adata.obsm = {util_sanitize_key(k, slash_replacement): v for k, v in adata.obsm.items()}
        adata.varm = {util_sanitize_key(k, slash_replacement): v for k, v in adata.varm.items()}
        adata.layers = {util_sanitize_key(k, slash_replacement): v for k, v in adata.layers.items()}

        # Sanitize DataFrame column names in obs and var
        adata.obs.columns = [util_sanitize_key(col, slash_replacement) for col in adata.obs.columns]
        adata.var.columns = [util_sanitize_key(col, slash_replacement) for col in adata.var.columns]

        # Sanitize DataFrame VALUES in obs and var (COMPREHENSIVE)
        # This prevents serialization errors when writing to H5AD
        for df in [adata.obs, adata.var]:
            columns_to_drop = []
            for col in df.columns:
                # Step 1: Check if column is entirely None/NaN
                if df[col].isna().all():
                    columns_to_drop.append(col)
                    logger.debug(f"Dropping column '{col}' - all values are None/NaN")
                    continue

                # Step 2: Handle boolean columns (convert to string)
                if df[col].dtype == bool or df[col].dtype == "boolean":
                    df[col] = df[col].astype(str)
                    logger.debug(f"Sanitized column '{col}' - converted bool to string")
                    continue

                # Step 3: Handle None/NaN in numeric columns
                if df[col].isna().any():
                    if np.issubdtype(df[col].dtype, np.number):
                        # Numeric column with NaN - fill with 0 or keep as NaN
                        # Keep NaN for numeric columns (HDF5 can handle numeric NaN)
                        pass
                    else:
                        # Object dtype with None - convert None to "NA"
                        df[col] = df[col].fillna("NA")
                        logger.debug(
                            f"Sanitized column '{col}' - converted None to 'NA'"
                        )

                # Step 4: Handle mixed-type columns (object dtype)
                if df[col].dtype == "object":
                    # Check if column has multiple types
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        unique_types = non_null_values.apply(
                            lambda x: type(x).__name__
                        ).unique()

                        if len(unique_types) > 1:
                            # Mixed types detected - convert all to string
                            df[col] = df[col].apply(
                                lambda x: str(x) if x is not None else "NA"
                            )
                            logger.debug(
                                f"Sanitized column '{col}' - mixed types {unique_types} converted to string"
                            )
                        else:
                            # Single type but object dtype - try numeric conversion
                            try:
                                # Attempt numeric conversion
                                df[col] = pd.to_numeric(df[col])
                                logger.debug(
                                    f"Sanitized column '{col}' - converted to numeric"
                                )
                            except (ValueError, TypeError):
                                # Not numeric - ensure all are strings
                                df[col] = df[col].astype(str)
                                logger.debug(
                                    f"Sanitized column '{col}' - ensured string type"
                                )

                # Step 5: Handle categorical columns with non-string categories
                if hasattr(df[col], "cat"):
                    # Check if categories contain non-strings
                    categories = df[col].cat.categories
                    if not all(isinstance(cat, str) for cat in categories):
                        df[col] = df[col].astype(str).astype("category")
                        logger.debug(
                            f"Sanitized column '{col}' - converted categorical to string categories"
                        )

            # Drop columns that are entirely None/NaN
            if columns_to_drop:
                df.drop(columns=columns_to_drop, inplace=True)
                logger.info(
                    f"Dropped {len(columns_to_drop)} empty columns: {columns_to_drop}"
                )

        return adata

    def save(self, adata: anndata.AnnData, path: Union[str, Path], **kwargs) -> None:
        """
        Save AnnData to H5AD file.

        Args:
            adata: AnnData object to save
            path: Destination path (local path or future S3 URI)
            **kwargs: Additional saving parameters:
                - compression: Override default compression
                - compression_opts: Override compression level
                - as_dense: Save sparse matrices as dense

        Raises:
            ValueError: If the data cannot be serialized
            PermissionError: If write access is denied
        """
        resolved_path = self._resolve_path_with_s3_support(path)

        # Check if this is an S3 path (for future implementation)
        if self._is_s3_path(path):
            return self._save_to_s3(adata, path, **kwargs)

        # Ensure parent directory exists
        self._ensure_directory(resolved_path)

        # Create backup if file exists
        # if resolved_path.exists():
        #     backup_path = self.create_backup(resolved_path)
        #     if backup_path:
        #         self.logger.info(f"Created backup: {backup_path}")

        try:
            # Extract saving parameters
            compression = kwargs.get("compression", self.compression)
            compression_opts = kwargs.get("compression_opts", self.compression_opts)
            as_dense = kwargs.get("as_dense", False)

            # saniztize Anndata before storing
            adata_sanitized = self.sanitize_anndata(adata)

            # make variables unique
            adata_sanitized.var_names_make_unique(join="__")

            # Prepare AnnData for saving
            adata_to_save = adata_sanitized.copy()

            if as_dense and hasattr(adata_to_save.X, "toarray"):
                # Convert sparse to dense if requested
                adata_to_save.X = adata_to_save.X.toarray()

            # Save to H5AD format
            adata_to_save.write_h5ad(
                resolved_path,
                compression=compression,
                compression_opts=compression_opts,
            )

            self._log_operation(
                "save",
                resolved_path,
                compression=compression,
                compression_opts=compression_opts,
                shape=adata.shape,
                size_mb=resolved_path.stat().st_size / 1024**2,
            )

        except Exception as e:
            # Remove failed file if it was created
            if resolved_path.exists():
                try:
                    resolved_path.unlink()
                except Exception:
                    pass
            raise ValueError(f"Failed to save H5AD file {resolved_path}: {e}")

    def supports_format(self, format_name: str) -> bool:
        """
        Check if the backend supports a specific file format.

        Args:
            format_name: Format to check

        Returns:
            bool: True if format is supported
        """
        return format_name.lower() in ["h5ad", "h5"]

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the storage backend.

        Returns:
            Dict[str, Any]: Storage backend information
        """
        info = super().get_storage_info()
        info.update(
            {
                "supported_formats": ["h5ad"],
                "compression": self.compression,
                "compression_opts": self.compression_opts,
                "s3_ready": True,
                "backed_mode_support": True,
            }
        )
        return info

    def optimize_for_reading(
        self, path: Union[str, Path], chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize H5AD file for reading performance.

        Args:
            path: Path to H5AD file
            chunk_size: Optimal chunk size for reading

        Returns:
            Dict[str, Any]: Optimization results
        """
        resolved_path = self._resolve_path_with_s3_support(path)

        if not resolved_path.exists():
            raise FileNotFoundError(f"H5AD file not found: {resolved_path}")

        try:
            # Load file to analyze structure
            adata = self.load(resolved_path, backed=True)

            # Analyze data characteristics
            analysis = {
                "shape": adata.shape,
                "is_sparse": hasattr(adata.X, "nnz"),
                "sparsity": None,
                "recommended_chunk_size": None,
                "estimated_memory_mb": None,
            }

            if hasattr(adata.X, "nnz"):
                # Sparse matrix analysis
                analysis["sparsity"] = 1.0 - (
                    adata.X.nnz / (adata.n_obs * adata.n_vars)
                )

                # Recommend chunk size based on sparsity and size
                if analysis["sparsity"] > 0.9:  # Very sparse
                    analysis["recommended_chunk_size"] = min(10000, adata.n_obs // 10)
                else:
                    analysis["recommended_chunk_size"] = min(5000, adata.n_obs // 20)

            # Estimate memory usage
            dtype_size = 4 if adata.X.dtype in [np.float32, np.int32] else 8
            if analysis["is_sparse"]:
                # Sparse matrix memory estimate
                analysis["estimated_memory_mb"] = (
                    adata.X.nnz * dtype_size * 3
                ) / 1024**2  # data + indices + indptr
            else:
                # Dense matrix memory estimate
                analysis["estimated_memory_mb"] = (
                    adata.n_obs * adata.n_vars * dtype_size
                ) / 1024**2

            return analysis

        except Exception as e:
            self.logger.warning(f"Failed to optimize file {path}: {e}")
            return {"error": str(e)}

    def validate_file_integrity(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate H5AD file integrity and structure.

        Args:
            path: Path to H5AD file

        Returns:
            Dict[str, Any]: Validation results
        """
        resolved_path = self._resolve_path_with_s3_support(path)

        validation = {
            "valid": False,
            "readable": False,
            "has_X": False,
            "has_obs": False,
            "has_var": False,
            "shape": None,
            "errors": [],
            "warnings": [],
        }

        try:
            # Check if file exists and is readable
            if not resolved_path.exists():
                validation["errors"].append("File does not exist")
                return validation

            # Try to load the file
            adata = self.load(resolved_path, backed=True)
            validation["readable"] = True

            # Check basic structure
            validation["shape"] = adata.shape
            validation["has_X"] = adata.X is not None
            validation["has_obs"] = len(adata.obs) > 0
            validation["has_var"] = len(adata.var) > 0

            # Check for common issues
            if adata.n_obs == 0:
                validation["warnings"].append("No observations in dataset")
            if adata.n_vars == 0:
                validation["warnings"].append("No variables in dataset")

            # Validate X matrix
            if adata.X is not None:
                if hasattr(adata.X, "dtype"):
                    if not np.issubdtype(adata.X.dtype, np.number):
                        validation["errors"].append("Non-numeric data in X matrix")

            validation["valid"] = len(validation["errors"]) == 0

        except Exception as e:
            validation["errors"].append(f"Failed to read file: {e}")

        return validation

    def _resolve_path_with_s3_support(self, path: Union[str, Path]) -> Path:
        """
        Resolve path with S3 support (future-ready).

        Args:
            path: Path to resolve

        Returns:
            Path: Resolved path (for local files)
        """
        # If it's an S3 path, we'll handle it differently in the future
        if self._is_s3_path(path):
            # For now, just return the string representation
            # In future S3 implementation, this would handle S3 URIs
            return Path(str(path))

        # Handle local paths normally
        return self._resolve_path(path)

    def _is_s3_path(self, path: Union[str, Path]) -> bool:
        """
        Check if path is an S3 URI.

        Args:
            path: Path to check

        Returns:
            bool: True if path is S3 URI
        """
        path_str = str(path)
        return path_str.startswith("s3://")

    def _load_from_s3(self, s3_path: str, **kwargs) -> anndata.AnnData:
        """
        Load AnnData from S3 (future implementation).

        Args:
            s3_path: S3 URI
            **kwargs: Loading parameters

        Returns:
            anndata.AnnData: Loaded data

        Raises:
            NotImplementedError: S3 support not yet implemented
        """
        # This is a placeholder for future S3 implementation
        raise NotImplementedError(
            "S3 support is not yet implemented. "
            "The current backend supports local files only, "
            "but the API is designed to support S3 in the future."
        )

    def _save_to_s3(self, adata: anndata.AnnData, s3_path: str, **kwargs) -> None:
        """
        Save AnnData to S3 (future implementation).

        Args:
            adata: AnnData object to save
            s3_path: S3 URI
            **kwargs: Saving parameters

        Raises:
            NotImplementedError: S3 support not yet implemented
        """
        # This is a placeholder for future S3 implementation
        raise NotImplementedError(
            "S3 support is not yet implemented. "
            "The current backend supports local files only, "
            "but the API is designed to support S3 in the future."
        )

    def configure_s3(
        self,
        bucket: str,
        region: str = "us-east-1",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        """
        Configure S3 settings (future implementation).

        Args:
            bucket: S3 bucket name
            region: AWS region
            access_key: AWS access key
            secret_key: AWS secret key
        """
        self.s3_config.update(
            {
                "bucket": bucket,
                "region": region,
                "access_key": access_key,
                "secret_key": secret_key,
            }
        )

        self.logger.info(
            f"S3 configuration set for bucket: {bucket} in region: {region}"
        )
        self.logger.warning(
            "S3 functionality is not yet implemented but configuration is stored"
        )

    def _detect_format(self, path: Union[str, Path]) -> str:
        """
        Detect file format from extension.

        Args:
            path: Path to analyze

        Returns:
            str: Detected format
        """
        path = Path(path)
        extension = path.suffix.lower()

        if extension == ".h5ad":
            return "h5ad"
        elif extension == ".h5":
            return "h5"
        else:
            return "unknown"
