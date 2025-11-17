"""
Flexible schema validation system with warning support.

This module provides validators that can operate in both strict mode
(errors cause failures) and permissive mode (warnings allow continued analysis).
"""

import logging
from typing import Any, Dict, Optional, Set

import anndata
import numpy as np

from lobster.core.interfaces.validator import IValidator, ValidationResult

logger = logging.getLogger(__name__)


class SchemaValidator(IValidator):
    """
    Base schema validator with flexible error handling.

    This validator can operate in strict mode (treat warnings as errors)
    or permissive mode (allow analysis to continue with warnings).
    """

    def __init__(self, schema: Dict[str, Any], name: str = "SchemaValidator"):
        """
        Initialize the schema validator.

        Args:
            schema: Schema definition dictionary
            name: Name for this validator instance
        """
        self.schema = schema
        self.name = name
        self.logger = logger

    def validate(
        self,
        adata: anndata.AnnData,
        strict: bool = False,
        check_types: bool = True,
        check_ranges: bool = True,
        check_completeness: bool = True,
    ) -> ValidationResult:
        """
        Validate AnnData object against schema.

        Args:
            adata: AnnData object to validate
            strict: If True, treat warnings as errors
            check_types: Whether to validate data types
            check_ranges: Whether to validate value ranges
            check_completeness: Whether to check for required fields

        Returns:
            ValidationResult: Validation results with errors/warnings
        """
        result = ValidationResult()

        # Validate against schema
        schema_result = self.validate_schema_compliance(adata, self.schema)
        result = result.merge(schema_result)

        # Additional validation checks
        if check_types:
            type_result = self._validate_data_types(adata)
            result = result.merge(type_result)

        if check_ranges:
            range_result = self._validate_value_ranges(adata)
            result = result.merge(range_result)

        if check_completeness:
            completeness_result = self._validate_completeness(adata)
            result = result.merge(completeness_result)

        # Convert warnings to errors if in strict mode
        if strict and result.has_warnings:
            for warning in result.warnings:
                result.add_error(f"[STRICT MODE] {warning}")
            result.warnings.clear()

        return result

    def validate_schema_compliance(
        self, adata: anndata.AnnData, schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate against a specific schema definition.

        Args:
            adata: AnnData object to validate
            schema: Schema definition to validate against

        Returns:
            ValidationResult: Schema validation results
        """
        result = ValidationResult()

        # Validate observation metadata
        if "obs" in schema:
            obs_result = self._validate_obs_schema(adata, schema["obs"])
            result = result.merge(obs_result)

        # Validate variable metadata
        if "var" in schema:
            var_result = self._validate_var_schema(adata, schema["var"])
            result = result.merge(var_result)

        # Validate layers
        if "layers" in schema:
            layers_result = self._validate_layers_schema(adata, schema["layers"])
            result = result.merge(layers_result)

        # Validate obsm (multi-dimensional observations)
        if "obsm" in schema:
            obsm_result = self._validate_obsm_schema(adata, schema["obsm"])
            result = result.merge(obsm_result)

        # Validate uns (unstructured metadata)
        if "uns" in schema:
            uns_result = self._validate_uns_schema(adata, schema["uns"])
            result = result.merge(uns_result)

        return result

    def _validate_obs_schema(
        self, adata: anndata.AnnData, obs_schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate observation metadata against schema."""
        result = ValidationResult()

        required_cols = obs_schema.get("required", [])
        optional_cols = obs_schema.get("optional", [])
        types = obs_schema.get("types", {})

        # Check required columns
        for col in required_cols:
            if col not in adata.obs.columns:
                result.add_error(f"Required obs column '{col}' is missing")
            elif adata.obs[col].isna().all():
                result.add_warning(
                    f"Required obs column '{col}' contains only NaN values"
                )

        # Check data types
        for col, expected_type in types.items():
            if col in adata.obs.columns:
                actual_type = adata.obs[col].dtype
                if not self._is_compatible_type(actual_type, expected_type):
                    result.add_warning(
                        f"obs column '{col}' has type {actual_type}, expected {expected_type}"
                    )

        # Check for unexpected columns (informational)
        expected_cols = set(required_cols + optional_cols)
        actual_cols = set(adata.obs.columns)
        unexpected = actual_cols - expected_cols
        if unexpected:
            result.add_info(f"Unexpected obs columns: {list(unexpected)}")

        return result

    def _validate_var_schema(
        self, adata: anndata.AnnData, var_schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate variable metadata against schema."""
        result = ValidationResult()

        required_cols = var_schema.get("required", [])
        var_schema.get("optional", [])
        types = var_schema.get("types", {})

        # Check required columns
        for col in required_cols:
            if col not in adata.var.columns:
                result.add_error(f"Required var column '{col}' is missing")
            elif adata.var[col].isna().all():
                result.add_warning(
                    f"Required var column '{col}' contains only NaN values"
                )

        # Check data types
        for col, expected_type in types.items():
            if col in adata.var.columns:
                actual_type = adata.var[col].dtype
                if not self._is_compatible_type(actual_type, expected_type):
                    result.add_warning(
                        f"var column '{col}' has type {actual_type}, expected {expected_type}"
                    )

        return result

    def _validate_layers_schema(
        self, adata: anndata.AnnData, layers_schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate data layers against schema."""
        result = ValidationResult()

        required_layers = layers_schema.get("required", [])
        layers_schema.get("optional", [])

        # Check required layers
        for layer_name in required_layers:
            if layer_name not in adata.layers:
                result.add_error(f"Required layer '{layer_name}' is missing")

        # Check layer dimensions
        for layer_name, layer_data in adata.layers.items():
            if hasattr(layer_data, "shape") and layer_data.shape != adata.X.shape:
                result.add_error(
                    f"Layer '{layer_name}' shape {layer_data.shape} doesn't match X shape {adata.X.shape}"
                )

        return result

    def _validate_obsm_schema(
        self, adata: anndata.AnnData, obsm_schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate multi-dimensional observations against schema."""
        result = ValidationResult()

        required_obsm = obsm_schema.get("required", [])

        # Check required obsm
        for obsm_name in required_obsm:
            if obsm_name not in adata.obsm:
                result.add_warning(f"Expected obsm '{obsm_name}' is missing")

        # Check obsm dimensions
        for obsm_name, obsm_data in adata.obsm.items():
            if hasattr(obsm_data, "shape") and obsm_data.shape[0] != adata.n_obs:
                result.add_error(
                    f"obsm '{obsm_name}' has {obsm_data.shape[0]} rows, expected {adata.n_obs}"
                )

        return result

    def _validate_uns_schema(
        self, adata: anndata.AnnData, uns_schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate unstructured metadata against schema."""
        result = ValidationResult()

        required_uns = uns_schema.get("required", [])

        # Check required uns keys
        for uns_key in required_uns:
            if uns_key not in adata.uns:
                result.add_warning(f"Expected uns key '{uns_key}' is missing")

        return result

    def _validate_data_types(self, adata: anndata.AnnData) -> ValidationResult:
        """Validate data types in the expression matrix."""
        result = ValidationResult()

        # Check if X matrix is numeric
        if hasattr(adata.X, "dtype"):
            if not np.issubdtype(adata.X.dtype, np.number):
                result.add_error(
                    f"Expression matrix has non-numeric dtype: {adata.X.dtype}"
                )

        return result

    def _validate_value_ranges(self, adata: anndata.AnnData) -> ValidationResult:
        """Validate value ranges in the data."""
        result = ValidationResult()

        # Check for extreme values, but only if data is not empty
        if hasattr(adata.X, "min") and hasattr(adata.X, "max") and adata.X.size > 0:
            min_val = adata.X.min()
            max_val = adata.X.max()

            if min_val < 0:
                result.add_warning(
                    f"Negative values found in expression matrix (min: {min_val})"
                )

            if max_val > 1e6:
                result.add_warning(
                    f"Very large values found in expression matrix (max: {max_val})"
                )

        return result

    def _validate_completeness(self, adata: anndata.AnnData) -> ValidationResult:
        """Validate data completeness."""
        result = ValidationResult()

        # Check for empty observations or variables
        if adata.n_obs == 0:
            result.add_error("No observations in dataset")
        if adata.n_vars == 0:
            result.add_error("No variables in dataset")

        # Check sparsity
        if hasattr(adata.X, "nnz"):
            total_elements = adata.X.shape[0] * adata.X.shape[1]
            if total_elements > 0:
                sparsity = 1.0 - (adata.X.nnz / total_elements)
                if sparsity > 0.95:
                    result.add_warning(f"Very sparse data: {sparsity:.1%} zeros")

        return result

    def _is_compatible_type(self, actual_type: np.dtype, expected_type: str) -> bool:
        """Check if actual dtype is compatible with expected type."""
        type_mapping = {
            "string": ["object", "string"],
            "categorical": ["category"],
            "numeric": [
                "int8",
                "int16",
                "int32",
                "int64",
                "float16",
                "float32",
                "float64",
            ],
            "boolean": ["bool"],
            "datetime": ["datetime64"],
        }

        expected_dtypes = type_mapping.get(expected_type, [expected_type])
        return (
            str(actual_type) in expected_dtypes or actual_type.name in expected_dtypes
        )


class FlexibleValidator(SchemaValidator):
    """
    Enhanced validator with flexible validation rules and warnings.

    This validator provides more nuanced validation with configurable
    strictness levels and the ability to selectively ignore certain
    validation checks.
    """

    def __init__(
        self,
        schema: Dict[str, Any],
        name: str = "FlexibleValidator",
        ignore_warnings: Optional[Set[str]] = None,
        custom_rules: Optional[Dict[str, callable]] = None,
    ):
        """
        Initialize the flexible validator.

        Args:
            schema: Schema definition dictionary
            name: Name for this validator instance
            ignore_warnings: Set of warning types to ignore
            custom_rules: Custom validation rules
        """
        super().__init__(schema, name)
        self.ignore_warnings = ignore_warnings or set()
        self.custom_rules = custom_rules or {}

    def validate(
        self,
        adata: anndata.AnnData,
        strict: bool = False,
        check_types: bool = True,
        check_ranges: bool = True,
        check_completeness: bool = True,
    ) -> ValidationResult:
        """
        Validate with flexible rules and custom filtering.

        Args:
            adata: AnnData object to validate
            strict: If True, treat warnings as errors
            check_types: Whether to validate data types
            check_ranges: Whether to validate value ranges
            check_completeness: Whether to check for required fields

        Returns:
            ValidationResult: Validation results with filtered warnings
        """
        # Get base validation results
        result = super().validate(
            adata,
            strict=False,
            check_types=check_types,
            check_ranges=check_ranges,
            check_completeness=check_completeness,
        )

        # Apply custom rules
        for rule_name, rule_func in self.custom_rules.items():
            try:
                custom_result = rule_func(adata)
                if isinstance(custom_result, ValidationResult):
                    result = result.merge(custom_result)
            except Exception as e:
                result.add_warning(f"Custom rule '{rule_name}' failed: {e}")

        # Filter out ignored warnings
        if self.ignore_warnings:
            result.warnings = [
                w
                for w in result.warnings
                if not any(ignore_type in w for ignore_type in self.ignore_warnings)
            ]

        # Apply strict mode after filtering
        if strict and result.has_warnings:
            for warning in result.warnings:
                result.add_error(f"[STRICT MODE] {warning}")
            result.warnings.clear()

        return result

    def add_custom_rule(self, name: str, rule_func: callable) -> None:
        """
        Add a custom validation rule.

        Args:
            name: Name of the custom rule
            rule_func: Function that takes AnnData and returns ValidationResult
        """
        self.custom_rules[name] = rule_func

    def ignore_warning_type(self, warning_type: str) -> None:
        """
        Add a warning type to ignore.

        Args:
            warning_type: String pattern to match in warning messages
        """
        self.ignore_warnings.add(warning_type)
