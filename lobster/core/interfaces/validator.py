"""
Validation interface definitions.

This module defines the interfaces and data structures for flexible
validation of biological data with support for warnings instead of
hard failures, enabling exploratory data analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anndata


@dataclass
class ValidationResult:
    """
    Result of a validation operation.
    
    This class encapsulates the results of validating biological data,
    supporting both errors (critical issues) and warnings (non-critical
    issues that don't prevent analysis).
    """
    
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_errors(self) -> bool:
        """Check if validation found any errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation found any warnings."""
        return len(self.warnings) > 0
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return not self.has_errors
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_info(self, message: str) -> None:
        """Add an informational message."""
        self.info.append(message)
    
    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """
        Merge another validation result into this one.
        
        Args:
            other: Another ValidationResult to merge
            
        Returns:
            ValidationResult: New merged result
        """
        return ValidationResult(
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            info=self.info + other.info,
            metadata={**self.metadata, **other.metadata}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "metadata": self.metadata,
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "is_valid": self.is_valid
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        parts = []
        
        if self.has_errors:
            parts.append(f"{len(self.errors)} error(s)")
        
        if self.has_warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        
        if self.info:
            parts.append(f"{len(self.info)} info message(s)")
        
        if not parts:
            return "Validation passed with no issues"
        
        return f"Validation completed with {', '.join(parts)}"
    
    def format_messages(self, include_info: bool = True) -> str:
        """Format all messages for display."""
        lines = []
        
        if self.errors:
            lines.append("ERRORS:")
            for error in self.errors:
                lines.append(f"  ❌ {error}")
        
        if self.warnings:
            if lines:
                lines.append("")
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
        
        if self.info and include_info:
            if lines:
                lines.append("")
            lines.append("INFO:")
            for info_msg in self.info:
                lines.append(f"  ℹ️  {info_msg}")
        
        return "\n".join(lines)


class IValidator(ABC):
    """
    Abstract interface for data validators.
    
    This interface defines the contract for validating biological data
    against schemas with flexible error handling that supports both
    strict validation (errors cause failures) and permissive validation
    (warnings allow continued analysis).
    """

    @abstractmethod
    def validate(
        self, 
        adata: anndata.AnnData,
        strict: bool = False,
        check_types: bool = True,
        check_ranges: bool = True,
        check_completeness: bool = True
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
        pass

    @abstractmethod
    def validate_schema_compliance(
        self, 
        adata: anndata.AnnData,
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate against a specific schema definition.

        Args:
            adata: AnnData object to validate
            schema: Schema definition to validate against

        Returns:
            ValidationResult: Schema validation results
        """
        pass

    def validate_obs_metadata(
        self, 
        adata: anndata.AnnData,
        required_columns: Optional[List[str]] = None,
        optional_columns: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate observation (cell/sample) metadata.

        Args:
            adata: AnnData object to validate
            required_columns: List of required obs columns
            optional_columns: List of optional obs columns

        Returns:
            ValidationResult: Obs metadata validation results
        """
        result = ValidationResult()
        
        if required_columns:
            for col in required_columns:
                if col not in adata.obs.columns:
                    result.add_error(f"Required obs column '{col}' is missing")
                elif adata.obs[col].isna().all():
                    result.add_warning(f"Required obs column '{col}' contains only NaN values")
        
        # Check for unexpected columns
        expected_columns = set((required_columns or []) + (optional_columns or []))
        actual_columns = set(adata.obs.columns)
        unexpected = actual_columns - expected_columns
        
        if unexpected:
            result.add_info(f"Unexpected obs columns found: {list(unexpected)}")
        
        return result

    def validate_var_metadata(
        self, 
        adata: anndata.AnnData,
        required_columns: Optional[List[str]] = None,
        optional_columns: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate variable (gene/protein) metadata.

        Args:
            adata: AnnData object to validate
            required_columns: List of required var columns
            optional_columns: List of optional var columns

        Returns:
            ValidationResult: Var metadata validation results
        """
        result = ValidationResult()
        
        if required_columns:
            for col in required_columns:
                if col not in adata.var.columns:
                    result.add_error(f"Required var column '{col}' is missing")
                elif adata.var[col].isna().all():
                    result.add_warning(f"Required var column '{col}' contains only NaN values")
        
        # Check for unexpected columns
        expected_columns = set((required_columns or []) + (optional_columns or []))
        actual_columns = set(adata.var.columns)
        unexpected = actual_columns - expected_columns
        
        if unexpected:
            result.add_info(f"Unexpected var columns found: {list(unexpected)}")
        
        return result

    def validate_layers(
        self, 
        adata: anndata.AnnData,
        expected_layers: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate data layers.

        Args:
            adata: AnnData object to validate
            expected_layers: List of expected layer names

        Returns:
            ValidationResult: Layers validation results
        """
        result = ValidationResult()
        
        if expected_layers:
            for layer_name in expected_layers:
                if layer_name not in adata.layers:
                    result.add_warning(f"Expected layer '{layer_name}' is missing")
        
        # Check layer shapes
        for layer_name, layer_data in adata.layers.items():
            if hasattr(layer_data, 'shape') and layer_data.shape != adata.X.shape:
                result.add_error(f"Layer '{layer_name}' shape {layer_data.shape} doesn't match X shape {adata.X.shape}")
        
        return result

    def validate_data_quality(self, adata: anndata.AnnData) -> ValidationResult:
        """
        Perform basic data quality checks.

        Args:
            adata: AnnData object to validate

        Returns:
            ValidationResult: Data quality validation results
        """
        result = ValidationResult()
        
        # Check for empty data
        if adata.n_obs == 0:
            result.add_error("No observations (cells/samples) in dataset")
        if adata.n_vars == 0:
            result.add_error("No variables (genes/proteins) in dataset")
        
        # Check for NaN values in X matrix
        if hasattr(adata.X, 'isnan'):
            nan_count = adata.X.isnan().sum()
            if nan_count > 0:
                nan_percentage = (nan_count / adata.X.size) * 100
                if nan_percentage > 50:
                    result.add_warning(f"High proportion of NaN values in X matrix: {nan_percentage:.1f}%")
                else:
                    result.add_info(f"NaN values in X matrix: {nan_percentage:.1f}%")
        
        # Check for negative values (if appropriate)
        if hasattr(adata.X, 'min'):
            min_val = adata.X.min()
            if min_val < 0:
                result.add_warning(f"Negative values found in X matrix (minimum: {min_val})")
        
        return result
