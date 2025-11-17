"""
Lobster core module with exception hierarchy and base classes.

This module provides the core exception hierarchy and shared functionality
for the Lobster AI system following a structured approach to error handling.
"""

# Import new exceptions from exceptions module
from lobster.core.exceptions import (
    DataOrientationError,
    DataTypeAmbiguityError,
    FeatureNotImplementedError,
    LobsterCoreError,
    UnsupportedFormatError,
    UnsupportedPlatformError,
)


class DataManagerError(LobsterCoreError):
    """Exception raised for data manager operations."""

    pass


class AdapterError(LobsterCoreError):
    """Exception raised for adapter-related operations."""

    pass


class ValidationError(LobsterCoreError):
    """Exception raised for data validation failures."""

    pass


class ProvenanceError(LobsterCoreError):
    """Exception raised for provenance tracking failures."""

    pass


# Pseudobulk-specific exceptions
class PseudobulkError(LobsterCoreError):
    """Base exception for pseudobulk operations."""

    pass


class AggregationError(PseudobulkError):
    """Raised when aggregation fails."""

    pass


class FormulaError(PseudobulkError):
    """Raised when formula parsing fails."""

    pass


class InsufficientCellsError(PseudobulkError):
    """Raised when too few cells for aggregation."""

    pass


class DesignMatrixError(PseudobulkError):
    """Raised when design matrix construction fails."""

    pass


# Export all exceptions for easy importing
__all__ = [
    # Base exception and new exceptions from exceptions.py
    "LobsterCoreError",
    "UnsupportedFormatError",
    "UnsupportedPlatformError",
    "FeatureNotImplementedError",
    "DataTypeAmbiguityError",
    "DataOrientationError",
    # Existing exceptions
    "DataManagerError",
    "AdapterError",
    "ValidationError",
    "ProvenanceError",
    "PseudobulkError",
    "AggregationError",
    "FormulaError",
    "InsufficientCellsError",
    "DesignMatrixError",
]
