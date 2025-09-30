"""
Lobster core module with exception hierarchy and base classes.

This module provides the core exception hierarchy and shared functionality
for the Lobster AI system following a structured approach to error handling.
"""


class LobsterCoreError(Exception):
    """
    Base exception class for all Lobster core errors.
    
    This serves as the root of the exception hierarchy, allowing for
    consistent error handling across all Lobster modules.
    """
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize the core error.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        """Return string representation of the error."""
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


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
    'LobsterCoreError',
    'DataManagerError', 
    'AdapterError',
    'ValidationError',
    'ProvenanceError',
    'PseudobulkError',
    'AggregationError',
    'FormulaError',
    'InsufficientCellsError',
    'DesignMatrixError'
]
