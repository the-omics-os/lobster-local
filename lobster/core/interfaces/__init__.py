"""
Core interfaces for the modular DataManager architecture.

This module defines the abstract base classes and interfaces that enable
a clean separation of concerns between data storage backends, modality-specific
adapters, and validation components.
"""

from .adapter import IModalityAdapter
from .backend import IDataBackend
from .validator import IValidator, ValidationResult

__all__ = ["IDataBackend", "IModalityAdapter", "IValidator", "ValidationResult"]
