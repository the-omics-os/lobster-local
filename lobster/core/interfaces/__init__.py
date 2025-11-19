"""
Core interfaces for the modular DataManager architecture.

This module defines the abstract base classes and interfaces that enable
a clean separation of concerns between data storage backends, modality-specific
adapters, validation components, and download services.
"""

from .adapter import IModalityAdapter
from .backend import IDataBackend
from .download_service import IDownloadService
from .validator import IValidator, ValidationResult

__all__ = [
    "IDataBackend",
    "IModalityAdapter",
    "IDownloadService",
    "IValidator",
    "ValidationResult",
]
