"""
Data backend implementations for the modular DataManager architecture.

This module provides concrete implementations of data storage backends
for different storage systems including local files and future cloud storage.
"""

from .base import BaseBackend
from .h5ad_backend import H5ADBackend

try:
    from .mudata_backend import MuDataBackend

    __all__ = ["BaseBackend", "H5ADBackend", "MuDataBackend"]
except ImportError:
    # MuData not available
    __all__ = ["BaseBackend", "H5ADBackend"]
