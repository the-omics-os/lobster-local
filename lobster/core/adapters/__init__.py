"""
Modality adapter implementations for the modular DataManager architecture.

This module provides concrete implementations of modality-specific data adapters
for different biological data types including transcriptomics and proteomics.
"""

from .base import BaseAdapter

# Import specific adapters - handle import errors gracefully
try:
    from .transcriptomics_adapter import TranscriptomicsAdapter
    _transcriptomics_available = True
except ImportError:
    _transcriptomics_available = False

try:
    from .proteomics_adapter import ProteomicsAdapter
    _proteomics_available = True
except ImportError:
    _proteomics_available = False

# Build __all__ dynamically based on what's available
__all__ = ["BaseAdapter"]

if _transcriptomics_available:
    __all__.append("TranscriptomicsAdapter")
if _proteomics_available:
    __all__.append("ProteomicsAdapter")
