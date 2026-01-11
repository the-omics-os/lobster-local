"""
Schema definitions and validation for the modular DataManager architecture.

This module provides schema definitions for different biological data modalities
and flexible validation that supports both strict and permissive modes.
"""

from .download_queue import DownloadQueueEntry, DownloadStatus, StrategyConfig
from .ontology import DiseaseConcept, DiseaseMatch, DiseaseOntologyConfig

__all__ = [
    "DownloadQueueEntry",
    "DownloadStatus",
    "StrategyConfig",
    "DiseaseConcept",
    "DiseaseMatch",
    "DiseaseOntologyConfig",
]
