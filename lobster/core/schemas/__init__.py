"""
Schema definitions and validation for the modular DataManager architecture.

This module provides schema definitions for different biological data modalities
and flexible validation that supports both strict and permissive modes.
"""

from .download_queue import DownloadQueueEntry, DownloadStatus, StrategyConfig

__all__ = [
    "DownloadQueueEntry",
    "DownloadStatus",
    "StrategyConfig",
]
