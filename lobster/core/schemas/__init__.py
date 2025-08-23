"""
Schema definitions and validation for the modular DataManager architecture.

This module provides schema definitions for different biological data modalities
and flexible validation that supports both strict and permissive modes.
"""

from .validation import SchemaValidator, FlexibleValidator
from .transcriptomics import TranscriptomicsSchema
from .proteomics import ProteomicsSchema

__all__ = [
    "SchemaValidator",
    "FlexibleValidator", 
    "TranscriptomicsSchema",
    "ProteomicsSchema"
]
