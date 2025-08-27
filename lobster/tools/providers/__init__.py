"""
Publication providers package for modular literature and dataset discovery.

This package implements the provider pattern for accessing different publication
sources including PubMed, bioRxiv, medRxiv, and others.
"""

from .base_provider import BasePublicationProvider
from .pubmed_provider import PubMedProvider

__all__ = [
    "BasePublicationProvider",
    "PubMedProvider",
]
