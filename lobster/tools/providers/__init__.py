"""
Publication providers for literature search and dataset discovery.

This package contains different provider implementations for accessing
various publication and dataset sources.
"""

from lobster.tools.providers.base_provider import (
    BasePublicationProvider,
    DatasetMetadata,
    DatasetType,
    PublicationMetadata,
    PublicationSource,
)
from lobster.tools.providers.geo_provider import GEOProvider, GEOProviderConfig
from lobster.tools.providers.pubmed_provider import PubMedProvider, PubMedProviderConfig

__all__ = [
    # Base classes
    "BasePublicationProvider",
    "PublicationSource",
    "DatasetType",
    "PublicationMetadata",
    "DatasetMetadata",
    # PubMed provider
    "PubMedProvider",
    "PubMedProviderConfig",
]
