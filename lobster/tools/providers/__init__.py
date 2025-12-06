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
from lobster.tools.providers.biorxiv_medrxiv_config import BioRxivMedRxivConfig
from lobster.tools.providers.biorxiv_medrxiv_provider import BioRxivMedRxivProvider
from lobster.tools.providers.geo_provider import GEOProvider, GEOProviderConfig
from lobster.tools.providers.massive_provider import (
    MassIVEProvider,
    MassIVEProviderConfig,
)
from lobster.tools.providers.pride_provider import PRIDEProvider, PRIDEProviderConfig
from lobster.tools.providers.pubmed_provider import PubMedProvider, PubMedProviderConfig

__all__ = [
    # Base classes
    "BasePublicationProvider",
    "PublicationSource",
    "DatasetType",
    "PublicationMetadata",
    "DatasetMetadata",
    # BioRxiv/MedRxiv provider
    "BioRxivMedRxivProvider",
    "BioRxivMedRxivConfig",
    # GEO provider
    "GEOProvider",
    "GEOProviderConfig",
    # PubMed provider
    "PubMedProvider",
    "PubMedProviderConfig",
    # PRIDE provider
    "PRIDEProvider",
    "PRIDEProviderConfig",
    # MassIVE provider
    "MassIVEProvider",
    "MassIVEProviderConfig",
]
