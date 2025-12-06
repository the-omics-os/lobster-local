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
from lobster.tools.providers.pubmed_provider import PubMedProvider, PubMedProviderConfig

# PRIDE provider is PREMIUM-only (proteomics)
# Import conditionally to avoid breaking FREE tier
try:
    from lobster.tools.providers.pride_provider import PRIDEProvider, PRIDEProviderConfig
    _PRIDE_AVAILABLE = True
except ImportError:
    _PRIDE_AVAILABLE = False
    PRIDEProvider = None  # type: ignore
    PRIDEProviderConfig = None  # type: ignore

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
    # MassIVE provider
    "MassIVEProvider",
    "MassIVEProviderConfig",
]

# Add PRIDE to exports if available (PREMIUM tier)
if _PRIDE_AVAILABLE:
    __all__.extend(["PRIDEProvider", "PRIDEProviderConfig"])
