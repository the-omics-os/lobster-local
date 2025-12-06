"""
GEO (Gene Expression Omnibus) data access package.

This package provides modular access to GEO data with clean separation of concerns:
- constants: Enums, dataclasses, and platform registry
- downloader: File download with retry logic
- parser: Multi-format file parsing
- strategy: Pipeline selection engine
- metadata/: Metadata fetching, extraction, and validation
- sample/: Sample downloading, validation, and storage
- utils/: URL classification and file pattern utilities

Usage:
    # Import GEOService from the legacy location (during transition)
    from lobster.services.data_access.geo_service import GEOService

    # Or use the sub-modules directly
    from lobster.services.data_access.geo.downloader import GEODownloadManager
    from lobster.services.data_access.geo.parser import GEOParser
    from lobster.services.data_access.geo.strategy import PipelineStrategyEngine

    service = GEOService(data_manager)
    metadata, validation = service.fetch_metadata_only("GSE12345")
    result = service.download_dataset("GSE12345")
"""

# Re-export constants and sub-modules (no GEOService to avoid circular import)
from lobster.services.data_access.geo.constants import (
    PLATFORM_REGISTRY,
    DownloadStrategy,
    GEODataSource,
    GEODataType,
    GEOFallbackError,
    GEOResult,
    GEOServiceError,
    PlatformCompatibility,
)

# Lazy import GEOService to avoid circular import
# Users should import from geo_service directly during transition


def __getattr__(name):
    """Lazy import GEOService to avoid circular dependency."""
    if name == "GEOService":
        from lobster.services.data_access.geo_service import GEOService

        return GEOService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GEOService",  # Available via lazy import
    "GEODataSource",
    "GEODataType",
    "GEOResult",
    "DownloadStrategy",
    "PlatformCompatibility",
    "GEOServiceError",
    "GEOFallbackError",
    "PLATFORM_REGISTRY",
]
