"""
GEO Service Facade - Thin orchestration layer.

This module provides the main GEOService class as a facade that coordinates
between the various sub-modules (downloader, parser, strategy, etc.).

During the refactoring transition, this facade imports from the legacy
geo_service.py while the functionality is being migrated to sub-modules.
"""

# For the transitional period, import from the legacy location
# Once fully refactored, GEOService will be implemented here using the sub-modules
from lobster.services.data_access.geo_service import GEOService

__all__ = ["GEOService"]
