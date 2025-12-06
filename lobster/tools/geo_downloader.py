"""
GEO database downloader and parser.

DEPRECATED: This module has been moved to lobster.services.data_access.geo.downloader.
Please update your imports. This alias will be removed in a future version.

This module handles downloading and parsing data from the Gene Expression Omnibus (GEO)
database, providing structured access to gene expression datasets.
"""

import warnings

warnings.warn(
    "lobster.tools.geo_downloader is deprecated. "
    "Use lobster.services.data_access.geo.downloader instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from new location for backward compatibility
from lobster.services.data_access.geo.downloader import (
    GEODownloadError,
    GEODownloadManager,
)

__all__ = ["GEODownloadError", "GEODownloadManager"]
