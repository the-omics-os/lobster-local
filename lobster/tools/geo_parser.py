"""
Advanced GEO data parser module with optimized performance.

DEPRECATED: This module has been moved to lobster.services.data_access.geo.parser.
Please update your imports. This alias will be removed in a future version.

This module handles parsing GEO files (SOFT, matrix, supplementary) with
modern optimization techniques including Polars integration, intelligent
delimiter detection, and memory-efficient chunked processing.
"""

import warnings

warnings.warn(
    "lobster.tools.geo_parser is deprecated. "
    "Use lobster.services.data_access.geo.parser instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from new location for backward compatibility
from lobster.services.data_access.geo.parser import (
    GEOFormatError,
    GEOParser,
    GEOParserError,
)

__all__ = ["GEOParser", "GEOParserError", "GEOFormatError"]
