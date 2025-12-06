"""
Mass spectrometry protocol extraction package for proteomics studies.

This package provides protocol extraction for mass spectrometry-based proteomics,
including acquisition mode, instrument detection, and sample preparation parsing.

Note: This is currently a stub implementation. Full functionality will be
added in future versions.

Examples:
    >>> from lobster.services.metadata.protocol_extraction.mass_spec import (
    ...     MassSpecProtocolService,
    ...     MassSpecProtocolDetails,
    ... )
    >>> service = MassSpecProtocolService()
    >>> service.domain
    'mass_spec'
"""

from lobster.services.metadata.protocol_extraction.mass_spec.details import (
    MassSpecProtocolDetails,
)
from lobster.services.metadata.protocol_extraction.mass_spec.service import (
    MassSpecProtocolService,
)

__all__ = [
    "MassSpecProtocolService",
    "MassSpecProtocolDetails",
]
