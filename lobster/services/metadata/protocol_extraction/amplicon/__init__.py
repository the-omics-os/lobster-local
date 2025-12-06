"""
Amplicon protocol extraction package for 16S/ITS microbiome studies.

This package provides protocol extraction for amplicon-based metagenomics,
including primer detection, V-region extraction, and PCR condition parsing.

Examples:
    >>> from lobster.services.metadata.protocol_extraction.amplicon import (
    ...     AmpliconProtocolService,
    ...     AmpliconProtocolDetails,
    ... )
    >>> service = AmpliconProtocolService()
    >>> details, result = service.extract_protocol(text)
"""

from lobster.services.metadata.protocol_extraction.amplicon.details import (
    V_REGIONS,
    AmpliconProtocolDetails,
)
from lobster.services.metadata.protocol_extraction.amplicon.service import (
    AmpliconProtocolService,
)

__all__ = [
    "AmpliconProtocolService",
    "AmpliconProtocolDetails",
    "V_REGIONS",
]
