"""
RNA-seq protocol extraction package for transcriptomics studies.

This package provides protocol extraction for RNA sequencing-based transcriptomics,
including library preparation, sequencing parameters, and analysis pipeline parsing.

Note: This is currently a stub implementation. Full functionality will be
added in future versions.

Examples:
    >>> from lobster.services.metadata.protocol_extraction.rnaseq import (
    ...     RNASeqProtocolService,
    ...     RNASeqProtocolDetails,
    ... )
    >>> service = RNASeqProtocolService()
    >>> service.domain
    'rnaseq'
"""

from lobster.services.metadata.protocol_extraction.rnaseq.details import (
    RNASeqProtocolDetails,
)
from lobster.services.metadata.protocol_extraction.rnaseq.service import (
    RNASeqProtocolService,
)

__all__ = [
    "RNASeqProtocolService",
    "RNASeqProtocolDetails",
]
