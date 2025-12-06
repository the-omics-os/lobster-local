"""
RNA-seq protocol extraction service for transcriptomics studies.

This service extracts technical protocol information from RNA-seq publication text.
Currently a stub implementation - full extraction logic to be implemented.
"""

from typing import Any, Dict, Tuple

from lobster.core.interfaces.validator import ValidationResult
from lobster.services.metadata.protocol_extraction.base import (
    IProtocolExtractionService,
)
from lobster.services.metadata.protocol_extraction.rnaseq.details import (
    RNASeqProtocolDetails,
)


class RNASeqProtocolService(IProtocolExtractionService):
    """
    Service for extracting RNA-seq transcriptomics protocol details.

    This is a stub implementation. Full extraction logic for library preparation,
    sequencing parameters, alignment, and differential expression analysis will
    be implemented in future versions.

    Planned extractions:
    - Library preparation (kit, strand specificity, selection method)
    - Sequencing parameters (platform, read length, depth)
    - Alignment settings (aligner, reference genome)
    - Quantification method (featureCounts, Salmon, kallisto)
    - Differential expression tools (DESeq2, edgeR, limma)
    - Quality control tools (FastQC, MultiQC)

    Examples:
        >>> service = RNASeqProtocolService()
        >>> service.domain
        'rnaseq'
        >>> service.supports_domain("transcriptomics")
        True
    """

    # Supported domain aliases
    SUPPORTED_DOMAINS = {"rnaseq", "transcriptomics", "bulk_rna", "scrna", "rna_seq"}

    def __init__(self):
        """Initialize the service."""
        pass

    @property
    def domain(self) -> str:
        """Return domain identifier."""
        return "rnaseq"

    @classmethod
    def supports_domain(cls, domain: str) -> bool:
        """Check if this service handles the given domain."""
        return domain.lower() in cls.SUPPORTED_DOMAINS

    def load_resources(self) -> Dict[str, Any]:
        """Load domain-specific reference data from JSON files."""
        # Stub: return empty dict until resources are created
        return {}

    def extract_protocol(
        self, text: str, source: str = "unknown"
    ) -> Tuple[RNASeqProtocolDetails, ValidationResult]:
        """
        Extract protocol details from publication text.

        Args:
            text: Publication text (methods section preferred)
            source: Source identifier for logging

        Returns:
            Tuple of (RNASeqProtocolDetails, ValidationResult)

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError(
            "RNASeqProtocolService.extract_protocol() is not yet implemented. "
            "This is a planned feature for transcriptomics protocol extraction."
        )
