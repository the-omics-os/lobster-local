"""
Protocol extraction service package for multi-omics studies.

This package provides modular protocol extraction services for different omics domains:
- amplicon: 16S/ITS microbiome studies (primers, V-regions, PCR, sequencing)
- mass_spec: Proteomics studies (DDA/DIA, MS parameters, database search)
- rnaseq: Transcriptomics studies (library prep, alignment, DE analysis)

Use the factory function get_protocol_service() to obtain the appropriate service
for a given domain.

Examples:
    >>> from lobster.services.metadata.protocol_extraction import get_protocol_service
    >>>
    >>> # Get amplicon service for 16S studies
    >>> service = get_protocol_service("amplicon")
    >>> details, result = service.extract_protocol(methods_text)
    >>>
    >>> # Alternative domain aliases work too
    >>> service = get_protocol_service("16s")  # Same as "amplicon"
    >>> service = get_protocol_service("metagenomics")  # Same as "amplicon"
"""

from typing import Dict

from lobster.services.metadata.protocol_extraction.base import (
    BaseProtocolDetails,
    IProtocolExtractionService,
)

# Lazy imports to avoid circular dependencies and heavy imports at module load
_service_cache: Dict[str, IProtocolExtractionService] = {}


def get_protocol_service(domain: str) -> IProtocolExtractionService:
    """
    Factory function to get the appropriate protocol extraction service.

    Args:
        domain: Domain identifier. Supported values:
            - "amplicon", "16s", "its", "metagenomics" → AmpliconProtocolService
            - "mass_spec", "proteomics", "dda", "dia" → MassSpecProtocolService
            - "rnaseq", "transcriptomics", "bulk_rna" → RNASeqProtocolService

    Returns:
        IProtocolExtractionService instance for the specified domain.

    Raises:
        ValueError: If the domain is not supported.

    Examples:
        >>> service = get_protocol_service("amplicon")
        >>> service.domain
        'amplicon'
        >>> service = get_protocol_service("16s")  # Same service
        >>> service.domain
        'amplicon'
    """
    domain_lower = domain.lower()

    # Check cache first (services are reused)
    if domain_lower in _service_cache:
        return _service_cache[domain_lower]

    # Amplicon/metagenomics domains
    if domain_lower in ("amplicon", "16s", "its", "metagenomics"):
        from lobster.services.metadata.protocol_extraction.amplicon.service import (
            AmpliconProtocolService,
        )

        service = AmpliconProtocolService()
        _service_cache[domain_lower] = service
        return service

    # Mass spectrometry/proteomics domains
    if domain_lower in ("mass_spec", "proteomics", "dda", "dia", "prm", "srm"):
        from lobster.services.metadata.protocol_extraction.mass_spec.service import (
            MassSpecProtocolService,
        )

        service = MassSpecProtocolService()
        _service_cache[domain_lower] = service
        return service

    # RNA-seq/transcriptomics domains
    if domain_lower in ("rnaseq", "transcriptomics", "bulk_rna", "scrna", "rna_seq"):
        from lobster.services.metadata.protocol_extraction.rnaseq.service import (
            RNASeqProtocolService,
        )

        service = RNASeqProtocolService()
        _service_cache[domain_lower] = service
        return service

    raise ValueError(
        f"Unknown domain: '{domain}'. Supported domains: "
        "amplicon/16s/its/metagenomics, mass_spec/proteomics/dda/dia, "
        "rnaseq/transcriptomics/bulk_rna"
    )


def clear_service_cache() -> None:
    """
    Clear the service cache.

    Useful for testing or when you want to force re-initialization of services.
    """
    _service_cache.clear()


__all__ = [
    # Base classes
    "IProtocolExtractionService",
    "BaseProtocolDetails",
    # Factory
    "get_protocol_service",
    "clear_service_cache",
]
