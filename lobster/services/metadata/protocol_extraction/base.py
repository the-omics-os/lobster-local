"""
Base interface for protocol extraction services.

This module defines the abstract base class for all protocol extraction services,
enabling modular support for different omics domains (amplicon, mass_spec, rnaseq, etc.).

Following the IDownloadService pattern from lobster.core.interfaces.download_service.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from lobster.core.interfaces.validator import ValidationResult


@dataclass
class BaseProtocolDetails:
    """
    Base protocol details - common across all omics domains.

    Subclasses add domain-specific fields (e.g., primers for amplicon,
    acquisition_mode for mass_spec, library_prep for rnaseq).
    """

    # Extraction confidence and validation
    confidence: float = 0.0
    extraction_notes: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values and empty lists."""
        return {k: v for k, v in self.__dict__.items() if v is not None and v != []}


class IProtocolExtractionService(ABC):
    """
    Abstract base class for protocol extraction services.

    Each omics domain (amplicon, mass_spec, rnaseq) implements this interface
    with domain-specific extraction logic and resource files.

    Examples:
        >>> from lobster.services.metadata.protocol_extraction import get_protocol_service
        >>> service = get_protocol_service("amplicon")
        >>> details, result = service.extract_protocol(methods_text)
        >>> details.to_dict()
        {'v_region': 'V3-V4', 'forward_primer': '515F', ...}
    """

    @property
    @abstractmethod
    def domain(self) -> str:
        """
        Return domain identifier.

        Returns:
            Domain string: 'amplicon', 'mass_spec', 'rnaseq', etc.
        """
        pass

    @classmethod
    @abstractmethod
    def supports_domain(cls, domain: str) -> bool:
        """
        Check if this service handles the given domain.

        Args:
            domain: Domain identifier to check.

        Returns:
            True if this service handles the domain.
        """
        pass

    @abstractmethod
    def extract_protocol(
        self, text: str, source: str = "unknown"
    ) -> Tuple[BaseProtocolDetails, ValidationResult]:
        """
        Extract protocol details from publication text.

        Args:
            text: Publication text (methods section preferred).
            source: Source identifier for logging.

        Returns:
            Tuple of (domain-specific ProtocolDetails, ValidationResult)
        """
        pass

    @abstractmethod
    def load_resources(self) -> Dict[str, Any]:
        """
        Load domain-specific reference data from JSON files.

        Returns:
            Dictionary containing loaded resource data.
        """
        pass

    def get_service_info(self) -> Dict[str, Any]:
        """
        Return service metadata.

        Returns:
            Dictionary with service_name and domain.
        """
        return {
            "service_name": self.__class__.__name__,
            "domain": self.domain,
        }
