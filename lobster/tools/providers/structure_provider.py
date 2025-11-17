"""
Abstract base class for structure providers (PDB, AlphaFold, etc.).

This module defines the interface for protein structure providers,
separate from publication and dataset providers, following the
Interface Segregation Principle (ISP).
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StructureSource(Enum):
    """Enum for different structure sources."""

    PDB = "pdb"  # RCSB Protein Data Bank
    ALPHAFOLD = "alphafold"  # AlphaFold DB
    MODBASE = "modbase"  # ModBase comparative models
    SWISS_MODEL = "swiss_model"  # SWISS-MODEL repository


class StructureMetadata(BaseModel):
    """Standard protein structure metadata."""

    pdb_id: str
    title: str
    experiment_method: str
    resolution: Optional[float] = None
    organism: Optional[str] = None
    chains: List[str] = []
    ligands: List[str] = []
    deposition_date: Optional[str] = None
    release_date: Optional[str] = None
    authors: List[str] = []
    publication_doi: Optional[str] = None
    citation: Optional[str] = None


class BaseStructureProvider(ABC):
    """
    Abstract base class for protein structure providers.

    This interface is specific to structure databases (PDB, AlphaFold)
    and does NOT force implementation of publication/dataset methods,
    following the Interface Segregation Principle.

    Use this for providers that:
    - Fetch protein structures (3D coordinates)
    - Provide structural metadata (resolution, method, etc.)
    - Do NOT handle publications or genomic datasets
    """

    @property
    @abstractmethod
    def source(self) -> StructureSource:
        """Return the structure source this provider handles."""
        pass

    @abstractmethod
    def search_structures(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[StructureMetadata]:
        """
        Search for protein structures using provider-specific implementation.

        Args:
            query: Search query string (protein name, gene name, keyword)
            max_results: Maximum number of results to return
            filters: Optional filters (organism, resolution, method, etc.)
            **kwargs: Provider-specific additional parameters

        Returns:
            List[StructureMetadata]: List of structure metadata results
        """
        pass

    @abstractmethod
    def get_structure_metadata(self, structure_id: str, **kwargs) -> StructureMetadata:
        """
        Extract standardized metadata for a specific structure.

        Args:
            structure_id: Structure identifier (e.g., PDB ID '1AKE')
            **kwargs: Provider-specific additional parameters

        Returns:
            StructureMetadata: Standardized structure metadata
        """
        pass

    @abstractmethod
    def download_structure(
        self,
        structure_id: str,
        output_path: str,
        format: str = "cif",
        **kwargs,
    ) -> str:
        """
        Download structure file.

        Args:
            structure_id: Structure identifier (e.g., PDB ID '1AKE')
            output_path: Path to save the structure file
            format: File format ('pdb', 'cif', 'mmcif', etc.)
            **kwargs: Provider-specific additional parameters

        Returns:
            str: Path to downloaded file
        """
        pass

    @property
    def priority(self) -> int:
        """
        Return provider priority for routing.

        Lower values = higher priority.

        Priority Guidelines:
        - 10: High priority (authoritative sources like RCSB PDB)
        - 50: Medium priority (predicted structures like AlphaFold)
        - 100: Low priority (user-submitted or low-quality sources)

        Returns:
            int: Provider priority (default 100)
        """
        return 100

    def get_supported_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities supported by this structure provider.

        Returns:
            Dict[str, bool]: Capability identifiers and support status
        """
        return {
            "search_structures": True,
            "get_metadata": True,
            "download_structure": True,
        }


class StructureProviderCapability:
    """
    Standard capability identifiers for structure providers.

    Defines operations that structure providers can support.
    """

    SEARCH_STRUCTURES = "search_structures"
    """Search for protein structures by name, keyword, or gene."""

    GET_METADATA = "get_metadata"
    """Extract structural metadata (resolution, method, organism, etc.)."""

    DOWNLOAD_STRUCTURE = "download_structure"
    """Download structure files in various formats (PDB, CIF, mmCIF)."""

    LINK_TO_GENES = "link_to_genes"
    """Link structures to gene expression data."""

    SEARCH_BY_SEQUENCE = "search_by_sequence"
    """Search structures by protein sequence similarity."""

    GET_BIOLOGICAL_ASSEMBLY = "get_biological_assembly"
    """Get biological assembly coordinates (multimer structures)."""
