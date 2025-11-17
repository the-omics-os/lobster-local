"""
PDB provider implementation for protein structure database access.

This provider implements search and download capabilities for RCSB Protein Data Bank
using the PDB REST API, supporting structure search, metadata extraction, and file downloads.
"""

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.structure_provider import (
    BaseStructureProvider,
    StructureMetadata,
    StructureSource,
    StructureProviderCapability,
)

logger = logging.getLogger(__name__)


class PDBFileFormat(str, Enum):
    """Supported PDB file formats."""

    PDB = "pdb"  # Legacy PDB format
    CIF = "cif"  # mmCIF format (official standard)
    MMCIF = "mmcif"  # Alternative name for CIF
    PDBX = "pdbx"  # PDBx format


class PDBExperimentMethod(str, Enum):
    """PDB experimental methods."""

    XRAY = "X-RAY DIFFRACTION"
    NMR = "SOLUTION NMR"
    CRYO_EM = "ELECTRON MICROSCOPY"
    NEUTRON = "NEUTRON DIFFRACTION"
    FIBER = "FIBER DIFFRACTION"
    ELECTRON_CRYST = "ELECTRON CRYSTALLOGRAPHY"
    SOLID_STATE_NMR = "SOLID-STATE NMR"
    PREDICTED = "PREDICTED"  # AlphaFold structures


# Note: PDBStructureMetadata is kept for backward compatibility but uses StructureMetadata from base
PDBStructureMetadata = StructureMetadata


class PDBSearchFilters(BaseModel):
    """Filters for PDB structure searches."""

    organism: Optional[str] = None  # Source organism
    experiment_method: Optional[str] = None  # X-ray, NMR, cryo-EM
    resolution_min: Optional[float] = None  # Minimum resolution (Angstroms)
    resolution_max: Optional[float] = None  # Maximum resolution (Angstroms)
    release_date_min: Optional[str] = None  # YYYY-MM-DD format
    release_date_max: Optional[str] = None  # YYYY-MM-DD format
    has_ligands: Optional[bool] = None  # Has bound ligands
    max_results: int = Field(default=10, ge=1, le=100)


class PDBProviderConfig(BaseModel):
    """Configuration for PDB provider."""

    # API settings
    base_url: str = "https://data.rcsb.org/rest/v1"
    search_url: str = "https://search.rcsb.org/rcsbsearch/v2/query"
    files_url: str = "https://files.rcsb.org/download"

    # Rate limiting
    max_requests_per_second: float = 5.0
    max_retry: int = 3
    retry_delay: float = 1.0

    # Download settings
    default_format: PDBFileFormat = PDBFileFormat.CIF
    cache_downloads: bool = True
    download_timeout: int = 30


class PDBProvider(BaseStructureProvider):
    """
    Provider for RCSB Protein Data Bank (PDB) database.

    This provider implements structure search, metadata extraction, and
    file download capabilities using the PDB REST API.

    Capabilities:
    - SEARCH_STRUCTURES: Search PDB by keywords
    - GET_METADATA: Get structure metadata
    - DOWNLOAD_STRUCTURE: Download structure files (PDB, mmCIF)
    """

    def __init__(
        self,
        data_manager: DataManagerV2,
        config: Optional[PDBProviderConfig] = None,
    ):
        """
        Initialize PDB provider.

        Args:
            data_manager: DataManagerV2 instance for provenance tracking
            config: Optional configuration, uses defaults if not provided
        """
        self.data_manager = data_manager
        self.config = config or PDBProviderConfig()

        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 1.0 / self.config.max_requests_per_second

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Lobster-Bioinformatics/1.0 (omics-os.com)"
        })

    @property
    def source(self) -> StructureSource:
        """Return PDB as the structure source."""
        return StructureSource.PDB

    @property
    def priority(self) -> int:
        """
        Return provider priority.

        PDB has high priority (10) as the authoritative source for protein structures.

        Returns:
            int: Priority 10 (high priority)
        """
        return 10

    def get_supported_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities supported by PDB provider.

        Returns:
            Dict[str, bool]: Capability mapping
        """
        return {
            StructureProviderCapability.SEARCH_STRUCTURES: True,
            StructureProviderCapability.GET_METADATA: True,
            StructureProviderCapability.DOWNLOAD_STRUCTURE: True,
        }

    def search_structures(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[StructureMetadata]:
        """
        Search PDB database for structures matching query.

        Args:
            query: Search query (protein name, PDB ID, keywords)
            max_results: Maximum number of results
            filters: Optional filters (not yet implemented)
            **kwargs: Additional search parameters

        Returns:
            List[StructureMetadata]: Search results
        """
        logger.info(f"Searching PDB for: {query} (max_results={max_results})")

        # Build search query
        search_payload = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "value": query
                }
            },
            "return_type": "entry",
            "request_options": {
                "results_content_type": ["experimental"],
                "return_all_hits": False,
                "sort": [{"sort_by": "score", "direction": "desc"}]
            }
        }

        try:
            self._rate_limit()
            response = self.session.post(
                self.config.search_url,
                json=search_payload,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            pdb_ids = [hit["identifier"] for hit in data.get("result_set", [])[:max_results]]

            # Get metadata for each PDB ID
            results = []
            for pdb_id in pdb_ids:
                try:
                    metadata = self.get_structure_metadata(pdb_id)
                    if metadata:
                        results.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to get metadata for {pdb_id}: {e}")
                    continue

            logger.info(f"Found {len(results)} PDB structures")
            return results

        except Exception as e:
            logger.error(f"PDB search failed: {e}")
            return []

    def get_structure_metadata(self, pdb_id: str, **kwargs) -> Optional[StructureMetadata]:
        """
        Get detailed metadata for a PDB structure.

        Args:
            pdb_id: PDB identifier (4 characters)
            **kwargs: Additional parameters (not used, for interface compatibility)

        Returns:
            Optional[StructureMetadata]: Structure metadata or None if not found
        """
        pdb_id = pdb_id.upper()
        logger.info(f"Fetching metadata for PDB ID: {pdb_id}")

        url = f"{self.config.base_url}/core/entry/{pdb_id}"

        try:
            self._rate_limit()
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Extract metadata
            metadata = StructureMetadata(
                pdb_id=pdb_id,
                title=data.get("struct", {}).get("title", "Unknown"),
                experiment_method=data.get("exptl", [{}])[0].get("method", "UNKNOWN"),
                resolution=self._extract_resolution(data),
                organism=self._extract_organism(data),
                chains=self._extract_chains(data),
                ligands=self._extract_ligands(data),
                deposition_date=data.get("rcsb_accession_info", {}).get("deposit_date"),
                release_date=data.get("rcsb_accession_info", {}).get("initial_release_date"),
                authors=self._extract_authors(data),
                publication_doi=self._extract_doi(data),
                citation=self._extract_citation(data),
            )

            return metadata

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"PDB ID not found: {pdb_id}")
            else:
                logger.error(f"HTTP error fetching {pdb_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch metadata for {pdb_id}: {e}")
            return None

    def download_structure(
        self,
        structure_id: str,
        output_path: str,
        format: str = "cif",
        **kwargs,
    ) -> str:
        """
        Download structure file from PDB.

        Args:
            structure_id: PDB identifier
            output_path: Output path for the downloaded file
            format: File format (pdb, cif, mmcif)
            **kwargs: Additional parameters (not used)

        Returns:
            str: Path to downloaded file

        Raises:
            ValueError: If format is unsupported or download fails
        """
        pdb_id = structure_id.upper()
        format = format.lower()

        if format not in ["pdb", "cif", "mmcif"]:
            logger.error(f"Unsupported format: {format}. Use 'pdb' or 'cif'")
            raise ValueError(f"Unsupported format: {format}. Use 'pdb' or 'cif'")

        # Normalize format names
        if format == "mmcif":
            format = "cif"

        # Construct download URL
        extension = "cif" if format == "cif" else "pdb"
        url = f"{self.config.files_url}/{pdb_id}.{extension}"

        # Convert output_path to Path object and ensure parent directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {pdb_id} ({format}) to {output_path}")

        try:
            self._rate_limit()
            response = self.session.get(
                url,
                timeout=self.config.download_timeout,
                stream=True
            )
            response.raise_for_status()

            # Write file
            with open(output_path_obj, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Successfully downloaded {pdb_id} to {output_path}")
            return str(output_path_obj)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Structure file not found: {pdb_id}.{extension}")
                raise ValueError(f"Structure file not found: {pdb_id}.{extension}")
            else:
                logger.error(f"HTTP error downloading {pdb_id}: {e}")
                raise ValueError(f"HTTP error downloading {pdb_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to download {pdb_id}: {e}")
            raise ValueError(f"Failed to download {pdb_id}: {e}")

    def validate_pdb_id(self, pdb_id: str) -> bool:
        """
        Validate PDB ID format and existence.

        Args:
            pdb_id: PDB identifier to validate

        Returns:
            bool: True if valid and exists, False otherwise
        """
        # Format check: 4 alphanumeric characters
        if not isinstance(pdb_id, str) or len(pdb_id) != 4:
            return False

        if not pdb_id.isalnum():
            return False

        # Existence check
        metadata = self.get_structure_metadata(pdb_id)
        return metadata is not None

    def _rate_limit(self):
        """Implement rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def _extract_resolution(self, data: Dict[str, Any]) -> Optional[float]:
        """Extract resolution from PDB metadata."""
        try:
            refine = data.get("refine", [])
            if refine:
                return float(refine[0].get("ls_d_res_high"))
        except (ValueError, TypeError, KeyError):
            pass

        try:
            # Try em_3d_reconstruction for cryo-EM
            em_3d = data.get("em_3d_reconstruction", [])
            if em_3d:
                return float(em_3d[0].get("resolution"))
        except (ValueError, TypeError, KeyError):
            pass

        return None

    def _extract_organism(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract source organism from PDB metadata."""
        try:
            entity_src_gen = data.get("entity_src_gen", [])
            if entity_src_gen:
                return entity_src_gen[0].get("pdbx_gene_src_scientific_name")
        except (TypeError, KeyError):
            pass

        try:
            entity_src_nat = data.get("entity_src_nat", [])
            if entity_src_nat:
                return entity_src_nat[0].get("pdbx_organism_scientific")
        except (TypeError, KeyError):
            pass

        return None

    def _extract_chains(self, data: Dict[str, Any]) -> List[str]:
        """Extract chain identifiers from PDB metadata."""
        chains = []
        try:
            struct_ref = data.get("struct_ref", [])
            for ref in struct_ref:
                chain_id = ref.get("pdbx_strand_id", "")
                if chain_id:
                    chains.extend(chain_id.split(","))
        except (TypeError, KeyError):
            pass

        return sorted(set(c.strip() for c in chains if c))

    def _extract_ligands(self, data: Dict[str, Any]) -> List[str]:
        """Extract bound ligands from PDB metadata."""
        ligands = []
        try:
            pdbx_entity_nonpoly = data.get("pdbx_entity_nonpoly", [])
            for entity in pdbx_entity_nonpoly:
                comp_id = entity.get("comp_id")
                if comp_id and comp_id not in ["HOH", "WAT"]:  # Exclude water
                    ligands.append(comp_id)
        except (TypeError, KeyError):
            pass

        return ligands

    def _extract_authors(self, data: Dict[str, Any]) -> List[str]:
        """Extract author names from PDB metadata."""
        authors = []
        try:
            audit_author = data.get("audit_author", [])
            for author in audit_author:
                name = author.get("name")
                if name:
                    authors.append(name)
        except (TypeError, KeyError):
            pass

        return authors

    def _extract_doi(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract publication DOI from PDB metadata."""
        try:
            citation = data.get("citation", [])
            if citation:
                return citation[0].get("pdbx_database_id_DOI")
        except (TypeError, KeyError):
            pass

        return None

    def _extract_citation(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract publication citation from PDB metadata."""
        try:
            citation = data.get("citation", [])
            if citation:
                cit = citation[0]
                title = cit.get("title", "")
                journal = cit.get("journal_abbrev", "")
                year = cit.get("year", "")
                return f"{title} ({journal}, {year})" if all([title, journal, year]) else None
        except (TypeError, KeyError):
            pass

        return None

    def search_by_filters(
        self, filters: PDBSearchFilters
    ) -> List[StructureMetadata]:
        """
        Search PDB with advanced filters.

        Args:
            filters: Search filters

        Returns:
            List[StructureMetadata]: Matching structures
        """
        # Build complex search query
        query_parts = []

        if filters.organism:
            query_parts.append({
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entity_source_organism.scientific_name",
                    "operator": "exact_match",
                    "value": filters.organism
                }
            })

        if filters.experiment_method:
            query_parts.append({
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "exptl.method",
                    "operator": "exact_match",
                    "value": filters.experiment_method
                }
            })

        if filters.resolution_max:
            query_parts.append({
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": filters.resolution_max
                }
            })

        # Combine query parts
        if len(query_parts) == 0:
            return []
        elif len(query_parts) == 1:
            query = query_parts[0]
        else:
            query = {
                "type": "group",
                "logical_operator": "and",
                "nodes": query_parts
            }

        search_payload = {
            "query": query,
            "return_type": "entry",
            "request_options": {
                "results_content_type": ["experimental"],
                "return_all_hits": False
            }
        }

        try:
            self._rate_limit()
            response = self.session.post(
                self.config.search_url,
                json=search_payload,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            pdb_ids = [hit["identifier"] for hit in data.get("result_set", [])[:filters.max_results]]

            # Get metadata for each
            results = []
            for pdb_id in pdb_ids:
                metadata = self.get_structure_metadata(pdb_id)
                if metadata:
                    results.append(metadata)

            return results

        except Exception as e:
            logger.error(f"Filtered PDB search failed: {e}")
            return []
