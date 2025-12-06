"""
Protein Structure Fetch Service for downloading and processing PDB structures.

This stateless service provides methods for fetching protein structures from
the RCSB PDB database, caching files locally, and extracting structural metadata
following the Lobster 3-tuple pattern (AnnData, stats, IR).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import anndata
from Bio import PDB

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.tools.providers.pdb_provider import PDBProvider, PDBStructureMetadata
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteinStructureFetchError(Exception):
    """Base exception for protein structure fetch operations."""

    pass


class ProteinStructureFetchService:
    """
    Stateless service for fetching protein structures from PDB database.

    This service implements the Lobster 3-tuple pattern:
    - Returns (AnnData, stats_dict, AnalysisStep)
    - Uses PDB Provider for API access
    - Caches downloaded structures
    - Extracts metadata for integration with omics data
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the protein structure fetch service.

        Args:
            config: Optional configuration dict (for future use)
            **kwargs: Additional arguments (ignored, for backward compatibility)
        """
        logger.debug("Initializing stateless ProteinStructureFetchService")
        self.config = config or {}
        logger.debug("ProteinStructureFetchService initialized successfully")

    def fetch_structure(
        self,
        pdb_id: str,
        format: str = "cif",
        cache_dir: Optional[Path] = None,
        extract_metadata: bool = True,
        data_manager=None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Fetch protein structure from PDB and extract metadata.

        Args:
            pdb_id: PDB identifier (4-character alphanumeric code)
            format: File format ('pdb' or 'cif', default: 'cif')
            cache_dir: Directory for caching downloaded structures
            extract_metadata: Whether to parse structure and extract metadata
            data_manager: Optional DataManagerV2 instance for provenance tracking

        Returns:
            Tuple[Dict, Dict, AnalysisStep]: Structure data dict, stats, and IR

        Raises:
            ProteinStructureFetchError: If fetch or parsing fails
        """
        try:
            logger.info(f"Fetching protein structure: {pdb_id} (format: {format})")

            # Validate PDB ID format
            pdb_id = pdb_id.upper().strip()
            if not self._validate_pdb_id_format(pdb_id):
                raise ProteinStructureFetchError(
                    f"Invalid PDB ID format: {pdb_id}. Must be 4 alphanumeric characters."
                )

            # Initialize PDB provider
            provider = PDBProvider(data_manager=data_manager)

            # Set cache directory from data_manager workspace
            if cache_dir is None:
                if data_manager is None:
                    raise ProteinStructureFetchError(
                        "Either cache_dir or data_manager must be provided. "
                        "Example: fetch_structure(pdb_id, data_manager=data_manager)"
                    )
                cache_dir = data_manager.cache_dir / "protein_structures"
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Check if structure is already cached
            file_extension = "cif" if format == "cif" else "pdb"
            cached_file = cache_dir / f"{pdb_id}.{file_extension}"

            if cached_file.exists():
                logger.info(f"Using cached structure: {cached_file}")
                structure_file = cached_file
                downloaded = False
            else:
                # Download structure
                logger.info(f"Downloading structure {pdb_id} from RCSB PDB...")
                try:
                    structure_file_str = provider.download_structure(
                        structure_id=pdb_id, output_path=str(cached_file), format=format
                    )
                    structure_file = Path(structure_file_str)
                except ValueError as e:
                    raise ProteinStructureFetchError(
                        f"Failed to download structure {pdb_id} from PDB: {e}"
                    )
                downloaded = True

            # Get metadata from provider
            metadata = provider.get_structure_metadata(pdb_id)
            if metadata is None:
                logger.warning(f"Could not retrieve metadata for {pdb_id}")
                metadata = self._create_minimal_metadata(pdb_id)

            # Extract structural information if requested
            structure_info = {}
            if extract_metadata and structure_file.exists():
                structure_info = self._parse_structure_file(structure_file, format)

            # Combine all structure data
            structure_data = {
                "pdb_id": pdb_id,
                "file_path": str(structure_file),
                "file_format": format,
                "metadata": self._metadata_to_dict(metadata),
                "structure_info": structure_info,
                "cached": not downloaded,
            }

            # Calculate statistics
            stats = {
                "pdb_id": pdb_id,
                "file_format": format,
                "file_size_mb": structure_file.stat().st_size / (1024 * 1024),
                "cached": not downloaded,
                "experiment_method": metadata.experiment_method,
                "resolution": metadata.resolution,
                "organism": metadata.organism,
                "n_chains": len(structure_info.get("chains", [])),
                "n_residues": structure_info.get("total_residues", 0),
                "n_atoms": structure_info.get("total_atoms", 0),
                "analysis_type": "protein_structure_fetch",
            }

            logger.info(
                f"Successfully fetched {pdb_id}: {stats['n_chains']} chains, {stats['n_residues']} residues"
            )

            # Create IR for notebook export
            ir = self._create_fetch_structure_ir(pdb_id=pdb_id, format=format)

            return structure_data, stats, ir

        except Exception as e:
            logger.exception(f"Error fetching protein structure {pdb_id}: {e}")
            raise ProteinStructureFetchError(
                f"Failed to fetch structure {pdb_id}: {str(e)}"
            )

    def link_structures_to_genes(
        self,
        adata: anndata.AnnData,
        gene_column: str = "gene_symbol",
        organism: str = "Homo sapiens",
        max_structures_per_gene: int = 5,
        data_manager=None,
    ) -> Tuple[anndata.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Link gene expression data to protein structures from PDB.

        Args:
            adata: AnnData object with gene/protein data
            gene_column: Column in adata.var containing gene symbols
            organism: Source organism for structure search
            max_structures_per_gene: Maximum structures to fetch per gene
            data_manager: Optional DataManagerV2 instance

        Returns:
            Tuple[AnnData, Dict, AnalysisStep]: Updated AnnData with structure links, stats, IR

        Raises:
            ProteinStructureFetchError: If linking fails
        """
        try:
            logger.info(
                f"Linking genes to protein structures (organism: {organism})..."
            )

            # Create working copy
            adata_linked = adata.copy()

            # Validate gene column exists
            if gene_column not in adata_linked.var.columns:
                raise ProteinStructureFetchError(
                    f"Gene column '{gene_column}' not found in adata.var. "
                    f"Available columns: {list(adata_linked.var.columns)}"
                )

            # Initialize PDB provider
            provider = PDBProvider(data_manager=data_manager)

            # Track structure mappings
            structure_mappings = {}
            genes_with_structures = 0
            total_structures_found = 0

            # Search for structures for each gene
            genes = adata_linked.var[gene_column].unique()
            logger.info(f"Searching structures for {len(genes)} genes...")

            for gene_symbol in genes[:100]:  # Limit to first 100 genes for performance
                try:
                    # Search PDB for this gene
                    search_query = f"{gene_symbol} {organism}"
                    results = provider.search_publications(
                        query=search_query, max_results=max_structures_per_gene
                    )

                    if results:
                        pdb_ids = [r.uid for r in results]
                        structure_mappings[gene_symbol] = pdb_ids
                        genes_with_structures += 1
                        total_structures_found += len(pdb_ids)

                except Exception as e:
                    logger.debug(f"No structures found for gene {gene_symbol}: {e}")
                    continue

            # Add structure mappings to AnnData
            adata_linked.var["pdb_structures"] = adata_linked.var[gene_column].map(
                lambda g: ",".join(structure_mappings.get(g, []))
            )
            adata_linked.var["has_structure"] = (
                adata_linked.var["pdb_structures"].str.len() > 0
            )

            # Store mapping details in uns
            adata_linked.uns["protein_structure_links"] = {
                "gene_column": gene_column,
                "organism": organism,
                "max_structures_per_gene": max_structures_per_gene,
                "genes_searched": len(genes),
                "genes_with_structures": genes_with_structures,
                "total_structures_found": total_structures_found,
                "structure_mappings": structure_mappings,
            }

            # Calculate statistics
            stats = {
                "genes_searched": len(genes),
                "genes_with_structures": genes_with_structures,
                "genes_with_structures_pct": (genes_with_structures / len(genes)) * 100,
                "total_structures_found": total_structures_found,
                "avg_structures_per_gene": (
                    total_structures_found / genes_with_structures
                    if genes_with_structures > 0
                    else 0
                ),
                "organism": organism,
                "analysis_type": "link_genes_to_structures",
            }

            logger.info(
                f"Linked {genes_with_structures} genes to {total_structures_found} structures"
            )

            # Create IR
            ir = self._create_link_genes_ir(
                gene_column=gene_column,
                organism=organism,
                max_structures_per_gene=max_structures_per_gene,
            )

            return adata_linked, stats, ir

        except Exception as e:
            logger.exception(f"Error linking genes to structures: {e}")
            raise ProteinStructureFetchError(
                f"Failed to link genes to structures: {str(e)}"
            )

    def _validate_pdb_id_format(self, pdb_id: str) -> bool:
        """Validate PDB ID format (4 alphanumeric characters)."""
        return len(pdb_id) == 4 and pdb_id.isalnum()

    def _parse_structure_file(
        self, structure_file: Path, format: str
    ) -> Dict[str, Any]:
        """Parse structure file using BioPython and extract information."""
        try:
            parser = (
                PDB.MMCIFParser(QUIET=True)
                if format == "cif"
                else PDB.PDBParser(QUIET=True)
            )
            structure = parser.get_structure("protein", structure_file)

            # Extract chain information
            chains = []
            total_residues = 0
            total_atoms = 0

            for model in structure:
                for chain in model:
                    chain_id = chain.get_id()
                    residues = list(chain.get_residues())
                    atoms = list(chain.get_atoms())

                    chains.append(
                        {
                            "chain_id": chain_id,
                            "n_residues": len(residues),
                            "n_atoms": len(atoms),
                        }
                    )

                    total_residues += len(residues)
                    total_atoms += len(atoms)

            return {
                "chains": chains,
                "total_residues": total_residues,
                "total_atoms": total_atoms,
                "n_models": len(structure),
            }

        except Exception as e:
            logger.warning(f"Failed to parse structure file: {e}")
            return {}

    def _metadata_to_dict(self, metadata: PDBStructureMetadata) -> Dict[str, Any]:
        """Convert PDBStructureMetadata to dictionary."""
        return {
            "pdb_id": metadata.pdb_id,
            "title": metadata.title,
            "experiment_method": metadata.experiment_method,
            "resolution": metadata.resolution,
            "organism": metadata.organism,
            "chains": metadata.chains,
            "ligands": metadata.ligands,
            "deposition_date": metadata.deposition_date,
            "release_date": metadata.release_date,
            "authors": metadata.authors,
            "publication_doi": metadata.publication_doi,
            "citation": metadata.citation,
        }

    def _create_minimal_metadata(self, pdb_id: str) -> PDBStructureMetadata:
        """Create minimal metadata when API fetch fails."""
        return PDBStructureMetadata(
            pdb_id=pdb_id,
            title="Unknown",
            experiment_method="UNKNOWN",
            resolution=None,
            organism=None,
            chains=[],
            ligands=[],
            deposition_date=None,
            release_date=None,
            authors=[],
            publication_doi=None,
            citation=None,
        )

    def _create_fetch_structure_ir(self, pdb_id: str, format: str) -> AnalysisStep:
        """Create Intermediate Representation for structure fetch operation."""
        parameter_schema = {
            "pdb_id": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=pdb_id,
                required=True,
                validation_rule="len(pdb_id) == 4 and pdb_id.isalnum()",
                description="PDB identifier (4-character code)",
            ),
            "format": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=format,
                required=False,
                validation_rule="format in ['pdb', 'cif']",
                description="Structure file format",
            ),
        }

        code_template = """# Fetch protein structure from RCSB PDB
import requests
from pathlib import Path

pdb_id = "{{ pdb_id }}"
format = "{{ format }}"

# Download structure file
download_url = f"https://files.rcsb.org/download/{pdb_id}.{format}"
response = requests.get(download_url)
response.raise_for_status()

# Save to file
output_file = Path(f"{pdb_id}.{format}")
output_file.write_bytes(response.content)

print(f"Downloaded structure {pdb_id} to {output_file}")
print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
"""

        return AnalysisStep(
            operation="pdb.fetch_structure",
            tool_name="fetch_protein_structure",
            description=f"Fetch protein structure {pdb_id} from RCSB PDB in {format} format",
            library="rcsb_pdb",
            code_template=code_template,
            imports=["import requests", "from pathlib import Path"],
            parameters={"pdb_id": pdb_id, "format": format},
            parameter_schema=parameter_schema,
            input_entities=[],
            output_entities=[f"{pdb_id}.{format}"],
            execution_context={
                "operation_type": "data_retrieval",
                "database": "RCSB PDB",
                "data_type": "protein_structure",
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_link_genes_ir(
        self, gene_column: str, organism: str, max_structures_per_gene: int
    ) -> AnalysisStep:
        """Create IR for linking genes to structures."""
        parameter_schema = {
            "gene_column": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=gene_column,
                required=True,
                description="Column in adata.var containing gene symbols",
            ),
            "organism": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=organism,
                required=False,
                description="Source organism for structure search",
            ),
            "max_structures_per_gene": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=max_structures_per_gene,
                required=False,
                description="Maximum structures to fetch per gene",
            ),
        }

        code_template = """# Link gene expression data to protein structures
import requests
import pandas as pd

gene_column = "{{ gene_column }}"
organism = "{{ organism }}"
max_structures = {{ max_structures_per_gene }}

# Search PDB for each gene
search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
structure_mappings = {}

for gene in adata.var[gene_column].unique()[:100]:
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {"value": f"{gene} {organism}"}
        },
        "return_type": "entry"
    }

    response = requests.post(search_url, json=query)
    if response.ok:
        results = response.json().get("result_set", [])
        pdb_ids = [r["identifier"] for r in results[:max_structures]]
        if pdb_ids:
            structure_mappings[gene] = pdb_ids

# Add structure links to AnnData
adata.var["pdb_structures"] = adata.var[gene_column].map(
    lambda g: ",".join(structure_mappings.get(g, []))
)
adata.var["has_structure"] = adata.var["pdb_structures"].str.len() > 0

print(f"Linked {len(structure_mappings)} genes to protein structures")
"""

        return AnalysisStep(
            operation="pdb.link_genes_to_structures",
            tool_name="link_to_expression_data",
            description=f"Link gene expression data to PDB structures for {organism}",
            library="rcsb_pdb",
            code_template=code_template,
            imports=["import requests", "import pandas as pd"],
            parameters={
                "gene_column": gene_column,
                "organism": organism,
                "max_structures_per_gene": max_structures_per_gene,
            },
            parameter_schema=parameter_schema,
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "operation_type": "data_integration",
                "database": "RCSB PDB",
                "data_type": "gene_structure_mapping",
            },
            validates_on_export=True,
            requires_validation=False,
        )
