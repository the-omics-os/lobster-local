"""
Protein structure data adapter with PDB and mmCIF format support.

This module provides the ProteinStructureAdapter that handles loading,
validation, and preprocessing of protein structure data from PDB, mmCIF,
and other structural formats using BioPython.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import anndata
import numpy as np
import pandas as pd

from lobster.core.adapters.base import BaseAdapter
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.protein_structure import ProteinStructureSchema

logger = logging.getLogger(__name__)


class ProteinStructureAdapter(BaseAdapter):
    """
    Adapter for protein structure data with schema enforcement.

    This adapter handles loading and validation of protein structure data
    from PDB, mmCIF, and other formats using BioPython's Bio.PDB module,
    converting structures to AnnData format where atoms are observations.
    """

    def __init__(
        self,
        strict_validation: bool = False,
        include_hetero: bool = True,
        include_waters: bool = False,
        model_number: int = 0,
    ):
        """
        Initialize the protein structure adapter.

        Args:
            strict_validation: Whether to use strict validation
            include_hetero: Whether to include HETATM records (ligands, etc.)
            include_waters: Whether to include water molecules
            model_number: Which model to use (for NMR structures with multiple models)
        """
        super().__init__(name="ProteinStructureAdapter")

        self.strict_validation = strict_validation
        self.include_hetero = include_hetero
        self.include_waters = include_waters
        self.model_number = model_number

        # Create validator
        self.validator = ProteinStructureSchema.create_validator(
            strict=strict_validation
        )

        # Get QC thresholds
        self.qc_thresholds = ProteinStructureSchema.get_recommended_qc_thresholds()

    def from_source(
        self, source: Union[str, Path, anndata.AnnData], **kwargs
    ) -> anndata.AnnData:
        """
        Convert source data to AnnData with protein structure schema.

        Args:
            source: Data source (file path or AnnData)
            **kwargs: Additional parameters:
                - pdb_id: PDB identifier (if not in filename)
                - model_number: Model to use for NMR structures (overrides init)
                - include_hetero: Include HETATM records (overrides init)
                - include_waters: Include water molecules (overrides init)

        Returns:
            anndata.AnnData: Loaded and validated structure data

        Raises:
            ValueError: If source data is invalid
            FileNotFoundError: If source file doesn't exist
            ImportError: If BioPython is not installed
        """
        self._log_operation("loading", source=str(source))

        try:
            # Handle different source types
            if isinstance(source, anndata.AnnData):
                adata = source.copy()
            elif isinstance(source, (str, Path)):
                adata = self._load_from_file(source, **kwargs)
            else:
                raise TypeError(f"Unsupported source type: {type(source)}")

            # Add basic metadata
            adata = self._add_basic_metadata(adata, source)

            # Add provenance information
            adata = self.add_provenance(
                adata,
                source_info={
                    "source": str(source),
                    "source_type": type(source).__name__,
                },
                processing_params=kwargs,
            )

            self.logger.info(
                f"Loaded protein structure: {adata.n_obs} atoms × {adata.n_vars} coords"
            )
            return adata

        except Exception as e:
            self.logger.error(f"Failed to load protein structure from {source}: {e}")
            raise

    def _load_from_file(self, path: Union[str, Path], **kwargs) -> anndata.AnnData:
        """Load protein structure from file with format detection."""
        try:
            from Bio.PDB import MMCIFParser, PDBParser
        except ImportError as e:
            raise ImportError(
                "BioPython is required for protein structure loading. "
                "Install with: pip install biopython"
            ) from e

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        format_type = self.detect_format(path)

        # Extract PDB ID from kwargs or filename
        pdb_id = kwargs.get("pdb_id")
        if pdb_id is None:
            # Try to extract from filename (e.g., "6FQF.pdb" -> "6FQF")
            pdb_id = path.stem[:4] if len(path.stem) >= 4 else path.stem

        # Override instance settings with kwargs if provided
        include_hetero = kwargs.get("include_hetero", self.include_hetero)
        include_waters = kwargs.get("include_waters", self.include_waters)
        model_number = kwargs.get("model_number", self.model_number)

        if format_type == "pdb":
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_id, str(path))
        elif format_type in ["cif", "mmcif"]:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure(pdb_id, str(path))
        elif format_type == "h5ad":
            # Already in AnnData format
            return self._load_h5ad_data(path)
        else:
            raise ValueError(
                f"Unsupported file format: {format_type}. "
                "Supported formats: .pdb, .cif, .mmcif, .h5ad"
            )

        # Convert Bio.PDB.Structure to AnnData
        adata = self._structure_to_anndata(
            structure,
            pdb_id=pdb_id,
            model_number=model_number,
            include_hetero=include_hetero,
            include_waters=include_waters,
        )

        # Extract and store structure metadata
        adata.uns["structure_metadata"] = self._extract_structure_metadata(
            structure, path
        )

        return adata

    def _structure_to_anndata(
        self,
        structure,
        pdb_id: str,
        model_number: int = 0,
        include_hetero: bool = True,
        include_waters: bool = False,
    ) -> anndata.AnnData:
        """
        Convert Bio.PDB.Structure object to AnnData.

        Args:
            structure: Bio.PDB.Structure object
            pdb_id: PDB identifier
            model_number: Model to use (for NMR structures)
            include_hetero: Whether to include HETATM records
            include_waters: Whether to include water molecules

        Returns:
            anndata.AnnData: Structure as AnnData with atoms as observations
        """
        # Get the specified model (or first model if not specified)
        try:
            model = structure[model_number]
        except KeyError:
            # If specified model doesn't exist, use first available
            model = list(structure.get_models())[0]
            self.logger.warning(
                f"Model {model_number} not found, using model {model.id}"
            )

        # Collect atom data
        atom_data = []
        coordinates = []

        for chain in model:
            chain_id = chain.id
            for residue in chain:
                # Filter residues based on settings
                hetero_flag = residue.id[0]

                # Skip waters unless explicitly included
                if hetero_flag == "W" and not include_waters:
                    continue

                # Skip other hetero atoms unless explicitly included
                if hetero_flag != " " and hetero_flag != "W" and not include_hetero:
                    continue

                residue_name = residue.resname
                residue_number = residue.id[1]

                for atom in residue:
                    atom_data.append(
                        {
                            "atom_name": atom.name,
                            "residue_name": residue_name,
                            "chain_id": chain_id,
                            "residue_number": residue_number,
                            "element": atom.element,
                            "b_factor": atom.bfactor,
                            "occupancy": atom.occupancy,
                            "alt_loc": atom.altloc,
                            "is_hetero": hetero_flag != " ",
                            "model_number": model.id,
                        }
                    )
                    coordinates.append(atom.coord)

        if len(atom_data) == 0:
            raise ValueError(
                "No atoms found in structure. "
                "Try adjusting include_hetero or include_waters settings."
            )

        # Create DataFrames
        obs = pd.DataFrame(atom_data)
        obs.index = [f"atom_{i}" for i in range(len(obs))]

        # Coordinates as X matrix (n_atoms x 3)
        X = np.array(coordinates, dtype=np.float32)

        # Variable names (coordinate axes)
        var = pd.DataFrame({"coordinate_axis": ["x", "y", "z"]}, index=["x", "y", "z"])

        # Create AnnData
        adata = anndata.AnnData(X=X, obs=obs, var=var)

        # Store 3D coordinates in obsm (standard for spatial data)
        adata.obsm["spatial"] = X.copy()

        # Store PDB ID in uns
        adata.uns["pdb_id"] = pdb_id

        # Add chain information summary
        chain_info = obs.groupby("chain_id").size().to_dict()
        adata.uns["chains"] = {
            "chain_ids": list(chain_info.keys()),
            "atom_counts": chain_info,
        }

        return adata

    def _extract_structure_metadata(self, structure, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from Bio.PDB.Structure object.

        Args:
            structure: Bio.PDB.Structure object
            file_path: Path to structure file

        Returns:
            Dict[str, Any]: Structure metadata
        """
        metadata = {
            "source_file": str(file_path),
            "structure_id": structure.id,
            "n_models": len(list(structure.get_models())),
        }

        # Try to extract header information (available for PDB parser)
        if hasattr(structure, "header") and structure.header:
            header = structure.header
            metadata.update(
                {
                    "experiment_method": header.get("structure_method", "UNKNOWN"),
                    "resolution": header.get("resolution"),
                    "deposition_date": header.get("deposition_date"),
                    "release_date": header.get("release_date"),
                    "organism": header.get("source", {}).get("organism_scientific"),
                }
            )

            # Add crystallographic information if available
            if "resolution" in header:
                metadata["resolution"] = header["resolution"]
            if "r_work" in header:
                metadata["r_factor"] = header["r_work"]
            if "r_free" in header:
                metadata["r_free"] = header["r_free"]

        return metadata

    def validate(
        self, adata: anndata.AnnData, strict: bool = False
    ) -> ValidationResult:
        """
        Validate protein structure data against schema.

        Args:
            adata: AnnData object to validate
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult: Validation results with errors/warnings

        Raises:
            ValueError: If strict=True and validation fails
        """
        result = self.validator.validate(adata)

        if strict and not result.is_valid():
            raise ValueError(f"Validation failed: {result.get_error_summary()}")

        return result

    def get_schema(self) -> Dict[str, Any]:
        """
        Return the protein structure schema.

        Returns:
            Dict[str, Any]: Schema definition
        """
        return ProteinStructureSchema.get_protein_structure_schema()

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.

        Returns:
            List[str]: Supported file extensions
        """
        return ["pdb", "cif", "mmcif", "h5ad"]

    def get_modality_name(self) -> str:
        """
        Get the modality name.

        Returns:
            str: Modality name
        """
        return "protein_structure"

    def detect_format(self, path: Union[str, Path]) -> str:
        """
        Detect file format from extension.

        Args:
            path: File path

        Returns:
            str: Detected format (pdb, cif, mmcif, h5ad)
        """
        path = Path(path)
        suffix = path.suffix.lower()

        format_map = {
            ".pdb": "pdb",
            ".ent": "pdb",  # Alternative PDB extension
            ".cif": "cif",
            ".mmcif": "mmcif",
            ".h5ad": "h5ad",
        }

        return format_map.get(suffix, suffix.lstrip("."))

    def preprocess_data(self, adata: anndata.AnnData, **kwargs) -> anndata.AnnData:
        """
        Apply basic preprocessing to structure data.

        Args:
            adata: Input AnnData
            **kwargs: Preprocessing parameters:
                - center: Whether to center coordinates at origin
                - calculate_distances: Whether to calculate pairwise distances

        Returns:
            anndata.AnnData: Preprocessed data
        """
        if kwargs.get("center", False):
            # Center coordinates at origin
            coords = adata.X.copy()
            center = coords.mean(axis=0)
            centered = coords - center
            adata.layers["centered_coordinates"] = centered
            self.logger.info("Centered coordinates at origin")

        return adata

    def add_provenance(
        self,
        adata: anndata.AnnData,
        source_info: Dict[str, Any],
        processing_params: Dict[str, Any],
    ) -> anndata.AnnData:
        """
        Add provenance tracking information.

        Args:
            adata: AnnData object
            source_info: Source information
            processing_params: Processing parameters used

        Returns:
            anndata.AnnData: Data with provenance information
        """
        if "provenance" not in adata.uns:
            adata.uns["provenance"] = {}

        adata.uns["provenance"]["adapter"] = {
            "name": self.name,
            "source": source_info,
            "parameters": processing_params,
            "adapter_version": "1.0.0",
        }

        return adata

    def _add_basic_metadata(
        self, adata: anndata.AnnData, source: Union[str, Path]
    ) -> anndata.AnnData:
        """
        Add basic metadata to AnnData object.

        Args:
            adata: AnnData object
            source: Data source

        Returns:
            anndata.AnnData: Data with basic metadata
        """
        # Add modality type
        adata.uns["modality"] = "protein_structure"

        # Add source info if not already present
        if "structure_metadata" not in adata.uns:
            adata.uns["structure_metadata"] = {}

        if isinstance(source, (str, Path)):
            adata.uns["structure_metadata"]["source_file"] = str(source)

        return adata

    def _log_operation(self, operation: str, **kwargs):
        """Log adapter operation with context."""
        context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"ProteinStructureAdapter: {operation} ({context})")
