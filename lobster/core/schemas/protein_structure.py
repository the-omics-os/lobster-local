"""
Protein structure schema definitions for 3D structural data from PDB and related sources.

This module defines the expected structure and metadata for protein structure
datasets including X-ray crystallography, NMR, cryo-EM data with appropriate
validation rules and atomic coordinate handling.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from lobster.core.interfaces.validator import ValidationResult
from .validation import FlexibleValidator


class ProteinStructureSchema:
    """
    Schema definitions for protein structure data modalities.

    This class provides schema definitions for 3D protein structures from
    PDB, AlphaFold, and other structural databases with appropriate metadata
    requirements and validation rules.
    """

    @staticmethod
    def get_protein_structure_schema() -> Dict[str, Any]:
        """
        Get schema for protein structure data (atoms as observations).

        Returns:
            Dict[str, Any]: Protein structure schema definition
        """
        return {
            "modality": "protein_structure",
            "description": "3D protein structure data from PDB or AlphaFold",
            "obs": {
                "required": ["atom_name", "residue_name", "chain_id", "residue_number"],
                "optional": [
                    "element",  # Chemical element symbol
                    "b_factor",  # Temperature factor (flexibility)
                    "occupancy",  # Occupancy value (0-1)
                    "alt_loc",  # Alternate location indicator
                    "secondary_structure",  # DSSP assignment (H, E, C, etc.)
                    "solvent_accessible",  # Solvent accessibility flag
                    "surface_area",  # Solvent accessible surface area
                    "residue_type",  # Amino acid type
                    "is_hetero",  # HETATM vs ATOM flag
                    "model_number",  # Model number (for NMR ensembles)
                ],
                "types": {
                    "atom_name": "string",
                    "residue_name": "string",
                    "chain_id": "string",
                    "residue_number": "int",
                    "element": "string",
                    "b_factor": "float",
                    "occupancy": "float",
                    "alt_loc": "string",
                    "secondary_structure": "categorical",
                    "solvent_accessible": "boolean",
                    "surface_area": "float",
                    "residue_type": "categorical",
                    "is_hetero": "boolean",
                    "model_number": "int",
                },
            },
            "var": {
                "required": ["coordinate_axis"],
                "optional": [],
                "types": {
                    "coordinate_axis": "string",  # x, y, z
                },
            },
            "layers": {
                "required": [],
                "optional": [
                    "raw_coordinates",  # Original atomic coordinates
                    "aligned_coordinates",  # Structurally aligned coordinates
                    "centered_coordinates",  # Centered at origin
                ],
            },
            "obsm": {
                "required": [],
                "optional": [
                    "spatial",  # 3D spatial coordinates (n_atoms x 3)
                    "pca_coords",  # PCA of conformational space
                ],
            },
            "uns": {
                "required": ["pdb_id", "structure_metadata"],
                "optional": [
                    "header_info",  # PDB header information
                    "resolution",  # Resolution in Angstroms
                    "experiment_method",  # X-ray, NMR, cryo-EM, etc.
                    "organism",  # Source organism
                    "expression_system",  # Expression system
                    "deposition_date",  # PDB deposition date
                    "release_date",  # PDB release date
                    "r_factor",  # R-factor (X-ray quality metric)
                    "r_free",  # R-free (validation metric)
                    "space_group",  # Crystallographic space group
                    "unit_cell",  # Unit cell parameters
                    "chains",  # List of chain information
                    "ligands",  # Bound ligands information
                    "secondary_structure_summary",  # DSSP summary statistics
                    "visualization_settings",  # ChimeraX visualization settings
                    "alignment_matrix",  # Transformation matrix for alignment
                    "provenance",  # Provenance tracking
                    # Cross-database accessions
                    "pdb_id",  # PDB identifier (1ABC, 6FQF, etc.)
                    "alphafold_id",  # AlphaFold DB identifier
                    "uniprot_id",  # UniProt accession
                    "publication_doi",  # Publication DOI
                ],
            },
        }

    @staticmethod
    def create_validator(
        strict: bool = False,
        ignore_warnings: Optional[List[str]] = None,
    ) -> FlexibleValidator:
        """
        Create a validator for protein structure data.

        Args:
            strict: Whether to use strict validation
            ignore_warnings: List of warning types to ignore

        Returns:
            FlexibleValidator: Configured validator
        """
        schema = ProteinStructureSchema.get_protein_structure_schema()

        ignore_set = set(ignore_warnings) if ignore_warnings else set()

        # Add default ignored warnings for protein structures
        ignore_set.update([
            "Unexpected obs columns",
            "Unexpected var columns",
        ])

        validator = FlexibleValidator(
            schema=schema,
            name="ProteinStructureValidator",
            ignore_warnings=ignore_set,
        )

        # Add protein structure-specific validation rules
        validator.add_custom_rule("check_atom_counts", _validate_atom_counts)
        validator.add_custom_rule("check_coordinates", _validate_coordinates)
        validator.add_custom_rule("check_pdb_id", _validate_pdb_id)
        validator.add_custom_rule("check_chain_consistency", _validate_chain_consistency)

        return validator

    @staticmethod
    def get_recommended_qc_thresholds() -> Dict[str, Any]:
        """
        Get recommended quality control thresholds for protein structures.

        Returns:
            Dict[str, Any]: QC thresholds and recommendations
        """
        return {
            "min_atoms": 100,  # Minimum atoms for valid structure
            "max_b_factor": 100.0,  # Maximum reasonable B-factor
            "min_occupancy": 0.0,  # Minimum occupancy
            "max_occupancy": 1.0,  # Maximum occupancy
            "resolution_high_quality": 2.0,  # <2.0 Å is high quality
            "resolution_acceptable": 3.5,  # <3.5 Å is acceptable
            "r_factor_good": 0.2,  # R-factor < 0.2 is good
            "r_free_good": 0.25,  # R-free < 0.25 is good
            "min_residues": 10,  # Minimum residues for peptide
        }


def _validate_atom_counts(adata) -> "ValidationResult":
    """Validate atom counts and structure size."""
    result = ValidationResult()

    # Check minimum atom count
    n_atoms = adata.n_obs
    if n_atoms < 100:
        result.add_warning(
            f"Very small structure: only {n_atoms} atoms (may be a fragment)"
        )
    elif n_atoms > 100000:
        result.add_warning(
            f"Very large structure: {n_atoms} atoms (may be slow to process)"
        )
    else:
        result.add_info(f"Structure contains {n_atoms} atoms")

    # Check for missing coordinates
    if "spatial" in adata.obsm:
        import numpy as np
        coords = adata.obsm["spatial"]
        if np.isnan(coords).any():
            nan_count = np.isnan(coords).sum()
            result.add_warning(f"Missing coordinates: {nan_count} NaN values found")

    return result


def _validate_coordinates(adata) -> "ValidationResult":
    """Validate 3D coordinates are reasonable."""
    import numpy as np

    result = ValidationResult()

    # Check X matrix dimensions (should be n_atoms x 3)
    if adata.n_vars != 3:
        result.add_error(
            f"Expected 3 coordinate dimensions (x, y, z), got {adata.n_vars}"
        )
        return result

    # Check coordinate ranges are reasonable (within typical protein size)
    coords = adata.X
    coord_range = np.ptp(coords, axis=0)  # Range for each dimension
    max_range = np.max(coord_range)

    if max_range > 1000:  # > 1000 Å is unusual
        result.add_warning(
            f"Unusually large coordinate range: {max_range:.1f} Å"
        )
    elif max_range < 5:  # < 5 Å is very small
        result.add_warning(
            f"Unusually small coordinate range: {max_range:.1f} Å"
        )
    else:
        result.add_info(f"Coordinate range: {max_range:.1f} Å")

    # Check for coordinate outliers (atoms very far from center)
    center = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    max_distance = np.max(distances)

    if max_distance > 500:  # > 500 Å from center
        result.add_warning(
            f"Atoms very far from center: max distance {max_distance:.1f} Å"
        )

    return result


def _validate_pdb_id(adata) -> "ValidationResult":
    """Validate PDB ID format."""
    result = ValidationResult()

    if "pdb_id" in adata.uns:
        pdb_id = adata.uns["pdb_id"]

        # PDB IDs should be 4 characters (alphanumeric)
        if not isinstance(pdb_id, str):
            result.add_error(f"PDB ID must be string, got {type(pdb_id)}")
        elif len(pdb_id) != 4:
            result.add_warning(
                f"PDB ID should be 4 characters, got '{pdb_id}' ({len(pdb_id)} chars)"
            )
        elif not pdb_id.isalnum():
            result.add_warning(
                f"PDB ID should be alphanumeric, got '{pdb_id}'"
            )
        else:
            result.add_info(f"Valid PDB ID: {pdb_id}")
    else:
        result.add_warning("No PDB ID found in uns['pdb_id']")

    return result


def _validate_chain_consistency(adata) -> "ValidationResult":
    """Validate chain assignments are consistent."""
    result = ValidationResult()

    if "chain_id" not in adata.obs.columns:
        result.add_warning("No chain_id column found")
        return result

    chains = adata.obs["chain_id"].unique()
    n_chains = len(chains)

    if n_chains == 0:
        result.add_error("No chains found in structure")
    elif n_chains == 1:
        result.add_info(f"Single chain structure: {chains[0]}")
    else:
        result.add_info(f"Multi-chain structure: {n_chains} chains ({', '.join(chains[:5])})")

        # Check chain sizes are reasonable
        chain_sizes = adata.obs["chain_id"].value_counts()
        if (chain_sizes < 10).any():
            small_chains = chain_sizes[chain_sizes < 10].index.tolist()
            result.add_warning(
                f"Very small chains detected: {', '.join(small_chains)} (<10 atoms)"
            )

    return result


# =============================================================================
# Pydantic Metadata Schemas for Structure Metadata Standardization
# =============================================================================


class ProteinStructureMetadataSchema(BaseModel):
    """
    Pydantic schema for protein structure metadata standardization.

    This schema defines the expected structure for protein structure metadata
    from PDB, AlphaFold, and other sources. It enforces controlled vocabularies
    and data types for consistent metadata representation.

    Used by metadata_assistant agent for:
    - Cross-database structure mapping
    - Metadata standardization and harmonization
    - Structure quality validation
    - Multi-omics integration with structural data

    Attributes:
        pdb_id: PDB identifier (required)
        experiment_method: Experimental method (X-ray, NMR, cryo-EM, etc.)
        resolution: Resolution in Angstroms (for X-ray and cryo-EM)
        organism: Source organism
        chains: List of chain identifiers
        ligands: List of bound ligands
        deposition_date: PDB deposition date
        additional_metadata: Flexible dict for custom fields
    """

    # Required fields
    pdb_id: str = Field(
        ...,
        description="PDB identifier (4 characters)",
        min_length=4,
        max_length=4
    )

    # Optional core fields
    experiment_method: Optional[str] = Field(
        None,
        description="Experimental method (X-RAY, NMR, CRYO-EM, PREDICTED)"
    )
    resolution: Optional[float] = Field(
        None,
        description="Resolution in Angstroms",
        gt=0.0,
        lt=100.0
    )
    organism: Optional[str] = Field(
        None,
        description="Source organism"
    )
    chains: Optional[List[str]] = Field(
        default_factory=list,
        description="List of chain identifiers"
    )
    ligands: Optional[List[str]] = Field(
        default_factory=list,
        description="List of bound ligands"
    )
    deposition_date: Optional[str] = Field(
        None,
        description="PDB deposition date (YYYY-MM-DD)"
    )

    # Flexible additional metadata
    additional_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional custom metadata fields"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "pdb_id": "6FQF",
                "experiment_method": "X-RAY",
                "resolution": 1.85,
                "organism": "Homo sapiens",
                "chains": ["A", "B"],
                "ligands": ["ATP", "MG"],
                "deposition_date": "2018-01-15",
                "additional_metadata": {
                    "r_factor": 0.178,
                    "r_free": 0.215,
                    "space_group": "P 21 21 21"
                }
            }
        }

    @field_validator("pdb_id")
    @classmethod
    def validate_pdb_id(cls, v: str) -> str:
        """Validate PDB ID format (4 alphanumeric characters)."""
        if not v or len(v) != 4:
            raise ValueError("PDB ID must be exactly 4 characters")
        if not v.isalnum():
            raise ValueError("PDB ID must be alphanumeric")
        return v.upper()

    @field_validator("experiment_method")
    @classmethod
    def validate_experiment_method(cls, v: Optional[str]) -> Optional[str]:
        """Validate experiment method is a known type."""
        if v is None:
            return v

        allowed = {
            "x-ray",
            "xray",
            "x_ray",
            "nmr",
            "cryo-em",
            "cryo_em",
            "cryoem",
            "electron microscopy",
            "predicted",
            "alphafold",
            "model",
            "solution nmr",
            "solid-state nmr",
            "fiber diffraction",
            "neutron diffraction",
            "electron crystallography",
        }

        v_lower = v.lower().replace("-", "_").replace(" ", "_")
        if v_lower not in allowed:
            # Allow unknown methods but normalize
            return v.upper()

        # Normalize to standard form
        if "xray" in v_lower or "x_ray" in v_lower:
            return "X-RAY"
        elif "nmr" in v_lower:
            return "NMR"
        elif "cryo" in v_lower or "electron microscopy" in v_lower:
            return "CRYO-EM"
        elif "predicted" in v_lower or "alphafold" in v_lower or "model" in v_lower:
            return "PREDICTED"
        else:
            return v.upper()

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: Optional[float]) -> Optional[float]:
        """Validate resolution is in reasonable range."""
        if v is None:
            return v
        if v <= 0:
            raise ValueError("Resolution must be positive")
        if v > 100:
            raise ValueError("Resolution must be less than 100 Angstroms")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary with all fields including additional_metadata
        """
        base_dict = self.model_dump(exclude={"additional_metadata"}, exclude_none=True)
        if self.additional_metadata:
            base_dict.update(self.additional_metadata)
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProteinStructureMetadataSchema":
        """
        Create schema from dictionary, automatically handling unknown fields.

        Args:
            data: Dictionary with metadata fields

        Returns:
            ProteinStructureMetadataSchema: Validated schema instance
        """
        # Extract known fields
        known_fields = set(cls.model_fields.keys()) - {"additional_metadata"}
        schema_data = {k: v for k, v in data.items() if k in known_fields}

        # Put remaining fields in additional_metadata
        additional = {k: v for k, v in data.items() if k not in known_fields}
        if additional:
            schema_data["additional_metadata"] = additional

        return cls(**schema_data)
