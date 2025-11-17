"""
Structure Analysis Service for analyzing protein structure properties.

This stateless service provides methods for calculating structural metrics,
comparing structures, and annotating structural features using BioPython,
following the Lobster 3-tuple pattern (Dict, stats, IR).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from Bio import PDB
from Bio.PDB import DSSP, Superimposer

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class StructureAnalysisError(Exception):
    """Base exception for structure analysis operations."""

    pass


class StructureAnalysisService:
    """
    Stateless service for analyzing protein structures.

    This service implements the Lobster 3-tuple pattern:
    - Returns (Dict, stats_dict, AnalysisStep)
    - Uses BioPython for structure parsing
    - Calculates RMSD, secondary structure, geometric properties
    - Compares multiple structures
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the structure analysis service.

        Args:
            config: Optional configuration dict (for future use)
            **kwargs: Additional arguments (ignored, for backward compatibility)
        """
        logger.debug("Initializing stateless StructureAnalysisService")
        self.config = config or {}
        logger.debug("StructureAnalysisService initialized successfully")

    def analyze_structure(
        self,
        structure_file: Path,
        analysis_type: str = "secondary_structure",
        chain_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Analyze protein structure and extract structural properties.

        Args:
            structure_file: Path to PDB/CIF structure file
            analysis_type: Type of analysis ('secondary_structure', 'geometry', 'residue_contacts')
            chain_id: Specific chain to analyze (None for all chains)

        Returns:
            Tuple[Dict, Dict, AnalysisStep]: Analysis results, stats, and IR

        Raises:
            StructureAnalysisError: If analysis fails
        """
        try:
            logger.info(
                f"Analyzing protein structure: {structure_file} (type: {analysis_type})"
            )

            # Validate structure file
            structure_file = Path(structure_file)
            if not structure_file.exists():
                raise StructureAnalysisError(
                    f"Structure file not found: {structure_file}"
                )

            # Parse structure
            structure = self._parse_structure(structure_file)

            # Perform requested analysis
            if analysis_type == "secondary_structure":
                analysis_results = self._analyze_secondary_structure(
                    structure, structure_file, chain_id
                )
            elif analysis_type == "geometry":
                analysis_results = self._analyze_geometry(structure, chain_id)
            elif analysis_type == "residue_contacts":
                analysis_results = self._analyze_residue_contacts(structure, chain_id)
            else:
                raise StructureAnalysisError(
                    f"Unknown analysis type: {analysis_type}. "
                    f"Choose from: 'secondary_structure', 'geometry', 'residue_contacts'"
                )

            # Calculate statistics
            stats = {
                "structure_file": structure_file.name,
                "pdb_id": structure_file.stem[:4].upper(),
                "analysis_type": analysis_type,
                "chain_analyzed": chain_id if chain_id else "all_chains",
                **analysis_results.get("summary_stats", {}),
            }

            logger.info(f"Structure analysis complete: {stats}")

            # Create IR
            ir = self._create_analysis_ir(
                structure_file=structure_file,
                analysis_type=analysis_type,
                chain_id=chain_id,
            )

            return analysis_results, stats, ir

        except Exception as e:
            logger.exception(f"Error analyzing protein structure: {e}")
            raise StructureAnalysisError(f"Failed to analyze structure: {str(e)}")

    def calculate_rmsd(
        self,
        structure_file1: Path,
        structure_file2: Path,
        chain_id1: Optional[str] = None,
        chain_id2: Optional[str] = None,
        align: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Calculate RMSD between two protein structures.

        Args:
            structure_file1: Path to first structure file
            structure_file2: Path to second structure file
            chain_id1: Chain ID in first structure (None for first available)
            chain_id2: Chain ID in second structure (None for first available)
            align: Whether to align structures before RMSD calculation

        Returns:
            Tuple[Dict, Dict, AnalysisStep]: RMSD results, stats, and IR

        Raises:
            StructureAnalysisError: If RMSD calculation fails
        """
        try:
            logger.info(
                f"Calculating RMSD between {structure_file1.name} and {structure_file2.name}"
            )

            # Parse structures
            structure1 = self._parse_structure(structure_file1)
            structure2 = self._parse_structure(structure_file2)

            # Get chains
            chain1 = self._get_chain(structure1, chain_id1)
            chain2 = self._get_chain(structure2, chain_id2)

            # Extract CA atoms for alignment
            atoms1 = [atom for atom in chain1.get_atoms() if atom.get_id() == "CA"]
            atoms2 = [atom for atom in chain2.get_atoms() if atom.get_id() == "CA"]

            # Ensure same number of atoms
            min_atoms = min(len(atoms1), len(atoms2))
            atoms1 = atoms1[:min_atoms]
            atoms2 = atoms2[:min_atoms]

            if len(atoms1) == 0:
                raise StructureAnalysisError(
                    "No CA atoms found for alignment. Cannot calculate RMSD."
                )

            # Calculate RMSD
            if align:
                # Align structures using Superimposer
                super_imposer = Superimposer()
                super_imposer.set_atoms(atoms1, atoms2)
                rmsd = super_imposer.rms
                rotation = super_imposer.rotran[0]
                translation = super_imposer.rotran[1]

                alignment_info = {
                    "aligned": True,
                    "rotation_matrix": rotation.tolist(),
                    "translation_vector": translation.tolist(),
                }
            else:
                # Calculate RMSD without alignment
                coords1 = np.array([atom.get_coord() for atom in atoms1])
                coords2 = np.array([atom.get_coord() for atom in atoms2])
                diff = coords1 - coords2
                rmsd = np.sqrt(np.sum(diff * diff) / len(atoms1))

                alignment_info = {"aligned": False}

            # Prepare results
            rmsd_results = {
                "rmsd": float(rmsd),
                "n_atoms_used": len(atoms1),
                "structure1": str(structure_file1),
                "structure2": str(structure_file2),
                "chain1": chain_id1 if chain_id1 else chain1.get_id(),
                "chain2": chain_id2 if chain_id2 else chain2.get_id(),
                **alignment_info,
            }

            # Calculate statistics
            stats = {
                "rmsd_angstroms": float(rmsd),
                "n_aligned_atoms": len(atoms1),
                "structure1_name": structure_file1.name,
                "structure2_name": structure_file2.name,
                "aligned": align,
                "analysis_type": "rmsd_calculation",
            }

            logger.info(f"RMSD calculated: {rmsd:.3f} Å ({len(atoms1)} atoms)")

            # Create IR
            ir = self._create_rmsd_ir(
                structure_file1=structure_file1,
                structure_file2=structure_file2,
                align=align,
            )

            return rmsd_results, stats, ir

        except Exception as e:
            logger.exception(f"Error calculating RMSD: {e}")
            raise StructureAnalysisError(f"Failed to calculate RMSD: {str(e)}")

    def _parse_structure(self, structure_file: Path):
        """Parse structure file using BioPython."""
        try:
            # Determine parser based on file extension
            if structure_file.suffix.lower() in [".cif", ".mmcif"]:
                parser = PDB.MMCIFParser(QUIET=True)
            else:
                parser = PDB.PDBParser(QUIET=True)

            structure = parser.get_structure("protein", structure_file)
            return structure

        except Exception as e:
            raise StructureAnalysisError(
                f"Failed to parse structure file {structure_file}: {e}"
            )

    def _get_chain(self, structure, chain_id: Optional[str]):
        """Get specific chain from structure."""
        model = structure[0]  # Use first model

        if chain_id:
            if chain_id not in model:
                available_chains = [c.get_id() for c in model.get_chains()]
                raise StructureAnalysisError(
                    f"Chain '{chain_id}' not found. Available chains: {available_chains}"
                )
            return model[chain_id]
        else:
            # Return first available chain
            chains = list(model.get_chains())
            if not chains:
                raise StructureAnalysisError("No chains found in structure")
            return chains[0]

    def _analyze_secondary_structure(
        self, structure, structure_file: Path, chain_id: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze secondary structure using DSSP."""
        try:
            model = structure[0]

            # Run DSSP (requires DSSP binary installed)
            try:
                dssp = DSSP(model, str(structure_file), dssp="mkdssp")
                has_dssp = True
            except Exception as e:
                logger.warning(
                    f"DSSP not available ({e}). Using simplified analysis."
                )
                has_dssp = False
                dssp = None

            if has_dssp and dssp:
                # Extract DSSP results
                ss_counts = {"H": 0, "B": 0, "E": 0, "G": 0, "I": 0, "T": 0, "S": 0, "-": 0}
                residue_ss = []

                for key in dssp:
                    ss = dssp[key][2]
                    ss_counts[ss] = ss_counts.get(ss, 0) + 1
                    residue_ss.append(
                        {
                            "chain": key[0],
                            "residue": key[1][1],
                            "secondary_structure": ss,
                            "accessibility": dssp[key][3],
                        }
                    )

                total_residues = sum(ss_counts.values())
                ss_percentages = {
                    k: (v / total_residues) * 100 for k, v in ss_counts.items()
                }

                results = {
                    "method": "DSSP",
                    "secondary_structure_counts": ss_counts,
                    "secondary_structure_percentages": ss_percentages,
                    "residue_annotations": residue_ss,
                    "summary_stats": {
                        "total_residues": total_residues,
                        "helix_percent": ss_percentages.get("H", 0),
                        "sheet_percent": ss_percentages.get("E", 0),
                        "coil_percent": ss_percentages.get("-", 0),
                    },
                }

            else:
                # Simplified analysis without DSSP
                chains = [self._get_chain(structure, chain_id)] if chain_id else list(model.get_chains())

                total_residues = sum(len(list(chain.get_residues())) for chain in chains)

                results = {
                    "method": "simplified",
                    "message": "DSSP not available. Install with: conda install -c salilab dssp",
                    "total_residues": total_residues,
                    "summary_stats": {
                        "total_residues": total_residues,
                    },
                }

            return results

        except Exception as e:
            logger.warning(f"Secondary structure analysis failed: {e}")
            return {
                "method": "error",
                "error": str(e),
                "message": "Secondary structure analysis failed",
            }

    def _analyze_geometry(self, structure, chain_id: Optional[str]) -> Dict[str, Any]:
        """Analyze geometric properties of structure."""
        model = structure[0]
        chain = self._get_chain(structure, chain_id) if chain_id else None

        if chain:
            chains_to_analyze = [chain]
        else:
            chains_to_analyze = list(model.get_chains())

        # Calculate geometric properties
        all_coords = []
        chain_data = []

        for chain in chains_to_analyze:
            atoms = list(chain.get_atoms())
            residues = list(chain.get_residues())

            if atoms:
                coords = np.array([atom.get_coord() for atom in atoms])
                all_coords.extend(coords)

                # Calculate center of mass
                center = np.mean(coords, axis=0)

                # Calculate radius of gyration
                distances = np.linalg.norm(coords - center, axis=1)
                radius_of_gyration = np.sqrt(np.mean(distances ** 2))

                chain_data.append(
                    {
                        "chain_id": chain.get_id(),
                        "n_residues": len(residues),
                        "n_atoms": len(atoms),
                        "center_of_mass": center.tolist(),
                        "radius_of_gyration": float(radius_of_gyration),
                    }
                )

        # Calculate overall geometric properties
        all_coords = np.array(all_coords)
        overall_center = np.mean(all_coords, axis=0)
        overall_radius = np.sqrt(np.mean(np.linalg.norm(all_coords - overall_center, axis=1) ** 2))

        results = {
            "chain_properties": chain_data,
            "overall_center_of_mass": overall_center.tolist(),
            "overall_radius_of_gyration": float(overall_radius),
            "total_atoms": len(all_coords),
            "summary_stats": {
                "n_chains": len(chain_data),
                "total_atoms": len(all_coords),
                "radius_of_gyration": float(overall_radius),
            },
        }

        return results

    def _analyze_residue_contacts(
        self, structure, chain_id: Optional[str], distance_cutoff: float = 8.0
    ) -> Dict[str, Any]:
        """Analyze residue-residue contacts."""
        model = structure[0]
        chain = self._get_chain(structure, chain_id) if chain_id else None

        if chain:
            residues = list(chain.get_residues())
        else:
            residues = list(model.get_residues())

        # Find contacts
        contacts = []
        for i, res1 in enumerate(residues):
            for res2 in residues[i + 1 :]:
                # Calculate minimum distance between residues
                min_dist = float("inf")
                for atom1 in res1:
                    for atom2 in res2:
                        dist = atom1 - atom2  # BioPython calculates distance
                        if dist < min_dist:
                            min_dist = dist

                if min_dist <= distance_cutoff:
                    contacts.append(
                        {
                            "residue1": f"{res1.get_parent().get_id()}:{res1.get_id()[1]}",
                            "residue2": f"{res2.get_parent().get_id()}:{res2.get_id()[1]}",
                            "distance": float(min_dist),
                        }
                    )

        results = {
            "distance_cutoff": distance_cutoff,
            "n_contacts": len(contacts),
            "contacts": contacts[:100],  # Limit to first 100 for performance
            "summary_stats": {
                "n_residues": len(residues),
                "n_contacts": len(contacts),
                "avg_contacts_per_residue": len(contacts) / len(residues)
                if residues
                else 0,
            },
        }

        return results

    def _create_analysis_ir(
        self, structure_file: Path, analysis_type: str, chain_id: Optional[str]
    ) -> AnalysisStep:
        """Create IR for structure analysis operation."""
        parameter_schema = {
            "structure_file": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=str(structure_file),
                required=True,
                description="Path to protein structure file",
            ),
            "analysis_type": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=analysis_type,
                required=False,
                validation_rule="analysis_type in ['secondary_structure', 'geometry', 'residue_contacts']",
                description="Type of structural analysis",
            ),
        }

        code_template = """# Analyze protein structure properties
from Bio import PDB
import numpy as np

structure_file = "{{ structure_file }}"
analysis_type = "{{ analysis_type }}"

# Parse structure
parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure("protein", structure_file)
model = structure[0]

if analysis_type == "secondary_structure":
    # Use DSSP for secondary structure analysis
    try:
        from Bio.PDB import DSSP
        dssp = DSSP(model, structure_file, dssp="mkdssp")

        ss_counts = {}
        for key in dssp:
            ss = dssp[key][2]
            ss_counts[ss] = ss_counts.get(ss, 0) + 1

        print("Secondary structure distribution:")
        for ss, count in ss_counts.items():
            print(f"  {ss}: {count} residues")
    except Exception as e:
        print(f"DSSP not available: {e}")
        print("Install with: conda install -c salilab dssp")

elif analysis_type == "geometry":
    # Calculate geometric properties
    atoms = list(model.get_atoms())
    coords = np.array([atom.get_coord() for atom in atoms])

    center = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    radius_of_gyration = np.sqrt(np.mean(distances ** 2))

    print(f"Center of mass: {center}")
    print(f"Radius of gyration: {radius_of_gyration:.2f} Å")
    print(f"Total atoms: {len(atoms)}")

elif analysis_type == "residue_contacts":
    # Find residue-residue contacts
    residues = list(model.get_residues())
    distance_cutoff = 8.0
    contacts = 0

    for i, res1 in enumerate(residues):
        for res2 in residues[i+1:]:
            min_dist = float('inf')
            for atom1 in res1:
                for atom2 in res2:
                    dist = atom1 - atom2
                    if dist < min_dist:
                        min_dist = dist
            if min_dist <= distance_cutoff:
                contacts += 1

    print(f"Residue contacts (≤{distance_cutoff}Å): {contacts}")
    print(f"Total residues: {len(residues)}")
"""

        return AnalysisStep(
            operation="biopython.analyze_structure",
            tool_name="analyze_protein_structure",
            description=f"Analyze protein structure: {analysis_type}",
            library="biopython",
            code_template=code_template,
            imports=["from Bio import PDB", "import numpy as np"],
            parameters={
                "structure_file": str(structure_file),
                "analysis_type": analysis_type,
                "chain_id": chain_id,
            },
            parameter_schema=parameter_schema,
            input_entities=[str(structure_file)],
            output_entities=[],
            execution_context={
                "operation_type": "structural_analysis",
                "tool": "BioPython",
                "analysis_type": analysis_type,
            },
            validates_on_export=True,
            requires_validation=False,
        )

    def _create_rmsd_ir(
        self, structure_file1: Path, structure_file2: Path, align: bool
    ) -> AnalysisStep:
        """Create IR for RMSD calculation."""
        parameter_schema = {
            "structure_file1": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=str(structure_file1),
                required=True,
                description="Path to first structure file",
            ),
            "structure_file2": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=str(structure_file2),
                required=True,
                description="Path to second structure file",
            ),
            "align": ParameterSpec(
                param_type="bool",
                papermill_injectable=True,
                default_value=align,
                required=False,
                description="Whether to align structures before RMSD",
            ),
        }

        code_template = """# Calculate RMSD between two protein structures
from Bio import PDB
from Bio.PDB import Superimposer
import numpy as np

structure_file1 = "{{ structure_file1 }}"
structure_file2 = "{{ structure_file2 }}"
align = {{ align }}

# Parse structures
parser = PDB.PDBParser(QUIET=True)
structure1 = parser.get_structure("structure1", structure_file1)
structure2 = parser.get_structure("structure2", structure_file2)

# Get CA atoms
model1 = structure1[0]
model2 = structure2[0]
chain1 = list(model1.get_chains())[0]
chain2 = list(model2.get_chains())[0]

atoms1 = [atom for atom in chain1.get_atoms() if atom.get_id() == "CA"]
atoms2 = [atom for atom in chain2.get_atoms() if atom.get_id() == "CA"]

# Match length
min_atoms = min(len(atoms1), len(atoms2))
atoms1 = atoms1[:min_atoms]
atoms2 = atoms2[:min_atoms]

if align:
    # Calculate RMSD with alignment
    super_imposer = Superimposer()
    super_imposer.set_atoms(atoms1, atoms2)
    rmsd = super_imposer.rms
    print(f"RMSD (aligned): {rmsd:.3f} Å")
    print(f"Atoms used: {min_atoms}")
else:
    # Calculate RMSD without alignment
    coords1 = np.array([atom.get_coord() for atom in atoms1])
    coords2 = np.array([atom.get_coord() for atom in atoms2])
    diff = coords1 - coords2
    rmsd = np.sqrt(np.sum(diff * diff) / len(atoms1))
    print(f"RMSD (unaligned): {rmsd:.3f} Å")
    print(f"Atoms used: {min_atoms}")
"""

        return AnalysisStep(
            operation="biopython.calculate_rmsd",
            tool_name="compare_structures",
            description=f"Calculate RMSD between two protein structures (align={align})",
            library="biopython",
            code_template=code_template,
            imports=[
                "from Bio import PDB",
                "from Bio.PDB import Superimposer",
                "import numpy as np",
            ],
            parameters={
                "structure_file1": str(structure_file1),
                "structure_file2": str(structure_file2),
                "align": align,
            },
            parameter_schema=parameter_schema,
            input_entities=[str(structure_file1), str(structure_file2)],
            output_entities=[],
            execution_context={
                "operation_type": "structure_comparison",
                "tool": "BioPython",
                "metric": "RMSD",
            },
            validates_on_export=True,
            requires_validation=False,
        )
