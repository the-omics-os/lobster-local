"""
Protein Structure Visualization Expert Agent for 3D protein structure analysis.

This agent specializes in fetching protein structures from PDB, creating visualizations
with ChimeraX, performing structural analysis, and linking structures to omics data.
"""

from datetime import date
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.state import ProteinStructureVisualizationExpertState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.pymol_visualization_service import (
    PyMOLVisualizationService,
    PyMOLVisualizationError,
)
from lobster.tools.protein_structure_fetch_service import (
    ProteinStructureFetchService,
    ProteinStructureFetchError,
)
from lobster.tools.structure_analysis_service import (
    StructureAnalysisService,
    StructureAnalysisError,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ProteinStructureVisualizationError(Exception):
    """Base exception for protein structure visualization operations."""

    pass


def protein_structure_visualization_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "protein_structure_visualization_expert_agent",
    handoff_tools: List = None,
):
    """
    Create protein structure visualization expert agent using DataManagerV2.

    This agent specializes in:
    - Fetching protein structures from RCSB PDB
    - Creating high-quality 3D visualizations with ChimeraX
    - Performing structural analysis (RMSD, secondary structure, geometry)
    - Linking protein structures to expression data
    - Comparing multiple protein structures

    Args:
        data_manager: DataManagerV2 instance for provenance tracking
        callback_handler: Optional callback handler for streaming
        agent_name: Name of the agent
        handoff_tools: Optional list of handoff tools to other agents

    Returns:
        LangGraph agent for protein structure visualization
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params(agent_name)
    llm = create_llm(agent_name, model_params)

    if callback_handler and hasattr(llm, "with_config"):
        llm = llm.with_config(callbacks=[callback_handler])

    # Initialize services
    fetch_service = ProteinStructureFetchService()
    viz_service = PyMOLVisualizationService()
    analysis_service = StructureAnalysisService()

    # -------------------------
    # STRUCTURE FETCH TOOLS
    # -------------------------

    @tool
    def fetch_protein_structure(pdb_id: str, format: str = "cif") -> str:
        """
        Fetch protein structure from RCSB PDB database.

        Args:
            pdb_id: PDB identifier (4-character alphanumeric code, e.g., '1AKE')
            format: File format ('pdb' or 'cif', default: 'cif')

        Returns:
            str: Summary of fetched structure with metadata

        Examples:
            - fetch_protein_structure("1AKE")
            - fetch_protein_structure("4HHB", format="pdb")
        """
        try:
            logger.info(f"Fetching structure {pdb_id} (format: {format})")

            # Determine cache directory (workspace or current directory)
            if data_manager.workspace_path:
                cache_dir = Path(data_manager.workspace_path) / "protein_structures"
            else:
                cache_dir = Path.cwd() / "protein_structures"

            # Fetch structure
            structure_data, stats, ir = fetch_service.fetch_structure(
                pdb_id=pdb_id,
                format=format,
                cache_dir=cache_dir,
                extract_metadata=True,
                data_manager=data_manager,
            )

            # Store structure data in DataManager
            modality_name = f"structure_{pdb_id}"
            data_manager.modalities[modality_name] = structure_data

            # Log tool usage with IR
            data_manager.log_tool_usage(
                tool_name="fetch_protein_structure",
                parameters={"pdb_id": pdb_id, "format": format},
                description=f"Fetched protein structure {pdb_id} from RCSB PDB",
                ir=ir,
            )

            # Format response
            metadata = structure_data["metadata"]
            structure_info = structure_data["structure_info"]

            response = f"""✅ Successfully fetched protein structure: {pdb_id}

📊 **Structure Information:**
- PDB ID: {pdb_id}
- Title: {metadata['title']}
- Organism: {metadata['organism'] or 'Unknown'}
- Experiment Method: {metadata['experiment_method']}
- Resolution: {metadata['resolution']} Å""" if metadata['resolution'] else f"""- Experiment Method: {metadata['experiment_method']}"""

            if structure_info:
                response += f"""
- Chains: {len(structure_info.get('chains', []))}
- Total Residues: {structure_info.get('total_residues', 0):,}
- Total Atoms: {structure_info.get('total_atoms', 0):,}"""

            response += f"""

💾 **File Information:**
- File path: {structure_data['file_path']}
- Format: {format.upper()}
- File size: {stats['file_size_mb']:.2f} MB
- Cached: {'✓' if stats['cached'] else '✗ (newly downloaded)'}"""

            if metadata.get("publication_doi"):
                response += f"""

📚 **Publication:**
- DOI: {metadata['publication_doi']}
- Citation: {metadata.get('citation', 'N/A')}"""

            response += """

🎯 **Next Steps:**
You can now:
- Visualize the structure with `visualize_with_chimerax()`
- Analyze structural properties with `analyze_protein_structure()`
- Compare with other structures using `compare_structures()`
- Link to expression data with `link_to_expression_data()`"""

            return response

        except ProteinStructureFetchError as e:
            logger.error(f"Error fetching structure {pdb_id}: {e}")
            return f"❌ Error fetching structure {pdb_id}: {str(e)}"
        except Exception as e:
            logger.exception(f"Unexpected error fetching structure {pdb_id}: {e}")
            return f"❌ Unexpected error: {str(e)}"

    @tool
    def link_to_expression_data(
        modality_name: str,
        gene_column: str = "gene_symbol",
        organism: str = "Homo sapiens",
        max_structures_per_gene: int = 5,
    ) -> str:
        """
        Link gene expression data to protein structures from PDB.

        This tool searches PDB for structures corresponding to genes in your
        expression data and adds structure information to the modality.

        Args:
            modality_name: Name of modality containing gene/protein data
            gene_column: Column in adata.var with gene symbols (default: 'gene_symbol')
            organism: Source organism (default: 'Homo sapiens')
            max_structures_per_gene: Max structures to find per gene (default: 5)

        Returns:
            str: Summary of structure links created

        Examples:
            - link_to_expression_data("rna_seq_data")
            - link_to_expression_data("proteomics_data", gene_column="protein_name")
        """
        try:
            logger.info(f"Linking structures to genes in modality '{modality_name}'")

            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                available = data_manager.list_modalities()
                return f"❌ Modality '{modality_name}' not found.\n\nAvailable modalities: {', '.join(available)}"

            # Get modality
            adata = data_manager.get_modality(modality_name)

            # Link structures
            adata_linked, stats, ir = fetch_service.link_structures_to_genes(
                adata=adata,
                gene_column=gene_column,
                organism=organism,
                max_structures_per_gene=max_structures_per_gene,
                data_manager=data_manager,
            )

            # Store updated modality
            new_modality_name = f"{modality_name}_structure_linked"
            data_manager.modalities[new_modality_name] = adata_linked

            # Log tool usage with IR
            data_manager.log_tool_usage(
                tool_name="link_to_expression_data",
                parameters={
                    "modality_name": modality_name,
                    "gene_column": gene_column,
                    "organism": organism,
                    "max_structures_per_gene": max_structures_per_gene,
                },
                description=f"Linked {stats['genes_with_structures']} genes to protein structures",
                ir=ir,
            )

            # Format response
            response = f"""✅ Successfully linked gene expression data to protein structures!

📊 **Linking Results:**
- Genes searched: {stats['genes_searched']}
- Genes with structures: {stats['genes_with_structures']} ({stats['genes_with_structures_pct']:.1f}%)
- Total structures found: {stats['total_structures_found']}
- Avg structures per gene: {stats['avg_structures_per_gene']:.1f}
- Organism: {organism}

💾 **New Modality Created:**
- Name: '{new_modality_name}'
- Added columns:
  * 'pdb_structures': Comma-separated PDB IDs
  * 'has_structure': Boolean flag

📁 **Structure Mapping Details:**
Stored in modality.uns['protein_structure_links']

🎯 **Next Steps:**
You can now fetch and visualize structures for specific genes:
- Use `fetch_protein_structure()` with the PDB IDs
- Filter genes: adata[adata.var['has_structure']]"""

            return response

        except ProteinStructureFetchError as e:
            logger.error(f"Error linking structures: {e}")
            return f"❌ Error linking structures: {str(e)}"
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return f"❌ Unexpected error: {str(e)}"

    # -------------------------
    # VISUALIZATION TOOLS
    # -------------------------

    @tool
    def visualize_with_pymol(
        pdb_id: str,
        mode: str = "interactive",
        style: str = "cartoon",
        color_by: str = "chain",
        width: int = 1920,
        height: int = 1080,
        execute: bool = True,
        highlight_residues: str = None,
        highlight_color: str = "red",
        highlight_style: str = "sticks",
        highlight_groups: str = None,
    ) -> str:
        """
        Create high-quality 3D visualization of protein structure using PyMOL with optional residue highlighting.

        Args:
            pdb_id: PDB ID of structure to visualize (must be fetched first)
            mode: Execution mode - 'interactive' (launch GUI for exploration) or 'batch' (save PNG and exit) (default: 'interactive')
            style: Representation style - 'cartoon', 'surface', 'sticks', 'spheres', 'ribbon' (default: 'cartoon')
            color_by: Coloring scheme - 'chain', 'secondary_structure', 'bfactor', 'element' (default: 'chain')
            width: Image width in pixels (default: 1920)
            height: Image height in pixels (default: 1080)
            execute: Execute PyMOL commands if installed (default: True)
            highlight_residues: Residues to highlight (e.g., "15,42,89" or "A:15-20,B:42") (default: None)
            highlight_color: Color for highlighted residues (default: "red")
            highlight_style: Visualization style for highlights - 'sticks', 'spheres', 'surface', etc. (default: "sticks")
            highlight_groups: Multiple highlight groups in format "residues|color|style;residues2|color2|style2" (default: None)

        Returns:
            str: Summary of visualization with file paths

        Examples:
            - visualize_with_pymol("1AKE")  # Interactive mode by default
            - visualize_with_pymol("4HHB", mode="batch", style="surface", color_by="bfactor")
            - visualize_with_pymol("1AKE", mode="interactive")  # Launch GUI for exploration
            - visualize_with_pymol("1AKE", highlight_residues="15,42,89", highlight_color="red", highlight_style="sticks")  # Highlight disease mutations
            - visualize_with_pymol("4HHB", highlight_residues="A:15-20,B:30-35")  # Chain-specific highlighting
            - visualize_with_pymol("1AKE", highlight_groups="15,42|red|sticks;100-120|blue|surface")  # Multiple highlight groups
        """
        try:
            logger.info(
                f"Creating PyMOL visualization for {pdb_id} (style: {style})"
            )

            # Check if structure has been fetched
            modality_name = f"structure_{pdb_id}"
            if modality_name not in data_manager.modalities:
                return f"❌ Structure {pdb_id} not found. Please fetch it first with `fetch_protein_structure('{pdb_id}')`"

            structure_data = data_manager.modalities[modality_name]
            structure_file = Path(structure_data["file_path"])

            # Create visualization
            viz_data, stats, ir = viz_service.visualize_structure(
                structure_file=structure_file,
                mode=mode,
                style=style,
                color_by=color_by,
                width=width,
                height=height,
                execute_commands=execute,
                highlight_residues=highlight_residues,
                highlight_color=highlight_color,
                highlight_style=highlight_style,
                highlight_groups=highlight_groups,
            )

            # Store visualization data in DataManager
            viz_modality_name = f"viz_{pdb_id}_{style}_{color_by}"
            data_manager.modalities[viz_modality_name] = viz_data

            # Log tool usage with IR
            log_params = {
                "pdb_id": pdb_id,
                "mode": mode,
                "style": style,
                "color_by": color_by,
                "width": width,
                "height": height,
            }

            # Add highlight parameters if used
            if highlight_residues or highlight_groups:
                log_params.update({
                    "highlight_residues": highlight_residues,
                    "highlight_color": highlight_color,
                    "highlight_style": highlight_style,
                    "highlight_groups": highlight_groups,
                })

            # Build description
            desc_parts = [f"Created {mode} {style} visualization of {pdb_id} colored by {color_by}"]
            if highlight_residues or highlight_groups:
                desc_parts.append("with residue highlights")
            description = " ".join(desc_parts)

            data_manager.log_tool_usage(
                tool_name="visualize_with_pymol",
                parameters=log_params,
                description=description,
                ir=ir,
            )

            # Check PyMOL installation status
            pymol_status = viz_service.check_pymol_installation()

            # Format response based on mode
            if mode == "interactive":
                # Interactive mode response
                response = f"""✅ PyMOL visualization for {pdb_id}!

🎨 **Visualization Settings:**
- Mode: Interactive (GUI)
- Style: {style}
- Color scheme: {color_by}
- Window size: {width} × {height} pixels

📁 **Generated Files:**
- Command script: {viz_data['script_file']}"""

                if viz_data["executed"]:
                    response += f"""

🚀 **PyMOL GUI Status:**
- Status: ✓ {viz_data['execution_message']}"""

                    if "pid" in viz_data:
                        response += f"""
- Process ID: {viz_data['pid']}

💡 **Interactive Mode:**
The PyMOL graphical interface is now open! You can:
- Rotate the structure: left-click and drag
- Zoom: scroll wheel or middle-click drag
- Move: right-click and drag
- Try different representations using PyMOL menus (Display > Representation)
- Change colors using PyMOL menus (Color)
- Save images manually via File > Save Image
- Close the window when you're done"""
                else:
                    response += f"""

❌ **Execution Failed:**
- Status: {viz_data['execution_message']}"""

                    if not pymol_status["installed"]:
                        response += f"""

⚠️  **PyMOL Not Installed:**
{pymol_status['message']}

📥 **Installation Instructions:**
1. Install via Homebrew (macOS/Linux): brew install brewsci/bio/pymol
2. Or download from: https://pymol.org/
3. Add to PATH or use full path in commands

💡 **Manual Execution:**
Run the command script manually to launch GUI:
```bash
pymol {viz_data['script_file']}
```"""

            else:
                # Batch mode response
                response = f"""✅ PyMOL visualization created for {pdb_id}!

🎨 **Visualization Settings:**
- Mode: Batch (PNG image)
- Style: {style}
- Color scheme: {color_by}
- Image dimensions: {width} × {height} pixels

📁 **Generated Files:**
- Command script: {viz_data['script_file']}
- Output image: {viz_data['output_image']}"""

                if viz_data["executed"]:
                    response += f"""

✓ **Execution Status:**
- Status: {viz_data['execution_message']}"""

                    if Path(viz_data["output_image"]).exists():
                        response += f"""
- Image file size: {stats.get('output_file_size_mb', 0):.2f} MB"""
                else:
                    response += f"""

✗ **Execution Status:**
- Status: {viz_data['execution_message']}"""

                    if not pymol_status["installed"]:
                        response += f"""

⚠️  **PyMOL Not Installed:**
{pymol_status['message']}

📥 **Installation Instructions:**
1. Install via Homebrew (macOS/Linux): brew install brewsci/bio/pymol
2. Or download from: https://pymol.org/
3. Add to PATH or use full path in commands

💡 **Manual Execution:**
Run the command script manually:
```bash
pymol -c {viz_data['script_file']}
```"""

            response += f"""

🔧 **PyMOL Commands ({len(viz_data['commands'])} commands):**
The visualization script contains professional PyMOL commands
that can be customized and re-executed as needed.

🎯 **Next Steps:**
- Try different styles: cartoon, surface, sticks, spheres, ribbon
- Try different color schemes: chain, secondary_structure, bfactor, element"""

            if mode == "interactive":
                response += """
- Switch to batch mode with mode="batch" to generate PNG images"""
            else:
                response += """
- Switch to interactive mode with mode="interactive" to explore in 3D"""

            return response

        except PyMOLVisualizationError as e:
            logger.error(f"Error creating visualization: {e}")
            return f"❌ Error creating visualization: {str(e)}"
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return f"❌ Unexpected error: {str(e)}"

    # -------------------------
    # STRUCTURE ANALYSIS TOOLS
    # -------------------------

    @tool
    def analyze_protein_structure(
        pdb_id: str,
        analysis_type: str = "secondary_structure",
        chain_id: Optional[str] = None,
    ) -> str:
        """
        Analyze protein structure and extract structural properties.

        Args:
            pdb_id: PDB ID of structure to analyze (must be fetched first)
            analysis_type: Type of analysis - 'secondary_structure', 'geometry', 'residue_contacts' (default: 'secondary_structure')
            chain_id: Specific chain to analyze (None for all chains)

        Returns:
            str: Analysis results with structural properties

        Examples:
            - analyze_protein_structure("1AKE")
            - analyze_protein_structure("4HHB", analysis_type="geometry")
            - analyze_protein_structure("1AKE", analysis_type="residue_contacts", chain_id="A")
        """
        try:
            logger.info(f"Analyzing structure {pdb_id} ({analysis_type})")

            # Check if structure has been fetched
            modality_name = f"structure_{pdb_id}"
            if modality_name not in data_manager.modalities:
                return f"❌ Structure {pdb_id} not found. Please fetch it first with `fetch_protein_structure('{pdb_id}')`"

            structure_data = data_manager.modalities[modality_name]
            structure_file = Path(structure_data["file_path"])

            # Perform analysis
            analysis_results, stats, ir = analysis_service.analyze_structure(
                structure_file=structure_file,
                analysis_type=analysis_type,
                chain_id=chain_id,
            )

            # Store analysis results in DataManager
            analysis_modality_name = f"analysis_{pdb_id}_{analysis_type}_{chain_id or 'all'}"
            data_manager.modalities[analysis_modality_name] = analysis_results

            # Log tool usage with IR
            data_manager.log_tool_usage(
                tool_name="analyze_protein_structure",
                parameters={
                    "pdb_id": pdb_id,
                    "analysis_type": analysis_type,
                    "chain_id": chain_id,
                },
                description=f"Analyzed {pdb_id}: {analysis_type}",
                ir=ir,
            )

            # Format response based on analysis type
            response = f"""✅ Structural analysis complete for {pdb_id}!

📊 **Analysis Type:** {analysis_type}
🔬 **Chain:** {chain_id if chain_id else 'All chains'}

"""

            if analysis_type == "secondary_structure":
                if "secondary_structure_percentages" in analysis_results:
                    ss_pct = analysis_results["secondary_structure_percentages"]
                    response += f"""**Secondary Structure Distribution:**
- Helix (H): {ss_pct.get('H', 0):.1f}%
- Sheet (E): {ss_pct.get('E', 0):.1f}%
- Coil (-): {ss_pct.get('-', 0):.1f}%
- Other: {sum(v for k, v in ss_pct.items() if k not in ['H', 'E', '-']):.1f}%

Total Residues: {analysis_results.get('summary_stats', {}).get('total_residues', 0)}

Method: {analysis_results.get('method', 'DSSP')}"""
                else:
                    response += f"""**Secondary Structure Analysis:**
{analysis_results.get('message', 'Analysis completed')}

Total Residues: {analysis_results.get('total_residues', 0)}"""

            elif analysis_type == "geometry":
                response += f"""**Geometric Properties:**
- Total atoms: {stats.get('total_atoms', 0):,}
- Chains analyzed: {analysis_results.get('summary_stats', {}).get('n_chains', 0)}
- Radius of gyration: {analysis_results.get('overall_radius_of_gyration', 0):.2f} Å

**Per-Chain Properties:**"""
                for chain_info in analysis_results.get("chain_properties", [])[:5]:
                    response += f"""
- Chain {chain_info['chain_id']}: {chain_info['n_residues']} residues, {chain_info['n_atoms']} atoms, Rg={chain_info['radius_of_gyration']:.2f} Å"""

            elif analysis_type == "residue_contacts":
                response += f"""**Residue Contact Analysis:**
- Total residues: {analysis_results.get('summary_stats', {}).get('n_residues', 0)}
- Total contacts (≤{analysis_results.get('distance_cutoff', 8.0)} Å): {analysis_results.get('n_contacts', 0)}
- Average contacts per residue: {analysis_results.get('summary_stats', {}).get('avg_contacts_per_residue', 0):.2f}

Top contacts shown: {min(len(analysis_results.get('contacts', [])), 10)} (of {analysis_results.get('n_contacts', 0)} total)"""

            response += """

💾 **Analysis Results Stored:**
Results are stored in memory and can be accessed for further processing.

🎯 **Next Steps:**
- Compare with other structures using `compare_structures()`
- Visualize the structure with `visualize_with_chimerax()`
- Try different analysis types to explore more properties"""

            return response

        except StructureAnalysisError as e:
            logger.error(f"Error analyzing structure: {e}")
            return f"❌ Error analyzing structure: {str(e)}"
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return f"❌ Unexpected error: {str(e)}"

    @tool
    def compare_structures(
        pdb_id1: str,
        pdb_id2: str,
        align: bool = True,
        chain_id1: Optional[str] = None,
        chain_id2: Optional[str] = None,
    ) -> str:
        """
        Compare two protein structures by calculating RMSD (Root Mean Square Deviation).

        Args:
            pdb_id1: First PDB ID (must be fetched)
            pdb_id2: Second PDB ID (must be fetched)
            align: Align structures before RMSD calculation (default: True)
            chain_id1: Specific chain in first structure (None for first chain)
            chain_id2: Specific chain in second structure (None for first chain)

        Returns:
            str: RMSD and structural comparison results

        Examples:
            - compare_structures("1AKE", "4AKE")
            - compare_structures("1AKE", "4AKE", align=False)
            - compare_structures("4HHB", "2HHB", chain_id1="A", chain_id2="A")
        """
        try:
            logger.info(f"Comparing structures {pdb_id1} and {pdb_id2}")

            # Check if both structures have been fetched
            modality_name1 = f"structure_{pdb_id1}"
            modality_name2 = f"structure_{pdb_id2}"
            if modality_name1 not in data_manager.modalities:
                return f"❌ Structure {pdb_id1} not found. Please fetch it first."
            if modality_name2 not in data_manager.modalities:
                return f"❌ Structure {pdb_id2} not found. Please fetch it first."

            structure_file1 = Path(
                data_manager.modalities[modality_name1]["file_path"]
            )
            structure_file2 = Path(
                data_manager.modalities[modality_name2]["file_path"]
            )

            # Calculate RMSD
            rmsd_results, stats, ir = analysis_service.calculate_rmsd(
                structure_file1=structure_file1,
                structure_file2=structure_file2,
                chain_id1=chain_id1,
                chain_id2=chain_id2,
                align=align,
            )

            # Store comparison results in DataManager
            comparison_modality_name = f"comparison_{pdb_id1}_vs_{pdb_id2}"
            data_manager.modalities[comparison_modality_name] = rmsd_results

            # Log tool usage with IR
            data_manager.log_tool_usage(
                tool_name="compare_structures",
                parameters={
                    "pdb_id1": pdb_id1,
                    "pdb_id2": pdb_id2,
                    "align": align,
                    "chain_id1": chain_id1,
                    "chain_id2": chain_id2,
                },
                description=f"Compared {pdb_id1} and {pdb_id2}: RMSD = {rmsd_results['rmsd']:.3f} Å",
                ir=ir,
            )

            # Format response
            rmsd_value = rmsd_results["rmsd"]

            # Interpret RMSD value
            if rmsd_value < 1.0:
                similarity = "Nearly identical structures"
            elif rmsd_value < 2.0:
                similarity = "Very similar structures"
            elif rmsd_value < 3.0:
                similarity = "Similar structures with minor differences"
            elif rmsd_value < 5.0:
                similarity = "Moderately similar structures"
            else:
                similarity = "Different structures (significant conformational changes)"

            response = f"""✅ Structural comparison complete!

📊 **RMSD Calculation:**
- Structure 1: {pdb_id1} (Chain {rmsd_results['chain1']})
- Structure 2: {pdb_id2} (Chain {rmsd_results['chain2']})
- RMSD: {rmsd_value:.3f} Å
- Aligned atoms: {rmsd_results['n_atoms_used']}
- Alignment method: {'Superposition' if align else 'Direct comparison'}
- Similarity: {similarity}

💡 **RMSD Interpretation:**
- < 1.0 Å: Nearly identical (e.g., same protein, different conditions)
- 1-2 Å: Very similar (e.g., close homologs, small conformational changes)
- 2-3 Å: Similar (e.g., homologs, moderate conformational changes)
- 3-5 Å: Moderately similar (e.g., distant homologs, domain movements)
- > 5 Å: Different (e.g., different folds, large conformational changes)"""

            if align and "rotation_matrix" in rmsd_results:
                response += """

🔧 **Alignment Details:**
- Superposition successfully applied
- Rotation and translation matrices computed
- Structures aligned for optimal RMSD"""

            response += """

🎯 **Next Steps:**
- Visualize both structures to see differences
- Analyze specific regions with different RMSD values
- Compare with additional structures for evolutionary analysis"""

            return response

        except StructureAnalysisError as e:
            logger.error(f"Error comparing structures: {e}")
            return f"❌ Error comparing structures: {str(e)}"
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return f"❌ Unexpected error: {str(e)}"

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        fetch_protein_structure,
        link_to_expression_data,
        visualize_with_pymol,
        analyze_protein_structure,
        compare_structures,
    ]

    tools = base_tools + (handoff_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are an expert in protein structure visualization and analysis, specializing in 3D structural biology and integration with multi-omics data.

<Role>
You fetch protein structures from RCSB PDB, create high-quality visualizations using PyMOL, perform structural analysis (RMSD, secondary structure, geometry), and link structures to gene expression and proteomics data. You work with the professional DataManagerV2 system with full provenance tracking.

**CRITICAL: You ONLY perform protein structure visualization tasks specifically requested by the supervisor. You report results back to the supervisor, never directly to users.**
</Role>

<Communication Flow>
**USER → SUPERVISOR → YOU → SUPERVISOR → USER**
- You receive structure visualization tasks from the supervisor
- You execute the requested structure analysis and visualization
- You report results back to the supervisor
- The supervisor communicates with the user
</Communication Flow>

<Task>
You perform protein structure tasks following best practices:
1. **Structure fetching** - Download structures from RCSB PDB with comprehensive metadata
2. **3D visualization** - Create publication-quality visualizations using PyMOL
3. **Structural analysis** - Calculate RMSD, analyze secondary structure, compute geometric properties
4. **Data integration** - Link protein structures to gene expression and proteomics datasets
5. **Structure comparison** - Compare multiple structures and calculate structural similarity
6. **Quality assurance** - Validate structure files and provide clear guidance on PyMOL installation
</Task>

<Available Tools>

## Structure Fetching:
- `fetch_protein_structure`: Download structure from PDB with metadata (organism, method, resolution, chains)
- `link_to_expression_data`: Link gene expression data to PDB structures by searching for genes

## Visualization:
- `visualize_with_pymol`: Create high-quality 3D visualizations with customizable styles, colors, and residue highlighting (e.g., disease mutations, binding sites, active sites)

## Analysis:
- `analyze_protein_structure`: Analyze secondary structure, geometry, or residue contacts
- `compare_structures`: Calculate RMSD between two structures (with optional alignment)

</Available Tools>

<Professional Workflows & Tool Usage Order>

## 1. BASIC STRUCTURE VISUALIZATION (Supervisor Request: "Visualize protein structure")

# Step 1: Fetch structure from PDB
fetch_protein_structure("1AKE")

# Step 2: Create visualization with PyMOL
visualize_with_pymol("1AKE", style="cartoon", color_by="chain")

# Step 3: Report visualization results to supervisor
# WAIT for supervisor instruction before proceeding


## 2. STRUCTURE ANALYSIS WORKFLOW (Supervisor Request: "Analyze protein structure")

# Step 1: Fetch structure if not already available
fetch_protein_structure("4HHB")

# Step 2: Perform structural analysis
analyze_protein_structure("4HHB", analysis_type="secondary_structure")

# Step 3: Optional - Analyze geometry
analyze_protein_structure("4HHB", analysis_type="geometry")

# Step 4: Report analysis results to supervisor


## 3. STRUCTURE COMPARISON (Supervisor Request: "Compare two structures")

# Step 1: Fetch both structures
fetch_protein_structure("1AKE")
fetch_protein_structure("4AKE")

# Step 2: Calculate RMSD
compare_structures("1AKE", "4AKE", align=True)

# Step 3: Optional - Visualize both for side-by-side comparison
visualize_with_pymol("1AKE", style="cartoon", color_by="chain")
visualize_with_pymol("4AKE", style="cartoon", color_by="chain")

# Step 4: Report comparison results to supervisor


## 4. LINK STRUCTURES TO OMICS DATA (Supervisor Request: "Link structures to gene expression")

# Step 1: Link structures to genes in modality
link_to_expression_data("rna_seq_normalized", gene_column="gene_symbol", organism="Homo sapiens")

# Step 2: Report linking results to supervisor
# New modality created with structure mappings in .var["pdb_structures"]

# Step 3: Optional - Fetch and visualize top expressed genes with structures
# This would be a follow-up task based on supervisor request


## 5. COMPREHENSIVE STRUCTURE WORKFLOW (Supervisor Request: "Complete structure analysis")

# Step 1: Fetch structure
fetch_protein_structure("1AKE")

# Step 2: Analyze structural properties
analyze_protein_structure("1AKE", analysis_type="secondary_structure")
analyze_protein_structure("1AKE", analysis_type="geometry")

# Step 3: Create multiple visualizations
visualize_with_pymol("1AKE", style="cartoon", color_by="secondary_structure")
visualize_with_pymol("1AKE", style="surface", color_by="bfactor")

# Step 4: Report comprehensive results to supervisor


## 6. MULTI-STRUCTURE COMPARISON (Supervisor Request: "Compare protein family members")

# Step 1: Fetch all family members
fetch_protein_structure("1AKE")
fetch_protein_structure("2AKE")
fetch_protein_structure("4AKE")

# Step 2: Pairwise comparisons
compare_structures("1AKE", "2AKE", align=True)
compare_structures("1AKE", "4AKE", align=True)
compare_structures("2AKE", "4AKE", align=True)

# Step 3: Report comparison matrix to supervisor


## 7. RESIDUE HIGHLIGHTING (Supervisor Request: "Highlight disease mutations" or "Show binding site")

# Step 1: Fetch structure
fetch_protein_structure("1AKE")

# Step 2: Single residue group highlighting (e.g., disease mutation sites)
visualize_with_pymol("1AKE", highlight_residues="15,42,89", highlight_color="red", highlight_style="sticks")

# Step 3: Chain-specific highlighting (e.g., interface residues)
visualize_with_pymol("4HHB", highlight_residues="A:15-20,B:30-35", highlight_color="yellow", highlight_style="sticks")

# Step 4: Multiple highlight groups (e.g., binding site + catalytic residues)
visualize_with_pymol("1AKE", highlight_groups="15,42|red|sticks;100-120|blue|surface;200,215,230|green|spheres")

# Step 5: Report visualization with highlighted features to supervisor

</Professional Workflows & Tool Usage Order>

<Parameter Guidelines>

**PDB IDs:**
- Format: 4-character alphanumeric code (e.g., '1AKE', '4HHB')
- Always uppercase when displaying to users
- Validate format before API calls

**Visualization Styles:**
- cartoon: Standard representation for overall structure (default)
- surface: Molecular surface for binding site analysis
- sticks: Detailed view of side chains and ligands
- spheres: Space-filling model
- ribbon: Alternative backbone representation
- lines: Simple wireframe representation

**Color Schemes:**
- chain: Color by chain (multi-chain complexes)
- secondary_structure: Helix (red), sheet (yellow), loop (green)
- bfactor: Temperature factors (flexibility, resolution quality)
- element: Color by atomic element (C, N, O, S, etc.)

**Residue Highlighting:**
- highlight_residues: Single group specification (e.g., "15,42,89" or "A:15-20,B:42")
- highlight_color: Color for highlights (red, blue, green, yellow, orange, purple, etc.)
- highlight_style: Visualization style (sticks, spheres, surface, cartoon, ribbon, lines)
- highlight_groups: Multiple groups format: "residues|color|style;residues2|color2|style2"

**Highlight Formats:**
- Single residues: "15,42,89" (residues 15, 42, and 89 in any chain)
- Residue ranges: "15-20,42-50" (ranges 15-20 and 42-50)
- Chain-specific: "A:15,B:42" (residue 15 in chain A, residue 42 in chain B)
- Chain ranges: "A:15-20,B:30-35" (ranges in specific chains)
- Multiple groups: "15,42|red|sticks;100-120|blue|surface" (two different highlight groups)

**Common Use Cases:**
- Disease mutations: Highlight known pathogenic variants
- Binding sites: Show ligand or substrate binding pockets
- Active sites: Emphasize catalytic residues
- Interface residues: Highlight protein-protein interaction sites
- Post-translational modifications: Show phosphorylation or methylation sites

**Analysis Types:**
- secondary_structure: DSSP analysis (requires DSSP binary)
- geometry: Center of mass, radius of gyration, chain properties
- residue_contacts: Spatial proximity analysis (default cutoff: 8 Å)

**RMSD Interpretation:**
- < 1.0 Å: Nearly identical
- 1-2 Å: Very similar (close homologs)
- 2-3 Å: Similar (moderate changes)
- 3-5 Å: Moderately similar (domain movements)
- > 5 Å: Different folds/conformations

</Parameter Guidelines>

<Critical Operating Principles>
1. **ONLY perform structure visualization tasks explicitly requested by the supervisor**
2. **Always report results back to the supervisor, never directly to users**
3. **Fetch structures before visualization or analysis**
4. **Provide ChimeraX installation guidance when visualization fails**
5. **Cache structures to avoid redundant downloads**
6. **Validate PDB IDs before API calls**
7. **Wait for supervisor instruction between major workflow steps**
8. **Store all results with proper provenance tracking**
9. **Explain RMSD values in biological context**
10. **Suggest next steps for iterative analysis**
11. **NEVER HALLUCINATE OR LIE** - never make up tasks you haven't completed

<PyMOL Installation Guidance>
If PyMOL execution fails, provide these instructions:

**Installation:**
1. Homebrew (macOS/Linux): brew install brewsci/bio/pymol
2. Download from: https://pymol.org/
3. Add to PATH or note installation path

**Manual Execution:**
Generated command scripts (*.pml) can be run manually:
```bash
# Interactive mode (with GUI)
pymol structure_commands.pml

# Batch mode (headless)
pymol -c structure_commands.pml
```

**Command Scripts:**
All visualizations generate PyMOL command scripts that can be:
- Modified for custom visualizations
- Re-executed with different parameters
- Used as templates for batch processing

</PyMOL Installation Guidance>

<Quality Assurance & Best Practices>
- All tools include professional error handling with structure-specific exception types
- Comprehensive logging tracks all structure operations with parameters
- Automatic validation ensures PDB ID format and file integrity
- Provenance tracking maintains complete analysis history with IR
- Professional reporting with structural metadata and file paths
- Integration with existing omics workflows (gene expression, proteomics)
- DSSP integration for secondary structure (with graceful fallback)
- BioPython for robust structure parsing and analysis

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=ProteinStructureVisualizationExpertState,
    )
