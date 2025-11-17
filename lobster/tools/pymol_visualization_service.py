"""
PyMOL Visualization Service for creating protein structure visualizations.

This stateless service provides methods for generating high-quality 3D protein
structure visualizations using PyMOL open-source, following the Lobster
3-tuple pattern (Dict, stats, IR).
"""

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class PyMOLVisualizationError(Exception):
    """Base exception for PyMOL visualization operations."""

    pass


class PyMOLVisualizationService:
    """
    Stateless service for creating protein structure visualizations with PyMOL.

    This service implements the Lobster 3-tuple pattern:
    - Returns (Dict, stats_dict, AnalysisStep)
    - Generates PyMOL command scripts (.pml files)
    - Creates high-quality structure images
    - Handles PyMOL installation checks
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the PyMOL visualization service.

        Args:
            config: Optional configuration dict (for future use)
            **kwargs: Additional arguments (ignored, for backward compatibility)
        """
        logger.debug("Initializing stateless PyMOLVisualizationService")
        self.config = config or {}
        self._pymol_available = None
        logger.debug("PyMOLVisualizationService initialized successfully")

    def visualize_structure(
        self,
        structure_file: Path,
        mode: str = "interactive",
        style: str = "cartoon",
        color_by: str = "chain",
        output_image: Optional[Path] = None,
        width: int = 1920,
        height: int = 1080,
        background: str = "white",
        execute_commands: bool = True,
        highlight_residues: Optional[str] = None,
        highlight_color: str = "red",
        highlight_style: str = "sticks",
        highlight_groups: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], AnalysisStep]:
        """
        Create 3D visualization of protein structure using PyMOL.

        Args:
            structure_file: Path to PDB/CIF structure file
            mode: Execution mode - 'interactive' (launch GUI) or 'batch' (save image and exit)
            style: Representation style ('cartoon', 'surface', 'sticks', 'spheres', 'ribbon')
            color_by: Coloring scheme ('chain', 'secondary_structure', 'bfactor', 'element')
            output_image: Path for output image (PNG), auto-generated if None
            width: Image width in pixels
            height: Image height in pixels
            background: Background color ('white', 'black', 'grey')
            execute_commands: Whether to execute PyMOL commands (requires installation)
            highlight_residues: Residues to highlight (e.g., "15,42,89" or "A:15-20,B:42")
            highlight_color: Color for highlighted residues (default: "red")
            highlight_style: Visualization style for highlights (default: "sticks")
            highlight_groups: Multiple highlight groups (format: "residues|color|style;...")

        Returns:
            Tuple[Dict, Dict, AnalysisStep]: Visualization metadata, stats, and IR

        Raises:
            PyMOLVisualizationError: If visualization fails
        """
        try:
            logger.info(
                f"Creating PyMOL visualization: {structure_file} (style: {style}, color: {color_by})"
            )

            # Validate structure file
            structure_file = Path(structure_file)
            if not structure_file.exists():
                raise PyMOLVisualizationError(
                    f"Structure file not found: {structure_file}"
                )

            # Set output image path
            if output_image is None:
                output_dir = structure_file.parent / "visualizations"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_image = (
                    output_dir / f"{structure_file.stem}_{style}_{color_by}.png"
                )
            output_image = Path(output_image)
            output_image.parent.mkdir(parents=True, exist_ok=True)

            # Parse highlight groups
            highlight_groups_parsed = self._parse_highlight_groups(
                highlight_residues=highlight_residues,
                highlight_color=highlight_color,
                highlight_style=highlight_style,
                highlight_groups=highlight_groups,
            )

            # Generate PyMOL command script
            commands = self._generate_pymol_commands(
                structure_file=structure_file,
                mode=mode,
                style=style,
                color_by=color_by,
                output_image=output_image,
                width=width,
                height=height,
                background=background,
                highlight_groups_parsed=highlight_groups_parsed,
            )

            # Save command script
            script_file = output_image.parent / f"{output_image.stem}_commands.pml"
            with open(script_file, "w") as f:
                f.write("\n".join(commands))

            logger.info(f"PyMOL command script saved: {script_file}")

            # Execute PyMOL commands if requested
            executed = False
            execution_message = "Script generated (execution skipped)"
            process_info = {}

            if execute_commands:
                pymol_installed = self.check_pymol_installation()
                if pymol_installed["installed"]:
                    try:
                        if mode == "interactive":
                            # Launch PyMOL GUI in interactive mode (non-blocking)
                            result = self._launch_pymol_interactive(
                                script_file, pymol_installed["path"]
                            )
                            executed = True
                            execution_message = result["message"]
                            process_info = {"pid": result["pid"], "mode": "interactive"}
                            logger.info(f"PyMOL GUI launched: {result['message']}")
                        else:
                            # Batch mode: execute and save image (blocking)
                            self._execute_pymol_batch(
                                script_file, pymol_installed["path"]
                            )
                            executed = True
                            execution_message = "Successfully executed with PyMOL (batch mode)"
                            logger.info(f"PyMOL visualization created: {output_image}")
                    except Exception as e:
                        logger.error(f"PyMOL execution failed: {e}")
                        execution_message = f"Execution failed: {e}"
                else:
                    execution_message = f"PyMOL not installed: {pymol_installed['message']}"
                    logger.warning(execution_message)

            # Prepare visualization data
            visualization_data = {
                "structure_file": str(structure_file),
                "output_image": str(output_image),
                "script_file": str(script_file),
                "mode": mode,
                "style": style,
                "color_by": color_by,
                "width": width,
                "height": height,
                "background": background,
                "commands": commands,
                "executed": executed,
                "execution_message": execution_message,
                **process_info,  # Add PID if interactive mode
            }

            # Calculate statistics
            stats = {
                "structure_file": structure_file.name,
                "output_image": output_image.name,
                "mode": mode,
                "style": style,
                "color_by": color_by,
                "image_dimensions": f"{width}x{height}",
                "executed": executed,
                "pymol_commands": len(commands),
                "analysis_type": "protein_structure_visualization",
            }

            # Check if image was created
            if output_image.exists():
                stats["output_file_size_mb"] = output_image.stat().st_size / (
                    1024 * 1024
                )

            logger.info(f"Visualization complete: {stats}")

            # Create IR for notebook export
            ir = self._create_visualization_ir(
                structure_file=structure_file,
                style=style,
                color_by=color_by,
                width=width,
                height=height,
                background=background,
                highlight_residues=highlight_residues,
                highlight_color=highlight_color,
                highlight_style=highlight_style,
                highlight_groups=highlight_groups,
            )

            return visualization_data, stats, ir

        except Exception as e:
            logger.exception(f"Error creating PyMOL visualization: {e}")
            raise PyMOLVisualizationError(
                f"Failed to create visualization: {str(e)}"
            )

    def check_pymol_installation(self) -> Dict[str, Any]:
        """
        Check if PyMOL is installed and accessible.

        Returns:
            Dict with installation status, path, and version information
        """
        if self._pymol_available is not None:
            return self._pymol_available

        # Check if pymol is in PATH (command-line)
        try:
            result = subprocess.run(
                ["pymol", "-c", "-Q"],  # -c = command-line mode, -Q = quit immediately
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Try to get version
                subprocess.run(
                    ["pymol", "-c", "-Q"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                version = "PyMOL (command-line)"
                self._pymol_available = {
                    "installed": True,
                    "path": "pymol",
                    "version": version,
                    "message": f"PyMOL found in PATH: {version}",
                }
                return self._pymol_available
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Common PyMOL installation paths
        potential_paths = [
            "/usr/local/bin/pymol",  # macOS/Linux (Homebrew)
            "/usr/bin/pymol",  # Linux (system)
            "/opt/homebrew/bin/pymol",  # macOS (ARM Homebrew)
            "C:\\Program Files\\PyMOL\\PyMOL\\pymol.exe",  # Windows
        ]

        # Check common installation paths
        for path in potential_paths:
            if Path(path).exists():
                try:
                    result = subprocess.run(
                        [path, "-c", "-Q"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        version = "PyMOL"
                        self._pymol_available = {
                            "installed": True,
                            "path": path,
                            "version": version,
                            "message": f"PyMOL found at {path}: {version}",
                        }
                        return self._pymol_available
                except Exception:
                    continue

        # Try Python module import (embedded PyMOL)
        try:
            import pymol

            version = getattr(pymol, "__version__", "Unknown version")
            self._pymol_available = {
                "installed": True,
                "path": "embedded",
                "version": version,
                "message": f"PyMOL Python module available: {version}",
            }
            return self._pymol_available
        except ImportError:
            pass

        # PyMOL not found
        self._pymol_available = {
            "installed": False,
            "path": None,
            "version": None,
            "message": "PyMOL not found. Install with: brew install brewsci/bio/pymol",
        }
        return self._pymol_available

    def _parse_highlight_groups(
        self,
        highlight_residues: Optional[str],
        highlight_color: str,
        highlight_style: str,
        highlight_groups: Optional[str],
    ) -> list:
        """
        Parse highlight parameters into standardized format for PyMOL commands.

        Args:
            highlight_residues: Single group residue specification
            highlight_color: Color for single group
            highlight_style: Style for single group
            highlight_groups: Multiple groups (format: "residues|color|style;...")

        Returns:
            List of dicts with keys: 'residues', 'color', 'style', 'label', 'pymol_selection'
        """
        groups = []

        if highlight_groups:
            # Parse multiple groups: "15,42|red|sticks;100-120|blue|surface"
            for i, group_spec in enumerate(highlight_groups.split(";")):
                parts = [p.strip() for p in group_spec.split("|")]
                if len(parts) != 3:
                    logger.warning(
                        f"Invalid highlight group format: {group_spec}. Expected 'residues|color|style'"
                    )
                    continue

                residues, color, style = parts
                pymol_selection = self._convert_to_pymol_selection(residues)

                groups.append({
                    "residues": residues,
                    "pymol_selection": pymol_selection,
                    "color": color,
                    "style": style,
                    "label": f"highlight_group_{i+1}",
                })

        elif highlight_residues:
            # Single group with specified color and style
            pymol_selection = self._convert_to_pymol_selection(highlight_residues)
            groups.append({
                "residues": highlight_residues,
                "pymol_selection": pymol_selection,
                "color": highlight_color,
                "style": highlight_style,
                "label": "highlight_residues",
            })

        return groups

    def _convert_to_pymol_selection(self, residue_spec: str) -> str:
        """
        Convert residue specification to PyMOL selection syntax.

        Supported formats:
        - "15,42,89" → "resi 15+42+89"
        - "15-20,42-50" → "resi 15-20+42-50"
        - "A:15,B:42" → "(chain A and resi 15) or (chain B and resi 42)"
        - "A:15-20,B:42-50" → "(chain A and resi 15-20) or (chain B and resi 42-50)"

        Args:
            residue_spec: Residue specification string

        Returns:
            PyMOL selection string
        """
        # Check if chain-specific (contains colons)
        if ":" in residue_spec:
            # Chain-specific format: A:15,B:42 or A:15-20,B:42-50
            selections = []
            for part in residue_spec.split(","):
                part = part.strip()
                if ":" not in part:
                    logger.warning(
                        f"Mixed chain-specific and non-chain-specific residues in: {residue_spec}"
                    )
                    continue

                chain, residues = part.split(":", 1)
                chain = chain.strip()
                residues = residues.strip()

                # Handle ranges and single residues
                if "-" in residues and not residues.replace("-", "").replace(" ", "").isdigit():
                    # Range format: 15-20
                    selections.append(f"(chain {chain} and resi {residues})")
                else:
                    # Single residue or list of residues
                    selections.append(f"(chain {chain} and resi {residues})")

            return " or ".join(selections)
        else:
            # Non-chain-specific format: 15,42,89 or 15-20,42-50
            # PyMOL uses '+' to combine selections and accepts ranges directly
            # "15,42,89" → "resi 15+42+89"
            # "15-20,42-50" → "resi 15-20+42-50"
            residues = residue_spec.replace(",", "+").replace(" ", "")
            return f"resi {residues}"

    def _generate_pymol_commands(
        self,
        structure_file: Path,
        mode: str,
        style: str,
        color_by: str,
        output_image: Path,
        width: int,
        height: int,
        background: str,
        highlight_groups_parsed: list = None,
    ) -> list:
        """Generate PyMOL command script for visualization."""
        commands = [
            "# PyMOL Protein Structure Visualization Script",
            "# Generated by Lobster - Protein Structure Visualization Service",
            "",
            "# Load structure file",
            f"load {structure_file}, protein1",
            "",
            "# Set background color",
            f"bg_color {background}",
            "",
            "# Hide all representations first",
            "hide everything",
            "",
            "# Apply representation style",
        ]

        # Style commands
        style_commands = {
            "cartoon": ["show cartoon, protein1"],
            "surface": ["show surface, protein1"],
            "sticks": ["show sticks, protein1"],
            "spheres": ["show spheres, protein1"],
            "ribbon": ["show ribbon, protein1"],
            "lines": ["show lines, protein1"],
        }

        commands.extend(style_commands.get(style, style_commands["cartoon"]))
        commands.append("")

        # Color commands
        commands.append("# Apply coloring scheme")
        if color_by == "chain":
            commands.append("util.cbc('protein1')")
        elif color_by == "secondary_structure":
            # Color by secondary structure elements
            commands.append("color red, ss h")  # Helices
            commands.append("color yellow, ss s")  # Sheets
            commands.append("color green, ss l+")  # Loops
        elif color_by == "bfactor":
            commands.append("spectrum b, rainbow, protein1")
        elif color_by == "element":
            commands.append("util.cnc('protein1')")  # Color by element
        else:
            commands.append("util.cbc('protein1')")  # Default to chain

        commands.extend(
            [
                "",
                "# Center and orient structure",
                "zoom all",
                "orient protein1",
                "",
            ]
        )

        # Add residue highlighting if specified
        if highlight_groups_parsed:
            commands.append("# Highlight specific residues (disease mutations, binding sites, etc.)")
            for group in highlight_groups_parsed:
                label = group['label']
                pymol_selection = group['pymol_selection']
                color = group['color']
                style_cmd = group['style']

                commands.extend([
                    f"select {label}, {pymol_selection}",
                    f"show {style_cmd}, {label}",
                    f"color {color}, {label}",
                ])
            commands.append("")

        # Mode-specific final commands
        if mode == "batch":
            commands.extend(
                [
                    "# Set viewport size",
                    f"viewport {width}, {height}",
                    "",
                    "# Save image and exit",
                    f"png {output_image}, width={width}, height={height}, dpi=300, ray=0",
                    "quit",
                ]
            )
        else:  # interactive mode
            commands.extend(
                [
                    "",
                    "# Interactive mode - PyMOL GUI will remain open",
                    "# You can now interact with the 3D structure",
                    "# Use mouse to rotate (left-click drag), zoom (scroll), and move (right-click drag)",
                    f"# To save an image: png {output_image.name}, width={width}, height={height}",
                ]
            )

        return commands

    def _execute_pymol_batch(self, script_file: Path, pymol_path: str):
        """Execute PyMOL command script in batch mode (headless, blocking)."""
        try:
            result = subprocess.run(
                [pymol_path, "-c", str(script_file)],  # -c = command-line (no GUI)
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise PyMOLVisualizationError(
                    f"PyMOL batch execution failed: {result.stderr}"
                )

            logger.debug(f"PyMOL batch output: {result.stdout}")

        except subprocess.TimeoutExpired:
            raise PyMOLVisualizationError("PyMOL batch execution timed out (60s)")
        except Exception as e:
            raise PyMOLVisualizationError(f"Failed to execute PyMOL in batch mode: {e}")

    def _launch_pymol_interactive(
        self, script_file: Path, pymol_path: str
    ) -> Dict[str, Any]:
        """
        Launch PyMOL GUI in interactive mode (non-blocking).

        Args:
            script_file: Path to PyMOL command script
            pymol_path: Path to PyMOL executable

        Returns:
            Dict with success status, PID, and message

        Raises:
            PyMOLVisualizationError: If launch fails
        """
        try:
            # Launch PyMOL with GUI (opens GUI window)
            # Use Popen for non-blocking execution
            process = subprocess.Popen(
                [pymol_path, str(script_file)],  # PyMOL runs script on launch
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from parent process
            )

            # Give PyMOL time to start
            time.sleep(2)

            # Check if process is still running
            poll_result = process.poll()
            if poll_result is not None:
                # Process terminated immediately - likely an error
                stderr_output = (
                    process.stderr.read().decode("utf-8")
                    if process.stderr
                    else "No error output"
                )
                raise PyMOLVisualizationError(
                    f"PyMOL failed to start (exit code {poll_result}): {stderr_output}"
                )

            logger.info(f"PyMOL GUI launched successfully (PID: {process.pid})")

            return {
                "success": True,
                "pid": process.pid,
                "message": f"PyMOL GUI launched successfully (PID {process.pid})",
            }

        except Exception as e:
            logger.error(f"Failed to launch PyMOL interactive mode: {e}")
            raise PyMOLVisualizationError(
                f"Failed to launch PyMOL GUI: {e}"
            )

    def _create_visualization_ir(
        self,
        structure_file: Path,
        style: str,
        color_by: str,
        width: int,
        height: int,
        background: str,
        highlight_residues: Optional[str] = None,
        highlight_color: str = "red",
        highlight_style: str = "sticks",
        highlight_groups: Optional[str] = None,
    ) -> AnalysisStep:
        """Create Intermediate Representation for visualization operation."""
        parameter_schema = {
            "structure_file": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=str(structure_file),
                required=True,
                description="Path to protein structure file",
            ),
            "style": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=style,
                required=False,
                validation_rule="style in ['cartoon', 'surface', 'sticks', 'spheres', 'ribbon', 'lines']",
                description="Representation style",
            ),
            "color_by": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=color_by,
                required=False,
                validation_rule="color_by in ['chain', 'secondary_structure', 'bfactor', 'element']",
                description="Coloring scheme",
            ),
            "width": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=width,
                required=False,
                description="Image width in pixels",
            ),
            "height": ParameterSpec(
                param_type="int",
                papermill_injectable=True,
                default_value=height,
                required=False,
                description="Image height in pixels",
            ),
            "highlight_residues": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=highlight_residues if highlight_residues else "None",
                required=False,
                description="Residues to highlight (e.g., '15,42,89' or 'A:15-20,B:42')",
            ),
            "highlight_color": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=highlight_color,
                required=False,
                description="Color for highlighted residues",
            ),
            "highlight_style": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=highlight_style,
                required=False,
                validation_rule="highlight_style in ['sticks', 'spheres', 'surface', 'cartoon', 'ribbon', 'lines']",
                description="Visualization style for highlighted residues",
            ),
            "highlight_groups": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=highlight_groups if highlight_groups else "None",
                required=False,
                description="Multiple highlight groups (format: 'residues|color|style;...')",
            ),
        }

        code_template = """# Create protein structure visualization with PyMOL
import subprocess
from pathlib import Path

structure_file = "{{ structure_file }}"
style = "{{ style }}"
color_by = "{{ color_by }}"
width = {{ width }}
height = {{ height }}
highlight_residues = "{{ highlight_residues }}"
highlight_color = "{{ highlight_color }}"
highlight_style = "{{ highlight_style }}"
highlight_groups = "{{ highlight_groups }}"

# Generate PyMOL commands
commands = [
    f"load {structure_file}, protein1",
    "bg_color white",
    "hide everything",
]

# Style command
if style == "cartoon":
    commands.append("show cartoon, protein1")
elif style == "surface":
    commands.append("show surface, protein1")
elif style in ["sticks", "spheres", "ribbon", "lines"]:
    commands.append(f"show {style}, protein1")

# Color commands
if color_by == "chain":
    commands.append("util.cbc('protein1')")
elif color_by == "secondary_structure":
    commands.extend([
        "color red, ss h",
        "color yellow, ss s",
        "color green, ss l+"
    ])
elif color_by == "bfactor":
    commands.append("spectrum b, rainbow, protein1")
elif color_by == "element":
    commands.append("util.cnc('protein1')")

# Orientation
commands.extend([
    "zoom all",
    "orient protein1",
])

# Residue highlighting
if highlight_groups and highlight_groups != "None":
    # Multiple highlight groups: "15,42|red|sticks;100-120|blue|surface"
    for i, group_spec in enumerate(highlight_groups.split(";")):
        parts = group_spec.split("|")
        if len(parts) == 3:
            residues, color, style_cmd = parts
            # Convert to PyMOL selection
            if ":" in residues:
                # Chain-specific format
                selections = []
                for part in residues.split(","):
                    if ":" in part:
                        chain, res = part.split(":", 1)
                        selections.append(f"(chain {chain} and resi {res})")
                pymol_selection = " or ".join(selections)
            else:
                # Simple format
                pymol_selection = f"resi {residues.replace(',', '+')}"

            label = f"highlight_group_{i+1}"
            commands.extend([
                f"select {label}, {pymol_selection}",
                f"show {style_cmd}, {label}",
                f"color {color}, {label}",
            ])
elif highlight_residues and highlight_residues != "None":
    # Single highlight group
    if ":" in highlight_residues:
        # Chain-specific format
        selections = []
        for part in highlight_residues.split(","):
            if ":" in part:
                chain, res = part.split(":", 1)
                selections.append(f"(chain {chain} and resi {res})")
        pymol_selection = " or ".join(selections)
    else:
        # Simple format
        pymol_selection = f"resi {highlight_residues.replace(',', '+')}"

    commands.extend([
        f"select highlight_residues, {pymol_selection}",
        f"show {highlight_style}, highlight_residues",
        f"color {highlight_color}, highlight_residues",
    ])

# Finalize
commands.extend([
    f"viewport {width}, {height}",
    f"png {Path(structure_file).stem}_visualization.png, width={width}, height={height}, ray=0",
    "quit"
])

# Save command script
script_file = Path(f"{Path(structure_file).stem}_pymol_commands.pml")
script_file.write_text("\\\\n".join(commands))

print(f"PyMOL command script created: {script_file}")
print("Run with: pymol -c {script_file}")

# Optional: Execute if PyMOL is installed
try:
    result = subprocess.run(
        ["pymol", "-c", str(script_file)],
        capture_output=True,
        timeout=60
    )
    if result.returncode == 0:
        print(f"Visualization created successfully")
    else:
        print(f"PyMOL execution failed: {result.stderr}")
except FileNotFoundError:
    print("PyMOL not found in PATH. Install with: brew install brewsci/bio/pymol")
"""

        # Build description with highlight info if applicable
        description_parts = [f"Create {style} visualization of protein structure colored by {color_by}"]
        if highlight_groups or highlight_residues:
            description_parts.append("with residue highlights")
        description = " ".join(description_parts)

        return AnalysisStep(
            operation="pymol.visualize_structure",
            tool_name="visualize_with_pymol",
            description=description,
            library="pymol",
            code_template=code_template,
            imports=["import subprocess", "from pathlib import Path"],
            parameters={
                "structure_file": str(structure_file),
                "style": style,
                "color_by": color_by,
                "width": width,
                "height": height,
                "background": background,
                "highlight_residues": highlight_residues if highlight_residues else None,
                "highlight_color": highlight_color,
                "highlight_style": highlight_style,
                "highlight_groups": highlight_groups if highlight_groups else None,
            },
            parameter_schema=parameter_schema,
            input_entities=[str(structure_file)],
            output_entities=[f"{structure_file.stem}_visualization.png"],
            execution_context={
                "operation_type": "visualization",
                "tool": "PyMOL",
                "visualization_type": "3d_structure",
                "has_highlights": bool(highlight_groups or highlight_residues),
            },
            validates_on_export=True,
            requires_validation=False,
        )
