"""
Jupyter Notebook Exporter for Pipeline Replay.

This module converts Lobster provenance records into executable Jupyter notebooks,
enabling reproducible analysis workflows using industry-standard tools.

The exporter uses Service-Emitted Intermediate Representation (IR) to automatically
generate code without manual mapping registries, achieving 95%+ executable notebooks.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import nbformat
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from lobster.core.analysis_ir import AnalysisStep, extract_unique_imports
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class NotebookExporter:
    """
    Convert Lobster provenance to executable Jupyter notebook using IR.

    This class transforms provenance tracking data with embedded Intermediate
    Representation (IR) into reproducible analysis pipelines using standard
    Jupyter notebook format compatible with Papermill for parameterized execution.

    The key innovation is that services emit AnalysisStep IR objects that contain
    complete code generation instructions, eliminating the need for manual mapping
    registries like TOOL_TO_CODE.

    Attributes:
        provenance: ProvenanceTracker instance with analysis history
        data_manager: DataManagerV2 instance for data context
    """

    def __init__(
        self, provenance: ProvenanceTracker, data_manager: DataManagerV2
    ) -> None:
        """
        Initialize notebook exporter.

        Args:
            provenance: ProvenanceTracker with recorded activities
            data_manager: DataManagerV2 for workspace context
        """
        self.provenance = provenance
        self.data_manager = data_manager
        logger.debug("Initialized NotebookExporter with IR support")

    def export(
        self,
        name: str,
        description: str = "",
        filter_strategy: str = "successful",
        validate_syntax: bool = True,
    ) -> Path:
        """
        Generate Jupyter notebook from current session using IR.

        Args:
            name: Notebook filename (without extension)
            description: Human-readable description for header
            filter_strategy: Activity filter ("successful" | "all" | "manual")
            validate_syntax: Whether to validate generated Python syntax

        Returns:
            Path to generated .ipynb file

        Raises:
            ValueError: If no activities to export or name is invalid
            RuntimeError: If IR extraction fails or code generation invalid
        """
        if not name or not name.strip():
            raise ValueError("Notebook name cannot be empty")

        if not self.provenance.activities:
            raise ValueError("No activities recorded - nothing to export")

        logger.info(f"Exporting notebook with IR: {name}")

        # Filter activities
        activities = self._filter_activities(filter_strategy)
        logger.debug(f"Filtered {len(activities)} activities for export")

        # Extract IRs from activities
        irs = self._extract_irs(activities)
        logger.info(
            f"Extracted {len(irs)} IR objects from {len(activities)} activities"
        )

        if len(irs) == 0:
            logger.warning(
                "No IR objects found in provenance. "
                "Services need to emit AnalysisStep objects. "
                "Notebook will contain placeholder code."
            )

        # Create new notebook
        notebook = new_notebook()

        # Add header
        notebook.cells.append(self._create_header_cell(name, description, len(irs)))

        # Collect and add imports
        imports_cell = self._create_imports_cell(irs)
        notebook.cells.append(imports_cell)

        # Add parameters cell (Papermill-tagged)
        notebook.cells.append(self._create_parameters_cell(irs))

        # Add data loading cell (critical for execution)
        notebook.cells.append(self._create_data_loading_cell())

        # Convert activities with IR to code cells
        for idx, activity in enumerate(activities, start=1):
            # Add documentation cell
            notebook.cells.append(self._create_doc_cell(activity, idx))

            # Add executable code cell from IR
            code_cell = self._activity_to_code(activity, validate=validate_syntax)
            if code_cell:
                notebook.cells.append(code_cell)

        # Add data saving cell
        notebook.cells.append(self._create_data_saving_cell())

        # Add footer cell
        notebook.cells.append(self._create_footer_cell())

        # Add notebook metadata
        notebook.metadata["lobster"] = self._create_metadata(len(irs), len(activities))

        # Save to .lobster/notebooks/
        output_dir = Path.home() / ".lobster" / "notebooks"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.ipynb"

        with open(output_path, "w") as f:
            nbformat.write(notebook, f)

        logger.info(f"Notebook exported: {output_path}")
        logger.info(
            f"IR coverage: {len(irs)}/{len(activities)} activities "
            f"({len(irs)/len(activities)*100:.1f}%)"
        )

        return output_path

    def _filter_activities(self, strategy: str) -> List[Dict[str, Any]]:
        """
        Filter provenance activities based on strategy.

        Args:
            strategy: Filter strategy ("successful" | "all" | "manual")

        Returns:
            List of filtered activity dictionaries
        """
        if strategy == "successful":
            # Exclude activities with error markers
            return [
                a
                for a in self.provenance.activities
                if not a.get("error") and a.get("type") != "failed_operation"
            ]
        elif strategy == "all":
            return self.provenance.activities.copy()
        else:
            # Manual filtering not implemented yet
            logger.warning("Manual filter strategy not implemented, using 'successful'")
            return self._filter_activities("successful")

    def _extract_irs(self, activities: List[Dict[str, Any]]) -> List[AnalysisStep]:
        """
        Extract AnalysisStep IR objects from provenance activities.

        Args:
            activities: List of activity dictionaries from provenance

        Returns:
            List of AnalysisStep objects (may be empty if no IRs present)
        """
        irs = []

        for activity in activities:
            ir_dict = activity.get("ir")

            if ir_dict is not None and isinstance(ir_dict, dict):
                try:
                    ir = AnalysisStep.from_dict(ir_dict)
                    # Check if this IR should be included in notebook export
                    if ir.exportable:
                        irs.append(ir)
                        logger.debug(f"Extracted IR for operation: {ir.operation}")
                    else:
                        logger.debug(
                            f"Skipping non-exportable IR for operation: {ir.operation}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize IR for activity "
                        f"{activity.get('type')}: {e}"
                    )

        return irs

    def _create_header_cell(
        self, name: str, description: str, n_irs: int
    ) -> NotebookNode:
        """
        Create markdown header cell for notebook.

        Args:
            name: Notebook name
            description: Optional description
            n_irs: Number of IR objects extracted

        Returns:
            Markdown cell with header content
        """
        coverage_pct = (
            n_irs / len(self.provenance.activities) * 100
            if self.provenance.activities
            else 0
        )

        header_content = f"""# {name}

**Generated from Lobster AI Session**

{description if description else "Reproducible bioinformatics analysis workflow"}

**Created:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Lobster Version:** {self._get_lobster_version()}
**IR Coverage:** {n_irs}/{len(self.provenance.activities)} activities ({coverage_pct:.1f}%)

---

## Workflow Summary

This notebook was automatically generated from a Lobster analysis session using
Service-Emitted Intermediate Representation (IR). Each analysis step contains
complete code generation instructions for reproducibility.

**Session Statistics:**
- Activities: {len(self.provenance.activities)}
- Entities: {len(self.provenance.entities)}
- Data Modalities: {len(self.data_manager.modalities)}
- IR Objects: {n_irs}

---
"""
        return new_markdown_cell(header_content)

    def _create_imports_cell(self, irs: List[AnalysisStep]) -> NotebookNode:
        """
        Create imports cell with deduplicated imports from all IRs.

        Args:
            irs: List of AnalysisStep objects

        Returns:
            Code cell with all required imports
        """
        if not irs:
            # Default imports if no IRs present
            imports = [
                "import numpy as np",
                "import pandas as pd",
                "import scanpy as sc",
                "import anndata",
            ]
        else:
            imports = extract_unique_imports(irs)

        imports_code = "# Required imports\n" + "\n".join(imports)

        logger.debug(f"Generated {len(imports)} import statements")

        return new_code_cell(imports_code)

    def _create_parameters_cell(self, irs: List[AnalysisStep]) -> NotebookNode:
        """
        Create Papermill parameters cell dynamically from IR schemas.

        Args:
            irs: List of AnalysisStep objects

        Returns:
            Code cell tagged for Papermill parameter injection
        """
        # Collect all Papermill-injectable parameters from IRs
        injectable_params: Dict[str, Any] = {}

        for ir in irs:
            papermill_params = ir.get_papermill_parameters()
            injectable_params.update(papermill_params)

        # Standard parameters
        code = """# Parameters (tagged for Papermill parameter injection)
# Override with: papermill notebook.ipynb output.ipynb -p param_name value

# Input/Output paths
input_data = "dataset.h5ad"  # Path to input H5AD file
output_prefix = "results"     # Prefix for output files

# Reproducibility
random_seed = 42              # Random seed for reproducibility
"""

        # Add IR-derived parameters if present
        if injectable_params:
            code += "\n# Analysis parameters (from workflow IR)\n"
            for param_name, param_value in sorted(injectable_params.items()):
                # Format value for Python
                if isinstance(param_value, str):
                    value_str = f'"{param_value}"'
                elif isinstance(param_value, (list, tuple)):
                    value_str = repr(param_value)
                else:
                    value_str = str(param_value)

                code += f"{param_name} = {value_str}\n"

        cell = new_code_cell(code)
        cell.metadata["tags"] = ["parameters"]

        logger.debug(
            f"Generated parameters cell with {len(injectable_params)} "
            f"IR-derived parameters"
        )

        return cell

    def _create_data_loading_cell(self) -> NotebookNode:
        """
        Create cell for loading input data using IR if available.

        Returns:
            Code cell with data loading code
        """
        # Try to find data loading IR from provenance
        for activity in self.provenance.activities:
            tool_name = activity.get("type", "")
            if tool_name == "load_dataset":
                ir_dict = activity.get("ir")
                if ir_dict:
                    try:
                        ir = AnalysisStep.from_dict(ir_dict)
                        code = ir.render()
                        logger.debug("Using IR for data loading cell")
                        return new_code_cell(code)
                    except Exception as e:
                        logger.warning(f"Failed to use IR for data loading: {e}")

        # Fallback to hardcoded cell if no IR found
        logger.debug("Using fallback hardcoded data loading cell")
        code = """# Load input data
import anndata as ad

adata = ad.read_h5ad(input_data)
print(f"Loaded data: {adata.n_obs} cells × {adata.n_vars} genes")
"""
        return new_code_cell(code)

    def _create_data_saving_cell(self) -> NotebookNode:
        """
        Create cell for saving output data using IR if available.

        Returns:
            Code cell with data saving code
        """
        # Try to find data saving IR from provenance
        for activity in reversed(self.provenance.activities):  # Check most recent first
            tool_name = activity.get("type", "")
            if tool_name == "save_dataset":
                ir_dict = activity.get("ir")
                if ir_dict:
                    try:
                        ir = AnalysisStep.from_dict(ir_dict)
                        code = ir.render()
                        logger.debug("Using IR for data saving cell")
                        return new_code_cell(code)
                    except Exception as e:
                        logger.warning(f"Failed to use IR for data saving: {e}")

        # Fallback to hardcoded cell if no IR found
        logger.debug("Using fallback hardcoded data saving cell")
        code = """# Save processed data
output_path = f"{output_prefix}_processed.h5ad"
adata.write_h5ad(output_path)
print(f"Saved processed data to: {output_path}")
"""
        return new_code_cell(code)

    def _create_doc_cell(
        self, activity: Dict[str, Any], step_number: int
    ) -> NotebookNode:
        """
        Create markdown documentation cell for activity.

        Args:
            activity: Activity dictionary from provenance
            step_number: Sequential step number

        Returns:
            Markdown cell documenting the analysis step
        """
        activity_type = activity.get("type", "unknown")
        params = activity.get("parameters", {})
        timestamp = activity.get("timestamp", "unknown")

        # Check if IR is present
        has_ir = activity.get("ir") is not None
        ir_status = "✓ IR available" if has_ir else "⚠ No IR (manual review needed)"

        doc_content = f"""## Step {step_number}: {activity_type}

**Timestamp:** {timestamp}
**Agent:** {activity.get("agent", "unknown")}
**Status:** {ir_status}

"""
        if activity.get("description"):
            doc_content += f"**Description:** {activity['description']}\n\n"

        if params:
            doc_content += "**Parameters:**\n"
            for key, value in params.items():
                # Format parameter value for display
                if isinstance(value, (list, tuple)) and len(value) > 5:
                    value_str = (
                        f"[{', '.join(str(x) for x in value[:3])}...] "
                        f"(length: {len(value)})"
                    )
                else:
                    value_str = str(value)
                doc_content += f"- `{key}`: {value_str}\n"

        return new_markdown_cell(doc_content)

    def _activity_to_code(
        self, activity: Dict[str, Any], validate: bool = True
    ) -> Optional[NotebookNode]:
        """
        Convert provenance activity to executable Python code using IR.

        Args:
            activity: Activity dictionary from provenance
            validate: Whether to validate syntax

        Returns:
            Code cell with executable Python, or None if cannot generate
        """
        ir_dict = activity.get("ir")

        if ir_dict is None:
            # No IR available - return placeholder
            tool_name = activity.get("type", "unknown")
            logger.warning(f"No IR for activity: {tool_name}")

            return new_code_cell(
                f"# TODO: Manual review needed for {tool_name}\n"
                f"# No IR available - service needs to emit AnalysisStep\n"
                f"# Parameters: {activity.get('parameters', {})}\n"
                f"pass"
            )

        # Deserialize IR
        try:
            ir = AnalysisStep.from_dict(ir_dict)
        except Exception as e:
            logger.error(f"Failed to deserialize IR: {e}")
            return new_code_cell(
                f"# ERROR: Failed to deserialize IR\n" f"# {str(e)}\n" f"pass"
            )

        # Render code from template
        try:
            code = ir.render()

            # Validate syntax if requested
            if validate and ir.validates_on_export:
                try:
                    ir.validate_rendered_code()
                    logger.debug(f"Validated syntax for {ir.operation}")
                except SyntaxError as e:
                    logger.error(
                        f"Syntax error in generated code for {ir.operation}: {e}"
                    )
                    code = (
                        f"# SYNTAX ERROR in generated code:\n"
                        f"# {str(e)}\n"
                        f"# Original template:\n"
                        f"# {ir.code_template}\n"
                        f"{code}"
                    )

            return new_code_cell(code)

        except Exception as e:
            logger.error(f"Failed to render code for {ir.operation}: {e}")
            return new_code_cell(
                f"# ERROR: Template rendering failed\n"
                f"# Operation: {ir.operation}\n"
                f"# Error: {str(e)}\n"
                f"pass"
            )

    def _create_footer_cell(self) -> NotebookNode:
        """
        Create markdown footer cell.

        Returns:
            Markdown cell with export instructions
        """
        footer_content = """---

## Results Export

This analysis is now complete. Results have been saved with provenance tracking.

### Next Steps

1. **Review Results**: Check output files and visualizations
2. **Version Control**: Commit this notebook to Git
3. **Share**: Distribute notebook for reproducibility
4. **Re-run**: Use Papermill to execute on new data

### Usage Example

```bash
# Execute notebook with new data
papermill notebook.ipynb output.ipynb \\
    -p input_data "new_dataset.h5ad" \\
    -p output_prefix "new_analysis"
```

### IR Coverage Note

This notebook was generated using Service-Emitted IR. Steps marked with "⚠ No IR"
require manual review, as the corresponding service hasn't been updated yet to emit
AnalysisStep objects. IR-enabled steps are fully reproducible.

---

*Generated with [Lobster AI](https://github.com/OmicsOS/lobster) - Multi-Agent Bioinformatics Platform*
*Using Service-Emitted IR Architecture for automatic code generation*
"""
        return new_markdown_cell(footer_content)

    def _create_metadata(self, n_irs: int, n_activities: int) -> Dict[str, Any]:
        """
        Create notebook metadata with provenance link and IR stats.

        Args:
            n_irs: Number of IR objects
            n_activities: Number of activities

        Returns:
            Metadata dictionary for notebook
        """
        import os

        return {
            "source_session_id": self.provenance.namespace,
            "created_by": os.getenv("USER", "unknown"),
            "created_at": datetime.now().isoformat(),
            "lobster_version": self._get_lobster_version(),
            "dependencies": self._snapshot_dependencies(),
            "source_provenance_summary": {
                "n_activities": len(self.provenance.activities),
                "n_entities": len(self.provenance.entities),
                "n_agents": len(self.provenance.agents),
            },
            "ir_statistics": {
                "n_irs_extracted": n_irs,
                "n_activities": n_activities,
                "coverage_percent": (
                    n_irs / n_activities * 100 if n_activities > 0 else 0
                ),
            },
        }

    def _snapshot_dependencies(self) -> Dict[str, str]:
        """
        Snapshot current package versions.

        Returns:
            Dictionary of package versions
        """
        versions = {}

        # Core bioinformatics packages
        packages = ["scanpy", "anndata", "pandas", "numpy", "scipy", "scikit-learn"]

        for package in packages:
            try:
                module = __import__(package)
                versions[package] = getattr(module, "__version__", "unknown")
            except ImportError:
                pass

        # Add Python version
        versions["python"] = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

        return versions

    def _get_lobster_version(self) -> str:
        """Get Lobster version."""
        try:
            from lobster.version import __version__

            return __version__
        except ImportError:
            return "unknown"
