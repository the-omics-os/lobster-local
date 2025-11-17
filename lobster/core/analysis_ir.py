"""
Intermediate Representation (IR) for reproducible analysis operations.

This module provides the Service-Emitted IR architecture that enables
automatic, deterministic Jupyter notebook generation. Services emit
AnalysisStep objects that contain everything needed to generate
executable code without manual mapping registries.
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpec:
    """
    Type and behavior specification for analysis parameters.

    This class defines how a parameter should be handled during
    notebook generation, including whether it can be overridden
    via Papermill parameter injection.

    Attributes:
        param_type: Python type as string (e.g., "int", "float", "List[str]", "Path")
        papermill_injectable: Whether this parameter can be overridden by Papermill
        default_value: Default value if not specified
        required: Whether this parameter must be provided
        validation_rule: Optional validation expression (e.g., "min_genes > 0")
        description: Human-readable parameter description

    Example:
        >>> spec = ParameterSpec(
        ...     param_type="int",
        ...     papermill_injectable=True,
        ...     default_value=200,
        ...     required=False,
        ...     validation_rule="min_genes > 0",
        ...     description="Minimum genes per cell threshold"
        ... )
    """

    param_type: str
    papermill_injectable: bool
    default_value: Any
    required: bool
    validation_rule: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to JSON-compatible dictionary.

        Returns:
            Dictionary with all parameter spec fields

        Raises:
            TypeError: If default_value is not JSON-serializable
        """
        import json

        # Validate JSON serializability of default_value
        try:
            json.dumps({"value": self.default_value})
        except TypeError as e:
            raise TypeError(
                f"ParameterSpec default_value is not JSON-serializable: "
                f"{type(self.default_value).__name__} = {self.default_value!r}. "
                f"Parameter: '{self.description or self.param_type}'. "
                f"Error: {e}"
            ) from e

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterSpec":
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary with parameter spec fields

        Returns:
            ParameterSpec instance
        """
        return cls(**data)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ParameterSpec(type={self.param_type}, "
            f"injectable={self.papermill_injectable}, "
            f"default={self.default_value})"
        )


@dataclass
class AnalysisStep:
    """
    Intermediate Representation for reproducible analysis operations.

    This is the canonical, single source of truth for how a service
    operation should be represented in an exported Jupyter notebook.
    Services emit AnalysisStep objects alongside their results to
    enable automatic notebook generation without manual code mapping.

    Attributes:
        operation: Fully-qualified operation name (e.g., "scanpy.pp.normalize_total")
        tool_name: Original tool name for provenance linking
        description: Human-readable description (appears in notebook markdown)
        library: Required library name (e.g., "scanpy", "pandas")
        code_template: Jinja2 template with {{ variable }} placeholders
        imports: List of import statements (e.g., ["import scanpy as sc"])
        parameters: Actual parameter values used in this execution
        parameter_schema: Type info and Papermill flags for each parameter
        input_entities: List of input data references
        output_entities: List of output data references
        execution_context: Random seeds, versions, timestamps, etc.
        validates_on_export: Whether to run AST syntax validation on export
        requires_validation: Whether full execution validation is recommended
        exportable: Whether to include in notebook export (default True)

    Example:
        >>> ir = AnalysisStep(
        ...     operation="scanpy.pp.normalize_total",
        ...     tool_name="normalize",
        ...     description="Total-count normalize expression data",
        ...     library="scanpy",
        ...     code_template="sc.pp.normalize_total(adata, target_sum={{ target }})",
        ...     imports=["import scanpy as sc"],
        ...     parameters={"target": 1e4},
        ...     parameter_schema={
        ...         "target": ParameterSpec(
        ...             param_type="float",
        ...             papermill_injectable=True,
        ...             default_value=1e4,
        ...             required=False,
        ...             description="Target sum for normalization"
        ...         )
        ...     },
        ...     input_entities=["adata"],
        ...     output_entities=["adata"],
        ...     execution_context={"scanpy_version": "1.9.3"}
        ... )
    """

    # Identity
    operation: str
    tool_name: str
    description: str

    # Code generation
    library: str
    code_template: str
    imports: List[str]

    # Parameters
    parameters: Dict[str, Any]
    parameter_schema: Dict[str, ParameterSpec]

    # Data flow
    input_entities: List[str] = field(default_factory=list)
    output_entities: List[str] = field(default_factory=list)

    # Metadata
    execution_context: Dict[str, Any] = field(default_factory=dict)

    # Validation flags
    validates_on_export: bool = True
    requires_validation: bool = False

    # Export control
    exportable: bool = True  # Whether to include in notebook export

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to JSON-compatible dictionary.

        This method handles the conversion of ParameterSpec objects
        within parameter_schema to dictionaries for JSON storage.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        data = asdict(self)

        # Convert ParameterSpec objects in parameter_schema
        data["parameter_schema"] = {
            k: v.to_dict() if isinstance(v, ParameterSpec) else v
            for k, v in self.parameter_schema.items()
        }

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisStep":
        """
        Deserialize from dictionary.

        This method reconstructs ParameterSpec objects from
        dictionaries in parameter_schema.

        Args:
            data: Dictionary representation from to_dict()

        Returns:
            AnalysisStep instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        required_fields = [
            "operation",
            "tool_name",
            "description",
            "library",
            "code_template",
            "imports",
            "parameters",
            "parameter_schema",
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Convert parameter_schema dicts back to ParameterSpec objects
        if "parameter_schema" in data:
            data["parameter_schema"] = {
                k: ParameterSpec.from_dict(v) if isinstance(v, dict) else v
                for k, v in data["parameter_schema"].items()
            }

        return cls(**data)

    def validate_template(self) -> bool:
        """
        Validate that the code template uses correct Jinja2 syntax.

        Returns:
            True if template is valid

        Raises:
            ValueError: If template has invalid syntax
        """
        try:
            from jinja2 import Template

            Template(self.code_template)
            return True
        except Exception as e:
            raise ValueError(f"Invalid Jinja2 template: {e}") from e

    def render(self, **override_params) -> str:
        """
        Render code template with parameters.

        Args:
            **override_params: Optional parameter overrides

        Returns:
            Rendered Python code string

        Raises:
            ValueError: If template rendering fails
        """
        try:
            from jinja2 import Template

            # Merge parameters with overrides
            params = {**self.parameters, **override_params}

            template = Template(self.code_template)
            code = template.render(**params)

            return code
        except Exception as e:
            logger.error(f"Failed to render template for {self.operation}: {e}")
            raise ValueError(f"Template rendering failed: {e}") from e

    def validate_rendered_code(self, **override_params) -> bool:
        """
        Render and validate generated code syntax.

        Args:
            **override_params: Optional parameter overrides

        Returns:
            True if rendered code has valid Python syntax

        Raises:
            SyntaxError: If generated code is invalid
        """
        import ast

        code = self.render(**override_params)

        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Invalid generated code for {self.operation}: {e}")
            raise

    def get_papermill_parameters(self) -> Dict[str, Any]:
        """
        Extract parameters that can be overridden via Papermill.

        Returns:
            Dictionary of injectable parameter names and current values
        """
        return {
            param_name: self.parameters.get(param_name, spec.default_value)
            for param_name, spec in self.parameter_schema.items()
            if spec.papermill_injectable
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AnalysisStep(operation={self.operation}, "
            f"tool={self.tool_name}, "
            f"params={len(self.parameters)})"
        )


# Utility functions for IR validation


def validate_ir_list(irs: List[Dict[str, Any]]) -> List[str]:
    """
    Validate a list of IR dictionaries.

    Args:
        irs: List of IR dictionaries from provenance

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    for idx, ir_dict in enumerate(irs):
        try:
            ir = AnalysisStep.from_dict(ir_dict)

            # Validate template syntax
            try:
                ir.validate_template()
            except ValueError as e:
                errors.append(f"IR {idx} ({ir.operation}): Invalid template - {e}")

            # Check for empty required fields
            if not ir.operation:
                errors.append(f"IR {idx}: Empty operation field")
            if not ir.code_template:
                errors.append(f"IR {idx}: Empty code_template field")
            if not ir.imports:
                errors.append(f"IR {idx}: No imports specified")

        except Exception as e:
            errors.append(f"IR {idx}: Deserialization failed - {e}")

    return errors


def extract_unique_imports(irs: List[AnalysisStep]) -> List[str]:
    """
    Extract and deduplicate imports from multiple IR objects.

    Imports are sorted in standard order: stdlib → third-party → local

    Args:
        irs: List of AnalysisStep objects

    Returns:
        Sorted list of unique import statements
    """
    imports = set()

    for ir in irs:
        imports.update(ir.imports)

    # Standard library modules for sorting
    stdlib_modules = {
        "os",
        "sys",
        "pathlib",
        "datetime",
        "json",
        "re",
        "math",
        "time",
        "typing",
    }

    def import_sort_key(import_str: str) -> tuple:
        """Sort key: stdlib (0) → third-party (1) → local (2)."""
        # Extract module name
        if import_str.startswith("from"):
            parts = import_str.split()
            module = parts[1].split(".")[0] if len(parts) > 1 else ""
        elif import_str.startswith("import"):
            parts = import_str.split()
            module = parts[1].split(".")[0] if len(parts) > 1 else ""
        else:
            module = ""

        # Categorize
        if module in stdlib_modules:
            return (0, import_str)  # stdlib first
        elif module.startswith("lobster"):
            return (2, import_str)  # local last
        else:
            return (1, import_str)  # third-party middle

    return sorted(imports, key=import_sort_key)


def create_minimal_ir(
    operation: str, tool_name: str, code: str, library: str = "unknown"
) -> AnalysisStep:
    """
    Create a minimal AnalysisStep for operations without full IR support.

    This is a fallback for tools that haven't been updated yet.
    The generated notebook will have a TODO comment.

    Args:
        operation: Operation name
        tool_name: Tool name for provenance
        code: Basic code to include
        library: Library name

    Returns:
        Minimal AnalysisStep with TODO markers
    """
    return AnalysisStep(
        operation=operation,
        tool_name=tool_name,
        description=f"TODO: Add proper IR support for {operation}",
        library=library,
        code_template=f"# TODO: Manual review needed for {operation}\n{code}",
        imports=[],
        parameters={},
        parameter_schema={},
        input_entities=[],
        output_entities=[],
        execution_context={"created": datetime.now().isoformat(), "ir_type": "minimal"},
        validates_on_export=False,
        requires_validation=True,
    )


def create_data_loading_ir(
    input_param_name: str = "input_data",
    description: str = "Load single-cell RNA-seq data from H5AD file",
) -> AnalysisStep:
    """
    Create IR for loading AnnData from H5AD file.

    Args:
        input_param_name: Name of parameter containing input file path
        description: Human-readable description of the loading operation

    Returns:
        AnalysisStep with data loading code template
    """
    # Parameter schema with Papermill flag
    parameter_schema = {
        input_param_name: ParameterSpec(
            param_type="str",
            papermill_injectable=True,
            default_value="dataset.h5ad",
            required=True,
            validation_rule=f"{input_param_name}.endswith('.h5ad')",
            description="Path to input H5AD file",
        ),
    }

    # Jinja2 template with parameter placeholder
    code_template = f"""# Load input data
import anndata as ad

adata = ad.read_h5ad({{{{ {input_param_name} }}}})
print(f"Loaded data: {{adata.n_obs}} cells × {{adata.n_vars}} genes")
"""

    return AnalysisStep(
        operation="anndata.read_h5ad",
        tool_name="load_dataset",
        description=description,
        library="anndata",
        code_template=code_template,
        imports=["import anndata as ad"],
        parameters={input_param_name: "dataset.h5ad"},
        parameter_schema=parameter_schema,
        input_entities=["file"],
        output_entities=["adata"],
        execution_context={
            "operation_type": "data_io",
            "io_direction": "input",
            "file_format": "h5ad",
        },
        validates_on_export=False,  # Data loading doesn't require validation
        requires_validation=False,
    )


def create_data_saving_ir(
    output_prefix_param: str = "output_prefix",
    filename_suffix: str = "processed",
    description: str = "Save processed single-cell data to H5AD file",
    compression: str = "gzip",
) -> AnalysisStep:
    """
    Create IR for saving AnnData to H5AD file.

    Args:
        output_prefix_param: Name of parameter containing output file prefix
        filename_suffix: Suffix to append to output filename
        description: Human-readable description of the saving operation
        compression: H5AD compression method ('gzip', 'lzf', None)

    Returns:
        AnalysisStep with data saving code template
    """
    # Parameter schema with Papermill flags
    parameter_schema = {
        output_prefix_param: ParameterSpec(
            param_type="str",
            papermill_injectable=True,
            default_value="results",
            required=True,
            validation_rule=f"len({output_prefix_param}) > 0",
            description="Prefix for output files",
        ),
        "compression": ParameterSpec(
            param_type="str",
            papermill_injectable=True,
            default_value=compression,
            required=False,
            validation_rule="compression in ['gzip', 'lzf', None, 'none']",
            description="Compression method for H5AD file",
        ),
    }

    # Jinja2 template with parameter placeholders
    code_template = f"""# Save processed data
output_path = f"{{{output_prefix_param}}}_{filename_suffix}.h5ad"
adata.write_h5ad(output_path, compression={{% if compression %}}{{{{ compression | tojson }}}}{{% else %}}None{{% endif %}})
print(f"Saved processed data to: {{output_path}}")
"""

    return AnalysisStep(
        operation="anndata.write_h5ad",
        tool_name="save_dataset",
        description=description,
        library="anndata",
        code_template=code_template,
        imports=["import anndata as ad"],
        parameters={
            output_prefix_param: "results",
            "compression": compression,
        },
        parameter_schema=parameter_schema,
        input_entities=["adata"],
        output_entities=["file"],
        execution_context={
            "operation_type": "data_io",
            "io_direction": "output",
            "file_format": "h5ad",
            "compression": compression,
        },
        validates_on_export=False,  # Data saving doesn't require validation
        requires_validation=False,
    )
