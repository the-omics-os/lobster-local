"""
Shared modality commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient
    import numpy as np
    import pandas as pd

from lobster.cli_internal.commands.output_adapter import OutputAdapter


# ============================================================================
# Helper Functions (Matrix/DataFrame Formatting)
# ============================================================================


def _get_matrix_info(matrix) -> Dict[str, Any]:
    """
    Get information about a matrix (sparse or dense).

    Args:
        matrix: numpy array or scipy sparse matrix

    Returns:
        Dictionary with matrix information (shape, dtype, sparsity, memory)
    """
    import scipy.sparse as sp

    info = {}
    info["shape"] = matrix.shape
    info["dtype"] = str(matrix.dtype)

    if sp.issparse(matrix):
        info["sparse"] = True
        info["format"] = matrix.format.upper()
        info["nnz"] = matrix.nnz
        info["density"] = (matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100
        info["memory_mb"] = (
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        ) / (1024**2)
    else:
        info["sparse"] = False
        info["format"] = "Dense"
        info["memory_mb"] = matrix.nbytes / (1024**2)
        info["density"] = 100.0

    return info


def _format_data_preview(matrix, max_rows: int = 5, max_cols: int = 5) -> Dict[str, Any]:
    """
    Format a data matrix preview as table data.

    Args:
        matrix: numpy array or scipy sparse matrix
        max_rows: Maximum rows to preview
        max_cols: Maximum columns to preview

    Returns:
        Table data dictionary for output adapter
    """
    import numpy as np
    import scipy.sparse as sp

    # Convert sparse to dense for preview if needed
    if sp.issparse(matrix):
        preview_rows = min(max_rows, matrix.shape[0])
        preview_cols = min(max_cols, matrix.shape[1])
        preview_data = matrix[:preview_rows, :preview_cols].toarray()
    else:
        preview_rows = min(max_rows, matrix.shape[0])
        preview_cols = min(max_cols, matrix.shape[1])
        preview_data = matrix[:preview_rows, :preview_cols]

    # Build table data
    columns = [{"name": "", "style": "bold grey50"}]  # Row index column
    for i in range(preview_cols):
        columns.append({"name": f"[{i}]", "style": "cyan"})

    rows = []
    for i in range(preview_rows):
        row_values = [f"[{i}]"]
        for j in range(preview_cols):
            val = preview_data[i, j]
            # Format the value
            if isinstance(val, (int, np.integer)):
                formatted = str(val)
            elif isinstance(val, (float, np.floating)):
                formatted = f"{val:.2f}"
            else:
                formatted = str(val)
            row_values.append(formatted)
        rows.append(row_values)

    # Add ellipsis row if there are more rows
    if matrix.shape[0] > max_rows or matrix.shape[1] > max_cols:
        ellipsis_row = ["..."] * (min(preview_cols, matrix.shape[1]) + 1)
        rows.append(ellipsis_row)

    return {"title": None, "columns": columns, "rows": rows}


def _format_dataframe_preview(df: "pd.DataFrame", max_rows: int = 5) -> Dict[str, Any]:
    """
    Format a DataFrame preview as table data.

    Args:
        df: pandas DataFrame
        max_rows: Maximum rows to preview

    Returns:
        Table data dictionary for output adapter
    """
    import numpy as np
    import pandas as pd

    # Build columns
    columns = [{"name": "Index", "style": "bold grey50"}]
    for col in df.columns[:10]:  # Limit to first 10 columns
        dtype_str = str(df[col].dtype)
        style = (
            "cyan"
            if dtype_str.startswith("int") or dtype_str.startswith("float")
            else "white"
        )
        columns.append({"name": str(col), "style": style})

    # Add rows
    rows = []
    preview_rows = min(max_rows, len(df))
    for idx in range(preview_rows):
        row_data = [str(df.index[idx])]
        for col in df.columns[:10]:
            val = df.iloc[idx][col]
            # Format based on type
            if pd.isna(val):
                formatted = "NaN"
            elif isinstance(val, (int, np.integer)):
                formatted = str(val)
            elif isinstance(val, (float, np.floating)):
                formatted = f"{val:.2f}"
            else:
                formatted = str(val)[:20]  # Truncate long strings
            row_data.append(formatted)
        rows.append(row_data)

    # Add ellipsis if there are more rows
    if len(df) > max_rows:
        ellipsis_row = ["..."] * (min(10, len(df.columns)) + 1)
        rows.append(ellipsis_row)

    # Add more columns indicator
    if len(df.columns) > 10:
        columns.append({"name": f"... +{len(df.columns) - 10} more", "style": "dim"})

    return {"title": None, "columns": columns, "rows": rows}


def _format_array_info(arrays_dict: Dict[str, "np.ndarray"]) -> Optional[Dict[str, Any]]:
    """
    Format array information (obsm/varm) as table data.

    Args:
        arrays_dict: Dictionary of array name to numpy array

    Returns:
        Table data dictionary for output adapter, or None if empty
    """
    if not arrays_dict:
        return None

    columns = [
        {"name": "Key", "style": "bold cyan"},
        {"name": "Shape", "style": "white"},
        {"name": "Dtype", "style": "grey70"},
    ]

    rows = []
    for key, arr in arrays_dict.items():
        shape_str = " × ".join(str(d) for d in arr.shape)
        dtype_str = str(arr.dtype)
        rows.append([key, shape_str, dtype_str])

    return {"title": None, "columns": columns, "rows": rows}


# ============================================================================
# Command Functions
# ============================================================================


def modalities_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show detailed information about all loaded modalities.

    Displays AnnData structure details for each modality including:
    - Shape (observations × variables)
    - Observation and variable columns
    - Layers
    - Multi-dimensional annotations (obsm)
    - Unstructured data (uns)

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    if not hasattr(client.data_manager, "list_modalities"):
        output.print(
            "[grey50]Modality information not available (using legacy DataManager)[/grey50]",
            style="info"
        )
        return None

    modalities = client.data_manager.list_modalities()

    if not modalities:
        output.print("[grey50]No modalities loaded[/grey50]", style="info")
        return "No modalities loaded"

    output.print("[bold red]🧬 Modality Details[/bold red]\n", style="info")

    modalities_shown = 0
    for modality_name in modalities:
        try:
            adata = client.data_manager.get_modality(modality_name)

            # Create modality table
            mod_table_data = {
                "title": f"🧬 {modality_name}",
                "columns": [
                    {"name": "Property", "style": "bold grey93"},
                    {"name": "Value", "style": "white"},
                ],
                "rows": []
            }

            mod_table_data["rows"].append([
                "Shape",
                f"{adata.n_obs} obs × {adata.n_vars} vars"
            ])

            # Show obs columns
            obs_cols = list(adata.obs.columns)
            if obs_cols:
                cols_preview = ", ".join(obs_cols[:5])
                if len(obs_cols) > 5:
                    cols_preview += f" ... (+{len(obs_cols)-5} more)"
                mod_table_data["rows"].append(["Obs Columns", cols_preview])

            # Show var columns
            var_cols = list(adata.var.columns)
            if var_cols:
                var_preview = ", ".join(var_cols[:5])
                if len(var_cols) > 5:
                    var_preview += f" ... (+{len(var_cols)-5} more)"
                mod_table_data["rows"].append(["Var Columns", var_preview])

            # Show layers
            if adata.layers:
                layers_str = ", ".join(list(adata.layers.keys()))
                mod_table_data["rows"].append(["Layers", layers_str])

            # Show obsm
            if adata.obsm:
                obsm_str = ", ".join(list(adata.obsm.keys()))
                mod_table_data["rows"].append(["Obsm", obsm_str])

            # Show some uns info
            if adata.uns:
                uns_keys = list(adata.uns.keys())[:5]
                uns_str = ", ".join(uns_keys)
                if len(adata.uns) > 5:
                    uns_str += f" ... (+{len(adata.uns)-5} more)"
                mod_table_data["rows"].append(["Uns Keys", uns_str])

            output.print_table(mod_table_data)
            output.print("")  # Blank line
            modalities_shown += 1

        except Exception as e:
            output.print(
                f"[red]Error accessing modality {modality_name}: {e}[/red]",
                style="error"
            )

    if modalities_shown > 0:
        return f"Displayed {modalities_shown} modalities"
    else:
        return "No modalities could be displayed"


def modality_describe(
    client: "AgentClient",
    output: OutputAdapter,
    modality_name: Optional[str] = None
) -> Optional[str]:
    """
    Show detailed information about a specific modality.

    Provides comprehensive breakdown of AnnData structure:
    - Basic information (shape, memory, matrix type)
    - Data matrix preview
    - Observations (obs) with column types and preview
    - Variables (var) with column types and preview
    - Additional data structures (layers, obsm, varm, obsp, varp, uns)
    - Metadata from DataManager

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        modality_name: Name of modality to describe (optional, prompts if missing)

    Returns:
        Summary string for conversation history, or None
    """
    import numpy as np
    import pandas as pd

    if not hasattr(client.data_manager, "list_modalities"):
        output.print(
            "[grey50]Describe command not available (using legacy DataManager)[/grey50]",
            style="info"
        )
        return None

    # Check if modality_name provided
    if not modality_name:
        output.print("[red]Usage: /describe <modality_name>[/red]", style="error")
        output.print("[dim]Available modalities:[/dim]", style="info")
        modalities = client.data_manager.list_modalities()
        for mod in modalities:
            output.print(f"  • {mod}", style="info")
        return None

    # Check if modality exists
    if modality_name not in client.data_manager.list_modalities():
        output.print(f"[red]Modality '{modality_name}' not found[/red]", style="error")
        output.print("[dim]Available modalities:[/dim]", style="info")
        for mod in client.data_manager.list_modalities():
            output.print(f"  • {mod}", style="info")
        return None

    try:
        # Get the modality
        adata = client.data_manager.get_modality(modality_name)

        # Create main header
        output.print("", style="info")
        output.print(
            f"[bold orange1]🧬 Modality: {modality_name}[/bold orange1]",
            style="info"
        )
        output.print("━" * 60, style="info")

        # ====================================================================
        # Basic Information
        # ====================================================================
        matrix_info = _get_matrix_info(adata.X)
        output.print("\n[bold white]📊 Basic Information[/bold white]", style="info")

        basic_table_data = {
            "title": None,
            "columns": [
                {"name": "Property", "style": "grey70"},
                {"name": "Value", "style": "white"},
            ],
            "rows": [
                ["Shape", f"{adata.n_obs:,} observations × {adata.n_vars:,} variables"],
                ["Memory", f"{matrix_info['memory_mb']:.1f} MB"],
            ]
        }

        if matrix_info["sparse"]:
            basic_table_data["rows"].append([
                "Matrix Type",
                f"Sparse ({matrix_info['format']}, {matrix_info['density']:.1f}% density)"
            ])
            basic_table_data["rows"].append([
                "Non-zero",
                f"{matrix_info['nnz']:,} elements"
            ])
        else:
            basic_table_data["rows"].append(["Matrix Type", "Dense array"])

        basic_table_data["rows"].append(["Data Type", matrix_info["dtype"]])

        output.print_table(basic_table_data)

        # ====================================================================
        # Data Matrix (X) Preview
        # ====================================================================
        output.print("\n[bold white]📈 Data Matrix (X)[/bold white]", style="info")
        output.print("[grey70]Preview (first 5×5 cells):[/grey70]", style="info")
        x_preview = _format_data_preview(adata.X)
        output.print_table(x_preview)

        # ====================================================================
        # Observations (obs)
        # ====================================================================
        if not adata.obs.empty:
            output.print(
                f"\n[bold white]🔬 Observations (obs) - {adata.n_obs:,} cells[/bold white]",
                style="info"
            )

            # Column information
            obs_info = []
            for col in adata.obs.columns:
                dtype = str(adata.obs[col].dtype)
                obs_info.append(f"{col} ({dtype})")

            output.print(
                f"[grey70]Columns ({len(adata.obs.columns)}):[/grey70] {', '.join(obs_info[:5])}",
                style="info"
            )
            if len(obs_info) > 5:
                output.print(
                    f"[grey50]... and {len(obs_info) - 5} more columns[/grey50]",
                    style="info"
                )

            # Preview table
            if len(adata.obs) > 0:
                output.print("[grey70]Preview:[/grey70]", style="info")
                obs_preview = _format_dataframe_preview(adata.obs)
                output.print_table(obs_preview)

        # ====================================================================
        # Variables (var)
        # ====================================================================
        if not adata.var.empty:
            output.print(
                f"\n[bold white]🧪 Variables (var) - {adata.n_vars:,} features[/bold white]",
                style="info"
            )

            # Column information
            var_info = []
            for col in adata.var.columns:
                dtype = str(adata.var[col].dtype)
                var_info.append(f"{col} ({dtype})")

            output.print(
                f"[grey70]Columns ({len(adata.var.columns)}):[/grey70] {', '.join(var_info[:5])}",
                style="info"
            )
            if len(var_info) > 5:
                output.print(
                    f"[grey50]... and {len(var_info) - 5} more columns[/grey50]",
                    style="info"
                )

            # Preview table
            if len(adata.var) > 0:
                output.print("[grey70]Preview:[/grey70]", style="info")
                var_preview = _format_dataframe_preview(adata.var)
                output.print_table(var_preview)

        # ====================================================================
        # Additional Data Structures
        # ====================================================================
        output.print(
            "\n[bold white]📦 Additional Data Structures[/bold white]",
            style="info"
        )

        # Layers
        if adata.layers:
            output.print(f"\n[cyan]Layers ({len(adata.layers)}):[/cyan]", style="info")
            for layer_name, layer_data in adata.layers.items():
                layer_info = _get_matrix_info(layer_data)
                output.print(
                    f"  • {layer_name}: {layer_info['shape'][0]}×{layer_info['shape'][1]} {layer_info['dtype']}",
                    style="info"
                )

        # Obsm (observation matrices)
        if adata.obsm:
            output.print("\n[cyan]Observation Matrices (obsm):[/cyan]", style="info")
            obsm_table = _format_array_info(dict(adata.obsm))
            if obsm_table:
                output.print_table(obsm_table)

        # Varm (variable matrices)
        if adata.varm:
            output.print("\n[cyan]Variable Matrices (varm):[/cyan]", style="info")
            varm_table = _format_array_info(dict(adata.varm))
            if varm_table:
                output.print_table(varm_table)

        # Obsp (observation pairwise)
        if adata.obsp:
            output.print("\n[cyan]Observation Pairwise (obsp):[/cyan]", style="info")
            for key in adata.obsp.keys():
                matrix = adata.obsp[key]
                output.print(
                    f"  • {key}: {matrix.shape[0]}×{matrix.shape[1]}",
                    style="info"
                )

        # Varp (variable pairwise)
        if adata.varp:
            output.print("\n[cyan]Variable Pairwise (varp):[/cyan]", style="info")
            for key in adata.varp.keys():
                matrix = adata.varp[key]
                output.print(
                    f"  • {key}: {matrix.shape[0]}×{matrix.shape[1]}",
                    style="info"
                )

        # Unstructured data (uns)
        if adata.uns:
            output.print("\n[cyan]Unstructured Data (uns):[/cyan]", style="info")
            uns_items = []
            for key, value in adata.uns.items():
                if isinstance(value, dict):
                    uns_items.append(f"{key} (dict with {len(value)} keys)")
                elif isinstance(value, (list, tuple)):
                    uns_items.append(f"{key} (list/tuple with {len(value)} items)")
                elif isinstance(value, np.ndarray):
                    uns_items.append(f"{key} (array {value.shape})")
                elif isinstance(value, pd.DataFrame):
                    uns_items.append(f"{key} (DataFrame {value.shape})")
                else:
                    type_name = type(value).__name__
                    uns_items.append(f"{key} ({type_name})")

            for item in uns_items[:10]:
                output.print(f"  • {item}", style="info")
            if len(uns_items) > 10:
                output.print(
                    f"[grey50]  ... and {len(uns_items) - 10} more items[/grey50]",
                    style="info"
                )

        # ====================================================================
        # Metadata from DataManager
        # ====================================================================
        if (
            hasattr(client.data_manager, "metadata_store")
            and modality_name in client.data_manager.metadata_store
        ):
            metadata = client.data_manager.metadata_store[modality_name]
            output.print("\n[bold white]📋 Metadata[/bold white]", style="info")

            meta_table_data = {
                "title": None,
                "columns": [
                    {"name": "Property", "style": "grey70"},
                    {"name": "Value", "style": "white"},
                ],
                "rows": []
            }

            if "source" in metadata:
                meta_table_data["rows"].append(["Source", metadata["source"]])
            if "created_at" in metadata:
                meta_table_data["rows"].append(["Created", metadata["created_at"]])
            if "geo_accession" in metadata:
                meta_table_data["rows"].append(["GEO Accession", metadata["geo_accession"]])

            if meta_table_data["rows"]:
                output.print_table(meta_table_data)

        output.print("", style="info")
        return f"Described modality: {modality_name}"

    except Exception as e:
        output.print(f"[red]Error describing modality: {e}[/red]", style="error")
        return None
