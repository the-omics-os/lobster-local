"""
Shared data commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter


def data_summary(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show current data summary with modalities table.

    Displays loaded datasets with shape, type, memory usage, and metadata.
    Handles both single modality and multiple modalities cases.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None if no data loaded
    """
    if not client.data_manager.has_data():
        output.print("[grey50]No data currently loaded[/grey50]", style="info")
        return None

    summary = client.data_manager.get_data_summary()

    # ========================================================================
    # Section 1: Main Data Summary Table
    # ========================================================================
    main_table_data = {
        "title": "🦞 Current Data Summary",
        "columns": [
            {"name": "Property", "style": "bold grey93"},
            {"name": "Value", "style": "white"},
        ],
        "rows": []
    }

    # Status row
    main_table_data["rows"].append(["Status", summary["status"]])

    # Handle shape - might be single modality or total for multiple modalities
    if "shape" in summary:
        main_table_data["rows"].append([
            "Shape",
            f"{summary['shape'][0]} × {summary['shape'][1]}"
        ])
    elif "total_obs" in summary and "total_vars" in summary:
        main_table_data["rows"].append([
            "Total Shape",
            f"{summary['total_obs']} × {summary['total_vars']}"
        ])

    # Memory usage
    if "memory_usage" in summary:
        main_table_data["rows"].append(["Memory Usage", summary["memory_usage"]])

    # Modality name (single modality case)
    if summary.get("modality_name"):
        main_table_data["rows"].append(["Modality", summary["modality_name"]])

    # Data type
    if summary.get("data_type"):
        main_table_data["rows"].append(["Data Type", summary["data_type"]])

    # Matrix type (sparse vs dense)
    if summary.get("is_sparse") is not None:
        sparse_status = "✓ Sparse" if summary["is_sparse"] else "✗ Dense"
        main_table_data["rows"].append(["Matrix Type", sparse_status])

    # Observation columns (prefer 'columns' key, fallback to 'obs_columns')
    obs_cols = summary.get("columns") or summary.get("obs_columns", [])
    if obs_cols:
        cols_preview = ", ".join(obs_cols[:5])
        if len(obs_cols) > 5:
            cols_preview += f" ... (+{len(obs_cols)-5} more)"
        main_table_data["rows"].append(["Obs Columns", cols_preview])

    # Variable columns
    if summary.get("var_columns"):
        var_cols = summary["var_columns"]
        var_preview = ", ".join(var_cols[:5])
        if len(var_cols) > 5:
            var_preview += f" ... (+{len(var_cols)-5} more)"
        main_table_data["rows"].append(["Var Columns", var_preview])

    # Sample names (prefer 'sample_names' key, fallback to 'obs_names')
    sample_names = summary.get("sample_names") or summary.get("obs_names", [])
    if sample_names:
        samples_preview = ", ".join(sample_names[:3])
        if len(sample_names) > 3:
            samples_preview += f" ... (+{len(sample_names)-3} more)"
        main_table_data["rows"].append(["Samples", samples_preview])

    # Layers
    if summary.get("layers"):
        layers_str = ", ".join(summary["layers"])
        main_table_data["rows"].append(["Layers", layers_str])

    # Obsm keys
    if summary.get("obsm"):
        obsm_str = ", ".join(summary["obsm"])
        main_table_data["rows"].append(["Obsm Keys", obsm_str])

    # Metadata keys
    if summary.get("metadata_keys"):
        meta_preview = ", ".join(summary["metadata_keys"][:3])
        if len(summary["metadata_keys"]) > 3:
            meta_preview += f" ... (+{len(summary['metadata_keys'])-3} more)"
        main_table_data["rows"].append(["Metadata Keys", meta_preview])

    # Processing log (recent steps)
    if summary.get("processing_log"):
        recent_steps = (
            summary["processing_log"][-2:]
            if len(summary["processing_log"]) > 2
            else summary["processing_log"]
        )
        main_table_data["rows"].append(["Recent Steps", "; ".join(recent_steps)])

    output.print_table(main_table_data)

    # ========================================================================
    # Section 2: Individual Modality Details (if multiple modalities)
    # ========================================================================
    if summary.get("modalities"):
        output.print("\n[bold red]🧬 Individual Modality Details[/bold red]\n", style="info")

        modalities_table_data = {
            "title": None,
            "columns": [
                {"name": "Modality", "style": "bold white"},
                {"name": "Shape", "style": "white"},
                {"name": "Type", "style": "cyan"},
                {"name": "Memory", "style": "grey74"},
                {"name": "Sparse", "style": "grey50"},
            ],
            "rows": []
        }

        for mod_name, mod_info in summary["modalities"].items():
            if isinstance(mod_info, dict) and not mod_info.get("error"):
                shape_str = f"{mod_info['shape'][0]} × {mod_info['shape'][1]}"
                data_type = mod_info.get("data_type", "unknown")
                memory = mod_info.get("memory_usage", "N/A")
                sparse = "✓" if mod_info.get("is_sparse") else "✗"

                modalities_table_data["rows"].append([
                    mod_name,
                    shape_str,
                    data_type,
                    memory,
                    sparse
                ])
            else:
                # Handle error case
                error_msg = (
                    mod_info.get("error", "Unknown error")
                    if isinstance(mod_info, dict)
                    else "Invalid data"
                )
                modalities_table_data["rows"].append([
                    mod_name,
                    "Error",
                    error_msg[:20] + "..." if len(error_msg) > 20 else error_msg,
                    "N/A",
                    "N/A"
                ])

        output.print_table(modalities_table_data)

    # ========================================================================
    # Section 3: Detailed Metadata (if available)
    # ========================================================================
    if (
        hasattr(client.data_manager, "current_metadata")
        and client.data_manager.current_metadata
    ):
        metadata = client.data_manager.current_metadata
        output.print("\n[bold red]📋 Detailed Metadata:[/bold red]\n", style="info")

        metadata_table_data = {
            "title": None,
            "columns": [
                {"name": "Key", "style": "bold grey93"},
                {"name": "Value", "style": "white"},
            ],
            "rows": []
        }

        # Show first 10 items
        for key, value in list(metadata.items())[:10]:
            # Format value for display
            if isinstance(value, (list, dict)):
                display_value = (
                    str(value)[:50] + "..."
                    if len(str(value)) > 50
                    else str(value)
                )
            else:
                display_value = (
                    str(value)[:50] + "..."
                    if len(str(value)) > 50
                    else str(value)
                )
            metadata_table_data["rows"].append([key, display_value])

        # Add "more items" indicator if needed
        if len(metadata) > 10:
            metadata_table_data["rows"].append(["...", f"(+{len(metadata)-10} more items)"])

        output.print_table(metadata_table_data)

    # Return summary
    if summary.get("modalities"):
        num_modalities = len(summary["modalities"])
        return f"Displayed data summary ({num_modalities} modalities)"
    else:
        return "Displayed data summary"
