"""
Shared metadata commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional
from datetime import datetime

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter


def metadata_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show metadata store contents and workspace metadata files.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    output.print("\n[bold red]📋 Metadata Information[/bold red]\n", style="info")

    entries_shown = 0

    # ========================================================================
    # Section 1: Metadata Store (Cached GEO/External Data)
    # ========================================================================
    if hasattr(client.data_manager, "metadata_store"):
        metadata_store = client.data_manager.metadata_store
        if metadata_store:
            output.print(
                "[bold white]🗄️  Metadata Store (Cached GEO/External Data):[/bold white]",
                style="info"
            )

            store_table_data = {
                "title": "🗄️ Metadata Store",
                "columns": [
                    {"name": "Dataset ID", "style": "bold white"},
                    {"name": "Type", "style": "cyan"},
                    {"name": "Title", "style": "white", "max_width": 40, "overflow": "ellipsis"},
                    {"name": "Samples", "style": "grey74"},
                    {"name": "Cached", "style": "grey50"},
                ],
                "rows": []
            }

            for dataset_id, metadata_info in metadata_store.items():
                metadata = metadata_info.get("metadata", {})
                validation = metadata_info.get("validation", {})

                # Extract key information
                title = str(metadata.get("title", "N/A"))
                if len(title) > 40:
                    title = title[:40] + "..."

                data_type = (
                    validation.get("predicted_data_type", "unknown")
                    .replace("_", " ")
                    .title()
                )

                samples = (
                    len(metadata.get("samples", {}))
                    if metadata.get("samples")
                    else "N/A"
                )

                # Parse timestamp
                timestamp = metadata_info.get("fetch_timestamp", "")
                try:
                    cached_time = datetime.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    )
                    cached_str = cached_time.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    cached_str = timestamp[:16] if timestamp else "N/A"

                store_table_data["rows"].append([
                    dataset_id,
                    data_type,
                    title,
                    str(samples),
                    cached_str
                ])

            output.print_table(store_table_data)
            output.print("")  # Blank line
            entries_shown += len(metadata_store)
        else:
            output.print("[grey50]No cached metadata in metadata store[/grey50]\n", style="info")

    # ========================================================================
    # Section 2: Current Data Metadata
    # ========================================================================
    if (
        hasattr(client.data_manager, "current_metadata")
        and client.data_manager.current_metadata
    ):
        output.print("[bold white]📊 Current Data Metadata:[/bold white]", style="info")
        metadata = client.data_manager.current_metadata

        metadata_table_data = {
            "title": None,
            "columns": [
                {"name": "Key", "style": "bold grey93", "width": 25},
                {"name": "Value", "style": "white", "width": 50, "overflow": "fold"},
            ],
            "rows": []
        }

        for key, value in metadata.items():
            # Format value for display
            if isinstance(value, dict):
                display_value = f"Dict with {len(value)} keys: {', '.join(list(value.keys())[:3])}"
                if len(value) > 3:
                    display_value += f" ... (+{len(value)-3} more)"
            elif isinstance(value, list):
                display_value = f"List with {len(value)} items"
                if len(value) > 0:
                    display_value += f": {', '.join(str(v) for v in value[:3])}"
                    if len(value) > 3:
                        display_value += f" ... (+{len(value)-3} more)"
            else:
                display_value = str(value)
                if len(display_value) > 60:
                    display_value = display_value[:60] + "..."

            metadata_table_data["rows"].append([key, display_value])

        output.print_table(metadata_table_data)
        entries_shown += len(metadata)
    else:
        output.print("[grey50]No current data metadata available[/grey50]", style="info")

    # ========================================================================
    # Section 3: Workspace Metadata Files
    # ========================================================================
    workspace_path = Path(client.data_manager.workspace_path)
    metadata_dir = workspace_path / "metadata"

    if metadata_dir.exists():
        json_files = sorted(metadata_dir.glob("*.json"))
        if json_files:
            output.print("\n[bold white]📁 Workspace Metadata Files:[/bold white]", style="info")

            files_table_data = {
                "title": None,
                "columns": [
                    {"name": "File", "style": "cyan", "width": 50, "overflow": "ellipsis"},
                    {"name": "Size", "style": "grey50", "width": 10},
                    {"name": "Modified", "style": "grey50", "width": 20},
                ],
                "rows": []
            }

            for json_file in json_files[:20]:  # Limit to 20 files
                stat = json_file.stat()
                size_kb = stat.st_size / 1024
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                files_table_data["rows"].append([
                    json_file.name,
                    f"{size_kb:.1f} KB",
                    modified
                ])

            output.print_table(files_table_data)

            if len(json_files) > 20:
                output.print(f"[grey50]... and {len(json_files) - 20} more files[/grey50]")
            output.print(f"[grey50]Path: {metadata_dir}[/grey50]")

    # ========================================================================
    # Section 4: Export Files
    # ========================================================================
    exports_dir = workspace_path / "exports"
    if exports_dir.exists():
        export_files = sorted(
            [f for f in exports_dir.iterdir() if f.is_file() and f.suffix in {".csv", ".tsv", ".xlsx"}],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        if export_files:
            output.print("\n[bold white]📤 Export Files:[/bold white]", style="info")

            exports_table_data = {
                "title": None,
                "columns": [
                    {"name": "File", "style": "green", "width": 50, "overflow": "ellipsis"},
                    {"name": "Size", "style": "grey50", "width": 10},
                    {"name": "Modified", "style": "grey50", "width": 20},
                ],
                "rows": []
            }

            for export_file in export_files[:15]:  # Limit to 15 files
                stat = export_file.stat()
                size_kb = stat.st_size / 1024
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                exports_table_data["rows"].append([
                    export_file.name,
                    f"{size_kb:.1f} KB",
                    modified
                ])

            output.print_table(exports_table_data)

            if len(export_files) > 15:
                output.print(f"[grey50]... and {len(export_files) - 15} more files[/grey50]")
            output.print(f"[grey50]Path: {exports_dir}[/grey50]")

    # ========================================================================
    # Section 5: Deprecated Export Location Warning
    # ========================================================================
    old_exports_dir = workspace_path / "metadata" / "exports"
    if old_exports_dir.exists():
        old_files = list(old_exports_dir.glob("*"))
        if old_files:
            output.print("\n[bold yellow]⚠️  Deprecated Export Location Detected[/bold yellow]", style="warning")
            output.print(f"[yellow]Found {len(old_files)} file(s) in old location: {old_exports_dir}[/yellow]")
            output.print("[yellow]New exports go to: workspace/exports/[/yellow]")
            output.print("[grey50]Migration: mv workspace/metadata/exports/* workspace/exports/[/grey50]")

    # Return summary
    if entries_shown > 0:
        return f"Displayed metadata information ({entries_shown} entries)"
    else:
        return "No metadata available"


def metadata_clear(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Clear metadata store.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    if not hasattr(client.data_manager, "metadata_store"):
        output.print("[yellow]⚠️  Metadata store not available[/yellow]", style="warning")
        return None

    metadata_store = client.data_manager.metadata_store
    num_entries = len(metadata_store)

    if num_entries == 0:
        output.print("[grey50]Metadata store is already empty[/grey50]", style="info")
        return "Metadata store already empty"

    # Confirm with user
    confirm = output.confirm(f"[yellow]Clear all {num_entries} metadata entries?[/yellow]")

    if confirm:
        metadata_store.clear()
        output.print(
            f"[green]✓ Cleared {num_entries} metadata entries from store[/green]",
            style="success"
        )
        return f"Cleared {num_entries} metadata entries"
    else:
        output.print("[cyan]Operation cancelled[/cyan]", style="info")
        return None
