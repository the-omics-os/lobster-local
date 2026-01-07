"""
Shared queue commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional
import shutil

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter


class QueueFileTypeNotSupported(Exception):
    """Raised when unsupported file type is provided to queue load."""
    pass


def show_queue_status(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Display status of download and publication queues.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    if client.publication_queue is None:
        output.print("[yellow]Publication queue not initialized[/yellow]", style="warning")
        return None

    stats = client.publication_queue.get_statistics()

    output.print("\n[bold cyan]📋 Queue Status[/bold cyan]\n", style="info")

    # Create status table
    table_data = {
        "title": None,
        "columns": [
            {"name": "Status", "style": "cyan"},
            {"name": "Count", "style": "white", "justify": "right"},
        ],
        "rows": [],
    }

    by_status = stats.get("by_status", {})
    total_entries = stats.get("total_entries", 0)

    # Build rows dynamically from all statuses (shows all 8 PublicationStatus values)
    rows = []
    for status_name, count in by_status.items():
        # Capitalize for display
        display_name = status_name.replace("_", " ").title()
        rows.append([display_name, str(count)])

    # Add total row
    rows.append(["[bold]Total[/bold]", f"[bold]{total_entries}[/bold]"])

    table_data["rows"] = rows

    output.print_table(table_data)

    output.print("\n[cyan]💡 Commands:[/cyan]", style="info")
    output.print("  • [white]/queue load <file>[/white] - Load file into queue")
    output.print("  • [white]/queue list[/white] - List queued items")
    output.print("  • [white]/queue clear[/white] - Clear publication queue")
    output.print("  • [white]/queue clear download[/white] - Clear download queue")
    output.print("  • [white]/queue clear all[/white] - Clear all queues")
    output.print("  • [white]/queue export[/white] - Export queue to workspace")

    return f"Queue status: {total_entries} total items"


def queue_load_file(
    client: "AgentClient",
    filename: str,
    output: OutputAdapter,
    current_directory: Optional[Path] = None,
) -> Optional[str]:
    """
    Load file into queue - type determines handler.

    Args:
        client: AgentClient instance
        filename: File path to load
        output: OutputAdapter for rendering
        current_directory: Current working directory (optional for dashboard)

    Returns:
        Summary string for conversation history, or None

    Raises:
        QueueFileTypeNotSupported: For unsupported file types
    """
    if not filename:
        output.print("[yellow]Usage: /queue load <file>[/yellow]", style="warning")
        return None

    # Resolve file path
    if current_directory:
        # CLI mode: use PathResolver for secure resolution
        from lobster.cli_internal.utils.path_resolution import PathResolver

        resolver = PathResolver(
            current_directory=current_directory,
            workspace_path=(
                client.data_manager.workspace_path
                if hasattr(client, "data_manager")
                else None
            ),
        )
        resolved = resolver.resolve(filename, search_workspace=True, must_exist=True)

        if not resolved.is_safe:
            output.print(f"[red]❌ Security error: {resolved.error}[/red]", style="error")
            return None

        if not resolved.exists:
            output.print(f"[red]❌ File not found: {filename}[/red]", style="error")
            return None

        file_path = resolved.path
    else:
        # Dashboard mode: treat as absolute or workspace-relative path
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = client.data_manager.workspace_path / filename

        if not file_path.exists():
            output.print(f"[red]❌ File not found: {filename}[/red]", style="error")
            return None

    ext = file_path.suffix.lower()

    # Supported: .ris files
    if ext in [".ris", ".txt"]:
        output.print(f"[cyan]📚 Loading into queue: {file_path.name}[/cyan]\n", style="info")

        try:
            result = client.load_publication_list(
                file_path=str(file_path),
                priority=5,
                schema_type="general",
                extraction_level="methods",
            )

            if result["added_count"] > 0:
                output.print(
                    f"[green]✅ Loaded {result['added_count']} items into queue[/green]\n",
                    style="success",
                )

                if result["skipped_count"] > 0:
                    output.print(
                        f"[yellow]⚠️  Skipped {result['skipped_count']} malformed entries[/yellow]",
                        style="warning",
                    )

                output.print(
                    "\n[bold cyan]What would you like to do with these publications?[/bold cyan]"
                )
                output.print("  • Extract methods and parameters")
                output.print("  • Search for related datasets (GEO)")
                output.print("  • Build citation network")
                output.print("  • Custom analysis (describe your intent)\n")

                return f"Loaded {result['added_count']} publications into queue from {file_path.name}. Awaiting user intent."
            else:
                output.print("[red]❌ No items could be loaded from file[/red]", style="error")
                if result.get("errors"):
                    for error in result["errors"][:3]:
                        output.print(f"  • {error}")
                return None

        except Exception as e:
            output.print(f"[red]❌ Failed to load file: {str(e)}[/red]", style="error")
            return None

    # Placeholder: .bib files (BibTeX)
    elif ext == ".bib":
        raise QueueFileTypeNotSupported(
            "BibTeX (.bib) support coming soon. "
            "Convert to .ris format or wait for future release."
        )

    # Placeholder: .csv files (custom lists)
    elif ext == ".csv":
        raise QueueFileTypeNotSupported(
            "CSV queue loading coming soon. "
            "Expected format: columns for DOI, PMID, or title."
        )

    # Placeholder: .json files (API exports)
    elif ext == ".json":
        raise QueueFileTypeNotSupported(
            "JSON queue loading coming soon. Planned support for PubMed API exports."
        )

    # Unknown type
    else:
        raise QueueFileTypeNotSupported(
            f"Unsupported file type: {ext}. "
            f"Currently supported: .ris. Coming soon: .bib, .csv, .json"
        )


def queue_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    List items in the publication queue.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    if client.publication_queue is None:
        output.print("[yellow]Publication queue not initialized[/yellow]", style="warning")
        return None

    entries = client.publication_queue.list_entries()

    if not entries:
        output.print("[yellow]Queue is empty[/yellow]", style="warning")
        return "Queue is empty"

    # Limit display to first 20 entries
    display_entries = entries[:20]
    total_count = len(entries)

    output.print(
        f"\n[bold cyan]📋 Queue Items ({len(display_entries)} of {total_count} shown)[/bold cyan]\n",
        style="info",
    )

    table_data = {
        "title": None,
        "columns": [
            {"name": "#", "style": "dim", "width": 4},
            {"name": "Title", "style": "white", "max_width": 50, "overflow": "ellipsis"},
            {"name": "Year", "style": "cyan", "width": 6},
            {"name": "Status", "style": "yellow", "width": 12},
            {"name": "PMID/DOI", "style": "dim", "width": 20},
        ],
        "rows": [],
    }

    for i, entry in enumerate(display_entries, 1):
        title = (
            entry.title[:47] + "..."
            if entry.title and len(entry.title) > 50
            else (entry.title or "N/A")
        )
        year = str(entry.year) if entry.year else "N/A"
        status = (
            entry.status.value if hasattr(entry.status, "value") else str(entry.status)
        )
        identifier = entry.pmid or entry.doi or "N/A"

        table_data["rows"].append([str(i), title, year, status, identifier])

    output.print_table(table_data)

    if total_count > 20:
        output.print(f"\n[dim]... and {total_count - 20} more items[/dim]")

    return f"Listed {len(display_entries)} of {total_count} items from queue"


def queue_clear(
    client: "AgentClient", output: OutputAdapter, queue_type: str = "publication"
) -> Optional[str]:
    """
    Clear items from specified queue(s).

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        queue_type: "publication", "download", or "all"

    Returns:
        Summary string for conversation history, or None
    """
    if queue_type == "all":
        # Clear both queues
        pub_total = 0
        dl_total = 0

        # Get publication queue stats
        if client.publication_queue is not None:
            pub_stats = client.publication_queue.get_statistics()
            pub_total = pub_stats.get("total_entries", 0)

        # Get download queue stats
        if hasattr(client, "data_manager") and client.data_manager:
            dl_stats = client.data_manager.download_queue.get_statistics()
            dl_total = dl_stats.get("total_entries", 0)

        grand_total = pub_total + dl_total

        if grand_total == 0:
            output.print("[yellow]All queues are already empty[/yellow]", style="warning")
            return "All queues were already empty"

        # Confirm with user
        output.print(f"[yellow]About to clear:[/yellow]", style="warning")
        output.print(f"  • Publication queue: {pub_total} items")
        output.print(f"  • Download queue: {dl_total} items")
        output.print(f"  • Total: {grand_total} items\n")
        confirm = output.confirm(f"[yellow]Clear all {grand_total} items from both queues?[/yellow]")

        if confirm:
            cleared = []
            if pub_total > 0:
                client.publication_queue.clear_queue()
                cleared.append(f"publication ({pub_total})")
            if dl_total > 0:
                client.data_manager.download_queue.clear_queue()
                cleared.append(f"download ({dl_total})")

            output.print(
                f"[green]✅ Cleared {grand_total} items from {', '.join(cleared)} queues[/green]",
                style="success",
            )
            return f"Cleared {grand_total} items from all queues"
        else:
            output.print("[cyan]Operation cancelled[/cyan]", style="info")
            return None

    elif queue_type == "download":
        # Clear download queue
        if not hasattr(client, "data_manager") or not client.data_manager:
            output.print("[yellow]Data manager not initialized[/yellow]", style="warning")
            return None

        download_queue = client.data_manager.download_queue
        stats = download_queue.get_statistics()
        total = stats.get("total_entries", 0)

        if total == 0:
            output.print("[yellow]Download queue is already empty[/yellow]", style="warning")
            return "Download queue was already empty"

        # Confirm with user
        confirm = output.confirm(f"[yellow]Clear all {total} items from download queue?[/yellow]")

        if confirm:
            download_queue.clear_queue()
            output.print(f"[green]✅ Cleared {total} items from download queue[/green]", style="success")
            return f"Cleared {total} items from download queue"
        else:
            output.print("[cyan]Operation cancelled[/cyan]", style="info")
            return None

    else:  # publication (default)
        if client.publication_queue is None:
            output.print("[yellow]Publication queue not initialized[/yellow]", style="warning")
            return None

        # Get count before clearing
        stats = client.publication_queue.get_statistics()
        total = stats.get("total_entries", 0)

        if total == 0:
            output.print("[yellow]Publication queue is already empty[/yellow]", style="warning")
            return "Publication queue was already empty"

        # Confirm with user
        confirm = output.confirm(f"[yellow]Clear all {total} items from publication queue?[/yellow]")

        if confirm:
            client.publication_queue.clear_queue()
            output.print(f"[green]✅ Cleared {total} items from publication queue[/green]", style="success")
            return f"Cleared {total} items from publication queue"
        else:
            output.print("[cyan]Operation cancelled[/cyan]", style="info")
            return None


def queue_export(
    client: "AgentClient", name: Optional[str], output: OutputAdapter
) -> Optional[str]:
    """
    Export queue to workspace for persistence.

    Args:
        client: AgentClient instance
        name: Optional name for the exported dataset
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    if client.publication_queue is None:
        output.print("[yellow]Publication queue not initialized[/yellow]", style="warning")
        return None

    stats = client.publication_queue.get_statistics()
    if stats.get("total_entries", 0) == 0:
        output.print("[yellow]Queue is empty, nothing to export[/yellow]", style="warning")
        return None

    # Generate export name if not provided
    if not name:
        from datetime import datetime

        name = f"queue_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output.print(f"[cyan]📦 Exporting queue to workspace as '{name}'...[/cyan]", style="info")

    try:
        # Export queue data by copying the queue file to workspace
        source_path = client.publication_queue.queue_file
        export_path = client.data_manager.workspace_path / f"{name}.jsonl"

        # Copy the queue file
        shutil.copy2(source_path, export_path)

        output.print(
            f"[green]✅ Exported {stats.get('total_entries', 0)} items to: {export_path}[/green]",
            style="success",
        )
        return f"Exported {stats.get('total_entries', 0)} queue items to workspace as '{name}'"
    except Exception as e:
        output.print(f"[red]❌ Export failed: {str(e)}[/red]", style="error")
        return None
