"""
Shared visualization commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional
from datetime import datetime
import logging

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter
from lobster.utils import open_path

logger = logging.getLogger(__name__)


def export_data(
    client: "AgentClient",
    output: OutputAdapter,
    include_png: bool = True,
    force_resave: bool = False,
    console=None,
) -> Optional[str]:
    """
    Export session data and plots to workspace exports directory.

    Exports all modalities and generated plots to workspace/exports/.
    For local clients with data, creates a comprehensive data package.
    For cloud clients or sessions without data, exports session metadata.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        include_png: Whether to generate PNG versions of plots (default: True)
        force_resave: Force re-serialization of all modalities (default: False)
        console: Optional Rich console for progress display

    Returns:
        Summary string for conversation history, or None
    """
    try:
        # Check if this is a local client with detailed export capabilities
        if hasattr(client, "data_manager") and hasattr(
            client.data_manager, "create_data_package"
        ):
            # For local client, check what we're exporting
            data_manager = client.data_manager
            has_data = data_manager.has_data()
            has_plots = bool(data_manager.latest_plots)

            if has_data:
                modality_count = len(data_manager.modalities)
                plot_count = len(data_manager.latest_plots)

                # Build progress message
                items = []
                if modality_count:
                    items.append(f"{modality_count} dataset(s)")
                if plot_count:
                    items.append(f"{plot_count} plot(s)")

                # Use Rich progress if console is available
                if console is not None:
                    export_stats = {"reused": 0, "saved": 0, "plots": 0}

                    with console.status(
                        f"[bold cyan]Exporting {', '.join(items)}...[/bold cyan]",
                        spinner="dots",
                    ) as status:

                        def update_progress(msg: str):
                            # Parse message for statistics
                            if "Reused existing" in msg:
                                export_stats["reused"] += 1
                            elif "Saving modality" in msg:
                                export_stats["saved"] += 1
                            elif "Saving plot" in msg:
                                export_stats["plots"] += 1
                            status.update(f"[bold cyan]{msg}[/bold cyan]")

                        export_path = data_manager.create_data_package(
                            output_dir=str(data_manager.exports_dir),
                            progress_callback=update_progress,
                            include_png=include_png,
                            reuse_autosave=not force_resave,
                        )
                        export_path = Path(export_path)

                    # Show summary
                    output.print(
                        f"[green]✓ Export complete:[/green]",
                        style="success"
                    )
                    output.print(f"  [green]•[/green] {export_path.name}", style="info")
                    if export_stats["reused"] > 0:
                        output.print(
                            f"  [dim]⚡ Reused {export_stats['reused']} cached file(s)[/dim]",
                            style="info"
                        )
                else:
                    # Fallback: simple progress callback
                    output.print(
                        f"[yellow]Exporting {', '.join(items)}...[/yellow]",
                        style="info"
                    )

                    def simple_progress(msg: str):
                        logger.debug(f"Export progress: {msg}")

                    export_path = data_manager.create_data_package(
                        output_dir=str(data_manager.exports_dir),
                        progress_callback=simple_progress,
                        include_png=include_png,
                        reuse_autosave=not force_resave,
                    )
                    export_path = Path(export_path)
                    output.print(
                        f"[green]✓ Session exported to:[/green] [grey74]{export_path}[/grey74]",
                        style="success"
                    )
            else:
                # Fallback to regular export_session for non-data exports
                if has_plots:
                    plot_count = len(data_manager.latest_plots)
                    output.print(
                        f"[yellow]Exporting {plot_count} plot(s)...[/yellow]",
                        style="info"
                    )
                export_path = client.export_session()
                output.print(
                    f"[green]✓ Session exported to:[/green] [grey74]{export_path}[/grey74]",
                    style="success"
                )
        else:
            # For cloud client or other clients
            output.print("[yellow]Exporting session data and plots...[/yellow]", style="info")
            export_path = client.export_session()
            output.print(
                f"[green]✓ Session exported to:[/green] [grey74]{export_path}[/grey74]",
                style="success"
            )

        return f"Session exported to: {export_path}"

    except Exception as e:
        output.print(f"[red]Export failed: {e}[/red]", style="error")
        logger.exception("Export error")
        return f"Export failed: {e}"


def plots_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show list of generated plots with IDs, titles, sources, and timestamps.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    try:
        history, stats, _ = client.data_manager.plot_manager.get_plot_history()

        if history:
            output.print("\n[bold red]📊 Generated Plots[/bold red]\n", style="info")

            table_data = {
                "title": "🦞 Generated Plots",
                "columns": [
                    {"name": "ID", "style": "bold white"},
                    {"name": "Title", "style": "white"},
                    {"name": "Source", "style": "grey74"},
                    {"name": "Created", "style": "grey50"},
                ],
                "rows": []
            }

            for plot in history:
                try:
                    created = datetime.fromisoformat(
                        plot["timestamp"].replace("Z", "+00:00")
                    )
                    created_str = created.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    created_str = plot["timestamp"][:16] if plot["timestamp"] else "N/A"

                table_data["rows"].append([
                    plot["id"],
                    plot["title"],
                    plot["source"] or "N/A",
                    created_str
                ])

            output.print_table(table_data)

            # Add usage hint
            output.print(
                "\n[grey50]Tip: Use /plot <ID> to open a specific plot[/grey50]",
                style="info"
            )

            return f"Displayed {len(history)} plot(s)"
        else:
            output.print("[grey50]No plots generated yet[/grey50]", style="info")
            return "No plots available"

    except Exception as e:
        output.print(f"[red]Failed to retrieve plot history: {e}[/red]", style="error")
        logger.exception("Plot list error")
        return None


def plot_show(
    client: "AgentClient",
    output: OutputAdapter,
    plot_identifier: Optional[str] = None
) -> Optional[str]:
    """
    Show or open a specific plot by ID or title match.

    If no plot_identifier is provided, opens the plots directory in file manager.
    If plot_identifier is provided, searches for plot by:
      1. Exact ID match
      2. Partial title match (case-insensitive)

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        plot_identifier: Plot ID or partial title to search for. None opens plots directory.

    Returns:
        Summary string for conversation history, or None
    """
    try:
        plots_dir = client.data_manager.workspace_path / "plots"

        # Case 1: No identifier - open plots directory
        if not plot_identifier:
            # Ensure plots directory exists
            if not plots_dir.exists():
                plots_dir.mkdir(parents=True, exist_ok=True)
                # Save any existing plots to the directory
                if client.data_manager.latest_plots:
                    saved_files, _, _ = client.data_manager.plot_manager.save_plots_to_workspace()
                    if saved_files:
                        output.print(
                            f"[green]✓ Saved {len(saved_files)} plot file(s) to workspace[/green]",
                            style="success"
                        )

            # Open the directory in file manager
            success, message = open_path(plots_dir)

            if success:
                output.print(f"[green]✓ {message}[/green]", style="success")
                return f"Opened plots directory: {plots_dir}"
            else:
                output.print(f"[red]Failed to open plots directory: {message}[/red]", style="error")
                output.print(f"[grey50]Plots directory: {plots_dir}[/grey50]", style="info")
                return f"Failed to open plots directory: {message}"

        # Case 2: Identifier provided - find and open specific plot
        found_plot = None
        plot_info = None

        for plot_entry in client.data_manager.latest_plots:
            # Check ID match
            if plot_entry["id"] == plot_identifier:
                found_plot = plot_entry
                plot_info = plot_entry
                break
            # Check title match (case-insensitive partial match)
            elif (
                plot_identifier.lower() in plot_entry["title"].lower()
                or plot_identifier.lower() in plot_entry["original_title"].lower()
            ):
                found_plot = plot_entry
                plot_info = plot_entry
                break

        if found_plot:
            # Ensure plots are saved to workspace
            if not plots_dir.exists():
                plots_dir.mkdir(parents=True, exist_ok=True)

            # Save plots and get actual file paths
            if client.data_manager.latest_plots:
                saved_files, _, _ = client.data_manager.plot_manager.save_plots_to_workspace()
            else:
                saved_files = []

            # Find the saved file for this plot using actual paths from save operation
            plot_id = plot_info["id"]
            matching_file = None

            for file_path in saved_files:
                file_path_obj = Path(file_path)
                # Check if filename starts with plot_id (handles both HTML and PNG)
                if file_path_obj.parent == plots_dir and file_path_obj.stem.startswith(plot_id):
                    matching_file = file_path_obj
                    # Prefer HTML over PNG
                    if file_path_obj.suffix == ".html":
                        break

            if matching_file and matching_file.exists():
                # Open plot using centralized system utility
                success, message = open_path(matching_file)

                if success:
                    output.print(
                        f"[green]✓ Opened plot:[/green] [grey74]{plot_info['original_title']}[/grey74]",
                        style="success"
                    )
                    return f"Opened plot: {plot_info['original_title']}"
                else:
                    output.print(
                        f"[red]Failed to open plot: {message}[/red]",
                        style="error"
                    )
                    output.print(
                        f"[grey50]Plot file: {matching_file}[/grey50]",
                        style="info"
                    )
                    return f"Failed to open plot: {message}"
            else:
                output.print(
                    "[red]Plot file not found. Try running /save first.[/red]",
                    style="error"
                )
                return "Plot file not found"
        else:
            output.print(
                f"[red]Plot not found: {plot_identifier}[/red]",
                style="error"
            )
            output.print(
                "[grey50]Use /plots to see available plot IDs and titles[/grey50]",
                style="info"
            )
            return f"Plot not found: {plot_identifier}"

    except Exception as e:
        output.print(f"[red]Failed to show plot: {e}[/red]", style="error")
        logger.exception("Plot show error")
        return None
