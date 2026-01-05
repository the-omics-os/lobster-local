"""
Shared pipeline commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional
import logging

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter

logger = logging.getLogger(__name__)


def pipeline_export(
    client: "AgentClient",
    output: OutputAdapter,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Optional[str]:
    """
    Export current session as Jupyter notebook.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        name: Notebook name (without extension). If None, prompts interactively.
        description: Optional notebook description. If None, prompts interactively.

    Returns:
        Summary string for conversation history, or None
    """
    try:
        # Check if data manager supports notebook export
        if not hasattr(client, "data_manager"):
            output.print(
                "[red]Notebook export not available for cloud client[/red]",
                style="error"
            )
            return "Notebook export only available for local client"

        if not hasattr(client.data_manager, "export_notebook"):
            output.print(
                "[red]Notebook export not available - update Lobster[/red]",
                style="error"
            )
            return "Notebook export not available"

        # Interactive prompts if not provided
        output.print(
            "[bold white]📓 Export Session as Jupyter Notebook[/bold white]\n",
            style="info"
        )

        if name is None:
            name = output.prompt("Notebook name (no extension)", default="analysis_workflow")
        if not name:
            output.print("[red]Name required[/red]", style="error")
            return "Export cancelled - no name provided"

        if description is None:
            description = output.prompt("Description (optional)", default="")

        # Export via DataManagerV2
        output.print("\n[yellow]Exporting notebook...[/yellow]", style="info")
        path = client.data_manager.export_notebook(name, description)

        output.print(f"\n[green]✓ Notebook exported:[/green] {path}", style="success")
        output.print("\n[bold white]Next steps:[/bold white]", style="info")
        output.print(f"  1. [yellow]Review:[/yellow]  jupyter notebook {path}", style="info")
        output.print(
            f"  2. [yellow]Commit:[/yellow]  git add {path} && git commit -m 'Add {name}'",
            style="info"
        )
        output.print(
            f"  3. [yellow]Run:[/yellow]     /pipeline run {path.name} <modality>",
            style="info"
        )

        return f"Exported notebook: {path}"

    except ValueError as e:
        output.print(f"[red]Export failed: {e}[/red]", style="error")
        return f"Export failed: {e}"
    except Exception as e:
        output.print(f"[red]Export error: {e}[/red]", style="error")
        logger.exception("Notebook export error")
        return f"Export error: {e}"


def pipeline_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    List available notebooks.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    try:
        if not hasattr(client, "data_manager"):
            output.print(
                "[red]Notebook listing not available for cloud client[/red]",
                style="error"
            )
            return "Notebook listing only available for local client"

        notebooks = client.data_manager.list_notebooks()

        if not notebooks:
            output.print(
                "[yellow]No notebooks found in .lobster/notebooks/[/yellow]",
                style="warning"
            )
            output.print("Export one with: [green]/pipeline export[/green]", style="info")
            return "No notebooks found"

        # Create table data
        table_data = {
            "title": "📓 Available Notebooks",
            "columns": [
                {"name": "Name", "style": "cyan"},
                {"name": "Steps", "justify": "right"},
                {"name": "Created By", "style": "white"},
                {"name": "Created", "style": "grey50"},
                {"name": "Size", "justify": "right", "style": "grey50"},
            ],
            "rows": []
        }

        for nb in notebooks:
            created_date = (
                nb["created_at"].split("T")[0] if nb["created_at"] else "unknown"
            )
            table_data["rows"].append([
                nb["name"],
                str(nb["n_steps"]),
                nb["created_by"],
                created_date,
                f"{nb['size_kb']:.1f} KB",
            ])

        output.print_table(table_data)
        return f"Found {len(notebooks)} notebooks"

    except Exception as e:
        output.print(f"[red]List error: {e}[/red]", style="error")
        logger.exception("Notebook list error")
        return f"List error: {e}"


def pipeline_run(
    client: "AgentClient",
    output: OutputAdapter,
    notebook_name: Optional[str] = None,
    input_modality: Optional[str] = None
) -> Optional[str]:
    """
    Run saved notebook with new data.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        notebook_name: Notebook filename. If None, prompts interactively.
        input_modality: Input modality name. If None, prompts interactively.

    Returns:
        Summary string for conversation history, or None
    """
    try:
        if not hasattr(client, "data_manager"):
            output.print(
                "[red]Notebook execution not available for cloud client[/red]",
                style="error"
            )
            return "Notebook execution only available for local client"

        # Get notebook name if not provided
        if notebook_name is None:
            notebooks = client.data_manager.list_notebooks()
            if not notebooks:
                output.print("[red]No notebooks available[/red]", style="error")
                return "No notebooks available"

            output.print("[bold]Available notebooks:[/bold]", style="info")
            for i, nb in enumerate(notebooks, 1):
                output.print(
                    f"  {i}. [cyan]{nb['name']}[/cyan] ({nb['n_steps']} steps)",
                    style="info"
                )

            selection = output.prompt("Select notebook number", default="1")
            try:
                idx = int(selection) - 1
                notebook_name = notebooks[idx]["filename"]
            except (ValueError, IndexError):
                output.print("[red]Invalid selection[/red]", style="error")
                return "Invalid notebook selection"

        # Get input modality if not provided
        if input_modality is None:
            modalities = client.data_manager.list_modalities()
            if not modalities:
                output.print("[red]No data loaded. Use /read first.[/red]", style="error")
                return "No data loaded"

            output.print("[bold]Available modalities:[/bold]", style="info")
            for i, mod in enumerate(modalities, 1):
                adata = client.data_manager.modalities[mod]
                output.print(
                    f"  {i}. [cyan]{mod}[/cyan] ({adata.n_obs} obs × {adata.n_vars} vars)",
                    style="info"
                )

            selection = output.prompt("Select modality number", default="1")
            try:
                idx = int(selection) - 1
                input_modality = modalities[idx]
            except (ValueError, IndexError):
                output.print("[red]Invalid selection[/red]", style="error")
                return "Invalid modality selection"

        # Dry run first
        output.print("\n[yellow]Running validation...[/yellow]", style="info")
        dry_result = client.data_manager.run_notebook(
            notebook_name, input_modality, dry_run=True
        )

        # Show validation
        validation = dry_result.get("validation")
        if (
            validation
            and hasattr(validation, "has_errors")
            and validation.has_errors
        ):
            output.print("[red]✗ Validation failed:[/red]", style="error")
            for error in validation.errors:
                output.print(f"  • {error}", style="error")
            return "Validation failed"

        if (
            validation
            and hasattr(validation, "has_warnings")
            and validation.has_warnings
        ):
            output.print("[yellow]⚠ Warnings:[/yellow]", style="warning")
            for warning in validation.warnings:
                output.print(f"  • {warning}", style="warning")

        output.print("\n[green]✓ Validation passed[/green]", style="success")
        output.print(f"  Steps to execute: {dry_result['steps_to_execute']}", style="info")
        output.print(
            f"  Estimated time: {dry_result['estimated_duration_minutes']} min",
            style="info"
        )

        # Confirm execution
        if not output.confirm("\nExecute notebook?"):
            output.print("Cancelled", style="info")
            return "Execution cancelled"

        # Execute
        output.print("\n[yellow]Executing notebook...[/yellow]", style="info")
        # Note: Progress handling is CLI-specific, skip for now
        result = client.data_manager.run_notebook(notebook_name, input_modality)

        if result["status"] == "success":
            output.print("\n[green]✓ Execution complete![/green]", style="success")
            output.print(f"  Output: {result['output_notebook']}", style="info")
            output.print(f"  Duration: {result['execution_time']:.1f}s", style="info")
            return (
                f"Notebook executed successfully in {result['execution_time']:.1f}s"
            )
        else:
            output.print("\n[red]✗ Execution failed[/red]", style="error")
            output.print(f"  Error: {result.get('error', 'Unknown')}", style="error")
            output.print(
                f"  Partial output: {result.get('output_notebook', 'N/A')}",
                style="error"
            )
            return f"Execution failed: {result.get('error', 'Unknown')}"

    except FileNotFoundError as e:
        output.print(f"[red]File not found: {e}[/red]", style="error")
        return f"Notebook not found: {e}"
    except Exception as e:
        output.print(f"[red]Execution error: {e}[/red]", style="error")
        logger.exception("Notebook execution error")
        return f"Execution error: {e}"


def pipeline_info(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show notebook details.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    try:
        if not hasattr(client, "data_manager"):
            output.print(
                "[red]Notebook info not available for cloud client[/red]",
                style="error"
            )
            return "Notebook info only available for local client"

        notebooks = client.data_manager.list_notebooks()
        if not notebooks:
            output.print("[red]No notebooks found[/red]", style="error")
            return "No notebooks found"

        output.print("[bold]Select notebook:[/bold]", style="info")
        for i, nb in enumerate(notebooks, 1):
            output.print(f"  {i}. [cyan]{nb['name']}[/cyan]", style="info")

        selection = output.prompt("Selection", default="1")
        try:
            idx = int(selection) - 1
            nb = notebooks[idx]
        except (ValueError, IndexError):
            output.print("[red]Invalid selection[/red]", style="error")
            return "Invalid selection"

        # Load full notebook
        import nbformat

        nb_path = Path(nb["path"])
        with open(nb_path) as f:
            notebook = nbformat.read(f, as_version=4)

        metadata = notebook.metadata.get("lobster", {})

        # Display info
        output.print(f"\n[bold cyan]{nb['name']}[/bold cyan]", style="info")
        output.print(f"Created by: {metadata.get('created_by', 'unknown')}", style="info")
        output.print(f"Date: {metadata.get('created_at', 'unknown')}", style="info")
        output.print(
            f"Lobster version: {metadata.get('lobster_version', 'unknown')}",
            style="info"
        )
        output.print("\nDependencies:", style="info")
        for pkg, ver in metadata.get("dependencies", {}).items():
            output.print(f"  {pkg}: {ver}", style="info")

        output.print(f"\nSteps: {nb['n_steps']}", style="info")
        output.print(f"Size: {nb['size_kb']:.1f} KB", style="info")

        return f"Notebook info: {nb['name']}"

    except Exception as e:
        output.print(f"[red]Info error: {e}[/red]", style="error")
        logger.exception("Notebook info error")
        return f"Info error: {e}"
