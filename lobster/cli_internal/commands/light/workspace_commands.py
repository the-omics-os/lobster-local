"""
Shared workspace commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional
from datetime import datetime
import fnmatch

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter


def truncate_middle(text: str, max_length: int = 60) -> str:
    """
    Truncate text in the middle with ellipsis, preserving start and end.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text with middle ellipsis
    """
    if len(text) <= max_length:
        return text

    # Calculate how much to keep on each side
    # Reserve 3 characters for "..."
    available_chars = max_length - 3
    start_length = (available_chars + 1) // 2  # Slightly prefer start
    end_length = available_chars // 2

    return f"{text[:start_length]}...{text[-end_length:]}"


def workspace_list(
    client: "AgentClient",
    output: OutputAdapter,
    force_refresh: bool = False
) -> Optional[str]:
    """
    List available datasets in workspace.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        force_refresh: Force refresh of workspace scan (default: False)

    Returns:
        Summary string for conversation history, or None
    """
    # BUG FIX #2: Use cached scan instead of explicit rescan (75% faster)
    if hasattr(client.data_manager, "get_available_datasets"):
        available = client.data_manager.get_available_datasets(
            force_refresh=force_refresh
        )
    else:
        # Fallback for older DataManager versions
        if hasattr(client.data_manager, "_scan_workspace"):
            client.data_manager._scan_workspace()
        available = client.data_manager.available_datasets

    loaded = set(client.data_manager.modalities.keys())

    if not available:
        # Handle empty case with helpful information
        workspace_path = client.data_manager.workspace_path
        data_dir = workspace_path / "data"

        output.print("[yellow]📂 No datasets found in workspace[/yellow]", style="warning")
        output.print(f"[grey70]Workspace: {workspace_path}[/grey70]", style="info")
        output.print(f"[grey70]Data directory: {data_dir}[/grey70]", style="info")

        if not data_dir.exists():
            output.print("[red]⚠️  Data directory doesn't exist[/red]", style="error")
            output.print(
                f"[cyan]💡 Create it with: mkdir -p {data_dir}[/cyan]",
                style="info"
            )
        else:
            # Check what files are actually in the data directory
            files = list(data_dir.glob("*"))
            if files:
                output.print(
                    f"[cyan]Found {len(files)} files in data directory, but none are supported datasets (.h5ad)[/cyan]",
                    style="info"
                )
                output.print("[grey70]Files found:[/grey70]", style="info")
                for f in files[:5]:  # Show first 5 files
                    output.print(f"  • {f.name}", style="info")
                if len(files) > 5:
                    output.print(f"  • ... and {len(files) - 5} more", style="info")
            else:
                output.print(
                    f"[cyan]💡 Add .h5ad files to {data_dir} to see them here[/cyan]",
                    style="info"
                )

        return "No datasets found in workspace"

    # Create table data structure
    table_data = {
        "title": "Available Datasets",
        "columns": [
            {"name": "#", "style": "dim", "width": 4},
            {"name": "Status", "style": "green", "width": 6},
            {"name": "Name", "style": "bold", "no_wrap": False},
            {"name": "Size", "style": "cyan", "width": 10},
            {"name": "Shape", "style": "white", "width": 15},
            {"name": "Modified", "style": "dim", "width": 12},
        ],
        "rows": []
    }

    for idx, (name, info) in enumerate(sorted(available.items()), start=1):
        status = "✓" if name in loaded else "○"
        size = f"{info['size_mb']:.1f} MB"
        shape = (
            f"{info['shape'][0]:,} × {info['shape'][1]:,}"
            if info["shape"]
            else "N/A"
        )
        modified = info["modified"].split("T")[0]

        # Use intelligent truncation for long names
        display_name = truncate_middle(name, max_length=60)

        table_data["rows"].append([str(idx), status, display_name, size, shape, modified])

    output.print_table(table_data)
    output.print("\n[dim]Use '/workspace info <#>' to see full details[/dim]", style="info")
    return f"Listed {len(available)} available datasets"


def workspace_info(
    client: "AgentClient",
    output: OutputAdapter,
    selector: str
) -> Optional[str]:
    """
    Show detailed information for specific dataset(s).

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        selector: Index (#) or pattern to match datasets

    Returns:
        Summary string for conversation history, or None
    """
    if not selector:
        output.print("[red]Usage: /workspace info <#|pattern>[/red]", style="error")
        output.print(
            "[dim]Examples: /workspace info 1, /workspace info gse12345, /workspace info *clustered*[/dim]",
            style="info"
        )
        return None

    # BUG FIX #2: Use cached scan for info command
    if hasattr(client.data_manager, "get_available_datasets"):
        available = client.data_manager.get_available_datasets(
            force_refresh=False
        )
    else:
        # Fallback for older DataManager versions
        if hasattr(client.data_manager, "_scan_workspace"):
            client.data_manager._scan_workspace()
        available = client.data_manager.available_datasets

    loaded = set(client.data_manager.modalities.keys())

    if not available:
        output.print("[yellow]No datasets found in workspace[/yellow]", style="warning")
        return None

    # Determine if selector is an index or pattern
    matched_datasets = []

    if selector.isdigit():
        # Index-based selection
        idx = int(selector)
        sorted_names = sorted(available.keys())
        if 1 <= idx <= len(sorted_names):
            matched_datasets = [
                (sorted_names[idx - 1], available[sorted_names[idx - 1]])
            ]
        else:
            output.print(
                f"[red]Index {idx} out of range (1-{len(sorted_names)})[/red]",
                style="error"
            )
            return None
    else:
        # Pattern-based selection
        for name, info in sorted(available.items()):
            if fnmatch.fnmatch(name.lower(), selector.lower()):
                matched_datasets.append((name, info))

        if not matched_datasets:
            output.print(
                f"[yellow]No datasets match pattern: {selector}[/yellow]",
                style="warning"
            )
            return None

    # Display detailed information for matched datasets
    for name, info in matched_datasets:
        status = "✓ Loaded" if name in loaded else "○ Not Loaded"

        # Create detailed info table
        detail_table_data = {
            "title": f"Dataset: {name}",
            "columns": [
                {"name": "Property", "style": "bold cyan"},
                {"name": "Value", "style": "white"},
            ],
            "show_header": False,
            "border_style": "cyan",
            "rows": []
        }

        detail_table_data["rows"].append(["Name", name])
        detail_table_data["rows"].append(["Status", status])
        detail_table_data["rows"].append(["Path", info["path"]])
        detail_table_data["rows"].append(["Size", f"{info['size_mb']:.2f} MB"])
        detail_table_data["rows"].append([
            "Shape",
            (
                f"{info['shape'][0]:,} observations × {info['shape'][1]:,} variables"
                if info["shape"]
                else "N/A"
            )
        ])
        detail_table_data["rows"].append(["Type", info["type"]])
        detail_table_data["rows"].append(["Modified", info["modified"]])

        # Try to detect lineage from name (basic version)
        if "_" in name:
            parts_list = name.split("_")
            possible_stages = [
                p
                for p in parts_list
                if any(
                    keyword in p.lower()
                    for keyword in [
                        "quality",
                        "filter",
                        "normal",
                        "doublet",
                        "cluster",
                        "marker",
                        "annot",
                        "pseudobulk",
                    ]
                )
            ]
            if possible_stages:
                detail_table_data["rows"].append([
                    "Processing Stages", " → ".join(possible_stages)
                ])

        output.print_table(detail_table_data)
        output.print("")  # Add spacing between datasets

    return f"Displayed details for {len(matched_datasets)} dataset(s)"


def workspace_load(
    client: "AgentClient",
    output: OutputAdapter,
    selector: str,
    current_directory: Path,
    path_resolver_class
) -> Optional[str]:
    """
    Load specific datasets by index, pattern, or file path.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        selector: Index (#), pattern, or file path
        current_directory: Current working directory for path resolution
        path_resolver_class: PathResolver class for secure path resolution

    Returns:
        Summary string for conversation history, or None
    """
    if not selector:
        output.print("[red]Usage: /workspace load <#|pattern|file>[/red]", style="error")
        output.print(
            "[dim]Examples: /workspace load 1, /workspace load recent, /workspace load data.h5ad[/dim]",
            style="info"
        )
        return None

    # BUG FIX #6: Use PathResolver for secure path resolution
    resolver = path_resolver_class(
        current_directory=current_directory,
        workspace_path=(
            client.data_manager.workspace_path
            if hasattr(client, "data_manager")
            else None
        ),
    )
    resolved = resolver.resolve(
        selector, search_workspace=True, must_exist=False
    )

    if not resolved.is_safe:
        output.print(f"[red]❌ Security error: {resolved.error}[/red]", style="error")
        return None

    file_path = resolved.path

    if file_path.exists() and file_path.is_file():
        # Load file directly into workspace
        output.print(
            f"[cyan]📂 Loading file into workspace: {file_path.name}[/cyan]\n",
            style="info"
        )

        try:
            result = client.load_data_file(str(file_path))

            if result.get("success"):
                output.print(
                    f"[green]✅ Loaded '{result['modality_name']}' "
                    f"({result['data_shape'][0]:,} × {result['data_shape'][1]:,})[/green]",
                    style="success"
                )
                return f"Loaded file '{file_path.name}' as modality '{result['modality_name']}'"
            else:
                output.print(
                    f"[red]❌ {result.get('error', 'Unknown error')}[/red]",
                    style="error"
                )
                if result.get("suggestion"):
                    output.print(f"[cyan]💡 {result['suggestion']}[/cyan]", style="info")
                return None

        except Exception as e:
            output.print(f"[red]❌ Failed to load file: {str(e)}[/red]", style="error")
            return None

    # BUG FIX #2: Use cached scan for load command
    if hasattr(client.data_manager, "get_available_datasets"):
        available = client.data_manager.get_available_datasets(
            force_refresh=False
        )
    else:
        # Fallback for older DataManager versions
        if hasattr(client.data_manager, "_scan_workspace"):
            client.data_manager._scan_workspace()
        available = client.data_manager.available_datasets

    if not available:
        output.print("[yellow]No datasets found in workspace[/yellow]", style="warning")
        output.print(
            f"[dim]Tip: If '{selector}' is a file, ensure the path is correct[/dim]",
            style="info"
        )
        return None

    # Determine if selector is an index or pattern
    if selector.isdigit():
        # Index-based loading (single dataset)
        idx = int(selector)
        sorted_names = sorted(available.keys())
        if 1 <= idx <= len(sorted_names):
            dataset_name = sorted_names[idx - 1]

            output.print(
                f"[yellow]Loading dataset: {dataset_name}...[/yellow]",
                style="info"
            )

            # Load single dataset directly
            success = client.data_manager.load_dataset(dataset_name)

            if success:
                output.print(
                    f"[green]✓ Loaded dataset: {dataset_name} ({available[dataset_name]['size_mb']:.1f} MB)[/green]",
                    style="success"
                )
                return "Loaded dataset from workspace"
            else:
                output.print(
                    f"[red]Failed to load dataset: {dataset_name}[/red]",
                    style="error"
                )
                return None
        else:
            output.print(
                f"[red]Index {idx} out of range (1-{len(sorted_names)})[/red]",
                style="error"
            )
            return None
    else:
        # Pattern-based loading (potentially multiple datasets)
        output.print(
            f"[yellow]Loading workspace datasets (pattern: {selector})...[/yellow]",
            style="info"
        )

        # Note: Progress bar creation is CLI-specific, so we skip it here
        # The CLI layer can add progress bars if needed

        # Perform workspace loading
        result = client.data_manager.restore_session(selector)

        # Display results
        if result["restored"]:
            output.print(
                f"[green]✓ Loaded {len(result['restored'])} datasets ({result['total_size_mb']:.1f} MB)[/green]",
                style="success"
            )
            for name in result["restored"]:
                output.print(f"  • {name}", style="info")
            return f"Loaded {len(result['restored'])} datasets from workspace"
        else:
            output.print("[yellow]No datasets loaded[/yellow]", style="warning")
            return None


def workspace_remove(
    client: "AgentClient",
    output: OutputAdapter,
    modality_name: str
) -> Optional[str]:
    """
    Remove a modality from memory.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        modality_name: Name of modality to remove

    Returns:
        Summary string for conversation history, or None
    """
    if not modality_name:
        output.print("[red]Usage: /workspace remove <modality_name>[/red]", style="error")
        output.print(
            "[dim]Examples: /workspace remove geo_gse12345_clustered[/dim]",
            style="info"
        )
        output.print("\n[dim]💡 Tip: Use '/modalities' to see loaded modalities[/dim]", style="info")
        return None

    # Check if modality exists
    if not hasattr(client.data_manager, "list_modalities"):
        output.print(
            "[red]❌ Modality management not available in this client[/red]",
            style="error"
        )
        return None

    available_modalities = client.data_manager.list_modalities()
    if modality_name not in available_modalities:
        output.print(
            f"[red]❌ Modality '{modality_name}' not found[/red]",
            style="error"
        )
        output.print(f"\n[yellow]Available modalities ({len(available_modalities)}):[/yellow]", style="warning")
        for mod in available_modalities:
            output.print(f"  • {mod}", style="info")
        return None

    try:
        # Import the service
        from lobster.services.data_management.modality_management_service import (
            ModalityManagementService,
        )

        # Create service instance and remove modality
        service = ModalityManagementService(client.data_manager)
        success, stats, ir = service.remove_modality(modality_name)

        if success:
            # Log to provenance
            client.data_manager.log_tool_usage(
                tool_name="remove_modality",
                parameters={"modality_name": modality_name},
                description=stats,
                ir=ir,
            )

            # Display success message
            output.print(
                f"[green]✓ Successfully removed modality: {stats['removed_modality']}[/green]",
                style="success"
            )
            output.print(
                f"[dim]  Shape: {stats['shape']['n_obs']} obs × {stats['shape']['n_vars']} vars[/dim]",
                style="info"
            )
            output.print(
                f"[dim]  Remaining modalities: {stats['remaining_modalities']}[/dim]",
                style="info"
            )

            return f"Removed modality: {modality_name}"
        else:
            output.print(
                f"[red]❌ Failed to remove modality: {modality_name}[/red]",
                style="error"
            )
            return None

    except Exception as e:
        output.print(
            f"[red]❌ Error removing modality: {str(e)}[/red]",
            style="error"
        )
        return None


def workspace_status(
    client: "AgentClient",
    output: OutputAdapter
) -> Optional[str]:
    """
    Show workspace status and information.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    output.print("[bold red]🏗️  Workspace Information[/bold red]\n", style="info")

    # Check if using DataManagerV2
    workspace_status_dict = {}
    if hasattr(client.data_manager, "get_workspace_status"):
        workspace_status_dict = client.data_manager.get_workspace_status()

    # Main workspace info table
    workspace_table_data = {
        "title": "🏗️ Workspace Status",
        "columns": [
            {"name": "Property", "style": "bold grey93"},
            {"name": "Value", "style": "white"},
        ],
        "border_style": "red",
        "show_header": True,
        "rows": [
            ["Workspace Path", workspace_status_dict.get("workspace_path", "N/A")],
            ["Modalities Loaded", str(workspace_status_dict.get("modalities_loaded", 0))],
            ["Registered Backends", str(len(workspace_status_dict.get("registered_backends", [])))],
            ["Registered Adapters", str(len(workspace_status_dict.get("registered_adapters", [])))],
            ["Default Backend", workspace_status_dict.get("default_backend", "N/A")],
            [
                "Provenance Enabled",
                "✓" if workspace_status_dict.get("provenance_enabled") else "✗"
            ],
            [
                "MuData Available",
                "✓" if workspace_status_dict.get("mudata_available") else "✗"
            ],
        ]
    }

    output.print_table(workspace_table_data)

    # Show directories
    if workspace_status_dict.get("directories"):
        dirs = workspace_status_dict["directories"]
        output.print("\n[bold white]📁 Directories:[/bold white]", style="info")
        for dir_type, path in dirs.items():
            output.print(f"  • {dir_type.title()}: [grey74]{path}[/grey74]", style="info")

    # Show loaded modalities
    if workspace_status_dict.get("modality_names"):
        output.print("\n[bold white]🧬 Loaded Modalities:[/bold white]", style="info")
        for modality in workspace_status_dict["modality_names"]:
            output.print(f"  • {modality}", style="info")

    # Show available backends and adapters
    output.print("\n[bold white]🔧 Available Backends:[/bold white]", style="info")
    for backend in workspace_status_dict.get("registered_backends", []):
        output.print(f"  • {backend}", style="info")

    output.print("\n[bold white]🔌 Available Adapters:[/bold white]", style="info")
    for adapter in workspace_status_dict.get("registered_adapters", []):
        output.print(f"  • {adapter}", style="info")

    # Show detailed modality information (similar to /modalities command)
    if hasattr(client.data_manager, "list_modalities"):
        modalities = client.data_manager.list_modalities()

        if modalities:
            output.print("\n[bold red]🧬 Modality Details[/bold red]\n", style="info")

            for modality_name in modalities:
                try:
                    adata = client.data_manager.get_modality(modality_name)

                    # Create modality detail table
                    modality_table_data = {
                        "title": f"🧬 {modality_name}",
                        "columns": [
                            {"name": "Property", "style": "bold grey93"},
                            {"name": "Value", "style": "white"},
                        ],
                        "border_style": "cyan",
                        "show_header": False,
                        "rows": []
                    }

                    # Shape
                    modality_table_data["rows"].append([
                        "Shape",
                        f"{adata.n_obs} obs × {adata.n_vars} vars"
                    ])

                    # Show obs columns
                    obs_cols = list(adata.obs.columns)
                    if obs_cols:
                        cols_preview = ", ".join(obs_cols[:5])
                        if len(obs_cols) > 5:
                            cols_preview += f" ... (+{len(obs_cols)-5} more)"
                        modality_table_data["rows"].append(["Obs Columns", cols_preview])

                    # Show var columns
                    var_cols = list(adata.var.columns)
                    if var_cols:
                        var_preview = ", ".join(var_cols[:5])
                        if len(var_cols) > 5:
                            var_preview += f" ... (+{len(var_cols)-5} more)"
                        modality_table_data["rows"].append(["Var Columns", var_preview])

                    # Show layers
                    if adata.layers:
                        layers_str = ", ".join(list(adata.layers.keys()))
                        modality_table_data["rows"].append(["Layers", layers_str])

                    # Show obsm
                    if adata.obsm:
                        obsm_str = ", ".join(list(adata.obsm.keys()))
                        modality_table_data["rows"].append(["Obsm", obsm_str])

                    # Show varm
                    if hasattr(adata, 'varm') and adata.varm:
                        varm_str = ", ".join(list(adata.varm.keys()))
                        modality_table_data["rows"].append(["Varm", varm_str])

                    # Show some uns info
                    if adata.uns:
                        uns_keys = list(adata.uns.keys())[:5]
                        uns_str = ", ".join(uns_keys)
                        if len(adata.uns) > 5:
                            uns_str += f" ... (+{len(adata.uns)-5} more)"
                        modality_table_data["rows"].append(["Uns Keys", uns_str])

                    output.print_table(modality_table_data)
                    output.print("")  # Add spacing between modalities

                except Exception as e:
                    output.print(
                        f"[red]Error accessing modality {modality_name}: {e}[/red]",
                        style="error"
                    )
        else:
            output.print("[grey50]No modalities loaded[/grey50]", style="info")
    else:
        output.print(
            "[grey50]Modality information not available (using legacy DataManager)[/grey50]",
            style="info"
        )

    return "Displayed workspace status and information"
