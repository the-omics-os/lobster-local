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
    selector: str
) -> Optional[str]:
    """
    Remove modality(ies) from memory by index, pattern, or exact name.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        selector: Index (#), pattern (with wildcards), or exact modality name

    Returns:
        Summary string for conversation history, or None
    """
    if not selector:
        output.print("[red]Usage: /workspace remove <#|pattern|name>[/red]", style="error")
        output.print(
            "[dim]Examples:[/dim]",
            style="info"
        )
        output.print(
            "[dim]  /workspace remove 1              - Remove by index[/dim]",
            style="info"
        )
        output.print(
            "[dim]  /workspace remove *              - Remove all modalities[/dim]",
            style="info"
        )
        output.print(
            "[dim]  /workspace remove *clustered*    - Remove matching pattern[/dim]",
            style="info"
        )
        output.print(
            "[dim]  /workspace remove geo_gse12345   - Remove by exact name[/dim]",
            style="info"
        )
        output.print("\n[dim]💡 Tip: Use '/modalities' to see loaded modalities with indexes[/dim]", style="info")
        return None

    # Check if modality management is available
    if not hasattr(client.data_manager, "list_modalities"):
        output.print(
            "[red]❌ Modality management not available in this client[/red]",
            style="error"
        )
        return None

    available_modalities = client.data_manager.list_modalities()

    if not available_modalities:
        output.print("[yellow]No modalities currently loaded[/yellow]", style="warning")
        return None

    # Determine which modalities to remove
    modalities_to_remove = []
    sorted_modalities = sorted(available_modalities)

    if selector.isdigit():
        # Index-based removal
        idx = int(selector)
        if 1 <= idx <= len(sorted_modalities):
            modalities_to_remove = [sorted_modalities[idx - 1]]
        else:
            output.print(
                f"[red]Index {idx} out of range (1-{len(sorted_modalities)})[/red]",
                style="error"
            )
            output.print(f"\n[yellow]Available modalities ({len(available_modalities)}):[/yellow]", style="warning")
            for i, mod in enumerate(sorted_modalities, start=1):
                output.print(f"  {i}. {mod}", style="info")
            return None
    elif "*" in selector or "?" in selector or "[" in selector:
        # Pattern-based removal (wildcards detected)
        for name in sorted_modalities:
            if fnmatch.fnmatch(name.lower(), selector.lower()):
                modalities_to_remove.append(name)

        if not modalities_to_remove:
            output.print(
                f"[yellow]No modalities match pattern: {selector}[/yellow]",
                style="warning"
            )
            output.print(f"\n[yellow]Available modalities ({len(available_modalities)}):[/yellow]", style="warning")
            for i, mod in enumerate(sorted_modalities, start=1):
                output.print(f"  {i}. {mod}", style="info")
            return None
    else:
        # Exact name match
        if selector in available_modalities:
            modalities_to_remove = [selector]
        else:
            output.print(
                f"[red]❌ Modality '{selector}' not found[/red]",
                style="error"
            )
            output.print(f"\n[yellow]Available modalities ({len(available_modalities)}):[/yellow]", style="warning")
            for i, mod in enumerate(sorted_modalities, start=1):
                output.print(f"  {i}. {mod}", style="info")
            return None

    # Confirm removal for multiple modalities
    if len(modalities_to_remove) > 1:
        output.print(
            f"[yellow]⚠️  About to remove {len(modalities_to_remove)} modalities:[/yellow]",
            style="warning"
        )
        for mod in modalities_to_remove:
            output.print(f"  • {mod}", style="info")
        output.print("")  # spacing

    try:
        # Import the service
        from lobster.services.data_management.modality_management_service import (
            ModalityManagementService,
        )

        # Create service instance
        service = ModalityManagementService(client.data_manager)

        # Remove each modality
        removed_count = 0
        failed_count = 0

        for modality_name in modalities_to_remove:
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
                    f"[green]✓ Removed: {stats['removed_modality']}[/green]",
                    style="success"
                )
                if len(modalities_to_remove) == 1:
                    # Show detailed info only for single removal
                    output.print(
                        f"[dim]  Shape: {stats['shape']['n_obs']} obs × {stats['shape']['n_vars']} vars[/dim]",
                        style="info"
                    )
                removed_count += 1
            else:
                output.print(
                    f"[red]✗ Failed to remove: {modality_name}[/red]",
                    style="error"
                )
                failed_count += 1

        # Summary for multiple removals
        if len(modalities_to_remove) > 1:
            remaining = client.data_manager.list_modalities()
            output.print(
                f"\n[dim]Summary: {removed_count} removed, {failed_count} failed, {len(remaining)} remaining[/dim]",
                style="info"
            )

        if removed_count > 0:
            if removed_count == 1:
                return f"Removed modality: {modalities_to_remove[0]}"
            else:
                return f"Removed {removed_count} modalities"
        return None

    except Exception as e:
        output.print(
            f"[red]❌ Error removing modality: {str(e)}[/red]",
            style="error"
        )
        return None


def _get_directory_stats(dir_path: str) -> tuple:
    """
    Get file count and total size for a directory.

    Args:
        dir_path: Path to directory

    Returns:
        Tuple of (file_count, total_size_str, exists)
    """
    path = Path(dir_path)
    if not path.exists():
        return 0, "-", False

    try:
        files = list(path.rglob("*"))
        file_count = sum(1 for f in files if f.is_file())
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        # Format size
        if total_size < 1024:
            size_str = f"{total_size} B"
        elif total_size < 1024 * 1024:
            size_str = f"{total_size / 1024:.1f} KB"
        elif total_size < 1024 * 1024 * 1024:
            size_str = f"{total_size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{total_size / (1024 * 1024 * 1024):.2f} GB"

        return file_count, size_str, True
    except Exception:
        return 0, "-", True


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
    # Check if using DataManagerV2
    workspace_status_dict = {}
    if hasattr(client.data_manager, "get_workspace_status"):
        workspace_status_dict = client.data_manager.get_workspace_status()

    workspace_path = workspace_status_dict.get("workspace_path", "N/A")

    # Header with workspace path
    output.print("\n[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]", style="info")
    output.print("[bold white]                            🏗️  WORKSPACE STATUS[/bold white]", style="info")
    output.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]\n", style="info")

    # Workspace path prominently displayed
    output.print(f"[bold white]📍 Location:[/bold white] [grey74]{workspace_path}[/grey74]\n", style="info")

    # Quick stats row
    modalities_count = workspace_status_dict.get("modalities_loaded", 0)
    provenance = "✓ Enabled" if workspace_status_dict.get("provenance_enabled") else "✗ Disabled"
    mudata = "✓ Available" if workspace_status_dict.get("mudata_available") else "✗ Not installed"

    output.print(f"[bold white]📊 Quick Stats:[/bold white]  Modalities: [cyan]{modalities_count}[/cyan]  │  Provenance: [green]{provenance}[/green]  │  MuData: {mudata}", style="info")

    # Directories section with enhanced display
    if workspace_status_dict.get("directories"):
        dirs = workspace_status_dict["directories"]

        output.print("\n[bold cyan]─────────────────────────────────────────────────────────────────────────────[/bold cyan]", style="info")
        output.print("[bold white]                              📁 DIRECTORIES[/bold white]", style="info")
        output.print("[bold cyan]─────────────────────────────────────────────────────────────────────────────[/bold cyan]\n", style="info")

        # Define icons for each directory type
        dir_icons = {
            "data": "💾",
            "exports": "📤",
            "cache": "🗄️",
            "literature_cache": "📚",
            "metadata": "🏷️",
            "notebooks": "📓",
            "queues": "📋",
        }

        # Create directory table with stats
        dir_table_data = {
            "title": "",
            "columns": [
                {"name": "Directory", "style": "bold white"},
                {"name": "Files", "style": "cyan", "justify": "right"},
                {"name": "Size", "style": "green", "justify": "right"},
                {"name": "Path", "style": "grey70"},
            ],
            "border_style": "dim cyan",
            "show_header": True,
            "rows": []
        }

        for dir_type, path in dirs.items():
            file_count, size_str, exists = _get_directory_stats(path)
            icon = dir_icons.get(dir_type, "📁")

            # Format display name
            display_name = dir_type.replace("_", " ").title()

            # Truncate path for display
            truncated_path = truncate_middle(path, 45)

            # Status indicator
            if not exists:
                status = "[dim red](not created)[/dim red]"
                truncated_path = f"{truncated_path} {status}"

            dir_table_data["rows"].append([
                f"{icon} {display_name}",
                str(file_count) if exists else "-",
                size_str,
                truncated_path
            ])

        output.print_table(dir_table_data)

    # Loaded modalities section
    modality_names = workspace_status_dict.get("modality_names", [])
    if modality_names:
        output.print("\n[bold cyan]─────────────────────────────────────────────────────────────────────────────[/bold cyan]", style="info")
        output.print("[bold white]                           🧬 LOADED MODALITIES[/bold white]", style="info")
        output.print("[bold cyan]─────────────────────────────────────────────────────────────────────────────[/bold cyan]\n", style="info")

        for modality in modality_names:
            output.print(f"  [green]●[/green] {modality}", style="info")
    else:
        output.print("\n[dim white]🧬 No modalities currently loaded[/dim white]", style="info")

    # System capabilities (collapsed view)
    output.print("\n[bold cyan]─────────────────────────────────────────────────────────────────────────────[/bold cyan]", style="info")
    output.print("[bold white]                           🔧 SYSTEM CAPABILITIES[/bold white]", style="info")
    output.print("[bold cyan]─────────────────────────────────────────────────────────────────────────────[/bold cyan]\n", style="info")

    backends = workspace_status_dict.get("registered_backends", [])
    adapters = workspace_status_dict.get("registered_adapters", [])

    output.print(f"[bold white]Backends ({len(backends)}):[/bold white] {', '.join(backends) if backends else 'None'}", style="info")
    output.print(f"[bold white]Adapters ({len(adapters)}):[/bold white] {', '.join(adapters[:5]) if adapters else 'None'}", style="info")
    if len(adapters) > 5:
        output.print(f"           [grey50]... and {len(adapters) - 5} more[/grey50]", style="info")

    # Show detailed modality information if modalities are loaded
    if hasattr(client.data_manager, "list_modalities"):
        modalities = client.data_manager.list_modalities()

        if modalities:
            output.print("\n[bold cyan]─────────────────────────────────────────────────────────────────────────────[/bold cyan]", style="info")
            output.print("[bold white]                          🔬 MODALITY DETAILS[/bold white]", style="info")
            output.print("[bold cyan]─────────────────────────────────────────────────────────────────────────────[/bold cyan]\n", style="info")

            for modality_name in modalities:
                try:
                    adata = client.data_manager.get_modality(modality_name)

                    # Create modality detail table with consistent styling
                    modality_table_data = {
                        "title": f"🧬 {modality_name}",
                        "columns": [
                            {"name": "Property", "style": "bold grey93"},
                            {"name": "Value", "style": "white"},
                        ],
                        "border_style": "dim cyan",
                        "show_header": False,
                        "rows": []
                    }

                    # Shape
                    modality_table_data["rows"].append([
                        "Shape",
                        f"[cyan]{adata.n_obs:,}[/cyan] obs × [cyan]{adata.n_vars:,}[/cyan] vars"
                    ])

                    # Show obs columns
                    obs_cols = list(adata.obs.columns)
                    if obs_cols:
                        cols_preview = ", ".join(obs_cols[:5])
                        if len(obs_cols) > 5:
                            cols_preview += f" [grey50]... (+{len(obs_cols)-5} more)[/grey50]"
                        modality_table_data["rows"].append(["Obs Columns", cols_preview])

                    # Show var columns
                    var_cols = list(adata.var.columns)
                    if var_cols:
                        var_preview = ", ".join(var_cols[:5])
                        if len(var_cols) > 5:
                            var_preview += f" [grey50]... (+{len(var_cols)-5} more)[/grey50]"
                        modality_table_data["rows"].append(["Var Columns", var_preview])

                    # Show layers
                    if adata.layers:
                        layers_str = ", ".join(list(adata.layers.keys()))
                        modality_table_data["rows"].append(["Layers", f"[green]{layers_str}[/green]"])

                    # Show obsm
                    if adata.obsm:
                        obsm_str = ", ".join(list(adata.obsm.keys()))
                        modality_table_data["rows"].append(["Obsm", f"[yellow]{obsm_str}[/yellow]"])

                    # Show varm
                    if hasattr(adata, 'varm') and adata.varm:
                        varm_str = ", ".join(list(adata.varm.keys()))
                        modality_table_data["rows"].append(["Varm", f"[yellow]{varm_str}[/yellow]"])

                    # Show some uns info
                    if adata.uns:
                        uns_keys = list(adata.uns.keys())[:5]
                        uns_str = ", ".join(uns_keys)
                        if len(adata.uns) > 5:
                            uns_str += f" [grey50]... (+{len(adata.uns)-5} more)[/grey50]"
                        modality_table_data["rows"].append(["Uns Keys", uns_str])

                    output.print_table(modality_table_data)
                    output.print("")  # Add spacing between modalities

                except Exception as e:
                    output.print(
                        f"  [red]✗ Error accessing {modality_name}: {e}[/red]",
                        style="error"
                    )

    # Footer
    output.print("[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]\n", style="info")

    return "Displayed workspace status and information"
