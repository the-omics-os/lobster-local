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


def _make_progress_bar(pct: float, width: int = 10) -> str:
    """Create ASCII progress bar for percentages."""
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def metadata_overview(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show smart metadata overview with key stats and next steps.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.services.metadata.metadata_overview_service import (
        MetadataOverviewService,
    )

    service = MetadataOverviewService(client.data_manager)
    overview = service.get_quick_overview()

    output.print("\n[bold cyan]📋 Metadata Overview[/bold cyan]\n", style="info")

    # Publication Queue section
    pq = overview.get("publication_queue", {})
    if pq.get("total", 0) > 0:
        output.print("[bold white]Publication Queue[/bold white]", style="info")

        status_table = {
            "title": None,
            "columns": [
                {"name": "Status", "style": "bold white", "width": 20},
                {"name": "Count", "style": "cyan", "width": 10},
            ],
            "rows": [],
        }

        status_emojis = {
            "pending": "⏳",
            "extracting": "🔄",
            "metadata_extracted": "📄",
            "metadata_enriched": "✨",
            "handoff_ready": "🤝",
            "completed": "✅",
            "failed": "❌",
            "paywalled": "🔒",
        }

        for status, count in pq.get("status_breakdown", {}).items():
            emoji = status_emojis.get(status, "📌")
            status_table["rows"].append([f"{emoji} {status}", str(count)])

        if status_table["rows"]:
            output.print_table(status_table)
            output.print(
                f"[grey50]Total: {pq['total']} | Workspace-ready: {pq.get('workspace_ready', 0)} | Extracted datasets: {pq.get('extracted_datasets', 0)}[/grey50]"
            )
        output.print("")

    # Sample Statistics section
    samples = overview.get("samples", {})
    if samples.get("total_samples", 0) > 0:
        output.print("[bold white]Sample Statistics[/bold white]", style="info")
        output.print(
            f"  Total: [cyan]{samples['total_samples']:,}[/cyan] samples from [cyan]{samples.get('bioproject_count', 0)}[/cyan] BioProjects"
        )

        if samples.get("has_aggregated"):
            filtered = samples.get("filtered_samples", 0)
            retention = samples.get("retention_rate", 0)
            output.print(
                f"  Filtered: [cyan]{filtered:,}[/cyan] ([yellow]{retention:.1f}%[/yellow] retention)"
            )

            coverage = samples.get("disease_coverage", 0)
            bar = _make_progress_bar(coverage)
            output.print(
                f"  Disease Coverage: {bar} [yellow]{coverage:.1f}%[/yellow]"
            )
        else:
            output.print(
                "[grey50]  → Run metadata filtering to generate aggregated statistics[/grey50]"
            )
        output.print("")

    # Workspace Files section
    workspace = overview.get("workspace", {})
    if workspace.get("metadata_files", 0) > 0 or workspace.get("export_files", 0) > 0:
        output.print("[bold white]Workspace Files[/bold white]", style="info")
        if workspace.get("metadata_files", 0) > 0:
            output.print(
                f"  Metadata: [cyan]{workspace['metadata_files']}[/cyan] files ([grey50]{workspace.get('total_size_mb', 0):.1f} MB[/grey50])"
            )
        if workspace.get("export_files", 0) > 0:
            output.print(f"  Exports: [cyan]{workspace['export_files']}[/cyan] files")
        if workspace.get("in_memory_entries", 0) > 0:
            output.print(
                f"  In-memory: [cyan]{workspace['in_memory_entries']}[/cyan] entries"
            )
        output.print("")

    # Next Steps
    next_steps = overview.get("next_steps", [])
    if next_steps:
        output.print("[bold yellow]💡 Next Steps[/bold yellow]", style="info")
        for step in next_steps:
            output.print(f"  • {step}")
        output.print("")

    # Deprecated warnings
    if overview.get("has_deprecated"):
        output.print(
            "[yellow]⚠️  Found files in deprecated metadata/exports/ location. Use /metadata workspace for details.[/yellow]"
        )

    # Help text
    output.print(
        "[grey50]Commands: /metadata publications | samples | workspace | exports | clear[/grey50]"
    )

    return f"Metadata overview: {pq.get('total', 0)} publications, {samples.get('total_samples', 0)} samples"


def metadata_publications(
    client: "AgentClient", output: OutputAdapter, status_filter: Optional[str] = None
) -> Optional[str]:
    """
    Show publication queue status breakdown with identifier coverage.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        status_filter: Optional status to filter by

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.services.metadata.metadata_overview_service import (
        MetadataOverviewService,
    )

    service = MetadataOverviewService(client.data_manager)
    summary = service.get_publication_queue_summary(status_filter=status_filter)

    if summary.get("total", 0) == 0:
        output.print(
            "[yellow]No publication queue found. Use research_agent to process publications.[/yellow]"
        )
        return None

    output.print(
        f"\n[bold cyan]📄 Publication Queue ({summary['total']} entries)[/bold cyan]\n",
        style="info",
    )

    # Status breakdown
    output.print("[bold white]Status Breakdown[/bold white]", style="info")
    status_table = {
        "columns": [
            {"name": "Status", "style": "bold white", "width": 25},
            {"name": "Count", "style": "cyan", "width": 10},
        ],
        "rows": [],
    }

    status_emojis = {
        "pending": "⏳",
        "extracting": "🔄",
        "metadata_extracted": "📄",
        "metadata_enriched": "✨",
        "handoff_ready": "🤝",
        "completed": "✅",
        "failed": "❌",
        "paywalled": "🔒",
    }

    for status, count in summary.get("status_breakdown", {}).items():
        emoji = status_emojis.get(status, "📌")
        status_table["rows"].append([f"{emoji} {status}", str(count)])

    output.print_table(status_table)
    output.print("")

    # Identifier coverage
    id_cov = summary.get("identifier_coverage", {})
    if id_cov:
        output.print("[bold white]Identifier Coverage[/bold white]", style="info")
        for id_type, stats in id_cov.items():
            count = stats.get("count", 0)
            pct = stats.get("pct", 0)
            bar = _make_progress_bar(pct, width=15)
            output.print(f"  {id_type.upper()}: {bar} {count}/{summary['total']} ({pct:.1f}%)")
        output.print("")

    # Extracted datasets
    extracted = summary.get("extracted_datasets", {})
    if extracted:
        output.print("[bold white]Extracted Identifiers[/bold white]", style="info")
        for db_type, count in sorted(extracted.items(), key=lambda x: -x[1]):
            output.print(f"  {db_type.upper()}: [cyan]{count}[/cyan] datasets")
        output.print("")

    # Workspace readiness
    ws_ready = summary.get("workspace_ready", 0)
    if ws_ready > 0:
        output.print(
            f"[bold white]Workspace Status:[/bold white] [green]{ws_ready}[/green] entries with metadata files"
        )
        output.print("")

    # Recent errors
    errors = summary.get("recent_errors", [])
    if errors:
        output.print("[bold red]Recent Errors[/bold red]", style="info")
        for err in errors:
            output.print(f"  • [yellow]{err['entry_id']}[/yellow]: {err['title']}")
            output.print(f"    [grey50]{err['error']}[/grey50]")
        output.print("")

    # Filter hint
    if not status_filter:
        output.print(
            "[grey50]Tip: Filter by status with /metadata publications --status=<status>[/grey50]"
        )

    return f"Publication queue: {summary['total']} entries"


def metadata_samples(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show aggregated sample statistics with disease coverage.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.services.metadata.metadata_overview_service import (
        MetadataOverviewService,
    )

    service = MetadataOverviewService(client.data_manager)
    stats = service.get_sample_statistics()

    if stats.get("total_samples", 0) == 0:
        output.print(
            "[yellow]No sample metadata found. Process publications with research_agent first.[/yellow]"
        )
        return None

    output.print(
        f"\n[bold cyan]🧬 Sample Statistics[/bold cyan]\n",
        style="info",
    )

    # Total samples
    total = stats.get("total_samples", 0)
    bioproject_count = stats.get("bioproject_count", 0)
    output.print(
        f"[bold white]Total Samples:[/bold white] [cyan]{total:,}[/cyan] from [cyan]{bioproject_count}[/cyan] BioProjects"
    )
    output.print("")

    if stats.get("has_aggregated"):
        # Filtered samples
        filtered = stats.get("filtered_samples", 0)
        retention = stats.get("retention_rate", 0)
        output.print(
            f"[bold white]Filtered Samples:[/bold white] [cyan]{filtered:,}[/cyan] ([yellow]{retention:.1f}%[/yellow] retention)"
        )

        # Disease coverage
        coverage = stats.get("disease_coverage", 0)
        bar = _make_progress_bar(coverage, width=20)
        output.print(
            f"[bold white]Disease Coverage:[/bold white] {bar} [yellow]{coverage:.1f}%[/yellow]"
        )
        output.print("")

        # Filter criteria
        criteria = stats.get("filter_criteria", "")
        if criteria:
            output.print(f"[bold white]Filter Criteria:[/bold white] [grey50]{criteria}[/grey50]")
            output.print("")

        # Filter breakdown
        breakdown = stats.get("filter_breakdown", {})
        if breakdown:
            output.print("[bold white]Filter Breakdown[/bold white]", style="info")
            for filter_name, filter_stats in breakdown.items():
                if isinstance(filter_stats, dict):
                    retained = filter_stats.get("retained", 0)
                    total_filtered = filter_stats.get("total", 0)
                    pct = (
                        retained / total_filtered * 100 if total_filtered > 0 else 0
                    )
                    bar = _make_progress_bar(pct, width=15)
                    output.print(
                        f"  {filter_name}: {bar} {retained}/{total_filtered} ({pct:.1f}%)"
                    )
            output.print("")

        output.print(
            "[green]✓ Aggregated metadata available. Use /metadata exports to see export files.[/green]"
        )
    else:
        output.print(
            "[yellow]⚠️  Samples not yet filtered. Use metadata_assistant to apply filters and generate aggregated statistics.[/yellow]"
        )
        output.print("")

        # Show sample sources
        sources = stats.get("sources", [])
        if sources:
            output.print(
                f"[bold white]Sample Sources:[/bold white] {len(sources)} BioProject(s)"
            )
            for src in sources[:10]:
                output.print(f"  • [grey50]{src}[/grey50]")
            if len(sources) > 10:
                output.print(f"  [grey50]... and {len(sources) - 10} more[/grey50]")

    return f"Sample stats: {total:,} samples, {bioproject_count} BioProjects"


def metadata_workspace(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show categorized file inventory across all storage locations.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.services.metadata.metadata_overview_service import (
        MetadataOverviewService,
    )

    service = MetadataOverviewService(client.data_manager)
    inventory = service.get_workspace_inventory()

    output.print("\n[bold cyan]📁 Workspace Inventory[/bold cyan]\n", style="info")

    # In-memory metadata store
    mem_count = inventory.get("metadata_store_count", 0)
    if mem_count > 0:
        output.print(
            f"[bold white]In-Memory Store:[/bold white] [cyan]{mem_count}[/cyan] entries",
            style="info",
        )
        categories = inventory.get("metadata_store_categories", {})
        if categories:
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                output.print(f"  • {cat}: {count}")
        output.print("")

    # Workspace files
    ws_files = inventory.get("workspace_files", {})
    if ws_files:
        total_files = inventory.get("workspace_files_total", 0)
        size_mb = inventory.get("total_size_mb", 0)
        output.print(
            f"[bold white]Workspace Files:[/bold white] [cyan]{total_files}[/cyan] files ([grey50]{size_mb:.1f} MB[/grey50])",
            style="info",
        )
        for cat, count in sorted(ws_files.items(), key=lambda x: -x[1]):
            output.print(f"  • {cat}: {count}")
        output.print("")

    # Export files
    exports = inventory.get("exports", [])
    if exports:
        total_exports = inventory.get("exports_total", 0)
        output.print(
            f"[bold white]Export Files:[/bold white] [cyan]{total_exports}[/cyan] files",
            style="info",
        )
        for exp in exports[:10]:
            output.print(
                f"  • {exp['name']} [grey50]({exp['size_kb']} KB, {exp['modified']})[/grey50]"
            )
        if len(exports) > 10:
            output.print(f"  [grey50]... and {total_exports - 10} more files[/grey50]")
        output.print("")

    # Deprecated warnings
    warnings = inventory.get("deprecated_warnings", [])
    if warnings:
        output.print("[bold yellow]⚠️  Deprecated Locations[/bold yellow]", style="warning")
        for warn in warnings:
            output.print(f"  • {warn}")
        output.print(
            "[yellow]Consider migrating: mv workspace/metadata/exports/* workspace/exports/[/yellow]"
        )

    return f"Workspace: {mem_count} in-memory, {inventory.get('workspace_files_total', 0)} files"


def metadata_exports(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show export files with categories and usage guidance.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.services.metadata.metadata_overview_service import (
        MetadataOverviewService,
    )

    service = MetadataOverviewService(client.data_manager)
    exports = service.get_export_summary()

    if exports.get("total_count", 0) == 0:
        output.print(
            "[yellow]No export files found. Use write_to_workspace() to export data.[/yellow]"
        )
        return None

    output.print(
        f"\n[bold cyan]📤 Export Files ({exports['total_count']} files)[/bold cyan]\n",
        style="info",
    )

    # Categories
    categories = exports.get("categories", {})
    if categories:
        output.print("[bold white]File Categories[/bold white]", style="info")
        cat_emojis = {
            "rich_export": "📊",
            "strict_mimarks": "✅",
            "provenance_log": "📜",
            "sample_data": "🧬",
            "analysis_results": "📈",
            "other": "📄",
        }
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            emoji = cat_emojis.get(cat, "📄")
            output.print(f"  {emoji} {cat}: {count}")
        output.print("")

    # File listing
    files = exports.get("files", [])
    if files:
        output.print("[bold white]Recent Files[/bold white]", style="info")
        files_table = {
            "columns": [
                {"name": "File", "style": "cyan", "width": 50, "overflow": "ellipsis"},
                {"name": "Size", "style": "grey50", "width": 10},
                {"name": "Modified", "style": "grey50", "width": 18},
            ],
            "rows": [],
        }
        for f in files[:15]:
            files_table["rows"].append(
                [f["name"], f"{f['size_kb']:.1f} KB", f["modified"]]
            )
        output.print_table(files_table)
        if len(files) > 15:
            output.print(f"[grey50]... and {exports['total_count'] - 15} more files[/grey50]")
        output.print("")

    # Usage hints
    hints = exports.get("usage_hints", {})
    if hints:
        output.print("[bold yellow]💡 Usage Tips[/bold yellow]", style="info")
        output.print(f"  • List exports: [cyan]{hints.get('list', 'N/A')}[/cyan]")
        output.print(f"  • Access in code: [cyan]{hints.get('access', 'N/A')}[/cyan]")
        output.print(f"  • CLI command: [cyan]{hints.get('cli', 'N/A')}[/cyan]")

    return f"Export files: {exports['total_count']} files"


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
    Clear metadata store (memory) AND workspace metadata files (disk).

    Clears:
    1. In-memory metadata_store (DataManagerV2)
    2. In-memory current_metadata (legacy)
    3. Workspace metadata files (workspace/metadata/*.json)

    NOTE: Does NOT clear export files. Use /metadata clear exports for that.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    # Count in-memory entries (metadata_store)
    metadata_store_count = 0
    if hasattr(client.data_manager, "metadata_store"):
        metadata_store_count = len(client.data_manager.metadata_store)

    # Count in-memory current_metadata (legacy)
    current_metadata_count = 0
    if hasattr(client.data_manager, "current_metadata"):
        current_metadata_count = len(client.data_manager.current_metadata)

    memory_count = metadata_store_count + current_metadata_count

    # Count disk files in workspace/metadata/
    disk_files = []
    metadata_dir = None
    if hasattr(client.data_manager, "workspace_path"):
        metadata_dir = Path(client.data_manager.workspace_path) / "metadata"
        if metadata_dir.exists():
            disk_files = list(metadata_dir.glob("*.json"))

    disk_count = len(disk_files)
    total_count = memory_count + disk_count

    if total_count == 0:
        output.print("[grey50]Metadata is already empty (memory + disk)[/grey50]", style="info")
        return "Metadata already empty"

    # Show what will be cleared
    output.print("\n[bold]About to clear:[/bold]")
    output.print(f"  • Memory (metadata_store): {metadata_store_count} entries")
    output.print(f"  • Memory (current_metadata): {current_metadata_count} entries")
    output.print(f"  • Disk (workspace/metadata/): {disk_count} files")
    output.print(f"  • Total: {total_count} items\n")

    # Confirm with user
    confirm = output.confirm(f"[yellow]Clear all {total_count} metadata items?[/yellow]")

    if confirm:
        # 1. Clear metadata_store (in-memory cache)
        cleared_store = 0
        if metadata_store_count > 0 and hasattr(client.data_manager, "metadata_store"):
            client.data_manager.metadata_store.clear()
            cleared_store = metadata_store_count

        # 2. Clear current_metadata (legacy in-memory)
        cleared_current = 0
        if current_metadata_count > 0 and hasattr(client.data_manager, "current_metadata"):
            client.data_manager.current_metadata.clear()
            cleared_current = current_metadata_count

        # 3. Clear disk files
        deleted_files = 0
        failed_files = 0
        for json_file in disk_files:
            try:
                json_file.unlink()
                deleted_files += 1
            except Exception as e:
                failed_files += 1
                output.print(
                    f"[yellow]⚠️  Could not delete {json_file.name}: {e}[/yellow]",
                    style="warning"
                )

        # Report results
        result_parts = []
        if cleared_store > 0:
            result_parts.append(f"{cleared_store} metadata_store entries")
        if cleared_current > 0:
            result_parts.append(f"{cleared_current} current_metadata entries")
        if deleted_files > 0:
            result_parts.append(f"{deleted_files} disk files")

        output.print(
            f"[green]✓ Cleared {' + '.join(result_parts)}[/green]",
            style="success"
        )

        if failed_files > 0:
            output.print(
                f"[yellow]⚠️  {failed_files} files could not be deleted[/yellow]",
                style="warning"
            )

        return f"Cleared {total_count} metadata items (memory + disk)"
    else:
        output.print("[cyan]Operation cancelled[/cyan]", style="info")
        return None


def metadata_clear_exports(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Clear export files from workspace/exports/.

    Clears:
    - All files in workspace/exports/ (*.csv, *.tsv, *.xlsx)

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    # Count export files
    export_files = []
    exports_dir = None
    if hasattr(client.data_manager, "workspace_path"):
        exports_dir = Path(client.data_manager.workspace_path) / "exports"
        if exports_dir.exists():
            export_files = [
                f for f in exports_dir.iterdir()
                if f.is_file() and f.suffix in {".csv", ".tsv", ".xlsx"}
            ]

    if not export_files:
        output.print("[grey50]No export files to clear[/grey50]", style="info")
        return "No export files to clear"

    # Show what will be cleared
    output.print("\n[bold]About to clear export files:[/bold]")
    output.print(f"  • Location: {exports_dir}")
    output.print(f"  • Files: {len(export_files)}\n")

    # Show first few files
    for f in export_files[:5]:
        size_kb = f.stat().st_size / 1024
        output.print(f"    • {f.name} ({size_kb:.1f} KB)")
    if len(export_files) > 5:
        output.print(f"    • ... and {len(export_files) - 5} more files\n")

    # Confirm with user
    confirm = output.confirm(
        f"[yellow]Delete all {len(export_files)} export files?[/yellow]"
    )

    if confirm:
        deleted = 0
        failed = 0
        for f in export_files:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                failed += 1
                output.print(
                    f"[yellow]⚠️  Could not delete {f.name}: {e}[/yellow]",
                    style="warning"
                )

        output.print(
            f"[green]✓ Deleted {deleted} export files[/green]",
            style="success"
        )

        if failed > 0:
            output.print(
                f"[yellow]⚠️  {failed} files could not be deleted[/yellow]",
                style="warning"
            )

        return f"Cleared {deleted} export files"
    else:
        output.print("[cyan]Operation cancelled[/cyan]", style="info")
        return None


def metadata_clear_all(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Clear ALL metadata: memory, workspace/metadata/, and workspace/exports/.

    This is the most comprehensive clear operation. Equivalent to:
    - /metadata clear + /metadata clear exports

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    # ========================================================================
    # Count all items
    # ========================================================================

    # Memory: metadata_store
    metadata_store_count = 0
    if hasattr(client.data_manager, "metadata_store"):
        metadata_store_count = len(client.data_manager.metadata_store)

    # Memory: current_metadata
    current_metadata_count = 0
    if hasattr(client.data_manager, "current_metadata"):
        current_metadata_count = len(client.data_manager.current_metadata)

    # Disk: workspace/metadata/*.json
    disk_files = []
    metadata_dir = None
    if hasattr(client.data_manager, "workspace_path"):
        metadata_dir = Path(client.data_manager.workspace_path) / "metadata"
        if metadata_dir.exists():
            disk_files = list(metadata_dir.glob("*.json"))

    # Disk: workspace/exports/*
    export_files = []
    exports_dir = None
    if hasattr(client.data_manager, "workspace_path"):
        exports_dir = Path(client.data_manager.workspace_path) / "exports"
        if exports_dir.exists():
            export_files = [
                f for f in exports_dir.iterdir()
                if f.is_file() and f.suffix in {".csv", ".tsv", ".xlsx"}
            ]

    # Totals
    memory_count = metadata_store_count + current_metadata_count
    disk_count = len(disk_files)
    export_count = len(export_files)
    total_count = memory_count + disk_count + export_count

    if total_count == 0:
        output.print(
            "[grey50]Nothing to clear (memory, metadata, exports all empty)[/grey50]",
            style="info"
        )
        return "Nothing to clear"

    # ========================================================================
    # Show what will be cleared
    # ========================================================================
    output.print("\n[bold red]⚠️  About to clear ALL metadata:[/bold red]")
    output.print(f"  • Memory (metadata_store): {metadata_store_count} entries")
    output.print(f"  • Memory (current_metadata): {current_metadata_count} entries")
    output.print(f"  • Disk (workspace/metadata/): {disk_count} files")
    output.print(f"  • Disk (workspace/exports/): {export_count} files")
    output.print(f"  • Total: {total_count} items\n")

    # Confirm with user
    confirm = output.confirm(
        f"[bold red]Clear ALL {total_count} items? This cannot be undone![/bold red]"
    )

    if confirm:
        results = []

        # 1. Clear metadata_store
        if metadata_store_count > 0 and hasattr(client.data_manager, "metadata_store"):
            client.data_manager.metadata_store.clear()
            results.append(f"{metadata_store_count} metadata_store entries")

        # 2. Clear current_metadata
        if current_metadata_count > 0 and hasattr(client.data_manager, "current_metadata"):
            client.data_manager.current_metadata.clear()
            results.append(f"{current_metadata_count} current_metadata entries")

        # 3. Clear workspace/metadata/*.json
        deleted_metadata = 0
        for f in disk_files:
            try:
                f.unlink()
                deleted_metadata += 1
            except Exception as e:
                output.print(
                    f"[yellow]⚠️  Could not delete {f.name}: {e}[/yellow]",
                    style="warning"
                )
        if deleted_metadata > 0:
            results.append(f"{deleted_metadata} metadata files")

        # 4. Clear workspace/exports/*
        deleted_exports = 0
        for f in export_files:
            try:
                f.unlink()
                deleted_exports += 1
            except Exception as e:
                output.print(
                    f"[yellow]⚠️  Could not delete {f.name}: {e}[/yellow]",
                    style="warning"
                )
        if deleted_exports > 0:
            results.append(f"{deleted_exports} export files")

        # Report
        output.print(
            f"[green]✓ Cleared: {', '.join(results)}[/green]",
            style="success"
        )

        return f"Cleared all metadata ({total_count} items)"
    else:
        output.print("[cyan]Operation cancelled[/cyan]", style="info")
        return None
