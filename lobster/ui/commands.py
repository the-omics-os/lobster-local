"""Command palette commands for Lobster OS.

Wraps CLI functionality for use in the Textual UI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
from pathlib import Path

from textual.command import Provider, Hits, Hit, DiscoveryHit

if TYPE_CHECKING:
    from lobster.core.client import AgentClient


class LobsterCommands(Provider):
    """Command provider for Lobster OS.

    Press Ctrl+P to open command palette.
    """

    @property
    def client(self) -> "AgentClient | None":
        """Get the AgentClient from the app."""
        return getattr(self.app, "client", None)

    async def discover(self) -> Hits:
        """Show available commands when palette opens."""
        # Data commands
        yield DiscoveryHit("Data: Show loaded modalities", self._cmd_data, "View current datasets")
        yield DiscoveryHit("Data: Show metadata", self._cmd_metadata, "View dataset metadata")
        yield DiscoveryHit("Files: List workspace files", self._cmd_files, "Show workspace contents")
        yield DiscoveryHit("Files: Show directory tree", self._cmd_tree, "Tree view of workspace")

        # Plot commands
        yield DiscoveryHit("Plots: List generated plots", self._cmd_plots, "View all plots")
        yield DiscoveryHit("Plots: Open plots folder", self._cmd_open_plots, "Open in file manager")

        # Queue commands
        yield DiscoveryHit("Queue: Show download status", self._cmd_queue_downloads, "Download queue status")
        yield DiscoveryHit("Queue: Show publication status", self._cmd_queue_publications, "Publication queue status")

        # Pipeline commands
        yield DiscoveryHit("Pipeline: Export notebook", self._cmd_pipeline_export, "Export as Jupyter notebook")
        yield DiscoveryHit("Pipeline: List notebooks", self._cmd_pipeline_list, "Show saved notebooks")

        # System commands
        yield DiscoveryHit("Session: Show session status", self._cmd_session, "Current session info")
        yield DiscoveryHit("Status: Show installation status", self._cmd_status, "Tier, packages, and agents")
        yield DiscoveryHit("Tokens: Show token usage", self._cmd_tokens, "Token usage and costs")
        yield DiscoveryHit("Workspace: Show info", self._cmd_workspace, "Workspace details")

        # Actions
        yield DiscoveryHit("Action: Save session", self._cmd_save, "Save current state")
        yield DiscoveryHit("Action: Clear results", self._cmd_clear, "Clear results display")
        yield DiscoveryHit("Action: Refresh all", self._cmd_refresh, "Refresh all panels")

    async def search(self, query: str) -> Hits:
        """Search commands by query."""
        query_lower = query.lower()

        commands = [
            ("data", "Show loaded modalities", self._cmd_data),
            ("metadata", "Show dataset metadata", self._cmd_metadata),
            ("files", "List workspace files", self._cmd_files),
            ("tree", "Show directory tree", self._cmd_tree),
            ("plots", "List generated plots", self._cmd_plots),
            ("open plots", "Open plots folder", self._cmd_open_plots),
            ("queue downloads", "Download queue status", self._cmd_queue_downloads),
            ("queue publications", "Publication queue status", self._cmd_queue_publications),
            ("pipeline export", "Export as Jupyter notebook", self._cmd_pipeline_export),
            ("pipeline list", "List saved notebooks", self._cmd_pipeline_list),
            ("session", "Session status", self._cmd_session),
            ("status", "Installation status", self._cmd_status),
            ("tokens", "Token usage", self._cmd_tokens),
            ("workspace", "Workspace info", self._cmd_workspace),
            ("save", "Save session", self._cmd_save),
            ("clear", "Clear results", self._cmd_clear),
            ("refresh", "Refresh panels", self._cmd_refresh),
        ]

        for name, help_text, callback in commands:
            if query_lower in name or query_lower in help_text.lower():
                yield Hit(
                    score=100 if name.startswith(query_lower) else 50,
                    match_display=name,
                    command=callback,
                    help=help_text,
                )

    # =========================================================================
    # Command implementations
    # =========================================================================

    async def _cmd_data(self) -> None:
        """Show loaded modalities/datasets."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        datasets = self.client.data_manager.available_datasets
        if not datasets:
            self.app.notify("No data loaded", severity="warning")
            return

        lines = ["**Loaded Datasets:**\n"]
        for name, info in datasets.items():
            shape = info.get("shape", "?")
            size = info.get("size_mb", 0)
            lines.append(f"- `{name}`: {shape} ({size:.1f} MB)")

        self._show_result("\n".join(lines))

    async def _cmd_metadata(self) -> None:
        """Show dataset metadata."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        datasets = self.client.data_manager.available_datasets
        if not datasets:
            self.app.notify("No data loaded", severity="warning")
            return

        lines = ["**Dataset Metadata:**\n"]
        for name, info in datasets.items():
            lines.append(f"### {name}")
            lines.append(f"- Shape: {info.get('shape', '?')}")
            lines.append(f"- Size: {info.get('size_mb', 0):.1f} MB")
            if "obs_columns" in info:
                lines.append(f"- Obs columns: {len(info['obs_columns'])}")
            if "var_columns" in info:
                lines.append(f"- Var columns: {len(info['var_columns'])}")
            lines.append("")

        self._show_result("\n".join(lines))

    async def _cmd_files(self) -> None:
        """List workspace files."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        workspace = self.client.workspace_path
        files = list(workspace.glob("*"))

        if not files:
            self.app.notify("Workspace is empty", severity="warning")
            return

        lines = [f"**Workspace:** `{workspace}`\n"]
        for f in sorted(files)[:20]:
            icon = "📁" if f.is_dir() else "📄"
            size = f.stat().st_size if f.is_file() else 0
            size_str = f"({size / 1024:.1f} KB)" if size > 0 else ""
            lines.append(f"{icon} {f.name} {size_str}")

        if len(files) > 20:
            lines.append(f"\n... and {len(files) - 20} more files")

        self._show_result("\n".join(lines))

    async def _cmd_tree(self) -> None:
        """Show directory tree."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        workspace = self.client.workspace_path
        lines = [f"**{workspace.name}/**"]

        def _tree(path: Path, prefix: str = "", depth: int = 0):
            if depth > 3:
                return
            try:
                items = sorted(path.iterdir())[:15]
                for i, item in enumerate(items):
                    is_last = i == len(items) - 1
                    connector = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{connector}{item.name}")
                    if item.is_dir():
                        new_prefix = prefix + ("    " if is_last else "│   ")
                        _tree(item, new_prefix, depth + 1)
            except PermissionError:
                pass

        _tree(workspace)
        self._show_result("\n".join(lines[:50]))

    async def _cmd_plots(self) -> None:
        """List generated plots."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        plots_dir = self.client.workspace_path / "plots"
        if not plots_dir.exists():
            self.app.notify("No plots directory", severity="warning")
            return

        plots = list(plots_dir.glob("*.html")) + list(plots_dir.glob("*.png"))
        if not plots:
            self.app.notify("No plots found", severity="warning")
            return

        lines = ["**Generated Plots:**\n"]
        for p in sorted(plots, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
            lines.append(f"- `{p.name}`")

        self._show_result("\n".join(lines))

    async def _cmd_open_plots(self) -> None:
        """Open plots folder in file manager."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        import subprocess
        import platform

        plots_dir = self.client.workspace_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        if platform.system() == "Darwin":
            subprocess.run(["open", str(plots_dir)])
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", str(plots_dir)])
        else:
            subprocess.run(["explorer", str(plots_dir)])

        self.app.notify(f"Opened {plots_dir.name}/", timeout=2)

    async def _cmd_queue_downloads(self) -> None:
        """Show download queue status."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        queue = self.client.data_manager.download_queue
        entries = queue.list_entries()

        if not entries:
            self.app.notify("Download queue is empty", severity="information")
            return

        lines = ["**Download Queue:**\n"]
        for e in entries[:10]:
            status_icon = {"pending": "○", "in_progress": "●", "completed": "✓", "failed": "✗"}.get(e.status, "?")
            lines.append(f"{status_icon} `{e.dataset_id}` - {e.status}")

        self._show_result("\n".join(lines))

    async def _cmd_queue_publications(self) -> None:
        """Show publication queue status."""
        if not self.client or not self.client.publication_queue:
            self.app.notify("Publication queue not available", severity="warning")
            return

        entries = self.client.publication_queue.list_entries()

        if not entries:
            self.app.notify("Publication queue is empty", severity="information")
            return

        lines = ["**Publication Queue:**\n"]
        for e in entries[:10]:
            status_icon = {"pending": "○", "in_progress": "●", "completed": "✓", "failed": "✗"}.get(e.status, "?")
            title = getattr(e, "title", "")[:40] or e.entry_id
            lines.append(f"{status_icon} {title} - {e.status}")

        self._show_result("\n".join(lines))

    async def _cmd_pipeline_export(self) -> None:
        """Export session as Jupyter notebook."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        try:
            result = self.client.data_manager.export_notebook()
            if result:
                self.app.notify(f"Exported: {result}", timeout=5)
            else:
                self.app.notify("No analysis steps to export", severity="warning")
        except Exception as e:
            self.app.notify(f"Export failed: {e}", severity="error")

    async def _cmd_pipeline_list(self) -> None:
        """List saved notebooks."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        notebooks_dir = self.client.workspace_path / "notebooks"
        if not notebooks_dir.exists():
            self.app.notify("No notebooks directory", severity="warning")
            return

        notebooks = list(notebooks_dir.glob("*.ipynb"))
        if not notebooks:
            self.app.notify("No notebooks found", severity="warning")
            return

        lines = ["**Saved Notebooks:**\n"]
        for nb in sorted(notebooks, key=lambda x: x.stat().st_mtime, reverse=True):
            lines.append(f"- `{nb.name}`")

        self._show_result("\n".join(lines))

    async def _cmd_session(self) -> None:
        """Show current session status."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        from lobster.config.llm_factory import LLMFactory
        from lobster.config.agent_config import get_agent_configurator

        # Get session info
        status = self.client.get_status()
        provider = LLMFactory.get_current_provider() or "unknown"
        configurator = get_agent_configurator()
        current_mode = configurator.get_current_profile()

        lines = [
            "**Session Status:**\n",
            f"- Session ID: `{status['session_id']}`",
            f"- Mode: {current_mode}",
            f"- Messages: {status['message_count']}",
            f"- Workspace: `{status['workspace']}`",
            f"- Provider: {provider}",
            f"- Data loaded: {'✓' if status['has_data'] else '✗'}",
        ]

        # Add data summary if available
        if status['has_data'] and status.get('data_summary'):
            summary = status['data_summary']
            lines.append(f"- Data shape: {summary.get('shape', 'N/A')}")
            lines.append(f"- Memory usage: {summary.get('memory_usage', 'N/A')}")

        self._show_result("\n".join(lines))

    async def _cmd_status(self) -> None:
        """Show installation status (tier, packages, agents)."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        from lobster.core.license_manager import get_entitlement_status
        from lobster.core.plugin_loader import get_installed_packages
        from lobster.config.agent_registry import get_worker_agents
        from lobster.config.subscription_tiers import is_agent_available

        # Get entitlement status
        try:
            entitlement = get_entitlement_status()
        except ImportError:
            entitlement = {"tier": "free", "tier_display": "Free", "source": "default"}

        # Get installed packages
        try:
            packages = get_installed_packages()
        except ImportError:
            packages = {"lobster-ai": "unknown"}

        # Get available agents
        try:
            worker_agents = get_worker_agents()
            tier = entitlement.get("tier", "free")
            available = [name for name in worker_agents if is_agent_available(name, tier)]
            restricted = [name for name in worker_agents if not is_agent_available(name, tier)]
        except ImportError:
            available = []
            restricted = []

        # Build output
        tier_display = entitlement.get("tier_display", "Free")
        tier_emoji = {"free": "🆓", "premium": "⭐", "enterprise": "🏢"}.get(
            entitlement.get("tier", "free"), "🆓"
        )

        lines = [
            "**Installation Status:**\n",
            f"- Tier: {tier_emoji} {tier_display}",
            f"- Source: {entitlement.get('source', 'default')}",
        ]

        if entitlement.get("expires_at"):
            days = entitlement.get("days_until_expiry")
            if days is not None and days < 30:
                lines.append(f"- ⚠️ Expires in {days} days")
            else:
                lines.append(f"- Expires: {entitlement.get('expires_at')}")

        lines.append(f"\n**Installed Packages ({len(packages)}):**")
        for pkg_name, version in packages.items():
            status_icon = "✓" if version not in ["missing", "dev"] else ("✗" if version == "missing" else "⚡")
            lines.append(f"- {status_icon} {pkg_name}: {version}")

        if available:
            lines.append(f"\n**Available Agents ({len(available)}):**")
            for agent in sorted(available)[:5]:  # Show first 5
                lines.append(f"- {agent}")
            if len(available) > 5:
                lines.append(f"- ... and {len(available) - 5} more")

        if restricted:
            lines.append(f"\n**Premium Agents ({len(restricted)}):**")
            for agent in sorted(restricted)[:3]:  # Show first 3
                lines.append(f"- {agent} (upgrade required)")
            if len(restricted) > 3:
                lines.append(f"- ... and {len(restricted) - 3} more")

        self._show_result("\n".join(lines))

    async def _cmd_tokens(self) -> None:
        """Show token usage and costs."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        try:
            usage = self.client.get_token_usage()
            if not usage or "error" in usage:
                self.app.notify("Token tracking not available", severity="warning")
                return

            lines = [
                "**Token Usage:**\n",
                f"- Input tokens: {usage.get('total_input_tokens', 0):,}",
                f"- Output tokens: {usage.get('total_output_tokens', 0):,}",
                f"- Estimated cost: ${usage.get('total_cost', 0):.4f}",
            ]

            self._show_result("\n".join(lines))
        except Exception:
            self.app.notify("Token tracking not available", severity="warning")

    async def _cmd_workspace(self) -> None:
        """Show workspace info."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        workspace = self.client.workspace_path

        # Count files
        n_files = sum(1 for _ in workspace.rglob("*") if _.is_file())

        # Calculate size
        total_size = sum(f.stat().st_size for f in workspace.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)

        lines = [
            "**Workspace Info:**\n",
            f"- Path: `{workspace}`",
            f"- Files: {n_files}",
            f"- Size: {size_mb:.1f} MB",
        ]

        # Show subdirectories
        subdirs = [d.name for d in workspace.iterdir() if d.is_dir()]
        if subdirs:
            lines.append(f"- Directories: {', '.join(subdirs[:5])}")

        self._show_result("\n".join(lines))

    async def _cmd_save(self) -> None:
        """Save current session state."""
        if not self.client:
            self.app.notify("No client available", severity="error")
            return

        try:
            saved_items = self.client.data_manager.auto_save_state()
            if saved_items:
                self.app.notify(f"Session saved ({len(saved_items)} items)", timeout=2)
            else:
                self.app.notify("Nothing to save", severity="warning", timeout=2)
        except Exception as e:
            self.app.notify(f"Save failed: {e}", severity="error")

    async def _cmd_clear(self) -> None:
        """Clear results display."""
        from lobster.ui.widgets import ResultsDisplay
        try:
            results = self.app.query_one(ResultsDisplay)
            results.clear_display()
            self.app.notify("Cleared", timeout=1)
        except Exception:
            pass

    async def _cmd_refresh(self) -> None:
        """Refresh all panels."""
        screen = self.app.screen
        if hasattr(screen, "refresh_all"):
            screen.refresh_all()
        self.app.notify("Refreshed", timeout=1)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _show_result(self, content: str) -> None:
        """Display result in the results panel."""
        from lobster.ui.widgets import ResultsDisplay
        try:
            results = self.app.query_one(ResultsDisplay)
            results.append_system_message(content)
        except Exception:
            self.app.notify("Could not display result", severity="error")
