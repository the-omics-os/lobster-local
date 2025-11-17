"""
Enhanced status display components with Rich Layout for Lobster AI.

This module provides sophisticated status displays using Rich Layout,
multi-panel views, and real-time system monitoring with orange theming.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
from rich import box
from rich.align import Align
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from ..console_manager import get_console_manager
from ..themes import LobsterTheme


class EnhancedStatusDisplay:
    """
    Enhanced status display system with Rich Layout and orange theming.

    Provides multi-panel status views, system health monitoring,
    and real-time workspace status with professional styling.
    """

    def __init__(self):
        """Initialize enhanced status display."""
        self.console_manager = get_console_manager()

    def create_system_health_layout(self, client) -> Layout:
        """
        Create a comprehensive system health layout.

        Args:
            client: AgentClient instance for system data

        Returns:
            Rich Layout with multi-panel system status
        """
        layout = Layout()

        # Split into header and body
        layout.split_column(Layout(name="header", size=3), Layout(name="body"))

        # Split body into left, center, and right panels
        layout["body"].split_row(
            Layout(name="left"), Layout(name="center"), Layout(name="right")
        )

        # Add header
        header_text = LobsterTheme.create_title_text("System Health Dashboard", "🦞")
        layout["header"].update(
            LobsterTheme.create_panel(
                Align.center(header_text),
                title=f"Updated: {datetime.now().strftime('%H:%M:%S')}",
            )
        )

        # Left panel: Core system status
        layout["left"].update(self._create_core_status_panel(client))

        # Center panel: Resource usage
        layout["center"].update(self._create_resource_panel())

        # Right panel: Agent status
        layout["right"].update(self._create_agent_status_panel(client))

        return layout

    def _create_core_status_panel(self, client) -> Panel:
        """Create core system status panel."""
        status = client.get_status()

        # Create status table
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Property", style="data.key", width=15)
        table.add_column("Value", style="data.value")
        table.add_column("Status", width=3)

        # Session info
        table.add_row("Session ID", status["session_id"][:8] + "...", "🆔")
        table.add_row("Messages", str(status["message_count"]), "💬")
        table.add_row("Workspace", Path(status["workspace"]).name, "📁")

        # Data status
        data_status = "✅" if status["has_data"] else "❌"
        data_text = "Loaded" if status["has_data"] else "None"
        table.add_row("Data", data_text, data_status)

        if status["has_data"] and status.get("data_summary"):
            summary = status["data_summary"]
            if summary.get("shape"):
                shape_text = f"{summary['shape'][0]} × {summary['shape'][1]}"
                table.add_row("Data Shape", shape_text, "📊")
            if summary.get("memory_usage"):
                table.add_row("Memory", summary["memory_usage"], "💾")

        return LobsterTheme.create_panel(table, title="📋 Core Status")

    def _create_resource_panel(self) -> Panel:
        """Create system resource monitoring panel."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(".")

            # Create resource table
            table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
            table.add_column("Resource", style="data.key", width=12)
            table.add_column("Usage", style="data.value", width=15)
            table.add_column("Bar", width=20)

            # CPU usage
            cpu_bar = self._create_usage_bar(cpu_percent)
            table.add_row("CPU", f"{cpu_percent:.1f}%", cpu_bar)

            # Memory usage
            memory_bar = self._create_usage_bar(memory.percent)
            memory_text = (
                f"{memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB"
            )
            table.add_row("Memory", memory_text, memory_bar)

            # Disk usage
            disk_percent = (disk.used / disk.total) * 100
            disk_bar = self._create_usage_bar(disk_percent)
            table.add_row("Disk", f"{disk_percent:.1f}%", disk_bar)

            # Process info
            current_process = psutil.Process()
            process_memory = current_process.memory_info().rss / (1024**2)  # MB
            table.add_row("Process", f"{process_memory:.1f}MB", "🦞")

        except Exception:
            # Fallback content if psutil fails
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_column("Info", style="data.value")
            table.add_row("Resource monitoring")
            table.add_row("unavailable")

        return LobsterTheme.create_panel(table, title="⚡ Resources")

    def _create_agent_status_panel(self, client) -> Panel:
        """Create agent status panel."""
        # Get current mode/profile
        try:
            from lobster.config.agent_config import get_agent_configurator

            configurator = get_agent_configurator()
            current_mode = configurator.get_current_profile()
        except Exception:
            current_mode = "unknown"

        # Create agent status table
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column("Agent", style="data.key", width=15)
        table.add_column("Status", style="data.value")
        table.add_column("Icon", width=3)

        # Mode info
        table.add_row("Mode", current_mode, "⚙️")

        # Active agents (simplified)
        agents = [
            ("Data Expert", "Ready", "📊"),
            ("SingleCell Expert", "Ready", "🧬"),
            ("Proteomics Expert", "Ready", "🔬"),
            ("Research Agent", "Ready", "📚"),
            ("Supervisor", "Active", "🎯"),
        ]

        for agent_name, status, icon in agents:
            status_style = (
                f"[{LobsterTheme.PRIMARY_ORANGE}]" if status == "Active" else "[green]"
            )
            table.add_row(
                agent_name, f"{status_style}{status}[/{status_style.strip('[]')}]", icon
            )

        return LobsterTheme.create_panel(table, title="🤖 Agents")

    def _create_usage_bar(self, percentage: float) -> str:
        """Create a colored usage bar."""
        bar_length = 20
        filled = int((percentage / 100) * bar_length)

        # Orange-themed color progression
        if percentage < 50:
            color = "green"
        elif percentage < 80:
            color = LobsterTheme.PRIMARY_ORANGE
        else:
            color = "red"

        bar = "█" * filled + "░" * (bar_length - filled)
        return f"[{color}]{bar}[/{color}]"

    def create_workspace_overview_layout(self, client) -> Layout:
        """
        Create workspace overview with file status and analysis progress.

        Args:
            client: AgentClient instance

        Returns:
            Rich Layout with workspace overview
        """
        layout = Layout()

        # Split into sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="workspace_info", size=8),
            Layout(name="files_and_data"),
        )

        # Header
        header_text = LobsterTheme.create_title_text("Workspace Overview", "🏗️")
        layout["header"].update(
            LobsterTheme.create_panel(
                Align.center(header_text), title="Workspace Status"
            )
        )

        # Workspace info
        layout["workspace_info"].update(self._create_workspace_info_panel(client))

        # Split files and data section
        layout["files_and_data"].split_row(
            Layout(name="files"), Layout(name="data_status")
        )

        layout["files"].update(self._create_recent_files_panel(client))
        layout["data_status"].update(self._create_data_status_panel(client))

        return layout

    def _create_workspace_info_panel(self, client) -> Panel:
        """Create workspace information panel."""
        try:
            if hasattr(client.data_manager, "get_workspace_status"):
                workspace_status = client.data_manager.get_workspace_status()

                # Create workspace table
                table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
                table.add_column("Property", style="data.key", width=18)
                table.add_column("Value", style="data.value")
                table.add_column("Icon", width=3)

                table.add_row(
                    "Workspace Path",
                    str(Path(workspace_status.get("workspace_path", "")).name),
                    "📁",
                )
                table.add_row(
                    "Modalities",
                    str(workspace_status.get("modalities_loaded", 0)),
                    "🧬",
                )
                table.add_row(
                    "Backends",
                    str(len(workspace_status.get("registered_backends", []))),
                    "💾",
                )
                table.add_row(
                    "Adapters",
                    str(len(workspace_status.get("registered_adapters", []))),
                    "🔌",
                )

                provenance = (
                    "✅" if workspace_status.get("provenance_enabled") else "❌"
                )
                table.add_row("Provenance", provenance, "📋")

                mudata = "✅" if workspace_status.get("mudata_available") else "❌"
                table.add_row("MuData Support", mudata, "🔗")

            else:
                # Fallback for older data manager
                table = Table(show_header=False, box=box.SIMPLE)
                table.add_column("Info", style="data.value")
                table.add_row("Basic workspace")
                table.add_row("information available")

        except Exception:
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_column("Error", style="red")
            table.add_row("Unable to load")
            table.add_row("workspace status")

        return LobsterTheme.create_panel(table, title="🏗️ Workspace Info")

    def _create_recent_files_panel(self, client) -> Panel:
        """Create recent files panel."""
        try:
            workspace_files = client.data_manager.list_workspace_files()

            # Collect all files and sort by modification time
            all_files = []
            for category, files in workspace_files.items():
                for file_info in files:
                    all_files.append((file_info, category))

            # Sort by modification time (newest first)
            all_files.sort(key=lambda x: x[0]["modified"], reverse=True)

            # Create files table
            table = Table(
                show_header=True,
                header_style=f"bold {LobsterTheme.PRIMARY_ORANGE}",
                box=box.SIMPLE,
                padding=(0, 1),
            )
            table.add_column("File", style="white", width=20)
            table.add_column("Type", style="cyan", width=8)
            table.add_column("Size", style="grey74", width=8)

            # Show top 5 files
            for (file_info, category), _ in zip(all_files[:5], range(5)):
                file_name = file_info["name"]
                if len(file_name) > 18:
                    file_name = file_name[:15] + "..."

                size_kb = file_info["size"] / 1024
                size_str = (
                    f"{size_kb:.1f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
                )

                # Category icon
                category_icons = {
                    "bioinformatics": "🧬",
                    "tabular": "📊",
                    "visualization": "📈",
                    "exports": "📤",
                }
                category_display = (
                    category_icons.get(category, "📄") + " " + category.title()[:6]
                )

                table.add_row(file_name, category_display, size_str)

            if len(all_files) > 5:
                table.add_row("...", f"+ {len(all_files) - 5} more", "")

        except Exception:
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_column("Info", style="grey50")
            table.add_row("No files found")

        return LobsterTheme.create_panel(table, title="📁 Recent Files")

    def _create_data_status_panel(self, client) -> Panel:
        """Create data status panel."""
        try:
            if client.data_manager.has_data():
                summary = client.data_manager.get_data_summary()

                # Create data table
                table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
                table.add_column("Property", style="data.key", width=12)
                table.add_column("Value", style="data.value")
                table.add_column("Icon", width=3)

                # Data shape
                if summary.get("shape"):
                    shape = summary["shape"]
                    table.add_row("Shape", f"{shape[0]:,} × {shape[1]:,}", "📊")

                # Memory usage
                if summary.get("memory_usage"):
                    table.add_row("Memory", summary["memory_usage"], "💾")

                # Data type
                if summary.get("data_type"):
                    table.add_row("Type", summary["data_type"], "🧬")

                # Modality info
                if summary.get("modality_name"):
                    modality = summary["modality_name"]
                    if len(modality) > 15:
                        modality = modality[:12] + "..."
                    table.add_row("Modality", modality, "🔬")

                # Processing status
                if summary.get("processing_log"):
                    recent_count = len(summary["processing_log"])
                    table.add_row("Operations", str(recent_count), "⚙️")

            else:
                table = Table(show_header=False, box=box.SIMPLE)
                table.add_column("Status", style="grey50")
                table.add_row("No data loaded")
                table.add_row("Use /read to load data")

        except Exception:
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_column("Error", style="red")
            table.add_row("Data status unavailable")

        return LobsterTheme.create_panel(table, title="📊 Data Status")

    def create_analysis_dashboard(self, client) -> Layout:
        """
        Create analysis progress dashboard.

        Args:
            client: AgentClient instance

        Returns:
            Rich Layout with analysis monitoring
        """
        layout = Layout()

        # Split into header and content
        layout.split_column(Layout(name="header", size=3), Layout(name="content"))

        # Split content into analysis and plots
        layout["content"].split_row(Layout(name="analysis"), Layout(name="plots"))

        # Header
        header_text = LobsterTheme.create_title_text("Analysis Dashboard", "🧬")
        layout["header"].update(
            LobsterTheme.create_panel(
                Align.center(header_text), title="Real-time Analysis Monitoring"
            )
        )

        # Analysis panel
        layout["analysis"].update(self._create_analysis_panel(client))

        # Plots panel
        layout["plots"].update(self._create_plots_panel(client))

        return layout

    def _create_analysis_panel(self, client) -> Panel:
        """Create analysis operations panel."""
        try:
            # Get tool usage history if available
            if hasattr(client.data_manager, "tool_usage_history"):
                history = client.data_manager.tool_usage_history[
                    -10:
                ]  # Last 10 operations

                table = Table(
                    show_header=True,
                    header_style=f"bold {LobsterTheme.PRIMARY_ORANGE}",
                    box=box.SIMPLE,
                    padding=(0, 1),
                )
                table.add_column("Operation", style="white", width=15)
                table.add_column("Status", style="data.value", width=10)
                table.add_column("Time", style="grey50", width=8)

                for operation in history:
                    op_name = operation.get("tool_name", "Unknown")
                    if len(op_name) > 13:
                        op_name = op_name[:10] + "..."

                    # Format timestamp
                    timestamp = operation.get("timestamp", "")
                    if timestamp:
                        try:
                            time_str = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            ).strftime("%H:%M")
                        except Exception:
                            time_str = "N/A"
                    else:
                        time_str = "N/A"

                    success = operation.get("success", True)
                    status = (
                        "[green]✅ Success[/green]"
                        if success
                        else "[red]❌ Failed[/red]"
                    )

                    table.add_row(op_name, status, time_str)

            else:
                table = Table(show_header=False, box=box.SIMPLE)
                table.add_column("Info", style="grey50")
                table.add_row("No recent operations")
                table.add_row("Start an analysis to")
                table.add_row("see activity here")

        except Exception:
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_column("Error", style="red")
            table.add_row("Analysis history")
            table.add_row("unavailable")

        return LobsterTheme.create_panel(table, title="⚙️ Recent Operations")

    def _create_plots_panel(self, client) -> Panel:
        """Create plots status panel."""
        try:
            plots = client.data_manager.get_plot_history()

            if plots:
                table = Table(
                    show_header=True,
                    header_style=f"bold {LobsterTheme.PRIMARY_ORANGE}",
                    box=box.SIMPLE,
                    padding=(0, 1),
                )
                table.add_column("Plot", style="white", width=15)
                table.add_column("Type", style="cyan", width=8)
                table.add_column("Created", style="grey50", width=8)

                # Show last 5 plots
                for plot in plots[-5:]:
                    plot_title = plot["title"]
                    if len(plot_title) > 13:
                        plot_title = plot_title[:10] + "..."

                    plot_type = "📈 Plot"  # Could be enhanced to detect plot type

                    # Format timestamp
                    try:
                        created = datetime.fromisoformat(
                            plot["timestamp"].replace("Z", "+00:00")
                        )
                        created_str = created.strftime("%H:%M")
                    except Exception:
                        created_str = "N/A"

                    table.add_row(plot_title, plot_type, created_str)

                if len(plots) > 5:
                    table.add_row("...", f"+ {len(plots) - 5} more", "")

            else:
                table = Table(show_header=False, box=box.SIMPLE)
                table.add_column("Info", style="grey50")
                table.add_row("No plots generated")
                table.add_row("Run analysis to")
                table.add_row("create visualizations")

        except Exception:
            table = Table(show_header=False, box=box.SIMPLE)
            table.add_column("Error", style="red")
            table.add_row("Plot history")
            table.add_row("unavailable")

        return LobsterTheme.create_panel(table, title="📈 Visualizations")


# Global enhanced status display instance
_status_display: Optional[EnhancedStatusDisplay] = None


def get_status_display() -> EnhancedStatusDisplay:
    """Get the global enhanced status display instance."""
    global _status_display
    if _status_display is None:
        _status_display = EnhancedStatusDisplay()
    return _status_display


def create_system_dashboard(client) -> Layout:
    """Quick function to create system health dashboard."""
    return get_status_display().create_system_health_layout(client)


def create_workspace_dashboard(client) -> Layout:
    """Quick function to create workspace overview dashboard."""
    return get_status_display().create_workspace_overview_layout(client)


def create_analysis_dashboard(client) -> Layout:
    """Quick function to create analysis dashboard."""
    return get_status_display().create_analysis_dashboard(client)
