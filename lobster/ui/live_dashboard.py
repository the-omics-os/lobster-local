"""
Live dashboard system for real-time monitoring in Lobster AI.

This module provides real-time monitoring capabilities using Rich Live,
including multi-panel displays, system health monitoring, and analysis
progress tracking with orange theming.
"""

import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from threading import Thread, Event
import queue
from contextlib import contextmanager

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich import box

from .themes import LobsterTheme
from .console_manager import get_console_manager


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AnalysisStatus:
    """Analysis operation status."""
    operation_id: str
    operation_name: str
    status: str  # 'running', 'completed', 'failed', 'queued'
    progress: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    details: str = ""
    agent_name: str = ""


class LiveDashboard:
    """
    Real-time dashboard for monitoring Lobster AI operations.

    Provides live updating displays for system metrics, analysis progress,
    and multi-panel monitoring with orange theming.
    """

    def __init__(self, refresh_rate: float = 2.0):
        """
        Initialize live dashboard.

        Args:
            refresh_rate: Refresh rate in seconds for live updates
        """
        self.console_manager = get_console_manager()
        self.refresh_rate = refresh_rate

        # Dashboard state
        self.system_metrics: List[SystemMetrics] = []
        self.analysis_operations: Dict[str, AnalysisStatus] = {}
        self.is_monitoring = False
        self.stop_event = Event()
        self.monitor_thread: Optional[Thread] = None

        # Update queue for thread-safe communication
        self.update_queue = queue.Queue()

        # Dashboard layout
        self.layout = self._create_layout()

    def _create_layout(self) -> Layout:
        """Create the main dashboard layout."""
        layout = Layout()

        # Split into header and body
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body")
        )

        # Split body into left and right panels
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        # Split left panel into system and operations
        layout["left"].split_column(
            Layout(name="system", size=12),
            Layout(name="operations")
        )

        # Right panel for analysis details
        layout["right"].split_column(
            Layout(name="analysis", size=15),
            Layout(name="logs")
        )

        return layout

    def _get_header_panel(self) -> Panel:
        """Create header panel with branding."""
        title = Text()
        title.append("🦞 ", style="")
        title.append("Lobster AI Live Dashboard", style="lobster.primary")
        title.append(f" - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    style="text.secondary")

        return LobsterTheme.create_panel(
            Align.center(title),
            title="Real-time Monitoring",
            box_style="minimal"
        )

    def _get_system_panel(self) -> Panel:
        """Create system metrics panel."""
        if not self.system_metrics:
            return LobsterTheme.create_panel(
                "Collecting system metrics...",
                title="📊 System Health"
            )

        latest = self.system_metrics[-1]

        # System metrics table
        table = Table(
            show_header=False,
            box=box.SIMPLE,
            padding=(0, 1)
        )
        table.add_column("Metric", style="data.key")
        table.add_column("Value", style="data.value")
        table.add_column("Bar", width=20)

        # CPU usage
        cpu_bar = self._create_usage_bar(latest.cpu_percent)
        table.add_row("CPU Usage", f"{latest.cpu_percent:.1f}%", cpu_bar)

        # Memory usage
        memory_bar = self._create_usage_bar(latest.memory_percent)
        table.add_row(
            "Memory Usage",
            f"{latest.memory_used_gb:.1f}GB / {latest.memory_total_gb:.1f}GB ({latest.memory_percent:.1f}%)",
            memory_bar
        )

        # Disk usage
        disk_bar = self._create_usage_bar(latest.disk_percent)
        table.add_row("Disk Usage", f"{latest.disk_percent:.1f}%", disk_bar)

        return LobsterTheme.create_panel(
            table,
            title="📊 System Metrics"
        )

    def _create_usage_bar(self, percentage: float) -> str:
        """Create a text-based usage bar."""
        bar_length = 20
        filled = int((percentage / 100) * bar_length)

        # Color based on usage level
        if percentage < 50:
            style = "green"
        elif percentage < 80:
            style = "yellow"
        else:
            style = "red"

        bar = "█" * filled + "░" * (bar_length - filled)
        return f"[{style}]{bar}[/{style}]"

    def _get_operations_panel(self) -> Panel:
        """Create operations status panel."""
        if not self.analysis_operations:
            return LobsterTheme.create_panel(
                "No active operations",
                title="🔄 Active Operations"
            )

        # Operations table
        table = Table(
            show_header=True,
            header_style="table.header",
            box=LobsterTheme.BOXES["primary"]
        )

        table.add_column("Operation", style="data.key")
        table.add_column("Agent", style="agent.name")
        table.add_column("Status", style="data.value")
        table.add_column("Progress", width=15)
        table.add_column("Duration", style="text.muted")

        for op in self.analysis_operations.values():
            # Status styling
            status_style = {
                'running': 'status.info',
                'completed': 'status.success',
                'failed': 'status.error',
                'queued': 'status.warning'
            }.get(op.status, 'data.value')

            # Progress bar
            progress_bar = self._create_progress_bar(op.progress)

            # Duration calculation
            if op.end_time:
                duration = op.end_time - op.start_time
            else:
                duration = datetime.now() - op.start_time

            duration_str = str(duration).split('.')[0]  # Remove microseconds

            table.add_row(
                op.operation_name,
                op.agent_name or "System",
                f"[{status_style}]{op.status.title()}[/{status_style}]",
                progress_bar,
                duration_str
            )

        return LobsterTheme.create_panel(
            table,
            title="🔄 Operations Status"
        )

    def _create_progress_bar(self, progress: float) -> str:
        """Create a progress bar visualization."""
        bar_length = 15
        filled = int((progress / 100) * bar_length)

        orange_bar = "■" * filled
        empty_bar = "□" * (bar_length - filled)

        return f"[{LobsterTheme.PRIMARY_ORANGE}]{orange_bar}[/{LobsterTheme.PRIMARY_ORANGE}]{empty_bar} {progress:.0f}%"

    def _get_analysis_panel(self) -> Panel:
        """Create analysis details panel."""
        # Show details of the most recent operation
        if not self.analysis_operations:
            return LobsterTheme.create_panel(
                "No analysis operations to display",
                title="🧬 Analysis Details"
            )

        # Get most recent operation
        latest_op = max(
            self.analysis_operations.values(),
            key=lambda x: x.start_time
        )

        # Create details display
        details = []
        details.append(f"[data.key]Operation:[/] {latest_op.operation_name}")
        details.append(f"[data.key]Agent:[/] {latest_op.agent_name or 'System'}")
        details.append(f"[data.key]Status:[/] {latest_op.status.title()}")
        details.append(f"[data.key]Started:[/] {latest_op.start_time.strftime('%H:%M:%S')}")

        if latest_op.end_time:
            details.append(f"[data.key]Completed:[/] {latest_op.end_time.strftime('%H:%M:%S')}")

        if latest_op.details:
            details.append(f"[data.key]Details:[/] {latest_op.details}")

        content = "\n".join(details)

        return LobsterTheme.create_panel(
            content,
            title="🧬 Analysis Details"
        )

    def _get_logs_panel(self) -> Panel:
        """Create logs panel (placeholder for future enhancement)."""
        return LobsterTheme.create_panel(
            "Log integration coming soon...\n\nReal-time log streaming will be\navailable in the next update.",
            title="📋 Recent Logs"
        )

    def _update_layout(self):
        """Update all layout panels."""
        self.layout["header"].update(self._get_header_panel())
        self.layout["system"].update(self._get_system_panel())
        self.layout["operations"].update(self._get_operations_panel())
        self.layout["analysis"].update(self._get_analysis_panel())
        self.layout["logs"].update(self._get_logs_panel())

    def _collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()

            # Disk usage for current directory
            disk = psutil.disk_usage('.')

            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_percent=(disk.used / disk.total) * 100,
                timestamp=datetime.now()
            )

            # Keep only last 60 measurements (2 minutes at 2Hz)
            self.system_metrics.append(metrics)
            if len(self.system_metrics) > 60:
                self.system_metrics.pop(0)

        except Exception:
            # Fallback to dummy metrics if psutil fails
            self.system_metrics.append(SystemMetrics())

    def _monitor_loop(self):
        """Main monitoring loop running in separate thread."""
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Process any queued updates
                try:
                    while True:
                        update = self.update_queue.get_nowait()
                        self._process_update(update)
                except queue.Empty:
                    pass

                # Sleep until next update
                self.stop_event.wait(self.refresh_rate)

            except Exception:
                # Continue monitoring even if there are errors
                time.sleep(self.refresh_rate)

    def _process_update(self, update: Dict[str, Any]):
        """Process an update from the queue."""
        update_type = update.get('type')

        if update_type == 'analysis_start':
            self.analysis_operations[update['id']] = AnalysisStatus(
                operation_id=update['id'],
                operation_name=update['name'],
                status='running',
                agent_name=update.get('agent', ''),
                details=update.get('details', '')
            )

        elif update_type == 'analysis_progress':
            if update['id'] in self.analysis_operations:
                self.analysis_operations[update['id']].progress = update['progress']
                self.analysis_operations[update['id']].details = update.get('details', '')

        elif update_type == 'analysis_complete':
            if update['id'] in self.analysis_operations:
                self.analysis_operations[update['id']].status = 'completed'
                self.analysis_operations[update['id']].progress = 100.0
                self.analysis_operations[update['id']].end_time = datetime.now()

        elif update_type == 'analysis_failed':
            if update['id'] in self.analysis_operations:
                self.analysis_operations[update['id']].status = 'failed'
                self.analysis_operations[update['id']].end_time = datetime.now()

    @contextmanager
    def live_monitor(self):
        """Context manager for live monitoring."""
        self.start_monitoring()

        with Live(
            self.layout,
            console=self.console_manager.console,
            refresh_per_second=self.refresh_rate,
            auto_refresh=False
        ) as live:
            try:
                while self.is_monitoring:
                    self._update_layout()
                    live.refresh()
                    time.sleep(1 / self.refresh_rate)

            except KeyboardInterrupt:
                pass
            finally:
                self.stop_monitoring()

    def start_monitoring(self):
        """Start background monitoring."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.stop_event.clear()
            self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        if self.is_monitoring:
            self.is_monitoring = False
            self.stop_event.set()
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)

    def add_analysis_operation(
        self,
        operation_id: str,
        name: str,
        agent: str = "",
        details: str = ""
    ):
        """Add a new analysis operation to monitor."""
        self.update_queue.put({
            'type': 'analysis_start',
            'id': operation_id,
            'name': name,
            'agent': agent,
            'details': details
        })

    def update_analysis_progress(
        self,
        operation_id: str,
        progress: float,
        details: str = ""
    ):
        """Update progress for an analysis operation."""
        self.update_queue.put({
            'type': 'analysis_progress',
            'id': operation_id,
            'progress': progress,
            'details': details
        })

    def complete_analysis_operation(self, operation_id: str):
        """Mark an analysis operation as completed."""
        self.update_queue.put({
            'type': 'analysis_complete',
            'id': operation_id
        })

    def fail_analysis_operation(self, operation_id: str):
        """Mark an analysis operation as failed."""
        self.update_queue.put({
            'type': 'analysis_failed',
            'id': operation_id
        })

    def cleanup_old_operations(self, max_age_hours: int = 24):
        """Clean up old completed operations."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        to_remove = []
        for op_id, operation in self.analysis_operations.items():
            if (operation.status in ['completed', 'failed'] and
                operation.end_time and
                operation.end_time < cutoff_time):
                to_remove.append(op_id)

        for op_id in to_remove:
            del self.analysis_operations[op_id]


# Global dashboard instance
_dashboard: Optional[LiveDashboard] = None


def get_dashboard() -> LiveDashboard:
    """Get the global dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = LiveDashboard()
    return _dashboard


def create_simple_monitor(title: str = "Processing") -> Live:
    """Create a simple live monitor for basic operations."""
    console = get_console_manager().console

    spinner_text = Text()
    spinner_text.append("🦞 ", style="")
    spinner_text.append(title, style="lobster.primary")
    spinner_text.append("...", style="text.secondary")

    return Live(
        Align.center(spinner_text),
        console=console,
        refresh_per_second=2,
        transient=True
    )