"""
Advanced multi-task progress system for Lobster AI.

This module provides sophisticated multi-task progress tracking with
live updates, concurrent operation monitoring, and orange theming.
"""

import queue
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional, Set

from rich import box
from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    TaskID,
)
from rich.table import Table
from rich.text import Text

from ..console_manager import get_console_manager
from ..progress_manager import get_progress_manager
from ..themes import LobsterTheme


@dataclass
class MultiTaskOperation:
    """Information about a multi-task operation."""

    operation_id: str
    name: str
    description: str
    subtasks: List[str] = field(default_factory=list)
    completed_subtasks: Set[str] = field(default_factory=set)
    failed_subtasks: Set[str] = field(default_factory=set)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiTaskProgressManager:
    """
    Advanced multi-task progress manager with orange theming.

    Provides sophisticated tracking of concurrent operations,
    real-time updates, and professional progress visualization.
    """

    def __init__(self, refresh_rate: float = 2.0):
        """
        Initialize multi-task progress manager.

        Args:
            refresh_rate: Refresh rate in seconds for live updates
        """
        self.console_manager = get_console_manager()
        self.progress_manager = get_progress_manager()
        self.refresh_rate = refresh_rate

        # Multi-task state
        self.operations: Dict[str, MultiTaskOperation] = {}
        self.task_progress: Dict[str, float] = {}  # task_id -> progress percentage
        self.is_monitoring = False
        self.stop_event = Event()
        self.monitor_thread: Optional[Thread] = None
        self.operations_lock = Lock()

        # Update queue for thread-safe communication
        self.update_queue = queue.Queue()

        # Progress tracking
        self.active_progress_bars: Dict[str, TaskID] = {}

    def create_multi_operation(
        self, name: str, description: str, subtasks: List[str], **metadata
    ) -> str:
        """
        Create a new multi-task operation.

        Args:
            name: Operation name
            description: Human-readable description
            subtasks: List of subtask names
            **metadata: Additional operation metadata

        Returns:
            Operation ID string
        """
        operation_id = str(uuid.uuid4())

        operation = MultiTaskOperation(
            operation_id=operation_id,
            name=name,
            description=description,
            subtasks=subtasks.copy(),
            metadata=metadata,
        )

        with self.operations_lock:
            self.operations[operation_id] = operation

        # Initialize progress tracking for each subtask
        for subtask in subtasks:
            task_id = f"{operation_id}_{subtask}"
            self.task_progress[task_id] = 0.0

        return operation_id

    def update_subtask_progress(
        self,
        operation_id: str,
        subtask: str,
        progress: float,
        details: Optional[str] = None,
    ):
        """
        Update progress for a specific subtask.

        Args:
            operation_id: Operation ID
            subtask: Subtask name
            progress: Progress percentage (0-100)
            details: Optional progress details
        """
        task_id = f"{operation_id}_{subtask}"

        with self.operations_lock:
            if operation_id in self.operations:
                self.task_progress[task_id] = progress

                # Update operation metadata
                operation = self.operations[operation_id]
                operation.metadata[f"{subtask}_progress"] = progress
                if details:
                    operation.metadata[f"{subtask}_details"] = details

                # Check if subtask is completed
                if progress >= 100.0:
                    operation.completed_subtasks.add(subtask)

    def complete_subtask(
        self,
        operation_id: str,
        subtask: str,
        success: bool = True,
        result: Optional[str] = None,
    ):
        """
        Mark a subtask as completed.

        Args:
            operation_id: Operation ID
            subtask: Subtask name
            success: Whether subtask completed successfully
            result: Optional result description
        """
        with self.operations_lock:
            if operation_id in self.operations:
                operation = self.operations[operation_id]

                if success:
                    operation.completed_subtasks.add(subtask)
                    self.task_progress[f"{operation_id}_{subtask}"] = 100.0
                else:
                    operation.failed_subtasks.add(subtask)

                if result:
                    operation.metadata[f"{subtask}_result"] = result

                # Check if entire operation is complete
                total_subtasks = len(operation.subtasks)
                completed_or_failed = len(operation.completed_subtasks) + len(
                    operation.failed_subtasks
                )

                if completed_or_failed >= total_subtasks:
                    operation.end_time = datetime.now()
                    if len(operation.failed_subtasks) == 0:
                        operation.status = "completed"
                    else:
                        operation.status = "failed"

    def get_operation_progress(self, operation_id: str) -> float:
        """
        Get overall progress for an operation.

        Args:
            operation_id: Operation ID

        Returns:
            Overall progress percentage (0-100)
        """
        with self.operations_lock:
            if operation_id not in self.operations:
                return 0.0

            operation = self.operations[operation_id]
            if not operation.subtasks:
                return 100.0 if operation.status == "completed" else 0.0

            total_progress = 0.0
            for subtask in operation.subtasks:
                task_id = f"{operation_id}_{subtask}"
                total_progress += self.task_progress.get(task_id, 0.0)

            return total_progress / len(operation.subtasks)

    def create_progress_layout(self) -> Layout:
        """
        Create a comprehensive progress monitoring layout.

        Returns:
            Rich Layout with multi-task progress visualization
        """
        layout = Layout()

        # Split into header and body
        layout.split_column(Layout(name="header", size=3), Layout(name="body"))

        # Split body into operations and details
        layout["body"].split_row(Layout(name="operations"), Layout(name="details"))

        # Add header
        header_text = LobsterTheme.create_title_text(
            "Multi-Task Progress Monitor", "🦞"
        )
        layout["header"].update(
            LobsterTheme.create_panel(
                Align.center(header_text),
                title=f"Active Operations: {len([op for op in self.operations.values() if op.status == 'running'])}",
            )
        )

        # Add operations overview
        layout["operations"].update(self._create_operations_panel())

        # Add detailed progress
        layout["details"].update(self._create_details_panel())

        return layout

    def _create_operations_panel(self) -> Panel:
        """Create operations overview panel."""
        with self.operations_lock:
            if not self.operations:
                empty_text = Text()
                empty_text.append("No active operations\n\n", style="grey50")
                empty_text.append(
                    "Multi-task operations will\nappear here when started",
                    style="dim grey50",
                )
                return LobsterTheme.create_panel(
                    Align.center(empty_text), title="🔄 Operations Overview"
                )

            # Create operations table
            table = Table(
                show_header=True,
                header_style=f"bold {LobsterTheme.PRIMARY_ORANGE}",
                box=LobsterTheme.BOXES["primary"],
                border_style=LobsterTheme.PRIMARY_ORANGE,
            )

            table.add_column("Operation", style="white", width=20)
            table.add_column("Progress", style="data.value", width=15)
            table.add_column("Status", style="data.value", width=10)
            table.add_column("Duration", style="grey50", width=10)
            table.add_column("Subtasks", style="grey74", width=8)

            for operation in self.operations.values():
                # Overall progress
                overall_progress = self.get_operation_progress(operation.operation_id)
                progress_bar = self._create_progress_bar(overall_progress)

                # Status styling
                status_styles = {
                    "running": f"[{LobsterTheme.PRIMARY_ORANGE}]Running[/{LobsterTheme.PRIMARY_ORANGE}]",
                    "completed": "[green]Completed[/green]",
                    "failed": "[red]Failed[/red]",
                }
                status_text = status_styles.get(operation.status, operation.status)

                # Duration
                if operation.end_time:
                    duration = operation.end_time - operation.start_time
                else:
                    duration = datetime.now() - operation.start_time

                duration_str = str(duration).split(".")[0]  # Remove microseconds

                # Subtasks summary
                completed = len(operation.completed_subtasks)
                failed = len(operation.failed_subtasks)
                total = len(operation.subtasks)
                subtasks_str = f"{completed}/{total}"
                if failed > 0:
                    subtasks_str += f" ({failed}❌)"

                # Operation name (truncated)
                op_name = operation.name
                if len(op_name) > 18:
                    op_name = op_name[:15] + "..."

                table.add_row(
                    op_name, progress_bar, status_text, duration_str, subtasks_str
                )

        return LobsterTheme.create_panel(table, title="🔄 Operations Overview")

    def _create_details_panel(self) -> Panel:
        """Create detailed progress panel for the most recent operation."""
        with self.operations_lock:
            if not self.operations:
                return LobsterTheme.create_panel(
                    "No operations to display", title="📊 Operation Details"
                )

            # Get most recent operation
            latest_operation = max(self.operations.values(), key=lambda x: x.start_time)

            # Create subtasks table
            table = Table(
                show_header=True,
                header_style=f"bold {LobsterTheme.PRIMARY_ORANGE}",
                box=box.SIMPLE,
                padding=(0, 1),
            )

            table.add_column("Subtask", style="white", width=18)
            table.add_column("Progress", style="data.value", width=12)
            table.add_column("Status", style="data.value", width=10)
            table.add_column("Details", style="grey50", width=15)

            for subtask in latest_operation.subtasks:
                task_id = f"{latest_operation.operation_id}_{subtask}"
                progress = self.task_progress.get(task_id, 0.0)

                # Progress bar
                progress_bar = self._create_mini_progress_bar(progress)

                # Status
                if subtask in latest_operation.completed_subtasks:
                    status = "[green]✅ Done[/green]"
                elif subtask in latest_operation.failed_subtasks:
                    status = "[red]❌ Failed[/red]"
                elif progress > 0:
                    status = f"[{LobsterTheme.PRIMARY_ORANGE}]🔄 Running[/{LobsterTheme.PRIMARY_ORANGE}]"
                else:
                    status = "[grey50]⏳ Pending[/grey50]"

                # Details
                details = latest_operation.metadata.get(f"{subtask}_details", "")
                if len(details) > 13:
                    details = details[:10] + "..."

                table.add_row(subtask, progress_bar, status, details)

            # Add operation info at the top
            op_info = f"[bold white]{latest_operation.name}[/bold white]\n"
            op_info += f"[grey50]{latest_operation.description}[/grey50]"

            content = Group(Text(op_info, justify="left"), Text(""), table)  # Spacer

        return LobsterTheme.create_panel(content, title="📊 Operation Details")

    def _create_progress_bar(self, percentage: float) -> str:
        """Create a progress bar visualization."""
        bar_length = 12
        filled = int((percentage / 100) * bar_length)

        orange_bar = "█" * filled
        empty_bar = "░" * (bar_length - filled)

        return f"[{LobsterTheme.PRIMARY_ORANGE}]{orange_bar}[/{LobsterTheme.PRIMARY_ORANGE}]{empty_bar} {percentage:.0f}%"

    def _create_mini_progress_bar(self, percentage: float) -> str:
        """Create a mini progress bar for subtasks."""
        bar_length = 8
        filled = int((percentage / 100) * bar_length)

        orange_bar = "■" * filled
        empty_bar = "□" * (bar_length - filled)

        return f"[{LobsterTheme.PRIMARY_ORANGE}]{orange_bar}[/{LobsterTheme.PRIMARY_ORANGE}]{empty_bar}"

    @contextmanager
    def live_progress_monitor(self):
        """Context manager for live progress monitoring."""
        layout = self.create_progress_layout()

        with Live(
            layout,
            console=self.console_manager.console,
            refresh_per_second=self.refresh_rate,
            auto_refresh=True,
        ) as live:
            self.start_monitoring()
            try:
                while self.is_monitoring:
                    # Update layout
                    updated_layout = self.create_progress_layout()
                    live.update(updated_layout)
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

    def stop_monitoring(self):
        """Stop background monitoring."""
        if self.is_monitoring:
            self.is_monitoring = False
            self.stop_event.set()

    def get_active_operations_count(self) -> int:
        """Get count of active operations."""
        with self.operations_lock:
            return len(
                [op for op in self.operations.values() if op.status == "running"]
            )

    def cleanup_completed_operations(self, max_age_hours: int = 24):
        """Clean up old completed operations."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        with self.operations_lock:
            to_remove = []
            for op_id, operation in self.operations.items():
                if (
                    operation.status in ["completed", "failed"]
                    and operation.end_time
                    and operation.end_time < cutoff_time
                ):
                    to_remove.append(op_id)

            for op_id in to_remove:
                # Clean up associated task progress
                operation = self.operations[op_id]
                for subtask in operation.subtasks:
                    task_id = f"{op_id}_{subtask}"
                    self.task_progress.pop(task_id, None)

                del self.operations[op_id]

    @contextmanager
    def track_multi_operation(
        self, name: str, description: str, subtasks: List[str], **metadata
    ):
        """
        Context manager for tracking a multi-task operation.

        Args:
            name: Operation name
            description: Operation description
            subtasks: List of subtask names
            **metadata: Additional metadata

        Yields:
            Operation ID for progress updates
        """
        operation_id = self.create_multi_operation(
            name, description, subtasks, **metadata
        )

        try:
            yield operation_id
        finally:
            # Ensure operation is marked as completed if not already
            with self.operations_lock:
                if operation_id in self.operations:
                    operation = self.operations[operation_id]
                    if operation.status == "running":
                        operation.status = "completed"
                        operation.end_time = datetime.now()


# Global multi-task progress manager instance
_multi_progress_manager: Optional[MultiTaskProgressManager] = None


def get_multi_progress_manager() -> MultiTaskProgressManager:
    """Get the global multi-task progress manager instance."""
    global _multi_progress_manager
    if _multi_progress_manager is None:
        _multi_progress_manager = MultiTaskProgressManager()
    return _multi_progress_manager


def create_multi_progress_layout() -> Layout:
    """Quick function to create multi-task progress layout."""
    return get_multi_progress_manager().create_progress_layout()


@contextmanager
def track_multi_task_operation(
    name: str, description: str, subtasks: List[str], **metadata
):
    """
    Quick context manager for tracking multi-task operations.

    Args:
        name: Operation name
        description: Operation description
        subtasks: List of subtask names
        **metadata: Additional metadata

    Yields:
        Tuple of (operation_id, progress_manager)
    """
    manager = get_multi_progress_manager()
    with manager.track_multi_operation(
        name, description, subtasks, **metadata
    ) as operation_id:
        yield operation_id, manager
