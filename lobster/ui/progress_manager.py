"""
Advanced progress tracking system for Lobster AI.

This module provides sophisticated progress tracking capabilities including
multi-task progress bars, download monitoring, analysis pipeline progress,
and parallel processing visualization with orange theming.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from threading import Lock

from rich.progress import (
    Progress, TaskID, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn,
    TransferSpeedColumn, DownloadColumn,
    MofNCompleteColumn
)
from rich.table import Table
from rich.live import Live
from rich.layout import Layout

from .themes import LobsterTheme
from .console_manager import get_console_manager


@dataclass
class TaskInfo:
    """Information about a progress task."""
    task_id: TaskID
    name: str
    description: str
    total: Optional[float]
    completed: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, paused
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressManager:
    """
    Advanced progress tracking manager with orange theming.

    Provides multi-task progress bars, download monitoring, and
    sophisticated progress visualization for Lobster AI operations.
    """

    def __init__(self, show_progress: bool = True):
        """
        Initialize progress manager.

        Args:
            show_progress: Whether to show progress bars by default
        """
        self.console_manager = get_console_manager()
        self.show_progress = show_progress
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_lock = Lock()

        # Create different progress instances for different contexts
        self._create_progress_instances()

    def _create_progress_instances(self):
        """Create specialized progress instances."""
        # General purpose progress
        self.general_progress = Progress(
            SpinnerColumn(style=LobsterTheme.PRIMARY_ORANGE),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                bar_width=None,
                complete_style=LobsterTheme.PRIMARY_ORANGE,
                finished_style=LobsterTheme.PRIMARY_ORANGE
            ),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console_manager.console,
            transient=True,
            expand=True
        )

        # Download progress with file size and speed
        self.download_progress = Progress(
            SpinnerColumn(style=LobsterTheme.PRIMARY_ORANGE),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                bar_width=None,
                complete_style=LobsterTheme.PRIMARY_ORANGE,
                finished_style=LobsterTheme.PRIMARY_ORANGE
            ),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=self.console_manager.console,
            transient=True,
            expand=True
        )

        # Analysis progress with step tracking
        self.analysis_progress = Progress(
            TextColumn("🧬 [progress.description]{task.description}"),
            BarColumn(
                bar_width=None,
                complete_style=LobsterTheme.PRIMARY_ORANGE,
                finished_style=LobsterTheme.PRIMARY_ORANGE
            ),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console_manager.console,
            transient=True,
            expand=True
        )

        # Batch processing progress
        self.batch_progress = Progress(
            SpinnerColumn(style=LobsterTheme.PRIMARY_ORANGE),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                bar_width=None,
                complete_style=LobsterTheme.PRIMARY_ORANGE,
                finished_style=LobsterTheme.PRIMARY_ORANGE
            ),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console_manager.console,
            transient=True,
            expand=True
        )

    def create_task(
        self,
        name: str,
        description: str,
        total: Optional[float] = None,
        progress_type: str = "general",
        **metadata
    ) -> str:
        """
        Create a new progress task.

        Args:
            name: Task name identifier
            description: Human-readable description
            total: Total work units (None for indeterminate progress)
            progress_type: Type of progress bar ("general", "download", "analysis", "batch")
            **metadata: Additional task metadata

        Returns:
            Task ID string
        """
        # Choose appropriate progress instance
        progress_instances = {
            "general": self.general_progress,
            "download": self.download_progress,
            "analysis": self.analysis_progress,
            "batch": self.batch_progress
        }

        progress = progress_instances.get(progress_type, self.general_progress)
        task_id = progress.add_task(description, total=total)

        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            name=name,
            description=description,
            total=total,
            metadata=metadata
        )

        with self.task_lock:
            unique_id = str(uuid.uuid4())
            self.tasks[unique_id] = task_info

        return unique_id

    def update_task(
        self,
        task_id: str,
        advance: Optional[float] = None,
        completed: Optional[float] = None,
        description: Optional[str] = None,
        **metadata
    ):
        """
        Update a progress task.

        Args:
            task_id: Task ID from create_task
            advance: Amount to advance progress
            completed: Set absolute completed amount
            description: Update task description
            **metadata: Update task metadata
        """
        with self.task_lock:
            if task_id not in self.tasks:
                return

            task_info = self.tasks[task_id]

            # Update progress in appropriate instance
            progress = self._get_progress_for_task(task_info)

            if advance is not None:
                progress.advance(task_info.task_id, advance)
                task_info.completed += advance
            elif completed is not None:
                current_completed = task_info.completed
                advance_amount = completed - current_completed
                if advance_amount > 0:
                    progress.advance(task_info.task_id, advance_amount)
                task_info.completed = completed

            if description is not None:
                progress.update(task_info.task_id, description=description)
                task_info.description = description

            # Update metadata
            task_info.metadata.update(metadata)

    def complete_task(self, task_id: str, success: bool = True):
        """
        Mark a task as completed.

        Args:
            task_id: Task ID from create_task
            success: Whether task completed successfully
        """
        with self.task_lock:
            if task_id not in self.tasks:
                return

            task_info = self.tasks[task_id]
            task_info.end_time = datetime.now()
            task_info.status = "completed" if success else "failed"

            # Complete progress
            progress = self._get_progress_for_task(task_info)
            if task_info.total:
                progress.update(task_info.task_id, completed=task_info.total)

            # Update description with completion status
            status_emoji = "✅" if success else "❌"
            status_text = "completed" if success else "failed"
            progress.update(
                task_info.task_id,
                description=f"{status_emoji} {task_info.description} ({status_text})"
            )

    def _get_progress_for_task(self, task_info: TaskInfo) -> Progress:
        """Get the appropriate progress instance for a task."""
        # This is a simplified approach - in a full implementation,
        # we'd track which progress instance each task belongs to
        return self.general_progress

    def create_download_tracker(
        self,
        filename: str,
        total_size: int,
        description: Optional[str] = None
    ) -> str:
        """
        Create a specialized download progress tracker.

        Args:
            filename: Name of file being downloaded
            total_size: Total file size in bytes
            description: Custom description (defaults to filename)

        Returns:
            Task ID string
        """
        if description is None:
            description = f"Downloading {filename}"

        task_id = self.download_progress.add_task(
            description,
            total=total_size,
            filename=filename
        )

        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            name=f"download_{filename}",
            description=description,
            total=total_size,
            metadata={"filename": filename, "type": "download"}
        )

        with self.task_lock:
            unique_id = str(uuid.uuid4())
            self.tasks[unique_id] = task_info

        return unique_id

    def update_download(
        self,
        task_id: str,
        bytes_downloaded: int,
        speed: Optional[int] = None
    ):
        """
        Update download progress.

        Args:
            task_id: Download task ID
            bytes_downloaded: Number of bytes downloaded
            speed: Download speed in bytes/sec
        """
        with self.task_lock:
            if task_id not in self.tasks:
                return

            task_info = self.tasks[task_id]
            advance_amount = bytes_downloaded - task_info.completed

            if advance_amount > 0:
                self.download_progress.advance(task_info.task_id, advance_amount)
                task_info.completed = bytes_downloaded

            # Update metadata
            if speed is not None:
                task_info.metadata["speed"] = speed

    def create_analysis_pipeline(
        self,
        pipeline_name: str,
        steps: List[str],
        description: Optional[str] = None
    ) -> str:
        """
        Create an analysis pipeline progress tracker.

        Args:
            pipeline_name: Name of the analysis pipeline
            steps: List of pipeline step names
            description: Custom description

        Returns:
            Task ID string
        """
        if description is None:
            description = f"Running {pipeline_name}"

        task_id = self.analysis_progress.add_task(
            description,
            total=len(steps)
        )

        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            name=pipeline_name,
            description=description,
            total=len(steps),
            metadata={
                "type": "analysis_pipeline",
                "steps": steps,
                "current_step": 0,
                "step_names": steps
            }
        )

        with self.task_lock:
            unique_id = str(uuid.uuid4())
            self.tasks[unique_id] = task_info

        return unique_id

    def advance_pipeline_step(
        self,
        task_id: str,
        step_name: Optional[str] = None,
        details: Optional[str] = None
    ):
        """
        Advance to the next step in an analysis pipeline.

        Args:
            task_id: Pipeline task ID
            step_name: Name of current step (for description update)
            details: Additional step details
        """
        with self.task_lock:
            if task_id not in self.tasks:
                return

            task_info = self.tasks[task_id]
            current_step = task_info.metadata.get("current_step", 0)

            # Advance progress
            self.analysis_progress.advance(task_info.task_id, 1)
            task_info.completed += 1
            task_info.metadata["current_step"] = current_step + 1

            # Update description with current step
            if step_name:
                description = f"🧬 {task_info.name}: {step_name}"
                if details:
                    description += f" - {details}"
                self.analysis_progress.update(
                    task_info.task_id,
                    description=description
                )

    @contextmanager
    def progress_context(self, progress_type: str = "general", clear_previous: bool = True):
        """
        Context manager for showing progress bars.

        Args:
            progress_type: Type of progress to show
            clear_previous: Whether to clear previous tasks from this progress type
        """
        progress_instances = {
            "general": self.general_progress,
            "download": self.download_progress,
            "analysis": self.analysis_progress,
            "batch": self.batch_progress
        }

        progress = progress_instances.get(progress_type, self.general_progress)

        # Clear previous tasks for this progress type if requested
        if clear_previous:
            try:
                task_ids_to_remove = list(progress.task_ids)
                for task_id in task_ids_to_remove:
                    try:
                        progress.remove_task(task_id)
                    except Exception:
                        pass
            except Exception:
                pass

        with progress:
            yield progress

    @contextmanager
    def multi_progress_context(self, progress_types: List[str] = None):
        """
        Context manager for showing multiple progress bars simultaneously.

        Args:
            progress_types: List of progress types to show
        """
        if progress_types is None:
            progress_types = ["general", "download", "analysis"]

        # Create layout for multiple progress bars
        layout = Layout()
        layout.split_column(*[Layout(name=ptype, size=3) for ptype in progress_types])

        # Create panel for each progress type
        panels = {}
        for ptype in progress_types:
            progress_instances = {
                "general": self.general_progress,
                "download": self.download_progress,
                "analysis": self.analysis_progress,
                "batch": self.batch_progress
            }
            progress = progress_instances.get(ptype, self.general_progress)
            panels[ptype] = progress

        # Start all progress instances
        for progress in panels.values():
            progress.start()

        try:
            with Live(layout, console=self.console_manager.console) as live:
                while True:
                    # Update layout with progress bars
                    for ptype, progress in panels.items():
                        layout[ptype].update(progress)
                    live.refresh()
                    time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            # Stop all progress instances
            for progress in panels.values():
                progress.stop()

    def get_task_summary(self) -> Table:
        """Get a summary table of all tasks."""
        table = Table(
            title="🦞 Task Summary",
            show_header=True,
            header_style="table.header",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            box=LobsterTheme.BOXES["primary"]
        )

        table.add_column("Task", style="data.key")
        table.add_column("Status", style="data.value")
        table.add_column("Progress", style="data.value")
        table.add_column("Duration", style="text.muted")

        with self.task_lock:
            for task_info in self.tasks.values():
                # Calculate duration
                if task_info.end_time:
                    duration = task_info.end_time - task_info.start_time
                else:
                    duration = datetime.now() - task_info.start_time

                duration_str = str(duration).split('.')[0]  # Remove microseconds

                # Format progress
                if task_info.total:
                    progress_pct = (task_info.completed / task_info.total) * 100
                    progress_str = f"{progress_pct:.1f}%"
                else:
                    progress_str = f"{task_info.completed} units"

                # Status styling
                status_styles = {
                    "running": "status.info",
                    "completed": "status.success",
                    "failed": "status.error",
                    "paused": "status.warning"
                }
                status_style = status_styles.get(task_info.status, "data.value")
                status_text = f"[{status_style}]{task_info.status.title()}[/{status_style}]"

                table.add_row(
                    task_info.name,
                    status_text,
                    progress_str,
                    duration_str
                )

        return table

    def clear_completed_tasks(self):
        """Remove completed and failed tasks."""
        with self.task_lock:
            to_remove = [
                task_id for task_id, task_info in self.tasks.items()
                if task_info.status in ["completed", "failed"]
            ]

            for task_id in to_remove:
                del self.tasks[task_id]

    def clear_all_progress_displays(self):
        """Clear all tasks from Rich Progress instances to prevent accumulation."""
        try:
            # Clear all tasks from each progress instance
            for progress_instance in [
                self.general_progress,
                self.download_progress,
                self.analysis_progress,
                self.batch_progress
            ]:
                # Get all task IDs and remove them
                task_ids_to_remove = list(progress_instance.task_ids)
                for task_id in task_ids_to_remove:
                    try:
                        progress_instance.remove_task(task_id)
                    except Exception:
                        # Ignore errors if task doesn't exist
                        pass
        except Exception:
            # Ignore any errors during cleanup
            pass

    def get_active_task_count(self) -> int:
        """Get count of active (running) tasks."""
        with self.task_lock:
            return sum(
                1 for task_info in self.tasks.values()
                if task_info.status == "running"
            )


# Global progress manager instance
_progress_manager: Optional[ProgressManager] = None


def get_progress_manager() -> ProgressManager:
    """Get the global progress manager instance."""
    global _progress_manager
    if _progress_manager is None:
        _progress_manager = ProgressManager()
    return _progress_manager


def create_simple_progress(description: str, total: Optional[float] = None) -> str:
    """Quick function to create a simple progress task."""
    return get_progress_manager().create_task(
        name=description,
        description=description,
        total=total
    )


def update_simple_progress(task_id: str, advance: float = 1.0):
    """Quick function to update a simple progress task."""
    get_progress_manager().update_task(task_id, advance=advance)


def complete_simple_progress(task_id: str, success: bool = True):
    """Quick function to complete a simple progress task."""
    get_progress_manager().complete_task(task_id, success=success)


def clear_progress_displays():
    """Quick function to clear all progress displays."""
    get_progress_manager().clear_all_progress_displays()