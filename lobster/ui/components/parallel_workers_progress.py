"""Parallel workers progress display for Lobster AI.

This module provides a Rich-based progress display for parallel queue processing,
showing stacked progress bars for each worker and overall progress.

Follows the lobster/ui/ architecture with orange theming from LobsterTheme.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ..themes import LobsterTheme


@dataclass
class WorkerState:
    """State of a single worker in the parallel processing pool."""

    worker_id: int
    task_id: int  # Rich Progress TaskID
    current_entry: Optional[str] = None
    status: str = "idle"  # idle, processing, done
    elapsed: float = 0.0


class ParallelWorkersProgress:
    """
    Progress display for parallel workers processing a queue.

    Shows stacked progress bars:
    - One bar per worker showing current entry being processed
    - Overall progress bar at bottom showing total completion

    Thread-safe for concurrent worker updates via Lock().

    Example usage:
        >>> with ParallelWorkersProgress(num_workers=3, total_entries=10) as progress:
        ...     # In worker thread:
        ...     progress.worker_start(worker_id=0, entry_title="GSE12345")
        ...     # ... do work ...
        ...     progress.worker_complete(worker_id=0, status="completed", elapsed=5.2)
        ...     progress.worker_done(worker_id=0)  # when no more work
    """

    def __init__(self, num_workers: int, total_entries: int):
        """
        Initialize parallel workers progress display.

        Args:
            num_workers: Number of concurrent workers to display
            total_entries: Total number of entries to process
        """
        # Create a fresh Console for live progress display
        # Note: We don't use get_console_manager().console here because it has
        # record=True which interferes with Rich's live refresh mechanism
        self.console = Console(theme=LobsterTheme.RICH_THEME)
        self.num_workers = num_workers
        self.total_entries = total_entries
        self.completed_count = 0
        self.lock = Lock()
        self.workers: Dict[int, WorkerState] = {}
        self.overall_task: Optional[int] = None

        # Create Rich Progress with orange theming
        # transient=True ensures clean in-place updates without line duplication
        self.progress = Progress(
            SpinnerColumn(style=LobsterTheme.PRIMARY_ORANGE),
            TextColumn("{task.description}"),
            BarColumn(
                bar_width=20,
                complete_style=LobsterTheme.PRIMARY_ORANGE,
                finished_style=LobsterTheme.PRIMARY_ORANGE,
            ),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
            refresh_per_second=4,
        )

    def __enter__(self) -> "ParallelWorkersProgress":
        """Enter context: create worker tasks and overall progress bar."""
        # Create worker tasks (stacked progress bars)
        for i in range(self.num_workers):
            task_id = self.progress.add_task(
                f"[cyan]Worker {i+1}[/cyan] [dim]idle[/dim]",
                total=None,  # Indeterminate until assigned
            )
            self.workers[i] = WorkerState(worker_id=i, task_id=task_id)

        # Create overall task at bottom
        self.overall_task = self.progress.add_task(
            f"[bold green]Overall[/bold green] 0/{self.total_entries} entries",
            total=self.total_entries,
        )

        self.progress.start()
        return self

    def __exit__(self, *args):
        """Exit context: stop progress display."""
        self.progress.stop()

    def worker_start(self, worker_id: int, entry_title: str):
        """
        Called when worker starts processing an entry.

        Args:
            worker_id: Index of the worker (0-based)
            entry_title: Title/ID of entry being processed (truncated to 35 chars)
        """
        with self.lock:
            if worker_id not in self.workers:
                return

            worker = self.workers[worker_id]
            worker.current_entry = entry_title[:35]
            worker.status = "processing"
            self.progress.update(
                worker.task_id,
                description=f"[cyan]Worker {worker_id+1}[/cyan] {worker.current_entry}...",
                total=100,
                completed=0,
            )

    def worker_update_step(self, worker_id: int, step_name: str, percent: int):
        """
        Update worker progress with current step information.

        Called during entry processing to show intermediate progress.

        Args:
            worker_id: Index of the worker (0-based)
            step_name: Name of current step (e.g., "ncbi_enrich", "metadata")
            percent: Completion percentage (0-100)
        """
        with self.lock:
            if worker_id not in self.workers:
                return

            worker = self.workers[worker_id]
            entry_name = worker.current_entry or "..."
            # Truncate entry name to make room for step indicator
            entry_short = entry_name[:25] if len(entry_name) > 25 else entry_name
            self.progress.update(
                worker.task_id,
                completed=percent,
                description=f"[cyan]Worker {worker_id+1}[/cyan] [yellow][{step_name}][/yellow] {entry_short}",
            )

    def worker_complete(self, worker_id: int, status: str, elapsed: float):
        """
        Called when worker completes an entry.

        Args:
            worker_id: Index of the worker (0-based)
            status: Completion status (e.g., "completed", "failed", "paywalled")
            elapsed: Time taken in seconds
        """
        with self.lock:
            if worker_id not in self.workers:
                return

            worker = self.workers[worker_id]
            worker.status = status
            worker.elapsed = elapsed
            self.progress.update(
                worker.task_id,
                completed=100,
                description=f"[cyan]Worker {worker_id+1}[/cyan] [dim]{status}[/dim] ({elapsed:.1f}s)",
            )

            # Update overall progress
            self.completed_count += 1
            if self.overall_task is not None:
                self.progress.update(
                    self.overall_task,
                    completed=self.completed_count,
                    description=f"[bold green]Overall[/bold green] {self.completed_count}/{self.total_entries} entries",
                )

    def worker_done(self, worker_id: int):
        """
        Called when worker has no more entries to process.

        Args:
            worker_id: Index of the worker (0-based)
        """
        with self.lock:
            if worker_id not in self.workers:
                return

            worker = self.workers[worker_id]
            worker.status = "done"
            self.progress.update(
                worker.task_id,
                description=f"[cyan]Worker {worker_id+1}[/cyan] [green]done[/green]",
            )


@contextmanager
def parallel_workers_progress(num_workers: int, total_entries: int):
    """
    Context manager for parallel workers progress display.

    Convenience wrapper around ParallelWorkersProgress class.

    Args:
        num_workers: Number of concurrent workers to display
        total_entries: Total number of entries to process

    Yields:
        ParallelWorkersProgress: Progress manager for worker updates

    Example:
        >>> with parallel_workers_progress(3, 10) as progress:
        ...     progress.worker_start(0, "Processing entry 1")
        ...     progress.worker_complete(0, "completed", 2.5)
    """
    progress = ParallelWorkersProgress(num_workers, total_entries)
    with progress:
        yield progress


__all__ = [
    "ParallelWorkersProgress",
    "WorkerState",
    "parallel_workers_progress",
]
