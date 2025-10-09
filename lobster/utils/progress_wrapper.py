"""
Progress wrapper utility for long-running black box operations.

Provides periodic progress updates for operations that don't have built-in
progress reporting using a simple threading approach.
"""

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def with_periodic_progress(
    operation_name: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    update_interval: int = 15,
    show_elapsed: bool = True,
):
    """
    Context manager that provides periodic progress updates for black box operations.

    This utility wraps long-running operations (like scanpy functions) that don't
    provide built-in progress reporting. It starts a background thread that sends
    periodic "still working" messages to keep users informed.

    Args:
        operation_name: Name of the operation (e.g., "Finding marker genes")
        progress_callback: Optional callback function to receive progress messages.
                         Should accept a single string parameter.
        update_interval: Seconds between progress updates (default: 15)
        show_elapsed: Whether to include elapsed time in messages (default: True)

    Usage:
        with with_periodic_progress("Finding marker genes", callback, 15):
            sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")

    Example progress messages:
        - "Finding marker genes..."
        - "Still finding marker genes (30s elapsed)..."
        - "Still finding marker genes (1m 15s elapsed)..."
    """
    if not progress_callback:
        # If no callback provided, just log to debug
        def default_callback(message: str):
            logger.debug(f"Progress: {message}")

        progress_callback = default_callback

    # Thread synchronization
    stop_event = threading.Event()

    def format_elapsed_time(seconds: int) -> str:
        """Format elapsed time in a human-readable way."""
        if seconds < 60:
            return f"{seconds}s"
        else:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            if remaining_seconds == 0:
                return f"{minutes}m"
            else:
                return f"{minutes}m {remaining_seconds}s"

    def progress_updater():
        """Background thread function that sends periodic progress updates."""
        start_time = time.time()

        # Send initial message
        progress_callback(f"{operation_name}...")

        while not stop_event.is_set():
            # Wait for the update interval or until stop is signaled
            if stop_event.wait(update_interval):
                break  # Stop was signaled

            # Calculate elapsed time
            elapsed_seconds = int(time.time() - start_time)

            # Format progress message
            if show_elapsed and elapsed_seconds > 0:
                elapsed_str = format_elapsed_time(elapsed_seconds)
                message = f"Still {operation_name.lower()} ({elapsed_str} elapsed)..."
            else:
                message = f"Still {operation_name.lower()}..."

            # Send progress update
            progress_callback(message)

    # Start the progress thread
    progress_thread = threading.Thread(target=progress_updater, daemon=True)
    progress_thread.start()

    try:
        # Yield control to the caller - this is where the actual work happens
        yield

    finally:
        # Stop the progress thread
        stop_event.set()

        # Wait for the thread to finish (with a reasonable timeout)
        progress_thread.join(timeout=1.0)

        # Send completion message
        if progress_callback:
            progress_callback(f"{operation_name} completed")


def wrap_with_progress(
    func: Callable,
    operation_name: str,
    progress_callback: Optional[Callable[[str], None]] = None,
    update_interval: int = 15,
    show_elapsed: bool = True,
) -> Callable:
    """
    Decorator version of with_periodic_progress for wrapping functions.

    Args:
        func: The function to wrap
        operation_name: Name of the operation for progress messages
        progress_callback: Optional callback for progress updates
        update_interval: Seconds between updates
        show_elapsed: Whether to show elapsed time

    Returns:
        Wrapped function that will show progress during execution

    Usage:
        @wrap_with_progress(find_markers, "Finding marker genes", callback)
        def find_markers_with_progress(*args, **kwargs):
            return find_markers(*args, **kwargs)
    """

    def wrapper(*args, **kwargs):
        with with_periodic_progress(
            operation_name, progress_callback, update_interval, show_elapsed
        ):
            return func(*args, **kwargs)

    return wrapper


class ProgressContext:
    """
    Helper class for managing progress state in complex operations.

    This can be used when you need more control over progress reporting
    than the simple context manager provides.
    """

    def __init__(
        self,
        operation_name: str,
        progress_callback: Optional[Callable[[str], None]] = None,
        update_interval: int = 15,
    ):
        self.operation_name = operation_name
        self.progress_callback = progress_callback or (lambda x: None)
        self.update_interval = update_interval
        self.start_time = time.time()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the progress reporting."""
        if self._thread is not None:
            return  # Already started

        def progress_updater():
            self.progress_callback(f"{self.operation_name}...")

            while not self._stop_event.is_set():
                if self._stop_event.wait(self.update_interval):
                    break

                elapsed = int(time.time() - self.start_time)
                if elapsed > 0:
                    elapsed_str = (
                        f"{elapsed // 60}m {elapsed % 60}s"
                        if elapsed >= 60
                        else f"{elapsed}s"
                    )
                    message = f"Still {self.operation_name.lower()} ({elapsed_str} elapsed)..."
                else:
                    message = f"Still {self.operation_name.lower()}..."

                self.progress_callback(message)

        self._thread = threading.Thread(target=progress_updater, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the progress reporting."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._thread = None

        # Send completion message
        self.progress_callback(f"{self.operation_name} completed")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
