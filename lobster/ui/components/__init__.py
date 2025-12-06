"""
Reusable Rich UI components with orange branding.

This module contains reusable UI components that follow the Lobster AI
design system with consistent orange (#e45c47) theming.
"""

from .file_tree import LobsterFileTree, create_file_tree, create_workspace_tree
from .multi_progress import (
    MultiTaskProgressManager,
    create_multi_progress_layout,
    get_multi_progress_manager,
    track_multi_task_operation,
)
from .parallel_workers_progress import (
    ParallelWorkersProgress,
    WorkerState,
    parallel_workers_progress,
)
from .status_display import (
    EnhancedStatusDisplay,
    create_analysis_dashboard,
    create_system_dashboard,
    create_workspace_dashboard,
    get_status_display,
)

__all__ = [
    "LobsterFileTree",
    "create_file_tree",
    "create_workspace_tree",
    "EnhancedStatusDisplay",
    "get_status_display",
    "create_system_dashboard",
    "create_workspace_dashboard",
    "create_analysis_dashboard",
    "MultiTaskProgressManager",
    "get_multi_progress_manager",
    "create_multi_progress_layout",
    "track_multi_task_operation",
    "ParallelWorkersProgress",
    "WorkerState",
    "parallel_workers_progress",
]
