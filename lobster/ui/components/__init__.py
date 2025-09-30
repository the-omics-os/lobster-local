"""
Reusable Rich UI components with orange branding.

This module contains reusable UI components that follow the Lobster AI
design system with consistent orange (#e45c47) theming.
"""

from .file_tree import LobsterFileTree, create_file_tree, create_workspace_tree
from .status_display import (
    EnhancedStatusDisplay, get_status_display,
    create_system_dashboard, create_workspace_dashboard, create_analysis_dashboard
)
from .multi_progress import (
    MultiTaskProgressManager, get_multi_progress_manager,
    create_multi_progress_layout, track_multi_task_operation
)

__all__ = [
    'LobsterFileTree',
    'create_file_tree',
    'create_workspace_tree',
    'EnhancedStatusDisplay',
    'get_status_display',
    'create_system_dashboard',
    'create_workspace_dashboard',
    'create_analysis_dashboard',
    'MultiTaskProgressManager',
    'get_multi_progress_manager',
    'create_multi_progress_layout',
    'track_multi_task_operation'
]