"""
Rich UI components and utilities for Lobster AI CLI.

This module provides advanced Rich-based UI components including:
- Themed console management with orange branding
- Live dashboard and monitoring displays
- Advanced progress tracking systems
- Professional styling and visual components
"""

from .themes import LobsterTheme
from .console_manager import get_console, setup_logging
from .live_dashboard import LiveDashboard
from .progress_manager import ProgressManager, get_progress_manager

__all__ = [
    'LobsterTheme',
    'get_console',
    'setup_logging',
    'LiveDashboard',
    'ProgressManager',
    'get_progress_manager'
]