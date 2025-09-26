"""
Utilities module for the LOBSTER AI bioinformatics platform.

This module contains various utility functions and classes:
- Logging configuration
- JSON extraction utilities
- Agent node creation helpers
- Terminal callback handling
- Cross-platform system operations
"""

from .logger import get_logger
from .callbacks import TerminalCallbackHandler, SimpleTerminalCallback
from .progress_wrapper import with_periodic_progress, wrap_with_progress, ProgressContext
from .system import (
    IS_MACOS, IS_LINUX, IS_WINDOWS,
    get_platform, is_macos, is_linux, is_windows,
    open_file, open_folder, open_path
)

__all__ = [
    'get_logger',
    'TerminalCallbackHandler',
    'SimpleTerminalCallback',
    # Progress utilities
    'with_periodic_progress',
    'wrap_with_progress',
    'ProgressContext',
    # System utilities
    'IS_MACOS', 'IS_LINUX', 'IS_WINDOWS',
    'get_platform', 'is_macos', 'is_linux', 'is_windows',
    'open_file', 'open_folder', 'open_path'
]
