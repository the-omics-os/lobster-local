"""
Utilities module for the Genie AI bioinformatics platform.

This module contains various utility functions and classes:
- Logging configuration
- JSON extraction utilities
- Agent node creation helpers
- Terminal callback handling
- Cross-platform system operations
"""

from .logger import get_logger
from .callbacks import TerminalCallbackHandler, SimpleTerminalCallback
from .system import (
    IS_MACOS, IS_LINUX, IS_WINDOWS,
    get_platform, is_macos, is_linux, is_windows,
    open_file, open_folder, open_path
)

__all__ = [
    'get_logger',
    'TerminalCallbackHandler',
    'SimpleTerminalCallback',
    # System utilities
    'IS_MACOS', 'IS_LINUX', 'IS_WINDOWS',
    'get_platform', 'is_macos', 'is_linux', 'is_windows',
    'open_file', 'open_folder', 'open_path'
]
