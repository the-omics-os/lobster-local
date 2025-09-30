"""
System utilities for cross-platform operations.

This module provides centralized system platform detection and file/folder
opening functionality to eliminate redundant platform.system() calls and
duplicate subprocess logic across the codebase.

Cloud-agnostic design: All operations run on the CLI side regardless of
whether agents are local or remote.
"""

import platform
import subprocess
from pathlib import Path
from typing import Tuple

# Initialize platform detection once at module import
_PLATFORM = platform.system()
IS_MACOS = _PLATFORM == "Darwin"
IS_LINUX = _PLATFORM == "Linux"
IS_WINDOWS = _PLATFORM == "Windows"


def get_platform() -> str:
    """
    Get the platform string.

    Returns:
        str: Platform string - 'Darwin', 'Linux', or 'Windows'
    """
    return _PLATFORM


def is_macos() -> bool:
    """Check if running on macOS."""
    return IS_MACOS


def is_linux() -> bool:
    """Check if running on Linux."""
    return IS_LINUX


def is_windows() -> bool:
    """Check if running on Windows."""
    return IS_WINDOWS


def open_file(file_path: Path) -> Tuple[bool, str]:
    """
    Open a file in the system's default application.

    Args:
        file_path: Path to the file to open

    Returns:
        Tuple[bool, str]: (success, message) - success boolean and status message
    """
    try:
        if IS_MACOS:
            subprocess.run(["open", str(file_path)], check=True)
            return True, f"Opened file: {file_path.name}"
        elif IS_LINUX:
            subprocess.run(["xdg-open", str(file_path)], check=True)
            return True, f"Opened file: {file_path.name}"
        elif IS_WINDOWS:
            subprocess.run(["start", "", str(file_path)], shell=True, check=True)
            return True, f"Opened file: {file_path.name}"
        else:
            return False, f"Unsupported operating system: {_PLATFORM}"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to open file '{file_path.name}': {e}"
    except Exception as e:
        return False, f"Error opening file '{file_path.name}': {e}"


def open_folder(folder_path: Path) -> Tuple[bool, str]:
    """
    Open a folder in the system's file manager.

    Args:
        folder_path: Path to the folder to open

    Returns:
        Tuple[bool, str]: (success, message) - success boolean and status message
    """
    try:
        if IS_MACOS:
            subprocess.run(["open", str(folder_path)], check=True)
            return True, f"Opened folder in Finder: {folder_path.name}"
        elif IS_LINUX:
            # Try common file managers in order of preference
            file_managers = ["xdg-open", "nautilus", "dolphin", "thunar", "pcmanfm"]
            for fm in file_managers:
                try:
                    subprocess.run([fm, str(folder_path)], check=True, stderr=subprocess.DEVNULL)
                    return True, f"Opened folder: {folder_path.name}"
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            return False, f"Could not find a suitable file manager to open: {folder_path.name}"
        elif IS_WINDOWS:
            subprocess.run(["explorer", str(folder_path)], check=True)
            return True, f"Opened folder in Explorer: {folder_path.name}"
        else:
            return False, f"Unsupported operating system: {_PLATFORM}"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to open folder '{folder_path.name}': {e}"
    except Exception as e:
        return False, f"Error opening folder '{folder_path.name}': {e}"


def open_path(path: Path) -> Tuple[bool, str]:
    """
    Open a file or folder in the appropriate system application.

    Automatically detects whether the path is a file or folder and calls
    the appropriate opening function.

    Args:
        path: Path to the file or folder to open

    Returns:
        Tuple[bool, str]: (success, message) - success boolean and status message
    """
    if not path.exists():
        return False, f"Path does not exist: {path}"

    if path.is_dir():
        return open_folder(path)
    else:
        return open_file(path)