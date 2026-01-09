"""
Version checking utilities for Lobster AI.

Checks PyPI for the latest version and notifies users if an update is available.
Includes caching to avoid repeated checks and graceful error handling for offline usage.
"""

import json
import logging
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple
from packaging import version as pkg_version

from lobster.version import __version__

logger = logging.getLogger(__name__)

# Cache settings
CACHE_FILE_NAME = ".version_check_cache.json"
CACHE_DURATION_SECONDS = 15 * 60  # 15 minutes - balance between responsiveness and API overhead
PYPI_URL = "https://pypi.org/pypi/lobster-ai/json"
REQUEST_TIMEOUT_SECONDS = 3  # Short timeout to avoid blocking startup


def _get_cache_path() -> Path:
    """Get the path to the version check cache file."""
    # Store in user's home directory under .lobster
    cache_dir = Path.home() / ".lobster"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / CACHE_FILE_NAME


def _read_cache() -> Optional[dict]:
    """Read cached version check result."""
    cache_path = _get_cache_path()
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _write_cache(latest_version: str) -> None:
    """Write version check result to cache."""
    cache_path = _get_cache_path()
    try:
        cache_data = {
            "latest_version": latest_version,
            "checked_at": time.time(),
            "current_version": __version__,
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)
    except IOError:
        # Silently fail - cache is optional
        pass


def _is_cache_valid(cache: dict) -> bool:
    """Check if cached result is still valid (within cache duration)."""
    checked_at = cache.get("checked_at", 0)
    return (time.time() - checked_at) < CACHE_DURATION_SECONDS


def _fetch_latest_version() -> Optional[str]:
    """
    Fetch the latest version from PyPI.

    Returns:
        The latest version string, or None if the check fails.
    """
    try:
        request = urllib.request.Request(
            PYPI_URL,
            headers={"Accept": "application/json", "User-Agent": f"lobster-ai/{__version__}"}
        )
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("info", {}).get("version")
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
        logger.debug(f"Failed to fetch latest version from PyPI: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        logger.debug(f"Unexpected error checking PyPI: {e}")
        return None


def _compare_versions(current: str, latest: str) -> bool:
    """
    Compare version strings to determine if an update is available.

    Returns:
        True if latest version is newer than current version.
    """
    try:
        return pkg_version.parse(latest) > pkg_version.parse(current)
    except Exception:
        # If parsing fails, fall back to string comparison
        return latest != current and latest > current


def check_for_updates() -> Tuple[bool, Optional[str]]:
    """
    Check if a newer version of lobster-ai is available.

    Uses caching to avoid repeated network requests. Fails silently if offline.

    Returns:
        Tuple of (update_available, latest_version).
        - update_available: True if a newer version exists
        - latest_version: The latest version string, or None if check failed
    """
    current_version = __version__

    # Check cache first
    cache = _read_cache()
    if cache and _is_cache_valid(cache):
        latest = cache.get("latest_version")
        if latest:
            update_available = _compare_versions(current_version, latest)
            return update_available, latest

    # Fetch from PyPI
    latest = _fetch_latest_version()
    if latest is None:
        return False, None

    # Update cache
    _write_cache(latest)

    update_available = _compare_versions(current_version, latest)
    return update_available, latest


def get_update_message(latest_version: str) -> str:
    """
    Generate a user-friendly update notification message.

    Args:
        latest_version: The latest available version.

    Returns:
        Formatted message string for display.
    """
    return (
        f"A new version of lobster-ai is available: {latest_version} "
        f"(current: {__version__})\n"
        f"Update with: uv pip install --upgrade lobster-ai"
    )


def maybe_show_update_notification(console) -> None:
    """
    Check for updates and display a notification if one is available.

    This is the main entry point for CLI integration.

    Args:
        console: Rich console instance for output.
    """
    try:
        update_available, latest_version = check_for_updates()
        if update_available and latest_version:
            message = get_update_message(latest_version)
            console.print(f"[dim cyan]ℹ {message}[/dim cyan]")
            console.print()
    except Exception as e:
        # Never let version check crash the CLI
        logger.debug(f"Version check failed: {e}")
