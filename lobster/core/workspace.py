"""
Centralized workspace path resolution for consistent workspace location across all entry points.

Resolution Order (first match wins):
1. Explicit path parameter (passed programmatically)
2. LOBSTER_WORKSPACE environment variable
3. Current working directory + ".lobster_workspace"

Example:
    >>> from lobster.core.workspace import resolve_workspace
    >>> workspace = resolve_workspace()  # Uses env or cwd fallback
    >>> workspace = resolve_workspace("/custom/path")  # Uses explicit path
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Standard workspace folder name
WORKSPACE_FOLDER_NAME = ".lobster_workspace"

# Environment variable for workspace override
WORKSPACE_ENV_VAR = "LOBSTER_WORKSPACE"


def resolve_workspace(
    explicit_path: Optional[Union[str, Path]] = None,
    create: bool = True,
) -> Path:
    """
    Resolve the workspace path using a consistent priority order.

    Args:
        explicit_path: Explicitly provided workspace path (highest priority)
        create: Whether to create the workspace directory if it doesn't exist

    Returns:
        Path: Resolved workspace path (absolute)

    Resolution Order:
        1. explicit_path (if provided)
        2. LOBSTER_WORKSPACE environment variable
        3. Path.cwd() / ".lobster_workspace" (default fallback)

    Example:
        >>> # Use environment variable or default
        >>> workspace = resolve_workspace()

        >>> # Override with explicit path
        >>> workspace = resolve_workspace("/custom/workspace")

        >>> # Check path without creating directory
        >>> workspace = resolve_workspace(create=False)
    """
    workspace: Path
    source: str

    if explicit_path is not None:
        workspace = Path(explicit_path)
        source = "explicit parameter"
    elif env_workspace := os.environ.get(WORKSPACE_ENV_VAR):
        workspace = Path(env_workspace)
        source = f"${WORKSPACE_ENV_VAR} environment variable"
    else:
        workspace = Path.cwd() / WORKSPACE_FOLDER_NAME
        source = "current working directory"

    # Convert to absolute path without following symlinks
    # Note: Using absolute() instead of resolve() to preserve symlinks
    # This avoids issues on macOS where /var -> /private/var
    if not workspace.is_absolute():
        workspace = Path.cwd() / workspace
    workspace = Path(os.path.abspath(workspace))

    logger.debug(f"Resolved workspace to {workspace} (source: {source})")

    if create:
        workspace.mkdir(parents=True, exist_ok=True)

    return workspace


def get_workspace_env_var() -> str:
    """Return the name of the workspace environment variable.

    Returns:
        str: The environment variable name (LOBSTER_WORKSPACE)
    """
    return WORKSPACE_ENV_VAR


def get_workspace_folder_name() -> str:
    """Return the default workspace folder name.

    Returns:
        str: The folder name (.lobster_workspace)
    """
    return WORKSPACE_FOLDER_NAME
