#!/usr/bin/env python3
"""
Modern, user-friendly CLI for the Multi-Agent Bioinformatics System.
Installable via pip or curl, with rich terminal interface.
"""

# Python 3.13 compatibility: Fix multiprocessing/tqdm file descriptor issue
# Must be done FIRST before any imports that might use multiprocessing
import sys
if sys.version_info >= (3, 13):
    import multiprocessing as _mp
    try:
        _mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # Already set

import html
import os
import random
import threading
from pathlib import Path
from typing import Optional, TYPE_CHECKING

os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "900000"
import ast
import inspect
import json
import logging
import shutil
import time
from typing import Any, Dict, Iterable, List

# Heavy imports moved to TYPE_CHECKING for lazy loading (saves ~5s startup time)
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from lobster.core.client import AgentClient
    from lobster.utils import SimpleTerminalCallback, TerminalCallbackHandler, open_path

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from lobster.cli_internal.utils.path_resolution import (  # BUG FIX #6: Secure path resolution
    PathResolver,
)
from lobster.cli_internal.commands import (
    ConsoleOutputAdapter,
    show_queue_status,
    queue_load_file,
    queue_list,
    queue_clear,
    queue_export,
    QueueFileTypeNotSupported,
    metadata_list,
    metadata_clear,
    workspace_list,
    workspace_info,
    workspace_load,
    workspace_remove,
    workspace_status,
    pipeline_export,
    pipeline_list,
    pipeline_run,
    pipeline_info,
    data_summary,
    file_read,
    archive_queue,
    config_show,
    config_provider_list,
    config_provider_switch,
    config_model_list,
    config_model_switch,
    modalities_list,
    modality_describe,
    export_data,
    plots_list,
    plot_show,
)
from lobster.config import provider_setup
from lobster.config.agent_config import (
    LobsterAgentConfigurator,
    get_agent_configurator,
    initialize_configurator,
)

# Import extraction cache manager (premium feature - graceful fallback if unavailable)
from lobster.core.component_registry import component_registry

ExtractionCacheManager = component_registry.get_service('extraction_cache')
HAS_EXTRACTION_CACHE = ExtractionCacheManager is not None

from lobster.core.queue_storage import queue_file_lock
from lobster.core.workspace import resolve_workspace

# Import new UI system
from lobster.ui import LobsterTheme, setup_logging
from lobster.ui.components import (
    create_file_tree,
    create_workspace_tree,
    get_multi_progress_manager,
    get_status_display,
)
from lobster.ui.console_manager import get_console_manager

# Note: SimpleTerminalCallback, TerminalCallbackHandler, open_path moved to lazy loading
from lobster.version import __version__

# Import prompt_toolkit for autocomplete functionality (optional dependency)
try:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import (
        CompleteEvent,
        Completer,
        Completion,
        ThreadedCompleter,
    )
    from prompt_toolkit.document import Document
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

# Module logger
logger = logging.getLogger(__name__)

_COMMAND_HISTORY_LOCK = threading.Lock()


# ============================================================================
# Queue Command Exceptions
# ============================================================================
# Note: QueueFileTypeNotSupported exception now imported from shared module
# (lobster.cli_internal.commands)
# ============================================================================
# Progress Management
# ============================================================================


class NoOpProgress:
    """No-operation progress context manager for verbose/reasoning modes."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_task(self, *args, **kwargs):
        """No-op task addition."""
        return None

    def update(self, *args, **kwargs):
        """No-op update."""
        pass

    def remove_task(self, *args, **kwargs):
        """No-op task removal."""
        pass


def should_show_progress(client_arg: Optional["AgentClient"] = None) -> bool:
    """
    Determine if progress indicators should be shown based on current mode.

    Returns False (no progress) when:
    - Reasoning mode is enabled
    - Verbose mode is enabled
    - Any callback has verbose/show_tools enabled

    Returns True (show progress) otherwise.
    """
    global client

    # Use provided client or global client
    c = client_arg or client
    if not c:
        return True  # Default to showing progress if no client

    # Don't show progress if reasoning mode is enabled
    if hasattr(c, "enable_reasoning") and c.enable_reasoning:
        return False

    # Check callbacks for verbose settings
    if hasattr(c, "callbacks") and c.callbacks:
        for callback in c.callbacks:
            if hasattr(callback, "verbose") and callback.verbose:
                return False
            if hasattr(callback, "show_tools") and callback.show_tools:
                return False

    # Check custom_callbacks for verbose settings
    if hasattr(c, "custom_callbacks") and c.custom_callbacks:
        for callback in c.custom_callbacks:
            if hasattr(callback, "verbose") and callback.verbose:
                return False
            if hasattr(callback, "show_tools") and callback.show_tools:
                return False

    return True


def create_progress(description: str = "", client_arg: Optional["AgentClient"] = None):
    """
    Create a progress indicator that respects verbose/reasoning mode.

    In verbose/reasoning mode: Returns no-op progress manager
    In normal mode: Returns actual Progress spinner

    Args:
        description: Initial progress description
        client_arg: Optional client to check mode (uses global if not provided)

    Returns:
        Either a Progress object or NoOpProgress based on mode
    """
    if not should_show_progress(client_arg):
        return NoOpProgress()

    # Create actual progress spinner for normal mode
    try:
        progress_console = Console(stderr=True, force_terminal=True)
    except Exception:
        progress_console = console

    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=progress_console,
        transient=True,  # Always transient to clean up properly
    )


# ============================================================================
# Autocomplete Infrastructure
# ============================================================================


class LobsterClientAdapter:
    """Adapter to handle both local and cloud clients uniformly for autocomplete."""

    def __init__(self, client):
        self.client = client
        # Detect client type
        self.is_cloud = hasattr(client, "list_workspace_files") and hasattr(
            client, "session"
        )
        self.is_local = hasattr(client, "data_manager")

    def get_workspace_files(self) -> List[Dict[str, Any]]:
        """Get workspace files from either local or cloud client."""
        try:
            if self.is_cloud:
                # Cloud client has direct list_workspace_files method
                cloud_files = self.client.list_workspace_files()
                # Ensure consistent format
                return [self._normalize_file_info(f) for f in cloud_files]
            elif self.is_local and hasattr(self.client, "data_manager"):
                # Local client uses data_manager
                workspace_files = self.client.data_manager.list_workspace_files()
                return self._format_local_files(workspace_files)
            else:
                return []
        except Exception as e:
            # Graceful fallback for any errors
            console = console_manager.get_console()
            console.print(f"[dim red]Error getting workspace files: {e}[/dim red]")
            return []

    def _normalize_file_info(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize file info to consistent format."""
        return {
            "name": file_info.get("name", ""),
            "path": file_info.get("path", ""),
            "size": file_info.get("size", 0),
            "type": file_info.get("type", "unknown"),
            "modified": file_info.get("modified", 0),
        }

    def _format_local_files(
        self, workspace_files: Dict[str, List]
    ) -> List[Dict[str, Any]]:
        """Format local workspace files to consistent format."""
        files = []
        for category, file_list in workspace_files.items():
            for file_info in file_list:
                files.append(
                    {
                        "name": file_info.get("name", ""),
                        "path": file_info.get("path", ""),
                        "size": file_info.get("size", 0),
                        "type": category,
                        "modified": file_info.get("modified", 0),
                    }
                )
        return files

    def can_read_files(self) -> bool:
        """Check if client supports file reading."""
        return self.is_cloud or (self.is_local and hasattr(self.client, "read_file"))


class CloudAwareCache:
    """Smart caching that adapts to client type."""

    def __init__(self, client):
        self.is_cloud = hasattr(client, "list_workspace_files") and hasattr(
            client, "session"
        )
        self.cache = {}
        self.timeouts = {
            "commands": float("inf"),  # Commands never change
            "files": 60 if self.is_cloud else 10,  # Longer cache for cloud
            "workspace": 30 if self.is_cloud else 5,
        }

    def get_or_fetch(self, key: str, fetch_func, category: str = "default"):
        """Get cached value or fetch if expired."""
        current_time = time.time()
        timeout = self.timeouts.get(category, 10)

        if (
            key not in self.cache
            or current_time - self.cache[key]["timestamp"] > timeout
        ):
            try:
                self.cache[key] = {"data": fetch_func(), "timestamp": current_time}
            except Exception as e:
                if self.is_cloud and (
                    "connection" in str(e).lower() or "timeout" in str(e).lower()
                ):
                    # For cloud connection errors, return stale cache if available
                    if key in self.cache:
                        console = console_manager.get_console()
                        console.print(
                            "[dim yellow]Using cached data due to connection issue[/dim yellow]"
                        )
                        return self.cache[key]["data"]
                raise e

        return self.cache[key]["data"]


# FIXME currenlty langraph implementation
def _add_command_to_history(
    client: "AgentClient", command: str, summary: str, is_error: bool = False
) -> bool:
    """
    Add command execution to conversation history for AI context.

    BUG FIX #4: Enhanced error handling with full logging and file backup.
    - Returns bool to indicate success/failure
    - Logs full error messages with stack traces (not truncated to 50 chars)
    - Implements file backup for audit trail and recovery
    - Provides detailed diagnostics for debugging

    Args:
        client: "AgentClient" instance
        command: Command that was executed
        summary: Summary of command result
        is_error: Whether this was an error result

    Returns:
        True if successfully logged, False otherwise
    """
    # 1. Validate inputs
    if not command or not summary:
        logger.warning("Empty command or summary provided to history logger")
        return False

    # 2. Check client compatibility
    if not hasattr(client, "messages") or not isinstance(client.messages, list):
        logger.info(
            f"Client type {type(client).__name__} doesn't support message history. "
            f"Commands will not be available in AI context."
        )
        return False

    # 3. Attempt primary logging (graph state)
    primary_logged = False
    try:
        # Import required message types
        from langchain_core.messages import AIMessage, HumanMessage

        # Format messages for conversation history
        human_message_command_usage = f"Command: {command}"
        status_prefix = "Error" if is_error else "Result"
        ai_message_command_response = f"Command {status_prefix}: {summary}"

        # Add messages directly to client.messages (the correct API)
        config = dict(configurable=dict(thread_id=client.session_id))
        human_msg = HumanMessage(content=human_message_command_usage)
        ai_msg = AIMessage(content=ai_message_command_response)

        # Add to client message history
        client.messages.append(human_msg)
        client.messages.append(ai_msg)

        # Update graph state
        client.graph.update_state(
            config,
            dict(messages=[human_msg, ai_msg]),
        )

        logger.debug(f"✓ Logged command to graph state: {command[:50]}")
        primary_logged = True

    except ImportError as e:
        logger.error(f"Cannot import langchain message types: {e}")
        return False

    except AttributeError as e:
        # BUG FIX #4: Full error logging with diagnostic info
        logger.error(
            f"Client missing required attributes for history logging: {e}. "
            f"Client type: {type(client).__name__}, "
            f"Has messages: {hasattr(client, 'messages')}, "
            f"Has graph: {hasattr(client, 'graph')}"
        )

    except Exception as e:
        # BUG FIX #4: Log FULL exception with stack trace (not truncated)
        logger.error(
            f"Failed to log command '{command}' to graph state: {e}",
            exc_info=True,  # Include full traceback for debugging
        )

    # 4. Backup to file (always, for audit trail and recovery)
    backup_logged = _backup_command_to_file(
        client, command, summary, is_error, primary_logged
    )

    return primary_logged or backup_logged


def _backup_command_to_file(
    client: "AgentClient",
    command: str,
    summary: str,
    is_error: bool,
    primary_logged: bool,
) -> bool:
    """
    Write command to backup file for audit trail and recovery.

    BUG FIX #4: Dual-channel logging - backup commands to file even if graph state succeeds.
    Provides audit trail, enables session reconstruction, supports compliance requirements.

    Args:
        client: "AgentClient" instance
        command: Command that was executed
        summary: Summary of command result
        is_error: Whether this was an error result
        primary_logged: Whether primary (graph state) logging succeeded

    Returns:
        True if backup successful, False otherwise
    """
    try:
        history_dir = client.data_manager.workspace_path / ".lobster"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_file = history_dir / "command_history.jsonl"
        lock_path = history_file.with_suffix(".lock")

        from datetime import datetime

        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": client.session_id,
            "command": command,
            "summary": summary,
            "is_error": is_error,
            "logged_to_graph": primary_logged,
        }

        with queue_file_lock(_COMMAND_HISTORY_LOCK, lock_path):
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        logger.debug(f"✓ Backed up command to file: {command[:50]}")
        return True

    except Exception as e:
        # BUG FIX #4: Log backup failures with full stack trace
        logger.error(f"Failed to write command backup: {e}", exc_info=True)
        return False


def check_for_missing_slash_command(user_input: str) -> Optional[str]:
    """Check if user input matches a command without the leading slash."""
    if not user_input or user_input.startswith("/"):
        return None

    # Get the first word (potential command)
    first_word = user_input.split()[0].lower()

    # Check against known commands (without the slash)
    available_commands = extract_available_commands()
    for cmd in available_commands.keys():
        cmd_without_slash = cmd[1:]  # Remove the leading slash
        if first_word == cmd_without_slash:
            return cmd

    return None


def extract_available_commands() -> Dict[str, str]:
    """Extract commands dynamically from _execute_command implementation."""

    # Static command definitions with descriptions (extracted from help text)
    command_descriptions = {
        "/help": "Show this help message",
        "/session": "Show current session status",
        "/status": "Show subscription tier, packages, and agents",
        "/input-features": "Show input capabilities and navigation features",
        "/dashboard": "Switch to interactive dashboard (Textual UI)",
        "/status-panel": "Show comprehensive system status panel",
        "/workspace-info": "Show detailed workspace overview",
        "/analysis-dash": "Show analysis monitoring dashboard",
        "/progress": "Show multi-task progress monitor",
        "/files": "List workspace files",
        "/tree": "Show directory tree view",
        "/data": "Show current data summary",
        "/metadata": "Show detailed metadata information",
        "/metadata list": "List metadata store entries",
        "/metadata clear": "Clear metadata store",
        # Queue commands (temporary, intent-driven)
        "/queue": "Show queue status",
        "/queue load": "Load file into queue (supports .ris, more coming)",
        "/queue list": "List queued items",
        "/queue clear": "Clear publication queue",
        "/queue clear download": "Clear download queue",
        "/queue clear all": "Clear all queues (publication + download)",
        "/queue export": "Export queue to workspace for persistence",
        # Workspace commands (persistent)
        "/workspace": "Show workspace status and information",
        "/workspace list": "List available datasets in workspace",
        "/workspace load": "Load dataset or file into workspace",
        "/workspace remove": "Remove modality from memory",
        "/workspace save": "Save modality to workspace",
        "/workspace info": "Show dataset information",
        "/restore": "Restore previous session datasets",
        "/modalities": "Show detailed modality information",
        "/describe": "Show detailed information about a specific modality",
        "/plots": "List all generated plots",
        "/plot": "Open plots directory or specific plot",
        "/open": "Open file or folder in system default application",
        "/save": "Save current state to workspace",
        "/read": "View file contents (inspection only)",
        "/export": "Export session data",
        "/reset": "Reset conversation",
        "/mode": "Change operation mode",
        "/modes": "List available modes",
        "/config": "Show current configuration (provider, profile, config files)",
        "/config provider": "List available LLM providers",
        "/config provider <name>": "Switch to specified provider (runtime only)",
        "/config provider <name> --save": "Switch provider and persist to workspace",
        "/config model": "List available Ollama models",
        "/config model <name>": "Switch to specified Ollama model (runtime only)",
        "/config model <name> --save": "Switch Ollama model and persist to workspace",
        "/clear": "Clear screen",
        "/exit": "Exit the chat",
    }

    # Try to extract dynamically as fallback, but use static definitions as primary
    try:
        # Get the source code of _execute_command
        source = inspect.getsource(_execute_command)
        tree = ast.parse(source)

        # Walk through AST to find command comparisons
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Look for patterns like: cmd == "/something"
                if (
                    isinstance(node.left, ast.Name)
                    and node.left.id == "cmd"
                    and len(node.ops) == 1
                    and isinstance(node.ops[0], ast.Eq)
                    and len(node.comparators) == 1
                    and isinstance(node.comparators[0], ast.Constant)
                ):

                    cmd = node.comparators[0].value
                    if isinstance(cmd, str) and cmd.startswith("/"):
                        # If not in our static definitions, add with generic description
                        if cmd not in command_descriptions:
                            command_descriptions[cmd] = f"Execute {cmd} command"

    except Exception:
        # Fallback to static definitions if AST parsing fails (silent)
        pass

    return command_descriptions


if PROMPT_TOOLKIT_AVAILABLE:

    class LobsterCommandCompleter(Completer):
        """Completer for Lobster slash commands with rich metadata."""

        def __init__(self):
            self.commands_cache = None
            self.cache_time = 0

        def get_completions(
            self, document: Document, complete_event: CompleteEvent
        ) -> Iterable[Completion]:
            """Generate hierarchical command completions.

            Implements depth-aware filtering to show only relevant commands:
            - Typing "/" → shows top-level commands (/help, /config, /queue)
            - Typing "/config " → shows subcommands (provider, model, show)
            - Typing "/config model " → shows model names

            This prevents overwhelming users with nested subcommands at the top level.
            """
            text_before_cursor = document.text_before_cursor.lstrip()

            # Only complete if we're typing a command (starts with /)
            if not text_before_cursor.startswith("/"):
                return

            # Get available commands (cached)
            commands = self._get_cached_commands()

            # Analyze input structure for hierarchical filtering
            has_trailing_space = text_before_cursor.endswith(" ")
            typed_text = text_before_cursor.rstrip()  # Remove trailing space for analysis
            typed_parts = typed_text.split() if typed_text else []
            current_depth = len(typed_parts)

            # Determine target depth for completions
            if has_trailing_space:
                # User typed "/config " (with space) → show next level subcommands
                # Target depth: current_depth + 1 (e.g., depth 1 → show depth 2)
                target_depth = current_depth + 1
                prefix_match = typed_text + " "  # Must start with "/config "
            else:
                # User typed "/conf" (no space) → complete current level
                # Target depth: current_depth (e.g., depth 1 → complete depth 1)
                target_depth = current_depth
                prefix_match = typed_text  # Must start with "/conf"

            # Generate completions with depth filtering
            for cmd, description in commands.items():
                # Calculate command depth (number of space-separated parts)
                cmd_depth = len(cmd.split())

                # Apply hierarchical filter: only show commands at target depth
                if cmd_depth != target_depth:
                    continue

                # Apply prefix filter: command must start with what user typed
                if not cmd.lower().startswith(prefix_match.lower()):
                    continue

                # Calculate completion text and cursor position
                if has_trailing_space:
                    # Show only the next part (e.g., "provider" from "/config provider")
                    completion_text = cmd[len(prefix_match):]  # Extract remaining part
                    start_position = 0
                else:
                    # Show full command and replace from beginning of current word
                    completion_text = cmd
                    current_word = typed_parts[-1] if typed_parts else ""
                    start_position = -len(current_word)

                # Escape HTML special characters to prevent XML parsing errors
                escaped_cmd = html.escape(completion_text)
                escaped_desc = html.escape(description)

                yield Completion(
                    text=completion_text,
                    start_position=start_position,
                    display=HTML(f"<ansired>{escaped_cmd}</ansired>"),
                    display_meta=HTML(f"<dim>{escaped_desc}</dim>"),
                    style="class:completion.command",
                )

        def _get_cached_commands(self) -> Dict[str, str]:
            """Get commands with caching."""
            current_time = time.time()
            # Cache commands for 5 minutes
            if self.commands_cache is None or current_time - self.cache_time > 300:
                self.commands_cache = extract_available_commands()
                self.cache_time = current_time
            return self.commands_cache

    class LobsterFileCompleter(Completer):
        """Completer for workspace files with cloud-aware caching."""

        def __init__(self, client):
            self.adapter = LobsterClientAdapter(client)
            self.cache = CloudAwareCache(client)

        def get_completions(
            self, document: Document, complete_event: CompleteEvent
        ) -> Iterable[Completion]:
            """Generate file completions."""
            word = document.get_word_before_cursor()

            # Get files with caching
            try:
                files = self.cache.get_or_fetch(
                    "workspace_files",
                    lambda: self.adapter.get_workspace_files(),
                    "files",
                )
            except Exception as e:
                # Graceful fallback
                console = console_manager.get_console()
                console.print(f"[dim red]File completion error: {e}[/dim red]")
                files = []

            # Generate completions
            for file_info in files:
                file_name = file_info.get("name", "")
                if file_name.lower().startswith(word.lower()):
                    # Format file metadata
                    file_size = file_info.get("size", 0)
                    file_type = file_info.get("type", "unknown")

                    # Format size
                    if file_size < 1024:
                        size_str = f"{file_size}B"
                    elif file_size < 1024**2:
                        size_str = f"{file_size / 1024:.1f}KB"
                    elif file_size < 1024**3:
                        size_str = f"{file_size / 1024 ** 2:.1f}MB"
                    else:
                        size_str = f"{file_size / 1024 ** 3:.1f}GB"

                    meta = f"{file_type} • {size_str}"

                    yield Completion(
                        text=file_name,
                        start_position=-len(word),
                        display=HTML(f"<ansicyan>{file_name}</ansicyan>"),
                        display_meta=HTML(f"<dim>{meta}</dim>"),
                        style="class:completion.file",
                    )

    class LobsterContextualCompleter(Completer):
        """Smart contextual completer that switches between commands and files."""

        def __init__(self, client):
            self.client = client
            self.adapter = LobsterClientAdapter(client)
            self.command_completer = LobsterCommandCompleter()
            self.file_completer = LobsterFileCompleter(client)

            # Commands that expect file arguments
            self.file_commands = {"/read", "/plot", "/open"}

        def get_completions(
            self, document: Document, complete_event: CompleteEvent
        ) -> Iterable[Completion]:
            """Generate context-aware completions."""
            text = document.text_before_cursor.strip()

            if not text:
                # Empty input - show all commands
                yield from self.command_completer.get_completions(
                    document, complete_event
                )

            elif text.startswith("/") and " " not in text:
                # Command completion (typing a command)
                yield from self.command_completer.get_completions(
                    document, complete_event
                )

            elif text.startswith("/workspace load "):
                # Suggest available dataset names
                prefix = text.replace("/workspace load ", "")
                try:
                    if hasattr(self.client.data_manager, "available_datasets"):
                        for (
                            name,
                            info,
                        ) in self.client.data_manager.available_datasets.items():
                            if name.lower().startswith(prefix.lower()):
                                size_mb = info.get("size_mb", 0)
                                shape = info.get("shape", (0, 0))
                                meta = f"{size_mb:.1f}MB • {shape[0]}×{shape[1]}"
                                yield Completion(
                                    text=name,
                                    start_position=-len(prefix),
                                    display=HTML(f"<ansicyan>{name}</ansicyan>"),
                                    display_meta=HTML(f"<dim>{meta}</dim>"),
                                    style="class:completion.dataset",
                                )
                except Exception:
                    pass

            elif text.startswith("/workspace remove "):
                # Suggest currently loaded modality names for removal
                prefix = text.replace("/workspace remove ", "")
                try:
                    if hasattr(self.client.data_manager, "list_modalities"):
                        modalities = self.client.data_manager.list_modalities()
                        for name in modalities:
                            if name.lower().startswith(prefix.lower()):
                                # Get modality info for metadata
                                try:
                                    adata = self.client.data_manager.get_modality(name)
                                    meta = f"{adata.n_obs}×{adata.n_vars}"
                                    yield Completion(
                                        text=name,
                                        start_position=-len(prefix),
                                        display=HTML(f"<ansired>{name}</ansired>"),
                                        display_meta=HTML(f"<dim>{meta}</dim>"),
                                        style="class:completion.modality",
                                    )
                                except Exception:
                                    # Fallback without metadata if getting modality fails
                                    yield Completion(
                                        text=name,
                                        start_position=-len(prefix),
                                        display=HTML(f"<ansired>{name}</ansired>"),
                                        style="class:completion.modality",
                                    )
                except Exception:
                    pass

            elif text.startswith("/restore "):
                # Suggest restore patterns
                patterns = ["recent", "all", "*"]
                prefix = text.replace("/restore ", "")
                for pattern in patterns:
                    if pattern.startswith(prefix):
                        yield Completion(
                            text=pattern,
                            start_position=-len(prefix),
                            display=HTML(f"<ansiyellow>{pattern}</ansiyellow>"),
                            display_meta=HTML("<dim>restore pattern</dim>"),
                            style="class:completion.pattern",
                        )

            elif text.startswith("/describe "):
                # Suggest modality names for describe command
                prefix = text.replace("/describe ", "")
                try:
                    if hasattr(self.client.data_manager, "list_modalities"):
                        modalities = self.client.data_manager.list_modalities()
                        for modality_name in modalities:
                            if modality_name.lower().startswith(prefix.lower()):
                                # Get basic info about the modality if possible
                                try:
                                    adata = self.client.data_manager.get_modality(
                                        modality_name
                                    )
                                    meta = (
                                        f"{adata.n_obs:,} obs × {adata.n_vars:,} vars"
                                    )
                                except Exception:
                                    meta = "modality"

                                yield Completion(
                                    text=modality_name,
                                    start_position=-len(prefix),
                                    display=HTML(
                                        f"<ansicyan>{modality_name}</ansicyan>"
                                    ),
                                    display_meta=HTML(f"<dim>{meta}</dim>"),
                                    style="class:completion.modality",
                                )
                except Exception:
                    pass

            elif text == "/config" or text == "/config ":
                # Show /config subcommands
                subcommands = [
                    ("provider", "Set/show LLM provider"),
                    ("model", "Set/show model for current provider"),
                    ("show", "Show current configuration"),
                    ("profile", "Set/show agent profile"),
                ]
                prefix = "" if text == "/config" else ""
                for subcmd, desc in subcommands:
                    yield Completion(
                        text=" " + subcmd if text == "/config" else subcmd,
                        start_position=0,
                        display=HTML(f"<ansicyan>{subcmd}</ansicyan>"),
                        display_meta=HTML(f"<dim>{desc}</dim>"),
                        style="class:completion.subcommand",
                    )

            elif text == "/config model" or text == "/config provider":
                # User typed /config model or /config provider without trailing space
                # Show available options
                if text == "/config model":
                    try:
                        from lobster.config.model_service import ModelServiceFactory
                        from lobster.config.llm_factory import LLMFactory

                        # Get current provider (from client, not adapter)
                        current_provider = getattr(self.client, 'provider_override', None) or LLMFactory.get_current_provider()

                        # Get model service for current provider
                        service = ModelServiceFactory.get_service(current_provider)
                        models = service.list_models()

                        for model in models:
                            meta = f"{model.display_name}"
                            if model.is_default:
                                meta += " (default)"

                            yield Completion(
                                text=" " + model.name,  # Add space before model name
                                start_position=0,
                                display=HTML(f"<ansiyellow>{model.name}</ansiyellow>"),
                                display_meta=HTML(f"<dim>{meta}</dim>"),
                                style="class:completion.model",
                            )
                    except Exception:
                        pass

                elif text == "/config provider":
                    # Dynamically fetch providers from ProviderRegistry
                    from lobster.config.providers import ProviderRegistry
                    try:
                        providers = ProviderRegistry.get_provider_names()
                    except Exception:
                        providers = ["anthropic", "bedrock", "ollama", "gemini"]  # Fallback

                    for provider in providers:
                        meta = f"Switch to {provider}"
                        yield Completion(
                            text=" " + provider,  # Add space before provider
                            start_position=0,
                            display=HTML(f"<ansiyellow>{provider}</ansiyellow>"),
                            display_meta=HTML(f"<dim>{meta}</dim>"),
                            style="class:completion.provider",
                        )

            elif text.startswith("/config model ") or text.startswith("/config provider "):
                # Tab completion for model names and provider names
                prefix = text.split(" ", 2)[-1]

                if text.startswith("/config model "):
                    # Provider-aware model completion
                    try:
                        from lobster.config.model_service import ModelServiceFactory
                        from lobster.config.llm_factory import LLMFactory

                        # Get current provider (from client, not adapter)
                        current_provider = getattr(self.client, 'provider_override', None) or LLMFactory.get_current_provider()

                        # Get model service for current provider
                        service = ModelServiceFactory.get_service(current_provider)
                        models = service.list_models()

                        for model in models:
                            if model.name.lower().startswith(prefix.lower()):
                                meta = f"{model.display_name}"
                                if model.is_default:
                                    meta += " (default)"

                                yield Completion(
                                    text=model.name,
                                    start_position=-len(prefix),
                                    display=HTML(f"<ansiyellow>{model.name}</ansiyellow>"),
                                    display_meta=HTML(f"<dim>{meta}</dim>"),
                                    style="class:completion.model",
                                )
                    except Exception:
                        # Silent fail for tab completion
                        pass

                elif text.startswith("/config provider "):
                    # Provider completion - dynamically fetch from ProviderRegistry
                    from lobster.config.providers import ProviderRegistry
                    try:
                        providers = ProviderRegistry.get_provider_names()
                    except Exception:
                        providers = ["anthropic", "bedrock", "ollama", "gemini"]  # Fallback

                    for provider in providers:
                        if provider.lower().startswith(prefix.lower()):
                            meta = f"Switch to {provider}"
                            yield Completion(
                                text=provider,
                                start_position=-len(prefix),
                                display=HTML(f"<ansiyellow>{provider}</ansiyellow>"),
                                display_meta=HTML(f"<dim>{meta}</dim>"),
                                style="class:completion.provider",
                            )

            elif any(text.startswith(cmd + " ") for cmd in self.file_commands):
                # File completion for file-accepting commands
                if self.adapter.can_read_files():
                    # Create a modified document that only includes the file part
                    # Find where the file argument starts
                    parts = text.split(" ", 1)
                    if len(parts) > 1:
                        file_part = parts[1]
                        # Create new document for file completion
                        from prompt_toolkit.document import Document

                        file_document = Document(
                            text=file_part, cursor_position=len(file_part)
                        )
                        yield from self.file_completer.get_completions(
                            file_document, complete_event
                        )

            elif text.startswith("/") and " " in text:
                # Other commands with arguments - could be extended for more specific completions
                pass


def change_mode(new_mode: str, current_client: "AgentClient") -> "AgentClient":
    """
    Change the operation mode and reinitialize client with the new configuration.

    Args:
        new_mode: The new mode/profile to switch to
        current_client: The current AgentClient instance

    Returns:
        Updated AgentClient instance
    """
    global client

    # Store current settings before reinitializing
    current_workspace = Path(current_client.workspace_path)
    current_reasoning = current_client.enable_reasoning

    # Initialize a new configurator with the specified profile
    initialize_configurator(profile=new_mode)

    # Reinitialize the client with the new profile settings
    client = init_client(workspace=current_workspace, reasoning=current_reasoning)

    return client


# Initialize Rich console with orange theming and Typer app
console_manager = get_console_manager()
console = console_manager.console

app = typer.Typer(
    name="lobster",
    help="🦞 Lobster by Omics-OS - Multi-Agent Bioinformatics Analysis System",
    add_completion=True,
    rich_markup_mode="rich",
)

# Create a subcommand for configuration management
config_app = typer.Typer(
    name="config",
    help="Configuration management for Lobster agents",
)
app.add_typer(config_app, name="config")


# App callback to show help when no subcommand is provided
@app.callback(invoke_without_command=True)
def default_callback(ctx: typer.Context):
    """
    Show friendly help guide when lobster is invoked without subcommands.
    """
    # If no subcommand was invoked, show the default help
    if ctx.invoked_subcommand is None:
        show_default_help()
        raise typer.Exit()


# Global client instance
client: Optional["AgentClient"] = None

# Global current directory tracking
current_directory = Path.cwd()

PROFILE_TIMINGS_ENV = "LOBSTER_PROFILE_TIMINGS"


def _str_to_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _resolve_profile_timings_flag(cli_flag: Optional[bool]) -> bool:
    if cli_flag is not None:
        return cli_flag
    env_value = _str_to_bool(os.environ.get(PROFILE_TIMINGS_ENV))
    return bool(env_value)


def _collect_profile_timings(
    client: "AgentClient", clear: bool = True
) -> Dict[str, Dict[str, float]]:
    timings: Dict[str, Dict[str, float]] = {}
    data_manager = getattr(client, "data_manager", None)
    if data_manager and hasattr(data_manager, "get_latest_timings"):
        dm_timings = data_manager.get_latest_timings(clear=clear)
        if dm_timings:
            timings["DataManager"] = dm_timings
    return timings


def _maybe_print_timings(client: "AgentClient", context: str) -> None:
    if not getattr(client, "profile_timings_enabled", False):
        return

    timing_sources = _collect_profile_timings(client, clear=True)
    if not timing_sources:
        return

    table = Table(title=f"{context} Timings", box=box.ROUNDED)
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Step", style="white")
    table.add_column("Seconds", justify="right")

    for component, entries in timing_sources.items():
        for step, value in sorted(
            entries.items(), key=lambda item: item[1], reverse=True
        ):
            table.add_row(component, step, f"{value:.2f}")

    console.print(table)


def init_client(
    workspace: Optional[Path] = None,
    reasoning: bool = False,
    verbose: bool = False,
    debug: bool = False,
    profile_timings: Optional[bool] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    session_id: Optional[str] = None,
) -> "AgentClient":
    """Initialize either local or cloud client based on environment."""
    # Lazy imports for performance (saves ~5s startup time)
    from lobster.core.client import AgentClient
    from lobster.config.settings import settings

    global client

    # Check for provider configuration using ConfigResolver
    from lobster.core.config_resolver import ConfigResolver, ConfigurationError
    from lobster.core.workspace import resolve_workspace

    # Resolve workspace (create if needed for proper config loading)
    workspace_path = resolve_workspace(explicit_path=workspace, create=True)
    resolver = ConfigResolver.get_instance(workspace_path)

    try:
        # This will raise ConfigurationError if provider not configured
        resolver.resolve_provider()
    except ConfigurationError as e:
        console.print(f"[red]❌ {str(e)}[/red]")
        console.print(f"\n[yellow]{e.help_text}[/yellow]")
        raise typer.Exit(code=1)

    # Check for cloud API key
    cloud_key = os.environ.get("LOBSTER_CLOUD_KEY")
    cloud_endpoint = os.environ.get("LOBSTER_ENDPOINT")

    if cloud_key:
        # Detect cloud key but provide better user experience
        console.print("[bold blue]🌩️  Cloud API key detected...[/bold blue]")

        try:
            from lobster.lobster_cloud.client import CloudLobsterClient

            console.print("[bold blue]   Initializing Lobster Cloud...[/bold blue]")
            if cloud_endpoint:
                console.print(f"[dim blue]   Endpoint: {cloud_endpoint}[/dim blue]")

            # Initialize cloud client with endpoint support
            client_kwargs = {"api_key": cloud_key}
            if cloud_endpoint:
                client_kwargs["endpoint"] = cloud_endpoint

            client = CloudLobsterClient(**client_kwargs)

            # Test connection with retry logic
            max_retries = 3
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    status_result = client.get_status()

                    if status_result.get("success", False):
                        console.print(
                            "[bold green]✅ Cloud connection established[/bold green]"
                        )
                        console.print(
                            f"[dim blue]   Status: {status_result.get('status', 'unknown')}[/dim blue]"
                        )
                        if status_result.get("version"):
                            console.print(
                                f"[dim blue]   Version: {status_result.get('version')}[/dim blue]"
                            )
                        return client
                    else:
                        error_msg = status_result.get("error", "Unknown error")
                        if attempt < max_retries - 1:
                            console.print(
                                f"[yellow]⚠️  Connection test failed (attempt {attempt + 1}): {error_msg}[/yellow]"
                            )
                            console.print(
                                f"[yellow]   Retrying in {retry_delay} seconds...[/yellow]"
                            )
                            import time

                            time.sleep(retry_delay)
                        else:
                            console.print(
                                f"[red]❌ Cloud connection failed after {max_retries} attempts: {error_msg}[/red]"
                            )
                            raise Exception(f"Connection test failed: {error_msg}")

                except Exception as e:
                    if "timeout" in str(e).lower():
                        error_type = "Connection timeout"
                        suggestion = "Check your internet connection and endpoint URL"
                    elif "401" in str(e) or "unauthorized" in str(e).lower():
                        error_type = "Authentication failed"
                        suggestion = "Verify your LOBSTER_CLOUD_KEY is correct"
                    elif "404" in str(e) or "not found" in str(e).lower():
                        error_type = "Endpoint not found"
                        suggestion = "Check your LOBSTER_ENDPOINT URL"
                    else:
                        error_type = "Connection error"
                        suggestion = "Check network connectivity and service status"

                    if attempt < max_retries - 1:
                        console.print(
                            f"[yellow]⚠️  {error_type} (attempt {attempt + 1}): {e}[/yellow]"
                        )
                        console.print(
                            f"[yellow]   Retrying in {retry_delay} seconds...[/yellow]"
                        )
                        import time

                        time.sleep(retry_delay)
                    else:
                        console.print(
                            f"[red]❌ {error_type} after {max_retries} attempts[/red]"
                        )
                        console.print(f"[red]   Error: {e}[/red]")
                        console.print(f"[yellow]   Suggestion: {suggestion}[/yellow]")
                        raise Exception(f"{error_type}: {e}")

        except ImportError:
            # Provide better guidance for cloud users
            console.print(
                "[bold yellow]☁️  Lobster Cloud Not Available Locally[/bold yellow]"
            )
            console.print(
                "[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]"
            )
            console.print(
                "[white]You have a [bold blue]LOBSTER_CLOUD_KEY[/bold blue] set, but this is the open-source version.[/white]"
            )
            console.print("")
            console.print("[bold white]🌟 Get Lobster Cloud Access:[/bold white]")
            console.print("   • Visit: [bold blue]https://cloud.lobster.ai[/bold blue]")
            console.print("   • Email: [bold blue]cloud@omics-os.com[/bold blue]")
            console.print("")
            console.print(
                "[bold white]💻 For now, using local mode with full functionality:[/bold white]"
            )
            console.print(
                "[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/cyan]"
            )

        except Exception as e:
            console.print(f"[red]❌ Cloud connection error: {e}[/red]")
            console.print("[yellow]   Falling back to local mode...[/yellow]")

    # Use local client (existing code)
    # Configure logging level based on debug flag
    import logging

    if debug:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.WARNING)  # Suppress INFO logs

    # Set workspace using centralized resolver
    # Resolution order: explicit --workspace > LOBSTER_WORKSPACE env var > cwd/.lobster_workspace
    workspace = resolve_workspace(explicit_path=workspace, create=True)

    # Initialize DataManagerV2 with workspace support and console for progress tracking
    from lobster.core.data_manager_v2 import DataManagerV2

    data_manager = DataManagerV2(workspace_path=workspace, console=console)

    profile_timings_enabled = _resolve_profile_timings_flag(profile_timings)
    if profile_timings_enabled and hasattr(data_manager, "enable_timing"):
        data_manager.enable_timing(True)

    # Create callback using the appropriate terminal_callback_handler
    # Configure callbacks based on reasoning and verbose flags independently
    from lobster.utils import SimpleTerminalCallback, TerminalCallbackHandler

    callbacks = []

    if reasoning or verbose:
        # Use full TerminalCallbackHandler when either reasoning or verbose is enabled
        callback = TerminalCallbackHandler(
            console=console,
            show_reasoning=reasoning,  # Only show reasoning if reasoning flag is True
            verbose=verbose,  # Control tool/agent verbosity independently
            show_tools=verbose,  # Only show detailed tool output if verbose is True
        )
        callbacks.append(callback)
    else:
        # Use minimal oh-my-zsh style output (◀ Agent Name) for clean default
        simple_callback = SimpleTerminalCallback(
            console=console,
            show_reasoning=False,
            minimal=True,  # Enable minimal mode for clean agent indicators
        )
        callbacks.append(simple_callback)

    # Initialize client with proper data_manager connection
    client = AgentClient(
        data_manager=data_manager,  # Pass the configured data_manager
        workspace_path=workspace,
        session_id=session_id,  # Pass custom session_id if provided
        enable_reasoning=reasoning,
        # enable_langfuse=debug,
        custom_callbacks=callbacks,  # Pass the proper callback
        provider_override=provider_override,  # Pass provider override from CLI flag
        model_override=model_override,  # Pass model override from CLI flag
    )

    client.profile_timings_enabled = profile_timings_enabled

    # Show graph visualization in debug mode
    if debug:
        try:
            # Get the graph from the client
            if hasattr(client, "graph") and client.graph:
                # Generate and save mermaid PNG to workspace
                mermaid_png = client.graph.get_graph().draw_mermaid_png()
                graph_file = workspace / "agent_graph.png"

                with open(graph_file, "wb") as f:
                    f.write(mermaid_png)

                console.print(
                    f"[green]📊 Graph visualization saved to: {graph_file}[/green]"
                )
        except Exception as e:
            console.print(
                f"[yellow]⚠️  Could not generate graph visualization: {e}[/yellow]"
            )

    return client


def get_user_input_with_editing(prompt_text: str, client=None) -> str:
    """
    Get user input with advanced arrow key navigation, command history, and autocomplete.

    Features:
    - Left/Right arrows for cursor movement
    - Up/Down arrows for command history navigation
    - Ctrl+R for reverse search through history
    - Home/End for line navigation
    - Backspace/Delete for editing
    - Tab completion for commands and files
    - Cloud-aware file completion
    - Full command history persistence
    """
    try:
        # Create history file path for persistent command history
        history_file = None
        if PROMPT_TOOLKIT_AVAILABLE:
            try:
                history_dir = Path.home() / ".lobster"
                history_dir.mkdir(exist_ok=True)
                history_file = FileHistory(str(history_dir / "lobster_history"))
            except Exception:
                # If history file creation fails, continue without it
                history_file = None

        # Try to use prompt_toolkit with autocomplete if available
        if PROMPT_TOOLKIT_AVAILABLE and client:
            # Clean prompt text - remove Rich markup, keep just the symbol
            import re
            clean_prompt = re.sub(r'\[.*?\]', '', prompt_text).strip()
            if not clean_prompt:
                clean_prompt = "❯ "

            # Create client-aware completer
            main_completer = ThreadedCompleter(LobsterContextualCompleter(client))

            # Custom style to match Rich orange theme
            style = Style.from_dict(
                {
                    "prompt": "#e45c47 bold",
                    "completion-menu.completion": "bg:#2d2d2d #00d7ff",  # Cyan text (colorblind safe, high contrast)
                    "completion-menu.completion.current": "bg:#0087ff #ffffff bold",  # Blue bg (colorblind safe)
                    "completion-menu.meta": "bg:#2d2d2d #ffd700",  # Gold text (colorblind safe, high contrast)
                    "completion-menu.meta.current": "bg:#0087ff #ffffff",  # Blue bg (colorblind safe)
                    "completion.command": "#00d7ff",  # Cyan (high contrast, colorblind safe)
                    "completion.file": "#ffd700",  # Gold/Yellow (high contrast, colorblind safe)
                }
            )

            # Use prompt_toolkit with autocomplete - orange prompt
            user_input = prompt(
                HTML(f"<style fg='#e45c47' bold='true'>{clean_prompt}</style>"),
                completer=main_completer,
                complete_while_typing=True,
                # Disable mouse support so terminal scroll remains usable
                mouse_support=False,  # FIXME change this back to True if needed. I deactivated to allow scrolling
                style=style,
                complete_style="multi-column",
                history=history_file,
            )
            return user_input.strip()

        elif PROMPT_TOOLKIT_AVAILABLE:
            # Clean prompt text for non-autocomplete mode too
            import re
            clean_prompt = re.sub(r'\[.*?\]', '', prompt_text).strip()
            if not clean_prompt:
                clean_prompt = "❯ "

            # Use prompt_toolkit without autocomplete (no client provided)
            user_input = prompt(
                HTML(f"<style fg='#e45c47' bold='true'>{clean_prompt}</style>"),
                # Disable mouse support so terminal scroll remains usable
                mouse_support=False,  # FIXME change this back to True if needed. I deactivated to allow scrolling
                history=history_file,
            )
            return user_input.strip()

        else:
            # Graceful fallback to current Rich input
            user_input = console_manager.console.input(
                prompt=prompt_text, markup=True, emoji=True
            )
            return user_input.strip()

    except (KeyboardInterrupt, EOFError):
        # Handle Ctrl+C or Ctrl+D gracefully
        raise KeyboardInterrupt
    except Exception as e:
        # Fallback on any other error (e.g., prompt_toolkit issues)
        console.print(f"[dim red]Input error, using fallback: {e}[/dim red]")
        try:
            user_input = console_manager.console.input(
                prompt=prompt_text, markup=True, emoji=True
            )
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            raise KeyboardInterrupt


def execute_shell_command(command: str) -> bool:
    """Execute shell commands and return True if successful."""
    global current_directory

    parts = command.strip().split()
    if not parts:
        return False

    cmd = parts[0].lower()

    try:
        if cmd == "cd":
            # Handle cd command
            if len(parts) == 1:
                # cd with no arguments goes to home
                new_dir = Path.home()
            else:
                target = " ".join(parts[1:])  # Handle paths with spaces
                if target == "~":
                    new_dir = Path.home()
                elif target.startswith("~/"):
                    new_dir = Path.home() / target[2:]
                else:
                    new_dir = (
                        current_directory / target
                        if not Path(target).is_absolute()
                        else Path(target)
                    )

                new_dir = new_dir.resolve()

            if new_dir.exists() and new_dir.is_dir():
                current_directory = new_dir
                os.chdir(current_directory)
                console.print(f"[grey74]{current_directory}[/grey74]")
                return True
            else:
                console.print(f"[red]cd: no such file or directory: {target}[/red]")
                return True  # We handled it, even if it failed

        elif cmd == "pwd":
            # Print working directory
            console.print(f"[grey74]{current_directory}[/grey74]")
            return True

        elif cmd == "ls":
            # List directory contents with structured output
            target_dir = current_directory
            show_path = ""
            if len(parts) > 1:
                target_path = parts[1]
                show_path = target_path
                if target_path.startswith("~/"):
                    target_dir = Path.home() / target_path[2:]
                else:
                    target_dir = (
                        current_directory / target_path
                        if not Path(target_path).is_absolute()
                        else Path(target_path)
                    )

            if target_dir.exists() and target_dir.is_dir():
                items = list(target_dir.iterdir())
                if not items:
                    console.print(
                        f"[grey50]Empty directory: {show_path or str(target_dir)}[/grey50]"
                    )
                    return True

                # Create a structured table for ls output
                table = Table(
                    title=f"📁 Directory Contents: {show_path or target_dir.name}",
                    box=box.SIMPLE,
                    border_style="blue",
                    show_header=True,
                    title_style="bold blue",
                )
                table.add_column("Name", style="white", min_width=20)
                table.add_column("Type", style="cyan", width=10)
                table.add_column("Size", style="grey74", width=10)
                table.add_column("Modified", style="grey50", width=16)

                # Sort: directories first, then files
                dirs = [item for item in items if item.is_dir()]
                files = [item for item in items if item.is_file()]
                sorted_items = sorted(dirs, key=lambda x: x.name.lower()) + sorted(
                    files, key=lambda x: x.name.lower()
                )

                for item in sorted_items:
                    try:
                        stat = item.stat()
                        if item.is_dir():
                            name = f"[bold blue]{item.name}/[/bold blue]"
                            type_str = "📁 DIR"
                            size_str = "-"
                        else:
                            name = f"[white]{item.name}[/white]"
                            type_str = "📄 FILE"
                            size = stat.st_size
                            if size < 1024:
                                size_str = f"{size}B"
                            elif size < 1024**2:
                                size_str = f"{size/1024:.1f}KB"
                            elif size < 1024**3:
                                size_str = f"{size/1024**2:.1f}MB"
                            else:
                                size_str = f"{size/1024**3:.1f}GB"

                        # Format modification time
                        from datetime import datetime

                        mod_time = datetime.fromtimestamp(stat.st_mtime)
                        mod_str = mod_time.strftime("%Y-%m-%d %H:%M")

                        table.add_row(name, type_str, size_str, mod_str)
                    except (OSError, PermissionError):
                        # If we can't get stats, just show the name
                        name = (
                            f"[bold blue]{item.name}/[/bold blue]"
                            if item.is_dir()
                            else f"[white]{item.name}[/white]"
                        )
                        table.add_row(name, "?", "?", "?")

                console.print(table)
                console.print(
                    f"\n[grey50]Total: {len(dirs)} directories, {len(files)} files[/grey50]"
                )
                return True
            else:
                console.print(
                    f"[red]ls: cannot access '{parts[1] if len(parts) > 1 else target_dir}': No such file or directory[/red]"
                )
                return True

        elif cmd == "cat":
            # Enhanced cat command with syntax highlighting
            if len(parts) < 2:
                console.print("[red]cat: missing file argument[/red]")
                return True

            file_path = " ".join(parts[1:])  # Handle paths with spaces
            if not file_path.startswith("/") and not file_path.startswith("~/"):
                file_path = current_directory / file_path
            else:
                file_path = Path(file_path).expanduser()

            try:
                if file_path.exists() and file_path.is_file():
                    content = file_path.read_text(encoding="utf-8", errors="replace")

                    # Try to guess syntax from extension for highlighting

                    ext = file_path.suffix.lower()

                    # Map common extensions to syntax highlighting
                    language_map = {
                        ".py": "python",
                        ".js": "javascript",
                        ".ts": "typescript",
                        ".html": "html",
                        ".css": "css",
                        ".json": "json",
                        ".xml": "xml",
                        ".yaml": "yaml",
                        ".yml": "yaml",
                        ".sh": "bash",
                        ".bash": "bash",
                        ".zsh": "bash",
                        ".sql": "sql",
                        ".md": "markdown",
                        ".txt": "text",
                        ".log": "text",
                        ".conf": "text",
                        ".cfg": "text",
                    }

                    language = language_map.get(ext, "text")

                    if content.strip():
                        syntax = Syntax(
                            content, language, theme="monokai", line_numbers=True
                        )
                        console.print(
                            Panel(
                                syntax,
                                title=f"[bold blue]📄 {file_path.name}[/bold blue]",
                                border_style="blue",
                                box=box.ROUNDED,
                            )
                        )
                    else:
                        console.print(
                            f"[grey50]📄 {file_path.name} (empty file)[/grey50]"
                        )
                else:
                    console.print(
                        f"[red]cat: {file_path}: No such file or directory[/red]"
                    )
            except PermissionError:
                console.print(f"[red]cat: {file_path}: Permission denied[/red]")
            except UnicodeDecodeError:
                console.print(
                    f"[red]cat: {file_path}: Binary file (cannot display)[/red]"
                )
            except Exception as e:
                console.print(f"[red]cat: {file_path}: {e}[/red]")

            return True

        elif cmd == "open":
            # Handle open command to open files or folders
            if len(parts) < 2:
                console.print("[red]open: missing file or folder argument[/red]")
                return True

            file_or_folder = " ".join(parts[1:])  # Handle paths with spaces

            # Resolve path relative to current directory if not absolute
            if not file_or_folder.startswith("/") and not file_or_folder.startswith(
                "~/"
            ):
                target_path = current_directory / file_or_folder
            else:
                target_path = Path(file_or_folder).expanduser()

            if not target_path.exists():
                console.print(
                    f"[red]open: '{file_or_folder}': No such file or directory[/red]"
                )
                return True

            # Open file or folder using centralized system utility
            success, message = open_path(target_path)

            if success:
                # Format success message with appropriate icon
                if target_path.is_dir():
                    console.print(f"[green]📁 {message}[/green]")
                else:
                    console.print(f"[green]📄 {message}[/green]")
            else:
                console.print(f"[red]open: {message}[/red]")

            return True

        elif cmd == "mkdir":
            # Create directory using pathlib (safe, no shell injection)
            if len(parts) < 2:
                console.print("[red]mkdir: missing operand[/red]")
                return True

            dir_path = " ".join(parts[1:])  # Handle paths with spaces
            target_dir = (
                current_directory / dir_path
                if not Path(dir_path).is_absolute()
                else Path(dir_path)
            )

            try:
                target_dir.mkdir(parents=True, exist_ok=False)
                console.print(f"[green]📁 Created directory: {parts[1]}[/green]")
            except FileExistsError:
                console.print(
                    f"[red]mkdir: cannot create directory '{parts[1]}': File exists[/red]"
                )
            except Exception as e:
                console.print(f"[red]mkdir: {e}[/red]")

            return True

        elif cmd == "touch":
            # Create file using pathlib (safe, no shell injection)
            if len(parts) < 2:
                console.print("[red]touch: missing file operand[/red]")
                return True

            file_path = " ".join(parts[1:])  # Handle paths with spaces
            target_file = (
                current_directory / file_path
                if not Path(file_path).is_absolute()
                else Path(file_path)
            )

            try:
                target_file.touch()
                console.print(f"[green]📄 Created file: {parts[1]}[/green]")
            except Exception as e:
                console.print(f"[red]touch: {e}[/red]")

            return True

        elif cmd == "cp":
            # Copy file using shutil (safe, no shell injection)
            if len(parts) < 3:
                console.print("[red]cp: missing file operand[/red]")
                return True

            src = parts[1]
            dst = parts[2]
            src_path = (
                current_directory / src if not Path(src).is_absolute() else Path(src)
            )
            dst_path = (
                current_directory / dst if not Path(dst).is_absolute() else Path(dst)
            )

            try:
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                console.print(f"[green]📋 Copied: {parts[1]} → {parts[2]}[/green]")
            except Exception as e:
                console.print(f"[red]cp: {e}[/red]")

            return True

        elif cmd == "mv":
            # Move file using shutil (safe, no shell injection)
            if len(parts) < 3:
                console.print("[red]mv: missing file operand[/red]")
                return True

            src = parts[1]
            dst = parts[2]
            src_path = (
                current_directory / src if not Path(src).is_absolute() else Path(src)
            )
            dst_path = (
                current_directory / dst if not Path(dst).is_absolute() else Path(dst)
            )

            try:
                shutil.move(str(src_path), str(dst_path))
                console.print(f"[green]📦 Moved: {parts[1]} → {parts[2]}[/green]")
            except Exception as e:
                console.print(f"[red]mv: {e}[/red]")

            return True

        elif cmd == "rm":
            # Remove file using pathlib (safe, no shell injection)
            if len(parts) < 2:
                console.print("[red]rm: missing operand[/red]")
                return True

            file_path = " ".join(parts[1:])  # Handle paths with spaces
            target_path = (
                current_directory / file_path
                if not Path(file_path).is_absolute()
                else Path(file_path)
            )

            try:
                if target_path.is_dir():
                    # For directories, require explicit -r flag
                    if "-r" in parts or "-rf" in parts:
                        shutil.rmtree(target_path)
                        console.print(
                            f"[green]🗑️  Removed directory: {parts[1]}[/green]"
                        )
                    else:
                        console.print(
                            f"[red]rm: cannot remove '{parts[1]}': Is a directory (use -r to remove directories)[/red]"
                        )
                else:
                    target_path.unlink()
                    console.print(f"[green]🗑️  Removed: {parts[1]}[/green]")
            except FileNotFoundError:
                console.print(
                    f"[red]rm: cannot remove '{parts[1]}': No such file or directory[/red]"
                )
            except Exception as e:
                console.print(f"[red]rm: {e}[/red]")

            return True

        else:
            # Not a recognized shell command
            return False

    except Exception as e:
        console.print(f"[red]Error executing command: {e}[/red]")
        return True  # We handled it, even if it failed


def get_current_agent_name() -> str:
    """Get the current active agent name for display."""
    global client
    if client and hasattr(client, "callbacks") and client.callbacks:
        for callback in client.callbacks:
            if isinstance(callback, TerminalCallbackHandler):
                if hasattr(callback, "current_agent") and callback.current_agent:
                    # Format the agent name properly
                    agent_name = callback.current_agent.replace("_", " ").title()
                    return f"🦞 {agent_name}"
                # Check if there are any recent events that might indicate the active agent
                elif hasattr(callback, "events") and callback.events:
                    # Get the most recent agent from events
                    for event in reversed(callback.events):
                        if (
                            event.agent_name
                            and event.agent_name != "system"
                            and event.agent_name != "unknown"
                        ):
                            agent_name = event.agent_name.replace("_", " ").title()
                            return f"🦞 {agent_name}"
                break
    return "🦞 Lobster"


def _dna_helix_animation(width: int, duration: float = 0.7):
    """
    DNA sequence animation with colorful bases (A, T, C, G) flowing across the terminal.
    Uses only capital DNA letters with biochemistry-inspired colors.
    """
    import sys
    import math

    # DNA base colors (biochemistry-inspired)
    base_colors = {
        'A': (0, 200, 83),    # Adenine - green (purine)
        'T': (255, 82, 82),   # Thymine - red (pyrimidine)
        'G': (255, 193, 7),   # Guanine - gold (purine)
        'C': (41, 121, 255),  # Cytosine - blue (pyrimidine)
    }

    lobster_orange = (228, 92, 71)
    white = (255, 255, 255)

    num_frames = max(40, int(duration * 90))
    frame_sleep = duration / num_frames

    # Generate a random DNA sequence (only capital letters A, T, C, G)
    sequence = ''.join(random.choices(['A', 'T', 'G', 'C'], k=width))

    # Phase 1: DNA sequence assembly (flowing bases)
    assembly_frames = int(num_frames * 0.45)
    for frame in range(assembly_frames):
        progress = (frame + 1) / assembly_frames
        reveal_pos = int(width * progress)

        line_parts = []
        for i in range(width):
            if i < reveal_pos:
                base = sequence[i]
                r, g, b = base_colors[base]
                # Add glow effect for recently revealed bases
                if i > reveal_pos - 5:
                    glow = 1.3 - (reveal_pos - i) * 0.06
                    r = min(255, int(r * glow))
                    g = min(255, int(g * glow))
                    b = min(255, int(b * glow))
                line_parts.append(f'\033[38;2;{r};{g};{b};1m{base}\033[0m')
            else:
                line_parts.append(' ')

        sys.stdout.write('\r' + ''.join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 0.5)

    # Phase 2: DNA wave intensity effect (only capital letters)
    helix_frames = int(num_frames * 0.35)
    for frame in range(helix_frames):
        line_parts = []
        wave_offset = frame * 0.4

        for i in range(width):
            base = sequence[i]
            # Sinusoidal wave for intensity variation (simulates depth)
            wave = math.sin((i * 0.25) + wave_offset)
            intensity = 0.5 + 0.5 * abs(wave)

            r, g, b = base_colors[base]
            r = int(r * intensity)
            g = int(g * intensity)
            b = int(b * intensity)

            line_parts.append(f'\033[38;2;{r};{g};{b};1m{base}\033[0m')

        sys.stdout.write('\r' + ''.join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 0.8)

    # Phase 3: Orange sweep with centered "─ omics-os ─" brand text
    sweep_frames = num_frames - assembly_frames - helix_frames
    brand_text = "─ omics-os ─"
    text_start = (width - len(brand_text)) // 2
    text_end = text_start + len(brand_text)

    for frame in range(sweep_frames + 1):
        progress = frame / sweep_frames if sweep_frames > 0 else 1
        sweep_pos = int(width * progress)

        line_parts = []
        for i in range(width):
            if i < sweep_pos:
                # Check if this position is part of the brand text
                if text_start <= i < text_end:
                    char_idx = i - text_start
                    char = brand_text[char_idx]
                    # White text on orange background
                    r, g, b = lobster_orange
                    line_parts.append(f'\033[48;2;{r};{g};{b}m\033[38;2;255;255;255;1m{char}\033[0m')
                else:
                    # Gradient orange fill
                    gradient = 1.0 - (sweep_pos - i) / max(width, 1) * 0.12
                    r = min(255, int(lobster_orange[0] * gradient))
                    g = min(255, int(lobster_orange[1] * gradient))
                    b = min(255, int(lobster_orange[2] * gradient))
                    line_parts.append(f'\033[48;2;{r};{g};{b}m \033[0m')
            elif i == sweep_pos:
                # Bright leading edge
                line_parts.append(f'\033[48;2;{white[0]};{white[1]};{white[2]}m \033[0m')
            else:
                # Fading DNA bases behind
                base = sequence[i]
                fade = max(0.15, 1.0 - (i - sweep_pos) * 0.08)
                r, g, b = base_colors[base]
                r, g, b = int(r * fade), int(g * fade), int(b * fade)
                line_parts.append(f'\033[38;2;{r};{g};{b};1m{base}\033[0m')

        sys.stdout.write('\r' + ''.join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 1.0)

    # Final orange bar with centered "─ omics-os ─"
    r, g, b = lobster_orange
    final_parts = []
    for i in range(width):
        if text_start <= i < text_end:
            char = brand_text[i - text_start]
            final_parts.append(f'\033[48;2;{r};{g};{b}m\033[38;2;255;255;255;1m{char}\033[0m')
        else:
            final_parts.append(f'\033[48;2;{r};{g};{b}m \033[0m')
    sys.stdout.write('\r' + ''.join(final_parts) + '\n')
    sys.stdout.flush()


def display_welcome():
    """Display DNA sequence animation as bioinformatics-themed startup visualization."""
    import sys

    term_width = shutil.get_terminal_size().columns
    sys.stdout.write("\n")
    _dna_helix_animation(term_width, duration=random.uniform(0.6, 0.8))
    sys.stdout.write("\n")


def _dna_agent_loading_phase(width: int, agent_names: List[str], ready_queue=None, timeout: float = 10.0):
    """
    DNA-themed agent loading animation showing real-time progress.

    Displays a progress bar made of DNA bases that fills as agents load,
    with each agent name appearing with a DNA sequencing effect.
    """
    import sys
    import math
    import queue as queue_module

    # DNA base colors (biochemistry-inspired)
    base_colors = {
        'A': (0, 200, 83),    # Adenine - green
        'T': (255, 82, 82),   # Thymine - red
        'G': (255, 193, 7),   # Guanine - gold
        'C': (41, 121, 255),  # Cytosine - blue
    }
    lobster_orange = (228, 92, 71)

    total_agents = len(agent_names)
    loaded_count = 0
    start_time = time.time()

    # Generate DNA sequence for progress bar
    sequence = ''.join(random.choices(['A', 'T', 'G', 'C'], k=width))

    def render_progress_bar(progress: float, agent_name: str = "", wave_frame: int = 0):
        """Render the DNA progress bar with current progress."""
        filled_width = int(width * progress)

        line_parts = []
        for i in range(width):
            base = sequence[i]
            r, g, b = base_colors[base]

            if i < filled_width:
                # Filled section: bright with wave effect
                wave = math.sin((i * 0.3) + (wave_frame * 0.3))
                intensity = 0.8 + 0.2 * abs(wave)
                r = min(255, int(r * intensity))
                g = min(255, int(g * intensity))
                b = min(255, int(b * intensity))
                line_parts.append(f'\033[38;2;{r};{g};{b};1m{base}\033[0m')
            elif i == filled_width and progress < 1.0:
                # Leading edge: white glow
                line_parts.append(f'\033[38;2;255;255;255;1m{base}\033[0m')
            else:
                # Unfilled section: dim
                r, g, b = int(r * 0.2), int(g * 0.2), int(b * 0.2)
                line_parts.append(f'\033[38;2;{r};{g};{b};1m{base}\033[0m')

        # Clear line and render
        sys.stdout.write('\r' + ''.join(line_parts))
        sys.stdout.flush()

        # Show agent name below progress bar (if provided)
        if agent_name:
            # Create compact DNA spinner
            spinner_bases = ['A', 'T', 'G', 'C']
            spinner_idx = wave_frame % len(spinner_bases)
            spinner_base = spinner_bases[spinner_idx]
            r, g, b = base_colors[spinner_base]

            # Format agent name with DNA spinner
            agent_text = f"  \033[38;2;{r};{g};{b};1m{spinner_base}\033[0m \033[38;2;200;200;200m{agent_name}\033[0m"

            # Move cursor down, print agent name, move cursor back up
            sys.stdout.write(f"\n{agent_text}\033[K")  # \033[K clears to end of line
            sys.stdout.write("\033[1A")  # Move cursor up 1 line
            sys.stdout.flush()

    # Initial render: empty progress bar
    render_progress_bar(0.0)
    sys.stdout.write("\n\n")  # Reserve space for agent name
    sys.stdout.write("\033[2A")  # Move cursor back to progress bar line
    sys.stdout.flush()

    # Animation loop
    wave_frame = 0
    last_agent_name = ""

    if ready_queue:
        # Real-time mode: wait for agent notifications
        while loaded_count < total_agents:
            # Check for timeout
            if time.time() - start_time > timeout:
                break

            try:
                # Non-blocking check for new agent
                agent_loaded = ready_queue.get(timeout=0.05)
                if agent_loaded == "__done__":
                    break
                loaded_count += 1
                last_agent_name = agent_loaded

                # Animate progress increase
                old_progress = (loaded_count - 1) / total_agents
                new_progress = loaded_count / total_agents

                # Smooth transition over 10 frames
                for frame in range(10):
                    progress = old_progress + (new_progress - old_progress) * (frame / 10)
                    render_progress_bar(progress, last_agent_name, wave_frame)
                    wave_frame += 1
                    time.sleep(0.03)

            except queue_module.Empty:
                # Keep animating while waiting
                progress = loaded_count / total_agents
                render_progress_bar(progress, last_agent_name, wave_frame)
                wave_frame += 1
                time.sleep(0.05)
    else:
        # Fallback mode: simulate loading (no real progress)
        estimated_time_per_agent = timeout / total_agents

        for agent_idx, agent_name in enumerate(agent_names):
            agent_start = time.time()

            # Animate this agent loading
            while time.time() - agent_start < estimated_time_per_agent:
                progress = (agent_idx + (time.time() - agent_start) / estimated_time_per_agent) / total_agents
                progress = min(progress, 1.0)
                render_progress_bar(progress, agent_name, wave_frame)
                wave_frame += 1
                time.sleep(0.05)

            loaded_count += 1

    # Final render: complete bar
    render_progress_bar(1.0, "ready", wave_frame)

    # Hold final state briefly
    for _ in range(10):
        render_progress_bar(1.0, "ready", wave_frame)
        wave_frame += 1
        time.sleep(0.04)

    # Clear the agent name line and progress bar
    sys.stdout.write("\r" + " " * width + "\n")  # Clear progress bar
    sys.stdout.write(" " * width + "\n")  # Clear agent name
    sys.stdout.write("\033[2A\r")  # Move cursor back up and to start
    sys.stdout.flush()


def _dna_exit_animation(width: int, duration: float = 0.5):
    """
    DNA exit animation - reverse of startup animation.
    Orange bar dissolves back into DNA bases, which then fade out.
    The "─ omics-os ─" brand text ALWAYS stays lobster orange.
    """
    import sys
    import math

    # DNA base colors (biochemistry-inspired)
    base_colors = {
        'A': (0, 200, 83),    # Adenine - green (purine)
        'T': (255, 82, 82),   # Thymine - red (pyrimidine)
        'G': (255, 193, 7),   # Guanine - gold (purine)
        'C': (41, 121, 255),  # Cytosine - blue (pyrimidine)
    }

    lobster_orange = (228, 92, 71)
    white = (255, 255, 255)

    num_frames = max(30, int(duration * 70))
    frame_sleep = duration / num_frames

    # Generate a random DNA sequence
    sequence = ''.join(random.choices(['A', 'T', 'G', 'C'], k=width))

    brand_text = "─ omics-os ─"
    text_start = (width - len(brand_text)) // 2
    text_end = text_start + len(brand_text)

    # Phase 1: Start with full orange bar, then reverse sweep (right to left dissolve)
    # Brand text stays full orange throughout
    dissolve_frames = int(num_frames * 0.5)
    for frame in range(dissolve_frames + 1):
        progress = frame / dissolve_frames if dissolve_frames > 0 else 1
        # Dissolve from right to left
        dissolve_pos = width - int(width * progress)

        line_parts = []
        for i in range(width):
            # Brand text ALWAYS stays lobster orange (no background during dissolve)
            if text_start <= i < text_end:
                char = brand_text[i - text_start]
                r, g, b = lobster_orange
                line_parts.append(f'\033[38;2;{r};{g};{b};1m{char}\033[0m')
            elif i >= dissolve_pos:
                # DNA bases emerge from dissolution
                base = sequence[i]
                r, g, b = base_colors[base]
                # Glow effect for recently revealed bases
                if i < dissolve_pos + 5:
                    glow = 1.3 - (i - dissolve_pos) * 0.06
                    r = min(255, int(r * glow))
                    g = min(255, int(g * glow))
                    b = min(255, int(b * glow))
                line_parts.append(f'\033[38;2;{r};{g};{b};1m{base}\033[0m')
            elif i == dissolve_pos - 1 and dissolve_pos > 0:
                # Bright dissolving edge
                line_parts.append(f'\033[48;2;{white[0]};{white[1]};{white[2]}m \033[0m')
            else:
                # Orange bar fading as it dissolves
                fade = max(0.4, 1.0 - progress * 0.6)
                r, g, b = lobster_orange
                r, g, b = int(r * fade), int(g * fade), int(b * fade)
                line_parts.append(f'\033[48;2;{r};{g};{b}m \033[0m')

        sys.stdout.write('\r' + ''.join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 0.8)

    # Phase 2: DNA wave effect with brand text staying orange
    wave_frames = int(num_frames * 0.25)
    for frame in range(wave_frames):
        line_parts = []
        wave_offset = frame * 0.5

        for i in range(width):
            # Brand text ALWAYS stays lobster orange
            if text_start <= i < text_end:
                char = brand_text[i - text_start]
                r, g, b = lobster_orange
                line_parts.append(f'\033[38;2;{r};{g};{b};1m{char}\033[0m')
            else:
                base = sequence[i]
                wave = math.sin((i * 0.25) + wave_offset)
                intensity = 0.5 + 0.5 * abs(wave)

                r, g, b = base_colors[base]
                r = int(r * intensity)
                g = int(g * intensity)
                b = int(b * intensity)

                line_parts.append(f'\033[38;2;{r};{g};{b};1m{base}\033[0m')

        sys.stdout.write('\r' + ''.join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 0.6)

    # Phase 3: Fade out DNA sequence, brand text stays orange then fades last
    fade_frames = num_frames - dissolve_frames - wave_frames
    for frame in range(fade_frames + 1):
        progress = frame / fade_frames if fade_frames > 0 else 1
        fade_intensity = 1.0 - progress
        # Brand fades slower - only starts fading at 70% progress
        brand_fade = max(0, 1.0 - max(0, (progress - 0.7) / 0.3))

        line_parts = []
        for i in range(width):
            # Brand text stays orange longer, then fades last
            if text_start <= i < text_end:
                char = brand_text[i - text_start]
                r, g, b = lobster_orange
                if brand_fade > 0.05:
                    r = int(r * brand_fade)
                    g = int(g * brand_fade)
                    b = int(b * brand_fade)
                    line_parts.append(f'\033[38;2;{r};{g};{b};1m{char}\033[0m')
                else:
                    line_parts.append(' ')
            elif fade_intensity > 0.05:
                base = sequence[i]
                r, g, b = base_colors[base]
                r = int(r * fade_intensity)
                g = int(g * fade_intensity)
                b = int(b * fade_intensity)
                line_parts.append(f'\033[38;2;{r};{g};{b};1m{base}\033[0m')
            else:
                line_parts.append(' ')

        sys.stdout.write('\r' + ''.join(line_parts))
        sys.stdout.flush()
        time.sleep(frame_sleep * 1.0)

    # Clear the line
    sys.stdout.write('\r' + ' ' * width + '\r')
    sys.stdout.flush()


def display_goodbye():
    """Display DNA exit animation as bioinformatics-themed farewell visualization."""
    import sys

    term_width = shutil.get_terminal_size().columns
    sys.stdout.write("\n")
    _dna_exit_animation(term_width, duration=random.uniform(0.4, 0.6))


def show_default_help():
    """Display default help guide when lobster is run without subcommands."""
    # Create branded header
    header_text = LobsterTheme.create_title_text("LOBSTER by Omics-OS", "🦞")

    help_content = f"""[bold white]Multi-Agent Bioinformatics Analysis System v{__version__}[/bold white]

[bold {LobsterTheme.PRIMARY_ORANGE}]AVAILABLE COMMANDS:[/bold {LobsterTheme.PRIMARY_ORANGE}]

  [{LobsterTheme.PRIMARY_ORANGE}]lobster chat[/{LobsterTheme.PRIMARY_ORANGE}]      [grey50]-[/grey50] Start interactive chat session with AI agents
  [{LobsterTheme.PRIMARY_ORANGE}]lobster query[/{LobsterTheme.PRIMARY_ORANGE}]     [grey50]-[/grey50] Send a single analysis query
  [{LobsterTheme.PRIMARY_ORANGE}]lobster serve[/{LobsterTheme.PRIMARY_ORANGE}]     [grey50]-[/grey50] Start API server for web services
  [{LobsterTheme.PRIMARY_ORANGE}]lobster config[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] Manage agent configuration

[bold {LobsterTheme.PRIMARY_ORANGE}]QUICK START:[/bold {LobsterTheme.PRIMARY_ORANGE}]

  [white]# Start interactive analysis with enhanced autocomplete[/white]
  [{LobsterTheme.PRIMARY_ORANGE}]lobster chat[/{LobsterTheme.PRIMARY_ORANGE}]

  [white]# Show agent reasoning during analysis[/white]
  [{LobsterTheme.PRIMARY_ORANGE}]lobster chat --reasoning[/{LobsterTheme.PRIMARY_ORANGE}]

  [white]# Send a single query and get results[/white]
  [{LobsterTheme.PRIMARY_ORANGE}]lobster query "Analyze my RNA-seq data"[/{LobsterTheme.PRIMARY_ORANGE}]

  [white]# Start API server on custom port[/white]
  [{LobsterTheme.PRIMARY_ORANGE}]lobster serve --port 8080[/{LobsterTheme.PRIMARY_ORANGE}]

[bold {LobsterTheme.PRIMARY_ORANGE}]CONFIGURATION:[/bold {LobsterTheme.PRIMARY_ORANGE}]

  [{LobsterTheme.PRIMARY_ORANGE}]lobster config list-models[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] List available AI models
  [{LobsterTheme.PRIMARY_ORANGE}]lobster config list-profiles[/{LobsterTheme.PRIMARY_ORANGE}]  [grey50]-[/grey50] List testing profiles
  [{LobsterTheme.PRIMARY_ORANGE}]lobster config show-config[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] Show current configuration

[bold {LobsterTheme.PRIMARY_ORANGE}]KEY FEATURES:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [white]Single-Cell & Bulk RNA-seq Analysis[/white]
• [white]Mass Spectrometry & Affinity Proteomics[/white]
• [white]GEO Dataset Access & Literature Mining[/white]
• [white]Interactive Visualizations & Reports[/white]
• [white]Natural Language Interface[/white]

[bold {LobsterTheme.PRIMARY_ORANGE}]HELP & DOCUMENTATION:[/bold {LobsterTheme.PRIMARY_ORANGE}]

  [{LobsterTheme.PRIMARY_ORANGE}]lobster --help[/{LobsterTheme.PRIMARY_ORANGE}]              [grey50]-[/grey50] Show detailed help
  [{LobsterTheme.PRIMARY_ORANGE}]lobster <command> --help[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] Show help for specific command

[dim grey50]🌐 Website: https://omics-os.com | 📚 Docs: https://github.com/the-omics-os[/dim grey50]
[dim grey50]Powered by LangGraph | © 2025 Omics-OS[/dim grey50]"""

    # Create branded help panel
    help_panel = LobsterTheme.create_panel(help_content, title=str(header_text))

    console_manager.print(help_panel)


# ============================================================================
# Helper Functions for Data Display
# ============================================================================


def _format_data_preview(matrix, max_rows: int = 5, max_cols: int = 5) -> Table:
    """Format a data matrix preview as a Rich table."""
    import numpy as np
    import scipy.sparse as sp

    # Convert sparse to dense for preview if needed
    if sp.issparse(matrix):
        # Get a small subset for preview
        preview_rows = min(max_rows, matrix.shape[0])
        preview_cols = min(max_cols, matrix.shape[1])
        preview_data = matrix[:preview_rows, :preview_cols].toarray()
    else:
        preview_rows = min(max_rows, matrix.shape[0])
        preview_cols = min(max_cols, matrix.shape[1])
        preview_data = matrix[:preview_rows, :preview_cols]

    # Create table
    table = Table(box=box.SIMPLE)

    # Add columns
    table.add_column("", style="bold grey50")  # Row index
    for i in range(preview_cols):
        table.add_column(f"[{i}]", style="cyan")

    # Add rows
    for i in range(preview_rows):
        row_values = ["[" + str(i) + "]"]
        for j in range(preview_cols):
            val = preview_data[i, j]
            # Format the value
            if isinstance(val, (int, np.integer)):
                formatted = str(val)
            elif isinstance(val, (float, np.floating)):
                formatted = f"{val:.2f}"
            else:
                formatted = str(val)
            row_values.append(formatted)
        table.add_row(*row_values)

    # Add ellipsis row if there are more rows
    if matrix.shape[0] > max_rows or matrix.shape[1] > max_cols:
        ellipsis_row = ["..."] * (min(preview_cols, matrix.shape[1]) + 1)
        table.add_row(*ellipsis_row, style="dim")

    return table


def _format_dataframe_preview(df: "pd.DataFrame", max_rows: int = 5) -> Table:
    """Format a DataFrame preview as a Rich table."""
    import numpy as np
    import pandas as pd

    table = Table(box=box.SIMPLE)

    # Add index column
    table.add_column("Index", style="bold grey50")

    # Add data columns
    for col in df.columns[:10]:  # Limit to first 10 columns
        dtype_str = str(df[col].dtype)
        style = (
            "cyan"
            if dtype_str.startswith("int") or dtype_str.startswith("float")
            else "white"
        )
        table.add_column(str(col), style=style)

    # Add rows
    preview_rows = min(max_rows, len(df))
    for idx in range(preview_rows):
        row_data = [str(df.index[idx])]
        for col in df.columns[:10]:
            val = df.iloc[idx][col]
            # Format based on type
            if pd.isna(val):
                formatted = "NaN"
            elif isinstance(val, (int, np.integer)):
                formatted = str(val)
            elif isinstance(val, (float, np.floating)):
                formatted = f"{val:.2f}"
            else:
                formatted = str(val)[:20]  # Truncate long strings
            row_data.append(formatted)
        table.add_row(*row_data)

    # Add ellipsis if there are more rows
    if len(df) > max_rows:
        ellipsis_row = ["..."] * (min(10, len(df.columns)) + 1)
        table.add_row(*ellipsis_row, style="dim")

    # Add more columns indicator
    if len(df.columns) > 10:
        table.add_column(f"... +{len(df.columns) - 10} more", style="dim")

    return table


def _format_array_info(arrays_dict: Dict[str, "np.ndarray"]) -> Table:
    """Format array information (obsm/varm) as a table."""
    import numpy as np

    if not arrays_dict:
        return None

    table = Table(box=box.SIMPLE)
    table.add_column("Key", style="bold cyan")
    table.add_column("Shape", style="white")
    table.add_column("Dtype", style="grey70")

    for key, arr in arrays_dict.items():
        shape_str = " × ".join(str(d) for d in arr.shape)
        dtype_str = str(arr.dtype)
        table.add_row(key, shape_str, dtype_str)

    return table


def _get_matrix_info(matrix) -> Dict[str, Any]:
    """Get information about a matrix (sparse or dense)."""
    import scipy.sparse as sp

    info = {}
    info["shape"] = matrix.shape
    info["dtype"] = str(matrix.dtype)

    if sp.issparse(matrix):
        info["sparse"] = True
        info["format"] = matrix.format.upper()
        info["nnz"] = matrix.nnz
        info["density"] = (matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100
        info["memory_mb"] = (
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        ) / (1024**2)
    else:
        info["sparse"] = False
        info["format"] = "Dense"
        info["memory_mb"] = matrix.nbytes / (1024**2)
        info["density"] = 100.0

    return info


def _display_status_info():
    """
    Display subscription tier, installed packages, and available agents.

    Shared implementation for both 'lobster status' CLI command and '/status' chat command.
    This function is client-independent and shows installation/license information.
    """
    from rich.table import Table

    console.print()
    console.print(
        Panel.fit(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]🦞 Lobster Status[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            padding=(0, 2),
        )
    )
    console.print()

    # Check initialization status
    env_file = Path.cwd() / ".env"
    is_initialized = env_file.exists()

    if is_initialized:
        console.print("[bold]Initialization:[/bold] ✅ Configured")
        # Try to detect provider from .env
        try:
            from dotenv import dotenv_values
            env_vars = dotenv_values(env_file)
            provider = env_vars.get("LOBSTER_LLM_PROVIDER")
            if provider:
                console.print(f"[dim]Provider: {provider}[/dim]")
            else:
                # Auto-detect from available keys
                if env_vars.get("ANTHROPIC_API_KEY"):
                    console.print("[dim]Provider: anthropic (auto-detected)[/dim]")
                elif env_vars.get("AWS_BEDROCK_ACCESS_KEY"):
                    console.print("[dim]Provider: bedrock (auto-detected)[/dim]")
                elif env_vars.get("OLLAMA_BASE_URL"):
                    console.print("[dim]Provider: ollama (auto-detected)[/dim]")
        except Exception:
            pass  # Silently skip if dotenv not available
        console.print(f"[dim]Config file: {env_file}[/dim]")
    else:
        console.print("[bold]Initialization:[/bold] ❌ Not configured")
        console.print(
            Panel.fit(
                "[yellow]⚠️  Lobster is not initialized yet[/yellow]\n\n"
                f"Run: [bold {LobsterTheme.PRIMARY_ORANGE}]lobster init[/bold {LobsterTheme.PRIMARY_ORANGE}]\n\n"
                "[dim]This will configure your LLM provider (Anthropic/Bedrock/Ollama)[/dim]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    console.print()

    # Get entitlement status
    try:
        from lobster.core.license_manager import get_entitlement_status

        entitlement = get_entitlement_status()
    except ImportError:
        entitlement = {"tier": "free", "tier_display": "Free", "source": "default"}

    # Get installed packages
    try:
        from lobster.core.plugin_loader import get_installed_packages

        packages = get_installed_packages()
    except ImportError:
        packages = {"lobster-ai": "unknown"}

    # Get available agents
    try:
        from lobster.config.agent_registry import get_worker_agents
        from lobster.config.subscription_tiers import is_agent_available

        worker_agents = get_worker_agents()
        tier = entitlement.get("tier", "free")
        available = [name for name in worker_agents if is_agent_available(name, tier)]
        restricted = [
            name for name in worker_agents if not is_agent_available(name, tier)
        ]
    except ImportError:
        available = []
        restricted = []
        tier = "free"

    # Subscription tier section
    tier_display = entitlement.get("tier_display", "Free")
    tier_emoji = {"free": "🆓", "premium": "⭐", "enterprise": "🏢"}.get(
        entitlement.get("tier", "free"), "🆓"
    )

    console.print(f"[bold]Subscription Tier:[/bold] {tier_emoji} {tier_display}")
    console.print(f"[dim]Source: {entitlement.get('source', 'default')}[/dim]")

    if entitlement.get("expires_at"):
        days = entitlement.get("days_until_expiry")
        if days is not None and days < 30:
            console.print(f"[yellow]⚠️  License expires in {days} days[/yellow]")
        else:
            console.print(f"[dim]Expires: {entitlement.get('expires_at')}[/dim]")

    if entitlement.get("warnings"):
        for warning in entitlement["warnings"]:
            console.print(f"[red]⚠️  {warning}[/red]")

    console.print()

    # Installed packages table
    console.print("[bold]Installed Packages:[/bold]")
    pkg_table = Table(box=box.ROUNDED, border_style="cyan", show_header=True)
    pkg_table.add_column("Package", style="white")
    pkg_table.add_column("Version", style="cyan")
    pkg_table.add_column("Status", style="green")

    for pkg_name, version in packages.items():
        if version == "missing":
            status_str = "[red]Missing[/red]"
        elif version == "dev":
            status_str = "[yellow]Development[/yellow]"
        else:
            status_str = "[green]Installed[/green]"
        pkg_table.add_row(pkg_name, version, status_str)

    console.print(pkg_table)
    console.print()

    # Available agents
    if available:
        console.print(f"[bold]Available Agents ({len(available)}):[/bold]")
        agent_list = ", ".join(sorted(available))
        console.print(f"[green]{agent_list}[/green]")
        console.print()

    # Restricted agents (upgrade prompt)
    if restricted:
        console.print(f"[bold]Premium Agents ({len(restricted)}):[/bold]")
        restricted_list = ", ".join(sorted(restricted))
        console.print(f"[dim]{restricted_list}[/dim]")
        console.print()
        console.print(
            Panel.fit(
                f"[yellow]⭐ Upgrade to Premium to unlock {len(restricted)} additional agents[/yellow]\n"
                f"[dim]Visit https://omics-os.com/pricing or run 'lobster activate <code>'[/dim]",
                border_style="yellow",
                padding=(0, 2),
            )
        )

    # Features
    features = entitlement.get("features", [])
    if features:
        console.print()
        console.print("[bold]Enabled Features:[/bold]")
        console.print(f"[cyan]{', '.join(features)}[/cyan]")


def display_session(client: "AgentClient"):
    """Display current session status with enhanced orange theming."""
    status = client.get_status()

    # Get current mode/profile
    configurator = get_agent_configurator()
    current_mode = configurator.get_current_profile()

    # Prepare status data for the themed status panel
    status_data = {
        "session_id": status["session_id"],
        "mode": current_mode,
        "messages": str(status["message_count"]),
        "workspace": status["workspace"],
        "data_loaded": status["has_data"],
    }

    # Add data summary if available
    if status["has_data"] and status["data_summary"]:
        summary = status["data_summary"]
        status_data["data_shape"] = str(summary.get("shape", "N/A"))
        status_data["memory_usage"] = summary.get("memory_usage", "N/A")

    # Use the themed status panel
    console_manager.print_status_panel(status_data, "Session Status")


def _show_workspace_prompt(client):
    """Display minimal oh-my-zsh style status with system information."""
    try:
        import psutil
        import shutil
        from lobster.core.license_manager import get_current_tier
        from lobster.config.agent_registry import AGENT_REGISTRY, _ensure_plugins_loaded
        from lobster.tools.gpu_detector import GPUDetector

        # Ensure plugins are loaded before accessing AGENT_REGISTRY
        _ensure_plugins_loaded()

        tier = get_current_tier()
        tier_color = {"free": "green", "premium": "yellow", "enterprise": "magenta"}.get(tier, "white")

        # Count agents
        child_agent_names = set()
        for config in AGENT_REGISTRY.values():
            if config.child_agents:
                child_agent_names.update(config.child_agents)
        agent_count = sum(
            1 for name, config in AGENT_REGISTRY.items()
            if config.supervisor_accessible is not False and name not in child_agent_names
        )

        # Get system information
        # RAM
        ram_gb = round(psutil.virtual_memory().total / (1024**3))

        # GPU
        hw_rec = GPUDetector.get_hardware_recommendation()
        gpu_label = hw_rec["device"].upper()  # CUDA, MPS, or CPU

        # Disk space (workspace)
        try:
            workspace_path = client.workspace_path if hasattr(client, 'workspace_path') else Path.cwd()
            disk_usage = shutil.disk_usage(workspace_path)
            disk_free_gb = round(disk_usage.free / (1024**3))
        except Exception:
            disk_free_gb = "?"

        # Line 1: Application status
        console.print(f"  [{tier_color}]●[/] lobster v{__version__} [{tier_color}]{tier}[/] [dim]│[/] {agent_count} agents [dim]│ local │[/] [dim italic]/help[/]")

        # Line 2-3: System resources (stacked for clarity)
        console.print(f"[dim]  └─ Compute:[/] {ram_gb}GB RAM [dim]│[/] {gpu_label}")
        console.print(f"[dim]     Storage:[/] {disk_free_gb}GB free [dim](workspace)[/]")
        console.print()
    except Exception as e:
        # Fallback to basic prompt if system info fails
        console.print(f"[dim red]Error loading system info: {e}[/dim red]")
        console.print(f"  [green]●[/] lobster v{__version__}")
        console.print()


def init_client_with_animation(
    workspace: Optional[Path] = None,
    reasoning: bool = False,
    verbose: bool = False,
    debug: bool = False,
    profile_timings: Optional[bool] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    session_id: Optional[str] = None,
) -> "AgentClient":
    """
    Initialize client. Fast startup thanks to lazy imports.
    """
    get_console_manager()

    # Initialize client - lazy imports make this fast
    client = init_client(workspace, reasoning, verbose, debug, profile_timings, provider_override, model_override, session_id)

    return client


@app.command()
def config_test(
    output_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON (machine-readable)",
    ),
):
    """Test API connectivity and validate configuration."""
    import json as json_module

    from dotenv import load_dotenv

    # Results structure for JSON output
    test_results = {
        "valid": False,
        "env_file": None,
        "checks": {
            "llm_provider": {"status": "fail", "provider": None, "message": None, "hint": None},
            "ncbi_api": {"status": "skip", "has_key": False, "message": None},
            "workspace": {"status": "fail", "path": None, "message": None},
        },
    }

    def log(msg: str, style: str = None):
        """Print message only if not in JSON mode."""
        if not output_json:
            if style:
                console.print(f"[{style}]{msg}[/{style}]")
            else:
                console.print(msg)

    # Check .env file exists
    env_file = Path.cwd() / ".env"
    test_results["env_file"] = str(env_file) if env_file.exists() else None

    if not env_file.exists():
        if output_json:
            test_results["checks"]["llm_provider"]["message"] = "No .env file found"
            print(json_module.dumps(test_results, indent=2))
        else:
            console.print(f"[red]❌ No .env file found in current directory[/red]")
            console.print(f"[dim]Run 'lobster init' to create configuration[/dim]")
        raise typer.Exit(1)

    load_dotenv()

    if not output_json:
        console.print()
        console.print(
            Panel.fit(
                f"[bold {LobsterTheme.PRIMARY_ORANGE}]🔍 Configuration Test[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                border_style=LobsterTheme.PRIMARY_ORANGE,
                padding=(0, 2),
            )
        )
        console.print()
        console.print(f"[green]✅ Found .env file:[/green] {env_file}")
        console.print()

    # Test LLM Provider
    log("Testing LLM Provider...", "bold")
    try:
        from lobster.config.llm_factory import LLMFactory, LLMProvider

        workspace_path = Path.cwd() / ".lobster_workspace"
        provider = LLMFactory.get_current_provider(workspace_path=workspace_path)
        if provider is None:
            test_results["checks"]["llm_provider"]["message"] = "No API keys found"
            test_results["checks"]["llm_provider"]["hint"] = "Set ANTHROPIC_API_KEY, AWS_BEDROCK_ACCESS_KEY, or run Ollama"
            log("❌ No LLM provider configured", "red")
        else:
            # provider is a string, not an enum
            test_results["checks"]["llm_provider"]["provider"] = provider
            log(f"  Detected provider: {provider}", "yellow")

            # Provider-specific pre-checks for Ollama
            if provider == "ollama":
                ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                log(f"  Checking Ollama server at {ollama_url}...", "yellow")
                try:
                    import requests
                    resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
                    if resp.status_code != 200:
                        raise Exception(f"Ollama server returned status {resp.status_code}")
                    models = resp.json().get("models", [])
                    if not models:
                        test_results["checks"]["llm_provider"]["message"] = "Ollama running but no models installed"
                        test_results["checks"]["llm_provider"]["hint"] = "Run: ollama pull llama3.2:3b"
                        log("❌ Ollama: No models installed", "red")
                        provider = None
                    else:
                        model_names = [m.get("name", "unknown") for m in models[:3]]
                        log(f"  Ollama server: Running ({len(models)} models)", "green")
                except requests.exceptions.ConnectionError:
                    test_results["checks"]["llm_provider"]["message"] = "Ollama server not running"
                    test_results["checks"]["llm_provider"]["hint"] = "Start Ollama: ollama serve"
                    log("❌ Ollama server not accessible", "red")
                    provider = None
                except Exception as e:
                    test_results["checks"]["llm_provider"]["message"] = f"Ollama error: {str(e)[:60]}"
                    log(f"❌ Ollama check failed: {e}", "red")
                    provider = None

            # Test LLM connectivity
            if provider is not None:
                try:
                    # provider is a string
                    if provider == "ollama":
                        test_config = {"model_id": "", "temperature": 1.0, "max_tokens": 50}
                    elif provider == "anthropic":
                        test_config = {"model_id": "claude-3-5-haiku-20241022", "temperature": 1.0, "max_tokens": 50}
                    elif provider == "gemini":
                        test_config = {"model_id": "gemini-2.0-flash-exp", "temperature": 1.0, "max_tokens": 50}
                    else:  # bedrock
                        test_config = {"model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0", "temperature": 1.0, "max_tokens": 50}

                    test_llm = LLMFactory.create_llm(test_config, "config_test")
                    log("  Testing API connectivity...", "yellow")
                    response = test_llm.invoke("Reply with just 'ok'")

                    test_results["checks"]["llm_provider"]["status"] = "pass"
                    test_results["checks"]["llm_provider"]["message"] = "Connected"
                    log(f"✅ {provider} API: Connected", "green")
                except Exception as e:
                    error_msg = str(e)
                    if "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                        hint = "Check your API key"
                    elif "rate_limit" in error_msg.lower():
                        hint = "Rate limited - wait and retry"
                    elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                        hint = "Model not available in your region/plan"
                    else:
                        hint = None

                    test_results["checks"]["llm_provider"]["message"] = error_msg[:100]
                    test_results["checks"]["llm_provider"]["hint"] = hint
                    log(f"❌ {provider} API: {error_msg[:60]}", "red")
    except Exception as e:
        test_results["checks"]["llm_provider"]["message"] = str(e)[:100]
        log(f"❌ LLM Provider test failed: {e}", "red")

    if not output_json:
        console.print()

    # Test NCBI API
    log("Testing NCBI API...", "bold")
    ncbi_key = os.environ.get("NCBI_API_KEY")
    test_results["checks"]["ncbi_api"]["has_key"] = bool(ncbi_key)

    if ncbi_key or os.environ.get("NCBI_EMAIL"):
        try:
            import urllib.request
            import xml.etree.ElementTree as ET

            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {"db": "pubmed", "term": "test", "retmax": "1", "retmode": "xml"}
            if ncbi_key:
                params["api_key"] = ncbi_key

            url = f"{base_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
            log("  Testing NCBI E-utilities...", "yellow")

            with urllib.request.urlopen(url, timeout=10) as response:
                root = ET.fromstring(response.read())
                error = root.find(".//ERROR")
                if error is not None:
                    test_results["checks"]["ncbi_api"]["status"] = "fail"
                    test_results["checks"]["ncbi_api"]["message"] = error.text
                    log(f"❌ NCBI API Error: {error.text}", "red")
                else:
                    test_results["checks"]["ncbi_api"]["status"] = "pass"
                    test_results["checks"]["ncbi_api"]["message"] = "Connected"
                    log(f"✅ NCBI API: Connected {'(with API key)' if ncbi_key else ''}", "green")
        except Exception as e:
            test_results["checks"]["ncbi_api"]["status"] = "fail"
            test_results["checks"]["ncbi_api"]["message"] = str(e)[:60]
            log(f"❌ NCBI API: {str(e)[:60]}", "red")
    else:
        test_results["checks"]["ncbi_api"]["status"] = "skip"
        test_results["checks"]["ncbi_api"]["message"] = "Not configured (optional)"
        log("⊘ NCBI API: Not configured (optional)", "dim")

    if not output_json:
        console.print()

    # Test Workspace
    log("Testing Workspace...", "bold")
    try:
        from lobster.core.workspace import resolve_workspace
        workspace_path = resolve_workspace(explicit_path=None, create=True)
        test_results["checks"]["workspace"]["path"] = str(workspace_path)

        test_file = workspace_path / ".config_test_write"
        try:
            test_file.write_text("test")
            test_file.unlink()
            test_results["checks"]["workspace"]["status"] = "pass"
            test_results["checks"]["workspace"]["message"] = "Writable"
            log("✅ Workspace: Writable", "green")
            log(f"  Path: {workspace_path}", "dim")
        except PermissionError:
            test_results["checks"]["workspace"]["message"] = "Permission denied"
            log("❌ Workspace: Permission denied", "red")
        except Exception as e:
            test_results["checks"]["workspace"]["message"] = str(e)[:60]
            log(f"❌ Workspace: {e}", "red")
    except Exception as e:
        test_results["checks"]["workspace"]["message"] = str(e)[:60]
        log(f"❌ Workspace setup failed: {e}", "red")

    # Determine overall validity
    test_results["valid"] = test_results["checks"]["llm_provider"]["status"] == "pass"

    # Output
    if output_json:
        print(json_module.dumps(test_results, indent=2))
    else:
        console.print()

        # Summary table
        table = Table(title="Configuration Test Summary", box=box.ROUNDED)
        table.add_column("Service", style="cyan", no_wrap=True)
        table.add_column("Status", style="bold", no_wrap=True)
        table.add_column("Details", style="dim")

        for check_name, check_data in test_results["checks"].items():
            status = check_data["status"]
            icon = "✅" if status == "pass" else ("❌" if status == "fail" else "⊘")
            style = "green" if status == "pass" else ("red" if status == "fail" else "dim")
            details = check_data.get("message") or ""
            if check_name == "llm_provider" and check_data.get("provider"):
                details = f"{check_data['provider']}: {details}"
            table.add_row(check_name.replace("_", " ").title(), f"[{style}]{icon}[/{style}]", details)

        console.print(table)
        console.print()

        if test_results["valid"]:
            console.print(
                Panel.fit(
                    "[bold green]✅ Configuration Valid[/bold green]\n\n"
                    "All required services are accessible.\n"
                    f"You can now run: [bold {LobsterTheme.PRIMARY_ORANGE}]lobster chat[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print(
                Panel.fit(
                    "[bold red]❌ Configuration Issues Detected[/bold red]\n\n"
                    "Please check your API keys in the .env file.\n"
                    f"Run: [bold {LobsterTheme.PRIMARY_ORANGE}]lobster init --force[/bold {LobsterTheme.PRIMARY_ORANGE}] to reconfigure",
                    border_style="red",
                    padding=(1, 2),
                )
            )

    if not test_results["valid"]:
        raise typer.Exit(1)


@app.command(name="config")
def config_command(
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to use",
    ),
):
    """Display current LLM configuration (provider, models, config files).

    This command shows:
    - Current LLM provider and profile
    - Configuration file locations
    - Per-agent model assignments
    - Available config commands
    """
    from lobster.core.workspace import resolve_workspace
    from lobster.core.client import AgentClient

    # Resolve workspace path
    workspace_path = resolve_workspace(workspace, create=False)

    # Create minimal client for config access
    try:
        client = AgentClient(workspace_path=str(workspace_path))
        output = ConsoleOutputAdapter(console)

        # Use unified config_show implementation from config_commands.py
        config_show(client, output)
    except Exception as e:
        console.print(f"[red]❌ Error displaying configuration: {str(e)}[/red]")
        console.print(f"[dim]Run 'lobster init' if not yet configured[/dim]")
        raise typer.Exit(1)


@app.command()
def status():
    """Display subscription tier, installed packages, and available agents."""
    _display_status_info()


@app.command()
def activate(
    access_code: str = typer.Argument(
        ..., help="Premium activation code from Omics-OS"
    ),
    server_url: Optional[str] = typer.Option(
        None,
        "--server",
        help="License server URL (defaults to https://licenses.omics-os.com)",
    ),
):
    """
    Activate a premium license using an access code.

    This command contacts the Omics-OS license server to validate your
    access code and activate premium features on this machine.

    Examples:
      lobster activate ABC123-XYZ789
      lobster activate ABC123-XYZ789 --server https://custom.server.com
    """
    console.print()
    console.print(
        Panel.fit(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]🦞 License Activation[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            padding=(0, 2),
        )
    )
    console.print()

    # Check if already activated
    try:
        from lobster.core.license_manager import get_entitlement_status

        current = get_entitlement_status()
        if (
            current.get("tier") not in ("free", None)
            and current.get("source") == "license_file"
        ):
            console.print(
                f"[yellow]⚠️  You already have an active {current.get('tier_display', 'Premium')} license[/yellow]"
            )
            console.print(f"[dim]Source: {current.get('source')}[/dim]")
            console.print()
            if not Confirm.ask("Replace existing license?", default=False):
                console.print("[yellow]Activation cancelled[/yellow]")
                raise typer.Exit(0)
    except ImportError:
        pass  # Continue with activation

    console.print("[dim]Contacting license server...[/dim]")

    try:
        from lobster.core.license_manager import activate_license

        result = activate_license(access_code, license_server_url=server_url)

        if result.get("success"):
            entitlement = result.get("entitlement", {})
            tier = entitlement.get("tier", "premium").title()
            packages_installed = result.get("packages_installed", [])
            packages_failed = result.get("packages_failed", [])

            # Build the success message
            msg_lines = [
                f"[bold green]✅ License Activated Successfully![/bold green]\n",
                f"Tier: [bold]{tier}[/bold]",
                f"Features: {', '.join(entitlement.get('features', []))}",
            ]

            # Show installed packages
            if packages_installed:
                msg_lines.append("")
                msg_lines.append(
                    f"[bold green]Custom Packages Installed ({len(packages_installed)}):[/bold green]"
                )
                for pkg in packages_installed:
                    msg_lines.append(
                        f"  [green]✓[/green] {pkg['name']} v{pkg['version']}"
                    )

            # Show failed packages with warnings
            if packages_failed:
                msg_lines.append("")
                msg_lines.append(
                    f"[bold yellow]⚠️  Package Installation Issues ({len(packages_failed)}):[/bold yellow]"
                )
                for pkg in packages_failed:
                    error = pkg.get("error", "Unknown error")
                    msg_lines.append(
                        f"  [yellow]✗[/yellow] {pkg['name']}: {error[:50]}..."
                    )
                msg_lines.append("")
                msg_lines.append(
                    "[dim]You can retry later with: pip install <package_name>[/dim]"
                )

            msg_lines.append("")
            msg_lines.append(
                f"Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster status[/bold {LobsterTheme.PRIMARY_ORANGE}] to see available agents."
            )

            console.print()
            console.print(
                Panel.fit(
                    "\n".join(msg_lines),
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            error = result.get("error", "Unknown error")
            console.print()
            console.print(
                Panel.fit(
                    f"[bold red]❌ Activation Failed[/bold red]\n\n"
                    f"Error: {error}\n\n"
                    f"[dim]If this problem persists, contact support@omics-os.com[/dim]",
                    border_style="red",
                    padding=(1, 2),
                )
            )
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[red]❌ License manager not available: {e}[/red]")
        console.print("[dim]This may indicate an incomplete installation.[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Activation error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def deactivate():
    """
    Deactivate the current premium license.

    This removes the local license file and reverts to the free tier.
    Your license can be re-activated on another machine or re-used later.
    """
    console.print()
    console.print(
        Panel.fit(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]🦞 License Deactivation[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            padding=(0, 2),
        )
    )
    console.print()

    # Check current status
    try:
        from lobster.core.license_manager import (
            clear_entitlement,
            get_entitlement_status,
        )

        current = get_entitlement_status()

        if current.get("source") == "cloud_key":
            console.print(
                "[yellow]⚠️  Your premium tier is from LOBSTER_CLOUD_KEY environment variable.[/yellow]"
            )
            console.print(
                "[dim]To deactivate, unset the LOBSTER_CLOUD_KEY environment variable.[/dim]"
            )
            raise typer.Exit(0)

        if current.get("source") == "environment":
            console.print(
                "[yellow]⚠️  Your tier is set via LOBSTER_SUBSCRIPTION_TIER environment variable.[/yellow]"
            )
            console.print(
                "[dim]To deactivate, unset the LOBSTER_SUBSCRIPTION_TIER environment variable.[/dim]"
            )
            raise typer.Exit(0)

        if current.get("tier") == "free" or current.get("source") == "default":
            console.print("[dim]No active license found. Already on free tier.[/dim]")
            raise typer.Exit(0)

        # Confirm deactivation
        tier_display = current.get("tier_display", "Premium")
        console.print(f"[bold]Current tier:[/bold] {tier_display}")
        console.print(f"[dim]Source: {current.get('source')}[/dim]")
        console.print()

        if not Confirm.ask(
            f"[yellow]Deactivate {tier_display} license and revert to Free tier?[/yellow]",
            default=False,
        ):
            console.print("[yellow]Deactivation cancelled[/yellow]")
            raise typer.Exit(0)

        # Remove license file
        if clear_entitlement():
            console.print()
            console.print(
                Panel.fit(
                    "[bold green]✅ License Deactivated[/bold green]\n\n"
                    "You are now on the Free tier.\n"
                    f"Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster activate <code>[/bold {LobsterTheme.PRIMARY_ORANGE}] to re-activate.",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print("[red]❌ Failed to remove license file[/red]")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[red]❌ License manager not available: {e}[/red]")
        raise typer.Exit(1)


def _create_workspace_config(config_dict: Dict[str, Any], workspace_path: Path) -> None:
    """
    Create workspace-scoped provider configuration from structured config dict.

    This function creates the provider_config.json file in the workspace directory,
    enabling the ConfigResolver to properly detect provider settings via layer 2
    (workspace config) instead of relying solely on .env (layer 4).

    Args:
        config_dict: Structured configuration dictionary with keys:
            - provider: str - LLM provider ("anthropic", "bedrock", "ollama")
            - profile: Optional[str] - Agent profile (for Anthropic/Bedrock)
            - ollama_model: Optional[str] - Ollama model name
            - ollama_host: Optional[str] - Ollama server URL
        workspace_path: Path to workspace directory

    Raises:
        IOError: If workspace config creation fails (non-fatal - logged as warning)
    """
    from lobster.config.workspace_config import WorkspaceProviderConfig

    try:
        # Create workspace config from structured dict
        workspace_config = WorkspaceProviderConfig()

        # Set provider (required)
        if "provider" in config_dict:
            workspace_config.global_provider = config_dict["provider"]

        # Set profile (for Anthropic/Bedrock)
        if "profile" in config_dict:
            workspace_config.profile = config_dict["profile"]

        # Set Ollama-specific settings
        if "ollama_model" in config_dict:
            workspace_config.ollama_model = config_dict["ollama_model"]

        if "ollama_host" in config_dict:
            workspace_config.ollama_host = config_dict["ollama_host"]

        # Ensure workspace directory exists and save config
        workspace_path.mkdir(parents=True, exist_ok=True)
        workspace_config.save(workspace_path)

        logger.info(f"Created workspace config at {workspace_path / 'provider_config.json'}")

    except Exception as e:
        # Non-fatal error - .env is still valid, workspace config is optimization
        logger.warning(f"Failed to create workspace config: {e}")
        console.print(f"[dim yellow]⚠️  Note: Workspace config creation skipped ({e})[/dim yellow]")


@app.command()
def init(
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing .env file"
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Non-interactive mode for CI/CD (requires API key flags)",
    ),
    anthropic_key: Optional[str] = typer.Option(
        None, "--anthropic-key", help="Claude API key (non-interactive mode)"
    ),
    bedrock_access_key: Optional[str] = typer.Option(
        None,
        "--bedrock-access-key",
        help="AWS Bedrock access key (non-interactive mode)",
    ),
    bedrock_secret_key: Optional[str] = typer.Option(
        None,
        "--bedrock-secret-key",
        help="AWS Bedrock secret key (non-interactive mode)",
    ),
    use_ollama: bool = typer.Option(
        False,
        "--use-ollama",
        help="Use Ollama (local LLM) provider (non-interactive mode)",
    ),
    ollama_model: Optional[str] = typer.Option(
        None,
        "--ollama-model",
        help="Ollama model name (default: llama3:8b-instruct, non-interactive mode)",
    ),
    gemini_key: Optional[str] = typer.Option(
        None, "--gemini-key", help="Google API key (non-interactive mode)"
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Agent profile (development, production, ultra, godmode). Only for Anthropic/Bedrock providers.",
    ),
    ncbi_key: Optional[str] = typer.Option(
        None, "--ncbi-key", help="NCBI API key (optional, non-interactive mode)"
    ),
    cloud_key: Optional[str] = typer.Option(
        None,
        "--cloud-key",
        help="Lobster Cloud API key (optional, enables premium tier)",
    ),
    cloud_endpoint: Optional[str] = typer.Option(
        None,
        "--cloud-endpoint",
        help="Custom cloud endpoint URL (optional)",
    ),
    skip_ssl_test: bool = typer.Option(
        False,
        "--skip-ssl-test",
        help="Skip SSL connectivity test during init",
    ),
    ssl_verify: Optional[bool] = typer.Option(
        None,
        "--ssl-verify/--no-ssl-verify",
        help="Enable/disable SSL verification (non-interactive mode)",
    ),
    ssl_cert_path: Optional[str] = typer.Option(
        None,
        "--ssl-cert-path",
        help="Path to custom CA certificate bundle",
    ),
):
    """
    Initialize Lobster AI configuration by creating a .env file with API keys.

    Interactive mode (default):
      Guides you through provider selection and API key entry with masked input.

    Non-interactive mode (CI/CD):
      Provide API keys via command-line flags for automated deployment.

    Examples:
      lobster init                                    # Interactive setup
      lobster init --force                            # Reconfigure (overwrite existing)
      lobster init --non-interactive \\
        --anthropic-key=sk-ant-xxx                   # CI/CD: Claude API (default: production)
      lobster init --non-interactive \\
        --anthropic-key=sk-ant-xxx \\
        --profile=development                        # CI/CD: Claude with dev profile
      lobster init --non-interactive \\
        --bedrock-access-key=AKIA... \\
        --bedrock-secret-key=xxx \\
        --profile=ultra                              # CI/CD: Bedrock with ultra profile
      lobster init --non-interactive \\
        --use-ollama                                 # CI/CD: Ollama (profile not applicable)
      lobster init --non-interactive \\
        --use-ollama --ollama-model=mixtral:8x7b-instruct  # CI/CD: Ollama with custom model
      lobster init --non-interactive \\
        --anthropic-key=sk-ant-xxx \\
        --cloud-key=cloud_xxx                        # CI/CD: With cloud access
    """
    import datetime

    env_path = Path.cwd() / ".env"

    # Check if .env already exists
    if env_path.exists() and not force:
        console.print()
        console.print(
            Panel.fit(
                "[bold yellow]⚠️  Configuration Already Exists[/bold yellow]\n\n"
                f"A .env file already exists at:\n[cyan]{env_path}[/cyan]\n\n"
                "To reconfigure, use:\n"
                f"[bold {LobsterTheme.PRIMARY_ORANGE}]lobster init --force[/bold {LobsterTheme.PRIMARY_ORANGE}]\n\n"
                "Or edit the file manually.",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        console.print()
        console.print(f"[dim]Configuration file: {env_path}[/dim]")
        raise typer.Exit(0)

    # If force flag and file exists, create backup and confirm
    backup_path = None
    if env_path.exists() and force:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path.cwd() / f".env.backup.{timestamp}"

        if not non_interactive:
            console.print(f"[yellow]⚠️  Existing .env will be backed up to:[/yellow]")
            console.print(f"[yellow]   {backup_path}[/yellow]")
            console.print()
            if not Confirm.ask("Continue with reconfiguration?", default=False):
                console.print("[yellow]Configuration cancelled[/yellow]")
                raise typer.Exit(0)

        # Create backup
        try:
            shutil.copy2(env_path, backup_path)
        except Exception as e:
            console.print(f"[red]❌ Failed to create backup: {str(e)}[/red]")
            raise typer.Exit(1)

    # Non-interactive mode: validate and create .env from parameters
    if non_interactive:
        env_lines = []
        env_lines.append("# Lobster AI Configuration")
        env_lines.append("# Generated by lobster init --non-interactive\n")

        # Initialize structured config dict for workspace config
        config_dict = {}

        # Validate provider configuration using provider_setup module
        has_anthropic = anthropic_key is not None
        has_bedrock = bedrock_access_key is not None and bedrock_secret_key is not None
        has_ollama = use_ollama
        has_gemini = gemini_key is not None

        # Validate at least one provider
        valid, error_msg = provider_setup.validate_provider_choice(
            has_anthropic, has_bedrock, has_ollama, has_gemini
        )
        if not valid:
            console.print(f"[red]❌ Error: {error_msg}[/red]")
            console.print()
            console.print("You must provide one of:")
            console.print("  • Claude API: --anthropic-key=xxx")
            console.print(
                "  • AWS Bedrock: --bedrock-access-key=xxx --bedrock-secret-key=xxx"
            )
            console.print("  • Ollama (Local): --use-ollama")
            console.print("  • Google Gemini: --gemini-key=xxx")
            raise typer.Exit(1)

        # Warn if multiple providers
        priority_warning = provider_setup.get_provider_priority_warning(
            has_anthropic, has_bedrock, has_ollama, has_gemini
        )
        if priority_warning:
            console.print(f"[yellow]⚠️  Warning: {priority_warning}[/yellow]")

        # Validate profile parameter (only relevant for Anthropic/Bedrock)
        valid_profiles = ["development", "production", "ultra", "godmode"]
        selected_profile = None

        if profile:
            # Profile provided - validate it
            if profile not in valid_profiles:
                console.print(f"[red]❌ Error: Invalid profile '{profile}'[/red]")
                console.print(f"Valid profiles: {', '.join(valid_profiles)}")
                raise typer.Exit(1)

            # Warn if profile used with Ollama (it will be ignored)
            if has_ollama and not has_anthropic and not has_bedrock:
                console.print(
                    f"[yellow]⚠️  Warning: --profile ignored for Ollama (uses local models)[/yellow]"
                )
            else:
                selected_profile = profile
        else:
            # No profile provided - use default for Anthropic/Bedrock only
            if has_anthropic or has_bedrock:
                selected_profile = "production"  # Default profile

        # Create provider configuration
        if has_anthropic:
            config = provider_setup.create_anthropic_config(anthropic_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "anthropic"
        elif has_bedrock:
            config = provider_setup.create_bedrock_config(
                bedrock_access_key, bedrock_secret_key
            )
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "bedrock"
        elif has_ollama:
            config = provider_setup.create_ollama_config(model_name=ollama_model)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "ollama"
                if ollama_model:
                    config_dict["ollama_model"] = ollama_model
                console.print("[green]✓ Ollama provider configured[/green]")
        elif has_gemini:
            config = provider_setup.create_gemini_config(gemini_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "gemini"
                console.print("[green]✓ Gemini provider configured[/green]")

        # Write profile configuration (only for Anthropic/Bedrock)
        if selected_profile:
            env_lines.append(f"\n# Agent Configuration Profile")
            env_lines.append(f"# Determines which Claude models are used for analysis")
            env_lines.append(f"LOBSTER_PROFILE={selected_profile}")
            config_dict["profile"] = selected_profile
            console.print(f"[green]✓ Profile set to: {selected_profile}[/green]")

        if ncbi_key:
            env_lines.append(f"\n# Optional: Enhanced literature search")
            env_lines.append(f"NCBI_API_KEY={ncbi_key.strip()}")

        if cloud_key:
            env_lines.append(f"\n# Lobster Cloud configuration (enables premium tier)")
            env_lines.append(f"LOBSTER_CLOUD_KEY={cloud_key.strip()}")
            if cloud_endpoint:
                env_lines.append(f"LOBSTER_ENDPOINT={cloud_endpoint.strip()}")
            console.print(
                "[green]✓ Cloud API key configured (premium tier enabled)[/green]"
            )

        # Handle SSL configuration from flags
        if ssl_verify is not None or ssl_cert_path:
            env_lines.append(f"\n# SSL/HTTPS configuration")
            if ssl_verify is False:
                env_lines.append(f"LOBSTER_SSL_VERIFY=false")
                console.print(
                    "[yellow]⚠️  SSL verification disabled (--no-ssl-verify)[/yellow]"
                )
            if ssl_cert_path:
                env_lines.append(f"LOBSTER_SSL_CERT_PATH={ssl_cert_path}")
                console.print(f"[green]✓ SSL cert path: {ssl_cert_path}[/green]")

        # Test SSL connectivity (non-interactive)
        if not skip_ssl_test and ssl_verify is not False:
            from lobster.config.ssl_setup import test_ssl_connectivity

            console.print("\n[dim]Testing connectivity to NCBI databases...[/dim]")
            success, is_ssl_error, error_msg = test_ssl_connectivity(timeout=10)

            if is_ssl_error:
                console.print(
                    f"\n[yellow]⚠️  SSL Certificate Issue Detected[/yellow]"
                )
                console.print(f"[dim]Error: {error_msg[:80]}...[/dim]\n")
                console.print(
                    "[yellow]To fix this in non-interactive mode, use one of:[/yellow]"
                )
                console.print("  --no-ssl-verify           # Disable SSL (testing only)")
                console.print("  --ssl-cert-path=/path/to/ca.pem  # Use corporate CA")
                console.print()
                console.print(
                    "[dim]Continuing with setup - you may encounter errors later.[/dim]"
                )
            elif success:
                console.print("[green]✓ NCBI connectivity: OK[/green]")
            else:
                # Non-SSL network error
                console.print(f"[yellow]⚠️  Network test failed: {error_msg[:60]}[/yellow]")
                console.print("[dim]You can still continue (may work later)[/dim]")

        # Write .env file
        try:
            with open(env_path, "w") as f:
                f.write("\n".join(env_lines))
                f.write("\n")
        except Exception as e:
            console.print(f"[red]❌ Failed to write .env file: {str(e)}[/red]")
            raise typer.Exit(1)

        # Create workspace config from structured config_dict
        workspace_path = Path.cwd() / ".lobster_workspace"
        _create_workspace_config(config_dict, workspace_path)

        # Success message for non-interactive
        console.print(f"[green]✅ Configuration saved:[/green]")
        console.print(f"  • Environment: {env_path}")
        console.print(f"  • Workspace:   {workspace_path / 'provider_config.json'}")
        if backup_path:
            console.print(f"  • Backup:      {backup_path}")
        raise typer.Exit(0)

    # Interactive mode: run wizard
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold white]🦞 Welcome to Lobster AI![/bold white]\n\n"
            "Let's set up your API keys.\n"
            "This wizard will create a [cyan].env[/cyan] file in your current directory.",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )
    console.print()

    try:
        # Provider selection
        console.print("[bold white]Select your LLM provider:[/bold white]")
        console.print(
            "  [cyan]1[/cyan] - Claude API (Anthropic) - Quick testing, development"
        )
        console.print("  [cyan]2[/cyan] - AWS Bedrock - Production, enterprise use")
        console.print("  [cyan]3[/cyan] - Ollama (Local) - Privacy, zero cost, offline")
        console.print("  [cyan]4[/cyan] - Google Gemini - Latest models with thinking support")
        console.print()

        provider = Prompt.ask(
            "[bold white]Choose provider[/bold white]", choices=["1", "2", "3", "4"], default="1"
        )

        env_lines = []
        env_lines.append("# Lobster AI Configuration")
        env_lines.append("# Generated by lobster init\n")

        # Initialize structured config dict for workspace config
        config_dict = {}

        if provider == "1":
            # Claude API setup
            console.print("\n[bold white]🔑 Claude API Configuration[/bold white]")
            console.print(
                "Get your API key from: [link]https://console.anthropic.com/[/link]\n"
            )

            api_key = Prompt.ask(
                "[bold white]Enter your Claude API key[/bold white]", password=True
            )

            if not api_key.strip():
                console.print("[red]❌ API key cannot be empty[/red]")
                raise typer.Exit(1)

            env_lines.append(f"ANTHROPIC_API_KEY={api_key.strip()}")
            config_dict["provider"] = "anthropic"

        elif provider == "2":
            # AWS Bedrock setup
            console.print("\n[bold white]🔑 AWS Bedrock Configuration[/bold white]")
            console.print(
                "You'll need AWS access key and secret key with Bedrock permissions.\n"
            )

            access_key = Prompt.ask(
                "[bold white]Enter your AWS access key[/bold white]", password=True
            )
            secret_key = Prompt.ask(
                "[bold white]Enter your AWS secret key[/bold white]", password=True
            )

            if not access_key.strip() or not secret_key.strip():
                console.print("[red]❌ AWS credentials cannot be empty[/red]")
                raise typer.Exit(1)

            env_lines.append(f"AWS_BEDROCK_ACCESS_KEY={access_key.strip()}")
            env_lines.append(f"AWS_BEDROCK_SECRET_ACCESS_KEY={secret_key.strip()}")
            config_dict["provider"] = "bedrock"

        elif provider == "3":  # provider == "3"
            # Ollama (local LLM) setup using provider_setup module
            console.print("\n[bold white]🏠 Ollama (Local LLM) Configuration[/bold white]")
            console.print("Ollama runs models locally - no API keys needed!\n")

            # Get Ollama status
            ollama_status = provider_setup.get_ollama_status()

            if not ollama_status.installed:
                # Ollama not installed - show instructions
                console.print(
                    "[yellow]⚠️  Ollama is not installed on this system.[/yellow]"
                )
                console.print()
                console.print("[bold white]To install Ollama:[/bold white]")
                install_instructions = provider_setup.get_ollama_install_instructions()
                console.print(f"  • macOS/Linux: {install_instructions['macos_linux']}")
                console.print(f"  • Windows: {install_instructions['windows']}")
                console.print()

                install_later = Confirm.ask(
                    "Configure for Ollama anyway? (you can install it later)",
                    default=True,
                )

                if not install_later:
                    console.print(
                        "[yellow]Please install Ollama first, then run 'lobster init' again[/yellow]"
                    )
                    raise typer.Exit(0)

                # User wants to configure anyway
                config = provider_setup.create_ollama_config()
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                env_lines.append(
                    "# Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
                )
                env_lines.append("# Then run: ollama pull llama3:8b-instruct")
                config_dict["provider"] = "ollama"
                console.print(
                    "[green]✓ Ollama provider configured (install Ollama to use)[/green]"
                )

            else:
                # Ollama is installed
                console.print("[green]✓ Ollama is installed[/green]")
                if ollama_status.version:
                    console.print(f"[dim]  Version: {ollama_status.version}[/dim]")

                if not ollama_status.running:
                    console.print("[yellow]⚠️  Ollama server is not running[/yellow]")
                    console.print("Start it with: ollama serve")
                    console.print()

                # Show available models if any
                if ollama_status.running and ollama_status.models:
                    console.print("\n[bold white]Available models:[/bold white]")
                    for model in ollama_status.models[:5]:  # Show first 5
                        console.print(f"  • {model}")
                    if len(ollama_status.models) > 5:
                        console.print(
                            f"  ... and {len(ollama_status.models) - 5} more"
                        )
                    console.print()

                    # Ask which model to use
                    use_custom_model = Confirm.ask(
                        "Specify a model? (default: llama3:8b-instruct)", default=False
                    )

                    if use_custom_model:
                        model_name = Prompt.ask(
                            "[bold white]Enter model name[/bold white]",
                            default="llama3:8b-instruct",
                        )
                        config = provider_setup.create_ollama_config(
                            model_name=model_name
                        )
                        config_dict["ollama_model"] = model_name
                    else:
                        config = provider_setup.create_ollama_config()

                    for key, value in config.env_vars.items():
                        env_lines.append(f"{key}={value}")

                    config_dict["provider"] = "ollama"

                else:
                    # No models available - show recommendations
                    console.print("\n[yellow]No models found. Pull a model first:[/yellow]")
                    recommended = provider_setup.get_recommended_models()
                    for model_info in recommended:
                        console.print(
                            f"  ollama pull {model_info['name']}  # {model_info['description']}"
                        )
                    console.print()

                    proceed = Confirm.ask("Configure for Ollama anyway?", default=True)

                    if not proceed:
                        console.print(
                            "[yellow]Setup cancelled. Pull a model first, then run 'lobster init'[/yellow]"
                        )
                        raise typer.Exit(0)

                    # Configure anyway
                    config = provider_setup.create_ollama_config()
                    for key, value in config.env_vars.items():
                        env_lines.append(f"{key}={value}")

                    config_dict["provider"] = "ollama"

                console.print("[green]✓ Ollama provider configured[/green]")

        elif provider == "4":
            # Google Gemini setup
            console.print("\n[bold white]🔑 Google Gemini Configuration[/bold white]")
            console.print(
                "Get your API key from: [link]https://aistudio.google.com/apikey[/link]\n"
            )

            api_key = Prompt.ask(
                "[bold white]Enter your Google API key[/bold white]", password=True
            )

            if not api_key.strip():
                console.print("[red]❌ API key cannot be empty[/red]")
                raise typer.Exit(1)

            config = provider_setup.create_gemini_config(api_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                config_dict["provider"] = "gemini"
                console.print("[green]✓ Gemini provider configured[/green]")

        # Profile selection (only for Anthropic/Bedrock providers)
        profile_to_write = None
        if provider in ["1", "2"]:  # Anthropic or Bedrock
            console.print("\n[bold white]⚙️  Agent Configuration Profile[/bold white]")
            console.print("Choose which Claude models to use for analysis:")
            console.print()
            console.print("  [cyan]1[/cyan] - Development  (Haiku 4.5 - fastest, most affordable)")
            console.print("  [cyan]2[/cyan] - Production   (Sonnet 4 - balanced quality & speed) [recommended]")
            console.print("  [cyan]3[/cyan] - Ultra        (Sonnet 4.5 - highest quality, slower)")
            console.print("  [cyan]4[/cyan] - Godmode      (Opus 4.1 - experimental, most expensive)")
            console.print()
            console.print("[dim]💡 Tip: Development profile is great for testing, Production for real analysis[/dim]")
            console.print("[dim]   You can change this later by editing .env or using the --profile flag[/dim]")
            console.print()

            profile_choice = Prompt.ask(
                "[bold white]Choose profile[/bold white]",
                choices=["1", "2", "3", "4"],
                default="2"
            )

            profile_map = {
                "1": "development",
                "2": "production",
                "3": "ultra",
                "4": "godmode"
            }
            profile_to_write = profile_map[profile_choice]

            env_lines.append(f"\n# Agent Configuration Profile")
            env_lines.append(f"# Determines which Claude models are used for analysis")
            env_lines.append(f"LOBSTER_PROFILE={profile_to_write}")
            config_dict["profile"] = profile_to_write
            console.print(f"[green]✓ Profile set to: {profile_to_write}[/green]")

        elif provider in ["3", "4"]:  # Ollama or Gemini
            # No profile needed - Ollama uses local models, Gemini uses its own models
            console.print("[dim]ℹ️  Note: Profile configuration not applicable for this provider[/dim]")

        # Optional NCBI key(s) - supports multiple for parallelization
        console.print("\n[bold white]📚 NCBI API Key(s) (Optional)[/bold white]")
        console.print("Enhances literature search capabilities (10 req/sec per key).")
        console.print(
            "Multiple keys enable parallel processing for large publication batches."
        )
        console.print(
            "Get key from: [link]https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/[/link]\n"
        )

        add_ncbi = Confirm.ask("Add NCBI API key?", default=False)

        if add_ncbi:
            ncbi_keys = []
            key_count = 0

            # First key (primary - NCBI_API_KEY)
            ncbi_key = Prompt.ask(
                "[bold white]Enter your NCBI API key[/bold white]", password=True
            )
            if ncbi_key.strip():
                ncbi_keys.append(("NCBI_API_KEY", ncbi_key.strip()))
                key_count += 1

                # Ask for additional keys
                while True:
                    add_another = Confirm.ask(
                        f"Add another NCBI API key? ({key_count} added)",
                        default=False,
                    )
                    if not add_another:
                        break

                    additional_key = Prompt.ask(
                        f"[bold white]Enter NCBI API key #{key_count + 1}[/bold white]",
                        password=True,
                    )
                    if additional_key.strip():
                        # Additional keys use NCBI_API_KEY_1, NCBI_API_KEY_2, etc.
                        ncbi_keys.append(
                            (f"NCBI_API_KEY_{key_count}", additional_key.strip())
                        )
                        key_count += 1
                    else:
                        console.print("[yellow]Empty key skipped[/yellow]")

            # Write all NCBI keys to env
            if ncbi_keys:
                env_lines.append(f"\n# NCBI API keys for literature search")
                if len(ncbi_keys) > 1:
                    env_lines.append(
                        f"# Multiple keys enable parallel processing (10 req/sec each)"
                    )
                for key_name, key_value in ncbi_keys:
                    env_lines.append(f"{key_name}={key_value}")
                console.print(
                    f"[green]✓ Added {len(ncbi_keys)} NCBI API key(s)[/green]"
                )

        # Optional Premium/Cloud configuration
        # First check if license is already activated
        existing_license = None
        try:
            from lobster.core.license_manager import load_entitlement
            entitlement = load_entitlement()
            if entitlement.get("tier") != "free" and entitlement.get("source") in ["license_file", "cloud_key"]:
                existing_license = entitlement
        except Exception:
            pass  # No license or error loading

        if existing_license and not force:
            # License already activated - show status and skip prompt (unless --force)
            tier = existing_license.get("tier", "premium").title()
            tier_emoji = {"premium": "⭐", "enterprise": "🏢"}.get(existing_license.get("tier"), "⭐")
            console.print(f"\n[bold white]{tier_emoji} Premium Features[/bold white]")
            console.print(f"[green]✓ License already activated: {tier} tier[/green]")
            console.print(f"[dim]Source: {existing_license.get('source')}[/dim]")

            # Show installed custom packages if any
            try:
                from lobster.core.plugin_loader import get_installed_packages
                packages = get_installed_packages()
                custom_pkgs = [p for p in packages.keys() if "lobster-custom-" in p]
                if custom_pkgs:
                    console.print(f"[dim]Custom packages: {', '.join(custom_pkgs)}[/dim]")
            except Exception:
                pass

            console.print()
            console.print(
                f"[dim]Your {tier} license is preserved. Use [bold]lobster init --force[/bold] to reconfigure.[/dim]"
            )
            console.print()
            # Skip premium configuration since already activated
            premium_choice = "1"  # Set to skip
        else:
            # No license OR force flag used - show options
            if existing_license and force:
                console.print("\n[yellow]⚠️  Existing license detected but --force flag used[/yellow]")
                tier = existing_license.get("tier", "premium").title()
                console.print(f"[dim]Current: {tier} tier from {existing_license.get('source')}[/dim]")
                console.print()

            console.print("[bold white]⭐ Premium Features (Optional)[/bold white]")
            console.print("Unlock advanced agents and cloud processing capabilities.")
            console.print("Options:")
            console.print("  [cyan]1[/cyan] - Skip (stay on Free tier)" + (" [yellow](will preserve existing license)[/yellow]" if existing_license and force else ""))
            console.print("  [cyan]2[/cyan] - I have an activation code")
            console.print("  [cyan]3[/cyan] - I have a cloud API key")
            console.print()

            premium_choice = Prompt.ask(
                "[bold white]Choose option[/bold white]",
                choices=["1", "2", "3"],
                default="1",
            )

        if premium_choice == "2":
            # Activation code - activate license immediately
            console.print("\n[bold white]🔑 License Activation[/bold white]")
            console.print(
                "Enter your activation code from Omics-OS to unlock premium features.\n"
            )
            activation_code = Prompt.ask(
                "[bold white]Enter activation code[/bold white]", password=False
            )

            if activation_code.strip():
                console.print("[dim]Contacting license server...[/dim]")
                try:
                    from lobster.core.license_manager import activate_license

                    result = activate_license(activation_code.strip())

                    if result.get("success"):
                        entitlement = result.get("entitlement", {})
                        tier = entitlement.get("tier", "premium").title()
                        console.print(
                            f"[green]✓ License activated: {tier} tier[/green]"
                        )
                        # Note: License is stored in ~/.lobster/license.json, not .env
                    else:
                        error = result.get("error", "Unknown error")
                        console.print(f"[yellow]⚠️  Activation failed: {error}[/yellow]")
                        console.print(
                            f"[dim]You can retry later with: lobster activate <code>[/dim]"
                        )
                except ImportError:
                    console.print("[yellow]⚠️  License manager not available[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]⚠️  Activation error: {e}[/yellow]")
                    console.print(
                        f"[dim]You can retry later with: lobster activate <code>[/dim]"
                    )

        elif premium_choice == "3":
            # Cloud API key - store in .env
            console.print("\n[bold white]🌩️  Cloud API Key[/bold white]")
            console.print("Enter your Lobster Cloud API key for cloud processing.\n")
            cloud_key = Prompt.ask(
                "[bold white]Enter your cloud API key[/bold white]", password=True
            )

            if cloud_key.strip():
                env_lines.append(f"\n# Lobster Cloud configuration")
                env_lines.append(f"LOBSTER_CLOUD_KEY={cloud_key.strip()}")
                console.print("[green]✓ Cloud API key configured[/green]")

                # Optional: custom endpoint
                custom_endpoint = Confirm.ask(
                    "Configure custom cloud endpoint?", default=False
                )
                if custom_endpoint:
                    endpoint = Prompt.ask(
                        "[bold white]Enter endpoint URL[/bold white]",
                        default="https://api.lobster.omics-os.com",
                    )
                    env_lines.append(f"LOBSTER_ENDPOINT={endpoint.strip()}")

        # Test SSL connectivity (interactive mode)
        if not skip_ssl_test:
            from lobster.config.ssl_setup import (
                test_ssl_connectivity,
                prompt_ssl_fix,
            )

            console.print()
            console.print("[dim]Testing connectivity to NCBI databases...[/dim]")
            success, is_ssl_error, error_msg = test_ssl_connectivity(timeout=10)

            if is_ssl_error:
                # Prompt user for SSL fix
                ssl_env_line = prompt_ssl_fix(console, error_msg)
                if ssl_env_line:
                    env_lines.append(f"\n# SSL/HTTPS configuration")
                    env_lines.append(ssl_env_line)
            elif success:
                console.print("[green]✓ NCBI connectivity: OK[/green]")
            else:
                # Non-SSL network error
                console.print(
                    f"[yellow]⚠️  Network test failed (non-SSL): {error_msg[:60]}[/yellow]"
                )
                console.print("[dim]You can still continue (may work later)[/dim]")

        # Write .env file
        with open(env_path, "w") as f:
            f.write("\n".join(env_lines))
            f.write("\n")

        # Create workspace config from structured config_dict
        workspace_path = Path.cwd() / ".lobster_workspace"
        _create_workspace_config(config_dict, workspace_path)

        console.print()
        success_message = f"[bold green]✅ Configuration saved![/bold green]\n\n"
        success_message += f"Environment: [cyan]{env_path}[/cyan]\n"
        success_message += f"Workspace:   [cyan]{workspace_path / 'provider_config.json'}[/cyan]\n"
        if backup_path:
            success_message += f"Backup: [cyan]{backup_path}[/cyan]\n\n"
        else:
            success_message += "\n"
        success_message += f"[bold white]Next step:[/bold white] Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster chat[/bold {LobsterTheme.PRIMARY_ORANGE}] to start analyzing!"

        console.print(Panel.fit(success_message, border_style="green"))
        console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Configuration cancelled[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Configuration failed: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def chat(
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory. Can also be set via LOBSTER_WORKSPACE env var. Default: ./.lobster_workspace",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Session ID to continue (use 'latest' for most recent session in workspace)",
    ),
    reasoning: bool = typer.Option(
        False,
        "--reasoning",
        is_flag=True,
        help="Show agent reasoning and thinking process",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed tool usage and agent activity"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug mode with enhanced error reporting"
    ),
    profile_timings: Optional[bool] = typer.Option(
        None,
        "--profile-timings/--no-profile-timings",
        help="Enable timing diagnostics for data manager operations",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider to use (bedrock, anthropic, ollama). Overrides auto-detection.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (e.g., claude-4-sonnet, llama3:70b-instruct, mixtral:8x7b). Overrides configuration.",
    ),
):
    """
    Start an interactive chat session with the multi-agent system.

    Use --session-id to continue a previous conversation:
      lobster chat --session-id latest
      lobster chat --session-id session_20241208_150000

    Use --reasoning to see agent thinking process. Use --verbose for detailed tool output.
    """
    # Enhanced error handling setup
    if debug:
        # Enable more detailed tracebacks in debug mode
        import rich.traceback

        rich.traceback.install(
            console=console_manager.error_console,
            width=None,
            extra_lines=5,
            theme="monokai",
            word_wrap=True,
            show_locals=True,  # Show local variables in debug mode
            suppress=[],
            max_frames=30,
        )

    # Configure logging level based on debug flag
    import logging

    if debug:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.WARNING)  # Suppress INFO logs

    # Check for configuration
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        console.print()
        console.print("[red]✗[/red] no config found")
        console.print(f"  run [bold #e45c47]lobster init[/bold #e45c47]")
        console.print()
        raise typer.Exit(1)

    # Handle session loading for continuity (similar to query command)
    session_file_to_load = None
    session_id_for_client = None

    if session_id:
        # Resolve workspace early to check for session files
        from lobster.core.workspace import resolve_workspace
        workspace_path = resolve_workspace(explicit_path=workspace, create=True)

        if session_id == "latest":
            # Find most recent session file
            session_files = list(workspace_path.glob("session_*.json"))
            if session_files:
                # Sort by modification time (most recent first)
                session_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                session_file_to_load = session_files[0]
                console.print(
                    f"[cyan]📂 Loading session: {session_file_to_load.name}[/cyan]"
                )
            else:
                console.print(
                    "[yellow]⚠️  No previous sessions found - creating new session[/yellow]"
                )
        else:
            # Explicit session ID provided
            # Try with session_ prefix first
            session_file_candidate = workspace_path / f"session_{session_id}.json"
            if not session_file_candidate.exists():
                # Try exact filename
                session_file_candidate = workspace_path / f"{session_id}.json"

            if session_file_candidate.exists():
                session_file_to_load = session_file_candidate
                console.print(
                    f"[cyan]📂 Loading session: {session_file_candidate.name}[/cyan]"
                )
            else:
                # Session file doesn't exist - use this session_id for new session
                session_id_for_client = session_id
                console.print(
                    f"[cyan]📂 Creating new session: {session_id}[/cyan]"
                )

    # Show DNA animation (starts instantly thanks to lazy imports)
    display_welcome()

    # Initialize client (heavy imports happen here)
    try:
        client = init_client_with_animation(
            workspace, reasoning, verbose, debug, profile_timings, provider, model,
            session_id_for_client
        )
    except Exception as e:
        console.print(f"\n[red]✗[/red] init failed: {str(e)[:80]}")
        raise

    # Load session if found
    if session_file_to_load:
        try:
            load_result = client.load_session(session_file_to_load)
            console.print(
                f"[green]✓ Loaded {load_result['messages_loaded']} "
                f"previous messages[/green]"
            )
            if load_result.get("original_session_id"):
                console.print(
                    f"[dim]  Original session: "
                    f"{load_result['original_session_id']}[/dim]"
                )
        except Exception as e:
            console.print(f"[red]❌ Failed to load session: {e}[/red]")
            console.print("[yellow]   Starting fresh session instead[/yellow]")

    # Show compact session status
    _show_workspace_prompt(client)

    while True:
        try:
            # Get current token usage for prompt prefix
            token_prefix = ""
            try:
                token_usage = client.get_token_usage()
                if token_usage and "error" not in token_usage:
                    cost = token_usage.get("total_cost_usd", 0.0)
                    tokens = token_usage.get("total_tokens", 0)
                    # Check if using Ollama (local = free)
                    from lobster.config.llm_factory import LLMFactory
                    current_provider = getattr(client, 'provider_override', None) or LLMFactory.get_current_provider()
                    if current_provider == "ollama":
                        # Ollama is free - show token count only
                        token_prefix = f"[dim grey42]🦙 FREE · {tokens:,}t[/dim grey42] "
                    else:
                        token_prefix = f"[dim grey42]${cost:.4f} · {tokens:,}t[/dim grey42] "
            except Exception:
                pass

            # Prompt with token costs on the left
            user_input = get_user_input_with_editing(
                f"\n{token_prefix}[bold #e45c47]❯[/bold #e45c47] ", client
            )

            # Skip processing if input is empty or just whitespace
            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/"):
                handle_command(user_input, client)
                continue

            # Check if user forgot the slash for a command
            potential_command = check_for_missing_slash_command(user_input)
            if potential_command:
                if Confirm.ask(
                    f"[yellow]Did you mean '{potential_command}'?[/yellow]",
                    default=True,
                ):
                    # Replace first word with the slash command
                    words = user_input.split()
                    words[0] = potential_command
                    corrected_input = " ".join(words)
                    handle_command(corrected_input, client)
                    continue

            # Check if it's a shell command first
            if execute_shell_command(user_input):
                continue

            # Minimal processing indicator
            if should_show_progress(client):
                console.print("[dim]...[/dim]", end="", flush=True)

            # Single code path - no duplication
            result = client.query(user_input, stream=False)

            if should_show_progress(client):
                console.print("\r   \r", end="", flush=True)

            # Minimal response display - no boxes
            if result["success"]:
                # Agent indicator line
                agent_name = result.get("last_agent", "supervisor")
                if agent_name and agent_name != "__end__":
                    agent_display = agent_name.replace("_", " ").title()
                else:
                    agent_display = "Lobster"
                console.print(f"\n[dim]◀ {agent_display}[/dim]")

                # Response as markdown (no panel)
                console.print(Markdown(result["response"]))

                # Plots indicator
                if result.get("plots"):
                    console.print(f"[dim #e45c47]◆ {len(result['plots'])} plot(s)[/dim #e45c47]")
            else:
                console.print(f"\n[red]✗[/red] {result['error']}")

            _maybe_print_timings(client, "Chat Query")

        except KeyboardInterrupt:
            if Confirm.ask("\n[dim]exit?[/dim]"):
                display_goodbye()

                # Display session ID for continuity
                console.print(
                    f"[dim]Session: {client.session_id} "
                    f"(use --session-id latest to continue)[/dim]"
                )

                console.print("[dim]◆ feedback: [link=https://forms.cloud.microsoft/e/AkNk8J8nE8]forms.cloud.microsoft/e/AkNk8J8nE8[/link][/dim]")

                # Display session token usage summary
                if hasattr(client, "token_tracker") and client.token_tracker:
                    summary = client.token_tracker.get_minimal_summary()
                    if summary:
                        console.print(f"[dim]{summary}[/dim]")
                console.print()  # Empty line before exit
                break
            continue
        except Exception as e:
            # Enhanced error reporting with context
            error_message = str(e)
            error_type = type(e).__name__

            # Provide context-aware suggestions
            suggestions = {
                "FileNotFoundError": "Check if the file path is correct and the file exists",
                "PermissionError": "Check file permissions or run with appropriate privileges",
                "ConnectionError": "Check your internet connection and API keys",
                "TimeoutError": "The operation timed out. Try again or check your connection",
                "ImportError": "Required dependency missing. Try reinstalling the package",
                "ValueError": "Invalid input provided. Check your command syntax",
                "KeyError": "Missing configuration or data. Check your setup",
            }

            suggestion = suggestions.get(
                error_type, "Check the error details and try again"
            )

            console_manager.print_error_panel(
                f"{error_type}: {error_message}", suggestion
            )

            # In debug mode, also print the full traceback
            if debug:
                console_manager.error_console.print_exception(
                    width=None,
                    extra_lines=3,
                    theme="monokai",
                    word_wrap=True,
                    show_locals=True,
                )


def handle_command(command: str, client: "AgentClient"):
    """Handle slash commands with enhanced error handling."""
    cmd = command.lower().strip()

    try:
        # Execute command and capture summary for history
        command_summary = _execute_command(cmd, client)

        # Add to conversation history if summary provided
        if command_summary:
            _add_command_to_history(client, command, command_summary)

        _maybe_print_timings(client, f"Command {cmd}")

    except Exception as e:
        # Enhanced command error handling
        error_message = str(e)
        error_type = type(e).__name__

        # Log command failure to history
        error_summary = f"Failed: {error_type}: {error_message[:100]}"
        _add_command_to_history(client, command, error_summary, is_error=True)

        # Command-specific error suggestions
        if cmd.startswith("/read"):
            suggestion = "Check if the file exists and you have read permissions"
        elif cmd.startswith("/plot"):
            suggestion = "Ensure plots have been generated and saved to workspace"
        elif cmd in ["/files", "/data", "/metadata"]:
            suggestion = "Check if workspace is properly initialized"
        else:
            suggestion = "Check command syntax with /help"

        console_manager.print_error_panel(
            f"Command failed ({error_type}): {error_message}", suggestion
        )


# ============================================================================
# Queue Command Helpers (MOVED TO SHARED MODULE)
# ============================================================================
# The following functions have been extracted to lobster/cli_internal/commands/
# for reuse across CLI and Dashboard interfaces:
#
# - show_queue_status()    (was _show_queue_status)
# - queue_load_file()      (was _queue_load_file)
# - queue_list()           (was _queue_list)
# - queue_clear()          (was _queue_clear)
# - queue_export()         (was _queue_export)
#
# Import these from lobster.cli_internal.commands instead of defining them here.
# This ensures CLI and Dashboard use the same implementation (Single Source of Truth).
# ============================================================================




def _execute_command(cmd: str, client: "AgentClient") -> Optional[str]:
    """Execute individual slash commands.

    Returns:
        Optional[str]: Summary of command execution for conversation history,
                      or None if command should not be logged to history.
    """
    from lobster.utils import open_path

    # -------------------------------------------------------------------------
    # Helper function for quantification directory detection
    # -------------------------------------------------------------------------
    def _is_quantification_directory(path: Path) -> Optional[str]:
        """
        Detect if directory contains Kallisto or Salmon quantification files.

        Args:
            path: Path object to check

        Returns:
            Tool type ('kallisto' or 'salmon') if quantification directory, None otherwise

        Detection criteria:
        - Kallisto: Looks for abundance.tsv, abundance.h5, or abundance.txt files
        - Salmon: Looks for quant.sf or quant.genes.sf files
        - Requires at least 2 sample subdirectories to avoid false positives
        """
        if not path.is_dir():
            return None

        kallisto_count = 0
        salmon_count = 0

        try:
            for subdir in path.iterdir():
                if not subdir.is_dir():
                    continue

                # Check for Kallisto signatures
                if (
                    (subdir / "abundance.tsv").exists()
                    or (subdir / "abundance.h5").exists()
                    or (subdir / "abundance.txt").exists()
                ):
                    kallisto_count += 1

                # Check for Salmon signatures
                if (subdir / "quant.sf").exists() or (
                    subdir / "quant.genes.sf"
                ).exists():
                    salmon_count += 1
        except (PermissionError, OSError):
            return None

        # Require at least 2 samples to identify as quantification directory
        # This avoids false positives from directories with only 1 sample
        if kallisto_count >= 2:
            return "kallisto"
        elif salmon_count >= 2:
            return "salmon"

        return None

    if cmd == "/help":
        help_text = f"""[bold white]Quick Start:[/bold white]

  [dim]• Just ask what you want to do in natural language
  • Example: "analyze my single-cell data" or "find breast cancer datasets"
  • Load data: /read <file> or drag-and-drop files (supports *.h5ad, *.csv, etc.)
  • View results: /data, /plots, /workspace[/dim]

[bold white]Key Commands:[/bold white]

[{LobsterTheme.PRIMARY_ORANGE}]/session[/{LobsterTheme.PRIMARY_ORANGE}]      [grey50]-[/grey50] Current session status
[{LobsterTheme.PRIMARY_ORANGE}]/status[/{LobsterTheme.PRIMARY_ORANGE}]       [grey50]-[/grey50] Subscription tier & agents
[{LobsterTheme.PRIMARY_ORANGE}]/data[/{LobsterTheme.PRIMARY_ORANGE}]         [grey50]-[/grey50] Current data summary
[{LobsterTheme.PRIMARY_ORANGE}]/workspace[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] Workspace info
[{LobsterTheme.PRIMARY_ORANGE}]/plots[/{LobsterTheme.PRIMARY_ORANGE}]        [grey50]-[/grey50] List generated plots
[{LobsterTheme.PRIMARY_ORANGE}]/read[/{LobsterTheme.PRIMARY_ORANGE}] <file>  [grey50]-[/grey50] Load data file (supports glob: *.h5ad)
[{LobsterTheme.PRIMARY_ORANGE}]/pipeline export[/{LobsterTheme.PRIMARY_ORANGE}] [grey50]-[/grey50] Export as Jupyter notebook
[{LobsterTheme.PRIMARY_ORANGE}]/dashboard[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] Interactive TUI dashboard
[{LobsterTheme.PRIMARY_ORANGE}]/tokens[/{LobsterTheme.PRIMARY_ORANGE}]       [grey50]-[/grey50] Token usage & costs
[{LobsterTheme.PRIMARY_ORANGE}]/clear[/{LobsterTheme.PRIMARY_ORANGE}]        [grey50]-[/grey50] Clear screen
[{LobsterTheme.PRIMARY_ORANGE}]/exit[/{LobsterTheme.PRIMARY_ORANGE}]         [grey50]-[/grey50] Exit

[dim]Shell commands:[/dim] [grey50]Standard shell commands work directly (cd, ls, pwd, mkdir, etc.)[/grey50]"""

        help_panel = LobsterTheme.create_panel(help_text, title="Help")
        console_manager.print(help_panel)

    elif cmd == "/data":
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        return data_summary(client, output)

    elif cmd == "/session":
        display_session(client)

    elif cmd == "/status":
        # Display subscription tier, packages, and agents (replicates CLI 'lobster status')
        _display_status_info()

    elif cmd == "/tokens":
        # Display token usage and cost information
        try:
            token_usage = client.get_token_usage()

            if not token_usage or "error" in token_usage:
                console_manager.print(
                    "[yellow]Token tracking not available for this client type[/yellow]"
                )
                return

            # Check if using Ollama (local = free)
            from lobster.config.llm_factory import LLMFactory
            current_provider = getattr(client, 'provider_override', None) or LLMFactory.get_current_provider()
            is_ollama = current_provider == "ollama"

            # Create summary table
            title = "🦙 Session Token Usage (FREE - Local)" if is_ollama else "💰 Session Token Usage & Cost"
            summary_table = Table(title=title, box=box.ROUNDED)
            summary_table.add_column("Metric", style="cyan", no_wrap=True)
            summary_table.add_column("Value", style="green")

            summary_table.add_row("Session ID", token_usage["session_id"])
            # Handle None provider gracefully
            provider_display = "🦙 Ollama (Local)" if is_ollama else (current_provider.capitalize() if current_provider else "Unknown")
            summary_table.add_row("Provider", provider_display)
            summary_table.add_row(
                "Total Input Tokens", f"{token_usage['total_input_tokens']:,}"
            )
            summary_table.add_row(
                "Total Output Tokens", f"{token_usage['total_output_tokens']:,}"
            )
            summary_table.add_row("Total Tokens", f"{token_usage['total_tokens']:,}")
            if is_ollama:
                summary_table.add_row("Total Cost", "[green]FREE[/green] (local model)")
            else:
                summary_table.add_row(
                    "Total Cost (USD)", f"${token_usage['total_cost_usd']:.4f}"
                )

            console_manager.print(summary_table)

            # Create per-agent breakdown table if agents have been used
            if token_usage.get("by_agent"):
                agent_title = "📊 Tokens by Agent" if is_ollama else "📊 Cost by Agent"
                agent_table = Table(title=agent_title, box=box.ROUNDED)
                agent_table.add_column("Agent", style="cyan")
                agent_table.add_column("Input", style="blue", justify="right")
                agent_table.add_column("Output", style="magenta", justify="right")
                agent_table.add_column("Total", style="yellow", justify="right")
                if not is_ollama:
                    agent_table.add_column("Cost (USD)", style="green", justify="right")
                agent_table.add_column("Calls", style="grey50", justify="right")

                for agent_name, stats in token_usage["by_agent"].items():
                    agent_display = agent_name.replace("_", " ").title()
                    if is_ollama:
                        agent_table.add_row(
                            agent_display,
                            f"{stats['input_tokens']:,}",
                            f"{stats['output_tokens']:,}",
                            f"{stats['total_tokens']:,}",
                            str(stats["invocation_count"]),
                        )
                    else:
                        agent_table.add_row(
                            agent_display,
                            f"{stats['input_tokens']:,}",
                            f"{stats['output_tokens']:,}",
                            f"{stats['total_tokens']:,}",
                            f"${stats['cost_usd']:.4f}",
                            str(stats["invocation_count"]),
                        )

                console_manager.print("\n")
                console_manager.print(agent_table)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to retrieve token usage: {str(e)}"
            )

    elif cmd == "/input-features":
        # Show input capabilities and navigation features
        input_features = console_manager.get_input_features()

        if PROMPT_TOOLKIT_AVAILABLE and input_features["arrow_navigation"]:
            features_text = f"""[bold white]✨ Enhanced Input Features Active[/bold white]

[bold {LobsterTheme.PRIMARY_ORANGE}]Available Navigation:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [green]←/→ Arrow keys[/green] - Navigate within your input text
• [green]↑/↓ Arrow keys[/green] - Browse command history
• [green]Ctrl+R[/green] - Reverse search through history
• [green]Home/End[/green] - Jump to beginning/end of line
• [green]Backspace/Delete[/green] - Edit text naturally

[bold {LobsterTheme.PRIMARY_ORANGE}]Autocomplete Features:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [green]Tab completion[/green] - Complete commands and file names
• [green]Smart context[/green] - Commands when typing /, files after /read
• [green]Live preview[/green] - See completions as you type
• [green]Rich metadata[/green] - File sizes, types, and descriptions
• [green]Cloud aware[/green] - Works with both local and cloud clients

[bold {LobsterTheme.PRIMARY_ORANGE}]History Features:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [green]Persistent history[/green] - Commands saved between sessions
• [green]History file[/green] - {input_features['history_file']}
• [green]Reverse search[/green] - Ctrl+R to find previous commands

[bold {LobsterTheme.PRIMARY_ORANGE}]Tips:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• Use ↑/↓ to recall previous commands and questions
• Use Ctrl+R followed by typing to search command history
• Press Tab to see available commands or files
• Edit recalled commands with arrow keys before pressing Enter"""
        elif PROMPT_TOOLKIT_AVAILABLE:
            features_text = f"""[bold white]✨ Autocomplete Features Active[/bold white]

[bold {LobsterTheme.PRIMARY_ORANGE}]Autocomplete Features:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [green]Tab completion[/green] - Complete commands and file names
• [green]Smart context[/green] - Commands when typing /, files after /read
• [green]Live preview[/green] - See completions as you type
• [green]Rich metadata[/green] - File sizes, types, and descriptions
• [green]Cloud aware[/green] - Works with both local and cloud clients

[bold {LobsterTheme.PRIMARY_ORANGE}]Available Input:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [yellow]Basic arrow navigation[/yellow] - Limited cursor control
• [yellow]Backspace/Delete[/yellow] - Edit text
• [yellow]Enter[/yellow] - Submit commands

[bold {LobsterTheme.PRIMARY_ORANGE}]Tips:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• Press Tab to see available commands or files
• Type / to see all available commands
• Type /read followed by Tab to see workspace files"""
        else:
            features_text = f"""[bold white]📝 Basic Input Mode[/bold white]

[bold {LobsterTheme.PRIMARY_ORANGE}]Current Capabilities:[/bold {LobsterTheme.PRIMARY_ORANGE}]
• [yellow]Basic text input[/yellow] - Standard terminal input
• [yellow]Backspace[/yellow] - Delete characters
• [yellow]Enter[/yellow] - Submit commands

[bold {LobsterTheme.PRIMARY_ORANGE}]Upgrade Available:[/bold {LobsterTheme.PRIMARY_ORANGE}]
🚀 [bold white]Get Enhanced Input Features & Autocomplete![/bold white]
Install prompt-toolkit for arrow key navigation, command history, and Tab completion:

[bold {LobsterTheme.PRIMARY_ORANGE}]pip install prompt-toolkit[/bold {LobsterTheme.PRIMARY_ORANGE}]

[bold white]After installation, you'll get:[/bold white]
• ←/→ Arrow keys for text navigation
• ↑/↓ Arrow keys for command history
• Ctrl+R for reverse search
• [green]Tab completion[/green] for commands and files
• [green]Smart autocomplete[/green] with file metadata
• [green]Cloud-aware completion[/green] for remote files
• Persistent command history between sessions"""

        features_panel = LobsterTheme.create_panel(
            features_text, title="🔤 Input Features & Navigation"
        )
        console_manager.print(features_panel)

    elif cmd == "/dashboard":
        # Launch interactive Textual dashboard
        try:
            from lobster.ui.os_app import run_lobster_os

            console_manager.print(
                f"[{LobsterTheme.PRIMARY_ORANGE}]Launching interactive dashboard...[/{LobsterTheme.PRIMARY_ORANGE}]"
            )
            # Get workspace from client
            workspace_path = getattr(client, 'workspace_path', None)
            run_lobster_os(workspace_path)
            return True  # Exit chat loop after dashboard closes
        except ImportError as e:
            console_manager.print_error_panel(
                f"Failed to import Textual UI: {str(e)}",
                "Ensure textual>=0.79.1 is installed: pip install textual",
            )
        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to launch dashboard: {e}",
                "Check system permissions and try again",
            )

    elif cmd == "/status-panel":
        # Show comprehensive system health dashboard (Rich panels)
        try:
            # Create a compact dashboard using individual panels instead of full-screen layout
            status_display = get_status_display()

            # Get individual panels from the dashboard components
            core_panel = status_display._create_core_status_panel(client)
            resource_panel = status_display._create_resource_panel()
            agent_panel = status_display._create_agent_status_panel(client)

            # Print panels individually instead of using full-screen layout
            console_manager.print(
                LobsterTheme.create_panel(
                    f"[bold {LobsterTheme.PRIMARY_ORANGE}]System Health Dashboard[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                    title="🦞 Status Panel",
                )
            )
            console_manager.print(core_panel)
            console_manager.print(resource_panel)
            console_manager.print(agent_panel)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to create status panel: {e}",
                "Check system permissions and try again",
            )

    elif cmd == "/workspace-info":
        # Show detailed workspace overview
        try:
            # Create a compact workspace overview using individual panels instead of full-screen layout
            status_display = get_status_display()

            # Get individual panels from the workspace dashboard components
            workspace_info_panel = status_display._create_workspace_info_panel(client)
            files_panel = status_display._create_recent_files_panel(client)
            data_panel = status_display._create_data_status_panel(client)

            # Print panels individually instead of using full-screen layout
            console_manager.print(
                LobsterTheme.create_panel(
                    f"[bold {LobsterTheme.PRIMARY_ORANGE}]Workspace Overview[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                    title="🏗️ Workspace",
                )
            )
            console_manager.print(workspace_info_panel)
            console_manager.print(files_panel)
            console_manager.print(data_panel)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to create workspace overview: {e}",
                "Check if workspace is properly initialized",
            )

    elif cmd == "/analysis-dash":
        # Show analysis monitoring dashboard
        try:
            # Create a compact analysis dashboard using individual panels instead of full-screen layout
            status_display = get_status_display()

            # Get individual panels from the analysis dashboard components
            analysis_panel = status_display._create_analysis_panel(client)
            plots_panel = status_display._create_plots_panel(client)

            # Print panels individually instead of using full-screen layout
            console_manager.print(
                LobsterTheme.create_panel(
                    f"[bold {LobsterTheme.PRIMARY_ORANGE}]Analysis Dashboard[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                    title="🧬 Analysis",
                )
            )
            console_manager.print(analysis_panel)
            console_manager.print(plots_panel)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to create analysis dashboard: {e}",
                "Check if analysis operations have been performed",
            )

    elif cmd == "/progress":
        # Show multi-task progress monitor
        try:
            progress_manager = get_multi_progress_manager()
            active_count = progress_manager.get_active_operations_count()

            if active_count > 0:
                # Create a compact progress display using individual panels instead of full-screen layout
                operations_panel = progress_manager._create_operations_panel()
                details_panel = progress_manager._create_details_panel()

                # Print panels individually instead of using full-screen layout
                console_manager.print(
                    LobsterTheme.create_panel(
                        f"[bold {LobsterTheme.PRIMARY_ORANGE}]Multi-Task Progress Monitor[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                        title=f"🔄 Progress ({active_count} active)",
                    )
                )
                console_manager.print(operations_panel)
                console_manager.print(details_panel)
            else:
                # Show information about the progress system
                info_text = f"""[bold white]Multi-Task Progress Monitor[/bold white]

[{LobsterTheme.PRIMARY_ORANGE}]Status:[/{LobsterTheme.PRIMARY_ORANGE}] No active multi-task operations

[bold white]Features:[/bold white]
• Real-time progress tracking for concurrent operations
• Subtask progress monitoring with detailed status
• Live updates with orange-themed progress bars
• Operation duration and completion tracking

[bold white]Usage:[/bold white]
The progress monitor automatically tracks multi-task operations
when they are started by agents or analysis workflows.

[grey50]Multi-task operations will appear here when active.[/grey50]"""

                info_panel = LobsterTheme.create_panel(
                    info_text, title="🔄 Progress Monitor"
                )
                console_manager.print(info_panel)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to create progress monitor: {e}",
                "Check system status and try again",
            )

    elif cmd.startswith("/metadata"):
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        parts = cmd.split()
        subcommand = parts[1] if len(parts) > 1 else None

        if subcommand == "clear":
            return metadata_clear(client, output)
        elif subcommand == "list" or subcommand is None:
            return metadata_list(client, output)
        else:
            console.print(f"[yellow]Unknown metadata subcommand: {subcommand}[/yellow]")
            console.print("[cyan]Available: list, clear[/cyan]")
            return None

    elif cmd == "/files":
        # Get categorized workspace files from data_manager
        workspace_files = client.data_manager.list_workspace_files()

        if any(workspace_files.values()):
            for category, files in workspace_files.items():
                if files:
                    # Sort files by modified date (descending: newest first)
                    files_sorted = sorted(
                        files, key=lambda f: f["modified"], reverse=True
                    )

                    # Create themed table
                    table = Table(
                        title=f"🦞 {category.title()} Files",
                        **LobsterTheme.get_table_style(),
                    )
                    table.add_column("Name", style="bold white")
                    table.add_column("Size", style="grey74")
                    table.add_column("Modified", style="grey50")
                    table.add_column("Path", style="dim grey50")

                    for f in files_sorted:
                        from datetime import datetime

                        size_kb = f["size"] / 1024
                        mod_time = datetime.fromtimestamp(f["modified"]).strftime(
                            "%Y-%m-%d %H:%M"
                        )
                        table.add_row(
                            f["name"],
                            f"{size_kb:.1f} KB",
                            mod_time,
                            Path(f["path"]).parent.name,
                        )

                    console_manager.print(table)
                    console_manager.print()  # Add spacing between categories
        else:
            console_manager.print("[grey50]No files in workspace[/grey50]")

    elif cmd == "/tree":
        # Show directory tree view
        try:
            # Show current directory tree
            current_tree = create_file_tree(
                root_path=current_directory,
                title=f"Current Directory: {current_directory.name}",
                show_hidden=False,
                max_depth=3,
            )

            tree_panel = LobsterTheme.create_panel(
                current_tree, title="📁 Directory Tree"
            )
            console_manager.print(tree_panel)

            # Also show workspace tree if it exists
            from lobster.core.workspace import resolve_workspace

            workspace_path = resolve_workspace(explicit_path=client.workspace_path, create=False)
            if workspace_path.exists():
                console_manager.print()  # Add spacing
                workspace_tree = create_workspace_tree(workspace_path)

                workspace_panel = LobsterTheme.create_panel(
                    workspace_tree, title="🦞 Workspace Tree"
                )
                console_manager.print(workspace_panel)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to create tree view: {e}",
                "Check directory permissions and try again",
            )

    elif cmd.startswith("/workspace"):
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        parts = cmd.split()

        # Default to showing status summary when no args (like /config and /queue)
        if len(parts) == 1:
            return workspace_status(client, output)

        subcommand = parts[1]

        if subcommand == "list":
            force_refresh = "--refresh" in cmd.lower()
            return workspace_list(client, output, force_refresh)
        elif subcommand == "info":
            selector = parts[2] if len(parts) > 2 else None
            return workspace_info(client, output, selector)
        elif subcommand == "load":
            selector = parts[2] if len(parts) > 2 else None
            return workspace_load(client, output, selector, current_directory, PathResolver)
        elif subcommand == "remove":
            modality_name = parts[2] if len(parts) > 2 else None
            return workspace_remove(client, output, modality_name)
        elif subcommand == "status":
            return workspace_status(client, output)
        else:
            console.print(f"[yellow]Unknown workspace subcommand: {subcommand}[/yellow]")
            console.print("[cyan]Available: status, list, info, load, remove[/cyan]")
            return None

    elif cmd.startswith("/queue"):
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        parts = cmd.split()

        if len(parts) == 1:
            # Show queue status
            return show_queue_status(client, output)

        subcommand = parts[1] if len(parts) > 1 else None

        if subcommand == "load":
            filename = parts[2] if len(parts) > 2 else None
            try:
                return queue_load_file(client, filename, output, current_directory)
            except QueueFileTypeNotSupported as e:
                console.print(f"[yellow]⚠️  {str(e)}[/yellow]")
                return None

        elif subcommand == "list":
            # Determine queue type from additional parameters
            queue_type = "publication"  # default
            if len(parts) > 2:
                if parts[2] == "download":
                    queue_type = "download"
                elif parts[2] == "publication":
                    queue_type = "publication"
                else:
                    console.print(f"[yellow]Unknown queue type: {parts[2]}[/yellow]")
                    console.print("[cyan]Usage: /queue list [download|publication][/cyan]")
                    return None
            return queue_list(client, output, queue_type=queue_type)

        elif subcommand == "clear":
            # Determine queue type from additional parameters
            queue_type = "publication"  # default
            if len(parts) > 2:
                if parts[2] == "download":
                    queue_type = "download"
                elif parts[2] == "all":
                    queue_type = "all"
                else:
                    console.print(f"[yellow]Unknown queue type: {parts[2]}[/yellow]")
                    console.print("[cyan]Usage: /queue clear [download|all][/cyan]")
                    return None
            return queue_clear(client, output, queue_type=queue_type)

        elif subcommand == "export":
            name = parts[2] if len(parts) > 2 else None
            return queue_export(client, name, output)

        else:
            console.print(f"[yellow]Unknown queue subcommand: {subcommand}[/yellow]")
            console.print("[cyan]Available: load, list, clear, export[/cyan]")
            return None

    # =========================================================================
    # /read - File Inspection Only (no state change, view-only)
    # =========================================================================
    elif cmd.startswith("/read "):
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        filename = cmd[6:].strip()
        return file_read(client, output, filename, current_directory, PathResolver)

    elif cmd.startswith("/archive"):
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        parts = cmd.split()
        subcommand = parts[1] if len(parts) > 1 else "help"
        args = parts[2:] if len(parts) > 2 else None
        return archive_queue(client, output, subcommand, args)

    elif cmd.startswith("/pipeline"):
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        parts = cmd.split()
        subcommand = parts[1] if len(parts) > 1 else None

        if subcommand == "export":
            name = parts[2] if len(parts) > 2 else None
            description = " ".join(parts[3:]) if len(parts) > 3 else None
            return pipeline_export(client, output, name, description)
        elif subcommand == "list":
            return pipeline_list(client, output)
        elif subcommand == "run":
            notebook_name = parts[2] if len(parts) > 2 else None
            input_modality = parts[3] if len(parts) > 3 else None
            return pipeline_run(client, output, notebook_name, input_modality)
        elif subcommand == "info":
            return pipeline_info(client, output)
        else:
            console.print(f"[yellow]Unknown pipeline subcommand: {subcommand}[/yellow]")
            console.print("[cyan]Available: export, list, run, info[/cyan]")
            return None

    elif cmd == "/export":
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        return export_data(client, output)

        # Show generated plots
        plots = client.data_manager.get_plot_history()

        if plots:
            table = Table(
                title="🦞 Generated Plots",
                box=box.ROUNDED,
                border_style="red",
                title_style="bold red on white",
            )
            table.add_column("ID", style="bold white")
            table.add_column("Title", style="white")
            table.add_column("Source", style="grey74")
            table.add_column("Created", style="grey50")

            for plot in plots:
                from datetime import datetime

                try:
                    created = datetime.fromisoformat(
                        plot["timestamp"].replace("Z", "+00:00")
                    )
                    created_str = created.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    created_str = plot["timestamp"][:16] if plot["timestamp"] else "N/A"

                table.add_row(
                    plot["id"], plot["title"], plot["source"] or "N/A", created_str
                )

            console.print(table)
        else:
            console.print("[grey50]No plots generated yet[/grey50]")


    elif cmd.startswith("/open "):
        # Handle /open command for files and folders
        file_or_folder = cmd[6:].strip()

        if not file_or_folder:
            console.print("[red]/open: missing file or folder argument[/red]")
            console.print("[grey50]Usage: /open <file_or_folder>[/grey50]")
            return "No file or folder specified for /open command"

    elif cmd == "/plots":
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        return plots_list(client, output)

    elif cmd == "/modalities":
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        return modalities_list(client, output)

    elif cmd.startswith("/describe"):
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        parts = cmd.split()
        modality_name = parts[1] if len(parts) > 1 else None
        return modality_describe(client, output, modality_name)

    elif cmd == "/save":
        # Auto-save current state
        saved_items = client.data_manager.auto_save_state()

        if saved_items:
            console.print("[bold red]✓[/bold red] [white]Saved to workspace:[/white]")
            for item in saved_items:
                console.print(f"  • {item}")
            return f"Saved {len(saved_items)} items to workspace: {', '.join(saved_items[:3])}{'...' if len(saved_items) > 3 else ''}"
        else:
            console.print("[grey50]Nothing to save (no data or plots loaded)[/grey50]")
            return "No data or plots to save"

    elif cmd.startswith("/restore"):
        # Restore previous session
        parts = cmd.split()
        pattern = "recent"  # default
        if len(parts) > 1:
            pattern = parts[1]

        # Show what will be restored
        console.print(f"[yellow]Restoring workspace (pattern: {pattern})...[/yellow]")

    elif cmd.startswith("/plot"):
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        parts = cmd.split()
        plot_identifier = parts[1] if len(parts) > 1 else None
        return plot_show(client, output, plot_identifier)

    elif cmd.startswith("/config"):
        # Use shared command implementation (unified with dashboard)
        output = ConsoleOutputAdapter(console)
        parts = cmd.split()
        
        if len(parts) == 1 or parts[1] == "show":
            return config_show(client, output)
        elif parts[1] == "provider":
            if len(parts) == 2 or parts[2] == "list":
                return config_provider_list(client, output)
            elif parts[2] == "switch":
                provider_name = parts[3] if len(parts) > 3 else None
                save = "--save" in parts
                return config_provider_switch(client, output, provider_name, save)
        elif parts[1] == "model":
            if len(parts) == 2 or parts[2] == "list":
                return config_model_list(client, output)
            elif parts[2] == "switch":
                model_name = parts[3] if len(parts) > 3 else None
                save = "--save" in parts
                return config_model_switch(client, output, model_name, save)
        else:
            console.print(f"[yellow]Unknown config subcommand: {parts[1]}[/yellow]")
            console.print("[cyan]Available: show, provider, model[/cyan]")
            return None

    elif cmd == "/clear":
        console.clear()

    elif cmd == "/exit":
        if Confirm.ask("[dim]exit?[/dim]"):
            display_goodbye()

            # Display session ID for continuity
            console.print(
                f"[dim]Session: {client.session_id} "
                f"(use --session-id latest to continue)[/dim]"
            )

            console.print("[dim]◆ feedback: [link=https://forms.cloud.microsoft/e/AkNk8J8nE8]forms.cloud.microsoft/e/AkNk8J8nE8[/link][/dim]")

            # Display session token usage summary
            if hasattr(client, "token_tracker") and client.token_tracker:
                summary = client.token_tracker.get_minimal_summary()
                if summary:
                    console.print(f"[dim]{summary}[/dim]")
            console.print()  # Empty line before exit
            raise KeyboardInterrupt

    else:
        console.print(
            f"[bold red on white] ⚠️  Error [/bold red on white] [red]Unknown command: {cmd}[/red]"
        )


@app.command(name="dashboard")
def dashboard_command(
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory. Can also be set via LOBSTER_WORKSPACE env var. Default: ./.lobster_workspace",
    ),
):
    """
    Launch interactive dashboard (Textual UI).

    This command starts a Textual-based interactive terminal UI with:
    - Multi-panel cockpit layout for real-time monitoring
    - Live agent activity tracking and handoff visualization
    - Query input with streaming responses
    - Token usage and system status panels

    Press ESC to quit, ^P for command palette.
    """
    try:
        from lobster.ui.os_app import run_lobster_os

        run_lobster_os(workspace)
    except ImportError as e:
        console.print(
            f"[red]❌ Failed to import Textual UI: {str(e)}[/red]\n"
            "[yellow]Ensure textual>=0.79.1 is installed: pip install textual[/yellow]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Failed to launch dashboard: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def query(
    question: str,
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory. Can also be set via LOBSTER_WORKSPACE env var. Default: ./.lobster_workspace",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Session ID to continue (use 'latest' for most recent session in workspace)",
    ),
    reasoning: bool = typer.Option(
        False,
        "--reasoning",
        is_flag=True,
        help="Show agent reasoning and thinking process",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed tool usage and agent activity"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug mode with detailed logging"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    profile_timings: Optional[bool] = typer.Option(
        None,
        "--profile-timings/--no-profile-timings",
        help="Enable timing diagnostics for data manager operations",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider to use (bedrock, anthropic, ollama). Overrides auto-detection.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use (e.g., claude-4-sonnet, llama3:70b-instruct, mixtral:8x7b). Overrides configuration.",
    ),
):
    """
    Send a single query to the agent system.

    Use --session-id to continue a previous conversation:
      lobster query "follow up question" --session-id latest
      lobster query "follow up question" --session-id session_20241208_150000

    Agent reasoning is shown by default. Use --no-reasoning to disable.
    """
    # Check for configuration
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        console.print(
            f"[red]❌ No configuration found. Run 'lobster init' first.[/red]"
        )
        raise typer.Exit(1)

    # Handle session loading for continuity
    # Determine session_id before client initialization
    session_file_to_load = None
    session_id_for_client = None

    if session_id:
        # Resolve workspace early to check for session files
        from lobster.core.workspace import resolve_workspace
        workspace_path = resolve_workspace(explicit_path=workspace, create=True)

        if session_id == "latest":
            # Find most recent session file
            session_files = list(workspace_path.glob("session_*.json"))
            if session_files:
                # Sort by modification time (most recent first)
                session_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                session_file_to_load = session_files[0]
                console.print(
                    f"[cyan]📂 Loading latest session: {session_file_to_load.name}[/cyan]"
                )
            else:
                console.print(
                    "[yellow]⚠️  No previous sessions found - creating new session[/yellow]"
                )
        else:
            # Explicit session ID provided
            # Try with session_ prefix first
            session_file_candidate = workspace_path / f"session_{session_id}.json"
            if not session_file_candidate.exists():
                # Try exact filename
                session_file_candidate = workspace_path / f"{session_id}.json"

            if session_file_candidate.exists():
                session_file_to_load = session_file_candidate
            else:
                # Session file doesn't exist - use this session_id for new session
                session_id_for_client = session_id
                console.print(
                    f"[cyan]📂 Creating new session: {session_id}[/cyan]"
                )

    # Initialize client with custom session_id if new session
    client = init_client(workspace, reasoning, verbose, debug, profile_timings, provider, model, session_id_for_client)

    # Load session if found
    if session_file_to_load:
        try:
            load_result = client.load_session(session_file_to_load)
            console.print(
                f"[green]✓ Loaded {load_result['messages_loaded']} "
                f"previous messages[/green]"
            )
            if load_result.get("original_session_id"):
                console.print(
                    f"[dim]  Original session: "
                    f"{load_result['original_session_id']}[/dim]"
                )
        except Exception as e:
            console.print(f"[red]❌ Failed to load session: {e}[/red]")
            console.print("[yellow]   Creating new session instead[/yellow]")

    # Process query
    if should_show_progress(client):
        with console.status("[red]🦞 Processing query...[/red]"):
            result = client.query(question)
    else:
        # In verbose/reasoning mode, no progress indication
        result = client.query(question)

    # Display or save result
    if result["success"]:
        if output:
            output.write_text(result["response"])
            console.print(
                f"[bold red]✓[/bold red] [white]Response saved to:[/white] [grey74]{output}[/grey74]"
            )
        else:
            console.print(
                Panel(
                    Markdown(result["response"]),
                    title="[bold white on red] 🦞 Lobster Response [/bold white on red]",
                    border_style="red",
                    box=box.DOUBLE,
                )
            )

        # Display session ID for continuity
        console.print(
            f"\n[dim]Session: {client.session_id} "
            f"(use --session-id latest for follow-ups)[/dim]"
        )

        # Display minimal token usage summary
        if hasattr(client, "token_tracker") and client.token_tracker:
            summary = client.token_tracker.get_minimal_summary()
            if summary:
                console.print(f"[dim]{summary}[/dim]")
    else:
        console.print(
            f"[bold red on white] ⚠️  Error [/bold red on white] [red]{result['error']}[/red]"
        )

    _maybe_print_timings(client, "Query")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host"),
):
    """
    Start the agent system as an API server (for React UI).
    """
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    # Create FastAPI app
    api = FastAPI(
        title="Lobster Agent API",
        description="🦞 Multi-Agent Bioinformatics System by Omics-OS",
        version="2.0",
    )

    class QueryRequest(BaseModel):
        question: str
        session_id: Optional[str] = None
        stream: bool = False

    @api.post("/query")
    async def query_endpoint(request: QueryRequest):
        try:
            client = init_client()
            result = client.query(request.question, stream=request.stream)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @api.get("/status")
    async def status_endpoint():
        client = init_client()
        return client.get_status()

    console.print(f"[red]🦞 Starting Lobster API server on {host}:{port}[/red]")
    uvicorn.run(api, host=host, port=port)


# Config subcommands
@config_app.command(name="list-models")
def list_models():
    """List all available model presets."""
    configurator = LobsterAgentConfigurator()
    models = configurator.list_available_models()

    console.print("\n[cyan]🤖 Available Model Presets[/cyan]")
    console.print("[cyan]" + "=" * 60 + "[/cyan]")

    table = Table(
        box=box.ROUNDED,
        border_style="cyan",
        title="🤖 Available Model Presets",
        title_style="bold cyan",
    )

    table.add_column("Preset Name", style="bold white")
    table.add_column("Tier", style="cyan")
    table.add_column("Region", style="white")
    table.add_column("Temperature", style="white")
    table.add_column("Description", style="white")

    for name, config in models.items():
        description = (
            config.description[:40] + "..."
            if len(config.description) > 40
            else config.description
        )
        table.add_row(
            name,
            config.tier.value.title(),
            config.region,
            f"{config.temperature}",
            description,
        )

    console.print(table)


@config_app.command(name="list-profiles")
def list_profiles():
    """List all available testing profiles."""
    configurator = LobsterAgentConfigurator()
    profiles = configurator.list_available_profiles()

    console.print("\n[cyan]⚙️  Available Testing Profiles[/cyan]")
    console.print("[cyan]" + "=" * 60 + "[/cyan]")

    for profile_name, config in profiles.items():
        console.print(f"\n[yellow]📋 {profile_name.title()}[/yellow]")
        for agent, model in config.items():
            console.print(f"   {agent}: {model}")


@config_app.command(name="show-config")
def show_config(
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-w", help="Workspace path (default: current directory)"
    ),
    show_all: bool = typer.Option(
        False,
        "--show-all",
        help="Show all configured agents regardless of license tier",
    ),
):
    """Show current runtime configuration from ConfigResolver and ProviderRegistry."""
    from lobster.core.config_resolver import ConfigResolver, ConfigurationError
    from lobster.config.providers import ProviderRegistry, get_provider
    from lobster.core.license_manager import get_current_tier
    from lobster.config.subscription_tiers import is_agent_available, get_tier_display_name
    from lobster.config.agent_registry import AGENT_REGISTRY
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.config.global_config import GlobalProviderConfig
    from lobster.core.workspace import resolve_workspace

    # Resolve workspace path
    workspace_path = resolve_workspace(workspace, create=False)

    # Get ConfigResolver instance
    resolver = ConfigResolver.get_instance(workspace_path)

    # Get license tier
    current_tier = get_current_tier()

    console.print()
    console.print(
        Panel.fit(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]🔧 Lobster AI Runtime Configuration[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            padding=(0, 2),
        )
    )
    console.print()

    # Section 1: License & Workspace
    console.print(f"[bold cyan]📍 Environment[/bold cyan]")
    console.print(f"   Workspace: [yellow]{workspace_path}[/yellow]")
    console.print(f"   License Tier: [green]{get_tier_display_name(current_tier)}[/green]")
    console.print()

    # Section 2: Provider Configuration
    console.print(f"[bold cyan]🔌 Provider Configuration[/bold cyan]")

    try:
        provider_name, provider_source = resolver.resolve_provider()
        provider_obj = get_provider(provider_name)

        console.print(f"   Active Provider: [green]{provider_obj.display_name if provider_obj else provider_name}[/green]")
        console.print(f"   Source: [dim]{provider_source}[/dim]")

        if provider_obj:
            # Show provider-specific information
            default_model = provider_obj.get_default_model()
            console.print(f"   Default Model: [yellow]{default_model}[/yellow]")

            # Show available models for this provider
            models = provider_obj.list_models()
            if models:
                console.print(f"   Available Models: [dim]{len(models)} model(s)[/dim]")

    except ConfigurationError as e:
        console.print(f"   [red]✗ No provider configured[/red]")
        console.print(f"   [dim]Error: {str(e)}[/dim]")
        console.print(f"   [yellow]💡 Run 'lobster init' to configure a provider[/yellow]")

    console.print()

    # Section 3: Profile Configuration
    console.print(f"[bold cyan]⚙️  Profile Configuration[/bold cyan]")

    try:
        profile, profile_source = resolver.resolve_profile()
        console.print(f"   Active Profile: [green]{profile}[/green]")
        console.print(f"   Source: [dim]{profile_source}[/dim]")
    except Exception:
        console.print(f"   Active Profile: [yellow]production[/yellow] [dim](default)[/dim]")

    console.print()

    # Section 4: Configuration Files
    console.print(f"[bold cyan]📁 Configuration Files[/bold cyan]")

    # Workspace config
    workspace_config_path = workspace_path / "provider_config.json"
    if workspace_config_path.exists():
        console.print(f"   [green]✓[/green] Workspace: {workspace_config_path}")
        try:
            ws_config = WorkspaceProviderConfig.load(workspace_path)
            if ws_config.global_provider:
                console.print(f"      Provider: [yellow]{ws_config.global_provider}[/yellow]")
            if ws_config.profile:
                console.print(f"      Profile: [yellow]{ws_config.profile}[/yellow]")
        except Exception as e:
            console.print(f"      [dim]Could not read: {str(e)}[/dim]")
    else:
        console.print(f"   [dim]○ Workspace: {workspace_config_path} (not found)[/dim]")

    # Global config
    global_config_path = Path.home() / ".config" / "lobster" / "providers.json"
    if global_config_path.exists():
        console.print(f"   [green]✓[/green] Global: {global_config_path}")
        try:
            global_config = GlobalProviderConfig.load()
            if global_config.global_default_provider:
                console.print(f"      Provider: [yellow]{global_config.global_default_provider}[/yellow]")
            if global_config.default_profile:
                console.print(f"      Profile: [yellow]{global_config.default_profile}[/yellow]")
        except Exception as e:
            console.print(f"      [dim]Could not read: {str(e)}[/dim]")
    else:
        console.print(f"   [dim]○ Global: {global_config_path} (not found)[/dim]")

    console.print()

    # Section 5: Per-Agent Configuration
    console.print(f"[bold cyan]🤖 Agent Configuration[/bold cyan]")
    console.print()

    # Get agent configurations from settings (for temperature/thinking)
    from lobster.config.settings import get_settings
    settings = get_settings()

    displayed_count = 0
    filtered_count = 0

    for agent_name, agent_cfg in AGENT_REGISTRY.items():
        # Check if agent is available for current tier
        if not show_all and not is_agent_available(agent_name, current_tier):
            filtered_count += 1
            continue

        displayed_count += 1

        try:
            # Get model parameters from settings (temperature, thinking)
            model_params = settings.get_agent_llm_params(agent_name)

            # Try to resolve model for this agent
            try:
                provider_name, _ = resolver.resolve_provider()
                model_id, model_source = resolver.resolve_model(
                    agent_name=agent_name,
                    runtime_override=None,
                    provider=provider_name
                )

                # If no model resolved, get provider's default model
                if not model_id:
                    provider_obj = get_provider(provider_name)
                    if provider_obj:
                        model_id = provider_obj.get_default_model()
                        model_source = "provider default"
                    else:
                        model_id = model_params.get("model_id", "unknown")
                        model_source = "profile config"

            except Exception:
                # Fallback: try to get from provider's default
                try:
                    provider_name, _ = resolver.resolve_provider()
                    provider_obj = get_provider(provider_name)
                    if provider_obj:
                        model_id = provider_obj.get_default_model()
                        model_source = "provider default"
                    else:
                        model_id = model_params.get("model_id", "unknown")
                        model_source = "profile config"
                except Exception:
                    model_id = model_params.get("model_id", "unknown")
                    model_source = "profile config"

            # Display agent configuration
            console.print(f"   [bold white]{agent_cfg.display_name}[/bold white] ([dim]{agent_name}[/dim])")
            console.print(f"      Model: [yellow]{model_id}[/yellow] [dim](from {model_source})[/dim]")
            console.print(f"      Temperature: [cyan]{model_params.get('temperature', 1.0)}[/cyan]")

            # Show thinking configuration if present
            additional_fields = model_params.get("additional_model_request_fields", {})
            if "thinking" in additional_fields:
                thinking = additional_fields["thinking"]
                budget = thinking.get("budget_tokens", "unknown")
                console.print(f"      [dim]🧠 Thinking: Enabled (Budget: {budget} tokens)[/dim]")

            console.print()

        except Exception as e:
            console.print(f"   [bold white]{agent_cfg.display_name}[/bold white]: [red]Error - {str(e)}[/red]")
            console.print()

    # Summary
    if filtered_count > 0 and not show_all:
        console.print(f"[dim]{'─' * 60}[/dim]")
        console.print(
            f"[cyan]📊 Summary:[/cyan] Showing {displayed_count} agents available for [green]{current_tier}[/green] tier"
        )
        console.print(f"   [dim]({filtered_count} premium agents hidden)[/dim]")
        console.print(f"   [yellow]💡 Use '--show-all' to see all configured agents[/yellow]")

    console.print()


@config_app.command(name="test")
def test(
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Profile to test (optional, auto-detects if not provided)"),
    agent: Optional[str] = typer.Option(
        None, "--agent", "-a", help="Specific agent to test"
    ),
):
    """Test LLM provider connectivity and configuration.

    If no --profile is specified, auto-detects the currently configured provider
    and tests basic connectivity. With --profile, tests the full agent configuration.
    """
    # Load .env file for provider detection
    from dotenv import load_dotenv
    load_dotenv()

    # If no profile specified, test current provider connectivity
    if not profile:
        console.print()
        console.print(
            Panel.fit(
                f"[bold {LobsterTheme.PRIMARY_ORANGE}]🔍 Testing LLM Provider[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                border_style=LobsterTheme.PRIMARY_ORANGE,
                padding=(0, 2),
            )
        )
        console.print()

        from lobster.config.providers import ProviderRegistry
        from lobster.config.llm_factory import LLMFactory

        # Auto-detect current provider (use ProviderRegistry directly)
        try:
            # Get list of configured providers
            configured = ProviderRegistry.get_configured_providers()

            if not configured:
                console.print("[red]❌ No LLM provider configured[/red]")
                console.print("[yellow]💡 Run 'lobster init' to configure a provider[/yellow]")
                raise typer.Exit(1)

            # Check environment variable first
            import os
            explicit_provider = os.getenv("LOBSTER_LLM_PROVIDER")
            if explicit_provider and explicit_provider in [p.name for p in configured]:
                provider = explicit_provider
            else:
                # Use first configured provider
                provider = configured[0].name

            if provider is None:
                console.print("[red]❌ No LLM provider configured[/red]")
                console.print("[yellow]💡 Run 'lobster init' to configure a provider[/yellow]")
                raise typer.Exit(1)

            # provider is a string, not an enum
            console.print(f"[cyan]Detected provider:[/cyan] {provider}")
            console.print()

            # Test provider connectivity
            console.print("[yellow]Testing API connectivity...[/yellow]")

            try:
                # Get provider object and create model directly (bypass ConfigResolver)
                from lobster.config.providers import get_provider
                provider_obj = get_provider(provider)

                if not provider_obj:
                    raise Exception(f"Provider '{provider}' not found in registry")

                # Get default model for provider
                default_model = provider_obj.get_default_model()

                # Create test model directly via provider
                test_llm = provider_obj.create_chat_model(
                    model_id=default_model,
                    temperature=1.0,
                    max_tokens=50
                )

                response = test_llm.invoke("Reply with just 'ok'")

                console.print()
                console.print(
                    Panel.fit(
                        f"[bold green]✅ Provider Test Successful[/bold green]\n\n"
                        f"Provider: [cyan]{provider}[/cyan]\n"
                        f"Status: [green]Connected[/green]\n\n"
                        f"Your {provider} provider is working correctly.\n"
                        f"Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster chat[/bold {LobsterTheme.PRIMARY_ORANGE}] to start analyzing!",
                        border_style="green",
                        padding=(1, 2),
                    )
                )
                return True

            except Exception as e:
                error_msg = str(e)
                console.print()
                console.print(
                    Panel.fit(
                        f"[bold red]❌ Provider Test Failed[/bold red]\n\n"
                        f"Provider: [cyan]{provider}[/cyan]\n"
                        f"Error: [red]{error_msg[:100]}[/red]\n\n"
                        f"[yellow]💡 Check your API keys in .env file[/yellow]\n"
                        f"Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster init --force[/bold {LobsterTheme.PRIMARY_ORANGE}] to reconfigure",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                raise typer.Exit(1)

        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            raise typer.Exit(1)

    # Profile-based testing (original functionality)
    try:
        configurator = initialize_configurator(profile=profile)

        if agent:
            # Test specific agent
            try:
                config = configurator.get_agent_model_config(agent)
                configurator.get_llm_params(agent)

                console.print(
                    f"\n[green]✅ Agent '{agent}' configuration is valid[/green]"
                )
                console.print(f"   Model: {config.model_config.model_id}")
                console.print(f"   Tier: {config.model_config.tier.value}")
                console.print(f"   Region: {config.model_config.region}")

            except KeyError:
                console.print(
                    f"\n[red]❌ Agent '{agent}' not found in profile '{profile}'[/red]"
                )
                return False
        else:
            # Test all agents dynamically
            console.print(f"\n[yellow]🧪 Testing Profile: {profile}[/yellow]")
            all_valid = True

            # Get all agents from the configurator's DEFAULT_AGENTS
            available_agents = configurator.DEFAULT_AGENTS

            for agent_name in available_agents:
                try:
                    config = configurator.get_agent_model_config(agent_name)
                    configurator.get_llm_params(agent_name)
                    console.print(
                        f"   [green]✅ {agent_name}: {config.model_config.model_id}[/green]"
                    )
                except Exception as e:
                    console.print(f"   [red]❌ {agent_name}: {str(e)}[/red]")
                    all_valid = False

            if all_valid:
                console.print(
                    f"\n[green]🎉 Profile '{profile}' is fully configured and valid![/green]"
                )
            else:
                console.print(
                    f"\n[yellow]⚠️  Profile '{profile}' has configuration issues[/yellow]"
                )

        return True

    except Exception as e:
        console.print(f"\n[red]❌ Error testing configuration: {str(e)}[/red]")
        return False


@config_app.command(name="create-custom")
def create_custom():
    """Interactive creation of custom configuration."""
    console.print("\n[cyan]🛠️  Create Custom Configuration[/cyan]")
    console.print("[cyan]" + "=" * 50 + "[/cyan]")

    configurator = LobsterAgentConfigurator()
    available_models = configurator.list_available_models()

    # Show available models
    console.print("\n[yellow]Available models:[/yellow]")
    for i, (name, config) in enumerate(available_models.items(), 1):
        console.print(f"{i:2}. {name} ({config.tier.value}, {config.region})")

    config_data = {"profile": "custom", "agents": {}}

    # Use dynamic agent list
    agents = configurator.DEFAULT_AGENTS

    for agent in agents:
        console.print(f"\n[yellow]Configuring {agent}:[/yellow]")
        console.print("Choose a model preset (enter number or name):")

        choice = Prompt.ask(f"Model for {agent}")

        # Handle numeric choice
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                model_name = list(available_models.keys())[idx]
            else:
                console.print(
                    "[yellow]Invalid choice, using default (claude-sonnet)[/yellow]"
                )
                model_name = "claude-sonnet"
        else:
            # Handle name choice
            if choice in available_models:
                model_name = choice
            else:
                console.print(
                    "[yellow]Invalid choice, using default (claude-sonnet)[/yellow]"
                )
                model_name = "claude-sonnet"

        model_config = available_models[model_name]
        config_data["agents"][agent] = {
            "model_config": {
                "provider": model_config.provider.value,
                "model_id": model_config.model_id,
                "tier": model_config.tier.value,
                "temperature": model_config.temperature,
                "region": model_config.region,
                "description": model_config.description,
            },
            "enabled": True,
            "custom_params": {},
        }

        console.print(f"   [green]Selected: {model_name}[/green]")

    # Save configuration
    config_file = "config/custom_agent_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    console.print(f"\n[green]✅ Custom configuration saved to: {config_file}[/green]")
    console.print("[yellow]To use this configuration, set:[/yellow]")
    console.print(f"   export LOBSTER_CONFIG_FILE={config_file}", style="yellow")


@config_app.command(name="generate-env")
def generate_env():
    """Generate .env template with all available options."""
    template = """# LOBSTER AI Configuration Template
# Copy this file to .env and configure as needed

# =============================================================================
# API KEYS (Required)
# =============================================================================
AWS_BEDROCK_ACCESS_KEY="your-aws-access-key-here"
AWS_BEDROCK_SECRET_ACCESS_KEY="your-aws-secret-key-here"
NCBI_API_KEY="your-ncbi-api-key-here"

# =============================================================================
# AGENT CONFIGURATION (Professional System)
# =============================================================================

# Profile-based configuration (recommended)
# Available profiles: development, production, ultra, godmode
LOBSTER_PROFILE=production

# OR use custom configuration file
# LOBSTER_CONFIG_FILE=config/custom_agent_config.json

# Per-agent model overrides (optional)
# Available models: claude-haiku, claude-sonnet, claude-sonnet-eu, claude-opus, claude-opus-eu, claude-3-7-sonnet, claude-3-7-sonnet-eu
# LOBSTER_SUPERVISOR_MODEL=claude-haiku
# LOBSTER_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus
# LOBSTER_METHOD_AGENT_MODEL=claude-sonnet
# LOBSTER_GENERAL_CONVERSATION_MODEL=claude-haiku

# Global model override (overrides all agents)
# LOBSTER_GLOBAL_MODEL=claude-sonnet

# Per-agent temperature overrides
# LOBSTER_SUPERVISOR_TEMPERATURE=0.5
# LOBSTER_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7
# LOBSTER_METHOD_AGENT_TEMPERATURE=0.3

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Server configuration
PORT=8501
HOST=0.0.0.0
DEBUG=False

# Data processing
LOBSTER_MAX_FILE_SIZE_MB=500
LOBSTER_CLUSTER_RESOLUTION=0.5
LOBSTER_CACHE_DIR=data/cache

# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================

# Example 1: Development setup (Claude Haiku 4.5 - fastest, most affordable)
# LOBSTER_PROFILE=development

# Example 2: Production setup (Claude Sonnet 4 - balanced quality & speed)
# LOBSTER_PROFILE=production

# Example 3: Ultra setup (Claude Sonnet 4.5 - highest quality)
# LOBSTER_PROFILE=ultra

# Example 4: Godmode setup (Claude Opus 4.1 - experimental, most expensive)
# LOBSTER_PROFILE=godmode
"""

    with open(".env.template", "w") as f:
        f.write(template)

    console.print("[green]✅ Environment template saved to: .env.template[/green]")
    console.print("[yellow]Copy this file to .env and configure your API keys[/yellow]")


if __name__ == "__main__":
    app()
