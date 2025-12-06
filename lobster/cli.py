#!/usr/bin/env python3
"""
Modern, user-friendly CLI for the Multi-Agent Bioinformatics System.
Installable via pip or curl, with rich terminal interface.
"""

import os
import threading
from pathlib import Path
from typing import Optional

os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "900000"
import ast
import inspect
import json
import logging
import shutil
import time
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
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
from lobster.config import provider_setup
from lobster.config.agent_config import (
    LobsterAgentConfigurator,
    get_agent_configurator,
    initialize_configurator,
)
from lobster.core.client import AgentClient

# Import extraction cache manager (premium feature - graceful fallback if unavailable)
try:
    from lobster.core.extraction_cache import ExtractionCacheManager
    HAS_EXTRACTION_CACHE = True
except ImportError:
    # Premium feature not available in open-core distribution
    ExtractionCacheManager = None
    HAS_EXTRACTION_CACHE = False

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

# Import the proper callback handler and system utilities
from lobster.utils import SimpleTerminalCallback, TerminalCallbackHandler, open_path
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


class QueueFileTypeNotSupported(NotImplementedError):
    """Raised when /queue load encounters an unsupported file type.

    This exception provides helpful messaging for users about which file types
    are currently supported and which are planned for future releases.
    """

    pass


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
    client: AgentClient, command: str, summary: str, is_error: bool = False
) -> bool:
    """
    Add command execution to conversation history for AI context.

    BUG FIX #4: Enhanced error handling with full logging and file backup.
    - Returns bool to indicate success/failure
    - Logs full error messages with stack traces (not truncated to 50 chars)
    - Implements file backup for audit trail and recovery
    - Provides detailed diagnostics for debugging

    Args:
        client: AgentClient instance
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
    client: AgentClient,
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
        client: AgentClient instance
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
        "/status": "Show system status",
        "/input-features": "Show input capabilities and navigation features",
        "/dashboard": "Show comprehensive system dashboard",
        "/workspace-info": "Show detailed workspace overview",
        "/analysis-dash": "Show analysis monitoring dashboard",
        "/progress": "Show multi-task progress monitor",
        "/files": "List workspace files",
        "/tree": "Show directory tree view",
        "/data": "Show current data summary",
        "/metadata": "Show detailed metadata information",
        # Queue commands (temporary, intent-driven)
        "/queue": "Show queue status",
        "/queue load": "Load file into queue (supports .ris, more coming)",
        "/queue list": "List queued items",
        "/queue clear": "Clear queue",
        "/queue export": "Export queue to workspace for persistence",
        # Workspace commands (persistent)
        "/workspace": "Show workspace status and information",
        "/workspace list": "List available datasets in workspace",
        "/workspace load": "Load dataset or file into workspace",
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
        "/provider": "List available LLM providers",
        "/provider list": "List available LLM providers with status",
        "/provider anthropic": "Switch to Anthropic provider",
        "/provider bedrock": "Switch to AWS Bedrock provider",
        "/provider ollama": "Switch to Ollama (local) provider",
        "/clear": "Clear screen",
        "/exit": "Exit the chat",
        # Deprecated commands
        "/load": "[DEPRECATED] Use /queue load instead",
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

    except Exception as e:
        # Fallback to static definitions if AST parsing fails
        console.print(f"[dim yellow]Command extraction fallback: {e}[/dim yellow]")

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
            """Generate command completions."""
            text_before_cursor = document.text_before_cursor.lstrip()

            # Only complete if we're typing a command (starts with /)
            if not text_before_cursor.startswith("/"):
                return

            # Extract the command being typed (from / until space or end)
            command_part = (
                text_before_cursor.split()[0]
                if " " in text_before_cursor
                else text_before_cursor
            )

            # Get available commands (cached)
            commands = self._get_cached_commands()

            # Generate completions
            for cmd, description in commands.items():
                if cmd.lower().startswith(command_part.lower()):
                    yield Completion(
                        text=cmd,
                        start_position=-len(command_part),
                        display=HTML(f"<ansired>{cmd}</ansired>"),
                        display_meta=HTML(f"<dim>{description}</dim>"),
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

            elif text.startswith("/provider "):
                # Suggest provider options (anthropic, bedrock, ollama, list)
                prefix = text.replace("/provider ", "")
                providers = ["anthropic", "bedrock", "ollama", "list"]

                for provider in providers:
                    if provider.lower().startswith(prefix.lower()):
                        # Get provider status using LLMFactory
                        from lobster.config.llm_factory import LLMFactory

                        if provider == "list":
                            meta = "Show providers with status"
                        else:
                            available = LLMFactory.get_available_providers()
                            status = "✓ configured" if provider in available else "✗ not configured"
                            meta = status

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


def change_mode(new_mode: str, current_client: AgentClient) -> AgentClient:
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
client: Optional[AgentClient] = None

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
    client: AgentClient, clear: bool = True
) -> Dict[str, Dict[str, float]]:
    timings: Dict[str, Dict[str, float]] = {}
    data_manager = getattr(client, "data_manager", None)
    if data_manager and hasattr(data_manager, "get_latest_timings"):
        dm_timings = data_manager.get_latest_timings(clear=clear)
        if dm_timings:
            timings["DataManager"] = dm_timings
    return timings


def _maybe_print_timings(client: AgentClient, context: str) -> None:
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
) -> AgentClient:
    """Initialize either local or cloud client based on environment."""
    global client

    # Check for configuration errors
    from lobster.config.settings import settings

    if hasattr(settings, "_config_error") and settings._config_error:
        console.print(settings._config_error)
        console.print(
            "\n[yellow]Tip:[/yellow] If you installed via pip, make sure to create a .env file in your current directory."
        )
        console.print(
            "[yellow]Tip:[/yellow] See README for installation instructions: https://github.com/the-omics-os/lobster-local"
        )
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
    console.print("[bold red]💻 Using Lobster Local[/bold red]")

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
        # Use simplified callback for minimal, clean output (default)
        simple_callback = SimpleTerminalCallback(console=console, show_reasoning=False)
        callbacks.append(simple_callback)

    # Initialize client with proper data_manager connection
    client = AgentClient(
        data_manager=data_manager,  # Pass the configured data_manager
        workspace_path=workspace,
        enable_reasoning=reasoning,
        # enable_langfuse=debug,
        custom_callbacks=callbacks,  # Pass the proper callback
        provider_override=provider_override,  # Pass provider override from CLI flag
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
            # Clean prompt text - remove Rich markup and emoji, keep it simple
            clean_prompt = (
                prompt_text.replace("[bold red]", "")
                .replace("[/bold red]", "")
                .replace("🦞 ", "")
            )

            # Create client-aware completer
            main_completer = ThreadedCompleter(LobsterContextualCompleter(client))

            # Custom style to match Rich orange theme
            style = Style.from_dict(
                {
                    "completion-menu.completion": "bg:#2d2d2d #ffffff",
                    "completion-menu.completion.current": "bg:#ff6600 #ffffff bold",
                    "completion-menu.meta": "bg:#2d2d2d #888888",
                    "completion-menu.meta.current": "bg:#ff6600 #ffffff",
                    "completion.command": "#ff6600",
                    "completion.file": "#00aa00",
                }
            )

            # Use prompt_toolkit with autocomplete - simple grey prompt
            user_input = prompt(
                HTML(f"<ansibrightblack>{clean_prompt}</ansibrightblack>"),
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
            clean_prompt = (
                prompt_text.replace("[bold red]", "")
                .replace("[/bold red]", "")
                .replace("🦞 ", "")
            )

            # Use prompt_toolkit without autocomplete (no client provided)
            user_input = prompt(
                HTML(f"<ansibrightblack>{clean_prompt}</ansibrightblack>"),
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


def display_welcome():
    """Display minimal welcome message with OS-like branding."""
    from lobster.core.license_manager import get_current_tier

    # Get current tier for badge
    tier = get_current_tier()
    tier_badge = f" - {tier.capitalize()}" if tier != "free" else ""

    # Create minimal welcome content
    welcome_content = f"""[bold white]Multi-Agent Bioinformatics Platform[/bold white]"""

    # Create branded welcome panel with tier badge in title
    welcome_panel = LobsterTheme.create_panel(
        welcome_content,
        title=f"🦞 Lobster OS v{__version__}{tier_badge}"
    )

    console_manager.print(welcome_panel)
    console_manager.print(
        f"\n[dim]💬 Ready. Type your analysis request or [{LobsterTheme.PRIMARY_ORANGE}]/help[/{LobsterTheme.PRIMARY_ORANGE}] for commands.[/dim]\n"
    )


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


def _format_dataframe_preview(df: pd.DataFrame, max_rows: int = 5) -> Table:
    """Format a DataFrame preview as a Rich table."""
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


def _format_array_info(arrays_dict: Dict[str, np.ndarray]) -> Table:
    """Format array information (obsm/varm) as a table."""
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


def display_status(client: AgentClient):
    """Display current system status with enhanced orange theming."""
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
    console_manager.print_status_panel(status_data, "System Status")


def _show_workspace_prompt(client):
    """Display compact session status line."""
    from lobster.config.agent_registry import AGENT_REGISTRY

    datasets = client.data_manager.available_datasets
    session_id = client.session_id

    # Count supervisor-accessible agents
    child_agent_names = set()
    for config in AGENT_REGISTRY.values():
        if config.child_agents:
            child_agent_names.update(config.child_agents)

    agent_count = sum(
        1 for name, config in AGENT_REGISTRY.items()
        if config.supervisor_accessible is not False and name not in child_agent_names
    )

    # Build compact status line
    if datasets:
        total_size = sum(d["size_mb"] for d in datasets.values())
        dataset_count = len(datasets)
        dataset_info = f"{dataset_count} dataset{'s' if dataset_count > 1 else ''} ({total_size:.1f} MB)"
    else:
        dataset_info = "no data loaded"

    # Display one-line status
    console.print(
        f"[dim]Session: {session_id} | {dataset_info} | {agent_count} agents ready[/dim]\n"
    )


def init_client_with_animation(
    workspace: Optional[Path] = None,
    reasoning: bool = False,
    verbose: bool = False,
    debug: bool = False,
    profile_timings: Optional[bool] = None,
    provider_override: Optional[str] = None,
) -> AgentClient:
    """Initialize client with minimal loading message."""
    from lobster.config.agent_registry import AGENT_REGISTRY

    get_console_manager()

    # Count supervisor-accessible agents
    child_agent_names = set()
    for config in AGENT_REGISTRY.values():
        if config.child_agents:
            child_agent_names.update(config.child_agents)

    supervisor_accessible_count = sum(
        1 for name, config in AGENT_REGISTRY.items()
        if config.supervisor_accessible is not False and name not in child_agent_names
    )

    # Single-line loading message
    with console.status(
        f"[{LobsterTheme.PRIMARY_ORANGE}]🦞 Loading {supervisor_accessible_count} agents...[/{LobsterTheme.PRIMARY_ORANGE}]"
    ):
        client = init_client(workspace, reasoning, verbose, debug, profile_timings, provider_override)

    console.print(f"[green]✓[/green] Lobster is ready\n")
    return client


@app.command()
def config_test():
    """Test API connectivity and validate configuration."""
    import datetime

    console.print()
    console.print(
        Panel.fit(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]🔍 Configuration Test[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            padding=(0, 2),
        )
    )
    console.print()

    # Check .env file exists
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        console.print(f"[red]❌ No .env file found in current directory[/red]")
        console.print(f"[dim]Run 'lobster init' to create configuration[/dim]")
        raise typer.Exit(1)

    console.print(f"[green]✅ Found .env file:[/green] {env_file}")
    console.print()

    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    # Test results
    results = []

    # Test LLM Provider
    console.print("[bold]Testing LLM Provider...[/bold]")
    try:
        from lobster.config.llm_factory import LLMFactory

        provider = LLMFactory.detect_provider()
        if provider is None:
            results.append(("LLM Provider", "❌", "No API keys found"))
            console.print("[red]❌ No LLM provider configured[/red]")
        else:
            console.print(f"[yellow]  Detected provider: {provider.value}[/yellow]")

            # Try to create a test LLM instance
            try:
                test_config = {
                    "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "temperature": 1.0,
                    "max_tokens": 100,
                }
                test_llm = LLMFactory.create_llm(test_config, "test")

                # Try a simple invoke to test connectivity
                console.print("[yellow]  Testing API connectivity...[/yellow]")
                response = test_llm.invoke("Hi")

                results.append(("LLM Provider", "✅", f"{provider.value} (connected)"))
                console.print(f"[green]✅ {provider.value} API: Connected[/green]")
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 60:
                    error_msg = error_msg[:60] + "..."
                results.append(("LLM Provider", "❌", f"{provider.value}: {error_msg}"))
                console.print(f"[red]❌ {provider.value} API: {error_msg}[/red]")
    except Exception as e:
        results.append(("LLM Provider", "❌", f"Error: {str(e)[:60]}"))
        console.print(f"[red]❌ LLM Provider test failed: {e}[/red]")

    console.print()

    # Test NCBI API (optional)
    console.print("[bold]Testing NCBI API...[/bold]")
    ncbi_key = os.environ.get("NCBI_API_KEY")
    ncbi_email = os.environ.get("NCBI_EMAIL")

    if ncbi_key or ncbi_email:
        try:
            import urllib.request
            import xml.etree.ElementTree as ET

            # Test with a simple esearch query
            base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {"db": "pubmed", "term": "cancer", "retmax": "1", "retmode": "xml"}
            if ncbi_email:
                params["email"] = ncbi_email
            if ncbi_key:
                params["api_key"] = ncbi_key

            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            url = f"{base_url}?{query_string}"

            console.print("[yellow]  Testing NCBI E-utilities...[/yellow]")
            with urllib.request.urlopen(url, timeout=10) as response:
                data = response.read()
                root = ET.fromstring(data)

                # Check for error
                error = root.find(".//ERROR")
                if error is not None:
                    results.append(("NCBI API", "❌", f"Error: {error.text}"))
                    console.print(f"[red]❌ NCBI API Error: {error.text}[/red]")
                else:
                    count = root.find(".//Count")
                    status = "with API key" if ncbi_key else "without API key"
                    results.append(("NCBI API", "✅", f"Connected ({status})"))
                    console.print(f"[green]✅ NCBI API: Connected ({status})[/green]")
                    if ncbi_key:
                        console.print("[dim]  Rate limit: 10 requests/second[/dim]")
                    else:
                        console.print(
                            "[dim]  Rate limit: 3 requests/second (add NCBI_API_KEY for higher limit)[/dim]"
                        )
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 60:
                error_msg = error_msg[:60] + "..."
            results.append(("NCBI API", "❌", error_msg))
            console.print(f"[red]❌ NCBI API: {error_msg}[/red]")
    else:
        results.append(("NCBI API", "⊘", "Not configured (optional)"))
        console.print("[dim]⊘ NCBI API: Not configured (optional)[/dim]")

    console.print()

    # Summary table
    table = Table(title="Configuration Test Summary", box=box.ROUNDED)
    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold", no_wrap=True)
    table.add_column("Details", style="dim")

    for service, status, details in results:
        status_style = (
            "green" if status == "✅" else ("red" if status == "❌" else "dim")
        )
        table.add_row(service, f"[{status_style}]{status}[/{status_style}]", details)

    console.print(table)
    console.print()

    # Final verdict
    all_required_ok = all(
        status == "✅" for service, status, _ in results if service == "LLM Provider"
    )

    if all_required_ok:
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
        raise typer.Exit(1)


@app.command()
def config_show():
    """Display current configuration with masked secrets."""
    console.print()
    console.print(
        Panel.fit(
            f"[bold {LobsterTheme.PRIMARY_ORANGE}]⚙️  Current Configuration[/bold {LobsterTheme.PRIMARY_ORANGE}]",
            border_style=LobsterTheme.PRIMARY_ORANGE,
            padding=(0, 2),
        )
    )
    console.print()

    # Check .env file
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        console.print(f"[red]❌ No .env file found in current directory[/red]")
        console.print(f"[dim]Run 'lobster init' to create configuration[/dim]")
        raise typer.Exit(1)

    console.print(f"[dim]Configuration file:[/dim] {env_file}")
    console.print()

    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv()

    def mask_secret(value: Optional[str], show_chars: int = 4) -> str:
        """Mask a secret value, showing only first few characters."""
        if not value:
            return "[dim]Not set[/dim]"
        if len(value) <= show_chars:
            return "[yellow]" + "*" * len(value) + "[/yellow]"
        return f"[yellow]{value[:show_chars]}{'*' * (len(value) - show_chars)}[/yellow]"

    # Create configuration table
    table = Table(
        title=None, box=box.SIMPLE, show_header=True, header_style="bold cyan"
    )
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    # LLM Provider
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    bedrock_access = os.environ.get("AWS_BEDROCK_ACCESS_KEY")
    bedrock_secret = os.environ.get("AWS_BEDROCK_SECRET_ACCESS_KEY")
    llm_provider = os.environ.get("LOBSTER_LLM_PROVIDER")
    ollama_model = os.environ.get("OLLAMA_DEFAULT_MODEL")

    table.add_row("", "")  # Spacing
    table.add_row("[bold]LLM Provider", "")

    # Show detected provider first
    from lobster.config.llm_factory import LLMFactory

    provider = LLMFactory.detect_provider()
    if provider:
        table.add_row("  [dim]Detected Provider[/dim]", f"[green]{provider.value}[/green]")

    # Anthropic API
    if anthropic_key:
        table.add_row("  ANTHROPIC_API_KEY", mask_secret(anthropic_key))
    else:
        table.add_row("  ANTHROPIC_API_KEY", "[dim]Not set[/dim]")

    # AWS Bedrock
    if bedrock_access or bedrock_secret:
        table.add_row("  AWS_BEDROCK_ACCESS_KEY", mask_secret(bedrock_access))
        table.add_row("  AWS_BEDROCK_SECRET_ACCESS_KEY", mask_secret(bedrock_secret))
    else:
        table.add_row("  AWS_BEDROCK_ACCESS_KEY", "[dim]Not set[/dim]")
        table.add_row("  AWS_BEDROCK_SECRET_ACCESS_KEY", "[dim]Not set[/dim]")

    # Ollama - use provider_setup module
    if llm_provider == "ollama" or provider and provider.value == "ollama":
        table.add_row("  LOBSTER_LLM_PROVIDER", "[green]ollama[/green]")

        # Get Ollama status using provider_setup
        ollama_status = provider_setup.get_ollama_status()

        if ollama_status.running:
            table.add_row("  [dim]Ollama Status[/dim]", "[green]✓ Running[/green]")
            if ollama_status.models:
                model_count = len(ollama_status.models)
                table.add_row(
                    "  [dim]Available Models[/dim]", f"[cyan]{model_count} model(s)[/cyan]"
                )
        else:
            table.add_row("  [dim]Ollama Status[/dim]", "[yellow]Not running[/yellow]")

        if ollama_model:
            table.add_row("  OLLAMA_DEFAULT_MODEL", f"[yellow]{ollama_model}[/yellow]")
        else:
            table.add_row(
                "  OLLAMA_DEFAULT_MODEL", "[dim]llama3:8b-instruct (default)[/dim]"
            )
    else:
        table.add_row("  LOBSTER_LLM_PROVIDER", "[dim]Not set (using cloud)[/dim]")

    # NCBI API (optional)
    table.add_row("", "")  # Spacing
    table.add_row("[bold]NCBI API (Optional)", "")

    ncbi_key = os.environ.get("NCBI_API_KEY")
    ncbi_email = os.environ.get("NCBI_EMAIL")

    table.add_row("  NCBI_API_KEY", mask_secret(ncbi_key))
    if ncbi_email:
        table.add_row("  NCBI_EMAIL", f"[yellow]{ncbi_email}[/yellow]")
    else:
        table.add_row("  NCBI_EMAIL", "[dim]Not set[/dim]")

    console.print(table)
    console.print()

    # Status
    from lobster.config.llm_factory import LLMFactory

    provider = LLMFactory.detect_provider()

    if provider:
        console.print(
            Panel.fit(
                f"[green]✅ Configuration looks valid[/green]\n\n"
                f"Primary provider: [bold]{provider.value}[/bold]\n"
                f"Run [bold {LobsterTheme.PRIMARY_ORANGE}]lobster config test[/bold {LobsterTheme.PRIMARY_ORANGE}] to verify connectivity",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        console.print(
            Panel.fit(
                "[red]❌ No LLM provider configured[/red]\n\n"
                "Please configure one of:\n"
                "  • Claude API (ANTHROPIC_API_KEY)\n"
                "  • AWS Bedrock (AWS_BEDROCK credentials)\n"
                "  • Ollama (local, LOBSTER_LLM_PROVIDER=ollama)\n\n"
                f"Run: [bold {LobsterTheme.PRIMARY_ORANGE}]lobster init[/bold {LobsterTheme.PRIMARY_ORANGE}]",
                border_style="red",
                padding=(1, 2),
            )
        )


@app.command()
def status():
    """Display subscription tier, installed packages, and available agents."""
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
        --anthropic-key=sk-ant-xxx                   # CI/CD: Claude API
      lobster init --non-interactive \\
        --bedrock-access-key=AKIA... \\
        --bedrock-secret-key=xxx                     # CI/CD: AWS Bedrock
      lobster init --non-interactive \\
        --use-ollama                                 # CI/CD: Ollama (local)
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

        # Validate provider configuration using provider_setup module
        has_anthropic = anthropic_key is not None
        has_bedrock = bedrock_access_key is not None and bedrock_secret_key is not None
        has_ollama = use_ollama

        # Validate at least one provider
        valid, error_msg = provider_setup.validate_provider_choice(
            has_anthropic, has_bedrock, has_ollama
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
            raise typer.Exit(1)

        # Warn if multiple providers
        priority_warning = provider_setup.get_provider_priority_warning(
            has_anthropic, has_bedrock, has_ollama
        )
        if priority_warning:
            console.print(f"[yellow]⚠️  Warning: {priority_warning}[/yellow]")

        # Create provider configuration
        if has_anthropic:
            config = provider_setup.create_anthropic_config(anthropic_key)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
        elif has_bedrock:
            config = provider_setup.create_bedrock_config(
                bedrock_access_key, bedrock_secret_key
            )
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
        elif has_ollama:
            config = provider_setup.create_ollama_config(model_name=ollama_model)
            if config.success:
                for key, value in config.env_vars.items():
                    env_lines.append(f"{key}={value}")
                console.print("[green]✓ Ollama provider configured[/green]")

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

        # Write .env file
        try:
            with open(env_path, "w") as f:
                f.write("\n".join(env_lines))
                f.write("\n")
        except Exception as e:
            console.print(f"[red]❌ Failed to write .env file: {str(e)}[/red]")
            raise typer.Exit(1)

        # Success message for non-interactive
        console.print(f"[green]✅ Configuration saved: {env_path}[/green]")
        if backup_path:
            console.print(f"[green]📦 Backup created: {backup_path}[/green]")
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
        console.print()

        provider = Prompt.ask(
            "[bold white]Choose provider[/bold white]", choices=["1", "2", "3"], default="1"
        )

        env_lines = []
        env_lines.append("# Lobster AI Configuration")
        env_lines.append("# Generated by lobster init\n")

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

        else:  # provider == "3"
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
                    else:
                        config = provider_setup.create_ollama_config()

                    for key, value in config.env_vars.items():
                        env_lines.append(f"{key}={value}")

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

                console.print("[green]✓ Ollama provider configured[/green]")

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
        console.print("\n[bold white]⭐ Premium Features (Optional)[/bold white]")
        console.print("Unlock advanced agents and cloud processing capabilities.")
        console.print("Options:")
        console.print("  [cyan]1[/cyan] - Skip (stay on Free tier)")
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

        # Write .env file
        with open(env_path, "w") as f:
            f.write("\n".join(env_lines))
            f.write("\n")

        console.print()
        success_message = f"[bold green]✅ Configuration saved![/bold green]\n\n"
        success_message += f"File: [cyan]{env_path}[/cyan]\n"
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
    reasoning: Optional[bool] = typer.Option(
        None,
        "--reasoning",
        hidden=True,
        help="[DEPRECATED] Reasoning is now enabled by default. Use --no-reasoning to disable.",
    ),
    no_reasoning: bool = typer.Option(
        False,
        "--no-reasoning",
        is_flag=True,
        help="Disable agent reasoning display (enabled by default)",
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
):
    """
    Start an interactive chat session with the multi-agent system.

    Agent reasoning is shown by default. Use --no-reasoning to disable.
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
        console.print(
            Panel.fit(
                "[bold red]⚠️  No Configuration Found[/bold red]\n\n"
                "Lobster requires API keys to function. Please run the setup wizard:\n\n"
                f"[bold {LobsterTheme.PRIMARY_ORANGE}]lobster init[/bold {LobsterTheme.PRIMARY_ORANGE}]\n\n"
                "This will guide you through API key configuration.",
                border_style="red",
                padding=(1, 2),
            )
        )
        console.print()
        console.print("[dim]For manual configuration, see:[/dim]")
        console.print(
            "[link]https://github.com/the-omics-os/lobster-local/wiki/03-configuration[/link]"
        )
        raise typer.Exit(1)

    display_welcome()

    # Initialize client with animated loading sequence
    try:
        client = init_client_with_animation(
            workspace, not no_reasoning, verbose, debug, profile_timings
        )
    except Exception as e:
        console_manager.print_error_panel(
            f"Failed to initialize Lobster: {str(e)}",
            "Check your configuration and try again",
        )
        raise

    # Show compact session status
    _show_workspace_prompt(client)

    while True:
        try:
            # Get user input with arrow key navigation support
            current_path = (
                str(current_directory.name) if current_directory != Path.home() else "~"
            )
            if current_directory == Path.cwd():
                current_path = str(current_directory.name)
            user_input = get_user_input_with_editing(
                f"\n[bold red]🦞 {current_path}[/bold red] ", client
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

            # BUG FIX #7: Remove fake progress bar overhead (saves ~100ms per query)
            # Process query with simple status message instead of spinner context manager
            if should_show_progress(client):
                # Normal mode: show simple processing message
                console_manager.print(
                    f"[dim cyan]🦞 Processing: {user_input[:50]}{'...' if len(user_input) > 50 else ''}[/dim cyan]",
                    end="",
                    flush=True,
                )

            # Single code path - no duplication
            result = client.query(user_input, stream=False)

            if should_show_progress(client):
                # Clear the status line after query completes
                console_manager.print("\r" + " " * 100 + "\r", end="", flush=True)

            # Display response with enhanced theming
            if result["success"]:
                # Show which agent provided the response if available
                agent_name = result.get("last_agent", "supervisor")
                if agent_name and agent_name != "__end__":
                    agent_display = agent_name.replace("_", " ").title()
                    title = f"🦞 {agent_display} Response"
                else:
                    title = "🦞 Lobster Response"

                response_panel = LobsterTheme.create_panel(
                    Markdown(result["response"]), title=title
                )
                console_manager.print(response_panel)

                # Show token usage and cost if available
                if result.get("token_usage"):
                    token_info = result["token_usage"]
                    latest_cost = token_info.get("latest_cost_usd", 0.0)
                    session_total = token_info.get("session_total_usd", 0.0)
                    total_tokens = token_info.get("total_tokens", 0)

                    # Format cost display
                    cost_display = f"💰 Session cost: ${session_total:.4f}"
                    if latest_cost > 0:
                        cost_display += f" (+${latest_cost:.4f} this response)"
                    cost_display += f" | Total tokens: {total_tokens:,}"

                    console_manager.print(f"[dim grey50]{cost_display}[/dim grey50]")

                # Show any generated plots with orange styling
                if result.get("plots"):
                    plot_text = f"📊 Generated {len(result['plots'])} visualization(s)"
                    console_manager.print(
                        f"[{LobsterTheme.PRIMARY_ORANGE}]{plot_text}[/{LobsterTheme.PRIMARY_ORANGE}]"
                    )
            else:
                console_manager.print_error_panel(result["error"])

            _maybe_print_timings(client, "Chat Query")

        except KeyboardInterrupt:
            if Confirm.ask(
                f"\n[{LobsterTheme.PRIMARY_ORANGE}]🦞 Exit Lobster?[/{LobsterTheme.PRIMARY_ORANGE}]"
            ):
                # Get final token usage
                try:
                    token_usage = client.get_token_usage()
                    if token_usage and "error" not in token_usage:
                        total_cost = token_usage.get("total_cost_usd", 0.0)
                        total_tokens = token_usage.get("total_tokens", 0)
                        cost_info = f"\n[bold white]💰 Session Summary:[/bold white]\nTotal tokens used: {total_tokens:,}\nTotal cost: ${total_cost:.4f}\n"
                    else:
                        cost_info = ""
                except Exception:
                    cost_info = ""

                goodbye_message = f"""👋 Thank you for using Lobster by Omics-OS!
{cost_info}
[bold white]🌟 Help us improve Lobster![/bold white]
Your feedback matters! Please take 1 minute to share your experience:

[bold {LobsterTheme.PRIMARY_ORANGE}]📝 Quick Survey:[/bold {LobsterTheme.PRIMARY_ORANGE}] [link=https://forms.cloud.microsoft/e/AkNk8J8nE8]https://forms.cloud.microsoft/e/AkNk8J8nE8[/link]

[dim grey50]Happy analyzing! 🧬🦞[/dim grey50]"""

                exit_panel = LobsterTheme.create_panel(
                    goodbye_message, title="🦞 Goodbye & Thank You!"
                )
                console_manager.print(exit_panel)
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


def handle_command(command: str, client: AgentClient):
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
# Queue Command Helpers
# ============================================================================


def _show_queue_status(client: AgentClient, console: Console) -> Optional[str]:
    """Display status of the publication queue.

    Returns:
        Summary string for conversation history, or None.
    """
    if client.publication_queue is None:
        console.print("[yellow]Publication queue not initialized[/yellow]")
        return None

    stats = client.publication_queue.get_statistics()

    console.print("\n[bold cyan]📋 Queue Status[/bold cyan]\n")

    # Create status table
    table = Table(box=box.ROUNDED)
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="white", justify="right")

    by_status = stats.get("by_status", {})
    total_entries = stats.get("total_entries", 0)

    table.add_row("Pending", str(by_status.get("pending", 0)))
    table.add_row("Extracting", str(by_status.get("extracting", 0)))
    table.add_row("Completed", str(by_status.get("completed", 0)))
    table.add_row("Failed", str(by_status.get("failed", 0)))
    table.add_row("[bold]Total[/bold]", f"[bold]{total_entries}[/bold]")

    console.print(table)

    console.print("\n[cyan]💡 Commands:[/cyan]")
    console.print("  • [white]/queue load <file>[/white] - Load file into queue")
    console.print("  • [white]/queue list[/white] - List queued items")
    console.print("  • [white]/queue clear[/white] - Clear the queue")

    return f"Queue status: {total_entries} total items"


def _queue_load_file(
    client: AgentClient,
    filename: str,
    console: Console,
    current_directory: Path,
) -> Optional[str]:
    """Load file into queue - type determines handler, user determines intent.

    Args:
        client: The AgentClient instance
        filename: File path to load
        console: Rich console for output
        current_directory: Current working directory

    Returns:
        Summary string for conversation history, or None

    Raises:
        QueueFileTypeNotSupported: For unsupported file types
    """
    if not filename:
        console.print("[yellow]Usage: /queue load <file>[/yellow]")
        return None

    # BUG FIX #6: Use PathResolver for secure path resolution
    resolver = PathResolver(
        current_directory=current_directory,
        workspace_path=(
            client.data_manager.workspace_path
            if hasattr(client, "data_manager")
            else None
        ),
    )
    resolved = resolver.resolve(filename, search_workspace=True, must_exist=True)

    if not resolved.is_safe:
        console.print(f"[red]❌ Security error: {resolved.error}[/red]")
        return None

    if not resolved.exists:
        console.print(f"[red]❌ File not found: {filename}[/red]")
        return None

    file_path = resolved.path

    ext = file_path.suffix.lower()

    # Supported: .ris files
    if ext in [".ris", ".txt"]:
        console.print(f"[cyan]📚 Loading into queue: {file_path.name}[/cyan]\n")

        try:
            result = client.load_publication_list(
                file_path=str(file_path),
                priority=5,
                schema_type="general",
                extraction_level="methods",
            )

            if result["added_count"] > 0:
                console.print(
                    f"[green]✅ Loaded {result['added_count']} items into queue[/green]\n"
                )

                if result["skipped_count"] > 0:
                    console.print(
                        f"[yellow]⚠️  Skipped {result['skipped_count']} malformed entries[/yellow]"
                    )

                console.print(
                    "\n[bold cyan]What would you like to do with these publications?[/bold cyan]"
                )
                console.print("  • Extract methods and parameters")
                console.print("  • Search for related datasets (GEO)")
                console.print("  • Build citation network")
                console.print("  • Custom analysis (describe your intent)\n")

                return f"Loaded {result['added_count']} publications into queue from {file_path.name}. Awaiting user intent."
            else:
                console.print("[red]❌ No items could be loaded from file[/red]")
                if result.get("errors"):
                    for error in result["errors"][:3]:
                        console.print(f"  • {error}")
                return None

        except Exception as e:
            console.print(f"[red]❌ Failed to load file: {str(e)}[/red]")
            return None

    # Placeholder: .bib files (BibTeX)
    elif ext == ".bib":
        raise QueueFileTypeNotSupported(
            "BibTeX (.bib) support coming soon. "
            "Convert to .ris format or wait for future release."
        )

    # Placeholder: .csv files (custom lists)
    elif ext == ".csv":
        raise QueueFileTypeNotSupported(
            "CSV queue loading coming soon. "
            "Expected format: columns for DOI, PMID, or title."
        )

    # Placeholder: .json files (API exports)
    elif ext == ".json":
        raise QueueFileTypeNotSupported(
            "JSON queue loading coming soon. " "Planned support for PubMed API exports."
        )

    # Unknown type
    else:
        raise QueueFileTypeNotSupported(
            f"Unsupported file type: {ext}. "
            f"Currently supported: .ris. Coming soon: .bib, .csv, .json"
        )


def _queue_list(client: AgentClient, console: Console) -> Optional[str]:
    """List items in the publication queue.

    Returns:
        Summary string for conversation history, or None.
    """
    if client.publication_queue is None:
        console.print("[yellow]Publication queue not initialized[/yellow]")
        return None

    entries = client.publication_queue.list_entries()

    if not entries:
        console.print("[yellow]Queue is empty[/yellow]")
        return "Queue is empty"

    # Limit display to first 20 entries
    display_entries = entries[:20]
    total_count = len(entries)

    console.print(
        f"\n[bold cyan]📋 Queue Items ({len(display_entries)} of {total_count} shown)[/bold cyan]\n"
    )

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", style="white", max_width=50, overflow="ellipsis")
    table.add_column("Year", style="cyan", width=6)
    table.add_column("Status", style="yellow", width=12)
    table.add_column("PMID/DOI", style="dim", width=20)

    for i, entry in enumerate(display_entries, 1):
        title = (
            entry.title[:47] + "..."
            if entry.title and len(entry.title) > 50
            else (entry.title or "N/A")
        )
        year = str(entry.year) if entry.year else "N/A"
        status = (
            entry.status.value if hasattr(entry.status, "value") else str(entry.status)
        )
        identifier = entry.pmid or entry.doi or "N/A"

        table.add_row(str(i), title, year, status, identifier)

    console.print(table)

    if total_count > 20:
        console.print(f"\n[dim]... and {total_count - 20} more items[/dim]")

    return f"Listed {len(display_entries)} of {total_count} items from queue"


def _queue_clear(client: AgentClient, console: Console) -> Optional[str]:
    """Clear all items from the publication queue.

    Returns:
        Summary string for conversation history, or None.
    """
    if client.publication_queue is None:
        console.print("[yellow]Publication queue not initialized[/yellow]")
        return None

    # Get count before clearing
    stats = client.publication_queue.get_statistics()
    total = stats.get("total_entries", 0)

    if total == 0:
        console.print("[yellow]Queue is already empty[/yellow]")
        return "Queue was already empty"

    # Confirm with user
    confirm = Confirm.ask(f"[yellow]Clear all {total} items from queue?[/yellow]")

    if confirm:
        client.publication_queue.clear_queue()
        console.print(f"[green]✅ Cleared {total} items from queue[/green]")
        return f"Cleared {total} items from queue"
    else:
        console.print("[cyan]Operation cancelled[/cyan]")
        return None


def _queue_export(
    client: AgentClient, name: Optional[str], console: Console
) -> Optional[str]:
    """Export queue to workspace for persistence.

    Args:
        client: The AgentClient instance
        name: Optional name for the exported dataset
        console: Rich console for output

    Returns:
        Summary string for conversation history, or None.
    """
    if client.publication_queue is None:
        console.print("[yellow]Publication queue not initialized[/yellow]")
        return None

    stats = client.publication_queue.get_statistics()
    if stats.get("total_entries", 0) == 0:
        console.print("[yellow]Queue is empty, nothing to export[/yellow]")
        return None

    # Generate export name if not provided
    if not name:
        from datetime import datetime

        name = f"queue_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    console.print(f"[cyan]📦 Exporting queue to workspace as '{name}'...[/cyan]")

    try:
        # Export queue data by copying the queue file to workspace
        source_path = client.publication_queue.queue_file
        export_path = client.data_manager.workspace_path / f"{name}.jsonl"

        # Copy the queue file
        shutil.copy2(source_path, export_path)

        console.print(
            f"[green]✅ Exported {stats.get('total', 0)} items to: {export_path}[/green]"
        )
        return f"Exported {stats.get('total', 0)} queue items to workspace as '{name}'"
    except Exception as e:
        console.print(f"[red]❌ Export failed: {str(e)}[/red]")
        return None


def _execute_command(cmd: str, client: AgentClient) -> Optional[str]:
    """Execute individual slash commands.

    Returns:
        Optional[str]: Summary of command execution for conversation history,
                      or None if command should not be logged to history.
    """

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
        help_text = f"""[bold white]Available Commands:[/bold white]

[{LobsterTheme.PRIMARY_ORANGE}]/help[/{LobsterTheme.PRIMARY_ORANGE}]         [grey50]-[/grey50] Show this help message
[{LobsterTheme.PRIMARY_ORANGE}]/status[/{LobsterTheme.PRIMARY_ORANGE}]       [grey50]-[/grey50] Show system status
[{LobsterTheme.PRIMARY_ORANGE}]/tokens[/{LobsterTheme.PRIMARY_ORANGE}]       [grey50]-[/grey50] Show token usage and cost for this session
[{LobsterTheme.PRIMARY_ORANGE}]/input-features[/{LobsterTheme.PRIMARY_ORANGE}] [grey50]-[/grey50] Show input capabilities and navigation features
[{LobsterTheme.PRIMARY_ORANGE}]/dashboard[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] Show comprehensive system dashboard
[{LobsterTheme.PRIMARY_ORANGE}]/workspace-info[/{LobsterTheme.PRIMARY_ORANGE}] [grey50]-[/grey50] Show detailed workspace overview
[{LobsterTheme.PRIMARY_ORANGE}]/analysis-dash[/{LobsterTheme.PRIMARY_ORANGE}] [grey50]-[/grey50] Show analysis monitoring dashboard
[{LobsterTheme.PRIMARY_ORANGE}]/progress[/{LobsterTheme.PRIMARY_ORANGE}]      [grey50]-[/grey50] Show multi-task progress monitor
[{LobsterTheme.PRIMARY_ORANGE}]/files[/{LobsterTheme.PRIMARY_ORANGE}]        [grey50]-[/grey50] List workspace files
[{LobsterTheme.PRIMARY_ORANGE}]/tree[/{LobsterTheme.PRIMARY_ORANGE}]         [grey50]-[/grey50] Show directory tree view
[{LobsterTheme.PRIMARY_ORANGE}]/data[/{LobsterTheme.PRIMARY_ORANGE}]         [grey50]-[/grey50] Show current data summary
[{LobsterTheme.PRIMARY_ORANGE}]/metadata[/{LobsterTheme.PRIMARY_ORANGE}]     [grey50]-[/grey50] Show detailed metadata information
[{LobsterTheme.PRIMARY_ORANGE}]/workspace[/{LobsterTheme.PRIMARY_ORANGE}]    [grey50]-[/grey50] Show workspace status and information
[{LobsterTheme.PRIMARY_ORANGE}]/workspace list[/{LobsterTheme.PRIMARY_ORANGE}] [grey50]-[/grey50] List available datasets in workspace
[{LobsterTheme.PRIMARY_ORANGE}]/restore[/{LobsterTheme.PRIMARY_ORANGE}]      [grey50]-[/grey50] Restore previous session datasets
[{LobsterTheme.PRIMARY_ORANGE}]/restore <pattern>[/{LobsterTheme.PRIMARY_ORANGE}] [grey50]-[/grey50] Restore datasets matching pattern (recent/all/*)
[{LobsterTheme.PRIMARY_ORANGE}]/modalities[/{LobsterTheme.PRIMARY_ORANGE}]   [grey50]-[/grey50] Show detailed modality information
[{LobsterTheme.PRIMARY_ORANGE}]/describe <name>[/{LobsterTheme.PRIMARY_ORANGE}] [grey50]-[/grey50] Show comprehensive details about a specific modality
[{LobsterTheme.PRIMARY_ORANGE}]/plots[/{LobsterTheme.PRIMARY_ORANGE}]        [grey50]-[/grey50] List all generated plots
[{LobsterTheme.PRIMARY_ORANGE}]/plot[/{LobsterTheme.PRIMARY_ORANGE}]         [grey50]-[/grey50] Open plots directory in file manager
[{LobsterTheme.PRIMARY_ORANGE}]/plot[/{LobsterTheme.PRIMARY_ORANGE}] <ID>    [grey50]-[/grey50] Open a specific plot by ID or name
[{LobsterTheme.PRIMARY_ORANGE}]/open[/{LobsterTheme.PRIMARY_ORANGE}] <file>  [grey50]-[/grey50] Open file or folder in system default application
[{LobsterTheme.PRIMARY_ORANGE}]/save[/{LobsterTheme.PRIMARY_ORANGE}]         [grey50]-[/grey50] Save current state to workspace
[{LobsterTheme.PRIMARY_ORANGE}]/read[/{LobsterTheme.PRIMARY_ORANGE}] <file>  [grey50]-[/grey50] Read a file from workspace (supports glob patterns like *.h5ad)
[{LobsterTheme.PRIMARY_ORANGE}]/load[/{LobsterTheme.PRIMARY_ORANGE}] <file>  [grey50]-[/grey50] Load .ris publication list into queue for processing
[{LobsterTheme.PRIMARY_ORANGE}]/export[/{LobsterTheme.PRIMARY_ORANGE}]       [grey50]-[/grey50] Export session data
[{LobsterTheme.PRIMARY_ORANGE}]/pipeline export[/{LobsterTheme.PRIMARY_ORANGE}] [grey50]-[/grey50] Export session as Jupyter notebook
[{LobsterTheme.PRIMARY_ORANGE}]/pipeline list[/{LobsterTheme.PRIMARY_ORANGE}] [grey50]-[/grey50] List available notebooks
[{LobsterTheme.PRIMARY_ORANGE}]/pipeline run[/{LobsterTheme.PRIMARY_ORANGE}] [grey50]-[/grey50] Execute saved notebook with new data
[{LobsterTheme.PRIMARY_ORANGE}]/pipeline info[/{LobsterTheme.PRIMARY_ORANGE}] [grey50]-[/grey50] Show notebook details
[{LobsterTheme.PRIMARY_ORANGE}]/reset[/{LobsterTheme.PRIMARY_ORANGE}]        [grey50]-[/grey50] Reset conversation
[{LobsterTheme.PRIMARY_ORANGE}]/mode[/{LobsterTheme.PRIMARY_ORANGE}] <name>  [grey50]-[/grey50] Change operation mode
[{LobsterTheme.PRIMARY_ORANGE}]/modes[/{LobsterTheme.PRIMARY_ORANGE}]        [grey50]-[/grey50] List available modes
[{LobsterTheme.PRIMARY_ORANGE}]/provider[/{LobsterTheme.PRIMARY_ORANGE}]     [grey50]-[/grey50] List available LLM providers
[{LobsterTheme.PRIMARY_ORANGE}]/provider[/{LobsterTheme.PRIMARY_ORANGE}] <name> [grey50]-[/grey50] Switch LLM provider (anthropic, bedrock, ollama)
[{LobsterTheme.PRIMARY_ORANGE}]/clear[/{LobsterTheme.PRIMARY_ORANGE}]        [grey50]-[/grey50] Clear screen
[{LobsterTheme.PRIMARY_ORANGE}]/exit[/{LobsterTheme.PRIMARY_ORANGE}]         [grey50]-[/grey50] Exit the chat

[bold white]File Loading Examples:[/bold white]

[{LobsterTheme.PRIMARY_ORANGE}]/read[/{LobsterTheme.PRIMARY_ORANGE}] data.h5ad      [grey50]-[/grey50] Load single file
[{LobsterTheme.PRIMARY_ORANGE}]/read[/{LobsterTheme.PRIMARY_ORANGE}] *.h5ad         [grey50]-[/grey50] Load all .h5ad files in current directory
[{LobsterTheme.PRIMARY_ORANGE}]/read[/{LobsterTheme.PRIMARY_ORANGE}] data/*.csv     [grey50]-[/grey50] Load all .csv files in data/ directory
[{LobsterTheme.PRIMARY_ORANGE}]/read[/{LobsterTheme.PRIMARY_ORANGE}] sample_*.h5ad  [grey50]-[/grey50] Load files matching pattern

[bold white]Shell Commands:[/bold white] [grey50](execute directly without /)[/grey50]

[{LobsterTheme.PRIMARY_ORANGE}]cd[/{LobsterTheme.PRIMARY_ORANGE}] <path>      [grey50]-[/grey50] Change directory
[{LobsterTheme.PRIMARY_ORANGE}]pwd[/{LobsterTheme.PRIMARY_ORANGE}]            [grey50]-[/grey50] Print current directory
[{LobsterTheme.PRIMARY_ORANGE}]ls[/{LobsterTheme.PRIMARY_ORANGE}] [path]      [grey50]-[/grey50] List directory contents
[{LobsterTheme.PRIMARY_ORANGE}]open[/{LobsterTheme.PRIMARY_ORANGE}] <file>    [grey50]-[/grey50] Open file or folder in system default application
[{LobsterTheme.PRIMARY_ORANGE}]mkdir[/{LobsterTheme.PRIMARY_ORANGE}] <dir>    [grey50]-[/grey50] Create directory
[{LobsterTheme.PRIMARY_ORANGE}]touch[/{LobsterTheme.PRIMARY_ORANGE}] <file>   [grey50]-[/grey50] Create file
[{LobsterTheme.PRIMARY_ORANGE}]cp[/{LobsterTheme.PRIMARY_ORANGE}] <src> <dst> [grey50]-[/grey50] Copy file/directory
[{LobsterTheme.PRIMARY_ORANGE}]mv[/{LobsterTheme.PRIMARY_ORANGE}] <src> <dst> [grey50]-[/grey50] Move/rename file/directory
[{LobsterTheme.PRIMARY_ORANGE}]rm[/{LobsterTheme.PRIMARY_ORANGE}] <file>      [grey50]-[/grey50] Remove file
[{LobsterTheme.PRIMARY_ORANGE}]cat[/{LobsterTheme.PRIMARY_ORANGE}] <file>     [grey50]-[/grey50] Display file contents"""

        help_panel = LobsterTheme.create_panel(help_text, title="🦞 Help Menu")
        console_manager.print(help_panel)

    elif cmd == "/status":
        display_status(client)

    elif cmd == "/tokens":
        # Display token usage and cost information
        try:
            token_usage = client.get_token_usage()

            if not token_usage or "error" in token_usage:
                console_manager.print(
                    "[yellow]Token tracking not available for this client type[/yellow]"
                )
                return

            # Create summary table
            summary_table = Table(
                title="💰 Session Token Usage & Cost", box=box.ROUNDED
            )
            summary_table.add_column("Metric", style="cyan", no_wrap=True)
            summary_table.add_column("Value", style="green")

            summary_table.add_row("Session ID", token_usage["session_id"])
            summary_table.add_row(
                "Total Input Tokens", f"{token_usage['total_input_tokens']:,}"
            )
            summary_table.add_row(
                "Total Output Tokens", f"{token_usage['total_output_tokens']:,}"
            )
            summary_table.add_row("Total Tokens", f"{token_usage['total_tokens']:,}")
            summary_table.add_row(
                "Total Cost (USD)", f"${token_usage['total_cost_usd']:.4f}"
            )

            console_manager.print(summary_table)

            # Create per-agent breakdown table if agents have been used
            if token_usage.get("by_agent"):
                agent_table = Table(title="📊 Cost by Agent", box=box.ROUNDED)
                agent_table.add_column("Agent", style="cyan")
                agent_table.add_column("Input", style="blue", justify="right")
                agent_table.add_column("Output", style="magenta", justify="right")
                agent_table.add_column("Total", style="yellow", justify="right")
                agent_table.add_column("Cost (USD)", style="green", justify="right")
                agent_table.add_column("Calls", style="grey50", justify="right")

                for agent_name, stats in token_usage["by_agent"].items():
                    agent_display = agent_name.replace("_", " ").title()
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
        # Show comprehensive system health dashboard
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
                    title="🦞 Dashboard",
                )
            )
            console_manager.print(core_panel)
            console_manager.print(resource_panel)
            console_manager.print(agent_panel)

        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to create system dashboard: {e}",
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

            workspace_path = resolve_workspace(explicit_path=workspace, create=False)
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

    # =========================================================================
    # /read - File Inspection Only (no state change, view-only)
    # =========================================================================
    elif cmd.startswith("/read "):
        filename = cmd[6:].strip()

        if not filename:
            console.print("[yellow]Usage: /read <file|pattern>[/yellow]")
            console.print("[grey50]  View file contents (text files only)[/grey50]")
            console.print(
                "[grey50]  Use /workspace load <file> to load data files[/grey50]"
            )
            return None

        # Check if filename contains glob patterns (before path resolution)
        import glob as glob_module

        is_glob_pattern = any(char in filename for char in ["*", "?", "[", "]"])

        # BUG FIX #6: Use PathResolver for secure path resolution (non-glob paths)
        if not is_glob_pattern:
            resolver = PathResolver(
                current_directory=current_directory,
                workspace_path=(
                    client.data_manager.workspace_path
                    if hasattr(client, "data_manager")
                    else None
                ),
            )
            resolved = resolver.resolve(
                filename, search_workspace=True, must_exist=False
            )

            if not resolved.is_safe:
                console.print(f"[red]❌ Security error: {resolved.error}[/red]")
                return None

            file_path = resolved.path
        else:
            # Glob patterns need special handling - construct search pattern
            file_path = (
                Path(filename)
                if Path(filename).is_absolute()
                else current_directory / filename
            )

        if is_glob_pattern:
            # Handle glob patterns - show contents of matching text files
            if not Path(filename).is_absolute():
                base_path = current_directory
                search_pattern = str(base_path / filename)
            else:
                search_pattern = filename

            # BUG FIX #3: Use lazy evaluation to prevent memory explosion
            # Only load first 10 file paths instead of all matches
            import itertools

            matching_files = list(
                itertools.islice(glob_module.iglob(search_pattern), 10)
            )

            if not matching_files:
                console_manager.print_error_panel(
                    f"No files found matching pattern: {filename}",
                    f"Searched in: {current_directory}",
                )
                return None

            # Count total matches without loading all paths
            total_count = sum(1 for _ in glob_module.iglob(search_pattern))

            matching_files.sort()
            console.print(
                f"[cyan]📁 Found {total_count} files matching '[white]{filename}[/white]', displaying first 10[/cyan]\n"
            )

            displayed_count = 0
            for match_path in matching_files:  # Already limited to 10
                match_file = Path(match_path)
                file_info = client.detect_file_type(match_file)

                # Only display text files
                if not file_info.get("binary", True):
                    try:
                        # BUG FIX #3: Add file size check before reading (10MB limit)
                        file_size = match_file.stat().st_size
                        if file_size > 10_000_000:  # 10MB
                            console.print(
                                f"[yellow]⚠️  {match_file.name} too large to display ({file_size / 1_000_000:.1f}MB, limit: 10MB)[/yellow]"
                            )
                            continue

                        content = match_file.read_text(encoding="utf-8")
                        lines = content.splitlines()

                        # Language detection
                        ext = match_file.suffix.lower()
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
                            ".md": "markdown",
                            ".txt": "text",
                            ".log": "text",
                            ".r": "r",
                            ".csv": "csv",
                            ".tsv": "csv",
                            ".ris": "text",
                        }
                        language = language_map.get(ext, "text")

                        syntax = Syntax(
                            content, language, theme="monokai", line_numbers=True
                        )
                        console.print(
                            Panel(
                                syntax,
                                title=f"[bold white] 📄 {match_file.name} [/bold white]",
                                subtitle=f"[grey50]{len(lines)} lines[/grey50]",
                                border_style="cyan",
                                box=box.ROUNDED,
                            )
                        )
                        displayed_count += 1
                    except Exception as e:
                        console.print(
                            f"[yellow]⚠️  Could not read {match_file.name}: {e}[/yellow]"
                        )
                else:
                    console.print(
                        f"[grey50]  • {match_file.name} (binary file - skipped)[/grey50]"
                    )

            if total_count > 10:
                console.print(
                    f"\n[grey50]... and {total_count - 10} more files (not loaded)[/grey50]"
                )

            return f"Displayed {displayed_count} text files matching '{filename}' (total: {total_count})"

        # Single file processing
        if not file_path.exists():
            # Try to locate via client (searches workspace directories)
            file_info = client.locate_file(filename)
            if not file_info["found"]:
                console.print(
                    f"[bold red on white] ⚠️  Error [/bold red on white] [red]{file_info['error']}[/red]"
                )
                if "searched_paths" in file_info:
                    console.print("[grey50]Searched in:[/grey50]")
                    for path in file_info["searched_paths"][:5]:
                        console.print(f"  • [grey50]{path}[/grey50]")
                return f"File '{filename}' not found"
            file_path = file_info["path"]

        # Get file info
        file_info = client.detect_file_type(file_path)
        file_description = file_info.get("description", "Unknown")
        file_category = file_info.get("category", "unknown")
        is_binary = file_info.get("binary", True)

        # Show file location
        console.print(f"[cyan]📄 File:[/cyan] [white]{file_path.name}[/white]")
        console.print(f"[grey50]   Path: {file_path}[/grey50]")
        console.print(f"[grey50]   Type: {file_description}[/grey50]")

        # Handle text files - display content
        if not is_binary:
            try:
                content = file_path.read_text(encoding="utf-8")
                lines = content.splitlines()

                # Language detection
                ext = file_path.suffix.lower()
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
                    ".md": "markdown",
                    ".txt": "text",
                    ".log": "text",
                    ".r": "r",
                    ".csv": "csv",
                    ".tsv": "csv",
                    ".ris": "text",
                }
                language = language_map.get(ext, "text")

                syntax = Syntax(content, language, theme="monokai", line_numbers=True)
                console.print(
                    Panel(
                        syntax,
                        title=f"[bold white on red] 📄 {file_path.name} [/bold white on red]",
                        border_style="red",
                        box=box.DOUBLE,
                    )
                )

                return f"Displayed text file '{filename}' ({file_description}, {len(lines)} lines)"

            except UnicodeDecodeError:
                console.print(
                    "[yellow]⚠️  File appears to be binary despite extension[/yellow]"
                )
                is_binary = True
            except Exception as e:
                console.print(f"[red]Error reading file: {e}[/red]")
                return f"Error reading file '{filename}': {str(e)}"

        # Handle binary/data files - show info only, suggest /workspace load
        if is_binary:
            console.print(
                "\n[bold yellow on black] ℹ️  File Info [/bold yellow on black]"
            )

            # Format file size
            size_bytes = file_path.stat().st_size
            if size_bytes < 1024:
                size_str = f"{size_bytes} bytes"
            elif size_bytes < 1024**2:
                size_str = f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                size_str = f"{size_bytes/1024**2:.1f} MB"
            else:
                size_str = f"{size_bytes/1024**3:.1f} GB"

            console.print(f"[white]Size: [yellow]{size_str}[/yellow][/white]")

            # Provide guidance based on file type
            if file_category == "bioinformatics":
                console.print(
                    f"\n[cyan]💡 This is a bioinformatics data file ({file_description}).[/cyan]"
                )
                console.print(
                    f"[cyan]   To load into workspace: [yellow]/workspace load {filename}[/yellow][/cyan]"
                )
            elif file_category == "tabular":
                console.print(
                    f"\n[cyan]💡 This is a tabular data file ({file_description}).[/cyan]"
                )
                console.print(
                    f"[cyan]   To load into workspace: [yellow]/workspace load {filename}[/yellow][/cyan]"
                )
            elif file_category == "archive":
                console.print(
                    f"\n[cyan]💡 This is an archive file ({file_description}).[/cyan]"
                )
                console.print(
                    f"[cyan]   To extract and load: [yellow]/workspace load {filename}[/yellow][/cyan]"
                )
            elif file_category == "image":
                console.print(
                    "[cyan]💡 This is an image file. Use your system's image viewer to open it.[/cyan]"
                )
            else:
                console.print(
                    "[cyan]💡 Binary file - use external tools to view.[/cyan]"
                )

            return f"Inspected file '{filename}' ({file_description}, {size_str}) - use /workspace load to load data files"

    # =========================================================================
    # /queue - Flexible Queue Operations (session-based, user determines intent)
    # =========================================================================
    elif cmd.startswith("/queue"):
        parts = cmd.split()

        if len(parts) == 1:
            # Show queue status
            return _show_queue_status(client, console)

        subcommand = parts[1] if len(parts) > 1 else None

        if subcommand == "load":
            filename = parts[2] if len(parts) > 2 else None
            try:
                return _queue_load_file(client, filename, console, current_directory)
            except QueueFileTypeNotSupported as e:
                console.print(f"[yellow]⚠️  {str(e)}[/yellow]")
                return None

        elif subcommand == "list":
            return _queue_list(client, console)

        elif subcommand == "clear":
            return _queue_clear(client, console)

        elif subcommand == "export":
            name = parts[2] if len(parts) > 2 else None
            return _queue_export(client, name, console)

        else:
            console.print(f"[yellow]Unknown queue subcommand: {subcommand}[/yellow]")
            console.print("[cyan]Available: load, list, clear, export[/cyan]")
            return None

    elif cmd.startswith("/load "):
        # DEPRECATED: Use /queue load instead
        console.print("[yellow]⚠️  Deprecation Warning: /load is deprecated.[/yellow]")
        console.print("[yellow]   Use /queue load <file> for queue operations[/yellow]")
        console.print("[yellow]   Use /workspace load <file> for data files[/yellow]\n")

        # Load publication list from .ris file
        filename = cmd[6:].strip()

        # BUG FIX #6: Use PathResolver for secure path resolution
        resolver = PathResolver(
            current_directory=current_directory,
            workspace_path=(
                client.data_manager.workspace_path
                if hasattr(client, "data_manager")
                else None
            ),
        )
        resolved = resolver.resolve(filename, search_workspace=True, must_exist=False)

        if not resolved.is_safe:
            console.print(f"[red]❌ Security error: {resolved.error}[/red]")
            return None

        file_path = resolved.path

        # Validate file extension
        if not file_path.suffix.lower() in [".ris", ".txt"]:
            console_manager.print_error_panel(
                f"Invalid file type: {file_path.suffix}",
                "Only .ris or .txt files are supported for /load command",
            )
            return None

        # Check if file exists
        if not file_path.exists():
            console_manager.print_error_panel(
                f"File not found: {file_path}",
                f"Searched in: {current_directory}",
            )
            return None

        console.print(
            f"[cyan]📚 Loading publication list from: {file_path.name}[/cyan]\n"
        )

        try:
            with create_progress(client_arg=client) as progress:
                task = progress.add_task(
                    "[cyan]Parsing .ris file and creating queue entries...",
                    total=None,
                )

                # Load publications into queue
                result = client.load_publication_list(
                    file_path=str(file_path),
                    priority=5,  # Default priority
                    schema_type="general",  # Default schema
                    extraction_level="methods",  # Default extraction level
                )

                progress.remove_task(task)

            # Display results
            if result["added_count"] > 0:
                console.print(
                    f"[green]✅ Successfully loaded {result['added_count']} publications[/green]\n"
                )

                console.print("[cyan]📊 Load Summary:[/cyan]")
                console.print(
                    f"  • Added to queue: [bold]{result['added_count']}[/bold] publications"
                )
                if result["skipped_count"] > 0:
                    console.print(
                        f"  • Skipped: [yellow]{result['skipped_count']}[/yellow] entries (malformed or invalid)"
                    )
                console.print(
                    f"  • Status: [bold]PENDING[/bold] (ready for processing)"
                )
                console.print(f"  • Queue file: publication_queue.jsonl\n")

                console.print("[cyan]💡 Next steps:[/cyan]")
                console.print(
                    "  • View queue: [white]get_content_from_workspace(workspace='publication_queue')[/white]"
                )
                console.print(
                    "  • Process queue: Ask the research agent to process pending publications\n"
                )

                if result.get("errors") and len(result["errors"]) > 0:
                    console.print(
                        f"[yellow]⚠️  {len(result['errors'])} error(s) occurred during parsing:[/yellow]"
                    )
                    for i, error in enumerate(
                        result["errors"][:3], 1
                    ):  # Show first 3 errors
                        console.print(f"  {i}. {error}")
                    if len(result["errors"]) > 3:
                        console.print(f"  ... and {len(result['errors']) - 3} more")

                # Return summary for conversation history
                return f"Loaded {result['added_count']} publications from {file_path.name} into publication queue (skipped {result['skipped_count']})"
            else:
                console.print(
                    f"[bold red on white] ⚠️  Error [/bold red on white] "
                    f"[red]No publications could be loaded from file[/red]"
                )
                if result.get("errors"):
                    console.print(f"\n[yellow]Errors:[/yellow]")
                    for error in result["errors"][:5]:
                        console.print(f"  • {error}")
                return None

        except FileNotFoundError as e:
            console_manager.print_error_panel(
                f"File not found: {str(e)}",
                "Check the file path and try again",
            )
            return None
        except ValueError as e:
            console_manager.print_error_panel(
                f"Invalid file format: {str(e)}",
                "Ensure the file is a valid .ris format",
            )
            return None
        except Exception as e:
            console_manager.print_error_panel(
                f"Failed to load publication list: {str(e)}",
                "Check the file format and try again",
            )
            return None

    elif cmd.startswith("/archive"):
        # BUG FIX #1: Handle nested archive commands with proper cache management
        # Use ExtractionCacheManager instead of client instance variables to prevent race conditions

        # Check if extraction cache is available (premium feature)
        if not HAS_EXTRACTION_CACHE:
            console.print(
                "[yellow]Archive caching is a premium feature not available in this distribution.[/yellow]"
            )
            console.print(
                "[dim]Use extract_and_load_archive() to load entire archives instead.[/dim]"
            )
            return None

        parts = cmd.split(maxsplit=2)
        subcommand = parts[1] if len(parts) > 1 else "help"

        # Initialize cache manager (thread-safe, per-request instance)
        cache_manager = ExtractionCacheManager(client.data_manager.workspace_path)
        recent_caches = cache_manager.list_all_caches()

        if not recent_caches:
            console.print("[red]❌ No cached archives found[/red]")
            console.print(
                "[yellow]💡 Run /read <archive.tar> first to inspect an archive[/yellow]"
            )
            return None

        # Select cache: use most recent if only one, otherwise prompt user
        if len(recent_caches) == 1:
            cache_id = recent_caches[0]["cache_id"]
        else:
            # Multiple caches available - show list and use most recent
            console.print(
                f"\n[cyan]📦 Found {len(recent_caches)} cached archives (using most recent):[/cyan]"
            )
            for i, cache in enumerate(recent_caches[:3], 1):
                age_hours = (time.time() - cache.get("timestamp", 0)) / 3600
                console.print(f"  {i}. {cache['cache_id']} ({age_hours:.1f}h ago)")
            cache_id = recent_caches[0]["cache_id"]  # Most recent

        # Get cache info and nested_info
        cache_info = cache_manager.get_cache_info(cache_id)
        if not cache_info:
            console.print(f"[red]❌ Cache {cache_id} metadata not found[/red]")
            return None

        nested_info = cache_info.get("nested_info")
        if not nested_info:
            console.print(
                f"[red]❌ Cache {cache_id} missing nested structure info[/red]"
            )
            return None

        if subcommand == "list":
            # Show detailed list of all nested samples
            console.print("\n[bold white]📋 Archive Contents:[/bold white]")
            console.print(f"[dim]Cache ID: {cache_id}[/dim]\n")

            samples_table = Table(
                box=box.ROUNDED, border_style="cyan", title="All Samples"
            )
            samples_table.add_column("GSM ID", style="bold orange1")
            samples_table.add_column("Condition", style="white")
            samples_table.add_column("Number", style="grey70")
            samples_table.add_column("Filename", style="dim")

            for condition, samples in nested_info.groups.items():
                for sample in samples:
                    samples_table.add_row(
                        sample["gsm_id"],
                        condition,
                        sample["number"],
                        sample["filename"],
                    )

            console.print(samples_table)
            return f"Listed {nested_info.total_count} samples from cached archive"

        elif subcommand == "groups":
            # Show condition groups summary (nested_info already loaded above)
            console.print("\n[bold white]📂 Condition Groups:[/bold white]\n")

            groups_table = Table(box=box.ROUNDED, border_style="cyan")
            groups_table.add_column("Condition", style="bold orange1")
            groups_table.add_column("Sample Count", style="white")
            groups_table.add_column("GSM IDs", style="grey70")

            for condition, samples in nested_info.groups.items():
                gsm_ids = [s["gsm_id"] for s in samples]
                groups_table.add_row(
                    condition,
                    str(len(samples)),
                    f"{min(gsm_ids)}-{max(gsm_ids)}" if gsm_ids else "N/A",
                )

            console.print(groups_table)
            return f"Displayed {len(nested_info.groups)} condition groups"

        elif subcommand == "load":
            # Load samples by pattern
            if len(parts) < 3:
                console.print(
                    "[yellow]Usage: /archive load <pattern|GSM_ID|condition>[/yellow]"
                )
                console.print("[dim]Examples:[/dim]")
                console.print("[dim]  /archive load GSM4710689[/dim]")
                console.print("[dim]  /archive load TISSUE[/dim]")
                console.print("[dim]  /archive load PDAC_* --limit 3[/dim]")
                return None

            pattern_arg = parts[2]

            # Parse limit flag
            limit = None
            if "--limit" in pattern_arg:
                pattern, limit_str = pattern_arg.split("--limit")
                pattern = pattern.strip()
                try:
                    limit = int(limit_str.strip())
                except ValueError:
                    console.print("[red]❌ Invalid limit value[/red]")
                    return None
            else:
                pattern = pattern_arg

            console.print(
                f"[cyan]🔄 Loading samples matching '[bold]{pattern}[/bold]'...[/cyan]"
            )

            with console.status("[cyan]Loading samples...[/cyan]"):
                result = client.load_from_cache(cache_id, pattern, limit)

            if result["success"]:
                console.print(f"\n[green]✓ {result['message']}[/green]")

                # Display merged dataset if auto-concatenation occurred
                if "merged_modality" in result:
                    merged_name = result["merged_modality"]

                    # Get merged dataset details
                    try:
                        merged_adata = client.data_manager.get_modality(merged_name)

                        # Create prominent merged dataset panel
                        merged_info = f"""[bold white]Merged Dataset:[/bold white] [orange1]{merged_name}[/orange1]

[white]Shape:[/white] [cyan]{merged_adata.n_obs:,} cells × {merged_adata.n_vars:,} genes[/cyan]
[white]Batches:[/white] [cyan]{result['loaded_count']} samples merged[/cyan]
[white]Batch key:[/white] [cyan]sample_id[/cyan]

[bold white]🎯 Ready for Analysis![/bold white]
[grey70]  • Say: "Show me a UMAP of this dataset"[/grey70]
[grey70]  • Say: "Perform quality control"[/grey70]
[grey70]  • Or use: /data to inspect the dataset[/grey70]"""

                        panel = Panel(
                            merged_info,
                            title="✨ Auto-Merged Dataset",
                            border_style="green",
                            padding=(1, 2),
                        )
                        console.print(panel)

                    except Exception as e:
                        console.print(
                            f"\n[yellow]⚠️  Could not display merged dataset details: {e}[/yellow]"
                        )

                    # Show individual modalities in collapsed format
                    console.print(
                        f"\n[dim]Individual modalities (merged into '{merged_name}'):[/dim]"
                    )
                    for i, modality in enumerate(result["modalities"][:5], 1):
                        console.print(f"  [dim]{i}. {modality}[/dim]")
                    if len(result["modalities"]) > 5:
                        console.print(
                            f"  [dim]... and {len(result['modalities'])-5} more[/dim]"
                        )

                else:
                    # Single sample or no auto-concatenation
                    console.print("\n[bold white]Loaded Modalities:[/bold white]")
                    for modality in result["modalities"]:
                        console.print(f"  • [cyan]{modality}[/cyan]")

                    # Suggest next steps
                    console.print("\n[bold white]🎯 Next Steps:[/bold white]")
                    console.print(
                        "[grey70]  • Use /data to inspect the dataset[/grey70]"
                    )
                    console.print(
                        "[grey70]  • Say: 'Analyze this dataset' for natural language analysis[/grey70]"
                    )

                if result["failed"]:
                    console.print(
                        f"\n[yellow]⚠️  Failed to load {len(result['failed'])} samples:[/yellow]"
                    )
                    for failed in result["failed"][:5]:
                        console.print(f"  • [dim]{failed}[/dim]")
                    if len(result["failed"]) > 5:
                        console.print(
                            f"  • [dim]... and {len(result['failed'])-5} more[/dim]"
                        )

                # Return summary
                if "merged_modality" in result:
                    return f"Merged {result['loaded_count']} samples into '{result['merged_modality']}'"
                else:
                    return f"Loaded {result['loaded_count']} samples: {', '.join(result['modalities'][:3])}{'...' if len(result['modalities']) > 3 else ''}"

            else:
                console.print(f"\n[red]❌ {result['error']}[/red]")
                if "suggestion" in result:
                    console.print(f"[yellow]💡 {result['suggestion']}[/yellow]")
                return f"Failed to load samples: {result['error']}"

        elif subcommand == "status":
            # Show cache status (uses top-level import, already checked HAS_EXTRACTION_CACHE)
            cache_manager = ExtractionCacheManager(client.workspace_path)
            all_caches = cache_manager.list_all_caches()

            console.print("\n[bold white]📊 Extraction Cache Status:[/bold white]\n")
            console.print(
                f"[white]Total cached extractions: [yellow]{len(all_caches)}[/yellow][/white]"
            )

            if all_caches:
                cache_table = Table(box=box.ROUNDED, border_style="cyan")
                cache_table.add_column("Cache ID", style="bold orange1")
                cache_table.add_column("Archive", style="white")
                cache_table.add_column("Samples", style="yellow")
                cache_table.add_column("Extracted At", style="dim")

                for cache in all_caches:
                    from datetime import datetime

                    extracted_at = datetime.fromisoformat(cache["extracted_at"])
                    cache_table.add_row(
                        cache["cache_id"],
                        Path(cache["archive_path"]).name,
                        str(cache["nested_info"]["total_count"]),
                        extracted_at.strftime("%Y-%m-%d %H:%M"),
                    )

                console.print(cache_table)

            return f"Cache status: {len(all_caches)} active extractions"

        elif subcommand == "cleanup":
            # Clean up old caches (uses top-level import, already checked HAS_EXTRACTION_CACHE)
            cache_manager = ExtractionCacheManager(client.workspace_path)

            console.print("[cyan]🧹 Cleaning up old cached extractions...[/cyan]")
            removed_count = cache_manager.cleanup_old_caches(max_age_days=7)

            console.print(f"[green]✓ Removed {removed_count} old cache(s)[/green]")
            return f"Cleaned up {removed_count} old cached extractions"

        else:
            # Show help
            console.print("\n[bold white]📦 /archive Commands:[/bold white]\n")
            console.print(
                "[orange1]/archive list[/orange1]             - List all samples in inspected archive"
            )
            console.print(
                "[orange1]/archive groups[/orange1]           - Show condition groups"
            )
            console.print(
                "[orange1]/archive load <pattern>[/orange1]   - Load samples by pattern"
            )
            console.print(
                "[orange1]/archive status[/orange1]           - Show extraction cache status"
            )
            console.print(
                "[orange1]/archive cleanup[/orange1]          - Clear old cached extractions\n"
            )

            console.print("[bold white]Loading Patterns:[/bold white]")
            console.print("[grey70]• GSM ID:[/grey70]        GSM4710689")
            console.print("[grey70]• Condition:[/grey70]     TISSUE, PDAC_TISSUE")
            console.print("[grey70]• Glob:[/grey70]          PDAC_*, *_TISSUE_*")
            console.print("[grey70]• With limit:[/grey70]    TISSUE --limit 3")

            return None

    elif cmd == "/export":
        # Create progress bar
        with create_progress(client_arg=client) as progress:
            task = progress.add_task("Preparing export...", total=None)

            # Check if this is a local client with detailed export capabilities
            if hasattr(client, "data_manager") and hasattr(
                client.data_manager, "create_data_package"
            ):
                # For local client, we can provide detailed progress
                def update_progress(message):
                    progress.update(task, description=message)

                # Check what we're exporting to show appropriate progress messages
                data_manager = client.data_manager
                has_data = data_manager.has_data()
                has_plots = bool(getattr(data_manager, "latest_plots", []))

                if has_data and has_plots:
                    modality_count = len(getattr(data_manager, "modalities", {}))
                    plot_count = len(getattr(data_manager, "latest_plots", []))
                    update_progress(
                        f"Exporting {modality_count} datasets and {plot_count} plots..."
                    )
                elif has_data:
                    modality_count = len(getattr(data_manager, "modalities", {}))
                    update_progress(f"Exporting {modality_count} datasets...")
                elif has_plots:
                    plot_count = len(getattr(data_manager, "latest_plots", []))
                    update_progress(f"Exporting {plot_count} plots...")

                # Call export with progress callback if supported
                if (
                    "progress_callback"
                    in client.data_manager.create_data_package.__code__.co_varnames
                ):
                    # Create a modified export that uses progress callback
                    if has_data:
                        export_path = data_manager.create_data_package(
                            output_dir=str(data_manager.exports_dir),
                            progress_callback=update_progress,
                        )
                        export_path = Path(export_path)
                    else:
                        # Fallback to regular export_session for non-data exports
                        export_path = client.export_session()
                else:
                    # Fallback to regular method
                    export_path = client.export_session()
            else:
                # For cloud client or other clients, show generic progress
                progress.update(task, description="Exporting session data and plots...")
                export_path = client.export_session()

        console.print(
            f"[bold red]✓[/bold red] [white]Session exported to:[/white] [grey74]{export_path}[/grey74]"
        )
        return f"Session exported to: {export_path}"

    elif cmd == "/pipeline export":
        """Export current session as Jupyter notebook."""
        try:
            # Check if data manager supports notebook export
            if not hasattr(client, "data_manager"):
                console.print(
                    "[red]Notebook export not available for cloud client[/red]"
                )
                return "Notebook export only available for local client"

            if not hasattr(client.data_manager, "export_notebook"):
                console.print(
                    "[red]Notebook export not available - update Lobster[/red]"
                )
                return "Notebook export not available"

            # Interactive prompts
            console.print(
                "[bold white]📓 Export Session as Jupyter Notebook[/bold white]\n"
            )

            name = Prompt.ask(
                "Notebook name (no extension)", default="analysis_workflow"
            )
            if not name:
                console.print("[red]Name required[/red]")
                return "Export cancelled - no name provided"

            description = Prompt.ask("Description (optional)", default="")

            # Export via DataManagerV2
            console.print("\n[yellow]Exporting notebook...[/yellow]")
            path = client.data_manager.export_notebook(name, description)

            console.print(f"\n[green]✓ Notebook exported:[/green] {path}")
            console.print("\n[bold white]Next steps:[/bold white]")
            console.print(f"  1. [yellow]Review:[/yellow]  jupyter notebook {path}")
            console.print(
                f"  2. [yellow]Commit:[/yellow]  git add {path} && git commit -m 'Add {name}'"
            )
            console.print(
                f"  3. [yellow]Run:[/yellow]     /pipeline run {path.name} <modality>"
            )

            return f"Exported notebook: {path}"

        except ValueError as e:
            console.print(f"[red]Export failed: {e}[/red]")
            return f"Export failed: {e}"
        except Exception as e:
            console.print(f"[red]Export error: {e}[/red]")
            logger.exception("Notebook export error")
            return f"Export error: {e}"

    elif cmd == "/pipeline list":
        """List available notebooks."""
        try:
            if not hasattr(client, "data_manager"):
                console.print(
                    "[red]Notebook listing not available for cloud client[/red]"
                )
                return "Notebook listing only available for local client"

            notebooks = client.data_manager.list_notebooks()

            if not notebooks:
                console.print(
                    "[yellow]No notebooks found in .lobster/notebooks/[/yellow]"
                )
                console.print("Export one with: [green]/pipeline export[/green]")
                return "No notebooks found"

            # Create table
            table = Table(
                title="📓 Available Notebooks",
                box=box.ROUNDED,
                border_style="blue",
                title_style="bold blue on white",
            )
            table.add_column("Name", style="cyan")
            table.add_column("Steps", justify="right")
            table.add_column("Created By")
            table.add_column("Created", style="dim")
            table.add_column("Size", justify="right")

            for nb in notebooks:
                created_date = (
                    nb["created_at"].split("T")[0] if nb["created_at"] else "unknown"
                )
                table.add_row(
                    nb["name"],
                    str(nb["n_steps"]),
                    nb["created_by"],
                    created_date,
                    f"{nb['size_kb']:.1f} KB",
                )

            console.print(table)
            return f"Found {len(notebooks)} notebooks"

        except Exception as e:
            console.print(f"[red]List error: {e}[/red]")
            logger.exception("Notebook list error")
            return f"List error: {e}"

    elif cmd.startswith("/pipeline run"):
        """Run saved notebook with new data."""
        try:
            if not hasattr(client, "data_manager"):
                console.print(
                    "[red]Notebook execution not available for cloud client[/red]"
                )
                return "Notebook execution only available for local client"

            parts = cmd.split()

            # Get notebook name
            if len(parts) > 2:
                notebook_name = parts[2]
            else:
                # Interactive selection
                notebooks = client.data_manager.list_notebooks()
                if not notebooks:
                    console.print("[red]No notebooks available[/red]")
                    return "No notebooks available"

                console.print("[bold]Available notebooks:[/bold]")
                for i, nb in enumerate(notebooks, 1):
                    console.print(
                        f"  {i}. [cyan]{nb['name']}[/cyan] ({nb['n_steps']} steps)"
                    )

                selection = Prompt.ask("Select notebook number", default="1")
                try:
                    idx = int(selection) - 1
                    notebook_name = notebooks[idx]["filename"]
                except (ValueError, IndexError):
                    console.print("[red]Invalid selection[/red]")
                    return "Invalid notebook selection"

            # Get input modality
            if len(parts) > 3:
                input_modality = parts[3]
            else:
                modalities = client.data_manager.list_modalities()
                if not modalities:
                    console.print("[red]No data loaded. Use /read first.[/red]")
                    return "No data loaded"

                console.print("[bold]Available modalities:[/bold]")
                for i, mod in enumerate(modalities, 1):
                    adata = client.data_manager.modalities[mod]
                    console.print(
                        f"  {i}. [cyan]{mod}[/cyan] ({adata.n_obs} obs × {adata.n_vars} vars)"
                    )

                selection = Prompt.ask("Select modality number", default="1")
                try:
                    idx = int(selection) - 1
                    input_modality = modalities[idx]
                except (ValueError, IndexError):
                    console.print("[red]Invalid selection[/red]")
                    return "Invalid modality selection"

            # Dry run first
            console.print("\n[yellow]Running validation...[/yellow]")
            dry_result = client.data_manager.run_notebook(
                notebook_name, input_modality, dry_run=True
            )

            # Show validation
            validation = dry_result.get("validation")
            if (
                validation
                and hasattr(validation, "has_errors")
                and validation.has_errors
            ):
                console.print("[red]✗ Validation failed:[/red]")
                for error in validation.errors:
                    console.print(f"  • {error}")
                return "Validation failed"

            if (
                validation
                and hasattr(validation, "has_warnings")
                and validation.has_warnings
            ):
                console.print("[yellow]⚠ Warnings:[/yellow]")
                for warning in validation.warnings:
                    console.print(f"  • {warning}")

            console.print("\n[green]✓ Validation passed[/green]")
            console.print(f"  Steps to execute: {dry_result['steps_to_execute']}")
            console.print(
                f"  Estimated time: {dry_result['estimated_duration_minutes']} min"
            )

            # Confirm execution
            if not Confirm.ask("\nExecute notebook?"):
                console.print("Cancelled")
                return "Execution cancelled"

            # Execute
            console.print("\n[yellow]Executing notebook...[/yellow]")
            with create_progress(client_arg=client) as progress:
                task = progress.add_task("Running analysis...", total=None)
                result = client.data_manager.run_notebook(notebook_name, input_modality)

            if result["status"] == "success":
                console.print("\n[green]✓ Execution complete![/green]")
                console.print(f"  Output: {result['output_notebook']}")
                console.print(f"  Duration: {result['execution_time']:.1f}s")
                return (
                    f"Notebook executed successfully in {result['execution_time']:.1f}s"
                )
            else:
                console.print("\n[red]✗ Execution failed[/red]")
                console.print(f"  Error: {result.get('error', 'Unknown')}")
                console.print(
                    f"  Partial output: {result.get('output_notebook', 'N/A')}"
                )
                return f"Execution failed: {result.get('error', 'Unknown')}"

        except FileNotFoundError as e:
            console.print(f"[red]File not found: {e}[/red]")
            return f"Notebook not found: {e}"
        except Exception as e:
            console.print(f"[red]Execution error: {e}[/red]")
            logger.exception("Notebook execution error")
            return f"Execution error: {e}"

    elif cmd == "/pipeline info":
        """Show notebook details."""
        try:
            if not hasattr(client, "data_manager"):
                console.print("[red]Notebook info not available for cloud client[/red]")
                return "Notebook info only available for local client"

            notebooks = client.data_manager.list_notebooks()
            if not notebooks:
                console.print("[red]No notebooks found[/red]")
                return "No notebooks found"

            console.print("[bold]Select notebook:[/bold]")
            for i, nb in enumerate(notebooks, 1):
                console.print(f"  {i}. [cyan]{nb['name']}[/cyan]")

            selection = Prompt.ask("Selection", default="1")
            try:
                idx = int(selection) - 1
                nb = notebooks[idx]
            except (ValueError, IndexError):
                console.print("[red]Invalid selection[/red]")
                return "Invalid selection"

            # Load full notebook
            import nbformat

            nb_path = Path(nb["path"])
            with open(nb_path) as f:
                notebook = nbformat.read(f, as_version=4)

            metadata = notebook.metadata.get("lobster", {})

            # Display info
            console.print(f"\n[bold cyan]{nb['name']}[/bold cyan]")
            console.print(f"Created by: {metadata.get('created_by', 'unknown')}")
            console.print(f"Date: {metadata.get('created_at', 'unknown')}")
            console.print(
                f"Lobster version: {metadata.get('lobster_version', 'unknown')}"
            )
            console.print("\nDependencies:")
            for pkg, ver in metadata.get("dependencies", {}).items():
                console.print(f"  {pkg}: {ver}")

            console.print(f"\nSteps: {nb['n_steps']}")
            console.print(f"Size: {nb['size_kb']:.1f} KB")

            return f"Notebook info: {nb['name']}"

        except Exception as e:
            console.print(f"[red]Info error: {e}[/red]")
            logger.exception("Notebook info error")
            return f"Info error: {e}"

    elif cmd == "/reset":
        if Confirm.ask("[red]🦞 Reset conversation?[/red]"):
            client.reset()
            console.print("[bold red]✓[/bold red] [white]Conversation reset[/white]")
            return "Conversation reset - cleared message history and session state"
        else:
            return "Reset cancelled by user"

    elif cmd == "/data":
        # Show current data summary with enhanced metadata display
        if client.data_manager.has_data():
            summary = client.data_manager.get_data_summary()

            table = Table(
                title="🦞 Current Data Summary",
                box=box.ROUNDED,
                border_style="red",
                title_style="bold red on white",
            )
            table.add_column("Property", style="bold grey93")
            table.add_column("Value", style="white")

            table.add_row("Status", summary["status"])

            # Handle shape - might be single modality or total for multiple modalities
            if "shape" in summary:
                table.add_row("Shape", f"{summary['shape'][0]} × {summary['shape'][1]}")
            elif "total_obs" in summary and "total_vars" in summary:
                table.add_row(
                    "Total Shape", f"{summary['total_obs']} × {summary['total_vars']}"
                )

            # Handle memory usage
            if "memory_usage" in summary:
                table.add_row("Memory Usage", summary["memory_usage"])

            # Show modality name if available
            if summary.get("modality_name"):
                table.add_row("Modality", summary["modality_name"])

            # Show data type if available
            if summary.get("data_type"):
                table.add_row("Data Type", summary["data_type"])

            # Show if sparse matrix
            if summary.get("is_sparse") is not None:
                sparse_status = "✓ Sparse" if summary["is_sparse"] else "✗ Dense"
                table.add_row("Matrix Type", sparse_status)

            # Handle observation columns (prefer 'columns' key, fallback to 'obs_columns')
            obs_cols = summary.get("columns") or summary.get("obs_columns", [])
            if obs_cols:
                cols_preview = ", ".join(obs_cols[:5])
                if len(obs_cols) > 5:
                    cols_preview += f" ... (+{len(obs_cols)-5} more)"
                table.add_row("Obs Columns", cols_preview)

            # Handle variable columns if available
            if summary.get("var_columns"):
                var_cols = summary["var_columns"]
                var_preview = ", ".join(var_cols[:5])
                if len(var_cols) > 5:
                    var_preview += f" ... (+{len(var_cols)-5} more)"
                table.add_row("Var Columns", var_preview)

            # Handle sample names (prefer 'sample_names' key, fallback to 'obs_names')
            sample_names = summary.get("sample_names") or summary.get("obs_names", [])
            if sample_names:
                samples_preview = ", ".join(sample_names[:3])
                if len(sample_names) > 3:
                    samples_preview += f" ... (+{len(sample_names)-3} more)"
                table.add_row("Samples", samples_preview)

            # Show layers if available
            if summary.get("layers"):
                layers_str = ", ".join(summary["layers"])
                table.add_row("Layers", layers_str)

            # Show obsm keys if available
            if summary.get("obsm"):
                obsm_str = ", ".join(summary["obsm"])
                table.add_row("Obsm Keys", obsm_str)

            # Show metadata keys
            if summary.get("metadata_keys"):
                meta_preview = ", ".join(summary["metadata_keys"][:3])
                if len(summary["metadata_keys"]) > 3:
                    meta_preview += f" ... (+{len(summary['metadata_keys'])-3} more)"
                table.add_row("Metadata Keys", meta_preview)

            # Show processing log
            if summary.get("processing_log"):
                recent_steps = (
                    summary["processing_log"][-2:]
                    if len(summary["processing_log"]) > 2
                    else summary["processing_log"]
                )
                table.add_row("Recent Steps", "; ".join(recent_steps))

            console.print(table)

            # Show individual modality details if multiple modalities are loaded
            if summary.get("modalities"):
                console.print("\n[bold red]🧬 Individual Modality Details[/bold red]")

                modalities_table = Table(
                    box=box.SIMPLE, border_style="red", show_header=True
                )
                modalities_table.add_column("Modality", style="bold white")
                modalities_table.add_column("Shape", style="white")
                modalities_table.add_column("Type", style="cyan")
                modalities_table.add_column("Memory", style="grey74")
                modalities_table.add_column("Sparse", style="grey50")

                for mod_name, mod_info in summary["modalities"].items():
                    if isinstance(mod_info, dict) and not mod_info.get("error"):
                        shape_str = f"{mod_info['shape'][0]} × {mod_info['shape'][1]}"
                        data_type = mod_info.get("data_type", "unknown")
                        memory = mod_info.get("memory_usage", "N/A")
                        sparse = "✓" if mod_info.get("is_sparse") else "✗"

                        modalities_table.add_row(
                            mod_name, shape_str, data_type, memory, sparse
                        )
                    else:
                        # Handle error case
                        error_msg = (
                            mod_info.get("error", "Unknown error")
                            if isinstance(mod_info, dict)
                            else "Invalid data"
                        )
                        modalities_table.add_row(
                            mod_name, "Error", error_msg[:20] + "...", "N/A", "N/A"
                        )

                console.print(modalities_table)

            # Show detailed metadata if available
            if (
                hasattr(client.data_manager, "current_metadata")
                and client.data_manager.current_metadata
            ):
                metadata = client.data_manager.current_metadata
                console.print("\n[bold red]📋 Detailed Metadata:[/bold red]")

                metadata_table = Table(
                    box=box.SIMPLE, border_style="red", show_header=True
                )
                metadata_table.add_column("Key", style="bold grey93")
                metadata_table.add_column("Value", style="white")

                for key, value in list(metadata.items())[:10]:  # Show first 10 items
                    # Format value for display
                    if isinstance(value, (list, dict)):
                        display_value = (
                            str(value)[:50] + "..."
                            if len(str(value)) > 50
                            else str(value)
                        )
                    else:
                        display_value = (
                            str(value)[:50] + "..."
                            if len(str(value)) > 50
                            else str(value)
                        )
                    metadata_table.add_row(key, display_value)

                if len(metadata) > 10:
                    metadata_table.add_row("...", f"(+{len(metadata)-10} more items)")

                console.print(metadata_table)
        else:
            console.print("[grey50]No data currently loaded[/grey50]")

    elif cmd == "/metadata":
        # Show metadata store contents (for DataManagerV2) and current metadata
        console.print("[bold red]📋 Metadata Information[/bold red]\n")

        # Check if using DataManagerV2 with metadata_store
        if hasattr(client.data_manager, "metadata_store"):
            metadata_store = client.data_manager.metadata_store
            if metadata_store:
                console.print(
                    "[bold white]🗄️  Metadata Store (Cached GEO/External Data):[/bold white]"
                )

                store_table = Table(
                    box=box.ROUNDED,
                    border_style="red",
                    title="🗄️ Metadata Store",
                    title_style="bold red on white",
                )
                store_table.add_column("Dataset ID", style="bold white")
                store_table.add_column("Type", style="cyan")
                store_table.add_column("Title", style="white")
                store_table.add_column("Samples", style="grey74")
                store_table.add_column("Cached", style="grey50")

                for dataset_id, metadata_info in metadata_store.items():
                    metadata = metadata_info.get("metadata", {})
                    validation = metadata_info.get("validation", {})

                    # Extract key information
                    title = (
                        str(metadata.get("title", "N/A"))[:40] + "..."
                        if len(str(metadata.get("title", "N/A"))) > 40
                        else str(metadata.get("title", "N/A"))
                    )
                    data_type = (
                        validation.get("predicted_data_type", "unknown")
                        .replace("_", " ")
                        .title()
                    )
                    samples = (
                        len(metadata.get("samples", {}))
                        if metadata.get("samples")
                        else "N/A"
                    )

                    # Parse timestamp
                    timestamp = metadata_info.get("fetch_timestamp", "")
                    try:
                        from datetime import datetime

                        cached_time = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                        cached_str = cached_time.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        cached_str = timestamp[:16] if timestamp else "N/A"

                    store_table.add_row(
                        dataset_id, data_type, title, str(samples), cached_str
                    )

                console.print(store_table)
                console.print()
            else:
                console.print("[grey50]No cached metadata in metadata store[/grey50]\n")

        # Show current data metadata
        if (
            hasattr(client.data_manager, "current_metadata")
            and client.data_manager.current_metadata
        ):
            console.print("[bold white]📊 Current Data Metadata:[/bold white]")
            metadata = client.data_manager.current_metadata

            metadata_table = Table(box=box.SIMPLE, border_style="red", show_header=True)
            metadata_table.add_column("Key", style="bold grey93", width=25)
            metadata_table.add_column("Value", style="white", width=50)

            for key, value in metadata.items():
                # Format value for display
                if isinstance(value, dict):
                    display_value = f"Dict with {len(value)} keys: {', '.join(list(value.keys())[:3])}"
                    if len(value) > 3:
                        display_value += f" ... (+{len(value)-3} more)"
                elif isinstance(value, list):
                    display_value = f"List with {len(value)} items"
                    if len(value) > 0:
                        display_value += f": {', '.join(str(v) for v in value[:3])}"
                        if len(value) > 3:
                            display_value += f" ... (+{len(value)-3} more)"
                else:
                    display_value = str(value)
                    if len(display_value) > 60:
                        display_value = display_value[:60] + "..."

                metadata_table.add_row(key, display_value)

            console.print(metadata_table)
        else:
            console.print("[grey50]No current data metadata available[/grey50]")

    elif cmd.startswith("/workspace"):
        # Workspace management commands
        parts = cmd.split()
        subcommand = parts[1] if len(parts) > 1 else "info"

        if subcommand == "list":
            # BUG FIX #2: Use cached scan instead of explicit rescan (75% faster)
            # Check if user wants to force refresh with --refresh flag
            force_refresh = "--refresh" in cmd.lower()
            if hasattr(client.data_manager, "get_available_datasets"):
                available = client.data_manager.get_available_datasets(
                    force_refresh=force_refresh
                )
            else:
                # Fallback for older DataManager versions
                if hasattr(client.data_manager, "_scan_workspace"):
                    client.data_manager._scan_workspace()
                available = client.data_manager.available_datasets
            loaded = set(client.data_manager.modalities.keys())

            if not available:
                # Handle empty case with helpful information
                workspace_path = client.data_manager.workspace_path
                data_dir = workspace_path / "data"

                console.print("[yellow]📂 No datasets found in workspace[/yellow]")
                console.print(f"[grey70]Workspace: {workspace_path}[/grey70]")
                console.print(f"[grey70]Data directory: {data_dir}[/grey70]")

                if not data_dir.exists():
                    console.print("[red]⚠️  Data directory doesn't exist[/red]")
                    console.print(
                        f"[cyan]💡 Create it with: mkdir -p {data_dir}[/cyan]"
                    )
                else:
                    # Check what files are actually in the data directory
                    files = list(data_dir.glob("*"))
                    if files:
                        console.print(
                            f"[cyan]Found {len(files)} files in data directory, but none are supported datasets (.h5ad)[/cyan]"
                        )
                        console.print("[grey70]Files found:[/grey70]")
                        for f in files[:5]:  # Show first 5 files
                            console.print(f"  • {f.name}")
                        if len(files) > 5:
                            console.print(f"  • ... and {len(files) - 5} more")
                    else:
                        console.print(
                            f"[cyan]💡 Add .h5ad files to {data_dir} to see them here[/cyan]"
                        )

                return "No datasets found in workspace"

            # Helper function for intelligent truncation (middle ellipsis)
            def truncate_middle(text: str, max_length: int = 60) -> str:
                """Truncate text in the middle with ellipsis, preserving start and end."""
                if len(text) <= max_length:
                    return text

                # Calculate how much to keep on each side
                # Reserve 3 characters for "..."
                available_chars = max_length - 3
                start_length = (available_chars + 1) // 2  # Slightly prefer start
                end_length = available_chars // 2

                return f"{text[:start_length]}...{text[-end_length:]}"

            table = Table(title="Available Datasets", box=box.ROUNDED)
            table.add_column("#", style="dim", width=4)
            table.add_column("Status", style="green", width=6)
            table.add_column("Name", style="bold", no_wrap=False)
            table.add_column("Size", style="cyan", width=10)
            table.add_column("Shape", style="white", width=15)
            table.add_column("Modified", style="dim", width=12)

            for idx, (name, info) in enumerate(sorted(available.items()), start=1):
                status = "✓" if name in loaded else "○"
                size = f"{info['size_mb']:.1f} MB"
                shape = (
                    f"{info['shape'][0]:,} × {info['shape'][1]:,}"
                    if info["shape"]
                    else "N/A"
                )
                modified = info["modified"].split("T")[0]

                # Use intelligent truncation for long names
                display_name = truncate_middle(name, max_length=60)

                table.add_row(str(idx), status, display_name, size, shape, modified)

            console.print(table)
            console.print("\n[dim]Use '/workspace info <#>' to see full details[/dim]")
            return f"Listed {len(available)} available datasets"

        elif subcommand == "info":
            # Show detailed information for specific dataset(s)
            if len(parts) < 3:
                console.print("[red]Usage: /workspace info <#|pattern>[/red]")
                console.print(
                    "[dim]Examples: /workspace info 1, /workspace info gse12345, /workspace info *clustered*[/dim]"
                )
                return None

            selector = parts[2]

            # BUG FIX #2: Use cached scan for info command
            if hasattr(client.data_manager, "get_available_datasets"):
                available = client.data_manager.get_available_datasets(
                    force_refresh=False
                )
            else:
                # Fallback for older DataManager versions
                if hasattr(client.data_manager, "_scan_workspace"):
                    client.data_manager._scan_workspace()
                available = client.data_manager.available_datasets

            loaded = set(client.data_manager.modalities.keys())

            if not available:
                console.print("[yellow]No datasets found in workspace[/yellow]")
                return None

            # Determine if selector is an index or pattern
            matched_datasets = []

            if selector.isdigit():
                # Index-based selection
                idx = int(selector)
                sorted_names = sorted(available.keys())
                if 1 <= idx <= len(sorted_names):
                    matched_datasets = [
                        (sorted_names[idx - 1], available[sorted_names[idx - 1]])
                    ]
                else:
                    console.print(
                        f"[red]Index {idx} out of range (1-{len(sorted_names)})[/red]"
                    )
                    return None
            else:
                # Pattern-based selection
                import fnmatch

                for name, info in sorted(available.items()):
                    if fnmatch.fnmatch(name.lower(), selector.lower()):
                        matched_datasets.append((name, info))

                if not matched_datasets:
                    console.print(
                        f"[yellow]No datasets match pattern: {selector}[/yellow]"
                    )
                    return None

            # Display detailed information for matched datasets
            for name, info in matched_datasets:
                status = "✓ Loaded" if name in loaded else "○ Not Loaded"

                # Create detailed info table
                detail_table = Table(
                    title=f"Dataset: {name}",
                    box=box.ROUNDED,
                    border_style="cyan",
                    show_header=False,
                )
                detail_table.add_column("Property", style="bold cyan")
                detail_table.add_column("Value", style="white")

                detail_table.add_row("Name", name)
                detail_table.add_row("Status", status)
                detail_table.add_row("Path", info["path"])
                detail_table.add_row("Size", f"{info['size_mb']:.2f} MB")
                detail_table.add_row(
                    "Shape",
                    (
                        f"{info['shape'][0]:,} observations × {info['shape'][1]:,} variables"
                        if info["shape"]
                        else "N/A"
                    ),
                )
                detail_table.add_row("Type", info["type"])
                detail_table.add_row("Modified", info["modified"])

                # Try to detect lineage from name (basic version)
                if "_" in name:
                    parts_list = name.split("_")
                    possible_stages = [
                        p
                        for p in parts_list
                        if any(
                            keyword in p.lower()
                            for keyword in [
                                "quality",
                                "filter",
                                "normal",
                                "doublet",
                                "cluster",
                                "marker",
                                "annot",
                                "pseudobulk",
                            ]
                        )
                    ]
                    if possible_stages:
                        detail_table.add_row(
                            "Processing Stages", " → ".join(possible_stages)
                        )

                console.print(detail_table)
                console.print()  # Add spacing between datasets

            return f"Displayed details for {len(matched_datasets)} dataset(s)"

        elif subcommand == "load":
            # Load specific datasets by index, pattern, or file path
            if len(parts) < 3:
                console.print("[red]Usage: /workspace load <#|pattern|file>[/red]")
                console.print(
                    "[dim]Examples: /workspace load 1, /workspace load recent, /workspace load data.h5ad[/dim]"
                )
                return None

            selector = parts[2]

            # BUG FIX #6: Use PathResolver for secure path resolution
            resolver = PathResolver(
                current_directory=current_directory,
                workspace_path=(
                    client.data_manager.workspace_path
                    if hasattr(client, "data_manager")
                    else None
                ),
            )
            resolved = resolver.resolve(
                selector, search_workspace=True, must_exist=False
            )

            if not resolved.is_safe:
                console.print(f"[red]❌ Security error: {resolved.error}[/red]")
                return None

            file_path = resolved.path

            if file_path.exists() and file_path.is_file():
                # Load file directly into workspace
                console.print(
                    f"[cyan]📂 Loading file into workspace: {file_path.name}[/cyan]\n"
                )

                try:
                    result = client.load_data_file(str(file_path))

                    if result.get("success"):
                        console.print(
                            f"[green]✅ Loaded '{result['modality_name']}' "
                            f"({result['data_shape'][0]:,} × {result['data_shape'][1]:,})[/green]"
                        )
                        return f"Loaded file '{file_path.name}' as modality '{result['modality_name']}'"
                    else:
                        console.print(
                            f"[red]❌ {result.get('error', 'Unknown error')}[/red]"
                        )
                        if result.get("suggestion"):
                            console.print(f"[cyan]💡 {result['suggestion']}[/cyan]")
                        return None

                except Exception as e:
                    console.print(f"[red]❌ Failed to load file: {str(e)}[/red]")
                    return None

            # BUG FIX #2: Use cached scan for load command
            if hasattr(client.data_manager, "get_available_datasets"):
                available = client.data_manager.get_available_datasets(
                    force_refresh=False
                )
            else:
                # Fallback for older DataManager versions
                if hasattr(client.data_manager, "_scan_workspace"):
                    client.data_manager._scan_workspace()
                available = client.data_manager.available_datasets

            if not available:
                console.print("[yellow]No datasets found in workspace[/yellow]")
                console.print(
                    f"[dim]Tip: If '{selector}' is a file, ensure the path is correct[/dim]"
                )
                return None

            # Determine if selector is an index or pattern
            if selector.isdigit():
                # Index-based loading (single dataset)
                idx = int(selector)
                sorted_names = sorted(available.keys())
                if 1 <= idx <= len(sorted_names):
                    dataset_name = sorted_names[idx - 1]

                    console.print(
                        f"[yellow]Loading dataset: {dataset_name}...[/yellow]"
                    )

                    # Load single dataset directly
                    success = client.data_manager.load_dataset(dataset_name)

                    if success:
                        console.print(
                            f"[green]✓ Loaded dataset: {dataset_name} ({available[dataset_name]['size_mb']:.1f} MB)[/green]"
                        )
                        return "Loaded dataset from workspace"
                    else:
                        console.print(
                            f"[red]Failed to load dataset: {dataset_name}[/red]"
                        )
                        return None
                else:
                    console.print(
                        f"[red]Index {idx} out of range (1-{len(sorted_names)})[/red]"
                    )
                    return None
            else:
                # Pattern-based loading (potentially multiple datasets)
                console.print(
                    f"[yellow]Loading workspace datasets (pattern: {selector})...[/yellow]"
                )

                # Create progress bar
                with create_progress(client_arg=client) as progress:
                    task = progress.add_task(
                        f"Loading datasets matching '{selector}'...", total=None
                    )

                    # Perform workspace loading
                    result = client.data_manager.restore_session(selector)

                # Display results
                if result["restored"]:
                    console.print(
                        f"[green]✓ Loaded {len(result['restored'])} datasets ({result['total_size_mb']:.1f} MB)[/green]"
                    )
                    for name in result["restored"]:
                        console.print(f"  • {name}")
                    return f"Loaded {len(result['restored'])} datasets from workspace"
                else:
                    console.print("[yellow]No datasets loaded[/yellow]")
                    return None

        else:  # Default to info subcommand
            # Show workspace status and information
            console.print("[bold red]🏗️  Workspace Information[/bold red]\n")

            # Check if using DataManagerV2
            if hasattr(client.data_manager, "get_workspace_status"):
                workspace_status = client.data_manager.get_workspace_status()

            # Main workspace info
            workspace_table = Table(
                title="🏗️ Workspace Status",
                box=box.ROUNDED,
                border_style="red",
                title_style="bold red on white",
            )
            workspace_table.add_column("Property", style="bold grey93")
            workspace_table.add_column("Value", style="white")

            workspace_table.add_row(
                "Workspace Path", workspace_status.get("workspace_path", "N/A")
            )
            workspace_table.add_row(
                "Modalities Loaded", str(workspace_status.get("modalities_loaded", 0))
            )
            workspace_table.add_row(
                "Registered Backends",
                str(len(workspace_status.get("registered_backends", []))),
            )
            workspace_table.add_row(
                "Registered Adapters",
                str(len(workspace_status.get("registered_adapters", []))),
            )
            workspace_table.add_row(
                "Default Backend", workspace_status.get("default_backend", "N/A")
            )
            workspace_table.add_row(
                "Provenance Enabled",
                "✓" if workspace_status.get("provenance_enabled") else "✗",
            )
            workspace_table.add_row(
                "MuData Available",
                "✓" if workspace_status.get("mudata_available") else "✗",
            )

            console.print(workspace_table)

            # Show directories
            if workspace_status.get("directories"):
                dirs = workspace_status["directories"]
                console.print("\n[bold white]📁 Directories:[/bold white]")
                for dir_type, path in dirs.items():
                    console.print(f"  • {dir_type.title()}: [grey74]{path}[/grey74]")

            # Show loaded modalities
            if workspace_status.get("modality_names"):
                console.print("\n[bold white]🧬 Loaded Modalities:[/bold white]")
                for modality in workspace_status["modality_names"]:
                    console.print(f"  • {modality}")

            # Show available backends and adapters
            console.print("\n[bold white]🔧 Available Backends:[/bold white]")
            for backend in workspace_status.get("registered_backends", []):
                console.print(f"  • {backend}")

            console.print("\n[bold white]🔌 Available Adapters:[/bold white]")
            for adapter in workspace_status.get("registered_adapters", []):
                console.print(f"  • {adapter}")

    elif cmd == "/modalities":
        # Show detailed modality information (DataManagerV2 specific)
        if hasattr(client.data_manager, "list_modalities"):
            modalities = client.data_manager.list_modalities()

            if modalities:
                console.print("[bold red]🧬 Modality Details[/bold red]\n")

                for modality_name in modalities:
                    try:
                        adata = client.data_manager.get_modality(modality_name)

                        # Create modality table
                        mod_table = Table(
                            title=f"🧬 {modality_name}",
                            box=box.ROUNDED,
                            border_style="cyan",
                            title_style="bold cyan on white",
                        )
                        mod_table.add_column("Property", style="bold grey93")
                        mod_table.add_column("Value", style="white")

                        mod_table.add_row(
                            "Shape", f"{adata.n_obs} obs × {adata.n_vars} vars"
                        )

                        # Show obs columns
                        obs_cols = list(adata.obs.columns)
                        if obs_cols:
                            cols_preview = ", ".join(obs_cols[:5])
                            if len(obs_cols) > 5:
                                cols_preview += f" ... (+{len(obs_cols)-5} more)"
                            mod_table.add_row("Obs Columns", cols_preview)

                        # Show var columns
                        var_cols = list(adata.var.columns)
                        if var_cols:
                            var_preview = ", ".join(var_cols[:5])
                            if len(var_cols) > 5:
                                var_preview += f" ... (+{len(var_cols)-5} more)"
                            mod_table.add_row("Var Columns", var_preview)

                        # Show layers
                        if adata.layers:
                            layers_str = ", ".join(list(adata.layers.keys()))
                            mod_table.add_row("Layers", layers_str)

                        # Show obsm
                        if adata.obsm:
                            obsm_str = ", ".join(list(adata.obsm.keys()))
                            mod_table.add_row("Obsm", obsm_str)

                        # Show some uns info
                        if adata.uns:
                            uns_keys = list(adata.uns.keys())[:5]
                            uns_str = ", ".join(uns_keys)
                            if len(adata.uns) > 5:
                                uns_str += f" ... (+{len(adata.uns)-5} more)"
                            mod_table.add_row("Uns Keys", uns_str)

                        console.print(mod_table)
                        console.print()

                    except Exception as e:
                        console.print(
                            f"[red]Error accessing modality {modality_name}: {e}[/red]"
                        )
            else:
                console.print("[grey50]No modalities loaded[/grey50]")
        else:
            console.print(
                "[grey50]Modality information not available (using legacy DataManager)[/grey50]"
            )

    elif cmd.startswith("/describe"):
        # Show detailed information about a specific modality
        parts = cmd.split()
        if len(parts) < 2:
            console.print("[red]Usage: /describe <modality_name>[/red]")
            console.print("[dim]Available modalities:[/dim]")
            if hasattr(client.data_manager, "list_modalities"):
                modalities = client.data_manager.list_modalities()
                for mod in modalities:
                    console.print(f"  • {mod}")
            return None

        modality_name = parts[1]

        # Check if modality exists
        if hasattr(client.data_manager, "list_modalities"):
            if modality_name not in client.data_manager.list_modalities():
                console.print(f"[red]Modality '{modality_name}' not found[/red]")
                console.print("[dim]Available modalities:[/dim]")
                for mod in client.data_manager.list_modalities():
                    console.print(f"  • {mod}")
                return None

            try:
                # Get the modality
                adata = client.data_manager.get_modality(modality_name)

                # Create main header
                console.print()
                console.print(
                    f"[bold {LobsterTheme.PRIMARY_ORANGE}]🧬 Modality: {modality_name}[/bold {LobsterTheme.PRIMARY_ORANGE}]"
                )
                console.print("━" * 60)

                # Basic Information
                matrix_info = _get_matrix_info(adata.X)
                console.print("\n[bold white]📊 Basic Information[/bold white]")
                basic_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
                basic_table.add_column("Property", style="grey70")
                basic_table.add_column("Value", style="white")

                basic_table.add_row(
                    "Shape",
                    f"{adata.n_obs:,} observations × {adata.n_vars:,} variables",
                )
                basic_table.add_row("Memory", f"{matrix_info['memory_mb']:.1f} MB")
                if matrix_info["sparse"]:
                    basic_table.add_row(
                        "Matrix Type",
                        f"Sparse ({matrix_info['format']}, {matrix_info['density']:.1f}% density)",
                    )
                    basic_table.add_row("Non-zero", f"{matrix_info['nnz']:,} elements")
                else:
                    basic_table.add_row("Matrix Type", "Dense array")
                basic_table.add_row("Data Type", matrix_info["dtype"])

                console.print(basic_table)

                # Data Matrix (X) Preview
                console.print("\n[bold white]📈 Data Matrix (X)[/bold white]")
                console.print("[grey70]Preview (first 5×5 cells):[/grey70]")
                x_preview = _format_data_preview(adata.X)
                console.print(x_preview)

                # Observations (obs)
                if not adata.obs.empty:
                    console.print(
                        f"\n[bold white]🔬 Observations (obs) - {adata.n_obs:,} cells[/bold white]"
                    )

                    # Column information
                    obs_info = []
                    for col in adata.obs.columns:
                        dtype = str(adata.obs[col].dtype)
                        obs_info.append(f"{col} ({dtype})")

                    console.print(
                        f"[grey70]Columns ({len(adata.obs.columns)}):[/grey70] {', '.join(obs_info[:5])}"
                    )
                    if len(obs_info) > 5:
                        console.print(
                            f"[grey50]... and {len(obs_info) - 5} more columns[/grey50]"
                        )

                    # Preview table
                    if len(adata.obs) > 0:
                        console.print("[grey70]Preview:[/grey70]")
                        obs_preview = _format_dataframe_preview(adata.obs)
                        console.print(obs_preview)

                # Variables (var)
                if not adata.var.empty:
                    console.print(
                        f"\n[bold white]🧪 Variables (var) - {adata.n_vars:,} features[/bold white]"
                    )

                    # Column information
                    var_info = []
                    for col in adata.var.columns:
                        dtype = str(adata.var[col].dtype)
                        var_info.append(f"{col} ({dtype})")

                    console.print(
                        f"[grey70]Columns ({len(adata.var.columns)}):[/grey70] {', '.join(var_info[:5])}"
                    )
                    if len(var_info) > 5:
                        console.print(
                            f"[grey50]... and {len(var_info) - 5} more columns[/grey50]"
                        )

                    # Preview table
                    if len(adata.var) > 0:
                        console.print("[grey70]Preview:[/grey70]")
                        var_preview = _format_dataframe_preview(adata.var)
                        console.print(var_preview)

                # Additional Data Structures
                console.print(
                    "\n[bold white]📦 Additional Data Structures[/bold white]"
                )

                # Layers
                if adata.layers:
                    console.print(f"\n[cyan]Layers ({len(adata.layers)}):[/cyan]")
                    for layer_name, layer_data in adata.layers.items():
                        layer_info = _get_matrix_info(layer_data)
                        console.print(
                            f"  • {layer_name}: {layer_info['shape'][0]}×{layer_info['shape'][1]} {layer_info['dtype']}"
                        )

                # Obsm (observation matrices)
                if adata.obsm:
                    console.print("\n[cyan]Observation Matrices (obsm):[/cyan]")
                    obsm_table = _format_array_info(dict(adata.obsm))
                    if obsm_table:
                        console.print(obsm_table)

                # Varm (variable matrices)
                if adata.varm:
                    console.print("\n[cyan]Variable Matrices (varm):[/cyan]")
                    varm_table = _format_array_info(dict(adata.varm))
                    if varm_table:
                        console.print(varm_table)

                # Obsp (observation pairwise)
                if adata.obsp:
                    console.print("\n[cyan]Observation Pairwise (obsp):[/cyan]")
                    for key in adata.obsp.keys():
                        matrix = adata.obsp[key]
                        console.print(f"  • {key}: {matrix.shape[0]}×{matrix.shape[1]}")

                # Varp (variable pairwise)
                if adata.varp:
                    console.print("\n[cyan]Variable Pairwise (varp):[/cyan]")
                    for key in adata.varp.keys():
                        matrix = adata.varp[key]
                        console.print(f"  • {key}: {matrix.shape[0]}×{matrix.shape[1]}")

                # Unstructured data (uns)
                if adata.uns:
                    console.print("\n[cyan]Unstructured Data (uns):[/cyan]")
                    uns_items = []
                    for key, value in adata.uns.items():
                        if isinstance(value, dict):
                            uns_items.append(f"{key} (dict with {len(value)} keys)")
                        elif isinstance(value, (list, tuple)):
                            uns_items.append(
                                f"{key} (list/tuple with {len(value)} items)"
                            )
                        elif isinstance(value, np.ndarray):
                            uns_items.append(f"{key} (array {value.shape})")
                        elif isinstance(value, pd.DataFrame):
                            uns_items.append(f"{key} (DataFrame {value.shape})")
                        else:
                            type_name = type(value).__name__
                            uns_items.append(f"{key} ({type_name})")

                    for item in uns_items[:10]:
                        console.print(f"  • {item}")
                    if len(uns_items) > 10:
                        console.print(
                            f"[grey50]  ... and {len(uns_items) - 10} more items[/grey50]"
                        )

                # Metadata from DataManager if available
                if (
                    hasattr(client.data_manager, "metadata_store")
                    and modality_name in client.data_manager.metadata_store
                ):
                    metadata = client.data_manager.metadata_store[modality_name]
                    console.print("\n[bold white]📋 Metadata[/bold white]")
                    meta_table = Table(
                        box=box.SIMPLE, show_header=False, padding=(0, 2)
                    )
                    meta_table.add_column("Property", style="grey70")
                    meta_table.add_column("Value", style="white")

                    if "source" in metadata:
                        meta_table.add_row("Source", metadata["source"])
                    if "created_at" in metadata:
                        meta_table.add_row("Created", metadata["created_at"])
                    if "geo_accession" in metadata:
                        meta_table.add_row("GEO Accession", metadata["geo_accession"])

                    console.print(meta_table)

                console.print()
                return f"Described modality: {modality_name}"

            except Exception as e:
                console.print(f"[red]Error describing modality: {e}[/red]")
                return None
        else:
            console.print(
                "[grey50]Describe command not available (using legacy DataManager)[/grey50]"
            )
            return None

    elif cmd == "/plots":
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

    elif cmd.startswith("/plot"):
        # Handle /plot command with optional plot ID/name
        parts = cmd.split(maxsplit=1)

        if len(parts) == 1:
            # /plot with no arguments - open plots directory
            plots_dir = client.data_manager.workspace_path / "plots"

            # Ensure plots directory exists
            if not plots_dir.exists():
                plots_dir.mkdir(parents=True, exist_ok=True)
                # Save any existing plots to the directory
                if client.data_manager.latest_plots:
                    saved_files = client.data_manager.save_plots_to_workspace()
                    if saved_files:
                        console.print(
                            f"[bold red]✓[/bold red] [white]Saved {len(saved_files)} plot files to workspace[/white]"
                        )

            # Open the directory in file manager using centralized system utility
            success, message = open_path(plots_dir)

            if success:
                console.print(f"[bold red]✓[/bold red] [white]{message}[/white]")
            else:
                console.print(
                    f"[bold red on white] ⚠️  Error [/bold red on white] [red]{message}[/red]"
                )
                console.print(
                    f"[white]Plots directory:[/white] [grey74]{plots_dir}[/grey74]"
                )

        else:
            # /plot <ID or name> - open specific plot
            plot_identifier = parts[1].strip()

            # FIRST: Find the plot by ID or partial title match (before expensive operations)
            found_plot = None
            plot_info = None

            for plot_entry in client.data_manager.latest_plots:
                # Check ID match
                if plot_entry["id"] == plot_identifier:
                    found_plot = plot_entry
                    plot_info = plot_entry
                    break
                # Check title match (case-insensitive partial match)
                elif (
                    plot_identifier.lower() in plot_entry["title"].lower()
                    or plot_identifier.lower() in plot_entry["original_title"].lower()
                ):
                    found_plot = plot_entry
                    plot_info = plot_entry
                    break

            if found_plot:
                # ONLY if plot exists: ensure plots are saved to workspace
                plots_dir = client.data_manager.workspace_path / "plots"
                if not plots_dir.exists():
                    plots_dir.mkdir(parents=True, exist_ok=True)

                # Save plots if needed
                if client.data_manager.latest_plots:
                    saved_files = client.data_manager.save_plots_to_workspace()

                # Construct the filename
                plot_id = plot_info["id"]
                plot_title = plot_info["title"]

                # Create sanitized filename (same logic as save_plots_to_workspace)
                safe_title = "".join(
                    c for c in plot_title if c.isalnum() or c in [" ", "_", "-"]
                ).rstrip()
                safe_title = safe_title.replace(" ", "_")
                filename_base = f"{plot_id}_{safe_title}" if safe_title else plot_id

                # Try to open HTML file first, then PNG
                html_path = plots_dir / f"{filename_base}.html"
                png_path = plots_dir / f"{filename_base}.png"

                file_to_open = html_path if html_path.exists() else png_path

                if file_to_open.exists():
                    # Open plot using centralized system utility
                    success, message = open_path(file_to_open)

                    if success:
                        console.print(
                            f"[bold red]✓[/bold red] [white]Opened plot:[/white] [grey74]{plot_info['original_title']}[/grey74]"
                        )
                    else:
                        console.print(
                            f"[bold red on white] ⚠️  Error [/bold red on white] [red]Failed to open plot: {message}[/red]"
                        )
                        console.print(
                            f"[white]Plot file:[/white] [grey74]{file_to_open}[/grey74]"
                        )
                else:
                    console.print(
                        "[bold red on white] ⚠️  Error [/bold red on white] [red]Plot file not found. Try running /save first.[/red]"
                    )
            else:
                console.print(
                    f"[bold red on white] ⚠️  Error [/bold red on white] [red]Plot not found: {plot_identifier}[/red]"
                )
                console.print(
                    "[grey50]Use /plots to see available plot IDs and titles[/grey50]"
                )

    elif cmd.startswith("/open "):
        # Handle /open command for files and folders
        file_or_folder = cmd[6:].strip()

        if not file_or_folder:
            console.print("[red]/open: missing file or folder argument[/red]")
            console.print("[grey50]Usage: /open <file_or_folder>[/grey50]")
            return "No file or folder specified for /open command"

        # BUG FIX #6: Use PathResolver for secure path resolution with workspace search
        resolver = PathResolver(
            current_directory=current_directory,
            workspace_path=(
                client.data_manager.workspace_path
                if hasattr(client, "data_manager")
                else None
            ),
        )
        resolved = resolver.resolve(
            file_or_folder, search_workspace=True, must_exist=True, allow_special=False
        )

        if not resolved.is_safe:
            console.print(f"[red]❌ Security error: {resolved.error}[/red]")
            return None

        if not resolved.exists:
            console.print(
                f"[red]/open: '{file_or_folder}': No such file or directory[/red]"
            )
            console.print(
                "[grey50]Check current directory, workspace, or use absolute path[/grey50]"
            )
            return f"File or folder '{file_or_folder}' not found"

        target_path = resolved.path

        # Open file or folder using centralized system utility
        success, message = open_path(target_path)

        if success:
            # Format success message with appropriate styling
            if target_path.is_dir():
                console.print(f"[bold red]✓[/bold red] [white]{message}[/white]")
            else:
                console.print(f"[bold red]✓[/bold red] [white]{message}[/white]")

            # Return summary for conversation history
            item_type = "folder" if target_path.is_dir() else "file"
            return (
                f"Opened {item_type} '{target_path.name}' in system default application"
            )
        else:
            console.print(f"[red]open: {message}[/red]")
            return f"Failed to open '{file_or_folder}': {message}"

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

        # Create progress bar
        with create_progress(client_arg=client) as progress:
            task = progress.add_task("Restoring datasets...", total=None)

            # Perform restoration
            result = client.data_manager.restore_session(pattern)

        # Display results
        if result["restored"]:
            console.print(
                f"[green]✓ Restored {len(result['restored'])} datasets ({result['total_size_mb']:.1f} MB)[/green]"
            )
            for name in result["restored"]:
                console.print(f"  • {name}")
            return f"Restored {len(result['restored'])} datasets from workspace"
        else:
            console.print("[yellow]No datasets to restore[/yellow]")

        if result["skipped"]:
            console.print(
                f"[dim]Skipped {len(result['skipped'])} datasets (size limit)[/dim]"
            )

        return None

    elif cmd == "/modes":
        # List all available modes/profiles
        configurator = get_agent_configurator()
        current_mode = configurator.get_current_profile()
        available_profiles = configurator.list_available_profiles()

        # Create modes table
        table = Table(
            title="🦞 Available Modes",
            box=box.ROUNDED,
            border_style="red",
            title_style="bold red on white",
        )
        table.add_column("Mode", style="bold white")
        table.add_column("Status", style="grey74")
        table.add_column("Description", style="grey50")

        for profile in sorted(available_profiles.keys()):
            # Add descriptions for each mode based on actual configurations
            description = ""
            if profile == "development":
                description = "Claude 3.7 Sonnet for all agents, 3.5 Sonnet v2 for assistant - fast development"
            elif profile == "production":
                description = "Claude 4 Sonnet for all agents, 3.5 Sonnet v2 for assistant - production ready"
            elif profile == "cost-optimized":
                description = "Claude 3.7 Sonnet for all agents, 3.5 Sonnet v2 for assistant - cost optimized"

            status = (
                "[bold green]ACTIVE[/bold green]" if profile == current_mode else ""
            )
            table.add_row(profile, status, description)

        console.print(table)

    elif cmd.startswith("/mode "):
        # Get the new mode name from the command
        new_mode = cmd[6:].strip()

        # Get available profiles
        configurator = get_agent_configurator()
        available_profiles = configurator.list_available_profiles()

        if new_mode in available_profiles:
            # Change the mode and update the client
            change_mode(new_mode, client)
            console.print(
                f"[bold red]✓[/bold red] [white]Mode changed to:[/white] [bold red]{new_mode}[/bold red]"
            )
            display_status(client)
            return f"Operation mode changed to '{new_mode}' - agent models and configurations updated"
        else:
            # Display available profilescan you
            console.print(
                f"[bold red on white] ⚠️  Error [/bold red on white] [red]Invalid mode: {new_mode}[/red]"
            )
            console.print("[white]Available modes:[/white]")
            for profile in sorted(available_profiles.keys()):
                if profile == configurator.get_current_profile():
                    console.print(f"  • [bold red]{profile}[/bold red] (current)")
                else:
                    console.print(f"  • {profile}")
            return f"Invalid mode '{new_mode}' - available modes: {', '.join(sorted(available_profiles.keys()))}"

    elif cmd.startswith("/provider"):
        # Handle provider switching and listing
        parts = cmd.split()

        if len(parts) == 1 or (len(parts) == 2 and parts[1] == "list"):
            # /provider or /provider list - Show available providers
            from lobster.config.llm_factory import LLMFactory

            available_providers = LLMFactory.get_available_providers()
            current_provider = client.provider_override or LLMFactory.get_current_provider()

            provider_table = Table(title="🔌 LLM Providers", box=box.ROUNDED)
            provider_table.add_column("Provider", style="cyan")
            provider_table.add_column("Status", style="white")
            provider_table.add_column("Active", style="green")

            for provider in ["anthropic", "bedrock", "ollama"]:
                configured = "✓ Configured" if provider in available_providers else "✗ Not configured"
                active = "●" if provider == current_provider else ""

                status_style = "green" if provider in available_providers else "grey50"
                provider_table.add_row(
                    provider.capitalize(),
                    f"[{status_style}]{configured}[/{status_style}]",
                    f"[bold green]{active}[/bold green]" if active else ""
                )

            console_manager.print(provider_table)

            console_manager.print(f"\n[cyan]💡 Usage:[/cyan]")
            console_manager.print("  • [white]/provider <name>[/white] - Switch to specified provider")
            console_manager.print("  • [white]/provider list[/white] - Show this list")
            console_manager.print("\n[cyan]Available providers:[/cyan] anthropic, bedrock, ollama")

            if current_provider:
                console_manager.print(f"\n[green]✓ Current provider: {current_provider}[/green]")

        elif len(parts) == 2:
            # /provider <name> - Switch provider
            new_provider = parts[1].lower()

            console_manager.print(f"[yellow]Switching to {new_provider} provider...[/yellow]")

            result = client.switch_provider(new_provider)

            if result["success"]:
                console_manager.print(
                    f"[green]✓ Successfully switched to {result['provider']} provider[/green]"
                )
            else:
                error_msg = result.get("error", "Unknown error")
                hint = result.get("hint", "")
                console_manager.print_error_panel(error_msg, hint)
        else:
            console_manager.print("[red]Invalid syntax. Use: /provider [list|<name>][/red]")

    elif cmd == "/clear":
        console.clear()

    elif cmd == "/exit":
        if Confirm.ask("[red]🦞 Exit Lobster?[/red]"):
            goodbye_message = f"""👋 Thank you for using Lobster by Omics-OS!

[bold white]🌟 Help us improve Lobster![/bold white]
Your feedback matters! Please take 1 minute to share your experience:

[bold {LobsterTheme.PRIMARY_ORANGE}]📝 Quick Survey:[/bold {LobsterTheme.PRIMARY_ORANGE}] [link=https://forms.cloud.microsoft/e/AkNk8J8nE8]https://forms.cloud.microsoft/e/AkNk8J8nE8[/link]

[dim grey50]Happy analyzing! 🧬🦞[/dim grey50]"""

            exit_panel = LobsterTheme.create_panel(
                goodbye_message, title="🦞 Goodbye & Thank You!"
            )
            console_manager.print(exit_panel)
            raise KeyboardInterrupt

    else:
        console.print(
            f"[bold red on white] ⚠️  Error [/bold red on white] [red]Unknown command: {cmd}[/red]"
        )


@app.command(name="os")
def os_command(
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory. Can also be set via LOBSTER_WORKSPACE env var. Default: ./.lobster_workspace",
    ),
):
    """
    Launch interactive OS-like workspace (Textual UI).

    This command starts a Textual-based interactive terminal UI with:
    - Multi-panel layout for browsing datasets
    - Keyboard-first navigation (vim-like bindings)
    - Non-blocking operations (Phase 3+)
    - Live streaming query responses (Phase 3+)

    Press 'Q' to quit, '?' for help.
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
        console.print(f"[red]❌ Failed to launch Lobster OS: {str(e)}[/red]")
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
    reasoning: Optional[bool] = typer.Option(
        None,
        "--reasoning",
        hidden=True,
        help="[DEPRECATED] Reasoning is now enabled by default. Use --no-reasoning to disable.",
    ),
    no_reasoning: bool = typer.Option(
        False,
        "--no-reasoning",
        help="Disable agent reasoning display (enabled by default)",
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
):
    """
    Send a single query to the agent system.

    Agent reasoning is shown by default. Use --no-reasoning to disable.
    """
    # Check for configuration
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        console.print(
            f"[red]❌ No configuration found. Run 'lobster init' first.[/red]"
        )
        raise typer.Exit(1)

    # Initialize client
    client = init_client(workspace, not no_reasoning, verbose, debug, profile_timings, provider)

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
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Profile to show"
    )
):
    """Show current configuration."""
    configurator = (
        initialize_configurator(profile=profile)
        if profile
        else LobsterAgentConfigurator()
    )
    configurator.print_current_config()


@config_app.command(name="test")
def test(
    profile: str = typer.Option(..., "--profile", "-p", help="Profile to test"),
    agent: Optional[str] = typer.Option(
        None, "--agent", "-a", help="Specific agent to test"
    ),
):
    """Test a specific configuration."""
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
# Available profiles: development, production, cost-optimized
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

# Example 1: Development setup (Claude 3.7 Sonnet for all agents)
# LOBSTER_PROFILE=development

# Example 2: Production setup (Claude 4 Sonnet for all agents except assistant)
# LOBSTER_PROFILE=production

# Example 3: Cost-optimized setup (Claude 3.7 Sonnet for all agents)
# LOBSTER_PROFILE=cost-optimized
"""

    with open(".env.template", "w") as f:
        f.write(template)

    console.print("[green]✅ Environment template saved to: .env.template[/green]")
    console.print("[yellow]Copy this file to .env and configure your API keys[/yellow]")


if __name__ == "__main__":
    app()
