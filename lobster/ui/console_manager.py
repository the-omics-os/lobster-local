"""
Centralized Rich console management for Lobster AI.

This module provides centralized console management with orange theming,
enhanced logging integration, and session capture capabilities.
"""

import logging
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Capture, Console
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.traceback import install as install_rich_traceback

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from .themes import DEFAULT_THEME, LobsterTheme


class HistoryConsole(Console):
    """
    Enhanced Rich Console with arrow key navigation and command history support.

    Integrates prompt-toolkit's PromptSession to provide:
    - Left/Right arrow keys for cursor movement
    - Up/Down arrow keys for command history
    - Ctrl+R for reverse search
    - Home/End for line navigation
    - Backspace/Delete for editing
    """

    def __init__(self, history_file="lobster_history", *args, **kwargs):
        """
        Initialize HistoryConsole with command history support.

        Args:
            history_file: Path to history file (relative to workspace)
            *args, **kwargs: Arguments passed to Rich Console
        """
        super().__init__(*args, **kwargs)

        if PROMPT_TOOLKIT_AVAILABLE:
            try:
                # Create history file in a safe location
                history_path = Path.home() / ".lobster" / history_file
                history_path.parent.mkdir(parents=True, exist_ok=True)

                self.history = FileHistory(str(history_path))
                self.session = PromptSession(history=self.history)
                self._has_history = True
            except Exception:
                # Fall back to no history if there are any issues
                self.session = None
                self._has_history = False
        else:
            self.session = None
            self._has_history = False
            # Only show this message once during initialization
            if not hasattr(HistoryConsole, "_showed_prompt_toolkit_message"):
                # Use a more subtle notification that doesn't break the Rich styling
                HistoryConsole._showed_prompt_toolkit_message = True

    def input(
        self, prompt="", markup=True, emoji=True, password=False, stream=None
    ) -> str:
        """
        Enhanced input method with arrow key navigation and history.

        Args:
            prompt: Prompt text with Rich markup support
            markup: Whether to parse Rich markup in prompt
            emoji: Whether to parse emojis in prompt
            password: Whether to hide input (uses getpass)
            stream: Input stream (for compatibility)

        Returns:
            User input string
        """
        # Print the prompt with Rich styling
        if prompt:
            self.print(prompt, markup=markup, emoji=emoji, end="")

        # Handle password input
        if password:
            return getpass("", stream=stream)

        # Handle stream input
        if stream:
            result = stream.readline()
            return result.rstrip("\n\r") if result else ""

        # Use prompt-toolkit for enhanced input if available
        if self._has_history and self.session:
            try:
                result = self.session.prompt("")
                return result
            except (KeyboardInterrupt, EOFError):
                raise
            except Exception:
                # Fall back to built-in input on any error
                pass

        # Fall back to built-in input
        return input()


class LobsterConsoleManager:
    """
    Centralized console manager for Lobster AI with orange theming.

    Provides a single point of configuration for all Rich console instances,
    logging, error handling, and session capture functionality.
    """

    _instance: Optional["LobsterConsoleManager"] = None
    _console: Optional[Console] = None
    _error_console: Optional[Console] = None
    _capture: Optional[Capture] = None

    def __new__(cls) -> "LobsterConsoleManager":
        """Singleton pattern for console manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize console manager if not already initialized."""
        if not hasattr(self, "_initialized"):
            self._setup_consoles()
            self._setup_logging()
            self._setup_traceback()
            self._initialized = True

    def _setup_consoles(self):
        """Setup main and error consoles with orange theming."""
        # Main console with orange theme and advanced input capabilities
        self._console = HistoryConsole(
            history_file="lobster_history",
            theme=DEFAULT_THEME,
            width=None,  # Auto-detect terminal width
            force_terminal=None,  # Auto-detect terminal capability
            color_system="auto",
            legacy_windows=False,
            safe_box=True,
            get_datetime=True,
            highlight=True,
            markup=True,
            emoji=True,
            record=False,  # Disable session recording to allow terminal scrolling
        )

        # Error console (stderr) with same theming
        self._error_console = Console(
            theme=DEFAULT_THEME,
            stderr=True,
            width=None,
            force_terminal=None,
            color_system="auto",
            legacy_windows=False,
            safe_box=True,
            highlight=True,
            markup=True,
            emoji=True,
            record=False,  # Disable session recording to allow terminal scrolling
        )

    def _setup_logging(self):
        """Setup Rich logging handlers with orange theming."""
        # Configure rich handler for main logging
        rich_handler = RichHandler(
            console=self._console,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            tracebacks_max_frames=10,
        )

        # Configure logging format
        rich_handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%X]"))

        # Setup root logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[rich_handler],
        )

        # Adjust levels for specific loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        # Suppress docling formatting clash warnings (non-fatal cosmetic issues)
        logging.getLogger("docling_core.transforms").setLevel(logging.ERROR)

    def _setup_traceback(self):
        """Setup Rich traceback integration."""
        install_rich_traceback(
            console=self._error_console,
            width=None,
            extra_lines=3,
            theme="monokai",
            word_wrap=True,
            show_locals=False,
            suppress=[],
            max_frames=20,
        )

    @property
    def console(self) -> Console:
        """Get the main console instance."""
        return self._console

    @property
    def error_console(self) -> Console:
        """Get the error console instance."""
        return self._error_console

    def print(self, *args, **kwargs):
        """Print to main console with theming support."""
        self._console.print(*args, **kwargs)

    def print_error(self, *args, **kwargs):
        """Print to error console."""
        self._error_console.print(*args, **kwargs)

    def clear(self):
        """Clear the console."""
        self._console.clear()

    def rule(self, title: str = "", style: str = "lobster.primary"):
        """Print a horizontal rule with optional title."""
        self._console.rule(title, style=style)

    def status(self, status: str, spinner: str = "dots"):
        """Create a status context manager."""
        return self._console.status(
            status, spinner=spinner, spinner_style="lobster.primary"
        )

    def create_panel(
        self,
        content: Any,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        style: str = "white on default",
    ) -> Panel:
        """Create a themed panel with orange branding."""
        return LobsterTheme.create_panel(content, title=title, subtitle=subtitle)

    def print_welcome(self, title: str = "Lobster AI", subtitle: str = None):
        """Print branded welcome message."""
        welcome_text = Text()
        welcome_text.append("🦞 ", style="")
        welcome_text.append(title, style="lobster.primary")
        if subtitle:
            welcome_text.append(f" - {subtitle}", style="text.secondary")

        panel = self.create_panel(
            welcome_text,
            title="Welcome",
        )
        self.print(panel)

    def print_status_panel(self, status_data: Dict[str, Any], title: str = "Status"):
        """Print a status panel with formatted data."""
        from rich.table import Table

        table = Table(
            show_header=True,
            header_style="table.header",
            border_style="border.primary",
            box=LobsterTheme.BOXES["primary"],
        )

        table.add_column("Property", style="data.key")
        table.add_column("Value", style="data.value")

        for key, value in status_data.items():
            # Format key nicely
            display_key = key.replace("_", " ").title()
            # Format value based on type
            if isinstance(value, bool):
                display_value = "✓" if value else "✗"
                style = "status.success" if value else "status.error"
                table.add_row(display_key, display_value, style=style)
            else:
                table.add_row(display_key, str(value))

        panel = self.create_panel(table, title=f"🦞 {title}")
        self.print(panel)

    def print_error_panel(self, error: str, suggestion: str = None):
        """Print an error panel with optional suggestion."""
        content = Text()
        content.append("❌ ", style="status.error")
        content.append("Error: ", style="status.error")
        content.append(error, style="text.primary")

        if suggestion:
            content.append("\n\n")
            content.append("💡 Suggestion: ", style="lobster.primary")
            content.append(suggestion, style="text.secondary")

        panel = Panel(
            content,
            title="Error",
            border_style="error",
            title_align="left",
            padding=(1, 2),
        )
        self.print(panel)

    def print_success_panel(self, message: str, details: str = None):
        """Print a success panel with optional details."""
        content = Text()
        content.append("✅ ", style="status.success")
        content.append(message, style="status.success")

        if details:
            content.append("\n")
            content.append(details, style="text.secondary")

        panel = Panel(
            content,
            title="Success",
            border_style="success",
            title_align="left",
            padding=(1, 2),
        )
        self.print(panel)

    def start_capture(self):
        """Start capturing console output for session export."""
        if self._capture is None:
            self._capture = Capture()
        self._console.begin_capture()

    def stop_capture(self) -> str:
        """Stop capturing and return captured output."""
        if self._console.is_recording:
            return self._console.end_capture()
        return ""

    def export_session(self, file_path: Optional[Path] = None) -> Path:
        """Export current session to file."""
        if file_path is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = Path(f"lobster_session_{timestamp}.html")

        # Export as HTML with styling
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self._console.export_html(inline_styles=True, code_format="{code}"))

        return file_path

    def get_terminal_size(self) -> tuple[int, int]:
        """Get terminal size as (width, height)."""
        size = self._console.size
        return size.width, size.height

    def is_interactive(self) -> bool:
        """Check if running in interactive terminal."""
        return self._console.is_terminal

    def supports_color(self) -> bool:
        """Check if terminal supports color."""
        return self._console.color_system is not None

    def create_live_context(self, renderable: Any, refresh_per_second: float = 4):
        """Create a Live context for real-time updates."""
        return Live(
            renderable,
            console=self._console,
            refresh_per_second=refresh_per_second,
            auto_refresh=True,
            vertical_overflow="visible",
            transient=True,  # Enable transient mode to allow terminal scrolling
        )

    def has_history_support(self) -> bool:
        """Check if the console has command history support enabled."""
        return (
            isinstance(self._console, HistoryConsole)
            and hasattr(self._console, "_has_history")
            and self._console._has_history
        )

    def get_input_features(self) -> dict:
        """Get information about available input features."""
        if self.has_history_support():
            return {
                "arrow_navigation": True,
                "command_history": True,
                "reverse_search": True,
                "line_editing": True,
                "history_file": str(Path.home() / ".lobster" / "lobster_history"),
            }
        else:
            return {
                "arrow_navigation": False,
                "command_history": False,
                "reverse_search": False,
                "line_editing": False,
                "suggestion": "Install prompt-toolkit for enhanced input: pip install prompt-toolkit",
            }


# Global console manager instance
_console_manager: Optional[LobsterConsoleManager] = None


def get_console() -> Console:
    """Get the main console instance."""
    global _console_manager
    if _console_manager is None:
        _console_manager = LobsterConsoleManager()
    return _console_manager.console


def get_console_manager() -> LobsterConsoleManager:
    """Get the console manager instance."""
    global _console_manager
    if _console_manager is None:
        _console_manager = LobsterConsoleManager()
    return _console_manager


def setup_logging(level: int = logging.INFO):
    """Setup logging with Rich handlers and update all handler levels."""
    get_console_manager()
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Update all existing handlers to respect the new level
    for handler in root_logger.handlers:
        handler.setLevel(level)


def print_lobster(*args, **kwargs):
    """Quick print function with lobster theming."""
    get_console().print(*args, **kwargs)


def clear_console():
    """Quick clear function."""
    get_console_manager().clear()


def create_lobster_status(message: str):
    """Quick status context manager."""
    return get_console_manager().status(message)
