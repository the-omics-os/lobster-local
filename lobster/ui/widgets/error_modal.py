"""Error modal for displaying critical errors with recovery options."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Callable

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Static, Button, Label
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel
from rich import box


class ErrorSeverity(Enum):
    """Error severity levels."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Structured error information."""
    title: str
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    details: Optional[str] = None
    suggestions: Optional[List[str]] = None
    can_retry: bool = False
    can_dismiss: bool = True


class ErrorModal(ModalScreen[bool]):
    """
    Modal screen for displaying critical errors.

    Inspired by Dolphie's ManualException pattern.
    Returns True if user clicked Retry, False otherwise.
    """

    DEFAULT_CSS = """
    ErrorModal {
        align: center middle;
    }

    ErrorModal > Vertical {
        width: 70;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $error;
        padding: 1 2;
    }

    ErrorModal .error-header {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $error;
        margin-bottom: 1;
    }

    ErrorModal .error-message {
        width: 100%;
        margin-bottom: 1;
    }

    ErrorModal .error-details {
        width: 100%;
        height: auto;
        max-height: 10;
        border: round $primary 30%;
        padding: 0 1;
        margin-bottom: 1;
        overflow-y: auto;
        color: $text 70%;
    }

    ErrorModal .suggestions {
        width: 100%;
        margin-bottom: 1;
    }

    ErrorModal .suggestion-item {
        color: $success;
    }

    ErrorModal .button-row {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
    }

    ErrorModal Button {
        margin: 0 1;
    }

    ErrorModal Button.retry {
        background: $warning;
    }

    ErrorModal Button.dismiss {
        background: $primary;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_modal", "Close", show=True),
        Binding("r", "retry", "Retry", show=True),
    ]

    def __init__(self, error_context: ErrorContext, **kwargs):
        super().__init__(**kwargs)
        self.error_context = error_context

    def compose(self) -> ComposeResult:
        ctx = self.error_context

        with Vertical():
            # Header with icon based on severity
            icon = self._get_severity_icon(ctx.severity)
            yield Static(f"{icon} {ctx.title}", classes="error-header")

            # Main error message
            yield Static(ctx.message, classes="error-message")

            # Details (collapsible/scrollable)
            if ctx.details:
                yield Static(ctx.details, classes="error-details")

            # Suggestions
            if ctx.suggestions:
                with Vertical(classes="suggestions"):
                    yield Static("Suggestions:", classes="suggestion-header")
                    for i, suggestion in enumerate(ctx.suggestions, 1):
                        yield Static(
                            f"  {i}. {suggestion}",
                            classes="suggestion-item"
                        )

            # Buttons
            with Horizontal(classes="button-row"):
                if ctx.can_retry:
                    yield Button("Retry", variant="warning", id="retry", classes="retry")
                if ctx.can_dismiss:
                    yield Button("Dismiss", variant="primary", id="dismiss", classes="dismiss")

    def _get_severity_icon(self, severity: ErrorSeverity) -> str:
        """Get icon for severity level."""
        icons = {
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🚨",
        }
        return icons.get(severity, "❌")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "retry":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_dismiss_modal(self) -> None:
        """Dismiss the modal."""
        self.dismiss(False)

    def action_retry(self) -> None:
        """Retry action."""
        if self.error_context.can_retry:
            self.dismiss(True)


class ConnectionErrorModal(ErrorModal):
    """Specialized modal for connection/API errors."""

    def __init__(
        self,
        error_message: str,
        provider: str = "Unknown",
        **kwargs
    ):
        context = ErrorContext(
            title="Connection Error",
            message=f"Failed to connect to {provider}",
            severity=ErrorSeverity.ERROR,
            details=error_message,
            suggestions=[
                "Check your internet connection",
                "Verify your API key is correct",
                "Try again in a few moments",
                "Check provider status page",
            ],
            can_retry=True,
            can_dismiss=True,
        )
        super().__init__(context, **kwargs)


class AgentErrorModal(ErrorModal):
    """Specialized modal for agent execution errors."""

    def __init__(
        self,
        error_message: str,
        agent_name: str = "Agent",
        **kwargs
    ):
        context = ErrorContext(
            title="Agent Error",
            message=f"{agent_name} encountered an error",
            severity=ErrorSeverity.ERROR,
            details=error_message,
            suggestions=[
                "Try rephrasing your query",
                "Check if required data is loaded",
                "Review the error details above",
            ],
            can_retry=True,
            can_dismiss=True,
        )
        super().__init__(context, **kwargs)


class DataErrorModal(ErrorModal):
    """Specialized modal for data loading/processing errors."""

    def __init__(
        self,
        error_message: str,
        data_source: str = "Data",
        **kwargs
    ):
        context = ErrorContext(
            title="Data Error",
            message=f"Failed to process {data_source}",
            severity=ErrorSeverity.ERROR,
            details=error_message,
            suggestions=[
                "Verify the data format is correct",
                "Check file permissions",
                "Ensure sufficient memory is available",
            ],
            can_retry=True,
            can_dismiss=True,
        )
        super().__init__(context, **kwargs)
