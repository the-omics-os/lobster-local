"""
Error handling service for Lobster OS UI.

Provides consistent error display, categorization, and recovery patterns.
Inspired by Dolphie's ManualException and Elia's notification patterns.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Any
import re

from textual.app import App
from textual.message import Message

from lobster.ui.widgets.error_modal import (
    ErrorModal,
    ErrorContext,
    ErrorSeverity,
    ConnectionErrorModal,
    AgentErrorModal,
    DataErrorModal,
)


# Error notification timeout constants (like Elia)
ERROR_TIMEOUT = 10
WARNING_TIMEOUT = 8
SUCCESS_TIMEOUT = 3
INFO_TIMEOUT = 5


class ErrorCategory(Enum):
    """Categories of errors for routing to appropriate handlers."""
    CONNECTION = "connection"      # API/network errors
    AUTHENTICATION = "auth"        # API key/credential errors
    AGENT = "agent"               # Agent execution errors
    DATA = "data"                 # Data loading/processing errors
    VALIDATION = "validation"     # Input validation errors
    SYSTEM = "system"             # System/resource errors
    CONTEXT_LIMIT = "context_limit"  # Token/context window exceeded
    UNKNOWN = "unknown"           # Uncategorized errors


@dataclass
class AgentQueryFailed(Message):
    """Posted when an agent query fails - for UI state recovery."""
    original_query: str
    error_message: str
    category: ErrorCategory
    can_retry: bool = True


@dataclass
class AgentQueryRecovered(Message):
    """Posted when user chooses to retry after error."""
    original_query: str


class ErrorService:
    """
    Centralized error handling service for Lobster OS UI.

    Features:
    - Error categorization and routing
    - Consistent notification display
    - Modal dialogs for critical errors
    - Recovery pattern support (like Elia's AgentResponseFailed)

    Usage:
        error_service = ErrorService(app)
        error_service.handle_error(exception, category=ErrorCategory.AGENT)
    """

    # Pattern matching for error categorization
    CONNECTION_PATTERNS = [
        r"connection\s*(refused|reset|timeout)",
        r"network\s*(error|unreachable)",
        r"ssl\s*error",
        r"timeout",
        r"502|503|504",
        r"service\s*unavailable",
    ]

    AUTH_PATTERNS = [
        r"(api|access)\s*key",
        r"unauthorized|401|403",
        r"authentication\s*failed",
        r"invalid\s*(credentials|token)",
        r"permission\s*denied",
    ]

    DATA_PATTERNS = [
        r"file\s*not\s*found",
        r"(invalid|corrupt)\s*(file|data|format)",
        r"memory\s*(error|exhausted)",
        r"parsing\s*(error|failed)",
        r"adata|anndata|h5ad",
    ]

    def __init__(self, app: App):
        self.app = app

    def handle_error(
        self,
        error: Exception,
        category: Optional[ErrorCategory] = None,
        context: str = "",
        show_modal: bool = False,
        can_retry: bool = False,
        on_retry: Optional[Callable[[], Any]] = None,
    ) -> None:
        """
        Handle an error with appropriate display.

        Args:
            error: The exception that occurred
            category: Error category (auto-detected if None)
            context: Additional context about where error occurred
            show_modal: Force modal dialog display
            can_retry: Whether retry is possible
            on_retry: Callback for retry action
        """
        error_msg = str(error)

        # Auto-detect category if not provided
        if category is None:
            category = self._categorize_error(error_msg)

        # Log to activity log if available
        self._log_error(error_msg, category)

        # Determine display method based on category and severity
        if show_modal or category in [ErrorCategory.CONNECTION, ErrorCategory.AUTHENTICATION]:
            self._show_modal(error_msg, category, can_retry, on_retry)
        else:
            self._show_notification(error_msg, category, context)

    def _categorize_error(self, error_msg: str) -> ErrorCategory:
        """
        Auto-detect error category from message.

        Uses StructuredErrorParser for typed detection of LLM errors,
        falls back to pattern matching for other error types.
        """
        # Try structured parsing first for LLM errors
        try:
            from lobster.utils.error_handlers import get_structured_parser, ErrorType
            parser = get_structured_parser()
            parsed = parser.parse(error_msg)

            # Map ErrorType to ErrorCategory
            if parsed.error_type == ErrorType.CONTEXT_LIMIT:
                return ErrorCategory.CONTEXT_LIMIT
            elif parsed.error_type == ErrorType.AUTHENTICATION:
                return ErrorCategory.AUTHENTICATION
            elif parsed.error_type in (ErrorType.RATE_LIMIT, ErrorType.NETWORK, ErrorType.MODEL_OVERLOADED):
                return ErrorCategory.CONNECTION
        except Exception:
            # Fallback to pattern matching if structured parsing fails
            pass

        # Pattern-based detection for non-LLM errors
        error_lower = error_msg.lower()

        for pattern in self.CONNECTION_PATTERNS:
            if re.search(pattern, error_lower):
                return ErrorCategory.CONNECTION

        for pattern in self.AUTH_PATTERNS:
            if re.search(pattern, error_lower):
                return ErrorCategory.AUTHENTICATION

        for pattern in self.DATA_PATTERNS:
            if re.search(pattern, error_lower):
                return ErrorCategory.DATA

        return ErrorCategory.UNKNOWN

    def _show_notification(
        self,
        error_msg: str,
        category: ErrorCategory,
        context: str = "",
    ) -> None:
        """Show error as notification (non-blocking)."""
        # Truncate long messages
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."

        title = self._get_notification_title(category)
        if context:
            title = f"{title}: {context}"

        # Use Textual's built-in notify
        self.app.notify(
            error_msg,
            title=title,
            severity="error",
            timeout=ERROR_TIMEOUT,
        )

    def _show_modal(
        self,
        error_msg: str,
        category: ErrorCategory,
        can_retry: bool,
        on_retry: Optional[Callable[[], Any]],
    ) -> None:
        """Show error as modal dialog (blocking)."""
        # Select appropriate modal type
        if category == ErrorCategory.CONNECTION:
            modal = ConnectionErrorModal(error_msg)
        elif category == ErrorCategory.AGENT:
            modal = AgentErrorModal(error_msg)
        elif category == ErrorCategory.DATA:
            modal = DataErrorModal(error_msg)
        else:
            context = ErrorContext(
                title=self._get_notification_title(category),
                message="An error occurred",
                severity=ErrorSeverity.ERROR,
                details=error_msg,
                can_retry=can_retry,
                can_dismiss=True,
            )
            modal = ErrorModal(context)

        # Show modal and handle result
        def handle_result(should_retry: bool) -> None:
            if should_retry and on_retry:
                on_retry()

        self.app.push_screen(modal, handle_result)

    def _log_error(self, error_msg: str, category: ErrorCategory) -> None:
        """Log error to activity log if available."""
        try:
            from lobster.ui.widgets import ActivityLogPanel
            activity_log = self.app.query_one(ActivityLogPanel)
            activity_log.log_tool_error(category.value, error_msg[:50])
        except Exception:
            pass  # Activity log not available

    def _get_notification_title(self, category: ErrorCategory) -> str:
        """Get notification title for category."""
        titles = {
            ErrorCategory.CONNECTION: "Connection Error",
            ErrorCategory.AUTHENTICATION: "Authentication Error",
            ErrorCategory.AGENT: "Agent Error",
            ErrorCategory.DATA: "Data Error",
            ErrorCategory.VALIDATION: "Validation Error",
            ErrorCategory.SYSTEM: "System Error",
            ErrorCategory.CONTEXT_LIMIT: "Context Limit Exceeded",
            ErrorCategory.UNKNOWN: "Error",
        }
        return titles.get(category, "Error")

    # Convenience methods for common error types

    def show_connection_error(
        self,
        error: Exception,
        provider: str = "API",
        can_retry: bool = True,
        on_retry: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Show connection error with modal."""
        self.handle_error(
            error,
            category=ErrorCategory.CONNECTION,
            context=provider,
            show_modal=True,
            can_retry=can_retry,
            on_retry=on_retry,
        )

    def show_agent_error(
        self,
        error: Exception,
        agent_name: str = "Agent",
        can_retry: bool = True,
    ) -> None:
        """Show agent error as notification."""
        self.handle_error(
            error,
            category=ErrorCategory.AGENT,
            context=agent_name,
            show_modal=False,
            can_retry=can_retry,
        )

    def show_data_error(
        self,
        error: Exception,
        data_source: str = "Data",
    ) -> None:
        """Show data error as notification."""
        self.handle_error(
            error,
            category=ErrorCategory.DATA,
            context=data_source,
            show_modal=False,
        )

    def show_validation_error(self, message: str) -> None:
        """Show validation error (user input issues)."""
        self.app.notify(
            message,
            title="Validation Error",
            severity="warning",
            timeout=WARNING_TIMEOUT,
        )

    def show_success(self, message: str, title: str = "Success") -> None:
        """Show success notification."""
        self.app.notify(
            message,
            title=title,
            severity="information",
            timeout=SUCCESS_TIMEOUT,
        )

    def show_warning(self, message: str, title: str = "Warning") -> None:
        """Show warning notification."""
        self.app.notify(
            message,
            title=title,
            severity="warning",
            timeout=WARNING_TIMEOUT,
        )
