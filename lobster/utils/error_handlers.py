"""
Modular Error Handling System for Lobster AI.

Provides a professional, extensible framework for handling different types of
LLM and API errors with user-friendly guidance and actionable solutions.

Architecture:
- ErrorGuidance: Structured error information
- ErrorHandler: Abstract base class for specific handlers
- ErrorHandlerRegistry: Manages and dispatches to handlers
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ErrorGuidance:
    """
    Structured error information for user display.

    Provides comprehensive guidance including problem description,
    actionable solutions, severity level, and support resources.
    """

    error_type: str
    title: str
    description: str
    solutions: List[str]
    severity: str = "error"  # "error", "warning", "critical"
    support_email: Optional[str] = "info@omics-os.com"
    documentation_url: Optional[str] = None
    can_retry: bool = False
    retry_delay: Optional[int] = None  # seconds
    metadata: dict = field(default_factory=dict)


class ErrorHandler(ABC):
    """
    Abstract base class for error handlers.

    Implement this class to create custom error handlers that can
    be registered with the ErrorHandlerRegistry.
    """

    @abstractmethod
    def can_handle(self, error: Exception, error_str: str) -> bool:
        """
        Check if this handler can process the given error.

        Args:
            error: The exception object
            error_str: String representation of the error

        Returns:
            True if this handler should process this error
        """
        pass

    @abstractmethod
    def handle(self, error: Exception, error_str: str) -> ErrorGuidance:
        """
        Process the error and return structured guidance.

        Args:
            error: The exception object
            error_str: String representation of the error

        Returns:
            ErrorGuidance with actionable information
        """
        pass


class RateLimitErrorHandler(ErrorHandler):
    """
    Handles Anthropic rate limit errors (429).

    Provides guidance for users who have exceeded their API rate limits,
    with recommendations to request increases or switch to AWS Bedrock.
    """

    def can_handle(self, error: Exception, error_str: str) -> bool:
        """Detect rate limit errors from various patterns."""
        patterns = [
            "429",
            "rate_limit_error",
            "rate limit",
            "too many requests",
            "exceeded your organization",
            "usage increase rate",
        ]
        error_lower = error_str.lower()
        return any(pattern in error_lower for pattern in patterns)

    def handle(self, error: Exception, error_str: str) -> ErrorGuidance:
        """Generate comprehensive rate limit guidance."""
        return ErrorGuidance(
            error_type="rate_limit",
            title="⚠️  Rate Limit Exceeded",
            description=(
                "You've hit Anthropic's API rate limit. This is common for new accounts "
                "which have conservative usage limits to prevent abuse."
            ),
            solutions=[
                "Wait a few minutes and try again (limits reset periodically)",
                "Request an Anthropic rate increase at: https://docs.anthropic.com/en/api/rate-limits",
                "Switch to AWS Bedrock (recommended for production): See installation docs",
                "Contact us for assistance: info@omics-os.com",
            ],
            severity="warning",
            documentation_url="https://docs.anthropic.com/en/api/rate-limits",
            can_retry=True,
            retry_delay=60,
            metadata={"original_error": error_str[:200], "provider": "anthropic"},
        )


class AuthenticationErrorHandler(ErrorHandler):
    """
    Handles API authentication errors (401, invalid keys).

    Guides users through resolving API key configuration issues.
    """

    def can_handle(self, error: Exception, error_str: str) -> bool:
        """Detect authentication-related errors."""
        patterns = [
            "401",
            "unauthorized",
            "invalid api key",
            "authentication failed",
            "invalid_api_key",
            "api key not found",
        ]
        error_lower = error_str.lower()
        return any(pattern in error_lower for pattern in patterns)

    def handle(self, error: Exception, error_str: str) -> ErrorGuidance:
        """Generate authentication troubleshooting guidance."""
        return ErrorGuidance(
            error_type="authentication",
            title="🔑 Authentication Failed",
            description=(
                "Your API key is invalid or not configured correctly. "
                "Lobster AI requires a valid API key to access LLM services."
            ),
            solutions=[
                "Check your .env file contains: ANTHROPIC_API_KEY=sk-ant-...",
                "Verify the API key is valid at: https://console.anthropic.com/",
                "Ensure .env file is in the project root directory",
                "Reload environment variables: source .env && lobster chat",
                "For AWS Bedrock: Verify AWS credentials are configured correctly",
            ],
            severity="error",
            documentation_url="https://github.com/the-omics-os/lobster/wiki/03-configuration",
            can_retry=False,
            metadata={"original_error": error_str[:200]},
        )


class NetworkErrorHandler(ErrorHandler):
    """
    Handles network connectivity errors.

    Covers timeouts, connection refused, DNS failures, and other
    network-related issues.
    """

    def can_handle(self, error: Exception, error_str: str) -> bool:
        """Detect network-related errors."""
        patterns = [
            "connection error",
            "timeout",
            "network",
            "connection refused",
            "unreachable",
            "dns",
            "timed out",
            "connection reset",
            "no route to host",
        ]
        error_lower = error_str.lower()
        return any(pattern in error_lower for pattern in patterns)

    def handle(self, error: Exception, error_str: str) -> ErrorGuidance:
        """Generate network troubleshooting guidance."""
        return ErrorGuidance(
            error_type="network",
            title="🌐 Network Error",
            description=(
                "Unable to connect to the API service. This could be due to "
                "internet connectivity issues, firewall settings, or service outages."
            ),
            solutions=[
                "Check your internet connection is working",
                "Verify firewall settings allow HTTPS connections (port 443)",
                "Try again in a few moments (may be temporary)",
                "If using a proxy, ensure it's configured correctly",
                "Check API service status: https://status.anthropic.com",
            ],
            severity="error",
            can_retry=True,
            retry_delay=30,
            metadata={"original_error": error_str[:200]},
        )


class QuotaExceededErrorHandler(ErrorHandler):
    """
    Handles quota/billing limit errors.

    Occurs when users have exceeded their monthly spending or usage quotas.
    """

    def can_handle(self, error: Exception, error_str: str) -> bool:
        """Detect quota-related errors."""
        patterns = [
            "quota exceeded",
            "insufficient_quota",
            "billing",
            "payment required",
            "402",
        ]
        error_lower = error_str.lower()
        return any(pattern in error_lower for pattern in patterns)

    def handle(self, error: Exception, error_str: str) -> ErrorGuidance:
        """Generate quota troubleshooting guidance."""
        return ErrorGuidance(
            error_type="quota",
            title="💳 Usage Quota Exceeded",
            description=(
                "You've exceeded your API usage quota or billing limit. "
                "This typically requires updating your billing settings."
            ),
            solutions=[
                "Check your billing settings at: https://console.anthropic.com/settings/billing",
                "Review your current usage and limits",
                "Upgrade your plan or add additional credits",
                "Contact Anthropic support for quota increases",
                "Consider AWS Bedrock for enterprise-level quotas",
            ],
            severity="error",
            documentation_url="https://docs.anthropic.com/en/api/rate-limits",
            can_retry=False,
            metadata={"original_error": error_str[:200]},
        )


class ErrorHandlerRegistry:
    """
    Registry that manages all error handlers.

    Provides a centralized system for registering handlers and dispatching
    errors to the appropriate handler based on detection patterns.
    """

    def __init__(self):
        """Initialize registry with default handlers."""
        self.handlers: List[ErrorHandler] = []
        self._register_default_handlers()
        logger.debug(
            "Error handler registry initialized with %d handlers", len(self.handlers)
        )

    def _register_default_handlers(self):
        """Register built-in error handlers in priority order."""
        # Order matters - first match wins
        self.register(RateLimitErrorHandler())
        self.register(AuthenticationErrorHandler())
        self.register(QuotaExceededErrorHandler())
        self.register(NetworkErrorHandler())

    def register(self, handler: ErrorHandler):
        """
        Register a new error handler.

        Args:
            handler: ErrorHandler instance to register
        """
        self.handlers.append(handler)
        logger.debug("Registered error handler: %s", handler.__class__.__name__)

    def handle_error(self, error: Union[Exception, KeyboardInterrupt]) -> ErrorGuidance:
        """
        Process error through registered handlers.

        Tries each handler in order until one accepts the error.
        Falls back to generic guidance if no handler matches.

        Args:
            error: The exception to handle

        Returns:
            ErrorGuidance with structured information
        """
        error_str = str(error)

        # Special case for KeyboardInterrupt
        if isinstance(error, KeyboardInterrupt):
            return ErrorGuidance(
                error_type="interrupt",
                title="⚠️  Operation Cancelled",
                description="Operation was cancelled by user (Ctrl+C)",
                solutions=["Operation stopped safely", "No action required"],
                severity="warning",
                can_retry=False,
            )

        # Try each handler in order (first match wins)
        for handler in self.handlers:
            try:
                if handler.can_handle(error, error_str):
                    logger.debug(
                        "Handler %s accepted error: %s",
                        handler.__class__.__name__,
                        error_str[:100],
                    )
                    return handler.handle(error, error_str)
            except Exception as handler_error:
                # Don't let handler errors break the system
                logger.error(
                    "Handler %s failed to process error: %s",
                    handler.__class__.__name__,
                    handler_error,
                )
                continue

        # Fallback to generic error
        logger.debug("No handler matched, using generic guidance")
        return self._create_generic_guidance(error, error_str)

    def _create_generic_guidance(
        self, error: Exception, error_str: str
    ) -> ErrorGuidance:
        """
        Fallback for unhandled errors.

        Provides generic troubleshooting guidance when no specific
        handler matches the error pattern.
        """
        # Truncate very long error messages
        display_error = error_str[:500]
        if len(error_str) > 500:
            display_error += "..."

        return ErrorGuidance(
            error_type="unknown",
            title="❌ Agent Error",
            description=f"An unexpected error occurred: {display_error}",
            solutions=[
                "Check the error message above for specific details",
                "Try running with --debug flag for more information: lobster chat --debug",
                "Review the documentation at: https://github.com/the-omics-os/lobster/wiki",
                "Search for similar issues: https://github.com/the-omics-os/lobster/issues",
                "Report this issue if it persists: https://github.com/the-omics-os/lobster/issues/new",
            ],
            severity="error",
            support_email="info@omics-os.com",
            can_retry=False,
            metadata={"original_error": error_str, "error_type": type(error).__name__},
        )


# Global registry instance (singleton pattern)
_error_registry: Optional[ErrorHandlerRegistry] = None


def get_error_registry() -> ErrorHandlerRegistry:
    """
    Get the global error handler registry.

    Creates the registry on first call (lazy initialization).

    Returns:
        The global ErrorHandlerRegistry instance
    """
    global _error_registry
    if _error_registry is None:
        _error_registry = ErrorHandlerRegistry()
    return _error_registry


def reset_error_registry():
    """
    Reset the global error handler registry.

    Useful for testing or when you need to reconfigure handlers.
    """
    global _error_registry
    _error_registry = None
