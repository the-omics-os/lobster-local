"""
Modular Error Handling System for Lobster AI.

Provides a professional, extensible framework for handling different types of
LLM and API errors with user-friendly guidance and actionable solutions.

Architecture:
- ErrorGuidance: Structured error information
- ErrorHandler: Abstract base class for specific handlers
- ProviderErrorParser: Strategy pattern for provider-specific error parsing
- ErrorHandlerRegistry: Manages and dispatches to handlers
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

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


# =============================================================================
# Provider Error Parsing System (Strategy Pattern)
# =============================================================================


@dataclass
class RateLimitInfo:
    """
    Parsed rate limit error information.

    Provides a unified structure for rate limit errors across all providers,
    enabling consistent handling regardless of the underlying API.
    """

    provider: str
    display_name: str
    retry_delay: int  # seconds
    quota_metric: Optional[str] = None
    quota_limit: Optional[str] = None
    quota_id: Optional[str] = None
    documentation_url: Optional[str] = None
    console_url: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


class ProviderErrorParser(ABC):
    """
    Abstract base class for provider-specific error parsing.

    Implement this class to add support for new LLM providers' rate limit errors.
    Each parser is responsible for detecting its provider and extracting
    rate limit information from error messages.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Canonical provider identifier (e.g., 'gemini', 'anthropic')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-friendly provider name (e.g., 'Google Gemini')."""
        pass

    @abstractmethod
    def can_parse(self, error_str: str) -> bool:
        """
        Check if this parser can handle the given error.

        Args:
            error_str: String representation of the error

        Returns:
            True if this parser recognizes the error format
        """
        pass

    @abstractmethod
    def parse(self, error_str: str) -> RateLimitInfo:
        """
        Parse the error string and extract rate limit information.

        Args:
            error_str: String representation of the error

        Returns:
            RateLimitInfo with extracted information
        """
        pass

    def _extract_number(self, pattern: str, text: str, default: int = 0) -> int:
        """Helper to extract numeric values from error text."""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(float(match.group(1)))
            except (ValueError, IndexError):
                pass
        return default

    def _format_large_number(self, value: int) -> str:
        """Format large numbers for readability (e.g., 1000000 -> '1M')."""
        if value >= 1_000_000:
            return f"{value // 1_000_000}M"
        elif value >= 1_000:
            return f"{value // 1_000}K"
        return str(value)


class GeminiErrorParser(ProviderErrorParser):
    """
    Parser for Google Gemini API rate limit errors.

    Handles RESOURCE_EXHAUSTED errors with quota information and retry delays.
    """

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google Gemini"

    def can_parse(self, error_str: str) -> bool:
        error_lower = error_str.lower()
        return (
            "resource_exhausted" in error_lower
            and ("gemini" in error_lower or "generativelanguage.googleapis.com" in error_lower)
        ) or ("429" in error_str and "google" in error_lower and "gemini" in error_lower)

    def parse(self, error_str: str) -> RateLimitInfo:
        # Extract retry delay from various patterns
        retry_delay = self._extract_number(r"retry in (\d+(?:\.\d+)?)s", error_str, default=0)
        if not retry_delay:
            retry_delay = self._extract_number(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", error_str, default=15)
        retry_delay = max(retry_delay + 1, 5)  # Add buffer, minimum 5s

        # Extract quota metric
        quota_metric = None
        metric_match = re.search(r"Quota exceeded for metric:\s*([^,\n]+)", error_str)
        if metric_match:
            raw_metric = metric_match.group(1).strip()
            if "input_token" in raw_metric:
                quota_metric = "Input tokens per minute"
            elif "output_token" in raw_metric:
                quota_metric = "Output tokens per minute"
            elif "request" in raw_metric.lower():
                quota_metric = "Requests per minute"
            else:
                quota_metric = raw_metric.split("/")[-1].replace("_", " ").title()

        # Extract quota limit
        quota_limit = None
        limit_match = re.search(r"limit:\s*(\d+)", error_str)
        if limit_match:
            quota_limit = self._format_large_number(int(limit_match.group(1)))

        # Extract quota ID
        quota_id = None
        id_match = re.search(r"quotaId['\"]?\s*:\s*['\"]?([^'\"}\s,]+)", error_str)
        if id_match:
            quota_id = id_match.group(1)

        return RateLimitInfo(
            provider=self.provider_name,
            display_name=self.display_name,
            retry_delay=retry_delay,
            quota_metric=quota_metric,
            quota_limit=quota_limit,
            quota_id=quota_id,
            documentation_url="https://ai.google.dev/gemini-api/docs/rate-limits",
            console_url="https://aistudio.google.com/apikey",
            additional_info={
                "usage_monitor": "https://ai.dev/rate-limit",
                "model_hint": "Consider Gemini Flash for higher rate limits",
            },
        )


class AnthropicErrorParser(ProviderErrorParser):
    """Parser for Anthropic Claude API rate limit errors."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        return "Anthropic Claude"

    def can_parse(self, error_str: str) -> bool:
        error_lower = error_str.lower()
        return (
            ("anthropic" in error_lower or "claude" in error_lower)
            and any(p in error_lower for p in ["rate_limit", "429", "too many requests"])
        ) or "exceeded your organization" in error_lower

    def parse(self, error_str: str) -> RateLimitInfo:
        retry_delay = self._extract_number(r"retry.?after['\"]?\s*[:=]\s*(\d+)", error_str, default=60)

        return RateLimitInfo(
            provider=self.provider_name,
            display_name=self.display_name,
            retry_delay=retry_delay,
            quota_metric="Requests or tokens per minute",
            documentation_url="https://docs.anthropic.com/en/api/rate-limits",
            console_url="https://console.anthropic.com/settings/limits",
            additional_info={
                "tier_info": "New accounts have conservative limits",
            },
        )


class BedrockErrorParser(ProviderErrorParser):
    """Parser for AWS Bedrock rate limit errors (ThrottlingException)."""

    @property
    def provider_name(self) -> str:
        return "bedrock"

    @property
    def display_name(self) -> str:
        return "AWS Bedrock"

    def can_parse(self, error_str: str) -> bool:
        error_lower = error_str.lower()
        return (
            "throttlingexception" in error_lower
            or "bedrock" in error_lower
            or ("converse" in error_lower and "throttl" in error_lower)
        )

    def parse(self, error_str: str) -> RateLimitInfo:
        retry_delay = self._extract_number(r"retry.?after['\"]?\s*[:=]\s*(\d+)", error_str, default=60)

        return RateLimitInfo(
            provider=self.provider_name,
            display_name=self.display_name,
            retry_delay=retry_delay,
            quota_metric="Requests or tokens per minute",
            documentation_url="https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html",
            console_url="https://console.aws.amazon.com/servicequotas/home/services/bedrock/quotas",
            additional_info={
                "quota_request": "Request increase via AWS Service Quotas console",
            },
        )


class OllamaErrorParser(ProviderErrorParser):
    """Parser for Ollama rate limit/resource errors."""

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama (Local)"

    def can_parse(self, error_str: str) -> bool:
        error_lower = error_str.lower()
        return "ollama" in error_lower and any(
            p in error_lower for p in ["rate", "limit", "resource", "busy", "overload"]
        )

    def parse(self, error_str: str) -> RateLimitInfo:
        return RateLimitInfo(
            provider=self.provider_name,
            display_name=self.display_name,
            retry_delay=5,
            quota_metric="Local resource constraints",
            documentation_url="https://ollama.ai/",
            additional_info={
                "hint": "Check GPU memory and concurrent request limits",
            },
        )


class ProviderErrorParserRegistry:
    """
    Registry of provider-specific error parsers.

    Maintains a list of parsers and dispatches errors to the appropriate one.
    Parsers are tried in registration order (first match wins).
    """

    _instance: Optional["ProviderErrorParserRegistry"] = None
    _parsers: List[ProviderErrorParser]

    def __new__(cls) -> "ProviderErrorParserRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._parsers = []
            cls._instance._register_default_parsers()
        return cls._instance

    def _register_default_parsers(self) -> None:
        """Register built-in parsers in priority order."""
        # More specific parsers first
        self._parsers = [
            GeminiErrorParser(),
            AnthropicErrorParser(),
            BedrockErrorParser(),
            OllamaErrorParser(),
        ]

    def register(self, parser: ProviderErrorParser) -> None:
        """Register a custom parser (added to front for priority)."""
        self._parsers.insert(0, parser)

    def parse(self, error_str: str) -> Optional[RateLimitInfo]:
        """
        Parse error using the first matching parser.

        Args:
            error_str: Error message to parse

        Returns:
            RateLimitInfo if a parser matched, None otherwise
        """
        for parser in self._parsers:
            try:
                if parser.can_parse(error_str):
                    logger.debug(f"Using {parser.display_name} parser for rate limit error")
                    return parser.parse(error_str)
            except Exception as e:
                logger.warning(f"Parser {parser.provider_name} failed: {e}")
                continue
        return None

    def detect_provider(self, error_str: str) -> Optional[str]:
        """Detect provider from error string without full parsing."""
        for parser in self._parsers:
            if parser.can_parse(error_str):
                return parser.provider_name
        return None


def get_parser_registry() -> ProviderErrorParserRegistry:
    """Get the singleton parser registry instance."""
    return ProviderErrorParserRegistry()


# =============================================================================
# Error Handlers
# =============================================================================


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
    Unified handler for LLM provider rate limit errors.

    Uses the ProviderErrorParserRegistry to dynamically detect the provider
    and parse error-specific information (retry delay, quota details).
    Supports all registered providers: Gemini, Anthropic, Bedrock, Ollama.
    """

    # Provider-agnostic detection patterns
    RATE_LIMIT_PATTERNS = [
        "429",
        "rate_limit",
        "rate limit",
        "too many requests",
        "throttling",
        "resource_exhausted",
        "exceeded your organization",  # Anthropic specific
        "usage increase rate",  # Anthropic specific
    ]

    def can_handle(self, error: Exception, error_str: str) -> bool:
        """Detect rate limit errors from various patterns."""
        error_lower = error_str.lower()
        return any(pattern in error_lower for pattern in self.RATE_LIMIT_PATTERNS)

    def handle(self, error: Exception, error_str: str) -> ErrorGuidance:
        """
        Generate dynamic rate limit guidance based on detected provider.

        Uses the parser registry to extract provider-specific information,
        then generates a unified guidance message with appropriate solutions.
        """
        parser_registry = get_parser_registry()
        rate_limit_info = parser_registry.parse(error_str)

        if rate_limit_info:
            return self._create_provider_guidance(rate_limit_info, error_str)
        else:
            return self._create_generic_guidance(error_str)

    def _create_provider_guidance(
        self, info: RateLimitInfo, error_str: str
    ) -> ErrorGuidance:
        """Generate guidance from parsed provider information."""
        # Build description dynamically
        description_parts = [f"You've hit {info.display_name}'s API rate limit."]
        if info.quota_metric:
            description_parts.append(f"Quota: {info.quota_metric}")
            if info.quota_limit:
                description_parts.append(f"(limit: {info.quota_limit})")
        description_parts.append(f"Suggested retry: ~{info.retry_delay}s.")

        # Build solutions dynamically
        solutions = [
            f"Wait {info.retry_delay} seconds and try again",
        ]

        if info.documentation_url:
            solutions.append(f"Review rate limits: {info.documentation_url}")

        if info.console_url:
            solutions.append(f"Manage quotas: {info.console_url}")

        # Add provider-specific hints from additional_info
        if info.additional_info:
            if "usage_monitor" in info.additional_info:
                solutions.append(f"Monitor usage: {info.additional_info['usage_monitor']}")
            if "model_hint" in info.additional_info:
                solutions.append(info.additional_info["model_hint"])
            if "tier_info" in info.additional_info:
                solutions.append(f"Note: {info.additional_info['tier_info']}")
            if "quota_request" in info.additional_info:
                solutions.append(info.additional_info["quota_request"])
            if "hint" in info.additional_info:
                solutions.append(info.additional_info["hint"])

        solutions.append("Consider switching to another provider (anthropic, bedrock, ollama, gemini)")
        solutions.append("Contact us for assistance: info@omics-os.com")

        return ErrorGuidance(
            error_type="rate_limit",
            title=f"⚠️  {info.display_name} Rate Limit Exceeded",
            description=" ".join(description_parts),
            solutions=solutions,
            severity="warning",
            documentation_url=info.documentation_url,
            can_retry=True,
            retry_delay=info.retry_delay,
            metadata={
                "original_error": error_str[:200],
                "provider": info.provider,
                "quota_metric": info.quota_metric,
                "quota_limit": info.quota_limit,
                "parsed_retry_delay": info.retry_delay,
            },
        )

    def _create_generic_guidance(self, error_str: str) -> ErrorGuidance:
        """Fallback guidance when provider cannot be determined."""
        # Try to detect provider from environment
        provider = "unknown"
        if os.environ.get("LOBSTER_LLM_PROVIDER"):
            provider = os.environ.get("LOBSTER_LLM_PROVIDER", "unknown")
        elif os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("AWS_BEDROCK_ACCESS_KEY"):
            provider = "bedrock"

        # Determine documentation URL based on detected provider
        doc_urls = {
            "gemini": "https://ai.google.dev/gemini-api/docs/rate-limits",
            "anthropic": "https://docs.anthropic.com/en/api/rate-limits",
            "bedrock": "https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html",
        }
        documentation_url = doc_urls.get(provider)

        return ErrorGuidance(
            error_type="rate_limit",
            title="⚠️  API Rate Limit Exceeded",
            description=(
                "You've hit your LLM provider's API rate limit. "
                "This occurs when you exceed the allowed requests per minute."
            ),
            solutions=[
                "Wait 60 seconds and try again (limits reset periodically)",
                "Gemini: https://ai.google.dev/gemini-api/docs/rate-limits",
                "Anthropic: https://docs.anthropic.com/en/api/rate-limits",
                "AWS Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html",
                "Consider switching to another provider",
                "Contact us for assistance: info@omics-os.com",
            ],
            severity="warning",
            documentation_url=documentation_url,
            can_retry=True,
            retry_delay=60,
            metadata={"original_error": error_str[:200], "provider": provider},
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


# =============================================================================
# LLM Retry Utilities
# =============================================================================


@dataclass
class LLMRetryConfig:
    """
    Configuration for LLM API retry behavior.

    Provides sensible defaults for handling rate limit errors with
    exponential backoff.
    """

    max_retries: int = 3
    base_delay: float = 5.0  # seconds
    max_delay: float = 120.0  # seconds
    backoff_multiplier: float = 2.0
    jitter: float = 0.1  # ±10% random jitter
    retryable_errors: List[str] = field(
        default_factory=lambda: [
            "429",
            "rate_limit",
            "resource_exhausted",
            "throttling",
            "too many requests",
            "overloaded",
            "503",
            "502",
        ]
    )


def is_retryable_error(error: Exception, config: Optional[LLMRetryConfig] = None) -> bool:
    """
    Check if an error is retryable based on configuration.

    Args:
        error: The exception to check
        config: Retry configuration (uses defaults if None)

    Returns:
        True if the error should trigger a retry
    """
    if config is None:
        config = LLMRetryConfig()

    error_str = str(error).lower()
    return any(pattern in error_str for pattern in config.retryable_errors)


def get_retry_delay(
    error: Exception,
    attempt: int,
    config: Optional[LLMRetryConfig] = None,
) -> float:
    """
    Calculate retry delay using parsed information or exponential backoff.

    First tries to parse the retry delay from the error message (provider-specific),
    then falls back to exponential backoff with jitter.

    Args:
        error: The exception that triggered the retry
        attempt: Current retry attempt (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds before next retry
    """
    import random

    if config is None:
        config = LLMRetryConfig()

    error_str = str(error)

    # Try to get provider-specific retry delay
    parser_registry = get_parser_registry()
    rate_limit_info = parser_registry.parse(error_str)

    if rate_limit_info and rate_limit_info.retry_delay > 0:
        # Use parsed delay with small jitter
        parsed_delay = float(rate_limit_info.retry_delay)
        jitter = random.uniform(-config.jitter, config.jitter) * parsed_delay
        return min(parsed_delay + jitter, config.max_delay)

    # Fallback to exponential backoff
    delay = config.base_delay * (config.backoff_multiplier ** attempt)
    jitter = random.uniform(-config.jitter, config.jitter) * delay
    return min(delay + jitter, config.max_delay)


def retry_on_rate_limit(
    func: Optional[Callable] = None,
    *,
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> Callable:
    """
    Decorator for retrying LLM calls on rate limit errors.

    Automatically retries the decorated function when rate limit errors occur,
    using provider-specific retry delays when available or exponential backoff
    as fallback.

    Args:
        func: Function to decorate (set automatically when used without parens)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff (seconds)
        max_delay: Maximum delay between retries (seconds)
        on_retry: Optional callback(error, attempt, delay) called before each retry

    Returns:
        Decorated function with retry behavior

    Example:
        @retry_on_rate_limit(max_retries=3)
        def call_llm(prompt: str) -> str:
            return llm.invoke(prompt)

        # Or without arguments for defaults:
        @retry_on_rate_limit
        def call_llm(prompt: str) -> str:
            return llm.invoke(prompt)
    """
    import functools
    import time

    config = LLMRetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(config.max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if not is_retryable_error(e, config):
                        raise

                    if attempt >= config.max_retries:
                        logger.error(
                            f"Max retries ({config.max_retries}) exhausted for {fn.__name__}"
                        )
                        raise

                    delay = get_retry_delay(e, attempt, config)

                    # Log retry information
                    error_str = str(e)[:100]
                    logger.warning(
                        f"Rate limit error in {fn.__name__}: {error_str}... "
                        f"Retry {attempt + 1}/{config.max_retries} after {delay:.1f}s"
                    )

                    # Call optional callback
                    if on_retry:
                        try:
                            on_retry(e, attempt, delay)
                        except Exception as callback_error:
                            logger.debug(f"Retry callback error: {callback_error}")

                    time.sleep(delay)

            # Should not reach here, but just in case
            if last_error:
                raise last_error

        return wrapper

    # Handle both @retry_on_rate_limit and @retry_on_rate_limit() syntax
    if func is not None:
        return decorator(func)
    return decorator


async def retry_on_rate_limit_async(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    **kwargs,
) -> Any:
    """
    Async utility for retrying LLM calls on rate limit errors.

    Similar to the decorator but for direct async function calls.

    Args:
        func: Async function to call
        *args: Arguments to pass to func
        max_retries: Maximum retry attempts
        base_delay: Base delay for backoff
        max_delay: Maximum delay
        on_retry: Optional callback for retry events
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of func(*args, **kwargs)

    Example:
        result = await retry_on_rate_limit_async(
            llm.ainvoke,
            "What is 2+2?",
            max_retries=3
        )
    """
    import asyncio

    config = LLMRetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    last_error = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            if not is_retryable_error(e, config):
                raise

            if attempt >= config.max_retries:
                logger.error(
                    f"Max retries ({config.max_retries}) exhausted for async call"
                )
                raise

            delay = get_retry_delay(e, attempt, config)

            error_str = str(e)[:100]
            logger.warning(
                f"Rate limit error (async): {error_str}... "
                f"Retry {attempt + 1}/{config.max_retries} after {delay:.1f}s"
            )

            if on_retry:
                try:
                    on_retry(e, attempt, delay)
                except Exception as callback_error:
                    logger.debug(f"Retry callback error: {callback_error}")

            await asyncio.sleep(delay)

    if last_error:
        raise last_error
