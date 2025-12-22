"""
Base provider interface for LLM providers.

This module defines the abstract interface that all LLM providers must implement,
enabling a clean, extensible architecture for supporting multiple providers.

Example:
    >>> class MyProvider(ILLMProvider):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_provider"
    ...
    ...     def is_configured(self) -> bool:
    ...         return bool(os.environ.get("MY_API_KEY"))
    ...     # ... implement other methods
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """
    Unified model information across all providers.

    Attributes:
        name: Model identifier (e.g., "claude-sonnet-4-20250514")
        display_name: Human-friendly name (e.g., "Claude Sonnet 4")
        description: Model description with capabilities
        provider: Provider name (anthropic | bedrock | ollama)
        context_window: Maximum context window size (tokens)
        is_default: Whether this is the provider's default model
        input_cost_per_million: Cost per million input tokens (USD)
        output_cost_per_million: Cost per million output tokens (USD)
    """

    name: str
    display_name: str
    description: str
    provider: str
    context_window: Optional[int] = None
    is_default: bool = False
    input_cost_per_million: Optional[float] = None
    output_cost_per_million: Optional[float] = None


class ILLMProvider(ABC):
    """
    Abstract interface for LLM providers.

    All LLM providers (Anthropic, Bedrock, Ollama, OpenAI, etc.) must implement
    this interface to be usable by Lobster's configuration system.

    The interface ensures:
    - Consistent model discovery across providers
    - Uniform health checking
    - Standardized model instantiation
    - Easy extensibility for new providers

    Example implementation:
        >>> class AnthropicProvider(ILLMProvider):
        ...     @property
        ...     def name(self) -> str:
        ...         return "anthropic"
        ...
        ...     def is_configured(self) -> bool:
        ...         return bool(os.environ.get("ANTHROPIC_API_KEY"))
        ...
        ...     def create_chat_model(self, model_id: str, **kwargs):
        ...         from langchain_anthropic import ChatAnthropic
        ...         return ChatAnthropic(model=model_id, **kwargs)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Provider identifier (lowercase, no spaces).

        Used as the key in ProviderRegistry and configuration files.

        Returns:
            str: Provider name (e.g., "anthropic", "bedrock", "ollama")
        """
        pass

    @property
    def display_name(self) -> str:
        """
        Human-friendly provider name for UI display.

        Returns:
            str: Display name (e.g., "Anthropic Direct API")
        """
        return self.name.title()

    @abstractmethod
    def is_configured(self) -> bool:
        """
        Check if provider has valid credentials/configuration.

        This checks for the presence of required configuration (API keys,
        endpoints, etc.) but does NOT verify they are valid.

        Returns:
            bool: True if required configuration is present

        Example:
            >>> provider = AnthropicProvider()
            >>> if not provider.is_configured():
            ...     print("Set ANTHROPIC_API_KEY in .env")
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if provider service is accessible (health check).

        This performs an actual connectivity test to verify the provider
        is reachable and credentials are valid.

        Returns:
            bool: True if provider is accessible and working

        Note:
            This may involve network calls. Use sparingly (cache results).
        """
        pass

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """
        List available models for this provider.

        For cloud providers (Anthropic, Bedrock), this returns a static catalog.
        For local providers (Ollama), this dynamically queries available models.

        Returns:
            List[ModelInfo]: Available models, sorted by capability (best first)

        Example:
            >>> for model in provider.list_models():
            ...     print(f"{model.name}: {model.description}")
        """
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """
        Get the recommended default model for this provider.

        Returns:
            str: Default model identifier

        Example:
            >>> model_id = provider.get_default_model()
            >>> llm = provider.create_chat_model(model_id)
        """
        pass

    @abstractmethod
    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """
        Create a LangChain chat model instance.

        Args:
            model_id: Model identifier (provider-specific format)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            **kwargs: Provider-specific parameters

        Returns:
            BaseChatModel: LangChain chat model instance

        Raises:
            ImportError: If required LangChain package not installed
            ValueError: If model_id is invalid for this provider

        Example:
            >>> llm = provider.create_chat_model(
            ...     "claude-sonnet-4-20250514",
            ...     temperature=0.7,
            ...     max_tokens=8192
            ... )
        """
        pass

    @abstractmethod
    def validate_model(self, model_id: str) -> bool:
        """
        Check if a model ID is valid for this provider.

        Args:
            model_id: Model identifier to validate

        Returns:
            bool: True if model exists and is available

        Example:
            >>> if provider.validate_model("claude-sonnet-4-20250514"):
            ...     llm = provider.create_chat_model("claude-sonnet-4-20250514")
        """
        pass

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Optional[ModelInfo]: Model information or None if not found
        """
        for model in self.list_models():
            if model.name == model_id:
                return model
        return None

    def get_model_names(self) -> List[str]:
        """
        Get list of model names (for tab completion).

        Returns:
            List[str]: Model identifiers
        """
        return [model.name for model in self.list_models()]

    def get_configuration_help(self) -> str:
        """
        Get help text for configuring this provider.

        Returns:
            str: Configuration instructions for the user
        """
        return f"Configure {self.display_name} by setting required environment variables."
