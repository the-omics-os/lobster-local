"""
Provider registry for LLM provider management.

This module provides a singleton registry for discovering and accessing
LLM providers. All providers must be registered here to be usable.

Usage:
    >>> from lobster.config.providers import ProviderRegistry, get_provider
    >>>
    >>> # Register a provider (done automatically on import)
    >>> ProviderRegistry.register(AnthropicProvider())
    >>>
    >>> # Get a specific provider
    >>> provider = get_provider("anthropic")
    >>>
    >>> # List available providers
    >>> for p in ProviderRegistry.get_available_providers():
    ...     print(f"{p.name} is ready")
"""

import logging
from typing import Dict, List, Optional

from lobster.config.providers.base_provider import ILLMProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Singleton registry for LLM providers.

    This class manages all registered LLM providers and provides methods
    for discovery and access. Providers are registered at import time.

    Thread Safety:
        This class is designed for single-threaded initialization with
        thread-safe reads. All providers should be registered during
        module import, before any concurrent access.

    Example:
        >>> ProviderRegistry.register(AnthropicProvider())
        >>> ProviderRegistry.register(BedrockProvider())
        >>>
        >>> provider = ProviderRegistry.get("anthropic")
        >>> if provider and provider.is_available():
        ...     llm = provider.create_chat_model(provider.get_default_model())
    """

    _providers: Dict[str, ILLMProvider] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, provider: ILLMProvider) -> None:
        """
        Register a provider in the registry.

        Args:
            provider: Provider instance implementing ILLMProvider

        Example:
            >>> ProviderRegistry.register(AnthropicProvider())
        """
        cls._providers[provider.name] = provider
        logger.debug(f"Registered provider: {provider.name}")

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove a provider from the registry.

        Args:
            name: Provider name to remove
        """
        if name in cls._providers:
            del cls._providers[name]
            logger.debug(f"Unregistered provider: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[ILLMProvider]:
        """
        Get a provider by name.

        Args:
            name: Provider name (e.g., "anthropic", "bedrock", "ollama")

        Returns:
            Optional[ILLMProvider]: Provider instance or None if not found

        Example:
            >>> provider = ProviderRegistry.get("anthropic")
            >>> if provider:
            ...     print(f"Found {provider.display_name}")
        """
        cls._ensure_initialized()
        return cls._providers.get(name)

    @classmethod
    def get_all(cls) -> List[ILLMProvider]:
        """
        Get all registered providers.

        Returns:
            List[ILLMProvider]: All registered providers
        """
        cls._ensure_initialized()
        return list(cls._providers.values())

    @classmethod
    def get_configured_providers(cls) -> List[ILLMProvider]:
        """
        Get all providers that have valid configuration.

        Returns:
            List[ILLMProvider]: Providers with is_configured() == True

        Example:
            >>> configured = ProviderRegistry.get_configured_providers()
            >>> print(f"{len(configured)} providers configured")
        """
        cls._ensure_initialized()
        return [p for p in cls._providers.values() if p.is_configured()]

    @classmethod
    def get_available_providers(cls) -> List[ILLMProvider]:
        """
        Get all providers that are accessible and ready.

        This performs health checks, so may be slower than get_configured_providers().

        Returns:
            List[ILLMProvider]: Providers with is_available() == True
        """
        cls._ensure_initialized()
        return [p for p in cls._providers.values() if p.is_available()]

    @classmethod
    def get_provider_names(cls) -> List[str]:
        """
        Get list of all registered provider names.

        Returns:
            List[str]: Provider names (for tab completion)
        """
        cls._ensure_initialized()
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a provider is registered.

        Args:
            name: Provider name

        Returns:
            bool: True if provider is registered
        """
        cls._ensure_initialized()
        return name in cls._providers

    @classmethod
    def _ensure_initialized(cls) -> None:
        """
        Ensure providers are initialized (lazy loading).

        This triggers import of concrete providers if not already done.
        """
        if cls._initialized:
            return

        # Import concrete providers to trigger registration
        try:
            from lobster.config.providers import anthropic_provider  # noqa: F401
        except ImportError:
            logger.debug("AnthropicProvider not available")

        try:
            from lobster.config.providers import bedrock_provider  # noqa: F401
        except ImportError:
            logger.debug("BedrockProvider not available")

        try:
            from lobster.config.providers import ollama_provider  # noqa: F401
        except ImportError:
            logger.debug("OllamaProvider not available")

        try:
            from lobster.config.providers import gemini_provider  # noqa: F401
        except ImportError:
            logger.debug("GeminiProvider not available")

        cls._initialized = True

    @classmethod
    def reset(cls) -> None:
        """
        Reset registry (for testing).

        Warning: This clears all registered providers!
        """
        cls._providers.clear()
        cls._initialized = False

    @classmethod
    def get_all_models_with_pricing(cls) -> Dict[str, Dict[str, float]]:
        """
        Generate unified pricing dictionary from all registered providers.

        This is the single source of truth for model pricing, collecting
        pricing data from each provider's ModelInfo definitions.

        Returns:
            Dict mapping model names to pricing info:
            {
                "gemini-3-pro-preview": {
                    "input_per_million": 2.00,
                    "output_per_million": 12.00,
                    "display_name": "Gemini 3 Pro"
                },
                ...
            }

        Note:
            - Models with pricing=None or pricing=0.0 are excluded (e.g., Ollama)
            - This method triggers provider initialization if not already done

        Example:
            >>> pricing = ProviderRegistry.get_all_models_with_pricing()
            >>> cost_info = pricing.get("gemini-3-pro-preview")
            >>> if cost_info:
            ...     print(f"Input: ${cost_info['input_per_million']}/M tokens")
        """
        cls._ensure_initialized()

        pricing_dict: Dict[str, Dict[str, float]] = {}

        for provider in cls._providers.values():
            # Skip local providers (Ollama) - they have no cost and require network calls
            # to list models. Cloud providers have static MODELS catalogs.
            if provider.name == "ollama":
                continue

            try:
                # Use static MODELS attribute if available (avoids unnecessary calls)
                if hasattr(provider, "MODELS"):
                    models = provider.MODELS
                else:
                    models = provider.list_models()

                for model in models:
                    # Only include models with defined pricing
                    if (
                        model.input_cost_per_million is not None
                        and model.input_cost_per_million > 0
                    ):
                        pricing_dict[model.name] = {
                            "input_per_million": model.input_cost_per_million,
                            "output_per_million": model.output_cost_per_million or 0.0,
                            "display_name": model.display_name,
                        }
            except Exception as e:
                # Log but don't fail - some providers may not be available
                logger.debug(f"Could not get models from {provider.name}: {e}")

        return pricing_dict


def get_provider(name: str) -> Optional[ILLMProvider]:
    """
    Convenience function to get a provider by name.

    Args:
        name: Provider name (e.g., "anthropic", "bedrock", "ollama")

    Returns:
        Optional[ILLMProvider]: Provider instance or None

    Example:
        >>> provider = get_provider("anthropic")
        >>> if provider and provider.is_available():
        ...     llm = provider.create_chat_model("claude-sonnet-4-20250514")
    """
    return ProviderRegistry.get(name)
