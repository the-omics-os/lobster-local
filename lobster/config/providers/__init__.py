"""
LLM Provider abstraction layer for Lobster AI.

This module provides a clean, extensible interface for LLM providers,
enabling easy addition of new providers (OpenAI, Nebius, etc.) in the future.

Architecture:
    ILLMProvider (ABC) - Interface all providers implement
    ProviderRegistry - Singleton registry for provider discovery
    Concrete providers - AnthropicProvider, BedrockProvider, OllamaProvider

Usage:
    >>> from lobster.config.providers import ProviderRegistry, get_provider
    >>>
    >>> # Get a registered provider
    >>> provider = get_provider("anthropic")
    >>> if provider.is_available():
    ...     llm = provider.create_chat_model("claude-sonnet-4-20250514")
    >>>
    >>> # List all configured providers
    >>> for p in ProviderRegistry.get_configured_providers():
    ...     print(f"{p.name}: {p.get_default_model()}")
"""

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo
from lobster.config.providers.registry import ProviderRegistry, get_provider

__all__ = [
    "ILLMProvider",
    "ModelInfo",
    "ProviderRegistry",
    "get_provider",
]
