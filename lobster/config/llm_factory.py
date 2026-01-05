"""
LLM Factory for provider-agnostic model instantiation.

This module provides a unified interface for creating LLM instances
using the ProviderRegistry. All provider-specific logic is delegated
to the individual provider implementations.

Architecture:
    LLMFactory → ConfigResolver → ProviderRegistry → ILLMProvider

Usage:
    >>> from lobster.config.llm_factory import LLMFactory, create_llm
    >>>
    >>> # Simple usage (uses ConfigResolver)
    >>> llm = create_llm("supervisor", {"temperature": 0.7})
    >>>
    >>> # With explicit overrides
    >>> llm = create_llm("supervisor", {}, provider_override="anthropic", model_override="claude-sonnet-4")
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """
    Supported LLM providers.

    Note: This enum is kept for backward compatibility. New code should use
    the string provider names directly with ProviderRegistry.
    """

    ANTHROPIC_DIRECT = "anthropic"
    BEDROCK_ANTHROPIC = "bedrock"
    OLLAMA = "ollama"
    GEMINI = "gemini"


class LLMFactory:
    """
    Factory for creating provider-agnostic LLM instances.

    This factory delegates to ProviderRegistry for actual model creation.
    It uses ConfigResolver to determine which provider and model to use
    based on the 3-layer priority system.

    Example:
        >>> llm = LLMFactory.create_llm(
        ...     model_config={"temperature": 0.7},
        ...     agent_name="supervisor",
        ...     workspace_path=Path(".lobster_workspace")
        ... )
    """

    @classmethod
    def create_llm(
        cls,
        model_config: Dict[str, Any],
        agent_name: Optional[str] = None,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None,
        workspace_path: Optional[Path] = None,
    ) -> Any:
        """
        Create an LLM instance based on configuration.

        Uses ConfigResolver to determine provider and model, then delegates
        to the appropriate provider implementation via ProviderRegistry.

        Args:
            model_config: Configuration dictionary with model parameters
                - temperature: float (default: 1.0)
                - max_tokens: int (default: 4096)
                - model_id: str (optional, can be overridden)
            agent_name: Optional agent name for logging and per-agent config
            provider_override: Explicit provider name from CLI flag
            model_override: Explicit model name from CLI flag
            workspace_path: Optional workspace path for ConfigResolver

        Returns:
            LLM instance (ChatAnthropic, ChatBedrockConverse, ChatOllama, etc.)

        Raises:
            ConfigurationError: If no provider is configured
            ValueError: If provider is not registered

        Example:
            >>> llm = LLMFactory.create_llm(
            ...     {"temperature": 0.7, "max_tokens": 8192},
            ...     agent_name="supervisor",
            ...     provider_override="anthropic"
            ... )
        """
        from lobster.config.providers import ProviderRegistry, get_provider
        from lobster.core.config_resolver import ConfigResolver, ConfigurationError

        # Get ConfigResolver instance
        resolver = ConfigResolver.get_instance(workspace_path)

        # Resolve provider using 3-layer priority
        try:
            provider_name, provider_source = resolver.resolve_provider(
                runtime_override=provider_override
            )
        except ConfigurationError:
            # Re-raise with context
            raise

        # Get provider from registry
        provider = get_provider(provider_name)
        if not provider:
            raise ValueError(
                f"Provider '{provider_name}' is not registered. "
                f"Available providers: {', '.join(ProviderRegistry.get_provider_names())}"
            )

        # Resolve model using priority (or use override directly)
        if model_override:
            model_id = model_override
            model_source = "runtime flag --model"
        elif model_config.get("model_id"):
            # Model specified in config (from profile system)
            model_id = model_config["model_id"]
            model_source = "model_config"
        else:
            # Resolve from workspace config or use provider default
            model_id, model_source = resolver.resolve_model(
                agent_name=agent_name,
                provider=provider_name,
            )
            if not model_id:
                model_id = provider.get_default_model()
                model_source = f"provider default ({provider_name})"

        # Log resolution
        if agent_name:
            logger.debug(
                f"Creating LLM for '{agent_name}': "
                f"provider={provider_name} ({provider_source}), "
                f"model={model_id} ({model_source})"
            )

        # Extract parameters
        temperature = model_config.get("temperature", 1.0)
        max_tokens = model_config.get("max_tokens", 4096)

        # Extract additional model request fields (thinking config, etc.)
        additional_fields = model_config.get("additional_model_request_fields", {})

        # Create model via provider
        # Pass additional_fields as kwargs to support provider-specific features:
        # - Bedrock: thinking config via additional_model_request_fields
        # - Anthropic: extended thinking via additional_model_request_fields
        return provider.create_chat_model(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **additional_fields,  # Pass thinking config and other provider-specific params
        )

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get list of providers that have valid configuration.

        Returns:
            List of provider names with is_configured() == True
        """
        from lobster.config.providers import ProviderRegistry

        return [p.name for p in ProviderRegistry.get_configured_providers()]

    @classmethod
    def get_current_provider(cls, workspace_path: Optional[Path] = None) -> Optional[str]:
        """
        Get the currently configured provider.

        Args:
            workspace_path: Optional workspace path for ConfigResolver

        Returns:
            Provider name or None if not configured
        """
        from lobster.core.config_resolver import ConfigResolver, ConfigurationError

        try:
            resolver = ConfigResolver.get_instance(workspace_path)
            provider_name, _ = resolver.resolve_provider()
            return provider_name
        except ConfigurationError:
            return None

    @classmethod
    def is_provider_available(cls, provider_name: str) -> bool:
        """
        Check if a specific provider is available.

        Args:
            provider_name: Provider to check (anthropic, bedrock, ollama)

        Returns:
            bool: True if provider is available
        """
        from lobster.config.providers import get_provider

        provider = get_provider(provider_name)
        return provider.is_available() if provider else False


# Convenience function for backward compatibility
def create_llm(
    agent_name: str,
    model_params: Dict[str, Any],
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    workspace_path: Optional[Path] = None,
) -> Any:
    """
    Create an LLM instance for a specific agent.

    This is a convenience function that maintains backward compatibility
    with the existing agent code.

    Args:
        agent_name: Name of the agent requesting the LLM
        model_params: Model configuration parameters
        provider_override: Optional explicit provider name (e.g., from CLI flag)
        model_override: Optional explicit model name (e.g., from CLI flag)
        workspace_path: Optional workspace path for configuration resolution

    Returns:
        LLM instance configured for the agent

    Example:
        >>> llm = create_llm("supervisor", {"temperature": 0.7})
    """
    return LLMFactory.create_llm(
        model_config=model_params,
        agent_name=agent_name,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
    )
