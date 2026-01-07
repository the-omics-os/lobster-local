"""
Unified model service for provider-aware model discovery and management.

This module provides model catalogs and discovery for all supported LLM providers:
- Anthropic: Static catalog of Claude models
- Bedrock: Static catalog of AWS Bedrock model IDs
- Ollama: Dynamic discovery via local HTTP API

Example:
    >>> from lobster.config.model_service import ModelServiceFactory
    >>>
    >>> # Get models for current provider
    >>> service = ModelServiceFactory.get_service("anthropic")
    >>> models = service.list_models()
    >>> for model in models:
    ...     print(f"{model.name}: {model.description}")
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

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
        context_window: Maximum context window size
        is_default: Whether this is the provider's default model
    """

    name: str
    display_name: str
    description: str
    provider: str
    context_window: Optional[int] = None
    is_default: bool = False


class BaseModelService(ABC):
    """Abstract base class for provider-specific model services."""

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """List all available models for this provider."""
        pass

    @abstractmethod
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model."""
        pass

    @abstractmethod
    def validate_model(self, model_name: str) -> bool:
        """Check if a model name is valid for this provider."""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default/recommended model for this provider."""
        pass

    def get_model_names(self) -> List[str]:
        """Get list of model names for tab completion."""
        return [model.name for model in self.list_models()]


class AnthropicModelService(BaseModelService):
    """
    Model service for Anthropic Direct API.

    Provides static catalog of available Claude models with their capabilities.
    Models are ordered by capability (most capable first).
    """

    # Static model catalog - update when Anthropic releases new models
    MODELS = [
        ModelInfo(
            name="claude-sonnet-4-20250514",
            display_name="Claude Sonnet 4",
            description="Latest Sonnet - best balance of speed and capability",
            provider="anthropic",
            context_window=200000,
            is_default=True,
        ),
        ModelInfo(
            name="claude-opus-4-20250514",
            display_name="Claude Opus 4",
            description="Most capable model - complex reasoning and analysis",
            provider="anthropic",
            context_window=200000,
        ),
        ModelInfo(
            name="claude-3-5-sonnet-20241022",
            display_name="Claude 3.5 Sonnet",
            description="Previous generation Sonnet - fast and capable",
            provider="anthropic",
            context_window=200000,
        ),
        ModelInfo(
            name="claude-3-5-haiku-20241022",
            display_name="Claude 3.5 Haiku",
            description="Fastest model - quick tasks and high throughput",
            provider="anthropic",
            context_window=200000,
        ),
        ModelInfo(
            name="claude-3-opus-20240229",
            display_name="Claude 3 Opus",
            description="Previous Opus - complex analysis (legacy)",
            provider="anthropic",
            context_window=200000,
        ),
        ModelInfo(
            name="claude-3-haiku-20240307",
            display_name="Claude 3 Haiku",
            description="Claude 3 Haiku - fast responses (legacy)",
            provider="anthropic",
            context_window=200000,
        ),
    ]

    def list_models(self) -> List[ModelInfo]:
        """List all available Anthropic models."""
        return self.MODELS.copy()

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific Anthropic model."""
        for model in self.MODELS:
            if model.name == model_name:
                return model
        return None

    def validate_model(self, model_name: str) -> bool:
        """Check if model name is a valid Anthropic model."""
        return model_name in [m.name for m in self.MODELS]

    def get_default_model(self) -> str:
        """Get the default Anthropic model (Claude Sonnet 4)."""
        for model in self.MODELS:
            if model.is_default:
                return model.name
        return self.MODELS[0].name  # Fallback to first


class BedrockModelService(BaseModelService):
    """
    Model service for AWS Bedrock.

    Provides static catalog of Claude models available through Bedrock.
    Model IDs follow Bedrock's naming convention.
    """

    # Static model catalog for Bedrock
    # Note: Availability may vary by AWS region
    MODELS = [
        ModelInfo(
            name="anthropic.claude-sonnet-4-20250514-v1:0",
            display_name="Claude Sonnet 4 (Bedrock)",
            description="Latest Sonnet via Bedrock - best balance",
            provider="bedrock",
            context_window=200000,
            is_default=True,
        ),
        ModelInfo(
            name="anthropic.claude-opus-4-20250514-v1:0",
            display_name="Claude Opus 4 (Bedrock)",
            description="Most capable via Bedrock - complex reasoning",
            provider="bedrock",
            context_window=200000,
        ),
        ModelInfo(
            name="anthropic.claude-3-5-sonnet-20241022-v2:0",
            display_name="Claude 3.5 Sonnet v2 (Bedrock)",
            description="Claude 3.5 Sonnet - fast and capable",
            provider="bedrock",
            context_window=200000,
        ),
        ModelInfo(
            name="anthropic.claude-3-5-haiku-20241022-v1:0",
            display_name="Claude 3.5 Haiku (Bedrock)",
            description="Fastest Claude via Bedrock",
            provider="bedrock",
            context_window=200000,
        ),
        ModelInfo(
            name="anthropic.claude-3-opus-20240229-v1:0",
            display_name="Claude 3 Opus (Bedrock)",
            description="Claude 3 Opus via Bedrock (legacy)",
            provider="bedrock",
            context_window=200000,
        ),
        ModelInfo(
            name="anthropic.claude-3-sonnet-20240229-v1:0",
            display_name="Claude 3 Sonnet (Bedrock)",
            description="Claude 3 Sonnet via Bedrock (legacy)",
            provider="bedrock",
            context_window=200000,
        ),
        ModelInfo(
            name="anthropic.claude-3-haiku-20240307-v1:0",
            display_name="Claude 3 Haiku (Bedrock)",
            description="Claude 3 Haiku via Bedrock (legacy)",
            provider="bedrock",
            context_window=200000,
        ),
    ]

    def list_models(self) -> List[ModelInfo]:
        """List all available Bedrock models."""
        return self.MODELS.copy()

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific Bedrock model."""
        for model in self.MODELS:
            if model.name == model_name:
                return model
        return None

    def validate_model(self, model_name: str) -> bool:
        """Check if model name is a valid Bedrock model ID."""
        return model_name in [m.name for m in self.MODELS]

    def get_default_model(self) -> str:
        """Get the default Bedrock model."""
        for model in self.MODELS:
            if model.is_default:
                return model.name
        return self.MODELS[0].name


class OllamaModelServiceAdapter(BaseModelService):
    """
    Adapter for OllamaService to conform to BaseModelService interface.

    Wraps the existing OllamaService for dynamic model discovery.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def list_models(self) -> List[ModelInfo]:
        """List all available Ollama models via HTTP API."""
        try:
            from lobster.config.ollama_service import OllamaService

            ollama_models = OllamaService.list_models(self.base_url)

            return [
                ModelInfo(
                    name=m.name,
                    display_name=m.name,
                    description=m.description,
                    provider="ollama",
                    context_window=None,  # Ollama doesn't expose this
                    is_default=(i == 0),  # First (largest) is default
                )
                for i, m in enumerate(ollama_models)
            ]
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific Ollama model."""
        models = self.list_models()
        for model in models:
            if model.name == model_name:
                return model
        return None

    def validate_model(self, model_name: str) -> bool:
        """Check if model is available in local Ollama."""
        try:
            from lobster.config.ollama_service import OllamaService

            return OllamaService.validate_model(model_name, self.base_url)
        except Exception:
            return False

    def get_default_model(self) -> str:
        """Get the default (largest) Ollama model."""
        try:
            from lobster.config.ollama_service import OllamaService

            model = OllamaService.select_best_model("auto", self.base_url)
            return model or "llama3:8b-instruct"
        except Exception:
            return "llama3:8b-instruct"

    def is_available(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            from lobster.config.ollama_service import OllamaService

            return OllamaService.is_available(self.base_url)
        except Exception:
            return False


class ModelServiceFactory:
    """
    Factory for creating provider-specific model services.

    Example:
        >>> service = ModelServiceFactory.get_service("anthropic")
        >>> models = service.list_models()
        >>> print(service.get_default_model())
    """

    _services = {
        "anthropic": AnthropicModelService,
        "bedrock": BedrockModelService,
        "ollama": OllamaModelServiceAdapter,
    }

    @classmethod
    def get_service(cls, provider: str, **kwargs) -> BaseModelService:
        """
        Get the model service for a specific provider.

        Args:
            provider: Provider name (anthropic | bedrock | ollama)
            **kwargs: Additional arguments (e.g., base_url for Ollama)

        Returns:
            BaseModelService: Provider-specific model service

        Raises:
            ValueError: If provider is not supported
        """
        if provider not in cls._services:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported: {', '.join(cls._services.keys())}"
            )

        service_class = cls._services[provider]
        return service_class(**kwargs)

    @classmethod
    def get_supported_providers(cls) -> List[str]:
        """Get list of supported providers."""
        return list(cls._services.keys())

    @classmethod
    def list_all_models(cls) -> List[ModelInfo]:
        """List models from all providers (for global search)."""
        all_models = []
        for provider in cls._services:
            try:
                service = cls.get_service(provider)
                all_models.extend(service.list_models())
            except Exception as e:
                logger.debug(f"Failed to list models for {provider}: {e}")
        return all_models
