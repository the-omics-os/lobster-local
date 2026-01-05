"""
Google Gemini provider implementation.

This module provides Google Gemini integration for Lobster's LLM system,
enabling access to Gemini 3.0 models via Google's Developer API.

Architecture:
    - Implements ILLMProvider interface for consistency
    - Uses static model catalog (Gemini 3 Pro + Flash)
    - Creates ChatGoogleGenerativeAI instances via LangChain

Example:
    >>> from lobster.config.providers.gemini_provider import GeminiProvider
    >>> provider = GeminiProvider()
    >>> if provider.is_configured():
    ...     models = provider.list_models()
    ...     llm = provider.create_chat_model("gemini-3-pro-preview")
"""

import logging
import os
from typing import Any, List, Optional

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

logger = logging.getLogger(__name__)


class GeminiProvider(ILLMProvider):
    """
    Google Gemini provider for Gemini 3.0 models.

    Provides access to Gemini models through Google's Developer API.
    Requires GOOGLE_API_KEY environment variable.

    Features:
        - Static model catalog with Gemini 3.0 models
        - Automatic model validation
        - ChatGoogleGenerativeAI integration
        - Temperature enforcement for Gemini 3+ (requires 1.0)

    Configuration:
        GOOGLE_API_KEY: Google API key (primary)
        GEMINI_API_KEY: Fallback API key (langchain compatibility)

    Usage:
        >>> provider = GeminiProvider()
        >>> if not provider.is_configured():
        ...     print("Set GOOGLE_API_KEY in .env")
        >>> models = provider.list_models()
        >>> llm = provider.create_chat_model(models[0].name)
    """

    # Static model catalog - Gemini 3.0 models
    # Source: https://ai.google.dev/gemini-api/docs/pricing
    MODELS = [
        ModelInfo(
            name="gemini-3-pro-preview",
            display_name="Gemini 3 Pro",
            description="Latest Gemini - best balance of speed and capability",
            provider="gemini",
            context_window=200000,
            is_default=True,
            input_cost_per_million=2.00,  # $2.00 for prompts ≤200k tokens, $4.00 for >200k
            output_cost_per_million=12.00,  # $12.00 for prompts ≤200k tokens, $18.00 for >200k
        ),
        ModelInfo(
            name="gemini-3-flash-preview",
            display_name="Gemini 3 Flash",
            description="Fastest Gemini - quick tasks with thinking support (free tier available)",
            provider="gemini",
            context_window=200000,
            is_default=False,
            input_cost_per_million=0.50,  # $0.50 text/image/video, $1.00 audio; free tier available
            output_cost_per_million=3.00,  # $3.00 (free tier available)
        ),
    ]

    @property
    def name(self) -> str:
        """
        Provider identifier.

        Returns:
            str: "gemini"
        """
        return "gemini"

    @property
    def display_name(self) -> str:
        """
        Human-friendly provider name.

        Returns:
            str: "Google Gemini"
        """
        return "Google Gemini"

    def is_configured(self) -> bool:
        """
        Check if Google API key is present.

        Checks for GOOGLE_API_KEY (primary) or GEMINI_API_KEY (fallback).
        Does NOT validate the key (use is_available() for that).

        Returns:
            bool: True if API key is set

        Example:
            >>> provider = GeminiProvider()
            >>> if not provider.is_configured():
            ...     print("Set GOOGLE_API_KEY=... in .env")
        """
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        return bool(api_key and api_key.strip())

    def is_available(self) -> bool:
        """
        Check if Gemini API is accessible.

        For cloud providers like Gemini, availability equals configuration
        (no local service to health-check). Actual key validation happens
        at first API call.

        Returns:
            bool: True if configured (cloud always available if configured)

        Note:
            Unlike Ollama, we don't ping the API here to avoid unnecessary
            latency and API charges. Invalid keys fail gracefully at runtime.
        """
        return self.is_configured()

    def list_models(self) -> List[ModelInfo]:
        """
        List all available Gemini models.

        Returns static catalog of Gemini models.
        Models are ordered by capability (most capable first).

        Returns:
            List[ModelInfo]: Available Gemini models

        Example:
            >>> provider = GeminiProvider()
            >>> for model in provider.list_models():
            ...     print(f"{model.display_name}: {model.description}")
        """
        return self.MODELS.copy()

    def get_default_model(self) -> str:
        """
        Get the recommended default model.

        Returns:
            str: "gemini-3-pro-preview" (Gemini 3 Pro)

        Example:
            >>> provider = GeminiProvider()
            >>> model_id = provider.get_default_model()
            >>> llm = provider.create_chat_model(model_id)
        """
        for model in self.MODELS:
            if model.is_default:
                return model.name
        return self.MODELS[0].name  # Fallback to first model

    def validate_model(self, model_id: str) -> bool:
        """
        Check if a model ID is valid for Gemini.

        Args:
            model_id: Model identifier (e.g., "gemini-3-pro-preview")

        Returns:
            bool: True if model exists in catalog

        Example:
            >>> provider = GeminiProvider()
            >>> if provider.validate_model("gemini-3-pro-preview"):
            ...     llm = provider.create_chat_model("gemini-3-pro-preview")
        """
        return model_id in [m.name for m in self.MODELS]

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """
        Create a ChatGoogleGenerativeAI instance.

        Instantiates a LangChain ChatGoogleGenerativeAI model with the specified
        parameters. Uses GOOGLE_API_KEY from environment.

        Args:
            model_id: Model identifier (must be valid Gemini model)
            temperature: Sampling temperature (0.0-2.0, default 1.0)
                WARNING: Gemini 3.0+ requires temperature=1.0 (lower values
                can cause infinite loops per Google documentation)
            max_tokens: Maximum tokens in response (default 4096)
            **kwargs: Additional ChatGoogleGenerativeAI parameters
                - api_key: Override API key

        Returns:
            ChatGoogleGenerativeAI: Configured LangChain chat model

        Raises:
            ImportError: If langchain-google-genai not installed
            ValueError: If API key not found

        Example:
            >>> provider = GeminiProvider()
            >>> llm = provider.create_chat_model(
            ...     "gemini-3-pro-preview",
            ...     temperature=1.0,
            ...     max_tokens=8192
            ... )
            >>> response = llm.invoke("Hello!")

        Notes:
            - Reads GOOGLE_API_KEY from environment by default
            - Override with api_key kwarg if needed
            - Temperature is enforced to 1.0 for Gemini 3+ models
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai package not installed. "
                "Install with: pip install langchain-google-genai"
            )

        # Validate model ID
        if not self.validate_model(model_id):
            logger.warning(
                f"Model '{model_id}' not in Gemini catalog. "
                f"Proceeding anyway (may fail at runtime)."
            )

        # Get API key (prefer kwarg, fallback to environment)
        api_key = kwargs.pop("api_key", None) or os.environ.get(
            "GOOGLE_API_KEY"
        ) or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment or kwargs. "
                "Set it with: export GOOGLE_API_KEY=your-key-here"
            )

        # CRITICAL: Gemini 3+ requires temperature=1.0 (lower values cause infinite loops)
        if "gemini-3" in model_id and temperature != 1.0:
            logger.warning(
                f"Gemini 3.0+ models require temperature=1.0 (you set {temperature}). "
                f"Lower values can cause infinite loops. Overriding to 1.0."
            )
            temperature = 1.0

        # Create ChatGoogleGenerativeAI instance
        # Note: Do NOT use **kwargs here - Google client doesn't support parameters
        # like max_retries that LangChain might try to pass
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens else None,
            include_thoughts=True,  # Enable thought signature handling for reasoning/function calling
        )

    def get_configuration_help(self) -> str:
        """
        Get help text for configuring Gemini provider.

        Returns:
            str: Configuration instructions for the user

        Example:
            >>> provider = GeminiProvider()
            >>> if not provider.is_configured():
            ...     print(provider.get_configuration_help())
        """
        return (
            "Configure Google Gemini:\n\n"
            "1. Get API key from: https://aistudio.google.com/apikey\n"
            "2. Set environment variable:\n"
            "   export GOOGLE_API_KEY=your-key-here\n\n"
            "Or add to .env file:\n"
            "   GOOGLE_API_KEY=your-key-here\n\n"
            f"Default model: {self.get_default_model()}\n"
            f"Available models: {', '.join(self.get_model_names())}\n\n"
            "Note: Gemini 3.0+ models require temperature=1.0."
        )


# Auto-register provider with registry
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(GeminiProvider())
