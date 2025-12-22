"""
Ollama provider for local LLM inference.

This provider enables Lobster to use locally-hosted Ollama models,
providing privacy, zero cost, and offline capability. It dynamically
discovers available models and auto-selects the best one based on size
and capability heuristics.

Example:
    >>> from lobster.config.providers import get_provider
    >>>
    >>> provider = get_provider("ollama")
    >>> if provider and provider.is_available():
    ...     models = provider.list_models()
    ...     llm = provider.create_chat_model(provider.get_default_model())
"""

import logging
import os
import re
from typing import Any, List, Optional

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

logger = logging.getLogger(__name__)


class OllamaProvider(ILLMProvider):
    """
    Provider for local Ollama models.

    Ollama enables running large language models locally with GPU acceleration.
    This provider dynamically discovers installed models and auto-selects the
    best available model based on size and capability heuristics.

    Key Features:
        - Dynamic model discovery via HTTP API
        - Smart model selection (prefers larger instruct models)
        - Zero-cost inference with local GPU
        - Privacy-focused (no data sent to cloud)
        - Offline capability

    Configuration:
        - OLLAMA_BASE_URL: Ollama server URL (default: "http://localhost:11434")
        - OLLAMA_DEFAULT_MODEL: Explicit model override (bypasses auto-selection)

    Auto-Selection Priority:
        1. OLLAMA_DEFAULT_MODEL environment variable (explicit override)
        2. Best available model by heuristic (70B > 8x7B > 13B > 8B)
        3. Fallback default: "gpt-oss:20b"

    Example:
        >>> provider = OllamaProvider()
        >>> if provider.is_available():
        ...     # List models
        ...     for model in provider.list_models():
        ...         print(f"{model.name}: {model.description}")
        ...
        ...     # Create LLM with auto-selected model
        ...     llm = provider.create_chat_model(
        ...         provider.get_default_model(),
        ...         temperature=0.7
        ...     )
    """

    def __init__(self):
        """Initialize Ollama provider with default configuration."""
        self._base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def name(self) -> str:
        """
        Provider identifier.

        Returns:
            str: Always returns "ollama"
        """
        return "ollama"

    @property
    def display_name(self) -> str:
        """
        Human-friendly provider name.

        Returns:
            str: Display name for UI/CLI
        """
        return "Ollama (Local)"

    def is_configured(self) -> bool:
        """
        Check if Ollama configuration is present.

        For Ollama, we consider it configured if either:
        1. OLLAMA_BASE_URL environment variable is set, OR
        2. Default localhost endpoint exists (always True)

        Returns:
            bool: Always True (Ollama uses default localhost if not configured)
        """
        # Ollama is considered "configured" if base URL is set or defaults to localhost
        return True

    def is_available(self) -> bool:
        """
        Check if Ollama server is accessible (health check).

        Performs an HTTP GET to /api/tags endpoint with 2-second timeout.
        This verifies the Ollama server is running and responsive.

        Returns:
            bool: True if Ollama server responds with 200 OK

        Example:
            >>> provider = OllamaProvider()
            >>> if provider.is_available():
            ...     print("Ollama is running")
            ... else:
            ...     print("Start Ollama: ollama serve")
        """
        try:
            import requests

            response = requests.get(f"{self._base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama server not accessible at {self._base_url}: {e}")
            return False

    def list_models(self) -> List[ModelInfo]:
        """
        List all available models from Ollama instance.

        Queries the /api/tags endpoint to dynamically discover installed models.
        Models are sorted by size (largest first) for better UX.

        Returns:
            List[ModelInfo]: Available models with metadata (name, size, parameters)

        Example:
            >>> provider = OllamaProvider()
            >>> for model in provider.list_models():
            ...     print(f"{model.name}: {model.description}")
            ...     # Output: "llama3:70b-instruct: 40.5GB - 70B params"
        """
        try:
            import requests

            response = requests.get(f"{self._base_url}/api/tags", timeout=5)

            if response.status_code != 200:
                logger.warning(
                    f"Ollama API returned status {response.status_code} at {self._base_url}"
                )
                return []

            data = response.json()
            models = data.get("models", [])

            model_infos = []
            for model in models:
                # Extract model details
                model_name = model.get("name", "unknown")
                size_bytes = model.get("size", 0)
                details = model.get("details", {})

                # Convert to ModelInfo
                model_info = ModelInfo(
                    name=model_name,
                    display_name=self._format_display_name(model_name),
                    description=self._generate_description(
                        size_bytes,
                        details.get("parameter_size"),
                        details.get("family"),
                    ),
                    provider="ollama",
                    context_window=self._estimate_context_window(model_name),
                    is_default=False,  # Set by get_default_model()
                    input_cost_per_million=0.0,  # Local inference is free
                    output_cost_per_million=0.0,
                )
                model_infos.append(model_info)

            # Sort by size (largest first) for better UX
            model_infos.sort(
                key=lambda m: self._extract_size_bytes_from_description(m.description),
                reverse=True,
            )

            # Mark default model
            default_model_name = self.get_default_model()
            for model in model_infos:
                if model.name == default_model_name:
                    model.is_default = True
                    break

            return model_infos

        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    def get_default_model(self) -> str:
        """
        Get the recommended default model for this provider.

        Selection strategy (in priority order):
        1. OLLAMA_DEFAULT_MODEL environment variable (explicit override)
        2. Best available model by heuristic (70B > 8x7B > 13B > 8B)
        3. Fallback default: "gpt-oss:20b"

        Heuristics:
        - Prefers instruct/chat models (better for agent tasks)
        - Prefers larger parameter counts (higher capability)
        - Prefers newer model versions (e.g., llama3 > llama2)

        Returns:
            str: Default model identifier

        Example:
            >>> provider = OllamaProvider()
            >>> model = provider.get_default_model()
            >>> print(model)  # "llama3:70b-instruct"
        """
        # 1. Check environment variable first
        env_model = os.environ.get("OLLAMA_DEFAULT_MODEL")
        if env_model:
            logger.debug(f"Using OLLAMA_DEFAULT_MODEL: {env_model}")
            return env_model

        # 2. Auto-select best available model
        models = self.list_models()

        if not models:
            logger.warning("No Ollama models detected, using default: gpt-oss:20b")
            return "gpt-oss:20b"

        # Filter to instruct/chat models (better for agents)
        instruct_models = [
            m
            for m in models
            if "instruct" in m.name.lower() or "chat" in m.name.lower()
        ]

        if not instruct_models:
            instruct_models = models  # Fallback to all models

        # Score models by quality heuristic
        best_model = max(instruct_models, key=lambda m: self._score_model(m.name))

        logger.info(
            f"Auto-selected Ollama model: {best_model.name} from {len(models)} available models"
        )
        return best_model.name

    def create_chat_model(
        self,
        model_id: str,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> Any:
        """
        Create a LangChain ChatOllama instance.

        Args:
            model_id: Ollama model identifier (e.g., "llama3:70b-instruct")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response (ignored by Ollama)
            **kwargs: Additional parameters for ChatOllama

        Returns:
            ChatOllama: LangChain chat model instance

        Raises:
            ImportError: If langchain-ollama package not installed

        Example:
            >>> provider = OllamaProvider()
            >>> llm = provider.create_chat_model(
            ...     "llama3:70b-instruct",
            ...     temperature=0.7
            ... )
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama package not installed. "
                "Please run: pip install langchain-ollama"
            )

        # Build parameters for ChatOllama
        ollama_params = {
            "model": model_id,
            "temperature": temperature,
            **kwargs,
        }

        # Add base URL if custom endpoint
        if self._base_url != "http://localhost:11434":
            ollama_params["base_url"] = self._base_url

        logger.debug(f"Creating ChatOllama with model '{model_id}'")
        return ChatOllama(**ollama_params)

    def validate_model(self, model_id: str) -> bool:
        """
        Check if a model exists locally in Ollama.

        Args:
            model_id: Model identifier to validate

        Returns:
            bool: True if model is installed

        Example:
            >>> provider = OllamaProvider()
            >>> if provider.validate_model("llama3:70b-instruct"):
            ...     print("Model is ready")
            ... else:
            ...     print("Pull model: ollama pull llama3:70b-instruct")
        """
        available_models = [m.name for m in self.list_models()]
        return model_id in available_models

    def get_configuration_help(self) -> str:
        """
        Get help text for configuring Ollama.

        Returns:
            str: Configuration instructions with quickstart commands
        """
        return (
            "Ollama (Local) Configuration:\n\n"
            "1. Install Ollama: https://ollama.ai/download\n"
            "2. Start Ollama: ollama serve\n"
            "3. Pull a model: ollama pull llama3:70b-instruct\n\n"
            "Environment Variables:\n"
            "  OLLAMA_BASE_URL: Server URL (default: http://localhost:11434)\n"
            "  OLLAMA_DEFAULT_MODEL: Model name (default: auto-select best)\n\n"
            "Current Configuration:\n"
            f"  Base URL: {self._base_url}\n"
            f"  Server Available: {self.is_available()}\n"
            f"  Default Model: {self.get_default_model()}"
        )

    # ---- Private Helper Methods ----

    def _format_display_name(self, model_name: str) -> str:
        """
        Format model name for display.

        Args:
            model_name: Raw model name (e.g., "llama3:70b-instruct")

        Returns:
            str: Human-friendly name (e.g., "Llama 3 70B Instruct")
        """
        # Replace colons/hyphens with spaces, title case
        display = model_name.replace(":", " ").replace("-", " ")
        return display.title()

    def _generate_description(
        self,
        size_bytes: int,
        parameter_size: Optional[str],
        family: Optional[str],
    ) -> str:
        """
        Generate model description from metadata.

        Args:
            size_bytes: Model size in bytes
            parameter_size: Parameter count (e.g., "70B")
            family: Model family (e.g., "llama")

        Returns:
            str: Formatted description (e.g., "40.5GB - 70B params (llama family)")
        """
        parts = [self._format_size(size_bytes)]

        if parameter_size:
            parts.append(f"{parameter_size} params")

        if family:
            parts.append(f"({family} family)")

        return " - ".join(parts)

    def _format_size(self, size_bytes: int) -> str:
        """
        Format bytes as human-readable size.

        Args:
            size_bytes: Size in bytes

        Returns:
            str: Formatted size (e.g., "40.5GB")
        """
        size = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}PB"

    def _extract_size_bytes_from_description(self, description: str) -> float:
        """
        Extract size in bytes from description for sorting.

        Args:
            description: Model description with size

        Returns:
            float: Size in bytes (0 if not found)
        """
        # Match patterns like "40.5GB", "4.7MB"
        match = re.search(r"(\d+\.?\d*)(GB|MB|KB|B)", description)
        if not match:
            return 0.0

        value = float(match.group(1))
        unit = match.group(2)

        # Convert to bytes
        multipliers = {
            "B": 1,
            "KB": 1024,
            "MB": 1024**2,
            "GB": 1024**3,
            "TB": 1024**4,
        }
        return value * multipliers.get(unit, 1)

    def _estimate_context_window(self, model_name: str) -> Optional[int]:
        """
        Estimate context window size from model name.

        Args:
            model_name: Model identifier

        Returns:
            Optional[int]: Context window size (tokens) or None
        """
        # Common context windows by model family
        model_lower = model_name.lower()

        if "llama3" in model_lower or "llama-3" in model_lower:
            return 8192  # Llama 3 has 8K context
        elif "mixtral" in model_lower:
            return 32768  # Mixtral has 32K context
        elif "gpt-oss" in model_lower:
            return 8192  # gpt-oss has 8K context
        else:
            return None  # Unknown

    def _score_model(self, model_name: str) -> int:
        """
        Score model by quality heuristic (higher = better).

        Scoring factors:
        - Instruct/chat models: +100 points
        - Parameter size: +1 point per billion parameters
        - Model version: +10 points per version number

        Args:
            model_name: Model identifier

        Returns:
            int: Quality score
        """
        score = 0
        model_lower = model_name.lower()

        # Prefer instruct/chat models
        if "instruct" in model_lower or "chat" in model_lower:
            score += 100

        # Size-based scoring (extract parameter count)
        # Examples: "llama3:8b", "gpt-oss:20b", "mixtral:8x7b"
        size_match = re.search(r"(\d+)(?:x)?(\d+)?b", model_lower)
        if size_match:
            main_size = int(size_match.group(1))
            multiplier = int(size_match.group(2)) if size_match.group(2) else 1
            total_params = main_size * multiplier
            score += total_params  # Larger models score higher

        # Prefer newer versions (llama3 > llama2)
        version_match = re.search(r"(\d+)", model_lower)
        if version_match:
            version = int(version_match.group(1))
            score += version * 10

        return score


# Auto-register with ProviderRegistry on import
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(OllamaProvider())
