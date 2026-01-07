"""
Ollama service for model discovery and management.

This module provides utilities for interacting with a local Ollama instance,
including model enumeration, metadata retrieval, and validation.

Example:
    >>> from lobster.config.ollama_service import OllamaService
    >>>
    >>> # List available models
    >>> models = OllamaService.list_models()
    >>> for model in models:
    ...     print(f"{model.name} ({model.size_human})")
    >>>
    >>> # Validate model exists
    >>> if OllamaService.validate_model("llama3:70b-instruct"):
    ...     print("Model is installed")
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OllamaModelInfo:
    """
    Information about an Ollama model.

    Attributes:
        name: Model name with tag (e.g., "llama3:70b-instruct")
        size_bytes: Model size in bytes
        modified_at: Last modification timestamp
        family: Model family (e.g., "llama", "mixtral")
        parameter_size: Parameter count (e.g., "70B", "8x7B")
        digest: Model digest hash
    """

    name: str
    size_bytes: int
    modified_at: str
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    digest: Optional[str] = None

    @property
    def size_human(self) -> str:
        """
        Return human-readable size (e.g., "40GB", "4.7GB").

        Returns:
            str: Formatted size string
        """
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}PB"

    @property
    def description(self) -> str:
        """
        Generate a description for the model.

        Returns:
            str: Model description with size and parameters
        """
        parts = [self.size_human]
        if self.parameter_size:
            parts.append(f"{self.parameter_size} params")
        if self.family:
            parts.append(f"({self.family} family)")
        return " - ".join(parts)


class OllamaService:
    """Service for Ollama model discovery and management."""

    @staticmethod
    def is_available(base_url: str = "http://localhost:11434") -> bool:
        """
        Check if Ollama server is accessible.

        Args:
            base_url: Ollama server URL

        Returns:
            bool: True if server is accessible

        Example:
            >>> if OllamaService.is_available():
            ...     models = OllamaService.list_models()
        """
        try:
            import requests

            response = requests.get(f"{base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama server not accessible: {e}")
            return False

    @staticmethod
    def list_models(base_url: str = "http://localhost:11434") -> List[OllamaModelInfo]:
        """
        List all available models from Ollama instance.

        Args:
            base_url: Ollama server URL

        Returns:
            List[OllamaModelInfo]: List of model information objects

        Example:
            >>> models = OllamaService.list_models()
            >>> for model in models:
            ...     print(f"{model.name}: {model.description}")
        """
        try:
            import requests

            response = requests.get(f"{base_url}/api/tags", timeout=5)

            if response.status_code != 200:
                logger.warning(f"Ollama API returned status {response.status_code}")
                return []

            data = response.json()
            models = data.get("models", [])

            model_infos = []
            for model in models:
                # Extract details
                details = model.get("details", {})

                model_info = OllamaModelInfo(
                    name=model.get("name", "unknown"),
                    size_bytes=model.get("size", 0),
                    modified_at=model.get("modified_at", ""),
                    family=details.get("family"),
                    parameter_size=details.get("parameter_size"),
                    digest=model.get("digest"),
                )
                model_infos.append(model_info)

            # Sort by size (largest first) for better UX
            model_infos.sort(key=lambda m: m.size_bytes, reverse=True)

            return model_infos

        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    @staticmethod
    def get_model_names(base_url: str = "http://localhost:11434") -> List[str]:
        """
        Get list of model names (for tab completion).

        Args:
            base_url: Ollama server URL

        Returns:
            List[str]: List of model names

        Example:
            >>> names = OllamaService.get_model_names()
            >>> print(names)
            ['llama3:70b-instruct', 'mixtral:8x7b-instruct', 'llama3:8b-instruct']
        """
        models = OllamaService.list_models(base_url)
        return [model.name for model in models]

    @staticmethod
    def validate_model(
        model_name: str, base_url: str = "http://localhost:11434"
    ) -> bool:
        """
        Check if a model exists locally.

        Args:
            model_name: Model name to validate
            base_url: Ollama server URL

        Returns:
            bool: True if model exists

        Example:
            >>> if OllamaService.validate_model("llama3:70b-instruct"):
            ...     print("Model is installed")
            ... else:
            ...     print("Model not found")
        """
        available_models = OllamaService.get_model_names(base_url)
        return model_name in available_models

    @staticmethod
    def get_model_info(
        model_name: str, base_url: str = "http://localhost:11434"
    ) -> Optional[OllamaModelInfo]:
        """
        Get detailed information about a specific model.

        Args:
            model_name: Model name to query
            base_url: Ollama server URL

        Returns:
            Optional[OllamaModelInfo]: Model info or None if not found

        Example:
            >>> info = OllamaService.get_model_info("llama3:70b-instruct")
            >>> if info:
            ...     print(f"Size: {info.size_human}")
            ...     print(f"Parameters: {info.parameter_size}")
        """
        models = OllamaService.list_models(base_url)
        for model in models:
            if model.name == model_name:
                return model
        return None

    @staticmethod
    def suggest_model_for_agent(agent_name: str) -> str:
        """
        Recommend a model based on agent complexity.

        Args:
            agent_name: Name of agent

        Returns:
            str: Recommended model preference (large/medium/small)

        Example:
            >>> suggestion = OllamaService.suggest_model_for_agent("supervisor")
            >>> print(suggestion)  # "large"
        """
        # High-complexity agents need larger models (FREE tier only)
        # Premium agents (custom_feature_agent, machine_learning_expert_agent,
        # proteomics_expert, etc.) get model recommendations from custom packages.
        high_complexity = [
            "supervisor",
        ]

        # Medium-complexity agents (FREE tier only)
        medium_complexity = [
            "transcriptomics_expert",
            "de_analysis_expert",
        ]

        if agent_name in high_complexity:
            return "large"
        elif agent_name in medium_complexity:
            return "medium"
        else:
            return "small"

    @staticmethod
    def select_best_model(
        preference: str = "auto", base_url: str = "http://localhost:11434"
    ) -> Optional[str]:
        """
        Auto-select best available model based on preference.

        Args:
            preference: Model size preference (auto/large/medium/small)
            base_url: Ollama server URL

        Returns:
            Optional[str]: Best model name or None if no models found

        Example:
            >>> model = OllamaService.select_best_model("large")
            >>> print(model)  # "llama3:70b-instruct"
        """
        models = OllamaService.list_models(base_url)

        if not models:
            return None

        # Filter by instruct/chat models (better for agents)
        instruct_models = [
            m for m in models if "instruct" in m.name.lower() or "chat" in m.name.lower()
        ]

        if not instruct_models:
            instruct_models = models  # Fallback to all models

        if preference == "large":
            # Return largest model
            return instruct_models[0].name

        elif preference == "small":
            # Return smallest model
            return instruct_models[-1].name

        elif preference == "medium":
            # Return middle-sized model
            mid_index = len(instruct_models) // 2
            return instruct_models[mid_index].name

        else:  # auto
            # Smart selection: prefer 70B > 8x7B > 13B > 8B
            for model in instruct_models:
                if "70b" in model.name.lower():
                    return model.name

            for model in instruct_models:
                if "8x7b" in model.name.lower():
                    return model.name

            for model in instruct_models:
                if "13b" in model.name.lower():
                    return model.name

            # Fallback to largest available
            return instruct_models[0].name if instruct_models else None
