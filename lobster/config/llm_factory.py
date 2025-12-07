"""
LLM Factory for provider-agnostic model instantiation.

This module provides a unified interface for creating LLM instances
regardless of the underlying provider (Claude API, AWS Bedrock, etc.).
"""

import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers."""

    ANTHROPIC_DIRECT = "anthropic"
    BEDROCK_ANTHROPIC = "bedrock"
    OLLAMA = "ollama"  # Local LLM support via Ollama


class LLMFactory:
    """Factory for creating provider-agnostic LLM instances."""

    # Model mapping: provider-agnostic name -> provider-specific IDs
    # 3 models for development, production, and godmode
    MODEL_MAPPINGS = {
        # Development Model - Claude 3.7 Sonnet
        "claude-4-5-haiku": {
            LLMProvider.BEDROCK_ANTHROPIC: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            LLMProvider.ANTHROPIC_DIRECT: "claude-haiku-4-5-20251001",  # Fallback for direct API
        },
        # Production Model - Claude 4 Sonnet
        "claude-4-sonnet": {
            LLMProvider.BEDROCK_ANTHROPIC: "us.anthropic.claude-sonnet-4-20250514-v1:0",
            LLMProvider.ANTHROPIC_DIRECT: "anthropic.claude-sonnet-4-20250514-v1:0",  # Fallback for direct API
        },
         # ultra Model - Claude 4.5 Sonnet
         "claude-4-5-sonnet": {
             LLMProvider.BEDROCK_ANTHROPIC: "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
             LLMProvider.ANTHROPIC_DIRECT: "claude-sonnet-4-5-20250929",  # Fallback for direct API
         },
         # Godmode Model - Claude 4.5 Sonnet
         "claude-4-1'opus": {
             LLMProvider.BEDROCK_ANTHROPIC: "us.anthropic.claude-opus-4-1-20250805-v1:0",
             LLMProvider.ANTHROPIC_DIRECT: "claude-opus-4-1-20250805",  # Fallback for direct API
        },
    }

    @classmethod
    def detect_provider(cls, explicit_override: Optional[str] = None) -> Optional[LLMProvider]:
        """
        Auto-detect available provider based on environment variables.

        Args:
            explicit_override: Optional explicit provider name (e.g., from CLI flag)
                             Takes highest priority over env vars and auto-detection

        Returns:
            LLMProvider enum value or None if no provider credentials found
        """
        # Check for explicit override (highest priority - from CLI flags)
        if explicit_override:
            try:
                return LLMProvider(explicit_override)
            except ValueError:
                print(
                    f"Warning: Invalid provider '{explicit_override}', falling back to auto-detection"
                )

        # Check for environment variable override
        provider_override = os.environ.get("LOBSTER_LLM_PROVIDER")
        if provider_override:
            try:
                return LLMProvider(provider_override)
            except ValueError:
                print(
                    f"Warning: Invalid LOBSTER_LLM_PROVIDER value '{provider_override}', falling back to auto-detection"
                )

        # Check for Ollama (can work without API keys)
        if os.environ.get("OLLAMA_BASE_URL") or cls._is_ollama_running():
            return LLMProvider.OLLAMA

        # Priority order: Direct API > Bedrock > Others
        if os.environ.get("ANTHROPIC_API_KEY"):
            return LLMProvider.ANTHROPIC_DIRECT
        elif os.environ.get("AWS_BEDROCK_ACCESS_KEY") and os.environ.get(
            "AWS_BEDROCK_SECRET_ACCESS_KEY"
        ):
            return LLMProvider.BEDROCK_ANTHROPIC
        return None

    @classmethod
    def _is_ollama_running(cls) -> bool:
        """
        Check if Ollama server is accessible at default or configured URL.

        Returns:
            bool: True if Ollama server is accessible, False otherwise
        """
        try:
            import requests

            ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            response = requests.get(f"{ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    @classmethod
    def _get_ollama_models(cls) -> list[str]:
        """
        Get list of available models from running Ollama instance.

        Returns:
            List of model names (e.g., ["llama3:8b-instruct", "gpt-oss:20b"])
            Empty list if Ollama is not accessible or no models found
        """
        try:
            import requests

            ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            response = requests.get(f"{ollama_url}/api/tags", timeout=2)

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                # Extract model names (format: "name:tag")
                model_names = [model.get("name") for model in models if model.get("name")]
                return model_names
            return []
        except Exception as e:
            logging.debug(f"Failed to fetch Ollama models: {e}")
            return []

    @classmethod
    def _select_best_ollama_model(cls) -> str:
        """
        Automatically select the best available Ollama model.

        Selection strategy (in priority order):
        1. Environment variable OLLAMA_DEFAULT_MODEL if set
        2. Largest available model from Ollama (by size/quality heuristic)
        3. Default fallback: "llama3:8b-instruct"

        Returns:
            Selected model name
        """
        # Check environment variable first
        env_model = os.environ.get("OLLAMA_DEFAULT_MODEL")
        if env_model:
            return env_model

        # Get available models from Ollama
        available_models = cls._get_ollama_models()

        if not available_models:
            logging.warning("No Ollama models detected, using default: llama3:8b-instruct")
            return "llama3:8b-instruct"

        # Selection heuristics (prefer larger/better models)
        # Priority order: larger parameter counts, instruct/chat variants
        def model_score(model_name: str) -> int:
            """Score model by quality (higher = better)."""
            score = 0
            model_lower = model_name.lower()

            # Prefer instruct/chat models
            if "instruct" in model_lower or "chat" in model_lower:
                score += 100

            # Size-based scoring (extract parameter count)
            # Examples: "llama3:8b", "gpt-oss:20b", "mixtral:8x7b"
            import re

            # Match patterns like "70b", "8b", "20b", "8x7b"
            size_match = re.search(r'(\d+)(?:x)?(\d+)?b', model_lower)
            if size_match:
                main_size = int(size_match.group(1))
                multiplier = int(size_match.group(2)) if size_match.group(2) else 1
                total_params = main_size * multiplier
                score += total_params  # Larger models score higher

            # Prefer newer versions (llama3 > llama2)
            version_match = re.search(r'(\d+)', model_lower)
            if version_match:
                version = int(version_match.group(1))
                score += version * 10

            return score

        # Sort by score and select best
        best_model = max(available_models, key=model_score)

        logging.info(
            f"Auto-selected Ollama model: {best_model} from {len(available_models)} available models"
        )
        return best_model

    @classmethod
    def create_llm(cls, model_config: Dict[str, Any], agent_name: str = None, provider_override: Optional[str] = None) -> Any:
        """
        Create an LLM instance based on configuration and detected provider.

        Args:
            model_config: Configuration dictionary with model parameters
            agent_name: Optional agent name for logging
            provider_override: Optional explicit provider name (e.g., from CLI flag)

        Returns:
            LLM instance (ChatAnthropic, ChatBedrockConverse, etc.)

        Raises:
            ValueError: If no provider credentials are found or provider not implemented
        """
        # Detect provider (with optional explicit override)
        provider = cls.detect_provider(explicit_override=provider_override)

        if not provider:
            raise ValueError(
                "No LLM provider credentials found. Please set one of the following:\n"
                "  - ANTHROPIC_API_KEY for Claude API\n"
                "  - AWS_BEDROCK_ACCESS_KEY and AWS_BEDROCK_SECRET_ACCESS_KEY for AWS Bedrock"
            )

        # Log provider selection if agent_name provided
        if agent_name:
            logging.debug(
                f"Creating LLM for agent '{agent_name}' using provider: {provider.value}"
            )

        # Get model ID from config
        model_id = model_config.get("model_id")

        # Create appropriate LLM instance based on provider
        if provider == LLMProvider.ANTHROPIC_DIRECT:
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                raise ImportError(
                    "langchain-anthropic package not installed. "
                    "Please run: pip install langchain-anthropic"
                )

            # Check if model_id is from another provider (Ollama)
            is_ollama_model = (
                model_id and
                not model_id.startswith("us.") and
                not model_id.startswith("claude-") and
                (":" in model_id or "llama" in model_id.lower())
            )

            # Use Anthropic-specific model or translate from Bedrock
            if is_ollama_model or not model_id:
                # Use default Anthropic model (Claude 3.5 Sonnet)
                anthropic_model = "claude-3-5-sonnet-20241022"
                if agent_name:
                    logging.info(
                        f"Switching to Anthropic - using model '{anthropic_model}' instead of '{model_id}'"
                    )
            else:
                # Translate Bedrock model ID to Anthropic if needed
                anthropic_model = cls._translate_model_id(model_id, provider)

            # Create ChatAnthropic instance
            return ChatAnthropic(
                model=anthropic_model,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                temperature=model_config.get("temperature", 1.0),
                max_tokens=model_config.get("max_tokens", 4096),
            )

        elif provider == LLMProvider.BEDROCK_ANTHROPIC:
            try:
                from langchain_aws import ChatBedrockConverse
            except ImportError:
                raise ImportError(
                    "langchain-aws package not installed. "
                    "Please run: pip install langchain-aws"
                )

            # Check if model_id is from another provider (Ollama)
            # Ollama models typically don't have provider prefixes
            is_ollama_model = (
                model_id and
                not model_id.startswith("us.") and
                not model_id.startswith("claude-") and
                "anthropic" not in model_id.lower() and
                (":" in model_id or "llama" in model_id.lower())  # Ollama format: model:tag
            )

            # Use Bedrock-specific model or default
            if is_ollama_model or not model_id:
                # Use default Bedrock model (Claude 4.5 Sonnet)
                bedrock_model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
                if agent_name:
                    logging.info(
                        f"Switching to Bedrock - using model '{bedrock_model_id}' instead of '{model_id}'"
                    )
            else:
                bedrock_model_id = model_id

            # For Bedrock, use the determined model ID
            bedrock_params = {
                "model_id": bedrock_model_id,
                "temperature": model_config.get("temperature", 1.0),
            }

            # Add region if specified
            if "region_name" in model_config:
                bedrock_params["region_name"] = model_config["region_name"]
            else:
                bedrock_params["region_name"] = "us-east-1"  # Default region

            # Add AWS credentials if explicitly provided
            aws_access_key = os.environ.get("AWS_BEDROCK_ACCESS_KEY")
            aws_secret_key = os.environ.get("AWS_BEDROCK_SECRET_ACCESS_KEY")

            if aws_access_key and aws_secret_key:
                bedrock_params["aws_access_key_id"] = aws_access_key
                bedrock_params["aws_secret_access_key"] = aws_secret_key

            return ChatBedrockConverse(**bedrock_params)

        elif provider == LLMProvider.OLLAMA:
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                raise ImportError(
                    "langchain-ollama package not installed. "
                    "Please run: pip install langchain-ollama"
                )

            # Get model ID from config
            config_model_id = model_config.get("model_id", "")

            # Check if model_id is from another provider (Bedrock/Anthropic)
            # Bedrock models start with "us." or contain "anthropic"
            # Anthropic models start with "claude-"
            is_other_provider_model = (
                config_model_id.startswith("us.") or
                "anthropic" in config_model_id.lower() or
                config_model_id.startswith("claude-")
            )

            # Use Ollama-specific model or auto-detect best available
            if is_other_provider_model or not config_model_id:
                # Automatically select best available Ollama model
                model_id = cls._select_best_ollama_model()
                if agent_name:
                    logging.info(
                        f"Switching to Ollama - auto-selected model '{model_id}'" +
                        (f" (was: '{config_model_id}')" if config_model_id else "")
                    )
            else:
                model_id = config_model_id

            ollama_params = {
                "model": model_id,
                "temperature": model_config.get("temperature", 1.0),
            }

            # Add custom base URL if specified
            ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
            if ollama_base_url:
                ollama_params["base_url"] = ollama_base_url

            # Log for debugging
            if agent_name:
                logging.debug(
                    f"Creating Ollama model '{model_id}' for agent '{agent_name}'"
                )

            return ChatOllama(**ollama_params)

        else:
            raise ValueError(f"Provider {provider.value} is not yet implemented")

    @classmethod
    def _translate_model_id(
        cls, bedrock_model_id: str, target_provider: LLMProvider
    ) -> str:
        """
        Translate Bedrock model ID to provider-specific ID.
        Supports Claude 3.7 Sonnet, Claude 4 Sonnet, and Claude 4.5 Sonnet.

        Args:
            bedrock_model_id: The Bedrock model ID to translate
            target_provider: The target provider to translate for

        Returns:
            Translated model ID for the target provider
        """
        # First, try to find exact match in mappings
        for model_name, mappings in cls.MODEL_MAPPINGS.items():
            if mappings.get(LLMProvider.BEDROCK_ANTHROPIC) == bedrock_model_id:
                translated = mappings.get(target_provider)
                if translated:
                    return translated

        # Fallback: pattern matching for the 3 supported models
        if "claude-3-7-sonnet" in bedrock_model_id:
            # Claude 3.7 Sonnet
            return "claude-3-5-sonnet-20241022"
        elif "claude-sonnet-4-20250514" in bedrock_model_id or (
            "claude-4-sonnet" in bedrock_model_id and "4-5" not in bedrock_model_id
        ):
            # Claude 4 Sonnet
            return "claude-3-5-sonnet-20241022"
        elif (
            "claude-sonnet-4-5" in bedrock_model_id
            or "claude-4-5-sonnet" in bedrock_model_id
        ):
            # Claude 4.5 Sonnet
            return "claude-3-5-sonnet-20241022"

        # Default fallback to Claude 3.5 Sonnet for direct API
        print(
            f"Warning: Could not translate model ID '{bedrock_model_id}', using default Claude 3.5 Sonnet"
        )
        return "claude-3-5-sonnet-20241022"

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """
        Get list of providers that have credentials configured.

        Returns:
            List of available provider names
        """
        available = []

        if os.environ.get("ANTHROPIC_API_KEY"):
            available.append(LLMProvider.ANTHROPIC_DIRECT.value)

        if os.environ.get("AWS_BEDROCK_ACCESS_KEY") and os.environ.get(
            "AWS_BEDROCK_SECRET_ACCESS_KEY"
        ):
            available.append(LLMProvider.BEDROCK_ANTHROPIC.value)

        if os.environ.get("OLLAMA_BASE_URL") or cls._is_ollama_running():
            available.append(LLMProvider.OLLAMA.value)

        return available

    @classmethod
    def get_current_provider(cls) -> Optional[str]:
        """
        Get the currently selected provider.

        Returns:
            Provider name or None if no provider available
        """
        provider = cls.detect_provider()
        return provider.value if provider else None


# Convenience function for backward compatibility
def create_llm(agent_name: str, model_params: Dict[str, Any], provider_override: Optional[str] = None) -> Any:
    """
    Create an LLM instance for a specific agent.

    This is a convenience function that maintains backward compatibility
    with the existing agent code.

    Args:
        agent_name: Name of the agent requesting the LLM
        model_params: Model configuration parameters
        provider_override: Optional explicit provider name (e.g., from CLI flag)

    Returns:
        LLM instance configured for the agent
    """
    return LLMFactory.create_llm(model_params, agent_name, provider_override=provider_override)
