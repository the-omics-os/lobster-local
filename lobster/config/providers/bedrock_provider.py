"""
AWS Bedrock provider implementation.

This module provides AWS Bedrock integration for Lobster, enabling Claude models
via Amazon's Bedrock service with cross-region support and IAM credentials.

Key features:
- Static model catalog (Claude models via Bedrock)
- AWS IAM credential management
- Cross-region support (default: us-east-1)
- ChatBedrockConverse LangChain integration

Example:
    >>> from lobster.config.providers.bedrock_provider import BedrockProvider
    >>>
    >>> provider = BedrockProvider()
    >>> if provider.is_configured():
    ...     llm = provider.create_chat_model("anthropic.claude-sonnet-4-20250514-v1:0")
"""

import logging
import os
from typing import Any, List

from lobster.config.providers.base_provider import ILLMProvider, ModelInfo

logger = logging.getLogger(__name__)


class BedrockProvider(ILLMProvider):
    """
    AWS Bedrock provider for Claude models.

    Provides access to Claude models through Amazon Bedrock with IAM credential
    management and cross-region support.

    Configuration:
        AWS_BEDROCK_ACCESS_KEY: AWS access key ID
        AWS_BEDROCK_SECRET_ACCESS_KEY: AWS secret access key
        AWS_REGION: AWS region (optional, default: us-east-1)

    Attributes:
        MODELS: Static catalog of Claude models available via Bedrock
        DEFAULT_REGION: Default AWS region for Bedrock API
    """

    # Static model catalog - extracted from BedrockModelService.MODELS
    # Note: Availability may vary by AWS region
    # Model IDs follow AWS Bedrock format: "anthropic." (standard) or "us.anthropic." (cross-region)
    MODELS = [
        # Cross-region model IDs (us. prefix) - Used by agent_config.py profiles
        ModelInfo(
            name="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            display_name="Claude Sonnet 4.5 (Bedrock Cross-Region)",
            description="Claude 4.5 Sonnet via Bedrock - ultra profile",
            provider="bedrock",
            context_window=200000,
            is_default=True,
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-sonnet-4-20250514-v1:0",
            display_name="Claude Sonnet 4 (Bedrock Cross-Region)",
            description="Claude 4 Sonnet via Bedrock - production profile",
            provider="bedrock",
            context_window=200000,
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ),
        ModelInfo(
            name="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            display_name="Claude Haiku 4.5 (Bedrock Cross-Region)",
            description="Claude 4.5 Haiku via Bedrock - development profile",
            provider="bedrock",
            context_window=200000,
            input_cost_per_million=1.0,
            output_cost_per_million=5.0,
        ),
        # Standard region model IDs (anthropic. prefix)
        ModelInfo(
            name="anthropic.claude-sonnet-4-20250514-v1:0",
            display_name="Claude Sonnet 4 (Bedrock)",
            description="Latest Sonnet via Bedrock - best balance",
            provider="bedrock",
            context_window=200000,
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ),
        ModelInfo(
            name="anthropic.claude-opus-4-20250514-v1:0",
            display_name="Claude Opus 4 (Bedrock)",
            description="Most capable via Bedrock - complex reasoning",
            provider="bedrock",
            context_window=200000,
            input_cost_per_million=15.0,
            output_cost_per_million=75.0,
        ),
        ModelInfo(
            name="anthropic.claude-3-5-sonnet-20241022-v2:0",
            display_name="Claude 3.5 Sonnet v2 (Bedrock)",
            description="Claude 3.5 Sonnet - fast and capable",
            provider="bedrock",
            context_window=200000,
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ),
        ModelInfo(
            name="anthropic.claude-3-5-haiku-20241022-v1:0",
            display_name="Claude 3.5 Haiku (Bedrock)",
            description="Fastest Claude via Bedrock",
            provider="bedrock",
            context_window=200000,
            input_cost_per_million=1.0,
            output_cost_per_million=5.0,
        ),
        ModelInfo(
            name="anthropic.claude-3-opus-20240229-v1:0",
            display_name="Claude 3 Opus (Bedrock)",
            description="Claude 3 Opus via Bedrock (legacy)",
            provider="bedrock",
            context_window=200000,
            input_cost_per_million=15.0,
            output_cost_per_million=75.0,
        ),
        ModelInfo(
            name="anthropic.claude-3-sonnet-20240229-v1:0",
            display_name="Claude 3 Sonnet (Bedrock)",
            description="Claude 3 Sonnet via Bedrock (legacy)",
            provider="bedrock",
            context_window=200000,
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        ),
        ModelInfo(
            name="anthropic.claude-3-haiku-20240307-v1:0",
            display_name="Claude 3 Haiku (Bedrock)",
            description="Claude 3 Haiku via Bedrock (legacy)",
            provider="bedrock",
            context_window=200000,
            input_cost_per_million=0.25,
            output_cost_per_million=1.25,
        ),
    ]

    DEFAULT_REGION = "us-east-1"

    @property
    def name(self) -> str:
        """Return provider identifier."""
        return "bedrock"

    @property
    def display_name(self) -> str:
        """Return human-friendly provider name."""
        return "AWS Bedrock"

    def is_configured(self) -> bool:
        """
        Check if AWS Bedrock credentials are configured.

        Returns:
            bool: True if both AWS_BEDROCK_ACCESS_KEY and
                  AWS_BEDROCK_SECRET_ACCESS_KEY are set

        Example:
            >>> provider = BedrockProvider()
            >>> if not provider.is_configured():
            ...     print("Set AWS_BEDROCK_ACCESS_KEY and AWS_BEDROCK_SECRET_ACCESS_KEY")
        """
        return bool(
            os.environ.get("AWS_BEDROCK_ACCESS_KEY")
            and os.environ.get("AWS_BEDROCK_SECRET_ACCESS_KEY")
        )

    def is_available(self) -> bool:
        """
        Check if AWS Bedrock is accessible.

        For Bedrock, this is equivalent to is_configured() since we cannot
        easily test connectivity without making an API call.

        Returns:
            bool: True if credentials are configured

        Note:
            Full connectivity test would require boto3 call, which is expensive.
            We rely on credential presence as proxy for availability.
        """
        return self.is_configured()

    def list_models(self) -> List[ModelInfo]:
        """
        List all available Bedrock models.

        Returns:
            List[ModelInfo]: Static catalog of Claude models via Bedrock

        Example:
            >>> for model in provider.list_models():
            ...     print(f"{model.name}: {model.description}")
        """
        return self.MODELS.copy()

    def get_default_model(self) -> str:
        """
        Get the default Bedrock model.

        Returns:
            str: Default model ID (Claude Sonnet 4 via Bedrock)

        Example:
            >>> provider = BedrockProvider()
            >>> model_id = provider.get_default_model()
            >>> # "anthropic.claude-sonnet-4-20250514-v1:0"
        """
        for model in self.MODELS:
            if model.is_default:
                return model.name
        return self.MODELS[0].name  # Fallback to first model

    def validate_model(self, model_id: str) -> bool:
        """
        Check if a model ID is valid for Bedrock.

        Args:
            model_id: Bedrock model identifier (e.g., "anthropic.claude-sonnet-4-20250514-v1:0")

        Returns:
            bool: True if model exists in catalog

        Example:
            >>> if provider.validate_model("anthropic.claude-sonnet-4-20250514-v1:0"):
            ...     llm = provider.create_chat_model("anthropic.claude-sonnet-4-20250514-v1:0")
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
        Create a ChatBedrockConverse instance.

        Extracted from llm_factory.py lines 290-340 with proper parameter handling
        and AWS credential management.

        Args:
            model_id: Bedrock model ID (e.g., "anthropic.claude-sonnet-4-20250514-v1:0")
            temperature: Sampling temperature (0.0-2.0, default: 1.0)
            max_tokens: Maximum tokens in response (default: 4096)
            **kwargs: Additional parameters:
                - region_name: AWS region (default: us-east-1)
                - aws_access_key_id: Override AWS access key
                - aws_secret_access_key: Override AWS secret key
                - additional_model_request_fields: Extended thinking config (passed to ChatBedrockConverse)

        Returns:
            ChatBedrockConverse: LangChain chat model instance

        Raises:
            ImportError: If langchain-aws not installed
            ValueError: If model_id is invalid or credentials missing

        Example:
            >>> provider = BedrockProvider()
            >>> llm = provider.create_chat_model(
            ...     "anthropic.claude-sonnet-4-20250514-v1:0",
            ...     temperature=0.7,
            ...     max_tokens=8192,
            ...     region_name="us-west-2"
            ... )

            >>> # With extended thinking (AWS Bedrock feature)
            >>> llm = provider.create_chat_model(
            ...     "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            ...     temperature=1.0,
            ...     additional_model_request_fields={
            ...         "thinking": {"type": "enabled", "budget_tokens": 5000}
            ...     }
            ... )
        """
        # Import check
        try:
            from langchain_aws import ChatBedrockConverse
        except ImportError:
            raise ImportError(
                "langchain-aws package not installed. "
                "Please run: pip install langchain-aws"
            )

        # Validate model
        if not self.validate_model(model_id):
            logger.warning(
                f"Model '{model_id}' not in catalog. "
                f"Attempting to use anyway (may work in your region)."
            )

        # Check credentials
        if not self.is_configured():
            raise ValueError(
                "AWS Bedrock credentials not configured. "
                "Set AWS_BEDROCK_ACCESS_KEY and AWS_BEDROCK_SECRET_ACCESS_KEY."
            )

        # Build parameters (extracted from llm_factory.py)
        bedrock_params = {
            "model_id": model_id,
            "temperature": temperature,
        }

        # Add region (from kwargs or environment or default)
        region = kwargs.pop("region_name", None)
        if not region:
            region = os.environ.get("AWS_REGION", self.DEFAULT_REGION)
        bedrock_params["region_name"] = region

        # Add AWS credentials (from kwargs or environment)
        aws_access_key = kwargs.pop("aws_access_key_id", None) or os.environ.get(
            "AWS_BEDROCK_ACCESS_KEY"
        )
        aws_secret_key = kwargs.pop("aws_secret_access_key", None) or os.environ.get(
            "AWS_BEDROCK_SECRET_ACCESS_KEY"
        )

        if aws_access_key and aws_secret_key:
            bedrock_params["aws_access_key_id"] = aws_access_key
            bedrock_params["aws_secret_access_key"] = aws_secret_key
        else:
            raise ValueError(
                "AWS credentials required but not found. "
                "Set AWS_BEDROCK_ACCESS_KEY and AWS_BEDROCK_SECRET_ACCESS_KEY."
            )

        # Pass through any additional kwargs
        bedrock_params.update(kwargs)

        logger.debug(
            f"Creating ChatBedrockConverse with model={model_id}, "
            f"region={region}, temperature={temperature}"
        )

        return ChatBedrockConverse(**bedrock_params)

    def get_configuration_help(self) -> str:
        """
        Get help text for configuring Bedrock.

        Returns:
            str: Configuration instructions

        Example:
            >>> print(provider.get_configuration_help())
        """
        return (
            "Configure AWS Bedrock by setting environment variables:\n\n"
            "Required:\n"
            "  AWS_BEDROCK_ACCESS_KEY=your_access_key_id\n"
            "  AWS_BEDROCK_SECRET_ACCESS_KEY=your_secret_access_key\n\n"
            "Optional:\n"
            "  AWS_REGION=us-east-1  # Default region for Bedrock API\n\n"
            "Note: Model availability varies by region. Claude Sonnet 4 is available in us-east-1."
        )


# Auto-register provider with registry
from lobster.config.providers.registry import ProviderRegistry

ProviderRegistry.register(BedrockProvider())
