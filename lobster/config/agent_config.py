"""
Professional agent configuration system for Lobster AI.

This module provides a flexible, type-safe configuration system that allows
per-agent model configuration for easy testing and deployment.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ModelProvider(Enum):
    """Supported model providers."""

    BEDROCK_ANTHROPIC = "bedrock_anthropic"
    OPENAI = "openai" #TODO
    BEDROCK_META = "bedrock_meta"
    BEDROCK_AMAZON = "bedrock_amazon"


class ModelTier(Enum):
    """Model performance tiers."""

    LIGHTWEIGHT = "lightweight"
    STANDARD = "standard"
    HEAVY = "heavy"
    ULTRA = "ultra"


@dataclass
class ThinkingConfig:
    """Configuration for model thinking/reasoning feature."""

    enabled: bool = False
    budget_tokens: int = 2000
    type: str = "enabled"  # AWS Bedrock uses "enabled" as the type value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API parameters."""
        if not self.enabled:
            return {}
        return {
            "thinking": {
                "type": self.type,  # AWS Bedrock expects "type": "enabled"
                "budget_tokens": self.budget_tokens,
            }
        }


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: ModelProvider
    model_id: str
    tier: ModelTier
    temperature: float = 1.0
    region: str = "us-east-2"
    description: str = ""
    supports_thinking: bool = False  # Flag for models that support thinking

    def __post_init__(self):
        if isinstance(self.provider, str):
            self.provider = ModelProvider(self.provider)
        if isinstance(self.tier, str):
            self.tier = ModelTier(self.tier)


@dataclass
class AgentModelConfig:
    """Model configuration for a specific agent."""

    name: str
    model_config: ModelConfig
    fallback_model: Optional[str] = None
    enabled: bool = True
    custom_params: Dict = field(default_factory=dict)
    thinking_config: Optional[ThinkingConfig] = None


class LobsterAgentConfigurator:
    """
    Professional configuration manager for Lobster AI agents.

    Features:
    - Per-agent model configuration
    - Environment-based overrides
    - Fallback mechanisms
    - Easy testing profiles
    - Production-ready validation
    - Thinking/reasoning support for compatible models
    """

    # Pre-defined model configurations - 3 models
    MODEL_PRESETS = {
        # Development Model - Claude 3.7 Sonnet
        "claude-4-5-haiku": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            tier=ModelTier.ULTRA,
            temperature=1.0,
            region="us-east-1",
            description="Claude 4.5 haiku for development and worker agents",
            supports_thinking=True,
        ),
        # Production Model - Claude 4 Sonnet
        "claude-4-sonnet": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            tier=ModelTier.ULTRA,
            temperature=1.0,
            region="us-east-1",
            description="Claude 4 Sonnet for production",
            supports_thinking=True,
        ),
        # ultra Model - Claude 4.5 Sonnet
        "claude-4-5-sonnet": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            tier=ModelTier.ULTRA,
            temperature=1.0,
            region="us-east-1",
            description="Claude 4.5 Sonnet for uktra mode",
            supports_thinking=True,
        ),
        # Godmode Model - Claude 4.5 Sonnet
        "claude-4-1-opus": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-opus-4-1-20250805-v1:0",
            tier=ModelTier.ULTRA,
            temperature=1.0,
            region="us-east-1",
            description="Claude 4.1 opus for godmode",
            supports_thinking=True,
        ),
    }

    # Default agents configuration - modify this to add/remove agents dynamically
    DEFAULT_AGENTS = [
        "assistant",
        "supervisor",
        "singlecell_expert_agent",
        "bulk_rnaseq_expert_agent",
        # "method_expert_agent",  # DEPRECATED v2.2+: merged into research_agent
        "research_agent",
        "metadata_assistant",  # Metadata operations and cross-dataset mapping
        "data_expert_agent",
        "machine_learning_expert_agent",
        "visualization_expert_agent",
        "ms_proteomics_expert_agent",
        "affinity_proteomics_expert_agent",
        "custom_feature_agent",  # META-AGENT for code generation
        "protein_structure_visualization_expert_agent",  # Protein structure visualization
    ]

    # Thinking configuration presets
    THINKING_PRESETS = {
        "disabled": ThinkingConfig(enabled=False),
        "light": ThinkingConfig(enabled=True, budget_tokens=1000),
        "standard": ThinkingConfig(enabled=True, budget_tokens=2000),
        "extended": ThinkingConfig(enabled=True, budget_tokens=5000),
        "deep": ThinkingConfig(enabled=True, budget_tokens=10000),
    }

    # Pre-defined testing profiles - 3 profiles
    TESTING_PROFILES = {
        "development": {
            # Supervisor and expert agents use Claude 4 Sonnet
            "supervisor": "claude-4-5-haiku",
            # Assistant uses Claude 3.7 Sonnet
            "assistant": "claude-4-5-haiku",
            # All expert agents use Claude 4 Sonnet
            "singlecell_expert_agent": "claude-4-5-haiku",
            "bulk_rnaseq_expert_agent": "claude-4-5-haiku",
            # "method_expert_agent": "claude-4-sonnet",  # DEPRECATED v2.2+
            "data_expert_agent": "claude-4-5-haiku",
            "machine_learning_expert_agent": "claude-4-5-haiku",
            "research_agent": "claude-4-5-haiku",
            "metadata_assistant": "claude-4-5-haiku",
            "ms_proteomics_expert_agent": "claude-4-5-haiku",
            "affinity_proteomics_expert_agent": "claude-4-5-haiku",
            "visualization_expert_agent": "claude-4-5-haiku",
            "protein_structure_visualization_expert_agent": "claude-4-5-haiku",
            "custom_feature_agent": "claude-4-5-sonnet",  # Use Sonnet for code generation
            "thinking": {},  # No thinking in development mode for faster testing
        },
        "production": {
            # Supervisor uses Claude 4.5 Sonnet
            "supervisor": "claude-4-5-sonnet",
            # Assistant uses Claude 3.7 Sonnet
            "assistant": "claude-4-5-haiku",
            # All expert agents use Claude 4 Sonnet
            "singlecell_expert_agent": "claude-4-5-haiku",
            "bulk_rnaseq_expert_agent": "claude-4-5-haiku",
            # "method_expert_agent": "claude-4-5-haiku",  # DEPRECATED v2.2+
            "data_expert_agent": "claude-4-5-haiku",
            "machine_learning_expert_agent": "claude-4-5-haiku",
            "research_agent": "claude-4-5-haiku",
            "metadata_assistant": "claude-4-5-haiku",
            "ms_proteomics_expert_agent": "claude-4-5-haiku",
            "affinity_proteomics_expert_agent": "claude-4-5-haiku",
            "visualization_expert_agent": "claude-4-5-haiku",
            "protein_structure_visualization_expert_agent": "claude-4-5-haiku",
            "custom_feature_agent": "claude-4-5-sonnet",  # Use Sonnet 4.5 for code generation
            "thinking": {},  # No thinking configured for production
        },
        "ultra": {
            # All agents including supervisor and assistant use Claude 4.5 Sonnet
            "supervisor": "claude-4-5-sonnet",
            "assistant": "claude-4-5-sonnet",
            "singlecell_expert_agent": "claude-4-5-sonnet",
            "bulk_rnaseq_expert_agent": "claude-4-5-sonnet",
            # "method_expert_agent": "claude-4-5-sonnet",  # DEPRECATED v2.2+
            "data_expert_agent": "claude-4-5-sonnet",
            "machine_learning_expert_agent": "claude-4-5-sonnet",
            "research_agent": "claude-4-5-sonnet",
            "metadata_assistant": "claude-4-5-sonnet",
            "ms_proteomics_expert_agent": "claude-4-5-sonnet",
            "affinity_proteomics_expert_agent": "claude-4-5-sonnet",
            "visualization_expert_agent": "claude-4-5-sonnet",
            "protein_structure_visualization_expert_agent": "claude-4-5-sonnet",
            "custom_feature_agent": "claude-4-5-sonnet",  # Use Sonnet 4.5 for code generation
            "thinking": {},  # No thinking configured for godmode
        },
        "godmode": {
            # All agents including supervisor and assistant use Claude 4.5 Sonnet
            "supervisor": "claude-4-1-opus",
            "assistant": "claude-4-5-sonnet",
            "singlecell_expert_agent": "claude-4-5-sonnet",
            "bulk_rnaseq_expert_agent": "claude-4-5-sonnet",
            # "method_expert_agent": "claude-4-5-sonnet",  # DEPRECATED v2.2+
            "data_expert_agent": "claude-4-5-sonnet",
            "machine_learning_expert_agent": "claude-4-5-sonnet",
            "research_agent": "claude-4-5-sonnet",
            "metadata_assistant": "claude-4-5-sonnet",
            "ms_proteomics_expert_agent": "claude-4-5-sonnet",
            "affinity_proteomics_expert_agent": "claude-4-5-sonnet",
            "visualization_expert_agent": "claude-4-5-sonnet",
            "protein_structure_visualization_expert_agent": "claude-4-5-sonnet",
            "custom_feature_agent": "claude-4-1-opus",  # Use Opus 4.1 for best code generation
            "thinking": {},  # No thinking configured for godmode
        },
    }

    def __init__(self, profile: str = None):
        """
        Initialize the configurator.

        Args:
            profile: Testing profile name (e.g., 'development', 'production')
            config_file: Path to custom configuration file
        """
        # Note: Environment variables still use LOBSTER_ prefix for backward compatibility
        self.profile = profile or os.environ.get("LOBSTER_PROFILE", "production")
        self._agent_configs = {}
        self._load_from_profile()

        # Apply environment overrides
        self._apply_env_overrides()

    def _load_from_profile(self):
        """Load configuration from a testing profile."""
        if self.profile not in self.TESTING_PROFILES:
            raise ValueError(
                f"Unknown profile: {self.profile}. Available: {list(self.TESTING_PROFILES.keys())}"
            )

        profile_config = self.TESTING_PROFILES[self.profile]

        # Load model configurations
        for agent_name, model_preset in profile_config.items():
            if agent_name == "thinking":
                continue  # Skip thinking configuration here

            if model_preset not in self.MODEL_PRESETS:
                raise ValueError(f"Unknown model preset: {model_preset}")

            model_config = self.MODEL_PRESETS[model_preset]

            # Initialize thinking config if specified in profile
            thinking_config = None
            if (
                "thinking" in profile_config
                and agent_name in profile_config["thinking"]
            ):
                thinking_preset = profile_config["thinking"][agent_name]
                if thinking_preset in self.THINKING_PRESETS:
                    thinking_config = self.THINKING_PRESETS[thinking_preset]
                elif isinstance(thinking_preset, dict):
                    thinking_config = ThinkingConfig(**thinking_preset)

                # Only apply thinking if model supports it
                if thinking_config and not model_config.supports_thinking:
                    thinking_config = None

            self._agent_configs[agent_name] = AgentModelConfig(
                name=agent_name,
                model_config=model_config,
                thinking_config=thinking_config,
            )

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Global overrides
        if os.environ.get("LOBSTER_GLOBAL_MODEL"):
            model_preset = os.environ.get("LOBSTER_GLOBAL_MODEL")
            if model_preset in self.MODEL_PRESETS:
                for agent_config in self._agent_configs.values():
                    agent_config.model_config = self.MODEL_PRESETS[model_preset]

        # Per-agent overrides
        for agent_name in self._agent_configs:
            env_key = f"LOBSTER_{agent_name.upper()}_MODEL"
            if os.environ.get(env_key):
                model_preset = os.environ.get(env_key)
                if model_preset in self.MODEL_PRESETS:
                    self._agent_configs[agent_name].model_config = self.MODEL_PRESETS[
                        model_preset
                    ]

        # Temperature overrides
        for agent_name in self._agent_configs:
            env_key = f"LOBSTER_{agent_name.upper()}_TEMPERATURE"
            if os.environ.get(env_key):
                try:
                    temperature = float(os.environ.get(env_key))
                    self._agent_configs[agent_name].model_config.temperature = (
                        temperature
                    )
                except ValueError:
                    pass

        # Thinking configuration overrides
        for agent_name in self._agent_configs:
            # Enable/disable thinking
            env_key = f"LOBSTER_{agent_name.upper()}_THINKING_ENABLED"
            if os.environ.get(env_key):
                enabled = os.environ.get(env_key).lower() == "true"
                if (
                    enabled
                    and self._agent_configs[agent_name].model_config.supports_thinking
                ):
                    if not self._agent_configs[agent_name].thinking_config:
                        self._agent_configs[agent_name].thinking_config = (
                            ThinkingConfig()
                        )
                    self._agent_configs[agent_name].thinking_config.enabled = True

            # Thinking token budget
            env_key = f"LOBSTER_{agent_name.upper()}_THINKING_BUDGET"
            if os.environ.get(env_key):
                try:
                    budget = int(os.environ.get(env_key))
                    if self._agent_configs[agent_name].thinking_config:
                        self._agent_configs[
                            agent_name
                        ].thinking_config.budget_tokens = budget
                except ValueError:
                    pass

        # Global thinking preset
        if os.environ.get("LOBSTER_GLOBAL_THINKING"):
            thinking_preset = os.environ.get("LOBSTER_GLOBAL_THINKING")
            if thinking_preset in self.THINKING_PRESETS:
                for agent_config in self._agent_configs.values():
                    if agent_config.model_config.supports_thinking:
                        agent_config.thinking_config = self.THINKING_PRESETS[
                            thinking_preset
                        ]

    def get_agent_model_config(self, agent_name: str) -> AgentModelConfig:
        """
        Get model configuration for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentModelConfig for the specified agent

        Raises:
            KeyError: If agent configuration not found
        """
        if agent_name not in self._agent_configs:
            raise KeyError(f"No configuration found for agent: {agent_name}")

        return self._agent_configs[agent_name]

    def get_model_config(self, agent_name: str) -> ModelConfig:
        """
        Get model configuration for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            ModelConfig for the specified agent
        """
        return self.get_agent_model_config(agent_name).model_config

    def get_thinking_config(self, agent_name: str) -> Optional[ThinkingConfig]:
        """
        Get thinking configuration for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            ThinkingConfig for the specified agent or None if not configured
        """
        agent_config = self.get_agent_model_config(agent_name)
        return agent_config.thinking_config

    def get_llm_params(self, agent_name: str) -> Dict:
        """
        Get LLM initialization parameters for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary of parameters for LLM initialization
        """
        agent_config = self.get_agent_model_config(agent_name)
        model_config = agent_config.model_config

        # Base parameters
        params = {
            "model_id": model_config.model_id,
            "temperature": model_config.temperature,
            "region_name": model_config.region,
        }

        # Add provider-specific parameters
        if model_config.provider == ModelProvider.BEDROCK_ANTHROPIC:
            params.update(
                {
                    "aws_access_key_id": os.environ.get("AWS_BEDROCK_ACCESS_KEY"),
                    "aws_secret_access_key": os.environ.get(
                        "AWS_BEDROCK_SECRET_ACCESS_KEY"
                    ),
                }
            )
        elif model_config.provider == ModelProvider.OPENAI: #TODO
            params.update(
                {
                    "openai_api_key": os.environ.get("OPENAI_API_KEY"),
                }
            )

        # Add thinking configuration if enabled
        if agent_config.thinking_config and agent_config.thinking_config.enabled:
            thinking_params = agent_config.thinking_config.to_dict()
            if thinking_params:
                params["additional_model_request_fields"] = thinking_params

        return params

    def list_available_models(self) -> Dict[str, ModelConfig]:
        """List all available model presets."""
        return self.MODEL_PRESETS.copy()

    def list_available_profiles(self) -> Dict[str, Dict]:
        """List all available testing profiles."""
        return self.TESTING_PROFILES.copy()

    def list_thinking_presets(self) -> Dict[str, ThinkingConfig]:
        """List all available thinking presets."""
        return self.THINKING_PRESETS.copy()

    def get_current_profile(self) -> str:
        """Get current active profile."""
        return self.profile

    def export_config(self, filepath: str):
        """
        Export current configuration to JSON file.

        Args:
            filepath: Path to save configuration file
        """
        config_data = {"profile": self.profile, "agents": {}}

        for agent_name, agent_config in self._agent_configs.items():
            config_data["agents"][agent_name] = {
                "model_config": {
                    "provider": agent_config.model_config.provider.value,
                    "model_id": agent_config.model_config.model_id,
                    "tier": agent_config.model_config.tier.value,
                    "temperature": agent_config.model_config.temperature,
                    "region": agent_config.model_config.region,
                    "description": agent_config.model_config.description,
                    "supports_thinking": agent_config.model_config.supports_thinking,
                },
                "fallback_model": agent_config.fallback_model,
                "enabled": agent_config.enabled,
                "custom_params": agent_config.custom_params,
                "thinking_config": (
                    {
                        "enabled": agent_config.thinking_config.enabled,
                        "budget_tokens": agent_config.thinking_config.budget_tokens,
                        "type": agent_config.thinking_config.type,
                    }
                    if agent_config.thinking_config
                    else None
                ),
            }

        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=2)

    def print_current_config(self):
        """Print current configuration in a readable format."""
        print("\n🔧 Lobster AI Configuration")
        print(f"Profile: {self.profile}")
        print(f"{'='*60}")

        for agent_name, agent_config in self._agent_configs.items():
            model = agent_config.model_config
            print(f"\n🤖 {agent_name.title()}")
            print(f"   Model: {model.model_id}")
            print(f"   Tier: {model.tier.value}")
            print(f"   Region: {model.region}")
            print(f"   Temperature: {model.temperature}")
            if model.description:
                print(f"   Description: {model.description}")
            if agent_config.thinking_config and agent_config.thinking_config.enabled:
                print(
                    f"   🧠 Thinking: Enabled (Budget: {agent_config.thinking_config.budget_tokens} tokens)"
                )
            elif model.supports_thinking:
                print("   🧠 Thinking: Available but disabled")


# Singleton instance
_configurator = None


def get_agent_configurator() -> LobsterAgentConfigurator:
    """
    Get the global agent configurator instance.

    Returns:
        LobsterAgentConfigurator instance
    """
    global _configurator
    if _configurator is None:
        _configurator = LobsterAgentConfigurator()
    return _configurator


def initialize_configurator(profile: str = None) -> LobsterAgentConfigurator:
    """
    Initialize or reinitialize the global configurator.

    Args:
        profile: Testing profile name
        config_file: Path to custom configuration file

    Returns:
        LobsterAgentConfigurator instance
    """
    global _configurator
    _configurator = LobsterAgentConfigurator(profile=profile)
    return _configurator
