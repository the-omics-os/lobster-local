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


@dataclass
class CustomAgentConfig:
    """
    Configuration for custom agent LLM settings (registered via entry points).

    Custom packages register these configs via the 'lobster.agent_configs' entry point
    group. They are discovered at runtime and merged into the LobsterAgentConfigurator.

    Supports two modes:
    1. Explicit model_preset: Specify a model directly (e.g., "claude-4-sonnet")
    2. model_reference: Copy config from another agent (e.g., "research_agent")

    Attributes:
        name: Agent name (must match the agent's name in AGENT_REGISTRY)
        model_preset: Optional key in MODEL_PRESETS (e.g., "claude-4-sonnet")
        model_reference: Optional agent name to copy config from (e.g., "research_agent")
        thinking_preset: Optional key in THINKING_PRESETS (e.g., "standard")
        temperature_override: Optional custom temperature (overrides preset default)
        custom_params: Additional parameters passed to agent

    Examples:
        ```python
        # Explicit model preset
        METADATA_ASSISTANT_CONFIG = CustomAgentConfig(
            name="metadata_assistant",
            model_preset="claude-4-sonnet",
            thinking_preset="standard",
        )

        # Dynamic reference (copies research_agent's current model)
        METADATA_ASSISTANT_CONFIG = CustomAgentConfig(
            name="metadata_assistant",
            model_reference="research_agent",  # Uses same model as research_agent
            thinking_preset="standard",
        )
        ```
    """

    name: str
    model_preset: Optional[str] = None  # Key in MODEL_PRESETS (e.g., "claude-4-sonnet")
    model_reference: Optional[str] = None  # Agent name to copy config from (e.g., "research_agent")
    thinking_preset: Optional[str] = None  # Key in THINKING_PRESETS
    temperature_override: Optional[float] = None
    custom_params: Dict = field(default_factory=dict)

    def to_agent_model_config(
        self, configurator: "LobsterAgentConfigurator"
    ) -> AgentModelConfig:
        """
        Convert to AgentModelConfig using configurator's preset registry.

        Supports two modes:
        1. model_preset: Explicit model selection
        2. model_reference: Copy config from another agent

        Args:
            configurator: The LobsterAgentConfigurator instance with MODEL_PRESETS

        Returns:
            AgentModelConfig ready for use in the agent system

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate that exactly one of model_preset or model_reference is specified
        if self.model_preset and self.model_reference:
            raise ValueError(
                f"Agent '{self.name}': Cannot specify both model_preset and model_reference. "
                f"Choose one: explicit preset ('{self.model_preset}') or reference ('{self.model_reference}')."
            )

        if not self.model_preset and not self.model_reference:
            raise ValueError(
                f"Agent '{self.name}': Must specify either model_preset or model_reference. "
                f"Examples: model_preset='claude-4-sonnet' or model_reference='research_agent'"
            )

        # Resolve model config
        if self.model_reference:
            # Dynamic reference: Copy config from another agent
            if self.model_reference not in configurator._agent_configs:
                raise ValueError(
                    f"Agent '{self.name}': Referenced agent '{self.model_reference}' not found in configurator. "
                    f"Available agents: {list(configurator._agent_configs.keys())}"
                )

            # Copy model config from referenced agent
            referenced_config = configurator._agent_configs[self.model_reference]
            model_config = referenced_config.model_config

        else:
            # Explicit model preset
            if self.model_preset not in configurator.MODEL_PRESETS:
                raise ValueError(
                    f"Agent '{self.name}': Unknown model preset '{self.model_preset}'. "
                    f"Available: {list(configurator.MODEL_PRESETS.keys())}"
                )

            model_config = configurator.MODEL_PRESETS[self.model_preset]

        # Apply temperature override if specified
        if self.temperature_override is not None:
            from dataclasses import replace

            model_config = replace(model_config, temperature=self.temperature_override)

        # Resolve thinking config
        thinking_config = None
        if self.thinking_preset:
            if self.thinking_preset not in configurator.THINKING_PRESETS:
                raise ValueError(
                    f"Agent '{self.name}': Unknown thinking preset '{self.thinking_preset}'. "
                    f"Available: {list(configurator.THINKING_PRESETS.keys())}"
                )
            thinking_config = configurator.THINKING_PRESETS[self.thinking_preset]

            # Only apply thinking if model supports it
            if not model_config.supports_thinking:
                thinking_config = None

        return AgentModelConfig(
            name=self.name,
            model_config=model_config,
            thinking_config=thinking_config,
            custom_params=self.custom_params,
        )


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
        # Development Model - Claude 4.5 Haiku (fastest, lightweight)
        "claude-4-5-haiku": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            tier=ModelTier.LIGHTWEIGHT,
            temperature=1.0,
            region="us-east-1",
            description="Claude 4.5 haiku for development and worker agents",
            supports_thinking=True,
        ),
        # Production Model - Claude 4 Sonnet (balanced performance)
        "claude-4-sonnet": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            tier=ModelTier.STANDARD,
            temperature=1.0,
            region="us-east-1",
            description="Claude 4 Sonnet for production",
            supports_thinking=True,
        ),
        # Ultra Mode - Claude 4.5 Sonnet (advanced capabilities)
        "claude-4-5-sonnet": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            tier=ModelTier.HEAVY,
            temperature=1.0,
            region="us-east-1",
            description="Claude 4.5 Sonnet for ultra mode",
            supports_thinking=True,
        ),
        # Godmode Model - Claude 4.1 Opus (most capable)
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

    # Default agents configuration (FREE tier only)
    # Premium agents (metadata_assistant, proteomics_expert, machine_learning_expert_agent,
    # protein_structure_visualization_expert_agent, custom_feature_agent) are loaded via
    # component_registry from lobster-premium or lobster-custom-* packages.
    DEFAULT_AGENTS = [
        "assistant",
        "supervisor",
        # Unified transcriptomics agents (v2.5+)
        "transcriptomics_expert",  # Parent: handles QC, clustering, orchestrates sub-agents
        "annotation_expert",  # Sub-agent: cell type annotation
        "de_analysis_expert",  # Sub-agent: differential expression
        "research_agent",
        "data_expert_agent",
        "visualization_expert_agent",
    ]

    # Thinking configuration presets
    THINKING_PRESETS = {
        "disabled": ThinkingConfig(enabled=False),
        "light": ThinkingConfig(enabled=True, budget_tokens=1000),
        "standard": ThinkingConfig(enabled=True, budget_tokens=2000),
        "extended": ThinkingConfig(enabled=True, budget_tokens=5000),
        "deep": ThinkingConfig(enabled=True, budget_tokens=10000),
    }

    # Pre-defined testing profiles (FREE tier agents only)
    # Premium agents get their model configs from custom packages via component_registry.
    TESTING_PROFILES = {
        "development": {
            "supervisor": "claude-4-5-haiku",
            "assistant": "claude-4-5-haiku",
            # Unified transcriptomics agents (v2.5+)
            "transcriptomics_expert": "claude-4-5-haiku",
            "annotation_expert": "claude-4-5-haiku",
            "de_analysis_expert": "claude-4-5-haiku",
            "data_expert_agent": "claude-4-5-haiku",
            "data_expert_assistant": "claude-4-5-haiku",  # LLM-based helper for data_expert
            "research_agent": "claude-4-5-haiku",
            "visualization_expert_agent": "claude-4-5-haiku",
            "thinking": {},  # No thinking in development mode for faster testing
        },
        "production": {
            "supervisor": "claude-4-5-sonnet",
            "assistant": "claude-4-sonnet",
            # Unified transcriptomics agents (v2.5+)
            "transcriptomics_expert": "claude-4-sonnet",
            "annotation_expert": "claude-4-sonnet",
            "de_analysis_expert": "claude-4-sonnet",
            "data_expert_agent": "claude-4-sonnet",
            "data_expert_assistant": "claude-4-sonnet",  # LLM-based helper for data_expert
            "research_agent": "claude-4-sonnet",
            "visualization_expert_agent": "claude-4-sonnet",
            "thinking": {},  # No thinking configured for production
        },
        "ultra": {
            "supervisor": "claude-4-5-sonnet",
            "assistant": "claude-4-5-sonnet",
            # Unified transcriptomics agents (v2.5+)
            "transcriptomics_expert": "claude-4-5-sonnet",
            "annotation_expert": "claude-4-5-sonnet",
            "de_analysis_expert": "claude-4-5-sonnet",
            "data_expert_agent": "claude-4-5-sonnet",
            "data_expert_assistant": "claude-4-5-sonnet",  # LLM-based helper for data_expert
            "research_agent": "claude-4-5-sonnet",
            "visualization_expert_agent": "claude-4-5-sonnet",
            "thinking": {},  # No thinking configured for ultra
        },
        "godmode": {
            "supervisor": "claude-4-1-opus",
            "assistant": "claude-4-5-sonnet",
            # Unified transcriptomics agents (v2.5+)
            "transcriptomics_expert": "claude-4-5-sonnet",
            "annotation_expert": "claude-4-5-sonnet",
            "de_analysis_expert": "claude-4-5-sonnet",
            "data_expert_agent": "claude-4-5-sonnet",
            "data_expert_assistant": "claude-4-5-sonnet",  # LLM-based helper for data_expert
            "research_agent": "claude-4-5-sonnet",
            "visualization_expert_agent": "claude-4-5-sonnet",
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
        self._custom_configs_loaded = False  # Track lazy loading
        self._load_from_profile()

        # Note: Custom agent configs are loaded lazily to avoid circular imports
        # They will be loaded on first access via _ensure_custom_configs_loaded()

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

    def _ensure_custom_configs_loaded(self):
        """
        Lazily load custom agent configs from entry points (on first access).

        This method is idempotent and defers loading to avoid circular imports
        during Settings initialization. Custom configs take precedence over
        profile defaults for premium agents.

        Entry points are discovered via the 'lobster.agent_configs' group.
        """
        if self._custom_configs_loaded:
            return

        import logging

        logger = logging.getLogger(__name__)

        try:
            from lobster.core.component_registry import component_registry

            # Trigger discovery
            component_registry.load_components()

            # Merge custom configs
            custom_configs = component_registry.list_agent_configs()

            if not custom_configs:
                self._custom_configs_loaded = True
                return

            for agent_name, custom_config in custom_configs.items():
                try:
                    # Convert CustomAgentConfig → AgentModelConfig
                    agent_model_config = custom_config.to_agent_model_config(self)

                    # Merge into registry (overwrites if agent_name already exists)
                    self._agent_configs[agent_name] = agent_model_config

                    ref_or_preset = custom_config.model_reference or custom_config.model_preset
                    logger.debug(
                        f"Loaded custom config for '{agent_name}' "
                        f"(config: {ref_or_preset})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load custom config for '{agent_name}': {e}"
                    )

            logger.info(
                f"Merged {len(custom_configs)} custom agent configs: "
                f"{list(custom_configs.keys())}"
            )

        except ImportError:
            # component_registry not available (shouldn't happen in normal installs)
            logger.debug(
                "component_registry not available, skipping custom agent configs"
            )
        except Exception as e:
            # Don't let component discovery failures break the system
            logger.warning(f"Failed to load custom agent configs: {e}")

        self._custom_configs_loaded = True

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
        # Lazily load custom configs on first access (avoids circular imports)
        self._ensure_custom_configs_loaded()

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

        Returns tier-related parameters only (temperature, thinking config).
        Model selection is handled by the provider-aware system:
        1. CLI --model flag
        2. workspace_config.{provider}_model
        3. Provider.get_default_model()

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary of parameters for LLM initialization (excludes model_id)
        """
        agent_config = self.get_agent_model_config(agent_name)
        model_config = agent_config.model_config

        # Return tier parameters only - model_id is handled by provider system
        # This enables provider-agnostic model selection (Gemini, Anthropic, Bedrock, Ollama)
        params = {
            "temperature": model_config.temperature,
        }

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

    def print_current_config(self, show_all: bool = False):
        """
        Print current configuration in a readable format.

        Args:
            show_all: If True, show all configured agents regardless of license tier.
                      If False (default), filter by current license tier.
        """
        # Import license and tier utilities
        from lobster.core.license_manager import get_current_tier
        from lobster.config.subscription_tiers import (
            is_agent_available,
            get_tier_display_name,
        )

        # Get current tier
        current_tier = get_current_tier()

        print("\n🔧 Lobster AI Configuration")
        print(f"Profile: {self.profile}")
        print(f"License Tier: {get_tier_display_name(current_tier)}")
        print(f"{'='*60}")

        # Filter agents based on tier availability
        displayed_count = 0
        filtered_count = 0

        for agent_name, agent_config in self._agent_configs.items():
            # Check if agent is available for current tier (unless show_all is True)
            if not show_all and not is_agent_available(agent_name, current_tier):
                filtered_count += 1
                continue

            displayed_count += 1
            model = agent_config.model_config
            print(f"\n🤖 {agent_name.title()}")
            print(f"   Model: {model.model_id}")
            print(f"   Performance: {model.tier.value}")
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

        # Show summary if agents were filtered
        if filtered_count > 0 and not show_all:
            print(f"\n{'='*60}")
            print(
                f"📊 Summary: Showing {displayed_count} agents available for {current_tier} tier"
            )
            print(f"   ({filtered_count} premium agents hidden - upgrade to access)")
            print(
                f"   Tip: Use 'lobster config show-config --show-all' to see all configured agents"
            )


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
