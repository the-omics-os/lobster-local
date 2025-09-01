"""
Professional agent configuration system for Genie AI.

This module provides a flexible, type-safe configuration system that allows
per-agent model configuration for easy testing and deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelProvider(Enum):
    """Supported model providers."""
    BEDROCK_ANTHROPIC = "bedrock_anthropic"
    OPENAI = "openai"
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
    type: str = "enabled"  # Could be extended to support different thinking modes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API parameters."""
        if not self.enabled:
            return {}
        return {
            "thinking": {
                "type": self.type,
                "budget_tokens": self.budget_tokens
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
    Professional configuration manager for Genie AI agents.
    
    Features:
    - Per-agent model configuration
    - Environment-based overrides
    - Fallback mechanisms
    - Easy testing profiles
    - Production-ready validation
    - Thinking/reasoning support for compatible models
    """
    
    # Pre-defined model configurations
    MODEL_PRESETS = {
        # Anthropic Claude Models - Lightweight (Haiku family)
        "claude-3-haiku": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
            tier=ModelTier.LIGHTWEIGHT,
            temperature=1.0,
            description="Fast, cost-effective Claude 3 Haiku for simple tasks",
            supports_thinking=False
        ),
        
        "claude-3-5-haiku": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
            tier=ModelTier.LIGHTWEIGHT,
            temperature=1.0,
            description="Fast, cost-effective Claude 3.5 Haiku for simple tasks",
            supports_thinking=False
        ),
        
        "claude-3-5-sonnet-v2": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            tier=ModelTier.STANDARD,
            temperature=1.0,
            description="Latest Claude 3.5 Sonnet v2 with enhanced capabilities",
            supports_thinking=False
        ),
        
        "claude-4-sonnet": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            tier=ModelTier.STANDARD,
            temperature=1.0,
            description="Next-generation Claude 4 Sonnet model",
            supports_thinking=True
        ),
        
        "claude-4-opus": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-opus-4-20250514-v1:0",
            tier=ModelTier.HEAVY,
            temperature=1.0,
            description="Advanced Claude 4 Opus for complex reasoning",
            supports_thinking=True
        ),
        
        "claude-4-1-opus": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-opus-4-1-20250805-v1:0",
            tier=ModelTier.HEAVY,
            temperature=1.0,
            description="Latest Claude 4.1 Opus with cutting-edge capabilities",
            supports_thinking=True
        ),
        
        # Ultra Performance Models
        "claude-3-7-sonnet": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            tier=ModelTier.ULTRA,
            temperature=1.0,
            description="Highest-performance Claude 3.7 Sonnet model with thinking support",
            supports_thinking=True
        ),
        
        # EU Region Models (for EU compliance)
        "claude-3-5-haiku-eu": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="eu.anthropic.claude-3-5-haiku-20241022-v1:0",
            tier=ModelTier.LIGHTWEIGHT,
            temperature=1.0,
            region="eu-central-1",
            description="EU region Claude 3.5 Haiku model",
            supports_thinking=False
        ),
        
        "claude-3-5-sonnet-v2-eu": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="eu.anthropic.claude-3-5-sonnet-20241022-v2:0",
            tier=ModelTier.STANDARD,
            temperature=1.0,
            region="eu-central-1",
            description="EU region Claude 3.5 Sonnet v2 model",
            supports_thinking=False
        ),
        
        "claude-4-opus-eu": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="eu.anthropic.claude-opus-4-20250514-v1:0",
            tier=ModelTier.HEAVY,
            temperature=1.0,
            region="eu-central-1",
            description="EU region Claude 4 Opus model",
            supports_thinking=True
        ),
        
        "claude-4-1-opus-eu": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="eu.anthropic.claude-opus-4-1-20250805-v1:0",
            tier=ModelTier.HEAVY,
            temperature=1.0,
            region="eu-central-1",
            description="EU region Claude 4.1 Opus model",
            supports_thinking=True
        ),
        
        "claude-3-7-sonnet-eu": ModelConfig(
            provider=ModelProvider.BEDROCK_ANTHROPIC,
            model_id="eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
            tier=ModelTier.ULTRA,
            temperature=1.0,
            region="eu-central-1",
            description="EU region Claude 3.7 Sonnet model with thinking support",
            supports_thinking=True
        )
    }
    
    # Default agents configuration - modify this to add/remove agents dynamically
    DEFAULT_AGENTS = [
        "assistant",
        "supervisor",
        "singlecell_expert", 
        "bulk_rnaseq_expert",
        "method_agent",
        "research_agent",
        "data_expert"
    ]
    
    # Thinking configuration presets
    THINKING_PRESETS = {
        "disabled": ThinkingConfig(enabled=False),
        "light": ThinkingConfig(enabled=True, budget_tokens=1000),
        "standard": ThinkingConfig(enabled=True, budget_tokens=2000),
        "extended": ThinkingConfig(enabled=True, budget_tokens=5000),
        "deep": ThinkingConfig(enabled=True, budget_tokens=10000)
    }
    
    # Pre-defined testing profiles - automatically includes all agents
    TESTING_PROFILES = {
        "development": {
            "supervisor": "claude-3-5-haiku",
            "singlecell_expert": "claude-3-5-haiku",
            "bulk_rnaseq_expert": "claude-3-5-haiku",
            "method_agent": "claude-3-5-haiku",
            "data_expert": "claude-3-5-haiku",
            "research_agent": "claude-3-5-haiku",
            "thinking": {}  # No thinking in development mode #FIXME
        },
        
        "production": {
            "assistant": "claude-3-7-sonnet",
            "supervisor": "claude-3-7-sonnet",
            "singlecell_expert": "claude-4-sonnet",
            "bulk_rnaseq_expert": "claude-4-sonnet",
            "method_agent": "claude-3-7-sonnet",
            "data_expert": "claude-3-7-sonnet",
            "research_agent": "claude-3-7-sonnet",
            "thinking": {
                "supervisor": "standard",
                "singlecell_expert": "standard",
                "bulk_rnaseq_expert": "standard",
                "method_agent": "standard",
                "data_expert": "standard",
                "research_agent": "standard"
                }
        },
        
        "high-performance": {
            "assistant": "claude-3-7-sonnet",
            "supervisor": "claude-4-opus",
            "singlecell_expert": "claude-3-7-sonnet",
            "bulk_rnaseq_expert": "claude-4-opus",
            "method_agent": "claude-4-sonnet",
            "data_expert": "claude-3-5-haiku",
            "research_agent": "claude-3-5-haiku",
            "thinking": { #FIXME
                "supervisor": "extended",
                "singlecell_expert": "standard",
                "bulk_rnaseq_expert": "extended"
            }
        },
        
        "ultra-performance": {
            "assistant": "claude-3-7-sonnet",
            "supervisor": "claude-4-sonnet",
            "singlecell_expert": "claude-4-sonnet",
            "bulk_rnaseq_expert": "claude-4-sonnet",
            "method_agent": "claude-4-sonnet",
            "data_expert": "claude-4-sonnet",
            "research_agent": "claude-4-sonnet",
            "thinking": {}  # Most models don't support thinking yet #FIXME
        },
        
        "cost-optimized": {
            "assistant": "claude-3-7-sonnet",
            "supervisor": "claude-3-haiku",
            "singlecell_expert": "claude-3-5-sonnet",
            "bulk_rnaseq_expert": "claude-3-5-haiku",
            "method_agent": "claude-3-haiku",
            "data_expert": "claude-3-5-haiku",
            "research_agent": "claude-3-haiku",
            "thinking": {}  # No thinking for cost optimization
        },
        
        "heavyweight": {
            "assistant": "claude-3-7-sonnet",
            "supervisor": "claude-4-1-opus",
            "singlecell_expert": "claude-4-1-opus",
            "bulk_rnaseq_expert": "claude-4-1-opus",
            "method_agent": "claude-4-opus",
            "data_expert": "claude-3-5-haiku",
            "research_agent": "claude-4-opus",
            "thinking": {}  # Opus models don't support thinking yet
        },
        
        "eu-compliant": {
            "assistant": "claude-3-7-sonnet",
            "supervisor": "claude-3-5-sonnet-v2-eu",
            "singlecell_expert": "claude-4-1-opus-eu",
            "bulk_rnaseq_expert": "claude-3-5-sonnet-v2-eu",
            "method_agent": "claude-3-5-sonnet-eu",
            "data_expert": "claude-3-5-haiku",
            "research_agent": "claude-3-5-sonnet-eu",
            "thinking": {}  # EU models configuration
        },
        
        "eu-high-performance": {
            "assistant": "claude-3-7-sonnet",
            "supervisor": "claude-3-7-sonnet-eu",
            "singlecell_expert": "claude-3-7-sonnet-eu",
            "bulk_rnaseq_expert": "claude-4-opus-eu",
            "method_agent": "claude-4-opus-eu",
            "data_expert": "claude-3-5-haiku",
            "research_agent": "claude-3-5-sonnet-v2-eu",
            "thinking": {
                "supervisor": "deep",
                "singlecell_expert": "extended"
            }
        }
    }
    
    def __init__(self, profile: str = None):
        """
        Initialize the configurator.
        
        Args:
            profile: Testing profile name (e.g., 'development', 'production')
            config_file: Path to custom configuration file
        """
        self.profile = profile or os.environ.get('GENIE_PROFILE', 'production')
        self._agent_configs = {}
        self._load_from_profile()
        
        # Apply environment overrides
        self._apply_env_overrides()
    
    def _load_from_profile(self):
        """Load configuration from a testing profile."""
        if self.profile not in self.TESTING_PROFILES:
            raise ValueError(f"Unknown profile: {self.profile}. Available: {list(self.TESTING_PROFILES.keys())}")
        
        profile_config = self.TESTING_PROFILES[self.profile]
        
        # Load model configurations
        for agent_name, model_preset in profile_config.items():
            if agent_name == 'thinking':
                continue  # Skip thinking configuration here
                
            if model_preset not in self.MODEL_PRESETS:
                raise ValueError(f"Unknown model preset: {model_preset}")
            
            model_config = self.MODEL_PRESETS[model_preset]
            
            # Initialize thinking config if specified in profile
            thinking_config = None
            if 'thinking' in profile_config and agent_name in profile_config['thinking']:
                thinking_preset = profile_config['thinking'][agent_name]
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
                thinking_config=thinking_config
            )
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Global overrides
        if os.environ.get('GENIE_GLOBAL_MODEL'):
            model_preset = os.environ.get('GENIE_GLOBAL_MODEL')
            if model_preset in self.MODEL_PRESETS:
                for agent_config in self._agent_configs.values():
                    agent_config.model_config = self.MODEL_PRESETS[model_preset]
        
        # Per-agent overrides
        for agent_name in self._agent_configs:
            env_key = f'GENIE_{agent_name.upper()}_MODEL'
            if os.environ.get(env_key):
                model_preset = os.environ.get(env_key)
                if model_preset in self.MODEL_PRESETS:
                    self._agent_configs[agent_name].model_config = self.MODEL_PRESETS[model_preset]
        
        # Temperature overrides
        for agent_name in self._agent_configs:
            env_key = f'GENIE_{agent_name.upper()}_TEMPERATURE'
            if os.environ.get(env_key):
                try:
                    temperature = float(os.environ.get(env_key))
                    self._agent_configs[agent_name].model_config.temperature = temperature
                except ValueError:
                    pass
        
        # Thinking configuration overrides
        for agent_name in self._agent_configs:
            # Enable/disable thinking
            env_key = f'GENIE_{agent_name.upper()}_THINKING_ENABLED'
            if os.environ.get(env_key):
                enabled = os.environ.get(env_key).lower() == 'true'
                if enabled and self._agent_configs[agent_name].model_config.supports_thinking:
                    if not self._agent_configs[agent_name].thinking_config:
                        self._agent_configs[agent_name].thinking_config = ThinkingConfig()
                    self._agent_configs[agent_name].thinking_config.enabled = True
            
            # Thinking token budget
            env_key = f'GENIE_{agent_name.upper()}_THINKING_BUDGET'
            if os.environ.get(env_key):
                try:
                    budget = int(os.environ.get(env_key))
                    if self._agent_configs[agent_name].thinking_config:
                        self._agent_configs[agent_name].thinking_config.budget_tokens = budget
                except ValueError:
                    pass
        
        # Global thinking preset
        if os.environ.get('GENIE_GLOBAL_THINKING'):
            thinking_preset = os.environ.get('GENIE_GLOBAL_THINKING')
            if thinking_preset in self.THINKING_PRESETS:
                for agent_config in self._agent_configs.values():
                    if agent_config.model_config.supports_thinking:
                        agent_config.thinking_config = self.THINKING_PRESETS[thinking_preset]
    
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
            params.update({
                "aws_access_key_id": os.environ.get('AWS_BEDROCK_ACCESS_KEY'),
                "aws_secret_access_key": os.environ.get('AWS_BEDROCK_SECRET_ACCESS_KEY'),
            })
        elif model_config.provider == ModelProvider.OPENAI:
            params.update({
                "openai_api_key": os.environ.get('OPENAI_API_KEY'),
            })
        
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
        config_data = {
            "profile": self.profile,
            "agents": {}
        }
        
        for agent_name, agent_config in self._agent_configs.items():
            config_data["agents"][agent_name] = {
                "model_config": {
                    "provider": agent_config.model_config.provider.value,
                    "model_id": agent_config.model_config.model_id,
                    "tier": agent_config.model_config.tier.value,
                    "temperature": agent_config.model_config.temperature,
                    "region": agent_config.model_config.region,
                    "description": agent_config.model_config.description,
                    "supports_thinking": agent_config.model_config.supports_thinking
                },
                "fallback_model": agent_config.fallback_model,
                "enabled": agent_config.enabled,
                "custom_params": agent_config.custom_params,
                "thinking_config": {
                    "enabled": agent_config.thinking_config.enabled,
                    "budget_tokens": agent_config.thinking_config.budget_tokens,
                    "type": agent_config.thinking_config.type
                } if agent_config.thinking_config else None
            }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def print_current_config(self):
        """Print current configuration in a readable format."""
        print("\n🔧 Genie AI Configuration")
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
                print(f"   🧠 Thinking: Enabled (Budget: {agent_config.thinking_config.budget_tokens} tokens)")
            elif model.supports_thinking:
                print(f"   🧠 Thinking: Available but disabled")

# Singleton instance
_configurator = None

def get_agent_configurator() -> LobsterAgentConfigurator:
    """
    Get the global agent configurator instance.
    
    Returns:
        GenieAgentConfigurator instance
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
        GenieAgentConfigurator instance
    """
    global _configurator
    _configurator = LobsterAgentConfigurator(profile=profile)
    return _configurator
