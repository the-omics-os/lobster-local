"""
Workspace-scoped provider and model configuration with Pydantic validation.

This module provides configuration management for workspace-specific LLM provider
and model preferences, separate from global .env settings.

Storage Location: .lobster_workspace/provider_config.json

Priority Resolution (highest to lowest):
1. Runtime overrides (CLI flags like --provider)
2. Workspace config (.lobster_workspace/provider_config.json)
3. Global user config (~/.config/lobster/providers.json)
4. Environment variables (.env file)
5. Auto-detection (Ollama running, API keys present)
6. Hardcoded defaults

Example:
    >>> from lobster.config.workspace_config import WorkspaceProviderConfig
    >>> from pathlib import Path
    >>>
    >>> # Load config from workspace
    >>> config = WorkspaceProviderConfig.load(Path(".lobster_workspace"))
    >>>
    >>> # Modify and save
    >>> config.global_provider = "ollama"
    >>> config.ollama_model = "llama3:70b-instruct"
    >>> config.save(Path(".lobster_workspace"))
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Configuration file name
CONFIG_FILE_NAME = "provider_config.json"


class WorkspaceProviderConfig(BaseModel):
    """
    Workspace-scoped provider and model configuration.

    Attributes:
        global_provider: LLM provider for all agents (bedrock | anthropic | ollama)
        anthropic_model: Anthropic model to use (e.g., "claude-sonnet-4-20250514")
        bedrock_model: Bedrock model ID to use (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
        ollama_model: Ollama model to use (e.g., "llama3:70b-instruct")
        ollama_host: Ollama server URL (default: http://localhost:11434)
        per_agent_providers: Provider override per agent (e.g., {"supervisor": "ollama"})
        per_agent_models: Model override per agent (e.g., {"supervisor": "llama3:70b"})
        profile: Agent configuration profile (development | production | ultra | godmode | hybrid)

    Example:
        >>> config = WorkspaceProviderConfig(
        ...     global_provider="anthropic",
        ...     anthropic_model="claude-sonnet-4-20250514",
        ...     profile="production"
        ... )
        >>> # Or for Ollama:
        >>> config.global_provider = "ollama"
        >>> config.ollama_model = "llama3:70b-instruct"
    """

    global_provider: Optional[str] = Field(
        None,
        description="Global LLM provider (bedrock | anthropic | ollama)"
    )

    # Per-provider model settings
    anthropic_model: Optional[str] = Field(
        None,
        description="Anthropic model (e.g., 'claude-sonnet-4-20250514', 'claude-3-5-haiku-20241022')"
    )

    bedrock_model: Optional[str] = Field(
        None,
        description="Bedrock model ID (e.g., 'anthropic.claude-3-5-sonnet-20241022-v2:0')"
    )

    ollama_model: Optional[str] = Field(
        None,
        description="Ollama model (e.g., 'llama3:70b-instruct', 'mixtral:8x7b')"
    )

    ollama_host: str = Field(
        "http://localhost:11434",
        description="Ollama server URL"
    )

    per_agent_providers: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-agent provider overrides (e.g., {'supervisor': 'ollama'})"
    )

    per_agent_models: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-agent model overrides (e.g., {'supervisor': 'llama3:70b'})"
    )

    profile: str = Field(
        "production",
        description="Agent configuration profile (development | production | ultra | godmode | hybrid)"
    )

    @field_validator("global_provider")
    @classmethod
    def validate_provider(cls, v):
        """Validate provider is one of the supported types."""
        if v and v not in ["bedrock", "anthropic", "ollama"]:
            raise ValueError(
                f"Invalid provider: '{v}'. Must be one of: bedrock, anthropic, ollama"
            )
        return v

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, v):
        """Validate profile is one of the defined profiles."""
        valid_profiles = ["development", "production", "ultra", "godmode", "hybrid"]
        if v not in valid_profiles:
            raise ValueError(
                f"Invalid profile: '{v}'. Must be one of: {', '.join(valid_profiles)}"
            )
        return v

    @field_validator("per_agent_providers")
    @classmethod
    def validate_agent_providers(cls, v):
        """Validate per-agent provider overrides."""
        valid_providers = ["bedrock", "anthropic", "ollama"]
        for agent, provider in v.items():
            if provider not in valid_providers:
                raise ValueError(
                    f"Invalid provider '{provider}' for agent '{agent}'. "
                    f"Must be one of: {', '.join(valid_providers)}"
                )
        return v

    def save(self, workspace_path: Path) -> None:
        """
        Save configuration to workspace with atomic write.

        Args:
            workspace_path: Path to workspace directory

        Raises:
            IOError: If write operation fails

        Example:
            >>> config = WorkspaceProviderConfig(global_provider="ollama")
            >>> config.save(Path(".lobster_workspace"))
        """
        config_path = workspace_path / CONFIG_FILE_NAME

        # Ensure workspace directory exists
        workspace_path.mkdir(parents=True, exist_ok=True)

        try:
            # Write with indentation for human readability
            config_path.write_text(self.model_dump_json(indent=2))
            logger.info(f"Saved workspace config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save workspace config: {e}")
            raise

    @classmethod
    def load(cls, workspace_path: Path) -> "WorkspaceProviderConfig":
        """
        Load configuration from workspace with graceful error handling.

        Handles:
        - Missing file: Returns default configuration
        - Corrupted JSON: Logs warning, returns defaults
        - Invalid schema: Logs validation errors, returns defaults

        Args:
            workspace_path: Path to workspace directory

        Returns:
            WorkspaceProviderConfig: Loaded or default configuration

        Example:
            >>> config = WorkspaceProviderConfig.load(Path(".lobster_workspace"))
            >>> if config.global_provider:
            ...     print(f"Using provider: {config.global_provider}")
        """
        config_path = workspace_path / CONFIG_FILE_NAME

        # File doesn't exist - return defaults
        if not config_path.exists():
            logger.debug(
                f"No workspace config found at {config_path}, using defaults"
            )
            return cls()

        try:
            # Parse JSON
            data = json.loads(config_path.read_text())
            config = cls(**data)
            logger.info(f"Loaded workspace config from {config_path}")
            return config

        except json.JSONDecodeError as e:
            logger.warning(
                f"Corrupted config file at {config_path}: {e}. Using defaults."
            )
            return cls()

        except Exception as e:
            # Catch Pydantic validation errors and other exceptions
            logger.warning(
                f"Invalid config schema at {config_path}: {e}. Using defaults."
            )
            return cls()

    @classmethod
    def exists(cls, workspace_path: Path) -> bool:
        """
        Check if workspace configuration file exists.

        Args:
            workspace_path: Path to workspace directory

        Returns:
            bool: True if configuration file exists

        Example:
            >>> if WorkspaceProviderConfig.exists(Path(".lobster_workspace")):
            ...     config = WorkspaceProviderConfig.load(Path(".lobster_workspace"))
        """
        config_path = workspace_path / CONFIG_FILE_NAME
        return config_path.exists()

    def reset(self) -> None:
        """
        Reset configuration to defaults (clears all overrides).

        Example:
            >>> config = WorkspaceProviderConfig.load(workspace)
            >>> config.reset()
            >>> config.save(workspace)  # Saves defaults
        """
        self.global_provider = None
        self.anthropic_model = None
        self.bedrock_model = None
        self.ollama_model = None
        self.ollama_host = "http://localhost:11434"
        self.per_agent_providers = {}
        self.per_agent_models = {}
        self.profile = "production"
        logger.info("Reset workspace config to defaults")

    def get_model_for_provider(self, provider: str) -> Optional[str]:
        """
        Get the configured model for a specific provider.

        Args:
            provider: Provider name (anthropic | bedrock | ollama)

        Returns:
            Model name/ID if configured, None otherwise

        Example:
            >>> config = WorkspaceProviderConfig(anthropic_model="claude-sonnet-4-20250514")
            >>> config.get_model_for_provider("anthropic")
            'claude-sonnet-4-20250514'
        """
        model_map = {
            "anthropic": self.anthropic_model,
            "bedrock": self.bedrock_model,
            "ollama": self.ollama_model,
        }
        return model_map.get(provider)

    def set_model_for_provider(self, provider: str, model: str) -> None:
        """
        Set the model for a specific provider.

        Args:
            provider: Provider name (anthropic | bedrock | ollama)
            model: Model name/ID to set

        Example:
            >>> config = WorkspaceProviderConfig()
            >>> config.set_model_for_provider("anthropic", "claude-sonnet-4-20250514")
        """
        if provider == "anthropic":
            self.anthropic_model = model
        elif provider == "bedrock":
            self.bedrock_model = model
        elif provider == "ollama":
            self.ollama_model = model
        else:
            raise ValueError(f"Unknown provider: {provider}")
