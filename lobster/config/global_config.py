"""
Global user-level provider configuration (cross-project defaults).

This module provides user-level configuration management that applies across
all Lobster workspaces unless overridden by workspace-specific settings.

Storage Location (platform-specific):
- Linux:   ~/.config/lobster/providers.json
- macOS:   ~/Library/Application Support/lobster/providers.json
- Windows: %APPDATA%\\lobster\\providers.json

Priority Hierarchy:
1. Runtime overrides (CLI flags)
2. Workspace config (.lobster_workspace/provider_config.json)
3. Global user config (platform-specific path) ← This module
4. Environment variables (.env file)
5. Auto-detection
6. Defaults

Example:
    >>> from lobster.config.global_config import GlobalProviderConfig
    >>>
    >>> # Load user-level config
    >>> config = GlobalProviderConfig.load()
    >>>
    >>> # Set defaults for all projects
    >>> config.default_provider = "ollama"
    >>> config.ollama_default_model = "mixtral:8x7b-instruct"
    >>> config.save()
"""

import json
import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Optional

from pydantic import Field

from lobster.config.base_config import ProviderConfigBase

logger = logging.getLogger(__name__)

# Configuration file location (cross-platform)
# Unix (Linux/macOS): ~/.config/lobster/ (CLI convention - consistent across Unix)
# Windows: %APPDATA%\lobster\ (Windows convention)
if platform.system() == "Windows":
    CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / "lobster"
else:
    # Unix: Use XDG_CONFIG_HOME or fallback to ~/.config (same for Linux and macOS)
    CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "lobster"

CONFIG_FILE_NAME = "providers.json"

# Legacy config location - no longer used, kept for migration only
LEGACY_CONFIG_DIR = None  # Migration removed since we're keeping ~/.config/ on Unix


class GlobalProviderConfig(ProviderConfigBase):
    """
    Global user-level defaults for LLM providers and models.

    These settings apply across all Lobster projects unless overridden
    by workspace-specific configurations.

    Attributes:
        default_provider: Default LLM provider (bedrock | anthropic | ollama | gemini)
        default_profile: Default agent configuration profile
        anthropic_default_model: Default Anthropic model for all projects
        bedrock_default_model: Default Bedrock model for all projects
        ollama_default_model: Default Ollama model for all projects
        ollama_default_host: Default Ollama server URL

    Example:
        >>> config = GlobalProviderConfig(
        ...     default_provider="anthropic",
        ...     anthropic_default_model="claude-sonnet-4-20250514",
        ...     default_profile="production"
        ... )
        >>> config.save()
    """

    @property
    def provider_field_name(self) -> str:
        return "default_provider"

    @property
    def model_field_suffix(self) -> str:
        return "_default_model"

    default_provider: Optional[str] = Field(
        None,
        description="Default LLM provider (bedrock | anthropic | ollama | gemini)"
    )

    default_profile: str = Field(
        "production",
        description="Default agent configuration profile"
    )

    # Per-provider default models
    anthropic_default_model: Optional[str] = Field(
        None,
        description="Default Anthropic model (e.g., 'claude-sonnet-4-20250514')"
    )

    bedrock_default_model: Optional[str] = Field(
        None,
        description="Default Bedrock model ID (e.g., 'anthropic.claude-3-5-sonnet-20241022-v2:0')"
    )

    ollama_default_model: Optional[str] = Field(
        None,
        description="Default Ollama model (e.g., 'llama3:70b-instruct')"
    )

    gemini_default_model: Optional[str] = Field(
        None,
        description="Default Gemini model (e.g., 'gemini-3-pro-preview')"
    )

    ollama_default_host: str = Field(
        "http://localhost:11434",
        description="Default Ollama server URL"
    )

    def save(self) -> None:
        """
        Save global configuration to user config directory.

        Creates ~/.config/lobster/ directory if it doesn't exist.

        Raises:
            IOError: If write operation fails

        Example:
            >>> config = GlobalProviderConfig.load()
            >>> config.default_provider = "ollama"
            >>> config.save()
        """
        config_path = CONFIG_DIR / CONFIG_FILE_NAME

        # Ensure config directory exists
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        try:
            # Write with indentation for human readability
            config_path.write_text(self.model_dump_json(indent=2))
            logger.info(f"Saved global config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save global config: {e}")
            raise

    @classmethod
    def load(cls) -> "GlobalProviderConfig":
        """
        Load global configuration with graceful error handling.

        Handles:
        - Missing file: Returns default configuration
        - Corrupted JSON: Logs warning, returns defaults
        - Invalid schema: Logs validation errors, returns defaults

        Returns:
            GlobalProviderConfig: Loaded or default configuration

        Example:
            >>> config = GlobalProviderConfig.load()
            >>> if config.default_provider:
            ...     print(f"Global default: {config.default_provider}")
        """
        config_path = CONFIG_DIR / CONFIG_FILE_NAME

        # File doesn't exist - return defaults
        if not config_path.exists():
            logger.debug(
                f"No global config found at {config_path}, using defaults"
            )
            return cls()

        try:
            # Parse JSON
            data = json.loads(config_path.read_text())
            config = cls(**data)
            logger.info(f"Loaded global config from {config_path}")
            return config

        except json.JSONDecodeError as e:
            logger.warning(
                f"Corrupted global config file at {config_path}: {e}. Using defaults."
            )
            return cls()

        except Exception as e:
            # Catch Pydantic validation errors and other exceptions
            logger.warning(
                f"Invalid global config schema at {config_path}: {e}. Using defaults."
            )
            return cls()

    @classmethod
    def exists(cls) -> bool:
        """
        Check if global configuration file exists.

        Returns:
            bool: True if configuration file exists

        Example:
            >>> if GlobalProviderConfig.exists():
            ...     config = GlobalProviderConfig.load()
            ... else:
            ...     print("No global config - using defaults")
        """
        config_path = CONFIG_DIR / CONFIG_FILE_NAME
        return config_path.exists()

    def reset(self) -> None:
        """
        Reset configuration to defaults.

        Example:
            >>> config = GlobalProviderConfig.load()
            >>> config.reset()
            >>> config.save()  # Saves defaults
        """
        self.default_provider = None
        self.default_profile = "production"
        self.anthropic_default_model = None
        self.bedrock_default_model = None
        self.ollama_default_model = None
        self.gemini_default_model = None
        self.ollama_default_host = "http://localhost:11434"
        logger.info("Reset global config to defaults")

    @classmethod
    def get_config_path(cls) -> Path:
        """
        Get the path to the global configuration file.

        Returns:
            Path: Path to ~/.config/lobster/providers.json

        Example:
            >>> path = GlobalProviderConfig.get_config_path()
            >>> print(f"Config location: {path}")
        """
        return CONFIG_DIR / CONFIG_FILE_NAME
