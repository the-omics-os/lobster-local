"""
Application settings and configuration.

This module centralizes all configuration settings for the application,
including the new professional agent configuration system.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from lobster.config.agent_config import initialize_configurator
from lobster.config.supervisor_config import SupervisorConfig

logger = logging.getLogger(__name__)


class Settings:
    """
    Application settings with environment variable support.

    This class manages application-wide settings with fallbacks
    and environment variable overrides for easier configuration
    in different environments, especially in containers.

    Credential Loading Priority (lowest to highest):
    1. Global credentials (~/.config/lobster/credentials.env)
    2. Workspace credentials (./.env or workspace/.env)
    3. Environment variables (explicit exports)

    Higher priority sources override lower priority ones.
    """

    def __init__(self):
        """Initialize application settings."""
        # Load credentials in priority order (lowest first, so higher priority overrides)
        self._load_credentials()

        # hackathon
        self.LINKUP_API_KEY = os.environ.get("LINKUP_API_KEY", "")

        # CDK variables (used by lobster-cloud deployment)
        self.STACK_NAME = "LobsterStack"
        self.CDK_DEPLY_ACCOUNT = "649207544517"
        # AWS Fargate CPU/Memory options summary:
        # - 256 (.25 vCPU): 512 MiB, 1 GB, 2 GB (Linux)
        # - 512 (.5 vCPU): 1 GB, 2 GB, 3 GB, 4 GB (Linux)
        # - 1024 (1 vCPU): 2-8 GB (Linux, Windows)
        # - 2048 (2 vCPU): 4-16 GB (Linux, Windows, 1 GB steps)
        # - 4096 (4 vCPU): 8-30 GB (Linux, Windows, 1 GB steps)
        # - 8192 (8 vCPU, Linux 1.4.0+): 16-60 GB (4 GB steps)
        # - 16384 (16 vCPU, Linux 1.4.0+): 32-120 GB (8 GB steps)
        # See AWS docs for full details.
        self.MEMORY = 24576
        self.CPU = 8192
        # Initialize agent configurator based on environment
        profile = os.environ.get("LOBSTER_PROFILE", "production")
        self.agent_configurator = initialize_configurator(profile=profile)

        # Initialize supervisor configuration
        self.supervisor_config = SupervisorConfig.from_env()

        # Base directories
        self.BASE_DIR = Path(__file__).resolve().parent.parent

        # API keys
        self.AWS_BEDROCK_ACCESS_KEY = os.environ.get("AWS_BEDROCK_ACCESS_KEY", "")
        self.AWS_BEDROCK_SECRET_ACCESS_KEY = os.environ.get(
            "AWS_BEDROCK_SECRET_ACCESS_KEY", ""
        )
        self.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
        self.NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
        self.NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")
        # Multi-key support for parallelization (NCBI_API_KEY_1, NCBI_API_KEY_2, etc.)
        self._ncbi_api_keys = self._load_ncbi_api_keys()

        # Ollama settings (local LLM support)
        self.OLLAMA_BASE_URL = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.OLLAMA_DEFAULT_MODEL = os.environ.get(
            "OLLAMA_DEFAULT_MODEL", "gpt-oss:20b"
        )

        # Logging settings
        self.LOG_LEVEL = os.environ.get("LOBSTER_LOG_LEVEL", "WARNING").upper()

        # Git automation settings (for sync scripts and CI/CD)
        self.GIT_USER_NAME = os.environ.get("GIT_USER_NAME", "")
        self.GIT_USER_EMAIL = os.environ.get("GIT_USER_EMAIL", "")

        # AWS region (fallback for backward compatibility)
        self.REGION = os.environ.get("AWS_REGION", "us-east-1")

        # Web server settings (for 'lobster serve' FastAPI server)
        self.PORT = int(os.environ.get("PORT", "8000"))
        self.HOST = os.environ.get("HOST", "0.0.0.0")
        self.DEBUG = os.environ.get("DEBUG", "False").lower() == "true"

        # Data processing settings
        self.MAX_FILE_SIZE_MB = int(os.environ.get("LOBSTER_MAX_FILE_SIZE_MB", "500"))
        self.DEFAULT_CLUSTER_RESOLUTION = float(
            os.environ.get("LOBSTER_CLUSTER_RESOLUTION", "0.5")
        )

        # SSL/HTTPS settings
        self.SSL_VERIFY = os.environ.get("LOBSTER_SSL_VERIFY", "true").lower() == "true"
        self.SSL_CERT_PATH = os.environ.get("LOBSTER_SSL_CERT_PATH", None)

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all settings as a dictionary.

        Returns:
            dict: All settings
        """
        settings_dict = {}
        for attr in dir(self):
            if not attr.startswith("_") and not callable(getattr(self, attr)):
                settings_dict[attr] = getattr(self, attr)
        return settings_dict

    def get_setting(self, name: str, default: Any = None) -> Any:
        """
        Get a specific setting.

        Args:
            name: Setting name
            default: Default value if setting doesn't exist

        Returns:
            Value of the setting or default
        """
        return getattr(self, name, default)

    def _load_credentials(self, workspace_path: Path = None) -> None:
        """
        Load credentials from global and workspace sources.

        Loading Order (lowest to highest priority):
        1. Global credentials (~/.config/lobster/credentials.env)
        2. Workspace credentials (workspace/.env or ./.env)

        Higher priority sources override lower priority ones.
        Environment variables (explicit exports) always take highest priority.

        Args:
            workspace_path: Optional explicit workspace path. If None, uses CWD.

        Example:
            >>> settings._load_credentials()  # Load from CWD
            >>> settings._load_credentials(Path("/path/to/workspace"))  # Explicit workspace
        """
        from lobster.config.global_config import (
            get_global_credentials_path,
            global_credentials_exist,
        )

        # Step 1: Load global credentials (lowest priority, override=False preserves existing env vars)
        if global_credentials_exist():
            global_creds_path = get_global_credentials_path()
            load_dotenv(global_creds_path, override=False)
            logger.debug(f"Loaded global credentials from {global_creds_path}")

        # Step 2: Load workspace credentials (higher priority, overrides global)
        # Determine workspace .env path
        if workspace_path:
            workspace_env = workspace_path / ".env"
        else:
            # Default: look in current working directory
            workspace_env = Path.cwd() / ".env"

        if workspace_env.exists():
            load_dotenv(workspace_env, override=True)
            logger.debug(f"Loaded workspace credentials from {workspace_env}")
        else:
            # Fallback: let load_dotenv find .env in default locations
            load_dotenv(override=True)

    def reload_credentials(self, workspace_path: Path) -> None:
        """
        Reload credentials for a specific workspace.

        Call this when switching workspaces to ensure correct credentials are loaded.

        Args:
            workspace_path: Path to the workspace directory

        Example:
            >>> settings.reload_credentials(Path("/new/workspace"))
        """
        self._load_credentials(workspace_path)
        # Refresh API key attributes
        self.AWS_BEDROCK_ACCESS_KEY = os.environ.get("AWS_BEDROCK_ACCESS_KEY", "")
        self.AWS_BEDROCK_SECRET_ACCESS_KEY = os.environ.get("AWS_BEDROCK_SECRET_ACCESS_KEY", "")
        self.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
        self.NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
        self._ncbi_api_keys = self._load_ncbi_api_keys()
        logger.info(f"Reloaded credentials for workspace: {workspace_path}")

    def _load_ncbi_api_keys(self) -> List[str]:
        """
        Load all NCBI API keys from environment variables.

        Supports multiple keys for parallelization:
        - NCBI_API_KEY (primary, backward compatible)
        - NCBI_API_KEY_1, NCBI_API_KEY_2, ... NCBI_API_KEY_9

        Returns:
            List of API keys found in environment
        """
        keys = []
        # Primary key (backward compatible)
        primary = os.environ.get("NCBI_API_KEY", "")
        if primary:
            keys.append(primary)
        # Additional keys for parallelization
        for i in range(1, 10):
            key = os.environ.get(f"NCBI_API_KEY_{i}", "")
            if key:
                keys.append(key)
        return keys

    def get_ncbi_api_key(self, index: int = 0) -> str:
        """
        Get NCBI API key by index (for parallelization).

        Uses round-robin if index exceeds available keys.

        Args:
            index: Worker index (0-based)

        Returns:
            API key string, or empty string if no keys configured
        """
        if not self._ncbi_api_keys:
            return ""
        return self._ncbi_api_keys[index % len(self._ncbi_api_keys)]

    def get_all_ncbi_api_keys(self) -> List[str]:
        """
        Get all available NCBI API keys for parallel workers.

        Returns:
            List of all configured API keys
        """
        return self._ncbi_api_keys.copy()

    def get_ncbi_key_count(self) -> int:
        """
        Get the number of NCBI API keys available.

        Returns:
            Number of keys (useful for determining parallelization factor)
        """
        return len(self._ncbi_api_keys)

    def get_agent_llm_params(self, agent_name: str) -> Dict[str, Any]:
        """
        Get LLM parameters for a specific agent using the new configuration system.

        Resolution order:
        1. Custom agent config (from entry points)
        2. Profile config (FREE tier agents)
        3. Fallback to data_expert_agent

        Args:
            agent_name: Name of the agent (e.g., 'supervisor', 'transcriptomics_expert', 'method_agent')

        Returns:
            Dictionary of LLM initialization parameters

        Note:
            Provider information is resolved by ConfigResolver, not Settings.
        """
        try:
            params = self.agent_configurator.get_llm_params(agent_name)
            return params
        except KeyError:
            # Check if this is a premium agent without custom config
            is_premium_agent = self._is_premium_agent(agent_name)

            if is_premium_agent:
                # Premium agent without custom config - use INFO log (expected behavior)
                logger.info(
                    f"No custom configuration for premium agent '{agent_name}'. "
                    f"Using default configuration (data_expert_agent). "
                    f"To customize: Add entry point in custom package."
                )
            else:
                # FREE tier agent missing from profiles - this may be a configuration issue
                logger.warning(
                    f"No configuration for agent '{agent_name}'. "
                    f"Falling back to data_expert_agent. "
                    f"To fix: Add '{agent_name}' to all profiles in agent_config.py"
                )

            try:
                # Use data_expert as fallback
                params = self.agent_configurator.get_llm_params("data_expert_agent")
                return params
            except KeyError as e:
                # If data_expert doesn't exist either, raise error
                raise KeyError(
                    f"Fallback configuration failed for '{agent_name}'. "
                    f"data_expert_agent not found: {e}"
                )

    def _is_premium_agent(self, agent_name: str) -> bool:
        """
        Check if an agent is a premium agent (not in FREE tier).

        Args:
            agent_name: Agent name to check

        Returns:
            True if the agent is premium-only (not in FREE tier)
        """
        try:
            from lobster.config.subscription_tiers import is_agent_available

            # Agent is premium if it's available in premium but not in free tier
            is_available_premium = is_agent_available(agent_name, "premium")
            is_available_free = is_agent_available(agent_name, "free")

            return is_available_premium and not is_available_free
        except ImportError:
            # subscription_tiers not available, assume not premium
            return False
        except Exception:
            # Any error, assume not premium
            return False

    def get_assistant_llm_params(self, agent_name: str) -> Dict[str, Any]:
        """
        Get LLM parameters for a specific assistant using the new configuration system.

        Args:
            agent_name: Name of the assistant (e.g., 'data_expert_assistant')

        Returns:
            Dictionary of LLM initialization parameters
        """
        try:
            return self.agent_configurator.get_llm_params(agent_name)
        except KeyError:
            # Assistants use INFO level logging for fallback (expected behavior)
            logger.info(
                f"No configuration for assistant '{agent_name}'. "
                f"Using default configuration (data_expert_agent)."
            )

            try:
                # Use data_expert as fallback
                return self.agent_configurator.get_llm_params("data_expert_agent")
            except KeyError as e:
                # If data_expert doesn't exist either, raise error
                raise KeyError(
                    f"Fallback configuration failed for assistant '{agent_name}'. "
                    f"data_expert_agent not found: {e}"
                )

    def get_agent_model_config(self, agent_name: str):
        """
        Get model configuration for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            ModelConfig object for the agent
        """
        return self.agent_configurator.get_model_config(agent_name)

    def get_supervisor_config(self) -> SupervisorConfig:
        """
        Get supervisor configuration.

        Returns:
            SupervisorConfig: Supervisor configuration instance
        """
        return self.supervisor_config

    def print_agent_configuration(self):
        """Print current agent configuration."""
        self.agent_configurator.print_current_config()


# Model pricing configuration (USD per million tokens)
# Dynamically loaded from ProviderRegistry - single source of truth
# Pricing is defined in each provider's ModelInfo objects:
#   - lobster/config/providers/anthropic_provider.py
#   - lobster/config/providers/bedrock_provider.py
#   - lobster/config/providers/gemini_provider.py
#   - lobster/config/providers/ollama_provider.py (free - pricing=0.0)

# Cache for lazy-loaded pricing (avoids circular imports)
_MODEL_PRICING_CACHE: Dict[str, Any] = None


def get_model_pricing() -> Dict[str, Any]:
    """
    Get model pricing dictionary from all registered providers.

    Uses lazy loading to avoid circular imports. Pricing is defined
    in each provider's ModelInfo objects (single source of truth).

    Returns:
        Dict mapping model names to pricing info:
        {
            "model-name": {
                "input_per_million": float,
                "output_per_million": float,
                "display_name": str
            }
        }

    Example:
        >>> pricing = get_model_pricing()
        >>> gemini_cost = pricing.get("gemini-3-pro-preview")
        >>> if gemini_cost:
        ...     print(f"Input: ${gemini_cost['input_per_million']}/M tokens")
    """
    global _MODEL_PRICING_CACHE

    if _MODEL_PRICING_CACHE is None:
        from lobster.config.providers.registry import ProviderRegistry

        _MODEL_PRICING_CACHE = ProviderRegistry.get_all_models_with_pricing()

    return _MODEL_PRICING_CACHE


def reset_model_pricing_cache() -> None:
    """
    Reset the pricing cache (for testing or after provider changes).

    This forces the next call to get_model_pricing() to reload from providers.
    """
    global _MODEL_PRICING_CACHE
    _MODEL_PRICING_CACHE = None


# Backward compatibility: MODEL_PRICING as module-level dict
# Populated on first access via get_model_pricing()
MODEL_PRICING: Dict[str, Any] = get_model_pricing()


# Default pricing for unknown models (use Sonnet as reasonable default)
DEFAULT_PRICING = {
    "input_per_million": 3.0,
    "output_per_million": 15.0,
}


def calculate_token_cost(
    input_tokens: int,
    output_tokens: int,
    model_id: str = None,
) -> float:
    """
    Calculate cost in USD for token usage.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_id: Model identifier (optional, uses default if not found)

    Returns:
        Cost in USD
    """
    pricing = MODEL_PRICING.get(model_id, DEFAULT_PRICING) if model_id else DEFAULT_PRICING

    input_cost = (input_tokens / 1_000_000) * pricing.get(
        "input_per_million", DEFAULT_PRICING["input_per_million"]
    )
    output_cost = (output_tokens / 1_000_000) * pricing.get(
        "output_per_million", DEFAULT_PRICING["output_per_million"]
    )

    return input_cost + output_cost


# Create singleton instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the application settings.

    Returns:
        Settings: Application settings
    """
    return settings
