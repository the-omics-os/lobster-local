"""
Application settings and configuration.

This module centralizes all configuration settings for the application,
including the new professional agent configuration system.
"""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from lobster.config.agent_config import initialize_configurator
from lobster.config.supervisor_config import SupervisorConfig


class Settings:
    """
    Application settings with environment variable support.

    This class manages application-wide settings with fallbacks
    and environment variable overrides for easier configuration
    in different environments, especially in containers.
    """

    def __init__(self):
        """Initialize application settings."""
        # Load dotenv
        load_dotenv()

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
        self.CACHE_DIR = os.environ.get(
            "LOBSTER_CACHE_DIR", str(self.BASE_DIR / "data" / "cache")
        )

        # Create cache directory if it doesn't exist
        Path(self.CACHE_DIR).mkdir(parents=True, exist_ok=True)

        # API keys
        self.OPENAI_API_KEY = os.environ.get(
            "OPENAI_API_KEY", ""
        )  # TODO future support
        self.AWS_BEDROCK_ACCESS_KEY = os.environ.get("AWS_BEDROCK_ACCESS_KEY", "")
        self.AWS_BEDROCK_SECRET_ACCESS_KEY = os.environ.get(
            "AWS_BEDROCK_SECRET_ACCESS_KEY", ""
        )
        self.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
        self.NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
        self.NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")

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

        # GEO database settings
        self.GEO_CACHE_DIR = os.path.join(self.CACHE_DIR, "geo")
        Path(self.GEO_CACHE_DIR).mkdir(parents=True, exist_ok=True)

        # SSL/HTTPS settings
        self.SSL_VERIFY = os.environ.get("LOBSTER_SSL_VERIFY", "true").lower() == "true"
        self.SSL_CERT_PATH = os.environ.get("LOBSTER_SSL_CERT_PATH", None)

        # Validate configuration and store error for later check in CLI
        is_valid, error_msg = self.validate_configuration()
        if not is_valid:
            # Don't raise here - let CLI handle it gracefully
            self._config_error = error_msg
        else:
            self._config_error = None

    @property
    def llm_provider(self) -> str:
        """Detect which LLM provider to use based on available credentials."""
        if os.environ.get("LOBSTER_LLM_PROVIDER"):
            return os.environ.get("LOBSTER_LLM_PROVIDER")
        elif self.ANTHROPIC_API_KEY:
            return "anthropic"
        elif self.AWS_BEDROCK_ACCESS_KEY and self.AWS_BEDROCK_SECRET_ACCESS_KEY:
            return "bedrock"
        else:
            return "bedrock"  # Default for backward compatibility

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

    def validate_configuration(self) -> tuple:
        """
        Validate that required configuration is present.

        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        if not self.ANTHROPIC_API_KEY and not (
            self.AWS_BEDROCK_ACCESS_KEY and self.AWS_BEDROCK_SECRET_ACCESS_KEY
        ):
            error_msg = """
❌ No LLM provider configured

Lobster AI requires API credentials to function.

Quick Setup:
1. Create a .env file in your current directory
2. Add ONE of the following:

   Option A - Claude API (Recommended for testing):
   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

   Option B - AWS Bedrock (Recommended for production):
   AWS_BEDROCK_ACCESS_KEY=your-access-key
   AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret-key

Get API Keys:
  • Claude API: https://console.anthropic.com/
  • AWS Bedrock: https://aws.amazon.com/bedrock/

For detailed setup instructions, see:
  https://github.com/the-omics-os/lobster-local/wiki/03-configuration
"""
            return False, error_msg

        return True, ""

    def get_agent_llm_params(self, agent_name: str) -> Dict[str, Any]:
        """
        Get LLM parameters for a specific agent using the new configuration system.

        Args:
            agent_name: Name of the agent (e.g., 'supervisor', 'transcriptomics_expert', 'method_agent')

        Returns:
            Dictionary of LLM initialization parameters
        """
        try:
            params = self.agent_configurator.get_llm_params(agent_name)
            # Add provider information
            params["provider"] = self.llm_provider
            return params
        except KeyError as k:
            # Fallback to data_expert settings with warning
            print(f"⚠️  WARNING: No configuration found for agent '{agent_name}'")
            print("⚠️  Falling back to data_expert_agent configuration")
            print(f"⚠️  To fix: Add '{agent_name}' to all profiles in agent_config.py")

            try:
                # Use data_expert as fallback
                params = self.agent_configurator.get_llm_params("data_expert_agent")
                params["provider"] = self.llm_provider
                return params
            except KeyError:
                # If data_expert doesn't exist either, raise original error
                raise KeyError(f"{k}")

    def get_assistant_llm_params(self, agent_name: str) -> Dict[str, Any]:
        """
        Get LLM parameters for a specific assistants using the new configuration system.

        Args:
            agent_name: Name of the agent (e.g., 'supervisor', 'transcriptomics_expert', 'method_agent')

        Returns:
            Dictionary of LLM initialization parameters
        """
        try:
            return self.agent_configurator.get_llm_params(agent_name)
        except KeyError as k:
            # Fallback to data_expert settings with warning
            print(f"⚠️  WARNING: No configuration found for assistant '{agent_name}'")
            print("⚠️  Falling back to data_expert_agent configuration")
            print(f"⚠️  To fix: Add '{agent_name}' to all profiles in agent_config.py")

            try:
                # Use data_expert as fallback
                return self.agent_configurator.get_llm_params("data_expert_agent")
            except KeyError:
                # If data_expert doesn't exist either, raise original error
                raise KeyError(f"{k}")

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
# Pricing as of January 2025 - Update as needed
# Source: https://www.anthropic.com/pricing
MODEL_PRICING = {
    # AWS Bedrock Model IDs - Official Pricing (as of Nov 2024)
    # Source: https://aws.amazon.com/bedrock/pricing/
    # Claude Haiku 4.5
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": {
        "input_per_million": 1.00,  # $0.001 per 1K tokens
        "output_per_million": 5.00,  # $0.005 per 1K tokens
        "display_name": "Claude 4.5 Haiku",
    },
    # Claude Sonnet 4
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {
        "input_per_million": 3.00,  # $0.003 per 1K tokens
        "output_per_million": 15.00,  # $0.015 per 1K tokens
        "display_name": "Claude 4 Sonnet",
    },
    # Claude Sonnet 4.5 (Standard Context)
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {
        "input_per_million": 3.00,  # $0.003 per 1K tokens
        "output_per_million": 15.00,  # $0.015 per 1K tokens
        "display_name": "Claude 4.5 Sonnet",
    },
    # Note: Long Context variants use different pricing:
    # - Sonnet 4 Long Context: $6.00/$22.50 per million ($0.006/$0.0225 per 1K)
    # - Sonnet 4.5 Long Context: $6.00/$22.50 per million ($0.006/$0.0225 per 1K)
    # Add model IDs here when they become available
    # Anthropic Direct API Model IDs (same pricing)
    "claude-4-5-haiku": {
        "input_per_million": 1.00,
        "output_per_million": 5.00,
        "display_name": "Claude 4.5 Haiku",
    },
    "claude-4-sonnet": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "display_name": "Claude 4 Sonnet",
    },
    "claude-4-5-sonnet": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "display_name": "Claude 4.5 Sonnet",
    },
    # Legacy Claude 3.5 models (for backward compatibility)
    "claude-3-5-sonnet-20240620": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "display_name": "Claude 3.5 Sonnet",
    },
    "claude-3-5-sonnet-20241022": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "display_name": "Claude 3.5 Sonnet (v2)",
    },
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "display_name": "Claude 3.5 Sonnet",
    },
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "display_name": "Claude 3.5 Sonnet (v2)",
    },
    # Claude 3 Opus (legacy)
    "claude-3-opus-20240229": {
        "input_per_million": 15.00,
        "output_per_million": 75.00,
        "display_name": "Claude 3 Opus",
    },
    "us.anthropic.claude-3-opus-20240229-v1:0": {
        "input_per_million": 15.00,
        "output_per_million": 75.00,
        "display_name": "Claude 3 Opus",
    },
}


# Create singleton instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the application settings.

    Returns:
        Settings: Application settings
    """
    return settings
