"""
Provider setup utilities for LLM configuration.

This module contains pure logic for detecting, validating, and configuring
LLM providers (Anthropic, Bedrock, Ollama). It is separated from CLI
presentation logic for better testability and reusability.
"""

import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class OllamaStatus:
    """Status of Ollama installation and server."""

    installed: bool
    version: Optional[str] = None
    running: bool = False
    models: List[str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = []


@dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""

    provider_type: str  # "anthropic", "bedrock", "ollama"
    env_vars: Dict[str, str]
    success: bool = True
    message: Optional[str] = None


# =============================================================================
# Ollama Detection & Checking
# =============================================================================


def check_ollama_installation() -> Tuple[bool, Optional[str]]:
    """
    Check if Ollama is installed on the system.

    Returns:
        Tuple of (installed: bool, version: Optional[str])
    """
    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Parse version from output (e.g., "ollama version 0.1.23")
            version = result.stdout.strip().split()[-1] if result.stdout else None
            return True, version
        return False, None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, None


def check_ollama_server() -> bool:
    """
    Check if Ollama server is running.

    Returns:
        bool: True if server is accessible, False otherwise
    """
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_ollama_models() -> List[str]:
    """
    Get list of available Ollama models.

    Returns:
        List[str]: List of model names, or empty list if none available
    """
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Parse model names from output
            lines = [
                line.strip()
                for line in result.stdout.split("\n")
                if line.strip() and not line.startswith("NAME")
            ]
            # Extract first column (model name)
            models = [line.split()[0] for line in lines if line.split()]
            return models
        return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def get_ollama_status() -> OllamaStatus:
    """
    Get comprehensive Ollama status.

    Returns:
        OllamaStatus: Complete status including installation, server, and models
    """
    installed, version = check_ollama_installation()

    if not installed:
        return OllamaStatus(installed=False)

    running = check_ollama_server()
    models = get_ollama_models() if running else []

    return OllamaStatus(
        installed=installed, version=version, running=running, models=models
    )


# =============================================================================
# Provider Configuration
# =============================================================================


def create_anthropic_config(api_key: str) -> ProviderConfig:
    """
    Create configuration for Anthropic Direct API.

    Args:
        api_key: Anthropic API key

    Returns:
        ProviderConfig with environment variables
    """
    if not api_key or not api_key.strip():
        return ProviderConfig(
            provider_type="anthropic",
            env_vars={},
            success=False,
            message="API key cannot be empty",
        )

    return ProviderConfig(
        provider_type="anthropic",
        env_vars={
            "LOBSTER_LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": api_key.strip(),
        },
        success=True,
    )


def create_bedrock_config(access_key: str, secret_key: str) -> ProviderConfig:
    """
    Create configuration for AWS Bedrock.

    Args:
        access_key: AWS access key
        secret_key: AWS secret key

    Returns:
        ProviderConfig with environment variables
    """
    if not access_key or not access_key.strip() or not secret_key or not secret_key.strip():
        return ProviderConfig(
            provider_type="bedrock",
            env_vars={},
            success=False,
            message="AWS credentials cannot be empty",
        )

    return ProviderConfig(
        provider_type="bedrock",
        env_vars={
            "LOBSTER_LLM_PROVIDER": "bedrock",
            "AWS_BEDROCK_ACCESS_KEY": access_key.strip(),
            "AWS_BEDROCK_SECRET_ACCESS_KEY": secret_key.strip(),
        },
        success=True,
    )


def create_ollama_config(
    model_name: Optional[str] = None, custom_url: Optional[str] = None
) -> ProviderConfig:
    """
    Create configuration for Ollama (local LLM).

    Args:
        model_name: Optional model name (default: llama3:8b-instruct)
        custom_url: Optional custom Ollama server URL

    Returns:
        ProviderConfig with environment variables
    """
    env_vars = {"LOBSTER_LLM_PROVIDER": "ollama"}

    if model_name:
        env_vars["OLLAMA_DEFAULT_MODEL"] = model_name.strip()

    if custom_url:
        env_vars["OLLAMA_BASE_URL"] = custom_url.strip()

    return ProviderConfig(provider_type="ollama", env_vars=env_vars, success=True)


# =============================================================================
# Installation Instructions
# =============================================================================


def get_ollama_install_instructions() -> Dict[str, str]:
    """
    Get platform-specific Ollama installation instructions.

    Returns:
        Dict with platform keys and installation command values
    """
    return {
        "macos_linux": "curl -fsSL https://ollama.com/install.sh | sh",
        "windows": "Download from https://ollama.com/download",
        "general": "Visit https://ollama.com for installation instructions",
    }


def get_recommended_models() -> List[Dict[str, str]]:
    """
    Get list of recommended Ollama models with descriptions.

    Returns:
        List of dicts with model info (name, ram_required, description)
    """
    return [
        {
            "name": "llama3:8b-instruct",
            "ram_required": "8-16GB",
            "description": "Recommended for testing (fast, good baseline)",
        },
        {
            "name": "mixtral:8x7b-instruct",
            "ram_required": "24-32GB",
            "description": "Better quality for production workflows",
        },
        {
            "name": "llama3:70b-instruct",
            "ram_required": "48GB VRAM",
            "description": "Maximum quality (requires GPU)",
        },
    ]


# =============================================================================
# Validation
# =============================================================================


def validate_provider_choice(
    has_anthropic: bool, has_bedrock: bool, has_ollama: bool
) -> Tuple[bool, Optional[str]]:
    """
    Validate that at least one provider is configured.

    Args:
        has_anthropic: Whether Anthropic API key is provided
        has_bedrock: Whether Bedrock credentials are provided
        has_ollama: Whether Ollama is selected

    Returns:
        Tuple of (valid: bool, error_message: Optional[str])
    """
    if not has_anthropic and not has_bedrock and not has_ollama:
        return False, "No provider specified. You must provide one of: Claude API, AWS Bedrock, or Ollama"

    return True, None


def get_provider_priority_warning(
    has_anthropic: bool, has_bedrock: bool, has_ollama: bool
) -> Optional[str]:
    """
    Get warning message if multiple providers are configured.

    Args:
        has_anthropic: Whether Anthropic is configured
        has_bedrock: Whether Bedrock is configured
        has_ollama: Whether Ollama is configured

    Returns:
        Optional warning message if multiple providers detected
    """
    providers_count = sum([has_anthropic, has_bedrock, has_ollama])

    if providers_count <= 1:
        return None

    # Return priority order message
    if has_anthropic:
        return "Multiple providers specified. Using Claude API (highest priority)."
    elif has_bedrock:
        return "Multiple providers specified. Using AWS Bedrock (second priority)."
    elif has_ollama:
        return "Multiple providers specified. Using Ollama (third priority)."

    return None
