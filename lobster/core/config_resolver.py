"""
Configuration resolver with transparent decision logging.

This module implements the 6-layer priority hierarchy for provider and model
selection, with clear logging of why each decision was made.

Priority Order (highest to lowest):
1. Runtime overrides (CLI flags like --provider, --model)
2. Workspace config (.lobster_workspace/provider_config.json)
3. Global user config (~/.config/lobster/providers.json)
4. Environment variables (.env file: LOBSTER_LLM_PROVIDER, etc.)
5. Auto-detection (Ollama running, API keys present)
6. Hardcoded defaults (production profile, bedrock provider)

Example:
    >>> from lobster.core.config_resolver import ConfigResolver
    >>> from pathlib import Path
    >>>
    >>> resolver = ConfigResolver(workspace_path=Path(".lobster_workspace"))
    >>> provider, source = resolver.resolve_provider()
    >>> print(f"Using {provider} (from {source})")
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from lobster.core.global_config import GlobalProviderConfig
from lobster.core.workspace_config import WorkspaceProviderConfig
from lobster.config.llm_factory import LLMFactory, LLMProvider

logger = logging.getLogger(__name__)


class ConfigResolver:
    """
    Resolve configuration with transparent decision logging.

    This class implements the priority hierarchy for provider and model selection,
    ensuring users understand why each decision was made.

    Attributes:
        workspace_path: Path to workspace directory
        workspace_config: Loaded workspace configuration
        global_config: Loaded global user configuration
    """

    def __init__(self, workspace_path: Optional[Path] = None):
        """
        Initialize resolver with optional workspace path.

        Args:
            workspace_path: Path to workspace directory (optional)

        Example:
            >>> resolver = ConfigResolver(Path(".lobster_workspace"))
        """
        self.workspace_path = workspace_path
        self.workspace_config = None
        self.global_config = None

        # Load configurations if workspace path provided
        if workspace_path:
            self.workspace_config = WorkspaceProviderConfig.load(workspace_path)
            self.global_config = GlobalProviderConfig.load()

    def resolve_provider(
        self,
        runtime_override: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Resolve LLM provider with transparent decision logging.

        Args:
            runtime_override: Explicit provider from CLI flag (highest priority)

        Returns:
            Tuple[str, str]: (provider_name, decision_source)

        Decision sources:
            - "runtime flag --provider"
            - "workspace config"
            - "global user config"
            - "environment variable LOBSTER_LLM_PROVIDER"
            - "auto-detected (Ollama running)"
            - "auto-detected (ANTHROPIC_API_KEY set)"
            - "auto-detected (AWS credentials set)"
            - "default (bedrock)"

        Example:
            >>> resolver = ConfigResolver(Path(".lobster_workspace"))
            >>> provider, source = resolver.resolve_provider()
            >>> logger.info(f"Using provider '{provider}' (from {source})")
        """
        # Layer 1: Runtime override (highest priority)
        if runtime_override:
            # Validate provider
            if runtime_override in ["bedrock", "anthropic", "ollama"]:
                return (runtime_override, "runtime flag --provider")
            else:
                logger.warning(
                    f"Invalid runtime provider '{runtime_override}', "
                    f"continuing to next priority level"
                )

        # Layer 2: Workspace config (only if file exists)
        if self.workspace_config and self.workspace_path:
            if WorkspaceProviderConfig.exists(self.workspace_path):
                if self.workspace_config.global_provider:
                    provider = self.workspace_config.global_provider
                    return (provider, "workspace config")

        # Layer 3: Global user config (only if file exists)
        if self.global_config:
            if GlobalProviderConfig.exists():
                if self.global_config.default_provider:
                    provider = self.global_config.default_provider
                    return (provider, "global user config")

        # Layer 4: Environment variable
        if env_provider := os.environ.get("LOBSTER_LLM_PROVIDER"):
            if env_provider in ["bedrock", "anthropic", "ollama"]:
                return (env_provider, "environment variable LOBSTER_LLM_PROVIDER")
            else:
                logger.warning(
                    f"Invalid LOBSTER_LLM_PROVIDER '{env_provider}', "
                    f"continuing to auto-detection"
                )

        # Layer 5: Auto-detection
        # Check Ollama first (local-first philosophy)
        if LLMFactory._is_ollama_running():
            return ("ollama", "auto-detected (Ollama running)")

        # Check Anthropic API key
        if os.environ.get("ANTHROPIC_API_KEY"):
            return ("anthropic", "auto-detected (ANTHROPIC_API_KEY set)")

        # Check AWS Bedrock credentials
        if os.environ.get("AWS_BEDROCK_ACCESS_KEY") and os.environ.get(
            "AWS_BEDROCK_SECRET_ACCESS_KEY"
        ):
            return ("bedrock", "auto-detected (AWS credentials set)")

        # Layer 6: Default fallback
        return ("bedrock", "default (no configuration found)")

    def resolve_model(
        self,
        agent_name: Optional[str] = None,
        runtime_override: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        """
        Resolve model for a specific agent with transparent decision logging.

        Now provider-aware: resolves the appropriate model for the current provider
        (anthropic, bedrock, or ollama).

        Args:
            agent_name: Name of agent (e.g., "supervisor", "data_expert")
            runtime_override: Explicit model from CLI flag (highest priority)
            provider: LLM provider (bedrock, anthropic, ollama)

        Returns:
            Tuple[Optional[str], str]: (model_name, decision_source)
                                       Returns (None, source) if no override

        Decision sources:
            - "runtime flag --model"
            - "workspace config (per-agent)"
            - "workspace config ({provider} model)"
            - "global user config ({provider} model)"
            - "environment variable"
            - "provider default"

        Example:
            >>> resolver = ConfigResolver(Path(".lobster_workspace"))
            >>> model, source = resolver.resolve_model("supervisor", provider="anthropic")
            >>> if model:
            ...     logger.info(f"Agent 'supervisor' → model '{model}' (from {source})")
        """
        # Layer 1: Runtime override
        if runtime_override:
            return (runtime_override, "runtime flag --model")

        # Layer 2: Workspace per-agent model override (only if file exists)
        if self.workspace_config and self.workspace_path and agent_name:
            if WorkspaceProviderConfig.exists(self.workspace_path):
                if agent_name in self.workspace_config.per_agent_models:
                    model = self.workspace_config.per_agent_models[agent_name]
                    return (model, f"workspace config (agent '{agent_name}')")

        # Layer 3: Workspace global model for the current provider (only if file exists)
        if self.workspace_config and self.workspace_path and provider:
            if WorkspaceProviderConfig.exists(self.workspace_path):
                model = self.workspace_config.get_model_for_provider(provider)
                if model:
                    return (model, f"workspace config ({provider} model)")

        # Layer 4: Global user config for the current provider (only if file exists)
        if self.global_config and provider:
            if GlobalProviderConfig.exists():
                model = self.global_config.get_model_for_provider(provider)
                if model:
                    return (model, f"global user config ({provider} model)")

        # Layer 5: Environment variable (provider-specific)
        env_var_map = {
            "ollama": "OLLAMA_DEFAULT_MODEL",
            "anthropic": "ANTHROPIC_MODEL",
            "bedrock": "BEDROCK_MODEL",
        }
        if provider and provider in env_var_map:
            env_var = env_var_map[provider]
            if env_model := os.environ.get(env_var):
                return (env_model, f"environment variable {env_var}")

        # Layer 6: Provider default model
        if provider:
            try:
                from lobster.config.model_service import ModelServiceFactory

                service = ModelServiceFactory.get_service(provider)
                default_model = service.get_default_model()
                if default_model:
                    return (default_model, f"provider default ({provider})")
            except Exception as e:
                logger.debug(f"Failed to get default model for {provider}: {e}")

        # No model override - let profile configuration handle it
        return (None, "profile configuration")

    def resolve_profile(
        self,
        runtime_override: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Resolve agent configuration profile with transparent decision logging.

        Args:
            runtime_override: Explicit profile from CLI flag (highest priority)

        Returns:
            Tuple[str, str]: (profile_name, decision_source)

        Decision sources:
            - "runtime flag --profile"
            - "workspace config"
            - "global user config"
            - "environment variable LOBSTER_PROFILE"
            - "default (production)"

        Example:
            >>> resolver = ConfigResolver(Path(".lobster_workspace"))
            >>> profile, source = resolver.resolve_profile()
            >>> logger.info(f"Using profile '{profile}' (from {source})")
        """
        # Layer 1: Runtime override
        if runtime_override:
            valid_profiles = ["development", "production", "ultra", "godmode", "hybrid"]
            if runtime_override in valid_profiles:
                return (runtime_override, "runtime flag --profile")
            else:
                logger.warning(
                    f"Invalid runtime profile '{runtime_override}', "
                    f"continuing to next priority level"
                )

        # Layer 2: Workspace config (only if file exists)
        if self.workspace_config and self.workspace_path:
            if WorkspaceProviderConfig.exists(self.workspace_path):
                profile = self.workspace_config.profile
                return (profile, "workspace config")

        # Layer 3: Global user config (only if file exists)
        if self.global_config:
            if GlobalProviderConfig.exists():
                profile = self.global_config.default_profile
                return (profile, "global user config")

        # Layer 4: Environment variable
        if env_profile := os.environ.get("LOBSTER_PROFILE"):
            valid_profiles = ["development", "production", "ultra", "godmode", "hybrid"]
            if env_profile in valid_profiles:
                return (env_profile, "environment variable LOBSTER_PROFILE")
            else:
                logger.warning(
                    f"Invalid LOBSTER_PROFILE '{env_profile}', using default"
                )

        # Layer 5: Default
        return ("production", "default (no configuration found)")

    def resolve_per_agent_provider(
        self,
        agent_name: str,
        global_provider: str,
    ) -> Tuple[str, str]:
        """
        Resolve provider for a specific agent (for mixed-provider workflows).

        Args:
            agent_name: Name of agent (e.g., "supervisor", "data_expert")
            global_provider: Global provider as fallback

        Returns:
            Tuple[str, str]: (provider_name, decision_source)

        Decision sources:
            - "workspace config (per-agent)"
            - "global provider"

        Example:
            >>> # Cost optimization: expensive agents on Ollama, cheap on Bedrock
            >>> resolver = ConfigResolver(Path(".lobster_workspace"))
            >>> provider, source = resolver.resolve_per_agent_provider(
            ...     "supervisor", "bedrock"
            ... )
            >>> # Returns ("ollama", "workspace config (per-agent)")
        """
        # Check workspace per-agent provider override
        if (
            self.workspace_config
            and agent_name in self.workspace_config.per_agent_providers
        ):
            provider = self.workspace_config.per_agent_providers[agent_name]
            return (provider, f"workspace config (agent '{agent_name}')")

        # Fallback to global provider
        return (global_provider, "global provider")

    def log_resolution_summary(
        self,
        provider: str,
        provider_source: str,
        profile: str,
        profile_source: str,
        agent_models: Optional[dict] = None,
    ) -> None:
        """
        Log a comprehensive summary of configuration resolution.

        This provides users with full transparency about why each decision was made.

        Args:
            provider: Resolved provider name
            provider_source: Source of provider decision
            profile: Resolved profile name
            profile_source: Source of profile decision
            agent_models: Optional dict of agent_name -> (model, source)

        Example:
            >>> resolver = ConfigResolver(Path(".lobster_workspace"))
            >>> provider, p_source = resolver.resolve_provider()
            >>> profile, pf_source = resolver.resolve_profile()
            >>> resolver.log_resolution_summary(provider, p_source, profile, pf_source)
        """
        logger.info("=" * 60)
        logger.info("Configuration Resolution Summary")
        logger.info("=" * 60)
        logger.info(f"Provider: {provider} (from {provider_source})")
        logger.info(f"Profile:  {profile} (from {profile_source})")

        if agent_models:
            logger.info("\nPer-Agent Model Resolution:")
            for agent_name, (model, source) in agent_models.items():
                logger.info(f"  {agent_name}: {model} (from {source})")

        logger.info("=" * 60)
