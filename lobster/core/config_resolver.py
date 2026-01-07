"""
Configuration resolver with transparent decision logging.

This module implements a 5-layer priority hierarchy for provider
and model selection, with explicit failure when not configured.

Priority Order (highest to lowest):
1. Runtime overrides (CLI flags like --provider, --model)
2. Workspace config (.lobster_workspace/provider_config.json)
3. Global user config (~/.config/lobster/providers.json)
4. Environment variable (LOBSTER_LLM_PROVIDER)
5. FAIL - require explicit configuration (no auto-detection, no defaults)

This design ensures:
- Users must explicitly configure their provider
- No silent defaults that may incur unexpected costs
- Clear error messages guide users to proper setup
- External workspaces can inherit from global config

Example:
    >>> from lobster.core.config_resolver import ConfigResolver
    >>> from pathlib import Path
    >>>
    >>> resolver = ConfigResolver.get_instance(Path(".lobster_workspace"))
    >>> try:
    ...     provider, source = resolver.resolve_provider()
    ...     print(f"Using {provider} (from {source})")
    ... except ConfigurationError as e:
    ...     print(f"Run 'lobster init' to configure: {e}")
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

# Environment variable for provider override
LOBSTER_LLM_PROVIDER_ENV = "LOBSTER_LLM_PROVIDER"

from lobster.config.constants import VALID_PROVIDERS, VALID_PROFILES
from lobster.config.global_config import GlobalProviderConfig
from lobster.config.workspace_config import WorkspaceProviderConfig

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """
    Raised when required configuration is missing.

    This error indicates the user needs to run 'lobster init' or
    manually create a provider_config.json file.
    """

    def __init__(self, message: str, help_text: Optional[str] = None):
        super().__init__(message)
        self.help_text = help_text or (
            "Run 'lobster init' to configure your provider, "
            "or create .lobster_workspace/provider_config.json manually."
        )


class ConfigResolver:
    """
    Resolve configuration with transparent decision logging.

    This class implements a 5-layer priority hierarchy:
    1. Runtime CLI flags (--provider, --model)
    2. Workspace JSON config (.lobster_workspace/provider_config.json)
    3. Global user config (~/.config/lobster/providers.json)
    4. Environment variable (LOBSTER_LLM_PROVIDER)
    5. FAIL with helpful error message

    Singleton Pattern:
        Use get_instance() to get the resolver with proper workspace context.

    Attributes:
        workspace_path: Path to workspace directory
        workspace_config: Loaded workspace configuration
        global_config: Loaded global user configuration

    Example:
        >>> resolver = ConfigResolver.get_instance(Path(".lobster_workspace"))
        >>> provider, source = resolver.resolve_provider()
    """

    _instance: Optional["ConfigResolver"] = None
    _instance_workspace: Optional[Path] = None

    def __init__(self, workspace_path: Optional[Path] = None):
        """
        Initialize resolver with optional workspace path.

        Args:
            workspace_path: Path to workspace directory (optional)

        Note:
            Prefer using get_instance() for singleton access.
        """
        self.workspace_path = workspace_path
        self.workspace_config: Optional[WorkspaceProviderConfig] = None
        self.global_config: Optional[GlobalProviderConfig] = None

        # Load configurations if workspace path provided
        if workspace_path:
            self.workspace_config = WorkspaceProviderConfig.load(workspace_path)
            # Global config still used for profile defaults
            self.global_config = GlobalProviderConfig.load()

    @classmethod
    def get_instance(cls, workspace_path: Optional[Path] = None, force_reload: bool = False) -> "ConfigResolver":
        """
        Get singleton instance of ConfigResolver.

        Args:
            workspace_path: Path to workspace directory
            force_reload: Force reload config even if workspace path unchanged

        Returns:
            ConfigResolver: Singleton instance

        Note:
            If workspace_path changes or force_reload=True, a new instance is created.
            This ensures config files are re-read after lobster init.

        Example:
            >>> resolver = ConfigResolver.get_instance(Path(".lobster_workspace"))
            >>> # After creating new config:
            >>> resolver = ConfigResolver.get_instance(Path(".lobster_workspace"), force_reload=True)
        """
        # Normalize workspace_path for comparison
        norm_workspace = workspace_path.resolve() if workspace_path else None
        norm_cached = cls._instance_workspace.resolve() if cls._instance_workspace else None

        if cls._instance is None or norm_workspace != norm_cached or force_reload:
            cls._instance = cls(workspace_path)
            cls._instance_workspace = workspace_path
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None
        cls._instance_workspace = None

    def resolve_provider(
        self,
        runtime_override: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Resolve LLM provider with 5-layer priority.

        Priority:
        1. Runtime CLI flag (--provider)
        2. Workspace config (provider_config.json)
        3. Global user config (~/.config/lobster/providers.json)
        4. Environment variable (LOBSTER_LLM_PROVIDER)
        5. FAIL - configuration required

        Args:
            runtime_override: Explicit provider from CLI flag (highest priority)

        Returns:
            Tuple[str, str]: (provider_name, decision_source)

        Raises:
            ConfigurationError: If no provider is configured or invalid

        Example:
            >>> resolver = ConfigResolver.get_instance(workspace)
            >>> provider, source = resolver.resolve_provider()
            >>> print(f"Using provider '{provider}' (from {source})")
        """
        # Layer 1: Runtime override (highest priority)
        if runtime_override:
            if runtime_override in VALID_PROVIDERS:
                return (runtime_override, "runtime flag --provider")
            else:
                raise ConfigurationError(
                    f"Invalid provider '{runtime_override}'. "
                    f"Valid providers: {', '.join(VALID_PROVIDERS)}"
                )

        # Layer 2: Workspace config
        if self.workspace_config and self.workspace_path:
            if WorkspaceProviderConfig.exists(self.workspace_path):
                if self.workspace_config.global_provider:
                    provider = self.workspace_config.global_provider
                    return (provider, "workspace config")

        # Layer 3: Global user config
        if self.global_config and GlobalProviderConfig.exists():
            if self.global_config.default_provider:
                provider = self.global_config.default_provider
                return (provider, "global user config (~/.config/lobster/)")

        # Layer 4: Environment variable
        env_provider = os.environ.get(LOBSTER_LLM_PROVIDER_ENV)
        if env_provider:
            if env_provider in VALID_PROVIDERS:
                return (env_provider, f"environment variable {LOBSTER_LLM_PROVIDER_ENV}")
            else:
                # Hard error for invalid env var (fail-fast)
                raise ConfigurationError(
                    f"Invalid provider '{env_provider}' in {LOBSTER_LLM_PROVIDER_ENV}. "
                    f"Valid providers: {', '.join(VALID_PROVIDERS)}"
                )

        # Layer 5: FAIL - require explicit configuration with diagnostic
        raise ConfigurationError(
            "No provider configured.",
            help_text=self._build_diagnostic_help_text(),
        )

    def _build_diagnostic_help_text(self) -> str:
        """Build diagnostic help text showing what was checked."""
        workspace_path_str = str(self.workspace_path) if self.workspace_path else "(not set)"
        workspace_config_exists = (
            WorkspaceProviderConfig.exists(self.workspace_path)
            if self.workspace_path else False
        )
        global_config_exists = GlobalProviderConfig.exists()
        env_var_set = os.environ.get(LOBSTER_LLM_PROVIDER_ENV)

        return (
            "Lobster requires explicit provider configuration.\n\n"
            "Checked (in priority order):\n"
            f"  ✗ Runtime flag: --provider not provided\n"
            f"  ✗ Workspace config: {workspace_path_str}/provider_config.json "
            f"{'(no provider set)' if workspace_config_exists else '(not found)'}\n"
            f"  ✗ Global config: ~/.config/lobster/providers.json "
            f"{'(no default_provider set)' if global_config_exists else '(not found)'}\n"
            f"  ✗ Environment: {LOBSTER_LLM_PROVIDER_ENV} "
            f"{'=' + env_var_set if env_var_set else '(not set)'}\n\n"
            "Quick Setup:\n"
            "  lobster init              # Configure this workspace\n"
            "  lobster init --global     # Set global defaults for all workspaces\n\n"
            "Or set environment variable:\n"
            f"  export {LOBSTER_LLM_PROVIDER_ENV}=anthropic\n\n"
            "Valid providers: " + ", ".join(VALID_PROVIDERS)
        )

    def resolve_model(
        self,
        agent_name: Optional[str] = None,
        runtime_override: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        """
        Resolve model for a specific agent with 4-layer priority.

        Priority:
        1. Runtime CLI flag (--model)
        2. Workspace config (per-agent or per-provider)
        3. Global user config (~/.config/lobster/providers.json)
        4. Provider default (from ProviderRegistry)

        Args:
            agent_name: Name of agent (e.g., "supervisor", "data_expert")
            runtime_override: Explicit model from CLI flag (highest priority)
            provider: LLM provider (bedrock, anthropic, ollama)

        Returns:
            Tuple[Optional[str], str]: (model_name, decision_source)
                                       Returns (None, source) if no model specified

        Example:
            >>> model, source = resolver.resolve_model("supervisor", provider="anthropic")
        """
        # Layer 1: Runtime override
        if runtime_override:
            return (runtime_override, "runtime flag --model")

        # Layer 2a: Workspace per-agent model override
        if self.workspace_config and self.workspace_path and agent_name:
            if WorkspaceProviderConfig.exists(self.workspace_path):
                if agent_name in self.workspace_config.per_agent_models:
                    model = self.workspace_config.per_agent_models[agent_name]
                    return (model, f"workspace config (agent '{agent_name}')")

        # Layer 2b: Workspace global model for the current provider
        if self.workspace_config and self.workspace_path and provider:
            if WorkspaceProviderConfig.exists(self.workspace_path):
                model = self.workspace_config.get_model_for_provider(provider)
                if model:
                    return (model, f"workspace config ({provider} model)")

        # Layer 3: Global user config model for the current provider
        if self.global_config and GlobalProviderConfig.exists() and provider:
            model = self.global_config.get_model_for_provider(provider)
            if model:
                return (model, f"global user config ({provider} model)")

        # Layer 4: Provider default (delegated to provider)
        if provider:
            try:
                from lobster.config.providers import get_provider

                prov = get_provider(provider)
                if prov:
                    default_model = prov.get_default_model()
                    if default_model:
                        return (default_model, f"provider default ({provider})")
            except Exception as e:
                logger.debug(f"Failed to get default model for {provider}: {e}")

        # No model specified - caller should use provider default
        return (None, "no model specified")

    def resolve_profile(
        self,
        runtime_override: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Resolve agent configuration profile with priority.

        Profile controls model selection per agent (development uses cheaper
        models, production uses best models, etc.).

        Priority:
        1. Runtime CLI flag (--profile)
        2. Workspace config
        3. Global user config
        4. Default: "production"

        Args:
            runtime_override: Explicit profile from CLI flag (highest priority)

        Returns:
            Tuple[str, str]: (profile_name, decision_source)

        Example:
            >>> profile, source = resolver.resolve_profile()
        """
        # Layer 1: Runtime override
        if runtime_override:
            if runtime_override in VALID_PROFILES:
                return (runtime_override, "runtime flag --profile")
            else:
                logger.warning(
                    f"Invalid runtime profile '{runtime_override}', "
                    f"valid profiles: {', '.join(VALID_PROFILES)}"
                )

        # Layer 2: Workspace config
        if self.workspace_config and self.workspace_path:
            if WorkspaceProviderConfig.exists(self.workspace_path):
                profile = self.workspace_config.profile
                if profile:
                    return (profile, "workspace config")

        # Layer 3: Global user config
        if self.global_config:
            if GlobalProviderConfig.exists():
                profile = self.global_config.default_profile
                if profile:
                    return (profile, "global user config")

        # Layer 4: Default (profiles have reasonable default)
        return ("production", "default")

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

        Example:
            >>> # Cost optimization: expensive agents on Ollama, cheap on Bedrock
            >>> provider, source = resolver.resolve_per_agent_provider("supervisor", "bedrock")
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

    def is_configured(self) -> bool:
        """
        Check if a valid provider configuration exists.

        Returns:
            bool: True if provider is configured

        Example:
            >>> if not resolver.is_configured():
            ...     print("Run 'lobster init' to configure")
        """
        try:
            self.resolve_provider()
            return True
        except ConfigurationError:
            return False

    def get_configuration_summary(self) -> dict:
        """
        Get a summary of current configuration.

        Returns:
            dict: Configuration summary for display

        Example:
            >>> summary = resolver.get_configuration_summary()
            >>> print(f"Provider: {summary['provider']} ({summary['provider_source']})")
        """
        result = {
            "configured": False,
            "provider": None,
            "provider_source": None,
            "profile": None,
            "profile_source": None,
            "workspace_path": str(self.workspace_path) if self.workspace_path else None,
        }

        try:
            provider, provider_source = self.resolve_provider()
            result["provider"] = provider
            result["provider_source"] = provider_source
            result["configured"] = True
        except ConfigurationError:
            pass

        try:
            profile, profile_source = self.resolve_profile()
            result["profile"] = profile
            result["profile_source"] = profile_source
        except Exception:
            result["profile"] = "production"
            result["profile_source"] = "default"

        return result

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

        Args:
            provider: Resolved provider name
            provider_source: Source of provider decision
            profile: Resolved profile name
            profile_source: Source of profile decision
            agent_models: Optional dict of agent_name -> (model, source)
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
