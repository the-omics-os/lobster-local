"""
Unified component registry for premium services and agents via entry points.

This module discovers and loads components from:
- lobster-premium: Shared premium features
- lobster-custom-*: Customer-specific features

Components are advertised via entry points in pyproject.toml:

    [project.entry-points."lobster.services"]
    publication_processing = "package.module:ServiceClass"

    [project.entry-points."lobster.agents"]
    metadata_assistant = "package.module:AGENT_CONFIG"

    [project.entry-points."lobster.agent_configs"]
    metadata_assistant = "package.module:CUSTOM_AGENT_CONFIG"

Usage:
    from lobster.core.component_registry import component_registry

    # Services
    ServiceClass = component_registry.get_service('publication_processing')

    # Agents
    agent_config = component_registry.get_agent('metadata_assistant')
    all_agents = component_registry.list_agents()  # Includes core + custom

    # Agent LLM Configs
    llm_config = component_registry.get_agent_config('metadata_assistant')
    all_configs = component_registry.list_agent_configs()
"""

import sys
import logging
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


class ComponentConflictError(Exception):
    """Raised when a custom component conflicts with a core component."""
    pass


class ComponentRegistry:
    """
    Unified registry for dynamically loaded services and agents.

    Services: Premium service classes loaded via entry points
    Agents: AgentRegistryConfig instances (core hardcoded + custom via entry points)
    """

    def __init__(self):
        self._services: Dict[str, Type[Any]] = {}
        self._custom_agents: Dict[str, Any] = {}  # AgentRegistryConfig instances
        self._custom_agent_configs: Dict[str, Any] = {}  # CustomAgentConfig instances
        self._loaded = False

    def load_components(self) -> None:
        """
        Discover and load all components from entry points.
        Idempotent - safe to call multiple times.
        """
        if self._loaded:
            return

        logger.debug("Discovering components via entry points...")

        # Load services from 'lobster.services' entry point
        self._load_entry_point_group('lobster.services', self._services)

        # Load custom agents from 'lobster.agents' entry point
        self._load_entry_point_group('lobster.agents', self._custom_agents)

        # Load custom agent LLM configs from 'lobster.agent_configs' entry point
        self._load_entry_point_group('lobster.agent_configs', self._custom_agent_configs)

        self._loaded = True
        logger.debug(
            f"Component discovery complete. "
            f"Services: {len(self._services)}, "
            f"Custom agents: {len(self._custom_agents)}, "
            f"Custom agent configs: {len(self._custom_agent_configs)}"
        )

    def _load_entry_point_group(self, group: str, target_dict: Dict[str, Any]) -> None:
        """Load all entry points from a specific group into target dict."""
        # Handle Python 3.10+ vs 3.9 API differences
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points
            discovered = entry_points(group=group)
        else:
            from importlib.metadata import entry_points
            eps = entry_points()
            discovered = eps.get(group, [])

        for entry in discovered:
            try:
                loaded = entry.load()
                target_dict[entry.name] = loaded
                logger.info(f"Loaded {group.split('.')[-1]} '{entry.name}' from {entry.value}")
            except Exception as e:
                logger.warning(f"Failed to load {group} '{entry.name}': {e}")

    # =========================================================================
    # SERVICE API
    # =========================================================================

    def get_service(self, name: str, required: bool = False) -> Optional[Type[Any]]:
        """
        Get a premium service class by name.

        Args:
            name: Service name (e.g., 'publication_processing')
            required: If True, raise error when service not found

        Returns:
            Service class if found, None otherwise

        Raises:
            ValueError: If required=True and service not found
        """
        if not self._loaded:
            self.load_components()

        service = self._services.get(name)

        if service is None and required:
            raise ValueError(
                f"Required service '{name}' not found. "
                f"Available services: {list(self._services.keys())}"
            )

        return service

    def has_service(self, name: str) -> bool:
        """Check if a service is available."""
        if not self._loaded:
            self.load_components()
        return name in self._services

    def list_services(self) -> Dict[str, str]:
        """List all available services with their module paths."""
        if not self._loaded:
            self.load_components()
        return {
            name: f"{cls.__module__}.{cls.__name__}"
            for name, cls in self._services.items()
        }

    # =========================================================================
    # AGENT API
    # =========================================================================

    def get_agent(self, name: str, required: bool = False) -> Optional[Any]:
        """
        Get an agent config by name (custom agents only).

        For unified agent access including core agents, use list_agents().

        Args:
            name: Agent name (e.g., 'metadata_assistant')
            required: If True, raise error when agent not found

        Returns:
            AgentRegistryConfig if found, None otherwise

        Raises:
            ValueError: If required=True and agent not found
        """
        if not self._loaded:
            self.load_components()

        agent = self._custom_agents.get(name)

        if agent is None and required:
            raise ValueError(
                f"Required agent '{name}' not found in custom agents. "
                f"Available custom agents: {list(self._custom_agents.keys())}"
            )

        return agent

    def has_agent(self, name: str) -> bool:
        """Check if a custom agent is available."""
        if not self._loaded:
            self.load_components()
        return name in self._custom_agents

    def list_custom_agents(self) -> Dict[str, Any]:
        """List custom agents only (from entry points)."""
        if not self._loaded:
            self.load_components()
        return dict(self._custom_agents)

    def list_agents(self) -> Dict[str, Any]:
        """
        List ALL agents (core + custom).

        This merges the hardcoded AGENT_REGISTRY with custom agents from entry points.
        Raises ComponentConflictError if custom agent names collide with core agents.

        Returns:
            Dict[str, AgentRegistryConfig] - All available agents

        Raises:
            ComponentConflictError: If a custom agent has the same name as a core agent
        """
        if not self._loaded:
            self.load_components()

        # Import core registry (hardcoded agents)
        from lobster.config.agent_registry import AGENT_REGISTRY

        # Detect name collisions before merging (strict validation)
        conflicts = set(AGENT_REGISTRY.keys()) & set(self._custom_agents.keys())
        if conflicts:
            raise ComponentConflictError(
                f"Custom agent name collision detected: {sorted(conflicts)}. "
                f"Custom agents must use unique names that don't conflict with core agents. "
                f"Core agent names: {sorted(AGENT_REGISTRY.keys())}"
            )

        # Merge: core agents + custom agents (no conflicts possible after check)
        all_agents = dict(AGENT_REGISTRY)
        all_agents.update(self._custom_agents)

        return all_agents

    # =========================================================================
    # AGENT CONFIG API
    # =========================================================================

    def get_agent_config(self, name: str, required: bool = False) -> Optional[Any]:
        """
        Get a custom agent LLM config by name.

        Args:
            name: Agent name (e.g., 'metadata_assistant')
            required: If True, raise error when config not found

        Returns:
            CustomAgentConfig if found, None otherwise

        Raises:
            ValueError: If required=True and config not found
        """
        if not self._loaded:
            self.load_components()

        config = self._custom_agent_configs.get(name)

        if config is None and required:
            raise ValueError(
                f"Required agent config '{name}' not found. "
                f"Available configs: {list(self._custom_agent_configs.keys())}"
            )

        return config

    def has_agent_config(self, name: str) -> bool:
        """Check if a custom agent config is available."""
        if not self._loaded:
            self.load_components()
        return name in self._custom_agent_configs

    def list_agent_configs(self) -> Dict[str, Any]:
        """
        List all custom agent LLM configs (from entry points).

        Returns:
            Dict[str, CustomAgentConfig] - All available agent configs
        """
        if not self._loaded:
            self.load_components()
        return dict(self._custom_agent_configs)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive registry info for diagnostics."""
        if not self._loaded:
            self.load_components()

        return {
            "services": {
                "count": len(self._services),
                "names": list(self._services.keys()),
            },
            "custom_agents": {
                "count": len(self._custom_agents),
                "names": list(self._custom_agents.keys()),
            },
            "custom_agent_configs": {
                "count": len(self._custom_agent_configs),
                "names": list(self._custom_agent_configs.keys()),
            },
            "total_agents": len(self.list_agents()),
        }

    def reset(self) -> None:
        """Reset the registry state (for testing)."""
        self._services.clear()
        self._custom_agents.clear()
        self._custom_agent_configs.clear()
        self._loaded = False


# Singleton instance
component_registry = ComponentRegistry()
