"""
Plugin discovery utilities for premium and custom packages.

This module provides diagnostic and compatibility utilities for plugin packages.
The actual plugin discovery is handled by ComponentRegistry (component_registry.py).

Plugin packages (lobster-premium, lobster-custom-*) register components via entry points:
- [project.entry-points."lobster.services"] - Premium service classes
- [project.entry-points."lobster.agents"] - Custom agent configurations

Usage:
    from lobster.core.plugin_loader import discover_plugins, get_installed_packages

    # Discover all agents (delegates to ComponentRegistry)
    plugin_agents = discover_plugins()

    # Check installed packages
    packages = get_installed_packages()

Note: For direct component access, use ComponentRegistry:
    from lobster.core.component_registry import component_registry
    component_registry.get_service('publication_processing')
    component_registry.list_agents()
"""

import importlib
import importlib.metadata
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_entitlement() -> Dict[str, Any]:
    """
    Load entitlement data from license manager.

    Returns:
        Entitlement dict with tier, custom_packages, etc.
        Returns minimal free tier entitlement if no license found.
    """
    try:
        from lobster.core.license_manager import load_entitlement

        return load_entitlement()
    except ImportError:
        logger.debug("License manager not available, using free tier defaults")
        return {
            "tier": "free",
            "custom_packages": [],
            "valid": True,
        }
    except Exception as e:
        logger.warning(f"Failed to load entitlement: {e}, using free tier defaults")
        return {
            "tier": "free",
            "custom_packages": [],
            "valid": True,
        }


def discover_plugins() -> Dict[str, Any]:
    """
    Discover all lobster plugin packages and load their registries.

    Now uses unified ComponentRegistry for both services and agents.

    Returns:
        Dict of agent configurations to merge into AGENT_REGISTRY.
        Keys are agent names, values are AgentRegistryConfig instances.
    """
    from lobster.core.component_registry import component_registry

    # Single call loads BOTH services and agents
    component_registry.load_components()

    # Return all agents (core + custom) for backward compatibility with graph.py
    return component_registry.list_agents()


def _discover_via_entry_points() -> Dict[str, Any]:
    """
    DEPRECATED: This function is no longer used.

    Use ComponentRegistry instead, which discovers both services and agents
    via the new entry point groups:
        - [project.entry-points."lobster.services"]
        - [project.entry-points."lobster.agents"]

    This old function discovered agents via the deprecated "lobster.plugins"
    entry point group. It's kept for backward compatibility but is not
    called by the new system.

    Returns:
        Dict of agent configurations from entry point packages (empty).
    """
    # DEPRECATED: No longer used, component_registry handles discovery
    discovered: Dict[str, Any] = {}

    try:
        # Python 3.10+ style entry points discovery
        entry_points = importlib.metadata.entry_points()

        # Handle both old and new entry_points API
        if hasattr(entry_points, "select"):
            # Python 3.10+
            lobster_eps = entry_points.select(group="lobster.plugins")
        else:
            # Python 3.9 compatibility
            lobster_eps = entry_points.get("lobster.plugins", [])

        for ep in lobster_eps:
            try:
                registry = ep.load()
                if isinstance(registry, dict):
                    discovered.update(registry)
                    logger.info(f"Loaded agents from entry point: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load entry point {ep.name}: {e}")

    except Exception as e:
        logger.debug(f"Entry point discovery not available: {e}")

    return discovered


def get_installed_packages() -> Dict[str, str]:
    """
    Return dict of installed lobster packages and their versions.

    Returns:
        Dict mapping package name to version string.
        Version is "installed" if version cannot be determined,
        "missing" if package is in entitlement but not installed.
    """
    packages: Dict[str, str] = {}

    # Core package is always present
    try:
        packages["lobster-ai"] = importlib.metadata.version("lobster-ai")
    except importlib.metadata.PackageNotFoundError:
        # Development install or editable mode
        packages["lobster-ai"] = "dev"

    # Check for premium package
    try:
        import lobster_premium

        version = getattr(lobster_premium, "__version__", None)
        if version is None:
            try:
                version = importlib.metadata.version("lobster-premium")
            except importlib.metadata.PackageNotFoundError:
                version = "installed"
        packages["lobster-premium"] = version
    except ImportError:
        pass  # Not installed, don't add to dict

    # Check for custom packages based on entitlement
    entitlement = _load_entitlement()
    for pkg_name in entitlement.get("custom_packages", []):
        try:
            version = importlib.metadata.version(pkg_name)
            packages[pkg_name] = version
        except importlib.metadata.PackageNotFoundError:
            # Package is in entitlement but not installed
            packages[pkg_name] = "missing"

    return packages


def get_plugin_info() -> Dict[str, Any]:
    """
    Get comprehensive plugin information for diagnostics.

    Returns:
        Dict with installed packages, discovered agents, and entitlement status.
    """
    entitlement = _load_entitlement()
    packages = get_installed_packages()
    discovered = discover_plugins()

    return {
        "entitlement": {
            "tier": entitlement.get("tier", "free"),
            "valid": entitlement.get("valid", False),
            "expires": entitlement.get("expires"),
            "custom_packages": entitlement.get("custom_packages", []),
        },
        "installed_packages": packages,
        "discovered_agents": list(discovered.keys()),
        "agent_count": {
            "core": 0,  # Will be set by caller with registry info
            "premium": len([a for a in discovered.keys() if a not in packages]),
            "custom": len([a for a in discovered.keys()]),
        },
    }


def is_premium_installed() -> bool:
    """Check if lobster-premium package is installed."""
    try:
        import lobster_premium

        return True
    except ImportError:
        return False


def get_custom_package_names() -> List[str]:
    """
    Get list of custom package names from entitlement.

    Returns:
        List of package names (e.g., ["lobster-custom-databiomix"])
    """
    entitlement = _load_entitlement()
    return entitlement.get("custom_packages", [])


def validate_plugin_compatibility(package_name: str) -> Dict[str, Any]:
    """
    Validate that a plugin package is compatible with current lobster version.

    Args:
        package_name: Name of the plugin package to validate

    Returns:
        Dict with compatibility status and any warnings
    """
    result = {
        "package": package_name,
        "compatible": True,
        "warnings": [],
        "errors": [],
    }

    try:
        module_name = package_name.replace("-", "_")
        module = importlib.import_module(module_name)

        # Check for minimum lobster version requirement
        min_version = getattr(module, "MIN_LOBSTER_VERSION", None)
        if min_version:
            try:
                current_version = importlib.metadata.version("lobster-ai")
                # Simple version comparison (could be enhanced with packaging.version)
                if current_version < min_version:
                    result["compatible"] = False
                    result["errors"].append(
                        f"Requires lobster-ai >= {min_version}, found {current_version}"
                    )
            except Exception:
                result["warnings"].append("Could not verify version compatibility")

        # Check for entry points (new pattern)
        try:
            eps = importlib.metadata.entry_points()
            if hasattr(eps, "select"):
                # Python 3.10+
                services = list(eps.select(group="lobster.services"))
                agents = list(eps.select(group="lobster.agents"))
            else:
                # Python 3.9
                services = list(eps.get("lobster.services", []))
                agents = list(eps.get("lobster.agents", []))

            if not services and not agents:
                # Check for old CUSTOM_REGISTRY pattern (deprecated)
                if hasattr(module, "CUSTOM_REGISTRY") or hasattr(module, "PREMIUM_REGISTRY"):
                    result["warnings"].append(
                        "Package uses deprecated CUSTOM_REGISTRY pattern. "
                        "Please migrate to entry points (lobster.services, lobster.agents)"
                    )
                else:
                    result["warnings"].append(
                        "Package has no 'lobster.services' or 'lobster.agents' entry points"
                    )
        except Exception:
            result["warnings"].append("Could not verify entry point registration")

    except ImportError as e:
        result["compatible"] = False
        result["errors"].append(f"Package not installed: {e}")
    except Exception as e:
        result["warnings"].append(f"Compatibility check error: {e}")

    return result
