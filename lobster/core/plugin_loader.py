"""
Plugin discovery for premium and custom packages.

This module discovers and loads agent registries from:
- lobster_premium: Shared premium features for paying customers
- lobster_custom_*: Customer-specific agent packages

The plugin system follows Python's entry point pattern, allowing
external packages to register agents that get merged into the
main AGENT_REGISTRY at runtime.

Usage:
    from lobster.core.plugin_loader import discover_plugins, get_installed_packages

    # Discover all plugin agents
    plugin_agents = discover_plugins()

    # Check installed packages
    packages = get_installed_packages()
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

    Discovers packages by naming convention:
    - lobster_premium: shared premium features
    - lobster_custom_*: customer-specific features

    The function checks the entitlement to determine which custom
    packages are authorized for the current installation.

    Returns:
        Dict of agent configurations to merge into AGENT_REGISTRY.
        Keys are agent names, values are AgentRegistryConfig instances.
    """
    discovered_agents: Dict[str, Any] = {}

    # 1. Try to load lobster-premium package
    try:
        from lobster_premium import PREMIUM_REGISTRY

        discovered_agents.update(PREMIUM_REGISTRY)
        logger.info(
            f"Loaded {len(PREMIUM_REGISTRY)} premium agents from lobster-premium"
        )
    except ImportError:
        logger.debug("lobster-premium not installed (expected for free tier)")
    except Exception as e:
        logger.warning(f"Failed to load lobster-premium: {e}")

    # 2. Discover lobster-custom-* packages based on entitlement
    entitlement = _load_entitlement()
    custom_packages = entitlement.get("custom_packages", [])

    for pkg_name in custom_packages:
        try:
            # Convert package name to module name (dashes to underscores)
            module_name = pkg_name.replace("-", "_")
            module = importlib.import_module(module_name)

            if hasattr(module, "CUSTOM_REGISTRY"):
                registry = getattr(module, "CUSTOM_REGISTRY")
                discovered_agents.update(registry)
                logger.info(f"Loaded {len(registry)} custom agents from {pkg_name}")
            else:
                logger.debug(f"Package {pkg_name} has no CUSTOM_REGISTRY export")

        except ImportError as e:
            logger.warning(f"Custom package {pkg_name} not installed: {e}")
        except Exception as e:
            logger.warning(f"Failed to load custom package {pkg_name}: {e}")

    # 3. Also discover any packages via entry points (future-proof)
    discovered_agents.update(_discover_via_entry_points())

    return discovered_agents


def _discover_via_entry_points() -> Dict[str, Any]:
    """
    Discover plugins via Python entry points.

    This allows third-party packages to register agents by declaring
    entry points in their pyproject.toml:

        [project.entry-points."lobster.plugins"]
        my_agent = "my_package:AGENT_REGISTRY"

    Returns:
        Dict of agent configurations from entry point packages.
    """
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

        # Check for required exports
        if not hasattr(module, "CUSTOM_REGISTRY") and not hasattr(
            module, "PREMIUM_REGISTRY"
        ):
            result["warnings"].append(
                "Package has no CUSTOM_REGISTRY or PREMIUM_REGISTRY export"
            )

    except ImportError as e:
        result["compatible"] = False
        result["errors"].append(f"Package not installed: {e}")
    except Exception as e:
        result["warnings"].append(f"Compatibility check error: {e}")

    return result
