"""
Subscription tier definitions for premium feature gating.

This module defines the three-tier subscription model:
- FREE: Open-core agents available to all users
- PREMIUM: Additional agents for paying customers
- ENTERPRISE: All agents including customer-specific packages

Tier Configuration:
- agents: List of agent names available at this tier
- restricted_handoffs: Dict mapping agent_name -> list of blocked handoff targets
- features: List of feature flags enabled at this tier
- compute_limits: Resource constraints for this tier
"""

from typing import Any, Dict, List, Optional

# =============================================================================
# SUBSCRIPTION TIER DEFINITIONS (Source of Truth)
# =============================================================================

SUBSCRIPTION_TIERS: Dict[str, Dict[str, Any]] = {
    "free": {
        "display_name": "Free",
        "agents": [
            # Core agents available to all users (6 agents)
            # These are synced to lobster-local (public repo)
            # Note: Names must match AGENT_REGISTRY keys exactly
            "research_agent",
            "data_expert_agent",
            "transcriptomics_expert",  # Unified single-cell + bulk RNA-seq
            "visualization_expert_agent",
            "annotation_expert",
            "de_analysis_expert",
        ],
        "restricted_handoffs": {
            # FREE tier: research_agent cannot handoff to metadata_assistant
            # This is a key premium differentiator
            "research_agent": ["metadata_assistant"],
        },
        "features": [
            "local_only",
            "community_support",
        ],
        "compute_limits": {
            "queries_per_day": 50,
            "max_datasets": 5,
        },
    },
    "premium": {
        "display_name": "Premium",
        "agents": [
            # All FREE tier agents (6)
            "research_agent",
            "data_expert_agent",
            "transcriptomics_expert",
            "visualization_expert_agent",
            "annotation_expert",
            "de_analysis_expert",
            # Premium-only agents (4) - NOT synced to lobster-local
            "metadata_assistant",  # Publication queue filtering, ID mapping
            "proteomics_expert",  # MS proteomics (DDA/DIA)
            "machine_learning_expert_agent",
            "protein_structure_visualization_expert_agent",
        ],
        "restricted_handoffs": {},  # No restrictions at premium tier
        "features": [
            "local_only",
            "cloud_compute",
            "email_support",
            "priority_processing",
        ],
        "compute_limits": {
            "queries_per_day": 500,
            "max_datasets": 50,
        },
    },
    "enterprise": {
        "display_name": "Enterprise",
        "agents": ["*"],  # Wildcard: all agents including custom packages
        "custom_packages": True,  # Allow lobster-custom-{customer} packages
        "restricted_handoffs": {},  # No restrictions
        "features": [
            "local_only",
            "cloud_compute",
            "dedicated_compute",
            "sla",
            "custom_development",
            "priority_support",
        ],
        "compute_limits": {
            "queries_per_day": None,  # Unlimited
            "max_datasets": None,  # Unlimited
        },
    },
}

# =============================================================================
# TIER HELPER FUNCTIONS
# =============================================================================


def get_tier_config(tier: str) -> Dict[str, Any]:
    """
    Get the full configuration for a subscription tier.

    Args:
        tier: Tier name (free, premium, enterprise)

    Returns:
        Tier configuration dict, defaults to 'free' if tier not found
    """
    return SUBSCRIPTION_TIERS.get(tier.lower(), SUBSCRIPTION_TIERS["free"])


def get_tier_agents(tier: str) -> List[str]:
    """
    Get list of agents available for a subscription tier.

    Args:
        tier: Tier name (free, premium, enterprise)

    Returns:
        List of agent names available at this tier
    """
    tier_config = get_tier_config(tier)
    return tier_config.get("agents", [])


def get_restricted_handoffs(tier: str, agent_name: str) -> List[str]:
    """
    Get list of handoffs restricted for an agent in a given tier.

    Args:
        tier: Tier name (free, premium, enterprise)
        agent_name: Name of the agent to check restrictions for

    Returns:
        List of agent names that cannot be handed off to
    """
    tier_config = get_tier_config(tier)
    restrictions = tier_config.get("restricted_handoffs", {})
    return restrictions.get(agent_name, [])


def is_agent_available(agent_name: str, tier: str) -> bool:
    """
    Check if an agent is available for a subscription tier.

    Args:
        agent_name: Name of the agent to check
        tier: Tier name (free, premium, enterprise)

    Returns:
        True if agent is available at this tier
    """
    tier_config = get_tier_config(tier)
    agents = tier_config.get("agents", [])
    # Wildcard "*" means all agents are available (enterprise tier)
    return "*" in agents or agent_name in agents


def is_handoff_allowed(source_agent: str, target_agent: str, tier: str) -> bool:
    """
    Check if a handoff from source to target agent is allowed at given tier.

    Args:
        source_agent: Agent initiating the handoff
        target_agent: Agent being handed off to
        tier: Subscription tier

    Returns:
        True if handoff is allowed
    """
    restricted = get_restricted_handoffs(tier, source_agent)
    return target_agent not in restricted


def get_tier_features(tier: str) -> List[str]:
    """
    Get list of features enabled for a subscription tier.

    Args:
        tier: Tier name

    Returns:
        List of feature flags
    """
    tier_config = get_tier_config(tier)
    return tier_config.get("features", [])


def get_compute_limits(tier: str) -> Dict[str, Optional[int]]:
    """
    Get compute limits for a subscription tier.

    Args:
        tier: Tier name

    Returns:
        Dict of limit_name -> limit_value (None means unlimited)
    """
    tier_config = get_tier_config(tier)
    return tier_config.get("compute_limits", {})


def get_all_tiers() -> List[str]:
    """Get list of all available tier names."""
    return list(SUBSCRIPTION_TIERS.keys())


def get_tier_display_name(tier: str) -> str:
    """Get human-readable display name for a tier."""
    tier_config = get_tier_config(tier)
    return tier_config.get("display_name", tier.title())


# =============================================================================
# TIER COMPARISON UTILITIES
# =============================================================================

# Tier hierarchy for comparison (higher index = higher tier)
_TIER_HIERARCHY = ["free", "premium", "enterprise"]


def compare_tiers(tier1: str, tier2: str) -> int:
    """
    Compare two tiers.

    Args:
        tier1: First tier name
        tier2: Second tier name

    Returns:
        -1 if tier1 < tier2, 0 if equal, 1 if tier1 > tier2
    """
    idx1 = (
        _TIER_HIERARCHY.index(tier1.lower()) if tier1.lower() in _TIER_HIERARCHY else 0
    )
    idx2 = (
        _TIER_HIERARCHY.index(tier2.lower()) if tier2.lower() in _TIER_HIERARCHY else 0
    )

    if idx1 < idx2:
        return -1
    elif idx1 > idx2:
        return 1
    return 0


def is_tier_at_least(current_tier: str, required_tier: str) -> bool:
    """
    Check if current tier meets or exceeds required tier.

    Args:
        current_tier: User's current tier
        required_tier: Minimum required tier

    Returns:
        True if current_tier >= required_tier
    """
    return compare_tiers(current_tier, required_tier) >= 0
