"""
Shared configuration constants for LLM provider system.

This module is the SINGLE SOURCE OF TRUTH for valid providers and profiles.
All other modules should import from here rather than defining their own lists.
"""

from typing import Final, List

# Valid LLM providers - add new providers here ONLY
VALID_PROVIDERS: Final[List[str]] = ["anthropic", "bedrock", "ollama", "gemini"]

# Valid agent configuration profiles
VALID_PROFILES: Final[List[str]] = [
    "development",
    "production",
    "ultra",
    "godmode",
    "hybrid",
]

# Provider display names for UI
PROVIDER_DISPLAY_NAMES: Final[dict] = {
    "anthropic": "Anthropic Direct API",
    "bedrock": "AWS Bedrock",
    "ollama": "Ollama (Local)",
    "gemini": "Google Gemini",
}
