"""
Shared tools for Lobster AI agents.

This module provides factory functions for creating shared tools that can be used
across multiple agents. Tools are created via factory pattern to inject dependencies
via closures, following the established workspace_tool.py pattern.

Available submodules:
    - custom_code_tool: Factory for execute_custom_code tool
    - workspace_tool: Factory for workspace access tools
    - providers/: External data providers (PubMed, GEO, SRA, etc.)

Usage:
    # Direct imports (recommended - avoids circular imports)
    from lobster.tools.custom_code_tool import create_execute_custom_code_tool
    from lobster.tools.workspace_tool import create_workspace_tools

Note:
    This __init__.py intentionally does NOT import submodules at module level
    to prevent circular imports. The lobster.tools namespace contains many
    submodules that may be imported independently (e.g., providers), and
    top-level imports would trigger loading of all dependencies.
"""

# Intentionally empty - consumers import directly from submodules
# This prevents circular imports when lobster.tools.providers.* is imported

__all__ = [
    # Document available exports without importing them
    # Consumers should import directly:
    #   from lobster.tools.custom_code_tool import create_execute_custom_code_tool
    #   from lobster.tools.custom_code_tool import metadata_store_post_processor
    #   from lobster.tools.custom_code_tool import PostProcessor
]
