"""Data Expert agent module for multi-omics data acquisition and management."""

# Temporary: Import data_expert only when it exists (during refactoring)
try:
    from lobster.agents.data_expert.data_expert import data_expert
    __all__ = ["data_expert"]
except ImportError:
    # During refactoring, data_expert.py may not exist yet
    __all__ = []
