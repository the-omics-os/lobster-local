"""
Shared command implementations for CLI and Dashboard.

Performance optimization: Commands are split into light/ (fast) and heavy/ (data-intensive).
Light commands import eagerly, heavy commands use lazy loading via __getattr__.

IMPORTANT: output_adapter.py is NOT imported eagerly to avoid circular imports.
It's re-exported via __getattr__ for backward compatibility.
"""

# ============================================================================
# EAGER IMPORTS (Light commands - no heavy dependencies)
# ============================================================================

# NOTE: output_adapter is NOT imported here to avoid triggering __init__.py
# when light commands import it. It's available via __getattr__ below.

# Light commands (from light/ subdirectory)
from lobster.cli_internal.commands.light.queue_commands import (
    show_queue_status,
    queue_load_file,
    queue_list,
    queue_clear,
    queue_export,
    queue_import,
    QueueFileTypeNotSupported,
)
from lobster.cli_internal.commands.light.metadata_commands import (
    metadata_overview,
    metadata_publications,
    metadata_samples,
    metadata_workspace,
    metadata_exports,
    metadata_list,
    metadata_clear,
    metadata_clear_exports,
    metadata_clear_all,
)
from lobster.cli_internal.commands.light.workspace_commands import (
    workspace_list,
    workspace_info,
    workspace_load,
    workspace_remove,
    workspace_status,
)
from lobster.cli_internal.commands.light.pipeline_commands import (
    pipeline_export,
    pipeline_list,
    pipeline_run,
    pipeline_info,
)
from lobster.cli_internal.commands.light.file_commands import (
    file_read,
    archive_queue,
)
from lobster.cli_internal.commands.light.config_commands import (
    config_show,
    config_provider_list,
    config_provider_switch,
    config_model_list,
    config_model_switch,
)
from lobster.cli_internal.commands.light.purge_commands import (
    purge,
    discover_purge_targets,
    PurgeScope,
)

# ============================================================================
# LAZY IMPORTS (Heavy commands AND output_adapter - via __getattr__)
# ============================================================================

# Map of lazy-loaded items to their import paths
_LAZY_IMPORTS = {
    # Output adapters (lazy to avoid circular import)
    "OutputAdapter": ("lobster.cli_internal.commands.output_adapter", "OutputAdapter"),
    "ConsoleOutputAdapter": ("lobster.cli_internal.commands.output_adapter", "ConsoleOutputAdapter"),
    "DashboardOutputAdapter": ("lobster.cli_internal.commands.output_adapter", "DashboardOutputAdapter"),
    # Heavy data commands
    "data_summary": ("lobster.cli_internal.commands.heavy.data_commands", "data_summary"),
    # Heavy modality commands
    "modalities_list": ("lobster.cli_internal.commands.heavy.modality_commands", "modalities_list"),
    "modality_describe": ("lobster.cli_internal.commands.heavy.modality_commands", "modality_describe"),
    # Heavy visualization commands
    "export_data": ("lobster.cli_internal.commands.heavy.visualization_commands", "export_data"),
    "plots_list": ("lobster.cli_internal.commands.heavy.visualization_commands", "plots_list"),
    "plot_show": ("lobster.cli_internal.commands.heavy.visualization_commands", "plot_show"),
}


def __getattr__(name: str):
    """
    Lazy loader for heavy commands and output adapters.

    Output adapters are lazy-loaded to avoid circular imports (light commands import them).
    Heavy commands are lazy-loaded to avoid ~2s numpy/pandas/anndata import penalty.

    Args:
        name: Attribute name being accessed

    Returns:
        The requested command function or class

    Raises:
        AttributeError: If the item doesn't exist
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        # Import the module
        from importlib import import_module
        module = import_module(module_path)
        # Get the specific function/class
        obj = getattr(module, attr_name)
        # Cache it in this module for future access
        globals()[name] = obj
        return obj

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Output adapters (lazy)
    "OutputAdapter",
    "ConsoleOutputAdapter",
    "DashboardOutputAdapter",
    # Queue commands (light)
    "show_queue_status",
    "queue_load_file",
    "queue_list",
    "queue_clear",
    "queue_export",
    "queue_import",
    "QueueFileTypeNotSupported",
    # Metadata commands (light)
    "metadata_overview",
    "metadata_publications",
    "metadata_samples",
    "metadata_workspace",
    "metadata_exports",
    "metadata_list",
    "metadata_clear",
    "metadata_clear_exports",
    "metadata_clear_all",
    # Workspace commands (light)
    "workspace_list",
    "workspace_info",
    "workspace_load",
    "workspace_remove",
    "workspace_status",
    # Pipeline commands (light)
    "pipeline_export",
    "pipeline_list",
    "pipeline_run",
    "pipeline_info",
    # Data commands (heavy - lazy)
    "data_summary",
    # File commands (light)
    "file_read",
    "archive_queue",
    # Config commands (light)
    "config_show",
    "config_provider_list",
    "config_provider_switch",
    "config_model_list",
    "config_model_switch",
    # Purge commands (light)
    "purge",
    "discover_purge_targets",
    "PurgeScope",
    # Modality commands (heavy - lazy)
    "modalities_list",
    "modality_describe",
    # Visualization commands (heavy - lazy)
    "export_data",
    "plots_list",
    "plot_show",
]
