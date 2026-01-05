"""Shared command implementations for CLI and Dashboard."""

from lobster.cli_internal.commands.output_adapter import (
    OutputAdapter,
    ConsoleOutputAdapter,
    DashboardOutputAdapter,
)
from lobster.cli_internal.commands.queue_commands import (
    show_queue_status,
    queue_load_file,
    queue_list,
    queue_clear,
    queue_export,
    QueueFileTypeNotSupported,
)
from lobster.cli_internal.commands.metadata_commands import (
    metadata_list,
    metadata_clear,
)
from lobster.cli_internal.commands.workspace_commands import (
    workspace_list,
    workspace_info,
    workspace_load,
    workspace_remove,
    workspace_status,
)
from lobster.cli_internal.commands.pipeline_commands import (
    pipeline_export,
    pipeline_list,
    pipeline_run,
    pipeline_info,
)
from lobster.cli_internal.commands.data_commands import (
    data_summary,
)
from lobster.cli_internal.commands.file_commands import (
    file_read,
    archive_queue,
)
from lobster.cli_internal.commands.config_commands import (
    config_show,
    config_provider_list,
    config_provider_switch,
    config_model_list,
    config_model_switch,
)
from lobster.cli_internal.commands.modality_commands import (
    modalities_list,
    modality_describe,
)
from lobster.cli_internal.commands.visualization_commands import (
    export_data,
    plots_list,
    plot_show,
)

__all__ = [
    # Output adapters
    "OutputAdapter",
    "ConsoleOutputAdapter",
    "DashboardOutputAdapter",
    # Queue commands
    "show_queue_status",
    "queue_load_file",
    "queue_list",
    "queue_clear",
    "queue_export",
    "QueueFileTypeNotSupported",
    # Metadata commands
    "metadata_list",
    "metadata_clear",
    # Workspace commands
    "workspace_list",
    "workspace_info",
    "workspace_load",
    "workspace_remove",
    "workspace_status",
    # Pipeline commands
    "pipeline_export",
    "pipeline_list",
    "pipeline_run",
    "pipeline_info",
    # Data commands
    "data_summary",
    # File commands
    "file_read",
    "archive_queue",
    # Config commands
    "config_show",
    "config_provider_list",
    "config_provider_switch",
    "config_model_list",
    "config_model_switch",
    # Modality commands
    "modalities_list",
    "modality_describe",
    # Visualization commands
    "export_data",
    "plots_list",
    "plot_show",
]
