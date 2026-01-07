"""
DataHub widget: Unified data management interface for the Lobster dashboard.

Provides dual-pane view:
- Workspace: Files on disk in .lobster_workspace (not yet loaded)
- Loaded: Modalities in memory (AnnData objects ready for analysis)

Supports drag-and-drop file ingestion via textual-filedrop pattern.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from textual import events
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, ListItem, ListView, Static

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Supported file formats for bioinformatics data
SUPPORTED_FORMATS = {
    ".h5ad": {"icon": "🧬", "type": "anndata", "adapter": "h5ad"},
    ".csv": {"icon": "📊", "type": "tabular", "adapter": "csv"},
    ".tsv": {"icon": "📊", "type": "tabular", "adapter": "tsv"},
    ".xlsx": {"icon": "📗", "type": "excel", "adapter": "excel"},
    ".xls": {"icon": "📗", "type": "excel", "adapter": "excel"},
    ".txt": {"icon": "📄", "type": "text", "adapter": "text"},
    ".mtx": {"icon": "🔢", "type": "10x_mtx", "adapter": "10x"},
    ".gz": {"icon": "📦", "type": "compressed", "adapter": "auto"},
    ".soft.gz": {"icon": "📦", "type": "geo_soft", "adapter": "geo_soft"},
    ".h5ad.gz": {"icon": "🧬", "type": "anndata", "adapter": "h5ad"},
}

# Icons for visual state indication
ICONS = {
    "workspace": "📁",  # File on disk, not loaded
    "loaded": "🔬",  # Modality in memory
    "loading": "⏳",  # Currently loading
    "error": "❌",  # Load failed
    "unsupported": "❓",  # Unsupported format
    "folder": "",  # Folder (10x MTX)
}


class FileState(Enum):
    """State of a file in the data pipeline."""

    WORKSPACE = "workspace"  # On disk, not loaded
    LOADING = "loading"  # Currently loading
    LOADED = "loaded"  # In memory as modality
    ERROR = "error"  # Load failed
    UNSUPPORTED = "unsupported"  # Format not supported


# ─────────────────────────────────────────────────────────────────────────────
# Messages
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FileDropped(Message):
    """Posted when files are dropped into the DataHub."""

    filepaths: List[str]
    filenames: List[str]


@dataclass
class FileLoadRequested(Message):
    """Posted when user requests to load a workspace file."""

    filepath: str
    filename: str


@dataclass
class ModalitySelected(Message):
    """Posted when user selects a loaded modality."""

    modality_name: str


@dataclass
class WorkspaceFileSelected(Message):
    """Posted when user selects a workspace file."""

    filepath: str
    filename: str


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def get_file_info(filepath: str) -> Dict[str, Any]:
    """
    Extract file information for display.

    Args:
        filepath: Absolute path to file

    Returns:
        Dict with name, ext, size_mb, icon, supported, adapter
    """
    path = Path(filepath)
    name = path.name
    ext = path.suffix.lower()

    # Handle double extensions like .h5ad.gz
    if ext == ".gz" and len(path.suffixes) > 1:
        ext = "".join(path.suffixes[-2:])

    # Get file size
    try:
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
    except (OSError, FileNotFoundError):
        size_mb = 0.0

    # Determine format support
    format_info = SUPPORTED_FORMATS.get(ext, {})
    supported = bool(format_info)
    icon = format_info.get("icon", ICONS["unsupported"]) if supported else ICONS["unsupported"]
    adapter = format_info.get("adapter", None)

    return {
        "path": filepath,
        "name": name,
        "ext": ext,
        "size_mb": size_mb,
        "icon": icon,
        "supported": supported,
        "adapter": adapter,
        "state": FileState.WORKSPACE if supported else FileState.UNSUPPORTED,
    }


def is_10x_folder(path: Path) -> bool:
    """Check if a folder contains 10x Genomics MTX data."""
    if not path.is_dir():
        return False

    # Look for standard 10x files
    expected_files = {"matrix.mtx", "barcodes.tsv", "features.tsv", "genes.tsv"}
    found_files = {f.name for f in path.iterdir() if f.is_file()}

    # Also check for .gz versions
    gz_files = {f.name.replace(".gz", "") for f in path.iterdir() if f.name.endswith(".gz")}

    all_files = found_files | gz_files

    # Need matrix.mtx and at least one of barcodes/features/genes
    has_matrix = "matrix.mtx" in all_files
    has_annotations = bool(all_files & {"barcodes.tsv", "features.tsv", "genes.tsv"})

    return has_matrix and has_annotations


# ─────────────────────────────────────────────────────────────────────────────
# List Item Widgets
# ─────────────────────────────────────────────────────────────────────────────


class WorkspaceFileItem(ListItem):
    """
    Single file in the Workspace pane.

    Displays: icon + name + size
    States: workspace, loading, error, unsupported
    """

    DEFAULT_CSS = """
    WorkspaceFileItem {
        height: auto;
        padding: 0 1;
    }
    WorkspaceFileItem.loading {
        color: $warning;
    }
    WorkspaceFileItem.error {
        color: $error;
    }
    WorkspaceFileItem.unsupported {
        color: $text-disabled;
    }
    """

    state = reactive(FileState.WORKSPACE)

    def __init__(self, file_info: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.file_info = file_info
        self.filepath = file_info["path"]
        self.filename = file_info["name"]
        self.state = file_info.get("state", FileState.WORKSPACE)

    def compose(self):
        """Render file item."""
        icon = self._get_icon()
        size = self.file_info.get("size_mb", 0)
        size_str = f"({size:.1f} MB)" if size > 0 else ""
        yield Label(f"{icon} {self.filename} {size_str}")

    def _get_icon(self) -> str:
        """Get icon based on current state."""
        if self.state == FileState.LOADING:
            return ICONS["loading"]
        elif self.state == FileState.ERROR:
            return ICONS["error"]
        elif self.state == FileState.UNSUPPORTED:
            return ICONS["unsupported"]
        else:
            return self.file_info.get("icon", ICONS["workspace"])

    def watch_state(self, new_state: FileState) -> None:
        """Update visual styling when state changes."""
        # Remove all state classes
        self.remove_class("loading", "error", "unsupported")

        # Add appropriate class
        if new_state == FileState.LOADING:
            self.add_class("loading")
        elif new_state == FileState.ERROR:
            self.add_class("error")
        elif new_state == FileState.UNSUPPORTED:
            self.add_class("unsupported")

        # Refresh to update icon
        self.refresh()


class LoadedModalityItem(ListItem):
    """
    Single modality in the Loaded pane.

    Displays: icon + name + shape info
    """

    DEFAULT_CSS = """
    LoadedModalityItem {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, modality_name: str, info: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.modality_name = modality_name
        self.info = info

    def compose(self):
        """Render modality item."""
        icon = ICONS["loaded"]
        size_mb = self.info.get("size_mb", 0)

        # Try to get shape info
        shape = self.info.get("shape", None)
        if shape:
            shape_str = f"({shape[0]:,} × {shape[1]:,})"
        else:
            shape_str = f"({size_mb:.1f} MB)"

        yield Label(f"{icon} {self.modality_name} {shape_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Pane Widgets
# ─────────────────────────────────────────────────────────────────────────────


class WorkspaceView(ListView):
    """
    Top pane: Files on disk in workspace directory.

    Features:
    - Scans workspace directory for data files
    - Shows file state (workspace/loading/error/unsupported)
    - Enter/click to request load
    """

    DEFAULT_CSS = """
    WorkspaceView {
        width: 1fr;
        height: 1fr;
        min-height: 4;
        border: round $primary;
    }
    """

    file_count = reactive(0)

    def __init__(self, workspace_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self.workspace_path = workspace_path
        self._files: Dict[str, Dict[str, Any]] = {}

    def on_mount(self) -> None:
        """Initialize on mount."""
        self.border_title = "Workspace (Drop files)"
        if self.workspace_path:
            self.scan_workspace()

    def scan_workspace(self) -> None:
        """Scan workspace directory for data files."""
        self.clear()
        self._files.clear()

        if not self.workspace_path or not self.workspace_path.exists():
            self._show_empty_message()
            return

        files_found = []

        # Scan for files
        for item in self.workspace_path.iterdir():
            if item.is_file():
                ext = item.suffix.lower()
                # Skip hidden files and session files
                if item.name.startswith("."):
                    continue
                # Only include data files
                if ext in SUPPORTED_FORMATS or ext == ".gz":
                    file_info = get_file_info(str(item))
                    files_found.append(file_info)
            elif item.is_dir() and is_10x_folder(item):
                # Handle 10x folders
                file_info = {
                    "path": str(item),
                    "name": item.name,
                    "ext": "",
                    "size_mb": sum(f.stat().st_size for f in item.rglob("*")) / (1024 * 1024),
                    "icon": ICONS["folder"],
                    "supported": True,
                    "adapter": "10x",
                    "state": FileState.WORKSPACE,
                }
                files_found.append(file_info)

        if not files_found:
            self._show_empty_message()
            return

        # Sort by name and add items
        files_found.sort(key=lambda x: x["name"].lower())
        for file_info in files_found:
            self._files[file_info["path"]] = file_info
            item = WorkspaceFileItem(file_info)
            self.append(item)

        self.file_count = len(files_found)
        self.border_title = f"Workspace ({self.file_count})"

    def _show_empty_message(self) -> None:
        """Show placeholder when workspace is empty."""
        self.append(ListItem(Label("[dim]Drag files here[/dim]")))
        self.file_count = 0
        self.border_title = "Workspace (Drop files)"

    def add_file(self, filepath: str) -> bool:
        """
        Add a file to the workspace view.

        Returns True if file was added, False if already exists or unsupported.
        """
        if filepath in self._files:
            return False

        file_info = get_file_info(filepath)
        self._files[filepath] = file_info

        # Remove empty message if present
        if self.file_count == 0:
            self.clear()

        item = WorkspaceFileItem(file_info)
        self.append(item)
        self.file_count = len(self._files)
        self.border_title = f"Workspace ({self.file_count})"
        return True

    def remove_file(self, filepath: str) -> None:
        """Remove a file from workspace view (e.g., after successful load)."""
        if filepath not in self._files:
            return

        del self._files[filepath]

        # Find and remove the item
        for item in self.query(WorkspaceFileItem):
            if item.filepath == filepath:
                item.remove()
                break

        self.file_count = len(self._files)
        if self.file_count == 0:
            self._show_empty_message()
        else:
            self.border_title = f"Workspace ({self.file_count})"

    def set_file_state(self, filepath: str, state: FileState) -> None:
        """Update the state of a file item."""
        if filepath in self._files:
            self._files[filepath]["state"] = state

        for item in self.query(WorkspaceFileItem):
            if item.filepath == filepath:
                item.state = state
                break

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle file selection - request load."""
        if isinstance(event.item, WorkspaceFileItem):
            file_info = event.item.file_info
            if file_info.get("supported", False) and event.item.state == FileState.WORKSPACE:
                self.post_message(
                    FileLoadRequested(filepath=event.item.filepath, filename=event.item.filename)
                )


class LoadedView(ListView):
    """
    Bottom pane: Modalities loaded in memory.

    Features:
    - Shows all modalities from DataManagerV2
    - Click to select for analysis
    """

    DEFAULT_CSS = """
    LoadedView {
        width: 1fr;
        height: 1fr;
        min-height: 4;
        border: round $accent;
    }
    """

    modality_count = reactive(0)

    def __init__(self, client=None, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def on_mount(self) -> None:
        """Initialize on mount."""
        self.border_title = "Loaded"
        self.refresh_modalities()

    def refresh_modalities(self) -> None:
        """Refresh the list of loaded modalities."""
        self.clear()

        if not self.client:
            self._show_empty_message()
            return

        try:
            datasets = self.client.data_manager.available_datasets
        except AttributeError:
            self._show_empty_message()
            return

        if not datasets:
            self._show_empty_message()
            return

        # Add each modality
        for name, info in datasets.items():
            item = LoadedModalityItem(name, info)
            self.append(item)

        self.modality_count = len(datasets)
        self.border_title = f"Loaded ({self.modality_count})"

    def _show_empty_message(self) -> None:
        """Show placeholder when no modalities loaded."""
        self.append(ListItem(Label("[dim]No data loaded[/dim]")))
        self.modality_count = 0
        self.border_title = "Loaded"

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle modality selection."""
        if isinstance(event.item, LoadedModalityItem):
            self.post_message(ModalitySelected(modality_name=event.item.modality_name))


# ─────────────────────────────────────────────────────────────────────────────
# Main DataHub Widget
# ─────────────────────────────────────────────────────────────────────────────


class DataHub(Static, can_focus=True):
    """
    Unified data management hub for the Lobster dashboard.

    Combines:
    - WorkspaceView: Files on disk (drag-and-drop target)
    - LoadedView: Modalities in memory

    Layout (stacked vertically):
    ┌─ DataHub ───────────────────────────────┐
    │ ┌─ Workspace (Drop files) ────────────┐ │
    │ │ 📁 data.h5ad (45 MB)                │ │
    │ │ 📁 metadata.csv (2 MB)              │ │
    │ │ ⏳ loading.h5ad [▆▆▆--] 30%         │ │
    │ │ ❓ unknown.zip (unsupported)        │ │
    │ └─────────────────────────────────────┘ │
    │ ┌─ Loaded ────────────────────────────┐ │
    │ │ 🔬 geo_gse12345 (1.2k × 15k)        │ │
    │ │ 🔬 bulk_expr (500 × 20k)            │ │
    │ └─────────────────────────────────────┘ │
    └─────────────────────────────────────────┘
    """

    DEFAULT_CSS = """
    DataHub {
        height: auto;
        min-height: 12;
        max-height: 20;
        border: round $primary;
        padding: 0;
    }

    DataHub > Vertical {
        height: 100%;
    }

    DataHub:focus-within {
        border: round $accent;
    }
    """

    def __init__(
        self,
        client=None,
        workspace_path: Optional[Path] = None,
        on_load_file: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client = client
        self.workspace_path = workspace_path
        self.on_load_file = on_load_file  # Callback for loading files
        self._workspace_view: Optional[WorkspaceView] = None
        self._loaded_view: Optional[LoadedView] = None

    def compose(self):
        """Compose the dual-pane layout (stacked vertically)."""
        with Vertical():
            self._workspace_view = WorkspaceView(
                workspace_path=self.workspace_path, id="workspace-view"
            )
            yield self._workspace_view

            self._loaded_view = LoadedView(client=self.client, id="loaded-view")
            yield self._loaded_view

    def on_mount(self) -> None:
        """Initialize on mount and set focus to enable drag-and-drop."""
        self.border_title = "Data Hub (Drop files here)"
        # Focus this widget to enable paste event handling for file drops
        self.focus()

    # ─────────────────────────────────────────────────────────────────────
    # File Drop Handling (textual-filedrop pattern)
    # ─────────────────────────────────────────────────────────────────────

    def on_paste(self, event: events.Paste) -> None:
        """
        Handle paste events (file drag-and-drop).

        Terminal file drops are typically handled as paste events with file paths.
        """
        text = event.text.strip()

        # Debug: Show notification that paste event was received
        logger.info(f"DataHub received paste event: {text[:100]}")

        if not text:
            self.notify("Paste event received but no text", severity="information")
            return

        # Extract file paths from pasted text
        filepaths = self._extract_filepaths(text)

        if filepaths:
            logger.info(f"Extracted {len(filepaths)} file paths: {filepaths}")
            event.prevent_default()
            self._handle_file_drop(filepaths)
        else:
            # Show notification if no valid files found
            self.notify(f"No valid files found in paste: {text[:50]}", severity="warning")

    def _extract_filepaths(self, text: str) -> List[str]:
        """
        Extract valid file paths from pasted text.

        Handles:
        - Single file paths
        - Multiple paths (space or newline separated)
        - Quoted paths (for paths with spaces)
        """
        import shlex

        filepaths = []

        try:
            # Try to parse as shell arguments (handles quotes)
            parts = shlex.split(text)
        except ValueError:
            # Fallback to simple split
            parts = text.split()

        for part in parts:
            # Clean up the path
            path_str = part.strip().replace("\x00", "").replace('"', "")

            if not path_str:
                continue

            path = Path(path_str)

            # Check if it's a valid file or directory
            if path.exists():
                if path.is_file():
                    filepaths.append(str(path.absolute()))
                elif path.is_dir():
                    # Check if it's a 10x folder
                    if is_10x_folder(path):
                        filepaths.append(str(path.absolute()))
                    else:
                        # Recursively add data files from directory
                        for item in path.rglob("*"):
                            if item.is_file() and item.suffix.lower() in SUPPORTED_FORMATS:
                                filepaths.append(str(item.absolute()))

        return filepaths

    def _handle_file_drop(self, filepaths: List[str]) -> None:
        """
        Process dropped files.

        1. Copy to workspace directory
        2. Add to WorkspaceView
        3. Show notification
        """
        if not self.workspace_path:
            self.notify("Workspace not configured", severity="error")
            return

        added_count = 0
        error_count = 0

        for filepath in filepaths:
            src_path = Path(filepath)

            if not src_path.exists():
                continue

            # Determine destination
            dest_path = self.workspace_path / src_path.name

            # Handle name conflicts
            if dest_path.exists():
                base = dest_path.stem
                suffix = dest_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = self.workspace_path / f"{base}_{counter}{suffix}"
                    counter += 1

            try:
                # Copy file or directory
                if src_path.is_dir():
                    shutil.copytree(src_path, dest_path)
                else:
                    shutil.copy2(src_path, dest_path)

                # Add to workspace view
                if self._workspace_view and self._workspace_view.add_file(str(dest_path)):
                    added_count += 1

            except (OSError, shutil.Error) as e:
                logger.error(f"Failed to copy {src_path}: {e}")
                error_count += 1

        # Show notification
        if added_count > 0:
            self.notify(f"Added {added_count} file(s) to workspace", severity="information")
        if error_count > 0:
            self.notify(f"Failed to copy {error_count} file(s)", severity="warning")

        # Post message for parent to handle
        self.post_message(
            FileDropped(
                filepaths=[str(self.workspace_path / Path(f).name) for f in filepaths[:added_count]],
                filenames=[Path(f).name for f in filepaths[:added_count]],
            )
        )

    # ─────────────────────────────────────────────────────────────────────
    # Load File Handling
    # ─────────────────────────────────────────────────────────────────────

    def on_file_load_requested(self, event: FileLoadRequested) -> None:
        """Handle request to load a workspace file."""
        # Update state to loading
        if self._workspace_view:
            self._workspace_view.set_file_state(event.filepath, FileState.LOADING)

        # Call the load callback if provided
        if self.on_load_file:
            try:
                self.on_load_file(event.filepath)
            except Exception as e:
                logger.error(f"Failed to load {event.filename}: {e}")
                if self._workspace_view:
                    self._workspace_view.set_file_state(event.filepath, FileState.ERROR)
                self.notify(f"Failed to load: {str(e)[:50]}", severity="error")
                return

        # On success, remove from workspace and refresh loaded
        if self._workspace_view:
            self._workspace_view.remove_file(event.filepath)
        if self._loaded_view:
            self._loaded_view.refresh_modalities()

        self.notify(f"Loaded {event.filename}", severity="information")

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def refresh_all(self) -> None:
        """Refresh both panes."""
        if self._workspace_view:
            self._workspace_view.scan_workspace()
        if self._loaded_view:
            self._loaded_view.refresh_modalities()

    def refresh_workspace(self) -> None:
        """Refresh workspace pane only."""
        if self._workspace_view:
            self._workspace_view.scan_workspace()

    def refresh_loaded(self) -> None:
        """Refresh loaded pane only."""
        if self._loaded_view:
            self._loaded_view.refresh_modalities()

    @property
    def workspace_view(self) -> Optional[WorkspaceView]:
        """Access workspace view."""
        return self._workspace_view

    @property
    def loaded_view(self) -> Optional[LoadedView]:
        """Access loaded view."""
        return self._loaded_view
