"""Modality list widget showing loaded datasets."""

from dataclasses import dataclass
from typing import Optional

from textual.widgets import ListView, ListItem, Label
from textual.message import Message
from textual.reactive import reactive


@dataclass
class ModalitySelected(Message):
    """Posted when user selects a modality."""

    modality_name: str


class ModalityListItem(ListItem):
    """Single modality list item with metadata."""

    def __init__(self, modality_name: str, size_mb: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modality_name = modality_name
        self.size_mb = size_mb

    def compose(self):
        """Render modality item."""
        # Show name + size
        yield Label(f"● {self.modality_name}  ({self.size_mb:.1f} MB)")


class ModalityList(ListView):
    """
    Scrollable list of loaded modalities/datasets.

    Features:
    - Real-time updates when modalities added/removed
    - Shows dataset size
    - Keyboard navigation (hjkl/arrows)
    - Selection posts ModalitySelected message
    """

    modality_count = reactive(0)

    def __init__(self, client=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client

    def on_mount(self) -> None:
        """Load modalities on mount."""
        self.border_title = "Modalities"
        self.refresh_modalities()

    def refresh_modalities(self) -> None:
        """Refresh the list of modalities."""
        self.clear()

        if not self.client:
            self.append(ListItem(Label("No client loaded")))
            return

        datasets = self.client.data_manager.available_datasets

        if not datasets:
            self.append(ListItem(Label("No data loaded")))
            self.modality_count = 0
            return

        # Add each modality
        for name, info in datasets.items():
            size_mb = info.get("size_mb", 0.0)
            self.append(ModalityListItem(name, size_mb))

        self.modality_count = len(datasets)
        self.border_title = f"Modalities ({self.modality_count})"

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle modality selection."""
        if isinstance(event.item, ModalityListItem):
            self.post_message(ModalitySelected(event.item.modality_name))
            self.notify(f"Selected: {event.item.modality_name}", timeout=2)

    def watch_modality_count(self, count: int) -> None:
        """Update border title when count changes."""
        self.border_title = f"Modalities ({count})" if count > 0 else "Modalities"
