"""Results display widget - conversation container (Elia pattern)."""

from typing import Optional

from textual.containers import VerticalScroll
from textual.reactive import reactive

from lobster.ui.widgets.chat_message import ChatMessage


class ResultsDisplay(VerticalScroll):
    """
    Conversation container with vertical scrolling (Elia pattern).

    Features:
    - User/agent message bubbles with differentiation
    - Vertical scrolling for history (NO horizontal scroll)
    - Smart auto-scroll (only if user at bottom)
    - Content wraps to width
    - Minimal styling (grayscale + orange)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.can_focus = False  # Prevent focus (like Elia)
        self.current_agent_message: Optional[ChatMessage] = None

    def on_mount(self) -> None:
        """Set initial state."""
        self.border_title = "Conversation"

    def append_user_message(self, text: str) -> None:
        """Add user message bubble."""
        message = ChatMessage(text, is_user=True)
        self.mount(message)

        # Smart auto-scroll (Elia pattern - only if at bottom)
        if self.scroll_offset.y >= self.max_scroll_y - 3:
            self.scroll_end(animate=False)

    def start_agent_message(self) -> ChatMessage:
        """Create new agent message bubble for streaming."""
        message = ChatMessage("", is_user=False)
        message.is_streaming = True
        self.mount(message)
        self.current_agent_message = message

        # Auto-scroll
        if self.scroll_offset.y >= self.max_scroll_y - 3:
            self.scroll_end(animate=False)

        return message

    def append_to_agent_message(self, chunk: str) -> None:
        """Append chunk to current agent message (streaming)."""
        if self.current_agent_message:
            self.current_agent_message.append_chunk(chunk)

            # Auto-scroll during streaming
            if self.scroll_offset.y >= self.max_scroll_y - 3:
                self.scroll_end(animate=False)

    def complete_agent_message(self) -> None:
        """Mark agent message as complete."""
        if self.current_agent_message:
            self.current_agent_message.is_streaming = False
            self.current_agent_message = None

    def show_error(self, error: str) -> None:
        """Show error message as agent message."""
        message = ChatMessage(f"❌ Error: {error}", is_user=False)
        message.add_class("error-message")
        self.mount(message)

        # Auto-scroll
        if self.scroll_offset.y >= self.max_scroll_y - 3:
            self.scroll_end(animate=False)

    def append_system_message(self, content: str) -> None:
        """Add system/command output message."""
        message = ChatMessage(content, is_user=False)
        message.add_class("system-message")
        self.mount(message)

        # Auto-scroll
        if self.scroll_offset.y >= self.max_scroll_y - 3:
            self.scroll_end(animate=False)

    def clear_display(self) -> None:
        """Clear all messages."""
        # Remove all ChatMessage children
        for child in self.query(ChatMessage):
            child.remove()
        self.current_agent_message = None
