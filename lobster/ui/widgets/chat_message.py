"""Chat message bubble widget for user/agent messages."""

from textual.widgets import Static, Markdown as TextualMarkdown
from textual.reactive import reactive


class ChatMessage(Static):
    """
    Message bubble for user or agent (Elia pattern).

    Features:
    - Different styling for user vs agent
    - Streaming indicator for agent messages
    - Markdown rendering with word wrap
    - Auto-sizing (height: auto)
    """

    is_streaming = reactive(False)

    def __init__(self, content: str, is_user: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.content = content
        self.is_user = is_user
        self._markdown = None

    def compose(self):
        """Render Markdown content."""
        self._markdown = TextualMarkdown(self.content)
        yield self._markdown

    def on_mount(self) -> None:
        """Set border and styling based on role."""
        if self.is_user:
            self.add_class("user-message")
            self.border_title = "You"
        else:
            self.add_class("agent-message")
            self.border_title = "Lobster"

    def append_chunk(self, chunk: str) -> None:
        """Append streaming chunk (agent messages only)."""
        self.content += chunk
        if self._markdown:
            self._markdown.update(self.content)

    def watch_is_streaming(self, streaming: bool) -> None:
        """Update styling during streaming."""
        if streaming:
            self.add_class("streaming")
            self.border_title = "Lobster (streaming...)"
        else:
            self.remove_class("streaming")
            self.border_title = "Lobster"
