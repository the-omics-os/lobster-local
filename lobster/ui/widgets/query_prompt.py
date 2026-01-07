"""Query prompt widget for user input."""

from dataclasses import dataclass
from textual.widgets import TextArea
from textual.message import Message
from textual.reactive import reactive
from textual.binding import Binding


class QueryPrompt(TextArea):
    """
    Multi-line text input for user queries (Elia pattern).

    Features:
    - Submit lock (prevents double-submission while agent works)
    - Enter to submit
    - Shift+Enter for newlines
    - Visual feedback when locked
    - Immediate clear on submit (Elia pattern)
    """

    @dataclass
    class QuerySubmitted(Message):
        """Posted when user submits a query."""

        text: str
        prompt_input: "QueryPrompt"

    submit_ready = reactive(True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No syntax highlighting needed for query input

    def on_key(self, event) -> None:
        """Handle key presses - intercept Enter before TextArea processes it."""
        if event.key == "enter":
            # Plain Enter = submit
            event.prevent_default()
            event.stop()
            self.action_submit_query()
        # Shift+Enter is handled as "shift+enter" key, which TextArea handles naturally for newlines

    def watch_submit_ready(self, ready: bool) -> None:
        """Update visual state when submit lock changes."""
        if not ready:
            self.add_class("locked")
            self.border_subtitle = "🔒 Agent is working..."
        else:
            self.remove_class("locked")
            self.border_subtitle = "Enter to submit | Shift+Enter for newline"

    def action_submit_query(self) -> None:
        """Submit query (Enter key) - Elia pattern."""
        # Validate not empty
        if self.text.strip() == "":
            self.notify("Cannot submit empty query!", severity="warning")
            return

        # Validate not locked
        if not self.submit_ready:
            self.app.bell()
            self.notify("Please wait for processing to complete.", severity="warning")
            return

        # Clear IMMEDIATELY (Elia pattern - before posting!)
        message = self.QuerySubmitted(self.text, prompt_input=self)
        self.clear()

        # Post message to parent screen
        self.post_message(message)

    def on_mount(self) -> None:
        """Set initial state."""
        self.border_title = "Query"
        self.border_subtitle = "Enter to submit | Shift+Enter for newline"
