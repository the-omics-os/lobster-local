"""
LobsterOS - Minimal terminal UI for Lobster bioinformatics platform.

Design: Terminal-native, transparent backgrounds, minimal color palette.
Press Ctrl+P to open command palette.
"""

# Python 3.13 compatibility: Fix multiprocessing/tqdm file descriptor issue
# Must be done before any imports that might use multiprocessing
import os
import sys

if sys.version_info >= (3, 13):
    # Disable tqdm's multiprocessing lock (alternative fix)
    os.environ.setdefault("TQDM_DISABLE", "0")  # Don't fully disable, just note we tried

    import multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # Already set

from pathlib import Path
from typing import Optional

from textual.app import App
from textual.binding import Binding

from lobster.core.client import AgentClient
from lobster.core.workspace import resolve_workspace
from lobster.ui.screens import AnalysisScreen
from lobster.ui.commands import LobsterCommands


class LobsterOS(App):
    """
    Minimal terminal-native bioinformatics workspace.

    Design principles:
    - Transparent backgrounds (inherit terminal theme)
    - Single accent color (Lobster Orange #CC2C18) for active/important elements
    - Clean borders, no visual noise

    Press Ctrl+P to open command palette with all available commands.
    """

    # Lobster Orange brand theme
    CSS = """
    * {
        scrollbar-size: 1 1;
    }

    /* Override Textual's default primary color with Lobster Orange */
    $primary: #CC2C18;
    $primary-lighten-1: #E84D3A;
    $primary-darken-1: #A82414;
    $accent: #CC2C18;

    Screen {
        background: transparent;
    }

    Header {
        background: transparent;
        color: #CC2C18;
    }

    Footer {
        background: transparent;
    }
    """

    # Register command providers for Ctrl+P palette
    COMMANDS = {LobsterCommands}

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Commands", key_display="^P"),
        Binding("q", "quit", "Quit", key_display="Q"),
        Binding("f5", "refresh", "Refresh", key_display="F5"),
    ]

    TITLE = "lobster"

    def __init__(self, workspace_path: Optional[Path] = None):
        super().__init__()
        self.workspace_path = resolve_workspace(workspace_path, create=True)
        self.client: Optional[AgentClient] = None

    def on_mount(self) -> None:
        """Initialize the application."""
        self.sub_title = str(self.workspace_path.name)

        try:
            self.client = AgentClient(
                workspace_path=self.workspace_path,
                enable_reasoning=True,
            )
            self.push_screen(AnalysisScreen(self.client))

        except Exception as e:
            self.notify(f"Init failed: {e}", severity="error", timeout=30)

    def action_refresh(self) -> None:
        """Refresh all panels."""
        screen = self.screen
        if hasattr(screen, "refresh_all"):
            screen.refresh_all()


def run_lobster_os(workspace_path: Optional[Path] = None):
    """Entry point for lobster os command."""
    # Ensure multiprocessing fix is applied (for direct function calls)
    if sys.version_info >= (3, 13):
        import multiprocessing as mp
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass  # Already set

    app = LobsterOS(workspace_path)
    app.run()
