"""
Rich theme system for Lobster AI with orange branding.

This module provides comprehensive theming for the Lobster AI CLI with
consistent orange (#e45c47) branding, professional styling, and support
for different data types and contexts.
"""

import os
from typing import Any, Dict, Optional

from rich import box
from rich.color import Color
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from rich.theme import Theme


class LobsterTheme:
    """
    Comprehensive theme system for Lobster AI CLI.

    Provides consistent orange branding (#e45c47) with professional
    styling for all Rich components and data types.
    """

    # Core Brand Colors
    PRIMARY_ORANGE = "#e45c47"
    BACKGROUND_WHITE = "white"
    TEXT_BLACK = "white"  # Changed to white for dark terminal compatibility

    # Extended Color Palette
    COLORS = {
        # Primary branding
        "lobster_orange": PRIMARY_ORANGE,
        "lobster_orange_bright": "#ff6b52",
        "lobster_orange_dark": "#c44836",
        "lobster_orange_dim": "#b23a29",
        # Background and text
        "background": BACKGROUND_WHITE,
        "text_primary": TEXT_BLACK,
        "text_secondary": "#cccccc",  # Lighter for dark backgrounds
        "text_tertiary": "#aaaaaa",  # Lighter for dark backgrounds
        "text_muted": "#888888",  # Lighter for dark backgrounds
        # Status colors with orange variations
        "success": "#28a745",
        "success_orange": "#e4a547",
        "warning": "#ffc107",
        "warning_orange": "#ff8c47",
        "error": "#dc3545",
        "error_orange": "#e44747",
        "info": "#17a2b8",
        "info_orange": "#e49c47",
        # Data type specific colors
        "genomics": "#4CAF50",
        "genomics_orange": "#e4af47",
        "proteomics": "#2196F3",
        "proteomics_orange": "#477ce4",
        "metabolomics": "#9C27B0",
        "metabolomics_orange": "#c447e4",
        # UI element colors
        "border": PRIMARY_ORANGE,
        "border_dim": "#e4a547",
        "highlight": PRIMARY_ORANGE,
        "selection": "#ffe4df",
        "hover": "#fff0ee",
        # Progress and status indicators
        "progress_complete": PRIMARY_ORANGE,
        "progress_incomplete": "#333333",  # Darker for visibility on dark backgrounds
        "loading": PRIMARY_ORANGE,
        "active": PRIMARY_ORANGE,
        "inactive": "#666666",  # More visible on dark backgrounds
    }

    # Rich Theme Styles
    RICH_THEME = Theme(
        {
            # Brand styles
            "lobster.primary": f"bold {PRIMARY_ORANGE}",
            "lobster.logo": f"bold {PRIMARY_ORANGE}",
            "lobster.highlight": f"{PRIMARY_ORANGE}",
            "lobster.accent": f"italic {PRIMARY_ORANGE}",
            # Text hierarchy
            "text.primary": TEXT_BLACK,
            "text.secondary": "#cccccc",  # Lighter for dark backgrounds
            "text.tertiary": "#aaaaaa",  # Lighter for dark backgrounds
            "text.muted": "#888888",  # Lighter for dark backgrounds
            "text.inverse": f"{BACKGROUND_WHITE} on {PRIMARY_ORANGE}",
            # Status styles
            "status.success": "bold #28a745",
            "status.warning": "bold #ffc107",
            "status.error": "bold #dc3545",
            "status.info": "bold #17a2b8",
            "status.loading": f"bold {PRIMARY_ORANGE}",
            # Data styles
            "data.header": f"bold {PRIMARY_ORANGE}",
            "data.value": TEXT_BLACK,
            "data.key": "#cccccc",  # Lighter for dark backgrounds
            "data.number": "#66aaff",  # Brighter blue for dark backgrounds
            "data.string": "#66ff66",  # Brighter green for dark backgrounds
            "data.boolean": "#ffaa66",  # Brighter orange for dark backgrounds
            # Interactive elements
            "interactive.prompt": f"bold {PRIMARY_ORANGE}",
            "interactive.choice": f"{PRIMARY_ORANGE}",
            "interactive.selected": f"reverse {PRIMARY_ORANGE}",
            "interactive.cursor": f"bold {PRIMARY_ORANGE}",
            # Progress elements
            "progress.bar": f"{PRIMARY_ORANGE}",
            "progress.complete": f"bold {PRIMARY_ORANGE}",
            "progress.percentage": f"bold {PRIMARY_ORANGE}",
            "progress.data": "#aaaaaa",  # Lighter for dark backgrounds
            # Panel and border styles
            "panel.title": f"bold {PRIMARY_ORANGE}",
            "panel.border": PRIMARY_ORANGE,
            "border.primary": PRIMARY_ORANGE,
            "border.secondary": "#e4a547",
            "error": "#dc3545",
            "success": "#28a745",
            # Tree styles
            "tree.line": PRIMARY_ORANGE,
            "tree.guide": "#e4a547",
            "tree.folder": f"bold {PRIMARY_ORANGE}",
            "tree.file": TEXT_BLACK,
            # Table styles
            "table.header": f"bold {PRIMARY_ORANGE}",  # Removed white background
            "table.cell": TEXT_BLACK,
            "table.row_odd": "default",  # Use terminal default background
            "table.row_even": "default",  # Use terminal default background
            # Agent and system styles
            "agent.name": f"bold {PRIMARY_ORANGE}",
            "agent.active": f"bold reverse {PRIMARY_ORANGE}",
            "system.healthy": "bold #28a745",
            "system.warning": "bold #ffc107",
            "system.error": "bold #dc3545",
            # Analysis specific styles
            "analysis.metric": f"bold {PRIMARY_ORANGE}",
            "analysis.result": TEXT_BLACK,
            "analysis.significant": "bold #28a745",
            "analysis.nonsignificant": "#aaaaaa",  # Lighter for dark backgrounds
        }
    )

    # Box styles with orange branding
    BOXES = {
        "primary": box.ROUNDED,
        "secondary": box.SIMPLE,
        "minimal": box.MINIMAL,
        "double": box.DOUBLE,
        "heavy": box.HEAVY,
    }

    @classmethod
    def get_theme(cls, variant: str = "default") -> Theme:
        """Get Rich theme variant."""
        return cls.RICH_THEME

    @classmethod
    def get_color(cls, name: str) -> str:
        """Get color by name from the palette."""
        return cls.COLORS.get(name, cls.PRIMARY_ORANGE)

    @classmethod
    def create_panel(
        cls,
        content: Any,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        border_style: str = "border.primary",
        title_style: str = "panel.title",
        box_style: str = "primary",
    ) -> Panel:
        """Create a themed panel with orange branding."""
        return Panel(
            content,
            title=title,
            subtitle=subtitle,
            border_style=cls.get_color("border"),
            title_align="left",
            box=cls.BOXES[box_style],
            padding=(1, 2),
        )

    @classmethod
    def create_title_text(cls, text: str, emoji: Optional[str] = None) -> Text:
        """Create styled title text with optional emoji."""
        if emoji:
            title = Text()
            title.append(f"{emoji} ", style="")
            title.append(text, style="lobster.primary")
            return title
        return Text(text, style="lobster.primary")

    @classmethod
    def create_status_text(cls, text: str, status: str = "info") -> Text:
        """Create status text with appropriate styling."""
        style_map = {
            "success": "status.success",
            "warning": "status.warning",
            "error": "status.error",
            "info": "status.info",
            "loading": "status.loading",
        }
        return Text(text, style=style_map.get(status, "status.info"))

    @classmethod
    def get_progress_style(cls) -> Dict[str, Any]:
        """Get progress bar styling configuration."""
        return {
            "bar_width": None,
            "progress_type": "[progress.percentage]{task.percentage:>3.0f}%",
            "speed": "[progress.data]{task.speed}",
            "time_remaining": "[progress.data]{task.time_remaining}",
            "completed": f"[bold {cls.PRIMARY_ORANGE}]",
            "remaining": "[#333333]",  # Darker for visibility on dark backgrounds
        }

    @classmethod
    def get_table_style(cls) -> Dict[str, Any]:
        """Get table styling configuration."""
        return {
            "header_style": f"bold {cls.PRIMARY_ORANGE}",  # Removed white background
            "border_style": cls.PRIMARY_ORANGE,
            "row_styles": ["default", "default"],  # Use terminal default backgrounds
            "show_header": True,
            "show_lines": True,
            "box": cls.BOXES["primary"],
        }

    @classmethod
    def get_tree_style(cls) -> Dict[str, Any]:
        """Get tree styling configuration."""
        return {
            "guide_style": cls.get_color("lobster_orange_dim"),
            "expanded": True,
            "highlight": True,
        }

    @classmethod
    def is_dark_mode(cls) -> bool:
        """Check if dark mode is enabled via environment variable."""
        return os.getenv("LOBSTER_DARK_MODE", "false").lower() == "true"

    @classmethod
    def get_environment_theme(cls) -> Theme:
        """Get theme based on environment settings."""
        if cls.is_dark_mode():
            # Dark mode variant (future enhancement)
            return cls.RICH_THEME
        return cls.RICH_THEME


# Pre-configured theme instances
DEFAULT_THEME = LobsterTheme.get_theme()
DARK_THEME = LobsterTheme.get_environment_theme()


# Quick access functions
def get_lobster_color(name: str = "lobster_orange") -> str:
    """Quick access to lobster colors."""
    return LobsterTheme.get_color(name)


def create_lobster_panel(content: Any, title: str = None) -> Panel:
    """Quick create orange-themed panel."""
    return LobsterTheme.create_panel(content, title=title)


def create_lobster_title(text: str, emoji: str = "🦞") -> Text:
    """Quick create lobster-themed title."""
    return LobsterTheme.create_title_text(text, emoji)
