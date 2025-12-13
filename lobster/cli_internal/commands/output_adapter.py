"""
Output adapter for command results - abstracts CLI vs Dashboard rendering.

Design: Commands return structured data + rendering hints. OutputAdapter
handles the actual formatting for Rich Console (CLI) or ResultsDisplay (Dashboard).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box


class OutputAdapter(ABC):
    """
    Abstract interface for command output.

    Allows commands to be UI-agnostic by providing a common API for
    displaying text, tables, and confirmations.
    """

    @abstractmethod
    def print(self, message: str, style: Optional[str] = None) -> None:
        """
        Print a message with optional style.

        Args:
            message: Message text (supports Rich markup for CLI, plain for Dashboard)
            style: Style hint - "info", "success", "warning", "error"
        """
        pass

    @abstractmethod
    def print_table(self, table_data: Dict[str, Any]) -> None:
        """
        Render a table from structured data.

        Args:
            table_data: Dict with keys:
                - columns: List[Dict] with "name", "style", "width" keys
                - rows: List[List[str]] - table data
                - title: Optional[str] - table title
        """
        pass

    @abstractmethod
    def confirm(self, question: str) -> bool:
        """
        Ask user for confirmation.

        Args:
            question: Question to ask

        Returns:
            bool: True if confirmed, False otherwise
        """
        pass

    @abstractmethod
    def print_code_block(self, code: str, language: str = "python") -> None:
        """
        Print formatted code block.

        Args:
            code: Code content
            language: Syntax highlighting language hint
        """
        pass


class ConsoleOutputAdapter(OutputAdapter):
    """OutputAdapter for Rich Console (CLI mode)."""

    def __init__(self, console: Console):
        self.console = console

    def print(self, message: str, style: Optional[str] = None) -> None:
        """Print with Rich markup."""
        self.console.print(message)

    def print_table(self, table_data: Dict[str, Any]) -> None:
        """Render Rich Table."""
        table = Table(box=box.ROUNDED, title=table_data.get("title"))

        # Add columns
        for col in table_data.get("columns", []):
            table.add_column(
                col["name"],
                style=col.get("style", "white"),
                width=col.get("width"),
                max_width=col.get("max_width"),
                overflow=col.get("overflow", "fold"),
                justify=col.get("justify", "left"),
            )

        # Add rows
        for row in table_data.get("rows", []):
            table.add_row(*row)

        self.console.print(table)

    def confirm(self, question: str) -> bool:
        """Ask for confirmation using Rich Confirm."""
        from rich.prompt import Confirm
        return Confirm.ask(question)

    def print_code_block(self, code: str, language: str = "python") -> None:
        """Print syntax-highlighted code."""
        from rich.syntax import Syntax
        syntax = Syntax(code, language, theme="monokai")
        self.console.print(syntax)


class DashboardOutputAdapter(OutputAdapter):
    """OutputAdapter for Textual ResultsDisplay (Dashboard mode)."""

    def __init__(self, results_display):
        """
        Initialize with ResultsDisplay widget.

        Args:
            results_display: ResultsDisplay widget instance
        """
        self.results_display = results_display

    def print(self, message: str, style: Optional[str] = None) -> None:
        """
        Print to ResultsDisplay as system message.

        Strips Rich markup for cleaner dashboard display.
        """
        # Strip Rich markup for dashboard
        clean_message = self._strip_markup(message)
        self.results_display.append_system_message(clean_message)

    def print_table(self, table_data: Dict[str, Any]) -> None:
        """
        Render table as formatted markdown in dashboard.

        Uses GitHub-flavored markdown table syntax.
        """
        # Build markdown table
        columns = table_data.get("columns", [])
        rows = table_data.get("rows", [])

        if not columns or not rows:
            return

        # Header
        header = "| " + " | ".join(col["name"] for col in columns) + " |"
        separator = "|" + "|".join("---" for _ in columns) + "|"

        # Rows
        table_rows = []
        for row in rows:
            table_rows.append("| " + " | ".join(row) + " |")

        # Combine
        title = table_data.get("title", "")
        markdown = f"**{title}**\n\n" if title else ""
        markdown += header + "\n" + separator + "\n" + "\n".join(table_rows)

        self.results_display.append_system_message(markdown)

    def confirm(self, question: str) -> bool:
        """
        Dashboard doesn't support interactive confirmations.

        Returns False (safe default - no destructive operations).
        Display message to user instead.
        """
        clean_question = self._strip_markup(question)
        self.results_display.append_system_message(
            f"⚠️ Confirmation required: {clean_question}\n"
            "Interactive confirmations not supported in dashboard. "
            "Use CLI mode for destructive operations."
        )
        return False

    def print_code_block(self, code: str, language: str = "python") -> None:
        """Print code as markdown code block."""
        markdown = f"```{language}\n{code}\n```"
        self.results_display.append_system_message(markdown)

    def _strip_markup(self, text: str) -> str:
        """
        Strip Rich markup tags from text.

        Args:
            text: Text with Rich markup (e.g., "[cyan]text[/cyan]")

        Returns:
            Plain text without markup
        """
        import re
        # Remove [style] and [/style] tags
        return re.sub(r'\[/?[^\]]+\]', '', text)
