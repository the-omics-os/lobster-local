"""
Rich Tree file navigation component for Lobster AI.

This module provides hierarchical file browsing with orange theming,
status indicators, and interactive navigation capabilities.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from rich.console import Console
from rich.filesize import decimal
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree

from ..themes import LobsterTheme


@dataclass
class FileInfo:
    """Information about a file or directory."""

    path: Path
    name: str
    is_dir: bool
    size: int = 0
    modified: float = 0.0
    has_data: bool = False
    file_type: str = "unknown"
    status: str = "normal"  # normal, active, processed, error


class LobsterFileTree:
    """
    Rich Tree component for hierarchical file navigation with orange theming.

    Provides expandable directory trees, file status indicators, and
    interactive navigation for the Lobster AI workspace.
    """

    def __init__(
        self,
        root_path: Path,
        show_hidden: bool = False,
        max_depth: int = 3,
        file_filter: Optional[Callable[[Path], bool]] = None,
    ):
        """
        Initialize file tree component.

        Args:
            root_path: Root directory to display
            show_hidden: Whether to show hidden files/directories
            max_depth: Maximum tree depth to display
            file_filter: Optional function to filter files
        """
        self.root_path = Path(root_path).resolve()
        self.show_hidden = show_hidden
        self.max_depth = max_depth
        self.file_filter = file_filter or (lambda p: True)

        # File type detection patterns
        self.data_extensions = {
            ".h5ad",
            ".h5",
            ".csv",
            ".tsv",
            ".txt",
            ".xlsx",
            ".xls",
            ".mtx",
            ".gz",
            ".bz2",
            ".zip",
            ".tar",
        }
        self.bio_extensions = {
            ".fastq",
            ".fq",
            ".fasta",
            ".fa",
            ".bam",
            ".sam",
            ".vcf",
            ".bed",
            ".gtf",
            ".gff",
            ".wig",
            ".bigwig",
            ".bedgraph",
        }
        self.analysis_extensions = {
            ".py",
            ".r",
            ".R",
            ".ipynb",
            ".rmd",
            ".md",
            ".sh",
            ".yaml",
            ".yml",
        }

        # Status tracking
        self.file_statuses: Dict[str, str] = {}
        self.processed_files: Set[str] = set()
        self.active_files: Set[str] = set()

    def _detect_file_type(self, path: Path) -> str:
        """Detect file type based on extension and content."""
        if path.is_dir():
            return "directory"

        suffix = path.suffix.lower()

        if suffix in self.data_extensions:
            return "data"
        elif suffix in self.bio_extensions:
            return "bioinformatics"
        elif suffix in self.analysis_extensions:
            return "analysis"
        elif suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html"}:
            return "visualization"
        elif suffix in {".log", ".out", ".err"}:
            return "log"
        else:
            return "unknown"

    def _get_file_info(self, path: Path) -> FileInfo:
        """Get comprehensive file information."""
        try:
            stat = path.stat()
            file_type = self._detect_file_type(path)
            status = self.file_statuses.get(str(path), "normal")

            # Check if it's a data file
            has_data = file_type in ["data", "bioinformatics"]

            return FileInfo(
                path=path,
                name=path.name,
                is_dir=path.is_dir(),
                size=stat.st_size if not path.is_dir() else 0,
                modified=stat.st_mtime,
                has_data=has_data,
                file_type=file_type,
                status=status,
            )
        except (OSError, PermissionError):
            return FileInfo(
                path=path,
                name=path.name,
                is_dir=path.is_dir(),
                file_type="error",
                status="error",
            )

    def _create_file_text(self, file_info: FileInfo) -> Text:
        """Create styled text for a file entry."""
        text = Text()

        # Add file type icon
        icons = {
            "directory": "📁",
            "data": "📊",
            "bioinformatics": "🧬",
            "analysis": "📝",
            "visualization": "📈",
            "log": "📋",
            "unknown": "📄",
            "error": "❌",
        }

        icon = icons.get(file_info.file_type, "📄")
        text.append(f"{icon} ", style="")

        # Add file name with appropriate styling
        name_style = "white"
        if file_info.is_dir:
            name_style = f"bold {LobsterTheme.PRIMARY_ORANGE}"
        elif file_info.has_data:
            name_style = f"{LobsterTheme.PRIMARY_ORANGE}"
        elif file_info.file_type == "analysis":
            name_style = "cyan"
        elif file_info.file_type == "visualization":
            name_style = "green"

        text.append(file_info.name, style=name_style)

        # Add status indicators
        if file_info.status == "active":
            text.append(" ●", style=f"bold {LobsterTheme.PRIMARY_ORANGE}")
        elif file_info.status == "processed":
            text.append(" ✓", style="bold green")
        elif file_info.status == "error":
            text.append(" ✗", style="bold red")

        # Add size for files
        if not file_info.is_dir and file_info.size > 0:
            size_str = decimal(file_info.size)
            text.append(f" ({size_str})", style="dim grey50")

        return text

    def _should_show_file(self, path: Path) -> bool:
        """Determine if a file should be shown in the tree."""
        # Skip hidden files if not showing hidden
        if not self.show_hidden and path.name.startswith("."):
            return False

        # Apply custom filter
        if not self.file_filter(path):
            return False

        return True

    def _build_tree_recursive(
        self, tree: Tree, directory: Path, current_depth: int = 0
    ):
        """Recursively build the file tree."""
        if current_depth >= self.max_depth:
            return

        try:
            # Get and sort directory contents
            items = []
            for item in directory.iterdir():
                if self._should_show_file(item):
                    items.append(item)

            # Sort: directories first, then files, both alphabetically
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

            for item in items:
                file_info = self._get_file_info(item)
                file_text = self._create_file_text(file_info)

                if item.is_dir():
                    # Add directory node
                    subtree = tree.add(file_text)
                    # Recursively add contents
                    self._build_tree_recursive(subtree, item, current_depth + 1)
                else:
                    # Add file node
                    tree.add(file_text)

        except (OSError, PermissionError):
            # Add error node for inaccessible directories
            error_text = Text()
            error_text.append("❌ ", style="red")
            error_text.append("Permission denied", style="red")
            tree.add(error_text)

    def create_tree(self, title: Optional[str] = None) -> Tree:
        """
        Create the file tree display.

        Args:
            title: Optional title for the tree root

        Returns:
            Rich Tree object ready for display
        """
        if title is None:
            title = f"📁 {self.root_path.name}"

        # Create root tree with orange styling
        root_text = Text()
        root_text.append("🦞 ", style="")
        root_text.append(title, style=f"bold {LobsterTheme.PRIMARY_ORANGE}")

        tree = Tree(root_text, **LobsterTheme.get_tree_style())

        # Build the tree structure
        self._build_tree_recursive(tree, self.root_path)

        return tree

    def create_workspace_tree(self, workspace_path: Path) -> Tree:
        """
        Create a specialized tree for Lobster workspace.

        Args:
            workspace_path: Path to the .lobster_workspace directory

        Returns:
            Rich Tree object for workspace navigation
        """
        workspace_tree = Tree(
            Text("🦞 Lobster Workspace", style=f"bold {LobsterTheme.PRIMARY_ORANGE}"),
            **LobsterTheme.get_tree_style(),
        )

        # Define workspace structure
        workspace_dirs = {
            "data": "📊 Data Files",
            "exports": "📤 Exports",
            "plots": "📈 Visualizations",
            "cache": "💾 Cache",
            "logs": "📋 Logs",
        }

        for dir_name, display_name in workspace_dirs.items():
            dir_path = workspace_path / dir_name
            if dir_path.exists():
                # Create subtree for each workspace directory
                dir_text = Text()
                dir_text.append(
                    display_name, style=f"bold {LobsterTheme.PRIMARY_ORANGE}"
                )

                subtree = workspace_tree.add(dir_text)

                # Add contents of each directory
                try:
                    items = sorted(
                        dir_path.iterdir(),
                        key=lambda x: (not x.is_dir(), x.name.lower()),
                    )
                    for item in items[:10]:  # Limit to first 10 items
                        if self._should_show_file(item):
                            file_info = self._get_file_info(item)
                            file_text = self._create_file_text(file_info)
                            subtree.add(file_text)

                    if len(list(dir_path.iterdir())) > 10:
                        more_text = Text(
                            f"... and {len(list(dir_path.iterdir())) - 10} more",
                            style="dim grey50",
                        )
                        subtree.add(more_text)

                except (OSError, PermissionError):
                    error_text = Text("Permission denied", style="red")
                    subtree.add(error_text)

        return workspace_tree

    def set_file_status(self, file_path: Path, status: str):
        """Set status for a specific file."""
        self.file_statuses[str(file_path)] = status

    def mark_active(self, file_path: Path):
        """Mark a file as currently active."""
        self.active_files.add(str(file_path))
        self.set_file_status(file_path, "active")

    def mark_processed(self, file_path: Path):
        """Mark a file as processed."""
        self.processed_files.add(str(file_path))
        self.set_file_status(file_path, "processed")

    def clear_active(self):
        """Clear all active file markers."""
        self.active_files.clear()
        # Reset statuses for previously active files
        for path, status in self.file_statuses.items():
            if status == "active":
                self.file_statuses[path] = "normal"

    def create_compact_tree(self, max_items: int = 20) -> Tree:
        """
        Create a compact tree view for limited space.

        Args:
            max_items: Maximum number of items to show

        Returns:
            Compact Rich Tree object
        """
        title_text = Text()
        title_text.append("📁 ", style="")
        title_text.append(
            f"{self.root_path.name} (compact)",
            style=f"bold {LobsterTheme.PRIMARY_ORANGE}",
        )

        tree = Tree(title_text, **LobsterTheme.get_tree_style())

        try:
            # Get all files and directories, prioritizing data files
            all_items = []
            for item in self.root_path.rglob("*"):
                if self._should_show_file(item) and item.is_file():
                    file_info = self._get_file_info(item)
                    all_items.append((item, file_info))

            # Sort by importance: data files first, then by type
            all_items.sort(
                key=lambda x: (
                    x[1].file_type not in ["data", "bioinformatics"],
                    x[1].file_type,
                    x[0].name.lower(),
                )
            )

            # Add top items
            for item, file_info in all_items[:max_items]:
                # Show relative path for context
                rel_path = item.relative_to(self.root_path)
                file_text = Text()

                # Add icon and styling
                if file_info.has_data:
                    file_text.append("📊 ", style="")
                    file_text.append(
                        str(rel_path), style=f"{LobsterTheme.PRIMARY_ORANGE}"
                    )
                else:
                    file_text.append("📄 ", style="")
                    file_text.append(str(rel_path), style="white")

                tree.add(file_text)

            if len(all_items) > max_items:
                more_text = Text(
                    f"... and {len(all_items) - max_items} more files",
                    style="dim grey50",
                )
                tree.add(more_text)

        except (OSError, PermissionError):
            error_text = Text("Error reading directory", style="red")
            tree.add(error_text)

        return tree


def create_file_tree(
    root_path: Path,
    title: Optional[str] = None,
    show_hidden: bool = False,
    max_depth: int = 3,
) -> Tree:
    """
    Quick function to create a file tree.

    Args:
        root_path: Root directory path
        title: Optional tree title
        show_hidden: Show hidden files
        max_depth: Maximum tree depth

    Returns:
        Rich Tree object
    """
    tree_component = LobsterFileTree(
        root_path=root_path, show_hidden=show_hidden, max_depth=max_depth
    )
    return tree_component.create_tree(title)


def create_workspace_tree(workspace_path: Path) -> Tree:
    """
    Quick function to create a workspace tree.

    Args:
        workspace_path: Path to workspace directory

    Returns:
        Rich Tree object for workspace
    """
    tree_component = LobsterFileTree(workspace_path)
    return tree_component.create_workspace_tree(workspace_path)
