"""
Stub DataManager class for legacy compatibility during testing.
"""
from pathlib import Path
from typing import Dict, Any, Optional


class DataManager:
    """Stub DataManager class for backward compatibility."""

    def __init__(self, workspace_path: Optional[Path] = None):
        self.workspace_path = workspace_path or Path.cwd()
        self.data_dir = self.workspace_path / "data"
        self.plots_dir = self.workspace_path / "plots"
        self.exports_dir = self.workspace_path / "exports"
        self._has_data = False

    def has_data(self) -> bool:
        """Check if data manager has loaded data."""
        return self._has_data

    def set_data(self, data_path: str) -> None:
        """Set data in the manager."""
        self._has_data = True

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of loaded data."""
        return {
            "has_data": self._has_data,
            "data_path": str(self.data_dir)
        }