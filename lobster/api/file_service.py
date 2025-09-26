"""
Stub file service for testing purposes only.
"""
from typing import List, Dict, Any
from uuid import UUID


class FileService:
    """Stub file service class."""

    @staticmethod
    def get_session_files_metadata(
        session_id: UUID,
        workspace_base_path: str,
        directory: str = None
    ) -> List[Dict[str, Any]]:
        """Stub method for getting session file metadata."""
        return [
            {
                "name": "test_file.csv",
                "path": "/workspace/test_file.csv",
                "relative_path": "test_file.csv",
                "size_bytes": 1024,
                "modified_at": "2023-01-01T00:00:00Z",
                "directory": "data",
                "file_type": "csv",
                "file_format": "delimited",
                "is_data_file": True,
                "row_count": 100,
                "column_count": 10,
                "has_header": True,
                "delimiter": ",",
                "compressed": False,
                "format_info": {"encoding": "utf-8"},
                "analysis_error": None
            }
        ]