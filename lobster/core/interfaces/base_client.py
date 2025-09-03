"""
Base Client Interface for Lobster AI System.

This module defines the abstract base class that all Lobster clients must implement,
ensuring consistency between local and cloud implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator


class BaseClient(ABC):
    """
    Abstract base class defining the interface for all Lobster client implementations.
    
    This ensures that both local (AgentClient) and cloud (CloudLobsterClient) 
    implementations provide the same interface to the CLI and other components.
    """
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the client with necessary configuration."""
        pass
    
    @abstractmethod
    def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]:
        """
        Process a user query through the system.
        
        Args:
            user_input: The user's question or request
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing:
                - success: bool
                - response: str
                - error: Optional[str]
                - session_id: str
                - has_data: bool
                - plots: List[Dict[str, Any]]
                - duration: float (optional)
                - last_agent: Optional[str] (optional)
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the client/system.
        
        Returns:
            Dictionary containing status information including:
                - session_id: str
                - message_count: int (for local) or status: str (for cloud)
                - has_data: bool
                - workspace: str
                - data_summary: Optional[Dict] (if data is loaded)
        """
        pass
    
    @abstractmethod
    def list_workspace_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
        """
        List files in the workspace.
        
        Args:
            pattern: Glob pattern for filtering files
            
        Returns:
            List of dictionaries containing file information:
                - name: str
                - path: str
                - size: int
                - modified: str (ISO format timestamp)
        """
        pass
    
    @abstractmethod
    def read_file(self, filename: str) -> Optional[str]:
        """
        Read a file from the workspace.
        
        Args:
            filename: Name or path of the file to read
            
        Returns:
            File contents as string, or None if not found
        """
        pass
    
    @abstractmethod
    def write_file(self, filename: str, content: str) -> bool:
        """
        Write a file to the workspace.
        
        Args:
            filename: Name or path of the file to write
            content: Content to write to the file
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of messages with role and content:
                - role: str ('user', 'assistant', or 'system')
                - content: str
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the conversation state."""
        pass
    
    @abstractmethod
    def export_session(self, export_path: Optional[Path] = None) -> Path:
        """
        Export the current session data.
        
        Args:
            export_path: Optional path for the export file
            
        Returns:
            Path to the exported file
        """
        pass
    
    # Optional methods that implementations may override
    def get_usage(self) -> Dict[str, Any]:
        """
        Get usage statistics (primarily for cloud clients).
        
        Returns:
            Dictionary with usage information or error
        """
        return {"error": "Usage tracking not available for this client type", "success": False}
    
    def list_models(self) -> Dict[str, Any]:
        """
        List available models (primarily for cloud clients).
        
        Returns:
            Dictionary with model list or error
        """
        return {"error": "Model listing not available for this client type", "success": False}
