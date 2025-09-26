"""
Lobster Cloud Client - Connects to Lobster Cloud API
"""

import os
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from lobster.core.interfaces.base_client import BaseClient


class CloudLobsterClient(BaseClient):
    """
    Cloud client for Lobster AI that connects to the remote API
    """
    
    def __init__(
        self, 
        api_key: str, 
        endpoint: Optional[str] = None,
        timeout: int = 30,
        workspace_path: Optional[Path] = None
    ):
        """
        Initialize the cloud client
        
        Args:
            api_key: API key for authentication
            endpoint: API endpoint URL (defaults to production)
            timeout: Request timeout in seconds
            workspace_path: Optional local workspace path for hybrid operations
        """
        self.api_key = api_key
        self.endpoint = endpoint or os.getenv(
            'LOBSTER_ENDPOINT', 
            'https://api.lobster.omics-os.com'
        )
        self.timeout = timeout
        self.workspace_path = workspace_path or Path.cwd()
        self.session_id = f"cloud_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create session with headers
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Lobster-Cloud-Client/2.0.0"
        })
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []
    
    def query(self, user_input: str, stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Process a user query via the cloud API
        
        Args:
            user_input: The user's question or request
            stream: Whether to stream the response (not implemented for cloud yet)
            **kwargs: Additional options (workspace, reasoning, etc.)
            
        Returns:
            Dict containing the response and metadata in format compatible with AgentClient
        """
        start_time = datetime.now()
        
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        try:
            payload = {
                "query": user_input,
                "session_id": self.session_id,
                "options": kwargs
            }
            
            response = self.session.post(
                f"{self.endpoint}/query",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            cloud_response = response.json()
            
            # Extract response content
            response_content = cloud_response.get("response", "")
            
            # Add assistant response to conversation history
            if response_content:
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response_content
                })
            
            # Standardize response format to match AgentClient
            standardized_response = {
                "success": cloud_response.get("success", True),
                "response": response_content,
                "session_id": self.session_id,
                "has_data": cloud_response.get("has_data", False),
                "plots": cloud_response.get("plots", []),
                "duration": (datetime.now() - start_time).total_seconds(),
                "last_agent": cloud_response.get("last_agent", "cloud_supervisor")
            }
            
            # Add error field if present
            if "error" in cloud_response:
                standardized_response["error"] = cloud_response["error"]
            
            return standardized_response
            
        except requests.exceptions.Timeout:
            error_response = {
                "error": "Request timed out. The cloud service may be experiencing high load.",
                "success": False,
                "response": "I apologize, but the request timed out. Please try again.",
                "session_id": self.session_id,
                "has_data": False,
                "plots": [],
                "duration": (datetime.now() - start_time).total_seconds()
            }
            return error_response
            
        except requests.exceptions.ConnectionError:
            error_response = {
                "error": "Could not connect to Lobster Cloud. Please check your internet connection.",
                "success": False,
                "response": "I couldn't connect to Lobster Cloud. Please check your internet connection and try again.",
                "session_id": self.session_id,
                "has_data": False,
                "plots": [],
                "duration": (datetime.now() - start_time).total_seconds()
            }
            return error_response
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                error_msg = "Invalid API key. Please check your LOBSTER_CLOUD_KEY."
            elif response.status_code == 429:
                error_msg = "Rate limit exceeded. Please try again later."
            else:
                error_msg = f"HTTP error: {e}"
                
            error_response = {
                "error": error_msg,
                "success": False,
                "response": f"I encountered an error: {error_msg}",
                "session_id": self.session_id,
                "has_data": False,
                "plots": [],
                "duration": (datetime.now() - start_time).total_seconds()
            }
            return error_response
            
        except Exception as e:
            error_response = {
                "error": f"Unexpected error: {e}",
                "success": False,
                "response": f"I encountered an unexpected error: {e}",
                "session_id": self.session_id,
                "has_data": False,
                "plots": [],
                "duration": (datetime.now() - start_time).total_seconds()
            }
            return error_response
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the cloud service
        
        Returns:
            Dict containing status information
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/status",
                timeout=10  # Shorter timeout for status checks
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Could not get status: {e}",
                "success": False
            }
    
    def get_usage(self) -> Dict[str, Any]:
        """
        Get usage statistics for the current API key
        
        Returns:
            Dict containing usage information
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/usage",
                timeout=10
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {
                "error": f"Could not get usage info: {e}",
                "success": False
            }
    
    def list_models(self) -> Dict[str, Any]:
        """
        List available models in the cloud
        
        Returns:
            Dict containing available models
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/models",
                timeout=10
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {
                "error": f"Could not list models: {e}",
                "success": False
            }
    
    def list_workspace_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
        """
        List files in the cloud workspace.
        
        Args:
            pattern: Glob pattern for filtering files
            
        Returns:
            List of dictionaries containing file information
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/workspace/files",
                params={"pattern": pattern},
                timeout=10
            )
            response.raise_for_status()
            
            return response.json().get("files", [])
            
        except Exception:
            # Fallback to empty list on error
            return []
    
    def read_file(self, filename: str) -> Optional[str]:
        """
        Read a file from the cloud workspace.
        
        Args:
            filename: Name or path of the file to read
            
        Returns:
            File contents as string, or None if not found
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/workspace/files/{filename}",
                timeout=30
            )
            response.raise_for_status()
            
            return response.json().get("content")
            
        except requests.exceptions.HTTPError as e:
            if hasattr(e.response, 'status_code') and e.response.status_code == 404:
                return None
            return f"Error reading file: {e}"
        except Exception as e:
            return f"Error reading file: {e}"
    
    def write_file(self, filename: str, content: str) -> bool:
        """
        Write a file to the cloud workspace.
        
        Args:
            filename: Name or path of the file to write
            content: Content to write to the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.put(
                f"{self.endpoint}/workspace/files/{filename}",
                json={"content": content},
                timeout=30
            )
            response.raise_for_status()
            
            return response.json().get("success", False)
            
        except Exception:
            return False
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of messages with role and content
        """
        # For cloud client, we maintain local conversation history
        # This could be enhanced to sync with cloud storage
        return self.conversation_history.copy()
    
    def reset(self) -> None:
        """Reset the conversation state."""
        self.conversation_history.clear()
        # Optionally notify cloud service of reset
        try:
            self.session.post(
                f"{self.endpoint}/session/reset",
                json={"session_id": self.session_id},
                timeout=10
            )
        except:
            # Silently fail - local reset is sufficient
            pass
    
    def export_session(self, export_path: Optional[Path] = None) -> Path:
        """
        Export the current session data.
        
        Args:
            export_path: Optional path for the export file
            
        Returns:
            Path to the exported file
        """
        # Default export path
        if export_path is None:
            export_path = self.workspace_path / f"{self.session_id}_export.json"
        
        # Prepare export data
        export_data = {
            "session_id": self.session_id,
            "endpoint": self.endpoint,
            "conversation_history": self.conversation_history,
            "exported_at": datetime.now().isoformat(),
            "client_type": "cloud"
        }
        
        # Try to get additional data from cloud
        try:
            response = self.session.get(
                f"{self.endpoint}/session/export",
                params={"session_id": self.session_id},
                timeout=30
            )
            if response.status_code == 200:
                cloud_data = response.json()
                export_data.update(cloud_data)
        except:
            # Continue with local data only
            pass
        
        # Write export file
        import json
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return export_path
