"""
Lobster AI - API AgentClient
Enhanced AgentClient specifically designed for API usage with WebSocket streaming.
"""

from typing import Any, Dict, List, Optional, AsyncGenerator
from datetime import datetime
from pathlib import Path
from uuid import UUID
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from lobster.core.client import AgentClient
from lobster.core.websocket_callback import APICallbackManager
from lobster.core.data_manager import DataManager
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class APIAgentClient:
    """
    API-enhanced AgentClient with WebSocket streaming and session management.
    
    This class extends the functionality of the base AgentClient to work
    seamlessly with the FastAPI web service, providing real-time streaming
    and session-based operation.
    """
    
    def __init__(
        self,
        session_id: UUID,
        session_manager,
        data_manager: Optional[DataManager] = None,
        workspace_path: Optional[Path] = None
    ):
        """
        Initialize the API AgentClient.
        
        Args:
            session_id: UUID of the session this client belongs to
            session_manager: Reference to the session manager for WebSocket broadcasting
            data_manager: Data manager instance (creates new if None)
            workspace_path: Path to workspace for file operations
        """
        self.session_id = session_id
        self.session_manager = session_manager
        
        # Set up workspace
        self.workspace_path = workspace_path or Path(f"workspaces/{session_id}")
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data manager with session workspace
        self.data_manager = data_manager or DataManager(workspace_path=self.workspace_path)
        
        # Set up API callbacks for WebSocket streaming
        self.callback_manager = APICallbackManager(session_id, session_manager)
        
        # Initialize the base AgentClient with API callbacks
        self.agent_client = AgentClient(
            data_manager=self.data_manager,
            session_id=str(session_id),
            workspace_path=self.workspace_path,
            custom_callbacks=self.callback_manager.get_callbacks()
        )
        
        logger.info(f"Initialized API AgentClient for session {session_id}")
    
    async def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]:
        """
        Process a user query through the agent system with API enhancements.
        
        Args:
            user_input: User's input text
            stream: Whether to stream the response (always True for API)
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Send progress update via WebSocket
            await self.callback_manager.send_progress_update(
                "Starting analysis...", 
                {"query_length": len(user_input)}
            )
            
            # Process the query using the base client
            if stream:
                return await self._stream_query(user_input)
            else:
                return await self._run_query(user_input)
                
        except Exception as e:
            logger.error(f"Error in API query processing: {e}", exc_info=True)
            
            # Send error via WebSocket
            from lobster.api.websocket import send_error
            await send_error(self.session_id, f"Query processing failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error while processing your request: {str(e)}",
                "session_id": str(self.session_id)
            }
    
    async def _run_query(self, user_input: str) -> Dict[str, Any]:
        """Process a query and return the complete response."""
        # Run the query in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self.agent_client.query, 
            user_input, 
            False  # stream=False
        )
        
        # Check if new data was loaded and notify via WebSocket
        if result.get("has_data") and self.data_manager.has_data():
            await self._notify_data_updates()
        
        # Check for new plots and notify via WebSocket
        if result.get("plots"):
            await self._notify_plot_updates(result["plots"])
        
        return result
    
    async def _stream_query(self, user_input: str) -> Dict[str, Any]:
        """Process a query with streaming response."""
        # Note: The streaming is handled by the WebSocket callbacks
        # This method processes the query and lets callbacks handle streaming
        return await self._run_query(user_input)
    
    async def _notify_data_updates(self):
        """Notify about data updates via WebSocket."""
        try:
            if self.data_manager.has_data():
                data_summary = self.data_manager.get_data_summary()
                dataset_info = {
                    "has_data": True,
                    "summary": data_summary,
                    "workspace_path": str(self.workspace_path)
                }
                await self.callback_manager.send_data_update(dataset_info)
        except Exception as e:
            logger.error(f"Error notifying data updates: {e}")
    
    async def _notify_plot_updates(self, plots: List[str]):
        """Notify about new plots via WebSocket."""
        try:
            for plot_path in plots:
                plot_info = {
                    "path": plot_path,
                    "name": Path(plot_path).name,
                    "session_id": str(self.session_id),
                    "created_at": datetime.utcnow().isoformat()
                }
                await self.callback_manager.send_plot_generated(plot_info)
        except Exception as e:
            logger.error(f"Error notifying plot updates: {e}")
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """Get formatted conversation history."""
        history = self.agent_client.get_conversation_history()
        if limit:
            history = history[-limit:]
        return history
    
    def get_status(self) -> Dict[str, Any]:
        """Get current client status with API-specific information."""
        base_status = self.agent_client.get_status()
        
        # Add API-specific status information
        api_status = {
            **base_status,
            "api_session_id": str(self.session_id),
            "websocket_callbacks": len(self.callback_manager.get_callbacks()),
            "workspace_path": str(self.workspace_path),
            "data_manager_status": {
                "has_data": self.data_manager.has_data(),
                "data_dir": str(self.data_manager.data_dir),
                "plots_dir": str(self.data_manager.plots_dir),
                "exports_dir": str(self.data_manager.exports_dir)
            }
        }
        
        return api_status
    
    async def upload_file(self, file_path: str, file_content: bytes) -> Dict[str, Any]:
        """
        Handle file upload for the session.
        
        Args:
            file_path: Path where to save the file
            file_content: File content as bytes
            
        Returns:
            Dictionary with upload result
        """
        try:
            # Save file to session workspace
            full_path = self.workspace_path / "data" / Path(file_path).name
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'wb') as f:
                f.write(file_content)
            
            # Try to load the data using data manager
            try:
                # This will attempt to load the data if it's a supported format
                self.data_manager.set_data(str(full_path))
                
                # Notify about data update
                await self._notify_data_updates()
                
                return {
                    "success": True,
                    "message": f"File uploaded and loaded successfully: {full_path.name}",
                    "file_path": str(full_path),
                    "file_size": len(file_content),
                    "data_loaded": True
                }
                
            except Exception as load_error:
                # File saved but couldn't be loaded as data
                logger.warning(f"File uploaded but couldn't be loaded as data: {load_error}")
                
                return {
                    "success": True,
                    "message": f"File uploaded successfully: {full_path.name}",
                    "file_path": str(full_path),
                    "file_size": len(file_content),
                    "data_loaded": False,
                    "load_warning": str(load_error)
                }
        
        except Exception as e:
            logger.error(f"Error uploading file: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to upload file: {str(e)}"
            }
    
    async def download_geo_dataset(self, geo_id: str) -> Dict[str, Any]:
        """
        Download a GEO dataset for the session.
        
        Args:
            geo_id: GEO dataset ID (e.g., GSE123456)
            
        Returns:
            Dictionary with download result
        """
        try:
            # Send progress update
            await self.callback_manager.send_progress_update(
                f"Starting GEO download for {geo_id}..."
            )
            
            # Use a query to trigger GEO download through the agent system
            query_text = f"Please download and analyze the GEO dataset {geo_id}"
            result = await self.query(query_text, stream=True)
            
            return {
                "success": result.get("success", True),
                "geo_id": geo_id,
                "message": f"GEO dataset {geo_id} download initiated",
                "query_result": result
            }
            
        except Exception as e:
            logger.error(f"Error downloading GEO dataset {geo_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "geo_id": geo_id,
                "message": f"Failed to download GEO dataset {geo_id}: {str(e)}"
            }
    
    def list_workspace_files(self, directory: str = None) -> List[Dict[str, Any]]:
        """List files in the session workspace."""
        try:
            if directory:
                search_path = self.workspace_path / directory
            else:
                search_path = self.workspace_path
            
            files = []
            if search_path.exists():
                for file_path in search_path.rglob("*"):
                    if file_path.is_file():
                        files.append({
                            "name": file_path.name,
                            "path": str(file_path.relative_to(self.workspace_path)),
                            "full_path": str(file_path),
                            "size": file_path.stat().st_size,
                            "modified": datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            ).isoformat(),
                            "directory": str(file_path.parent.relative_to(self.workspace_path))
                        })
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing workspace files: {e}")
            return []
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.agent_client.reset()
    
    async def cleanup(self):
        """Clean up resources for this API client."""
        try:
            # Any cleanup needed for the API client
            logger.info(f"Cleaning up API AgentClient for session {self.session_id}")
        except Exception as e:
            logger.error(f"Error during API client cleanup: {e}")
