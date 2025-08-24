"""
Clean Agent Client Interface for LangGraph Multi-Agent System.
Provides a simple, extensible interface for both CLI and future UI implementations.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime
import json

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from langfuse.langchain import CallbackHandler as LangfuseCallback

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.agents.graph import create_bioinformatics_graph

# Configure logging
logger = logging.getLogger(__name__)


class AgentClient:
    def __init__(
        self,
        data_manager: Optional[DataManagerV2] = None,
        session_id: str = None,
        enable_reasoning: bool = True,
        enable_langfuse: bool = False,
        workspace_path: Optional[Path] = None,
        custom_callbacks: Optional[List] = None,
        manual_model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agent client with DataManagerV2.
        
        Args:
            data_manager: DataManagerV2 instance (creates new if None)
            session_id: Unique session identifier
            enable_reasoning: Show agent reasoning/thinking process
            enable_langfuse: Enable Langfuse debugging callback
            workspace_path: Path to workspace for file operations
            custom_callbacks: Additional callback handlers
            manual_model_params: Manual model parameter overrides
        """
        # Set up session
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_reasoning = enable_reasoning
        
        # Set up workspace
        self.workspace_path = workspace_path or Path.cwd()
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize DataManagerV2
        if data_manager is None:
            from rich.console import Console
            console = Console() if custom_callbacks else None
            self.data_manager = DataManagerV2(
                workspace_path=self.workspace_path,
                console=console
            )
            logger.info("Initialized with DataManagerV2 (modular multi-omics)")
        else:
            self.data_manager = data_manager
        
        # Set up callbacks
        self.callbacks = []
        if enable_langfuse and os.getenv("LANGFUSE_PUBLIC_KEY"):
            self.callbacks.append(LangfuseCallback())
        if custom_callbacks:
            self.callbacks.extend(custom_callbacks)
        
        self.checkpointer = InMemorySaver()
        # Initialize graph - pass all callbacks
        self.graph = create_bioinformatics_graph(
            data_manager=self.data_manager,
            checkpointer=self.checkpointer,
            callback_handler=self.callbacks,  # Pass the list of callbacks
            manual_model_params=manual_model_params  # Placeholder for future manual model params
        )        
        
        # Conversation state
        self.messages: List[BaseMessage] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "session_id": self.session_id,
            "workspace": str(self.workspace_path)
        }
    
    def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]:
        """
        Process a user query through the agent system.
        
        Args:
            user_input: User's input text
            stream: Whether to stream the response
            
        Returns:
            Dictionary with response and metadata
        """
        # Add user message
        self.messages.append(HumanMessage(content=user_input))
        
        # Prepare graph input
        graph_input = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        config = {
            "configurable": {"thread_id": self.session_id},
            "callbacks": self.callbacks
        }
        
        if stream:
            return self._stream_query(graph_input, config)
        else:
            return self._run_query(graph_input, config)
    
    def _run_query(self, graph_input: Dict, config: Dict) -> Dict[str, Any]:
        """Run a query and return the complete response."""
        try:
            # Track execution
            start_time = datetime.now()
            events = []
            
            # Execute graph
            for event in self.graph.stream(
                input=graph_input, 
                config=config,
                stream_mode='updates'
                ):
                events.append(event)
            
            # Extract final response from the last event
            final_response = self._extract_response(events)
            
            # Update messages with the final response (not the raw events)
            if final_response:
                self.messages.append(AIMessage(content=final_response))
            
            return {
                "success": True,
                "response": final_response,
                "duration": (datetime.now() - start_time).total_seconds(),
                "events_count": len(events),
                "session_id": self.session_id,
                "has_data": self.data_manager.has_data(),
                "plots": self.data_manager.get_latest_plots(5) if self.data_manager.has_data() else []
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error: {str(e)}",
                "session_id": self.session_id
            }
    
    def _stream_query(self, graph_input: Dict, config: Dict) -> Generator[Dict[str, Any], None, None]:
        """Stream query execution with intermediate results."""
        try:
            start_time = datetime.now()
            
            for event in self.graph.stream(graph_input, config):
                # Process each event
                for node_name, node_output in event.items():
                    # Extract meaningful content
                    content = self._extract_event_content(node_output)
                    
                    if content:
                        yield {
                            "type": "stream",
                            "node": node_name,
                            "content": content,
                            "timestamp": datetime.now().isoformat()
                        }
            
            # Final response
            yield {
                "type": "complete",
                "duration": (datetime.now() - start_time).total_seconds(),
                "session_id": self.session_id
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "session_id": self.session_id
            }
    
    def _extract_response(self, events: List[Dict]) -> str:
        """Extract the final response from events, expecting supervisor responses."""
        if not events:
            return "No response generated."
        
        # Process events in reverse chronological order to find the last supervisor response
        for event in reversed(events):
            # Check for supervisor key first
            if 'supervisor' not in event:
                # Log any unexpected keys
                unexpected_keys = [key for key in event.keys() if key != 'supervisor']
                if unexpected_keys:
                    logger.warning(f"Unexpected event keys found (expected 'supervisor'): {unexpected_keys}")
                continue
            
            supervisor_data = event['supervisor']
            if not isinstance(supervisor_data, dict) or 'messages' not in supervisor_data:
                continue
                
            messages = supervisor_data['messages']
            if not isinstance(messages, list):
                continue
            
            # Find the last AIMessage in the supervisor's messages
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                    content = msg.content.strip()
                    if content:
                        return content
        
        return "No response generated."
    
    def _extract_event_content(self, node_output: Dict) -> Optional[str]:
        """Extract displayable content from a node output."""
        if not isinstance(node_output, dict):
            return None
        
        # Check for messages - only return content from AI messages
        if "messages" in node_output and node_output["messages"]:
            # Look for the last AI message in this event
            for msg in reversed(node_output["messages"]):
                if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                    return msg.content
        
        # Check for other relevant fields
        for key in ["analysis_results", "next", "data_context"]:
            if key in node_output and node_output[key]:
                return f"{key}: {node_output[key]}"
        
        return None
    
    # Workspace operations
    def list_workspace_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
        """List files in the workspace."""
        files = []
        for path in self.workspace_path.glob(pattern):
            if path.is_file():
                files.append({
                    "name": path.name,
                    "path": str(path),
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                })
        return files
    
    def read_file(self, filename: str) -> Optional[str]:
        """
        Read a file from the workspace or absolute path.
        
        Args:
            filename: Either a relative filename (searched in workspace) or absolute path
            
        Returns:
            File content as string, or None if not found
        """
        file_path = Path(filename)
        
        # If it's an absolute path, try to read directly
        if file_path.is_absolute():
            if file_path.exists() and file_path.is_file():
                try:
                    return file_path.read_text()
                except Exception as e:
                    return f"Error reading file {file_path}: {e}"
            else:
                return f"File not found: {file_path}"
        
        # For relative paths, search in workspace and data directories
        search_paths = [
            self.workspace_path / filename,
            self.data_manager.data_dir / filename,
            self.data_manager.workspace_path / "plots" / filename,
            self.data_manager.exports_dir / filename,
            self.data_manager.cache_dir / filename
        ]
        
        for search_path in search_paths:
            if search_path.exists() and search_path.is_file():
                try:
                    return search_path.read_text()
                except Exception as e:
                    return f"Error reading file: {e}"
        
        return f"File not found in workspace: {filename}"
    
    def write_file(self, filename: str, content: str) -> bool:
        """Write a file to the workspace."""
        try:
            file_path = self.workspace_path / filename
            file_path.write_text(content)
            return True
        except Exception:
            return False
    
    # State management
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history."""
        history = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "system"
            
            history.append({
                "role": role,
                "content": msg.content if hasattr(msg, 'content') else str(msg)
            })
        return history
    
    def get_status(self) -> Dict[str, Any]:
        """Get current client status."""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "has_data": self.data_manager.has_data(),
            "data_summary": self.data_manager.get_data_summary() if self.data_manager.has_data() else None,
            "workspace": str(self.workspace_path),
            "reasoning_enabled": self.enable_reasoning,
            "callbacks_count": len(self.callbacks)
        }
    
    def reset(self):
        """Reset the conversation state."""
        self.messages = []
        self.metadata["reset_at"] = datetime.now().isoformat()
    
    def export_session(self, export_path: Optional[Path] = None) -> Path:
        """Export the current session data."""
        if self.data_manager.has_data():
            export_path = self.data_manager.create_data_package(
                output_dir=str(self.data_manager.exports_dir)
            )
            return Path(export_path)
        
        export_path = export_path or self.workspace_path / f"session_{self.session_id}.json"
        
        session_data = {
            "session_id": self.session_id,
            "metadata": self.metadata,
            "conversation": self.get_conversation_history(),
            "status": self.get_status(),
            "workspace_status": self.data_manager.get_workspace_status(),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(export_path, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        return export_path
