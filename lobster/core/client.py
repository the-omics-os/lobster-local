"""
Clean Agent Client Interface for LangGraph Multi-Agent System.
Provides a simple, extensible interface for both CLI and future UI implementations.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Callable
from datetime import datetime
import json

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langfuse.langchain import CallbackHandler as LangfuseCallback

from .data_manager import DataManager
from ..agents.graph import create_bioinformatics_graph


class AgentClient:
    def __init__(
        self,
        data_manager: Optional[DataManager] = None,
        session_id: str = None,
        enable_reasoning: bool = True,
        enable_langfuse: bool = True,
        workspace_path: Optional[Path] = None,
        custom_callbacks: Optional[List] = None  # Changed from List[Callable]
    ):
        """
        Initialize the agent client.
        
        Args:
            data_manager: Data manager instance (creates new if None)
            session_id: Unique session identifier
            enable_reasoning: Show agent reasoning/thinking process
            enable_langfuse: Enable Langfuse debugging callback
            workspace_path: Path to workspace for file operations
            custom_callbacks: Additional callback handlers
        """
        # Set up session
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_reasoning = enable_reasoning
        
        # Set up workspace
        self.workspace_path = workspace_path or Path.cwd()
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data manager
        self.data_manager = data_manager or DataManager()
        
        # Set up callbacks
        self.callbacks = []
        if enable_langfuse and os.getenv("LANGFUSE_PUBLIC_KEY"):
            from langfuse.langchain import CallbackHandler as LangfuseCallback
            self.callbacks.append(LangfuseCallback())
        if custom_callbacks:
            self.callbacks.extend(custom_callbacks)
        
        self.checkpointer = MemorySaver()
        # Initialize graph - pass all callbacks
        self.graph = create_bioinformatics_graph(
            data_manager=self.data_manager,
            checkpointer=self.checkpointer,
            callback_handler=self.callbacks  # Pass the list of callbacks
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
        """Extract the final response from graph events."""
        if not events:
            return "No response generated."
        
        # The last event should contain the final state
        # Structure: [{'node_name': {'messages': [HumanMessage, AIMessage, ...]}}]
        last_event = events[-1]
        
        # Extract the node output (could be supervisor, transcriptomics_expert_agent, etc.)
        for node_name, node_output in last_event.items():
            if isinstance(node_output, dict) and "messages" in node_output:
                messages = node_output["messages"]
                
                # Find the last AI message in the conversation
                for msg in reversed(messages):
                    # Check if it's an AIMessage and has content
                    if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content and msg.content.strip():
                        return msg.content
                    # Also handle cases where msg might have different attributes
                    elif hasattr(msg, 'content') and msg.content and not isinstance(msg, HumanMessage):
                        # This catches any message with content that isn't a HumanMessage
                        return msg.content
        
        # If no AI message found in the last event, check all events
        for event in reversed(events):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict) and "messages" in node_output:
                    messages = node_output["messages"]
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content and msg.content.strip():
                            return msg.content
        
        return "Analysis completed."
    
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
        """Read a file from the workspace."""
        file_path = self.workspace_path / filename
        if file_path.exists() and file_path.is_file():
            try:
                return file_path.read_text()
            except Exception as e:
                return f"Error reading file: {e}"
        return None
    
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
        export_path = export_path or self.workspace_path / f"session_{self.session_id}.json"
        
        session_data = {
            "session_id": self.session_id,
            "metadata": self.metadata,
            "conversation": self.get_conversation_history(),
            "status": self.get_status(),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(export_path, 'w') as f:
            json.dumps(session_data, f, indent=2, default=str)
        
        return export_path
