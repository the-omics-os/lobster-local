"""
Lobster AI - WebSocket Callback Handler
WebSocket-aware callback handler for streaming agent responses to API clients.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from lobster.api.models import WSEventType, WSMessage
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class WebSocketCallbackHandler(BaseCallbackHandler):
    """Callback handler that streams agent responses via WebSocket connections."""
    
    def __init__(self, session_id: UUID, session_manager=None):
        """
        Initialize the WebSocket callback handler.
        
        Args:
            session_id: UUID of the session to broadcast to
            session_manager: Reference to the session manager for broadcasting
        """
        super().__init__()
        self.session_id = session_id
        self.session_manager = session_manager
        self.current_agent = None
        
    async def _send_websocket_message(self, event_type: WSEventType, data: Dict[str, Any]):
        """Send a message via WebSocket to all connections for this session."""
        if not self.session_manager:
            return
            
        try:
            # Get the session
            session = self.session_manager.get_session(self.session_id)
            if not session:
                logger.warning(f"Session {self.session_id} not found for WebSocket message")
                return
            
            # Create WebSocket message
            ws_message = WSMessage(
                event_type=event_type,
                session_id=self.session_id,
                data=data
            )
            
            # Broadcast to all WebSocket connections for this session
            await session.broadcast_message(ws_message.dict())
            
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> None:
        """Called when LLM starts running."""
        try:
            self._schedule_websocket_message(
                WSEventType.AGENT_THINKING,
                {
                    "status": "llm_start",
                    "agent": self.current_agent or "unknown",
                    "message": "Agent is thinking..."
                }
            )
        except Exception as e:
            logger.error(f"Error in on_llm_start: {e}")
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated."""
        try:
            self._schedule_websocket_message(
                WSEventType.CHAT_STREAM,
                {
                    "token": token,
                    "agent": self.current_agent or "unknown",
                    "type": "token"
                }
            )
        except Exception as e:
            logger.error(f"Error in on_llm_new_token: {e}")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        try:
            self._schedule_websocket_message(
                WSEventType.AGENT_THINKING,
                {
                    "status": "llm_end", 
                    "agent": self.current_agent or "unknown",
                    "message": "Agent finished thinking"
                }
            )
        except Exception as e:
            logger.error(f"Error in on_llm_end: {e}")
    
    def on_chain_start(
        self, 
        serialized: Dict[str, Any], 
        inputs: Dict[str, Any], 
        **kwargs: Any
    ) -> None:
        """Called when chain/agent starts running."""
        try:
            # Extract agent name from the chain (handle None serialized)
            if serialized is not None:
                chain_name = serialized.get("name", "unknown")
            else:
                chain_name = "unknown"
            
            self.current_agent = chain_name
            
            self._schedule_websocket_message(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "agent_start",
                    "agent": chain_name,
                    "message": f"Starting {chain_name}..."
                }
            )
        except Exception as e:
            logger.error(f"Error in on_chain_start: {e}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain/agent ends running."""
        try:
            self._schedule_websocket_message(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "agent_end",
                    "agent": self.current_agent or "unknown",
                    "message": f"Completed {self.current_agent or 'unknown'}"
                }
            )
        except Exception as e:
            logger.error(f"Error in on_chain_end: {e}")
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when chain/agent encounters an error."""
        try:
            self._schedule_websocket_message(
                WSEventType.ERROR,
                {
                    "status": "error",
                    "agent": self.current_agent or "unknown", 
                    "error": str(error),
                    "message": f"Error in {self.current_agent or 'unknown'}: {str(error)}"
                }
            )
        except Exception as e:
            logger.error(f"Error in on_chain_error: {e}")
    
    def on_tool_start(
        self, 
        serialized: Dict[str, Any], 
        input_str: str, 
        **kwargs: Any
    ) -> None:
        """Called when tool starts running."""
        try:
            # Handle None serialized parameter
            if serialized is not None:
                tool_name = serialized.get("name", "unknown_tool")
            else:
                tool_name = "unknown_tool"
                
            self._schedule_websocket_message(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "tool_start",
                    "tool": tool_name,
                    "agent": self.current_agent or "unknown",
                    "message": f"Using tool: {tool_name}"
                }
            )
        except Exception as e:
            logger.error(f"Error in on_tool_start: {e}")
    
    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Called when tool ends running."""
        try:
            # Convert output to string representation safely
            try:
                if hasattr(output, 'content'):
                    # Handle ToolMessage objects
                    output_str = str(output.content)
                elif hasattr(output, '__str__'):
                    # Handle any object with string representation
                    output_str = str(output)
                else:
                    # Fallback for other types
                    output_str = repr(output)
            except Exception as convert_error:
                logger.warning(f"Error converting tool output to string: {convert_error}")
                output_str = "Tool output conversion failed"
            
            # Create preview safely
            try:
                if len(output_str) > 200:
                    output_preview = output_str[:200] + "..."
                else:
                    output_preview = output_str
            except Exception:
                output_preview = "Output preview unavailable"
            
            self._schedule_websocket_message(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "tool_end",
                    "agent": self.current_agent or "unknown",
                    "message": "Tool completed",
                    "output_preview": output_preview
                }
            )
        except Exception as e:
            logger.error(f"Error in on_tool_end: {e}")
    
    def on_text(self, text: str, **kwargs: Any) -> None:
        """Called when agent outputs text."""
        try:
            # Only send substantial text content
            if text and len(text.strip()) > 0:
                self._schedule_websocket_message(
                    WSEventType.CHAT_STREAM,
                    {
                        "content": text,
                        "agent": self.current_agent or "unknown",
                        "type": "text"
                    }
                )
        except Exception as e:
            logger.error(f"Error in on_text: {e}")
    
    def _schedule_websocket_message(self, event_type: WSEventType, data: Dict[str, Any]):
        """Schedule a WebSocket message to be sent asynchronously."""
        try:
            # Try to get current event loop
            loop = asyncio.get_running_loop()
            # Schedule the coroutine to run
            loop.create_task(self._send_websocket_message(event_type, data))
        except RuntimeError:
            # No running event loop - this is expected in sync callback context
            # Store the message to be sent later or ignore it
            logger.debug(f"No event loop available for WebSocket message: {event_type}")
        except Exception as e:
            logger.error(f"Error scheduling WebSocket message: {e}")


class APICallbackManager:
    """Manager for API-specific callbacks including WebSocket streaming."""
    
    def __init__(self, session_id: UUID, session_manager=None):
        """
        Initialize the callback manager.
        
        Args:
            session_id: Session to manage callbacks for
            session_manager: Reference to session manager for WebSocket broadcasting
        """
        self.session_id = session_id
        self.session_manager = session_manager
        self.websocket_handler = WebSocketCallbackHandler(session_id, session_manager)
        
    def get_callbacks(self) -> List[BaseCallbackHandler]:
        """Get list of callback handlers for this session."""
        callbacks = [self.websocket_handler]
        
        # Add other callbacks as needed (e.g., Langfuse if configured)
        import os
        if os.getenv("LANGFUSE_PUBLIC_KEY"):
            try:
                from langfuse.langchain import CallbackHandler as LangfuseCallback
                callbacks.append(LangfuseCallback())
            except ImportError:
                logger.warning("Langfuse callback requested but not installed")
        
        return callbacks
    
    async def send_progress_update(self, message: str, details: Optional[Dict] = None):
        """Send a custom progress update via WebSocket."""
        await self.websocket_handler._send_websocket_message(
            WSEventType.ANALYSIS_PROGRESS,
            {
                "status": "custom_update",
                "message": message,
                "details": details or {},
                "agent": "system"
            }
        )
    
    async def send_data_update(self, dataset_info: Dict):
        """Send data update notification via WebSocket."""
        await self.websocket_handler._send_websocket_message(
            WSEventType.DATA_UPDATED,
            {
                "status": "data_loaded",
                "dataset": dataset_info,
                "message": "New dataset loaded"
            }
        )
    
    async def send_plot_generated(self, plot_info: Dict):
        """Send plot generation notification via WebSocket."""
        await self.websocket_handler._send_websocket_message(
            WSEventType.PLOT_GENERATED,
            {
                "status": "plot_created",
                "plot": plot_info,
                "message": "New plot generated"
            }
        )
