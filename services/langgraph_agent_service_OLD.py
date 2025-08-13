"""
LangGraph Agent Service for managing multi-agent bioinformatics interactions.

Updated to use the new clean agent client architecture.
Acts as a compatibility layer for existing Streamlit integration.
"""

from typing import List, Dict, Any, Optional

from utils.logger import get_logger
from core.data_manager import DataManager
from clients.agent_client import LobsterClient

logger = get_logger(__name__)


class LangGraphAgentService:
    """
    Service for managing LangGraph multi-agent interactions.
    
    Updated to use the new clean agent client architecture.
    Provides compatibility for existing Streamlit integration.
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        callback_handler: Optional[Any] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the langgraph agent service using the new client architecture.
        
        Args:
            data_manager: DataManager instance
            callback_handler: Optional callback handler or list of handlers for streaming
            chat_history: Optional list of chat messages from session state
        """
        logger.info("Initializing LangGraph Agent Service with new client architecture")
        
        self.data_manager = data_manager
        
        # Handle callback handler(s) - can be a single handler or a list
        if callback_handler is None:
            self.callback_handlers = []
        elif isinstance(callback_handler, list):
            self.callback_handlers = callback_handler
        else:
            self.callback_handlers = [callback_handler]
        
        self.chat_history = chat_history or []
        
        # Create the new agent client
        self.agent_client = LobsterClient(
            data_manager=data_manager,
            callback_handlers=self.callback_handlers,
            session_id="streamlit_session"
        )
        
        logger.info("LangGraph Agent Service initialized with new architecture")
    
    def _sync_chat_history_with_client(self):
        """Sync chat history with the agent client."""
        try:
            # Get current conversation from client
            client_history = self.agent_client.get_conversation_history()
            
            # Update our chat history to match
            self.chat_history = client_history
            
        except Exception as e:
            logger.warning(f"Error syncing chat history: {e}")
    
    def run_agent(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Run the multi-agent system with a user query using the new client architecture.
        
        Args:
            query: User query string
            chat_history: Optional updated chat history
            
        Returns:
            str: Agent response
        """
        logger.debug(f"Running agent with query: {query[:50]}...")
        
        # Update chat history if provided
        if chat_history:
            self.chat_history = chat_history
        
        try:
            # Use the new agent client to process the query
            response = self.agent_client.run_query(query)
            
            # Sync chat history with client
            self._sync_chat_history_with_client()
            
            return response
            
        except Exception as e:
            logger.exception(f"Error running agent: {e}")
            return f"I encountered an error: {str(e)}"
    
    async def arun_agent(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Async version of run_agent using the new client architecture.
        
        Args:
            query: User query string
            chat_history: Optional updated chat history
            
        Returns:
            str: Agent response
        """
        # Update chat history if provided
        if chat_history:
            self.chat_history = chat_history
        
        try:
            # Use the async method of the agent client
            response = await self.agent_client.arun_query(query)
            
            # Sync chat history with client
            self._sync_chat_history_with_client()
            
            return response
            
        except Exception as e:
            logger.exception(f"Error running agent async: {e}")
            return f"I encountered an error: {str(e)}"
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the conversation using the new client architecture.
        
        Returns:
            Dict containing current state information
        """
        try:
            # Get state from the agent client
            client_status = self.agent_client.get_status()
            graph_state = client_status.get('graph_state', {})
            
            # Convert to the expected format for backward compatibility
            return {
                "has_research_brief": bool(graph_state.get("current_task")),
                "current_expert": graph_state.get("current_agent"),
                "analysis_complete": graph_state.get("analysis_complete", False),
                "awaiting_clarification": graph_state.get("awaiting_user_response", False),
                "session_id": client_status.get("session_id"),
                "conversation_length": client_status.get("conversation_length", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            return {}
    
    def reset_conversation(self):
        """Reset the conversation state using the new client architecture."""
        self.agent_client.reset()
        self.chat_history = []
        logger.info("Conversation reset")
