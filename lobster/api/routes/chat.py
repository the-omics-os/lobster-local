"""
Lobster AI - Chat Routes
Chat and conversation endpoints for agent interaction.
"""

from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse

from lobster.api.models import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    MessageRole,
    BaseResponse
)
from lobster.api.session_manager import SessionManager
from lobster.core.api_client import APIAgentClient
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Create router instance
router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get the session manager from app state."""
    return request.app.state.session_manager


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def send_message(
    session_id: UUID,
    chat_request: ChatRequest,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Send a message to the agent and get a response.
    
    This endpoint processes user messages through the multi-agent system
    and returns the agent's response. If streaming is enabled, the response
    will be sent via WebSocket connections.
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Get or create API agent client for this session
        if not hasattr(session, '_api_client') or session._api_client is None:
            session._api_client = APIAgentClient(
                session_id=session_id,
                session_manager=session_manager,
                data_manager=session.data_manager if hasattr(session, 'data_manager') else None,
                workspace_path=session.workspace_path
            )
        
        api_client = session._api_client
        
        # Process the message through the agent system
        result = await api_client.query(
            user_input=chat_request.message,
            stream=chat_request.stream
        )
        
        # Update session activity
        session.update_activity()
        session.message_count += 1
        
        # Create chat message for response
        if result.get("success", True):
            response_message = ChatMessage(
                id=str(uuid4()),
                role=MessageRole.ASSISTANT,
                content=result.get("response", "No response generated"),
                timestamp=datetime.utcnow().isoformat(),
                metadata={
                    "duration": result.get("duration"),
                    "events_count": result.get("events_count"),
                    "has_data": result.get("has_data", False),
                    "session_id": str(session_id)
                },
                plots=result.get("plots", [])
            )
            
            # Create successful response
            chat_response = ChatResponse(
                success=True,
                message="Message processed successfully",
                timestamp=datetime.utcnow().isoformat(),
                chat_message=response_message,
                conversation_id=session_id,
                plots=result.get("plots", []),
                data_updated=result.get("has_data", False)
            )
            
            logger.info(f"Chat message processed for session {session_id}")
            return chat_response
            
        else:
            # Handle error case
            error_message = ChatMessage(
                id=str(uuid4()),
                role=MessageRole.ASSISTANT,
                content=result.get("response", "I encountered an error processing your request"),
                timestamp=datetime.utcnow().isoformat(),
                metadata={
                    "error": result.get("error"),
                    "session_id": str(session_id)
                }
            )
            
            chat_response = ChatResponse(
                success=False,
                message=f"Error processing message: {result.get('error', 'Unknown error')}",
                timestamp=datetime.utcnow().isoformat(),
                chat_message=error_message,
                conversation_id=session_id
            )
            
            return chat_response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat message: {str(e)}"
        )


@router.get("/sessions/{session_id}/chat/history")
async def get_chat_history(
    session_id: UUID,
    limit: Optional[int] = 50,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get the conversation history for a session.
    
    Returns the chat history with optional limit on number of messages.
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Get conversation history
        if hasattr(session, '_api_client') and session._api_client:
            history = session._api_client.get_conversation_history(limit=limit)
        else:
            # Return empty history if no agent client exists yet
            history = []
        
        # Convert to ChatMessage objects
        chat_messages = []
        for msg in history:
            chat_message = ChatMessage(
                id=msg.get("id", str(uuid4())),
                role=MessageRole(msg["role"]),
                content=msg["content"],
                timestamp=msg.get("timestamp", datetime.utcnow().isoformat()),
                metadata=msg.get("metadata", {"session_id": str(session_id)})
            )
            chat_messages.append(chat_message)
        
        return {
            "success": True,
            "message": f"Retrieved {len(chat_messages)} messages",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": str(session_id),
            "messages": chat_messages,
            "total_messages": len(chat_messages)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve chat history: {str(e)}"
        )


@router.delete("/sessions/{session_id}/chat/history", response_model=BaseResponse)
async def clear_chat_history(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Clear the conversation history for a session.
    
    Resets the conversation while keeping the session and workspace intact.
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Clear conversation history
        if hasattr(session, '_api_client') and session._api_client:
            session._api_client.reset_conversation()
        
        # Reset message count
        session.message_count = 0
        session.update_activity()
        
        logger.info(f"Cleared chat history for session {session_id}")
        
        return BaseResponse(
            success=True,
            message="Chat history cleared successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing chat history for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear chat history: {str(e)}"
        )


@router.get("/sessions/{session_id}/status")
async def get_session_status(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get detailed status information for a session.
    
    Returns session info, agent status, data status, and workspace information.
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Get basic session info
        session_info = session.to_session_info()
        
        # Get agent client status if available
        agent_status = None
        if hasattr(session, '_api_client') and session._api_client:
            agent_status = session._api_client.get_status()
        
        # Get workspace file listing
        workspace_files = []
        if hasattr(session, '_api_client') and session._api_client:
            workspace_files = session._api_client.list_workspace_files()
        
        return {
            "success": True,
            "message": "Session status retrieved successfully",
            "session": session_info.dict(),
            "agent_status": agent_status,
            "workspace_files": workspace_files,
            "websocket_connections": len(session.websocket_connections)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status for {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session status: {str(e)}"
        )


@router.post("/sessions/{session_id}/geo-download")
async def download_geo_dataset(
    session_id: UUID,
    geo_request: dict,  # Simple dict for now, can be enhanced with proper model
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Download a GEO dataset for analysis in the session.
    
    This endpoint initiates the download of a GEO dataset and loads it
    into the session's data manager for analysis.
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Get GEO ID from request
        geo_id = geo_request.get("geo_id")
        if not geo_id:
            raise HTTPException(
                status_code=400,
                detail="GEO ID is required"
            )
        
        # Validate GEO ID format
        if not geo_id.startswith("GSE"):
            raise HTTPException(
                status_code=400,
                detail="Invalid GEO ID format. Must start with 'GSE'"
            )
        
        # Get or create API agent client
        if not hasattr(session, '_api_client') or session._api_client is None:
            session._api_client = APIAgentClient(
                session_id=session_id,
                session_manager=session_manager,
                workspace_path=session.workspace_path
            )
        
        # Initiate GEO download
        result = await session._api_client.download_geo_dataset(geo_id)
        
        # Update session activity
        session.update_activity()
        
        return {
            "success": result.get("success", True),
            "message": result.get("message", f"GEO dataset {geo_id} download initiated"),
            "geo_id": geo_id,
            "session_id": str(session_id),
            "details": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading GEO dataset for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download GEO dataset: {str(e)}"
        )
