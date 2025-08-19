"""
Lobster AI - Session Management Routes
CRUD endpoints for managing user sessions.
"""

from uuid import UUID
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request

from lobster.api.models import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionListResponse,
    SessionInfo,
    BaseResponse
)
from lobster.api.session_manager import SessionManager, Session
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Create router instance
router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get the session manager from app state."""
    return request.app.state.session_manager


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    session_data: SessionCreate,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Create a new user session.
    
    Creates a new isolated session with its own workspace and agent instance.
    """
    try:
        session = await session_manager.create_session(
            name=session_data.name,
            description=session_data.description,
            user_id=session_data.user_id,
            timeout_minutes=session_data.timeout_minutes or 30
        )
        
        response = SessionResponse(
            success=True,
            message="Session created successfully",
            session=session.to_session_info()
        )
        
        logger.info(f"Created session {session.session_id} via API")
        return response
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create session: {str(e)}"
        )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get session information by ID.
    
    Returns session details including status, workspace path, and activity info.
    """
    try:
        session = session_manager.get_session(session_id)
        
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        response = SessionResponse(
            success=True,
            message="Session retrieved successfully",
            session=session.to_session_info()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session: {str(e)}"
        )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user_id: Optional[str] = None,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    List all active sessions.
    
    Optionally filter by user_id. Returns session information for all active sessions.
    """
    try:
        sessions = session_manager.list_sessions(user_id=user_id)
        
        session_infos = [session.to_session_info() for session in sessions]
        
        response = SessionListResponse(
            success=True,
            message=f"Retrieved {len(session_infos)} sessions",
            sessions=session_infos,
            total=len(session_infos)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sessions: {str(e)}"
        )


@router.put("/sessions/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: UUID,
    session_data: SessionUpdate,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Update session information.
    
    Allows updating session name, description, and timeout settings.
    """
    try:
        session = session_manager.get_session(session_id)
        
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Update session fields if provided
        if session_data.name is not None:
            session.name = session_data.name
        
        if session_data.description is not None:
            session.description = session_data.description
        
        if session_data.timeout_minutes is not None:
            session.timeout_minutes = session_data.timeout_minutes
        
        # Update activity timestamp
        session.update_activity()
        
        response = SessionResponse(
            success=True,
            message="Session updated successfully",
            session=session.to_session_info()
        )
        
        logger.info(f"Updated session {session_id} via API")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update session: {str(e)}"
        )


@router.delete("/sessions/{session_id}", response_model=BaseResponse)
async def delete_session(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Delete a session and clean up its resources.
    
    Removes the session, closes WebSocket connections, and cleans up workspace.
    """
    try:
        success = await session_manager.delete_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
        
        response = BaseResponse(
            success=True,
            message="Session deleted successfully"
        )
        
        logger.info(f"Deleted session {session_id} via API")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(e)}"
        )


@router.post("/sessions/{session_id}/extend", response_model=SessionResponse)
async def extend_session(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Extend session timeout by updating activity timestamp.
    
    Resets the session timeout counter by updating the last activity time.
    """
    try:
        session = session_manager.get_session(session_id)
        
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Update activity (this extends the timeout)
        session.update_activity()
        
        response = SessionResponse(
            success=True,
            message="Session extended successfully",
            session=session.to_session_info()
        )
        
        logger.debug(f"Extended session {session_id} via API")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extend session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extend session: {str(e)}"
        )


@router.get("/sessions/stats")
async def get_session_stats(
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get session statistics.
    
    Returns count of total, active, and expired sessions.
    """
    try:
        stats = session_manager.get_stats()
        
        return {
            "success": True,
            "message": "Session statistics retrieved",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get session stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session statistics: {str(e)}"
        )


@router.post("/sessions/cleanup")
async def cleanup_expired_sessions(
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Manually trigger cleanup of expired sessions.
    
    Removes all expired sessions and their resources.
    """
    try:
        cleaned_count = await session_manager.cleanup_expired_sessions()
        
        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} expired sessions",
            "cleaned_count": cleaned_count
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup expired sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup expired sessions: {str(e)}"
        )


@router.get("/sessions/{session_id}/workspace/files")
async def get_workspace_files(
    session_id: UUID,
    file_type: Optional[str] = None,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get files in the session workspace.
    """
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Get workspace files
        files = []
        if hasattr(session, '_api_client') and session._api_client:
            try:
                workspace_files = session._api_client.list_workspace_files()
                
                # Filter by file type if specified
                if file_type:
                    workspace_files = [f for f in workspace_files if f.get('file_type') == file_type]
                
                # Convert to FileInfo format
                for file_info in workspace_files:
                    files.append({
                        "name": file_info.get("name", ""),
                        "path": file_info.get("path", ""),
                        "size_bytes": file_info.get("size", 0),
                        "created_at": file_info.get("created_at", "2025-01-01T00:00:00"),
                        "modified_at": file_info.get("modified_at", "2025-01-01T00:00:00"),
                        "file_type": file_info.get("file_type", "unknown")
                    })
            except Exception as e:
                logger.warning(f"Failed to get workspace files from api_client: {e}")
                # Return empty list if workspace files can't be retrieved
                files = []
        
        return {
            "success": True,
            "message": f"Retrieved {len(files)} workspace files",
            "timestamp": "2025-01-01T00:00:00",
            "files": files,
            "total": len(files),
            "total_size_bytes": sum(f["size_bytes"] for f in files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workspace files for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve workspace files: {str(e)}"
        )
