"""
Lobster AI - Session Manager
Manages user sessions, agent instances, and workspace isolation.
"""

import asyncio
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Set
from uuid import UUID, uuid4
from weakref import WeakSet

from fastapi import WebSocket

from lobster.api.models import SessionInfo, SessionStatus
from lobster.core.client import AgentClient
from lobster.core.data_manager import DataManager
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class Session:
    """Individual session instance with agent client and workspace."""
    
    def __init__(
        self,
        session_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout_minutes: int = 30
    ):
        self.session_id = session_id
        self.name = name or f"Session {session_id.hex[:8]}"
        self.description = description
        self.user_id = user_id
        self.timeout_minutes = timeout_minutes
        
        # Timestamps
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Status
        self.status = SessionStatus.ACTIVE
        
        # Workspace setup
        self.workspace_path = Path(f"workspaces/{session_id}")
        self._setup_workspace()
        
        # Agent client - will be initialized lazily
        self._agent_client: Optional[AgentClient] = None
        
        # WebSocket connections for this session
        self.websocket_connections: Set[WebSocket] = set()
        
        # Conversation history
        self.message_count = 0
        self.datasets: list = []
        
        logger.info(f"Created session {session_id} with workspace {self.workspace_path}")
    
    def _setup_workspace(self):
        """Create isolated workspace directory structure."""
        try:
            # Create main workspace directory
            self.workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.workspace_path / "data").mkdir(exist_ok=True)
            (self.workspace_path / "plots").mkdir(exist_ok=True)
            (self.workspace_path / "exports").mkdir(exist_ok=True)
            (self.workspace_path / "temp").mkdir(exist_ok=True)
            
            logger.debug(f"Workspace setup complete: {self.workspace_path}")
            
        except Exception as e:
            logger.error(f"Failed to setup workspace {self.workspace_path}: {e}")
            self.status = SessionStatus.ERROR
            raise
    
    @property
    def agent_client(self) -> AgentClient:
        """Get or create the agent client for this session."""
        if self._agent_client is None:
            try:
                # Create data manager with session-specific workspace
                data_manager = DataManager(workspace_path=self.workspace_path)
                
                # Create agent client with the session's data manager
                self._agent_client = AgentClient(data_manager=data_manager)
                
                logger.debug(f"Initialized agent client for session {self.session_id}")
                
            except Exception as e:
                logger.error(f"Failed to initialize agent client for session {self.session_id}: {e}")
                self.status = SessionStatus.ERROR
                raise
        
        return self._agent_client
    
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = datetime.utcnow()
        if self.status == SessionStatus.INACTIVE:
            self.status = SessionStatus.ACTIVE
    
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        if self.status == SessionStatus.EXPIRED:
            return True
            
        timeout_delta = timedelta(minutes=self.timeout_minutes)
        return datetime.utcnow() - self.last_activity > timeout_delta
    
    def add_websocket(self, websocket: WebSocket):
        """Add a WebSocket connection to this session."""
        self.websocket_connections.add(websocket)
        logger.debug(f"Added WebSocket to session {self.session_id}. Total: {len(self.websocket_connections)}")
    
    def remove_websocket(self, websocket: WebSocket):
        """Remove a WebSocket connection from this session."""
        self.websocket_connections.discard(websocket)
        logger.debug(f"Removed WebSocket from session {self.session_id}. Total: {len(self.websocket_connections)}")
    
    async def broadcast_message(self, message: dict):
        """Broadcast a message to all WebSocket connections for this session."""
        if not self.websocket_connections:
            return
            
        # Create a copy of connections to avoid modification during iteration
        connections = self.websocket_connections.copy()
        
        # Convert any UUID objects to strings for JSON serialization
        serializable_message = self._make_json_serializable(message)
        
        for websocket in connections:
            try:
                await websocket.send_json(serializable_message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket in session {self.session_id}: {e}")
                # Remove failed connection
                self.websocket_connections.discard(websocket)
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def to_session_info(self) -> SessionInfo:
        """Convert to SessionInfo model for API responses."""
        return SessionInfo(
            session_id=self.session_id,
            name=self.name,
            description=self.description,
            user_id=self.user_id,
            status=self.status,
            created_at=self.created_at,
            last_activity=self.last_activity,
            timeout_minutes=self.timeout_minutes,
            workspace_path=str(self.workspace_path),
            datasets=self.datasets.copy(),
            message_count=self.message_count
        )
    
    async def cleanup(self):
        """Clean up session resources."""
        logger.info(f"Cleaning up session {self.session_id}")
        
        # Close WebSocket connections
        connections = self.websocket_connections.copy()
        for websocket in connections:
            try:
                await websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
        
        self.websocket_connections.clear()
        
        # Clean up agent client if it exists
        if self._agent_client:
            try:
                # The agent client doesn't have async cleanup, but we can reset it
                self._agent_client = None
            except Exception as e:
                logger.warning(f"Error cleaning up agent client: {e}")
        
        # Optionally remove workspace directory (commented out for data persistence)
        # try:
        #     if self.workspace_path.exists():
        #         shutil.rmtree(self.workspace_path)
        #         logger.debug(f"Removed workspace directory: {self.workspace_path}")
        # except Exception as e:
        #     logger.warning(f"Failed to remove workspace directory {self.workspace_path}: {e}")
        
        self.status = SessionStatus.EXPIRED


class SessionManager:
    """Manages multiple user sessions with cleanup and persistence."""
    
    def __init__(self):
        self.sessions: Dict[UUID, Session] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        
        logger.info("SessionManager initialized")
    
    def _start_cleanup_task(self):
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                logger.info("Session cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in session cleanup task: {e}")
    
    async def create_session(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        timeout_minutes: int = 30
    ) -> Session:
        """Create a new session."""
        session_id = uuid4()
        
        session = Session(
            session_id=session_id,
            name=name,
            description=description,
            user_id=user_id,
            timeout_minutes=timeout_minutes
        )
        
        self.sessions[session_id] = session
        
        logger.info(f"Created session {session_id} for user {user_id or 'anonymous'}")
        return session
    
    def get_session(self, session_id: UUID) -> Optional[Session]:
        """Get a session by ID."""
        session = self.sessions.get(session_id)
        
        if session is None:
            return None
        
        if session.is_expired():
            logger.info(f"Session {session_id} has expired")
            # Mark as expired but don't remove immediately
            session.status = SessionStatus.EXPIRED
            return None
        
        # Update activity timestamp
        session.update_activity()
        return session
    
    def list_sessions(self, user_id: Optional[str] = None) -> list[Session]:
        """List all active sessions, optionally filtered by user_id."""
        sessions = []
        
        for session in self.sessions.values():
            if session.is_expired():
                continue
                
            if user_id is None or session.user_id == user_id:
                sessions.append(session)
        
        return sessions
    
    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session and clean up its resources."""
        session = self.sessions.get(session_id)
        
        if session is None:
            return False
        
        # Clean up session resources
        await session.cleanup()
        
        # Remove from sessions dict
        del self.sessions[session_id]
        
        logger.info(f"Deleted session {session_id}")
        return True
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up all expired sessions."""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
        
        cleaned_count = 0
        for session_id in expired_sessions:
            try:
                await self.delete_session(session_id)
                cleaned_count += 1
            except Exception as e:
                logger.error(f"Error cleaning up expired session {session_id}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired sessions")
        
        return cleaned_count
    
    async def cleanup_all_sessions(self):
        """Clean up all sessions (used during shutdown)."""
        logger.info("Cleaning up all sessions...")
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all sessions
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            try:
                await self.delete_session(session_id)
            except Exception as e:
                logger.error(f"Error cleaning up session {session_id}: {e}")
        
        logger.info("All sessions cleaned up")
    
    def get_stats(self) -> dict:
        """Get session statistics."""
        active_sessions = len([s for s in self.sessions.values() if not s.is_expired()])
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "expired_sessions": len(self.sessions) - active_sessions
        }
