"""
Lobster AI - WebSocket Connection Management
Handles WebSocket connections for real-time communication with API clients.
"""

from typing import Dict, Set
from uuid import UUID
import json

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.routing import APIRouter

from lobster.api.session_manager import SessionManager
from lobster.api.models import WSMessage, WSEventType
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Create router for WebSocket endpoints
router = APIRouter()


class WebSocketManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        # Store active connections by session ID
        self.connections: Dict[UUID, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: UUID, session_manager: SessionManager):
        """
        Accept a new WebSocket connection and associate it with a session.
        
        Args:
            websocket: The WebSocket connection
            session_id: Session UUID to associate with
            session_manager: Session manager to validate session exists
        """
        # Validate session exists
        session = session_manager.get_session(session_id)
        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return False
        
        # Accept the connection
        await websocket.accept()
        
        # Add to session's WebSocket connections
        session.add_websocket(websocket)
        
        # Also track in our global manager
        if session_id not in self.connections:
            self.connections[session_id] = set()
        self.connections[session_id].add(websocket)
        
        logger.info(f"WebSocket connected to session {session_id}")
        
        # Send welcome message
        welcome_message = {
            "event_type": "connection",
            "session_id": str(session_id),
            "data": {
                "status": "connected",
                "message": "WebSocket connected successfully",
                "session_info": {
                    "session_id": str(session.session_id),
                    "created_at": session.created_at.isoformat(),
                    "workspace_path": str(session.workspace_path.name)
                }
            },
            "timestamp": session.created_at.isoformat()
        }
        
        await websocket.send_json(welcome_message)
        return True
    
    async def disconnect(self, websocket: WebSocket, session_id: UUID, session_manager: SessionManager):
        """
        Handle WebSocket disconnection.
        
        Args:
            websocket: The WebSocket connection to remove
            session_id: Session UUID to remove from
            session_manager: Session manager to update session state
        """
        # Remove from session
        session = session_manager.get_session(session_id)
        if session:
            session.remove_websocket(websocket)
        
        # Remove from global tracking
        if session_id in self.connections:
            self.connections[session_id].discard(websocket)
            if not self.connections[session_id]:
                del self.connections[session_id]
        
        logger.info(f"WebSocket disconnected from session {session_id}")
    
    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal WebSocket message: {e}")
    
    async def broadcast_to_session(self, session_id: UUID, message: dict):
        """Broadcast a message to all WebSocket connections for a session."""
        if session_id not in self.connections:
            return
        
        # Create a copy of connections to avoid modification during iteration
        connections = self.connections[session_id].copy()
        
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket in session {session_id}: {e}")
                # Remove failed connection
                self.connections[session_id].discard(websocket)
    
    def get_session_connection_count(self, session_id: UUID) -> int:
        """Get the number of active connections for a session."""
        return len(self.connections.get(session_id, set()))
    
    def get_total_connections(self) -> int:
        """Get total number of active WebSocket connections."""
        return sum(len(connections) for connections in self.connections.values())
    
    def get_active_sessions(self) -> Set[UUID]:
        """Get set of session IDs with active WebSocket connections."""
        return set(self.connections.keys())


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


def get_websocket_manager() -> WebSocketManager:
    """Dependency to get the WebSocket manager."""
    return websocket_manager


@router.websocket("/ws/sessions/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: UUID,
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """
    WebSocket endpoint for real-time communication with a specific session.
    
    This endpoint handles:
    - Real-time agent streaming
    - Progress updates during analysis
    - Data and plot generation notifications
    - Error reporting
    """
    try:
        # Get session manager from app state since we can't inject it directly
        session_manager = None
        if hasattr(websocket, 'scope') and 'app' in websocket.scope:
            session_manager = websocket.scope['app'].state.session_manager
        
        if not session_manager:
            await websocket.close(code=1008, reason="Session manager not available")
            return
        
        # Connect to the session
        connected = await ws_manager.connect(websocket, session_id, session_manager)
        if not connected:
            return
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    await handle_websocket_message(websocket, session_id, message, session_manager)
                except json.JSONDecodeError:
                    # Handle non-JSON messages
                    await ws_manager.send_personal_message(websocket, {
                        "event_type": "error",
                        "data": {
                            "status": "invalid_message",
                            "message": "Invalid JSON format"
                        }
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket message handling: {e}")
                await ws_manager.send_personal_message(websocket, {
                    "event_type": "error",
                    "data": {
                        "status": "server_error",
                        "message": "Server error occurred"
                    }
                })
                break
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        # Clean up connection
        await ws_manager.disconnect(websocket, session_id, session_manager)


async def handle_websocket_message(
    websocket: WebSocket,
    session_id: UUID,
    message: dict,
    session_manager: SessionManager
):
    """
    Handle incoming WebSocket messages from clients.
    
    Args:
        websocket: The WebSocket connection
        session_id: Session UUID
        message: Parsed JSON message from client
        session_manager: Session manager instance
    """
    try:
        message_type = message.get("type", "unknown")
        
        if message_type == "ping":
            # Handle ping/pong for connection health
            await websocket.send_json({
                "type": "pong",
                "timestamp": message.get("timestamp")
            })
        
        elif message_type == "session_status":
            # Send current session status
            session = session_manager.get_session(session_id)
            if session:
                session_info = session.to_session_info()
                await websocket.send_json({
                    "event_type": "session_status",
                    "data": {
                        "session": session_info.dict(),
                        "connections": websocket_manager.get_session_connection_count(session_id)
                    }
                })
            else:
                await websocket.send_json({
                    "event_type": "error",
                    "data": {
                        "status": "session_not_found",
                        "message": "Session not found"
                    }
                })
        
        elif message_type == "subscribe":
            # Handle subscription to specific event types
            event_types = message.get("event_types", [])
            await websocket.send_json({
                "event_type": "subscription_confirmed",
                "data": {
                    "subscribed_events": event_types,
                    "message": "Subscription confirmed"
                }
            })
        
        else:
            # Unknown message type
            await websocket.send_json({
                "event_type": "error",
                "data": {
                    "status": "unknown_message_type",
                    "message": f"Unknown message type: {message_type}"
                }
            })
    
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        await websocket.send_json({
            "event_type": "error",
            "data": {
                "status": "message_handling_error",
                "message": "Error processing message"
            }
        })


# Utility functions for sending specific types of WebSocket messages

async def send_agent_progress(session_id: UUID, agent_name: str, progress: dict):
    """Send agent progress update to session WebSocket connections."""
    message = {
        "event_type": str(WSEventType.ANALYSIS_PROGRESS),
        "session_id": str(session_id),
        "data": {
            "agent": str(agent_name),
            "progress": progress,
            "message": f"Progress from {agent_name}"
        },
        "timestamp": "2025-01-01T00:00:00"
    }
    await websocket_manager.broadcast_to_session(session_id, message)


async def send_data_update(session_id: UUID, dataset_info: dict):
    """Send data update notification to session WebSocket connections."""
    message = {
        "event_type": str(WSEventType.DATA_UPDATED),
        "session_id": str(session_id),
        "data": {
            "dataset": dataset_info,
            "message": "Dataset updated"
        },
        "timestamp": "2025-01-01T00:00:00"
    }
    await websocket_manager.broadcast_to_session(session_id, message)


async def send_plot_generated(session_id: UUID, plot_info: dict):
    """Send plot generation notification to session WebSocket connections."""
    message = {
        "event_type": str(WSEventType.PLOT_GENERATED),
        "session_id": str(session_id),
        "data": {
            "plot": plot_info,
            "message": "New plot generated"
        },
        "timestamp": "2025-01-01T00:00:00"
    }
    await websocket_manager.broadcast_to_session(session_id, message)


async def send_error(session_id: UUID, error_message: str, error_details: dict = None):
    """Send error notification to session WebSocket connections."""
    message = {
        "event_type": str(WSEventType.ERROR),
        "session_id": str(session_id),
        "data": {
            "message": str(error_message),
            "details": error_details or {},
            "error": True
        },
        "timestamp": "2025-01-01T00:00:00"
    }
    await websocket_manager.broadcast_to_session(session_id, message)
