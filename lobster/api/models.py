"""
Stub models for API components - for testing purposes only.
"""
from enum import Enum
from typing import Any, Dict


class WSEventType(Enum):
    """WebSocket event types."""
    MESSAGE = "message"
    PROGRESS = "progress"
    ERROR = "error"
    DATA_UPDATE = "data_update"
    DATA_UPDATED = "data_updated"
    PLOT_GENERATED = "plot_generated"
    AGENT_THINKING = "agent_thinking"
    CHAT_STREAM = "chat_stream"
    ANALYSIS_PROGRESS = "analysis_progress"


class WSMessage:
    """WebSocket message structure."""

    def __init__(self, event_type: WSEventType, data: Dict[str, Any], session_id: str = None):
        self.event_type = event_type
        self.data = data
        self.session_id = session_id

    def dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "session_id": str(self.session_id) if self.session_id else None
        }