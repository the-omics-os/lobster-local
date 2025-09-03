"""
WebSocket Logging Handler
Forwards Python log messages to WebSocket callback system for real-time UI updates.
"""

import logging
import asyncio
from typing import Optional, Set, Dict, Any
from lobster.api.models import WSEventType
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class WebSocketLoggingHandler(logging.Handler):
    """
    Custom logging handler that forwards log messages to WebSocket callbacks.
    
    This handler captures log messages from specified loggers and forwards them
    to the WebSocket callback system for real-time display in the UI.
    """
    
    def __init__(
        self, 
        callback_manager=None,
        level=logging.INFO,
        logger_prefixes: Optional[Set[str]] = None,
        max_message_length: int = 500
    ):
        """
        Initialize the WebSocket logging handler.
        
        Args:
            callback_manager: WebSocket callback manager instance
            level: Minimum logging level to handle
            logger_prefixes: Set of logger name prefixes to capture (e.g., {'lobster.tools', 'lobster.agents'})
            max_message_length: Maximum length of log messages to forward
        """
        super().__init__(level)
        self.callback_manager = callback_manager
        self.logger_prefixes = logger_prefixes or {
            'lobster.tools',
            'lobster.agents', 
            'lobster.core'
        }
        self.max_message_length = max_message_length
        
        # Message deduplication
        self._recent_messages: Dict[str, float] = {}
        self._dedupe_window = 1.0  # seconds
        
    def emit(self, record: logging.LogRecord):
        """
        Emit a log record to WebSocket callbacks.
        
        Args:
            record: The log record to process
        """
        try:
            # Check if this logger should be forwarded
            if not self._should_forward_logger(record.name):
                return
                
            # Check if callback manager is available
            if not self.callback_manager:
                return
                
            # Format the message
            message = self.format(record)
            
            # Truncate if too long
            if len(message) > self.max_message_length:
                message = message[:self.max_message_length - 3] + "..."
            
            # Deduplicate recent messages
            if self._is_duplicate_message(message):
                return
                
            # Map log level to step type and status
            step_type, status = self._map_log_level_to_step_type(record.levelno)
            
            # Create progress update data
            progress_data = {
                'status': 'progress_log',
                'message': message,
                'logger_name': record.name,
                'level': record.levelname,
                'agent': self._extract_agent_name(record.name),
                'tool': self._extract_tool_name(record.name)
            }
            
            # Schedule WebSocket message
            self._schedule_websocket_message(step_type, progress_data)
            
        except Exception as e:
            # Don't let logging errors break the application
            logger.debug(f"Error in WebSocket logging handler: {e}")
    
    def _should_forward_logger(self, logger_name: str) -> bool:
        """Check if this logger should be forwarded to WebSocket."""
        return any(logger_name.startswith(prefix) for prefix in self.logger_prefixes)
    
    def _is_duplicate_message(self, message: str) -> bool:
        """Check if this message was recently sent to avoid spam."""
        import time
        current_time = time.time()
        
        # Clean old messages
        cutoff_time = current_time - self._dedupe_window
        self._recent_messages = {
            msg: timestamp for msg, timestamp in self._recent_messages.items()
            if timestamp > cutoff_time
        }
        
        # Check if message is duplicate
        if message in self._recent_messages:
            return True
            
        # Store this message
        self._recent_messages[message] = current_time
        return False
    
    def _map_log_level_to_step_type(self, level: int) -> tuple:
        """Map Python log level to step type and status."""
        if level >= logging.ERROR:
            return 'error', 'error'
        elif level >= logging.WARNING:
            return 'progress', 'active'
        else:  # INFO, DEBUG
            return 'progress', 'completed'
    
    def _extract_agent_name(self, logger_name: str) -> Optional[str]:
        """Extract agent name from logger name."""
        if 'agents' in logger_name:
            parts = logger_name.split('.')
            if 'agents' in parts:
                agent_idx = parts.index('agents')
                if len(parts) > agent_idx + 1:
                    return parts[agent_idx + 1].replace('_', ' ').title()
        return 'system'
    
    def _extract_tool_name(self, logger_name: str) -> Optional[str]:
        """Extract tool name from logger name."""
        if 'tools' in logger_name:
            parts = logger_name.split('.')
            if 'tools' in parts:
                tool_idx = parts.index('tools')
                if len(parts) > tool_idx + 1:
                    return parts[tool_idx + 1].replace('_service', '').replace('_', ' ').title()
        return None
    
    def _schedule_websocket_message(self, event_type: str, data: Dict[str, Any]):
        """Schedule a WebSocket message to be sent."""
        if not self.callback_manager:
            return
            
        try:
            # Map our event types to WSEventType
            ws_event_type = WSEventType.ANALYSIS_PROGRESS
            if event_type == 'error':
                ws_event_type = WSEventType.ERROR
            
            # Check if this is an APICallbackManager
            if hasattr(self.callback_manager, 'websocket_handler'):
                # This is an APICallbackManager - use its websocket_handler
                websocket_handler = self.callback_manager.websocket_handler
                if hasattr(websocket_handler, '_schedule_websocket_message'):
                    websocket_handler._schedule_websocket_message(ws_event_type, data)
                else:
                    # Fallback: try to send directly
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            websocket_handler._send_websocket_message(ws_event_type, data)
                        )
                    except RuntimeError:
                        # No event loop - this is expected in some contexts
                        logger.debug("No event loop available for WebSocket message")
            elif hasattr(self.callback_manager, '_schedule_websocket_message'):
                # This is likely a WebSocketCallbackHandler directly
                self.callback_manager._schedule_websocket_message(ws_event_type, data)
            elif hasattr(self.callback_manager, '_send_websocket_message'):
                # Direct WebSocket sender
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        self.callback_manager._send_websocket_message(ws_event_type, data)
                    )
                except RuntimeError:
                    # No event loop - this is expected in some contexts
                    logger.debug("No event loop available for WebSocket message")
                
        except Exception as e:
            logger.debug(f"Error scheduling WebSocket message: {e}")


def setup_websocket_logging(callback_manager, level=logging.INFO) -> WebSocketLoggingHandler:
    """
    Set up WebSocket logging handler for the application.
    
    Args:
        callback_manager: WebSocket callback manager instance
        level: Minimum logging level to forward
        
    Returns:
        WebSocketLoggingHandler: The configured handler
    """
    handler = WebSocketLoggingHandler(
        callback_manager=callback_manager,
        level=level
    )
    
    # Add handler to relevant loggers
    loggers_to_enhance = [
        'lobster.tools',
        'lobster.agents', 
        'lobster.core'
    ]
    
    for logger_name in loggers_to_enhance:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.addHandler(handler)
        # Ensure the logger level allows INFO messages
        if logger_instance.level > level:
            logger_instance.setLevel(level)
    
    logger.info("WebSocket logging handler configured")
    return handler


def remove_websocket_logging(handler: WebSocketLoggingHandler):
    """
    Remove WebSocket logging handler from all loggers.
    
    Args:
        handler: The handler to remove
    """
    loggers_to_clean = [
        'lobster.tools',
        'lobster.agents',
        'lobster.core'
    ]
    
    for logger_name in loggers_to_clean:
        logger_instance = logging.getLogger(logger_name)
        if handler in logger_instance.handlers:
            logger_instance.removeHandler(handler)
    
    logger.info("WebSocket logging handler removed")
