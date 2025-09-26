"""
Comprehensive unit tests for WebSocket logging handler.
Tests WebSocketLoggingHandler and related logging functionality.
"""

import asyncio
import logging
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from uuid import uuid4

from lobster.api.models import WSEventType
from lobster.core.websocket_logging_handler import (
    WebSocketLoggingHandler,
    setup_websocket_logging,
    remove_websocket_logging
)


class TestWebSocketLoggingHandler:
    """Test suite for WebSocketLoggingHandler."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.mock_callback_manager = Mock()
        self.mock_websocket_handler = Mock()
        self.mock_websocket_handler._schedule_websocket_message = Mock()
        self.mock_callback_manager.websocket_handler = self.mock_websocket_handler

        self.handler = WebSocketLoggingHandler(
            callback_manager=self.mock_callback_manager,
            level=logging.INFO
        )

    def test_handler_initialization_default(self):
        """Test WebSocket logging handler initialization with defaults."""
        handler = WebSocketLoggingHandler()

        assert handler.callback_manager is None
        assert handler.level == logging.INFO  # Default level is INFO, not NOTSET
        assert handler.logger_prefixes == {
            'lobster.tools',
            'lobster.agents',
            'lobster.core'
        }
        assert handler.max_message_length == 500
        assert handler._dedupe_window == 1.0

    def test_handler_initialization_custom(self):
        """Test WebSocket logging handler initialization with custom values."""
        custom_prefixes = {'custom.module', 'another.module'}
        handler = WebSocketLoggingHandler(
            callback_manager=self.mock_callback_manager,
            level=logging.DEBUG,
            logger_prefixes=custom_prefixes,
            max_message_length=1000
        )

        assert handler.callback_manager == self.mock_callback_manager
        assert handler.level == logging.DEBUG
        assert handler.logger_prefixes == custom_prefixes
        assert handler.max_message_length == 1000

    def test_should_forward_logger_matching_prefix(self):
        """Test logger forwarding for matching prefixes."""
        assert self.handler._should_forward_logger('lobster.tools.service') is True
        assert self.handler._should_forward_logger('lobster.agents.expert') is True
        assert self.handler._should_forward_logger('lobster.core.client') is True

    def test_should_forward_logger_non_matching_prefix(self):
        """Test logger forwarding for non-matching prefixes."""
        assert self.handler._should_forward_logger('other.module') is False
        assert self.handler._should_forward_logger('lobster.invalid') is False
        assert self.handler._should_forward_logger('external.service') is False

    def test_is_duplicate_message_new_message(self):
        """Test duplicate detection for new messages."""
        message = "Test log message"
        assert self.handler._is_duplicate_message(message) is False

        # Same message should now be duplicate
        assert self.handler._is_duplicate_message(message) is True

    def test_is_duplicate_message_expired_message(self):
        """Test duplicate detection for expired messages."""
        message = "Test log message"

        # Add message to recent messages with old timestamp
        old_time = time.time() - 2.0  # 2 seconds ago (beyond dedupe window)
        self.handler._recent_messages[message] = old_time

        # Should not be considered duplicate (expired)
        assert self.handler._is_duplicate_message(message) is False

        # Should now be in recent messages again
        assert self.handler._is_duplicate_message(message) is True

    def test_is_duplicate_message_cleanup(self):
        """Test cleanup of old messages during duplicate check."""
        # Add multiple messages with different timestamps
        current_time = time.time()
        self.handler._recent_messages = {
            "old_message": current_time - 2.0,  # Expired
            "recent_message": current_time - 0.5,  # Still fresh
        }

        new_message = "new_test_message"
        assert self.handler._is_duplicate_message(new_message) is False

        # Old message should be cleaned up
        assert "old_message" not in self.handler._recent_messages
        assert "recent_message" in self.handler._recent_messages
        assert new_message in self.handler._recent_messages

    def test_map_log_level_to_step_type_error(self):
        """Test log level mapping for ERROR level."""
        step_type, status = self.handler._map_log_level_to_step_type(logging.ERROR)
        assert step_type == 'error'
        assert status == 'error'

        step_type, status = self.handler._map_log_level_to_step_type(logging.CRITICAL)
        assert step_type == 'error'
        assert status == 'error'

    def test_map_log_level_to_step_type_warning(self):
        """Test log level mapping for WARNING level."""
        step_type, status = self.handler._map_log_level_to_step_type(logging.WARNING)
        assert step_type == 'progress'
        assert status == 'active'

    def test_map_log_level_to_step_type_info_debug(self):
        """Test log level mapping for INFO and DEBUG levels."""
        step_type, status = self.handler._map_log_level_to_step_type(logging.INFO)
        assert step_type == 'progress'
        assert status == 'completed'

        step_type, status = self.handler._map_log_level_to_step_type(logging.DEBUG)
        assert step_type == 'progress'
        assert status == 'completed'

    def test_extract_agent_name_from_agents_logger(self):
        """Test agent name extraction from agents logger."""
        agent_name = self.handler._extract_agent_name('lobster.agents.singlecell_expert')
        assert agent_name == 'Singlecell Expert'

        agent_name = self.handler._extract_agent_name('lobster.agents.bulk_rnaseq_expert')
        assert agent_name == 'Bulk Rnaseq Expert'

    def test_extract_agent_name_from_non_agents_logger(self):
        """Test agent name extraction from non-agents logger."""
        agent_name = self.handler._extract_agent_name('lobster.tools.service')
        assert agent_name == 'system'

        agent_name = self.handler._extract_agent_name('lobster.core.client')
        assert agent_name == 'system'

    def test_extract_agent_name_incomplete_path(self):
        """Test agent name extraction from incomplete path."""
        agent_name = self.handler._extract_agent_name('lobster.agents')
        assert agent_name == 'system'

    def test_extract_tool_name_from_tools_logger(self):
        """Test tool name extraction from tools logger."""
        tool_name = self.handler._extract_tool_name('lobster.tools.preprocessing_service')
        assert tool_name == 'Preprocessing'

        tool_name = self.handler._extract_tool_name('lobster.tools.quality_service')
        assert tool_name == 'Quality'

        tool_name = self.handler._extract_tool_name('lobster.tools.custom_tool')
        assert tool_name == 'Custom Tool'

    def test_extract_tool_name_from_non_tools_logger(self):
        """Test tool name extraction from non-tools logger."""
        tool_name = self.handler._extract_tool_name('lobster.agents.expert')
        assert tool_name is None

        tool_name = self.handler._extract_tool_name('lobster.core.client')
        assert tool_name is None

    def test_extract_tool_name_incomplete_path(self):
        """Test tool name extraction from incomplete path."""
        tool_name = self.handler._extract_tool_name('lobster.tools')
        assert tool_name is None

    def test_emit_successful_forwarding(self):
        """Test successful log record emission and forwarding."""
        # Create a log record
        record = logging.LogRecord(
            name='lobster.tools.test_service',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test log message',
            args=(),
            exc_info=None
        )

        # Mock the format method
        self.handler.format = Mock(return_value='Formatted test message')

        self.handler.emit(record)

        # Verify websocket_handler method was called
        self.mock_websocket_handler._schedule_websocket_message.assert_called_once_with(
            WSEventType.ANALYSIS_PROGRESS,
            {
                'status': 'progress_log',
                'message': 'Formatted test message',
                'logger_name': 'lobster.tools.test_service',
                'level': 'INFO',
                'agent': 'system',
                'tool': 'Test'
            }
        )

    def test_emit_non_forwarded_logger(self):
        """Test log record emission for non-forwarded logger."""
        record = logging.LogRecord(
            name='external.module',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test log message',
            args=(),
            exc_info=None
        )

        self.handler.emit(record)

        # Should not call websocket handler
        self.mock_websocket_handler._schedule_websocket_message.assert_not_called()

    def test_emit_no_callback_manager(self):
        """Test log record emission without callback manager."""
        handler = WebSocketLoggingHandler(callback_manager=None)
        record = logging.LogRecord(
            name='lobster.tools.test_service',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test log message',
            args=(),
            exc_info=None
        )

        # Should not raise exception
        handler.emit(record)

    def test_emit_message_truncation(self):
        """Test log message truncation for long messages."""
        long_message = "x" * 600  # Longer than max_message_length (500)

        record = logging.LogRecord(
            name='lobster.tools.test_service',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test log message',
            args=(),
            exc_info=None
        )

        # Mock the format method to return long message
        self.handler.format = Mock(return_value=long_message)

        self.handler.emit(record)

        # Verify message was truncated
        expected_message = "x" * 497 + "..."  # 500 - 3 for "..."
        call_args = self.mock_websocket_handler._schedule_websocket_message.call_args[0][1]
        assert call_args['message'] == expected_message

    def test_emit_duplicate_message_filtering(self):
        """Test duplicate message filtering."""
        record = logging.LogRecord(
            name='lobster.tools.test_service',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test log message',
            args=(),
            exc_info=None
        )

        self.handler.format = Mock(return_value='Same message')

        # First emission should succeed
        self.handler.emit(record)
        assert self.mock_websocket_handler._schedule_websocket_message.call_count == 1

        # Second emission with same message should be filtered
        self.handler.emit(record)
        assert self.mock_websocket_handler._schedule_websocket_message.call_count == 1

    def test_emit_error_level_mapping(self):
        """Test error level mapping in emission."""
        record = logging.LogRecord(
            name='lobster.agents.test_agent',
            level=logging.ERROR,
            pathname='test.py',
            lineno=1,
            msg='Error message',
            args=(),
            exc_info=None
        )

        self.handler.format = Mock(return_value='Formatted error')

        self.handler.emit(record)

        # Should use ERROR event type for error level
        self.mock_websocket_handler._schedule_websocket_message.assert_called_once_with(
            WSEventType.ERROR,
            {
                'status': 'progress_log',
                'message': 'Formatted error',
                'logger_name': 'lobster.agents.test_agent',
                'level': 'ERROR',
                'agent': 'Test Agent',
                'tool': None
            }
        )

    def test_emit_exception_handling(self):
        """Test exception handling during emission."""
        record = logging.LogRecord(
            name='lobster.tools.test_service',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test log message',
            args=(),
            exc_info=None
        )

        # Mock format to raise exception
        self.handler.format = Mock(side_effect=Exception("Format error"))

        # Should not raise exception
        self.handler.emit(record)

    def test_schedule_websocket_message_api_callback_manager(self):
        """Test WebSocket message scheduling with APICallbackManager."""
        event_type = 'progress'
        data = {'test': 'data'}

        self.handler._schedule_websocket_message(event_type, data)

        # Should call websocket_handler's schedule method
        self.mock_websocket_handler._schedule_websocket_message.assert_called_once_with(
            WSEventType.ANALYSIS_PROGRESS, data
        )

    def test_schedule_websocket_message_websocket_handler_direct(self):
        """Test WebSocket message scheduling with WebSocketCallbackHandler directly."""
        # Mock callback manager that has _schedule_websocket_message method
        mock_handler = Mock()
        mock_handler._schedule_websocket_message = Mock()
        # Remove websocket_handler attribute to trigger the direct handler path
        if hasattr(mock_handler, 'websocket_handler'):
            del mock_handler.websocket_handler
        self.handler.callback_manager = mock_handler

        event_type = 'error'
        data = {'error': 'test error'}

        self.handler._schedule_websocket_message(event_type, data)

        mock_handler._schedule_websocket_message.assert_called_once_with(
            WSEventType.ERROR, data
        )

    def test_schedule_websocket_message_direct_sender(self):
        """Test WebSocket message scheduling with direct sender."""
        mock_sender = Mock()
        mock_sender._send_websocket_message = AsyncMock()
        # Remove both websocket_handler and _schedule_websocket_message to trigger direct sender path
        if hasattr(mock_sender, 'websocket_handler'):
            del mock_sender.websocket_handler
        if hasattr(mock_sender, '_schedule_websocket_message'):
            del mock_sender._schedule_websocket_message
        self.handler.callback_manager = mock_sender

        mock_loop = Mock()
        mock_task = Mock()
        mock_loop.create_task.return_value = mock_task

        with patch('asyncio.get_running_loop', return_value=mock_loop):
            self.handler._schedule_websocket_message('progress', {'test': 'data'})

            mock_loop.create_task.assert_called_once()

    def test_schedule_websocket_message_no_event_loop(self):
        """Test WebSocket message scheduling without event loop."""
        mock_sender = Mock()
        mock_sender._send_websocket_message = AsyncMock()
        self.handler.callback_manager = mock_sender

        with patch('asyncio.get_running_loop', side_effect=RuntimeError("No event loop")):
            # Should not raise exception
            self.handler._schedule_websocket_message('progress', {'test': 'data'})

    def test_schedule_websocket_message_exception_handling(self):
        """Test exception handling in WebSocket message scheduling."""
        # Mock callback manager that raises exception
        mock_handler = Mock()
        mock_handler._schedule_websocket_message = Mock(side_effect=Exception("Test error"))
        self.handler.callback_manager = mock_handler

        # Should not raise exception
        self.handler._schedule_websocket_message('progress', {'test': 'data'})

    def test_schedule_websocket_message_no_callback_manager(self):
        """Test WebSocket message scheduling without callback manager."""
        self.handler.callback_manager = None

        # Should not raise exception
        self.handler._schedule_websocket_message('progress', {'test': 'data'})


class TestWebSocketLoggingSetup:
    """Test suite for WebSocket logging setup and teardown."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_callback_manager = Mock()

    def test_setup_websocket_logging(self):
        """Test WebSocket logging setup."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_loggers = {}
            for name in ['lobster.tools', 'lobster.agents', 'lobster.core']:
                mock_logger = Mock()
                mock_logger.level = logging.WARNING  # Higher than INFO
                mock_logger.addHandler = Mock()
                mock_logger.setLevel = Mock()
                mock_loggers[name] = mock_logger

            mock_get_logger.side_effect = lambda name: mock_loggers.get(name, Mock())

            handler = setup_websocket_logging(self.mock_callback_manager, logging.INFO)

            # Verify handler configuration
            assert isinstance(handler, WebSocketLoggingHandler)
            assert handler.callback_manager == self.mock_callback_manager

            # Verify loggers were configured
            for logger_name in ['lobster.tools', 'lobster.agents', 'lobster.core']:
                mock_logger = mock_loggers[logger_name]
                mock_logger.addHandler.assert_called_once_with(handler)
                mock_logger.setLevel.assert_called_once_with(logging.INFO)

    def test_setup_websocket_logging_logger_level_not_changed(self):
        """Test WebSocket logging setup when logger level is already appropriate."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.level = logging.DEBUG  # Lower than INFO - should not be changed
            mock_logger.addHandler = Mock()
            mock_logger.setLevel = Mock()

            mock_get_logger.return_value = mock_logger

            handler = setup_websocket_logging(self.mock_callback_manager, logging.INFO)

            # setLevel should not be called for logger with lower level
            mock_logger.setLevel.assert_not_called()

    def test_remove_websocket_logging(self):
        """Test WebSocket logging removal."""
        # Create a real handler instance
        handler = WebSocketLoggingHandler(self.mock_callback_manager)

        with patch('logging.getLogger') as mock_get_logger:
            mock_loggers = {}
            for name in ['lobster.tools', 'lobster.agents', 'lobster.core']:
                mock_logger = Mock()
                mock_logger.handlers = [handler, Mock()]  # Include our handler
                mock_logger.removeHandler = Mock()
                mock_loggers[name] = mock_logger

            mock_get_logger.side_effect = lambda name: mock_loggers.get(name, Mock())

            remove_websocket_logging(handler)

            # Verify handler was removed from all loggers
            for logger_name in ['lobster.tools', 'lobster.agents', 'lobster.core']:
                mock_logger = mock_loggers[logger_name]
                mock_logger.removeHandler.assert_called_once_with(handler)

    def test_remove_websocket_logging_handler_not_present(self):
        """Test WebSocket logging removal when handler is not present."""
        handler = WebSocketLoggingHandler(self.mock_callback_manager)

        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.handlers = [Mock()]  # Different handler
            mock_logger.removeHandler = Mock()

            mock_get_logger.return_value = mock_logger

            # Should not raise exception
            remove_websocket_logging(handler)

            # removeHandler should not be called
            mock_logger.removeHandler.assert_not_called()


class TestWebSocketLoggingIntegration:
    """Integration tests for WebSocket logging."""

    def setup_method(self):
        """Set up test fixtures for integration tests."""
        self.mock_callback_manager = Mock()
        self.mock_websocket_handler = Mock()
        self.mock_websocket_handler._schedule_websocket_message = Mock()
        self.mock_callback_manager.websocket_handler = self.mock_websocket_handler

    def test_real_logging_integration(self):
        """Test integration with real Python logging."""
        handler = setup_websocket_logging(self.mock_callback_manager, logging.INFO)

        try:
            # Create a logger that should be captured
            test_logger = logging.getLogger('lobster.tools.test_service')

            # Log a message
            test_logger.info("Test integration message")

            # Verify the message was processed
            self.mock_websocket_handler._schedule_websocket_message.assert_called_once()

        finally:
            # Clean up
            remove_websocket_logging(handler)

    def test_multiple_loggers_integration(self):
        """Test integration with multiple loggers."""
        handler = setup_websocket_logging(self.mock_callback_manager, logging.INFO)

        try:
            # Create multiple loggers
            tool_logger = logging.getLogger('lobster.tools.preprocessing')
            agent_logger = logging.getLogger('lobster.agents.singlecell')
            core_logger = logging.getLogger('lobster.core.client')

            # Log messages from each
            tool_logger.info("Tool processing data")
            agent_logger.warning("Agent needs attention")
            core_logger.error("Core system error")

            # Should have captured all three messages
            assert self.mock_websocket_handler._schedule_websocket_message.call_count == 3

        finally:
            # Clean up
            remove_websocket_logging(handler)

    def test_filtered_logger_integration(self):
        """Test that non-matching loggers are filtered out."""
        handler = setup_websocket_logging(self.mock_callback_manager, logging.INFO)

        try:
            # Create loggers - one matching, one not
            matching_logger = logging.getLogger('lobster.tools.test')
            non_matching_logger = logging.getLogger('external.service')

            # Log messages from both
            matching_logger.info("Should be captured")
            non_matching_logger.info("Should be ignored")

            # Only the matching logger should have been processed
            assert self.mock_websocket_handler._schedule_websocket_message.call_count == 1

        finally:
            # Clean up
            remove_websocket_logging(handler)

    def test_log_level_filtering_integration(self):
        """Test log level filtering integration."""
        handler = setup_websocket_logging(self.mock_callback_manager, logging.WARNING)

        try:
            test_logger = logging.getLogger('lobster.tools.test')

            # Log messages at different levels
            test_logger.debug("Debug message")    # Should be ignored
            test_logger.info("Info message")      # Should be ignored
            test_logger.warning("Warning message") # Should be captured
            test_logger.error("Error message")     # Should be captured

            # Only warning and error should be captured
            assert self.mock_websocket_handler._schedule_websocket_message.call_count == 2

        finally:
            # Clean up
            remove_websocket_logging(handler)

    def test_performance_high_volume_logging(self):
        """Test performance with high volume logging."""
        handler = setup_websocket_logging(self.mock_callback_manager, logging.INFO)

        try:
            test_logger = logging.getLogger('lobster.tools.performance_test')

            start_time = time.time()

            # Log many messages rapidly
            for i in range(1000):
                test_logger.info(f"Message {i}")

            end_time = time.time()
            duration = end_time - start_time

            # Should complete reasonably quickly (less than 1 second)
            assert duration < 1.0

            # Due to deduplication, actual calls may be less than 1000
            # but should still be processing messages
            assert self.mock_websocket_handler._schedule_websocket_message.call_count > 0

        finally:
            # Clean up
            remove_websocket_logging(handler)

    def test_memory_usage_stability(self):
        """Test memory usage stability during prolonged logging."""
        handler = setup_websocket_logging(self.mock_callback_manager, logging.INFO)

        try:
            test_logger = logging.getLogger('lobster.tools.memory_test')

            # Generate many unique messages to test memory cleanup
            for i in range(100):
                test_logger.info(f"Unique message {i} at {time.time()}")
                time.sleep(0.01)  # Small delay to ensure different timestamps

            # Handler's recent messages should be cleaned up periodically
            # The exact number depends on timing, but should not grow unbounded
            assert len(handler._recent_messages) < 100

        finally:
            # Clean up
            remove_websocket_logging(handler)