"""
Integration tests for WebSocket components with API client.
Tests real-world usage patterns and integration with the broader system.
"""

import asyncio
import logging
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from pathlib import Path
import tempfile

from lobster.api.models import WSEventType, WSMessage
from lobster.core.websocket_callback import WebSocketCallbackHandler, APICallbackManager
from lobster.core.websocket_logging_handler import (
    WebSocketLoggingHandler,
    setup_websocket_logging,
    remove_websocket_logging
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    session_manager = Mock()
    session = Mock()
    session.broadcast_message = AsyncMock()
    session_manager.get_session.return_value = session
    return session_manager


@pytest.fixture
def api_callback_manager(mock_session_manager):
    """Create an API callback manager for testing."""
    session_id = uuid4()
    return APICallbackManager(session_id, mock_session_manager)


class TestWebSocketAPIClientIntegration:
    """Test WebSocket integration with API client patterns."""

    @pytest.mark.asyncio
    async def test_callback_manager_with_mock_agent_client(self, api_callback_manager, mock_session_manager):
        """Test callback manager integration with mock agent client workflow."""
        # Simulate AgentClient initialization pattern
        callbacks = api_callback_manager.get_callbacks()
        assert len(callbacks) >= 1
        assert api_callback_manager.websocket_handler in callbacks

        # Simulate agent workflow with callbacks
        handler = api_callback_manager.websocket_handler

        # Agent starts
        handler.on_chain_start({"name": "singlecell_expert"}, {"input": "analyze data"})

        # LLM processing
        handler.on_llm_start({"name": "gpt-4"}, ["Analyze this single-cell data..."])
        handler.on_llm_new_token("I'll")
        handler.on_llm_new_token(" analyze")
        handler.on_llm_new_token(" the")
        handler.on_llm_new_token(" data")
        handler.on_llm_end(Mock())

        # Tool usage
        handler.on_tool_start({"name": "load_data"}, "loading dataset")
        handler.on_tool_end("Data loaded successfully with 1000 cells")

        # More tool usage
        handler.on_tool_start({"name": "quality_control"}, "running QC")
        handler.on_tool_end("QC completed: 950 cells passed filters")

        # Agent completes
        handler.on_chain_end({"output": "Analysis complete"})

        # Allow async operations to complete
        await asyncio.sleep(0.1)

        # Verify session manager was called multiple times
        assert mock_session_manager.get_session.call_count > 0

        # Verify current agent was tracked
        assert handler.current_agent == "singlecell_expert"

    @pytest.mark.asyncio
    async def test_websocket_logging_integration_with_api_workflow(self, api_callback_manager):
        """Test WebSocket logging integration with API workflow."""
        # Set up WebSocket logging
        ws_logging_handler = setup_websocket_logging(api_callback_manager, logging.INFO)

        try:
            # Create loggers that would be used during analysis
            tool_logger = logging.getLogger('lobster.tools.preprocessing_service')
            agent_logger = logging.getLogger('lobster.agents.singlecell_expert')
            core_logger = logging.getLogger('lobster.core.data_manager_v2')

            # Simulate realistic log messages during analysis
            tool_logger.info("Loading dataset from H5AD file")
            time.sleep(0.01)  # Small delay to avoid deduplication

            agent_logger.info("Starting quality control analysis")
            time.sleep(0.01)

            tool_logger.info("Filtering cells with < 200 genes")
            time.sleep(0.01)

            tool_logger.warning("Found 50 cells with high mitochondrial percentage")
            time.sleep(0.01)

            core_logger.info("Saved filtered data to workspace")
            time.sleep(0.01)

            agent_logger.info("Quality control completed successfully")

            # Allow logging to be processed
            await asyncio.sleep(0.1)

            # Verify messages were processed
            # (Exact count may vary due to deduplication)
            assert api_callback_manager.websocket_handler._schedule_websocket_message.call_count > 0

        finally:
            # Clean up logging handler
            remove_websocket_logging(ws_logging_handler)

    @pytest.mark.asyncio
    async def test_progress_and_data_updates(self, api_callback_manager, mock_session_manager):
        """Test progress and data update notifications."""
        # Send various types of updates
        await api_callback_manager.send_progress_update(
            "Loading dataset from GEO",
            {"current_step": 1, "total_steps": 5}
        )

        await api_callback_manager.send_data_update({
            "name": "geo_gse12345",
            "type": "single-cell",
            "cells": 2500,
            "genes": 18000
        })

        await api_callback_manager.send_plot_generated({
            "type": "umap",
            "file": "umap_plot.html",
            "title": "UMAP Visualization"
        })

        # Verify all updates were sent
        session = mock_session_manager.get_session.return_value
        assert session.broadcast_message.call_count == 3

        # Verify message types
        calls = session.broadcast_message.call_args_list

        # Progress update
        progress_msg = calls[0][0][0]
        assert progress_msg["event_type"] == WSEventType.ANALYSIS_PROGRESS.value
        assert "Loading dataset from GEO" in progress_msg["data"]["message"]

        # Data update
        data_msg = calls[1][0][0]
        assert data_msg["event_type"] == WSEventType.DATA_UPDATED.value
        assert data_msg["data"]["dataset"]["name"] == "geo_gse12345"

        # Plot update
        plot_msg = calls[2][0][0]
        assert plot_msg["event_type"] == WSEventType.PLOT_GENERATED.value
        assert plot_msg["data"]["plot"]["type"] == "umap"

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, api_callback_manager, mock_session_manager):
        """Test error handling integration across components."""
        handler = api_callback_manager.websocket_handler

        # Simulate error during agent execution
        handler.on_chain_start({"name": "bulk_rnaseq_expert"}, {"input": "analyze"})

        # Simulate tool error
        tool_error = Exception("Failed to load data: File not found")
        handler.on_chain_error(tool_error)

        # Allow async operations to complete
        await asyncio.sleep(0.1)

        # Verify error was broadcasted
        session = mock_session_manager.get_session.return_value
        error_calls = [call for call in session.broadcast_message.call_args_list
                      if call[0][0]["event_type"] == WSEventType.ERROR.value]

        assert len(error_calls) >= 1
        error_data = error_calls[0][0][0]["data"]
        assert "Failed to load data" in error_data["error"]
        assert error_data["agent"] == "bulk_rnaseq_expert"

    @pytest.mark.asyncio
    async def test_session_isolation(self, mock_session_manager):
        """Test that different sessions are properly isolated."""
        session_id_1 = uuid4()
        session_id_2 = uuid4()

        manager_1 = APICallbackManager(session_id_1, mock_session_manager)
        manager_2 = APICallbackManager(session_id_2, mock_session_manager)

        # Send messages from both sessions
        await manager_1.send_progress_update("Session 1 progress")
        await manager_2.send_progress_update("Session 2 progress")

        # Verify session manager was called with correct session IDs
        get_session_calls = mock_session_manager.get_session.call_args_list
        called_session_ids = [call[0][0] for call in get_session_calls]

        assert session_id_1 in called_session_ids
        assert session_id_2 in called_session_ids

    def test_callback_handler_with_langfuse(self, api_callback_manager):
        """Test callback handler integration with Langfuse when available."""
        with patch.dict('os.environ', {'LANGFUSE_PUBLIC_KEY': 'test_key'}):
            with patch('lobster.core.websocket_callback.CallbackHandler') as mock_langfuse:
                mock_langfuse_instance = Mock()
                mock_langfuse.return_value = mock_langfuse_instance

                callbacks = api_callback_manager.get_callbacks()

                assert len(callbacks) == 2
                assert api_callback_manager.websocket_handler in callbacks
                assert mock_langfuse_instance in callbacks

    def test_callback_handler_without_langfuse(self, api_callback_manager):
        """Test callback handler without Langfuse configuration."""
        with patch.dict('os.environ', {}, clear=True):
            callbacks = api_callback_manager.get_callbacks()

            assert len(callbacks) == 1
            assert api_callback_manager.websocket_handler in callbacks


class TestWebSocketResilience:
    """Test WebSocket connection resilience and error recovery."""

    @pytest.mark.asyncio
    async def test_broadcast_failure_resilience(self, mock_session_manager):
        """Test resilience when WebSocket broadcast fails."""
        session_id = uuid4()

        # Configure session to fail broadcast
        session = Mock()
        session.broadcast_message = AsyncMock(side_effect=Exception("Connection lost"))
        mock_session_manager.get_session.return_value = session

        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # Should not raise exception even when broadcast fails
        await handler._send_websocket_message(WSEventType.ERROR, {"test": "data"})

        # Verify attempt was made
        session.broadcast_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_not_found_resilience(self, mock_session_manager):
        """Test resilience when session is not found."""
        session_id = uuid4()

        # Configure session manager to return None (session not found)
        mock_session_manager.get_session.return_value = None

        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # Should not raise exception when session not found
        await handler._send_websocket_message(WSEventType.ANALYSIS_PROGRESS, {"test": "data"})

        # Verify session lookup was attempted
        mock_session_manager.get_session.assert_called_once_with(session_id)

    def test_logging_handler_without_callback_manager(self):
        """Test logging handler resilience without callback manager."""
        handler = WebSocketLoggingHandler(callback_manager=None)

        # Create a log record
        record = logging.LogRecord(
            name='lobster.tools.test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )

        # Should not raise exception
        handler.emit(record)

    def test_callback_handler_without_session_manager(self):
        """Test callback handler resilience without session manager."""
        session_id = uuid4()
        handler = WebSocketCallbackHandler(session_id, session_manager=None)

        # All callback methods should work without raising exceptions
        handler.on_chain_start({"name": "test_agent"}, {})
        handler.on_llm_start({}, [])
        handler.on_llm_new_token("token")
        handler.on_tool_start({"name": "test_tool"}, "input")
        handler.on_tool_end("output")
        handler.on_text("text")
        handler.on_chain_end({})
        handler.on_chain_error(Exception("test error"))


class TestWebSocketPerformance:
    """Test WebSocket performance characteristics."""

    @pytest.mark.asyncio
    async def test_high_frequency_messages(self, api_callback_manager, mock_session_manager):
        """Test handling of high frequency messages."""
        handler = api_callback_manager.websocket_handler

        start_time = time.time()

        # Send many messages rapidly
        for i in range(100):
            handler.on_llm_new_token(f"token_{i}")

        # Allow async operations to complete
        await asyncio.sleep(0.1)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete quickly (less than 1 second)
        assert duration < 1.0

    def test_message_deduplication_performance(self):
        """Test performance of message deduplication."""
        handler = WebSocketLoggingHandler()

        start_time = time.time()

        # Test deduplication with repeated messages
        for i in range(1000):
            handler._is_duplicate_message("repeated message")

        end_time = time.time()
        duration = end_time - start_time

        # Should complete quickly even with many duplicates
        assert duration < 0.1

    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load."""
        handler = WebSocketLoggingHandler()

        # Generate many unique messages to test cleanup
        for i in range(1000):
            handler._is_duplicate_message(f"unique_message_{i}_{time.time()}")

        # Memory should be bounded due to cleanup
        assert len(handler._recent_messages) < 1000

    @pytest.mark.asyncio
    async def test_concurrent_websocket_operations(self, mock_session_manager):
        """Test concurrent WebSocket operations."""
        session_id = uuid4()
        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # Create many concurrent async operations
        tasks = []
        for i in range(50):
            task = handler._send_websocket_message(
                WSEventType.CHAT_STREAM,
                {"token": f"token_{i}"}
            )
            tasks.append(task)

        # Wait for all to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should complete without errors
        session = mock_session_manager.get_session.return_value
        assert session.broadcast_message.call_count == 50


class TestWebSocketMessageFormat:
    """Test WebSocket message formatting and structure."""

    def test_ws_message_creation_and_serialization(self):
        """Test WebSocket message creation and serialization."""
        session_id = uuid4()
        event_type = WSEventType.ANALYSIS_PROGRESS
        data = {
            "status": "tool_start",
            "tool": "preprocessing_service",
            "agent": "singlecell_expert",
            "message": "Starting data preprocessing"
        }

        message = WSMessage(event_type=event_type, data=data, session_id=session_id)
        message_dict = message.dict()

        assert message_dict["event_type"] == event_type.value
        assert message_dict["data"] == data
        assert message_dict["session_id"] == str(session_id)

    def test_message_data_integrity(self):
        """Test that message data maintains integrity through serialization."""
        complex_data = {
            "nested": {
                "values": [1, 2, 3],
                "strings": ["a", "b", "c"]
            },
            "unicode": "Hello 🌍",
            "numbers": {
                "int": 42,
                "float": 3.14159
            }
        }

        message = WSMessage(
            event_type=WSEventType.DATA_UPDATED,
            data=complex_data,
            session_id=uuid4()
        )

        serialized = message.dict()

        # Data should be preserved exactly
        assert serialized["data"] == complex_data

    @pytest.mark.asyncio
    async def test_event_type_mapping_consistency(self, api_callback_manager, mock_session_manager):
        """Test that event types are consistently mapped."""
        handler = api_callback_manager.websocket_handler

        # Test different callback types and their expected event types
        test_cases = [
            ("on_chain_start", WSEventType.ANALYSIS_PROGRESS),
            ("on_chain_end", WSEventType.ANALYSIS_PROGRESS),
            ("on_tool_start", WSEventType.ANALYSIS_PROGRESS),
            ("on_tool_end", WSEventType.ANALYSIS_PROGRESS),
            ("on_llm_start", WSEventType.AGENT_THINKING),
            ("on_llm_end", WSEventType.AGENT_THINKING),
            ("on_llm_new_token", WSEventType.CHAT_STREAM),
            ("on_text", WSEventType.CHAT_STREAM),
            ("on_chain_error", WSEventType.ERROR),
        ]

        for callback_method, expected_event_type in test_cases:
            # Reset mock
            mock_session_manager.reset_mock()
            session = Mock()
            session.broadcast_message = AsyncMock()
            mock_session_manager.get_session.return_value = session

            # Call the appropriate callback method
            if callback_method == "on_chain_start":
                handler.on_chain_start({"name": "test"}, {})
            elif callback_method == "on_chain_end":
                handler.on_chain_end({})
            elif callback_method == "on_tool_start":
                handler.on_tool_start({"name": "test_tool"}, "input")
            elif callback_method == "on_tool_end":
                handler.on_tool_end("output")
            elif callback_method == "on_llm_start":
                handler.on_llm_start({}, [])
            elif callback_method == "on_llm_end":
                handler.on_llm_end(Mock())
            elif callback_method == "on_llm_new_token":
                handler.on_llm_new_token("token")
            elif callback_method == "on_text":
                handler.on_text("text")
            elif callback_method == "on_chain_error":
                handler.on_chain_error(Exception("test"))

            # Allow async operations to complete
            await asyncio.sleep(0.1)

            # Verify correct event type was used
            if session.broadcast_message.called:
                message_data = session.broadcast_message.call_args[0][0]
                assert message_data["event_type"] == expected_event_type.value