"""
Comprehensive unit tests for WebSocket callback handling components.
Tests WebSocketCallbackHandler and APICallbackManager functionality.
"""

import asyncio
import logging
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from uuid import uuid4, UUID
from typing import Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from lobster.api.models import WSEventType, WSMessage
from lobster.core.websocket_callback import WebSocketCallbackHandler, APICallbackManager


class TestWebSocketCallbackHandler:
    """Test suite for WebSocketCallbackHandler."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.session_id = uuid4()
        self.mock_session_manager = Mock()
        self.mock_session = Mock()
        self.mock_session.broadcast_message = AsyncMock()

        # Configure session manager to return our mock session
        self.mock_session_manager.get_session.return_value = self.mock_session

        self.handler = WebSocketCallbackHandler(
            session_id=self.session_id,
            session_manager=self.mock_session_manager
        )

    def test_handler_initialization(self):
        """Test WebSocket callback handler initialization."""
        assert self.handler.session_id == self.session_id
        assert self.handler.session_manager == self.mock_session_manager
        assert self.handler.current_agent is None

    def test_handler_initialization_without_session_manager(self):
        """Test handler initialization without session manager."""
        handler = WebSocketCallbackHandler(session_id=self.session_id)
        assert handler.session_id == self.session_id
        assert handler.session_manager is None
        assert handler.current_agent is None

    @pytest.mark.asyncio
    async def test_send_websocket_message_success(self):
        """Test successful WebSocket message sending."""
        event_type = WSEventType.ANALYSIS_PROGRESS
        data = {"status": "test", "message": "test message"}

        await self.handler._send_websocket_message(event_type, data)

        # Verify session lookup
        self.mock_session_manager.get_session.assert_called_once_with(self.session_id)

        # Verify broadcast call
        self.mock_session.broadcast_message.assert_called_once()
        broadcast_data = self.mock_session.broadcast_message.call_args[0][0]

        assert broadcast_data["event_type"] == event_type.value
        assert broadcast_data["data"] == data
        assert broadcast_data["session_id"] == str(self.session_id)

    @pytest.mark.asyncio
    async def test_send_websocket_message_no_session_manager(self):
        """Test WebSocket message sending without session manager."""
        handler = WebSocketCallbackHandler(session_id=self.session_id)

        # Should not raise exception
        await handler._send_websocket_message(WSEventType.ERROR, {"test": "data"})

    @pytest.mark.asyncio
    async def test_send_websocket_message_session_not_found(self):
        """Test WebSocket message sending when session is not found."""
        self.mock_session_manager.get_session.return_value = None

        # Should not raise exception
        await self.handler._send_websocket_message(WSEventType.ERROR, {"test": "data"})

        # Verify session lookup was attempted
        self.mock_session_manager.get_session.assert_called_once_with(self.session_id)

    @pytest.mark.asyncio
    async def test_send_websocket_message_broadcast_error(self):
        """Test WebSocket message sending when broadcast fails."""
        self.mock_session.broadcast_message.side_effect = Exception("Broadcast error")

        # Should not raise exception (error handling)
        await self.handler._send_websocket_message(WSEventType.ERROR, {"test": "data"})

    def test_on_llm_start(self):
        """Test LLM start callback."""
        serialized = {"name": "test_llm"}
        prompts = ["test prompt"]

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_llm_start(serialized, prompts)

            mock_schedule.assert_called_once_with(
                WSEventType.AGENT_THINKING,
                {
                    "status": "llm_start",
                    "agent": "unknown",  # current_agent is None
                    "message": "Agent is thinking..."
                }
            )

    def test_on_llm_start_with_current_agent(self):
        """Test LLM start callback with current agent set."""
        self.handler.current_agent = "test_agent"
        serialized = {"name": "test_llm"}
        prompts = ["test prompt"]

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_llm_start(serialized, prompts)

            mock_schedule.assert_called_once_with(
                WSEventType.AGENT_THINKING,
                {
                    "status": "llm_start",
                    "agent": "test_agent",
                    "message": "Agent is thinking..."
                }
            )

    def test_on_llm_start_error_handling(self):
        """Test LLM start callback error handling."""
        with patch.object(self.handler, '_schedule_websocket_message', side_effect=Exception("Test error")):
            # Should not raise exception
            self.handler.on_llm_start({}, [])

    def test_on_llm_new_token(self):
        """Test LLM new token callback."""
        token = "test_token"
        self.handler.current_agent = "test_agent"

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_llm_new_token(token)

            mock_schedule.assert_called_once_with(
                WSEventType.CHAT_STREAM,
                {
                    "token": token,
                    "agent": "test_agent",
                    "type": "token"
                }
            )

    def test_on_llm_new_token_error_handling(self):
        """Test LLM new token callback error handling."""
        with patch.object(self.handler, '_schedule_websocket_message', side_effect=Exception("Test error")):
            # Should not raise exception
            self.handler.on_llm_new_token("token")

    def test_on_llm_end(self):
        """Test LLM end callback."""
        self.handler.current_agent = "test_agent"
        response = LLMResult(generations=[[]])

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_llm_end(response)

            mock_schedule.assert_called_once_with(
                WSEventType.AGENT_THINKING,
                {
                    "status": "llm_end",
                    "agent": "test_agent",
                    "message": "Agent finished thinking"
                }
            )

    def test_on_chain_start(self):
        """Test chain start callback."""
        serialized = {"name": "test_chain"}
        inputs = {"input": "test"}

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_chain_start(serialized, inputs)

            assert self.handler.current_agent == "test_chain"
            mock_schedule.assert_called_once_with(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "agent_start",
                    "agent": "test_chain",
                    "message": "Starting test_chain..."
                }
            )

    def test_on_chain_start_none_serialized(self):
        """Test chain start callback with None serialized."""
        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_chain_start(None, {})

            assert self.handler.current_agent == "unknown"
            mock_schedule.assert_called_once_with(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "agent_start",
                    "agent": "unknown",
                    "message": "Starting unknown..."
                }
            )

    def test_on_chain_end(self):
        """Test chain end callback."""
        self.handler.current_agent = "test_agent"
        outputs = {"output": "test"}

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_chain_end(outputs)

            mock_schedule.assert_called_once_with(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "agent_end",
                    "agent": "test_agent",
                    "message": "Completed test_agent"
                }
            )

    def test_on_chain_error(self):
        """Test chain error callback."""
        self.handler.current_agent = "test_agent"
        error = Exception("Test error")

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_chain_error(error)

            mock_schedule.assert_called_once_with(
                WSEventType.ERROR,
                {
                    "status": "error",
                    "agent": "test_agent",
                    "error": "Test error",
                    "message": "Error in test_agent: Test error"
                }
            )

    def test_on_tool_start(self):
        """Test tool start callback."""
        self.handler.current_agent = "test_agent"
        serialized = {"name": "test_tool"}
        input_str = "test input"

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_tool_start(serialized, input_str)

            mock_schedule.assert_called_once_with(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "tool_start",
                    "tool": "test_tool",
                    "agent": "test_agent",
                    "message": "Using tool: test_tool"
                }
            )

    def test_on_tool_start_none_serialized(self):
        """Test tool start callback with None serialized."""
        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_tool_start(None, "input")

            mock_schedule.assert_called_once_with(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "tool_start",
                    "tool": "unknown_tool",
                    "agent": "unknown",
                    "message": "Using tool: unknown_tool"
                }
            )

    def test_on_tool_end_string_output(self):
        """Test tool end callback with string output."""
        output = "Simple string output"

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_tool_end(output)

            mock_schedule.assert_called_once_with(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "tool_end",
                    "agent": "unknown",
                    "message": "Tool completed",
                    "output_preview": "Simple string output"
                }
            )

    def test_on_tool_end_long_output(self):
        """Test tool end callback with long output (truncation)."""
        output = "x" * 300  # Long output to test truncation

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_tool_end(output)

            # Check that output was truncated
            expected_preview = "x" * 200 + "..."
            mock_schedule.assert_called_once_with(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "tool_end",
                    "agent": "unknown",
                    "message": "Tool completed",
                    "output_preview": expected_preview
                }
            )

    def test_on_tool_end_object_with_content(self):
        """Test tool end callback with object having content attribute."""
        mock_output = Mock()
        mock_output.content = "Tool message content"

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_tool_end(mock_output)

            mock_schedule.assert_called_once_with(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "tool_end",
                    "agent": "unknown",
                    "message": "Tool completed",
                    "output_preview": "Tool message content"
                }
            )

    def test_on_tool_end_conversion_error(self):
        """Test tool end callback when output conversion fails."""
        # Create a mock that will fail during string conversion
        mock_output = Mock()
        # Remove content attribute
        if hasattr(mock_output, 'content'):
            del mock_output.content

        # Create a mock that raises exception on str()
        def failing_str():
            raise Exception("Conversion error")

        mock_output.__str__ = Mock(side_effect=failing_str)

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_tool_end(mock_output)

            mock_schedule.assert_called_once_with(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "tool_end",
                    "agent": "unknown",
                    "message": "Tool completed",
                    "output_preview": "Tool output conversion failed"
                }
            )

    def test_on_text(self):
        """Test text callback."""
        text = "Agent response text"
        self.handler.current_agent = "test_agent"

        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_text(text)

            mock_schedule.assert_called_once_with(
                WSEventType.CHAT_STREAM,
                {
                    "content": text,
                    "agent": "test_agent",
                    "type": "text"
                }
            )

    def test_on_text_empty_text(self):
        """Test text callback with empty text."""
        with patch.object(self.handler, '_schedule_websocket_message') as mock_schedule:
            self.handler.on_text("")
            self.handler.on_text("   ")  # Whitespace only

            # Should not call schedule for empty text
            mock_schedule.assert_not_called()

    def test_on_text_error_handling(self):
        """Test text callback error handling."""
        with patch.object(self.handler, '_schedule_websocket_message', side_effect=Exception("Test error")):
            # Should not raise exception
            self.handler.on_text("test text")

    def test_schedule_websocket_message_with_event_loop(self):
        """Test scheduling WebSocket message with running event loop."""
        mock_loop = Mock()
        mock_task = Mock()
        mock_loop.create_task.return_value = mock_task

        with patch('asyncio.get_running_loop', return_value=mock_loop):
            self.handler._schedule_websocket_message(WSEventType.ERROR, {"test": "data"})

            mock_loop.create_task.assert_called_once()

    def test_schedule_websocket_message_no_event_loop(self):
        """Test scheduling WebSocket message without event loop."""
        with patch('asyncio.get_running_loop', side_effect=RuntimeError("No event loop")):
            # Should not raise exception
            self.handler._schedule_websocket_message(WSEventType.ERROR, {"test": "data"})

    def test_schedule_websocket_message_exception(self):
        """Test scheduling WebSocket message with exception."""
        with patch('asyncio.get_running_loop', side_effect=Exception("Unexpected error")):
            # Should not raise exception
            self.handler._schedule_websocket_message(WSEventType.ERROR, {"test": "data"})


class TestAPICallbackManager:
    """Test suite for APICallbackManager."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.session_id = uuid4()
        self.mock_session_manager = Mock()
        self.manager = APICallbackManager(
            session_id=self.session_id,
            session_manager=self.mock_session_manager
        )

    def test_manager_initialization(self):
        """Test API callback manager initialization."""
        assert self.manager.session_id == self.session_id
        assert self.manager.session_manager == self.mock_session_manager
        assert isinstance(self.manager.websocket_handler, WebSocketCallbackHandler)
        assert self.manager.websocket_handler.session_id == self.session_id

    def test_get_callbacks_basic(self):
        """Test getting basic callbacks."""
        with patch.dict('os.environ', {}, clear=True):
            callbacks = self.manager.get_callbacks()

            assert len(callbacks) == 1
            assert self.manager.websocket_handler in callbacks

    def test_get_callbacks_with_langfuse(self):
        """Test getting callbacks with Langfuse enabled."""
        mock_langfuse_callback = Mock()

        with patch.dict('os.environ', {'LANGFUSE_PUBLIC_KEY': 'test_key'}):
            with patch('langfuse.langchain.CallbackHandler', return_value=mock_langfuse_callback):
                callbacks = self.manager.get_callbacks()

                assert len(callbacks) == 2
                assert self.manager.websocket_handler in callbacks
                assert mock_langfuse_callback in callbacks

    def test_get_callbacks_langfuse_import_error(self):
        """Test getting callbacks with Langfuse import error."""
        with patch.dict('os.environ', {'LANGFUSE_PUBLIC_KEY': 'test_key'}):
            with patch('langfuse.langchain.CallbackHandler', side_effect=ImportError):
                callbacks = self.manager.get_callbacks()

                # Should fall back to just WebSocket handler
                assert len(callbacks) == 1
                assert self.manager.websocket_handler in callbacks

    @pytest.mark.asyncio
    async def test_send_progress_update(self):
        """Test sending progress update."""
        message = "Test progress"
        details = {"step": 1, "total": 5}

        with patch.object(self.manager.websocket_handler, '_send_websocket_message') as mock_send:
            await self.manager.send_progress_update(message, details)

            mock_send.assert_called_once_with(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "custom_update",
                    "message": message,
                    "details": details,
                    "agent": "system"
                }
            )

    @pytest.mark.asyncio
    async def test_send_progress_update_no_details(self):
        """Test sending progress update without details."""
        message = "Test progress"

        with patch.object(self.manager.websocket_handler, '_send_websocket_message') as mock_send:
            await self.manager.send_progress_update(message)

            mock_send.assert_called_once_with(
                WSEventType.ANALYSIS_PROGRESS,
                {
                    "status": "custom_update",
                    "message": message,
                    "details": {},
                    "agent": "system"
                }
            )

    @pytest.mark.asyncio
    async def test_send_data_update(self):
        """Test sending data update notification."""
        dataset_info = {"name": "test_dataset", "rows": 1000, "columns": 20}

        with patch.object(self.manager.websocket_handler, '_send_websocket_message') as mock_send:
            await self.manager.send_data_update(dataset_info)

            mock_send.assert_called_once_with(
                WSEventType.DATA_UPDATED,
                {
                    "status": "data_loaded",
                    "dataset": dataset_info,
                    "message": "New dataset loaded"
                }
            )

    @pytest.mark.asyncio
    async def test_send_plot_generated(self):
        """Test sending plot generated notification."""
        plot_info = {"type": "scatter", "file": "plot.html", "title": "Test Plot"}

        with patch.object(self.manager.websocket_handler, '_send_websocket_message') as mock_send:
            await self.manager.send_plot_generated(plot_info)

            mock_send.assert_called_once_with(
                WSEventType.PLOT_GENERATED,
                {
                    "status": "plot_created",
                    "plot": plot_info,
                    "message": "New plot generated"
                }
            )


class TestWSMessage:
    """Test suite for WSMessage model."""

    def test_ws_message_initialization(self):
        """Test WebSocket message initialization."""
        event_type = WSEventType.ANALYSIS_PROGRESS
        data = {"status": "test", "message": "test message"}
        session_id = "test-session-id"

        message = WSMessage(event_type=event_type, data=data, session_id=session_id)

        assert message.event_type == event_type
        assert message.data == data
        assert message.session_id == session_id

    def test_ws_message_dict_conversion(self):
        """Test WebSocket message dict conversion."""
        event_type = WSEventType.ERROR
        data = {"error": "test error", "code": 500}
        session_id = uuid4()

        message = WSMessage(event_type=event_type, data=data, session_id=session_id)
        message_dict = message.dict()

        assert message_dict["event_type"] == event_type.value
        assert message_dict["data"] == data
        assert message_dict["session_id"] == str(session_id)

    def test_ws_message_dict_conversion_no_session_id(self):
        """Test WebSocket message dict conversion without session ID."""
        event_type = WSEventType.CHAT_STREAM
        data = {"content": "test content"}

        message = WSMessage(event_type=event_type, data=data)
        message_dict = message.dict()

        assert message_dict["event_type"] == event_type.value
        assert message_dict["data"] == data
        assert message_dict["session_id"] is None


class TestWebSocketIntegration:
    """Integration tests for WebSocket components."""

    def setup_method(self):
        """Set up test fixtures for integration tests."""
        self.session_id = uuid4()
        self.mock_session_manager = Mock()
        self.mock_session = Mock()
        self.mock_session.broadcast_message = AsyncMock()
        self.mock_session_manager.get_session.return_value = self.mock_session

    @pytest.mark.asyncio
    async def test_full_callback_flow(self):
        """Test full callback flow from handler through manager."""
        manager = APICallbackManager(self.session_id, self.mock_session_manager)
        handler = manager.websocket_handler

        # Simulate LLM workflow
        handler.on_chain_start({"name": "test_agent"}, {"input": "test"})
        handler.on_llm_start({"name": "test_llm"}, ["test prompt"])
        handler.on_llm_new_token("Hello")
        handler.on_llm_new_token(" ")
        handler.on_llm_new_token("World")
        handler.on_llm_end(LLMResult(generations=[[]]))
        handler.on_tool_start({"name": "test_tool"}, "tool input")
        handler.on_tool_end("tool output")
        handler.on_chain_end({"output": "final result"})

        # Let async tasks complete
        await asyncio.sleep(0.1)

        # Verify current agent was set
        assert handler.current_agent == "test_agent"

    @pytest.mark.asyncio
    async def test_error_flow(self):
        """Test error handling flow."""
        manager = APICallbackManager(self.session_id, self.mock_session_manager)
        handler = manager.websocket_handler

        # Set up agent
        handler.on_chain_start({"name": "error_agent"}, {"input": "test"})

        # Simulate error
        error = Exception("Test error occurred")
        handler.on_chain_error(error)

        # Let async tasks complete
        await asyncio.sleep(0.1)

        assert handler.current_agent == "error_agent"

    def test_concurrent_callbacks(self):
        """Test handling concurrent callbacks safely."""
        handler = WebSocketCallbackHandler(self.session_id, self.mock_session_manager)

        # Simulate rapid concurrent callbacks
        for i in range(100):
            handler.on_llm_new_token(f"token_{i}")
            handler.on_text(f"text_{i}")

        # Should complete without errors
        assert True

    @pytest.mark.asyncio
    async def test_memory_cleanup(self):
        """Test that handlers don't hold unnecessary references."""
        handler = WebSocketCallbackHandler(self.session_id, self.mock_session_manager)

        # Create some activity
        handler.on_chain_start({"name": "test_agent"}, {})
        handler.on_llm_start({}, [])
        handler.on_text("test")

        # Clear references
        handler.session_manager = None

        # Should not cause errors
        handler.on_text("more test")
        await asyncio.sleep(0.1)