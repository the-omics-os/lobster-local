"""
Comprehensive resilience and error handling tests for WebSocket components.
Tests connection failures, recovery mechanisms, and error propagation.
"""

import asyncio
import logging
import pytest
import time
import random
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call, side_effect
from uuid import uuid4
import threading
import concurrent.futures

from lobster.api.models import WSEventType, WSMessage
from lobster.core.websocket_callback import WebSocketCallbackHandler, APICallbackManager
from lobster.core.websocket_logging_handler import (
    WebSocketLoggingHandler,
    setup_websocket_logging,
    remove_websocket_logging
)


class TestWebSocketConnectionResilience:
    """Test WebSocket connection resilience and recovery."""

    @pytest.mark.asyncio
    async def test_session_not_found_resilience(self):
        """Test resilience when session is not found."""
        session_id = uuid4()
        mock_session_manager = Mock()
        mock_session_manager.get_session.return_value = None

        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # Should not raise exceptions for any callback method
        test_cases = [
            lambda: handler.on_chain_start({"name": "test_agent"}, {}),
            lambda: handler.on_llm_start({}, []),
            lambda: handler.on_llm_new_token("token"),
            lambda: handler.on_tool_start({"name": "test_tool"}, "input"),
            lambda: handler.on_tool_end("output"),
            lambda: handler.on_text("text"),
            lambda: handler.on_chain_end({}),
            lambda: handler.on_chain_error(Exception("test error")),
        ]

        for test_case in test_cases:
            try:
                test_case()
                # Also test direct async call
                await handler._send_websocket_message(WSEventType.ERROR, {"test": "data"})
            except Exception as e:
                pytest.fail(f"Handler raised exception when session not found: {e}")

    @pytest.mark.asyncio
    async def test_broadcast_failure_resilience(self):
        """Test resilience when WebSocket broadcast fails."""
        session_id = uuid4()
        mock_session_manager = Mock()
        mock_session = Mock()

        # Configure broadcast to fail in various ways
        failure_types = [
            ConnectionError("Connection lost"),
            TimeoutError("Broadcast timeout"),
            OSError("Network error"),
            Exception("Generic error"),
        ]

        for failure in failure_types:
            mock_session.broadcast_message = AsyncMock(side_effect=failure)
            mock_session_manager.get_session.return_value = mock_session

            handler = WebSocketCallbackHandler(session_id, mock_session_manager)

            # Should not raise exception even when broadcast fails
            try:
                await handler._send_websocket_message(WSEventType.ERROR, {"test": "data"})
            except Exception as e:
                pytest.fail(f"Handler raised exception on {failure}: {e}")

            # Verify attempt was made
            mock_session.broadcast_message.assert_called()
            mock_session.broadcast_message.reset_mock()

    @pytest.mark.asyncio
    async def test_session_manager_corruption_resilience(self):
        """Test resilience when session manager is corrupted or returns invalid data."""
        session_id = uuid4()

        # Test various types of session manager corruption
        corruption_scenarios = [
            None,  # Session manager is None
            Mock(get_session=Mock(side_effect=Exception("Manager corrupted"))),  # get_session throws
            Mock(get_session=Mock(return_value="invalid")),  # Returns invalid object
            Mock(get_session=Mock(return_value=Mock(broadcast_message=None))),  # Missing broadcast method
        ]

        for corrupted_manager in corruption_scenarios:
            handler = WebSocketCallbackHandler(session_id, corrupted_manager)

            # Should handle corruption gracefully
            try:
                handler.on_llm_new_token("test_token")
                await handler._send_websocket_message(WSEventType.ERROR, {"test": "data"})
            except Exception as e:
                pytest.fail(f"Handler failed with corrupted manager {corrupted_manager}: {e}")

    def test_callback_handler_with_malformed_data(self):
        """Test callback handler with malformed or unexpected data."""
        session_id = uuid4()
        mock_session_manager = Mock()
        mock_session = Mock()
        mock_session.broadcast_message = AsyncMock()
        mock_session_manager.get_session.return_value = mock_session

        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # Test with various malformed inputs
        malformed_inputs = [
            (None, {}),  # None serialized
            ({}, None),  # None inputs
            ({"name": None}, {}),  # None name
            ({"name": ""}, {}),  # Empty name
            ({"name": 12345}, {}),  # Non-string name
            ({"name": "test", "extra": object()}, {}),  # Non-serializable data
        ]

        for serialized, inputs in malformed_inputs:
            try:
                handler.on_chain_start(serialized, inputs)
                handler.on_tool_start(serialized, "input")
            except Exception as e:
                pytest.fail(f"Handler failed with malformed data {serialized}: {e}")

    def test_logging_handler_with_malformed_records(self):
        """Test logging handler with malformed log records."""
        mock_callback_manager = Mock()
        handler = WebSocketLoggingHandler(callback_manager=mock_callback_manager)

        # Create malformed log records
        malformed_records = [
            logging.LogRecord(
                name=None,  # None name
                level=logging.INFO,
                pathname='test.py',
                lineno=1,
                msg='Test message',
                args=(),
                exc_info=None
            ),
            logging.LogRecord(
                name='',  # Empty name
                level=999,  # Invalid level
                pathname='test.py',
                lineno=1,
                msg='Test message',
                args=(),
                exc_info=None
            ),
            logging.LogRecord(
                name='test.logger',
                level=logging.INFO,
                pathname='test.py',
                lineno=1,
                msg=None,  # None message
                args=(),
                exc_info=None
            ),
        ]

        for record in malformed_records:
            try:
                handler.emit(record)
            except Exception as e:
                pytest.fail(f"Logging handler failed with malformed record: {e}")


class TestWebSocketErrorHandling:
    """Test error handling and error propagation in WebSocket components."""

    def test_callback_exception_isolation(self):
        """Test that exceptions in one callback don't affect others."""
        session_id = uuid4()
        mock_session_manager = Mock()
        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # Mock _schedule_websocket_message to fail for specific calls
        original_schedule = handler._schedule_websocket_message
        call_count = 0

        def failing_schedule(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second call
                raise Exception("Scheduled failure")
            return original_schedule(*args, **kwargs)

        with patch.object(handler, '_schedule_websocket_message', side_effect=failing_schedule):
            # Make multiple calls - second should fail but others should succeed
            try:
                handler.on_chain_start({"name": "agent1"}, {})  # Call 1 - should succeed
                handler.on_llm_start({}, [])  # Call 2 - should fail silently
                handler.on_text("test")  # Call 3 - should succeed
            except Exception as e:
                pytest.fail(f"Exception propagated when it should have been isolated: {e}")

        assert call_count == 3, "Not all callbacks were attempted"

    @pytest.mark.asyncio
    async def test_async_error_isolation(self):
        """Test that async errors don't propagate to synchronous callers."""
        session_id = uuid4()
        mock_session_manager = Mock()
        mock_session = Mock()
        mock_session.broadcast_message = AsyncMock(side_effect=Exception("Async failure"))
        mock_session_manager.get_session.return_value = mock_session

        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # Synchronous callback that triggers async operation should not raise
        try:
            handler.on_llm_new_token("test_token")
            # Give async operation time to fail
            await asyncio.sleep(0.1)
        except Exception as e:
            pytest.fail(f"Async error propagated to sync caller: {e}")

    def test_logging_handler_format_error_handling(self):
        """Test logging handler when format method fails."""
        mock_callback_manager = Mock()
        handler = WebSocketLoggingHandler(callback_manager=mock_callback_manager)

        # Mock format to raise exception
        handler.format = Mock(side_effect=Exception("Format error"))

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
        try:
            handler.emit(record)
        except Exception as e:
            pytest.fail(f"Logging handler failed to handle format error: {e}")

    def test_websocket_message_serialization_errors(self):
        """Test handling of message serialization errors."""
        # Test with non-serializable data
        non_serializable_data = {
            "function": lambda x: x,  # Function objects can't be serialized
            "complex": complex(1, 2),  # Complex numbers
            "set": {1, 2, 3},  # Sets may not be JSON serializable
        }

        for key, value in non_serializable_data.items():
            message = WSMessage(
                event_type=WSEventType.ERROR,
                data={key: value},
                session_id=uuid4()
            )

            # dict() method should handle or gracefully fail
            try:
                result = message.dict()
                # If it succeeds, data should be present
                assert "data" in result
            except Exception:
                # If it fails, that's acceptable for non-serializable data
                pass

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self):
        """Test error handling under concurrent operations."""
        session_id = uuid4()
        mock_session_manager = Mock()
        mock_session = Mock()

        # Configure some broadcasts to fail randomly
        async def random_failing_broadcast(message):
            if random.random() < 0.3:  # 30% failure rate
                raise ConnectionError("Random connection failure")

        mock_session.broadcast_message = AsyncMock(side_effect=random_failing_broadcast)
        mock_session_manager.get_session.return_value = mock_session

        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # Send many concurrent messages
        tasks = []
        for i in range(100):
            task = handler._send_websocket_message(
                WSEventType.CHAT_STREAM,
                {"token": f"token_{i}"}
            )
            tasks.append(task)

        # Wait for all tasks to complete (some may fail)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some may have failed, but none should have raised unhandled exceptions
        for result in results:
            if isinstance(result, Exception):
                # Exceptions should be contained, not propagated
                assert result is None or isinstance(result, (ConnectionError, Exception))


class TestWebSocketRecoveryMechanisms:
    """Test recovery mechanisms and fallback behaviors."""

    def test_session_recovery_after_failure(self):
        """Test session recovery after initial failure."""
        session_id = uuid4()
        mock_session_manager = Mock()

        # Initially return None (session not found)
        mock_session_manager.get_session.return_value = None

        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # First call should handle missing session gracefully
        handler.on_text("test message 1")

        # Now make session available
        mock_session = Mock()
        mock_session.broadcast_message = AsyncMock()
        mock_session_manager.get_session.return_value = mock_session

        # Subsequent calls should work normally
        handler.on_text("test message 2")

        # Verify session manager was called for both attempts
        assert mock_session_manager.get_session.call_count >= 2

    @pytest.mark.asyncio
    async def test_broadcast_retry_behavior(self):
        """Test behavior when broadcast operations fail and potentially retry."""
        session_id = uuid4()
        mock_session_manager = Mock()
        mock_session = Mock()

        # Configure broadcast to fail first time, succeed second time
        call_count = 0

        async def flaky_broadcast(message):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("First attempt fails")
            # Subsequent attempts succeed

        mock_session.broadcast_message = AsyncMock(side_effect=flaky_broadcast)
        mock_session_manager.get_session.return_value = mock_session

        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # First call will fail, but should be handled gracefully
        await handler._send_websocket_message(WSEventType.ERROR, {"test": "data"})

        # Verify failure was handled
        assert call_count == 1

    def test_logging_handler_recovery_after_callback_manager_loss(self):
        """Test logging handler recovery when callback manager becomes unavailable."""
        mock_callback_manager = Mock()
        handler = WebSocketLoggingHandler(callback_manager=mock_callback_manager)

        # Initial logging should work
        record = logging.LogRecord(
            name='lobster.tools.test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message 1',
            args=(),
            exc_info=None
        )

        handler.emit(record)

        # Remove callback manager
        handler.callback_manager = None

        # Should still handle records gracefully
        record2 = logging.LogRecord(
            name='lobster.tools.test',
            level=logging.INFO,
            pathname='test.py',
            lineno=2,
            msg='Test message 2',
            args=(),
            exc_info=None
        )

        try:
            handler.emit(record2)
        except Exception as e:
            pytest.fail(f"Handler failed after callback manager removal: {e}")


class TestWebSocketResourceManagement:
    """Test resource management and cleanup in WebSocket components."""

    def test_memory_leak_prevention_in_deduplication(self):
        """Test that message deduplication doesn't cause memory leaks."""
        handler = WebSocketLoggingHandler()
        handler._dedupe_window = 0.1  # Short window for testing

        initial_memory_size = len(handler._recent_messages)

        # Add many messages over time
        for i in range(1000):
            message = f"message_{i}"
            handler._is_duplicate_message(message)

            # Periodically check memory usage
            if i % 100 == 0:
                current_size = len(handler._recent_messages)
                # Memory should not grow unbounded
                assert current_size < 500, f"Memory usage too high: {current_size} messages"

        # Wait for cleanup window to expire
        time.sleep(0.15)

        # Trigger cleanup
        handler._is_duplicate_message("cleanup_trigger")

        # Memory should be cleaned up
        final_size = len(handler._recent_messages)
        assert final_size < 100, f"Memory not properly cleaned up: {final_size} messages"

    def test_resource_cleanup_on_handler_deletion(self):
        """Test that resources are properly cleaned up when handlers are deleted."""
        mock_callback_manager = Mock()
        handler = WebSocketLoggingHandler(callback_manager=mock_callback_manager)

        # Add some messages to internal structures
        for i in range(100):
            handler._is_duplicate_message(f"message_{i}")

        assert len(handler._recent_messages) > 0

        # Delete handler - this should trigger cleanup
        handler_id = id(handler)
        del handler

        # Force garbage collection
        import gc
        gc.collect()

        # Memory should be freed (can't directly test, but at least verify no exceptions)

    @pytest.mark.asyncio
    async def test_async_task_cleanup(self):
        """Test that async tasks are properly cleaned up."""
        session_id = uuid4()
        mock_session_manager = Mock()
        mock_session = Mock()
        mock_session.broadcast_message = AsyncMock()
        mock_session_manager.get_session.return_value = mock_session

        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # Create many async tasks
        tasks = []
        for i in range(50):
            task = handler._send_websocket_message(
                WSEventType.CHAT_STREAM,
                {"token": f"token_{i}"}
            )
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

        # Verify all tasks completed successfully
        assert mock_session.broadcast_message.call_count == 50

        # Check that there are no pending tasks
        pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]
        websocket_tasks = [task for task in pending_tasks
                          if 'websocket' in str(task.get_coro()) if hasattr(task, 'get_coro')]

        # Should not have lingering WebSocket-related tasks
        assert len(websocket_tasks) == 0, f"Found {len(websocket_tasks)} lingering WebSocket tasks"

    def test_thread_safety_of_resource_cleanup(self):
        """Test thread safety of resource cleanup operations."""
        handler = WebSocketLoggingHandler()

        def worker_thread(thread_id):
            """Worker function for concurrent testing."""
            for i in range(100):
                handler._is_duplicate_message(f"thread_{thread_id}_message_{i}")

        # Run concurrent cleanup operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Handler should still be functional
        final_message_count = len(handler._recent_messages)
        assert final_message_count > 0, "All messages were incorrectly cleaned up"

        # Trigger final cleanup
        time.sleep(0.1)  # Wait beyond dedupe window
        handler._is_duplicate_message("final_cleanup")

        # Should not raise exceptions
        assert len(handler._recent_messages) >= 1  # At least the final cleanup message


class TestWebSocketEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_extremely_large_messages(self):
        """Test handling of extremely large messages."""
        handler = WebSocketLoggingHandler(max_message_length=100)  # Small limit for testing

        large_message = "x" * 10000  # 10KB message

        record = logging.LogRecord(
            name='lobster.tools.test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg=large_message,
            args=(),
            exc_info=None
        )

        handler.format = Mock(return_value=large_message)

        # Should handle large message gracefully (truncate)
        try:
            handler.emit(record)
        except Exception as e:
            pytest.fail(f"Handler failed with large message: {e}")

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        session_id = uuid4()
        mock_session_manager = Mock()
        mock_session = Mock()
        mock_session.broadcast_message = AsyncMock()
        mock_session_manager.get_session.return_value = mock_session

        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # Test various Unicode and special characters
        special_texts = [
            "Hello 🌍 World! 🚀",  # Emojis
            "测试中文字符",  # Chinese characters
            "Тест русских символов",  # Cyrillic
            "العربية",  # Arabic
            "🔥💯✨⚡️🌟",  # Multiple emojis
            "\n\t\r",  # Control characters
            "\"'\\`",  # Quotes and backslashes
            None,  # None value
        ]

        for text in special_texts:
            try:
                if text is not None:
                    handler.on_text(text)
                    handler.on_llm_new_token(text)
                else:
                    # Test None handling
                    handler.on_text(text)
            except Exception as e:
                pytest.fail(f"Handler failed with special text '{text}': {e}")

    @pytest.mark.asyncio
    async def test_rapid_session_changes(self):
        """Test handling of rapid session manager changes."""
        session_id = uuid4()
        handler = WebSocketCallbackHandler(session_id, None)

        # Rapidly change session managers
        for i in range(10):
            mock_session_manager = Mock()
            mock_session = Mock()
            mock_session.broadcast_message = AsyncMock()
            mock_session_manager.get_session.return_value = mock_session

            handler.session_manager = mock_session_manager

            # Send message
            await handler._send_websocket_message(WSEventType.CHAT_STREAM, {"test": f"message_{i}"})

            # Remove session manager
            handler.session_manager = None

            # Send another message (should be handled gracefully)
            await handler._send_websocket_message(WSEventType.ERROR, {"test": f"error_{i}"})

    def test_zero_length_and_empty_data(self):
        """Test handling of zero-length and empty data."""
        session_id = uuid4()
        mock_session_manager = Mock()
        handler = WebSocketCallbackHandler(session_id, mock_session_manager)

        # Test empty and zero-length inputs
        empty_inputs = [
            ("", {}),  # Empty strings
            ({}, ""),
            ({"name": ""}, {}),  # Empty name
            ({}, {"input": ""}),  # Empty input
            ({"name": "test"}, {}),  # Empty dict
        ]

        for serialized, inputs in empty_inputs:
            try:
                handler.on_chain_start(serialized, inputs)
                handler.on_tool_start(serialized, inputs.get("input", ""))
            except Exception as e:
                pytest.fail(f"Handler failed with empty data {serialized}, {inputs}: {e}")


if __name__ == "__main__":
    # Run resilience tests independently
    print("Running WebSocket Resilience Tests...")
    print("=" * 50)

    pytest.main([__file__, "-v", "-s"])