"""
Performance and load tests for WebSocket messaging components.
Tests throughput, latency, memory usage, and scalability.
"""

import asyncio
import logging
import pytest
import time
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, AsyncMock
from uuid import uuid4
import psutil
import os

from lobster.api.models import WSEventType, WSMessage
from lobster.core.websocket_callback import WebSocketCallbackHandler, APICallbackManager
from lobster.core.websocket_logging_handler import (
    WebSocketLoggingHandler,
    setup_websocket_logging,
    remove_websocket_logging
)


class PerformanceMetrics:
    """Helper class to collect performance metrics."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_end = None
        self.message_count = 0

    def start(self):
        """Start performance measurement."""
        gc.collect()  # Force garbage collection
        self.start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    def stop(self):
        """Stop performance measurement."""
        self.end_time = time.time()
        gc.collect()  # Force garbage collection
        self.memory_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    @property
    def duration(self):
        """Total duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    @property
    def throughput(self):
        """Messages per second."""
        if self.duration > 0:
            return self.message_count / self.duration
        return 0

    @property
    def memory_delta(self):
        """Memory usage change in MB."""
        if self.memory_start and self.memory_end:
            return self.memory_end - self.memory_start
        return 0


@pytest.fixture
def fast_mock_session_manager():
    """Create a fast mock session manager for performance tests."""
    session_manager = Mock()
    session = Mock()
    # Use a simple mock that doesn't involve async operations for speed
    session.broadcast_message = Mock(return_value=None)
    session_manager.get_session.return_value = session
    return session_manager


class TestWebSocketCallbackPerformance:
    """Performance tests for WebSocket callback handling."""

    def test_callback_handler_throughput(self, fast_mock_session_manager):
        """Test callback handler throughput with high message volume."""
        session_id = uuid4()
        handler = WebSocketCallbackHandler(session_id, fast_mock_session_manager)

        metrics = PerformanceMetrics()
        metrics.start()

        # Send many callback messages
        message_count = 10000
        for i in range(message_count):
            handler.on_llm_new_token(f"token_{i}")
            if i % 100 == 0:
                handler.on_text(f"Progress: {i/message_count*100:.1f}%")

        metrics.message_count = message_count
        metrics.stop()

        print(f"Callback throughput: {metrics.throughput:.0f} messages/sec")
        print(f"Total duration: {metrics.duration:.3f} seconds")
        print(f"Memory delta: {metrics.memory_delta:.2f} MB")

        # Performance assertions
        assert metrics.throughput > 5000, f"Throughput too low: {metrics.throughput:.0f} msg/sec"
        assert metrics.memory_delta < 50, f"Memory usage too high: {metrics.memory_delta:.2f} MB"

    @pytest.mark.asyncio
    async def test_async_websocket_message_throughput(self, fast_mock_session_manager):
        """Test async WebSocket message sending throughput."""
        session_id = uuid4()
        handler = WebSocketCallbackHandler(session_id, fast_mock_session_manager)

        # Configure for async testing
        session = Mock()
        session.broadcast_message = AsyncMock()
        fast_mock_session_manager.get_session.return_value = session

        metrics = PerformanceMetrics()
        metrics.start()

        # Send many async messages
        message_count = 1000
        tasks = []

        for i in range(message_count):
            task = handler._send_websocket_message(
                WSEventType.CHAT_STREAM,
                {"token": f"token_{i}", "index": i}
            )
            tasks.append(task)

        # Wait for all messages to be sent
        await asyncio.gather(*tasks)

        metrics.message_count = message_count
        metrics.stop()

        print(f"Async message throughput: {metrics.throughput:.0f} messages/sec")
        print(f"Total duration: {metrics.duration:.3f} seconds")

        # Verify all messages were sent
        assert session.broadcast_message.call_count == message_count

        # Performance assertions
        assert metrics.throughput > 500, f"Async throughput too low: {metrics.throughput:.0f} msg/sec"

    def test_callback_handler_memory_stability(self, fast_mock_session_manager):
        """Test callback handler memory stability over extended use."""
        session_id = uuid4()
        handler = WebSocketCallbackHandler(session_id, fast_mock_session_manager)

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Simulate extended usage with periodic cleanup
        cycles = 10
        messages_per_cycle = 1000

        for cycle in range(cycles):
            for i in range(messages_per_cycle):
                handler.on_chain_start({"name": f"agent_{i}"}, {})
                handler.on_llm_new_token(f"token_{i}")
                handler.on_tool_start({"name": f"tool_{i}"}, f"input_{i}")
                handler.on_tool_end(f"output_{i}")
                handler.on_chain_end({})

            # Force garbage collection between cycles
            gc.collect()

            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory

            print(f"Cycle {cycle + 1}: Memory growth = {memory_growth:.2f} MB")

            # Memory growth should be bounded
            assert memory_growth < 100, f"Memory growth too high: {memory_growth:.2f} MB"

    def test_concurrent_callback_handling(self, fast_mock_session_manager):
        """Test concurrent callback handling from multiple threads."""
        session_id = uuid4()
        handler = WebSocketCallbackHandler(session_id, fast_mock_session_manager)

        metrics = PerformanceMetrics()
        metrics.start()

        def worker_thread(thread_id, message_count):
            """Worker function for concurrent testing."""
            for i in range(message_count):
                handler.on_llm_new_token(f"thread_{thread_id}_token_{i}")
                handler.on_text(f"thread_{thread_id}_text_{i}")

        # Run concurrent threads
        thread_count = 5
        messages_per_thread = 1000

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(worker_thread, i, messages_per_thread)
                for i in range(thread_count)
            ]

            # Wait for all threads to complete
            for future in as_completed(futures):
                future.result()

        metrics.message_count = thread_count * messages_per_thread
        metrics.stop()

        print(f"Concurrent throughput: {metrics.throughput:.0f} messages/sec")
        print(f"Total duration: {metrics.duration:.3f} seconds")

        # Performance assertions
        assert metrics.throughput > 2000, f"Concurrent throughput too low: {metrics.throughput:.0f} msg/sec"


class TestWebSocketLoggingPerformance:
    """Performance tests for WebSocket logging handler."""

    def test_logging_handler_throughput(self):
        """Test logging handler throughput with high log volume."""
        mock_callback_manager = Mock()
        mock_websocket_handler = Mock()
        mock_websocket_handler._schedule_websocket_message = Mock()
        mock_callback_manager.websocket_handler = mock_websocket_handler

        handler = WebSocketLoggingHandler(
            callback_manager=mock_callback_manager,
            level=logging.INFO
        )

        metrics = PerformanceMetrics()
        metrics.start()

        # Create many log records
        message_count = 10000
        for i in range(message_count):
            record = logging.LogRecord(
                name='lobster.tools.test_service',
                level=logging.INFO,
                pathname='test.py',
                lineno=i,
                msg=f'Log message {i}',
                args=(),
                exc_info=None
            )
            handler.emit(record)

        metrics.message_count = message_count
        metrics.stop()

        print(f"Logging throughput: {metrics.throughput:.0f} messages/sec")
        print(f"Total duration: {metrics.duration:.3f} seconds")
        print(f"Memory delta: {metrics.memory_delta:.2f} MB")

        # Performance assertions
        assert metrics.throughput > 8000, f"Logging throughput too low: {metrics.throughput:.0f} msg/sec"
        assert metrics.memory_delta < 20, f"Logging memory usage too high: {metrics.memory_delta:.2f} MB"

    def test_deduplication_performance(self):
        """Test performance of message deduplication under high load."""
        handler = WebSocketLoggingHandler()

        metrics = PerformanceMetrics()
        metrics.start()

        # Test with mostly duplicate messages (realistic scenario)
        unique_messages = 100
        total_checks = 10000

        messages = [f"Log message {i}" for i in range(unique_messages)]

        for i in range(total_checks):
            message = messages[i % unique_messages]
            handler._is_duplicate_message(message)

        metrics.message_count = total_checks
        metrics.stop()

        print(f"Deduplication throughput: {metrics.throughput:.0f} checks/sec")
        print(f"Recent messages count: {len(handler._recent_messages)}")

        # Performance assertions
        assert metrics.throughput > 50000, f"Deduplication too slow: {metrics.throughput:.0f} checks/sec"
        assert len(handler._recent_messages) <= unique_messages, "Memory leak in deduplication"

    def test_logging_memory_cleanup(self):
        """Test that logging handler properly cleans up old messages."""
        handler = WebSocketLoggingHandler()
        handler._dedupe_window = 0.1  # Short window for testing

        initial_time = time.time()

        # Add messages over time
        for i in range(1000):
            handler._recent_messages[f"message_{i}"] = initial_time + (i * 0.001)

        assert len(handler._recent_messages) == 1000

        # Trigger cleanup by checking a new message after the window
        time.sleep(0.15)  # Wait longer than dedupe window
        handler._is_duplicate_message("new_message")

        # Old messages should be cleaned up
        assert len(handler._recent_messages) < 1000
        print(f"Messages after cleanup: {len(handler._recent_messages)}")

    def test_concurrent_logging_performance(self):
        """Test concurrent logging performance from multiple threads."""
        mock_callback_manager = Mock()
        mock_websocket_handler = Mock()
        mock_websocket_handler._schedule_websocket_message = Mock()
        mock_callback_manager.websocket_handler = mock_websocket_handler

        handler = WebSocketLoggingHandler(
            callback_manager=mock_callback_manager,
            level=logging.INFO
        )

        metrics = PerformanceMetrics()
        metrics.start()

        def logging_worker(thread_id, log_count):
            """Worker function for concurrent logging."""
            for i in range(log_count):
                record = logging.LogRecord(
                    name=f'lobster.tools.thread_{thread_id}',
                    level=logging.INFO,
                    pathname='test.py',
                    lineno=i,
                    msg=f'Thread {thread_id} message {i}',
                    args=(),
                    exc_info=None
                )
                handler.emit(record)

        # Run concurrent logging threads
        thread_count = 5
        logs_per_thread = 2000

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(logging_worker, i, logs_per_thread)
                for i in range(thread_count)
            ]

            for future in as_completed(futures):
                future.result()

        metrics.message_count = thread_count * logs_per_thread
        metrics.stop()

        print(f"Concurrent logging throughput: {metrics.throughput:.0f} messages/sec")

        # Performance assertions
        assert metrics.throughput > 5000, f"Concurrent logging too slow: {metrics.throughput:.0f} msg/sec"


class TestWebSocketIntegrationPerformance:
    """Performance tests for integrated WebSocket components."""

    def test_full_integration_performance(self, fast_mock_session_manager):
        """Test performance of full WebSocket integration."""
        session_id = uuid4()
        manager = APICallbackManager(session_id, fast_mock_session_manager)

        # Set up logging integration
        logging_handler = setup_websocket_logging(manager, logging.INFO)

        try:
            metrics = PerformanceMetrics()
            metrics.start()

            # Simulate realistic workflow with mixed callbacks and logging
            callback_handler = manager.websocket_handler
            test_logger = logging.getLogger('lobster.tools.integration_test')

            workflow_cycles = 100
            for cycle in range(workflow_cycles):
                # Agent starts
                callback_handler.on_chain_start({"name": f"agent_{cycle}"}, {})
                test_logger.info(f"Starting agent cycle {cycle}")

                # LLM processing
                callback_handler.on_llm_start({"name": "gpt-4"}, [])
                for token in ["Analyzing", " data", " for", " cycle", f" {cycle}"]:
                    callback_handler.on_llm_new_token(token)
                callback_handler.on_llm_end(Mock())

                # Tool execution
                callback_handler.on_tool_start({"name": "process_data"}, "input")
                test_logger.info(f"Processing data for cycle {cycle}")
                callback_handler.on_tool_end("Processed successfully")

                # Agent completes
                callback_handler.on_chain_end({})
                test_logger.info(f"Completed cycle {cycle}")

            metrics.message_count = workflow_cycles * 10  # Approximate message count
            metrics.stop()

            print(f"Integration throughput: {metrics.throughput:.0f} messages/sec")
            print(f"Total duration: {metrics.duration:.3f} seconds")
            print(f"Memory delta: {metrics.memory_delta:.2f} MB")

            # Performance assertions
            assert metrics.throughput > 1000, f"Integration throughput too low: {metrics.throughput:.0f} msg/sec"
            assert metrics.memory_delta < 30, f"Integration memory usage too high: {metrics.memory_delta:.2f} MB"

        finally:
            remove_websocket_logging(logging_handler)

    @pytest.mark.asyncio
    async def test_async_message_batching_performance(self, fast_mock_session_manager):
        """Test performance of batched async message sending."""
        session_id = uuid4()
        handler = WebSocketCallbackHandler(session_id, fast_mock_session_manager)

        # Configure async mock
        session = Mock()
        session.broadcast_message = AsyncMock()
        fast_mock_session_manager.get_session.return_value = session

        metrics = PerformanceMetrics()
        metrics.start()

        # Send messages in batches
        batch_size = 100
        batch_count = 10

        for batch in range(batch_count):
            # Create batch of messages
            batch_tasks = []
            for i in range(batch_size):
                task = handler._send_websocket_message(
                    WSEventType.ANALYSIS_PROGRESS,
                    {"batch": batch, "message": i, "data": f"payload_{batch}_{i}"}
                )
                batch_tasks.append(task)

            # Send batch concurrently
            await asyncio.gather(*batch_tasks)

        metrics.message_count = batch_size * batch_count
        metrics.stop()

        print(f"Batched async throughput: {metrics.throughput:.0f} messages/sec")

        # Verify all messages were sent
        assert session.broadcast_message.call_count == batch_size * batch_count

        # Performance assertions
        assert metrics.throughput > 1000, f"Batched throughput too low: {metrics.throughput:.0f} msg/sec"

    def test_memory_usage_under_sustained_load(self, fast_mock_session_manager):
        """Test memory usage under sustained load over time."""
        session_id = uuid4()
        manager = APICallbackManager(session_id, fast_mock_session_manager)
        logging_handler = setup_websocket_logging(manager, logging.INFO)

        try:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            callback_handler = manager.websocket_handler
            test_logger = logging.getLogger('lobster.tools.memory_test')

            # Sustained load test
            load_duration = 10  # seconds
            start_time = time.time()
            message_count = 0

            while time.time() - start_time < load_duration:
                # Mixed workload
                callback_handler.on_llm_new_token(f"token_{message_count}")
                test_logger.info(f"Log message {message_count}")
                callback_handler.on_text(f"Text {message_count}")

                message_count += 3

                # Periodic memory checks
                if message_count % 1000 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_growth = current_memory - initial_memory
                    print(f"Messages: {message_count}, Memory growth: {memory_growth:.2f} MB")

                    # Memory should not grow excessively
                    assert memory_growth < 200, f"Memory growth too high: {memory_growth:.2f} MB"

            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            total_growth = final_memory - initial_memory

            print(f"Total messages: {message_count}")
            print(f"Total memory growth: {total_growth:.2f} MB")
            print(f"Messages per MB: {message_count / max(total_growth, 1):.0f}")

            # Final assertions
            assert total_growth < 150, f"Total memory growth too high: {total_growth:.2f} MB"

        finally:
            remove_websocket_logging(logging_handler)

    @pytest.mark.asyncio
    async def test_websocket_protocol_compliance_performance(self, fast_mock_session_manager):
        """Test WebSocket protocol compliance under high load."""
        session_id = uuid4()
        handler = WebSocketCallbackHandler(session_id, fast_mock_session_manager)

        # Configure mock to track message structure
        session = Mock()
        sent_messages = []

        async def track_message(message):
            sent_messages.append(message)

        session.broadcast_message = AsyncMock(side_effect=track_message)
        fast_mock_session_manager.get_session.return_value = session

        # Send various message types
        message_types = [
            (WSEventType.CHAT_STREAM, {"token": "test"}),
            (WSEventType.ANALYSIS_PROGRESS, {"status": "running"}),
            (WSEventType.ERROR, {"error": "test error"}),
            (WSEventType.DATA_UPDATED, {"dataset": "test"}),
            (WSEventType.PLOT_GENERATED, {"plot": "test.html"}),
        ]

        tasks = []
        for _ in range(200):  # 200 messages of each type
            for event_type, data in message_types:
                task = handler._send_websocket_message(event_type, data)
                tasks.append(task)

        # Send all messages
        await asyncio.gather(*tasks)

        # Verify message structure compliance
        assert len(sent_messages) == 200 * len(message_types)

        for message in sent_messages:
            # Every message should have required fields
            assert "event_type" in message
            assert "data" in message
            assert "session_id" in message

            # Event type should be valid
            assert message["event_type"] in [e.value for e in WSEventType]

            # Session ID should be present
            assert message["session_id"] == str(session_id)


if __name__ == "__main__":
    # Run performance tests independently
    import sys

    print("Running WebSocket Performance Tests...")
    print("=" * 50)

    # You can run specific performance tests here
    # This allows for manual performance profiling

    pytest.main([__file__, "-v", "-s"])