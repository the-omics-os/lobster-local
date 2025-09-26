"""
Performance and memory leak tests for Lobster client components.

These tests verify memory usage, resource cleanup, and performance
characteristics under various load conditions.
"""

import gc
import psutil
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import tempfile

import pytest
import numpy as np
import pandas as pd

from lobster.core.client import AgentClient
from lobster.core.api_client import APIAgentClient
from lobster.core.data_manager_v2 import DataManagerV2


# ===============================================================================
# Performance Test Utilities
# ===============================================================================

class MemoryMonitor:
    """Monitor memory usage during test execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            current_memory = self.process.memory_info().rss / 1024 / 1024
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            time.sleep(0.1)

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        return {
            "initial_mb": self.initial_memory,
            "current_mb": current_memory,
            "peak_mb": self.peak_memory,
            "increase_mb": current_memory - self.initial_memory,
            "peak_increase_mb": self.peak_memory - self.initial_memory
        }


@pytest.fixture
def memory_monitor():
    """Memory monitoring fixture."""
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    yield monitor
    monitor.stop_monitoring()


@pytest.fixture
def temp_perf_workspace():
    """Temporary workspace for performance tests."""
    with tempfile.TemporaryDirectory(prefix="lobster_perf_") as temp_dir:
        yield Path(temp_dir)


# ===============================================================================
# Memory Leak Tests
# ===============================================================================

@pytest.mark.performance
class TestClientMemoryLeaks:
    """Test for memory leaks in client components."""

    def test_agent_client_memory_leak(self, temp_perf_workspace, memory_monitor):
        """Test AgentClient for memory leaks during repeated operations."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Test response")]}}
        ]

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            clients = []

            # Create and use multiple clients
            for i in range(50):
                client = AgentClient(workspace_path=temp_perf_workspace / f"client_{i}")

                # Perform operations
                result = client.query(f"Test query {i}")
                assert result["success"] is True

                # Store weak reference to track garbage collection
                clients.append(weakref.ref(client))

                # Explicitly delete client
                del client

            # Force garbage collection
            gc.collect()
            time.sleep(0.1)  # Allow cleanup

            # Check memory usage
            memory_stats = memory_monitor.get_memory_stats()

            # Memory increase should be reasonable (less than 100MB for 50 clients)
            assert memory_stats["increase_mb"] < 100, f"Memory increase too high: {memory_stats}"

            # Most clients should be garbage collected
            alive_clients = sum(1 for ref in clients if ref() is not None)
            assert alive_clients < 10, f"Too many clients still alive: {alive_clients}"

    def test_conversation_history_memory(self, temp_perf_workspace, memory_monitor):
        """Test memory usage with large conversation history."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Response")]}}
        ]

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            client = AgentClient(workspace_path=temp_perf_workspace)

            # Generate large conversation
            for i in range(1000):
                client.query(f"Query {i}")

            memory_stats = memory_monitor.get_memory_stats()

            # Memory usage should be reasonable for 1000 messages
            assert memory_stats["increase_mb"] < 50, f"Memory usage too high: {memory_stats}"

            # Test memory after reset
            initial_reset_memory = memory_stats["current_mb"]
            client.reset()
            gc.collect()
            time.sleep(0.1)

            final_memory_stats = memory_monitor.get_memory_stats()
            memory_freed = initial_reset_memory - final_memory_stats["current_mb"]

            # Some memory should be freed after reset
            assert memory_freed > 0, "No memory freed after reset"

    def test_file_operations_memory(self, temp_perf_workspace, memory_monitor):
        """Test memory usage during file operations."""
        with patch('lobster.core.client.create_bioinformatics_graph'):
            client = AgentClient(workspace_path=temp_perf_workspace)

            # Create multiple test files
            for i in range(100):
                test_file = temp_perf_workspace / f"test_file_{i}.csv"
                data = pd.DataFrame({
                    'col1': np.random.randn(1000),
                    'col2': np.random.randn(1000)
                })
                data.to_csv(test_file, index=False)

            # Perform file operations
            for i in range(100):
                filename = f"test_file_{i}.csv"

                # Test file operations
                client.detect_file_type(temp_perf_workspace / filename)
                client.locate_file(filename)
                client.read_file(filename)

            memory_stats = memory_monitor.get_memory_stats()

            # Memory usage should be reasonable
            assert memory_stats["increase_mb"] < 200, f"File operations memory too high: {memory_stats}"


# ===============================================================================
# Performance Benchmarks
# ===============================================================================

@pytest.mark.performance
class TestClientPerformanceBenchmarks:
    """Performance benchmarks for client operations."""

    def test_query_processing_performance(self, temp_perf_workspace):
        """Benchmark query processing performance."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Benchmark response")]}}
        ]

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            client = AgentClient(workspace_path=temp_perf_workspace)

            # Benchmark single query
            start_time = time.time()
            result = client.query("Benchmark query")
            single_query_time = time.time() - start_time

            assert result["success"] is True
            assert single_query_time < 1.0, f"Single query too slow: {single_query_time}s"

            # Benchmark batch queries
            start_time = time.time()
            for i in range(100):
                result = client.query(f"Batch query {i}")
                assert result["success"] is True
            batch_time = time.time() - start_time

            avg_query_time = batch_time / 100
            assert avg_query_time < 0.1, f"Average query time too slow: {avg_query_time}s"

    def test_concurrent_query_performance(self, temp_perf_workspace):
        """Test performance under concurrent load."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Concurrent response")]}}
        ]

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            client = AgentClient(workspace_path=temp_perf_workspace)

            def worker_task(worker_id):
                """Worker task for concurrent testing."""
                start_time = time.time()
                result = client.query(f"Concurrent query from worker {worker_id}")
                end_time = time.time()
                return {
                    "worker_id": worker_id,
                    "success": result["success"],
                    "duration": end_time - start_time
                }

            # Test with different thread counts
            for num_threads in [5, 10, 20]:
                start_time = time.time()

                with ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(worker_task, i) for i in range(num_threads)]
                    results = [future.result() for future in as_completed(futures)]

                total_time = time.time() - start_time

                # All tasks should succeed
                assert all(r["success"] for r in results)

                # Total time should be reasonable
                assert total_time < 5.0, f"Concurrent execution too slow: {total_time}s with {num_threads} threads"

                # Average response time should be reasonable
                avg_response_time = sum(r["duration"] for r in results) / len(results)
                assert avg_response_time < 1.0, f"Average response time too slow: {avg_response_time}s"

    def test_file_operation_performance(self, temp_perf_workspace):
        """Benchmark file operations performance."""
        with patch('lobster.core.client.create_bioinformatics_graph'):
            client = AgentClient(workspace_path=temp_perf_workspace)

            # Create test files of various sizes
            file_sizes = [1000, 10000, 100000]  # rows
            test_files = []

            for size in file_sizes:
                filename = f"test_data_{size}.csv"
                file_path = temp_perf_workspace / filename

                data = pd.DataFrame({
                    f'col_{i}': np.random.randn(size)
                    for i in range(10)
                })
                data.to_csv(file_path, index=False)
                test_files.append((filename, size))

            # Benchmark file operations
            for filename, size in test_files:
                # File detection
                start_time = time.time()
                file_info = client.detect_file_type(temp_perf_workspace / filename)
                detection_time = time.time() - start_time

                assert file_info["type"] == "delimited_data"
                assert detection_time < 0.1, f"File detection too slow for {size} rows: {detection_time}s"

                # File location
                start_time = time.time()
                location_result = client.locate_file(filename)
                location_time = time.time() - start_time

                assert location_result["found"] is True
                assert location_time < 0.5, f"File location too slow for {size} rows: {location_time}s"

            # Benchmark file listing
            start_time = time.time()
            files = client.list_workspace_files()
            listing_time = time.time() - start_time

            assert len(files) >= len(test_files)
            assert listing_time < 1.0, f"File listing too slow: {listing_time}s"

    def test_memory_efficiency_under_load(self, temp_perf_workspace, memory_monitor):
        """Test memory efficiency under sustained load."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Load test response")]}}
        ]

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            client = AgentClient(workspace_path=temp_perf_workspace)

            # Sustained load test
            for batch in range(10):
                # Process batch of queries
                for i in range(50):
                    result = client.query(f"Load test batch {batch} query {i}")
                    assert result["success"] is True

                # Check memory after each batch
                memory_stats = memory_monitor.get_memory_stats()

                # Memory shouldn't grow excessively
                assert memory_stats["current_mb"] < memory_stats["initial_mb"] + 100, \
                    f"Memory growth too high after batch {batch}: {memory_stats}"

                # Force garbage collection between batches
                gc.collect()


# ===============================================================================
# Resource Cleanup Tests
# ===============================================================================

@pytest.mark.performance
class TestResourceCleanup:
    """Test proper resource cleanup and garbage collection."""

    def test_client_resource_cleanup(self, temp_perf_workspace):
        """Test that client resources are properly cleaned up."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Cleanup test")]}}
        ]

        client_refs = []

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            # Create multiple clients
            for i in range(20):
                workspace = temp_perf_workspace / f"client_{i}"
                workspace.mkdir(exist_ok=True)

                client = AgentClient(workspace_path=workspace)
                client.query(f"Test query {i}")

                # Store weak reference
                client_refs.append(weakref.ref(client))

                # Delete client reference
                del client

        # Force garbage collection
        gc.collect()
        time.sleep(0.1)

        # Check that most clients were garbage collected
        alive_clients = sum(1 for ref in client_refs if ref() is not None)
        assert alive_clients < 5, f"Too many clients not garbage collected: {alive_clients}"

    def test_thread_cleanup(self, temp_perf_workspace):
        """Test that background threads are properly cleaned up."""
        initial_thread_count = threading.active_count()

        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Thread test")]}}
        ]

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            clients = []

            # Create clients that might spawn threads
            for i in range(10):
                client = AgentClient(workspace_path=temp_perf_workspace / f"thread_test_{i}")
                client.query(f"Thread test {i}")
                clients.append(client)

            # Check thread count during operation
            during_thread_count = threading.active_count()

            # Clean up clients
            for client in clients:
                # Simulate cleanup operations
                if hasattr(client, 'cleanup'):
                    client.cleanup()
                del client

            clients.clear()

        # Force cleanup
        gc.collect()
        time.sleep(0.5)  # Allow thread cleanup

        final_thread_count = threading.active_count()

        # Thread count should return close to initial
        thread_increase = final_thread_count - initial_thread_count
        assert thread_increase <= 2, f"Too many threads left active: {thread_increase}"

    def test_file_handle_cleanup(self, temp_perf_workspace):
        """Test that file handles are properly closed."""
        with patch('lobster.core.client.create_bioinformatics_graph'):
            client = AgentClient(workspace_path=temp_perf_workspace)

            # Create many files
            for i in range(100):
                test_file = temp_perf_workspace / f"handle_test_{i}.txt"
                test_file.write_text(f"Test content {i}")

            # Perform many file operations
            for i in range(100):
                filename = f"handle_test_{i}.txt"

                # These operations should properly close file handles
                content = client.read_file(filename)
                assert f"Test content {i}" in content

                # File detection should also handle files properly
                file_info = client.detect_file_type(temp_perf_workspace / filename)
                assert file_info is not None

            # Test writing files
            for i in range(50):
                success = client.write_file(f"output_{i}.txt", f"Output content {i}")
                assert success is True

        # If we get here without "too many open files" error, file handles are managed properly


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])