"""
Integration tests for Lobster client components.

These tests verify end-to-end functionality, client interactions,
and real-world usage scenarios for AgentClient and APIAgentClient.
"""

import asyncio
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from uuid import uuid4

import pytest
import pandas as pd
import numpy as np
import anndata as ad

from lobster.core.client import AgentClient
from lobster.core.api_client import APIAgentClient
from lobster.core.interfaces.base_client import BaseClient
from lobster.core.data_manager_v2 import DataManagerV2


# ===============================================================================
# Test Fixtures
# ===============================================================================

@pytest.fixture
def temp_integration_workspace():
    """Create temporary workspace for integration tests."""
    with tempfile.TemporaryDirectory(prefix="lobster_integration_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    n_obs, n_vars = 100, 50
    X = np.random.poisson(2, size=(n_obs, n_vars)).astype(np.float32)

    obs = pd.DataFrame({
        'cell_id': [f'cell_{i}' for i in range(n_obs)],
        'sample_id': np.random.choice(['sample_A', 'sample_B'], n_obs),
        'condition': np.random.choice(['treated', 'control'], n_obs),
        'n_genes': np.random.randint(1000, 5000, n_obs)
    }, index=[f'cell_{i}' for i in range(n_obs)])

    var = pd.DataFrame({
        'gene_id': [f'gene_{i}' for i in range(n_vars)],
        'gene_name': [f'GENE{i}' for i in range(n_vars)],
    }, index=[f'gene_{i}' for i in range(n_vars)])

    return ad.AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def sample_csv_data(temp_integration_workspace):
    """Create sample CSV data file."""
    data = pd.DataFrame({
        'sample': [f'sample_{i}' for i in range(20)],
        'gene1': np.random.randn(20),
        'gene2': np.random.randn(20),
        'gene3': np.random.randn(20),
        'treatment': np.random.choice(['A', 'B'], 20)
    })

    csv_path = temp_integration_workspace / "test_data.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


# ===============================================================================
# Client Integration Tests
# ===============================================================================

@pytest.mark.integration
class TestAgentClientIntegration:
    """Integration tests for AgentClient functionality."""

    def test_client_initialization_with_real_workspace(self, temp_integration_workspace):
        """Test client initialization with real workspace."""
        with patch('lobster.core.client.create_bioinformatics_graph') as mock_graph:
            mock_graph.return_value = Mock()

            client = AgentClient(workspace_path=temp_integration_workspace)

            # Verify workspace structure
            assert client.workspace_path == temp_integration_workspace
            assert client.workspace_path.exists()
            assert isinstance(client.data_manager, DataManagerV2)
            assert client.session_id.startswith("session_")

    def test_file_operations_integration(self, temp_integration_workspace, sample_csv_data):
        """Test integrated file operations."""
        with patch('lobster.core.client.create_bioinformatics_graph') as mock_graph:
            mock_graph.return_value = Mock()

            client = AgentClient(workspace_path=temp_integration_workspace)

            # Test file detection
            file_info = client.detect_file_type(sample_csv_data)
            assert file_info["type"] == "delimited_data"
            assert file_info["description"] == "Comma-separated values"

            # Test file location
            location_result = client.locate_file(sample_csv_data.name)
            assert location_result["found"] is True
            assert location_result["readable"] is True

            # Test file reading
            content = client.read_file(str(sample_csv_data))
            assert "sample,gene1,gene2,gene3,treatment" in content

            # Test workspace file listing
            files = client.list_workspace_files()
            file_names = [f["name"] for f in files]
            assert sample_csv_data.name in file_names

    def test_conversation_flow_integration(self, temp_integration_workspace):
        """Test complete conversation flow."""
        mock_graph = Mock()
        mock_events = [
            {"supervisor": {
                "messages": [Mock(content="I'll help you analyze your data.")]
            }},
            {"data_expert": {
                "messages": [Mock(content="Loading your dataset...")]
            }},
            {"supervisor": {
                "messages": [Mock(content="Analysis complete.")]
            }}
        ]
        mock_graph.stream.return_value = mock_events

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            client = AgentClient(workspace_path=temp_integration_workspace)

            # Test conversation flow
            queries = [
                "Hello, can you help me analyze my data?",
                "Load the dataset from my workspace",
                "Perform quality control analysis"
            ]

            results = []
            for query in queries:
                result = client.query(query)
                results.append(result)
                assert result["success"] is True
                assert "response" in result
                assert result["session_id"] == client.session_id

            # Verify conversation history
            history = client.get_conversation_history()
            assert len(history) == 6  # 3 user + 3 assistant messages

            # Test status after conversation
            status = client.get_status()
            assert status["message_count"] == 6
            assert status["session_id"] == client.session_id

    def test_session_export_integration(self, temp_integration_workspace):
        """Test session export functionality."""
        mock_data_manager = Mock(spec=DataManagerV2)
        mock_data_manager.has_data.return_value = False
        mock_data_manager.get_workspace_status.return_value = {
            "workspace_path": str(temp_integration_workspace),
            "modalities": 0,
            "plots": 0
        }

        with patch('lobster.core.client.create_bioinformatics_graph'):
            client = AgentClient(
                data_manager=mock_data_manager,
                workspace_path=temp_integration_workspace
            )

            # Add some conversation history
            client.query("Test query 1")
            client.query("Test query 2")

            # Export session
            export_path = client.export_session()

            assert export_path.exists()
            assert export_path.suffix == ".json"

            # Verify export content
            with open(export_path) as f:
                export_data = json.load(f)

            assert export_data["session_id"] == client.session_id
            assert len(export_data["conversation"]) == 4  # 2 user + 2 assistant
            assert "metadata" in export_data
            assert "exported_at" in export_data

    def test_error_recovery_integration(self, temp_integration_workspace):
        """Test error recovery in real scenarios."""
        mock_graph = Mock()

        # First call succeeds
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="First query successful")]}}
        ]

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            client = AgentClient(workspace_path=temp_integration_workspace)

            # First query succeeds
            result1 = client.query("First query")
            assert result1["success"] is True

            # Second query fails
            mock_graph.stream.side_effect = Exception("Network error")
            result2 = client.query("Second query")
            assert result2["success"] is False
            assert "Network error" in result2["error"]

            # Third query succeeds (recovery)
            mock_graph.stream.side_effect = None
            mock_graph.stream.return_value = [
                {"supervisor": {"messages": [Mock(content="Recovery successful")]}}
            ]
            result3 = client.query("Third query after recovery")
            assert result3["success"] is True
            assert "Recovery successful" in result3["response"]

            # Verify conversation history includes both success and failure
            history = client.get_conversation_history()
            assert len(history) == 6  # 3 user + 3 assistant messages


@pytest.mark.integration
@pytest.mark.asyncio
class TestAPIAgentClientIntegration:
    """Integration tests for APIAgentClient functionality."""

    async def test_api_client_initialization_integration(self, temp_integration_workspace):
        """Test APIAgentClient initialization with real dependencies."""
        session_id = uuid4()
        mock_session_manager = Mock()

        with patch('lobster.core.api_client.AgentClient') as mock_agent_client:
            with patch('lobster.core.api_client.APICallbackManager'):
                with patch('lobster.core.api_client.setup_websocket_logging'):
                    client = APIAgentClient(
                        session_id=session_id,
                        session_manager=mock_session_manager,
                        workspace_path=temp_integration_workspace
                    )

        assert client.session_id == session_id
        assert client.workspace_path == temp_integration_workspace
        assert client.workspace_path.exists()

    async def test_file_upload_integration(self, temp_integration_workspace):
        """Test file upload functionality."""
        session_id = uuid4()
        mock_session_manager = Mock()

        with patch('lobster.core.api_client.AgentClient'):
            with patch('lobster.core.api_client.APICallbackManager') as mock_callback_manager:
                mock_callback_instance = Mock()
                mock_callback_instance.send_data_update = AsyncMock()
                mock_callback_manager.return_value = mock_callback_instance

                with patch('lobster.core.api_client.setup_websocket_logging'):
                    mock_data_manager = Mock()
                    mock_data_manager.set_data = Mock()

                    client = APIAgentClient(
                        session_id=session_id,
                        session_manager=mock_session_manager,
                        workspace_path=temp_integration_workspace
                    )
                    client.data_manager = mock_data_manager
                    client.callback_manager = mock_callback_instance

        # Test file upload
        test_content = b"sample,value1,value2\nA,1,2\nB,3,4"
        result = await client.upload_file("test_upload.csv", test_content)

        assert result["success"] is True
        assert result["data_loaded"] is True
        assert "uploaded and loaded successfully" in result["message"]

        # Verify file was saved
        uploaded_file = temp_integration_workspace / "data" / "test_upload.csv"
        assert uploaded_file.exists()
        assert uploaded_file.read_bytes() == test_content

    async def test_workspace_files_integration(self, temp_integration_workspace, sample_csv_data):
        """Test workspace file listing integration."""
        session_id = uuid4()
        mock_session_manager = Mock()

        with patch('lobster.core.api_client.AgentClient'):
            with patch('lobster.core.api_client.APICallbackManager'):
                with patch('lobster.core.api_client.setup_websocket_logging'):
                    client = APIAgentClient(
                        session_id=session_id,
                        session_manager=mock_session_manager,
                        workspace_path=temp_integration_workspace
                    )

        # Test file listing
        files = client.list_workspace_files()

        # Should return at least the sample CSV file
        assert len(files) >= 1
        file_names = [f["name"] for f in files]
        assert sample_csv_data.name in file_names

        # Verify file metadata
        csv_file_info = next(f for f in files if f["name"] == sample_csv_data.name)
        assert csv_file_info["file_type"] == "csv"
        assert csv_file_info["size_bytes"] > 0


# ===============================================================================
# Performance and Stress Tests
# ===============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestClientPerformanceIntegration:
    """Performance tests for client components."""

    def test_concurrent_client_operations(self, temp_integration_workspace):
        """Test concurrent client operations."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Concurrent response")]}}
        ]

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            client = AgentClient(workspace_path=temp_integration_workspace)

            results = []
            errors = []

            def worker(worker_id):
                try:
                    result = client.query(f"Query from worker {worker_id}")
                    results.append((worker_id, result))
                except Exception as e:
                    errors.append((worker_id, e))

            # Create and start threads
            threads = []
            for i in range(10):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            # Verify results
            assert len(errors) == 0, f"Concurrent errors: {errors}"
            assert len(results) == 10

            # All results should be successful
            for worker_id, result in results:
                assert result["success"] is True
                assert result["session_id"] == client.session_id

    def test_large_file_handling(self, temp_integration_workspace):
        """Test handling of large files."""
        with patch('lobster.core.client.create_bioinformatics_graph'):
            client = AgentClient(workspace_path=temp_integration_workspace)

            # Create large CSV file
            large_data = pd.DataFrame({
                f'feature_{i}': np.random.randn(1000)
                for i in range(100)
            })

            large_file_path = temp_integration_workspace / "large_dataset.csv"
            large_data.to_csv(large_file_path, index=False)

            # Test file operations with large file
            start_time = time.time()

            # Test file detection
            file_info = client.detect_file_type(large_file_path)
            assert file_info["type"] == "delimited_data"

            # Test file location
            location_result = client.locate_file("large_dataset.csv")
            assert location_result["found"] is True

            # Test file listing
            files = client.list_workspace_files()
            large_file_info = next(f for f in files if f["name"] == "large_dataset.csv")
            assert large_file_info["size"] > 100000  # Should be reasonably large

            end_time = time.time()

            # Should complete within reasonable time (less than 5 seconds)
            assert (end_time - start_time) < 5.0

    def test_memory_usage_with_long_conversations(self, temp_integration_workspace):
        """Test memory usage with extensive conversation history."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Response")]}}
        ]

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            client = AgentClient(workspace_path=temp_integration_workspace)

            # Simulate long conversation
            for i in range(500):
                result = client.query(f"Query number {i}")
                assert result["success"] is True

            # Verify conversation history
            history = client.get_conversation_history()
            assert len(history) == 1000  # 500 user + 500 assistant messages

            # Test status with large history
            status = client.get_status()
            assert status["message_count"] == 1000

            # Test session export with large history
            export_path = client.export_session()
            assert export_path.exists()

            # Verify export is reasonable size (not too large)
            export_size = export_path.stat().st_size
            assert export_size < 10 * 1024 * 1024  # Less than 10MB


# ===============================================================================
# Edge Cases and Error Handling
# ===============================================================================

@pytest.mark.integration
class TestClientEdgeCasesIntegration:
    """Test edge cases and error scenarios."""

    def test_workspace_permission_issues(self):
        """Test handling of workspace permission issues."""
        # Test with non-writable directory (if possible to simulate)
        with tempfile.TemporaryDirectory() as temp_dir:
            restricted_path = Path(temp_dir) / "restricted"

            with patch('pathlib.Path.mkdir') as mock_mkdir:
                mock_mkdir.side_effect = PermissionError("Permission denied")

                with pytest.raises(PermissionError):
                    with patch('lobster.core.client.create_bioinformatics_graph'):
                        AgentClient(workspace_path=restricted_path)

    def test_corrupted_file_handling(self, temp_integration_workspace):
        """Test handling of corrupted or invalid files."""
        with patch('lobster.core.client.create_bioinformatics_graph'):
            client = AgentClient(workspace_path=temp_integration_workspace)

            # Create corrupted CSV file
            corrupted_file = temp_integration_workspace / "corrupted.csv"
            corrupted_file.write_text("invalid,csv,content\n\x00\x01\x02")

            # Test file detection should still work
            file_info = client.detect_file_type(corrupted_file)
            assert file_info["type"] == "delimited_data"

            # Test reading corrupted file
            content = client.read_file(str(corrupted_file))
            assert "invalid,csv,content" in content

    def test_network_timeout_simulation(self, temp_integration_workspace):
        """Test network timeout simulation."""
        mock_graph = Mock()

        # Simulate slow/hanging network call
        def slow_stream(*args, **kwargs):
            time.sleep(0.1)  # Small delay to simulate network
            return [{"supervisor": {"messages": [Mock(content="Delayed response")]}}]

        mock_graph.stream.side_effect = slow_stream

        with patch('lobster.core.client.create_bioinformatics_graph', return_value=mock_graph):
            client = AgentClient(workspace_path=temp_integration_workspace)

            start_time = time.time()
            result = client.query("Query with network delay")
            end_time = time.time()

            assert result["success"] is True
            assert (end_time - start_time) >= 0.1  # Should include the delay
            assert result["duration"] >= 0.1

    @pytest.mark.asyncio
    async def test_api_client_cleanup_integration(self, temp_integration_workspace):
        """Test proper cleanup of API client resources."""
        session_id = uuid4()
        mock_session_manager = Mock()

        with patch('lobster.core.api_client.AgentClient'):
            with patch('lobster.core.api_client.APICallbackManager'):
                with patch('lobster.core.api_client.setup_websocket_logging'):
                    with patch('lobster.core.api_client.remove_websocket_logging') as mock_remove:
                        client = APIAgentClient(
                            session_id=session_id,
                            session_manager=mock_session_manager,
                            workspace_path=temp_integration_workspace
                        )

                        # Set up mock logging handler
                        client.websocket_logging_handler = Mock()

                        # Test cleanup
                        await client.cleanup()

                        # Verify cleanup was called
                        mock_remove.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])