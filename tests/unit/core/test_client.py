"""
Comprehensive unit tests for AgentClient and related client functionality.

This module provides thorough testing of the AgentClient class and related components,
covering all major functionality areas including initialization, query processing,
cloud/local switching, session management, WebSocket streaming, error handling,
and BaseClient interface compliance.

Test coverage target: 95%+ with meaningful assertions and edge case coverage.
"""

import asyncio
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator
from unittest.mock import Mock, MagicMock, patch, AsyncMock, call
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest
import anndata as ad
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from pytest_mock import MockerFixture

from lobster.core.client import AgentClient
from lobster.core.api_client import APIAgentClient
from lobster.core.interfaces.base_client import BaseClient
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.websocket_callback import APICallbackManager

from tests.mock_data.generators import (
    generate_synthetic_single_cell,
    generate_synthetic_bulk_rnaseq,
    generate_synthetic_proteomics,
    generate_mock_geo_response,
    generate_test_workspace_state
)
from tests.mock_data.base import MEDIUM_DATASET_CONFIG


# ===============================================================================
# Test Fixtures
# ===============================================================================

@pytest.fixture
def mock_data_manager_v2():
    """Create a mock DataManagerV2 with comprehensive behavior."""
    mock_dm = Mock(spec=DataManagerV2)
    
    # Mock basic properties
    mock_dm.modalities = {}
    mock_dm.metadata_store = {}
    mock_dm.latest_plots = []
    mock_dm.tool_usage_history = []
    mock_dm.workspace_path = Path("/tmp/test_workspace")
    mock_dm.data_dir = Path("/tmp/test_workspace/data")
    mock_dm.exports_dir = Path("/tmp/test_workspace/exports")
    mock_dm.cache_dir = Path("/tmp/test_workspace/cache")
    
    # Mock methods with realistic behavior
    mock_dm.list_modalities.return_value = list(mock_dm.modalities.keys())
    mock_dm.get_modality.side_effect = lambda name: mock_dm.modalities.get(name)
    mock_dm.has_data.side_effect = lambda: len(mock_dm.modalities) > 0
    mock_dm.get_data_summary.return_value = {
        "modality_count": len(mock_dm.modalities),
        "total_cells": 1000,
        "total_features": 2000
    }
    mock_dm.get_latest_plots.return_value = []
    mock_dm.get_workspace_status.return_value = {
        "workspace_path": str(mock_dm.workspace_path),
        "modalities": len(mock_dm.modalities),
        "plots": len(mock_dm.latest_plots)
    }
    mock_dm.create_data_package.return_value = "/tmp/test_workspace/exports/package.zip"
    mock_dm.log_tool_usage.return_value = None
    
    return mock_dm


@pytest.fixture
def mock_langgraph_graph():
    """Create a mock LangGraph with realistic stream behavior."""
    mock_graph = Mock()
    
    def mock_stream(*args, **kwargs):
        """Mock streaming behavior that yields realistic events."""
        events = [
            {"supervisor": {
                "messages": [AIMessage(content="I'll help you with that analysis.")]
            }},
            {"data_expert": {
                "messages": [AIMessage(content="Loading your dataset...")]
            }},
            {"supervisor": {
                "messages": [AIMessage(content="Analysis complete. Here are the results.")]
            }},
            {"__end__": {}}
        ]
        for event in events:
            yield event
    
    mock_graph.stream.side_effect = mock_stream
    return mock_graph


@pytest.fixture
def mock_create_bioinformatics_graph(mock_langgraph_graph):
    """Mock the graph creation function."""
    with patch('lobster.core.client.create_bioinformatics_graph') as mock_create:
        mock_create.return_value = mock_langgraph_graph
        yield mock_create


@pytest.fixture
def agent_client_config():
    """Configuration for AgentClient tests."""
    return {
        "session_id": "test_session_123",
        "enable_reasoning": True,
        "enable_langfuse": False,
        "manual_model_params": {"temperature": 0.1, "max_tokens": 1000}
    }


@pytest.fixture
def mock_session_manager():
    """Mock session manager for APIAgentClient."""
    manager = Mock()
    manager.broadcast_to_session = AsyncMock()
    manager.get_session = Mock()
    return manager


@pytest.fixture
def synthetic_test_data():
    """Generate synthetic test data for various scenarios."""
    return {
        "single_cell": generate_synthetic_single_cell(n_cells=100, n_genes=500),
        "bulk_rnaseq": generate_synthetic_bulk_rnaseq(n_samples=12, n_genes=500),
        "proteomics": generate_synthetic_proteomics(n_samples=24, n_proteins=100)
    }


# ===============================================================================
# AgentClient Core Functionality Tests
# ===============================================================================

@pytest.mark.unit
class TestAgentClientInitialization:
    """Test AgentClient initialization and configuration."""
    
    def test_basic_initialization(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test basic AgentClient initialization."""
        client = AgentClient(workspace_path=temp_workspace)
        
        assert client.session_id.startswith("session_")
        assert client.workspace_path == temp_workspace
        assert client.enable_reasoning is True
        assert isinstance(client.data_manager, DataManagerV2)
        assert isinstance(client.messages, list)
        assert len(client.messages) == 0
        assert isinstance(client.metadata, dict)
        assert "created_at" in client.metadata
        assert "session_id" in client.metadata
        
        # Verify graph creation was called
        mock_create_bioinformatics_graph.assert_called_once()
    
    def test_initialization_with_custom_params(self, temp_workspace, mock_create_bioinformatics_graph, 
                                               agent_client_config, mock_data_manager_v2):
        """Test initialization with custom parameters."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            session_id=agent_client_config["session_id"],
            enable_reasoning=agent_client_config["enable_reasoning"],
            enable_langfuse=agent_client_config["enable_langfuse"],
            workspace_path=temp_workspace,
            manual_model_params=agent_client_config["manual_model_params"]
        )
        
        assert client.session_id == agent_client_config["session_id"]
        assert client.enable_reasoning == agent_client_config["enable_reasoning"]
        assert client.data_manager == mock_data_manager_v2
        assert client.workspace_path == temp_workspace
        
        # Verify graph creation with custom parameters
        mock_create_bioinformatics_graph.assert_called_once()
        args, kwargs = mock_create_bioinformatics_graph.call_args
        assert kwargs["data_manager"] == mock_data_manager_v2
        assert kwargs["manual_model_params"] == agent_client_config["manual_model_params"]
    
    @patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "test_key"})
    def test_initialization_with_langfuse(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test initialization with Langfuse enabled."""
        with patch('lobster.core.client.LangfuseCallback') as mock_langfuse:
            mock_callback = Mock()
            mock_langfuse.return_value = mock_callback
            
            client = AgentClient(
                workspace_path=temp_workspace,
                enable_langfuse=True
            )
            
            assert mock_callback in client.callbacks
    
    def test_initialization_with_custom_callbacks(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test initialization with custom callbacks."""
        custom_callback = Mock()
        
        client = AgentClient(
            workspace_path=temp_workspace,
            custom_callbacks=[custom_callback]
        )
        
        assert custom_callback in client.callbacks
    
    def test_workspace_creation(self, mock_create_bioinformatics_graph):
        """Test that workspace directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "new_workspace"
            assert not workspace_path.exists()
            
            client = AgentClient(workspace_path=workspace_path)
            
            assert workspace_path.exists()
            assert workspace_path.is_dir()


@pytest.mark.unit
class TestAgentClientQueryProcessing:
    """Test query processing functionality."""
    
    def test_query_basic_flow(self, temp_workspace, mock_create_bioinformatics_graph, 
                              mock_data_manager_v2):
        """Test basic query processing flow."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        user_input = "Analyze my single-cell data"
        result = client.query(user_input)
        
        assert result["success"] is True
        assert "response" in result
        assert result["session_id"] == client.session_id
        assert "duration" in result
        assert "events_count" in result
        assert len(client.messages) == 2  # User message + AI response
        assert isinstance(client.messages[0], HumanMessage)
        assert client.messages[0].content == user_input
    
    def test_query_with_reasoning_enabled(self, temp_workspace, mock_create_bioinformatics_graph,
                                          mock_data_manager_v2):
        """Test query processing with reasoning enabled."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {
                "messages": [AIMessage(content=[
                    {"type": "reasoning_content", "reasoning_content": {"text": "I need to analyze this data"}},
                    {"type": "text", "text": "I'll help you analyze your data."}
                ])]
            }}
        ]
        mock_create_bioinformatics_graph.return_value = mock_graph
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace,
            enable_reasoning=True
        )
        
        result = client.query("Analyze my data")
        
        assert result["success"] is True
        assert "[Thinking:" in result["response"]
        assert "I'll help you analyze your data." in result["response"]
    
    def test_query_with_reasoning_disabled(self, temp_workspace, mock_create_bioinformatics_graph,
                                          mock_data_manager_v2):
        """Test query processing with reasoning disabled."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {
                "messages": [AIMessage(content=[
                    {"type": "reasoning_content", "reasoning_content": {"text": "I need to analyze this data"}},
                    {"type": "text", "text": "I'll help you analyze your data."}
                ])]
            }}
        ]
        mock_create_bioinformatics_graph.return_value = mock_graph
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace,
            enable_reasoning=False
        )
        
        result = client.query("Analyze my data")
        
        assert result["success"] is True
        assert "[Thinking:" not in result["response"]
        assert "I'll help you analyze your data." in result["response"]
    
    def test_query_streaming_mode(self, temp_workspace, mock_create_bioinformatics_graph,
                                  mock_data_manager_v2):
        """Test query processing in streaming mode."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        stream_gen = client.query("Analyze my data", stream=True)
        
        assert hasattr(stream_gen, '__iter__')
        events = list(stream_gen)
        
        assert len(events) > 0
        assert any(event.get("type") == "complete" for event in events)
    
    def test_query_error_handling(self, temp_workspace, mock_create_bioinformatics_graph,
                                  mock_data_manager_v2):
        """Test error handling in query processing."""
        mock_graph = Mock()
        mock_graph.stream.side_effect = Exception("Graph execution failed")
        mock_create_bioinformatics_graph.return_value = mock_graph
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        result = client.query("Analyze my data")
        
        assert result["success"] is False
        assert "error" in result
        assert "Graph execution failed" in result["error"]
        assert result["session_id"] == client.session_id
    
    def test_query_with_data_plots_response(self, temp_workspace, mock_create_bioinformatics_graph,
                                           mock_data_manager_v2):
        """Test query response when data manager has data and plots."""
        mock_data_manager_v2.has_data.return_value = True
        mock_data_manager_v2.get_latest_plots.return_value = [
            {"name": "umap", "path": "/tmp/umap.html"}
        ]
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        result = client.query("Show me the data")
        
        assert result["success"] is True
        assert result["has_data"] is True
        assert len(result["plots"]) == 1
        assert result["plots"][0]["name"] == "umap"
    
    @pytest.mark.parametrize("user_input,expected_in_response", [
        ("Load dataset GSE123456", "GSE123456"),
        ("Perform quality control", "quality"),
        ("Create UMAP visualization", "visualization"),
        ("", ""),  # Empty input
        ("A" * 1000, "A" * 50)  # Very long input (should be truncated in logs)
    ])
    def test_query_with_various_inputs(self, temp_workspace, mock_create_bioinformatics_graph,
                                       mock_data_manager_v2, user_input, expected_in_response):
        """Test query processing with various input types."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        result = client.query(user_input)
        
        assert result["success"] is True
        if expected_in_response and user_input:
            # For non-empty inputs, verify some content handling
            assert len(client.messages) == 2


@pytest.mark.unit  
class TestAgentClientFileOperations:
    """Test file operation functionality."""
    
    def test_detect_file_type_bioinformatics_formats(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test file type detection for bioinformatics formats."""
        client = AgentClient(workspace_path=temp_workspace)
        
        test_cases = [
            ("data.h5ad", "single_cell_data", "Single-cell RNA-seq data (H5AD format)"),
            ("data.h5mu", "multimodal_data", "Multi-modal omics data (H5MU format)"),
            ("matrix.mtx", "matrix_data", "Matrix Market sparse matrix"),
            ("data.csv", "delimited_data", "Comma-separated values"),
            ("data.xlsx", "spreadsheet_data", "Excel spreadsheet"),
            ("config.json", "structured_data", "JSON metadata"),
            ("script.py", "python_script", "Python script")
        ]
        
        for filename, expected_type, expected_desc in test_cases:
            file_path = Path(filename)
            result = client.detect_file_type(file_path)
            
            assert result["type"] == expected_type
            assert result["description"] == expected_desc
    
    def test_detect_file_type_compressed_files(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test file type detection for compressed files."""
        client = AgentClient(workspace_path=temp_workspace)
        
        file_path = Path("data.csv.gz")
        result = client.detect_file_type(file_path)
        
        assert result["type"] == "delimited_data"
        assert result["compressed"] is True
        assert "gzip compressed" in result["description"]
    
    def test_locate_file_absolute_path(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test file location with absolute path."""
        client = AgentClient(workspace_path=temp_workspace)
        
        # Create a test file
        test_file = temp_workspace / "test_data.csv"
        test_file.write_text("col1,col2\n1,2\n3,4")
        
        result = client.locate_file(str(test_file))
        
        assert result["found"] is True
        assert result["path"] == test_file
        assert result["size_bytes"] > 0
        assert result["readable"] is True
        assert result["type"] == "delimited_data"
    
    def test_locate_file_relative_path_search(self, temp_workspace, mock_create_bioinformatics_graph,
                                             mock_data_manager_v2):
        """Test file location with relative path searching."""
        mock_data_manager_v2.data_dir = temp_workspace / "data"
        mock_data_manager_v2.workspace_path = temp_workspace
        mock_data_manager_v2.exports_dir = temp_workspace / "exports"
        mock_data_manager_v2.cache_dir = temp_workspace / "cache"
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Create test file in data directory
        data_dir = temp_workspace / "data"
        data_dir.mkdir(exist_ok=True)
        test_file = data_dir / "test_data.h5ad"
        test_file.write_text("mock h5ad content")
        
        result = client.locate_file("test_data.h5ad")
        
        assert result["found"] is True
        assert result["path"].name == "test_data.h5ad"
        assert "searched_paths" in result
    
    def test_locate_file_not_found(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test file location when file doesn't exist."""
        client = AgentClient(workspace_path=temp_workspace)
        
        result = client.locate_file("nonexistent_file.csv")
        
        assert result["found"] is False
        assert "error" in result
        assert "searched_paths" in result
    
    def test_load_data_file_h5ad_format(self, temp_workspace, mock_create_bioinformatics_graph,
                                        mock_data_manager_v2, synthetic_test_data):
        """Test loading H5AD format data file."""
        # Setup mock data manager
        test_adata = synthetic_test_data["single_cell"]
        mock_data_manager_v2.load_modality.return_value = test_adata
        mock_data_manager_v2.list_modalities.return_value = []
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Create mock file
        test_file = temp_workspace / "test_data.h5ad"
        test_file.write_text("mock h5ad content")
        
        result = client.load_data_file("test_data.h5ad")
        
        assert result["success"] is True
        assert result["modality_name"] == "test_data"
        assert result["data_shape"] == (test_adata.n_obs, test_adata.n_vars)
        mock_data_manager_v2.load_modality.assert_called_once()
    
    def test_load_data_file_csv_format(self, temp_workspace, mock_create_bioinformatics_graph,
                                       mock_data_manager_v2, synthetic_test_data):
        """Test loading CSV format data file."""
        # Setup mock data manager
        test_adata = synthetic_test_data["bulk_rnaseq"]
        mock_data_manager_v2.load_modality.return_value = test_adata
        mock_data_manager_v2.list_modalities.return_value = []
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Create test CSV file
        test_file = temp_workspace / "test_data.csv"
        test_df = pd.DataFrame({
            "gene1": [1, 2, 3],
            "gene2": [4, 5, 6],
            "gene3": [7, 8, 9]
        })
        test_df.to_csv(test_file, index=False)
        
        with patch('pandas.read_csv', return_value=test_df):
            result = client.load_data_file("test_data.csv")
        
        assert result["success"] is True
        assert result["modality_name"] == "test_data"
        mock_data_manager_v2.load_modality.assert_called_once()
    
    def test_load_data_file_name_collision(self, temp_workspace, mock_create_bioinformatics_graph,
                                          mock_data_manager_v2):
        """Test loading file when modality name already exists."""
        mock_data_manager_v2.list_modalities.side_effect = lambda: ["test_data", "test_data_1"]
        mock_data_manager_v2.load_modality.return_value = Mock(n_obs=100, n_vars=200)
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Create test file
        test_file = temp_workspace / "test_data.h5ad"
        test_file.write_text("mock content")
        
        result = client.load_data_file("test_data.h5ad")
        
        assert result["success"] is True
        assert result["modality_name"] == "test_data_2"  # Should increment to avoid collision
    
    def test_load_data_file_unsupported_format(self, temp_workspace, mock_create_bioinformatics_graph,
                                               mock_data_manager_v2):
        """Test loading unsupported file format."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Create unsupported file type
        test_file = temp_workspace / "test_data.pdf"
        test_file.write_text("PDF content")
        
        result = client.load_data_file("test_data.pdf")
        
        assert result["success"] is False
        assert "not a supported data format" in result["error"]
    
    def test_list_workspace_files(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test listing workspace files."""
        client = AgentClient(workspace_path=temp_workspace)
        
        # Create test files
        (temp_workspace / "file1.csv").write_text("data1")
        (temp_workspace / "file2.h5ad").write_text("data2")
        subdir = temp_workspace / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("data3")
        
        files = client.list_workspace_files()
        
        assert len(files) >= 2  # At least the files we created
        file_names = [f["name"] for f in files]
        assert "file1.csv" in file_names
        assert "file2.h5ad" in file_names
    
    def test_read_file_absolute_path(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test reading file with absolute path."""
        client = AgentClient(workspace_path=temp_workspace)
        
        test_file = temp_workspace / "test_file.txt"
        test_content = "This is test content"
        test_file.write_text(test_content)
        
        content = client.read_file(str(test_file))
        
        assert content == test_content
    
    def test_read_file_relative_path(self, temp_workspace, mock_create_bioinformatics_graph,
                                    mock_data_manager_v2):
        """Test reading file with relative path search."""
        mock_data_manager_v2.data_dir = temp_workspace / "data"
        mock_data_manager_v2.workspace_path = temp_workspace
        mock_data_manager_v2.exports_dir = temp_workspace / "exports"
        mock_data_manager_v2.cache_dir = temp_workspace / "cache"
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Create file in data directory
        data_dir = temp_workspace / "data"
        data_dir.mkdir(exist_ok=True)
        test_file = data_dir / "test_file.txt"
        test_content = "Test content in data dir"
        test_file.write_text(test_content)
        
        content = client.read_file("test_file.txt")
        
        assert content == test_content
    
    def test_read_file_not_found(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test reading non-existent file."""
        client = AgentClient(workspace_path=temp_workspace)
        
        content = client.read_file("nonexistent.txt")
        
        assert "File not found" in content
    
    def test_write_file(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test writing file to workspace."""
        client = AgentClient(workspace_path=temp_workspace)
        
        filename = "output.txt"
        content = "Test output content"
        
        success = client.write_file(filename, content)
        
        assert success is True
        assert (temp_workspace / filename).exists()
        assert (temp_workspace / filename).read_text() == content


@pytest.mark.unit
class TestAgentClientStateManagement:
    """Test state management functionality."""
    
    def test_get_conversation_history(self, temp_workspace, mock_create_bioinformatics_graph,
                                     mock_data_manager_v2):
        """Test getting conversation history."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Add some messages
        client.messages.extend([
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?")
        ])
        
        history = client.get_conversation_history()
        
        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there!"
        assert history[2]["role"] == "user"
        assert history[2]["content"] == "How are you?"
    
    def test_get_status_without_data(self, temp_workspace, mock_create_bioinformatics_graph,
                                    mock_data_manager_v2):
        """Test getting status when no data is loaded."""
        mock_data_manager_v2.has_data.return_value = False
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        status = client.get_status()
        
        assert status["session_id"] == client.session_id
        assert status["message_count"] == 0
        assert status["has_data"] is False
        assert status["data_summary"] is None
        assert status["workspace"] == str(temp_workspace)
        assert status["reasoning_enabled"] is True
    
    def test_get_status_with_data(self, temp_workspace, mock_create_bioinformatics_graph,
                                 mock_data_manager_v2):
        """Test getting status when data is loaded."""
        mock_data_manager_v2.has_data.return_value = True
        mock_data_manager_v2.get_data_summary.return_value = {
            "modalities": 2,
            "total_cells": 5000
        }
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Add some messages
        client.messages.extend([
            HumanMessage(content="Test"),
            AIMessage(content="Response")
        ])
        
        status = client.get_status()
        
        assert status["message_count"] == 2
        assert status["has_data"] is True
        assert status["data_summary"]["modalities"] == 2
    
    def test_reset_conversation(self, temp_workspace, mock_create_bioinformatics_graph,
                               mock_data_manager_v2):
        """Test resetting conversation state."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Add some messages
        client.messages.extend([
            HumanMessage(content="Test"),
            AIMessage(content="Response")
        ])
        
        original_metadata_keys = set(client.metadata.keys())
        
        client.reset()
        
        assert len(client.messages) == 0
        assert "reset_at" in client.metadata
        assert set(client.metadata.keys()) > original_metadata_keys
    
    def test_export_session_with_data(self, temp_workspace, mock_create_bioinformatics_graph,
                                     mock_data_manager_v2):
        """Test exporting session when data manager has data."""
        mock_data_manager_v2.has_data.return_value = True
        mock_data_manager_v2.create_data_package.return_value = str(temp_workspace / "package.zip")
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        export_path = client.export_session()
        
        assert export_path == Path(temp_workspace / "package.zip")
        mock_data_manager_v2.create_data_package.assert_called_once()
    
    def test_export_session_without_data(self, temp_workspace, mock_create_bioinformatics_graph,
                                        mock_data_manager_v2):
        """Test exporting session when no data is loaded."""
        mock_data_manager_v2.has_data.return_value = False
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Add some conversation history
        client.messages.append(HumanMessage(content="Test message"))
        
        export_path = client.export_session()
        
        assert export_path.exists()
        assert export_path.suffix == ".json"
        
        # Verify export content
        with open(export_path) as f:
            export_data = json.load(f)
        
        assert export_data["session_id"] == client.session_id
        assert len(export_data["conversation"]) == 1
        assert "exported_at" in export_data


# ===============================================================================
# BaseClient Interface Compliance Tests
# ===============================================================================

@pytest.mark.unit
class TestBaseClientCompliance:
    """Test that AgentClient properly implements BaseClient interface."""
    
    def test_implements_base_client(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test that AgentClient is instance of BaseClient."""
        client = AgentClient(workspace_path=temp_workspace)
        
        assert isinstance(client, BaseClient)
    
    def test_all_abstract_methods_implemented(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test that all abstract methods from BaseClient are implemented."""
        client = AgentClient(workspace_path=temp_workspace)
        
        # Test that all abstract methods exist and are callable
        abstract_methods = [
            "query", "get_status", "list_workspace_files", "read_file",
            "write_file", "get_conversation_history", "reset", "export_session"
        ]
        
        for method_name in abstract_methods:
            assert hasattr(client, method_name)
            assert callable(getattr(client, method_name))
    
    def test_optional_methods_implemented(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test that optional methods are properly handled."""
        client = AgentClient(workspace_path=temp_workspace)
        
        # Test get_usage (should return error for local client)
        usage_result = client.get_usage()
        assert usage_result["success"] is False
        assert "not available" in usage_result["error"]
        
        # Test list_models (should return error for local client)
        models_result = client.list_models()
        assert models_result["success"] is False
        assert "not available" in models_result["error"]


# ===============================================================================
# APIAgentClient Tests
# ===============================================================================

@pytest.mark.unit
class TestAPIAgentClientInitialization:
    """Test APIAgentClient initialization and configuration."""
    
    def test_basic_initialization(self, temp_workspace, mock_session_manager):
        """Test basic APIAgentClient initialization."""
        session_id = uuid4()
        
        with patch('lobster.core.api_client.AgentClient') as mock_agent_client:
            with patch('lobster.core.api_client.APICallbackManager') as mock_callback_manager:
                with patch('lobster.core.api_client.setup_websocket_logging'):
                    client = APIAgentClient(
                        session_id=session_id,
                        session_manager=mock_session_manager,
                        workspace_path=temp_workspace
                    )
        
        assert client.session_id == session_id
        assert client.session_manager == mock_session_manager
        assert client.workspace_path == temp_workspace
    
    def test_initialization_with_data_manager(self, temp_workspace, mock_session_manager,
                                             mock_data_manager_v2):
        """Test initialization with custom data manager."""
        session_id = uuid4()
        
        with patch('lobster.core.api_client.AgentClient') as mock_agent_client:
            with patch('lobster.core.api_client.APICallbackManager'):
                with patch('lobster.core.api_client.setup_websocket_logging'):
                    client = APIAgentClient(
                        session_id=session_id,
                        session_manager=mock_session_manager,
                        data_manager=mock_data_manager_v2,
                        workspace_path=temp_workspace
                    )
        
        assert client.data_manager == mock_data_manager_v2


@pytest.mark.unit 
@pytest.mark.asyncio
class TestAPIAgentClientAsyncOperations:
    """Test APIAgentClient async operations."""
    
    async def test_query_async_success(self, temp_workspace, mock_session_manager):
        """Test successful async query processing."""
        session_id = uuid4()
        
        with patch('lobster.core.api_client.AgentClient') as mock_agent_client:
            mock_client_instance = Mock()
            mock_client_instance.query.return_value = {
                "success": True,
                "response": "Analysis complete",
                "has_data": False,
                "plots": []
            }
            mock_agent_client.return_value = mock_client_instance
            
            with patch('lobster.core.api_client.APICallbackManager') as mock_callback_manager:
                mock_callback_instance = Mock()
                mock_callback_instance.send_progress_update = AsyncMock()
                mock_callback_manager.return_value = mock_callback_instance
                
                with patch('lobster.core.api_client.setup_websocket_logging'):
                    client = APIAgentClient(
                        session_id=session_id,
                        session_manager=mock_session_manager,
                        workspace_path=temp_workspace
                    )
                    
                    client.callback_manager = mock_callback_instance
                    client.agent_client = mock_client_instance
        
        result = await client.query("Test query")
        
        assert result["success"] is True
        assert result["response"] == "Analysis complete"
        mock_callback_instance.send_progress_update.assert_called_once()
    
    async def test_query_async_error_handling(self, temp_workspace, mock_session_manager):
        """Test async query error handling."""
        session_id = uuid4()
        
        with patch('lobster.core.api_client.AgentClient') as mock_agent_client:
            mock_client_instance = Mock()
            mock_client_instance.query.side_effect = Exception("Query failed")
            mock_agent_client.return_value = mock_client_instance
            
            with patch('lobster.core.api_client.APICallbackManager') as mock_callback_manager:
                mock_callback_instance = Mock()
                mock_callback_instance.send_progress_update = AsyncMock()
                mock_callback_manager.return_value = mock_callback_instance
                
                with patch('lobster.core.api_client.setup_websocket_logging'):
                    with patch('lobster.core.api_client.send_error', new_callable=AsyncMock) as mock_send_error:
                        client = APIAgentClient(
                            session_id=session_id,
                            session_manager=mock_session_manager,
                            workspace_path=temp_workspace
                        )
                        
                        client.callback_manager = mock_callback_instance
                        client.agent_client = mock_client_instance
        
        result = await client.query("Test query")
        
        assert result["success"] is False
        assert "Query failed" in result["error"]
        mock_send_error.assert_called_once()
    
    async def test_upload_file_success(self, temp_workspace, mock_session_manager):
        """Test successful file upload."""
        session_id = uuid4()
        
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
                        workspace_path=temp_workspace
                    )
                    
                    client.callback_manager = mock_callback_instance
                    client.data_manager = mock_data_manager
        
        file_content = b"test,data\n1,2\n3,4"
        result = await client.upload_file("test_data.csv", file_content)
        
        assert result["success"] is True
        assert "uploaded and loaded successfully" in result["message"]
        assert result["data_loaded"] is True
    
    async def test_upload_file_load_error(self, temp_workspace, mock_session_manager):
        """Test file upload when data loading fails."""
        session_id = uuid4()
        
        with patch('lobster.core.api_client.AgentClient'):
            with patch('lobster.core.api_client.APICallbackManager') as mock_callback_manager:
                mock_callback_instance = Mock()
                mock_callback_manager.return_value = mock_callback_instance
                
                with patch('lobster.core.api_client.setup_websocket_logging'):
                    mock_data_manager = Mock()
                    mock_data_manager.set_data.side_effect = Exception("Failed to load data")
                    
                    client = APIAgentClient(
                        session_id=session_id,
                        session_manager=mock_session_manager,
                        workspace_path=temp_workspace
                    )
                    
                    client.data_manager = mock_data_manager
        
        file_content = b"invalid,data"
        result = await client.upload_file("test_data.csv", file_content)
        
        assert result["success"] is True  # File saved successfully
        assert result["data_loaded"] is False  # But data loading failed
        assert "Failed to load data" in result["load_warning"]
    
    async def test_download_geo_dataset(self, temp_workspace, mock_session_manager):
        """Test GEO dataset download."""
        session_id = uuid4()
        
        with patch('lobster.core.api_client.AgentClient'):
            with patch('lobster.core.api_client.APICallbackManager') as mock_callback_manager:
                mock_callback_instance = Mock()
                mock_callback_instance.send_progress_update = AsyncMock()
                mock_callback_manager.return_value = mock_callback_instance
                
                with patch('lobster.core.api_client.setup_websocket_logging'):
                    client = AIAgentClient(
                        session_id=session_id,
                        session_manager=mock_session_manager,
                        workspace_path=temp_workspace
                    )
                    
                    client.callback_manager = mock_callback_instance
                    # Mock the query method to return success
                    client.query = AsyncMock(return_value={"success": True})
        
        result = await client.download_geo_dataset("GSE123456")
        
        assert result["success"] is True
        assert result["geo_id"] == "GSE123456"
        mock_callback_instance.send_progress_update.assert_called_once()
    
    async def test_cleanup(self, temp_workspace, mock_session_manager):
        """Test cleanup of API client resources."""
        session_id = uuid4()
        
        with patch('lobster.core.api_client.AgentClient'):
            with patch('lobster.core.api_client.APICallbackManager'):
                with patch('lobster.core.api_client.setup_websocket_logging'):
                    with patch('lobster.core.api_client.remove_websocket_logging') as mock_remove_logging:
                        client = APIAgentClient(
                            session_id=session_id,
                            session_manager=mock_session_manager,
                            workspace_path=temp_workspace
                        )
                        
                        client.websocket_logging_handler = Mock()
        
        await client.cleanup()
        
        mock_remove_logging.assert_called_once()


# ===============================================================================
# Error Handling and Edge Cases Tests
# ===============================================================================

@pytest.mark.unit
class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge case scenarios."""
    
    def test_query_with_empty_events(self, temp_workspace, mock_create_bioinformatics_graph,
                                    mock_data_manager_v2):
        """Test query processing when graph returns empty events."""
        mock_graph = Mock()
        mock_graph.stream.return_value = []
        mock_create_bioinformatics_graph.return_value = mock_graph
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        result = client.query("Test query")
        
        assert result["success"] is True
        assert result["response"] == "No response generated."
    
    def test_query_with_malformed_events(self, temp_workspace, mock_create_bioinformatics_graph,
                                        mock_data_manager_v2):
        """Test query processing with malformed events."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"unexpected_key": "value"},
            {"supervisor": "not_a_dict"},
            {"supervisor": {"messages": "not_a_list"}},
            {"supervisor": {"messages": []}},
        ]
        mock_create_bioinformatics_graph.return_value = mock_graph
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        result = client.query("Test query")
        
        assert result["success"] is True
        assert result["response"] == "No response generated."
    
    def test_extract_content_from_various_message_formats(self, temp_workspace, 
                                                         mock_create_bioinformatics_graph,
                                                         mock_data_manager_v2):
        """Test content extraction from various message formats."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Test string content
        result = client._extract_content_from_message("Simple string content")
        assert result == "Simple string content"
        
        # Test list content with text blocks
        list_content = [
            {"type": "text", "text": "Main response"},
            {"type": "reasoning_content", "reasoning_content": {"text": "Thinking process"}}
        ]
        client.enable_reasoning = True
        result = client._extract_content_from_message(list_content)
        assert "[Thinking: Thinking process]" in result
        assert "Main response" in result
        
        # Test list content with reasoning disabled
        client.enable_reasoning = False
        result = client._extract_content_from_message(list_content)
        assert "[Thinking:" not in result
        assert "Main response" in result
        
        # Test empty content
        result = client._extract_content_from_message("")
        assert result == ""
        
        # Test None content
        result = client._extract_content_from_message(None)
        assert result == ""
    
    def test_workspace_permissions_error(self, mock_create_bioinformatics_graph):
        """Test handling of workspace permission errors."""
        # Try to create workspace in non-writable location
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError):
                AgentClient(workspace_path=Path("/root/restricted"))
    
    def test_file_operations_with_permission_errors(self, temp_workspace, mock_create_bioinformatics_graph):
        """Test file operations with permission errors."""
        client = AgentClient(workspace_path=temp_workspace)
        
        # Test write file with permission error
        with patch('pathlib.Path.write_text') as mock_write:
            mock_write.side_effect = PermissionError("Permission denied")
            
            success = client.write_file("restricted.txt", "content")
            assert success is False
    
    def test_data_manager_initialization_error(self, temp_workspace):
        """Test error handling when DataManagerV2 initialization fails."""
        with patch('lobster.core.client.DataManagerV2') as mock_dm:
            mock_dm.side_effect = Exception("DataManager initialization failed")
            
            with patch('lobster.core.client.create_bioinformatics_graph'):
                with pytest.raises(Exception) as exc_info:
                    AgentClient(workspace_path=temp_workspace)
                
                assert "DataManager initialization failed" in str(exc_info.value)


# ===============================================================================
# Performance and Concurrency Tests
# ===============================================================================

@pytest.mark.unit
@pytest.mark.performance
class TestPerformanceAndConcurrency:
    """Test performance characteristics and concurrency handling."""
    
    def test_query_performance_baseline(self, temp_workspace, mock_create_bioinformatics_graph,
                                       mock_data_manager_v2, benchmark):
        """Benchmark basic query performance."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        def run_query():
            return client.query("Test query for performance")
        
        result = benchmark(run_query)
        
        assert result["success"] is True
    
    def test_concurrent_queries(self, temp_workspace, mock_create_bioinformatics_graph,
                               mock_data_manager_v2):
        """Test handling of concurrent queries."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        def run_query(query_id):
            return client.query(f"Concurrent query {query_id}")
        
        # Run multiple queries in threads
        results = []
        threads = []
        
        for i in range(5):
            thread = threading.Thread(target=lambda i=i: results.append(run_query(i)))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        assert all(result["success"] for result in results)
    
    def test_memory_usage_with_large_conversation(self, temp_workspace, mock_create_bioinformatics_graph,
                                                 mock_data_manager_v2):
        """Test memory usage with large conversation history."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Add many messages to conversation
        for i in range(1000):
            client.messages.extend([
                HumanMessage(content=f"User message {i}"),
                AIMessage(content=f"AI response {i}")
            ])
        
        # Test operations still work with large history
        history = client.get_conversation_history()
        assert len(history) == 2000
        
        status = client.get_status()
        assert status["message_count"] == 2000
    
    @pytest.mark.parametrize("n_files", [10, 50, 100])
    def test_file_listing_performance(self, temp_workspace, mock_create_bioinformatics_graph,
                                     n_files):
        """Test file listing performance with varying numbers of files."""
        client = AgentClient(workspace_path=temp_workspace)
        
        # Create many test files
        for i in range(n_files):
            (temp_workspace / f"file_{i:03d}.txt").write_text(f"Content {i}")
        
        start_time = time.time()
        files = client.list_workspace_files()
        end_time = time.time()
        
        assert len(files) >= n_files
        assert (end_time - start_time) < 5.0  # Should complete within 5 seconds
    
    def test_streaming_query_performance(self, temp_workspace, mock_create_bioinformatics_graph,
                                        mock_data_manager_v2):
        """Test streaming query performance and behavior."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        start_time = time.time()
        stream_events = list(client.query("Stream test query", stream=True))
        end_time = time.time()
        
        assert len(stream_events) > 0
        assert any(event.get("type") == "complete" for event in stream_events)
        assert (end_time - start_time) < 10.0  # Should complete within 10 seconds


# ===============================================================================
# Integration-style Tests for Client Interactions
# ===============================================================================

@pytest.mark.unit
class TestClientIntegrationScenarios:
    """Test realistic client usage scenarios."""
    
    def test_full_analysis_workflow(self, temp_workspace, mock_create_bioinformatics_graph,
                                   mock_data_manager_v2, synthetic_test_data):
        """Test a complete analysis workflow through the client."""
        # Setup mock responses
        mock_data_manager_v2.has_data.return_value = True
        mock_data_manager_v2.get_latest_plots.return_value = [
            {"name": "qc_plot", "path": "/tmp/qc.html"},
            {"name": "umap_plot", "path": "/tmp/umap.html"}
        ]
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Step 1: Load data
        result1 = client.query("Load the single-cell dataset")
        assert result1["success"] is True
        
        # Step 2: Perform QC
        result2 = client.query("Perform quality control analysis")
        assert result2["success"] is True
        
        # Step 3: Create visualization
        result3 = client.query("Create UMAP visualization")
        assert result3["success"] is True
        assert len(result3["plots"]) == 2
        
        # Verify conversation history
        history = client.get_conversation_history()
        assert len(history) == 6  # 3 user messages + 3 AI responses
        
        # Verify status
        status = client.get_status()
        assert status["has_data"] is True
        assert status["message_count"] == 6
    
    def test_error_recovery_workflow(self, temp_workspace, mock_create_bioinformatics_graph,
                                    mock_data_manager_v2):
        """Test error recovery in analysis workflow."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # First query succeeds
        result1 = client.query("Load dataset")
        assert result1["success"] is True
        
        # Second query fails
        mock_graph = Mock()
        mock_graph.stream.side_effect = Exception("Analysis failed")
        client.graph = mock_graph
        
        result2 = client.query("Perform analysis")
        assert result2["success"] is False
        assert "Analysis failed" in result2["error"]
        
        # Third query succeeds (recovery)
        mock_graph.stream.side_effect = None
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [AIMessage(content="Recovery successful")]}}
        ]
        
        result3 = client.query("Try analysis again")
        assert result3["success"] is True
        assert "Recovery successful" in result3["response"]
    
    def test_session_export_and_metadata(self, temp_workspace, mock_create_bioinformatics_graph,
                                        mock_data_manager_v2):
        """Test session export with comprehensive metadata."""
        mock_data_manager_v2.has_data.return_value = False
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace,
            session_id="test_export_session"
        )
        
        # Perform some interactions
        client.query("Hello")
        client.query("Load some data")
        client.query("Create a plot")
        
        # Export session
        export_path = client.export_session()
        
        assert export_path.exists()
        
        # Verify export content
        with open(export_path) as f:
            export_data = json.load(f)
        
        assert export_data["session_id"] == "test_export_session"
        assert len(export_data["conversation"]) == 6  # 3 user + 3 AI messages
        assert "metadata" in export_data
        assert "status" in export_data
        assert "workspace_status" in export_data
        assert "exported_at" in export_data
    
    def test_client_resource_cleanup(self, temp_workspace, mock_create_bioinformatics_graph,
                                    mock_data_manager_v2):
        """Test proper cleanup of client resources."""
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace
        )
        
        # Perform some operations
        client.query("Test query")
        client.write_file("test.txt", "content")
        
        # Reset should clear conversation but not affect files
        client.reset()
        
        assert len(client.messages) == 0
        assert (temp_workspace / "test.txt").exists()
        
        # Status should reflect reset
        status = client.get_status()
        assert status["message_count"] == 0
        assert "reset_at" in client.metadata


# ===============================================================================
# Parametrized Tests for Various Scenarios
# ===============================================================================

@pytest.mark.unit
class TestParametrizedScenarios:
    """Test various scenarios using parametrized tests."""
    
    @pytest.mark.parametrize("reasoning_enabled", [True, False])
    def test_reasoning_toggle_behavior(self, temp_workspace, mock_create_bioinformatics_graph,
                                      mock_data_manager_v2, reasoning_enabled):
        """Test behavior with reasoning enabled/disabled."""
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {
                "messages": [AIMessage(content=[
                    {"type": "reasoning_content", "reasoning_content": {"text": "Thinking..."}},
                    {"type": "text", "text": "Response text"}
                ])]
            }}
        ]
        mock_create_bioinformatics_graph.return_value = mock_graph
        
        client = AgentClient(
            data_manager=mock_data_manager_v2,
            workspace_path=temp_workspace,
            enable_reasoning=reasoning_enabled
        )
        
        result = client.query("Test query")
        
        assert result["success"] is True
        if reasoning_enabled:
            assert "[Thinking:" in result["response"]
        else:
            assert "[Thinking:" not in result["response"]
        assert "Response text" in result["response"]
    
    @pytest.mark.parametrize("file_extension,expected_type", [
        (".h5ad", "single_cell_data"),
        (".h5mu", "multimodal_data"),
        (".csv", "delimited_data"),
        (".tsv", "delimited_data"),
        (".xlsx", "spreadsheet_data"),
        (".json", "structured_data"),
        (".py", "python_script"),
        (".unknown", "unknown")
    ])
    def test_file_type_detection_parametrized(self, temp_workspace, mock_create_bioinformatics_graph,
                                             file_extension, expected_type):
        """Test file type detection for various extensions."""
        client = AgentClient(workspace_path=temp_workspace)
        
        file_path = Path(f"test_file{file_extension}")
        result = client.detect_file_type(file_path)
        
        assert result["type"] == expected_type
    
    @pytest.mark.parametrize("session_id,enable_reasoning,enable_langfuse", [
        ("custom_session_1", True, False),
        ("custom_session_2", False, False),
        ("custom_session_3", True, True),
        (None, True, False),  # Auto-generated session ID
    ])
    def test_initialization_parameter_combinations(self, temp_workspace, mock_create_bioinformatics_graph,
                                                  session_id, enable_reasoning, enable_langfuse):
        """Test various initialization parameter combinations."""
        with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "test_key"} if enable_langfuse else {}):
            with patch('lobster.core.client.LangfuseCallback', Mock()):
                client = AgentClient(
                    workspace_path=temp_workspace,
                    session_id=session_id,
                    enable_reasoning=enable_reasoning,
                    enable_langfuse=enable_langfuse
                )
        
        if session_id:
            assert client.session_id == session_id
        else:
            assert client.session_id.startswith("session_")
        
        assert client.enable_reasoning == enable_reasoning
        
        if enable_langfuse:
            assert len(client.callbacks) > 0
        else:
            assert len(client.callbacks) == 0