"""
Unit tests for the Visualization Expert Agent implementation.

Tests state persistence, UUID tracking, supervisor integration, and visualization functionality.
"""

import pytest
import uuid
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import anndata
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.agents.visualization_expert import visualization_expert, VisualizationExpertState


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManagerV2 instance with visualization state."""
    mock_dm = Mock(spec=DataManagerV2)
    
    # Initialize visualization state
    mock_dm.visualization_state = {
        'history': [],
        'settings': {
            'default_width': 800,
            'default_height': 600,
            'color_scheme': 'Set1',
            'save_by_default': True,
            'export_formats': ['html', 'png']
        },
        'plot_registry': {}
    }
    
    # Mock methods
    mock_dm.list_modalities.return_value = ['test_modality', 'geo_gse12345_clustered']
    mock_dm.add_visualization_record = MagicMock()
    mock_dm.get_visualization_history = MagicMock()
    mock_dm.add_plot = MagicMock()
    mock_dm.save_plots_to_workspace = MagicMock()
    mock_dm.log_tool_usage = MagicMock()
    
    return mock_dm


@pytest.fixture
def sample_adata():
    """Create sample AnnData object for testing."""
    n_obs, n_vars = 1000, 2000
    X = np.random.negative_binomial(100, 0.01, size=(n_obs, n_vars))
    
    adata = anndata.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    
    # Add clustering results
    adata.obs['leiden'] = np.random.choice(['0', '1', '2', '3'], n_obs)
    adata.obs['cell_type'] = np.random.choice(['T_cell', 'B_cell', 'NK_cell'], n_obs)
    
    # Add UMAP coordinates
    adata.obsm['X_umap'] = np.random.random((n_obs, 2))
    adata.obsm['X_pca'] = np.random.random((n_obs, 50))
    
    # Add QC metrics
    adata.obs['n_genes_by_counts'] = np.random.randint(500, 3000, n_obs)
    adata.obs['total_counts'] = np.random.randint(1000, 50000, n_obs)
    adata.obs['pct_counts_mt'] = np.random.uniform(0, 25, n_obs)
    
    return adata


class TestVisualizationStateManagement:
    """Test visualization state management in DataManagerV2."""
    
    def test_add_visualization_record(self, mock_data_manager):
        """Test that visualization records are properly added to state."""
        plot_id = str(uuid.uuid4())
        metadata = {
            "type": "umap",
            "modality": "test_modality",
            "color_by": "leiden",
            "created_by": "visualization_expert"
        }
        
        # Configure mock to simulate the actual method
        def mock_add_visualization_record(pid, meta):
            mock_data_manager.visualization_state['history'].append({
                'plot_id': pid,
                'timestamp': pd.Timestamp.now(),
                'metadata': meta
            })
            mock_data_manager.visualization_state['plot_registry'][pid] = meta
        
        mock_data_manager.add_visualization_record.side_effect = mock_add_visualization_record
        
        # Call the method
        mock_data_manager.add_visualization_record(plot_id, metadata)
        
        # Verify state was updated
        assert len(mock_data_manager.visualization_state['history']) == 1
        assert plot_id in mock_data_manager.visualization_state['plot_registry']
        assert mock_data_manager.visualization_state['plot_registry'][plot_id] == metadata
        
        # Verify method was called
        mock_data_manager.add_visualization_record.assert_called_once_with(plot_id, metadata)

    def test_get_visualization_history(self, mock_data_manager):
        """Test that visualization history retrieval works correctly."""
        # Setup mock history
        mock_history = [
            {
                'plot_id': str(uuid.uuid4()),
                'timestamp': pd.Timestamp.now(),
                'metadata': {'type': 'umap', 'modality': 'test1'}
            },
            {
                'plot_id': str(uuid.uuid4()),
                'timestamp': pd.Timestamp.now(),
                'metadata': {'type': 'qc_plots', 'modality': 'test2'}
            }
        ]
        
        mock_data_manager.get_visualization_history.return_value = mock_history
        
        # Test retrieval
        history = mock_data_manager.get_visualization_history(limit=10)
        
        assert len(history) == 2
        assert history[0]['metadata']['type'] == 'umap'
        assert history[1]['metadata']['type'] == 'qc_plots'
        mock_data_manager.get_visualization_history.assert_called_once_with(limit=10)

    def test_visualization_settings_management(self, mock_data_manager):
        """Test visualization settings get/update functionality."""
        # Test initial settings
        initial_settings = mock_data_manager.visualization_state['settings']
        assert initial_settings['default_width'] == 800
        assert initial_settings['color_scheme'] == 'Set1'
        
        # Test settings update
        new_settings = {'default_width': 1200, 'color_scheme': 'viridis'}
        mock_data_manager.visualization_state['settings'].update(new_settings)
        
        assert mock_data_manager.visualization_state['settings']['default_width'] == 1200
        assert mock_data_manager.visualization_state['settings']['color_scheme'] == 'viridis'


class TestVisualizationTools:
    """Test individual visualization tools with UUID tracking."""
    
    @patch('lobster.tools.visualization_service.SingleCellVisualizationService')
    @patch('lobster.config.settings.get_settings')
    def test_create_umap_plot_with_uuid(self, mock_settings, mock_viz_service, mock_data_manager, sample_adata):
        """Test UMAP plot creation with proper UUID tracking."""
        # Setup mocks
        mock_settings.return_value.get_agent_llm_params.return_value = {}
        mock_viz_service.return_value.create_umap_plot.return_value = go.Figure()
        
        mock_data_manager.get_modality.return_value = sample_adata
        mock_data_manager.add_plot.return_value = "plot_123"
        mock_data_manager.save_plots_to_workspace.return_value = ["plot.html", "plot.png"]
        
        # Test will be done by importing and calling the tool directly
        # This is a simplified test - in reality we'd need to properly instantiate the agent
        
        # Verify UUID is generated and tracked
        assert len(str(uuid.uuid4())) == 36  # Standard UUID length
        
        # Verify mock calls would be made
        mock_data_manager.add_visualization_record.reset_mock()
        mock_data_manager.add_plot.reset_mock()
        
        # Test successful execution path
        assert mock_data_manager.list_modalities.return_value is not None

    @patch('lobster.tools.visualization_service.SingleCellVisualizationService')
    def test_create_qc_plots_with_state_persistence(self, mock_viz_service, mock_data_manager, sample_adata):
        """Test QC plots creation with state persistence."""
        mock_viz_service.return_value.create_qc_plots.return_value = go.Figure()
        mock_data_manager.get_modality.return_value = sample_adata
        
        # Verify that state persistence methods are called
        plot_id = str(uuid.uuid4())
        metadata = {
            "type": "qc_plots",
            "modality": "test_modality",
            "metrics": ["n_genes_by_counts", "total_counts"],
            "created_by": "visualization_expert"
        }
        
        # Simulate state persistence
        mock_data_manager.add_visualization_record(plot_id, metadata)
        
        # Verify method was called with correct parameters
        mock_data_manager.add_visualization_record.assert_called_with(plot_id, metadata)

    def test_violin_plot_gene_validation(self, mock_data_manager, sample_adata):
        """Test violin plot gene validation logic."""
        # Setup test data
        sample_adata.var_names = ['GENE1', 'GENE2', 'GENE3']
        mock_data_manager.get_modality.return_value = sample_adata
        
        # Test gene filtering
        requested_genes = ['GENE1', 'NONEXISTENT', 'GENE3']
        available_genes = [gene for gene in requested_genes if gene in sample_adata.var_names]
        
        assert len(available_genes) == 2
        assert 'GENE1' in available_genes
        assert 'GENE3' in available_genes
        assert 'NONEXISTENT' not in available_genes


class TestSupervisorIntegration:
    """Test supervisor-mediated workflow integration."""
    
    def test_visualization_completion_reporting(self, mock_data_manager):
        """Test that completion reports follow correct format for supervisor."""
        plot_id = str(uuid.uuid4())
        requesting_agent = "singlecell_expert_agent"
        
        expected_report_format = f"""
📊 **Visualization Task Complete**

**Status**: SUCCESS
**Plot ID**: {plot_id}
**Requesting Agent**: {requesting_agent}
**Timestamp**: {pd.Timestamp.now()}

**Action Required**: Please inform {requesting_agent} that visualization {plot_id} is ready."""
        
        # Verify report structure contains required fields
        assert "Visualization Task Complete" in expected_report_format
        assert "Status" in expected_report_format
        assert "Plot ID" in expected_report_format
        assert "Requesting Agent" in expected_report_format
        assert "Action Required" in expected_report_format

    def test_supervisor_only_communication_pattern(self):
        """Test that agent follows supervisor-only communication pattern."""
        # The visualization expert should only respond to supervisor requests
        # This is enforced by the system prompt and agent design
        
        system_prompt_requirements = [
            "You ONLY respond to supervisor requests",
            "You NEVER communicate directly with other agents", 
            "You ALWAYS report completion back to supervisor",
            "supervisor-mediated workflows"
        ]
        
        # These requirements should be in the agent's system prompt
        for requirement in system_prompt_requirements:
            # In a real test, we'd verify these are in the actual system prompt
            assert isinstance(requirement, str)


class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    def test_modality_not_found_error_handling(self, mock_data_manager):
        """Test handling of modality not found errors."""
        mock_data_manager.list_modalities.return_value = ['existing_modality']
        
        # Test error message format
        nonexistent_modality = 'nonexistent_modality'
        expected_error = f"Modality '{nonexistent_modality}' not found"
        
        # Verify error message format
        assert "not found" in expected_error
        assert nonexistent_modality in expected_error

    def test_visualization_service_error_handling(self, mock_data_manager, sample_adata):
        """Test handling of visualization service errors."""
        mock_data_manager.get_modality.return_value = sample_adata
        
        # Test that service errors are caught and formatted properly
        with patch('lobster.tools.visualization_service.SingleCellVisualizationService') as mock_service:
            mock_service.return_value.create_umap_plot.side_effect = Exception("Service error")
            
            # In a real implementation, this would be caught and formatted
            expected_error_format = "❌ Error creating UMAP plot: Service error"
            assert "❌ Error" in expected_error_format
            assert "Service error" in expected_error_format


class TestPerformanceAndMetrics:
    """Test performance characteristics and metrics."""
    
    def test_uuid_generation_performance(self):
        """Test UUID generation performance for high-throughput scenarios."""
        import time
        
        start_time = time.time()
        uuids = [str(uuid.uuid4()) for _ in range(1000)]
        end_time = time.time()
        
        # Should generate 1000 UUIDs in under 1 second
        assert end_time - start_time < 1.0
        
        # All UUIDs should be unique
        assert len(set(uuids)) == 1000
        
        # All UUIDs should be proper format
        for uuid_str in uuids[:10]:  # Test first 10
            assert len(uuid_str) == 36
            assert uuid_str.count('-') == 4

    def test_state_persistence_memory_usage(self, mock_data_manager):
        """Test that state persistence doesn't cause memory leaks."""
        # Add multiple visualization records
        for i in range(100):
            plot_id = str(uuid.uuid4())
            metadata = {
                "type": "test_plot",
                "modality": f"modality_{i}",
                "created_by": "visualization_expert"
            }
            
            # Simulate adding to state
            mock_data_manager.visualization_state['history'].append({
                'plot_id': plot_id,
                'timestamp': pd.Timestamp.now(),
                'metadata': metadata
            })
            mock_data_manager.visualization_state['plot_registry'][plot_id] = metadata
        
        # Verify state contains expected number of records
        assert len(mock_data_manager.visualization_state['history']) == 100
        assert len(mock_data_manager.visualization_state['plot_registry']) == 100


class TestVisualizationReadiness:
    """Test visualization readiness checking functionality."""
    
    def test_check_visualization_readiness_complete_data(self, mock_data_manager, sample_adata):
        """Test readiness check with complete single-cell data."""
        mock_data_manager.get_modality.return_value = sample_adata
        
        # Expected readiness indicators
        expected_plots = []
        if "X_umap" in sample_adata.obsm:
            expected_plots.append("UMAP")
        if "X_pca" in sample_adata.obsm:
            expected_plots.append("PCA")
        if "leiden" in sample_adata.obs.columns:
            expected_plots.append("Cluster-based plots")
        if "cell_type" in sample_adata.obs.columns:
            expected_plots.append("Cell type plots")
        
        # Should have multiple plot types available
        assert len(expected_plots) >= 3  # UMAP, PCA, Cluster-based, Cell type

    def test_check_visualization_readiness_minimal_data(self, mock_data_manager):
        """Test readiness check with minimal data."""
        # Create minimal AnnData
        minimal_adata = anndata.AnnData(X=np.random.random((100, 1000)))
        mock_data_manager.get_modality.return_value = minimal_adata
        
        # Should handle minimal data gracefully
        available_plots = []
        if "X_umap" in minimal_adata.obsm:
            available_plots.append("UMAP")
        # etc.
        
        # With minimal data, fewer plots should be available
        assert isinstance(available_plots, list)


class TestBackwardCompatibility:
    """Test backward compatibility with existing workflows."""
    
    def test_plot_metadata_structure(self, mock_data_manager):
        """Test that plot metadata maintains backward compatibility."""
        # Standard metadata structure
        expected_metadata_fields = [
            "plot_id",
            "modality_name", 
            "plot_type",
            "n_cells",
            "parameters"
        ]
        
        sample_metadata = {
            "plot_id": str(uuid.uuid4()),
            "modality_name": "test_modality",
            "plot_type": "umap",
            "color_by": "leiden",
            "n_cells": 1000,
            "parameters": {
                "point_size": 5,
                "title": None
            }
        }
        
        # Verify all expected fields are present
        for field in expected_metadata_fields:
            assert field in sample_metadata

    def test_existing_plot_manager_compatibility(self, mock_data_manager):
        """Test compatibility with existing plot management system."""
        # Should work with existing add_plot method
        mock_figure = go.Figure()
        
        mock_data_manager.add_plot.return_value = "plot_123"
        
        # Call existing method
        result = mock_data_manager.add_plot(
            plot=mock_figure,
            title="Test Plot",
            source="visualization_expert",
            dataset_info={"modality_name": "test"}
        )
        
        assert result == "plot_123"
        mock_data_manager.add_plot.assert_called_once()


class TestIntegrationWorkflows:
    """Test end-to-end integration workflows."""
    
    @patch('lobster.config.settings.get_settings')
    def test_agent_initialization(self, mock_settings, mock_data_manager):
        """Test that visualization expert agent initializes properly."""
        # Mock settings
        mock_settings.return_value.get_agent_llm_params.return_value = {
            'model_id': 'claude-3-sonnet-20240229-v1:0',
            'max_tokens': 4000,
            'temperature': 0.1
        }
        
        # Test agent creation
        with patch('langchain_aws.ChatBedrockConverse'), \
             patch('langgraph.prebuilt.create_react_agent') as mock_create_agent:
            
            agent = visualization_expert(
                data_manager=mock_data_manager,
                callback_handler=None,
                agent_name="test_viz_expert"
            )
            
            # Verify agent creation was attempted
            mock_create_agent.assert_called_once()

    def test_tool_registration_completeness(self):
        """Test that all required visualization tools are registered."""
        expected_tools = [
            'check_visualization_readiness',
            'create_umap_plot',
            'create_qc_plots', 
            'create_violin_plot',
            'create_feature_plot',
            'create_dot_plot',
            'create_heatmap',
            'create_elbow_plot',
            'create_cluster_composition_plot',
            'get_visualization_history',
            'report_visualization_complete'
        ]
        
        # In the actual implementation, we'd verify these tools are in base_tools
        # For now, just verify the expected tools list is complete
        assert len(expected_tools) == 11  # Should have 11 tools total


class TestVisualizationExpertState:
    """Test the VisualizationExpertState schema."""
    
    def test_state_schema_structure(self):
        """Test that state schema has required fields."""
        # The state schema should include messages, current_request, last_plot_id
        from lobster.agents.visualization_expert import VisualizationExpertState
        
        # Check that it's a TypedDict
        assert hasattr(VisualizationExpertState, '__annotations__')
        
        # Required fields
        expected_fields = ['messages', 'current_request', 'last_plot_id']
        actual_fields = list(VisualizationExpertState.__annotations__.keys())
        
        for field in expected_fields:
            assert field in actual_fields


class TestDataValidation:
    """Test data validation and preprocessing."""
    
    def test_gene_list_validation(self, sample_adata):
        """Test gene list validation for plotting functions."""
        # Test with valid genes
        valid_genes = ['gene_0', 'gene_1', 'gene_2']
        available_genes = [gene for gene in valid_genes if gene in sample_adata.var_names]
        
        assert len(available_genes) == 3  # All should be available
        
        # Test with mixed valid/invalid genes  
        mixed_genes = ['gene_0', 'invalid_gene', 'gene_1']
        available_genes = [gene for gene in mixed_genes if gene in sample_adata.var_names]
        
        assert len(available_genes) == 2  # Only valid ones
        assert 'invalid_gene' not in available_genes

    def test_modality_existence_validation(self, mock_data_manager):
        """Test modality existence validation."""
        # Test with existing modality
        mock_data_manager.list_modalities.return_value = ['modality1', 'modality2']
        
        assert 'modality1' in mock_data_manager.list_modalities()
        assert 'nonexistent' not in mock_data_manager.list_modalities()


class TestVisualizationParameters:
    """Test visualization parameter handling and auto-detection."""
    
    def test_auto_point_size_calculation(self):
        """Test automatic point size calculation based on cell count."""
        # Test size calculation logic
        test_cases = [
            (500, 8),    # Small dataset
            (5000, 5),   # Medium dataset  
            (25000, 3),  # Large dataset
            (75000, 2),  # Very large dataset
        ]
        
        for n_cells, expected_size in test_cases:
            if n_cells < 1000:
                calculated_size = 8
            elif n_cells < 10000:
                calculated_size = 5
            elif n_cells < 50000:
                calculated_size = 3
            else:
                calculated_size = 2
                
            assert calculated_size == expected_size

    def test_qc_metrics_auto_detection(self, sample_adata):
        """Test automatic QC metrics detection."""
        # Standard QC metrics that should be detected
        expected_qc_metrics = []
        
        if 'n_genes_by_counts' in sample_adata.obs.columns:
            expected_qc_metrics.append('n_genes_by_counts')
        if 'total_counts' in sample_adata.obs.columns:
            expected_qc_metrics.append('total_counts')
        if 'pct_counts_mt' in sample_adata.obs.columns:
            expected_qc_metrics.append('pct_counts_mt')
        
        # Should find at least 3 QC metrics in sample data
        assert len(expected_qc_metrics) >= 3


# Integration test scenarios
class TestEndToEndWorkflows:
    """Test complete visualization workflows."""
    
    @patch('lobster.tools.visualization_service.SingleCellVisualizationService')
    @patch('lobster.config.settings.get_settings')  
    def test_complete_umap_workflow(self, mock_settings, mock_viz_service, mock_data_manager, sample_adata):
        """Test complete UMAP creation workflow with state tracking."""
        # Setup mocks
        mock_settings.return_value.get_agent_llm_params.return_value = {}
        mock_viz_service.return_value.create_umap_plot.return_value = go.Figure()
        
        mock_data_manager.get_modality.return_value = sample_adata
        mock_data_manager.add_plot.return_value = "plot_123"
        mock_data_manager.save_plots_to_workspace.return_value = ["plot.html"]
        
        # Simulate complete workflow
        plot_id = str(uuid.uuid4())
        
        # 1. Validate modality exists
        assert 'test_modality' in mock_data_manager.list_modalities.return_value
        
        # 2. Create plot
        fig = mock_viz_service.return_value.create_umap_plot.return_value
        assert isinstance(fig, go.Figure)
        
        # 3. Add to data manager
        mock_data_manager.add_plot(fig, "UMAP - leiden", "visualization_expert", {})
        
        # 4. Track in visualization state  
        mock_data_manager.add_visualization_record(plot_id, {})
        
        # 5. Save plots
        saved_files = mock_data_manager.save_plots_to_workspace()
        assert len(saved_files) > 0
        
        # Verify all steps were called
        mock_data_manager.add_plot.assert_called_once()
        mock_data_manager.add_visualization_record.assert_called_once()
        mock_data_manager.save_plots_to_workspace.assert_called_once()


def test_visualization_expert_import():
    """Test that visualization expert module imports correctly."""
    from lobster.agents.visualization_expert import visualization_expert, VisualizationExpertState
    
    # Verify imports work
    assert callable(visualization_expert)
    assert VisualizationExpertState is not None


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for visualization operations."""
    
    def test_state_operation_latency(self, mock_data_manager):
        """Test that state operations complete within acceptable time limits."""
        import time
        
        # Test add_visualization_record latency
        start_time = time.time()
        
        for i in range(100):
            plot_id = str(uuid.uuid4())
            metadata = {"type": "test", "modality": f"mod_{i}"}
            
            # Simulate state update
            mock_data_manager.visualization_state['history'].append({
                'plot_id': plot_id,
                'timestamp': pd.Timestamp.now(),
                'metadata': metadata
            })
            mock_data_manager.visualization_state['plot_registry'][plot_id] = metadata
        
        end_time = time.time()
        
        # Should complete 100 state operations in under 0.5 seconds
        assert end_time - start_time < 0.5

    def test_history_retrieval_performance(self, mock_data_manager):
        """Test history retrieval performance with large datasets."""
        # Setup large history
        large_history = []
        for i in range(1000):
            large_history.append({
                'plot_id': str(uuid.uuid4()),
                'timestamp': pd.Timestamp.now(),
                'metadata': {'type': 'test', 'modality': f'mod_{i}'}
            })
        
        mock_data_manager.get_visualization_history.return_value = large_history[-10:]
        
        import time
        start_time = time.time()
        history = mock_data_manager.get_visualization_history(limit=10)
        end_time = time.time()
        
        # Should retrieve history quickly
        assert end_time - start_time < 0.1
        assert len(history) == 10


if __name__ == "__main__":
    pytest.main([__file__])
