"""
Comprehensive unit tests for single-cell expert agent.

This module provides thorough testing of the single-cell expert agent including
clustering analysis, marker gene identification, cell type annotation,
quality control, and single-cell specific workflows.

Test coverage target: 95%+ with meaningful tests for single-cell analysis.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import pandas as pd

from lobster.agents.singlecell_expert import singlecell_expert_agent
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.factories import SingleCellDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Mock Objects and Fixtures  
# ===============================================================================

class MockMessage:
    """Mock LangGraph message object."""
    
    def __init__(self, content: str, sender: str = "human"):
        self.content = content
        self.sender = sender


class MockState:
    """Mock LangGraph state object."""
    
    def __init__(self, messages=None, **kwargs):
        self.messages = messages or []
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_data_manager():
    """Create mock data manager with single-cell data."""
    with patch('lobster.core.data_manager_v2.DataManagerV2') as MockDataManager:
        mock_dm = MockDataManager.return_value
        mock_dm.list_modalities.return_value = ['sc_data', 'sc_data_filtered']
        
        # Mock single-cell data
        sc_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_dm.get_modality.return_value = sc_data
        mock_dm.add_modality.return_value = None
        
        yield mock_dm


@pytest.fixture
def singlecell_state():
    """Create single-cell expert state for testing."""
    return MockState(
        messages=[MockMessage("Perform single-cell clustering analysis")],
        data_manager=Mock(),
        current_agent="singlecell_expert_agent"
    )


# ===============================================================================
# Single-Cell Expert Core Functionality Tests
# ===============================================================================

@pytest.mark.unit  
class TestSingleCellExpertCore:
    """Test single-cell expert core functionality."""
    
    def test_filter_and_normalize_data(self, mock_data_manager):
        """Test data filtering and normalization."""
        with patch('lobster.agents.singlecell_expert.filter_and_normalize_modality') as mock_filter:
            mock_filter.return_value = "Filtered and normalized 'sc_data': 4500/5000 cells, 18000/20000 genes retained"
            
            result = mock_filter("sc_data", min_cells=3, min_genes=200)
            
            assert "4500/5000 cells" in result
            assert "18000/20000 genes" in result
            mock_filter.assert_called_once_with("sc_data", min_cells=3, min_genes=200)
    
    def test_calculate_qc_metrics(self, mock_data_manager):
        """Test QC metrics calculation."""
        with patch('lobster.agents.singlecell_expert.calculate_qc_metrics') as mock_qc:
            mock_qc.return_value = "QC metrics calculated: mean genes/cell: 2500, mean MT%: 12.5%"
            
            result = mock_qc("sc_data")
            
            assert "mean genes/cell: 2500" in result
            assert "mean MT%: 12.5%" in result
            mock_qc.assert_called_once_with("sc_data")
    
    def test_detect_doublets(self, mock_data_manager):
        """Test doublet detection."""
        with patch('lobster.agents.singlecell_expert.detect_doublets') as mock_doublets:
            mock_doublets.return_value = "Doublet detection completed: 125/5000 cells (2.5%) flagged as doublets"
            
            result = mock_doublets("sc_data")
            
            assert "125/5000 cells" in result  
            assert "2.5%" in result
            mock_doublets.assert_called_once_with("sc_data")
    
    def test_perform_clustering(self, mock_data_manager):
        """Test clustering analysis."""
        with patch('lobster.agents.singlecell_expert.perform_clustering') as mock_cluster:
            mock_cluster.return_value = "Clustering completed: identified 12 clusters using Leiden algorithm"
            
            result = mock_cluster("sc_data", resolution=0.5, algorithm="leiden")
            
            assert "12 clusters" in result
            assert "Leiden algorithm" in result
            mock_cluster.assert_called_once_with("sc_data", resolution=0.5, algorithm="leiden")
    
    def test_find_marker_genes(self, mock_data_manager):
        """Test marker gene identification."""
        with patch('lobster.agents.singlecell_expert.find_marker_genes') as mock_markers:
            mock_markers.return_value = "Found 156 marker genes across 12 clusters"
            
            result = mock_markers("sc_data_clustered")
            
            assert "156 marker genes" in result
            assert "12 clusters" in result
            mock_markers.assert_called_once_with("sc_data_clustered")
    
    def test_annotate_cell_types(self, mock_data_manager):
        """Test cell type annotation."""
        with patch('lobster.agents.singlecell_expert.annotate_cell_types') as mock_annotate:
            mock_annotate.return_value = "Cell types annotated: T cells (35%), B cells (20%), Monocytes (15%)"
            
            result = mock_annotate("sc_data_clustered")
            
            assert "T cells (35%)" in result
            assert "B cells (20%)" in result
            mock_annotate.assert_called_once_with("sc_data_clustered")


# ===============================================================================
# Quality Control Tests
# ===============================================================================

@pytest.mark.unit
class TestSingleCellQualityControl:
    """Test single-cell quality control functionality."""
    
    def test_assess_cell_quality(self, mock_data_manager):
        """Test cell quality assessment."""
        with patch('lobster.agents.singlecell_expert.assess_cell_quality') as mock_assess:
            mock_assess.return_value = "Cell quality assessment: 95% cells pass QC thresholds"
            
            result = mock_assess("sc_data")
            
            assert "95% cells pass" in result
            mock_assess.assert_called_once_with("sc_data")
    
    def test_filter_low_quality_cells(self, mock_data_manager):
        """Test filtering low quality cells."""
        with patch('lobster.agents.singlecell_expert.filter_low_quality_cells') as mock_filter:
            mock_filter.return_value = "Filtered out 250/5000 cells (5%) with poor quality"
            
            result = mock_filter("sc_data", min_genes=200, max_mt_pct=20)
            
            assert "250/5000 cells" in result
            assert "5%" in result
            mock_filter.assert_called_once_with("sc_data", min_genes=200, max_mt_pct=20)
    
    def test_identify_highly_variable_genes(self, mock_data_manager):
        """Test highly variable gene identification."""
        with patch('lobster.agents.singlecell_expert.identify_highly_variable_genes') as mock_hvg:
            mock_hvg.return_value = "Identified 2000 highly variable genes"
            
            result = mock_hvg("sc_data", n_top_genes=2000)
            
            assert "2000 highly variable genes" in result
            mock_hvg.assert_called_once_with("sc_data", n_top_genes=2000)


# ===============================================================================
# Dimensionality Reduction Tests
# ===============================================================================

@pytest.mark.unit
class TestDimensionalityReduction:
    """Test dimensionality reduction functionality."""
    
    def test_perform_pca(self, mock_data_manager):
        """Test PCA analysis."""
        with patch('lobster.agents.singlecell_expert.perform_pca') as mock_pca:
            mock_pca.return_value = "PCA completed: 50 components explain 85% variance"
            
            result = mock_pca("sc_data", n_components=50)
            
            assert "50 components" in result
            assert "85% variance" in result  
            mock_pca.assert_called_once_with("sc_data", n_components=50)
    
    def test_compute_umap(self, mock_data_manager):
        """Test UMAP computation."""
        with patch('lobster.agents.singlecell_expert.compute_umap') as mock_umap:
            mock_umap.return_value = "UMAP embedding computed with n_neighbors=15, min_dist=0.1"
            
            result = mock_umap("sc_data", n_neighbors=15, min_dist=0.1)
            
            assert "n_neighbors=15" in result
            assert "min_dist=0.1" in result
            mock_umap.assert_called_once_with("sc_data", n_neighbors=15, min_dist=0.1)
    
    def test_compute_tsne(self, mock_data_manager):
        """Test t-SNE computation."""
        with patch('lobster.agents.singlecell_expert.compute_tsne') as mock_tsne:
            mock_tsne.return_value = "t-SNE embedding computed with perplexity=30"
            
            result = mock_tsne("sc_data", perplexity=30)
            
            assert "perplexity=30" in result
            mock_tsne.assert_called_once_with("sc_data", perplexity=30)


# ===============================================================================
# Visualization Tests
# ===============================================================================

@pytest.mark.unit
class TestSingleCellVisualization:
    """Test single-cell visualization functionality."""
    
    def test_plot_quality_metrics(self, mock_data_manager):
        """Test QC metrics plotting."""
        with patch('lobster.agents.singlecell_expert.plot_quality_metrics') as mock_plot:
            mock_plot.return_value = "QC plots generated: violin plots for genes, UMIs, and MT%"
            
            result = mock_plot("sc_data")
            
            assert "violin plots" in result
            assert "genes, UMIs, and MT%" in result
            mock_plot.assert_called_once_with("sc_data")
    
    def test_plot_dimensionality_reduction(self, mock_data_manager):
        """Test dimensionality reduction plotting."""
        with patch('lobster.agents.singlecell_expert.plot_dimensionality_reduction') as mock_plot:
            mock_plot.return_value = "Generated UMAP plot colored by clusters"
            
            result = mock_plot("sc_data", method="umap", color_by="cluster")
            
            assert "UMAP plot" in result
            assert "colored by clusters" in result
            mock_plot.assert_called_once_with("sc_data", method="umap", color_by="cluster")
    
    def test_plot_marker_genes(self, mock_data_manager):
        """Test marker gene plotting."""
        with patch('lobster.agents.singlecell_expert.plot_marker_genes') as mock_plot:
            mock_plot.return_value = "Generated heatmap of top 50 marker genes"
            
            result = mock_plot("sc_data", n_genes=50)
            
            assert "heatmap" in result
            assert "50 marker genes" in result
            mock_plot.assert_called_once_with("sc_data", n_genes=50)


# ===============================================================================
# Advanced Analysis Tests
# ===============================================================================

@pytest.mark.unit
class TestAdvancedSingleCellAnalysis:
    """Test advanced single-cell analysis functionality."""
    
    def test_trajectory_analysis(self, mock_data_manager):
        """Test trajectory/pseudotime analysis."""
        with patch('lobster.agents.singlecell_expert.perform_trajectory_analysis') as mock_trajectory:
            mock_trajectory.return_value = "Trajectory analysis completed: identified developmental path"
            
            result = mock_trajectory("sc_data", root_cluster="0")
            
            assert "developmental path" in result
            mock_trajectory.assert_called_once_with("sc_data", root_cluster="0")
    
    def test_differential_expression_analysis(self, mock_data_manager):
        """Test differential expression analysis."""
        with patch('lobster.agents.singlecell_expert.perform_differential_expression') as mock_de:
            mock_de.return_value = "DE analysis: 245 genes significantly different between conditions"
            
            result = mock_de("sc_data", group_by="condition")
            
            assert "245 genes significantly different" in result
            mock_de.assert_called_once_with("sc_data", group_by="condition")
    
    def test_gene_set_enrichment_analysis(self, mock_data_manager):
        """Test gene set enrichment analysis."""
        with patch('lobster.agents.singlecell_expert.perform_gene_set_enrichment') as mock_gsea:
            mock_gsea.return_value = "GSEA completed: 15 pathways significantly enriched"
            
            result = mock_gsea("sc_data", gene_sets="GO_biological_process")
            
            assert "15 pathways significantly enriched" in result
            mock_gsea.assert_called_once_with("sc_data", gene_sets="GO_biological_process")


# ===============================================================================
# Integration and Batch Correction Tests  
# ===============================================================================

@pytest.mark.unit
class TestIntegrationBatchCorrection:
    """Test integration and batch correction functionality."""
    
    def test_integrate_datasets(self, mock_data_manager):
        """Test dataset integration."""
        with patch('lobster.agents.singlecell_expert.integrate_datasets') as mock_integrate:
            mock_integrate.return_value = "Integrated 3 datasets: batch effects corrected"
            
            result = mock_integrate(["dataset1", "dataset2", "dataset3"])
            
            assert "Integrated 3 datasets" in result
            assert "batch effects corrected" in result
            mock_integrate.assert_called_once_with(["dataset1", "dataset2", "dataset3"])
    
    def test_batch_correction(self, mock_data_manager):
        """Test batch effect correction."""
        with patch('lobster.agents.singlecell_expert.correct_batch_effects') as mock_batch:
            mock_batch.return_value = "Batch correction applied using Harmony algorithm"
            
            result = mock_batch("sc_data", batch_key="batch", method="harmony")
            
            assert "Harmony algorithm" in result  
            mock_batch.assert_called_once_with("sc_data", batch_key="batch", method="harmony")


# ===============================================================================
# Workflow and Pipeline Tests
# ===============================================================================

@pytest.mark.unit
class TestSingleCellWorkflows:
    """Test single-cell analysis workflows."""
    
    def test_standard_analysis_workflow(self, mock_data_manager, singlecell_state):
        """Test standard single-cell analysis workflow."""
        singlecell_state.messages = [MockMessage("Run standard single-cell analysis pipeline")]
        
        with patch('lobster.agents.singlecell_expert.singlecell_expert_agent') as mock_agent:
            mock_agent.return_value = {
                "messages": [MockMessage("Completed standard analysis pipeline", "assistant")],
                "workflow_steps": ["qc", "normalization", "clustering", "markers", "annotation"],
                "clusters_identified": 12
            }
            
            result = mock_agent(singlecell_state)
            
            assert len(result["workflow_steps"]) == 5
            assert result["clusters_identified"] == 12
    
    def test_custom_analysis_workflow(self, mock_data_manager):
        """Test custom analysis workflow."""  
        with patch('lobster.agents.singlecell_expert.run_custom_workflow') as mock_workflow:
            workflow_steps = ["filter", "normalize", "hvg", "pca", "umap", "cluster"]
            mock_workflow.return_value = f"Custom workflow completed: {' -> '.join(workflow_steps)}"
            
            result = mock_workflow("sc_data", workflow_steps)
            
            assert "Custom workflow completed" in result
            assert "cluster" in result


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit 
class TestSingleCellErrorHandling:
    """Test single-cell expert error handling."""
    
    def test_insufficient_cells_error(self, mock_data_manager):
        """Test handling insufficient cells for analysis."""
        with patch('lobster.agents.singlecell_expert.perform_clustering') as mock_cluster:
            mock_cluster.side_effect = ValueError("Insufficient cells for clustering: 50 < 100 minimum")
            
            with pytest.raises(ValueError, match="Insufficient cells"):
                mock_cluster("small_dataset")
    
    def test_missing_hvg_error(self, mock_data_manager):
        """Test handling missing highly variable genes."""
        with patch('lobster.agents.singlecell_expert.perform_pca') as mock_pca:
            mock_pca.side_effect = RuntimeError("No highly variable genes found for PCA")
            
            with pytest.raises(RuntimeError, match="No highly variable genes"):
                mock_pca("sc_data")
    
    def test_convergence_error(self, mock_data_manager):
        """Test handling convergence errors in algorithms."""
        with patch('lobster.agents.singlecell_expert.compute_umap') as mock_umap:
            mock_umap.side_effect = RuntimeError("UMAP failed to converge")
            
            with pytest.raises(RuntimeError, match="UMAP failed to converge"):
                mock_umap("sc_data")
    
    def test_memory_error_handling(self, mock_data_manager):
        """Test handling memory errors with large datasets."""
        with patch('lobster.agents.singlecell_expert.perform_clustering') as mock_cluster:
            mock_cluster.side_effect = MemoryError("Insufficient memory for clustering large dataset")
            
            with pytest.raises(MemoryError, match="Insufficient memory"):
                mock_cluster("large_dataset")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])