"""
Unit tests for Manual Cell Type Annotation Service

Tests the core functionality of the manual annotation service including:
- Annotation session initialization
- Cluster assignment and collapsing
- Debris identification  
- Export/import functionality
- Validation and coverage metrics
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from unittest.mock import Mock, patch, MagicMock

from lobster.tools.manual_annotation_service import (
    ManualAnnotationService,
    ClusterInfo,
    AnnotationState
)


@pytest.fixture
def mock_adata():
    """Create mock AnnData object for testing."""
    n_obs, n_vars = 1000, 2000
    
    # Create mock AnnData
    adata = sc.AnnData(
        X=np.random.rand(n_obs, n_vars),
        obs=pd.DataFrame({
            'leiden': np.random.randint(0, 10, n_obs).astype(str),
            'total_counts': np.random.normal(5000, 1000, n_obs),
            'n_genes_by_counts': np.random.normal(1500, 300, n_obs),
            'pct_counts_mt': np.random.normal(15, 5, n_obs)
        }),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    )
    
    # Add UMAP coordinates
    adata.obsm['X_umap'] = np.random.rand(n_obs, 2)
    
    return adata


@pytest.fixture
def annotation_service():
    """Create ManualAnnotationService instance."""
    # Mock the Rich Console to avoid terminal interactions in tests
    with patch('lobster.tools.manual_annotation_service.Console'):
        return ManualAnnotationService()


class TestManualAnnotationService:
    """Test suite for ManualAnnotationService."""
    
    def test_initialization(self, annotation_service):
        """Test service initialization."""
        assert annotation_service.state is None
        assert annotation_service.adata is None
        assert annotation_service.cluster_colors == {}
        assert annotation_service.templates_loaded == {}
    
    def test_initialize_annotation_session(self, annotation_service, mock_adata):
        """Test annotation session initialization."""
        
        # Initialize session
        state = annotation_service.initialize_annotation_session(
            adata=mock_adata,
            cluster_key='leiden'
        )
        
        # Check state was created
        assert isinstance(state, AnnotationState)
        assert annotation_service.state == state
        assert annotation_service.adata == mock_adata
        
        # Check clusters were extracted
        unique_clusters = mock_adata.obs['leiden'].unique()
        assert len(state.clusters) == len(unique_clusters)
        
        # Check cluster info
        for cluster_id in unique_clusters:
            assert str(cluster_id) in state.clusters
            cluster_info = state.clusters[str(cluster_id)]
            assert isinstance(cluster_info, ClusterInfo)
            assert cluster_info.cluster_id == str(cluster_id)
            assert cluster_info.cell_count > 0
            assert cluster_info.color.startswith('#')
            
        # Check initial state
        assert state.cell_type_mapping == {}
        assert state.debris_clusters == set()
        assert state.annotation_history == []
    
    def test_apply_annotations_to_adata(self, annotation_service, mock_adata):
        """Test applying annotations to AnnData object."""
        
        # Initialize session
        state = annotation_service.initialize_annotation_session(mock_adata, 'leiden')
        
        # Add some manual annotations
        state.cell_type_mapping['0'] = 'T cells'
        state.cell_type_mapping['1'] = 'B cells'
        state.debris_clusters.add('9')
        
        # Apply annotations
        adata_annotated = annotation_service.apply_annotations_to_adata(
            adata=mock_adata,
            cluster_key='leiden',
            cell_type_column='cell_type_manual'
        )
        
        # Check annotations were applied
        assert 'cell_type_manual' in adata_annotated.obs.columns
        
        # Check specific annotations
        t_cell_mask = adata_annotated.obs['leiden'] == '0'
        assert all(adata_annotated.obs.loc[t_cell_mask, 'cell_type_manual'] == 'T cells')
        
        b_cell_mask = adata_annotated.obs['leiden'] == '1'
        assert all(adata_annotated.obs.loc[b_cell_mask, 'cell_type_manual'] == 'B cells')
        
        debris_mask = adata_annotated.obs['leiden'] == '9'
        assert all(adata_annotated.obs.loc[debris_mask, 'cell_type_manual'] == 'Debris')
        
        # Check metadata was stored
        assert 'manual_annotation_metadata' in adata_annotated.uns
        metadata = adata_annotated.uns['manual_annotation_metadata']
        assert metadata['cluster_key_used'] == 'leiden'
        assert metadata['cell_type_column'] == 'cell_type_manual'
    
    def test_suggest_debris_clusters(self, annotation_service, mock_adata):
        """Test smart debris cluster suggestions."""
        
        # Create clusters with poor QC metrics
        mock_adata.obs.loc[mock_adata.obs['leiden'] == '0', 'n_genes_by_counts'] = 100  # Very low
        mock_adata.obs.loc[mock_adata.obs['leiden'] == '1', 'pct_counts_mt'] = 60      # Very high
        mock_adata.obs.loc[mock_adata.obs['leiden'] == '2', 'total_counts'] = 200      # Very low
        
        # Get suggestions
        suggestions = annotation_service.suggest_debris_clusters(
            adata=mock_adata,
            min_genes=200,
            max_mt_percent=50,
            min_umi=500
        )
        
        # Check suggestions include clusters with poor QC
        assert '0' in suggestions  # Low gene count
        assert '1' in suggestions  # High MT%
        assert '2' in suggestions  # Low UMI count
    
    def test_validate_annotation_coverage(self, annotation_service, mock_adata):
        """Test annotation coverage validation."""
        
        # Create mock annotations
        mock_adata.obs['cell_type_manual'] = 'Unassigned'
        mock_adata.obs.loc[mock_adata.obs['leiden'].isin(['0', '1']), 'cell_type_manual'] = 'T cells'
        mock_adata.obs.loc[mock_adata.obs['leiden'] == '2', 'cell_type_manual'] = 'B cells'
        mock_adata.obs.loc[mock_adata.obs['leiden'] == '9', 'cell_type_manual'] = 'Debris'
        
        # Validate coverage
        validation = annotation_service.validate_annotation_coverage(
            adata=mock_adata,
            annotation_col='cell_type_manual'
        )
        
        # Check validation results
        assert validation['valid'] is True
        assert validation['total_cells'] == mock_adata.n_obs
        assert validation['unique_cell_types'] >= 2  # At least T cells and B cells
        assert 'T cells' in validation['cell_type_names']
        assert 'B cells' in validation['cell_type_names']
        assert validation['coverage_percentage'] > 0
    
    def test_cluster_info_creation(self):
        """Test ClusterInfo dataclass creation."""
        
        cluster_info = ClusterInfo(
            cluster_id="0",
            color="#ff0000",
            cell_count=100,
            assigned_type="T cells",
            qc_scores={'mean_genes': 1500, 'mean_mt_pct': 10.5}
        )
        
        assert cluster_info.cluster_id == "0"
        assert cluster_info.color == "#ff0000"
        assert cluster_info.cell_count == 100
        assert cluster_info.assigned_type == "T cells"
        assert cluster_info.is_debris is False
        assert cluster_info.qc_scores['mean_genes'] == 1500
        assert cluster_info.notes == ""
    
    def test_annotation_state_creation(self):
        """Test AnnotationState dataclass creation."""
        
        clusters = {
            "0": ClusterInfo("0", "#ff0000", 100),
            "1": ClusterInfo("1", "#00ff00", 150)
        }
        
        state = AnnotationState(
            clusters=clusters,
            cell_type_mapping={"0": "T cells"},
            debris_clusters={"1"},
            annotation_history=[]
        )
        
        assert len(state.clusters) == 2
        assert state.cell_type_mapping["0"] == "T cells"
        assert "1" in state.debris_clusters
        assert state.annotation_history == []
        assert state.current_step == 0
    
    @patch('builtins.open')
    @patch('json.dump')
    def test_export_functionality(self, mock_json_dump, mock_open, annotation_service, mock_adata):
        """Test annotation export functionality."""
        
        # Initialize session with annotations
        state = annotation_service.initialize_annotation_session(mock_adata, 'leiden')
        state.cell_type_mapping['0'] = 'T cells'
        state.cell_type_mapping['1'] = 'B cells'
        state.debris_clusters.add('9')
        
        # Mock Rich console input
        with patch.object(annotation_service.console, 'input', return_value='test_export.json'):
            with patch.object(annotation_service.console, 'print'):
                annotation_service._export_annotations()
        
        # Verify export was called
        mock_open.assert_called_once_with('test_export.json', 'w')
        mock_json_dump.assert_called_once()
        
        # Check exported data structure
        export_call_args = mock_json_dump.call_args
        exported_data = export_call_args[0][0]  # First argument to json.dump
        
        assert 'cell_type_mapping' in exported_data
        assert 'debris_clusters' in exported_data
        assert 'cluster_info' in exported_data
        assert 'export_timestamp' in exported_data
        
        assert exported_data['cell_type_mapping']['0'] == 'T cells'
        assert exported_data['cell_type_mapping']['1'] == 'B cells'
        assert '9' in exported_data['debris_clusters']
    
    @patch('builtins.open')
    @patch('json.load')
    def test_import_functionality(self, mock_json_load, mock_open, annotation_service, mock_adata):
        """Test annotation import functionality."""
        
        # Mock import data
        import_data = {
            'cell_type_mapping': {'0': 'T cells', '1': 'B cells'},
            'debris_clusters': ['9'],
            'cluster_info': {
                '0': {'assigned_type': 'T cells', 'is_debris': False, 'notes': ''},
                '1': {'assigned_type': 'B cells', 'is_debris': False, 'notes': ''},
                '9': {'assigned_type': None, 'is_debris': True, 'notes': 'Low quality'}
            }
        }
        
        mock_json_load.return_value = import_data
        
        # Initialize session
        state = annotation_service.initialize_annotation_session(mock_adata, 'leiden')
        
        # Mock Rich console input and print
        with patch.object(annotation_service.console, 'input', return_value='test_import.json'):
            with patch.object(annotation_service.console, 'print'):
                annotation_service._import_annotations()
        
        # Verify import was called
        mock_open.assert_called_once_with('test_import.json', 'r')
        mock_json_load.assert_called_once()
        
        # Check annotations were imported
        assert state.cell_type_mapping['0'] == 'T cells'
        assert state.cell_type_mapping['1'] == 'B cells'
        assert '9' in state.debris_clusters
    
    def test_color_generation(self, annotation_service, mock_adata):
        """Test cluster color generation."""
        
        # Initialize session
        state = annotation_service.initialize_annotation_session(mock_adata, 'leiden')
        
        # Check colors were generated
        assert len(annotation_service.cluster_colors) > 0
        
        # Check each cluster has a color
        for cluster_id in mock_adata.obs['leiden'].unique():
            assert str(cluster_id) in annotation_service.cluster_colors
            color = annotation_service.cluster_colors[str(cluster_id)]
            assert color.startswith('#')
            assert len(color) == 7  # #RRGGBB format
    
    def test_qc_score_calculation(self, annotation_service, mock_adata):
        """Test QC score calculation for clusters."""
        
        # Initialize session
        state = annotation_service.initialize_annotation_session(mock_adata, 'leiden')
        
        # Check QC scores were calculated
        for cluster_id, cluster_info in state.clusters.items():
            assert 'mean_total_counts' in cluster_info.qc_scores
            assert 'mean_genes' in cluster_info.qc_scores
            assert 'mean_mt_pct' in cluster_info.qc_scores
            
            # Check scores are reasonable
            assert cluster_info.qc_scores['mean_total_counts'] > 0
            assert cluster_info.qc_scores['mean_genes'] > 0
            assert cluster_info.qc_scores['mean_mt_pct'] >= 0


class TestAnnotationWorkflows:
    """Test complete annotation workflows."""
    
    def test_end_to_end_workflow(self, annotation_service, mock_adata):
        """Test complete annotation workflow without Rich interface."""
        
        # Step 1: Initialize session
        state = annotation_service.initialize_annotation_session(mock_adata, 'leiden')
        assert state is not None
        
        # Step 2: Get debris suggestions
        debris_suggestions = annotation_service.suggest_debris_clusters(mock_adata)
        assert isinstance(debris_suggestions, list)
        
        # Step 3: Manual annotations (simulate)
        state.cell_type_mapping['0'] = 'T cells CD4+'
        state.cell_type_mapping['1'] = 'T cells CD8+'  
        state.cell_type_mapping['2'] = 'B cells'
        state.debris_clusters.add('9')
        
        # Step 4: Apply to AnnData
        adata_annotated = annotation_service.apply_annotations_to_adata(
            mock_adata, 'leiden', 'cell_type_manual'
        )
        
        # Step 5: Validate results
        validation = annotation_service.validate_annotation_coverage(
            adata_annotated, 'cell_type_manual'
        )
        
        assert validation['valid'] is True
        assert validation['unique_cell_types'] >= 3
        assert 'T cells CD4+' in validation['cell_type_names']
        assert 'T cells CD8+' in validation['cell_type_names'] 
        assert 'B cells' in validation['cell_type_names']


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_cluster_key(self, annotation_service, mock_adata):
        """Test handling of invalid cluster key."""
        
        with pytest.raises(KeyError):
            annotation_service.initialize_annotation_session(
                mock_adata, 'nonexistent_column'
            )
    
    def test_empty_adata(self, annotation_service):
        """Test handling of empty AnnData object."""
        
        # Create empty AnnData
        empty_adata = sc.AnnData(
            X=np.empty((0, 100)),
            obs=pd.DataFrame(columns=['leiden'])
        )
        
        # Should handle gracefully
        state = annotation_service.initialize_annotation_session(empty_adata, 'leiden')
        assert len(state.clusters) == 0
    
    def test_validation_without_session(self, annotation_service, mock_adata):
        """Test validation methods without initialized session."""
        
        # Should raise error when no session exists
        with pytest.raises(ValueError, match="No annotation session available"):
            annotation_service.apply_annotations_to_adata(mock_adata, 'leiden')
    
    def test_missing_annotation_column(self, annotation_service, mock_adata):
        """Test validation with missing annotation column."""
        
        validation = annotation_service.validate_annotation_coverage(
            mock_adata, 'nonexistent_column'
        )
        
        assert validation['valid'] is False
        assert 'error' in validation


if __name__ == "__main__":
    pytest.main([__file__])
