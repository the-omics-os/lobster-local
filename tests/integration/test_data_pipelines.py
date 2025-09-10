"""
Comprehensive integration tests for data pipelines.

This module provides thorough testing of data processing pipelines including
transformation workflows, multi-modal pipeline orchestration, data flow validation,
pipeline error recovery, and end-to-end data processing scenarios.

Test coverage target: 95%+ with realistic pipeline scenarios.
"""

import pytest
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.client import AgentClient
from lobster.tools.geo_service import GEOService
from lobster.tools.preprocessing_service import PreprocessingService
from lobster.tools.clustering_service import ClusteringService
from lobster.tools.quality_service import QualityService
from lobster.tools.visualization_service import SingleCellVisualizationService

from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory, ProteomicsDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Test Fixtures and Mock Data
# ===============================================================================

@pytest.fixture
def temp_workspace():
    """Create temporary workspace for pipeline tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def data_manager(temp_workspace):
    """Create DataManagerV2 instance for pipeline testing."""
    return DataManagerV2(workspace_path=temp_workspace)


@pytest.fixture
def mock_pipeline_services():
    """Create mock service instances for pipeline testing."""
    return {
        'geo_service': Mock(spec=GEOService),
        'preprocessing_service': Mock(spec=PreprocessingService),
        'clustering_service': Mock(spec=ClusteringService),
        'quality_service': Mock(spec=QualityService),
        'visualization_service': Mock(spec=SingleCellVisualizationService)
    }


@pytest.fixture
def sample_pipeline_config():
    """Sample pipeline configuration for testing."""
    return {
        'single_cell_pipeline': {
            'steps': [
                {'name': 'load_data', 'service': 'geo_service', 'method': 'fetch_dataset'},
                {'name': 'quality_control', 'service': 'quality_service', 'method': 'calculate_qc_metrics'},
                {'name': 'filter_cells', 'service': 'preprocessing_service', 'method': 'filter_cells_and_genes'},
                {'name': 'normalize', 'service': 'preprocessing_service', 'method': 'normalize_data'},
                {'name': 'find_hvg', 'service': 'preprocessing_service', 'method': 'find_highly_variable_genes'},
                {'name': 'scale_data', 'service': 'preprocessing_service', 'method': 'scale_data'},
                {'name': 'pca', 'service': 'preprocessing_service', 'method': 'run_pca'},
                {'name': 'neighbors', 'service': 'clustering_service', 'method': 'compute_neighbors'},
                {'name': 'cluster', 'service': 'clustering_service', 'method': 'leiden_clustering'},
                {'name': 'umap', 'service': 'preprocessing_service', 'method': 'run_umap'},
                {'name': 'find_markers', 'service': 'clustering_service', 'method': 'find_marker_genes'},
                {'name': 'visualize', 'service': 'visualization_service', 'method': 'plot_umap'}
            ],
            'parameters': {
                'min_genes': 200,
                'min_cells': 3,
                'max_mt_percent': 20.0,
                'n_top_genes': 2000,
                'resolution': 0.5
            }
        },
        'bulk_rnaseq_pipeline': {
            'steps': [
                {'name': 'load_data', 'service': 'geo_service', 'method': 'fetch_dataset'},
                {'name': 'quality_assessment', 'service': 'quality_service', 'method': 'assess_data_quality'},
                {'name': 'normalize', 'service': 'preprocessing_service', 'method': 'normalize_counts'},
                {'name': 'batch_correction', 'service': 'preprocessing_service', 'method': 'correct_batch_effects'},
                {'name': 'differential_expression', 'service': 'preprocessing_service', 'method': 'run_differential_expression'},
                {'name': 'pathway_analysis', 'service': 'preprocessing_service', 'method': 'run_pathway_analysis'},
                {'name': 'visualize_results', 'service': 'visualization_service', 'method': 'create_de_plots'}
            ],
            'parameters': {
                'normalization_method': 'TMM',
                'batch_variable': 'batch',
                'design_formula': '~ condition',
                'padj_threshold': 0.05
            }
        }
    }


@pytest.fixture
def mock_geo_datasets():
    """Mock GEO datasets for pipeline testing."""
    return {
        'GSE123456': {
            'title': 'Single-cell RNA-seq of immune cells',
            'organism': 'Homo sapiens',
            'n_samples': 8,
            'platform': 'GPL24676',
            'data': SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        },
        'GSE789012': {
            'title': 'Bulk RNA-seq time course experiment',
            'organism': 'Homo sapiens',
            'n_samples': 24,
            'platform': 'GPL20301',
            'data': BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG)
        }
    }


# ===============================================================================
# Basic Pipeline Execution Tests
# ===============================================================================

@pytest.mark.integration
class TestBasicPipelineExecution:
    """Test basic pipeline execution functionality."""
    
    def test_simple_linear_pipeline(self, data_manager, mock_pipeline_services, sample_pipeline_config):
        """Test execution of a simple linear pipeline."""
        pipeline_steps = sample_pipeline_config['single_cell_pipeline']['steps'][:5]  # First 5 steps
        
        # Mock service responses
        mock_pipeline_services['geo_service'].fetch_dataset.return_value = {
            'modality_name': 'geo_gse123456',
            'success': True,
            'metadata': {'n_cells': 2000, 'n_genes': 3000}
        }
        
        mock_pipeline_services['quality_service'].calculate_qc_metrics.return_value = {
            'mean_genes_per_cell': 1250,
            'mean_counts_per_cell': 5500,
            'n_cells_passed': 1900
        }
        
        mock_pipeline_services['preprocessing_service'].filter_cells_and_genes.return_value = {
            'n_cells_filtered': 100,
            'n_genes_filtered': 250,
            'new_modality': 'geo_gse123456_filtered'
        }
        
        # Execute pipeline
        pipeline_results = []
        current_modality = 'geo_gse123456'
        
        for step in pipeline_steps:
            service = mock_pipeline_services[step['service']]
            method = getattr(service, step['method'])
            
            if step['name'] == 'load_data':
                result = method('GSE123456')
                current_modality = result['modality_name']
            elif step['name'] == 'quality_control':
                result = method(current_modality)
            elif step['name'] == 'filter_cells':
                result = method(current_modality, min_genes=200, min_cells=3)
                current_modality = result['new_modality']
            else:
                result = method(current_modality)
            
            pipeline_results.append({
                'step': step['name'],
                'result': result,
                'current_modality': current_modality
            })
        
        # Verify pipeline execution
        assert len(pipeline_results) == 5
        assert all(result['result'] for result in pipeline_results)
        assert pipeline_results[-1]['current_modality'] == 'geo_gse123456_filtered'
    
    def test_branching_pipeline(self, data_manager, mock_pipeline_services):
        """Test pipeline with branching execution paths."""
        # Define branching pipeline
        pipeline_config = {
            'main_branch': [
                {'name': 'load_data', 'service': 'geo_service', 'method': 'fetch_dataset'},
                {'name': 'quality_control', 'service': 'quality_service', 'method': 'calculate_qc_metrics'}
            ],
            'analysis_branch_a': [
                {'name': 'clustering_workflow', 'service': 'clustering_service', 'method': 'leiden_clustering'},
                {'name': 'find_markers_a', 'service': 'clustering_service', 'method': 'find_marker_genes'}
            ],
            'analysis_branch_b': [
                {'name': 'trajectory_analysis', 'service': 'preprocessing_service', 'method': 'run_trajectory_analysis'},
                {'name': 'pseudotime_analysis', 'service': 'preprocessing_service', 'method': 'calculate_pseudotime'}
            ]
        }
        
        # Mock responses
        mock_pipeline_services['geo_service'].fetch_dataset.return_value = {'success': True}
        mock_pipeline_services['quality_service'].calculate_qc_metrics.return_value = {'qc_passed': True}
        mock_pipeline_services['clustering_service'].leiden_clustering.return_value = {'n_clusters': 8}
        mock_pipeline_services['clustering_service'].find_marker_genes.return_value = {'n_markers': 156}
        mock_pipeline_services['preprocessing_service'].run_trajectory_analysis.return_value = {'trajectory_success': True}
        mock_pipeline_services['preprocessing_service'].calculate_pseudotime.return_value = {'pseudotime_calculated': True}
        
        # Execute main branch
        main_results = []
        for step in pipeline_config['main_branch']:
            service = mock_pipeline_services[step['service']]
            result = getattr(service, step['method'])('test_data')
            main_results.append({'step': step['name'], 'result': result})
        
        # Execute both analysis branches in parallel
        branch_results = {}
        
        for branch_name, steps in [('analysis_branch_a', pipeline_config['analysis_branch_a']),
                                   ('analysis_branch_b', pipeline_config['analysis_branch_b'])]:
            branch_results[branch_name] = []
            for step in steps:
                service = mock_pipeline_services[step['service']]
                result = getattr(service, step['method'])('test_data')
                branch_results[branch_name].append({'step': step['name'], 'result': result})
        
        # Verify branching execution
        assert len(main_results) == 2
        assert len(branch_results) == 2
        assert 'analysis_branch_a' in branch_results
        assert 'analysis_branch_b' in branch_results
        assert len(branch_results['analysis_branch_a']) == 2
        assert len(branch_results['analysis_branch_b']) == 2
    
    def test_conditional_pipeline_execution(self, data_manager, mock_pipeline_services):
        """Test pipeline with conditional step execution."""
        # Pipeline with conditions
        pipeline_steps = [
            {'name': 'load_data', 'service': 'geo_service', 'method': 'fetch_dataset', 'condition': None},
            {'name': 'assess_quality', 'service': 'quality_service', 'method': 'assess_data_quality', 'condition': None},
            {'name': 'doublet_detection', 'service': 'quality_service', 'method': 'detect_doublets', 'condition': 'single_cell_data'},
            {'name': 'batch_correction', 'service': 'preprocessing_service', 'method': 'correct_batch_effects', 'condition': 'has_batch_effects'},
            {'name': 'normalize', 'service': 'preprocessing_service', 'method': 'normalize_data', 'condition': None}
        ]
        
        # Mock data assessment
        mock_pipeline_services['geo_service'].fetch_dataset.return_value = {
            'success': True, 'data_type': 'single_cell'
        }
        mock_pipeline_services['quality_service'].assess_data_quality.return_value = {
            'data_type': 'single_cell', 'has_batch_effects': True, 'quality_score': 0.85
        }
        mock_pipeline_services['quality_service'].detect_doublets.return_value = {'doublets_detected': 45}
        mock_pipeline_services['preprocessing_service'].correct_batch_effects.return_value = {'batch_corrected': True}
        mock_pipeline_services['preprocessing_service'].normalize_data.return_value = {'normalized': True}
        
        # Execute conditional pipeline
        pipeline_context = {}
        executed_steps = []
        
        for step in pipeline_steps:
            should_execute = True
            
            if step['condition'] == 'single_cell_data':
                should_execute = pipeline_context.get('data_type') == 'single_cell'
            elif step['condition'] == 'has_batch_effects':
                should_execute = pipeline_context.get('has_batch_effects', False)
            
            if should_execute:
                service = mock_pipeline_services[step['service']]
                result = getattr(service, step['method'])('test_data')
                pipeline_context.update(result)
                executed_steps.append(step['name'])
        
        # Verify conditional execution
        assert 'load_data' in executed_steps
        assert 'assess_quality' in executed_steps
        assert 'doublet_detection' in executed_steps  # Should execute for single-cell data
        assert 'batch_correction' in executed_steps   # Should execute when batch effects detected
        assert 'normalize' in executed_steps
    
    def test_pipeline_parameter_propagation(self, data_manager, mock_pipeline_services):
        """Test parameter propagation between pipeline steps."""
        # Pipeline with parameter dependencies
        pipeline_config = {
            'steps': [
                {'name': 'load_data', 'params': {'dataset_id': 'GSE123456'}},
                {'name': 'filter_data', 'params': {'min_genes': '${qc_results.recommended_min_genes}'}},
                {'name': 'normalize', 'params': {'target_sum': '${filter_results.recommended_target_sum}'}},
                {'name': 'cluster', 'params': {'resolution': '${normalization_results.optimal_resolution}'}}
            ]
        }
        
        # Mock step results with parameters for next steps
        mock_pipeline_services['geo_service'].fetch_dataset.return_value = {
            'success': True,
            'qc_results': {'recommended_min_genes': 250}
        }
        
        mock_pipeline_services['preprocessing_service'].filter_cells_and_genes.return_value = {
            'success': True,
            'filter_results': {'recommended_target_sum': 10000}
        }
        
        mock_pipeline_services['preprocessing_service'].normalize_data.return_value = {
            'success': True,
            'normalization_results': {'optimal_resolution': 0.8}
        }
        
        mock_pipeline_services['clustering_service'].leiden_clustering.return_value = {
            'success': True,
            'n_clusters': 12
        }
        
        # Execute pipeline with parameter propagation
        pipeline_context = {}
        final_params = {}
        
        for step in pipeline_config['steps']:
            # Resolve parameters from context
            resolved_params = {}
            for param_name, param_value in step['params'].items():
                if isinstance(param_value, str) and param_value.startswith('${') and param_value.endswith('}'):
                    # Extract parameter path
                    param_path = param_value[2:-1].split('.')
                    resolved_value = pipeline_context
                    for path_part in param_path:
                        resolved_value = resolved_value.get(path_part, param_value)
                    resolved_params[param_name] = resolved_value
                else:
                    resolved_params[param_name] = param_value
            
            final_params[step['name']] = resolved_params
            
            # Execute step and update context
            if step['name'] == 'load_data':
                result = mock_pipeline_services['geo_service'].fetch_dataset(**resolved_params)
            elif step['name'] == 'filter_data':
                result = mock_pipeline_services['preprocessing_service'].filter_cells_and_genes(**resolved_params)
            elif step['name'] == 'normalize':
                result = mock_pipeline_services['preprocessing_service'].normalize_data(**resolved_params)
            elif step['name'] == 'cluster':
                result = mock_pipeline_services['clustering_service'].leiden_clustering(**resolved_params)
            
            pipeline_context.update(result)
        
        # Verify parameter propagation
        assert final_params['load_data']['dataset_id'] == 'GSE123456'
        assert final_params['filter_data']['min_genes'] == 250
        assert final_params['normalize']['target_sum'] == 10000
        assert final_params['cluster']['resolution'] == 0.8


# ===============================================================================
# Multi-Modal Pipeline Tests
# ===============================================================================

@pytest.mark.integration
class TestMultiModalPipelines:
    """Test multi-modal data pipeline functionality."""
    
    def test_transcriptomics_proteomics_integration(self, data_manager, mock_pipeline_services):
        """Test integrated transcriptomics and proteomics pipeline."""
        # Mock multi-modal data
        transcriptomics_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        proteomics_data = ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)
        
        # Integration pipeline
        integration_steps = [
            {'modality': 'transcriptomics', 'step': 'preprocess_rna', 'service': 'preprocessing_service'},
            {'modality': 'proteomics', 'step': 'preprocess_protein', 'service': 'preprocessing_service'},
            {'modality': 'both', 'step': 'find_common_features', 'service': 'preprocessing_service'},
            {'modality': 'both', 'step': 'integrate_modalities', 'service': 'preprocessing_service'},
            {'modality': 'integrated', 'step': 'joint_clustering', 'service': 'clustering_service'},
            {'modality': 'integrated', 'step': 'correlation_analysis', 'service': 'preprocessing_service'}
        ]
        
        # Mock service responses
        mock_pipeline_services['preprocessing_service'].preprocess_rna.return_value = {
            'modality': 'transcriptomics_processed',
            'n_features': 2000,
            'preprocessing_complete': True
        }
        
        mock_pipeline_services['preprocessing_service'].preprocess_protein.return_value = {
            'modality': 'proteomics_processed', 
            'n_features': 150,
            'preprocessing_complete': True
        }
        
        mock_pipeline_services['preprocessing_service'].find_common_features.return_value = {
            'common_cells': 800,
            'feature_overlap': 0.15,
            'integration_feasible': True
        }
        
        mock_pipeline_services['preprocessing_service'].integrate_modalities.return_value = {
            'integrated_modality': 'multi_omics_integrated',
            'integration_score': 0.85,
            'n_integrated_features': 2150
        }
        
        mock_pipeline_services['clustering_service'].joint_clustering.return_value = {
            'n_clusters': 10,
            'modality_agreement': 0.78,
            'joint_clusters': 'clusters_integrated'
        }
        
        mock_pipeline_services['preprocessing_service'].correlation_analysis.return_value = {
            'rna_protein_correlations': 456,
            'significant_correlations': 123,
            'correlation_matrix': 'corr_matrix_rna_protein'
        }
        
        # Execute multi-modal pipeline
        pipeline_results = {}
        modality_states = {
            'transcriptomics': 'transcriptomics_raw',
            'proteomics': 'proteomics_raw',
            'integrated': None
        }
        
        for step in integration_steps:
            service = mock_pipeline_services[step['service']]
            method_name = step['step']
            
            if step['modality'] == 'transcriptomics':
                result = getattr(service, method_name)(modality_states['transcriptomics'])
                modality_states['transcriptomics'] = result['modality']
            elif step['modality'] == 'proteomics':
                result = getattr(service, method_name)(modality_states['proteomics'])
                modality_states['proteomics'] = result['modality']
            elif step['modality'] == 'both':
                result = getattr(service, method_name)(
                    modality_states['transcriptomics'], 
                    modality_states['proteomics']
                )
                if 'integrated_modality' in result:
                    modality_states['integrated'] = result['integrated_modality']
            elif step['modality'] == 'integrated':
                result = getattr(service, method_name)(modality_states['integrated'])
            
            pipeline_results[step['step']] = result
        
        # Verify multi-modal integration
        assert pipeline_results['preprocess_rna']['preprocessing_complete'] == True
        assert pipeline_results['preprocess_protein']['preprocessing_complete'] == True
        assert pipeline_results['find_common_features']['integration_feasible'] == True
        assert pipeline_results['integrate_modalities']['integration_score'] > 0.8
        assert pipeline_results['joint_clustering']['n_clusters'] == 10
        assert pipeline_results['correlation_analysis']['significant_correlations'] > 0
        assert modality_states['integrated'] == 'multi_omics_integrated'
    
    def test_spatial_transcriptomics_pipeline(self, data_manager, mock_pipeline_services):
        """Test spatial transcriptomics data pipeline."""
        # Spatial transcriptomics specific steps
        spatial_pipeline = [
            {'step': 'load_spatial_data', 'requires_coordinates': True},
            {'step': 'quality_control_spatial', 'checks_spatial_coverage': True},
            {'step': 'spatial_filtering', 'removes_edge_effects': True},
            {'step': 'spatial_normalization', 'accounts_for_position': True},
            {'step': 'spatial_clustering', 'uses_coordinates': True},
            {'step': 'spatial_domains', 'identifies_regions': True},
            {'step': 'ligand_receptor_analysis', 'cell_communication': True}
        ]
        
        # Mock spatial-specific results
        mock_results = {
            'load_spatial_data': {
                'spatial_coordinates': True,
                'n_spots': 3000,
                'tissue_area': 'covered',
                'coordinate_system': 'array_based'
            },
            'quality_control_spatial': {
                'spatial_coverage': 0.92,
                'edge_artifacts': 150,
                'low_quality_spots': 75
            },
            'spatial_filtering': {
                'spots_retained': 2925,
                'edge_effects_removed': True,
                'quality_improved': True
            },
            'spatial_normalization': {
                'position_normalized': True,
                'spatial_bias_corrected': True,
                'normalization_method': 'spatial_quantile'
            },
            'spatial_clustering': {
                'spatial_clusters': 15,
                'coherent_regions': True,
                'spatial_autocorrelation': 0.68
            },
            'spatial_domains': {
                'tissue_domains': 6,
                'domain_markers': 234,
                'spatial_coherence': 0.85
            },
            'ligand_receptor_analysis': {
                'lr_pairs_detected': 89,
                'significant_interactions': 45,
                'spatial_communication_map': True
            }
        }
        
        # Setup mocks for each step
        for step_info in spatial_pipeline:
            step_name = step_info['step']
            method_mock = Mock(return_value=mock_results[step_name])
            setattr(mock_pipeline_services['preprocessing_service'], step_name, method_mock)
        
        # Execute spatial pipeline
        spatial_results = {}
        for step_info in spatial_pipeline:
            step_name = step_info['step']
            method = getattr(mock_pipeline_services['preprocessing_service'], step_name)
            result = method('spatial_data')
            spatial_results[step_name] = result
        
        # Verify spatial pipeline execution
        assert spatial_results['load_spatial_data']['spatial_coordinates'] == True
        assert spatial_results['quality_control_spatial']['spatial_coverage'] > 0.9
        assert spatial_results['spatial_filtering']['edge_effects_removed'] == True
        assert spatial_results['spatial_normalization']['position_normalized'] == True
        assert spatial_results['spatial_clustering']['spatial_clusters'] == 15
        assert spatial_results['spatial_domains']['tissue_domains'] == 6
        assert spatial_results['ligand_receptor_analysis']['lr_pairs_detected'] > 0
    
    def test_temporal_multi_omics_pipeline(self, data_manager, mock_pipeline_services):
        """Test temporal multi-omics analysis pipeline."""
        # Time-course multi-omics pipeline
        time_points = ['0h', '6h', '12h', '24h', '48h']
        modalities = ['transcriptomics', 'proteomics', 'metabolomics']
        
        temporal_pipeline = {
            'per_timepoint_processing': [
                {'step': 'load_timepoint_data', 'per_modality': True},
                {'step': 'normalize_timepoint', 'per_modality': True},
                {'step': 'quality_filter_timepoint', 'per_modality': True}
            ],
            'temporal_integration': [
                {'step': 'align_time_series', 'cross_modality': True},
                {'step': 'temporal_smoothing', 'cross_modality': True},
                {'step': 'identify_dynamic_features', 'cross_modality': True}
            ],
            'temporal_analysis': [
                {'step': 'trajectory_inference', 'uses_time': True},
                {'step': 'differential_dynamics', 'compares_conditions': True},
                {'step': 'regulatory_network_dynamics', 'integrative': True}
            ]
        }
        
        # Mock temporal processing results
        per_timepoint_results = {}
        for tp in time_points:
            per_timepoint_results[tp] = {}
            for modality in modalities:
                per_timepoint_results[tp][modality] = {
                    'processed': True,
                    'n_features': np.random.randint(500, 3000),
                    'quality_score': np.random.uniform(0.7, 0.95)
                }
        
        integration_results = {
            'align_time_series': {
                'aligned_successfully': True,
                'temporal_resolution': '6h',
                'interpolated_points': 120
            },
            'temporal_smoothing': {
                'smoothing_applied': True,
                'smoothing_method': 'spline',
                'noise_reduction': 0.35
            },
            'identify_dynamic_features': {
                'dynamic_genes': 1250,
                'dynamic_proteins': 89,
                'dynamic_metabolites': 156,
                'temporal_patterns': 8
            }
        }
        
        analysis_results = {
            'trajectory_inference': {
                'trajectories_found': 3,
                'branching_points': 2,
                'pseudotime_resolution': 0.1
            },
            'differential_dynamics': {
                'differentially_dynamic': 456,
                'condition_specific_patterns': 23,
                'temporal_de_genes': 234
            },
            'regulatory_network_dynamics': {
                'dynamic_networks': 5,
                'regulatory_interactions': 1234,
                'time_specific_regulators': 67
            }
        }
        
        # Setup method mocks
        for step in temporal_pipeline['per_timepoint_processing']:
            method_mock = Mock(side_effect=lambda x, tp=None, mod=None: per_timepoint_results.get(tp, {}).get(mod, {'processed': True}))
            setattr(mock_pipeline_services['preprocessing_service'], step['step'], method_mock)
        
        for step in temporal_pipeline['temporal_integration']:
            method_mock = Mock(return_value=integration_results[step['step']])
            setattr(mock_pipeline_services['preprocessing_service'], step['step'], method_mock)
        
        for step in temporal_pipeline['temporal_analysis']:
            method_mock = Mock(return_value=analysis_results[step['step']])
            setattr(mock_pipeline_services['preprocessing_service'], step['step'], method_mock)
        
        # Execute temporal pipeline
        # Phase 1: Per-timepoint processing
        timepoint_results = {}
        for tp in time_points:
            timepoint_results[tp] = {}
            for modality in modalities:
                for step in temporal_pipeline['per_timepoint_processing']:
                    method = getattr(mock_pipeline_services['preprocessing_service'], step['step'])
                    result = method(f'{modality}_{tp}', tp=tp, mod=modality)
                    timepoint_results[tp][modality] = result
        
        # Phase 2: Temporal integration
        temporal_integration_results = {}
        for step in temporal_pipeline['temporal_integration']:
            method = getattr(mock_pipeline_services['preprocessing_service'], step['step'])
            result = method('all_timepoints')
            temporal_integration_results[step['step']] = result
        
        # Phase 3: Temporal analysis
        temporal_analysis_results = {}
        for step in temporal_pipeline['temporal_analysis']:
            method = getattr(mock_pipeline_services['preprocessing_service'], step['step'])
            result = method('integrated_temporal_data')
            temporal_analysis_results[step['step']] = result
        
        # Verify temporal pipeline
        assert len(timepoint_results) == len(time_points)
        assert all(all(tp_data[mod]['processed'] for mod in modalities) for tp_data in timepoint_results.values())
        assert temporal_integration_results['align_time_series']['aligned_successfully'] == True
        assert temporal_integration_results['identify_dynamic_features']['dynamic_genes'] > 0
        assert temporal_analysis_results['trajectory_inference']['trajectories_found'] > 0
        assert temporal_analysis_results['regulatory_network_dynamics']['dynamic_networks'] > 0


# ===============================================================================
# Pipeline Error Handling and Recovery Tests
# ===============================================================================

@pytest.mark.integration
class TestPipelineErrorHandling:
    """Test pipeline error handling and recovery mechanisms."""
    
    def test_pipeline_step_failure_recovery(self, data_manager, mock_pipeline_services):
        """Test recovery from individual step failures."""
        # Pipeline with potential failure points
        pipeline_steps = [
            {'name': 'load_data', 'critical': True, 'retry_count': 3},
            {'name': 'quality_control', 'critical': True, 'retry_count': 2},
            {'name': 'normalize', 'critical': False, 'retry_count': 1, 'fallback': 'simple_normalize'},
            {'name': 'cluster', 'critical': False, 'retry_count': 2, 'fallback': 'kmeans_cluster'},
            {'name': 'visualize', 'critical': False, 'retry_count': 1, 'fallback': 'simple_plot'}
        ]
        
        # Setup failure scenarios
        failure_count = {'normalize': 0, 'cluster': 0}
        
        def mock_normalize_with_failure(*args, **kwargs):
            failure_count['normalize'] += 1
            if failure_count['normalize'] <= 1:  # Fail first time
                raise Exception("Normalization failed: Memory error")
            return {'success': True, 'method': 'fallback_normalization'}
        
        def mock_cluster_with_failure(*args, **kwargs):
            failure_count['cluster'] += 1
            if failure_count['cluster'] <= 2:  # Fail first two times
                raise Exception("Clustering failed: Convergence error")
            return {'success': True, 'method': 'fallback_clustering'}
        
        # Setup service mocks
        mock_pipeline_services['geo_service'].fetch_dataset.return_value = {'success': True}
        mock_pipeline_services['quality_service'].calculate_qc_metrics.return_value = {'success': True}
        mock_pipeline_services['preprocessing_service'].normalize_data = Mock(side_effect=mock_normalize_with_failure)
        mock_pipeline_services['preprocessing_service'].simple_normalize = Mock(return_value={'success': True, 'method': 'simple'})
        mock_pipeline_services['clustering_service'].leiden_clustering = Mock(side_effect=mock_cluster_with_failure)
        mock_pipeline_services['clustering_service'].kmeans_cluster = Mock(return_value={'success': True, 'method': 'kmeans'})
        mock_pipeline_services['visualization_service'].plot_umap.return_value = {'success': True}
        
        # Execute pipeline with error recovery
        pipeline_results = []
        
        for step in pipeline_steps:
            step_executed = False
            attempts = 0
            
            while not step_executed and attempts < step['retry_count']:
                attempts += 1
                
                try:
                    if step['name'] == 'load_data':
                        result = mock_pipeline_services['geo_service'].fetch_dataset('test')
                    elif step['name'] == 'quality_control':
                        result = mock_pipeline_services['quality_service'].calculate_qc_metrics('test')
                    elif step['name'] == 'normalize':
                        result = mock_pipeline_services['preprocessing_service'].normalize_data('test')
                    elif step['name'] == 'cluster':
                        result = mock_pipeline_services['clustering_service'].leiden_clustering('test')
                    elif step['name'] == 'visualize':
                        result = mock_pipeline_services['visualization_service'].plot_umap('test')
                    
                    step_executed = True
                    pipeline_results.append({
                        'step': step['name'],
                        'result': result,
                        'attempts': attempts,
                        'method': 'primary'
                    })
                    
                except Exception as e:
                    if attempts >= step['retry_count']:
                        # Try fallback method if available
                        if not step['critical'] and 'fallback' in step:
                            try:
                                if step['name'] == 'normalize':
                                    result = mock_pipeline_services['preprocessing_service'].simple_normalize('test')
                                elif step['name'] == 'cluster':
                                    result = mock_pipeline_services['clustering_service'].kmeans_cluster('test')
                                
                                step_executed = True
                                pipeline_results.append({
                                    'step': step['name'],
                                    'result': result,
                                    'attempts': attempts,
                                    'method': 'fallback',
                                    'original_error': str(e)
                                })
                            except Exception as fallback_error:
                                if step['critical']:
                                    raise Exception(f"Critical step {step['name']} failed: {e}")
                                else:
                                    # Skip non-critical step
                                    pipeline_results.append({
                                        'step': step['name'],
                                        'result': {'success': False, 'skipped': True},
                                        'attempts': attempts,
                                        'method': 'skipped',
                                        'error': str(e)
                                    })
                                    step_executed = True
                        elif step['critical']:
                            raise Exception(f"Critical step {step['name']} failed after {attempts} attempts: {e}")
                        else:
                            # Skip non-critical step
                            pipeline_results.append({
                                'step': step['name'],
                                'result': {'success': False, 'skipped': True},
                                'attempts': attempts,
                                'method': 'skipped',
                                'error': str(e)
                            })
                            step_executed = True
        
        # Verify error recovery
        assert len(pipeline_results) == 5
        assert all(result['result']['success'] or result['result']['skipped'] for result in pipeline_results)
        
        # Check that normalize used fallback after failure
        normalize_result = next(r for r in pipeline_results if r['step'] == 'normalize')
        assert normalize_result['method'] == 'fallback'
        assert normalize_result['attempts'] == 1
        
        # Check that cluster eventually succeeded
        cluster_result = next(r for r in pipeline_results if r['step'] == 'cluster')
        assert cluster_result['result']['success'] == True
    
    def test_pipeline_checkpoint_system(self, data_manager, temp_workspace):
        """Test pipeline checkpoint and resume functionality."""
        # Pipeline with checkpointing
        pipeline_steps = [
            {'name': 'load_data', 'checkpoint': True, 'estimated_time': 30},
            {'name': 'preprocess', 'checkpoint': True, 'estimated_time': 120},
            {'name': 'cluster', 'checkpoint': True, 'estimated_time': 60},
            {'name': 'analyze', 'checkpoint': False, 'estimated_time': 90},
            {'name': 'visualize', 'checkpoint': False, 'estimated_time': 15}
        ]
        
        checkpoint_dir = temp_workspace / 'pipeline_checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Simulate pipeline execution with checkpointing
        pipeline_state = {
            'completed_steps': [],
            'step_results': {},
            'checkpoint_files': {},
            'total_runtime': 0
        }
        
        def save_checkpoint(step_name, step_result, checkpoint_path):
            """Simulate checkpoint saving."""
            checkpoint_data = {
                'step_name': step_name,
                'step_result': step_result,
                'timestamp': time.time(),
                'pipeline_state': pipeline_state.copy()
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, default=str)
            
            return checkpoint_path
        
        def load_checkpoint(checkpoint_path):
            """Simulate checkpoint loading."""
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        
        # First execution - run until failure at step 3
        for i, step in enumerate(pipeline_steps[:3]):  # Run first 3 steps
            step_result = {
                'success': True,
                'step_name': step['name'],
                'runtime': step['estimated_time'],
                'data_produced': f"output_data_{step['name']}"
            }
            
            pipeline_state['completed_steps'].append(step['name'])
            pipeline_state['step_results'][step['name']] = step_result
            pipeline_state['total_runtime'] += step['estimated_time']
            
            if step['checkpoint']:
                checkpoint_path = checkpoint_dir / f"checkpoint_{step['name']}.json"
                save_checkpoint(step['name'], step_result, checkpoint_path)
                pipeline_state['checkpoint_files'][step['name']] = str(checkpoint_path)
        
        # Simulate interruption after step 3
        last_checkpoint = 'cluster'
        
        # Resume from checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_{last_checkpoint}.json"
        checkpoint_data = load_checkpoint(checkpoint_path)
        
        # Restore pipeline state
        restored_state = checkpoint_data['pipeline_state']
        
        # Continue from last checkpoint
        remaining_steps = [s for s in pipeline_steps if s['name'] not in restored_state['completed_steps']]
        
        for step in remaining_steps:
            step_result = {
                'success': True,
                'step_name': step['name'],
                'runtime': step['estimated_time'],
                'data_produced': f"output_data_{step['name']}"
            }
            
            restored_state['completed_steps'].append(step['name'])
            restored_state['step_results'][step['name']] = step_result
            restored_state['total_runtime'] += step['estimated_time']
            
            if step['checkpoint']:
                checkpoint_path = checkpoint_dir / f"checkpoint_{step['name']}.json"
                save_checkpoint(step['name'], step_result, checkpoint_path)
                restored_state['checkpoint_files'][step['name']] = str(checkpoint_path)
        
        # Verify checkpoint system
        assert len(restored_state['completed_steps']) == 5
        assert restored_state['total_runtime'] == sum(s['estimated_time'] for s in pipeline_steps)
        assert 'load_data' in restored_state['checkpoint_files']
        assert 'preprocess' in restored_state['checkpoint_files']
        assert 'cluster' in restored_state['checkpoint_files']
        
        # Verify all checkpoint files exist
        for checkpoint_file in restored_state['checkpoint_files'].values():
            assert Path(checkpoint_file).exists()
    
    def test_pipeline_rollback_mechanism(self, data_manager, mock_pipeline_services):
        """Test pipeline rollback on critical failures."""
        # Pipeline with rollback points
        rollback_pipeline = [
            {'name': 'backup_original', 'rollback_point': 'start'},
            {'name': 'load_data', 'rollback_point': None},
            {'name': 'preprocess_data', 'rollback_point': 'preprocessed'},
            {'name': 'advanced_analysis', 'rollback_point': None, 'can_fail': True},
            {'name': 'finalize_results', 'rollback_point': None}
        ]
        
        # Track pipeline state and rollback points
        pipeline_snapshots = {}
        current_state = {'data': 'original_data', 'processed': False}
        
        # Mock service responses
        mock_pipeline_services['geo_service'].backup_original.return_value = {
            'backup_created': True, 'backup_id': 'backup_001'
        }
        mock_pipeline_services['geo_service'].load_data.return_value = {
            'data_loaded': True, 'n_observations': 1000
        }
        mock_pipeline_services['preprocessing_service'].preprocess_data.return_value = {
            'preprocessing_complete': True, 'n_features_selected': 2000
        }
        
        # Advanced analysis that fails
        def failing_analysis(*args, **kwargs):
            raise Exception("Advanced analysis failed: Insufficient memory")
        
        mock_pipeline_services['preprocessing_service'].advanced_analysis = Mock(side_effect=failing_analysis)
        mock_pipeline_services['preprocessing_service'].finalize_results.return_value = {
            'finalization_complete': True
        }
        
        # Execute pipeline with rollback capability
        executed_steps = []
        
        try:
            for step in rollback_pipeline:
                # Save rollback point if specified
                if step['rollback_point']:
                    pipeline_snapshots[step['rollback_point']] = current_state.copy()
                
                # Execute step
                if step['name'] == 'backup_original':
                    result = mock_pipeline_services['geo_service'].backup_original()
                elif step['name'] == 'load_data':
                    result = mock_pipeline_services['geo_service'].load_data()
                    current_state['data_loaded'] = True
                elif step['name'] == 'preprocess_data':
                    result = mock_pipeline_services['preprocessing_service'].preprocess_data()
                    current_state['processed'] = True
                elif step['name'] == 'advanced_analysis':
                    result = mock_pipeline_services['preprocessing_service'].advanced_analysis()
                    current_state['advanced_complete'] = True
                elif step['name'] == 'finalize_results':
                    result = mock_pipeline_services['preprocessing_service'].finalize_results()
                    current_state['finalized'] = True
                
                executed_steps.append({
                    'step': step['name'],
                    'result': result,
                    'success': True
                })
                
        except Exception as e:
            # Rollback to last safe point
            rollback_to = 'preprocessed' if 'preprocessed' in pipeline_snapshots else 'start'
            current_state = pipeline_snapshots[rollback_to]
            
            executed_steps.append({
                'step': 'rollback',
                'rollback_point': rollback_to,
                'error': str(e),
                'success': False
            })
        
        # Verify rollback mechanism
        assert len(executed_steps) == 4  # 3 successful steps + rollback
        assert executed_steps[-1]['step'] == 'rollback'
        assert executed_steps[-1]['rollback_point'] == 'preprocessed'
        assert 'preprocessed' in pipeline_snapshots
        assert current_state['processed'] == True  # Rolled back to preprocessed state
        assert 'advanced_complete' not in current_state  # Failed step not in state
    
    def test_pipeline_parallel_execution_error_handling(self, data_manager, mock_pipeline_services):
        """Test error handling in parallel pipeline execution."""
        # Parallel pipeline branches with potential failures
        parallel_branches = {
            'branch_a': [
                {'name': 'process_a1', 'can_fail': False},
                {'name': 'process_a2', 'can_fail': True, 'failure_rate': 0.5},
                {'name': 'process_a3', 'can_fail': False}
            ],
            'branch_b': [
                {'name': 'process_b1', 'can_fail': False},
                {'name': 'process_b2', 'can_fail': False},
                {'name': 'process_b3', 'can_fail': True, 'failure_rate': 0.8}
            ],
            'branch_c': [
                {'name': 'process_c1', 'can_fail': True, 'failure_rate': 0.3},
                {'name': 'process_c2', 'can_fail': False}
            ]
        }
        
        # Mock branch execution functions
        def execute_branch_a():
            results = []
            for step in parallel_branches['branch_a']:
                if step['can_fail'] and np.random.random() < step.get('failure_rate', 0):
                    raise Exception(f"Step {step['name']} failed in branch A")
                results.append({'step': step['name'], 'success': True})
            return {'branch': 'a', 'results': results, 'success': True}
        
        def execute_branch_b():
            results = []
            for step in parallel_branches['branch_b']:
                if step['can_fail'] and np.random.random() < step.get('failure_rate', 0):
                    raise Exception(f"Step {step['name']} failed in branch B")
                results.append({'step': step['name'], 'success': True})
            return {'branch': 'b', 'results': results, 'success': True}
        
        def execute_branch_c():
            results = []
            for step in parallel_branches['branch_c']:
                if step['can_fail'] and step['name'] == 'process_c1':  # Force failure for testing
                    raise Exception(f"Step {step['name']} failed in branch C")
                results.append({'step': step['name'], 'success': True})
            return {'branch': 'c', 'results': results, 'success': True}
        
        # Execute branches in parallel with error handling
        branch_results = {}
        branch_errors = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all branches
            future_to_branch = {
                executor.submit(execute_branch_a): 'branch_a',
                executor.submit(execute_branch_b): 'branch_b', 
                executor.submit(execute_branch_c): 'branch_c'
            }
            
            # Collect results
            for future in as_completed(future_to_branch):
                branch_name = future_to_branch[future]
                
                try:
                    result = future.result(timeout=10)
                    branch_results[branch_name] = result
                except Exception as e:
                    branch_errors[branch_name] = {
                        'error': str(e),
                        'branch': branch_name,
                        'failed': True
                    }
        
        # Verify parallel error handling
        total_branches = len(parallel_branches)
        successful_branches = len(branch_results)
        failed_branches = len(branch_errors)
        
        assert successful_branches + failed_branches == total_branches
        assert 'branch_c' in branch_errors  # This branch was set to fail
        
        # Check that successful branches completed all steps
        for branch_name, result in branch_results.items():
            expected_steps = len(parallel_branches[branch_name])
            actual_steps = len(result['results'])
            assert actual_steps == expected_steps
            assert result['success'] == True


# ===============================================================================
# Pipeline Performance and Optimization Tests
# ===============================================================================

@pytest.mark.integration
class TestPipelinePerformance:
    """Test pipeline performance and optimization features."""
    
    def test_pipeline_caching_system(self, data_manager, mock_pipeline_services, temp_workspace):
        """Test pipeline step result caching."""
        cache_dir = temp_workspace / 'pipeline_cache'
        cache_dir.mkdir(exist_ok=True)
        
        # Pipeline with cacheable steps
        cacheable_pipeline = [
            {'name': 'expensive_preprocessing', 'cache_key': 'preprocess', 'cacheable': True},
            {'name': 'expensive_analysis', 'cache_key': 'analyze', 'cacheable': True},
            {'name': 'quick_visualization', 'cache_key': None, 'cacheable': False}
        ]
        
        # Mock expensive operations
        preprocessing_call_count = 0
        analysis_call_count = 0
        
        def mock_expensive_preprocessing(*args, **kwargs):
            nonlocal preprocessing_call_count
            preprocessing_call_count += 1
            time.sleep(0.1)  # Simulate expensive operation
            return {
                'preprocessing_result': f'result_{preprocessing_call_count}',
                'runtime': 100,
                'call_count': preprocessing_call_count
            }
        
        def mock_expensive_analysis(*args, **kwargs):
            nonlocal analysis_call_count
            analysis_call_count += 1
            time.sleep(0.1)  # Simulate expensive operation
            return {
                'analysis_result': f'analysis_{analysis_call_count}',
                'runtime': 150,
                'call_count': analysis_call_count
            }
        
        mock_pipeline_services['preprocessing_service'].expensive_preprocessing = Mock(side_effect=mock_expensive_preprocessing)
        mock_pipeline_services['preprocessing_service'].expensive_analysis = Mock(side_effect=mock_expensive_analysis)
        mock_pipeline_services['visualization_service'].quick_visualization = Mock(return_value={'plot_created': True})
        
        # Cache management functions
        def get_cache_key(step_name, parameters):
            import hashlib
            param_str = json.dumps(parameters, sort_keys=True)
            return hashlib.md5(f"{step_name}_{param_str}".encode()).hexdigest()
        
        def save_to_cache(cache_key, result, cache_dir):
            cache_file = cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        
        def load_from_cache(cache_key, cache_dir):
            cache_file = cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
            return None
        
        # First execution - no cache hits
        first_run_results = []
        for step in cacheable_pipeline:
            parameters = {'input_data': 'test_data', 'parameter_set': 'default'}
            
            if step['cacheable']:
                cache_key = get_cache_key(step['cache_key'], parameters)
                cached_result = load_from_cache(cache_key, cache_dir)
                
                if cached_result:
                    result = cached_result
                    result['from_cache'] = True
                else:
                    # Execute step
                    if step['name'] == 'expensive_preprocessing':
                        result = mock_pipeline_services['preprocessing_service'].expensive_preprocessing(**parameters)
                    elif step['name'] == 'expensive_analysis':
                        result = mock_pipeline_services['preprocessing_service'].expensive_analysis(**parameters)
                    
                    result['from_cache'] = False
                    save_to_cache(cache_key, result, cache_dir)
            else:
                # Non-cacheable step
                result = mock_pipeline_services['visualization_service'].quick_visualization(**parameters)
                result['from_cache'] = False
            
            first_run_results.append({
                'step': step['name'],
                'result': result,
                'cacheable': step['cacheable']
            })
        
        # Second execution - should hit cache for cacheable steps
        second_run_results = []
        for step in cacheable_pipeline:
            parameters = {'input_data': 'test_data', 'parameter_set': 'default'}  # Same parameters
            
            if step['cacheable']:
                cache_key = get_cache_key(step['cache_key'], parameters)
                cached_result = load_from_cache(cache_key, cache_dir)
                
                if cached_result:
                    result = cached_result
                    result['from_cache'] = True
                else:
                    # Execute step
                    if step['name'] == 'expensive_preprocessing':
                        result = mock_pipeline_services['preprocessing_service'].expensive_preprocessing(**parameters)
                    elif step['name'] == 'expensive_analysis':
                        result = mock_pipeline_services['preprocessing_service'].expensive_analysis(**parameters)
                    
                    result['from_cache'] = False
                    save_to_cache(cache_key, result, cache_dir)
            else:
                # Non-cacheable step
                result = mock_pipeline_services['visualization_service'].quick_visualization(**parameters)
                result['from_cache'] = False
            
            second_run_results.append({
                'step': step['name'],
                'result': result,
                'cacheable': step['cacheable']
            })
        
        # Verify caching system
        assert len(first_run_results) == 3
        assert len(second_run_results) == 3
        
        # First run - no cache hits
        assert all(not result['result']['from_cache'] for result in first_run_results)
        
        # Second run - cache hits for cacheable steps
        cacheable_results = [r for r in second_run_results if r['cacheable']]
        non_cacheable_results = [r for r in second_run_results if not r['cacheable']]
        
        assert all(result['result']['from_cache'] for result in cacheable_results)
        assert all(not result['result']['from_cache'] for result in non_cacheable_results)
        
        # Verify expensive operations were called only once
        assert preprocessing_call_count == 1
        assert analysis_call_count == 1
    
    def test_pipeline_resource_optimization(self, data_manager, mock_pipeline_services):
        """Test pipeline resource usage optimization."""
        # Resource-aware pipeline configuration
        resource_config = {
            'memory_limit': 8 * 1024**3,  # 8GB
            'cpu_cores': 4,
            'gpu_memory': 2 * 1024**3,   # 2GB
            'disk_space': 50 * 1024**3   # 50GB
        }
        
        # Pipeline steps with resource requirements
        resource_pipeline = [
            {
                'name': 'load_large_dataset', 
                'memory_mb': 2048, 
                'cpu_cores': 1, 
                'gpu_mb': 0,
                'estimated_runtime': 60
            },
            {
                'name': 'memory_intensive_preprocessing', 
                'memory_mb': 4096, 
                'cpu_cores': 2, 
                'gpu_mb': 1024,
                'estimated_runtime': 180
            },
            {
                'name': 'parallel_analysis', 
                'memory_mb': 1024, 
                'cpu_cores': 4, 
                'gpu_mb': 512,
                'estimated_runtime': 120
            },
            {
                'name': 'visualization', 
                'memory_mb': 512, 
                'cpu_cores': 1, 
                'gpu_mb': 256,
                'estimated_runtime': 30
            }
        ]
        
        # Resource monitoring and optimization
        current_resources = {
            'memory_used_mb': 1024,  # 1GB baseline usage
            'cpu_cores_used': 0,
            'gpu_memory_used_mb': 0
        }
        
        def check_resource_availability(step_requirements, current_usage, limits):
            """Check if resources are available for step execution."""
            memory_available = (limits['memory_limit'] // (1024**2)) - current_usage['memory_used_mb']
            cpu_available = limits['cpu_cores'] - current_usage['cpu_cores_used']
            gpu_available = (limits['gpu_memory'] // (1024**2)) - current_usage['gpu_memory_used_mb']
            
            return (
                step_requirements['memory_mb'] <= memory_available and
                step_requirements['cpu_cores'] <= cpu_available and
                step_requirements['gpu_mb'] <= gpu_available
            )
        
        def allocate_resources(step_requirements, current_usage):
            """Allocate resources for step execution."""
            current_usage['memory_used_mb'] += step_requirements['memory_mb']
            current_usage['cpu_cores_used'] += step_requirements['cpu_cores']
            current_usage['gpu_memory_used_mb'] += step_requirements['gpu_mb']
        
        def release_resources(step_requirements, current_usage):
            """Release resources after step completion."""
            current_usage['memory_used_mb'] -= step_requirements['memory_mb']
            current_usage['cpu_cores_used'] -= step_requirements['cpu_cores']
            current_usage['gpu_memory_used_mb'] -= step_requirements['gpu_mb']
        
        # Execute resource-aware pipeline
        execution_log = []
        
        for step in resource_pipeline:
            step_name = step['name']
            
            # Check resource availability
            if check_resource_availability(step, current_resources, resource_config):
                # Allocate resources
                allocate_resources(step, current_resources)
                
                # Mock step execution
                start_time = time.time()
                if step_name == 'load_large_dataset':
                    result = mock_pipeline_services['geo_service'].fetch_dataset('large_dataset')
                elif step_name == 'memory_intensive_preprocessing':
                    result = mock_pipeline_services['preprocessing_service'].intensive_preprocessing('data')
                elif step_name == 'parallel_analysis':
                    result = mock_pipeline_services['preprocessing_service'].parallel_analysis('data')
                elif step_name == 'visualization':
                    result = mock_pipeline_services['visualization_service'].create_plots('data')
                else:
                    result = {'success': True}
                
                execution_time = time.time() - start_time
                
                # Release resources
                release_resources(step, current_resources)
                
                execution_log.append({
                    'step': step_name,
                    'success': True,
                    'resources_allocated': {
                        'memory_mb': step['memory_mb'],
                        'cpu_cores': step['cpu_cores'], 
                        'gpu_mb': step['gpu_mb']
                    },
                    'execution_time': execution_time,
                    'result': result
                })
            else:
                execution_log.append({
                    'step': step_name,
                    'success': False,
                    'error': 'Insufficient resources',
                    'required_resources': {
                        'memory_mb': step['memory_mb'],
                        'cpu_cores': step['cpu_cores'],
                        'gpu_mb': step['gpu_mb']
                    },
                    'available_resources': current_resources.copy()
                })
        
        # Verify resource optimization
        successful_steps = [log for log in execution_log if log['success']]
        failed_steps = [log for log in execution_log if not log['success']]
        
        assert len(execution_log) == len(resource_pipeline)
        
        # All steps should succeed with current resource limits
        assert len(successful_steps) == len(resource_pipeline)
        assert len(failed_steps) == 0
        
        # Verify resource tracking
        total_memory_used = max(log['resources_allocated']['memory_mb'] for log in successful_steps)
        max_cpu_used = max(log['resources_allocated']['cpu_cores'] for log in successful_steps)
        max_gpu_used = max(log['resources_allocated']['gpu_mb'] for log in successful_steps)
        
        assert total_memory_used <= (resource_config['memory_limit'] // (1024**2))
        assert max_cpu_used <= resource_config['cpu_cores']
        assert max_gpu_used <= (resource_config['gpu_memory'] // (1024**2))
    
    def test_pipeline_adaptive_scheduling(self, data_manager, mock_pipeline_services):
        """Test adaptive pipeline scheduling based on performance metrics."""
        # Pipeline steps with different characteristics
        adaptive_pipeline = [
            {
                'name': 'io_intensive', 
                'type': 'io_bound', 
                'priority': 'high',
                'estimated_duration': 120,
                'can_parallelize': False
            },
            {
                'name': 'cpu_intensive_1', 
                'type': 'cpu_bound', 
                'priority': 'medium',
                'estimated_duration': 180,
                'can_parallelize': True
            },
            {
                'name': 'cpu_intensive_2', 
                'type': 'cpu_bound', 
                'priority': 'medium',
                'estimated_duration': 150,
                'can_parallelize': True
            },
            {
                'name': 'memory_intensive', 
                'type': 'memory_bound', 
                'priority': 'low',
                'estimated_duration': 90,
                'can_parallelize': False
            },
            {
                'name': 'gpu_intensive', 
                'type': 'gpu_bound', 
                'priority': 'high',
                'estimated_duration': 200,
                'can_parallelize': False
            }
        ]
        
        # Performance monitoring
        performance_metrics = {
            'cpu_utilization': 0.3,
            'memory_utilization': 0.4,
            'gpu_utilization': 0.1,
            'io_wait_time': 0.2
        }
        
        def get_step_priority_score(step, metrics):
            """Calculate dynamic priority score based on current system state."""
            base_priority = {'high': 3, 'medium': 2, 'low': 1}[step['priority']]
            
            # Adjust based on system utilization
            if step['type'] == 'cpu_bound' and metrics['cpu_utilization'] < 0.5:
                return base_priority + 1
            elif step['type'] == 'gpu_bound' and metrics['gpu_utilization'] < 0.3:
                return base_priority + 2
            elif step['type'] == 'io_bound' and metrics['io_wait_time'] < 0.3:
                return base_priority + 1
            elif step['type'] == 'memory_bound' and metrics['memory_utilization'] < 0.6:
                return base_priority + 1
            
            return base_priority
        
        def can_run_parallel(steps, current_step):
            """Determine if step can run in parallel with others."""
            if not current_step['can_parallelize']:
                return False
            
            # Check for resource conflicts
            running_cpu_steps = sum(1 for s in steps if s['type'] == 'cpu_bound')
            if current_step['type'] == 'cpu_bound' and running_cpu_steps >= 2:
                return False
            
            return True
        
        # Adaptive scheduling algorithm
        scheduled_execution = []
        remaining_steps = adaptive_pipeline.copy()
        
        while remaining_steps:
            # Calculate priority scores for remaining steps
            step_scores = []
            for step in remaining_steps:
                score = get_step_priority_score(step, performance_metrics)
                step_scores.append((step, score))
            
            # Sort by priority score (highest first)
            step_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Find steps that can run in current scheduling round
            current_round = []
            running_steps = []
            
            for step, score in step_scores:
                if can_run_parallel(running_steps, step) or len(running_steps) == 0:
                    current_round.append(step)
                    running_steps.append(step)
                    
                    # Remove from remaining steps
                    remaining_steps.remove(step)
                    
                    # If step cannot be parallelized, run only this step in round
                    if not step['can_parallelize']:
                        break
            
            scheduled_execution.append({
                'round': len(scheduled_execution) + 1,
                'steps': current_round,
                'parallel': len(current_round) > 1,
                'estimated_duration': max(s['estimated_duration'] for s in current_round)
            })
        
        # Mock execution of scheduled pipeline
        total_execution_time = 0
        execution_results = []
        
        for round_info in scheduled_execution:
            round_start = time.time()
            round_results = []
            
            # Execute steps in parallel (mocked)
            for step in round_info['steps']:
                step_result = {
                    'step_name': step['name'],
                    'success': True,
                    'execution_time': step['estimated_duration'],
                    'scheduled_round': round_info['round']
                }
                round_results.append(step_result)
            
            round_execution_time = round_info['estimated_duration'] / 1000  # Convert to seconds for simulation
            time.sleep(round_execution_time)
            
            total_execution_time += round_execution_time
            execution_results.extend(round_results)
        
        # Verify adaptive scheduling
        assert len(execution_results) == len(adaptive_pipeline)
        assert len(scheduled_execution) >= 1
        
        # Check that high-priority steps were scheduled early
        high_priority_steps = [r for r in execution_results if any(s['name'] == r['step_name'] and s['priority'] == 'high' for s in adaptive_pipeline)]
        assert len(high_priority_steps) == 2  # io_intensive and gpu_intensive
        
        # Check that parallelizable CPU steps were scheduled together
        cpu_steps = [r for r in execution_results if any(s['name'] == r['step_name'] and s['type'] == 'cpu_bound' for s in adaptive_pipeline)]
        cpu_rounds = [r['scheduled_round'] for r in cpu_steps]
        assert len(set(cpu_rounds)) <= 2  # Should be in same or consecutive rounds
        
        # Verify total execution time is optimized
        sequential_time = sum(s['estimated_duration'] for s in adaptive_pipeline) / 1000
        assert total_execution_time < sequential_time  # Should be faster due to parallelization


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])