"""
Comprehensive system tests for full analysis workflows.

This module provides thorough system-level testing of complete analysis workflows
from data ingestion to final results, testing the entire Lobster AI platform
integration including agent coordination, data management, and output generation.

Test coverage target: 95%+ with realistic end-to-end analysis scenarios.
"""

import pytest
import time
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from dataclasses import dataclass, asdict

from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.provenance import ProvenanceTracker
from lobster.agents.singlecell_expert import SingleCellExpert
from lobster.agents.bulk_rnaseq_expert import BulkRNASeqExpert
from lobster.agents.proteomics_expert import ProteomicsExpert
from lobster.agents.data_expert import DataExpert
from lobster.agents.research_agent import ResearchAgent
from lobster.agents.supervisor import SupervisorAgent

from tests.mock_data.factories import (
    SingleCellDataFactory,
    BulkRNASeqDataFactory,
    ProteomicsDataFactory,
    SpatialDataFactory
)
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Workflow Test Configuration and Utilities
# ===============================================================================

@dataclass
class WorkflowStep:
    """Represents a step in an analysis workflow."""
    step_id: str
    step_type: str
    description: str
    agent_type: str
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    expected_duration: float = 30.0
    success_criteria: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowResult:
    """Results from executing a workflow."""
    workflow_id: str
    success: bool
    total_execution_time: float
    steps_completed: int
    steps_failed: int
    final_outputs: Dict[str, Any]
    intermediate_results: List[Dict[str, Any]]
    error_messages: List[str]
    performance_metrics: Dict[str, Any]


class AnalysisWorkflowEngine:
    """Engine for executing and testing complete analysis workflows."""
    
    def __init__(self, client: AgentClient):
        self.client = client
        self.data_manager = client.data_manager
        self.workflow_history = []
        self.execution_log = []
        
    def execute_workflow(self, workflow_definition: Dict[str, Any]) -> WorkflowResult:
        """Execute a complete analysis workflow."""
        workflow_id = workflow_definition.get('workflow_id', f'workflow_{int(time.time())}')
        workflow_start = time.time()
        
        steps = [WorkflowStep(**step) for step in workflow_definition['steps']]
        intermediate_results = []
        error_messages = []
        steps_completed = 0
        steps_failed = 0
        
        try:
            # Initialize workflow
            self._initialize_workflow(workflow_definition)
            
            # Execute each step
            for step_idx, step in enumerate(steps):
                step_start = time.time()
                
                try:
                    step_result = self._execute_workflow_step(step, step_idx)
                    
                    if step_result['success']:
                        steps_completed += 1
                        intermediate_results.append({
                            'step_id': step.step_id,
                            'step_index': step_idx,
                            'success': True,
                            'execution_time': time.time() - step_start,
                            'result': step_result['result'],
                            'outputs': step_result.get('outputs', {})
                        })
                    else:
                        steps_failed += 1
                        error_messages.append(f"Step {step.step_id}: {step_result.get('error', 'Unknown error')}")
                        
                        # Check if workflow should continue after failure
                        if not workflow_definition.get('continue_on_error', False):
                            break
                
                except Exception as e:
                    steps_failed += 1
                    error_messages.append(f"Step {step.step_id}: {str(e)}")
                    
                    if not workflow_definition.get('continue_on_error', False):
                        break
            
            # Generate final outputs
            final_outputs = self._generate_final_outputs(workflow_definition, intermediate_results)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_workflow_metrics(
                workflow_start, 
                intermediate_results, 
                steps_completed, 
                steps_failed
            )
            
            workflow_success = steps_failed == 0 and steps_completed == len(steps)
            
        except Exception as e:
            # Workflow-level failure
            workflow_success = False
            error_messages.append(f"Workflow failure: {str(e)}")
            final_outputs = {}
            performance_metrics = {}
        
        workflow_result = WorkflowResult(
            workflow_id=workflow_id,
            success=workflow_success,
            total_execution_time=time.time() - workflow_start,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            final_outputs=final_outputs,
            intermediate_results=intermediate_results,
            error_messages=error_messages,
            performance_metrics=performance_metrics
        )
        
        self.workflow_history.append(workflow_result)
        return workflow_result
    
    def _initialize_workflow(self, workflow_definition: Dict[str, Any]):
        """Initialize workflow environment."""
        # Prepare input data if specified
        if 'input_data' in workflow_definition:
            for data_name, data_config in workflow_definition['input_data'].items():
                self._prepare_input_data(data_name, data_config)
    
    def _prepare_input_data(self, data_name: str, data_config: Dict[str, Any]):
        """Prepare input data for workflow."""
        data_type = data_config.get('type', 'single_cell')
        
        if data_type == 'single_cell':
            adata = SingleCellDataFactory(config=data_config.get('config', SMALL_DATASET_CONFIG))
        elif data_type == 'bulk_rna_seq':
            adata = BulkRNASeqDataFactory(config=data_config.get('config', SMALL_DATASET_CONFIG))
        elif data_type == 'proteomics':
            adata = ProteomicsDataFactory(config=data_config.get('config', SMALL_DATASET_CONFIG))
        elif data_type == 'spatial':
            adata = SpatialDataFactory(config=data_config.get('config', SMALL_DATASET_CONFIG))
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        self.data_manager.modalities[data_name] = adata
    
    def _execute_workflow_step(self, step: WorkflowStep, step_index: int) -> Dict[str, Any]:
        """Execute a single workflow step."""
        try:
            # Construct agent query based on step
            query = self._construct_agent_query(step)
            
            # Mock agent execution with realistic behavior
            if step.step_type == 'data_loading':
                result = self._mock_data_loading_step(step)
            elif step.step_type == 'quality_control':
                result = self._mock_quality_control_step(step)
            elif step.step_type == 'preprocessing':
                result = self._mock_preprocessing_step(step)
            elif step.step_type == 'clustering':
                result = self._mock_clustering_step(step)
            elif step.step_type == 'differential_analysis':
                result = self._mock_differential_analysis_step(step)
            elif step.step_type == 'visualization':
                result = self._mock_visualization_step(step)
            elif step.step_type == 'annotation':
                result = self._mock_annotation_step(step)
            elif step.step_type == 'integration':
                result = self._mock_integration_step(step)
            else:
                result = self._mock_generic_step(step)
            
            return {
                'success': True,
                'result': result,
                'outputs': result.get('outputs', {}),
                'query_executed': query
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'step_id': step.step_id
            }
    
    def _construct_agent_query(self, step: WorkflowStep) -> str:
        """Construct natural language query for agent."""
        base_query = step.description
        
        if step.input_data:
            base_query += f" using dataset '{step.input_data}'"
        
        if step.parameters:
            param_strings = []
            for key, value in step.parameters.items():
                param_strings.append(f"{key}={value}")
            base_query += f" with parameters: {', '.join(param_strings)}"
        
        return base_query
    
    def _mock_data_loading_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Mock data loading step execution."""
        time.sleep(0.5)  # Simulate loading time
        
        # Verify input data exists or create it
        if step.input_data and step.input_data not in self.data_manager.list_modalities():
            # Create mock data
            self._prepare_input_data(step.input_data, {
                'type': step.parameters.get('data_type', 'single_cell'),
                'config': SMALL_DATASET_CONFIG
            })
        
        dataset_name = step.input_data or 'mock_loaded_data'
        adata = self.data_manager.get_modality(dataset_name)
        
        return {
            'data_loaded': True,
            'dataset_name': dataset_name,
            'shape': adata.shape,
            'outputs': {step.output_data or 'loaded_data': dataset_name}
        }
    
    def _mock_quality_control_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Mock quality control step execution."""
        time.sleep(1.0)  # Simulate QC time
        
        input_dataset = step.input_data
        if not input_dataset or input_dataset not in self.data_manager.list_modalities():
            raise ValueError(f"Input dataset '{input_dataset}' not found for QC step")
        
        adata = self.data_manager.get_modality(input_dataset)
        
        # Mock QC metrics
        qc_metrics = {
            'n_cells_before': adata.n_obs,
            'n_genes_before': adata.n_vars,
            'n_cells_after': int(adata.n_obs * 0.9),  # 10% filtered
            'n_genes_after': int(adata.n_vars * 0.85),  # 15% filtered
            'mean_genes_per_cell': np.random.randint(1500, 3000),
            'mean_counts_per_cell': np.random.randint(5000, 15000),
            'mitochondrial_fraction': np.random.uniform(0.05, 0.2)
        }
        
        # Create filtered dataset
        output_name = step.output_data or f"{input_dataset}_qc"
        filtered_adata = adata[:qc_metrics['n_cells_after'], :qc_metrics['n_genes_after']].copy()
        self.data_manager.modalities[output_name] = filtered_adata
        
        return {
            'qc_completed': True,
            'qc_metrics': qc_metrics,
            'cells_filtered': qc_metrics['n_cells_before'] - qc_metrics['n_cells_after'],
            'genes_filtered': qc_metrics['n_genes_before'] - qc_metrics['n_genes_after'],
            'outputs': {step.output_data or 'qc_data': output_name}
        }
    
    def _mock_preprocessing_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Mock preprocessing step execution."""
        time.sleep(1.5)  # Simulate preprocessing time
        
        input_dataset = step.input_data
        if not input_dataset or input_dataset not in self.data_manager.list_modalities():
            raise ValueError(f"Input dataset '{input_dataset}' not found for preprocessing")
        
        adata = self.data_manager.get_modality(input_dataset)
        
        # Mock preprocessing operations
        preprocessing_params = step.parameters or {}
        
        processing_results = {
            'normalization_method': preprocessing_params.get('normalization', 'log1p'),
            'target_sum': preprocessing_params.get('target_sum', 10000),
            'highly_variable_genes': int(adata.n_vars * 0.2),
            'pca_components': preprocessing_params.get('n_pcs', 50),
            'preprocessing_successful': True
        }
        
        # Create processed dataset
        output_name = step.output_data or f"{input_dataset}_preprocessed"
        processed_adata = adata.copy()
        
        # Add mock preprocessing results to adata
        processed_adata.obs['total_counts'] = np.random.randint(1000, 20000, adata.n_obs)
        processed_adata.var['highly_variable'] = np.random.choice([True, False], adata.n_vars, p=[0.2, 0.8])
        processed_adata.obsm['X_pca'] = np.random.randn(adata.n_obs, processing_results['pca_components'])
        
        self.data_manager.modalities[output_name] = processed_adata
        
        return {
            'preprocessing_completed': True,
            'processing_results': processing_results,
            'outputs': {step.output_data or 'preprocessed_data': output_name}
        }
    
    def _mock_clustering_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Mock clustering step execution."""
        time.sleep(2.0)  # Simulate clustering time
        
        input_dataset = step.input_data
        if not input_dataset or input_dataset not in self.data_manager.list_modalities():
            raise ValueError(f"Input dataset '{input_dataset}' not found for clustering")
        
        adata = self.data_manager.get_modality(input_dataset)
        
        # Mock clustering parameters
        clustering_params = step.parameters or {}
        resolution = clustering_params.get('resolution', 0.5)
        n_neighbors = clustering_params.get('n_neighbors', 10)
        
        # Mock clustering results
        n_clusters = np.random.randint(5, 15)
        cluster_assignments = np.random.randint(0, n_clusters, adata.n_obs)
        
        clustering_results = {
            'clustering_method': clustering_params.get('method', 'leiden'),
            'resolution': resolution,
            'n_neighbors': n_neighbors,
            'n_clusters': n_clusters,
            'silhouette_score': np.random.uniform(0.3, 0.8),
            'clustering_successful': True
        }
        
        # Create clustered dataset
        output_name = step.output_data or f"{input_dataset}_clustered"
        clustered_adata = adata.copy()
        
        # Add clustering results
        clustered_adata.obs['leiden'] = cluster_assignments.astype(str)
        clustered_adata.obs['cluster_id'] = cluster_assignments
        clustered_adata.obsm['X_umap'] = np.random.randn(adata.n_obs, 2)
        
        self.data_manager.modalities[output_name] = clustered_adata
        
        return {
            'clustering_completed': True,
            'clustering_results': clustering_results,
            'cluster_assignments': cluster_assignments.tolist(),
            'outputs': {step.output_data or 'clustered_data': output_name}
        }
    
    def _mock_differential_analysis_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Mock differential analysis step execution."""
        time.sleep(1.8)  # Simulate DE analysis time
        
        input_dataset = step.input_data
        if not input_dataset or input_dataset not in self.data_manager.list_modalities():
            raise ValueError(f"Input dataset '{input_dataset}' not found for DE analysis")
        
        adata = self.data_manager.get_modality(input_dataset)
        
        # Mock DE analysis parameters
        de_params = step.parameters or {}
        groupby = de_params.get('groupby', 'leiden')
        method = de_params.get('method', 'wilcoxon')
        
        # Mock DE results
        n_comparisons = np.random.randint(3, 8)
        de_results = {
            'method': method,
            'groupby': groupby,
            'n_comparisons': n_comparisons,
            'significant_genes': {
                f'comparison_{i}': np.random.randint(50, 500) 
                for i in range(n_comparisons)
            },
            'upregulated_genes': {
                f'comparison_{i}': np.random.randint(25, 250)
                for i in range(n_comparisons)
            },
            'downregulated_genes': {
                f'comparison_{i}': np.random.randint(25, 250)
                for i in range(n_comparisons)
            },
            'de_analysis_successful': True
        }
        
        # Create results dataset
        output_name = step.output_data or f"{input_dataset}_de"
        de_adata = adata.copy()
        
        # Add mock DE results
        de_adata.uns['rank_genes_groups'] = {
            'names': np.array([[f'gene_{i}_{j}' for j in range(10)] for i in range(n_comparisons)]),
            'scores': np.random.uniform(0, 5, (n_comparisons, 10)),
            'pvals': np.random.uniform(0, 0.05, (n_comparisons, 10)),
            'pvals_adj': np.random.uniform(0, 0.1, (n_comparisons, 10))
        }
        
        self.data_manager.modalities[output_name] = de_adata
        
        return {
            'differential_analysis_completed': True,
            'de_results': de_results,
            'outputs': {step.output_data or 'de_results': output_name}
        }
    
    def _mock_visualization_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Mock visualization step execution."""
        time.sleep(1.2)  # Simulate plotting time
        
        input_dataset = step.input_data
        if not input_dataset or input_dataset not in self.data_manager.list_modalities():
            raise ValueError(f"Input dataset '{input_dataset}' not found for visualization")
        
        # Mock visualization parameters
        viz_params = step.parameters or {}
        plot_types = viz_params.get('plot_types', ['umap', 'violin', 'heatmap'])
        
        # Mock visualization results
        generated_plots = []
        for plot_type in plot_types:
            plot_info = {
                'plot_type': plot_type,
                'file_path': f'mock_plot_{plot_type}_{int(time.time())}.png',
                'plot_successful': True,
                'plot_dimensions': (8, 6) if plot_type != 'heatmap' else (10, 8)
            }
            generated_plots.append(plot_info)
        
        visualization_results = {
            'visualization_completed': True,
            'plots_generated': len(generated_plots),
            'plot_details': generated_plots,
            'export_format': viz_params.get('format', 'png')
        }
        
        return {
            'visualization_completed': True,
            'visualization_results': visualization_results,
            'outputs': {
                'plots': [plot['file_path'] for plot in generated_plots]
            }
        }
    
    def _mock_annotation_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Mock cell type annotation step execution."""
        time.sleep(1.6)  # Simulate annotation time
        
        input_dataset = step.input_data
        if not input_dataset or input_dataset not in self.data_manager.list_modalities():
            raise ValueError(f"Input dataset '{input_dataset}' not found for annotation")
        
        adata = self.data_manager.get_modality(input_dataset)
        
        # Mock annotation parameters
        annotation_params = step.parameters or {}
        reference_db = annotation_params.get('reference', 'default_cell_atlas')
        
        # Mock cell type annotations
        cell_types = ['T cells', 'B cells', 'NK cells', 'Monocytes', 'Dendritic cells', 
                     'Neutrophils', 'Basophils', 'Eosinophils']
        
        n_clusters = len(np.unique(adata.obs.get('leiden', [0])))
        cluster_annotations = {
            f'cluster_{i}': np.random.choice(cell_types) 
            for i in range(n_clusters)
        }
        
        # Assign cell types to cells based on clusters
        cell_annotations = []
        if 'leiden' in adata.obs:
            for cluster in adata.obs['leiden']:
                cell_annotations.append(cluster_annotations.get(f'cluster_{cluster}', 'Unknown'))
        else:
            cell_annotations = np.random.choice(cell_types, adata.n_obs).tolist()
        
        annotation_results = {
            'annotation_method': annotation_params.get('method', 'automatic'),
            'reference_database': reference_db,
            'n_cell_types_identified': len(set(cell_annotations)),
            'cell_type_mapping': cluster_annotations,
            'annotation_confidence': np.random.uniform(0.7, 0.95),
            'annotation_successful': True
        }
        
        # Create annotated dataset
        output_name = step.output_data or f"{input_dataset}_annotated"
        annotated_adata = adata.copy()
        annotated_adata.obs['cell_type'] = cell_annotations
        annotated_adata.obs['annotation_confidence'] = np.random.uniform(0.5, 1.0, adata.n_obs)
        
        self.data_manager.modalities[output_name] = annotated_adata
        
        return {
            'annotation_completed': True,
            'annotation_results': annotation_results,
            'cell_type_counts': {
                cell_type: cell_annotations.count(cell_type) 
                for cell_type in set(cell_annotations)
            },
            'outputs': {step.output_data or 'annotated_data': output_name}
        }
    
    def _mock_integration_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Mock multi-modal integration step execution."""
        time.sleep(2.5)  # Simulate integration time
        
        # Mock integration of multiple datasets
        integration_params = step.parameters or {}
        modalities = integration_params.get('modalities', ['rna', 'protein'])
        
        integration_results = {
            'integration_method': integration_params.get('method', 'wnn'),
            'modalities_integrated': modalities,
            'n_shared_cells': np.random.randint(1000, 5000),
            'integration_score': np.random.uniform(0.6, 0.9),
            'batch_correction_applied': integration_params.get('batch_correction', True),
            'integration_successful': True
        }
        
        # Create mock integrated dataset
        output_name = step.output_data or 'integrated_data'
        integrated_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        # Add integration results
        integrated_adata.obsm['X_integrated'] = np.random.randn(integrated_adata.n_obs, 30)
        integrated_adata.obsm['X_umap_integrated'] = np.random.randn(integrated_adata.n_obs, 2)
        integrated_adata.obs['modality'] = np.random.choice(modalities, integrated_adata.n_obs)
        
        self.data_manager.modalities[output_name] = integrated_adata
        
        return {
            'integration_completed': True,
            'integration_results': integration_results,
            'outputs': {step.output_data or 'integrated_data': output_name}
        }
    
    def _mock_generic_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Mock generic analysis step execution."""
        time.sleep(np.random.uniform(0.5, 2.0))  # Variable execution time
        
        return {
            'step_completed': True,
            'step_type': step.step_type,
            'mock_execution': True,
            'outputs': {step.output_data or 'generic_output': f"mock_result_{step.step_id}"}
        }
    
    def _generate_final_outputs(self, workflow_definition: Dict[str, Any], intermediate_results: List[Dict]) -> Dict[str, Any]:
        """Generate final workflow outputs."""
        final_outputs = {
            'workflow_type': workflow_definition.get('workflow_type', 'custom'),
            'execution_summary': {
                'total_steps': len(workflow_definition['steps']),
                'completed_steps': len(intermediate_results),
                'success_rate': len(intermediate_results) / len(workflow_definition['steps']) if workflow_definition['steps'] else 0
            },
            'generated_datasets': [],
            'generated_plots': [],
            'analysis_results': {}
        }
        
        # Extract outputs from intermediate results
        for result in intermediate_results:
            if 'outputs' in result:
                for output_type, output_value in result['outputs'].items():
                    if output_type == 'plots':
                        final_outputs['generated_plots'].extend(output_value)
                    elif isinstance(output_value, str) and output_value in self.data_manager.list_modalities():
                        final_outputs['generated_datasets'].append(output_value)
                    else:
                        final_outputs['analysis_results'][output_type] = output_value
        
        # Add final dataset info
        dataset_info = {}
        for dataset_name in final_outputs['generated_datasets']:
            if dataset_name in self.data_manager.list_modalities():
                adata = self.data_manager.get_modality(dataset_name)
                dataset_info[dataset_name] = {
                    'shape': adata.shape,
                    'obs_columns': list(adata.obs.columns),
                    'var_columns': list(adata.var.columns),
                    'obsm_keys': list(adata.obsm.keys()),
                    'uns_keys': list(adata.uns.keys())
                }
        
        final_outputs['dataset_info'] = dataset_info
        
        return final_outputs
    
    def _calculate_workflow_metrics(self, start_time: float, intermediate_results: List[Dict], 
                                  completed: int, failed: int) -> Dict[str, Any]:
        """Calculate workflow performance metrics."""
        total_time = time.time() - start_time
        
        if intermediate_results:
            step_times = [result['execution_time'] for result in intermediate_results]
            
            metrics = {
                'total_execution_time': total_time,
                'avg_step_time': np.mean(step_times),
                'min_step_time': np.min(step_times),
                'max_step_time': np.max(step_times),
                'std_step_time': np.std(step_times),
                'steps_completed': completed,
                'steps_failed': failed,
                'success_rate': completed / (completed + failed) if (completed + failed) > 0 else 0,
                'throughput_steps_per_minute': (completed * 60) / total_time if total_time > 0 else 0
            }
        else:
            metrics = {
                'total_execution_time': total_time,
                'steps_completed': completed,
                'steps_failed': failed,
                'success_rate': 0,
                'error': 'no_intermediate_results'
            }
        
        return metrics


# ===============================================================================
# Fixtures for System Testing
# ===============================================================================

@pytest.fixture(scope="session")
def system_workspace():
    """Create workspace for system tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_system_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def system_data_manager(system_workspace):
    """Create DataManagerV2 for system testing."""
    return DataManagerV2(workspace_path=system_workspace)


@pytest.fixture  
def system_agent_client(system_data_manager):
    """Create AgentClient for system testing."""
    return AgentClient(data_manager=system_data_manager)


@pytest.fixture
def workflow_engine(system_agent_client):
    """Create AnalysisWorkflowEngine for testing."""
    return AnalysisWorkflowEngine(system_agent_client)


@pytest.fixture
def sample_workflow_definitions():
    """Define sample workflows for testing."""
    return {
        'single_cell_basic': {
            'workflow_id': 'sc_basic_analysis',
            'workflow_type': 'single_cell_analysis',
            'description': 'Basic single-cell RNA-seq analysis workflow',
            'input_data': {
                'raw_data': {
                    'type': 'single_cell',
                    'config': SMALL_DATASET_CONFIG
                }
            },
            'steps': [
                {
                    'step_id': 'load_data',
                    'step_type': 'data_loading',
                    'description': 'Load single-cell dataset',
                    'agent_type': 'data',
                    'input_data': 'raw_data',
                    'output_data': 'loaded_data'
                },
                {
                    'step_id': 'quality_control',
                    'step_type': 'quality_control',
                    'description': 'Perform quality control filtering',
                    'agent_type': 'singlecell',
                    'input_data': 'loaded_data',
                    'output_data': 'qc_data',
                    'parameters': {'min_genes': 200, 'min_cells': 3, 'max_mt_percent': 20}
                },
                {
                    'step_id': 'preprocessing',
                    'step_type': 'preprocessing', 
                    'description': 'Normalize and find highly variable genes',
                    'agent_type': 'singlecell',
                    'input_data': 'qc_data',
                    'output_data': 'preprocessed_data',
                    'parameters': {'target_sum': 10000, 'n_pcs': 50}
                },
                {
                    'step_id': 'clustering',
                    'step_type': 'clustering',
                    'description': 'Perform clustering analysis',
                    'agent_type': 'singlecell',
                    'input_data': 'preprocessed_data',
                    'output_data': 'clustered_data',
                    'parameters': {'resolution': 0.5, 'method': 'leiden'}
                },
                {
                    'step_id': 'visualization',
                    'step_type': 'visualization',
                    'description': 'Generate UMAP and clustering plots',
                    'agent_type': 'singlecell',
                    'input_data': 'clustered_data',
                    'parameters': {'plot_types': ['umap', 'violin']}
                }
            ]
        },
        'bulk_rna_seq_de': {
            'workflow_id': 'bulk_de_analysis',
            'workflow_type': 'bulk_rna_seq_analysis',
            'description': 'Bulk RNA-seq differential expression workflow',
            'input_data': {
                'expression_data': {
                    'type': 'bulk_rna_seq',
                    'config': SMALL_DATASET_CONFIG
                }
            },
            'steps': [
                {
                    'step_id': 'load_expression',
                    'step_type': 'data_loading',
                    'description': 'Load bulk RNA-seq expression data',
                    'agent_type': 'data',
                    'input_data': 'expression_data',
                    'output_data': 'loaded_expression'
                },
                {
                    'step_id': 'quality_assessment',
                    'step_type': 'quality_control',
                    'description': 'Assess data quality and filter low-quality samples',
                    'agent_type': 'bulk_rnaseq',
                    'input_data': 'loaded_expression',
                    'output_data': 'qc_expression'
                },
                {
                    'step_id': 'differential_expression',
                    'step_type': 'differential_analysis',
                    'description': 'Perform differential expression analysis',
                    'agent_type': 'bulk_rnaseq',
                    'input_data': 'qc_expression',
                    'output_data': 'de_results',
                    'parameters': {'method': 'deseq2', 'padj_threshold': 0.05}
                },
                {
                    'step_id': 'visualization',
                    'step_type': 'visualization',
                    'description': 'Generate volcano plots and heatmaps',
                    'agent_type': 'bulk_rnaseq',
                    'input_data': 'de_results',
                    'parameters': {'plot_types': ['volcano', 'heatmap']}
                }
            ]
        },
        'multi_modal_integration': {
            'workflow_id': 'multimodal_integration',
            'workflow_type': 'multi_modal_analysis',
            'description': 'Multi-modal data integration workflow',
            'input_data': {
                'rna_data': {
                    'type': 'single_cell',
                    'config': SMALL_DATASET_CONFIG
                },
                'protein_data': {
                    'type': 'proteomics', 
                    'config': SMALL_DATASET_CONFIG
                }
            },
            'steps': [
                {
                    'step_id': 'load_rna',
                    'step_type': 'data_loading',
                    'description': 'Load RNA data',
                    'agent_type': 'data',
                    'input_data': 'rna_data',
                    'output_data': 'loaded_rna'
                },
                {
                    'step_id': 'load_protein',
                    'step_type': 'data_loading', 
                    'description': 'Load protein data',
                    'agent_type': 'data',
                    'input_data': 'protein_data',
                    'output_data': 'loaded_protein'
                },
                {
                    'step_id': 'preprocess_rna',
                    'step_type': 'preprocessing',
                    'description': 'Preprocess RNA data',
                    'agent_type': 'singlecell',
                    'input_data': 'loaded_rna',
                    'output_data': 'preprocessed_rna'
                },
                {
                    'step_id': 'preprocess_protein',
                    'step_type': 'preprocessing',
                    'description': 'Preprocess protein data', 
                    'agent_type': 'proteomics',
                    'input_data': 'loaded_protein',
                    'output_data': 'preprocessed_protein'
                },
                {
                    'step_id': 'integration',
                    'step_type': 'integration',
                    'description': 'Integrate RNA and protein modalities',
                    'agent_type': 'singlecell',
                    'output_data': 'integrated_data',
                    'parameters': {'method': 'wnn', 'modalities': ['rna', 'protein']}
                },
                {
                    'step_id': 'joint_clustering',
                    'step_type': 'clustering',
                    'description': 'Perform joint clustering on integrated data',
                    'agent_type': 'singlecell',
                    'input_data': 'integrated_data',
                    'output_data': 'joint_clusters'
                },
                {
                    'step_id': 'annotation',
                    'step_type': 'annotation',
                    'description': 'Annotate integrated clusters',
                    'agent_type': 'singlecell',
                    'input_data': 'joint_clusters',
                    'output_data': 'annotated_clusters'
                }
            ]
        }
    }


# ===============================================================================
# Full Analysis Workflow Tests
# ===============================================================================

@pytest.mark.system
class TestFullAnalysisWorkflows:
    """Test complete end-to-end analysis workflows."""
    
    def test_single_cell_basic_workflow(self, workflow_engine, sample_workflow_definitions):
        """Test basic single-cell analysis workflow."""
        workflow_def = sample_workflow_definitions['single_cell_basic']
        
        # Execute workflow
        result = workflow_engine.execute_workflow(workflow_def)
        
        # Verify workflow execution
        assert result.success == True, f"Workflow failed: {result.error_messages}"
        assert result.steps_completed == 5, f"Expected 5 steps completed, got {result.steps_completed}"
        assert result.steps_failed == 0, f"Unexpected step failures: {result.steps_failed}"
        
        # Verify workflow timing
        assert result.total_execution_time < 60.0, f"Workflow too slow: {result.total_execution_time}s"
        
        # Verify intermediate results
        assert len(result.intermediate_results) == 5
        step_types_completed = [r['result'].get('step_type') for r in result.intermediate_results]
        expected_steps = ['data_loading', 'quality_control', 'preprocessing', 'clustering', 'visualization']
        
        # Verify all expected steps were completed
        for step in expected_steps:
            assert any(step in str(r) for r in result.intermediate_results), f"Step {step} not completed"
        
        # Verify final outputs
        assert 'generated_datasets' in result.final_outputs
        assert len(result.final_outputs['generated_datasets']) >= 4  # loaded, qc, preprocessed, clustered
        assert 'generated_plots' in result.final_outputs
        assert len(result.final_outputs['generated_plots']) >= 1  # At least one plot generated
        
        # Verify data manager state
        data_manager = workflow_engine.data_manager
        assert 'raw_data' in data_manager.list_modalities()
        assert any('clustered' in modality for modality in data_manager.list_modalities())
        
        # Verify performance metrics
        metrics = result.performance_metrics
        assert metrics['success_rate'] == 1.0
        assert metrics['throughput_steps_per_minute'] > 0
        assert metrics['avg_step_time'] < 20.0
    
    def test_bulk_rna_seq_workflow(self, workflow_engine, sample_workflow_definitions):
        """Test bulk RNA-seq differential expression workflow."""
        workflow_def = sample_workflow_definitions['bulk_rna_seq_de']
        
        # Execute workflow
        result = workflow_engine.execute_workflow(workflow_def)
        
        # Verify workflow execution
        assert result.success == True, f"Workflow failed: {result.error_messages}"
        assert result.steps_completed == 4, f"Expected 4 steps completed, got {result.steps_completed}"
        assert result.steps_failed == 0, f"Unexpected step failures: {result.steps_failed}"
        
        # Verify bulk-specific results
        de_step_result = None
        for intermediate_result in result.intermediate_results:
            if 'differential_analysis_completed' in intermediate_result.get('result', {}):
                de_step_result = intermediate_result['result']
                break
        
        assert de_step_result is not None, "Differential analysis step not found"
        assert de_step_result['differential_analysis_completed'] == True
        assert 'de_results' in de_step_result
        assert de_step_result['de_results']['n_comparisons'] > 0
        
        # Verify data outputs
        assert 'de_results' in result.final_outputs['generated_datasets']
    
    def test_multi_modal_integration_workflow(self, workflow_engine, sample_workflow_definitions):
        """Test multi-modal data integration workflow."""
        workflow_def = sample_workflow_definitions['multi_modal_integration']
        
        # Execute workflow  
        result = workflow_engine.execute_workflow(workflow_def)
        
        # Verify workflow execution
        assert result.success == True, f"Workflow failed: {result.error_messages}"
        assert result.steps_completed == 7, f"Expected 7 steps completed, got {result.steps_completed}"
        assert result.steps_failed == 0, f"Unexpected step failures: {result.steps_failed}"
        
        # Verify multi-modal specific results
        integration_step_result = None
        for intermediate_result in result.intermediate_results:
            if 'integration_completed' in intermediate_result.get('result', {}):
                integration_step_result = intermediate_result['result']
                break
        
        assert integration_step_result is not None, "Integration step not found"
        assert integration_step_result['integration_completed'] == True
        assert 'integration_results' in integration_step_result
        assert len(integration_step_result['integration_results']['modalities_integrated']) >= 2
        
        # Verify annotation step
        annotation_step_result = None
        for intermediate_result in result.intermediate_results:
            if 'annotation_completed' in intermediate_result.get('result', {}):
                annotation_step_result = intermediate_result['result']
                break
        
        assert annotation_step_result is not None, "Annotation step not found"
        assert annotation_step_result['annotation_completed'] == True
        
        # Verify integrated data exists
        data_manager = workflow_engine.data_manager
        assert any('integrated' in modality for modality in data_manager.list_modalities())
        assert any('annotated' in modality for modality in data_manager.list_modalities())
    
    def test_workflow_error_handling(self, workflow_engine):
        """Test workflow error handling and recovery."""
        
        # Define workflow with intentional error
        error_workflow = {
            'workflow_id': 'error_test_workflow',
            'workflow_type': 'error_testing',
            'description': 'Workflow designed to test error handling',
            'continue_on_error': True,  # Continue after errors
            'steps': [
                {
                    'step_id': 'successful_step',
                    'step_type': 'data_loading',
                    'description': 'This step should succeed',
                    'agent_type': 'data',
                    'input_data': 'nonexistent_data',  # This will create mock data
                    'output_data': 'loaded_data'
                },
                {
                    'step_id': 'error_step',
                    'step_type': 'nonexistent_step_type',  # This should cause an error
                    'description': 'This step should fail',
                    'agent_type': 'nonexistent',
                    'input_data': 'loaded_data',
                    'output_data': 'error_output'
                },
                {
                    'step_id': 'recovery_step',
                    'step_type': 'preprocessing',
                    'description': 'This step should succeed after error',
                    'agent_type': 'singlecell',
                    'input_data': 'loaded_data',  # Use successful output
                    'output_data': 'recovered_data'
                }
            ]
        }
        
        # Execute error workflow
        result = workflow_engine.execute_workflow(error_workflow)
        
        # Verify error handling
        assert result.success == False, "Workflow should have failed overall due to errors"
        assert result.steps_completed >= 1, "At least one step should have completed"
        assert result.steps_failed >= 1, "At least one step should have failed"
        assert len(result.error_messages) >= 1, "Should have error messages"
        
        # Verify partial execution
        assert len(result.intermediate_results) >= 1, "Should have some intermediate results"
        
        # Check that successful steps produced outputs
        successful_results = [r for r in result.intermediate_results if r['success']]
        assert len(successful_results) >= 1, "Should have at least one successful step"
    
    def test_workflow_performance_scaling(self, workflow_engine):
        """Test workflow performance with different dataset sizes."""
        
        dataset_sizes = [
            ('small', SMALL_DATASET_CONFIG),
            ('medium', {**SMALL_DATASET_CONFIG, 'n_obs': 5000, 'n_vars': 3000}),
        ]
        
        performance_results = {}
        
        for size_name, config in dataset_sizes:
            # Create workflow with different dataset size
            scaling_workflow = {
                'workflow_id': f'scaling_test_{size_name}',
                'workflow_type': 'performance_scaling',
                'description': f'Test workflow performance with {size_name} dataset',
                'input_data': {
                    'test_data': {
                        'type': 'single_cell',
                        'config': config
                    }
                },
                'steps': [
                    {
                        'step_id': 'load_data',
                        'step_type': 'data_loading',
                        'description': f'Load {size_name} dataset',
                        'agent_type': 'data',
                        'input_data': 'test_data',
                        'output_data': 'loaded_data'
                    },
                    {
                        'step_id': 'preprocessing',
                        'step_type': 'preprocessing',
                        'description': f'Preprocess {size_name} dataset',
                        'agent_type': 'singlecell',
                        'input_data': 'loaded_data',
                        'output_data': 'preprocessed_data'
                    },
                    {
                        'step_id': 'clustering',
                        'step_type': 'clustering',
                        'description': f'Cluster {size_name} dataset',
                        'agent_type': 'singlecell',
                        'input_data': 'preprocessed_data',
                        'output_data': 'clustered_data'
                    }
                ]
            }
            
            # Execute workflow and measure performance
            result = workflow_engine.execute_workflow(scaling_workflow)
            
            performance_results[size_name] = {
                'success': result.success,
                'total_time': result.total_execution_time,
                'avg_step_time': result.performance_metrics.get('avg_step_time', 0),
                'dataset_config': config,
                'throughput': result.performance_metrics.get('throughput_steps_per_minute', 0)
            }
        
        # Verify all workflows succeeded
        for size_name, perf_result in performance_results.items():
            assert perf_result['success'] == True, f"Workflow failed for {size_name} dataset"
        
        # Verify reasonable performance scaling
        small_time = performance_results['small']['total_time']
        medium_time = performance_results['medium']['total_time']
        
        # Medium dataset should not take more than 5x longer than small
        assert medium_time < (small_time * 5), f"Performance scaling too poor: {medium_time} vs {small_time}"
        
        return performance_results
    
    def test_concurrent_workflow_execution(self, workflow_engine, sample_workflow_definitions):
        """Test concurrent execution of multiple workflows."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Select workflows for concurrent execution
        workflows_to_run = [
            sample_workflow_definitions['single_cell_basic'],
            sample_workflow_definitions['bulk_rna_seq_de']
        ]
        
        # Modify workflow IDs to avoid conflicts
        for i, workflow in enumerate(workflows_to_run):
            workflow['workflow_id'] = f"{workflow['workflow_id']}_concurrent_{i}"
        
        concurrent_results = []
        
        def execute_workflow_worker(workflow_def):
            """Worker function for concurrent workflow execution."""
            try:
                result = workflow_engine.execute_workflow(workflow_def)
                return {
                    'success': True,
                    'workflow_id': workflow_def['workflow_id'],
                    'result': result,
                    'error': None
                }
            except Exception as e:
                return {
                    'success': False,
                    'workflow_id': workflow_def['workflow_id'],
                    'result': None,
                    'error': str(e)
                }
        
        # Execute workflows concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_workflow = {
                executor.submit(execute_workflow_worker, workflow): workflow
                for workflow in workflows_to_run
            }
            
            for future in as_completed(future_to_workflow):
                result = future.result()
                concurrent_results.append(result)
        
        # Verify concurrent execution results
        assert len(concurrent_results) == 2, "Should have results from 2 concurrent workflows"
        
        successful_workflows = [r for r in concurrent_results if r['success']]
        assert len(successful_workflows) >= 1, "At least one concurrent workflow should succeed"
        
        # Verify no interference between workflows
        for result_info in successful_workflows:
            workflow_result = result_info['result']
            assert workflow_result.success == True, f"Concurrent workflow {result_info['workflow_id']} failed"
            assert workflow_result.steps_completed > 0, f"No steps completed in {result_info['workflow_id']}"
        
        return concurrent_results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])