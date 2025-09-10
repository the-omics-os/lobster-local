"""
Comprehensive performance tests for large dataset processing.

This module provides thorough performance testing of large-scale biological
dataset processing including memory efficiency, processing speed, resource
utilization, and scalability metrics across different dataset sizes.

Test coverage target: 95%+ with realistic large dataset scenarios.
"""

import pytest
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
from unittest.mock import Mock, MagicMock, patch
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import gc
import resource
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from memory_profiler import profile
import functools

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.agents.singlecell_expert import SingleCellExpert
from lobster.agents.bulk_rnaseq_expert import BulkRNASeqExpert
from lobster.tools.preprocessing_service import PreprocessingService
from lobster.tools.clustering_service import ClusteringService
from lobster.tools.quality_service import QualityService

from tests.mock_data.factories import (
    SingleCellDataFactory, 
    BulkRNASeqDataFactory,
    SpatialDataFactory
)
from tests.mock_data.base import LARGE_DATASET_CONFIG


# ===============================================================================
# Performance Test Configuration and Utilities
# ===============================================================================

class PerformanceMetrics:
    """Collects and analyzes performance metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.cpu_percent = []
        self.memory_samples = []
        self.monitoring_active = False
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / (1024**3)  # GB
        self.peak_memory = self.start_memory
        self.cpu_percent = []
        self.memory_samples = []
        self.monitoring_active = True
        
        # Start background monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring and return metrics."""
        self.monitoring_active = False
        self.end_time = time.time()
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        execution_time = self.end_time - self.start_time
        current_memory = psutil.virtual_memory().used / (1024**3)
        memory_increase = current_memory - self.start_memory
        
        return {
            'execution_time': execution_time,
            'start_memory_gb': self.start_memory,
            'peak_memory_gb': self.peak_memory,
            'memory_increase_gb': memory_increase,
            'avg_cpu_percent': np.mean(self.cpu_percent) if self.cpu_percent else 0,
            'max_cpu_percent': np.max(self.cpu_percent) if self.cpu_percent else 0,
            'memory_samples': len(self.memory_samples),
            'memory_efficiency': (memory_increase / execution_time) if execution_time > 0 else 0
        }
    
    def _monitor_resources(self):
        """Background resource monitoring."""
        while self.monitoring_active:
            try:
                # CPU monitoring
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_percent.append(cpu_percent)
                
                # Memory monitoring
                current_memory = psutil.virtual_memory().used / (1024**3)
                self.memory_samples.append(current_memory)
                self.peak_memory = max(self.peak_memory, current_memory)
                
                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                break


def memory_benchmark(func):
    """Decorator to benchmark memory usage of functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        try:
            result = func(*args, **kwargs)
            performance_data = metrics.stop_monitoring()
            
            # Add performance metrics to result if it's a dict
            if isinstance(result, dict):
                result['performance_metrics'] = performance_data
            
            return result
        except Exception as e:
            metrics.stop_monitoring()
            raise e
    
    return wrapper


def create_large_dataset(n_obs: int, n_vars: int, sparse_ratio: float = 0.8) -> ad.AnnData:
    """Create large synthetic biological dataset for performance testing."""
    # Create sparse count matrix for realistic memory usage
    from scipy.sparse import random as sparse_random
    
    # Generate sparse matrix with realistic biological distributions
    X_sparse = sparse_random(n_obs, n_vars, density=1-sparse_ratio, format='csr')
    
    # Scale to realistic count ranges
    X_sparse.data = np.random.negative_binomial(20, 0.3, size=len(X_sparse.data))
    
    # Create observations metadata
    obs_data = {
        'cell_id': [f'CELL_{i:07d}' for i in range(n_obs)],
        'batch': np.random.choice(['batch_1', 'batch_2', 'batch_3', 'batch_4'], n_obs),
        'condition': np.random.choice(['control', 'treatment_a', 'treatment_b'], n_obs),
        'cell_type': np.random.choice(['T_cells', 'B_cells', 'NK_cells', 'Monocytes', 'Dendritic_cells'], n_obs),
        'total_counts': np.random.randint(1000, 50000, n_obs),
        'n_genes_by_counts': np.random.randint(500, 8000, n_obs)
    }
    obs_df = pd.DataFrame(obs_data)
    obs_df.index = obs_data['cell_id']
    
    # Create variables metadata
    var_data = {
        'gene_id': [f'ENSG{i:011d}' for i in range(n_vars)],
        'gene_symbol': [f'GENE_{i}' for i in range(n_vars)],
        'feature_type': 'Gene Expression',
        'highly_variable': np.random.choice([True, False], n_vars, p=[0.15, 0.85])
    }
    var_df = pd.DataFrame(var_data)
    var_df.index = var_data['gene_id']
    
    # Create AnnData object
    adata = ad.AnnData(X=X_sparse, obs=obs_df, var=var_df)
    
    return adata


# ===============================================================================
# Fixtures for Performance Testing
# ===============================================================================

@pytest.fixture(scope="session")
def temp_performance_workspace():
    """Create temporary workspace for performance tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_performance_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture(scope="session")
def performance_data_manager(temp_performance_workspace):
    """Create DataManagerV2 instance for performance testing."""
    return DataManagerV2(workspace_path=temp_performance_workspace)


@pytest.fixture(params=[
    (10000, 5000),    # Medium dataset
    (50000, 20000),   # Large dataset
    (100000, 30000)   # Extra large dataset
])
def large_dataset_sizes(request):
    """Parameterized fixture for different dataset sizes."""
    return request.param


@pytest.fixture
def large_single_cell_data():
    """Create large single-cell dataset for performance testing."""
    return create_large_dataset(n_obs=50000, n_vars=20000, sparse_ratio=0.85)


@pytest.fixture
def extra_large_single_cell_data():
    """Create extra large single-cell dataset for stress testing."""
    return create_large_dataset(n_obs=100000, n_vars=30000, sparse_ratio=0.9)


@pytest.fixture
def performance_services():
    """Create analysis services for performance testing."""
    return {
        'preprocessing_service': PreprocessingService(),
        'clustering_service': ClusteringService(),
        'quality_service': QualityService()
    }


# ===============================================================================
# Large Dataset Loading and Memory Management Tests
# ===============================================================================

@pytest.mark.performance
class TestLargeDatasetLoading:
    """Test loading and memory management of large datasets."""
    
    @memory_benchmark
    def test_large_h5ad_loading_performance(self, temp_performance_workspace, large_single_cell_data):
        """Test performance of loading large H5AD files."""
        
        class LargeDatasetLoader:
            """Handles loading of large biological datasets."""
            
            def __init__(self, workspace_path):
                self.workspace_path = Path(workspace_path)
                self.loading_metrics = {}
                
            def save_and_load_h5ad(self, adata, test_name):
                """Save and reload H5AD file to test I/O performance."""
                file_path = self.workspace_path / f"{test_name}.h5ad"
                
                # Measure save performance
                save_start = time.time()
                adata.write_h5ad(file_path)
                save_time = time.time() - save_start
                
                # Get file size
                file_size_mb = file_path.stat().st_size / (1024**2)
                
                # Force garbage collection
                del adata
                gc.collect()
                
                # Measure load performance
                load_start = time.time()
                loaded_adata = ad.read_h5ad(file_path)
                load_time = time.time() - load_start
                
                return {
                    'save_time': save_time,
                    'load_time': load_time,
                    'file_size_mb': file_size_mb,
                    'load_speed_mbps': file_size_mb / load_time if load_time > 0 else 0,
                    'save_speed_mbps': file_size_mb / save_time if save_time > 0 else 0,
                    'loaded_adata': loaded_adata
                }
            
            def test_chunked_loading(self, file_path, chunk_size=5000):
                """Test chunked loading for memory efficiency."""
                chunked_results = []
                
                # Load full dataset first to get dimensions
                full_adata = ad.read_h5ad(file_path)
                n_obs_total = full_adata.n_obs
                
                del full_adata
                gc.collect()
                
                # Load in chunks
                chunk_start = time.time()
                for start_idx in range(0, n_obs_total, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_obs_total)
                    
                    # Read specific slice
                    chunk_adata = ad.read_h5ad(file_path)[start_idx:end_idx, :]
                    
                    # Process chunk (mock analysis)
                    chunk_stats = {
                        'chunk_start': start_idx,
                        'chunk_end': end_idx,
                        'chunk_size': chunk_adata.n_obs,
                        'memory_usage_mb': chunk_adata.X.nbytes / (1024**2)
                    }
                    chunked_results.append(chunk_stats)
                    
                    # Clean up chunk
                    del chunk_adata
                    gc.collect()
                
                chunk_total_time = time.time() - chunk_start
                
                return {
                    'chunked_loading_successful': True,
                    'total_chunks': len(chunked_results),
                    'chunk_processing_time': chunk_total_time,
                    'avg_time_per_chunk': chunk_total_time / len(chunked_results),
                    'chunk_details': chunked_results
                }
        
        # Test large dataset loading
        loader = LargeDatasetLoader(temp_performance_workspace)
        
        # Save and load test
        io_results = loader.save_and_load_h5ad(large_single_cell_data, "large_dataset_performance")
        
        # Verify performance benchmarks
        assert io_results['save_time'] < 60.0, f"Save time too slow: {io_results['save_time']}s"
        assert io_results['load_time'] < 30.0, f"Load time too slow: {io_results['load_time']}s"
        assert io_results['file_size_mb'] > 0, "File was not saved properly"
        assert io_results['load_speed_mbps'] > 1.0, "Load speed too slow"
        
        # Test chunked loading
        file_path = temp_performance_workspace / "large_dataset_performance.h5ad"
        chunked_results = loader.test_chunked_loading(file_path, chunk_size=5000)
        
        assert chunked_results['chunked_loading_successful'] == True
        assert chunked_results['total_chunks'] > 1
        assert chunked_results['avg_time_per_chunk'] < 5.0, "Chunk processing too slow"
        
        return {
            'io_performance': io_results,
            'chunked_performance': chunked_results
        }
    
    def test_memory_efficient_dataset_creation(self, large_dataset_sizes, performance_data_manager):
        """Test memory-efficient creation of datasets of varying sizes."""
        
        class MemoryEfficientCreator:
            """Creates datasets with memory optimization."""
            
            def __init__(self, data_manager):
                self.data_manager = data_manager
                
            def create_dataset_with_monitoring(self, n_obs, n_vars, modality_name):
                """Create dataset while monitoring memory usage."""
                creation_metrics = PerformanceMetrics()
                creation_metrics.start_monitoring()
                
                # Create dataset in chunks to manage memory
                chunk_size = min(10000, n_obs // 4)
                chunks = []
                
                for start_idx in range(0, n_obs, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_obs)
                    chunk_n_obs = end_idx - start_idx
                    
                    # Create chunk
                    chunk_adata = create_large_dataset(chunk_n_obs, n_vars, sparse_ratio=0.85)
                    
                    # Adjust cell IDs to maintain uniqueness
                    chunk_adata.obs.index = [f'CELL_{i+start_idx:07d}' for i in range(chunk_n_obs)]
                    chunks.append(chunk_adata)
                
                # Concatenate chunks
                if len(chunks) > 1:
                    combined_adata = ad.concat(chunks, axis=0, join='outer')
                else:
                    combined_adata = chunks[0]
                
                # Clean up chunk references
                del chunks
                gc.collect()
                
                # Store in data manager
                self.data_manager.modalities[modality_name] = combined_adata
                
                performance_data = creation_metrics.stop_monitoring()
                
                return {
                    'creation_successful': True,
                    'final_n_obs': combined_adata.n_obs,
                    'final_n_vars': combined_adata.n_vars,
                    'performance_metrics': performance_data,
                    'memory_efficiency_score': self._calculate_memory_efficiency(
                        combined_adata, performance_data
                    )
                }
            
            def _calculate_memory_efficiency(self, adata, performance_data):
                """Calculate memory efficiency score."""
                # Theoretical minimum memory (sparse matrix storage)
                theoretical_memory_gb = (adata.X.nnz * 8) / (1024**3)  # 8 bytes per non-zero
                actual_memory_increase = performance_data['memory_increase_gb']
                
                if actual_memory_increase > 0:
                    efficiency = theoretical_memory_gb / actual_memory_increase
                    return min(efficiency, 1.0)  # Cap at 1.0 for perfect efficiency
                return 0.0
        
        # Test dataset creation with different sizes
        n_obs, n_vars = large_dataset_sizes
        modality_name = f"performance_test_{n_obs}x{n_vars}"
        
        creator = MemoryEfficientCreator(performance_data_manager)
        creation_result = creator.create_dataset_with_monitoring(n_obs, n_vars, modality_name)
        
        # Verify creation success and performance
        assert creation_result['creation_successful'] == True
        assert creation_result['final_n_obs'] == n_obs
        assert creation_result['final_n_vars'] == n_vars
        
        # Performance benchmarks
        metrics = creation_result['performance_metrics']
        assert metrics['execution_time'] < 300.0, f"Creation too slow: {metrics['execution_time']}s"
        assert metrics['memory_increase_gb'] < 10.0, f"Memory usage too high: {metrics['memory_increase_gb']}GB"
        
        # Memory efficiency should be reasonable
        assert creation_result['memory_efficiency_score'] > 0.1, "Memory efficiency too low"
        
        # Verify dataset is accessible in data manager
        assert modality_name in performance_data_manager.list_modalities()
        retrieved_adata = performance_data_manager.get_modality(modality_name)
        assert retrieved_adata.n_obs == n_obs
        assert retrieved_adata.n_vars == n_vars
    
    def test_concurrent_dataset_loading(self, temp_performance_workspace, performance_data_manager):
        """Test concurrent loading of multiple datasets."""
        
        class ConcurrentLoader:
            """Handles concurrent dataset loading operations."""
            
            def __init__(self, workspace_path, data_manager):
                self.workspace_path = Path(workspace_path)
                self.data_manager = data_manager
                self.loading_results = {}
                
            def prepare_multiple_datasets(self, dataset_configs):
                """Prepare multiple datasets for concurrent loading test."""
                prepared_files = []
                
                for config in dataset_configs:
                    dataset_name = config['name']
                    n_obs = config['n_obs']
                    n_vars = config['n_vars']
                    
                    # Create dataset
                    adata = create_large_dataset(n_obs, n_vars, sparse_ratio=0.8)
                    
                    # Save to file
                    file_path = self.workspace_path / f"{dataset_name}.h5ad"
                    adata.write_h5ad(file_path)
                    
                    prepared_files.append({
                        'name': dataset_name,
                        'file_path': str(file_path),
                        'expected_shape': (n_obs, n_vars)
                    })
                    
                    del adata
                    gc.collect()
                
                return prepared_files
            
            def load_dataset_worker(self, file_info):
                """Worker function for concurrent dataset loading."""
                worker_start = time.time()
                worker_memory_start = psutil.virtual_memory().used / (1024**3)
                
                try:
                    # Load dataset
                    adata = ad.read_h5ad(file_info['file_path'])
                    
                    # Store in data manager
                    self.data_manager.modalities[file_info['name']] = adata
                    
                    worker_end = time.time()
                    worker_memory_end = psutil.virtual_memory().used / (1024**3)
                    
                    return {
                        'name': file_info['name'],
                        'success': True,
                        'loading_time': worker_end - worker_start,
                        'memory_increase': worker_memory_end - worker_memory_start,
                        'dataset_shape': adata.shape,
                        'expected_shape': file_info['expected_shape']
                    }
                    
                except Exception as e:
                    return {
                        'name': file_info['name'],
                        'success': False,
                        'error': str(e)
                    }
            
            def test_concurrent_loading(self, prepared_files, max_workers=3):
                """Test loading multiple datasets concurrently."""
                concurrent_start = time.time()
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(self.load_dataset_worker, file_info): file_info
                        for file_info in prepared_files
                    }
                    
                    results = []
                    for future in future_to_file:
                        result = future.result()
                        results.append(result)
                
                concurrent_end = time.time()
                
                return {
                    'concurrent_loading_successful': all(r['success'] for r in results),
                    'total_loading_time': concurrent_end - concurrent_start,
                    'individual_results': results,
                    'datasets_loaded': len([r for r in results if r['success']]),
                    'average_loading_time': np.mean([r['loading_time'] for r in results if r['success']]),
                    'total_memory_increase': sum([r.get('memory_increase', 0) for r in results if r['success']])
                }
        
        # Test concurrent loading
        loader = ConcurrentLoader(temp_performance_workspace, performance_data_manager)
        
        # Prepare test datasets
        dataset_configs = [
            {'name': 'concurrent_dataset_1', 'n_obs': 20000, 'n_vars': 5000},
            {'name': 'concurrent_dataset_2', 'n_obs': 15000, 'n_vars': 8000},
            {'name': 'concurrent_dataset_3', 'n_obs': 25000, 'n_vars': 6000}
        ]
        
        prepared_files = loader.prepare_multiple_datasets(dataset_configs)
        
        # Test concurrent loading
        concurrent_results = loader.test_concurrent_loading(prepared_files, max_workers=3)
        
        # Verify concurrent loading success
        assert concurrent_results['concurrent_loading_successful'] == True
        assert concurrent_results['datasets_loaded'] == 3
        assert concurrent_results['total_loading_time'] < 120.0, "Concurrent loading too slow"
        assert concurrent_results['average_loading_time'] < 60.0, "Average loading time too slow"
        
        # Verify all datasets are accessible
        for config in dataset_configs:
            assert config['name'] in performance_data_manager.list_modalities()
            adata = performance_data_manager.get_modality(config['name'])
            assert adata.n_obs == config['n_obs']
            assert adata.n_vars == config['n_vars']


# ===============================================================================
# Large-Scale Processing Performance Tests
# ===============================================================================

@pytest.mark.performance
class TestLargeScaleProcessing:
    """Test performance of large-scale data processing operations."""
    
    @memory_benchmark
    def test_large_scale_preprocessing_performance(self, large_single_cell_data, performance_services):
        """Test preprocessing performance on large datasets."""
        
        class LargeScalePreprocessor:
            """Handles large-scale preprocessing with performance optimization."""
            
            def __init__(self, preprocessing_service):
                self.preprocessing_service = preprocessing_service
                
            def preprocess_with_monitoring(self, adata, processing_params):
                """Preprocess data with comprehensive performance monitoring."""
                preprocessing_steps = []
                
                # Step 1: Quality Control Filtering
                qc_start = time.time()
                qc_memory_start = psutil.virtual_memory().used / (1024**3)
                
                # Calculate QC metrics
                adata.var['mt'] = adata.var.index.str.startswith('MT-')
                adata.obs['pct_counts_mt'] = (
                    np.array(adata[:, adata.var['mt']].X.sum(axis=1)).flatten() / 
                    np.array(adata.X.sum(axis=1)).flatten()
                ) * 100
                
                # Filter cells and genes
                n_obs_before = adata.n_obs
                n_vars_before = adata.n_vars
                
                # Filter cells
                sc.pp.filter_cells(adata, min_genes=processing_params.get('min_genes', 200))
                adata = adata[adata.obs['pct_counts_mt'] < processing_params.get('max_mt_percent', 20), :]
                
                # Filter genes  
                sc.pp.filter_genes(adata, min_cells=processing_params.get('min_cells', 3))
                
                qc_end = time.time()
                qc_memory_end = psutil.virtual_memory().used / (1024**3)
                
                preprocessing_steps.append({
                    'step': 'quality_control_filtering',
                    'execution_time': qc_end - qc_start,
                    'memory_increase': qc_memory_end - qc_memory_start,
                    'cells_before': n_obs_before,
                    'genes_before': n_vars_before,
                    'cells_after': adata.n_obs,
                    'genes_after': adata.n_vars,
                    'cells_filtered': n_obs_before - adata.n_obs,
                    'genes_filtered': n_vars_before - adata.n_vars
                })
                
                # Step 2: Normalization
                norm_start = time.time()
                norm_memory_start = psutil.virtual_memory().used / (1024**3)
                
                # Store raw counts
                adata.raw = adata
                
                # Normalize to 10,000 reads per cell
                sc.pp.normalize_total(adata, target_sum=processing_params.get('target_sum', 1e4))
                
                # Log transform
                sc.pp.log1p(adata)
                
                norm_end = time.time()
                norm_memory_end = psutil.virtual_memory().used / (1024**3)
                
                preprocessing_steps.append({
                    'step': 'normalization_log_transform',
                    'execution_time': norm_end - norm_start,
                    'memory_increase': norm_memory_end - norm_memory_start,
                    'target_sum': processing_params.get('target_sum', 1e4)
                })
                
                # Step 3: Highly Variable Genes
                hvg_start = time.time()
                hvg_memory_start = psutil.virtual_memory().used / (1024**3)
                
                # Find highly variable genes
                sc.pp.highly_variable_genes(
                    adata, 
                    n_top_genes=processing_params.get('n_top_genes', 2000),
                    batch_key=processing_params.get('batch_key')
                )
                
                n_hvg = sum(adata.var['highly_variable'])
                
                # Keep only HVGs for downstream analysis
                adata_hvg = adata[:, adata.var['highly_variable']].copy()
                
                hvg_end = time.time()
                hvg_memory_end = psutil.virtual_memory().used / (1024**3)
                
                preprocessing_steps.append({
                    'step': 'highly_variable_genes',
                    'execution_time': hvg_end - hvg_start,
                    'memory_increase': hvg_memory_end - hvg_memory_start,
                    'n_hvg_found': n_hvg,
                    'hvg_percentage': (n_hvg / adata.n_vars) * 100
                })
                
                # Step 4: Principal Component Analysis  
                pca_start = time.time()
                pca_memory_start = psutil.virtual_memory().used / (1024**3)
                
                # Scale data
                sc.pp.scale(adata_hvg, max_value=10)
                
                # PCA
                sc.tl.pca(adata_hvg, n_comps=processing_params.get('n_pcs', 50))
                
                pca_end = time.time()
                pca_memory_end = psutil.virtual_memory().used / (1024**3)
                
                preprocessing_steps.append({
                    'step': 'pca_computation',
                    'execution_time': pca_end - pca_start,
                    'memory_increase': pca_memory_end - pca_memory_start,
                    'n_components': processing_params.get('n_pcs', 50),
                    'explained_variance': float(adata_hvg.uns['pca']['variance_ratio'][:10].sum())
                })
                
                return {
                    'preprocessing_successful': True,
                    'processed_adata': adata_hvg,
                    'preprocessing_steps': preprocessing_steps,
                    'total_execution_time': sum(step['execution_time'] for step in preprocessing_steps),
                    'total_memory_increase': sum(step['memory_increase'] for step in preprocessing_steps),
                    'final_dataset_shape': adata_hvg.shape
                }
        
        # Test large-scale preprocessing
        preprocessor = LargeScalePreprocessor(performance_services['preprocessing_service'])
        
        processing_params = {
            'min_genes': 200,
            'min_cells': 3,
            'max_mt_percent': 20,
            'target_sum': 1e4,
            'n_top_genes': 2000,
            'n_pcs': 50
        }
        
        preprocessing_result = preprocessor.preprocess_with_monitoring(
            large_single_cell_data.copy(), 
            processing_params
        )
        
        # Verify preprocessing success and performance
        assert preprocessing_result['preprocessing_successful'] == True
        assert len(preprocessing_result['preprocessing_steps']) == 4
        assert preprocessing_result['total_execution_time'] < 300.0, "Preprocessing too slow"
        assert preprocessing_result['total_memory_increase'] < 15.0, "Memory usage too high"
        
        # Verify processing quality
        processed_adata = preprocessing_result['processed_adata']
        assert processed_adata.n_obs > 0, "No cells remaining after filtering"
        assert processed_adata.n_vars > 0, "No genes remaining after filtering"
        assert 'X_pca' in processed_adata.obsm, "PCA not computed"
        assert 'highly_variable' in processed_adata.var, "HVG not identified"
        
        return preprocessing_result
    
    def test_large_scale_clustering_performance(self, large_single_cell_data, performance_services):
        """Test clustering performance on large datasets."""
        
        class LargeScaleClusterer:
            """Handles large-scale clustering with performance optimization."""
            
            def __init__(self, clustering_service):
                self.clustering_service = clustering_service
                
            def cluster_with_monitoring(self, adata, clustering_params):
                """Perform clustering with performance monitoring."""
                clustering_metrics = PerformanceMetrics()
                clustering_metrics.start_monitoring()
                
                clustering_steps = []
                
                # Prepare data (basic preprocessing if needed)
                if 'X_pca' not in adata.obsm:
                    # Quick preprocessing for clustering
                    prep_start = time.time()
                    
                    # Normalize and log-transform
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    
                    # Find HVGs and compute PCA
                    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
                    adata_hvg = adata[:, adata.var['highly_variable']].copy()
                    sc.pp.scale(adata_hvg, max_value=10)
                    sc.tl.pca(adata_hvg, n_comps=50)
                    
                    adata = adata_hvg
                    
                    prep_end = time.time()
                    clustering_steps.append({
                        'step': 'preprocessing_for_clustering',
                        'execution_time': prep_end - prep_start
                    })
                
                # Step 1: Neighborhood Graph Construction
                neighbors_start = time.time()
                
                sc.pp.neighbors(
                    adata,
                    n_neighbors=clustering_params.get('n_neighbors', 10),
                    n_pcs=clustering_params.get('n_pcs', 40)
                )
                
                neighbors_end = time.time()
                clustering_steps.append({
                    'step': 'neighbors_computation',
                    'execution_time': neighbors_end - neighbors_start,
                    'n_neighbors': clustering_params.get('n_neighbors', 10)
                })
                
                # Step 2: Leiden Clustering
                leiden_start = time.time()
                
                resolutions = clustering_params.get('resolutions', [0.3, 0.5, 0.7, 1.0])
                clustering_results = {}
                
                for resolution in resolutions:
                    sc.tl.leiden(adata, resolution=resolution, key_added=f'leiden_{resolution}')
                    
                    n_clusters = len(adata.obs[f'leiden_{resolution}'].unique())
                    clustering_results[f'leiden_{resolution}'] = {
                        'resolution': resolution,
                        'n_clusters': n_clusters,
                        'largest_cluster_size': adata.obs[f'leiden_{resolution}'].value_counts().max(),
                        'smallest_cluster_size': adata.obs[f'leiden_{resolution}'].value_counts().min()
                    }
                
                leiden_end = time.time()
                clustering_steps.append({
                    'step': 'leiden_clustering',
                    'execution_time': leiden_end - leiden_start,
                    'resolutions_tested': len(resolutions),
                    'clustering_results': clustering_results
                })
                
                # Step 3: UMAP Embedding
                umap_start = time.time()
                
                sc.tl.umap(adata, n_components=2)
                
                umap_end = time.time()
                clustering_steps.append({
                    'step': 'umap_embedding',
                    'execution_time': umap_end - umap_start,
                    'embedding_dimensions': adata.obsm['X_umap'].shape[1]
                })
                
                # Calculate clustering quality metrics
                quality_metrics = self._assess_clustering_quality(adata, clustering_results)
                
                performance_data = clustering_metrics.stop_monitoring()
                
                return {
                    'clustering_successful': True,
                    'clustered_adata': adata,
                    'clustering_steps': clustering_steps,
                    'clustering_results': clustering_results,
                    'quality_metrics': quality_metrics,
                    'performance_metrics': performance_data,
                    'total_execution_time': sum(step['execution_time'] for step in clustering_steps)
                }
            
            def _assess_clustering_quality(self, adata, clustering_results):
                """Assess quality of clustering results."""
                quality_metrics = {}
                
                for clustering_key in clustering_results:
                    if clustering_key in adata.obs:
                        clusters = adata.obs[clustering_key]
                        
                        # Basic quality metrics
                        n_clusters = len(clusters.unique())
                        cluster_sizes = clusters.value_counts()
                        
                        quality_metrics[clustering_key] = {
                            'n_clusters': n_clusters,
                            'mean_cluster_size': float(cluster_sizes.mean()),
                            'cluster_size_cv': float(cluster_sizes.std() / cluster_sizes.mean()),
                            'singleton_clusters': int(sum(cluster_sizes == 1)),
                            'largest_cluster_fraction': float(cluster_sizes.max() / len(clusters))
                        }
                
                return quality_metrics
        
        # Test large-scale clustering
        clusterer = LargeScaleClusterer(performance_services['clustering_service'])
        
        clustering_params = {
            'n_neighbors': 15,
            'n_pcs': 40,
            'resolutions': [0.3, 0.5, 0.8, 1.2]
        }
        
        clustering_result = clusterer.cluster_with_monitoring(
            large_single_cell_data.copy(),
            clustering_params
        )
        
        # Verify clustering success and performance
        assert clustering_result['clustering_successful'] == True
        assert len(clustering_result['clustering_steps']) >= 3
        assert clustering_result['total_execution_time'] < 600.0, "Clustering too slow"
        
        # Verify clustering quality
        clustered_adata = clustering_result['clustered_adata']
        assert 'X_umap' in clustered_adata.obsm, "UMAP not computed"
        
        # Check clustering results
        for resolution in clustering_params['resolutions']:
            clustering_key = f'leiden_{resolution}'
            assert clustering_key in clustered_adata.obs, f"Clustering {clustering_key} not found"
            
            quality = clustering_result['quality_metrics'][clustering_key]
            assert quality['n_clusters'] > 1, f"Too few clusters for {clustering_key}"
            assert quality['n_clusters'] < clustered_adata.n_obs // 10, f"Too many clusters for {clustering_key}"
            assert quality['largest_cluster_fraction'] < 0.8, f"Dominant cluster too large for {clustering_key}"
        
        return clustering_result
    
    def test_memory_usage_scaling(self, extra_large_single_cell_data, temp_performance_workspace):
        """Test how memory usage scales with dataset size."""
        
        class MemoryScalingAnalyzer:
            """Analyzes memory usage scaling patterns."""
            
            def __init__(self, workspace_path):
                self.workspace_path = Path(workspace_path)
                
            def test_scaling_with_subsampling(self, full_adata, subsample_fractions):
                """Test memory scaling by subsampling large dataset."""
                scaling_results = []
                
                for fraction in subsample_fractions:
                    n_cells_subsample = int(full_adata.n_obs * fraction)
                    
                    # Create subsample
                    subsample_metrics = PerformanceMetrics()
                    subsample_metrics.start_monitoring()
                    
                    # Random subsample
                    subsample_indices = np.random.choice(
                        full_adata.n_obs, 
                        size=n_cells_subsample, 
                        replace=False
                    )
                    subsample_adata = full_adata[subsample_indices, :].copy()
                    
                    # Perform basic analysis on subsample
                    analysis_start = time.time()
                    
                    # Basic preprocessing
                    sc.pp.filter_genes(subsample_adata, min_cells=3)
                    sc.pp.normalize_total(subsample_adata, target_sum=1e4)
                    sc.pp.log1p(subsample_adata)
                    sc.pp.highly_variable_genes(subsample_adata, n_top_genes=min(2000, subsample_adata.n_vars))
                    
                    analysis_end = time.time()
                    
                    performance_data = subsample_metrics.stop_monitoring()
                    
                    # Calculate theoretical vs actual memory usage
                    theoretical_memory_mb = (subsample_adata.X.nbytes) / (1024**2)
                    actual_memory_mb = performance_data['memory_increase_gb'] * 1024
                    
                    scaling_results.append({
                        'subsample_fraction': fraction,
                        'n_cells': n_cells_subsample,
                        'n_genes': subsample_adata.n_vars,
                        'execution_time': performance_data['execution_time'],
                        'analysis_time': analysis_end - analysis_start,
                        'theoretical_memory_mb': theoretical_memory_mb,
                        'actual_memory_mb': actual_memory_mb,
                        'memory_overhead_ratio': actual_memory_mb / theoretical_memory_mb if theoretical_memory_mb > 0 else 0,
                        'peak_memory_gb': performance_data['peak_memory_gb']
                    })
                    
                    # Clean up
                    del subsample_adata
                    gc.collect()
                
                # Analyze scaling patterns
                scaling_analysis = self._analyze_scaling_patterns(scaling_results)
                
                return {
                    'scaling_results': scaling_results,
                    'scaling_analysis': scaling_analysis
                }
            
            def _analyze_scaling_patterns(self, results):
                """Analyze memory and time scaling patterns."""
                if len(results) < 2:
                    return {'error': 'Insufficient data points for scaling analysis'}
                
                # Extract data for analysis
                n_cells = [r['n_cells'] for r in results]
                execution_times = [r['execution_time'] for r in results]
                memory_usage = [r['actual_memory_mb'] for r in results]
                
                # Fit scaling relationships (simple linear approximation)
                time_scaling = np.polyfit(n_cells, execution_times, 1)
                memory_scaling = np.polyfit(n_cells, memory_usage, 1)
                
                return {
                    'time_scaling_slope': float(time_scaling[0]),
                    'time_scaling_intercept': float(time_scaling[1]),
                    'memory_scaling_slope': float(memory_scaling[0]),
                    'memory_scaling_intercept': float(memory_scaling[1]),
                    'max_memory_overhead': max([r['memory_overhead_ratio'] for r in results]),
                    'min_memory_overhead': min([r['memory_overhead_ratio'] for r in results]),
                    'avg_memory_overhead': np.mean([r['memory_overhead_ratio'] for r in results])
                }
        
        # Test memory scaling
        analyzer = MemoryScalingAnalyzer(temp_performance_workspace)
        
        subsample_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        scaling_result = analyzer.test_scaling_with_subsampling(
            extra_large_single_cell_data, 
            subsample_fractions
        )
        
        # Verify scaling analysis
        assert len(scaling_result['scaling_results']) == len(subsample_fractions)
        
        scaling_analysis = scaling_result['scaling_analysis']
        assert 'time_scaling_slope' in scaling_analysis
        assert 'memory_scaling_slope' in scaling_analysis
        
        # Memory overhead should be reasonable
        assert scaling_analysis['max_memory_overhead'] < 10.0, "Memory overhead too high"
        assert scaling_analysis['avg_memory_overhead'] > 0.5, "Memory overhead too low (suspicious)"
        
        # Time scaling should be reasonable (sub-quadratic)
        largest_dataset = max(scaling_result['scaling_results'], key=lambda x: x['n_cells'])
        assert largest_dataset['execution_time'] < 600.0, "Largest dataset processing too slow"
        
        return scaling_result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])