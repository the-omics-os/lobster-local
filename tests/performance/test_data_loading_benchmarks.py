"""
Comprehensive performance benchmarks for data loading operations.

This module provides thorough performance testing of data loading operations
including file I/O, format conversions, data validation, caching mechanisms,
and streaming data processing across various biological data formats.

Test coverage target: 95%+ with realistic data loading scenarios.
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
import gzip
import pickle
import json
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from dataclasses import dataclass
from collections import defaultdict

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
from lobster.core.adapters.proteomics_adapter import ProteomicsAdapter
from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.tools.geo_service import GEOService

from tests.mock_data.factories import (
    SingleCellDataFactory, 
    BulkRNASeqDataFactory,
    ProteomicsDataFactory,
    SpatialDataFactory
)
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Data Loading Benchmark Configuration and Utilities
# ===============================================================================

@dataclass
class LoadingBenchmark:
    """Represents a data loading benchmark."""
    benchmark_id: str
    file_format: str
    data_type: str
    file_size_mb: float
    load_time: float
    validation_time: float
    conversion_time: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None


class DataLoadingProfiler:
    """Profiles data loading operations."""
    
    def __init__(self):
        self.benchmarks = []
        self.io_metrics = defaultdict(list)
        self.format_performance = {}
        
    def profile_loading_operation(self, operation_name: str, func, *args, **kwargs):
        """Profile a data loading operation."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)  # MB
        
        try:
            result = func(*args, **kwargs)
            success = True
            error_msg = None
        except Exception as e:
            result = None
            success = False
            error_msg = str(e)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024**2)  # MB
        
        profile_result = {
            'operation': operation_name,
            'execution_time': end_time - start_time,
            'memory_increase_mb': end_memory - start_memory,
            'success': success,
            'error': error_msg,
            'result': result
        }
        
        self.io_metrics[operation_name].append(profile_result)
        return profile_result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'total_operations': sum(len(ops) for ops in self.io_metrics.values()),
            'operation_breakdown': {},
            'overall_stats': {}
        }
        
        all_times = []
        all_memory = []
        success_count = 0
        
        for operation, metrics in self.io_metrics.items():
            successful = [m for m in metrics if m['success']]
            failed = [m for m in metrics if not m['success']]
            
            if successful:
                times = [m['execution_time'] for m in successful]
                memory = [m['memory_increase_mb'] for m in successful]
                
                summary['operation_breakdown'][operation] = {
                    'total_operations': len(metrics),
                    'successful': len(successful),
                    'failed': len(failed),
                    'success_rate': len(successful) / len(metrics),
                    'avg_time': float(np.mean(times)),
                    'min_time': float(np.min(times)),
                    'max_time': float(np.max(times)),
                    'avg_memory_mb': float(np.mean(memory)),
                    'total_memory_mb': float(np.sum(memory))
                }
                
                all_times.extend(times)
                all_memory.extend(memory)
                success_count += len(successful)
        
        if all_times:
            summary['overall_stats'] = {
                'total_successful_operations': success_count,
                'overall_success_rate': success_count / summary['total_operations'],
                'avg_execution_time': float(np.mean(all_times)),
                'total_execution_time': float(np.sum(all_times)),
                'avg_memory_usage_mb': float(np.mean(all_memory)),
                'total_memory_usage_mb': float(np.sum(all_memory)),
                'throughput_ops_per_second': success_count / np.sum(all_times) if np.sum(all_times) > 0 else 0
            }
        
        return summary


def create_test_files(workspace_path: Path, file_configs: List[Dict]) -> List[Path]:
    """Create test files for loading benchmarks."""
    created_files = []
    
    for config in file_configs:
        file_path = workspace_path / config['filename']
        
        if config['format'] == 'h5ad':
            # Create AnnData file
            adata = SingleCellDataFactory(config=config.get('data_config', SMALL_DATASET_CONFIG))
            adata.write_h5ad(file_path)
            
        elif config['format'] == 'csv':
            # Create CSV file
            n_rows = config.get('n_rows', 1000)
            n_cols = config.get('n_cols', 50)
            
            data = pd.DataFrame(
                np.random.randn(n_rows, n_cols),
                columns=[f'feature_{i}' for i in range(n_cols)],
                index=[f'sample_{i}' for i in range(n_rows)]
            )
            data.to_csv(file_path)
            
        elif config['format'] == 'tsv':
            # Create TSV file
            n_rows = config.get('n_rows', 1000)
            n_cols = config.get('n_cols', 50)
            
            data = pd.DataFrame(
                np.random.randn(n_rows, n_cols),
                columns=[f'feature_{i}' for i in range(n_cols)],
                index=[f'sample_{i}' for i in range(n_rows)]
            )
            data.to_csv(file_path, sep='\t')
            
        elif config['format'] == 'mtx':
            # Create Matrix Market file
            from scipy.io import mmwrite
            from scipy.sparse import random
            
            n_rows = config.get('n_rows', 2000)
            n_cols = config.get('n_cols', 1000)
            sparse_matrix = random(n_rows, n_cols, density=0.1, format='coo')
            mmwrite(file_path, sparse_matrix)
            
        elif config['format'] == 'hdf5':
            # Create HDF5 file
            with h5py.File(file_path, 'w') as f:
                n_rows = config.get('n_rows', 1000)
                n_cols = config.get('n_cols', 100)
                
                data = np.random.randn(n_rows, n_cols)
                f.create_dataset('expression_matrix', data=data)
                f.create_dataset('gene_names', data=[f'gene_{i}'.encode() for i in range(n_cols)])
                f.create_dataset('cell_names', data=[f'cell_{i}'.encode() for i in range(n_rows)])
        
        created_files.append(file_path)
    
    return created_files


# ===============================================================================
# Fixtures for Data Loading Benchmarks
# ===============================================================================

@pytest.fixture(scope="session")
def benchmark_workspace():
    """Create workspace for data loading benchmarks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_benchmark_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def benchmark_data_manager(benchmark_workspace):
    """Create DataManagerV2 for benchmarking."""
    return DataManagerV2(workspace_path=benchmark_workspace)


@pytest.fixture
def test_file_configurations():
    """Define test file configurations for benchmarking."""
    return [
        {
            'filename': 'small_single_cell.h5ad',
            'format': 'h5ad',
            'data_config': SMALL_DATASET_CONFIG,
            'expected_size_mb': 5.0
        },
        {
            'filename': 'large_single_cell.h5ad', 
            'format': 'h5ad',
            'data_config': {**LARGE_DATASET_CONFIG, 'n_obs': 20000, 'n_vars': 10000},
            'expected_size_mb': 50.0
        },
        {
            'filename': 'bulk_expression.csv',
            'format': 'csv',
            'n_rows': 20000,
            'n_cols': 500,
            'expected_size_mb': 40.0
        },
        {
            'filename': 'sparse_matrix.mtx',
            'format': 'mtx',
            'n_rows': 30000,
            'n_cols': 15000,
            'expected_size_mb': 15.0
        },
        {
            'filename': 'hdf5_data.h5',
            'format': 'hdf5',
            'n_rows': 10000,
            'n_cols': 2000,
            'expected_size_mb': 80.0
        }
    ]


@pytest.fixture
def benchmark_test_files(benchmark_workspace, test_file_configurations):
    """Create test files for benchmarking."""
    return create_test_files(benchmark_workspace, test_file_configurations)


# ===============================================================================
# File Format Loading Performance Tests
# ===============================================================================

@pytest.mark.performance
class TestFileFormatLoadingPerformance:
    """Test loading performance across different file formats."""
    
    def test_h5ad_loading_performance(self, benchmark_test_files, benchmark_data_manager):
        """Test H5AD file loading performance."""
        
        class H5ADLoadingBenchmark:
            """Benchmarks H5AD file loading operations."""
            
            def __init__(self, data_manager):
                self.data_manager = data_manager
                self.profiler = DataLoadingProfiler()
                
            def benchmark_h5ad_operations(self, h5ad_files: List[Path]):
                """Benchmark various H5AD loading operations."""
                benchmark_results = []
                
                for file_path in h5ad_files:
                    if not file_path.name.endswith('.h5ad'):
                        continue
                    
                    file_size_mb = file_path.stat().st_size / (1024**2)
                    
                    # Test direct loading
                    load_result = self.profiler.profile_loading_operation(
                        'direct_h5ad_load',
                        self._load_h5ad_direct,
                        file_path
                    )
                    
                    # Test loading with scanpy
                    scanpy_result = self.profiler.profile_loading_operation(
                        'scanpy_h5ad_load', 
                        self._load_h5ad_scanpy,
                        file_path
                    )
                    
                    # Test loading with DataManager
                    dm_result = self.profiler.profile_loading_operation(
                        'datamanager_h5ad_load',
                        self._load_h5ad_datamanager,
                        file_path
                    )
                    
                    # Test memory-mapped loading
                    mmap_result = self.profiler.profile_loading_operation(
                        'mmap_h5ad_load',
                        self._load_h5ad_mmap,
                        file_path
                    )
                    
                    benchmark_results.append({
                        'file_path': str(file_path),
                        'file_size_mb': file_size_mb,
                        'direct_load': load_result,
                        'scanpy_load': scanpy_result,
                        'datamanager_load': dm_result,
                        'mmap_load': mmap_result
                    })
                
                return {
                    'h5ad_benchmark_results': benchmark_results,
                    'performance_summary': self.profiler.get_performance_summary()
                }
            
            def _load_h5ad_direct(self, file_path: Path) -> ad.AnnData:
                """Load H5AD file directly."""
                return ad.read_h5ad(file_path)
            
            def _load_h5ad_scanpy(self, file_path: Path) -> ad.AnnData:
                """Load H5AD file using scanpy."""
                adata = sc.read_h5ad(file_path)
                return adata
            
            def _load_h5ad_datamanager(self, file_path: Path) -> str:
                """Load H5AD file using DataManager."""
                modality_name = f"benchmark_{file_path.stem}"
                self.data_manager.load_modality_from_file(modality_name, file_path)
                return modality_name
            
            def _load_h5ad_mmap(self, file_path: Path) -> ad.AnnData:
                """Load H5AD file with memory mapping."""
                return ad.read_h5ad(file_path, backed='r')
        
        # Run H5AD loading benchmarks
        h5ad_benchmark = H5ADLoadingBenchmark(benchmark_data_manager)
        h5ad_results = h5ad_benchmark.benchmark_h5ad_operations(benchmark_test_files)
        
        # Verify benchmark results
        assert len(h5ad_results['h5ad_benchmark_results']) > 0
        
        performance_summary = h5ad_results['performance_summary']
        assert performance_summary['overall_stats']['overall_success_rate'] >= 0.8
        
        # H5AD loading should be reasonably fast
        direct_load_stats = performance_summary['operation_breakdown'].get('direct_h5ad_load', {})
        if direct_load_stats:
            assert direct_load_stats['avg_time'] < 30.0, "H5AD loading too slow"
            assert direct_load_stats['success_rate'] >= 0.9, "H5AD loading failure rate too high"
        
        return h5ad_results
    
    def test_csv_tsv_loading_performance(self, benchmark_test_files, benchmark_data_manager):
        """Test CSV/TSV file loading performance."""
        
        class CSVTSVLoadingBenchmark:
            """Benchmarks CSV/TSV file loading operations."""
            
            def __init__(self, data_manager):
                self.data_manager = data_manager
                self.profiler = DataLoadingProfiler()
                
            def benchmark_csv_tsv_operations(self, files: List[Path]):
                """Benchmark CSV/TSV loading operations."""
                benchmark_results = []
                
                for file_path in files:
                    if not (file_path.name.endswith('.csv') or file_path.name.endswith('.tsv')):
                        continue
                    
                    file_size_mb = file_path.stat().st_size / (1024**2)
                    separator = '\t' if file_path.name.endswith('.tsv') else ','
                    
                    # Test pandas loading
                    pandas_result = self.profiler.profile_loading_operation(
                        'pandas_csv_load',
                        self._load_csv_pandas,
                        file_path, separator
                    )
                    
                    # Test chunked loading
                    chunked_result = self.profiler.profile_loading_operation(
                        'chunked_csv_load',
                        self._load_csv_chunked,
                        file_path, separator
                    )
                    
                    # Test optimized loading
                    optimized_result = self.profiler.profile_loading_operation(
                        'optimized_csv_load',
                        self._load_csv_optimized,
                        file_path, separator
                    )
                    
                    # Test AnnData conversion
                    anndata_result = self.profiler.profile_loading_operation(
                        'csv_to_anndata',
                        self._load_csv_to_anndata,
                        file_path, separator
                    )
                    
                    benchmark_results.append({
                        'file_path': str(file_path),
                        'file_size_mb': file_size_mb,
                        'file_type': 'tsv' if separator == '\t' else 'csv',
                        'pandas_load': pandas_result,
                        'chunked_load': chunked_result,
                        'optimized_load': optimized_result,
                        'anndata_conversion': anndata_result
                    })
                
                return {
                    'csv_tsv_benchmark_results': benchmark_results,
                    'performance_summary': self.profiler.get_performance_summary()
                }
            
            def _load_csv_pandas(self, file_path: Path, separator: str) -> pd.DataFrame:
                """Load CSV/TSV using pandas."""
                return pd.read_csv(file_path, sep=separator, index_col=0)
            
            def _load_csv_chunked(self, file_path: Path, separator: str) -> List[pd.DataFrame]:
                """Load CSV/TSV in chunks."""
                chunks = []
                chunk_size = 1000
                
                for chunk in pd.read_csv(file_path, sep=separator, index_col=0, chunksize=chunk_size):
                    chunks.append(chunk)
                
                return chunks
            
            def _load_csv_optimized(self, file_path: Path, separator: str) -> pd.DataFrame:
                """Load CSV/TSV with optimizations."""
                return pd.read_csv(
                    file_path, 
                    sep=separator, 
                    index_col=0,
                    low_memory=False,
                    engine='c'  # Use C parser for speed
                )
            
            def _load_csv_to_anndata(self, file_path: Path, separator: str) -> ad.AnnData:
                """Load CSV/TSV and convert to AnnData."""
                df = pd.read_csv(file_path, sep=separator, index_col=0)
                
                # Transpose if needed (genes as columns)
                if df.shape[1] > df.shape[0]:
                    df = df.T
                
                adata = ad.AnnData(X=df.values)
                adata.obs.index = df.index
                adata.var.index = df.columns
                
                return adata
        
        # Run CSV/TSV loading benchmarks
        csv_tsv_benchmark = CSVTSVLoadingBenchmark(benchmark_data_manager)
        csv_tsv_results = csv_tsv_benchmark.benchmark_csv_tsv_operations(benchmark_test_files)
        
        # Verify benchmark results
        if csv_tsv_results['csv_tsv_benchmark_results']:
            performance_summary = csv_tsv_results['performance_summary']
            assert performance_summary['overall_stats']['overall_success_rate'] >= 0.8
            
            # CSV loading should be reasonably efficient
            pandas_stats = performance_summary['operation_breakdown'].get('pandas_csv_load', {})
            if pandas_stats:
                assert pandas_stats['avg_time'] < 60.0, "CSV loading too slow"
        
        return csv_tsv_results
    
    def test_sparse_matrix_loading_performance(self, benchmark_test_files):
        """Test sparse matrix format loading performance."""
        
        class SparseMatrixBenchmark:
            """Benchmarks sparse matrix loading operations."""
            
            def __init__(self):
                self.profiler = DataLoadingProfiler()
                
            def benchmark_sparse_operations(self, files: List[Path]):
                """Benchmark sparse matrix loading operations."""
                benchmark_results = []
                
                for file_path in files:
                    if not file_path.name.endswith('.mtx'):
                        continue
                    
                    file_size_mb = file_path.stat().st_size / (1024**2)
                    
                    # Test scipy loading
                    scipy_result = self.profiler.profile_loading_operation(
                        'scipy_mtx_load',
                        self._load_mtx_scipy,
                        file_path
                    )
                    
                    # Test scanpy loading
                    scanpy_result = self.profiler.profile_loading_operation(
                        'scanpy_mtx_load',
                        self._load_mtx_scanpy,
                        file_path
                    )
                    
                    # Test format conversion
                    conversion_result = self.profiler.profile_loading_operation(
                        'mtx_format_conversion',
                        self._convert_mtx_formats,
                        file_path
                    )
                    
                    benchmark_results.append({
                        'file_path': str(file_path),
                        'file_size_mb': file_size_mb,
                        'scipy_load': scipy_result,
                        'scanpy_load': scanpy_result,
                        'format_conversion': conversion_result
                    })
                
                return {
                    'sparse_matrix_results': benchmark_results,
                    'performance_summary': self.profiler.get_performance_summary()
                }
            
            def _load_mtx_scipy(self, file_path: Path):
                """Load MTX file using scipy."""
                from scipy.io import mmread
                return mmread(file_path)
            
            def _load_mtx_scanpy(self, file_path: Path):
                """Load MTX file using scanpy."""
                return sc.read_mtx(file_path)
            
            def _convert_mtx_formats(self, file_path: Path):
                """Convert MTX to different sparse formats."""
                from scipy.io import mmread
                
                matrix = mmread(file_path)
                
                # Convert to different formats
                csr_matrix = matrix.tocsr()
                csc_matrix = matrix.tocsc()
                coo_matrix = matrix.tocoo()
                
                return {
                    'csr_shape': csr_matrix.shape,
                    'csc_shape': csc_matrix.shape,
                    'coo_shape': coo_matrix.shape,
                    'nnz': matrix.nnz
                }
        
        # Run sparse matrix benchmarks
        sparse_benchmark = SparseMatrixBenchmark()
        sparse_results = sparse_benchmark.benchmark_sparse_operations(benchmark_test_files)
        
        # Verify benchmark results
        if sparse_results['sparse_matrix_results']:
            performance_summary = sparse_results['performance_summary']
            assert performance_summary['overall_stats']['overall_success_rate'] >= 0.8
        
        return sparse_results
    
    def test_hdf5_loading_performance(self, benchmark_test_files):
        """Test HDF5 file loading performance."""
        
        class HDF5LoadingBenchmark:
            """Benchmarks HDF5 file loading operations."""
            
            def __init__(self):
                self.profiler = DataLoadingProfiler()
                
            def benchmark_hdf5_operations(self, files: List[Path]):
                """Benchmark HDF5 loading operations."""
                benchmark_results = []
                
                for file_path in files:
                    if not file_path.name.endswith('.h5'):
                        continue
                    
                    file_size_mb = file_path.stat().st_size / (1024**2)
                    
                    # Test direct h5py loading
                    h5py_result = self.profiler.profile_loading_operation(
                        'h5py_load',
                        self._load_hdf5_h5py,
                        file_path
                    )
                    
                    # Test partial loading
                    partial_result = self.profiler.profile_loading_operation(
                        'h5py_partial_load',
                        self._load_hdf5_partial,
                        file_path
                    )
                    
                    # Test compression analysis
                    compression_result = self.profiler.profile_loading_operation(
                        'hdf5_compression_analysis',
                        self._analyze_hdf5_compression,
                        file_path
                    )
                    
                    benchmark_results.append({
                        'file_path': str(file_path),
                        'file_size_mb': file_size_mb,
                        'h5py_load': h5py_result,
                        'partial_load': partial_result,
                        'compression_analysis': compression_result
                    })
                
                return {
                    'hdf5_benchmark_results': benchmark_results,
                    'performance_summary': self.profiler.get_performance_summary()
                }
            
            def _load_hdf5_h5py(self, file_path: Path):
                """Load HDF5 file using h5py."""
                with h5py.File(file_path, 'r') as f:
                    data = {}
                    for key in f.keys():
                        data[key] = f[key][:]
                    return data
            
            def _load_hdf5_partial(self, file_path: Path):
                """Load HDF5 file partially."""
                with h5py.File(file_path, 'r') as f:
                    # Load only first 100 rows of expression matrix
                    if 'expression_matrix' in f:
                        partial_data = f['expression_matrix'][:100, :]
                        return {'partial_expression': partial_data}
                    return {}
            
            def _analyze_hdf5_compression(self, file_path: Path):
                """Analyze HDF5 file compression."""
                with h5py.File(file_path, 'r') as f:
                    compression_info = {}
                    for key in f.keys():
                        dataset = f[key]
                        compression_info[key] = {
                            'compression': dataset.compression,
                            'compression_opts': dataset.compression_opts,
                            'chunks': dataset.chunks,
                            'dtype': str(dataset.dtype),
                            'shape': dataset.shape
                        }
                    return compression_info
        
        # Run HDF5 benchmarks
        hdf5_benchmark = HDF5LoadingBenchmark()
        hdf5_results = hdf5_benchmark.benchmark_hdf5_operations(benchmark_test_files)
        
        # Verify benchmark results
        if hdf5_results['hdf5_benchmark_results']:
            performance_summary = hdf5_results['performance_summary']
            assert performance_summary['overall_stats']['overall_success_rate'] >= 0.8
        
        return hdf5_results


# ===============================================================================
# Data Streaming and Pipeline Performance Tests
# ===============================================================================

@pytest.mark.performance
class TestDataStreamingPerformance:
    """Test data streaming and pipeline performance."""
    
    def test_streaming_data_processing(self, benchmark_workspace, benchmark_data_manager):
        """Test streaming data processing performance."""
        
        class StreamingDataProcessor:
            """Handles streaming data processing."""
            
            def __init__(self, data_manager, workspace_path):
                self.data_manager = data_manager
                self.workspace_path = Path(workspace_path)
                self.profiler = DataLoadingProfiler()
                
            def benchmark_streaming_operations(self, stream_configs: List[Dict]):
                """Benchmark various streaming operations."""
                streaming_results = []
                
                for config in stream_configs:
                    # Test batch processing
                    batch_result = self.profiler.profile_loading_operation(
                        f'batch_processing_{config["name"]}',
                        self._process_data_batches,
                        config
                    )
                    
                    # Test continuous streaming
                    stream_result = self.profiler.profile_loading_operation(
                        f'continuous_streaming_{config["name"]}',
                        self._process_continuous_stream,
                        config
                    )
                    
                    # Test buffered processing
                    buffered_result = self.profiler.profile_loading_operation(
                        f'buffered_processing_{config["name"]}',
                        self._process_buffered_stream,
                        config
                    )
                    
                    streaming_results.append({
                        'config': config,
                        'batch_processing': batch_result,
                        'continuous_streaming': stream_result,
                        'buffered_processing': buffered_result
                    })
                
                return {
                    'streaming_results': streaming_results,
                    'performance_summary': self.profiler.get_performance_summary()
                }
            
            def _process_data_batches(self, config: Dict):
                """Process data in batches."""
                batch_size = config.get('batch_size', 1000)
                total_samples = config.get('total_samples', 10000)
                
                processed_batches = []
                
                for batch_start in range(0, total_samples, batch_size):
                    batch_end = min(batch_start + batch_size, total_samples)
                    batch_size_actual = batch_end - batch_start
                    
                    # Simulate batch data creation
                    batch_data = np.random.randn(batch_size_actual, config.get('n_features', 100))
                    
                    # Simulate processing
                    processed_data = self._simulate_processing(batch_data)
                    
                    processed_batches.append({
                        'batch_id': len(processed_batches),
                        'batch_size': batch_size_actual,
                        'processing_result': processed_data.shape
                    })
                
                return {
                    'total_batches': len(processed_batches),
                    'total_samples_processed': total_samples,
                    'batch_details': processed_batches
                }
            
            def _process_continuous_stream(self, config: Dict):
                """Process continuous data stream."""
                stream_duration = config.get('stream_duration', 5.0)  # seconds
                samples_per_second = config.get('samples_per_second', 1000)
                
                start_time = time.time()
                processed_samples = 0
                
                while (time.time() - start_time) < stream_duration:
                    # Simulate incoming data
                    n_samples = min(samples_per_second // 10, 100)  # Process in small chunks
                    stream_data = np.random.randn(n_samples, config.get('n_features', 100))
                    
                    # Simulate processing
                    _ = self._simulate_processing(stream_data)
                    
                    processed_samples += n_samples
                    time.sleep(0.1)  # Simulate processing time
                
                actual_duration = time.time() - start_time
                
                return {
                    'processed_samples': processed_samples,
                    'actual_duration': actual_duration,
                    'throughput_samples_per_second': processed_samples / actual_duration,
                    'target_throughput': samples_per_second
                }
            
            def _process_buffered_stream(self, config: Dict):
                """Process buffered data stream."""
                buffer_size = config.get('buffer_size', 5000)
                total_samples = config.get('total_samples', 20000)
                
                buffer = []
                processed_buffers = 0
                total_processed = 0
                
                for sample_idx in range(total_samples):
                    # Add sample to buffer
                    sample = np.random.randn(config.get('n_features', 100))
                    buffer.append(sample)
                    
                    # Process buffer when full
                    if len(buffer) >= buffer_size:
                        buffer_array = np.array(buffer)
                        _ = self._simulate_processing(buffer_array)
                        
                        total_processed += len(buffer)
                        processed_buffers += 1
                        buffer = []
                
                # Process remaining buffer
                if buffer:
                    buffer_array = np.array(buffer)
                    _ = self._simulate_processing(buffer_array)
                    total_processed += len(buffer)
                    processed_buffers += 1
                
                return {
                    'processed_buffers': processed_buffers,
                    'total_samples_processed': total_processed,
                    'buffer_size': buffer_size,
                    'avg_buffer_utilization': total_processed / (processed_buffers * buffer_size) if processed_buffers > 0 else 0
                }
            
            def _simulate_processing(self, data: np.ndarray) -> np.ndarray:
                """Simulate data processing operations."""
                # Simulate normalization
                normalized = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
                
                # Simulate filtering
                filtered = normalized[np.sum(np.abs(normalized), axis=1) > 0.1]
                
                return filtered
        
        # Test streaming operations
        processor = StreamingDataProcessor(benchmark_data_manager, benchmark_workspace)
        
        stream_configs = [
            {
                'name': 'small_batches',
                'batch_size': 500,
                'total_samples': 5000,
                'n_features': 100
            },
            {
                'name': 'medium_stream',
                'stream_duration': 3.0,
                'samples_per_second': 2000,
                'n_features': 200
            },
            {
                'name': 'large_buffer',
                'buffer_size': 10000,
                'total_samples': 50000,
                'n_features': 150
            }
        ]
        
        streaming_results = processor.benchmark_streaming_operations(stream_configs)
        
        # Verify streaming performance
        performance_summary = streaming_results['performance_summary']
        assert performance_summary['overall_stats']['overall_success_rate'] >= 0.9
        
        # Check specific streaming metrics
        for result in streaming_results['streaming_results']:
            config_name = result['config']['name']
            
            if 'batch_processing' in result and result['batch_processing']['success']:
                batch_result = result['batch_processing']['result']
                assert batch_result['total_batches'] > 0, f"No batches processed for {config_name}"
            
            if 'continuous_streaming' in result and result['continuous_streaming']['success']:
                stream_result = result['continuous_streaming']['result']
                assert stream_result['throughput_samples_per_second'] > 0, f"No throughput for {config_name}"
                
            if 'buffered_processing' in result and result['buffered_processing']['success']:
                buffer_result = result['buffered_processing']['result']
                assert buffer_result['processed_buffers'] > 0, f"No buffers processed for {config_name}"
        
        return streaming_results
    
    def test_concurrent_file_loading(self, benchmark_test_files, benchmark_data_manager):
        """Test concurrent file loading performance."""
        
        class ConcurrentFileLoader:
            """Handles concurrent file loading operations."""
            
            def __init__(self, data_manager):
                self.data_manager = data_manager
                self.profiler = DataLoadingProfiler()
                
            def benchmark_concurrent_loading(self, files: List[Path], max_workers: int = 4):
                """Benchmark concurrent file loading."""
                
                def load_file_worker(file_path: Path):
                    """Worker function for concurrent file loading."""
                    start_time = time.time()
                    start_memory = psutil.virtual_memory().used / (1024**2)
                    
                    try:
                        # Determine file type and load appropriately
                        if file_path.name.endswith('.h5ad'):
                            result = ad.read_h5ad(file_path)
                        elif file_path.name.endswith('.csv'):
                            result = pd.read_csv(file_path, index_col=0)
                        elif file_path.name.endswith('.tsv'):
                            result = pd.read_csv(file_path, sep='\t', index_col=0)
                        elif file_path.name.endswith('.mtx'):
                            from scipy.io import mmread
                            result = mmread(file_path)
                        elif file_path.name.endswith('.h5'):
                            with h5py.File(file_path, 'r') as f:
                                result = {key: f[key][:] for key in f.keys()}
                        else:
                            result = None
                        
                        success = True
                        error_msg = None
                        
                    except Exception as e:
                        result = None
                        success = False
                        error_msg = str(e)
                    
                    end_time = time.time()
                    end_memory = psutil.virtual_memory().used / (1024**2)
                    
                    return {
                        'file_path': str(file_path),
                        'success': success,
                        'error': error_msg,
                        'load_time': end_time - start_time,
                        'memory_usage_mb': end_memory - start_memory,
                        'file_size_mb': file_path.stat().st_size / (1024**2),
                        'result_type': type(result).__name__ if result is not None else None
                    }
                
                # Execute concurrent loading
                concurrent_start = time.time()
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(load_file_worker, file_path): file_path
                        for file_path in files
                    }
                    
                    results = []
                    for future in as_completed(future_to_file):
                        result = future.result()
                        results.append(result)
                
                concurrent_end = time.time()
                
                # Analyze results
                successful_loads = [r for r in results if r['success']]
                failed_loads = [r for r in results if not r['success']]
                
                total_file_size = sum(r['file_size_mb'] for r in results)
                total_load_time = sum(r['load_time'] for r in successful_loads)
                
                return {
                    'concurrent_loading_successful': len(failed_loads) == 0,
                    'total_files': len(files),
                    'successful_loads': len(successful_loads),
                    'failed_loads': len(failed_loads),
                    'success_rate': len(successful_loads) / len(results) if results else 0,
                    'total_execution_time': concurrent_end - concurrent_start,
                    'total_file_size_mb': total_file_size,
                    'total_load_time': total_load_time,
                    'avg_load_time': total_load_time / len(successful_loads) if successful_loads else 0,
                    'throughput_mb_per_second': total_file_size / (concurrent_end - concurrent_start),
                    'individual_results': results
                }
        
        # Test concurrent file loading
        concurrent_loader = ConcurrentFileLoader(benchmark_data_manager)
        
        # Test with different worker counts
        worker_counts = [2, 4, 6]
        concurrent_results = {}
        
        for max_workers in worker_counts:
            result = concurrent_loader.benchmark_concurrent_loading(
                benchmark_test_files, 
                max_workers=max_workers
            )
            concurrent_results[f'workers_{max_workers}'] = result
        
        # Verify concurrent loading performance
        for worker_count, result in concurrent_results.items():
            assert result['success_rate'] >= 0.8, f"Low success rate for {worker_count}"
            assert result['throughput_mb_per_second'] > 0, f"No throughput for {worker_count}"
            assert result['total_execution_time'] < 300.0, f"Concurrent loading too slow for {worker_count}"
        
        # Compare performance across worker counts
        throughputs = [result['throughput_mb_per_second'] for result in concurrent_results.values()]
        max_throughput = max(throughputs)
        
        # At least one configuration should achieve reasonable throughput
        assert max_throughput > 1.0, "Maximum throughput too low"
        
        return concurrent_results


# ===============================================================================
# Caching and Data Validation Performance Tests
# ===============================================================================

@pytest.mark.performance
class TestCachingAndValidationPerformance:
    """Test caching mechanisms and data validation performance."""
    
    def test_data_caching_performance(self, benchmark_workspace, benchmark_data_manager):
        """Test data caching performance."""
        
        class DataCachingBenchmark:
            """Benchmarks data caching operations."""
            
            def __init__(self, data_manager, workspace_path):
                self.data_manager = data_manager
                self.workspace_path = Path(workspace_path)
                self.cache_dir = self.workspace_path / 'cache'
                self.cache_dir.mkdir(exist_ok=True)
                self.profiler = DataLoadingProfiler()
                
            def benchmark_caching_operations(self, cache_configs: List[Dict]):
                """Benchmark various caching operations."""
                caching_results = []
                
                for config in cache_configs:
                    # Test cache write performance
                    write_result = self.profiler.profile_loading_operation(
                        f'cache_write_{config["name"]}',
                        self._benchmark_cache_write,
                        config
                    )
                    
                    # Test cache read performance
                    read_result = self.profiler.profile_loading_operation(
                        f'cache_read_{config["name"]}',
                        self._benchmark_cache_read,
                        config
                    )
                    
                    # Test cache invalidation
                    invalidation_result = self.profiler.profile_loading_operation(
                        f'cache_invalidation_{config["name"]}',
                        self._benchmark_cache_invalidation,
                        config
                    )
                    
                    caching_results.append({
                        'config': config,
                        'write_performance': write_result,
                        'read_performance': read_result,
                        'invalidation_performance': invalidation_result
                    })
                
                return {
                    'caching_results': caching_results,
                    'performance_summary': self.profiler.get_performance_summary()
                }
            
            def _benchmark_cache_write(self, config: Dict):
                """Benchmark cache write operations."""
                data_size = config.get('data_size', 'medium')
                cache_format = config.get('format', 'pickle')
                
                # Generate test data
                if data_size == 'small':
                    test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                else:
                    test_data = SingleCellDataFactory(config={
                        **SMALL_DATASET_CONFIG,
                        'n_obs': 10000,
                        'n_vars': 5000
                    })
                
                cache_file = self.cache_dir / f"cache_test_{config['name']}.{cache_format}"
                
                # Write to cache
                if cache_format == 'pickle':
                    with open(cache_file, 'wb') as f:
                        pickle.dump(test_data, f)
                elif cache_format == 'h5ad':
                    test_data.write_h5ad(cache_file)
                elif cache_format == 'json':
                    # Convert to serializable format
                    cache_data = {
                        'shape': test_data.shape,
                        'obs_columns': list(test_data.obs.columns),
                        'var_columns': list(test_data.var.columns)
                    }
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f)
                
                return {
                    'cache_file_size_mb': cache_file.stat().st_size / (1024**2),
                    'original_data_shape': test_data.shape,
                    'cache_format': cache_format
                }
            
            def _benchmark_cache_read(self, config: Dict):
                """Benchmark cache read operations."""
                cache_format = config.get('format', 'pickle')
                cache_file = self.cache_dir / f"cache_test_{config['name']}.{cache_format}"
                
                if not cache_file.exists():
                    raise FileNotFoundError(f"Cache file not found: {cache_file}")
                
                # Read from cache
                if cache_format == 'pickle':
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                elif cache_format == 'h5ad':
                    cached_data = ad.read_h5ad(cache_file)
                elif cache_format == 'json':
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                
                return {
                    'cache_hit': True,
                    'cached_data_type': type(cached_data).__name__,
                    'cache_file_size_mb': cache_file.stat().st_size / (1024**2)
                }
            
            def _benchmark_cache_invalidation(self, config: Dict):
                """Benchmark cache invalidation operations."""
                cache_format = config.get('format', 'pickle')
                cache_file = self.cache_dir / f"cache_test_{config['name']}.{cache_format}"
                
                # Create cache hash for validation
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        cache_hash = hashlib.md5(f.read()).hexdigest()
                    
                    # Simulate cache invalidation check
                    time.sleep(0.01)  # Simulate hash computation time
                    
                    # Remove cache file
                    cache_file.unlink()
                    
                    return {
                        'cache_invalidated': True,
                        'cache_hash': cache_hash,
                        'file_removed': not cache_file.exists()
                    }
                else:
                    return {
                        'cache_invalidated': False,
                        'reason': 'cache_file_not_found'
                    }
        
        # Test caching performance
        caching_benchmark = DataCachingBenchmark(benchmark_data_manager, benchmark_workspace)
        
        cache_configs = [
            {'name': 'small_pickle', 'data_size': 'small', 'format': 'pickle'},
            {'name': 'medium_h5ad', 'data_size': 'medium', 'format': 'h5ad'},  
            {'name': 'metadata_json', 'data_size': 'small', 'format': 'json'}
        ]
        
        caching_results = caching_benchmark.benchmark_caching_operations(cache_configs)
        
        # Verify caching performance
        performance_summary = caching_results['performance_summary']
        assert performance_summary['overall_stats']['overall_success_rate'] >= 0.8
        
        # Check cache operation performance
        for result in caching_results['caching_results']:
            config_name = result['config']['name']
            
            # Write performance should be reasonable
            if result['write_performance']['success']:
                assert result['write_performance']['execution_time'] < 30.0, f"Cache write too slow for {config_name}"
            
            # Read performance should be fast
            if result['read_performance']['success']:
                assert result['read_performance']['execution_time'] < 10.0, f"Cache read too slow for {config_name}"
        
        return caching_results
    
    def test_data_validation_performance(self, benchmark_test_files, benchmark_data_manager):
        """Test data validation performance."""
        
        class DataValidationBenchmark:
            """Benchmarks data validation operations."""
            
            def __init__(self, data_manager):
                self.data_manager = data_manager
                self.profiler = DataLoadingProfiler()
                
            def benchmark_validation_operations(self, files: List[Path]):
                """Benchmark various validation operations."""
                validation_results = []
                
                for file_path in files:
                    file_size_mb = file_path.stat().st_size / (1024**2)
                    
                    # Test basic format validation
                    format_result = self.profiler.profile_loading_operation(
                        'format_validation',
                        self._validate_file_format,
                        file_path
                    )
                    
                    # Test schema validation
                    schema_result = self.profiler.profile_loading_operation(
                        'schema_validation',
                        self._validate_data_schema,
                        file_path
                    )
                    
                    # Test integrity validation
                    integrity_result = self.profiler.profile_loading_operation(
                        'integrity_validation',
                        self._validate_data_integrity,
                        file_path
                    )
                    
                    validation_results.append({
                        'file_path': str(file_path),
                        'file_size_mb': file_size_mb,
                        'format_validation': format_result,
                        'schema_validation': schema_result,
                        'integrity_validation': integrity_result
                    })
                
                return {
                    'validation_results': validation_results,
                    'performance_summary': self.profiler.get_performance_summary()
                }
            
            def _validate_file_format(self, file_path: Path):
                """Validate file format."""
                file_extension = file_path.suffix.lower()
                
                format_checks = {
                    '.h5ad': self._check_h5ad_format,
                    '.csv': self._check_csv_format,
                    '.tsv': self._check_tsv_format,
                    '.mtx': self._check_mtx_format,
                    '.h5': self._check_hdf5_format
                }
                
                if file_extension in format_checks:
                    return format_checks[file_extension](file_path)
                else:
                    return {'valid': False, 'reason': 'unsupported_format'}
            
            def _validate_data_schema(self, file_path: Path):
                """Validate data schema."""
                try:
                    if file_path.name.endswith('.h5ad'):
                        adata = ad.read_h5ad(file_path)
                        
                        # Basic schema checks
                        schema_valid = True
                        issues = []
                        
                        if adata.n_obs == 0:
                            schema_valid = False
                            issues.append('no_observations')
                        
                        if adata.n_vars == 0:
                            schema_valid = False
                            issues.append('no_variables')
                        
                        if not hasattr(adata, 'X') or adata.X is None:
                            schema_valid = False
                            issues.append('missing_expression_matrix')
                        
                        return {
                            'valid': schema_valid,
                            'issues': issues,
                            'shape': adata.shape,
                            'obs_columns': len(adata.obs.columns),
                            'var_columns': len(adata.var.columns)
                        }
                    else:
                        return {'valid': True, 'reason': 'schema_validation_skipped'}
                
                except Exception as e:
                    return {'valid': False, 'error': str(e)}
            
            def _validate_data_integrity(self, file_path: Path):
                """Validate data integrity."""
                try:
                    # Calculate file checksum
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    # Check for corruption (basic checks)
                    integrity_checks = {
                        'file_readable': True,
                        'checksum': file_hash,
                        'size_consistent': file_path.stat().st_size > 0
                    }
                    
                    # Format-specific integrity checks
                    if file_path.name.endswith('.h5ad'):
                        try:
                            adata = ad.read_h5ad(file_path)
                            integrity_checks['data_loadable'] = True
                            integrity_checks['matrix_integrity'] = adata.X is not None
                        except Exception:
                            integrity_checks['data_loadable'] = False
                            integrity_checks['matrix_integrity'] = False
                    
                    overall_integrity = all(
                        integrity_checks[key] for key in integrity_checks 
                        if isinstance(integrity_checks[key], bool)
                    )
                    
                    return {
                        'valid': overall_integrity,
                        'checks': integrity_checks
                    }
                
                except Exception as e:
                    return {'valid': False, 'error': str(e)}
            
            def _check_h5ad_format(self, file_path: Path):
                """Check H5AD format validity."""
                try:
                    # Try to read metadata without loading full data
                    with h5py.File(file_path, 'r') as f:
                        required_keys = ['obs', 'var', 'X']
                        present_keys = list(f.keys())
                        
                        format_valid = all(key in present_keys for key in required_keys)
                        
                        return {
                            'valid': format_valid,
                            'present_keys': present_keys,
                            'missing_keys': [key for key in required_keys if key not in present_keys]
                        }
                except Exception as e:
                    return {'valid': False, 'error': str(e)}
            
            def _check_csv_format(self, file_path: Path):
                """Check CSV format validity."""
                try:
                    # Read first few lines to check format
                    with open(file_path, 'r') as f:
                        lines = [f.readline() for _ in range(3)]
                    
                    # Check for consistent delimiter
                    delimiters = [',', '\t', ';']
                    delimiter_counts = {d: lines[0].count(d) for d in delimiters}
                    primary_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                    
                    return {
                        'valid': delimiter_counts[primary_delimiter] > 0,
                        'delimiter': primary_delimiter,
                        'delimiter_count': delimiter_counts[primary_delimiter],
                        'first_line_length': len(lines[0]) if lines else 0
                    }
                except Exception as e:
                    return {'valid': False, 'error': str(e)}
            
            def _check_tsv_format(self, file_path: Path):
                """Check TSV format validity."""
                return self._check_csv_format(file_path)  # Similar logic
            
            def _check_mtx_format(self, file_path: Path):
                """Check MTX format validity."""
                try:
                    with open(file_path, 'r') as f:
                        first_line = f.readline().strip()
                        
                        # Check for Matrix Market header
                        mm_header = first_line.startswith('%%MatrixMarket')
                        
                        return {
                            'valid': mm_header,
                            'header': first_line,
                            'has_mm_header': mm_header
                        }
                except Exception as e:
                    return {'valid': False, 'error': str(e)}
            
            def _check_hdf5_format(self, file_path: Path):
                """Check HDF5 format validity."""
                try:
                    with h5py.File(file_path, 'r') as f:
                        return {
                            'valid': True,
                            'keys': list(f.keys()),
                            'n_datasets': len(f.keys())
                        }
                except Exception as e:
                    return {'valid': False, 'error': str(e)}
        
        # Test validation performance
        validation_benchmark = DataValidationBenchmark(benchmark_data_manager)
        validation_results = validation_benchmark.benchmark_validation_operations(benchmark_test_files)
        
        # Verify validation performance
        performance_summary = validation_results['performance_summary']
        assert performance_summary['overall_stats']['overall_success_rate'] >= 0.8
        
        # Validation should be fast
        for operation, stats in performance_summary['operation_breakdown'].items():
            assert stats['avg_time'] < 10.0, f"Validation operation {operation} too slow"
        
        return validation_results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])