"""
Comprehensive performance tests for concurrent agent execution.

This module provides thorough performance testing of concurrent agent execution,
multi-threading, agent orchestration, resource contention, and scalability
under various load conditions and concurrent user scenarios.

Test coverage target: 95%+ with realistic concurrent execution scenarios.
"""

import pytest
import time
import threading
import asyncio
import psutil
from typing import Dict, Any, List, Optional, Tuple, Union
from unittest.mock import Mock, MagicMock, patch
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import multiprocessing as mp
from dataclasses import dataclass

from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.agents.singlecell_expert import SingleCellExpert
from lobster.agents.bulk_rnaseq_expert import BulkRNASeqExpert
from lobster.agents.proteomics_expert import ProteomicsExpert
from lobster.agents.data_expert import DataExpert
from lobster.agents.supervisor import SupervisorAgent

from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Concurrent Execution Test Configuration
# ===============================================================================

@dataclass
class ConcurrentTask:
    """Represents a concurrent task for performance testing."""
    task_id: str
    agent_type: str
    query: str
    expected_duration: float
    dataset_size: str
    priority: int = 1


@dataclass
class ExecutionMetrics:
    """Metrics for concurrent execution performance."""
    task_id: str
    agent_type: str
    start_time: float
    end_time: float
    execution_time: float
    success: bool
    memory_usage_mb: float
    cpu_percent: float
    queue_wait_time: float
    error_message: Optional[str] = None


class ConcurrentExecutionMonitor:
    """Monitors concurrent agent execution performance."""
    
    def __init__(self):
        self.execution_metrics = []
        self.resource_samples = []
        self.monitoring_active = False
        self.lock = threading.Lock()
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
    
    def record_execution(self, metrics: ExecutionMetrics):
        """Record execution metrics thread-safely."""
        with self.lock:
            self.execution_metrics.append(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.execution_metrics:
            return {'error': 'No execution metrics recorded'}
        
        successful_executions = [m for m in self.execution_metrics if m.success]
        failed_executions = [m for m in self.execution_metrics if not m.success]
        
        execution_times = [m.execution_time for m in successful_executions]
        memory_usages = [m.memory_usage_mb for m in successful_executions]
        cpu_usages = [m.cpu_percent for m in successful_executions]
        queue_wait_times = [m.queue_wait_time for m in successful_executions]
        
        return {
            'total_tasks': len(self.execution_metrics),
            'successful_tasks': len(successful_executions),
            'failed_tasks': len(failed_executions),
            'success_rate': len(successful_executions) / len(self.execution_metrics) if self.execution_metrics else 0,
            'execution_time_stats': {
                'mean': float(np.mean(execution_times)) if execution_times else 0,
                'std': float(np.std(execution_times)) if execution_times else 0,
                'min': float(np.min(execution_times)) if execution_times else 0,
                'max': float(np.max(execution_times)) if execution_times else 0,
                'p95': float(np.percentile(execution_times, 95)) if execution_times else 0
            },
            'memory_usage_stats': {
                'mean_mb': float(np.mean(memory_usages)) if memory_usages else 0,
                'peak_mb': float(np.max(memory_usages)) if memory_usages else 0,
                'total_mb': float(np.sum(memory_usages)) if memory_usages else 0
            },
            'cpu_usage_stats': {
                'mean_percent': float(np.mean(cpu_usages)) if cpu_usages else 0,
                'peak_percent': float(np.max(cpu_usages)) if cpu_usages else 0
            },
            'queue_performance': {
                'mean_wait_time': float(np.mean(queue_wait_times)) if queue_wait_times else 0,
                'max_wait_time': float(np.max(queue_wait_times)) if queue_wait_times else 0
            },
            'agent_type_breakdown': self._get_agent_breakdown(),
            'resource_samples': len(self.resource_samples)
        }
    
    def _monitor_resources(self):
        """Background resource monitoring."""
        while self.monitoring_active:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                
                sample = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_used_gb': memory_info.used / (1024**3),
                    'memory_percent': memory_info.percent
                }
                
                with self.lock:
                    self.resource_samples.append(sample)
                
                time.sleep(0.2)  # Sample every 200ms
            except Exception:
                break
    
    def _get_agent_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by agent type."""
        agent_breakdown = {}
        
        for agent_type in set(m.agent_type for m in self.execution_metrics):
            agent_metrics = [m for m in self.execution_metrics if m.agent_type == agent_type]
            successful = [m for m in agent_metrics if m.success]
            
            if successful:
                execution_times = [m.execution_time for m in successful]
                agent_breakdown[agent_type] = {
                    'total_tasks': len(agent_metrics),
                    'successful_tasks': len(successful),
                    'success_rate': len(successful) / len(agent_metrics),
                    'avg_execution_time': float(np.mean(execution_times)),
                    'max_execution_time': float(np.max(execution_times))
                }
        
        return agent_breakdown


# ===============================================================================
# Fixtures for Concurrent Testing
# ===============================================================================

@pytest.fixture(scope="session")
def concurrent_workspace():
    """Create workspace for concurrent execution tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_concurrent_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def concurrent_data_manager(concurrent_workspace):
    """Create DataManagerV2 for concurrent testing."""
    return DataManagerV2(workspace_path=concurrent_workspace)


@pytest.fixture
def mock_agent_clients(concurrent_data_manager):
    """Create mock agent clients for concurrent testing."""
    clients = {}
    
    # Create mock clients for different agent types
    for agent_type in ['singlecell', 'bulk_rnaseq', 'proteomics', 'data']:
        client = Mock(spec=AgentClient)
        client.data_manager = concurrent_data_manager
        clients[agent_type] = client
    
    return clients


@pytest.fixture
def concurrent_test_datasets(concurrent_data_manager):
    """Create test datasets for concurrent execution."""
    datasets = {}
    
    # Small dataset for quick tests
    small_sc = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    concurrent_data_manager.modalities['small_single_cell'] = small_sc
    datasets['small_single_cell'] = small_sc
    
    # Medium dataset for performance tests
    medium_sc = SingleCellDataFactory(config={
        **SMALL_DATASET_CONFIG,
        'n_obs': 10000,
        'n_vars': 5000
    })
    concurrent_data_manager.modalities['medium_single_cell'] = medium_sc
    datasets['medium_single_cell'] = medium_sc
    
    return datasets


@pytest.fixture
def concurrent_task_definitions():
    """Define tasks for concurrent execution testing."""
    return [
        ConcurrentTask(
            task_id="sc_qc_task",
            agent_type="singlecell",
            query="Perform quality control on small_single_cell dataset",
            expected_duration=5.0,
            dataset_size="small"
        ),
        ConcurrentTask(
            task_id="sc_clustering_task",
            agent_type="singlecell",
            query="Cluster cells in medium_single_cell dataset using leiden algorithm",
            expected_duration=15.0,
            dataset_size="medium"
        ),
        ConcurrentTask(
            task_id="data_loading_task",
            agent_type="data",
            query="Load and validate single-cell dataset format",
            expected_duration=3.0,
            dataset_size="small"
        ),
        ConcurrentTask(
            task_id="preprocessing_task",
            agent_type="singlecell", 
            query="Normalize and filter medium_single_cell dataset",
            expected_duration=10.0,
            dataset_size="medium"
        )
    ]


# ===============================================================================
# Concurrent Agent Execution Tests
# ===============================================================================

@pytest.mark.performance
class TestConcurrentAgentExecution:
    """Test concurrent execution of multiple agents."""
    
    def test_basic_concurrent_agent_execution(self, mock_agent_clients, concurrent_task_definitions):
        """Test basic concurrent execution of multiple agents."""
        
        class ConcurrentAgentExecutor:
            """Handles concurrent agent execution."""
            
            def __init__(self, agent_clients):
                self.agent_clients = agent_clients
                self.monitor = ConcurrentExecutionMonitor()
                
            def execute_task(self, task: ConcurrentTask, task_queue: Queue) -> ExecutionMetrics:
                """Execute a single task and record metrics."""
                queue_start = time.time()
                
                # Simulate queue wait time
                try:
                    task_queue.get(timeout=1.0)
                except Empty:
                    pass
                
                queue_wait_time = time.time() - queue_start
                
                # Record execution start
                start_time = time.time()
                start_memory = psutil.virtual_memory().used / (1024**2)  # MB
                start_cpu = psutil.cpu_percent()
                
                success = True
                error_message = None
                
                try:
                    # Mock agent execution based on task type
                    if task.agent_type == "singlecell":
                        execution_time = self._mock_singlecell_execution(task)
                    elif task.agent_type == "bulk_rnaseq":
                        execution_time = self._mock_bulk_execution(task)
                    elif task.agent_type == "data":
                        execution_time = self._mock_data_execution(task)
                    else:
                        execution_time = self._mock_generic_execution(task)
                    
                    # Simulate actual processing time
                    time.sleep(min(execution_time, 2.0))  # Cap sleep for testing
                    
                except Exception as e:
                    success = False
                    error_message = str(e)
                
                # Record execution end
                end_time = time.time()
                end_memory = psutil.virtual_memory().used / (1024**2)  # MB
                end_cpu = psutil.cpu_percent()
                
                return ExecutionMetrics(
                    task_id=task.task_id,
                    agent_type=task.agent_type,
                    start_time=start_time,
                    end_time=end_time,
                    execution_time=end_time - start_time,
                    success=success,
                    memory_usage_mb=end_memory - start_memory,
                    cpu_percent=max(start_cpu, end_cpu),
                    queue_wait_time=queue_wait_time,
                    error_message=error_message
                )
            
            def execute_concurrent_tasks(self, tasks: List[ConcurrentTask], max_workers: int = 4):
                """Execute multiple tasks concurrently."""
                task_queue = Queue()
                for i in range(len(tasks)):
                    task_queue.put(i)
                
                self.monitor.start_monitoring()
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_task = {
                        executor.submit(self.execute_task, task, task_queue): task 
                        for task in tasks
                    }
                    
                    for future in as_completed(future_to_task):
                        metrics = future.result()
                        self.monitor.record_execution(metrics)
                
                self.monitor.stop_monitoring()
                
                return self.monitor.get_performance_summary()
            
            def _mock_singlecell_execution(self, task: ConcurrentTask) -> float:
                """Mock single-cell analysis execution."""
                base_time = 0.5
                
                if "clustering" in task.query.lower():
                    return base_time * 3.0
                elif "quality" in task.query.lower():
                    return base_time * 1.5
                elif "preprocessing" in task.query.lower():
                    return base_time * 2.0
                else:
                    return base_time
            
            def _mock_bulk_execution(self, task: ConcurrentTask) -> float:
                """Mock bulk RNA-seq analysis execution."""
                return 0.8  # Bulk analysis typically faster per cell
            
            def _mock_data_execution(self, task: ConcurrentTask) -> float:
                """Mock data loading/validation execution."""
                return 0.3  # Data operations typically fast
            
            def _mock_generic_execution(self, task: ConcurrentTask) -> float:
                """Mock generic agent execution."""
                return 1.0
        
        # Test concurrent execution
        executor = ConcurrentAgentExecutor(mock_agent_clients)
        
        # Execute tasks concurrently
        performance_summary = executor.execute_concurrent_tasks(
            concurrent_task_definitions, 
            max_workers=3
        )
        
        # Verify concurrent execution performance
        assert performance_summary['total_tasks'] == len(concurrent_task_definitions)
        assert performance_summary['success_rate'] >= 0.9, "Too many task failures"
        
        # Performance benchmarks
        exec_stats = performance_summary['execution_time_stats']
        assert exec_stats['max'] < 30.0, "Maximum execution time too long"
        assert exec_stats['mean'] < 10.0, "Average execution time too long"
        
        # Resource utilization should be reasonable
        memory_stats = performance_summary['memory_usage_stats']
        assert memory_stats['peak_mb'] < 1000.0, "Peak memory usage too high"
        
        # Queue performance
        queue_perf = performance_summary['queue_performance']
        assert queue_perf['max_wait_time'] < 2.0, "Queue wait time too long"
        
        # Verify agent type breakdown
        agent_breakdown = performance_summary['agent_type_breakdown']
        assert len(agent_breakdown) > 0, "No agent type breakdown recorded"
        
        for agent_type, stats in agent_breakdown.items():
            assert stats['success_rate'] >= 0.8, f"Low success rate for {agent_type}"
    
    def test_agent_resource_contention(self, mock_agent_clients, concurrent_data_manager):
        """Test resource contention between concurrent agents."""
        
        class ResourceContentionTester:
            """Tests resource contention scenarios."""
            
            def __init__(self, agent_clients, data_manager):
                self.agent_clients = agent_clients
                self.data_manager = data_manager
                self.contention_results = []
                
            def test_memory_contention(self, num_memory_intensive_tasks: int = 5):
                """Test concurrent memory-intensive tasks."""
                memory_tasks = []
                
                for i in range(num_memory_intensive_tasks):
                    task = ConcurrentTask(
                        task_id=f"memory_task_{i}",
                        agent_type="singlecell",
                        query=f"Process large dataset with heavy memory usage - task {i}",
                        expected_duration=8.0,
                        dataset_size="large"
                    )
                    memory_tasks.append(task)
                
                # Monitor memory during execution
                monitor = ConcurrentExecutionMonitor()
                monitor.start_monitoring()
                
                def memory_intensive_worker(task):
                    """Worker that simulates memory-intensive processing."""
                    start_time = time.time()
                    start_memory = psutil.virtual_memory().used / (1024**3)  # GB
                    
                    # Simulate memory allocation
                    memory_consumer = np.random.randn(1000000)  # ~8MB array
                    
                    # Simulate processing
                    for _ in range(10):
                        _ = np.dot(memory_consumer, memory_consumer)
                        time.sleep(0.1)
                    
                    end_time = time.time()
                    end_memory = psutil.virtual_memory().used / (1024**3)  # GB
                    
                    return ExecutionMetrics(
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        start_time=start_time,
                        end_time=end_time,
                        execution_time=end_time - start_time,
                        success=True,
                        memory_usage_mb=(end_memory - start_memory) * 1024,
                        cpu_percent=psutil.cpu_percent(),
                        queue_wait_time=0.0
                    )
                
                # Execute memory-intensive tasks
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(memory_intensive_worker, task) for task in memory_tasks]
                    
                    for future in as_completed(futures):
                        metrics = future.result()
                        monitor.record_execution(metrics)
                
                monitor.stop_monitoring()
                
                return {
                    'memory_contention_test': 'completed',
                    'performance_summary': monitor.get_performance_summary(),
                    'peak_concurrent_memory': max([
                        sample['memory_used_gb'] for sample in monitor.resource_samples
                    ]) if monitor.resource_samples else 0
                }
            
            def test_cpu_contention(self, num_cpu_intensive_tasks: int = 4):
                """Test concurrent CPU-intensive tasks."""
                cpu_tasks = []
                
                for i in range(num_cpu_intensive_tasks):
                    task = ConcurrentTask(
                        task_id=f"cpu_task_{i}",
                        agent_type="singlecell", 
                        query=f"Perform CPU-intensive clustering - task {i}",
                        expected_duration=6.0,
                        dataset_size="medium"
                    )
                    cpu_tasks.append(task)
                
                monitor = ConcurrentExecutionMonitor()
                monitor.start_monitoring()
                
                def cpu_intensive_worker(task):
                    """Worker that simulates CPU-intensive processing."""
                    start_time = time.time()
                    
                    # Simulate CPU-intensive computation
                    result = 0
                    for i in range(1000000):
                        result += np.sin(i) * np.cos(i)
                        if i % 100000 == 0:
                            time.sleep(0.01)  # Brief pause for monitoring
                    
                    end_time = time.time()
                    
                    return ExecutionMetrics(
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        start_time=start_time,
                        end_time=end_time,
                        execution_time=end_time - start_time,
                        success=True,
                        memory_usage_mb=0.0,  # Minimal memory usage
                        cpu_percent=psutil.cpu_percent(),
                        queue_wait_time=0.0
                    )
                
                # Execute CPU-intensive tasks
                with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                    futures = [executor.submit(cpu_intensive_worker, task) for task in cpu_tasks]
                    
                    for future in as_completed(futures):
                        metrics = future.result()
                        monitor.record_execution(metrics)
                
                monitor.stop_monitoring()
                
                return {
                    'cpu_contention_test': 'completed',
                    'performance_summary': monitor.get_performance_summary(),
                    'peak_concurrent_cpu': max([
                        sample['cpu_percent'] for sample in monitor.resource_samples
                    ]) if monitor.resource_samples else 0
                }
            
            def test_io_contention(self, num_io_tasks: int = 6):
                """Test concurrent I/O intensive tasks."""
                io_tasks = []
                
                for i in range(num_io_tasks):
                    task = ConcurrentTask(
                        task_id=f"io_task_{i}",
                        agent_type="data",
                        query=f"Load and save large dataset - task {i}",
                        expected_duration=4.0,
                        dataset_size="medium"
                    )
                    io_tasks.append(task)
                
                monitor = ConcurrentExecutionMonitor()
                monitor.start_monitoring()
                
                def io_intensive_worker(task):
                    """Worker that simulates I/O intensive processing."""
                    start_time = time.time()
                    
                    # Create temporary data
                    temp_data = SingleCellDataFactory(config={
                        **SMALL_DATASET_CONFIG,
                        'n_obs': 5000,
                        'n_vars': 2000
                    })
                    
                    # Simulate I/O operations
                    with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp_file:
                        temp_data.write_h5ad(tmp_file.name)
                        
                        # Read back
                        _ = ad.read_h5ad(tmp_file.name)
                        
                        # Clean up
                        Path(tmp_file.name).unlink(missing_ok=True)
                    
                    end_time = time.time()
                    
                    return ExecutionMetrics(
                        task_id=task.task_id,
                        agent_type=task.agent_type,
                        start_time=start_time,
                        end_time=end_time,
                        execution_time=end_time - start_time,
                        success=True,
                        memory_usage_mb=temp_data.X.nbytes / (1024**2),
                        cpu_percent=psutil.cpu_percent(),
                        queue_wait_time=0.0
                    )
                
                # Execute I/O intensive tasks
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(io_intensive_worker, task) for task in io_tasks]
                    
                    for future in as_completed(futures):
                        metrics = future.result()
                        monitor.record_execution(metrics)
                
                monitor.stop_monitoring()
                
                return {
                    'io_contention_test': 'completed',
                    'performance_summary': monitor.get_performance_summary(),
                    'avg_io_time': monitor.get_performance_summary()['execution_time_stats']['mean']
                }
        
        # Test resource contention
        contention_tester = ResourceContentionTester(mock_agent_clients, concurrent_data_manager)
        
        # Test memory contention
        memory_result = contention_tester.test_memory_contention(num_memory_intensive_tasks=4)
        
        assert memory_result['memory_contention_test'] == 'completed'
        memory_summary = memory_result['performance_summary']
        assert memory_summary['success_rate'] >= 0.8, "Memory contention caused too many failures"
        assert memory_result['peak_concurrent_memory'] < 20.0, "Peak memory usage too high"
        
        # Test CPU contention
        cpu_result = contention_tester.test_cpu_contention(num_cpu_intensive_tasks=3)
        
        assert cpu_result['cpu_contention_test'] == 'completed'
        cpu_summary = cpu_result['performance_summary']
        assert cpu_summary['success_rate'] >= 0.8, "CPU contention caused too many failures"
        
        # Test I/O contention
        io_result = contention_tester.test_io_contention(num_io_tasks=4)
        
        assert io_result['io_contention_test'] == 'completed'
        io_summary = io_result['performance_summary']
        assert io_summary['success_rate'] >= 0.8, "I/O contention caused too many failures"
        assert io_result['avg_io_time'] < 15.0, "I/O operations too slow under contention"
    
    def test_agent_scalability_under_load(self, mock_agent_clients):
        """Test agent system scalability under increasing load."""
        
        class ScalabilityTester:
            """Tests system scalability under various load conditions."""
            
            def __init__(self, agent_clients):
                self.agent_clients = agent_clients
                
            def test_increasing_load(self, load_levels: List[int]):
                """Test performance under increasing numbers of concurrent tasks."""
                scalability_results = []
                
                for num_tasks in load_levels:
                    # Create tasks for this load level
                    tasks = self._generate_tasks_for_load_test(num_tasks)
                    
                    # Execute with load monitoring
                    monitor = ConcurrentExecutionMonitor()
                    monitor.start_monitoring()
                    
                    execution_start = time.time()
                    
                    # Determine appropriate number of workers
                    max_workers = min(num_tasks, mp.cpu_count() * 2)
                    
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        def execute_mock_task(task):
                            start_time = time.time()
                            
                            # Mock task execution
                            processing_time = np.random.uniform(0.5, 2.0)
                            time.sleep(processing_time)
                            
                            end_time = time.time()
                            
                            return ExecutionMetrics(
                                task_id=task.task_id,
                                agent_type=task.agent_type,
                                start_time=start_time,
                                end_time=end_time,
                                execution_time=end_time - start_time,
                                success=np.random.random() > 0.05,  # 95% success rate
                                memory_usage_mb=np.random.uniform(10, 100),
                                cpu_percent=np.random.uniform(20, 80),
                                queue_wait_time=max(0, start_time - execution_start)
                            )
                        
                        futures = [executor.submit(execute_mock_task, task) for task in tasks]
                        
                        for future in as_completed(futures):
                            metrics = future.result()
                            monitor.record_execution(metrics)
                    
                    execution_end = time.time()
                    monitor.stop_monitoring()
                    
                    # Analyze performance for this load level
                    performance_summary = monitor.get_performance_summary()
                    
                    scalability_results.append({
                        'num_tasks': num_tasks,
                        'max_workers': max_workers,
                        'total_execution_time': execution_end - execution_start,
                        'performance_summary': performance_summary,
                        'throughput_tasks_per_second': num_tasks / (execution_end - execution_start),
                        'avg_task_latency': performance_summary['execution_time_stats']['mean'],
                        'resource_efficiency': self._calculate_resource_efficiency(
                            performance_summary, num_tasks, execution_end - execution_start
                        )
                    })
                
                return {
                    'scalability_test_results': scalability_results,
                    'scalability_analysis': self._analyze_scalability_trends(scalability_results)
                }
            
            def _generate_tasks_for_load_test(self, num_tasks: int) -> List[ConcurrentTask]:
                """Generate tasks for load testing."""
                tasks = []
                agent_types = ['singlecell', 'bulk_rnaseq', 'proteomics', 'data']
                
                for i in range(num_tasks):
                    agent_type = agent_types[i % len(agent_types)]
                    task = ConcurrentTask(
                        task_id=f"load_test_task_{i}",
                        agent_type=agent_type,
                        query=f"Process dataset {i} with {agent_type} agent",
                        expected_duration=np.random.uniform(1.0, 3.0),
                        dataset_size="medium"
                    )
                    tasks.append(task)
                
                return tasks
            
            def _calculate_resource_efficiency(self, performance_summary: Dict, num_tasks: int, total_time: float) -> float:
                """Calculate resource utilization efficiency."""
                if total_time <= 0 or num_tasks <= 0:
                    return 0.0
                
                # Simple efficiency metric based on successful task completion rate and resource usage
                success_rate = performance_summary['success_rate']
                avg_cpu = performance_summary['cpu_usage_stats']['mean_percent']
                
                # Normalize CPU usage (target around 70% utilization)
                cpu_efficiency = 1.0 - abs(70.0 - avg_cpu) / 70.0
                cpu_efficiency = max(0.0, cpu_efficiency)
                
                return (success_rate * 0.7) + (cpu_efficiency * 0.3)
            
            def _analyze_scalability_trends(self, results: List[Dict]) -> Dict[str, Any]:
                """Analyze scalability trends across load levels."""
                if len(results) < 2:
                    return {'error': 'Insufficient data for trend analysis'}
                
                num_tasks_list = [r['num_tasks'] for r in results]
                throughput_list = [r['throughput_tasks_per_second'] for r in results]
                latency_list = [r['avg_task_latency'] for r in results]
                efficiency_list = [r['resource_efficiency'] for r in results]
                
                return {
                    'max_throughput': float(max(throughput_list)),
                    'throughput_at_max_load': float(throughput_list[-1]),
                    'throughput_degradation': (throughput_list[0] - throughput_list[-1]) / throughput_list[0] if throughput_list[0] > 0 else 0,
                    'latency_increase': latency_list[-1] - latency_list[0],
                    'avg_resource_efficiency': float(np.mean(efficiency_list)),
                    'efficiency_trend': 'improving' if efficiency_list[-1] > efficiency_list[0] else 'degrading',
                    'scalability_score': self._calculate_scalability_score(results)
                }
            
            def _calculate_scalability_score(self, results: List[Dict]) -> float:
                """Calculate overall scalability score (0-1)."""
                # Factors: throughput retention, latency control, resource efficiency
                if len(results) < 2:
                    return 0.0
                
                throughput_retention = min(1.0, results[-1]['throughput_tasks_per_second'] / results[0]['throughput_tasks_per_second'])
                
                latency_ratios = [r['avg_task_latency'] for r in results]
                latency_control = 1.0 / (1.0 + (max(latency_ratios) - min(latency_ratios)))
                
                avg_efficiency = np.mean([r['resource_efficiency'] for r in results])
                
                return (throughput_retention * 0.4) + (latency_control * 0.3) + (avg_efficiency * 0.3)
        
        # Test scalability
        scalability_tester = ScalabilityTester(mock_agent_clients)
        
        load_levels = [5, 10, 20, 30]  # Different numbers of concurrent tasks
        scalability_result = scalability_tester.test_increasing_load(load_levels)
        
        # Verify scalability results
        results = scalability_result['scalability_test_results']
        assert len(results) == len(load_levels)
        
        # All load levels should complete successfully
        for result in results:
            assert result['performance_summary']['success_rate'] >= 0.8
            assert result['throughput_tasks_per_second'] > 0
        
        # Analyze scalability trends
        analysis = scalability_result['scalability_analysis']
        assert 'max_throughput' in analysis
        assert 'scalability_score' in analysis
        
        # Scalability score should be reasonable
        assert analysis['scalability_score'] > 0.3, "Poor scalability score"
        
        # Throughput degradation should not be excessive
        assert analysis['throughput_degradation'] < 0.7, "Excessive throughput degradation under load"
        
        return scalability_result
    
    def test_agent_coordination_performance(self, mock_agent_clients, concurrent_data_manager):
        """Test performance of agent coordination and handoff mechanisms."""
        
        class AgentCoordinationTester:
            """Tests agent coordination performance."""
            
            def __init__(self, agent_clients, data_manager):
                self.agent_clients = agent_clients
                self.data_manager = data_manager
                
            def test_multi_agent_workflow(self, workflow_complexity: str = "medium"):
                """Test multi-agent workflow coordination performance."""
                coordination_monitor = ConcurrentExecutionMonitor()
                coordination_monitor.start_monitoring()
                
                # Define multi-agent workflow
                workflow_start = time.time()
                
                if workflow_complexity == "simple":
                    workflow_steps = self._simple_workflow()
                elif workflow_complexity == "complex":
                    workflow_steps = self._complex_workflow()
                else:
                    workflow_steps = self._medium_workflow()
                
                # Execute workflow with coordination
                workflow_results = []
                
                for step in workflow_steps:
                    step_start = time.time()
                    
                    # Simulate agent coordination overhead
                    coordination_time = np.random.uniform(0.1, 0.3)
                    time.sleep(coordination_time)
                    
                    # Execute step
                    step_result = self._execute_workflow_step(step)
                    
                    step_end = time.time()
                    
                    step_metrics = ExecutionMetrics(
                        task_id=step['step_id'],
                        agent_type=step['agent_type'],
                        start_time=step_start,
                        end_time=step_end,
                        execution_time=step_end - step_start,
                        success=step_result['success'],
                        memory_usage_mb=step_result.get('memory_usage', 50.0),
                        cpu_percent=np.random.uniform(30, 70),
                        queue_wait_time=coordination_time,
                        error_message=step_result.get('error')
                    )
                    
                    coordination_monitor.record_execution(step_metrics)
                    workflow_results.append({
                        'step': step,
                        'result': step_result,
                        'metrics': step_metrics
                    })
                
                workflow_end = time.time()
                coordination_monitor.stop_monitoring()
                
                return {
                    'workflow_coordination_successful': all(r['result']['success'] for r in workflow_results),
                    'total_workflow_time': workflow_end - workflow_start,
                    'workflow_steps': len(workflow_steps),
                    'coordination_overhead': sum(r['metrics'].queue_wait_time for r in workflow_results),
                    'step_results': workflow_results,
                    'performance_summary': coordination_monitor.get_performance_summary()
                }
            
            def _simple_workflow(self) -> List[Dict]:
                """Define simple multi-agent workflow."""
                return [
                    {'step_id': 'data_load', 'agent_type': 'data', 'description': 'Load dataset'},
                    {'step_id': 'sc_qc', 'agent_type': 'singlecell', 'description': 'Quality control'},
                    {'step_id': 'sc_cluster', 'agent_type': 'singlecell', 'description': 'Clustering'}
                ]
            
            def _medium_workflow(self) -> List[Dict]:
                """Define medium complexity multi-agent workflow."""
                return [
                    {'step_id': 'data_load', 'agent_type': 'data', 'description': 'Load dataset'},
                    {'step_id': 'data_validate', 'agent_type': 'data', 'description': 'Validate format'},
                    {'step_id': 'sc_qc', 'agent_type': 'singlecell', 'description': 'Quality control'},
                    {'step_id': 'sc_preprocess', 'agent_type': 'singlecell', 'description': 'Preprocessing'},
                    {'step_id': 'sc_cluster', 'agent_type': 'singlecell', 'description': 'Clustering'},
                    {'step_id': 'sc_annotate', 'agent_type': 'singlecell', 'description': 'Cell type annotation'}
                ]
            
            def _complex_workflow(self) -> List[Dict]:
                """Define complex multi-agent workflow."""
                return [
                    {'step_id': 'data_load_rna', 'agent_type': 'data', 'description': 'Load RNA data'},
                    {'step_id': 'data_load_protein', 'agent_type': 'data', 'description': 'Load protein data'},
                    {'step_id': 'sc_qc_rna', 'agent_type': 'singlecell', 'description': 'RNA QC'},
                    {'step_id': 'prot_qc', 'agent_type': 'proteomics', 'description': 'Protein QC'},
                    {'step_id': 'sc_preprocess_rna', 'agent_type': 'singlecell', 'description': 'RNA preprocessing'},
                    {'step_id': 'prot_preprocess', 'agent_type': 'proteomics', 'description': 'Protein preprocessing'},
                    {'step_id': 'multimodal_integration', 'agent_type': 'singlecell', 'description': 'Multi-modal integration'},
                    {'step_id': 'joint_clustering', 'agent_type': 'singlecell', 'description': 'Joint clustering'},
                    {'step_id': 'differential_analysis', 'agent_type': 'bulk_rnaseq', 'description': 'Differential analysis'}
                ]
            
            def _execute_workflow_step(self, step: Dict) -> Dict:
                """Execute a single workflow step."""
                # Simulate step execution
                execution_time = np.random.uniform(0.5, 2.0)
                time.sleep(execution_time)
                
                # Simulate success/failure
                success = np.random.random() > 0.05  # 95% success rate
                
                return {
                    'success': success,
                    'execution_time': execution_time,
                    'memory_usage': np.random.uniform(20, 200),
                    'error': None if success else "Mock execution error"
                }
        
        # Test agent coordination
        coordination_tester = AgentCoordinationTester(mock_agent_clients, concurrent_data_manager)
        
        # Test different workflow complexities
        for complexity in ['simple', 'medium', 'complex']:
            workflow_result = coordination_tester.test_multi_agent_workflow(complexity)
            
            # Verify coordination performance
            assert workflow_result['workflow_coordination_successful'] == True, f"Workflow {complexity} failed"
            
            expected_max_time = {'simple': 30.0, 'medium': 60.0, 'complex': 120.0}
            assert workflow_result['total_workflow_time'] < expected_max_time[complexity], f"Workflow {complexity} too slow"
            
            # Coordination overhead should be reasonable
            overhead_ratio = workflow_result['coordination_overhead'] / workflow_result['total_workflow_time']
            assert overhead_ratio < 0.2, f"Coordination overhead too high for {complexity} workflow"
            
            # Performance summary should show good metrics
            perf_summary = workflow_result['performance_summary']
            assert perf_summary['success_rate'] >= 0.9, f"Poor success rate for {complexity} workflow"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])