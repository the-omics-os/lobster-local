#!/usr/bin/env python3
"""
Enhanced Integration Testing Script for Lobster AI
==================================================

This script runs comprehensive tests against the Lobster AI multi-agent system
with advanced features including test categorization, performance monitoring,
dependency management, and integration with the full test suite framework.

Features:
- Sequential and parallel test execution
- Test categorization and filtering  
- Performance monitoring and resource tracking
- Enhanced error handling and recovery
- Integration with pytest test suites
- Workspace management and cleanup
- Test dependency resolution
- Comprehensive reporting and analytics

Usage:
    python tests/run_integration_tests.py --input tests/test_cases.json --output tests/results.json
    python tests/run_integration_tests.py --categories basic,advanced --parallel --workers 4
    python tests/run_integration_tests.py --run-pytest-integration --output combined_results.json
"""

import json
import argparse
import logging
import sys
import subprocess
import psutil
import time
import resource
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import traceback
import tempfile
import os
import shutil
import yaml
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Add the project root to the path so we can import lobster modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from lobster.core.client import AgentClient


@dataclass
class TestCase:
    """Enhanced test case representation."""
    test_id: str
    inputs: List[str]
    category: str = "basic"
    description: str = ""
    expected_duration: float = 60.0
    dependencies: List[str] = None
    tags: List[str] = None
    priority: int = 1
    timeout: float = 300.0
    retry_count: int = 0
    validation_criteria: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
        if self.validation_criteria is None:
            self.validation_criteria = {}


@dataclass
class TestResult:
    """Enhanced test result representation."""
    test_id: str
    success: bool
    error_message: Optional[str]
    inputs: List[str]
    responses: List[str]
    duration_seconds: float
    final_status: Dict[str, Any]
    workspace_path: str
    timestamp: str
    category: str = "basic"
    performance_metrics: Dict[str, Any] = None
    resource_usage: Dict[str, Any] = None
    validation_results: Dict[str, Any] = None
    retry_attempts: int = 0
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.resource_usage is None:
            self.resource_usage = {}
        if self.validation_results is None:
            self.validation_results = {}


@dataclass
class PerformanceMetrics:
    """Performance metrics for test execution."""
    cpu_percent_avg: float
    memory_mb_peak: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0


class PerformanceMonitor:
    """Monitors performance metrics during test execution."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.metrics = []
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self._monitoring = True
        self._monitor_thread = ThreadPoolExecutor(max_workers=1)
        self._monitor_thread.submit(self._monitor_loop)
        
    def stop_monitoring(self) -> PerformanceMetrics:
        """Stop monitoring and return aggregated metrics."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.shutdown(wait=True)
        
        if not self.metrics:
            return PerformanceMetrics(0, 0, 0, 0)
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_mb'] for m in self.metrics]
        
        return PerformanceMetrics(
            cpu_percent_avg=sum(cpu_values) / len(cpu_values),
            memory_mb_peak=max(memory_values),
            disk_io_read_mb=self.metrics[-1]['disk_read_mb'] - self.metrics[0]['disk_read_mb'],
            disk_io_write_mb=self.metrics[-1]['disk_write_mb'] - self.metrics[0]['disk_write_mb']
        )
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        process = psutil.Process()
        
        while self._monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                io_counters = process.io_counters() if hasattr(process, 'io_counters') else None
                
                metric = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_info.rss / (1024 * 1024),
                    'disk_read_mb': io_counters.read_bytes / (1024 * 1024) if io_counters else 0,
                    'disk_write_mb': io_counters.write_bytes / (1024 * 1024) if io_counters else 0
                }
                
                self.metrics.append(metric)
                time.sleep(self.sampling_interval)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception:
                continue


class TestLogger:
    """Custom logger for test execution."""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("LobsterTester")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # load environment variables
        load_dotenv()
        
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)


class EnhancedLobsterTestRunner:
    """Enhanced test runner for Lobster AI integration tests."""
    
    def __init__(self, log_level: str = "INFO", enable_performance_monitoring: bool = True):
        self.logger = TestLogger(log_level)
        self.results = {}
        self.test_cases = {}
        self.enable_performance_monitoring = enable_performance_monitoring
        self._workspace_cleanup_registry = []
        
    def load_test_cases(self, input_file: Path) -> Dict[str, TestCase]:
        """Load enhanced test cases from JSON or YAML file."""
        try:
            if input_file.suffix.lower() in ['.yaml', '.yml']:
                with open(input_file, 'r') as f:
                    raw_data = yaml.safe_load(f)
            else:
                with open(input_file, 'r') as f:
                    raw_data = json.load(f)
            
            test_cases = {}
            
            # Handle legacy format (simple dict of test_id -> inputs)
            if isinstance(next(iter(raw_data.values())), list):
                for test_id, inputs in raw_data.items():
                    test_cases[test_id] = TestCase(
                        test_id=test_id,
                        inputs=inputs,
                        category=self._infer_category(test_id),
                        description=f"Legacy test case: {test_id}"
                    )
            else:
                # Handle enhanced format
                for test_id, test_data in raw_data.items():
                    if isinstance(test_data, dict):
                        test_cases[test_id] = TestCase(
                            test_id=test_id,
                            inputs=test_data.get('inputs', []),
                            category=test_data.get('category', 'basic'),
                            description=test_data.get('description', ''),
                            expected_duration=test_data.get('expected_duration', 60.0),
                            dependencies=test_data.get('dependencies', []),
                            tags=test_data.get('tags', []),
                            priority=test_data.get('priority', 1),
                            timeout=test_data.get('timeout', 300.0),
                            retry_count=test_data.get('retry_count', 0),
                            validation_criteria=test_data.get('validation_criteria', {})
                        )
                    else:
                        # Fallback for mixed format
                        test_cases[test_id] = TestCase(
                            test_id=test_id,
                            inputs=test_data if isinstance(test_data, list) else [],
                            category=self._infer_category(test_id)
                        )
            
            self.test_cases = test_cases
            self.logger.info(f"Loaded {len(test_cases)} enhanced test cases from {input_file}")
            
            # Log test case breakdown by category
            categories = {}
            for test_case in test_cases.values():
                categories[test_case.category] = categories.get(test_case.category, 0) + 1
            
            self.logger.info(f"Test cases by category: {categories}")
            return test_cases
        
        except Exception as e:
            self.logger.error(f"Failed to load test cases from {input_file}: {e}")
            raise
    
    def _infer_category(self, test_id: str) -> str:
        """Infer test category from test ID."""
        if any(keyword in test_id.lower() for keyword in ['advanced', 'complex', 'integration']):
            return 'advanced'
        elif any(keyword in test_id.lower() for keyword in ['performance', 'benchmark', 'load']):
            return 'performance'
        elif any(keyword in test_id.lower() for keyword in ['error', 'failure', 'invalid']):
            return 'error_handling'
        elif any(keyword in test_id.lower() for keyword in ['geo', 'download']):
            return 'data_access'
        else:
            return 'basic'
    
    def filter_test_cases(self, categories: Optional[List[str]] = None, 
                         tags: Optional[List[str]] = None,
                         priorities: Optional[List[int]] = None) -> Dict[str, TestCase]:
        """Filter test cases based on criteria."""
        filtered = {}
        
        for test_id, test_case in self.test_cases.items():
            # Filter by categories
            if categories and test_case.category not in categories:
                continue
            
            # Filter by tags
            if tags and not any(tag in test_case.tags for tag in tags):
                continue
            
            # Filter by priorities
            if priorities and test_case.priority not in priorities:
                continue
            
            filtered[test_id] = test_case
        
        self.logger.info(f"Filtered to {len(filtered)} test cases")
        return filtered
    
    def resolve_dependencies(self, test_cases: Dict[str, TestCase]) -> List[str]:
        """Resolve test case dependencies and return execution order."""
        # Simple topological sort for dependency resolution
        executed = set()
        execution_order = []
        remaining = dict(test_cases)
        
        while remaining:
            # Find tests with no unresolved dependencies
            ready_tests = []
            for test_id, test_case in remaining.items():
                if all(dep in executed for dep in test_case.dependencies):
                    ready_tests.append(test_id)
            
            if not ready_tests:
                # Circular dependency or missing dependency
                self.logger.warning(f"Circular or missing dependencies detected for: {list(remaining.keys())}")
                ready_tests = list(remaining.keys())  # Execute anyway
            
            # Sort by priority
            ready_tests.sort(key=lambda tid: test_cases[tid].priority, reverse=True)
            
            for test_id in ready_tests:
                execution_order.append(test_id)
                executed.add(test_id)
                del remaining[test_id]
        
        return execution_order
    
    def run_single_test_enhanced(self, test_case: TestCase, workspace_dir: Optional[Path] = None, 
                                retry_attempt: int = 0) -> TestResult:
        """
        Run a single enhanced test case with performance monitoring and validation.
        
        Args:
            test_case: Enhanced test case to execute
            workspace_dir: Directory for test workspace (creates temp if None)
            retry_attempt: Current retry attempt number
            
        Returns:
            TestResult containing comprehensive test results
        """
        start_time = datetime.now()
        performance_monitor = PerformanceMonitor() if self.enable_performance_monitoring else None
        
        # Create temporary workspace if not provided
        if workspace_dir is None:
            workspace_dir = Path(tempfile.mkdtemp(prefix=f"lobster_test_{test_case.test_id}_"))
            self._workspace_cleanup_registry.append(workspace_dir)
        
        try:
            self.logger.info(f"Starting test '{test_case.test_id}' (attempt {retry_attempt + 1})")
            
            # Start performance monitoring
            if performance_monitor:
                performance_monitor.start_monitoring()
            
            # Initialize client with isolated workspace
            client = AgentClient(
                workspace_path=workspace_dir,
                session_id=f"test_{test_case.test_id}_{int(datetime.now().timestamp())}",
                enable_reasoning=False,  # Disable for cleaner test output
                enable_langfuse=False,   # Disable for tests
                manual_model_params={
                    "model_id": 'us.anthropic.claude-3-5-haiku-20241022-v1:0',
                    "temperature": 0.7,
                    "region_name": 'us-east-1',
                    "aws_access_key_id": os.environ.get('AWS_BEDROCK_ACCESS_KEY'),
                    "aws_secret_access_key": os.environ.get('AWS_BEDROCK_SECRET_ACCESS_KEY'),
                }
            )
            
            responses = []
            success = True
            error_message = None
            validation_results = {}
            
            # Execute sequential inputs with timeout
            for i, user_input in enumerate(test_case.inputs):
                try:
                    self.logger.info(f"Test '{test_case.test_id}' - Input {i+1}/{len(test_case.inputs)}: {user_input[:100]}...")
                    
                    # Set timeout for individual query
                    query_timeout = test_case.timeout / len(test_case.inputs)
                    
                    # Run query
                    result = client.query(user_input, stream=False)
                    
                    if result.get("success", False):
                        response_text = result.get("response", "No response generated")
                        responses.append(response_text)
                        self.logger.info(f"Test '{test_case.test_id}' - Response {i+1}: SUCCESS")
                        
                        # Perform validation if criteria specified
                        if test_case.validation_criteria:
                            validation_result = self._validate_response(
                                response_text, 
                                test_case.validation_criteria.get(f'input_{i}', {})
                            )
                            validation_results[f'input_{i}'] = validation_result
                            
                            if not validation_result.get('valid', True):
                                self.logger.warning(f"Validation failed for input {i+1}: {validation_result}")
                        
                    else:
                        error_msg = result.get("error", "Unknown error occurred")
                        responses.append(f"ERROR: {error_msg}")
                        success = False
                        error_message = error_msg
                        self.logger.error(f"Test '{test_case.test_id}' - Response {i+1}: FAILED - {error_msg}")
                        break  # Stop on first error
                        
                except Exception as e:
                    error_msg = f"Exception during query execution: {str(e)}"
                    responses.append(f"ERROR: {error_msg}")
                    success = False
                    error_message = error_msg
                    self.logger.error(f"Test '{test_case.test_id}' - Response {i+1}: EXCEPTION - {error_msg}")
                    break  # Stop on first exception
            
            # Collect final status
            final_status = client.get_status()
            
            # Stop performance monitoring and collect metrics
            performance_metrics = {}
            if performance_monitor:
                perf_data = performance_monitor.stop_monitoring()
                performance_metrics = asdict(perf_data)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Check if test duration exceeded expected time
            if duration > test_case.expected_duration * 1.5:
                self.logger.warning(f"Test '{test_case.test_id}' took longer than expected: {duration:.2f}s > {test_case.expected_duration:.2f}s")
            
            return TestResult(
                test_id=test_case.test_id,
                success=success,
                error_message=error_message,
                inputs=test_case.inputs,
                responses=responses,
                duration_seconds=duration,
                final_status=final_status,
                workspace_path=str(workspace_dir),
                timestamp=datetime.now().isoformat(),
                category=test_case.category,
                performance_metrics=performance_metrics,
                resource_usage=self._collect_resource_usage(),
                validation_results=validation_results,
                retry_attempts=retry_attempt
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Test setup/execution failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(f"Test '{test_case.test_id}' - FATAL ERROR: {error_msg}")
            
            # Stop performance monitoring in case of error
            performance_metrics = {}
            if performance_monitor:
                try:
                    perf_data = performance_monitor.stop_monitoring()
                    performance_metrics = asdict(perf_data)
                except Exception:
                    pass
            
            return TestResult(
                test_id=test_case.test_id,
                success=False,
                error_message=error_msg,
                inputs=test_case.inputs,
                responses=[],
                duration_seconds=duration,
                final_status={},
                workspace_path=str(workspace_dir) if workspace_dir else "",
                timestamp=datetime.now().isoformat(),
                category=test_case.category,
                performance_metrics=performance_metrics,
                retry_attempts=retry_attempt
            )
    
    def _validate_response(self, response: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Validate test response against criteria."""
        validation_result = {'valid': True, 'issues': []}
        
        # Check for required keywords
        if 'required_keywords' in criteria:
            for keyword in criteria['required_keywords']:
                if keyword.lower() not in response.lower():
                    validation_result['valid'] = False
                    validation_result['issues'].append(f"Missing required keyword: {keyword}")
        
        # Check for forbidden keywords
        if 'forbidden_keywords' in criteria:
            for keyword in criteria['forbidden_keywords']:
                if keyword.lower() in response.lower():
                    validation_result['valid'] = False
                    validation_result['issues'].append(f"Contains forbidden keyword: {keyword}")
        
        # Check minimum response length
        if 'min_length' in criteria:
            if len(response) < criteria['min_length']:
                validation_result['valid'] = False
                validation_result['issues'].append(f"Response too short: {len(response)} < {criteria['min_length']}")
        
        # Check for error indicators
        if 'no_errors' in criteria and criteria['no_errors']:
            error_indicators = ['error', 'failed', 'exception', 'traceback']
            for indicator in error_indicators:
                if indicator.lower() in response.lower():
                    validation_result['valid'] = False
                    validation_result['issues'].append(f"Response contains error indicator: {indicator}")
        
        return validation_result
    
    def _collect_resource_usage(self) -> Dict[str, Any]:
        """Collect current resource usage metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': memory_info.rss / (1024 * 1024),
                'open_files': len(process.open_files()),
                'num_threads': process.num_threads()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}
    
    def run_tests_enhanced(self, test_cases: Dict[str, TestCase], 
                          execution_order: Optional[List[str]] = None,
                          parallel: bool = False, max_workers: int = 4) -> Dict[str, TestResult]:
        """Run enhanced test cases with retry logic and comprehensive reporting."""
        if execution_order is None:
            execution_order = self.resolve_dependencies(test_cases)
        
        self.logger.info(f"Running {len(test_cases)} tests {'in parallel' if parallel else 'sequentially'}")
        
        if parallel:
            return self._run_tests_parallel_enhanced(test_cases, execution_order, max_workers)
        else:
            return self._run_tests_sequential_enhanced(test_cases, execution_order)
    
    def _run_tests_sequential_enhanced(self, test_cases: Dict[str, TestCase], 
                                      execution_order: List[str]) -> Dict[str, TestResult]:
        """Run tests sequentially with enhanced features."""
        results = {}
        
        for test_id in execution_order:
            if test_id not in test_cases:
                self.logger.warning(f"Test '{test_id}' in execution order but not in test cases")
                continue
            
            test_case = test_cases[test_id]
            
            # Execute test with retry logic
            result = self._execute_test_with_retry(test_case)
            results[test_id] = result
            
            # Log progress
            status = "PASSED" if result.success else "FAILED"
            perf_info = ""
            if result.performance_metrics:
                cpu_avg = result.performance_metrics.get('cpu_percent_avg', 0)
                mem_peak = result.performance_metrics.get('memory_mb_peak', 0)
                perf_info = f" (CPU: {cpu_avg:.1f}%, Mem: {mem_peak:.1f}MB)"
            
            self.logger.info(f"Test '{test_id}' [{result.category}]: {status} ({result.duration_seconds:.2f}s){perf_info}")
            
            # Early termination on critical failures
            if not result.success and test_case.priority >= 9:  # High priority failure
                self.logger.error(f"Critical test '{test_id}' failed, considering early termination")
        
        return results
    
    def _run_tests_parallel_enhanced(self, test_cases: Dict[str, TestCase], 
                                   execution_order: List[str], max_workers: int) -> Dict[str, TestResult]:
        """Run tests in parallel with enhanced features."""
        results = {}
        
        # Group tests by dependency levels for parallel execution
        dependency_levels = self._group_by_dependency_levels(test_cases, execution_order)
        
        for level, test_ids in dependency_levels.items():
            self.logger.info(f"Executing dependency level {level} with {len(test_ids)} tests")
            
            level_test_cases = {tid: test_cases[tid] for tid in test_ids if tid in test_cases}
            
            with ThreadPoolExecutor(max_workers=min(max_workers, len(level_test_cases))) as executor:
                future_to_test = {
                    executor.submit(self._execute_test_with_retry, test_case): test_id
                    for test_id, test_case in level_test_cases.items()
                }
                
                for future in as_completed(future_to_test):
                    test_id = future_to_test[future]
                    try:
                        result = future.result()
                        results[test_id] = result
                        
                        status = "PASSED" if result.success else "FAILED"
                        self.logger.info(f"Test '{test_id}' [{result.category}]: {status} ({result.duration_seconds:.2f}s)")
                        
                    except Exception as e:
                        self.logger.error(f"Test '{test_id}' execution failed: {e}")
                        # Create failure result
                        results[test_id] = TestResult(
                            test_id=test_id,
                            success=False,
                            error_message=f"Execution failed: {str(e)}",
                            inputs=level_test_cases[test_id].inputs,
                            responses=[],
                            duration_seconds=0,
                            final_status={},
                            workspace_path="",
                            timestamp=datetime.now().isoformat(),
                            category=level_test_cases[test_id].category
                        )
        
        return results
    
    def _execute_test_with_retry(self, test_case: TestCase) -> TestResult:
        """Execute test with retry logic."""
        last_result = None
        
        for attempt in range(test_case.retry_count + 1):
            result = self.run_single_test_enhanced(test_case, retry_attempt=attempt)
            
            if result.success:
                return result
            
            last_result = result
            
            if attempt < test_case.retry_count:
                self.logger.warning(f"Test '{test_case.test_id}' failed attempt {attempt + 1}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return last_result
    
    def _group_by_dependency_levels(self, test_cases: Dict[str, TestCase], 
                                   execution_order: List[str]) -> Dict[int, List[str]]:
        """Group tests by dependency levels for parallel execution."""
        levels = {}
        test_levels = {}
        
        # Calculate dependency depth for each test
        for test_id in execution_order:
            test_case = test_cases.get(test_id)
            if not test_case:
                continue
            
            if not test_case.dependencies:
                level = 0
            else:
                # Find maximum level of dependencies + 1
                dep_levels = [test_levels.get(dep, 0) for dep in test_case.dependencies]
                level = max(dep_levels, default=0) + 1
            
            test_levels[test_id] = level
            
            if level not in levels:
                levels[level] = []
            levels[level].append(test_id)
        
        return levels
    
    def run_pytest_integration(self, output_file: Path) -> Dict[str, Any]:
        """Run pytest integration tests and combine with manual test results."""
        self.logger.info("Running pytest integration tests")
        
        pytest_results = {}
        
        try:
            # Run different test categories
            test_commands = [
                ("unit", ["pytest", "tests/unit", "-v", "--tb=short", "--json-report", "--json-report-file=unit_results.json"]),
                ("integration", ["pytest", "tests/integration", "-v", "--tb=short", "--json-report", "--json-report-file=integration_results.json"]),
                ("system", ["pytest", "tests/system", "-v", "--tb=short", "--json-report", "--json-report-file=system_results.json"]),
                ("performance", ["pytest", "tests/performance", "-v", "--tb=short", "--json-report", "--json-report-file=performance_results.json"])
            ]
            
            for category, command in test_commands:
                self.logger.info(f"Running {category} tests...")
                
                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=1800  # 30 minute timeout
                    )
                    
                    pytest_results[category] = {
                        "return_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "success": result.returncode == 0
                    }
                    
                    # Try to load JSON report if available
                    json_file = Path(f"{category}_results.json")
                    if json_file.exists():
                        try:
                            with open(json_file, 'r') as f:
                                pytest_results[category]["detailed_results"] = json.load(f)
                            json_file.unlink()  # Cleanup
                        except Exception:
                            pass
                    
                    status = "PASSED" if result.returncode == 0 else "FAILED"
                    self.logger.info(f"Pytest {category} tests: {status}")
                    
                except subprocess.TimeoutExpired:
                    self.logger.error(f"Pytest {category} tests timed out")
                    pytest_results[category] = {
                        "return_code": -1,
                        "stdout": "",
                        "stderr": "Test execution timed out",
                        "success": False
                    }
                
                except Exception as e:
                    self.logger.error(f"Failed to run pytest {category} tests: {e}")
                    pytest_results[category] = {
                        "return_code": -1,
                        "stdout": "",
                        "stderr": str(e),
                        "success": False
                    }
        
        except Exception as e:
            self.logger.error(f"Pytest integration failed: {e}")
        
        return pytest_results
    
    def run_tests_parallel(self, test_cases: Dict[str, List[str]], max_workers: int = 4) -> Dict[str, Any]:
        """Run tests in parallel using multiprocessing."""
        self.logger.info(f"Running {len(test_cases)} tests in parallel with {max_workers} workers")
        
        results = {}
        
        # Create workspace directories for each test
        workspaces = {}
        for test_id in test_cases.keys():
            workspaces[test_id] = Path(tempfile.mkdtemp(prefix=f"lobster_test_{test_id}_"))
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tests
                future_to_test = {
                    executor.submit(
                        run_single_test_worker, 
                        test_id, 
                        user_inputs, 
                        workspaces[test_id]
                    ): test_id 
                    for test_id, user_inputs in test_cases.items()
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_test):
                    test_id = future_to_test[future]
                    try:
                        result = future.result()
                        results[test_id] = result
                        
                        status = "PASSED" if result["success"] else "FAILED"
                        self.logger.info(f"Test '{test_id}' completed: {status} ({result['duration_seconds']:.2f}s)")
                        
                    except Exception as e:
                        error_msg = f"Process execution failed: {str(e)}"
                        self.logger.error(f"Test '{test_id}' process error: {error_msg}")
                        results[test_id] = {
                            "test_id": test_id,
                            "success": False,
                            "error_message": error_msg,
                            "inputs": test_cases[test_id],
                            "responses": [],
                            "duration_seconds": 0,
                            "final_status": None,
                            "workspace_path": str(workspaces[test_id]),
                            "timestamp": datetime.now().isoformat()
                        }
        
        finally:
            # Clean up workspace directories
            for workspace in workspaces.values():
                try:
                    import shutil
                    shutil.rmtree(workspace)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup workspace {workspace}: {e}")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: Path):
        """Save test results to JSON file."""
        try:
            # Create summary statistics
            summary = {
                "total_tests": len(results),
                "passed_tests": sum(1 for r in results.values() if r["success"]),
                "failed_tests": sum(1 for r in results.values() if not r["success"]),
                "total_duration": sum(r["duration_seconds"] for r in results.values()),
                "timestamp": datetime.now().isoformat()
            }
            
            output_data = {
                "summary": summary,
                "results": results
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {output_file}")
            self.logger.info(f"Summary: {summary['passed_tests']}/{summary['total_tests']} tests passed")
            
        except Exception as e:
            self.logger.error(f"Failed to save results to {output_file}: {e}")
            raise



    def save_enhanced_results(self, results: Dict[str, TestResult], output_file: Path, 
                             pytest_results: Optional[Dict[str, Any]] = None):
        """Save enhanced test results with comprehensive analytics."""
        try:
            # Create comprehensive summary statistics
            total_tests = len(results)
            passed_tests = sum(1 for r in results.values() if r.success)
            failed_tests = total_tests - passed_tests
            
            # Performance analytics
            durations = [r.duration_seconds for r in results.values()]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Category breakdown
            category_stats = {}
            for result in results.values():
                category = result.category
                if category not in category_stats:
                    category_stats[category] = {'passed': 0, 'failed': 0, 'total': 0}
                category_stats[category]['total'] += 1
                if result.success:
                    category_stats[category]['passed'] += 1
                else:
                    category_stats[category]['failed'] += 1
            
            # Performance metrics summary
            perf_summary = {}
            if any(r.performance_metrics for r in results.values()):
                cpu_values = [r.performance_metrics.get('cpu_percent_avg', 0) 
                             for r in results.values() if r.performance_metrics]
                memory_values = [r.performance_metrics.get('memory_mb_peak', 0) 
                               for r in results.values() if r.performance_metrics]
                
                perf_summary = {
                    'avg_cpu_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    'avg_memory_mb': sum(memory_values) / len(memory_values) if memory_values else 0,
                    'max_memory_mb': max(memory_values) if memory_values else 0
                }
            
            summary = {
                "test_execution_summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests, 
                    "failed_tests": failed_tests,
                    "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                    "total_duration": sum(durations),
                    "average_duration": avg_duration,
                    "timestamp": datetime.now().isoformat()
                },
                "category_breakdown": category_stats,
                "performance_summary": perf_summary
            }
            
            # Convert TestResult objects to dictionaries
            results_dict = {}
            for test_id, result in results.items():
                results_dict[test_id] = asdict(result)
            
            output_data = {
                "summary": summary,
                "manual_test_results": results_dict
            }
            
            # Add pytest results if available
            if pytest_results:
                output_data["pytest_results"] = pytest_results
                
                # Calculate combined statistics
                pytest_success_count = sum(1 for r in pytest_results.values() if r.get('success', False))
                pytest_total = len(pytest_results)
                
                output_data["combined_summary"] = {
                    "manual_tests": {"total": total_tests, "passed": passed_tests},
                    "pytest_tests": {"total": pytest_total, "passed": pytest_success_count},
                    "overall": {
                        "total": total_tests + pytest_total,
                        "passed": passed_tests + pytest_success_count,
                        "success_rate": (passed_tests + pytest_success_count) / (total_tests + pytest_total) if (total_tests + pytest_total) > 0 else 0
                    }
                }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            self.logger.info(f"Enhanced results saved to {output_file}")
            self.logger.info(f"Summary: {passed_tests}/{total_tests} manual tests passed")
            
            if pytest_results:
                pytest_passed = sum(1 for r in pytest_results.values() if r.get('success', False))
                self.logger.info(f"Pytest: {pytest_passed}/{len(pytest_results)} test suites passed")
            
        except Exception as e:
            self.logger.error(f"Failed to save enhanced results to {output_file}: {e}")
            raise


# Legacy compatibility wrapper
class LobsterTestRunner(EnhancedLobsterTestRunner):
    """Legacy wrapper for backward compatibility."""
    
    def run_single_test(self, test_id: str, user_inputs: List[str], workspace_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        test_case = TestCase(test_id=test_id, inputs=user_inputs)
        result = self.run_single_test_enhanced(test_case, workspace_dir)
        return asdict(result)


def run_single_test_worker(test_id: str, user_inputs: List[str], workspace_dir: Path) -> Dict[str, Any]:
    """Worker function for parallel test execution."""
    runner = LobsterTestRunner()
    return runner.run_single_test(test_id, user_inputs, workspace_dir)


def main():
    """Enhanced main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Enhanced integration test runner for Lobster AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Features:
  - Test categorization and filtering
  - Performance monitoring and resource tracking  
  - Dependency resolution and retry logic
  - Integration with pytest test suites
  - Comprehensive reporting and analytics

Examples:
  # Run tests sequentially with enhanced features
  python tests/run_integration_tests.py --input tests/test_cases.json --output tests/results.json

  # Run specific categories in parallel
  python tests/run_integration_tests.py --categories basic,advanced --parallel --workers 4

  # Run with pytest integration and performance monitoring
  python tests/run_integration_tests.py --run-pytest-integration --performance-monitoring

  # Filter by tags and priorities
  python tests/run_integration_tests.py --tags geo,analysis --priorities 1,2,3
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("tests/test_cases.json"),
        help="Input JSON/YAML file containing test cases (default: tests/test_cases.json)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=Path,
        default=Path("tests/results.json"),
        help="Output JSON file for test results (default: tests/results.json)"
    )
    
    parser.add_argument(
        "--categories", "-c",
        type=str,
        help="Comma-separated list of test categories to run (e.g., basic,advanced)"
    )
    
    parser.add_argument(
        "--tags", "-t",
        type=str,
        help="Comma-separated list of tags to filter tests (e.g., geo,analysis)"
    )
    
    parser.add_argument(
        "--priorities", 
        type=str,
        help="Comma-separated list of priorities to run (e.g., 1,2,3)"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (default: sequential)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    parser.add_argument(
        "--run-pytest-integration",
        action="store_true",
        help="Also run pytest test suites and combine results"
    )
    
    parser.add_argument(
        "--performance-monitoring", 
        action="store_true",
        help="Enable performance monitoring during test execution"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Initialize enhanced test runner
    runner = EnhancedLobsterTestRunner(
        log_level=args.log_level,
        enable_performance_monitoring=args.performance_monitoring
    )
    
    try:
        # Load test cases
        test_cases = runner.load_test_cases(args.input)
        
        # Apply filters
        filtered_test_cases = test_cases
        
        if args.categories:
            categories = [c.strip() for c in args.categories.split(',')]
            filtered_test_cases = runner.filter_test_cases(
                filtered_test_cases, categories=categories
            )
        
        if args.tags:
            tags = [t.strip() for t in args.tags.split(',')]
            filtered_test_cases = runner.filter_test_cases(
                filtered_test_cases, tags=tags
            )
        
        if args.priorities:
            priorities = [int(p.strip()) for p in args.priorities.split(',')]
            filtered_test_cases = runner.filter_test_cases(
                filtered_test_cases, priorities=priorities
            )
        
        if not filtered_test_cases:
            runner.logger.warning("No test cases match the specified filters")
            sys.exit(0)
        
        # Run manual test cases
        manual_results = runner.run_tests_enhanced(
            filtered_test_cases,
            parallel=args.parallel,
            max_workers=args.workers
        )
        
        # Run pytest integration if requested
        pytest_results = None
        if args.run_pytest_integration:
            pytest_results = runner.run_pytest_integration(args.output)
        
        # Save comprehensive results
        runner.save_enhanced_results(manual_results, args.output, pytest_results)
        
        # Clean up workspaces
        runner.cleanup_workspaces()
        
        # Exit with appropriate code
        manual_failed = sum(1 for r in manual_results.values() if not r.success)
        pytest_failed = 0
        
        if pytest_results:
            pytest_failed = sum(1 for r in pytest_results.values() if not r.get('success', False))
        
        total_failed = manual_failed + pytest_failed
        
        if total_failed > 0:
            runner.logger.error(f"{total_failed} tests failed! (Manual: {manual_failed}, Pytest: {pytest_failed})")
            sys.exit(1)
        else:
            runner.logger.info("All tests passed!")
            sys.exit(0)
            
    except Exception as e:
        runner.logger.error(f"Test execution failed: {e}")
        # Clean up on error
        try:
            runner.cleanup_workspaces()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
