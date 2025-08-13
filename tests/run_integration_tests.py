#!/usr/bin/env python3
"""
Integration Testing Script for Lobster AI
=========================================

This script runs comprehensive tests against the Lobster AI multi-agent system
by reading test cases from JSON files and executing them sequentially or in parallel.

Usage:
    python tests/run_integration_tests.py --input tests/test_cases.json --output tests/results.json
    python tests/run_integration_tests.py --input tests/test_cases.json --parallel --workers 4
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import tempfile
import os
from dotenv import load_dotenv

# Add the project root to the path so we can import lobster modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from lobster.core.client import AgentClient
from lobster.core.data_manager import DataManager


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


class LobsterTestRunner:
    """Main test runner for Lobster AI integration tests."""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = TestLogger(log_level)
        self.results = {}
        
    def load_test_cases(self, input_file: Path) -> Dict[str, List[str]]:
        """Load test cases from JSON file."""
        try:
            with open(input_file, 'r') as f:
                test_cases = json.load(f)
            
            self.logger.info(f"Loaded {len(test_cases)} test cases from {input_file}")
            return test_cases
        
        except Exception as e:
            self.logger.error(f"Failed to load test cases from {input_file}: {e}")
            raise
    
    def run_single_test(self, test_id: str, user_inputs: List[str], workspace_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run a single test case with sequential user inputs.
        
        Args:
            test_id: Unique identifier for the test
            user_inputs: List of sequential user inputs
            workspace_dir: Directory for test workspace (creates temp if None)
            
        Returns:
            Dictionary containing test results
        """
        start_time = datetime.now()
        
        # Create temporary workspace if not provided
        if workspace_dir is None:
            workspace_dir = Path(tempfile.mkdtemp(prefix=f"lobster_test_{test_id}_"))
        
        try:
            self.logger.info(f"Starting test '{test_id}' with {len(user_inputs)} inputs")
            
            # Initialize client with isolated workspace
            client = AgentClient(
                workspace_path=workspace_dir,
                session_id=f"test_{test_id}_{int(datetime.now().timestamp())}",
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
            
            # Execute sequential inputs
            for i, user_input in enumerate(user_inputs):
                try:
                    self.logger.info(f"Test '{test_id}' - Input {i+1}/{len(user_inputs)}: {user_input[:100]}...")
                    
                    # Run query
                    result = client.query(user_input, stream=False)
                    
                    if result.get("success", False):
                        response_text = result.get("response", "No response generated")
                        responses.append(response_text)
                        self.logger.info(f"Test '{test_id}' - Response {i+1}: SUCCESS")
                    else:
                        error_msg = result.get("error", "Unknown error occurred")
                        responses.append(f"ERROR: {error_msg}")
                        success = False
                        error_message = error_msg
                        self.logger.error(f"Test '{test_id}' - Response {i+1}: FAILED - {error_msg}")
                        break  # Stop on first error
                        
                except Exception as e:
                    error_msg = f"Exception during query execution: {str(e)}"
                    responses.append(f"ERROR: {error_msg}")
                    success = False
                    error_message = error_msg
                    self.logger.error(f"Test '{test_id}' - Response {i+1}: EXCEPTION - {error_msg}")
                    break  # Stop on first exception
            
            # Collect final status
            final_status = client.get_status()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "test_id": test_id,
                "success": success,
                "error_message": error_message,
                "inputs": user_inputs,
                "responses": responses,
                "duration_seconds": duration,
                "final_status": final_status,
                "workspace_path": str(workspace_dir),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Test setup/execution failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(f"Test '{test_id}' - FATAL ERROR: {error_msg}")
            
            return {
                "test_id": test_id,
                "success": False,
                "error_message": error_msg,
                "inputs": user_inputs,
                "responses": [],
                "duration_seconds": duration,
                "final_status": None,
                "workspace_path": str(workspace_dir) if workspace_dir else None,
                "timestamp": datetime.now().isoformat()
            }
    
    def run_tests_sequential(self, test_cases: Dict[str, List[str]]) -> Dict[str, Any]:
        """Run all tests sequentially."""
        self.logger.info(f"Running {len(test_cases)} tests sequentially")
        
        results = {}
        for test_id, user_inputs in test_cases.items():
            result = self.run_single_test(test_id, user_inputs)
            results[test_id] = result
            
            # Log progress
            status = "PASSED" if result["success"] else "FAILED"
            self.logger.info(f"Test '{test_id}' completed: {status} ({result['duration_seconds']:.2f}s)")
        
        return results
    
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


def run_single_test_worker(test_id: str, user_inputs: List[str], workspace_dir: Path) -> Dict[str, Any]:
    """Worker function for parallel test execution."""
    # This function runs in a separate process, so we need to recreate the test runner
    runner = LobsterTestRunner()
    return runner.run_single_test(test_id, user_inputs, workspace_dir)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run integration tests for Lobster AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tests sequentially
  python tests/run_integration_tests.py --input tests/test_cases.json --output tests/results.json

  # Run tests in parallel with 4 workers  
  python tests/run_integration_tests.py --input tests/test_cases.json --output tests/results.json --parallel --workers 4

  # Run with debug logging
  python tests/run_integration_tests.py --input tests/test_cases.json --output tests/results.json --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("tests/test_cases.json"),
        help="Input JSON file containing test cases (default: tests/test_cases.json)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=Path,
        default=Path("tests/results.json"),
        help="Output JSON file for test results (default: tests/results.json)"
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
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = LobsterTestRunner(log_level=args.log_level)
    
    try:
        # Load test cases
        test_cases = runner.load_test_cases(args.input)
        
        # Run tests
        if args.parallel:
            results = runner.run_tests_parallel(test_cases, args.workers)
        else:
            results = runner.run_tests_sequential(test_cases)
        
        # Save results
        runner.save_results(results, args.output)
        
        # Exit with appropriate code
        failed_count = sum(1 for r in results.values() if not r["success"])
        if failed_count > 0:
            runner.logger.error(f"{failed_count} tests failed!")
            sys.exit(1)
        else:
            runner.logger.info("All tests passed!")
            sys.exit(0)
            
    except Exception as e:
        runner.logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
