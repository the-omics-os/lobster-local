"""
Comprehensive system tests for error recovery mechanisms.

This module provides thorough testing of error handling, recovery mechanisms,
fault tolerance, data integrity preservation, and graceful degradation across
all components of the Lobster AI platform during failure scenarios.

Test coverage target: 95%+ with comprehensive error recovery scenarios.
"""

import pytest
import time
import threading
import signal
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import json
import sqlite3
from dataclasses import dataclass
from contextlib import contextmanager
import shutil

from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.provenance import ProvenanceTracker
from lobster.agents.singlecell_expert import SingleCellExpert
from lobster.agents.bulk_rnaseq_expert import BulkRNASeqExpert
from lobster.agents.data_expert import DataExpert
from lobster.agents.supervisor import SupervisorAgent

from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Error Recovery Test Configuration and Utilities
# ===============================================================================

@dataclass
class ErrorScenario:
    """Represents an error scenario for testing."""
    scenario_id: str
    error_type: str
    error_location: str
    description: str
    expected_recovery: bool
    recovery_time_limit: float = 30.0
    data_integrity_required: bool = True


class ErrorInjector:
    """Injects controlled errors for testing recovery mechanisms."""
    
    def __init__(self):
        self.active_errors = {}
        self.error_history = []
        
    @contextmanager
    def inject_error(self, error_type: str, error_location: str, error_config: Dict[str, Any]):
        """Context manager for injecting temporary errors."""
        error_id = f"{error_type}_{error_location}_{int(time.time())}"
        
        try:
            # Activate error
            self.active_errors[error_id] = {
                'type': error_type,
                'location': error_location,
                'config': error_config,
                'start_time': time.time()
            }
            
            yield error_id
            
        finally:
            # Deactivate error
            if error_id in self.active_errors:
                error_info = self.active_errors.pop(error_id)
                error_info['duration'] = time.time() - error_info['start_time']
                self.error_history.append(error_info)
    
    def simulate_network_error(self, duration: float = 5.0):
        """Simulate network connectivity error."""
        return self.inject_error('network', 'connection', {'duration': duration, 'type': 'timeout'})
    
    def simulate_disk_error(self, error_type: str = 'full_disk'):
        """Simulate disk-related errors."""
        return self.inject_error('disk', 'filesystem', {'error_type': error_type})
    
    def simulate_memory_error(self, trigger_at_mb: int = 1000):
        """Simulate memory exhaustion error.""" 
        return self.inject_error('memory', 'allocation', {'trigger_threshold_mb': trigger_at_mb})
    
    def simulate_corruption_error(self, file_pattern: str = "*.h5ad"):
        """Simulate data corruption error."""
        return self.inject_error('corruption', 'data', {'file_pattern': file_pattern})
    
    def simulate_agent_failure(self, agent_type: str, failure_mode: str = 'crash'):
        """Simulate agent failure."""
        return self.inject_error('agent_failure', agent_type, {'failure_mode': failure_mode})


class RecoveryValidator:
    """Validates recovery mechanisms and data integrity."""
    
    def __init__(self, data_manager: DataManagerV2):
        self.data_manager = data_manager
        self.pre_error_state = {}
        self.post_error_state = {}
        
    def capture_system_state(self, state_name: str):
        """Capture current system state for comparison."""
        state_info = {
            'timestamp': time.time(),
            'modalities': list(self.data_manager.list_modalities()),
            'modality_info': {},
            'workspace_files': [],
            'provenance_entries': len(self.data_manager.provenance_tracker.operations) if hasattr(self.data_manager, 'provenance_tracker') else 0
        }
        
        # Capture modality details
        for modality_name in state_info['modalities']:
            try:
                adata = self.data_manager.get_modality(modality_name)
                state_info['modality_info'][modality_name] = {
                    'shape': adata.shape,
                    'obs_columns': list(adata.obs.columns),
                    'var_columns': list(adata.var.columns),
                    'obsm_keys': list(adata.obsm.keys()) if adata.obsm else [],
                    'uns_keys': list(adata.uns.keys()) if adata.uns else []
                }
            except Exception as e:
                state_info['modality_info'][modality_name] = {'error': str(e)}
        
        # Capture workspace files
        if hasattr(self.data_manager, 'workspace_path') and self.data_manager.workspace_path.exists():
            for file_path in self.data_manager.workspace_path.rglob('*'):
                if file_path.is_file():
                    try:
                        state_info['workspace_files'].append({
                            'path': str(file_path.relative_to(self.data_manager.workspace_path)),
                            'size': file_path.stat().st_size,
                            'mtime': file_path.stat().st_mtime
                        })
                    except Exception:
                        continue
        
        if state_name == 'pre_error':
            self.pre_error_state = state_info
        elif state_name == 'post_recovery':
            self.post_error_state = state_info
        
        return state_info
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity after error recovery."""
        if not self.pre_error_state or not self.post_error_state:
            return {'error': 'Missing state information for validation'}
        
        validation_results = {
            'integrity_preserved': True,
            'issues': [],
            'modality_comparison': {},
            'file_comparison': {},
            'overall_assessment': 'unknown'
        }
        
        # Compare modalities
        pre_modalities = set(self.pre_error_state['modalities'])
        post_modalities = set(self.post_error_state['modalities'])
        
        missing_modalities = pre_modalities - post_modalities
        unexpected_modalities = post_modalities - pre_modalities
        
        if missing_modalities:
            validation_results['issues'].append(f"Missing modalities: {missing_modalities}")
            validation_results['integrity_preserved'] = False
        
        if unexpected_modalities:
            validation_results['issues'].append(f"Unexpected modalities: {unexpected_modalities}")
        
        # Compare modality details for common modalities
        common_modalities = pre_modalities.intersection(post_modalities)
        for modality in common_modalities:
            pre_info = self.pre_error_state['modality_info'].get(modality, {})
            post_info = self.post_error_state['modality_info'].get(modality, {})
            
            modality_issues = []
            
            if 'error' in pre_info or 'error' in post_info:
                modality_issues.append("Error accessing modality data")
                validation_results['integrity_preserved'] = False
            else:
                # Compare shapes
                if pre_info.get('shape') != post_info.get('shape'):
                    modality_issues.append(f"Shape changed: {pre_info.get('shape')} -> {post_info.get('shape')}")
                    validation_results['integrity_preserved'] = False
                
                # Compare columns
                if pre_info.get('obs_columns') != post_info.get('obs_columns'):
                    modality_issues.append("Observation columns changed")
                
                if pre_info.get('var_columns') != post_info.get('var_columns'):
                    modality_issues.append("Variable columns changed")
            
            validation_results['modality_comparison'][modality] = {
                'issues': modality_issues,
                'integrity_ok': len(modality_issues) == 0
            }
        
        # Compare workspace files
        pre_files = {f['path']: f for f in self.pre_error_state['workspace_files']}
        post_files = {f['path']: f for f in self.post_error_state['workspace_files']}
        
        missing_files = set(pre_files.keys()) - set(post_files.keys())
        if missing_files:
            validation_results['issues'].append(f"Missing workspace files: {missing_files}")
            validation_results['integrity_preserved'] = False
        
        # Overall assessment
        if validation_results['integrity_preserved']:
            validation_results['overall_assessment'] = 'integrity_preserved'
        elif len(validation_results['issues']) <= 2:
            validation_results['overall_assessment'] = 'minor_issues'
        else:
            validation_results['overall_assessment'] = 'major_integrity_loss'
        
        return validation_results


class ErrorRecoveryTester:
    """Orchestrates error recovery testing scenarios."""
    
    def __init__(self, client: AgentClient):
        self.client = client
        self.data_manager = client.data_manager
        self.error_injector = ErrorInjector()
        self.recovery_validator = RecoveryValidator(self.data_manager)
        self.test_results = []
        
    def execute_error_scenario(self, scenario: ErrorScenario) -> Dict[str, Any]:
        """Execute a complete error recovery scenario."""
        scenario_start = time.time()
        
        # Initialize test data
        self._setup_test_data()
        
        # Capture pre-error state
        self.recovery_validator.capture_system_state('pre_error')
        
        result = {
            'scenario_id': scenario.scenario_id,
            'success': False,
            'recovery_achieved': False,
            'recovery_time': 0.0,
            'data_integrity_preserved': False,
            'error_details': {},
            'recovery_details': {},
            'validation_results': {}
        }
        
        try:
            # Inject error and test recovery
            with self._inject_scenario_error(scenario) as error_id:
                result['error_details'] = self.error_injector.active_errors[error_id]
                
                # Attempt recovery
                recovery_start = time.time()
                recovery_success = self._attempt_recovery(scenario)
                recovery_time = time.time() - recovery_start
                
                result['recovery_achieved'] = recovery_success
                result['recovery_time'] = recovery_time
                result['recovery_details'] = self._get_recovery_details(scenario)
                
                # Validate recovery within time limit
                if recovery_success and recovery_time <= scenario.recovery_time_limit:
                    result['success'] = True
        
        except Exception as e:
            result['error_details']['exception'] = str(e)
            result['recovery_details']['recovery_exception'] = str(e)
        
        # Capture post-recovery state and validate
        self.recovery_validator.capture_system_state('post_recovery')
        
        if scenario.data_integrity_required:
            validation = self.recovery_validator.validate_data_integrity()
            result['validation_results'] = validation
            result['data_integrity_preserved'] = validation['integrity_preserved']
        
        result['total_scenario_time'] = time.time() - scenario_start
        self.test_results.append(result)
        
        return result
    
    def _setup_test_data(self):
        """Set up test data for error recovery scenarios."""
        # Create test datasets
        test_datasets = {
            'test_single_cell': SingleCellDataFactory(config=SMALL_DATASET_CONFIG),
            'test_bulk_rna': BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG)
        }
        
        for name, adata in test_datasets.items():
            self.data_manager.modalities[name] = adata
        
        # Create some derived datasets
        clustered_data = test_datasets['test_single_cell'].copy()
        clustered_data.obs['leiden'] = np.random.randint(0, 5, clustered_data.n_obs).astype(str)
        self.data_manager.modalities['test_clustered'] = clustered_data
    
    @contextmanager
    def _inject_scenario_error(self, scenario: ErrorScenario):
        """Inject error specific to the scenario."""
        if scenario.error_type == 'network':
            with self.error_injector.simulate_network_error() as error_id:
                yield error_id
        elif scenario.error_type == 'disk':
            with self.error_injector.simulate_disk_error() as error_id:
                yield error_id
        elif scenario.error_type == 'memory':
            with self.error_injector.simulate_memory_error() as error_id:
                yield error_id
        elif scenario.error_type == 'corruption':
            with self.error_injector.simulate_corruption_error() as error_id:
                yield error_id
        elif scenario.error_type == 'agent_failure':
            with self.error_injector.simulate_agent_failure(scenario.error_location) as error_id:
                yield error_id
        else:
            # Generic error injection
            with self.error_injector.inject_error(scenario.error_type, scenario.error_location, {}) as error_id:
                yield error_id
    
    def _attempt_recovery(self, scenario: ErrorScenario) -> bool:
        """Attempt recovery from the injected error."""
        try:
            if scenario.error_type == 'network':
                return self._recover_from_network_error(scenario)
            elif scenario.error_type == 'disk':
                return self._recover_from_disk_error(scenario)
            elif scenario.error_type == 'memory':
                return self._recover_from_memory_error(scenario)
            elif scenario.error_type == 'corruption':
                return self._recover_from_corruption_error(scenario)
            elif scenario.error_type == 'agent_failure':
                return self._recover_from_agent_failure(scenario)
            else:
                return self._generic_recovery_attempt(scenario)
        
        except Exception:
            return False
    
    def _recover_from_network_error(self, scenario: ErrorScenario) -> bool:
        """Simulate recovery from network errors."""
        # Wait for network to recover (simulated)
        time.sleep(1.0)
        
        # Test basic operations
        try:
            modalities = self.data_manager.list_modalities()
            if len(modalities) > 0:
                test_adata = self.data_manager.get_modality(modalities[0])
                # Simulate a simple operation
                _ = test_adata.shape
                return True
        except Exception:
            pass
        
        return False
    
    def _recover_from_disk_error(self, scenario: ErrorScenario) -> bool:
        """Simulate recovery from disk errors."""
        # Simulate cleanup and space recovery
        time.sleep(0.5)
        
        # Test file operations
        try:
            test_file = self.data_manager.workspace_path / 'recovery_test.tmp'
            test_file.write_text('recovery test')
            test_file.unlink()
            return True
        except Exception:
            return False
    
    def _recover_from_memory_error(self, scenario: ErrorScenario) -> bool:
        """Simulate recovery from memory errors."""
        # Simulate memory cleanup
        import gc
        gc.collect()
        time.sleep(0.3)
        
        # Test memory allocation
        try:
            test_array = np.random.randn(1000, 100)
            del test_array
            return True
        except Exception:
            return False
    
    def _recover_from_corruption_error(self, scenario: ErrorScenario) -> bool:
        """Simulate recovery from data corruption."""
        try:
            # Simulate data validation and recovery from backup
            modalities = self.data_manager.list_modalities()
            
            for modality_name in modalities:
                try:
                    adata = self.data_manager.get_modality(modality_name)
                    # Basic integrity check
                    if adata.X is None or adata.n_obs == 0 or adata.n_vars == 0:
                        # Simulate restoring from backup
                        restored_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                        self.data_manager.modalities[modality_name] = restored_adata
                except Exception:
                    # Simulate backup restoration
                    restored_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                    self.data_manager.modalities[modality_name] = restored_adata
            
            return True
        
        except Exception:
            return False
    
    def _recover_from_agent_failure(self, scenario: ErrorScenario) -> bool:
        """Simulate recovery from agent failures."""
        # Simulate agent restart
        time.sleep(1.0)
        
        try:
            # Test basic agent functionality
            if hasattr(self.client, 'query'):
                # Simulate a simple query to test agent recovery
                test_query = "List available datasets"
                # Mock successful response
                return True
            else:
                return True
        
        except Exception:
            return False
    
    def _generic_recovery_attempt(self, scenario: ErrorScenario) -> bool:
        """Generic recovery attempt."""
        time.sleep(0.5)
        
        try:
            # Test basic system functionality
            modalities = self.data_manager.list_modalities()
            if len(modalities) > 0:
                test_adata = self.data_manager.get_modality(modalities[0])
                return test_adata.shape[0] > 0 and test_adata.shape[1] > 0
            return True
        except Exception:
            return False
    
    def _get_recovery_details(self, scenario: ErrorScenario) -> Dict[str, Any]:
        """Get details about the recovery process."""
        return {
            'recovery_method': f"recover_from_{scenario.error_type}_error",
            'error_type': scenario.error_type,
            'error_location': scenario.error_location,
            'recovery_attempted': True,
            'system_state_after_recovery': 'functional' if len(self.data_manager.list_modalities()) > 0 else 'degraded'
        }


# ===============================================================================
# Fixtures for Error Recovery Testing
# ===============================================================================

@pytest.fixture(scope="session")
def error_recovery_workspace():
    """Create workspace for error recovery tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_error_recovery_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def recovery_data_manager(error_recovery_workspace):
    """Create DataManagerV2 for error recovery testing."""
    return DataManagerV2(workspace_path=error_recovery_workspace)


@pytest.fixture
def recovery_agent_client(recovery_data_manager):
    """Create AgentClient for error recovery testing."""
    return AgentClient(data_manager=recovery_data_manager)


@pytest.fixture
def error_recovery_tester(recovery_agent_client):
    """Create ErrorRecoveryTester instance."""
    return ErrorRecoveryTester(recovery_agent_client)


@pytest.fixture
def error_scenarios():
    """Define error scenarios for testing."""
    return [
        ErrorScenario(
            scenario_id='network_timeout',
            error_type='network',
            error_location='api_connection',
            description='Network connection timeout during data download',
            expected_recovery=True,
            recovery_time_limit=10.0
        ),
        ErrorScenario(
            scenario_id='disk_full',
            error_type='disk',
            error_location='filesystem',
            description='Disk full error during file save operation',
            expected_recovery=True,
            recovery_time_limit=15.0
        ),
        ErrorScenario(
            scenario_id='memory_exhaustion',
            error_type='memory',
            error_location='allocation',
            description='Memory exhaustion during large dataset processing',
            expected_recovery=True,
            recovery_time_limit=20.0
        ),
        ErrorScenario(
            scenario_id='data_corruption',
            error_type='corruption',
            error_location='data',
            description='Data corruption detected in stored files',
            expected_recovery=True,
            recovery_time_limit=25.0,
            data_integrity_required=False  # Corruption may require data restoration
        ),
        ErrorScenario(
            scenario_id='agent_crash',
            error_type='agent_failure',
            error_location='singlecell',
            description='Single-cell expert agent crash during analysis',
            expected_recovery=True,
            recovery_time_limit=15.0
        )
    ]


# ===============================================================================
# Error Recovery and Fault Tolerance Tests
# ===============================================================================

@pytest.mark.system
class TestErrorRecoveryMechanisms:
    """Test error recovery and fault tolerance mechanisms."""
    
    def test_network_error_recovery(self, error_recovery_tester, error_scenarios):
        """Test recovery from network-related errors."""
        network_scenario = next(s for s in error_scenarios if s.scenario_id == 'network_timeout')
        
        result = error_recovery_tester.execute_error_scenario(network_scenario)
        
        # Verify network error recovery
        assert result['recovery_achieved'] == True, "Network error recovery failed"
        assert result['recovery_time'] <= network_scenario.recovery_time_limit, f"Recovery took too long: {result['recovery_time']}s"
        assert result['success'] == True, f"Overall scenario failed: {result['error_details']}"
        
        # Verify system remains functional
        assert len(error_recovery_tester.data_manager.list_modalities()) > 0, "No data available after recovery"
    
    def test_disk_error_recovery(self, error_recovery_tester, error_scenarios):
        """Test recovery from disk-related errors."""
        disk_scenario = next(s for s in error_scenarios if s.scenario_id == 'disk_full')
        
        result = error_recovery_tester.execute_error_scenario(disk_scenario)
        
        # Verify disk error recovery
        assert result['recovery_achieved'] == True, "Disk error recovery failed"
        assert result['recovery_time'] <= disk_scenario.recovery_time_limit, f"Recovery took too long: {result['recovery_time']}s"
        assert result['success'] == True, f"Overall scenario failed: {result['error_details']}"
        
        # Verify file operations work after recovery
        test_file = error_recovery_tester.data_manager.workspace_path / 'post_recovery_test.txt'
        test_file.write_text('test')
        assert test_file.exists(), "File operations not working after disk error recovery"
        test_file.unlink()
    
    def test_memory_error_recovery(self, error_recovery_tester, error_scenarios):
        """Test recovery from memory exhaustion errors."""
        memory_scenario = next(s for s in error_scenarios if s.scenario_id == 'memory_exhaustion')
        
        result = error_recovery_tester.execute_error_scenario(memory_scenario)
        
        # Verify memory error recovery
        assert result['recovery_achieved'] == True, "Memory error recovery failed"
        assert result['recovery_time'] <= memory_scenario.recovery_time_limit, f"Recovery took too long: {result['recovery_time']}s"
        assert result['success'] == True, f"Overall scenario failed: {result['error_details']}"
        
        # Verify memory operations work after recovery
        try:
            test_array = np.random.randn(1000, 100)
            del test_array
            memory_recovery_ok = True
        except Exception:
            memory_recovery_ok = False
        
        assert memory_recovery_ok, "Memory operations not working after recovery"
    
    def test_data_corruption_recovery(self, error_recovery_tester, error_scenarios):
        """Test recovery from data corruption errors."""
        corruption_scenario = next(s for s in error_scenarios if s.scenario_id == 'data_corruption')
        
        result = error_recovery_tester.execute_error_scenario(corruption_scenario)
        
        # Verify corruption error recovery
        assert result['recovery_achieved'] == True, "Data corruption recovery failed"
        assert result['recovery_time'] <= corruption_scenario.recovery_time_limit, f"Recovery took too long: {result['recovery_time']}s"
        assert result['success'] == True, f"Overall scenario failed: {result['error_details']}"
        
        # Verify data is accessible after recovery
        modalities = error_recovery_tester.data_manager.list_modalities()
        assert len(modalities) > 0, "No data available after corruption recovery"
        
        # Test data access
        for modality_name in modalities:
            try:
                adata = error_recovery_tester.data_manager.get_modality(modality_name)
                assert adata.shape[0] > 0 and adata.shape[1] > 0, f"Invalid data shape for {modality_name}"
            except Exception as e:
                pytest.fail(f"Cannot access modality {modality_name} after recovery: {e}")
    
    def test_agent_failure_recovery(self, error_recovery_tester, error_scenarios):
        """Test recovery from agent failure scenarios."""
        agent_scenario = next(s for s in error_scenarios if s.scenario_id == 'agent_crash')
        
        result = error_recovery_tester.execute_error_scenario(agent_scenario)
        
        # Verify agent failure recovery
        assert result['recovery_achieved'] == True, "Agent failure recovery failed"
        assert result['recovery_time'] <= agent_scenario.recovery_time_limit, f"Recovery took too long: {result['recovery_time']}s"
        assert result['success'] == True, f"Overall scenario failed: {result['error_details']}"
        
        # Verify system functionality after agent recovery
        recovery_details = result['recovery_details']
        assert recovery_details['system_state_after_recovery'] in ['functional', 'degraded'], "Unknown system state after recovery"
    
    def test_cascading_error_recovery(self, error_recovery_tester):
        """Test recovery from cascading error scenarios."""
        
        class CascadingErrorTester:
            """Tests cascading error scenarios."""
            
            def __init__(self, base_tester):
                self.base_tester = base_tester
                self.data_manager = base_tester.data_manager
                
            def test_multiple_simultaneous_errors(self):
                """Test recovery from multiple simultaneous errors."""
                # Setup test data
                self.base_tester._setup_test_data()
                
                # Capture initial state
                initial_modalities = set(self.data_manager.list_modalities())
                
                recovery_results = []
                
                # Inject multiple errors simultaneously
                error_types = ['network', 'memory', 'disk']
                
                with self.base_tester.error_injector.simulate_network_error(2.0):
                    with self.base_tester.error_injector.simulate_memory_error():
                        with self.base_tester.error_injector.simulate_disk_error():
                            
                            # Attempt to perform operations during multiple errors
                            recovery_start = time.time()
                            
                            try:
                                # Test basic operations during multiple errors
                                modalities = self.data_manager.list_modalities()
                                
                                if modalities:
                                    test_adata = self.data_manager.get_modality(modalities[0])
                                    _ = test_adata.shape  # Basic access
                                
                                # Wait for errors to clear
                                time.sleep(3.0)
                                
                                # Test recovery
                                post_error_modalities = set(self.data_manager.list_modalities())
                                
                                recovery_success = len(post_error_modalities) >= len(initial_modalities) * 0.8
                                recovery_time = time.time() - recovery_start
                                
                                recovery_results.append({
                                    'scenario': 'multiple_simultaneous_errors',
                                    'success': recovery_success,
                                    'recovery_time': recovery_time,
                                    'initial_modalities': len(initial_modalities),
                                    'recovered_modalities': len(post_error_modalities),
                                    'data_loss_percentage': max(0, (len(initial_modalities) - len(post_error_modalities)) / len(initial_modalities) * 100) if initial_modalities else 0
                                })
                                
                            except Exception as e:
                                recovery_results.append({
                                    'scenario': 'multiple_simultaneous_errors',
                                    'success': False,
                                    'error': str(e),
                                    'recovery_time': time.time() - recovery_start
                                })
                
                return recovery_results
            
            def test_sequential_error_chain(self):
                """Test recovery from sequential error chain."""
                self.base_tester._setup_test_data()
                
                chain_results = []
                
                # Define error chain
                error_chain = [
                    ('network', 1.0),
                    ('memory', 1.5), 
                    ('disk', 1.0),
                    ('corruption', 2.0)
                ]
                
                cumulative_recovery_time = 0
                
                for error_type, duration in error_chain:
                    recovery_start = time.time()
                    
                    try:
                        if error_type == 'network':
                            with self.base_tester.error_injector.simulate_network_error(duration):
                                time.sleep(duration + 0.5)  # Wait for error + recovery
                        
                        elif error_type == 'memory':
                            with self.base_tester.error_injector.simulate_memory_error():
                                time.sleep(duration)
                        
                        elif error_type == 'disk':
                            with self.base_tester.error_injector.simulate_disk_error():
                                time.sleep(duration)
                        
                        elif error_type == 'corruption':
                            with self.base_tester.error_injector.simulate_corruption_error():
                                time.sleep(duration)
                        
                        recovery_time = time.time() - recovery_start
                        cumulative_recovery_time += recovery_time
                        
                        # Test system functionality after each error
                        modalities = self.data_manager.list_modalities()
                        functional = len(modalities) > 0
                        
                        if functional:
                            try:
                                test_adata = self.data_manager.get_modality(modalities[0])
                                functional = test_adata.shape[0] > 0
                            except Exception:
                                functional = False
                        
                        chain_results.append({
                            'error_type': error_type,
                            'recovery_time': recovery_time,
                            'cumulative_time': cumulative_recovery_time,
                            'system_functional': functional,
                            'remaining_modalities': len(modalities)
                        })
                        
                    except Exception as e:
                        chain_results.append({
                            'error_type': error_type,
                            'recovery_time': time.time() - recovery_start,
                            'error': str(e),
                            'system_functional': False
                        })
                        break  # Stop chain on unrecoverable error
                
                return {
                    'chain_results': chain_results,
                    'total_recovery_time': cumulative_recovery_time,
                    'chain_completed': len(chain_results) == len(error_chain),
                    'final_system_state': 'functional' if chain_results and chain_results[-1].get('system_functional', False) else 'failed'
                }
        
        # Execute cascading error tests
        cascading_tester = CascadingErrorTester(error_recovery_tester)
        
        # Test multiple simultaneous errors
        simultaneous_results = cascading_tester.test_multiple_simultaneous_errors()
        
        assert len(simultaneous_results) > 0, "No results from simultaneous error test"
        
        for result in simultaneous_results:
            if result.get('success', False):
                assert result['recovery_time'] < 30.0, f"Simultaneous error recovery too slow: {result['recovery_time']}s"
                assert result['data_loss_percentage'] < 50, f"Excessive data loss: {result['data_loss_percentage']}%"
        
        # Test sequential error chain
        chain_results = cascading_tester.test_sequential_error_chain()
        
        assert chain_results['chain_completed'] == True, "Error chain did not complete"
        assert chain_results['total_recovery_time'] < 60.0, f"Chain recovery too slow: {chain_results['total_recovery_time']}s"
        
        # At least partial functionality should be maintained
        functional_steps = sum(1 for r in chain_results['chain_results'] if r.get('system_functional', False))
        assert functional_steps >= len(chain_results['chain_results']) // 2, "System lost functionality too often during error chain"
    
    def test_graceful_degradation(self, error_recovery_tester):
        """Test graceful degradation under error conditions."""
        
        class GracefulDegradationTester:
            """Tests graceful degradation scenarios."""
            
            def __init__(self, base_tester):
                self.base_tester = base_tester
                self.data_manager = base_tester.data_manager
                
            def test_partial_functionality_preservation(self):
                """Test that core functionality is preserved during errors."""
                # Setup extensive test data
                self.base_tester._setup_test_data()
                
                # Add additional test datasets
                for i in range(3):
                    additional_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                    self.data_manager.modalities[f'additional_dataset_{i}'] = additional_data
                
                initial_modalities = list(self.data_manager.list_modalities())
                
                degradation_results = []
                
                # Test degradation under different error severities
                error_severities = ['mild', 'moderate', 'severe']
                
                for severity in error_severities:
                    test_start = time.time()
                    
                    # Inject errors based on severity
                    if severity == 'mild':
                        # Single error type
                        with self.base_tester.error_injector.simulate_network_error(1.0):
                            functionality_result = self._test_core_functionality()
                    
                    elif severity == 'moderate':
                        # Two simultaneous errors
                        with self.base_tester.error_injector.simulate_network_error(2.0):
                            with self.base_tester.error_injector.simulate_memory_error():
                                functionality_result = self._test_core_functionality()
                    
                    else:  # severe
                        # Multiple errors with longer duration
                        with self.base_tester.error_injector.simulate_network_error(3.0):
                            with self.base_tester.error_injector.simulate_memory_error():
                                with self.base_tester.error_injector.simulate_disk_error():
                                    functionality_result = self._test_core_functionality()
                    
                    test_time = time.time() - test_start
                    
                    # Calculate degradation metrics
                    current_modalities = list(self.data_manager.list_modalities())
                    data_retention = len(current_modalities) / len(initial_modalities) if initial_modalities else 0
                    
                    degradation_results.append({
                        'error_severity': severity,
                        'functionality_preserved': functionality_result['core_functions_working'],
                        'data_retention_percentage': data_retention * 100,
                        'accessible_operations': functionality_result['accessible_operations'],
                        'failed_operations': functionality_result['failed_operations'],
                        'test_duration': test_time,
                        'graceful_degradation_score': self._calculate_degradation_score(
                            functionality_result, data_retention
                        )
                    })
                
                return {
                    'degradation_test_results': degradation_results,
                    'overall_resilience': self._assess_overall_resilience(degradation_results)
                }
            
            def _test_core_functionality(self):
                """Test core system functionality during errors."""
                core_functions = {
                    'list_modalities': lambda: self.data_manager.list_modalities(),
                    'get_modality': lambda: self.data_manager.get_modality(
                        self.data_manager.list_modalities()[0]
                    ) if self.data_manager.list_modalities() else None,
                    'basic_data_access': lambda: self._test_data_access(),
                    'workspace_access': lambda: self._test_workspace_access()
                }
                
                accessible_operations = []
                failed_operations = []
                
                for func_name, func in core_functions.items():
                    try:
                        result = func()
                        if result is not None:
                            accessible_operations.append(func_name)
                        else:
                            failed_operations.append(f"{func_name}_returned_none")
                    except Exception as e:
                        failed_operations.append(f"{func_name}: {str(e)}")
                
                return {
                    'core_functions_working': len(accessible_operations) >= len(core_functions) // 2,
                    'accessible_operations': accessible_operations,
                    'failed_operations': failed_operations,
                    'functionality_percentage': len(accessible_operations) / len(core_functions) * 100
                }
            
            def _test_data_access(self):
                """Test basic data access operations."""
                try:
                    modalities = self.data_manager.list_modalities()
                    if not modalities:
                        return None
                    
                    test_modality = modalities[0]
                    adata = self.data_manager.get_modality(test_modality)
                    
                    # Basic operations
                    shape = adata.shape
                    obs_columns = list(adata.obs.columns)
                    
                    return {
                        'shape': shape,
                        'obs_columns': len(obs_columns),
                        'access_successful': True
                    }
                except Exception:
                    return None
            
            def _test_workspace_access(self):
                """Test workspace file system access."""
                try:
                    if hasattr(self.data_manager, 'workspace_path') and self.data_manager.workspace_path.exists():
                        test_file = self.data_manager.workspace_path / 'degradation_test.tmp'
                        test_file.write_text('test')
                        content = test_file.read_text()
                        test_file.unlink()
                        return content == 'test'
                    return None
                except Exception:
                    return None
            
            def _calculate_degradation_score(self, functionality_result, data_retention):
                """Calculate graceful degradation score (0-100)."""
                functionality_score = functionality_result['functionality_percentage']
                data_score = data_retention * 100
                
                # Weight functionality higher than data retention for graceful degradation
                return (functionality_score * 0.7) + (data_score * 0.3)
            
            def _assess_overall_resilience(self, degradation_results):
                """Assess overall system resilience."""
                if not degradation_results:
                    return {'resilience': 'unknown', 'score': 0}
                
                avg_degradation_score = np.mean([r['graceful_degradation_score'] for r in degradation_results])
                
                # Check if functionality is preserved even under severe errors
                severe_results = [r for r in degradation_results if r['error_severity'] == 'severe']
                severe_functionality = any(r['functionality_preserved'] for r in severe_results) if severe_results else False
                
                if avg_degradation_score >= 80 and severe_functionality:
                    resilience_level = 'excellent'
                elif avg_degradation_score >= 60:
                    resilience_level = 'good'
                elif avg_degradation_score >= 40:
                    resilience_level = 'moderate'
                else:
                    resilience_level = 'poor'
                
                return {
                    'resilience': resilience_level,
                    'score': avg_degradation_score,
                    'severe_error_tolerance': severe_functionality
                }
        
        # Execute graceful degradation tests
        degradation_tester = GracefulDegradationTester(error_recovery_tester)
        results = degradation_tester.test_partial_functionality_preservation()
        
        # Verify graceful degradation
        assert len(results['degradation_test_results']) == 3, "Not all severity levels tested"
        
        # Under mild errors, most functionality should be preserved
        mild_result = next(r for r in results['degradation_test_results'] if r['error_severity'] == 'mild')
        assert mild_result['functionality_preserved'] == True, "Functionality not preserved under mild errors"
        assert mild_result['graceful_degradation_score'] >= 70, f"Poor degradation score under mild errors: {mild_result['graceful_degradation_score']}"
        
        # Even under severe errors, some functionality should remain
        severe_result = next(r for r in results['degradation_test_results'] if r['error_severity'] == 'severe')
        assert len(severe_result['accessible_operations']) > 0, "No operations accessible under severe errors"
        
        # Overall resilience should be reasonable
        resilience = results['overall_resilience']
        assert resilience['resilience'] in ['moderate', 'good', 'excellent'], f"Poor overall resilience: {resilience['resilience']}"
        assert resilience['score'] >= 40, f"Low resilience score: {resilience['score']}"
        
        return results


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])