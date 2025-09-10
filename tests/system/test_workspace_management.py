"""
Comprehensive system tests for workspace management operations.

This module provides thorough system-level testing of workspace initialization,
cleanup, backup/restore functionality, migration, file system operations,
and workspace integrity across different storage backends and configurations.

Test coverage target: 95%+ with realistic workspace management scenarios.
"""

import pytest
import time
import shutil
import json
import os
import stat
import threading
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import zipfile
import tarfile
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.client import AgentClient
from lobster.core.provenance import ProvenanceTracker
from lobster.core.backends.h5ad_backend import H5ADBackend

from tests.mock_data.factories import (
    SingleCellDataFactory,
    BulkRNASeqDataFactory,
    ProteomicsDataFactory
)
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Workspace Management Test Configuration and Utilities
# ===============================================================================

@dataclass
class WorkspaceState:
    """Represents the state of a workspace for comparison."""
    workspace_path: Path
    modalities: Dict[str, Dict[str, Any]]
    metadata_files: List[str]
    export_files: List[str]
    cache_files: List[str]
    directory_structure: Dict[str, Any]
    total_size_mb: float
    file_count: int
    created_time: datetime
    modified_time: datetime


class WorkspaceAnalyzer:
    """Analyzes workspace structure and contents."""
    
    def __init__(self):
        self.analysis_history = []
        
    def analyze_workspace(self, workspace_path: Path) -> WorkspaceState:
        """Perform comprehensive workspace analysis."""
        if not workspace_path.exists():
            raise FileNotFoundError(f"Workspace not found: {workspace_path}")
        
        # Analyze modalities
        modalities = {}
        data_path = workspace_path / "data"
        if data_path.exists():
            for modality_file in data_path.glob("*.h5ad"):
                try:
                    adata = ad.read_h5ad(modality_file)
                    modalities[modality_file.stem] = {
                        'shape': adata.shape,
                        'file_size_mb': modality_file.stat().st_size / (1024**2),
                        'obs_columns': list(adata.obs.columns),
                        'var_columns': list(adata.var.columns),
                        'obsm_keys': list(adata.obsm.keys()),
                        'uns_keys': list(adata.uns.keys())
                    }
                except Exception as e:
                    modalities[modality_file.stem] = {'error': str(e)}
        
        # Collect file lists
        metadata_files = self._collect_files(workspace_path, pattern="*.json")
        export_files = self._collect_files(workspace_path / "exports", pattern="*") if (workspace_path / "exports").exists() else []
        cache_files = self._collect_files(workspace_path / "cache", pattern="*") if (workspace_path / "cache").exists() else []
        
        # Directory structure
        directory_structure = self._analyze_directory_structure(workspace_path)
        
        # Size and file count
        total_size_mb, file_count = self._calculate_workspace_size(workspace_path)
        
        # Timestamps
        stat_info = workspace_path.stat()
        
        workspace_state = WorkspaceState(
            workspace_path=workspace_path,
            modalities=modalities,
            metadata_files=metadata_files,
            export_files=export_files,
            cache_files=cache_files,
            directory_structure=directory_structure,
            total_size_mb=total_size_mb,
            file_count=file_count,
            created_time=datetime.fromtimestamp(stat_info.st_ctime),
            modified_time=datetime.fromtimestamp(stat_info.st_mtime)
        )
        
        self.analysis_history.append(workspace_state)
        return workspace_state
    
    def _collect_files(self, path: Path, pattern: str = "*") -> List[str]:
        """Collect files matching pattern."""
        if not path.exists():
            return []
        
        files = []
        for file_path in path.rglob(pattern):
            if file_path.is_file():
                try:
                    relative_path = file_path.relative_to(path)
                    files.append(str(relative_path))
                except ValueError:
                    continue
        
        return sorted(files)
    
    def _analyze_directory_structure(self, workspace_path: Path) -> Dict[str, Any]:
        """Analyze directory structure."""
        structure = {}
        
        try:
            for item in workspace_path.iterdir():
                if item.is_dir():
                    structure[item.name] = {
                        'type': 'directory',
                        'file_count': len(list(item.rglob('*'))),
                        'size_mb': sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024**2)
                    }
                elif item.is_file():
                    structure[item.name] = {
                        'type': 'file',
                        'size_mb': item.stat().st_size / (1024**2),
                        'extension': item.suffix
                    }
        except PermissionError:
            structure['_error'] = 'permission_denied'
        
        return structure
    
    def _calculate_workspace_size(self, workspace_path: Path) -> Tuple[float, int]:
        """Calculate total workspace size and file count."""
        total_size = 0
        file_count = 0
        
        try:
            for item in workspace_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
        except PermissionError:
            pass
        
        return total_size / (1024**2), file_count


class WorkspaceManager:
    """Manages workspace operations including backup, restore, and migration."""
    
    def __init__(self, base_workspace_path: Optional[Path] = None):
        self.base_workspace_path = base_workspace_path
        self.operation_history = []
        self.analyzer = WorkspaceAnalyzer()
        
    def create_workspace(self, workspace_path: Path, initialize_structure: bool = True) -> Dict[str, Any]:
        """Create a new workspace with optional structure initialization."""
        operation_start = time.time()
        
        try:
            # Create main directory
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            if initialize_structure:
                # Create standard subdirectories
                (workspace_path / "data").mkdir(exist_ok=True)
                (workspace_path / "exports").mkdir(exist_ok=True)
                (workspace_path / "cache").mkdir(exist_ok=True)
                
                # Create metadata file
                metadata = {
                    'workspace_id': workspace_path.name,
                    'created_time': datetime.now().isoformat(),
                    'version': '2.0',
                    'structure_version': '1.0',
                    'description': 'Lobster AI analysis workspace'
                }
                
                metadata_file = workspace_path / "workspace_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Create empty provenance file
                provenance_file = workspace_path / "provenance.json"
                with open(provenance_file, 'w') as f:
                    json.dump({'operations': [], 'created': datetime.now().isoformat()}, f)
            
            operation_time = time.time() - operation_start
            
            result = {
                'success': True,
                'workspace_path': str(workspace_path),
                'initialized_structure': initialize_structure,
                'operation_time': operation_time,
                'initial_state': self.analyzer.analyze_workspace(workspace_path)
            }
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'operation_time': time.time() - operation_start
            }
        
        self.operation_history.append({
            'operation': 'create_workspace',
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
        
        return result
    
    def backup_workspace(self, workspace_path: Path, backup_path: Path, 
                        compression: str = 'zip', include_cache: bool = False) -> Dict[str, Any]:
        """Create workspace backup with various compression options."""
        operation_start = time.time()
        
        try:
            # Analyze workspace before backup
            pre_backup_state = self.analyzer.analyze_workspace(workspace_path)
            
            # Prepare backup directory
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            if compression == 'zip':
                backup_file = backup_path.with_suffix('.zip')
                success = self._create_zip_backup(workspace_path, backup_file, include_cache)
            elif compression == 'tar':
                backup_file = backup_path.with_suffix('.tar.gz')
                success = self._create_tar_backup(workspace_path, backup_file, include_cache)
            else:
                backup_file = backup_path
                success = self._create_directory_backup(workspace_path, backup_file, include_cache)
            
            operation_time = time.time() - operation_start
            
            if success:
                backup_size_mb = backup_file.stat().st_size / (1024**2) if backup_file.is_file() else 0
                
                result = {
                    'success': True,
                    'backup_path': str(backup_file),
                    'backup_size_mb': backup_size_mb,
                    'compression': compression,
                    'include_cache': include_cache,
                    'operation_time': operation_time,
                    'compression_ratio': backup_size_mb / pre_backup_state.total_size_mb if pre_backup_state.total_size_mb > 0 else 0,
                    'original_workspace_state': pre_backup_state
                }
            else:
                result = {
                    'success': False,
                    'error': 'backup_creation_failed',
                    'operation_time': operation_time
                }
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'operation_time': time.time() - operation_start
            }
        
        self.operation_history.append({
            'operation': 'backup_workspace',
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
        
        return result
    
    def restore_workspace(self, backup_path: Path, restore_path: Path, 
                         overwrite_existing: bool = False) -> Dict[str, Any]:
        """Restore workspace from backup."""
        operation_start = time.time()
        
        try:
            # Check if restore path exists
            if restore_path.exists() and not overwrite_existing:
                return {
                    'success': False,
                    'error': 'restore_path_exists_and_overwrite_false',
                    'operation_time': time.time() - operation_start
                }
            
            # Remove existing if overwriting
            if restore_path.exists() and overwrite_existing:
                shutil.rmtree(restore_path)
            
            # Determine backup type and restore
            if backup_path.suffix == '.zip':
                success = self._restore_from_zip(backup_path, restore_path)
            elif backup_path.suffix in ['.tar', '.gz'] or backup_path.name.endswith('.tar.gz'):
                success = self._restore_from_tar(backup_path, restore_path)
            else:
                success = self._restore_from_directory(backup_path, restore_path)
            
            operation_time = time.time() - operation_start
            
            if success and restore_path.exists():
                restored_state = self.analyzer.analyze_workspace(restore_path)
                
                result = {
                    'success': True,
                    'restored_path': str(restore_path),
                    'backup_source': str(backup_path),
                    'operation_time': operation_time,
                    'restored_workspace_state': restored_state
                }
            else:
                result = {
                    'success': False,
                    'error': 'restoration_failed',
                    'operation_time': operation_time
                }
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'operation_time': time.time() - operation_start
            }
        
        self.operation_history.append({
            'operation': 'restore_workspace',
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
        
        return result
    
    def migrate_workspace(self, source_path: Path, target_path: Path, 
                         migration_type: str = 'copy') -> Dict[str, Any]:
        """Migrate workspace between locations."""
        operation_start = time.time()
        
        try:
            # Analyze source workspace
            source_state = self.analyzer.analyze_workspace(source_path)
            
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            if migration_type == 'copy':
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                migration_success = True
            elif migration_type == 'move':
                shutil.move(str(source_path), str(target_path))
                migration_success = True
            else:
                migration_success = False
            
            operation_time = time.time() - operation_start
            
            if migration_success and target_path.exists():
                target_state = self.analyzer.analyze_workspace(target_path)
                
                # Update workspace metadata
                self._update_workspace_metadata(target_path, {
                    'migrated_from': str(source_path),
                    'migration_time': datetime.now().isoformat(),
                    'migration_type': migration_type
                })
                
                result = {
                    'success': True,
                    'source_path': str(source_path),
                    'target_path': str(target_path),
                    'migration_type': migration_type,
                    'operation_time': operation_time,
                    'source_state': source_state,
                    'target_state': target_state,
                    'data_integrity_preserved': self._verify_migration_integrity(source_state, target_state)
                }
            else:
                result = {
                    'success': False,
                    'error': 'migration_failed',
                    'operation_time': operation_time
                }
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'operation_time': time.time() - operation_start
            }
        
        self.operation_history.append({
            'operation': 'migrate_workspace',
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
        
        return result
    
    def cleanup_workspace(self, workspace_path: Path, cleanup_options: Dict[str, bool]) -> Dict[str, Any]:
        """Clean up workspace based on specified options."""
        operation_start = time.time()
        
        try:
            # Analyze workspace before cleanup
            pre_cleanup_state = self.analyzer.analyze_workspace(workspace_path)
            
            cleanup_summary = {
                'files_removed': [],
                'directories_removed': [],
                'space_freed_mb': 0,
                'errors': []
            }
            
            # Clean cache if requested
            if cleanup_options.get('clear_cache', False):
                cache_path = workspace_path / "cache"
                if cache_path.exists():
                    try:
                        cache_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                        shutil.rmtree(cache_path)
                        cache_path.mkdir(exist_ok=True)
                        cleanup_summary['directories_removed'].append('cache')
                        cleanup_summary['space_freed_mb'] += cache_size / (1024**2)
                    except Exception as e:
                        cleanup_summary['errors'].append(f"Cache cleanup error: {str(e)}")
            
            # Clean temporary files if requested
            if cleanup_options.get('remove_temp_files', False):
                temp_patterns = ['*.tmp', '*.temp', '*~', '.DS_Store']
                for pattern in temp_patterns:
                    for temp_file in workspace_path.rglob(pattern):
                        try:
                            file_size = temp_file.stat().st_size
                            temp_file.unlink()
                            cleanup_summary['files_removed'].append(str(temp_file.relative_to(workspace_path)))
                            cleanup_summary['space_freed_mb'] += file_size / (1024**2)
                        except Exception as e:
                            cleanup_summary['errors'].append(f"Temp file cleanup error: {str(e)}")
            
            # Clean old exports if requested
            if cleanup_options.get('remove_old_exports', False):
                exports_path = workspace_path / "exports"
                if exports_path.exists():
                    cutoff_time = datetime.now() - timedelta(days=cleanup_options.get('export_age_days', 30))
                    for export_file in exports_path.rglob('*'):
                        if export_file.is_file():
                            try:
                                file_mtime = datetime.fromtimestamp(export_file.stat().st_mtime)
                                if file_mtime < cutoff_time:
                                    file_size = export_file.stat().st_size
                                    export_file.unlink()
                                    cleanup_summary['files_removed'].append(str(export_file.relative_to(workspace_path)))
                                    cleanup_summary['space_freed_mb'] += file_size / (1024**2)
                            except Exception as e:
                                cleanup_summary['errors'].append(f"Export cleanup error: {str(e)}")
            
            # Optimize data storage if requested
            if cleanup_options.get('optimize_storage', False):
                data_path = workspace_path / "data"
                if data_path.exists():
                    optimization_result = self._optimize_data_storage(data_path)
                    cleanup_summary['space_freed_mb'] += optimization_result.get('space_saved_mb', 0)
                    cleanup_summary['optimization_details'] = optimization_result
            
            operation_time = time.time() - operation_start
            
            # Analyze workspace after cleanup
            post_cleanup_state = self.analyzer.analyze_workspace(workspace_path)
            
            result = {
                'success': True,
                'cleanup_summary': cleanup_summary,
                'operation_time': operation_time,
                'pre_cleanup_state': pre_cleanup_state,
                'post_cleanup_state': post_cleanup_state,
                'space_reduction_percentage': (
                    (pre_cleanup_state.total_size_mb - post_cleanup_state.total_size_mb) / 
                    pre_cleanup_state.total_size_mb * 100
                ) if pre_cleanup_state.total_size_mb > 0 else 0
            }
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'operation_time': time.time() - operation_start
            }
        
        self.operation_history.append({
            'operation': 'cleanup_workspace',
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
        
        return result
    
    def _create_zip_backup(self, workspace_path: Path, backup_file: Path, include_cache: bool) -> bool:
        """Create ZIP backup of workspace."""
        try:
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in workspace_path.rglob('*'):
                    if file_path.is_file():
                        # Skip cache files if not included
                        if not include_cache and 'cache' in file_path.parts:
                            continue
                        
                        relative_path = file_path.relative_to(workspace_path)
                        zip_file.write(file_path, relative_path)
            return True
        except Exception:
            return False
    
    def _create_tar_backup(self, workspace_path: Path, backup_file: Path, include_cache: bool) -> bool:
        """Create TAR backup of workspace."""
        try:
            with tarfile.open(backup_file, 'w:gz') as tar_file:
                for file_path in workspace_path.rglob('*'):
                    if file_path.is_file():
                        # Skip cache files if not included
                        if not include_cache and 'cache' in file_path.parts:
                            continue
                        
                        relative_path = file_path.relative_to(workspace_path)
                        tar_file.add(file_path, arcname=relative_path)
            return True
        except Exception:
            return False
    
    def _create_directory_backup(self, workspace_path: Path, backup_path: Path, include_cache: bool) -> bool:
        """Create directory copy backup of workspace."""
        try:
            if include_cache:
                shutil.copytree(workspace_path, backup_path, dirs_exist_ok=True)
            else:
                # Copy selectively without cache
                backup_path.mkdir(parents=True, exist_ok=True)
                for item in workspace_path.iterdir():
                    if item.name != 'cache':
                        if item.is_dir():
                            shutil.copytree(item, backup_path / item.name, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, backup_path / item.name)
            return True
        except Exception:
            return False
    
    def _restore_from_zip(self, backup_path: Path, restore_path: Path) -> bool:
        """Restore workspace from ZIP backup."""
        try:
            with zipfile.ZipFile(backup_path, 'r') as zip_file:
                zip_file.extractall(restore_path)
            return True
        except Exception:
            return False
    
    def _restore_from_tar(self, backup_path: Path, restore_path: Path) -> bool:
        """Restore workspace from TAR backup."""
        try:
            with tarfile.open(backup_path, 'r:*') as tar_file:
                tar_file.extractall(restore_path)
            return True
        except Exception:
            return False
    
    def _restore_from_directory(self, backup_path: Path, restore_path: Path) -> bool:
        """Restore workspace from directory backup."""
        try:
            shutil.copytree(backup_path, restore_path, dirs_exist_ok=True)
            return True
        except Exception:
            return False
    
    def _update_workspace_metadata(self, workspace_path: Path, updates: Dict[str, Any]):
        """Update workspace metadata file."""
        metadata_file = workspace_path / "workspace_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        metadata.update(updates)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _verify_migration_integrity(self, source_state: WorkspaceState, target_state: WorkspaceState) -> bool:
        """Verify data integrity after migration."""
        try:
            # Compare modality counts
            if len(source_state.modalities) != len(target_state.modalities):
                return False
            
            # Compare modality details
            for modality_name in source_state.modalities:
                if modality_name not in target_state.modalities:
                    return False
                
                source_info = source_state.modalities[modality_name]
                target_info = target_state.modalities[modality_name]
                
                # Skip if either has errors
                if 'error' in source_info or 'error' in target_info:
                    continue
                
                # Compare shapes
                if source_info.get('shape') != target_info.get('shape'):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _optimize_data_storage(self, data_path: Path) -> Dict[str, Any]:
        """Optimize data storage in workspace."""
        optimization_result = {
            'files_processed': 0,
            'space_saved_mb': 0,
            'errors': []
        }
        
        try:
            for h5ad_file in data_path.glob('*.h5ad'):
                try:
                    original_size = h5ad_file.stat().st_size
                    
                    # Read and rewrite to potentially improve compression
                    adata = ad.read_h5ad(h5ad_file)
                    temp_file = h5ad_file.with_suffix('.h5ad.tmp')
                    adata.write_h5ad(temp_file, compression='gzip')
                    
                    new_size = temp_file.stat().st_size
                    
                    # Keep optimized version if smaller
                    if new_size < original_size:
                        temp_file.replace(h5ad_file)
                        optimization_result['space_saved_mb'] += (original_size - new_size) / (1024**2)
                    else:
                        temp_file.unlink()
                    
                    optimization_result['files_processed'] += 1
                    
                except Exception as e:
                    optimization_result['errors'].append(f"Error optimizing {h5ad_file.name}: {str(e)}")
            
        except Exception as e:
            optimization_result['errors'].append(f"General optimization error: {str(e)}")
        
        return optimization_result


# ===============================================================================
# Fixtures for Workspace Management Testing
# ===============================================================================

@pytest.fixture(scope="session")
def workspace_test_base():
    """Create base directory for workspace tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir) / "workspace_tests"
        base_path.mkdir(parents=True, exist_ok=True)
        yield base_path


@pytest.fixture
def workspace_manager():
    """Create WorkspaceManager instance."""
    return WorkspaceManager()


@pytest.fixture
def sample_workspace_with_data(workspace_test_base, workspace_manager):
    """Create sample workspace with test data."""
    workspace_path = workspace_test_base / "sample_workspace"
    
    # Create workspace
    workspace_manager.create_workspace(workspace_path, initialize_structure=True)
    
    # Add test data
    data_manager = DataManagerV2(workspace_path=workspace_path)
    
    # Create sample datasets
    test_datasets = {
        'sample_single_cell': SingleCellDataFactory(config=SMALL_DATASET_CONFIG),
        'sample_bulk_rna': BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG),
        'sample_proteomics': ProteomicsDataFactory(config=SMALL_DATASET_CONFIG)
    }
    
    for name, adata in test_datasets.items():
        data_manager.modalities[name] = adata
    
    # Create some export files
    exports_path = workspace_path / "exports"
    (exports_path / "analysis_report.html").write_text("<html>Mock report</html>")
    (exports_path / "results_summary.json").write_text('{"results": "mock"}')
    
    # Create some cache files
    cache_path = workspace_path / "cache"
    (cache_path / "temp_data.tmp").write_text("temporary data")
    (cache_path / "processing_cache.cache").write_text("cache data")
    
    yield workspace_path, data_manager


@pytest.fixture
def concurrent_workspace_configs():
    """Define configurations for concurrent workspace testing."""
    return [
        {
            'workspace_id': 'concurrent_ws_1',
            'data_config': SMALL_DATASET_CONFIG,
            'n_modalities': 3,
            'operation_type': 'create_and_populate'
        },
        {
            'workspace_id': 'concurrent_ws_2', 
            'data_config': {**SMALL_DATASET_CONFIG, 'n_obs': 3000, 'n_vars': 1500},
            'n_modalities': 5,
            'operation_type': 'backup_and_restore'
        },
        {
            'workspace_id': 'concurrent_ws_3',
            'data_config': SMALL_DATASET_CONFIG,
            'n_modalities': 2,
            'operation_type': 'migrate_and_cleanup'
        }
    ]


# ===============================================================================
# Workspace Management System Tests
# ===============================================================================

@pytest.mark.system
class TestWorkspaceManagement:
    """Test comprehensive workspace management operations."""
    
    def test_workspace_creation_and_initialization(self, workspace_test_base, workspace_manager):
        """Test workspace creation with proper initialization."""
        workspace_path = workspace_test_base / "test_creation_workspace"
        
        # Test workspace creation
        creation_result = workspace_manager.create_workspace(workspace_path, initialize_structure=True)
        
        # Verify creation success
        assert creation_result['success'] == True
        assert workspace_path.exists()
        assert creation_result['operation_time'] < 10.0
        
        # Verify directory structure
        assert (workspace_path / "data").exists()
        assert (workspace_path / "exports").exists()
        assert (workspace_path / "cache").exists()
        
        # Verify metadata file
        metadata_file = workspace_path / "workspace_metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['workspace_id'] == workspace_path.name
        assert 'created_time' in metadata
        assert metadata['version'] == '2.0'
        
        # Verify provenance file
        provenance_file = workspace_path / "provenance.json"
        assert provenance_file.exists()
        
        # Test workspace state analysis
        initial_state = creation_result['initial_state']
        assert initial_state.file_count >= 2  # At least metadata and provenance files
        assert initial_state.total_size_mb > 0
    
    def test_workspace_backup_operations(self, sample_workspace_with_data, workspace_test_base, workspace_manager):
        """Test comprehensive workspace backup operations."""
        workspace_path, data_manager = sample_workspace_with_data
        
        # Test ZIP backup
        zip_backup_path = workspace_test_base / "backups" / "workspace_backup"
        zip_result = workspace_manager.backup_workspace(
            workspace_path, 
            zip_backup_path, 
            compression='zip', 
            include_cache=False
        )
        
        # Verify ZIP backup
        assert zip_result['success'] == True
        assert zip_result['compression'] == 'zip'
        assert zip_result['backup_size_mb'] > 0
        assert zip_result['compression_ratio'] < 1.0  # Should be compressed
        assert Path(zip_result['backup_path']).exists()
        
        # Test TAR backup  
        tar_backup_path = workspace_test_base / "backups" / "workspace_tar_backup"
        tar_result = workspace_manager.backup_workspace(
            workspace_path,
            tar_backup_path,
            compression='tar',
            include_cache=True
        )
        
        # Verify TAR backup
        assert tar_result['success'] == True
        assert tar_result['compression'] == 'tar'
        assert tar_result['include_cache'] == True
        assert Path(tar_result['backup_path']).exists()
        
        # Test directory backup
        dir_backup_path = workspace_test_base / "backups" / "workspace_dir_backup"
        dir_result = workspace_manager.backup_workspace(
            workspace_path,
            dir_backup_path,
            compression='none',
            include_cache=False
        )
        
        # Verify directory backup
        assert dir_result['success'] == True
        assert Path(dir_result['backup_path']).exists()
        assert Path(dir_result['backup_path']).is_dir()
        
        # Compare backup sizes (cache inclusion should affect size)
        assert tar_result['backup_size_mb'] >= zip_result['backup_size_mb']  # TAR includes cache
    
    def test_workspace_restore_operations(self, sample_workspace_with_data, workspace_test_base, workspace_manager):
        """Test workspace restore from various backup formats."""
        workspace_path, data_manager = sample_workspace_with_data
        
        # Create backup
        backup_path = workspace_test_base / "test_restore_backup"
        backup_result = workspace_manager.backup_workspace(
            workspace_path,
            backup_path,
            compression='zip',
            include_cache=True
        )
        
        assert backup_result['success'] == True
        
        # Test restore to new location
        restore_path = workspace_test_base / "restored_workspace"
        restore_result = workspace_manager.restore_workspace(
            Path(backup_result['backup_path']),
            restore_path,
            overwrite_existing=False
        )
        
        # Verify restore success
        assert restore_result['success'] == True
        assert restore_path.exists()
        assert restore_result['operation_time'] < 30.0
        
        # Verify restored structure
        assert (restore_path / "data").exists()
        assert (restore_path / "exports").exists()
        assert (restore_path / "cache").exists()
        
        # Verify data integrity
        restored_data_manager = DataManagerV2(workspace_path=restore_path)
        original_modalities = set(data_manager.list_modalities())
        restored_modalities = set(restored_data_manager.list_modalities())
        
        assert original_modalities == restored_modalities
        
        # Compare modality data
        for modality_name in original_modalities:
            original_adata = data_manager.get_modality(modality_name)
            restored_adata = restored_data_manager.get_modality(modality_name)
            
            assert original_adata.shape == restored_adata.shape
            assert list(original_adata.obs.columns) == list(restored_adata.obs.columns)
            assert list(original_adata.var.columns) == list(restored_adata.var.columns)
        
        # Test overwrite protection
        overwrite_result = workspace_manager.restore_workspace(
            Path(backup_result['backup_path']),
            restore_path,
            overwrite_existing=False
        )
        
        assert overwrite_result['success'] == False
        assert 'overwrite' in overwrite_result['error']
        
        # Test successful overwrite
        overwrite_success_result = workspace_manager.restore_workspace(
            Path(backup_result['backup_path']),
            restore_path,
            overwrite_existing=True
        )
        
        assert overwrite_success_result['success'] == True
    
    def test_workspace_migration_operations(self, sample_workspace_with_data, workspace_test_base, workspace_manager):
        """Test workspace migration between locations."""
        workspace_path, data_manager = sample_workspace_with_data
        
        # Test copy migration
        target_copy_path = workspace_test_base / "migrated_workspace_copy"
        copy_result = workspace_manager.migrate_workspace(
            workspace_path,
            target_copy_path,
            migration_type='copy'
        )
        
        # Verify copy migration
        assert copy_result['success'] == True
        assert copy_result['data_integrity_preserved'] == True
        assert workspace_path.exists()  # Original should still exist
        assert target_copy_path.exists()  # Target should exist
        
        # Verify data integrity
        migrated_data_manager = DataManagerV2(workspace_path=target_copy_path)
        original_modalities = set(data_manager.list_modalities())
        migrated_modalities = set(migrated_data_manager.list_modalities())
        
        assert original_modalities == migrated_modalities
        
        # Test move migration
        target_move_path = workspace_test_base / "migrated_workspace_move"
        move_result = workspace_manager.migrate_workspace(
            target_copy_path,  # Use copy as source to preserve original
            target_move_path,
            migration_type='move'
        )
        
        # Verify move migration
        assert move_result['success'] == True
        assert move_result['data_integrity_preserved'] == True
        assert not target_copy_path.exists()  # Source should be gone
        assert target_move_path.exists()  # Target should exist
        
        # Verify metadata update
        metadata_file = target_move_path / "workspace_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert 'migrated_from' in metadata
        assert 'migration_time' in metadata
        assert metadata['migration_type'] == 'move'
    
    def test_workspace_cleanup_operations(self, sample_workspace_with_data, workspace_manager):
        """Test comprehensive workspace cleanup operations."""
        workspace_path, data_manager = sample_workspace_with_data
        
        # Add more test files for cleanup
        cache_path = workspace_path / "cache"
        (cache_path / "large_temp_file.tmp").write_text("x" * 10000)
        (cache_path / "another_temp.temp").write_text("temporary content")
        
        exports_path = workspace_path / "exports"
        old_export = exports_path / "old_analysis.json"
        old_export.write_text('{"old": "analysis"}')
        
        # Set old timestamp
        old_time = time.time() - (40 * 24 * 60 * 60)  # 40 days ago
        os.utime(old_export, (old_time, old_time))
        
        # Test comprehensive cleanup
        cleanup_options = {
            'clear_cache': True,
            'remove_temp_files': True,
            'remove_old_exports': True,
            'export_age_days': 30,
            'optimize_storage': True
        }
        
        cleanup_result = workspace_manager.cleanup_workspace(workspace_path, cleanup_options)
        
        # Verify cleanup success
        assert cleanup_result['success'] == True
        assert cleanup_result['space_reduction_percentage'] > 0
        
        # Verify cache cleanup
        cache_contents = list(cache_path.iterdir())
        assert len(cache_contents) == 0  # Should be empty after cache clear
        
        # Verify old export removal
        assert not old_export.exists()
        
        # Verify temp file removal
        cleanup_summary = cleanup_result['cleanup_summary']
        assert len(cleanup_summary['files_removed']) > 0
        assert cleanup_summary['space_freed_mb'] > 0
        
        # Test selective cleanup
        selective_cleanup_options = {
            'clear_cache': False,
            'remove_temp_files': True,
            'remove_old_exports': False,
            'optimize_storage': False
        }
        
        selective_result = workspace_manager.cleanup_workspace(workspace_path, selective_cleanup_options)
        assert selective_result['success'] == True
    
    def test_workspace_concurrent_operations(self, workspace_test_base, concurrent_workspace_configs):
        """Test concurrent workspace operations."""
        
        class ConcurrentWorkspaceOperator:
            """Handles concurrent workspace operations."""
            
            def __init__(self, base_path: Path):
                self.base_path = base_path
                self.results = []
                self.workspace_manager = WorkspaceManager()
                
            def execute_concurrent_operations(self, configs: List[Dict]) -> List[Dict]:
                """Execute workspace operations concurrently."""
                
                def workspace_operation_worker(config: Dict) -> Dict:
                    """Worker function for concurrent operations."""
                    operation_start = time.time()
                    workspace_id = config['workspace_id']
                    workspace_path = self.base_path / workspace_id
                    
                    try:
                        if config['operation_type'] == 'create_and_populate':
                            # Create workspace
                            create_result = self.workspace_manager.create_workspace(
                                workspace_path, 
                                initialize_structure=True
                            )
                            
                            if create_result['success']:
                                # Populate with data
                                data_manager = DataManagerV2(workspace_path=workspace_path)
                                
                                for i in range(config['n_modalities']):
                                    adata = SingleCellDataFactory(config=config['data_config'])
                                    data_manager.modalities[f'{workspace_id}_modality_{i}'] = adata
                                
                                return {
                                    'workspace_id': workspace_id,
                                    'success': True,
                                    'operation': 'create_and_populate',
                                    'execution_time': time.time() - operation_start,
                                    'modalities_created': config['n_modalities']
                                }
                            else:
                                return {
                                    'workspace_id': workspace_id,
                                    'success': False,
                                    'operation': 'create_and_populate',
                                    'error': create_result.get('error', 'unknown'),
                                    'execution_time': time.time() - operation_start
                                }
                        
                        elif config['operation_type'] == 'backup_and_restore':
                            # Create workspace first
                            create_result = self.workspace_manager.create_workspace(workspace_path)
                            
                            if create_result['success']:
                                # Add some data
                                data_manager = DataManagerV2(workspace_path=workspace_path)
                                for i in range(config['n_modalities']):
                                    adata = SingleCellDataFactory(config=config['data_config'])
                                    data_manager.modalities[f'{workspace_id}_data_{i}'] = adata
                                
                                # Backup
                                backup_path = self.base_path / f"{workspace_id}_backup"
                                backup_result = self.workspace_manager.backup_workspace(
                                    workspace_path, backup_path, compression='zip'
                                )
                                
                                if backup_result['success']:
                                    # Restore
                                    restore_path = self.base_path / f"{workspace_id}_restored"
                                    restore_result = self.workspace_manager.restore_workspace(
                                        Path(backup_result['backup_path']), restore_path
                                    )
                                    
                                    return {
                                        'workspace_id': workspace_id,
                                        'success': restore_result['success'],
                                        'operation': 'backup_and_restore',
                                        'execution_time': time.time() - operation_start,
                                        'backup_size_mb': backup_result['backup_size_mb']
                                    }
                            
                            return {
                                'workspace_id': workspace_id,
                                'success': False,
                                'operation': 'backup_and_restore',
                                'error': 'backup_failed',
                                'execution_time': time.time() - operation_start
                            }
                        
                        elif config['operation_type'] == 'migrate_and_cleanup':
                            # Create workspace
                            create_result = self.workspace_manager.create_workspace(workspace_path)
                            
                            if create_result['success']:
                                # Add data
                                data_manager = DataManagerV2(workspace_path=workspace_path)
                                for i in range(config['n_modalities']):
                                    adata = SingleCellDataFactory(config=config['data_config'])
                                    data_manager.modalities[f'{workspace_id}_item_{i}'] = adata
                                
                                # Add cache files
                                cache_path = workspace_path / "cache"
                                (cache_path / "temp1.tmp").write_text("temp data 1")
                                (cache_path / "temp2.tmp").write_text("temp data 2")
                                
                                # Migrate
                                migrate_path = self.base_path / f"{workspace_id}_migrated"
                                migrate_result = self.workspace_manager.migrate_workspace(
                                    workspace_path, migrate_path, migration_type='copy'
                                )
                                
                                if migrate_result['success']:
                                    # Cleanup
                                    cleanup_result = self.workspace_manager.cleanup_workspace(
                                        migrate_path, 
                                        {'clear_cache': True, 'remove_temp_files': True}
                                    )
                                    
                                    return {
                                        'workspace_id': workspace_id,
                                        'success': cleanup_result['success'],
                                        'operation': 'migrate_and_cleanup',
                                        'execution_time': time.time() - operation_start,
                                        'space_freed_mb': cleanup_result['cleanup_summary']['space_freed_mb']
                                    }
                            
                            return {
                                'workspace_id': workspace_id,
                                'success': False,
                                'operation': 'migrate_and_cleanup',
                                'error': 'migration_failed',
                                'execution_time': time.time() - operation_start
                            }
                    
                    except Exception as e:
                        return {
                            'workspace_id': workspace_id,
                            'success': False,
                            'operation': config['operation_type'],
                            'error': str(e),
                            'execution_time': time.time() - operation_start
                        }
                
                # Execute operations concurrently
                concurrent_start = time.time()
                
                with ThreadPoolExecutor(max_workers=len(configs)) as executor:
                    future_to_config = {
                        executor.submit(workspace_operation_worker, config): config
                        for config in configs
                    }
                    
                    results = []
                    for future in as_completed(future_to_config):
                        result = future.result()
                        results.append(result)
                
                concurrent_end = time.time()
                
                return {
                    'concurrent_results': results,
                    'total_execution_time': concurrent_end - concurrent_start,
                    'concurrent_success_rate': sum(1 for r in results if r['success']) / len(results),
                    'operations_summary': {
                        op_type: [r for r in results if r.get('operation') == op_type]
                        for op_type in set(r.get('operation') for r in results)
                    }
                }
        
        # Test concurrent operations
        concurrent_operator = ConcurrentWorkspaceOperator(workspace_test_base)
        concurrent_results = concurrent_operator.execute_concurrent_operations(concurrent_workspace_configs)
        
        # Verify concurrent execution
        assert len(concurrent_results['concurrent_results']) == len(concurrent_workspace_configs)
        assert concurrent_results['concurrent_success_rate'] >= 0.8  # At least 80% success
        assert concurrent_results['total_execution_time'] < 180.0  # Should complete within 3 minutes
        
        # Verify individual operations
        successful_operations = [r for r in concurrent_results['concurrent_results'] if r['success']]
        assert len(successful_operations) >= 2  # At least 2 operations should succeed
        
        # Verify operation types completed
        operations_summary = concurrent_results['operations_summary']
        for operation_type in ['create_and_populate', 'backup_and_restore', 'migrate_and_cleanup']:
            if operation_type in operations_summary:
                type_results = operations_summary[operation_type]
                successful_type_ops = [r for r in type_results if r['success']]
                assert len(successful_type_ops) > 0, f"No successful {operation_type} operations"
        
        return concurrent_results
    
    def test_workspace_integrity_validation(self, sample_workspace_with_data, workspace_manager):
        """Test workspace integrity validation and corruption detection."""
        workspace_path, data_manager = sample_workspace_with_data
        
        class WorkspaceIntegrityValidator:
            """Validates workspace integrity and detects corruption."""
            
            def __init__(self, workspace_path: Path):
                self.workspace_path = workspace_path
                self.analyzer = WorkspaceAnalyzer()
                
            def validate_workspace_integrity(self) -> Dict[str, Any]:
                """Perform comprehensive workspace integrity validation."""
                validation_start = time.time()
                
                validation_results = {
                    'overall_integrity': True,
                    'issues_found': [],
                    'validation_details': {},
                    'suggestions': []
                }
                
                try:
                    # Validate structure
                    structure_validation = self._validate_directory_structure()
                    validation_results['validation_details']['structure'] = structure_validation
                    
                    if not structure_validation['valid']:
                        validation_results['overall_integrity'] = False
                        validation_results['issues_found'].extend(structure_validation['issues'])
                    
                    # Validate metadata
                    metadata_validation = self._validate_metadata_files()
                    validation_results['validation_details']['metadata'] = metadata_validation
                    
                    if not metadata_validation['valid']:
                        validation_results['overall_integrity'] = False
                        validation_results['issues_found'].extend(metadata_validation['issues'])
                    
                    # Validate data files
                    data_validation = self._validate_data_files()
                    validation_results['validation_details']['data'] = data_validation
                    
                    if not data_validation['valid']:
                        validation_results['overall_integrity'] = False
                        validation_results['issues_found'].extend(data_validation['issues'])
                    
                    # Performance checks
                    performance_validation = self._validate_performance_characteristics()
                    validation_results['validation_details']['performance'] = performance_validation
                    
                    # Generate suggestions
                    validation_results['suggestions'] = self._generate_improvement_suggestions(validation_results)
                    
                    validation_results['validation_time'] = time.time() - validation_start
                    
                except Exception as e:
                    validation_results['overall_integrity'] = False
                    validation_results['issues_found'].append(f"Validation error: {str(e)}")
                
                return validation_results
            
            def _validate_directory_structure(self) -> Dict[str, Any]:
                """Validate workspace directory structure."""
                required_dirs = ['data', 'exports', 'cache']
                required_files = ['workspace_metadata.json']
                
                structure_issues = []
                
                for req_dir in required_dirs:
                    dir_path = self.workspace_path / req_dir
                    if not dir_path.exists():
                        structure_issues.append(f"Missing required directory: {req_dir}")
                    elif not dir_path.is_dir():
                        structure_issues.append(f"Required directory is not a directory: {req_dir}")
                
                for req_file in required_files:
                    file_path = self.workspace_path / req_file
                    if not file_path.exists():
                        structure_issues.append(f"Missing required file: {req_file}")
                    elif not file_path.is_file():
                        structure_issues.append(f"Required file is not a file: {req_file}")
                
                return {
                    'valid': len(structure_issues) == 0,
                    'issues': structure_issues,
                    'required_dirs_present': len(required_dirs) - sum(
                        1 for d in required_dirs 
                        if not (self.workspace_path / d).exists()
                    ),
                    'required_files_present': len(required_files) - sum(
                        1 for f in required_files 
                        if not (self.workspace_path / f).exists()
                    )
                }
            
            def _validate_metadata_files(self) -> Dict[str, Any]:
                """Validate metadata files."""
                metadata_issues = []
                
                # Check workspace metadata
                metadata_file = self.workspace_path / "workspace_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        required_fields = ['workspace_id', 'created_time', 'version']
                        for field in required_fields:
                            if field not in metadata:
                                metadata_issues.append(f"Missing metadata field: {field}")
                        
                        # Validate version
                        if metadata.get('version') not in ['1.0', '2.0']:
                            metadata_issues.append(f"Unsupported workspace version: {metadata.get('version')}")
                    
                    except json.JSONDecodeError:
                        metadata_issues.append("Workspace metadata is not valid JSON")
                    except Exception as e:
                        metadata_issues.append(f"Error reading workspace metadata: {str(e)}")
                
                # Check provenance file
                provenance_file = self.workspace_path / "provenance.json"
                if provenance_file.exists():
                    try:
                        with open(provenance_file, 'r') as f:
                            provenance = json.load(f)
                        
                        if 'operations' not in provenance:
                            metadata_issues.append("Provenance file missing operations field")
                    
                    except json.JSONDecodeError:
                        metadata_issues.append("Provenance file is not valid JSON")
                
                return {
                    'valid': len(metadata_issues) == 0,
                    'issues': metadata_issues
                }
            
            def _validate_data_files(self) -> Dict[str, Any]:
                """Validate data files integrity."""
                data_issues = []
                validated_files = 0
                corrupted_files = 0
                
                data_path = self.workspace_path / "data"
                if data_path.exists():
                    for h5ad_file in data_path.glob("*.h5ad"):
                        try:
                            # Try to read the file
                            adata = ad.read_h5ad(h5ad_file)
                            
                            # Basic validation
                            if adata.n_obs == 0:
                                data_issues.append(f"Data file {h5ad_file.name} has no observations")
                            
                            if adata.n_vars == 0:
                                data_issues.append(f"Data file {h5ad_file.name} has no variables")
                            
                            if adata.X is None:
                                data_issues.append(f"Data file {h5ad_file.name} missing expression matrix")
                            
                            validated_files += 1
                        
                        except Exception as e:
                            data_issues.append(f"Cannot read data file {h5ad_file.name}: {str(e)}")
                            corrupted_files += 1
                
                return {
                    'valid': len(data_issues) == 0,
                    'issues': data_issues,
                    'validated_files': validated_files,
                    'corrupted_files': corrupted_files,
                    'corruption_rate': corrupted_files / (validated_files + corrupted_files) if (validated_files + corrupted_files) > 0 else 0
                }
            
            def _validate_performance_characteristics(self) -> Dict[str, Any]:
                """Validate workspace performance characteristics."""
                workspace_state = self.analyzer.analyze_workspace(self.workspace_path)
                
                performance_issues = []
                
                # Check workspace size
                if workspace_state.total_size_mb > 10000:  # 10GB
                    performance_issues.append("Workspace size is very large (>10GB)")
                
                # Check file count
                if workspace_state.file_count > 10000:
                    performance_issues.append("High file count may impact performance")
                
                # Check cache size
                cache_path = self.workspace_path / "cache"
                if cache_path.exists():
                    cache_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file()) / (1024**2)
                    if cache_size > 1000:  # 1GB
                        performance_issues.append("Cache directory is very large (>1GB)")
                
                return {
                    'total_size_mb': workspace_state.total_size_mb,
                    'file_count': workspace_state.file_count,
                    'performance_issues': performance_issues,
                    'performance_score': max(0, 100 - len(performance_issues) * 20)  # Simple scoring
                }
            
            def _generate_improvement_suggestions(self, validation_results: Dict) -> List[str]:
                """Generate suggestions for workspace improvement."""
                suggestions = []
                
                # Structure suggestions
                if 'structure' in validation_results['validation_details']:
                    structure = validation_results['validation_details']['structure']
                    if not structure['valid']:
                        suggestions.append("Run workspace repair to fix missing directories/files")
                
                # Performance suggestions
                if 'performance' in validation_results['validation_details']:
                    perf = validation_results['validation_details']['performance']
                    if 'Cache directory is very large' in perf['performance_issues']:
                        suggestions.append("Run workspace cleanup to reduce cache size")
                    
                    if perf['total_size_mb'] > 5000:
                        suggestions.append("Consider archiving old data to reduce workspace size")
                
                # Data suggestions
                if 'data' in validation_results['validation_details']:
                    data = validation_results['validation_details']['data']
                    if data['corruption_rate'] > 0:
                        suggestions.append("Backup workspace before corruption spreads")
                        suggestions.append("Restore corrupted files from backup if available")
                
                return suggestions
        
        # Test integrity validation
        validator = WorkspaceIntegrityValidator(workspace_path)
        validation_result = validator.validate_workspace_integrity()
        
        # Verify validation completed
        assert 'overall_integrity' in validation_result
        assert 'validation_details' in validation_result
        assert validation_result['validation_time'] < 30.0
        
        # Should be valid for clean workspace
        assert validation_result['overall_integrity'] == True
        assert len(validation_result['issues_found']) == 0
        
        # Test corruption detection by corrupting a file
        data_path = workspace_path / "data"
        if list(data_path.glob("*.h5ad")):
            test_file = list(data_path.glob("*.h5ad"))[0]
            
            # Corrupt the file
            with open(test_file, 'wb') as f:
                f.write(b"corrupted content")
            
            # Re-validate
            corrupted_validation = validator.validate_workspace_integrity()
            
            # Should detect corruption
            assert corrupted_validation['overall_integrity'] == False
            assert len(corrupted_validation['issues_found']) > 0
            assert any('Cannot read data file' in issue for issue in corrupted_validation['issues_found'])
            assert corrupted_validation['validation_details']['data']['corrupted_files'] > 0
        
        return validation_result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])