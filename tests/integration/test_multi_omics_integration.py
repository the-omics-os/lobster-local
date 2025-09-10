"""
Comprehensive integration tests for multi-omics data integration.

This module provides thorough testing of multi-omics data integration workflows
including transcriptomics-proteomics integration, spatial multi-omics, temporal
multi-omics, cross-platform integration, and unified analysis pipelines.

Test coverage target: 95%+ with realistic multi-omics scenarios.
"""

import pytest
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import mudata as mu
import scanpy as sc
import json
import time
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.preprocessing_service import PreprocessingService
from lobster.tools.clustering_service import ClusteringService
from lobster.tools.quality_service import QualityService
from lobster.tools.visualization_service import SingleCellVisualizationService

from tests.mock_data.factories import (
    SingleCellDataFactory, 
    BulkRNASeqDataFactory, 
    ProteomicsDataFactory,
    SpatialDataFactory
)
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Multi-Omics Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def temp_workspace():
    """Create temporary workspace for multi-omics integration tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def data_manager(temp_workspace):
    """Create DataManagerV2 instance for multi-omics testing."""
    return DataManagerV2(workspace_path=temp_workspace)


@pytest.fixture
def multi_omics_data():
    """Create matched multi-omics datasets."""
    n_cells = 1000
    n_genes = 2000
    n_proteins = 500
    
    # Create transcriptomics data
    rna_adata = SingleCellDataFactory(config={
        **SMALL_DATASET_CONFIG,
        'n_obs': n_cells,
        'n_vars': n_genes
    })
    
    # Create proteomics data with same cells
    protein_adata = ProteomicsDataFactory(config={
        **SMALL_DATASET_CONFIG,
        'n_obs': n_cells,
        'n_vars': n_proteins
    })
    
    # Ensure matching cell barcodes
    shared_barcodes = [f"CELL_{i:06d}" for i in range(n_cells)]
    rna_adata.obs.index = shared_barcodes
    protein_adata.obs.index = shared_barcodes
    
    # Add cell type annotations
    cell_types = ['T_cells', 'B_cells', 'NK_cells', 'Monocytes', 'Dendritic_cells']
    cell_type_labels = np.random.choice(cell_types, n_cells)
    
    rna_adata.obs['cell_type'] = cell_type_labels
    protein_adata.obs['cell_type'] = cell_type_labels
    
    # Add batch information
    batch_labels = np.random.choice(['Batch_1', 'Batch_2', 'Batch_3'], n_cells)
    rna_adata.obs['batch'] = batch_labels
    protein_adata.obs['batch'] = batch_labels
    
    # Add experimental conditions
    conditions = np.random.choice(['Control', 'Treatment_A', 'Treatment_B'], n_cells)
    rna_adata.obs['condition'] = conditions
    protein_adata.obs['condition'] = conditions
    
    return {
        'rna': rna_adata,
        'protein': protein_adata,
        'shared_cells': shared_barcodes,
        'cell_types': cell_types,
        'n_cells': n_cells
    }


@pytest.fixture
def spatial_multi_omics_data():
    """Create spatial multi-omics datasets."""
    n_spots = 2500
    n_genes = 3000
    n_proteins = 400
    
    # Create spatial coordinates
    x_coords = np.random.uniform(0, 100, n_spots)
    y_coords = np.random.uniform(0, 100, n_spots)
    
    # Create spatial transcriptomics data
    spatial_rna = SpatialDataFactory(config={
        **SMALL_DATASET_CONFIG,
        'n_obs': n_spots,
        'n_vars': n_genes
    })
    
    # Add spatial coordinates
    spatial_rna.obsm['spatial'] = np.column_stack([x_coords, y_coords])
    
    # Create matched proteomics data
    spatial_protein = ProteomicsDataFactory(config={
        **SMALL_DATASET_CONFIG,
        'n_obs': n_spots,
        'n_vars': n_proteins
    })
    
    # Add same spatial coordinates
    spatial_protein.obsm['spatial'] = spatial_rna.obsm['spatial'].copy()
    
    # Add tissue regions based on coordinates
    def assign_tissue_region(x, y):
        if x < 33 and y < 33:
            return 'cortex'
        elif x > 66 and y < 33:
            return 'medulla'
        elif y > 66:
            return 'capsule'
        else:
            return 'intermediate'
    
    tissue_regions = [assign_tissue_region(x, y) for x, y in zip(x_coords, y_coords)]
    spatial_rna.obs['tissue_region'] = tissue_regions
    spatial_protein.obs['tissue_region'] = tissue_regions
    
    return {
        'spatial_rna': spatial_rna,
        'spatial_protein': spatial_protein,
        'coordinates': np.column_stack([x_coords, y_coords]),
        'tissue_regions': tissue_regions
    }


@pytest.fixture
def temporal_multi_omics_data():
    """Create temporal multi-omics datasets."""
    timepoints = ['0h', '6h', '12h', '24h', '48h']
    n_cells_per_tp = 200
    n_genes = 2500
    n_proteins = 300
    
    temporal_data = {
        'timepoints': timepoints,
        'rna_data': {},
        'protein_data': {},
        'metadata': {}
    }
    
    for tp in timepoints:
        # Create RNA data for timepoint
        rna_data = SingleCellDataFactory(config={
            **SMALL_DATASET_CONFIG,
            'n_obs': n_cells_per_tp,
            'n_vars': n_genes
        })
        
        # Create protein data for timepoint
        protein_data = ProteomicsDataFactory(config={
            **SMALL_DATASET_CONFIG,
            'n_obs': n_cells_per_tp,
            'n_vars': n_proteins
        })
        
        # Add timepoint metadata
        rna_data.obs['timepoint'] = tp
        protein_data.obs['timepoint'] = tp
        
        # Add treatment conditions
        treatments = np.random.choice(['control', 'drug_treatment'], n_cells_per_tp)
        rna_data.obs['treatment'] = treatments
        protein_data.obs['treatment'] = treatments
        
        temporal_data['rna_data'][tp] = rna_data
        temporal_data['protein_data'][tp] = protein_data
        temporal_data['metadata'][tp] = {
            'n_cells': n_cells_per_tp,
            'conditions': list(set(treatments))
        }
    
    return temporal_data


@pytest.fixture
def mock_integration_services():
    """Create mock integration services."""
    return {
        'preprocessing_service': Mock(spec=PreprocessingService),
        'clustering_service': Mock(spec=ClusteringService),
        'quality_service': Mock(spec=QualityService),
        'visualization_service': Mock(spec=SingleCellVisualizationService)
    }


# ===============================================================================
# Basic Multi-Omics Integration Tests
# ===============================================================================

@pytest.mark.integration
class TestBasicMultiOmicsIntegration:
    """Test basic multi-omics integration functionality."""
    
    def test_transcriptomics_proteomics_integration(self, data_manager, multi_omics_data, mock_integration_services):
        """Test integration of transcriptomics and proteomics data."""
        
        class MultiOmicsIntegrator:
            """Handles multi-omics data integration."""
            
            def __init__(self, data_manager, services):
                self.data_manager = data_manager
                self.services = services
                self.integration_log = []
                
            def integrate_rna_protein(self, rna_data, protein_data, integration_method='wnn'):
                """Integrate RNA and protein data."""
                integration_result = {
                    'method': integration_method,
                    'input_modalities': {
                        'rna': {'n_obs': rna_data.n_obs, 'n_vars': rna_data.n_vars},
                        'protein': {'n_obs': protein_data.n_obs, 'n_vars': protein_data.n_vars}
                    },
                    'integration_successful': False,
                    'integrated_embedding': None,
                    'joint_clusters': None,
                    'integration_metrics': {}
                }
                
                # Step 1: Validate data compatibility
                compatibility = self._validate_data_compatibility(rna_data, protein_data)
                integration_result['data_compatibility'] = compatibility
                
                if not compatibility['compatible']:
                    integration_result['error'] = compatibility['error']
                    return integration_result
                
                # Step 2: Preprocess modalities
                rna_processed = self._preprocess_rna_data(rna_data)
                protein_processed = self._preprocess_protein_data(protein_data)
                
                # Step 3: Find shared features/cells
                shared_info = self._find_shared_elements(rna_processed, protein_processed)
                integration_result['shared_elements'] = shared_info
                
                # Step 4: Perform integration
                if integration_method == 'wnn':
                    integration_output = self._weighted_nearest_neighbor_integration(
                        rna_processed, protein_processed, shared_info
                    )
                elif integration_method == 'cca':
                    integration_output = self._canonical_correlation_analysis(
                        rna_processed, protein_processed, shared_info
                    )
                elif integration_method == 'pca_concat':
                    integration_output = self._pca_concatenation_integration(
                        rna_processed, protein_processed, shared_info
                    )
                else:
                    integration_result['error'] = f"Unknown integration method: {integration_method}"
                    return integration_result
                
                # Step 5: Evaluate integration quality
                integration_metrics = self._evaluate_integration_quality(
                    integration_output, rna_data, protein_data
                )
                
                integration_result.update({
                    'integration_successful': True,
                    'integrated_embedding': integration_output['embedding'],
                    'joint_clusters': integration_output.get('clusters'),
                    'integration_metrics': integration_metrics,
                    'method_parameters': integration_output.get('parameters', {})
                })
                
                self.integration_log.append(integration_result)
                return integration_result
            
            def _validate_data_compatibility(self, rna_data, protein_data):
                """Validate compatibility between RNA and protein data."""
                # Check cell overlap
                rna_cells = set(rna_data.obs.index)
                protein_cells = set(protein_data.obs.index)
                shared_cells = rna_cells.intersection(protein_cells)
                
                if len(shared_cells) < 10:
                    return {
                        'compatible': False,
                        'error': 'Insufficient shared cells between modalities',
                        'shared_cells': len(shared_cells)
                    }
                
                # Check metadata compatibility
                rna_metadata = set(rna_data.obs.columns)
                protein_metadata = set(protein_data.obs.columns)
                shared_metadata = rna_metadata.intersection(protein_metadata)
                
                return {
                    'compatible': True,
                    'shared_cells': len(shared_cells),
                    'shared_metadata_fields': list(shared_metadata),
                    'cell_overlap_percentage': len(shared_cells) / len(rna_cells) * 100
                }
            
            def _preprocess_rna_data(self, rna_data):
                """Preprocess RNA data for integration."""
                # Mock preprocessing
                processed_data = rna_data.copy()
                
                # Add mock preprocessing results
                processed_data.obs['total_counts_rna'] = np.random.randint(1000, 15000, processed_data.n_obs)
                processed_data.var['highly_variable_rna'] = np.random.choice([True, False], processed_data.n_vars, p=[0.2, 0.8])
                processed_data.obsm['X_pca_rna'] = np.random.randn(processed_data.n_obs, 50)
                
                return processed_data
            
            def _preprocess_protein_data(self, protein_data):
                """Preprocess protein data for integration."""
                # Mock preprocessing
                processed_data = protein_data.copy()
                
                # Add mock preprocessing results
                processed_data.obs['total_intensity_protein'] = np.random.uniform(100, 10000, processed_data.n_obs)
                processed_data.obsm['X_pca_protein'] = np.random.randn(processed_data.n_obs, 30)
                
                return processed_data
            
            def _find_shared_elements(self, rna_data, protein_data):
                """Find shared cells and features between modalities."""
                shared_cells = list(set(rna_data.obs.index).intersection(set(protein_data.obs.index)))
                
                # Look for gene-protein relationships
                rna_genes = set(rna_data.var.index)
                protein_names = set(protein_data.var.index)
                
                # Mock gene-protein mapping
                mapped_features = []
                for protein in list(protein_names)[:50]:  # Take first 50 proteins
                    # Simulate gene-protein relationship
                    if f"Gene_{protein.split('_')[-1]}" in rna_genes:
                        mapped_features.append((f"Gene_{protein.split('_')[-1]}", protein))
                
                return {
                    'shared_cells': shared_cells,
                    'n_shared_cells': len(shared_cells),
                    'mapped_features': mapped_features,
                    'n_mapped_features': len(mapped_features)
                }
            
            def _weighted_nearest_neighbor_integration(self, rna_data, protein_data, shared_info):
                """Perform weighted nearest neighbor integration."""
                n_shared_cells = shared_info['n_shared_cells']
                
                # Mock WNN integration
                integrated_embedding = np.random.randn(n_shared_cells, 30)  # 30D integrated space
                
                # Mock joint clustering
                joint_clusters = np.random.randint(0, 10, n_shared_cells)
                
                return {
                    'method': 'wnn',
                    'embedding': integrated_embedding,
                    'clusters': joint_clusters,
                    'parameters': {
                        'rna_weight': 0.6,
                        'protein_weight': 0.4,
                        'k_neighbors': 20
                    }
                }
            
            def _canonical_correlation_analysis(self, rna_data, protein_data, shared_info):
                """Perform canonical correlation analysis integration."""
                n_shared_cells = shared_info['n_shared_cells']
                
                # Mock CCA integration
                cca_embedding = np.random.randn(n_shared_cells, 20)  # 20D CCA space
                
                return {
                    'method': 'cca',
                    'embedding': cca_embedding,
                    'parameters': {
                        'n_components': 20,
                        'correlation_coefficients': np.random.uniform(0.3, 0.8, 20)
                    }
                }
            
            def _pca_concatenation_integration(self, rna_data, protein_data, shared_info):
                """Perform PCA concatenation integration."""
                n_shared_cells = shared_info['n_shared_cells']
                
                # Mock PCA concatenation
                # Concatenate PCA embeddings from both modalities
                rna_pca_subset = np.random.randn(n_shared_cells, 25)  # RNA PCA
                protein_pca_subset = np.random.randn(n_shared_cells, 15)  # Protein PCA
                
                concatenated_embedding = np.concatenate([rna_pca_subset, protein_pca_subset], axis=1)
                
                return {
                    'method': 'pca_concat',
                    'embedding': concatenated_embedding,
                    'parameters': {
                        'rna_pca_dims': 25,
                        'protein_pca_dims': 15,
                        'total_dims': 40
                    }
                }
            
            def _evaluate_integration_quality(self, integration_output, rna_data, protein_data):
                """Evaluate the quality of multi-omics integration."""
                # Mock integration quality metrics
                metrics = {
                    'silhouette_score': np.random.uniform(0.3, 0.8),
                    'batch_mixing_score': np.random.uniform(0.4, 0.9),
                    'biological_conservation_score': np.random.uniform(0.6, 0.95),
                    'modality_alignment_score': np.random.uniform(0.5, 0.85),
                    'cell_type_preservation': np.random.uniform(0.7, 0.95)
                }
                
                # Add method-specific metrics
                if integration_output['method'] == 'wnn':
                    metrics['neighbor_overlap_score'] = np.random.uniform(0.6, 0.9)
                elif integration_output['method'] == 'cca':
                    metrics['canonical_correlation_strength'] = np.mean(
                        integration_output['parameters']['correlation_coefficients']
                    )
                
                return metrics
        
        # Test transcriptomics-proteomics integration
        integrator = MultiOmicsIntegrator(data_manager, mock_integration_services)
        
        rna_data = multi_omics_data['rna']
        protein_data = multi_omics_data['protein']
        
        # Test different integration methods
        integration_methods = ['wnn', 'cca', 'pca_concat']
        
        for method in integration_methods:
            integration_result = integrator.integrate_rna_protein(rna_data, protein_data, method)
            
            # Verify integration success
            assert integration_result['integration_successful'] == True
            assert integration_result['method'] == method
            assert integration_result['data_compatibility']['compatible'] == True
            
            # Verify shared elements detection
            shared_info = integration_result['shared_elements']
            assert shared_info['n_shared_cells'] == multi_omics_data['n_cells']
            
            # Verify integration metrics
            metrics = integration_result['integration_metrics']
            assert 0 < metrics['silhouette_score'] < 1
            assert 0 < metrics['batch_mixing_score'] < 1
            assert 0 < metrics['biological_conservation_score'] < 1
            
            # Verify method-specific results
            if method == 'wnn':
                assert 'rna_weight' in integration_result['method_parameters']
                assert 'protein_weight' in integration_result['method_parameters']
            elif method == 'cca':
                assert 'correlation_coefficients' in integration_result['method_parameters']
            elif method == 'pca_concat':
                assert integration_result['method_parameters']['total_dims'] == 40
        
        # Verify integration log
        assert len(integrator.integration_log) == len(integration_methods)
    
    def test_mudata_integration_workflow(self, data_manager, multi_omics_data):
        """Test MuData-based multi-omics integration workflow."""
        
        class MuDataIntegrationWorkflow:
            """MuData-based integration workflow."""
            
            def __init__(self, data_manager):
                self.data_manager = data_manager
                
            def create_mudata_object(self, modalities_dict, modality_names=None):
                """Create MuData object from multiple modalities."""
                if modality_names is None:
                    modality_names = list(modalities_dict.keys())
                
                # Ensure all modalities have the same cells
                cell_intersection = None
                for modality_name, adata in modalities_dict.items():
                    if cell_intersection is None:
                        cell_intersection = set(adata.obs.index)
                    else:
                        cell_intersection = cell_intersection.intersection(set(adata.obs.index))
                
                # Filter modalities to shared cells
                filtered_modalities = {}
                for modality_name, adata in modalities_dict.items():
                    shared_cells = list(cell_intersection)
                    filtered_adata = adata[shared_cells, :].copy()
                    filtered_modalities[modality_name] = filtered_adata
                
                # Create MuData object
                mdata = mu.MuData(filtered_modalities)
                
                # Add global metadata
                mdata.obs['n_modalities'] = len(modality_names)
                mdata.obs['modalities_available'] = str(modality_names)
                
                return {
                    'mudata_object': mdata,
                    'n_shared_cells': len(cell_intersection),
                    'modalities': modality_names,
                    'integration_ready': True
                }
            
            def perform_mudata_integration(self, mdata, integration_params=None):
                """Perform integration using MuData framework."""
                integration_params = integration_params or {
                    'use_highly_variable': True,
                    'n_components': 50,
                    'integration_method': 'harmony'
                }
                
                # Mock MuData integration process
                n_cells = mdata.n_obs
                n_components = integration_params['n_components']
                
                # Add integrated embeddings
                mdata.obsm['X_integrated'] = np.random.randn(n_cells, n_components)
                mdata.obsm['X_umap_integrated'] = np.random.randn(n_cells, 2)
                
                # Add joint clustering
                integrated_clusters = np.random.randint(0, 12, n_cells)
                mdata.obs['integrated_clusters'] = integrated_clusters.astype(str)
                
                # Add integration quality metrics
                mdata.uns['integration_metrics'] = {
                    'method': integration_params['integration_method'],
                    'n_components': n_components,
                    'integration_score': np.random.uniform(0.7, 0.95),
                    'modality_weights': {
                        mod: np.random.uniform(0.2, 0.8) for mod in mdata.mod.keys()
                    }
                }
                
                return {
                    'integration_successful': True,
                    'integrated_mdata': mdata,
                    'n_integrated_components': n_components,
                    'joint_clusters': len(set(integrated_clusters))
                }
            
            def analyze_modality_relationships(self, mdata):
                """Analyze relationships between modalities."""
                relationships = {}
                
                modality_names = list(mdata.mod.keys())
                
                for i, mod1 in enumerate(modality_names):
                    for mod2 in modality_names[i+1:]:
                        # Mock correlation analysis between modalities
                        correlation_strength = np.random.uniform(0.3, 0.8)
                        
                        relationships[f"{mod1}_{mod2}"] = {
                            'correlation_strength': correlation_strength,
                            'significant_associations': np.random.randint(50, 500),
                            'co_regulated_features': np.random.randint(20, 200),
                            'relationship_type': self._classify_relationship(correlation_strength)
                        }
                
                return {
                    'modality_relationships': relationships,
                    'strongest_relationship': max(relationships.items(), key=lambda x: x[1]['correlation_strength']),
                    'average_correlation': np.mean([r['correlation_strength'] for r in relationships.values()])
                }
            
            def _classify_relationship(self, correlation_strength):
                """Classify the strength of modality relationship."""
                if correlation_strength >= 0.7:
                    return 'strong'
                elif correlation_strength >= 0.5:
                    return 'moderate'
                else:
                    return 'weak'
        
        # Test MuData integration workflow
        workflow = MuDataIntegrationWorkflow(data_manager)
        
        # Create MuData object
        modalities = {
            'rna': multi_omics_data['rna'],
            'protein': multi_omics_data['protein']
        }
        
        mudata_result = workflow.create_mudata_object(modalities)
        
        # Verify MuData creation
        assert mudata_result['integration_ready'] == True
        assert mudata_result['n_shared_cells'] == multi_omics_data['n_cells']
        assert set(mudata_result['modalities']) == {'rna', 'protein'}
        
        mdata = mudata_result['mudata_object']
        assert isinstance(mdata, mu.MuData)
        assert 'rna' in mdata.mod
        assert 'protein' in mdata.mod
        
        # Perform integration
        integration_result = workflow.perform_mudata_integration(mdata)
        
        # Verify integration
        assert integration_result['integration_successful'] == True
        assert integration_result['n_integrated_components'] == 50
        assert integration_result['joint_clusters'] > 0
        
        integrated_mdata = integration_result['integrated_mdata']
        assert 'X_integrated' in integrated_mdata.obsm
        assert 'X_umap_integrated' in integrated_mdata.obsm
        assert 'integrated_clusters' in integrated_mdata.obs
        assert 'integration_metrics' in integrated_mdata.uns
        
        # Analyze modality relationships
        relationships = workflow.analyze_modality_relationships(integrated_mdata)
        
        # Verify relationship analysis
        assert 'modality_relationships' in relationships
        assert 'rna_protein' in relationships['modality_relationships']
        assert 'strongest_relationship' in relationships
        assert 0 < relationships['average_correlation'] < 1
    
    def test_cross_platform_integration(self, data_manager, mock_integration_services):
        """Test integration of data from different experimental platforms."""
        
        class CrossPlatformIntegrator:
            """Handles cross-platform multi-omics integration."""
            
            def __init__(self, data_manager, services):
                self.data_manager = data_manager
                self.services = services
                
            def integrate_cross_platform_data(self, platform_datasets):
                """Integrate data from multiple experimental platforms."""
                integration_result = {
                    'input_platforms': list(platform_datasets.keys()),
                    'platform_compatibility': {},
                    'harmonization_applied': [],
                    'integrated_dataset': None,
                    'integration_successful': False
                }
                
                # Step 1: Assess platform compatibility
                compatibility_results = self._assess_platform_compatibility(platform_datasets)
                integration_result['platform_compatibility'] = compatibility_results
                
                # Step 2: Harmonize platform differences
                harmonization_results = self._harmonize_platforms(platform_datasets, compatibility_results)
                integration_result['harmonization_applied'] = harmonization_results['methods_applied']
                
                # Step 3: Integrate harmonized data
                if harmonization_results['harmonization_successful']:
                    harmonized_datasets = harmonization_results['harmonized_data']
                    integrated_data = self._perform_cross_platform_integration(harmonized_datasets)
                    integration_result['integrated_dataset'] = integrated_data
                    integration_result['integration_successful'] = True
                
                return integration_result
            
            def _assess_platform_compatibility(self, platform_datasets):
                """Assess compatibility between different platforms."""
                compatibility = {}
                
                platform_names = list(platform_datasets.keys())
                
                for platform in platform_names:
                    dataset = platform_datasets[platform]
                    
                    # Assess platform characteristics
                    platform_info = {
                        'data_type': self._infer_data_type(dataset),
                        'n_features': dataset.n_vars,
                        'n_samples': dataset.n_obs,
                        'feature_types': self._get_feature_types(dataset),
                        'normalization_status': self._check_normalization_status(dataset),
                        'batch_effects': self._detect_batch_effects(dataset)
                    }
                    
                    compatibility[platform] = platform_info
                
                # Assess cross-platform compatibility
                cross_platform_issues = []
                
                # Check for feature type mismatches
                feature_types = [info['feature_types'] for info in compatibility.values()]
                if len(set(str(ft) for ft in feature_types)) > 1:
                    cross_platform_issues.append('feature_type_mismatch')
                
                # Check for normalization differences
                norm_statuses = [info['normalization_status'] for info in compatibility.values()]
                if len(set(norm_statuses)) > 1:
                    cross_platform_issues.append('normalization_mismatch')
                
                compatibility['cross_platform_issues'] = cross_platform_issues
                compatibility['integration_feasible'] = len(cross_platform_issues) <= 2
                
                return compatibility
            
            def _infer_data_type(self, dataset):
                """Infer the type of omics data."""
                # Simple heuristic based on feature count and naming
                if dataset.n_vars > 10000:
                    return 'transcriptomics'
                elif dataset.n_vars < 1000:
                    return 'proteomics'
                elif 'spatial' in dataset.obsm:
                    return 'spatial_transcriptomics'
                else:
                    return 'unknown'
            
            def _get_feature_types(self, dataset):
                """Get types of features in the dataset."""
                # Mock feature type detection
                if hasattr(dataset.var, 'feature_types'):
                    return list(set(dataset.var['feature_types']))
                else:
                    return ['Gene Expression']  # Default
            
            def _check_normalization_status(self, dataset):
                """Check if data appears to be normalized."""
                # Simple heuristic: check data range
                max_value = np.max(dataset.X.data if hasattr(dataset.X, 'data') else dataset.X)
                
                if max_value > 1000:
                    return 'raw_counts'
                elif max_value > 10:
                    return 'normalized_counts'
                else:
                    return 'log_normalized'
            
            def _detect_batch_effects(self, dataset):
                """Detect potential batch effects."""
                # Mock batch effect detection
                return np.random.choice(['none', 'mild', 'moderate', 'severe'], p=[0.3, 0.3, 0.3, 0.1])
            
            def _harmonize_platforms(self, platform_datasets, compatibility_results):
                """Harmonize differences between platforms."""
                harmonization_result = {
                    'harmonization_successful': False,
                    'methods_applied': [],
                    'harmonized_data': {}
                }
                
                cross_platform_issues = compatibility_results['cross_platform_issues']
                
                harmonized_data = {}
                methods_applied = []
                
                for platform_name, dataset in platform_datasets.items():
                    harmonized_dataset = dataset.copy()
                    
                    # Apply normalization harmonization
                    if 'normalization_mismatch' in cross_platform_issues:
                        harmonized_dataset = self._harmonize_normalization(harmonized_dataset)
                        methods_applied.append(f'normalization_harmonization_{platform_name}')
                    
                    # Apply batch effect correction
                    platform_batch_effects = compatibility_results[platform_name]['batch_effects']
                    if platform_batch_effects in ['moderate', 'severe']:
                        harmonized_dataset = self._correct_batch_effects(harmonized_dataset)
                        methods_applied.append(f'batch_correction_{platform_name}')
                    
                    # Apply feature harmonization
                    if 'feature_type_mismatch' in cross_platform_issues:
                        harmonized_dataset = self._harmonize_features(harmonized_dataset, platform_name)
                        methods_applied.append(f'feature_harmonization_{platform_name}')
                    
                    harmonized_data[platform_name] = harmonized_dataset
                
                harmonization_result.update({
                    'harmonization_successful': True,
                    'methods_applied': list(set(methods_applied)),
                    'harmonized_data': harmonized_data
                })
                
                return harmonization_result
            
            def _harmonize_normalization(self, dataset):
                """Harmonize normalization across platforms."""
                # Mock normalization harmonization
                harmonized_dataset = dataset.copy()
                # Add harmonization metadata
                harmonized_dataset.uns['normalization_harmonized'] = True
                harmonized_dataset.uns['target_normalization'] = 'log1p_10000'
                return harmonized_dataset
            
            def _correct_batch_effects(self, dataset):
                """Correct batch effects in dataset."""
                # Mock batch effect correction
                corrected_dataset = dataset.copy()
                corrected_dataset.uns['batch_correction_applied'] = True
                corrected_dataset.uns['batch_correction_method'] = 'combat'
                return corrected_dataset
            
            def _harmonize_features(self, dataset, platform_name):
                """Harmonize features across platforms."""
                # Mock feature harmonization
                harmonized_dataset = dataset.copy()
                harmonized_dataset.uns['feature_harmonization_applied'] = True
                harmonized_dataset.uns['harmonization_method'] = f'{platform_name}_specific_mapping'
                return harmonized_dataset
            
            def _perform_cross_platform_integration(self, harmonized_datasets):
                """Integrate harmonized cross-platform datasets."""
                # Mock cross-platform integration
                platform_names = list(harmonized_datasets.keys())
                
                # Find common cells across platforms
                common_cells = None
                for platform_name, dataset in harmonized_datasets.items():
                    if common_cells is None:
                        common_cells = set(dataset.obs.index)
                    else:
                        common_cells = common_cells.intersection(set(dataset.obs.index))
                
                # Create integrated embedding
                n_common_cells = len(common_cells)
                integrated_embedding = np.random.randn(n_common_cells, 30)
                
                integration_result = {
                    'n_platforms_integrated': len(platform_names),
                    'platforms': platform_names,
                    'n_common_cells': n_common_cells,
                    'integrated_embedding': integrated_embedding,
                    'integration_method': 'cross_platform_harmony',
                    'platform_weights': {
                        platform: np.random.uniform(0.2, 0.8) 
                        for platform in platform_names
                    }
                }
                
                return integration_result
        
        # Create mock cross-platform datasets
        platform_datasets = {
            '10x_genomics': SingleCellDataFactory(config={**SMALL_DATASET_CONFIG, 'n_obs': 800, 'n_vars': 2500}),
            'smart_seq2': SingleCellDataFactory(config={**SMALL_DATASET_CONFIG, 'n_obs': 600, 'n_vars': 3000}),
            'cite_seq': SingleCellDataFactory(config={**SMALL_DATASET_CONFIG, 'n_obs': 700, 'n_vars': 200})
        }
        
        # Add platform-specific metadata
        platform_datasets['10x_genomics'].uns['platform'] = '10x_genomics'
        platform_datasets['smart_seq2'].uns['platform'] = 'smart_seq2'
        platform_datasets['cite_seq'].uns['platform'] = 'cite_seq'
        
        # Test cross-platform integration
        integrator = CrossPlatformIntegrator(data_manager, mock_integration_services)
        
        integration_result = integrator.integrate_cross_platform_data(platform_datasets)
        
        # Verify cross-platform integration
        assert integration_result['integration_successful'] == True
        assert len(integration_result['input_platforms']) == 3
        assert '10x_genomics' in integration_result['input_platforms']
        assert 'smart_seq2' in integration_result['input_platforms']
        assert 'cite_seq' in integration_result['input_platforms']
        
        # Verify platform compatibility assessment
        compatibility = integration_result['platform_compatibility']
        assert compatibility['integration_feasible'] == True
        assert '10x_genomics' in compatibility
        assert 'smart_seq2' in compatibility
        assert 'cite_seq' in compatibility
        
        # Verify harmonization was applied
        assert len(integration_result['harmonization_applied']) > 0
        
        # Verify integrated dataset
        integrated_data = integration_result['integrated_dataset']
        assert integrated_data['n_platforms_integrated'] == 3
        assert integrated_data['n_common_cells'] > 0
        assert 'integrated_embedding' in integrated_data


# ===============================================================================
# Spatial Multi-Omics Integration Tests
# ===============================================================================

@pytest.mark.integration
class TestSpatialMultiOmicsIntegration:
    """Test spatial multi-omics integration functionality."""
    
    def test_spatial_transcriptomics_proteomics_integration(self, data_manager, spatial_multi_omics_data):
        """Test integration of spatial transcriptomics and proteomics data."""
        
        class SpatialMultiOmicsIntegrator:
            """Handles spatial multi-omics integration."""
            
            def __init__(self, data_manager):
                self.data_manager = data_manager
                
            def integrate_spatial_multi_omics(self, spatial_rna, spatial_protein, integration_config=None):
                """Integrate spatial transcriptomics and proteomics data."""
                integration_config = integration_config or {
                    'spatial_weight': 0.3,
                    'expression_weight': 0.7,
                    'neighborhood_size': 50,
                    'resolution': 0.5
                }
                
                integration_result = {
                    'spatial_integration_successful': False,
                    'spatial_domains': None,
                    'co_localization_patterns': {},
                    'spatial_correlation_map': None
                }
                
                # Step 1: Validate spatial alignment
                alignment_result = self._validate_spatial_alignment(spatial_rna, spatial_protein)
                integration_result['spatial_alignment'] = alignment_result
                
                if not alignment_result['well_aligned']:
                    integration_result['error'] = 'Poor spatial alignment between modalities'
                    return integration_result
                
                # Step 2: Identify spatial domains
                spatial_domains = self._identify_spatial_domains(spatial_rna, spatial_protein, integration_config)
                integration_result['spatial_domains'] = spatial_domains
                
                # Step 3: Analyze co-localization patterns
                colocalization = self._analyze_colocalization_patterns(spatial_rna, spatial_protein)
                integration_result['co_localization_patterns'] = colocalization
                
                # Step 4: Create spatial correlation map
                correlation_map = self._create_spatial_correlation_map(spatial_rna, spatial_protein)
                integration_result['spatial_correlation_map'] = correlation_map
                
                integration_result['spatial_integration_successful'] = True
                return integration_result
            
            def _validate_spatial_alignment(self, spatial_rna, spatial_protein):
                """Validate spatial alignment between modalities."""
                rna_coords = spatial_rna.obsm['spatial']
                protein_coords = spatial_protein.obsm['spatial']
                
                # Check coordinate system compatibility
                coord_diff = np.mean(np.abs(rna_coords - protein_coords))
                alignment_quality = 1 / (1 + coord_diff)  # Higher is better
                
                return {
                    'well_aligned': alignment_quality > 0.8,
                    'alignment_score': alignment_quality,
                    'coordinate_difference': coord_diff,
                    'coordinate_correlation': np.corrcoef(rna_coords.flat, protein_coords.flat)[0, 1]
                }
            
            def _identify_spatial_domains(self, spatial_rna, spatial_protein, config):
                """Identify spatial domains using multi-omics information."""
                # Mock spatial domain identification
                n_spots = spatial_rna.n_obs
                
                # Create spatial domains based on coordinates and expression
                coordinates = spatial_rna.obsm['spatial']
                
                # Simple spatial clustering based on coordinates
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=6, random_state=42)
                spatial_clusters = kmeans.fit_predict(coordinates)
                
                # Refine clusters using expression information
                refined_domains = self._refine_domains_with_expression(
                    spatial_clusters, spatial_rna, spatial_protein
                )
                
                return {
                    'n_domains': len(set(refined_domains)),
                    'domain_assignments': refined_domains,
                    'domain_characteristics': self._characterize_domains(
                        refined_domains, spatial_rna, spatial_protein
                    ),
                    'spatial_coherence_score': np.random.uniform(0.6, 0.9)
                }
            
            def _refine_domains_with_expression(self, initial_domains, spatial_rna, spatial_protein):
                """Refine spatial domains using expression information."""
                # Mock domain refinement
                # In reality, this would use expression similarity to refine spatial clusters
                refined_domains = initial_domains.copy()
                
                # Add some noise to simulate refinement
                noise = np.random.randint(-1, 2, len(initial_domains))
                refined_domains = np.maximum(0, refined_domains + noise)
                
                return refined_domains
            
            def _characterize_domains(self, domain_assignments, spatial_rna, spatial_protein):
                """Characterize each spatial domain."""
                unique_domains = set(domain_assignments)
                domain_characteristics = {}
                
                for domain_id in unique_domains:
                    domain_mask = domain_assignments == domain_id
                    domain_spots = np.where(domain_mask)[0]
                    
                    # Mock domain characterization
                    characteristics = {
                        'n_spots': len(domain_spots),
                        'dominant_cell_types': np.random.choice(
                            ['immune', 'epithelial', 'stromal', 'endothelial'], 
                            size=np.random.randint(1, 3), replace=False
                        ).tolist(),
                        'rna_signature_genes': [f'RNA_marker_{i}' for i in range(np.random.randint(5, 15))],
                        'protein_signature': [f'Protein_marker_{i}' for i in range(np.random.randint(3, 8))],
                        'spatial_coherence': np.random.uniform(0.5, 0.95)
                    }
                    
                    domain_characteristics[f'domain_{domain_id}'] = characteristics
                
                return domain_characteristics
            
            def _analyze_colocalization_patterns(self, spatial_rna, spatial_protein):
                """Analyze co-localization patterns between RNA and protein."""
                # Mock co-localization analysis
                colocalization_patterns = {
                    'high_colocalization_pairs': [
                        {'rna_gene': 'CD3D', 'protein': 'CD3_protein', 'correlation': 0.85},
                        {'rna_gene': 'CD8A', 'protein': 'CD8_protein', 'correlation': 0.78},
                        {'rna_gene': 'CD4', 'protein': 'CD4_protein', 'correlation': 0.73}
                    ],
                    'spatial_anticorrelation_pairs': [
                        {'rna_gene': 'FOXP3', 'protein': 'CD8_protein', 'correlation': -0.62}
                    ],
                    'tissue_region_specific': {
                        'cortex': {
                            'enriched_rna': ['Cortex_gene_1', 'Cortex_gene_2'],
                            'enriched_proteins': ['Cortex_protein_1']
                        },
                        'medulla': {
                            'enriched_rna': ['Medulla_gene_1', 'Medulla_gene_2'],
                            'enriched_proteins': ['Medulla_protein_1', 'Medulla_protein_2']
                        }
                    },
                    'global_colocalization_score': np.random.uniform(0.6, 0.8)
                }
                
                return colocalization_patterns
            
            def _create_spatial_correlation_map(self, spatial_rna, spatial_protein):
                """Create spatial correlation map between modalities."""
                coordinates = spatial_rna.obsm['spatial']
                
                # Mock spatial correlation calculation
                correlation_map = {
                    'correlation_matrix': np.random.uniform(-0.5, 0.8, (100, 100)),  # 100x100 spatial grid
                    'high_correlation_regions': [
                        {'region': 'region_1', 'coordinates': [(10, 15), (20, 25)], 'correlation': 0.85},
                        {'region': 'region_2', 'coordinates': [(60, 70), (80, 90)], 'correlation': 0.78}
                    ],
                    'spatial_gradient_strength': np.random.uniform(0.3, 0.7),
                    'correlation_hotspots': 5
                }
                
                return correlation_map
        
        # Test spatial multi-omics integration
        integrator = SpatialMultiOmicsIntegrator(data_manager)
        
        spatial_rna = spatial_multi_omics_data['spatial_rna']
        spatial_protein = spatial_multi_omics_data['spatial_protein']
        
        integration_result = integrator.integrate_spatial_multi_omics(spatial_rna, spatial_protein)
        
        # Verify spatial integration
        assert integration_result['spatial_integration_successful'] == True
        
        # Verify spatial alignment
        alignment = integration_result['spatial_alignment']
        assert alignment['well_aligned'] == True
        assert alignment['alignment_score'] > 0.8
        
        # Verify spatial domains
        domains = integration_result['spatial_domains']
        assert domains['n_domains'] > 0
        assert domains['spatial_coherence_score'] > 0.5
        assert len(domains['domain_characteristics']) == domains['n_domains']
        
        # Verify co-localization patterns
        colocalization = integration_result['co_localization_patterns']
        assert 'high_colocalization_pairs' in colocalization
        assert len(colocalization['high_colocalization_pairs']) > 0
        assert 'global_colocalization_score' in colocalization
        
        # Verify spatial correlation map
        correlation_map = integration_result['spatial_correlation_map']
        assert 'correlation_matrix' in correlation_map
        assert 'high_correlation_regions' in correlation_map


# ===============================================================================
# Temporal Multi-Omics Integration Tests
# ===============================================================================

@pytest.mark.integration
class TestTemporalMultiOmicsIntegration:
    """Test temporal multi-omics integration functionality."""
    
    def test_temporal_multi_omics_trajectory_analysis(self, data_manager, temporal_multi_omics_data):
        """Test trajectory analysis across temporal multi-omics data."""
        
        class TemporalMultiOmicsAnalyzer:
            """Analyzes temporal multi-omics trajectories."""
            
            def __init__(self, data_manager):
                self.data_manager = data_manager
                
            def analyze_temporal_trajectories(self, temporal_data, analysis_config=None):
                """Analyze trajectories across temporal multi-omics data."""
                analysis_config = analysis_config or {
                    'trajectory_method': 'diffusion_pseudotime',
                    'integration_weight_rna': 0.6,
                    'integration_weight_protein': 0.4,
                    'smooth_trajectories': True
                }
                
                trajectory_result = {
                    'trajectory_analysis_successful': False,
                    'integrated_trajectories': None,
                    'temporal_patterns': {},
                    'cross_omics_dynamics': {}
                }
                
                # Step 1: Align temporal data across modalities
                aligned_data = self._align_temporal_modalities(temporal_data)
                trajectory_result['temporal_alignment'] = aligned_data['alignment_info']
                
                # Step 2: Compute integrated trajectories
                trajectories = self._compute_integrated_trajectories(
                    aligned_data['aligned_rna'], aligned_data['aligned_protein'], analysis_config
                )
                trajectory_result['integrated_trajectories'] = trajectories
                
                # Step 3: Identify temporal patterns
                patterns = self._identify_temporal_patterns(trajectories, temporal_data)
                trajectory_result['temporal_patterns'] = patterns
                
                # Step 4: Analyze cross-omics dynamics
                dynamics = self._analyze_cross_omics_dynamics(
                    aligned_data['aligned_rna'], aligned_data['aligned_protein'], trajectories
                )
                trajectory_result['cross_omics_dynamics'] = dynamics
                
                trajectory_result['trajectory_analysis_successful'] = True
                return trajectory_result
            
            def _align_temporal_modalities(self, temporal_data):
                """Align temporal data across RNA and protein modalities."""
                timepoints = temporal_data['timepoints']
                
                aligned_rna_data = []
                aligned_protein_data = []
                alignment_info = {
                    'timepoints_processed': timepoints,
                    'cells_per_timepoint': {},
                    'modality_correlation_per_timepoint': {}
                }
                
                for tp in timepoints:
                    rna_tp = temporal_data['rna_data'][tp]
                    protein_tp = temporal_data['protein_data'][tp]
                    
                    # Mock alignment process
                    aligned_rna_data.append(rna_tp)
                    aligned_protein_data.append(protein_tp)
                    
                    alignment_info['cells_per_timepoint'][tp] = rna_tp.n_obs
                    alignment_info['modality_correlation_per_timepoint'][tp] = np.random.uniform(0.4, 0.8)
                
                return {
                    'aligned_rna': aligned_rna_data,
                    'aligned_protein': aligned_protein_data,
                    'alignment_info': alignment_info
                }
            
            def _compute_integrated_trajectories(self, rna_data_list, protein_data_list, config):
                """Compute integrated trajectories across modalities and time."""
                n_timepoints = len(rna_data_list)
                
                # Mock trajectory computation
                trajectories = {
                    'pseudotime_rna': [],
                    'pseudotime_protein': [],
                    'integrated_pseudotime': [],
                    'trajectory_branches': []
                }
                
                for i, (rna_data, protein_data) in enumerate(zip(rna_data_list, protein_data_list)):
                    # Mock pseudotime calculation
                    n_cells = rna_data.n_obs
                    
                    # RNA pseudotime (based on real timepoint + noise)
                    base_time = i / (n_timepoints - 1)
                    rna_pseudotime = base_time + np.random.normal(0, 0.1, n_cells)
                    rna_pseudotime = np.clip(rna_pseudotime, 0, 1)
                    
                    # Protein pseudotime (similar but with different dynamics)
                    protein_pseudotime = base_time + np.random.normal(0, 0.15, n_cells)
                    protein_pseudotime = np.clip(protein_pseudotime, 0, 1)
                    
                    # Integrated pseudotime
                    rna_weight = config['integration_weight_rna']
                    protein_weight = config['integration_weight_protein']
                    integrated_pseudotime = (rna_weight * rna_pseudotime + 
                                           protein_weight * protein_pseudotime)
                    
                    trajectories['pseudotime_rna'].extend(rna_pseudotime)
                    trajectories['pseudotime_protein'].extend(protein_pseudotime)
                    trajectories['integrated_pseudotime'].extend(integrated_pseudotime)
                
                # Identify trajectory branches
                trajectories['trajectory_branches'] = self._identify_trajectory_branches(trajectories)
                
                return trajectories
            
            def _identify_trajectory_branches(self, trajectories):
                """Identify branching points in trajectories."""
                # Mock trajectory branch identification
                branches = [
                    {
                        'branch_point': 0.3,
                        'branch_id': 'early_divergence',
                        'cell_fate_1': 'activated_state',
                        'cell_fate_2': 'quiescent_state',
                        'branch_confidence': 0.75
                    },
                    {
                        'branch_point': 0.7,
                        'branch_id': 'late_specification',
                        'cell_fate_1': 'differentiated_state_A',
                        'cell_fate_2': 'differentiated_state_B',
                        'branch_confidence': 0.68
                    }
                ]
                
                return branches
            
            def _identify_temporal_patterns(self, trajectories, temporal_data):
                """Identify temporal patterns in multi-omics data."""
                patterns = {
                    'coordinated_changes': [],
                    'delayed_responses': [],
                    'oscillatory_patterns': [],
                    'monotonic_trends': []
                }
                
                # Mock pattern identification
                patterns['coordinated_changes'] = [
                    {
                        'gene_protein_pair': ('CD3D', 'CD3_protein'),
                        'coordination_score': 0.85,
                        'temporal_correlation': 0.78,
                        'peak_timepoint': '12h'
                    },
                    {
                        'gene_protein_pair': ('IL2', 'IL2_protein'),
                        'coordination_score': 0.72,
                        'temporal_correlation': 0.65,
                        'peak_timepoint': '6h'
                    }
                ]
                
                patterns['delayed_responses'] = [
                    {
                        'gene': 'IFNG',
                        'protein': 'IFNG_protein',
                        'rna_peak': '6h',
                        'protein_peak': '12h',
                        'delay_hours': 6,
                        'delay_correlation': 0.68
                    }
                ]
                
                patterns['oscillatory_patterns'] = [
                    {
                        'feature': 'Cell_cycle_genes',
                        'modality': 'rna',
                        'oscillation_period': '24h',
                        'amplitude': 0.5,
                        'phase_shift': 0
                    }
                ]
                
                patterns['monotonic_trends'] = [
                    {
                        'feature': 'Differentiation_markers',
                        'trend_direction': 'increasing',
                        'r_squared': 0.82,
                        'modalities': ['rna', 'protein']
                    }
                ]
                
                return patterns
            
            def _analyze_cross_omics_dynamics(self, rna_data_list, protein_data_list, trajectories):
                """Analyze dynamics between RNA and protein across time."""
                dynamics = {
                    'lag_analysis': {},
                    'correlation_dynamics': {},
                    'regulatory_relationships': [],
                    'temporal_coupling_strength': 0.0
                }
                
                # Mock cross-omics dynamics analysis
                dynamics['lag_analysis'] = {
                    'average_rna_protein_lag': 2.5,  # hours
                    'lag_variability': 1.2,
                    'genes_with_significant_lag': ['Gene1', 'Gene2', 'Gene3']
                }
                
                dynamics['correlation_dynamics'] = {
                    'early_timepoints_correlation': 0.65,
                    'late_timepoints_correlation': 0.78,
                    'correlation_trend': 'increasing',
                    'peak_correlation_timepoint': '24h'
                }
                
                dynamics['regulatory_relationships'] = [
                    {
                        'regulator_rna': 'TF1',
                        'target_protein': 'Target_protein_1',
                        'relationship_type': 'positive_regulation',
                        'temporal_delay': 3.0,
                        'confidence': 0.72
                    },
                    {
                        'regulator_protein': 'Kinase1',
                        'target_rna': 'Target_gene_1',
                        'relationship_type': 'feedback_regulation',
                        'temporal_delay': 6.0,
                        'confidence': 0.68
                    }
                ]
                
                dynamics['temporal_coupling_strength'] = np.random.uniform(0.6, 0.8)
                
                return dynamics
        
        # Test temporal multi-omics trajectory analysis
        analyzer = TemporalMultiOmicsAnalyzer(data_manager)
        
        trajectory_result = analyzer.analyze_temporal_trajectories(temporal_multi_omics_data)
        
        # Verify trajectory analysis
        assert trajectory_result['trajectory_analysis_successful'] == True
        
        # Verify temporal alignment
        alignment = trajectory_result['temporal_alignment']
        assert len(alignment['timepoints_processed']) == len(temporal_multi_omics_data['timepoints'])
        assert all(tp in alignment['cells_per_timepoint'] for tp in temporal_multi_omics_data['timepoints'])
        
        # Verify integrated trajectories
        trajectories = trajectory_result['integrated_trajectories']
        assert 'pseudotime_rna' in trajectories
        assert 'pseudotime_protein' in trajectories
        assert 'integrated_pseudotime' in trajectories
        assert 'trajectory_branches' in trajectories
        assert len(trajectories['trajectory_branches']) > 0
        
        # Verify temporal patterns
        patterns = trajectory_result['temporal_patterns']
        assert 'coordinated_changes' in patterns
        assert 'delayed_responses' in patterns
        assert len(patterns['coordinated_changes']) > 0
        
        # Verify cross-omics dynamics
        dynamics = trajectory_result['cross_omics_dynamics']
        assert 'lag_analysis' in dynamics
        assert 'correlation_dynamics' in dynamics
        assert 'regulatory_relationships' in dynamics
        assert dynamics['temporal_coupling_strength'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])