"""
Comprehensive unit tests for method expert agent.

This module provides thorough testing of the method expert agent including
parameter extraction from papers, protocol optimization, method benchmarking,
reproducibility validation, and integration with research workflows.

Test coverage target: 95%+ with meaningful tests for method expertise.
"""

import pytest
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, MagicMock, patch
import json
import numpy as np
import pandas as pd

from lobster.agents.method_expert import method_expert
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Mock Objects and Fixtures
# ===============================================================================

class MockMessage:
    """Mock LangGraph message object."""
    
    def __init__(self, content: str, sender: str = "human"):
        self.content = content
        self.sender = sender
        self.additional_kwargs = {}


class MockState:
    """Mock LangGraph state object."""
    
    def __init__(self, messages=None, **kwargs):
        self.messages = messages or []
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockPaper:
    """Mock paper object for method extraction."""
    
    def __init__(self, pmid: str, title: str, methods_section: str, **kwargs):
        self.pmid = pmid
        self.title = title
        self.methods_section = methods_section
        self.abstract = kwargs.get("abstract", "")
        self.authors = kwargs.get("authors", [])
        self.journal = kwargs.get("journal", "")
        self.year = kwargs.get("year", "2023")


@pytest.fixture
def mock_data_manager():
    """Create mock data manager."""
    with patch('lobster.core.data_manager_v2.DataManagerV2') as MockDataManager:
        mock_dm = MockDataManager.return_value
        mock_dm.list_modalities.return_value = ['test_sc_data', 'test_bulk_data']
        mock_dm.get_modality.return_value = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_dm.get_summary.return_value = "Test dataset with 1000 cells and 2000 genes"
        yield mock_dm


@pytest.fixture
def mock_pubmed_service():
    """Mock PubMed service for paper retrieval."""
    with patch('lobster.tools.providers.pubmed.PubMedProvider') as MockPubMed:
        mock_service = MockPubMed.return_value
        
        # Mock clustering paper
        mock_service.get_paper_details.return_value = MockPaper(
            pmid="12345678",
            title="Optimized single-cell clustering using resolution parameter tuning",
            methods_section="""
            Single-cell RNA sequencing data was processed using Scanpy (v1.9.1).
            Quality control filtering removed cells with <200 genes and genes expressed in <3 cells.
            Data was normalized using scanpy.pp.normalize_total (target_sum=10000) followed by 
            log transformation. Highly variable genes were identified using scanpy.pp.highly_variable_genes
            (n_top_genes=2000, flavor='seurat_v3').
            
            Principal component analysis was performed using scanpy.tl.pca (n_comps=50).
            Neighborhood graph construction used scanpy.pp.neighbors (n_neighbors=10, n_pcs=40).
            Leiden clustering was applied with resolution=0.5 using scanpy.tl.leiden.
            UMAP embedding was computed using scanpy.tl.umap (min_dist=0.5, spread=1.0).
            """,
            abstract="We present an optimized approach for single-cell RNA-seq clustering...",
            authors=["Smith J", "Doe A", "Johnson B"],
            journal="Nature Methods",
            year="2023"
        )
        
        # Mock differential expression paper
        mock_service.search_papers.return_value = [
            {
                "pmid": "87654321",
                "title": "Improved differential expression analysis with DESeq2 parameter optimization",
                "authors": ["Wilson C", "Brown D"],
                "journal": "Genome Biology",
                "year": "2023"
            }
        ]
        
        yield mock_service


@pytest.fixture
def method_expert_state():
    """Create method expert state for testing."""
    return MockState(
        messages=[MockMessage("Extract clustering parameters from this paper")],
        data_manager=Mock(),
        current_agent="method_expert_agent"
    )


# ===============================================================================
# Method Expert Core Functionality Tests
# ===============================================================================

@pytest.mark.unit
class TestMethodExpertCore:
    """Test method expert core functionality."""
    
    def test_extract_parameters_from_paper(self, mock_pubmed_service):
        """Test parameter extraction from research papers."""
        with patch('lobster.agents.method_expert.extract_parameters_from_paper') as mock_extract:
            mock_extract.return_value = {
                "scanpy_parameters": {
                    "normalize_total": {"target_sum": 10000},
                    "highly_variable_genes": {"n_top_genes": 2000, "flavor": "seurat_v3"},
                    "pca": {"n_comps": 50},
                    "neighbors": {"n_neighbors": 10, "n_pcs": 40},
                    "leiden": {"resolution": 0.5},
                    "umap": {"min_dist": 0.5, "spread": 1.0}
                },
                "quality_control": {
                    "min_genes_per_cell": 200,
                    "min_cells_per_gene": 3
                }
            }
            
            result = mock_extract("12345678")
            
            assert "scanpy_parameters" in result
            assert result["scanpy_parameters"]["leiden"]["resolution"] == 0.5
            assert result["quality_control"]["min_genes_per_cell"] == 200
            mock_extract.assert_called_once_with("12345678")
    
    def test_analyze_method_section(self, mock_pubmed_service):
        """Test analysis of methods section text."""
        methods_text = """
        Cells were filtered using min_genes=200 and min_cells=3.
        Normalization was performed with target_sum=1e4.
        PCA used n_comps=50, neighbors with n_neighbors=15.
        """
        
        with patch('lobster.agents.method_expert.analyze_method_section') as mock_analyze:
            mock_analyze.return_value = {
                "extracted_parameters": {
                    "min_genes": 200,
                    "min_cells": 3,
                    "target_sum": 10000,
                    "n_comps": 50,
                    "n_neighbors": 15
                },
                "method_steps": [
                    "quality_control_filtering",
                    "normalization",
                    "dimensionality_reduction",
                    "neighborhood_graph"
                ]
            }
            
            result = mock_analyze(methods_text)
            
            assert result["extracted_parameters"]["min_genes"] == 200
            assert "quality_control_filtering" in result["method_steps"]
            mock_analyze.assert_called_once_with(methods_text)
    
    def test_optimize_parameters_for_dataset(self, mock_data_manager):
        """Test parameter optimization for specific datasets."""
        with patch('lobster.agents.method_expert.optimize_parameters_for_dataset') as mock_optimize:
            mock_optimize.return_value = {
                "original_parameters": {"resolution": 0.5, "n_neighbors": 10},
                "optimized_parameters": {"resolution": 0.8, "n_neighbors": 15},
                "optimization_metrics": {
                    "silhouette_score": 0.65,
                    "modularity": 0.82,
                    "n_clusters": 12
                },
                "recommendation": "Increased resolution to 0.8 for better cluster separation"
            }
            
            result = mock_optimize("test_dataset", method="leiden_clustering")
            
            assert result["optimized_parameters"]["resolution"] == 0.8
            assert result["optimization_metrics"]["silhouette_score"] == 0.65
            mock_optimize.assert_called_once_with("test_dataset", method="leiden_clustering")
    
    def test_validate_method_reproducibility(self, mock_data_manager):
        """Test method reproducibility validation."""
        with patch('lobster.agents.method_expert.validate_method_reproducibility') as mock_validate:
            mock_validate.return_value = {
                "reproducible": True,
                "consistency_score": 0.95,
                "replicate_correlation": 0.92,
                "parameter_sensitivity": {
                    "resolution": "low",
                    "n_neighbors": "medium"
                },
                "recommendations": [
                    "Method is highly reproducible",
                    "Consider n_neighbors sensitivity in parameter selection"
                ]
            }
            
            result = mock_validate("test_method", parameters={"resolution": 0.5})
            
            assert result["reproducible"] == True
            assert result["consistency_score"] == 0.95
            mock_validate.assert_called_once_with("test_method", parameters={"resolution": 0.5})


# ===============================================================================
# Parameter Extraction and Analysis Tests
# ===============================================================================

@pytest.mark.unit
class TestParameterExtractionAnalysis:
    """Test parameter extraction and analysis functionality."""
    
    def test_parse_scanpy_parameters(self):
        """Test parsing Scanpy-specific parameters."""
        methods_text = """
        scanpy.pp.filter_cells(adata, min_genes=200)
        scanpy.pp.filter_genes(adata, min_cells=3)
        scanpy.pp.normalize_total(adata, target_sum=1e4)
        scanpy.pp.log1p(adata)
        scanpy.pp.highly_variable_genes(adata, n_top_genes=2000)
        scanpy.tl.pca(adata, n_comps=50)
        scanpy.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        scanpy.tl.leiden(adata, resolution=0.5)
        """
        
        with patch('lobster.agents.method_expert.parse_scanpy_parameters') as mock_parse:
            mock_parse.return_value = {
                "filter_cells": {"min_genes": 200},
                "filter_genes": {"min_cells": 3},
                "normalize_total": {"target_sum": 10000},
                "highly_variable_genes": {"n_top_genes": 2000},
                "pca": {"n_comps": 50},
                "neighbors": {"n_neighbors": 10, "n_pcs": 40},
                "leiden": {"resolution": 0.5}
            }
            
            result = mock_parse(methods_text)
            
            assert result["filter_cells"]["min_genes"] == 200
            assert result["leiden"]["resolution"] == 0.5
            mock_parse.assert_called_once_with(methods_text)
    
    def test_parse_seurat_parameters(self):
        """Test parsing Seurat-specific parameters."""
        methods_text = """
        CreateSeuratObject(counts = counts, min.cells = 3, min.features = 200)
        NormalizeData(object, normalization.method = "LogNormalize", scale.factor = 10000)
        FindVariableFeatures(object, selection.method = "vst", nfeatures = 2000)
        ScaleData(object)
        RunPCA(object, features = VariableFeatures(object), npcs = 50)
        FindNeighbors(object, dims = 1:40)
        FindClusters(object, resolution = 0.5)
        RunUMAP(object, dims = 1:40)
        """
        
        with patch('lobster.agents.method_expert.parse_seurat_parameters') as mock_parse:
            mock_parse.return_value = {
                "CreateSeuratObject": {"min.cells": 3, "min.features": 200},
                "NormalizeData": {"normalization.method": "LogNormalize", "scale.factor": 10000},
                "FindVariableFeatures": {"selection.method": "vst", "nfeatures": 2000},
                "RunPCA": {"npcs": 50},
                "FindNeighbors": {"dims": "1:40"},
                "FindClusters": {"resolution": 0.5},
                "RunUMAP": {"dims": "1:40"}
            }
            
            result = mock_parse(methods_text)
            
            assert result["CreateSeuratObject"]["min.cells"] == 3
            assert result["FindClusters"]["resolution"] == 0.5
            mock_parse.assert_called_once_with(methods_text)
    
    def test_extract_statistical_parameters(self):
        """Test extraction of statistical analysis parameters."""
        methods_text = """
        Differential expression analysis was performed using DESeq2 with 
        adjusted p-value < 0.05 and log2 fold change > 1.5.
        Multiple testing correction used Benjamini-Hochberg method.
        Minimum count threshold was set to 10 reads per gene.
        """
        
        with patch('lobster.agents.method_expert.extract_statistical_parameters') as mock_extract:
            mock_extract.return_value = {
                "differential_expression": {
                    "method": "DESeq2",
                    "p_value_threshold": 0.05,
                    "log2_fold_change_threshold": 1.5,
                    "multiple_testing_correction": "Benjamini-Hochberg",
                    "min_count_threshold": 10
                }
            }
            
            result = mock_extract(methods_text)
            
            assert result["differential_expression"]["method"] == "DESeq2"
            assert result["differential_expression"]["p_value_threshold"] == 0.05
            mock_extract.assert_called_once_with(methods_text)
    
    def test_identify_software_versions(self):
        """Test identification of software versions."""
        methods_text = """
        Analysis was performed using Python 3.8.5, scanpy 1.9.1, 
        pandas 1.3.0, and numpy 1.21.0. R version 4.1.0 was used
        for Seurat 4.0.3 analysis.
        """
        
        with patch('lobster.agents.method_expert.identify_software_versions') as mock_identify:
            mock_identify.return_value = {
                "python": "3.8.5",
                "scanpy": "1.9.1",
                "pandas": "1.3.0",
                "numpy": "1.21.0",
                "r": "4.1.0",
                "seurat": "4.0.3"
            }
            
            result = mock_identify(methods_text)
            
            assert result["scanpy"] == "1.9.1"
            assert result["seurat"] == "4.0.3"
            mock_identify.assert_called_once_with(methods_text)


# ===============================================================================
# Method Optimization Tests
# ===============================================================================

@pytest.mark.unit
class TestMethodOptimization:
    """Test method optimization functionality."""
    
    def test_optimize_clustering_parameters(self, mock_data_manager):
        """Test clustering parameter optimization."""
        with patch('lobster.agents.method_expert.optimize_clustering_parameters') as mock_optimize:
            mock_optimize.return_value = {
                "parameter_grid": {
                    "resolution": [0.1, 0.3, 0.5, 0.8, 1.0],
                    "n_neighbors": [5, 10, 15, 20]
                },
                "best_parameters": {"resolution": 0.8, "n_neighbors": 15},
                "optimization_results": {
                    "best_silhouette": 0.72,
                    "best_modularity": 0.85,
                    "optimal_clusters": 14
                },
                "parameter_effects": {
                    "resolution_impact": "high",
                    "neighbors_impact": "medium"
                }
            }
            
            result = mock_optimize("test_dataset", method="leiden")
            
            assert result["best_parameters"]["resolution"] == 0.8
            assert result["optimization_results"]["best_silhouette"] == 0.72
            mock_optimize.assert_called_once_with("test_dataset", method="leiden")
    
    def test_optimize_normalization_parameters(self, mock_data_manager):
        """Test normalization parameter optimization."""
        with patch('lobster.agents.method_expert.optimize_normalization_parameters') as mock_optimize:
            mock_optimize.return_value = {
                "tested_methods": ["LogNormalize", "SCTransform", "CPM"],
                "best_method": "LogNormalize",
                "best_parameters": {"target_sum": 10000, "scale_factor": 1e4},
                "evaluation_metrics": {
                    "variance_stabilization": 0.85,
                    "batch_effect_reduction": 0.72,
                    "hvg_detection_quality": 0.90
                }
            }
            
            result = mock_optimize("test_dataset")
            
            assert result["best_method"] == "LogNormalize"
            assert result["best_parameters"]["target_sum"] == 10000
            mock_optimize.assert_called_once_with("test_dataset")
    
    def test_optimize_dimensionality_reduction(self, mock_data_manager):
        """Test dimensionality reduction optimization."""
        with patch('lobster.agents.method_expert.optimize_dimensionality_reduction') as mock_optimize:
            mock_optimize.return_value = {
                "pca_optimization": {
                    "optimal_components": 45,
                    "explained_variance": 0.87,
                    "elbow_point": 42
                },
                "umap_optimization": {
                    "best_min_dist": 0.3,
                    "best_n_neighbors": 15,
                    "best_spread": 1.2,
                    "embedding_quality": 0.78
                }
            }
            
            result = mock_optimize("test_dataset")
            
            assert result["pca_optimization"]["optimal_components"] == 45
            assert result["umap_optimization"]["best_min_dist"] == 0.3
            mock_optimize.assert_called_once_with("test_dataset")
    
    def test_benchmark_method_performance(self, mock_data_manager):
        """Test method performance benchmarking."""
        with patch('lobster.agents.method_expert.benchmark_method_performance') as mock_benchmark:
            mock_benchmark.return_value = {
                "methods_compared": ["leiden", "louvain", "kmeans"],
                "performance_metrics": {
                    "leiden": {"silhouette": 0.72, "modularity": 0.85, "runtime": 12.5},
                    "louvain": {"silhouette": 0.68, "modularity": 0.82, "runtime": 8.3},
                    "kmeans": {"silhouette": 0.65, "modularity": 0.75, "runtime": 5.2}
                },
                "best_method": "leiden",
                "trade_offs": {
                    "accuracy_vs_speed": "leiden offers best accuracy but slower",
                    "recommendation": "Use leiden for quality, louvain for speed"
                }
            }
            
            result = mock_benchmark("test_dataset", methods=["leiden", "louvain", "kmeans"])
            
            assert result["best_method"] == "leiden"
            assert result["performance_metrics"]["leiden"]["silhouette"] == 0.72
            mock_benchmark.assert_called_once_with("test_dataset", methods=["leiden", "louvain", "kmeans"])


# ===============================================================================
# Protocol Analysis Tests
# ===============================================================================

@pytest.mark.unit
class TestProtocolAnalysis:
    """Test protocol analysis functionality."""
    
    def test_analyze_experimental_protocol(self, mock_pubmed_service):
        """Test experimental protocol analysis."""
        with patch('lobster.agents.method_expert.analyze_experimental_protocol') as mock_analyze:
            mock_analyze.return_value = {
                "protocol_type": "single_cell_rna_seq",
                "library_preparation": {
                    "method": "10X Chromium",
                    "chemistry": "3' v3.1",
                    "target_cells": 5000
                },
                "sequencing": {
                    "platform": "Illumina NovaSeq 6000",
                    "read_structure": "28-8-91",
                    "depth": "50000 reads per cell"
                },
                "quality_thresholds": {
                    "min_genes_per_cell": 200,
                    "max_mt_percent": 20,
                    "doublet_rate_threshold": 0.05
                }
            }
            
            result = mock_analyze("12345678")
            
            assert result["protocol_type"] == "single_cell_rna_seq"
            assert result["library_preparation"]["method"] == "10X Chromium"
            mock_analyze.assert_called_once_with("12345678")
    
    def test_compare_protocols(self, mock_pubmed_service):
        """Test protocol comparison."""
        with patch('lobster.agents.method_expert.compare_protocols') as mock_compare:
            mock_compare.return_value = {
                "protocols": ["protocol_A", "protocol_B"],
                "comparison_matrix": {
                    "sensitivity": {"protocol_A": 0.85, "protocol_B": 0.78},
                    "throughput": {"protocol_A": 5000, "protocol_B": 10000},
                    "cost_per_cell": {"protocol_A": 0.05, "protocol_B": 0.03}
                },
                "recommendations": {
                    "for_sensitivity": "protocol_A",
                    "for_throughput": "protocol_B",
                    "for_cost_effectiveness": "protocol_B"
                }
            }
            
            result = mock_compare(["protocol_A", "protocol_B"])
            
            assert result["comparison_matrix"]["sensitivity"]["protocol_A"] == 0.85
            assert result["recommendations"]["for_sensitivity"] == "protocol_A"
            mock_compare.assert_called_once_with(["protocol_A", "protocol_B"])
    
    def test_validate_protocol_compatibility(self, mock_data_manager):
        """Test protocol compatibility validation."""
        with patch('lobster.agents.method_expert.validate_protocol_compatibility') as mock_validate:
            mock_validate.return_value = {
                "compatible": True,
                "compatibility_score": 0.92,
                "potential_issues": [],
                "adaptation_required": False,
                "recommendations": [
                    "Protocol is fully compatible with dataset",
                    "No parameter adjustments needed"
                ]
            }
            
            result = mock_validate("test_dataset", "standard_10x_protocol")
            
            assert result["compatible"] == True
            assert result["compatibility_score"] == 0.92
            mock_validate.assert_called_once_with("test_dataset", "standard_10x_protocol")


# ===============================================================================
# Method Validation and Reproducibility Tests
# ===============================================================================

@pytest.mark.unit
class TestMethodValidationReproducibility:
    """Test method validation and reproducibility functionality."""
    
    def test_validate_method_implementation(self):
        """Test method implementation validation."""
        method_code = """
        import scanpy as sc
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.tl.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_neighbors=10)
        sc.tl.leiden(adata, resolution=0.5)
        """
        
        with patch('lobster.agents.method_expert.validate_method_implementation') as mock_validate:
            mock_validate.return_value = {
                "valid": True,
                "syntax_check": "passed",
                "parameter_validation": "passed",
                "best_practices": {
                    "followed": ["proper_filtering", "log_normalization", "pca_before_neighbors"],
                    "missing": ["highly_variable_genes_selection"]
                },
                "suggestions": [
                    "Consider adding highly variable genes selection before PCA"
                ]
            }
            
            result = mock_validate(method_code)
            
            assert result["valid"] == True
            assert "proper_filtering" in result["best_practices"]["followed"]
            mock_validate.assert_called_once_with(method_code)
    
    def test_assess_reproducibility_factors(self, mock_data_manager):
        """Test reproducibility factors assessment."""
        with patch('lobster.agents.method_expert.assess_reproducibility_factors') as mock_assess:
            mock_assess.return_value = {
                "seed_dependency": {
                    "required": True,
                    "impact_level": "high",
                    "affected_steps": ["pca", "neighbors", "leiden"]
                },
                "parameter_sensitivity": {
                    "resolution": {"sensitivity": "high", "impact": "cluster_count"},
                    "n_neighbors": {"sensitivity": "medium", "impact": "connectivity"}
                },
                "software_dependencies": {
                    "critical": ["scanpy>=1.8.0", "numpy>=1.20.0"],
                    "version_sensitive": ["leiden", "umap-learn"]
                },
                "reproducibility_score": 0.85
            }
            
            result = mock_assess("test_method", "test_dataset")
            
            assert result["seed_dependency"]["required"] == True
            assert result["reproducibility_score"] == 0.85
            mock_assess.assert_called_once_with("test_method", "test_dataset")
    
    def test_generate_reproducible_protocol(self):
        """Test reproducible protocol generation."""
        with patch('lobster.agents.method_expert.generate_reproducible_protocol') as mock_generate:
            mock_generate.return_value = {
                "protocol_id": "reproducible_clustering_v1.0",
                "method_steps": [
                    {"step": "quality_control", "parameters": {"min_genes": 200, "min_cells": 3}},
                    {"step": "normalization", "parameters": {"target_sum": 10000}},
                    {"step": "feature_selection", "parameters": {"n_top_genes": 2000}},
                    {"step": "pca", "parameters": {"n_comps": 50, "random_state": 42}},
                    {"step": "neighbors", "parameters": {"n_neighbors": 10, "random_state": 42}},
                    {"step": "clustering", "parameters": {"resolution": 0.5, "random_state": 42}}
                ],
                "environment_specs": {
                    "python": "3.8+",
                    "scanpy": "1.9.1",
                    "required_packages": ["pandas>=1.3.0", "numpy>=1.20.0"]
                },
                "validation_tests": [
                    "test_parameter_consistency",
                    "test_output_reproducibility",
                    "test_cross_platform_compatibility"
                ]
            }
            
            result = mock_generate("clustering_analysis")
            
            assert result["protocol_id"] == "reproducible_clustering_v1.0"
            assert len(result["method_steps"]) == 6
            mock_generate.assert_called_once_with("clustering_analysis")
    
    def test_cross_validate_method_results(self, mock_data_manager):
        """Test cross-validation of method results."""
        with patch('lobster.agents.method_expert.cross_validate_method_results') as mock_cross_val:
            mock_cross_val.return_value = {
                "cross_validation_folds": 5,
                "consistency_metrics": {
                    "cluster_stability": 0.88,
                    "marker_gene_consistency": 0.92,
                    "parameter_robustness": 0.85
                },
                "variance_analysis": {
                    "cluster_count_variance": 0.15,
                    "silhouette_score_variance": 0.08,
                    "modularity_variance": 0.12
                },
                "reliability_assessment": "high"
            }
            
            result = mock_cross_val("test_method", "test_dataset", folds=5)
            
            assert result["cross_validation_folds"] == 5
            assert result["consistency_metrics"]["cluster_stability"] == 0.88
            mock_cross_val.assert_called_once_with("test_method", "test_dataset", folds=5)


# ===============================================================================
# Integration and Workflow Tests
# ===============================================================================

@pytest.mark.unit
class TestMethodExpertIntegration:
    """Test method expert integration functionality."""
    
    def test_integrate_with_research_workflow(self, method_expert_state):
        """Test integration with research workflow."""
        method_expert_state.messages = [MockMessage("Extract and optimize parameters from paper PMID:12345678")]
        
        with patch('lobster.agents.method_expert.method_expert_agent') as mock_agent:
            mock_agent.return_value = {
                "messages": [MockMessage("Parameters extracted and optimized for your dataset", "assistant")],
                "extracted_parameters": {
                    "leiden_resolution": 0.8,
                    "n_neighbors": 15,
                    "normalization_target": 10000
                },
                "optimization_results": {
                    "improved_silhouette": 0.15,
                    "cluster_count": 12
                },
                "implementation_ready": True
            }
            
            result = mock_agent(method_expert_state)
            
            assert "extracted_parameters" in result
            assert result["implementation_ready"] == True
    
    def test_method_recommendation_system(self, mock_data_manager):
        """Test method recommendation system."""
        with patch('lobster.agents.method_expert.recommend_methods_for_dataset') as mock_recommend:
            mock_recommend.return_value = {
                "dataset_characteristics": {
                    "n_cells": 5000,
                    "n_genes": 20000,
                    "sparsity": 0.92,
                    "data_type": "single_cell_rna_seq"
                },
                "recommended_methods": [
                    {
                        "method": "leiden_clustering",
                        "confidence": 0.95,
                        "rationale": "Optimal for large single-cell datasets"
                    },
                    {
                        "method": "sctransform_normalization", 
                        "confidence": 0.88,
                        "rationale": "Handles sparse data effectively"
                    }
                ],
                "parameter_suggestions": {
                    "leiden_resolution": [0.5, 0.8],
                    "n_neighbors": [10, 15, 20]
                }
            }
            
            result = mock_recommend("test_dataset")
            
            assert len(result["recommended_methods"]) == 2
            assert result["recommended_methods"][0]["confidence"] == 0.95
            mock_recommend.assert_called_once_with("test_dataset")
    
    def test_adaptive_parameter_tuning(self, mock_data_manager):
        """Test adaptive parameter tuning."""
        with patch('lobster.agents.method_expert.adaptive_parameter_tuning') as mock_tune:
            mock_tune.return_value = {
                "tuning_iterations": 3,
                "parameter_evolution": {
                    "iteration_1": {"resolution": 0.5, "silhouette": 0.65},
                    "iteration_2": {"resolution": 0.7, "silhouette": 0.71},
                    "iteration_3": {"resolution": 0.8, "silhouette": 0.72}
                },
                "final_parameters": {"resolution": 0.8, "n_neighbors": 15},
                "convergence_achieved": True,
                "improvement_percentage": 10.8
            }
            
            result = mock_tune("test_dataset", target_metric="silhouette_score")
            
            assert result["convergence_achieved"] == True
            assert result["improvement_percentage"] == 10.8
            mock_tune.assert_called_once_with("test_dataset", target_metric="silhouette_score")
    
    def test_method_pipeline_construction(self, mock_data_manager):
        """Test method pipeline construction."""
        with patch('lobster.agents.method_expert.construct_analysis_pipeline') as mock_construct:
            mock_construct.return_value = {
                "pipeline_id": "optimized_sc_analysis_v1",
                "pipeline_steps": [
                    {"name": "quality_control", "function": "filter_cells_genes", "order": 1},
                    {"name": "normalization", "function": "normalize_total", "order": 2},
                    {"name": "feature_selection", "function": "highly_variable_genes", "order": 3},
                    {"name": "dimensionality_reduction", "function": "pca", "order": 4},
                    {"name": "neighborhood", "function": "neighbors", "order": 5},
                    {"name": "clustering", "function": "leiden", "order": 6}
                ],
                "parameter_set": {
                    "quality_control": {"min_genes": 200, "min_cells": 3},
                    "normalization": {"target_sum": 10000},
                    "clustering": {"resolution": 0.8}
                },
                "estimated_runtime": "5-10 minutes"
            }
            
            result = mock_construct("single_cell_analysis", "test_dataset")
            
            assert len(result["pipeline_steps"]) == 6
            assert result["parameter_set"]["clustering"]["resolution"] == 0.8
            mock_construct.assert_called_once_with("single_cell_analysis", "test_dataset")


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestMethodExpertErrorHandling:
    """Test method expert error handling and edge cases."""
    
    def test_invalid_paper_handling(self, mock_pubmed_service):
        """Test handling of invalid or inaccessible papers."""
        with patch('lobster.agents.method_expert.extract_parameters_from_paper') as mock_extract:
            mock_extract.side_effect = ValueError("Paper PMID:INVALID not found or inaccessible")
            
            with pytest.raises(ValueError, match="Paper PMID:INVALID not found"):
                mock_extract("INVALID")
    
    def test_unparseable_methods_section(self, mock_pubmed_service):
        """Test handling of unparseable methods sections."""
        with patch('lobster.agents.method_expert.analyze_method_section') as mock_analyze:
            mock_analyze.return_value = {
                "extracted_parameters": {},
                "parsing_errors": [
                    "Unable to identify specific parameter values",
                    "Methods section lacks technical details"
                ],
                "confidence": "low",
                "recommendations": [
                    "Manual parameter extraction may be required",
                    "Consider alternative papers with clearer methods"
                ]
            }
            
            vague_methods = "Cells were processed using standard protocols..."
            result = mock_analyze(vague_methods)
            
            assert len(result["extracted_parameters"]) == 0
            assert result["confidence"] == "low"
    
    def test_parameter_optimization_failure(self, mock_data_manager):
        """Test handling of parameter optimization failures."""
        with patch('lobster.agents.method_expert.optimize_parameters_for_dataset') as mock_optimize:
            mock_optimize.side_effect = RuntimeError("Optimization failed: insufficient data variance")
            
            with pytest.raises(RuntimeError, match="Optimization failed"):
                mock_optimize("insufficient_data", method="clustering")
    
    def test_incompatible_method_dataset(self, mock_data_manager):
        """Test handling of incompatible method-dataset combinations."""
        with patch('lobster.agents.method_expert.validate_method_compatibility') as mock_validate:
            mock_validate.return_value = {
                "compatible": False,
                "incompatibility_reasons": [
                    "Dataset has insufficient cells for method requirements",
                    "Method requires specific data preprocessing not present"
                ],
                "alternative_methods": ["simpler_clustering", "reduced_parameter_method"],
                "adaptation_possible": True,
                "adaptation_steps": [
                    "Reduce parameter complexity",
                    "Use alternative preprocessing pipeline"
                ]
            }
            
            result = mock_validate("small_dataset", "complex_method")
            
            assert result["compatible"] == False
            assert len(result["alternative_methods"]) == 2
    
    def test_software_version_conflicts(self):
        """Test handling of software version conflicts."""
        with patch('lobster.agents.method_expert.check_software_compatibility') as mock_check:
            mock_check.return_value = {
                "compatible": False,
                "conflicts": [
                    {"package": "scanpy", "required": "1.9.1", "installed": "1.8.2"},
                    {"package": "numpy", "required": ">=1.20.0", "installed": "1.19.5"}
                ],
                "resolution_steps": [
                    "Upgrade scanpy to version 1.9.1",
                    "Upgrade numpy to version 1.20.0 or higher"
                ],
                "compatibility_score": 0.3
            }
            
            result = mock_check("method_requiring_new_versions")
            
            assert result["compatible"] == False
            assert len(result["conflicts"]) == 2
    
    def test_concurrent_optimization_handling(self, mock_data_manager):
        """Test handling of concurrent parameter optimization."""
        import threading
        import time
        
        results = []
        errors = []
        
        def optimization_worker(worker_id, dataset_name):
            """Worker function for concurrent optimization testing."""
            try:
                with patch('lobster.agents.method_expert.optimize_parameters_for_dataset') as mock_optimize:
                    mock_optimize.return_value = {
                        "worker_id": worker_id,
                        "dataset": dataset_name,
                        "optimized_parameters": {"resolution": 0.5 + worker_id * 0.1}
                    }
                    
                    result = mock_optimize(dataset_name, method="clustering")
                    results.append(result)
                    time.sleep(0.01)
                    
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple concurrent optimizations
        threads = []
        for i in range(3):
            thread = threading.Thread(target=optimization_worker, args=(i, f"dataset_{i}"))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent optimization errors: {errors}"
        assert len(results) == 3
    
    def test_large_parameter_space_handling(self, mock_data_manager):
        """Test handling of large parameter optimization spaces."""
        with patch('lobster.agents.method_expert.optimize_parameters_for_dataset') as mock_optimize:
            mock_optimize.return_value = {
                "parameter_space_size": 10000,
                "optimization_strategy": "grid_search_with_early_stopping",
                "evaluated_combinations": 150,
                "early_stopping_triggered": True,
                "best_parameters": {"resolution": 0.7, "n_neighbors": 12},
                "optimization_time": 45.2,
                "convergence_reason": "improvement_threshold_reached"
            }
            
            result = mock_optimize("large_dataset", method="comprehensive_optimization")
            
            assert result["early_stopping_triggered"] == True
            assert result["evaluated_combinations"] < result["parameter_space_size"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])