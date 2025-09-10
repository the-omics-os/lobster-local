"""
Comprehensive pytest configuration and fixtures for Lobster AI testing framework.

This module provides all core fixtures, mock configurations, and test utilities
needed for testing the multi-agent bioinformatics analysis platform.
"""

import os
import tempfile
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Generator
from unittest.mock import Mock, MagicMock, patch
import json
import numpy as np
import pandas as pd
import anndata as ad
from datetime import datetime

import pytest
from pytest_mock import MockerFixture
import responses
from faker import Faker

# Suppress warnings during testing
logging.getLogger("scanpy").setLevel(logging.ERROR)
logging.getLogger("anndata").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# Initialize faker for generating test data
fake = Faker()
Faker.seed(42)  # For reproducible test data

# Test constants
TEST_WORKSPACE_PREFIX = "lobster_test_"
DEFAULT_TEST_TIMEOUT = 300
MOCK_API_BASE_URL = "https://api.mock-lobster.test"

# Configure pytest plugins
pytest_plugins = [
    "pytest_mock",
    "pytest_benchmark",
    "pytest_html",
]


# ==============================================================================
# Pytest Configuration Hooks
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers",
        "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", 
        "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers",
        "system: mark test as a system test"
    )
    config.addinivalue_line(
        "markers",
        "performance: mark test as a performance benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Auto-mark based on test path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "system" in str(item.fspath):
            item.add_marker(pytest.mark.system)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)


# ==============================================================================
# Core Infrastructure Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Global test configuration."""
    return {
        "workspace_prefix": TEST_WORKSPACE_PREFIX,
        "timeout": DEFAULT_TEST_TIMEOUT,
        "mock_api_url": MOCK_API_BASE_URL,
        "enable_logging": False,
        "cleanup_workspaces": True,
        "synthetic_data_seed": 42,
        "default_cell_count": 1000,
        "default_gene_count": 2000,
    }


@pytest.fixture(scope="function")
def temp_workspace(test_config: Dict[str, Any]) -> Generator[Path, None, None]:
    """Create isolated temporary workspace for each test."""
    workspace_path = Path(tempfile.mkdtemp(prefix=test_config["workspace_prefix"]))
    
    # Create standard workspace structure
    (workspace_path / "data").mkdir(exist_ok=True)
    (workspace_path / "exports").mkdir(exist_ok=True)
    (workspace_path / "cache").mkdir(exist_ok=True)
    
    try:
        yield workspace_path
    finally:
        # Cleanup workspace after test
        if test_config["cleanup_workspaces"] and workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)


@pytest.fixture(scope="function") 
def isolated_environment(temp_workspace: Path, monkeypatch):
    """Create completely isolated environment for testing."""
    # Set temporary workspace as working directory
    original_cwd = os.getcwd()
    monkeypatch.chdir(temp_workspace)
    
    # Mock environment variables
    test_env = {
        "LOBSTER_WORKSPACE": str(temp_workspace),
        "OPENAI_API_KEY": "test-openai-key",
        "AWS_BEDROCK_ACCESS_KEY": "test-aws-access-key",
        "AWS_BEDROCK_SECRET_ACCESS_KEY": "test-aws-secret-key",
        "NCBI_API_KEY": "test-ncbi-key",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    yield temp_workspace
    
    # Restore original working directory
    os.chdir(original_cwd)


# ==============================================================================
# Mock Data Generation Fixtures
# ==============================================================================

@pytest.fixture(scope="function")
def synthetic_single_cell_data(test_config: Dict[str, Any]) -> ad.AnnData:
    """Generate realistic synthetic single-cell RNA-seq data."""
    n_obs = test_config["default_cell_count"]
    n_vars = test_config["default_gene_count"]
    
    # Set random seed for reproducibility
    np.random.seed(test_config["synthetic_data_seed"])
    
    # Generate count matrix with negative binomial distribution
    # Simulate realistic single-cell count distributions
    X = np.random.negative_binomial(n=5, p=0.3, size=(n_obs, n_vars)).astype(np.float32)
    
    # Add some zeros to make it realistic (sparse)
    zero_mask = np.random.random((n_obs, n_vars)) < 0.7
    X[zero_mask] = 0
    
    # Create gene names
    var_names = [f"Gene_{i:04d}" for i in range(n_vars)]
    
    # Create cell barcodes
    obs_names = [f"Cell_{fake.uuid4()[:8]}" for _ in range(n_obs)]
    
    # Create AnnData object
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=var_names), obs=pd.DataFrame(index=obs_names))
    
    # Add realistic metadata
    adata.var["gene_ids"] = [f"ENSG{i:011d}" for i in range(n_vars)]
    adata.var["feature_types"] = ["Gene Expression"] * n_vars
    adata.var["chromosome"] = np.random.choice(
        [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"], 
        size=n_vars
    )
    
    # Add cell metadata
    adata.obs["total_counts"] = np.array(X.sum(axis=1))
    adata.obs["n_genes_by_counts"] = np.array((X > 0).sum(axis=1))
    adata.obs["pct_counts_mt"] = np.random.uniform(0, 30, n_obs)  # Mitochondrial gene percentage
    adata.obs["pct_counts_ribo"] = np.random.uniform(0, 50, n_obs)  # Ribosomal gene percentage
    
    # Add simulated cell types
    cell_types = ["T_cell", "B_cell", "NK_cell", "Monocyte", "Dendritic_cell"]
    adata.obs["cell_type"] = np.random.choice(cell_types, size=n_obs)
    
    # Add batch information
    adata.obs["batch"] = np.random.choice(["Batch1", "Batch2", "Batch3"], size=n_obs)
    
    return adata


@pytest.fixture(scope="function")
def synthetic_bulk_rnaseq_data(test_config: Dict[str, Any]) -> ad.AnnData:
    """Generate realistic synthetic bulk RNA-seq data."""
    n_obs = 24  # Typical sample count for bulk RNA-seq
    n_vars = test_config["default_gene_count"]
    
    np.random.seed(test_config["synthetic_data_seed"])
    
    # Generate count matrix with higher counts than single-cell
    X = np.random.negative_binomial(n=20, p=0.1, size=(n_obs, n_vars)).astype(np.float32)
    
    # Create sample and gene names
    obs_names = [f"Sample_{i:02d}" for i in range(n_obs)]
    var_names = [f"Gene_{i:04d}" for i in range(n_vars)]
    
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
    
    # Add realistic bulk RNA-seq metadata
    adata.var["gene_ids"] = [f"ENSG{i:011d}" for i in range(n_vars)]
    adata.var["gene_name"] = [f"GENE{i}" for i in range(n_vars)]
    adata.var["biotype"] = np.random.choice(
        ["protein_coding", "lncRNA", "miRNA", "pseudogene"], 
        size=n_vars, 
        p=[0.7, 0.15, 0.05, 0.1]
    )
    
    # Add sample metadata
    adata.obs["condition"] = ["Treatment"] * 12 + ["Control"] * 12
    adata.obs["batch"] = (["Batch1"] * 6 + ["Batch2"] * 6) * 2
    adata.obs["sex"] = np.random.choice(["M", "F"], size=n_obs)
    adata.obs["age"] = np.random.randint(20, 80, size=n_obs)
    
    return adata


@pytest.fixture(scope="function") 
def synthetic_proteomics_data(test_config: Dict[str, Any]) -> ad.AnnData:
    """Generate realistic synthetic proteomics data."""
    n_obs = 48  # Typical proteomics sample count
    n_vars = 500  # Typical protein count
    
    np.random.seed(test_config["synthetic_data_seed"])
    
    # Generate intensity matrix with log-normal distribution
    X = np.random.lognormal(mean=10, sigma=2, size=(n_obs, n_vars)).astype(np.float32)
    
    # Add missing values (common in proteomics)
    missing_mask = np.random.random((n_obs, n_vars)) < 0.2
    X[missing_mask] = np.nan
    
    obs_names = [f"Sample_{i:03d}" for i in range(n_obs)]
    var_names = [f"Protein_{i:03d}" for i in range(n_vars)]
    
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))
    
    # Add protein metadata
    adata.var["protein_ids"] = [f"P{i:05d}" for i in range(n_vars)]
    adata.var["protein_names"] = [f"PROT{i}" for i in range(n_vars)]
    adata.var["molecular_weight"] = np.random.uniform(10, 200, n_vars)
    
    # Add sample metadata
    adata.obs["condition"] = (["Disease"] * 16 + ["Healthy"] * 16 + ["Control"] * 16)
    adata.obs["tissue"] = np.random.choice(["Brain", "Liver", "Kidney"], size=n_obs)
    adata.obs["batch"] = np.random.choice(["Batch1", "Batch2", "Batch3", "Batch4"], size=n_obs)
    
    return adata


@pytest.fixture(scope="function")
def mock_geo_response() -> Dict[str, Any]:
    """Generate mock GEO dataset response."""
    return {
        "gse_id": "GSE123456",
        "title": "Test Single-Cell RNA-seq Dataset",
        "summary": "This is a synthetic dataset for testing purposes",
        "organism": "Homo sapiens",
        "platform": "GPL24676",
        "samples": [
            {
                "gsm_id": "GSM1234567",
                "title": "Sample 1",
                "characteristics": {
                    "cell type": "T cell",
                    "tissue": "PBMC",
                    "treatment": "Control"
                }
            },
            {
                "gsm_id": "GSM1234568", 
                "title": "Sample 2",
                "characteristics": {
                    "cell type": "B cell",
                    "tissue": "PBMC",
                    "treatment": "Treatment"
                }
            }
        ],
        "supplementary_files": [
            "GSE123456_matrix.mtx.gz",
            "GSE123456_features.tsv.gz", 
            "GSE123456_barcodes.tsv.gz"
        ]
    }


# ==============================================================================
# Core Component Mocks
# ==============================================================================

@pytest.fixture(scope="function")
def mock_data_manager_v2(temp_workspace: Path) -> Mock:
    """Mock DataManagerV2 with realistic behavior."""
    mock_dm = Mock()
    
    # Mock basic properties
    mock_dm.workspace_path = temp_workspace
    mock_dm.modalities = {}
    mock_dm.metadata_store = {}
    mock_dm.latest_plots = []
    mock_dm.tool_usage_history = []
    
    # Mock methods
    mock_dm.list_modalities.return_value = list(mock_dm.modalities.keys())
    mock_dm.get_modality.side_effect = lambda name: mock_dm.modalities.get(name)
    mock_dm.add_modality.side_effect = lambda name, data: mock_dm.modalities.update({name: data})
    mock_dm.remove_modality.side_effect = lambda name: mock_dm.modalities.pop(name, None)
    
    # Mock file operations
    mock_dm.save_modality.return_value = True
    mock_dm.load_modality.return_value = True
    mock_dm.export_workspace.return_value = temp_workspace / "export.zip"
    
    return mock_dm


@pytest.fixture(scope="function")
def mock_agent_client(temp_workspace: Path) -> Mock:
    """Mock AgentClient for testing agent interactions."""
    mock_client = Mock()
    
    # Mock basic properties
    mock_client.session_id = f"test_session_{fake.uuid4()[:8]}"
    mock_client.workspace_path = temp_workspace
    
    # Mock query method with realistic responses
    def mock_query(user_input: str, stream: bool = False):
        return {
            "success": True,
            "response": f"Mock response to: {user_input[:50]}...",
            "agent_used": "supervisor_agent",
            "execution_time": 1.23,
            "tools_used": ["list_available_modalities"]
        }
    
    mock_client.query.side_effect = mock_query
    
    # Mock status method
    mock_client.get_status.return_value = {
        "session_id": mock_client.session_id,
        "workspace_path": str(temp_workspace),
        "active_modalities": 0,
        "total_interactions": 0,
        "last_activity": datetime.now().isoformat()
    }
    
    return mock_client


@pytest.fixture(scope="function")
def mock_llm_responses(mocker: MockerFixture) -> Mock:
    """Mock LLM API responses for consistent agent testing."""
    mock_responses = {
        "supervisor": "I understand your request. Let me delegate this to the appropriate expert agent.",
        "data_expert": "I can help you load and analyze your dataset. Let me check the data format.",
        "singlecell_expert": "I'll perform single-cell RNA-seq analysis including QC, normalization, and clustering.",
        "research_agent": "I can search for relevant datasets and literature for your research question.",
        "method_expert": "I'll extract optimal parameters from recent publications for your analysis."
    }
    
    # Mock OpenAI API calls
    mock_openai = mocker.patch("openai.resources.chat.completions.Completions.create")
    mock_openai.return_value.choices = [
        Mock(message=Mock(content=mock_responses["supervisor"]))
    ]
    
    # Mock AWS Bedrock calls
    mock_bedrock = mocker.patch("boto3.client")
    mock_bedrock.return_value.invoke_model.return_value = {
        "body": Mock(read=lambda: json.dumps({
            "content": [{"text": mock_responses["supervisor"]}]
        }).encode())
    }
    
    return mock_openai


# ==============================================================================
# External Service Mocks
# ==============================================================================

@pytest.fixture(scope="function")
def mock_geo_service(mocker: MockerFixture) -> Mock:
    """Mock GEO service for testing data download."""
    mock_service = Mock()
    
    # Mock successful download
    mock_service.download_gse.return_value = {
        "success": True,
        "gse_id": "GSE123456",
        "files_downloaded": 3,
        "local_path": "/mock/path/GSE123456"
    }
    
    # Mock GEO metadata fetch
    mock_service.get_gse_metadata.return_value = {
        "gse_id": "GSE123456",
        "title": "Test Dataset",
        "organism": "Homo sapiens",
        "sample_count": 24,
        "platform": "GPL24676"
    }
    
    return mock_service


@pytest.fixture(scope="function")
def mock_external_apis():
    """Mock external API calls using responses library."""
    with responses.RequestsMock() as rsps:
        # Mock NCBI/GEO API
        rsps.add(
            responses.GET,
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            json={"esearchresult": {"idlist": ["123456"], "count": "1"}},
            status=200
        )
        
        # Mock PubMed API
        rsps.add(
            responses.GET,
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            xml='<?xml version="1.0"?><PubmedArticle></PubmedArticle>',
            status=200
        )
        
        yield rsps


# ==============================================================================
# Performance Testing Fixtures
# ==============================================================================

@pytest.fixture(scope="function")
def benchmark_config() -> Dict[str, Any]:
    """Configuration for performance benchmarking."""
    return {
        "min_rounds": 3,
        "max_time": 10.0,
        "timer": "time.perf_counter",
        "disable_gc": True,
        "warmup": True
    }


# ==============================================================================
# Test Utilities
# ==============================================================================

@pytest.fixture(scope="session")
def test_data_registry() -> Dict[str, str]:
    """Registry of test data files and their descriptions."""
    return {
        "small_single_cell": "Small single-cell dataset (100 cells, 500 genes)",
        "medium_single_cell": "Medium single-cell dataset (1000 cells, 2000 genes)",
        "large_single_cell": "Large single-cell dataset (10000 cells, 5000 genes)",
        "bulk_rnaseq": "Bulk RNA-seq dataset (24 samples, 2000 genes)",
        "proteomics": "Proteomics dataset (48 samples, 500 proteins)",
        "multimodal": "Multi-modal dataset (single-cell + proteomics)"
    }


def create_mock_file(file_path: Path, content: str = "") -> Path:
    """Utility function to create mock files for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path


def assert_adata_equal(adata1: ad.AnnData, adata2: ad.AnnData, 
                      check_dtype: bool = True) -> None:
    """Assert that two AnnData objects are equal."""
    assert adata1.shape == adata2.shape, "Shape mismatch"
    
    # Check data matrix
    if hasattr(adata1.X, 'toarray'):
        assert np.allclose(adata1.X.toarray(), adata2.X.toarray(), equal_nan=True)
    else:
        assert np.allclose(adata1.X, adata2.X, equal_nan=True)
    
    # Check obs and var
    pd.testing.assert_frame_equal(adata1.obs, adata2.obs)
    pd.testing.assert_frame_equal(adata1.var, adata2.var)


# ==============================================================================
# Cleanup and Finalization
# ==============================================================================

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """Cleanup test artifacts at the end of test session."""
    yield
    
    # Cleanup any remaining temporary files
    temp_dir = Path(tempfile.gettempdir())
    for path in temp_dir.glob(f"{TEST_WORKSPACE_PREFIX}*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)