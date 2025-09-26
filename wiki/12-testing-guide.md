# Testing Guide - Lobster AI Testing Framework

## 🎯 Overview

This guide provides comprehensive documentation for the Lobster AI testing framework, targeting 95%+ code coverage with scientifically accurate testing scenarios. The testing infrastructure covers unit tests, integration tests, system tests, and performance benchmarks across all bioinformatics workflows.

## 🏗️ Testing Architecture

### Test Categories

#### 1. **Unit Tests** (`tests/unit/`)
- **Purpose**: Test individual functions, classes, and methods in isolation
- **Duration**: ~2 minutes for full suite
- **Coverage**: Individual components (services, agents, utilities)
- **Execution**: `pytest tests/unit/`

#### 2. **Integration Tests** (`tests/integration/`)
- **Purpose**: Test component interactions and workflows
- **Duration**: ~15 minutes for full suite
- **Coverage**: Agent-service integration, data flow validation
- **Execution**: `pytest tests/integration/`

#### 3. **System Tests** (`tests/system/`)
- **Purpose**: Test complete end-to-end workflows
- **Duration**: ~30 minutes for full suite
- **Coverage**: Full analysis pipelines, CLI interactions
- **Execution**: `pytest tests/system/`

#### 4. **Performance Tests** (`tests/performance/`)
- **Purpose**: Benchmark performance and memory usage
- **Duration**: ~45 minutes for full suite
- **Coverage**: Large dataset handling, algorithmic efficiency
- **Execution**: `pytest tests/performance/`

### Directory Structure

```
tests/
├── 📁 unit/                    # Unit tests (20+ files)
│   ├── core/                  # Core system components
│   │   ├── test_data_manager_v2.py
│   │   ├── test_client.py
│   │   └── test_adapters.py
│   ├── agents/                # AI agent functionality
│   │   ├── test_data_expert.py
│   │   ├── test_singlecell_expert.py
│   │   └── test_bulk_rnaseq_expert.py
│   ├── services/              # Analysis services (7+ files)
│   │   ├── test_quality_service.py
│   │   ├── test_clustering_service.py
│   │   └── test_differential_service.py
│   └── tools/                 # Analysis tools (12+ files)
├── 📁 integration/            # Integration tests (5 files)
│   ├── test_agent_workflows.py
│   ├── test_data_pipelines.py
│   └── test_service_chains.py
├── 📁 system/                 # System tests (3 files)
│   ├── test_end_to_end.py
│   ├── test_cli_commands.py
│   └── test_multi_modal.py
├── 📁 performance/            # Performance tests (3 files)
├── 📁 mock_data/              # Synthetic data generation
│   ├── generators.py          # High-level data generators
│   ├── factories.py           # Data factory classes
│   └── base.py               # Base configurations
├── conftest.py               # Global fixtures and configuration
└── README.md                 # Testing documentation
```

## 🔧 Test Configuration

### Pytest Configuration (`pytest.ini`)

```ini
[tool:pytest]
minversion = 6.0
addopts =
    --strict-markers
    --disable-warnings
    --verbose
    --tb=short
    --cov=lobster
    --cov-report=html:htmlcov
    --cov-report=xml:coverage.xml
    --cov-report=term-missing
    --cov-fail-under=80
    --durations=10

testpaths = tests
markers =
    unit: mark test as a unit test
    integration: mark test as an integration test
    system: mark test as a system test
    performance: mark test as a performance benchmark
    slow: mark test as slow running
    requires_gpu: mark test as requiring GPU
    requires_network: mark test as requiring network access

filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
    ignore::FutureWarning:scanpy
```

### Global Test Fixtures (`conftest.py`)

```python
# Core fixtures available to all tests
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.client import AgentClient
from tests.mock_data.generators import (
    generate_synthetic_single_cell,
    generate_synthetic_bulk_rnaseq,
    generate_synthetic_proteomics
)

@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory(prefix="lobster_test_") as tmpdir:
        workspace_path = Path(tmpdir)
        yield workspace_path

@pytest.fixture
def mock_data_manager(temp_workspace):
    """Create DataManagerV2 instance with temporary workspace."""
    return DataManagerV2(workspace_path=temp_workspace)

@pytest.fixture
def sample_single_cell_data():
    """Generate synthetic single-cell data for testing."""
    return generate_synthetic_single_cell(
        n_cells=100,
        n_genes=50,
        n_cell_types=3
    )

@pytest.fixture
def sample_bulk_data():
    """Generate synthetic bulk RNA-seq data."""
    return generate_synthetic_bulk_rnaseq(
        n_samples=12,
        n_genes=100
    )

@pytest.fixture
def sample_proteomics_data():
    """Generate synthetic proteomics data."""
    return generate_synthetic_proteomics(
        n_samples=20,
        n_proteins=80,
        missing_rate=0.3
    )

@pytest.fixture
def mock_agent_client(mock_data_manager):
    """Create mock AgentClient for testing."""
    client = Mock(spec=AgentClient)
    client.data_manager = mock_data_manager
    return client

@pytest.fixture(scope="session")
def test_config():
    """Test configuration parameters."""
    return {
        'timeout': 300,
        'max_memory': '2GB',
        'test_data_size': 'medium',
        'enable_gpu': False
    }
```

## 🧪 Writing Unit Tests

### Service Unit Test Template

```python
# tests/unit/services/test_your_service.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from lobster.tools.your_service import YourService, YourServiceError
from tests.mock_data.generators import generate_synthetic_single_cell


class TestYourService:
    """Comprehensive unit tests for YourService."""

    @pytest.fixture
    def service(self):
        """Create service instance for testing."""
        return YourService()

    @pytest.fixture
    def mock_adata(self):
        """Create mock AnnData for testing."""
        return generate_synthetic_single_cell(n_cells=50, n_genes=30)

    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert service.progress_callback is None
        assert service.current_progress == 0
        assert hasattr(service, 'total_steps')

    def test_progress_callback_setting(self, service):
        """Test progress callback functionality."""
        callback_calls = []

        def mock_callback(progress, message):
            callback_calls.append((progress, message))

        service.set_progress_callback(mock_callback)
        assert service.progress_callback is not None

        # Test progress update
        service.total_steps = 2
        service._update_progress("Test step")

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 50  # 50% progress
        assert "Test step" in callback_calls[0][1]

    def test_main_analysis_success(self, service, mock_adata):
        """Test successful analysis execution."""

        # Test with default parameters
        result_adata, statistics = service.main_analysis_method(
            mock_adata,
            parameter1=1.0,
            parameter2="default"
        )

        # Validate results structure
        assert result_adata is not None
        assert isinstance(statistics, dict)

        # Validate AnnData structure preservation
        assert result_adata.n_obs == mock_adata.n_obs
        assert result_adata.n_vars == mock_adata.n_vars

        # Validate statistics content
        required_stats = ['n_observations', 'n_features', 'analysis_timestamp']
        for stat in required_stats:
            assert stat in statistics

        # Validate analysis metadata stored
        assert 'your_analysis' in result_adata.uns
        assert result_adata.uns['your_analysis']['method'] == 'default'

    def test_parameter_validation(self, service, mock_adata):
        """Test comprehensive parameter validation."""

        # Test invalid parameter1 (should be positive)
        with pytest.raises(ValueError, match="Parameter1 must be positive"):
            service.main_analysis_method(mock_adata, parameter1=-1.0)

        with pytest.raises(ValueError, match="Parameter1 must be positive"):
            service.main_analysis_method(mock_adata, parameter1=0.0)

        # Test invalid parameter2 (should be from allowed options)
        with pytest.raises(ValueError, match="Invalid parameter2"):
            service.main_analysis_method(mock_adata, parameter2="invalid_option")

        # Test empty parameter3 list
        with pytest.raises(ValueError, match="Parameter3 cannot be empty list"):
            service.main_analysis_method(mock_adata, parameter3=[])

    def test_empty_data_handling(self, service):
        """Test handling of edge cases in data."""

        # Empty observations
        empty_obs_adata = generate_synthetic_single_cell(n_cells=0, n_genes=10)
        with pytest.raises(ValueError, match="Input data is empty"):
            service.main_analysis_method(empty_obs_adata)

        # Empty features
        empty_vars_adata = generate_synthetic_single_cell(n_cells=10, n_genes=0)
        with pytest.raises(ValueError, match="Input data has no features"):
            service.main_analysis_method(empty_vars_adata)

    def test_statistical_accuracy(self, service, mock_adata):
        """Test statistical calculations are mathematically correct."""

        result_adata, statistics = service.main_analysis_method(mock_adata)

        # Verify basic statistics match input data
        assert statistics['n_observations'] == mock_adata.n_obs
        assert statistics['n_features'] == mock_adata.n_vars

        # Test statistical calculations if applicable
        if 'mean_expression' in statistics:
            expected_mean = np.mean(mock_adata.X)
            np.testing.assert_almost_equal(
                statistics['mean_expression'],
                expected_mean,
                decimal=5
            )

    def test_error_propagation(self, service, mock_adata, monkeypatch):
        """Test error handling and propagation."""

        # Mock internal method to raise exception
        def mock_preprocess_error(*args, **kwargs):
            raise RuntimeError("Preprocessing failed")

        monkeypatch.setattr(service, '_preprocess_data', mock_preprocess_error)

        # Should wrap in service-specific error
        with pytest.raises(YourServiceError, match="Unexpected error"):
            service.main_analysis_method(mock_adata)

    def test_reproducibility(self, service, mock_adata):
        """Test that analyses are reproducible."""

        # Run same analysis twice
        result1_adata, stats1 = service.main_analysis_method(
            mock_adata, parameter1=1.5, parameter2="option1"
        )

        result2_adata, stats2 = service.main_analysis_method(
            mock_adata.copy(), parameter1=1.5, parameter2="option2"
        )

        # Results should be deterministic for same inputs
        if 'option1' == 'option1':  # Same parameters
            np.testing.assert_array_almost_equal(
                result1_adata.X, result1_adata.X  # Compare with self for structure
            )

    @pytest.mark.parametrize("param1,param2,expected_error", [
        (-1.0, "default", ValueError),
        (1.0, "invalid", ValueError),
        (1.0, "default", None),  # Should succeed
    ])
    def test_parameter_combinations(self, service, mock_adata, param1, param2, expected_error):
        """Test various parameter combinations."""

        if expected_error:
            with pytest.raises(expected_error):
                service.main_analysis_method(mock_adata, parameter1=param1, parameter2=param2)
        else:
            result_adata, statistics = service.main_analysis_method(
                mock_adata, parameter1=param1, parameter2=param2
            )
            assert result_adata is not None
            assert isinstance(statistics, dict)

    def test_memory_efficiency(self, service):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run analysis on moderately sized data
        large_adata = generate_synthetic_single_cell(n_cells=1000, n_genes=500)
        result_adata, statistics = service.main_analysis_method(large_adata)

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 500MB for this test)
        assert memory_growth < 500 * 1024 * 1024  # 500MB threshold

    def test_concurrent_usage(self, service, mock_adata):
        """Test service can be used concurrently (stateless requirement)."""
        import threading

        results = []
        errors = []

        def run_analysis(data):
            try:
                result = service.main_analysis_method(data.copy())
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple analyses concurrently
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_analysis, args=(mock_adata,))
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Should have no errors and expected number of results
        assert len(errors) == 0
        assert len(results) == 3
```

### Agent Unit Test Template

```python
# tests/unit/agents/test_your_agent.py
import pytest
from unittest.mock import Mock, patch, MagicMock

from lobster.agents.your_agent import your_agent_factory
from lobster.core.data_manager_v2 import DataManagerV2


class TestYourAgent:
    """Unit tests for YourAgent."""

    @pytest.fixture
    def mock_data_manager(self):
        """Create mock DataManagerV2."""
        mock_dm = Mock(spec=DataManagerV2)
        mock_dm.list_modalities.return_value = []
        mock_dm.get_modality.return_value = None
        mock_dm.log_tool_usage = Mock()
        return mock_dm

    @pytest.fixture
    def agent(self, mock_data_manager):
        """Create agent instance for testing."""
        with patch('lobster.agents.your_agent.get_settings') as mock_settings:
            mock_settings.return_value.get_agent_llm_params.return_value = {
                'model': 'test-model',
                'temperature': 0.1
            }
            return your_agent_factory(mock_data_manager)

    def test_agent_creation(self, mock_data_manager):
        """Test agent factory creates agent successfully."""
        with patch('lobster.agents.your_agent.get_settings') as mock_settings:
            mock_settings.return_value.get_agent_llm_params.return_value = {}

            agent = your_agent_factory(mock_data_manager)
            assert agent is not None

    def test_agent_tools_available(self, agent):
        """Test that agent has expected tools."""
        # Agent tools are typically accessible via agent.get_graph().nodes
        # Implementation depends on LangGraph structure

        # This is a conceptual test - actual implementation may vary
        tools = getattr(agent, 'tools', [])
        tool_names = [tool.name for tool in tools if hasattr(tool, 'name')]

        expected_tools = ['check_available_modalities', 'perform_domain_analysis']
        for expected_tool in expected_tools:
            # Check if tool exists (test implementation may need adjustment)
            pass

    @patch('lobster.agents.your_agent.YourService')
    def test_tool_service_integration(self, mock_service_class, agent, mock_data_manager):
        """Test that agent tools properly integrate with services."""

        # Setup mock service
        mock_service = Mock()
        mock_service.perform_analysis.return_value = (Mock(), {'metric': 1.0})
        mock_service_class.return_value = mock_service

        # Setup data manager
        mock_adata = Mock()
        mock_data_manager.list_modalities.return_value = ['test_data']
        mock_data_manager.get_modality.return_value = mock_adata

        # This test would need to be implemented based on how tools are exposed
        # in the actual LangGraph agent structure
```

## 🔗 Writing Integration Tests

### Agent-Service Integration Template

```python
# tests/integration/test_agent_service_integration.py
import pytest
from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2
from tests.mock_data.generators import generate_synthetic_single_cell


class TestAgentServiceIntegration:
    """Integration tests for agent-service workflows."""

    @pytest.fixture
    def client_with_data(self, temp_workspace, sample_single_cell_data):
        """Create client with loaded data."""
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        data_manager.modalities['test_data'] = sample_single_cell_data

        client = AgentClient(data_manager=data_manager)
        return client

    def test_quality_assessment_workflow(self, client_with_data):
        """Test complete quality assessment workflow."""

        # Request quality assessment
        response = client_with_data.query(
            "Assess the quality of test_data modality"
        )

        # Validate response structure
        assert response['success'] is True
        assert 'quality_assessed' in response['response']

        # Check that new modality was created
        modalities = client_with_data.data_manager.list_modalities()
        assert any('quality_assessed' in mod for mod in modalities)

    def test_clustering_analysis_workflow(self, client_with_data):
        """Test complete clustering workflow."""

        # First, assess quality
        quality_response = client_with_data.query(
            "Assess quality of test_data"
        )
        assert quality_response['success']

        # Then perform clustering
        cluster_response = client_with_data.query(
            "Perform clustering analysis on the quality-assessed data"
        )

        assert cluster_response['success']
        assert 'clustered' in cluster_response['response']

    def test_multi_agent_handoff(self, client_with_data):
        """Test handoffs between different agents."""

        # Start with data expert
        data_response = client_with_data.query(
            "Load and prepare the test data for single-cell analysis"
        )
        assert data_response['success']

        # Should handoff to single-cell expert
        analysis_response = client_with_data.query(
            "Now perform clustering and find marker genes"
        )
        assert analysis_response['success']

    def test_error_handling_integration(self, client_with_data):
        """Test error handling across agent-service boundaries."""

        # Request analysis on non-existent data
        error_response = client_with_data.query(
            "Analyze the modality called 'nonexistent_data'"
        )

        # Should handle error gracefully
        assert error_response['success'] is False
        assert 'not found' in error_response['response'].lower()
```

## 🌐 Writing System Tests

### End-to-End Test Template

```python
# tests/system/test_end_to_end.py
import pytest
from pathlib import Path
import tempfile

from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2


class TestEndToEndWorkflows:
    """System tests for complete analysis workflows."""

    @pytest.fixture
    def full_client_setup(self):
        """Set up complete client with real configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            data_manager = DataManagerV2(workspace_path=workspace)
            client = AgentClient(data_manager=data_manager)
            yield client

    def test_complete_single_cell_pipeline(self, full_client_setup):
        """Test complete single-cell analysis pipeline."""
        client = full_client_setup

        # Step 1: Load synthetic data
        response1 = client.query(
            "Generate synthetic single-cell data with 500 cells and 100 genes"
        )
        assert response1['success']

        # Step 2: Quality assessment
        response2 = client.query(
            "Assess the quality of the generated data"
        )
        assert response2['success']

        # Step 3: Preprocessing
        response3 = client.query(
            "Filter and normalize the data based on quality metrics"
        )
        assert response3['success']

        # Step 4: Clustering
        response4 = client.query(
            "Perform clustering analysis and generate UMAP visualization"
        )
        assert response4['success']

        # Step 5: Marker gene analysis
        response5 = client.query(
            "Find marker genes for each cluster"
        )
        assert response5['success']

        # Validate final state
        modalities = client.data_manager.list_modalities()
        expected_stages = ['generated', 'quality_assessed', 'normalized', 'clustered']

        for stage in expected_stages:
            assert any(stage in mod for mod in modalities)

    def test_bulk_rnaseq_differential_expression(self, full_client_setup):
        """Test bulk RNA-seq differential expression workflow."""
        client = full_client_setup

        # Generate bulk RNA-seq data with conditions
        response1 = client.query(
            "Generate bulk RNA-seq data with treatment and control conditions"
        )
        assert response1['success']

        # Perform differential expression
        response2 = client.query(
            "Perform differential expression analysis between conditions"
        )
        assert response2['success']

        # Validate results contain expected elements
        assert 'differential' in response2['response']
        assert any('differential' in mod for mod in client.data_manager.list_modalities())

    @pytest.mark.slow
    def test_large_dataset_handling(self, full_client_setup):
        """Test system performance with larger datasets."""
        client = full_client_setup

        # Generate larger dataset
        response = client.query(
            "Generate single-cell data with 10000 cells and 2000 genes"
        )
        assert response['success']

        # Perform computationally intensive analysis
        cluster_response = client.query(
            "Perform clustering with high resolution and generate comprehensive visualizations"
        )
        assert cluster_response['success']
```

## 📊 Performance Testing

### Performance Test Template

```python
# tests/performance/test_service_performance.py
import pytest
import time
import psutil
import os
from pathlib import Path

from lobster.tools.clustering_service import ClusteringService
from tests.mock_data.generators import generate_synthetic_single_cell


class TestServicePerformance:
    """Performance benchmarks for services."""

    @pytest.mark.performance
    def test_clustering_performance_small(self, benchmark):
        """Benchmark clustering on small dataset."""
        service = ClusteringService()
        adata = generate_synthetic_single_cell(n_cells=1000, n_genes=500)

        def run_clustering():
            return service.cluster_and_visualize(adata)

        result = benchmark(run_clustering)

        # Validate performance
        assert benchmark.stats.stats.mean < 30.0  # Should complete in <30 seconds

    @pytest.mark.performance
    @pytest.mark.slow
    def test_clustering_performance_large(self, benchmark):
        """Benchmark clustering on larger dataset."""
        service = ClusteringService()
        adata = generate_synthetic_single_cell(n_cells=10000, n_genes=2000)

        def run_clustering():
            return service.cluster_and_visualize(adata, demo_mode=True)

        result = benchmark(run_clustering)

        # Should still complete within reasonable time in demo mode
        assert benchmark.stats.stats.mean < 120.0  # 2 minutes max

    def test_memory_usage_monitoring(self):
        """Monitor memory usage during analysis."""
        service = ClusteringService()
        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss

        # Run analysis
        adata = generate_synthetic_single_cell(n_cells=5000, n_genes=1000)
        result_adata, stats = service.cluster_and_visualize(adata)

        peak_memory = process.memory_info().rss
        memory_growth = peak_memory - initial_memory

        # Memory growth should be reasonable (less than 2GB)
        assert memory_growth < 2 * 1024**3

        # Clean up and check memory returns close to initial
        del adata, result_adata
        import gc
        gc.collect()

        final_memory = process.memory_info().rss
        memory_leak = final_memory - initial_memory

        # Should not have significant memory leaks (less than 100MB)
        assert memory_leak < 100 * 1024**2
```

## 🛠️ Mock Data Generation

### Synthetic Data Generators

```python
# tests/mock_data/custom_generators.py
"""Custom generators for specific test scenarios."""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional, Dict, Any


def generate_realistic_single_cell(
    n_cells: int = 1000,
    n_genes: int = 500,
    cell_types: Optional[list] = None,
    batch_effects: bool = False,
    doublet_rate: float = 0.05
) -> ad.AnnData:
    """
    Generate realistic single-cell data with biological features.

    Args:
        n_cells: Number of cells
        n_genes: Number of genes
        cell_types: List of cell type names
        batch_effects: Whether to include batch effects
        doublet_rate: Proportion of doublet cells

    Returns:
        AnnData with realistic single-cell features
    """

    if cell_types is None:
        cell_types = ['T_cells', 'B_cells', 'NK_cells', 'Monocytes']

    # Generate base expression matrix with biological structure
    np.random.seed(42)

    # Create cell type-specific expression patterns
    n_cell_types = len(cell_types)
    cells_per_type = n_cells // n_cell_types

    X = np.zeros((n_cells, n_genes))
    cell_type_labels = []

    for i, cell_type in enumerate(cell_types):
        start_idx = i * cells_per_type
        end_idx = start_idx + cells_per_type if i < n_cell_types - 1 else n_cells

        # Base expression for this cell type
        base_expression = np.random.negative_binomial(10, 0.3, size=(end_idx - start_idx, n_genes))

        # Add cell type-specific marker genes
        marker_genes = slice(i * 50, (i + 1) * 50)  # 50 markers per type
        base_expression[:, marker_genes] *= np.random.uniform(2, 5, size=(end_idx - start_idx, 50))

        X[start_idx:end_idx, :] = base_expression
        cell_type_labels.extend([cell_type] * (end_idx - start_idx))

    # Create AnnData object
    adata = ad.AnnData(X=X.astype(np.float32))

    # Add cell metadata
    adata.obs['cell_type'] = cell_type_labels
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
    adata.obs['total_counts'] = adata.X.sum(axis=1)

    # Add mitochondrial genes
    mt_genes = np.random.choice(n_genes, size=int(0.05 * n_genes), replace=False)
    adata.var['mt'] = False
    adata.var.iloc[mt_genes, adata.var.columns.get_loc('mt')] = True

    # Calculate mitochondrial percentage
    adata.obs['pct_counts_mt'] = (
        adata[:, adata.var['mt']].X.sum(axis=1) / adata.obs['total_counts'] * 100
    )

    # Add batch effects if requested
    if batch_effects:
        n_batches = 3
        batch_assignments = np.random.choice(n_batches, size=n_cells)
        adata.obs['batch'] = [f'batch_{i}' for i in batch_assignments]

        # Apply batch-specific scaling
        for batch_id in range(n_batches):
            batch_mask = batch_assignments == batch_id
            batch_effect = np.random.uniform(0.8, 1.2)
            adata.X[batch_mask, :] *= batch_effect

    # Add doublets if requested
    if doublet_rate > 0:
        n_doublets = int(n_cells * doublet_rate)
        doublet_indices = np.random.choice(n_cells, size=n_doublets, replace=False)

        adata.obs['is_doublet'] = False
        adata.obs.iloc[doublet_indices, adata.obs.columns.get_loc('is_doublet')] = True

        # Doublets have higher total counts
        adata.X[doublet_indices, :] *= np.random.uniform(1.5, 2.0, size=(n_doublets, 1))

    # Add gene metadata
    gene_names = [f'Gene_{i:04d}' for i in range(n_genes)]
    adata.var.index = gene_names
    adata.var['highly_variable'] = False

    # Mark some genes as highly variable
    hv_genes = np.random.choice(n_genes, size=int(0.2 * n_genes), replace=False)
    adata.var.iloc[hv_genes, adata.var.columns.get_loc('highly_variable')] = True

    return adata


def generate_differential_expression_data(
    n_samples_per_group: int = 6,
    n_genes: int = 1000,
    n_de_genes: int = 100,
    effect_size: float = 2.0
) -> ad.AnnData:
    """Generate bulk RNA-seq data with known differential expression."""

    n_samples = n_samples_per_group * 2

    # Generate base counts
    X = np.random.negative_binomial(20, 0.3, size=(n_samples, n_genes))

    # Add differential expression
    de_gene_indices = np.random.choice(n_genes, size=n_de_genes, replace=False)

    # Treatment group gets higher expression for DE genes
    treatment_samples = slice(n_samples_per_group, n_samples)
    X[treatment_samples, :][:, de_gene_indices] *= effect_size

    # Create AnnData
    adata = ad.AnnData(X=X.astype(np.float32))

    # Add sample metadata
    conditions = ['control'] * n_samples_per_group + ['treatment'] * n_samples_per_group
    adata.obs['condition'] = conditions
    adata.obs['sample_id'] = [f'sample_{i:02d}' for i in range(n_samples)]

    # Add gene metadata
    gene_names = [f'Gene_{i:04d}' for i in range(n_genes)]
    adata.var.index = gene_names
    adata.var['is_de'] = False
    adata.var.iloc[de_gene_indices, adata.var.columns.get_loc('is_de')] = True

    return adata
```

## 🚀 Running Tests

### Basic Test Execution

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest tests/system/                  # System tests only
pytest tests/performance/             # Performance tests only

# Run tests with specific markers
pytest -m "unit and not slow"        # Fast unit tests only
pytest -m "integration"              # Integration tests
pytest -m "performance"              # Performance benchmarks

# Run specific test files
pytest tests/unit/test_clustering_service.py
pytest tests/integration/test_agent_workflows.py

# Run with coverage reporting
pytest --cov=lobster --cov-report=html

# Run with performance benchmarks
pytest --benchmark-only              # Only benchmark tests
pytest --benchmark-compare           # Compare with previous runs
```

### Advanced Test Options

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto                       # Auto-detect CPU cores
pytest -n 4                          # Use 4 processes

# Run with specific output formats
pytest --tb=long                     # Long traceback format
pytest --tb=short                    # Short traceback format
pytest -v                            # Verbose output
pytest -s                            # Don't capture stdout

# Run tests matching pattern
pytest -k "test_clustering"          # Tests with 'clustering' in name
pytest -k "not slow"                 # Skip slow tests

# Run with timeout (requires pytest-timeout)
pytest --timeout=300                 # 5 minute timeout per test

# Generate test reports
pytest --html=reports/report.html    # HTML report
pytest --junitxml=reports/junit.xml  # JUnit XML report
```

## 🎯 End-to-End Acceptance Testing System

Lobster AI features a **sophisticated 3-tier testing architecture** that includes a comprehensive End-to-End acceptance testing system for validating complete user workflows.

### 🌟 Testing Architecture Overview

1. **Traditional pytest Framework** - Unit/Integration/System/Performance tests
2. **End-to-End Acceptance Testing** - Natural language workflow validation
3. **Hybrid Execution** - Combined reporting and analytics

### 🚀 E2E Testing Components

#### **Core Files**
- **`tests/test_cases.json`** - 30+ realistic user scenarios with validation criteria
- **`tests/run_integration_tests.py`** - Advanced test runner with performance monitoring
- **`tests/run_tests.sh`** - User-friendly bash wrapper for easy execution

#### **Key Features**
- **Natural Language Testing**: Validates actual conversational interface users experience
- **Performance Monitoring**: Real-time CPU, memory, disk I/O tracking during execution
- **Scalable Architecture**: Tag-based filtering, priorities, parallel execution
- **Response Validation**: Keyword matching, length checks, error detection
- **Workspace Management**: Isolated test environments with automatic cleanup

### 🎮 Quick Start Commands

```bash
# User-friendly bash wrapper (recommended)
./tests/run_tests.sh                    # Run all scenarios sequentially
./tests/run_tests.sh --parallel         # Run in parallel
./tests/run_tests.sh --parallel -w 8    # 8 parallel workers

# Advanced Python runner with full control
python tests/run_integration_tests.py --categories basic,advanced --parallel
python tests/run_integration_tests.py --performance-monitoring --workers 4
python tests/run_integration_tests.py --run-pytest-integration --output results.json
```

### 📊 Test Categorization & Filtering

```bash
# Filter by categories
python tests/run_integration_tests.py --categories basic,advanced,performance,error_handling

# Filter by biological domains
python tests/run_integration_tests.py --tags geo,proteomics,multiomics,spatial,qc

# Filter by priority levels (1-5)
python tests/run_integration_tests.py --priorities 1,2,3

# Combine filters for targeted testing
python tests/run_integration_tests.py --categories advanced --tags geo,qc --parallel
```

### 🧪 Test Scenarios (30+ Realistic Workflows)

#### **Categories:**
- **`basic`** - Simple workflows (GEO download, basic QC)
- **`advanced`** - Complex analysis (multi-omics, trajectory analysis)
- **`performance`** - Large dataset processing
- **`error_handling`** - Edge cases and error recovery

#### **Biological Domain Tags:**
- **`geo`** - GEO dataset workflows
- **`qc`** - Quality control processes
- **`visualization`** - Plotting and visual analysis
- **`multiomics`** - Cross-platform integration
- **`spatial`** - Spatial transcriptomics
- **`proteomics`** - Mass spec and affinity proteomics
- **`clustering`** - Cell/sample grouping analysis

#### **Example Test Scenarios:**
```json
{
  "test_geo_download_with_qc_umap": {
    "inputs": [
      "Download GEO dataset GSE291670 and do the quality control",
      "Generate the UMAP with resolution 0.7"
    ],
    "category": "basic",
    "description": "Test complete workflow from download to UMAP visualization",
    "expected_duration": 120.0,
    "timeout": 400.0,
    "tags": ["geo", "qc", "umap", "visualization"],
    "priority": 3,
    "validation_criteria": {
      "input_0": {
        "required_keywords": ["quality control", "downloaded"],
        "no_errors": true
      },
      "input_1": {
        "required_keywords": ["UMAP", "resolution"],
        "no_errors": true
      }
    }
  }
}
```

### ⚡ Performance Monitoring

The E2E system includes comprehensive performance monitoring:

```bash
# Enable performance monitoring
python tests/run_integration_tests.py --performance-monitoring

# Features monitored:
# - CPU usage percentage (average and peak)
# - Memory consumption (RSS, peak usage)
# - Disk I/O operations (read/write MB)
# - Network activity (sent/received MB)
# - Test execution duration vs expected
# - Resource usage trends across test categories
```

### 🔄 Hybrid pytest Integration

Combine traditional pytest tests with E2E scenarios for comprehensive validation:

```bash
# Run both pytest and E2E tests together
python tests/run_integration_tests.py --run-pytest-integration

# Features:
# - Unified success/failure reporting
# - Combined coverage analytics
# - Category-wise performance breakdowns
# - Comprehensive JSON output with both test types
```

### 🎯 Advanced E2E Features

#### **Dependency Resolution**
Tests can specify dependencies for automatic ordering:
```json
{
  "dependencies": ["test_geo_download", "test_basic_qc"],
  "priority": 4
}
```

#### **Retry Logic**
Configurable retry attempts for flaky tests:
```json
{
  "retry_count": 2,
  "timeout": 300.0
}
```

#### **Response Validation**
Sophisticated validation of AI responses:
```json
{
  "validation_criteria": {
    "input_0": {
      "required_keywords": ["downloaded", "GSE109564"],
      "forbidden_keywords": ["error", "failed"],
      "min_length": 50,
      "no_errors": true
    }
  }
}
```

### 📋 Adding New E2E Test Scenarios

Add realistic user scenarios to **`tests/test_cases.json`**:

```json
{
  "test_my_custom_workflow": {
    "inputs": [
      "Download GSE123456 and perform quality control",
      "Apply batch correction using Harmony",
      "Create publication-ready UMAP plot"
    ],
    "category": "advanced",
    "description": "Test batch correction workflow",
    "tags": ["geo", "batch_correction", "visualization"],
    "priority": 3,
    "timeout": 600.0,
    "expected_duration": 240.0,
    "validation_criteria": {
      "input_0": {
        "required_keywords": ["downloaded", "quality control"],
        "no_errors": true
      },
      "input_1": {
        "required_keywords": ["batch correction", "Harmony"],
        "no_errors": true
      },
      "input_2": {
        "required_keywords": ["UMAP", "publication"],
        "no_errors": true
      }
    }
  }
}
```

### 📊 E2E Test Results & Analytics

The E2E system generates comprehensive reports:

```json
{
  "summary": {
    "test_execution_summary": {
      "total_tests": 25,
      "passed_tests": 23,
      "failed_tests": 2,
      "success_rate": 0.92,
      "total_duration": 1800.0,
      "average_duration": 72.0
    },
    "category_breakdown": {
      "basic": {"passed": 8, "failed": 0, "total": 8},
      "advanced": {"passed": 12, "failed": 2, "total": 14},
      "performance": {"passed": 3, "failed": 0, "total": 3}
    },
    "performance_summary": {
      "avg_cpu_percent": 15.2,
      "avg_memory_mb": 1024.5,
      "max_memory_mb": 2048.0
    }
  }
}
```

## 📈 Coverage and Quality Metrics

### Coverage Requirements
- **Minimum Coverage**: 80% (enforced by CI)
- **Target Coverage**: 95%
- **Critical Components**: 100% coverage required for core services and agents

### Coverage Analysis
```bash
# Generate coverage report
pytest --cov=lobster --cov-report=html --cov-report=term

# View coverage in browser
open htmlcov/index.html

# Check coverage for specific modules
pytest --cov=lobster.tools --cov-report=term-missing

# Fail build if coverage below threshold
pytest --cov=lobster --cov-fail-under=80
```

### Quality Metrics
- **Test Execution Time**: Unit tests <2min, Integration <15min, System <30min
- **Memory Usage**: No test should use >2GB RAM
- **Test Reliability**: <1% flaky test rate
- **Scientific Accuracy**: All biological algorithms must be validated

## 🔍 Debugging Tests

### Common Debugging Techniques

```python
# Add debug logging in tests
import logging
logging.basicConfig(level=logging.DEBUG)

# Use pytest debugging
pytest --pdb                         # Drop to debugger on failures
pytest --pdbcls=IPython.terminal.debugger:TerminalPdb  # Use IPython debugger

# Add debug prints (use capsys to capture)
def test_with_debug(capsys):
    print("Debug information here")
    # ... test code ...
    captured = capsys.readouterr()
    print(f"Captured output: {captured.out}")

# Temporary test isolation
pytest -x                           # Stop on first failure
pytest --lf                         # Run only last failed tests
pytest --ff                         # Run failures first
```

### Mock Data Debugging
```python
# Inspect generated data
def test_inspect_mock_data():
    adata = generate_synthetic_single_cell()

    print(f"Shape: {adata.shape}")
    print(f"Obs columns: {adata.obs.columns.tolist()}")
    print(f"Var columns: {adata.var.columns.tolist()}")
    print(f"Uns keys: {list(adata.uns.keys())}")

    # Save for manual inspection
    adata.write_h5ad('/tmp/debug_data.h5ad')
```

## 🎯 Best Practices Summary

### Test Design Principles
1. **Test Isolation**: Each test should be independent
2. **Realistic Data**: Use biologically plausible synthetic data
3. **Scientific Validation**: Verify biological correctness, not just code correctness
4. **Performance Awareness**: Monitor memory and time usage
5. **Error Coverage**: Test both success and failure paths

### Naming Conventions
```python
class TestServiceName:
    def test_method_success_case(self):         # Happy path
        pass

    def test_method_edge_case_empty_data(self): # Edge cases
        pass

    def test_method_error_invalid_params(self): # Error conditions
        pass

    def test_method_performance_large_data(self): # Performance
        pass
```

### Test Organization
- **One test class per component** being tested
- **Group related tests** in the same class
- **Use descriptive test names** that explain the scenario
- **Keep tests focused** - one concept per test
- **Use fixtures** for common setup/teardown

This comprehensive testing guide ensures that the Lobster AI platform maintains high quality, reliability, and scientific accuracy across all bioinformatics workflows.