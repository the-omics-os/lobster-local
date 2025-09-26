# 🧪 Lobster AI Testing Framework

Comprehensive testing infrastructure for the Lobster AI multi-agent bioinformatics platform, targeting 95%+ code coverage with scientifically accurate testing scenarios.

## 🎯 **Overview**

This testing framework provides robust validation across all components of Lobster AI, from individual functions to complete bioinformatics workflows. The framework emphasizes:

- **Scientific Accuracy**: Realistic biological data and analysis scenarios
- **Comprehensive Coverage**: Unit, integration, system, and performance testing
- **Production Readiness**: CI/CD integration with automated quality gates
- **Developer Experience**: Fast feedback loops and clear test organization

## 📁 **Test Structure**

```
tests/
├── 📁 unit/                    # Unit tests (13 files)
│   ├── core/                  # Core system components
│   ├── agents/                # AI agent functionality
│   └── tools/                 # Analysis services
├── 📁 integration/            # Integration tests (5 files)
│   ├── test_agent_workflows.py
│   ├── test_data_pipelines.py
│   ├── test_cloud_local_switching.py
│   ├── test_geo_download_workflows.py
│   └── test_multi_omics_integration.py
├── 📁 system/                 # System tests (3 files)
│   ├── test_full_analysis_workflows.py
│   ├── test_error_recovery.py
│   └── test_workspace_management.py
├── 📁 performance/            # Performance tests (3 files)
│   ├── test_large_dataset_processing.py
│   ├── test_concurrent_agent_execution.py
│   └── test_data_loading_benchmarks.py
├── 📁 mock_data/              # Synthetic data generation
│   ├── factories.py           # Data factory classes
│   ├── base.py                # Base configurations
│   └── generators/            # Specialized generators
├── 🔧 conftest.py             # Pytest configuration & fixtures
├── 🔧 test_config.yaml        # Test environment settings
├── 🔧 data_registry.json      # Test dataset registry
├── 🚀 run_integration_tests.py # Enhanced test runner
└── 📋 test_cases.json         # Test case definitions
```

## 🚀 **Quick Start**

### Basic Testing Commands

```bash
# Install development dependencies
make dev-install

# Run all tests with coverage
make test

# Fast parallel execution
make test-fast

# Run specific test categories
pytest tests/unit/          # Unit tests (~2 min)
pytest tests/integration/   # Integration tests (~15 min)
pytest tests/system/        # System tests (~30 min)
pytest tests/performance/   # Performance tests (~45 min)

# Generate coverage report
pytest --cov=lobster --cov-report=html
open htmlcov/index.html     # View coverage report
```

### Advanced Test Execution

```bash
# Run tests by biological focus
pytest -m "singlecell"      # Single-cell RNA-seq tests
pytest -m "proteomics"      # Proteomics analysis tests
pytest -m "multiomics"      # Multi-omics integration tests
pytest -m "geo"             # GEO database integration tests

# Run tests by complexity
pytest -m "fast"            # Quick tests (<5 sec)
pytest -m "slow"            # Longer tests (>30 sec)
pytest -m "external"        # Tests requiring external services

# Filter by test characteristics
pytest -m "memory_intensive" # High memory usage tests
pytest -m "gpu"             # GPU-accelerated tests
pytest -m "real_data"       # Tests with real biological datasets

# Development workflows
pytest --maxfail=5 -v      # Fail-fast with verbose output
pytest -x                  # Stop on first failure
pytest --lf               # Run last-failed tests only
pytest --ff               # Run failed tests first
```

## 🧬 **Mock Data Framework**

### Biological Data Factories

The testing framework includes sophisticated synthetic data generation that mimics real biological datasets:

```python
from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG, MEDIUM_DATASET_CONFIG

# Generate single-cell RNA-seq data
sc_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
print(f"Generated: {sc_data.shape[0]} cells × {sc_data.shape[1]} genes")

# Generate bulk RNA-seq data  
bulk_data = BulkRNASeqDataFactory(config={
    'n_samples': 24,
    'n_genes': 5000,
    'conditions': ['control', 'treatment'],
    'replicates': 12
})

# Generate proteomics data
from tests.mock_data.factories import ProteomicsDataFactory
proteomics_data = ProteomicsDataFactory(config={
    'n_samples': 48,
    'n_proteins': 2000,
    'missing_value_rate': 0.15
})
```

### Available Dataset Configurations

| **Configuration** | **Use Case** | **Size** | **Generation Time** |
|------------------|--------------|----------|---------------------|
| `SMALL_DATASET_CONFIG` | Unit tests, fast iteration | 1K cells, 2K genes | ~5 seconds |
| `MEDIUM_DATASET_CONFIG` | Integration tests | 5K cells, 3K genes | ~25 seconds |
| `LARGE_DATASET_CONFIG` | Performance tests | 20K cells, 10K genes | ~3 minutes |

### Realistic Biological Features

- **Cell Type Labels**: Biologically accurate cell type annotations
- **Mitochondrial/Ribosomal Genes**: Proper gene categorization
- **Batch Effects**: Realistic technical variation simulation
- **Missing Values**: Proteomics-appropriate missingness patterns
- **Spatial Coordinates**: 2D/3D spatial transcriptomics data
- **Temporal Progression**: Time-series and trajectory datasets

## 📊 **Test Categories**

### 🔬 **Unit Tests** (`tests/unit/`)

**Purpose**: Validate individual components in isolation  
**Runtime**: < 2 minutes  
**Coverage**: Core functions, data structures, algorithms

```bash
# Core system tests
pytest tests/unit/core/test_data_manager_v2.py
pytest tests/unit/core/test_client.py
pytest tests/unit/core/test_adapters.py
pytest tests/unit/core/test_schemas.py

# Agent system tests  
pytest tests/unit/agents/test_singlecell_expert.py
pytest tests/unit/agents/test_bulk_rnaseq_expert.py
pytest tests/unit/agents/test_proteomics_expert.py

# Service layer tests
pytest tests/unit/tools/test_geo_service.py
pytest tests/unit/tools/test_preprocessing_service.py
pytest tests/unit/tools/test_clustering_service.py
```

### 🔄 **Integration Tests** (`tests/integration/`)

**Purpose**: Test component interactions and workflows  
**Runtime**: < 15 minutes  
**Coverage**: Agent coordination, data pipelines, external integrations

```bash
# Agent workflow coordination
pytest tests/integration/test_agent_workflows.py

# Data pipeline orchestration
pytest tests/integration/test_data_pipelines.py

# Cloud/local execution switching
pytest tests/integration/test_cloud_local_switching.py

# GEO database workflows  
pytest tests/integration/test_geo_download_workflows.py

# Multi-omics data integration
pytest tests/integration/test_multi_omics_integration.py
```

### 🌐 **System Tests** (`tests/system/`)

**Purpose**: End-to-end validation of complete workflows  
**Runtime**: < 30 minutes  
**Coverage**: Full analysis pipelines, error recovery, workspace management

```bash
# Complete analysis workflows
pytest tests/system/test_full_analysis_workflows.py

# Error handling and recovery
pytest tests/system/test_error_recovery.py

# Workspace lifecycle management
pytest tests/system/test_workspace_management.py
```

### ⚡ **Performance Tests** (`tests/performance/`)

**Purpose**: Benchmark performance and scalability  
**Runtime**: < 45 minutes  
**Coverage**: Large datasets, concurrent execution, resource usage

```bash
# Large dataset processing
pytest tests/performance/test_large_dataset_processing.py

# Concurrent agent execution
pytest tests/performance/test_concurrent_agent_execution.py

# Data loading benchmarks
pytest tests/performance/test_data_loading_benchmarks.py

# Generate performance report
pytest tests/performance/ --benchmark-only --benchmark-json=benchmark_results.json
```

## 🎛️ **Enhanced Test Runner**

The `run_integration_tests.py` script provides advanced test execution with performance monitoring and comprehensive reporting.

### Basic Usage

```bash
# Run all test cases
python tests/run_integration_tests.py

# Run specific categories
python tests/run_integration_tests.py --categories basic,advanced

# Run with performance monitoring
python tests/run_integration_tests.py --performance-monitoring

# Parallel execution
python tests/run_integration_tests.py --parallel --workers 4

# Include pytest integration
python tests/run_integration_tests.py --run-pytest-integration
```

### Advanced Options

```bash
# Filter by test characteristics
python tests/run_integration_tests.py --tags geo,analysis
python tests/run_integration_tests.py --priorities 1,2,3

# Custom output and logging
python tests/run_integration_tests.py \
  --input tests/test_cases.json \
  --output detailed_results.json \
  --log-level DEBUG

# Environment-specific testing
LOBSTER_TEST_ENV=ci python tests/run_integration_tests.py
```

### Performance Monitoring Features

- **Resource Tracking**: CPU usage, memory consumption, disk I/O
- **Execution Timing**: Per-test and cumulative timing analysis
- **Error Analysis**: Detailed failure categorization and recovery metrics
- **Dependency Resolution**: Automatic test ordering based on dependencies
- **Retry Logic**: Configurable retry attempts with exponential backoff

## ⚙️ **Configuration**

### Test Environment Configuration (`test_config.yaml`)

```yaml
# Environment-specific settings
environments:
  development:
    log_level: "DEBUG" 
    enable_langfuse: false
    model_config:
      model_id: "us.anthropic.claude-3-5-haiku-20241022-v1:0"
      temperature: 0.7

  ci:
    log_level: "INFO"
    timeout_multiplier: 2.0
    resource_limits:
      max_memory_mb: 4096
      max_concurrent_tests: 2

# Test categories with specific parameters
test_categories:
  basic:
    default_timeout: 180
    parallel_safe: true
    required_for_ci: true
    
  performance:
    default_timeout: 1800
    parallel_safe: false
    resource_intensive: true
```

### Dataset Registry (`data_registry.json`)

Centralized registry of available test datasets with metadata:

```json
{
  "datasets": {
    "geo_reference_datasets": {
      "GSE109564": {
        "title": "Human PBMC single-cell RNA-seq",
        "cell_count": 4340,
        "gene_count": 33538,
        "availability": {
          "development": true,
          "ci": false,
          "staging": true
        }
      }
    },
    "mock_datasets": {
      "small_single_cell_pbmc": {
        "n_obs": 1000,
        "n_vars": 2000,
        "generation_time_seconds": 5.0,
        "test_usage": {
          "categories": ["unit", "basic", "integration"]
        }
      }
    }
  }
}
```

### Pytest Configuration (`pytest.ini`)

Comprehensive pytest configuration with:

- **Test Markers**: 30+ markers for test categorization
- **Coverage Settings**: Branch coverage, HTML/XML reporting
- **Timeout Management**: Per-test and global timeout settings  
- **Logging Configuration**: Structured logging for debugging
- **Parallel Execution**: xdist configuration for parallel testing

## 🚦 **CI/CD Integration**

### GitHub Actions Workflows

#### Comprehensive CI (`ci.yml`)

- **Multi-Platform Testing**: Ubuntu, macOS, Windows
- **Python Version Matrix**: 3.11, 3.12
- **Test Categories**: Unit → Integration → System → Performance
- **Code Quality**: Pre-commit hooks, linting, type checking
- **Security Scanning**: Bandit, Safety, dependency analysis
- **Coverage Reporting**: Codecov integration

#### Pull Request Validation (`pr-validation.yml`)

- **Change Analysis**: Automatic complexity scoring and labeling
- **Fast Feedback**: Unit tests and linting within 15 minutes
- **Security Checks**: Vulnerability scanning for all PRs
- **Performance Impact**: Benchmark comparisons for core changes
- **Cross-Platform**: Conditional testing based on PR complexity

#### Release Automation (`release.yml`)

- **Pre-Release Testing**: Comprehensive test suite validation
- **Security Scanning**: Enhanced security analysis for releases
- **Build Artifacts**: Wheel and source distribution creation
- **PyPI Publishing**: Automated package publishing
- **Post-Release Validation**: Installation testing across platforms

#### Dependency Maintenance (`dependency-updates.yml`)

- **Weekly Security Scans**: Automated vulnerability detection
- **Dependency Updates**: Smart update strategies (patch/minor/major)
- **Test Validation**: Ensure updates don't break functionality
- **Automated PRs**: Create pull requests for approved updates

### Quality Gates

Every pull request must pass:

- ✅ **Code Formatting**: Black, isort
- ✅ **Linting**: Flake8 with bioinformatics-specific rules
- ✅ **Type Checking**: MyPy static analysis
- ✅ **Security**: Bandit security linting
- ✅ **Unit Tests**: >80% coverage required
- ✅ **Integration Tests**: Critical workflow validation
- ✅ **Performance**: No significant regressions

## 📈 **Performance Benchmarking**

### Benchmark Categories

```bash
# Data loading performance
pytest tests/performance/test_data_loading_benchmarks.py::test_h5ad_loading_performance

# Analysis algorithm benchmarks  
pytest tests/performance/test_large_dataset_processing.py::test_clustering_performance

# Concurrent execution scaling
pytest tests/performance/test_concurrent_agent_execution.py::test_agent_scalability

# Memory usage profiling
pytest tests/performance/ --benchmark-only --benchmark-sort=mean
```

### Performance Thresholds

| **Operation** | **Small Dataset** | **Medium Dataset** | **Large Dataset** |
|---------------|------------------|-------------------|------------------|
| **Data Loading** | <5 seconds | <30 seconds | <2 minutes |
| **Quality Control** | <10 seconds | <60 seconds | <5 minutes |
| **Clustering** | <30 seconds | <3 minutes | <15 minutes |
| **Visualization** | <15 seconds | <90 seconds | <8 minutes |

### Memory Guidelines

- **Unit Tests**: <100 MB peak memory usage
- **Integration Tests**: <500 MB peak memory usage  
- **System Tests**: <2 GB peak memory usage
- **Performance Tests**: <8 GB peak memory usage

## 🛠️ **Development Workflow**

### Pre-Commit Hooks

Automatic code quality enforcement:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files

# Custom testing hooks
pre-commit run test-structure-validation
pre-commit run mock-data-validation
pre-commit run run-critical-tests
```

### Test-Driven Development

1. **Write Tests First**: Create failing tests for new features
2. **Implement Incrementally**: Build functionality to pass tests
3. **Refactor Safely**: Comprehensive test coverage enables confident refactoring
4. **Validate Integration**: Ensure new features work with existing system

### Debugging Failed Tests

```bash
# Run with verbose output and keep temporary files
pytest tests/unit/core/test_data_manager_v2.py -v -s --tb=long

# Debug specific test with pdb
pytest tests/unit/core/test_data_manager_v2.py::test_modality_management --pdb

# Capture stdout/stderr
pytest tests/integration/test_agent_workflows.py --capture=no

# Run with custom workspace (preserved after test)
LOBSTER_WORKSPACE_PATH=/tmp/debug_workspace pytest tests/system/test_full_analysis_workflows.py -k "test_single_cell_workflow"
```

## 📋 **Test Markers Reference**

### Category Markers

- `unit`: Unit tests for individual components
- `integration`: Integration tests across multiple components
- `system`: End-to-end system tests
- `performance`: Performance and benchmark tests

### Characteristic Markers

- `fast`: Tests completing under 5 seconds
- `slow`: Tests taking longer than 30 seconds  
- `external`: Tests requiring external services
- `memory_intensive`: Tests requiring >1GB memory
- `gpu`: Tests requiring GPU acceleration

### Biological Domain Markers

- `bio`: General bioinformatics algorithm tests
- `singlecell`: Single-cell RNA-seq analysis tests
- `bulk_rna`: Bulk RNA-seq analysis tests
- `proteomics`: Proteomics analysis tests
- `multiomics`: Multi-omics integration tests
- `geo`: GEO database integration tests

### Infrastructure Markers

- `agents`: Agent system functionality tests
- `workflows`: Multi-agent workflow tests
- `data_manager`: Data management tests
- `security`: Security-related tests
- `error_handling`: Error recovery tests

## 🔍 **Troubleshooting**

### Common Issues

#### Test Discovery Problems

```bash
# Verify test structure
python -c "
import os
required = ['tests/unit', 'tests/integration', 'tests/system', 'tests/performance']
missing = [d for d in required if not os.path.exists(d)]
print(f'Missing: {missing}' if missing else 'All directories present ✓')
"

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Mock Data Generation Issues

```bash
# Validate mock data factories
python -c "
import sys; sys.path.insert(0, 'tests')
from mock_data.factories import SingleCellDataFactory
from mock_data.base import SMALL_DATASET_CONFIG
data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
print(f'Generated: {data.shape[0]} × {data.shape[1]} ✓')
"
```

#### Memory Issues with Large Tests

```bash
# Run with memory monitoring
pytest tests/performance/ --memray

# Limit concurrent execution
pytest tests/ -n 2  # Only 2 parallel workers

# Skip memory-intensive tests
pytest tests/ -m "not memory_intensive"
```

#### CI/CD Pipeline Failures

```bash
# Run CI simulation locally
act --job unit-tests  # Requires 'act' tool

# Debug GitHub Actions
pytest tests/ --github-actions-output

# Test matrix simulation
tox -e py312-ubuntu  # Requires tox configuration
```

### Getting Help

- **🐛 Test Issues**: [Report on GitHub Issues](https://github.com/the-omics-os/lobster/issues)
- **💬 General Questions**: [Discord Community](https://discord.gg/HDTRbWJ8omicsos)
- **📧 Direct Support**: [Email Testing Team](mailto:info@omics-os.com)

## 🎯 **Contributing to Tests**

### Adding New Tests

1. **Choose Appropriate Category**: Unit/Integration/System/Performance
2. **Follow Naming Conventions**: `test_*.py` for files, `test_*` for functions
3. **Use Appropriate Markers**: Add relevant pytest markers
4. **Include Documentation**: Docstrings explaining test purpose
5. **Mock External Dependencies**: Use fixtures for external services
6. **Validate with Real Scenarios**: Ensure biological accuracy

### Test Quality Guidelines

- **Atomic Tests**: Each test should validate one specific behavior
- **Deterministic**: Tests must produce consistent results
- **Fast Feedback**: Unit tests should complete quickly
- **Clear Assertions**: Use descriptive assertion messages
- **Cleanup**: Properly clean up resources and temporary files
- **Documentation**: Include docstrings explaining complex test logic

### Example Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from tests.mock_data.factories import SingleCellDataFactory
from lobster.core.data_manager_v2 import DataManagerV2

@pytest.mark.unit
@pytest.mark.singlecell
class TestSingleCellAnalysis:
    """Test single-cell RNA-seq analysis workflows."""
    
    def test_quality_control_filtering(self, temp_workspace, mock_single_cell_data):
        """Test quality control filtering removes low-quality cells."""
        # Arrange
        data_manager = DataManagerV2(workspace_path=temp_workspace)
        data_manager.modalities['test_data'] = mock_single_cell_data
        
        # Act
        filtered_data = data_manager.apply_quality_control(
            modality_name='test_data',
            min_genes=200,
            max_mt_percent=20
        )
        
        # Assert
        assert filtered_data.n_obs < mock_single_cell_data.n_obs
        assert 'passed_qc' in filtered_data.obs.columns
        assert filtered_data.obs['passed_qc'].all()
```

---

**🦞 Ready to contribute to Lobster AI's robust testing infrastructure?**

[Testing Guidelines](../CONTRIBUTING.md#testing) • [Code Style Guide](../CONTRIBUTING.md#style) • [CI/CD Documentation](../.github/workflows/README.md)