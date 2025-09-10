# 🧪 Lobster AI Testing Framework

Comprehensive testing infrastructure for the Lobster AI multi-agent bioinformatics platform, targeting 95%+ code coverage with scientifically accurate testing scenarios.

## 📖 **Complete Documentation**

📚 **[View Complete Testing Guide →](../docs/testing.md)**

The full testing documentation covers:
- **Test Structure & Categories** - Unit, integration, system, and performance tests
- **Mock Data Framework** - Biological data generation for realistic testing
- **Enhanced Test Runner** - Advanced execution with performance monitoring
- **CI/CD Integration** - GitHub Actions workflows and quality gates
- **Performance Benchmarking** - Thresholds and memory guidelines
- **Development Workflow** - Pre-commit hooks, debugging, and TDD practices
- **Configuration** - Test environment settings and dataset registry
- **Troubleshooting** - Common issues and solutions

## 🚀 **Quick Start**

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
```

## 📁 **Test Structure**

```
tests/
├── 📁 unit/                    # Unit tests (13 files)
├── 📁 integration/            # Integration tests (5 files)  
├── 📁 system/                 # System tests (3 files)
├── 📁 performance/            # Performance tests (3 files)
├── 📁 mock_data/              # Synthetic data generation
├── 🔧 conftest.py             # Pytest configuration & fixtures
├── 🔧 test_config.yaml        # Test environment settings
├── 🔧 data_registry.json      # Test dataset registry
└── 🚀 run_integration_tests.py # Enhanced test runner
```

## 🧬 **Mock Data Framework**

Generate realistic biological datasets for testing:

```python
from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG

# Generate single-cell RNA-seq data
sc_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
print(f"Generated: {sc_data.shape[0]} cells × {sc_data.shape[1]} genes")
```

## 🏃 **Enhanced Test Runner**

```bash
# Run all test cases with performance monitoring
python tests/run_integration_tests.py --performance-monitoring

# Run specific categories in parallel
python tests/run_integration_tests.py --categories basic,advanced --parallel
```

## 📋 **Test Markers**

Filter tests by category, complexity, or biological domain:

```bash
# By category
pytest -m "unit"              # Unit tests only
pytest -m "integration"       # Integration tests only

# By biological focus  
pytest -m "singlecell"        # Single-cell RNA-seq tests
pytest -m "proteomics"        # Proteomics analysis tests

# By characteristics
pytest -m "fast"              # Quick tests (<5 sec)
pytest -m "memory_intensive"  # High memory usage tests
```

---

**🦞 Ready to contribute to Lobster AI's robust testing infrastructure?**

📚 **[Complete Testing Guide](../docs/testing.md)** • [Testing Guidelines](../CONTRIBUTING.md#testing) • [Code Style Guide](../CONTRIBUTING.md#style)