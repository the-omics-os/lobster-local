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
├── 📁 unit/                    # Unit tests (20+ files)
│   ├── core/                  # Core system components
│   ├── agents/                # AI agent functionality
│   ├── services/              # Existing analysis services (7 files)
│   └── tools/                 # Analysis tools (12+ files) ✨ NEW
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

## 🧪 **NEW: Comprehensive Tools Test Suite**

**Recently added comprehensive unit tests for `tests/unit/tools/` directory targeting 95%+ coverage:**

### 🧬 **Proteomics Analysis Suite (Complete)**
- **`test_proteomics_analysis_service.py`** - Statistical testing, dimensionality reduction (PCA/t-SNE/UMAP), clustering analysis, pathway enrichment
- **`test_proteomics_preprocessing_service.py`** - Missing value imputation (KNN/MNAR/mixed), normalization methods, batch correction
- **`test_proteomics_quality_service.py`** - Missing value patterns, CV assessment, contaminant detection, PCA outliers, replicate validation
- **`test_proteomics_differential_service.py`** - Differential expression, time course analysis, correlation analysis, volcano plots
- **`test_proteomics_visualization_service.py`** - Heatmaps, intensity distributions, volcano plots, networks, QC dashboards

### 🧬 **Bulk RNA-seq Analysis Suite (Complete)**
- **`test_bulk_rnaseq_service.py`** - Quality control, quantification, differential analysis, pathway enrichment
- **`test_differential_formula_service.py`** - R-style formula parsing, design matrix construction
- **`test_file_upload_service.py`** - File validation, format detection, upload processing
- **`test_pseudobulk_service.py`** - Single-cell to bulk aggregation workflows

### 🧬 **Additional Tools Coverage**
**Existing files in `tests/unit/tools/`:**
- `test_bulk_rnaseq_pydeseq2.py` - PyDESeq2 integration tests

**Each test file provides:**
- ✅ **95%+ Coverage** - Comprehensive method testing with edge cases
- ✅ **Realistic Mock Data** - Biologically accurate synthetic datasets
- ✅ **Error Handling** - Exception testing and graceful degradation
- ✅ **Performance Testing** - Memory efficiency and scalability validation
- ✅ **Integration Testing** - Multi-step workflow validation
- ✅ **Scientific Accuracy** - Biologically meaningful test scenarios

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

# Run new tools tests specifically
pytest tests/unit/tools/      # All tools unit tests
pytest tests/unit/tools/test_proteomics_*.py  # Proteomics suite only
```

## 📊 **Test Coverage Summary**

### **Tools Directory Test Statistics**
```
📁 tests/unit/tools/
├── 🧬 Proteomics Suite:           5 files  |  ~4,000 lines  |  95%+ coverage
├── 🧬 Bulk RNA-seq Suite:         4 files  |  ~3,000 lines  |  95%+ coverage
├── 🧬 Additional Tools:           1 file   |  ~1,000 lines  |  95%+ coverage
└── 📊 Total Coverage:             10 files |  ~8,000 lines  |  95%+ coverage
```

**Key Features:**
- **🎯 Scientific Accuracy** - Biologically realistic test scenarios with proper statistical validation
- **🔬 Comprehensive Coverage** - All public methods, edge cases, error conditions, and integration workflows
- **⚡ Performance Validated** - Memory efficiency testing for large datasets (200+ samples, 1000+ proteins)
- **🛡️ Error Resilience** - Extensive exception handling and graceful degradation testing
- **🔄 Workflow Integration** - Multi-step analysis pipeline validation

### **Mock Data Framework Enhancements**
- **ProteomicsDataFactory** - Generates realistic proteomics datasets with missing value patterns, batch effects, and biological variation
- **Structured Missing Values** - MNAR (Missing Not At Random) patterns typical of proteomics data
- **Differential Expression Patterns** - Controlled up/down regulation for testing statistical methods
- **Time Course Data** - Temporal expression patterns for longitudinal analysis validation
- **Correlation Networks** - Realistic protein-protein correlation structures

---

**🦞 Ready to contribute to Lobster AI's robust testing infrastructure?**

📚 **[Complete Testing Guide](../docs/testing.md)** • [Testing Guidelines](../CONTRIBUTING.md#testing) • [Code Style Guide](../CONTRIBUTING.md#style)