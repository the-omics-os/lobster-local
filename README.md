# 🦞 Lobster AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**AI-Powered Multi-Omics Bioinformatics Analysis Platform**

Transform your bioinformatics research with intelligent AI agents that understand your data and provide expert analysis insights.

## ✨ What is Lobster AI?

Lobster AI is a revolutionary bioinformatics platform that combines the power of **specialized AI agents** with proven scientific tools to analyze complex multi-omics data. Instead of wrestling with command-line tools and scripts, simply describe your analysis needs in natural language.

**✨ Now featuring a stunning Rich CLI interface with professional orange branding, multi-panel dashboards, and real-time progress monitoring!**

### 🎯 **Perfect For**
- **Bioinformatics Researchers** analyzing RNA-seq, proteomics, and multi-omics data
- **Computational Biologists** seeking intelligent analysis workflows  
- **Life Science Teams** requiring reproducible, publication-ready results
- **Students & Educators** learning modern bioinformatics approaches

## 🚀 Key Features

### 🤖 **Intelligent AI Agents**
- **Data Expert**: Handles data loading, format conversion, and quality assessment
- **Research Agent**: Discovers relevant datasets and literature for your analysis
- **Transcriptomics Expert**: Specialized in single-cell and bulk RNA-seq analysis
- **MS Proteomics Expert**: Mass spectrometry proteomics with DDA/DIA workflows and missing value handling
- **Affinity Proteomics Expert**: Targeted protein panels, Olink arrays, and antibody-based assays
- **Method Expert**: Extracts optimal parameters from scientific literature

### 🧬 **Advanced Analysis Capabilities**
- **Single-Cell RNA-seq**: Quality control, clustering, cell type annotation, trajectory analysis
- **Pseudobulk Aggregation**: Convert single-cell to sample-level matrices for statistical analysis
- **Bulk RNA-seq & pyDESeq2**: Pure Python differential expression with complex experimental designs
- **Formula-Based Statistics**: R-style formulas (`~condition + batch`) with design matrix construction
- **Mass Spectrometry Proteomics**: Missing value pattern analysis, intensity normalization, peptide-to-protein mapping
- **Affinity Proteomics**: Targeted panels, coefficient of variation analysis, antibody validation workflows
- **Proteomics Visualization**: Volcano plots, correlation networks, pathway enrichment, QC dashboards
- **Multi-Omics Integration**: Cross-platform analysis using MuData framework
- **Literature Mining**: Automated parameter optimization from publications

### 🎨 **Enhanced User Experience**
- **Natural Language Interface**: Describe analyses in plain English
- **Rich CLI with Orange Branding**: Professional terminal interface with #e45c47 theming
- **Interactive Chat Mode**: Conversational analysis workflow with real-time feedback
- **Multi-Panel Dashboards**: System health, workspace overview, and analysis monitoring
- **Advanced Navigation**: File tree browsing with status indicators and hierarchical views
- **Live Progress Tracking**: Multi-task progress monitoring with concurrent operation support
- **Intelligent Error Handling**: Context-aware suggestions with Rich traceback integration
- **Automatic Visualization**: Publication-ready plots and reports
- **Complete Provenance**: Reproducible analysis trails
- **Format Flexibility**: CSV, Excel, H5AD, 10X MTX, and more

## 🎬 **Quick Examples**

### Single-Cell RNA-seq Analysis
```bash
# Install Lobster AI
git clone https://github.com/homara-ai/lobster.git
cd lobster && make install

# Start analyzing in seconds
lobster chat

# Natural language analysis
🦞 You: "Download GSE109564 and perform single-cell clustering analysis"

🦞 Lobster: I'll download and analyze this single-cell dataset for you...

✓ Downloaded 5,000 cells × 20,000 genes
✓ Quality control: filtered to 4,477 high-quality cells  
✓ Identified 12 distinct cell clusters
✓ Generated UMAP visualization and marker gene analysis

Analysis complete! Found 12 cell populations with distinct expression signatures.
```

### Mass Spectrometry Proteomics Analysis
```bash
🦞 You: "Analyze my proteomics data with missing value assessment and differential analysis"

🦞 Lobster: I'll analyze your mass spectrometry proteomics data...

✓ Loaded proteomics data: 2,847 proteins × 24 samples
✓ Missing value analysis: 42% missing (typical for MS data)
✓ Applied log2 transformation and normalization
✓ Identified 156 significantly regulated proteins (p < 0.05)
✓ Generated volcano plot and pathway enrichment analysis

Analysis complete! Found significant protein expression changes with comprehensive QC metrics.
```

### Affinity Proteomics (Olink) Analysis
```bash
🦞 You: "Process my Olink panel data and create QC dashboard"

🦞 Lobster: I'll process your targeted proteomics panel...

✓ Loaded Olink data: 384 proteins × 96 samples
✓ QC assessment: 18% missing values (excellent for affinity)
✓ Coefficient of variation analysis completed
✓ Generated comprehensive QC dashboard
✓ Created protein correlation network visualization

Analysis complete! High-quality targeted proteomics data with professional QC metrics.
```

## 🎨 **Enhanced CLI Experience**

Lobster AI features a **stunning Rich CLI interface** with professional orange branding (#e45c47) and advanced terminal capabilities:

### 🚀 **Professional Interface Features**

#### **Multi-Panel Dashboards**
```bash
🦞 You: "/dashboard"
# Shows comprehensive system health with CPU, memory, and agent status

🦞 You: "/workspace-info"
# Detailed workspace overview with file status and data summary

🦞 You: "/analysis-dash"
# Real-time analysis monitoring with operation history
```

#### **Advanced Navigation**
```bash
🦞 You: "/tree"
# Hierarchical directory tree with orange highlights and file type icons
📁 Current Directory: project_data
├── 🧬 single_cell_data.h5ad (2.1MB)
├── 📊 metadata.csv (45KB) ✓
├── 📈 results/
│   ├── 📈 umap_plot.html ✓
│   └── 📊 de_genes.csv ✓
└── 🦞 Lobster Workspace/
    ├── 📊 Data Files/
    └── 📈 Visualizations/
```

#### **Enhanced Input & Command History**
```bash
# Professional text input with full navigation support
🦞 ~/project ▸ analyze my single-cell data
                ↑ ← → Use arrow keys to navigate and edit
                ↑ ↓ Browse command history
                Ctrl+R Reverse search through history

🦞 You: "/input-features"
# Shows current input capabilities and navigation features
✨ Enhanced input: Arrow navigation, command history, and reverse search enabled
```

**New Features (v2.1+):**
- **← → Arrow keys** - Navigate within text naturally
- **↑ ↓ Arrow keys** - Browse persistent command history
- **Ctrl+R** - Reverse search through previous commands
- **Auto-disappearing progress bars** - Clean interface without clutter
- **Optimized startup** - Reduced log noise for faster initialization

#### **Live Progress Monitoring**
```bash
🦞 You: "/progress"
# Multi-task progress tracking with real-time updates

🔄 Operations Overview
┌─────────────────────┬──────────────────┬──────────┬──────────┐
│ Operation           │ Progress         │ Status   │ Duration │
├─────────────────────┼──────────────────┼──────────┼──────────┤
│ GEO Download        │ ████████████ 85% │ Running  │ 00:02:34 │
│ Quality Control     │ ██████░░░░░░ 50%  │ Running  │ 00:01:15 │
│ Clustering Analysis │ ░░░░░░░░░░░░  0%  │ Pending  │ 00:00:00 │
└─────────────────────┴──────────────────┴──────────┴──────────┘
```

#### **Intelligent Error Handling**
```bash
🦞 You: "/read nonexistent_file.h5ad"

❌ Error Panel
┌─ Command Failed ─────────────────────────────┐
│ FileNotFoundError: File not found           │
│                                              │
│ 💡 Suggestion: Check if the file exists     │
│    and you have read permissions            │
└──────────────────────────────────────────────┘
```

### 🎯 **Enhanced Command Suite**

| Command | Description | Visual Enhancement |
|---------|-------------|-------------------|
| `/dashboard` | System health dashboard | Multi-panel layout with orange progress bars |
| `/workspace-info` | Detailed workspace overview | File status indicators and data summaries |
| `/analysis-dash` | Analysis monitoring | Real-time operation tracking |
| `/tree` | Directory tree navigation | Hierarchical view with file type icons |
| `/progress` | Multi-task progress monitor | Live concurrent operation tracking |
| `/files` | Enhanced file listing | Orange-themed tables with metadata |
| `/help` | Comprehensive help system | Professional styling with orange highlights |

### 🔥 **Professional Features**

- **🎨 Orange Brand Theming**: Consistent #e45c47 branding throughout
- **📊 Rich Layout System**: Multi-panel displays with professional styling
- **⚡ Real-time Updates**: Live dashboard monitoring and progress tracking
- **🌳 File Tree Navigation**: Hierarchical browsing with status indicators
- **🔍 Enhanced Error Messages**: Context-aware suggestions with Rich tracebacks
- **📈 Progress Visualization**: Advanced multi-task operation monitoring
- **💫 Interactive Elements**: Sophisticated prompts and selection interfaces

## 🔬 **Comprehensive Proteomics Platform**

Lobster AI provides **industry-leading proteomics analysis** with specialized agents and professional-grade algorithms:

### 🎯 **Proteomics-Specific Features**

#### **🔬 Mass Spectrometry Support**
- **DDA/DIA Workflows**: Complete data-dependent and data-independent acquisition pipelines
- **Missing Value Intelligence**: Sophisticated handling of 30-70% missing values typical in MS data
- **Peptide-to-Protein Mapping**: Professional aggregation algorithms with statistical validation
- **Intensity Normalization**: Multiple normalization strategies (TMM, quantile, VSN)
- **Database Search Artifact Removal**: Quality-based filtering of unreliable identifications

#### **🎯 Affinity Proteomics Excellence**
- **Olink Panel Support**: Specialized workflows for targeted protein panels
- **Antibody Validation**: Quality assessment tools for antibody-based assays
- **Coefficient of Variation Analysis**: Technical reproducibility assessment
- **Low Missing Values**: Optimized for <30% missing values in affinity data
- **Panel Comparison**: Cross-panel harmonization and batch effect correction

#### **📊 Professional Visualization Suite**
- **Missing Value Heatmaps**: Pattern analysis across samples and proteins
- **Intensity Distribution Plots**: Platform-specific data quality assessment
- **Volcano Plots**: Publication-ready differential expression visualization
- **Protein Correlation Networks**: Interactive NetworkX-based protein interaction maps
- **Pathway Enrichment Plots**: Functional analysis with statistical significance
- **Comprehensive QC Dashboards**: Multi-metric quality control reports

#### **🧬 Advanced Statistical Analysis**
- **Differential Expression**: Multiple testing correction with FDR control
- **Pathway Analysis**: Gene set enrichment with protein-specific databases
- **Quality Control**: Multi-level assessment (PSM, peptide, protein levels)
- **Batch Effect Detection**: Automated identification and correction strategies
- **Statistical Modeling**: Linear mixed models for complex experimental designs

### 🤖 **Specialized Proteomics AI Agents**

#### **MS Proteomics Expert**
- **DDA/DIA Pipeline Management**: Automated workflow selection and optimization
- **Missing Value Pattern Analysis**: MNAR vs MCAR classification and handling
- **Database Search Integration**: Support for MaxQuant, MSFragger, and other tools
- **Quality Assessment**: Multi-level QC from spectrum to protein identification

#### **Affinity Proteomics Expert**
- **Panel-Specific Optimization**: Tailored analysis for Olink, SOMAscan, MSD platforms
- **Antibody Performance Metrics**: Validation and quality scoring algorithms
- **Cross-Platform Harmonization**: Integration across different affinity technologies
- **Targeted Analysis Workflows**: Hypothesis-driven protein subset analysis

### 🔧 **Professional Service Architecture**

#### **ProteomicsPreprocessingService**
- **Multi-Platform Support**: MS and affinity proteomics data loading
- **Intelligent Filtering**: Protein and sample quality-based filtering
- **Normalization Strategies**: Platform-appropriate normalization methods
- **Missing Value Handling**: Imputation strategies with statistical validation

#### **ProteomicsQualityService**
- **Comprehensive QC Metrics**: Sample and protein-level quality assessment
- **Missing Value Analysis**: Pattern detection and classification
- **Technical Reproducibility**: CV analysis and batch effect detection
- **Platform-Specific Thresholds**: Evidence-based quality criteria

#### **ProteomicsAnalysisService**
- **Statistical Testing**: Multiple hypothesis testing with appropriate corrections
- **Dimensionality Reduction**: PCA, t-SNE optimized for proteomics data
- **Clustering Analysis**: Protein and sample clustering with validation metrics
- **Pathway Analysis**: Protein-centric functional enrichment

#### **ProteomicsDifferentialService**
- **Advanced Statistical Models**: Linear models with empirical Bayes moderation
- **Multiple Comparisons**: FDR control across protein and contrast levels
- **Effect Size Estimation**: Fold change calculations with confidence intervals
- **Result Interpretation**: Automated significance assessment and reporting

#### **ProteomicsVisualizationService**
- **Publication-Quality Plots**: Plotly-based interactive visualizations
- **Missing Value Visualizations**: Heatmaps and pattern analysis plots
- **Statistical Result Plots**: Volcano plots, MA plots, p-value distributions
- **Network Visualizations**: Protein interaction and correlation networks
- **QC Dashboards**: Comprehensive multi-panel quality control reports

### 📈 **Industry Integration**

#### **File Format Support**
- **MaxQuant Output**: proteinGroups.txt, peptides.txt processing
- **Spectronaut Results**: DirectDIA and library-based workflows
- **Olink Data**: NPX values with quality flags and metadata
- **Generic Formats**: CSV, Excel, HDF5 with flexible schema detection

#### **Database Integration**
- **UniProt Mapping**: Automatic protein annotation and ID conversion
- **Pathway Databases**: Reactome, KEGG, GO integration for functional analysis
- **PPI Networks**: STRING, BioGRID protein interaction data
- **Literature Mining**: Automated parameter extraction from proteomics publications

This comprehensive proteomics platform ensures publication-ready results with professional-grade algorithms and industry-standard workflows.

## 📦 **Quick Installation**

### Local Installation
```bash
git clone https://github.com/homara-ai/lobster.git
cd lobster
make install
lobster chat  # Start with enhanced Rich CLI experience!
```

### First Launch Experience
When you start Lobster AI, you'll be greeted with a professional orange-branded interface:

```bash
🦞 lobster chat

┌─ Welcome ────────────────────────────────────────────────────────────┐
│  🦞 LOBSTER by homara AI                                             │
│  Multi-Agent Bioinformatics Analysis System v2.0                    │
│                                                                      │
│  🧬 Key Tasks:                                                       │
│  • Analyze RNA-seq & genomics data                                  │
│  • Generate visualizations and plots                                │
│  • Extract insights from bioinformatics datasets                    │
│  • Access GEO & literature databases                               │
│                                                                      │
│  📋 Essential Commands:                                              │
│  /dashboard    - System health dashboard                            │
│  /tree         - Directory tree navigation                          │
│  /help         - Complete command reference                         │
└──────────────────────────────────────────────────────────────────────┘

🦞 ~/projects ▸
```

### Requirements
- Python 3.12+
- 4GB+ RAM recommended
- Modern terminal with Unicode support (for Rich CLI features)
- API keys for LLM providers (OpenAI, AWS Bedrock)

## ☁️ **Lobster Cloud: Seamless Cloud Integration**

Experience the power of cloud computing with **automatic cloud detection**:
- ☁️ **Zero Configuration** - Just set your API key and go
- 🚀 **Scalable Computing** - Handle large datasets without local hardware limits  
- 🔄 **Seamless Switching** - Automatic fallback to local mode if needed
- 🔒 **Secure Processing** - Enterprise-grade security for your data

### **Getting Started with Cloud**

1. **Get your API key** from [cloud.lobster.ai](mailto:cloud@homara.ai?subject=Lobster%20Cloud%20API%20Key%20Request)
2. **Set your environment variable**:
   ```bash
   # Add to your .env file
   LOBSTER_CLOUD_KEY=your-api-key-here
   ```
3. **Run Lobster as usual** - it automatically detects and uses cloud mode:
   ```bash
   lobster chat  # Automatically uses cloud when key is present
   ```

### **Smart Local Fallback**
- **No Cloud Key?** → Runs locally with full functionality
- **Cloud Unavailable?** → Automatically falls back to local mode
- **Same Experience** → Identical interface whether cloud or local

**[Request Cloud Access →](mailto:cloud@homara.ai?subject=Lobster%20Cloud%20API%20Key%20Request)**

## 🧪 **Testing Framework**

Lobster AI includes a **comprehensive testing framework** with 95%+ code coverage targeting, ensuring reliability across all bioinformatics workflows.

### 🎯 **Test Categories**

| **Test Type** | **Purpose** | **Coverage** | **Runtime** |
|---------------|-------------|--------------|-------------|
| **Unit Tests** | Core component validation | Individual functions/classes | < 2 minutes |
| **Integration Tests** | Multi-component workflows | Agent interactions, data pipelines | < 15 minutes |
| **System Tests** | End-to-end scenarios | Complete analysis workflows | < 30 minutes |
| **Performance Tests** | Benchmarking & scalability | Large datasets, concurrent execution | < 45 minutes |

### 🚀 **Quick Testing Commands**

```bash
# Run all tests (recommended for development)
make test

# Fast parallel execution
make test-fast

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests
pytest tests/system/        # System tests
pytest tests/performance/   # Performance benchmarks

# Run tests with coverage reporting
pytest --cov=lobster --cov-report=html

# Run enhanced integration test runner
python tests/run_integration_tests.py --categories basic,advanced
```

### 📊 **Test Infrastructure**

- **🧬 Biological Data Mocking**: Realistic synthetic datasets (single-cell, proteomics, multi-omics)
- **⚡ Performance Monitoring**: Memory, CPU, and execution time tracking
- **🔄 CI/CD Automation**: GitHub Actions with multi-environment testing
- **📈 Coverage Reporting**: Detailed HTML reports with branch coverage
- **🛡️ Security Scanning**: Automated dependency and vulnerability checks

### 🎛️ **Advanced Testing Features**

```bash
# Test with specific biological scenarios
pytest -m "singlecell and geo"        # Single-cell + GEO integration
pytest -m "performance and large_data" # Performance with large datasets
pytest -m "multiomics"                 # Multi-omics integration tests

# Run tests by priority
pytest tests/ --maxfail=5 -v          # Fail fast development mode
pytest tests/ -x                      # Stop on first failure

# Generate performance benchmarks
pytest tests/performance/ --benchmark-only --benchmark-json=results.json
```

### 🔧 **Test Configuration**

The testing framework uses centralized configuration:

- **`tests/test_config.yaml`** - Environment settings, test parameters, performance thresholds
- **`tests/data_registry.json`** - Test dataset registry with metadata and availability
- **`pytest.ini`** - Pytest configuration with markers and coverage settings
- **`.pre-commit-config.yaml`** - Code quality gates and validation hooks

### 🚦 **CI/CD Pipeline**

Automated testing runs on every pull request with:

- **✅ Code Quality**: Black formatting, linting, type checking
- **🧪 Multi-Platform**: Ubuntu, macOS, Windows (Python 3.11, 3.12)
- **🔒 Security**: Bandit, Safety, vulnerability scanning
- **📊 Performance**: Benchmark comparisons and regression detection
- **📈 Coverage**: Automated coverage reporting to Codecov

### 🎯 **Quality Standards**

- **Minimum Coverage**: 80% (targeting 95%+)
- **Test Execution Time**: < 2 minutes for unit tests, < 45 minutes total
- **Biological Accuracy**: Scientifically validated mock data and algorithms
- **Error Recovery**: Comprehensive fault tolerance and graceful degradation testing

## 📚 **Learn More**

- 📖 **[Full Documentation](docs/)** - Complete guides and tutorials
- 🏗️ **[Architecture Overview](docs/architecture_diagram.md)** - Technical deep-dive
- 🧪 **[Example Analyses](examples/)** - Real-world use cases
- 🎓 **[Video Tutorials](https://youtube.com/@homaraai)** - Step-by-step walkthroughs including new Rich CLI features
- ⚗️ **[Testing Guide](tests/README.md)** - Comprehensive testing documentation
- 🎨 **[CLI Enhancement Guide](CLAUDE.md)** - Rich CLI features and orange theming details

## 🔍 **Data Quality & Compliance**

Lobster AI maintains **publication-grade data quality standards** for transcriptomics and proteomics analysis -> [source publication](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9926151): 

### ✅ **Quality Assessment Results**
- **✅ 60% COMPLIANT** - Strong foundational QC infrastructure
- **⚠️ 26% PARTIAL** - Areas identified for enhancement  
- **❌ 14% MISSING** - Clear roadmap for remaining features

### 🏗️ **Robust QC Architecture**
- **Comprehensive Provenance Tracking** - W3C-PROV compliant analysis history
- **Automated Quality Control** - Built-in metrics for genes, cells, and proteins
- **Schema Validation** - Structured metadata for reproducible research
- **Batch Effect Management** - Detection and correction workflows
- **Reproducible Workflows** - Containerized analysis with parameter logging

### 📋 **Key QC Components Analyzed**
- `lobster/tools/quality_service.py` - Quality assessment algorithms
- `lobster/tools/preprocessing_service.py` - Normalization and batch correction  
- `lobster/core/provenance.py` - Complete analysis history tracking
- `lobster/core/schemas/` - Transcriptomics and proteomics metadata validation
- `AGENT_DATA_QC_CHECKLIST.md` - Comprehensive quality requirements checklist

### 🎯 **Next Steps for Highest Quality**
**Priority improvements identified:**
1. **Missing Data Handling** - Implement imputation strategies for proteomics
2. **Reference Harmonization** - Add Ensembl/UniProt version management
3. **Statistical Rigor** - Systematic FDR control across all analyses
4. **Proteomics Enhancement** - Multi-level PSM/peptide/protein QC

📊 **[View Full Quality Report →](AGENT_DATA_QC_CHECKLIST_report.md)**

## 🤝 **Community & Support**

- 💬 **[Discord Community](https://discord.gg/homaraai)** - Chat with users and developers
- 🐛 **[Report Issues](https://github.com/homara-ai/lobster/issues)** - Bug reports and feature requests
- 📧 **[Email Support](mailto:support@homara.ai)** - Direct help from our team
- 🐦 **[Follow Updates](https://twitter.com/homaraai)** - Latest news and releases

## 🏢 **Enterprise Solutions**

Need custom integrations, dedicated support, or on-premise deployment?

**[Contact Enterprise Sales →](mailto:enterprise@homara.ai)**

## 📄 **License**

Open source under [MIT License](LICENSE) - Use freely in academic and commercial projects.

---

<div align="center">

**🦞 Transform Your Bioinformatics Research Today**

[Get Started](https://github.com/homara-ai/lobster) • [Documentation](docs/) • [Community](https://discord.gg/homaraai)

*Made with ❤️ by [Homara AI](https://homara.ai)*

</div>
