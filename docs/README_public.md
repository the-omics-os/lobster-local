# 🦞 Lobster AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**AI-Powered Multi-Omics Bioinformatics Analysis Platform**

Transform your bioinformatics research with intelligent AI agents that understand your data and provide expert analysis insights. Lobster AI combines specialized AI agents with proven scientific tools to analyze complex multi-omics data through natural language interactions.

## 🚀 Key Features

### 🤖 **Specialized AI Agents**
- **Data Expert**: Multi-omics data loading, format conversion, and quality assessment
- **Research Agent**: Literature discovery and dataset identification from GEO/PubMed
- **Transcriptomics Expert**: Single-cell and bulk RNA-seq analysis workflows
- **MS Proteomics Expert**: Mass spectrometry with DDA/DIA workflows and missing value handling
- **Affinity Proteomics Expert**: Olink, SOMAscan, and antibody-based assay analysis
- **Method Expert**: Computational parameter extraction from scientific literature

### 🧬 **Analysis Capabilities**

**Single-Cell RNA-seq**
- Quality control and filtering with mitochondrial gene flagging
- Normalization, clustering (Leiden algorithm), and cell type annotation
- Trajectory analysis and marker gene identification
- Doublet detection and batch effect correction

**Bulk RNA-seq**
- Differential expression analysis with multiple testing correction
- Pathway enrichment and functional analysis
- Batch correction and normalization strategies

**Mass Spectrometry Proteomics**
- Missing value pattern analysis (30-70% missing data handling)
- Peptide-to-protein aggregation with statistical validation
- Intensity normalization (TMM, quantile, VSN)
- Database search artifact removal and quality filtering

**Affinity Proteomics (Olink/SOMAscan)**
- Coefficient of variation analysis for technical reproducibility
- Cross-platform harmonization and panel comparison
- Antibody validation and quality scoring

**Multi-Omics Integration**
- Cross-modal correlation analysis using MuData framework
- Integrated dimensionality reduction and joint clustering
- Protein-RNA correlation networks

### 📊 **Professional Visualization**
- Publication-ready plots: volcano plots, heatmaps, UMAP/t-SNE
- Interactive NetworkX-based protein interaction maps
- Comprehensive QC dashboards with missing value analysis
- Statistical result plots with p-value distributions

## 🔬 **Industry-Leading Proteomics Platform**

Lobster AI provides **comprehensive proteomics analysis** with specialized agents and professional-grade algorithms:

### **🎯 Mass Spectrometry Excellence**
- **DDA/DIA Workflows**: Complete data-dependent and data-independent acquisition pipelines
- **Missing Value Intelligence**: Sophisticated handling of 30-70% missing values typical in MS data
- **Peptide-to-Protein Mapping**: Professional aggregation algorithms with statistical validation
- **Database Search Integration**: Support for MaxQuant, MSFragger, Spectronaut, and other tools

### **🎯 Affinity Proteomics Specialization**
- **Olink Panel Support**: Specialized workflows for targeted protein panels
- **Antibody Validation**: Quality assessment tools for antibody-based assays
- **Low Missing Values**: Optimized for <30% missing values in affinity data
- **Cross-Platform Harmonization**: Integration across different affinity technologies

### **📊 Advanced Statistical Analysis**
- **Differential Expression**: Multiple testing correction with FDR control
- **Pathway Analysis**: Gene set enrichment with protein-specific databases
- **Quality Control**: Multi-level assessment (PSM, peptide, protein levels)
- **Statistical Modeling**: Linear mixed models for complex experimental designs

## 🎨 **User-Friendly Experience**

### **Natural Language Interface**
- **Conversational Analysis**: Describe your analysis needs in plain English
- **Intelligent Agent Coordination**: AI agents automatically hand off tasks to specialists
- **Interactive Chat Mode**: Real-time analysis workflow with immediate feedback
- **Zero Learning Curve**: No command-line expertise or scripting required

### **Professional Output & Reproducibility**
- **Automatic Visualization**: Publication-ready plots generated automatically
- **Complete Provenance**: W3C-PROV compliant analysis trail for reproducibility
- **Format Flexibility**: Support for CSV, Excel, H5AD, 10X MTX, MaxQuant, Olink NPX, and more
- **Export Capabilities**: Save complete analysis sessions with full metadata

### **🎯 Perfect For**
- **Bioinformatics Researchers** analyzing RNA-seq, proteomics, and multi-omics data
- **Computational Biologists** seeking intelligent analysis workflows  
- **Life Science Teams** requiring reproducible, publication-ready results
- **Students & Educators** learning modern bioinformatics approaches

## 🎬 Quick Start

### Installation
```bash
git clone https://github.com/omics-os/lobster.git
cd lobster
make install
lobster chat  # Start analyzing immediately!
```

### Usage Examples

**Single-Cell Analysis**
```bash
🦞 You: "Download GSE109564 and perform single-cell clustering analysis"

🦞 Lobster: I'll download and analyze this single-cell dataset for you...
✓ Downloaded 5,000 cells × 20,000 genes
✓ Quality control: filtered to 4,477 high-quality cells  
✓ Identified 12 distinct cell clusters
✓ Generated UMAP visualization and marker gene analysis
```

**Proteomics Analysis**
```bash
🦞 You: "Analyze my proteomics data with missing value assessment"

🦞 Lobster: I'll analyze your mass spectrometry proteomics data...
✓ Loaded 2,847 proteins × 24 samples
✓ Missing value analysis: 42% missing (typical for MS data)
✓ Identified 156 significantly regulated proteins (p < 0.05)
✓ Generated volcano plot and pathway enrichment analysis
```

## 🔧 Configuration

### Environment Setup
```env
# Required API Keys
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key  
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key

# Optional
NCBI_API_KEY=your-ncbi-api-key  # Enhanced literature search
GENIE_PROFILE=production         # Model configuration preset
```

### Command Line Interface
```bash
lobster chat                     # Interactive analysis mode
lobster query "your question"    # Single query mode
lobster serve --port 8501        # Web interface
```

### Essential Commands
- `/help` - Show all available commands
- `/data` - Show current data summary
- `/plots` - List generated visualizations
- `/export` - Export session data with full provenance

## 🏗️ Architecture

**Modular Design**: 4 packages (lobster-core, lobster-local, lobster-cloud, lobster-server)  
**Data Formats**: CSV, TSV, Excel, H5AD, 10X MTX, MaxQuant, Spectronaut, Olink NPX  
**Quality Standards**: 95%+ test coverage, W3C-PROV compliant provenance tracking  
**Cloud Integration**: Smart CLI router with automatic fallback to local mode

## ☁️ Lobster Cloud: Seamless Cloud Integration

Experience the power of cloud computing with **automatic cloud detection**:
- ☁️ **Zero Configuration** - Just set your API key and go
- 🚀 **Scalable Computing** - Handle large datasets without local hardware limits  
- 🔄 **Seamless Switching** - Automatic fallback to local mode if needed
- 🔒 **Secure Processing** - Enterprise-grade security for your data

### **Getting Started with Cloud**

1. **Get your API key** from [cloud.lobster.ai](mailto:kevin.yar@omics-os.com?subject=Lobster%20Cloud%20API%20Key%20Request)
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

**[Request Cloud Access →](mailto:kevin.yar@omics-os.com?subject=Lobster%20Cloud%20API%20Key%20Request)**

## 🧪 Testing & Quality

Comprehensive testing framework with 95%+ coverage:
```bash
make test                        # Run all tests
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests
pytest --cov=lobster            # Coverage reporting
```

**Quality Standards**:
- ✅ 60% COMPLIANT with publication-grade data quality standards
- ⚠️ 26% PARTIAL compliance with enhancement roadmap
- Automated CI/CD with multi-platform testing (Ubuntu, macOS, Windows)

## 📞 Support & Community

- 📚 **[Documentation](docs/)** - Complete guides and tutorials
- 🐛 **[GitHub Issues](https://github.com/omics-os/lobster/issues)** - Bug reports
- 📧 **[Email Support](mailto:kevin.yar@omics-os.com)** - Direct assistance
- 🌐 **[Website](https://www.omics-os.com)** - Latest updates and news

## 📄 License & Attribution

- **License**: MIT License - Free for academic and commercial use
- **Built with**: LangGraph, LangChain, Scanpy, BioPython
- **Created by**: [Omics-OS](https://www.omics-os.com) - Founded by Kevin Yar

---

<div align="center">

**🦞 Transform Your Bioinformatics Research Today**

[Get Started](https://github.com/omics-os/lobster) • [Visit Website](https://www.omics-os.com) • [Cloud Waitlist](mailto:kevin.yar@omics-os.com)

*Experience the future of bioinformatics analysis today, with cloud deployment coming soon.*

</div>