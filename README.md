# 🦞 Lobster AI - Local Installation Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**Complete Local Installation & Configuration Guide**

This guide provides comprehensive instructions for installing and configuring Lobster AI locally. For a quick overview, see the main [README.md](README.md).

## 🌟 **Cloud Platform Coming Soon!**

While you can use Lobster AI locally today, we're actively developing a **cloud-hosted platform** that will provide:

- ☁️ **Zero Setup Experience** - Run analyses directly in your browser
- 🚀 **Unlimited Scalability** - Process massive datasets on cloud infrastructure
- 💾 **Persistent Storage** - Your analyses saved and accessible anywhere
- 👥 **Team Collaboration** - Share projects and work together in real-time
- 🔧 **Managed Infrastructure** - No need to manage dependencies or API keys
- 📊 **Enhanced Analytics** - Advanced monitoring and usage insights

**Expected Launch: Q1 2025** | **[Join Early Access Waitlist →](mailto:cloud@homara.ai?subject=Lobster%20Cloud%20Early%20Access)**

For now, this local installation provides the complete Lobster AI experience on your own hardware.

## 🏗️ **Architecture Overview**

Lobster AI features a **modular, cloud-ready architecture** designed for both local and distributed deployment:

### 📦 **Package Structure**

Lobster AI is organized into 4 modular packages:

| Package | Purpose | Status |
|---------|---------|---------|
| **lobster-core** | Shared interfaces and base classes | ✅ Available |
| **lobster-local** | Full local implementation (this guide) | ✅ Available |
| **lobster-cloud** | Minimal cloud client | 🚧 Coming Soon |
| **lobster-server** | AWS serverless backend | 🚧 Coming Soon |

### 🔄 **Smart CLI Router**

The Lobster CLI automatically detects your environment:

```bash
# Local mode (current)
lobster chat
# Output: 💻 Using Lobster Local

# Cloud mode (coming soon)
export LOBSTER_CLOUD_KEY=your-api-key
lobster chat  
# Output: 🌩️ Using Lobster Cloud
```

## 🚀 **Core Capabilities**

### 🤖 **Multi-Agent System**
- **Data Expert**: Multi-omics data management and loading
- **Research Agent**: Literature discovery and dataset identification  
- **Method Expert**: Computational parameter extraction from publications
- **Transcriptomics Expert**: Single-cell and bulk RNA-seq analysis
- **Proteomics Expert**: Mass spectrometry and protein analysis

### 🧬 **Analysis Features**
- **Single-Cell RNA-seq**: Quality control, filtering, normalization, clustering, marker genes
- **Bulk RNA-seq**: Differential expression, pathway analysis, batch correction
- **Proteomics**: Missing value handling, statistical analysis, protein networks
- **Literature Mining**: PubMed integration for method parameters and validation
- **Multi-Modal Integration**: Cross-omics analysis using MuData framework

### 📊 **Data Management**
- **GEO Integration**: Automatic dataset download and processing
- **Format Support**: CSV, TSV, Excel, H5AD, 10X MTX, and more
- **Schema Validation**: Flexible validation with exploratory analysis support
- **Provenance Tracking**: Complete W3C-PROV-like audit trails

## 📦 **Installation Instructions**

### 🎯 **Quick Install (Recommended)**

```bash
git clone https://github.com/homara-ai/lobster.git
cd lobster
make install
```

This automated installer will:
- ✅ Verify Python 3.12+ installation
- ✅ Create isolated virtual environment
- ✅ Install all required dependencies
- ✅ Set up configuration templates
- ✅ Run verification tests

### 🔧 **Manual Installation**

If you prefer manual control or the automated installer fails:

```bash
# Clone repository
git clone https://github.com/homara-ai/lobster.git
cd lobster

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in development mode
pip install -e .

# Install additional dependencies
pip install -r requirements.txt
```

### 🏗️ **Modular Development Installation**

For developers working with the modular architecture:

```bash
# Install all packages in development mode
./dev_install.sh

# Or manually:
pip install -e ./lobster-core
pip install -e ./lobster-local
pip install -e ./lobster-cloud  # Cloud components (optional)
```

### 🐳 **Docker Installation**

```bash
# Build and run with Docker
docker build -f Dockerfile -t lobster-ai:latest .
docker run -p 8501:8501 --env-file .env lobster-ai:latest

# Or use Docker Compose
docker-compose up
```

## 🔧 **Configuration**

### 📋 **Environment Setup**

Create a `.env` file in your working directory:

```env
# Required API Keys
OPENAI_API_KEY=your-openai-key-here
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key  
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key

# Optional Configurations
NCBI_API_KEY=your-ncbi-api-key  # For enhanced literature search
GENIE_PROFILE=production         # Model configuration preset

# Advanced Settings
GENIE_MAX_FILE_SIZE_MB=500
GENIE_CLUSTER_RESOLUTION=0.5
GENIE_CACHE_DIR=data/cache
```

### 🎛️ **Model Profiles**

Configure AI model usage with built-in profiles:

```bash
# High-performance for complex analyses
export GENIE_PROFILE=high-performance

# Cost-optimized for routine tasks  
export GENIE_PROFILE=cost-optimized

# Development profile for testing
export GENIE_PROFILE=development
```

Available profiles: `development`, `production`, `high-performance`, `cost-optimized`, `eu-compliant`

### 🔍 **Configuration Management**

```bash
# Interactive configuration setup
lobster config generate-env

# View current configuration
lobster config show-config

# Test configuration
lobster config test --profile production

# List available models
lobster config list-models
```

## 🖥️ **Command Line Interface**

### 💬 **Interactive Chat Mode**

```bash
# Start interactive session
lobster chat

# With debugging enabled
lobster chat --reasoning --debug

# Custom workspace
lobster chat --workspace ./my-analysis
```

### 📝 **Single Query Mode**

```bash
# Process single queries directly
lobster query "Download and analyze GSE109564 from GEO"

# Save output to file
lobster query "Perform clustering analysis" --output results.md
```

### 🌐 **Web Interface**

```bash
# Launch Streamlit web interface
streamlit run lobster/streamlit_app.py

# Or use the built-in command
lobster serve --port 8501
```

### 🔧 **Essential Commands**

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/status` | Display system status |
| `/files` | List workspace files |
| `/data` | Show current data summary |
| `/plots` | List generated visualizations |
| `/read <file>` | Display file contents |
| `/save` | Save current state |
| `/export` | Export session data |
| `/reset` | Reset conversation |

## 🧬 **Usage Examples**

### 🔬 **Single-Cell RNA-seq Analysis**

```bash
🦞 You: Download GSE109564 and perform single-cell analysis

🦞 Lobster: I'll download and analyze the single-cell dataset...

[Data Expert Agent]
✓ Downloaded GSE109564 as modality 'geo_gse109564'
✓ Loaded: 5,000 cells × 20,000 genes using transcriptomics_single_cell adapter
✓ Schema validation: 2 warnings, no errors

[Transcriptomics Expert Agent]  
✓ Quality assessment: Mean 3,500 genes/cell, 8.2% mitochondrial
✓ Filtered modality: 4,477 cells × 18,543 genes (removed low-quality data)
✓ Clustered modality: 12 clusters identified using Leiden algorithm
✓ Marker genes: Found distinctive markers for each cluster

Analysis complete! The dataset shows 12 distinct cell populations.
```

### 🧪 **Multi-Omics Integration**

```bash
🦞 You: Load transcriptomics data GSE12345 and proteomics data proteins.csv, then perform integrated analysis

🦞 Lobster: I'll set up multi-omics analysis with both datasets...

[Data Expert Agent]
✓ Loaded transcriptomics: 'geo_gse12345' (3,200 cells × 15,000 genes)
✓ Loaded proteomics: 'ms_proteins' (3,200 cells × 1,247 proteins) 
✓ Created MuData: integrated_analysis.h5mu

[Multi-Modal Analysis]
✓ Cross-modal correlation analysis
✓ Integrated dimensionality reduction  
✓ Multi-omics clustering with 8 joint clusters
✓ Protein-RNA correlation networks

The integrated analysis reveals 8 multi-omics cell states with distinct signatures.
```

## 🔬 **Advanced Features**

### 📚 **Literature Integration**

```python
# Programmatic usage with literature mining
from lobster import LobsterClient

client = LobsterClient()
result = client.query(
    "Find optimal clustering parameters for my single-cell data based on recent publications"
)
```

### 🏗️ **Modular Architecture Usage**

```python
# Direct DataManagerV2 usage
from lobster.core.data_manager_v2 import DataManagerV2

# Initialize data manager
dm = DataManagerV2(workspace_path="./analysis")

# Load multiple data modalities
dm.load_modality("rna_seq", "data.csv", "transcriptomics_single_cell")
dm.load_modality("proteins", "proteins.csv", "proteomics_ms")

# Integrated analysis
mudata = dm.to_mudata()
dm.save_mudata("integrated_study.h5mu")
```

### 🎯 **Available Data Adapters**

| Adapter | Data Type | Formats | Features |
|---------|-----------|---------|----------|
| `transcriptomics_single_cell` | Single-cell RNA-seq | CSV, TSV, H5AD, MTX | Mitochondrial flagging, doublet detection |
| `transcriptomics_bulk` | Bulk RNA-seq | CSV, TSV, H5AD | Batch correction, DE analysis |
| `proteomics_ms` | Mass Spectrometry | CSV, TSV | Missing value imputation, contaminant removal |
| `proteomics_affinity` | Antibody Arrays | CSV, TSV | Signal normalization, background correction |

## 📊 **Workspace Management**

### 📁 **File Organization**

```
.lobster_workspace/
├── data/           # Raw and processed datasets
├── plots/          # Generated visualizations
├── exports/        # Analysis reports and exports
├── cache/          # Cached computations
└── provenance/     # Analysis history and logs
```

### 💾 **Data Export & Reproducibility**

```bash
# Export complete analysis
/export

# This creates a ZIP containing:
# - Raw and processed data files
# - Interactive HTML + static PNG plots  
# - Complete methodology report
# - Tool parameters and timestamps
# - Full provenance trail
```

## 🔧 **Troubleshooting**

### 🚨 **Common Issues**

**Installation Problems:**
```bash
# Clear package cache
pip cache purge

# Reinstall with no cache
pip install --no-cache-dir -e .

# Check Python version
python --version  # Must be 3.12+
```

**API Key Issues:**
```bash
# Verify environment variables
echo $OPENAI_API_KEY
echo $AWS_BEDROCK_ACCESS_KEY

# Test API connectivity
lobster config test
```

**Memory Issues:**
```bash
# Reduce memory usage
export GENIE_MAX_FILE_SIZE_MB=100

# Use lightweight models
export GENIE_PROFILE=cost-optimized
```

### 📞 **Getting Help**

- 📚 **[Full Documentation](docs/)** - Comprehensive guides
- 💬 **[Discord Community](https://discord.gg/homaraai)** - Real-time help
- 🐛 **[GitHub Issues](https://github.com/homara-ai/lobster/issues)** - Bug reports
- 📧 **[Email Support](mailto:support@homara.ai)** - Direct assistance

## 🛣️ **Roadmap & Cloud Migration**

### 🚀 **Upcoming Cloud Features**

**Q1 2025 - Cloud Beta Launch:**
- ☁️ Fully managed cloud infrastructure
- 🔄 Seamless local-to-cloud migration tools
- 📊 Enhanced web interface with real-time collaboration
- 🔒 Enterprise security and compliance features

**Q2 2025 - Advanced Features:**
- 🤖 Enhanced AI models with domain-specific training
- 📈 Advanced analytics and experiment tracking
- 🔗 Integration with popular data platforms
- 📱 Mobile companion app

### 🔄 **Migration Path**

When the cloud platform launches, migrating will be seamless:

1. **Export Current Work**: Use `/export` to package your analyses
2. **Cloud Account Setup**: Automatic migration of workspace data  
3. **Hybrid Usage**: Continue using local installation alongside cloud
4. **Full Migration**: Optional complete transition to cloud-only usage

Your local installation will remain fully functional and continue receiving updates.

## 🤝 **Contributing**

We welcome contributions to Lobster AI! See our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Development setup
make dev-install
make test
make format

# Run local development
lobster chat --reasoning --debug
```

## 📄 **License & Acknowledgments**

- **License**: MIT License - see [LICENSE](LICENSE) for details
- **Built with**: [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain)
- **Bioinformatics**: [Scanpy](https://scanpy.readthedocs.io/), [BioPython](https://biopython.org/)
- **Created by**: [Homara AI](https://homara.ai)

---

<div align="center">

**🦞 Ready to Transform Your Bioinformatics Research?**

[Get Started Now](https://github.com/homara-ai/lobster) • [Join Community](https://discord.gg/homaraai) • [Cloud Waitlist](mailto:cloud@homara.ai)

*Experience the future of bioinformatics analysis today, with cloud deployment coming soon.*

</div>
