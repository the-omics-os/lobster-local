# 🦞 Lobster AI

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation: CC BY 4.0](https://img.shields.io/badge/Documentation-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**Transform your bioinformatics research with AI agents that understand your data and provide expert analysis insights.**
  
## 📋 Table of Contents

- [✨ What is Lobster AI?](#-what-is-lobster-ai)
- [🚀 Quick Start](#-quick-start)
- [💡 Example Usage](#-example-usage)
- [🧬 Features](#-features)
- [🔬 Literature Mining & Metadata](#-literature-mining--metadata)
- [🔧 Configuration](#-configuration)
- [📚 Documentation](#-documentation)
- [🤝 Community & Support](#-community--support)
- [📄 License](#-license)

## ✨ What is Lobster AI?

Lobster AI is a bioinformatics platform that combines specialized AI agents with open-source tools to analyze complex multi-omics data, discover relevant literature, and manage metadata across datasets. Simply describe your analysis needs in natural language - no coding required.

**Perfect for:**
- Bioinformatics researchers analyzing RNA-seq data
- Computational biologists seeking intelligent analysis workflows
- Life science teams requiring reproducible, publication-ready results
- Students learning modern bioinformatics approaches

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher (for native installation)
- An LLM API key (Claude or AWS Bedrock)
- Docker Desktop (for Docker installation - recommended for Windows users)

### Installation by Platform

#### 🍎 macOS Installation (Native - Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local

# 2. Install with make (automatically creates .env file)
make install

# 3. Configure your API key
# The .env file is automatically created during installation
open .env
# Add your API key: ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# 4. Activate the virtual environment
source .venv/bin/activate

# 5. Start analyzing!
lobster chat
# or if you want to see the reasoning
lobster chat --reasoning

# Optional: Install globally to use 'lobster' from any directory
make install-global
```

#### 🐧 Ubuntu/Debian Installation (Native)

```bash
# 1. Install system dependencies (REQUIRED)
sudo apt update
sudo apt install -y \
    build-essential \
    python3.12-dev \
    python3.12-venv \
    libhdf5-dev \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libssl-dev \
    libblas-dev \
    liblapack-dev

# 2. Clone the repository
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local

# 3. Install Lobster
make install

# 4. Configure your API key
nano .env
# Add your API key: ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# 5. Activate and run
source .venv/bin/activate
lobster chat
```

**Alternative for Ubuntu**: Use the helper script
```bash
./install-ubuntu.sh  # Auto-detects and installs system dependencies
```

#### 🪟 Windows Installation

**Option 1: Docker Desktop (Recommended)**

Docker provides the most reliable experience on Windows:

```powershell
# 1. Install Docker Desktop for Windows
# Download from: https://www.docker.com/products/docker-desktop/

# 2. Clone the repository
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local

# 3. Configure your API key
copy .env.example .env
notepad .env
# Add your API key: ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# 4. Run Lobster in Docker
docker-compose run --rm lobster-cli

# OR run as web service
docker-compose up lobster-server
# Access at http://localhost:8000
```

**Option 2: Native Installation (Experimental)**

Native Windows installation is currently experimental. For the best experience, use Docker Desktop.

```powershell
# 1. Install Python 3.12 from python.org
# Download from: https://www.python.org/downloads/

# 2. Clone the repository
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local

# 3. Run the Windows installer
.\install.ps1

# 4. Configure your API key
notepad .env
# Add your API key: ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# 5. Activate and run
.\.venv\Scripts\Activate.ps1
lobster chat
```

**Troubleshooting Windows**: See detailed guide at `docs/WINDOWS_INSTALLATION.md`

---

#### 🐳 Docker Installation (Cross-Platform - Production Ready)

**Prerequisites:**
- Docker 20.10+ and Docker Compose 2.0+ installed
- `.env` file with API keys (see Configuration section)

**Quick Start:**

```bash
# 1. Clone the repository
git clone https://github.com/the-omics-os/lobster-local.git
cd lobster-local

# 2. Configure your API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY or AWS Bedrock credentials

# 3. Build Docker images
make docker-build

# 4. Run interactively (CLI mode)
make docker-run-cli

# OR run as a web service (FastAPI server)
make docker-run-server

# Check server health
curl http://localhost:8000/health
```

**Docker Compose (Multi-Service):**

```bash
# Run CLI interactively
make docker-compose-cli

# Start FastAPI server in background
make docker-compose-up

# Stop all services
make docker-compose-down
```

**Why Docker?**
- ✅ **Isolated environment** - No Python version conflicts
- ✅ **Production-ready** - Includes healthchecks and resource limits
- ✅ **Cloud deployment** - Ready for AWS ECS, Kubernetes, or Docker Swarm
- ✅ **Consistent setup** - Same environment across all machines

📚 **See [Docker Deployment Guide](wiki/43-docker-deployment-guide.md) for:**
- AWS ECS/Fargate deployment
- Kubernetes manifests
- Volume management strategies
- Troubleshooting and best practices

### ⚠️ Important: API Keys & Rate Limits

**Rate Limits**

Anthropic's API has conservative rate limits for new accounts. If you encounter rate limit errors:

1. **Wait and retry** - Limits reset after a short period (typically 60 seconds)
2. **Request increase** - Visit [Anthropic Rate Limits Documentation](https://docs.anthropic.com/en/api/rate-limits)
3. **Use AWS Bedrock** - Recommended for production use with higher limits
4. **Contact us** - Email [info@omics-os.com](mailto:info@omics-os.com) for assistance

**Recommended Setup by Use Case:**

| Use Case | Recommended Provider | Notes |
|----------|---------------------|-------|
| **Quick Testing** | Claude API | May encounter rate limits |
| **Development** | Claude API + Rate Increase | Request higher limits from Anthropic |
| **Production** | AWS Bedrock | Enterprise-grade limits |
| **Heavy Analysis** | AWS Bedrock | Required for large datasets |

For AWS Bedrock setup, see the [Configuration Guide](wiki/03-configuration.md).

### ⚠️ Cell Type Annotation - Development Status

**IMPORTANT: Built-in marker gene lists are preliminary and not scientifically validated.**

The current cell type annotation templates use **hardcoded marker lists** without:
- Evidence scoring (AUC, logFC, specificity metrics)
- Validation against reference atlases (Azimuth, CellTypist, Human Cell Atlas)
- Tissue/context-specific optimization
- Species separation (some mouse genes may be present)
- State handling (activation/injury markers mixed with baseline identity)

**Current limitations:**
- **SASP/Senescence detection**: Not reliable with RNA-seq data alone (removed in v0.1.0)
- **Tumor cell detection**: Should use CNV inference (inferCNV/CopyKAT), not proliferation markers
- **Cross-tissue transfer**: Markers optimized for one tissue may not work in others

**Recommended approach for production analysis:**
1. **Provide custom validated markers** specific to your tissue/context
2. Use reference-based tools: [Azimuth](https://azimuth.hubmapconsortium.org/), [CellTypist](https://www.celltypist.org/), [scANVI](https://docs.scvi-tools.org/)
3. Validate annotations manually with known markers

**Planned improvements:**
- Integration with Azimuth/CellTypist pretrained models
- Reference atlas-derived markers with evidence scores
- UCell/AUCell signature scoring
- CNV-based tumor/normal classification
- Cell Ontology (CL ID) annotations

When using Lobster for annotation, agents will **prompt you for custom markers**. Only use built-in templates if you explicitly acknowledge these limitations.

See [Manual Annotation Guide](wiki/35-manual-annotation-service.md) for details on providing custom markers.

## 💡 Example Usage

### Interactive Chat Mode

```bash
🦞 lobster chat

Welcome to Lobster AI - Your bioinformatics analysis assistant

🦞 You: Download GSE109564 and perform single-cell clustering analysis

🦞 Lobster: I'll download and analyze this single-cell dataset for you...

✓ Downloaded 5,000 cells × 20,000 genes
✓ Quality control: filtered to 4,477 high-quality cells
✓ Identified 12 distinct cell clusters
✓ Generated UMAP visualization and marker gene analysis

Analysis complete! Results saved to workspace.
```

### Single Query Mode

For non-interactive analysis, you can run single queries directly:

```bash
# Basic syntax
lobster query "your analysis request"

# Examples
lobster query "download GSE109564 and perform quality control"
lobster query "load my_data.h5ad and create UMAP plot"
lobster query "differential expression between control and treatment"

# With workspace context
lobster query --workspace ~/my_analysis "cluster the loaded dataset"

# Show reasoning process
lobster query --reasoning "quality control on data.csv"

# Save output to file
lobster query "analyze GSE12345" --output results.txt

# Verbose logging
lobster query --verbose "load and normalize data"
```

**When to use:**

| Use `lobster query` | Use `lobster chat` |
|---------------------|-------------------|
| Scripting/automation | Exploratory analysis |
| Single-task analysis | Multi-step workflows |
| CI/CD pipelines | Interactive debugging |
| Batch processing | Iterative refinement |

### Available Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/files` | List workspace files |
| `/read <file>` | Load a dataset |
| `/archive <file>` | Load data from archives (tar/zip with 10X, Kallisto/Salmon) |
| `/data` | Show current dataset info |
| `/plots` | List generated visualizations |
| `/workspace` | Show workspace information |
| `/workspace list` | List available datasets |
| `/workspace load <name>` | Load specific dataset |

### Natural Language Examples

```bash
# Download and analyze GEO datasets
🦞 You: "Download GSE12345 and perform quality control"

# Analyze your own data
🦞 You: "Load my_data.csv and identify differentially expressed genes"

# Generate visualizations
🦞 You: "Create a UMAP plot colored by cell type"

# Perform complex analyses
🦞 You: "Run pseudobulk aggregation and differential expression between conditions"
```

## 🧬 Features

### Current Capabilities

#### **Single-Cell RNA-seq Analysis**
- Quality control and filtering
- Normalization and scaling
- Clustering and UMAP visualization
- Cell type annotation
- Marker gene identification
- Pseudobulk aggregation

#### **Bulk RNA-seq Analysis**
- Differential expression with pyDESeq2
- R-style formula-based statistics
- Complex experimental designs
- Batch effect correction

#### **Data Management**
- Support for CSV, Excel, H5AD, 10X formats
- Multi-source dataset discovery (GEO, SRA, PRIDE, ENA)
- Literature mining and full-text retrieval
- Cross-dataset metadata harmonization
- Sample ID mapping and validation
- Automatic visualization generation

### 🗓️ Roadmap & Premium Features

Lobster follows an **open-core model**: core transcriptomics capabilities are open source, while advanced features are available in premium tiers.

#### **Open-Core (Public - lobster-local)**
✅ **Available Now:**
- Single-cell & bulk RNA-seq analysis
- Literature mining & dataset discovery
- Research agents & workflow automation
- Protein structure visualization (v2.4+)

#### **Premium Features (Private)**

##### **Q1 2025 - Proteomics Platform** *(In Development)*
- 🔬 Mass spectrometry proteomics (DDA/DIA workflows)
- 🧬 Affinity proteomics (Olink panels, antibody arrays)
- 📊 Missing value handling and normalization
- 🧪 Peptide-to-protein aggregation
- 📈 Differential expression analysis

##### **Q2 2025 - AI Agent Toolkit** *(Private Beta)*
- 🤖 Custom feature agent (code generation with Claude Code SDK)
- 🛠️ Agent creation templates & frameworks
- 🔧 Unified agent development patterns

##### **Q2-Q3 2025 - Multi-Omics Integration**
- 🔗 Cross-platform data integration (RNA + Protein)
- 🎯 Multi-modal analysis workflows
- 📊 Integrated visualization suite

##### **Q3 2025 - Lobster Cloud** *(Launching)*
- ☁️ Scalable cloud computing (AWS Bedrock optimization)
- 🚀 No local hardware requirements
- 🔐 HIPAA/GDPR compliance (SOC2 in progress)
- 💼 Enterprise SaaS ($6K-$30K/year)

**Target:** 50 paying customers, $810K ARR by Month 18

---

**Note:** Premium features are available for enterprise customers and research collaborations. [Contact us](mailto:info@omics-os.com) for access.

## 🔬 Literature Mining & Metadata

### Literature Discovery

Lobster AI can automatically search scientific literature and extract key information to inform your analyses:

- **Search across databases** - Find relevant papers from PubMed, bioRxiv, and other scientific repositories
- **Full-text retrieval** - Automatically access complete articles when available
- **Methods extraction** - Extract experimental protocols, software parameters, and statistical approaches
- **Citation networks** - Discover related papers and build comprehensive literature reviews
- **Batch processing** - Analyze multiple publications simultaneously

### Dataset Discovery & Validation

Before downloading or analyzing data, Lobster helps you find and evaluate datasets:

- **Multi-source search** - Search across GEO, SRA, PRIDE, and ENA databases
- **Automatic metadata extraction** - Get platform details, sample counts, and experimental conditions
- **Publication linking** - Connect datasets to their associated research papers
- **Pre-download validation** - Check dataset quality, control samples, and platform consistency
- **Compatibility assessment** - Evaluate whether datasets meet your analysis requirements

### Cross-Dataset Metadata Operations

Harmonize and validate metadata across multiple studies:

- **Sample ID mapping** - Match samples between different omics datasets (e.g., RNA-seq to proteomics)
- **Metadata standardization** - Convert diverse metadata formats to common schemas
- **Quality control** - Validate experimental designs and detect potential issues
- **Meta-analysis preparation** - Harmonize metadata across multiple studies for combined analysis
- **Multi-omics integration** - Ensure sample compatibility across different data types

### Natural Language Examples

```bash
# Literature discovery
🦞 You: "Find recent papers about CRISPR screens in cancer and extract their methods"

# Dataset search and validation
🦞 You: "Search GEO for single-cell datasets of pancreatic beta cells with at least 50 samples"

# Cross-dataset operations
🦞 You: "Map sample IDs between my RNA-seq and proteomics datasets"

# Meta-analysis preparation
🦞 You: "Check if these three breast cancer datasets are compatible for meta-analysis"

# Automated metadata extraction
🦞 You: "What analysis parameters did the authors use in PMID:35042229?"
```

## 🔧 Configuration

### API Keys

The `.env` file is automatically created during installation (`make install` calls `setup-env`). You just need to edit it with your API credentials.

**Edit the .env file:**
```bash
# macOS - Opens in default text editor (TextEdit, etc.)
open .env

# Linux - Use nano or your preferred editor
nano .env

# Windows (⚠️ untested platform)
notepad .env
```

**Choose ONE LLM provider:**

**Option 1: Claude API (Recommended for quick start)**
```bash
# Get your key from https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```
⚠️ Note: Claude API has rate limits for new accounts. For production use, consider AWS Bedrock.

**Option 2: AWS Bedrock (Recommended for production)**
```bash
# Requires AWS account with Bedrock access
AWS_BEDROCK_ACCESS_KEY=AKIA...
AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret-key
```

### Optional Configuration

```bash
# Enhanced literature search (optional)
NCBI_API_KEY=your-ncbi-api-key  # Get from NCBI

# Force specific provider (auto-detected by default)
LOBSTER_LLM_PROVIDER=anthropic  # or "bedrock"

# Cloud mode (optional - contact info@omics-os.com for access)
LOBSTER_CLOUD_KEY=your-cloud-api-key
```

### Platform Support

| Platform | Native Installation | Docker | Status |
|----------|---------------------|--------|--------|
| **macOS** | ✅ Fully supported | ✅ Supported | Production ready |
| **Ubuntu/Debian** | ✅ Supported (system deps required) | ✅ Supported | Production ready |
| **Other Linux** | ⚠️ Manual setup needed | ✅ Supported | Community tested |
| **Windows 10/11** | ⚠️ Experimental (use install.ps1) | ✅ Fully supported | Docker recommended |

**Recommended Installation Method:**
- **macOS/Ubuntu**: Native installation (simpler, faster)
- **Windows**: Docker Desktop (most reliable)
- **Enterprise/Production**: Docker (consistent across environments)

## 📚 Documentation

- [Full Documentation](https://github.com/the-omics-os/lobster-local/wiki) - Guides and tutorials
- [Example Analyses](https://github.com/the-omics-os/lobster-local/wiki/27-examples-cookbook) - Real-world use cases
- [Architecture Overview](https://github.com/the-omics-os/lobster-local/wiki/18-architecture-overview) - Technical details

## 🤝 Community & Support

- 🐛 [Report Issues](https://github.com/the-omics-os/lobster-local/issues) - Bug reports and feature requests
- 📧 [Email Support](mailto:info@omics-os.com) - Direct help from our team

### Enterprise Solutions

Need custom integrations or dedicated support? [Contact us](mailto:info@omics-os.com)

## 📄 License

Lobster AI is open source under the Apache License 2.0 (see `LICENSE`). Documentation is licensed CC-BY-4.0.
Contributions are accepted under a Contributor License Agreement to preserve future licensing flexibility.

---

<div align="center">

**Transform Your Bioinformatics Research Today**

[Get Started](https://github.com/the-omics-os/lobster-local) • [Documentation](docs/)

*Made with ❤️ by [Omics-OS](https://omics-os.com)*

</div>