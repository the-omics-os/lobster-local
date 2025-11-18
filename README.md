# 🦞 Lobster AI

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL%203.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation: CC BY 4.0](https://img.shields.io/badge/Documentation-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**Transform your bioinformatics research with AI agents that understand your data and provide expert analysis insights.**

## 📋 Table of Contents

- [✨ What is Lobster AI?](#-what-is-lobster-ai)
- [⚡ Quick Start](#-quick-start)
- [💡 Example Usage](#-example-usage)
- [🧬 Features](#-features)
- [🚀 Installation](#-installation)
- [🔬 Literature Mining & Metadata](#-literature-mining--metadata)
- [🔧 Configuration](#-configuration)
- [🗓️ Roadmap](#-roadmap)
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

## ⚡ Quick Start (60 seconds)

```bash
# 1. Install
pip install lobster-ai

# 2. Configure API key (add NCBI for higher requests)
open .env
ANTHROPIC_API_KEY=<your key>

# 3. Run
lobster chat
```

**Get API keys:** [Claude API](https://console.anthropic.com/) | [AWS Bedrock](https://aws.amazon.com/bedrock/)

**First analysis:**
```bash
lobster query "Download GSE109564 and perform clustering"
```

[See detailed installation options](#-installation) | [Configuration guide](https://github.com/the-omics-os/lobster-local/wiki/03-configuration)

## 💡 Example Usage

### Interactive Chat Mode

```bash
lobster chat

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

For non-interactive analysis and automation:

```bash
# Basic usage
lobster query "download GSE109564 and perform quality control"

# With workspace context
lobster query --workspace ~/my_analysis "cluster the loaded dataset"

# Show reasoning process
lobster query --reasoning "differential expression between conditions"
```

### Natural Language Examples

```bash
# Download and analyze GEO datasets
🦞 You: "Download GSE12345 and perform quality control"

# Analyze your own data
🦞 You: "Load my_data.csv and identify differentially expressed genes"

# Generate visualizations
🦞 You: "Create a UMAP plot colored by cell type"

# Complex analyses
🦞 You: "Run pseudobulk aggregation and differential expression"
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

## 🚀 Installation

### Primary Method: PyPI (Recommended)

Install Lobster AI with a single command:

```bash
pip install lobster-ai
```

**Configure API Keys:**

Create a `.env` file in your working directory:

```bash
# Required: Choose ONE LLM provider

# Option 1: Claude API (Quick testing)
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Option 2: AWS Bedrock (Production)
AWS_BEDROCK_ACCESS_KEY=your-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret-key

# Optional: Enhanced literature search
NCBI_API_KEY=your-ncbi-key
NCBI_EMAIL=your.email@example.com
```

**Get API Keys:**
- **Claude API**: https://console.anthropic.com/
- **AWS Bedrock**: https://aws.amazon.com/bedrock/
- **NCBI API**: https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/

**Note:** The `.env` file must be in your current working directory when running Lobster.

---

### Alternative: Docker (Cross-Platform)

**Best for:** Windows users, production deployments, isolated environments

```bash
# 1. Configure API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# 2. Run with Docker
docker run -it --rm \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  ghcr.io/the-omics-os/lobster-local:latest chat
```

**Why Docker?**
- ✅ No Python installation required
- ✅ Consistent environment across all platforms
- ✅ Isolated from system dependencies

---

### Platform-Specific Installation

For native installation (development, advanced users):

- **macOS**: [Native Installation Guide](https://github.com/the-omics-os/lobster-local/wiki/02-installation#macos)
- **Linux**: [Ubuntu/Debian Guide](https://github.com/the-omics-os/lobster-local/wiki/02-installation#linux-ubuntudebian)
- **Windows**: [Docker Recommended](https://github.com/the-omics-os/lobster-local/wiki/02-installation#windows) or [WSL Guide](https://github.com/the-omics-os/lobster-local/wiki/02-installation#windows)

**Complete installation guide:** [wiki/02-installation.md](https://github.com/the-omics-os/lobster-local/wiki/02-installation)

---

### ⚠️ Important: API Rate Limits

**Claude API:**
- ⚠️ Conservative rate limits for new accounts
- ✅ Best for: Testing, development, small datasets
- 📈 Upgrade: [Request limit increase](https://docs.anthropic.com/en/api/rate-limits)

**AWS Bedrock:**
- ✅ Enterprise-grade rate limits (recommended for production)
- ✅ Best for: Large-scale analysis, production deployments
- 🔗 Setup: [AWS Bedrock Guide](https://github.com/the-omics-os/lobster-local/wiki/02-installation#aws-bedrock-enhanced-setup)

If you encounter rate limit errors: [Troubleshooting Guide](https://github.com/the-omics-os/lobster-local/wiki/28-troubleshooting)

## 🔬 Literature Mining & Metadata

Lobster AI automatically searches scientific literature and extracts key information to inform your analyses:

- **Search across databases** - Find relevant papers from PubMed, bioRxiv, and other repositories
- **Full-text retrieval** - Automatically access complete articles when available
- **Methods extraction** - Extract experimental protocols, software parameters, and statistical approaches
- **Dataset discovery** - Search across GEO, SRA, PRIDE, and ENA databases
- **Metadata harmonization** - Convert diverse metadata formats to common schemas
- **Sample ID mapping** - Match samples between different omics datasets

### Natural Language Examples

```bash
# Literature discovery
🦞 You: "Find recent papers about CRISPR screens in cancer"

# Dataset search
🦞 You: "Search GEO for single-cell datasets of pancreatic beta cells"

# Cross-dataset operations
🦞 You: "Concatenate multiple single-cell RNA-seq batches and correct for batch effects"

# Automated extraction
🦞 You: "What analysis parameters did the authors use in PMID:35042229?"
```

## 🔧 Configuration

Lobster AI is configured via the `.env` file in your working directory.

**Minimal configuration** (choose one):

```bash
# Option A: Claude API
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Option B: AWS Bedrock
AWS_BEDROCK_ACCESS_KEY=your-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret-key
```

**Optional settings:**
```bash
# Enhanced literature search
NCBI_API_KEY=your-ncbi-api-key
NCBI_EMAIL=your.email@example.com

# Performance tuning
LOBSTER_PROFILE=production
LOBSTER_MAX_FILE_SIZE_MB=500
```

**Complete configuration guide:** [wiki/03-configuration.md](https://github.com/the-omics-os/lobster-local/wiki/03-configuration)

## 🗓️ Roadmap

Lobster follows an **open-core model**: core transcriptomics is open source, advanced features in premium tiers.

**Open Source (lobster-local):**
- ✅ Single-cell & bulk RNA-seq analysis
- ✅ Literature mining & dataset discovery
- ✅ Protein structure visualization

**Premium Features:**
- Q1 2025: Proteomics platform (DDA/DIA workflows)
- Q2 2025: AI agent toolkit & custom feature generation
- Q3 2025: Lobster Cloud (SaaS, $6K-$30K/year)

**Target:** 50 paying customers, $810K ARR by Month 18

[Full roadmap & pricing](https://github.com/the-omics-os/lobster-local/wiki) | [Contact for enterprise access](mailto:info@omics-os.com)

## 📚 Documentation

- [Full Documentation](https://github.com/the-omics-os/lobster-local/wiki) - Guides and tutorials
- [Example Analyses](https://github.com/the-omics-os/lobster-local/wiki/27-examples-cookbook) - Real-world use cases
- [Architecture Overview](https://github.com/the-omics-os/lobster-local/wiki/18-architecture-overview) - Technical details
- [API Reference](https://github.com/the-omics-os/lobster-local/wiki/13-api-overview) - Complete API documentation

## 🤝 Community & Support

- 🐛 [Report Issues](https://github.com/the-omics-os/lobster-local/issues) - Bug reports and feature requests
- 📧 [Email Support](mailto:info@omics-os.com) - Direct help from our team

### Enterprise Solutions

Need custom integrations or dedicated support? [Contact us](mailto:info@omics-os.com)

## 📄 License

Lobster AI is open source under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). This license ensures that all users, including those accessing the software over a network, receive the freedoms to use, study, share, and modify the software. The AGPL-3.0 license is compatible with GPL-licensed dependencies used in our bioinformatics toolchain.

For commercial licensing options or questions about license compatibility, please contact us at info@omics-os.com.

Documentation is licensed CC-BY-4.0. Contributions are accepted under a Contributor License Agreement to preserve future licensing flexibility.

---

<div align="center">

**Transform Your Bioinformatics Research Today**

[Get Started](#-quick-start) • [Documentation](https://github.com/the-omics-os/lobster-local/wiki)

*Made with ❤️ by [Omics-OS](https://omics-os.com)*

</div>
