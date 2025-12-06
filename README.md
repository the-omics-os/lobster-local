# 🦞 Lobster AI

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL%203.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation: CC BY 4.0](https://img.shields.io/badge/Documentation-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Bioinformatics co-pilot to automate redundant tasks so you can focus on science**

## 📋 Table of Contents

- [✨ What is Lobster AI?](#-what-is-lobster-ai)
- [⚡ Quick Start](#-quick-start)
- [💡 Example Usage](#-example-usage)
- [🧬 Features](#-features)
- [🚀 Installation](#-installation)
- [🔬 Literature Mining & Metadata](#-literature-mining--metadata)
- [🔧 Configuration](#-configuration)
- [🏠 Local LLM Support](#-local-llm-support-new)
- [⭐ Premium Features](#-premium-features)
- [🗓️ Roadmap](#-roadmap)
- [📚 Documentation](#-documentation)
- [🤝 Community & Support](#-community--support)
- [🛠️ For Developers](#-for-developers)
- [📄 License](#-license)

## ✨ What is Lobster AI?

Lobster AI is a bioinformatics platform that combines specialized AI agents with open-source tools to analyze complex multi-omics data, discover relevant literature, and manage metadata across datasets. Simply describe your analysis needs in natural language - no coding required.

**Perfect for:**
- Bioinformatics researchers analyzing RNA-seq data
- Computational biologists seeking intelligent analysis workflows
- Life science teams requiring reproducible, publication-ready results
- Students learning modern bioinformatics approaches

## ⚡ Quick Start

### Option 1: Global Installation (Recommended for CLI Use)

```bash
# Install uv if not already installed
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install Lobster globally
uv tool install lobster-ai

# Configure API keys
lobster init

# Start using Lobster
lobster chat
```

**Benefits**: Accessible from anywhere, clean uninstall, isolated environment.

### Option 2: Local Installation (For Projects/Development)

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Lobster in virtual environment
uv pip install lobster-ai
# or: pip install lobster-ai

# Configure API keys
lobster init

# Start using Lobster
lobster chat
```

**Benefits**: Project-specific installation, doesn't affect system Python.

---

**Get API keys:** [Claude API](https://console.anthropic.com/) | [AWS Bedrock](https://aws.amazon.com/bedrock/)

**Setup wizard:** Run `lobster init` to launch the interactive configuration wizard. It will guide you through API key setup and save configuration to a `.env` file in your working directory.

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

🦞 You: Download GSE109564 do a QC run all preprocessing steps and perform single-cell clustering analysis

🦞 Lobster: I'll download and analyze this single-cell dataset for you...

✓ Downloaded 5,000 cells × 20,000 genes
✓ Quality control: filtered to 4,477 high-quality cells
✓ Identified 12 distinct cell clusters
✓ Generated UMAP visualization and marker gene analysis

Analysis complete! Results saved to workspace.

🦞 You: Now fetch the methods from the original publication. 
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
# Recommended: Use uv for faster installation
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
uv pip install lobster-ai

# Alternative: pip install lobster-ai
```

**Configure API Keys:**

Run the configuration wizard to set up your API keys:

```bash
# Launch interactive configuration wizard
lobster init
```

The wizard will:
- Prompt you to choose between Claude API or AWS Bedrock
- Securely collect your API keys (input is masked)
- Optionally configure NCBI API key for enhanced literature search
- Create a `.env` file in your working directory

**Additional configuration commands:**
```bash
lobster config test   # Test API connectivity
lobster config show   # Display current configuration (secrets masked)
```

**Get API Keys:**
- **Claude API**: https://console.anthropic.com/
- **AWS Bedrock**: https://aws.amazon.com/bedrock/
- **NCBI API** (optional): https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/

### Optional: Machine Learning Features (PREMIUM)

Lobster's core features work out of the box. For advanced machine learning capabilities (deep learning embeddings, scVI integration), install the ML extras:

```bash
# Install with ML features (adds PyTorch + scVI-tools)
pip install lobster-ai[ml]
```

**What's included:**
- ✅ **scVI integration**: Deep learning-based dimensionality reduction and batch correction
- ✅ **GPU acceleration**: Automatic CUDA/MPS detection for faster training
- ✅ **Advanced embeddings**: State-of-the-art single-cell embeddings

**Note**: ML extras add ~500MB of dependencies. Most users don't need this - the standard installation covers all common bioinformatics workflows.

**Advanced: Manual Configuration**

If you prefer, you can manually create a `.env` file in your working directory:

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

---

### Platform-Specific Installation

For native installation (development, advanced users):

- **macOS**: [Native Installation Guide](https://github.com/the-omics-os/lobster-local/wiki/02-installation#macos)
- **Linux**: [Ubuntu/Debian Guide](https://github.com/the-omics-os/lobster-local/wiki/02-installation#linux-ubuntudebian)
- **Windows**: [WSL Guide (Recommended)](https://github.com/the-omics-os/lobster-local/wiki/02-installation#windows)

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

---

### Uninstalling Lobster AI

#### Remove Package

**If installed globally with uv tool:**
```bash
uv tool uninstall lobster-ai
```

**If installed locally in virtual environment:**
```bash
# Activate the virtual environment first
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Uninstall
pip uninstall lobster-ai

# Remove virtual environment (optional)
deactivate
rm -rf .venv
```

**If installed with make (developers):**
```bash
cd /path/to/lobster
make uninstall-global  # Remove global symlink
make uninstall         # Remove virtual environment
```

#### Remove User Data (Optional)

⚠️ **Warning**: This deletes all your analysis data, notebooks, and workspaces!

```bash
# Remove all user data
rm -rf ~/.lobster
rm -rf ~/.lobster_workspace

# Remove project configuration
rm .env  # In your project directory
```

#### Verify Complete Removal

```bash
# Check command removed
which lobster  # Should output nothing

# Check tool not listed (if using uv tool)
uv tool list | grep lobster  # Should output nothing
```

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

**Works for both global and local installations:**

```bash
# Interactive configuration wizard
lobster init

# Test configuration
lobster config test

# View current configuration
lobster config show
```

**Manual configuration** (advanced users - edit `.env` file):

```bash
# Option A: Claude API
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Option B: AWS Bedrock
AWS_BEDROCK_ACCESS_KEY=your-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret-key

# Optional: Enhanced literature search
NCBI_API_KEY=your-ncbi-api-key
NCBI_EMAIL=your.email@example.com

# Optional: Performance tuning
LOBSTER_PROFILE=production
LOBSTER_MAX_FILE_SIZE_MB=500

# Optional: Logging level (default: WARNING for clean interface)
# Set to DEBUG for verbose output, INFO for standard verbosity
LOBSTER_LOG_LEVEL=WARNING
```

**CI/CD and automation:**
```bash
# Non-interactive mode for scripts and CI/CD
lobster init --non-interactive --anthropic-key=sk-ant-xxx
lobster init --non-interactive --bedrock-access-key=xxx --bedrock-secret-key=yyy
```

**Complete configuration guide:** [wiki/03-configuration.md](https://github.com/the-omics-os/lobster-local/wiki/03-configuration)

## 🏠 Local LLM Support (New!)

Run Lobster AI **completely locally** with Ollama - no cloud dependencies, no API costs, complete privacy.

### Quick Start with Ollama

```bash
# 1. Install Ollama (one-time setup)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model (one-time, ~4GB download)
ollama pull llama3:8b-instruct

# 3. Set environment variable
export LOBSTER_LLM_PROVIDER=ollama

# 4. Run Lobster - now 100% local!
lobster chat
```

### Why Use Local LLMs?

| Advantage | Description |
|-----------|-------------|
| ✅ **Zero API Costs** | No per-token charges |
| ✅ **Complete Privacy** | Data never leaves your machine |
| ✅ **No Rate Limits** | Use as much as you want |
| ✅ **Offline Capable** | Works without internet |

### Model Recommendations

| Model | RAM Required | Best For |
|-------|--------------|----------|
| `llama3:8b-instruct` | 8-16GB | Testing, light analysis |
| `mixtral:8x7b-instruct` | 24-32GB | Production workflows |
| `llama3:70b-instruct` | 48GB VRAM | Maximum quality (requires GPU) |

### Configuration Options

```bash
# Use Ollama (explicit)
export LOBSTER_LLM_PROVIDER=ollama

# Optional: Specify model
export OLLAMA_DEFAULT_MODEL=mixtral:8x7b-instruct

# Optional: Custom Ollama server
export OLLAMA_BASE_URL=http://localhost:11434
```

### Switching Between Cloud and Local

```bash
# Use local LLMs (Ollama)
export LOBSTER_LLM_PROVIDER=ollama
lobster chat

# Use cloud LLMs (Bedrock/Anthropic)
unset LOBSTER_LLM_PROVIDER  # Auto-detects based on API keys
lobster chat
```

### Running Multiple Sessions with Different Providers

**Use Case:** Test the same analysis with different LLM providers simultaneously, or use local for development and cloud for production.

**Method 1: Different Terminals (Easiest)**
```bash
# Terminal 1: Local development with Ollama
export LOBSTER_LLM_PROVIDER=ollama
cd ~/project1
lobster chat

# Terminal 2: Production with Claude (simultaneously)
export LOBSTER_LLM_PROVIDER=anthropic
cd ~/project2
lobster chat

# Terminal 3: Enterprise with Bedrock
export LOBSTER_LLM_PROVIDER=bedrock
cd ~/project3
lobster chat
```

**Method 2: Per-Command Override (Coming Soon)**
```bash
# Future: CLI flag support
lobster chat --provider ollama    # Use Ollama for this session
lobster query --provider anthropic "Analyze data"  # Use Claude for this query
```

**Method 3: Workspace-Specific Configuration (Coming Soon)**
```bash
# Each workspace can have its own provider
cd project1/.lobster_workspace/
# config.json specifies "ollama"

cd project2/.lobster_workspace/
# config.json specifies "anthropic"
```

**Current Best Practice:**
- Use separate terminal windows/tabs with different `LOBSTER_LLM_PROVIDER` env vars
- Each terminal maintains its own provider configuration
- Environment variables are terminal-specific (don't interfere with each other)

**Provider Selection Priority:**
1. Explicit `LOBSTER_LLM_PROVIDER` environment variable
2. Auto-detection: Ollama (if running) → Anthropic API → Bedrock
3. Default: Fails with helpful error message

**Hardware Requirements:**
- Laptop (16GB RAM): Use `llama3:8b-instruct`
- Workstation (32-64GB RAM): Use `mixtral:8x7b-instruct`
- Server with GPU (48GB+ VRAM): Use `llama3:70b-instruct`

**Trade-offs:** Local models (<70B parameters) offer privacy and zero costs but may require more prompt engineering compared to Claude. For critical production workflows, cloud models may still be preferred. For exploratory work or privacy-sensitive data, local models are excellent.

**Resources:** [Ollama docs](https://github.com/ollama/ollama) | [Model library](https://ollama.com/library) | [Troubleshooting](#troubleshooting-local-llms)

## ⭐ Premium Features

Unlock advanced capabilities with a Lobster Cloud subscription.

**Premium includes:**
- Proteomics analysis (DDA/DIA workflows, missing value handling)
- Metadata assistant for cross-dataset harmonization
- Priority support and cloud compute options
- Custom agent packages for enterprise customers

**Activate your subscription:**

```bash
# Activate with your cloud key
lobster activate <your-cloud-key>

# Check your current tier and features
lobster status
```

When you activate, any custom packages included in your subscription are automatically installed.

**Get a cloud key:** [Contact sales](mailto:info@omics-os.com) | [Pricing info](https://omics-os.com)

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

## 🛠️ For Developers

Lobster follows an **open-core model** with a single source of truth architecture:

```
lobster/config/subscription_tiers.py  (defines FREE vs PREMIUM)
                ↓
scripts/generate_allowlist.py         (generates file list)
                ↓
scripts/public_allowlist.txt          (DO NOT EDIT - auto-generated)
                ↓
lobster-local                         (public PyPI package)
```

**Key files for contributors:**
- `subscription_tiers.py` - Defines which agents/features are FREE vs PREMIUM
- `generate_allowlist.py --write` - Regenerates the sync allowlist
- `CLAUDE.md` - Complete developer guide with architecture details

**CI enforces** that `public_allowlist.txt` stays in sync with `subscription_tiers.py`.

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
