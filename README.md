# 🦞 Lobster AI

[![PyPI version](https://badge.fury.io/py/lobster-ai.svg)](https://badge.fury.io/py/lobster-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Multi-Agent Bioinformatics Analysis System powered by LangGraph**

Lobster AI is a powerful command-line tool that uses specialized AI agents to analyze RNA sequencing data. It combines state-of-the-art language models with proven bioinformatics tools to provide intelligent, reproducible analyses.

## 🚀 Features

- **Multi-Agent System**: Specialized agents for different analysis tasks
- **GEO Integration**: Download and analyze datasets from Gene Expression Omnibus
- **Single-Cell Analysis**: Quality control, clustering, cell type annotation
- **Literature Mining**: PubMed integration for method parameters and validation
- **Reproducible Workflows**: Complete audit trail and export capabilities
- **Flexible Configuration**: Support for multiple LLM providers and models

## 📦 Installation

### Quick Install (Recommended)

```bash
git clone https://github.com/homara-ai/lobster-ai.git
cd lobster-ai
make install
```

This will:
- ✅ Check Python 3.9+ is installed
- ✅ Create isolated virtual environment
- ✅ Install all dependencies
- ✅ Set up environment configuration
- ✅ Provide clear next steps

### Alternative Methods

#### Using pip

```bash
pip install lobster-ai
```

#### Using Docker

```bash
docker run -it --rm \
  -v ~/.lobster:/root/.lobster \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  homaraai/lobster:latest
```

#### One-line installer

```bash
curl -sSL https://get.lobster-ai.com | bash
```

## 🔧 Configuration

### Quick Setup

```bash
lobster configure
```

This interactive command will help you set up:
- API keys (OpenAI, AWS Bedrock, NCBI)
- Model preferences
- Default settings

### Manual Setup

Create a `.env` file in your working directory:

```env
# Required
OPENAI_API_KEY=your-openai-key
AWS_BEDROCK_ACCESS_KEY=your-aws-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret

# Optional
NCBI_API_KEY=your-ncbi-key
GENIE_PROFILE=production
```

## 🎯 Quick Start

### Interactive Chat Mode

```bash
lobster chat
```

### Single Query

```bash
lobster query "Download and analyze GSE109564 from GEO"
```

### With Custom Workspace

```bash
lobster chat --workspace ./my-analysis
```

## 💬 Example Usage

```bash
🦞 You: Download GSE109564 and perform quality control

🦞 Lobster: I'll download GSE109564 and perform quality control analysis...

[Downloading GEO dataset...]
✓ Downloaded GSE109564: 5,000 cells × 20,000 genes
✓ Study: Single-cell RNA-seq of mouse neurons

[Performing quality control...]
✓ Mitochondrial gene percentage: 5-15% (healthy range)
✓ Gene counts per cell: 2,000-8,000 (good coverage)
✓ Identified 523 low-quality cells for removal

[Visualizations created:]
- QC metrics violin plot
- Gene count distribution
- Mitochondrial percentage scatter plot

The data quality looks good overall. Would you like me to proceed with clustering?
```

## 🧬 Available Analyses

- **Data Download**: GEO datasets, CSV/H5 file uploads
- **Quality Control**: Cell/gene filtering, doublet detection
- **Clustering**: Leiden/Louvain algorithms, UMAP visualization
- **Cell Annotation**: Marker-based and reference-based methods
- **Differential Expression**: Between clusters or conditions
- **Pathway Analysis**: GO/KEGG enrichment
- **Literature Integration**: Find parameters and validation from PubMed

## 📊 Export & Reproducibility

Export your entire analysis:

```bash
lobster export
```

This creates a ZIP file containing:
- Raw and processed data
- All generated plots (interactive HTML + static PNG)
- Complete methodology report
- Tool parameters and timestamps

## 🛠️ Advanced Usage

### Using Different Models

```bash
# Use Claude for complex analyses
export GENIE_PROFILE=high-performance
lobster chat

# Use lightweight models for quick tasks
export GENIE_PROFILE=cost-optimized
lobster chat
```

### Programmatic Usage

```python
from lobster import LobsterClient

client = LobsterClient()
result = client.query("Analyze my single-cell data for T cell markers")
print(result['response'])
```

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Options](docs/configuration.md)
- [API Reference](docs/api.md)
- [Example Notebooks](docs/examples/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Setup development environment
make dev-install

# Run tests
make test

# Format code
make format
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Bioinformatics tools: [Scanpy](https://scanpy.readthedocs.io/), [BioPython](https://biopython.org/)
- Created by [Homara AI](https://homara.ai)

## 📞 Support

- 📧 Email: support@homara.ai
- 💬 Discord: [Join our community](https://discord.gg/homaraai)
- 🐛 Issues: [GitHub Issues](https://github.com/homara-ai/lobster-ai/issues)

---

<p align="center">
  Made with ❤️ by <a href="https://homara.ai">Homara AI</a>
</p>
