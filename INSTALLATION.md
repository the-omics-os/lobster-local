# 🦞 Lobster AI Installation Guide

## Quick Start (Recommended)

The fastest way to get Lobster AI running:

```bash
git clone https://github.com/homara-ai/lobster.git
cd lobster-ai
make install
```

## What the Installation Does

The `make install` command handles everything automatically:

1. **✅ Python Version Check** - Ensures Python 3.9+ is installed
2. **✅ Virtual Environment** - Creates isolated `.venv` directory
3. **✅ Dependency Installation** - Installs all required packages
4. **✅ Environment Setup** - Creates `.env` file for API keys
5. **✅ Post-Install Guidance** - Shows you exactly what to do next

## After Installation

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate
```

Or use the helper command:
```bash
make activate
```

### 2. Configure API Keys

Edit the `.env` file with your API keys:

```bash
nano .env
```

Required keys:
- `OPENAI_API_KEY` - From OpenAI dashboard
- `AWS_BEDROCK_ACCESS_KEY` - AWS access key  
- `AWS_BEDROCK_SECRET_ACCESS_KEY` - AWS secret key

Optional:
- `NCBI_API_KEY` - For enhanced GEO downloads
- `GENIE_PROFILE` - Model performance profile

### 3. Test Installation

```bash
lobster --help
lobster chat
```

## Development Installation

For contributors and developers:

```bash
make dev-install
```

This includes:
- All runtime dependencies
- Development tools (pytest, black, pylint, etc.)
- Pre-commit hooks
- Documentation tools

## Alternative Installation Methods

### System-wide pip install

```bash
pip install lobster-ai
```

⚠️ **Note**: This doesn't include environment setup or isolation.

### Docker

```bash
docker run -it --rm \
  -v ~/.lobster:/root/.lobster \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  homaraai/lobster:latest
```

### One-line installer (when available)

```bash
curl -sSL https://get.lobster-ai.com | bash
```

## Requirements

### System Requirements

- **Operating System**: macOS, Linux, or Windows with WSL
- **Python**: 3.9 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for dependencies and data

### Python Dependencies

Core dependencies (automatically installed):
- LangGraph & LangChain for AI orchestration
- Scanpy & BioPython for bioinformatics
- Pandas & NumPy for data processing
- Plotly for interactive visualizations

### API Requirements

At least one of:
- **OpenAI API Key** - For GPT models
- **AWS Bedrock Access** - For Claude and other models

## Troubleshooting

### Python Version Issues

```bash
# Check your Python version
python3 --version

# If you need to install Python 3.9+:
# macOS with Homebrew
brew install python@3.11

# Ubuntu/Debian
sudo apt update && sudo apt install python3.11

# Windows - download from python.org
```

### Virtual Environment Issues

```bash
# Clean install
make clean-install

# Manual cleanup
rm -rf .venv
make install
```

### Permission Issues

```bash
# If pip fails with permissions
python3 -m pip install --user --upgrade pip

# Then retry
make install
```

### Import Errors

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Check installation
lobster --version
```

### API Key Issues

```bash
# Check environment variables
make check-env

# Verify .env file exists and has correct format
cat .env
```

## Environment Management

### Check Status

```bash
make check-env
```

### Update Dependencies

```bash
make update-deps
```

### Clean Installation

```bash
make clean-install
```

### Uninstall

```bash
make uninstall  # Removes virtual environment
make clean-all  # Removes all generated files
```

## Advanced Configuration

### Custom Virtual Environment Location

```bash
# Edit Makefile to change VENV_PATH
VENV_PATH := /path/to/custom/venv
```

### Using Different Python Version

```bash
# Use specific Python version
PYTHON := python3.11 make install
```

### Development Workflow

```bash
# Setup development environment
make dev-install

# Run tests
make test

# Format code  
make format

# Type checking
make type-check

# Build documentation
make docs
```

## Getting Help

If you encounter issues:

1. **Check this guide** - Most common issues are covered above
2. **GitHub Issues** - [Report bugs or request features](https://github.com/homara-ai/lobster-ai/issues)
3. **Discord** - [Join our community](https://discord.gg/homaraai)
4. **Email** - support@homara.ai

## Next Steps

After successful installation:

- 📖 Read the [User Guide](README.md) for usage examples
- 🧬 Try the [Example Analyses](docs/examples/)  
- 🤝 Check out [Contributing Guide](CONTRIBUTING.md) to contribute
- 🐛 Report any issues on [GitHub](https://github.com/homara-ai/lobster-ai/issues)

---

**Happy analyzing with Lobster AI! 🦞**
