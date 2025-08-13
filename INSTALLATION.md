# 🦞 Lobster Installation Guide

## Quick Start (Recommended)

The fastest way to get Lobster AI running:

```bash
git clone https://github.com/homara-ai/lobster.git
cd lobster
make install
```

### When to Use Makefile vs pip

- **Use `make install`**: For most users, this is the recommended approach. It sets up everything automatically (virtual environment, dependencies, configuration files).

- **Use `make dev-install`**: For developers who want to contribute to the project. This installs extra tools for development (testing, linting, documentation).

- **Use `pip`/`pip3` directly**: Only if you're an advanced user with specific needs or if the Makefile doesn't work for your environment.

### Package Manager Notes

#### UV (Preferred)

If you have the [uv](https://github.com/astral-sh/uv) package manager installed, the installation will automatically use it for faster installs:

```bash
# Install UV if you don't have it yet (recommended)
curl -sSf https://install.astral.sh | sh

# Then use uv directly for installs
uv pip install -e ".[dev]"  # Note the quotes for zsh
```

#### macOS-specific Notes

On macOS, Python is typically installed with the `pip3` command rather than `pip`:

```bash
# If using pip directly on macOS
pip3 install -e ".[dev]"  # Note the quotes for zsh
```

You can verify your available package managers with:
```bash
which uv      # Check if UV is installed (recommended)
which pip     # May not exist on macOS
which pip3    # Should point to your Python 3 pip
```

## What the Installation Does

The `make install` command handles everything automatically:

1. **✅ Python Version Check** - Ensures Python 3.12+ is installed
2. **✅ Virtual Environment** - Creates isolated `.venv` directory
3. **✅ Dependency Installation** - Installs all required packages from pyproject.toml
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

### 2. Install Dependencies

When using `make dev-install`, everything is handled automatically. But if you need to install dependencies manually:

#### macOS (using zsh and pip3):

```bash
# Option 1: Use quotes to prevent zsh globbing
pip3 install -e ".[dev]"

# Option 2: Use noglob command
noglob pip3 install -e .[dev]
```

#### Linux (using bash and pip):

```bash
pip install -e .[dev]
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
- **Python**: 3.12 or higher
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
brew install python@3.12

# Ubuntu/Debian
sudo apt update && sudo apt install python3.12

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

### ZSH Square Bracket Issues

If using zsh (default shell on macOS) and getting `zsh: no matches found: .[dev]` error:

```bash
# Solution 1: Use quotes around the argument
pip3 install -e ".[dev]"  
# or with uv:
uv pip install -e ".[dev]"

# Solution 2: Use noglob command
noglob pip3 install -e .[dev]
# or with uv:
noglob uv pip install -e .[dev]

# Solution 3: Escape the square brackets
pip3 install -e .\[dev\]

# Solution 4: Use make which already handles this (RECOMMENDED)
make dev-install
```

> **NOTE**: Using make or uv is recommended as they handle all edge cases automatically.

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
PYTHON := python3.12 make install
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

# Manage dependencies
# Edit pyproject.toml directly to add or modify dependencies
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
