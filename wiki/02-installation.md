# Comprehensive Installation Guide

This guide covers all installation methods for Lobster AI, from quick setup to advanced development configurations.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Verification](#verification)
- [Development Installation](#development-installation)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **Python**: 3.11+ (Python 3.12+ strongly recommended)
- **Memory**: 4GB RAM (8GB+ recommended for large datasets)
- **Storage**: 2GB free space (more for data analysis)
- **Network**: Internet connection for API access and data downloads

**Recommended Setup:**
- **Python**: 3.12+
- **Memory**: 16GB+ RAM
- **Storage**: 10GB+ free space
- **CPU**: Multi-core processor for parallel analysis

### Package Manager Recommendations

Lobster AI automatically detects and uses the best available package manager:

1. **uv** (fastest, recommended): `pip install uv`
2. **pip3** (macOS default)
3. **pip** (fallback)

### Required API Keys

Choose ONE of the following LLM providers:

1. **Claude API Key** (Recommended for most users)
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Create account and generate API key
   - Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`

2. **AWS Bedrock Access** (For AWS users)
   - AWS account with Bedrock access
   - Create IAM user with Bedrock permissions
   - Add to `.env`:
     ```
     AWS_BEDROCK_ACCESS_KEY=...
     AWS_BEDROCK_SECRET_ACCESS_KEY=...
     ```

3. **NCBI API Key** (Optional)
   - Visit [NCBI E-utilities](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)
   - Enhances literature search capabilities

## Installation Methods

### Method 1: Quick Install (Recommended)

The easiest and most reliable installation method:

```bash
# Clone repository
git clone https://github.com/the-omics-os/lobster.git
cd lobster

# One-command installation
make install
```

**What this does:**
1. Verifies Python 3.12+ installation
2. Creates virtual environment at `.venv`
3. Installs all dependencies automatically
4. Sets up configuration files
5. Provides activation instructions

### Method 2: Development Install

For contributors and developers:

```bash
git clone https://github.com/the-omics-os/lobster.git
cd lobster

# Install with development dependencies
make dev-install
```

**Additional features:**
- Testing framework (pytest, coverage)
- Code quality tools (black, isort, pylint, mypy)
- Pre-commit hooks
- Documentation tools

### Method 3: Manual Installation

For full control over the installation process:

```bash
# Clone and enter directory
git clone https://github.com/the-omics-os/lobster.git
cd lobster

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip and install build tools
pip install --upgrade pip build wheel

# Install Lobster AI
pip install -e .
```

### Method 4: Global Installation

Install the `lobster` command globally (Unix/macOS):

```bash
# First, install locally
make install

# Then install globally
make install-global
```

This creates a symlink in `/usr/local/bin/lobster` allowing you to run `lobster` from anywhere.

### Method 5: Package Manager Installation

When available on PyPI (coming soon):

```bash
# Standard installation
pip install lobster-ai

# Development installation
pip install lobster-ai[dev]

# All extras
pip install lobster-ai[all]
```

## Platform-Specific Instructions

### macOS

**Homebrew Setup (Recommended):**
```bash
# Install Python 3.12+
brew install python@3.12

# Optional: Install uv for faster package management
brew install uv

# Clone and install Lobster
git clone https://github.com/the-omics-os/lobster.git
cd lobster
make install
```

**Issues with System Python:**
If using system Python causes issues:
```bash
# Use Homebrew Python explicitly
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Linux (Ubuntu/Debian)

**Install Dependencies:**
```bash
# Update package list
sudo apt update

# Install Python 3.12+
sudo apt install python3.12 python3.12-venv python3.12-dev

# Install build essentials (for compiled dependencies)
sudo apt install build-essential

# Clone and install
git clone https://github.com/the-omics-os/lobster.git
cd lobster
make install
```

**CentOS/RHEL/Fedora:**
```bash
# Install Python 3.12+
sudo dnf install python3.12 python3.12-devel

# Install development tools
sudo dnf groupinstall "Development Tools"

# Clone and install
git clone https://github.com/the-omics-os/lobster.git
cd lobster
make install
```

### Windows

**Using PowerShell:**
```powershell
# Install Python 3.12+ from python.org
# Ensure Python is in PATH

# Clone repository
git clone https://github.com/the-omics-os/lobster.git
cd lobster

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install
pip install --upgrade pip wheel
pip install -e .
```

**Using Windows Subsystem for Linux (WSL):**
Follow the Linux installation instructions within WSL.

### Python Version Considerations

**Python 3.12+ Requirements:**
- **pyproject.toml specifies**: `>=3.11`
- **Makefile enforces**: `>=3.12`
- **Recommendation**: Use Python 3.12+ for best performance

**Installing Specific Python Version:**
```bash
# macOS with pyenv
brew install pyenv
pyenv install 3.12.0
pyenv local 3.12.0

# Ubuntu with deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12 python3.12-venv
```

## Verification

### Test Installation

After installation, verify everything works:

```bash
# Activate environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Test CLI
lobster --help

# Test imports
python -c "import lobster; print('✅ Lobster imported successfully')"

# Run verification script
python verify_installation.py
```

### Check System Status

```bash
# Start Lobster and check status
lobster chat

# In the chat interface, type:
/status
```

Expected output:
```
✅ System Status: Healthy
✅ Environment: Virtual environment active
✅ Dependencies: All packages installed
✅ Configuration: .env file present
⚠️  API Keys: Configure in .env file
```

### Verify API Connectivity

```bash
# Test API keys (after configuration)
lobster config test
```

## Development Installation

### Full Development Setup

```bash
git clone https://github.com/the-omics-os/lobster.git
cd lobster

# Development installation
make dev-install

# This installs:
# - All runtime dependencies
# - Testing framework (pytest, pytest-cov, pytest-xdist)
# - Code quality (black, isort, flake8, pylint, mypy)
# - Security tools (bandit)
# - Documentation (mkdocs)
# - Pre-commit hooks
```

### Development Commands

```bash
# Run tests
make test

# Fast parallel testing
make test-fast

# Code formatting
make format

# Linting
make lint

# Type checking
make type-check

# Clean installation
make clean-install
```

### Pre-commit Hooks

Development installation automatically sets up pre-commit hooks:

```bash
# Manual setup if needed
make setup-pre-commit

# Run on all files
pre-commit run --all-files
```

## Docker Deployment

### Build Docker Image

```bash
# Build image
make docker-build

# Or manually
docker build -t lobster-ai:latest .
```

### Run with Docker

```bash
# Create .env file first (see Configuration Guide)

# Run interactive container
make docker-run

# Or manually with custom settings (Claude API example)
docker run -it --rm \
  -v ~/.lobster:/root/.lobster \
  -v $(pwd)/data:/app/data \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  lobster-ai:latest

# Or with AWS Bedrock
docker run -it --rm \
  -v ~/.lobster:/root/.lobster \
  -v $(pwd)/data:/app/data \
  -e AWS_BEDROCK_ACCESS_KEY=$AWS_BEDROCK_ACCESS_KEY \
  -e AWS_BEDROCK_SECRET_ACCESS_KEY=$AWS_BEDROCK_SECRET_ACCESS_KEY \
  lobster-ai:latest
```

### Docker Compose

```bash
# Start with docker-compose
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

**docker-compose.yml configuration:**
```yaml
version: '3.8'
services:
  lobster:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ~/.lobster:/root/.lobster
```

## Troubleshooting

### Common Installation Issues

#### Python Version Problems

**Error**: `Python 3.12+ is required`

**Solutions:**
```bash
# Check Python version
python --version
python3 --version

# Install Python 3.12+
# macOS: brew install python@3.12
# Ubuntu: sudo apt install python3.12
# Windows: Download from python.org

# Use specific Python version
python3.12 -m venv .venv
```

#### Virtual Environment Issues

**Error**: `Failed to create virtual environment`

**Solutions:**
```bash
# Install venv module (Ubuntu/Debian)
sudo apt install python3.12-venv

# Clear existing environment
rm -rf .venv

# Create manually
python3 -m venv .venv --clear

# Alternative method
python3 -m venv .venv --without-pip
source .venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py | python
```

#### Dependency Installation Failures

**Error**: `Failed building wheel for [package]`

**Solutions:**
```bash
# Install development headers (Linux)
sudo apt install python3.12-dev build-essential

# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Clear pip cache
pip cache purge

# Install with no cache
pip install --no-cache-dir -e .

# Use uv for faster, more reliable installs
pip install uv
uv pip install -e .
```

#### Permission Errors

**Error**: `Permission denied`

**Solutions:**
```bash
# Don't use sudo with pip in virtual environment
# Instead, ensure virtual environment ownership
chown -R $USER:$USER .venv

# For global installation (Unix only)
sudo make install-global
```

#### Memory Issues During Installation

**Error**: `Killed` or memory-related errors

**Solutions:**
```bash
# Increase swap space (Linux)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Install with limited parallelism
pip install -e . --no-build-isolation

# Use lighter installation
export LOBSTER_PROFILE=cost-optimized
make install
```

### Runtime Issues

#### API Key Problems

**Error**: API key not found or invalid

**Solutions:**
```bash
# Check environment variables
echo $ANTHROPIC_API_KEY    # For Claude API
echo $AWS_BEDROCK_ACCESS_KEY  # For AWS Bedrock
source .env  # Load from file

# Test API connectivity
lobster config test

# Regenerate API keys if needed
```

#### Import Errors

**Error**: `ModuleNotFoundError: No module named 'lobster'`

**Solutions:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall in development mode
pip install -e .

# Check PYTHONPATH
python -c "import sys; print(sys.path)"
```

#### Memory Issues During Analysis

**Solutions:**
```bash
# Use lighter model profile
export LOBSTER_PROFILE=cost-optimized

# Reduce file size limits
export LOBSTER_MAX_FILE_SIZE_MB=100

# Monitor memory usage
htop  # Linux/macOS
# Task Manager on Windows
```

### Getting Additional Help

#### Check System Health
```bash
lobster chat
/dashboard  # Comprehensive system overview
/status     # Quick status check
```

#### Enable Debug Mode
```bash
# Verbose logging
lobster chat --debug --verbose

# Show reasoning
lobster chat --reasoning
```

#### Log Files
```bash
# Check logs in workspace
ls .lobster_workspace/logs/

# Enable detailed logging
export LOBSTER_LOG_LEVEL=DEBUG
```

#### Community Support

- **GitHub Issues**: [Report bugs](https://github.com/the-omics-os/lobster/issues)
- **Discord**: [Join community](https://discord.gg/HDTRbWJ8omicsos)
- **Email**: [Direct support](mailto:info@omics-os.com)
- **Documentation**: [Full docs](../README.md)

### Clean Reinstallation

If all else fails, perform a clean reinstallation:

```bash
# Remove everything
make uninstall      # Remove virtual environment
make clean          # Remove build artifacts
rm -rf .lobster_workspace  # Remove workspace (optional)

# Fresh installation
make clean-install

# Or manually
rm -rf .venv
git clean -fdx  # Warning: removes all untracked files
make install
```

---

**Next Steps**: Once installation is complete, see the [Configuration Guide](03-configuration.md) to set up API keys and customize your Lobster AI environment.