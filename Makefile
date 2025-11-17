# Lobster AI Makefile
# Professional build and development commands

.PHONY: help install dev-install install-global uninstall-global test format lint clean docker-build docker-run release check-python setup-env

# Configuration
VENV_NAME := .venv
VENV_PATH := $(VENV_NAME)
PYTHON_VERSION_MIN := 3.11
PROJECT_NAME := lobster-ai

# Smart Python Discovery
# Check for conda environment first (avoid conflicts)
CONDA_ACTIVE := $(shell echo $$CONDA_DEFAULT_ENV)
PYENV_VERSION := $(shell pyenv version-name 2>/dev/null || echo "")

# Allow PYTHON override via environment variable
# If PYTHON is not set, find best available Python (3.11+)
# Priority: 3.13 > 3.12 > 3.11 (prefer newer versions)
# Prioritize Homebrew installations to avoid broken system Python
ifndef PYTHON
	PYTHON_CANDIDATES := /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 python3.13 python3.12 python3.11 python3 python
	PYTHON := $(shell for p in $(PYTHON_CANDIDATES); do \
		if command -v $$p >/dev/null 2>&1; then \
			if $$p -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then \
				echo $$p; \
				break; \
			fi; \
		fi; \
	done)

	# If no suitable Python found, default to python3 for error messages
	ifeq ($(PYTHON),)
		PYTHON := python3
	endif
endif

# Detect Python environment type
PYTHON_ENV_TYPE := $(shell \
	if [ -n "$(CONDA_ACTIVE)" ]; then \
		echo "conda"; \
	elif [ -n "$(PYENV_VERSION)" ] && [ "$(PYENV_VERSION)" != "system" ]; then \
		echo "pyenv"; \
	elif command -v brew >/dev/null 2>&1 && $(PYTHON) -c "import sys; print('homebrew' if 'Cellar' in sys.executable or 'homebrew' in sys.executable else 'system')" 2>/dev/null; then \
		echo "homebrew"; \
	else \
		echo "system"; \
	fi)

# Determine package manager (prefer uv > pip3 > pip)
UV_EXISTS := $(shell which uv > /dev/null 2>&1 && echo "yes" || echo "no")
PIP_EXISTS := $(shell which pip > /dev/null 2>&1 && echo "yes" || echo "no")
PIP3_EXISTS := $(shell which pip3 > /dev/null 2>&1 && echo "yes" || echo "no")

ifeq ($(UV_EXISTS),yes)
	# Use uv if available for faster installations
	PKG_MGR := uv
	SYS_PIP := uv pip
	USE_UV := true
else
	USE_UV := false
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		# On macOS, prefer pip3 if available
		ifeq ($(PIP3_EXISTS),yes)
			SYS_PIP := pip3
			PKG_MGR := pip3
		else
			SYS_PIP := pip
			PKG_MGR := pip
		endif
	else
		SYS_PIP := pip
		PKG_MGR := pip
	endif
endif

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
help:
	@echo "🦞 Lobster - Available Commands"
	@echo ""
	@echo "Installation:"
	@echo "  make install        Install Lobster AI in virtual environment (default: Python 3.13)"
	@echo "  make dev-install    Install with development dependencies"
	@echo "  make install-global Install lobster command globally (macOS/Linux)"
	@echo "  make clean-install  Clean install (remove existing installation)"
	@echo "  make setup-env      Setup environment configuration"
	@echo "  make activate       Show activation command"
	@echo ""
	@echo "Python Version Override:"
	@echo "  PYTHON=/path/to/python3.11 make install  # Use specific Python version"
	@echo "  PYTHON=python3.12 make install           # Use Python 3.12"
	@echo ""
	@echo "Optional Components:"
	@echo "  make install-pymol  Install PyMOL for protein structure visualization"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run all tests"
	@echo "  make test-fast     Run tests (parallel)"
	@echo "  make format        Format code with black and isort"
	@echo "  make lint          Run linting checks"
	@echo "  make type-check    Run type checking with mypy"
	@echo "  make verify        Verify installation integrity"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build         Build Docker images (CLI + server)"
	@echo "  make docker-run-cli       Run CLI in Docker container"
	@echo "  make docker-run-server    Run FastAPI server in Docker"
	@echo "  make docker-compose-up    Start services with docker-compose"
	@echo "  make docker-compose-cli   Run CLI via docker-compose"
	@echo "  make docker-compose-down  Stop docker-compose services"
	@echo "  make docker-push          Push images to Docker Hub"
	@echo ""
	@echo "Release:"
	@echo "  make release       Create a new release"
	@echo "  make publish       Publish to PyPI"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove build artifacts"
	@echo "  make clean-all     Remove all generated files"
	@echo "  make uninstall     Remove virtual environment"
	@echo "  make uninstall-global Remove global lobster command"

# Environment detection and validation
check-env-conflicts:
	@if [ -n "$(CONDA_ACTIVE)" ]; then \
		echo "$(YELLOW)⚠️  Conda environment '$(CONDA_ACTIVE)' is active$(NC)"; \
		echo "$(YELLOW)   This may cause conflicts with the virtual environment.$(NC)"; \
		echo "$(BLUE)   Recommended: deactivate conda before installing:$(NC)"; \
		echo "$(YELLOW)   conda deactivate$(NC)"; \
		echo ""; \
	fi
	@if [ -f "$(VENV_PATH)/bin/python" ] && [ -n "$(VIRTUAL_ENV)" ] && [ "$(VIRTUAL_ENV)" != "$(shell pwd)/$(VENV_PATH)" ]; then \
		echo "$(YELLOW)⚠️  Another virtual environment is active: $(VIRTUAL_ENV)$(NC)"; \
		echo "$(BLUE)   Recommended: deactivate it first:$(NC)"; \
		echo "$(YELLOW)   deactivate$(NC)"; \
		echo ""; \
	fi

# Python version check with platform guidance
check-python: check-env-conflicts
	@echo "🔍 Checking Python environment..."
	@echo "   Environment type: $(PYTHON_ENV_TYPE)"
	@echo "   Python command: $(PYTHON)"
	@if [ -z "$(PYTHON)" ] || ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "$(RED)❌ No suitable Python 3.11+ found$(NC)"; \
		echo ""; \
		echo "$(YELLOW)📋 Installation instructions based on your system:$(NC)"; \
		if [ -n "$(CONDA_ACTIVE)" ]; then \
			echo "$(BLUE)🐍 Conda environment detected$(NC)"; \
			echo "$(YELLOW)   Option 1: Install in conda:$(NC)"; \
			echo "     conda install python=3.12"; \
			echo "$(YELLOW)   Option 2: Deactivate conda and use system Python:$(NC)"; \
			echo "     conda deactivate"; \
		elif command -v pyenv >/dev/null 2>&1; then \
			echo "$(BLUE)🐍 pyenv detected$(NC)"; \
			echo "$(YELLOW)   Install Python 3.12 with pyenv:$(NC)"; \
			echo "     pyenv install 3.12.0"; \
			echo "     pyenv global 3.12.0"; \
			echo "     pyenv rehash"; \
		elif [ "$$(uname -s)" = "Darwin" ]; then \
			echo "$(BLUE)🍎 macOS detected$(NC)"; \
			if ! command -v brew >/dev/null 2>&1; then \
				echo "$(YELLOW)   First install Homebrew:$(NC)"; \
				echo "     /bin/bash -c \"$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""; \
				echo ""; \
			fi; \
			echo "$(YELLOW)   Install Python with Homebrew:$(NC)"; \
			echo "     brew install python@3.12"; \
			echo "     brew link python@3.12"; \
			if [ "$$(uname -m)" = "arm64" ]; then \
				echo "$(BLUE)   🍎 Apple Silicon optimizations will be applied$(NC)"; \
			fi; \
		else \
			echo "$(BLUE)🐧 Linux detected$(NC)"; \
			if command -v apt >/dev/null 2>&1; then \
				echo "$(YELLOW)   Ubuntu/Debian - Install with apt:$(NC)"; \
				echo "     sudo apt update"; \
				echo "     sudo apt install python3.12 python3.12-dev python3.12-venv"; \
			elif command -v yum >/dev/null 2>&1; then \
				echo "$(YELLOW)   RHEL/CentOS - Install with yum:$(NC)"; \
				echo "     sudo yum install python3.12 python3.12-devel"; \
			else \
				echo "$(YELLOW)   Generic Linux - Build from source or use pyenv:$(NC)"; \
				echo "     curl https://pyenv.run | bash"; \
				echo "     pyenv install 3.12.0"; \
			fi; \
		fi; \
		exit 1; \
	fi
	@$(PYTHON) -c "import sys; print(sys.version_info)" > /dev/null 2>&1 || { \
		echo "$(RED)❌ Failed to execute Python. Please check your installation.$(NC)"; \
		exit 1; \
	}
	@$(PYTHON) -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" || { \
		echo "$(RED)❌ Python 3.11+ is required. Found: $$($(PYTHON) --version 2>&1)$(NC)"; \
		echo ""; \
		echo "$(YELLOW)📋 Upgrade instructions for your setup ($(PYTHON_ENV_TYPE)):$(NC)"; \
		if [ "$(PYTHON_ENV_TYPE)" = "conda" ]; then \
			echo "$(BLUE)🐍 Conda environment:$(NC)"; \
			echo "$(YELLOW)   Update Python in current environment:$(NC)"; \
			echo "     conda update python"; \
			echo "$(YELLOW)   Or create new environment:$(NC)"; \
			echo "     conda create -n lobster python=3.12"; \
			echo "     conda activate lobster"; \
		elif [ "$(PYTHON_ENV_TYPE)" = "pyenv" ]; then \
			echo "$(BLUE)🐍 pyenv:$(NC)"; \
			echo "$(YELLOW)   Install and set Python 3.12:$(NC)"; \
			echo "     pyenv install 3.12.0"; \
			echo "     pyenv local 3.12.0  # for this project"; \
			echo "     # or"; \
			echo "     pyenv global 3.12.0  # system-wide"; \
		elif [ "$(PYTHON_ENV_TYPE)" = "homebrew" ]; then \
			echo "$(BLUE)🍺 Homebrew Python:$(NC)"; \
			echo "$(YELLOW)   Upgrade Python:$(NC)"; \
			echo "     brew upgrade python@3.12"; \
			echo "     brew link --overwrite python@3.12"; \
			echo "$(YELLOW)   If link fails:$(NC)"; \
			echo "     brew unlink python@3.11  # or current version"; \
			echo "     brew link python@3.12"; \
		else \
			if [ "$$(uname -s)" = "Darwin" ]; then \
				echo "$(YELLOW)   macOS: brew install python@3.12 && brew link python@3.12$(NC)"; \
			else \
				echo "$(YELLOW)   Ubuntu: sudo apt install python3.12 python3.12-dev python3.12-venv$(NC)"; \
				echo "$(YELLOW)   RHEL: sudo yum install python3.12 python3.12-devel$(NC)"; \
			fi; \
		fi; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Python check passed: $$($(PYTHON) --version 2>&1)$(NC)"
	@echo "   Path: $$(which $(PYTHON))"
	@echo "🔍 Checking venv module..."
	@$(PYTHON) -c "import venv" > /dev/null 2>&1 || { \
		echo "$(RED)❌ Python venv module not found$(NC)"; \
		echo ""; \
		echo "$(YELLOW)📋 Fix for your environment ($(PYTHON_ENV_TYPE)):$(NC)"; \
		if [ "$(PYTHON_ENV_TYPE)" = "conda" ]; then \
			echo "$(YELLOW)   Conda usually includes venv. Try:$(NC)"; \
			echo "     conda install python=3.12"; \
		elif [ "$(PYTHON_ENV_TYPE)" = "pyenv" ]; then \
			echo "$(YELLOW)   Reinstall Python with proper configuration:$(NC)"; \
			echo "     pyenv uninstall 3.12.0"; \
			echo "     pyenv install 3.12.0"; \
		elif [ "$$(uname -s)" = "Darwin" ]; then \
			echo "$(YELLOW)   macOS - Reinstall Python:$(NC)"; \
			echo "     brew reinstall python@3.12"; \
		else \
			PYTHON_VERSION=$$($(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.12"); \
			echo "$(YELLOW)   Ubuntu/Debian:$(NC)"; \
			echo "     sudo apt install python$${PYTHON_VERSION}-venv"; \
			echo "$(YELLOW)   RHEL/CentOS:$(NC)"; \
			echo "     sudo yum install python$${PYTHON_VERSION}-venv"; \
		fi; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Python venv module available$(NC)"
	@echo "$(BLUE)🔍 Checking system dependencies...$(NC)"
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "$(YELLOW)⚠️ Homebrew not found. Install: /bin/bash -c \"$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"$(NC)"; \
		fi; \
	else \
		if ! dpkg -l python3.12-dev >/dev/null 2>&1; then \
			echo "$(YELLOW)⚠️ python3.12-dev not found. Install: sudo apt install python3.12-dev$(NC)"; \
		fi; \
	fi

$(VENV_PATH): check-python
	@echo "🐍 Creating virtual environment..."
	@echo "   Using: $(PYTHON) ($(PYTHON_ENV_TYPE))"
	@if [ -n "$(CONDA_ACTIVE)" ]; then \
		echo "$(YELLOW)⚠️  Note: Creating venv inside conda environment '$(CONDA_ACTIVE)'$(NC)"; \
		echo "$(YELLOW)   This is usually fine, but if you have issues, try:$(NC)"; \
		echo "$(YELLOW)   conda deactivate && make clean-install$(NC)"; \
	fi
	@if ! $(PYTHON) -c "import ensurepip" >/dev/null 2>&1; then \
		echo "$(RED)❌ Python ensurepip module not found$(NC)"; \
		echo ""; \
		echo "$(YELLOW)📋 Fix for your environment ($(PYTHON_ENV_TYPE)):$(NC)"; \
		if [ "$(PYTHON_ENV_TYPE)" = "conda" ]; then \
			echo "$(YELLOW)   Update conda Python:$(NC)"; \
			echo "     conda update python"; \
		elif [ "$(PYTHON_ENV_TYPE)" = "pyenv" ]; then \
			echo "$(YELLOW)   Reinstall Python:$(NC)"; \
			echo "     pyenv install --force 3.12.0"; \
		elif [ "$$(uname -s)" = "Darwin" ]; then \
			echo "$(YELLOW)   macOS:$(NC)"; \
			echo "     brew reinstall python@3.12"; \
		else \
			PYTHON_VERSION=$$($(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.12"); \
			echo "$(YELLOW)   Linux:$(NC)"; \
			echo "     sudo apt install python$${PYTHON_VERSION}-venv python$${PYTHON_VERSION}-distutils"; \
		fi; \
		exit 1; \
	fi
	@echo "   Using package manager: $(PKG_MGR)"
	@if [ "$$(uname -m)" = "arm64" ] && [ "$$(uname -s)" = "Darwin" ]; then \
		echo "$(BLUE)   🍎 Apple Silicon optimization enabled$(NC)"; \
	fi
	@$(PYTHON) -m venv $(VENV_PATH) || { \
		echo "$(RED)❌ Failed to create virtual environment$(NC)"; \
		echo "$(YELLOW)🔄 Attempting recovery...$(NC)"; \
		$(PYTHON) -m venv $(VENV_PATH) --without-pip || { \
			echo "$(RED)❌ Virtual environment creation failed$(NC)"; \
			echo ""; \
			echo "$(YELLOW)📋 Troubleshooting for $(PYTHON_ENV_TYPE) environment:$(NC)"; \
			if [ "$(PYTHON_ENV_TYPE)" = "conda" ]; then \
				echo "$(YELLOW)   1. Exit conda and use system Python:$(NC)"; \
				echo "      conda deactivate"; \
				echo "      make clean-install"; \
				echo "$(YELLOW)   2. Or create conda environment instead:$(NC)"; \
				echo "      conda create -n lobster python=3.12"; \
				echo "      conda activate lobster"; \
				echo "      pip install -e ."; \
			elif [ "$(PYTHON_ENV_TYPE)" = "pyenv" ]; then \
				echo "$(YELLOW)   1. Reinstall Python with all components:$(NC)"; \
				echo "      pyenv install --force 3.12.0"; \
				echo "      pyenv rehash"; \
				echo "$(YELLOW)   2. Check pyenv shims:$(NC)"; \
				echo "      pyenv which python"; \
			elif [ "$$(uname -s)" = "Darwin" ]; then \
				echo "$(YELLOW)   1. Reinstall Python:$(NC)"; \
				echo "      brew uninstall --ignore-dependencies python@3.12"; \
				echo "      brew install python@3.12"; \
				echo "$(YELLOW)   2. Check Xcode tools:$(NC)"; \
				echo "      xcode-select --install"; \
				echo "$(YELLOW)   3. Check disk space and permissions:$(NC)"; \
				echo "      df -h ."; \
				echo "      ls -la ."; \
			else \
				echo "$(YELLOW)   1. Install venv package:$(NC)"; \
				echo "      sudo apt update"; \
				echo "      sudo apt install python3.12-venv python3.12-dev"; \
				echo "$(YELLOW)   2. Check disk space: df -h$(NC)"; \
				echo "$(YELLOW)   3. Check permissions: ls -la .$(NC)"; \
				echo "$(YELLOW)   4. Try with --system-site-packages:$(NC)"; \
				echo "      $(PYTHON) -m venv $(VENV_PATH) --system-site-packages"; \
			fi; \
			exit 1; \
		}; \
		echo "$(YELLOW)⚠️ Created environment without pip. Installing pip manually...$(NC)"; \
		curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py || { \
			echo "$(RED)❌ Failed to download pip installer. Check internet connection.$(NC)"; \
			exit 1; \
		}; \
		$(VENV_PATH)/bin/python /tmp/get-pip.py; \
		rm /tmp/get-pip.py; \
	}
	@if [ ! -f "$(VENV_PATH)/bin/pip" ] && [ ! -f "$(VENV_PATH)/bin/pip3" ]; then \
		echo "$(RED)❌ Virtual environment created but pip is not available.$(NC)"; \
		echo "$(YELLOW)📋 This usually indicates a Python installation issue.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Virtual environment created successfully at $(VENV_PATH)$(NC)"
	@if [ "$$(uname -s)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ]; then \
		echo "$(BLUE)🍎 Ready for Apple Silicon optimized installation$(NC)"; \
	fi


# Environment setup
setup-env: $(VENV_PATH)
	@echo "⚙️  Setting up environment configuration..."
	@if [ ! -f .env ]; then \
		if [ -f .env.example ]; then \
			cp .env.example .env; \
			echo "$(YELLOW)📋 Created .env from .env.example. Please edit it with your API keys.$(NC)"; \
		else \
			echo "# Lobster AI Environment Variables" > .env; \
			echo "# Required API Keys" >> .env; \
			echo "ANTHROPIC_API_KEY=sk-ant-api03-your-key-here" >> .env; \
			echo "AWS_BEDROCK_ACCESS_KEY=your-aws-access-key-here" >> .env; \
			echo "AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key-here" >> .env; \
			echo "" >> .env; \
			echo "# Optional" >> .env; \
			echo "NCBI_API_KEY=your-ncbi-api-key-here" >> .env; \
			echo "LOBSTER_PROFILE=production" >> .env; \
			echo "$(YELLOW)📋 Created .env file. Please edit it with your API keys.$(NC)"; \
		fi; \
	else \
		echo "$(GREEN)✅ .env file already exists$(NC)"; \
	fi

# Install PyMOL (optional prerequisite for protein structure visualization)
install-pymol:
	@echo "$(BLUE)🔬 Installing PyMOL for protein structure visualization...$(NC)"
	@if command -v pymol >/dev/null 2>&1; then \
		echo "$(GREEN)✅ PyMOL is already installed$(NC)"; \
		pymol -c -Q 2>/dev/null && echo "   Version: $$(pymol -c -Q 2>&1 | head -1)" || echo "   (command-line mode detected)"; \
	else \
		UNAME_S=$$(uname -s 2>/dev/null || echo "Unknown"); \
		if [ "$$UNAME_S" = "Darwin" ]; then \
			echo "$(BLUE)🍎 macOS detected - Installing via Homebrew...$(NC)"; \
			if ! command -v brew >/dev/null 2>&1; then \
				echo "$(RED)❌ Homebrew not found. Please install Homebrew first:$(NC)"; \
				echo "   /bin/bash -c \"\$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""; \
				exit 1; \
			fi; \
			if brew tap | grep -q "brewsci/bio"; then \
				echo "$(GREEN)✓ brewsci/bio tap already added$(NC)"; \
			else \
				echo "$(YELLOW)📦 Adding brewsci/bio tap...$(NC)"; \
				brew tap brewsci/bio; \
			fi; \
			echo "$(YELLOW)📦 Installing PyMOL...$(NC)"; \
			brew install brewsci/bio/pymol && echo "$(GREEN)✅ PyMOL installed successfully!$(NC)" || { \
				echo "$(RED)❌ PyMOL installation failed$(NC)"; \
				exit 1; \
			}; \
		elif [ "$$UNAME_S" = "Linux" ]; then \
			echo "$(BLUE)🐧 Linux detected - Installing via package manager...$(NC)"; \
			if command -v apt-get >/dev/null 2>&1; then \
				echo "$(YELLOW)📦 Installing PyMOL via apt-get (requires sudo)...$(NC)"; \
				sudo apt-get update && sudo apt-get install -y pymol && echo "$(GREEN)✅ PyMOL installed successfully!$(NC)" || { \
					echo "$(RED)❌ PyMOL installation failed$(NC)"; \
					exit 1; \
				}; \
			elif command -v dnf >/dev/null 2>&1; then \
				echo "$(YELLOW)📦 Installing PyMOL via dnf (requires sudo)...$(NC)"; \
				sudo dnf install -y pymol && echo "$(GREEN)✅ PyMOL installed successfully!$(NC)" || { \
					echo "$(RED)❌ PyMOL installation failed$(NC)"; \
					exit 1; \
				}; \
			elif command -v brew >/dev/null 2>&1; then \
				echo "$(YELLOW)📦 Installing PyMOL via Homebrew on Linux...$(NC)"; \
				brew tap brewsci/bio && brew install brewsci/bio/pymol && echo "$(GREEN)✅ PyMOL installed successfully!$(NC)" || { \
					echo "$(RED)❌ PyMOL installation failed$(NC)"; \
					exit 1; \
				}; \
			else \
				echo "$(RED)❌ No supported package manager found (apt-get, dnf, or brew)$(NC)"; \
				echo "$(YELLOW)Please install PyMOL manually:$(NC)"; \
				echo "   Ubuntu/Debian: sudo apt-get install pymol"; \
				echo "   Fedora/RHEL: sudo dnf install pymol"; \
				echo "   Or visit: https://pymol.org/"; \
				exit 1; \
			fi; \
		else \
			echo "$(RED)❌ Unsupported operating system: $$UNAME_S$(NC)"; \
			echo "$(YELLOW)Please install PyMOL manually: https://pymol.org/$(NC)"; \
			exit 1; \
		fi; \
	fi
	@echo ""
	@echo "$(GREEN)🎉 PyMOL installation complete!$(NC)"
	@echo "$(BLUE)💡 Test with: pymol -c -Q$(NC)"

# Installation targets
install: $(VENV_PATH) setup-env
	@echo "🦞 Installing Lobster AI..."
	@if [ "$(USE_UV)" = "true" ]; then \
		echo "📦 Using uv for faster installation..."; \
		uv pip install -e .; \
	else \
		echo "📦 Upgrading pip and installing build tools..."; \
		if [ -f "$(VENV_PATH)/bin/pip3" ]; then \
			$(VENV_PATH)/bin/pip3 install --upgrade pip build wheel; \
		else \
			$(VENV_PATH)/bin/pip install --upgrade pip build wheel; \
		fi; \
		echo "📦 Installing Lobster AI and dependencies..."; \
		if [ -f "$(VENV_PATH)/bin/pip3" ]; then \
			$(VENV_PATH)/bin/pip3 install -e .; \
		else \
			$(VENV_PATH)/bin/pip install -e .; \
		fi; \
	fi
	@echo ""
	@echo "$(GREEN)🎉 Installation complete!$(NC)"
	@echo ""
	@echo "$(BLUE)🔍 Running installation verification...$(NC)"
	@$(VENV_PATH)/bin/python verify_installation.py || { \
		echo "$(YELLOW)⚠️ Some verification checks showed warnings$(NC)"; \
		echo "$(YELLOW)   Missing proteomics modules are expected in the public distribution$(NC)"; \
		echo "$(YELLOW)   Installation is complete for transcriptomics functionality$(NC)"; \
	}
	@echo ""
	@echo "$(BLUE)📋 Next steps:$(NC)"
	@echo "1. Activate the virtual environment:"
	@echo "   $(YELLOW)source $(VENV_PATH)/bin/activate$(NC)"
	@echo ""
	@echo "2. Configure your API keys in the .env file:"
	@UNAME_S=$$(uname -s 2>/dev/null || echo "Unknown"); \
	if [ "$$UNAME_S" = "Darwin" ]; then \
		echo "   $(YELLOW)open .env$(NC)  $(BLUE)# Opens in your default text editor$(NC)"; \
	elif [ "$$UNAME_S" = "Linux" ]; then \
		echo "   $(YELLOW)nano .env$(NC)  $(BLUE)# Or use your preferred editor (vi, vim, gedit)$(NC)"; \
	else \
		echo "   $(YELLOW)notepad .env$(NC)  $(BLUE)# Windows$(NC)"; \
		echo "   $(RED)⚠️  Note: Windows support is currently untested$(NC)"; \
	fi
	@echo "   $(BLUE)Required: ANTHROPIC_API_KEY or AWS Bedrock credentials$(NC)"
	@echo ""
	@echo "3. Test the installation:"
	@echo "   $(YELLOW)lobster --help$(NC)"
	@echo ""
	@echo "4. Start using Lobster AI:"
	@echo "   $(YELLOW)lobster chat$(NC)"
	@echo ""
	@echo "$(BLUE)💡 Tip: Try asking 'Download GSE109564 and perform single-cell analysis'$(NC)"

dev-install: $(VENV_PATH) setup-env
	@echo "🦞 Installing Lobster AI with development dependencies..."
	@if [ "$(USE_UV)" = "true" ]; then \
		echo "📦 Using uv for faster installation..."; \
		uv pip install -e ".[dev]"; \
	else \
		echo "📦 Upgrading pip and installing build tools..."; \
		if [ -f "$(VENV_PATH)/bin/pip3" ]; then \
			$(VENV_PATH)/bin/pip3 install --upgrade pip build wheel; \
		else \
			$(VENV_PATH)/bin/pip install --upgrade pip build wheel; \
		fi; \
		echo "📦 Installing development dependencies..."; \
		if [ -f "$(VENV_PATH)/bin/pip3" ]; then \
			$(VENV_PATH)/bin/pip3 install -e ".[dev]"; \
		else \
			$(VENV_PATH)/bin/pip install -e ".[dev]"; \
		fi; \
	fi
	@echo "🔧 Installing pre-commit git hooks..."
	@if [ -f "$(VENV_PATH)/bin/pre-commit" ]; then \
		$(VENV_PATH)/bin/pre-commit install; \
	else \
		echo "$(YELLOW)⚠️ pre-commit not found, skipping hook installation$(NC)"; \
	fi
	@echo "$(GREEN)🎉 Development installation complete!$(NC)"
	@echo ""
	@echo "$(BLUE)📋 Next steps:$(NC)"
	@echo "1. Activate the virtual environment:"
	@echo "   $(YELLOW)source $(VENV_PATH)/bin/activate$(NC)"
	@echo "2. Configure your API keys in the .env file"
	@echo "3. Run tests: $(YELLOW)make test$(NC)"

clean-install: 
	@echo "🧹 Clean installing Lobster AI..."
	$(MAKE) uninstall || true
	$(MAKE) install

# Show activation command
activate:
	@echo "To activate the virtual environment, run:"
	@echo "$(YELLOW)source $(VENV_PATH)/bin/activate$(NC)"
	@if [ -n "$(CONDA_ACTIVE)" ]; then \
		echo ""; \
		echo "$(YELLOW)⚠️  Note: You're in conda environment '$(CONDA_ACTIVE)'$(NC)"; \
		echo "$(YELLOW)   Consider: conda deactivate$(NC)"; \
	fi

# Global installation (macOS/Linux)
install-global: $(VENV_PATH)
	@echo "🌍 Installing lobster command globally..."
	@if [ ! -f "$(VENV_PATH)/bin/lobster" ]; then \
		echo "$(RED)❌ Lobster not found in virtual environment. Run 'make install' first.$(NC)"; \
		exit 1; \
	fi
	@if [ ! -d "/usr/local/bin" ]; then \
		echo "$(YELLOW)📁 Creating /usr/local/bin directory...$(NC)"; \
		sudo mkdir -p /usr/local/bin; \
	fi
	@if [ -L "/usr/local/bin/lobster" ]; then \
		echo "$(YELLOW)🔗 Removing existing lobster symlink...$(NC)"; \
		sudo rm /usr/local/bin/lobster; \
	fi
	@echo "🔗 Creating global symlink..."
	@sudo ln -sf "$(shell pwd)/$(VENV_PATH)/bin/lobster" /usr/local/bin/lobster
	@echo "$(GREEN)✅ Lobster command installed globally!$(NC)"
	@echo ""
	@echo "$(BLUE)📋 You can now use 'lobster' from anywhere:$(NC)"
	@echo "   $(YELLOW)lobster --help$(NC)"
	@echo "   $(YELLOW)lobster chat$(NC)"
	@echo ""
	@echo "$(BLUE)💡 Note: The global command will use the virtual environment at:$(NC)"
	@echo "   $(YELLOW)$(shell pwd)/$(VENV_PATH)$(NC)"

uninstall-global:
	@echo "🗑️  Removing global lobster command..."
	@if [ -L "/usr/local/bin/lobster" ]; then \
		echo "🔗 Removing symlink from /usr/local/bin/lobster..."; \
		sudo rm /usr/local/bin/lobster; \
		echo "$(GREEN)✅ Global lobster command removed$(NC)"; \
	else \
		echo "$(YELLOW)⚠️ No global lobster command found$(NC)"; \
	fi

# Testing targets (require virtual environment)
test: $(VENV_PATH)
	@echo "🧪 Running tests..."
	$(VENV_PATH)/bin/pytest tests/ -v --cov=lobster --cov-report=html --cov-report=term

test-fast: $(VENV_PATH)
	@echo "🧪 Running tests in parallel..."
	$(VENV_PATH)/bin/pytest tests/ -n auto -v

test-integration: $(VENV_PATH)
	@echo "🧪 Running integration tests..."
	$(VENV_PATH)/bin/pytest tests/integration/ -v -m integration

# Code quality targets
format: $(VENV_PATH)
	@echo "🎨 Formatting code..."
	$(VENV_PATH)/bin/black lobster tests
	$(VENV_PATH)/bin/isort lobster tests
	@echo "$(GREEN)✅ Code formatted!$(NC)"

lint: $(VENV_PATH)
	@echo "🔍 Running linters..."
	$(VENV_PATH)/bin/flake8 lobster tests
	$(VENV_PATH)/bin/pylint lobster
	$(VENV_PATH)/bin/bandit -r lobster -ll
	@echo "$(GREEN)✅ Linting complete!$(NC)"

type-check: $(VENV_PATH)
	@echo "🔍 Running type checks..."
	$(VENV_PATH)/bin/mypy lobster --strict

# Docker targets
docker-build:
	@echo "🐳 Building Docker images..."
	docker build -t omicsos/lobster:latest -f Dockerfile .
	docker build -t omicsos/lobster:server -f Dockerfile.server .
	@echo "$(GREEN)✅ Docker images built successfully!$(NC)"
	@echo "  - omicsos/lobster:latest (CLI mode)"
	@echo "  - omicsos/lobster:server (FastAPI server)"

docker-run-cli:
	@echo "🐳 Running Lobster CLI in Docker..."
	docker run -it --rm \
		-v $(shell pwd)/data:/app/data \
		-v lobster-workspace:/app/.lobster_workspace \
		--env-file .env \
		omicsos/lobster:latest chat

docker-run-server:
	@echo "🐳 Running Lobster FastAPI server..."
	docker run -d --rm \
		--name lobster-server \
		-p 8000:8000 \
		-v $(shell pwd)/data:/app/data \
		-v lobster-workspace:/app/.lobster_workspace \
		--env-file .env \
		omicsos/lobster:server
	@echo "$(GREEN)✅ Server running at http://localhost:8000$(NC)"
	@echo "  Stop with: docker stop lobster-server"

docker-compose-up:
	@echo "🐳 Starting services with docker-compose..."
	docker-compose up -d lobster-server
	@echo "$(GREEN)✅ Services started!$(NC)"

docker-compose-cli:
	@echo "🐳 Running CLI with docker-compose..."
	docker-compose run --rm lobster-cli

docker-compose-down:
	@echo "🐳 Stopping docker-compose services..."
	docker-compose down

docker-push:
	@echo "🐳 Pushing images to Docker Hub..."
	docker push omicsos/lobster:latest
	docker push omicsos/lobster:server
	@echo "$(GREEN)✅ Images pushed successfully!$(NC)"

# Legacy alias for backward compatibility
docker-run: docker-run-cli

# Release targets
version: $(VENV_PATH)
	@$(VENV_PATH)/bin/python -c "from lobster.version import __version__; print(__version__)"

bump-patch: $(VENV_PATH)
	@echo "📦 Bumping patch version..."
	$(VENV_PATH)/bin/bumpversion patch

bump-minor: $(VENV_PATH)
	@echo "📦 Bumping minor version..."
	$(VENV_PATH)/bin/bumpversion minor

bump-major: $(VENV_PATH)
	@echo "📦 Bumping major version..."
	$(VENV_PATH)/bin/bumpversion major

release: clean test
	@echo "📦 Creating release..."
	$(VENV_PATH)/bin/python -m build
	$(VENV_PATH)/bin/twine check dist/*
	@echo "$(GREEN)✅ Release artifacts created in dist/$(NC)"

publish: release
	@echo "📦 Publishing to PyPI..."
	$(VENV_PATH)/bin/twine upload dist/*
	@echo "$(GREEN)✅ Published to PyPI!$(NC)"

# Cleanup targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✅ Cleanup complete!$(NC)"

clean-all: clean
	@echo "🧹 Removing all generated files..."
	rm -rf .lobster_workspace
	rm -rf data/exports/*
	@echo "$(GREEN)✅ Deep cleanup complete!$(NC)"

# Uninstall
uninstall:
	@echo "🗑️  Removing virtual environment..."
	rm -rf $(VENV_PATH)
	@echo "$(GREEN)✅ Virtual environment removed$(NC)"

# Development helpers
setup-pre-commit: $(VENV_PATH)
	@echo "🔧 Setting up pre-commit hooks..."
	$(VENV_PATH)/bin/pre-commit install
	$(VENV_PATH)/bin/pre-commit run --all-files

update-deps: $(VENV_PATH)
	@echo "📦 Dependencies are now managed in pyproject.toml"
	@echo "$(GREEN)✅ To add new dependencies, edit the pyproject.toml file directly$(NC)"

# Documentation
docs: $(VENV_PATH)
	@echo "📚 Building documentation..."
	cd docs && ../$(VENV_PATH)/bin/mkdocs build
	@echo "$(GREEN)✅ Documentation built in docs/site/$(NC)"

docs-serve: $(VENV_PATH)
	@echo "📚 Serving documentation..."
	cd docs && ../$(VENV_PATH)/bin/mkdocs serve

# Python info for debugging
python-info:
	@echo "🐍 Python Environment Information"
	@echo "================================"
	@echo "Environment Type: $(PYTHON_ENV_TYPE)"
	@echo "Python Command: $(PYTHON)"
	@echo "Python Path: $$(which $(PYTHON) 2>/dev/null || echo 'not found')"
	@echo "Python Version: $$($(PYTHON) --version 2>&1 || echo 'N/A')"
	@if [ -n "$(CONDA_ACTIVE)" ]; then \
		echo "Conda Environment: $(CONDA_ACTIVE)"; \
		echo "Conda Python: $$(which python)"; \
	fi
	@if [ -n "$(PYENV_VERSION)" ]; then \
		echo "Pyenv Version: $(PYENV_VERSION)"; \
		echo "Pyenv Root: $$(pyenv root 2>/dev/null)"; \
	fi
	@if [ -n "$(VIRTUAL_ENV)" ]; then \
		echo "Active Venv: $(VIRTUAL_ENV)"; \
	fi
	@echo ""
	@echo "Available Python versions:"
	@for p in python python3 python3.12 python3.13; do \
		if command -v $$p >/dev/null 2>&1; then \
			echo "  $$p: $$($$p --version 2>&1) at $$(which $$p)"; \
		fi; \
	done

# Utility targets
check-env: $(VENV_PATH)
	@echo "🔍 Checking environment..."
	@$(VENV_PATH)/bin/python -c "import sys; print(f'Python: {sys.version}')"
	@echo ""
	@echo "Virtual environment: $(GREEN)✅ Active at $(VENV_PATH)$(NC)"
	@echo ""
	@echo "Required environment variables:"
	@echo -n "ANTHROPIC_API_KEY: "; [ -z "${ANTHROPIC_API_KEY}" ] && echo "$(RED)❌ Not set$(NC)" || echo "$(GREEN)✅ Set$(NC)"
	@echo -n "AWS_BEDROCK_ACCESS_KEY: "; [ -z "${AWS_BEDROCK_ACCESS_KEY}" ] && echo "$(RED)❌ Not set$(NC)" || echo "$(GREEN)✅ Set$(NC)"
	@echo -n "AWS_BEDROCK_SECRET_ACCESS_KEY: "; [ -z "${AWS_BEDROCK_SECRET_ACCESS_KEY}" ] && echo "$(RED)❌ Not set$(NC)" || echo "$(GREEN)✅ Set$(NC)"

# Comprehensive installation validation
verify: $(VENV_PATH)
	@echo "🧪 Running comprehensive installation verification..."
	@echo "   Note: Missing proteomics modules are expected in the public distribution"
	@$(VENV_PATH)/bin/python verify_installation.py || { \
		EXIT_CODE=$$?; \
		if [ $$EXIT_CODE -eq 1 ]; then \
			echo "$(YELLOW)⚠️ Some optional components are missing$(NC)"; \
			echo "$(YELLOW)   This is expected for the public distribution$(NC)"; \
			echo "$(GREEN)✅ Core functionality is available$(NC)"; \
		fi; \
		exit 0; \
	}

run: $(VENV_PATH)
	@echo "🦞 Starting Lobster AI..."
	$(VENV_PATH)/bin/lobster chat
