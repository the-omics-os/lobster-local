# Lobster AI Makefile
# Professional build and development commands

.PHONY: help install dev-install install-global uninstall-global test format lint clean docker-build docker-run release check-python setup-env

# Configuration
PYTHON := python3
VENV_NAME := .venv
VENV_PATH := $(VENV_NAME)
PYTHON_VERSION_MIN := 3.12
PROJECT_NAME := lobster-ai

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
	@echo "  make install        Install Lobster AI in virtual environment"
	@echo "  make dev-install    Install with development dependencies"
	@echo "  make install-global Install lobster command globally (macOS/Linux)"
	@echo "  make clean-install  Clean install (remove existing installation)"
	@echo "  make setup-env      Setup environment configuration"
	@echo "  make activate       Show activation command"
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
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make docker-push   Push to Docker Hub"
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

# Python version check with platform guidance
check-python:
	@echo "🔍 Checking Python version..."
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "$(RED)❌ $(PYTHON) command not found. Please install Python 3.12+.$(NC)"; \
		echo "$(YELLOW)📋 Installation instructions:$(NC)"; \
		if [ "$$(uname -s)" = "Darwin" ]; then \
			echo "$(YELLOW)   macOS: brew install python@3.12$(NC)"; \
			if [ "$$(uname -m)" = "arm64" ]; then \
				echo "$(BLUE)   🍎 Apple Silicon detected - using optimized installation$(NC)"; \
			fi; \
		else \
			echo "$(YELLOW)   Ubuntu: sudo apt install python3.12 python3.12-dev$(NC)"; \
		fi; \
		exit 1; \
	fi
	@$(PYTHON) -c "import sys; print(sys.version_info)" > /dev/null 2>&1 || { \
		echo "$(RED)❌ Failed to execute Python. Please check your installation.$(NC)"; \
		exit 1; \
	}
	@$(PYTHON) -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)" || { \
		echo "$(RED)❌ Python 3.12+ is required. Found: $$($(PYTHON) --version)$(NC)"; \
		echo "$(YELLOW)📋 Upgrade instructions:$(NC)"; \
		if [ "$$(uname -s)" = "Darwin" ]; then \
			echo "$(YELLOW)   macOS: brew install python@3.12 && brew link python@3.12$(NC)"; \
		else \
			echo "$(YELLOW)   Ubuntu: sudo apt install python3.12$(NC)"; \
		fi; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Python 3.12+ check passed: $$($(PYTHON) --version)$(NC)"
	@echo "🔍 Checking venv module..."
	@$(PYTHON) -c "import venv" > /dev/null 2>&1 || { \
		echo "$(RED)❌ Python venv module not found. Please install python3-venv package.$(NC)"; \
		echo "$(YELLOW)📋 Fix instructions:$(NC)"; \
		echo "$(YELLOW)  Ubuntu/Debian: sudo apt install python3.12-venv$(NC)"; \
		echo "$(YELLOW)  macOS: brew reinstall python@3.12$(NC)"; \
		exit 1; \
	}
	@echo "$(GREEN)✅ Python venv module available$(NC)"
	@echo "$(BLUE)🔍 Checking system dependencies...$(NC)"
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		if ! command -v brew >/dev/null 2>&1; then \
			echo "$(YELLOW)⚠️ Homebrew not found. Install: /bin/bash -c \"$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"$(NC)"; \
		fi; \
		if ! brew list hdf5 >/dev/null 2>&1; then \
			echo "$(YELLOW)⚠️ HDF5 not found. Install: brew install hdf5$(NC)"; \
		fi; \
	else \
		if ! dpkg -l python3.12-dev >/dev/null 2>&1; then \
			echo "$(YELLOW)⚠️ python3.12-dev not found. Install: sudo apt install python3.12-dev$(NC)"; \
		fi; \
		if ! dpkg -l libhdf5-dev >/dev/null 2>&1; then \
			echo "$(YELLOW)⚠️ libhdf5-dev not found. Install: sudo apt install libhdf5-dev$(NC)"; \
		fi; \
	fi

$(VENV_PATH):
	@echo "🐍 Creating virtual environment with $(PYTHON)..."
	@if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "$(RED)❌ $(PYTHON) command not found. Please install Python 3.12+.$(NC)"; \
		echo "$(YELLOW)   macOS: brew install python@3.12$(NC)"; \
		echo "$(YELLOW)   Ubuntu/Debian: sudo apt install python3.12$(NC)"; \
		exit 1; \
	fi
	@if ! $(PYTHON) -c "import ensurepip" >/dev/null 2>&1; then \
		echo "$(RED)❌ Python ensurepip module not found. Your Python might be missing venv support.$(NC)"; \
		if [ "$$(uname -s)" = "Darwin" ]; then \
			echo "$(YELLOW)   Try: brew reinstall python@3.12$(NC)"; \
		else \
			echo "$(YELLOW)   Try: sudo apt install python3.12-venv python3.12-distutils$(NC)"; \
		fi; \
		exit 1; \
	fi
	@echo "   Using package manager: $(PKG_MGR)"
	@if [ "$$(uname -m)" = "arm64" ] && [ "$$(uname -s)" = "Darwin" ]; then \
		echo "$(BLUE)   🍎 Apple Silicon optimization enabled$(NC)"; \
	fi
	@$(PYTHON) -m venv $(VENV_PATH) || { \
		echo "$(RED)❌ Failed to create virtual environment. Trying recovery options...$(NC)"; \
		echo "$(YELLOW)🔄 Attempting recovery with --without-pip flag...$(NC)"; \
		$(PYTHON) -m venv $(VENV_PATH) --without-pip || { \
			echo "$(RED)❌ Virtual environment creation completely failed.$(NC)"; \
			echo "$(YELLOW)📋 Troubleshooting steps:$(NC)"; \
			if [ "$$(uname -s)" = "Darwin" ]; then \
				echo "$(YELLOW)   1. brew uninstall python@3.12 && brew install python@3.12$(NC)"; \
				echo "$(YELLOW)   2. Check Xcode Command Line Tools: xcode-select --install$(NC)"; \
			else \
				echo "$(YELLOW)   1. sudo apt update && sudo apt install python3.12-venv$(NC)"; \
				echo "$(YELLOW)   2. Check disk space: df -h$(NC)"; \
				echo "$(YELLOW)   3. Check permissions: ls -la .$(NC)"; \
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
		if [ -f .env.template ]; then \
			cp .env.template .env; \
			echo "$(YELLOW)📋 Created .env from template. Please edit it with your API keys.$(NC)"; \
		else \
			echo "# Lobster AI Environment Variables" > .env; \
			echo "# Required API Keys" >> .env; \
			echo "ANTHROPIC_API_KEY=sk-ant-api03-your-key-here" >> .env; \
			echo "AWS_BEDROCK_ACCESS_KEY=your-aws-access-key-here" >> .env; \
			echo "AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key-here" >> .env; \
			echo "" >> .env; \
			echo "# Optional" >> .env; \
			echo "NCBI_API_KEY=your-ncbi-api-key-here" >> .env; \
			echo "GENIE_PROFILE=production" >> .env; \
			echo "$(YELLOW)📋 Created .env file. Please edit it with your API keys.$(NC)"; \
		fi; \
	else \
		echo "$(GREEN)✅ .env file already exists$(NC)"; \
	fi

# Installation targets
install: check-python $(VENV_PATH) setup-env
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
		echo "$(YELLOW)⚠️ Some verification tests failed, but installation may still work$(NC)"; \
		echo "$(YELLOW)   Continue with configuration if basic imports succeeded$(NC)"; \
	}
	@echo ""
	@echo "$(BLUE)📋 Next steps:$(NC)"
	@echo "1. Activate the virtual environment:"
	@echo "   $(YELLOW)source $(VENV_PATH)/bin/activate$(NC)"
	@echo ""
	@echo "2. Configure your API keys in the .env file:"
	@echo "   $(YELLOW)nano .env$(NC)"
	@echo "   $(BLUE)Required: ANTHROPIC_API_KEY or AWS Bedrock credentials$(NC)"
	@echo ""
	@echo "3. Test the installation:"
	@echo "   $(YELLOW)lobster --help$(NC)"
	@echo ""
	@echo "4. Start using Lobster AI:"
	@echo "   $(YELLOW)lobster chat$(NC)"
	@echo ""
	@echo "$(BLUE)💡 Tip: Try asking 'Download GSE109564 and perform single-cell analysis'$(NC)"

dev-install: check-python $(VENV_PATH) setup-env
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
	$(VENV_PATH)/bin/pytest tests/test_lobster.py -v --cov=lobster --cov-report=html --cov-report=term

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
	@echo "🐳 Building Docker image..."
	docker build -t homaraai/lobster:latest .
	@echo "$(GREEN)✅ Docker image built!$(NC)"

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -it --rm \
		-v ~/.lobster:/root/.lobster \
		-e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
		-e AWS_BEDROCK_ACCESS_KEY=${AWS_BEDROCK_ACCESS_KEY} \
		-e AWS_BEDROCK_SECRET_ACCESS_KEY=${AWS_BEDROCK_SECRET_ACCESS_KEY} \
		homaraai/lobster:latest

docker-push:
	@echo "🐳 Pushing to Docker Hub..."
	docker push homaraai/lobster:latest

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
	@$(VENV_PATH)/bin/python verify_installation.py

run: $(VENV_PATH)
	@echo "🦞 Starting Lobster AI..."
	$(VENV_PATH)/bin/lobster chat
