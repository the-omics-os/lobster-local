# Lobster AI Makefile
# Professional build and development commands

.PHONY: help install dev-install test format lint clean docker-build docker-run release check-python setup-env

# Configuration
PYTHON := python3
VENV_NAME := .venv
VENV_PATH := $(VENV_NAME)
PYTHON_VERSION_MIN := 3.9
PROJECT_NAME := lobster-ai

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

# Python version check
check-python:
	@echo "🔍 Checking Python version..."
	@$(PYTHON) -c "import sys; exit(0 if sys.version_info >= ($(shell echo $(PYTHON_VERSION_MIN) | tr '.' ',')) else 1)" || \
	(echo "$(RED)❌ Python $(PYTHON_VERSION_MIN)+ required. Found: $$($(PYTHON) --version)$(NC)" && exit 1)
	@echo "$(GREEN)✅ Python version check passed: $$($(PYTHON) --version)$(NC)"

# Virtual environment creation
$(VENV_PATH):
	@echo "🐍 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_PATH)
	@echo "$(GREEN)✅ Virtual environment created at $(VENV_PATH)$(NC)"

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
			echo "OPENAI_API_KEY=your-openai-api-key-here" >> .env; \
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
	@echo "📦 Upgrading pip and installing build tools..."
	$(VENV_PATH)/bin/pip install --upgrade pip setuptools wheel
	@echo "📦 Installing Lobster AI and dependencies..."
	$(VENV_PATH)/bin/pip install -e .
	@echo ""
	@echo "$(GREEN)🎉 Installation complete!$(NC)"
	@echo ""
	@echo "$(BLUE)📋 Next steps:$(NC)"
	@echo "1. Activate the virtual environment:"
	@echo "   $(YELLOW)source $(VENV_PATH)/bin/activate$(NC)"
	@echo ""
	@echo "2. Configure your API keys in the .env file:"
	@echo "   $(YELLOW)nano .env$(NC)"
	@echo ""
	@echo "3. Test the installation:"
	@echo "   $(YELLOW)lobster --help$(NC)"
	@echo ""
	@echo "4. Start using Lobster AI:"
	@echo "   $(YELLOW)lobster chat$(NC)"

dev-install: check-python $(VENV_PATH) setup-env
	@echo "🦞 Installing Lobster AI with development dependencies..."
	@echo "📦 Upgrading pip and installing build tools..."
	$(VENV_PATH)/bin/pip install --upgrade pip setuptools wheel
	@echo "📦 Installing development dependencies..."
	$(VENV_PATH)/bin/pip install -e ".[dev]"
	$(VENV_PATH)/bin/pre-commit install
	@echo ""
	@echo "$(GREEN)🎉 Development installation complete!$(NC)"
	@echo ""
	@echo "$(BLUE)📋 Next steps:$(NC)"
	@echo "1. Activate the virtual environment:"
	@echo "   $(YELLOW)source $(VENV_PATH)/bin/activate$(NC)"
	@echo ""
	@echo "2. Configure your API keys in the .env file"
	@echo "3. Run tests: $(YELLOW)make test$(NC)"

clean-install: 
	uninstall install

# Show activation command
activate:
	@echo "To activate the virtual environment, run:"
	@echo "$(YELLOW)source $(VENV_PATH)/bin/activate$(NC)"

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
	@echo "🐳 Building Docker image..."
	docker build -t homaraai/lobster:latest .
	@echo "$(GREEN)✅ Docker image built!$(NC)"

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -it --rm \
		-v ~/.lobster:/root/.lobster \
		-e OPENAI_API_KEY=${OPENAI_API_KEY} \
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
	@echo "📦 Updating dependencies..."
	$(VENV_PATH)/bin/pip-compile requirements.in -o requirements.txt
	$(VENV_PATH)/bin/pip-compile requirements-dev.in -o requirements-dev.txt
	@echo "$(GREEN)✅ Dependencies updated!$(NC)"

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
	@echo -n "OPENAI_API_KEY: "; [ -z "${OPENAI_API_KEY}" ] && echo "$(RED)❌ Not set$(NC)" || echo "$(GREEN)✅ Set$(NC)"
	@echo -n "AWS_BEDROCK_ACCESS_KEY: "; [ -z "${AWS_BEDROCK_ACCESS_KEY}" ] && echo "$(RED)❌ Not set$(NC)" || echo "$(GREEN)✅ Set$(NC)"
	@echo -n "AWS_BEDROCK_SECRET_ACCESS_KEY: "; [ -z "${AWS_BEDROCK_SECRET_ACCESS_KEY}" ] && echo "$(RED)❌ Not set$(NC)" || echo "$(GREEN)✅ Set$(NC)"

run: $(VENV_PATH)
	@echo "🦞 Starting Lobster AI..."
	$(VENV_PATH)/bin/lobster chat
