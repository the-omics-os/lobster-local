# Lobster AI - Refactoring Plan

## Project Overview
- **Repository Name**: lobster-ai
- **Product Name**: Lobster
- **Description**: Multi-Agent Bioinformatics Analysis System
- **Installation**: pip install lobster-ai

## New Directory Structure
```
lobster-ai/
├── README.md                 # Main documentation
├── LICENSE                   # MIT License
├── CONTRIBUTING.md          # Contribution guidelines
├── CHANGELOG.md             # Version history
├── Makefile                 # Installation and development commands
├── Dockerfile               # Container definition
├── docker-compose.yml       # Development environment
├── setup.py                 # Python package setup
├── pyproject.toml          # Modern Python packaging
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── .env.example            # Example environment variables
├── .gitignore              # Git ignore rules
│
├── lobster/                # Main package directory
│   ├── __init__.py
│   ├── __main__.py         # Entry point for `python -m lobster`
│   ├── cli.py              # Main CLI application (renamed from agent_cli.py)
│   ├── version.py          # Version information
│   │
│   ├── agents/             # Multi-agent system
│   │   ├── __init__.py
│   │   ├── graph.py        # LangGraph orchestration
│   │   ├── supervisor.py   # Supervisor agent
│   │   ├── transcriptomics.py  # Transcriptomics expert
│   │   ├── methods.py      # Methods expert
│   │   └── state.py        # Agent states
│   │
│   ├── tools/              # Analysis tools
│   │   ├── __init__.py
│   │   ├── geo.py          # GEO data tools
│   │   ├── clustering.py   # Clustering analysis
│   │   ├── quality.py      # Quality control
│   │   └── pubmed.py       # PubMed integration
│   │
│   ├── core/               # Core functionality
│   │   ├── __init__.py
│   │   ├── data_manager.py # Data management
│   │   ├── config.py       # Configuration management
│   │   └── client.py       # Agent client
│   │
│   └── utils/              # Utilities
│       ├── __init__.py
│       ├── logger.py       # Logging
│       └── callbacks.py    # Callback handlers
│
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── conftest.py        # Pytest configuration
│   ├── test_agents.py     # Agent tests
│   ├── test_tools.py      # Tool tests
│   └── test_cli.py        # CLI tests
│
├── docs/                   # Documentation
│   ├── installation.md     # Installation guide
│   ├── quickstart.md      # Getting started
│   ├── configuration.md   # Configuration guide
│   ├── api.md            # API reference
│   └── examples/         # Example notebooks
│
└── scripts/              # Utility scripts
    ├── install.sh        # Curl installation script
    ├── setup_env.sh      # Environment setup
    └── get_api_key.py    # API key helper
```

## Key Changes

### 1. Remove Streamlit Components
- Delete `app.py`, `app/` directory
- Remove `services/langgraph_agent_service_OLD.py`
- Clean up Streamlit dependencies from requirements.txt

### 2. Rename and Reorganize
- `agent_cli.py` → `lobster/cli.py`
- `lobster/` directory → `lobster/agents/`
- `clients/agent_client.py` → `lobster/core/client.py`
- Consolidate tools in `lobster/tools/`

### 3. Professional Entry Points
- Main CLI: `lobster` command
- Python module: `python -m lobster`
- Docker: `docker run lobster-ai`

### 4. Installation Methods

#### pip install
```bash
pip install lobster-ai
```

#### Homebrew
```bash
brew tap homara-ai/tap
brew install lobster
```

#### Curl one-liner
```bash
curl -sSL https://get.lobster-ai.com | bash
```

### 5. Configuration Strategy
- Default: Use `.env` file
- Advanced: Custom configuration files
- Easy mode: `lobster configure` command to set up API keys
- Hosted option: Use Homara AI API keys

## Implementation Steps

1. **Phase 1: Core Restructuring**
   - Create new directory structure
   - Move and rename files
   - Remove Streamlit dependencies

2. **Phase 2: CLI Enhancement**
   - Refactor `agent_cli.py` to professional CLI
   - Add proper command structure
   - Implement configuration wizard

3. **Phase 3: Packaging**
   - Create `setup.py` and `pyproject.toml`
   - Set up entry points
   - Configure package metadata

4. **Phase 4: Installation**
   - Create Makefile with common tasks
   - Optimize Dockerfile
   - Create installation scripts

5. **Phase 5: Documentation**
   - Rewrite README.md
   - Create user guides
   - Add API documentation

6. **Phase 6: Testing**
   - Set up pytest framework
   - Add unit tests
   - Create integration tests
