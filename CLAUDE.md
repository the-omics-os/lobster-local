# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lobster AI is a multi-agent bioinformatics analysis platform that combines specialized AI agents with proven scientific tools to analyze complex multi-omics data. Users interact through natural language to perform RNA-seq, proteomics, and multi-omics analysis.

## Essential Development Commands

### Environment Setup
```bash
# Initial setup with development dependencies
make dev-install

# Basic installation
make install

# Clean installation (removes existing environment)
make clean-install
```

### Testing and Quality Assurance
```bash
# Run all tests
make test

# Run tests in parallel
make test-fast

# Code formatting (black + isort)
make format

# Linting (flake8, pylint, bandit)
make lint

# Type checking
make type-check
```

### Running the Application
```bash
# Start interactive chat mode with enhanced autocomplete
lobster chat

# Show help
lobster --help

# Alternative module execution
python -m lobster
```

## CLI Interface

The CLI (`lobster/cli.py`) features a modern terminal interface with comprehensive autocomplete functionality:

### Enhanced Input Features
- **Tab Autocomplete**: Smart completion for commands and workspace files
- **Context-Aware**: Commands when typing `/`, files after `/read` or `/plot`
- **Cloud Integration**: Works seamlessly with both local and cloud clients
- **Rich Metadata**: Shows file sizes, types, and descriptions in completion menu
- **Arrow Navigation**: Full prompt_toolkit integration for enhanced editing
- **Command History**: Persistent history with Ctrl+R reverse search

### Key Implementation Details
- **LobsterClientAdapter**: Unified interface for local `AgentClient` and `CloudLobsterClient`
- **Dynamic Command Discovery**: Automatically extracts available commands from `_execute_command()`
- **Intelligent Caching**: 60s cache for cloud, 10s for local file operations
- **Graceful Fallback**: Falls back to Rich input if prompt_toolkit unavailable
- **Orange Theme Integration**: Completion menu matches existing Lobster branding

### Essential Commands
- `/help` - Show all available commands with descriptions
- `/status` - Show system status and client type
- `/files` - List workspace files by category
- `/read <file>` - Read and load files (supports Tab completion)
- `/data` - Show current dataset information
- `/plots` - List generated visualizations
- `/workspace` - Show workspace information
- `/modes` - List available operation modes

## Architecture Overview

### Core Components

- **`lobster/agents/`** - Specialized AI agents
  - `singlecell_expert.py` - Single-cell RNA-seq analysis
  - `bulk_rnaseq_expert.py` - Bulk RNA-seq analysis
  - `ms_proteomics_expert.py` - Mass spectrometry proteomics
  - `affinity_proteomics_expert.py` - Affinity proteomics (Olink panels)
  - `data_expert.py` - Data loading and quality assessment
  - `research_agent.py` - Literature mining and dataset discovery
  - `supervisor.py` - Agent coordination and workflow management

- **`lobster/core/`** - Data management and client infrastructure
  - `client.py` - AgentClient (local LangGraph processing)
  - `api_client.py` - APIAgentClient (WebSocket streaming)
  - `data_manager_v2.py` - DataManagerV2 (multi-omics orchestrator)
  - `interfaces/base_client.py` - BaseClient interface for cloud/local consistency

- **`lobster/tools/`** - Analysis services
  - `preprocessing_service.py` - Data preprocessing and normalization
  - `clustering_service.py` - Clustering and cell type annotation
  - `visualization_service.py` - Publication-quality plotting
  - `proteomics_*_service.py` - Professional proteomics analysis suite
  - `publication_service.py` - Literature mining and GEO search

- **`lobster/config/`** - Configuration management
  - `agent_config.py` - Agent configuration and LLM settings
  - `agent_registry.py` - Centralized agent registry (single source of truth)

### Key Design Principles

1. **Agent-Based Architecture** - Specialist agents with centralized registry
2. **Cloud/Local Hybrid** - Seamless switching between local and cloud execution
3. **Modular Services** - Stateless analysis services for bioinformatics workflows
4. **Natural Language Interface** - Users describe analyses in plain English
5. **Publication-Quality Output** - Interactive Plotly visualizations with scientific rigor

### Client Architecture

The system supports multiple client types through the `BaseClient` interface:

```python
# lobster/core/interfaces/base_client.py
class BaseClient(ABC):
    @abstractmethod
    def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]
    @abstractmethod
    def get_status(self) -> Dict[str, Any]
    @abstractmethod
    def export_session(self, export_path: Optional[Path] = None) -> Path
```

**Client Types:**
- **AgentClient** - Local LangGraph processing with DataManagerV2
- **APIAgentClient** - WebSocket streaming for web services
- **CloudLobsterClient** - HTTP REST API for cloud services (external package)

### Data Management

**DataManagerV2** handles multi-modal data orchestration:
- Named biological datasets (`Dict[str, AnnData]`)
- Metadata store for GEO and source metadata
- Tool usage history for provenance tracking
- Backend/adapter registry for extensible data handling

**Professional Naming Convention:**
```
geo_gse12345                     # Raw downloaded data
├── geo_gse12345_filtered        # Preprocessed data
├── geo_gse12345_clustered      # Clustering results
└── geo_gse12345_annotated     # Cell type annotations
```

## Development Guidelines

### Code Style and Quality
- Follow PEP 8 Python style guidelines
- Use type hints for all functions and methods
- Line length: 88 characters (Black formatting)
- Add comprehensive docstrings to all public functions
- Prioritize scientific accuracy over performance optimizations

### Working with the CLI
- All CLI functionality is in `lobster/cli.py`
- Commands are handled in `_execute_command()` function
- Autocomplete classes are defined at the top of the file after imports
- Use `PROMPT_TOOLKIT_AVAILABLE` flag for optional dependency handling
- Maintain backward compatibility with Rich input fallback

### Adding New Commands
1. Add command logic to `_execute_command()`
2. Update command descriptions in `extract_available_commands()`
3. Commands automatically appear in autocomplete
4. Test with both local and cloud clients

### Agent Tool Pattern
```python
@tool
def analyze_modality(modality_name: str, **params) -> str:
    # 1. Validate modality exists
    adata = data_manager.get_modality(modality_name)

    # 2. Call stateless service
    result_adata, stats = service.analyze(adata, **params)

    # 3. Store results with descriptive naming
    new_modality = f"{modality_name}_analyzed"
    data_manager.modalities[new_modality] = result_adata

    # 4. Log operation for provenance
    data_manager.log_tool_usage("analyze_modality", params, stats)

    return formatted_response(stats, new_modality)
```

## Environment Configuration

### Required Environment Variables
```bash
# API Keys (required)
OPENAI_API_KEY=your-openai-api-key
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key

# Optional
NCBI_API_KEY=your-ncbi-api-key
LOBSTER_CLOUD_KEY=your-cloud-api-key  # Enables cloud mode
```

### Cloud Integration
- Set `LOBSTER_CLOUD_KEY` to enable cloud mode
- System automatically detects and switches between local/cloud
- Autocomplete works with both client types
- Fallback to local mode if cloud unavailable

## Testing Framework

```bash
# Run all tests with coverage
make test

# Fast parallel execution
make test-fast

# Run specific categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
```

**Coverage Requirements:**
- Minimum 80% coverage (targeting 95%+)
- Test with real bioinformatics data when possible
- Include edge cases and error conditions

## Critical Rules

- **NEVER** modify `pyproject.toml` - all installations requested through user
- Always prefer editing existing files over creating new ones
- Maintain backward compatibility when updating CLI
- Test autocomplete with both local and cloud clients
- Follow scientific accuracy standards for bioinformatics algorithms

## Common Troubleshooting

### Installation Issues
- Ensure Python 3.12+ is installed
- Use `make clean-install` for fresh environment
- For enhanced CLI features: `pip install prompt-toolkit`

### CLI Issues
- Check `PROMPT_TOOLKIT_AVAILABLE` flag for autocomplete functionality
- Verify client type detection in `LobsterClientAdapter`
- Test fallback to Rich input if prompt_toolkit fails
- Check file permissions for workspace access

### Cloud Integration
- Verify `LOBSTER_CLOUD_KEY` environment variable
- Check network connectivity for cloud operations
- Monitor cache timeouts (60s cloud, 10s local)
- Confirm BaseClient interface compliance