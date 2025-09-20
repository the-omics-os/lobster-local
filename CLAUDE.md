# CLAUDE.md

This file provides comprehensive guidance for AI agents and bots contributing to the Lobster AI codebase. It contains architectural context, connectivity between components, and critical implementation details.

## Project Overview

Lobster AI is a professional **multi-agent bioinformatics analysis platform** that combines specialized AI agents with proven scientific tools to analyze complex multi-omics data. Users interact through natural language to perform RNA-seq, proteomics, and multi-omics analysis.

### Core Capabilities
- **Single-Cell RNA-seq**: Quality control, clustering, cell type annotation, trajectory analysis, pseudobulk aggregation
- **Bulk RNA-seq**: Differential expression with pyDESeq2, R-style formula-based statistics, complex experimental designs
- **Mass Spectrometry Proteomics**: DDA/DIA workflows, missing value handling (30-70% typical), peptide-to-protein mapping, intensity normalization
- **Affinity Proteomics**: Olink panels, antibody arrays, targeted protein panels, CV analysis, low missing values (<30%)
- **Multi-Omics Integration**: (Future feature) Cross-platform analysis using MuData framework
- **Literature Mining**: Automated parameter extraction from publications via PubMed and GEO

### Supported Data Formats
- **Input**: CSV, Excel, H5AD, 10X MTX, MaxQuant output, Spectronaut results, Olink NPX values
- **Databases**: GEO (GSE datasets), PubMed, UniProt, Reactome, KEGG, STRING, BioGRID
- **Storage**: H5AD (single modality), MuData (multi-modal), S3-ready backends

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

#### **`lobster/agents/`** - Specialized AI agents
- `singlecell_expert.py` - Single-cell RNA-seq analysis with formula-guided DE
- `bulk_rnaseq_expert.py` - Bulk RNA-seq analysis with pyDESeq2 integration
- `ms_proteomics_expert.py` - Mass spectrometry proteomics (DDA/DIA, MNAR/MCAR missing value patterns)
- `affinity_proteomics_expert.py` - Affinity proteomics (Olink panels, antibody validation)
- `data_expert.py` - Data loading, quality assessment, sample concatenation
- `research_agent.py` - Literature mining and dataset discovery
- `method_expert.py` - Computational parameter extraction from publications
- `supervisor.py` - Agent coordination and workflow management

#### **`lobster/core/`** - Data management and client infrastructure
- `client.py` - AgentClient (local LangGraph processing)
- `api_client.py` - APIAgentClient (WebSocket streaming)
- `data_manager_v2.py` - DataManagerV2 (multi-omics orchestrator with modality management)
- `interfaces/base_client.py` - BaseClient interface for cloud/local consistency
- `provenance.py` - W3C-PROV compliant analysis history tracking
- `schemas/` - Transcriptomics and proteomics metadata validation

#### **`lobster/tools/`** - Stateless analysis services
- **Transcriptomics Services:**
  - `preprocessing_service.py` - Filter, normalize, batch correction
  - `quality_service.py` - Multi-metric QC assessment
  - `clustering_service.py` - Leiden clustering, UMAP, cell annotation
  - `enhanced_singlecell_service.py` - Doublet detection, marker genes
  - `bulk_rnaseq_service.py` - pyDESeq2 differential expression
  - `pseudobulk_service.py` - Single-cell to pseudobulk aggregation
  - `differential_formula_service.py` - R-style formula parsing, design matrices
  - `concatenation_service.py` - Memory-efficient sample merging (eliminates 450+ lines of duplication)

- **Proteomics Services:**
  - `proteomics_preprocessing_service.py` - MS/affinity filtering, normalization (TMM, quantile, VSN)
  - `proteomics_quality_service.py` - Missing value patterns, CV analysis, batch detection
  - `proteomics_analysis_service.py` - Statistical testing, PCA, clustering
  - `proteomics_differential_service.py` - Linear models with empirical Bayes, FDR control
  - `proteomics_visualization_service.py` - Volcano plots, correlation networks, QC dashboards

- **Data & Publication Services:**
  - `geo_service.py` - GEO dataset downloading and metadata extraction
  - `publication_service.py` - PubMed literature mining, GEO dataset search
  - `visualization_service.py` - Plotly-based interactive visualizations

#### **`lobster/config/`** - Configuration management
- `agent_config.py` - Agent configuration and LLM settings
- `agent_registry.py` - **Centralized agent registry (single source of truth)**
  - Eliminates redundancy when adding new agents
  - Automatic handoff tool generation
  - Dynamic agent loading with factory functions
  - Type-safe AgentConfig dataclass
- `settings.py` - System configuration and environment management

### Key Design Principles

1. **Agent-Based Architecture** - Specialist agents with centralized registry (single source of truth)
2. **Cloud/Local Hybrid** - Seamless switching between local and cloud execution
3. **Modular Services** - Stateless analysis services for bioinformatics workflows
4. **Natural Language Interface** - Users describe analyses in plain English
5. **Publication-Quality Output** - Interactive Plotly visualizations with scientific rigor
6. **Professional & Extensible** - Modular architecture designed for easy addition of future features
7. **Data Quality Compliance** - Publication-grade standards with 60% compliant, 26% partial implementation

### Client Architecture & Cloud/Local Switching

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

**Cloud/Local Switching:**
- System detects `LOBSTER_CLOUD_KEY` environment variable
- Automatic fallback to local if cloud unavailable
- CLI adapter (`LobsterClientAdapter`) provides unified interface

**⚠️ Cloud Integration Considerations:**
When modifying the codebase, be aware of cloud dependencies:
- **BaseClient Interface**: Changes must maintain compatibility
- **CLI Commands**: Must work with both local and cloud clients
- **File Operations**: Cloud uses different caching (60s vs 10s local)
- **DataManagerV2**: Cloud client may not have direct access to modalities
- **Agent Registry**: Changes affect both local graph creation and cloud handoffs

### Data Management & Scientific Workflows

**DataManagerV2** handles multi-modal data orchestration:
- Named biological datasets (`Dict[str, AnnData]`)
- Metadata store for GEO and source metadata
- Tool usage history for provenance tracking (W3C-PROV compliant)
- Backend/adapter registry for extensible data handling
- Schema validation for transcriptomics and proteomics data

**Professional Naming Convention:**
```
geo_gse12345                          # Raw downloaded data
├── geo_gse12345_quality_assessed     # QC metrics added
├── geo_gse12345_filtered_normalized  # Preprocessed data
├── geo_gse12345_doublets_detected    # Doublet annotations
├── geo_gse12345_clustered           # Leiden clustering + UMAP
├── geo_gse12345_markers              # Differential expression
├── geo_gse12345_annotated           # Cell type annotations
└── geo_gse12345_pseudobulk          # Aggregated for DE analysis
```

**Scientific Analysis Workflows:**

1. **Single-Cell RNA-seq Pipeline:**
   - Quality control (mitochondrial%, ribosomal%, gene counts)
   - Normalization (log1p, scaling)
   - Highly variable gene selection
   - PCA → Neighbors → Leiden clustering → UMAP
   - Marker gene identification (Wilcoxon rank-sum)
   - Cell type annotation (manual or automated)
   - Pseudobulk aggregation for DE analysis

2. **Bulk RNA-seq with pyDESeq2:**
   - Count matrix normalization
   - R-style formula construction (~condition + batch)
   - Design matrix generation
   - Differential expression with FDR control
   - Iterative analysis with result comparison

3. **Mass Spectrometry Proteomics:**
   - Missing value pattern analysis (MNAR vs MCAR)
   - Intensity normalization (TMM, quantile, VSN)
   - Peptide-to-protein aggregation
   - Batch effect detection and correction
   - Statistical testing with multiple correction
   - Pathway enrichment analysis

4. **Affinity Proteomics (Olink/Antibody Arrays):**
   - NPX value processing
   - Lower missing values (<30%)
   - Coefficient of variation analysis
   - Antibody validation metrics
   - Panel comparison and harmonization

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
    """Standard pattern for all agent tools."""
    try:
        # 1. Validate modality exists
        if modality_name not in data_manager.list_modalities():
            raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

        adata = data_manager.get_modality(modality_name)

        # 2. Call stateless service (returns tuple)
        result_adata, stats = service.analyze(adata, **params)

        # 3. Store results with descriptive naming
        new_modality = f"{modality_name}_analyzed"
        data_manager.modalities[new_modality] = result_adata

        # 4. Log operation for provenance
        data_manager.log_tool_usage("analyze_modality", params, stats)

        return formatted_response(stats, new_modality)

    except ServiceError as e:
        logger.error(f"Service error: {e}")
        return f"Analysis failed: {str(e)}"
```

### Agent Registry Pattern
```python
# lobster/config/agent_registry.py
@dataclass
class AgentConfig:
    name: str                          # Unique identifier
    display_name: str                  # Human-readable name
    description: str                   # Agent capabilities
    factory_function: str             # Module path to factory
    handoff_tool_name: Optional[str] # Auto-generated tool name
    handoff_tool_description: Optional[str]

# Adding new agents only requires registry entry:
AGENT_REGISTRY = {
    'new_agent': AgentConfig(
        name='new_agent',
        display_name='New Agent',
        description='Agent purpose',
        factory_function='lobster.agents.new_agent.new_agent',
        handoff_tool_name='handoff_to_new_agent',
        handoff_tool_description='Task description'
    )
}
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

## Critical Rules & Architectural Patterns

### Development Rules
- **NEVER** modify `pyproject.toml` - all installations requested through user
- Always prefer editing existing files over creating new ones
- Maintain backward compatibility when updating CLI
- Test autocomplete with both local and cloud clients
- Follow scientific accuracy standards for bioinformatics algorithms
- Use the centralized agent registry for new agents (no manual graph.py edits)
- Maintain stateless service design (services work with AnnData, return tuples)
- Follow the professional naming convention for modalities

### Architectural Patterns to Maintain

1. **Service Pattern**: Stateless, returns `(processed_adata, statistics_dict)`
2. **Tool Pattern**: Validates modality → calls service → stores result → logs provenance
3. **Error Hierarchy**: Use specific exceptions (ModalityNotFoundError, ServiceError, etc.)
4. **Registry Pattern**: Single source of truth for agent configuration
5. **Adapter Pattern**: Unified interfaces for different data types/clients

### Code Deduplication Principles
- Use `ConcatenationService` for all sample merging (no duplication)
- Delegate to services rather than implementing in agents
- Reuse validation logic through shared utilities
- Centralize configuration in registry and settings

### Data Quality Standards
- Maintain W3C-PROV compliant provenance tracking
- Enforce schema validation for all data types
- Include comprehensive QC metrics at each step
- Support batch effect detection and correction
- Implement proper missing value handling strategies

## Common Troubleshooting & Connectivity

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

### Component Connectivity Map

```
CLI (lobster/cli.py)
├── LobsterClientAdapter → BaseClient implementations
│   ├── AgentClient → LangGraph → Agent Registry → All Agents
│   └── CloudLobsterClient → HTTP API (external)
│
Agents (lobster/agents/)
├── Use DataManagerV2 for modality management
├── Call Services for analysis (stateless)
└── Return formatted responses to CLI
│
Services (lobster/tools/)
├── Receive AnnData objects
├── Process with scientific algorithms
└── Return (processed_adata, statistics)
│
DataManagerV2 (lobster/core/data_manager_v2.py)
├── Manages named modalities (Dict[str, AnnData])
├── Tracks provenance and tool usage
├── Validates schemas (transcriptomics/proteomics)
└── Delegates to backends (H5AD, MuData)
```

### Testing Connectivity
- Agent Registry: `python tests/test_agent_registry.py`
- Service Integration: `pytest tests/integration/`
- CLI Commands: Test with both `AgentClient` and mock `CloudLobsterClient`
- Data Flow: Verify modality naming convention maintained throughout pipeline