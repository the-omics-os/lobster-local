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

# Run integration tests  
make test-integration

# Code formatting (black + isort)
make format

# Linting (flake8, pylint, bandit)
make lint

# Type checking
make type-check
```

### Running the Application
```bash
# Start interactive chat mode
lobster chat

# Show help
lobster --help

# Alternative module execution
python -m lobster
```

## Architecture Overview

The codebase follows a modular, agent-based architecture:

### Core Components

- **`lobster/agents/`** - Specialized AI agents for different analysis types
  - `singlecell_expert.py` - Single-cell RNA-seq analysis
  - `bulk_rnaseq_expert.py` - Bulk RNA-seq analysis  
  - `proteomics_expert.py` - Proteomics and mass spectrometry
  - `data_expert.py` - Data loading, format conversion, quality assessment
  - `research_agent.py` - Literature mining and dataset discovery
  - `method_expert.py` - Parameter optimization from publications
  - `machine_learning_expert.py` - Advanced ML workflows
  - `supervisor.py` - Agent coordination and workflow management

- **`lobster/core/`** - Data management and client infrastructure
  - **Client Architecture**:
    - `client.py` - AgentClient (local LangGraph processing)
    - `api_client.py` - APIAgentClient (WebSocket streaming for web services)
    - `interfaces/base_client.py` - BaseClient interface for cloud/local consistency
  - **Data Management**:
    - `data_manager_v2.py` - DataManagerV2 (modular multi-omics orchestrator)
    - `data_manager_old.py` - Legacy DataManager (deprecated)
    - `provenance.py` - ProvenanceTracker (analysis history and reproducibility)
  - **Modular Architecture**:
    - `adapters/` - Modality-specific data adapters
    - `backends/` - Storage backend implementations
    - `schemas/` - Data validation and schema enforcement
    - `interfaces/` - Abstract interfaces for extensible architecture
  - **WebSocket Integration**:
    - `websocket_callback.py` - Real-time streaming callback manager
    - `websocket_logging_handler.py` - WebSocket logging integration

- **`lobster/tools/`** - Analysis services and bioinformatics tools
  - `geo_service.py` - GEO database integration and downloading
  - `preprocessing_service.py` - Data preprocessing and normalization
  - `clustering_service.py` - Clustering algorithms and cell type annotation
  - `visualization_service.py` - Publication-ready plot generation
  - `quality_service.py` - Quality control metrics and filtering

- **`lobster/config/`** - Configuration management
  - `agent_config.py` - Agent configuration and LLM settings
  - `settings.py` - Global application settings
  - `config_manager.py` - Environment and API key management

### Key Design Principles

1. **Agent-Based Architecture** - Each specialist agent handles specific analysis domains
2. **Unified Data Management** - Single data manager coordinates all modalities (AnnData, MuData)
3. **Natural Language Interface** - Users describe analyses in plain English
4. **Reproducibility** - Complete provenance tracking of all operations
5. **Cloud Integration** - Seamless local/cloud execution with automatic detection

## Core Architecture Deep Dive

### Client System Architecture

The `lobster/core/` module implements a sophisticated client architecture that provides seamless switching between local and cloud execution modes:

#### BaseClient Interface Pattern

```python
# lobster/core/interfaces/base_client.py
class BaseClient(ABC):
    """Abstract interface ensuring consistency between local and cloud clients."""
    
    @abstractmethod
    def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]
    @abstractmethod
    def get_status(self) -> Dict[str, Any]
    @abstractmethod
    def export_session(self, export_path: Optional[Path] = None) -> Path
    # ... other unified methods
```

#### Client Implementations

1. **AgentClient** (`client.py`) - Local LangGraph Processing
   - Manages LangGraph multi-agent workflows locally
   - Uses DataManagerV2 for modality orchestration
   - Provides rich console output and debugging capabilities
   - Supports custom callbacks and model parameter overrides

2. **APIAgentClient** (`api_client.py`) - WebSocket Streaming
   - Designed for FastAPI web service integration
   - Real-time streaming through WebSocket connections
   - Session-based operation with UUID tracking
   - Specialized for web UI and API consumption

3. **CloudLobsterClient** (external) - Cloud Integration
   - HTTP REST API communication with cloud services
   - Fallback to local AgentClient when cloud unavailable
   - Unified interface through BaseClient abstraction

### DataManagerV2 Architecture

#### Modular Orchestration System

```json
{
  "DataManagerV2": {
    "core_storage": {
      "modalities": "Dict[str, AnnData] - Named biological datasets",
      "metadata_store": "Dict[str, Dict] - GEO and source metadata",
      "tool_usage_history": "List[Dict] - Complete operation log",
      "latest_plots": "List[Dict] - Generated visualizations"
    },
    "backend_registry": {
      "h5ad": "H5ADBackend - Single modality storage",
      "mudata": "MuDataBackend - Multi-modal integrated storage"
    },
    "adapter_registry": {
      "transcriptomics_single_cell": "Single-cell RNA-seq processing",
      "transcriptomics_bulk": "Bulk RNA-seq processing", 
      "proteomics_mass_spec": "Mass spectrometry proteomics",
      "proteomics_affinity": "Antibody array proteomics"
    },
    "workspace_structure": {
      "data/": "Persistent modality storage",
      "exports/": "Analysis outputs and visualizations",
      "cache/": "Temporary processing files"
    }
  }
}
```

#### Modality Management Pattern

```python
# Professional naming convention for analysis workflows
geo_gse12345                           # Raw downloaded data
├── geo_gse12345_quality_assessed     # With QC metrics  
├── geo_gse12345_filtered_normalized  # Preprocessed data
├── geo_gse12345_doublets_detected    # Doublet annotations
├── geo_gse12345_clustered           # Clustering results
├── geo_gse12345_markers            # Marker genes
└── geo_gse12345_annotated         # Cell type annotations
```

### Interface-Driven Modular Architecture

#### IModalityAdapter Interface

```python
# lobster/core/interfaces/adapter.py
class IModalityAdapter(ABC):
    """Contract for modality-specific data processing."""
    
    @abstractmethod
    def from_source(self, source: Union[str, Path, pd.DataFrame]) -> anndata.AnnData
    @abstractmethod
    def validate(self, adata: anndata.AnnData, strict: bool = False) -> ValidationResult
    @abstractmethod  
    def get_schema(self) -> Dict[str, Any]
    @abstractmethod
    def get_supported_formats(self) -> List[str]
```

**Current Adapter Implementations:**
- `TranscriptomicsAdapter` - RNA-seq data processing with schema validation
- `ProteomicsAdapter` - Protein data processing with missing value handling

#### IDataBackend Interface

```python
# lobster/core/interfaces/backend.py  
class IDataBackend(ABC):
    """Contract for storage backend implementations."""
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> anndata.AnnData
    @abstractmethod
    def save(self, adata: anndata.AnnData, path: Union[str, Path]) -> None
    @abstractmethod
    def exists(self, path: Union[str, Path]) -> bool
```

**Current Backend Implementations:**
- `H5ADBackend` - Single modality HDF5 storage (S3-ready)
- `MuDataBackend` - Multi-modal integrated storage

#### IValidator Interface

```python
# lobster/core/interfaces/validator.py
class IValidator(ABC):
    """Contract for data validation implementations."""
    
    @abstractmethod
    def validate(self, adata: anndata.AnnData) -> ValidationResult
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]
```

### Schema System

#### Modality-Specific Schemas

**Transcriptomics Schema** (`schemas/transcriptomics.py`):
```json
{
  "required_obs": ["total_counts", "n_genes_by_counts"],
  "optional_obs": ["pct_counts_mt", "pct_counts_ribo", "doublet_score"],
  "required_var": ["gene_ids", "feature_types"],
  "layers": ["counts", "logcounts"],
  "obsm": ["X_pca", "X_umap", "X_tsne"],
  "uns": ["hvg", "pca", "neighbors", "leiden"]
}
```

**Proteomics Schema** (`schemas/proteomics.py`):
```json
{
  "required_obs": ["sample_id", "total_protein_intensity"],  
  "optional_obs": ["batch", "condition", "missing_value_pct"],
  "required_var": ["protein_ids", "protein_names"],
  "layers": ["raw_intensity", "normalized_intensity"],
  "uns": ["normalization_params", "imputation_method"]
}
```

#### Flexible Validation System

```python
# Warning-based validation (default)
validation_result = adapter.validate(adata, strict=False)
if validation_result.warnings:
    logger.warning(f"Schema warnings: {validation_result.warnings}")

# Strict validation (raises exceptions)  
try:
    validation_result = adapter.validate(adata, strict=True)
except ValueError as e:
    logger.error(f"Schema validation failed: {e}")
```

### Provenance System

#### Complete Analysis Tracking

```python
# lobster/core/provenance.py
class ProvenanceTracker:
    """Tracks complete analysis history for reproducibility."""
    
    def log_operation(self, operation: str, parameters: Dict, results: Dict):
        """Log analysis operation with parameters and outcomes."""
        
    def get_analysis_history(self) -> List[Dict]:
        """Retrieve complete chronological analysis history."""
        
    def export_workflow(self, format: str = "json") -> str:
        """Export reproducible workflow specification."""
```

#### Provenance Data Structure

```json
{
  "operation_id": "uuid4_string",
  "timestamp": "2024-09-06T10:30:00.000Z",
  "operation": "filter_and_normalize_modality", 
  "agent": "singlecell_expert_agent",
  "parameters": {
    "modality_name": "geo_gse12345",
    "min_genes": 200,
    "min_cells": 3,
    "normalize_total": 10000
  },
  "inputs": {
    "n_obs": 5000,
    "n_vars": 20000,
    "modality": "geo_gse12345"
  },
  "outputs": {
    "n_obs": 4500,
    "n_vars": 18500, 
    "new_modality": "geo_gse12345_filtered_normalized"
  },
  "execution_time": 45.2,
  "success": true
}
```

### WebSocket Integration

#### Real-Time Streaming Architecture

```python
# lobster/core/websocket_callback.py
class APICallbackManager:
    """Manages real-time streaming of agent operations."""
    
    async def broadcast_agent_transition(self, from_agent: str, to_agent: str):
        """Stream agent handoffs in real-time."""
        
    async def broadcast_tool_execution(self, tool_name: str, status: str):
        """Stream tool execution progress."""
        
    async def broadcast_plot_generation(self, plot_data: Dict):
        """Stream generated visualizations."""
```

### Error Handling & Exception Hierarchy

#### Professional Error Management

```python
# Custom exception hierarchy for precise error handling
class LobsterCoreError(Exception):
    """Base exception for lobster core operations."""

class ModalityNotFoundError(LobsterCoreError):
    """Raised when requested modality doesn't exist."""

class ValidationError(LobsterCoreError):
    """Raised when data validation fails."""
    
class BackendError(LobsterCoreError):
    """Raised when storage backend operations fail."""
    
class AdapterError(LobsterCoreError):  
    """Raised when modality adapter operations fail."""
```

### Usage Patterns

#### Agent Tool Integration Pattern

```python
@tool
def analyze_modality(modality_name: str, **params) -> str:
    """Standard pattern for agent tools."""
    try:
        # 1. Validate modality exists
        if modality_name not in data_manager.list_modalities():
            raise ModalityNotFoundError(f"Modality '{modality_name}' not found")
            
        # 2. Get data through adapter
        adata = data_manager.get_modality(modality_name)
        
        # 3. Call stateless analysis service
        result_adata, stats = service.analyze(adata, **params)
        
        # 4. Store results with descriptive naming
        new_modality = f"{modality_name}_analyzed"
        data_manager.modalities[new_modality] = result_adata
        
        # 5. Log operation for provenance
        data_manager.log_tool_usage("analyze_modality", params, stats)
        
        return f"Analysis complete. Results stored in: {new_modality}"
        
    except LobsterCoreError as e:
        logger.error(f"Core error: {e}")
        return f"Analysis failed: {str(e)}"
```

#### DataManagerV2 Initialization Pattern

```python  
# Standard initialization with full capabilities
data_manager = DataManagerV2(
    default_backend="h5ad",
    workspace_path="/path/to/.lobster_workspace", 
    enable_provenance=True,
    console=rich_console_instance
)

# Register custom adapters
data_manager.register_adapter("custom_modality", CustomAdapter())

# Register custom backends  
data_manager.register_backend("s3", S3Backend(bucket="my-bucket"))
```

## Development Guidelines

### Code Style and Quality
- Follow PEP 8 Python style guidelines
- Use type hints for all functions and methods
- Line length: 88 characters (Black formatting)
- Add comprehensive docstrings to all public functions
- Prioritize scientific accuracy over performance optimizations

### Testing Requirements
- Add tests for new functionality in `tests/` directory
- Ensure existing tests pass before committing
- Test with real bioinformatics data when possible
- Include edge cases and error conditions
- Use pytest for test execution

### Bioinformatics Standards
- Include literature references for novel methods
- Follow established bioinformatics conventions (e.g., AnnData for single-cell)
- Consider reproducibility in all analyses
- Validate scientific accuracy of algorithms

### Working with Core Components

#### Adding New Modality Adapters
When supporting new biological data types:

1. **Create Adapter Class**:
   ```python
   # lobster/core/adapters/new_modality_adapter.py
   class NewModalityAdapter(IModalityAdapter):
       def from_source(self, source, **kwargs) -> anndata.AnnData:
           # Convert source to standardized AnnData
           pass
           
       def validate(self, adata, strict=False) -> ValidationResult:
           # Validate against modality schema
           pass
   ```

2. **Define Schema**:
   ```python
   # lobster/core/schemas/new_modality.py
   NEW_MODALITY_SCHEMA = {
       "required_obs": ["sample_id", "condition"],
       "optional_obs": ["batch", "replicate"],
       "required_var": ["feature_id", "feature_name"]
   }
   ```

3. **Register in DataManagerV2**:
   ```python
   data_manager.register_adapter("new_modality", NewModalityAdapter())
   ```

#### Adding Storage Backends
When implementing new storage systems:

1. **Implement IDataBackend Interface**:
   ```python
   # lobster/core/backends/cloud_backend.py
   class CloudBackend(IDataBackend):
       def load(self, path: Union[str, Path]) -> anndata.AnnData:
           # Load from cloud storage
           pass
           
       def save(self, adata: anndata.AnnData, path: Union[str, Path]) -> None:
           # Save to cloud storage 
           pass
   ```

2. **Register Backend**:
   ```python
   data_manager.register_backend("cloud", CloudBackend(config))
   ```

#### Extending Client Implementations
When adding new client types:

1. **Inherit from BaseClient**:
   ```python
   # lobster/core/new_client.py
   class NewClient(BaseClient):
       def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]:
           # Custom query processing
           pass
   ```

2. **Implement Required Methods**: All abstract methods from BaseClient interface
3. **Integration**: Update initialization logic in CLI or API layers

### Adding New Analysis Methods
When implementing new analysis tools:

1. Create service in `lobster/tools/` (e.g., `new_analysis_service.py`)
2. Add agent tool integration in relevant expert agent
3. Implement error handling and result validation
4. Add comprehensive tests and documentation
5. Update data manager if new data types are introduced

### Pre-commit Hooks
The project uses pre-commit hooks for code quality:
- Trailing whitespace removal
- YAML/JSON validation
- Black code formatting
- isort import sorting
- Flake8 linting
- Bandit security checking

## Environment Configuration

### Required Environment Variables
```bash
# API Keys (required)
OPENAI_API_KEY=your-openai-api-key
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key  
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key

# Optional
NCBI_API_KEY=your-ncbi-api-key
LOBSTER_CLOUD_KEY=your-cloud-api-key
```

### Workspace Structure
- `.lobster_workspace/` - Local analysis workspace and results
- `data/` - Input datasets and cached files  
- `exports/` - Generated outputs and visualizations

## Testing Strategy

### Test Organization
```bash
# Run specific test categories
pytest tests/test_lobster.py -v                    # Core functionality
pytest tests/integration/ -v -m integration        # Integration tests
pytest tests/ -n auto -v                          # Parallel execution
```

### Coverage Requirements
- Maintain test coverage reports with `--cov=lobster`
- Generate HTML coverage reports for detailed analysis
- Focus on critical analysis pathways and data handling

## Cloud Integration

The system supports automatic cloud/local mode switching:
- **Cloud Mode** - Activated when `LOBSTER_CLOUD_KEY` is set
- **Local Mode** - Fallback when cloud is unavailable
- **Seamless Switching** - Same interface regardless of execution mode

## Docker Support

```bash
# Build Docker image
make docker-build

# Run containerized version
make docker-run

# Push to registry
make docker-push
```

## Performance Considerations

- Single-cell datasets can be memory-intensive (4GB+ RAM recommended)
- Use appropriate chunking for large datasets
- Leverage scanpy/AnnData optimizations for sparse matrices
- Consider computation complexity when adding new algorithms

## Common Troubleshooting

### Installation Issues
- Ensure Python 3.12+ is installed
- For macOS: Use `brew install python@3.12`
- For Ubuntu/Debian: Install `python3.12-venv` package
- Use `make clean-install` for fresh environment

### Runtime Issues
- Check API key configuration in `.env`
- Verify virtual environment activation
- Review workspace permissions in `.lobster_workspace/`
- Check memory availability for large datasets

## Release Process

```bash
# Version management
make bump-patch    # Bug fixes
make bump-minor    # New features  
make bump-major    # Breaking changes

# Create release artifacts
make release

# Publish to PyPI
make publish
```