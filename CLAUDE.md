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
  - `ms_proteomics_expert.py` - Mass spectrometry proteomics with DDA/DIA workflows and missing value handling
  - `affinity_proteomics_expert.py` - Affinity proteomics including Olink panels and antibody arrays
  - `data_expert.py` - Data loading, format conversion, quality assessment
  - `research_agent.py` - Literature mining and dataset discovery
  - `method_expert.py` - Parameter optimization from publications
  - `machine_learning_expert.py` - Advanced ML workflows
  - `supervisor.py` - Agent coordination and workflow management
  - `langgraph_supervisor/` - LangGraph supervisor implementation
    - `supervisor.py` - Core supervisor orchestration logic
    - `handoff.py` - Agent handoff coordination
    - `agent_name.py` - Agent identification and naming

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
  - **Visualization Services** - Comprehensive interactive visualization suite
    - `visualization_service.py` - SingleCellVisualizationService with publication-quality plotting:
      - `create_umap_plot()` - Interactive UMAP with customizable coloring, auto-scaling, and hover data
      - `create_pca_plot()` - PCA plots with variance explained and component selection
      - `create_elbow_plot()` - PCA variance elbow plots with dual y-axes
      - `create_violin_plot()` - Gene expression violin plots across groups with statistical overlays
      - `create_feature_plot()` - Gene expression feature plots on UMAP with multi-gene support
      - `create_dot_plot()` - Marker gene dot plots with expression intensity and percentage
      - `create_heatmap()` - Gene expression heatmaps with hierarchical clustering and z-score normalization
      - `create_qc_plots()` - Comprehensive 16-panel QC report with statistical thresholds and quality metrics
      - `create_cluster_composition_plot()` - Cluster composition analysis with batch effect visualization
      - `save_all_plots()` - Batch export to HTML/PNG with high-resolution options
  - `quality_service.py` - Quality control metrics and filtering
  - **Proteomics Services** - Professional-grade proteomics analysis
    - `proteomics_preprocessing_service.py` - MS/Affinity filtering & normalization
    - `proteomics_quality_service.py` - Missing value & CV analysis
    - `proteomics_analysis_service.py` - Statistical testing & PCA
    - `proteomics_differential_service.py` - Differential expression & pathways
    - `proteomics_visualization_service.py` - Volcano plots & networks
  - **Publication Services** - Literature mining and dataset discovery
    - `publication_service.py` - Multi-provider orchestrator
    - `providers/pubmed_provider.py` - PubMed literature search
    - `providers/geo_provider.py` - Direct GEO DataSets search with advanced filtering

- **`lobster/config/`** - Configuration management
  - `agent_config.py` - Agent configuration and LLM settings
  - `agent_registry.py` - Centralized agent registry system (single source of truth)
  - `settings.py` - Global application settings
  - `config_manager.py` - Environment and API key management

### Key Design Principles

1. **Agent-Based Architecture** - Each specialist agent handles specific analysis domains with centralized registry
2. **Modular Service Architecture** - Stateless analysis services for professional-grade bioinformatics workflows
3. **Unified Data Management** - Single data manager coordinates all modalities (AnnData, MuData)
4. **Professional Proteomics Support** - Specialized MS and affinity proteomics with missing value handling
5. **Publication-Quality Visualization** - Interactive Plotly-based plots with scientific accuracy and statistical rigor
6. **Natural Language Interface** - Users describe analyses in plain English
7. **Reproducibility** - Complete provenance tracking of all operations
8. **Cloud Integration** - Seamless local/cloud execution with automatic detection and fallback

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
      "proteomics_mass_spec": "Mass spectrometry proteomics with missing value handling",
      "proteomics_affinity": "Antibody array proteomics (Olink panels, targeted assays)"
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

### Modular Service Architecture

The system employs a clean separation between agent orchestration and analysis implementation:

#### Stateless Analysis Services

**Transcriptomics Services:**
- `PreprocessingService` - Quality-based filtering and normalization
- `QualityService` - Comprehensive QC assessment with statistical metrics
- `ClusteringService` - Leiden clustering, PCA, UMAP visualization
- `EnhancedSingleCellService` - Doublet detection and cell type annotation

**Visualization Services (Publication-Grade):**
- `SingleCellVisualizationService` - Interactive Plotly-based visualization suite
  - Multi-panel QC reports with statistical thresholds and automated outlier detection
  - Dimensionality reduction plots (UMAP/PCA) with intelligent scaling and coloring
  - Gene expression visualizations (violin, feature, dot plots, heatmaps)
  - Cluster analysis and composition plots with batch effect detection
  - Professional export pipeline for publication-quality figures (HTML/PNG)

**Proteomics Services (Professional Grade):**
- `ProteomicsPreprocessingService` - MS/Affinity filtering with platform-appropriate normalization
- `ProteomicsQualityService` - Missing value analysis and coefficient of variation assessment
- `ProteomicsAnalysisService` - Statistical testing and dimensionality reduction
- `ProteomicsDifferentialService` - Advanced differential expression with FDR control
- `ProteomicsVisualizationService` - Publication-ready volcano plots and protein networks

**Publication Services:**
- `PublicationService` - Multi-provider orchestrator for literature mining
- `PubMedProvider` - PubMed literature search and method extraction
- `GEOProvider` - Direct GEO DataSets search with advanced filtering capabilities

#### Agent-Service Integration Pattern

Agents focus on orchestration and user interaction while delegating analysis to specialized services:

```python
# Standard agent tool pattern
@tool
def analyze_modality(modality_name: str, **params) -> str:
    # 1. Validate modality exists and get data
    adata = data_manager.get_modality(modality_name)
    
    # 2. Call stateless service for analysis
    result_adata, stats = service.analyze_method(adata, **params)
    
    # 3. Store results with descriptive naming
    new_modality = f"{modality_name}_analyzed"
    data_manager.modalities[new_modality] = result_adata
    
    # 4. Log operation for provenance
    data_manager.log_tool_usage("analyze_modality", params, stats)
    
    return formatted_response(stats, new_modality)
```

### Advanced Visualization Architecture

Lobster AI features a sophisticated visualization system built on Plotly for interactive, publication-quality scientific visualizations with professional color schemes and statistical accuracy.

#### SingleCellVisualizationService

The core visualization service provides comprehensive plotting capabilities for single-cell RNA-seq analysis with scientific rigor and publication standards.

```python
# lobster/tools/visualization_service.py
class SingleCellVisualizationService:
    """Professional visualization service for single-cell RNA-seq data."""

    # Sophisticated color palettes for different data types
    cluster_colors = px.colors.qualitative.Set1          # Categorical data
    continuous_colors = px.colors.sequential.Viridis     # Gene expression
    diverging_colors = px.colors.diverging.RdBu_r       # Differential expression
    expression_colorscale = [                            # Scientific gene expression
        [0, 'lightgray'], [0.01, 'lightblue'], [0.1, 'blue'],
        [0.5, 'red'], [0.8, 'darkred'], [1.0, 'black']
    ]
```

#### Interactive Dimensionality Reduction Visualizations

**UMAP and PCA Plots** with advanced features:
- Automatic point size scaling based on dataset size
- Multi-modal coloring (categorical clusters, continuous gene expression)
- Interactive hover data with cell metadata
- Logarithmic scaling for count data
- Professional axis styling and legends

```python
# Dynamic visualization with intelligent defaults
fig = service.create_umap_plot(
    adata,
    color_by="leiden",           # Cluster coloring
    point_size=None,             # Auto-scaled: 8→5→3→2 based on cell count
    alpha=0.8,                   # Optimal transparency
    show_legend=True             # Professional legend positioning
)

# Gene expression overlay with scientific color mapping
fig = service.create_umap_plot(adata, color_by="CD4", alpha=0.6)
```

#### Comprehensive Quality Control Suite

**16-Panel Professional QC Report** with statistical rigor:

```python
# Creates publication-ready multi-panel QC figure
qc_fig = service.create_qc_plots(adata)

# Comprehensive panels include:
# A. Transcriptional Complexity (UMI vs Genes with MT% coloring)
# B. Mitochondrial QC (UMI vs MT% with outlier detection)
# C. Ribosomal Content Analysis
# D-F. Statistical Distribution Analysis (violin + box plots)
# G. Library Saturation Curves
# H. Cell Quality Classification (pie chart)
# I-L. Advanced Metrics (detection rates, correlations, summaries)
# M-P. Specialized Analysis (doublets, thresholds, filtering impact)
```

**Automated Statistical Thresholds**:
- MAD-based outlier detection (Median Absolute Deviation)
- Data-driven axis scaling with intelligent padding
- Batch effect detection and visualization
- Quality score computation with traffic light indicators

#### Gene Expression Visualizations

**Multi-Modal Expression Analysis**:

```python
# Violin plots with statistical overlays
violin_fig = service.create_violin_plot(
    adata,
    genes=["CD4", "CD8A", "IL2RA"],
    groupby="leiden",
    log_scale=True,              # Automatic log transformation
    use_raw=True                 # Raw vs normalized data selection
)

# Feature plots with multi-gene support
feature_fig = service.create_feature_plot(
    adata,
    genes=["FOXP3", "IL2RA", "CTLA4", "CD25"],
    ncols=2,                     # Automatic subplot layout
    vmin=0, vmax=5              # Custom expression ranges
)

# Professional dot plots with dual encoding
dot_fig = service.create_dot_plot(
    adata,
    genes=marker_genes,
    groupby="leiden",
    standard_scale="var"         # Statistical normalization options
)
```

#### Publication-Quality Heatmaps

**Hierarchical Expression Analysis**:
- Automatic marker gene selection from scanpy results
- Z-score normalization with statistical validation
- Professional color schemes (diverging for normalized, sequential for raw)
- Intelligent gene filtering and duplicate removal

```python
# Auto-generated from marker gene analysis
heatmap_fig = service.create_heatmap(
    adata,
    genes=None,                  # Auto-selects top markers per cluster
    n_top_genes=5,              # Configurable marker count
    standard_scale=True,         # Z-score normalization
    use_raw=True                # Data source selection
)
```

#### Advanced Cluster Analysis

**Composition and Batch Effect Analysis**:

```python
# Sophisticated cluster composition with batch correction
composition_fig = service.create_cluster_composition_plot(
    adata,
    cluster_col="leiden",
    sample_col=None,             # Auto-detects batch columns
    normalize=True               # Percentage vs absolute counts
)
```

#### Professional Export System

**Multi-Format Publication Pipeline**:

```python
# Batch export with publication settings
plots_dict = {
    "umap_clusters": umap_fig,
    "qc_comprehensive": qc_fig,
    "marker_heatmap": heatmap_fig,
    "expression_violins": violin_fig
}

saved_files = service.save_all_plots(
    plots_dict,
    output_dir="publication_figures/",
    format="both"                # HTML (interactive) + PNG (high-res)
)

# PNG exports: 3200x2400px, scale=2 for publication quality
# HTML exports: Full interactivity with hover data and zoom
```

#### Visualization Design Principles

**Scientific Accuracy**:
- Color-blind friendly palettes with sufficient contrast
- Statistically appropriate scaling (log for count data, linear for percentages)
- Professional typography and axis labeling
- Consistent styling across all plot types

**Performance Optimization**:
- Intelligent point sampling for large datasets (>50K cells)
- ScatterGL for high-performance rendering
- Automatic memory management for sparse matrices
- Progressive loading for complex multi-panel figures

**Interactive Features**:
- Rich hover tooltips with cell metadata
- Zoom and pan capabilities maintained across all plots
- Legend positioning optimized for different screen sizes
- Export-ready formatting for both web and print media

### Centralized Agent Registry System

The system features a centralized agent registry that serves as the single source of truth for all agent configurations, eliminating redundancy and reducing errors.

#### Agent Registry Architecture

```python
# lobster/config/agent_registry.py
@dataclass
class AgentConfig:
    name: str                              # Unique agent identifier
    display_name: str                     # Human-readable name
    description: str                      # Agent's purpose/capability
    factory_function: str                 # Module path to factory function
    handoff_tool_name: Optional[str]     # Name of handoff tool
    handoff_tool_description: Optional[str]  # Tool description
```

#### Current Agent Registry

```python
AGENT_REGISTRY: Dict[str, AgentConfig] = {
    'data_expert_agent': AgentConfig(...),
    'singlecell_expert_agent': AgentConfig(...),
    'bulk_rnaseq_expert_agent': AgentConfig(...),
    'research_agent': AgentConfig(...),
    'method_expert_agent': AgentConfig(...),
    'ms_proteomics_expert_agent': AgentConfig(
        name='ms_proteomics_expert_agent',
        display_name='MS Proteomics Expert',
        description='Handles mass spectrometry proteomics data analysis including DDA/DIA workflows',
        factory_function='lobster.agents.ms_proteomics_expert.ms_proteomics_expert'
    ),
    'affinity_proteomics_expert_agent': AgentConfig(
        name='affinity_proteomics_expert_agent',
        display_name='Affinity Proteomics Expert',
        description='Handles affinity proteomics including Olink and targeted protein panels',
        factory_function='lobster.agents.affinity_proteomics_expert.affinity_proteomics_expert'
    ),
}
```

#### Benefits of Centralized Registry

**Before (Legacy System):**
```
Adding new agents required updating:
├── lobster/agents/graph.py          # Import statements
├── lobster/agents/graph.py          # Agent creation code
├── lobster/agents/graph.py          # Handoff tool definitions
├── lobster/utils/callbacks.py       # Agent name hardcoded list
└── Multiple imports throughout codebase
```

**After (Registry System):**
```
Adding new agents only requires:
└── lobster/config/agent_registry.py  # Single registry entry

Everything else is handled automatically:
├── ✅ Dynamic agent loading
├── ✅ Automatic handoff tool creation
├── ✅ Callback system integration
├── ✅ Type-safe configuration
└── ✅ Professional error handling
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

## Comprehensive Testing Framework

Lobster AI includes a robust testing infrastructure targeting 95%+ code coverage with scientifically accurate testing scenarios.

### Test Structure

```
tests/
├── 📁 unit/                    # Unit tests (13 files)
│   ├── core/                  # Core system components
│   ├── agents/                # AI agent functionality
│   └── tools/                 # Analysis services
├── 📁 integration/            # Integration tests (5 files)
│   ├── test_agent_workflows.py
│   ├── test_data_pipelines.py
│   ├── test_cloud_local_switching.py
│   ├── test_geo_download_workflows.py
│   └── test_multi_omics_integration.py
├── 📁 system/                 # System tests (3 files)
├── 📁 performance/            # Performance tests (3 files)
├── 🚀 run_integration_tests.py # Enhanced test runner
└── 📋 test_cases.json         # Test case definitions
```

### Test Categories & Runtime

| **Test Type** | **Purpose** | **Coverage** | **Runtime** |
|---------------|-------------|--------------|-------------|
| **Unit Tests** | Core component validation | Individual functions/classes | < 2 minutes |
| **Integration Tests** | Multi-component workflows | Agent interactions, data pipelines | < 15 minutes |
| **System Tests** | End-to-end scenarios | Complete analysis workflows | < 30 minutes |
| **Performance Tests** | Benchmarking & scalability | Large datasets, concurrent execution | < 45 minutes |

### Quick Testing Commands

```bash
# Run all tests with coverage
make test

# Fast parallel execution
make test-fast

# Run specific categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/system/        # System tests
pytest tests/performance/   # Performance tests

# Run by biological focus
pytest -m "singlecell"      # Single-cell RNA-seq tests
pytest -m "proteomics"      # Proteomics analysis tests
pytest -m "multiomics"      # Multi-omics integration tests
pytest -m "geo"             # GEO database integration tests

# Enhanced test runner with performance monitoring
python tests/run_integration_tests.py --categories basic,advanced --performance-monitoring
```

### Mock Data Framework

Sophisticated synthetic data generation with realistic biological features:

```python
from tests.mock_data.factories import SingleCellDataFactory, ProteomicsDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG

# Generate single-cell RNA-seq data
sc_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

# Generate proteomics data with missing values
proteomics_data = ProteomicsDataFactory(config={
    'n_samples': 48,
    'n_proteins': 2000,
    'missing_value_rate': 0.15
})
```

**Realistic Biological Features:**
- Cell type labels and gene categorization
- Mitochondrial/ribosomal gene patterns
- Batch effects and technical variation
- Proteomics-appropriate missingness patterns
- Spatial and temporal data structures

### CI/CD Integration

**Quality Gates (every PR must pass):**
- ✅ Code Formatting (Black, isort)
- ✅ Linting (Flake8 with bioinformatics-specific rules)
- ✅ Type Checking (MyPy static analysis)
- ✅ Security (Bandit security linting)
- ✅ Unit Tests (>80% coverage required)
- ✅ Integration Tests (critical workflow validation)
- ✅ Performance (no significant regressions)

**Multi-Platform Testing:**
- Ubuntu, macOS, Windows
- Python 3.11, 3.12
- Automated dependency updates and vulnerability scanning

### Coverage Requirements
- **Minimum Coverage**: 80% (targeting 95%+)
- **Test Execution Time**: < 2 minutes for unit tests, < 45 minutes total
- **Biological Accuracy**: Scientifically validated mock data and algorithms
- **Error Recovery**: Comprehensive fault tolerance testing

## Cloud Integration Architecture

Lobster AI features sophisticated cloud/local architecture with seamless mode switching and automatic fallback capabilities.

### Smart Client Detection

The system automatically detects your configuration and routes requests appropriately:

```python
# Automatic client initialization
def init_client():
    if os.getenv('LOBSTER_CLOUD_KEY'):
        try:
            # Attempt cloud client initialization
            cloud_client = CloudLobsterClient(api_key=cloud_key)
            if cloud_client.test_connection():
                return cloud_client  # ☁️ Cloud mode active
        except ConnectionError:
            pass  # Fallback to local
    
    # 💻 Local mode (default or fallback)
    return AgentClient()
```

### Execution Modes

#### ☁️ **Cloud Mode** (when `LOBSTER_CLOUD_KEY` is set)
- **CloudLobsterClient**: HTTP REST API communication with cloud services
- **Scalable Computing**: Handle large datasets without local hardware limits
- **Automatic Fallback**: Falls back to local mode if cloud unavailable
- **Same Interface**: Identical user experience through BaseClient abstraction

#### 💻 **Local Mode** (default or fallback)
- **AgentClient**: Complete local LangGraph processing
- **Full Functionality**: All analysis capabilities included
- **Privacy**: Data never leaves your computer
- **Offline Capable**: Works without internet connection

### BaseClient Interface Pattern

All clients implement a unified interface ensuring consistency:

```python
class BaseClient(ABC):
    @abstractmethod
    def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]
    @abstractmethod
    def get_status(self) -> Dict[str, Any]
    @abstractmethod
    def export_session(self, export_path: Optional[Path] = None) -> Path
```

### Seamless Mode Switching Flow

```bash
# User runs same command regardless of mode
lobster chat

# System automatically:
1. Checks for LOBSTER_CLOUD_KEY
2. Tests cloud connection if key present
3. Falls back to local if cloud unavailable
4. Provides identical interface in both modes
```

### Configuration

```bash
# Cloud mode activation
export LOBSTER_CLOUD_KEY=your-api-key

# System status indicators
🌩️ "Cloud mode active"           # Cloud connection successful
💻 "Using local mode"            # No cloud key set
💻 "Using local mode (cloud unavailable)"  # Fallback occurred
```

This architecture ensures users have a consistent, reliable experience whether running locally or in the cloud.

## Advanced Dataset Discovery

### GEO Provider with Direct Database Search

The system includes a sophisticated GEO provider that enables direct search of NCBI's Gene Expression Omnibus (GEO) DataSets database with advanced filtering capabilities.

#### Key Features

- **Direct GEO DataSets Search**: Search the GEO database directly without going through PubMed
- **Advanced Filtering**: Support for organisms, platforms, entry types, date ranges, and supplementary file types
- **Official API Compliance**: Implements all query patterns from NCBI documentation
- **WebEnv/QueryKey Support**: Efficient result pagination using NCBI's history server

#### Usage Examples

```python
# Simple search
results = service.search_datasets_directly(
    query="single-cell RNA-seq",
    data_type="geo",
    max_results=10
)

# Advanced filtering
filters = {
    "organisms": ["human", "mouse"],
    "entry_types": ["gse"],  # Series only
    "published_last_n_months": 6,
    "supplementary_file_types": ["h5", "h5ad"],
    "max_results": 20
}

results = service.search_datasets_directly(
    query="ATAC-seq chromatin accessibility",
    data_type="geo",
    filters=filters
)
```

#### Research Agent Integration

```bash
# Natural language dataset discovery
🦞 You: "Find recent human single-cell datasets with H5 files"

🦞 Lobster: Searching GEO database with advanced filters...
✓ Found 15 datasets matching criteria
✓ Filtered by: human organism, GSE series, H5 supplementary files
✓ Published in last 3 months
```

### Publication Service Architecture

Multi-provider orchestrator for comprehensive literature mining:

- **PubMedProvider**: Traditional literature search and method extraction
- **GEOProvider**: Direct dataset discovery with biological context
- **Unified Interface**: Consistent API across all providers
- **Extensible Design**: Easy addition of new data sources

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