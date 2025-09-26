# Core API Reference

## Overview

The Core API provides the foundational layer of the Lobster AI system, including data management, client interfaces, and system orchestration. This module handles multi-modal data storage, provenance tracking, and provides unified interfaces for both local and cloud execution.

## DataManagerV2

The `DataManagerV2` class is the central orchestrator for multi-omics data management with complete provenance tracking and modular backend support.

### Class Definition

```python
class DataManagerV2:
    """
    Modular data manager for multi-omics analysis.

    This class orchestrates modality adapters, storage backends,
    and validation to provide a unified interface for managing
    multi-modal biological data with complete provenance tracking.
    """
```

### Constructor

```python
def __init__(
    self,
    default_backend: str = "h5ad",
    workspace_path: Optional[Union[str, Path]] = None,
    enable_provenance: bool = True,
    console=None,
    auto_scan: bool = True
) -> None
```

**Parameters:**
- `default_backend` (str): Default storage backend to use ("h5ad", "mudata")
- `workspace_path` (Optional[Union[str, Path]]): Optional workspace directory for data storage
- `enable_provenance` (bool): Whether to enable provenance tracking
- `console`: Optional Rich console instance for progress tracking
- `auto_scan` (bool): Whether to automatically scan workspace for available datasets

### Core Data Management Methods

#### Loading and Saving Data

```python
def load_modality(
    self,
    name: str,
    source: Union[str, Path, pd.DataFrame, anndata.AnnData],
    adapter: str,
    validate: bool = True,
    **kwargs
) -> anndata.AnnData
```

Load data for a specific modality using the specified adapter.

**Parameters:**
- `name` (str): Name for the modality
- `source` (Union[str, Path, pd.DataFrame, anndata.AnnData]): Data source
- `adapter` (str): Name of adapter to use
- `validate` (bool): Whether to validate the loaded data
- `**kwargs`: Additional parameters passed to adapter

**Returns:**
- `anndata.AnnData`: Loaded and validated data

**Raises:**
- `ValueError`: If adapter is not registered or validation fails

```python
def save_modality(
    self,
    name: str,
    path: Union[str, Path],
    backend: Optional[str] = None,
    **kwargs
) -> str
```

Save a modality using specified backend.

**Parameters:**
- `name` (str): Name of modality to save
- `path` (Union[str, Path]): Destination path
- `backend` (Optional[str]): Backend to use (default: default_backend)
- `**kwargs`: Additional parameters passed to backend

**Returns:**
- `str`: Path where data was saved

#### Modality Management

```python
def get_modality(self, name: str) -> anndata.AnnData
```

Get a specific modality.

**Returns:**
- `anndata.AnnData`: The requested modality

**Raises:**
- `ValueError`: If modality not found

```python
def list_modalities(self) -> List[str]
```

List all loaded modalities.

**Returns:**
- `List[str]`: List of modality names

```python
def remove_modality(self, name: str) -> None
```

Remove a modality from memory.

**Raises:**
- `ValueError`: If modality not found

#### Multi-Modal Data Integration

```python
def to_mudata(self, modalities: Optional[List[str]] = None) -> Any
```

Convert modalities to MuData object.

**Parameters:**
- `modalities` (Optional[List[str]]): List of modality names to include (default: all)

**Returns:**
- `mudata.MuData`: MuData object containing specified modalities

**Raises:**
- `ImportError`: If MuData is not available
- `ValueError`: If no modalities are loaded

### Backend and Adapter Registration

```python
def register_backend(self, name: str, backend: IDataBackend) -> None
```

Register a storage backend.

**Parameters:**
- `name` (str): Name for the backend
- `backend` (IDataBackend): Backend implementation

**Raises:**
- `ValueError`: If backend name already exists

```python
def register_adapter(self, name: str, adapter: IModalityAdapter) -> None
```

Register a modality adapter.

**Parameters:**
- `name` (str): Name for the adapter
- `adapter` (IModalityAdapter): Adapter implementation

**Raises:**
- `ValueError`: If adapter name already exists

### Quality and Validation Methods

```python
def get_quality_metrics(self, modality: Optional[str] = None) -> Dict[str, Any]
```

Get quality metrics for modalities.

**Parameters:**
- `modality` (Optional[str]): Specific modality name (default: all modalities)

**Returns:**
- `Dict[str, Any]`: Quality metrics

```python
def validate_modalities(self, strict: bool = False) -> Dict[str, ValidationResult]
```

Validate all loaded modalities.

**Parameters:**
- `strict` (bool): Whether to use strict validation

**Returns:**
- `Dict[str, ValidationResult]`: Validation results for each modality

### Workspace Management

```python
def get_workspace_status(self) -> Dict[str, Any]
```

Get comprehensive workspace status.

**Returns:**
- `Dict[str, Any]`: Workspace status information including:
  - `workspace_path`: Path to workspace directory
  - `modalities_loaded`: Number of loaded modalities
  - `modality_names`: List of modality names
  - `registered_backends`: List of available backends
  - `registered_adapters`: List of available adapters
  - `modality_details`: Detailed information about each modality

```python
def list_workspace_files(self) -> Dict[str, List[Dict[str, Any]]]
```

List all files in the workspace organized by category.

**Returns:**
- `Dict[str, List[Dict[str, Any]]]`: Files organized by category ("data", "exports", "cache")

### Machine Learning Integration

```python
def check_ml_readiness(self, modality: str = None) -> Dict[str, Any]
```

Check if modalities are ready for machine learning workflows.

**Parameters:**
- `modality` (str): Specific modality to check (default: all modalities)

**Returns:**
- `Dict[str, Any]`: ML readiness assessment with scores and recommendations

```python
def prepare_ml_features(
    self,
    modality: str,
    feature_selection: str = "variance",
    n_features: int = 2000,
    normalization: str = "log1p",
    scaling: str = "standard"
) -> Dict[str, Any]
```

Prepare ML-ready feature matrices from biological data.

**Parameters:**
- `modality` (str): Name of modality to process
- `feature_selection` (str): Method for feature selection ('variance', 'correlation', 'chi2', 'mutual_info')
- `n_features` (int): Number of features to select
- `normalization` (str): Normalization method ('log1p', 'cpm', 'none')
- `scaling` (str): Scaling method ('standard', 'minmax', 'robust', 'none')

**Returns:**
- `Dict[str, Any]`: Processed feature information and metadata

### Plot Management

```python
def add_plot(
    self,
    plot: go.Figure,
    title: str = None,
    source: str = None,
    dataset_info: Dict[str, Any] = None,
    analysis_params: Dict[str, Any] = None,
) -> str
```

Add a plot to the collection with comprehensive metadata.

**Parameters:**
- `plot` (go.Figure): Plotly Figure object
- `title` (str): Optional title for the plot
- `source` (str): Optional source identifier (e.g., service name)
- `dataset_info` (Dict[str, Any]): Optional information about the dataset used
- `analysis_params` (Dict[str, Any]): Optional parameters used for the analysis

**Returns:**
- `str`: The unique ID assigned to the plot

```python
def get_latest_plots(self, n: int = None) -> List[Dict[str, Any]]
```

Get the n most recent plots with their metadata.

**Parameters:**
- `n` (int): Number of plots to return (None for all)

**Returns:**
- `List[Dict[str, Any]]`: List of plot entries with metadata

### Legacy Compatibility Methods

```python
def log_tool_usage(
    self,
    tool_name: str,
    parameters: Dict[str, Any],
    description: str = None
) -> None
```

Log tool usage for reproducibility tracking.

```python
def has_data(self) -> bool
```

Check if any modalities are loaded.

## Client Interfaces

### BaseClient

Abstract base class defining the interface for all Lobster client implementations.

```python
class BaseClient(ABC):
    """
    Abstract base class defining the interface for all Lobster client implementations.

    This ensures that both local (AgentClient) and cloud (CloudLobsterClient)
    implementations provide the same interface to the CLI and other components.
    """
```

#### Core Methods

```python
@abstractmethod
def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]
```

Process a user query through the system.

**Parameters:**
- `user_input` (str): The user's question or request
- `stream` (bool): Whether to stream the response

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `success`: bool
  - `response`: str
  - `error`: Optional[str]
  - `session_id`: str
  - `has_data`: bool
  - `plots`: List[Dict[str, Any]]
  - `duration`: float (optional)
  - `last_agent`: Optional[str] (optional)

```python
@abstractmethod
def get_status(self) -> Dict[str, Any]
```

Get the current status of the client/system.

**Returns:**
- `Dict[str, Any]`: Status information

```python
@abstractmethod
def list_workspace_files(self, pattern: str = "*") -> List[Dict[str, Any]]
```

List files in the workspace.

**Parameters:**
- `pattern` (str): Glob pattern for filtering files

**Returns:**
- `List[Dict[str, Any]]`: List of file information dictionaries

### AgentClient

Local client implementation using LangGraph multi-agent system.

```python
class AgentClient(BaseClient):
    def __init__(
        self,
        data_manager: Optional[DataManagerV2] = None,
        session_id: str = None,
        enable_reasoning: bool = True,
        enable_langfuse: bool = False,
        workspace_path: Optional[Path] = None,
        custom_callbacks: Optional[List] = None,
        manual_model_params: Optional[Dict[str, Any]] = None
    )
```

**Parameters:**
- `data_manager` (Optional[DataManagerV2]): DataManagerV2 instance (creates new if None)
- `session_id` (str): Unique session identifier
- `enable_reasoning` (bool): Show agent reasoning/thinking process
- `enable_langfuse` (bool): Enable Langfuse debugging callback
- `workspace_path` (Optional[Path]): Path to workspace for file operations
- `custom_callbacks` (Optional[List]): Additional callback handlers
- `manual_model_params` (Optional[Dict[str, Any]]): Manual model parameter overrides

#### Implementation-Specific Methods

```python
def reset(self) -> None
```

Reset the conversation state.

```python
def export_session(self, export_path: Optional[Path] = None) -> Path
```

Export the current session data.

**Parameters:**
- `export_path` (Optional[Path]): Optional path for the export file

**Returns:**
- `Path`: Path to the exported file

### APIAgentClient

WebSocket streaming client for web services.

```python
class APIAgentClient(BaseClient):
    def __init__(
        self,
        websocket_url: str,
        data_manager: Optional[DataManagerV2] = None,
        session_id: str = None
    )
```

**Parameters:**
- `websocket_url` (str): WebSocket server URL
- `data_manager` (Optional[DataManagerV2]): Optional local data manager
- `session_id` (str): Session identifier

## Provenance Tracking

### ProvenanceTracker

W3C-PROV-like provenance tracking system.

```python
class ProvenanceTracker:
    """
    W3C-PROV-like provenance tracking system.

    This class tracks data processing activities, entities, and agents
    to provide a complete audit trail and enable reproducibility.
    """
```

#### Constructor

```python
def __init__(self, namespace: str = "lobster") -> None
```

**Parameters:**
- `namespace` (str): Namespace for provenance identifiers

#### Core Methods

```python
def create_activity(
    self,
    activity_type: str,
    agent: str,
    inputs: Optional[List[Dict[str, Any]]] = None,
    outputs: Optional[List[Dict[str, Any]]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None
) -> str
```

Create a provenance activity record.

**Parameters:**
- `activity_type` (str): Type of activity performed
- `agent` (str): Agent responsible for the activity
- `inputs` (Optional[List[Dict[str, Any]]]): Input entities
- `outputs` (Optional[List[Dict[str, Any]]]): Output entities
- `parameters` (Optional[Dict[str, Any]]): Activity parameters
- `description` (Optional[str]): Human-readable description

**Returns:**
- `str`: Unique activity identifier

```python
def create_entity(
    self,
    entity_type: str,
    uri: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

Create a provenance entity record.

**Parameters:**
- `entity_type` (str): Type of entity
- `uri` (Optional[str]): Entity URI or path
- `metadata` (Optional[Dict[str, Any]]): Entity metadata

**Returns:**
- `str`: Unique entity identifier

```python
def to_dict(self) -> Dict[str, Any]
```

Export provenance information as dictionary.

**Returns:**
- `Dict[str, Any]`: Complete provenance record

```python
def add_to_anndata(self, adata: anndata.AnnData) -> anndata.AnnData
```

Add provenance information to AnnData object.

**Parameters:**
- `adata` (anndata.AnnData): AnnData object to annotate

**Returns:**
- `anndata.AnnData`: Annotated AnnData object

## Schema Validation

### ValidationResult

Data class for validation results with flexible error handling.

```python
@dataclass
class ValidationResult:
    """
    Result of a validation operation.

    This class encapsulates the results of validating biological data,
    supporting both errors (critical issues) and warnings (non-critical
    issues that don't prevent analysis).
    """

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Properties

```python
@property
def has_errors(self) -> bool
```

Check if validation found any errors.

```python
@property
def has_warnings(self) -> bool
```

Check if validation found any warnings.

```python
@property
def is_valid(self) -> bool
```

Check if validation passed (no errors).

#### Methods

```python
def add_error(self, message: str) -> None
```

Add an error message.

```python
def add_warning(self, message: str) -> None
```

Add a warning message.

```python
def merge(self, other: "ValidationResult") -> "ValidationResult"
```

Merge another validation result into this one.

```python
def summary(self) -> str
```

Generate a human-readable summary.

```python
def format_messages(self, include_info: bool = True) -> str
```

Format all messages for display.

## Usage Examples

### Basic DataManagerV2 Usage

```python
from lobster.core.data_manager_v2 import DataManagerV2
import pandas as pd

# Initialize data manager
data_manager = DataManagerV2(
    workspace_path="./my_workspace",
    enable_provenance=True
)

# Load data using adapter
adata = data_manager.load_modality(
    name="my_dataset",
    source="/path/to/data.csv",
    adapter="transcriptomics_single_cell"
)

# Get quality metrics
metrics = data_manager.get_quality_metrics("my_dataset")

# Save processed data
data_manager.save_modality("my_dataset", "processed_data.h5ad")
```

### Client Usage

```python
from lobster.core.client import AgentClient

# Create local client
client = AgentClient()

# Query the system
result = client.query("Load GSE194247 and perform quality assessment")

print(result['response'])
if result['plots']:
    print(f"Generated {len(result['plots'])} plots")
```

### Provenance Tracking

```python
from lobster.core.provenance import ProvenanceTracker

# Initialize tracker
provenance = ProvenanceTracker()

# Record an activity
activity_id = provenance.create_activity(
    activity_type="data_preprocessing",
    agent="preprocessing_service",
    parameters={"method": "log1p_normalization"},
    description="Applied log1p normalization to expression data"
)

# Export provenance
prov_data = provenance.to_dict()
```

## Error Handling

The Core API uses a hierarchical exception structure:

- `ValueError`: For invalid parameters or missing data
- `FileNotFoundError`: For missing files or paths
- `ImportError`: For missing optional dependencies
- `PermissionError`: For access-denied scenarios

All methods provide detailed error messages and maintain system state consistency even when operations fail.

## Thread Safety

The DataManagerV2 class is not thread-safe. For multi-threaded applications, use appropriate locking mechanisms or create separate instances per thread.

## Memory Management

DataManagerV2 holds all modalities in memory. For large datasets:
- Use the `remove_modality()` method to free memory
- Monitor memory usage with `get_workspace_status()`
- Consider using the `auto_save_state()` method for persistence