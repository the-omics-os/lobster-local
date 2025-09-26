# Interfaces API Reference

## Overview

The Interfaces API defines the abstract contracts and protocols that ensure consistent behavior across different implementations in the Lobster AI system. These interfaces enable modularity, extensibility, and maintainability by providing clear contracts for data backends, modality adapters, validators, and client implementations.

## Client Interface

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

#### Core Abstract Methods

```python
@abstractmethod
def __init__(self, *args, **kwargs):
    """Initialize the client with necessary configuration."""
    pass

@abstractmethod
def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]:
    """
    Process a user query through the system.

    Args:
        user_input: The user's question or request
        stream: Whether to stream the response

    Returns:
        Dictionary containing:
            - success: bool
            - response: str
            - error: Optional[str]
            - session_id: str
            - has_data: bool
            - plots: List[Dict[str, Any]]
            - duration: float (optional)
            - last_agent: Optional[str] (optional)
    """
    pass

@abstractmethod
def get_status(self) -> Dict[str, Any]:
    """
    Get the current status of the client/system.

    Returns:
        Dictionary containing status information including:
            - session_id: str
            - message_count: int (for local) or status: str (for cloud)
            - has_data: bool
            - workspace: str
            - data_summary: Optional[Dict] (if data is loaded)
    """
    pass

@abstractmethod
def list_workspace_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
    """
    List files in the workspace.

    Args:
        pattern: Glob pattern for filtering files

    Returns:
        List of dictionaries containing file information:
            - name: str
            - path: str
            - size: int
            - modified: str (ISO format timestamp)
    """
    pass

@abstractmethod
def reset(self) -> None:
    """Reset the conversation state."""
    pass

@abstractmethod
def export_session(self, export_path: Optional[Path] = None) -> Path:
    """
    Export the current session data.

    Args:
        export_path: Optional path for the export file

    Returns:
        Path to the exported file
    """
    pass
```

#### Optional Methods

```python
def get_usage(self) -> Dict[str, Any]:
    """
    Get usage statistics (primarily for cloud clients).

    Returns:
        Dictionary with usage information or error
    """
    return {"error": "Usage tracking not available for this client type", "success": False}

def list_models(self) -> Dict[str, Any]:
    """
    List available models (primarily for cloud clients).

    Returns:
        Dictionary with model list or error
    """
    return {"error": "Model listing not available for this client type", "success": False}
```

## Data Storage Interfaces

### IDataBackend

Abstract interface for data storage backends enabling support for different storage systems.

```python
class IDataBackend(ABC):
    """
    Abstract interface for data storage backends.

    This interface defines the contract for storing and retrieving
    bioinformatics data in various formats and storage systems.
    All backends must implement these core operations to ensure
    consistent behavior across different storage solutions.
    """
```

#### Core Abstract Methods

```python
@abstractmethod
def load(self, path: Union[str, Path], **kwargs) -> anndata.AnnData:
    """
    Load data from storage.

    Args:
        path: Path to the data file (local path or URI)
        **kwargs: Backend-specific loading parameters

    Returns:
        anndata.AnnData: Loaded data object

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is unsupported or corrupted
        PermissionError: If access is denied
    """
    pass

@abstractmethod
def save(self, adata: anndata.AnnData, path: Union[str, Path], **kwargs) -> None:
    """
    Save data to storage.

    Args:
        adata: AnnData object to save
        path: Destination path (local path or URI)
        **kwargs: Backend-specific saving parameters

    Raises:
        ValueError: If the data cannot be serialized
        PermissionError: If write access is denied
        OSError: If storage operation fails
    """
    pass

@abstractmethod
def exists(self, path: Union[str, Path]) -> bool:
    """
    Check if data exists at the specified path.

    Args:
        path: Path to check (local path or URI)

    Returns:
        bool: True if data exists, False otherwise
    """
    pass

@abstractmethod
def delete(self, path: Union[str, Path]) -> None:
    """
    Delete data at the specified path.

    Args:
        path: Path to delete (local path or URI)

    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If delete access is denied
    """
    pass

@abstractmethod
def list_files(self, directory: Union[str, Path], pattern: str = "*") -> list[str]:
    """
    List files in a directory matching the given pattern.

    Args:
        directory: Directory to search (local path or URI)
        pattern: File pattern to match (glob-style)

    Returns:
        list[str]: List of file paths matching the pattern

    Raises:
        FileNotFoundError: If the directory doesn't exist
        PermissionError: If read access is denied
    """
    pass

@abstractmethod
def get_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get metadata about a file.

    Args:
        path: Path to the file (local path or URI)

    Returns:
        Dict[str, Any]: Metadata dictionary containing:
            - size: File size in bytes
            - modified: Last modification timestamp
            - checksum: File checksum (if available)
            - format: Detected file format

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    pass
```

#### Default Implementation Methods

```python
def get_storage_info(self) -> Dict[str, Any]:
    """
    Get information about the storage backend.

    Returns:
        Dict[str, Any]: Storage backend information including:
            - backend_type: Type of backend (e.g., 'local', 's3')
            - capabilities: List of supported operations
            - configuration: Backend configuration details
    """
    return {
        "backend_type": self.__class__.__name__,
        "capabilities": ["load", "save", "exists", "delete", "list_files", "get_metadata"],
        "configuration": {}
    }

def validate_path(self, path: Union[str, Path]) -> Union[str, Path]:
    """
    Validate and normalize a path for this backend.

    Args:
        path: Path to validate

    Returns:
        Union[str, Path]: Validated and normalized path

    Raises:
        ValueError: If the path is invalid for this backend
    """
    return path

def supports_format(self, format_name: str) -> bool:
    """
    Check if the backend supports a specific file format.

    Args:
        format_name: Format to check (e.g., 'h5ad', 'csv', 'h5mu')

    Returns:
        bool: True if format is supported, False otherwise
    """
    return format_name.lower() in ['h5ad', 'csv']
```

## Data Adapter Interfaces

### IModalityAdapter

Abstract interface for modality-specific data adapters enabling support for different biological data modalities.

```python
class IModalityAdapter(ABC):
    """
    Abstract interface for modality-specific data adapters.

    This interface defines the contract for converting raw data from various
    sources into standardized AnnData objects with modality-specific schemas.
    Each adapter handles the specific requirements and conventions of its
    biological data modality.
    """
```

#### Core Abstract Methods

```python
@abstractmethod
def from_source(
    self,
    source: Union[str, Path, pd.DataFrame],
    **kwargs
) -> anndata.AnnData:
    """
    Convert source data to AnnData with appropriate schema.

    Args:
        source: Data source (file path, DataFrame, or other format)
        **kwargs: Modality-specific conversion parameters

    Returns:
        anndata.AnnData: Standardized data object with proper schema

    Raises:
        ValueError: If source data is invalid or cannot be converted
        FileNotFoundError: If source file doesn't exist
        TypeError: If source format is not supported
    """
    pass

@abstractmethod
def validate(
    self,
    adata: anndata.AnnData,
    strict: bool = False
) -> "ValidationResult":
    """
    Validate AnnData against modality schema.

    Args:
        adata: AnnData object to validate
        strict: If True, treat warnings as errors

    Returns:
        ValidationResult: Validation results with errors/warnings

    Raises:
        ValueError: If strict=True and validation fails
    """
    pass

@abstractmethod
def get_schema(self) -> Dict[str, Any]:
    """
    Return the expected schema for this modality.

    Returns:
        Dict[str, Any]: Schema definition containing:
            - required_obs: Required observation (cell/sample) metadata
            - required_var: Required variable (gene/protein) metadata
            - optional_obs: Optional observation metadata
            - optional_var: Optional variable metadata
            - layers: Expected data layers
            - obsm: Expected multi-dimensional observations
            - uns: Expected unstructured metadata
    """
    pass

@abstractmethod
def get_supported_formats(self) -> List[str]:
    """
    Get list of supported input formats.

    Returns:
        List[str]: List of supported file extensions or format names
    """
    pass
```

#### Default Implementation Methods

```python
def get_modality_name(self) -> str:
    """
    Get the name of this modality.

    Returns:
        str: Modality name (e.g., 'transcriptomics', 'proteomics')
    """
    return self.__class__.__name__.lower().replace('adapter', '')

def detect_format(self, source: Union[str, Path]) -> Optional[str]:
    """
    Detect the format of a source file.

    Args:
        source: Path to the source file

    Returns:
        Optional[str]: Detected format name, None if unknown
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        extension = path.suffix.lower()

        format_mapping = {
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.txt': 'txt',
            '.h5ad': 'h5ad',
            '.h5': 'h5',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.mtx': 'mtx',
            '.h5mu': 'h5mu'
        }

        return format_mapping.get(extension)

    return None

def preprocess_data(
    self,
    adata: anndata.AnnData,
    **kwargs
) -> anndata.AnnData:
    """
    Apply modality-specific preprocessing steps.

    Args:
        adata: Input AnnData object
        **kwargs: Preprocessing parameters

    Returns:
        anndata.AnnData: Preprocessed data object
    """
    return adata

def get_quality_metrics(self, adata: anndata.AnnData) -> Dict[str, Any]:
    """
    Calculate modality-specific quality metrics.

    Args:
        adata: AnnData object to analyze

    Returns:
        Dict[str, Any]: Quality metrics dictionary
    """
    return {
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "sparsity": 1.0 - (adata.X != 0).sum() / adata.X.size if hasattr(adata.X, 'size') else 0.0,
        "memory_usage": adata.X.nbytes if hasattr(adata.X, 'nbytes') else 0
    }

def add_provenance(
    self,
    adata: anndata.AnnData,
    source_info: Dict[str, Any],
    processing_params: Optional[Dict[str, Any]] = None
) -> anndata.AnnData:
    """
    Add provenance information to AnnData object.

    Args:
        adata: AnnData object to annotate
        source_info: Information about data source
        processing_params: Parameters used in processing

    Returns:
        anndata.AnnData: AnnData with provenance information
    """
    import datetime

    provenance = {
        "adapter": self.__class__.__name__,
        "modality": self.get_modality_name(),
        "source": source_info,
        "processing_params": processing_params or {},
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0.0"
    }

    if "provenance" not in adata.uns:
        adata.uns["provenance"] = []

    adata.uns["provenance"].append(provenance)

    return adata
```

## Validation Interfaces

### IValidator

Abstract interface for data validators with flexible error handling.

```python
class IValidator(ABC):
    """
    Abstract interface for data validators.

    This interface defines the contract for validating biological data
    against schemas with flexible error handling that supports both
    strict validation (errors cause failures) and permissive validation
    (warnings allow continued analysis).
    """
```

#### Core Abstract Methods

```python
@abstractmethod
def validate(
    self,
    adata: anndata.AnnData,
    strict: bool = False,
    check_types: bool = True,
    check_ranges: bool = True,
    check_completeness: bool = True
) -> ValidationResult:
    """
    Validate AnnData object against schema.

    Args:
        adata: AnnData object to validate
        strict: If True, treat warnings as errors
        check_types: Whether to validate data types
        check_ranges: Whether to validate value ranges
        check_completeness: Whether to check for required fields

    Returns:
        ValidationResult: Validation results with errors/warnings
    """
    pass

@abstractmethod
def validate_schema_compliance(
    self,
    adata: anndata.AnnData,
    schema: Dict[str, Any]
) -> ValidationResult:
    """
    Validate against a specific schema definition.

    Args:
        adata: AnnData object to validate
        schema: Schema definition to validate against

    Returns:
        ValidationResult: Schema validation results
    """
    pass
```

#### Default Implementation Methods

```python
def validate_obs_metadata(
    self,
    adata: anndata.AnnData,
    required_columns: Optional[List[str]] = None,
    optional_columns: Optional[List[str]] = None
) -> ValidationResult:
    """
    Validate observation (cell/sample) metadata.

    Args:
        adata: AnnData object to validate
        required_columns: List of required obs columns
        optional_columns: List of optional obs columns

    Returns:
        ValidationResult: Obs metadata validation results
    """
    result = ValidationResult()

    if required_columns:
        for col in required_columns:
            if col not in adata.obs.columns:
                result.add_error(f"Required obs column '{col}' is missing")
            elif adata.obs[col].isna().all():
                result.add_warning(f"Required obs column '{col}' contains only NaN values")

    # Check for unexpected columns
    expected_columns = set((required_columns or []) + (optional_columns or []))
    actual_columns = set(adata.obs.columns)
    unexpected = actual_columns - expected_columns

    if unexpected:
        result.add_info(f"Unexpected obs columns found: {list(unexpected)}")

    return result

def validate_data_quality(self, adata: anndata.AnnData) -> ValidationResult:
    """
    Perform basic data quality checks.

    Args:
        adata: AnnData object to validate

    Returns:
        ValidationResult: Data quality validation results
    """
    result = ValidationResult()

    # Check for empty data
    if adata.n_obs == 0:
        result.add_error("No observations (cells/samples) in dataset")
    if adata.n_vars == 0:
        result.add_error("No variables (genes/proteins) in dataset")

    # Check for NaN values in X matrix
    if hasattr(adata.X, 'isnan'):
        nan_count = adata.X.isnan().sum()
        if nan_count > 0:
            nan_percentage = (nan_count / adata.X.size) * 100
            if nan_percentage > 50:
                result.add_warning(f"High proportion of NaN values: {nan_percentage:.1f}%")
            else:
                result.add_info(f"NaN values in X matrix: {nan_percentage:.1f}%")

    return result
```

### ValidationResult

Data class for validation results with comprehensive error handling.

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
def has_errors(self) -> bool:
    """Check if validation found any errors."""
    return len(self.errors) > 0

@property
def has_warnings(self) -> bool:
    """Check if validation found any warnings."""
    return len(self.warnings) > 0

@property
def is_valid(self) -> bool:
    """Check if validation passed (no errors)."""
    return not self.has_errors
```

#### Methods

```python
def add_error(self, message: str) -> None:
    """Add an error message."""
    self.errors.append(message)

def add_warning(self, message: str) -> None:
    """Add a warning message."""
    self.warnings.append(message)

def add_info(self, message: str) -> None:
    """Add an informational message."""
    self.info.append(message)

def merge(self, other: "ValidationResult") -> "ValidationResult":
    """
    Merge another validation result into this one.

    Args:
        other: Another ValidationResult to merge

    Returns:
        ValidationResult: New merged result
    """
    return ValidationResult(
        errors=self.errors + other.errors,
        warnings=self.warnings + other.warnings,
        info=self.info + other.info,
        metadata={**self.metadata, **other.metadata}
    )

def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary representation."""
    return {
        "errors": self.errors,
        "warnings": self.warnings,
        "info": self.info,
        "metadata": self.metadata,
        "has_errors": self.has_errors,
        "has_warnings": self.has_warnings,
        "is_valid": self.is_valid
    }

def summary(self) -> str:
    """Generate a human-readable summary."""
    parts = []

    if self.has_errors:
        parts.append(f"{len(self.errors)} error(s)")

    if self.has_warnings:
        parts.append(f"{len(self.warnings)} warning(s)")

    if self.info:
        parts.append(f"{len(self.info)} info message(s)")

    if not parts:
        return "Validation passed with no issues"

    return f"Validation completed with {', '.join(parts)}"

def format_messages(self, include_info: bool = True) -> str:
    """Format all messages for display."""
    lines = []

    if self.errors:
        lines.append("ERRORS:")
        for error in self.errors:
            lines.append(f"  ❌ {error}")

    if self.warnings:
        if lines:
            lines.append("")
        lines.append("WARNINGS:")
        for warning in self.warnings:
            lines.append(f"  ⚠️  {warning}")

    if self.info and include_info:
        if lines:
            lines.append("")
        lines.append("INFO:")
        for info_msg in self.info:
            lines.append(f"  ℹ️  {info_msg}")

    return "\n".join(lines)
```

## Implementation Examples

### Custom Backend Implementation

```python
class S3Backend(IDataBackend):
    """Example S3 backend implementation."""

    def __init__(self, bucket_name: str, aws_credentials: Dict[str, str]):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', **aws_credentials)

    def load(self, path: Union[str, Path], **kwargs) -> anndata.AnnData:
        """Load data from S3."""
        # Download from S3 to temporary file
        with tempfile.NamedTemporaryFile() as tmp:
            self.s3_client.download_fileobj(
                self.bucket_name, str(path), tmp
            )
            tmp.seek(0)
            return anndata.read_h5ad(tmp.name)

    def save(self, adata: anndata.AnnData, path: Union[str, Path], **kwargs) -> None:
        """Save data to S3."""
        with tempfile.NamedTemporaryFile() as tmp:
            adata.write_h5ad(tmp.name)
            tmp.seek(0)
            self.s3_client.upload_fileobj(
                tmp, self.bucket_name, str(path)
            )

    # Implement other required methods...
```

### Custom Adapter Implementation

```python
class CustomOmicsAdapter(IModalityAdapter):
    """Example custom omics adapter."""

    def from_source(
        self,
        source: Union[str, Path, pd.DataFrame],
        **kwargs
    ) -> anndata.AnnData:
        """Convert custom format to AnnData."""
        if isinstance(source, pd.DataFrame):
            # Convert DataFrame to AnnData
            adata = anndata.AnnData(X=source.values)
            adata.obs_names = source.index
            adata.var_names = source.columns
        else:
            # Load from file
            df = pd.read_csv(source)
            adata = anndata.AnnData(X=df.iloc[:, 1:].values)
            adata.obs_names = df.iloc[:, 0]
            adata.var_names = df.columns[1:]

        return self.add_provenance(adata, {"source": str(source)})

    def validate(
        self,
        adata: anndata.AnnData,
        strict: bool = False
    ) -> ValidationResult:
        """Validate custom omics data."""
        result = ValidationResult()

        # Custom validation logic
        if adata.n_vars < 100:
            result.add_warning("Low number of features detected")

        return result

    def get_schema(self) -> Dict[str, Any]:
        """Return expected schema."""
        return {
            "required_obs": ["sample_id"],
            "optional_obs": ["batch", "condition"],
            "required_var": [],
            "optional_var": ["gene_biotype"],
            "layers": ["raw"],
            "obsm": [],
            "uns": ["processing_info"]
        }

    def get_supported_formats(self) -> List[str]:
        """Return supported formats."""
        return ["csv", "tsv", "xlsx"]
```

### Custom Validator Implementation

```python
class BioinformaticsValidator(IValidator):
    """Example bioinformatics data validator."""

    def validate(
        self,
        adata: anndata.AnnData,
        strict: bool = False,
        check_types: bool = True,
        check_ranges: bool = True,
        check_completeness: bool = True
    ) -> ValidationResult:
        """Validate bioinformatics data."""
        result = ValidationResult()

        # Basic structure validation
        quality_result = self.validate_data_quality(adata)
        result = result.merge(quality_result)

        # Data type validation
        if check_types:
            if not np.issubdtype(adata.X.dtype, np.number):
                result.add_error("Expression matrix must contain numeric data")

        # Range validation
        if check_ranges:
            if hasattr(adata.X, 'min') and adata.X.min() < 0:
                result.add_warning("Negative values detected in expression data")

        # Convert warnings to errors if strict mode
        if strict and result.has_warnings:
            result.errors.extend(result.warnings)
            result.warnings = []

        return result

    def validate_schema_compliance(
        self,
        adata: anndata.AnnData,
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate against specific schema."""
        result = ValidationResult()

        # Validate required columns
        obs_result = self.validate_obs_metadata(
            adata,
            schema.get('required_obs', []),
            schema.get('optional_obs', [])
        )
        result = result.merge(obs_result)

        # Validate expected layers
        expected_layers = schema.get('layers', [])
        for layer_name in expected_layers:
            if layer_name not in adata.layers:
                result.add_warning(f"Expected layer '{layer_name}' not found")

        return result
```

## Interface Integration

### Registration with DataManagerV2

```python
# Register custom implementations
data_manager = DataManagerV2()

# Register backend
s3_backend = S3Backend(bucket_name="my-bucket", aws_credentials=creds)
data_manager.register_backend("s3", s3_backend)

# Register adapter
custom_adapter = CustomOmicsAdapter()
data_manager.register_adapter("custom_omics", custom_adapter)

# Use registered implementations
adata = data_manager.load_modality(
    name="my_data",
    source="s3://my-bucket/data.csv",
    adapter="custom_omics"
)
```

### Validation Pipeline

```python
# Create validation pipeline
validator = BioinformaticsValidator()

# Validate with different strictness levels
result = validator.validate(adata, strict=False)
if result.has_errors:
    print("Validation failed:", result.format_messages())
elif result.has_warnings:
    print("Validation passed with warnings:", result.format_messages())
else:
    print("Validation passed successfully")
```

## Interface Benefits

### Modularity
- **Pluggable Components**: Easily swap implementations without changing core logic
- **Separation of Concerns**: Clear boundaries between different system layers
- **Testability**: Mock implementations for unit testing

### Extensibility
- **Custom Backends**: Support new storage systems (S3, GCS, databases)
- **New Modalities**: Add support for emerging data types
- **Flexible Validation**: Implement domain-specific validation rules

### Consistency
- **Uniform APIs**: Same interface regardless of underlying implementation
- **Error Handling**: Consistent exception hierarchy and error reporting
- **Documentation**: Self-documenting through interface contracts

The Interfaces API provides the foundation for Lobster AI's modular architecture, enabling seamless integration of new components while maintaining backward compatibility and system reliability.