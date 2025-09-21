# Creating Adapters - Lobster AI Adapter Development Guide

## 🎯 Overview

This guide covers how to create adapters in the Lobster AI system. Adapters serve two primary purposes: **Modality Adapters** convert raw data from various sources into standardized AnnData objects with proper schemas, and **Backend Adapters** handle data storage and retrieval across different storage systems (local files, S3, etc.).

## 🏗️ Adapter Architecture

### Adapter Types

#### 1. **Modality Adapters**
- Convert raw biological data to standardized AnnData format
- Enforce modality-specific schemas (transcriptomics, proteomics, etc.)
- Handle data validation and quality control
- Located in: `lobster/core/adapters/`

#### 2. **Backend Adapters**
- Handle data storage and retrieval operations
- Support multiple storage systems (H5AD, MuData, S3-ready)
- Provide consistent API across storage backends
- Located in: `lobster/core/backends/`

### Design Principles
- **Interface Compliance**: All adapters implement abstract base interfaces
- **Schema Validation**: Enforce data quality and structure standards
- **Format Flexibility**: Support multiple input/output formats
- **Future-Proof**: S3-ready architecture for cloud scaling
- **Error Handling**: Comprehensive validation and error reporting

## 📋 Interface Definitions

### Modality Adapter Interface
```python
# lobster/core/interfaces/adapter.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from pathlib import Path
import anndata
import pandas as pd

class IModalityAdapter(ABC):
    """Abstract interface for modality-specific data adapters."""

    @abstractmethod
    def from_source(
        self,
        source: Union[str, Path, pd.DataFrame],
        **kwargs
    ) -> anndata.AnnData:
        """Convert source data to AnnData with appropriate schema."""
        pass

    @abstractmethod
    def validate(
        self,
        adata: anndata.AnnData,
        strict: bool = False
    ) -> "ValidationResult":
        """Validate AnnData against modality schema."""
        pass

    @abstractmethod
    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the expected schema."""
        pass

    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Return list of supported input formats."""
        pass
```

### Backend Interface
```python
# lobster/core/interfaces/backend.py
from abc import ABC, abstractmethod
from typing import Union, Dict, Any
from pathlib import Path
import anndata

class IDataBackend(ABC):
    """Abstract interface for data storage backends."""

    @abstractmethod
    def load(self, path: Union[str, Path], **kwargs) -> anndata.AnnData:
        """Load data from storage."""
        pass

    @abstractmethod
    def save(self, adata: anndata.AnnData, path: Union[str, Path], **kwargs) -> None:
        """Save data to storage."""
        pass

    @abstractmethod
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if data exists at path."""
        pass

    @abstractmethod
    def list_objects(self, prefix: str = "") -> List[str]:
        """List available data objects."""
        pass

    @abstractmethod
    def delete(self, path: Union[str, Path]) -> None:
        """Delete data at path."""
        pass
```

## 🔬 Creating Modality Adapters

### Step 1: Create Base Adapter Class

```python
# lobster/core/adapters/your_modality_adapter.py
"""
Your Modality Adapter for handling specific biological data type.

This adapter converts raw [modality type] data into standardized AnnData
format with appropriate schema validation and quality control.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata
import numpy as np
import pandas as pd
from lobster.core.adapters.base import BaseAdapter
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.your_schema import YourModalitySchema
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class YourModalityAdapter(BaseAdapter):
    """
    Adapter for your specific modality data type.

    This adapter handles loading, validation, and preprocessing of
    [specific modality] data with appropriate schema enforcement.
    """

    def __init__(
        self,
        data_subtype: str = "default",
        strict_validation: bool = False,
        **kwargs
    ):
        """
        Initialize the modality adapter.

        Args:
            data_subtype: Subtype of the modality (e.g., 'single_cell', 'bulk')
            strict_validation: Whether to use strict validation
            **kwargs: Additional configuration parameters
        """
        super().__init__(name="YourModalityAdapter")

        self.data_subtype = data_subtype
        self.strict_validation = strict_validation

        # Initialize validator
        self.validator = YourModalitySchema.create_validator(
            schema_type=data_subtype,
            strict=strict_validation
        )

        # Get recommended parameters
        self.recommended_params = YourModalitySchema.get_recommended_params(data_subtype)

        logger.info(f"Initialized {self.name} for {data_subtype} data")

    def from_source(
        self,
        source: Union[str, Path, pd.DataFrame],
        **kwargs
    ) -> anndata.AnnData:
        """
        Convert source data to AnnData with modality schema.

        Args:
            source: Data source (file path, DataFrame, or existing AnnData)
            **kwargs: Conversion parameters:
                - transpose: Whether to transpose the data matrix
                - obs_metadata: Sample/observation metadata
                - var_metadata: Feature/variable metadata
                - delimiter: File delimiter for text files
                - sheet_name: Excel sheet name
                - [modality-specific parameters]

        Returns:
            anndata.AnnData: Standardized data object with proper schema

        Raises:
            ValueError: If source data is invalid
            FileNotFoundError: If source file doesn't exist
            TypeError: If source format is not supported
        """
        try:
            logger.info(f"Converting source data: {type(source).__name__}")

            # Handle different source types
            if isinstance(source, (str, Path)):
                adata = self._load_from_file(Path(source), **kwargs)
            elif isinstance(source, pd.DataFrame):
                adata = self._load_from_dataframe(source, **kwargs)
            elif isinstance(source, anndata.AnnData):
                adata = self._load_from_anndata(source, **kwargs)
            else:
                raise TypeError(f"Unsupported source type: {type(source)}")

            # Apply modality-specific preprocessing
            adata = self._apply_modality_preprocessing(adata, **kwargs)

            # Validate the result
            validation_result = self.validate(adata, strict=self.strict_validation)
            if not validation_result.is_valid and self.strict_validation:
                raise ValueError(f"Validation failed: {validation_result.error_summary}")

            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(f"Schema validation warning: {warning}")

            logger.info(f"Successfully converted to AnnData: {adata.n_obs} obs, {adata.n_vars} vars")
            return adata

        except Exception as e:
            logger.error(f"Failed to convert source data: {str(e)}")
            raise

    def _load_from_file(self, file_path: Path, **kwargs) -> anndata.AnnData:
        """Load data from file based on extension."""

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == '.h5ad':
            return anndata.read_h5ad(file_path)

        elif suffix in ['.csv', '.tsv', '.txt']:
            delimiter = kwargs.get('delimiter', '\t' if suffix == '.tsv' else ',')
            df = pd.read_csv(file_path, delimiter=delimiter, index_col=0)
            return self._dataframe_to_anndata(df, **kwargs)

        elif suffix in ['.xlsx', '.xls']:
            sheet_name = kwargs.get('sheet_name', 0)
            df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
            return self._dataframe_to_anndata(df, **kwargs)

        elif suffix in ['.h5', '.hdf5']:
            # Handle HDF5 files (implement based on your format)
            return self._load_hdf5(file_path, **kwargs)

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_from_dataframe(self, df: pd.DataFrame, **kwargs) -> anndata.AnnData:
        """Convert DataFrame to AnnData."""
        return self._dataframe_to_anndata(df, **kwargs)

    def _load_from_anndata(self, adata: anndata.AnnData, **kwargs) -> anndata.AnnData:
        """Process existing AnnData to ensure schema compliance."""
        # Create a copy to avoid modifying the original
        result_adata = adata.copy()

        # Apply any necessary transformations
        return self._apply_modality_preprocessing(result_adata, **kwargs)

    def _dataframe_to_anndata(self, df: pd.DataFrame, **kwargs) -> anndata.AnnData:
        """Convert DataFrame to AnnData with proper structure."""

        # Handle transposition if needed
        transpose = kwargs.get('transpose', True)
        if transpose:
            df = df.T

        # Create AnnData object
        adata = anndata.AnnData(X=df.values)
        adata.obs.index = df.index
        adata.var.index = df.columns

        # Add observation metadata if provided
        obs_metadata = kwargs.get('obs_metadata')
        if obs_metadata is not None:
            if isinstance(obs_metadata, pd.DataFrame):
                # Align metadata with observations
                common_indices = adata.obs.index.intersection(obs_metadata.index)
                adata.obs = adata.obs.join(obs_metadata.loc[common_indices])
            else:
                logger.warning("obs_metadata should be a DataFrame")

        # Add variable metadata if provided
        var_metadata = kwargs.get('var_metadata')
        if var_metadata is not None:
            if isinstance(var_metadata, pd.DataFrame):
                common_indices = adata.var.index.intersection(var_metadata.index)
                adata.var = adata.var.join(var_metadata.loc[common_indices])

        return adata

    def _apply_modality_preprocessing(self, adata: anndata.AnnData, **kwargs) -> anndata.AnnData:
        """Apply modality-specific preprocessing and annotations."""

        # Add modality type
        adata.uns['modality'] = self.data_subtype
        adata.uns['adapter'] = self.name
        adata.uns['processing_timestamp'] = pd.Timestamp.now().isoformat()

        # Apply modality-specific processing
        if self.data_subtype == 'mass_spectrometry':
            adata = self._apply_ms_preprocessing(adata, **kwargs)
        elif self.data_subtype == 'affinity_proteomics':
            adata = self._apply_affinity_preprocessing(adata, **kwargs)
        # Add more subtypes as needed

        return adata

    def _apply_ms_preprocessing(self, adata: anndata.AnnData, **kwargs) -> anndata.AnnData:
        """Apply mass spectrometry specific preprocessing."""

        # Handle missing values typical in MS data
        missing_threshold = kwargs.get('missing_threshold', 0.5)

        # Calculate missing value percentages
        missing_per_protein = (adata.X == 0).sum(axis=0) / adata.n_obs
        missing_per_sample = (adata.X == 0).sum(axis=1) / adata.n_vars

        # Store in AnnData
        adata.var['missing_pct'] = missing_per_protein
        adata.obs['missing_pct'] = missing_per_sample

        # Log transform if requested
        if kwargs.get('log_transform', True):
            adata.layers['log2_intensities'] = np.log2(adata.X + 1)
            adata.X = adata.layers['log2_intensities']

        return adata

    def validate(
        self,
        adata: anndata.AnnData,
        strict: bool = False
    ) -> ValidationResult:
        """
        Validate AnnData against modality schema.

        Args:
            adata: AnnData object to validate
            strict: Whether to use strict validation

        Returns:
            ValidationResult: Validation results with errors/warnings
        """
        return self.validator.validate(adata, strict=strict)

    def get_schema_info(self) -> Dict[str, Any]:
        """Get information about the expected schema."""
        return {
            'name': self.name,
            'modality': self.data_subtype,
            'required_obs': self.validator.required_obs_columns,
            'required_var': self.validator.required_var_columns,
            'recommended_params': self.recommended_params,
            'supported_formats': self.supported_formats()
        }

    def supported_formats(self) -> List[str]:
        """Return list of supported input formats."""
        return ['.csv', '.tsv', '.txt', '.xlsx', '.xls', '.h5ad', '.h5', '.hdf5']

    def get_conversion_examples(self) -> Dict[str, str]:
        """Provide usage examples for common scenarios."""
        return {
            'csv_file': "adapter.from_source('data.csv', transpose=True)",
            'excel_file': "adapter.from_source('data.xlsx', sheet_name='Sheet1')",
            'with_metadata': "adapter.from_source(df, obs_metadata=sample_info)",
            'dataframe': "adapter.from_source(expression_df, transpose=False)"
        }
```

### Step 2: Create Schema Validation

```python
# lobster/core/schemas/your_schema.py
"""Schema definitions for your modality."""

from typing import Dict, List, Any, Optional
from lobster.core.schemas.base import BaseSchema
from lobster.core.interfaces.validator import ValidationResult


class YourModalitySchema(BaseSchema):
    """Schema for your specific modality data."""

    @classmethod
    def get_required_obs_columns(cls, schema_type: str = "default") -> List[str]:
        """Required observation (sample) columns."""
        base_columns = ['sample_id']

        if schema_type == 'mass_spectrometry':
            return base_columns + ['sample_type', 'batch']
        elif schema_type == 'affinity_proteomics':
            return base_columns + ['panel_type', 'dilution_factor']

        return base_columns

    @classmethod
    def get_required_var_columns(cls, schema_type: str = "default") -> List[str]:
        """Required variable (feature) columns."""
        base_columns = ['feature_id']

        if schema_type == 'mass_spectrometry':
            return base_columns + ['protein_group', 'gene_name']
        elif schema_type == 'affinity_proteomics':
            return base_columns + ['antibody_id', 'target_protein']

        return base_columns

    @classmethod
    def get_recommended_qc_thresholds(cls, schema_type: str) -> Dict[str, float]:
        """Get QC thresholds for the modality."""
        if schema_type == 'mass_spectrometry':
            return {
                'min_peptides_per_protein': 2,
                'max_missing_per_protein': 0.5,
                'min_intensity': 1.0
            }
        elif schema_type == 'affinity_proteomics':
            return {
                'max_cv_within_group': 0.2,
                'min_signal_to_noise': 3.0,
                'max_missing_per_sample': 0.3
            }

        return {}
```

## 🗄️ Creating Backend Adapters

### Backend Template

```python
# lobster/core/backends/your_backend.py
"""
Your Backend for handling data storage and retrieval.

This backend provides data persistence for [specific storage system]
with S3-ready architecture for future cloud scaling.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import anndata
from lobster.core.backends.base import BaseBackend
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class YourBackend(BaseBackend):
    """
    Backend for [storage system] with S3-ready path handling.

    This backend handles data storage and retrieval for [specific format]
    with path parsing ready for future S3 integration.
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
        **storage_config
    ):
        """
        Initialize the backend.

        Args:
            base_path: Optional base path for all operations
            **storage_config: Storage-specific configuration
        """
        super().__init__(base_path=base_path)

        # Storage-specific configuration
        self.config = storage_config

        # S3-ready configuration (for future use)
        self.s3_config = {
            "bucket": storage_config.get("s3_bucket"),
            "region": storage_config.get("s3_region"),
            "prefix": storage_config.get("s3_prefix", "")
        }

        logger.info(f"Initialized {self.__class__.__name__}")

    def load(self, path: Union[str, Path], **kwargs) -> anndata.AnnData:
        """
        Load data from storage.

        Args:
            path: Path to the data (local path or future S3 URI)
            **kwargs: Loading parameters

        Returns:
            anndata.AnnData: Loaded data object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        try:
            resolved_path = self._resolve_path(path)

            if self._is_s3_path(path):
                return self._load_from_s3(resolved_path, **kwargs)
            else:
                return self._load_from_local(resolved_path, **kwargs)

        except Exception as e:
            logger.error(f"Failed to load data from {path}: {str(e)}")
            raise

    def save(
        self,
        adata: anndata.AnnData,
        path: Union[str, Path],
        **kwargs
    ) -> None:
        """
        Save data to storage.

        Args:
            adata: AnnData object to save
            path: Destination path
            **kwargs: Saving parameters

        Raises:
            ValueError: If data cannot be serialized
            PermissionError: If write access is denied
        """
        try:
            resolved_path = self._resolve_path(path)

            if self._is_s3_path(path):
                self._save_to_s3(adata, resolved_path, **kwargs)
            else:
                self._save_to_local(adata, resolved_path, **kwargs)

            logger.info(f"Successfully saved data to {path}")

        except Exception as e:
            logger.error(f"Failed to save data to {path}: {str(e)}")
            raise

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if data exists at path."""
        try:
            resolved_path = self._resolve_path(path)

            if self._is_s3_path(path):
                return self._exists_in_s3(resolved_path)
            else:
                return Path(resolved_path).exists()

        except Exception as e:
            logger.warning(f"Error checking existence of {path}: {str(e)}")
            return False

    def list_objects(self, prefix: str = "") -> List[str]:
        """List available data objects."""
        try:
            if self.s3_config.get("bucket"):
                return self._list_s3_objects(prefix)
            else:
                return self._list_local_objects(prefix)

        except Exception as e:
            logger.error(f"Failed to list objects with prefix {prefix}: {str(e)}")
            return []

    def delete(self, path: Union[str, Path]) -> None:
        """Delete data at path."""
        try:
            resolved_path = self._resolve_path(path)

            if self._is_s3_path(path):
                self._delete_from_s3(resolved_path)
            else:
                Path(resolved_path).unlink()

            logger.info(f"Successfully deleted {path}")

        except Exception as e:
            logger.error(f"Failed to delete {path}: {str(e)}")
            raise

    def _load_from_local(self, path: Path, **kwargs) -> anndata.AnnData:
        """Load data from local file system."""

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Implement format-specific loading
        suffix = path.suffix.lower()

        if suffix == '.h5ad':
            return anndata.read_h5ad(path)
        elif suffix in ['.h5', '.hdf5']:
            return self._load_hdf5(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _save_to_local(self, adata: anndata.AnnData, path: Path, **kwargs) -> None:
        """Save data to local file system."""

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Implement format-specific saving
        suffix = path.suffix.lower()

        if suffix == '.h5ad':
            compression = kwargs.get('compression', 'gzip')
            adata.write_h5ad(path, compression=compression)
        elif suffix in ['.h5', '.hdf5']:
            self._save_hdf5(adata, path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _is_s3_path(self, path: Union[str, Path]) -> bool:
        """Check if path is an S3 URI."""
        return str(path).startswith(('s3://', 's3a://', 's3n://'))

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path with base_path if needed."""
        path_obj = Path(path)

        if not path_obj.is_absolute() and self.base_path:
            return self.base_path / path_obj

        return path_obj

    # S3 methods (implement when S3 support is added)
    def _load_from_s3(self, path: str, **kwargs) -> anndata.AnnData:
        """Load data from S3 (future implementation)."""
        raise NotImplementedError("S3 support not yet implemented")

    def _save_to_s3(self, adata: anndata.AnnData, path: str, **kwargs) -> None:
        """Save data to S3 (future implementation)."""
        raise NotImplementedError("S3 support not yet implemented")

    def _exists_in_s3(self, path: str) -> bool:
        """Check if object exists in S3 (future implementation)."""
        raise NotImplementedError("S3 support not yet implemented")

    def _list_s3_objects(self, prefix: str) -> List[str]:
        """List S3 objects (future implementation)."""
        raise NotImplementedError("S3 support not yet implemented")

    def _delete_from_s3(self, path: str) -> None:
        """Delete from S3 (future implementation)."""
        raise NotImplementedError("S3 support not yet implemented")
```

## 📊 Testing Adapters

### Unit Test Template
```python
# tests/unit/adapters/test_your_adapter.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from lobster.core.adapters.your_adapter import YourModalityAdapter
from tests.mock_data.generators import generate_mock_data


class TestYourModalityAdapter:

    @pytest.fixture
    def adapter(self):
        return YourModalityAdapter(data_subtype="default")

    @pytest.fixture
    def sample_dataframe(self):
        return generate_mock_data(n_samples=10, n_features=100)

    def test_adapter_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter.name == "YourModalityAdapter"
        assert adapter.data_subtype == "default"
        assert adapter.validator is not None

    def test_from_dataframe(self, adapter, sample_dataframe):
        """Test conversion from DataFrame."""
        adata = adapter.from_source(sample_dataframe)

        assert adata.n_obs == sample_dataframe.shape[1]  # transposed
        assert adata.n_vars == sample_dataframe.shape[0]
        assert 'modality' in adata.uns

    def test_validation(self, adapter, sample_dataframe):
        """Test schema validation."""
        adata = adapter.from_source(sample_dataframe)

        validation_result = adapter.validate(adata)
        assert validation_result is not None
        # Add specific validation checks

    def test_file_loading(self, adapter, tmp_path):
        """Test loading from various file formats."""
        # Create test CSV file
        test_data = pd.DataFrame(np.random.randn(10, 20))
        csv_path = tmp_path / "test.csv"
        test_data.to_csv(csv_path)

        adata = adapter.from_source(csv_path)
        assert adata.n_obs > 0
        assert adata.n_vars > 0

    def test_error_handling(self, adapter):
        """Test error handling for invalid inputs."""
        with pytest.raises(FileNotFoundError):
            adapter.from_source("nonexistent_file.csv")

        with pytest.raises(TypeError):
            adapter.from_source(12345)  # Invalid type
```

## 🎯 Best Practices

### 1. Schema Design
- **Flexible Validation**: Support both strict and permissive validation modes
- **Biological Relevance**: Include biologically meaningful QC thresholds
- **Extensible**: Design schemas to accommodate future data types
- **Documentation**: Provide clear schema documentation and examples

### 2. Error Handling
```python
def robust_conversion(self, source, **kwargs):
    """Example of robust error handling."""
    try:
        # Attempt conversion
        result = self._convert_data(source, **kwargs)

        # Validate result
        validation = self.validate(result)
        if not validation.is_valid:
            if self.strict_validation:
                raise ValueError(f"Validation failed: {validation.errors}")
            else:
                logger.warning(f"Validation warnings: {validation.warnings}")

        return result

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Input file not found: {e}")
    except pd.errors.EmptyDataError:
        raise ValueError("Input file is empty or contains no valid data")
    except Exception as e:
        logger.error(f"Unexpected error during conversion: {e}")
        raise
```

### 3. Performance Optimization
```python
def memory_efficient_loading(self, path, **kwargs):
    """Handle large files efficiently."""

    # Check file size first
    file_size = Path(path).stat().st_size

    if file_size > 1e9:  # 1GB threshold
        logger.info("Large file detected, using chunked reading")
        return self._load_in_chunks(path, **kwargs)
    else:
        return self._load_standard(path, **kwargs)

def _load_in_chunks(self, path, chunk_size=10000):
    """Load large files in chunks."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)
```

### 4. S3-Ready Design
```python
def s3_ready_path_handling(self, path):
    """Parse paths for current and future S3 support."""

    if isinstance(path, str) and path.startswith('s3://'):
        # Parse S3 URI
        parsed = urlparse(path)
        return {
            'type': 's3',
            'bucket': parsed.netloc,
            'key': parsed.path.lstrip('/'),
            'region': self.s3_config.get('region')
        }
    else:
        # Local path
        return {
            'type': 'local',
            'path': Path(path),
            'absolute': Path(path).resolve()
        }
```

## 🔍 Integration with DataManagerV2

Adapters integrate with the DataManagerV2 through the adapter registry:

```python
# Register your adapter
from lobster.core.data_manager_v2 import DataManagerV2

data_manager = DataManagerV2()

# Register modality adapter
data_manager.register_modality_adapter(
    'your_modality',
    YourModalityAdapter
)

# Register backend adapter
data_manager.register_backend_adapter(
    'your_format',
    YourBackend
)

# Use in loading
adata = data_manager.load_data(
    'data.csv',
    modality_type='your_modality',
    adapter_params={'data_subtype': 'specific_type'}
)
```

## 📚 Complete Example Usage

```python
# Example: Creating a metabolomics adapter
from lobster.core.adapters.metabolomics_adapter import MetabolomicsAdapter

# Initialize adapter
adapter = MetabolomicsAdapter(
    data_subtype='targeted_metabolomics',
    strict_validation=False
)

# Convert from various sources
adata_from_csv = adapter.from_source(
    'metabolite_data.csv',
    transpose=True,
    obs_metadata=sample_metadata
)

adata_from_excel = adapter.from_source(
    'data.xlsx',
    sheet_name='Concentrations',
    log_transform=True
)

# Validate data
validation_result = adapter.validate(adata_from_csv)
if validation_result.warnings:
    print("Schema warnings:", validation_result.warnings)

# Get schema information
schema_info = adapter.get_schema_info()
print("Required columns:", schema_info['required_obs'])
```

This comprehensive guide provides everything needed to create robust, extensible adapters that handle data conversion, validation, and storage while maintaining compatibility with the Lobster AI platform's architecture and future cloud scaling requirements.