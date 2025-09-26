"""
Comprehensive unit tests for interface implementations.

This module provides thorough testing of the interface system including
IModalityAdapter, IDataBackend, IValidator, and BaseClient interfaces,
compliance validation, abstract method enforcement, and implementation contracts.

Test coverage target: 95%+ with meaningful tests for interface contracts.
"""

import abc
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from lobster.core.interfaces.adapter import IModalityAdapter
from lobster.core.interfaces.backend import IDataBackend  
from lobster.core.interfaces.validator import IValidator, ValidationResult
from lobster.core.interfaces.base_client import BaseClient

from tests.mock_data.factories import SingleCellDataFactory, ProteomicsDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Mock Implementations for Testing
# ===============================================================================

class MockModalityAdapter(IModalityAdapter):
    """Mock implementation of IModalityAdapter for testing."""
    
    def __init__(self, modality_name="mock_modality"):
        self.modality_name = modality_name
        self.supported_formats = ["csv", "h5ad"]
        self.schema = {"required_obs": ["sample_id"], "required_var": ["feature_id"]}
    
    def from_source(self, source: Any, **kwargs) -> ad.AnnData:
        """Mock from_source implementation."""
        if isinstance(source, ad.AnnData):
            return source
        elif isinstance(source, pd.DataFrame):
            return ad.AnnData(X=source.values, obs=pd.DataFrame(index=source.index), 
                            var=pd.DataFrame(index=source.columns))
        else:
            return SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    
    def validate(self, adata: ad.AnnData, strict: bool = False) -> ValidationResult:
        """Mock validate implementation."""
        result = ValidationResult()
        if adata.n_obs == 0 or adata.n_vars == 0:
            result.errors.append("Empty dataset")
        return result
    
    def get_schema(self) -> Dict[str, Any]:
        """Mock get_schema implementation.""" 
        return self.schema
    
    def get_supported_formats(self) -> List[str]:
        """Mock get_supported_formats implementation."""
        return self.supported_formats


class MockDataBackend(IDataBackend):
    """Mock implementation of IDataBackend for testing."""

    def __init__(self, backend_type="mock_backend"):
        self.backend_type = backend_type
        self.storage_info = {"type": backend_type, "compression": True}

    def load(self, path: Union[str, Path], **kwargs) -> ad.AnnData:
        """Mock load implementation."""
        return SingleCellDataFactory(config=SMALL_DATASET_CONFIG)

    def save(self, adata: ad.AnnData, path: Union[str, Path], **kwargs) -> None:
        """Mock save implementation."""
        pass  # Mock save does nothing

    def exists(self, path: Union[str, Path]) -> bool:
        """Mock exists implementation."""
        return True  # Always return True for testing

    def delete(self, path: Union[str, Path]) -> None:
        """Mock delete implementation."""
        pass  # Mock delete does nothing

    def list_files(self, directory: Union[str, Path], pattern: str = "*") -> list[str]:
        """Mock list_files implementation."""
        return ["file1.h5ad", "file2.csv", "file3.h5mu"]

    def get_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Mock get_metadata implementation."""
        return {
            "size": 1024,
            "modified": "2023-01-01T00:00:00Z",
            "checksum": "abc123",
            "format": "h5ad"
        }

    def get_storage_info(self) -> Dict[str, Any]:
        """Mock get_storage_info implementation."""
        return self.storage_info


class MockValidator(IValidator):
    """Mock implementation of IValidator for testing."""

    def __init__(self, schema=None):
        self.schema = schema or {"required_obs": ["sample_id"], "required_var": ["feature_id"]}

    def validate(
        self,
        adata: ad.AnnData,
        strict: bool = False,
        check_types: bool = True,
        check_ranges: bool = True,
        check_completeness: bool = True
    ) -> ValidationResult:
        """Mock validate implementation."""
        result = ValidationResult()
        if adata.n_obs < 10:
            result.warnings.append("Low sample count")
        if adata.n_vars < 100:
            result.warnings.append("Low feature count")
        return result

    def validate_schema_compliance(
        self,
        adata: ad.AnnData,
        schema: Dict[str, Any]
    ) -> ValidationResult:
        """Mock validate_schema_compliance implementation."""
        result = ValidationResult()
        # Simple mock validation - check if required keys exist
        required_obs = schema.get("required_obs", [])
        for col in required_obs:
            if col not in adata.obs.columns:
                result.add_error(f"Missing required obs column: {col}")
        return result

    def get_schema(self) -> Dict[str, Any]:
        """Mock get_schema implementation."""
        return self.schema


class MockBaseClient(BaseClient):
    """Mock implementation of BaseClient for testing."""

    def __init__(self):
        self.session_id = "mock_session_123"
        self.query_count = 0
        self.conversation_history = []
        self.workspace_files = [
            {"name": "test1.csv", "path": "/workspace/test1.csv", "size": 1024, "modified": "2023-01-01T00:00:00Z"},
            {"name": "test2.h5ad", "path": "/workspace/test2.h5ad", "size": 2048, "modified": "2023-01-02T00:00:00Z"}
        ]

    def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]:
        """Mock query implementation."""
        self.query_count += 1
        response = f"Mock response to: {user_input}"
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        return {
            "response": response,
            "session_id": self.session_id,
            "query_count": self.query_count,
            "success": True,
            "has_data": False,
            "plots": []
        }

    def get_status(self) -> Dict[str, Any]:
        """Mock get_status implementation."""
        return {
            "status": "active",
            "session_id": self.session_id,
            "query_count": self.query_count,
            "has_data": False,
            "workspace": "/mock/workspace"
        }

    def list_workspace_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
        """Mock list_workspace_files implementation."""
        return self.workspace_files

    def read_file(self, filename: str) -> Optional[str]:
        """Mock read_file implementation."""
        return f"Mock content of {filename}"

    def write_file(self, filename: str, content: str) -> bool:
        """Mock write_file implementation."""
        return True

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Mock get_conversation_history implementation."""
        return self.conversation_history

    def reset(self) -> None:
        """Mock reset implementation."""
        self.conversation_history = []
        self.query_count = 0

    def export_session(self, export_path: Optional[Path] = None) -> Path:
        """Mock export_session implementation."""
        if export_path is None:
            export_path = Path("/tmp/mock_session_export.zip")
        return export_path


class IncompleteModalityAdapter:
    """Incomplete implementation missing required methods."""
    
    def from_source(self, source: Any) -> ad.AnnData:
        return SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    
    # Missing validate, get_schema, get_supported_formats methods


class IncompleteDataBackend:
    """Incomplete implementation missing required methods."""
    
    def load(self, path: Union[str, Path]) -> ad.AnnData:
        return SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
    
    # Missing save, exists methods


# ===============================================================================
# IModalityAdapter Interface Tests
# ===============================================================================

@pytest.mark.unit
class TestIModalityAdapterInterface:
    """Test IModalityAdapter interface compliance and contracts."""
    
    def test_interface_is_abstract(self):
        """Test that IModalityAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IModalityAdapter()
    
    def test_mock_implementation_valid(self):
        """Test that mock implementation satisfies interface."""
        adapter = MockModalityAdapter()
        
        # Should be instance of interface
        assert isinstance(adapter, IModalityAdapter)
        
        # Should have all required methods
        assert hasattr(adapter, 'from_source')
        assert hasattr(adapter, 'validate')
        assert hasattr(adapter, 'get_schema')
        assert hasattr(adapter, 'get_supported_formats')
        
        # Methods should be callable
        assert callable(adapter.from_source)
        assert callable(adapter.validate)
        assert callable(adapter.get_schema)
        assert callable(adapter.get_supported_formats)
    
    def test_from_source_method_contract(self):
        """Test from_source method contract."""
        adapter = MockModalityAdapter()
        
        # Should accept AnnData and return AnnData
        test_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        result = adapter.from_source(test_adata)
        assert isinstance(result, ad.AnnData)
        
        # Should accept DataFrame and return AnnData
        test_df = pd.DataFrame(np.random.rand(10, 20))
        result = adapter.from_source(test_df)
        assert isinstance(result, ad.AnnData)
        
        # Should accept file paths and return AnnData
        result = adapter.from_source("test_file.csv")
        assert isinstance(result, ad.AnnData)
    
    def test_validate_method_contract(self):
        """Test validate method contract."""
        adapter = MockModalityAdapter()
        test_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        # Should accept AnnData and return ValidationResult
        result = adapter.validate(test_adata)
        assert isinstance(result, ValidationResult)
        
        # Should accept strict parameter
        result_strict = adapter.validate(test_adata, strict=True)
        assert isinstance(result_strict, ValidationResult)
        
        # Should handle empty data
        empty_adata = ad.AnnData(X=np.array([]).reshape(0, 0))
        result_empty = adapter.validate(empty_adata)
        assert isinstance(result_empty, ValidationResult)
        assert result_empty.has_errors
    
    def test_get_schema_method_contract(self):
        """Test get_schema method contract."""
        adapter = MockModalityAdapter()
        
        schema = adapter.get_schema()
        
        # Should return dictionary
        assert isinstance(schema, dict)
        
        # Should contain expected structure
        assert "required_obs" in schema
        assert "required_var" in schema
    
    def test_get_supported_formats_method_contract(self):
        """Test get_supported_formats method contract."""
        adapter = MockModalityAdapter()
        
        formats = adapter.get_supported_formats()
        
        # Should return list
        assert isinstance(formats, list)
        
        # Should contain at least one format
        assert len(formats) > 0
        
        # All items should be strings
        assert all(isinstance(fmt, str) for fmt in formats)
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            # This should fail because IncompleteModalityAdapter doesn't implement all abstract methods
            class TestIncomplete(IncompleteModalityAdapter, IModalityAdapter):
                pass
            TestIncomplete()
    
    @pytest.mark.parametrize("method_name,args", [
        ("from_source", ("test_source",)),
        ("validate", (SingleCellDataFactory(config=SMALL_DATASET_CONFIG),)),
        ("get_schema", ()),
        ("get_supported_formats", ())
    ])
    def test_method_signatures(self, method_name, args):
        """Test that interface methods have correct signatures."""
        adapter = MockModalityAdapter()
        
        # Method should exist and be callable
        method = getattr(adapter, method_name)
        assert callable(method)
        
        # Should be able to call with expected arguments
        try:
            result = method(*args)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Method {method_name} failed with args {args}: {e}")


# ===============================================================================
# IDataBackend Interface Tests  
# ===============================================================================

@pytest.mark.unit
class TestIDataBackendInterface:
    """Test IDataBackend interface compliance and contracts."""
    
    def test_interface_is_abstract(self):
        """Test that IDataBackend cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IDataBackend()
    
    def test_mock_implementation_valid(self):
        """Test that mock implementation satisfies interface."""
        backend = MockDataBackend()
        
        # Should be instance of interface
        assert isinstance(backend, IDataBackend)
        
        # Should have all required methods
        assert hasattr(backend, 'load')
        assert hasattr(backend, 'save')
        assert hasattr(backend, 'exists')
        
        # Methods should be callable
        assert callable(backend.load)
        assert callable(backend.save)
        assert callable(backend.exists)
    
    def test_load_method_contract(self):
        """Test load method contract."""
        backend = MockDataBackend()
        
        # Should accept string path
        result = backend.load("test_file.h5ad")
        assert isinstance(result, ad.AnnData)
        
        # Should accept Path object
        result = backend.load(Path("test_file.h5ad"))
        assert isinstance(result, ad.AnnData)
    
    def test_save_method_contract(self):
        """Test save method contract."""
        backend = MockDataBackend()
        test_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        # Should accept AnnData and path string - should not raise exception
        try:
            backend.save(test_adata, "test_output.h5ad")
        except Exception as e:
            pytest.fail(f"save method failed: {e}")
        
        # Should accept Path object
        try:
            backend.save(test_adata, Path("test_output.h5ad"))
        except Exception as e:
            pytest.fail(f"save method failed with Path: {e}")
    
    def test_exists_method_contract(self):
        """Test exists method contract."""
        backend = MockDataBackend()
        
        # Should accept string path and return boolean
        result = backend.exists("test_file.h5ad")
        assert isinstance(result, bool)
        
        # Should accept Path object
        result = backend.exists(Path("test_file.h5ad"))
        assert isinstance(result, bool)
    
    def test_get_storage_info_method(self):
        """Test optional get_storage_info method."""
        backend = MockDataBackend()
        
        if hasattr(backend, 'get_storage_info'):
            info = backend.get_storage_info()
            assert isinstance(info, dict)
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            class TestIncomplete(IncompleteDataBackend, IDataBackend):
                pass
            TestIncomplete()


# ===============================================================================
# IValidator Interface Tests
# ===============================================================================

@pytest.mark.unit
class TestIValidatorInterface:
    """Test IValidator interface compliance and contracts."""
    
    def test_interface_is_abstract(self):
        """Test that IValidator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IValidator()
    
    def test_mock_implementation_valid(self):
        """Test that mock implementation satisfies interface."""
        validator = MockValidator()
        
        # Should be instance of interface
        assert isinstance(validator, IValidator)
        
        # Should have all required methods
        assert hasattr(validator, 'validate')
        assert hasattr(validator, 'get_schema')
        
        # Methods should be callable
        assert callable(validator.validate)
        assert callable(validator.get_schema)
    
    def test_validate_method_contract(self):
        """Test validate method contract."""
        validator = MockValidator()
        test_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        # Should accept AnnData and return ValidationResult
        result = validator.validate(test_adata)
        assert isinstance(result, ValidationResult)
        
        # Should handle various data sizes
        small_adata = ad.AnnData(X=np.random.rand(5, 50))
        result_small = validator.validate(small_adata)
        assert isinstance(result_small, ValidationResult)
        assert result_small.has_warnings  # Should warn about low counts
    
    def test_get_schema_method_contract(self):
        """Test get_schema method contract."""
        validator = MockValidator()
        
        schema = validator.get_schema()
        
        # Should return dictionary
        assert isinstance(schema, dict)


# ===============================================================================
# BaseClient Interface Tests
# ===============================================================================

@pytest.mark.unit
class TestBaseClientInterface:
    """Test BaseClient interface compliance and contracts."""
    
    def test_interface_is_abstract(self):
        """Test that BaseClient cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseClient()
    
    def test_mock_implementation_valid(self):
        """Test that mock implementation satisfies interface."""
        client = MockBaseClient()
        
        # Should be instance of interface
        assert isinstance(client, BaseClient)
        
        # Should have all required methods
        assert hasattr(client, 'query')
        assert hasattr(client, 'get_status')
        assert hasattr(client, 'export_session')
        
        # Methods should be callable
        assert callable(client.query)
        assert callable(client.get_status)
        assert callable(client.export_session)
    
    def test_query_method_contract(self):
        """Test query method contract."""
        client = MockBaseClient()
        
        # Should accept string input
        result = client.query("test query")
        assert isinstance(result, dict)
        assert "response" in result
        
        # Should accept stream parameter
        result_stream = client.query("test query", stream=True)
        assert isinstance(result_stream, dict)
    
    def test_get_status_method_contract(self):
        """Test get_status method contract."""
        client = MockBaseClient()
        
        status = client.get_status()
        
        # Should return dictionary
        assert isinstance(status, dict)
        assert "status" in status
    
    def test_export_session_method_contract(self):
        """Test export_session method contract."""
        client = MockBaseClient()
        
        # Should work without path
        result = client.export_session()
        assert isinstance(result, Path)
        
        # Should accept Path parameter
        custom_path = Path("/tmp/custom_export.zip")
        result_custom = client.export_session(custom_path)
        assert isinstance(result_custom, Path)
        assert result_custom == custom_path


# ===============================================================================
# ValidationResult Tests
# ===============================================================================

@pytest.mark.unit
class TestValidationResult:
    """Test ValidationResult utility class."""
    
    def test_initialization(self):
        """Test ValidationResult initialization."""
        result = ValidationResult()
        
        assert result.errors == []
        assert result.warnings == []
        assert not result.has_errors
        assert not result.has_warnings
    
    def test_add_error(self):
        """Test adding errors."""
        result = ValidationResult()
        
        result.errors.append("Test error")
        assert result.has_errors
        assert "Test error" in result.errors
    
    def test_add_warning(self):
        """Test adding warnings."""
        result = ValidationResult()
        
        result.warnings.append("Test warning")
        assert result.has_warnings
        assert "Test warning" in result.warnings
    
    def test_combined_errors_and_warnings(self):
        """Test result with both errors and warnings."""
        result = ValidationResult()
        
        result.errors.append("Critical error")
        result.warnings.append("Minor warning")
        
        assert result.has_errors
        assert result.has_warnings
        assert len(result.errors) == 1
        assert len(result.warnings) == 1


# ===============================================================================
# Interface Registry and Discovery Tests
# ===============================================================================

@pytest.mark.unit
class TestInterfaceRegistryPatterns:
    """Test interface registry and discovery patterns."""
    
    def test_multiple_adapter_implementations(self):
        """Test registry pattern with multiple adapter implementations."""
        # Simulate an adapter registry
        adapter_registry = {}
        
        # Register different adapters
        adapter_registry["rna_seq"] = MockModalityAdapter("rna_seq")
        adapter_registry["proteomics"] = MockModalityAdapter("proteomics")
        
        # Test registry functionality
        assert "rna_seq" in adapter_registry
        assert "proteomics" in adapter_registry
        
        # All should be valid adapters
        for name, adapter in adapter_registry.items():
            assert isinstance(adapter, IModalityAdapter)
            assert adapter.get_schema() is not None
            assert len(adapter.get_supported_formats()) > 0
    
    def test_multiple_backend_implementations(self):
        """Test registry pattern with multiple backend implementations."""
        # Simulate a backend registry
        backend_registry = {}
        
        # Register different backends
        backend_registry["h5ad"] = MockDataBackend("h5ad")
        backend_registry["zarr"] = MockDataBackend("zarr")
        backend_registry["csv"] = MockDataBackend("csv")
        
        # Test registry functionality
        assert len(backend_registry) == 3
        
        # All should be valid backends
        for name, backend in backend_registry.items():
            assert isinstance(backend, IDataBackend)
            # Test basic operations
            test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            backend.save(test_data, f"test.{name}")
            assert backend.exists(f"test.{name}")
    
    def test_interface_polymorphism(self):
        """Test polymorphic behavior of interface implementations."""
        # Create different implementations
        adapters = [
            MockModalityAdapter("type1"),
            MockModalityAdapter("type2"),
            MockModalityAdapter("type3")
        ]
        
        # Should be able to treat them polymorphically
        for adapter in adapters:
            # All should support the same interface
            test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            validation_result = adapter.validate(test_data)
            assert isinstance(validation_result, ValidationResult)
            
            schema = adapter.get_schema()
            assert isinstance(schema, dict)
            
            formats = adapter.get_supported_formats()
            assert isinstance(formats, list)
    
    def test_dynamic_interface_compliance_checking(self):
        """Test dynamic compliance checking for interfaces."""
        def check_adapter_compliance(obj) -> bool:
            """Check if object complies with IModalityAdapter interface."""
            required_methods = ['from_source', 'validate', 'get_schema', 'get_supported_formats']
            
            # Check if it's an instance of the interface
            if not isinstance(obj, IModalityAdapter):
                return False
            
            # Check if all required methods exist and are callable
            for method_name in required_methods:
                if not hasattr(obj, method_name) or not callable(getattr(obj, method_name)):
                    return False
            
            return True
        
        # Test with valid implementation
        valid_adapter = MockModalityAdapter()
        assert check_adapter_compliance(valid_adapter)
        
        # Test with invalid object
        invalid_adapter = object()
        assert not check_adapter_compliance(invalid_adapter)


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestInterfaceErrorHandling:
    """Test error handling in interface implementations."""
    
    def test_interface_method_error_handling(self):
        """Test error handling in interface method implementations."""
        
        class ErrorProneAdapter(IModalityAdapter):
            """Adapter that raises errors for testing."""
            
            def from_source(self, source: Any, **kwargs) -> ad.AnnData:
                if source == "error":
                    raise ValueError("Simulated error")
                return SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            
            def validate(self, adata: ad.AnnData, strict: bool = False) -> ValidationResult:
                if adata.n_obs == 999:  # Special error condition
                    raise RuntimeError("Validation error")
                return ValidationResult()
            
            def get_schema(self) -> Dict[str, Any]:
                return {"test": "schema"}
            
            def get_supported_formats(self) -> List[str]:
                return ["csv"]
        
        adapter = ErrorProneAdapter()
        
        # Should handle normal cases
        result = adapter.from_source("normal_input")
        assert isinstance(result, ad.AnnData)
        
        # Should propagate errors appropriately
        with pytest.raises(ValueError, match="Simulated error"):
            adapter.from_source("error")
        
        # Validation error handling
        error_data = ad.AnnData(X=np.random.rand(999, 10))  # Triggers error
        with pytest.raises(RuntimeError, match="Validation error"):
            adapter.validate(error_data)
    
    def test_interface_type_safety(self):
        """Test type safety in interface implementations."""
        adapter = MockModalityAdapter()
        
        # Test with correct types
        test_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        result = adapter.validate(test_adata)
        assert isinstance(result, ValidationResult)
        
        # Test type checking behavior (implementation dependent)
        # Some implementations might check types, others might not
        try:
            # This might work or raise TypeError depending on implementation
            adapter.validate("not_anndata")
        except (TypeError, AttributeError):
            # Expected behavior for strict type checking
            pass
    
    def test_concurrent_interface_usage(self):
        """Test thread safety considerations for interface implementations."""
        import threading
        import time
        
        adapter = MockModalityAdapter()
        results = []
        errors = []
        
        def worker(worker_id):
            """Worker function for concurrent testing."""
            try:
                # Simulate concurrent usage
                test_data = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                result = adapter.validate(test_data)
                results.append((worker_id, result))
                time.sleep(0.01)  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        
        # All results should be ValidationResult instances
        for worker_id, result in results:
            assert isinstance(result, ValidationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])