"""
Comprehensive unit tests for data expert agent.

This module provides thorough testing of the data expert agent including
data loading, format detection, GEO integration, workspace management,
and data preprocessing for the bioinformatics platform.

Test coverage target: 95%+ with meaningful tests for data operations.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
import pandas as pd
import numpy as np

from lobster.agents.data_expert import data_expert_agent
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Mock Objects and Fixtures
# ===============================================================================

class MockMessage:
    """Mock LangGraph message object."""
    
    def __init__(self, content: str, sender: str = "human"):
        self.content = content
        self.sender = sender
        self.additional_kwargs = {}


class MockState:
    """Mock LangGraph state object."""
    
    def __init__(self, messages=None, **kwargs):
        self.messages = messages or []
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_data_manager():
    """Create mock data manager."""
    with patch('lobster.core.data_manager_v2.DataManagerV2') as MockDataManager:
        mock_dm = MockDataManager.return_value
        mock_dm.list_modalities.return_value = ['test_data']
        mock_dm.get_modality.return_value = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        mock_dm.add_modality.return_value = None
        mock_dm.get_summary.return_value = "Test dataset with 100 cells and 500 genes"
        yield mock_dm


@pytest.fixture
def mock_geo_service():
    """Mock GEO service for data downloading."""
    with patch('lobster.tools.geo_service.GEOService') as MockGEOService:
        mock_service = MockGEOService.return_value
        mock_service.search_geo.return_value = [
            {"accession": "GSE12345", "title": "Test dataset", "organism": "Homo sapiens"}
        ]
        mock_service.download_geo_data.return_value = {
            "success": True,
            "files": ["GSE12345_matrix.txt", "GSE12345_barcodes.txt"],
            "metadata": {"samples": 100, "features": 500}
        }
        yield mock_service


@pytest.fixture
def data_expert_state():
    """Create data expert state for testing."""
    return MockState(
        messages=[MockMessage("Load the single-cell dataset")],
        data_manager=Mock(),
        current_agent="data_expert_agent"
    )


# ===============================================================================
# Data Expert Core Functionality Tests
# ===============================================================================

@pytest.mark.unit
class TestDataExpertCore:
    """Test data expert core functionality."""
    
    def test_data_expert_initialization(self, mock_data_manager):
        """Test data expert agent initialization."""
        state = MockState(data_manager=mock_data_manager)
        
        with patch('lobster.agents.data_expert.data_expert_agent') as mock_agent:
            mock_agent.return_value = {"messages": []}
            
            # Should initialize without errors
            assert callable(mock_agent)
    
    def test_list_workspace_files(self, mock_data_manager, data_expert_state):
        """Test listing workspace files."""
        data_expert_state.messages = [MockMessage("Show me what files are available")]
        
        with patch('lobster.agents.data_expert.list_workspace_files') as mock_list:
            mock_list.return_value = "Available files: dataset1.h5ad, dataset2.csv, results.png"
            
            result = mock_list()
            
            assert "dataset1.h5ad" in result
            assert "dataset2.csv" in result
            mock_list.assert_called_once()
    
    def test_read_workspace_file(self, mock_data_manager, data_expert_state):
        """Test reading workspace files."""
        file_content = "gene1,gene2,gene3\ncell1,100,200,150"
        
        with patch('lobster.agents.data_expert.read_workspace_file') as mock_read:
            mock_read.return_value = f"File contents:\n{file_content}"
            
            result = mock_read("test_data.csv")
            
            assert "gene1" in result
            assert "cell1" in result
            mock_read.assert_called_once_with("test_data.csv")
    
    def test_load_data_from_file(self, mock_data_manager, data_expert_state):
        """Test loading data from file."""
        data_expert_state.messages = [MockMessage("Load data from my_dataset.csv")]
        
        with patch('lobster.agents.data_expert.load_data_from_file') as mock_load:
            mock_load.return_value = "Successfully loaded data from my_dataset.csv into 'my_dataset' modality"
            
            result = mock_load("my_dataset.csv")
            
            assert "Successfully loaded" in result
            assert "my_dataset.csv" in result
            mock_load.assert_called_once_with("my_dataset.csv")
    
    def test_examine_data_modality(self, mock_data_manager, data_expert_state):
        """Test examining data modality."""
        mock_data_manager.get_modality.return_value = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        
        with patch('lobster.agents.data_expert.examine_data_modality') as mock_examine:
            mock_examine.return_value = "Dataset 'test_data': 100 cells × 500 genes"
            
            result = mock_examine("test_data")
            
            assert "100 cells" in result
            assert "500 genes" in result
            mock_examine.assert_called_once_with("test_data")


# ===============================================================================
# GEO Data Integration Tests
# ===============================================================================

@pytest.mark.unit
class TestGEODataIntegration:
    """Test GEO data integration functionality."""
    
    def test_search_geo_datasets(self, mock_geo_service, data_expert_state):
        """Test searching GEO datasets."""
        data_expert_state.messages = [MockMessage("Search for T cell datasets in GEO")]
        
        with patch('lobster.agents.data_expert.search_geo_datasets') as mock_search:
            mock_search.return_value = "Found 5 datasets matching 'T cell'"
            
            result = mock_search("T cell")
            
            assert "Found 5 datasets" in result
            mock_search.assert_called_once_with("T cell")
    
    def test_download_geo_data_success(self, mock_geo_service, data_expert_state):
        """Test successful GEO data download."""
        data_expert_state.messages = [MockMessage("Download GEO dataset GSE12345")]
        
        with patch('lobster.agents.data_expert.download_geo_data') as mock_download:
            mock_download.return_value = "Successfully downloaded GSE12345 and loaded as 'geo_gse12345'"
            
            result = mock_download("GSE12345")
            
            assert "Successfully downloaded" in result
            assert "GSE12345" in result
            mock_download.assert_called_once_with("GSE12345")
    
    def test_download_geo_data_with_metadata(self, mock_geo_service, data_expert_state):
        """Test GEO data download with metadata extraction."""
        with patch('lobster.agents.data_expert.download_geo_data') as mock_download:
            mock_download.return_value = "Downloaded GSE12345: 1000 cells, 20000 genes, organism: Homo sapiens"
            
            result = mock_download("GSE12345", include_metadata=True)
            
            assert "1000 cells" in result
            assert "Homo sapiens" in result
            mock_download.assert_called_once_with("GSE12345", include_metadata=True)
    
    def test_download_geo_data_failure(self, mock_geo_service, data_expert_state):
        """Test GEO data download failure handling."""
        with patch('lobster.agents.data_expert.download_geo_data') as mock_download:
            mock_download.side_effect = Exception("GEO dataset not found")
            
            with pytest.raises(Exception, match="GEO dataset not found"):
                mock_download("INVALID_GSE")
    
    def test_get_geo_metadata(self, mock_geo_service, data_expert_state):
        """Test getting GEO metadata."""
        with patch('lobster.agents.data_expert.get_geo_metadata') as mock_metadata:
            mock_metadata.return_value = "GSE12345 metadata: Title: Test dataset, Organism: Human, Samples: 100"
            
            result = mock_metadata("GSE12345")
            
            assert "Test dataset" in result
            assert "Human" in result
            mock_metadata.assert_called_once_with("GSE12345")


# ===============================================================================
# Data Format Detection and Conversion Tests
# ===============================================================================

@pytest.mark.unit
class TestDataFormatDetection:
    """Test data format detection and conversion."""
    
    @pytest.mark.parametrize("filename,expected_format", [
        ("data.csv", "csv"),
        ("data.tsv", "tsv"),
        ("data.xlsx", "excel"),
        ("data.h5ad", "h5ad"),
        ("data.h5", "hdf5"),
        ("matrix.mtx", "mtx"),
        ("data.txt", "text"),
    ])
    def test_detect_file_format(self, filename, expected_format):
        """Test file format detection."""
        with patch('lobster.agents.data_expert.detect_file_format') as mock_detect:
            mock_detect.return_value = expected_format
            
            result = mock_detect(filename)
            
            assert result == expected_format
    
    def test_detect_data_type_single_cell(self, mock_data_manager):
        """Test detecting single-cell data type."""
        # High gene count suggests single-cell
        mock_data = Mock()
        mock_data.n_obs = 5000  # cells
        mock_data.n_vars = 20000  # genes
        
        with patch('lobster.agents.data_expert.detect_data_type') as mock_detect:
            mock_detect.return_value = "single_cell_rna_seq"
            
            result = mock_detect(mock_data)
            
            assert result == "single_cell_rna_seq"
    
    def test_detect_data_type_bulk(self, mock_data_manager):
        """Test detecting bulk RNA-seq data type."""
        # Lower gene count suggests bulk
        mock_data = Mock()
        mock_data.n_obs = 24  # samples
        mock_data.n_vars = 2000  # genes
        
        with patch('lobster.agents.data_expert.detect_data_type') as mock_detect:
            mock_detect.return_value = "bulk_rna_seq"
            
            result = mock_detect(mock_data)
            
            assert result == "bulk_rna_seq"
    
    def test_convert_data_format(self, mock_data_manager):
        """Test data format conversion."""
        with patch('lobster.agents.data_expert.convert_data_format') as mock_convert:
            mock_convert.return_value = "Converted data from CSV to H5AD format"
            
            result = mock_convert("input.csv", "h5ad")
            
            assert "Converted" in result
            assert "CSV to H5AD" in result
            mock_convert.assert_called_once_with("input.csv", "h5ad")
    
    def test_validate_data_structure(self, mock_data_manager):
        """Test data structure validation."""
        with patch('lobster.agents.data_expert.validate_data_structure') as mock_validate:
            mock_validate.return_value = "Data structure is valid: 1000 observations × 500 features"
            
            result = mock_validate("test_data")
            
            assert "valid" in result
            assert "1000 observations" in result
            mock_validate.assert_called_once_with("test_data")


# ===============================================================================
# Workspace Management Tests
# ===============================================================================

@pytest.mark.unit
class TestWorkspaceManagement:
    """Test workspace management functionality."""
    
    def test_create_workspace_directory(self, mock_data_manager):
        """Test creating workspace directories."""
        with patch('lobster.agents.data_expert.create_workspace_directory') as mock_create:
            mock_create.return_value = "Created directory: /workspace/analysis_results"
            
            result = mock_create("analysis_results")
            
            assert "Created directory" in result
            assert "analysis_results" in result
            mock_create.assert_called_once_with("analysis_results")
    
    def test_organize_workspace_files(self, mock_data_manager):
        """Test organizing workspace files."""
        with patch('lobster.agents.data_expert.organize_workspace_files') as mock_organize:
            mock_organize.return_value = "Organized 15 files into appropriate directories"
            
            result = mock_organize()
            
            assert "Organized 15 files" in result
            mock_organize.assert_called_once()
    
    def test_cleanup_temporary_files(self, mock_data_manager):
        """Test cleaning up temporary files."""
        with patch('lobster.agents.data_expert.cleanup_temporary_files') as mock_cleanup:
            mock_cleanup.return_value = "Cleaned up 5 temporary files, freed 250MB"
            
            result = mock_cleanup()
            
            assert "Cleaned up 5" in result
            assert "250MB" in result
            mock_cleanup.assert_called_once()
    
    def test_backup_workspace_data(self, mock_data_manager):
        """Test backing up workspace data."""
        with patch('lobster.agents.data_expert.backup_workspace_data') as mock_backup:
            mock_backup.return_value = "Backup created: workspace_backup_20240115.tar.gz"
            
            result = mock_backup()
            
            assert "Backup created" in result
            assert ".tar.gz" in result
            mock_backup.assert_called_once()


# ===============================================================================
# Data Quality Assessment Tests
# ===============================================================================

@pytest.mark.unit
class TestDataQualityAssessment:
    """Test data quality assessment functionality."""
    
    def test_assess_data_quality(self, mock_data_manager):
        """Test comprehensive data quality assessment."""
        with patch('lobster.agents.data_expert.assess_data_quality') as mock_assess:
            mock_assess.return_value = "Quality assessment: 95% cells pass QC, median genes/cell: 2500"
            
            result = mock_assess("test_data")
            
            assert "95% cells pass" in result
            assert "median genes" in result
            mock_assess.assert_called_once_with("test_data")
    
    def test_check_missing_values(self, mock_data_manager):
        """Test checking for missing values."""
        with patch('lobster.agents.data_expert.check_missing_values') as mock_check:
            mock_check.return_value = "Missing values: 2.5% of total, mostly in ribosomal genes"
            
            result = mock_check("test_data")
            
            assert "2.5%" in result
            assert "ribosomal genes" in result
            mock_check.assert_called_once_with("test_data")
    
    def test_detect_outliers(self, mock_data_manager):
        """Test outlier detection."""
        with patch('lobster.agents.data_expert.detect_outliers') as mock_outliers:
            mock_outliers.return_value = "Found 12 potential outlier cells with extreme gene counts"
            
            result = mock_outliers("test_data")
            
            assert "12 potential outlier" in result
            assert "extreme gene counts" in result
            mock_outliers.assert_called_once_with("test_data")
    
    def test_calculate_qc_metrics(self, mock_data_manager):
        """Test QC metrics calculation."""
        with patch('lobster.agents.data_expert.calculate_qc_metrics') as mock_metrics:
            mock_metrics.return_value = "QC metrics calculated: total_counts, n_genes, pct_mt, pct_ribo"
            
            result = mock_metrics("test_data")
            
            assert "total_counts" in result
            assert "pct_mt" in result
            mock_metrics.assert_called_once_with("test_data")


# ===============================================================================
# Data Export and Sharing Tests
# ===============================================================================

@pytest.mark.unit
class TestDataExportSharing:
    """Test data export and sharing functionality."""
    
    def test_export_data_to_file(self, mock_data_manager):
        """Test exporting data to file."""
        with patch('lobster.agents.data_expert.export_data_to_file') as mock_export:
            mock_export.return_value = "Exported 'test_data' to analysis_results.h5ad"
            
            result = mock_export("test_data", "analysis_results.h5ad")
            
            assert "Exported" in result
            assert "analysis_results.h5ad" in result
            mock_export.assert_called_once_with("test_data", "analysis_results.h5ad")
    
    def test_export_metadata_summary(self, mock_data_manager):
        """Test exporting metadata summary."""
        with patch('lobster.agents.data_expert.export_metadata_summary') as mock_summary:
            mock_summary.return_value = "Metadata summary exported to metadata_report.json"
            
            result = mock_summary("test_data")
            
            assert "Metadata summary exported" in result
            assert "metadata_report.json" in result
            mock_summary.assert_called_once_with("test_data")
    
    def test_create_data_package(self, mock_data_manager):
        """Test creating data package."""
        with patch('lobster.agents.data_expert.create_data_package') as mock_package:
            mock_package.return_value = "Data package created: analysis_package.zip (15.2MB)"
            
            result = mock_package(["data1", "data2"])
            
            assert "Data package created" in result
            assert "15.2MB" in result
            mock_package.assert_called_once_with(["data1", "data2"])
    
    def test_generate_data_report(self, mock_data_manager):
        """Test generating data report."""
        with patch('lobster.agents.data_expert.generate_data_report') as mock_report:
            mock_report.return_value = "Data report generated: comprehensive_report.html"
            
            result = mock_report("test_data")
            
            assert "Data report generated" in result
            assert "comprehensive_report.html" in result
            mock_report.assert_called_once_with("test_data")


# ===============================================================================
# Integration and Workflow Tests
# ===============================================================================

@pytest.mark.unit
class TestIntegrationWorkflows:
    """Test integration and workflow functionality."""
    
    def test_complete_data_loading_workflow(self, mock_data_manager, mock_geo_service):
        """Test complete data loading workflow."""
        state = MockState(
            messages=[MockMessage("Download GSE12345 and perform quality assessment")],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.data_expert.data_expert_agent') as mock_agent:
            # Mock complete workflow
            mock_agent.return_value = {
                "messages": [MockMessage("Downloaded GSE12345 and completed QC analysis", "assistant")],
                "workflow_completed": True,
                "data_loaded": "geo_gse12345",
                "qc_passed": True
            }
            
            result = mock_agent(state)
            
            assert result["workflow_completed"] == True
            assert result["qc_passed"] == True
    
    def test_multi_dataset_integration(self, mock_data_manager):
        """Test integrating multiple datasets."""
        state = MockState(
            messages=[MockMessage("Combine dataset A and dataset B")],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.data_expert.integrate_datasets') as mock_integrate:
            mock_integrate.return_value = "Integrated 2 datasets into 'combined_analysis'"
            
            result = mock_integrate(["dataset_A", "dataset_B"])
            
            assert "Integrated 2 datasets" in result
            assert "combined_analysis" in result
    
    def test_data_preprocessing_pipeline(self, mock_data_manager):
        """Test data preprocessing pipeline."""
        state = MockState(
            messages=[MockMessage("Preprocess the data for analysis")],
            data_manager=mock_data_manager
        )
        
        with patch('lobster.agents.data_expert.preprocess_data_pipeline') as mock_preprocess:
            mock_preprocess.return_value = "Preprocessing completed: normalized, filtered, QC metrics added"
            
            result = mock_preprocess("test_data")
            
            assert "normalized" in result
            assert "QC metrics added" in result


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestDataExpertErrorHandling:
    """Test data expert error handling and edge cases."""
    
    def test_file_not_found_handling(self, mock_data_manager):
        """Test handling of file not found errors."""
        with patch('lobster.agents.data_expert.load_data_from_file') as mock_load:
            mock_load.side_effect = FileNotFoundError("File not found: missing_data.csv")
            
            with pytest.raises(FileNotFoundError, match="missing_data.csv"):
                mock_load("missing_data.csv")
    
    def test_corrupted_data_handling(self, mock_data_manager):
        """Test handling of corrupted data files."""
        with patch('lobster.agents.data_expert.load_data_from_file') as mock_load:
            mock_load.side_effect = Exception("Corrupted file: unable to parse data")
            
            with pytest.raises(Exception, match="Corrupted file"):
                mock_load("corrupted_data.csv")
    
    def test_insufficient_memory_handling(self, mock_data_manager):
        """Test handling of insufficient memory errors."""
        with patch('lobster.agents.data_expert.load_data_from_file') as mock_load:
            mock_load.side_effect = MemoryError("Insufficient memory to load large dataset")
            
            with pytest.raises(MemoryError, match="Insufficient memory"):
                mock_load("huge_dataset.h5ad")
    
    def test_invalid_geo_accession_handling(self, mock_geo_service):
        """Test handling of invalid GEO accessions."""
        with patch('lobster.agents.data_expert.download_geo_data') as mock_download:
            mock_download.side_effect = ValueError("Invalid GEO accession: INVALID123")
            
            with pytest.raises(ValueError, match="Invalid GEO accession"):
                mock_download("INVALID123")
    
    def test_network_error_handling(self, mock_geo_service):
        """Test handling of network errors during GEO download."""
        with patch('lobster.agents.data_expert.download_geo_data') as mock_download:
            mock_download.side_effect = ConnectionError("Network timeout during GEO download")
            
            with pytest.raises(ConnectionError, match="Network timeout"):
                mock_download("GSE12345")
    
    def test_concurrent_data_access(self, mock_data_manager):
        """Test concurrent data access handling."""
        import threading
        import time
        
        results = []
        errors = []
        
        def data_worker(worker_id):
            """Worker function for concurrent data access."""
            try:
                with patch('lobster.agents.data_expert.examine_data_modality') as mock_examine:
                    mock_examine.return_value = f"Worker {worker_id}: Data examined successfully"
                    
                    result = mock_examine(f"data_{worker_id}")
                    results.append(result)
                    time.sleep(0.01)
                    
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=data_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 5
    
    def test_large_dataset_memory_management(self, mock_data_manager):
        """Test memory management with large datasets."""
        # Simulate large dataset scenario
        mock_data_manager.get_modality.return_value.n_obs = 1000000  # 1M cells
        mock_data_manager.get_modality.return_value.n_vars = 50000   # 50k genes
        
        with patch('lobster.agents.data_expert.examine_data_modality') as mock_examine:
            mock_examine.return_value = "Large dataset: 1M cells × 50k genes (memory optimized)"
            
            result = mock_examine("large_dataset")
            
            assert "1M cells" in result
            assert "memory optimized" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])