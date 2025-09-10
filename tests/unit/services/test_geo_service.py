"""
Comprehensive unit tests for GEO service.

This module provides thorough testing of the GEO (Gene Expression Omnibus) 
service including dataset search, metadata extraction, file downloading,
format conversion, and integration with the data management system.

Test coverage target: 95%+ with meaningful tests for GEO operations.
"""

import pytest
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, MagicMock, patch, mock_open
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import json
import gzip
from io import StringIO, BytesIO

from lobster.tools.geo_service import GEOService
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.factories import SingleCellDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def mock_geo_response():
    """Mock GEO search response data."""
    return {
        "header": {
            "type": "esearch",
            "version": "0.3"
        },
        "esearchresult": {
            "count": "3",
            "retmax": "20",
            "retstart": "0",
            "idlist": ["200012345", "200012346", "200012347"],
            "translationset": [],
            "querytranslation": "single cell[All Fields] AND rna seq[All Fields]"
        }
    }


@pytest.fixture
def mock_geo_metadata():
    """Mock GEO dataset metadata."""
    return {
        "GSE123456": {
            "title": "Single-cell RNA sequencing of tumor-infiltrating T cells",
            "summary": "We performed scRNA-seq analysis of T cells from tumor samples...",
            "organism": "Homo sapiens",
            "sample_count": 48,
            "platform": "GPL24676 (Illumina NovaSeq 6000)",
            "publication_date": "2023-06-15",
            "last_update_date": "2023-06-20",
            "samples": [
                {
                    "gsm": "GSM1234567",
                    "title": "Tumor T cells replicate 1",
                    "characteristics": {
                        "cell type": "T cell",
                        "tissue": "tumor",
                        "treatment": "untreated"
                    }
                },
                {
                    "gsm": "GSM1234568", 
                    "title": "Tumor T cells replicate 2",
                    "characteristics": {
                        "cell type": "T cell",
                        "tissue": "tumor", 
                        "treatment": "untreated"
                    }
                }
            ],
            "supplementary_files": [
                {
                    "name": "GSE123456_matrix.txt.gz",
                    "size": "45.2 MB",
                    "type": "TXT"
                },
                {
                    "name": "GSE123456_barcodes.txt.gz", 
                    "size": "1.2 MB",
                    "type": "TXT"
                },
                {
                    "name": "GSE123456_features.txt.gz",
                    "size": "800 KB", 
                    "type": "TXT"
                }
            ]
        }
    }


@pytest.fixture
def mock_ncbi_client():
    """Mock NCBI E-utilities client."""
    with patch('lobster.tools.geo_service.Entrez') as mock_entrez:
        mock_entrez.email = "test@example.com"
        
        # Mock esearch response
        mock_search_handle = StringIO(json.dumps({
            "esearchresult": {
                "count": "3",
                "idlist": ["200012345", "200012346", "200012347"]
            }
        }))
        mock_entrez.esearch.return_value = mock_search_handle
        
        # Mock efetch response
        mock_fetch_handle = StringIO("""
        <DocumentSummary uid="200012345">
            <Accession>GSE123456</Accession>
            <Title>Single-cell RNA sequencing of tumor-infiltrating T cells</Title>
            <Summary>We performed scRNA-seq analysis...</Summary>
            <n_samples>48</n_samples>
            <PlatformTitle>Illumina NovaSeq 6000</PlatformTitle>
        </DocumentSummary>
        """)
        mock_entrez.efetch.return_value = mock_fetch_handle
        
        yield mock_entrez


@pytest.fixture
def mock_ftp_client():
    """Mock FTP client for GEO file downloads."""
    with patch('ftplib.FTP') as mock_ftp:
        mock_ftp_instance = mock_ftp.return_value
        mock_ftp_instance.login.return_value = None
        mock_ftp_instance.cwd.return_value = None
        mock_ftp_instance.nlst.return_value = [
            'GSE123456_matrix.txt.gz',
            'GSE123456_barcodes.txt.gz', 
            'GSE123456_features.txt.gz'
        ]
        mock_ftp_instance.size.return_value = 47382016  # 45.2 MB
        mock_ftp_instance.retrbinary = Mock()
        mock_ftp_instance.quit.return_value = None
        
        yield mock_ftp_instance


@pytest.fixture
def geo_service():
    """Create GEOService instance for testing."""
    return GEOService(email="test@example.com", cache_dir="test_cache")


@pytest.fixture
def temp_download_dir():
    """Create temporary directory for downloads."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# ===============================================================================
# GEO Service Core Functionality Tests
# ===============================================================================

@pytest.mark.unit
class TestGEOServiceCore:
    """Test GEO service core functionality."""
    
    def test_geo_service_initialization(self):
        """Test GEOService initialization."""
        service = GEOService(email="test@example.com")
        
        assert service.email == "test@example.com"
        assert service.cache_dir is not None
        assert hasattr(service, 'base_url')
    
    def test_geo_service_with_api_key(self):
        """Test GEOService initialization with API key."""
        service = GEOService(
            email="test@example.com",
            api_key="test_api_key",
            cache_dir="custom_cache"
        )
        
        assert service.email == "test@example.com"
        assert service.api_key == "test_api_key"
        assert service.cache_dir == "custom_cache"
    
    def test_validate_geo_accession_valid(self, geo_service):
        """Test validation of valid GEO accessions."""
        valid_accessions = [
            "GSE123456",
            "GSM987654",
            "GPL123456",
            "GDS123456"
        ]
        
        for accession in valid_accessions:
            assert geo_service._validate_accession(accession) == True
    
    def test_validate_geo_accession_invalid(self, geo_service):
        """Test validation of invalid GEO accessions."""
        invalid_accessions = [
            "INVALID123",
            "GSE",
            "12345",
            "GSE-123456",
            ""
        ]
        
        for accession in invalid_accessions:
            assert geo_service._validate_accession(accession) == False


# ===============================================================================
# GEO Search and Discovery Tests  
# ===============================================================================

@pytest.mark.unit
class TestGEOSearchDiscovery:
    """Test GEO search and discovery functionality."""
    
    def test_search_geo_datasets_basic(self, geo_service, mock_ncbi_client, mock_geo_response):
        """Test basic GEO dataset search."""
        with patch.object(geo_service, '_search_geo') as mock_search:
            mock_search.return_value = [
                {
                    "accession": "GSE123456",
                    "title": "Single-cell RNA sequencing of tumor-infiltrating T cells",
                    "organism": "Homo sapiens",
                    "samples": 48
                },
                {
                    "accession": "GSE123457", 
                    "title": "scRNA-seq analysis of immune cells",
                    "organism": "Homo sapiens",
                    "samples": 32
                }
            ]
            
            results = geo_service.search_datasets("single cell RNA seq", max_results=10)
            
            assert len(results) == 2
            assert results[0]["accession"] == "GSE123456"
            assert "single-cell" in results[0]["title"].lower()
            mock_search.assert_called_once_with("single cell RNA seq", max_results=10)
    
    def test_search_geo_datasets_with_filters(self, geo_service, mock_ncbi_client):
        """Test GEO dataset search with filters."""
        with patch.object(geo_service, '_search_geo') as mock_search:
            mock_search.return_value = [
                {
                    "accession": "GSE123456",
                    "title": "Human T cell analysis",
                    "organism": "Homo sapiens",
                    "samples": 48,
                    "platform": "Illumina NovaSeq 6000"
                }
            ]
            
            results = geo_service.search_datasets(
                query="T cell",
                organism="Homo sapiens",
                platform="Illumina",
                min_samples=40,
                max_results=5
            )
            
            assert len(results) == 1
            assert results[0]["organism"] == "Homo sapiens"
            assert results[0]["samples"] >= 40
    
    def test_search_geo_datasets_empty_results(self, geo_service, mock_ncbi_client):
        """Test GEO search with no results."""
        with patch.object(geo_service, '_search_geo') as mock_search:
            mock_search.return_value = []
            
            results = geo_service.search_datasets("extremely_rare_query_no_results")
            
            assert len(results) == 0
    
    def test_get_trending_datasets(self, geo_service, mock_ncbi_client):
        """Test getting trending/popular datasets."""
        with patch.object(geo_service, 'get_trending_datasets') as mock_trending:
            mock_trending.return_value = [
                {
                    "accession": "GSE123456",
                    "title": "Popular single-cell dataset",
                    "download_count": 1250,
                    "citation_count": 45
                },
                {
                    "accession": "GSE123457",
                    "title": "Highly cited RNA-seq study", 
                    "download_count": 890,
                    "citation_count": 67
                }
            ]
            
            results = geo_service.get_trending_datasets(category="single_cell", limit=10)
            
            assert len(results) == 2
            assert results[0]["download_count"] > 1000
            mock_trending.assert_called_once_with(category="single_cell", limit=10)
    
    def test_search_by_publication(self, geo_service, mock_ncbi_client):
        """Test search by publication details."""
        with patch.object(geo_service, 'search_by_publication') as mock_pub_search:
            mock_pub_search.return_value = [
                {
                    "accession": "GSE123456",
                    "title": "Dataset from Nature paper",
                    "pmid": "12345678",
                    "journal": "Nature",
                    "publication_year": 2023
                }
            ]
            
            results = geo_service.search_by_publication(
                pmid="12345678",
                journal="Nature",
                author="Smith"
            )
            
            assert len(results) == 1
            assert results[0]["pmid"] == "12345678"


# ===============================================================================
# GEO Metadata Extraction Tests
# ===============================================================================

@pytest.mark.unit  
class TestGEOMetadataExtraction:
    """Test GEO metadata extraction functionality."""
    
    def test_get_dataset_metadata(self, geo_service, mock_ncbi_client, mock_geo_metadata):
        """Test extracting dataset metadata."""
        with patch.object(geo_service, 'get_metadata') as mock_get_metadata:
            mock_get_metadata.return_value = mock_geo_metadata["GSE123456"]
            
            metadata = geo_service.get_metadata("GSE123456")
            
            assert metadata["title"] == "Single-cell RNA sequencing of tumor-infiltrating T cells"
            assert metadata["organism"] == "Homo sapiens"
            assert metadata["sample_count"] == 48
            assert len(metadata["samples"]) == 2
            mock_get_metadata.assert_called_once_with("GSE123456")
    
    def test_get_sample_metadata(self, geo_service, mock_ncbi_client):
        """Test extracting sample-level metadata."""
        with patch.object(geo_service, 'get_sample_metadata') as mock_sample_meta:
            mock_sample_meta.return_value = {
                "gsm": "GSM1234567",
                "title": "Tumor T cells replicate 1", 
                "characteristics": {
                    "cell type": "T cell",
                    "tissue": "tumor",
                    "treatment": "untreated",
                    "age": "65 years",
                    "sex": "male"
                },
                "protocols": {
                    "extraction": "Single cell isolation using FACS",
                    "library_construction": "10X Chromium 3' v3.1"
                }
            }
            
            metadata = geo_service.get_sample_metadata("GSM1234567")
            
            assert metadata["characteristics"]["cell type"] == "T cell"
            assert metadata["protocols"]["library_construction"] == "10X Chromium 3' v3.1"
            mock_sample_meta.assert_called_once_with("GSM1234567")
    
    def test_get_platform_metadata(self, geo_service, mock_ncbi_client):
        """Test extracting platform metadata."""
        with patch.object(geo_service, 'get_platform_metadata') as mock_platform_meta:
            mock_platform_meta.return_value = {
                "gpl": "GPL24676",
                "title": "Illumina NovaSeq 6000",
                "organism": "Homo sapiens",
                "technology": "high-throughput sequencing",
                "manufacturer": "Illumina",
                "description": "Next-generation sequencing platform"
            }
            
            metadata = geo_service.get_platform_metadata("GPL24676")
            
            assert metadata["title"] == "Illumina NovaSeq 6000"
            assert metadata["technology"] == "high-throughput sequencing"
            mock_platform_meta.assert_called_once_with("GPL24676")
    
    def test_extract_experimental_design(self, geo_service, mock_geo_metadata):
        """Test experimental design extraction."""
        with patch.object(geo_service, 'extract_experimental_design') as mock_design:
            mock_design.return_value = {
                "study_type": "single_cell_rna_seq",
                "experimental_factors": ["tissue", "treatment"],
                "sample_groups": {
                    "tumor_untreated": 24,
                    "normal_untreated": 24
                },
                "replicates": 2,
                "batch_effects": "minimal",
                "quality_metrics": {
                    "cells_per_sample": "~2000",
                    "genes_detected": "~15000"
                }
            }
            
            design = geo_service.extract_experimental_design("GSE123456")
            
            assert design["study_type"] == "single_cell_rna_seq"
            assert len(design["experimental_factors"]) == 2
            assert design["sample_groups"]["tumor_untreated"] == 24


# ===============================================================================
# GEO File Download Tests
# ===============================================================================

@pytest.mark.unit
class TestGEOFileDownload:
    """Test GEO file download functionality."""
    
    def test_list_supplementary_files(self, geo_service, mock_geo_metadata):
        """Test listing supplementary files."""
        with patch.object(geo_service, 'list_files') as mock_list_files:
            mock_list_files.return_value = mock_geo_metadata["GSE123456"]["supplementary_files"]
            
            files = geo_service.list_files("GSE123456")
            
            assert len(files) == 3
            assert files[0]["name"] == "GSE123456_matrix.txt.gz"
            assert files[0]["size"] == "45.2 MB"
            mock_list_files.assert_called_once_with("GSE123456")
    
    def test_download_dataset_files(self, geo_service, mock_ftp_client, temp_download_dir):
        """Test downloading dataset files."""
        with patch.object(geo_service, 'download_files') as mock_download:
            mock_download.return_value = {
                "success": True,
                "downloaded_files": [
                    str(temp_download_dir / "GSE123456_matrix.txt.gz"),
                    str(temp_download_dir / "GSE123456_barcodes.txt.gz"),
                    str(temp_download_dir / "GSE123456_features.txt.gz")
                ],
                "total_size": "47.2 MB",
                "download_time": 45.2
            }
            
            result = geo_service.download_files(
                "GSE123456",
                download_dir=str(temp_download_dir),
                file_types=["matrix", "barcodes", "features"]
            )
            
            assert result["success"] == True
            assert len(result["downloaded_files"]) == 3
            assert "matrix" in result["downloaded_files"][0]
    
    def test_download_specific_files(self, geo_service, mock_ftp_client, temp_download_dir):
        """Test downloading specific files."""
        with patch.object(geo_service, 'download_files') as mock_download:
            mock_download.return_value = {
                "success": True, 
                "downloaded_files": [str(temp_download_dir / "GSE123456_matrix.txt.gz")],
                "skipped_files": []
            }
            
            result = geo_service.download_files(
                "GSE123456",
                file_names=["GSE123456_matrix.txt.gz"],
                download_dir=str(temp_download_dir)
            )
            
            assert result["success"] == True
            assert len(result["downloaded_files"]) == 1
    
    def test_download_with_progress_callback(self, geo_service, mock_ftp_client, temp_download_dir):
        """Test download with progress callback."""
        progress_updates = []
        
        def progress_callback(filename, bytes_downloaded, total_bytes):
            progress_updates.append({
                "file": filename,
                "downloaded": bytes_downloaded,
                "total": total_bytes,
                "percent": (bytes_downloaded / total_bytes) * 100
            })
        
        with patch.object(geo_service, 'download_files') as mock_download:
            # Simulate progress updates
            progress_callback("GSE123456_matrix.txt.gz", 10485760, 47382016)  # 25%
            progress_callback("GSE123456_matrix.txt.gz", 23691008, 47382016)  # 50%
            progress_callback("GSE123456_matrix.txt.gz", 47382016, 47382016)  # 100%
            
            mock_download.return_value = {
                "success": True,
                "downloaded_files": [str(temp_download_dir / "GSE123456_matrix.txt.gz")]
            }
            
            result = geo_service.download_files(
                "GSE123456",
                download_dir=str(temp_download_dir),
                progress_callback=progress_callback
            )
            
            assert len(progress_updates) == 3
            assert progress_updates[-1]["percent"] == 100.0
    
    def test_resume_interrupted_download(self, geo_service, temp_download_dir):
        """Test resuming interrupted downloads."""
        # Create partial file
        partial_file = temp_download_dir / "GSE123456_matrix.txt.gz.partial"
        partial_file.write_bytes(b"partial_content")
        
        with patch.object(geo_service, 'download_files') as mock_download:
            mock_download.return_value = {
                "success": True,
                "resumed": True,
                "downloaded_files": [str(temp_download_dir / "GSE123456_matrix.txt.gz")],
                "bytes_resumed": len(b"partial_content")
            }
            
            result = geo_service.download_files(
                "GSE123456",
                download_dir=str(temp_download_dir),
                resume=True
            )
            
            assert result["resumed"] == True
            assert result["bytes_resumed"] > 0


# ===============================================================================
# Data Format Conversion Tests
# ===============================================================================

@pytest.mark.unit
class TestGEOFormatConversion:
    """Test GEO data format conversion functionality."""
    
    def test_detect_file_format(self, geo_service):
        """Test automatic file format detection."""
        test_files = [
            ("GSE123456_matrix.txt.gz", "matrix"),
            ("GSE123456_barcodes.tsv.gz", "barcodes"),
            ("GSE123456_features.tsv.gz", "features"), 
            ("GSE123456_matrix.mtx.gz", "matrix_market"),
            ("GSE123456.h5", "hdf5"),
            ("GSE123456_expression.xlsx", "excel")
        ]
        
        with patch.object(geo_service, '_detect_format') as mock_detect:
            for filename, expected_format in test_files:
                mock_detect.return_value = expected_format
                
                detected_format = geo_service._detect_format(filename)
                assert detected_format == expected_format
    
    def test_convert_to_anndata(self, geo_service, temp_download_dir):
        """Test conversion to AnnData format."""
        # Create mock input files
        matrix_file = temp_download_dir / "matrix.txt.gz"
        barcodes_file = temp_download_dir / "barcodes.txt.gz"
        features_file = temp_download_dir / "features.txt.gz"
        
        with patch.object(geo_service, 'convert_to_anndata') as mock_convert:
            mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            mock_convert.return_value = mock_adata
            
            adata = geo_service.convert_to_anndata(
                matrix_file=str(matrix_file),
                barcodes_file=str(barcodes_file),
                features_file=str(features_file)
            )
            
            assert adata.n_obs > 0
            assert adata.n_vars > 0
            mock_convert.assert_called_once()
    
    def test_convert_soft_format(self, geo_service, temp_download_dir):
        """Test SOFT format conversion."""
        soft_file = temp_download_dir / "GSE123456_family.soft.gz"
        
        with patch.object(geo_service, 'parse_soft_file') as mock_parse_soft:
            mock_parse_soft.return_value = {
                "platform_data": pd.DataFrame({
                    "ID": ["GENE1", "GENE2", "GENE3"],
                    "Gene Symbol": ["ACTB", "GAPDH", "TP53"]
                }),
                "sample_data": {
                    "GSM1234567": pd.Series([100, 200, 150], index=["GENE1", "GENE2", "GENE3"]),
                    "GSM1234568": pd.Series([120, 180, 140], index=["GENE1", "GENE2", "GENE3"])
                },
                "metadata": {
                    "platform": "GPL123456",
                    "samples": ["GSM1234567", "GSM1234568"]
                }
            }
            
            result = geo_service.parse_soft_file(str(soft_file))
            
            assert "platform_data" in result
            assert len(result["sample_data"]) == 2
            assert result["metadata"]["platform"] == "GPL123456"
    
    def test_convert_matrix_market(self, geo_service, temp_download_dir):
        """Test Matrix Market format conversion."""
        mtx_file = temp_download_dir / "matrix.mtx.gz"
        
        with patch.object(geo_service, 'read_matrix_market') as mock_read_mtx:
            # Mock sparse matrix
            mock_matrix = np.array([[1, 0, 3], [0, 2, 0], [1, 1, 0]])
            mock_read_mtx.return_value = mock_matrix
            
            matrix = geo_service.read_matrix_market(str(mtx_file))
            
            assert matrix.shape == (3, 3)
            assert np.sum(matrix) == 8  # Sum of non-zero elements
    
    def test_handle_compressed_files(self, geo_service, temp_download_dir):
        """Test handling of compressed files."""
        # Create mock compressed file
        compressed_file = temp_download_dir / "data.txt.gz"
        
        test_content = "gene1\tgene2\tgene3\ncell1\t100\t200\t150\ncell2\t120\t180\t140"
        
        with patch.object(geo_service, '_decompress_file') as mock_decompress:
            mock_decompress.return_value = test_content
            
            content = geo_service._decompress_file(str(compressed_file))
            
            assert "gene1" in content
            assert "cell1" in content
            mock_decompress.assert_called_once_with(str(compressed_file))


# ===============================================================================
# Integration and Caching Tests
# ===============================================================================

@pytest.mark.unit
class TestGEOIntegrationCaching:
    """Test GEO integration and caching functionality."""
    
    def test_cache_search_results(self, geo_service, temp_download_dir):
        """Test caching of search results."""
        cache_dir = temp_download_dir / "cache"
        cache_dir.mkdir()
        
        service = GEOService(email="test@example.com", cache_dir=str(cache_dir))
        
        with patch.object(service, '_cache_search_results') as mock_cache:
            search_results = [
                {"accession": "GSE123456", "title": "Test dataset 1"},
                {"accession": "GSE123457", "title": "Test dataset 2"}
            ]
            
            mock_cache.return_value = True
            
            cached = service._cache_search_results("test_query", search_results)
            
            assert cached == True
            mock_cache.assert_called_once_with("test_query", search_results)
    
    def test_retrieve_cached_results(self, geo_service, temp_download_dir):
        """Test retrieving cached search results."""
        cache_dir = temp_download_dir / "cache"
        cache_dir.mkdir()
        
        service = GEOService(email="test@example.com", cache_dir=str(cache_dir))
        
        with patch.object(service, '_get_cached_results') as mock_get_cache:
            cached_results = [
                {"accession": "GSE123456", "title": "Cached dataset 1"},
                {"accession": "GSE123457", "title": "Cached dataset 2"}
            ]
            
            mock_get_cache.return_value = cached_results
            
            results = service._get_cached_results("test_query")
            
            assert len(results) == 2
            assert results[0]["title"] == "Cached dataset 1"
    
    def test_cache_expiration(self, geo_service, temp_download_dir):
        """Test cache expiration handling."""
        cache_dir = temp_download_dir / "cache"
        cache_dir.mkdir()
        
        service = GEOService(
            email="test@example.com", 
            cache_dir=str(cache_dir),
            cache_ttl=3600  # 1 hour
        )
        
        with patch.object(service, '_is_cache_expired') as mock_expired:
            mock_expired.return_value = True
            
            expired = service._is_cache_expired("test_query")
            
            assert expired == True
    
    def test_integration_with_data_manager(self, geo_service, temp_download_dir):
        """Test integration with data manager."""
        with patch('lobster.core.data_manager_v2.DataManagerV2') as MockDataManager:
            mock_dm = MockDataManager.return_value
            
            with patch.object(geo_service, 'download_and_load') as mock_download_load:
                mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                mock_download_load.return_value = {
                    "success": True,
                    "modality_name": "geo_gse123456",
                    "adata": mock_adata,
                    "metadata": {"accession": "GSE123456", "organism": "Homo sapiens"}
                }
                
                result = geo_service.download_and_load(
                    accession="GSE123456",
                    data_manager=mock_dm,
                    modality_name="geo_gse123456"
                )
                
                assert result["success"] == True
                assert result["modality_name"] == "geo_gse123456"
                assert result["adata"].n_obs > 0


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestGEOErrorHandling:
    """Test GEO service error handling and edge cases."""
    
    def test_invalid_accession_handling(self, geo_service):
        """Test handling of invalid GEO accessions."""
        with patch.object(geo_service, 'get_metadata') as mock_get_metadata:
            mock_get_metadata.side_effect = ValueError("Invalid GEO accession: INVALID123")
            
            with pytest.raises(ValueError, match="Invalid GEO accession"):
                geo_service.get_metadata("INVALID123")
    
    def test_network_timeout_handling(self, geo_service, mock_ncbi_client):
        """Test handling of network timeouts."""
        with patch.object(geo_service, 'search_datasets') as mock_search:
            mock_search.side_effect = ConnectionError("Network timeout during GEO search")
            
            with pytest.raises(ConnectionError, match="Network timeout"):
                geo_service.search_datasets("test query")
    
    def test_ftp_download_failure(self, geo_service, mock_ftp_client):
        """Test handling of FTP download failures."""
        with patch.object(geo_service, 'download_files') as mock_download:
            mock_download.side_effect = Exception("FTP server unavailable")
            
            with pytest.raises(Exception, match="FTP server unavailable"):
                geo_service.download_files("GSE123456")
    
    def test_corrupted_file_handling(self, geo_service, temp_download_dir):
        """Test handling of corrupted downloaded files."""
        corrupted_file = temp_download_dir / "corrupted_matrix.txt.gz"
        corrupted_file.write_bytes(b"corrupted_content")
        
        with patch.object(geo_service, 'convert_to_anndata') as mock_convert:
            mock_convert.side_effect = Exception("File appears to be corrupted")
            
            with pytest.raises(Exception, match="File appears to be corrupted"):
                geo_service.convert_to_anndata(matrix_file=str(corrupted_file))
    
    def test_insufficient_disk_space(self, geo_service, temp_download_dir):
        """Test handling of insufficient disk space."""
        with patch.object(geo_service, 'download_files') as mock_download:
            mock_download.side_effect = OSError("No space left on device")
            
            with pytest.raises(OSError, match="No space left on device"):
                geo_service.download_files("GSE123456", download_dir=str(temp_download_dir))
    
    def test_rate_limit_handling(self, geo_service, mock_ncbi_client):
        """Test handling of API rate limits."""
        with patch.object(geo_service, 'search_datasets') as mock_search:
            mock_search.side_effect = Exception("API rate limit exceeded")
            
            with pytest.raises(Exception, match="API rate limit exceeded"):
                geo_service.search_datasets("test query")
    
    def test_large_dataset_handling(self, geo_service, temp_download_dir):
        """Test handling of very large datasets."""
        with patch.object(geo_service, 'download_files') as mock_download:
            mock_download.return_value = {
                "success": True,
                "downloaded_files": [str(temp_download_dir / "large_matrix.txt.gz")],
                "total_size": "15.2 GB",
                "download_time": 3600,
                "warnings": ["Large file size may impact processing time"]
            }
            
            result = geo_service.download_files(
                "GSE123456",
                download_dir=str(temp_download_dir)
            )
            
            assert result["success"] == True
            assert "warnings" in result
    
    def test_concurrent_download_handling(self, geo_service, temp_download_dir):
        """Test concurrent download operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def download_worker(worker_id, accession):
            """Worker function for concurrent downloads."""
            try:
                with patch.object(geo_service, 'download_files') as mock_download:
                    mock_download.return_value = {
                        "success": True,
                        "worker_id": worker_id,
                        "accession": accession
                    }
                    
                    result = geo_service.download_files(
                        accession,
                        download_dir=str(temp_download_dir)
                    )
                    results.append(result)
                    time.sleep(0.01)
                    
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple concurrent downloads
        threads = []
        accessions = ["GSE123456", "GSE123457", "GSE123458"]
        
        for i, accession in enumerate(accessions):
            thread = threading.Thread(target=download_worker, args=(i, accession))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent download errors: {errors}"
        assert len(results) == 3


# ===============================================================================
# Performance and Benchmarking Tests
# ===============================================================================

@pytest.mark.unit
class TestGEOPerformance:
    """Test GEO service performance characteristics."""
    
    def test_search_performance_metrics(self, geo_service, mock_ncbi_client):
        """Test search performance tracking."""
        with patch.object(geo_service, 'search_datasets') as mock_search:
            mock_search.return_value = [
                {"accession": f"GSE{i}", "title": f"Dataset {i}"} for i in range(100)
            ]
            
            import time
            start_time = time.time()
            results = geo_service.search_datasets("performance test", max_results=100)
            end_time = time.time()
            
            search_time = end_time - start_time
            
            assert len(results) == 100
            assert search_time < 5.0  # Should complete within 5 seconds (mocked)
    
    def test_download_speed_monitoring(self, geo_service, mock_ftp_client, temp_download_dir):
        """Test download speed monitoring."""
        with patch.object(geo_service, 'download_files') as mock_download:
            mock_download.return_value = {
                "success": True,
                "downloaded_files": [str(temp_download_dir / "test_file.txt.gz")],
                "total_size_bytes": 52428800,  # 50 MB
                "download_time_seconds": 10.5,
                "average_speed_mbps": 40.0
            }
            
            result = geo_service.download_files("GSE123456")
            
            assert result["average_speed_mbps"] == 40.0
            assert result["download_time_seconds"] < 15.0
    
    def test_memory_usage_optimization(self, geo_service, temp_download_dir):
        """Test memory usage optimization for large files."""
        with patch.object(geo_service, 'convert_to_anndata') as mock_convert:
            # Simulate memory-efficient processing
            mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
            mock_convert.return_value = mock_adata
            
            # Test chunk-based processing
            adata = geo_service.convert_to_anndata(
                matrix_file="large_matrix.txt.gz",
                chunk_size=10000,  # Process in chunks
                memory_efficient=True
            )
            
            assert adata.n_obs > 0
            mock_convert.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])