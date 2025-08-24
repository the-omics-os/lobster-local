"""
Comprehensive test suite for Lobster AI - Multi-Agent Bioinformatics Analysis System.

This test suite covers all major functionality including data management,
GEO downloads, quality assessment, clustering, agent client, and analysis services.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import urllib
import scanpy as sc

# Import modules to test - Updated for new structure
from lobster.core import DataManager
from lobster.tools.geo_service import GEOService
from lobster.tools import QualityService
from lobster.tools import ClusteringService
from lobster.tools import BulkRNASeqService
from lobster.tools import EnhancedSingleCellService
# from lobster.tools import FileUploadService
from lobster.core import AgentClient
from lobster.tools.pubmed_service import PubMedService
from lobster.utils import get_logger


logger = get_logger(__name__)


# Test Data Fixtures
@pytest.fixture
def sample_expression_data():
    """Create sample expression data for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Create realistic single-cell expression data
    n_cells = 500
    n_genes = 1000
    
    # Generate sparse expression data (typical of single-cell)
    data = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))
    
    # Add some zero inflation (typical of single-cell)
    zero_mask = np.random.random((n_cells, n_genes)) < 0.7
    data[zero_mask] = 0
    
    # Create DataFrame
    cell_names = [f"Cell_{i:04d}" for i in range(n_cells)]
    
    # Create gene names with biologically relevant markers
    known_genes = ['CD3D', 'CD3E', 'CD8A', 'CD4', 'CD19', 'MS4A1', 'GNLY', 'NKG7', 
                   'CD14', 'LYZ', 'FCGR3A', 'MS4A7', 'CST3', 'GZMB', 'NKG7', 
                   'MT-CO1', 'MT-CO2', 'MT-ATP6', 'FCER1A', 'CST3', 'TCL1A']
    
    # Ensure we have enough known genes for the test
    gene_names = known_genes.copy()
    # Fill the rest with generic gene names
    for i in range(n_genes - len(known_genes)):
        gene_names.append(f"GENE_{i:04d}")
    
    df = pd.DataFrame(data, index=cell_names, columns=gene_names)
    return df

@pytest.fixture
def sample_bulk_data():
    """Create sample bulk RNA-seq data for testing."""
    np.random.seed(42)
    
    # Create bulk RNA-seq expression data (higher counts, less sparse)
    n_samples = 12  # 6 control + 6 treatment
    n_genes = 2000
    
    # Generate count data - use float64 to avoid casting errors during multiplication
    base_expression = np.random.negative_binomial(n=20, p=0.1, size=(n_samples, n_genes)).astype(np.float64)
    
    # Add differential expression for some genes
    de_genes = np.random.choice(n_genes, size=200, replace=False)
    fold_changes = np.random.normal(0, 2, len(de_genes))
    
    for i, gene_idx in enumerate(de_genes):
        # Apply fold change to treatment samples (last 6 samples)
        base_expression[6:, gene_idx] = base_expression[6:, gene_idx] * np.exp(fold_changes[i])
    
    sample_names = [f"Control_{i+1}" for i in range(6)] + [f"Treatment_{i+1}" for i in range(6)]
    gene_names = [f"GENE_{i:05d}" for i in range(n_genes)]
    
    df = pd.DataFrame(base_expression, index=sample_names, columns=gene_names)
    return df

@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        'source': 'test_data',
        'n_samples': 500,
        'n_genes': 1000,
        'organism': 'Homo sapiens',
        'tissue': 'PBMC'
    }

@pytest.fixture
def data_manager():
    """Create DataManager instance for testing."""
    return DataManager()

@pytest.fixture
def temp_directory():
    """Create temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# Core Data Manager Tests
class TestDataManager:
    """Test cases for DataManager functionality."""
    
    def test_empty_state(self, data_manager):
        """Test DataManager empty state."""
        assert not data_manager.has_data()
        assert data_manager.current_data is None
        assert data_manager.adata is None
        assert len(data_manager.latest_plots) == 0
    
    def test_set_data_valid(self, data_manager, sample_expression_data, sample_metadata):
        """Test setting valid data."""
        data_manager.set_data(sample_expression_data, sample_metadata)
        
        assert data_manager.has_data()
        assert data_manager.current_data.shape == sample_expression_data.shape
        assert data_manager.adata is not None
        assert data_manager.current_metadata == sample_metadata
    
    def test_set_data_invalid(self, data_manager):
        """Test setting invalid data."""
        # Test None data
        with pytest.raises(ValueError):
            data_manager.set_data(None)
        
        # Test empty DataFrame
        with pytest.raises(ValueError):
            data_manager.set_data(pd.DataFrame())
    
    def test_data_summary(self, data_manager, sample_expression_data):
        """Test data summary generation."""
        data_manager.set_data(sample_expression_data)
        summary = data_manager.get_data_summary()
        
        assert summary['status'] == 'Data loaded'
        assert summary['shape'] == sample_expression_data.shape
        assert 'memory_usage' in summary
        assert 'columns' in summary
    
    def test_plot_management(self, data_manager):
        """Test plot addition and clearing."""
        import plotly.graph_objects as go
        
        # Add plots
        plot1 = go.Figure()
        plot2 = go.Figure()
        
        data_manager.add_plot(plot1, "Test Plot 1")
        data_manager.add_plot(plot2, "Test Plot 2")
        
        assert len(data_manager.latest_plots) == 2
        
        # Clear plots
        data_manager.clear_plots()
        assert len(data_manager.latest_plots) == 0
    
    def test_tool_usage_tracking(self, data_manager):
        """Test tool usage tracking for reproducibility."""
        # Initial state
        assert len(data_manager.tool_usage_history) == 0
        
        # Log tool usage
        data_manager.log_tool_usage(
            tool_name="test_tool",
            parameters={"param1": "value1", "param2": 42},
            description="Test description"
        )
        
        # Verify tool was logged
        assert len(data_manager.tool_usage_history) == 1
        logged_tool = data_manager.tool_usage_history[0]
        assert logged_tool["tool"] == "test_tool"
        assert logged_tool["parameters"]["param1"] == "value1"
        assert logged_tool["parameters"]["param2"] == 42
        assert logged_tool["description"] == "Test description"
        assert "timestamp" in logged_tool
    
    def test_technical_summary_generation(self, data_manager, sample_expression_data):
        """Test technical summary generation."""
        # Set data
        data_manager.set_data(sample_expression_data)
        
        # Log some tool usage
        data_manager.log_tool_usage(
            tool_name="clustering",
            parameters={"resolution": 0.8},
            description="Cell clustering"
        )
        data_manager.log_tool_usage(
            tool_name="qc",
            parameters={"min_genes": 200},
            description="Quality control"
        )
        
        # Generate technical summary
        summary = data_manager.get_technical_summary()
        
        # Verify summary content
        assert "# Technical Summary" in summary
        assert "## Data Information" in summary
        assert "## Tool Usage History" in summary
        assert "clustering" in summary
        assert "qc" in summary
        assert "resolution: 0.8" in summary
        assert "min_genes: 200" in summary
    
    def test_data_package_creation(self, data_manager, sample_expression_data, temp_directory):
        """Test creating a downloadable data package."""
        # Set data and log tool usage
        data_manager.set_data(sample_expression_data)
        data_manager.log_tool_usage(
            tool_name="test_analysis",
            parameters={"param": "value"},
            description="Test analysis"
        )
        
        # Add a plot
        import plotly.graph_objects as go
        plot = go.Figure()
        data_manager.add_plot(plot, "Test Plot")
        
        # Create package in temp directory
        output_dir = os.path.join(temp_directory, "exports")
        package_path = data_manager.create_data_package(output_dir=output_dir)
        
        # Verify package was created and is a zip file
        assert os.path.exists(package_path)
        assert package_path.endswith('.zip')
        
        # Verify zip file contains expected files
        import zipfile
        with zipfile.ZipFile(package_path) as zipf:
            file_list = zipf.namelist()
            assert "technical_summary.md" in file_list
            assert "raw_data.csv" in file_list


# GEO Service Tests
class TestGEOService:
    """Test cases for GEO data service."""
    
    def test_gse_id_extraction(self, data_manager):
        """Test GSE ID extraction from queries."""
        geo_service = GEOService(data_manager)
        
        # Valid GSE IDs
        assert geo_service._extract_gse_id("GSE291670") == "GSE291670"
        assert geo_service._extract_gse_id("Download GSE123456 please") == "GSE123456"
        assert geo_service._extract_gse_id("GSE291670") == "GSE291670"  # Case insensitive
        
        # Invalid queries
        assert geo_service._extract_gse_id("no accession here") is None
        assert geo_service._extract_gse_id("") is None
    
    @patch('lobster.tools.geo_service.GEODownloadManager')
    @patch('lobster.tools.geo_service.GEOParser')
    def test_download_dataset_success(self, mock_parser, mock_downloader, data_manager, sample_expression_data):
        """Test successful dataset download."""
        # Mock the downloader and parser
        mock_downloader_instance = Mock()
        mock_parser_instance = Mock()
        
        mock_downloader.return_value = mock_downloader_instance
        mock_parser.return_value = mock_parser_instance
        
        # Mock successful download
        mock_downloader_instance.download_geo_data.return_value = (Path("/fake/soft"), Path("/fake/suppl"))
        mock_parser_instance.parse_soft_file.return_value = (None, {'series': {'Series_title': 'Test Study'}})
        mock_parser_instance.parse_supplementary_file.return_value = sample_expression_data
        
        geo_service = GEOService(data_manager)
        result = geo_service.download_dataset("GSE291670")
        
        assert "Successfully downloaded GSE291670" in result
        assert data_manager.has_data()
    
    def test_download_dataset_invalid_id(self, data_manager):
        """Test download with invalid GSE ID."""
        geo_service = GEOService(data_manager)
        result = geo_service.download_dataset("invalid accession")
        
        assert "Please provide a valid GSE accession number" in result


# Quality Service Tests
class TestQualityService:
    """Test cases for quality assessment service."""
    
    def test_quality_assessment_no_data(self, data_manager):
        """Test quality assessment with no data."""
        quality_service = QualityService(data_manager)
        result = quality_service.assess_quality()
        
        assert "No data loaded" in result
    
    def test_quality_assessment_with_data(self, data_manager, sample_expression_data):
        """Test quality assessment with valid data."""
        data_manager.set_data(sample_expression_data)
        quality_service = QualityService(data_manager)
        
        result = quality_service.assess_quality()
        
        assert "Quality Assessment Complete" in result
        assert len(data_manager.latest_plots) > 0
        assert 'qc_metrics' in data_manager.current_metadata
    
    def test_qc_metrics_calculation(self, data_manager, sample_expression_data):
        """Test QC metrics calculation."""
        data_manager.set_data(sample_expression_data)
        quality_service = QualityService(data_manager)
        
        qc_metrics = quality_service._calculate_qc_metrics(sample_expression_data)
        
        # Check basic metrics
        assert 'total_counts' in qc_metrics.columns
        assert 'n_genes' in qc_metrics.columns
        assert 'mt_pct' in qc_metrics.columns
        assert len(qc_metrics) == len(sample_expression_data)
        
        # Check enhanced metrics
        assert 'ribo_pct' in qc_metrics.columns
        assert 'housekeeping_score' in qc_metrics.columns
    
    def test_qc_with_parameters(self, data_manager, sample_expression_data):
        """Test quality assessment with custom parameters."""
        data_manager.set_data(sample_expression_data)
        quality_service = QualityService(data_manager)
        
        # Run with custom parameters
        result = quality_service.assess_quality(
            min_genes=600,
            max_mt_pct=10.0,
            max_ribo_pct=30.0,
            min_housekeeping_score=2.0
        )
        
        # Check that parameters were applied
        assert "Quality Assessment Complete" in result
        assert "Minimum genes per cell: 600" in result
        assert "Maximum mitochondrial percentage: 10.0%" in result
        assert "Maximum ribosomal percentage: 30.0%" in result
        assert "Minimum housekeeping gene score: 2.0" in result
        
        # Verify QC results in metadata
        assert 'qc_metrics' in data_manager.current_metadata
        assert 'filter_params' in data_manager.current_metadata['qc_metrics']
        assert data_manager.current_metadata['qc_metrics']['filter_params']['min_genes'] == 600
        assert data_manager.current_metadata['qc_metrics']['filter_params']['max_mt_pct'] == 10.0


# Clustering Service Tests
class TestClusteringService:
    """Test cases for clustering service."""
    
    def test_clustering_no_data(self, data_manager):
        """Test clustering with no data."""
        clustering_service = ClusteringService(data_manager)
        result = clustering_service.cluster_and_visualize()
        
        assert "No data loaded" in result
    
    def test_clustering_with_data(self, data_manager, sample_expression_data):
        """Test clustering with valid data."""
        data_manager.set_data(sample_expression_data)
        clustering_service = ClusteringService(data_manager)
        
        result = clustering_service.cluster_and_visualize(resolution=0.5)
        
        assert "Clustering Completed" in result
        assert len(data_manager.latest_plots) > 0
        assert 'clusters' in data_manager.current_metadata
        assert 'n_clusters' in data_manager.current_metadata
    
    def test_prepare_adata(self, data_manager, sample_expression_data):
        """Test AnnData preparation."""
        data_manager.set_data(sample_expression_data)
        clustering_service = ClusteringService(data_manager)
        
        adata = clustering_service._prepare_adata()
        
        assert isinstance(adata, sc.AnnData)
        assert adata.shape == sample_expression_data.shape


# Bulk RNA-seq Service Tests
class TestBulkRNASeqService:
    """Test cases for bulk RNA-seq analysis service."""
    
    def test_deseq2_analysis_no_data(self, data_manager):
        """Test DESeq2 analysis with no data."""
        bulk_service = BulkRNASeqService(data_manager)
        result = bulk_service.run_deseq2_analysis()
        
        assert "No count data available" in result
    
    def test_deseq2_analysis_with_data(self, data_manager, sample_bulk_data):
        """Test DESeq2 analysis with bulk data."""
        data_manager.set_data(sample_bulk_data)
        bulk_service = BulkRNASeqService(data_manager)
        
        result = bulk_service.run_deseq2_analysis()
        
        assert "DESeq2 Differential Expression Analysis Complete" in result
        assert len(data_manager.latest_plots) > 0
        assert data_manager.has_data()  # Results should be stored
    
    @patch('subprocess.run')
    def test_fastqc_execution(self, mock_subprocess, data_manager, temp_directory):
        """Test FastQC execution."""
        # Mock successful subprocess execution
        mock_subprocess.return_value = Mock(returncode=0, stderr="")
        
        bulk_service = BulkRNASeqService(data_manager)
        
        # Create fake FASTQ files
        fastq_files = []
        for i in range(2):
            fastq_file = temp_directory / f"sample_{i}.fastq"
            fastq_file.write_text("@read1\nACGT\n+\nIIII\n")
            fastq_files.append(str(fastq_file))
        
        result = bulk_service.run_fastqc(fastq_files)
        
        assert "FastQC Analysis Complete" in result
        assert mock_subprocess.called
    
    def test_enrichment_analysis_no_data(self, data_manager):
        """Test enrichment analysis with no data."""
        bulk_service = BulkRNASeqService(data_manager)
        result = bulk_service.run_enrichment_analysis()
        
        assert "No data available" in result


# Enhanced Single-cell Service Tests
class TestEnhancedSingleCellService:
    """Test cases for enhanced single-cell service."""
    
    def test_doublet_detection_no_data(self, data_manager):
        """Test doublet detection with no data."""
        sc_service = EnhancedSingleCellService(data_manager)
        result = sc_service.detect_doublets()
        
        assert "No data loaded" in result
    
    def test_doublet_detection_with_data(self, data_manager, sample_expression_data):
        """Test doublet detection with valid data."""
        data_manager.set_data(sample_expression_data)
        sc_service = EnhancedSingleCellService(data_manager)
        
        result = sc_service.detect_doublets()
        
        assert "Doublet Detection Complete" in result
        assert len(data_manager.latest_plots) > 0
        assert 'doublet_scores' in data_manager.current_metadata
        
    def test_doublet_detection_with_custom_threshold(self, data_manager, sample_expression_data):
        """Test doublet detection with custom threshold."""
        data_manager.set_data(sample_expression_data)
        sc_service = EnhancedSingleCellService(data_manager)
        
        # Run with custom parameters from publication
        result = sc_service.detect_doublets(expected_doublet_rate=0.025, threshold=0.22)
        
        assert "Doublet Detection Complete" in result
        assert "**Expected Doublet Rate:** 2.5%" in result
        assert 'doublet_scores' in data_manager.current_metadata
        assert 'predicted_doublets' in data_manager.current_metadata
    
    def test_cell_type_annotation_no_data(self, data_manager):
        """Test cell type annotation with no data."""
        sc_service = EnhancedSingleCellService(data_manager)
        result = sc_service.annotate_cell_types()
        
        assert "No data loaded" in result
    
    def test_marker_gene_calculation(self, data_manager, sample_expression_data):
        """Test marker gene score calculation."""
        # First cluster the data
        data_manager.set_data(sample_expression_data)
        clustering_service = ClusteringService(data_manager)
        clustering_service.cluster_and_visualize()
        
        # Then test annotation
        sc_service = EnhancedSingleCellService(data_manager)
        markers = {'T cells': ['CD3D', 'CD8A'], 'B cells': ['CD19', 'MS4A1']}
        
        scores = sc_service._calculate_marker_scores(markers)
        
        assert isinstance(scores, dict)
        assert len(scores) > 0  # Should have cluster scores


# # File Upload Service Tests
# class TestFileUploadService:
#     """Test cases for file upload service."""
    
#     def test_csv_processing(self, data_manager, temp_directory, sample_expression_data):
#         """Test CSV file processing."""
#         upload_service = FileUploadService(data_manager)
        
#         # Create test CSV file
#         csv_file = temp_directory / "test_data.csv"
#         sample_expression_data.to_csv(csv_file)
        
#         data, metadata = upload_service._process_csv(csv_file)
        
#         assert data is not None
#         assert data.shape == sample_expression_data.shape
#         assert metadata['format'] == 'csv'
    
#     def test_excel_processing(self, data_manager, temp_directory, sample_expression_data):
#         """Test Excel file processing."""
#         upload_service = FileUploadService(data_manager)
        
#         # Create test Excel file
#         excel_file = temp_directory / "test_data.xlsx"
#         sample_expression_data.to_excel(excel_file)
        
#         data, metadata = upload_service._process_excel(excel_file)
        
#         assert data is not None
#         assert data.shape == sample_expression_data.shape
#         assert metadata['format'] == 'excel'
    
#     def test_data_type_detection(self, data_manager, sample_expression_data, sample_bulk_data):
#         """Test automatic data type detection."""
#         upload_service = FileUploadService(data_manager)
        
#         # Test single-cell detection (many cells)
#         sc_type = upload_service._detect_data_type(sample_expression_data)
#         assert sc_type == "single_cell"
        
#         # Test bulk RNA-seq detection (few samples)
#         bulk_type = upload_service._detect_data_type(sample_bulk_data)
#         assert bulk_type == "bulk_rnaseq"
    
#     def test_fastq_processing(self, data_manager, temp_directory):
#         """Test FASTQ file processing."""
#         upload_service = FileUploadService(data_manager)
        
#         # Create test FASTQ file
#         fastq_file = temp_directory / "test.fastq"
#         fastq_content = "@read1\nACGTACGT\n+\nIIIIIIII\n@read2\nTCGATCGA\n+\nIIIIIIII\n"
#         fastq_file.write_text(fastq_content)
        
#         data, metadata = upload_service._process_fastq(fastq_file)
        
#         assert metadata['format'] == 'fastq'
#         assert metadata['requires_processing'] == True


# Agent Client Tests
class TestAgentClient:
    """Test cases for the agent client system."""

    @patch('lobster.core.client.create_bioinformatics_graph')
    def test_client_initialization(self, mock_create_graph, data_manager, temp_directory):
        """Test agent client initialization."""
        mock_graph = Mock()
        mock_create_graph.return_value = mock_graph
        
        client = AgentClient(
            data_manager=data_manager,
            workspace_path=temp_directory,
            enable_reasoning=False
        )
        
        assert client.data_manager == data_manager
        assert client.workspace_path == temp_directory
        assert client.enable_reasoning == True
        assert len(client.messages) == 0
    
    @patch('lobster.core.client.create_bioinformatics_graph')
    def test_client_query(self, mock_create_graph, data_manager, temp_directory):
        """Test client query processing."""
        # Mock the graph
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": [Mock(content="Test response")]}}
        ]
        mock_create_graph.return_value = mock_graph
        
        client = AgentClient(
            data_manager=data_manager,
            workspace_path=temp_directory
        )
        
        result = client.query("test query")
        
        assert result["success"] == True
        assert "Test response" in result["response"]
        assert len(client.messages) == 2  # Human + AI message
    
    @patch('lobster.core.client.create_bioinformatics_graph')
    def test_client_status(self, mock_create_graph, data_manager, temp_directory):
        """Test client status reporting."""
        mock_create_graph.return_value = Mock()
        
        client = AgentClient(
            data_manager=data_manager,
            workspace_path=temp_directory
        )
        
        status = client.get_status()
        
        assert "session_id" in status
        assert "message_count" in status
        assert "has_data" in status
        assert status["workspace"] == str(temp_directory)
    
    @patch('lobster.core.client.create_bioinformatics_graph')
    def test_workspace_operations(self, mock_create_graph, data_manager, temp_directory):
        """Test workspace file operations."""
        mock_create_graph.return_value = Mock()
        
        client = AgentClient(
            data_manager=data_manager,
            workspace_path=temp_directory
        )
        
        # Test file writing and reading
        test_content = "This is a test file"
        assert client.write_file("test.txt", test_content) == True
        
        read_content = client.read_file("test.txt")
        assert read_content == test_content
        
        # Test file listing
        files = client.list_workspace_files()
        assert len(files) == 1
        assert files[0]["name"] == "test.txt"


# PubMed Service Tests
class TestPubMedService:
    """Test cases for PubMed service."""
    
    @patch('urllib.request.urlopen')
    def test_pubmed_service_initialization(self, mock_urlopen, data_manager):
        """Test PubMed service initialization."""
        # Create a PubMed service instance
        pubmed_service = PubMedService(parse=None, data_manager=data_manager)
        
        assert pubmed_service.data_manager == data_manager
        assert pubmed_service.top_k_results == 3
        assert pubmed_service.MAX_QUERY_LENGTH == 300
    
    @patch('lobster.tools.pubmed_service.PubMedService.load')
    def test_search_pubmed(self, mock_load, data_manager):
        """Test PubMed search functionality."""
        # Create mock results
        mock_results = [
            {
                'uid': '12345678',
                'Title': 'Test Article Title',
                'Published': '2023-01-01',
                'Summary': 'This is a test abstract for the article.'
            }
        ]
        mock_load.return_value = mock_results
        
        # Initialize service
        pubmed_service = PubMedService(parse=None, data_manager=data_manager)
        
        # Perform search
        result = pubmed_service.search_pubmed("test query")
        
        # Verify results
        mock_load.assert_called_once_with("test query")
    
    @patch('lobster.tools.pubmed_service.PubMedService.load')
    def test_find_geo_from_doi(self, mock_load, data_manager):
        """Test finding GEO accession numbers from DOI."""
        # Create mock results with GEO accessions
        mock_results = [
            {
                'uid': '12345678',
                'Title': 'Test Article with GEO',
                'Published': '2023-01-01',
                'Summary': 'The data is available in GEO with accession GSE123456.'
            }
        ]
        mock_load.return_value = mock_results
        
        # Initialize service
        pubmed_service = PubMedService(parse=None, data_manager=data_manager)
        
        # Find GEO from DOI
        result = pubmed_service.find_geo_from_doi("10.1234/test.doi")
        
        # Verify results
        assert "GSE123456" in result
        mock_load.assert_called_once()
        assert "GEO" in mock_load.call_args[0][0]
        assert "10.1234/test.doi" in mock_load.call_args[0][0]
    
    @patch('lobster.tools.pubmed_service.PubMedService.load')
    def test_find_marker_genes(self, mock_load, data_manager):
        """Test finding marker genes from literature."""
        # Create mock results
        mock_results = [
            {
                'uid': '12345678',
                'Title': 'T Cell Markers in Cancer',
                'Published': '2023-01-01',
                'Summary': 'We identified CD3, CD4, and CD8 as key markers.'
            }
        ]
        mock_load.return_value = mock_results
        
        # Initialize service
        pubmed_service = PubMedService(parse=None, data_manager=data_manager)
        
        # Find marker genes
        result = pubmed_service.find_marker_genes("T cells", disease="cancer")
        
        # Verify results
        assert "T Cell Markers in Cancer" in result
        assert "CD3, CD4, and CD8" in result
        mock_load.assert_called_once()
        assert "T cells" in mock_load.call_args[0][0]
        assert "cancer" in mock_load.call_args[0][0]
    
    @patch('lobster.tools.pubmed_service.PubMedService.load')
    def test_find_protocol_information(self, mock_load, data_manager):
        """Test finding protocol information."""
        # Create mock results
        mock_results = [
            {
                'uid': '12345678',
                'Title': 'RNA-seq Protocol for Low Input Samples',
                'Published': '2023-01-01',
                'Summary': 'This protocol describes RNA-seq for low input samples.'
            }
        ]
        mock_load.return_value = mock_results
        
        # Initialize service
        pubmed_service = PubMedService(parse=None, data_manager=data_manager)
        
        # Find protocol information
        result = pubmed_service.find_protocol_information("RNA-seq low input")
        
        # Verify results
        assert "RNA-seq Protocol for Low Input Samples" in result
        mock_load.assert_called_once()
        assert "RNA-seq low input" in mock_load.call_args[0][0]
    
    @patch('urllib.request.urlopen')
    def test_error_handling(self, mock_urlopen, data_manager):
        """Test error handling during API calls."""
        # Mock error response
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://test.com",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None
        )
        
        # Initialize service
        pubmed_service = PubMedService(parse=Mock(), data_manager=data_manager)
        
        # Test error handling in search
        result = pubmed_service.search_pubmed("test query")
        assert "PubMed search encountered an error" in result


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_single_cell_workflow(self, data_manager, sample_expression_data):
        """Test complete single-cell analysis workflow."""
        # 1. Load data
        data_manager.set_data(sample_expression_data)
        assert data_manager.has_data()
        
        # 2. Quality assessment
        quality_service = QualityService(data_manager)
        qa_result = quality_service.assess_quality()
        assert "Quality Assessment Complete" in qa_result
        
        # 3. Clustering
        clustering_service = ClusteringService(data_manager)
        cluster_result = clustering_service.cluster_and_visualize()
        assert "Clustering Completed" in cluster_result
        
        # 4. Enhanced analysis
        sc_service = EnhancedSingleCellService(data_manager)
        doublet_result = sc_service.detect_doublets()
        assert "Doublet Detection Complete" in doublet_result
        
        # Check that all steps produced results
        assert len(data_manager.latest_plots) > 0
        assert 'qc_metrics' in data_manager.current_metadata
        assert 'clusters' in data_manager.current_metadata
    
    def test_bulk_rnaseq_workflow(self, data_manager, sample_bulk_data):
        """Test bulk RNA-seq analysis workflow."""
        # 1. Load data
        data_manager.set_data(sample_bulk_data)
        assert data_manager.has_data()
        
        # 2. Differential expression
        bulk_service = BulkRNASeqService(data_manager)
        de_result = bulk_service.run_deseq2_analysis()
        assert "DESeq2 Differential Expression Analysis Complete" in de_result
        
        # 3. Enrichment analysis
        enrichment_result = bulk_service.run_enrichment_analysis()
        assert "Enrichment Analysis Complete" in enrichment_result
        
        # Check results
        assert len(data_manager.latest_plots) > 0
    
    def test_pubmed_integration_workflow(self, data_manager):
        """Test PubMed integration workflow."""
        # Create mocked PubMedService with patched methods
        with patch('lobster.tools.pubmed_service.PubMedService.load') as mock_load:
            # Mock search results
            mock_search_results = [
                {
                    'uid': '12345678',
                    'Title': 'Single-cell RNA-seq Analysis of T Cells',
                    'Published': '2023-01-01',
                    'Summary': 'This study used scRNA-seq to identify T cell markers.'
                }
            ]
            
            # Mock results with GEO accessions
            mock_geo_results = [
                {
                    'uid': '23456789',
                    'Title': 'Gene Expression Study with GEO Data',
                    'Published': '2023-02-01',
                    'Summary': 'Data available in GSE123456 and GSE789012.'
                }
            ]
            
            # Configure mock to return different results based on query
            def side_effect_load(query):
                if "T cells marker genes" in query:
                    return mock_search_results
                elif "GEO" in query:
                    return mock_geo_results
                return []
            
            mock_load.side_effect = side_effect_load
            
            # Create PubMed service
            pubmed_service = PubMedService(parse=None, data_manager=data_manager)
            
            # 1. Search PubMed
            search_result = pubmed_service.search_pubmed("T cells marker genes")
            assert "Single-cell RNA-seq Analysis of T Cells" in search_result
            assert "12345678" in search_result
            
            # Verify tool usage was logged
            assert len(data_manager.tool_usage_history) == 1
            assert data_manager.tool_usage_history[0]["tool"] == "search_pubmed"
            
            # 2. Find GEO accessions from DOI
            geo_result = pubmed_service.find_geo_from_doi("10.1234/example.doi")
            assert "GSE123456" in geo_result
            assert "GSE789012" in geo_result
            
            # Verify tool usage was logged
            assert len(data_manager.tool_usage_history) == 2
            assert data_manager.tool_usage_history[1]["tool"] == "find_geo_from_doi"
            
            # 3. Generate technical summary with PubMed searches
            summary = data_manager.get_technical_summary()
            assert "search_pubmed" in summary
            assert "find_geo_from_doi" in summary
    
    def test_data_export_workflow(self, data_manager, sample_expression_data, temp_directory):
        """Test complete data export workflow with tool tracking."""
        # 1. Load data
        data_manager.set_data(sample_expression_data, {'data_type': 'single_cell'})
        assert data_manager.has_data()
        
        # 2. Perform analysis with tool tracking
        quality_service = QualityService(data_manager)
        clustering_service = ClusteringService(data_manager)
        sc_service = EnhancedSingleCellService(data_manager)
        
        # Log all tool usages
        data_manager.log_tool_usage(
            tool_name="load_data",
            parameters={"shape": sample_expression_data.shape},
            description="Loaded single-cell expression matrix"
        )
        
        # Run quality assessment
        qa_result = quality_service.assess_quality()
        data_manager.log_tool_usage(
            tool_name="assess_quality",
            parameters={},
            description="Quality control assessment of single-cell data"
        )
        
        # Run clustering
        cluster_result = clustering_service.cluster_and_visualize(resolution=0.8)
        data_manager.log_tool_usage(
            tool_name="cluster_cells",
            parameters={"resolution": 0.8},
            description="Cell clustering and UMAP visualization"
        )
        
        # Find marker genes
        marker_result = sc_service.find_marker_genes()
        data_manager.log_tool_usage(
            tool_name="find_marker_genes",
            parameters={},
            description="Identification of marker genes for cell clusters"
        )
        
        # Verify tools were logged
        assert len(data_manager.tool_usage_history) == 4
        
        # 3. Create data package
        export_dir = os.path.join(temp_directory, "exports")
        export_path = data_manager.create_data_package(output_dir=export_dir)
        assert os.path.exists(export_path)
        
        # 4. Verify package contents
        import zipfile
        with zipfile.ZipFile(export_path) as zipf:
            contents = zipf.namelist()
            # Check for all expected files
            assert "technical_summary.md" in contents
            assert "raw_data.csv" in contents
            assert any(f.startswith("plots/") for f in contents)
            
            # Extract and check technical summary content
            with zipf.open("technical_summary.md") as summary_file:
                summary_content = summary_file.read().decode('utf-8')
                # Verify all tools are documented
                assert "load_data" in summary_content
                assert "assess_quality" in summary_content
                assert "cluster_cells" in summary_content
                assert "find_marker_genes" in summary_content
                assert "resolution: 0.8" in summary_content
    
    @patch('lobster.core.client.create_bioinformatics_graph')
    def test_agent_client_integration(self, mock_create_graph, data_manager, temp_directory, sample_expression_data):
        """Test agent client integration with data analysis."""
        # Mock the graph to simulate agent responses
        mock_messages = [
            Mock(content="Data loaded successfully."),
            Mock(content="Quality assessment complete. Found 15 clusters."),
            Mock(content="Analysis complete. Generated 3 plots.")
        ]
        
        mock_graph = Mock()
        mock_graph.stream.return_value = [
            {"supervisor": {"messages": mock_messages}}
        ]
        mock_create_graph.return_value = mock_graph
        
        # Create client
        client = AgentClient(
            data_manager=data_manager,
            workspace_path=temp_directory
        )
        
        # Load data manually
        data_manager.set_data(sample_expression_data)
        
        # Test query
        result = client.query("Analyze this single-cell data")
        
        assert result["success"] == True
        assert "Analysis complete" in result["response"]
        assert len(client.messages) == 2  # Human + AI message
        
        # Test status
        status = client.get_status()
        assert status["has_data"] == True
        assert "data_summary" in status


# Performance Tests
class TestPerformance:
    """Performance tests for large datasets."""
    
    def test_large_dataset_handling(self, data_manager):
        """Test handling of larger datasets."""
        # Create larger dataset
        large_data = pd.DataFrame(
            np.random.randint(0, 100, (5000, 2000)),
            index=[f"Cell_{i}" for i in range(5000)],
            columns=[f"Gene_{i}" for i in range(2000)]
        )
        
        # Test data loading
        import time
        start_time = time.time()
        data_manager.set_data(large_data)
        load_time = time.time() - start_time
        
        assert data_manager.has_data()
        assert load_time < 10  # Should load within 10 seconds
    
    def test_memory_usage(self, data_manager, sample_expression_data):
        """Test memory usage tracking."""
        data_manager.set_data(sample_expression_data)
        summary = data_manager.get_data_summary()
        
        assert 'memory_usage' in summary
        assert 'MB' in summary['memory_usage']


# CLI Tests
class TestCLI:
    """Test cases for CLI functionality."""
    
    @patch('lobster.core.client.AgentClient')
    def test_cli_query_command(self, mock_client_class, temp_directory):
        """Test CLI query command."""
        from lobster.cli import app
        from typer.testing import CliRunner
        
        # Mock client
        mock_client = Mock()
        mock_client.query.return_value = {
            "success": True,
            "response": "Test response from CLI"
        }
        mock_client_class.return_value = mock_client
        
        runner = CliRunner()
        result = runner.invoke(app, ["query", "test query"])
        
        assert result.exit_code == 0
        mock_client.query.assert_called_once_with("test query")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
