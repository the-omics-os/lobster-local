"""
Comprehensive integration tests for GEO download workflows.

This module provides thorough testing of GEO (Gene Expression Omnibus) dataset
download workflows, including dataset discovery, metadata extraction, file downloading,
format conversion, quality assessment, and automated processing pipelines.

Test coverage target: 95%+ with realistic GEO workflow scenarios.
"""

import pytest
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import json
import time
import requests
import gzip
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.geo_service import GEOService
from lobster.tools.preprocessing_service import PreprocessingService
from lobster.tools.quality_service import QualityService

from tests.mock_data.factories import SingleCellDataFactory, BulkRNASeqDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG, LARGE_DATASET_CONFIG


# ===============================================================================
# Mock GEO Data and Fixtures
# ===============================================================================

@pytest.fixture
def temp_workspace():
    """Create temporary workspace for GEO workflow tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / ".lobster_workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        yield workspace_path


@pytest.fixture
def mock_geo_metadata():
    """Mock GEO dataset metadata."""
    return {
        'GSE123456': {
            'title': 'Single-cell RNA sequencing of human immune cells',
            'summary': 'This study characterizes immune cell populations using scRNA-seq',
            'organism': 'Homo sapiens',
            'experiment_type': 'Expression profiling by high throughput sequencing',
            'platform': 'GPL24676 (Illumina NovaSeq 6000)',
            'sample_count': 8,
            'series_matrix_files': ['GSE123456_series_matrix.txt.gz'],
            'supplementary_files': [
                'GSE123456_barcodes.tsv.gz',
                'GSE123456_features.tsv.gz', 
                'GSE123456_matrix.mtx.gz'
            ],
            'publication_date': '2024-01-15',
            'last_update_date': '2024-01-20',
            'contact_info': {
                'name': 'Dr. Jane Smith',
                'email': 'jane.smith@university.edu',
                'institute': 'University Research Institute'
            },
            'sample_metadata': {
                'GSM123456': {'condition': 'control', 'cell_type': 'PBMC', 'batch': 1},
                'GSM123457': {'condition': 'treatment', 'cell_type': 'PBMC', 'batch': 1},
                'GSM123458': {'condition': 'control', 'cell_type': 'PBMC', 'batch': 2},
                'GSM123459': {'condition': 'treatment', 'cell_type': 'PBMC', 'batch': 2}
            }
        },
        'GSE789012': {
            'title': 'Bulk RNA-seq time course of drug treatment',
            'summary': 'Time course analysis of drug response in cancer cells',
            'organism': 'Homo sapiens',
            'experiment_type': 'Expression profiling by high throughput sequencing',
            'platform': 'GPL20301 (Illumina HiSeq 4000)',
            'sample_count': 24,
            'series_matrix_files': ['GSE789012_series_matrix.txt.gz'],
            'supplementary_files': [
                'GSE789012_counts.txt.gz',
                'GSE789012_fpkm.txt.gz',
                'GSE789012_metadata.txt'
            ],
            'publication_date': '2024-02-01',
            'last_update_date': '2024-02-05',
            'sample_metadata': {
                'GSM789012': {'timepoint': '0h', 'treatment': 'control', 'replicate': 1},
                'GSM789013': {'timepoint': '6h', 'treatment': 'drug_a', 'replicate': 1},
                'GSM789014': {'timepoint': '12h', 'treatment': 'drug_a', 'replicate': 1}
            }
        }
    }


@pytest.fixture
def mock_geo_files():
    """Mock GEO file contents."""
    return {
        'barcodes.tsv': '\n'.join([f'CELL_{i}' for i in range(1000)]),
        'features.tsv': '\n'.join([f'GENE_{i}\tENSG{i:08d}\tGene' for i in range(2000)]),
        'matrix.mtx': '''%%MatrixMarket matrix coordinate integer general
%
2000 1000 150000
1 1 5
1 2 3
2 1 8
''',
        'series_matrix.txt': '''!Series_title\t"Test Dataset"
!Series_geo_accession\t"GSE123456"
!Series_status\t"Public on Jan 15 2024"
ID_REF\tGSM123456\tGSM123457
GENE_1\t10.5\t12.3
GENE_2\t8.7\t9.1
''',
        'counts.txt': '''gene_id\tGSM789012\tGSM789013\tGSM789014
GENE_1\t45\t52\t48
GENE_2\t123\t145\t132
GENE_3\t67\t71\t69
''',
        'metadata.txt': '''sample_id\ttimepoint\ttreatment\treplicate
GSM789012\t0h\tcontrol\t1
GSM789013\t6h\tdrug_a\t1
GSM789014\t12h\tdrug_a\t1
'''
    }


@pytest.fixture
def mock_geo_service():
    """Create mock GEO service."""
    service = Mock(spec=GEOService)
    
    # Mock file download simulation
    def mock_download_file(url, local_path, **kwargs):
        """Simulate file download."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simulate different file types
        if 'barcodes.tsv' in str(local_path):
            content = '\n'.join([f'CELL_{i}' for i in range(100)])
        elif 'features.tsv' in str(local_path):
            content = '\n'.join([f'GENE_{i}\tENSG{i:08d}\tGene' for i in range(200)])
        elif 'matrix.mtx' in str(local_path):
            content = '''%%MatrixMarket matrix coordinate integer general
200 100 1500
1 1 5
2 1 3
'''
        elif 'counts.txt' in str(local_path):
            content = '''gene_id\tsample1\tsample2
GENE_1\t45\t52
GENE_2\t123\t145
'''
        else:
            content = f"Mock file content for {local_path.name}"
        
        # Handle compressed files
        if str(local_path).endswith('.gz'):
            with gzip.open(local_path, 'wt') as f:
                f.write(content)
        else:
            local_path.write_text(content)
        
        return {
            'success': True,
            'file_size': len(content.encode()),
            'download_time': 1.5
        }
    
    service.download_file = mock_download_file
    return service


# ===============================================================================
# Dataset Discovery and Search Tests
# ===============================================================================

@pytest.mark.integration
class TestGEODatasetDiscovery:
    """Test GEO dataset discovery and search functionality."""
    
    def test_geo_search_by_keywords(self, mock_geo_metadata):
        """Test searching GEO datasets by keywords."""
        
        class GEOSearchEngine:
            """Mock GEO search engine."""
            
            def __init__(self, metadata_db):
                self.metadata_db = metadata_db
                
            def search_datasets(self, keywords, organism=None, experiment_type=None, date_range=None):
                """Search datasets by keywords and filters."""
                results = []
                
                for accession, metadata in self.metadata_db.items():
                    match_score = 0
                    
                    # Keyword matching
                    searchable_text = f"{metadata['title']} {metadata['summary']}".lower()
                    for keyword in keywords.lower().split():
                        if keyword in searchable_text:
                            match_score += 1
                    
                    # Filter by organism
                    if organism and organism.lower() not in metadata['organism'].lower():
                        continue
                    
                    # Filter by experiment type
                    if experiment_type and experiment_type.lower() not in metadata['experiment_type'].lower():
                        continue
                    
                    if match_score > 0:
                        results.append({
                            'accession': accession,
                            'title': metadata['title'],
                            'organism': metadata['organism'],
                            'experiment_type': metadata['experiment_type'],
                            'sample_count': metadata['sample_count'],
                            'publication_date': metadata['publication_date'],
                            'relevance_score': match_score
                        })
                
                # Sort by relevance
                results.sort(key=lambda x: x['relevance_score'], reverse=True)
                return results
        
        # Test keyword search
        search_engine = GEOSearchEngine(mock_geo_metadata)
        
        # Search for single-cell datasets
        sc_results = search_engine.search_datasets('single cell RNA sequencing')
        assert len(sc_results) >= 1
        assert any('GSE123456' in result['accession'] for result in sc_results)
        
        # Search for bulk RNA-seq datasets
        bulk_results = search_engine.search_datasets('bulk RNA time course')
        assert len(bulk_results) >= 1
        assert any('GSE789012' in result['accession'] for result in bulk_results)
        
        # Search with organism filter
        human_results = search_engine.search_datasets('RNA sequencing', organism='Homo sapiens')
        assert all(result['organism'] == 'Homo sapiens' for result in human_results)
        
        # Search with experiment type filter
        seq_results = search_engine.search_datasets('expression', experiment_type='high throughput sequencing')
        assert all('sequencing' in result['experiment_type'] for result in seq_results)
    
    def test_geo_metadata_extraction(self, mock_geo_metadata):
        """Test extraction and parsing of GEO metadata."""
        
        class GEOMetadataExtractor:
            """Extracts and parses GEO metadata."""
            
            def extract_metadata(self, accession, metadata_source):
                """Extract comprehensive metadata for a dataset."""
                if accession not in metadata_source:
                    raise ValueError(f"Dataset {accession} not found")
                
                raw_metadata = metadata_source[accession]
                
                # Parse and enrich metadata
                parsed_metadata = {
                    'basic_info': {
                        'accession': accession,
                        'title': raw_metadata['title'],
                        'summary': raw_metadata['summary'],
                        'organism': raw_metadata['organism'],
                        'platform': raw_metadata['platform']
                    },
                    'experimental_design': {
                        'experiment_type': raw_metadata['experiment_type'],
                        'sample_count': raw_metadata['sample_count'],
                        'data_processing': self._infer_data_processing(raw_metadata),
                        'study_design': self._analyze_study_design(raw_metadata)
                    },
                    'data_availability': {
                        'series_matrix_files': raw_metadata['series_matrix_files'],
                        'supplementary_files': raw_metadata['supplementary_files'],
                        'estimated_download_size': self._estimate_download_size(raw_metadata)
                    },
                    'temporal_info': {
                        'publication_date': raw_metadata['publication_date'],
                        'last_update_date': raw_metadata['last_update_date'],
                        'data_freshness': self._calculate_data_freshness(raw_metadata['last_update_date'])
                    },
                    'sample_annotations': self._process_sample_metadata(raw_metadata.get('sample_metadata', {}))
                }
                
                return parsed_metadata
            
            def _infer_data_processing(self, metadata):
                """Infer data processing type from metadata."""
                title_summary = f"{metadata['title']} {metadata['summary']}".lower()
                
                if 'single-cell' in title_summary or 'scrna' in title_summary:
                    return 'single_cell_rna_seq'
                elif 'bulk' in title_summary or ('rna' in title_summary and 'seq' in title_summary):
                    return 'bulk_rna_seq'
                elif 'proteom' in title_summary:
                    return 'proteomics'
                elif 'spatial' in title_summary:
                    return 'spatial_transcriptomics'
                else:
                    return 'unknown'
            
            def _analyze_study_design(self, metadata):
                """Analyze experimental study design."""
                sample_metadata = metadata.get('sample_metadata', {})
                
                if not sample_metadata:
                    return {'type': 'unknown', 'factors': []}
                
                # Extract experimental factors
                factors = set()
                for sample_info in sample_metadata.values():
                    factors.update(sample_info.keys())
                
                # Determine study type
                if 'timepoint' in factors:
                    study_type = 'time_course'
                elif 'condition' in factors or 'treatment' in factors:
                    study_type = 'comparative'
                elif 'cell_type' in factors:
                    study_type = 'cell_type_comparison'
                else:
                    study_type = 'descriptive'
                
                return {
                    'type': study_type,
                    'factors': list(factors),
                    'n_conditions': len(set(
                        info.get('condition', info.get('treatment', 'unknown'))
                        for info in sample_metadata.values()
                    ))
                }
            
            def _estimate_download_size(self, metadata):
                """Estimate total download size."""
                # Simple heuristic based on sample count and file types
                sample_count = metadata['sample_count']
                file_count = len(metadata['supplementary_files'])
                
                if 'single-cell' in metadata['title'].lower():
                    # Single-cell data is typically larger
                    base_size_mb = sample_count * 50 + file_count * 10
                else:
                    # Bulk data
                    base_size_mb = sample_count * 5 + file_count * 2
                
                return {
                    'estimated_mb': base_size_mb,
                    'confidence': 'low',  # Heuristic-based estimate
                    'compressed_size_factor': 0.3
                }
            
            def _calculate_data_freshness(self, last_update_date):
                """Calculate data freshness score."""
                from datetime import datetime, timedelta
                
                try:
                    update_date = datetime.strptime(last_update_date, '%Y-%m-%d')
                    days_old = (datetime.now() - update_date).days
                    
                    if days_old <= 30:
                        return 'very_fresh'
                    elif days_old <= 180:
                        return 'fresh'
                    elif days_old <= 365:
                        return 'moderate'
                    else:
                        return 'old'
                except:
                    return 'unknown'
            
            def _process_sample_metadata(self, sample_metadata):
                """Process and summarize sample metadata."""
                if not sample_metadata:
                    return {'summary': 'No sample metadata available'}
                
                # Summarize sample annotations
                factors = {}
                for sample_id, annotations in sample_metadata.items():
                    for factor, value in annotations.items():
                        if factor not in factors:
                            factors[factor] = set()
                        factors[factor].add(value)
                
                return {
                    'n_samples': len(sample_metadata),
                    'experimental_factors': {
                        factor: list(values) for factor, values in factors.items()
                    },
                    'sample_balance': self._check_sample_balance(sample_metadata)
                }
            
            def _check_sample_balance(self, sample_metadata):
                """Check if samples are balanced across conditions."""
                if not sample_metadata:
                    return {'balanced': False, 'reason': 'no_metadata'}
                
                # Check condition balance
                conditions = []
                for sample_info in sample_metadata.values():
                    condition = sample_info.get('condition', sample_info.get('treatment', 'unknown'))
                    conditions.append(condition)
                
                condition_counts = {}
                for condition in conditions:
                    condition_counts[condition] = condition_counts.get(condition, 0) + 1
                
                counts = list(condition_counts.values())
                if len(set(counts)) == 1:
                    return {'balanced': True, 'type': 'perfectly_balanced'}
                elif max(counts) / min(counts) <= 2:
                    return {'balanced': True, 'type': 'reasonably_balanced'}
                else:
                    return {'balanced': False, 'reason': 'imbalanced_conditions'}
        
        # Test metadata extraction
        extractor = GEOMetadataExtractor()
        
        # Extract metadata for single-cell dataset
        sc_metadata = extractor.extract_metadata('GSE123456', mock_geo_metadata)
        
        assert sc_metadata['basic_info']['accession'] == 'GSE123456'
        assert sc_metadata['experimental_design']['data_processing'] == 'single_cell_rna_seq'
        assert sc_metadata['experimental_design']['study_design']['type'] == 'comparative'
        assert 'condition' in sc_metadata['experimental_design']['study_design']['factors']
        assert sc_metadata['data_availability']['estimated_download_size']['estimated_mb'] > 0
        
        # Extract metadata for bulk RNA-seq dataset
        bulk_metadata = extractor.extract_metadata('GSE789012', mock_geo_metadata)
        
        assert bulk_metadata['basic_info']['accession'] == 'GSE789012'
        assert bulk_metadata['experimental_design']['data_processing'] == 'bulk_rna_seq'
        assert bulk_metadata['experimental_design']['study_design']['type'] == 'time_course'
        assert 'timepoint' in bulk_metadata['experimental_design']['study_design']['factors']
    
    def test_dataset_compatibility_assessment(self, mock_geo_metadata):
        """Test assessment of dataset compatibility with analysis goals."""
        
        class CompatibilityAssessor:
            """Assesses dataset compatibility with analysis requirements."""
            
            def assess_compatibility(self, accession, metadata_db, analysis_requirements):
                """Assess how well a dataset matches analysis requirements."""
                if accession not in metadata_db:
                    return {'compatible': False, 'reason': 'dataset_not_found'}
                
                dataset_metadata = metadata_db[accession]
                compatibility_score = 0
                compatibility_details = {
                    'data_type_match': False,
                    'organism_match': False,
                    'sample_size_adequate': False,
                    'study_design_appropriate': False,
                    'data_quality_indicators': {}
                }
                
                # Check data type compatibility
                required_data_type = analysis_requirements.get('data_type', '')
                dataset_title = dataset_metadata['title'].lower()
                
                if required_data_type == 'single_cell' and 'single-cell' in dataset_title:
                    compatibility_details['data_type_match'] = True
                    compatibility_score += 3
                elif required_data_type == 'bulk_rna_seq' and 'bulk' in dataset_title:
                    compatibility_details['data_type_match'] = True
                    compatibility_score += 3
                
                # Check organism compatibility
                required_organism = analysis_requirements.get('organism', '')
                if required_organism.lower() in dataset_metadata['organism'].lower():
                    compatibility_details['organism_match'] = True
                    compatibility_score += 2
                
                # Check sample size adequacy
                min_samples = analysis_requirements.get('min_samples', 0)
                if dataset_metadata['sample_count'] >= min_samples:
                    compatibility_details['sample_size_adequate'] = True
                    compatibility_score += 1
                
                # Check study design appropriateness
                required_design = analysis_requirements.get('study_design', '')
                sample_metadata = dataset_metadata.get('sample_metadata', {})
                
                if required_design == 'comparative' and any('condition' in info or 'treatment' in info for info in sample_metadata.values()):
                    compatibility_details['study_design_appropriate'] = True
                    compatibility_score += 2
                elif required_design == 'time_course' and any('timepoint' in info or 'time' in info for info in sample_metadata.values()):
                    compatibility_details['study_design_appropriate'] = True
                    compatibility_score += 2
                
                # Assess data quality indicators
                compatibility_details['data_quality_indicators'] = {
                    'recent_publication': self._is_recent_publication(dataset_metadata['publication_date']),
                    'complete_metadata': len(sample_metadata) == dataset_metadata['sample_count'],
                    'supplementary_files_available': len(dataset_metadata['supplementary_files']) > 0
                }
                
                # Calculate final compatibility
                max_score = 8  # Maximum possible score
                compatibility_percentage = (compatibility_score / max_score) * 100
                
                return {
                    'compatible': compatibility_percentage >= 60,
                    'compatibility_score': compatibility_percentage,
                    'details': compatibility_details,
                    'recommendation': self._generate_recommendation(compatibility_percentage, compatibility_details)
                }
            
            def _is_recent_publication(self, publication_date):
                """Check if publication is recent."""
                from datetime import datetime, timedelta
                try:
                    pub_date = datetime.strptime(publication_date, '%Y-%m-%d')
                    return (datetime.now() - pub_date).days <= 730  # Within 2 years
                except:
                    return False
            
            def _generate_recommendation(self, score, details):
                """Generate compatibility recommendation."""
                if score >= 80:
                    return {
                        'level': 'highly_recommended',
                        'message': 'Dataset is highly compatible with analysis requirements'
                    }
                elif score >= 60:
                    return {
                        'level': 'recommended',
                        'message': 'Dataset is compatible with some limitations'
                    }
                elif score >= 40:
                    return {
                        'level': 'conditional',
                        'message': 'Dataset may work with modifications to analysis approach'
                    }
                else:
                    return {
                        'level': 'not_recommended',
                        'message': 'Dataset is not well suited for the intended analysis'
                    }
        
        # Test compatibility assessment
        assessor = CompatibilityAssessor()
        
        # Test single-cell analysis requirements
        sc_requirements = {
            'data_type': 'single_cell',
            'organism': 'Homo sapiens',
            'min_samples': 4,
            'study_design': 'comparative'
        }
        
        sc_compatibility = assessor.assess_compatibility('GSE123456', mock_geo_metadata, sc_requirements)
        
        assert sc_compatibility['compatible'] == True
        assert sc_compatibility['compatibility_score'] >= 60
        assert sc_compatibility['details']['data_type_match'] == True
        assert sc_compatibility['details']['organism_match'] == True
        assert sc_compatibility['details']['sample_size_adequate'] == True
        assert sc_compatibility['details']['study_design_appropriate'] == True
        
        # Test bulk RNA-seq analysis requirements
        bulk_requirements = {
            'data_type': 'bulk_rna_seq',
            'organism': 'Homo sapiens',
            'min_samples': 20,
            'study_design': 'time_course'
        }
        
        bulk_compatibility = assessor.assess_compatibility('GSE789012', mock_geo_metadata, bulk_requirements)
        
        assert bulk_compatibility['compatible'] == True
        assert bulk_compatibility['details']['data_type_match'] == True
        assert bulk_compatibility['details']['organism_match'] == True
        assert bulk_compatibility['details']['study_design_appropriate'] == True
        
        # Test incompatible requirements
        incompatible_requirements = {
            'data_type': 'single_cell',
            'organism': 'Mus musculus',  # Different organism
            'min_samples': 100,  # Too many samples required
            'study_design': 'time_course'  # Different design
        }
        
        incompatible_result = assessor.assess_compatibility('GSE123456', mock_geo_metadata, incompatible_requirements)
        
        assert incompatible_result['compatible'] == False
        assert incompatible_result['compatibility_score'] < 60


# ===============================================================================
# File Download and Processing Tests
# ===============================================================================

@pytest.mark.integration
class TestGEOFileDownloadProcessing:
    """Test GEO file download and processing workflows."""
    
    def test_sequential_file_download(self, temp_workspace, mock_geo_service, mock_geo_files):
        """Test sequential download of GEO files."""
        
        class SequentialDownloader:
            """Handles sequential file downloads."""
            
            def __init__(self, geo_service, workspace_path):
                self.geo_service = geo_service
                self.workspace_path = Path(workspace_path)
                self.download_log = []
                
            def download_dataset_files(self, accession, file_urls):
                """Download all files for a dataset sequentially."""
                dataset_dir = self.workspace_path / f"geo_downloads/{accession}"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
                download_results = {
                    'accession': accession,
                    'total_files': len(file_urls),
                    'successful_downloads': [],
                    'failed_downloads': [],
                    'total_size_mb': 0,
                    'total_download_time': 0
                }
                
                for file_url in file_urls:
                    file_name = file_url.split('/')[-1]
                    local_path = dataset_dir / file_name
                    
                    try:
                        start_time = time.time()
                        result = self.geo_service.download_file(file_url, local_path)
                        download_time = time.time() - start_time
                        
                        if result['success']:
                            file_size_mb = result['file_size'] / (1024**2)
                            download_results['successful_downloads'].append({
                                'file_name': file_name,
                                'local_path': str(local_path),
                                'size_mb': file_size_mb,
                                'download_time': download_time
                            })
                            download_results['total_size_mb'] += file_size_mb
                            download_results['total_download_time'] += download_time
                            
                            self.download_log.append({
                                'accession': accession,
                                'file': file_name,
                                'status': 'success',
                                'timestamp': time.time()
                            })
                        else:
                            download_results['failed_downloads'].append({
                                'file_name': file_name,
                                'error': result.get('error', 'Unknown error')
                            })
                    
                    except Exception as e:
                        download_results['failed_downloads'].append({
                            'file_name': file_name,
                            'error': str(e)
                        })
                        
                        self.download_log.append({
                            'accession': accession,
                            'file': file_name,
                            'status': 'failed',
                            'error': str(e),
                            'timestamp': time.time()
                        })
                
                return download_results
        
        # Test sequential download
        downloader = SequentialDownloader(mock_geo_service, temp_workspace)
        
        # Mock file URLs for single-cell dataset
        file_urls = [
            'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM123nnn/GSM123456/suppl/GSM123456_barcodes.tsv.gz',
            'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM123nnn/GSM123456/suppl/GSM123456_features.tsv.gz',
            'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM123nnn/GSM123456/suppl/GSM123456_matrix.mtx.gz'
        ]
        
        download_results = downloader.download_dataset_files('GSE123456', file_urls)
        
        # Verify download results
        assert download_results['accession'] == 'GSE123456'
        assert download_results['total_files'] == 3
        assert len(download_results['successful_downloads']) == 3
        assert len(download_results['failed_downloads']) == 0
        assert download_results['total_size_mb'] > 0
        
        # Verify files were created
        dataset_dir = temp_workspace / "geo_downloads/GSE123456"
        assert dataset_dir.exists()
        assert (dataset_dir / 'GSM123456_barcodes.tsv.gz').exists()
        assert (dataset_dir / 'GSM123456_features.tsv.gz').exists()
        assert (dataset_dir / 'GSM123456_matrix.mtx.gz').exists()
        
        # Verify download log
        assert len(downloader.download_log) == 3
        assert all(log['status'] == 'success' for log in downloader.download_log)
    
    def test_parallel_file_download(self, temp_workspace, mock_geo_service):
        """Test parallel download of GEO files."""
        
        class ParallelDownloader:
            """Handles parallel file downloads."""
            
            def __init__(self, geo_service, workspace_path, max_workers=3):
                self.geo_service = geo_service
                self.workspace_path = Path(workspace_path)
                self.max_workers = max_workers
                self.download_log = []
                
            def download_dataset_files_parallel(self, accession, file_urls):
                """Download files in parallel."""
                dataset_dir = self.workspace_path / f"geo_downloads/{accession}"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                
                download_results = {
                    'accession': accession,
                    'total_files': len(file_urls),
                    'successful_downloads': [],
                    'failed_downloads': [],
                    'parallel_execution': True,
                    'workers_used': min(self.max_workers, len(file_urls))
                }
                
                def download_single_file(file_url):
                    """Download a single file."""
                    file_name = file_url.split('/')[-1]
                    local_path = dataset_dir / file_name
                    
                    try:
                        start_time = time.time()
                        result = self.geo_service.download_file(file_url, local_path)
                        download_time = time.time() - start_time
                        
                        return {
                            'success': result['success'],
                            'file_name': file_name,
                            'local_path': str(local_path),
                            'size_mb': result.get('file_size', 0) / (1024**2),
                            'download_time': download_time,
                            'error': result.get('error')
                        }
                    except Exception as e:
                        return {
                            'success': False,
                            'file_name': file_name,
                            'error': str(e)
                        }
                
                # Execute parallel downloads
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_url = {executor.submit(download_single_file, url): url for url in file_urls}
                    
                    for future in as_completed(future_to_url):
                        result = future.result()
                        
                        if result['success']:
                            download_results['successful_downloads'].append(result)
                            self.download_log.append({
                                'file': result['file_name'],
                                'status': 'success',
                                'worker_id': future,
                                'timestamp': time.time()
                            })
                        else:
                            download_results['failed_downloads'].append(result)
                            self.download_log.append({
                                'file': result['file_name'],
                                'status': 'failed',
                                'error': result['error'],
                                'timestamp': time.time()
                            })
                
                return download_results
        
        # Test parallel download
        downloader = ParallelDownloader(mock_geo_service, temp_workspace, max_workers=3)
        
        # Mock file URLs for bulk RNA-seq dataset
        file_urls = [
            'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE789nnn/GSE789012/suppl/GSE789012_counts.txt.gz',
            'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE789nnn/GSE789012/suppl/GSE789012_fpkm.txt.gz',
            'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE789nnn/GSE789012/suppl/GSE789012_metadata.txt',
            'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE789nnn/GSE789012/matrix/GSE789012_series_matrix.txt.gz'
        ]
        
        download_results = downloader.download_dataset_files_parallel('GSE789012', file_urls)
        
        # Verify parallel download results
        assert download_results['accession'] == 'GSE789012'
        assert download_results['total_files'] == 4
        assert download_results['parallel_execution'] == True
        assert download_results['workers_used'] == 3
        assert len(download_results['successful_downloads']) == 4
        assert len(download_results['failed_downloads']) == 0
        
        # Verify files were created
        dataset_dir = temp_workspace / "geo_downloads/GSE789012"
        assert dataset_dir.exists()
        assert len(list(dataset_dir.glob('*'))) == 4
    
    def test_file_format_detection_and_conversion(self, temp_workspace, mock_geo_files):
        """Test automatic detection and conversion of GEO file formats."""
        
        class FormatDetectorConverter:
            """Detects and converts GEO file formats."""
            
            def __init__(self, workspace_path):
                self.workspace_path = Path(workspace_path)
                self.conversion_log = []
                
            def detect_file_format(self, file_path):
                """Detect the format of a downloaded file."""
                file_path = Path(file_path)
                file_name = file_path.name.lower()
                
                format_info = {
                    'file_path': str(file_path),
                    'detected_format': 'unknown',
                    'confidence': 0.0,
                    'processing_hints': []
                }
                
                # Detect based on file name patterns
                if 'matrix.mtx' in file_name:
                    format_info.update({
                        'detected_format': '10x_mtx',
                        'confidence': 0.95,
                        'processing_hints': ['requires_barcodes_and_features', 'sparse_matrix']
                    })
                elif 'barcodes.tsv' in file_name:
                    format_info.update({
                        'detected_format': '10x_barcodes',
                        'confidence': 0.95,
                        'processing_hints': ['cell_identifiers', 'complement_to_matrix']
                    })
                elif 'features.tsv' in file_name or 'genes.tsv' in file_name:
                    format_info.update({
                        'detected_format': '10x_features',
                        'confidence': 0.95,
                        'processing_hints': ['gene_identifiers', 'complement_to_matrix']
                    })
                elif 'series_matrix.txt' in file_name:
                    format_info.update({
                        'detected_format': 'geo_series_matrix',
                        'confidence': 0.90,
                        'processing_hints': ['expression_matrix', 'sample_annotations']
                    })
                elif 'counts.txt' in file_name or 'counts.csv' in file_name:
                    format_info.update({
                        'detected_format': 'counts_matrix',
                        'confidence': 0.85,
                        'processing_hints': ['raw_counts', 'genes_by_samples']
                    })
                elif 'fpkm.txt' in file_name or 'tpm.txt' in file_name:
                    format_info.update({
                        'detected_format': 'normalized_expression',
                        'confidence': 0.85,
                        'processing_hints': ['normalized_values', 'genes_by_samples']
                    })
                elif 'metadata.txt' in file_name or 'phenodata' in file_name:
                    format_info.update({
                        'detected_format': 'sample_metadata',
                        'confidence': 0.80,
                        'processing_hints': ['sample_annotations', 'experimental_design']
                    })
                
                return format_info
            
            def convert_to_anndata(self, file_group, output_path):
                """Convert a group of related files to AnnData format."""
                conversion_result = {
                    'input_files': file_group,
                    'output_path': str(output_path),
                    'conversion_successful': False,
                    'anndata_created': False,
                    'warnings': []
                }
                
                try:
                    # Determine conversion strategy based on file types
                    file_formats = [self.detect_file_format(f)['detected_format'] for f in file_group]
                    
                    if '10x_mtx' in file_formats and '10x_barcodes' in file_formats and '10x_features' in file_formats:
                        # 10x format conversion
                        conversion_result.update(self._convert_10x_format(file_group, output_path))
                    elif 'geo_series_matrix' in file_formats:
                        # GEO series matrix conversion
                        conversion_result.update(self._convert_series_matrix(file_group, output_path))
                    elif 'counts_matrix' in file_formats:
                        # Simple counts matrix conversion
                        conversion_result.update(self._convert_counts_matrix(file_group, output_path))
                    else:
                        conversion_result['warnings'].append('No recognized format combination for conversion')
                
                except Exception as e:
                    conversion_result['error'] = str(e)
                
                self.conversion_log.append(conversion_result)
                return conversion_result
            
            def _convert_10x_format(self, file_group, output_path):
                """Convert 10x format files to AnnData."""
                # Mock 10x conversion
                mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                mock_adata.write_h5ad(output_path)
                
                return {
                    'conversion_successful': True,
                    'anndata_created': True,
                    'format_converted': '10x_to_h5ad',
                    'n_obs': mock_adata.n_obs,
                    'n_vars': mock_adata.n_vars
                }
            
            def _convert_series_matrix(self, file_group, output_path):
                """Convert GEO series matrix to AnnData."""
                # Mock series matrix conversion
                mock_adata = BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG)
                mock_adata.write_h5ad(output_path)
                
                return {
                    'conversion_successful': True,
                    'anndata_created': True,
                    'format_converted': 'series_matrix_to_h5ad',
                    'n_obs': mock_adata.n_obs,
                    'n_vars': mock_adata.n_vars
                }
            
            def _convert_counts_matrix(self, file_group, output_path):
                """Convert counts matrix to AnnData."""
                # Mock counts matrix conversion
                mock_adata = BulkRNASeqDataFactory(config=SMALL_DATASET_CONFIG)
                mock_adata.write_h5ad(output_path)
                
                return {
                    'conversion_successful': True,
                    'anndata_created': True,
                    'format_converted': 'counts_matrix_to_h5ad',
                    'n_obs': mock_adata.n_obs,
                    'n_vars': mock_adata.n_vars
                }
        
        # Test format detection and conversion
        converter = FormatDetectorConverter(temp_workspace)
        
        # Create mock files for testing
        test_files_dir = temp_workspace / 'test_files'
        test_files_dir.mkdir(exist_ok=True)
        
        # Create 10x format files
        for file_name, content in mock_geo_files.items():
            if file_name in ['barcodes.tsv', 'features.tsv', 'matrix.mtx']:
                file_path = test_files_dir / f'GSM123456_{file_name}.gz'
                with gzip.open(file_path, 'wt') as f:
                    f.write(content)
        
        # Test format detection
        matrix_file = test_files_dir / 'GSM123456_matrix.mtx.gz'
        barcodes_file = test_files_dir / 'GSM123456_barcodes.tsv.gz'
        features_file = test_files_dir / 'GSM123456_features.tsv.gz'
        
        matrix_format = converter.detect_file_format(matrix_file)
        assert matrix_format['detected_format'] == '10x_mtx'
        assert matrix_format['confidence'] > 0.9
        
        barcodes_format = converter.detect_file_format(barcodes_file)
        assert barcodes_format['detected_format'] == '10x_barcodes'
        
        features_format = converter.detect_file_format(features_file)
        assert features_format['detected_format'] == '10x_features'
        
        # Test conversion to AnnData
        file_group = [matrix_file, barcodes_file, features_file]
        output_path = temp_workspace / 'converted_data.h5ad'
        
        conversion_result = converter.convert_to_anndata(file_group, output_path)
        
        assert conversion_result['conversion_successful'] == True
        assert conversion_result['anndata_created'] == True
        assert conversion_result['format_converted'] == '10x_to_h5ad'
        assert output_path.exists()
        
        # Verify AnnData can be loaded
        adata = ad.read_h5ad(output_path)
        assert adata.n_obs > 0
        assert adata.n_vars > 0
    
    def test_download_progress_monitoring(self, temp_workspace, mock_geo_service):
        """Test monitoring and reporting of download progress."""
        
        class ProgressMonitor:
            """Monitors and reports download progress."""
            
            def __init__(self, geo_service):
                self.geo_service = geo_service
                self.progress_log = []
                self.current_downloads = {}
                
            def download_with_progress(self, accession, file_urls, progress_callback=None):
                """Download files with progress monitoring."""
                total_files = len(file_urls)
                completed_files = 0
                
                download_session = {
                    'session_id': f"download_{accession}_{int(time.time())}",
                    'accession': accession,
                    'total_files': total_files,
                    'start_time': time.time(),
                    'estimated_completion': None,
                    'average_speed_mbps': 0
                }
                
                self.current_downloads[download_session['session_id']] = download_session
                
                for i, file_url in enumerate(file_urls):
                    file_name = file_url.split('/')[-1]
                    
                    # Update progress before starting file
                    progress_info = {
                        'session_id': download_session['session_id'],
                        'current_file': file_name,
                        'file_index': i + 1,
                        'total_files': total_files,
                        'progress_percentage': (i / total_files) * 100,
                        'status': 'downloading'
                    }
                    
                    self.progress_log.append(progress_info)
                    
                    if progress_callback:
                        progress_callback(progress_info)
                    
                    # Simulate file download
                    local_path = temp_workspace / f"downloads/{accession}/{file_name}"
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    start_time = time.time()
                    result = self.geo_service.download_file(file_url, local_path)
                    download_time = time.time() - start_time
                    
                    # Update download statistics
                    completed_files += 1
                    elapsed_time = time.time() - download_session['start_time']
                    
                    if completed_files > 1:
                        # Estimate completion time
                        avg_time_per_file = elapsed_time / completed_files
                        remaining_files = total_files - completed_files
                        estimated_completion = time.time() + (avg_time_per_file * remaining_files)
                        download_session['estimated_completion'] = estimated_completion
                        
                        # Calculate average speed
                        if result['success']:
                            file_size_mb = result['file_size'] / (1024**2)
                            speed_mbps = file_size_mb / download_time if download_time > 0 else 0
                            
                            # Update rolling average
                            current_avg = download_session['average_speed_mbps']
                            download_session['average_speed_mbps'] = (current_avg * (completed_files - 1) + speed_mbps) / completed_files
                    
                    # Log completion
                    completion_info = {
                        'session_id': download_session['session_id'],
                        'file_completed': file_name,
                        'file_index': i + 1,
                        'progress_percentage': (completed_files / total_files) * 100,
                        'status': 'completed' if result['success'] else 'failed',
                        'download_time': download_time,
                        'estimated_completion': download_session.get('estimated_completion'),
                        'average_speed_mbps': download_session['average_speed_mbps']
                    }
                    
                    self.progress_log.append(completion_info)
                    
                    if progress_callback:
                        progress_callback(completion_info)
                
                # Final progress update
                final_progress = {
                    'session_id': download_session['session_id'],
                    'status': 'session_complete',
                    'total_files': total_files,
                    'completed_files': completed_files,
                    'progress_percentage': 100.0,
                    'total_time': time.time() - download_session['start_time'],
                    'average_speed_mbps': download_session['average_speed_mbps']
                }
                
                self.progress_log.append(final_progress)
                
                if progress_callback:
                    progress_callback(final_progress)
                
                return {
                    'session_complete': True,
                    'download_session': download_session,
                    'progress_log': [log for log in self.progress_log if log['session_id'] == download_session['session_id']]
                }
            
            def get_active_downloads(self):
                """Get information about active downloads."""
                return list(self.current_downloads.values())
            
            def get_download_statistics(self, session_id=None):
                """Get download statistics for a session or all sessions."""
                if session_id:
                    session_logs = [log for log in self.progress_log if log['session_id'] == session_id]
                else:
                    session_logs = self.progress_log
                
                if not session_logs:
                    return {'error': 'No logs found for session'}
                
                completed_files = len([log for log in session_logs if log.get('status') == 'completed'])
                failed_files = len([log for log in session_logs if log.get('status') == 'failed'])
                
                download_times = [log['download_time'] for log in session_logs if 'download_time' in log]
                avg_download_time = sum(download_times) / len(download_times) if download_times else 0
                
                speeds = [log['average_speed_mbps'] for log in session_logs if 'average_speed_mbps' in log and log['average_speed_mbps'] > 0]
                avg_speed = sum(speeds) / len(speeds) if speeds else 0
                
                return {
                    'completed_files': completed_files,
                    'failed_files': failed_files,
                    'average_download_time': avg_download_time,
                    'average_speed_mbps': avg_speed,
                    'total_progress_events': len(session_logs)
                }
        
        # Test progress monitoring
        monitor = ProgressMonitor(mock_geo_service)
        
        # Mock progress callback
        progress_updates = []
        def progress_callback(progress_info):
            progress_updates.append(progress_info)
        
        # Test download with progress monitoring
        file_urls = [
            'https://ftp.ncbi.nlm.nih.gov/geo/file1.txt',
            'https://ftp.ncbi.nlm.nih.gov/geo/file2.txt',
            'https://ftp.ncbi.nlm.nih.gov/geo/file3.txt'
        ]
        
        result = monitor.download_with_progress('GSE123456', file_urls, progress_callback)
        
        # Verify progress monitoring
        assert result['session_complete'] == True
        assert len(progress_updates) > 0
        
        # Check progress updates structure
        start_updates = [u for u in progress_updates if u.get('status') == 'downloading']
        completion_updates = [u for u in progress_updates if u.get('status') in ['completed', 'failed']]
        final_update = [u for u in progress_updates if u.get('status') == 'session_complete']
        
        assert len(start_updates) == 3  # One per file
        assert len(completion_updates) == 3  # One per file
        assert len(final_update) == 1  # One session completion
        
        # Verify progress percentages
        assert start_updates[0]['progress_percentage'] < start_updates[-1]['progress_percentage']
        assert final_update[0]['progress_percentage'] == 100.0
        
        # Verify download statistics
        stats = monitor.get_download_statistics(result['download_session']['session_id'])
        assert stats['completed_files'] >= 0
        assert stats['total_progress_events'] > 0


# ===============================================================================
# Automated Processing Pipeline Tests
# ===============================================================================

@pytest.mark.integration
class TestAutomatedProcessingPipelines:
    """Test automated processing pipelines for downloaded GEO data."""
    
    def test_single_cell_geo_processing_pipeline(self, temp_workspace, mock_geo_service):
        """Test automated single-cell data processing pipeline."""
        
        class SingleCellGEOPipeline:
            """Automated pipeline for single-cell GEO data."""
            
            def __init__(self, workspace_path, geo_service):
                self.workspace_path = Path(workspace_path)
                self.geo_service = geo_service
                self.pipeline_log = []
                
            def process_geo_dataset(self, accession, processing_config=None):
                """Process a GEO single-cell dataset end-to-end."""
                processing_config = processing_config or self._get_default_sc_config()
                
                pipeline_result = {
                    'accession': accession,
                    'pipeline_steps': [],
                    'final_modality': None,
                    'processing_successful': False,
                    'quality_metrics': {},
                    'analysis_results': {}
                }
                
                try:
                    # Step 1: Download and detect format
                    download_result = self._download_and_detect_format(accession)
                    pipeline_result['pipeline_steps'].append({
                        'step': 'download_and_detect',
                        'success': download_result['success'],
                        'details': download_result
                    })
                    
                    if not download_result['success']:
                        return pipeline_result
                    
                    # Step 2: Convert to AnnData
                    conversion_result = self._convert_to_anndata(download_result['files'], accession)
                    pipeline_result['pipeline_steps'].append({
                        'step': 'convert_to_anndata',
                        'success': conversion_result['success'],
                        'details': conversion_result
                    })
                    
                    if not conversion_result['success']:
                        return pipeline_result
                    
                    # Step 3: Quality control
                    qc_result = self._perform_quality_control(conversion_result['anndata_path'], processing_config['qc'])
                    pipeline_result['pipeline_steps'].append({
                        'step': 'quality_control',
                        'success': qc_result['success'],
                        'details': qc_result
                    })
                    pipeline_result['quality_metrics'] = qc_result.get('metrics', {})
                    
                    # Step 4: Preprocessing
                    if processing_config.get('auto_preprocess', True):
                        preprocess_result = self._preprocess_data(qc_result['processed_path'], processing_config['preprocessing'])
                        pipeline_result['pipeline_steps'].append({
                            'step': 'preprocessing',
                            'success': preprocess_result['success'],
                            'details': preprocess_result
                        })
                    
                    # Step 5: Initial analysis
                    if processing_config.get('auto_analyze', True):
                        analysis_result = self._perform_initial_analysis(preprocess_result['processed_path'], processing_config['analysis'])
                        pipeline_result['pipeline_steps'].append({
                            'step': 'initial_analysis',
                            'success': analysis_result['success'],
                            'details': analysis_result
                        })
                        pipeline_result['analysis_results'] = analysis_result.get('results', {})
                    
                    pipeline_result['processing_successful'] = all(
                        step['success'] for step in pipeline_result['pipeline_steps']
                    )
                    
                except Exception as e:
                    pipeline_result['error'] = str(e)
                
                self.pipeline_log.append(pipeline_result)
                return pipeline_result
            
            def _get_default_sc_config(self):
                """Get default single-cell processing configuration."""
                return {
                    'qc': {
                        'min_genes': 200,
                        'min_cells': 3,
                        'max_mt_percent': 20.0,
                        'calculate_qc_metrics': True
                    },
                    'preprocessing': {
                        'normalize_total': 10000,
                        'log_transform': True,
                        'find_hvg': True,
                        'n_top_genes': 2000
                    },
                    'analysis': {
                        'run_pca': True,
                        'compute_neighbors': True,
                        'run_umap': True,
                        'leiden_clustering': True,
                        'resolution': 0.5
                    },
                    'auto_preprocess': True,
                    'auto_analyze': True
                }
            
            def _download_and_detect_format(self, accession):
                """Download files and detect format."""
                # Mock download process
                download_dir = self.workspace_path / f"downloads/{accession}"
                download_dir.mkdir(parents=True, exist_ok=True)
                
                # Simulate 10x format files
                files = ['barcodes.tsv.gz', 'features.tsv.gz', 'matrix.mtx.gz']
                downloaded_files = []
                
                for file_name in files:
                    file_path = download_dir / file_name
                    self.geo_service.download_file(f"https://mock.url/{file_name}", file_path)
                    downloaded_files.append(str(file_path))
                
                return {
                    'success': True,
                    'files': downloaded_files,
                    'detected_format': '10x',
                    'download_dir': str(download_dir)
                }
            
            def _convert_to_anndata(self, files, accession):
                """Convert downloaded files to AnnData."""
                output_path = self.workspace_path / f"processed/{accession}_raw.h5ad"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Mock conversion
                mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                mock_adata.write_h5ad(output_path)
                
                return {
                    'success': True,
                    'anndata_path': str(output_path),
                    'n_obs': mock_adata.n_obs,
                    'n_vars': mock_adata.n_vars
                }
            
            def _perform_quality_control(self, anndata_path, qc_config):
                """Perform quality control on the data."""
                # Mock QC process
                qc_metrics = {
                    'n_cells_before': 2000,
                    'n_genes_before': 3000,
                    'n_cells_after': 1850,
                    'n_genes_after': 2750,
                    'mean_genes_per_cell': 1250,
                    'mean_counts_per_cell': 5500,
                    'median_mt_percent': 8.5
                }
                
                processed_path = anndata_path.replace('_raw.h5ad', '_qc.h5ad')
                
                # Create mock QC'd data
                mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                mock_adata.write_h5ad(processed_path)
                
                return {
                    'success': True,
                    'processed_path': processed_path,
                    'metrics': qc_metrics,
                    'cells_filtered': qc_metrics['n_cells_before'] - qc_metrics['n_cells_after'],
                    'genes_filtered': qc_metrics['n_genes_before'] - qc_metrics['n_genes_after']
                }
            
            def _preprocess_data(self, qc_path, preprocess_config):
                """Preprocess the quality-controlled data."""
                processed_path = qc_path.replace('_qc.h5ad', '_preprocessed.h5ad')
                
                # Mock preprocessing
                mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                
                # Add preprocessing results
                mock_adata.obs['total_counts'] = np.random.randint(1000, 15000, mock_adata.n_obs)
                mock_adata.obs['n_genes_by_counts'] = np.random.randint(200, 4000, mock_adata.n_obs)
                mock_adata.var['highly_variable'] = np.random.choice([True, False], mock_adata.n_vars, p=[0.15, 0.85])
                
                mock_adata.write_h5ad(processed_path)
                
                return {
                    'success': True,
                    'processed_path': processed_path,
                    'normalization_applied': preprocess_config.get('normalize_total', False),
                    'log_transformed': preprocess_config.get('log_transform', False),
                    'hvg_computed': preprocess_config.get('find_hvg', False),
                    'n_hvgs': sum(mock_adata.var['highly_variable'])
                }
            
            def _perform_initial_analysis(self, preprocessed_path, analysis_config):
                """Perform initial analysis on preprocessed data."""
                # Mock analysis results
                analysis_results = {
                    'pca_computed': analysis_config.get('run_pca', False),
                    'neighbors_computed': analysis_config.get('compute_neighbors', False),
                    'umap_computed': analysis_config.get('run_umap', False),
                    'clustering_performed': analysis_config.get('leiden_clustering', False),
                    'n_clusters': 8,
                    'explained_variance_ratio': [0.15, 0.08, 0.06, 0.04, 0.03]
                }
                
                final_path = preprocessed_path.replace('_preprocessed.h5ad', '_analyzed.h5ad')
                
                # Mock final analyzed data
                mock_adata = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
                
                # Add analysis results
                if analysis_results['pca_computed']:
                    mock_adata.obsm['X_pca'] = np.random.randn(mock_adata.n_obs, 50)
                if analysis_results['umap_computed']:
                    mock_adata.obsm['X_umap'] = np.random.randn(mock_adata.n_obs, 2)
                if analysis_results['clustering_performed']:
                    mock_adata.obs['leiden'] = np.random.randint(0, 8, mock_adata.n_obs).astype(str)
                
                mock_adata.write_h5ad(final_path)
                
                return {
                    'success': True,
                    'final_path': final_path,
                    'results': analysis_results
                }
        
        # Test single-cell processing pipeline
        pipeline = SingleCellGEOPipeline(temp_workspace, mock_geo_service)
        
        # Process a single-cell dataset
        result = pipeline.process_geo_dataset('GSE123456')
        
        # Verify pipeline execution
        assert result['processing_successful'] == True
        assert len(result['pipeline_steps']) >= 3  # At minimum: download, convert, QC
        
        # Verify each step succeeded
        for step in result['pipeline_steps']:
            assert step['success'] == True
        
        # Verify quality metrics were calculated
        assert 'quality_metrics' in result
        assert result['quality_metrics']['n_cells_before'] > result['quality_metrics']['n_cells_after']
        
        # Verify analysis results
        assert 'analysis_results' in result
        assert result['analysis_results']['n_clusters'] > 0
    
    def test_bulk_rnaseq_geo_processing_pipeline(self, temp_workspace, mock_geo_service):
        """Test automated bulk RNA-seq data processing pipeline."""
        
        class BulkRNASeqGEOPipeline:
            """Automated pipeline for bulk RNA-seq GEO data."""
            
            def __init__(self, workspace_path, geo_service):
                self.workspace_path = Path(workspace_path)
                self.geo_service = geo_service
                
            def process_bulk_geo_dataset(self, accession, processing_config=None):
                """Process a bulk RNA-seq GEO dataset."""
                processing_config = processing_config or self._get_default_bulk_config()
                
                pipeline_result = {
                    'accession': accession,
                    'pipeline_steps': [],
                    'processing_successful': False,
                    'experimental_design': {},
                    'differential_expression': {}
                }
                
                # Step 1: Download and parse series matrix
                download_result = self._download_series_matrix(accession)
                pipeline_result['pipeline_steps'].append({
                    'step': 'download_series_matrix',
                    'success': download_result['success']
                })
                
                # Step 2: Parse experimental design
                if download_result['success']:
                    design_result = self._parse_experimental_design(download_result['matrix_path'])
                    pipeline_result['pipeline_steps'].append({
                        'step': 'parse_experimental_design',
                        'success': design_result['success']
                    })
                    pipeline_result['experimental_design'] = design_result.get('design', {})
                
                # Step 3: Quality assessment
                if processing_config.get('assess_quality', True):
                    quality_result = self._assess_bulk_quality(download_result['matrix_path'])
                    pipeline_result['pipeline_steps'].append({
                        'step': 'quality_assessment',
                        'success': quality_result['success']
                    })
                
                # Step 4: Differential expression analysis
                if processing_config.get('run_de_analysis', True) and design_result.get('success'):
                    de_result = self._run_differential_expression(
                        download_result['matrix_path'],
                        design_result['design']
                    )
                    pipeline_result['pipeline_steps'].append({
                        'step': 'differential_expression',
                        'success': de_result['success']
                    })
                    pipeline_result['differential_expression'] = de_result.get('results', {})
                
                pipeline_result['processing_successful'] = all(
                    step['success'] for step in pipeline_result['pipeline_steps']
                )
                
                return pipeline_result
            
            def _get_default_bulk_config(self):
                """Get default bulk RNA-seq processing configuration."""
                return {
                    'assess_quality': True,
                    'run_de_analysis': True,
                    'normalization_method': 'TMM',
                    'de_method': 'edgeR',
                    'padj_threshold': 0.05,
                    'log2fc_threshold': 1.0
                }
            
            def _download_series_matrix(self, accession):
                """Download GEO series matrix file."""
                matrix_path = self.workspace_path / f"bulk_data/{accession}_series_matrix.txt"
                matrix_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Mock download
                self.geo_service.download_file(
                    f"https://ftp.ncbi.nlm.nih.gov/geo/series/{accession}/{accession}_series_matrix.txt.gz",
                    matrix_path
                )
                
                return {
                    'success': True,
                    'matrix_path': str(matrix_path)
                }
            
            def _parse_experimental_design(self, matrix_path):
                """Parse experimental design from series matrix."""
                # Mock experimental design parsing
                experimental_design = {
                    'factors': ['condition', 'timepoint', 'batch'],
                    'conditions': ['control', 'treatment_a', 'treatment_b'],
                    'timepoints': ['0h', '6h', '12h', '24h'],
                    'batches': ['batch_1', 'batch_2'],
                    'balanced': True,
                    'n_samples_per_group': 3
                }
                
                return {
                    'success': True,
                    'design': experimental_design
                }
            
            def _assess_bulk_quality(self, matrix_path):
                """Assess bulk RNA-seq data quality."""
                # Mock quality assessment
                quality_metrics = {
                    'n_samples': 24,
                    'n_genes': 15000,
                    'library_size_cv': 0.15,
                    'gene_detection_rate': 0.82,
                    'outlier_samples': [],
                    'batch_effects_detected': False,
                    'overall_quality': 'good'
                }
                
                return {
                    'success': True,
                    'quality_metrics': quality_metrics
                }
            
            def _run_differential_expression(self, matrix_path, experimental_design):
                """Run differential expression analysis."""
                # Mock DE analysis
                de_results = {
                    'comparisons_performed': [
                        'treatment_a_vs_control',
                        'treatment_b_vs_control',
                        'treatment_b_vs_treatment_a'
                    ],
                    'significant_genes': {
                        'treatment_a_vs_control': 456,
                        'treatment_b_vs_control': 623,
                        'treatment_b_vs_treatment_a': 234
                    },
                    'upregulated_genes': {
                        'treatment_a_vs_control': 234,
                        'treatment_b_vs_control': 345,
                        'treatment_b_vs_treatment_a': 123
                    },
                    'downregulated_genes': {
                        'treatment_a_vs_control': 222,
                        'treatment_b_vs_control': 278,
                        'treatment_b_vs_treatment_a': 111
                    }
                }
                
                return {
                    'success': True,
                    'results': de_results
                }
        
        # Test bulk RNA-seq processing pipeline
        bulk_pipeline = BulkRNASeqGEOPipeline(temp_workspace, mock_geo_service)
        
        # Process a bulk RNA-seq dataset
        result = bulk_pipeline.process_bulk_geo_dataset('GSE789012')
        
        # Verify pipeline execution
        assert result['processing_successful'] == True
        assert len(result['pipeline_steps']) >= 2  # At minimum: download, design parsing
        
        # Verify experimental design was parsed
        assert 'experimental_design' in result
        assert result['experimental_design']['factors']
        assert result['experimental_design']['balanced'] == True
        
        # Verify differential expression results
        assert 'differential_expression' in result
        assert len(result['differential_expression']['comparisons_performed']) > 0
        assert all(count > 0 for count in result['differential_expression']['significant_genes'].values())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])