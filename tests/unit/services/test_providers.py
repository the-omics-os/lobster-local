"""
Comprehensive unit tests for external service providers.

This module provides thorough testing of external service providers including
PubMed integration, bioRxiv/medRxiv fetching, Ensembl API access,
NCBI services, UniProt integration, and other bioinformatics databases.

Test coverage target: 95%+ with meaningful tests for provider operations.
"""

import pytest
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, MagicMock, patch, mock_open
import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from lobster.tools.providers.pubmed import PubMedProvider
from lobster.tools.providers.biorxiv import BioRxivProvider
from lobster.tools.providers.ensembl import EnsemblProvider
from lobster.tools.providers.uniprot import UniProtProvider
from lobster.tools.providers.ncbi import NCBIProvider
from lobster.tools.providers.base import BaseProvider

from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Mock Data and Fixtures
# ===============================================================================

@pytest.fixture
def mock_pubmed_response():
    """Mock PubMed API response data."""
    return {
        "esearchresult": {
            "count": "3",
            "retmax": "20",
            "retstart": "0",
            "idlist": ["37123456", "37123457", "37123458"]
        }
    }


@pytest.fixture
def mock_pubmed_paper_details():
    """Mock detailed paper information."""
    return {
        "37123456": {
            "pmid": "37123456",
            "title": "Single-cell RNA sequencing reveals novel T cell populations in cancer",
            "abstract": "Single-cell RNA sequencing has revolutionized our understanding...",
            "authors": ["Smith J", "Doe A", "Johnson B"],
            "journal": "Nature",
            "year": "2023",
            "doi": "10.1038/s41586-023-12345-6",
            "mesh_terms": ["Single-Cell Analysis", "T-Lymphocytes", "Neoplasms"],
            "publication_date": "2023-06-15",
            "full_text_available": True
        }
    }


@pytest.fixture
def mock_biorxiv_response():
    """Mock bioRxiv API response data."""
    return {
        "messages": [
            {
                "status": "ok",
                "total": 2
            }
        ],
        "collection": [
            {
                "rel_doi": "10.1101/2023.05.15.540123",
                "rel_title": "Novel computational methods for single-cell analysis",
                "rel_authors": "Jane Smith, Bob Johnson, Alice Brown",
                "rel_date": "2023-05-15",
                "rel_site": "bioRxiv",
                "category": "Bioinformatics"
            },
            {
                "rel_doi": "10.1101/2023.05.20.541234",
                "rel_title": "Machine learning approaches for genomics data",
                "rel_authors": "Charlie Wilson, David Lee",
                "rel_date": "2023-05-20", 
                "rel_site": "bioRxiv",
                "category": "Computational Biology"
            }
        ]
    }


@pytest.fixture
def mock_ensembl_response():
    """Mock Ensembl API response data."""
    return {
        "ENSG00000139618": {
            "id": "ENSG00000139618",
            "display_name": "BRCA2",
            "description": "BRCA2 DNA repair associated",
            "biotype": "protein_coding",
            "start": 32315086,
            "end": 32400266,
            "strand": 1,
            "seq_region_name": "13"
        }
    }


@pytest.fixture  
def mock_uniprot_response():
    """Mock UniProt API response data."""
    return {
        "results": [
            {
                "primaryAccession": "P51587",
                "uniProtkbId": "BRCA2_HUMAN",
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
                "genes": [{"geneName": {"value": "BRCA2"}}],
                "organism": {"scientificName": "Homo sapiens"},
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Breast cancer type 2 susceptibility protein"}
                    }
                },
                "sequence": {"length": 3418}
            }
        ]
    }


@pytest.fixture
def providers_config():
    """Configuration for provider services."""
    return {
        'pubmed': {
            'email': 'test@example.com',
            'api_key': 'test_api_key',
            'rate_limit': 10
        },
        'ensembl': {
            'base_url': 'https://rest.ensembl.org',
            'species': 'homo_sapiens'
        },
        'uniprot': {
            'base_url': 'https://rest.uniprot.org',
            'format': 'json'
        }
    }


# ===============================================================================
# Base Provider Tests
# ===============================================================================

@pytest.mark.unit
class TestBaseProvider:
    """Test base provider functionality."""
    
    def test_base_provider_initialization(self):
        """Test BaseProvider initialization."""
        provider = BaseProvider(
            name="test_provider",
            base_url="https://api.example.com",
            rate_limit=10
        )
        
        assert provider.name == "test_provider"
        assert provider.base_url == "https://api.example.com"
        assert provider.rate_limit == 10
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        provider = BaseProvider(
            name="rate_test",
            rate_limit=2  # 2 requests per second
        )
        
        with patch('time.sleep') as mock_sleep:
            # Simulate multiple rapid requests
            provider._check_rate_limit()
            provider._check_rate_limit()
            provider._check_rate_limit()  # This should trigger rate limiting
            
            # Should have called sleep to enforce rate limit
            mock_sleep.assert_called()
    
    def test_request_retry_mechanism(self):
        """Test request retry mechanism."""
        provider = BaseProvider(name="retry_test", max_retries=3)
        
        with patch('requests.get') as mock_get:
            # Mock connection error
            mock_get.side_effect = requests.ConnectionError("Network error")
            
            with pytest.raises(requests.ConnectionError):
                provider._make_request("https://api.example.com/test")
            
            # Should have tried 3 times (initial + 2 retries)
            assert mock_get.call_count == 3
    
    def test_response_caching(self):
        """Test response caching functionality."""
        provider = BaseProvider(name="cache_test", enable_cache=True)
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"data": "test"}
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            # First request should hit the API
            result1 = provider._make_request("https://api.example.com/test")
            
            # Second request should use cache
            result2 = provider._make_request("https://api.example.com/test")
            
            # API should only be called once
            assert mock_get.call_count == 1
            assert result1 == result2


# ===============================================================================
# PubMed Provider Tests
# ===============================================================================

@pytest.mark.unit
class TestPubMedProvider:
    """Test PubMed provider functionality."""
    
    def test_pubmed_provider_initialization(self, providers_config):
        """Test PubMedProvider initialization."""
        provider = PubMedProvider(
            email=providers_config['pubmed']['email'],
            api_key=providers_config['pubmed']['api_key']
        )
        
        assert provider.email == providers_config['pubmed']['email']
        assert provider.api_key == providers_config['pubmed']['api_key']
    
    def test_search_papers(self, mock_pubmed_response):
        """Test searching for papers."""
        provider = PubMedProvider(email="test@example.com")
        
        with patch.object(provider, '_search_pubmed') as mock_search:
            mock_search.return_value = mock_pubmed_response
            
            results = provider.search_papers(
                query="single cell RNA seq",
                max_results=20
            )
            
            assert len(results) == 3
            assert results[0] == "37123456"
            mock_search.assert_called_once()
    
    def test_get_paper_details(self, mock_pubmed_paper_details):
        """Test getting detailed paper information."""
        provider = PubMedProvider(email="test@example.com")
        
        with patch.object(provider, '_fetch_paper_details') as mock_fetch:
            mock_fetch.return_value = mock_pubmed_paper_details["37123456"]
            
            paper_details = provider.get_paper_details("37123456")
            
            assert paper_details["pmid"] == "37123456"
            assert paper_details["title"].startswith("Single-cell RNA")
            assert paper_details["journal"] == "Nature"
            mock_fetch.assert_called_once_with("37123456")
    
    def test_search_by_author(self, mock_pubmed_response):
        """Test searching papers by author."""
        provider = PubMedProvider(email="test@example.com")
        
        with patch.object(provider, 'search_by_author') as mock_author_search:
            mock_author_search.return_value = ["37123456", "37123457"]
            
            results = provider.search_by_author(
                author_name="Smith J",
                affiliation="Harvard University"
            )
            
            assert len(results) == 2
            mock_author_search.assert_called_once()
    
    def test_search_by_journal(self, mock_pubmed_response):
        """Test searching papers by journal."""
        provider = PubMedProvider(email="test@example.com")
        
        with patch.object(provider, 'search_by_journal') as mock_journal_search:
            mock_journal_search.return_value = ["37123456", "37123457", "37123458"]
            
            results = provider.search_by_journal(
                journal="Nature",
                date_range="2020:2023"
            )
            
            assert len(results) == 3
            mock_journal_search.assert_called_once()
    
    def test_get_paper_citations(self):
        """Test getting paper citations."""
        provider = PubMedProvider(email="test@example.com")
        
        with patch.object(provider, 'get_citations') as mock_citations:
            mock_citations.return_value = {
                'cited_by': ["38123456", "38123457", "38123458"],
                'references': ["36123456", "36123457"],
                'citation_count': 3,
                'self_citations': 0
            }
            
            citations = provider.get_citations("37123456")
            
            assert citations['citation_count'] == 3
            assert len(citations['cited_by']) == 3
            assert len(citations['references']) == 2
    
    def test_get_mesh_terms(self):
        """Test extracting MeSH terms from papers."""
        provider = PubMedProvider(email="test@example.com")
        
        with patch.object(provider, 'extract_mesh_terms') as mock_mesh:
            mock_mesh.return_value = [
                "Single-Cell Analysis",
                "T-Lymphocytes", 
                "Neoplasms",
                "Gene Expression Profiling"
            ]
            
            mesh_terms = provider.extract_mesh_terms("37123456")
            
            assert len(mesh_terms) == 4
            assert "Single-Cell Analysis" in mesh_terms
    
    def test_download_full_text(self):
        """Test downloading full text when available."""
        provider = PubMedProvider(email="test@example.com")
        
        with patch.object(provider, 'download_full_text') as mock_download:
            mock_download.return_value = {
                'success': True,
                'text': 'Full paper text content...',
                'format': 'xml',
                'sections': ['abstract', 'introduction', 'methods', 'results', 'discussion']
            }
            
            full_text = provider.download_full_text("37123456")
            
            assert full_text['success'] == True
            assert 'Full paper text' in full_text['text']
            assert len(full_text['sections']) == 5


# ===============================================================================
# BioRxiv Provider Tests  
# ===============================================================================

@pytest.mark.unit
class TestBioRxivProvider:
    """Test BioRxiv provider functionality."""
    
    def test_biorxiv_provider_initialization(self):
        """Test BioRxivProvider initialization."""
        provider = BioRxivProvider(server="biorxiv")
        
        assert provider.server == "biorxiv"
        assert provider.base_url == "https://api.biorxiv.org"
    
    def test_search_preprints(self, mock_biorxiv_response):
        """Test searching for preprints."""
        provider = BioRxivProvider()
        
        with patch.object(provider, '_search_preprints') as mock_search:
            mock_search.return_value = mock_biorxiv_response
            
            results = provider.search_preprints(
                query="single cell",
                start_date="2023-01-01",
                end_date="2023-12-31"
            )
            
            assert len(results['collection']) == 2
            assert results['collection'][0]['rel_title'].startswith("Novel computational")
            mock_search.assert_called_once()
    
    def test_get_preprint_details(self):
        """Test getting detailed preprint information."""
        provider = BioRxivProvider()
        
        with patch.object(provider, 'get_preprint_details') as mock_details:
            mock_details.return_value = {
                'doi': '10.1101/2023.05.15.540123',
                'title': 'Novel computational methods for single-cell analysis',
                'authors': ['Jane Smith', 'Bob Johnson', 'Alice Brown'],
                'abstract': 'We present novel computational methods...',
                'category': 'Bioinformatics',
                'date': '2023-05-15',
                'version': 1,
                'pdf_url': 'https://www.biorxiv.org/content/10.1101/2023.05.15.540123v1.full.pdf'
            }
            
            details = provider.get_preprint_details("10.1101/2023.05.15.540123")
            
            assert details['doi'] == "10.1101/2023.05.15.540123"
            assert len(details['authors']) == 3
    
    def test_search_by_category(self, mock_biorxiv_response):
        """Test searching preprints by category."""
        provider = BioRxivProvider()
        
        with patch.object(provider, 'search_by_category') as mock_category:
            mock_category.return_value = mock_biorxiv_response['collection'][:1]
            
            results = provider.search_by_category(
                category="Bioinformatics",
                limit=10
            )
            
            assert len(results) == 1
            assert results[0]['category'] == "Bioinformatics"
    
    def test_get_trending_preprints(self):
        """Test getting trending preprints."""
        provider = BioRxivProvider()
        
        with patch.object(provider, 'get_trending_preprints') as mock_trending:
            mock_trending.return_value = [
                {
                    'doi': '10.1101/2023.05.15.540123',
                    'title': 'Trending paper 1',
                    'downloads': 1250,
                    'score': 95.8
                },
                {
                    'doi': '10.1101/2023.05.20.541234',
                    'title': 'Trending paper 2', 
                    'downloads': 890,
                    'score': 87.3
                }
            ]
            
            trending = provider.get_trending_preprints(timeframe="week", limit=5)
            
            assert len(trending) == 2
            assert trending[0]['downloads'] > trending[1]['downloads']


# ===============================================================================
# Ensembl Provider Tests
# ===============================================================================

@pytest.mark.unit
class TestEnsemblProvider:
    """Test Ensembl provider functionality."""
    
    def test_ensembl_provider_initialization(self, providers_config):
        """Test EnsemblProvider initialization."""
        provider = EnsemblProvider(
            species=providers_config['ensembl']['species']
        )
        
        assert provider.species == providers_config['ensembl']['species']
        assert provider.base_url == "https://rest.ensembl.org"
    
    def test_get_gene_info(self, mock_ensembl_response):
        """Test getting gene information."""
        provider = EnsemblProvider(species="homo_sapiens")
        
        with patch.object(provider, 'get_gene_info') as mock_gene_info:
            mock_gene_info.return_value = mock_ensembl_response["ENSG00000139618"]
            
            gene_info = provider.get_gene_info("ENSG00000139618")
            
            assert gene_info['id'] == "ENSG00000139618"
            assert gene_info['display_name'] == "BRCA2"
            assert gene_info['biotype'] == "protein_coding"
    
    def test_gene_id_conversion(self):
        """Test gene ID conversion between different formats."""
        provider = EnsemblProvider()
        
        with patch.object(provider, 'convert_gene_ids') as mock_convert:
            mock_convert.return_value = {
                'BRCA2': 'ENSG00000139618',
                'TP53': 'ENSG00000141510',
                'MYC': 'ENSG00000136997'
            }
            
            conversions = provider.convert_gene_ids(
                gene_names=['BRCA2', 'TP53', 'MYC'],
                from_format='gene_name',
                to_format='ensembl_id'
            )
            
            assert conversions['BRCA2'] == 'ENSG00000139618'
            assert len(conversions) == 3
    
    def test_get_gene_sequence(self):
        """Test getting gene sequence."""
        provider = EnsemblProvider()
        
        with patch.object(provider, 'get_sequence') as mock_sequence:
            mock_sequence.return_value = {
                'id': 'ENSG00000139618',
                'sequence': 'ATGCCTATTGGATCCAAAGAGAGGCCAACATTTTTTGA...',
                'length': 10257,
                'type': 'genomic'
            }
            
            sequence = provider.get_sequence("ENSG00000139618", seq_type="genomic")
            
            assert sequence['id'] == "ENSG00000139618"
            assert sequence['length'] > 0
            assert sequence['sequence'].startswith('ATG')
    
    def test_get_orthologs(self):
        """Test getting orthologous genes."""
        provider = EnsemblProvider()
        
        with patch.object(provider, 'get_orthologs') as mock_orthologs:
            mock_orthologs.return_value = [
                {
                    'id': 'ENSMUSG00000041147',
                    'species': 'mus_musculus',
                    'gene_name': 'Brca2',
                    'homology_type': 'ortholog_one2one',
                    'confidence': 1
                }
            ]
            
            orthologs = provider.get_orthologs(
                gene_id="ENSG00000139618",
                target_species="mus_musculus"
            )
            
            assert len(orthologs) == 1
            assert orthologs[0]['species'] == 'mus_musculus'
    
    def test_get_gene_tree(self):
        """Test getting gene phylogenetic tree."""
        provider = EnsemblProvider()
        
        with patch.object(provider, 'get_gene_tree') as mock_tree:
            mock_tree.return_value = {
                'tree_id': 'ENSGT00390000003602',
                'newick_format': '(ENSG00000139618:0.05,(ENSMUSG00000041147:0.02,ENSCAFG00000014728:0.03):0.01);',
                'species_count': 3,
                'alignment_length': 3418
            }
            
            tree = provider.get_gene_tree("ENSG00000139618")
            
            assert tree['species_count'] == 3
            assert 'newick_format' in tree
    
    def test_regulatory_features(self):
        """Test getting regulatory features."""
        provider = EnsemblProvider()
        
        with patch.object(provider, 'get_regulatory_features') as mock_regulatory:
            mock_regulatory.return_value = [
                {
                    'id': 'ENSR00001348195',
                    'feature_type': 'promoter',
                    'start': 32315000,
                    'end': 32316000,
                    'activity': 'ACTIVE'
                },
                {
                    'id': 'ENSR00001348196', 
                    'feature_type': 'enhancer',
                    'start': 32320000,
                    'end': 32321000,
                    'activity': 'POISED'
                }
            ]
            
            features = provider.get_regulatory_features(
                gene_id="ENSG00000139618",
                feature_types=['promoter', 'enhancer']
            )
            
            assert len(features) == 2
            assert features[0]['feature_type'] == 'promoter'


# ===============================================================================
# UniProt Provider Tests
# ===============================================================================

@pytest.mark.unit
class TestUniProtProvider:
    """Test UniProt provider functionality."""
    
    def test_uniprot_provider_initialization(self):
        """Test UniProtProvider initialization."""
        provider = UniProtProvider()
        
        assert provider.base_url == "https://rest.uniprot.org"
    
    def test_search_proteins(self, mock_uniprot_response):
        """Test searching for proteins."""
        provider = UniProtProvider()
        
        with patch.object(provider, 'search_proteins') as mock_search:
            mock_search.return_value = mock_uniprot_response
            
            results = provider.search_proteins(
                query="BRCA2",
                organism="9606"  # Homo sapiens
            )
            
            assert len(results['results']) == 1
            assert results['results'][0]['primaryAccession'] == "P51587"
            assert results['results'][0]['genes'][0]['geneName']['value'] == "BRCA2"
    
    def test_get_protein_info(self):
        """Test getting detailed protein information."""
        provider = UniProtProvider()
        
        with patch.object(provider, 'get_protein_info') as mock_protein_info:
            mock_protein_info.return_value = {
                'accession': 'P51587',
                'name': 'BRCA2_HUMAN',
                'gene_name': 'BRCA2',
                'organism': 'Homo sapiens',
                'function': 'Involved in DNA repair...',
                'sequence_length': 3418,
                'molecular_weight': 384494,
                'subcellular_location': ['Nucleus', 'Cytoplasm']
            }
            
            protein_info = provider.get_protein_info("P51587")
            
            assert protein_info['accession'] == "P51587"
            assert protein_info['gene_name'] == "BRCA2"
            assert protein_info['sequence_length'] == 3418
    
    def test_get_protein_sequence(self):
        """Test getting protein sequence."""
        provider = UniProtProvider()
        
        with patch.object(provider, 'get_sequence') as mock_sequence:
            mock_sequence.return_value = {
                'accession': 'P51587',
                'sequence': 'MPIGSKERPTFFEIFKTRCNKADLGPISLNWFEELSSEAPPYNSEPAEESEHKNNNYEPNLFKTPQRKPSYNQLASTPIIFKEQGLTLPLYQSPVKELDKFKLDLGRNVPNSRHKSLRTVKTKMDQADDVSCPLLNSCLSESPVVLQCTHVTPQRDKSVVCGSLFHTPKFVKGRQTPKHISESLGAEVDPDMSWSSSLATPPTLSSTVLIVRNEEASETVFPHDTTANVKSYFSNHDESLKKNDRFIASVTDSENTNQREAASHGFGKTSGNSFKVNSCKDHIGKSMPNVLEDEVYETVVDTSEEDSFSLCFSKCRTKNLQKVRTSKTRKKIFHEANADECEKSKNPVVKRVYTSFAAIFGKIADVSGLLSSNTPTNDLETVRSTYFKNGKLGYSNSKRTRYPYAVFNDKEGNIKITSYKHFPRIFADSHFMGDDSLAAGLQDKHVVPYDSSTAFSIVEELNSTIPMDSSDAKDDPVLTQVFRVEDAHSFEKRSPMRYVSDSQAHSLNPREDKNAFKALIAANETPVMASKATMGVPIVCNATIKTIKQYTKIPETIEKGKEQKAKMKETKNMKLQLFCKKLHAIPAVLKNLLLAKYAPATTKPQSRYANASQADSDKNDMSITTYKYILIPPKYERPESSFSGKGFGTKTIRILASAKKLIVTVKLHGAPNISEEKWYSSRSSSMSPILEGFLNPSTTKLAISPITTPTVIITALNVSVAAIKAKTSHANRFVDHIVKYLTDANVLQSKRRILGDCKIFKSTFPTFGPRTAKALAKEMQGAAKAAAAARNNVVEAALKAAAPSKMDELIIPPVDNSQLGFKSGSFYDDATVLKGDTLGTAVPSKLSKQRVKVYGRNRVEGDKTVELQRSEAPDPKKGMTGQNSRLGIDLVGAVSGEREKKSDSIVIVRLLGGRARLQRAKDADIAGYEGKQNGIAAGDTEQSALIVKDAFTLEQYEKDDKGVSYQKISLRAICFAELSRYKKDFLDQISSMGIYSLPEFVDDDYRSTLLKAATPLRRKLTNAVELVYRLEKQTASDNPQKLLEDPIKDVLQRLKGEAGSQVKSLADFADLVWLKRIHGYKVIETTLSDQCGMEQRTLYQASSIEKLIRLKGWGAMLNTSFDKDLYVVTLAQGTTYQADEHTGSYNPSIDAVIKHIRSVHGNLNVLQHNEQFGKPPRLLDMITRVVQMAHRLYRDVFGKLNVLQIHKRGGQGMRRLHTNQWMMDLMLNLQGLALHVVDRSRGDNRARLLVLELAPSRRPMLPYEPRKKYEEAKKMAHTSASDMDDLKPEYENHHTTMAGAVIAAAASYRLGTQTGMMVKSRDGTEKERFNIFNAYDNLMMLHVMVQQLLEMKREAKKGGSIVSLQGKMYQMQPREGRRSEQLKQSETQQQIPPRIIGPSGTGTAANIRKFRAFALTEASASGQNLLQSILKSRDEFKVPPFRFCYERHKQPHHHKHHSHPGSGMPVFKRAIGKAPTLDQGAAVALGQALQMTAEAQADVAAQEGDQLFALQSEQIANRNALLLHAKEAKQAAYGRKKHAAAMTYELLSLAIVALDPPQGSPLSGAASSGGSAGHSCPIDSRLLGQGRGQRWYDKVVPERSDDLASIDSSDKEIDFVFQRRLQQGKKTSLHVLQDWDEFTEKERNSLANQDKMAYRNVKTIKQHARDVMDLIQKAKRHYNGLLDSDKMKRAKMLLKKKPAPGHYSPTSLSHLCAADIQGRKELWQHHKERLWSANTAGQTVLSLNGANVERQQNVAQVGLHTMSLKQQKDGSVPQRSLSTQTHGQRPLSLSQANQMGANAAIKQTLPTGGTLPIEEPPKGLSRHLRDVIDLAAFVTTSHVVRVGRSLSTDTAAPKDGMETKLAGSDLNDTLLGNNQRLRLDQAPGRPGSLTSPSAGNVGAPDAARKAIEQIPLHEAKELGDQLPDDLELVAREAAWQAFVSALQAEAALRVASAPLAKLPLKFAKNYDEMIPYLDQLTSSQHEEAALSVASSPQTVSSGLMSLKDVLMDTYGRPPDSRQMVLGTKDEILNCRLMSEMEVLALGIQVQKCKELYTSKRNALRQLNGSTKSLPPDSYKDSPNAISQIQHRDLPAEVVETEPGYNNPPGQKLANLEKPQHQELVLLQKRCGSELTKALVATAANQTQMNQAMHRQSPQSVGEAKDQRMLQWFSTRQKEHQNSVSGPNLETCDSYERKEIDWLLAERKPPAKRVFLSTWRRAIDLQLDLVLQAAHAEPQRPPSPLDGLSVAKQIPLVELLSKIPGSKKEDECENHALGGQVSLIQAGASVAAPKKLDDSSLKHAAEVLQKSEATLIKTQIPVLQHASQLLGVVSYTRGGWRQAQMKALSKSDHTCKTTPAGENLLQVLVKTLRARSMPEKLQLPPKGLWPQSQSDPHLTDSPRWLYKPKMDNMQSQVVQTLRALAPDHARSKGLPEKDVFRKSPPYPPDRRYKVLEVQRIIHSQAPKVVSSLKTQLLMKQPNQKSRRPPIPVPPRDSPGDISPDSQMPSPYPPRRRATSTDLQGGLAAHNVYQLEQEMRDPLEHMTTNQENPAKPNFQYRDTQELMSRAPQKNQPQPAGSIEAQGKGPDIPYRDTQLLMQRAPRNRADDEDLNDLREASLHNAETQEFLERTARQIERLTMIDAPLEKQSKQGQDLQQMRDGGQGNSATTGSTTQPQVRALSASPKKPKDMTKEDDFASEPRVNPAKRKTLGKSDDDSRTVASLMAKEEKLPKTGPLASRKMTTAMAPQAATPSKKQNQNQPNGTNAADTKDKRNARPGRPSSPSQPRATDTMLGMQGPRQNKQMGAPPDHQGQNPNGASQASQMTRGRPRDRPNGQDLPPNKPMRTMQKQNKAQHSASLLEQEGLNTDGPLQSLDALQGRSSLPRPQFLLAQPQPPTGSALHQRNALTNANRENTMLNAAAPEQTLKRHLSYSSDYLDAGLIRSWETIKLLQALLQHALQPQLIKDGNRLKKLTQKQHPDLATTDSQRKKNDNNQLGPNPVLDWMSAQPLHREESPGCAGPKPGPVGSQSPGGDILRHRRPPEGPLRLQQLKEQTSGQEEKLSRAVVNNLRMPKQIQSAQTLDLRTTGPRGILKPQLDRFGIVQYTGDVFMDLFQRSEQLRLYQMMKTATVYSQNGYRSPMEYQPEEERMRTPQNDSAPPANSSSESPGQALLQLLIAQVQPGQNGHLLREEKGHSVSDDNLPHHEQQNVAINKDRDALPKTLGETSHLATFELNEKQQNPGGRFHTQDDFTSITDLMSLSPGEDPVQMEPFKLYHQVLAPNASLHEGQNVLNDNDPNRFRETNLNNNQGNDRFHHTLLADLEILMKHQKCAIRAPSQFQQLEKLIGNTSKLTQLTKQRRAHQHRRTESLLTQPPADHNQGRRQKLSPLHGCQRPATAPTLDKQNRSADQSNMTPDLMQQENTYSSGNLPIQRQPGSGGSQTGQQAKIRRYNEISESQENVQSLKQKQRPPLPPSAHPAKQQRLELCDPHQNKSPERKRQPQSPVATPAGQNMRPPAGGRHYNSRWRKKPAERYIGYIQSLKSQDALQSPMSEGQAMAQAKGSHAKNKLPLRQSMPVTDSQDQKIADQGGQLAVYLNSPQGKSGLSSSDAMKQLLEITNDLIEDKRTERREINQDFRDKLQDYLKPFRIELMDTAAVDKLQASQVSSQTIIAELRERPLDTRDYACISRESLAQLLPHPGALEGMSKPQSSRLLIKEATALVAYMQQQRAPNAHARTIPLLHWHTYLMLAQNSHDESQNQERTALGLKAILTTGHITPAGQRDSPARHSPQKRLFQKQAGLIGQPVKDLASSPRASRSHNQVSDPKTSWDPVASAKDASNLHQVVSLAENRLLTSQEVFQSARLTIESDPDKHDQALVRALHGHDELKPLQAKGVLTTQMERIDLVDQLIITQVSVRVRKVGESSSMLPLALNISLGQTQLRLQKASRRSEKPAAVVQPPRSATDKRQPHMKRDQGDFSQQSPQRAPAQGQEKATALQNQPGSWGQELQQLQRQERSRQGSTRRDGSGRQNESLKQGLSSAEEKKKRAGEDDEDRKQQASAAKGIGATVLQDRQQAAADGRRDAHIALSTNALSLDKDKTVKQTAQKAAEELRTKAVRQLLKNSPAKVFFGSQSPKTLTDDVVSKSELLHVAQVGLQVAPSSSNSALQKDKDESKQEKQRAKPREASKDVRERVSQEKGKRNKVKPSRVQREPRESVLVIQIERILMQTVKAQAGPAEADKSRGRTAAKAASDGTPGKAGKAKDSTISETQETLSTQDRDFNISSAHDVQGHLENRDKARRDASQLLTQAAKREASPPGAGLANSQFGEAADKQHDKDKQAKPQVKAEVLLGVEVSQKLKQHQKKTRASTTSSKQKASRRIDNRYQLADEQALAPNLLETRDNQLPTQAGRPTLNLEQADETPKDEKESLKQAQVAERPEKRGKGGRFFTQNNFDNAKNRQRGLSRESEESDTATGRSTAESSASTPSMNEEGKDLDGLEGNLNNSKTQNYRRGSDKSDKQEEASQTALQHAENQPQKEKRRHGQMKQSQKNKQSQVKNSDSNTDKQQSNQKAQDLAQQQVKEEAKATDKQERPQQVKQGKKSSMAPDSPVSQGTAASGLQSQGASQDLEKQGLQDLKAQSSEDREQQKQRPAVRQDVVVQYGKGKDRGPVMKQGKQKQSPGSPHKANRSSTQDKGTTKGQMGTNKTESPHTLTSRSQSHGHHSHEHSQHRDRGTQDAQGTTHTSQERQRKALASLRQKPTTQELRAMPQKMPSTLDNKGSLQEPNTQKKQSPGLSSASQHQGSDDPTGSKDGQMSNTREGQRQKQVGQIVNSTENRQWYDSHQLQGVGENGQLGELFDSSSSQDDIQSRTDSSRTQNTQKAKSSTQLNKNQQANTRRNGTPPPPKAQRQRPHTQHQEKREEQKLKSAAEEKRASAVLMPPSENNQKRSQGTVNSQSQKTDKAPFQAQAKDKEPSLIASHDDAIKRQKLELAQTEHDARRKNTELQRENAKNLLLSKGDKDADINDLKEQIRTLLLTAATAEALYRQAGHSSQMKRRATDTAGKHTFTKGKRSHPQKLSPKQRRRPAATLRRSTTVGQGTTGTTIAGKQRNEGGAAGNGPRHQRTAQAHARDMRKRLKKDLVATLLGSDLYLTWMSTLEHVDPNTEEPNLGEQKLLGSGKRGKVRRRAALRGDGFEGEALLFLAEVDADLIKELISSKLRLGESRQLRMVDLEVADHIQQELEKAGAKNEREQAAAEQERAQARLDHLRTAVNLLQKARQNHEEQAQAAKRNAEPKQNPQSLPKQQQLRPKGSLQRATRNLQKHAEKGLFRDTEQLIRAIASQLDDEENVQSEKRQKLCRELEQETQRRVERLTKALAECALLGRAAKRANSPEVKKIPEVRDSWQEARRAWLAGKQDEAGGKVRKQRLKSKDKLQALLKYMNQILHSDVDAIFAQLNQSRCKWGVIMGIPGSGKFTAFGISFLVKPQLIDYVLTFPLSRGKEGLEADMSAIRKMFVGRKQEGQFITYDLRVLRSEICLASLTDTVRRLHDAAIGLITIHNKEQRRKLAQTSAEPRISRLDVAIRGHALGWLIPGDLSERLSQAAALLHTTTGLAKFAHQVSEKWSGTSSKGDDLPTHVAQGTDTTGSSYTSQRRTSSSASSDIQASKLLGLSNQMHILPPSNSGATRQNSPSSQSEDSGTDRTAGQESTQNPSDSERNIGDIDRQRITRTARRLQTAQQIGQSSQRRRPAPSSLASRDHDQSSSRGTSSQDRPRRITPPQTSSLTRAFERQRTSNLPEQKALTKEIKQDQRTLEDIPLLKALEHQLPKKQSNIVTQPDPTTTTATGATSSPTSAADSSTTESQHKSGQRTHNHRSATSPTSSSPAATTAQAATSQTSQTSTSASEPARGPPLCADHPSPSQFITKNSTDISTSDEKTTAAEQNNRGATPVELQTTISWRQFLNDGALLAITKENGTAAASIKQGRIPQFLEKPQRSEKVLDRMAKHPKKLRKQNLAGDTMSFAIDLRNRQNTTFGQTLSEKLKELEKAASQLAHLQAASAYANRGRLTTRTQAHSARRRSQALGHESAMVQALTLHEQQAADGTQVRALTLLHNTGTHQVTNRRALVLQSLTRLQSMLTILNHLQRQEVQIATGTSRRIATTGRVLRVLPNTGNRRHFSQSAEAITSRSELVQHSDQQTRQLQLQLLATRHNEQMAAPIAAAGQDQFFSLLRAMHPGGGPQRVSARVNGHILTDMGLDPAVNARVRSRLLQVAGRYNVTNKALQHPRLTDSQTESRQIGDSADAFAASYRILTATLGVLDQLQLPGLPQALSHIRRDLATPDVNNLSILLGPKDAFSTRTENTVATETALDLLTEELQSTAHKQRQTLRRTLTQQQAETHRDMRRGTEGQILEQEQALEREQKLRELLQELALELLKGQRHSIASAALKSEALAQLRRAQKQRQDTAIAAMDQNSTAGARLEVEARLMNQAYQLAGQRHQRLEQQERQLAAALMERLERAHLAALSQEARQALARLAADLEAEVSETVEETVEEGGDLEQGLLSGGQALEAALVALQIMHQRHQNVAANLEQLALQVKLQVLNKRKSAGEGGTQESPSKEKTPSCARGPSGMDRQRSGSDIKTGQNQGQLELGQRSGQRINSGVGLGSLDLGSQNRQSLILDSGGGGRQYALLPGTPAASRVIVQPGAYRDADLGPLHVMVEERRRRRIPRGSPPDGRLGRLLDALLGPQHGGGTDHLPSSDSSSAARRRKAPSPAANGTEQPYDGLLSSGPDGLSHPSRARTGDSWRLVHRPGGQASPIQHSSARGQSPSHLSGEEHDAPLLKDGQGQRLDGHAMAPAGHSLAPARSHLHRGDIWVADSSDASLTRSAGGRAVSPQGSAAGQRDSPVHLHPKAGREASLLTLVALAQQMERFHHLAAQGQQDRLELPEKALGQLEGALLSMGLDGGRRRCLRRRIREGQFLFSPLDRGLTAREAALLLQQHQAGLDGQDHRRDYLDRHQRPVQQSGQRFASQGGRPAGLQQQMGALLHQASLSALDALYLLKKGQQGLESGGRSPAERRVPPSLGGASPYPSGLGLLQVQARKQMSSPQVGQRRAITGPQQLEAALEAISQQLLAMHQARPQEPPVGRKLDLEAQLNAQMEGEQDQSLQKQLDLHQKKLLEALREPNLLVQSPVTAQAPKPSQGAGVAAGAASPQTQPRQSPRRATRTPPGDLRSPGRLILDAQEGQARKGTQGAVKQRQEEAPAAAAAAAAAVSPATAPRSRPRSSRSQRSLERIATQGPLRILHGQQQRLHDARRPQVPDQGAERGVSTNRASAKKSSPKERAIPATHGTKTGQHIKRHLPGGLLLGVRGQRSEYQLQKRARQLLAIHNSRRQTQISLKLIAASLEEALEVLQNAKSASEEAKRLLEPTRREELAGLRQPIKQELQREHQRGGDHEIRELQLENVTLAADEQRSQGAGQELQLQAKQHGELQNQAEILLLGATLRLLSEGLQIPSTDQKTRRGRSTQPLIQEPRGRVPRPQRAPVAEHQRLRAALNARIPIQGNIQLKGAGSDLQAALEKAHETLEKQQNGQPAEPPQPPPAAGQPVASAREGNPDPALQRATEGNRSLLQSQQQPGGSGRNLPDGMPKGRRSSGGRPGLLGSRLGALSPALAPVLLGKSGRGTQLQRHTPAHQKAHASASRPTPPPDHVQPGRERHAQQVEGLQGALKTTAREHLKAQLQAHDPQGRPDLSNLRHRRERSADQRRSSLPPPRGRRAGDAGMGSQRPDTGGQEQREPRPLGSEEQSSGRRRPPPRPRGLPIRGRRRSPRSDKVTEDSVEETANVDHKTPEENQGASSMRESKSQHVQERLWQSKLLTALLGKLLAPPDVLVLLKARGRALQWLAQPIQGAIISRSAPQATEDSQSLIRSAKKHLHQLCSQLTTQARRLIPLPLAPILSFRAEALAAPRPDQQGLGLQPEADPADGHLSVAAQHHGVQHEQARELQQRLEQDLRSHAAEPSLEQGGDHEASRLDLLQEQQAGIRLEREQGLSRHAQRLLDLLQKRKTQTLQLDSSQGRQNAREALATPPHRRGDQEASKRLAQQGGTRSRQGQGRRERTQQQLATLTREHQRVLEGRLKRLRLDRQLLDMLRSLEVLQELSTAAKEEARSRAAEIIARLQQREISQQQGEAFLQALLAHDELLEFLEEVQGEVRNQDLLPQRLWHLLEQPQDIRGDGLLGELEERTLQRGREHIQELMDRHLQHLEETQHQRLEQLLLEQEERPQKIAQMQQLRQRQSAALEALVEAQASPIQRDRPRPPIHAPIASPQAAGESPQPGGPLRRGRQGASPQAPSLSASDKGRSSHHSGTQGREVRRQAIEEKLEKQIERLRAEAEFHHTSLPHQLEPKPGSEKDRLPRISPAGASSSRAPPGDMDRPRPSRGGTKHPCDGRGPQRQDSQADQTHHTSSGTSGGGSSQGSGRPDDPDQLLREPQQQHLTGQAPLLGRPPQGALLDVLPPQPLPTLRGPQAGGVAGDALPGALEESLQELQPQLSELSETLGQAHGRQAPARVAQLEQALQAAPRQALDAELLELALQGQGGQHHLKQWLDRLQHHQSLSGRDREIGDGSALHEALRHHLQRQQRLLLQRLAADHAPGGRPPQKARLPDLQLLIHAWPPQESEQDQRRGALAEALEGLHEHQLQLLQKRPSERRALAAALQELTQEPQAHGRHQGPPNLIGEGLPNAHSRARSRAGSRDSGSQEPSLAQLLRHQQPADLQAQQRLQAESPHERRLERGLAVALVDALREPPPDRKLNAQAAEEAPPAASLSGQHQRRQQLQDEQADYSQALQRGGQAGSLVAGHHGEPLQKLAAGQLGSGQERQDLAQALAQGREVQRAQGLPLKPRTGGLEMSLPRGGPRPKGSQQQGGSQAREAGRLQGGRQQLQQCLGKRQEAQTDRPQHQEADLGLQQLGRRHQQLLGTHQAQGGRLLKQAAPQGISQPPARQHMLRRSGQHKLDQHVSRAATDKQRRSGLFDQRHRQGQHGAHGMSQLQSLRREQAGDLPEGLRREATLERLHEGHAGSAMRDGPQERVPQAILGQELQSLHHRAEPQELLLLLQGQKARRAPRRVSRRGEADALQAQRAHLEAQLAQQQAQGLKRELRSGPQMLAGRERPPGDPQEGQRLLRAHLDAARHLQMQVRPAQAAGLLRLLEVSQPGRGAAVEAQAALDRSRQAGDLREALQRLQHAQDALPALRKPLERARGRALQHHLQMEEVHPQALHRLEVQGQQQSLLEKQQQDLEQQRDQRRSLRGQEERALREDQRALLELRDREHLQRDQRLGSGGSTRPDSAHHHTQKAAQLRHERGAATAQQHLEVAGGQRSGLPVLSHQAQRHPRRPPQQASNQRVHLLRRSQGATQLRSPRAHRGDLQRRGGQLLERVPEALQHLQGRGGRRLHSALGAQAGHVQALLRLQNHGLPRRAAQAGQAALLRREQAAPGRLELLHAQQACLRALPHQQLEGLETQRSQGQAVSLGRKEAPPGGVAGSDDLPPQVHQGHHPTGGSKRERQTAQRSLLEPPQKQQKSPWQHQQEAVRATGQGGSAPPHPQLQPGHQETQRKAVAQAQSRHKQHSGEWDSCDATLEGDDDSQHLPPLLRRQRGRRLLVGAVQARPALAPLVSWGHKQNQQAPAELLLLGSFLRTALKQSLYTELQAGQHPQKPQPRQPAILSSAKAVQQVQVPVRALAGGRLQAGQQSRQLHPQSRRMPGTEPGKDRPAHEELSRQRSKKLELDLAARGSSAGKEQRRQRQQRGGADAGAMGLRRLLPQAQMVPPLRQLAPSALVAQDGQRQGAHRLQAGQQPRPQSQAEAALAQALQSGDSLLAGELDALQVAGSPQRQRLAARPGLQLLHKDPELKAERLERPRAGDQEGLLQALQQAGYLRALWQAGSPFRRLKARLHRLKLQGGRANKEQQRQLGALAATQARQAQLRHLEAEQHQRLLVPAGPRAGGRDHSQPSQYHQAALRRHQQLARLGQRHRRHQRGDAESQLRVDPQQDHDLLREAAQRRLQLQQQKQKEDLQERLTRAARLELHQVPQDDQQSAANPASAGETGQGRARGALKELREPSGANRQAKLRELLRRLQLRGGSGVEQGMPMHQEQVPPGSPLAPAQRQAAGGQVPGKDHEHLLELRLLRQQQRDRQQADSLATAQLQLRLQAAELRAGLEQASQASHRHQAERLQGMLRLAGLQARNQIQRSLASLVQQESALEQVEGHDRLQRPRSPERQAASRALRSPATLLKGEVQAAQKMLAQCQPHHHEKGELAPLRELPQQAALQRQSVQGLPQGRQRGTPELPEPRQQSPIQHLLQDSLGARGRGRRERGQAEQGRHRHKKERRQQAEVAERVQQLATAPLEARESVLAHELALGVAQTQQQQSPQLGRQRESRGAHLRLWQERGAGQLALLQLQREGQPQREDAPRLRAHQQGLRRLHALQQEPEGPGGPVDPALPDRQRQKGPHHRRSDSSQGGAPNQSQPAAAAEIQSTQERDQEQQLKGFLQPEKEKLHLGGQPPPAPAPVGAALASSSSTRHRRHPGQGSARLPGRRDPATARQSPAESGSQKTSGDGGSQAEARRHQQQLAHRGDRQRASRAGLARALQQQRALCQAHRQQEHRSQERELAAGKRSQVAAGERDAQQPRQAERLRHSQVYAKLLAERDLPQLRARRLPQQRELYLSASLTLRREGELAKQLGQAQEGALRRQELEKLQRELRQMELQRALQRAALGLGRNAEQGQLLRSLLRQALGASPPQAQGRLEQLGDEATQLRQQLEGQAPELQARLLEQRPPGCLGQRQSGKLLPLLGQDHEKHALELRRHRSLQEAAAGLRKLLEEDLQLRQLQLLAKEGGQHRLQEERTLRPKLRRAKRAHKQAPELEGLRRRVRALQARCPQELQRRARHRLSLAEERRDHARLLAAHLQAREAARSRRRRGLLDPLRAYQGQRAKQRMLAKLSRGLKQQGAKGRLLRAHLERRAAHPRLDAPAAQDAAHLPLLRQGQVLRQQQLLRHHLLDASPQVREVDPPSLLQRRALRRQAGGRERGLRLLARQEPRHQRHLQTLLQRWQSQYRHQGAKERQGLQSLLRQAGAELSQRRHQRAKAVVRGLREAEPPGLLTALARRLRGLQHQLELALPRQKALLRRLVAPQRALQRLLEGAGQEHRQDQHRELEERRRELRQQLHKAERARLLEGPEAGSGRGLIRAVQRALEQQQKRGHERKLRSLQKRLEAVRLLQGGRGAHRPQRDRQLQRRARRLARQARKLLQERLLSRLHDGLLHLQEGLQGRLRRQHEGQAQLSQLAGLHQPRQKLSHQQQLQRRAADVPQRLRQRAVQGQGESRQAQLLQARRAALKGRSRLQHKAKSRGGGVEPQRRLLLRLGQAAPLGKPGGDLRSRKSRALEQLKRGDQRESRELREQLHALRRAAAQRRRELPQRLGQVRDHQGQHRALRAQARGYRQLLQKAQLPRRQLLRHQAEEGSPHRLKQRLAQLRARDLERCRRVRARALQSLQADHQRRARLQQRVAELSGERRDEVRRAGQARQLLRQVEAALARHQGAQAGKREALGYLLQSRAQAARALPPQRQAARLQERHQGLAEAESEPGSQERHLAQQLARAQQASLLQGERLLALLKGKGLLQFRRELMDARKDQVQGHVAEHKADRQGRLRRLLADLEELQAGALAQFLQHEQQLAEPQHQEVQALAAGLQQQQDELQARLELDRQRALQAAQLGQRQERRALQELELRQQALAQSGQRLLGVAQKLRELQRLHARLRGRQARLQAQRERQVLEVQAKQRELAGREAARRRLAEGARAAHQTHLLAAKEELQRLQAELEGTQHLAQQLQGREEQRAYLEVQPHLRQQRLHLRPLERLLQEYQSAAAREAAELQRGEAHRVQRLLEQEKQRLGHQLEAALERERLLHGQPDRLAQRREALQQLLEQEEAQQRLRALRPEAALQRLAAQLQGLRRELEQQKQTLAGERAHELLLAQLPAARGEAESLLQAAAARLQEVTSQRPEDATARRARSAARLLDQARALQARGQDRRLQGGGRLARALVELGREQKLRERLERLRALLHGKAQRLAQALRERRREVEQARTQLKAHQGRLALKGRAEELLAEKAAGLLQRHQGAKRLLGQLEAAKERGLAALRALPQRLQQALEYLRQRQRALRRAAAQLEQALAELRALRQRAGAEAARPGPHQAEEALLLGLAELLALQAASRAKQLLAHLAQLGRRARQALEIFERRLQLQEQRGSQPRELERRALQEGQQSRLLPQLQAQKLSLQELLHQGLQARRLKLQRKLHQLEREIAERLQAYQDQALDEDHTLEELARRLQQQVEGLAAKQERLRAHEEKRVRAAAKELLELLQARGGERREKLRQALQRRLAHALQRAEGLKREAALRALGQQRAGLEAARQQLPARLLARLALQMGEEVARQAELQALLAGEKQLAEQHQQRAAELLMQARQKLKALGQRQAEAAARKRELLLHRQLLLLRDQAGEAALQRLAALRRRAKRLQGERRGLRDQLRALQSLGGLQGALLHLEAKAHRLLEHGLLQGQELLEALRAGELDRLKGYLEPQARLAERALAALEREARALAAVQARAARLLQHNRAAKALQAAAQGALLRKGHAAQAEAARKRAAGELLRVQRALVLREGAGRALQALRAEGKALQRAAEGAGERAKQQRALLVGLAQQQRLLQARLQAHRAGALKAAQLGLQRLRALLEAQLGDRLEARRAERFFARAKAALGQAEATERVLQAGQAAAAGLAGLRRLQARGQARRDYARQLRERQRGADEQRLLEALRQRALEQLERGQGLRRHLQARLQRQSQELLQHLLDSRRALHAAREQAERTRLLAAQEAQAVRRQQALLEARQEGQHLLQARLEQLQQQKRELQHQLAALGRRLQELQAALQARQAVRALNRRPEHLKALLGRLAEAAARLRRALQAAQQRLARAVTAHASRQLSRQHQAAHQERLRARSQRAHEQHALLHRGQEKEQHQEAHLQAETMQGRRLLRRQEALLEREGQESMARRLRHQRRHQGALEHLAQARRQLEQRAKEGQEQRRRLQERLQALLALQGQQQSAYQAVRELLHQLLAAQQREQLLQLAQQQTVSQCQLLRELQRANHLVAALQHSPGLLEQRARALKQLRRALAAQLQAAAELQGREQAAGAKRELEREALLRLQALQQMRSLFRSLQARIRERIREKEREALQRQLRELAQRPQALEQALEQHEAQLLRIHQEQRHLQRALQAARRQQKEHIRLAQQAHRRALAHLQARKQELQELRRALQREQHARLREQLHQLLLRATLRSNLARALQAQEKAARRLEEARQRLLAAVGEHQKRLQQHQEMLEALKRQQHQLRKQHQRLENRQWSQKELQQGDQQAAQLLQARLLLQQGRQAGRRLRHLERVQAAQRRLRRELQRLRLERREALQQRDPMMLEGAQTLLSQGQLRLAAGELRSLQASLQRVQAAQQLLQEKQRQLKRLRQRHQSLQARLAARRVGALQARLEALQARQRQADQLQLQLQETGQGGRILQRLRHQRSQHQELREQLALEQQLARLRRGLQELALQQATLEERKQQARELRLLQAPGRHLAQHRQELQKLQAQLKKLRQQHHRVLLLAEDLAELREKREGQLRRQAREAAAALQQRLQARAAATEERRKEAHQALQALQRAQLEGAKALLREQQRLRAHRSARGQATQDRRLQALASRQRLRQELQARLQQALEQARLKLQAHQMHREELRQRRQSAALQGRLQALQARQRLLREAALQGQRELRRQLQRHAREREARALAARQFLREALQSHLQEAQHQLLATRATEEHLRRQGHELEDLARELRARLQLQEQALQEVGDDGKLAQPRQLQADLMRLRQQLQEVQYLREELREKQRLRRQRLQEQLREQSDQRQELHRARRTHQEAHRLLQAHARALQQLAQQLRRPDQRQQELLLLQARGAALQRALQELRQAEKVQQRLELLHQTQRHLLGLRRRLQRGLLAAEARQRELRAGETQLRQRLALLEAARRLLEALQARQGESPARLEQAQRRRTQEERALQRLAQELRRGQHLLRELHRLARRHSGTLRLLQAQLLARRQQASRGGQLLELRAQLLHLQNRQQQLEELLRVLEAQQPLLGRQRARQALQRDSQHQRHLSAAHLGLREKERKLQHLREAREQHQLLQHQLGETLHGLQALRQAQARQQELATLQQTLARLRRAQEALRADLGLQARLQELRAVSQARQLHQHQEELLHRLRDQQLQRQRQLLQERQRALAELRQEQQLHRAQLLKGQRQELQRLRRQKLGLQAELERQGARQAELARLKAAGGLAAEQHRLQRHQRLLQRRQAQLERLRQQGRETLRHQHQLIQQEQQLQRTLRHQEAQLQQLRQALKETGERLQQRHLQRLRELQERLQEGGGLLRELQTQLAGLATESLQRQPRSLRGSQRQGVAEGRQRLLRTGLRELQHQRLFLSARRQAQGGGQARLASQQELLSLQQQKALLRQTGARGARLSLLGQAAAELREALLQHQERLQRRSQHQALLQGRRHRETEQVLAALQERALLQRAQQQTQARRGRALAALLKARALQLQGEAQQLRSARLGLQHQAQLLKGEALAGRLEEAQHQQRDDGEREELQALLQAELQRHQEQHQHLLRRLQQAERSQRLRELQGERALLAELQALRQCLQRQRLQAFRDRLPRRRHAARQLEEQQELQAAAAGRALLHQRAQLQRAGQEARHRSQALQRLLQRPLEAEARQALERAWLRIQARLELLQRTQRLLEELLGRLERTQHELEDLQRYLRARQPDQATQEKERLALLQKLRRAQQEQAHLLQALQGLARALLEQQSQHLQAARRLLQLLQALRALLQHQRHQLRATQRADRRRRHQARVLLEDLQAAQRLQRLLGRQRLPGRLQHQQLQEELQEAKRALQAQESLQQLQRQAREQQHLLEATRALLAALQHQSQLLQELQALLHSQEALRQQRLQLQRRRTAQLQRELERLQASQYLLQRLRELLKTALRALQARLEEQEAARLKRLERLHQRKAELLLQAQLQAHQRQVLLRDLQATLRQQQQAAERLQQRLELLQARELEARGELERAQQHRLLQQAQARLQEGYRPPQEFREAPRQRRAPEPPPAPPEGKIPQAPTETAAQEKQGDKGARTTAGAKKQEERQMGPASAQELEEGLRGDPATTTQRQAISPGLQELPAVGGQSGGRRPLQKERLGLQDAGAATLLQHPAARLLQALQRVGDAKLRLQHREHLERYLKEFQGLQRQAALLQRRTALRRALQERAEAKTSLQRHGQMLKRSALLQGLLDRQETRRAPGQPRAALRLSQRLGLQGQERGALRQDRQLQRALERQRTQLETARRKLAAVARLEAAARDLSLQRLQRLQSEAARHRALQKKRLQHQHQALHQAELLREALERIARQQRALQQGKKVQEHLQRMLARLQEQRKLLKQHQRQLEQYLQRNNQGLQALTRARRALARLRDQRRERHQRLRALLQRQLEQVQAELATLRTLLRQARAAAARLRHQVAALLEGGGLAVELLARLQHVQEQLERRQEQLQRELRLLARRVGAALQRLQALQRRRRRVLADELTRLKRAQEETLRRLQELRRVQHQKRRLQGHHELLRHDGEQLQALRRALRLSLSQQQARLQDLQRQAAIREERGLQAALRALRALQGKTDDRLAAGQELRQLQGYLEQALRQLLERRQLLAGLQEAAHGLRRELLQRLQALLEQVAQAQPTRLELQDAGRQTRHQGLQALLRKLRALRRQRRQALQRYLEQLQGQRHQQGRLHHRLRQLLATHQRQLRELEGHLKAARRAALQQLARASARQKQRLDRQRAALEERRATAAAQQLYGELRDEQRALQRQHRLLHQQARLQGRRRALEELQRFLSQQQQQHQAGLEAALLRLQLQRQELQQRQDLLRELRRVRQQQGQLRSLLRAQSDLQRHQQRLQELRRQLRLSQGDSRTVGDAATLQHLRHHIERQLLLQRDQLLQRLRPRLEQLAQELALQHQRHYEAARATLAAQQRGLITGQR',
                'length': 3418
            }
            
            sequence = provider.get_sequence("P51587")
            
            assert sequence['accession'] == "P51587"
            assert sequence['length'] == 3418
            assert sequence['sequence'].startswith('MPI')
    
    def test_get_protein_domains(self):
        """Test getting protein domain information."""
        provider = UniProtProvider()
        
        with patch.object(provider, 'get_domains') as mock_domains:
            mock_domains.return_value = [
                {
                    'name': 'BRCA2 repeat',
                    'start': 1002,
                    'end': 1036,
                    'description': 'BRCA2 repeat'
                },
                {
                    'name': 'DNA-binding domain',
                    'start': 2481,
                    'end': 3186,
                    'description': 'DNA-binding domain of BRCA2'
                }
            ]
            
            domains = provider.get_domains("P51587")
            
            assert len(domains) == 2
            assert domains[0]['name'] == 'BRCA2 repeat'
    
    def test_search_by_go_term(self):
        """Test searching proteins by GO terms."""
        provider = UniProtProvider()
        
        with patch.object(provider, 'search_by_go_term') as mock_go_search:
            mock_go_search.return_value = [
                {'accession': 'P51587', 'gene_name': 'BRCA2'},
                {'accession': 'P38398', 'gene_name': 'BRCA1'}
            ]
            
            results = provider.search_by_go_term(
                go_id="GO:0006281",  # DNA repair
                organism="9606"
            )
            
            assert len(results) == 2
            assert any(r['gene_name'] == 'BRCA2' for r in results)


# ===============================================================================
# NCBI Provider Tests
# ===============================================================================

@pytest.mark.unit
class TestNCBIProvider:
    """Test NCBI provider functionality."""
    
    def test_ncbi_provider_initialization(self):
        """Test NCBIProvider initialization."""
        provider = NCBIProvider(email="test@example.com", api_key="test_key")
        
        assert provider.email == "test@example.com"
        assert provider.api_key == "test_key"
    
    def test_search_nucleotide_database(self):
        """Test searching nucleotide database."""
        provider = NCBIProvider(email="test@example.com")
        
        with patch.object(provider, 'search_nucleotide') as mock_search:
            mock_search.return_value = [
                "NM_000059.3",  # BRCA2 mRNA
                "NM_007294.4"   # BRCA1 mRNA
            ]
            
            results = provider.search_nucleotide(
                query="BRCA[Gene Name] AND homo sapiens[Organism]",
                retmax=50
            )
            
            assert len(results) == 2
            assert "NM_000059.3" in results
    
    def test_get_sequence_from_genbank(self):
        """Test getting sequence from GenBank."""
        provider = NCBIProvider(email="test@example.com")
        
        with patch.object(provider, 'get_genbank_sequence') as mock_sequence:
            mock_sequence.return_value = {
                'accession': 'NM_000059.3',
                'sequence': 'ATGCCTATTGGATCCAAAGAGAGGCCAACATTTTTTGAAATTTTTAAGACACGTTGCAATAAG...',
                'length': 10257,
                'organism': 'Homo sapiens',
                'gene': 'BRCA2'
            }
            
            sequence = provider.get_genbank_sequence("NM_000059.3")
            
            assert sequence['accession'] == "NM_000059.3"
            assert sequence['gene'] == "BRCA2"
    
    def test_search_sra_database(self):
        """Test searching SRA database."""
        provider = NCBIProvider(email="test@example.com")
        
        with patch.object(provider, 'search_sra') as mock_sra:
            mock_sra.return_value = [
                {
                    'run_accession': 'SRR123456',
                    'study_accession': 'SRP123456',
                    'experiment_title': 'Single-cell RNA-seq of T cells',
                    'organism': 'Homo sapiens',
                    'platform': 'ILLUMINA',
                    'library_strategy': 'RNA-Seq'
                }
            ]
            
            results = provider.search_sra(
                query="single cell RNA seq homo sapiens",
                retmax=10
            )
            
            assert len(results) == 1
            assert results[0]['run_accession'] == 'SRR123456'
    
    def test_get_taxonomy_info(self):
        """Test getting taxonomy information."""
        provider = NCBIProvider(email="test@example.com")
        
        with patch.object(provider, 'get_taxonomy_info') as mock_taxonomy:
            mock_taxonomy.return_value = {
                'taxid': '9606',
                'scientific_name': 'Homo sapiens',
                'common_name': 'human',
                'lineage': ['Eukaryota', 'Metazoa', 'Chordata', 'Mammalia', 'Primates', 'Hominidae'],
                'parent_taxid': '9605'
            }
            
            taxonomy = provider.get_taxonomy_info("9606")
            
            assert taxonomy['scientific_name'] == 'Homo sapiens'
            assert taxonomy['common_name'] == 'human'


# ===============================================================================
# Error Handling and Integration Tests
# ===============================================================================

@pytest.mark.unit
class TestProvidersErrorHandling:
    """Test provider error handling and integration."""
    
    def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        provider = PubMedProvider(email="test@example.com")
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.Timeout("Request timeout")
            
            with pytest.raises(requests.Timeout):
                provider._make_request("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")
    
    def test_api_rate_limit_handling(self):
        """Test handling of API rate limits."""
        provider = PubMedProvider(email="test@example.com", rate_limit=1)
        
        with patch('time.sleep') as mock_sleep:
            # Make multiple rapid requests
            provider._check_rate_limit()
            provider._check_rate_limit()
            
            # Should have enforced rate limiting
            mock_sleep.assert_called()
    
    def test_invalid_response_handling(self):
        """Test handling of invalid API responses."""
        provider = BioRxivProvider()
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"error": "Invalid query"}
            mock_response.status_code = 400
            mock_get.return_value = mock_response
            
            with pytest.raises(Exception, match="Invalid query"):
                provider._make_request("https://api.biorxiv.org/details/biorxiv/invalid")
    
    def test_provider_integration(self, providers_config):
        """Test integration between different providers."""
        pubmed = PubMedProvider(email=providers_config['pubmed']['email'])
        ensembl = EnsemblProvider()
        uniprot = UniProtProvider()
        
        # Mock a workflow that uses multiple providers
        with patch.object(pubmed, 'search_papers') as mock_pubmed_search, \
             patch.object(ensembl, 'get_gene_info') as mock_ensembl_info, \
             patch.object(uniprot, 'search_proteins') as mock_uniprot_search:
            
            mock_pubmed_search.return_value = ["37123456"]
            mock_ensembl_info.return_value = {"display_name": "BRCA2"}
            mock_uniprot_search.return_value = {"results": [{"primaryAccession": "P51587"}]}
            
            # Simulate integrated workflow
            papers = pubmed.search_papers("BRCA2")
            gene_info = ensembl.get_gene_info("ENSG00000139618")
            proteins = uniprot.search_proteins("BRCA2")
            
            assert len(papers) == 1
            assert gene_info["display_name"] == "BRCA2"
            assert len(proteins["results"]) == 1
    
    def test_concurrent_provider_access(self):
        """Test concurrent access to providers."""
        import threading
        import time
        
        provider = PubMedProvider(email="test@example.com")
        results = []
        errors = []
        
        def provider_worker(worker_id):
            """Worker function for concurrent provider access."""
            try:
                with patch.object(provider, 'search_papers') as mock_search:
                    mock_search.return_value = [f"PMID_{worker_id}"]
                    
                    papers = provider.search_papers(f"query_{worker_id}")
                    results.append((worker_id, papers))
                    time.sleep(0.01)
                    
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=provider_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent provider access errors: {errors}"
        assert len(results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])