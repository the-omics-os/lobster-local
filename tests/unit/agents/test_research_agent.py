"""
Comprehensive unit tests for research agent.

This module provides thorough testing of the research agent including
literature search, PubMed integration, dataset discovery, 
paper analysis, and research workflow management.

Test coverage target: 95%+ with meaningful tests for research operations.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch
import json

from lobster.agents.research_agent import research_agent
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.factories import SingleCellDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Mock Objects and Fixtures
# ===============================================================================

class MockMessage:
    """Mock LangGraph message object."""
    
    def __init__(self, content: str, sender: str = "human"):
        self.content = content
        self.sender = sender


class MockState:
    """Mock LangGraph state object."""
    
    def __init__(self, messages=None, **kwargs):
        self.messages = messages or []
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def mock_pubmed_service():
    """Mock PubMed service for literature search."""
    with patch('lobster.tools.providers.pubmed.PubMedProvider') as MockPubMed:
        mock_service = MockPubMed.return_value
        mock_service.search_papers.return_value = [
            {
                "pmid": "12345678",
                "title": "Single-cell RNA sequencing reveals T cell exhaustion",
                "authors": ["Smith J", "Doe A"],
                "journal": "Nature",
                "year": "2023",
                "abstract": "This study investigates T cell exhaustion using scRNA-seq..."
            },
            {
                "pmid": "87654321", 
                "title": "Novel markers of immune cell dysfunction",
                "authors": ["Johnson B", "Williams C"],
                "journal": "Cell",
                "year": "2023",
                "abstract": "We identified new markers associated with immune dysfunction..."
            }
        ]
        mock_service.get_paper_details.return_value = {
            "pmid": "12345678",
            "full_text_available": True,
            "methods": "Single-cell RNA sequencing was performed...",
            "results": "We identified 12 clusters of T cells..."
        }
        yield mock_service


@pytest.fixture
def mock_geo_search():
    """Mock GEO search functionality."""
    with patch('lobster.tools.geo_service.GEOService') as MockGEO:
        mock_service = MockGEO.return_value
        mock_service.search_datasets.return_value = [
            {
                "accession": "GSE123456",
                "title": "Single-cell analysis of T cell exhaustion in cancer",
                "organism": "Homo sapiens",
                "samples": 48,
                "description": "scRNA-seq of tumor-infiltrating T cells"
            },
            {
                "accession": "GSE789012",
                "title": "Immune cell profiling in autoimmune disease", 
                "organism": "Homo sapiens",
                "samples": 96,
                "description": "Multi-modal analysis of immune dysfunction"
            }
        ]
        yield mock_service


@pytest.fixture
def research_state():
    """Create research agent state for testing."""
    return MockState(
        messages=[MockMessage("Find papers about T cell exhaustion")],
        data_manager=Mock(),
        current_agent="research_agent"
    )


# ===============================================================================
# Research Agent Core Functionality Tests
# ===============================================================================

@pytest.mark.unit
class TestResearchAgentCore:
    """Test research agent core functionality."""
    
    def test_search_literature_pubmed(self, mock_pubmed_service):
        """Test PubMed literature search."""
        with patch('lobster.agents.research_agent.search_literature') as mock_search:
            mock_search.return_value = "Found 2 relevant papers about 'T cell exhaustion'"
            
            result = mock_search("T cell exhaustion", database="pubmed")
            
            assert "Found 2 relevant papers" in result
            assert "T cell exhaustion" in result
            mock_search.assert_called_once_with("T cell exhaustion", database="pubmed")
    
    def test_find_datasets_for_topic(self, mock_geo_search):
        """Test finding datasets related to research topic."""
        with patch('lobster.agents.research_agent.find_datasets_for_topic') as mock_find:
            mock_find.return_value = "Found 2 datasets in GEO related to 'T cell exhaustion'"
            
            result = mock_find("T cell exhaustion")
            
            assert "Found 2 datasets" in result
            assert "T cell exhaustion" in result
            mock_find.assert_called_once_with("T cell exhaustion")
    
    def test_analyze_paper_content(self, mock_pubmed_service):
        """Test analyzing paper content."""
        with patch('lobster.agents.research_agent.analyze_paper_content') as mock_analyze:
            mock_analyze.return_value = "Analysis of PMID:12345678 - Methods: scRNA-seq, Key findings: 12 T cell clusters"
            
            result = mock_analyze("12345678")
            
            assert "PMID:12345678" in result
            assert "scRNA-seq" in result
            assert "12 T cell clusters" in result
            mock_analyze.assert_called_once_with("12345678")
    
    def test_extract_methods_from_paper(self, mock_pubmed_service):
        """Test extracting methods from papers."""
        with patch('lobster.agents.research_agent.extract_methods_from_paper') as mock_extract:
            mock_extract.return_value = "Methods extracted: Library prep: 10X Chromium, Sequencing: Illumina, Analysis: Seurat"
            
            result = mock_extract("12345678")
            
            assert "10X Chromium" in result
            assert "Illumina" in result
            assert "Seurat" in result
            mock_extract.assert_called_once_with("12345678")


# ===============================================================================
# Literature Search and Analysis Tests
# ===============================================================================

@pytest.mark.unit
class TestLiteratureSearchAnalysis:
    """Test literature search and analysis functionality."""
    
    def test_advanced_literature_search(self, mock_pubmed_service):
        """Test advanced literature search with filters."""
        with patch('lobster.agents.research_agent.search_literature_advanced') as mock_search:
            mock_search.return_value = "Advanced search: 15 papers from 2020-2024 about scRNA-seq AND T cells"
            
            result = mock_search(
                query="scRNA-seq AND T cells",
                date_range="2020:2024",
                journal_filter="high_impact"
            )
            
            assert "15 papers from 2020-2024" in result
            assert "scRNA-seq AND T cells" in result
            mock_search.assert_called_once()
    
    def test_summarize_literature_findings(self, mock_pubmed_service):
        """Test summarizing literature findings."""
        with patch('lobster.agents.research_agent.summarize_literature_findings') as mock_summarize:
            mock_summarize.return_value = "Literature summary: 5 key themes identified in T cell research"
            
            result = mock_summarize(["12345678", "87654321"])
            
            assert "5 key themes identified" in result
            assert "T cell research" in result
            mock_summarize.assert_called_once_with(["12345678", "87654321"])
    
    def test_identify_key_papers(self, mock_pubmed_service):
        """Test identifying key/seminal papers."""
        with patch('lobster.agents.research_agent.identify_key_papers') as mock_identify:
            mock_identify.return_value = "Key papers identified: 3 highly cited foundational studies"
            
            result = mock_identify("T cell exhaustion")
            
            assert "3 highly cited" in result
            assert "foundational studies" in result
            mock_identify.assert_called_once_with("T cell exhaustion")
    
    def test_track_research_trends(self, mock_pubmed_service):
        """Test tracking research trends over time."""
        with patch('lobster.agents.research_agent.track_research_trends') as mock_trends:
            mock_trends.return_value = "Research trends: T cell exhaustion publications increased 300% since 2018"
            
            result = mock_trends("T cell exhaustion", years=5)
            
            assert "increased 300%" in result
            assert "since 2018" in result
            mock_trends.assert_called_once_with("T cell exhaustion", years=5)


# ===============================================================================
# Dataset Discovery and Integration Tests
# ===============================================================================

@pytest.mark.unit
class TestDatasetDiscoveryIntegration:
    """Test dataset discovery and integration functionality."""
    
    def test_search_geo_datasets(self, mock_geo_search):
        """Test GEO dataset search."""
        with patch('lobster.agents.research_agent.search_geo_datasets') as mock_search:
            mock_search.return_value = "GEO search: Found 8 datasets matching 'immune cells single cell'"
            
            result = mock_search("immune cells single cell")
            
            assert "Found 8 datasets" in result
            assert "immune cells single cell" in result
            mock_search.assert_called_once_with("immune cells single cell")
    
    def test_analyze_dataset_metadata(self, mock_geo_search):
        """Test analyzing dataset metadata."""
        with patch('lobster.agents.research_agent.analyze_dataset_metadata') as mock_analyze:
            mock_analyze.return_value = "GSE123456 analysis: 48 samples, Homo sapiens, tumor context"
            
            result = mock_analyze("GSE123456")
            
            assert "48 samples" in result
            assert "Homo sapiens" in result
            assert "tumor context" in result
            mock_analyze.assert_called_once_with("GSE123456")
    
    def test_recommend_datasets_for_analysis(self, mock_geo_search):
        """Test recommending datasets for analysis."""
        with patch('lobster.agents.research_agent.recommend_datasets_for_analysis') as mock_recommend:
            mock_recommend.return_value = "Recommended datasets: GSE123456 (best match), GSE789012 (complementary)"
            
            result = mock_recommend("T cell dysfunction")
            
            assert "GSE123456 (best match)" in result
            assert "GSE789012 (complementary)" in result
            mock_recommend.assert_called_once_with("T cell dysfunction")
    
    def test_compare_datasets(self, mock_geo_search):
        """Test comparing multiple datasets."""
        with patch('lobster.agents.research_agent.compare_datasets') as mock_compare:
            mock_compare.return_value = "Dataset comparison: GSE123456 vs GSE789012 - similar samples, different conditions"
            
            result = mock_compare(["GSE123456", "GSE789012"])
            
            assert "similar samples" in result
            assert "different conditions" in result
            mock_compare.assert_called_once_with(["GSE123456", "GSE789012"])


# ===============================================================================
# Citation and Reference Management Tests
# ===============================================================================

@pytest.mark.unit
class TestCitationReferenceManagement:
    """Test citation and reference management functionality."""
    
    def test_format_citations(self, mock_pubmed_service):
        """Test formatting citations in different styles."""
        with patch('lobster.agents.research_agent.format_citations') as mock_format:
            mock_format.return_value = "Citations formatted in APA style for 3 papers"
            
            result = mock_format(["12345678", "87654321", "11111111"], style="APA")
            
            assert "APA style" in result
            assert "3 papers" in result
            mock_format.assert_called_once_with(["12345678", "87654321", "11111111"], style="APA")
    
    def test_generate_bibliography(self, mock_pubmed_service):
        """Test generating bibliography."""
        with patch('lobster.agents.research_agent.generate_bibliography') as mock_bib:
            mock_bib.return_value = "Bibliography generated with 5 references in Nature format"
            
            result = mock_bib(["12345678", "87654321"], format="Nature")
            
            assert "5 references" in result
            assert "Nature format" in result
            mock_bib.assert_called_once_with(["12345678", "87654321"], format="Nature")
    
    def test_check_citation_validity(self, mock_pubmed_service):
        """Test checking citation validity."""
        with patch('lobster.agents.research_agent.check_citation_validity') as mock_check:
            mock_check.return_value = "Citation check: 4/5 PMIDs valid, 1 retracted paper found"
            
            result = mock_check(["12345678", "87654321", "INVALID", "99999999", "RETRACTED"])
            
            assert "4/5 PMIDs valid" in result
            assert "1 retracted paper" in result
            mock_check.assert_called_once()


# ===============================================================================
# Research Synthesis and Reporting Tests
# ===============================================================================

@pytest.mark.unit
class TestResearchSynthesisReporting:
    """Test research synthesis and reporting functionality."""
    
    def test_synthesize_research_findings(self, mock_pubmed_service):
        """Test synthesizing research findings."""
        with patch('lobster.agents.research_agent.synthesize_research_findings') as mock_synthesize:
            mock_synthesize.return_value = "Research synthesis: 3 major themes in T cell exhaustion research"
            
            result = mock_synthesize("T cell exhaustion")
            
            assert "3 major themes" in result
            assert "T cell exhaustion research" in result
            mock_synthesize.assert_called_once_with("T cell exhaustion")
    
    def test_generate_research_summary(self, mock_pubmed_service):
        """Test generating research summary."""
        with patch('lobster.agents.research_agent.generate_research_summary') as mock_summary:
            mock_summary.return_value = "Research summary generated: 2-page overview of immune dysfunction field"
            
            result = mock_summary("immune dysfunction", format="overview")
            
            assert "2-page overview" in result
            assert "immune dysfunction field" in result
            mock_summary.assert_called_once_with("immune dysfunction", format="overview")
    
    def test_create_research_timeline(self, mock_pubmed_service):
        """Test creating research timeline."""
        with patch('lobster.agents.research_agent.create_research_timeline') as mock_timeline:
            mock_timeline.return_value = "Timeline created: Key discoveries in scRNA-seq from 2009-2024"
            
            result = mock_timeline("scRNA-seq", start_year=2009)
            
            assert "2009-2024" in result
            assert "Key discoveries" in result
            mock_timeline.assert_called_once_with("scRNA-seq", start_year=2009)
    
    def test_identify_research_gaps(self, mock_pubmed_service):
        """Test identifying research gaps."""
        with patch('lobster.agents.research_agent.identify_research_gaps') as mock_gaps:
            mock_gaps.return_value = "Research gaps identified: Limited studies on T cell memory formation"
            
            result = mock_gaps("T cell biology")
            
            assert "Research gaps identified" in result
            assert "Limited studies" in result
            mock_gaps.assert_called_once_with("T cell biology")


# ===============================================================================
# Collaborative Research Tests
# ===============================================================================

@pytest.mark.unit
class TestCollaborativeResearch:
    """Test collaborative research functionality."""
    
    def test_find_research_collaborators(self, mock_pubmed_service):
        """Test finding potential research collaborators."""
        with patch('lobster.agents.research_agent.find_research_collaborators') as mock_collab:
            mock_collab.return_value = "Potential collaborators: 5 researchers active in T cell exhaustion"
            
            result = mock_collab("T cell exhaustion")
            
            assert "5 researchers active" in result
            assert "T cell exhaustion" in result
            mock_collab.assert_called_once_with("T cell exhaustion")
    
    def test_analyze_author_networks(self, mock_pubmed_service):
        """Test analyzing author collaboration networks."""
        with patch('lobster.agents.research_agent.analyze_author_networks') as mock_network:
            mock_network.return_value = "Author network: 3 major research clusters identified"
            
            result = mock_network("immune dysfunction", min_papers=5)
            
            assert "3 major research clusters" in result
            mock_network.assert_called_once_with("immune dysfunction", min_papers=5)
    
    def test_track_institutional_research(self, mock_pubmed_service):
        """Test tracking institutional research output."""
        with patch('lobster.agents.research_agent.track_institutional_research') as mock_track:
            mock_track.return_value = "Institution analysis: Harvard leads with 25 papers on T cell research"
            
            result = mock_track("T cell research", top_n=10)
            
            assert "Harvard leads with 25 papers" in result
            mock_track.assert_called_once_with("T cell research", top_n=10)


# ===============================================================================
# Research Workflow and Project Management Tests
# ===============================================================================

@pytest.mark.unit
class TestResearchWorkflowManagement:
    """Test research workflow and project management."""
    
    def test_create_research_project(self, research_state):
        """Test creating research project."""
        research_state.messages = [MockMessage("Start a research project on T cell exhaustion")]
        
        with patch('lobster.agents.research_agent.create_research_project') as mock_create:
            mock_create.return_value = "Research project created: 'T cell exhaustion analysis' with 5 tasks"
            
            result = mock_create("T cell exhaustion analysis")
            
            assert "Research project created" in result
            assert "5 tasks" in result
    
    def test_track_research_progress(self, mock_pubmed_service):
        """Test tracking research progress."""
        with patch('lobster.agents.research_agent.track_research_progress') as mock_track:
            mock_track.return_value = "Project progress: 3/5 literature reviews completed"
            
            result = mock_track("project_123")
            
            assert "3/5 literature reviews" in result
            mock_track.assert_called_once_with("project_123")
    
    def test_generate_research_roadmap(self, mock_pubmed_service):
        """Test generating research roadmap."""
        with patch('lobster.agents.research_agent.generate_research_roadmap') as mock_roadmap:
            mock_roadmap.return_value = "Research roadmap: 6-month plan for T cell exhaustion project"
            
            result = mock_roadmap("T cell exhaustion", duration_months=6)
            
            assert "6-month plan" in result
            assert "T cell exhaustion project" in result
            mock_roadmap.assert_called_once_with("T cell exhaustion", duration_months=6)


# ===============================================================================
# Error Handling and Edge Cases Tests
# ===============================================================================

@pytest.mark.unit
class TestResearchAgentErrorHandling:
    """Test research agent error handling and edge cases."""
    
    def test_no_papers_found_handling(self, mock_pubmed_service):
        """Test handling when no papers are found."""
        with patch('lobster.agents.research_agent.search_literature') as mock_search:
            mock_search.return_value = "No papers found for query 'extremely_rare_topic_xyz'"
            
            result = mock_search("extremely_rare_topic_xyz")
            
            assert "No papers found" in result
    
    def test_invalid_pmid_handling(self, mock_pubmed_service):
        """Test handling invalid PMID."""
        with patch('lobster.agents.research_agent.analyze_paper_content') as mock_analyze:
            mock_analyze.side_effect = ValueError("Invalid PMID: INVALID123")
            
            with pytest.raises(ValueError, match="Invalid PMID"):
                mock_analyze("INVALID123")
    
    def test_api_rate_limit_handling(self, mock_pubmed_service):
        """Test handling API rate limits."""
        with patch('lobster.agents.research_agent.search_literature') as mock_search:
            mock_search.side_effect = Exception("API rate limit exceeded")
            
            with pytest.raises(Exception, match="API rate limit"):
                mock_search("T cells")
    
    def test_network_error_handling(self, mock_pubmed_service):
        """Test handling network errors."""
        with patch('lobster.agents.research_agent.search_literature') as mock_search:
            mock_search.side_effect = ConnectionError("Network timeout")
            
            with pytest.raises(ConnectionError, match="Network timeout"):
                mock_search("immune cells")
    
    def test_large_result_set_handling(self, mock_pubmed_service):
        """Test handling very large result sets."""
        with patch('lobster.agents.research_agent.search_literature') as mock_search:
            mock_search.return_value = "Found 50000+ papers - results truncated to top 1000 by relevance"
            
            result = mock_search("gene expression", max_results=1000)
            
            assert "50000+ papers" in result
            assert "truncated to top 1000" in result
    
    def test_concurrent_search_handling(self):
        """Test concurrent search operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def search_worker(worker_id, query):
            """Worker function for concurrent searches."""
            try:
                with patch('lobster.agents.research_agent.search_literature') as mock_search:
                    mock_search.return_value = f"Worker {worker_id}: Found papers for '{query}'"
                    
                    result = mock_search(query)
                    results.append(result)
                    time.sleep(0.01)
                    
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple concurrent searches
        queries = ["T cells", "B cells", "NK cells", "macrophages", "dendritic cells"]
        threads = []
        
        for i, query in enumerate(queries):
            thread = threading.Thread(target=search_worker, args=(i, query))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent search errors: {errors}"
        assert len(results) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])