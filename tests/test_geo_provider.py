"""
Unit tests for GEO provider implementation.

Tests the direct GEO DataSets search capabilities including all query patterns
from the official NCBI API examples.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from lobster.tools.providers.geo_provider import GEOProvider, GEOProviderConfig
from lobster.tools.providers.geo_query_builder import (
    GEOQueryBuilder, 
    GEOSearchFilters, 
    GEOEntryType,
    GEOFieldTag
)
from lobster.tools.providers.base_provider import DatasetType, PublicationSource
from lobster.core.data_manager_v2 import DataManagerV2


class TestGEOQueryBuilder:
    """Test the GEO query builder functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = GEOQueryBuilder()
    
    def test_escape_query_term(self):
        """Test query term escaping."""
        assert self.builder.escape_query_term("single-cell") == "single-cell"
        assert self.builder.escape_query_term("single cell RNA-seq") == '"single cell RNA-seq"'
        assert self.builder.escape_query_term('test "quotes"') == '"test \\"quotes\\""'
    
    def test_build_entry_type_filter(self):
        """Test entry type filter construction."""
        # Single entry type
        result = self.builder.build_entry_type_filter([GEOEntryType.SERIES])
        assert result == "gse[ETYP]"
        
        # Multiple entry types
        result = self.builder.build_entry_type_filter([GEOEntryType.SERIES, GEOEntryType.DATASET])
        assert result == "(gse[ETYP] OR gds[ETYP])"
    
    def test_build_organism_filter(self):
        """Test organism filter construction."""
        # Single organism
        result = self.builder.build_organism_filter(["human"])
        assert result == "human[ORGN]"
        
        # Multiple organisms  
        result = self.builder.build_organism_filter(["human", "mouse"])
        assert result == "(human[ORGN] OR mouse[ORGN])"
        
        # Organism name mapping
        result = self.builder.build_organism_filter(["homo sapiens"])
        assert result == "human[ORGN]"
    
    def test_build_platform_filter(self):
        """Test platform filter construction."""
        # Single platform
        result = self.builder.build_platform_filter(["GPL96"])
        assert result == "GPL96[ACCN]"
        
        # Platform number without GPL prefix
        result = self.builder.build_platform_filter(["570"])
        assert result == "GPL570[ACCN]"
    
    def test_build_date_filter(self):
        """Test date range filter construction."""
        # Date range
        date_range = {"start": "2007/01", "end": "2007/03"}
        result = self.builder.build_date_filter(date_range)
        assert result == "2007/01:2007/03[PDAT]"
        
        # Single start date
        date_range = {"start": "2023"}
        result = self.builder.build_date_filter(date_range)
        assert result == "2023[PDAT]"
    
    def test_build_recent_filter(self):
        """Test recent publication filter."""
        result = self.builder.build_recent_filter(3)
        assert result == '"published last 3 months"[Filter]'
    
    def test_build_supplementary_filter(self):
        """Test supplementary file filter."""
        # Single file type
        result = self.builder.build_supplementary_filter(["cel"])
        assert result == "cel[suppFile]"
        
        # Multiple file types
        result = self.builder.build_supplementary_filter(["cel", "h5"])
        assert result == "(cel[suppFile] OR h5[suppFile])"
    
    def test_official_example_queries(self):
        """Test that we can reproduce the official NCBI example queries."""
        examples = self.builder.get_example_queries()
        
        # Example I: GSE[ETYP] AND "published last 3 months"[Filter]
        filters = GEOSearchFilters(
            entry_types=[GEOEntryType.SERIES],
            published_last_n_months=3
        )
        result = self.builder.build_geo_query("", filters)
        assert 'gse[ETYP]' in result
        assert '"published last 3 months"[Filter]' in result
        
        # Example II: yeast[ORGN] AND 2007/01:2007/03[PDAT]
        filters = GEOSearchFilters(
            organisms=["yeast"],
            date_range={"start": "2007/01", "end": "2007/03"}
        )
        result = self.builder.build_geo_query("", filters)
        assert 'yeast[ORGN]' in result
        assert '2007/01:2007/03[PDAT]' in result
        
        # Example III: GPL96[ACCN] AND gse[ETYP] AND cel[suppFile]
        filters = GEOSearchFilters(
            platforms=["GPL96"],
            entry_types=[GEOEntryType.SERIES],
            supplementary_file_types=["cel"]
        )
        result = self.builder.build_geo_query("", filters)
        assert 'GPL96[ACCN]' in result
        assert 'gse[ETYP]' in result
        assert 'cel[suppFile]' in result
    
    def test_validate_query_syntax(self):
        """Test query syntax validation."""
        assert self.builder.validate_query_syntax("human[ORGN]")
        assert self.builder.validate_query_syntax("(human[ORGN] OR mouse[ORGN])")
        assert self.builder.validate_query_syntax('"single-cell RNA-seq"')
        
        # Invalid queries
        assert not self.builder.validate_query_syntax("(human[ORGN")  # Unbalanced parentheses
        assert not self.builder.validate_query_syntax('test "quotes')  # Unbalanced quotes


class TestGEOProvider:
    """Test the GEO provider functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_manager = Mock(spec=DataManagerV2)
        self.config = GEOProviderConfig()
        self.provider = GEOProvider(self.mock_data_manager, self.config)
    
    def test_provider_properties(self):
        """Test basic provider properties."""
        assert self.provider.source == PublicationSource.GEO
        assert DatasetType.GEO in self.provider.supported_dataset_types
    
    def test_validate_identifier(self):
        """Test GEO identifier validation."""
        # Valid identifiers
        assert self.provider.validate_identifier("GSE123456")
        assert self.provider.validate_identifier("GDS1234")
        assert self.provider.validate_identifier("GPL96")
        assert self.provider.validate_identifier("GSM567890")
        
        # Invalid identifiers
        assert not self.provider.validate_identifier("PMID123456")
        assert not self.provider.validate_identifier("10.1038/nature")
        assert not self.provider.validate_identifier("not-a-geo-id")
    
    def test_get_supported_features(self):
        """Test supported features reporting."""
        features = self.provider.get_supported_features()
        assert features["dataset_discovery"] == True
        assert features["advanced_filtering"] == True
        assert features["organism_filtering"] == True
        assert features["platform_filtering"] == True
        assert features["literature_search"] == False  # GEO doesn't search literature
    
    @patch('urllib.request.urlopen')
    def test_search_geo_datasets(self, mock_urlopen):
        """Test GEO dataset search functionality."""
        # Mock eSearch response
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            "esearchresult": {
                "count": "2",
                "idlist": ["123", "456"],
                "webenv": "test_webenv",
                "querykey": "1"
            }
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        # Test search
        filters = GEOSearchFilters(
            organisms=["human"],
            entry_types=[GEOEntryType.SERIES]
        )
        result = self.provider.search_geo_datasets("single-cell RNA-seq", filters)
        
        assert result.count == 2
        assert result.ids == ["123", "456"]
        assert result.web_env == "test_webenv"
        assert result.query_key == "1"
    
    @patch('urllib.request.urlopen')
    def test_get_dataset_summaries(self, mock_urlopen):
        """Test dataset summary retrieval."""
        # Mock eSummary response
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({
            "result": {
                "123": {
                    "uid": "123",
                    "title": "Test Dataset 1",
                    "summary": "Test description",
                    "taxon": "Homo sapiens",
                    "GPL": "GPL24676",
                    "n_samples": "100",
                    "PDAT": "2023/12/01"
                },
                "456": {
                    "uid": "456", 
                    "title": "Test Dataset 2",
                    "summary": "Another test",
                    "taxon": "Mus musculus"
                }
            }
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        # Create search result to test with
        from lobster.tools.providers.geo_provider import GEOSearchResult
        search_result = GEOSearchResult(
            count=2,
            ids=["123", "456"],
            web_env="test_webenv",
            query_key="1"
        )
        
        summaries = self.provider.get_dataset_summaries(search_result)
        
        assert len(summaries) == 2
        assert summaries[0]["title"] == "Test Dataset 1"
        assert summaries[0]["taxon"] == "Homo sapiens"
        assert summaries[1]["title"] == "Test Dataset 2"
    
    @patch('urllib.request.urlopen')
    def test_search_publications_integration(self, mock_urlopen):
        """Test the search_publications method (which does dataset search for GEO)."""
        # Mock both eSearch and eSummary responses
        def mock_urlopen_side_effect(request):
            mock_response = Mock()
            url = request.get_full_url() if hasattr(request, 'get_full_url') else str(request)
            
            if 'esearch.fcgi' in url:
                # eSearch response
                mock_response.read.return_value = json.dumps({
                    "esearchresult": {
                        "count": "1",
                        "idlist": ["123"],
                        "webenv": "test_webenv",
                        "querykey": "1"
                    }
                }).encode('utf-8')
            elif 'esummary.fcgi' in url:
                # eSummary response
                mock_response.read.return_value = json.dumps({
                    "result": {
                        "123": {
                            "uid": "123",
                            "Accession": "GDS123",
                            "title": "Single-cell RNA-seq of human T cells",
                            "summary": "Comprehensive analysis of T cell states",
                            "taxon": "Homo sapiens",
                            "GPL": "GPL24676",
                            "n_samples": "50",
                            "PDAT": "2023/12/01"
                        }
                    }
                }).encode('utf-8')
            
            return mock_response.__enter__.return_value
        
        mock_urlopen.side_effect = mock_urlopen_side_effect
        
        # Test search with filters
        filters = {
            "organisms": ["human"],
            "entry_types": ["gse"]
        }
        
        result = self.provider.search_publications(
            query="single-cell RNA-seq",
            max_results=5,
            filters=filters
        )
        
        # Verify the result is a formatted string
        assert isinstance(result, str)
        assert "GEO DataSets Search Results" in result
        assert "single-cell RNA-seq" in result
        assert "GDS123" in result
    
    def test_convert_filters(self):
        """Test filter conversion from generic dict to GEOSearchFilters."""
        filters_dict = {
            "organisms": ["human", "mouse"],
            "entry_types": ["gse", "gds"],
            "published_last_n_months": 6,
            "supplementary_file_types": ["h5", "cel"],
            "max_results": 10
        }
        
        geo_filters = self.provider._convert_filters(filters_dict)
        
        assert geo_filters.organisms == ["human", "mouse"]
        assert len(geo_filters.entry_types) == 2
        assert GEOEntryType.SERIES in geo_filters.entry_types
        assert GEOEntryType.DATASET in geo_filters.entry_types
        assert geo_filters.published_last_n_months == 6
        assert geo_filters.supplementary_file_types == ["h5", "cel"]
        assert geo_filters.max_results == 10


class TestGEOProviderIntegration:
    """Integration tests for GEO provider with PublicationService."""
    
    @patch('urllib.request.urlopen')
    def test_publication_service_integration(self, mock_urlopen):
        """Test GEO provider integration with PublicationService."""
        from lobster.tools.publication_service import PublicationService
        
        # Mock data manager
        mock_data_manager = Mock(spec=DataManagerV2)
        
        # Create publication service (should auto-register GEO provider)
        service = PublicationService(mock_data_manager)
        
        # Verify GEO provider is registered
        geo_provider = service.registry.get_provider(PublicationSource.GEO)
        assert geo_provider is not None
        assert isinstance(geo_provider, GEOProvider)
        
        # Test that GEO dataset type is supported
        supporting_providers = service.registry.get_providers_for_dataset_type(DatasetType.GEO)
        assert len(supporting_providers) >= 1
        assert any(p.source == PublicationSource.GEO for p in supporting_providers)


def test_official_ncbi_examples():
    """Test that we can construct all four official NCBI example queries."""
    builder = GEOQueryBuilder()
    
    # Example I: GSE[ETYP] AND "published last 3 months"[Filter]
    filters1 = GEOSearchFilters(
        entry_types=[GEOEntryType.SERIES],
        published_last_n_months=3
    )
    query1 = builder.build_geo_query("", filters1)
    expected1 = 'gse[ETYP] AND "published last 3 months"[Filter]'
    assert query1 == expected1
    
    # Example II: yeast[ORGN] AND 2007/01:2007/03[PDAT]  
    filters2 = GEOSearchFilters(
        organisms=["yeast"],
        date_range={"start": "2007/01", "end": "2007/03"}
    )
    query2 = builder.build_geo_query("", filters2)
    expected2 = "yeast[ORGN] AND 2007/01:2007/03[PDAT]"
    assert query2 == expected2
    
    # Example III: GPL96[ACCN] AND gse[ETYP] AND cel[suppFile]
    filters3 = GEOSearchFilters(
        platforms=["GPL96"],
        entry_types=[GEOEntryType.SERIES],
        supplementary_file_types=["cel"]
    )
    query3 = builder.build_geo_query("", filters3)
    expected3 = "GPL96[ACCN] AND gse[ETYP] AND cel[suppFile]"
    assert query3 == expected3
    
    # Example IV: Complex query with keyword and filters
    filters4 = GEOSearchFilters(
        organisms=["human"],
        entry_types=[GEOEntryType.SERIES],
        supplementary_file_types=["h5"]
    )
    query4 = builder.build_geo_query("single-cell RNA-seq", filters4)
    expected_parts = ['"single-cell RNA-seq"', 'human[ORGN]', 'gse[ETYP]', 'h5[suppFile]']
    for part in expected_parts:
        assert part in query4
    assert query4.count(" AND ") == 3  # Should have 3 AND connectors


if __name__ == "__main__":
    pytest.main([__file__])
