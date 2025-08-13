"""
Test script to verify the enhanced GEO dataset detection from publications.

This test focuses on the improved find_geo_from_doi method in PubMedService,
which now detects GEO, SRA, and platform accessions in publication data.
"""

import unittest
from unittest.mock import MagicMock, patch

from lobster.core import DataManager
from lobster.tools import PubMedService


class TestGEODetection(unittest.TestCase):
    """Test the enhanced GEO detection capabilities."""
    
    def setUp(self):
        """Set up the test environment."""
        self.data_manager = MagicMock(spec=DataManager)
        self.data_manager.log_tool_usage = MagicMock()
        
        # Create mock publication data with various accessions
        self.mock_results = [
            {
                'uid': '12345678',
                'Title': 'Test Article with GEO Accession',
                'Journal': 'Bioinformatics Journal',
                'Published': '2023-01-01',
                'Summary': 'Methods: Data was deposited to GEO under accession GSE123456. Additional data available as SRP987654.'
            },
            {
                'uid': '23456789',
                'Title': 'Test Article with Multiple Accessions',
                'Journal': 'Genomics Journal',
                'Published': '2023-02-01',
                'Summary': 'We analyzed RNA-seq data (GSE234567) using the platform GPL1234. Raw reads are available as SRR123456.'
            },
            {
                'uid': '34567890',
                'Title': 'Test Article with No Accessions',
                'Journal': 'Biology Journal',
                'Published': '2023-03-01',
                'Summary': 'We conducted a literature review of RNA-seq studies without generating new data.'
            }
        ]
        
        # Bypass the xmltodict import check in PubMedService
        with patch('services.pubmed_service.PubMedService.validate_environment') as mock_validate:
            mock_validate.return_value = {"parse": lambda x: x}
            self.pubmed_service = PubMedService(parse=lambda x: x, data_manager=self.data_manager)
    
    def test_geo_detection(self):
        """Test that GEO accessions are correctly detected."""
        with patch.object(self.pubmed_service, 'load', return_value=self.mock_results):
            result = self.pubmed_service.find_geo_from_doi("10.1234/test")
            
            # Check that GEO accessions were found
            self.assertIn("GSE123456", result)
            self.assertIn("GSE234567", result)
            self.assertIn("Found 2 GEO dataset accession", result)
    
    def test_sra_detection(self):
        """Test that SRA accessions are correctly detected."""
        with patch.object(self.pubmed_service, 'load', return_value=self.mock_results):
            result = self.pubmed_service.find_geo_from_doi("10.1234/test")
            
            # Check that SRA accessions were found
            self.assertIn("SRP987654", result)
            self.assertIn("SRR123456", result)
            self.assertIn("Found 2 SRA accession", result)
    
    def test_platform_detection(self):
        """Test that GEO platform accessions are correctly detected."""
        with patch.object(self.pubmed_service, 'load', return_value=self.mock_results):
            result = self.pubmed_service.find_geo_from_doi("10.1234/test")
            
            # Check that platform accessions were found
            self.assertIn("GPL1234", result)
            self.assertIn("Found 1 GEO platform accession", result)
    
    def test_publication_details(self):
        """Test that publication details are included in the response."""
        with patch.object(self.pubmed_service, 'load', return_value=self.mock_results):
            result = self.pubmed_service.find_geo_from_doi("10.1234/test")
            
            # Check that publication details are included
            self.assertIn("Publication Details", result)
            self.assertIn("Test Article with GEO Accession", result)
            self.assertIn("Test Article with Multiple Accessions", result)
            self.assertIn("Test Article with No Accessions", result)
    
    def test_no_accessions(self):
        """Test behavior when no accessions are found."""
        # Create a mock result with no accessions
        no_accession_results = [
            {
                'uid': '12345678',
                'Title': 'Test Article with No Accessions',
                'Journal': 'Biology Journal',
                'Published': '2023-01-01',
                'Summary': 'This paper contains no mention of any GEO or SRA data.'
            }
        ]
        
        with patch.object(self.pubmed_service, 'load', return_value=no_accession_results):
            result = self.pubmed_service.find_geo_from_doi("10.1234/test")
            
            # Check that the appropriate message is included
            self.assertIn("No GEO or SRA accession numbers were found", result)
    
    def test_multiple_accessions_in_single_article(self):
        """Test detection of multiple accessions in a single article."""
        # Create a mock result with multiple accessions in one article
        multi_accession_results = [
            {
                'uid': '12345678',
                'Title': 'Test Article with Multiple Accessions',
                'Journal': 'Genomics Journal',
                'Published': '2023-01-01',
                'Summary': 'We used multiple datasets: GSE111, GSE222, GSE333, and platforms GPL111, GPL222. ' + 
                           'Raw data available as SRR111, SRR222, and SRR333.'
            }
        ]
        
        with patch.object(self.pubmed_service, 'load', return_value=multi_accession_results):
            result = self.pubmed_service.find_geo_from_doi("10.1234/test")
            
            # Check that all accessions were found
            for acc in ["GSE111", "GSE222", "GSE333", "GPL111", "GPL222", "SRR111", "SRR222", "SRR333"]:
                self.assertIn(acc, result)
            
            # Check accession counts
            self.assertIn("Found 3 GEO dataset accession", result)
            self.assertIn("Found 3 SRA accession", result)
            self.assertIn("Found 2 GEO platform accession", result)


if __name__ == "__main__":
    unittest.main()
