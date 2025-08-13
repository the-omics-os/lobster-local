"""
Test script to specifically verify DOI validation improvements.

This test focuses on validating the regex pattern used for DOI validation
in the PubMed service to ensure it properly identifies valid and invalid DOIs.
"""

import unittest
import re


class TestDOIValidation(unittest.TestCase):
    """Test the DOI validation regex pattern."""
    
    def setUp(self):
        """Set up the test environment with the DOI pattern."""
        # This is the exact pattern from the PubMedService class
        self.doi_pattern_str = r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+'
        self.doi_pattern = re.compile(self.doi_pattern_str, re.IGNORECASE)
    
    def test_valid_doi_formats(self):
        """Test that valid DOI formats pass validation."""
        valid_dois = [
            "10.1038/s41586-021-03659-0",
            "10.1126/science.aaf1420",
            "10.1093/nar/gkab346",
            "10.1101/2020.12.08.416503",
            "10.1016/j.cell.2019.05.031"
        ]
        
        for doi in valid_dois:
            with self.subTest(doi=doi):
                # Check using the regex directly
                self.assertTrue(self.doi_pattern.match(doi), 
                               f"DOI format validation failed for valid DOI: {doi}")
    
    def test_invalid_doi_formats(self):
        """Test that invalid DOI formats fail validation."""
        invalid_dois = [
            "doi:10.1038/s41586-021-03659-0",  # with doi: prefix
            "https://doi.org/10.1038/s41586-021-03659-0",  # full URL
            "https://pubmed.ncbi.nlm.nih.gov/12345678/",  # PubMed URL
            "PMC12345678",  # PubMed Central ID
            "PMID:12345678",  # PubMed ID with prefix
            "12345678",  # Just a number
            "not-a-doi",  # Random text
            "11.1234/abcd",  # Wrong prefix (11 instead of 10)
        ]
        
        for doi in invalid_dois:
            with self.subTest(doi=doi):
                # Check using the regex directly
                self.assertFalse(self.doi_pattern.match(doi), 
                                f"DOI format validation incorrectly passed for invalid DOI: {doi}")
    
    def test_doi_extraction_simulation(self):
        """Test simulating the DOI extraction logic."""
        test_cases = [
            # (input, should_pass_after_extraction)
            ("https://doi.org/10.1038/s41586-021-03659-0", True),  # Valid DOI in URL
            ("doi:10.1038/s41586-021-03659-0", False),  # Not properly formatted URL
            ("https://pubmed.ncbi.nlm.nih.gov/12345678/", False),  # PubMed URL, no DOI
        ]
        
        for input_value, should_pass in test_cases:
            with self.subTest(input=input_value):
                # Simulate the extraction logic in find_geo_from_doi
                extracted = input_value
                if "doi.org" in input_value:
                    extracted = input_value.split("doi.org/")[-1]
                
                if should_pass:
                    self.assertTrue(self.doi_pattern.match(extracted),
                                  f"Extracted DOI should pass validation: {extracted}")
                else:
                    self.assertFalse(self.doi_pattern.match(extracted),
                                   f"Extracted DOI should fail validation: {extracted}")


if __name__ == "__main__":
    unittest.main()
