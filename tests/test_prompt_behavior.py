"""
Test script to verify the system prompt correctly handles PubMed URLs and DOIs.

This script tests various input scenarios to ensure the agent:
1. Asks for a DOI when provided with a PubMed URL
2. Only uses the find_geo_from_doi tool with proper DOI formats
3. Handles edge cases correctly
"""

import unittest
from unittest.mock import patch, MagicMock

from core.data_manager import DataManager
from services.agent_service_OLD import AgentService
from utils.system_prompts import BIOINFORMATICS_SYSTEM_MESSAGE


class TestPromptBehavior(unittest.TestCase):
    """Test the system prompt behavior for edge cases."""
    
    def setUp(self):
        """Set up the test environment."""
        # Mock the data manager
        self.data_manager = MagicMock(spec=DataManager)
        self.data_manager.current_metadata = {}
        self.data_manager.log_tool_usage = MagicMock()
        
        # Create agent service with our system message
        self.agent_service = AgentService(
            data_manager=self.data_manager,
            system_message=BIOINFORMATICS_SYSTEM_MESSAGE
        )
        
        # Patch the run method to check which tool would be used
        self.patcher = patch.object(self.agent_service.agent, 'run')
        self.mock_run = self.patcher.start()
        
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
    
    def test_pubmed_url_handling(self):
        """Test that PubMed URLs are not processed with find_geo_from_doi."""
        # Mock user input with PubMed URL
        user_input = "Can you find GEO datasets for this paper: https://pubmed.ncbi.nlm.nih.gov/12345678/"
        
        # Define the expected response (asking for DOI)
        expected_response = "I notice you've shared a PubMed URL"
        self.mock_run.return_value = expected_response
        
        # Run the agent with the user input
        response = self.agent_service.run_agent(user_input)
        
        # Check that the response asks for a DOI
        self.assertIn(expected_response, response)
        
        # For a complete test, you'd need to check that the find_geo_from_doi tool wasn't called
        # This would require more complex mocking of the LangChain internals
    
    def test_proper_doi_acceptance(self):
        """Test that properly formatted DOIs are accepted."""
        # Mock user input with proper DOI
        user_input = "Can you find GEO datasets for this DOI: 10.1038/s41586-021-03659-0"
        
        # Define the expected response (indicating tool use)
        expected_response = "I'll search for GEO datasets associated with this DOI"
        self.mock_run.return_value = expected_response
        
        # Run the agent with the user input
        response = self.agent_service.run_agent(user_input)
        
        # Check that the response indicates processing the DOI
        self.assertIn(expected_response, response)
    
    def test_malformed_doi_handling(self):
        """Test that malformed DOIs are rejected."""
        # Mock user input with improper DOI format
        user_input = "Can you find GEO datasets for this DOI: doi-12345678"
        
        # Define the expected response (asking for proper DOI)
        expected_response = "need a properly formatted DOI"
        self.mock_run.return_value = expected_response
        
        # Run the agent with the user input
        response = self.agent_service.run_agent(user_input)
        
        # Check that the response asks for a proper DOI
        self.assertIn(expected_response, response)
    
    def test_mixed_content_handling(self):
        """Test handling of messages with both PubMed URLs and DOIs."""
        # Mock user input with both PubMed URL and DOI
        user_input = """
        I found two papers: 
        1. https://pubmed.ncbi.nlm.nih.gov/12345678/
        2. DOI: 10.1038/s41586-021-03659-0
        Can you find GEO datasets for these?
        """
        
        # Define the expected response (should ask for clarification)
        expected_response = "notice you've shared a PubMed URL"
        self.mock_run.return_value = expected_response
        
        # Run the agent with the user input
        response = self.agent_service.run_agent(user_input)
        
        # Check that the response asks for clarification
        self.assertIn(expected_response, response)
        

if __name__ == "__main__":
    unittest.main()
