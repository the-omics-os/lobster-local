"""
Lobster Cloud Client - Connects to Lobster Cloud API
"""

import os
import requests
from typing import Dict, Any, Optional
from lobster_core.interfaces.base_client import BaseLobsterClient


class CloudLobsterClient(BaseLobsterClient):
    """
    Cloud client for Lobster AI that connects to the remote API
    """
    
    def __init__(
        self, 
        api_key: str, 
        endpoint: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize the cloud client
        
        Args:
            api_key: API key for authentication
            endpoint: API endpoint URL (defaults to production)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.endpoint = endpoint or os.getenv(
            'LOBSTER_ENDPOINT', 
            'https://api.lobster.homara.ai'
        )
        self.timeout = timeout
        
        # Create session with headers
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Lobster-Cloud-Client/2.0.0"
        })
    
    def query(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """
        Process a user query via the cloud API
        
        Args:
            user_input: The user's question or request
            **kwargs: Additional options (workspace, reasoning, etc.)
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            payload = {
                "query": user_input,
                "options": kwargs
            }
            
            response = self.session.post(
                f"{self.endpoint}/query",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.Timeout:
            return {
                "error": "Request timed out. The cloud service may be experiencing high load.",
                "success": False
            }
        except requests.exceptions.ConnectionError:
            return {
                "error": "Could not connect to Lobster Cloud. Please check your internet connection.",
                "success": False
            }
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                return {
                    "error": "Invalid API key. Please check your LOBSTER_CLOUD_KEY.",
                    "success": False
                }
            elif response.status_code == 429:
                return {
                    "error": "Rate limit exceeded. Please try again later.",
                    "success": False
                }
            else:
                return {
                    "error": f"HTTP error: {e}",
                    "success": False
                }
        except Exception as e:
            return {
                "error": f"Unexpected error: {e}",
                "success": False
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the cloud service
        
        Returns:
            Dict containing status information
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/status",
                timeout=10  # Shorter timeout for status checks
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Could not get status: {e}",
                "success": False
            }
    
    def get_usage(self) -> Dict[str, Any]:
        """
        Get usage statistics for the current API key
        
        Returns:
            Dict containing usage information
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/usage",
                timeout=10
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {
                "error": f"Could not get usage info: {e}",
                "success": False
            }
    
    def list_models(self) -> Dict[str, Any]:
        """
        List available models in the cloud
        
        Returns:
            Dict containing available models
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/models",
                timeout=10
            )
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {
                "error": f"Could not list models: {e}",
                "success": False
            }
