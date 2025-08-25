"""
Application settings and configuration.

This module centralizes all configuration settings for the application,
including the new professional agent configuration system.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from lobster.config.agent_config import initialize_configurator
from lobster.config.agent_registry import get_all_agent_names, get_worker_agents, get_agent_config

class Settings:
    """
    Application settings with environment variable support.
    
    This class manages application-wide settings with fallbacks
    and environment variable overrides for easier configuration
    in different environments, especially in containers.
    """
    
    def __init__(self):
        """Initialize application settings."""
        # Load dotenv
        load_dotenv()

        #CDK variables
        self.STACK_NAME = 'LobsterStack'
        self.CUSTOM_HEADER_VALUE = 'HomaraBeatsKepler'
        self.SECRETS_MANAGER_ID = f"{self.STACK_NAME}ParamCognitoSecret"
        self.CDK_DEPLY_ACCOUNT = '649207544517'
        # AWS Fargate CPU/Memory options summary:
        # - 256 (.25 vCPU): 512 MiB, 1 GB, 2 GB (Linux)
        # - 512 (.5 vCPU): 1 GB, 2 GB, 3 GB, 4 GB (Linux)
        # - 1024 (1 vCPU): 2-8 GB (Linux, Windows)
        # - 2048 (2 vCPU): 4-16 GB (Linux, Windows, 1 GB steps)
        # - 4096 (4 vCPU): 8-30 GB (Linux, Windows, 1 GB steps)
        # - 8192 (8 vCPU, Linux 1.4.0+): 16-60 GB (4 GB steps)
        # - 16384 (16 vCPU, Linux 1.4.0+): 32-120 GB (8 GB steps)
        # See AWS docs for full details.        
        self.MEMORY = 24576
        self.CPU = 8192
        # Initialize agent configurator based on environment
        profile = os.environ.get('GENIE_PROFILE', 'production')
        config_file = os.environ.get('GENIE_CONFIG_FILE')
        self.agent_configurator = initialize_configurator(profile=profile, config_file=config_file)

        # Base directories
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.CACHE_DIR = os.environ.get('GENIE_CACHE_DIR', 
                                      str(self.BASE_DIR / 'data' / 'cache'))
        
        # Create cache directory if it doesn't exist
        Path(self.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        
        # API keys
        self.OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
        self.AWS_BEDROCK_ACCESS_KEY = os.environ.get('AWS_BEDROCK_ACCESS_KEY', '')
        self.AWS_BEDROCK_SECRET_ACCESS_KEY = os.environ.get('AWS_BEDROCK_SECRET_ACCESS_KEY', '')
        self.NCBI_API_KEY = os.environ.get('NCBI_API_KEY', '')

        # AWS region (fallback for backward compatibility)
        self.REGION = os.environ.get('AWS_REGION', 'us-east-1')
        
        # Web server settings
        self.PORT = int(os.environ.get('PORT', '8501'))
        self.HOST = os.environ.get('HOST', '0.0.0.0')
        self.DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
        
        # Data processing settings
        self.MAX_FILE_SIZE_MB = int(os.environ.get('GENIE_MAX_FILE_SIZE_MB', '500'))
        self.DEFAULT_CLUSTER_RESOLUTION = float(os.environ.get('GENIE_CLUSTER_RESOLUTION', '0.5'))
        
        # GEO database settings
        self.GEO_CACHE_DIR = os.path.join(self.CACHE_DIR, 'geo')
        Path(self.GEO_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all settings as a dictionary.
        
        Returns:
            dict: All settings
        """
        settings_dict = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                settings_dict[attr] = getattr(self, attr)
        return settings_dict

    def get_setting(self, name: str, default: Any = None) -> Any:
        """
        Get a specific setting.
        
        Args:
            name: Setting name
            default: Default value if setting doesn't exist
            
        Returns:
            Value of the setting or default
        """
        return getattr(self, name, default)
    
    def get_agent_llm_params(self, agent_name: str) -> Dict[str, Any]:
        """
        Get LLM parameters for a specific agent using the new configuration system.
        
        Args:
            agent_name: Name of the agent (e.g., 'supervisor', 'transcriptomics_expert', 'method_agent')
            
        Returns:
            Dictionary of LLM initialization parameters
        """
        try:
            return self.agent_configurator.get_llm_params(agent_name)
        except KeyError:
            # Fallback to legacy settings if agent not configured
            return {
                "model_id": self.LLM_MODEL,
                "temperature": self.LLM_TEMPERATURE,
                "region_name": self.REGION,
                "aws_access_key_id": self.AWS_BEDROCK_ACCESS_KEY,
                "aws_secret_access_key": self.AWS_BEDROCK_SECRET_ACCESS_KEY,
            }
    
    def get_agent_model_config(self, agent_name: str):
        """
        Get model configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            ModelConfig object for the agent
        """
        return self.agent_configurator.get_model_config(agent_name)
    
    def print_agent_configuration(self):
        """Print current agent configuration."""
        self.agent_configurator.print_current_config()

# Create singleton instance
settings = Settings()

def get_settings() -> Settings:
    """
    Get the application settings.
    
    Returns:
        Settings: Application settings
    """
    return settings
