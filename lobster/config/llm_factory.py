"""
LLM Factory for provider-agnostic model instantiation.

This module provides a unified interface for creating LLM instances
regardless of the underlying provider (Claude API, AWS Bedrock, etc.).
"""

from typing import Dict, Any, Optional
from enum import Enum
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC_DIRECT = "anthropic"
    BEDROCK_ANTHROPIC = "bedrock"
    OPENAI = "openai"  # Future support
    AZURE_OPENAI = "azure_openai"  # Future support


class LLMFactory:
    """Factory for creating provider-agnostic LLM instances."""
    
    # Model mapping: provider-agnostic name -> provider-specific IDs
    MODEL_MAPPINGS = {
        # Ultra tier models
        "claude-3-7-sonnet": {
            LLMProvider.BEDROCK_ANTHROPIC: "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            LLMProvider.ANTHROPIC_DIRECT: "claude-3-5-sonnet-20241022"
        },
        # Heavy tier models  
        "claude-4-opus": {
            LLMProvider.BEDROCK_ANTHROPIC: "us.anthropic.claude-opus-4-20250514-v1:0",
            LLMProvider.ANTHROPIC_DIRECT: "claude-3-opus-20240229"
        },
        # Standard tier models
        "claude-4-sonnet": {
            LLMProvider.BEDROCK_ANTHROPIC: "us.anthropic.claude-sonnet-4-20250514-v1:0",
            LLMProvider.ANTHROPIC_DIRECT: "claude-3-5-sonnet-20241022"
        },
        # Lightweight tier models
        "claude-3-haiku": {
            LLMProvider.BEDROCK_ANTHROPIC: "us.anthropic.claude-3-haiku-20240307-v1:0",
            LLMProvider.ANTHROPIC_DIRECT: "claude-3-haiku-20240307"
        }
    }
    
    @classmethod
    def detect_provider(cls) -> Optional[LLMProvider]:
        """
        Auto-detect available provider based on environment variables.
        
        Returns:
            LLMProvider enum value or None if no provider credentials found
        """
        # Check for explicit provider override
        provider_override = os.environ.get('LOBSTER_LLM_PROVIDER')
        if provider_override:
            try:
                return LLMProvider(provider_override)
            except ValueError:
                print(f"Warning: Invalid LOBSTER_LLM_PROVIDER value '{provider_override}', falling back to auto-detection")
        
        # Priority order: Direct API > Bedrock > Others
        if os.environ.get('ANTHROPIC_API_KEY'):
            return LLMProvider.ANTHROPIC_DIRECT
        elif os.environ.get('AWS_BEDROCK_ACCESS_KEY') and os.environ.get('AWS_BEDROCK_SECRET_ACCESS_KEY'):
            return LLMProvider.BEDROCK_ANTHROPIC
        elif os.environ.get('OPENAI_API_KEY'):
            return LLMProvider.OPENAI
        
        return None
    
    @classmethod
    def create_llm(cls, model_config: Dict[str, Any], agent_name: str = None) -> Any:
        """
        Create an LLM instance based on configuration and detected provider.
        
        Args:
            model_config: Configuration dictionary with model parameters
            agent_name: Optional agent name for logging
            
        Returns:
            LLM instance (ChatAnthropic, ChatBedrockConverse, etc.)
            
        Raises:
            ValueError: If no provider credentials are found or provider not implemented
        """
        # Detect provider
        provider = cls.detect_provider()
        
        if not provider:
            raise ValueError(
                "No LLM provider credentials found. Please set one of the following:\n"
                "  - ANTHROPIC_API_KEY for Claude API\n"
                "  - AWS_BEDROCK_ACCESS_KEY and AWS_BEDROCK_SECRET_ACCESS_KEY for AWS Bedrock"
            )
        
        # Log provider selection if agent_name provided
        if agent_name:
            print(f"Creating LLM for agent '{agent_name}' using provider: {provider.value}")
        
        # Get model ID from config
        model_id = model_config.get('model_id')
        
        # Create appropriate LLM instance based on provider
        if provider == LLMProvider.ANTHROPIC_DIRECT:
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                raise ImportError(
                    "langchain-anthropic package not installed. "
                    "Please run: pip install langchain-anthropic"
                )
            
            # Translate model ID if needed
            translated_model = cls._translate_model_id(model_id, provider)
            
            # Create ChatAnthropic instance
            return ChatAnthropic(
                model=translated_model,
                api_key=os.environ.get('ANTHROPIC_API_KEY'),
                temperature=model_config.get('temperature', 1.0),
                max_tokens=model_config.get('max_tokens', 4096)
            )
            
        elif provider == LLMProvider.BEDROCK_ANTHROPIC:
            try:
                from langchain_aws import ChatBedrockConverse
            except ImportError:
                raise ImportError(
                    "langchain-aws package not installed. "
                    "Please run: pip install langchain-aws"
                )
            
            # For Bedrock, use the original model ID (no translation needed)
            bedrock_params = {
                'model_id': model_id,
                'temperature': model_config.get('temperature', 1.0)
            }
            
            # Add region if specified
            if 'region_name' in model_config:
                bedrock_params['region_name'] = model_config['region_name']
            else:
                bedrock_params['region_name'] = 'us-east-1'  # Default region
            
            # Add AWS credentials if explicitly provided
            aws_access_key = os.environ.get('AWS_BEDROCK_ACCESS_KEY')
            aws_secret_key = os.environ.get('AWS_BEDROCK_SECRET_ACCESS_KEY')
            
            if aws_access_key and aws_secret_key:
                bedrock_params['aws_access_key_id'] = aws_access_key
                bedrock_params['aws_secret_access_key'] = aws_secret_key
            
            return ChatBedrockConverse(**bedrock_params)
            
        elif provider == LLMProvider.OPENAI:
            raise NotImplementedError(
                "OpenAI provider support is planned for future release. "
                "Please use Claude API or AWS Bedrock for now."
            )
            
        else:
            raise ValueError(f"Provider {provider.value} is not yet implemented")
    
    @classmethod
    def _translate_model_id(cls, bedrock_model_id: str, target_provider: LLMProvider) -> str:
        """
        Translate Bedrock model ID to provider-specific ID.
        
        Args:
            bedrock_model_id: The Bedrock model ID to translate
            target_provider: The target provider to translate for
            
        Returns:
            Translated model ID for the target provider
        """
        # First, try to find exact match in mappings
        for model_name, mappings in cls.MODEL_MAPPINGS.items():
            if mappings.get(LLMProvider.BEDROCK_ANTHROPIC) == bedrock_model_id:
                translated = mappings.get(target_provider)
                if translated:
                    return translated
        
        # Fallback: try to extract model version from Bedrock ID
        if "claude-3-5-sonnet" in bedrock_model_id or "claude-3-7-sonnet" in bedrock_model_id:
            return "claude-3-5-sonnet-20241022"
        elif "claude-3-haiku" in bedrock_model_id:
            return "claude-3-haiku-20240307"
        elif "claude-3-opus" in bedrock_model_id or "claude-opus-4" in bedrock_model_id:
            return "claude-3-opus-20240229"
        elif "claude-sonnet-4" in bedrock_model_id:
            return "claude-3-5-sonnet-20241022"
        
        # Default fallback to Claude 3.5 Sonnet
        print(f"Warning: Could not translate model ID '{bedrock_model_id}', using default Claude 3.5 Sonnet")
        return "claude-3-5-sonnet-20241022"
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """
        Get list of providers that have credentials configured.
        
        Returns:
            List of available provider names
        """
        available = []
        
        if os.environ.get('ANTHROPIC_API_KEY'):
            available.append(LLMProvider.ANTHROPIC_DIRECT.value)
        
        if os.environ.get('AWS_BEDROCK_ACCESS_KEY') and os.environ.get('AWS_BEDROCK_SECRET_ACCESS_KEY'):
            available.append(LLMProvider.BEDROCK_ANTHROPIC.value)
        
        if os.environ.get('OPENAI_API_KEY'):
            available.append(LLMProvider.OPENAI.value)
        
        return available
    
    @classmethod
    def get_current_provider(cls) -> Optional[str]:
        """
        Get the currently selected provider.
        
        Returns:
            Provider name or None if no provider available
        """
        provider = cls.detect_provider()
        return provider.value if provider else None


# Convenience function for backward compatibility
def create_llm(agent_name: str, model_params: Dict[str, Any]) -> Any:
    """
    Create an LLM instance for a specific agent.
    
    This is a convenience function that maintains backward compatibility
    with the existing agent code.
    
    Args:
        agent_name: Name of the agent requesting the LLM
        model_params: Model configuration parameters
        
    Returns:
        LLM instance configured for the agent
    """
    return LLMFactory.create_llm(model_params, agent_name)
