"""
Unit tests for the LLM Factory module.

Tests provider detection, LLM creation, and model ID translation.
"""

import pytest
from unittest.mock import patch, MagicMock, call
import os
from lobster.config.llm_factory import LLMFactory, LLMProvider, create_llm


class TestLLMProvider:
    """Test the LLMProvider enum."""
    
    def test_provider_values(self):
        """Test that provider enum has expected values."""
        assert LLMProvider.ANTHROPIC_DIRECT.value == "anthropic"
        assert LLMProvider.BEDROCK_ANTHROPIC.value == "bedrock"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.AZURE_OPENAI.value == "azure_openai"


class TestProviderDetection:
    """Test provider detection logic."""
    
    def test_detect_anthropic_provider(self):
        """Test detection when only Anthropic API key is present."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}, clear=True):
            provider = LLMFactory.detect_provider()
            assert provider == LLMProvider.ANTHROPIC_DIRECT
    
    def test_detect_bedrock_provider(self):
        """Test detection when only AWS Bedrock credentials are present."""
        with patch.dict('os.environ', {
            'AWS_BEDROCK_ACCESS_KEY': 'test-key',
            'AWS_BEDROCK_SECRET_ACCESS_KEY': 'test-secret'
        }, clear=True):
            provider = LLMFactory.detect_provider()
            assert provider == LLMProvider.BEDROCK_ANTHROPIC
    
    def test_detect_no_provider(self):
        """Test detection when no credentials are present."""
        with patch.dict('os.environ', {}, clear=True):
            provider = LLMFactory.detect_provider()
            assert provider is None
    
    def test_provider_priority_anthropic_over_bedrock(self):
        """Test that Anthropic API takes priority over Bedrock."""
        with patch.dict('os.environ', {
            'ANTHROPIC_API_KEY': 'test-key',
            'AWS_BEDROCK_ACCESS_KEY': 'test-key',
            'AWS_BEDROCK_SECRET_ACCESS_KEY': 'test-secret'
        }, clear=True):
            provider = LLMFactory.detect_provider()
            assert provider == LLMProvider.ANTHROPIC_DIRECT
    
    def test_provider_override(self):
        """Test LOBSTER_LLM_PROVIDER override."""
        with patch.dict('os.environ', {
            'ANTHROPIC_API_KEY': 'test-key',
            'LOBSTER_LLM_PROVIDER': 'bedrock'
        }, clear=True):
            provider = LLMFactory.detect_provider()
            assert provider == LLMProvider.BEDROCK_ANTHROPIC
    
    def test_invalid_provider_override(self):
        """Test invalid LOBSTER_LLM_PROVIDER value falls back to auto-detection."""
        with patch.dict('os.environ', {
            'ANTHROPIC_API_KEY': 'test-key',
            'LOBSTER_LLM_PROVIDER': 'invalid_provider'
        }, clear=True):
            with patch('builtins.print') as mock_print:
                provider = LLMFactory.detect_provider()
                assert provider == LLMProvider.ANTHROPIC_DIRECT
                mock_print.assert_called_with(
                    "Warning: Invalid LOBSTER_LLM_PROVIDER value 'invalid_provider', falling back to auto-detection"
                )


class TestLLMCreation:
    """Test LLM instance creation."""
    
    @patch('langchain_anthropic.ChatAnthropic')
    def test_create_anthropic_llm(self, mock_anthropic):
        """Test creating ChatAnthropic instance."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}, clear=True):
            model_config = {
                'model_id': 'us.anthropic.claude-3-haiku-20240307-v1:0',
                'temperature': 0.7,
                'max_tokens': 2048
            }
            
            llm = LLMFactory.create_llm(model_config, 'test_agent')
            
            mock_anthropic.assert_called_once_with(
                model='claude-3-haiku-20240307',
                api_key='test-key',
                temperature=0.7,
                max_tokens=2048
            )
    
    @patch('langchain_aws.ChatBedrockConverse')
    def test_create_bedrock_llm(self, mock_bedrock):
        """Test creating ChatBedrockConverse instance."""
        with patch.dict('os.environ', {
            'AWS_BEDROCK_ACCESS_KEY': 'test-key',
            'AWS_BEDROCK_SECRET_ACCESS_KEY': 'test-secret'
        }, clear=True):
            model_config = {
                'model_id': 'us.anthropic.claude-3-haiku-20240307-v1:0',
                'temperature': 0.7,
                'region_name': 'us-west-2'
            }
            
            llm = LLMFactory.create_llm(model_config, 'test_agent')
            
            mock_bedrock.assert_called_once_with(
                model_id='us.anthropic.claude-3-haiku-20240307-v1:0',
                temperature=0.7,
                region_name='us-west-2',
                aws_access_key_id='test-key',
                aws_secret_access_key='test-secret'
            )
    
    def test_create_llm_no_credentials(self):
        """Test error when no credentials are available."""
        with patch.dict('os.environ', {}, clear=True):
            model_config = {'model_id': 'test-model'}
            
            with pytest.raises(ValueError) as exc_info:
                LLMFactory.create_llm(model_config)
            
            assert "No LLM provider credentials found" in str(exc_info.value)
    
    def test_create_openai_not_implemented(self):
        """Test that OpenAI provider raises NotImplementedError."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}, clear=True):
            model_config = {'model_id': 'gpt-4'}
            
            with pytest.raises(NotImplementedError) as exc_info:
                LLMFactory.create_llm(model_config)
            
            assert "OpenAI provider support is planned" in str(exc_info.value)


class TestModelTranslation:
    """Test model ID translation between providers."""
    
    def test_translate_haiku_model(self):
        """Test translating Haiku model ID."""
        bedrock_id = "us.anthropic.claude-3-haiku-20240307-v1:0"
        translated = LLMFactory._translate_model_id(bedrock_id, LLMProvider.ANTHROPIC_DIRECT)
        assert translated == "claude-3-haiku-20240307"
    
    def test_translate_sonnet_model(self):
        """Test translating Sonnet model ID."""
        bedrock_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        translated = LLMFactory._translate_model_id(bedrock_id, LLMProvider.ANTHROPIC_DIRECT)
        assert translated == "claude-3-5-sonnet-20241022"
    
    def test_translate_opus_model(self):
        """Test translating Opus model ID."""
        bedrock_id = "us.anthropic.claude-opus-4-20250514-v1:0"
        translated = LLMFactory._translate_model_id(bedrock_id, LLMProvider.ANTHROPIC_DIRECT)
        assert translated == "claude-3-opus-20240229"
    
    def test_translate_unknown_model_fallback(self):
        """Test fallback for unknown model ID."""
        with patch('builtins.print') as mock_print:
            bedrock_id = "unknown-model-id"
            translated = LLMFactory._translate_model_id(bedrock_id, LLMProvider.ANTHROPIC_DIRECT)
            
            assert translated == "claude-3-5-sonnet-20241022"
            mock_print.assert_called_with(
                "Warning: Could not translate model ID 'unknown-model-id', using default Claude 3.5 Sonnet"
            )
    
    def test_translate_partial_match_sonnet(self):
        """Test translation with partial model name match."""
        bedrock_id = "some-prefix-claude-3-5-sonnet-suffix"
        translated = LLMFactory._translate_model_id(bedrock_id, LLMProvider.ANTHROPIC_DIRECT)
        assert translated == "claude-3-5-sonnet-20241022"


class TestUtilityMethods:
    """Test utility methods of LLMFactory."""
    
    def test_get_available_providers_all(self):
        """Test getting list of available providers when all are configured."""
        with patch.dict('os.environ', {
            'ANTHROPIC_API_KEY': 'test-key',
            'AWS_BEDROCK_ACCESS_KEY': 'test-key',
            'AWS_BEDROCK_SECRET_ACCESS_KEY': 'test-secret',
            'OPENAI_API_KEY': 'test-key'
        }, clear=True):
            providers = LLMFactory.get_available_providers()
            assert set(providers) == {'anthropic', 'bedrock', 'openai'}
    
    def test_get_available_providers_none(self):
        """Test getting list of available providers when none are configured."""
        with patch.dict('os.environ', {}, clear=True):
            providers = LLMFactory.get_available_providers()
            assert providers == []
    
    def test_get_current_provider(self):
        """Test getting the current provider."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}, clear=True):
            provider = LLMFactory.get_current_provider()
            assert provider == 'anthropic'
    
    def test_get_current_provider_none(self):
        """Test getting current provider when none available."""
        with patch.dict('os.environ', {}, clear=True):
            provider = LLMFactory.get_current_provider()
            assert provider is None


class TestConvenienceFunction:
    """Test the convenience function for backward compatibility."""
    
    @patch.object(LLMFactory, 'create_llm')
    def test_create_llm_convenience_function(self, mock_factory_create):
        """Test that convenience function calls factory correctly."""
        model_params = {'model_id': 'test-model', 'temperature': 0.5}
        agent_name = 'test_agent'
        
        create_llm(agent_name, model_params)
        
        mock_factory_create.assert_called_once_with(model_params, agent_name)


class TestBackwardCompatibility:
    """Test backward compatibility with existing Bedrock configurations."""
    
    @patch('langchain_aws.ChatBedrockConverse')
    def test_existing_bedrock_config_works(self, mock_bedrock):
        """Test that existing Bedrock configurations continue to work."""
        with patch.dict('os.environ', {
            'AWS_BEDROCK_ACCESS_KEY': 'existing-key',
            'AWS_BEDROCK_SECRET_ACCESS_KEY': 'existing-secret'
        }, clear=True):
            # Simulate existing model config from agent_config.py
            model_config = {
                'model_id': 'us.anthropic.claude-3-haiku-20240307-v1:0',
                'temperature': 1.0,
                'region_name': 'us-east-1'
            }
            
            llm = LLMFactory.create_llm(model_config)
            
            # Verify Bedrock is called with original model ID
            mock_bedrock.assert_called_once()
            call_args = mock_bedrock.call_args[1]
            assert call_args['model_id'] == 'us.anthropic.claude-3-haiku-20240307-v1:0'
            assert call_args['temperature'] == 1.0
            assert call_args['region_name'] == 'us-east-1'
