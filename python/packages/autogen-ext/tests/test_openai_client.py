from unittest.mock import MagicMock, patch
from openai import AsyncOpenAI

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily, ModelInfo

# Dummy API key for testing
DUMMY_API_KEY = "sk-dummy-test-key"

def test_known_model_default_configuration():
    """Test configuration for a known OpenAI model."""
    client_config = {
        "model": "gpt-4",
        "api_key": DUMMY_API_KEY
    }
    
    # Mock AsyncOpenAI client creation
    with patch('openai.AsyncOpenAI') as MockAsyncOpenAI:
        mock_client = MagicMock(spec=AsyncOpenAI)
        MockAsyncOpenAI.return_value = mock_client
        
        client = OpenAIChatCompletionClient(**client_config)
        
        # Verify model info is correctly set
        assert client._model_info['vision'] == False
        assert client._model_info['function_calling'] == True
        assert client._model_info['json_output'] == True
        assert client._model_info['family'] == ModelFamily.GPT_4

def test_custom_model_with_explicit_capabilities():
    """Test configuration for a custom model with explicit capabilities."""
    client_config = {
        "model": "custom-model",
        "base_url": "https://custom-endpoint.com/v1",
        "api_key": DUMMY_API_KEY,
        "vision_support": True,
        "function_calling_support": True,
        "json_output_support": False
    }
    
    # Mock AsyncOpenAI client creation
    with patch('openai.AsyncOpenAI') as MockAsyncOpenAI:
        mock_client = MagicMock(spec=AsyncOpenAI)
        MockAsyncOpenAI.return_value = mock_client
        
        client = OpenAIChatCompletionClient(**client_config)
        
        # Verify model info reflects explicit capabilities
        assert client._model_info['vision'] == True
        assert client._model_info['function_calling'] == True
        assert client._model_info['json_output'] == False
        assert client._model_info['family'] == ModelFamily.UNKNOWN

def test_model_info_override():
    """Test overriding model info with a custom ModelInfo."""
    custom_model_info = ModelInfo(
        vision=True,
        function_calling=False,
        json_output=True,
        family=ModelFamily.GPT_4O
    )
    
    client_config = {
        "model": "gpt-4",
        "api_key": DUMMY_API_KEY,
        "model_info": custom_model_info
    }
    
    # Mock AsyncOpenAI client creation
    with patch('openai.AsyncOpenAI') as MockAsyncOpenAI:
        mock_client = MagicMock(spec=AsyncOpenAI)
        MockAsyncOpenAI.return_value = mock_client
        
        client = OpenAIChatCompletionClient(**client_config)
        
        # Verify model info reflects the custom model info
        assert client._model_info['vision'] == True
        assert client._model_info['function_calling'] == False
        assert client._model_info['json_output'] == True
        assert client._model_info['family'] == ModelFamily.GPT_4O

def test_partial_capability_override():
    """Test partially overriding model capabilities."""
    client_config = {
        "model": "gpt-4",
        "api_key": DUMMY_API_KEY,
        "vision_support": True
    }
    
    # Mock AsyncOpenAI client creation
    with patch('openai.AsyncOpenAI') as MockAsyncOpenAI:
        mock_client = MagicMock(spec=AsyncOpenAI)
        MockAsyncOpenAI.return_value = mock_client
        
        client = OpenAIChatCompletionClient(**client_config)
        
        # Verify only vision is overridden
        assert client._model_info['vision'] == True
        assert client._model_info['function_calling'] == True
        assert client._model_info['json_output'] == True
        assert client._model_info['family'] == ModelFamily.GPT_4

def test_model_resolution():
    """Test model name resolution for known and unknown models."""
    known_model_config = {
        "model": "gpt-4",
        "api_key": DUMMY_API_KEY
    }
    
    custom_model_config = {
        "model": "custom-model",
        "base_url": "https://custom-endpoint.com/v1",
        "api_key": DUMMY_API_KEY
    }
    
    # Mock AsyncOpenAI client creation
    with patch('openai.AsyncOpenAI') as MockAsyncOpenAI:
        mock_client = MagicMock(spec=AsyncOpenAI)
        MockAsyncOpenAI.return_value = mock_client
        
        known_client = OpenAIChatCompletionClient(**known_model_config)
        custom_client = OpenAIChatCompletionClient(**custom_model_config)
        
        # Verify resolved model names
        assert known_client._resolved_model == "gpt-4-0613"  # Known model resolution
        assert custom_client._resolved_model == "custom-model"  # Fallback to original model name