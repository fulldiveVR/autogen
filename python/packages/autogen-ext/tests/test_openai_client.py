import logging
import os
from unittest.mock import MagicMock, patch
from openai import AsyncOpenAI

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily, ModelInfo, UserMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dummy API key for testing
DUMMY_API_KEY = "sk-dummy-test-key"

def test_known_model_default_configuration():
    """Test configuration for a known OpenAI model."""
    client_config = {
        "model": "gpt-4",
        "api_key": DUMMY_API_KEY
    }
    
    logger.info(f"Testing configuration for known model: {client_config}")
    
    # Mock AsyncOpenAI client creation
    with patch('openai.AsyncOpenAI') as MockAsyncOpenAI:
        mock_client = MagicMock(spec=AsyncOpenAI)
        MockAsyncOpenAI.return_value = mock_client
        
        client = OpenAIChatCompletionClient(**client_config)
        
        # Verify model info is correctly set
        logger.info(f"Model Info: {client._model_info}")
        assert client._model_info['vision'] == False
        assert client._model_info['function_calling'] == True
        assert client._model_info['json_output'] == True
        assert client._model_info['family'] == ModelFamily.GPT_4

def test_openrouter_model():
    """Test configuration for an OpenRouterAI model."""
    # OpenRouter configuration example
    client_config = {
        "model": "anthropic/claude-2.1",  # Example OpenRouter model
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY", "test-openrouter-key"),
        # Explicitly set capabilities for the model
        "vision_support": False,
        "function_calling_support": True,
        "json_output_support": True,
        "model_family": "anthropic"
    }
    
    logger.info(f"Testing OpenRouter model configuration: {client_config}")
    
    # Detailed mocking of AsyncOpenAI client
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()
    
    # Create a nested mock structure that mimics AsyncOpenAI's structure
    mock_client.chat = mock_chat
    mock_client.chat.completions = mock_completions
    
    # Simulate a create method response
    mock_completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test response"))]
    )
    
    # Patch AsyncOpenAI to return our mock client
    with patch('openai.AsyncOpenAI', return_value=mock_client):
        client = OpenAIChatCompletionClient(**client_config)
        
        # Verify model info reflects OpenRouter model capabilities
        logger.info(f"OpenRouter Model Info: {client._model_info}")
        assert client._model_info['vision'] == False
        assert client._model_info['function_calling'] == True
        assert client._model_info['json_output'] == True
        assert client._model_info['family'] == "anthropic"
        
        # Optional: Test actual message creation (mocked)
        try:
            result = client.create([UserMessage(content="Test message", source="user")])
            logger.info(f"Mocked create result: {result}")
        except Exception as e:
            logger.error(f"Error in create method: {e}")
            raise

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
    
    logger.info(f"Testing custom model configuration: {client_config}")
    
    # Mock AsyncOpenAI client creation
    with patch('openai.AsyncOpenAI') as MockAsyncOpenAI:
        mock_client = MagicMock(spec=AsyncOpenAI)
        MockAsyncOpenAI.return_value = mock_client
        
        client = OpenAIChatCompletionClient(**client_config)
        
        # Verify model info reflects explicit capabilities
        logger.info(f"Custom Model Info: {client._model_info}")
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
    
    logger.info(f"Testing model info override: {client_config}")
    
    # Mock AsyncOpenAI client creation
    with patch('openai.AsyncOpenAI') as MockAsyncOpenAI:
        mock_client = MagicMock(spec=AsyncOpenAI)
        MockAsyncOpenAI.return_value = mock_client
        
        client = OpenAIChatCompletionClient(**client_config)
        
        # Verify model info reflects the custom model info
        logger.info(f"Overridden Model Info: {client._model_info}")
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
    
    logger.info(f"Testing partial capability override: {client_config}")
    
    # Mock AsyncOpenAI client creation
    with patch('openai.AsyncOpenAI') as MockAsyncOpenAI:
        mock_client = MagicMock(spec=AsyncOpenAI)
        MockAsyncOpenAI.return_value = mock_client
        
        client = OpenAIChatCompletionClient(**client_config)
        
        # Verify only vision is overridden
        logger.info(f"Partially Overridden Model Info: {client._model_info}")
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
    
    logger.info(f"Testing model resolution for known and unknown models")
    
    # Mock AsyncOpenAI client creation
    with patch('openai.AsyncOpenAI') as MockAsyncOpenAI:
        mock_client = MagicMock(spec=AsyncOpenAI)
        MockAsyncOpenAI.return_value = mock_client
        
        known_client = OpenAIChatCompletionClient(**known_model_config)
        custom_client = OpenAIChatCompletionClient(**custom_model_config)
        
        # Verify resolved model names
        logger.info(f"Resolved model names: known={known_client._resolved_model}, custom={custom_client._resolved_model}")
        assert known_client._resolved_model == "gpt-4-0613"  # Known model resolution
        assert custom_client._resolved_model == "custom-model"  # Fallback to original model name