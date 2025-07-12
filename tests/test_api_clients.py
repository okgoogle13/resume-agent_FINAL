# tests/test_api_clients.py
import pytest
from unittest.mock import patch, MagicMock
from api_clients import GeminiClient, PerplexityClient
import httpx
import tenacity

# --- GeminiClient Tests ---

def test_gemini_client_initialization_success():
    """Test successful initialization of GeminiClient."""
    client = GeminiClient(api_key="fake_key")
    assert client is not None

def test_gemini_client_initialization_no_key():
    """Test GeminiClient initialization with no API key."""
    with pytest.raises(ValueError, match="Gemini API key is required."):
        GeminiClient(api_key=None)

@patch('google.generativeai.GenerativeModel.generate_content')
def test_gemini_generate_text_success(mock_generate_content):
    """Test successful text generation with GeminiClient."""
    mock_response = MagicMock()
    mock_response.text = "Generated text"
    mock_generate_content.return_value = mock_response

    client = GeminiClient(api_key="fake_key")
    result = client.generate_text("test prompt")

    assert result == "Generated text"
    mock_generate_content.assert_called_once_with("test prompt")

@patch('google.generativeai.GenerativeModel.generate_content')
def test_gemini_generate_text_retry(mock_generate_content):
    """Test the retry logic of GeminiClient's generate_text."""
    mock_generate_content.side_effect = [Exception("API Error"), Exception("API Error"), MagicMock(text="Success")]

    client = GeminiClient(api_key="fake_key")
    # This test is tricky because the mock is not configured to have a `text` attribute on the final call
    # Let's adjust the mock to handle this
    final_response = MagicMock()
    final_response.text = "Success"
    mock_generate_content.side_effect = [Exception("API Error"), Exception("API Error"), final_response]

    result = client.generate_text("test prompt")
    assert result == "Success"
    assert mock_generate_content.call_count == 3


# --- PerplexityClient Tests ---

def test_perplexity_client_initialization_success():
    """Test successful initialization of PerplexityClient."""
    client = PerplexityClient(api_key="fake_key")
    assert client is not None

def test_perplexity_client_initialization_no_key():
    """Test PerplexityClient initialization with no API key."""
    with pytest.raises(ValueError, match="Perplexity API key is required."):
        PerplexityClient(api_key=None)

@patch('httpx.Client.post')
def test_perplexity_search_success(mock_post):
    """Test successful search with PerplexityClient."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": "some data"}
    mock_post.return_value = mock_response

    client = PerplexityClient(api_key="fake_key")
    result = client.search("test query")

    assert result == {"result": "some data"}
    mock_post.assert_called_once()

@patch('httpx.Client.post')
def test_perplexity_search_retry(mock_post):
    """Test the retry logic of PerplexityClient's search."""
    mock_post.side_effect = [httpx.RequestError("Network Error"), httpx.RequestError("Network Error"), MagicMock(status_code=200, json=lambda: {"result": "Success"})]

    client = PerplexityClient(api_key="fake_key")
    result = client.search("test query")

    assert result == {"result": "Success"}
    assert mock_post.call_count == 3

@patch('httpx.Client.post')
def test_perplexity_search_http_error(mock_post):
    """Test that an HTTP error from Perplexity raises an exception."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("Error", request=MagicMock(), response=MagicMock(status_code=500, text="Internal Server Error"))
    mock_post.return_value = mock_response

    client = PerplexityClient(api_key="fake_key")
    with pytest.raises(tenacity.RetryError):
        client.search("test query")
