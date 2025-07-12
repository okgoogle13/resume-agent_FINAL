# tests/test_intelligence_booster.py
import pytest
from unittest.mock import MagicMock, patch
from intelligence_booster import SemanticCache, QueryGenerator, IntelligenceBoosterModule
from api_clients import PerplexityClient

# --- SemanticCache Tests ---

def test_semantic_cache_miss():
    """Test that the cache returns None for a new query."""
    cache = SemanticCache()
    assert cache.get("new query") is None

def test_semantic_cache_set_and_get():
    """Test setting and retrieving an item with an exact match."""
    cache = SemanticCache()
    cache.set("query 1", {"data": "result 1"})
    result = cache.get("query 1")
    assert result is not None
    assert result["data"] == "result 1"

@patch('intelligence_booster.cosine_similarity')
def test_semantic_cache_semantic_hit(mock_cosine_similarity):
    """Test a cache hit with a semantically similar query."""
    mock_cosine_similarity.return_value = [[0.95]]
    cache = SemanticCache(similarity_threshold=0.9)
    cache.set("What is the company's mission?", {"data": "mission result"})
    result = cache.get("What is their mission statement?")
    assert result is not None
    assert result["data"] == "mission result"

def test_semantic_cache_semantic_miss():
    """Test a cache miss with a dissimilar query."""
    cache = SemanticCache(similarity_threshold=0.95)
    cache.set("What are the company values?", {"data": "values result"})
    result = cache.get("Tell me about their recent news.")
    assert result is None

# --- QueryGenerator Tests ---

def test_query_generator():
    """Test the generation of queries."""
    generator = QueryGenerator()
    queries = generator.generate_for_company("Future Inc.", "Developer")

    assert "values_mission" in queries
    assert "Future Inc." in queries["values_mission"]
    assert "recent_news" in queries
    assert "Developer" in queries["role_context"]

# --- IntelligenceBoosterModule Tests ---

@pytest.fixture
def mock_perplexity_client():
    """Fixture for a mocked PerplexityClient."""
    client = MagicMock(spec=PerplexityClient)

    # Simple mock response for any search query
    def mock_search(query):
        return {"choices": [{"message": {"content": f"Mock response for: {query}"}}]}

    client.search.side_effect = mock_search
    return client

@pytest.fixture
def intelligence_booster(mock_perplexity_client):
    """Fixture for an IntelligenceBoosterModule with a mocked client."""
    return IntelligenceBoosterModule(perplexity_client=mock_perplexity_client)

def test_get_intelligence_no_cache(intelligence_booster, mock_perplexity_client):
    """Test fetching intelligence when the cache is empty."""
    intel = intelligence_booster.get_intelligence("New Corp", "Analyst")

    # Should call the API for all generated queries
    assert mock_perplexity_client.search.call_count == 3
    assert "values_mission" in intel
    assert "Mock response for" in intel["values_mission"]

def test_get_intelligence_with_cache_hit(intelligence_booster, mock_perplexity_client):
    """Test that a cache hit prevents an API call."""
    # Pre-populate the cache
    query = "What are the publicly stated values, mission, or vision of the organization 'Old Corp'?"
    intelligence_booster.cache.set(query, {"content": "Cached values"})

    # This should result in one cache hit and two API calls
    intel = intelligence_booster.get_intelligence("Old Corp", "Manager")

    assert mock_perplexity_client.search.call_count == 2
    assert intel["values_mission"] == "Cached values"
    assert "Mock response for" in intel["recent_news"]

def test_get_intelligence_api_error(intelligence_booster, mock_perplexity_client):
    """Test how the module handles an API error."""
    mock_perplexity_client.search.side_effect = Exception("API Failure")

    intel = intelligence_booster.get_intelligence("Fail Corp", "Tester")

    assert "Error fetching data: API Failure" in intel["values_mission"]
    assert "Error fetching data: API Failure" in intel["recent_news"]
    assert "Error fetching data: API Failure" in intel["role_context"]
