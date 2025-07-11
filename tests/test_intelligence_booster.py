# tests/test_intelligence_booster.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Assuming intelligence_booster.py is in the root directory
from intelligence_booster import QueryGenerator, SemanticCache, IntelligenceBoosterModule
from api_clients import PerplexityClient # Needed for type hinting if not directly instantiating

# --- Tests for QueryGenerator ---
def test_query_generator_generate_for_company():
    generator = QueryGenerator()
    company_name = "FutureTech"
    role_title = "AI Ethicist"
    queries = generator.generate_for_company(company_name, role_title)

    assert "values_mission" in queries
    assert "recent_news" in queries
    assert "role_context" in queries
    assert queries["values_mission"] == f"What are the publicly stated values, mission, or vision of the organization '{company_name}'?"
    assert queries["recent_news"] == f"Summarize recent news, projects, or developments for '{company_name}' in the last 6 months."
    assert queries["role_context"] == f"What are the typical challenges or objectives for a '{role_title}' within the community services or social work sector in Australia?"

# --- Tests for SemanticCache ---
@pytest.fixture
def mock_embedding_model():
    """Mocks the sentence_transformers.SentenceTransformer model."""
    mock_model = MagicMock()
    # Simple mock: return a unique embedding for each unique string
    # and the same embedding for the same string.
    # More sophisticated mocking might be needed for complex similarity scenarios.
    mock_model.encode = MagicMock(side_effect=lambda x: [np.array([hash(s) % 1000]) for s in x])
    return mock_model

def test_semantic_cache_set_and_get_miss(mock_embedding_model):
    """Test setting an item and then trying to get a dissimilar item."""
    with patch('intelligence_booster.embedding_model', mock_embedding_model):
        cache = SemanticCache(similarity_threshold=0.95)
        query1 = "What is the mission of Company A?"
        result1 = {"content": "Mission A"}
        cache.set(query1, result1)

        # Mock cosine_similarity to control the outcome
        with patch('intelligence_booster.cosine_similarity') as mock_cosine_sim:
            # Simulate a low similarity score for a different query
            mock_cosine_sim.return_value = np.array([[0.5]])

            query2 = "Tell me about Company B's vision."
            retrieved = cache.get(query2)
            assert retrieved is None
            mock_embedding_model.encode.assert_any_call([query1]) # Called during set
            mock_embedding_model.encode.assert_any_call([query2]) # Called during get
            mock_cosine_sim.assert_called_once()

def test_semantic_cache_set_and_get_hit(mock_embedding_model):
    """Test setting an item and then getting a similar item."""
    with patch('intelligence_booster.embedding_model', mock_embedding_model):
        cache = SemanticCache(similarity_threshold=0.95)
        query1 = "What are the core values of Innovate Corp?"
        result1 = {"content": "Values: Innovation, Integrity."}
        cache.set(query1, result1)

        # Mock cosine_similarity for a cache hit
        with patch('intelligence_booster.cosine_similarity') as mock_cosine_sim:
            mock_cosine_sim.return_value = np.array([[0.98]]) # High similarity

            query_similar = "Tell me about Innovate Corp's values." # Semantically similar
            retrieved = cache.get(query_similar)
            assert retrieved == result1
            mock_embedding_model.encode.assert_any_call([query1])
            mock_embedding_model.encode.assert_any_call([query_similar])
            mock_cosine_sim.assert_called_once()

def test_semantic_cache_empty_get(mock_embedding_model):
    with patch('intelligence_booster.embedding_model', mock_embedding_model):
        cache = SemanticCache()
        assert cache.get("Any query") is None

def test_semantic_cache_threshold(mock_embedding_model):
    with patch('intelligence_booster.embedding_model', mock_embedding_model):
        cache = SemanticCache(similarity_threshold=0.9)
        query1 = "Query X"
        result1 = {"content": "Result X"}
        cache.set(query1, result1)

        with patch('intelligence_booster.cosine_similarity') as mock_cosine_sim:
            # Test just below threshold
            mock_cosine_sim.return_value = np.array([[0.89]])
            assert cache.get("Similar Query Y") is None

            # Test just at/above threshold
            mock_cosine_sim.return_value = np.array([[0.90]])
            assert cache.get("Similar Query Z") == result1


# --- Tests for IntelligenceBoosterModule ---
@pytest.fixture
def mock_perplexity_client(mock_perplexity_api_key):
    """Mocks the PerplexityClient."""
    client = MagicMock(spec=PerplexityClient)
    # Ensure api_key attribute exists if accessed by PerplexityClient constructor if not fully mocked
    client.api_key = mock_perplexity_api_key
    return client

@pytest.fixture
def mock_semantic_cache_instance():
    """Mocks an instance of SemanticCache."""
    cache = MagicMock(spec=SemanticCache)
    return cache

def test_intelligence_booster_get_intelligence_cache_hit(mock_perplexity_client, mock_semantic_cache_instance):
    """Test get_intelligence when all items are found in cache."""
    booster = IntelligenceBoosterModule(perplexity_client=mock_perplexity_client)
    booster.cache = mock_semantic_cache_instance # Inject the mock cache

    company_name = "Cached Inc."
    role_title = "Cache Manager"

    # Setup mock cache to return values for all generated queries
    mock_semantic_cache_instance.get.side_effect = [
        {"content": "Cached values"},
        {"content": "Cached news"},
        {"content": "Cached role context"}
    ]

    intelligence = booster.get_intelligence(company_name, role_title)

    assert intelligence["values_mission"] == "Cached values"
    assert intelligence["recent_news"] == "Cached news"
    assert intelligence["role_context"] == "Cached role context"

    assert mock_semantic_cache_instance.get.call_count == 3
    mock_perplexity_client.search.assert_not_called() # API should not be called
    mock_semantic_cache_instance.set.assert_not_called() # Nothing new to set

def test_intelligence_booster_get_intelligence_cache_miss_api_call(mock_perplexity_client, mock_semantic_cache_instance):
    """Test get_intelligence with cache misses, triggering API calls and caching results."""
    booster = IntelligenceBoosterModule(perplexity_client=mock_perplexity_client)
    booster.cache = mock_semantic_cache_instance

    company_name = "FreshData Corp"
    role_title = "Data Analyst"

    # Simulate cache miss (get returns None)
    mock_semantic_cache_instance.get.return_value = None

    # Mock Perplexity API responses
    api_responses = {
        f"What are the publicly stated values, mission, or vision of the organization '{company_name}'?": {"choices": [{"message": {"content": "Mission: Freshness"}}]},
        f"Summarize recent news, projects, or developments for '{company_name}' in the last 6 months.": {"choices": [{"message": {"content": "News: Launched new product."}}]},
        f"What are the typical challenges or objectives for a '{role_title}' within the community services or social work sector in Australia?": {"choices": [{"message": {"content": "Challenges: Funding, outreach."}}]}
    }
    mock_perplexity_client.search.side_effect = lambda query: api_responses[query]

    intelligence = booster.get_intelligence(company_name, role_title)

    assert intelligence["values_mission"] == "Mission: Freshness"
    assert intelligence["recent_news"] == "News: Launched new product."
    assert intelligence["role_context"] == "Challenges: Funding, outreach."

    assert mock_semantic_cache_instance.get.call_count == 3
    assert mock_perplexity_client.search.call_count == 3
    assert mock_semantic_cache_instance.set.call_count == 3
    # Verify that results from API are cached
    mock_semantic_cache_instance.set.assert_any_call(
        f"What are the publicly stated values, mission, or vision of the organization '{company_name}'?",
        {'content': "Mission: Freshness"}
    )

def test_intelligence_booster_get_intelligence_api_error(mock_perplexity_client, mock_semantic_cache_instance):
    """Test get_intelligence when Perplexity API call fails."""
    booster = IntelligenceBoosterModule(perplexity_client=mock_perplexity_client)
    booster.cache = mock_semantic_cache_instance

    company_name = "ErrorProne LLC"
    role_title = "Risk Taker"

    mock_semantic_cache_instance.get.return_value = None # Cache miss
    mock_perplexity_client.search.side_effect = Exception("API Failure") # Simulate API error

    intelligence = booster.get_intelligence(company_name, role_title)

    # Expect error messages in the results
    assert "Error fetching data: API Failure" in intelligence["values_mission"]
    assert "Error fetching data: API Failure" in intelligence["recent_news"]
    assert "Error fetching data: API Failure" in intelligence["role_context"]

    assert mock_perplexity_client.search.call_count == 3
    mock_semantic_cache_instance.set.assert_not_called() # Should not cache errors in this way
                                                        # (or if it does, test that specific behavior)

def test_intelligence_booster_partial_cache_hit(mock_perplexity_client, mock_semantic_cache_instance):
    """Test get_intelligence with a mix of cache hits and misses."""
    booster = IntelligenceBoosterModule(perplexity_client=mock_perplexity_client)
    booster.cache = mock_semantic_cache_instance

    company_name = "Hybrid Systems"
    role_title = "Integrator"

    # First query hits cache, second misses, third hits
    mock_semantic_cache_instance.get.side_effect = [
        {"content": "Cached values_mission Hybrid"}, # HIT
        None,                                       # MISS
        {"content": "Cached role_context Hybrid"}    # HIT
    ]

    # API response for the cache miss
    query_for_news = f"Summarize recent news, projects, or developments for '{company_name}' in the last 6 months."
    api_response_news = {"choices": [{"message": {"content": "News: Hybrid merger complete."}}]}
    mock_perplexity_client.search.side_effect = lambda query: api_response_news if query == query_for_news else None


    intelligence = booster.get_intelligence(company_name, role_title)

    assert intelligence["values_mission"] == "Cached values_mission Hybrid"
    assert intelligence["recent_news"] == "News: Hybrid merger complete."
    assert intelligence["role_context"] == "Cached role_context Hybrid"

    assert mock_semantic_cache_instance.get.call_count == 3
    mock_perplexity_client.search.assert_called_once_with(query_for_news)
    mock_semantic_cache_instance.set.assert_called_once_with(query_for_news, {'content': "News: Hybrid merger complete."})

# Ensure the global embedding_model in intelligence_booster is patched for SemanticCache tests
# that don't use the fixture explicitly (though the above tests do for clarity)
@pytest.fixture(autouse=True)
def auto_patch_embedding_model(mock_embedding_model):
    with patch('intelligence_booster.embedding_model', mock_embedding_model):
        yield
