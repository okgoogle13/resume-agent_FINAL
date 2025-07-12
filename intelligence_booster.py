# intelligence_booster.py
"""
Module to enrich context by gathering live intelligence about a company.
This version is enhanced with a Semantic Cache to avoid redundant API calls
for similar queries, saving costs and improving response time.
"""
from api_clients import PerplexityClient
from typing import Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the embedding model once when the module is loaded.
# This model is lightweight and efficient for creating vector embeddings of text.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class SemanticCache:
    """
    A cache that stores and retrieves data based on semantic similarity of keys.
    Instead of exact string matching, it uses vector embeddings to find queries
    that are conceptually similar.
    """
    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initializes the cache.
        
        Args:
            similarity_threshold: The cosine similarity score required to consider
                                  two queries a match. Defaults to 0.95.
        """
        self._cache: Dict[str, Dict] = {} # Stores original query string and its result
        self._embeddings: Dict[str, np.ndarray] = {} # Stores query string and its vector embedding
        self.similarity_threshold = similarity_threshold

    def get(self, query: str) -> Optional[Dict]:
        """
        Tries to retrieve a result from the cache based on semantic similarity.

        Args:
            query: The new query string.

        Returns:
            The cached result if a similar query is found, otherwise None.
        """
        if not self._cache:
            return None

        query_embedding = embedding_model.encode([query]).reshape(1, -1)
        
        # Retrieve all cached embeddings and their corresponding keys
        cached_keys = list(self._embeddings.keys())
        cached_embeddings = np.array([self._embeddings[key] for key in cached_keys])

        # Calculate cosine similarities between the new query and all cached queries
        similarities = cosine_similarity(query_embedding, cached_embeddings)[0]

        # Find the highest similarity score
        max_similarity_idx = np.argmax(similarities)
        max_similarity_score = similarities[max_similarity_idx]

        if max_similarity_score >= self.similarity_threshold:
            matched_key = cached_keys[max_similarity_idx]
            print(f"Cache HIT: Found similar query with score {max_similarity_score:.2f}. Returning cached result.")
            return self._cache[matched_key]
        
        print("Cache MISS: No sufficiently similar query found in cache.")
        return None

    def set(self, query: str, result: Dict):
        """
        Adds a new query and its result to the cache.

        Args:
            query: The query string to cache.
            result: The result from the API call to cache.
        """
        query_embedding = embedding_model.encode([query]).reshape(1, -1)
        self._cache[query] = result
        self._embeddings[query] = query_embedding[0] # Store the 1D array
        print(f"Cache SET: Added new query to cache.")


class QueryGenerator:
    """Generates targeted questions for the Perplexity API."""
    def generate_for_company(self, company_name: str, role_title: str) -> Dict[str, str]:
        return {
            "values_mission": f"What are the publicly stated values, mission, or vision of the organization '{company_name}'?",
            "recent_news": f"Summarize recent news, projects, or developments for '{company_name}' in the last 6 months.",
            "role_context": f"What are the typical challenges or objectives for a '{role_title}' within the community services or social work sector in Australia?"
        }

class IntelligenceBoosterModule:
    """Facade for the Company Researcher, orchestrating queries and caching."""
    def __init__(self, perplexity_client: PerplexityClient):
        self.client = perplexity_client
        self.query_generator = QueryGenerator()
        # Use the new SemanticCache instead of TTLCache
        self.cache = SemanticCache()

    def get_intelligence(self, company_name: str, role_title: str) -> Dict[str, str]:
        """Main method to fetch and structure company intelligence using semantic caching."""
        queries = self.query_generator.generate_for_company(company_name, role_title)
        intelligence = {}

        for key, query in queries.items():
            # 1. Check the cache first
            cached_result = self.cache.get(query)
            if cached_result:
                intelligence[key] = cached_result['content']
                continue

            # 2. If not in cache, call the API
            try:
                print(f"API CALL: Calling Perplexity for query: '{query}'")
                response = self.client.search(query)
                content = response['choices'][0]['message']['content']
                intelligence[key] = content
                
                # 3. Store the new result in the cache
                self.cache.set(query, {'content': content})

            except Exception as e:
                error_message = f"Error fetching data: {e}"
                print(f"Could not fetch intelligence for query '{query}': {e}")
                intelligence[key] = error_message

        return intelligence
