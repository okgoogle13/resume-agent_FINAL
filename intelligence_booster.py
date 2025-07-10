# intelligence_booster.py
"""
Module to enrich context by gathering live intelligence about a company.
"""
from cachetools import TTLCache
from api_clients import PerplexityClient
from typing import Dict

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
        self.cache = TTLCache(maxsize=100, ttl=86400)

    def get_intelligence(self, company_name: str, role_title: str) -> Dict[str, str]:
        """Main method to fetch and structure company intelligence."""
        cache_key = f"{company_name}_{role_title}".lower()
        if cache_key in self.cache:
            return self.cache[cache_key]

        queries = self.query_generator.generate_for_company(company_name, role_title)
        intelligence = {}

        for key, query in queries.items():
            try:
                response = self.client.search(query)
                intelligence[key] = response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Could not fetch intelligence for query '{query}': {e}")
                intelligence[key] = f"Error fetching data: {e}"

        self.cache[cache_key] = intelligence
        return intelligence
