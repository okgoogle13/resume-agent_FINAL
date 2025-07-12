# api_clients.py
"""
Centralized clients for interacting with external APIs like Google Gemini and Perplexity.
Includes robust error handling and automatic retries.
"""
import google.generativeai as genai
import httpx
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

class GeminiClient:
    """Client for interacting with the Google Gemini API."""
    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_text(self, prompt: str) -> str:
        """Generates text content from a given prompt with retry logic."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise

class PerplexityClient:
    """Client for interacting with the Perplexity Sonar API."""
    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            raise ValueError("Perplexity API key is required.")
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def search(self, query: str) -> Dict[str, Any]:
        """Performs a search using the Perplexity API with retry logic."""
        payload = {
            "model": "sonar-small-online",
            "messages": [{"role": "user", "content": query}]
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting from Perplexity: {e}")
            raise
