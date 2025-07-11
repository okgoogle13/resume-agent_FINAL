# pd_scraper.py
"""
A fully functional module to scrape job description content from a given URL.
It uses Playwright for robust browser automation and BeautifulSoup for HTML parsing.
"""
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from api_clients import GeminiClient
import json

class PDScraperModule:
    def __init__(self, gemini_client: GeminiClient):
        """Initializes the scraper module and the Gemini client."""
        self.gemini_client = gemini_client

    async def _get_page_content(self, url: str) -> str:
        """
        Uses Playwright to navigate to a URL and return its HTML content.
        This can handle dynamically loaded content (JavaScript).
        """
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=15000)
                content = await page.content()
                await browser.close()
                return content
        except Exception as e:
            print(f"Error with Playwright navigation: {e}")
            return f"<html><body>Error fetching page: {e}</body></html>"

    def _extract_text_from_html(self, html: str) -> str:
        """
        Uses BeautifulSoup to parse HTML and extract clean, readable text.
        It focuses on common tags where job descriptions are found.
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # Prioritize common content tags
        for tag in ['main', 'article', '[role="main"]']:
            if soup.select_one(tag):
                soup = soup.select_one(tag)
                break
        
        # Get text and clean it up
        text = soup.get_text(separator='\n', strip=True)
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)

    def _summarize_text_with_ai(self, text: str) -> dict:
        """Uses Gemini to summarize the scraped text into a structured JSON format."""
        prompt = f"""
        **Task:** Analyze the following job description text and extract key information.

        **Job Description Text:**
        ---
        {text[:8000]} 
        ---

        **Output Format:**
        Return a single, valid JSON object only. Do not include any other text.
        {{
            "role_title": "...",
            "company_name": "...",
            "key_responsibilities": ["...", "..."],
            "essential_skills": ["...", "..."]
        }}
        """
        try:
            response = self.gemini_client.generate_text(prompt)
            json_str = response.strip().replace("```json", "").replace("```", "")
            return json.loads(json_str)
        except Exception as e:
            print(f"AI summarization failed: {e}")
            return {"error": f"AI summarization failed: {e}"}

    def process_url(self, url: str) -> dict:
        """
        Orchestrates the scraping and summarization process.
        This is an async-to-sync wrapper.
        """
        try:
            return asyncio.run(self._async_process_url(url))
        except Exception as e:
            return {"error": f"Failed to process URL: {e}"}

    async def _async_process_url(self, url: str) -> dict:
        """The core async processing function."""
        print(f"Starting to scrape URL: {url}")
        html_content = await self._get_page_content(url)
        
        if "Error fetching page" in html_content:
            return {"error": "Could not retrieve page content."}
            
        full_text = self._extract_text_from_html(html_content)
        
        if not full_text:
            return {"error": "Could not extract any meaningful text from the URL."}
            
        result = {"full_text": full_text}
        
        print("Scraping complete. Summarizing with AI...")
        summary = self._summarize_text_with_ai(full_text)
        
        result.update(summary)
        print(f"Processing complete for {url}")
        return result

