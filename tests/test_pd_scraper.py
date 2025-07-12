# tests/test_pd_scraper.py
import pytest
from unittest.mock import patch, MagicMock
from pd_scraper import PDScraperModule
from api_clients import GeminiClient

@pytest.fixture
def mock_gemini_client():
    """Fixture for a mocked GeminiClient."""
    client = MagicMock(spec=GeminiClient)
    client.generate_text.return_value = '{"role_title": "Software Engineer", "company_name": "TestCo", "key_responsibilities": ["Coding"], "essential_skills": ["Python"]}'
    return client

@pytest.fixture
def scraper(mock_gemini_client):
    """Fixture for a PDScraperModule with a mocked client."""
    return PDScraperModule(gemini_client=mock_gemini_client)

@pytest.fixture
def mock_playwright():
    """Fixture to mock the entire Playwright process."""
    with patch('pd_scraper.sync_playwright') as mock_sync_playwright:
        mock_page = MagicMock()
        mock_page.content.return_value = "<html><body><h1>Job Title</h1><p>Description</p></body></html>"

        mock_browser = MagicMock()
        mock_browser.new_page.return_value = mock_page

        mock_playwright_manager = MagicMock()
        mock_playwright_manager.chromium.launch.return_value = mock_browser

        # This makes the `with sync_playwright() as p:` block work
        mock_sync_playwright.return_value.__enter__.return_value = mock_playwright_manager

        yield mock_sync_playwright

def test_extract_text_from_html():
    """Test the pure text extraction logic from HTML."""
    # This function is internal but crucial. We test it directly.
    scraper = PDScraperModule(gemini_client=MagicMock())
    html = """
    <html>
        <head><title>Ignore Me</title><style>p {color: red;}</style></head>
        <body>
            <header>Header stuff</header>
            <main>
                <h1>Job Title</h1>
                <p>This is the job description.</p>
                <script>alert('hello');</script>
            </main>
            <footer>Footer stuff</footer>
        </body>
    </html>
    """
    text = scraper._extract_text_from_html(html)
    assert text == "Job Title\nThis is the job description."

def test_process_url_success(scraper, mock_playwright):
    """Test the end-to-end process_url method with mocks."""
    url = "http://example.com/job"
    result = scraper.process_url(url)

    assert "full_text" in result
    assert "Job Title\nDescription" in result["full_text"]
    assert result["role_title"] == "Software Engineer"
    assert result["company_name"] == "TestCo"

    # Check that playwright was called
    mock_playwright.assert_called_once()

    # Check that Gemini was called
    scraper.gemini_client.generate_text.assert_called_once()

def test_process_url_playwright_error(scraper, mock_playwright):
    """Test how process_url handles a Playwright navigation error."""
    # Configure the mock to simulate a failure
    mock_playwright.return_value.__enter__.side_effect = Exception("Browser failed")

    url = "http://example.com/job"
    result = scraper.process_url(url)

    assert "error" in result
    assert "An unexpected error occurred" in result["error"]

def test_process_url_no_text_extracted(scraper, mock_playwright):
    """Test the case where no text can be extracted from the HTML."""
    mock_playwright.return_value.__enter__.return_value.chromium.launch.return_value.new_page.return_value.content.return_value = "<html><body></body></html>"

    url = "http://example.com/job"
    result = scraper.process_url(url)

    assert "error" in result
    assert "Could not extract any meaningful text" in result["error"]

def test_summarize_text_with_ai_error(scraper):
    """Test handling of an error during AI summarization."""
    scraper.gemini_client.generate_text.side_effect = Exception("AI API is down")

    # We call the internal method directly to isolate this test
    result = scraper._summarize_text_with_ai("Some text")

    assert "error" in result
    assert "AI summarization failed" in result["error"]
