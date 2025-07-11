import httpx
from bs4 import BeautifulSoup
import streamlit as st # For potential error messages if needed directly, though better to return errors

# Common selectors - this list can be expanded.
# Prioritize more specific selectors if known for common job boards.
COMMON_JOB_DESC_SELECTORS = [
    {'tag': 'div', 'attrs': {'class': 'job-description'}},
    {'tag': 'div', 'attrs': {'id': 'job-details'}},
    {'tag': 'div', 'attrs': {'class': 'job-details-content'}},
    {'tag': 'div', 'attrs': {'class': 'jobsearch-JobComponent-description'}}, # Indeed
    {'tag': 'article', 'attrs': {}}, # General article tag
    {'tag': 'div', 'attrs': {'role': 'document'}}, # Some modern sites
    # LinkedIn specific (often dynamic, might need Playwright/Selenium for robust LinkedIn)
    {'tag': 'div', 'attrs': {'class': 'description__text'}},
    {'tag': 'section', 'attrs': {'class': 'description'}},
]

def fetch_page_content(url: str) -> str | None:
    """
    Fetches HTML content from a given URL.
    Returns HTML content as a string, or None if an error occurs.
    """
    try:
        # Add a common user-agent to avoid simple blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            return response.text
    except httpx.HTTPStatusError as e:
        st.warning(f"HTTP error fetching URL '{url}': {e.response.status_code} {e.response.reason_phrase}")
        return None
    except httpx.RequestError as e:
        st.warning(f"Request error fetching URL '{url}': {e}")
        return None
    except Exception as e:
        st.warning(f"An unexpected error occurred while fetching URL '{url}': {e}")
        return None

def extract_job_description_from_html(html_content: str, url: str) -> str:
    """
    Extracts job description text from HTML content using common selectors.
    This is a best-effort extraction and might not work for all websites.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, 'html.parser')

    extracted_text = ""

    # Try common selectors
    for selector_info in COMMON_JOB_DESC_SELECTORS:
        tag = selector_info['tag']
        attrs = selector_info['attrs']
        # find_all can also be used if multiple sections are expected and need merging
        element = soup.find(tag, attrs=attrs)
        if element:
            # Basic cleaning: get_text with separator, strip leading/trailing whitespace
            text_content = element.get_text(separator='\n', strip=True)

            # Further cleaning (optional, can be expanded)
            # Remove excessive blank lines
            lines = [line.strip() for line in text_content.splitlines() if line.strip()]
            cleaned_text = "\n".join(lines)

            # Heuristic: if the text is reasonably long, assume it's the main content.
            # This helps avoid picking up small, irrelevant divs.
            if len(cleaned_text) > 300: # Arbitrary length threshold
                extracted_text = cleaned_text
                # st.info(f"Found content using: {tag} with {attrs}") # For debugging
                break # Found a good candidate

    if not extracted_text:
        # Fallback: if no specific selectors work, try to get text from the main body
        # This is very broad and might include headers, footers, ads.
        # Often, the main content is within an <article> or a <main> tag.
        main_content_tags = soup.find('main') or soup.find('article') or soup.body
        if main_content_tags:
            text_content = main_content_tags.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text_content.splitlines() if line.strip()]
            cleaned_text = "\n".join(lines)

            # Only use this broad fallback if it's substantial, to avoid noise
            if len(cleaned_text) > 500: # Higher threshold for broad fallback
                 # st.info("Used broad fallback (main/article/body)") # For debugging
                 extracted_text = cleaned_text
            else:
                # st.warning(f"Could not find a substantial job description block on {url} using common selectors or broad fallback.")
                pass # Keep extracted_text empty or with a very short message if nothing good found

    if not extracted_text:
       return f"[Could not automatically extract detailed job description from {url}. You may need to copy/paste manually if the generated documents are not adequate.]"

    return extracted_text

if __name__ == '__main__':
    # Basic test
    # test_url = "https://www.linkedin.com/jobs/view/software-engineer-at-linkedin-123456789/" # LinkedIn is hard to scrape directly
    # test_url = "https://www.indeed.com/viewjob?jk=...." # Replace with a real Indeed job URL
    # test_url = "https://jobs.careers.google.com/jobs/results/..." # Replace with a real Google job URL

    # Example (replace with a URL you can test, preferably one that's not heavily JS-reliant for critical content)
    # Note: Most major job boards are heavily JS-reliant and may not work well with simple httpx/BeautifulSoup.
    # This basic scraper is more likely to work on simpler, static HTML pages or company career pages.

    test_urls = [
        "https://www.google.com", # Just to see if fetch works
        # Add a known, simple job posting URL here if you have one for testing
    ]

    for url_to_test in test_urls:
        print(f"Fetching: {url_to_test}")
        html = fetch_page_content(url_to_test)
        if html:
            print(f"Fetched {len(html)} bytes. Extracting job description...")
            jd = extract_job_description_from_html(html, url_to_test)
            print("--- Extracted JD ---")
            print(jd[:1000] + "..." if len(jd) > 1000 else jd)
            print("--------------------")
        else:
            print(f"Failed to fetch {url_to_test}")
        print("\n")
