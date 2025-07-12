# tests/test_document_generator.py
import pytest
from unittest.mock import MagicMock
from document_generator import DocumentGenerator, find_relevant_experiences
from api_clients import GeminiClient

@pytest.fixture
def mock_gemini_client():
    """Fixture for a mocked GeminiClient."""
    client = MagicMock(spec=GeminiClient)
    client.generate_text.return_value = "Mocked AI Content"
    return client

@pytest.fixture
def doc_generator(mock_gemini_client):
    """Fixture for a DocumentGenerator with a mocked client."""
    return DocumentGenerator(gemini_client=mock_gemini_client)

@pytest.fixture
def sample_experiences():
    """Fixture for sample career experiences."""
    return [
        {"id": 1, "title": "Software Engineer", "company": "Tech Corp", "dates": "2020-2022", "situation": "A", "task": "B", "action": "C", "result": "D", "resume_bullets": "Did a thing"},
        {"id": 2, "title": "Data Scientist", "company": "Data Inc.", "dates": "2018-2020", "situation": "E", "task": "F", "action": "G", "result": "H", "resume_bullets": "Analyzed data"},
        {"id": 3, "title": "Project Manager", "company": "Biz LLC", "dates": "2016-2018", "situation": "I", "task": "J", "action": "K", "result": "L", "resume_bullets": "Managed project"},
    ]

@pytest.fixture
def sample_user_profile():
    """Fixture for a sample user profile."""
    return {
        "full_name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "111-222-3333",
        "address": "123 Main St, Anytown",
        "linkedin_url": "linkedin.com/in/johndoe",
        "professional_summary": "A summary.",
        "style_profile": "Professional and concise."
    }

# --- find_relevant_experiences Tests ---

def test_find_relevant_experiences(sample_experiences):
    """Test finding relevant experiences based on a query."""
    query = "software development"
    relevant = find_relevant_experiences(query, sample_experiences, top_k=1)
    assert len(relevant) == 1
    assert relevant[0]['title'] == "Software Engineer"

def test_find_relevant_experiences_no_match(sample_experiences):
    """Test that an unrelated query returns the least dissimilar items."""
    # This test is based on the idea that even with no good match, it will return the top_k anyway
    query = "cooking and baking"
    relevant = find_relevant_experiences(query, sample_experiences, top_k=2)
    assert len(relevant) == 2

def test_find_relevant_experiences_empty_input():
    """Test empty inputs for find_relevant_experiences."""
    assert find_relevant_experiences("query", []) == []
    assert find_relevant_experiences("", [{"title": "test"}]) == []

# --- DocumentGenerator Tests ---

def test_generate_resume_markdown(doc_generator, sample_user_profile, sample_experiences):
    """Test the generation of resume Markdown."""
    markdown = doc_generator.generate_resume_markdown(sample_user_profile, sample_experiences)
    assert "# John Doe" in markdown
    assert "## PROFESSIONAL EXPERIENCE" in markdown
    assert "### Software Engineer" in markdown
    assert "- Did a thing" in markdown

def test_generate_cover_letter_markdown(doc_generator, sample_user_profile, sample_experiences):
    """Test the generation of a cover letter."""
    job_details = {"full_text": "Looking for a software engineer"}
    company_intel = {"values_mission": "We value coding."}

    doc_generator.generate_cover_letter_markdown(sample_user_profile, sample_experiences, job_details, company_intel)

    # Check that the prompt passed to the AI contains the right elements
    prompt = doc_generator.gemini_client.generate_text.call_args[0][0]
    assert "John Doe" not in prompt # The prompt should not contain the user's name directly
    assert "in my role as a Software Engineer" in prompt
    assert "We value coding." in prompt

def test_generate_ksc_response(doc_generator, sample_user_profile, sample_experiences):
    """Test the generation of a KSC response."""
    ksc_question = "Describe your project management skills."
    company_intel = {"values_mission": "We value organization."}

    result = doc_generator.generate_ksc_response(ksc_question, sample_user_profile, sample_experiences, company_intel, "PM Role")

    # Check the generated content and the prompt
    assert result['html'] == "Mocked AI Content"
    prompt = doc_generator.gemini_client.generate_text.call_args[0][0]
    assert "Describe your project management skills." in prompt
    assert "Title: Project Manager" in prompt # Should find the most relevant experience
    assert "We value organization." in prompt

def test_score_resume(doc_generator):
    """Test the resume scoring functionality."""
    doc_generator.gemini_client.generate_text.return_value = '{"match_score": 85, "strengths": ["Good skills"], "suggestions": ["Add more metrics"]}'

    score = doc_generator.score_resume("My resume text", "Job description text")

    assert score['match_score'] == 85
    assert "Good skills" in score['strengths']

    prompt = doc_generator.gemini_client.generate_text.call_args[0][0]
    assert "My resume text" in prompt
    assert "Job description text" in prompt

def test_score_resume_json_error(doc_generator):
    """Test handling of invalid JSON from the AI during resume scoring."""
    doc_generator.gemini_client.generate_text.return_value = "This is not JSON."

    result = doc_generator.score_resume("resume", "job desc")

    assert "error" in result
    assert result['raw_text'] == "This is not JSON."

# --- Document Creation Tests ---

def test_create_docx_from_markdown(doc_generator):
    """Test creating a DOCX file from Markdown."""
    markdown_content = "# Title\n\n- Bullet point\n\nJust text."
    docx_bytes = doc_generator._create_docx_from_markdown(markdown_content)

    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 0
    # A more robust test would be to open the DOCX and check contents, but that's complex.
    # For now, we just check that it produces a non-empty file.

def test_create_pdf_from_markdown(doc_generator):
    """Test creating a PDF file from Markdown."""
    markdown_content = "# PDF Title\n\nThis is a test."
    pdf_bytes = doc_generator._create_pdf_from_markdown(markdown_content)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b'%PDF-')
