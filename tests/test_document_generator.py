# tests/test_document_generator.py
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import json

# Assuming document_generator.py is in the root directory
from document_generator import DocumentGenerator, find_relevant_experiences
from api_clients import GeminiClient # For type hinting

# --- Tests for find_relevant_experiences ---

@pytest.fixture
def mock_embedding_model_dg(): # Suffix dg for Document Generator
    """Mocks the sentence_transformers.SentenceTransformer model for document_generator."""
    mock_model = MagicMock()
    # Predefined embeddings for specific texts
    # This allows for predictable similarity scores
    # For simplicity, using 1D arrays. Real embeddings are higher dimensional.
    predefined_embeddings = {
        "project management experience": np.array([1, 0, 0]),
        "managed a team for Project X": np.array([0.9, 0.1, 0]), # Similar to "project management"
        "developed a new software": np.array([0, 1, 0]), # Different
        "led customer support team": np.array([0, 0, 1]) # Different
    }

    def encode_side_effect(texts):
        return [predefined_embeddings.get(text, np.random.rand(3)) for text in texts]

    mock_model.encode = MagicMock(side_effect=encode_side_effect)
    return mock_model

@pytest.fixture
def sample_experiences_for_search():
    return [
        {"id": 1, "title": "Project Lead", "situation": "S1", "task": "Managed a team for Project X", "action": "A1", "result": "R1"},
        {"id": 2, "title": "Developer", "situation": "S2", "task": "Developed a new software", "action": "A2", "result": "R2"},
        {"id": 3, "title": "Support Manager", "situation": "S3", "task": "Led customer support team", "action": "A3", "result": "R3"},
    ]

def test_find_relevant_experiences_selects_top_k(mock_embedding_model_dg, sample_experiences_for_search):
    """Test that correct top_k experiences are returned based on mocked similarity."""
    with patch('document_generator.embedding_model', mock_embedding_model_dg):
        # Mock cosine_similarity to return predictable scores
        # query "project management experience"
        # exp1 "Project Lead. Managed a team for Project X" -> high similarity
        # exp2 "Developer. Developed a new software" -> low similarity
        # exp3 "Support Manager. Led customer support team" -> low similarity

        # Mocking the actual cosine_similarity calculation based on predefined embeddings:
        # Query embedding: [1,0,0]
        # Exp1 text: "Project Lead. S1 Managed a team for Project X A1 R1" -> let's assume its combined embedding is close to [0.9, 0.1, 0]
        # Exp2 text: "Developer. S2 Developed a new software A2 R2" -> close to [0,1,0]
        # Exp3 text: "Support Manager. S3 Led customer support team A3 R3" -> close to [0,0,1]

        # (q_emb) (exp_emb_matrix.T)
        # [1,0,0] * [[0.9, 0, 0], [0.1, 1, 0], [0, 0, 1]] = [0.9, 0, 0] -> Sum these for actual similarity
        # This is a simplification. Actual cosine similarity is more nuanced.
        # We will mock the output of cosine_similarity directly for better control.

        with patch('document_generator.cosine_similarity') as mock_cosine_sim:
            # query_embedding vs [exp1_embedding, exp2_embedding, exp3_embedding]
            mock_cosine_sim.return_value = np.array([[0.9, 0.2, 0.1]]) # Exp1 is most similar

            query = "project management experience"
            relevant_experiences = find_relevant_experiences(query, sample_experiences_for_search, top_k=1)
            assert len(relevant_experiences) == 1
            assert relevant_experiences[0]["id"] == 1 # Project Lead

            mock_cosine_sim.return_value = np.array([[0.9, 0.8, 0.1]]) # Exp1 and Exp2 are most similar
            relevant_experiences_top2 = find_relevant_experiences(query, sample_experiences_for_search, top_k=2)
            assert len(relevant_experiences_top2) == 2
            assert relevant_experiences_top2[0]["id"] == 1 # Project Lead (score 0.9)
            assert relevant_experiences_top2[1]["id"] == 2 # Developer (score 0.8) - order matters

def test_find_relevant_experiences_empty_inputs(mock_embedding_model_dg):
    with patch('document_generator.embedding_model', mock_embedding_model_dg):
        assert find_relevant_experiences("", []) == []
        assert find_relevant_experiences("query", []) == []
        assert find_relevant_experiences("", [{"id":1, "task":"task"}]) == []

# --- Tests for DocumentGenerator ---

@pytest.fixture
def mock_gemini_client_dg(mock_gemini_api_key): # Suffix dg
    """Mocks the GeminiClient for document_generator tests."""
    client = MagicMock(spec=GeminiClient)
    client.api_key = mock_gemini_api_key
    client.generate_text = MagicMock(return_value="Mocked AI Content")
    return client

@pytest.fixture
def doc_generator(mock_gemini_client_dg):
    """Provides a DocumentGenerator instance with a mocked GeminiClient."""
    return DocumentGenerator(gemini_client=mock_gemini_client_dg)

# Patch database functions for all DocumentGenerator tests
@pytestfixture(autouse=True)
def mock_db_calls(sample_user_profile_data, sample_experiences_data):
    with patch('document_generator.db.get_user_profile', return_value=sample_user_profile_data) as mock_get_profile, \
         patch('document_generator.db.get_all_experiences', return_value=sample_experiences_data) as mock_get_experiences:
        yield mock_get_profile, mock_get_experiences

# Patch find_relevant_experiences for focused testing of generator logic
@pytestfixture(autouse=True)
def mock_find_relevant_experiences_dg(sample_experiences_data):
    # By default, return the first experience or a subset if top_k is used
    def side_effect(query, experiences, top_k=3):
        return experiences[:top_k]

    with patch('document_generator.find_relevant_experiences', side_effect=side_effect) as mock_fre:
        yield mock_fre


def test_generate_resume_markdown(doc_generator, sample_user_profile_data, sample_experiences_data):
    """Test resume generation builds correct Markdown structure (not AI content)."""
    # This test focuses on the structure, not the AI-generated part (which is part of user_profile)
    # The AI part for resume is mostly if a summary is AI generated elsewhere and saved in profile.
    # Here, generate_resume_markdown is more of a template filler.

    # Reset Gemini mock for this specific test as it's not used for resume structure
    doc_generator.gemini_client.generate_text.reset_mock()

    markdown_content = doc_generator.generate_resume_markdown(sample_user_profile_data, sample_experiences_data)

    assert f"# {sample_user_profile_data['full_name']}" in markdown_content
    assert sample_user_profile_data['email'] in markdown_content
    assert "## PROFESSIONAL SUMMARY" in markdown_content
    assert sample_user_profile_data['professional_summary'] in markdown_content
    assert "## PROFESSIONAL EXPERIENCE" in markdown_content
    exp1 = sample_experiences_data[0]
    assert f"### {exp1['title']}" in markdown_content
    assert f"**{exp1['company']}**" in markdown_content
    # Check for bullet points from resume_bullets
    assert f"- {exp1['resume_bullets'].splitlines()[0].lstrip('- ')}" in markdown_content

    doc_generator.gemini_client.generate_text.assert_not_called()


def test_generate_ksc_response(doc_generator, mock_find_relevant_experiences_dg, sample_user_profile_data, sample_experiences_data, sample_ksc_question, sample_company_intel):
    role_title = "Coordinator"
    expected_ai_response = "This is the KSC response for project management."
    doc_generator.gemini_client.generate_text.return_value = expected_ai_response

    # Set mock_find_relevant_experiences_dg to return specific experiences for this test
    mock_find_relevant_experiences_dg.return_value = [sample_experiences_data[0]]


    response_data = doc_generator.generate_ksc_response(
        ksc_question=sample_ksc_question,
        user_profile=sample_user_profile_data,
        experiences=sample_experiences_data, # This will be filtered by mocked find_relevant_experiences
        company_intel=sample_company_intel,
        role_title=role_title
    )

    assert response_data["html"] == expected_ai_response

    # Check that find_relevant_experiences was called
    mock_find_relevant_experiences_dg.assert_called_once_with(sample_ksc_question, sample_experiences_data)

    # Check the prompt sent to Gemini
    args, _ = doc_generator.gemini_client.generate_text.call_args
    prompt = args[0]
    assert sample_ksc_question in prompt
    assert sample_user_profile_data['style_profile'] in prompt
    assert sample_company_intel['values_mission'] in prompt
    # Check that the relevant experience (mocked to be sample_experiences_data[0]) is in the prompt
    assert sample_experiences_data[0]['situation'] in prompt
    assert sample_experiences_data[0]['action'] in prompt


def test_generate_cover_letter_markdown(doc_generator, mock_find_relevant_experiences_dg, sample_user_profile_data, sample_experiences_data, sample_job_details, sample_company_intel):
    expected_ai_response = "Dear Hiring Manager, this is a cover letter."
    doc_generator.gemini_client.generate_text.return_value = expected_ai_response

    # Mock find_relevant_experiences to return the first experience for the snippet
    mock_find_relevant_experiences_dg.return_value = [sample_experiences_data[0]]

    markdown_content = doc_generator.generate_cover_letter_markdown(
        user_profile=sample_user_profile_data,
        experiences=sample_experiences_data,
        job_details=sample_job_details,
        company_intel=sample_company_intel
    )

    assert markdown_content == expected_ai_response
    mock_find_relevant_experiences_dg.assert_called_once_with(sample_job_details['full_text'], sample_experiences_data, top_k=1)

    args, _ = doc_generator.gemini_client.generate_text.call_args
    prompt = args[0]
    assert sample_user_profile_data['style_profile'] in prompt
    assert sample_company_intel['values_mission'] in prompt
    # Check that the snippet from the relevant experience is in the prompt
    exp0 = sample_experiences_data[0]
    assert f"in my role as a {exp0['title']} at {exp0['company']}" in prompt
    assert exp0['result'] in prompt


def test_score_resume_valid_json_response(doc_generator):
    resume_text = "My resume..."
    job_description = "Job description..."
    expected_score = {"match_score": 85, "strengths": ["Good skills"], "suggestions": ["Add more keywords"]}

    # Mock Gemini to return a valid JSON string
    doc_generator.gemini_client.generate_text.return_value = json.dumps(expected_score)

    score_data = doc_generator.score_resume(resume_text, job_description)

    assert score_data == expected_score
    args, _ = doc_generator.gemini_client.generate_text.call_args
    prompt = args[0]
    assert resume_text in prompt
    assert job_description in prompt
    assert "Return a single, valid JSON object only." in prompt

def test_score_resume_malformed_json_response(doc_generator):
    resume_text = "My resume..."
    job_description = "Job description..."
    malformed_json_string = '{"match_score": 70, "strengths": ["Relevant experience"], "suggestions": ["Quantify achievements"' # Missing closing brace and quote

    doc_generator.gemini_client.generate_text.return_value = malformed_json_string

    score_data = doc_generator.score_resume(resume_text, job_description)

    assert "error" in score_data
    assert "Could not parse AI response." in score_data["error"]
    assert score_data["raw_text"] == malformed_json_string

def test_score_resume_json_with_markdown_wrapped(doc_generator):
    resume_text = "My resume..."
    job_description = "Job description..."
    expected_score = {"match_score": 90, "strengths": ["Perfect match"], "suggestions": ["None"]}
    json_with_markdown = f"```json\n{json.dumps(expected_score)}\n```"

    doc_generator.gemini_client.generate_text.return_value = json_with_markdown

    score_data = doc_generator.score_resume(resume_text, job_description)
    assert score_data == expected_score


# Basic tests for format converters - just checking they return bytes
def test_create_docx_from_markdown(doc_generator):
    markdown_input = "# Hello\nThis is a test."
    docx_bytes = doc_generator._create_docx_from_markdown(markdown_input)
    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 0 # Check it's not empty

def test_create_pdf_from_markdown(doc_generator):
    markdown_input = "# Hello PDF\nAnother test."
    # We need to mock weasyprint.HTML and its write_pdf method
    with patch('document_generator.HTML') as mock_weasy_html:
        mock_html_instance = MagicMock()
        mock_html_instance.write_pdf.return_value = b"pdf_bytes_content"
        mock_weasy_html.return_value = mock_html_instance

        pdf_bytes = doc_generator._create_pdf_from_markdown(markdown_input)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes == b"pdf_bytes_content"
        mock_weasy_html.assert_called_once() # Check HTML() was called
        mock_html_instance.write_pdf.assert_called_once()


# Ensure the global embedding_model in document_generator is patched for find_relevant_experiences tests
# that don't use the fixture explicitly (though the above tests do for clarity)
@pytest.fixture(autouse=True)
def auto_patch_embedding_model_dg(mock_embedding_model_dg):
    with patch('document_generator.embedding_model', mock_embedding_model_dg):
        yield

# Redefine autouse fixtures here because pytest processes conftest.py first, then test files.
# Fixtures defined with autouse=True in a test file are active only for tests within that file.
# We need to ensure the mocks for db and find_relevant_experiences are active for all tests in THIS file.

@pytest.fixture(autouse=True)
def auto_mock_db_calls_for_doc_gen(sample_user_profile_data, sample_experiences_data):
    with patch('document_generator.db.get_user_profile', return_value=sample_user_profile_data) as mock_get_profile, \
         patch('document_generator.db.get_all_experiences', return_value=sample_experiences_data) as mock_get_experiences:
        # Allow re-patching or specific configuration within tests if needed
        mock_get_profile.return_value = sample_user_profile_data
        mock_get_experiences.return_value = sample_experiences_data
        yield mock_get_profile, mock_get_experiences

@pytest.fixture(autouse=True)
def auto_mock_find_relevant_experiences_for_doc_gen(sample_experiences_data):
    # Default mock: returns top_k experiences from the sample data
    def default_side_effect(query, experiences, top_k=3):
        # Ensure the 'experiences' argument here is the one passed during the call,
        # not captured from the fixture's scope at definition time.
        return experiences[:top_k]

    with patch('document_generator.find_relevant_experiences', side_effect=default_side_effect) as mock_fre:
        # Allow re-patching or specific configuration within tests if needed
        mock_fre.side_effect = default_side_effect
        yield mock_fre

# Note: I had to rename the autouse fixtures slightly (`_for_doc_gen`) because pytest was
# getting confused about redefining fixtures with the same name as non-autouse ones.
# This is a bit of a workaround; a cleaner way might involve more careful fixture scoping or
# explicit application in test classes/modules if this became a larger issue.
# For now, this should work. The key is that the `patch` calls inside these autouse fixtures
# correctly target 'document_generator.db.*' and 'document_generator.find_relevant_experiences'.

# Corrected fixture name for clarity
@pytest.fixture
def mock_db_get_user_profile(sample_user_profile_data):
    with patch('database.get_user_profile', return_value=sample_user_profile_data) as m:
        yield m

@pytest.fixture
def mock_db_get_all_experiences(sample_experiences_data):
    with patch('database.get_all_experiences', return_value=sample_experiences_data) as m:
        yield m

# Re-patching the autouse fixtures to use the correctly named dependent fixtures
# to avoid conflicts and ensure clarity.
@pytest.fixture(autouse=True)
def auto_mock_db_calls_final(mock_db_get_user_profile, mock_db_get_all_experiences):
    # These mocks are now active due to the dependent fixtures
    yield mock_db_get_user_profile, mock_db_get_all_experiences


@pytest.fixture(autouse=True)
def auto_mock_fre_final(mock_find_relevant_experiences_dg): # Using the specific mock for doc_gen
     # This mock is now active due to the dependent fixture
    yield mock_find_relevant_experiences_dg
