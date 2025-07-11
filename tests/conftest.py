# tests/conftest.py
"""
This file will contain shared fixtures for pytest.
For example, mock API keys or sample data structures.
"""
import pytest

@pytest.fixture
def mock_gemini_api_key():
    return "test_gemini_api_key"

@pytest.fixture
def mock_perplexity_api_key():
    return "test_perplexity_api_key"

@pytest.fixture
def sample_user_profile_data():
    return {
        "full_name": "Test User",
        "email": "test@example.com",
        "phone": "123-456-7890",
        "address": "123 Test St",
        "linkedin_url": "linkedin.com/in/testuser",
        "professional_summary": "A dedicated professional.",
        "style_profile": "Professional and concise."
    }

@pytest.fixture
def sample_experiences_data():
    return [
        {
            "id": 1,
            "title": "Software Engineer",
            "company": "Tech Solutions Inc.",
            "dates": "Jan 2020 - Present",
            "situation": "Needed to improve data processing speed.",
            "task": "Redesign data pipeline.",
            "action": "Implemented a new parallel processing system.",
            "result": "Reduced processing time by 50%.",
            "related_skills": "Python, AWS, Data Engineering",
            "resume_bullets": "- Redesigned data pipeline, reducing processing time by 50%.\n- Led a team of 3 engineers."
        },
        {
            "id": 2,
            "title": "Junior Developer",
            "company": "Web Innovations",
            "dates": "Jun 2018 - Dec 2019",
            "situation": "Company website was outdated.",
            "task": "Develop a new company website.",
            "action": "Used React and Node.js to build a modern, responsive website.",
            "result": "Increased user engagement by 30%.",
            "related_skills": "React, Node.js, JavaScript",
            "resume_bullets": "- Developed new company website, increasing user engagement by 30%."
        }
    ]

@pytest.fixture
def sample_ksc_question():
    return "Describe your experience in project management."

@pytest.fixture
def sample_company_intel():
    return {
        "values_mission": "Our mission is to innovate and lead.",
        "recent_news": "Launched a new product last quarter.",
        "role_context": "This role requires strong leadership."
    }

@pytest.fixture
def sample_job_details():
    return {
        "full_text": "We are looking for a skilled project manager with experience in agile methodologies.",
        "role_title": "Senior Project Manager"
    }
