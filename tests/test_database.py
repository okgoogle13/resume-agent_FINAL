# tests/test_database.py
import sqlite3
import pytest
from database import (
    get_db_connection,
    initialize_db,
    add_experience,
    get_all_experiences,
    get_experience_by_id,
    update_experience,
    delete_experience,
    save_user_profile,
    get_user_profile,
    save_style_profile,
    add_job,
    update_job_scrape_data,
    update_job_summary,
    get_all_saved_jobs,
    delete_job,
)

@pytest.fixture
def db_path(tmp_path):
    """Fixture to create a temporary database for testing."""
    return tmp_path / "test_career_history.db"

@pytest.fixture(autouse=True)
def setup_database(db_path, monkeypatch):
    """Fixture to set up and tear down the database for each test."""
    monkeypatch.setattr("database.DB_FILE", db_path)
    initialize_db()
    yield
    # No teardown needed as tmp_path handles it

def test_initialize_db(db_path):
    """Test that the database and tables are created."""
    assert db_path.exists()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    assert "career_history" in tables
    assert "user_profile" in tables
    assert "saved_jobs" in tables
    conn.close()

def test_add_and_get_experience(db_path):
    """Test adding and retrieving a career experience."""
    add_experience("Test Title", "Test Company", "2022-2023", "Situation", "Task", "Action", "Result", "Skills", "Bullets")
    experiences = get_all_experiences()
    assert len(experiences) == 1
    assert experiences[0]["title"] == "Test Title"

def test_get_experience_by_id(db_path):
    """Test retrieving an experience by its ID."""
    add_experience("Test Title", "Test Company", "2022-2023", "Situation", "Task", "Action", "Result", "Skills", "Bullets")
    exp_id = get_all_experiences()[0]['id']
    experience = get_experience_by_id(exp_id)
    assert experience is not None
    assert experience["title"] == "Test Title"

def test_update_experience(db_path):
    """Test updating an existing experience."""
    add_experience("Old Title", "Old Company", "2021-2022", "Old Situation", "Old Task", "Old Action", "Old Result", "Old Skills", "Old Bullets")
    exp_id = get_all_experiences()[0]['id']
    update_experience(exp_id, "New Title", "New Company", "2022-2023", "New Situation", "New Task", "New Action", "New Result", "New Skills", "New Bullets")
    experience = get_experience_by_id(exp_id)
    assert experience["title"] == "New Title"
    assert experience["company"] == "New Company"

def test_delete_experience(db_path):
    """Test deleting an experience."""
    add_experience("Test Title", "Test Company", "2022-2023", "Situation", "Task", "Action", "Result", "Skills", "Bullets")
    exp_id = get_all_experiences()[0]['id']
    delete_experience(exp_id)
    experiences = get_all_experiences()
    assert len(experiences) == 0

def test_save_and_get_user_profile(db_path):
    """Test saving and retrieving a user profile."""
    profile_data = {
        "full_name": "Test User",
        "email": "test@example.com",
        "phone": "1234567890",
        "address": "123 Test St",
        "linkedin_url": "linkedin.com/test",
        "professional_summary": "A test summary.",
        "style_profile": "A test style profile."
    }
    save_user_profile(profile_data)
    profile = get_user_profile()
    assert profile is not None
    assert profile["full_name"] == "Test User"
    assert profile["email"] == "test@example.com"

def test_save_style_profile(db_path):
    """Test saving a style profile."""
    save_style_profile("New style profile.")
    profile = get_user_profile()
    assert profile is not None
    assert profile["style_profile"] == "New style profile."

def test_add_and_get_saved_jobs(db_path):
    """Test adding and retrieving a saved job."""
    add_job("http://example.com/job1")
    jobs = get_all_saved_jobs()
    assert len(jobs) == 1
    assert jobs[0]["url"] == "http://example.com/job1"
    assert jobs[0]["status"] == "Saved"

def test_add_duplicate_job(db_path):
    """Test that adding a duplicate job URL fails gracefully."""
    add_job("http://example.com/job1")
    result = add_job("http://example.com/job1")
    assert result is None
    jobs = get_all_saved_jobs()
    assert len(jobs) == 1

def test_update_job_scrape_data(db_path):
    """Test updating a job with scraped data."""
    job_id = add_job("http://example.com/job1")
    update_job_scrape_data(job_id, "This is the scraped job text.")
    jobs = get_all_saved_jobs()
    assert jobs[0]["full_text"] == "This is the scraped job text."
    assert jobs[0]["status"] == "Scraped"

def test_update_job_summary(db_path):
    """Test updating a job with a summary."""
    job_id = add_job("http://example.com/job1")
    summary_data = {"role": "Software Engineer", "requirements": ["Python", "SQL"]}
    update_job_summary(job_id, summary_data, "Test Corp", "Senior Developer")
    jobs = get_all_saved_jobs()
    assert jobs[0]["company_name"] == "Test Corp"
    assert jobs[0]["role_title"] == "Senior Developer"
    assert jobs[0]["status"] == "Summarized"
    import json
    assert json.loads(jobs[0]["summary_json"]) == summary_data

def test_delete_job(db_path):
    """Test deleting a saved job."""
    job_id = add_job("http://example.com/job1")
    delete_job(job_id)
    jobs = get_all_saved_jobs()
    assert len(jobs) == 0
