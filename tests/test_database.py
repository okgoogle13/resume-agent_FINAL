# tests/test_database.py
import pytest
import sqlite3
from unittest.mock import patch
# Make sure the database module can be imported
# This might need adjustment based on your project structure.
# If 'database.py' is in the root, and 'tests' is a subdir,
# you might need to adjust sys.path or use relative imports if it's part of a package.
# For now, assuming direct import works or will be configured via PYTHONPATH.
import database

@pytest.fixture
def memory_db():
    """Fixture to use an in-memory SQLite database for tests."""
    # Patch DB_FILE for all db operations within database.py to use :memory:
    with patch('database.DB_FILE', ':memory:'):
        # Create a single connection for the test session
        conn = database.get_db_connection()
        # Initialize the schema using this specific connection
        database.initialize_db(conn=conn)
        yield conn
        # The connection will be closed when the test session ends or if an error occurs
        conn.close()

def test_initialize_db(memory_db):
    """Test that tables are created by initialize_db."""
    cursor = memory_db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profile'")
    assert cursor.fetchone() is not None, "user_profile table should exist"
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='career_history'")
    assert cursor.fetchone() is not None, "career_history table should exist"
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='saved_jobs'")
    assert cursor.fetchone() is not None, "saved_jobs table should exist"

def test_save_and_get_user_profile(memory_db, sample_user_profile_data):
    """Test saving and retrieving a user profile."""
    database.save_user_profile(sample_user_profile_data, conn=memory_db)
    profile = database.get_user_profile(conn=memory_db)

    assert profile is not None
    assert profile["full_name"] == sample_user_profile_data["full_name"]
    assert profile["email"] == sample_user_profile_data["email"]
    assert profile["style_profile"] == sample_user_profile_data["style_profile"]

def test_get_user_profile_empty(memory_db):
    """Test retrieving a user profile when none is saved."""
    profile = database.get_user_profile(conn=memory_db)
    assert profile is None

def test_add_and_get_all_experiences(memory_db, sample_experiences_data):
    """Test adding and retrieving career experiences."""
    exp1 = sample_experiences_data[0]
    exp2 = sample_experiences_data[1]

    database.add_experience(
        title=exp1["title"], company=exp1["company"], dates=exp1["dates"],
        situation=exp1["situation"], task=exp1["task"], action=exp1["action"],
        result=exp1["result"], skills=exp1["related_skills"], bullets=exp1["resume_bullets"],
        conn=memory_db
    )
    database.add_experience(
        title=exp2["title"], company=exp2["company"], dates=exp2["dates"],
        situation=exp2["situation"], task=exp2["task"], action=exp2["action"],
        result=exp2["result"], skills=exp2["related_skills"], bullets=exp2["resume_bullets"],
        conn=memory_db
    )

    experiences = database.get_all_experiences(conn=memory_db)
    assert len(experiences) == 2
    # Experiences are returned in DESC order of ID
    assert experiences[0]["title"] == exp2["title"]
    assert experiences[1]["title"] == exp1["title"]

def test_get_all_experiences_empty(memory_db):
    """Test retrieving experiences when none are saved."""
    experiences = database.get_all_experiences(conn=memory_db)
    assert len(experiences) == 0

# Example of how you might test other CRUD, if you expand later:
# def test_add_and_get_experience_by_id(memory_db, sample_experiences_data):
#     exp1 = sample_experiences_data[0]
#     database.add_experience(
#         title=exp1["title"], company=exp1["company"], dates=exp1["dates"],
#         situation=exp1["situation"], task=exp1["task"], action=exp1["action"],
#         result=exp1["result"], skills=exp1["related_skills"], bullets=exp1["resume_bullets"],
#         conn=memory_db
#     )
#     # Assuming add_experience auto-increments ID starting from 1
#     retrieved_exp = database.get_experience_by_id(1, conn=memory_db)
#     assert retrieved_exp is not None
#     assert retrieved_exp["title"] == exp1["title"]

# def test_update_experience(memory_db, sample_experiences_data):
#     exp1 = sample_experiences_data[0]
#     database.add_experience(
#         title=exp1["title"], company=exp1["company DATES (Dates)":exp1["dates"],
#         situation=exp1["situation"], task=exp1["task"], action=exp1["action"],
#         result=exp1["result"], skills=exp1["related_skills"], bullets=exp1["resume_bullets"],
#         conn=memory_db
#     )
#     updated_title = "Senior Software Engineer"
#     database.update_experience(
#         1, title=updated_title, company=exp1["company"], dates=exp1["dates"],
#         situation=exp1["situation"], task=exp1["task"], action=exp1["action"],
#         result=exp1["result"], skills=exp1["related_skills"], bullets=exp1["resume_bullets"],
#         conn=memory_db
#     )
#     updated_exp = database.get_experience_by_id(1, conn=memory_db)
#     assert updated_exp["title"] == updated_title

# def test_delete_experience(memory_db, sample_experiences_data):
#     exp1 = sample_experiences_data[0]
#     database.add_experience(
#         title=exp1["title"], company=exp1["company"], dates=exp1["dates"],
#         situation=exp1["situation"], task=exp1["task"], action=exp1["action"],
#         result=exp1["result"], skills=exp1["related_skills"], bullets=exp1["resume_bullets"],
#         conn=memory_db
#     )
#     database.delete_experience(1, conn=memory_db)
#     assert database.get_experience_by_id(1, conn=memory_db) is None
#     assert len(database.get_all_experiences(conn=memory_db)) == 0

# --- Tests for Style Profile ---
def test_save_and_get_style_profile(memory_db):
    """Test saving and retrieving the style profile specifically."""
    style_text = "Very eloquent and persuasive."
    # First save a basic profile, as save_style_profile updates an existing one or creates one if id=1 doesn't exist
    database.save_user_profile({"full_name": "Test User"}, conn=memory_db)
    database.save_style_profile(style_text, conn=memory_db)

    profile = database.get_user_profile(conn=memory_db)
    assert profile is not None
    assert profile["style_profile"] == style_text

def test_save_style_profile_on_empty_db(memory_db):
    """Test save_style_profile when user_profile table is empty."""
    style_text = "Direct and to the point."
    database.save_style_profile(style_text, conn=memory_db) # Should insert a new row with id=1

    profile = database.get_user_profile(conn=memory_db)
    assert profile is not None
    assert profile["id"] == 1 # Check that it created the default row
    assert profile["style_profile"] == style_text
    assert profile["full_name"] is None # Other fields should be None initially
