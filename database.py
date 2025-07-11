# database.py
"""
Handles all database operations for the user's career history, profile, and saved jobs.
Uses SQLite for simple, local, file-based storage.
This version is refactored to use context managers for safer database connections.
"""
import sqlite3
from typing import List, Dict, Any, Optional
import json

DB_FILE = "career_history.db"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    """Creates the necessary tables if they don't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Career History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS career_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                company TEXT,
                dates TEXT,
                situation TEXT NOT NULL,
                task TEXT NOT NULL,
                action TEXT NOT NULL,
                result TEXT NOT NULL,
                related_skills TEXT,
                resume_bullets TEXT
            )
        """)
        # User Profile Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                full_name TEXT,
                email TEXT,
                phone TEXT,
                address TEXT,
                linkedin_url TEXT,
                professional_summary TEXT,
                style_profile TEXT
            )
        """)
        # Saved Jobs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS saved_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                company_name TEXT,
                role_title TEXT,
                full_text TEXT, # Scraped full text of the job ad by the user
                summary_json TEXT, # AI summary of the job ad
                status TEXT NOT NULL, # e.g., 'Saved', 'Scraped', 'Summarized', 'Applied'
                saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                applied_at TIMESTAMP, # New field: Date of application
                jd_text_for_application TEXT, # New field: JD text used at time of application
                generated_resume_path TEXT, # New field
                generated_cover_letter_path TEXT, # New field
                generated_ksc_path TEXT # New field
            )
        """)
        # Ensure new columns are added if the table already exists (for existing databases)
        # This is a simple way; more robust migration would check columns individually.
        try:
            cursor.execute("ALTER TABLE saved_jobs ADD COLUMN applied_at TIMESTAMP;")
            cursor.execute("ALTER TABLE saved_jobs ADD COLUMN jd_text_for_application TEXT;")
            cursor.execute("ALTER TABLE saved_jobs ADD COLUMN generated_resume_path TEXT;")
            cursor.execute("ALTER TABLE saved_jobs ADD COLUMN generated_cover_letter_path TEXT;")
            cursor.execute("ALTER TABLE saved_jobs ADD COLUMN generated_ksc_path TEXT;")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                pass # Columns already exist, that's fine
            else:
                raise # Other operational error
        conn.commit()

# --- Career History Functions ---
def add_experience(title: str, company: str, dates: str, situation: str, task: str, action: str, result: str, skills: str, bullets: str):
    """Adds a new career experience to the database."""
    with get_db_connection() as conn:
        conn.execute("INSERT INTO career_history (title, company, dates, situation, task, action, result, related_skills, resume_bullets) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (title, company, dates, situation, task, action, result, skills, bullets))
        conn.commit()

def get_all_experiences() -> List[Dict[str, Any]]:
    """Retrieves all career experiences from the database."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM career_history ORDER BY id DESC")
        return [dict(row) for row in cursor.fetchall()]

def get_experience_by_id(exp_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single career experience by its ID."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM career_history WHERE id = ?", (exp_id,))
        experience = cursor.fetchone()
        return dict(experience) if experience else None

def update_experience(exp_id: int, title: str, company: str, dates: str, situation: str, task: str, action: str, result: str, skills: str, bullets: str):
    """Updates an existing career experience in the database."""
    with get_db_connection() as conn:
        conn.execute("UPDATE career_history SET title = ?, company = ?, dates = ?, situation = ?, task = ?, action = ?, result = ?, related_skills = ?, resume_bullets = ? WHERE id = ?", (title, company, dates, situation, task, action, result, skills, bullets, exp_id))
        conn.commit()

def delete_experience(exp_id: int):
    """Deletes a career experience from the database."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM career_history WHERE id = ?", (exp_id,))
        conn.commit()

# --- User Profile Functions ---
def save_user_profile(profile_data: Dict[str, str]):
    """Saves or updates the user's profile."""
    with get_db_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO user_profile (id, full_name, email, phone, address, linkedin_url, professional_summary, style_profile) VALUES (1, ?, ?, ?, ?, ?, ?, ?)", (profile_data.get('full_name'), profile_data.get('email'), profile_data.get('phone'), profile_data.get('address'), profile_data.get('linkedin_url'), profile_data.get('professional_summary'), profile_data.get('style_profile')))
        conn.commit()

def get_user_profile() -> Optional[Dict[str, Any]]:
    """Retrieves the user's profile."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM user_profile WHERE id = 1")
        profile = cursor.fetchone()
        return dict(profile) if profile else None

def save_style_profile(style_profile_text: str):
    """Saves the user's writing style profile."""
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO user_profile (id) VALUES (1)")
        conn.execute("UPDATE user_profile SET style_profile = ? WHERE id = 1", (style_profile_text,))
        conn.commit()

# --- Saved Jobs Functions ---
def add_job(url: str) -> Optional[int]:
    """Adds a new job by URL, ensuring it's unique."""
    with get_db_connection() as conn:
        try:
            cursor = conn.execute("INSERT INTO saved_jobs (url, status) VALUES (?, 'Saved')", (url,))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None # Job with this URL already exists

def update_job_scrape_data(job_id: int, full_text: str):
    """Updates a job record with the scraped text content."""
    with get_db_connection() as conn:
        conn.execute("UPDATE saved_jobs SET full_text = ?, status = 'Scraped' WHERE id = ?", (full_text, job_id))
        conn.commit()

def update_job_summary(job_id: int, summary: Dict, company_name: str, role_title: str):
    """Updates a job record with the AI-generated summary."""
    summary_text = json.dumps(summary)
    with get_db_connection() as conn:
        conn.execute("UPDATE saved_jobs SET summary_json = ?, company_name = ?, role_title = ?, status = 'Summarized' WHERE id = ?", (summary_text, company_name, role_title, job_id))
        conn.commit()

def get_all_saved_jobs() -> List[Dict[str, Any]]:
    """Retrieves all saved jobs from the database."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM saved_jobs ORDER BY saved_at DESC")
        return [dict(row) for row in cursor.fetchall()]

def delete_job(job_id: int):
    """Deletes a saved job from the database."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM saved_jobs WHERE id = ?", (job_id,))
        conn.commit()

def log_application_to_job(
    url: str,
    job_description_text: str,
    company_name: Optional[str] = None,
    role_title: Optional[str] = None,
    generated_doc_paths: Optional[Dict[str, str]] = None
) -> Optional[int]:
    """
    Logs an application attempt for a given job URL.
    If the job URL doesn't exist, it creates a new record.
    If it exists, it updates the record with application details.
    Returns the ID of the job record.
    """
    if generated_doc_paths is None:
        generated_doc_paths = {}

    application_time = sqlite3.Timestamp.now() # Use sqlite3.Timestamp for SQLite datetime functions

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Check if job exists
        cursor.execute("SELECT id FROM saved_jobs WHERE url = ?", (url,))
        job_row = cursor.fetchone()

        if job_row: # Job exists, update it
            job_id = job_row['id']
            cursor.execute("""
                UPDATE saved_jobs
                SET company_name = COALESCE(?, company_name),
                    role_title = COALESCE(?, role_title),
                    status = 'Applied',
                    applied_at = ?,
                    jd_text_for_application = ?,
                    generated_resume_path = ?,
                    generated_cover_letter_path = ?,
                    generated_ksc_path = ?
                WHERE id = ?
            """, (
                company_name, role_title, application_time, job_description_text,
                generated_doc_paths.get('resume'),
                generated_doc_paths.get('cover_letter'),
                generated_doc_paths.get('ksc'),
                job_id
            ))
            conn.commit()
            return job_id
        else: # Job doesn't exist, create it
            try:
                cursor.execute("""
                    INSERT INTO saved_jobs (
                        url, company_name, role_title, status, saved_at,
                        applied_at, jd_text_for_application,
                        generated_resume_path, generated_cover_letter_path, generated_ksc_path
                    ) VALUES (?, ?, ?, 'Applied', ?, ?, ?, ?, ?, ?)
                """, (
                    url, company_name, role_title, application_time, # saved_at = applied_at for new entry
                    application_time, job_description_text,
                    generated_doc_paths.get('resume'),
                    generated_doc_paths.get('cover_letter'),
                    generated_doc_paths.get('ksc')
                ))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError: # Should not happen if previous check was done, but as a safeguard
                # This implies a race condition or logic error if reached.
                # Try to fetch the ID again if it was inserted by another process.
                cursor.execute("SELECT id FROM saved_jobs WHERE url = ?", (url,))
                job_row_after_integrity_error = cursor.fetchone()
                if job_row_after_integrity_error:
                    return job_row_after_integrity_error['id']
                return None # Failed to insert or find
