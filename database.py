# database.py
"""
Handles all database operations for the user's career history, profile, and saved jobs.
Uses SQLite for simple, local, file-based storage.
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
                full_text TEXT,
                summary_json TEXT,
                status TEXT NOT NULL,
                saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

# --- Career History Functions (Omitted for brevity) ---
def add_experience(title: str, company: str, dates: str, situation: str, task: str, action: str, result: str, skills: str, bullets: str):
    with get_db_connection() as conn:
        conn.execute("INSERT INTO career_history (title, company, dates, situation, task, action, result, related_skills, resume_bullets) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (title, company, dates, situation, task, action, result, skills, bullets))
        conn.commit()
def get_all_experiences() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM career_history ORDER BY id DESC")
        return [dict(row) for row in cursor.fetchall()]
def get_experience_by_id(exp_id: int) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM career_history WHERE id = ?", (exp_id,))
        experience = cursor.fetchone()
        return dict(experience) if experience else None
def update_experience(exp_id: int, title: str, company: str, dates: str, situation: str, task: str, action: str, result: str, skills: str, bullets: str):
    with get_db_connection() as conn:
        conn.execute("UPDATE career_history SET title = ?, company = ?, dates = ?, situation = ?, task = ?, action = ?, result = ?, related_skills = ?, resume_bullets = ? WHERE id = ?", (title, company, dates, situation, task, action, result, skills, bullets, exp_id))
        conn.commit()
def delete_experience(exp_id: int):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM career_history WHERE id = ?", (exp_id,))
        conn.commit()

# --- User Profile Functions ---
def save_user_profile(profile_data: Dict[str, str]):
    with get_db_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO user_profile (id, full_name, email, phone, address, linkedin_url, professional_summary, style_profile) VALUES (1, ?, ?, ?, ?, ?, ?, ?)", (profile_data.get('full_name'), profile_data.get('email'), profile_data.get('phone'), profile_data.get('address'), profile_data.get('linkedin_url'), profile_data.get('professional_summary'), profile_data.get('style_profile')))
        conn.commit()
def get_user_profile() -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM user_profile WHERE id = 1")
        profile = cursor.fetchone()
        return dict(profile) if profile else None
def save_style_profile(style_profile_text: str):
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO user_profile (id) VALUES (1)")
        conn.execute("UPDATE user_profile SET style_profile = ? WHERE id = 1", (style_profile_text,))
        conn.commit()

# --- Saved Jobs Functions (Omitted for brevity) ---
def add_job(url: str) -> Optional[int]:
    with get_db_connection() as conn:
        try:
            cursor = conn.execute("INSERT INTO saved_jobs (url, status) VALUES (?, 'Saved')", (url,))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None # Job with this URL already exists
def update_job_scrape_data(job_id: int, full_text: str):
    with get_db_connection() as conn:
        conn.execute("UPDATE saved_jobs SET full_text = ?, status = 'Scraped' WHERE id = ?", (full_text, job_id))
        conn.commit()
def update_job_summary(job_id: int, summary: Dict, company_name: str, role_title: str):
    summary_text = json.dumps(summary)
    with get_db_connection() as conn:
        conn.execute("UPDATE saved_jobs SET summary_json = ?, company_name = ?, role_title = ?, status = 'Summarized' WHERE id = ?", (summary_text, company_name, role_title, job_id))
        conn.commit()
def get_all_saved_jobs() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM saved_jobs ORDER BY saved_at DESC")
        return [dict(row) for row in cursor.fetchall()]
def delete_job(job_id: int):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM saved_jobs WHERE id = ?", (job_id,))
        conn.commit()
