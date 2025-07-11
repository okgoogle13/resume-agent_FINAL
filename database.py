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

def initialize_db(conn: Optional[sqlite3.Connection] = None):
    """
    Creates the necessary tables if they don't exist.
    If a connection is provided, it uses it; otherwise, it creates a new one.
    """
    if conn:
        # Use the provided connection directly, don't wrap in context manager
        # as the caller (fixture) will manage its lifecycle.
        cursor = conn.cursor()
    else:
        # Original behavior: create and manage connection internally
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__() # Manually enter for this path
        cursor = conn.cursor()

    # Career History Table
    try:
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
    finally:
        # If db_conn_cm was created in this function, ensure it's closed.
        if 'db_conn_cm' in locals() and db_conn_cm:
            db_conn_cm.__exit__(None, None, None)

# --- Career History Functions ---
def add_experience(title: str, company: str, dates: str, situation: str, task: str, action: str, result: str, skills: str, bullets: str, conn: Optional[sqlite3.Connection] = None):
    """Adds a new career experience. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()
    try:
        conn.execute("INSERT INTO career_history (title, company, dates, situation, task, action, result, related_skills, resume_bullets) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (title, company, dates, situation, task, action, result, skills, bullets))
        conn.commit()
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)

def get_all_experiences(conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """Retrieves all career experiences. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()

    experiences = []
    try:
        cursor = conn.execute("SELECT * FROM career_history ORDER BY id DESC")
        experiences = [dict(row) for row in cursor.fetchall()]
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)
    return experiences

def get_experience_by_id(exp_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single career experience by ID. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()

    experience_dict = None
    try:
        cursor = conn.execute("SELECT * FROM career_history WHERE id = ?", (exp_id,))
        experience_row = cursor.fetchone()
        experience_dict = dict(experience_row) if experience_row else None
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)
    return experience_dict

def update_experience(exp_id: int, title: str, company: str, dates: str, situation: str, task: str, action: str, result: str, skills: str, bullets: str, conn: Optional[sqlite3.Connection] = None):
    """Updates an existing career experience. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()
    try:
        conn.execute("UPDATE career_history SET title = ?, company = ?, dates = ?, situation = ?, task = ?, action = ?, result = ?, related_skills = ?, resume_bullets = ? WHERE id = ?", (title, company, dates, situation, task, action, result, skills, bullets, exp_id))
        conn.commit()
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)

def delete_experience(exp_id: int, conn: Optional[sqlite3.Connection] = None):
    """Deletes a career experience. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()
    try:
        conn.execute("DELETE FROM career_history WHERE id = ?", (exp_id,))
        conn.commit()
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)

# --- User Profile Functions ---
def save_user_profile(profile_data: Dict[str, str], conn: Optional[sqlite3.Connection] = None):
    """Saves or updates the user's profile. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()

    try:
        conn.execute("INSERT OR REPLACE INTO user_profile (id, full_name, email, phone, address, linkedin_url, professional_summary, style_profile) VALUES (1, ?, ?, ?, ?, ?, ?, ?)", (profile_data.get('full_name'), profile_data.get('email'), profile_data.get('phone'), profile_data.get('address'), profile_data.get('linkedin_url'), profile_data.get('professional_summary'), profile_data.get('style_profile')))
        conn.commit()
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)


def get_user_profile(conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the user's profile. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()

    profile = None
    try:
        cursor = conn.execute("SELECT * FROM user_profile WHERE id = 1")
        profile_row = cursor.fetchone()
        profile = dict(profile_row) if profile_row else None
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)
    return profile

def save_style_profile(style_profile_text: str, conn: Optional[sqlite3.Connection] = None):
    """Saves the user's writing style profile. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()

    try:
        conn.execute("INSERT OR IGNORE INTO user_profile (id) VALUES (1)")
        conn.execute("UPDATE user_profile SET style_profile = ? WHERE id = 1", (style_profile_text,))
        conn.commit()
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)

# --- Saved Jobs Functions ---
def add_job(url: str, conn: Optional[sqlite3.Connection] = None) -> Optional[int]:
    """Adds a new job by URL, ensuring it's unique. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()

    last_id = None
    try:
        cursor = conn.execute("INSERT INTO saved_jobs (url, status) VALUES (?, 'Saved')", (url,))
        conn.commit()
        last_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        last_id = None # Job with this URL already exists
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)
    return last_id

def update_job_scrape_data(job_id: int, full_text: str, conn: Optional[sqlite3.Connection] = None):
    """Updates a job record with the scraped text content. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()
    try:
        conn.execute("UPDATE saved_jobs SET full_text = ?, status = 'Scraped' WHERE id = ?", (full_text, job_id))
        conn.commit()
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)

def update_job_summary(job_id: int, summary: Dict, company_name: str, role_title: str, conn: Optional[sqlite3.Connection] = None):
    """Updates a job record with the AI-generated summary. Uses provided conn or creates new."""
    summary_text = json.dumps(summary)
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()
    try:
        conn.execute("UPDATE saved_jobs SET summary_json = ?, company_name = ?, role_title = ?, status = 'Summarized' WHERE id = ?", (summary_text, company_name, role_title, job_id))
        conn.commit()
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)

def get_all_saved_jobs(conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """Retrieves all saved jobs from the database. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()

    jobs = []
    try:
        cursor = conn.execute("SELECT * FROM saved_jobs ORDER BY saved_at DESC")
        jobs = [dict(row) for row in cursor.fetchall()]
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)
    return jobs

def delete_job(job_id: int, conn: Optional[sqlite3.Connection] = None):
    """Deletes a saved job from the database. Uses provided conn or creates new."""
    db_conn_cm = None
    if not conn:
        db_conn_cm = get_db_connection()
        conn = db_conn_cm.__enter__()
    try:
        conn.execute("DELETE FROM saved_jobs WHERE id = ?", (job_id,))
        conn.commit()
    finally:
        if db_conn_cm:
            db_conn_cm.__exit__(None, None, None)
