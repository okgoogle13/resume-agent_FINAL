#!/bin/bash
# Setup script for the Resume Agent 2.0 (Final Polished Edition)
# This script creates the full project structure and files.

echo "Setting up project structure..."
mkdir -p pages

echo "Creating project files..."

# --- requirements.txt ---
cat <<'EOF' > requirements.txt
# Core dependencies
streamlit
google-generativeai
httpx
cachetools
tenacity
markdown

# Document processing
python-docx
pypdf
weasyprint

# Web Scraping & Semantic Search
playwright
beautifulsoup4
sentence-transformers
scikit-learn
torch
torchvision

# Utilities
python-dotenv
EOF

# --- database.py ---
cat <<'EOF' > database.py
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
    conn = get_db_connection()
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
    conn.close()

# --- Career History Functions (Omitted for brevity) ---
def add_experience(title: str, company: str, dates: str, situation: str, task: str, action: str, result: str, skills: str, bullets: str):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("INSERT INTO career_history (title, company, dates, situation, task, action, result, related_skills, resume_bullets) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (title, company, dates, situation, task, action, result, skills, bullets))
    conn.commit(); conn.close()
def get_all_experiences() -> List[Dict[str, Any]]:
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT * FROM career_history ORDER BY id DESC")
    experiences = [dict(row) for row in cursor.fetchall()]; conn.close(); return experiences
def get_experience_by_id(exp_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT * FROM career_history WHERE id = ?", (exp_id,)); experience = cursor.fetchone(); conn.close(); return dict(experience) if experience else None
def update_experience(exp_id: int, title: str, company: str, dates: str, situation: str, task: str, action: str, result: str, skills: str, bullets: str):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("UPDATE career_history SET title = ?, company = ?, dates = ?, situation = ?, task = ?, action = ?, result = ?, related_skills = ?, resume_bullets = ? WHERE id = ?", (title, company, dates, situation, task, action, result, skills, bullets, exp_id))
    conn.commit(); conn.close()
def delete_experience(exp_id: int):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("DELETE FROM career_history WHERE id = ?", (exp_id,)); conn.commit(); conn.close()

# --- User Profile Functions ---
def save_user_profile(profile_data: Dict[str, str]):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO user_profile (id, full_name, email, phone, address, linkedin_url, professional_summary, style_profile) VALUES (1, ?, ?, ?, ?, ?, ?, ?)", (profile_data.get('full_name'), profile_data.get('email'), profile_data.get('phone'), profile_data.get('address'), profile_data.get('linkedin_url'), profile_data.get('professional_summary'), profile_data.get('style_profile')))
    conn.commit(); conn.close()
def get_user_profile() -> Optional[Dict[str, Any]]:
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_profile WHERE id = 1"); profile = cursor.fetchone(); conn.close(); return dict(profile) if profile else None
def save_style_profile(style_profile_text: str):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO user_profile (id) VALUES (1)")
    cursor.execute("UPDATE user_profile SET style_profile = ? WHERE id = 1", (style_profile_text,))
    conn.commit(); conn.close()

# --- Saved Jobs Functions (Omitted for brevity) ---
def add_job(url: str) -> Optional[int]:
    conn = get_db_connection(); cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO saved_jobs (url, status) VALUES (?, 'Saved')", (url,)); conn.commit(); job_id = cursor.lastrowid
    except sqlite3.IntegrityError: job_id = None
    finally: conn.close()
    return job_id
def update_job_scrape_data(job_id: int, full_text: str):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("UPDATE saved_jobs SET full_text = ?, status = 'Scraped' WHERE id = ?", (full_text, job_id)); conn.commit(); conn.close()
def update_job_summary(job_id: int, summary: Dict, company_name: str, role_title: str):
    conn = get_db_connection(); cursor = conn.cursor(); summary_text = json.dumps(summary)
    cursor.execute("UPDATE saved_jobs SET summary_json = ?, company_name = ?, role_title = ?, status = 'Summarized' WHERE id = ?", (summary_text, company_name, role_title, job_id))
    conn.commit(); conn.close()
def get_all_saved_jobs() -> List[Dict[str, Any]]:
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT * FROM saved_jobs ORDER BY saved_at DESC"); jobs = [dict(row) for row in cursor.fetchall()]; conn.close(); return jobs
def delete_job(job_id: int):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("DELETE FROM saved_jobs WHERE id = ?", (job_id,)); conn.commit(); conn.close()
EOF

# --- api_clients.py ---
cat <<'EOF' > api_clients.py
# api_clients.py
"""
Centralized clients for interacting with external APIs like Google Gemini and Perplexity.
Includes robust error handling and automatic retries.
"""
import google.generativeai as genai
import httpx
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

class GeminiClient:
    """Client for interacting with the Google Gemini API."""
    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_text(self, prompt: str) -> str:
        """Generates text content from a given prompt with retry logic."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise

class PerplexityClient:
    """Client for interacting with the Perplexity Sonar API."""
    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            raise ValueError("Perplexity API key is required.")
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def search(self, query: str) -> Dict[str, Any]:
        """Performs a search using the Perplexity API with retry logic."""
        payload = {
            "model": "sonar-small-online",
            "messages": [{"role": "user", "content": query}]
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting from Perplexity: {e}")
            raise
        except httpx.HTTPStatusError as e:
            print(f"Perplexity API returned an error: {e.response.status_code} - {e.response.text}")
            raise
EOF

# --- intelligence_booster.py ---
cat <<'EOF' > intelligence_booster.py
# intelligence_booster.py
"""
Module to enrich context by gathering live intelligence about a company.
"""
from cachetools import TTLCache
from api_clients import PerplexityClient
from typing import Dict

class QueryGenerator:
    """Generates targeted questions for the Perplexity API."""
    def generate_for_company(self, company_name: str, role_title: str) -> Dict[str, str]:
        return {
            "values_mission": f"What are the publicly stated values, mission, or vision of the organization '{company_name}'?",
            "recent_news": f"Summarize recent news, projects, or developments for '{company_name}' in the last 6 months.",
            "role_context": f"What are the typical challenges or objectives for a '{role_title}' within the community services or social work sector in Australia?"
        }

class IntelligenceBoosterModule:
    """Facade for the Company Researcher, orchestrating queries and caching."""
    def __init__(self, perplexity_client: PerplexityClient):
        self.client = perplexity_client
        self.query_generator = QueryGenerator()
        self.cache = TTLCache(maxsize=100, ttl=86400)

    def get_intelligence(self, company_name: str, role_title: str) -> Dict[str, str]:
        """Main method to fetch and structure company intelligence."""
        cache_key = f"{company_name}_{role_title}".lower()
        if cache_key in self.cache:
            return self.cache[cache_key]

        queries = self.query_generator.generate_for_company(company_name, role_title)
        intelligence = {}

        for key, query in queries.items():
            try:
                response = self.client.search(query)
                intelligence[key] = response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Could not fetch intelligence for query '{query}': {e}")
                intelligence[key] = f"Error fetching data: {e}"

        self.cache[cache_key] = intelligence
        return intelligence
EOF

# --- file_parser.py ---
cat <<'EOF' > file_parser.py
# file_parser.py
"""
A utility module to parse text content from different file types like PDF and DOCX.
"""
import io
from typing import List
from docx import Document
from pypdf import PdfReader
import streamlit as st

def parse_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """Parses a list of uploaded files and returns their combined text content."""
    full_text = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            text = parse_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = parse_docx(file)
        elif file.type == "text/plain":
            text = file.getvalue().decode("utf-8")
        else:
            text = f"Unsupported file type: {file.name}"
        full_text.append(f"--- Document: {file.name} ---\n{text}\n\n")
    return "".join(full_text)

def parse_pdf(file: io.BytesIO) -> str:
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error parsing PDF: {e}"

def parse_docx(file: io.BytesIO) -> str:
    """Extracts text from an uploaded DOCX file."""
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error parsing DOCX: {e}"
EOF

# --- document_generator.py ---
cat <<'EOF' > document_generator.py
# document_generator.py
"""
Handles the core logic of generating documents as structured Markdown text.
This version includes improved export quality for DOCX and PDF.
"""
from api_clients import GeminiClient
from typing import List, Dict, Any
from docx import Document
from docx.shared import Pt
from weasyprint import HTML, CSS
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import markdown
import json

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def find_relevant_experiences(question: str, experiences: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    if not experiences: return []
    experience_texts = [f"{exp['title']}. {exp['situation']} {exp['task']} {exp['action']} {exp['result']}" for exp in experiences]
    question_embedding = embedding_model.encode([question])
    experience_embeddings = embedding_model.encode(experience_texts)
    similarities = cosine_similarity(question_embedding, experience_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [experiences[i] for i in top_indices]

class DocumentGenerator:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client

    def _generate_ai_content(self, prompt: str) -> str:
        return self.gemini_client.generate_text(prompt)

    def _create_docx_from_markdown(self, markdown_content: str) -> bytes:
        """Creates a formatted DOCX from Markdown content."""
        doc = Document()
        for line in markdown_content.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                doc.add_heading(line[2:], level=1)
            elif line.startswith('## '):
                doc.add_heading(line[3:], level=2)
            elif line.startswith('### '):
                p = doc.add_paragraph()
                p.add_run(line[4:]).bold = True
            elif line.startswith('- '):
                doc.add_paragraph(line[2:], style='List Bullet')
            elif line:
                doc.add_paragraph(line)
        
        bio = io.BytesIO()
        doc.save(bio)
        return bio.getvalue()

    def _create_pdf_from_markdown(self, markdown_content: str) -> bytes:
        """Creates a styled PDF from Markdown content."""
        html_content = markdown.markdown(markdown_content)
        # Professional styling for the PDF
        css = CSS(string='''
            @page { size: A4; margin: 2cm; }
            body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.5; }
            h1 { font-size: 24pt; color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 5px; }
            h2 { font-size: 16pt; color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 3px; margin-top: 25px; }
            h3 { font-size: 12pt; color: #34495e; font-weight: bold; }
            ul { list-style-type: disc; }
        ''')
        return HTML(string=html_content).write_pdf(stylesheets=[css])

    def generate_resume_markdown(self, user_profile: Dict, experiences: List[Dict[str, Any]]) -> str:
        md_parts = [f"# {user_profile.get('full_name', 'Your Name')}"]
        contact_info = " | ".join(filter(None, [user_profile.get('phone'), user_profile.get('email'), user_profile.get('address'), user_profile.get('linkedin_url')]))
        md_parts.append(contact_info)
        if user_profile.get('professional_summary'):
            md_parts.append("\n## PROFESSIONAL SUMMARY")
            md_parts.append(f"{user_profile.get('professional_summary')}")
        if experiences:
            md_parts.append("\n## PROFESSIONAL EXPERIENCE")
            for exp in experiences:
                md_parts.append(f"\n### {exp.get('title')}")
                md_parts.append(f"**{exp.get('company')}** | *{exp.get('dates')}*")
                bullets = exp.get('resume_bullets', '').split('\n')
                for bullet in bullets:
                    if bullet.strip(): md_parts.append(f"- {bullet.strip()}")
        return "\n".join(md_parts)

    def generate_ksc_response(self, ksc_question: str, user_profile: Dict, experiences: List[Dict[str, Any]], company_intel: Dict[str, str], role_title: str) -> Dict[str, Any]:
        relevant_experiences = find_relevant_experiences(ksc_question, experiences)
        experience_text = "\n\n".join([f"Title: {exp['title']}\nSituation: {exp['situation']}\nTask: {exp['task']}\nAction: {exp['action']}\nResult: {exp['result']}" for exp in relevant_experiences])
        prompt = f"""
        **Persona:** You are an expert career coach for the Australian Community Services sector. Your tone is professional and authentic, mirroring the user's personal style if provided.
        **User's Personal Style Profile:** {user_profile.get('style_profile', 'N/A')}
        **Company Intelligence:** {company_intel.get('values_mission', 'N/A')}
        **Task:** Write a compelling KSC response to the question below.
        **KSC Question:** "{ksc_question}"
        **Reasoning Framework:**
        1.  **Deconstruct:** Identify the core competency.
        2.  **Select Evidence:** Choose the strongest parts of the STAR stories to prove this competency.
        3.  **Draft:** Structure the response using the STAR method, adopting the user's personal style and aligning with company intelligence.
        **Candidate's Most Relevant Career Examples:**
        ---
        {experience_text if experience_text else "No specific examples provided."}
        ---
        **Output Format:** Generate clean Markdown.
        """
        markdown_content = self._generate_ai_content(prompt)
        return {"html": markdown_content, "docx": self._create_docx_from_markdown(markdown_content), "pdf": self._create_pdf_from_markdown(markdown_content)}

    def generate_cover_letter_markdown(self, user_profile: Dict, experiences: List[Dict[str, Any]], job_details: Dict[str, Any], company_intel: Dict[str, Any]) -> str:
        most_relevant_experience = find_relevant_experiences(job_details.get('full_text', ''), experiences, top_k=1)
        experience_snippet = ""
        if most_relevant_experience:
            exp = most_relevant_experience[0]
            experience_snippet = f"In my role as a {exp['title']} at {exp['company']}, I was responsible for {exp['task']}. I successfully {exp['action']}, which resulted in {exp['result']}."
        prompt = f"""
        **Persona:** You are an expert career advisor, writing a cover letter for the Australian Community Services sector. Your tone is professional and warm, mirroring the user's personal style if provided.
        **User's Personal Style Profile:** {user_profile.get('style_profile', 'N/A')}
        **Company Intelligence:** {company_intel.get('values_mission', 'I am deeply impressed by your commitment to the community.')}
        **Task:** Write a compelling three-paragraph cover letter.
        **Reasoning Framework:**
        1.  **Opening:** State the role and express enthusiasm for the company, referencing company intel.
        2.  **Body:** Connect skills to job requirements, integrating the "Most Relevant Career Example" to show a key achievement.
        3.  **Closing:** Reiterate interest and include a clear call to action.
        **Most Relevant Career Example:**
        ---
        {experience_snippet if experience_snippet else "The applicant has extensive experience in community services."}
        ---
        **Output Format:** Generate clean Markdown.
        """
        return self._generate_ai_content(prompt)

    def score_resume(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Scores a resume against a job description using AI."""
        prompt = f"""
        **Persona:** You are an expert ATS (Applicant Tracking System) and a senior recruiter for the Community Services sector.
        **Task:** Analyze the provided resume against the job description. Provide a match score and actionable feedback.
        **Reasoning Framework:**
        1.  **Keyword Analysis:** Extract key skills, qualifications, and duties from the job description.
        2.  **Resume Parsing:** Identify skills, experiences, and achievements in the resume.
        3.  **Alignment Scoring:** Calculate a percentage score based on how well the resume matches the key requirements. Score harshly.
        4.  **Feedback Generation:** Provide a list of strengths (what matched well) and a list of concrete suggestions for improvement.
        
        **Job Description:**
        ---
        {job_description}
        ---
        
        **Candidate's Resume:**
        ---
        {resume_text}
        ---
        
        **Output Format:**
        Return a single, valid JSON object only. Do not include any other text.
        {{
          "match_score": <integer_percentage>,
          "strengths": ["...", "..."],
          "suggestions": ["...", "..."]
        }}
        """
        response_text = self._generate_ai_content(prompt)
        try:
            json_str = response_text.strip().replace("```json", "").replace("```", "")
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Could not parse AI response.", "raw_text": response_text}
EOF

# --- main_app.py ---
cat <<'EOF' > main_app.py
# main_app.py
import streamlit as st
import database as db
import os
from api_clients import GeminiClient, PerplexityClient
from intelligence_booster import IntelligenceBoosterModule
from document_generator import DocumentGenerator

# --- Session State Management ---
if 'doc_type' not in st.session_state: st.session_state.doc_type = "Resume"
if 'job_desc' not in st.session_state: st.session_state.job_desc = ""
if 'company_name' not in st.session_state: st.session_state.company_name = ""
if 'role_title' not in st.session_state: st.session_state.role_title = ""
if 'generated_content' not in st.session_state: st.session_state.generated_content = None

st.set_page_config(page_title="Resume Agent", page_icon="🤖", layout="wide")
db.initialize_db()

def get_api_clients():
    gemini_key = st.session_state.get("gemini_api_key")
    perplexity_key = st.session_state.get("perplexity_api_key")
    
    if not gemini_key or not perplexity_key:
        st.sidebar.warning("Please enter your API keys in the Settings page.")
        return None, None
    try:
        gemini_client = GeminiClient(api_key=gemini_key)
        perplexity_client = PerplexityClient(api_key=perplexity_key)
        return gemini_client, perplexity_client
    except Exception as e:
        st.sidebar.error(f"Failed to initialize API clients: {e}")
        return None, None

st.title("🤖 Resume Agent")
st.write("Welcome! This tool helps you create tailored job application documents.")
st.info("Start by filling out your User Profile and Career History. Use the Style Analyzer to teach the agent your writing style.")

# --- Sidebar for Inputs ---
st.sidebar.header("Document Generation Controls")
st.session_state.doc_type = st.sidebar.selectbox("Select Document Type", ("Resume", "KSC Response", "Cover Letter"), key="doc_type_key")

st.sidebar.subheader("Job Details")
st.session_state.job_desc = st.sidebar.text_area("Or Paste Job Description / KSC", value=st.session_state.job_desc, height=200, key="job_desc_key", help="For KSC/Cover Letter, paste the text here. For Resumes, this is optional.")
st.session_state.company_name = st.sidebar.text_input("Company / Organization Name", value=st.session_state.company_name, key="company_name_key")
st.session_state.role_title = st.sidebar.text_input("Role Title", value=st.session_state.role_title, key="role_title_key")

if st.sidebar.button("✨ Generate Document", type="primary", use_container_width=True):
    gemini_client, perplexity_client = get_api_clients()
    if not all([gemini_client, perplexity_client]): st.stop()
    if not st.session_state.job_desc and st.session_state.doc_type != "Resume":
        st.sidebar.error("Please paste the job description or KSC question."); st.stop()

    with st.spinner("Processing..."):
        try:
            job_details = {"full_text": st.session_state.job_desc, "role_title": st.session_state.role_title}
            user_profile = db.get_user_profile() or {}
            experiences = db.get_all_experiences()
            
            # Get company intelligence
            intel_booster = IntelligenceBoosterModule(perplexity_client)
            company_intel = {}
            if st.session_state.company_name and st.session_state.role_title:
                company_intel = intel_booster.get_intelligence(st.session_state.company_name, st.session_state.role_title)
            
            doc_generator = DocumentGenerator(gemini_client)
            if st.session_state.doc_type == "Resume":
                markdown_content = doc_generator.generate_resume_markdown(user_profile, experiences)
            elif st.session_state.doc_type == "KSC Response":
                markdown_content = doc_generator.generate_ksc_response(st.session_state.job_desc, user_profile, experiences, company_intel, st.session_state.role_title).get('html')
            elif st.session_state.doc_type == "Cover Letter":
                markdown_content = doc_generator.generate_cover_letter_markdown(user_profile, experiences, job_details, company_intel)
            
            st.session_state.generated_content = {
                "html": markdown_content,
                "docx": doc_generator._create_docx_from_markdown(markdown_content),
                "pdf": doc_generator._create_pdf_from_markdown(markdown_content)
            }
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.generated_content = None

if st.session_state.generated_content:
    st.divider()
    st.header("Generated Document")
    content = st.session_state.generated_content
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download as DOCX", content.get("docx", b""), "generated_document.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    with col2:
        st.download_button("📥 Download as PDF", content.get("pdf", b""), "generated_document.pdf", "application/pdf")
    st.markdown("---")
    st.subheader("Preview")
    st.markdown(content.get("html", "<p>No content generated.</p>"), unsafe_allow_html=True)
EOF

# --- pages/0_User_Profile.py ---
cat <<'EOF' > pages/0_User_Profile.py
# pages/0_User_Profile.py
import streamlit as st
import database as db

st.set_page_config(page_title="User Profile", layout="wide")
st.title("👤 User Profile")
st.write("This information will be used to populate your documents. It is saved locally in your `career_history.db` file.")

profile = db.get_user_profile() or {}

with st.form(key="profile_form"):
    st.header("Contact Information")
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name", value=profile.get("full_name", "Nishant Jonas Dougall"))
        email = st.text_input("Email Address", value=profile.get("email", ""))
        phone = st.text_input("Phone Number", value=profile.get("phone", "+61412202666"))
    with col2:
        address = st.text_input("Address", value=profile.get("address", "Unit 2 418 high street, Northcote VICTORIA 3070, Australia"))
        linkedin_url = st.text_input("LinkedIn Profile URL", value=profile.get("linkedin_url", ""))
    st.header("Professional Summary")
    professional_summary = st.text_area("Summary / Personal Statement", value=profile.get("professional_summary", ""), height=150, placeholder="Write a brief 2-4 sentence summary of your career, skills, and goals.")
    
    style_profile_text = profile.get("style_profile", "")

    submit_button = st.form_submit_button("Save Profile")
    if submit_button:
        profile_data = {"full_name": full_name, "email": email, "phone": phone, "address": address, "linkedin_url": linkedin_url, "professional_summary": professional_summary, "style_profile": style_profile_text}
        db.save_user_profile(profile_data)
        st.toast("✅ Profile saved successfully!")
EOF

# --- pages/1_Manage_Career_History.py ---
cat <<'EOF' > pages/1_Manage_Career_History.py
# pages/1_Manage_Career_History.py
import streamlit as st
import database as db

st.set_page_config(page_title="Manage Career History", layout="wide")
st.title("📝 Manage Career History")
st.write("Add, edit, or delete your career examples here. These examples, including your 'gold standard' resume bullet points, will be used by the AI to tailor your job applications.")

st.header("Add or Edit Experience")
query_params = st.experimental_get_query_params()
edit_id = query_params.get("edit", [None])[0]
initial_data = {}
if edit_id:
    initial_data = db.get_experience_by_id(int(edit_id))
    if not initial_data:
        st.error("Experience not found."); edit_id = None

with st.form(key="experience_form", clear_on_submit=not edit_id):
    col1, col2, col3 = st.columns(3)
    with col1: title = st.text_input("Job Title", value=initial_data.get("title", ""), placeholder="e.g., Community Support Worker")
    with col2: company = st.text_input("Company / Organization", value=initial_data.get("company", ""), placeholder="e.g., Hope Services")
    with col3: dates = st.text_input("Dates of Employment", value=initial_data.get("dates", ""), placeholder="e.g., Jan 2022 - Present")
    st.subheader("STAR Method Example")
    situation = st.text_area("Situation", value=initial_data.get("situation", ""), placeholder="Describe the context or background.")
    task = st.text_area("Task", value=initial_data.get("task", ""), placeholder="What was your specific goal or responsibility?")
    action = st.text_area("Action", value=initial_data.get("action", ""), placeholder="What steps did you take?")
    result = st.text_area("Result", value=initial_data.get("result", ""), placeholder="What was the outcome? Use quantifiable data if possible.")
    skills = st.text_input("Related Skills (comma-separated)", value=initial_data.get("related_skills", ""), placeholder="e.g., crisis-intervention, client-advocacy")
    st.markdown("---")
    st.subheader("Gold Standard Resume Bullet Points")
    st.info("Add your best, pre-written resume bullet points for this experience (one per line). The AI will use these to build your resume.")
    resume_bullets = st.text_area("Resume Bullet Points", value=initial_data.get("resume_bullets", ""), height=150, placeholder="e.g., Achieved X by doing Y, resulting in Z.")
    submit_button = st.form_submit_button(label="Save Experience" if not edit_id else "Update Experience")
    if submit_button:
        if not all([title, company, dates, situation, task, action, result]):
            st.warning("Please fill out all fields.")
        else:
            if edit_id:
                db.update_experience(int(edit_id), title, company, dates, situation, task, action, result, skills, resume_bullets)
                st.toast("✅ Experience updated successfully!"); st.experimental_set_query_params()
            else:
                db.add_experience(title, company, dates, situation, task, action, result, skills, resume_bullets)
                st.toast("✅ Experience added successfully!")
            st.experimental_rerun()

st.header("Your Saved Experiences")
all_experiences = db.get_all_experiences()
if not all_experiences:
    st.info("You haven't added any experiences yet. Use the form above to get started.")
else:
    for exp in all_experiences:
        with st.expander(f"**{exp['title']} at {exp['company']}** (ID: {exp['id']})"):
            st.markdown(f"**Dates:** {exp['dates']}"); st.markdown(f"**Situation:** {exp['situation']}")
            st.markdown(f"**Task:** {exp['task']}"); st.markdown(f"**Action:** {exp['action']}")
            st.markdown(f"**Result:** {exp['result']}"); st.markdown(f"**Skills:** `{exp['related_skills']}`")
            if exp.get('resume_bullets'):
                st.markdown("**Resume Bullets:**"); st.code(exp['resume_bullets'], language='text')
            col1, col2 = st.columns([0.1, 1])
            with col1:
                if st.button("Edit", key=f"edit_{exp['id']}"):
                    st.experimental_set_query_params(edit=exp['id']); st.experimental_rerun()
            with col2:
                if st.button("Delete", key=f"delete_{exp['id']}", type="primary"):
                    db.delete_experience(exp['id']); st.experimental_rerun()
EOF

# --- pages/2_Settings.py ---
cat <<'EOF' > pages/2_Settings.py
# pages/2_Settings.py
import streamlit as st
import database as db
import os
import shutil

st.set_page_config(page_title="Settings", layout="centered")
st.title("⚙️ Settings")

# --- API Key Management ---
with st.expander("API Keys", expanded=True):
    st.write("Configure your API keys here. These keys are stored in the app's temporary session state and are not saved permanently.")
    st.markdown("""
    - Get your Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    - Get your Perplexity API Key from the [Perplexity Labs Platform](https://docs.perplexity.ai/docs/getting-started).
    """)
    gemini_key = st.text_input("Google Gemini API Key", type="password", value=st.session_state.get("gemini_api_key", ""))
    perplexity_key = st.text_input("Perplexity API Key", type="password", value=st.session_state.get("perplexity_api_key", ""))
    if st.button("Save Keys"):
        if gemini_key: st.session_state["gemini_api_key"] = gemini_key; st.toast("✅ Gemini API Key saved for this session.")
        else: st.warning("Please enter a Gemini API Key.")
        if perplexity_key: st.session_state["perplexity_api_key"] = perplexity_key; st.toast("✅ Perplexity API Key saved for this session.")
        else: st.warning("Please enter a Perplexity API Key.")
    st.info("Your keys are only stored for your current browser session.")

# --- Data Import/Export ---
with st.expander("Data Management", expanded=True):
    st.write("Download your entire career database as a backup, or upload a previous backup to restore your data.")
    
    col1, col2 = st.columns(2)
    with col1:
        try:
            with open(db.DB_FILE, "rb") as fp:
                st.download_button(
                    label="📥 Download Database",
                    data=fp,
                    file_name="career_history_backup.db",
                    mime="application/octet-stream"
                )
        except FileNotFoundError:
            st.info("No database file found to download. Add some data first.")

    with col2:
        uploaded_db = st.file_uploader("📤 Upload Database Backup", type=['db'])
        if uploaded_db is not None:
            if st.button("Restore Database"):
                with st.spinner("Restoring database..."):
                    # Create a backup of the current database before overwriting
                    if os.path.exists(db.DB_FILE):
                        shutil.copy(db.DB_FILE, f"{db.DB_FILE}.bak")
                    
                    # Write the new database file
                    with open(db.DB_FILE, "wb") as f:
                        f.write(uploaded_db.getbuffer())
                    st.success("Database restored successfully! The app will now reload.")
                    st.experimental_rerun()
    st.warning("Restoring will overwrite your current data. A backup of your current database will be created as `career_history.db.bak`.")
EOF

# --- pages/3_Job_Vault.py ---
cat <<'EOF' > pages/3_Job_Vault.py
# pages/3_Job_Vault.py
import streamlit as st
import database as db
from pd_scraper import PDScraperModule
from api_clients import GeminiClient
import json

st.set_page_config(page_title="Job Vault", layout="wide")
st.title("🏦 Job Vault")
st.write("Save job opportunities here by pasting a URL. The agent will scrape the content and summarize it for you.")

# --- Add New Job ---
st.header("Add New Job Opportunity")
new_job_url = st.text_input("Paste Job Ad URL here")
if st.button("Save and Scrape Job"):
    if new_job_url:
        job_id = db.add_job(new_job_url)
        if job_id:
            st.toast(f"Job from {new_job_url} saved! Now processing...")
            gemini_key = st.session_state.get("gemini_api_key")
            if not gemini_key: st.error("Gemini API key not set in Settings. Cannot summarize."); st.stop()
            gemini_client = GeminiClient(api_key=gemini_key)
            scraper_module = PDScraperModule(gemini_client)
            try:
                with st.spinner("Scraping and summarizing..."):
                    summary_data = scraper_module.process_url(new_job_url)
                    if "error" in summary_data: st.error(f"Failed: {summary_data['error']}")
                    else:
                        db.update_job_scrape_data(job_id, summary_data['full_text'])
                        db.update_job_summary(job_id, summary_data, summary_data.get('role_title', 'N/A'), summary_data.get('role_title', 'N/A'))
                        st.toast("✅ Scraping and summarization complete!"); st.experimental_rerun()
            except Exception as e: st.error(f"An error occurred during processing: {e}")
        else: st.warning("This URL has already been saved.")
    else: st.warning("Please enter a URL.")

# --- Display Saved Jobs ---
st.header("Saved Jobs")
all_jobs = db.get_all_saved_jobs()
if not all_jobs:
    st.info("You haven't saved any jobs yet. Use the form above to get started.")
else:
    for job in all_jobs:
        summary = json.loads(job['summary_json']) if job['summary_json'] else {}
        role_title = job.get('role_title') or summary.get('role_title', 'Processing...')
        company_name = job.get('company_name', 'Processing...')
        with st.expander(f"**{role_title}** at **{company_name}** (Status: {job['status']})"):
            st.markdown(f"**URL:** [{job['url']}]({job['url']})")
            if job['status'] == 'Summarized' and summary:
                st.markdown("**AI Summary:**"); st.markdown(f"**Key Responsibilities:**")
                for resp in summary.get('key_responsibilities', []): st.markdown(f"- {resp}")
                st.markdown(f"**Essential Skills:**")
                for skill in summary.get('essential_skills', []): st.markdown(f"- {skill}")
            elif job['status'] == 'Scraped': st.info("This job has been scraped but is awaiting summarization.")
            else: st.info("This job is saved and waiting to be processed.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load this Job", key=f"load_{job['id']}"):
                    st.session_state.job_desc = job.get('full_text', ''); st.session_state.company_name = job.get('company_name', ''); st.session_state.role_title = job.get('role_title', '')
                    st.toast(f"Loaded job '{role_title}' into the main generator. Navigate to 'Document Generator' to proceed.", icon='✅')
            with col2:
                if st.button("Delete Job", key=f"delete_{job['id']}", type="primary"):
                    db.delete_job(job['id']); st.experimental_rerun()
EOF

# --- pages/4_Style_Analyzer.py ---
cat <<'EOF' > pages/4_Style_Analyzer.py
# pages/4_Style_Analyzer.py
import streamlit as st
import database as db
from api_clients import GeminiClient
from file_parser import parse_files

st.set_page_config(page_title="Style Analyzer", layout="wide")
st.title("🎨 Style Analyzer")
st.write("Upload examples of your past resumes or cover letters. The agent will analyze them to learn your unique writing style, which will be used to make future generated documents sound more like you.")

# --- File Uploader ---
st.header("Upload Your Documents")
uploaded_files = st.file_uploader(
    "Choose one or more files (.pdf, .docx, .txt)",
    accept_multiple_files=True,
    type=['pdf', 'docx', 'txt']
)

if uploaded_files:
    if st.button("Analyze My Writing Style"):
        gemini_key = st.session_state.get("gemini_api_key")
        if not gemini_key:
            st.error("Please set your Gemini API key in the Settings page to use this feature.")
            st.stop()
        
        with st.spinner("Parsing files and analyzing your style..."):
            try:
                # 1. Parse files to get text
                combined_text = parse_files(uploaded_files)

                # 2. Send to AI for analysis
                gemini_client = GeminiClient(api_key=gemini_key)
                prompt = f"""
                **Persona:** You are an expert writing coach and linguistic analyst.
                **Task:** Analyze the following text from a user's past professional documents. Identify the key characteristics of their writing style.
                
                **Reasoning Framework:**
                1.  **Tone Analysis:** Is the tone formal, conversational, direct, academic, warm, etc.?
                2.  **Vocabulary:** What kind of action verbs are commonly used? Is the language simple or sophisticated?
                3.  **Sentence Structure:** Are sentences typically short and punchy, or long and detailed?
                4.  **Key Themes:** What are the recurring themes or values expressed (e.g., collaboration, efficiency, innovation)?
                
                **Documents Text:**
                ---
                {combined_text[:8000]}
                ---
                
                **Output Format:**
                Provide a concise summary of the user's writing style in 3-4 bullet points. This summary will be used as a style guide for the AI.
                """
                style_profile = gemini_client.generate_text(prompt)

                # 3. Save the style profile to the database
                db.save_style_profile(style_profile)
                st.success("✅ Your writing style has been analyzed and saved!")
                st.balloons()
                st.subheader("Your Personal Style Profile:")
                st.markdown(style_profile)

            except Exception as e:
                st.error(f"An error occurred during style analysis: {e}")

# Display current style profile
st.header("Current Style Profile")
profile = db.get_user_profile()
if profile and profile.get("style_profile"):
    st.markdown(profile["style_profile"])
else:
    st.info("No style profile has been generated yet. Upload some documents to create one.")
EOF

# --- pages/5_Resume_Scorer.py ---
cat <<'EOF' > pages/5_Resume_Scorer.py
# pages/5_Resume_Scorer.py
import streamlit as st
import database as db
from api_clients import GeminiClient
from file_parser import parse_files, parse_pdf, parse_docx
import io
import json

st.set_page_config(page_title="Resume Scorer", layout="wide")
st.title("🎯 Resume Scorer")
st.write("Upload your final resume and the job description to get an AI-powered match score and actionable feedback.")

# --- Inputs ---
st.header("Inputs")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Final Resume")
    resume_file = st.file_uploader("Upload your resume (.pdf, .docx, .txt)", type=['pdf', 'docx', 'txt'])

with col2:
    st.subheader("Target Job Description")
    job_desc_text = st.text_area("Paste the full job description here", height=250)

if st.button("Score My Resume", type="primary", disabled=not (resume_file and job_desc_text)):
    gemini_key = st.session_state.get("gemini_api_key")
    if not gemini_key:
        st.error("Please set your Gemini API key in the Settings page to use this feature.")
        st.stop()

    with st.spinner("Parsing documents and scoring your resume..."):
        try:
            # 1. Parse resume file
            if resume_file.type == "application/pdf":
                resume_text = parse_pdf(io.BytesIO(resume_file.getvalue()))
            elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = parse_docx(io.BytesIO(resume_file.getvalue()))
            else:
                resume_text = resume_file.getvalue().decode("utf-8")

            # 2. Get score from AI
            from document_generator import DocumentGenerator
            doc_generator = DocumentGenerator(GeminiClient(api_key=gemini_key))
            score_data = doc_generator.score_resume(resume_text, job_desc_text)

            # 3. Display results
            if "error" in score_data:
                st.error(f"Scoring failed: {score_data['error']}")
                st.code(score_data.get('raw_text'))
            else:
                st.header("📊 Scoring Results")
                score = score_data.get("match_score", 0)
                
                # Display score with a progress bar and color
                st.subheader(f"Overall Match Score: {score}%")
                progress_color = "red"
                if score > 75: progress_color = "green"
                elif score > 50: progress_color = "orange"
                st.progress(score / 100)

                st.subheader("✅ Strengths")
                for strength in score_data.get("strengths", []):
                    st.markdown(f"- {strength}")

                st.subheader("💡 Suggestions for Improvement")
                for suggestion in score_data.get("suggestions", []):
                    st.markdown(f"- {suggestion}")

        except Exception as e:
            st.error(f"An error occurred during scoring: {e}")
EOF

echo "\n✅ Project setup complete!"
echo "\nNext steps:"
echo "1. Create and activate a Python virtual environment:"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo "2. Install the required packages:"
echo "   pip install -r requirements.txt"
echo "3. Run the Streamlit application:"
echo "   streamlit run main_app.py\n"

   python3 -m venv venv"#!/bin/bash
# Setup script for the Resume Agent 2.0 (Final Polished Edition)
# This script creates the full project structure and files.

echo "Setting up project structure..."
mkdir -p pages

echo "Creating project files..."

# --- requirements.txt ---
cat <<'EOF' > requirements.txt
# Core dependencies
streamlit
google-generativeai
httpx
cachetools
tenacity
markdown

# Document processing
python-docx
pypdf
weasyprint

# Web Scraping & Semantic Search
playwright
beautifulsoup4
sentence-transformers
scikit-learn

# Utilities
python-dotenv
EOF

# --- database.py ---
cat <<'EOF' > database.py
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
    conn = get_db_connection()
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
    conn.close()

# --- Career History Functions (Omitted for brevity) ---
def add_experience(title: str, company: str, dates: str, situation: str, task: str, action: str, result: str, skills: str, bullets: str):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("INSERT INTO career_history (title, company, dates, situation, task, action, result, related_skills, resume_bullets) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (title, company, dates, situation, task, action, result, skills, bullets))
    conn.commit(); conn.close()
def get_all_experiences() -> List[Dict[str, Any]]:
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT * FROM career_history ORDER BY id DESC")
    experiences = [dict(row) for row in cursor.fetchall()]; conn.close(); return experiences
def get_experience_by_id(exp_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT * FROM career_history WHERE id = ?", (exp_id,)); experience = cursor.fetchone(); conn.close(); return dict(experience) if experience else None
def update_experience(exp_id: int, title: str, company: str, dates: str, situation: str, task: str, action: str, result: str, skills: str, bullets: str):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("UPDATE career_history SET title = ?, company = ?, dates = ?, situation = ?, task = ?, action = ?, result = ?, related_skills = ?, resume_bullets = ? WHERE id = ?", (title, company, dates, situation, task, action, result, skills, bullets, exp_id))
    conn.commit(); conn.close()
def delete_experience(exp_id: int):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("DELETE FROM career_history WHERE id = ?", (exp_id,)); conn.commit(); conn.close()

# --- User Profile Functions ---
def save_user_profile(profile_data: Dict[str, str]):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO user_profile (id, full_name, email, phone, address, linkedin_url, professional_summary, style_profile) VALUES (1, ?, ?, ?, ?, ?, ?, ?)", (profile_data.get('full_name'), profile_data.get('email'), profile_data.get('phone'), profile_data.get('address'), profile_data.get('linkedin_url'), profile_data.get('professional_summary'), profile_data.get('style_profile')))
    conn.commit(); conn.close()
def get_user_profile() -> Optional[Dict[str, Any]]:
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_profile WHERE id = 1"); profile = cursor.fetchone(); conn.close(); return dict(profile) if profile else None
def save_style_profile(style_profile_text: str):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO user_profile (id) VALUES (1)")
    cursor.execute("UPDATE user_profile SET style_profile = ? WHERE id = 1", (style_profile_text,))
    conn.commit(); conn.close()

# --- Saved Jobs Functions (Omitted for brevity) ---
def add_job(url: str) -> Optional[int]:
    conn = get_db_connection(); cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO saved_jobs (url, status) VALUES (?, 'Saved')", (url,)); conn.commit(); job_id = cursor.lastrowid
    except sqlite3.IntegrityError: job_id = None
    finally: conn.close()
    return job_id
def update_job_scrape_data(job_id: int, full_text: str):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("UPDATE saved_jobs SET full_text = ?, status = 'Scraped' WHERE id = ?", (full_text, job_id)); conn.commit(); conn.close()
def update_job_summary(job_id: int, summary: Dict, company_name: str, role_title: str):
    conn = get_db_connection(); cursor = conn.cursor(); summary_text = json.dumps(summary)
    cursor.execute("UPDATE saved_jobs SET summary_json = ?, company_name = ?, role_title = ?, status = 'Summarized' WHERE id = ?", (summary_text, company_name, role_title, job_id))
    conn.commit(); conn.close()
def get_all_saved_jobs() -> List[Dict[str, Any]]:
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT * FROM saved_jobs ORDER BY saved_at DESC"); jobs = [dict(row) for row in cursor.fetchall()]; conn.close(); return jobs
def delete_job(job_id: int):
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("DELETE FROM saved_jobs WHERE id = ?", (job_id,)); conn.commit(); conn.close()
EOF

# --- api_clients.py ---
cat <<'EOF' > api_clients.py
# api_clients.py
"""
Centralized clients for interacting with external APIs like Google Gemini and Perplexity.
Includes robust error handling and automatic retries.
"""
import google.generativeai as genai
import httpx
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

class GeminiClient:
    """Client for interacting with the Google Gemini API."""
    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def generate_text(self, prompt: str) -> str:
        """Generates text content from a given prompt with retry logic."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            raise

class PerplexityClient:
    """Client for interacting with the Perplexity Sonar API."""
    def __init__(self, api_key: Optional[str] = None):
        if not api_key:
            raise ValueError("Perplexity API key is required.")
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def search(self, query: str) -> Dict[str, Any]:
        """Performs a search using the Perplexity API with retry logic."""
        payload = {
            "model": "sonar-small-online",
            "messages": [{"role": "user", "content": query}]
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting from Perplexity: {e}")
            raise
        except httpx.HTTPStatusError as e:
            print(f"Perplexity API returned an error: {e.response.status_code} - {e.response.text}")
            raise
EOF

# --- intelligence_booster.py ---
cat <<'EOF' > intelligence_booster.py
# intelligence_booster.py
"""
Module to enrich context by gathering live intelligence about a company.
"""
from cachetools import TTLCache
from api_clients import PerplexityClient
from typing import Dict

class QueryGenerator:
    """Generates targeted questions for the Perplexity API."""
    def generate_for_company(self, company_name: str, role_title: str) -> Dict[str, str]:
        return {
            "values_mission": f"What are the publicly stated values, mission, or vision of the organization '{company_name}'?",
            "recent_news": f"Summarize recent news, projects, or developments for '{company_name}' in the last 6 months.",
            "role_context": f"What are the typical challenges or objectives for a '{role_title}' within the community services or social work sector in Australia?"
        }

class IntelligenceBoosterModule:
    """Facade for the Company Researcher, orchestrating queries and caching."""
    def __init__(self, perplexity_client: PerplexityClient):
        self.client = perplexity_client
        self.query_generator = QueryGenerator()
        self.cache = TTLCache(maxsize=100, ttl=86400)

    def get_intelligence(self, company_name: str, role_title: str) -> Dict[str, str]:
        """Main method to fetch and structure company intelligence."""
        cache_key = f"{company_name}_{role_title}".lower()
        if cache_key in self.cache:
            return self.cache[cache_key]

        queries = self.query_generator.generate_for_company(company_name, role_title)
        intelligence = {}

        for key, query in queries.items():
            try:
                response = self.client.search(query)
                intelligence[key] = response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Could not fetch intelligence for query '{query}': {e}")
                intelligence[key] = f"Error fetching data: {e}"

        self.cache[cache_key] = intelligence
        return intelligence
EOF

# --- file_parser.py ---
cat <<'EOF' > file_parser.py
# file_parser.py
"""
A utility module to parse text content from different file types like PDF and DOCX.
"""
import io
from typing import List
from docx import Document
from pypdf import PdfReader
import streamlit as st

def parse_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """Parses a list of uploaded files and returns their combined text content."""
    full_text = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            text = parse_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = parse_docx(file)
        elif file.type == "text/plain":
            text = file.getvalue().decode("utf-8")
        else:
            text = f"Unsupported file type: {file.name}"
        full_text.append(f"--- Document: {file.name} ---\n{text}\n\n")
    return "".join(full_text)

def parse_pdf(file: io.BytesIO) -> str:
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error parsing PDF: {e}"

def parse_docx(file: io.BytesIO) -> str:
    """Extracts text from an uploaded DOCX file."""
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error parsing DOCX: {e}"
EOF

# --- document_generator.py ---
cat <<'EOF' > document_generator.py
# document_generator.py
"""
Handles the core logic of generating documents as structured Markdown text.
This version includes improved export quality for DOCX and PDF.
"""
from api_clients import GeminiClient
from typing import List, Dict, Any
from docx import Document
from docx.shared import Pt
from weasyprint import HTML, CSS
import io
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import markdown
import json

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def find_relevant_experiences(question: str, experiences: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    if not experiences: return []
    experience_texts = [f"{exp['title']}. {exp['situation']} {exp['task']} {exp['action']} {exp['result']}" for exp in experiences]
    question_embedding = embedding_model.encode([question])
    experience_embeddings = embedding_model.encode(experience_texts)
    similarities = cosine_similarity(question_embedding, experience_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [experiences[i] for i in top_indices]

class DocumentGenerator:
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client

    def _generate_ai_content(self, prompt: str) -> str:
        return self.gemini_client.generate_text(prompt)

    def _create_docx_from_markdown(self, markdown_content: str) -> bytes:
        """Creates a formatted DOCX from Markdown content."""
        doc = Document()
        for line in markdown_content.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                doc.add_heading(line[2:], level=1)
            elif line.startswith('## '):
                doc.add_heading(line[3:], level=2)
            elif line.startswith('### '):
                p = doc.add_paragraph()
                p.add_run(line[4:]).bold = True
            elif line.startswith('- '):
                doc.add_paragraph(line[2:], style='List Bullet')
            elif line:
                doc.add_paragraph(line)
        
        bio = io.BytesIO()
        doc.save(bio)
        return bio.getvalue()

    def _create_pdf_from_markdown(self, markdown_content: str) -> bytes:
        """Creates a styled PDF from Markdown content."""
        html_content = markdown.markdown(markdown_content)
        # Professional styling for the PDF
        css = CSS(string='''
            @page { size: A4; margin: 2cm; }
            body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.5; }
            h1 { font-size: 24pt; color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 5px; }
            h2 { font-size: 16pt; color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 3px; margin-top: 25px; }
            h3 { font-size: 12pt; color: #34495e; font-weight: bold; }
            ul { list-style-type: disc; }
        ''')
        return HTML(string=html_content).write_pdf(stylesheets=[css])

    def generate_resume_markdown(self, user_profile: Dict, experiences: List[Dict[str, Any]]) -> str:
        md_parts = [f"# {user_profile.get('full_name', 'Your Name')}"]
        contact_info = " | ".join(filter(None, [user_profile.get('phone'), user_profile.get('email'), user_profile.get('address'), user_profile.get('linkedin_url')]))
        md_parts.append(contact_info)
        if user_profile.get('professional_summary'):
            md_parts.append("\n## PROFESSIONAL SUMMARY")
            md_parts.append(f"{user_profile.get('professional_summary')}")
        if experiences:
            md_parts.append("\n## PROFESSIONAL EXPERIENCE")
            for exp in experiences:
                md_parts.append(f"\n### {exp.get('title')}")
                md_parts.append(f"**{exp.get('company')}** | *{exp.get('dates')}*")
                bullets = exp.get('resume_bullets', '').split('\n')
                for bullet in bullets:
                    if bullet.strip(): md_parts.append(f"- {bullet.strip()}")
        return "\n".join(md_parts)

    def generate_ksc_response(self, ksc_question: str, user_profile: Dict, experiences: List[Dict[str, Any]], company_intel: Dict[str, str], role_title: str) -> Dict[str, Any]:
        relevant_experiences = find_relevant_experiences(ksc_question, experiences)
        experience_text = "\n\n".join([f"Title: {exp['title']}\nSituation: {exp['situation']}\nTask: {exp['task']}\nAction: {exp['action']}\nResult: {exp['result']}" for exp in relevant_experiences])
        prompt = f"""
        **Persona:** You are an expert career coach for the Australian Community Services sector. Your tone is professional and authentic, mirroring the user's personal style if provided.
        **User's Personal Style Profile:** {user_profile.get('style_profile', 'N/A')}
        **Company Intelligence:** {company_intel.get('values_mission', 'N/A')}
        **Task:** Write a compelling KSC response to the question below.
        **KSC Question:** "{ksc_question}"
        **Reasoning Framework:**
        1.  **Deconstruct:** Identify the core competency.
        2.  **Select Evidence:** Choose the strongest parts of the STAR stories to prove this competency.
        3.  **Draft:** Structure the response using the STAR method, adopting the user's personal style and aligning with company intelligence.
        **Candidate's Most Relevant Career Examples:**
        ---
        {experience_text if experience_text else "No specific examples provided."}
        ---
        **Output Format:** Generate clean Markdown.
        """
        markdown_content = self._generate_ai_content(prompt)
        return {"html": markdown_content, "docx": self._create_docx_from_markdown(markdown_content), "pdf": self._create_pdf_from_markdown(markdown_content)}

    def generate_cover_letter_markdown(self, user_profile: Dict, experiences: List[Dict[str, Any]], job_details: Dict[str, Any], company_intel: Dict[str, Any]) -> str:
        most_relevant_experience = find_relevant_experiences(job_details.get('full_text', ''), experiences, top_k=1)
        experience_snippet = ""
        if most_relevant_experience:
            exp = most_relevant_experience[0]
            experience_snippet = f"In my role as a {exp['title']} at {exp['company']}, I was responsible for {exp['task']}. I successfully {exp['action']}, which resulted in {exp['result']}."
        prompt = f"""
        **Persona:** You are an expert career advisor, writing a cover letter for the Australian Community Services sector. Your tone is professional and warm, mirroring the user's personal style if provided.
        **User's Personal Style Profile:** {user_profile.get('style_profile', 'N/A')}
        **Company Intelligence:** {company_intel.get('values_mission', 'I am deeply impressed by your commitment to the community.')}
        **Task:** Write a compelling three-paragraph cover letter.
        **Reasoning Framework:**
        1.  **Opening:** State the role and express enthusiasm for the company, referencing company intel.
        2.  **Body:** Connect skills to job requirements, integrating the "Most Relevant Career Example" to show a key achievement.
        3.  **Closing:** Reiterate interest and include a clear call to action.
        **Most Relevant Career Example:**
        ---
        {experience_snippet if experience_snippet else "The applicant has extensive experience in community services."}
        ---
        **Output Format:** Generate clean Markdown.
        """
        return self._generate_ai_content(prompt)

    def score_resume(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Scores a resume against a job description using AI."""
        prompt = f"""
        **Persona:** You are an expert ATS (Applicant Tracking System) and a senior recruiter for the Community Services sector.
        **Task:** Analyze the provided resume against the job description. Provide a match score and actionable feedback.
        **Reasoning Framework:**
        1.  **Keyword Analysis:** Extract key skills, qualifications, and duties from the job description.
        2.  **Resume Parsing:** Identify skills, experiences, and achievements in the resume.
        3.  **Alignment Scoring:** Calculate a percentage score based on how well the resume matches the key requirements. Score harshly.
        4.  **Feedback Generation:** Provide a list of strengths (what matched well) and a list of concrete suggestions for improvement.
        
        **Job Description:**
        ---
        {job_description}
        ---
        
        **Candidate's Resume:**
        ---
        {resume_text}
        ---
        
        **Output Format:**
        Return a single, valid JSON object only. Do not include any other text.
        {{
          "match_score": <integer_percentage>,
          "strengths": ["...", "..."],
          "suggestions": ["...", "..."]
        }}
        """
        response_text = self._generate_ai_content(prompt)
        try:
            json_str = response_text.strip().replace("```json", "").replace("```", "")
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Could not parse AI response.", "raw_text": response_text}
EOF

# --- main_app.py ---
cat <<'EOF' > main_app.py
# main_app.py
import streamlit as st
import database as db
import os
from api_clients import GeminiClient, PerplexityClient
from intelligence_booster import IntelligenceBoosterModule
from document_generator import DocumentGenerator

# --- Session State Management ---
if 'doc_type' not in st.session_state: st.session_state.doc_type = "Resume"
if 'job_desc' not in st.session_state: st.session_state.job_desc = ""
if 'company_name' not in st.session_state: st.session_state.company_name = ""
if 'role_title' not in st.session_state: st.session_state.role_title = ""
if 'generated_content' not in st.session_state: st.session_state.generated_content = None

st.set_page_config(page_title="Resume Agent", page_icon="🤖", layout="wide")
db.initialize_db()

def get_api_clients():
    gemini_key = st.session_state.get("gemini_api_key")
    perplexity_key = st.session_state.get("perplexity_api_key")
    
    if not gemini_key or not perplexity_key:
        st.sidebar.warning("Please enter your API keys in the Settings page.")
        return None, None
    try:
        gemini_client = GeminiClient(api_key=gemini_key)
        perplexity_client = PerplexityClient(api_key=perplexity_key)
        return gemini_client, perplexity_client
    except Exception as e:
        st.sidebar.error(f"Failed to initialize API clients: {e}")
        return None, None

st.title("🤖 Resume Agent")
st.write("Welcome! This tool helps you create tailored job application documents.")
st.info("Start by filling out your User Profile and Career History. Use the Style Analyzer to teach the agent your writing style.")

# --- Sidebar for Inputs ---
st.sidebar.header("Document Generation Controls")
st.session_state.doc_type = st.sidebar.selectbox("Select Document Type", ("Resume", "KSC Response", "Cover Letter"), key="doc_type_key")

st.sidebar.subheader("Job Details")
st.session_state.job_desc = st.sidebar.text_area("Or Paste Job Description / KSC", value=st.session_state.job_desc, height=200, key="job_desc_key", help="For KSC/Cover Letter, paste the text here. For Resumes, this is optional.")
st.session_state.company_name = st.sidebar.text_input("Company / Organization Name", value=st.session_state.company_name, key="company_name_key")
st.session_state.role_title = st.sidebar.text_input("Role Title", value=st.session_state.role_title, key="role_title_key")

if st.sidebar.button("✨ Generate Document", type="primary", use_container_width=True):
    gemini_client, perplexity_client = get_api_clients()
    if not all([gemini_client, perplexity_client]): st.stop()
    if not st.session_state.job_desc and st.session_state.doc_type != "Resume":
        st.sidebar.error("Please paste the job description or KSC question."); st.stop()

    with st.spinner("Processing..."):
        try:
            job_details = {"full_text": st.session_state.job_desc, "role_title": st.session_state.role_title}
            user_profile = db.get_user_profile() or {}
            experiences = db.get_all_experiences()
            
            # Get company intelligence
            intel_booster = IntelligenceBoosterModule(perplexity_client)
            company_intel = {}
            if st.session_state.company_name and st.session_state.role_title:
                company_intel = intel_booster.get_intelligence(st.session_state.company_name, st.session_state.role_title)
            
            doc_generator = DocumentGenerator(gemini_client)
            if st.session_state.doc_type == "Resume":
                markdown_content = doc_generator.generate_resume_markdown(user_profile, experiences)
            elif st.session_state.doc_type == "KSC Response":
                markdown_content = doc_generator.generate_ksc_response(st.session_state.job_desc, user_profile, experiences, company_intel, st.session_state.role_title).get('html')
            elif st.session_state.doc_type == "Cover Letter":
                markdown_content = doc_generator.generate_cover_letter_markdown(user_profile, experiences, job_details, company_intel)
            
            st.session_state.generated_content = {
                "html": markdown_content,
                "docx": doc_generator._create_docx_from_markdown(markdown_content),
                "pdf": doc_generator._create_pdf_from_markdown(markdown_content)
            }
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.generated_content = None

if st.session_state.generated_content:
    st.divider()
    st.header("Generated Document")
    content = st.session_state.generated_content
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download as DOCX", content.get("docx", b""), "generated_document.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    with col2:
        st.download_button("📥 Download as PDF", content.get("pdf", b""), "generated_document.pdf", "application/pdf")
    st.markdown("---")
    st.subheader("Preview")
    st.markdown(content.get("html", "<p>No content generated.</p>"), unsafe_allow_html=True)
EOF

# --- pages/0_User_Profile.py ---
cat <<'EOF' > pages/0_User_Profile.py
# pages/0_User_Profile.py
import streamlit as st
import database as db

st.set_page_config(page_title="User Profile", layout="wide")
st.title("👤 User Profile")
st.write("This information will be used to populate your documents. It is saved locally in your `career_history.db` file.")

profile = db.get_user_profile() or {}

with st.form(key="profile_form"):
    st.header("Contact Information")
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name", value=profile.get("full_name", "Nishant Jonas Dougall"))
        email = st.text_input("Email Address", value=profile.get("email", ""))
        phone = st.text_input("Phone Number", value=profile.get("phone", "+61412202666"))
    with col2:
        address = st.text_input("Address", value=profile.get("address", "Unit 2 418 high street, Northcote VICTORIA 3070, Australia"))
        linkedin_url = st.text_input("LinkedIn Profile URL", value=profile.get("linkedin_url", ""))
    st.header("Professional Summary")
    professional_summary = st.text_area("Summary / Personal Statement", value=profile.get("professional_summary", ""), height=150, placeholder="Write a brief 2-4 sentence summary of your career, skills, and goals.")
    
    style_profile_text = profile.get("style_profile", "")

    submit_button = st.form_submit_button("Save Profile")
    if submit_button:
        profile_data = {"full_name": full_name, "email": email, "phone": phone, "address": address, "linkedin_url": linkedin_url, "professional_summary": professional_summary, "style_profile": style_profile_text}
        db.save_user_profile(profile_data)
        st.toast("✅ Profile saved successfully!")
EOF

# --- pages/1_Manage_Career_History.py ---
cat <<'EOF' > pages/1_Manage_Career_History.py
# pages/1_Manage_Career_History.py
import streamlit as st
import database as db

st.set_page_config(page_title="Manage Career History", layout="wide")
st.title("📝 Manage Career History")
st.write("Add, edit, or delete your career examples here. These examples, including your 'gold standard' resume bullet points, will be used by the AI to tailor your job applications.")

st.header("Add or Edit Experience")
query_params = st.experimental_get_query_params()
edit_id = query_params.get("edit", [None])[0]
initial_data = {}
if edit_id:
    initial_data = db.get_experience_by_id(int(edit_id))
    if not initial_data:
        st.error("Experience not found."); edit_id = None

with st.form(key="experience_form", clear_on_submit=not edit_id):
    col1, col2, col3 = st.columns(3)
    with col1: title = st.text_input("Job Title", value=initial_data.get("title", ""), placeholder="e.g., Community Support Worker")
    with col2: company = st.text_input("Company / Organization", value=initial_data.get("company", ""), placeholder="e.g., Hope Services")
    with col3: dates = st.text_input("Dates of Employment", value=initial_data.get("dates", ""), placeholder="e.g., Jan 2022 - Present")
    st.subheader("STAR Method Example")
    situation = st.text_area("Situation", value=initial_data.get("situation", ""), placeholder="Describe the context or background.")
    task = st.text_area("Task", value=initial_data.get("task", ""), placeholder="What was your specific goal or responsibility?")
    action = st.text_area("Action", value=initial_data.get("action", ""), placeholder="What steps did you take?")
    result = st.text_area("Result", value=initial_data.get("result", ""), placeholder="What was the outcome? Use quantifiable data if possible.")
    skills = st.text_input("Related Skills (comma-separated)", value=initial_data.get("related_skills", ""), placeholder="e.g., crisis-intervention, client-advocacy")
    st.markdown("---")
    st.subheader("Gold Standard Resume Bullet Points")
    st.info("Add your best, pre-written resume bullet points for this experience (one per line). The AI will use these to build your resume.")
    resume_bullets = st.text_area("Resume Bullet Points", value=initial_data.get("resume_bullets", ""), height=150, placeholder="e.g., Achieved X by doing Y, resulting in Z.")
    submit_button = st.form_submit_button(label="Save Experience" if not edit_id else "Update Experience")
    if submit_button:
        if not all([title, company, dates, situation, task, action, result]):
            st.warning("Please fill out all fields.")
        else:
            if edit_id:
                db.update_experience(int(edit_id), title, company, dates, situation, task, action, result, skills, resume_bullets)
                st.toast("✅ Experience updated successfully!"); st.experimental_set_query_params()
            else:
                db.add_experience(title, company, dates, situation, task, action, result, skills, resume_bullets)
                st.toast("✅ Experience added successfully!")
            st.experimental_rerun()

st.header("Your Saved Experiences")
all_experiences = db.get_all_experiences()
if not all_experiences:
    st.info("You haven't added any experiences yet. Use the form above to get started.")
else:
    for exp in all_experiences:
        with st.expander(f"**{exp['title']} at {exp['company']}** (ID: {exp['id']})"):
            st.markdown(f"**Dates:** {exp['dates']}"); st.markdown(f"**Situation:** {exp['situation']}")
            st.markdown(f"**Task:** {exp['task']}"); st.markdown(f"**Action:** {exp['action']}")
            st.markdown(f"**Result:** {exp['result']}"); st.markdown(f"**Skills:** `{exp['related_skills']}`")
            if exp.get('resume_bullets'):
                st.markdown("**Resume Bullets:**"); st.code(exp['resume_bullets'], language='text')
            col1, col2 = st.columns([0.1, 1])
            with col1:
                if st.button("Edit", key=f"edit_{exp['id']}"):
                    st.experimental_set_query_params(edit=exp['id']); st.experimental_rerun()
            with col2:
                if st.button("Delete", key=f"delete_{exp['id']}", type="primary"):
                    db.delete_experience(exp['id']); st.experimental_rerun()
EOF

# --- pages/2_Settings.py ---
cat <<'EOF' > pages/2_Settings.py
# pages/2_Settings.py
import streamlit as st
import database as db
import os
import shutil

st.set_page_config(page_title="Settings", layout="centered")
st.title("⚙️ Settings")

# --- API Key Management ---
with st.expander("API Keys", expanded=True):
    st.write("Configure your API keys here. These keys are stored in the app's temporary session state and are not saved permanently.")
    st.markdown("""
    - Get your Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    - Get your Perplexity API Key from the [Perplexity Labs Platform](https://docs.perplexity.ai/docs/getting-started).
    """)
    gemini_key = st.text_input("Google Gemini API Key", type="password", value=st.session_state.get("gemini_api_key", ""))
    perplexity_key = st.text_input("Perplexity API Key", type="password", value=st.session_state.get("perplexity_api_key", ""))
    if st.button("Save Keys"):
        if gemini_key: st.session_state["gemini_api_key"] = gemini_key; st.toast("✅ Gemini API Key saved for this session.")
        else: st.warning("Please enter a Gemini API Key.")
        if perplexity_key: st.session_state["perplexity_api_key"] = perplexity_key; st.toast("✅ Perplexity API Key saved for this session.")
        else: st.warning("Please enter a Perplexity API Key.")
    st.info("Your keys are only stored for your current browser session.")

# --- Data Import/Export ---
with st.expander("Data Management", expanded=True):
    st.write("Download your entire career database as a backup, or upload a previous backup to restore your data.")
    
    col1, col2 = st.columns(2)
    with col1:
        try:
            with open(db.DB_FILE, "rb") as fp:
                st.download_button(
                    label="📥 Download Database",
                    data=fp,
                    file_name="career_history_backup.db",
                    mime="application/octet-stream"
                )
        except FileNotFoundError:
            st.info("No database file found to download. Add some data first.")

    with col2:
        uploaded_db = st.file_uploader("📤 Upload Database Backup", type=['db'])
        if uploaded_db is not None:
            if st.button("Restore Database"):
                with st.spinner("Restoring database..."):
                    # Create a backup of the current database before overwriting
                    if os.path.exists(db.DB_FILE):
                        shutil.copy(db.DB_FILE, f"{db.DB_FILE}.bak")
                    
                    # Write the new database file
                    with open(db.DB_FILE, "wb") as f:
                        f.write(uploaded_db.getbuffer())
                    st.success("Database restored successfully! The app will now reload.")
                    st.experimental_rerun()
    st.warning("Restoring will overwrite your current data. A backup of your current database will be created as `career_history.db.bak`.")
EOF

# --- pages/3_Job_Vault.py ---
cat <<'EOF' > pages/3_Job_Vault.py
# pages/3_Job_Vault.py
import streamlit as st
import database as db
from pd_scraper import PDScraperModule
from api_clients import GeminiClient
import json

st.set_page_config(page_title="Job Vault", layout="wide")
st.title("🏦 Job Vault")
st.write("Save job opportunities here by pasting a URL. The agent will scrape the content and summarize it for you.")

# --- Add New Job ---
st.header("Add New Job Opportunity")
new_job_url = st.text_input("Paste Job Ad URL here")
if st.button("Save and Scrape Job"):
    if new_job_url:
        job_id = db.add_job(new_job_url)
        if job_id:
            st.toast(f"Job from {new_job_url} saved! Now processing...")
            gemini_key = st.session_state.get("gemini_api_key")
            if not gemini_key: st.error("Gemini API key not set in Settings. Cannot summarize."); st.stop()
            gemini_client = GeminiClient(api_key=gemini_key)
            scraper_module = PDScraperModule(gemini_client)
            try:
                with st.spinner("Scraping and summarizing..."):
                    summary_data = scraper_module.process_url(new_job_url)
                    if "error" in summary_data: st.error(f"Failed: {summary_data['error']}")
                    else:
                        db.update_job_scrape_data(job_id, summary_data['full_text'])
                        db.update_job_summary(job_id, summary_data, summary_data.get('role_title', 'N/A'), summary_data.get('role_title', 'N/A'))
                        st.toast("✅ Scraping and summarization complete!"); st.experimental_rerun()
            except Exception as e: st.error(f"An error occurred during processing: {e}")
        else: st.warning("This URL has already been saved.")
    else: st.warning("Please enter a URL.")

# --- Display Saved Jobs ---
st.header("Saved Jobs")
all_jobs = db.get_all_saved_jobs()
if not all_jobs:
    st.info("You haven't saved any jobs yet. Use the form above to get started.")
else:
    for job in all_jobs:
        summary = json.loads(job['summary_json']) if job['summary_json'] else {}
        role_title = job.get('role_title') or summary.get('role_title', 'Processing...')
        company_name = job.get('company_name', 'Processing...')
        with st.expander(f"**{role_title}** at **{company_name}** (Status: {job['status']})"):
            st.markdown(f"**URL:** [{job['url']}]({job['url']})")
            if job['status'] == 'Summarized' and summary:
                st.markdown("**AI Summary:**"); st.markdown(f"**Key Responsibilities:**")
                for resp in summary.get('key_responsibilities', []): st.markdown(f"- {resp}")
                st.markdown(f"**Essential Skills:**")
                for skill in summary.get('essential_skills', []): st.markdown(f"- {skill}")
            elif job['status'] == 'Scraped': st.info("This job has been scraped but is awaiting summarization.")
            else: st.info("This job is saved and waiting to be processed.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load this Job", key=f"load_{job['id']}"):
                    st.session_state.job_desc = job.get('full_text', ''); st.session_state.company_name = job.get('company_name', ''); st.session_state.role_title = job.get('role_title', '')
                    st.toast(f"Loaded job '{role_title}' into the main generator. Navigate to 'Document Generator' to proceed.", icon='✅')
            with col2:
                if st.button("Delete Job", key=f"delete_{job['id']}", type="primary"):
                    db.delete_job(job['id']); st.experimental_rerun()
EOF

# --- pages/4_Style_Analyzer.py ---
cat <<'EOF' > pages/4_Style_Analyzer.py
# pages/4_Style_Analyzer.py
import streamlit as st
import database as db
from api_clients import GeminiClient
from file_parser import parse_files

st.set_page_config(page_title="Style Analyzer", layout="wide")
st.title("🎨 Style Analyzer")
st.write("Upload examples of your past resumes or cover letters. The agent will analyze them to learn your unique writing style, which will be used to make future generated documents sound more like you.")

# --- File Uploader ---
st.header("Upload Your Documents")
uploaded_files = st.file_uploader(
    "Choose one or more files (.pdf, .docx, .txt)",
    accept_multiple_files=True,
    type=['pdf', 'docx', 'txt']
)

if uploaded_files:
    if st.button("Analyze My Writing Style"):
        gemini_key = st.session_state.get("gemini_api_key")
        if not gemini_key:
            st.error("Please set your Gemini API key in the Settings page to use this feature.")
            st.stop()
        
        with st.spinner("Parsing files and analyzing your style..."):
            try:
                # 1. Parse files to get text
                combined_text = parse_files(uploaded_files)

                # 2. Send to AI for analysis
                gemini_client = GeminiClient(api_key=gemini_key)
                prompt = f"""
                **Persona:** You are an expert writing coach and linguistic analyst.
                **Task:** Analyze the following text from a user's past professional documents. Identify the key characteristics of their writing style.
                
                **Reasoning Framework:**
                1.  **Tone Analysis:** Is the tone formal, conversational, direct, academic, warm, etc.?
                2.  **Vocabulary:** What kind of action verbs are commonly used? Is the language simple or sophisticated?
                3.  **Sentence Structure:** Are sentences typically short and punchy, or long and detailed?
                4.  **Key Themes:** What are the recurring themes or values expressed (e.g., collaboration, efficiency, innovation)?
                
                **Documents Text:**
                ---
                {combined_text[:8000]}
                ---
                
                **Output Format:**
                Provide a concise summary of the user's writing style in 3-4 bullet points. This summary will be used as a style guide for the AI.
                """
                style_profile = gemini_client.generate_text(prompt)

                # 3. Save the style profile to the database
                db.save_style_profile(style_profile)
                st.success("✅ Your writing style has been analyzed and saved!")
                st.balloons()
                st.subheader("Your Personal Style Profile:")
                st.markdown(style_profile)

            except Exception as e:
                st.error(f"An error occurred during style analysis: {e}")

# Display current style profile
st.header("Current Style Profile")
profile = db.get_user_profile()
if profile and profile.get("style_profile"):
    st.markdown(profile["style_profile"])
else:
    st.info("No style profile has been generated yet. Upload some documents to create one.")
EOF

# --- pages/5_Resume_Scorer.py ---
cat <<'EOF' > pages/5_Resume_Scorer.py
# pages/5_Resume_Scorer.py
import streamlit as st
import database as db
from api_clients import GeminiClient
from file_parser import parse_files, parse_pdf, parse_docx
import io
import json

st.set_page_config(page_title="Resume Scorer", layout="wide")
st.title("🎯 Resume Scorer")
st.write("Upload your final resume and the job description to get an AI-powered match score and actionable feedback.")

# --- Inputs ---
st.header("Inputs")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Final Resume")
    resume_file = st.file_uploader("Upload your resume (.pdf, .docx, .txt)", type=['pdf', 'docx', 'txt'])

with col2:
    st.subheader("Target Job Description")
    job_desc_text = st.text_area("Paste the full job description here", height=250)

if st.button("Score My Resume", type="primary", disabled=not (resume_file and job_desc_text)):
    gemini_key = st.session_state.get("gemini_api_key")
    if not gemini_key:
        st.error("Please set your Gemini API key in the Settings page to use this feature.")
        st.stop()

    with st.spinner("Parsing documents and scoring your resume..."):
        try:
            # 1. Parse resume file
            if resume_file.type == "application/pdf":
                resume_text = parse_pdf(io.BytesIO(resume_file.getvalue()))
            elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = parse_docx(io.BytesIO(resume_file.getvalue()))
            else:
                resume_text = resume_file.getvalue().decode("utf-8")

            # 2. Get score from AI
            from document_generator import DocumentGenerator
            doc_generator = DocumentGenerator(GeminiClient(api_key=gemini_key))
            score_data = doc_generator.score_resume(resume_text, job_desc_text)

            # 3. Display results
            if "error" in score_data:
                st.error(f"Scoring failed: {score_data['error']}")
                st.code(score_data.get('raw_text'))
            else:
                st.header("📊 Scoring Results")
                score = score_data.get("match_score", 0)
                
                # Display score with a progress bar and color
                st.subheader(f"Overall Match Score: {score}%")
                progress_color = "red"
                if score > 75: progress_color = "green"
                elif score > 50: progress_color = "orange"
                st.progress(score / 100)

                st.subheader("✅ Strengths")
                for strength in score_data.get("strengths", []):
                    st.markdown(f"- {strength}")

                st.subheader("💡 Suggestions for Improvement")
                for suggestion in score_data.get("suggestions", []):
                    st.markdown(f"- {suggestion}")

        except Exception as e:
            st.error(f"An error occurred during scoring: {e}")
EOF

echo "\n✅ Project setup complete!"
echo "\nNext steps:"
echo "1. Create and activate a Python virtual environment:"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo "2. Install the required packages:"
echo "   pip install -r requirements.txt"
echo "3. Run the Streamlit application:"
echo "   streamlit run main_app.py



