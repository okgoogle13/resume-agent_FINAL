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
                st.session_state.generated_content = doc_generator.generate_resume_markdown(user_profile, experiences)
            elif st.session_state.doc_type == "KSC Response":
                st.session_state.generated_content = doc_generator.generate_ksc_response(st.session_state.job_desc, user_profile, experiences, company_intel, st.session_state.role_title)
            elif st.session_state.doc_type == "Cover Letter":
                st.session_state.generated_content = doc_generator.generate_cover_letter_markdown(user_profile, experiences, job_details, company_intel)
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
