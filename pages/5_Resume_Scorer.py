# pages/5_Resume_Scorer.py
import streamlit as st
from api_clients import GeminiClient
from file_parser import parse_single_uploaded_file # Updated import
from document_generator import DocumentGenerator
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
            # 1. Parse resume file using the centralized parser
            resume_text = parse_single_uploaded_file(resume_file)

            # Check if parsing returned an error message (they usually start with "[")
            if resume_text.startswith("["):
                st.error(f"Failed to parse resume: {resume_text}")
                st.stop()

            # 2. Get score from AI
            doc_generator = DocumentGenerator(GeminiClient(api_key=gemini_key))
            score_data = doc_generator.score_resume(resume_text, job_desc_text)

            # 3. Display results
            if "error" in score_data:
                st.error(f"Scoring failed: {score_data['error']}")
                st.code(score_data.get('raw_text'))
            else:
                st.header("📊 Scoring Results")
                score = score_data.get("match_score", 0)
                
                # --- FIX APPLIED HERE ---
                # Display score and use score/100 for the progress bar value.
                st.subheader(f"Overall Match Score: {score}%")
                st.progress(score / 100)

                st.subheader("✅ Strengths")
                for strength in score_data.get("strengths", []):
                    st.markdown(f"- {strength}")

                st.subheader("💡 Suggestions for Improvement")
                for suggestion in score_data.get("suggestions", []):
                    st.markdown(f"- {suggestion}")

        except Exception as e:
            st.error(f"An error occurred during scoring: {e}")
