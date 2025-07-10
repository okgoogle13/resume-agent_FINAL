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
