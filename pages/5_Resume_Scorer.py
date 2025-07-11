# pages/5_Resume_Scorer.py
import streamlit as st
from api_clients import GeminiClient
from file_parser import parse_pdf, parse_docx
from document_generator import DocumentGenerator
import io
import json

st.set_page_config(page_title="Resume Analyzer", layout="wide") # Changed title
st.title("🎯 Resume Analyzer") # Changed title
st.write("Upload your resume and the job description to get an AI-powered analysis, including keyword matching, ATS friendliness, and actionable feedback.")

# --- Inputs ---
st.header("Inputs")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume") # Changed subheader
    resume_file = st.file_uploader("Upload your resume (.pdf, .docx, .txt)", type=['pdf', 'docx', 'txt'], key="resume_upload")

with col2:
    st.subheader("Target Job Description")
    job_desc_text = st.text_area("Paste the full job description here", height=250, key="job_desc_input")

# Initialize session state for analysis results
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if st.button("🚀 Analyze Resume", type="primary", use_container_width=True, disabled=not (resume_file and job_desc_text)): # Changed button text
    gemini_key = st.session_state.get("gemini_api_key")
    if not gemini_key:
        st.error("Please set your Gemini API key in the Settings page to use this feature.")
        st.stop()

    with st.spinner("Parsing documents and analyzing your resume... This might take a moment."): # Updated spinner text
        try:
            # 1. Parse resume file
            if resume_file.type == "application/pdf":
                resume_text = parse_pdf(io.BytesIO(resume_file.getvalue()))
            elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                resume_text = parse_docx(io.BytesIO(resume_file.getvalue()))
            else:
                resume_text = resume_file.getvalue().decode("utf-8", errors="ignore") # Added errors ignore

            # 2. Get analysis from AI
            doc_generator = DocumentGenerator(GeminiClient(api_key=gemini_key))
            # Call the new analysis function
            analysis_data = doc_generator.analyze_document_for_keywords_ats(resume_text, job_desc_text)
            st.session_state.analysis_results = analysis_data

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.session_state.analysis_results = {"error": f"An error occurred during analysis: {e}"}

# --- Display Analysis Results ---
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    st.divider()
    st.header("📊 Analysis Results")

    if "error" in results:
        st.error(f"Analysis failed: {results['error']}")
        if "raw_jd_response" in results:
            st.subheader("Raw Job Description AI Response (Debug):")
            st.code(results["raw_jd_response"])
        if "raw_resume_response" in results:
            st.subheader("Raw Resume Analysis AI Response (Debug):")
            st.code(results["raw_resume_response"])
    else:
        # Using tabs for better organization
        tab1, tab2, tab3 = st.tabs(["🔑 Keyword Analysis", "🤖 ATS Friendliness", "💡 General Suggestions"])

        with tab1:
            st.subheader("Keyword Matching & Suggestions")
            keyword_analysis = results.get("keyword_analysis", [])
            extracted_keywords = results.get("extracted_critical_keywords", []) # Get the originally extracted keywords

            if not keyword_analysis and not extracted_keywords:
                st.info("No keyword analysis data available.")
            else:
                # Create a DataFrame for display if keyword_analysis is present
                if keyword_analysis:
                    # We need to merge info from extracted_keywords (category) into keyword_analysis
                    # For simplicity, we'll rebuild a display list
                    display_keywords_data = []
                    for ka_item in keyword_analysis:
                        category = "N/A"
                        for ek_item in extracted_keywords:
                            if ek_item["term"] == ka_item["term"]:
                                category = ek_item.get("category", "N/A")
                                break
                        display_keywords_data.append({
                            "Keyword/Phrase": ka_item["term"],
                            "Category": category,
                            "Present in Resume?": "✅ Yes" if ka_item["present"] else "❌ No",
                            "Suggestion (if missing)": ka_item.get("suggestion", "N/A") if not ka_item["present"] else "N/A"
                        })

                    if display_keywords_data:
                        st.dataframe(display_keywords_data, use_container_width=True)
                    else:
                        st.info("No keyword matching data to display.")
                elif extracted_keywords: # Fallback if only extraction happened but not analysis
                    st.info("Keywords were extracted from the job description, but detailed matching against the resume was not completed.")
                    st.write("Extracted Keywords:")
                    df_extracted = [{"Keyword/Phrase": kw.get("term"), "Category": kw.get("category")} for kw in extracted_keywords]
                    st.dataframe(df_extracted, use_container_width=True)


        with tab2:
            st.subheader("ATS Friendliness Report")
            ats_friendliness = results.get("ats_friendliness", {})
            if not ats_friendliness:
                st.info("No ATS friendliness data available.")
            else:
                score = ats_friendliness.get("overall_score", 0)
                st.progress(score / 100)
                st.markdown(f"**Overall ATS Friendliness Score: {score}%**")

                issues = ats_friendliness.get("issues", [])
                if issues:
                    st.markdown("**Potential Issues Found:**")
                    for issue in issues:
                        with st.container(border=True):
                            st.error(f"**{issue.get('issue_type', 'Unknown Issue')}**")
                            st.write(f"*{issue.get('description', 'No description.')}*")
                            st.caption(f"Suggestion: {issue.get('suggestion', 'No suggestion.')}")
                else:
                    st.success("✅ No major ATS issues detected based on the analysis!")

        with tab3:
            st.subheader("Overall Improvement Suggestions")
            general_suggestions = results.get("general_suggestions", [])
            if general_suggestions:
                for suggestion in general_suggestions:
                    st.markdown(f"- {suggestion}")
            else:
                st.info("No general suggestions provided in this analysis.")
