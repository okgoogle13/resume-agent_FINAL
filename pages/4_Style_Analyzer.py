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
