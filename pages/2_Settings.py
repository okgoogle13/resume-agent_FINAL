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
