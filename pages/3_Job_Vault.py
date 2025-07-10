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
