import streamlit as st

def main():
    st.set_page_config(layout="wide")
    st.title("AI Resume Builder")

    st.header("1. Input Job Details")
    job_url_input = st.text_input("Enter Job Posting URL:", key="job_url_input", placeholder="https://example.com/job-listing/123")
    # We'll need a variable to hold the actual job description text, whether from URL or manual fallback (later)
    # For now, the tailoring functions expect job_description_text.
    # This will be populated by the web scraping logic later in this phase.
    # We also need to handle the case where job_url_input is used directly by the button logic.

from web_scraper import fetch_page_content, extract_job_description_from_html
from database import initialize_db, log_application_to_job # Import database functions

# Initialize database (ensure table exists)
initialize_db()

    st.header("2. Upload Your Base Documents")
    base_resume = st.file_uploader(
        "Upload Base Resume (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        key="base_resume"
    )
    base_cover_letter = st.file_uploader(
        "Upload Base Cover Letter Template (PDF, DOCX, TXT, optional)",
        type=['pdf', 'docx', 'txt'],
        key="base_cover_letter"
    )
    base_ksc = st.file_uploader(
        "Upload Base KSC Examples (PDF, DOCX, TXT, optional)",
        type=['pdf', 'docx', 'txt'],
        key="base_ksc"
    )

# --- Placeholder Document Processing Logic ---
USER_NAME = "Nishant Jonas Dougall"

def extract_text_from_file(uploaded_file_object):
    """
    Placeholder function to extract text from an uploaded file.
    For Phase 1, this will be very basic and might assume TXT or just return a filename.
    Proper PDF/DOCX parsing will be added later.
    """
    if uploaded_file_object is None:
        return ""

    try:
        file_type = uploaded_file_object.type
        file_name = uploaded_file_object.name

        if file_type == "text/plain":
            return uploaded_file_object.getvalue().decode("utf-8")
        elif file_type == "application/pdf":
            import pypdf
            pdf_reader = pypdf.PdfReader(uploaded_file_object)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() or "" # Add null check for empty pages
            if not text.strip(): # Check if extracted text is empty or just whitespace
                 return f"[Could not extract text from PDF: {file_name} - The document might be image-based or empty.]"
            return text
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            import docx
            document = docx.Document(uploaded_file_object)
            text = "\n".join([para.text for para in document.paragraphs])
            if not text.strip(): # Check if extracted text is empty or just whitespace
                return f"[Could not extract text from DOCX: {file_name} - The document might be empty or structured in a way that text isn't in paragraphs (e.g. tables only).]"
            return text
        else:
            return f"[Unsupported file type: {file_type} for file: {file_name} - Cannot extract text.]"
    except Exception as e:
        st.error(f"Error processing file {uploaded_file_object.name}: {e}")
        return f"[Error extracting text from {uploaded_file_object.name}: {str(e)}]"

def generate_tailored_resume(job_description_text: str, base_resume_text: str) -> str:
    """Placeholder for AI-driven resume tailoring."""
    if not job_description_text or not base_resume_text:
        return "Missing job description or base resume for tailoring."
    return (f"--- TAILORED RESUME (Placeholder) ---\n"
            f"Based on Job Description (first 50 chars): '{job_description_text[:50]}...'\n"
            f"And Base Resume (first 50 chars): '{base_resume_text[:50]}...'\n"
            f"This resume has been automatically tailored for {USER_NAME}.")

def generate_tailored_cover_letter(job_description_text: str, base_cover_letter_text: str, user_name: str) -> str:
    """Placeholder for AI-driven cover letter tailoring."""
    if not job_description_text:
        return "Missing job description for cover letter tailoring."
    if not base_cover_letter_text: # Optional base
        base_cover_letter_text = "[No base cover letter provided - generating generic one]"
    return (f"--- TAILORED COVER LETTER (Placeholder) ---\n"
            f"To: Hiring Manager\n"
            f"From: {user_name}\n"
            f"Regarding Job (first 50 chars): '{job_description_text[:50]}...'\n"
            f"Content based on: '{base_cover_letter_text[:50]}...'\n"
            f"This cover letter expresses {user_name}'s keen interest.")

def generate_tailored_ksc(job_description_text: str, base_ksc_text: str) -> str:
    """Placeholder for AI-driven KSC response tailoring."""
    if not job_description_text:
        return "Missing job description for KSC tailoring."
    if not base_ksc_text: # Optional base
        base_ksc_text = "[No base KSC examples provided - generating generic responses]"
    return (f"--- TAILORED KSC RESPONSE (Placeholder) ---\n"
            f"For Job (first 50 chars): '{job_description_text[:50]}...'\n"
            f"Drawing from KSC examples: '{base_ksc_text[:50]}...'\n"
            f"Key Selection Criteria responses for {USER_NAME}.")

# --- End of Placeholder Logic ---

    st.header("3. Generate Tailored Documents")
    if st.button("Generate Tailored Documents", key="generate_button"):
        # In this phase, job_description_text will be populated by web scraping.
        # For now, the error check is on job_url_input.
        # The actual text will be passed to tailoring functions later.
        job_description_text_for_tailoring = "" # This will be filled by scraper output

        if not job_url_input: # Check if URL is provided
            st.error("Please enter the Job Posting URL above.")
        elif not base_resume:
            st.error("Please upload your Base Resume.")
        else:
            # Web scraping logic will go here in the next step.
            # For now, we'll use a placeholder for job_description_text_for_tailoring
            # to allow the rest of the flow to be connected.
            # In a real scenario, if scraping fails, we might ask for manual input.

            # Placeholder: Simulate that job_url_input itself is the JD for now,
            # or that scraping will happen and populate job_description_text_for_tailoring.
            # This will be replaced by actual scraped content.
            # job_description_text_for_tailoring = f"Job Description from URL: {job_url_input} (Actual scraping pending)" # Placeholder

            with st.spinner("Fetching and processing job description from URL..."):
                html_content = fetch_page_content(job_url_input)
                if html_content:
                    job_description_text_for_tailoring = extract_job_description_from_html(html_content, job_url_input)
                    if "[Could not automatically extract detailed job description" in job_description_text_for_tailoring or not job_description_text_for_tailoring.strip():
                        st.warning(f"Could not extract a detailed job description from the URL. The AI will proceed with the information available (which might be limited or just a placeholder). For best results, you might need to find a more direct link or ensure the page content is accessible. The message from scraper was: {job_description_text_for_tailoring}")
                        # Provide a way to see what was (or wasn't) scraped.
                        st.expander("View Scraped Content (or error message)").write(job_description_text_for_tailoring)
                        # Fallback to a generic message if scraping truly yields nothing usable by AI.
                        if not job_description_text_for_tailoring.strip() or "[Could not automatically extract detailed job description" in job_description_text_for_tailoring :
                             job_description_text_for_tailoring = f"No detailed job description could be extracted from URL: {job_url_input}. Proceeding with minimal information."

                else:
                    st.error(f"Failed to fetch content from the URL: {job_url_input}. Please check the URL or your network connection.")
                    st.stop() # Stop further processing if URL fetch fails

            if not job_description_text_for_tailoring.strip():
                 st.error("Job description could not be obtained from the URL. Cannot proceed.")
                 st.stop()


            with st.spinner("Tailoring documents... Please wait."):
                # Extract text from uploaded files
                base_resume_text = extract_text_from_file(base_resume)
                base_cover_letter_text = extract_text_from_file(base_cover_letter) if base_cover_letter else ""
                base_ksc_text = extract_text_from_file(base_ksc) if base_ksc else ""

                # Call placeholder generation functions
                tailored_resume = generate_tailored_resume(job_description, base_resume_text)
                tailored_cover_letter = generate_tailored_cover_letter(job_description, base_cover_letter_text, USER_NAME)
                tailored_ksc_response = generate_tailored_ksc(job_description, base_ksc_text)

                st.subheader("Tailored Resume")
                st.text_area("Resume TXT", value=tailored_resume, height=200, key="resume_output_area")
                st.download_button(
                    label="Download Resume.txt",
                    data=tailored_resume,
                    file_name="Tailored_Resume.txt",
                    mime="text/plain"
                )

                st.subheader("Tailored Cover Letter")
                st.text_area("Cover Letter TXT", value=tailored_cover_letter, height=200, key="cover_letter_output_area")
                st.download_button(
                    label="Download Cover_Letter.txt",
                    data=tailored_cover_letter,
                    file_name="Tailored_Cover_Letter.txt",
                    mime="text/plain"
                )

                st.subheader("Tailored KSC Response")
                st.text_area("KSC Response TXT", value=tailored_ksc_response, height=200, key="ksc_output_area")
                st.download_button(
                    label="Download KSC_Response.txt",
                    data=tailored_ksc_response,
                    file_name="Tailored_KSC_Response.txt",
                    mime="text/plain"
                )
                st.success("Documents generated successfully!")

                # Log application to database
                # For Phase 2, actual paths to saved docs are not implemented yet.
                # Company name and job title extraction from JD is also a future step.
                # We'll log the URL and the scraped JD text.
                logged_app_id = log_application_to_job(
                    url=job_url_input,
                    job_description_text=job_description_text_for_tailoring,
                    company_name=None, # Placeholder for now
                    role_title=None,   # Placeholder for now
                    generated_doc_paths={} # Placeholder for now
                )
                if logged_app_id:
                    st.info(f"Application attempt for {job_url_input} logged to database (ID: {logged_app_id}).")
                else:
                    # Error/warning for logging failure is handled within log_application_to_job via st.warning/st.error
                    pass


if __name__ == "__main__":
    main()
