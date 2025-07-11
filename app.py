import streamlit as st

def main():
    st.set_page_config(layout="wide")
    st.title("AI Resume Builder")

    st.header("1. Input Job Details")
    job_description = st.text_area("Paste the full Job Description here:", height=200, key="job_description")

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
from config import USER_NAME_PLACEHOLDER

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
            f"This resume has been automatically tailored for {USER_NAME_PLACEHOLDER}.")

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
            f"Key Selection Criteria responses for {USER_NAME_PLACEHOLDER}.")

# --- End of Placeholder Logic ---

    st.header("3. Generate Tailored Documents")
    if st.button("Generate Tailored Documents", key="generate_button"):
        if not job_description:
            st.error("Please paste the Job Description above.")
        elif not base_resume:
            st.error("Please upload your Base Resume.")
        else:
            with st.spinner("Processing documents... Please wait."):
                # Extract text from uploaded files
                base_resume_text = extract_text_from_file(base_resume)
                base_cover_letter_text = extract_text_from_file(base_cover_letter) if base_cover_letter else ""
                base_ksc_text = extract_text_from_file(base_ksc) if base_ksc else ""

                # Call placeholder generation functions
                tailored_resume = generate_tailored_resume(job_description, base_resume_text)
                tailored_cover_letter = generate_tailored_cover_letter(job_description, base_cover_letter_text, USER_NAME_PLACEHOLDER)
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

if __name__ == "__main__":
    main()
