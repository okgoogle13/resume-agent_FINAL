# file_parser.py
"""
A utility module to parse text content from different file types like PDF, DOCX, and TXT.
Handles Streamlit UploadedFile objects.
"""
import io
from typing import List, Union
import docx # python-docx
import pypdf # pypdf
import streamlit as st # For type hinting UploadedFile

# Define a type alias for clarity if needed, though UploadedFile is quite specific
UploadedFile = st.runtime.uploaded_file_manager.UploadedFile

def parse_pdf(file_object: UploadedFile) -> str:
    """
    Extracts text from an uploaded PDF file.
    Returns extracted text or an error message string.
    """
    file_name = file_object.name
    try:
        pdf_reader = pypdf.PdfReader(file_object)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() or ""
        if not text.strip():
            return f"[Could not extract text from PDF: {file_name} - The document might be image-based or empty.]"
        return text
    except Exception as e:
        return f"[Error parsing PDF {file_name}: {str(e)}]"

def parse_docx(file_object: UploadedFile) -> str:
    """
    Extracts text from an uploaded DOCX file.
    Returns extracted text or an error message string.
    """
    file_name = file_object.name
    try:
        document = docx.Document(file_object)
        text = "\n".join([para.text for para in document.paragraphs])
        if not text.strip():
            return f"[Could not extract text from DOCX: {file_name} - The document might be empty or structured in a way that text isn't in paragraphs (e.g. tables only).]"
        return text
    except Exception as e:
        return f"[Error parsing DOCX {file_name}: {str(e)}]"

def parse_txt(file_object: UploadedFile) -> str:
    """
    Extracts text from an uploaded TXT file.
    Returns extracted text or an error message string.
    """
    file_name = file_object.name
    try:
        return file_object.getvalue().decode("utf-8")
    except Exception as e:
        return f"[Error parsing TXT {file_name}: {str(e)}]"

def parse_single_uploaded_file(uploaded_file: UploadedFile) -> str:
    """
    Parses a single Streamlit UploadedFile object and returns its text content.
    Returns extracted text or an error/info message string.
    """
    if uploaded_file is None:
        return "" # Or perhaps an info message like "[No file provided]"

    file_type = uploaded_file.type
    file_name = uploaded_file.name

    if file_type == "text/plain":
        return parse_txt(uploaded_file)
    elif file_type == "application/pdf":
        return parse_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return parse_docx(uploaded_file)
    else:
        return f"[Unsupported file type: {file_type} for file: {file_name} - Cannot extract text.]"

def parse_multiple_uploaded_files(uploaded_files: List[UploadedFile]) -> str:
    """
    Parses a list of Streamlit UploadedFile objects and returns their combined text content,
    with each document's content separated and identified.
    """
    full_text_parts = []
    for file in uploaded_files:
        text_content = parse_single_uploaded_file(file)
        full_text_parts.append(f"--- Document: {file.name} ---\n{text_content}\n\n")
    return "".join(full_text_parts)

# For backward compatibility or alternative naming if preferred by other modules.
# parse_files can be an alias or the main function name.
parse_files = parse_multiple_uploaded_files
