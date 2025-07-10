# file_parser.py
"""
A utility module to parse text content from different file types like PDF and DOCX.
"""
import io
from typing import List
from docx import Document
from pypdf import PdfReader
import streamlit as st

def parse_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """Parses a list of uploaded files and returns their combined text content."""
    full_text = []
    for file in uploaded_files:
        if file.type == "application/pdf":
            text = parse_pdf(file)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = parse_docx(file)
        elif file.type == "text/plain":
            text = file.getvalue().decode("utf-8")
        else:
            text = f"Unsupported file type: {file.name}"
        full_text.append(f"--- Document: {file.name} ---\n{text}\n\n")
    return "".join(full_text)

def parse_pdf(file: io.BytesIO) -> str:
    """Extracts text from an uploaded PDF file."""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error parsing PDF: {e}"

def parse_docx(file: io.BytesIO) -> str:
    """Extracts text from an uploaded DOCX file."""
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error parsing DOCX: {e}"
