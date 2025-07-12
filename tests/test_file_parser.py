# tests/test_file_parser.py
import pytest
import io
from unittest.mock import MagicMock
from file_parser import parse_files, parse_pdf, parse_docx
from docx import Document
from pypdf import PdfWriter, PageObject

# Helper to create a dummy PDF in memory
def create_dummy_pdf_bytes():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    # This is a bit of a hack. pypdf doesn't have a simple "add text" API.
    # We'll rely on mocking the extraction part instead of creating a complex PDF.
    pdf_bytes = io.BytesIO()
    writer.write(pdf_bytes)
    pdf_bytes.seek(0)
    return pdf_bytes

# Helper to create a dummy DOCX in memory
def create_dummy_docx_bytes(text="Hello, docx!"):
    doc = Document()
    doc.add_paragraph(text)
    docx_bytes = io.BytesIO()
    doc.save(docx_bytes)
    docx_bytes.seek(0)
    return docx_bytes

@pytest.fixture
def mock_uploaded_file():
    """Factory fixture to create a mock UploadedFile."""
    def _mock_file(filename, content_type, content_bytes):
        mock_file = MagicMock()
        mock_file.name = filename
        mock_file.type = content_type
        # The getvalue method should return the bytes content
        mock_file.getvalue.return_value = content_bytes
        # The file-like object itself for parsers
        mock_file.read.return_value = content_bytes
        mock_file.seek.return_value = 0
        return mock_file
    return _mock_file

def test_parse_docx():
    """Test parsing a DOCX file from a byte stream."""
    docx_bytes = create_dummy_docx_bytes("This is a test.")
    text = parse_docx(docx_bytes)
    assert "This is a test." in text

def test_parse_pdf(monkeypatch):
    """Test parsing a PDF file from a byte stream."""
    # Mock the PdfReader and its pages
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is PDF text."
    mock_reader.pages = [mock_page]

    monkeypatch.setattr("file_parser.PdfReader", lambda x: mock_reader)

    pdf_bytes = create_dummy_pdf_bytes() # Content doesn't matter due to mocking
    text = parse_pdf(pdf_bytes)
    assert "This is PDF text." in text

def test_parse_files_all_types(mock_uploaded_file, monkeypatch):
    """Test parse_files with a mix of supported file types."""
    # Mock PDF parsing as before
    mock_reader = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF content."
    mock_reader.pages = [mock_page]
    monkeypatch.setattr("file_parser.PdfReader", lambda x: mock_reader)

    # Create mock files
    docx_file = mock_uploaded_file("resume.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", create_dummy_docx_bytes("DOCX content.").getvalue())
    pdf_file = mock_uploaded_file("jd.pdf", "application/pdf", create_dummy_pdf_bytes().getvalue())
    txt_file = mock_uploaded_file("notes.txt", "text/plain", b"TXT content.")

    uploaded_files = [docx_file, pdf_file, txt_file]

    combined_text = parse_files(uploaded_files)

    assert "--- Document: resume.docx ---" in combined_text
    assert "DOCX content." in combined_text
    assert "--- Document: jd.pdf ---" in combined_text
    assert "PDF content." in combined_text
    assert "--- Document: notes.txt ---" in combined_text
    assert "TXT content." in combined_text

def test_parse_files_unsupported_type(mock_uploaded_file):
    """Test that an unsupported file type is handled gracefully."""
    img_file = mock_uploaded_file("image.jpg", "image/jpeg", b"imagedata")

    combined_text = parse_files([img_file])

    assert "Unsupported file type: image.jpg" in combined_text

def test_parse_pdf_error_handling():
    """Test error handling for a corrupt PDF."""
    corrupt_pdf = io.BytesIO(b"%PDF-not-a-real-pdf")
    result = parse_pdf(corrupt_pdf)
    assert "Error parsing PDF" in result

def test_parse_docx_error_handling():
    """Test error handling for a corrupt DOCX."""
    corrupt_docx = io.BytesIO(b"not-a-real-docx-file")
    result = parse_docx(corrupt_docx)
    assert "Error parsing DOCX" in result
