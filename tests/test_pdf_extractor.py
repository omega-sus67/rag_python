# tests/test_pdf_extractor.py

from app.utils.pdf_extractor import clean_extracted_text

def test_clean_extracted_text_removes_firefox_header():
    """
    PURPOSE: Verifies that the 'firefox' keyword page headers are removed.
    CAPABILITIES:
    - Case-insensitive matching filters 'Firefox' or 'firefox'.
    """
    dirty_text = "Firefox \nThis is a real sentence."
    expected = "This is a real sentence."
    assert clean_extracted_text(dirty_text) == expected

def test_clean_extracted_text_removes_page_numbers():
    """
    PURPOSE: Verifies that page indicators of pattern 'X of Y' are discarded.
    CAPABILITIES:
    - Filters numbers separated by 'of' keyword.
    """
    dirty_text = "1 of 95 \nWendy was sleeping."
    expected = "Wendy was sleeping."
    assert clean_extracted_text(dirty_text) == expected

def test_clean_extracted_text_removes_dates():
    """
    PURPOSE: Verifies removal of timestamp markers added during conversions.
    CAPABILITIES:
    - Regex pattern correctly matches date-time pairs.
    """
    dirty_text = "6/6/26, 23:58 \nPeter Pan flew in."
    expected = "Peter Pan flew in."
    assert clean_extracted_text(dirty_text) == expected

def test_clean_extracted_text_removes_gutenberg_url():
    """
    PURPOSE: Verifies URL cleanup of Gutenberg projects to prevent search model index pollution.
    CAPABILITIES:
    - Case-insensitive domain string filtering.
    """
    dirty_text = "http://www.gutenberg.org/ebooks/16 \nKeep this line."
    expected = "Keep this line."
    assert clean_extracted_text(dirty_text) == expected

from unittest.mock import patch
import pytest
from fastapi import HTTPException
from app.utils.pdf_extractor import parsePdf

@patch("pymupdf4llm.to_markdown")
def test_parse_pdf_success(mock_to_markdown):
    """
    PURPOSE: Verifies successful extraction, layout conversions, and title generation on valid paths.
    CAPABILITIES:
    - Conversions to markdown are called.
    - Path base name is title-cased.
    """
    mock_to_markdown.return_value = "Firefox \nSome text inside the PDF."
    doc = parsePdf("/dummy/path/my_awesome_file.pdf")
    
    assert doc.title == "My_Awesome_File"
    assert doc.extracted_text == "Some text inside the PDF."
    mock_to_markdown.assert_called_once_with("/dummy/path/my_awesome_file.pdf")

@patch("pymupdf4llm.to_markdown")
def test_parse_pdf_file_not_found(mock_to_markdown):
    """
    PURPOSE: Verifies that FileNotFound translates to a 404 HTTP Exception.
    CAPABILITIES:
    - Wraps system IOError to correct API status codes.
    """
    mock_to_markdown.side_effect = FileNotFoundError()
    
    with pytest.raises(HTTPException) as exc_info:
        parsePdf("/dummy/path/missing.pdf")
        
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "file not found"

@patch("pymupdf4llm.to_markdown")
def test_parse_pdf_generic_exception(mock_to_markdown):
    """
    PURPOSE: Verifies general error trapping inside parser function.
    CAPABILITIES:
    - Converts unexpected exceptions into 500 error responses with trace details.
    """
    mock_to_markdown.side_effect = Exception("Permission denied")
    
    with pytest.raises(HTTPException) as exc_info:
        parsePdf("/dummy/path/error.pdf")
        
    assert exc_info.value.status_code == 500
    assert "Failed to parse PDF: Permission denied" in exc_info.value.detail

# --- RIGOROUS EXTENDED TESTS ---

def test_clean_extracted_text_empty_and_spaces():
    """
    PURPOSE: Verifies cleaning behaviors when text content contains only noise lines.
    CAPABILITIES:
    - Filters text consisting entirely of firefox headers and page numbers.
    - Yields a correct empty/whitespace layout preserving breaks.
    """
    noise_only = "firefox\n1 of 10\n\n6/6/26, 12:00"
    # Firefox, page count, and date are matched and ignored; the blank line is kept.
    assert clean_extracted_text(noise_only) == ""

@patch("pymupdf4llm.to_markdown")
def test_parse_pdf_title_derivation_edge_cases(mock_to_markdown):
    """
    PURPOSE: Tests file title extraction from complex file paths.
    CAPABILITIES:
    - Extracts correct title when path base contains multiple dots, hyphens, and case differences.
    - Converts filename elements to Title Case format.
    """
    mock_to_markdown.return_value = "Content"
    doc = parsePdf("/complex-path/another.directory/my.awesome-document-v2.0.pdf")
    
    # Path extraction logic: split("/")[-1] -> "my.awesome-document-v2.0.pdf" -> replace(".pdf", "")
    # -> "my.awesome-document-v2.0" -> title() -> "My.Awesome-Document-V2.0"
    assert doc.title == "My.Awesome-Document-V2.0"