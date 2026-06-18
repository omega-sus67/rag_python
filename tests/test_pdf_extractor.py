from app.utils.pdf_extractor import clean_extracted_text

def test_clean_extracted_text_removes_firefox_header():
    dirty_text = "Firefox \nThis is a real sentence."
    expected = "This is a real sentence."
    assert clean_extracted_text(dirty_text) == expected

def test_clean_extracted_text_removes_page_numbers():
    dirty_text = "1 of 95 \nWendy was sleeping."
    expected = "Wendy was sleeping."
    assert clean_extracted_text(dirty_text) == expected

def test_clean_extracted_text_removes_dates():
    dirty_text = "6/6/26, 23:58 \nPeter Pan flew in."
    expected = "Peter Pan flew in."
    assert clean_extracted_text(dirty_text) == expected

def test_clean_extracted_text_removes_gutenberg_url():
    dirty_text = "http://www.gutenberg.org/ebooks/16 \nKeep this line."
    expected = "Keep this line."
    assert clean_extracted_text(dirty_text) == expected

from unittest.mock import patch
import pytest
from fastapi import HTTPException
from app.utils.pdf_extractor import parsePdf

@patch("pymupdf4llm.to_markdown")
def test_parse_pdf_success(mock_to_markdown):
    mock_to_markdown.return_value = "Firefox \nSome text inside the PDF."
    doc = parsePdf("/dummy/path/my_awesome_file.pdf")
    
    assert doc.title == "My_Awesome_File"
    assert doc.extracted_text == "Some text inside the PDF."
    mock_to_markdown.assert_called_once_with("/dummy/path/my_awesome_file.pdf")

@patch("pymupdf4llm.to_markdown")
def test_parse_pdf_file_not_found(mock_to_markdown):
    mock_to_markdown.side_effect = FileNotFoundError()
    
    with pytest.raises(HTTPException) as exc_info:
        parsePdf("/dummy/path/missing.pdf")
        
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "file not found"

@patch("pymupdf4llm.to_markdown")
def test_parse_pdf_generic_exception(mock_to_markdown):
    mock_to_markdown.side_effect = Exception("Permission denied")
    
    with pytest.raises(HTTPException) as exc_info:
        parsePdf("/dummy/path/error.pdf")
        
    assert exc_info.value.status_code == 500
    assert "Failed to parse PDF: Permission denied" in exc_info.value.detail