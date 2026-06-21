from fastapi import HTTPException
from pydantic import BaseModel
import pymupdf4llm
import re
from dataclasses import dataclass

@dataclass
class Document:
    """Simple container for storing extracted document details."""
    title : str
    extracted_text : str

def clean_extracted_text(text: str) -> str:
    """
    Cleans up boilerplate, page headers, dates, and project-specific URLs
    that pollute the content extraction from converted markdown/PDF lines.
    """
    lines = text.split("\n")
    cleaned_lines = []
    
    # Regex patterns targeting recurring headers, footers, and metadata timestamps.
    firefox_pattern = re.compile(r'^firefox\s*$', re.IGNORECASE)
    date_pattern = re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*$') # Matches "MM/DD/YYYY, HH:MM"
    page_pattern = re.compile(r'^\d+\s+of\s+\d+\s*$')                            # Matches "X of Y" page indicators
    gutenberg_url_pattern = re.compile(r'^(https?://)?(www\.)?[a-zA-Z0-9.-]*gutenberg\.org\S*\s*$', re.IGNORECASE)
    
    for line in lines:
        stripped = line.strip()
        # Preserve empty lines as structural breaks/paragraph boundaries.
        if not stripped:
            cleaned_lines.append("")
            continue
            
        # Ignore boilerplate lines matching headers, footers, page counts or catalog URLs.
        if firefox_pattern.match(stripped):
            continue
        if date_pattern.match(stripped):
            continue
        if page_pattern.match(stripped):
            continue
        if gutenberg_url_pattern.match(stripped):
            continue
            
        # Keep normal valid text lines.
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def parsePdf(path : str) -> Document :
    """
    Converts a PDF file to structural markdown layout using PyMuPDF4LLM.
    Wraps execution in a try-except block to translate system exceptions into proper HTTP responses.
    """
    try:
        # Extract markdown structure directly from the file.
        extText = pymupdf4llm.to_markdown(path)
        # Apply cleaning to purge headers, page numbers, and links.
        extText = clean_extracted_text(extText)
    except FileNotFoundError:
        # Explicitly map missing files to 404 resource errors.
        raise HTTPException(status_code=404 , detail="file not found")
    except Exception as e:
        # Map any other parsing, permission, or library errors to a general 500 error.
        raise HTTPException(status_code=500 , detail=f"Failed to parse PDF: {str(e)}")
    
    return Document(
        # Derive document title from filename by extracting base and converting casing (Title Case).
        title = path.split("/")[-1].replace(".pdf", "").title(),
        extracted_text = extText
    )
