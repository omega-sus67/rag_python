from fastapi import HTTPException
from pydantic import BaseModel
import pymupdf4llm
import re
from dataclasses import dataclass

@dataclass
class Document:
    title : str
    extracted_text : str

def clean_extracted_text(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    
    firefox_pattern = re.compile(r'^firefox\s*$', re.IGNORECASE)
    date_pattern = re.compile(r'^\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*$')
    page_pattern = re.compile(r'^\d+\s+of\s+\d+\s*$')
    gutenberg_url_pattern = re.compile(r'^(https?://)?(www\.)?[a-zA-Z0-9.-]*gutenberg\.org\S*\s*$', re.IGNORECASE)
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
            
        if firefox_pattern.match(stripped):
            continue
        if date_pattern.match(stripped):
            continue
        if page_pattern.match(stripped):
            continue
        if gutenberg_url_pattern.match(stripped):
            continue
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

# parser function that will read the pdf file and return the extracted text as Document object
def parsePdf(path : str) -> Document :
    try:
        extText = pymupdf4llm.to_markdown(path)
        extText = clean_extracted_text(extText)
    except FileNotFoundError:
        raise HTTPException(status_code=404 , detail="file not found")
    except Exception as e:
        raise HTTPException(status_code=500 , detail=f"Failed to parse PDF: {str(e)}")
    
    return Document(
        title = path.split("/")[-1].replace(".pdf", "").title(),
        extracted_text = extText
    )



