from fastapi import HTTPException
from pydantic import BaseModel
import pymupdf4llm
from dataclasses import dataclass

@dataclass
class Document:
    title : str
    extracted_text : str

# parser function that will read the pdf file and return the extracted text as Document object
def parsePdf(path : str) -> Document :
    try:
        extText = pymupdf4llm.to_markdown(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404 , detail="file not found")
    except Exception as e:
        raise HTTPException(status_code=500 , detail=f"Failed to parse PDF: {str(e)}")
    
    return Document(
        title = path.split("/")[-1].replace(".pdf", "").title(),
        extracted_text = extText
    )


