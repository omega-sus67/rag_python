from fast_api import HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
from dataclasses import dataclass

@dataclass
class Document:
    title : str
    extracted_text : str

# parser function that will read the pdf file and return the extracted text as Document object
def parsePdf(path : str) -> Document :
    try:
        reader = PdfReader(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404 , detail="file not found")
    
    extText = ""

    for txt in reader.pages:
        extText += txt.extract_text() 
    
    return Document(
        title = path.split("/")[-1].replace(".pdf", "").title(),
        extracted_text = extText
    )

