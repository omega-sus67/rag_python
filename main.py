from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pdf_operator import parsePdf, Document
from db_operator import createTable, saveDocument, fetchDoc
from pydantic import BaseModel
@asynccontextmanager
async def lifespan(app : FastAPI):
    print("starting up")
    await createTable()
    yield
    print("Shutting down")

class FilePathRequest(BaseModel):
    path: str

class DocumentID(BaseModel):
    id: str

class DocumentResponse(BaseModel):
    id: str
    title: str
    extracted_text: str
    class Config:
        from_attributes = True

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message" : "Welcome to the PDF Parser API(v1.0)"}

@app.post("/upload")
async def upload_file(path : FilePathRequest):
    try:
        doc = parsePdf(path.path)
        db_doc = await saveDocument(doc)
        return db_doc
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500 , detail=f"Something went wrong {e}")
    

@app.get("/docFetch/{id}", response_model=DocumentResponse)
async def getFileByID(id : str):
    try:
        return await fetchDoc(id)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500 , detail=f"Something went wrong {e}")



    



