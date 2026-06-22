from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.controllers.main_controller import MainController

# Instantiate our central orchestrator
controller = MainController()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up pipeline resources...")
    await controller.initialize_system()
    yield
    print("Shutting down pipeline resources...")

class FilePathRequest(BaseModel):
    path: str

class DocumentResponse(BaseModel):
    id: str
    title: str
    extracted_text: str
    
    class Config:
        from_attributes = True

class AgentQueryRequest(BaseModel):
    query: str

# Initialize FastAPI app with the orchestrated lifespan
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Welcome to the PDF Parser API(v1.0)"}

@app.post("/upload")
async def upload_file(path: FilePathRequest):
    """
    Parses the given PDF, stores its text in the database, 
    and chunks/embeds its contents seamlessly via the controller.
    """
    return await controller.process_and_ingest_pdf(path.path)

@app.get("/docFetch/{id}", response_model=DocumentResponse)
async def getFileByID(id: str):
    """
    Retrieves the raw document metadata by its hashed ID.
    """
    return await controller.fetch_document(id)

@app.post("/agent/query")
async def query_agent(request: AgentQueryRequest):
    """
    Solves user questions dynamically using a ReAct reasoning agent loop.
    """
    return await controller.ask_agent(request.query)