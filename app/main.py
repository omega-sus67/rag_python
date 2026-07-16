from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from celery.result import AsyncResult

from app.controllers.main_controller import MainController
from app.worker import process_pdf_task, celery_app

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

@app.post("/upload", status_code=202)
async def upload_file(path: FilePathRequest):
    """
    FastAPI receives the path, immediately queues the task,
    and returns a 202 Accepted status with the task ID.
    """
    # Trigger the Celery task asynchronously via Redis
    task = process_pdf_task.delay(path.path)
    return {
        "message": "Document ingestion queued.",
        "task_id": task.id
    }

@app.get("/task/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Endpoint to check the status of a background ingestion task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": task_result.status, # e.g. PENDING, STARTED, SUCCESS, FAILURE
        "result": task_result.result if task_result.ready() else None
    }

@app.get("/docFetch/{id}", response_model=DocumentResponse)
async def getFileByID(id: str):
    return await controller.fetch_document(id)

@app.post("/agent/query")
async def query_agent(request: AgentQueryRequest):
    return await controller.ask_agent(request.query)
