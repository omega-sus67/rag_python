import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from celery.result import AsyncResult

from app.controllers.main_controller import MainController
from app.core.config import settings
from app.worker import process_pdf_task, celery_app

# Instantiate our central orchestrator
controller = MainController()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up pipeline resources...")
    await controller.initialize_system()
    os.makedirs(settings.upload_dir, exist_ok=True)
    yield
    print("Shutting down pipeline resources...")

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
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a PDF as a multipart upload, streams it to the shared upload
    directory, queues the ingestion task, and returns a 202 with the task ID.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Sanitize to a bare filename, then isolate each upload in a UUID
    # subdirectory: concurrent same-named uploads cannot overwrite each
    # other, and the basename stays clean for document title derivation.
    safe_name = os.path.basename(file.filename)
    upload_subdir = os.path.join(settings.upload_dir, uuid.uuid4().hex[:8])
    os.makedirs(upload_subdir, exist_ok=True)
    dest_path = os.path.join(upload_subdir, safe_name)

    # Stream to disk in 1 MB pieces so large PDFs never sit fully in memory.
    with open(dest_path, "wb") as out:
        while True:
            piece = await file.read(1024 * 1024)
            if not piece:
                break
            out.write(piece)

    # Trigger the Celery task asynchronously via Redis
    task = process_pdf_task.delay(dest_path)
    return {
        "message": "Document ingestion queued.",
        "filename": safe_name,
        "task_id": task.id
    }

@app.get("/task/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Endpoint to check the status of a background ingestion task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    result = task_result.result if task_result.ready() else None
    # Failed tasks store the raised exception as the result; stringify it so
    # the response stays JSON-serializable instead of erroring the endpoint.
    if isinstance(result, BaseException):
        result = {"error": str(result)}
    return {
        "task_id": task_id,
        "status": task_result.status, # e.g. PENDING, STARTED, SUCCESS, FAILURE
        "result": result
    }

@app.get("/docFetch/{id}", response_model=DocumentResponse)
async def getFileByID(id: str):
    return await controller.fetch_document(id)

@app.post("/agent/query")
async def query_agent(request: AgentQueryRequest):
    return await controller.ask_agent(request.query)
