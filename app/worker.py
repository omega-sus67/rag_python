import os
import asyncio
from celery import Celery
from app.controllers.main_controller import MainController

# Connect to the Redis Broker and Result Backend
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("rag_tasks", broker=REDIS_URL, backend=REDIS_URL)

@celery_app.task(bind=True, name="process_pdf_task")
def process_pdf_task(self, file_path: str):
    """
    Background task to process and ingest a PDF.
    Runs asynchronously inside a dedicated worker process.
    """
    controller = MainController()
    
    # Celery tasks are synchronous by default.
    # We initialize an asyncio event loop to run our async controller logic.
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    result = loop.run_until_complete(controller.process_and_ingest_pdf(file_path))
    return result
