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

    # Celery tasks are synchronous, so each task runs the async controller
    # logic inside its own fresh event loop. asyncio.run() creates the loop,
    # runs the coroutine, and tears the loop down cleanly — unlike the
    # deprecated get_event_loop() pattern, which breaks when no loop exists
    # in the worker thread or a previous task left a closed one behind.
    return asyncio.run(controller.process_and_ingest_pdf(file_path))
