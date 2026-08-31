import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
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
    # Schema bootstrap is a deploy-time step in production: concurrent replicas
    # booting at once would otherwise race on CREATE TABLE. Local dev opts in
    # via BOOTSTRAP_DB_ON_STARTUP=true so `docker compose up` still just works.
    if settings.bootstrap_db_on_startup:
        await controller.initialize_system()
    os.makedirs(settings.upload_dir, exist_ok=True)
    yield
    print("Shutting down pipeline resources...")
    # Hand the broker its connections back on a clean shutdown, so a redeploy
    # does not leave sockets pinned against a capped free-tier client limit.
    global _redis_probe_client
    if _redis_probe_client is not None:
        await _redis_probe_client.aclose()
        _redis_probe_client = None
    await controller.db_manager.engine.dispose()

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

@app.get("/health")
async def health():
    """
    Dependency-aware health check.

    Returns 503 when Postgres or Redis is unreachable, because a web process
    that answers 200 while its database is down is worse than one that admits
    it: the platform's health check is what decides whether to keep routing
    traffic here or restart the container.

    Each probe is capped so a hung dependency cannot make the health check
    itself hang, which would look to the platform like a stuck process.

    The 5-second budget is measured, not guessed: a cold connection to managed
    Postgres costs ~2.0s (TLS handshake plus endpoint wake) against ~0.6s once
    the pool is warm. A 2s cap sat exactly on that cold-start number and made
    the check fail right after boot — which on a PaaS means a restart loop,
    since the platform kills instances that fail their health check.
    """
    async def probe(coro):
        try:
            await asyncio.wait_for(coro, timeout=5.0)
            return "ok", None
        except asyncio.TimeoutError:
            # TimeoutError stringifies to "", which reads as a passing check
            # with an empty error. Say what actually happened.
            return "error", "timed out after 5s"
        except Exception as exc:
            return "error", str(exc) or exc.__class__.__name__

    db_status, db_error = await probe(controller.db_manager.ping())
    redis_status, redis_error = await probe(ping_redis())

    healthy = db_status == "ok" and redis_status == "ok"
    body = {
        "status": "ok" if healthy else "degraded",
        "checks": {
            "database": {"status": db_status, "error": db_error},
            "redis": {"status": redis_status, "error": redis_error},
        },
    }
    return JSONResponse(status_code=200 if healthy else 503, content=body)

# One lazily-created client, reused for every probe.
#
# The first version opened and closed a fresh connection per health check, which
# exhausted the broker during testing: `max number of clients reached`. Managed
# Redis free tiers cap total clients (Redis Cloud is ~30), and a health endpoint
# is the most frequently hit route on the service — the platform polls it, and a
# keep-warm pinger polls it again. Per-request connections turn the cheapest
# endpoint into the one that runs the broker out of connections.
#
# Deliberately not Celery's broker pool: that pool belongs to the producer side,
# and a stale connection in it would make the probe report on the pool's health
# rather than on Redis being reachable right now.
_redis_probe_client = None


def _get_redis_probe_client():
    global _redis_probe_client
    if _redis_probe_client is None:
        import redis.asyncio as aioredis

        _redis_probe_client = aioredis.from_url(
            settings.redis_url,
            max_connections=2,
            socket_connect_timeout=5,
            socket_timeout=5,
            health_check_interval=30,
        )
    return _redis_probe_client


async def ping_redis() -> None:
    """PINGs Redis over the shared probe connection."""
    await _get_redis_probe_client().ping()

@app.post("/upload", status_code=202)
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a PDF as a multipart upload, parks the bytes in Postgres, queues the
    ingestion task by blob id, and returns a 202 with the task ID.

    The bytes go to Postgres rather than to disk because web and worker are
    separate containers in production with separate filesystems — a path handed
    across the queue would not resolve on the other side. They do not ride the
    queue itself either: Redis free tiers cap at ~30 MB total, so a queue full
    of PDF payloads would evict the queue.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = os.path.basename(file.filename)

    # Read in 1 MB pieces and enforce the cap as we go, so an oversized upload
    # is rejected mid-stream instead of after we have buffered all of it.
    pieces = bytearray()
    while True:
        piece = await file.read(1024 * 1024)
        if not piece:
            break
        pieces.extend(piece)
        if len(pieces) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"PDF exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB upload limit.",
            )

    blob_id = await controller.db_manager.store_upload_blob(safe_name, bytes(pieces))

    # Trigger the Celery task asynchronously via Redis
    task = process_pdf_task.delay(blob_id)
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
