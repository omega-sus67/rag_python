import asyncio
import os
import shutil
import ssl
import tempfile

from celery import Celery

from app.controllers.main_controller import MainController
from app.core.config import settings

# Connect to the Redis Broker and Result Backend
REDIS_URL = settings.redis_url
celery_app = Celery("rag_tasks", broker=REDIS_URL, backend=REDIS_URL)

# Managed Redis hands out a rediss:// URL, and Celery does not infer TLS from
# the scheme — it needs the ssl options set explicitly on both the broker and
# the result backend, or every connection fails with a protocol error.
if REDIS_URL.startswith("rediss://"):
    _ssl_options = {"ssl_cert_reqs": ssl.CERT_REQUIRED}
    celery_app.conf.broker_use_ssl = _ssl_options
    celery_app.conf.redis_backend_use_ssl = _ssl_options

# Connection budget.
#
# Managed Redis free tiers cap *total* clients (Redis Cloud Essentials is ~30),
# and Celery is not frugal by default: broker_pool_limit alone is 10, before the
# result backend, the control plane, and every API process publishing tasks.
# Exhausting that cap does not degrade gracefully — the broker starts refusing
# every new client with "max number of clients reached", which takes down
# publishing and consuming at once. Hit exactly that during deployment testing.
#
# These limits trade a little throughput headroom for staying inside the tier.
# Raise them together with the tier when load testing needs the concurrency.
celery_app.conf.broker_pool_limit = 2
celery_app.conf.broker_transport_options = {"max_connections": 4}
celery_app.conf.result_backend_transport_options = {"max_connections": 4}
celery_app.conf.redis_max_connections = 4


async def _ingest_from_blob(blob_id: str) -> dict:
    """
    Pulls the uploaded PDF's bytes out of Postgres, materializes them in this
    container's own temp directory, and runs the normal ingestion pipeline.

    The temp file exists because the parser (PyMuPDF) works on a path, not a
    buffer. It is local to the worker and deleted afterwards, so no shared
    filesystem is implied between web and worker.
    """
    controller = MainController()

    blob = await controller.db_manager.fetch_upload_blob(blob_id)
    if blob is None:
        # Either the id was never valid, or a retry arrived after a previous
        # attempt already consumed and deleted the blob.
        raise FileNotFoundError(f"No queued upload found for blob id '{blob_id}'.")

    filename, content = blob

    # Keep the original filename inside the temp directory: the document title
    # is derived from the basename, so a random temp name would corrupt titles.
    workdir = tempfile.mkdtemp(prefix="ingest_")
    file_path = os.path.join(workdir, filename)
    with open(file_path, "wb") as handle:
        handle.write(content)

    try:
        return await controller.process_and_ingest_pdf(file_path)
    finally:
        # Always reclaim both the temp file and the handoff row, including on
        # failure — otherwise a failing document leaks its bytes into a 512 MB
        # database forever.
        shutil.rmtree(workdir, ignore_errors=True)
        await controller.db_manager.delete_upload_blob(blob_id)


@celery_app.task(bind=True, name="process_pdf_task")
def process_pdf_task(self, blob_id: str):
    """
    Background task to process and ingest a PDF.
    Runs asynchronously inside a dedicated worker process.

    Accepts the id of an upload blob stored in Postgres. A filesystem path is
    still accepted for local CLI use, where web and worker are the same machine.
    """
    # Celery tasks are synchronous, so each task runs the async controller
    # logic inside its own fresh event loop. asyncio.run() creates the loop,
    # runs the coroutine, and tears the loop down cleanly — unlike the
    # deprecated get_event_loop() pattern, which breaks when no loop exists
    # in the worker thread or a previous task left a closed one behind.
    if os.path.exists(blob_id):
        controller = MainController()
        return asyncio.run(controller.process_and_ingest_pdf(blob_id))

    return asyncio.run(_ingest_from_blob(blob_id))
