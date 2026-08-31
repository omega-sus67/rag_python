#!/usr/bin/env sh
#
# Single-container fallback: runs the Celery worker and the API side by side,
# for platforms whose free tier only offers a web service.
#
# This is a deployment compromise, not the architecture. The system is designed
# as two independently scalable processes and runs that way under
# docker-compose, under the two-service render.yaml block, and on any platform
# with a worker tier. Collapsing them means the worker's CPU-bound ingestion
# competes with the API's event loop, and neither can scale without the other —
# which is exactly the kind of thing to say out loud in an interview rather
# than hide.
#
# Usage: set the container command to `sh scripts/start-combined.sh`.

set -e

echo "[start-combined] launching Celery worker..."
# --pool=solo, not the default prefork.
#
# Prefork forks a child process per concurrency slot, so the interpreter, the
# imports and the loaded libraries are all paid for twice inside one container.
# On a 512 MB free tier that is the difference between ingesting a document and
# being OOM-killed by the platform mid-task — which is exactly what happened:
# Render killed the worker, and because Celery acknowledges a task before
# running it, the task died with no result and no retry, leaving /task/status
# reporting PENDING forever.
#
# solo runs the task in the worker process itself. It gives up parallelism the
# free tier could not afford anyway (concurrency was already 1).
#
# --without-gossip/mingle/heartbeat drop inter-worker chatter that a single
# worker has no use for, and each of those costs broker connections on a tier
# that caps total clients.
celery -A app.worker.celery_app worker --loglevel=info --pool=solo \
    --without-gossip --without-mingle --without-heartbeat &
WORKER_PID=$!

echo "[start-combined] launching API on port ${PORT:-8000}..."
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" &
API_PID=$!

# Supervise both. If either dies, bring the whole container down so the platform
# restarts it as a unit.
#
# This matters more than it looks. Without it, a dead worker leaves the API
# happily answering 200 on /health while nothing drains the queue: uploads are
# accepted with 202 and then sit in PENDING forever. A half-dead service that
# still passes its health check is worse than one that is plainly down, because
# nothing triggers a restart. (Observed exactly this during testing.)
#
# Polling rather than `wait -n`, which is a bashism and this runs under sh.
while kill -0 "$WORKER_PID" 2>/dev/null && kill -0 "$API_PID" 2>/dev/null; do
    sleep 5
done

if kill -0 "$WORKER_PID" 2>/dev/null; then
    echo "[start-combined] API exited — stopping worker and failing the container."
else
    echo "[start-combined] worker exited — stopping API and failing the container."
fi

kill -TERM "$WORKER_PID" "$API_PID" 2>/dev/null || true
wait 2>/dev/null || true

# Non-zero so the platform treats this as a crash and restarts, rather than as
# a clean shutdown.
exit 1
