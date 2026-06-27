# Scaling to 10,000 Users: Implementation Task List

This to-do list outlines the engineering steps to scale the FastAPI backend.

- [ ] **Step 1: Unblock the Event Loop (Async Fixes)**
  - [ ] Wrap `SemanticEngine.model.encode` in `asyncio.to_thread` or a thread pool.
  - [ ] Ensure `DatabaseManager` and `HierarchicalRAGRetriever` calls use async properly without hanging the loop.

- [ ] **Step 2: Database Optimizations**
  - [ ] Initialize `Alembic` for schema migrations (remove `create_all()` on startup).
  - [ ] Add an `HNSW` vector index to the `dbChunk.embeddings` column to speed up cosine similarity search.
  - [ ] Introduce **PgBouncer** (via Docker) to handle connection pooling for 10,000 incoming requests.

- [ ] **Step 3: Message Queue & Background Workers**
  - [ ] Install and configure **Redis** as a message broker.
  - [ ] Install **Celery** (or ARQ/RQ) for background task processing.
  - [ ] Move the `process_and_ingest_pdf` logic out of the FastAPI endpoint and into a Celery task.
  - [ ] Update FastAPI ingestion endpoint to return a `202 Accepted` with a `task_id` for polling.

- [ ] **Step 4: Frontend Framework Setup**
  - [ ] Choose and initialize the frontend framework (e.g., Next.js or Streamlit) in a new `ui/` directory.
  - [ ] Implement UI for uploading PDFs.
  - [ ] Implement UI for the chat interface to query the ReAct agent.
  - [ ] Connect the frontend to the FastAPI backend (handling CORS if necessary).

- [ ] **Step 5: Dockerization & Orchestration**
  - [ ] Write a `docker-compose.yml` to orchestrate the services: FastAPI, Celery Worker, Redis, Postgres, PgBouncer, and the Frontend.
  - [ ] Ensure environment variables (`.env`) are correctly passed to all containers.

- [ ] **Step 6: Load Testing (Optional but Recommended)**
  - [ ] Write a `locustfile.py` to simulate concurrent API requests.
  - [ ] Run Locust to verify the API remains stable under heavy load without dropping connections.
