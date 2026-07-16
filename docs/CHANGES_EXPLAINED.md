# Changes Explained — Interview Prep Notes

Every change made during the "make it respectable" cleanup, with the reasoning
behind it and the interview questions each one is likely to trigger. Read this
top to bottom once, then use it as a lookup before interviews.

---

## 1. Repo hygiene

### What changed
- Rewrote `.gitignore`. The old one had a line ignoring **`.gitignore` itself** —
  which does nothing once the file is tracked (git ignores the ignore rules for
  already-tracked files), but it looks like a copy-paste accident to anyone
  reading the repo. Added `ingestion.log`, `.coverage`, `.pytest_cache/`,
  `data/`, and `*.pdf` (large, often copyrighted — they don't belong in git).
- Moved the job-description PDF out of the repo (it's on your Desktop now).
- Split ~350 lines of uncommitted work into **five logical commits** (gitignore,
  parallel PDF parsing, Celery/Redis infra, ingestion logging, tests/utilities)
  instead of one "misc changes" blob.
- Added an MIT `LICENSE` (copyright line says `omega-sus67` — put your real
  name there).

### Why it matters
Interviewers *do* click through commit history. Five commits that each do one
thing, with messages explaining *why*, signal that you understand code review
culture. A repo with someone's job application PDF in the root signals the
opposite.

### Likely questions
- **"Why is `trialData/` gitignored?"** — The corpus is novels (Peter Pan,
  Harry Potter): large binaries that bloat clone size, and redistributing them
  raises copyright questions. The eval dataset (questions + expected answers)
  IS committed; the PDFs it runs against are fetched locally.
- **"Why MIT?"** — Permissive, standard for portfolio projects; lets anyone
  read/reuse without legal friction. No reason to be restrictive here.

---

## 2. Multipart file upload (`app/main.py`)

### What changed
Before, `/upload` took a JSON body with a **server-side file path**:

```python
class FilePathRequest(BaseModel):
    path: str

@app.post("/upload", status_code=202)
async def upload_file(path: FilePathRequest):
    task = process_pdf_task.delay(path.path)
```

Now it takes the actual file bytes as a multipart upload:

```python
@app.post("/upload", status_code=202)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_name = os.path.basename(file.filename)
    dest_path = os.path.join(settings.upload_dir, f"{uuid.uuid4().hex[:8]}_{safe_name}")

    with open(dest_path, "wb") as out:
        while True:
            piece = await file.read(1024 * 1024)
            if not piece:
                break
            out.write(piece)

    task = process_pdf_task.delay(dest_path)
```

### Why
Three separate reasons — know all three:

1. **It didn't work remotely.** A path-based API assumes the client and the
   server share a filesystem. The moment the API runs in Docker or on another
   machine, the client's `/home/me/doc.pdf` means nothing to the server.
2. **It was a security hole.** The old endpoint let any caller make the server
   open *any path the process could read* (`{"path": "/etc/passwd"}`). That's
   a classic **arbitrary file read / path traversal** vector. The new endpoint
   only touches bytes the client actually sent, `os.path.basename()` strips
   any directory components from the client-supplied filename, and a UUID
   prefix prevents two uploads named `report.pdf` from clobbering each other.
3. **Memory safety.** `await file.read(1MB)` in a loop streams the upload to
   disk in pieces. `await file.read()` with no size would buffer the entire
   PDF in RAM — fine for 1 MB, bad for 500 MB.

One refinement found during live verification: prefixing the UUID onto the
*filename* polluted document titles (titles derive from the file basename),
so each upload now goes into a UUID **subdirectory** instead —
`data/uploads/<uuid>/report.pdf` — keeping collision safety and clean titles.

Also fixed: `/task/status` used to return `task_result.result` raw. For a
**failed** Celery task, `.result` is the raised *exception object*, which
FastAPI cannot serialize to JSON — so checking the status of a failed task
would itself 500. It's now stringified into `{"error": ...}`.

### Likely questions
- **"How does the file get from the API container to the worker container?"**
  Both mount the same `data/uploads` volume; the API writes the file there
  and passes the *path* through Redis (small message), not the bytes.
  Payloads through a message broker should be small — you pass references,
  not blobs.
- **"Why 202 and not 200?"** 202 Accepted means "request received, processing
  not complete." The client gets a `task_id` and polls `/task/status/{id}`.
  This is the standard async-job HTTP pattern.
- **"What's `python-multipart` for?"** FastAPI parses
  `multipart/form-data` bodies with it; it's a runtime dependency of
  `UploadFile`.

---

## 3. Celery worker event loop (`app/worker.py`)

### What changed
```python
# before
loop = asyncio.get_event_loop()
if loop.is_closed():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
result = loop.run_until_complete(controller.process_and_ingest_pdf(file_path))

# after
return asyncio.run(controller.process_and_ingest_pdf(file_path))
```

### Why
`asyncio.get_event_loop()` is **deprecated** (Python 3.10+) when there is no
running loop: it either warns and creates one implicitly or raises, depending
on version and thread. Inside a Celery worker there is no ambient event loop —
each task should create one, run the coroutine, and tear it down.
`asyncio.run()` does exactly that: new loop, run to completion, close loop,
clean up async generators. The old code also had a subtle leak: loops it
created were never closed.

### Likely questions
- **"Why do you need an event loop inside Celery at all?"** The whole
  ingestion stack is async (SQLAlchemy AsyncSession over asyncpg, and
  `asyncio.to_thread` for embedding). Celery tasks are plain sync functions,
  so the task is the *bridge*: sync Celery entrypoint → `asyncio.run()` →
  async pipeline.
- **"Why not make Celery itself async?"** Celery doesn't natively execute
  coroutine tasks; `asyncio.run()` per task is the standard workaround. The
  alternative is a natively-async queue (arq, or FastAPI BackgroundTasks for
  small jobs), but Celery gives retries, routing, and monitoring for free.

---

## 4. Ingestion audit logging (`app/controllers/main_controller.py`)

### What changed
Manual file writes to a **hardcoded path in the repo root**:

```python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_file_path = os.path.join(project_root, "ingestion.log")
...
with open(log_file_path, "a") as f:
    f.write(log_line)
```

became a named logger with a `FileHandler` at a configurable path
(`settings.ingestion_log_path`, default `data/ingestion.log`):

```python
ingestion_logger = logging.getLogger("rag.ingestion")

def _configure_ingestion_logger():
    if not ingestion_logger.handlers:
        ...
        handler = logging.FileHandler(settings.ingestion_log_path)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", ...))
```

Also: `raise e` → bare `raise` in the except block.

### Why
- Manual `open(..., "a")` scattered in business logic mixes concerns; the
  `logging` module gives timestamps, levels (INFO for success, ERROR for
  failure), and lets deployment decide where logs go via one env var.
  Writing logs into the repo root is how `ingestion.log` ended up in git
  status in the first place.
- The `if not ingestion_logger.handlers` guard prevents **duplicate handlers**
  — `logging.getLogger(name)` returns a process-wide singleton, so calling
  setup twice would otherwise double every log line.
- **`raise` vs `raise e`**: bare `raise` re-raises the active exception with
  its original traceback intact. `raise e` re-raises *from the current frame*,
  which rewrites the traceback origin and makes debugging harder. Rule of
  thumb: inside `except`, use bare `raise` unless you're wrapping in a new
  exception type (then `raise New(...) from e`).

### Likely questions
- **"Why a named logger instead of the root logger?"** Namespacing
  (`rag.ingestion`) lets you set levels/handlers for your subsystem without
  affecting third-party library logging, and log lines identify their source.

---

## 5. Docker portability (`docker-compose.yml`, `Dockerfile`, `.dockerignore`, `.env.example`)

### What changed
The old worker service was this:

```yaml
worker:
  image: python:3.12-slim
  volumes:
    - .:/home/omega_sus/Desktop/rag_python
  working_dir: /home/omega_sus/Desktop/rag_python
  environment:
    - PYTHONPATH=/home/omega_sus/Desktop/rag_python:/home/omega_sus/Desktop/rag_python/.venv/lib/python3.12/site-packages
```

Three things made it single-machine-only: the volume target is *your* absolute
home path, the `PYTHONPATH` reaches **into the host's `.venv`** (Linux-built
wheels borrowed by a container — happens to work only because host and
container are both Linux x86 with Python 3.12), and there was no `api` service
at all, so `docker compose up` never actually served the API. The `Dockerfile`
existed but nothing used it.

Now:
- **`api` and `worker` both `build: .`** from the Dockerfile — dependencies are
  baked into the image, so the stack runs anywhere Docker runs.
- **`.env.example`** documents every variable; compose substitutes them with
  `${VAR:-default}` so secrets never live in the compose file.
- **Shared `./data` volume** between api and worker: the API writes uploaded
  PDFs there, the worker reads them by path. This is why multipart upload +
  Celery works across containers.
- **`hf_cache` named volume** keeps the downloaded SentenceTransformer model
  between rebuilds (~400 MB you don't want to re-download every `up --build`).
- **GPU behind a profile**: `worker-gpu` only starts with
  `docker compose --profile gpu up`. Without the profile, the default CPU
  worker runs — previously the compose file demanded an NVIDIA device and
  would fail on machines without one.
- **Healthcheck + `depends_on: condition: service_healthy`** on Postgres:
  plain `depends_on` only waits for the *container* to start, not for
  Postgres to accept connections — a classic startup race.
- **`.dockerignore`**: `COPY . .` was copying `.git`, `.venv`, and the PDF
  corpus into the image. Image bloat, slow builds, and your git history
  inside a shippable artifact.
- **YAML anchor `&app_env`** defines the app environment once and `*app_env`
  reuses it for api/worker/worker-gpu — no drift between the three copies.

### Likely questions
- **"Why pgbouncer, and why transaction pool mode?"** Postgres forks a
  process per connection (~ MBs each); hundreds of short-lived connections
  from async apps and worker bursts would exhaust it. pgbouncer multiplexes
  many client connections onto a small pool of real ones. *Transaction mode*
  returns the server connection to the pool after each transaction (vs
  session mode, which pins it) — highest reuse, with the caveat that
  session-level state (prepared statements, `SET` variables, advisory locks)
  can't be relied on across transactions.
- **"Why does the DB also expose port 5435 directly?"** Admin operations
  (creating the database/extension at bootstrap) and anything needing
  session state bypass the pooler — that's the `db_direct_port` setting in
  `config.py`.
- **"Why not pass the PDF bytes through Redis?"** Brokers are for small
  control messages. Blobs go to shared storage (here a volume; in production
  S3) and you enqueue the *reference*.

---

## 6. Retrieval-quality evaluation (`app/eval/retrieval_eval.py`) — THE BIG ONE

This is the section to know cold. It turns "I built a fancy chunker" into
"I measured my chunker, found it losing to a naive baseline, diagnosed why,
fixed it, and can tell you exactly where it wins and loses."

### What was built
- **Two strategies, same everything else.** The eval ingests each corpus PDF
  twice: once through the real pipeline (AST parse → sliding-window semantic
  chunking → merge pass) and once through a control — fixed 400-token windows
  with 100-token overlap (reusing `TokenSizeOptimizer.optimize_block`, i.e.
  the exact splitter the pipeline already uses for oversized blocks). Same
  embedding model, same pgvector cosine search. The *only* variable is the
  chunking strategy — that's what makes it an experiment instead of a demo.
- Both corpora live in the same tables under namespaced IDs
  (`eval_semantic_<sha256>`, `eval_baseline_<sha256>`) so they never collide
  with real documents; searches filter with `dbFile.id.like("eval_semantic_%")`.
- **Dataset:** 18 questions over 4 documents (Peter Pan, White Nights,
  Gift of the Magi, ISO 27001). Every `expected_substrings` entry was verified
  to literally occur in the extracted text *before* being added — an expected
  answer that never appears in the corpus can only produce false misses.
- **Hit criterion:** a retrieved chunk is a hit only if it (a) belongs to the
  question's target document AND (b) contains an expected substring. The
  document constraint kills false positives (the word "hair" appears in
  Peter Pan too, not just Gift of the Magi).

### The metrics
- **hit-rate@k** — fraction of queries with ≥1 hit in the top k. Answers
  "if I stuff the top-k chunks into the LLM prompt, does the answer exist
  in there?" — which is exactly RAG's job.
- **MRR (mean reciprocal rank)** — average of 1/rank of the *first* hit
  (1/1, 1/2, 1/3...; 0 if none in top 5). Rewards putting the right chunk
  first. Two systems can have identical hit-rate@5 and very different MRR.

### The story arc (memorize this)
1. **First run: semantic LOST.** hit-rate@5 was 38.9% vs baseline's 50.0%.
2. **Diagnosis.** Dumped chunk statistics: the semantic index held 4,508
   chunks averaging 268 chars — **688 of them under 80 chars, minimum 1
   character**. Top results for missed queries were fragments like
   `'Peter Pan'`, `'WHITE NIGHTS'`, `'Della went to him.'`. Root cause #1:
   short chunks embed deceptively close to short queries (both are compact,
   name-heavy strings — cosine similarity loves that). Root cause #2, the
   actual bug: in `_decompose_node`, the min-size guard
   (`min_sentences`/`min_words`) only applied when a *boundary* fired; the
   remainder-flush branch (`if current_sentences:`) appended whatever was
   left with **no size check**. Fiction is thousands of one-sentence dialogue
   paragraphs; every one of them is a single AST node that lands in the
   remainder branch → tiny chunk.
3. **Fix:** `_merge_undersized_chunks()` — a post-pass that folds prose
   chunks below `min_words` into their predecessor, strips the duplicate
   `Context:` prefix from the absorbed text, and resets the merged chunk's
   embedding to `None` so the final embedding pass re-encodes the combined
   text. Tables and lists are exempt: they are structurally distinct and
   legitimately small (merging a table into a paragraph would be wrong —
   and two existing tests correctly failed when the first version tried).
4. **Result:** semantic hit-rate@5 went 38.9% → **50.0%**, MRR 0.272 → 0.312.
   Parity with baseline overall; wins on Magi and Peter Pan, loses slightly
   on ISO 27001, both fail on the 670k-char anthology.
5. **One metric bug found too:** semantic's #1 result for "who is the captain
   of the Jolly Roger" was *"I am James Hook, captain of the Jolly Roger"* —
   a perfect answer scored as a miss because the substrings only listed
   "Jas. Hook"/"Captain Hook". Added "James Hook". Lesson: inspect your
   eval's failures before trusting its numbers; metric bugs and system bugs
   look identical in a summary table.

### Likely questions
- **"Why substring matching instead of an LLM judge?"** Deterministic, free,
  offline, and reproducible in CI. An LLM judge scores *answer* quality;
  substring-plus-document-constraint is a sharp proxy for *retrieval*
  quality, which is the component being measured. Judge-based answer eval is
  the natural next layer, but you evaluate components before pipelines.
- **"Why is hit-rate@1 only ~22%?"** Small embedding model, novel-length
  documents, and questions phrased unlike the prose ("What is Wendy's full
  name?" vs a passing mention in a paragraph). The honest takeaways: absolute
  numbers depend on corpus difficulty; the *comparison* between strategies on
  identical conditions is what's informative.
- **"Your custom chunker only ties the baseline. Why keep it?"** (a) Before
  measuring, it was *losing* — the eval bought back 11 points; (b) it wins
  where documents have exploitable structure and at MRR on shorter docs;
  (c) a 100-token-overlap baseline stores ~33% redundant tokens, the semantic
  chunker doesn't; (d) knowing precisely where it doesn't help (fiction with
  misdetected headings) is the finding — most people can't say where their
  clever component underperforms.
- **"Why does the semantic strategy struggle on novels?"** pymupdf4llm
  misclassifies drop caps and short emphatic lines as headings, so AST
  breadcrumbs on fiction become noise (`Context: "Dive!"`). Structure-aware
  parsing needs actual structure; on flat prose it can only add overhead.

---

## 7. Chunker fix details (`app/engines/hierarchical_chunker.py`)

```python
def _merge_undersized_chunks(self, chunks, min_words=settings.min_words):
    merged = []
    for chunk in chunks:
        content = self._content_of(chunk.text)   # strips "Context: ...\nContent:"
        if (merged and self._is_prose(chunk) and self._is_prose(merged[-1])
                and len(content.split()) < min_words):
            prev = merged[-1]
            prev.text = f"{prev.text} {content}"
            prev.embeddings = None                # force re-embed of merged text
            prev.metadata["token_count"] = self.token_optimizer.count_tokens(prev.text)
        else:
            merged.append(chunk)
    return merged
```

Details interviewers may poke at:
- **Why reset `embeddings = None`?** The absorbing chunk's old vector no
  longer represents its new text. In hybrid mode everything re-embeds anyway;
  in vector-math mode only `None`-embedding chunks re-embed — resetting makes
  the merge correct under both strategies.
- **Why merge into the *previous* chunk?** A tiny dialogue line usually
  continues the preceding narration. Forward-merging was the alternative;
  backward keeps the pass single-direction and O(n).
- **Edge case:** a tiny *first* chunk has no predecessor and survives —
  covered by a unit test.

Also in this commit: embedding batch size 256 → 64 and made configurable
(`embedding_batch_size`). 256 was fine for short sentences but OOM'd a 6 GB
GPU on 400-token chunks — batch memory scales with batch × sequence length,
a classic thing to know.

---

## 8. CI + lint (`.github/workflows/ci.yml`, `pyproject.toml`)

- Two jobs: `lint` (ruff) and `test` (58 tests). Every push and PR.
- **CPU-only torch trick:** `pip install torch --index-url
  https://download.pytorch.org/whl/cpu` *before* `-r requirements-dev.txt`.
  Default torch pulls multi-GB CUDA wheels; CI has no GPU, so this cuts
  install time/size drastically. Worth mentioning in any "how do you keep CI
  fast" conversation.
- Tests need no database or model download — the suite mocks the vector
  engine and DB sessions, which is *why* it can run in CI in ~30s.
- Ruff config keeps only correctness rules (pyflakes + pycodestyle errors),
  line length 160 — linting for bugs, not style wars. First run found 27
  issues: unused imports, an f-string with no placeholders, an unused
  variable.
- **Found via CI prep:** `tiktoken`, `rich`, and `httpx` were imported but
  missing from `requirements.txt` — it only worked because the local venv
  happened to have them. A fresh `pip install -r requirements.txt` would
  have crashed. This is exactly what CI on a clean machine exists to catch.

---

## 9. README rewrite

- **Deleted the "Resume-Ready Technical Highlights" section.** A README that
  addresses recruiters signals the project exists for the resume. The same
  facts now live in the architecture narrative, addressed to users.
- The evaluation section leads with what the numbers *mean*, including the
  unflattering parts (baseline parity, White Nights failure). If an
  interviewer has read the README, every question about weaknesses is one
  you've already answered on your own terms.
- Quickstart is now the docker path with a real multipart `curl` — every
  command in it was actually run against the stack before being written down.

---

## Quick reference: numbers to remember

| Fact | Value |
|---|---|
| Tests | 58, all mocked, ~30s in CI |
| Eval dataset | 18 questions, 4 documents, substrings verified against extracted text |
| Semantic before → after merge fix (hit@5) | 38.9% → 50.0% |
| Tiny chunks removed by fix | 688 of 4,508 (min length was 1 char) |
| Baseline | 400-token windows, 100-token overlap, same embedder |
| Embedding model | BAAI/bge-base-en-v1.5, 768-dim, HNSW (m=16, ef_construction=200) |
| Batch size | 256 → 64 (6 GB GPU OOM on 400-token sequences) |
