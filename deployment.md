# Deployment

How this service goes from "runs on my laptop" to a public URL, what had to change
to make that possible, and what is still weak about it.

**Status: the deployment was built, measured, and then retired on purpose.** Every fix
below is real and still in the code. The URL is gone because the only free host that
fit required `PDF_PARSER=fast`, which disables the hierarchical parsing this project
exists to demonstrate — the arithmetic is in §4.7. A demo that quietly turns off the
feature it is demonstrating is a worse artifact than no demo.

Written to be read start-to-finish once, then used as a runbook if it is ever revived.

---

## 1. Why it could not be deployed as-is

Five things blocked deployment. Four were latent bugs that local Docker Compose was
actively hiding, which is the interesting part: **Compose was not a smaller version
of production, it was a different architecture.** Every blocker below is a place
where Compose papered over an assumption that a real platform does not grant you.

| # | Blocker | Why Compose hid it |
|---|---|---|
| 1 | Connection URL was assembled from `user`/`password`/`host`/`port`/`name` | Locally those five fields exist. Managed Postgres gives you *one* URL, with libpq-only query params on it that **asyncpg rejects outright**. |
| 2 | `create_tables()` issued `CREATE DATABASE` | Locally you are superuser on your own container. On Neon you get a database that already exists and a role that cannot create siblings. |
| 3 | Web wrote the PDF to disk and passed the **path** through Redis | Compose bind-mounts `./data` into both containers, so the path resolves. In production web and worker are separate containers on separate hosts, and the worker gets a path to a file that does not exist. |
| 4 | `torch` + `sentence-transformers` + `bge-base-en-v1.5` | A laptop with 16 GB and a GPU does not notice ~2.5 GB of CUDA wheels and ~1.5 GB resident. A 512 MB free-tier container cannot start it. |
| 5 | `/` returned a static string; port was hardcoded to 8000 | Nothing polls your laptop for health, and nothing reassigns its ports. |

Blocker 3 is the architectural one. The other four are configuration.

---

## 2. What changed, and why

### 2.1 One connection URL, and the asyncpg TLS trap

`DATABASE_URL` now overrides the discrete fields when set
([app/core/config.py](app/core/config.py)).

The subtlety is TLS. Neon issues:

```
postgresql://user:pass@host/db?sslmode=require&channel_binding=require
```

Both `sslmode` and `channel_binding` are **libpq** parameters. asyncpg is not libpq —
it raises on them rather than ignoring them. So `async_database_url` drops the query
string entirely and rewrites the scheme to `postgresql+asyncpg`, and the TLS intent is
re-applied separately through `db_connect_args` as asyncpg's own `ssl=` parameter,
passed to `create_async_engine(connect_args=...)`.

This is why the fix is "drop the whole query string", not "special-case `sslmode`" —
there were two offending params, and a provider could add a third.

Also set: `pool_pre_ping=True`. Neon closes idle connections; without pre-ping the
first query after an idle period dies on a stale socket.

### 2.2 Schema bootstrap split from database creation

`create_tables()` no longer opens a second engine against the `postgres` maintenance
database ([app/db/database_manager.py](app/db/database_manager.py)). It now does only:

```
CREATE EXTENSION IF NOT EXISTS vector;   -- idempotent
Base.metadata.create_all                 -- idempotent
```

Exposed as `python cli.py init-db`, run as an explicit deploy step.

It is deliberately **not** run on app startup in production
(`BOOTSTRAP_DB_ON_STARTUP=false`). Two web replicas booting simultaneously would race
on `CREATE TABLE`. Local Compose sets it to `true` because one container, and
convenience wins there.

### 2.3 Upload transport: bytes through Postgres, id through the queue

This is the real change.

**Before:** `/upload` streamed the PDF to `settings.upload_dir` and put the *path* on
the queue. Correct locally, broken anywhere web and worker do not share a disk.

**After:** `/upload` reads the bytes, writes them to an `upload_blobs` table, and puts
the **blob id** on the queue. The worker fetches the row, materializes it to its own
container-local temp file (PyMuPDF needs a path, not a buffer), ingests, and then
deletes both the temp file and the row in a `finally` block.

Two rejected alternatives, and why:

- **Bytes on the queue.** Redis free tier is 30 MB *total*. A queue of PDF payloads
  evicts the queue. A queue should carry references, not cargo.
- **Object storage (R2 / Supabase Storage).** Architecturally the right answer, and
  what this should become. Rejected for now purely on setup time: another account,
  another set of credentials, another SDK. Postgres was already a dependency.

Honest cost of the choice: the web process now holds the whole PDF in memory while
reading it, where before it streamed to disk in 1 MB pieces. That is bounded by
`MAX_UPLOAD_BYTES` (25 MB default), which is also what keeps one upload from eating a
512 MB free-tier database. Both limits are the same knob, which is a little crude.

### 2.4 Pluggable embeddings — the thing that makes it fit

New [app/engines/embeddings.py](app/engines/embeddings.py) puts one seam between the
chunker and whatever computes vectors:

- `local` — sentence-transformers in-process. What the README's hit@5 / MRR numbers
  were measured with. Needs torch.
- `gemini` — hosted API over HTTP. No torch, no weights.

`SemanticEngine` now delegates to a provider instead of owning a `SentenceTransformer`
directly, and the `sentence_transformers` import moved *inside* the local provider so
the production image can omit the package entirely.

Dimensions are the constraint that makes or breaks a provider swap. The column is
`Vector(768)`, matching `bge-base-en-v1.5`, so a hosted model must return exactly 768
or the stored vectors and the new ones are not comparable and inserts fail.

`python cli.py check-embeddings` exists to verify that in five seconds, and it earned
its keep immediately — see §2.4.1.

Providers, and how each reaches 768:

| Provider | Model | Dimensions |
|---|---|---|
| Jina (default) | `jina-embeddings-v2-base-en` | 768 natively |
| Together | `BAAI/bge-base-en-v1.5` | 768 — literally the model the eval used |
| OpenAI | `text-embedding-3-small` | 1536, truncated via `dimensions: 768` |
| Google | `gemini-embedding-001` | 3072, truncated via `outputDimensionality: 768` |

Truncated vectors are not unit length, so `embed()` re-normalizes; every provider
therefore returns unit vectors and cosine distance means the same thing on all of them.

### 2.4.1 Why the provider is generic — a lesson paid for in real time

The original plan named one vendor. Three things went wrong in sequence:

1. `text-embedding-004`, the model the plan was written around, had been **retired**.
   The API returns 404.
2. Its replacement returns 3072 dimensions, not 768 — every insert would have failed.
3. Then the Google account itself was **blocked**. `ListModels` returned 200 while
   every embedding and generation call returned
   `403 Your project has been denied access`. Read access worked; nothing billable did.

Only the third one is unfixable in code, and it is the one that matters: **a hosted
dependency can revoke you between writing the config and shipping it.** So the
embedding seam is now a generic OpenAI-wire-format client
(`OpenAICompatibleEmbeddingProvider`) rather than a vendor class. Jina, Together,
DeepInfra, Mistral, and OpenAI are all reachable by changing two environment
variables, with no new code.

The same property already held on the LLM side by accident: `llm.py` had an
OpenAI-compatible branch, so moving the agent from Gemini to Groq was pure
configuration.

That is the honest version of "pluggable" — not a nice abstraction, but the thing that
turned a hard stop into an env-var edit.

The API provider also does two things a naive client would not:
- Batches at 100 items (the endpoint's cap) — a 300-chunk document is 3 calls, not one
  400 error.
- Retries 429 and 5xx with exponential backoff — otherwise one rate-limit response
  fails an entire document ingestion.

`requirements-prod.txt` is the torch-free set the Docker image installs.
`requirements.txt` keeps torch for local dev, eval, and CI, selected via the
`REQUIREMENTS` build arg so Compose still builds the local-model image.

### 2.5 Health check, `$PORT`, TLS Redis

`GET /health` ([app/main.py](app/main.py)) probes Postgres and Redis and returns
**503** if either is down, so the platform stops routing to a half-dead instance
instead of letting it serve errors.

The timeout on those probes is measured rather than guessed, and this caught a real
bug during testing:

```
ping 1: 2.008s  (cold)
ping 2: 0.689s  (warm)
ping 3: 0.596s  (warm)
```

The first version used a 2.0s cap. A cold connection costs 2.008s — TLS handshake plus
Neon endpoint wake. So the health check failed *immediately after every boot*, and on a
PaaS a failing health check means the instance gets killed and restarted: a restart
loop that never stabilizes. Budget is now 5s. `asyncio.TimeoutError` also stringifies
to `""`, which rendered as an errored check with a blank reason; it now says
`timed out after 5s`.

`Dockerfile` `CMD` is in shell form so `${PORT:-8000}` expands at container start.

Celery sets `broker_use_ssl` / `redis_backend_use_ssl` when `REDIS_URL` starts with
`rediss://` — Celery does not infer TLS from the scheme on its own.

---

## 3. Verification actually performed

Run against the real managed services, not mocks:

| Check | Result |
|---|---|
| asyncpg → Neon over the rewritten URL | PostgreSQL **17.11**, pgvector **0.8.0** |
| `python cli.py init-db` on an empty database | created `files`, `doc_chunks`, `upload_blobs` + `ix_doc_chunks_embeddings_hnsw` |
| `GET /health`, everything up | `200`, `{"status":"ok"}` |
| `GET /health`, Redis pointed at a dead port | `503`, `db=ok`, `redis=error` — discriminates correctly |
| `POST /upload` → worker → ingest, end to end | `202` → `SUCCESS` in ~21s; 12 chunks, 768-dim, title `1-The_Gift_Of_The_Magi_0` |
| `upload_blobs` after ingest | `0` rows — handoff cleaned up |
| Files written to shared upload dir during that run | `0` — no shared-filesystem dependency remains |
| Vector search on Neon | returns semantically correct chunks |
| Single-container mode (`scripts/start-combined.sh`), the free-tier deploy shape | `/health` 200, full upload→ingest of a 290 KB novel: **293 chunks**, 0 leftover blobs |
| Full ingest via **hosted** embeddings (Jina), production config | `iso27001.pdf` → **196 chunks**, 768-dim, 0 orphans, 0 leftover blobs |
| Corpus after testing | 4 documents / **1,384 chunks** on Neon, no orphaned rows |
| Redis client footprint, 1 worker + API idle | **10** of a ~30 free-tier cap |
| `/docs`, `/openapi.json` | 200 — Swagger UI serves |
| `pytest` | **58 passed** |
| `ruff check .` | clean |

Ingestion timing worth knowing before Phase 2: that novel produced 3,235 sentences and
293 chunks, and sentence embedding alone took ~87 s on CPU. Ingestion is
embedding-bound, not parse-bound or database-bound — which is the first hypothesis the
load testing should try to falsify.

**Not verified: the `gemini` embedding provider.** There is no API key in `.env`, so
that path has never executed. It fails loudly rather than mysteriously without a key,
and `python cli.py check-embeddings` smoke-tests it in about five seconds — **run that
before deploying**, because production is the first place this code path runs.

---

## 4. Runbook — getting the URL

Postgres and Redis are already provisioned (Neon + Redis Cloud, both `ap-southeast-1`).

### Step 0 — get two API keys and verify the embedding one

Both have free tiers and neither asks for a card.

| What | Where | Goes into |
|---|---|---|
| Embeddings | <https://jina.ai/embeddings/> → API key | `EMBEDDING_API_KEY` |
| LLM (the agent) | <https://console.groq.com/keys> | `LLM_API_KEY` |

Add to `.env`, single-quoting anything containing `&`:

```
EMBEDDING_API_KEY='jina_...'
LLM_API_KEY='gsk_...'
```

Then verify — this is the step that catches a dead model, a wrong dimension, or a
blocked account before Render does:

```bash
python cli.py check-embeddings
```

Must report **768 dimensions** and an L2 norm of ~1.0. If the dimension differs, stop —
the stored vectors and new ones would not be comparable and inserts would fail.

### Step 1 — bootstrap the schema

Already done against your Neon database, but it is idempotent and re-runnable:

```bash
python cli.py init-db
```

### Step 2 — deploy

Push to GitHub, then on Render: **New → Blueprint**, point it at the repo. It reads
[render.yaml](render.yaml), which ships in its free single-service form — one Web
Service running the API and worker together via `scripts/start-combined.sh`. No card
required. The two-service block at the bottom of that file is the upgrade path once
the ~$7/mo worker is worth paying for.

Set the four secrets in the dashboard (declared `sync: false`, so they are never
committed): `DATABASE_URL`, `REDIS_URL`, `EMBEDDING_API_KEY`, `LLM_API_KEY`.

Region is pinned to **Singapore** in the blueprint: Neon and Redis are both
`ap-southeast-1`, and a US region would add ~200 ms to every database round trip.

### Step 3 — verify the deployment

```bash
curl -i https://<your-service>.onrender.com/health
curl -F "file=@trialData/1-the_gift_of_the_magi_0.pdf" https://<your-service>.onrender.com/upload
curl https://<your-service>.onrender.com/task/status/<task_id>
```

`/health` must be `200` before anything else is worth trying.

### Step 4 — the link to actually share

Put **`/docs`** on the résumé, not `/`.

FastAPI serves an interactive Swagger UI there: every endpoint, its schema, and a
**Try it out** button. A reader can upload a PDF and run a query without a terminal.
`/` returns a one-line JSON greeting, which is a much weaker first impression for a
backend project with no frontend.

### Keeping it warm

Render's free plan spins the instance down after ~15 minutes idle, and the next request
pays a cold start of roughly a minute. A recruiter clicking a link that appears to hang
is the failure mode to avoid.

Point a free uptime pinger (UptimeRobot, cron-job.org) at `/health` every 10 minutes.
That is also a genuinely useful signal: `/health` returns 503 when Postgres or Redis is
down, so the pinger doubles as real monitoring rather than just a keep-alive.

Note the honest tension: a keep-warm pinger means the free instance is never idle, and
any latency number measured on it is a *warm* number. Say which one you are quoting.

---

## 4.5 Bugs the hosted-embedding switch exposed

Every one of these was invisible while embeddings ran in-process. Moving that call
onto the network turned latent assumptions into failures — which is the argument for
doing it before load testing rather than after.

**1. Partial ingestion left permanently un-ingestable documents.** `save_document()`
commits the `files` row, and only then does chunking run. Any failure after that point
left a document with zero chunks: invisible to search, and impossible to retry because
the dedupe check rejected every re-upload as a duplicate. Silent corruption, and it
predated the deployment work entirely — local embeddings just never failed.
Fixed by rolling back the document row when ingestion fails
([main_controller.py](app/controllers/main_controller.py)).

**2. `/health` opened a new Redis connection per request.** Managed Redis free tiers
cap *total* clients (~30). A health endpoint is the most-polled route on the service —
the platform polls it, a keep-warm pinger polls it again — so per-request connections
made the cheapest endpoint the one that exhausted the broker. Observed as
`max number of clients reached`, which takes down publishing and consuming at once.
Now one shared client; Celery's own pools are capped too
([worker.py](app/worker.py)). Idle footprint measured at **10 connections**, down from
~9 *per worker*.

**3. Oversized inputs failed whole documents.** Local sentence-transformers silently
truncate at 512 tokens; hosted APIs reject instead. `iso27001.pdf` contains a table
that sentence-splitting turns into one 21k-character pseudo-sentence. Notably,
tiktoken measured it at 5,039 tokens — under Jina's 8,192 limit — and it still failed,
because Jina's BERT-style tokenizer splits tables far more finely. **The lesson is
that you cannot count tokens in someone else's tokenizer**; the clip is therefore in
characters, the only unit every provider agrees on.

**4. Batches of 100 timed out.** The API accepts 100 items, but 100 chunks of ~1.5k
chars routinely ran past a 60s read timeout on a free tier — and a timeout burned the
entire retry budget. Batch size is now 32 with a 120s timeout, and read timeouts are
treated as retryable rather than fatal. More requests, but each finishes comfortably.

**5. A dead worker still reported healthy.** In single-container mode the supervisor
originally sent `kill -TERM 1`, which only works when the script is PID 1. When the
worker died, the API kept answering `200` on `/health` while nothing drained the queue:
uploads returned `202` and sat in `PENDING` forever. A half-dead service that passes
its own health check is worse than one that is plainly down, because nothing triggers a
restart. The supervisor now watches both PIDs and exits non-zero if either dies.

## 4.6 The ReAct agent was not doing retrieval

Worth its own section, because it is the flaw most likely to be caught in a demo and
the hardest to spot from the outside: the answers looked good.

**The loop exited on the first "Final Answer:" it saw.** `run()` matched the final-answer
pattern *before* the action pattern and returned immediately, so nothing ever obliged the
model to search. Two runs, both against a live corpus:

- Asked about *The Gift of the Magi*, the agent called `list_documents` (titles only),
  recognised the story, and answered from training data — **getting it wrong**: it said
  Della *received* the watch chain, when the chain is what she bought for Jim.
- Asked about ISO 27001 clause 6.1.3 — a document the model has never seen — it made
  **zero tool calls** and invented a citation, `【ISO 27001 Document – Clau…`.

A retrieval system answering from parametric knowledge is not a retrieval system. It is
also the failure a hostile reader finds in one question, by asking about something in
the corpus that contradicts what the model already believes.

Fixed with a **retrieval gate**: a Final Answer is rejected until at least one
content-returning tool (`search_all_documents`, `search_document`, `read_chunk`) has
succeeded. `list_documents` deliberately does not count — it returns titles, and a title
is exactly what lets a model bluff.

Three consequences, each needing its own fix:

1. **Iteration budget.** Forcing retrieval costs turns. At the old limit of 5 the agent
   did all the retrieval correctly and then timed out before writing the answer — the
   worst outcome, paying full cost for nothing. Now 8.
2. **Context growth.** Observations are fed back verbatim and the whole context is
   re-sent every iteration, so a multi-KB chunk of JSON is paid for once per remaining
   turn. That exhausted Groq's free 8k tokens/minute mid-run. Observations are now
   truncated to 1,200 chars *in the context only*; the full text stays in `history` for
   the UI. The LLM client also retries 429s, preferring the delay the API itself
   suggests over guessing.
3. **Model choice is not free.** `openai/gpt-oss-120b` emits *native* tool calls, which
   this hand-rolled text-format loop does not consume — Groq then rejects the request
   outright (`tool choice is none, but model called a tool`). `qwen/qwen3.8-27b` follows
   the text protocol. A hand-rolled ReAct loop constrains which models you can use, and
   that is a real cost of not using a framework.

**Citations are a separate problem from grounded prose.** Even once retrieval was
working and the summary of clause 6.1.3 was accurate, the agent cited 8 chunk IDs of
which **2 did not exist** — fabricated in the same `chk_xxxxxxxx` shape as the real ones
and indistinguishable by eye. A citation nobody can follow is worse than no citation,
because it reads as evidence. `validate_citations()` now checks every cited ID against
the database, replaces unverifiable ones, and returns them in the response as
`unverified_citations` rather than hiding the fact.

Verified after the fixes, through the HTTP API on production config:

| Query | Behaviour |
|---|---|
| "What must the Statement of Applicability contain?" | 3 tool calls, grounded answer, **0 unverified citations** |
| "What is the capital of Australia?" (not in corpus) | searched, found nothing, and **said so** instead of answering from memory |

## 4.7 Why a 0.38 MB PDF exhausted 512 MB

Render OOM-killed the service (exit 137) on a 0.38 MB README. File size turned out to
be almost unrelated to the memory that actually gets allocated, and measuring it was the
only way to find out where the memory went.

Where the worker's memory goes, measured by RSS at each stage:

| Stage | RSS |
|---|---|
| bare Python | 9 MB |
| + fastapi, celery, sqlalchemy, httpx, numpy | 66 MB |
| + pymupdf, pymupdf4llm | 143 MB |
| + MainController (engine, agent, chunker) | 194 MB |
| **+ parsing one 475 KB PDF** | **438 MB** |

The parse alone costs **~244 MB** — roughly 500× the file it is reading. And it is a
*fixed* cost, not one that scales with the document:

| Document | pages | peak RSS |
|---|---|---|
| iso27001.pdf | 26 | 364 MB |
| Peter-Pan.pdf | 95 | 357 MB |

A 95-page document peaks *lower* than a 26-page one. That rules out per-page
accumulation, and it is why **batching pages does not help** — tried it, peak stayed at
364 MB. The cost is `pymupdf4llm`'s layout-analysis working set, allocated on first use
regardless of what it is handed.

The comparison that isolates it, same documents, same extracted text:

| Parser | chars extracted | peak RSS |
|---|---|---|
| plain `fitz.get_text()` | 57,392 | **51 MB** |
| `pymupdf4llm.to_markdown()` | 60,186 | **364 MB** |

**7× the memory for 5% more text.** What that 7× buys is real — markdown headings, which
is what the hierarchical parser turns into `Context: Chapter 2 > Section 2.1` breadcrumbs.
Through a full ingestion the trade is visible:

| `PDF_PARSER` | chunks | with breadcrumb | peak RSS |
|---|---|---|---|
| `fast` | 49 | 0 | 211 MB |
| `structured` | 196 | 196 | 446 MB |

So the arithmetic on a 512 MB single-container tier is simply lost: 446 MB of worker
plus ~150 MB of API process is over the limit before anything else happens. No tuning
closes a 90 MB gap against a fixed library cost. `PDF_PARSER=fast` exists as the escape
hatch, and it is honestly a downgrade — a quarter of the chunks and none of the
structural context.

The intended fix was to stop trying to fit and move to a host with room — a Hugging Face
Space is 16 GB and would run `structured` with margin. That turned out to be a dead end:
Hugging Face moved Docker Spaces behind a paid plan. CPU Basic hardware is still free;
creating a Docker Space is not.

So the deployed instance runs `PDF_PARSER=fast` and the README says so. The honest
summary is that no free host was found with enough memory for the structured parser,
and the demo is degraded rather than absent. Oracle Cloud's always-free tier (24 GB ARM)
would run it, at the cost of owning TLS, a domain and the box; AWS free tier would too,
but expires after twelve months.

One portability bug fell out of this. Spaces run the container as a **non-root uid**,
while everything `COPY`d in is root-owned — and the app creates `data/uploads` and opens
a log file under `/app` at startup. That is a `PermissionError` before the first request.
The Dockerfile now creates and opens that directory at build time. Verified by running
the built image with `--user 1000:1000`: startup writes succeed and `/docs` serves.

---

## 5. Known weaknesses

Worth stating before someone else finds them.

- **Uploads through Postgres is a stopgap.** Object storage is the correct answer.
  Current design puts multi-MB writes on the same connection pool that serves queries,
  which is exactly the kind of contention Phase 2 load testing should expose.
- **Free-tier cold starts.** Render free spins down after ~15 minutes idle; the first
  request afterwards pays a cold start. Any latency number taken from a cold instance
  is meaningless — warm the service before benchmarking.
- **No retry policy on the ingestion task yet.** A worker killed mid-task loses the
  task; the blob row is deleted in `finally` even on failure, so a retry would find
  nothing. That is Phase 3 work and it is a genuine correctness gap, not a polish item.
- **The retrieval gate is a blunt instrument.** It requires *a* successful content
  lookup, not a *relevant* one: an agent that searches badly, gets nothing useful, and
  then answers from memory would still pass. The honest fix is grounding verification —
  checking the answer's claims against the retrieved text — which this does not do.
- **Citation validation only checks existence, not support.** A cited chunk ID that
  exists but does not actually support the sentence it is attached to passes silently.
- **Chunk IDs are `chk_` + 8 hex characters**, which is exactly why the model can
  fabricate plausible ones. Longer or structurally checkable IDs would make invented
  citations self-evident instead of needing a database round trip to catch.
- **Switching to hosted embeddings has not been re-evaluated for quality.** The
  hit@5 = 50.0% / MRR figures were measured with `bge-base-en-v1.5`. Same dimension is
  not the same model — the eval should be re-run against `gemini-embedding-001` before
  those numbers are quoted alongside a deployment that does not use that model.
- **`pool_pre_ping` costs a round trip per checkout.** Correct for Neon's idle
  disconnects, but it is a latency floor worth measuring in Phase 2.
