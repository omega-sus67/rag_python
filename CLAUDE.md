# CLAUDE.md — engineering log & coaching state

This file is the memory of a 3-week push (brief: [docs/task.md](docs/task.md)) to turn this
repo from *architecture with no evidence* into a project that survives 20 minutes of hostile
questioning from a backend engineer.

**Read this first in any new session.** Current day and next action are at the bottom.

---

## Working agreement

- One day at a time. The next day is not shown until the previous one is reported back.
- Budget: ~2.5–3 h weekdays (labs 3–6 PM daily, five courses), ~6 h weekends. If a day
  doesn't fit the budget, scope gets cut, not overflowed.
- Each day: 3–5 ordered tasks, each with a definition of done the user verifies alone.
- **The user writes all the code.** Claude gives reasoning, approach, gotchas, and short
  illustrative snippets — never complete implementations. If the user is stuck after a real
  attempt, they paste what they have and we debug together.
- Every task must produce evidence a stranger can verify. If it doesn't, it doesn't get given.
- Every day ends with the exact resume bullet it earned, real numbers filled in. If the day
  earned no bullet, that is said explicitly.
- Push back on gold-plating, rabbit-holing, and rewriting things that already work.

---

## The goal (definition of done for the whole 3 weeks)

1. A live URL that can be pasted into a cold message to a founder.
2. A benchmark report: p50 / p95 / p99 latency and throughput, under a realistic mixed load.
3. At least one bottleneck found, fixed, and re-measured — with before/after numbers.
4. Documented failure behaviour: worker dies mid-task, Redis drops, connection pool exhausts.

---

## Phase roadmap

| Phase | Days | Outcome |
|---|---|---|
| 1 | 1–2 | **Deployed and publicly reachable.** Hard deadline. Managed Postgres+pgvector, managed Redis, web + worker, schema bootstrap on deploy, secrets via env, health check, a URL. |
| 2 | 3–8 | Load testing and the bottleneck hunt. k6 or Locust, realistic ingest+query mix, percentiles, profile, fix, re-measure. |
| 3 | 9–14 | Failure semantics and observability. Kill workers mid-task, prove or fix idempotency, retries with backoff, structured logging, basic metrics. |
| 4 | 15–21 | README rewrite as the primary deliverable. Architecture diagram, benchmark graphs, the bottleneck story, honest limitations. |

---

## Repo facts (so a cold session doesn't re-derive them)

**Stack:** Python 3.12, FastAPI, Celery + Redis, PostgreSQL + pgvector (HNSW),
SQLAlchemy 2.0 async, asyncpg, pgbouncer (dev only), Docker Compose, GitHub Actions, pytest.
No LangChain — parser, chunker, and ReAct agent are hand-rolled.

**Entry points:**
- [app/main.py](app/main.py) — FastAPI app. `/upload` (202 + task_id), `/task/status/{id}`,
  `/docFetch/{id}`, `/agent/query`.
- [app/worker.py](app/worker.py) — Celery app + `process_pdf_task`, runs the async controller
  inside `asyncio.run()` per task.
- [app/controllers/main_controller.py](app/controllers/main_controller.py) — orchestrator:
  parse → dedupe/save → chunk → embed → store.
- [app/db/database_manager.py](app/db/database_manager.py) — models (`files`, `doc_chunks`),
  HNSW index definition, schema bootstrap.
- [app/core/config.py](app/core/config.py) — pydantic-settings; all tunables.
- [app/eval/retrieval_eval.py](app/eval/retrieval_eval.py) — `python cli.py evaluate-retrieval`.

**Measured numbers that already exist** (retrieval quality only — this is the entire
evidence base right now, which is the problem):
- hit@5 = 50.0% after the merge-undersized-chunks fix (was 38.9% before).
- Semantic chunker MRR 0.312 vs fixed-400-token baseline MRR 0.347 — i.e. parity/slight
  loss, honestly reported. Corpus: 18 questions across 4 documents.
- 58 tests, mocked, green in CI (ruff + pytest).
- Embedding batch size 64 — 256 OOMs a 6 GB GPU on 400-token chunks.

**Zero numbers exist for:** latency, throughput, concurrency, memory, failure recovery.
That gap is the entire point of the next three weeks.

---

## Deployment blockers found in the Day-0 audit

These were found by reading the code, not by trying to deploy. All five must be cleared
before the URL exists.

1. **No single-URL DB config.** [config.py:44-50](app/core/config.py#L44-L50) assembles the
   connection URL from user/password/host/port/name. Every managed Postgres hands out one
   URL, with `?sslmode=require` on it — and **asyncpg rejects `sslmode` outright** (that's
   libpq syntax; asyncpg wants `ssl=` via connect_args). Needs a `database_url` override
   that wins when set.
2. **`create_tables()` issues `CREATE DATABASE`.**
   [database_manager.py:77-104](app/db/database_manager.py#L77-L104) opens a *second* engine
   against the `postgres` maintenance database first. On Neon/Supabase this fails on
   permissions or endpoint routing. Deploy-time bootstrap must reduce to: connect to the DB
   that already exists → `CREATE EXTENSION IF NOT EXISTS vector` → `create_all`.
3. **Web and worker share a filesystem.** `/upload` streams the PDF to `settings.upload_dir`
   ([main.py:53-65](app/main.py#L53-L65)) and passes the *path* through Redis
   ([main.py:68](app/main.py#L68)). Compose fakes this with a bind mount. On a PaaS, web and
   worker are separate containers with separate disks — the worker gets a path to a file that
   does not exist. **This is the real architectural blocker.** Note the Redis free tier is 30 MB
   total, so PDF bytes cannot simply ride the queue either.
4. **Memory.** `torch` + `sentence-transformers` + `bge-base-en-v1.5` is ~1–1.5 GB RSS in the
   worker, and `torch==2.12.0` in [requirements.txt](requirements.txt) pulls the CUDA wheel
   (multi-GB image). A 512 MB free tier cannot run this.
5. **No real health endpoint, no `$PORT`, no TLS Redis.** `/` returns a static message and
   checks nothing. PaaS platforms inject `$PORT`. If the Redis URL is
   `rediss://`, Celery needs explicit `broker_use_ssl` / `redis_backend_use_ssl`.

`pgbouncer` is dropped in production (Neon ships its own pooler); it stays in compose for
local dev.

---

## Dev-machine gotcha — `/usr/bin/psql` is a wrapper that hangs on connection URIs

Cost ~1 hour on Day 1 and produced an entirely wrong diagnosis before being nailed down.
**Neon connectivity is fine.** Confirmed working from the dev laptop: PostgreSQL 17.11,
pgvector 0.8.0, over both asyncpg and the real psql binary.

On Ubuntu, `/usr/bin/psql` is not psql — it is `/usr/share/postgresql-common/pg_wrapper`,
a Perl script that picks a version and cluster from the *local* installation before
dispatching. This machine has a local PG16 cluster on port **5433**. Given a
`postgresql://…` URI the wrapper mishandles it and hangs forever rather than erroring.

- **Use `/usr/lib/postgresql/16/bin/psql` for anything involving a remote URI.** Worth an
  alias in `~/.bashrc`.
- Symptom to recognise: TCP connects, `openssl s_client -starttls postgres` completes with
  a valid cert, the HTTP SQL endpoint answers in ~1 s, but psql hangs and
  `PGCONNECT_TIMEOUT` never fires.
- `role "<unix-username>" does not exist` from psql is the *other* wrapper symptom — it
  means the connection string was empty and it fell back to the local cluster.

**Debugging lesson, more valuable than the fix:** ~45 minutes went into theories about
campus firewalls, IPv6 path MTU, SNI routing and GSSAPI, all because the DoD was written
around `psql` — a tool the application never uses. Testing **asyncpg**, the driver the app
actually depends on, answered the question in one shot. When a CLI tool disagrees with your
application stack, test the application stack first.

---

## Decisions (locked — do not relitigate)

| Decision | Choice | Why |
|---|---|---|
| Hosting | Managed PaaS, free tiers: **Neon** (Postgres+pgvector) + **Redis Cloud Essentials** (Redis) + Render/Fly for web & worker | Matches the brief. Forces blockers 3 and 4 to be fixed properly instead of hidden behind a single VM. A single droplet would ship faster but invites "so it's one box?" |
| Embeddings | **Pluggable provider behind an env var** — local `sentence-transformers` for dev/eval, hosted API in prod | Gemini `text-embedding-004` is 768-dim, matching the existing `Vector(768)` column, so no migration and eval numbers stay comparable. Also puts a real network call inside the ingestion path, which is a more interesting thing to load-test. Lands Day 2–3. |
| This file | Root `CLAUDE.md`, committed | Auto-loads every session, which is what makes the day-by-day sequencing survive. Reads as an engineering log to anyone else. |
| Upload transport (blocker 3) | **Open** — Postgres `bytea` vs object storage (R2/Supabase Storage) | Decided Day 2. Neon free tier is 512 MB storage, which argues against bytea for a public demo. |

---

## Day log

Append-only. Records what actually happened, never what is planned.

### Day 0 — 2026-08-23 — audit & setup
- Read the full repo, produced the five blockers above.
- Locked hosting, embedding, and state-file decisions.
- Created this file.
- **Resume bullet earned: none.** Reading your own code is not an accomplishment.

### Day 1 — given 2026-08-23, in progress
Tasks given (details in chat):
1. ✅ Provision managed Postgres (with `vector` extension) and managed Redis.
2. ⬜ `DATABASE_URL` / `REDIS_URL` overrides in `config.py`, with the asyncpg fix.
3. ⬜ Split schema bootstrap from database creation — idempotent `init-db` command.
4. ⬜ Real `/health` that checks Postgres and Redis and returns 503 when either is down.

**Task 1 done.** Neon `ap-southeast-1`, PostgreSQL **17.11**, `pgvector` **0.8.0** —
verified from the dev laptop over asyncpg *and* psql. Redis Cloud Essentials
`ap-southeast-1`, `PONG`. Local compose DB pinned from the abandoned
`ankane/pgvector:latest` (which was **PG 15.4**) to `pgvector/pgvector:pg17`, so local and
production now match on major version.

Redis provider changed from Upstash to **Redis Cloud Essentials** at provisioning time:
Upstash meters by command count, and a Celery worker's `BRPOP` loop plus result-backend
polling would exhaust a monthly free quota within days — before Phase 2 load testing even
starts. Redis Cloud's free tier caps on memory (30 MB) instead, which is the right shape
for a queue with small payloads.

Two gotchas cost real time and are worth remembering:
- Neon's URL contains `&`. Unquoted in `.env`, `set -a; source .env` parses the `&` as a
  background-job operator, the assignment runs in a subshell, and the variable silently
  stays **unset**. psql then falls back to a local socket — the tell is
  `role "<your-unix-username>" does not exist`, which always means "empty connection
  string", never "remote server rejected me". All URL values in `.env` are now single-quoted.
- Neon issues `?sslmode=require&channel_binding=require`. **Both** are libpq-only params
  that asyncpg rejects. Task 2 must drop the whole query string, not special-case `sslmode`.

Reported back: Tasks 2–4 _pending_
Resume bullet: expected **none** — the URL is a Day 2 artifact.

---

## Current state

- **Phase:** 1 (deploy). **Day:** 1, in progress.
- **Next action:** user reports back on Day 1's four DoDs; then Day 2 (deploy + blocker 3
  upload-transport decision + pluggable embedding provider) gets issued.
- **Hard deadline:** publicly reachable URL by end of Day 2.
