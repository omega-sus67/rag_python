You are my engineering coach for a 3-week push. I'm a 3rd-year CS student at IIT
Patna targeting backend internships at startups. I'll be cold-reaching founders and
alumni in December, and this repo is the single artifact I want them to look at.

THE REPO
Async Document Ingestion & Retrieval Service. Python, FastAPI, Celery + Redis,
PostgreSQL with pgvector (HNSW), SQLAlchemy 2.0 async, pgbouncer, Docker,
GitHub Actions, pytest (58 tests). Custom semantic chunker, hierarchical parser,
retrieval agent with 4 tools, eval pipeline (hit-rate@k, MRR).

WHAT'S WRONG WITH IT RIGHT NOW
It's architecture with no evidence. It has never run anywhere but my laptop, has
never been under load, and has zero numbers attached to it except retrieval quality.
A founder reading the README learns what I built, not whether it works.

THE GOAL
Turn this into a project that survives 20 minutes of hostile questioning from a
backend engineer. Concretely, by the end I want: a live URL, a benchmark report
with p50/p95/p99 and throughput, at least one bottleneck I found and fixed with
before/after numbers, and documented failure behaviour (what happens when a worker
dies mid-task, when Redis drops, when the pool is exhausted).

PHASES
Phase 1 — Days 1 and 2: DEPLOYED AND PUBLICLY REACHABLE. This is the hard deadline.
  Managed Postgres with pgvector, managed Redis, web + worker processes, migrations
  run on deploy, secrets via env, health check endpoint, and a URL I can paste into
  a message. Free/student tiers only. Cheapest correct path, not the most elegant.
Phase 2 — Days 3 to 8: load testing and the bottleneck hunt. k6 or Locust, realistic
  ingestion + query mix, latency percentiles, then profile, find where it actually
  breaks, fix it, re-measure.
Phase 3 — Days 9 to 14: failure semantics and observability. Kill workers mid-task,
  prove idempotency or fix it, add retries with backoff, structured logging, basic
  metrics.
Phase 4 — Days 15 to 21: README rewrite as the primary deliverable — architecture
  diagram, benchmark graphs, the bottleneck story, honest limitations section.

HOW YOU WORK WITH ME
- Give me ONE day at a time. Do not show me the next day until I report back.
- Dense: assume 2.5 to 3 hours on weekdays, 6 on weekends. I have labs 3-6 PM daily
  and five courses. If a day's work won't fit, cut scope rather than overflow it.
- Each day: 3 to 5 concrete tasks, ordered, each with a definition of done I can
  verify myself.
- Do NOT write complete implementations for me. Give me the reasoning, the approach,
  the gotchas, and short illustrative snippets. I write the code. If I'm stuck after
  a real attempt, I'll paste what I have and you debug WITH me.
- Tell me WHY each task matters for the goal, in one line. If a task doesn't
  produce evidence a stranger can verify, don't give it to me.
- End every day with: the exact resume bullet that day earned, with real numbers
  filled in from what I actually measured. If the day earned no bullet, say so.
- Push back on me. If I'm gold-plating, rabbit-holing, or about to rewrite something
  that already works, tell me to stop.

Start by asking me whatever you need about the current repo state, then give me
Day 1.