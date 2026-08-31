---
title: Agentic RAG Pipeline
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
short_description: Async RAG service with a ReAct agent that must retrieve before it answers
---

# Agentic RAG Pipeline

A Retrieval-Augmented Generation service built from scratch in Python — no LangChain.
Hierarchical PDF parsing, sliding-window semantic chunking, async ingestion through
Celery + Redis into PostgreSQL/pgvector, and a ReAct agent that is structurally
prevented from answering without retrieving.

**Interactive API:** append `/docs` to this Space's URL.

- `POST /upload` — accepts a PDF, returns `202` plus a task id
- `GET  /task/status/{id}` — ingestion progress
- `POST /agent/query` — ask a question; the agent must search before answering
- `GET  /health` — returns 503 when Postgres or Redis is unreachable

**Source, full engineering write-up, and the honest limitations:**
<https://github.com/omega-sus67/rag_python>

The README there documents what broke getting this deployed and what each fix cost —
including the measurement showing `pymupdf4llm` needs ~364 MB of working set
regardless of document size, which is why this runs here rather than on a 512 MB tier.
