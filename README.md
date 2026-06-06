# RAG Pipeline — CLI Instructions

A high-performance, hierarchical Retrieval-Augmented Generation (RAG) pipeline built with Python, PostgreSQL (`pgvector`), and SentenceTransformers. This system is specifically tuned for processing complex literary texts, dialogue, and novels by using advanced structural context accumulation.

## CLI Usage

The pipeline is entirely managed through a beautiful, Hermes-style Command Line Interface (`cli.py`).

### 1. Ingesting Documents

To parse, chunk, embed, and save a PDF document into the database, use the `ingest` command. This will output a unique **Document ID** upon completion.

```bash
python3 cli.py ingest /path/to/your/document.pdf
```
**Example:**
```bash
python3 cli.py ingest trialData/Peter-Pan.pdf
```

### 2. Semantic Querying

To search a previously ingested document for relevant context blocks, use the `query` command. You must provide the exact Document ID returned during ingestion.

Because the system uses an asymmetric Q&A embedding model (`multi-qa-MiniLM-L6-cos-v1`), queries should ideally be phrased as **natural language questions** (e.g., "Who is Captain Hook?" rather than just "Captain Hook").

```bash
python3 cli.py query <document_id> "<your_question>"
```

**Options:**
- `--top <number>`: Specifies how many context chunks to return (default is 3).

**Example:**
```bash
python3 cli.py query d1b339c0f6f5dff9721aa57c8cd70952701baf49ecd4713dc691070d0a76300c "Why does Peter Pan refuse to grow up?" --top 5
```

## Architecture Notes
- **Context Accumulation**: The markdown parser automatically accumulates paragraphs across dialogue line-breaks, ensuring the LLM receives thick, descriptive narrative blocks instead of fragmented single sentences.
- **Hierarchical Chunking**: Context paths (like Headings and Titles) are automatically prepended to semantic chunks to ensure positional awareness.
- **Vector Space**: Embeddings are 384-dimensional dense vectors stored natively in Postgres via `pgvector` and queried using Cosine Distance.
