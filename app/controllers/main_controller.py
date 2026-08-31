import os
import time
import logging
from fastapi import HTTPException
from app.utils.pdf_extractor import parsePdf
from app.db.database_manager import DatabaseManager, RAGIngestionManager
from app.db.retrieval_manager import HierarchicalRAGRetriever
from app.engines.semantic_chunker import SemanticEngine
from app.core.config import settings
from app.engines.agent_engine import ReActAgent

# Dedicated ingestion audit logger writing to a configurable file path,
# so the log location is an environment concern instead of a hardcoded
# file in the repo root.
ingestion_logger = logging.getLogger("rag.ingestion")

def _configure_ingestion_logger():
    if not ingestion_logger.handlers:
        log_dir = os.path.dirname(settings.ingestion_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(settings.ingestion_log_path)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        ingestion_logger.addHandler(handler)
        ingestion_logger.setLevel(logging.INFO)

class MainController:
    """
    Central orchestrator for the RAG pipeline.
    Coordinates document parsing, database connection lifecycle, vector index builds,
    and semantic search execution.
    """
    def __init__(self):
        # Initialize database connection wrappers.
        self.db_manager = DatabaseManager()
        # Initialize inference engine.
        self.vector_engine = SemanticEngine()
        
        # Tie database operations and vector encoding together via ingestion controller.
        self.ingestion_manager = RAGIngestionManager(
            db_manager=self.db_manager,
            vector_engine=self.vector_engine
        )

        self.agent = ReActAgent(
            db_manager=self.db_manager,
            vector_engine=self.vector_engine
        )

    async def initialize_system(self):
        """Prepares database and tables. Creates vector extensions if needed."""
        await self.db_manager.create_tables()

    async def process_and_ingest_pdf(self, file_path: str) -> dict:
        """
        Executes end-to-end PDF ingestion pipeline:
        1. Parses PDF structure to markdown layout.
        2. Deduplicates document and saves metadata.
        3. Segments document and ingests vectors.
        """
        start_time = time.perf_counter()
        filename = os.path.basename(file_path)
        file_size_bytes = 0
        try:
            if os.path.exists(file_path):
                file_size_bytes = os.path.getsize(file_path)
        except Exception:
            pass

        _configure_ingestion_logger()

        # Tracks the document row this call created, so a later failure can undo
        # it. Stays None when save_document raises (e.g. a duplicate), which
        # matters: we must never delete a document a *previous* run ingested.
        created_doc_id = None

        try:
            # Step 1: Parse PDF to markdown representation.
            try:
                print(f"[Ingestion] Phase 1: Parsing PDF file '{filename}' (size: {file_size_bytes / (1024 * 1024):.2f} MB) using PyMuPDF...", flush=True)
                doc = parsePdf(file_path)
            except HTTPException as he:
                raise he
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"PDF Parsing Failed: {str(e)}")

            # Step 2: Save document metadata. Will throw 400 error on duplicate hashes.
            print("[Ingestion] Phase 2: PDF parsed successfully. Saving document metadata and checking for duplicates...", flush=True)
            db_doc = await self.db_manager.save_document(doc)
            created_doc_id = db_doc.id

            # Step 3: Run segment and ingestion transaction.
            print(f"[Ingestion] Phase 3: Document metadata saved (ID: {db_doc.id}). Starting segmentation and embedding ingestion...", flush=True)
            chunk_count = await self.ingestion_manager.ingest_document(
                raw_text=doc.extracted_text,
                file_id=db_doc.id,
                source_name=doc.title
            )

            elapsed_time = time.perf_counter() - start_time

            ingestion_logger.info(
                f"SUCCESS | File: {filename} | "
                f"Size: {file_size_bytes / (1024 * 1024):.2f} MB | "
                f"Duration: {elapsed_time:.2f}s | "
                f"Chunks: {chunk_count} | "
                f"Doc ID: {db_doc.id}"
            )

            return {
                "message": "Document successfully parsed and ingested.",
                "document_id": db_doc.id,
                "title": db_doc.title,
                "chunks_processed": chunk_count
            }
        except Exception as e:
            # save_document commits the metadata row before chunking runs, so a
            # failure after that point leaves a document with zero chunks:
            # invisible to search, and permanently un-ingestable because the
            # dedupe check rejects every retry as a duplicate. Undo it, so a
            # failed ingestion leaves no trace and the upload can be retried.
            if created_doc_id is not None:
                try:
                    await self.db_manager.delete_document(created_doc_id)
                    ingestion_logger.info(f"ROLLBACK | Removed partial document {created_doc_id}")
                except Exception as cleanup_error:
                    # Surfacing the original failure matters more than this one.
                    ingestion_logger.error(
                        f"ROLLBACK FAILED | doc {created_doc_id} left with no chunks | {cleanup_error}"
                    )

            elapsed_time = time.perf_counter() - start_time
            ingestion_logger.error(
                f"FAILURE | File: {filename} | "
                f"Size: {file_size_bytes / (1024 * 1024):.2f} MB | "
                f"Duration: {elapsed_time:.2f}s | "
                f"Error: {str(e)}"
            )
            # Bare raise preserves the original traceback instead of
            # re-raising from this frame.
            raise

    async def fetch_document(self, doc_id: str):
        """Fetches document metadata by its hash ID."""
        return await self.db_manager.fetch_document(doc_id)

    async def query_document(self, query_text: str, file_id: str, top_k: int = 3):
        """
        Performs vector similarity queries on document chunks.
        Spawns retriever with a scoped session connection block.
        """
        async with self.db_manager.SessionLocal() as session:
            retriever = HierarchicalRAGRetriever(db_session=session, vector_engine=self.vector_engine)
            results = await retriever.retrieve_relevant_chunks(query_text, file_id, top_k)
            return results

    async def ask_agent(self, user_query: str) -> dict:
        """
        Routes queries through the ReAct agent.
        Returns the final answer along with intermediate reasoning steps.
        """
        answer, steps = await self.agent.run(user_query)

        # The agent cites chunk IDs, and sometimes invents ones shaped exactly
        # like the real ones. Verify each against the database before the answer
        # leaves the system, and report any that could not be verified rather
        # than quietly dropping them.
        answer, unverified = await self.agent.validate_citations(answer)

        return {
            "query": user_query,
            "answer": answer,
            "steps": steps,
            "unverified_citations": unverified
        }