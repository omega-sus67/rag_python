from fastapi import HTTPException
from app.utils.pdf_extractor import parsePdf
from app.db.database_manager import DatabaseManager, RAGIngestionManager
from app.db.retrieval_manager import HierarchicalRAGRetriever
from app.engines.semantic_chunker import SemanticEngine

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
        # Step 1: Parse PDF to markdown representation.
        try:
            doc = parsePdf(file_path)
        except HTTPException as he:
            # Re-raise explicit HTTP exceptions from utility.
            raise he
        except Exception as e:
            # Wrap unexpected library errors into standard Internal Server Error.
            raise HTTPException(status_code=500, detail=f"PDF Parsing Failed: {str(e)}")

        # Step 2: Save document metadata. Will throw 400 error on duplicate hashes.
        db_doc = await self.db_manager.save_document(doc)
        
        # Step 3: Run segment and ingestion transaction.
        try:
            chunk_count = await self.ingestion_manager.ingest_document(
                raw_text=doc.extracted_text,
                file_id=db_doc.id,
                source_name=doc.title
            )
            return {
                "message": "Document successfully parsed and ingested.",
                "document_id": db_doc.id,
                "title": db_doc.title,
                "chunks_processed": chunk_count
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ingestion Failed: {str(e)}")

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
