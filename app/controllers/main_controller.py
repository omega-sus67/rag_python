from fastapi import HTTPException
from app.utils.pdf_operator import parsePdf
from app.db.database_manager import DatabaseManager, RAGIngestionManager
from app.engines.semantic_engine import SemanticEngine

class MainController:
    """
    Central orchestrator for the RAG pipeline.
    Coordinates document parsing, database operations, and vector ingestion
    in a unified, class-based architecture.
    """
    
    def __init__(self):
        # Initialize our core components
        self.db_manager = DatabaseManager()
        self.vector_engine = SemanticEngine()
        
        # Ingestion manager ties the DB and Vector Engine together
        self.ingestion_manager = RAGIngestionManager(
            db_manager=self.db_manager,
            vector_engine=self.vector_engine
        )

    async def initialize_system(self):
        """Initializes system resources, such as creating database tables."""
        await self.db_manager.create_tables()

    async def process_and_ingest_pdf(self, file_path: str) -> dict:
        """
        End-to-end pipeline for a single PDF:
        1. Parse PDF into raw text.
        2. Save document metadata to the database.
        3. Chunk, embed, and ingest text vectors into the database.
        """
        # Step 1: Parse the PDF
        try:
            doc = parsePdf(file_path)
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF Parsing Failed: {str(e)}")

        # Step 2: Save metadata to get a unique document ID
        # Note: DatabaseManager.save_document already throws a 400 if it's a duplicate
        db_doc = await self.db_manager.save_document(doc)
        
        # Step 3: Run the chunking and embedding ingestion pipeline
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
        """Retrieves document metadata by ID."""
        return await self.db_manager.fetch_document(doc_id)
