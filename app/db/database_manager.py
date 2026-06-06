import hashlib
import numpy as np
from typing import List, Optional
from fastapi import HTTPException
from pydantic import BaseModel

from sqlalchemy import Column, Integer, String, text, select, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from pgvector.sqlalchemy import Vector

# Import our custom classes from earlier phases
from app.utils.pdf_operator import Document
from app.engines.engine_operator import HierarchicalSemanticEngine, RenderedChunk
from app.engines.semantic_engine import SemanticEngine

ASYNC_DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5433/tutorial_db"

Base = declarative_base()

class dbFile(Base):
    __tablename__ = "files"
    
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    extracted_text = Column(String, nullable=False)


class dbChunk(Base):
    __tablename__ = "doc_chunks"

    id = Column(String, primary_key=True, index=True)
    file_id = Column(String, ForeignKey("files.id"))
    chunk_index = Column(Integer, nullable=False)
    text_data = Column(String, nullable=False)
    embeddings = Column(Vector(384))

    @classmethod
    def from_rendered_chunk(cls, rendered_chunk: RenderedChunk, file_id: str, index: int, vector: List[float]):
        """Factory method to translate a RenderedChunk into a database record."""
        return cls(
            id=rendered_chunk.chunk_id,
            file_id=file_id,
            chunk_index=index,
            text_data=rendered_chunk.text,
            embeddings=vector
        )


class DatabaseManager:
    """Encapsulates database connection setup and core document operations."""
    
    def __init__(self, database_url: str = ASYNC_DATABASE_URL):
        self.engine = create_async_engine(database_url, echo=True)
        self.SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    async def create_tables(self):
        """Initializes the database and creates all tables."""
        temp_url = "postgresql+asyncpg://postgres:password@localhost:5433/postgres"
        async_temp_engine = create_async_engine(temp_url, isolation_level="AUTOCOMMIT")
        
        async with async_temp_engine.connect() as conn:
            res = await conn.execute(text("SELECT 1 FROM pg_database WHERE datname='tutorial_db'"))
            if not res.scalar():
                print("Database 'tutorial_db' does not exist. Creating it...")
                await conn.execute(text("CREATE DATABASE tutorial_db"))
            else:
                print("Database 'tutorial_db' already exists.")
                
        await async_temp_engine.dispose()

        # Connect to 'tutorial_db' and create tables
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)

    async def save_document(self, doc: Document) -> dbFile:
        """Saves a Document into the files table, ensuring uniqueness via text hash."""
        unique_id = hashlib.sha256(doc.extracted_text.encode('utf-8')).hexdigest()

        async with self.SessionLocal() as session:
            chk = select(dbFile).where(dbFile.id == unique_id)
            res = await session.execute(chk)

            if res.scalar():
                raise HTTPException(status_code=400, detail="File already exists")

            db_doc = dbFile(
                id=unique_id,
                title=doc.title,
                extracted_text=doc.extracted_text
            )
            session.add(db_doc)
            await session.commit()
            await session.refresh(db_doc)
            return db_doc

    async def fetch_document(self, file_id: str) -> dbFile:
        """Fetches a saved Document by its ID."""
        async with self.SessionLocal() as session:
            chk = select(dbFile).where(dbFile.id == file_id)
            res = await session.execute(chk)
            doc = res.scalar()
            if not doc:
                raise HTTPException(status_code=404, detail="File not found")
            return doc


class RAGIngestionManager:
    """Orchestrates text chunking, bulk vector generation, and database persistence."""
    
    def __init__(self, db_manager: DatabaseManager, vector_engine: SemanticEngine):
        self.db_manager = db_manager
        self.vector_engine = vector_engine
        # Initialize our unified chunking engine using the shared vector engine
        self.chunking_engine = HierarchicalSemanticEngine(
            vector_engine=self.vector_engine,
            window_size=3,
            threshold_factor=0.8
        )

    async def ingest_document(self, raw_text: str, file_id: str, source_name: str) -> int:
        """
        Processes a raw document string, generates optimized database records, 
        and commits them to PostgreSQL. Returns total count of chunks ingested.
        """
        # Step 1: Run the document tree parsing and semantic boundary split logic
        rendered_chunks = self.chunking_engine.process_document(raw_text, source_name=source_name)
        
        if not rendered_chunks:
            return 0

        # Step 2: Extract the raw text strings from our optimized chunks
        final_texts_to_embed = [chunk.text for chunk in rendered_chunks]

        # Step 3: Compute the fresh, clean, final ingestion embeddings in bulk batches
        # Assuming vector_engine has get_embeddings based on semantic_engine.py
        ingestion_embeddings = self.vector_engine.get_embeddings(final_texts_to_embed)

        # Step 4: Map the DTO chunks and their vectors to our dbChunk models
        async with self.db_manager.SessionLocal() as session:
            for index, rendered_chunk in enumerate(rendered_chunks):
                # Extract the correct vector row from our bulk matrix and convert to a plain float list
                vector_row = ingestion_embeddings[index].tolist()

                # Translate fields cleanly using the factory method
                db_record = dbChunk.from_rendered_chunk(
                    rendered_chunk=rendered_chunk,
                    file_id=file_id,
                    index=index,
                    vector=vector_row
                )
                
                # Stage the record inside the database transaction block
                session.add(db_record)

            # Step 5: Flush all staged records safely to the disk in a single transaction commit
            try:
                await session.commit()
                return len(rendered_chunks)
            except Exception as e:
                await session.rollback()  # Rollback transaction if writing fails to ensure data integrity
                raise e
