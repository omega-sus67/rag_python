import hashlib
from typing import List, Optional
from fastapi import HTTPException

from sqlalchemy import Column, Integer, String, text, select, ForeignKey, Index
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from pgvector.sqlalchemy import Vector

# Import custom domain structures and engines
from app.utils.pdf_extractor import Document
from app.engines.hierarchical_chunker import HierarchicalSemanticEngine, RenderedChunk
from app.engines.semantic_chunker import SemanticEngine
from app.core.config import settings

Base = declarative_base()

class dbFile(Base):
    """
    SQLAlchemy model representing the parent document file record.
    Stores the full extracted text content alongside document metadata.
    """
    __tablename__ = "files"
    
    # Unique ID is generated as the SHA-256 hash of the extracted text to prevent duplicate uploads.
    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    extracted_text = Column(String, nullable=False)


class dbChunk(Base):
    """
    SQLAlchemy model representing an individual semantic text chunk.
    Linked to its parent dbFile. Stores content and its vector embeddings coordinate.
    """
    __tablename__ = "doc_chunks"

    id = Column(String, primary_key=True, index=True)
    file_id = Column(String, ForeignKey("files.id"))
    chunk_index = Column(Integer, nullable=False)
    text_data = Column(String, nullable=False)
    # The Vector column stores multi-dimensional floating-point embeddings for vector queries.
    embeddings = Column(Vector(settings.embedding_dimension))

    @classmethod
    def from_rendered_chunk(cls, rendered_chunk: RenderedChunk, file_id: str, index: int, vector: List[float]):
        """
        Factory mapping method to convert a RenderedChunk DTO into a database record.
        Encapsulates translation rules in a clean, maintainable method.
        """
        return cls(
            id=rendered_chunk.chunk_id,
            file_id=file_id,
            chunk_index=index,
            text_data=rendered_chunk.text,
            embeddings=vector
        )
Index(
        "ix_doc_chunks_embeddings_hnsw",
        dbChunk.embeddings,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 200},
        postgresql_ops={"embeddings": "vector_cosine_ops"}
    )


class DatabaseManager:
    """
    Manages database connection lifecycle, pool setup, tables initialization,
    and core CRUD queries for documents.
    """
    def __init__(self, database_url: str = settings.async_database_url):
        # Create asynchronous SQLAlchemy engine using the configured postgres credentials.
        self.engine = create_async_engine(database_url, echo=False)
        self.SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    async def create_tables(self):
        """
        Initializes postgres schemas:
        1. Checks database existence using a temporary autocommit connection.
        2. Creates database and installs the pgvector extension if missing.
        3. Generates tables based on defined SQLAlchemy models.
        """
        temp_url = settings.async_postgres_url
        # AUTOCOMMIT isolation level is mandatory for executing 'CREATE DATABASE' command.
        async_temp_engine = create_async_engine(temp_url, isolation_level="AUTOCOMMIT")
        
        async with async_temp_engine.connect() as conn:
            # Verify if the target database name exists in system tables.
            res = await conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{settings.db_name}'"))
            if not res.scalar():
                print(f"Database '{settings.db_name}' does not exist. Creating it...")
                await conn.execute(text(f"CREATE DATABASE {settings.db_name}"))
            else:
                print(f"Database '{settings.db_name}' already exists.")
                
        await async_temp_engine.dispose()

        # Connect to newly created/verified database and create tables.
        async with self.engine.begin() as conn:
            # Enable the vector indexing support in Postgres.
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            # Generate all tables described by Base class.
            await conn.run_sync(Base.metadata.create_all)

    async def save_document(self, doc: Document) -> dbFile:
        """
        Inserves a Document metadata row.
        Generates a SHA-256 hash to ensure that files with identical text content are rejected.
        """
        unique_id = hashlib.sha256(doc.extracted_text.encode('utf-8')).hexdigest()

        async with self.SessionLocal() as session:
            # Query the database to check if this SHA-256 hash already exists.
            chk = select(dbFile).where(dbFile.id == unique_id)
            res = await session.execute(chk)

            # If scalar() yields a record, it's a duplicate upload. Raise HTTP 400.
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

    async def delete_document(self, file_id: str) -> bool:
        """
        Deletes a document and all its associated semantic chunks.
        Used to reset database state for clean-slate ingestion benchmarks.
        """
        async with self.SessionLocal() as session:
            # 1. Delete matching rows in doc_chunks
            await session.execute(text("DELETE FROM doc_chunks WHERE file_id = :file_id"), {"file_id": file_id})
            # 2. Delete the parent document row in files
            await session.execute(text("DELETE FROM files WHERE id = :file_id"), {"file_id": file_id})
            await session.commit()
            return True

    async def fetch_document(self, file_id: str) -> dbFile:
        """
        Fetches document metadata by its ID.
        Raises HTTP 404 if the document cannot be found.
        """
        async with self.SessionLocal() as session:
            chk = select(dbFile).where(dbFile.id == file_id)
            res = await session.execute(chk)
            doc = res.scalar()
            if not doc:
                raise HTTPException(status_code=404, detail="File not found")
            return doc
    
    async def list_all_documents(self) -> List[dict]:
        """
        Retrieves a list of all ingested files.
        This serves as a discovery tool for the Agent to know what content is available.
        """
        async with self.SessionLocal() as session:
            statement = select(dbFile.id, dbFile.title)
            results = await session.execute(statement)
            return [{"id": row[0], "title": row[1]} for row in results.all()]

    async def fetch_chunk_by_id(self, chunk_id: str) -> Optional[dict]:
        """
        Retrieves a specific chunk's raw content by its unique ID.
        Useful when the agent needs to inspect a particular segment in detail.
        """
        async with self.SessionLocal() as session:
            statement = select(dbChunk).where(dbChunk.id == chunk_id)
            result = await session.execute(statement)
            chunk = result.scalar()
            if chunk:
                return {
                    "id": chunk.id,
                    "file_id": chunk.file_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text_data
                }
            return None

    async def resolve_file_id(self, identifier: str) -> Optional[str]:
        """
        Resolves a document identifier.
        1. Checks if the identifier is an exact match for a document's hash ID.
        2. If not, checks if it matches a document's title (case-insensitive substring).
        Returns the resolved SHA-256 hash ID, or None if no match is found.
        """
        async with self.SessionLocal() as session:
            # 1. Check if it's a valid ID
            statement = select(dbFile.id).where(dbFile.id == identifier)
            result = await session.execute(statement)
            resolved = result.scalar()
            if resolved:
                return resolved
                
            # 2. Check if it matches a title
            statement = select(dbFile.id).where(dbFile.title.ilike(f"%{identifier}%"))
            result = await session.execute(statement)
            resolved = result.scalar()
            return resolved


class RAGIngestionManager:
    """
    Coordinates markdown parsing, chunking, and bulk vector embedding generation,
    before storing the resulting records in a single database transaction.
    """
    def __init__(self, db_manager: DatabaseManager, vector_engine: SemanticEngine):
        self.db_manager = db_manager
        self.vector_engine = vector_engine
        # Initialize Hierarchical Chunker with configured window settings.
        self.chunking_engine = HierarchicalSemanticEngine(
            vector_engine=self.vector_engine,
            window_size=settings.window_size,
            threshold_factor=settings.threshold_factor
        )

    async def ingest_document(self, raw_text: str, file_id: str, source_name: str) -> int:
        """
        Ingests a document:
        1. Breaks raw markdown into structured, context-enriched text blocks and computes their embeddings.
        2. Inserts chunks inside a transaction.
        3. Rolls back database changes on failure.
        """
        # Step 1: Run hierarchical parsing, semantic segmentation, and embedding computation.
        rendered_chunks = await self.chunking_engine.process_document(raw_text, source_name=source_name)
        
        # If no chunks were generated, exit early.
        if not rendered_chunks:
            return 0

        # Step 2: Map DTO chunks to database records and insert them.
        print(f"[Ingestion] Preparing {len(rendered_chunks)} chunks for database insertion...", flush=True)
        async with self.db_manager.SessionLocal() as session:
            for index, rendered_chunk in enumerate(rendered_chunks):
                # Retrieve pre-computed chunk vector coordinates
                vector_row = rendered_chunk.embeddings

                # Map metadata fields via factory.
                db_record = dbChunk.from_rendered_chunk(
                    rendered_chunk=rendered_chunk,
                    file_id=file_id,
                    index=index,
                    vector=vector_row
                )
                
                # Add record to transaction session queue.
                session.add(db_record)

            # Step 3: Flush and commit all inserts to PostgreSQL.
            try:
                print(f"[Ingestion] Committing transaction for {len(rendered_chunks)} chunks to PostgreSQL...", flush=True)
                await session.commit()
                print(f"[Ingestion] Database transaction committed successfully for '{source_name}'.", flush=True)
                return len(rendered_chunks)
            except Exception as e:
                # Rollback transaction to protect database state integrity on failures.
                print("[Ingestion] Database transaction failed. Rolling back...", flush=True)
                await session.rollback()
                raise e

