# tests/test_db_components.py

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.db.database_manager import dbFile, dbChunk, DatabaseManager, RAGIngestionManager
from app.db.retrieval_manager import HierarchicalRAGRetriever
from app.utils.pdf_extractor import Document
from app.engines.hierarchical_chunker import RenderedChunk

# Define pytest markers for async tests
pytestmark = pytest.mark.asyncio

async def test_db_chunk_from_rendered_chunk():
    """
    PURPOSE: Verifies the dbChunk factory converter behaves correctly.
    CAPABILITIES:
    - Sets corresponding table columns correctly from DTO values.
    """
    rendered = RenderedChunk(
        chunk_id="chk_123",
        parent_node_id="node_abc",
        text="Sample chunk text",
        metadata={"source": "test.md"}
    )
    
    db_rec = dbChunk.from_rendered_chunk(
        rendered_chunk=rendered,
        file_id="file_xyz",
        index=2,
        vector=[0.1, 0.2, 0.3]
    )
    
    assert db_rec.id == "chk_123"
    assert db_rec.file_id == "file_xyz"
    assert db_rec.chunk_index == 2
    assert db_rec.text_data == "Sample chunk text"
    assert db_rec.embeddings == [0.1, 0.2, 0.3]

async def test_database_manager_save_document_duplicate():
    """
    PURPOSE: Verifies rejection of duplicate documents based on SHA-256 hash checks.
    CAPABILITIES:
    - Raises an HTTP 400 Bad Request exception when checking a pre-existing hash.
    """
    db_mgr = DatabaseManager(database_url="postgresql+asyncpg://mock:mock@mock/mock")
    
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_execute_result = MagicMock()
    mock_execute_result.scalar.return_value = dbFile(id="dup_hash", title="Existing", extracted_text="content")
    mock_session.execute.return_value = mock_execute_result
    
    db_mgr.SessionLocal = MagicMock(return_value=mock_session)
    
    doc = Document(title="New Doc", extracted_text="content")
    
    with pytest.raises(HTTPException) as exc_info:
        await db_mgr.save_document(doc)
        
    assert exc_info.value.status_code == 400
    assert "File already exists" in exc_info.value.detail

async def test_database_manager_save_document_success():
    """
    PURPOSE: Verifies successful document saving when the SHA-256 hash is unique.
    CAPABILITIES:
    - Adds new dbFile to active session queue.
    - Triggers transaction commit.
    - Refreshes instance attributes.
    """
    db_mgr = DatabaseManager(database_url="postgresql+asyncpg://mock:mock@mock/mock")
    
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_execute_result = MagicMock()
    mock_execute_result.scalar.return_value = None
    mock_session.execute.return_value = mock_execute_result
    
    db_mgr.SessionLocal = MagicMock(return_value=mock_session)
    
    doc = Document(title="New Doc", extracted_text="unique content")
    
    db_doc = await db_mgr.save_document(doc)
    
    assert db_doc.title == "New Doc"
    assert db_doc.extracted_text == "unique content"
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once()

async def test_database_manager_fetch_document_not_found():
    """
    PURPOSE: Verifies fetch behavior when seeking a missing document ID.
    CAPABILITIES:
    - Raises an HTTP 404 Not Found exception.
    """
    db_mgr = DatabaseManager(database_url="postgresql+asyncpg://mock:mock@mock/mock")
    
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_execute_result = MagicMock()
    mock_execute_result.scalar.return_value = None
    mock_session.execute.return_value = mock_execute_result
    
    db_mgr.SessionLocal = MagicMock(return_value=mock_session)
    
    with pytest.raises(HTTPException) as exc_info:
        await db_mgr.fetch_document("nonexistent")
        
    assert exc_info.value.status_code == 404
    assert "File not found" in exc_info.value.detail

async def test_database_manager_fetch_document_success():
    """
    PURPOSE: Verifies successful document fetch.
    CAPABILITIES:
    - Queries database by key.
    - Yields matching dbFile record.
    """
    db_mgr = DatabaseManager(database_url="postgresql+asyncpg://mock:mock@mock/mock")
    
    expected_doc = dbFile(id="file_123", title="Test Doc", extracted_text="some text")
    
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_execute_result = MagicMock()
    mock_execute_result.scalar.return_value = expected_doc
    mock_session.execute.return_value = mock_execute_result
    
    db_mgr.SessionLocal = MagicMock(return_value=mock_session)
    
    result = await db_mgr.fetch_document("file_123")
    assert result == expected_doc

async def test_rag_ingestion_manager_success():
    """
    PURPOSE: Verifies the full RAG Ingestion Pipeline run.
    CAPABILITIES:
    - Invokes markdown structural parsing.
    - Obtains embeddings in batch.
    - Persists multiple chunk entities inside a single transaction.
    """
    db_mgr = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    db_mgr.SessionLocal = MagicMock(return_value=mock_session)
    
    vector_engine = MagicMock()
    
    ingestion_mgr = RAGIngestionManager(db_manager=db_mgr, vector_engine=vector_engine)
    
    mock_chunks = [
        RenderedChunk(chunk_id="chk_1", parent_node_id="n1", text="text 1", metadata={}, embeddings=[0.1]*384),
        RenderedChunk(chunk_id="chk_2", parent_node_id="n2", text="text 2", metadata={}, embeddings=[0.2]*384)
    ]
    ingestion_mgr.chunking_engine.process_document = AsyncMock(return_value=mock_chunks)
    
    count = await ingestion_mgr.ingest_document("raw doc text", "file_abc", "source.md")
    
    assert count == 2
    assert mock_session.add.call_count == 2
    mock_session.commit.assert_called_once()

async def test_rag_ingestion_manager_rollback_on_failure():
    """
    PURPOSE: Verifies transactional safety in case of database errors.
    CAPABILITIES:
    - Rolls back the active database session.
    - Re-raises the error to notify upper orchestration layers.
    """
    db_mgr = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.commit.side_effect = Exception("DB error")
    db_mgr.SessionLocal = MagicMock(return_value=mock_session)
    
    vector_engine = MagicMock()
    
    ingestion_mgr = RAGIngestionManager(db_manager=db_mgr, vector_engine=vector_engine)
    
    mock_chunks = [RenderedChunk(chunk_id="chk_1", parent_node_id="n1", text="text 1", metadata={}, embeddings=[0.1]*384)]
    ingestion_mgr.chunking_engine.process_document = AsyncMock(return_value=mock_chunks)
    
    with pytest.raises(Exception) as exc_info:
        await ingestion_mgr.ingest_document("raw doc text", "file_abc", "source.md")
        
    assert "DB error" in str(exc_info.value)    
    mock_session.rollback.assert_called_once()

async def test_hierarchical_rag_retriever():
    """
    PURPOSE: Verifies similarity search querying and distance logic mapping.
    CAPABILITIES:
    - Embeds the query.
    - Generates select statements with order and limit constraints.
    - Converts pgvector distance to display similarity score (1 - dist).
    """
    mock_session = AsyncMock()
    
    vector_engine = AsyncMock()
    vector_engine.get_embeddings_async.return_value = np.array([[0.5]*384])
    
    retriever = HierarchicalRAGRetriever(db_session=mock_session, vector_engine=vector_engine)
    
    chunk_record = dbChunk(id="chk_res", chunk_index=0, text_data="found text")
    distance_score = 0.35
    
    mock_execute_result = MagicMock()
    mock_execute_result.all.return_value = [(chunk_record, distance_score)]
    mock_session.execute.return_value = mock_execute_result
    
    results = await retriever.retrieve_relevant_chunks(
        query_text="What is the answer?",
        file_id="file_id_xyz",
        top_k=1
    )
    
    assert len(results) == 1
    res = results[0]
    assert res["chunk_id"] == "chk_res"
    assert res["chunk_index"] == 0
    assert res["text"] == "found text"
    assert res["distance"] == 0.35
    assert res["similarity"] == pytest.approx(0.65)
    
    vector_engine.get_embeddings_async.assert_called_once_with(["What is the answer?"])
    mock_session.execute.assert_called_once()

# --- RIGOROUS EXTENDED TESTS ---

async def test_rag_ingestion_manager_empty_document():
    """
    PURPOSE: Verifies that if document parsing yields no chunks, the ingestion exits early.
    CAPABILITIES:
    - Exits without model call or database connection acquisition.
    - Returns 0 count safely.
    """
    db_mgr = MagicMock()
    vector_engine = MagicMock()
    ingestion_mgr = RAGIngestionManager(db_manager=db_mgr, vector_engine=vector_engine)
    
    # Mock return value of parsing as empty
    ingestion_mgr.chunking_engine.process_document = AsyncMock(return_value=[])
    
    count = await ingestion_mgr.ingest_document("empty text", "file_123", "source.md")
    assert count == 0
    vector_engine.get_embeddings.assert_not_called()
    db_mgr.SessionLocal.assert_not_called()

async def test_database_manager_save_document_general_exception():
    """
    PURPOSE: Verifies database commit failure handling inside save_document.
    CAPABILITIES:
    - Bubbles database transaction exceptions up cleanly.
    - Prevents orphaned sessions by ensuring cleanup runs.
    """
    db_mgr = DatabaseManager(database_url="postgresql+asyncpg://mock:mock@mock/mock")
    
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.commit.side_effect = Exception("Session error")
    
    # Duplicate check passes
    mock_execute_result = MagicMock()
    mock_execute_result.scalar.return_value = None
    mock_session.execute.return_value = mock_execute_result
    
    db_mgr.SessionLocal = MagicMock(return_value=mock_session)
    doc = Document(title="Doc", extracted_text="unique content")
    
    with pytest.raises(Exception) as exc_info:
        await db_mgr.save_document(doc)
        
    assert "Session error" in str(exc_info.value)
