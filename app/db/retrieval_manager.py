# retrieval_manager.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
# Import pgvector Vector distance operator
from pgvector.sqlalchemy import Vector

from app.engines.semantic_chunker import SemanticEngine
from app.db.database_manager import dbChunk, dbFile

class HierarchicalRAGRetriever:
    """
    Handles semantic vector similarity queries against the pgvector database schema.
    Retrieves the most contextually relevant chunks for a given user query.
    """
    def __init__(self, db_session: AsyncSession, vector_engine: SemanticEngine):
        self.db = db_session
        self.vector_engine = vector_engine

    async def retrieve_relevant_chunks(self, query_text: str, file_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar database chunks:
        1. Embeds the user question using the same embedding model.
        2. Constructs a pgvector cosine_distance query scoped to the current file.
        3. Executes the query and outputs records mapped with explicit similarity scores.
        """
        # Step 1: Embed the incoming query.
        # This converts user text into a 1D vector (e.g. size 384).
        query_vector_ndarray = self.vector_engine.get_embeddings([query_text])[0]
        
        # Convert the NumPy array representation into a standard float list for database execution.
        query_vector_list = query_vector_ndarray.tolist()

        # Step 2: Set up the pgvector cosine distance database query.
        # pgvector uses the .cosine_distance() method to compute vector difference in the database.
        # Formula: Distance = 1.0 - CosineSimilarity. Lower distance means closer meaning.
        distance_expression = dbChunk.embeddings.cosine_distance(query_vector_list)

        query_statement = (
            select(dbChunk, distance_expression.label("distance"))
            .where(dbChunk.file_id == file_id)  # Scope query to the target document
            .order_by("distance")               # Order ascending so most similar chunks are first
            .limit(top_k)                       # Enforce top_k limit to stay within LLM context boundaries
        )

        # Step 3: Run search query and package returned values.
        results = (await self.db.execute(query_statement)).all()
        
        matched_payloads = []
        for row in results:
            chunk_record = row[0]
            distance_score = row[1]
            
            matched_payloads.append({
                "chunk_id": chunk_record.id,
                "chunk_index": chunk_record.chunk_index,
                "text": chunk_record.text_data,
                "distance": float(distance_score),
                # Convert distance back to similarity score (Similarity = 1.0 - Distance) for user display.
                "similarity": float(1.0 - distance_score)
            })

        return matched_payloads
    
    async def retrieve_across_all_documents(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic vector similarity query across ALL documents in the database.
        Merges results and maps them with source documents.
        """
        # 1. Embed incoming query
        query_vector_ndarray = self.vector_engine.get_embeddings([query_text])[0]
        query_vector_list = query_vector_ndarray.tolist()

        # 2. Query pgvector cosine distance across all files, joining with the files table to get titles
        distance_expression = dbChunk.embeddings.cosine_distance(query_vector_list)
        
        # We perform a join between doc_chunks and files so the agent knows which document each chunk came from
        query_statement = (
            select(dbChunk, dbFile.title, distance_expression.label("distance"))
            .join(dbFile, dbChunk.file_id == dbFile.id)
            .order_by("distance")
            .limit(top_k)
        )

        results = (await self.db.execute(query_statement)).all()
        
        matched_payloads = []
        for row in results:
            chunk_record = row[0]
            doc_title = row[1]
            distance_score = row[2]
            
            matched_payloads.append({
                "chunk_id": chunk_record.id,
                "file_id": chunk_record.file_id,
                "document_title": doc_title,
                "chunk_index": chunk_record.chunk_index,
                "text": chunk_record.text_data,
                "distance": float(distance_score),
                "similarity": float(1.0 - distance_score)
            })

        return matched_payloads
