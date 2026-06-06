# retriever.py
from sqlalchemy import select
from sqlalchemy.orm import Session
from typing import List, Dict, Any
# Assuming pgvector provides the cosine_distance operator via its SQLAlchemy extension
from pgvector.sqlalchemy import Vector

from app.engines.semantic_engine import SemanticEngine
from app.db.database_manager import dbChunk

class HierarchicalRAGRetriever:
    """Handles high-performance semantic vector queries against the pgvector database schema."""
    def __init__(self, db_session: Session, vector_engine: SemanticEngine):
        self.db = db_session
        self.vector_engine = vector_engine

    def retrieve_relevant_chunks(self, query_text: str, file_id: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Converts a raw question into a vector coordinate and extracts 
        the top_k closest structural text blocks from PostgreSQL.
        """
        # Step 1: Embed the user's incoming question using the exact same vector model
        # This yields a 1D numpy array of shape (384,)
        query_vector_ndarray = self.vector_engine.get_embeddings([query_text])[0]
        
        # Convert the raw numpy array back into a plain Python list of floats for the SQL driver
        query_vector_list = query_vector_ndarray.tolist()

        # Step 2: Construct the pgvector distance query
        # We use the built-in '.cosine_distance()' operator provided by pgvector
        # Formula running inside Postgres: 1 - CosineSimilarity
        distance_expression = dbChunk.embeddings.cosine_distance(query_vector_list)

        query_statement = (
            select(dbChunk, distance_expression.label("distance"))
            .where(dbChunk.file_id == file_id)  # Scope search to the target document
            .order_by("distance")               # Ascending order: lowest distance means closest meaning
            .limit(top_k)                       # Restrict results to protect our LLM context window
        )

        # Step 3: Execute the transaction and package results
        results = self.db.execute(query_statement).all()
        
        matched_payloads = []
        for row in results:
            chunk_record = row[0]
            distance_score = row[1]
            
            matched_payloads.append({
                "chunk_id": chunk_record.id,
                "chunk_index": chunk_record.chunk_index,
                "text": chunk_record.text_data,
                "distance": float(distance_score),
                "similarity": float(1.0 - distance_score) # Invert distance back to a readable similarity score
            })

        return matched_payloads