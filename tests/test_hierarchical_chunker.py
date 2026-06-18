# tests/test_hierarchical_chunker.py

import unittest
from unittest.mock import MagicMock
import numpy as np

from app.engines.hierarchical_chunker import HierarchicalSemanticEngine, RenderedChunk
from app.engines.semantic_chunker import SemanticEngine

class MockVectorEngine:
    def get_embeddings(self, texts):
        # Return a mock array of appropriate shape
        return np.ones((len(texts), 768))

    def calculate_cosine_similarity(self, vec_a, vec_b):
        return 1.0

def test_hierarchical_semantic_engine_processes_document():
    # Construct HierarchicalSemanticEngine with mocked vector engine
    mock_vector_engine = MockVectorEngine()
    engine = HierarchicalSemanticEngine(vector_engine=mock_vector_engine)
    
    # We construct a document that:
    # 1. Has H1 and H2
    # 2. Has a paragraph with at least 2 sentences (to satisfy min_sentences=2)
    # 3. Has a table
    raw_markdown = (
        "# Main Topic\n"
        "## Subtopic A\n"
        "This is sentence one. This is sentence two.\n"
        "\n"
        "| Header |\n"
        "|--------|\n"
        "| Cell   |"
    )
    
    chunks = engine.process_document(raw_markdown, source_name="test_doc.md")
    
    # We expect 2 chunks: 
    # - 1 for the paragraph (under Main Topic > Subtopic A)
    # - 1 for the table (under Main Topic > Subtopic A)
    assert len(chunks) == 2
    
    para_chunk = chunks[0]
    table_chunk = chunks[1]
    
    # Verify Paragraph Chunk properties
    assert para_chunk.metadata["source"] == "test_doc.md"
    assert para_chunk.metadata["node_type"] == "PARAGRAPH"
    assert para_chunk.metadata["structural_path"] == ["Main Topic", "Subtopic A"]
    assert "Context: Main Topic > Subtopic A\nContent: This is sentence one. This is sentence two." in para_chunk.text
    assert para_chunk.metadata["token_count"] > 0
    assert "line_number" in para_chunk.metadata
    
    # Verify Table Chunk properties
    assert table_chunk.metadata["source"] == "test_doc.md"
    assert table_chunk.metadata["node_type"] == "TABLE"
    assert table_chunk.metadata["structural_path"] == ["Main Topic", "Subtopic A"]
    assert "Context: Main Topic > Subtopic A\nTable Data:\n| Header |\n|--------|\n| Cell   |" in table_chunk.text
    assert "line_number" in table_chunk.metadata
    assert table_chunk.metadata["row_count"] == 3
