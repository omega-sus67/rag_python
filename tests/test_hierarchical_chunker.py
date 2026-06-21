# tests/test_hierarchical_chunker.py

import unittest
from unittest.mock import MagicMock
import numpy as np

from app.engines.hierarchical_chunker import HierarchicalSemanticEngine, RenderedChunk
from app.engines.semantic_chunker import SemanticEngine

class MockVectorEngine:
    """Mock helper representing vector model embedding interface."""
    def get_embeddings(self, texts):
        return np.ones((len(texts), 384))

    def calculate_cosine_similarity(self, vec_a, vec_b):
        return 1.0

def test_hierarchical_semantic_engine_processes_document():
    """
    PURPOSE: Verifies full document parsing, AST extraction, and hierarchical context path prepending.
    CAPABILITIES:
    - Extracts context paths like ['Main Topic', 'Subtopic A'].
    - Enriches chunk text content by prepending structural heading prefixes.
    - Handles tables and paragraphs correctly.
    """
    mock_vector_engine = MockVectorEngine()
    engine = HierarchicalSemanticEngine(vector_engine=mock_vector_engine)
    
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
    
    assert len(chunks) == 2
    
    para_chunk = chunks[0]
    table_chunk = chunks[1]
    
    assert para_chunk.metadata["source"] == "test_doc.md"
    assert para_chunk.metadata["node_type"] == "PARAGRAPH"
    assert para_chunk.metadata["structural_path"] == ["Main Topic", "Subtopic A"]
    assert "Context: Main Topic > Subtopic A\nContent: This is sentence one. This is sentence two." in para_chunk.text
    assert para_chunk.metadata["token_count"] > 0
    assert "line_number" in para_chunk.metadata
    
    assert table_chunk.metadata["source"] == "test_doc.md"
    assert table_chunk.metadata["node_type"] == "TABLE"
    assert table_chunk.metadata["structural_path"] == ["Main Topic", "Subtopic A"]
    assert "Context: Main Topic > Subtopic A\nTable Data:\n| Header |\n|--------|\n| Cell   |" in table_chunk.text
    assert "line_number" in table_chunk.metadata
    assert table_chunk.metadata["row_count"] == 3

# --- RIGOROUS EXTENDED TESTS ---

def test_hierarchical_semantic_engine_empty_ast():
    """
    PURPOSE: Tests engine decomposition on empty inputs.
    CAPABILITIES:
    - Empty strings return zero chunks.
    - Runs cleanly without throwing null pointer or recursion errors.
    """
    mock_vector_engine = MockVectorEngine()
    engine = HierarchicalSemanticEngine(vector_engine=mock_vector_engine)
    
    chunks = engine.process_document("", source_name="empty.md")
    assert len(chunks) == 0

def test_hierarchical_semantic_engine_long_list_token_split():
    """
    PURPOSE: Verifies that long list blocks exceeding limit budgets are split safely by TokenSizeOptimizer.
    CAPABILITIES:
    - Lists exceeding settings.max_tokens are broken down into sub-chunks.
    - Each sub-chunk receives the appropriate prepended contextual path.
    """
    mock_vector_engine = MockVectorEngine()
    engine = HierarchicalSemanticEngine(vector_engine=mock_vector_engine)
    
    # Configure low max token size on the optimizer for testing
    engine.token_optimizer.max_tokens = 5
    engine.token_optimizer.overlap_tokens = 2
    
    raw_markdown = (
        "# Parent Heading\n"
        "- Bullet 1\n"
        "- Bullet 2\n"
        "- Bullet 3\n"
        "- Bullet 4\n"
        "- Bullet 5\n"
    )
    
    chunks = engine.process_document(raw_markdown, source_name="long_list.md")
    # Due to small token limits, the list should be partitioned into multiple sub-chunks
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["node_type"] == "LIST"
        assert chunk.metadata["structural_path"] == ["Parent Heading"]
        assert "Context: Parent Heading" in chunk.text
