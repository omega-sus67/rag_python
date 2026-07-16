# tests/test_semantic_chunker.py

import numpy as np
from unittest.mock import MagicMock
from app.engines.semantic_chunker import SlidingSemanticChunker, SemanticEngine, SemanticBoundaryDetector

def test_sentence_splitter_iggnores_abbreviations():
    """
    PURPOSE: Verifies that sentence segmentation correctly skips punctuation
    belonging to common abbreviations (e.g. Dr., U.S.A.).
    CAPABILITIES:
    - Splits text on terminal punctuation (. or ! or ?).
    - Ignores internal periods inside abbreviations to keep sentences structurally complete.
    """
    engine = SemanticEngine()
    chunker = SlidingSemanticChunker(vector_engine=engine)
    
    test_text = "Dr. Smith lives in the U.S.A. He is a good man."
    expected = ["Dr. Smith lives in the U.S.A.", "He is a good man."]
    
    assert chunker.split_into_sentences(test_text) == expected

def test_generate_chunks_enforces_min_size():
    """
    PURPOSE: Verifies size criteria rules in chunk creation.
    CAPABILITIES:
    - Rejects boundary creation if the sentence block fails both min_sentences AND min_words.
    - Accretes text forward to prevent tiny fragment chunks.
    """
    engine = SemanticEngine()
    chunker = SlidingSemanticChunker(vector_engine=engine)
    
    sentences = ["This is a tiny sentence."]
    analysis = [{"index": 0, "sentence": sentences[0], "distance": 0.5, "is_boundary": True}]
    
    chunks = chunker.generate_chunks(sentences, analysis, min_sentences=2, min_words=50)
    assert len(chunks) == 0

def test_boundary_detector_zero_variance():
    """
    PURPOSE: Verifies boundary detector behavior when all adjacent distances are identical (zero variance).
    CAPABILITIES:
    - Gracefully handles zero variance without divide-by-zero crashes.
    - Correctly marks no boundaries since there are no outlier semantic shifts.
    """
    mock_vector = MagicMock()
    mock_vector.calculate_cosine_similarity.return_value = 0.8
    mock_vector.get_embeddings.return_value = np.ones((3, 768))
    
    detector = SemanticBoundaryDetector(vector_engine=mock_vector, threshold_factor=0.8)
    sentences = ["Sentence one.", "Sentence two.", "Sentence three."]
    
    results = detector.detect_boundaries(sentences)
    assert len(results) == 3
    for r in results:
        assert not r["is_boundary"]

def test_boundary_detector_edge_inputs():
    """
    PURPOSE: Verifies boundary detection boundary cases for small array sizes.
    CAPABILITIES:
    - Empty sentence list yields an empty analysis output.
    - Single sentence input returns no boundary mark and 0.0 distance safely.
    """
    mock_vector = MagicMock()
    detector = SemanticBoundaryDetector(vector_engine=mock_vector)
    
    assert detector.detect_boundaries([]) == []
    
    results = detector.detect_boundaries(["Just one sentence."])
    assert len(results) == 1
    assert results[0]["is_boundary"] is False
    assert results[0]["distance_to_next"] == 0.0

def test_sliding_chunker_edge_inputs():
    """
    PURPOSE: Verifies sliding window chunker behavior on inputs shorter than twice the window size.
    CAPABILITIES:
    - Bypasses embedding computation to save processing overhead.
    - Safely returns default non-boundary results.
    """
    mock_vector = MagicMock()
    chunker = SlidingSemanticChunker(vector_engine=mock_vector, window_size=3)
    
    sentences = ["One.", "Two.", "Three."]
    analysis = chunker.compute_window_bounds(sentences)
    
    assert len(analysis) == 3
    for a in analysis:
        assert a["is_boundary"] is False
        assert a["distance"] == 0.0
    mock_vector.get_embeddings.assert_not_called()

# --- RIGOROUS EXTENDED TESTS ---

def test_semantic_engine_get_embeddings_empty():
    """
    PURPOSE: Verifies that get_embeddings handles empty list input gracefully.
    CAPABILITIES:
    - Returns an empty NumPy array with dimension matching settings.
    - Prevents model execution roundtrips.
    """
    from app.core.config import settings
    engine = SemanticEngine()
    result = engine.get_embeddings([])
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, settings.embedding_dimension)

def test_semantic_engine_cosine_similarity_zero_norm():
    """
    PURPOSE: Verifies divide-by-zero protection in cosine similarity.
    CAPABILITIES:
    - Returns 0.0 similarity if either vector is all-zeros (zero norm).
    - Prevents runtime floating point errors or NaN outputs.
    """
    vec_a = np.zeros(384)
    vec_b = np.ones(384)
    
    sim = SemanticEngine.calculate_cosine_similarity(vec_a, vec_b)
    assert sim == 0.0

def test_semantic_engine_similarity_matrix_zero_norm():
    """
    PURPOSE: Verifies that pairwise similarity matrix calculation handles zero-norm rows safely.
    CAPABILITIES:
    - Vectorized division handles zero norms by substituting 1.0.
    - Returns a valid similarity matrix without producing NaNs.
    """
    embeddings = np.array([
        [0.0] * 384,
        [1.0] * 384
    ])
    matrix = SemanticEngine.calculate_similarity_matrix(embeddings)
    assert matrix.shape == (2, 2)
    assert not np.isnan(matrix).any()

def test_boundary_detector_cluster_sentences_simple():
    """
    PURPOSE: Verifies clustering behavior from boundary analysis mappings.
    CAPABILITIES:
    - Correctly groups adjacent sentence streams.
    - Splits clusters at designated boundary index slots.
    """
    detector = SemanticBoundaryDetector(vector_engine=MagicMock())
    sentences = ["S1.", "S2.", "S3.", "S4."]
    analysis = [
        {"is_boundary": False},
        {"is_boundary": True},  # Boundary after S2
        {"is_boundary": False},
        {"is_boundary": False}
    ]
    clusters = detector.cluster_sentences(sentences, analysis)
    assert len(clusters) == 2
    assert clusters[0] == "S1. S2."
    assert clusters[1] == "S3. S4."

def test_sliding_chunker_sufficient_sentences():
    """
    PURPOSE: Verifies sliding window calculations on valid input sizes.
    CAPABILITIES:
    - Computes window vectors for indices where left/right contexts fit.
    - Sets boundary indicator when window distance exceeds dynamic thresholds.
    """
    mock_vector = MagicMock()
    # 6 sentences total. Let's make S1, S2, S3 semantically similar (topic A),
    # and S4, S5, S6 semantically similar (topic B).
    # We return mock embeddings showing this transition.
    embedding_dim = 384
    mock_embeddings = np.zeros((6, embedding_dim))
    mock_embeddings[0:3, 0] = 1.0  # Topic A: [1, 0, ...]
    mock_embeddings[3:6, 1] = 1.0  # Topic B: [0, 1, ...]
    
    mock_vector.get_embeddings.return_value = mock_embeddings
    
    chunker = SlidingSemanticChunker(vector_engine=mock_vector, window_size=2)
    sentences = ["S1", "S2", "S3", "S4", "S5", "S6"]
    
    analysis = chunker.compute_window_bounds(sentences)
    assert len(analysis) == 6
    # With window_size=2:
    # Split index 2 (split after S3) compares left window (S2, S3) -> Topic A,
    # with right window (S4, S5) -> Topic B. This is the boundary of topic shift!
    assert analysis[2]["is_boundary"] is True

def test_semantic_engine_lazy_loading():
    """
    PURPOSE: Verifies that the embedding model is loaded lazily and only when required.
    """
    engine = SemanticEngine()
    # The model should start as None before any calls or access
    assert engine._model is None
    
    # Accessing the model property should trigger loading
    _ = engine.model
    assert engine._model is not None

def test_semantic_engine_auto_unload():
    """
    PURPOSE: Verifies that the model is automatically unloaded after the configured delay.
    """
    import time
    from app.core.config import settings
    
    # Override settings for the test
    original_unload = settings.auto_unload_embeddings
    original_delay = settings.auto_unload_delay
    
    settings.auto_unload_embeddings = True
    settings.auto_unload_delay = 0.1  # 100 milliseconds
    
    try:
        engine = SemanticEngine()
        # Trigger lazy load
        _ = engine.model
        assert engine._model is not None
        
        # Wait for longer than 0.1 seconds to trigger timer
        time.sleep(0.25)
        
        # Check that the model has been cleared
        assert engine._model is None
    finally:
        # Restore settings
        settings.auto_unload_embeddings = original_unload
        settings.auto_unload_delay = original_delay



