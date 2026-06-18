from app.engines.semantic_chunker import SlidingSemanticChunker, SemanticEngine


def test_sentence_splitter_iggnores_abbreviations():
    engine = SemanticEngine()
    chunker = SlidingSemanticChunker(vector_engine = engine)
    
    test_text = "Dr. Smith lives in the U.S.A. He is a good man."
    expected = ["Dr. Smith lives in the U.S.A.", "He is a good man."]
    
    assert chunker.split_into_sentences(test_text) == expected
    
    

def test_generate_chunks_enforces_min_size():
    engine = SemanticEngine()
    chunker = SlidingSemanticChunker(vector_engine=engine)
    
    # We pass a list of sentences and a mock boundary analysis
    sentences = ["This is a tiny sentence."]
    
    # We set is_boundary to True, meaning the algorithm wants to create a chunk here
    analysis = [{"index": 0, "sentence": sentences[0], "distance": 0.5, "is_boundary": True}]
    
    # Run chunk generation with a constraint of min 2 sentences OR 50 words
    chunks = chunker.generate_chunks(sentences, analysis, min_sentences=2, min_words=50)
    
    # Since the single sentence fails both the sentence count (1 < 2) and word count (5 < 50),
    # it must NOT be saved as a chunk.
    assert len(chunks) == 0

from unittest.mock import patch, MagicMock
from app.engines.semantic_chunker import SemanticBoundaryDetector
import numpy as np

def test_boundary_detector_zero_variance():
    # Setup mock vector engine
    mock_vector = MagicMock()
    # Mock calculate_cosine_similarity to always return 0.8
    # which makes all distances exactly 0.2 (zero variance)
    mock_vector.calculate_cosine_similarity.return_value = 0.8
    mock_vector.get_embeddings.return_value = np.ones((3, 768))
    
    detector = SemanticBoundaryDetector(vector_engine=mock_vector, threshold_factor=0.8)
    sentences = ["Sentence one.", "Sentence two.", "Sentence three."]
    
    results = detector.detect_boundaries(sentences)
    assert len(results) == 3
    for r in results:
        assert not r["is_boundary"]
        
def test_boundary_detector_edge_inputs():
    mock_vector = MagicMock()
    detector = SemanticBoundaryDetector(vector_engine=mock_vector)
    
    # 0 sentences
    assert detector.detect_boundaries([]) == []
    
    # 1 sentence
    results = detector.detect_boundaries(["Just one sentence."])
    assert len(results) == 1
    assert results[0]["is_boundary"] is False
    assert results[0]["distance_to_next"] == 0.0

def test_sliding_chunker_edge_inputs():
    mock_vector = MagicMock()
    # window_size is 3, so window_size * 2 = 6 sentences needed to compute window bounds.
    chunker = SlidingSemanticChunker(vector_engine=mock_vector, window_size=3)
    
    sentences = ["One.", "Two.", "Three."]
    analysis = chunker.compute_window_bounds(sentences)
    
    # Since len(sentences) < 6, it should immediately return analysis without embedding
    assert len(analysis) == 3
    for a in analysis:
        assert a["is_boundary"] is False
        assert a["distance"] == 0.0
    mock_vector.get_embeddings.assert_not_called()

