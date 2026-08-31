import numpy as np
from typing import List, Dict, Any, Optional
import re
import threading
import gc
from app.core.config import settings
from app.engines.embeddings import EmbeddingProvider, build_provider
import asyncio

class SemanticEngine:
    """
    Handles text-to-vector embedding generation and mathematical similarity calculations.
    Delegates the actual encoding to a pluggable provider (see app/engines/embeddings.py)
    so the same pipeline runs against a local model or a hosted API.
    Supports lazy loading and inactivity auto-unloading to save RAM/VRAM.
    """
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._timer = None

    @property
    def model(self) -> EmbeddingProvider:
        """
        Thread-safe lazy loader for the configured embedding provider.
        Resets the inactivity timer upon access.
        """
        with self._lock:
            if self._model is None:
                provider = build_provider()
                # Pay the model-loading cost here rather than on the first
                # encode, so lazy-loading stays observable from outside.
                if hasattr(provider, "warm"):
                    provider.warm()
                self._model = provider

            if settings.auto_unload_embeddings:
                self._reset_timer()

            return self._model

    def _reset_timer(self):
        """
        Cancels any active inactivity timer and spawns a new one.
        """
        if self._timer is not None:
            self._timer.cancel()

        self._timer = threading.Timer(settings.auto_unload_delay, self._unload_model)
        self._timer.daemon = True
        self._timer.start()

    def _unload_model(self):
        """
        Thread-safe method to delete the model from memory and release VRAM/RAM cache.
        """
        with self._lock:
            if self._model is not None:
                # Let the provider release whatever it holds (torch weights,
                # CUDA cache) before we drop our reference to it.
                if hasattr(self._model, "unload"):
                    self._model.unload()
                self._model = None
                gc.collect()

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates normalized dense vector embeddings for a list of string segments.
        Handles empty input lists gracefully by returning a correctly-shaped empty matrix.
        """
        if not texts:
            return np.empty((0, settings.embedding_dimension))

        # Providers are contracted to return L2-normalized vectors, so cosine
        # similarity reduces to a dot product downstream.
        return self.model.embed(texts)

    async def get_embeddings_async(self, texts: List[str]) -> np.ndarray:
        """
        ARCHITECT'S NOTE:
        We use asyncio.to_thread to run the blocking get_embeddings function 
        in a separate thread pool. This releases the main event loop to serve 
        other users while the GPU/CPU churns on the vector math.
        """
        return await asyncio.to_thread(self.get_embeddings, texts)

    @staticmethod
    def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculates cosine similarity between two 1D vector arrays.
        Protects against zero-norm inputs (division by zero) by returning 0.0.
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        # Guard against zero-length vectors to avoid numerical errors/NaNs.
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))   

    @staticmethod
    def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """
        Efficiently calculates all pairwise cosine similarities for a matrix of embeddings
        using vectorized matrix multiplication.
        """
        # Compute L2 norms along the horizontal axis (each embedding row).
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero by setting any zero norms to 1.0.
        norms[norms == 0.0] = 1.0
        normalized_embeddings = embeddings / norms
 
        # The dot product of normalized matrices yields the complete pairwise similarity matrix.
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        return similarity_matrix

class SemanticBoundaryDetector:
    """
    Detects topic shifts by comparing semantic distance between adjacent sentence pairs.
    Splits text when distance exceeds a dynamically computed standard deviation threshold.
    """
    def __init__(self, vector_engine : SemanticEngine, threshold_factor : float = 0.8):
        self.vector_engine = vector_engine
        self.threshold_factor = threshold_factor
        # Looks for sentence endings followed by spaces and a capital letter, excluding common abbreviations.
        self.sentence_end_regex = re.compile(r'(?<!\b[A-Z]\b)(?<!\b[a-zA-Z]\.)(?<!\b[a-zA-Z][a-zA-Z]\.)(?<!\b[a-zA-Z][a-zA-Z][a-zA-Z]\.)(?<=[.!?])\s+(?=[A-Z])')
    
    def split_into_sentences(self, text : str) -> List[str] :
        """
        Splits text block into sentences using negative lookbehind assertions to ignore
        punctuation that belongs to common abbreviations (e.g. Mr., Mrs., Dr.).
        """
        if not text.strip() :
            return []
        
        splits = re.split(r'(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bJr\.)(?<!\bSr\.)(?<!\s[A-Z]\.)(?<!^[A-Z]\.)(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in splits if s.strip()]
    
    def detect_boundaries(self, sentences : List[str]) -> List[Dict[str, Any]] :
        """
        Analyzes semantic differences between adjacent sentences.
        Identifies boundaries using a dynamic threshold (mean distance + factor * std dev).
        """
        # Edge case: If we have fewer than 2 sentences, we cannot compare adjacencies.
        if len(sentences) < 2:
            return [{"sentence": s, "distance_to_next": 0.0, "is_boundary": False} for s in sentences]
        
        # Embed all sentences to compare their semantic distance.
        embeddings = self.vector_engine.get_embeddings(sentences)

        distances = []

        # Loop calculates the distance (1.0 - similarity) between each consecutive sentence pair.
        for i in range(len(sentences) - 1) :
            sim = self.vector_engine.calculate_cosine_similarity(embeddings[i], embeddings[i + 1])
            distances.append(1.0 - sim)
            mean_dist = float(np.mean(distances))
        
        # Calculate deviation to construct a statistical threshold for detecting outliers (boundaries).
        std_dist = float(np.std(distances)) if len(distances) > 1 else 0.0
        dynamic_threshold = mean_dist + (self.threshold_factor * std_dist)

        results = []
        # Build boundary analysis mappings.
        for i, sentence in enumerate(sentences):
            if i < len(distances):
                dist = distances[i]
                # If distance exceeds our statistical threshold, mark a boundary.
                is_boundary = dist > dynamic_threshold
            else:
                dist = 0.0
                is_boundary = False
                
            results.append({
                "index": i,
                "sentence": sentence,
                "distance_to_next": dist,
                "is_boundary": is_boundary
            })
            
        return results
    
    def cluster_sentences(self, sentences: List[str], boundary_analysis: List[Dict[str, Any]]) -> List[str]:
        """
        Iterates over sentences and joins them into larger chunks.
        Splits and starts a new chunk whenever a boundary index is encountered.
        """
        chunks = []
        current_chunk_sentences = []

        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)
            # When boundary is True, flush the current list of sentences as a single chunk.
            if boundary_analysis[i]["is_boundary"]:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []

        # Flush any remaining sentences after processing loop.
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

class SlidingSemanticChunker:
    """
    A more robust chunker that compares contextual 'left' and 'right' windows of sentences.
    Provides smoother transitions and prevents narrow local semantic shifts from over-segmenting.
    """
    def __init__(self, vector_engine: SemanticEngine, window_size: int = settings.window_size, threshold_factor: float = settings.threshold_factor):
        self.vector_engine = vector_engine
        self.window_size = window_size
        self.threshold_factor = threshold_factor

    def split_into_sentences(self, text : str) -> List[str] :
        """
        Identical sentence splitter logic using negative lookbehinds for abbreviations.
        """
        if not text.strip() :
            return []
        
        splits = re.split(r'(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bJr\.)(?<!\bSr\.)(?<!\s[A-Z]\.)(?<!^[A-Z]\.)(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in splits if s.strip()]

    def compute_window_bounds(self, sentences : List[str], sentence_embeddings: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Computes sliding window semantic differences using NumPy mean pooling on pre-computed sentence embeddings.
        """
        num_sentences = len(sentences)
        # Guard clause: If text has fewer sentences than the window size on both sides,
        # we don't have enough context to compare windows. Return default non-boundaries.
        if num_sentences < (self.window_size * 2):
            return [{"index": i, "distance": 0.0, "is_boundary": False} for i in range(num_sentences)]

        # If sentence embeddings are not pre-computed, calculate them on the fly
        if sentence_embeddings is None:
            sentence_embeddings = self.vector_engine.get_embeddings(sentences)

        # Valid indexes where we can fit a full left and right window.
        valid_split_indices = list(range(self.window_size - 1, num_sentences - self.window_size))
        
        # Build window embeddings by taking the mean of sentence embeddings in each window
        left_win_embs = [
            np.mean(sentence_embeddings[idx - self.window_size + 1 : idx + 1], axis=0)
            for idx in valid_split_indices
        ]
        right_win_embs = [
            np.mean(sentence_embeddings[idx + 1 : idx + 1 + self.window_size], axis=0)
            for idx in valid_split_indices
        ]

        left_embeddings = np.array(left_win_embs)
        right_embeddings = np.array(right_win_embs)

        # Normalize the window embeddings (crucial for cosine similarity)
        left_norms = np.linalg.norm(left_embeddings, axis=1, keepdims=True)
        left_norms[left_norms == 0.0] = 1.0
        left_embeddings_normalized = left_embeddings / left_norms

        right_norms = np.linalg.norm(right_embeddings, axis=1, keepdims=True)
        right_norms[right_norms == 0.0] = 1.0
        right_embeddings_normalized = right_embeddings / right_norms

        # Compute cosine similarity using row-wise dot products
        similarities = np.sum(left_embeddings_normalized * right_embeddings_normalized, axis=1)
        window_distances = 1.0 - similarities

        # Dynamic statistical threshold.
        mean_dist = float(np.mean(window_distances))
        std_dist = float(np.std(window_distances)) if len(window_distances) > 1 else 0.0
        dynamic_threshold = mean_dist + (self.threshold_factor * std_dist)

        boundary_map = {i: False for i in range(num_sentences)}
        
        # Mark index as a boundary if window distance exceeds the calculated statistical threshold.
        for i, idx in enumerate(valid_split_indices):
            if window_distances[i] > dynamic_threshold:
                boundary_map[idx] = True

        analysis = []
        distance_idx_ptr = 0
        for i in range(num_sentences):
            # Map distance values only to indices that could be evaluated with full windows.
            if i in valid_split_indices:
                dist = window_distances[distance_idx_ptr]
                distance_idx_ptr += 1
            else:
                dist = 0.0
                
            analysis.append({
                "index": i,
                "sentence": sentences[i],
                "distance": dist,
                "is_boundary": boundary_map[i]
            })

        return analysis

    def generate_chunks(self, sentences: List[str], analysis: List[Dict[str, Any]], min_sentences: int = settings.min_sentences, min_words: int = settings.min_words) -> List[str]:
        """
        Aggregates sentences into final chunks based on window boundaries.
        Enforces minimum paragraph size constraints (sentence count or word count)
        to prevent generating tiny, uninformative chunks.
        """
        chunks = []
        buffer = []
        
        for i, sentence in enumerate(sentences):
            buffer.append(sentence)
            # When boundary is detected, verify if the buffer meets the minimum size requirements.
            if analysis[i]["is_boundary"]:
                current_text = " ".join(buffer)
                word_count = len(current_text.split())
                # If size is sufficient, finalize chunk; otherwise, accumulate further to preserve context.
                if len(buffer) >= min_sentences or word_count >= min_words:
                    chunks.append(current_text)
                    buffer = []
                
        # Flush remainder if it meets size criteria or we are at the end.
        if buffer:
            current_text = " ".join(buffer)
            word_count = len(current_text.split())
            if len(buffer) >= min_sentences or word_count >= min_words:
                chunks.append(current_text)
            
        return chunks