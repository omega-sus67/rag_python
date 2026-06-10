import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict, Any
import re
from app.core.config import settings


class SemanticEngine:
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, settings.embedding_dimension))
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True
        )
        return embeddings
    @staticmethod
    def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))   
    @staticmethod
    def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized_embeddings = embeddings / norms
 
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        return similarity_matrix

class SemanticBoundaryDetector:
    def __init__(self,vector_engine : SemanticEngine, threshold_factor : float = 0.8):
        self.vector_engine = vector_engine
        self.threshold_factor = threshold_factor
        self.sentence_end_regex = re.compile(r'(?<!\b\p{Lu}\b)(?<!\b\p{L}{1,3}\.)(?<=[.!?])\s+(?=\p{Lu})', re.UNICODE)
    
    def split_into_sentences(self, text : str) -> List[str] :
        if not text.strip() :
            return []
        
        splits = re.split(r'(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bJr\.)(?<!\bSr\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in splits if s.strip()]
    
    def detect_boundaries(self, sentences : List[str]) -> List[Dict[str, Any]] :

        if len(sentences) < 2:
            return [{"sentence": s, "distance_to_next": 0.0, "is_boundary": False} for s in sentences]
        
        embeddings = self.vector_engine.get_embeddings(sentences)

        distances = []

        for i in range(len(sentences) - 1) :
            sim = self.vector_engine.calculate_cosine_similarity(embeddings[i], embeddings[i + 1])
            distances.append(1.0 - sim)
            mean_dist = float(np.mean(distances))
        
        std_dist = float(np.std(distances)) if len(distances) > 1 else 0.0
        dynamic_threshold = mean_dist + (self.threshold_factor * std_dist)

        results = []
        for i, sentence in enumerate(sentences):
            if i < len(distances):
                dist = distances[i]
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
        chunks = []
        current_chunk_sentences = []

        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)
            if boundary_analysis[i]["is_boundary"]:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

class SlidingSemanticChunker:
    def __init__(self, vector_engine: SemanticEngine, window_size: int = settings.window_size, threshold_factor: float = settings.threshold_factor):
        self.vector_engine = vector_engine
        self.window_size = window_size
        self.threshold_factor = threshold_factor

    def split_into_sentences(self, text : str) -> List[str] :
        if not text.strip() :
            return []
        
        splits = re.split(r'(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bJr\.)(?<!\bSr\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in splits if s.strip()]

    def compute_window_bounds(self, sentences : List[str]) -> List[Dict[str, Any]]:
        num_sentences = len(sentences)
        if num_sentences < (self.window_size * 2):
            return [{"index": i, "distance": 0.0, "is_boundary": False} for i in range(num_sentences)]

        left_window_strings = []
        right_window_strings = []
        
        valid_split_indices = list(range(self.window_size - 1, num_sentences - self.window_size))
        
        for idx in valid_split_indices:
            left_win = sentences[idx - self.window_size + 1 : idx + 1]
            right_win = sentences[idx + 1 : idx + 1 + self.window_size]
            
            left_window_strings.append(" ".join(left_win))
            right_window_strings.append(" ".join(right_win))

        left_embeddings = self.vector_engine.get_embeddings(left_window_strings)
        right_embeddings = self.vector_engine.get_embeddings(right_window_strings)

        window_distances = []
        for i in range(len(valid_split_indices)):
            sim = self.vector_engine.calculate_cosine_similarity(left_embeddings[i], right_embeddings[i])
            window_distances.append(1.0 - sim)

        mean_dist = float(np.mean(window_distances))
        std_dist = float(np.std(window_distances)) if len(window_distances) > 1 else 0.0
        dynamic_threshold = mean_dist + (self.threshold_factor * std_dist)

        boundary_map = {i: False for i in range(num_sentences)}
        
        for i, idx in enumerate(valid_split_indices):
            if window_distances[i] > dynamic_threshold:
                boundary_map[idx] = True

        analysis = []
        distance_idx_ptr = 0
        for i in range(num_sentences):
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
        chunks = []
        buffer = []
        
        for i, sentence in enumerate(sentences):
            buffer.append(sentence)
            if analysis[i]["is_boundary"]:
                current_text = " ".join(buffer)
                word_count = len(current_text.split())
                if len(buffer) >= min_sentences or word_count >= min_words:
                    chunks.append(current_text)
                    buffer = []
                
        if buffer:
            current_text = " ".join(buffer)
            word_count = len(current_text.split())
            if len(buffer) >= min_sentences or word_count >= min_words:
                chunks.append(current_text)
            
        return chunks