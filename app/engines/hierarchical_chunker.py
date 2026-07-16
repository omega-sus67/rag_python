import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from app.engines.semantic_chunker import SemanticEngine, SlidingSemanticChunker
from app.engines.markdown_parser import markdownParser, DocNode, NodeType
from app.utils.token_optimizer import TokenSizeOptimizer

from app.core.config import settings

class RenderedChunk:
    """
    A unified data transfer object (DTO) representing a completed RAG chunk.
    Enriched with heading context path and metadata fields ready for database insertion.
    """
    def __init__(self, chunk_id: str, parent_node_id: str, text: str, metadata: Dict[str, Any], embeddings: Optional[List[float]] = None):
        self.chunk_id = chunk_id
        # Reference back to the original AST DocNode.
        self.parent_node_id = parent_node_id
        # Contains context path prepended to the actual text content.
        self.text = text
        self.metadata = metadata
        self.embeddings = embeddings

    def __repr__(self) -> str:
        return f"<RenderedChunk id={self.chunk_id} parent={self.parent_node_id} text_len={len(self.text)}>"


class HierarchicalSemanticEngine:
    """
    Orchestrator that combines document structure parsing, semantic boundary detection,
    and token length constraints to construct optimal contextual RAG chunks.
    """
    def __init__(self, vector_engine: SemanticEngine, window_size: int = settings.window_size, threshold_factor: float = settings.threshold_factor):
        # markdownParser handles structural AST hierarchy.
        self.structure_parser = markdownParser()
        # SlidingSemanticChunker handles sentence clustering.
        self.boundary_detector = SlidingSemanticChunker(
            vector_engine=vector_engine, 
            window_size=window_size, 
            threshold_factor=threshold_factor
        )
        # TokenSizeOptimizer prevents context window overflow.
        self.token_optimizer = TokenSizeOptimizer()

    def _collect_sentences(self, node: DocNode, sentence_list: List[str], node_sentence_indices: Dict[str, range]) -> None:
        """
        Recursively traverses the AST to extract and collect all sentences from paragraphs and list items.
        """
        if node.node_type in [NodeType.PARAGRAPH, NodeType.LIST_ITEM]:
            sentences = self.boundary_detector.split_into_sentences(node.text)
            if sentences:
                start_idx = len(sentence_list)
                sentence_list.extend(sentences)
                end_idx = len(sentence_list)
                node_sentence_indices[node.node_id] = range(start_idx, end_idx)
        
        for child in node.children:
            self._collect_sentences(child, sentence_list, node_sentence_indices)

    async def process_document(self, raw_markdown_text: str, source_name: str = "unknown") -> List[RenderedChunk]:
        """
        Orchestrates document ingestion:
        1. Parses markdown text into a structural node tree (AST).
        2. Gathers and batch-embeds all unique sentences at the document level.
        3. Traverses and decomposes the AST recursively into semantic chunks.
        4. Computes final chunk embeddings based on the configured strategy.
        """
        print(f"[Ingestion] Parsing raw markdown for '{source_name}' to construct AST tree...", flush=True)
        root_node = self.structure_parser.parse(raw_markdown_text)
        
        # Step 1: Collect all sentences across the entire AST
        sentence_list: List[str] = []
        node_sentence_indices: Dict[str, range] = {}
        self._collect_sentences(root_node, sentence_list, node_sentence_indices)
        print(f"[Ingestion] Extracted {len(sentence_list)} sentences from AST. Generating sentence-level embeddings...", flush=True)
        
        # Step 2: Batch-embed all sentences in one call
        sentence_embeddings = await self.boundary_detector.vector_engine.get_embeddings_async(sentence_list)
        print(f"[Ingestion] Completed sentence embeddings. Decomposing AST tree into semantic chunks...", flush=True)
        
        # Step 3: Decompose the tree
        final_chunks: List[RenderedChunk] = []
        self._decompose_node(root_node, final_chunks, source_name, sentence_list, node_sentence_indices, sentence_embeddings)
        print(f"[Ingestion] Decomposed AST into {len(final_chunks)} chunks. Generating final chunk embeddings (strategy: '{settings.chunk_embedding_strategy}')...", flush=True)
        
        # Step 4: Populate final chunk embeddings
        if final_chunks:
            if settings.chunk_embedding_strategy == "hybrid":
                # Batch embed all final chunk texts
                chunk_texts = [c.text for c in final_chunks]
                print(f"[Ingestion] Hybrid strategy: generating embeddings for all {len(chunk_texts)} chunks...", flush=True)
                chunk_vectors = await self.boundary_detector.vector_engine.get_embeddings_async(chunk_texts)
                for idx, chunk in enumerate(final_chunks):
                    chunk.embeddings = chunk_vectors[idx].tolist()
            else:
                # Pure Vector Math strategy (Option B)
                # For chunks that do not have embeddings pre-populated (like lists/tables),
                # we batch-embed their texts. Paragraphs are already pre-populated.
                chunks_to_embed_indices = []
                texts_to_embed = []
                for idx, chunk in enumerate(final_chunks):
                    if chunk.embeddings is None:
                        chunks_to_embed_indices.append(idx)
                        texts_to_embed.append(chunk.text)
                
                if texts_to_embed:
                    print(f"[Ingestion] Vector Math strategy: embedding remaining {len(texts_to_embed)} non-paragraph chunks (lists, tables)...", flush=True)
                    chunk_vectors = await self.boundary_detector.vector_engine.get_embeddings_async(texts_to_embed)
                    for i, idx in enumerate(chunks_to_embed_indices):
                        final_chunks[idx].embeddings = chunk_vectors[i].tolist()
        
        print(f"[Ingestion] Completed final chunk embedding generation.", flush=True)
        return final_chunks

    def _decompose_node(
           self, 
           node: DocNode, 
           chunk_accumulator: List[RenderedChunk], 
           source_name: str,
           sentence_list: List[str],
           node_sentence_indices: Dict[str, range],
           sentence_embeddings: np.ndarray
       ) -> None:
           """
           Recursively decomposes AST tree nodes into rendered chunks using pre-computed sentence embeddings.
           """
           # --- Handle Paragraphs and Lists Items ---
           if node.node_type in [NodeType.PARAGRAPH, NodeType.LIST_ITEM]:
               # Retrieve the pre-computed sentences and embeddings slice for this node
               sentences = []
               embeddings_slice = np.empty((0, settings.embedding_dimension))
               if node.node_id in node_sentence_indices:
                   r = node_sentence_indices[node.node_id]
                   sentences = [sentence_list[i] for i in r]
                   embeddings_slice = sentence_embeddings[r]
               # Only process if we have sentences to chunk.
               if len(sentences) > 0:
                   # Detect boundaries using the sliding window semantic differences on pre-computed embeddings
                   analysis = self.boundary_detector.compute_window_bounds(sentences, embeddings_slice)
                   
                   # Segment sentences and calculate mean embeddings for each semantic block
                   blocks_with_embeddings = []
                   current_sentences = []
                   current_indices = []
                   
                   for i, sentence in enumerate(sentences):
                       current_sentences.append(sentence)
                       current_indices.append(i)
                       
                       if analysis[i]["is_boundary"]:
                           current_text = " ".join(current_sentences)
                           word_count = len(current_text.split())
                           if len(current_sentences) >= settings.min_sentences or word_count >= settings.min_words:
                               block_emb = None
                               if settings.chunk_embedding_strategy == "vector_math":
                                   block_emb = np.mean(embeddings_slice[current_indices], axis=0)
                                   norm = np.linalg.norm(block_emb)
                                   if norm > 0.0:
                                       block_emb = (block_emb / norm).tolist()
                               blocks_with_embeddings.append((current_text, block_emb))
                               current_sentences = []
                               current_indices = []
                               
                   if current_sentences:
                       current_text = " ".join(current_sentences)
                       block_emb = None
                       if settings.chunk_embedding_strategy == "vector_math":
                           block_emb = np.mean(embeddings_slice[current_indices], axis=0)
                           norm = np.linalg.norm(block_emb)
                           if norm > 0.0:
                               block_emb = (block_emb / norm).tolist()
                       blocks_with_embeddings.append((current_text, block_emb))
                   
                   # Iterate through each clustered block to check token size limits.
                   for block, block_emb in blocks_with_embeddings:
                       # Optimize/split blocks if they violate maximum token thresholds.
                       optimized_sub_texts = self.token_optimizer.optimize_block(block)
       
                       context_path = node.get_contextual_path()
                       context_prefix = " > ".join(context_path)
       
                       for sub_text in optimized_sub_texts:
                           # Prepend contextual heading path (e.g. H1 > H2) to help vector retrieval match correctly.
                           enriched_text = f"Context: {context_prefix}\nContent: {sub_text}" if context_path else sub_text
           
                           chunk_metadata = {
                               "source": source_name,
                               "node_type": node.node_type,
                               "structural_path": context_path,
                               "token_count": self.token_optimizer.count_tokens(sub_text),
                               **node.metadata
                           }
           
                           rendered = RenderedChunk(
                               chunk_id=f"chk_{uuid.uuid4().hex[:8]}",
                               parent_node_id=node.node_id,
                               text=enriched_text,
                               metadata=chunk_metadata,
                               embeddings=block_emb
                           )
                           chunk_accumulator.append(rendered)
           
           # --- Handle Grouped LIST Blocks ---
           elif node.node_type == NodeType.LIST:
               context_path = node.get_contextual_path()
               context_prefix = " > ".join(context_path)
               
               optimized_sub_texts = self.token_optimizer.optimize_block(node.text)
               for sub_text in optimized_sub_texts:
                   enriched_text = f"Context: {context_prefix}\nContent:\n{sub_text}" if context_path else sub_text
                   chunk_metadata = {
                       "source": source_name,
                       "node_type": NodeType.LIST,
                       "structural_path": context_path,
                       "token_count": self.token_optimizer.count_tokens(sub_text),
                       **node.metadata
                   }
                   rendered = RenderedChunk(
                       chunk_id=f"chk_{uuid.uuid4().hex[:8]}",
                       parent_node_id=node.node_id,
                       text=enriched_text,
                       metadata=chunk_metadata
                   )
                   chunk_accumulator.append(rendered)
   
           # --- Handle Structured TABLE Blocks ---
           elif node.node_type == NodeType.TABLE:
               context_path = node.get_contextual_path()
               context_prefix = " > ".join(context_path)
               enriched_text = f"Context: {context_prefix}\nTable Data:\n{node.text}" if context_path else node.text
               
               table_chunk = RenderedChunk(
                   chunk_id=f"chk_{uuid.uuid4().hex[:8]}",
                   parent_node_id=node.node_id,
                   text=enriched_text,
                   metadata={"source": source_name, "node_type": NodeType.TABLE, "structural_path": context_path, **node.metadata}
               )
               chunk_accumulator.append(table_chunk)
   
           # --- Traverse Down the AST ---
           for child in node.children:
               self._decompose_node(child, chunk_accumulator, source_name, sentence_list, node_sentence_indices, sentence_embeddings)