import uuid
from typing import List, Dict, Any
from app.engines.semantic_chunker import SemanticEngine, SlidingSemanticChunker
from app.engines.markdown_parser import markdownParser, DocNode, NodeType
from app.utils.token_optimizer import TokenSizeOptimizer

from app.core.config import settings

class RenderedChunk:
    """
    A unified data transfer object (DTO) representing a completed RAG chunk.
    Enriched with heading context path and metadata fields ready for database insertion.
    """
    def __init__(self, chunk_id: str, parent_node_id: str, text: str, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        # Reference back to the original AST DocNode.
        self.parent_node_id = parent_node_id
        # Contains context path prepended to the actual text content.
        self.text = text
        self.metadata = metadata

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

    def process_document(self, raw_markdown_text: str, source_name: str = "unknown") -> List[RenderedChunk]:
        """
        Orchestrates document ingestion:
        1. Parses markdown text into a structural node tree (AST).
        2. Traverses and decomposes the AST recursively into semantic chunks.
        """
        root_node = self.structure_parser.parse(raw_markdown_text)
        
        final_chunks: List[RenderedChunk] = []
        # Begin recursive decomposition from the Root node.
        self._decompose_node(root_node, final_chunks, source_name)
        
        return final_chunks

    def _decompose_node(self, node: DocNode, chunk_accumulator: List[RenderedChunk], source_name: str) -> None:
        """
        Recursively decomposes AST tree nodes into rendered chunks based on NodeType.
        Handles text content, tables, and lists differently to optimize context preservation.
        """
        # --- Handle Paragraphs and Lists Items ---
        if node.node_type in [NodeType.PARAGRAPH, NodeType.LIST_ITEM]:
            sentences = self.boundary_detector.split_into_sentences(node.text)
            
            # Only process if we have sentences to chunk.
            if len(sentences) > 0:
                # Detect boundaries using the sliding window semantic differences.
                analysis = self.boundary_detector.compute_window_bounds(sentences)
                semantic_blocks = self.boundary_detector.generate_chunks(sentences, analysis)

                # Iterate through each clustered block to check token size limits.
                for block in semantic_blocks:
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
                            metadata=chunk_metadata
                        )
                        chunk_accumulator.append(rendered)
        
        # --- Handle Grouped LIST Blocks ---
        elif node.node_type == NodeType.LIST:
            context_path = node.get_contextual_path()
            context_prefix = " > ".join(context_path)
            
            # Pass the entire list block directly to the Token Optimizer.
            # We treat lists as atomic context units. Only split them if they exceed absolute limits.
            optimized_sub_texts = self.token_optimizer.optimize_block(node.text)
            for sub_text in optimized_sub_texts:
                # Prepend heading path context to maintain logical association.
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
            # Tables represent structured data and are never semantic-split; kept completely intact.
            enriched_text = f"Context: {context_prefix}\nTable Data:\n{node.text}" if context_path else node.text
            
            table_chunk = RenderedChunk(
                chunk_id=f"chk_{uuid.uuid4().hex[:8]}",
                parent_node_id=node.node_id,
                text=enriched_text,
                metadata={"source": source_name, "node_type": NodeType.TABLE, "structural_path": context_path, **node.metadata}
            )
            chunk_accumulator.append(table_chunk)

        # --- Traverse Down the AST ---
        # Recursively call child nodes to process the whole document tree.
        for child in node.children:
            self._decompose_node(child, chunk_accumulator, source_name)