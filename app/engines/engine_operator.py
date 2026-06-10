import uuid
from typing import List, Dict, Any
from app.engines.semantic_engine import SemanticEngine, SlidingSemanticChunker
from app.engines.chunking_engine import markdownParser, DocNode, NodeType
from app.utils.token_optimizer import TokenSizeOptimizer

from app.core.config import settings

class RenderedChunk:
    def __init__(self, chunk_id: str, parent_node_id: str, text: str, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.parent_node_id = parent_node_id
        self.text = text
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"<RenderedChunk id={self.chunk_id} parent={self.parent_node_id} text_len={len(self.text)}>"


class HierarchicalSemanticEngine:
    def __init__(self, vector_engine: SemanticEngine, window_size: int = settings.window_size, threshold_factor: float = settings.threshold_factor):
        self.structure_parser = markdownParser()
        self.boundary_detector = SlidingSemanticChunker(
            vector_engine=vector_engine, 
            window_size=window_size, 
            threshold_factor=threshold_factor
        )
        self.token_optimizer = TokenSizeOptimizer()

    def process_document(self, raw_markdown_text: str, source_name: str = "unknown") -> List[RenderedChunk]:
        root_node = self.structure_parser.parse(raw_markdown_text)
        
        final_chunks: List[RenderedChunk] = []
        self._decompose_node(root_node, final_chunks, source_name)
        
        return final_chunks

    def _decompose_node(self, node: DocNode, chunk_accumulator: List[RenderedChunk], source_name: str) -> None:
        if node.node_type in [NodeType.PARAGRAPH, NodeType.LIST_ITEM]:
            sentences = self.boundary_detector.split_into_sentences(node.text)
            
            if len(sentences) > 0:
                analysis = self.boundary_detector.compute_window_bounds(sentences)
                semantic_blocks = self.boundary_detector.generate_chunks(sentences, analysis)

                for block in semantic_blocks:
                    optimized_sub_texts = self.token_optimizer.optimize_block(block)
    
                    context_path = node.get_contextual_path()
                    context_prefix = " > ".join(context_path)
    
                    for sub_text in optimized_sub_texts:
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

        for child in node.children:
            self._decompose_node(child, chunk_accumulator, source_name)