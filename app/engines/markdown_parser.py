import re
import uuid
from typing import List, Dict, Optional, Any

class NodeType:
    """
    Defines the structural types of nodes in the Document AST.
    Using string constants for simplicity and ease of serialization/rendering.
    """
    ROOT = "ROOT"
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    TABLE = "TABLE"
    LIST_ITEM = "LIST_ITEM"  # Kept for compatibility, though list elements are grouped under NodeType.LIST
    LIST = "LIST"

class DocNode:
    """
    Represents a single node in the document's Abstract Syntax Tree (AST).
    Maintains a parent-child relationship tree to allow path reconstruction.
    """
    def __init__(
        self, 
        node_type: NodeType, 
        text: str, 
        level: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        # Generate an 8-character unique identifier for tracking/debugging.
        self.node_id : str = str(uuid.uuid4())[:8]
        self.node_type : NodeType = node_type
        # Contains raw text content for the node (e.g., table cells, paragraph body, heading title).
        self.text : str = text
        # Level tracks structural depth. Crucial for headings (e.g., level 1 for #, level 2 for ##).
        self.level : int = level
        self.metadata : Dict[str, Any] = metadata if metadata is not None else {}

        # Traversal pointers to navigate the hierarchical tree.
        self.parent : Optional[DocNode] = None
        self.children : List[DocNode] = []

    def add_child(self, child_node: 'DocNode') -> None:
        """
        Links a child node to this parent node.
        Updates parent reference on child to keep the bidirectional relationship valid.
        """
        child_node.parent = self
        self.children.append(child_node)

    def get_contextual_path(self) -> List[str]:
        """
        Traverses upwards from the current node to the root, collecting all heading text.
        This provides the hierarchical contextual 'path' (breadcrumbs) of the chunk.
        """
        path : List[str] = []
        curr = self
        # Loop ascends the parent tree, ignoring the root node text which contains placeholder info.
        while curr is not None and curr.node_type != NodeType.ROOT:
            if curr.node_type == NodeType.HEADING:
                # Insert at the beginning of the list to preserve top-down order (H1 > H2 > H3).
                path.insert(0, curr.text)
            curr = curr.parent
        return path

    def __repr__(self) -> str:
        return f"DocNode(id={self.node_id}, type={self.node_type}, level={self.level}, children={len(self.children)})"

class markdownParser:
    """
    A line-by-line Markdown parser that constructs a hierarchical AST.
    Uses regex matching to detect block structures (headings, tables, lists).
    """
    def __init__(self):
        # Matches markdown headers (1 to 6 hashes).
        self.heading_regex = re.compile(r'^(#{1,6})\s+(.*)$')
        # Matches list item bullet syntax (handles -, *, + with optional indentation).
        self.list_regex = re.compile(r'^(\s*)[-\*\+]\s+(.*)$')
        # Matches simple markdown table rows that start and end with a pipe.
        self.table_row_regex = re.compile(r'^\|.*\|$')
    
    def _flush_table(
        self,
        parent: DocNode,
        buffer: List[str],
        start_line: int
    ) -> None:
        """
        Joins buffered table rows, builds a TABLE DocNode, and hooks it to the active parent.
        Clears the buffer to prepare for subsequent structures.
        """
        if not buffer:
            return
        # Table rows are joined with newlines to preserve tabular format for downstream parsing.
        table_text = "\n".join(buffer)
        table_node = DocNode(
            node_type=NodeType.TABLE,
            text=table_text,
            level=parent.level,
            metadata={"line_number": start_line, "row_count": len(buffer)}
        )
        parent.add_child(table_node)
        buffer.clear()

    def _flush_paragraph(
        self,
        parent: DocNode,
        buffer: List[str],
        start_line: int
    ) -> None:
        """
        Merges buffered paragraph lines into a single space-separated prose block,
        creates a PARAGRAPH DocNode, attaches it to the parent, and clears the buffer.
        """
        if not buffer:
            return
        # Join with space to reconstitute standard multi-line paragraph text into a single cohesive string.
        paragraph_text = " ".join(buffer)
        paragraph_node = DocNode(
            node_type=NodeType.PARAGRAPH,
            text=paragraph_text,
            level=parent.level,
            metadata={"line_number": start_line}
        )
        parent.add_child(paragraph_node)
        buffer.clear()

    def _flush_list(
        self,
        parent: DocNode,
        buffer: List[str],
        start_line: int
    ) -> None:
        """
        Groups accumulated consecutive list items into a single LIST block node,
        preserving newlines and indentation, preventing list fracturing.
        """
        if not buffer:
            return
        # Join with newlines to preserve bullets and nested hierarchical structure.
        list_text = "\n".join(buffer)
        list_node = DocNode(
            node_type=NodeType.LIST,
            text=list_text,
            level=parent.level,
            metadata={"line_number": start_line}
        )
        parent.add_child(list_node)
        buffer.clear()

    def parse(self, text : str) -> DocNode:
        """
        Orchestrates AST generation by parsing input text line-by-line.
        Tracks open block states (table, list, paragraph) using state flags.
        """
        root = DocNode(node_type = NodeType.ROOT, text = "ROOT_TEXT", level = 0 , metadata=None)

        # Buffers for collecting multi-line structures.
        buffer : List[str] = []       # For table lines
        para_buffer : List[str] = []  # For normal paragraph text lines
        list_buffer : List[str] = []  # For grouping consecutive list items
        
        # Keep track of original line numbers for metadata tracing.
        start_line : int = 0
        para_start_line : int = 0
        list_start_line : int = 0
        
        current_parent : DocNode = root
        
        # State flags to track active block collection.
        in_table : bool = False
        in_list : bool = False

        lines = text.split("\n")

        for i , line in enumerate(lines):
            strip_line = line.strip()

            # Empty lines signify structural boundaries and force flushing of all active buffers.
            if not strip_line:
                if in_table:   
                    self._flush_table(current_parent, buffer, start_line)
                    in_table = False
                elif in_list:
                    self._flush_list(current_parent, list_buffer, list_start_line)
                    in_list = False
                else:
                    self._flush_paragraph(current_parent, para_buffer, para_start_line)
                continue
            
            # --- Table Handling ---
            if self.table_row_regex.match(strip_line):
                # If we encounter a table, flush any active paragraph or list first.
                if para_buffer:
                    self._flush_paragraph(current_parent, para_buffer, para_start_line)
                if in_list:
                    self._flush_list(current_parent, list_buffer, list_start_line)
                    in_list = False
                if not in_table:
                    in_table = True
                    start_line = i
                buffer.append(line)
                continue
            elif in_table:
                # If the line does not match the table regex, but we were in a table block, flush it.
                self._flush_table(current_parent, buffer, start_line)
                in_table = False
            
            # --- Heading Handling ---
            heading_match = self.heading_regex.match(strip_line)
            if heading_match:
                # Headings interrupt any current paragraph or list; flush them first.
                if para_buffer:
                    self._flush_paragraph(current_parent, para_buffer, para_start_line)
                if in_list:
                    self._flush_list(current_parent, list_buffer, list_start_line)
                    in_list = False
                
                hashes, heading_text = heading_match.groups()
                level = len(hashes)

                heading_node = DocNode(
                    node_type=NodeType.HEADING,
                    text=heading_text,
                    level=level,
                    metadata={"line_number": i}
                )
                
                # Backtrack up the AST tree to locate the correct parent node.
                # If the current parent has level >= new heading level, it means we have finished
                # the current sub-section and are moving back up to a broader section or sibling.
                while current_parent.parent is not None and current_parent.level >= level:
                    current_parent = current_parent.parent
                
                current_parent.add_child(heading_node)
                current_parent = heading_node  # Set new heading as parent for subsequent children.
                continue
            
            # --- List Handling ---
            # Use original line (not stripped) to match list regex, as indentation (leading spaces) is crucial.
            list_match = self.list_regex.match(line)
            if list_match:
                # List items interrupt regular paragraphs.
                if para_buffer:
                    self._flush_paragraph(current_parent, para_buffer, para_start_line)
                if not in_list:
                    in_list = True
                    list_start_line = i
                list_buffer.append(line)
                continue
            elif in_list:
                # If we encounter a line that isn't a list item, flush the list.
                self._flush_list(current_parent, list_buffer, list_start_line)
                in_list = False

            # --- Paragraph Fallback ---
            # If line doesn't match any block patterns, it is treated as a paragraph line.
            if not para_buffer:
                para_start_line = i
            para_buffer.append(strip_line)

        # --- Cleanup at EOF ---
        # Ensure any open buffers are flushed when we reach the end of the text.
        if in_table:
            self._flush_table(current_parent, buffer, start_line)
        if in_list:
            self._flush_list(current_parent, list_buffer, list_start_line)
        if para_buffer:
            self._flush_paragraph(current_parent, para_buffer, para_start_line)
            
        return root