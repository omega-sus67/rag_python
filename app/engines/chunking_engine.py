import re
import uuid
from typing import List, Dict, Optional, Any

class NodeType:
    ROOT = "ROOT"
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    TABLE = "TABLE"
    LIST_ITEM = "LIST_ITEM"

class DocNode:
    def __init__(
    self, 
    node_type: NodeType, 
    text: str, 
    level: int = 0,
    metadata: Optional[Dict[str, Any]] = None
    ):
        self.node_id : str = str(uuid.uuid4())[:8]
        self.node_type : NodeType = node_type
        self.text : str = text
        self.level : int = level
        self.metadata : Dict[str, Any] = metadata if metadata is not None else {}

        self.parent : Optional[DocNode] = None
        self.children : List[DocNode] = []

    def add_child(self, child_node: 'DocNode') -> None:
        child_node.parent = self
        self.children.append(child_node)

    def get_contextual_path(self) -> List[str]:
        path : List[str] = []
        curr = self
        while curr is not None and curr.node_type != NodeType.ROOT:
            if curr.node_type == NodeType.HEADING:
                path.insert(0, curr.text)
            curr = curr.parent
        return path
    def __repr__(self) -> str:
        return f"DocNode(id={self.node_id}, type={self.node_type}, level={self.level}, children={len(self.children)})"

class markdownParser:
    def __init__(self):
        self.heading_regex = re.compile(r'^(#{1,6})\s+(.*)$')
        self.list_regex = re.compile(r'^(\s*)[-\*\+]\s+(.*)$')
        self.table_row_regex = re.compile(r'^\|.*\|$')
    
    def _flush_table(
    self,
    parent: DocNode,
    buffer: List[str],
    start_line: int
    ) -> None:
        table_text = "\n".join(buffer)
        table_node = DocNode(
            node_type=NodeType.TABLE,
            text=table_text,
            level=parent.level,
            metadata={"line_number": start_line, "row_count": len(buffer)}
        )
        parent.add_child(table_node)
        buffer.clear()

    def parse(self, text : str) -> DocNode:
        root = DocNode(node_type = NodeType.ROOT, text = "ROOT_TEXT", level = 0 , metadata=None)

        buffer : List[str] = []
        start_line : int = 0
        current_parent : DocNode = root
        in_table : bool = False

        lines = text.split("\n")

        for i , line in enumerate(lines):
            strip_line = line.strip()

            if not strip_line:
                if in_table:   
                    self._flush_table(current_parent, buffer, start_line)
                    in_table = False
                continue
            
            if self.table_row_regex.match(strip_line):
                if not in_table:
                    in_table = True
                    start_line = i
                buffer.append(line)
                continue
            elif in_table:
                self._flush_table(current_parent, buffer, start_line)
                in_table = False
            
            heading_match = self.heading_regex.match(strip_line)
            if heading_match:
                hashes, heading_text = heading_match.groups()
                level = len(hashes)

                heading_node = DocNode(
                    node_type=NodeType.HEADING,
                    text=heading_text,
                    level=level,
                    metadata={"line_number": i}
                )
                while current_parent.parent is not None and current_parent.level >= level:
                    current_parent = current_parent.parent
                current_parent.add_child(heading_node)
                current_parent = heading_node
                continue
            list_match = self.list_regex.match(strip_line)
            if list_match:
                indent, list_item_text = list_match.groups()
                listNode = DocNode(
                    node_type = NodeType.LIST_ITEM,
                    text = list_item_text,
                    level = current_parent.level,
                    metadata = {"line_number": i, "indentation" : len(indent)}
                )
                current_parent.add_child(listNode)
                continue

            paragraph_node = DocNode(
                node_type=NodeType.PARAGRAPH,
                text=strip_line,
                level=current_parent.level,
                metadata={"line_number": i}
            )
            current_parent.add_child(paragraph_node)

        if in_table:
            self._flush_table(current_parent, buffer, start_line)
            
        return root
                
                
                
                
                
                    
        