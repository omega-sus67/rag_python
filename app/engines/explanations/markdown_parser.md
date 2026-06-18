# Markdown Parser – Detailed Explanation

The `markdown_parser.py` file implements a lightweight **Markdown‑to‑AST** parser that turns a plain‑text Markdown document into a tree of `DocNode` objects.  This structure is later consumed by higher‑level components (e.g., the hierarchical semantic engine) to provide context for embeddings and retrieval.

---

## 1. Core Data Structures

### `NodeType`
```python
class NodeType:
    ROOT = "ROOT"
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    TABLE = "TABLE"
    LIST_ITEM = "LIST_ITEM"
    LIST = "LIST"
```
A simple enum‑like container that identifies the kind of node we are dealing with.  It is used throughout the parser and later stages to decide how to treat a node (e.g., headings become hierarchical context, tables are stored as raw text, etc.).

### `DocNode`
```python
class DocNode:
    def __init__(self, node_type: NodeType, text: str, level: int = 0,
                 metadata: Optional[Dict[str, Any]] = None):
        self.node_id: str = str(uuid.uuid4())[:8]
        self.node_type: NodeType = node_type
        self.text: str = text
        self.level: int = level
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self.parent: Optional[DocNode] = None
        self.children: List[DocNode] = []
```
* **Purpose** – Represents a single element of the document (heading, paragraph, table, list item, or the artificial root).  Each node carries:
  * `node_type` – what kind of element it is.
  * `text` – the raw content of that element.
  * `level` – the hierarchical depth (used for headings and to keep list items at the same level as their parent).  For non‑hierarchical nodes this stays at the parent’s level.
  * `metadata` – optional auxiliary information such as the original line number or indentation depth.
* **Relationships** – Nodes form a **tree** via `parent` and `children`.  The helper methods below manipulate this tree.

#### `add_child`
```python
def add_child(self, child_node: 'DocNode') -> None:
    child_node.parent = self
    self.children.append(child_node)
```
Links a child node to its parent and stores it in the parent’s `children` list.  This is the only place we set `parent`, guaranteeing a single source of truth for the tree.

#### `get_contextual_path`
```python
def get_contextual_path(self) -> List[str]:
    path: List[str] = []
    curr = self
    while curr is not None and curr.node_type != NodeType.ROOT:
        if curr.node_type == NodeType.HEADING:
            path.insert(0, curr.text)
        curr = curr.parent
    return path
```
Returns a list of heading texts that lead from the root to the node.  This is crucial for the **HierarchicalSemanticEngine** because it prefixes each chunk with the breadcrumb trail, giving the LLM additional context.

#### `__repr__`
A convenience for debugging – prints a compact summary of the node.

---

## 2. `markdownParser` – The actual parser

```python
class markdownParser:
    def __init__(self):
        self.heading_regex = re.compile(r'^(#{1,6})\s+(.*)$')
        self.list_regex = re.compile(r'^(\s*)[-\*\+]\s+(.*)$')
        self.table_row_regex = re.compile(r'^\|.*\|$')
```
* **Regexes** – Pre‑compiled regular expressions for the three structural constructs we care about:
  * **Headings** – `#{1,6}` captures the leading `#` characters; the number of `#` determines the heading level.
  * **List items** – Captures optional leading whitespace (`indent`) and the actual list text.  The whitespace length determines nesting depth.
  * **Table rows** – Very simple detection of Markdown table rows (start and end with `|`).  We treat a block of consecutive rows as a single table node.

### 2.1 Private helper: `_flush_table`
```python
def _flush_table(self, parent: DocNode, buffer: List[str], start_line: int) -> None:
    if not buffer:
        return
    table_text = "\n".join(buffer)
    table_node = DocNode(
        node_type=NodeType.TABLE,
        text=table_text,
        level=parent.level,
        metadata={"line_number": start_line, "row_count": len(buffer)}
    )
    parent.add_child(table_node)
    buffer.clear()
```
* **When called** – After a sequence of table rows ends (either because a blank line or another structure appears).
* **What it does** – Joins the collected rows into a single string, creates a `TABLE` node with the same hierarchical level as its parent, stores the starting line number and row count for possible debugging, attaches it to the tree, and empties the buffer for reuse.

### 2.2 Private helper: `_flush_paragraph`
```python
def _flush_paragraph(self, parent: DocNode, buffer: List[str], start_line: int) -> None:
    if not buffer:
        return
    paragraph_text = " ".join(buffer)
    paragraph_node = DocNode(
        node_type=NodeType.PARAGRAPH,
        text=paragraph_text,
        level=parent.level,
        metadata={"line_number": start_line}
    )
    parent.add_child(paragraph_node)
    buffer.clear()
```
* **When called** – When a non‑paragraph construct appears (heading, list, table) or when we encounter a blank line signalling the end of the current paragraph.
* **What it does** – Joins the buffered stripped lines with spaces (preserving natural sentence spacing), creates a `PARAGRAPH` node, adds it to the tree, and clears the buffer.

### 2.3 Private helper: `_flush_list`
```python
def _flush_list(self, parent: DocNode, buffer: List[str], start_line: int) -> None:
    if not buffer:
        return
    list_text = "\n".join(buffer)
    list_node = DocNode(
        node_type=NodeType.LIST,
        text=list_text,
        level=parent.level,
        metadata={"line_number": start_line}
    )
    parent.add_child(list_node)
    buffer.clear()
```
* **When called** – When consecutive list item rows end because a blank line or non-list construct is encountered.
* **What it does** – Joins the accumulated raw list item lines with newlines, preserving the bullet formats and indentation structure, constructs a `LIST` node, attaches it to the parent, and clears the buffer.

### 2.4 The public `parse` method – orchestrator
```python
def parse(self, text: str) -> DocNode:
    root = DocNode(node_type=NodeType.ROOT, text="ROOT_TEXT", level=0, metadata=None)
    buffer: List[str] = []                # generic buffer for table rows
    para_buffer: List[str] = []           # buffer for paragraph lines
    list_buffer: List[str] = []           # buffer for list lines
    start_line: int = 0
    para_start_line: int = 0
    list_start_line: int = 0
    current_parent: DocNode = root
    in_table: bool = False
    in_list: bool = False
    lines = text.split("\n")
    for i, line in enumerate(lines):
        strip_line = line.strip()
        # (1) Blank line handling – terminates any open block
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
        # (2) Table detection – consecutive rows starting with '|'
        if self.table_row_regex.match(strip_line):
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
            self._flush_table(current_parent, buffer, start_line)
            in_table = False
        # (3) Heading detection
        heading_match = self.heading_regex.match(strip_line)
        if heading_match:
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
            while current_parent.parent is not None and current_parent.level >= level:
                current_parent = current_parent.parent
            current_parent.add_child(heading_node)
            current_parent = heading_node
            continue
        # (4) List detection – uses the **original line** (not stripped) to retain indentation
        list_match = self.list_regex.match(line)
        if list_match:
            if para_buffer:
                self._flush_paragraph(current_parent, para_buffer, para_start_line)
            if not in_list:
                in_list = True
                list_start_line = i
            list_buffer.append(line)
            continue
        elif in_list:
            self._flush_list(current_parent, list_buffer, list_start_line)
            in_list = False
        # (5) Anything else is part of a paragraph
        if not para_buffer:
            para_start_line = i
        para_buffer.append(strip_line)
    # End of file – flush any lingering buffers
    if in_table:
        self._flush_table(current_parent, buffer, start_line)
    if in_list:
        self._flush_list(current_parent, list_buffer, list_start_line)
    if para_buffer:
        self._flush_paragraph(current_parent, para_buffer, para_start_line)
    return root
```
#### High‑level flow
1. **Initialize the root node** – This node never appears in the final output but gives us a stable anchor for the tree.
2. **Iterate line‑by‑line** – `enumerate(lines)` provides the line number (`i`). The parser maintains three buffers:
   * `buffer` → accumulates raw table rows.
   * `para_buffer` → accumulates stripped paragraph lines.
   * `list_buffer` → accumulates raw list item lines.
3. **Blank line handling** – A blank line ends any active *paragraph*, *list*, or *table* block. The parser flushes the appropriate buffer and resets state.
4. **Table detection** – If a line matches `table_row_regex`, we treat it as part of a table. Before starting a table we ensure any ongoing paragraph/list is flushed, then we start/continue the table buffer.
5. **Heading detection** – When a line matches the heading pattern, we create a `HEADING` node. The while‑loop walks **up** the tree until we locate the correct parent according to heading level (`current_parent.level >= level`). This reproduces the hierarchical structure you would expect from nested headings. Any active paragraph/list is flushed first.
6. **List detection** – List items are parsed using the **original line** (preserving indentation). Instead of creating individual list item nodes, consecutive list items are accumulated into `list_buffer` and flushed together as a single `NodeType.LIST` block node. This prevents list fracturing and keeps the entire list semantic structure intact.
7. **Paragraph accumulation** – Anything that is not a table, heading, or list is considered a normal paragraph line. Its stripped version is stored in `para_buffer`. Any active list is flushed.
8. **End‑of‑file cleanup** – After the loop finishes we flush any table, list, or paragraph that might still be open.
9. **Return the root** – The calling code walks the root’s children to extract chunks for downstream processing.

---

## 3. Why the Design Choices?

| Design Element | Reason / Benefits |
|----------------|-------------------|
| **AST‑style `DocNode` tree** | Keeps the document’s logical hierarchy explicit (headings → sub‑headings). Later stages can easily compute contextual paths, which dramatically improves retrieval relevance. |
| **Separate buffers for tables & paragraphs** | Tables have a very different semantic role (structured data) and should not be merged with surrounding prose. Using a dedicated buffer guarantees we treat an entire table as a single node.
| **Grouped list blocks (`NodeType.LIST`)** | Lists convey complex context and hierarchical relationships (nested bullets). Grouping consecutive items into a single node prevents semantic fracturing, ensuring list items are not separated from their introduction sentence or neighboring steps during retrieval.
| **Flushing on blank lines** | A blank line in Markdown is the canonical delimiter between paragraphs, tables, headings, etc. Flushing ensures we close the previous block exactly where the author intended.
| **Walking up the tree for heading levels** | Markdown allows you to jump from `##` back to `#` and then to `###`. The while‑loop correctly re‑attaches new headings under the most recent appropriate ancestor.
| **Storing line numbers** (metadata) | Helpful for debugging, error reporting, and any UI that needs to highlight the original source location.

---

## 4. Interaction with the Rest of the System
* **`HierarchicalSemanticEngine`** consumes the AST returned by `markdownParser.parse`. For each leaf node (paragraph, table, list block) it calls `node.get_contextual_path()` to prepend the heading breadcrumb, creating a `RenderedChunk`.
* **List Chunking Safety** – `NodeType.LIST` chunks are processed as a single block to preserve their bullet formatting (indentations, newlines). If a list is extremely long, it is split safely using the overlapping token-level optimizer to avoid exceeding LLM context window limits.
* **Embedding / Retrieval** – The contextual path dramatically improves semantic similarity because the vector representation now contains both the content and its surrounding section titles.
* **Testing** – The test suite (`tests/test_markdown_parser.py`) verifies that each helper correctly creates nodes, heading hierarchy is respected, list blocks group consecutive items properly, and paragraph and table flushing works as expected.

---

## 5. Summary of the Changes
To fix the semantic hazard of list fracturing, the chunking parser has been updated to accumulate all consecutive list items into a single unified `NodeType.LIST` block. This ensures that the logical relation between consecutive items, as well as nested indentation, remains fully intact inside a single vector chunk, rather than being chopped up into separate sentences/items.

---

*File created at:* `app/engines/explanations/markdown_parser.md`.

---

*End of explanation.*
