# tests/test_markdown_parser.py

from app.engines.markdown_parser import DocNode, NodeType, markdownParser

def test_doc_node_initialization_and_hierarchy():
    """
    PURPOSE: Verifies that DocNode initializes correct defaults (unique IDs, empty structures)
    and correctly links bidirectionally when children are appended.
    CAPABILITIES:
    - Bidirectional parent-child linkage.
    - Path computation returns empty list for root node.
    - Path computation returns H1 heading text for single level children.
    """
    root = DocNode(node_type=NodeType.ROOT, text="ROOT")
    child = DocNode(node_type=NodeType.HEADING, text="Heading 1", level=1, metadata={"line_number": 0})
    
    root.add_child(child)
    
    assert child.parent == root
    assert child in root.children
    assert root.get_contextual_path() == []
    assert child.get_contextual_path() == ["Heading 1"]

def test_doc_node_contextual_path_nested():
    """
    PURPOSE: Verifies get_contextual_path properly climbs the parent hierarchy
    reconstructing structural breadcrumbs.
    CAPABILITIES:
    - Traverses grand-parent relationship.
    - Correctly accumulates heading path text chronologically (top-down).
    """
    root = DocNode(node_type=NodeType.ROOT, text="ROOT")
    h1 = DocNode(node_type=NodeType.HEADING, text="Heading 1", level=1)
    h2 = DocNode(node_type=NodeType.HEADING, text="Heading 2", level=2)
    p = DocNode(node_type=NodeType.PARAGRAPH, text="Some text")
    
    root.add_child(h1)
    h1.add_child(h2)
    h2.add_child(p)
    
    assert h1.get_contextual_path() == ["Heading 1"]
    assert h2.get_contextual_path() == ["Heading 1", "Heading 2"]
    assert p.get_contextual_path() == ["Heading 1", "Heading 2"]

def test_markdown_parser_headings():
    """
    PURPOSE: Verifies heading nesting rules under standard incrementing levels.
    CAPABILITIES:
    - Attaches H1 to root.
    - Attaches sub-headings to their immediate parent heading.
    """
    parser = markdownParser()
    text = (
        "# H1\n"
        "## H1.1\n"
        "# H2\n"
        "## H2.1\n"
        "### H2.1.1"
    )
    root = parser.parse(text)
    
    assert len(root.children) == 2
    h1, h2 = root.children
    assert h1.text == "H1"
    assert h1.level == 1
    
    assert h2.text == "H2"
    assert h2.level == 1
    
    assert len(h1.children) == 1
    h1_1 = h1.children[0]
    assert h1_1.text == "H1.1"
    assert h1_1.level == 2
    
    assert len(h2.children) == 1
    h2_1 = h2.children[0]
    assert h2_1.text == "H2.1"
    assert h2_1.level == 2
    
    assert len(h2_1.children) == 1
    h2_1_1 = h2_1.children[0]
    assert h2_1_1.text == "H2.1.1"
    assert h2_1_1.level == 3

def test_markdown_parser_paragraphs():
    """
    PURPOSE: Verifies paragraph accumulation logic.
    CAPABILITIES:
    - Merges multi-line paragraphs with spacing.
    - Segments paragraphs split by double newlines.
    """
    parser = markdownParser()
    text = (
        "This is line 1.\n"
        "This is line 2.\n"
        "\n"
        "This is paragraph 2."
    )
    root = parser.parse(text)
    
    assert len(root.children) == 2
    p1, p2 = root.children
    
    assert p1.node_type == NodeType.PARAGRAPH
    assert p1.text == "This is line 1. This is line 2."
    assert p1.metadata["line_number"] == 0
    
    assert p2.node_type == NodeType.PARAGRAPH
    assert p2.text == "This is paragraph 2."
    assert p2.metadata["line_number"] == 3

def test_markdown_parser_lists():
    """
    PURPOSE: Verifies consecutive list bullets collection.
    CAPABILITIES:
    - Aggregates adjacent lists (using -, *, + markers) into a single Node.
    - Retains original line spacings/indentations inside the list text.
    """
    parser = markdownParser()
    text = (
        "- Item 1\n"
        "  - Nested Item 1.1\n"
        "* Item 2\n"
        "    + Nested Item 2.1"
    )
    root = parser.parse(text)
    
    assert len(root.children) == 1
    list_node = root.children[0]
    assert list_node.node_type == NodeType.LIST
    assert list_node.text == text
    assert list_node.metadata["line_number"] == 0

def test_markdown_parser_tables():
    """
    PURPOSE: Verifies markdown table extraction and metadata collection.
    CAPABILITIES:
    - Identifies table rows start and end characters (|).
    - Measures row counts in metadata.
    """
    parser = markdownParser()
    text = (
        "| Header 1 | Header 2 |\n"
        "| -------- | -------- |\n"
        "| Value 1  | Value 2  |"
    )
    root = parser.parse(text)
    
    assert len(root.children) == 1
    table = root.children[0]
    assert table.node_type == NodeType.TABLE
    assert table.text == text
    assert table.metadata["line_number"] == 0
    assert table.metadata["row_count"] == 3

def test_markdown_parser_empty_or_whitespace():
    """
    PURPOSE: Verifies parser behavior on empty string or whitespace-only layouts.
    CAPABILITIES:
    - Empty inputs yield no AST child nodes.
    - Whitespace streams are discarded and do not create empty paragraphs.
    """
    parser = markdownParser()
    root = parser.parse("   \n\n   ")
    assert len(root.children) == 0

def test_markdown_parser_mixed_document():
    """
    PURPOSE: Verifies full system AST builder on general documents.
    CAPABILITIES:
    - Resolves mixed sequences of headings, paragraphs, lists, and tables.
    - Maintains correct nesting structure under active headers.
    """
    parser = markdownParser()
    text = (
        "# Document Title\n"
        "Introductory paragraph here.\n"
        "\n"
        "## Sub-section\n"
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "\n"
        "Next paragraph after table.\n"
        "- List element 1\n"
        "- List element 2\n"
    )
    root = parser.parse(text)
    
    assert len(root.children) == 1
    h1 = root.children[0]
    assert h1.text == "Document Title"
    assert h1.node_type == NodeType.HEADING
    
    assert len(h1.children) == 2
    intro_para, h2 = h1.children
    assert intro_para.node_type == NodeType.PARAGRAPH
    assert intro_para.text == "Introductory paragraph here."
    
    assert h2.node_type == NodeType.HEADING
    assert h2.text == "Sub-section"
    
    assert len(h2.children) == 3
    table, post_table_para, list_node = h2.children
    
    assert table.node_type == NodeType.TABLE
    assert post_table_para.node_type == NodeType.PARAGRAPH
    assert post_table_para.text == "Next paragraph after table."
    
    assert list_node.node_type == NodeType.LIST
    assert list_node.text == "- List element 1\n- List element 2"

# --- RIGOROUS EXTENDED TESTS ---

def test_markdown_parser_backtracking_level_skip():
    """
    PURPOSE: Tests backtrack parent selection when skipping heading levels or going back to root.
    CAPABILITIES:
    - Skipping H1 -> H3 correctly handles missing intermediate level.
    - Re-ascending H3 -> H2 jumps back up to H1 as parent.
    - Returning to subsequent H1 anchors directly to AST root.
    """
    parser = markdownParser()
    text = (
        "# H1\n"
        "### H3\n"
        "## H2\n"
        "# H1-Sibling"
    )
    root = parser.parse(text)
    
    assert len(root.children) == 2
    h1_first, h1_second = root.children
    assert h1_first.text == "H1"
    assert h1_second.text == "H1-Sibling"
    
    # H1 should have H3 nested under it (skipped H2)
    assert len(h1_first.children) == 2
    h3_node = h1_first.children[0]
    h2_node = h1_first.children[1]
    
    assert h3_node.text == "H3"
    assert h3_node.level == 3
    
    assert h2_node.text == "H2"
    assert h2_node.level == 2

def test_markdown_parser_malformed_tables():
    """
    PURPOSE: Verifies parser resilience when handling malformed tables.
    CAPABILITIES:
    - Ignores spaces and unclosed pipes if they violate row regex.
    - Closes table blocks correctly when encountering non-table text.
    """
    parser = markdownParser()
    text = (
        "| Valid Table Header |\n"
        "|--------------------|\n"
        "| Valid Table Cell |\n"
        "Invalid line breaking the table\n"
        "| Another single cell |"
    )
    root = parser.parse(text)
    
    # We expect:
    # 1. TABLE block (first 3 rows)
    # 2. PARAGRAPH block (the breaking line)
    # 3. TABLE block (the last row)
    assert len(root.children) == 3
    t1, p, t2 = root.children
    
    assert t1.node_type == NodeType.TABLE
    assert t1.metadata["row_count"] == 3
    
    assert p.node_type == NodeType.PARAGRAPH
    assert p.text == "Invalid line breaking the table"
    
    assert t2.node_type == NodeType.TABLE
    assert t2.metadata["row_count"] == 1

def test_markdown_parser_separated_lists():
    """
    PURPOSE: Verifies list segmentation rules.
    CAPABILITIES:
    - Consecutive list lines form a single LIST node.
    - Two list blocks separated by a blank line form two separate LIST nodes.
    """
    parser = markdownParser()
    text = (
        "- List 1 - Item A\n"
        "- List 1 - Item B\n"
        "\n"
        "- List 2 - Item A\n"
        "- List 2 - Item B"
    )
    root = parser.parse(text)
    
    assert len(root.children) == 2
    l1, l2 = root.children
    
    assert l1.node_type == NodeType.LIST
    assert l1.text == "- List 1 - Item A\n- List 1 - Item B"
    
    assert l2.node_type == NodeType.LIST
    assert l2.text == "- List 2 - Item A\n- List 2 - Item B"

def test_markdown_parser_extreme_spacing():
    """
    PURPOSE: Verifies parser cleaning mechanics on text files containing excessive spaces.
    CAPABILITIES:
    - Purges trailing spacing while maintaining multi-line paragraph joins.
    - Discards empty rows without breaking heading parenting context.
    """
    parser = markdownParser()
    text = (
        "  \t  \n"
        "# Head \t \n"
        "Paragraph text with trailing space.    \n"
        "Next line of paragraph.\t\t\n"
        "\n\n\n"
        "Second paragraph."
    )
    root = parser.parse(text)
    
    assert len(root.children) == 1
    h = root.children[0]
    assert h.text == "Head"
    
    assert len(h.children) == 2
    p1, p2 = h.children
    
    assert p1.node_type == NodeType.PARAGRAPH
    assert p1.text == "Paragraph text with trailing space. Next line of paragraph."
    assert p2.text == "Second paragraph."
