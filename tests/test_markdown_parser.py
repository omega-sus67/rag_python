# tests/test_markdown_parser.py

from app.engines.markdown_parser import DocNode, NodeType, markdownParser

def test_doc_node_initialization_and_hierarchy():
    root = DocNode(node_type=NodeType.ROOT, text="ROOT")
    child = DocNode(node_type=NodeType.HEADING, text="Heading 1", level=1, metadata={"line_number": 0})
    
    root.add_child(child)
    
    assert child.parent == root
    assert child in root.children
    assert root.get_contextual_path() == []
    assert child.get_contextual_path() == ["Heading 1"]

def test_doc_node_contextual_path_nested():
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
    parser = markdownParser()
    text = (
        "# H1\n"
        "## H1.1\n"
        "# H2\n"
        "## H2.1\n"
        "### H2.1.1"
    )
    root = parser.parse(text)
    
    # Check H1 and H2 are children of root
    assert len(root.children) == 2
    h1, h2 = root.children
    assert h1.text == "H1"
    assert h1.level == 1
    
    assert h2.text == "H2"
    assert h2.level == 1
    
    # H1 should have H1.1 as child
    assert len(h1.children) == 1
    h1_1 = h1.children[0]
    assert h1_1.text == "H1.1"
    assert h1_1.level == 2
    
    # H2 should have H2.1
    assert len(h2.children) == 1
    h2_1 = h2.children[0]
    assert h2_1.text == "H2.1"
    assert h2_1.level == 2
    
    # H2.1 should have H2.1.1
    assert len(h2_1.children) == 1
    h2_1_1 = h2_1.children[0]
    assert h2_1_1.text == "H2.1.1"
    assert h2_1_1.level == 3

def test_markdown_parser_paragraphs():
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
    parser = markdownParser()
    root = parser.parse("   \n\n   ")
    assert len(root.children) == 0

def test_markdown_parser_mixed_document():
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
    
    # Structure verification
    assert len(root.children) == 1
    h1 = root.children[0]
    assert h1.text == "Document Title"
    assert h1.node_type == NodeType.HEADING
    
    # Children of h1: paragraph, h2
    assert len(h1.children) == 2
    intro_para, h2 = h1.children
    assert intro_para.node_type == NodeType.PARAGRAPH
    assert intro_para.text == "Introductory paragraph here."
    
    assert h2.node_type == NodeType.HEADING
    assert h2.text == "Sub-section"
    
    # Children of h2: table, paragraph, list
    assert len(h2.children) == 3
    table, post_table_para, list_node = h2.children
    
    assert table.node_type == NodeType.TABLE
    assert post_table_para.node_type == NodeType.PARAGRAPH
    assert post_table_para.text == "Next paragraph after table."
    
    assert list_node.node_type == NodeType.LIST
    assert list_node.text == "- List element 1\n- List element 2"
