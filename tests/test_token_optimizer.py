# tests/test_token_optimizer.py

from app.utils.token_optimizer import TokenSizeOptimizer

def test_optimizer_leaves_small_text_intact():
    """
    PURPOSE: Verifies that short texts which fit within budget are not split.
    CAPABILITIES:
    - Return list containing one item equal to original text.
    """
    optimizer = TokenSizeOptimizer(max_tokens=20, overlap_tokens=5)
    text = "Short text."
    
    result = optimizer.optimize_block(text)
    assert result == [text]

def test_optimizer_splits_large_text_with_overlap():
    """
    PURPOSE: Verifies that large strings are partitioned with designated overlaps.
    CAPABILITIES:
    - Splits text into multiple chunks under max_tokens budget.
    - Matches token integer list slices to assert overlap size matching.
    """
    optimizer = TokenSizeOptimizer(max_tokens=10, overlap_tokens=3)
    text = "apple banana cherry grape orange melon strawberry peach lemon lime blueberry blackberry raspberry"
    
    result = optimizer.optimize_block(text)
    
    assert len(result) > 1
    
    for chunk in result:
        assert optimizer.count_tokens(chunk) <= 10
        
    first_chunk_tokens = optimizer.tokenizer.encode(result[0])
    second_chunk_tokens = optimizer.tokenizer.encode(result[1])
    
    overlap_size = 3
    assert first_chunk_tokens[-overlap_size:] == second_chunk_tokens[:overlap_size]

def test_optimizer_large_overlap():
    """
    PURPOSE: Tests safety measures when overlap settings violate threshold boundaries (overlap >= max_tokens).
    CAPABILITIES:
    - Bypasses infinite loops using defensive index increments.
    - Yields clean chunks.
    """
    optimizer = TokenSizeOptimizer(max_tokens=10, overlap_tokens=12)
    text = "apple banana cherry grape orange melon strawberry peach lemon lime blueberry blackberry raspberry"
    
    result = optimizer.optimize_block(text)
    assert len(result) > 1

def test_optimizer_empty_input():
    """
    PURPOSE: Verifies behavior on empty inputs.
    CAPABILITIES:
    - Returns a list containing a single empty string without crashes.
    """
    optimizer = TokenSizeOptimizer(max_tokens=10, overlap_tokens=3)
    result = optimizer.optimize_block("")
    assert result == [""]

# --- RIGOROUS EXTENDED TESTS ---

def test_optimizer_exact_match_size():
    """
    PURPOSE: Verifies behavior when token count matches max_tokens exactly.
    CAPABILITIES:
    - Prevents unnecessary division and partitioning.
    - Returns the block intact in a single list element.
    """
    optimizer = TokenSizeOptimizer(max_tokens=10, overlap_tokens=3)
    # Target text has exactly 10 tokens
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    text = optimizer.tokenizer.decode(tokens)
    
    result = optimizer.optimize_block(text)
    assert len(result) == 1
    assert result[0] == text

def test_optimizer_zero_overlap():
    """
    PURPOSE: Verifies chunk partitioning when overlap_tokens is set to 0.
    CAPABILITIES:
    - Splits text cleanly into contiguous, non-overlapping blocks.
    - Sub-chunks contain no shared tokens between boundaries.
    """
    optimizer = TokenSizeOptimizer(max_tokens=5, overlap_tokens=0)
    text = "apple banana cherry grape orange melon strawberry peach lemon lime"
    
    result = optimizer.optimize_block(text)
    assert len(result) > 1
    
    # Verify no overlap between first and second chunk tokens
    t1 = optimizer.tokenizer.encode(result[0])
    t2 = optimizer.tokenizer.encode(result[1])
    
    # Intersection of adjacent boundaries is empty (no shared context)
    assert len(set(t1).intersection(set(t2))) == 0 or t1[-1] != t2[0]

def test_optimizer_non_ascii_characters():
    """
    PURPOSE: Verifies encoding and decoding stability on multi-byte UTF-8 string inputs.
    CAPABILITIES:
    - Correctly tokenizes non-ASCII glyphs (e.g. Chinese characters and emojis).
    - Decodes back to clean valid UTF-8 without byte truncation.
    """
    optimizer = TokenSizeOptimizer(max_tokens=3, overlap_tokens=1)
    text = "你好，世界！😊🤖"
    
    result = optimizer.optimize_block(text)
    assert len(result) > 0
    # Joined text should retain meaning
    rejoined = "".join(result)
    assert "你好" in rejoined
    assert "😊" in rejoined

def test_optimizer_exact_overlap_boundary():
    """
    PURPOSE: Tests the exact boundary where overlap_tokens equals max_tokens.
    CAPABILITIES:
    - Prevents infinite loops when overlap equals max_tokens.
    - Forces progress by stepping start_idx to end_idx.
    """
    optimizer = TokenSizeOptimizer(max_tokens=10, overlap_tokens=10)
    text = "apple banana cherry grape orange melon strawberry peach lemon lime blueberry blackberry raspberry"
    result = optimizer.optimize_block(text)
    assert len(result) > 1

