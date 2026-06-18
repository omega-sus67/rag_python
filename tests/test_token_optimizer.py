# tests/test_token_optimizer.py

from app.utils.token_optimizer import TokenSizeOptimizer

def test_optimizer_leaves_small_text_intact():
    optimizer = TokenSizeOptimizer(max_tokens=20, overlap_tokens=5)
    text = "Short text."
    
    # Since the text is small, it should return a single block identical to the input
    result = optimizer.optimize_block(text)
    assert result == [text]

def test_optimizer_splits_large_text_with_overlap():
    # Let's create an optimizer with very small limits to make testing easy
    optimizer = TokenSizeOptimizer(max_tokens=10, overlap_tokens=3)
    
    # Create a long repeating string
    # We construct a sentence that is guaranteed to exceed 10 tokens
    text = "apple banana cherry grape orange melon strawberry peach lemon lime blueberry blackberry raspberry"
    
    result = optimizer.optimize_block(text)
    
    # It must split the text into multiple chunks
    assert len(result) > 1
    
    # Let's check that the maximum tokens of any chunk never exceeds 10
    for chunk in result:
        assert optimizer.count_tokens(chunk) <= 10
        
    # Let's verify that the last 3 tokens of the first chunk overlap with the start of the second chunk
    first_chunk_tokens = optimizer.tokenizer.encode(result[0])
    second_chunk_tokens = optimizer.tokenizer.encode(result[1])
    
    overlap_size = 3
    assert first_chunk_tokens[-overlap_size:] == second_chunk_tokens[:overlap_size]

def test_optimizer_large_overlap():
    # overlap_tokens (12) >= max_tokens (10)
    optimizer = TokenSizeOptimizer(max_tokens=10, overlap_tokens=12)
    text = "apple banana cherry grape orange melon strawberry peach lemon lime blueberry blackberry raspberry"
    
    # This should run without an infinite loop and produce chunks
    result = optimizer.optimize_block(text)
    assert len(result) > 1

def test_optimizer_empty_input():
    optimizer = TokenSizeOptimizer(max_tokens=10, overlap_tokens=3)
    result = optimizer.optimize_block("")
    assert result == [""]

