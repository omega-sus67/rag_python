import tiktoken
from typing import List, Dict, Any

class TokenSizeOptimizer:
    """Enforces absolute token count guardrails on raw text strings using forced overlapping splits."""
    def __init__(self, encoding_name: str = "cl100k_base", max_tokens: int = 400, overlap_tokens: int = 100):
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, text: str) -> int:
        """Computes the exact number of tokens a string yields under the specified tokenizer."""
        return len(self.tokenizer.encode(text))

    def optimize_block(self, text: str) -> List[str]:
        """
        Takes a raw string block and slices it into sub-strings that strictly 
        obey max_tokens boundaries, using token-level sliding overlaps.
        """
        # Step 1: Convert raw string into a list of individual token IDs
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)

        # If the text safely fits inside our budget, return it completely intact
        if total_tokens <= self.max_tokens:
            return [text]

        sub_blocks = []
        start_idx = 0

        # Step 2: Slide across token IDs, enforcing constraints
        while start_idx < total_tokens:
            end_idx = min(start_idx + self.max_tokens, total_tokens)
            
            # Extract the subset array of token integers
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Step 3: Decode the token IDs back into a clean UTF-8 string string
            decoded_text = self.tokenizer.decode(chunk_tokens)
            sub_blocks.append(decoded_text)
            
            # Advance our pointer forward, subtracting the overlap window
            if end_idx == total_tokens:
                break
            start_idx = end_idx - self.overlap_tokens
            
            # Defensive guard: prevent infinite loops if misconfigured
            if start_idx >= end_idx:
                start_idx = end_idx

        return sub_blocks