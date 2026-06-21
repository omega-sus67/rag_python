import tiktoken
from typing import List, Dict, Any

from app.core.config import settings

class TokenSizeOptimizer:
    """
    Enforces absolute token count limits on raw text strings using sliding overlaps.
    Token-level boundaries are used instead of character-level boundaries because
    different token patterns consume different context slots inside LLMs.
    """
    def __init__(self, encoding_name: str = "cl100k_base", max_tokens: int = settings.max_tokens, overlap_tokens: int = settings.overlap_tokens):
        # Retrieve the cl100k_base encoding scheme used by OpenAI and compatible models.
        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, text: str) -> int:
        """
        Calculates the exact number of tokens a text block converts to.
        Used to measure chunk sizes for metadata tracking and vector optimization.
        """
        return len(self.tokenizer.encode(text))

    def optimize_block(self, text: str) -> List[str]:
        """
        Slices a text block into multiple overlapping chunks that strictly obey self.max_tokens.
        Uses token integer lists to divide and decode strings back into valid UTF-8.
        """
        # Encode the entire string to get the full list of token integers.
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)

        # Quick path: if the block fits, return it immediately without any partitioning overhead.
        if total_tokens <= self.max_tokens:
            return [text]

        sub_blocks = []
        start_idx = 0

        # Slide the window across the token integer array.
        while start_idx < total_tokens:
            # Determine the upper bound index for the current chunk.
            end_idx = min(start_idx + self.max_tokens, total_tokens)
            
            # Slice the list of tokens.
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode the selected integer slice back into a clean string.
            decoded_text = self.tokenizer.decode(chunk_tokens)
            sub_blocks.append(decoded_text)
            
            # If we've reached the end of the list, exit the loop.
            if end_idx == total_tokens:
                break
                
            prev_start_idx = start_idx
            # Slide the window forward, keeping overlap_tokens of overlap.
            start_idx = end_idx - self.overlap_tokens
            
            # Safety check: avoid infinite loops.
            # If the window doesn't progress forward (e.g. if overlap is set too high or max_tokens is too low),
            # force progress by jumping start_idx directly to end_idx.
            if start_idx >= end_idx or start_idx <= prev_start_idx:
                start_idx = end_idx

        return sub_blocks