"""
BPE tokenizer backed by tiktoken's GPT-2 encoding.

Uses the same 50,257-token vocabulary that GPT-2 was trained with.
This gives us:
  - Subword units learned from a huge English corpus, so common security
    terms like "overflow", "injection", "CVE" are likely single tokens.
  - Much shorter sequences than char-level (~4–5× compression ratio).
  - No training step — we borrow the pre-built vocabulary for free.

The tradeoff: vocab_size is fixed at 50,257 regardless of corpus size,
so most of the embedding table is wasted capacity for this tiny model.
"""

import tiktoken


class BPETokenizer:
    def __init__(self):
        # gpt2 encoding: cl100k_base is more modern but gpt2 is more comparable
        # to nanoGPT and easier to reason about at this scale.
        self._enc = tiktoken.get_encoding("gpt2")
        self.vocab_size: int = self._enc.n_vocab  # 50,257

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, ids: list[int]) -> str:
        # errors="replace" prevents crashes on partial multi-byte tokens
        return self._enc.decode(ids)

    # save/load are no-ops — tiktoken manages its own cache
    def save(self, path) -> None:
        pass

    def load(self, path) -> None:
        pass
