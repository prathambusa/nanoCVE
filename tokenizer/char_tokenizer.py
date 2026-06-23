"""
Character-level tokenizer.

Every unique character in the corpus gets an integer ID. Dead simple,
but produces very long sequences — a 200-character CVE description becomes
200 tokens, so the model must learn spelling, punctuation, and words all
from scratch.

Vocab size is typically 100–200 for English security text.
"""

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self):
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self.vocab_size: int = 0

    def build_vocab(self, corpus: str) -> None:
        """Scan corpus and assign an integer to every unique character."""
        chars = sorted(set(corpus))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        # Unknown characters (not seen during vocab build) are silently skipped.
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.idx2char.get(i, "") for i in ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"char2idx": self.char2idx}, f)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self.char2idx = data["char2idx"]
        self.idx2char = {int(i): ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
