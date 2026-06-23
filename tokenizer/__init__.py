from .char_tokenizer import CharTokenizer
from .bpe_tokenizer import BPETokenizer

def get_tokenizer(name: str, corpus: str = ""):
    """Return a tokenizer by name. 'corpus' is only used by CharTokenizer to build vocab."""
    if name == "char":
        tok = CharTokenizer()
        if corpus:
            tok.build_vocab(corpus)
        return tok
    elif name == "bpe":
        return BPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer: {name!r}. Choose 'char' or 'bpe'.")
