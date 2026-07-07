"""
Small config — designed to run on CPU in a few minutes.
Good for smoke-testing the full pipeline.
~16M params with BPE tokenizer (embedding dominates at this scale too).
~2.5M params with char tokenizer.
"""
from dataclasses import dataclass

@dataclass
class Config:
    # Model
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    block_size: int = 128
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    learning_rate: float = 3e-4
    lr_warmup_iters: int = 100
    min_lr: float = 3e-5
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    grad_accum: int = 1

    # Misc
    tokenizer: str = "bpe"
    run_name: str = "small"
    seed: int = 42
