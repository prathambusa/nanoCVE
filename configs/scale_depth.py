"""
Scaling experiment — doubles n_layer vs. default (6 → 12), all else equal.
Used to observe how depth affects val loss, training time, and overfitting.
~41M params with BPE tokenizer (embedding table alone is ~19M).
"""
from dataclasses import dataclass

@dataclass
class Config:
    # Model — only n_layer changes vs. default
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.1

    # Training — identical to default for a fair comparison
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 100
    learning_rate: float = 3e-4
    lr_warmup_iters: int = 200
    min_lr: float = 3e-5
    grad_clip: float = 1.0
    weight_decay: float = 0.1

    # Misc
    tokenizer: str = "bpe"
    run_name: str = "scale_deep"
    seed: int = 42
