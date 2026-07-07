"""
Default config — nanoGPT scale, targets a single GPU or fast MPS device.
~30M params with BPE tokenizer (vocab_size=50257) — embedding table dominates.
~11M params with char tokenizer (vocab_size~150).
"""
from dataclasses import dataclass

@dataclass
class Config:
    # Model
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 100
    learning_rate: float = 3e-4
    lr_warmup_iters: int = 200
    min_lr: float = 3e-5
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    grad_accum: int = 1        # gradient accumulation steps; effective_batch = batch_size × grad_accum

    # Misc
    tokenizer: str = "bpe"
    run_name: str = "default"
    seed: int = 42
