"""
model.py — Decoder-only transformer (GPT-style) for nanoCVE.

Architecture overview
---------------------
  Token embedding  →  Positional embedding
         ↓
  N × TransformerBlock
    ├─ LayerNorm  (pre-LN: norm before attention, not after)
    ├─ CausalSelfAttention
    ├─ residual add
    ├─ LayerNorm
    ├─ MLP
    └─ residual add
         ↓
  Final LayerNorm
         ↓
  Linear head  (weight-tied to token embedding)
         ↓
  Logits over vocab

Key design choices
------------------
- Pre-LN (norm before sub-layer): more stable training than post-LN,
  used by GPT-J, PaLM, and nanoGPT.
- Weight tying: the output projection matrix is shared with the token
  embedding. This halves the embedding parameters (~40% of total at
  small scale) and often improves perplexity because the model must
  learn a single representation space for both input and output.
- Causal mask: implemented via torch.tril so each position can only
  attend to itself and earlier positions (no cheating on future tokens).
- GELU activation: smoother than ReLU, standard in GPT-2 and beyond.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    vocab_size: int = 50257   # overwritten at init time based on tokenizer
    block_size: int = 256     # maximum sequence length (context window)
    n_layer: int = 6          # number of transformer blocks
    n_head: int = 6           # number of attention heads
    n_embd: int = 384         # embedding dimension (must be divisible by n_head)
    dropout: float = 0.1      # dropout probability (0.0 to disable)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )


# ── Causal Self-Attention ─────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Each head independently computes attention over the sequence, then all
    head outputs are concatenated and projected back to n_embd.

    "Causal" means we mask out future positions so token at position t
    can only attend to positions 0..t. This is what makes the model
    autoregressive — it can only predict the next token from past context.

    Dimensions throughout:
      B = batch size
      T = sequence length (≤ block_size)
      C = n_embd  (embedding dimension)
      hs = C // n_head  (head size, dimension per head)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # Single linear that produces Q, K, V for all heads in one shot.
        # Output dim is 3*n_embd; we'll split it into thirds below.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # Output projection — maps concatenated head outputs back to n_embd.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of shape (block_size, block_size).
        # Registered as a buffer so it moves to the right device with .to(device)
        # but is NOT a learnable parameter.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Compute Q, K, V and split into n_head pieces
        qkv = self.c_attn(x)                        # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)     # each (B, T, C)

        # Reshape to (B, n_head, T, head_size) for batched attention
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Scaled dot-product attention
        # Scale by 1/sqrt(head_size) to keep dot products from growing too large
        # (large dot products → very peaked softmax → vanishing gradients).
        scale = 1.0 / math.sqrt(self.head_size)
        att = (q @ k.transpose(-2, -1)) * scale     # (B, n_head, T, T)

        # Apply causal mask: positions beyond t are set to -inf → zero after softmax
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        y = att @ v                                  # (B, n_head, T, head_size)

        # Concatenate heads and project back to n_embd
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


# ── MLP block ─────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Position-wise feed-forward network.

    Each token's representation is independently transformed through:
      Linear (C → 4C)  →  GELU  →  Linear (4C → C)

    The 4× expansion is standard GPT practice. GELU is smoother than ReLU
    (it has a non-zero gradient near 0) and is what GPT-2 and GPT-3 use.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# ── Transformer Block ─────────────────────────────────────────────────────────

class Block(nn.Module):
    """
    One transformer block: pre-LN → attention → residual, pre-LN → MLP → residual.

    Pre-LN means we normalize the input BEFORE each sub-layer (vs. GPT-1/BERT
    which normalize after). Pre-LN gives smoother loss landscapes and is
    generally easier to train at small scale without a learning rate warm-up.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections let gradients flow directly to early layers,
        # making it much easier to train deep networks.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ── Full GPT Model ────────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    Decoder-only transformer (GPT architecture).

    Parameters
    ----------
    config : GPTConfig
        Holds all architecture hyperparameters.

    Forward pass
    ------------
    Input:  token indices  (B, T)  — integers in [0, vocab_size)
    Output: logits         (B, T, vocab_size) — unnormalized next-token scores
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            # Token embedding: maps each integer token id to a learned vector.
            "wte": nn.Embedding(config.vocab_size, config.n_embd),

            # Positional embedding: learned offset for each position 0..block_size-1.
            # Unlike sinusoidal positions (original Transformer), learned positions
            # work just as well at this scale and are simpler.
            "wpe": nn.Embedding(config.block_size, config.n_embd),

            "drop": nn.Dropout(config.dropout),

            # Stack of transformer blocks
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),

            # Final layer norm before the output projection
            "ln_f": nn.LayerNorm(config.n_embd),
        })

        # Output projection: maps embeddings to logits over the vocabulary.
        # bias=False is standard; the bias would be redundant given LayerNorm.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share the token embedding matrix with the output projection.
        # Rationale: the model needs to map token ids → vectors (embedding) and
        # vectors → token ids (lm_head). It makes sense for both to use the same
        # "meaning" space, and it halves the parameter count for this large matrix.
        self.transformer["wte"].weight = self.lm_head.weight

        # Initialize weights using GPT-2's scheme
        self.apply(self._init_weights)

        # Scale residual projections by 1/sqrt(n_layer) so that the variance of
        # activations stays roughly constant regardless of depth (from GPT-2 paper).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """Standard GPT-2 weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args
        ----
        idx     : (B, T) integer token indices
        targets : (B, T) integer token indices shifted by 1 (next-token labels).
                  If None, only logits are returned (inference mode).

        Returns
        -------
        logits : (B, T, vocab_size)
        loss   : scalar cross-entropy loss, or None if targets is None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # Build position indices [0, 1, ..., T-1] for each item in the batch
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)

        # Token + positional embeddings, then dropout
        tok_emb = self.transformer["wte"](idx)   # (B, T, n_embd)
        pos_emb = self.transformer["wpe"](pos)   # (1, T, n_embd)
        x = self.transformer["drop"](tok_emb + pos_emb)

        # Pass through all transformer blocks
        for block in self.transformer["h"]:
            x = block(x)

        # Final layer norm
        x = self.transformer["ln_f"](x)

        if targets is not None:
            # Training: compute logits for all positions and measure cross-entropy.
            logits = self.lm_head(x)             # (B, T, vocab_size)
            # Flatten to (B*T, vocab_size) and (B*T,) for F.cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        else:
            # Inference: only compute logit for the last position (efficiency).
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        At each step:
          1. Feed the current context (up to block_size) through the model.
          2. Take the logit at the last position and divide by temperature.
          3. Optionally zero out all but the top-k logits (nucleus sampling lite).
          4. Sample from the resulting distribution.
          5. Append the new token and repeat.

        Args
        ----
        idx            : (B, T) seed token indices
        max_new_tokens : how many new tokens to generate
        temperature    : >1 → more random, <1 → more greedy
        top_k          : if set, restrict sampling to the k most likely tokens
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)              # (B, 1, vocab_size)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            if top_k is not None:
                # Zero out all logits except the top-k
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters (excluding tied weights from double-counting)."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
