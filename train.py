"""
train.py — Hand-written pretraining loop for nanoCVE.

What this script does
---------------------
1. Loads a config (from configs/) with optional CLI overrides.
2. Builds numpy memmap datasets for train and val splits.
3. Instantiates the GPT model.
4. Runs a training loop with:
     - AdamW optimizer (separate weight-decay groups)
     - Cosine LR schedule with linear warmup
     - Gradient clipping
     - Gradient accumulation (simulate large batches without the memory cost)
     - Automatic mixed precision (bfloat16 on CUDA, float32 on MPS/CPU)
     - Periodic evaluation on the val set
     - Loss logging to runs/<run_name>/losses.csv
     - Best-checkpoint saving (by val loss)
5. After training, generates a loss_curve.png.

Usage
-----
  python train.py --config configs/default.py --tokenizer bpe
  python train.py --config configs/small.py   --tokenizer char
  python train.py --config configs/default.py --tokenizer bpe --max_iters 1000
  python train.py --config configs/default.py --tokenizer bpe --grad_accum 4

All config fields can be overridden from the command line via --<field> <value>.
"""

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model import GPT, GPTConfig


# ── Device selection ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(config_path: str):
    """Dynamically import a config module and return its Config instance."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Config()


def apply_cli_overrides(cfg, overrides: dict) -> None:
    """Apply --key value overrides from CLI onto a config dataclass."""
    for key, val in overrides.items():
        if not hasattr(cfg, key):
            print(f"Warning: config has no field '{key}', skipping")
            continue
        field_type = type(getattr(cfg, key))
        try:
            setattr(cfg, key, field_type(val))
        except (ValueError, TypeError) as e:
            print(f"Warning: could not cast --{key} {val!r} to {field_type}: {e}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenDataset:
    """
    Wraps a flat numpy memmap of token IDs.

    get_batch() samples B random windows of length block_size from the array.
    The target is the same window shifted right by 1 (next-token prediction).
    """

    def __init__(self, bin_path: Path, block_size: int, dtype: np.dtype):
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self.block_size = block_size
        print(f"  Dataset {bin_path.name}: {len(self.data):,} tokens")

    def get_batch(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Sample random start positions (avoid overflowing the array)
        ix = np.random.randint(0, len(self.data) - self.block_size, size=(batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64))
            for i in ix
        ])
        # Target = input shifted by 1: predicting token t+1 given tokens 0..t
        y = torch.stack([
            torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])
        return x.to(device), y.to(device)


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, cfg) -> float:
    """
    Cosine decay with linear warmup.

    - Steps 0..warmup_iters: LR increases linearly from 0 to learning_rate.
    - Steps warmup_iters..max_iters: LR decays via cosine from learning_rate
      down to min_lr.
    - After max_iters: LR stays at min_lr.

    This avoids a large gradient spike at the very start of training (warmup)
    and gives the model time to settle near a minimum at the end (cosine).
    """
    if step < cfg.lr_warmup_iters:
        return cfg.learning_rate * (step + 1) / cfg.lr_warmup_iters

    if step > cfg.max_iters:
        return cfg.min_lr

    # Cosine decay from learning_rate → min_lr
    decay_ratio = (step - cfg.lr_warmup_iters) / (cfg.max_iters - cfg.lr_warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ── Optimizer setup ───────────────────────────────────────────────────────────

def build_optimizer(model: GPT, cfg) -> torch.optim.AdamW:
    """
    AdamW with weight decay applied only to weight matrices, not biases or
    LayerNorm parameters.

    Why? Weight decay is L2 regularization — it prevents large weights. But
    biases and LayerNorm scale/shift parameters don't need this constraint
    (they don't contribute to representational complexity in the same way).
    Decaying them just makes training harder without benefit.
    """
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay_params.append(p)     # weight matrices, embeddings
        else:
            no_decay_params.append(p)  # biases, LayerNorm scale/shift

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in no_decay_params)
    print(f"  Optimizer: {n_decay:,} params with weight decay, "
          f"{n_nodecay:,} without")

    return torch.optim.AdamW(param_groups, lr=cfg.learning_rate, betas=(0.9, 0.95))


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_ds: TokenDataset,
    val_ds: TokenDataset,
    cfg,
    device: torch.device,
) -> dict[str, float]:
    """
    Estimate mean loss over eval_iters random batches from each split.
    Uses torch.no_grad() + model.eval() to disable dropout.
    """
    model.eval()
    out = {}
    for split, ds in [("train", train_ds), ("val", val_ds)]:
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg, tokenizer_name: str) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device()
    print(f"\nDevice: {device}")

    run_name = getattr(cfg, "run_name", "run")
    run_dir = ROOT / "runs" / f"{run_name}_{tokenizer_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    cache = ROOT / "data" / "cache"
    if tokenizer_name == "char":
        dtype = np.uint16
        train_bin = cache / "train_char.bin"
        val_bin   = cache / "val_char.bin"
    else:
        dtype = np.uint32
        train_bin = cache / "train_bpe.bin"
        val_bin   = cache / "val_bpe.bin"

    for p in (train_bin, val_bin):
        if not p.exists():
            print(f"ERROR: {p} not found. Run `make prepare` first.")
            sys.exit(1)

    print("\nLoading datasets:")
    train_ds = TokenDataset(train_bin, cfg.block_size, dtype)
    val_ds   = TokenDataset(val_bin,   cfg.block_size, dtype)

    # ── Vocab size ────────────────────────────────────────────────────────────
    if tokenizer_name == "char":
        from tokenizer import CharTokenizer
        tok = CharTokenizer()
        tok.load(cache / "char_vocab.json")
        vocab_size = tok.vocab_size
    else:
        from tokenizer import BPETokenizer
        vocab_size = BPETokenizer().vocab_size

    print(f"\nVocab size: {vocab_size:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPT(model_cfg).to(device)
    n_params = model.num_parameters()
    print(f"Model: {n_params/1e6:.2f}M parameters")
    print(f"       {cfg.n_layer}L {cfg.n_head}H {cfg.n_embd}E {cfg.block_size}ctx")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    print("\nSetting up optimizer:")
    optimizer = build_optimizer(model, cfg)

    # ── Logging ───────────────────────────────────────────────────────────────
    csv_path = run_dir / "losses.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])
    print(f"Logging losses to {csv_path}")

    best_val_loss = float("inf")
    best_ckpt_path = run_dir / "ckpt_best.pt"

    # ── Training loop ─────────────────────────────────────────────────────────
    grad_accum = cfg.grad_accum
    eff_batch = cfg.batch_size * grad_accum
    print(f"\n{'='*60}")
    print(f"Training for {cfg.max_iters} steps  |  "
          f"batch={cfg.batch_size}×{grad_accum}={eff_batch}  |  "
          f"block={cfg.block_size}  |  tokenizer={tokenizer_name}")
    print(f"{'='*60}\n")

    # ── Mixed precision setup ─────────────────────────────────────────────────
    # bfloat16 on CUDA: halves memory, ~30% faster matmuls, no loss scaling needed
    # (bfloat16 has the same exponent range as float32, unlike float16).
    # MPS and CPU stay in float32 — bfloat16 support is incomplete on MPS.
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16
    # bfloat16 has the same exponent range as float32, so no GradScaler needed.
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_amp else
        torch.autocast(device_type="cpu", enabled=False)  # no-op context
    )
    if use_amp:
        print(f"Mixed precision: bfloat16 (CUDA)")
    else:
        print(f"Mixed precision: disabled (float32 on {device.type})")

    # ── Gradient accumulation ─────────────────────────────────────────────────
    # Accumulate gradients over `grad_accum` micro-batches before stepping.
    # Effective batch size = batch_size × grad_accum.
    # This lets us simulate large batches (e.g. 64) on hardware that only fits
    # small ones (e.g. 16) without changing the learning dynamics.
    if grad_accum > 1:
        print(f"Gradient accumulation: {grad_accum} steps "
              f"(effective batch = {cfg.batch_size * grad_accum})")

    model.train()
    t_start = time.time()

    for step in range(cfg.max_iters + 1):

        # ── Eval checkpoint ───────────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_ds, val_ds, cfg, device)
            elapsed = time.time() - t_start
            lr_now = get_lr(step, cfg)

            print(
                f"step {step:>5d}/{cfg.max_iters}  |  "
                f"train {losses['train']:.4f}  |  "
                f"val {losses['val']:.4f}  |  "
                f"lr {lr_now:.2e}  |  "
                f"{elapsed:.0f}s"
            )

            csv_writer.writerow([step, losses["train"], losses["val"], lr_now, elapsed])
            csv_file.flush()

            # Save best checkpoint
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                import dataclasses
                checkpoint = {
                    "step": step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    # Store configs as plain dicts so torch.save can pickle them
                    # regardless of how the config module was imported.
                    "model_config": dataclasses.asdict(model_cfg),
                    "train_config": dataclasses.asdict(cfg),
                    "tokenizer": tokenizer_name,
                }
                torch.save(checkpoint, best_ckpt_path)
                print(f"  ✓ New best val loss {best_val_loss:.4f} — checkpoint saved")

        if step == cfg.max_iters:
            break  # final eval done, stop before taking another gradient step

        # ── Learning rate update ──────────────────────────────────────────────
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ── Forward + backward (with gradient accumulation) ───────────────────
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum):
            x, y = train_ds.get_batch(cfg.batch_size, device)
            with autocast_ctx:
                _, loss = model(x, y)
            # Scale loss so gradients are averaged across accumulation steps,
            # not summed — keeps the effective LR independent of grad_accum.
            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping prevents occasional large gradient spikes from
        # destabilising training. 1.0 is the standard threshold for GPT models.
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

    # ── Wrap up ───────────────────────────────────────────────────────────────
    csv_file.close()
    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time/60:.1f} min")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {best_ckpt_path}")

    # Generate loss curve
    print("\nGenerating loss curve...")
    _plot_loss(csv_path, run_dir / "loss_curve.png", run_name, tokenizer_name)
    print(f"Loss curve: {run_dir}/loss_curve.png")


def _plot_loss(csv_path: Path, out_path: Path, run_name: str, tokenizer: str) -> None:
    """Read losses CSV and write a loss_curve.png."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe on headless servers
    import matplotlib.pyplot as plt

    steps, train_losses, val_losses = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_losses, label="train", linewidth=2)
    ax.plot(steps, val_losses,   label="val",   linewidth=2, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"nanoCVE — {run_name} ({tokenizer})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train nanoCVE")
    parser.add_argument("--config", default="configs/default.py",
                        help="Path to config file (default: configs/default.py)")
    parser.add_argument("--tokenizer", default=None,
                        choices=["char", "bpe"],
                        help="Override tokenizer (char or bpe)")
    parser.add_argument("--run_name", default=None,
                        help="Override run name (used for run directory)")

    # Allow overriding any config field: --n_layer 12 --batch_size 32 etc.
    parser.add_argument("--n_layer",       type=int,   default=None)
    parser.add_argument("--n_head",        type=int,   default=None)
    parser.add_argument("--n_embd",        type=int,   default=None)
    parser.add_argument("--block_size",    type=int,   default=None)
    parser.add_argument("--dropout",       type=float, default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--max_iters",     type=int,   default=None)
    parser.add_argument("--eval_interval", type=int,   default=None)
    parser.add_argument("--eval_iters",    type=int,   default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--grad_accum",    type=int,   default=None,
                        help="Gradient accumulation steps (effective_batch = batch_size × grad_accum)")
    parser.add_argument("--seed",          type=int,   default=None)

    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply explicit CLI overrides (skip None values)
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("config", "tokenizer", "run_name")}
    apply_cli_overrides(cfg, overrides)

    if args.tokenizer:
        cfg.tokenizer = args.tokenizer
    if args.run_name:
        cfg.run_name = args.run_name

    train(cfg, cfg.tokenizer)


if __name__ == "__main__":
    main()
