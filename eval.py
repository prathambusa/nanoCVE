"""
eval.py — Evaluate a nanoCVE checkpoint on the validation set.

Reports:
  - Cross-entropy loss (nats per token)
  - Perplexity  = exp(loss)
  - Bits-per-character (BPC) — the fair cross-tokenizer metric

BPC derivation
--------------
Cross-entropy is measured in nats per TOKEN, but tokens differ in size:
  char:  1 token  = 1 character  → nats/token = nats/char
  BPE:   1 token ≈ 4.25 chars   → nats/char = nats/token ÷ chars_per_token

BPC = nats_per_char / ln(2)

This puts char and BPE models on equal footing for comparison.

Usage
-----
  python eval.py --run_name default_bpe
  python eval.py --run_name default_char
  python eval.py --run_name default_bpe --eval_iters 200
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model import GPT, GPTConfig
from train import TokenDataset


@torch.no_grad()
def evaluate(run_name: str, eval_iters: int = 100, batch_size: int = 16) -> dict:
    # ── Load checkpoint ───────────────────────────────────────────────────────
    run_dir = ROOT / "runs" / run_name
    if not run_dir.exists():
        for p in (ROOT / "runs").glob(f"{run_name}*/ckpt_best.pt"):
            run_dir = p.parent
            break
    ckpt_path = run_dir / "ckpt_best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: No checkpoint at {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_cfg = ckpt["model_config"]
    model_cfg = GPTConfig(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    tokenizer_name = ckpt.get("tokenizer", "bpe")
    step = ckpt.get("step", "?")
    saved_val_loss = ckpt.get("val_loss", float("nan"))

    model = GPT(model_cfg).to(device)
    # strict=False: tolerate legacy 'mask' buffers saved before Flash Attention
    # was added. Missing keys are fine — the mask is no longer a registered buffer.
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # ── Load val dataset ──────────────────────────────────────────────────────
    cache = ROOT / "data" / "cache"
    if tokenizer_name == "char":
        val_bin = cache / "val_char.bin"
        char_bin = cache / "val_char.bin"
        dtype = np.uint16
    else:
        val_bin = cache / "val_bpe.bin"
        char_bin = cache / "val_char.bin"
        dtype = np.uint32

    val_ds = TokenDataset(val_bin, model_cfg.block_size, dtype)

    # Chars per token: ratio of val char tokens to val BPE tokens
    if tokenizer_name == "bpe" and (cache / "val_char.bin").exists():
        n_val_chars = len(np.fromfile(cache / "val_char.bin", dtype=np.uint16))
        n_val_bpe   = len(np.fromfile(cache / "val_bpe.bin",  dtype=np.uint32))
        chars_per_token = n_val_chars / n_val_bpe
    else:
        chars_per_token = 1.0  # char tokenizer: 1 token = 1 char

    # ── Evaluate ──────────────────────────────────────────────────────────────
    losses = []
    for _ in range(eval_iters):
        x, y = val_ds.get_batch(batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())

    val_loss = float(np.mean(losses))
    perplexity = math.exp(val_loss)
    bpc = (val_loss / chars_per_token) / math.log(2)

    return {
        "run_name": run_name,
        "step": step,
        "tokenizer": tokenizer_name,
        "n_layer": model_cfg.n_layer,
        "n_embd": model_cfg.n_embd,
        "val_loss": val_loss,
        "perplexity": perplexity,
        "bpc": bpc,
        "chars_per_token": chars_per_token,
        "saved_val_loss": saved_val_loss,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a nanoCVE checkpoint")
    parser.add_argument("--run_name", required=True,
                        help="Run name (or space-separated list for comparison)")
    parser.add_argument("--eval_iters", type=int, default=100,
                        help="Batches to average over (default: 100)")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    run_names = args.run_name.split()

    results = []
    for name in run_names:
        print(f"Evaluating {name} ...")
        r = evaluate(name, eval_iters=args.eval_iters, batch_size=args.batch_size)
        results.append(r)

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    print(f"{'Run':<28} {'Tok':<5} {'Step':>5} {'Val loss':>9} {'Perplexity':>11} {'BPC':>7}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['run_name']:<28} {r['tokenizer']:<5} {str(r['step']):>5} "
            f"{r['val_loss']:>9.4f} {r['perplexity']:>11.2f} {r['bpc']:>7.4f}"
        )

    if len(results) == 2:
        bpc_a, bpc_b = results[0]["bpc"], results[1]["bpc"]
        better = results[0]["run_name"] if bpc_a < bpc_b else results[1]["run_name"]
        ratio = max(bpc_a, bpc_b) / min(bpc_a, bpc_b)
        print(f"\n{better} is {ratio:.2f}× more efficient in bits/char")


if __name__ == "__main__":
    main()
