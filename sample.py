"""
sample.py — Load a nanoCVE checkpoint and generate text from a prompt.

Usage
-----
  python sample.py --run_name default_bpe --prompt "A vulnerability in"
  python sample.py --run_name default_bpe --prompt "CVE-2024" --max_new_tokens 200
  python sample.py --run_name default_bpe --temperature 0.8 --top_k 40
  python sample.py --run_name small_char  --prompt "Buffer overflow"

The script auto-detects the tokenizer from the checkpoint.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model import GPT


def load_checkpoint(run_name: str, device: torch.device):
    """Load model and metadata from best checkpoint for a given run."""
    ckpt_path = ROOT / "runs" / run_name / "ckpt_best.pt"
    if not ckpt_path.exists():
        # Try with tokenizer suffix patterns
        for p in (ROOT / "runs").glob(f"{run_name}*/ckpt_best.pt"):
            ckpt_path = p
            break
    if not ckpt_path.exists():
        print(f"ERROR: No checkpoint found for run '{run_name}'.")
        print(f"  Looked in: {ROOT / 'runs' / run_name}")
        print(f"  Available runs: {[p.name for p in (ROOT / 'runs').iterdir() if p.is_dir()]}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt


def get_tokenizer(tokenizer_name: str):
    """Build the tokenizer used during training."""
    cache = ROOT / "data" / "cache"
    if tokenizer_name == "char":
        from tokenizer import CharTokenizer
        tok = CharTokenizer()
        vocab_path = cache / "char_vocab.json"
        if not vocab_path.exists():
            print(f"ERROR: char vocab not found at {vocab_path}. Run prepare first.")
            sys.exit(1)
        tok.load(vocab_path)
        return tok
    else:
        from tokenizer import BPETokenizer
        return BPETokenizer()


def generate(
    prompt: str,
    run_name: str,
    max_new_tokens: int = 150,
    temperature: float = 1.0,
    top_k: int | None = 50,
    num_samples: int = 1,
    device: torch.device | None = None,
) -> list[str]:
    """
    Generate text from a prompt using a saved checkpoint.

    Returns a list of generated strings (length = num_samples).
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    ckpt = load_checkpoint(run_name, device)

    # model_config is stored as a plain dict (to survive dynamic module pickling)
    from model import GPTConfig
    raw_cfg = ckpt["model_config"]
    model_cfg = GPTConfig(**raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
    tokenizer_name = ckpt.get("tokenizer", "bpe")
    val_loss = ckpt.get("val_loss", float("nan"))
    step = ckpt.get("step", "?")

    print(f"  Tokenizer : {tokenizer_name}")
    print(f"  Step      : {step}")
    print(f"  Val loss  : {val_loss:.4f}")
    print(f"  Config    : {model_cfg.n_layer}L {model_cfg.n_head}H "
          f"{model_cfg.n_embd}E {model_cfg.block_size}ctx")

    # Rebuild model and load weights
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tok = get_tokenizer(tokenizer_name)

    # Encode prompt
    prompt_ids = tok.encode(prompt)
    if not prompt_ids:
        print("WARNING: prompt encoded to empty — using a single padding token.")
        prompt_ids = [0]
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    results = []
    for i in range(num_samples):
        out = model.generate(
            prompt_tensor.clone(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Decode only the newly generated tokens (after the prompt)
        new_ids = out[0, len(prompt_ids):].tolist()
        generated = tok.decode(new_ids)
        full_text = prompt + generated
        results.append(full_text)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate text from a nanoCVE checkpoint")
    parser.add_argument("--run_name", required=True,
                        help="Run name (subdirectory of runs/, e.g. 'default_bpe')")
    parser.add_argument("--prompt", default="A vulnerability in",
                        help="Seed text for generation")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                        help="Number of tokens to generate (default: 150)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (>1=more random, <1=more focused)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling (0 to disable)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to generate")
    args = parser.parse_args()

    top_k = args.top_k if args.top_k > 0 else None

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Settings: max_new_tokens={args.max_new_tokens}, "
          f"temperature={args.temperature}, top_k={top_k}\n")

    samples = generate(
        prompt=args.prompt,
        run_name=args.run_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        num_samples=args.num_samples,
    )

    print("\n" + "=" * 60)
    for i, s in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(s)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
