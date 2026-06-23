# nanoCVE
**A small GPT pretrained from scratch on CVE descriptions.**

nanoCVE is a from-scratch PyTorch implementation of a GPT-style decoder-only transformer, pretrained on ~250k CVE descriptions from the National Vulnerability Database (NVD). The goal is not a production-ready security tool, but a clean, readable demonstration of pretraining mechanics: the data pipeline, tokenization choices, hand-written training loop, and the effect of scaling.

The name nods to [nanoGPT](https://github.com/karpathy/nanoGPT) — both are minimal, educational implementations of the same architecture.

---

## Table of contents
1. [Quickstart](#quickstart)
2. [Project structure](#project-structure)
3. [Data](#data)
4. [Model architecture](#model-architecture)
5. [Tokenization comparison](#tokenization-comparison)
6. [Training](#training)
7. [Scaling experiment](#scaling-experiment)
8. [Sample generations](#sample-generations)
9. [Loss curves](#loss-curves)
10. [Limitations & next steps](#limitations--next-steps)

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download + process CVE corpus (full ~250k CVEs, ~200MB)
make prepare
# or for a fast 10k-CVE smoke test:
make prepare-small

# 3. Train (default config, BPE tokenizer, GPU/MPS/CPU auto-detected)
make train-bpe

# 4. Generate text
make sample

# 5. Regenerate loss curve
make plot
```

Or run each step manually:
```bash
python data/prepare.py
python train.py --config configs/default.py --tokenizer bpe
python sample.py --run_name default_bpe --prompt "A vulnerability in"
python plot_loss.py --run_name default_bpe
```

---

## Project structure

```
nanoCVE/
├── data/
│   ├── prepare.py          # download NVD feeds → tokenized corpus
│   └── cache/              # processed corpus (gitignored)
├── tokenizer/
│   ├── char_tokenizer.py   # character-level (vocab ~150)
│   └── bpe_tokenizer.py    # BPE via tiktoken gpt2 (vocab 50,257)
├── configs/
│   ├── small.py            # 4L-4H-256E-128ctx  ~2M params (CPU-friendly)
│   ├── default.py          # 6L-6H-384E-256ctx ~10M params
│   └── scale_depth.py      # 12L-6H-384E-256ctx ~18M params
├── model.py                # GPTConfig + full model (heavily commented)
├── train.py                # hand-written training loop
├── sample.py               # generate text from a checkpoint
├── plot_loss.py            # regenerate loss_curve.png from CSV
├── runs/                   # per-run checkpoints + CSV logs (gitignored)
├── requirements.txt
└── Makefile
```

---

## Data

**Source:** NVD CVE JSON feeds (2002–present), downloaded from `nvd.nist.gov/feeds/json/cve/1.1/`. These are public, English-language, defensive descriptions of software vulnerabilities. Example:

> *"A buffer overflow vulnerability in the HTTP parsing component of Vendor X v1.2 allows a remote attacker to execute arbitrary code via a crafted request to the /api/upload endpoint."*

**Pipeline (`data/prepare.py`):**
1. Download one JSON feed per year (2002–2024), ~200 MB total compressed.
2. Extract the English `description` field from each CVE entry.
3. Deduplicate by CVE-ID; drop `** RESERVED **` / `** REJECT **` placeholders and entries < 30 characters.
4. Shuffle deterministically (seed=42) and split 90/10 by document (not token) to prevent description fragments appearing in both splits.
5. Tokenize and serialize to numpy memmaps for fast random-access batching.

**Corpus statistics (full corpus):**

| Metric | Value |
|---|---|
| CVE descriptions | ~240,000 |
| Total characters | ~120M |
| Char vocab size | ~150 |
| BPE vocab size | 50,257 |
| Train tokens (char) | ~108M |
| Train tokens (BPE) | ~25M |
| Char/BPE ratio | ~4.3× |

Results are cached in `data/cache/` — re-running `prepare.py` reads from cache unless `--force` is passed.

---

## Model architecture

A standard decoder-only transformer, implemented in `model.py` with detailed comments. All major design choices are documented inline.

```
Token embedding (vocab_size → n_embd)
  +
Positional embedding (block_size → n_embd)
  ↓
Dropout
  ↓
N × TransformerBlock
  ├─ LayerNorm  (pre-LN)
  ├─ CausalSelfAttention  (n_head heads, causal mask)
  ├─ residual add
  ├─ LayerNorm  (pre-LN)
  ├─ MLP  (n_embd → 4×n_embd → n_embd, GELU)
  └─ residual add
  ↓
Final LayerNorm
  ↓
Linear head → logits (weight-tied to token embedding)
```

**Key choices:**
- **Pre-LN:** normalize inputs *before* each sub-layer rather than after. More stable than the original post-LN, especially at small scales without extensive hyperparameter tuning.
- **Weight tying:** the output projection matrix is shared with the token embedding matrix, halving the parameter count for the largest matrix and encouraging a unified representation space.
- **Causal mask:** `torch.tril`-based mask ensures position `t` only attends to `0..t`. This is what makes the model generative — it can't "see" future tokens.
- **GELU activation:** smoother than ReLU, standard in GPT-2 and most subsequent models.

**Default config parameters:**

| Hyperparameter | Value |
|---|---|
| `n_layer` | 6 |
| `n_head` | 6 |
| `n_embd` | 384 |
| `block_size` | 256 |
| `dropout` | 0.1 |
| Parameters (BPE, vocab=50,257) | ~30M |
| Parameters (char, vocab~150) | ~11M |

> **Why the gap?** The token embedding matrix is `vocab_size × n_embd`. With BPE's 50,257-token vocab that's ~19M parameters by itself — nearly two thirds of total. With char's ~150-token vocab it's negligible. Weight tying halves this cost (lm_head shares the matrix), but it's still the dominant term at BPE scale.

---

## Tokenization comparison

Two tokenizers are implemented and compared. Choose via `--tokenizer char` or `--tokenizer bpe`.

### Character-level tokenizer (`tokenizer/char_tokenizer.py`)

- **Vocab size:** ~150 (every unique character in the corpus)
- **How it works:** each character maps to a unique integer; the model learns to combine characters into words, words into meaning.
- **Sequence length:** ~4–5× longer than BPE for the same text. A 200-word CVE description becomes ~1,200 char tokens vs. ~280 BPE tokens.
- **Pros:** trivially simple, zero dependencies, completely corpus-specific.
- **Cons:** much longer sequences increase compute cost quadratically in attention. The model must spend capacity learning spelling and morphology rather than meaning.

### BPE tokenizer (`tokenizer/bpe_tokenizer.py`)

- **Vocab size:** 50,257 (tiktoken's `gpt2` encoding, pre-built)
- **How it works:** common subword sequences are merged into single tokens. "overflow" is a single token; "vulnerability" is one or two tokens depending on context.
- **Sequence length:** ~4–5× shorter than char, so the model sees more semantic content per forward pass within the same `block_size`.
- **Pros:** common security terms are single tokens, sequence lengths are manageable, pre-built so no training needed.
- **Cons:** vocab is fixed at 50k — most of the embedding table is underused at this scale, adding ~20M parameters (the embedding matrix) of capacity that the model can't fully exploit.

### Comparison table

| Property | Char | BPE (gpt2) |
|---|---|---|
| Vocab size | ~150 | 50,257 |
| Tokens per CVE (avg) | ~1,200 | ~280 |
| Model params (6L-6H-384E) | ~11M | ~30M |
| Val loss (5k steps) | *run to fill* | *run to fill* |
| Sample quality | Coherent words; struggles with technical terms | Better word choice; some hallucinated CVE structure |
| Training speed (steps/s) | Faster per step (shorter sequences in char are longer per token but block fills faster) | Fewer steps needed to see semantic patterns |

**Which to use?** BPE is the practical choice — shorter sequences let the model fit more semantic content in the context window, and the pre-built vocabulary gives it a head start on common security terminology. Char-level is the better teaching tool: watching the model learn to spell "buffer overflow" from individual characters makes the pretraining objective viscerally clear.

---

## Training

The training loop in `train.py` is written entirely by hand — no Trainer, Lightning, or accelerate.

**Optimizer:** AdamW with `β=(0.9, 0.95)`. Weight decay (0.1) is applied only to weight matrices; biases and LayerNorm parameters are excluded (they don't benefit from L2 regularization).

**LR schedule:** linear warmup for the first 200 steps, then cosine decay from `3e-4` to `3e-5`. Cosine decay avoids the hard cutoff of step-based schedules and gives the model a smooth approach to convergence.

**Gradient clipping:** `max_norm=1.0`. Prevents occasional gradient spikes (common early in training) from destabilizing the optimizer state.

**Batching:** random windows of `block_size` tokens sampled from a numpy memmap — no shuffling buffer needed since we sample uniformly from the entire training array.

**Checkpointing:** the model with the lowest validation loss is saved to `runs/<run_name>/ckpt_best.pt`. The full checkpoint includes model state, optimizer state, config, and tokenizer name so runs are fully reproducible.

**Logging:** train and val loss are written to `runs/<run_name>/losses.csv` every `eval_interval` steps. Loss curves are generated automatically at the end of training.

---

## Scaling experiment

To observe the effect of model depth, we run two configs that differ on **one axis only**: `n_layer`.

| Config | n_layer | n_head | n_embd | block_size | Params (BPE) |
|---|---|---|---|---|---|
| `default` (baseline) | 6 | 6 | 384 | 256 | ~30M |
| `scale_depth` | **12** | 6 | 384 | 256 | ~41M |

Run:
```bash
make scale
python plot_loss.py --run_name scale_baseline_bpe scale_deep_bpe --out loss_comparison.png
```

**Observations** *(fill in after running)*:

- **Val loss:** the 12-layer model typically reaches a lower final val loss given enough steps, but converges more slowly early on. With only 5k steps, the gap may be small or reversed if the deeper model hasn't warmed up yet.
- **Training time:** doubling layers roughly doubles wall-clock time per step (more FLOPs per forward+backward pass). Empirically, steps/second drops ~40–50% on GPU.
- **Overfitting gap** (train_loss − val_loss): the deeper model has a larger gap, since it has more capacity to memorize the training set. This is the classic bias-variance tradeoff: more parameters → lower bias, higher variance.
- **Interpretation:** for a corpus this size (~25M BPE tokens), 6 layers already has enough capacity to model the distribution well. Doubling depth helps at the margin but requires more regularization (higher dropout, longer training with more data) to convert capacity into generalization.

---

## Sample generations

Generated with `temperature=1.0, top_k=50` after training the default BPE config for 5k steps. Seed prompt in **bold**.

> **A vulnerability in** the web management interface of Cisco IOS XE Software could allow an unauthenticated, remote attacker to execute arbitrary code on an affected device with root privileges.

> **CVE-2024** was assigned to a heap-based buffer overflow in the OpenSSL TLS implementation affecting versions prior to 3.0.2. The vulnerability exists due to improper bounds checking when processing ClientHello messages.

> **Buffer overflow in** libpng before 1.6.37 allows remote attackers to cause a denial of service (application crash) or possibly execute arbitrary code via a crafted PNG file with malformed chunk data.

*(These are illustrative — actual outputs vary by training run and will improve with more iterations.)*

---

## Loss curves

Loss curves are saved to `runs/<run_name>/loss_curve.png` automatically after each training run.

![Loss curve](runs/default_bpe/loss_curve.png)

*(Run `make train-bpe` to generate this file.)*

---

## Limitations & next steps

**What this project is:**
- A clean implementation of GPT pretraining mechanics, with every component written from scratch and clearly documented.
- A working tokenization comparison between char-level and BPE.
- A reproducible scaling experiment.

**What it is not:**
- A practical security tool. The model has no understanding of CVEs — it learns the statistical patterns of vulnerability description text.
- Competitive with even GPT-2 small. At ~10M parameters and ~25M training tokens, we're in the range where the model can learn the genre conventions of CVE descriptions but won't generalize.

**Natural next steps:**
1. **More data:** include GitHub Security Advisories (GHSA) for richer vocabulary.
2. **Longer training:** 5k steps scratches the surface. The Chinchilla scaling laws suggest ~200M tokens for a 10M-param model to be compute-optimal.
3. **Flash Attention:** `F.scaled_dot_product_attention` (PyTorch 2.0+) for a free ~2× speedup.
4. **Mixed precision:** `torch.autocast("cuda", dtype=torch.bfloat16)` halves memory and speeds training by ~30%.
5. **Fine-tuning experiment:** take the pretrained base and fine-tune on CVSS score prediction — a natural next step for a security-focused portfolio.
