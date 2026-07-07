.PHONY: prepare prepare-small train-bpe train-char train-small eval sample scale plot help

# Reduce batch_size from the config default of 64 to 16 to fit in MPS/GPU memory.
# Override with: make train-bpe BATCH=32
BATCH ?= 16
ITERS ?= 2000

# ── Data ──────────────────────────────────────────────────────────────────────
prepare:
	python data/prepare.py

prepare-small:
	python data/prepare.py --limit 10000

# ── Training ──────────────────────────────────────────────────────────────────
train-bpe:
	python train.py --config configs/default.py --tokenizer bpe \
	  --run_name default_bpe --batch_size $(BATCH) --max_iters $(ITERS)

train-char:
	python train.py --config configs/default.py --tokenizer char \
	  --run_name default_char --batch_size $(BATCH) --max_iters $(ITERS)

train-small:
	python train.py --config configs/small.py --tokenizer bpe \
	  --run_name small_bpe --batch_size $(BATCH) --max_iters $(ITERS)

# ── Scaling experiment ────────────────────────────────────────────────────────
scale:
	python train.py --config configs/default.py     --tokenizer bpe \
	  --run_name scale_baseline --batch_size $(BATCH) --max_iters $(ITERS)
	python train.py --config configs/scale_depth.py --tokenizer bpe \
	  --run_name scale_deep     --batch_size $(BATCH) --max_iters $(ITERS)

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
	python eval.py --run_name scale_baseline_bpe default_char_char

# ── Sampling ──────────────────────────────────────────────────────────────────
sample:
	python sample.py --run_name default_bpe --prompt "A vulnerability in"

sample-char:
	python sample.py --run_name default_char --prompt "A vulnerability in"

# ── Plots ─────────────────────────────────────────────────────────────────────
plot:
	python plot_loss.py --run_name default_bpe

plot-scale:
	python plot_loss.py --run_name scale_baseline_bpe scale_deep_bpe \
	  --out loss_comparison.png

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "nanoCVE — GPT pretrained on CVE descriptions"
	@echo ""
	@echo "  make prepare          Download + process NVD CVE feeds (~360k CVEs)"
	@echo "  make prepare-small    Quick run with 10k CVEs"
	@echo "  make train-bpe        Train default config with BPE tokenizer"
	@echo "  make train-char       Train default config with char tokenizer"
	@echo "  make train-small      Train tiny config (CPU-friendly)"
	@echo "  make scale            Run baseline + deep configs for scaling exp"
	@echo "  make eval             Compute BPC/perplexity for BPE and char checkpoints"
	@echo "  make sample           Generate text from BPE checkpoint"
	@echo "  make sample-char      Generate text from char checkpoint"
	@echo "  make plot             Regenerate BPE loss_curve.png"
	@echo "  make plot-scale       Regenerate scaling comparison plot"
	@echo ""
	@echo "  Override defaults:  make train-bpe BATCH=32 ITERS=5000"
