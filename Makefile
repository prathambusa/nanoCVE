.PHONY: prepare train-bpe train-char sample scale plot help

# ── Data ──────────────────────────────────────────────────────────────────────
prepare:
	python data/prepare.py

prepare-small:
	python data/prepare.py --limit 10000

# ── Training ──────────────────────────────────────────────────────────────────
train-bpe:
	python train.py --config configs/default.py --tokenizer bpe

train-char:
	python train.py --config configs/default.py --tokenizer char

train-small:
	python train.py --config configs/small.py --tokenizer bpe

# ── Scaling experiment ────────────────────────────────────────────────────────
scale:
	python train.py --config configs/default.py     --tokenizer bpe --run_name scale_baseline
	python train.py --config configs/scale_depth.py --tokenizer bpe --run_name scale_deep

# ── Sampling ──────────────────────────────────────────────────────────────────
sample:
	python sample.py --run_name default_bpe --prompt "A vulnerability in"

# ── Plots ─────────────────────────────────────────────────────────────────────
plot:
	python plot_loss.py --run_name default_bpe

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo "nanoCVE — GPT pretrained on CVE descriptions"
	@echo ""
	@echo "  make prepare        Download + process NVD CVE feeds"
	@echo "  make prepare-small  Quick run with 10k CVEs"
	@echo "  make train-bpe      Train default config with BPE tokenizer"
	@echo "  make train-char     Train default config with char tokenizer"
	@echo "  make train-small    Train tiny config (CPU-friendly)"
	@echo "  make scale          Run baseline + deep configs for scaling exp"
	@echo "  make sample         Generate text from best checkpoint"
	@echo "  make plot           Regenerate loss_curve.png"
