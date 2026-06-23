"""
plot_loss.py — Regenerate loss curves from saved CSV logs.

Can plot a single run or overlay multiple runs for comparison
(used for the scaling experiment).

Usage
-----
  python plot_loss.py --run_name default_bpe
  python plot_loss.py --run_name scale_baseline_bpe scale_deep_bpe --out scaling_comparison.png
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def load_csv(csv_path: Path) -> tuple[list, list, list]:
    steps, train_losses, val_losses = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
    return steps, train_losses, val_losses


def plot(run_names: list[str], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.tab10.colors
    for i, run_name in enumerate(run_names):
        run_dir = ROOT / "runs" / run_name
        csv_path = run_dir / "losses.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping {run_name}")
            continue

        steps, train_losses, val_losses = load_csv(csv_path)
        c = colors[i % len(colors)]
        label_base = run_name
        ax.plot(steps, train_losses, color=c, linewidth=2,
                label=f"{label_base} train", alpha=0.9)
        ax.plot(steps, val_losses, color=c, linewidth=2, linestyle="--",
                label=f"{label_base} val", alpha=0.9)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Cross-entropy loss", fontsize=12)
    title = "nanoCVE — Loss Curves" if len(run_names) > 1 else f"nanoCVE — {run_names[0]}"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot nanoCVE loss curves")
    parser.add_argument("--run_name", nargs="+", required=True,
                        help="One or more run names (subdirs of runs/)")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: runs/<run>/loss_curve.png "
                             "for single, or loss_comparison.png for multiple)")
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    elif len(args.run_name) == 1:
        out_path = ROOT / "runs" / args.run_name[0] / "loss_curve.png"
    else:
        out_path = ROOT / "loss_comparison.png"

    plot(args.run_name, out_path)


if __name__ == "__main__":
    main()
