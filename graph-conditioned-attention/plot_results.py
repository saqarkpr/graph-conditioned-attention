"""
Plots train/dev UAS (and loss) curves per epoch for baseline vs g2g from a
comparison.csv (written by train.py, columns: model, epoch, train_loss,
train_UAS, dev_loss, dev_UAS), or mean+/-std dev UAS across seeds from a
seed_study_summary.csv.

Usage:
    python plot_results.py --comparison_csv results/baseline/comparison.csv \
        --out results/baseline/uas_curves.png

    python plot_results.py --seed_study_csv results/seed_study_summary.csv \
        --out results/seed_study.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_comparison(csv_path, out_path):
    series = {}  # model -> {epoch, train_loss, train_UAS, dev_loss, dev_UAS}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            m = row["model"]
            series.setdefault(m, {"epoch": [], "train_loss": [], "train_UAS": [], "dev_loss": [], "dev_UAS": []})
            series[m]["epoch"].append(int(row["epoch"]))
            series[m]["train_loss"].append(float(row["train_loss"]))
            series[m]["train_UAS"].append(float(row["train_UAS"]))
            series[m]["dev_loss"].append(float(row["dev_loss"]))
            series[m]["dev_UAS"].append(float(row["dev_UAS"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, d in series.items():
        axes[0].plot(d["epoch"], d["train_loss"], "--", alpha=0.5, label=f"{m} (train)")
        axes[0].plot(d["epoch"], d["dev_loss"], "-", label=f"{m} (dev)")
        axes[1].plot(d["epoch"], d["train_UAS"], "--", alpha=0.5, label=f"{m} (train)")
        axes[1].plot(d["epoch"], d["dev_UAS"], "-", label=f"{m} (dev)")

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("arc loss"); axes[0].set_title("Loss")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("unlabeled attachment accuracy"); axes[1].set_title("UAS")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_seed_study(csv_path, out_path):
    rows = list(csv.DictReader(open(csv_path)))
    mean_row = next(r for r in rows if r["seed"] == "mean")
    std_row = next(r for r in rows if r["seed"] == "std")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    models = ["baseline", "g2g"]
    means = [float(mean_row["baseline_dev_UAS"]), float(mean_row["g2g_dev_UAS"])]
    stds = [float(std_row["baseline_dev_UAS"]), float(std_row["g2g_dev_UAS"])]
    ax.bar(models, means, yerr=stds, capsize=6, color=["#888", "#4c72b0"])
    ax.set_ylabel("dev UAS (mean +/- std across seeds)")
    ax.set_title("Final dev UAS across seeds")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comparison_csv", type=str, default=None)
    p.add_argument("--seed_study_csv", type=str, default=None)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    if args.comparison_csv:
        plot_comparison(args.comparison_csv, args.out)
    elif args.seed_study_csv:
        plot_seed_study(args.seed_study_csv, args.out)
    else:
        raise SystemExit("provide --comparison_csv or --seed_study_csv")


if __name__ == "__main__":
    main()
