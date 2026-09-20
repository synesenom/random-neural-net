"""Generate the H1/H2/H3 plots from saved CSVs only (never from in-memory
training state, per PLAN.md Section 9)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODEL_LABELS = {
    "rgsn": "RGSN (full + rewiring)",
    "rgsn_no_rewire": "RGSN (weights only)",
    "reservoir": "Fixed reservoir + readout",
    "mlp": "MLP (param-matched)",
}
MODEL_COLORS = {
    "rgsn": "#1b9e77",
    "rgsn_no_rewire": "#66c2a5",
    "reservoir": "#7570b3",
    "mlp": "#d95f02",
}

RECALL_LABELS = {
    "rgsn": "RGSN, population-rate readout",
    "rgsn_linear_readout": "RGSN, trained linear readout",
    "memoryless_mlp": "Memoryless MLP (online, no history)",
    "windowed_mlp": "Windowed MLP (offline, full sequence)",
}
RECALL_COLORS = {
    "rgsn": "#1b9e77",
    "rgsn_linear_readout": "#66c2a5",
    "memoryless_mlp": "#d95f02",
    "windowed_mlp": "#7570b3",
}


def plot_sample_efficiency(summary_csv: Path, out_dir: Path):
    df = pd.read_csv(summary_csv)
    agg = df.groupby(["model", "n_train"])["final_test_acc"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model in MODEL_LABELS:
        sub = agg[agg.model == model].sort_values("n_train")
        if sub.empty:
            continue
        ax.errorbar(
            sub.n_train,
            sub["mean"],
            yerr=sub["std"].fillna(0),
            marker="o",
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Test accuracy (mean ± std over seeds)")
    ax.set_title("H1: Sample efficiency (MNIST)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "h1_sample_efficiency.png", dpi=150)
    plt.close(fig)


def plot_learning_speed(h1_root: Path, n_train: int, seed: int, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model in MODEL_LABELS:
        run_dir = h1_root / f"{model}_n{n_train}_s{seed}"
        epochs_csv = run_dir / "epochs.csv"
        if not epochs_csv.exists():
            continue
        df = pd.read_csv(epochs_csv)
        ax.plot(
            df.epoch, df.test_acc, marker=".", label=MODEL_LABELS[model], color=MODEL_COLORS[model]
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"H2: Learning speed (n_train={n_train}, seed={seed})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "h2_learning_speed.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model in MODEL_LABELS:
        run_dir = h1_root / f"{model}_n{n_train}_s{seed}"
        epochs_csv = run_dir / "epochs.csv"
        if not epochs_csv.exists():
            continue
        df = pd.read_csv(epochs_csv)
        ax.plot(
            df.wall_time_s,
            df.test_acc,
            marker=".",
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"H2: Learning speed vs wall time (n_train={n_train}, seed={seed})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "h2_learning_speed_walltime.png", dpi=150)
    plt.close(fig)


def plot_ablation(ablation_csv: Path, out_dir: Path):
    df = pd.read_csv(ablation_csv)
    for target in df.target.unique():
        sub_t = df[df.target == target]
        agg = sub_t.groupby(["model", "fraction"])["acc"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for model in MODEL_LABELS:
            sub = agg[agg.model == model].sort_values("fraction")
            if sub.empty:
                continue
            ax.errorbar(
                sub.fraction,
                sub["mean"],
                yerr=sub["std"].fillna(0),
                marker="o",
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
            )
        ax.set_xlabel(f"Fraction of {target} units/edges removed")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"H3: Robustness to {target} ablation")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"h3_ablation_{target}.png", dpi=150)
        plt.close(fig)


def plot_delayed_recall(recall_csv: Path, out_dir: Path):
    df = pd.read_csv(recall_csv)
    agg = df.groupby(["model", "delay"])["test_acc"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model in RECALL_LABELS:
        sub = agg[agg.model == model].sort_values("delay")
        if sub.empty:
            continue
        ax.errorbar(
            sub.delay,
            sub["mean"],
            yerr=sub["std"].fillna(0),
            marker="o",
            label=RECALL_LABELS[model],
            color=RECALL_COLORS[model],
        )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel("Delay between cue and query (timesteps)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Delayed recall: online memory vs. delay length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.4, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "delayed_recall.png", dpi=150)
    plt.close(fig)


def plot_shrink(shrink_csv: Path, out_dir: Path):
    df = pd.read_csv(shrink_csv)
    rgsn = df[df.model == "rgsn"]
    agg = rgsn.groupby("n")[["test_acc", "params", "train_time_s"]].agg(["mean", "std"])
    mlp = df[df.model == "windowed_mlp"]
    mlp_acc, mlp_params, mlp_time = (
        mlp.test_acc.mean(),
        mlp.params.mean(),
        mlp.train_time_s.mean(),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.errorbar(
        agg.index,
        agg[("test_acc", "mean")],
        yerr=agg[("test_acc", "std")].fillna(0),
        marker="o",
        color="#1b9e77",
        label="RGSN",
    )
    ax.axhline(mlp_acc, color="#7570b3", linestyle="--", label="Windowed MLP")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
    ax.set_xlabel("N (neurons)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. network size (delay=20)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(agg.index, agg[("params", "mean")], marker="o", color="#1b9e77", label="RGSN")
    ax.axhline(mlp_params, color="#7570b3", linestyle="--", label="Windowed MLP")
    ax.set_xlabel("N (neurons)")
    ax.set_ylabel("Trainable parameters")
    ax.set_yscale("log")
    ax.set_title("Parameter count vs. network size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(agg.index, agg[("train_time_s", "mean")], marker="o", color="#1b9e77", label="RGSN")
    ax.axhline(mlp_time, color="#7570b3", linestyle="--", label="Windowed MLP")
    ax.set_xlabel("N (neurons)")
    ax.set_ylabel("Training wall-clock time (s)")
    ax.set_title("Training cost vs. network size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "shrink_delayed_recall.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-root", default="runs/h1_sample_efficiency")
    parser.add_argument("--h3-csv", default="runs/h3_ablation/summary.csv")
    parser.add_argument("--recall-csv", default="runs/delayed_recall/summary.csv")
    parser.add_argument("--shrink-csv", default="runs/shrink_delayed_recall/summary.csv")
    parser.add_argument("--out-dir", default="runs/plots")
    parser.add_argument("--learning-speed-n", type=int, default=4000)
    parser.add_argument("--learning-speed-seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h1_root = Path(args.h1_root)

    summary_csv = h1_root / "summary.csv"
    if summary_csv.exists():
        plot_sample_efficiency(summary_csv, out_dir)
        plot_learning_speed(h1_root, args.learning_speed_n, args.learning_speed_seed, out_dir)

    h3_csv = Path(args.h3_csv)
    if h3_csv.exists():
        plot_ablation(h3_csv, out_dir)

    recall_csv = Path(args.recall_csv)
    if recall_csv.exists():
        plot_delayed_recall(recall_csv, out_dir)

    shrink_csv = Path(args.shrink_csv)
    if shrink_csv.exists():
        plot_shrink(shrink_csv, out_dir)

    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
