"""H3 (robustness) driver: load trained checkpoints from the H1 sample-
efficiency runs (largest training set, one row per seed) and measure test
accuracy as a function of the fraction of hidden neurons, output neurons, or
edges removed at evaluation time. RGSN's population coding is compared
against the MLP baseline's hidden-unit ablation (its closest analogue: "zero
hidden units" per PLAN.md Section 7)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from rgsn.baselines import hidden_size_for_budget
from rgsn.data import get_mnist_loaders
from rgsn.evaluate import (
    mlp_hidden_ablation_curve,
    rgsn_edge_ablation_curve,
    rgsn_neuron_ablation_curve,
)
from rgsn.train import N_CLASSES_MNIST, N_FEATURES_MNIST, load_model

FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-root", default="runs/h1_sample_efficiency")
    parser.add_argument("--n-train", type=int, default=4000)
    default_models = ["rgsn", "rgsn_no_rewire", "reservoir", "mlp"]
    parser.add_argument("--models", nargs="+", default=default_models)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default="runs/h3_ablation/summary.csv")
    args = parser.parse_args()

    h1_root = Path(args.h1_root)
    rows = []
    for model_kind in args.models:
        for seed in args.seeds:
            run_dir = h1_root / f"{model_kind}_n{args.n_train}_s{seed}"
            if not (run_dir / "checkpoint.pt").exists():
                print(f"skip missing {run_dir}")
                continue
            model, cfg = load_model(run_dir)
            gen = torch.Generator().manual_seed(1000 + seed)
            _, test_loader = get_mnist_loaders(
                n_train=cfg.train.n_train_samples,
                n_test=cfg.train.n_test_samples,
                batch_size=cfg.train.batch_size,
                seed=cfg.train.seed,
            )

            def add_rows(curve: dict[float, float], target: str) -> None:
                for frac, acc in curve.items():
                    row = {
                        "model": model_kind,
                        "seed": seed,
                        "target": target,
                        "fraction": frac,
                        "acc": acc,
                    }
                    rows.append(row)

            if model.mlp is not None:
                h = hidden_size_for_budget(
                    cfg.train.mlp_param_budget, N_FEATURES_MNIST, N_CLASSES_MNIST
                )
                curve = mlp_hidden_ablation_curve(
                    model, test_loader, FRACTIONS, gen, hidden_sizes=[h]
                )
                add_rows(curve, "hidden")
            else:
                hidden_curve = rgsn_neuron_ablation_curve(
                    model, test_loader, FRACTIONS, gen, target="hidden"
                )
                output_curve = rgsn_neuron_ablation_curve(
                    model, test_loader, FRACTIONS, gen, target="output"
                )
                edge_curve = rgsn_edge_ablation_curve(model, test_loader, FRACTIONS, gen)
                add_rows(hidden_curve, "hidden")
                add_rows(output_curve, "output")
                add_rows(edge_curve, "edge")
            print(f"done {model_kind} seed={seed}")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.groupby(["model", "target", "fraction"])["acc"].mean().unstack())


if __name__ == "__main__":
    main()
