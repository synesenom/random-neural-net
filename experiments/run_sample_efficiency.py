"""H1 (sample efficiency) + H2 (learning speed) driver.

Trains every model kind at several training-set sizes and several seeds,
using the same base config (`configs/rgsn_mnist.yaml`) and the same
epoch/tuning budget for every model, per PLAN.md Section 7 / "Unfair
comparisons" risk mitigation. Each run's full step-by-step and epoch-by-epoch
CSVs (saved by `train_run`) already give H2's learning-speed curves; this
script additionally writes one consolidated summary CSV across all runs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from rgsn.config import RunConfig
from rgsn.train import train_run
from rgsn.utils import dataclass_from_dict, load_yaml

MODELS = ["rgsn", "rgsn_no_rewire", "reservoir", "mlp"]
TRAIN_SIZES = [200, 500, 1000, 2000, 4000]
SEEDS = [0, 1, 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/rgsn_mnist.yaml")
    parser.add_argument("--out-root", default="runs/h1_sample_efficiency")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--sizes", nargs="+", type=int, default=TRAIN_SIZES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    base = load_yaml(args.base_config)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    t_start = time.time()
    for model in args.models:
        for n_train in args.sizes:
            for seed in args.seeds:
                raw = {**base}
                raw["model"] = model
                raw["train"] = {**base["train"], "n_train_samples": n_train, "seed": seed}
                if args.epochs is not None:
                    raw["train"]["epochs"] = args.epochs
                if model == "mlp":
                    raw["train"]["lr"] = 0.001  # MLP is far better conditioned; same epoch budget.
                if model == "reservoir":
                    # PLAN.md 5.3: fixed reservoir + trained *linear* readout, not population code.
                    raw["decoding"] = {**base["decoding"], "scheme": "linear_readout"}
                raw["name"] = f"{model}_n{n_train}_s{seed}"
                cfg = dataclass_from_dict(RunConfig, raw)
                run_dir = out_root / raw["name"]
                print(f">>> {raw['name']}")
                t0 = time.time()
                train_run(cfg, run_dir)
                elapsed = time.time() - t0

                epochs_df = pd.read_csv(run_dir / "epochs.csv")
                tail = epochs_df.tail(3)
                rows.append(
                    {
                        "model": model,
                        "n_train": n_train,
                        "seed": seed,
                        "final_test_acc": tail["test_acc"].mean(),
                        "final_test_acc_last": epochs_df["test_acc"].iloc[-1],
                        "best_test_acc": epochs_df["test_acc"].max(),
                        "param_total": epochs_df["param_total"].iloc[-1],
                        "param_nonzero": epochs_df["param_nonzero"].iloc[-1],
                        "wall_time_s": elapsed,
                        "n_epochs": len(epochs_df),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "summary.csv", index=False)
    print(f"Total time: {time.time() - t_start:.1f}s")
    print(summary)


if __name__ == "__main__":
    main()
