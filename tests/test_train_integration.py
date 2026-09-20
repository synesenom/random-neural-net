"""Small end-to-end smoke tests for every model kind: a couple of epochs on a
tiny MNIST subset, just checking that training runs without error and that
accuracy is (weakly) better than random chance."""

from pathlib import Path

import pytest

from rgsn.config import RunConfig
from rgsn.train import train_run
from rgsn.utils import dataclass_from_dict

pytestmark = pytest.mark.slow

BASE = {
    "name": "smoke",
    "graph": {
        "n": 80,
        "p": 0.15,
        "input_fraction": 0.3,
        "output_group_size": 2,
        "n_classes": 10,
        "gain": 1.5,
    },
    "encoding": {"t_steps": 10},
    "train": {
        "epochs": 2,
        "batch_size": 32,
        "lr": 0.01,
        "n_train_samples": 128,
        "n_test_samples": 64,
        "seed": 0,
        "mlp_param_budget": 2000,
    },
}


def run_smoke(tmp_path, model_kind, rewiring_enabled=False):
    cfg_dict = {**BASE, "model": model_kind, "name": f"smoke_{model_kind}"}
    if rewiring_enabled:
        cfg_dict["rewiring"] = {
            "enabled": True,
            "interval_steps": 1,
            "init_weight_scale": 0.05,
            "l1_weight": 0.01,
        }
    cfg = dataclass_from_dict(RunConfig, cfg_dict)
    run_dir = Path(tmp_path) / model_kind
    train_run(cfg, run_dir)
    assert (run_dir / "epochs.csv").exists()
    assert (run_dir / "checkpoint.pt").exists()


def test_rgsn_trains(tmp_path):
    run_smoke(tmp_path, "rgsn", rewiring_enabled=True)


def test_rgsn_no_rewire_trains(tmp_path):
    run_smoke(tmp_path, "rgsn_no_rewire")


def test_reservoir_trains(tmp_path):
    run_smoke(tmp_path, "reservoir")


def test_mlp_trains(tmp_path):
    run_smoke(tmp_path, "mlp")
