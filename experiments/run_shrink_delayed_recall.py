"""How small can RGSN be and still match the windowed ("cheating", full-
sequence-access) MLP on the delayed-recall task, and how much cheaper is
training it at that size?

Fixes delay=20 (~1 membrane time constant), the regime where RGSN reliably
matches the windowed MLP with the plain population-rate decoder (see
README). Sweeps the neuron count N down, keeping density and output-group
size fixed in proportion to N so the graph stays connected and the readout
still fits, and reports test accuracy, trainable parameter count, and
training wall-clock time at each size against the windowed MLP baseline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from rgsn.config import GraphConfig, NeuronConfig
from rgsn.decoding import PopulationRateDecoder
from rgsn.delayed_recall import RecallTaskSpec, WindowedClassifier, make_loaders
from rgsn.graph import build_graph
from rgsn.network import RandomGraphSNN
from rgsn.utils import seed_everything

N_CLASSES = 2


def density_for_n(n: int, target_avg_degree: float = 12.0) -> float:
    """Keep average out-degree roughly constant as N shrinks, instead of a
    fixed edge probability, so small graphs don't become disconnected."""
    return min(0.6, target_avg_degree / max(1, n - 1))


def output_group_size_for_n(n: int) -> int:
    """Output group size shrinks with N too (fixed neuron *counts* would
    eventually exceed N), floored at 2 neurons/class."""
    return max(2, round(0.05 * n))


def train_rgsn_sized(
    spec: RecallTaskSpec, n: int, seed: int, epochs, lr, grad_clip, n_train, n_test, batch_size
):
    gen = seed_everything(seed)
    k = output_group_size_for_n(n)
    gcfg = GraphConfig(
        n=n,
        p=density_for_n(n),
        input_fraction=0.2,
        output_group_size=k,
        n_classes=N_CLASSES,
        gain=4.0,
    )
    ncfg = NeuronConfig(tau_mem_ms=20.0, dt_ms=1.0, v_th=1.0, surrogate_slope=1.0)
    graph_spec = build_graph(gcfg, gen)
    net = RandomGraphSNN(graph_spec, ncfg, n_features=spec.n_features, gain=gcfg.gain, seed=seed)
    decoder = PopulationRateDecoder(graph_spec.output_idx, learnable_temperature=True)
    params = list(net.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader, test_loader = make_loaders(spec, n_train, n_test, batch_size, seed)

    def run_epoch(loader, train: bool):
        correct, total = 0, 0
        for x, y in loader:
            spikes = net(x)
            logits = decoder(spikes[:, spec.query_start :, :])
            loss = loss_fn(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
        return correct / total

    t0 = time.time()
    for _ in range(epochs):
        run_epoch(train_loader, train=True)
    train_time = time.time() - t0
    test_acc = run_epoch(test_loader, train=False)
    param_count = sum(p.numel() for p in params if p.requires_grad)
    return test_acc, param_count, train_time, net.edge_count()


def train_windowed(spec, seed, epochs, lr, n_train, n_test, batch_size, hidden=32):
    seed_everything(seed)
    model = WindowedClassifier(spec.t_total, spec.n_features, N_CLASSES, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader, test_loader = make_loaders(spec, n_train, n_test, batch_size, seed)

    def run_epoch(loader, train: bool):
        correct, total = 0, 0
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
        return correct / total

    t0 = time.time()
    for _ in range(epochs):
        run_epoch(train_loader, train=True)
    train_time = time.time() - t0
    test_acc = run_epoch(test_loader, train=False)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return test_acc, param_count, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[150, 100, 75, 50, 30, 20, 15, 12])
    parser.add_argument("--delay", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default="runs/shrink_delayed_recall/summary.csv")
    args = parser.parse_args()

    spec = RecallTaskSpec(cue_steps=5, delay=args.delay, query_steps=5)
    rows = []
    t0 = time.time()

    for seed in args.seeds:
        acc, params, train_time = train_windowed(
            spec,
            seed,
            args.epochs,
            lr=0.01,
            n_train=args.n_train,
            n_test=args.n_test,
            batch_size=args.batch_size,
        )
        rows.append(
            {
                "model": "windowed_mlp",
                "n": None,
                "seed": seed,
                "test_acc": acc,
                "params": params,
                "edges": None,
                "train_time_s": train_time,
            }
        )
        print(rows[-1], f"t={time.time() - t0:.0f}s")

    for n in args.sizes:
        for seed in args.seeds:
            acc, params, train_time, edges = train_rgsn_sized(
                spec,
                n,
                seed,
                args.epochs,
                lr=0.01,
                grad_clip=5.0,
                n_train=args.n_train,
                n_test=args.n_test,
                batch_size=args.batch_size,
            )
            rows.append(
                {
                    "model": "rgsn",
                    "n": n,
                    "seed": seed,
                    "test_acc": acc,
                    "params": params,
                    "edges": edges,
                    "train_time_s": train_time,
                }
            )
            print(rows[-1], f"t={time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.groupby(["model", "n"], dropna=False)[["test_acc", "params", "train_time_s"]].mean())


if __name__ == "__main__":
    main()
