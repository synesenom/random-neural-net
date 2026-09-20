"""Delayed-recall experiment: sweep the gap between when a value is shown and
when it must be reported, and compare:
  - RGSN (recurrent spiking, online: one timestep of input at a time)
  - a memoryless per-timestep MLP (online, but provably chance-level for any
    delay since its output at query time cannot depend on x[t < query])
  - a windowed MLP (offline: sees the whole flattened sequence at once, an
    upper-bound sanity check, not a fair online baseline)

This tests PLAN.md's H4 (temporal tasks) hypothesis without needing the SHD
dataset: RGSN's recurrent/leaky state is a *structural* source of memory
that a stateless feedforward net cannot have no matter how it's trained.
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
from rgsn.delayed_recall import (
    MemorylessClassifier,
    RecallTaskSpec,
    WindowedClassifier,
    make_loaders,
)
from rgsn.graph import build_graph
from rgsn.network import RandomGraphSNN
from rgsn.utils import seed_everything

N_CLASSES = 2


def train_rgsn(
    spec: RecallTaskSpec,
    seed: int,
    epochs: int,
    lr: float,
    grad_clip: float,
    n_train,
    n_test,
    batch_size,
):
    gen = seed_everything(seed)
    gcfg = GraphConfig(
        n=150, p=0.1, input_fraction=0.2, output_group_size=8, n_classes=N_CLASSES, gain=4.0
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
        correct, total, loss_sum = 0, 0, 0.0
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
            loss_sum += loss.item() * y.numel()
        return correct / total, loss_sum / total

    for _ in range(epochs):
        run_epoch(train_loader, train=True)
    test_acc, _ = run_epoch(test_loader, train=False)
    param_count = sum(p.numel() for p in params if p.requires_grad)
    return test_acc, param_count


def train_memoryless(spec, seed, epochs, lr, n_train, n_test, batch_size, hidden=32):
    seed_everything(seed)
    model = MemorylessClassifier(spec.n_features, N_CLASSES, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader, test_loader = make_loaders(spec, n_train, n_test, batch_size, seed)

    def run_epoch(loader, train: bool):
        correct, total = 0, 0
        for x, y in loader:
            logits_t = model(x[:, spec.query_start :, :])  # (B, query_steps, C)
            logits_mean = logits_t.mean(dim=1)
            y_rep = y.unsqueeze(1).expand(-1, logits_t.shape[1]).reshape(-1)
            loss = loss_fn(logits_t.reshape(-1, N_CLASSES), y_rep)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            correct += (logits_mean.argmax(dim=1) == y).sum().item()
            total += y.numel()
        return correct / total

    for _ in range(epochs):
        run_epoch(train_loader, train=True)
    test_acc = run_epoch(test_loader, train=False)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return test_acc, param_count


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

    for _ in range(epochs):
        run_epoch(train_loader, train=True)
    test_acc = run_epoch(test_loader, train=False)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return test_acc, param_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delays", nargs="+", type=int, default=[0, 10, 20, 40, 80])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default="runs/delayed_recall/summary.csv")
    args = parser.parse_args()

    rows = []
    t0 = time.time()
    for delay in args.delays:
        spec = RecallTaskSpec(cue_steps=5, delay=delay, query_steps=5)
        for seed in args.seeds:
            acc, params = train_rgsn(
                spec,
                seed,
                args.epochs,
                lr=0.01,
                grad_clip=5.0,
                n_train=args.n_train,
                n_test=args.n_test,
                batch_size=args.batch_size,
            )
            rows.append(
                {"model": "rgsn", "delay": delay, "seed": seed, "test_acc": acc, "params": params}
            )
            print(rows[-1], f"t={time.time() - t0:.0f}s")

            acc, params = train_memoryless(
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
                    "model": "memoryless_mlp",
                    "delay": delay,
                    "seed": seed,
                    "test_acc": acc,
                    "params": params,
                }
            )
            print(rows[-1], f"t={time.time() - t0:.0f}s")

            acc, params = train_windowed(
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
                    "delay": delay,
                    "seed": seed,
                    "test_acc": acc,
                    "params": params,
                }
            )
            print(rows[-1], f"t={time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.groupby(["model", "delay"])["test_acc"].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
