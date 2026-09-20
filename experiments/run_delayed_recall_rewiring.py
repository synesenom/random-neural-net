"""Does DEEP-R structural rewiring help RGSN's delayed-recall memory,
especially at long delays where the plain (no-rewiring) network's memory
degrades?

Everything so far used a static topology (no rewiring). This runs the full
2x2 factorial -- {no rewiring, DEEP-R rewiring} x {population-rate readout,
trained linear readout} -- across delays, to separate two different
questions:
  - Does rewiring change what the *recurrent dynamics* can represent at
    long delays? (isolated by the linear-readout condition, which removes
    the fixed-output-group bottleneck identified earlier.)
  - Does rewiring change how *reliably the population-rate readout* can
    read out whatever is there? (the population-rate condition.)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from rgsn.config import GraphConfig, NeuronConfig
from rgsn.decoding import LinearReadoutDecoder, PopulationRateDecoder
from rgsn.delayed_recall import RecallTaskSpec, make_loaders
from rgsn.graph import build_graph
from rgsn.network import RandomGraphSNN
from rgsn.rewiring import deep_r_step
from rgsn.utils import seed_everything

N_CLASSES = 2


def train_rgsn(
    spec: RecallTaskSpec,
    seed: int,
    epochs: int,
    lr: float,
    grad_clip: float,
    n_train: int,
    n_test: int,
    batch_size: int,
    decoder_scheme: str,
    rewiring_enabled: bool,
    l1_weight: float = 0.0005,
    init_weight_scale: float = 0.02,
):
    gen = seed_everything(seed)
    gcfg = GraphConfig(
        n=150, p=0.1, input_fraction=0.2, output_group_size=8, n_classes=N_CLASSES, gain=4.0
    )
    ncfg = NeuronConfig(tau_mem_ms=20.0, dt_ms=1.0, v_th=1.0, surrogate_slope=1.0)
    graph_spec = build_graph(gcfg, gen)
    net = RandomGraphSNN(graph_spec, ncfg, n_features=spec.n_features, gain=gcfg.gain, seed=seed)
    if decoder_scheme == "linear_readout":
        decoder = LinearReadoutDecoder(net.n, N_CLASSES)
    else:
        decoder = PopulationRateDecoder(graph_spec.output_idx, learnable_temperature=True)
    params = list(net.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader, test_loader = make_loaders(spec, n_train, n_test, batch_size, seed)

    def run_epoch(loader, train: bool):
        correct, total = 0, 0
        n_pruned_total, n_edges_final = 0, net.edge_count()
        for x, y in loader:
            spikes = net(x)
            logits = decoder(spikes[:, spec.query_start :, :])
            loss = loss_fn(logits, y)
            if rewiring_enabled and l1_weight > 0:
                loss = loss + l1_weight * net.W_mag[net.mask].abs().mean()
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
                if rewiring_enabled:
                    stats = deep_r_step(net, init_weight_scale, gen)
                    n_pruned_total += stats["n_pruned"]
                    n_edges_final = stats["n_edges"]
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
        return correct / total, n_pruned_total, n_edges_final

    total_pruned = 0
    for _ in range(epochs):
        _, pruned, edges_final = run_epoch(train_loader, train=True)
        total_pruned += pruned
    test_acc, _, _ = run_epoch(test_loader, train=False)
    param_count = sum(p.numel() for p in params if p.requires_grad)
    return test_acc, param_count, total_pruned, edges_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delays", nargs="+", type=int, default=[0, 20, 40, 80])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default="runs/delayed_recall_rewiring/summary.csv")
    args = parser.parse_args()

    rows = []
    t0 = time.time()
    for delay in args.delays:
        spec = RecallTaskSpec(cue_steps=5, delay=delay, query_steps=5)
        for decoder_scheme in ["population_rate", "linear_readout"]:
            for rewiring_enabled in [False, True]:
                for seed in args.seeds:
                    acc, params, pruned, edges = train_rgsn(
                        spec,
                        seed,
                        args.epochs,
                        lr=0.01,
                        grad_clip=5.0,
                        n_train=args.n_train,
                        n_test=args.n_test,
                        batch_size=args.batch_size,
                        decoder_scheme=decoder_scheme,
                        rewiring_enabled=rewiring_enabled,
                    )
                    row = {
                        "delay": delay,
                        "decoder": decoder_scheme,
                        "rewiring": rewiring_enabled,
                        "seed": seed,
                        "test_acc": acc,
                        "params": params,
                        "total_pruned": pruned,
                        "final_edges": edges,
                    }
                    rows.append(row)
                    print(row, f"t={time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.groupby(["decoder", "rewiring", "delay"])["test_acc"].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
