"""Evaluation helpers: plain accuracy plus ablation curves (H3) that remove a
fraction of hidden neurons / output neurons / edges (RGSN) or hidden units
(MLP baseline) and re-measure test accuracy."""

from __future__ import annotations

import torch

from rgsn.ablation import EdgeAblation, NeuronAblation, random_neuron_subset
from rgsn.baselines import random_hidden_ablation_masks
from rgsn.train import Model, evaluate


def rgsn_neuron_ablation_curve(
    model: Model,
    loader,
    fractions: list[float],
    gen: torch.Generator,
    target: str = "hidden",
    max_batches: int | None = None,
) -> dict[float, float]:
    """target: "hidden" (any non-input, non-output neuron) or "output" (output
    neurons only, still respecting population coding's graceful degradation)."""
    net = model.net
    all_output = net.output_idx.flatten()
    if target == "output":
        pool = all_output
        exclude = None
    else:
        exclude = torch.cat([net.input_idx, all_output])
        pool = None

    results = {}
    for frac in fractions:
        if frac == 0.0:
            results[frac] = evaluate(model, loader, max_batches)
            continue
        if target == "output":
            n_ablate = int(round(frac * pool.numel()))
            idx = pool[torch.randperm(pool.numel(), generator=gen)[:n_ablate]]
        else:
            idx = random_neuron_subset(net.n, frac, gen, exclude=exclude)
        with NeuronAblation(net, idx):
            results[frac] = evaluate(model, loader, max_batches)
    return results


def rgsn_edge_ablation_curve(
    model: Model,
    loader,
    fractions: list[float],
    gen: torch.Generator,
    max_batches: int | None = None,
) -> dict[float, float]:
    results = {}
    for frac in fractions:
        if frac == 0.0:
            results[frac] = evaluate(model, loader, max_batches)
            continue
        with EdgeAblation(model.net, frac, gen):
            results[frac] = evaluate(model, loader, max_batches)
    return results


def mlp_hidden_ablation_curve(
    model: Model,
    loader,
    fractions: list[float],
    gen: torch.Generator,
    hidden_sizes: list[int],
    max_batches: int | None = None,
) -> dict[float, float]:
    results = {}
    for frac in fractions:
        masks = None if frac == 0.0 else random_hidden_ablation_masks(hidden_sizes, frac, gen)
        correct, total = 0, 0
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            with torch.no_grad():
                logits = model.mlp(x, hidden_masks=masks)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
        results[frac] = correct / max(1, total)
    return results
