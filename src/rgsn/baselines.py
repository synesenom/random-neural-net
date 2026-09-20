"""Baseline models matched by trainable parameter count (Section 5 of PLAN.md):
a plain MLP as the "decent common neural net", and helpers to build an RGSN
variant with topology frozen (isolates the effect of rewiring) and to size
an MLP to match a target parameter budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """1-2 hidden layer ReLU MLP. `forward(x, hidden_masks=...)` accepts an
    optional list of per-layer boolean/float masks (one per hidden layer) to
    simulate neuron ablation at eval time, matching the RGSN robustness test."""

    def __init__(self, n_features: int, n_classes: int, hidden_sizes: list[int]):
        super().__init__()
        sizes = [n_features] + list(hidden_sizes)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
        self.out = nn.Linear(sizes[-1], n_classes)

    def forward(
        self, x: torch.Tensor, hidden_masks: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.hidden_layers):
            h = torch.relu(layer(h))
            if hidden_masks is not None and hidden_masks[i] is not None:
                h = h * hidden_masks[i]
        return self.out(h)


def mlp_param_count(n_features: int, n_classes: int, hidden_sizes: list[int]) -> int:
    sizes = [n_features] + list(hidden_sizes)
    total = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))
    total += sizes[-1] * n_classes + n_classes
    return total


def hidden_size_for_budget(
    param_budget: int, n_features: int, n_classes: int, n_hidden_layers: int = 1
) -> int:
    """Solve (approximately, for a single hidden layer) for the hidden width
    that makes an MLP's parameter count closest to `param_budget`, then
    verify/adjust by search for >1 hidden layer."""
    if n_hidden_layers == 1:
        h = round((param_budget - n_classes) / (n_features + 1 + n_classes))
        return max(1, h)
    # Small search for multi-layer case.
    best_h, best_diff = 1, float("inf")
    for h in range(1, param_budget):
        hidden_sizes = [h] * n_hidden_layers
        diff = abs(mlp_param_count(n_features, n_classes, hidden_sizes) - param_budget)
        if diff < best_diff:
            best_h, best_diff = h, diff
        if mlp_param_count(n_features, n_classes, hidden_sizes) > param_budget * 1.5:
            break
    return best_h


def random_hidden_ablation_masks(
    hidden_sizes: list[int], fraction: float, gen: torch.Generator
) -> list[torch.Tensor]:
    masks = []
    for h in hidden_sizes:
        n_ablate = int(round(fraction * h))
        mask = torch.ones(h)
        if n_ablate > 0:
            idx = torch.randperm(h, generator=gen)[:n_ablate]
            mask[idx] = 0.0
        masks.append(mask)
    return masks
