"""Ablation at evaluation time: remove a fraction of hidden neurons, output
neurons, or edges and measure how much accuracy degrades. Used for H3
(robustness). All ablations are temporary and reversible via a context
manager so the same trained model can be re-evaluated at many fractions.
"""

from __future__ import annotations

import torch

from rgsn.network import RandomGraphSNN


class NeuronAblation:
    """Context manager that silences a set of neurons in an RGSN: both their
    incoming and outgoing recurrent edges, and their input projection, are
    temporarily zeroed so they can never spike and never influence others."""

    def __init__(self, net: RandomGraphSNN, neuron_idx: torch.Tensor):
        self.net = net
        self.neuron_idx = neuron_idx

    def __enter__(self):
        self._saved_mask = self.net.mask.clone()
        self._saved_w_in = self.net.W_in.data.clone()
        alive = torch.ones(self.net.n, dtype=torch.bool)
        alive[self.neuron_idx] = False
        alive_matrix = alive.unsqueeze(0) & alive.unsqueeze(1)  # (post, pre) both alive
        self.net.mask = self.net.mask & alive_matrix
        self.net.W_in.data = self.net.W_in.data * alive.unsqueeze(1).float()
        return self.net

    def __exit__(self, *exc):
        self.net.mask = self._saved_mask
        self.net.W_in.data = self._saved_w_in
        return False


class EdgeAblation:
    """Context manager that removes a random fraction of currently active
    edges for the duration of the block."""

    def __init__(self, net: RandomGraphSNN, fraction: float, gen: torch.Generator):
        self.net = net
        self.fraction = fraction
        self.gen = gen

    def __enter__(self):
        self._saved_mask = self.net.mask.clone()
        active_idx = self.net.mask.nonzero(as_tuple=False)
        n_remove = int(round(self.fraction * active_idx.shape[0]))
        if n_remove > 0:
            perm = torch.randperm(active_idx.shape[0], generator=self.gen)[:n_remove]
            chosen = active_idx[perm]
            new_mask = self.net.mask.clone()
            new_mask[chosen[:, 0], chosen[:, 1]] = False
            self.net.mask = new_mask
        return self.net

    def __exit__(self, *exc):
        self.net.mask = self._saved_mask
        return False


def random_neuron_subset(
    n: int, fraction: float, gen: torch.Generator, exclude: torch.Tensor | None = None
) -> torch.Tensor:
    candidates = torch.arange(n)
    if exclude is not None:
        keep = torch.ones(n, dtype=torch.bool)
        keep[exclude] = False
        candidates = candidates[keep]
    n_ablate = int(round(fraction * candidates.numel()))
    if n_ablate == 0:
        return torch.empty(0, dtype=torch.long)
    perm = torch.randperm(candidates.numel(), generator=gen)[:n_ablate]
    return candidates[perm]
