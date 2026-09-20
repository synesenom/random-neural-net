"""Random directed graph generation for the RGSN reservoir.

Produces a boolean adjacency mask plus derived input/output neuron subsets.
Weights themselves live in `network.py` (they are learnable parameters);
this module only decides *which* entries of the N x N weight matrix are
structurally allowed to be nonzero, and which neurons are I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rgsn.config import GraphConfig


@dataclass
class GraphSpec:
    mask: torch.Tensor  # (N, N) bool, mask[i, j] = edge j -> i (row = post, col = pre)
    sign: torch.Tensor  # (N,) float, +1/-1 per presynaptic neuron (Dale's law) or all 1
    input_idx: torch.Tensor  # (N_in,) long
    output_idx: torch.Tensor  # (n_classes, k) long, disjoint groups
    n: int


def _erdos_renyi_mask(n: int, p: float, self_loops: bool, gen: torch.Generator) -> torch.Tensor:
    mask = torch.rand((n, n), generator=gen) < p
    if not self_loops:
        mask.fill_diagonal_(False)
    return mask


def _watts_strogatz_mask(
    n: int, k: int, beta: float, self_loops: bool, gen: torch.Generator
) -> torch.Tensor:
    """Directed small-world graph: ring lattice with k nearest neighbors (each
    direction), then random rewiring of each edge with probability beta."""
    mask = torch.zeros((n, n), dtype=torch.bool)
    half = max(1, k // 2)
    idx = torch.arange(n)
    for offset in range(1, half + 1):
        mask[idx, (idx + offset) % n] = True
        mask[(idx + offset) % n, idx] = True
    edges = mask.nonzero(as_tuple=False)
    rewire = torch.rand(edges.shape[0], generator=gen) < beta
    for e in range(edges.shape[0]):
        if rewire[e]:
            i, j = edges[e, 0].item(), edges[e, 1].item()
            mask[i, j] = False
            new_j = torch.randint(0, n, (1,), generator=gen).item()
            tries = 0
            while (new_j == i or mask[i, new_j]) and tries < 10:
                new_j = torch.randint(0, n, (1,), generator=gen).item()
                tries += 1
            mask[i, new_j] = True
    if not self_loops:
        mask.fill_diagonal_(False)
    return mask


def build_graph(cfg: GraphConfig, gen: torch.Generator) -> GraphSpec:
    n = cfg.n
    if cfg.graph_type == "erdos_renyi":
        mask = _erdos_renyi_mask(n, cfg.p, cfg.self_loops, gen)
    elif cfg.graph_type == "watts_strogatz":
        mask = _watts_strogatz_mask(n, cfg.ws_k, cfg.ws_beta, cfg.self_loops, gen)
    else:
        raise ValueError(f"Unknown graph_type: {cfg.graph_type}")

    if cfg.dales_law:
        n_exc = int(round(cfg.excitatory_fraction * n))
        perm = torch.randperm(n, generator=gen)
        sign = torch.ones(n)
        sign[perm[n_exc:]] = -1.0
    else:
        sign = torch.ones(n)

    n_in = max(1, int(round(cfg.input_fraction * n)))
    n_out = cfg.n_classes * cfg.output_group_size
    if n_in + n_out > n:
        raise ValueError(f"input_fraction*N ({n_in}) + n_classes*k ({n_out}) exceeds N ({n})")
    perm = torch.randperm(n, generator=gen)
    input_idx = perm[:n_in].clone()
    output_flat = perm[n_in : n_in + n_out].clone()
    output_idx = output_flat.view(cfg.n_classes, cfg.output_group_size)

    return GraphSpec(mask=mask, sign=sign, input_idx=input_idx, output_idx=output_idx, n=n)
