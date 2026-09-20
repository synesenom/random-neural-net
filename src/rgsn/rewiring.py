"""DEEP-R-style structural plasticity: prune edges whose magnitude has
collapsed to (near) zero, then regrow the same number of edges elsewhere,
keeping the total active-edge count constant.

Two regrowth strategies:
  - "deep_r": regrow at uniformly random inactive positions (Bellec et al. 2018).
  - "rigl": regrow at the inactive positions with the largest dense gradient
    magnitude (Evci et al. 2020), which requires a dense gradient w.r.t. the
    full N x N weight matrix computed just before the call.
"""

from __future__ import annotations

import torch

from rgsn.network import RandomGraphSNN


@torch.no_grad()
def prune_collapsed_edges(net: RandomGraphSNN, eps: float = 1e-6) -> torch.Tensor:
    """Any active edge whose magnitude parameter has been pushed to <= eps by
    gradient descent is structurally removed (softplus(W_mag) -> ~0 current).
    Returns a bool mask of newly pruned positions."""
    active = net.mask
    collapsed = active & (net.W_mag <= eps)
    if collapsed.any():
        net.mask = net.mask & ~collapsed
        net.W_mag.data[collapsed] = 0.0
        net.edge_sign.data[collapsed] = 0.0
    return collapsed


@torch.no_grad()
def regrow_random(
    net: RandomGraphSNN, n_regrow: int, init_weight_scale: float, gen: torch.Generator
) -> int:
    """Regrow `n_regrow` edges at uniformly random currently-inactive, allowed
    (non-self-loop) positions. Returns the number actually regrown (may be
    less than requested if too few candidates remain)."""
    candidates = (~net.mask).clone()
    candidates.fill_diagonal_(False)
    idx = candidates.nonzero(as_tuple=False)
    if idx.shape[0] == 0 or n_regrow <= 0:
        return 0
    n_regrow = min(n_regrow, idx.shape[0])
    perm = torch.randperm(idx.shape[0], generator=gen)[:n_regrow]
    chosen = idx[perm]
    rows, cols = chosen[:, 0], chosen[:, 1]
    net.mask[rows, cols] = True
    new_sign = torch.randint(0, 2, (n_regrow,), generator=gen).float() * 2 - 1
    net.edge_sign.data[rows, cols] = new_sign
    new_mag = torch.rand(n_regrow, generator=gen) * init_weight_scale
    net.W_mag.data[rows, cols] = new_mag
    return n_regrow


@torch.no_grad()
def regrow_rigl(
    net: RandomGraphSNN,
    n_regrow: int,
    init_weight_scale: float,
    dense_grad: torch.Tensor,
) -> int:
    """Regrow at the `n_regrow` currently-inactive positions with the largest
    |dense_grad|. `dense_grad` must be the gradient of the loss w.r.t. a dense
    (unmasked) copy of the recurrent weight matrix, shape (N, N)."""
    n = net.n
    candidates = (~net.mask).clone()
    candidates.fill_diagonal_(False)
    if n_regrow <= 0 or not candidates.any():
        return 0
    scores = dense_grad.abs() * candidates.float()
    n_regrow = min(n_regrow, int(candidates.sum().item()))
    flat_scores = scores.flatten()
    top = torch.topk(flat_scores, n_regrow).indices
    rows, cols = top // n, top % n
    net.mask[rows, cols] = True
    sign = torch.sign(dense_grad[rows, cols])
    sign[sign == 0] = 1.0
    # Regrow in the direction that *reduces* the loss, i.e. opposite the grad sign.
    net.edge_sign.data[rows, cols] = -sign
    net.W_mag.data[rows, cols] = torch.rand(n_regrow) * init_weight_scale
    return n_regrow


@torch.no_grad()
def deep_r_step(
    net: RandomGraphSNN, init_weight_scale: float, gen: torch.Generator, eps: float = 1e-6
) -> dict:
    """One DEEP-R prune+regrow cycle. Call after each optimizer.step() (or
    every `interval_steps`)."""
    pruned = prune_collapsed_edges(net, eps=eps)
    n_pruned = int(pruned.sum().item())
    n_regrown = regrow_random(net, n_pruned, init_weight_scale, gen)
    return {"n_pruned": n_pruned, "n_regrown": n_regrown, "n_edges": net.edge_count()}
