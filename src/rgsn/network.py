"""RandomGraphSNN: simulates a sparse random directed graph of LIF neurons
over T discrete time steps, with a designated input neuron subset and
disjoint output neuron groups (population coding).

Each structurally active edge (i <- j) carries a parameter with a fixed sign
(`edge_sign`), following the DEEP-R weight parameterization: the effective
weight is `sign * relu(magnitude)`, which is always consistent with the
assigned sign. `rewiring.py` prunes edges whose magnitude has been driven to
(near) zero and regrows the same number elsewhere.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rgsn.config import NeuronConfig
from rgsn.graph import GraphSpec
from rgsn.neurons import LIFCell, beta_from_tau


class RandomGraphSNN(nn.Module):
    def __init__(
        self,
        graph_spec: GraphSpec,
        neuron_cfg: NeuronConfig,
        n_features: int,
        gain: float = 1.0,
        train_input_weights: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        n = graph_spec.n
        self.n = n
        self.n_classes, self.k = graph_spec.output_idx.shape

        gen = torch.Generator().manual_seed(seed)

        self.register_buffer("mask", graph_spec.mask.clone())
        self.register_buffer("input_idx", graph_spec.input_idx.clone())
        self.register_buffer("output_idx", graph_spec.output_idx.clone())

        in_row_mask = torch.zeros(n, dtype=torch.bool)
        in_row_mask[graph_spec.input_idx] = True
        self.register_buffer("in_row_mask", in_row_mask)

        # Per-edge fixed sign. If Dale's law was requested, graph_spec.sign
        # already encodes one sign per presynaptic neuron; broadcast over rows
        # (post-synaptic index) so every outgoing edge from a neuron shares it.
        if torch.any(graph_spec.sign != 1.0):
            edge_sign = graph_spec.sign.unsqueeze(0).expand(n, n).clone()
        else:
            edge_sign = torch.randint(0, 2, (n, n), generator=gen).float() * 2 - 1
        edge_sign = edge_sign * self.mask.float()
        self.register_buffer("edge_sign", edge_sign)

        density = max(self.mask.float().mean().item(), 1.0 / n)
        std = gain / math.sqrt(n * density)
        mag_init = torch.rand((n, n), generator=gen) * std  # positive magnitude
        self.W_mag = nn.Parameter(mag_init * self.mask.float())

        in_std = gain / math.sqrt(max(1, n_features))
        w_in_init = torch.randn((n, n_features), generator=gen) * in_std
        w_in_init = w_in_init * in_row_mask.unsqueeze(1).float()
        self.W_in = nn.Parameter(w_in_init, requires_grad=train_input_weights)

        beta = beta_from_tau(neuron_cfg.tau_mem_ms, neuron_cfg.dt_ms)
        self.cell = LIFCell(
            beta, neuron_cfg.v_th, neuron_cfg.surrogate_slope, neuron_cfg.refractory_steps
        )
        self.refractory_steps = neuron_cfg.refractory_steps

    def effective_recurrent_weight(self) -> torch.Tensor:
        """W_eff[i, j] = sign_ij * relu(magnitude_ij), masked to active edges.

        This is the DEEP-R weight parameterization (Bellec et al. 2018): the
        magnitude is rectified so it can never disagree with the edge's fixed
        sign, relu(0) = 0 keeps a small init close to zero current (unlike
        softplus, whose nonzero offset at 0 would inflate initial weight
        scale), and an edge whose magnitude is driven <= 0 by gradient descent
        carries zero current and zero gradient, i.e. it is already pruned.
        """
        return self.edge_sign * F.relu(self.W_mag) * self.mask.float()

    def effective_input_weight(self) -> torch.Tensor:
        return self.W_in * self.in_row_mask.unsqueeze(1).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) encoded input. Returns spikes (B, T, N)."""
        b, t_steps, _ = x.shape
        device = x.device
        v = torch.zeros(b, self.n, device=device)
        refrac = torch.zeros(b, self.n, device=device) if self.refractory_steps > 0 else None
        s = torch.zeros(b, self.n, device=device)

        w_rec = self.effective_recurrent_weight()
        w_in = self.effective_input_weight()

        spikes_out = []
        for t in range(t_steps):
            i_ext = x[:, t, :] @ w_in.t()
            i_rec = s @ w_rec.t()
            v, s, refrac = self.cell.step(v, i_ext + i_rec, refrac)
            spikes_out.append(s)
        return torch.stack(spikes_out, dim=1)

    def firing_rate_hz(self, spikes: torch.Tensor, dt_ms: float) -> torch.Tensor:
        """Mean firing rate per neuron across batch and time, in Hz. spikes: (B,T,N)."""
        t_steps = spikes.shape[1]
        duration_s = t_steps * dt_ms / 1000.0
        return spikes.mean(dim=(0, 1)) / duration_s

    def edge_count(self) -> int:
        return int(self.mask.sum().item())
