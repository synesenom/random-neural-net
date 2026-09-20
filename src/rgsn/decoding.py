"""Population readouts: turn (B, T, N) spike trains into (B, C) class logits.

Output neurons are grouped disjointly, `output_idx: (C, k)`. Because the
logit is a mean over a group, a neuron dropped by ablation (its row of
spikes forced to zero) simply lowers that group's mean instead of producing
an undefined or crashing readout — this is what gives population coding its
graceful degradation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PopulationRateDecoder(nn.Module):
    """Logit_c = temperature * mean spike count of group c over the window."""

    def __init__(
        self,
        output_idx: torch.Tensor,
        learnable_temperature: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("output_idx", output_idx)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(float(temperature))))

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """spikes: (B, T, N) -> logits (B, C)."""
        group_spikes = spikes[:, :, self.output_idx]  # (B, T, C, k)
        rate = group_spikes.mean(dim=(1, 3))  # (B, C)
        return rate * torch.exp(self.log_temperature)


class MembranePotentialDecoder(nn.Module):
    """Logit_c = mean membrane potential of group c over the window (needs the
    membrane-potential trace, not just spikes)."""

    def __init__(self, output_idx: torch.Tensor):
        super().__init__()
        self.register_buffer("output_idx", output_idx)

    def forward(self, v_trace: torch.Tensor) -> torch.Tensor:
        group_v = v_trace[:, :, self.output_idx]  # (B, T, C, k)
        return group_v.mean(dim=(1, 3))


class LinearReadoutDecoder(nn.Module):
    """Learned linear readout over *all* neurons' mean spike rate (comparison
    decoder only, not the main population-coding model)."""

    def __init__(self, n: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n, n_classes)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        rate = spikes.mean(dim=1)  # (B, N)
        return self.linear(rate)
