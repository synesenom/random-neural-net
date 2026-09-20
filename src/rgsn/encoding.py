"""Input encoders: turn static features or event streams into spike trains
of shape (B, T, F) consumed by `RandomGraphSNN.forward`."""

from __future__ import annotations

import torch


def poisson_rate_encode(
    x: torch.Tensor,
    t_steps: int,
    max_rate_hz: float,
    dt_ms: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Independent Poisson (Bernoulli-per-step) encoding.

    x: (B, F) values in [0, 1] (e.g. normalized pixel intensities).
    Returns (B, T, F) binary spike train; per-step firing probability is
    `x * max_rate_hz * dt_s`, i.e. higher intensity -> higher instantaneous
    spike rate.
    """
    p = (x.clamp(0.0, 1.0) * max_rate_hz * dt_ms / 1000.0).clamp(0.0, 1.0)
    p = p.unsqueeze(1).expand(-1, t_steps, -1)
    return torch.bernoulli(p, generator=generator)


def passthrough_encode(events: torch.Tensor) -> torch.Tensor:
    """Events already provided as a (B, T, F) binary/float tensor: no-op."""
    return events
