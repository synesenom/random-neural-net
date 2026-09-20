"""Synthetic delayed-recall task: a binary value is shown for `cue_steps`,
followed by `delay` steps of silence, followed by a "go" pulse for
`query_steps` during which the network must report the value it saw.

This isolates a capability gap that a classifier evaluated online (seeing
only the current timestep, no buffered history) cannot have regardless of
training: at query time the only visible input is the go pulse, which is
identical for both classes, so a *memoryless* per-timestep classifier is
mathematically forced to chance accuracy for any delay >= 0. A recurrent
model (RGSN's spiking dynamics, or an LSTM) can carry the value forward in
its own state and is not subject to that bound. See PLAN.md's H4
(temporal tasks) hypothesis.

Two features per timestep: channel 0 carries the value during the cue
window (Bernoulli spikes at a class-dependent rate), channel 1 carries the
go pulse (Bernoulli spikes at a fixed rate, same for both classes).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RecallTaskSpec:
    cue_steps: int
    delay: int
    query_steps: int
    n_features: int = 2

    @property
    def t_total(self) -> int:
        return self.cue_steps + self.delay + self.query_steps

    @property
    def query_start(self) -> int:
        return self.cue_steps + self.delay


def generate_dataset(
    spec: RecallTaskSpec,
    n_samples: int,
    seed: int,
    cue_rate_hi: float = 0.8,
    cue_rate_lo: float = 0.1,
    go_rate: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (X, y): X is (n_samples, T, 2) binary spikes, y is (n_samples,)
    in {0, 1}."""
    gen = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n_samples,), generator=gen)
    x = torch.zeros(n_samples, spec.t_total, spec.n_features)

    cue_rate = torch.where(y == 1, cue_rate_hi, cue_rate_lo).unsqueeze(1)  # (n_samples, 1)
    cue_probs = cue_rate.expand(n_samples, spec.cue_steps)
    x[:, : spec.cue_steps, 0] = torch.bernoulli(cue_probs, generator=gen)

    go_probs = torch.full((n_samples, spec.query_steps), go_rate)
    x[:, spec.query_start :, 1] = torch.bernoulli(go_probs, generator=gen)

    return x, y


def make_loaders(
    spec: RecallTaskSpec,
    n_train: int,
    n_test: int,
    batch_size: int,
    seed: int,
    **rate_kwargs,
):
    from torch.utils.data import DataLoader, TensorDataset

    x_train, y_train = generate_dataset(spec, n_train, seed, **rate_kwargs)
    x_test, y_test = generate_dataset(spec, n_test, seed + 1, **rate_kwargs)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class MemorylessClassifier(nn.Module):
    """A shared-weight MLP applied *independently* to each timestep: no
    recurrent connection carries information from one call to the next, so
    its output at time t is a pure function of x[t] alone. Structurally
    incapable of solving any delay > 0 of this task, whatever it learns."""

    def __init__(self, n_features: int, n_classes: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> per-timestep logits (B, T, C)."""
        b, t, f = x.shape
        return self.net(x.reshape(b * t, f)).reshape(b, t, -1)


class WindowedClassifier(nn.Module):
    """An offline MLP given the whole flattened sequence at once (external
    buffering supplies the memory a feedforward net lacks). Included as a
    sanity-check upper bound, not a fair online baseline."""

    def __init__(self, t_total: int, n_features: int, n_classes: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(t_total * n_features, hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return self.net(x.reshape(b, -1))
