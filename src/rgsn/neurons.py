"""Discrete-time LIF neuron dynamics with a surrogate-gradient spike function."""

from __future__ import annotations

import math

import torch


class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside step in the forward pass; fast-sigmoid surrogate in backward."""

    @staticmethod
    def forward(ctx, v_minus_th: torch.Tensor, slope: float):
        ctx.save_for_backward(v_minus_th)
        ctx.slope = slope
        return (v_minus_th > 0).to(v_minus_th.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v_minus_th,) = ctx.saved_tensors
        slope = ctx.slope
        denom = (1.0 + slope * v_minus_th.abs()) ** 2
        grad = grad_output * slope / denom
        return grad, None


def spike_fn(v_minus_th: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    return FastSigmoidSpike.apply(v_minus_th, slope)


def beta_from_tau(tau_mem_ms: float, dt_ms: float) -> float:
    return math.exp(-dt_ms / tau_mem_ms)


class LIFCell:
    """Stateless functional LIF update, reset-by-subtraction, optional refractory period.

    v[t+1] = beta * v[t] + i_syn[t] - s[t] * v_th   (reset happens using the spike
                                                        emitted *this* step, applied
                                                        to the membrane potential
                                                        that produced it)
    s[t]   = H(v[t] - v_th)
    """

    def __init__(self, beta: float, v_th: float, surrogate_slope: float, refractory_steps: int = 0):
        self.beta = beta
        self.v_th = v_th
        self.surrogate_slope = surrogate_slope
        self.refractory_steps = refractory_steps

    def step(
        self,
        v: torch.Tensor,
        i_syn: torch.Tensor,
        refrac: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """One time step.

        v: (B, N) membrane potential at t
        i_syn: (B, N) total input current at t (external + recurrent)
        refrac: (B, N) remaining refractory steps, or None if refractory disabled

        Returns (v_next, spikes, refrac_next).
        """
        v_new = self.beta * v + i_syn
        if refrac is not None:
            active = (refrac <= 0).to(v.dtype)
            v_new = v_new * active  # neurons in refractory stay clamped near 0
        spikes = spike_fn(v_new - self.v_th, self.surrogate_slope)
        v_next = v_new - spikes.detach() * self.v_th
        refrac_next = None
        if refrac is not None:
            refrac_next = torch.clamp(refrac - 1, min=0)
            refrac_next = torch.where(
                spikes.detach() > 0,
                torch.full_like(refrac_next, float(self.refractory_steps)),
                refrac_next,
            )
        return v_next, spikes, refrac_next
