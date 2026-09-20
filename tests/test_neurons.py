import torch

from rgsn.neurons import LIFCell, beta_from_tau, spike_fn


def test_beta_from_tau():
    beta = beta_from_tau(tau_mem_ms=20.0, dt_ms=1.0)
    assert 0.0 < beta < 1.0


def test_spike_fn_forward_is_heaviside():
    v = torch.tensor([-1.0, -0.001, 0.001, 1.0])
    s = spike_fn(v, slope=25.0)
    assert torch.equal(s, torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_spike_fn_backward_nonzero_gradient():
    v = torch.tensor([0.0], requires_grad=True)
    s = spike_fn(v, slope=25.0)
    s.backward()
    assert v.grad is not None
    assert v.grad.item() > 0


def test_lif_silent_with_zero_input():
    cell = LIFCell(beta=0.9, v_th=1.0, surrogate_slope=25.0)
    v = torch.zeros(4, 10)
    i_syn = torch.zeros(4, 10)
    for _ in range(20):
        v, s, _ = cell.step(v, i_syn)
        assert torch.all(s == 0)


def test_lif_reset_by_subtraction():
    cell = LIFCell(beta=1.0, v_th=1.0, surrogate_slope=25.0)
    v = torch.zeros(1, 1)
    i_syn = torch.full((1, 1), 1.5)
    v, s, _ = cell.step(v, i_syn)
    assert s.item() == 1.0
    assert torch.isclose(v, torch.tensor([[0.5]]))


def test_lif_firing_rate_increases_with_input_gain():
    cell = LIFCell(beta=0.9, v_th=1.0, surrogate_slope=25.0)

    def total_spikes(gain):
        v = torch.zeros(1, 1)
        total = 0.0
        for _ in range(200):
            v, s, _ = cell.step(v, torch.full((1, 1), gain))
            total += s.item()
        return total

    assert total_spikes(0.3) < total_spikes(0.8)


def test_lif_refractory_suppresses_immediate_refire():
    cell = LIFCell(beta=1.0, v_th=1.0, surrogate_slope=25.0, refractory_steps=3)
    v = torch.zeros(1, 1)
    refrac = torch.zeros(1, 1)
    i_syn = torch.full((1, 1), 2.0)
    v, s0, refrac = cell.step(v, i_syn, refrac)
    assert s0.item() == 1.0
    v, s1, refrac = cell.step(v, i_syn, refrac)
    assert s1.item() == 0.0  # still refractory, input current gets clamped away
