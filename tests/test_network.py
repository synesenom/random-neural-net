import time

import torch

from rgsn.config import GraphConfig, NeuronConfig
from rgsn.graph import build_graph
from rgsn.network import RandomGraphSNN


def make_net(n=60, p=0.1, gain=1.0, n_features=20, seed=0):
    gcfg = GraphConfig(n=n, p=p, input_fraction=0.3, output_group_size=2, n_classes=3, gain=gain)
    gen = torch.Generator().manual_seed(seed)
    spec = build_graph(gcfg, gen)
    ncfg = NeuronConfig()
    net = RandomGraphSNN(spec, ncfg, n_features=n_features, gain=gain, seed=seed)
    return net, spec


def test_forward_shapes():
    net, spec = make_net()
    x = torch.rand(5, 15, 20)
    spikes = net(x)
    assert spikes.shape == (5, 15, net.n)


def test_masked_edges_carry_no_current_and_no_gradient():
    net, spec = make_net(n=30, p=0.2, n_features=10)
    off_mask = ~net.mask
    assert torch.all(net.effective_recurrent_weight()[off_mask] == 0)
    x = torch.rand(2, 8, 10, requires_grad=False)
    spikes = net(x)
    spikes.sum().backward()
    assert net.W_mag.grad is not None
    assert torch.all(net.W_mag.grad[off_mask] == 0)


def test_silent_with_zero_input():
    net, spec = make_net(n=40, p=0.1, n_features=10)
    x = torch.zeros(3, 10, 10)
    spikes = net(x)
    assert torch.all(spikes == 0)


def test_firing_rate_increases_with_gain():
    def total_rate(gain):
        net, spec = make_net(n=50, p=0.1, gain=gain, n_features=15, seed=7)
        x = (torch.rand(4, 30, 15) < 0.5).float()
        spikes = net(x)
        return spikes.mean().item()

    assert total_rate(0.3) < total_rate(3.0)


def test_deterministic_under_seed():
    net1, _ = make_net(seed=123)
    net2, _ = make_net(seed=123)
    assert torch.equal(net1.W_mag, net2.W_mag)
    assert torch.equal(net1.W_in, net2.W_in)
    x = torch.rand(2, 5, 20)
    s1 = net1(x)
    s2 = net2(x)
    assert torch.equal(s1, s2)


def test_gradients_flow_to_w_and_w_in():
    net, spec = make_net(n=25, p=0.2, n_features=8)
    x = torch.rand(2, 6, 8)
    spikes = net(x)
    loss = spikes.sum()
    loss.backward()
    assert net.W_mag.grad is not None and net.W_mag.grad.abs().sum() > 0
    assert net.W_in.grad is not None and net.W_in.grad.abs().sum() > 0


def test_forward_pass_speed_batch64_n1000_t100():
    gcfg = GraphConfig(n=1000, p=0.05, input_fraction=0.2, output_group_size=10, n_classes=10)
    gen = torch.Generator().manual_seed(0)
    spec = build_graph(gcfg, gen)
    ncfg = NeuronConfig()
    net = RandomGraphSNN(spec, ncfg, n_features=784, seed=0)
    x = torch.rand(64, 100, 784)
    start = time.time()
    spikes = net(x)
    elapsed = time.time() - start
    assert spikes.shape == (64, 100, 1000)
    assert elapsed < 30.0
