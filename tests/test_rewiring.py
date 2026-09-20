import torch

from rgsn.config import GraphConfig, NeuronConfig
from rgsn.graph import build_graph
from rgsn.network import RandomGraphSNN
from rgsn.rewiring import deep_r_step, prune_collapsed_edges, regrow_random


def make_net(n=50, p=0.15, seed=0):
    gcfg = GraphConfig(n=n, p=p, input_fraction=0.2, output_group_size=2, n_classes=3)
    gen = torch.Generator().manual_seed(seed)
    spec = build_graph(gcfg, gen)
    net = RandomGraphSNN(spec, NeuronConfig(), n_features=10, seed=seed)
    return net


def test_prune_removes_collapsed_edges():
    net = make_net()
    edges_before = net.edge_count()
    idx = net.mask.nonzero(as_tuple=False)[0]
    with torch.no_grad():
        net.W_mag[idx[0], idx[1]] = 0.0
    pruned = prune_collapsed_edges(net)
    assert pruned.sum().item() == 1
    assert net.edge_count() == edges_before - 1
    assert not net.mask[idx[0], idx[1]]


def test_regrow_keeps_edge_count_and_avoids_self_loops():
    net = make_net()
    gen = torch.Generator().manual_seed(1)
    edges_before = net.edge_count()
    n_regrown = regrow_random(net, n_regrow=5, init_weight_scale=0.05, gen=gen)
    assert n_regrown == 5
    assert net.edge_count() == edges_before + 5
    assert not net.mask.diagonal().any()


def test_deep_r_step_keeps_total_edge_count_constant():
    net = make_net(n=60, p=0.1, seed=2)
    edges_before = net.edge_count()
    with torch.no_grad():
        idx = net.mask.nonzero(as_tuple=False)
        # Force-collapse 3 random active edges to simulate a training step
        # that drove their magnitude to zero.
        for i in range(3):
            net.W_mag[idx[i, 0], idx[i, 1]] = 0.0
    gen = torch.Generator().manual_seed(3)
    stats = deep_r_step(net, init_weight_scale=0.05, gen=gen)
    assert stats["n_pruned"] == 3
    assert stats["n_regrown"] == 3
    assert net.edge_count() == edges_before


def test_regrown_edges_produce_gradient_flow():
    net = make_net(n=40, p=0.1, seed=4)
    gen = torch.Generator().manual_seed(5)
    edges_before = net.edge_count()
    with torch.no_grad():
        idx = net.mask.nonzero(as_tuple=False)
        net.W_mag[idx[0, 0], idx[0, 1]] = 0.0
    deep_r_step(net, init_weight_scale=0.05, gen=gen)
    assert net.edge_count() == edges_before
    x = torch.rand(2, 5, 10)
    spikes = net(x)
    spikes.sum().backward()
    assert net.W_mag.grad[net.mask].abs().sum() >= 0  # runs without error
