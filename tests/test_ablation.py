import torch

from rgsn.ablation import EdgeAblation, NeuronAblation, random_neuron_subset
from rgsn.config import GraphConfig, NeuronConfig
from rgsn.graph import build_graph
from rgsn.network import RandomGraphSNN


def make_net(n=60, p=0.15, seed=0):
    gcfg = GraphConfig(n=n, p=p, input_fraction=0.3, output_group_size=3, n_classes=3)
    gen = torch.Generator().manual_seed(seed)
    spec = build_graph(gcfg, gen)
    net = RandomGraphSNN(spec, NeuronConfig(), n_features=10, seed=seed)
    return net


def test_neuron_ablation_silences_and_restores():
    net = make_net()
    x = torch.rand(3, 20, 10)
    idx = torch.tensor([0, 1, 2])
    with NeuronAblation(net, idx):
        spikes = net(x)
        assert torch.all(spikes[:, :, idx] == 0)
    net2 = make_net()  # fresh net with identical seed for comparison
    assert net.edge_count() == net2.edge_count()  # mask restored to original


def test_edge_ablation_reduces_and_restores_edge_count():
    net = make_net()
    gen = torch.Generator().manual_seed(9)
    edges_before = net.edge_count()
    with EdgeAblation(net, fraction=0.3, gen=gen):
        edges_during = net.edge_count()
        assert edges_during < edges_before
        assert abs(edges_during - edges_before * 0.7) <= 2
    assert net.edge_count() == edges_before


def test_random_neuron_subset_excludes_given_indices():
    gen = torch.Generator().manual_seed(0)
    exclude = torch.tensor([0, 1, 2, 3, 4])
    subset = random_neuron_subset(20, 0.5, gen, exclude=exclude)
    assert set(subset.tolist()).isdisjoint(set(exclude.tolist()))
    assert subset.numel() == 8  # 50% of the remaining 15 candidates, rounded
