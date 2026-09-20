import torch

from rgsn.config import GraphConfig
from rgsn.graph import build_graph


def test_shapes_and_no_self_loops():
    cfg = GraphConfig(n=50, p=0.1, input_fraction=0.2, output_group_size=3, n_classes=4)
    gen = torch.Generator().manual_seed(0)
    spec = build_graph(cfg, gen)
    assert spec.mask.shape == (50, 50)
    assert spec.mask.dtype == torch.bool
    assert not spec.mask.diagonal().any()
    assert spec.input_idx.numel() == 10
    assert spec.output_idx.shape == (4, 3)


def test_input_output_disjoint():
    cfg = GraphConfig(n=100, p=0.05, input_fraction=0.2, output_group_size=5, n_classes=5)
    gen = torch.Generator().manual_seed(1)
    spec = build_graph(cfg, gen)
    in_set = set(spec.input_idx.tolist())
    out_set = set(spec.output_idx.flatten().tolist())
    assert in_set.isdisjoint(out_set)
    assert len(out_set) == 25  # all output neurons distinct across classes


def test_deterministic_under_seed():
    cfg = GraphConfig(n=60, p=0.08, input_fraction=0.2, output_group_size=2, n_classes=3)
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    spec1 = build_graph(cfg, gen1)
    spec2 = build_graph(cfg, gen2)
    assert torch.equal(spec1.mask, spec2.mask)
    assert torch.equal(spec1.input_idx, spec2.input_idx)


def test_dales_law_sign_per_presynaptic_neuron():
    cfg = GraphConfig(
        n=80,
        p=0.1,
        dales_law=True,
        excitatory_fraction=0.8,
        input_fraction=0.2,
        output_group_size=2,
        n_classes=3,
    )
    gen = torch.Generator().manual_seed(3)
    spec = build_graph(cfg, gen)
    n_inhibitory = (spec.sign < 0).sum().item()
    assert abs(n_inhibitory - 16) <= 2  # ~20% of 80


def test_watts_strogatz_runs():
    cfg = GraphConfig(
        n=40,
        graph_type="watts_strogatz",
        ws_k=4,
        ws_beta=0.2,
        input_fraction=0.2,
        output_group_size=2,
        n_classes=3,
    )
    gen = torch.Generator().manual_seed(0)
    spec = build_graph(cfg, gen)
    assert spec.mask.shape == (40, 40)
    assert spec.mask.sum().item() > 0
