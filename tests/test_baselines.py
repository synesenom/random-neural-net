import torch

from rgsn.baselines import (
    MLP,
    hidden_size_for_budget,
    mlp_param_count,
    random_hidden_ablation_masks,
)


def test_hidden_size_for_budget_matches_closely():
    budget = 50_000
    h = hidden_size_for_budget(budget, n_features=784, n_classes=10)
    actual = mlp_param_count(784, 10, [h])
    assert abs(actual - budget) / budget < 0.05


def test_mlp_forward_shape():
    model = MLP(n_features=784, n_classes=10, hidden_sizes=[64])
    x = torch.rand(8, 784)
    out = model(x)
    assert out.shape == (8, 10)


def test_hidden_ablation_zeroes_activations():
    model = MLP(n_features=20, n_classes=3, hidden_sizes=[16])
    gen = torch.Generator().manual_seed(0)
    masks = random_hidden_ablation_masks([16], fraction=0.5, gen=gen)
    assert masks[0].sum().item() == 8
    x = torch.rand(4, 20)
    out_full = model(x)
    out_ablated = model(x, hidden_masks=masks)
    assert not torch.allclose(out_full, out_ablated)
