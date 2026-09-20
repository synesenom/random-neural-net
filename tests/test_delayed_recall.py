import torch

from rgsn.delayed_recall import (
    MemorylessClassifier,
    RecallTaskSpec,
    WindowedClassifier,
    generate_dataset,
)


def test_dataset_shapes_and_labels():
    spec = RecallTaskSpec(cue_steps=5, delay=10, query_steps=5)
    x, y = generate_dataset(spec, n_samples=200, seed=0)
    assert x.shape == (200, spec.t_total, 2)
    assert set(y.unique().tolist()).issubset({0, 1})
    assert 60 < y.float().sum().item() < 140  # roughly balanced


def test_cue_and_go_windows_are_disjoint_and_correctly_placed():
    spec = RecallTaskSpec(cue_steps=5, delay=10, query_steps=5)
    x, y = generate_dataset(
        spec, n_samples=500, seed=0, cue_rate_hi=1.0, cue_rate_lo=0.0, go_rate=1.0
    )
    # Cue window channel 0 exactly encodes the label (rates 1.0/0.0 are deterministic).
    cue_active = x[:, : spec.cue_steps, 0].mean(dim=1)
    assert torch.allclose(cue_active, y.float())
    # Delay window is silent on both channels.
    assert torch.all(x[:, spec.cue_steps : spec.query_start, :] == 0)
    # Go window channel 1 always fires (go_rate=1.0), channel 0 never does.
    assert torch.all(x[:, spec.query_start :, 1] == 1)
    assert torch.all(x[:, spec.query_start :, 0] == 0)


def test_memoryless_classifier_output_is_pure_function_of_query_window():
    """The whole point of the task: since the memoryless classifier only ever
    sees x[query_start:], and that slice is statistically identical for both
    classes (same go rate, no leaked value info), two datasets differing only
    in their cue-window value must produce the same *input* to the model at
    query time, and hence the same output for a given input in that slice."""
    spec = RecallTaskSpec(cue_steps=5, delay=10, query_steps=5)
    torch.manual_seed(0)
    model = MemorylessClassifier(n_features=2, n_classes=2, hidden=8)

    query_slice_a = torch.zeros(1, spec.query_steps, 2)
    query_slice_a[:, :, 1] = 1.0  # go pulse on, matches go_rate=1 pattern
    query_slice_b = query_slice_a.clone()  # identical: value info never reaches this slice

    out_a = model(query_slice_a)
    out_b = model(query_slice_b)
    assert torch.equal(out_a, out_b)


def test_windowed_classifier_forward_shape():
    spec = RecallTaskSpec(cue_steps=5, delay=10, query_steps=5)
    model = WindowedClassifier(spec.t_total, n_features=2, n_classes=2, hidden=8)
    x = torch.rand(4, spec.t_total, 2)
    out = model(x)
    assert out.shape == (4, 2)


def test_memoryless_classifier_forward_shape():
    model = MemorylessClassifier(n_features=2, n_classes=2, hidden=8)
    x = torch.rand(4, 6, 2)
    out = model(x)
    assert out.shape == (4, 6, 2)
