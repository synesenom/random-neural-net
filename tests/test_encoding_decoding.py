import torch

from rgsn.decoding import PopulationRateDecoder
from rgsn.encoding import poisson_rate_encode


def test_encoder_rate_matches_pixel_intensity():
    torch.manual_seed(0)
    x = torch.tensor([[0.0, 0.5, 1.0]]).expand(200, 3).clone()
    spikes = poisson_rate_encode(x, t_steps=100, max_rate_hz=100.0, dt_ms=1.0)
    mean_rate = spikes.mean(dim=(0, 1))  # per-feature average firing prob
    assert mean_rate[0].item() < 0.02  # ~0 intensity -> near-silent
    assert 0.03 < mean_rate[1].item() < 0.07  # ~0.5 intensity -> ~5% prob/step
    assert mean_rate[2].item() > 0.08  # max intensity -> ~10% prob/step (clamped)


def test_encoder_output_shape_and_binary():
    x = torch.rand(4, 10)
    spikes = poisson_rate_encode(x, t_steps=20, max_rate_hz=50.0, dt_ms=1.0)
    assert spikes.shape == (4, 20, 10)
    assert set(torch.unique(spikes).tolist()).issubset({0.0, 1.0})


def test_decoder_ignores_non_output_neurons():
    output_idx = torch.tensor([[0, 1], [2, 3]])
    decoder = PopulationRateDecoder(output_idx)
    spikes = torch.zeros(1, 5, 10)
    spikes[:, :, 0] = 1.0  # class-0 group neuron fires
    spikes[:, :, 5] = 1.0  # non-output neuron fires a lot, should be ignored
    logits = decoder(spikes)
    assert logits.shape == (1, 2)
    assert logits[0, 0].item() == 0.5  # mean of [1,0] over group of size 2
    assert logits[0, 1].item() == 0.0


def test_ablating_output_neuron_lowers_group_mean_but_stays_finite():
    output_idx = torch.tensor([[0, 1, 2]])
    decoder = PopulationRateDecoder(output_idx)
    spikes = torch.ones(1, 4, 3)
    full_logit = decoder(spikes)
    spikes_ablated = spikes.clone()
    spikes_ablated[:, :, 0] = 0.0  # ablate one neuron in the group
    ablated_logit = decoder(spikes_ablated)
    assert torch.isfinite(ablated_logit).all()
    assert ablated_logit.item() < full_logit.item()
    assert ablated_logit.item() > 0.0  # group mean fades, doesn't zero out
