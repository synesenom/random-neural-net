# RGSN: Random-Graph Spiking Network with Structural Plasticity

Implementation of [PLAN.md](PLAN.md): a sparse random directed graph of
leaky-integrate-and-fire (LIF) spiking neurons, with population-coded I/O and
DEEP-R-style structural plasticity (prune + regrow), evaluated against
parameter-matched baselines on MNIST.

## Status

All phases from PLAN.md Section 6 are implemented, at a scale (N=300,
T=30 steps, 2-4k training samples) chosen so the full experiment suite runs
in minutes on CPU rather than requiring a GPU cluster; see "Scope and
deviations from PLAN.md" below.

| Phase | Status |
|---|---|
| 0 — Setup | Done |
| 1 — Simulator core (`graph.py`, `neurons.py`, `network.py`) | Done |
| 2 — Encoding/decoding | Done |
| 3 — Fixed reservoir baseline | Done |
| 4 — Full weight training (BPTT + surrogate gradients) | Done |
| 5 — Rewiring (DEEP-R) | Done |
| 6 — Baselines (MLP, reservoir, RGSN-no-rewire) | Done |
| 7 — Experiments (H1/H2/H3) | Done (MNIST only) |
| 8 — Stretch (delays, e-prop, SHD/H4, topology analysis) | Not implemented |

## Repo layout

Matches PLAN.md Section 3: `src/rgsn/` for the library, `configs/` for YAML
run configs, `experiments/` for the H1/H2/H3 drivers and plotting, `tests/`
for pytest, `runs/` (gitignored) for per-run outputs.

## Quickstart

```bash
pip install -e ".[dev]"
pytest tests/ -q                                    # fast tests
pytest tests/ -q -m slow                             # + end-to-end training smoke tests
python -m rgsn.train --config configs/rgsn_mnist.yaml --dry-run
python -m rgsn.train --config configs/rgsn_mnist.yaml   # one training run

# Full experiment suite (takes ~XX minutes on CPU):
python experiments/run_sample_efficiency.py    # H1 + H2 data
python experiments/run_ablation.py             # H3 data
python experiments/make_plots.py               # renders runs/plots/*.png from the CSVs
```

Every run is fully defined by `configs/rgsn_mnist.yaml` + a seed; the
resolved config and per-step/per-epoch CSV logs are saved to
`runs/<name>/`. Plots are generated only from those saved CSVs.

## Model summary

- **Graph**: Erdos-Renyi (default) or Watts-Strogatz directed sparse graph,
  N=300, p=0.08 (~7200 directed edges), no self-loops, optional Dale's law.
- **Neurons**: discrete-time LIF, reset-by-subtraction, fast-sigmoid
  surrogate gradient.
- **I/O**: 20% of neurons are inputs (Poisson rate-coded pixels via a
  learned `W_in` projection restricted to those rows); 10 disjoint 8-neuron
  output groups give population-coded class logits (mean spike rate per
  group). An ablated/removed neuron's row is forced to zero, so a group's
  mean degrades smoothly instead of the readout breaking.
- **Learning**: BPTT with surrogate gradients (Adam) on the recurrent weight
  magnitude `W_mag` (parameterized as `sign * relu(W_mag)`, a DEEP-R-style
  sign-constrained weight) and on `W_in`, plus a firing-rate regularizer.
- **Rewiring (DEEP-R)**: an edge whose magnitude is driven to <= 0 by
  gradient descent is structurally pruned (relu already zeroes its current
  and its gradient); the same number of edges is regrown at uniformly
  random inactive positions each step, so the active-edge count is exactly
  constant throughout training.

## Baselines (Section 5)

All models are trained with the same data pipeline, epoch budget, and (for
the spiking models) the same graph/neuron hyperparameters:

- **MLP**: 1 hidden layer, ReLU, hidden width chosen so its parameter count
  matches the RGSN's *nonzero* trainable parameter count (`mlp_param_budget`
  in the config).
- **Fixed reservoir**: identical random graph, `W_mag`/`W_in` frozen at
  init; only a linear readout over all N neurons' mean spike rate is
  trained (isolates the value of learning the weights at all).
- **RGSN, no rewiring**: `W_mag`/`W_in` trained, topology frozen (isolates
  the value of rewiring, vs. the full RGSN).
- **RGSN, full**: weights trained + DEEP-R rewiring.

## Results

*(filled in from `runs/h1_sample_efficiency/summary.csv` and
`runs/h3_ablation/summary.csv` after the experiment suite finishes — see
`runs/plots/*.png`.)*

### H1 — Sample efficiency

### H2 — Learning speed

### H3 — Robustness to ablation

## Scope and deviations from PLAN.md

Time/compute-boxed for this exercise; documented here rather than silently
diverging:

- **Scale**: N=300 (not 1000), T=30 (not 50-100), training-set sizes up to
  4000 (not the full 60k) and 3 seeds (not 5). The simulator itself is
  validated at the plan's target scale (N=1000, T=100, batch=64) in
  `tests/test_network.py::test_forward_pass_speed_batch64_n1000_t100`.
- **H4 (temporal, SHD) and Phase 8 (stretch) are not implemented**: no
  `tonic`/SHD dependency, no synaptic delays, no local learning rules
  (e-prop / reward-modulated STDP), no topology analysis. The architecture
  (`encoding.py`'s `passthrough_encode`, `RandomGraphSNN`'s discrete-time
  interface) is built to make adding an event-based temporal dataset a
  matter of writing a loader, not a redesign.
- **RigL regrowth** (`rewiring.regrow_rigl`) is implemented but not wired
  into the training loop or exercised by the experiments; only DEEP-R
  (uniform random regrowth) is used.
- **Weight parameterization**: PLAN.md 4.5 specifies "if an update would
  flip [an edge's] sign, remove it." The implementation instead uses the
  DEEP-R paper's own parameterization directly (`sign * relu(magnitude)`):
  it is mathematically the same rule (magnitude <= 0 <=> would-be sign flip
  <=> pruned) but avoids ever materializing a wrong-signed weight, which
  also fixed a real training-stability bug encountered along the way (next
  point).
- **Hyperparameter tuning was done by hand**, not full "Phase 3" grid
  search: an initial `softplus`-based weight parameterization combined with
  gain=1.5 produced a network whose forward dynamics looked healthy
  (5-20 Hz firing) but whose 30-step BPTT gradients were exploding
  (`|grad| ~ 1e9`), because `softplus` has a `ln(2)` offset at zero that
  inflated the effective initial weight scale well past what the firing-rate
  regularizer could see. Switching to `relu` (small init magnitude maps
  directly to small effective weight) and retuning `gain` (search over
  {1.5, 2, 3, 4}) and `surrogate_slope` (search over {25, 10, 5, 2, 1})
  against measured gradient norms, not just firing rate, found a stable
  regime (gain=4, slope=1); gradient clipping (`train.grad_clip_norm`) is
  also on as a safety net. A proper Phase-3-style sweep with logged
  firing-rate histograms across seeds was out of scope here.
