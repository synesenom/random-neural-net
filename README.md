# RGSN: Random-Graph Spiking Network with Structural Plasticity

Implementation of [PLAN.md](PLAN.md): a sparse random directed graph of
leaky-integrate-and-fire (LIF) spiking neurons, with population-coded I/O and
DEEP-R-style structural plasticity (prune + regrow), evaluated against
parameter-matched baselines on MNIST, plus a synthetic online delayed-recall
task designed to test PLAN.md's H4 (temporal) hypothesis directly.

**Headline result**: RGSN loses badly to a parameter-matched MLP on static
MNIST classification (as PLAN.md's own "expectation check" predicts), but
wins decisively on **online temporal memory**: on a task where a value must
be reported some delay after it was shown, processing one timestep at a
time with no external buffer, a memoryless MLP is mathematically capped at
chance accuracy for *any* delay, while RGSN's recurrent spiking state
solves it (~96-97%, matching an MLP given the whole sequence at once) for
delays up to about one membrane time constant, degrading gracefully beyond
that. See "Delayed recall" under Results.

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
| 8 — Stretch (delays, e-prop, SHD/H4, topology analysis) | Partial: H4 tested via a custom synthetic delayed-recall task (`rgsn.delayed_recall`) instead of SHD; no synaptic delays, e-prop, or topology analysis |

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

# Full experiment suite (~15-20 min total on a 4-core CPU):
python experiments/run_sample_efficiency.py    # H1 + H2 data
python experiments/run_ablation.py             # H3 data (needs H1's checkpoints first)
python experiments/run_delayed_recall.py       # delayed-recall / H4 data (independent of the above)
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

Generated from `runs/h1_sample_efficiency/summary.csv`,
`runs/h3_ablation/summary.csv`, and `runs/delayed_recall/summary.csv`
(gitignored; regenerate with the Quickstart commands). Plots: `runs/plots/*.png`.

### H1 — Sample efficiency (MNIST, test accuracy, mean over 3 seeds)

| Model | n=200 | n=500 | n=1000 | n=2000 | n=4000 |
|---|---|---|---|---|---|
| MLP (param-matched) | 73.1% | 86.7% | 89.0% | 90.5% | **92.0%** |
| Fixed reservoir + linear readout | 25.6% | 35.9% | 40.0% | 44.4% | 49.5% |
| RGSN, no rewiring | 20.0% | 22.9% | 27.6% | 38.6% | 44.0% |
| RGSN, full (+ rewiring) | 13.6% | 24.7% | 28.0% | 36.4% | 41.4% |

The MLP dominates at every training-set size, matching PLAN.md's own
expectation. More surprising: the *frozen* reservoir with a trained linear
readout beats *both* trained RGSN variants (which use the population-code
readout) at every size, and rewiring doesn't measurably help over
`rgsn_no_rewire`. See "Update rule vs. output encoding" below for why.

### H2 — Learning speed (n_train=4000, seed 0)

MLP reaches 82.8% after a single epoch (7.2s total wall-clock for 15
epochs). The spiking models take 33-57s for the same 15 epochs (BPTT
through T=30 steps is the bottleneck) to reach 41-53%.

### H3 — Robustness to ablation

Fraction of each model's own *starting* accuracy retained at 50%
hidden-unit / edge / output-neuron ablation:

| Model | hidden | edge | output |
|---|---|---|---|
| MLP (hidden-unit ablation) | **84%** | - | - |
| Fixed reservoir | 71% | 68% | 79% |
| RGSN, no rewiring | 37% | 33% | 43% |
| RGSN, full | 50% | 28% | 62% |

This is a negative result relative to the original "population coding
degrades gracefully" hypothesis: the MLP degrades *more* gracefully in
relative terms, and starts from a much higher absolute accuracy, so it wins
outright. Likely cause: the MLP's hidden layer is densely connected to all
784 inputs and all 10 outputs (large redundancy), whereas RGSN's sparse
graph (avg out-degree ~=24) and small output groups (k=8 neurons/class)
have far less slack to absorb random damage.

### Update rule vs. output encoding

H1's odd ordering (frozen reservoir + linear readout beating trained RGSN +
population code) prompted a direct 2x2 factorial: {frozen, trained} weights
x {population-rate, dense linear} readout, n_train=4000, 15 epochs:

| | population-rate decoder | linear readout (all 300 neurons) |
|---|---|---|
| **frozen weights** | 10.2% (~chance) | 45.7% |
| **trained weights** | 41.6% | 76.1% |

Both factors matter about equally and combine **additively** (10.2 + 35.5
+ 31.4 ~= 77.1 ~= the actual 76.1, i.e. no strong interaction):
switching the decoder is worth +34-36 points regardless of whether the
weights are trained; switching from frozen to trained weights is worth
+30-31 points regardless of decoder. The extreme case is the most telling:
frozen weights + population decoder is barely above chance, because in
that configuration the *only* trainable parameter is the decoder's single
temperature scalar. Population coding's small, fixed-membership output
groups are a real capacity bottleneck that weight training alone only
partly compensates for.

### Delayed recall (online temporal memory — a targeted test of H4)

A binary value is shown for 5 steps, then `delay` steps of silence, then a
"go" pulse for 5 steps during which the model must report the value —
processing one timestep at a time, with no access to earlier input beyond
what its own state carries forward. Test accuracy, mean over 3 seeds:

| Delay (steps) | RGSN (online, recurrent) | Memoryless MLP (online) | Windowed MLP (offline, full sequence) |
|---|---|---|---|
| 0 | 97.3% | 50.1% | 97.1% |
| 10 | 97.0% | 50.1% | 97.4% |
| 20 (~1 membrane tau) | 96.0% | 50.1% | 96.9% |
| 40 (~2 tau) | 65.9% (+/-24%) | 50.1% | 96.9% |
| 80 (~4 tau) | 59.7% (+/-16%) | 50.1% | 96.6% |

The memoryless MLP is pinned at exactly chance for every delay *by
construction*: at query time its only visible input (the go pulse) is
statistically identical for both classes, so no amount of training can move
it off chance. RGSN, with no explicit buffer, matches the windowed MLP
(which is handed the entire sequence at once) almost exactly for delays up
to about one membrane time constant (tau_mem=20 steps here), then degrades
and becomes unreliable (high seed-to-seed variance) beyond ~2 tau — a
believable capacity limit of passive leaky-membrane memory, not a training
failure. This is a genuine structural capability gap: RGSN has a real,
non-tunable source of memory that a stateless feedforward net cannot have
regardless of training, in the specific (but common) setting of bounded
per-step online processing without an external history buffer.

## Scope and deviations from PLAN.md

Time/compute-boxed for this exercise; documented here rather than silently
diverging:

- **Scale**: N=300 (not 1000), T=30 (not 50-100), training-set sizes up to
  4000 (not the full 60k) and 3 seeds (not 5). The simulator itself is
  validated at the plan's target scale (N=1000, T=100, batch=64) in
  `tests/test_network.py::test_forward_pass_speed_batch64_n1000_t100`.
- **H4 was tested with a custom synthetic task, not SHD**: no
  `tonic`/SHD dependency was added; instead `rgsn.delayed_recall` implements
  a small online delayed-recall task purpose-built to isolate a temporal-
  memory capability gap (see Results). The architecture (`encoding.py`'s
  `passthrough_encode`, `RandomGraphSNN`'s discrete-time interface) is built
  to make adding a real event-based dataset like SHD a matter of writing a
  loader, not a redesign.
- **Phase 8 stretch goals are not implemented**: no synaptic delays, no
  local learning rules (e-prop / reward-modulated STDP), no topology
  analysis.
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
