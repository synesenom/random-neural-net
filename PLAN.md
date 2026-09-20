# PLAN.md — Random-Graph Spiking Network with Structural Plasticity

## 1. Goal

Build and evaluate a neural network that:

- starts as a **sparse random directed graph** of spiking neurons (no layers),
- receives input on an **input subset** of nodes,
- propagates activity as **spikes flowing through the graph over time**,
- produces output as **statistics over an output subset** (population coding), so losing nodes degrades output gracefully instead of breaking it,
- **learns by changing weights and rewiring edges** (add/remove) during training.

Then test whether it beats standard networks on:

- **H1 — Sample efficiency:** higher accuracy with little training data.
- **H2 — Learning speed:** fewer gradient steps / epochs to reach a target accuracy.
- **H3 — Robustness:** slower accuracy decay when neurons are ablated.
- **H4 — Temporal tasks:** better performance on inherently temporal data.

Related literature (for context, not dependencies): Liquid State Machines (Maass et al. 2002), DEEP R (Bellec et al. 2018), synaptic sampling (Kappel et al. 2015), SET (Mocanu et al. 2018), RigL (Evci et al. 2020), surrogate gradients (Neftci et al. 2019), e-prop (Bellec et al. 2020), polychronization (Izhikevich 2006).

## 2. Tech stack

- Python 3.11+, PyTorch 2.x
- Own LIF simulator in plain PyTorch (small, transparent, easy to modify). snnTorch may be used for its surrogate gradient functions only.
- `tonic` for neuromorphic datasets (SHD), `torchvision` for MNIST
- Config: dataclasses + YAML files
- Logging: CSV + TensorBoard
- Tests: pytest
- Tooling: `ruff` for lint/format, `uv` or `pip` + `pyproject.toml`

## 3. Repo layout

```
.
├── PLAN.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── rgsn_mnist.yaml
│   ├── rgsn_shd.yaml
│   ├── reservoir_mnist.yaml
│   └── baselines/*.yaml
├── src/rgsn/
│   ├── graph.py        # random graph generation, masks, Dale's law option
│   ├── neurons.py      # LIF dynamics, surrogate spike function
│   ├── network.py      # RandomGraphSNN: simulate T steps, input/output subsets
│   ├── encoding.py     # Poisson rate encoding, event data passthrough
│   ├── decoding.py     # population readouts (rate, first-spike, etc.)
│   ├── rewiring.py     # DEEP R / SET-style prune + regrow
│   ├── ablation.py     # neuron/edge removal at eval time
│   ├── baselines.py    # MLP, LSTM, fixed reservoir + linear readout
│   ├── data.py         # datasets, fixed-size subsets with seeds
│   ├── train.py        # training loop, entrypoint
│   ├── evaluate.py     # accuracy, ablation curves
│   └── utils.py        # seeding, logging, param counting
├── experiments/
│   ├── run_sample_efficiency.py
│   ├── run_ablation.py
│   └── run_learning_speed.py
├── notebooks/          # analysis and plots only
└── tests/
```

## 4. Model specification

### 4.1 Graph
- `N` neurons (default 1000). Directed, sparse, connection probability `p` (default 0.05). Self-loops off.
- Stored as dense weight matrix `W` (N×N) plus boolean mask `M`. Dense+mask is fine up to N≈5000; move to sparse tensors later only if needed.
- Graph types: Erdős–Rényi (default), Watts–Strogatz small-world (option).
- Optional **Dale's law**: 80% excitatory, 20% inhibitory; sign fixed per presynaptic neuron.
- Weight init: scaled so the spectral radius / average input keeps activity in a non-silent, non-saturated regime. Expose a `gain` parameter.

### 4.2 Neurons (discrete-time LIF)
```
v[t+1] = beta * v[t] + W_in @ x[t] + (W * M) @ s[t] - s[t] * v_th
s[t+1] = H(v[t+1] - v_th)      # surrogate gradient in backward pass
```
- `beta` = exp(-dt/tau_mem), default tau_mem = 20 ms, dt = 1 ms.
- Reset by subtraction. Optional refractory period.
- Surrogate: fast sigmoid (slope 25), configurable.
- Later option: synaptic delays (ring buffer of past spikes per delay bucket).

### 4.3 Input
- `N_in` input neurons (default 20% of N), a fixed random subset.
- Input projection `W_in` maps features to input neurons only. Option to learn or freeze it.
- MNIST: Poisson rate encoding, T = 50–100 steps. SHD: native spike events binned to dt.

### 4.4 Output (population readout)
- `C` classes, each with a disjoint output group of `k` neurons (default k = 10), disjoint from input neurons.
- Logit for class c = mean spike count of group c over the readout window (optionally scaled by a learnable temperature).
- Ablated neurons contribute zero, so the mean fades instead of breaking.
- Alternative decoders to try: mean membrane potential; learned linear readout over *all* output neurons (for comparison, not the main model).

### 4.5 Learning
- Loss: cross-entropy on population logits + small firing-rate regularizer (keep rates in e.g. 1–50 Hz band).
- Weight learning: BPTT with surrogate gradients, Adam.
- **Structural learning (DEEP R-style):**
  - Each active edge has a parameter with a fixed sign. If an update would flip its sign, the edge is removed.
  - After each step (or every `r` steps), regrow the same number of edges at random inactive positions with small initial weight, keeping total edge count constant.
  - Variant (RigL-style): regrow where the dense gradient magnitude is highest. Requires computing the gradient for masked entries every `r` steps.
- Stretch: e-prop (online, local), STDP + reward modulation, evolutionary topology search.

## 5. Baselines

All baselines matched by **trainable parameter count** (report both total and nonzero counts).

1. **MLP** (1–2 hidden layers, ReLU).
2. **LSTM / GRU** for temporal tasks (SHD).
3. **Fixed reservoir (LSM):** same random SNN, frozen, train linear readout only.
4. **RGSN without rewiring:** weights trained, topology frozen (isolates the effect of rewiring).
5. **Layered SNN:** feedforward spiking network with the same neuron model (isolates the effect of the random graph vs layers).

## 6. Phases and milestones

Each phase ends with passing tests and a short note in `README.md`.

### Phase 0 — Setup
- `pyproject.toml`, ruff, pytest, seeding utility, config loading, logging.
- **Done when:** `pytest` runs, `python -m rgsn.train --config configs/rgsn_mnist.yaml --dry-run` works.

### Phase 1 — Simulator core
- `graph.py`, `neurons.py`, `network.py`.
- Tests: shapes; masked edges carry no current and get no gradient; with zero input, network is silent; firing rate increases with gain; deterministic under fixed seed.
- **Done when:** forward pass of batch 64, N=1000, T=100 runs on CPU in a few seconds, and gradients flow to `W` and `W_in`.

### Phase 2 — Encoding and decoding
- Poisson encoder, SHD loader, population decoder.
- Tests: encoder rates match pixel intensity; decoder output ignores non-output neurons; ablating an output neuron lowers its group mean but keeps it finite.

### Phase 3 — Fixed reservoir baseline
- Freeze graph, train readout on MNIST.
- Tune `gain`, `p`, `tau_mem` so activity is in a healthy range. Log firing-rate histograms.
- **Done when:** clearly above chance on MNIST (target > 85%), results logged.

### Phase 4 — Full weight training
- Surrogate-gradient BPTT on `W` (and optionally `W_in`) with population readout.
- **Done when:** beats Phase 3 reservoir on MNIST; training is stable across 3 seeds.

### Phase 5 — Rewiring
- Implement DEEP R prune/regrow; then RigL variant.
- Log edges pruned/regrown per step, total edge count (must stay constant), degree distribution over time.
- **Done when:** runs stably and matches or beats Phase 4 at the same edge count.

### Phase 6 — Baselines
- Implement and tune all baselines from Section 5 with the same data pipeline and budget.

### Phase 7 — Experiments (the actual research)
- **Sample efficiency (H1):** train on 100 / 300 / 1000 / 3000 / 10000 / full MNIST samples, 5 seeds each. Plot test accuracy vs training set size for every model.
- **Learning speed (H2):** accuracy vs gradient steps and vs wall-clock time; steps to reach fixed accuracy thresholds.
- **Robustness (H3):** at eval, remove random 0–50% of (a) hidden neurons, (b) output neurons, (c) edges. Do the same for baselines (zero hidden units). Plot accuracy vs fraction removed.
- **Temporal (H4):** repeat H1–H3 on SHD against LSTM/GRU and layered SNN.
- Report mean ± std over seeds. Use the same hyperparameter search budget for every model.

### Phase 8 — Stretch
- Synaptic delays (polychronous dynamics).
- Local learning rules (e-prop, reward-modulated STDP) for comparison with BPTT.
- Growing/shrinking the neuron count, not just edges.
- Topology analysis: do learned graphs develop modules, hubs, or motifs?

## 7. Default hyperparameters (starting point)

| Param | Default |
|---|---|
| N | 1000 |
| p (connection prob.) | 0.05 |
| input fraction | 0.2 |
| output group size k | 10 |
| tau_mem | 20 ms |
| dt | 1 ms |
| T (MNIST) | 50 |
| v_th | 1.0 |
| optimizer | Adam, lr 1e-3 |
| batch size | 128 |
| rewiring interval r | 1 step (DEEP R), 100 steps (RigL) |
| RigL regrow fraction | 0.1, cosine-decayed |
| rate regularizer target | 5–20 spikes/neuron/sample window |

## 8. Risks and mitigations

- **Silent or exploding activity:** monitor firing rates every epoch; rate regularizer; tune gain in Phase 3 before training.
- **Vanishing gradients through long recurrent chains:** keep T small first; truncated BPTT; check gradient norms per step.
- **Slow training (BPTT over T steps):** use GPU; start with small N and T; profile before optimizing.
- **Unfair comparisons:** match parameter counts, tuning budget, and data; report all seeds; include the "no rewiring" and "layered SNN" ablations so any gain is attributable.
- **Expectation check:** SNNs and reservoirs usually do not beat standard nets on static-task accuracy. The most likely wins are robustness (H3) and small-data temporal tasks (H4). Negative results are still results; record them.

## 9. Conventions for Claude Code

- Implement one phase at a time; write tests with each module.
- Every run is fully defined by a config file + seed; save the resolved config next to results in `runs/<timestamp>_<name>/`.
- No hidden global state; all randomness through a passed `torch.Generator` or seeded at entry.
- Keep modules small and typed (type hints, docstrings on public functions).
- Plots are generated from saved CSVs, never from in-memory training state.
