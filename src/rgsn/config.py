"""Dataclass configs resolved from YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraphConfig:
    n: int = 1000
    p: float = 0.05
    graph_type: str = "erdos_renyi"  # or "watts_strogatz"
    ws_k: int = 10
    ws_beta: float = 0.1
    input_fraction: float = 0.2
    output_group_size: int = 10
    n_classes: int = 10
    dales_law: bool = False
    excitatory_fraction: float = 0.8
    gain: float = 1.0
    self_loops: bool = False


@dataclass
class NeuronConfig:
    tau_mem_ms: float = 20.0
    dt_ms: float = 1.0
    v_th: float = 1.0
    surrogate_slope: float = 25.0
    refractory_steps: int = 0


@dataclass
class EncodingConfig:
    scheme: str = "poisson"  # "poisson" or "passthrough"
    t_steps: int = 50
    max_rate_hz: float = 100.0
    dt_ms: float = 1.0


@dataclass
class DecodingConfig:
    scheme: str = "population_rate"  # "population_rate", "membrane", "linear_readout"
    temperature: float = 1.0
    learnable_temperature: bool = False


@dataclass
class RewiringConfig:
    enabled: bool = False
    strategy: str = "deep_r"  # "deep_r" or "rigl"
    interval_steps: int = 1
    regrow_fraction: float = 1.0
    init_weight_scale: float = 0.01
    l1_weight: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    optimizer: str = "adam"
    rate_reg_weight: float = 0.0
    rate_reg_target_low: float = 5.0
    rate_reg_target_high: float = 20.0
    train_input_weights: bool = True
    seed: int = 0
    n_train_samples: int | None = None
    n_test_samples: int | None = None
    mlp_param_budget: int = 50_000
    grad_clip_norm: float = 1.0


@dataclass
class RunConfig:
    name: str = "rgsn_run"
    model: str = "rgsn"  # "rgsn", "reservoir", "rgsn_no_rewire", "mlp", "layered_snn"
    dataset: str = "mnist"
    device: str = "cpu"
    graph: GraphConfig = field(default_factory=GraphConfig)
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    rewiring: RewiringConfig = field(default_factory=RewiringConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
