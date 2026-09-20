"""Training entrypoint. Every run is fully defined by a config + seed; the
resolved config and a step-by-step CSV log are saved to
`runs/<timestamp>_<name>/`.

Supported `model` values (config.model):
  - "rgsn":            full weight training + structural rewiring (if enabled)
  - "rgsn_no_rewire":  full weight training, topology frozen (isolates rewiring)
  - "reservoir":       graph + weights frozen at init, only a linear readout trained
  - "mlp":             standard feed-forward baseline, parameter-matched
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from rgsn.baselines import MLP, hidden_size_for_budget
from rgsn.config import RunConfig
from rgsn.data import get_mnist_loaders
from rgsn.decoding import LinearReadoutDecoder, PopulationRateDecoder
from rgsn.encoding import poisson_rate_encode
from rgsn.graph import build_graph
from rgsn.network import RandomGraphSNN
from rgsn.rewiring import deep_r_step
from rgsn.utils import CsvLogger, count_params, load_yaml, save_yaml, seed_everything

N_FEATURES_MNIST = 28 * 28
N_CLASSES_MNIST = 10


class Model:
    """Thin wrapper unifying the four model types behind one train/eval API."""

    def __init__(self, cfg: RunConfig, gen: torch.Generator):
        self.cfg = cfg
        self.gen = gen
        self.kind = cfg.model
        self.net: RandomGraphSNN | None = None
        self.decoder: nn.Module | None = None
        self.mlp: MLP | None = None

        if self.kind in ("rgsn", "rgsn_no_rewire", "reservoir"):
            spec = build_graph(cfg.graph, gen)
            self.net = RandomGraphSNN(
                spec,
                cfg.neuron,
                n_features=N_FEATURES_MNIST,
                gain=cfg.graph.gain,
                train_input_weights=cfg.train.train_input_weights,
                seed=cfg.train.seed,
            )
            if self.kind == "reservoir":
                self.net.W_mag.requires_grad_(False)
                self.net.W_in.requires_grad_(False)
                self.decoder = LinearReadoutDecoder(self.net.n, cfg.graph.n_classes)
            else:
                self.decoder = PopulationRateDecoder(
                    spec.output_idx,
                    learnable_temperature=cfg.decoding.learnable_temperature,
                    temperature=cfg.decoding.temperature,
                )
        elif self.kind == "mlp":
            h = hidden_size_for_budget(
                param_budget=cfg.train.mlp_param_budget,
                n_features=N_FEATURES_MNIST,
                n_classes=N_CLASSES_MNIST,
            )
            self.mlp = MLP(N_FEATURES_MNIST, N_CLASSES_MNIST, hidden_sizes=[h])
        else:
            raise ValueError(f"Unknown model kind: {self.kind}")

    def parameters(self):
        if self.mlp is not None:
            return list(self.mlp.parameters())
        params = list(self.decoder.parameters())
        if self.net.W_mag.requires_grad:
            params.append(self.net.W_mag)
        if self.net.W_in.requires_grad:
            params.append(self.net.W_in)
        return params

    def param_counts(self) -> dict:
        if self.mlp is not None:
            return count_params(self.mlp)
        total = 0
        nonzero = 0
        for p in self.parameters():
            total += p.numel()
            nonzero += int((p != 0).sum().item())
        return {"total": total, "nonzero": nonzero}

    def logits(self, x_raw: torch.Tensor) -> torch.Tensor:
        if self.mlp is not None:
            return self.mlp(x_raw)
        x_enc = poisson_rate_encode(
            x_raw,
            t_steps=self.cfg.encoding.t_steps,
            max_rate_hz=self.cfg.encoding.max_rate_hz,
            dt_ms=self.cfg.encoding.dt_ms,
            generator=self.gen,
        )
        spikes = self.net(x_enc)
        logits = self.decoder(spikes)
        return logits, spikes

    def firing_rate_penalty(self, spikes: torch.Tensor) -> torch.Tensor:
        rate = self.net.firing_rate_hz(spikes, self.cfg.encoding.dt_ms)
        low, high = self.cfg.train.rate_reg_target_low, self.cfg.train.rate_reg_target_high
        below = torch.relu(low - rate)
        above = torch.relu(rate - high)
        return (below**2 + above**2).mean()


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def evaluate(model: Model, loader, max_batches: int | None = None) -> float:
    correct, total = 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        with torch.no_grad():
            out = model.logits(x)
            logits = out[0] if isinstance(out, tuple) else out
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train_run(cfg: RunConfig, run_dir: Path, dry_run: bool = False) -> Path:
    gen = seed_everything(cfg.train.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(asdict(cfg), run_dir / "config.yaml")

    train_loader, test_loader = get_mnist_loaders(
        n_train=cfg.train.n_train_samples,
        n_test=cfg.train.n_test_samples,
        batch_size=cfg.train.batch_size,
        seed=cfg.train.seed,
    )

    model = Model(cfg, gen)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.train.lr) if params else None
    loss_fn = nn.CrossEntropyLoss()

    step_logger = CsvLogger(run_dir / "steps.csv")
    epoch_logger = CsvLogger(run_dir / "epochs.csv")
    pcounts = model.param_counts()

    if dry_run:
        x, y = next(iter(train_loader))
        out = model.logits(x)
        logits = out[0] if isinstance(out, tuple) else out
        print(f"[dry-run] logits shape={tuple(logits.shape)} params={pcounts}")
        return run_dir

    global_step = 0
    t_start = time.time()
    for epoch in range(cfg.train.epochs):
        model_train_correct, model_train_total = 0, 0
        for x, y in train_loader:
            if opt is not None:
                opt.zero_grad()
            out = model.logits(x)
            if isinstance(out, tuple):
                logits, spikes = out
                loss = loss_fn(logits, y)
                if cfg.train.rate_reg_weight > 0:
                    penalty = model.firing_rate_penalty(spikes)
                    loss = loss + cfg.train.rate_reg_weight * penalty
                if cfg.rewiring.enabled and cfg.rewiring.l1_weight > 0:
                    active_mag = model.net.W_mag[model.net.mask].abs().mean()
                    loss = loss + cfg.rewiring.l1_weight * active_mag
            else:
                logits = out
                loss = loss_fn(logits, y)

            if opt is not None:
                loss.backward()
                if cfg.train.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip_norm)
                opt.step()

            if (
                cfg.model == "rgsn"
                and cfg.rewiring.enabled
                and global_step % cfg.rewiring.interval_steps == 0
            ):
                rew_stats = deep_r_step(model.net, cfg.rewiring.init_weight_scale, gen)
            else:
                n_edges = model.net.edge_count() if model.net else 0
                rew_stats = {"n_pruned": 0, "n_regrown": 0, "n_edges": n_edges}

            acc = _accuracy(logits, y)
            model_train_correct += (logits.argmax(dim=1) == y).sum().item()
            model_train_total += y.numel()
            step_logger.log(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss.item(),
                    "train_acc": acc,
                    "wall_time_s": time.time() - t_start,
                    **rew_stats,
                }
            )
            global_step += 1

        test_acc = evaluate(model, test_loader)
        train_acc = model_train_correct / max(1, model_train_total)
        epoch_logger.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "wall_time_s": time.time() - t_start,
                "n_edges": model.net.edge_count() if model.net else None,
                "param_total": pcounts["total"],
                "param_nonzero": pcounts["nonzero"],
            }
        )
        print(f"[{cfg.name}] epoch {epoch}: train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    torch.save(
        {
            "model_kind": model.kind,
            "state": (
                model.mlp.state_dict()
                if model.mlp is not None
                else {
                    "net": model.net.state_dict(),
                    "decoder": model.decoder.state_dict(),
                }
            ),
        },
        run_dir / "checkpoint.pt",
    )
    return run_dir


def load_model(run_dir: Path) -> tuple[Model, RunConfig]:
    """Rebuild a `Model` from a saved run directory (config.yaml + checkpoint.pt)."""
    from rgsn.utils import dataclass_from_dict

    raw = load_yaml(run_dir / "config.yaml")
    cfg = dataclass_from_dict(RunConfig, raw)
    gen = seed_everything(cfg.train.seed)
    model = Model(cfg, gen)
    ckpt = torch.load(run_dir / "checkpoint.pt", weights_only=True)
    if model.mlp is not None:
        model.mlp.load_state_dict(ckpt["state"])
    else:
        model.net.load_state_dict(ckpt["state"]["net"])
        model.decoder.load_state_dict(ckpt["state"]["decoder"])
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    from rgsn.utils import dataclass_from_dict

    raw = load_yaml(args.config)
    cfg = dataclass_from_dict(RunConfig, raw)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"{ts}_{cfg.name}"

    train_run(cfg, run_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
