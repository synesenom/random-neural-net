"""Seeding, config loading, logging, and parameter-counting helpers."""

from __future__ import annotations

import csv
import dataclasses
import random
import typing
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> torch.Generator:
    """Seed python/numpy/torch global RNGs and return a dedicated generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def dataclass_from_dict(cls: type, data: dict[str, Any]):
    """Build a (possibly nested) dataclass instance from a plain dict, ignoring
    keys that aren't fields and filling in defaults for missing ones."""
    field_types = typing.get_type_hints(cls)
    kwargs = {}
    for name, ftype in field_types.items():
        if name not in data:
            continue
        value = data[name]
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            value = dataclass_from_dict(ftype, value)
        kwargs[name] = value
    return cls(**kwargs)


def count_params(module: torch.nn.Module) -> dict[str, int]:
    """Return total and nonzero trainable parameter counts for a module."""
    total = 0
    nonzero = 0
    for p in module.parameters():
        if not p.requires_grad:
            continue
        total += p.numel()
        nonzero += int((p != 0).sum().item())
    return {"total": total, "nonzero": nonzero}


class CsvLogger:
    """Append-only CSV logger; writes header on first row."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None
        if self.path.exists():
            self.path.unlink()

    def log(self, row: dict[str, Any]) -> None:
        write_header = self._fieldnames is None
        if write_header:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
