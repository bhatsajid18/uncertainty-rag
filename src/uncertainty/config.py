"""Load configs/uncertainty.yaml and apply command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path

DEFAULTS = {
    "data": {"root": "data/torchvision", "batch_size": 128, "num_workers": 2},
    "model": {"width": 64, "dropout": 0.1},
    "train": {"epochs": 30, "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
              "amp": True, "ensemble_size": 5, "edl_loss": "digamma",
              "edl_anneal_epochs": 10},
    "predict": {"mc_samples": 20, "corruption_subset": 2000},
    "paths": {"checkpoints": "checkpoints", "results": "results/uncertainty"},
}
DEFAULT_PATH = Path("configs/uncertainty.yaml")


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path | None = None, overrides: dict | None = None) -> dict:
    """Defaults, then the YAML file if present, then explicit overrides
    (a nested dict; None values are ignored so unset CLI flags don't clobber)."""
    cfg = copy.deepcopy(DEFAULTS)
    path = path or DEFAULT_PATH
    if path and Path(path).exists():
        import yaml
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    for section, values in (overrides or {}).items():
        cfg[section] = _merge(cfg[section], {k: v for k, v in values.items()
                                             if v is not None})
    return cfg
