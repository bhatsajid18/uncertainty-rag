"""
Run the trained models on every evaluation set and save their outputs.

For each method and each set (CIFAR-10 test, SVHN, CIFAR-100, and the
corrupted CIFAR-10 variants) this writes one .npz with the predictive
probabilities, the labels (-1 for OOD sets) and every uncertainty score:

  softmax     one network (seed 0)                  msp, entropy
  ensemble    five networks (seeds 0-4), averaged    msp, entropy, mutual_information
  mcdropout   one network, `mc_samples` passes with  msp, entropy, mutual_information
              dropout left on
  edl         one network, Dirichlet output          msp, entropy, vacuity

Metrics are computed separately (evaluate.py) from these files, so changing a
metric or a plot never needs the GPU again. This is the step that is slow on a
laptop - MC Dropout makes 20 passes - and fast on Kaggle: run it wherever the
checkpoints are, and copy results/uncertainty/ if that's not your Mac.

Usage (from the repo root):
  python src/uncertainty/predict.py                      # all methods, all sets
  python src/uncertainty/predict.py --methods edl softmax --no-corruptions
  python src/uncertainty/predict.py --fake               # after train.py --fake
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# src/ on the path, so these are imported as the `uncertainty` package and
# cannot collide with same-named modules elsewhere (src/evaluation/metrics.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncertainty.config import load_config  # noqa: E402
from uncertainty.data import DataConfig, eval_datasets, eval_loader  # noqa: E402
from uncertainty.models import build_model, enable_mc_dropout, pick_device  # noqa: E402
from uncertainty.scores import (  # noqa: E402
    scores_from_alpha, scores_from_probs, scores_from_samples,
)

METHODS = ("softmax", "ensemble", "mcdropout", "edl")


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mc = ckpt["config"]["model"]
    model = build_model(ckpt["method"], dropout=mc["dropout"], width=mc["width"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def _run(model, loader, device, fn) -> tuple[np.ndarray, np.ndarray]:
    outs, labels = [], []
    for x, y in loader:
        outs.append(fn(model, x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(outs).numpy(), torch.cat(labels).numpy()


def softmax_probs(model, x):
    return torch.softmax(model(x).float(), dim=1)


def dirichlet_alpha(model, x):
    return model.evidence(model(x).float()) + 1.0


def predict_set(method: str, models: list, loader, device, mc_samples: int) -> dict:
    """Outputs and scores of one method on one evaluation set."""
    if method in ("softmax", "ensemble"):
        members = [_run(m, loader, device, softmax_probs) for m in models]
        labels = members[0][1]
        if method == "softmax":
            return {**scores_from_probs(members[0][0]), "labels": labels}
        return {**scores_from_samples(np.stack([p for p, _ in members])),
                "labels": labels}
    if method == "mcdropout":
        model = models[0]
        enable_mc_dropout(model)
        samples = [_run(model, loader, device, softmax_probs) for _ in range(mc_samples)]
        model.eval()
        return {**scores_from_samples(np.stack([p for p, _ in samples])),
                "labels": samples[0][1]}
    alpha, labels = _run(models[0], loader, device, dirichlet_alpha)
    return {**scores_from_alpha(alpha), "labels": labels}


def checkpoints_for(method: str, root: Path, ensemble_size: int) -> list[Path]:
    if method == "softmax":
        return [root / "softmax" / "seed0.pt"]
    if method == "ensemble":
        return [root / "softmax" / f"seed{s}.pt" for s in range(ensemble_size)]
    return [root / method / "seed0.pt"]


def set_filename(name: str) -> str:
    return name.replace("/", "__")


def main():
    ap = argparse.ArgumentParser(description="Predict with the trained models.")
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--mc-samples", type=int)
    ap.add_argument("--corruption-subset", type=int)
    ap.add_argument("--no-corruptions", action="store_true",
                    help="Skip the corruption sets (much faster on a laptop).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--force", action="store_true", help="Recompute existing outputs.")
    args = ap.parse_args()

    cfg = load_config(args.config, {"predict": {"mc_samples": args.mc_samples,
                                                "corruption_subset": args.corruption_subset}})
    pc = cfg["predict"]
    sub = "fake" if args.fake else ""
    ckpt_root = Path(cfg["paths"]["checkpoints"]) / sub
    out_dir = Path(cfg["paths"]["results"]) / sub / "predictions"
    data_cfg = DataConfig(root=Path(cfg["data"]["root"]), batch_size=cfg["data"]["batch_size"],
                          num_workers=0 if args.fake else cfg["data"]["num_workers"],
                          fake=args.fake)
    sets = eval_datasets(data_cfg, corruption_subset=pc["corruption_subset"],
                         corruptions=not args.no_corruptions)
    device = pick_device(args.device)
    mc_samples = min(pc["mc_samples"], 3) if args.fake else pc["mc_samples"]
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = []
    for method in args.methods:
        paths = checkpoints_for(method, ckpt_root, cfg["train"]["ensemble_size"])
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"  {method}: missing {', '.join(map(str, missing))}; skipping "
                  "(run train.py first)")
            skipped.append(method)
            continue
        todo = {n: d for n, d in sets.items()
                if args.force or not (out_dir / f"{method}__{set_filename(n)}.npz").exists()}
        if not todo:
            print(f"  {method}: all outputs exist")
            continue
        models = [load_model(p, device)[0] for p in paths]
        print(f"  {method}: {len(todo)} set(s) on {device}")
        for name, ds in todo.items():
            res = predict_set(method, models, eval_loader(ds, data_cfg), device, mc_samples)
            out = out_dir / f"{method}__{set_filename(name)}.npz"
            tmp = out.with_name(out.name + ".tmp")  # no half-written file on a crash
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **{k: np.asarray(v, dtype=np.float32 if k != "labels"
                                                        else np.int64)
                                          for k, v in res.items()})
            os.replace(tmp, out)
            print(f"    {name:32s} {len(res['labels']):>6d} inputs")
    print(f"\nOutputs in {out_dir}/. Next: python src/uncertainty/evaluate.py"
          + (" --fake" if args.fake else ""))
    if skipped:
        # Fail loudly: a notebook run must not package a benchmark with methods
        # silently missing from it.
        sys.exit(f"No checkpoints for: {', '.join(skipped)}. Train them first, or "
                 "leave them out with --methods.")


if __name__ == "__main__":
    main()
