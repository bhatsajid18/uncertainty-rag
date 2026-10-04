"""
Train the models for the uncertainty benchmark. Meant for a free Kaggle GPU.

  softmax     ResNet-18 + cross-entropy. Seeds 0..4 are the Deep Ensemble;
              seed 0 alone is the softmax baseline, so no extra training.
  mcdropout   the same network with dropout in every residual block.
  edl         the same network with an evidential (Dirichlet) output and the
              EDL loss.

All use the same schedule (SGD + Nesterov, cosine LR, 30 epochs, standard
crop/flip augmentation), so the comparison is between methods, not recipes.

Each model is saved to checkpoints/<method>/seed<N>.pt with its training
history; a model whose checkpoint exists is skipped, so a run cut short by a
Kaggle session limit resumes with the next model.

Usage (from the repo root):
  python src/uncertainty/train.py --method all                 # all 7 models
  python src/uncertainty/train.py --method all --gpus 2        # Kaggle T4 x2
  python src/uncertainty/train.py --method edl --epochs 5
  python src/uncertainty/train.py --method all --fake --epochs 1   # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

# src/ on the path, so these are imported as the `uncertainty` package and
# cannot collide with same-named modules elsewhere (src/evaluation/metrics.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncertainty.config import load_config  # noqa: E402
from uncertainty.data import DataConfig, cifar10_test, eval_loader, train_loader  # noqa: E402
from uncertainty.losses import edl_loss  # noqa: E402
from uncertainty.models import build_model, pick_device  # noqa: E402

METHODS = ("softmax", "mcdropout", "edl")


def all_jobs(ensemble_size: int) -> list[tuple[str, int]]:
    return ([("softmax", s) for s in range(ensemble_size)]
            + [("mcdropout", 0), ("edl", 0)])


def checkpoint_path(root: Path, method: str, seed: int) -> Path:
    return Path(root) / method / f"seed{seed}.pt"


@torch.no_grad()
def test_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / max(1, total)


def train_one(method: str, seed: int, cfg: dict, data_cfg: DataConfig,
              device: torch.device, out_path: Path, log=print) -> dict:
    """Train one model and save it. Returns the saved metadata."""
    torch.manual_seed(seed)
    tc, mc = cfg["train"], cfg["model"]
    model = build_model(method, dropout=mc["dropout"], width=mc["width"]).to(device)
    loader = train_loader(data_cfg, seed=seed)
    test_dl = eval_loader(cifar10_test(data_cfg), data_cfg)

    opt = torch.optim.SGD(model.parameters(), lr=tc["lr"], momentum=tc["momentum"],
                          weight_decay=tc["weight_decay"], nesterov=True)
    steps = tc["epochs"] * len(loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    use_amp = bool(tc["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    for epoch in range(tc["epochs"]):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # autocast only on CUDA: some torch versions reject device_type="mps"
            # even with enabled=False
            with (torch.autocast(device_type="cuda", dtype=torch.float16)
                  if use_amp else nullcontext()):
                logits = model(x)
            logits = logits.float()  # losses in float32: lgamma/digamma need it
            if method == "edl":
                loss = edl_loss(model.evidence(logits), y, epoch,
                                tc["edl_anneal_epochs"], tc["edl_loss"])
            else:
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        row = {"epoch": epoch + 1, "loss": loss_sum / seen, "train_acc": correct / seen,
               "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
        if (epoch + 1) % 5 == 0 or epoch + 1 == tc["epochs"]:
            row["test_acc"] = test_accuracy(model, test_dl, device)
        history.append(row)
        log(f"  {method} seed{seed} epoch {epoch + 1:>3}/{tc['epochs']}  "
            f"loss {row['loss']:.4f}  train {row['train_acc']:.3f}"
            + (f"  test {row['test_acc']:.3f}" if "test_acc" in row else "")
            + f"  ({row['seconds']}s)")

    meta = {"method": method, "seed": seed, "config": cfg, "fake_data": data_cfg.fake,
            "test_acc": history[-1].get("test_acc"), "history": history,
            "device": str(device)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename: a run cut off mid-save must not leave a truncated .pt
    # that the resume logic would then skip and predict.py fail to load.
    tmp = out_path.parent / (out_path.name + ".tmp")
    torch.save({"state_dict": model.state_dict(), **meta}, tmp)
    os.replace(tmp, out_path)
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return meta


def _spawn_on_gpus(jobs, n_gpus: int, args) -> int:
    """One worker process per GPU, each training its share of the models one at
    a time. Kaggle's T4 x2 roughly halves the wall-clock time."""
    passthrough = []
    for flag, value in (("--config", args.config), ("--epochs", args.epochs),
                        ("--lr", args.lr), ("--batch-size", args.batch_size),
                        ("--edl-loss", args.edl_loss)):
        if value is not None:
            passthrough += [flag, str(value)]
    passthrough += ["--force"] if args.force else []
    passthrough += ["--fake"] if args.fake else []
    procs = []
    for gpu in range(n_gpus):
        share = jobs[gpu::n_gpus]
        if not share:
            continue
        spec = ",".join(f"{m}:{s}" for m, s in share)
        print(f"  cuda:{gpu} <- {spec}")
        procs.append((gpu, subprocess.Popen([sys.executable, __file__, *passthrough,
                                             "--jobs", spec, "--device", f"cuda:{gpu}"])))
    codes = [(gpu, p.wait()) for gpu, p in procs]
    # A worker killed by a signal (out of memory, say) returns a NEGATIVE code,
    # so max() would read it as success.
    failed = [f"cuda:{gpu} (exit {c})" for gpu, c in codes if c != 0]
    if failed:
        print(f"Training failed on {', '.join(failed)}. Finished models are saved; "
              "run the same command again to train the rest.", file=sys.stderr)
        return 1
    return 0


def parse_jobs(spec: str) -> list[tuple[str, int]]:
    """"softmax:0,edl:0" -> [("softmax", 0), ("edl", 0)]"""
    out = []
    for item in spec.split(","):
        method, seed = item.split(":")
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        out.append((method, int(seed)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Train uncertainty-benchmark models.")
    ap.add_argument("--method", choices=(*METHODS, "all"), default="all")
    ap.add_argument("--seeds", type=int, nargs="+",
                    help="Seeds to train (default: 0; softmax with --method all: "
                         "0..ensemble_size-1).")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--edl-loss", choices=("mse", "log", "digamma"))
    ap.add_argument("--device", default=None, help="cuda, cuda:1, mps or cpu.")
    ap.add_argument("--gpus", type=int, default=1,
                    help="Spread --method all over this many GPUs.")
    ap.add_argument("--fake", action="store_true",
                    help="Tiny random datasets, no downloads: checks the code runs.")
    ap.add_argument("--force", action="store_true", help="Retrain existing models.")
    ap.add_argument("--jobs", help=argparse.SUPPRESS)  # used by --gpus workers
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "train": {"epochs": args.epochs, "lr": args.lr, "edl_loss": args.edl_loss},
        "data": {"batch_size": args.batch_size}})
    if args.fake:
        # A smoke test: small and gentle, so it runs in a minute on a CPU and the
        # models neither diverge nor saturate (a saturated model makes every MC
        # Dropout pass identical, and the check would test nothing).
        cfg["data"]["batch_size"] = min(cfg["data"]["batch_size"], 32)
        cfg["model"]["width"] = min(cfg["model"]["width"], 16)
        if args.lr is None:
            cfg["train"]["lr"] = 0.01
    data_cfg = DataConfig(root=Path(cfg["data"]["root"]),
                          batch_size=cfg["data"]["batch_size"],
                          num_workers=0 if args.fake else cfg["data"]["num_workers"],
                          fake=args.fake)
    ckpt_root = Path(cfg["paths"]["checkpoints"]) / ("fake" if args.fake else "")

    if args.jobs:
        jobs = parse_jobs(args.jobs)
    elif args.method == "all":
        jobs = all_jobs(cfg["train"]["ensemble_size"])
    else:
        jobs = [(args.method, s) for s in (args.seeds or [0])]
    todo = [(m, s) for m, s in jobs
            if args.force or not checkpoint_path(ckpt_root, m, s).exists()]
    for m, s in jobs:
        if (m, s) not in todo:
            print(f"  {m} seed{s}: checkpoint exists, skipping (--force to retrain)")
    if not todo:
        print("Nothing to train.")
        return

    if args.gpus > 1 and not args.jobs:
        if torch.cuda.device_count() < args.gpus:
            sys.exit(f"--gpus {args.gpus} but only {torch.cuda.device_count()} "
                     "CUDA device(s) are visible.")
        if not args.fake:
            # Download once here: two workers fetching CIFAR-10 into the same
            # folder at the same time can leave a corrupt archive behind.
            # (SVHN and CIFAR-100 are only needed later, by predict.py.)
            print("Downloading CIFAR-10 once before starting the workers ...")
            train_loader(data_cfg)
            cifar10_test(data_cfg)
        sys.exit(_spawn_on_gpus(todo, args.gpus, args))

    device = pick_device(args.device)
    print(f"Training {len(todo)} model(s) on {device}"
          + (" with fake data" if args.fake else "") + ".")
    for m, s in todo:
        meta = train_one(m, s, cfg, data_cfg, device, checkpoint_path(ckpt_root, m, s))
        print(f"  saved {m} seed{s}: test accuracy {meta['test_acc']:.4f}")
    print(f"\nCheckpoints in {ckpt_root}/. Next: python src/uncertainty/predict.py"
          + (" --fake" if args.fake else ""))


if __name__ == "__main__":
    main()
