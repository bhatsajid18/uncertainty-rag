"""
Datasets for the uncertainty benchmark.

  in-distribution    CIFAR-10 (train for training, test for evaluation)
  near OOD           CIFAR-100 test: same kind of photos, different classes
  far OOD            SVHN test: house-number digits, a different domain
  dataset shift      CIFAR-10 test with synthetic corruptions at five
                     severities, in the spirit of CIFAR-10-C (Hendrycks &
                     Dietterich 2019) without the 3 GB download

Every evaluation set is normalised with the CIFAR-10 training statistics: the
model sees OOD inputs exactly as it would see any other input.

`fake=True` swaps in small random-image datasets so the whole pipeline (train,
predict, evaluate) can be smoke-tested in a minute without downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10

# Severity 1-5 parameters, taken from or modelled on CIFAR-10-C.
CORRUPTIONS = {
    "gaussian_noise": (0.04, 0.06, 0.08, 0.09, 0.10),   # noise std
    "gaussian_blur": (0.4, 0.6, 0.7, 0.8, 1.0),        # blur sigma (pixels)
    "contrast": (0.75, 0.5, 0.4, 0.3, 0.15),            # contrast factor
    "brightness": (0.05, 0.1, 0.15, 0.2, 0.3),          # added brightness
    "pixelate": (0.95, 0.9, 0.85, 0.75, 0.65),          # downscale factor
}
OOD_SETS = ("svhn", "cifar100")


def _normalize():
    import torchvision.transforms as T
    return T.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def train_transform():
    import torchvision.transforms as T
    return T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(), _normalize()])


def test_transform(normalize: bool = True):
    import torchvision.transforms as T
    return T.Compose([T.ToTensor(), _normalize()]) if normalize else T.ToTensor()


def corrupt(x: torch.Tensor, kind: str, severity: int,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply one corruption to an image tensor in [0, 1], shape (3, H, W)."""
    import torchvision.transforms.functional as TF

    if kind not in CORRUPTIONS or not 1 <= severity <= 5:
        raise ValueError(f"unknown corruption {kind!r} / severity {severity}")
    p = CORRUPTIONS[kind][severity - 1]
    if kind == "gaussian_noise":
        noise = torch.randn(x.shape, generator=generator) * p
        out = x + noise
    elif kind == "gaussian_blur":
        k = 2 * int(round(3 * p)) + 1
        out = TF.gaussian_blur(x, kernel_size=[k, k], sigma=[p, p])
    elif kind == "contrast":
        mean = x.mean(dim=(1, 2), keepdim=True)
        out = (x - mean) * p + mean
    elif kind == "brightness":
        out = x + p
    else:  # pixelate: average down (area = box filter), blow back up blocky
        import torch.nn.functional as F

        h, w = x.shape[-2:]
        small = F.interpolate(x[None], size=(max(1, int(h * p)), max(1, int(w * p))),
                              mode="area")
        out = F.interpolate(small, size=(h, w), mode="nearest")[0]
    return out.clamp(0.0, 1.0)


class CorruptedDataset(Dataset):
    """A test set with one corruption applied, deterministically per index."""

    def __init__(self, base: Dataset, kind: str, severity: int, seed: int = 0):
        self.base, self.kind, self.severity, self.seed = base, kind, severity, seed
        self.normalize = _normalize()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]  # base yields un-normalised tensors in [0, 1]
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        return self.normalize(corrupt(x, self.kind, self.severity, g)), y


class _Relabel(Dataset):
    """OOD sets carry label -1: their labels mean nothing to a CIFAR-10 model."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, -1


def _fake(n: int, seed: int, transform, shift: float = 0.0):
    """Random images with a class-dependent tint, so a model can learn a little
    and OOD sets (shifted colours) are separable - enough to exercise the code."""
    import torchvision.transforms.functional as TF

    class Fake(Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(seed * 100_003 + i)
            y = i % NUM_CLASSES
            x = torch.rand(3, 32, 32, generator=g) * 0.5
            x[y % 3] += 0.3 + 0.02 * y
            x = (x + shift).clamp(0, 1)
            return (transform(TF.to_pil_image(x)) if transform else x), y
    return Fake()


@dataclass
class DataConfig:
    root: Path = Path("data/torchvision")
    batch_size: int = 128
    num_workers: int = 2
    fake: bool = False
    fake_size: int = 256


def train_loader(cfg: DataConfig, seed: int = 0) -> DataLoader:
    if cfg.fake:
        ds = _fake(cfg.fake_size, seed=1, transform=train_transform())
    else:
        from torchvision.datasets import CIFAR10
        ds = CIFAR10(cfg.root, train=True, download=True, transform=train_transform())
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=g,
                      num_workers=cfg.num_workers, drop_last=True,
                      pin_memory=torch.cuda.is_available())


def cifar10_test(cfg: DataConfig, normalize=True):
    """The CIFAR-10 test set alone (training needs only this, not the OOD sets)."""
    if cfg.fake:
        return _fake(cfg.fake_size // 2, seed=2, transform=test_transform(normalize))
    from torchvision.datasets import CIFAR10
    return CIFAR10(cfg.root, train=False, download=True,
                   transform=test_transform(normalize))


def eval_datasets(cfg: DataConfig, corruption_subset: int = 2000,
                  corruptions: bool = True, seed: int = 0) -> dict[str, Dataset]:
    """Every evaluation set by name: cifar10, svhn, cifar100, and
    cifar10-c/<kind>/<severity> for the corruptions (a fixed random subset of
    the test set per corruption, to keep MC Dropout affordable)."""
    sets: dict[str, Dataset] = {"cifar10": cifar10_test(cfg)}
    if cfg.fake:
        sets["svhn"] = _Relabel(_fake(cfg.fake_size // 2, 3, test_transform(), 0.4))
        sets["cifar100"] = _Relabel(_fake(cfg.fake_size // 2, 4, test_transform(), 0.1))
    else:
        from torchvision.datasets import CIFAR100, SVHN
        sets["svhn"] = _Relabel(SVHN(cfg.root, split="test", download=True,
                                     transform=test_transform()))
        sets["cifar100"] = _Relabel(CIFAR100(cfg.root, train=False, download=True,
                                             transform=test_transform()))
    if corruptions:
        raw = cifar10_test(cfg, normalize=False)
        n = min(corruption_subset, len(raw))
        idx = torch.randperm(len(raw), generator=torch.Generator().manual_seed(seed))[:n]
        sub = Subset(raw, idx.tolist())
        for kind in CORRUPTIONS:
            for sev in range(1, 6):
                sets[f"cifar10-c/{kind}/{sev}"] = CorruptedDataset(sub, kind, sev, seed)
    return sets


def eval_loader(ds: Dataset, cfg: DataConfig) -> DataLoader:
    return DataLoader(ds, batch_size=max(cfg.batch_size, 256), shuffle=False,
                      num_workers=cfg.num_workers,
                      pin_memory=torch.cuda.is_available())
