"""
ResNet-18 for 32x32 images, with the switches each method needs.

The CIFAR variant of ResNet-18 (3x3 stem, no max-pool, He et al. 2016) is the
standard backbone in the uncertainty literature at this scale, and trains to
~94% on CIFAR-10 in 30 epochs on a free Kaggle GPU.

  dropout > 0   Dropout inside every residual block (between its two convs)
                and before the classifier. At test time MC Dropout keeps these
                layers active and averages many stochastic passes.
  evidential    The output is non-negative evidence per class
                (softplus of the logits); alpha = evidence + 1 parameterises a
                Dirichlet over class probabilities (Sensoy et al. 2018).

The backbone is otherwise identical across methods, so differences in the
results come from the method, not the architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(self.drop(out)))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, dropout=0.0, evidential=False, width=64):
        super().__init__()
        self.evidential = evidential
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        layers, in_planes = [], width
        for i, planes in enumerate([width, width * 2, width * 4, width * 8]):
            stride = 1 if i == 0 else 2
            layers += [BasicBlock(in_planes, planes, stride, dropout),
                       BasicBlock(planes, planes, 1, dropout)]
            in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(self.drop(out))

    def evidence(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softplus(logits)


def build_model(method: str, dropout: float = 0.1, width: int = 64,
                num_classes: int = 10) -> ResNet18:
    return ResNet18(num_classes=num_classes,
                    dropout=dropout if method == "mcdropout" else 0.0,
                    evidential=(method == "edl"), width=width)


def enable_mc_dropout(model: nn.Module) -> int:
    """Eval mode everywhere (BatchNorm uses its running statistics), except the
    dropout layers, which stay stochastic. Returns how many were switched on."""
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n += 1
    return n


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
