"""
Evidential Deep Learning loss (Sensoy, Kaplan & Kandemir, NeurIPS 2018).

The network outputs evidence e_k >= 0 per class; alpha = e + 1 is a Dirichlet
over class probabilities, with strength S = sum(alpha). The loss has two parts:

  fit     how well the Dirichlet predicts the one-hot label y. Three variants
          from the paper; "mse" (Eq. 5) is the one the paper recommends:
            mse       sum_k (y_k - p_k)^2 + p_k (1 - p_k) / (S + 1),  p = alpha/S
            log       sum_k y_k (log S - log alpha_k)
            digamma   sum_k y_k (digamma(S) - digamma(alpha_k))
  KL      KL(Dir(alpha_tilde) || Dir(1)), where alpha_tilde removes the
          evidence for the true class: it penalises evidence for the WRONG
          classes, pushing the model towards "I don't know" rather than
          confidently wrong. Its weight is annealed from 0 to 1 over the first
          `anneal_epochs` epochs, so the model learns to fit before it is
          pushed to be modest.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_to_uniform_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1, ..., 1)) per row."""
    k = alpha.shape[1]
    s = alpha.sum(dim=1, keepdim=True)
    return (torch.lgamma(s).squeeze(1)
            - torch.lgamma(torch.tensor(float(k), device=alpha.device))
            - torch.lgamma(alpha).sum(dim=1)
            + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(s))).sum(dim=1))


def edl_loss(evidence: torch.Tensor, target: torch.Tensor, epoch: int,
             anneal_epochs: int = 10, variant: str = "mse") -> torch.Tensor:
    num_classes = evidence.shape[1]
    y = F.one_hot(target, num_classes).float()
    alpha = evidence + 1.0
    s = alpha.sum(dim=1, keepdim=True)
    if variant == "mse":
        p = alpha / s
        fit = ((y - p) ** 2 + p * (1 - p) / (s + 1)).sum(dim=1)
    elif variant == "log":
        fit = (y * (torch.log(s) - torch.log(alpha))).sum(dim=1)
    elif variant == "digamma":
        fit = (y * (torch.digamma(s) - torch.digamma(alpha))).sum(dim=1)
    else:
        raise ValueError(f"unknown EDL loss variant {variant!r}")
    alpha_tilde = y + (1 - y) * alpha
    coef = min(1.0, epoch / max(1, anneal_epochs))
    return (fit + coef * kl_to_uniform_dirichlet(alpha_tilde)).mean()
