"""
Uncertainty scores: turn a method's output into "how unsure is it?".

Higher always means more uncertain, so every score can be fed straight to the
OOD metrics (which treat OOD as the positive class).

  msp        1 - max softmax probability (Hendrycks & Gimpel 2017).
  entropy    entropy of the (mean) predictive distribution: total uncertainty.
  mutual_information
             entropy of the mean minus mean entropy of the samples: the part of
             the uncertainty that comes from the samples DISAGREEING (epistemic,
             "the model doesn't know"), as opposed to each sample being unsure
             (aleatoric, "the input is ambiguous"). Only defined for methods that
             produce several predictions: MC Dropout and Deep Ensembles.
  vacuity    K / sum(alpha) for a Dirichlet with concentration alpha: the
             Evidential Deep Learning "I have no evidence" score, 1 when the
             network produced no evidence for any class.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def msp(probs: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(probs).max(axis=-1)


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def mutual_information(samples: np.ndarray) -> np.ndarray:
    """samples: (n_samples, n_inputs, n_classes) probability vectors."""
    samples = np.asarray(samples, dtype=np.float64)
    return entropy(samples.mean(axis=0)) - entropy(samples).mean(axis=0)


def dirichlet_mean(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float64)
    return alpha / alpha.sum(axis=-1, keepdims=True)


def vacuity(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=np.float64)
    return alpha.shape[-1] / alpha.sum(axis=-1)


def scores_from_samples(samples: np.ndarray) -> dict:
    """MC Dropout / ensemble: mean prediction and all three scores."""
    mean = np.asarray(samples).mean(axis=0)
    return {"probs": mean, "msp": msp(mean), "entropy": entropy(mean),
            "mutual_information": mutual_information(samples)}


def scores_from_probs(probs: np.ndarray) -> dict:
    return {"probs": probs, "msp": msp(probs), "entropy": entropy(probs)}


def scores_from_alpha(alpha: np.ndarray) -> dict:
    mean = dirichlet_mean(alpha)
    return {"probs": mean, "msp": msp(mean), "entropy": entropy(mean),
            "vacuity": vacuity(alpha)}


# The score each method is judged by for OOD detection in the headline table
# (the others are reported too). Chosen as each method's own notion of
# uncertainty: mutual information and vacuity measure what the model has not
# seen (epistemic uncertainty), which is what flags an unfamiliar input.
PRIMARY_SCORE = {"softmax": "msp", "mcdropout": "mutual_information",
                 "ensemble": "mutual_information", "edl": "vacuity"}

# The score for selective prediction (which predictions to trust), for every
# method: 1 - the confidence of the (averaged) prediction. A wrong answer on a
# familiar input is total, mostly aleatoric uncertainty, which mutual information
# and vacuity largely ignore; and one score for all methods compares like with like.
SELECTIVE_SCORE = "msp"
