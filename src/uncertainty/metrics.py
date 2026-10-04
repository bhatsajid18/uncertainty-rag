"""
Metrics for uncertainty estimation and out-of-distribution (OOD) detection.

Plain numpy, no scikit-learn at runtime (the tests cross-check against it), so
the definitions are visible and the conventions explicit:

OOD detection - "positive" means OUT-of-distribution, and a higher uncertainty
score should mean more likely OOD.
  auroc           probability that a random OOD input gets a higher uncertainty
                  than a random in-distribution (ID) input (ties count half).
                  0.5 = chance, 1.0 = perfect separation.
  aupr_out        average precision with OOD as the positive class.
  aupr_in         average precision with ID as the positive class (scores
                  negated). Both are reported because AUPR depends on which
                  class is rarer.
  fpr_at_95_tpr   the threshold that keeps 95% of ID inputs ("true positive
                  rate" on ID = 95%), and the share of OOD inputs that still
                  pass it as ID. Lower is better. This is the convention of
                  Hendrycks & Gimpel (2017) and most OOD papers since.

Selective prediction - answer only the inputs the model is most sure about:
  risk_coverage   sort inputs from least to most uncertain; at each coverage
                  (the share answered) the risk is the error rate among the
                  answered ones.
  aurc            area under that curve (mean risk over all coverages). Lower
                  is better: a good uncertainty score refuses its mistakes first.

Calibration - on in-distribution test data:
  ece             expected calibration error: |accuracy - confidence| averaged
                  over 15 equal-width confidence bins, weighted by bin size.
  brier           mean squared error between the probability vector and the
                  one-hot label, summed over classes.
  nll             mean negative log-likelihood of the true class.
"""

from __future__ import annotations

import numpy as np


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (scipy's rankdata)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(id_scores, ood_scores) -> float:
    """Area under the ROC curve for separating OOD (positive) from ID.

    Computed from the Mann-Whitney U statistic, which is exact and handles ties.
    """
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0:
        return float("nan")
    ranks = _average_ranks(np.concatenate([ood_scores, id_scores]))
    u = ranks[:n_ood].sum() - n_ood * (n_ood + 1) / 2
    return float(u / (n_ood * n_id))


def average_precision(scores, labels) -> float:
    """Average precision (area under precision-recall), as scikit-learn defines
    it: sum over thresholds of (recall step) x precision."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    # evaluate only at the last position of each distinct score (tie groups)
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def aupr_out(id_scores, ood_scores) -> float:
    s = np.concatenate([np.asarray(ood_scores), np.asarray(id_scores)])
    y = np.r_[np.ones(len(ood_scores)), np.zeros(len(id_scores))]
    return average_precision(s, y)


def aupr_in(id_scores, ood_scores) -> float:
    s = -np.concatenate([np.asarray(id_scores), np.asarray(ood_scores)])
    y = np.r_[np.ones(len(id_scores)), np.zeros(len(ood_scores))]
    return average_precision(s, y)


def fpr_at_tpr(id_scores, ood_scores, tpr: float = 0.95) -> float:
    """Share of OOD inputs accepted as ID when the threshold keeps `tpr` of ID.

    An input is "accepted as ID" when its uncertainty is at or below the
    threshold; the threshold is the tpr-quantile of the ID uncertainties.
    """
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return float("nan")
    threshold = np.quantile(id_scores, tpr, method="higher")
    return float(np.mean(ood_scores <= threshold))


def ece(probs, labels, n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(total)


def reliability(probs, labels, n_bins: int = 15) -> dict:
    """Per-bin confidence and accuracy, for reliability diagrams."""
    probs = np.asarray(probs, dtype=np.float64)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == np.asarray(labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out = {"confidence": [], "accuracy": [], "count": [], "edges": edges.tolist()}
    for b in range(n_bins):
        m = bins == b
        out["count"].append(int(m.sum()))
        out["confidence"].append(float(conf[m].mean()) if m.any() else None)
        out["accuracy"].append(float(correct[m].mean()) if m.any() else None)
    return out


def brier(probs, labels) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    onehot = np.eye(probs.shape[1])[np.asarray(labels)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def nll(probs, labels, eps: float = 1e-12) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    p_true = probs[np.arange(len(probs)), np.asarray(labels)]
    return float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))


def accuracy(probs, labels) -> float:
    return float(np.mean(np.asarray(probs).argmax(axis=1) == np.asarray(labels)))


def roc_curve(id_scores, ood_scores, points: int = 200) -> dict:
    """(FPR, TPR) pairs for plotting, OOD as positive, at evenly spaced
    thresholds over the score range."""
    s = np.concatenate([np.asarray(id_scores), np.asarray(ood_scores)])
    thresholds = np.quantile(s, np.linspace(0, 1, points))[::-1]
    id_scores, ood_scores = np.asarray(id_scores), np.asarray(ood_scores)
    tpr = [float(np.mean(ood_scores >= t)) for t in thresholds]
    fpr = [float(np.mean(id_scores >= t)) for t in thresholds]
    return {"fpr": [0.0, *fpr, 1.0], "tpr": [0.0, *tpr, 1.0]}


def risk_coverage(uncertainty, correct) -> tuple[np.ndarray, np.ndarray]:
    """(coverage, risk) when inputs are answered most-certain first."""
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if len(uncertainty) == 0:
        return np.array([]), np.array([])
    order = np.argsort(uncertainty, kind="mergesort")
    k = np.arange(1, len(order) + 1)
    risk = np.cumsum(~correct[order]) / k
    return k / len(order), risk


def aurc(uncertainty, correct) -> float:
    _, risk = risk_coverage(uncertainty, correct)
    return float(risk.mean()) if len(risk) else float("nan")


def accuracy_at_coverage(uncertainty, correct, coverage: float) -> float:
    """Accuracy on the `coverage` share of inputs the model is most sure about."""
    cov, risk = risk_coverage(uncertainty, correct)
    if not len(cov):
        return float("nan")
    i = min(len(cov) - 1, max(0, int(np.ceil(coverage * len(cov))) - 1))
    return float(1.0 - risk[i])


def ood_metrics(id_scores, ood_scores) -> dict:
    return {"auroc": auroc(id_scores, ood_scores),
            "aupr_in": aupr_in(id_scores, ood_scores),
            "aupr_out": aupr_out(id_scores, ood_scores),
            "fpr95": fpr_at_tpr(id_scores, ood_scores, 0.95)}


def calibration_metrics(probs, labels) -> dict:
    return {"accuracy": accuracy(probs, labels), "nll": nll(probs, labels),
            "ece": ece(probs, labels), "brier": brier(probs, labels)}
