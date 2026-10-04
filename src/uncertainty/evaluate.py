"""
Turn saved predictions into the benchmark's tables and figures. CPU only.

Reads the .npz files predict.py wrote and computes, for every method:

  headline     CIFAR-10 accuracy and calibration (NLL, ECE, Brier), and OOD
               detection (AUROC, AUPR-In, AUPR-Out, FPR@95TPR) against SVHN
               (far OOD) and CIFAR-100 (near OOD), each method scored by its
               own notion of uncertainty (scores.PRIMARY_SCORE).
  all scores   the same OOD metrics for every score a method produces (MSP,
               entropy, mutual information, vacuity), so a method is not
               judged only by one choice of score.
  near vs far  how much harder near OOD is than far OOD for each method.
  corruption   accuracy, ECE and mean uncertainty on CIFAR-10 under five
               synthetic corruptions at five severities, plus shift AUROC:
               how well the uncertainty separates clean from corrupted inputs.
               A useful uncertainty rises as accuracy falls.
  selective    risk-coverage on CIFAR-10: answer only the most confident inputs
               (every method ranked by scores.SELECTIVE_SCORE). AURC and
               accuracy at 80% / 90% coverage.

Outputs, in results/uncertainty/ (results/uncertainty/fake/ with --fake):
  summary.json         everything below in one file (the dashboard reads it)
  headline.csv         one row per method
  ood_all_scores.csv   method x score x OOD set
  corruption.csv       method x corruption x severity
  selective.csv        method: AURC, accuracy at 80% and 90% coverage
  report.md            the tables above, ready to paste into a README
  figures/*.png        ROC curves, corruption curves, reliability diagrams,
                       risk-coverage curves

Usage (from the repo root):
  python src/uncertainty/evaluate.py
  python src/uncertainty/evaluate.py --fake          # after the fake pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# src/ on the path, so these are imported as the `uncertainty` package and
# cannot collide with same-named modules elsewhere (src/evaluation/metrics.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uncertainty.config import load_config  # noqa: E402
from uncertainty.data import CORRUPTIONS  # noqa: E402
from uncertainty.metrics import (  # noqa: E402
    accuracy_at_coverage, aurc, auroc, calibration_metrics, ood_metrics,
    reliability, risk_coverage, roc_curve,
)
from uncertainty.scores import PRIMARY_SCORE, SELECTIVE_SCORE  # noqa: E402

METHODS = ("softmax", "ensemble", "mcdropout", "edl")
LABELS = {"softmax": "Softmax (MSP)", "ensemble": "Deep Ensemble (5)",
          "mcdropout": "MC Dropout", "edl": "Evidential (EDL)"}
OOD_SETS = {"svhn": "far", "cifar100": "near"}
SCORE_KEYS = ("msp", "entropy", "mutual_information", "vacuity")


def load_predictions(pred_dir: Path) -> dict[str, dict[str, dict]]:
    """{method: {set name: {probs, labels, <scores>}}} from predict.py's files."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(pred_dir.glob("*.npz")):
        method, _, set_part = path.stem.partition("__")
        with np.load(path) as z:
            out.setdefault(method, {})[set_part.replace("__", "/")] = {
                k: z[k] for k in z.files}
    return out


def _correct(p: dict) -> np.ndarray:
    return p["probs"].argmax(axis=1) == p["labels"]


def headline_row(method: str, sets: dict) -> dict | None:
    if "cifar10" not in sets:
        return None
    score = PRIMARY_SCORE[method]
    clean = sets["cifar10"]
    row = {"method": method, "label": LABELS.get(method, method), "score": score,
           **calibration_metrics(clean["probs"], clean["labels"])}
    for ood, kind in OOD_SETS.items():
        if ood in sets:
            m = ood_metrics(clean[score], sets[ood][score])
            row.update({f"{ood}_{k}": v for k, v in m.items()})
    row["aurc"] = aurc(clean[SELECTIVE_SCORE], _correct(clean))
    return row


def all_score_rows(method: str, sets: dict) -> list[dict]:
    rows = []
    clean = sets.get("cifar10")
    if clean is None:
        return rows
    for score in SCORE_KEYS:
        if score not in clean:
            continue
        for ood, kind in OOD_SETS.items():
            if ood in sets:
                rows.append({"method": method, "score": score, "ood": ood,
                             "kind": kind, "primary": score == PRIMARY_SCORE[method],
                             **ood_metrics(clean[score], sets[ood][score])})
    return rows


def corruption_rows(method: str, sets: dict) -> list[dict]:
    """Per corruption and severity, plus severity 0 = the clean test set."""
    score = PRIMARY_SCORE[method]
    clean = sets.get("cifar10")
    rows = []
    if clean is not None:
        cal = calibration_metrics(clean["probs"], clean["labels"])
        rows.append({"method": method, "corruption": "none", "severity": 0,
                     "accuracy": cal["accuracy"], "ece": cal["ece"],
                     "uncertainty": float(np.mean(clean[score])), "shift_auroc": 0.5})
    for kind in CORRUPTIONS:
        for sev in range(1, 6):
            p = sets.get(f"cifar10-c/{kind}/{sev}")
            if p is None:
                continue
            cal = calibration_metrics(p["probs"], p["labels"])
            rows.append({
                "method": method, "corruption": kind, "severity": sev,
                "accuracy": cal["accuracy"], "ece": cal["ece"],
                "uncertainty": float(np.mean(p[score])),
                # do corrupted inputs look more uncertain than clean ones?
                "shift_auroc": (auroc(clean[score], p[score])
                                if clean is not None else float("nan")),
            })
    return rows


def by_severity(rows: list[dict]) -> list[dict]:
    """Corruption rows averaged over corruption kinds, per method and severity."""
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        for sev in sorted({r["severity"] for r in rows if r["method"] == method}):
            group = [r for r in rows if r["method"] == method and r["severity"] == sev]
            out.append({"method": method, "severity": sev, **{
                k: float(np.mean([r[k] for r in group]))
                for k in ("accuracy", "ece", "uncertainty", "shift_auroc")}})
    return out


def selective_row(method: str, sets: dict) -> dict | None:
    clean = sets.get("cifar10")
    if clean is None:
        return None
    unc, correct = clean[SELECTIVE_SCORE], _correct(clean)
    return {"method": method, "score": SELECTIVE_SCORE, "aurc": aurc(unc, correct),
            "acc_at_80": accuracy_at_coverage(unc, correct, 0.8),
            "acc_at_90": accuracy_at_coverage(unc, correct, 0.9),
            "acc_at_100": float(correct.mean())}


def near_far_rows(headline: list[dict]) -> list[dict]:
    rows = []
    for r in headline:
        if "svhn_auroc" in r and "cifar100_auroc" in r:
            rows.append({"method": r["method"], "far_auroc": r["svhn_auroc"],
                         "near_auroc": r["cifar100_auroc"],
                         "gap": r["svhn_auroc"] - r["cifar100_auroc"]})
    return rows


# --- writing ------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v
                        for k, v in r.items()})


def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{100 * v:.1f}" if pct else f"{v:.3f}"


def markdown_report(summary: dict) -> str:
    h = summary["headline"]
    lines = ["# Uncertainty benchmark: CIFAR-10 vs SVHN / CIFAR-100", "",
             f"Generated {summary['generated']}"
             + (" from FAKE data (smoke test, numbers meaningless)"
                if summary.get("fake") else "") + ".", "",
             "## Headline", "",
             "Accuracy, AUROC and FPR95 in %. OOD detection uses each method's own "
             "uncertainty score (the Score column); AURC ranks every method by "
             "confidence (1 - max probability). Lower is better for NLL, ECE, Brier, "
             "FPR95 and AURC.",
             "",
             "| Method | Score | Acc | NLL | ECE | Brier | SVHN AUROC | SVHN FPR95 "
             "| C100 AUROC | C100 FPR95 | AURC |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in h:
        lines.append(
            f"| {r['label']} | {r['score']} | {_fmt(r['accuracy'], True)} "
            f"| {_fmt(r['nll'])} | {_fmt(r['ece'])} | {_fmt(r['brier'])} "
            f"| {_fmt(r.get('svhn_auroc'), True)} | {_fmt(r.get('svhn_fpr95'), True)} "
            f"| {_fmt(r.get('cifar100_auroc'), True)} "
            f"| {_fmt(r.get('cifar100_fpr95'), True)} | {_fmt(r['aurc'])} |")
    if summary["near_far"]:
        lines += ["", "## Near vs far OOD", "",
                  "AUROC (%) against far OOD (SVHN) and near OOD (CIFAR-100).", "",
                  "| Method | Far (SVHN) | Near (CIFAR-100) | Gap |", "|---|---|---|---|"]
        for r in summary["near_far"]:
            lines.append(f"| {LABELS.get(r['method'], r['method'])} "
                         f"| {_fmt(r['far_auroc'], True)} | {_fmt(r['near_auroc'], True)} "
                         f"| {_fmt(r['gap'], True)} |")
    sev = summary["corruption_by_severity"]
    if sev:
        lines += ["", "## Under corruption (mean over 5 corruption types)", "",
                  "Severity 0 is the clean test set. Shift AUROC: how well the "
                  "uncertainty separates corrupted from clean inputs.", "",
                  "| Method | Severity | Accuracy | ECE | Mean uncertainty | Shift AUROC |",
                  "|---|---|---|---|---|---|"]
        for r in sev:
            lines.append(f"| {LABELS.get(r['method'], r['method'])} | {r['severity']} "
                         f"| {_fmt(r['accuracy'], True)} | {_fmt(r['ece'])} "
                         f"| {_fmt(r['uncertainty'])} | {_fmt(r['shift_auroc'], True)} |")
    if summary["selective"]:
        lines += ["", "## Selective prediction (CIFAR-10)", "",
                  "Answer only the most confident inputs (every method ranked by "
                  "1 - max probability of its averaged prediction). Accuracy (%) "
                  "at a given coverage; AURC lower is better.", "",
                  "| Method | AURC | Acc @ 80% | Acc @ 90% | Acc @ 100% |",
                  "|---|---|---|---|---|"]
        for r in summary["selective"]:
            lines.append(f"| {LABELS.get(r['method'], r['method'])} | {_fmt(r['aurc'])} "
                         f"| {_fmt(r['acc_at_80'], True)} | {_fmt(r['acc_at_90'], True)} "
                         f"| {_fmt(r['acc_at_100'], True)} |")
    return "\n".join(lines) + "\n"


def make_figures(preds: dict, summary: dict, fig_dir: Path) -> list[str]:
    """PNG figures; skipped quietly if matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    methods = [m for m in METHODS if m in preds]

    for ood in OOD_SETS:
        if not any("cifar10" in preds[m] and ood in preds[m] for m in methods):
            continue
        fig, ax = plt.subplots(figsize=(4.5, 4))
        for m in methods:
            sets, score = preds[m], PRIMARY_SCORE[m]
            if "cifar10" in sets and ood in sets:
                c = roc_curve(sets["cifar10"][score], sets[ood][score])
                a = auroc(sets["cifar10"][score], sets[ood][score])
                ax.plot(c["fpr"], c["tpr"], label=f"{LABELS[m]} ({100 * a:.1f})")
        ax.plot([0, 1], [0, 1], ls=":", c="grey")
        ax.set(xlabel="False positive rate (ID flagged)",
               ylabel="True positive rate (OOD flagged)",
               title=f"CIFAR-10 vs {'SVHN' if ood == 'svhn' else 'CIFAR-100'}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = fig_dir / f"roc_{ood}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

    sev = summary["corruption_by_severity"]
    if sev:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for key, ax, title in (("accuracy", axes[0], "Accuracy"),
                               ("ece", axes[1], "ECE"),
                               ("shift_auroc", axes[2], "Shift AUROC")):
            for m in methods:
                pts = [r for r in sev if r["method"] == m]
                ax.plot([r["severity"] for r in pts], [r[key] for r in pts],
                        marker="o", label=LABELS[m])
            ax.set(xlabel="Corruption severity", title=title)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        path = fig_dir / "corruption.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path.name)

    fig, axes = plt.subplots(1, len(methods), figsize=(3.2 * len(methods), 3.2),
                             squeeze=False)
    for ax, m in zip(axes[0], methods):
        rel = summary["reliability"].get(m)
        if not rel:
            continue
        centres = [(a + b) / 2 for a, b in zip(rel["edges"][:-1], rel["edges"][1:])]
        acc = [a if a is not None else 0 for a in rel["accuracy"]]
        ax.bar(centres, acc, width=1 / len(centres), edgecolor="black", alpha=0.7)
        ax.plot([0, 1], [0, 1], ls=":", c="grey")
        ax.set(title=LABELS[m], xlabel="Confidence", ylim=(0, 1))
    axes[0][0].set_ylabel("Accuracy")
    fig.tight_layout()
    path = fig_dir / "reliability.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    for m in methods:
        sets = preds[m]
        if "cifar10" in sets:
            cov, risk = risk_coverage(sets["cifar10"][SELECTIVE_SCORE],
                                      _correct(sets["cifar10"]))
            ax.plot(cov, risk, label=LABELS[m])
    ax.set(xlabel="Coverage", ylabel="Risk (error rate)", title="Selective prediction")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = fig_dir / "risk_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path.name)
    return written


def evaluate(pred_dir: Path, out_dir: Path, fake: bool = False,
             figures: bool = True) -> dict:
    preds = load_predictions(pred_dir)
    if not preds:
        raise SystemExit(f"No predictions in {pred_dir}. Run predict.py first.")
    methods = [m for m in METHODS if m in preds]
    headline = [r for r in (headline_row(m, preds[m]) for m in methods) if r]
    corruption = [r for m in methods for r in corruption_rows(m, preds[m])]
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fake": fake, "methods": methods,
        "primary_score": {m: PRIMARY_SCORE[m] for m in methods},
        "selective_score": SELECTIVE_SCORE,
        "headline": headline,
        "ood_all_scores": [r for m in methods for r in all_score_rows(m, preds[m])],
        "near_far": near_far_rows(headline),
        "corruption": corruption,
        "corruption_by_severity": by_severity(corruption),
        "selective": [r for r in (selective_row(m, preds[m]) for m in methods) if r],
        "reliability": {m: reliability(preds[m]["cifar10"]["probs"],
                                       preds[m]["cifar10"]["labels"])
                        for m in methods if "cifar10" in preds[m]},
        "roc": {m: {ood: roc_curve(preds[m]["cifar10"][PRIMARY_SCORE[m]],
                                   preds[m][ood][PRIMARY_SCORE[m]], points=60)
                    for ood in OOD_SETS if ood in preds[m] and "cifar10" in preds[m]}
                for m in methods},
        "n_inputs": {m: {s: int(len(p["labels"])) for s, p in preds[m].items()}
                     for m in methods},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    _write_csv(out_dir / "headline.csv", headline)
    _write_csv(out_dir / "ood_all_scores.csv", summary["ood_all_scores"])
    _write_csv(out_dir / "corruption.csv", corruption)
    _write_csv(out_dir / "selective.csv", summary["selective"])
    (out_dir / "report.md").write_text(markdown_report(summary))
    summary["figures"] = make_figures(preds, summary, out_dir / "figures") if figures else []
    return summary


def main():
    ap = argparse.ArgumentParser(description="Tables and figures from predictions.")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["results"]) / ("fake" if args.fake else "")
    summary = evaluate(root / "predictions", root, fake=args.fake,
                       figures=not args.no_figures)
    print((root / "report.md").read_text())
    print(f"Written to {root}/ (summary.json, *.csv, report.md"
          + (", figures/" if summary["figures"] else "") + ")")


if __name__ == "__main__":
    main()
