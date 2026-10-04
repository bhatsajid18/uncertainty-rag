"""Tests for the uncertainty benchmark: metrics, scores, the EDL loss, models,
data, and the train -> predict -> evaluate pipeline on tiny fake data.

Runs on CPU in a few seconds; nothing is downloaded.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uncertainty import data as D  # noqa: E402
from uncertainty import evaluate as E  # noqa: E402
from uncertainty import metrics as M  # noqa: E402
from uncertainty import predict as P  # noqa: E402
from uncertainty import scores as S  # noqa: E402
from uncertainty import train as T  # noqa: E402
from uncertainty.losses import edl_loss, kl_to_uniform_dirichlet  # noqa: E402
from uncertainty.models import build_model, enable_mc_dropout  # noqa: E402

rng = np.random.default_rng(0)


# --- metrics -------------------------------------------------------------------------


def test_auroc_and_ap_match_sklearn_with_ties():
    sk = pytest.importorskip("sklearn.metrics")
    id_s = np.round(rng.normal(0, 1, 300), 1)       # rounding makes ties
    ood_s = np.round(rng.normal(0.8, 1, 200), 1)
    y = np.r_[np.zeros(300), np.ones(200)]
    s = np.r_[id_s, ood_s]
    assert M.auroc(id_s, ood_s) == pytest.approx(sk.roc_auc_score(y, s))
    assert M.aupr_out(id_s, ood_s) == pytest.approx(sk.average_precision_score(y, s))
    assert M.aupr_in(id_s, ood_s) == pytest.approx(
        sk.average_precision_score(1 - y, -s))


def test_auroc_extremes():
    assert M.auroc([0, 1, 2], [5, 6]) == 1.0
    assert M.auroc([5, 6], [0, 1]) == 0.0
    assert M.auroc([1, 1], [1, 1]) == 0.5
    assert np.isnan(M.auroc([], [1]))


def test_fpr95():
    id_s = np.arange(100, dtype=float)
    assert M.fpr_at_tpr(id_s, id_s + 1000) == 0.0, "OOD all above the threshold"
    assert M.fpr_at_tpr(id_s, np.full(10, -1.0)) == 1.0, "OOD all look like ID"
    assert 0.9 <= M.fpr_at_tpr(id_s, id_s) <= 1.0


def test_ece_calibrated_vs_overconfident():
    n = 20000
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.uniform(size=n) < conf              # calibrated by construction
    labels = np.where(correct, 0, 1)
    probs = np.stack([conf, 1 - conf], axis=1)
    assert M.ece(probs, labels) < 0.02
    over = np.stack([np.full(n, 0.99), np.full(n, 0.01)], axis=1)
    half = np.where(rng.uniform(size=n) < 0.5, 0, 1)
    assert M.ece(over, half) == pytest.approx(0.49, abs=0.02)


def test_perfect_predictions_have_zero_loss():
    labels = np.array([0, 1, 2])
    probs = np.eye(3)[labels]
    assert M.brier(probs, labels) == 0.0
    assert M.nll(probs, labels) == pytest.approx(0.0, abs=1e-9)
    assert M.accuracy(probs, labels) == 1.0


def test_selective_prediction_rewards_refusing_mistakes_first():
    correct = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    good = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])   # mistakes most uncertain
    bad = good[::-1].copy()
    assert M.aurc(good, correct) < M.aurc(bad, correct)
    assert M.accuracy_at_coverage(good, correct, 4 / 6) == 1.0
    cov, risk = M.risk_coverage(good, correct)
    assert cov[-1] == 1.0 and risk[-1] == pytest.approx(2 / 6)


# --- scores --------------------------------------------------------------------------


def test_mutual_information_is_disagreement():
    same = np.tile(np.array([[[0.7, 0.2, 0.1]]]), (5, 1, 1))
    assert S.mutual_information(same)[0] == pytest.approx(0.0, abs=1e-9)
    split = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
    assert S.mutual_information(split)[0] == pytest.approx(np.log(2), abs=1e-6)


def test_vacuity_and_entropy_ranges():
    assert S.vacuity(np.ones((1, 10)))[0] == 1.0, "no evidence = fully vacuous"
    assert S.vacuity(np.array([[101.0] + [1.0] * 9]))[0] < 0.1
    assert S.entropy(np.full((1, 4), 0.25))[0] == pytest.approx(np.log(4))
    r = S.scores_from_alpha(np.array([[5.0, 1.0, 1.0]]))
    assert r["probs"].sum() == pytest.approx(1.0) and set(r) >= {"vacuity", "msp"}


# --- EDL loss ------------------------------------------------------------------------


def test_kl_is_zero_at_the_uniform_dirichlet():
    assert kl_to_uniform_dirichlet(torch.ones(2, 10)).abs().max() < 1e-5
    assert (kl_to_uniform_dirichlet(torch.tensor([[5.0, 1.0, 1.0]])) > 0).all()


@pytest.mark.parametrize("variant", ["mse", "log", "digamma"])
def test_edl_loss_prefers_evidence_for_the_true_class(variant):
    y = torch.tensor([0, 1])
    right = torch.tensor([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
    wrong = torch.tensor([[0.0, 20.0, 0.0], [20.0, 0.0, 0.0]])
    l_right = edl_loss(right, y, epoch=10, anneal_epochs=10, variant=variant)
    l_wrong = edl_loss(wrong, y, epoch=10, anneal_epochs=10, variant=variant)
    assert torch.isfinite(l_right) and l_right < l_wrong


def test_kl_term_is_annealed_in():
    y = torch.tensor([0])
    ev = torch.tensor([[1.0, 5.0, 5.0]])               # evidence for wrong classes
    assert edl_loss(ev, y, epoch=0) < edl_loss(ev, y, epoch=10)
    with pytest.raises(ValueError):
        edl_loss(ev, y, epoch=0, variant="nope")


# --- models --------------------------------------------------------------------------


def test_models_share_a_backbone_and_differ_where_they_should():
    x = torch.randn(2, 3, 32, 32)
    for method in ("softmax", "mcdropout", "edl"):
        m = build_model(method, width=8).eval()
        assert m(x).shape == (2, 10)
    def n_drop(model):
        return sum(isinstance(layer, torch.nn.Dropout) for layer in model.modules())
    assert n_drop(build_model("softmax", width=8)) == 0
    assert n_drop(build_model("mcdropout", width=8)) == 9   # 8 blocks + classifier
    ev = build_model("edl", width=8).evidence(torch.tensor([[-5.0, 0.0, 5.0]]))
    assert (ev >= 0).all()


def test_mc_dropout_keeps_batchnorm_frozen():
    m = build_model("mcdropout", width=8)
    before = m.bn1.running_mean.clone()
    assert enable_mc_dropout(m) == 9
    with torch.no_grad():
        a, b = m(torch.randn(4, 3, 32, 32)), None
        b = m(torch.randn(4, 3, 32, 32))
    assert torch.equal(before, m.bn1.running_mean), "BN statistics untouched"
    assert not m.bn1.training and a.shape == b.shape


# --- data ----------------------------------------------------------------------------


def test_corruptions_stay_in_range_and_are_deterministic():
    x = torch.rand(3, 32, 32)
    for kind in D.CORRUPTIONS:
        for sev in (1, 5):
            out = D.corrupt(x, kind, sev, torch.Generator().manual_seed(1))
            assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    base = D._fake(8, seed=0, transform=None)
    ds = D.CorruptedDataset(base, "gaussian_noise", 3)
    assert torch.equal(ds[2][0], ds[2][0]), "same index, same noise"
    with pytest.raises(ValueError):
        D.corrupt(x, "fog", 1)


def test_fake_eval_sets():
    cfg = D.DataConfig(fake=True, fake_size=32, num_workers=0)
    sets = D.eval_datasets(cfg, corruption_subset=8)
    assert {"cifar10", "svhn", "cifar100"} <= set(sets)
    assert len([n for n in sets if n.startswith("cifar10-c/")]) == 25
    assert sets["svhn"][0][1] == -1, "OOD labels are meaningless to the model"


# --- the pipeline, end to end ----------------------------------------------------------


def _cfg(tmp_path):
    return {"data": {"root": str(tmp_path), "batch_size": 16, "num_workers": 0},
            "model": {"width": 8, "dropout": 0.1},
            "train": {"epochs": 1, "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4,
                      "amp": False, "ensemble_size": 2, "edl_loss": "digamma",
                      "edl_anneal_epochs": 1},
            "predict": {"mc_samples": 2, "corruption_subset": 8},
            "paths": {"checkpoints": str(tmp_path / "ck"),
                      "results": str(tmp_path / "res")}}


def test_train_predict_evaluate_on_fake_data(tmp_path):
    cfg = _cfg(tmp_path)
    dcfg = D.DataConfig(fake=True, fake_size=32, batch_size=16, num_workers=0)
    dev = torch.device("cpu")
    for method, seed in [("softmax", 0), ("softmax", 1), ("mcdropout", 0), ("edl", 0)]:
        path = T.checkpoint_path(tmp_path / "ck", method, seed)
        meta = T.train_one(method, seed, cfg, dcfg, dev, path, log=lambda *_: None)
        assert path.exists() and path.with_suffix(".json").exists()
        assert meta["history"][-1]["test_acc"] is not None

    sets = D.eval_datasets(dcfg, corruption_subset=8)
    keep = ["cifar10", "svhn", "cifar100", "cifar10-c/contrast/5"]
    pred_dir = tmp_path / "res" / "predictions"
    pred_dir.mkdir(parents=True)
    for method in P.METHODS:
        paths = P.checkpoints_for(method, tmp_path / "ck", ensemble_size=2)
        models = [P.load_model(p, dev)[0] for p in paths]
        for name in keep:
            res = P.predict_set(method, models, D.eval_loader(sets[name], dcfg), dev, 2)
            assert np.allclose(res["probs"].sum(axis=1), 1.0, atol=1e-5)
            np.savez(pred_dir / f"{method}__{P.set_filename(name)}.npz", **res)

    summary = E.evaluate(pred_dir, tmp_path / "res", fake=True, figures=False)
    assert [r["method"] for r in summary["headline"]] == list(E.METHODS)
    assert {r["score"] for r in summary["headline"]} == {
        "msp", "mutual_information", "vacuity"}
    assert all(0 <= r["svhn_auroc"] <= 1 for r in summary["headline"])
    sev = {(r["method"], r["severity"]) for r in summary["corruption_by_severity"]}
    assert ("edl", 0) in sev and ("edl", 5) in sev
    report = (tmp_path / "res" / "report.md").read_text()
    assert "| Deep Ensemble (5) |" in report and "FAKE data" in report
    assert json.loads((tmp_path / "res" / "summary.json").read_text())["fake"] is True


def test_evaluate_on_synthetic_predictions(tmp_path):
    """Known structure in, known ordering out: OOD inputs made more uncertain
    than ID inputs must score AUROC near 1, and figures get written."""
    pytest.importorskip("matplotlib")
    pred = tmp_path / "predictions"
    pred.mkdir()
    n = 400
    labels = rng.integers(0, 10, n)
    probs = np.full((n, 10), 0.01)
    probs[np.arange(n), labels] = 0.91
    for method in ("softmax", "edl"):
        score = S.PRIMARY_SCORE[method]
        base = {"probs": probs, "labels": labels, "msp": rng.uniform(0, 0.3, n),
                "entropy": rng.uniform(0, 0.3, n)}
        if score == "vacuity":
            base["vacuity"] = rng.uniform(0, 0.3, n)
        np.savez(pred / f"{method}__cifar10.npz", **base)
        ood = dict(base, labels=np.full(n, -1))
        for k in ("msp", "entropy", "vacuity"):
            if k in ood:
                ood[k] = rng.uniform(0.5, 1.0, n)
        np.savez(pred / f"{method}__svhn.npz", **ood)
    summary = E.evaluate(pred, tmp_path, figures=True)
    for r in summary["headline"]:
        assert r["svhn_auroc"] == 1.0 and r["svhn_fpr95"] == 0.0
        assert r["accuracy"] == 1.0
    assert "roc_svhn.png" in summary["figures"]
    assert (tmp_path / "figures" / "reliability.png").exists()


def test_selective_prediction_ranks_every_method_by_confidence():
    """OOD detection uses MI for the ensemble; selective prediction must not.
    Here MI is uninformative about mistakes while MSP flags them exactly."""
    n = 200
    labels = rng.integers(0, 10, n)
    wrong = np.arange(n) < 40
    preds = np.where(wrong, (labels + 1) % 10, labels)
    probs = np.full((n, 10), 0.02)
    probs[np.arange(n), preds] = 0.82
    clean = {"probs": probs, "labels": labels,
             "msp": np.where(wrong, 0.9, 0.1),          # confident only when right
             "entropy": np.where(wrong, 0.9, 0.1),
             "mutual_information": rng.uniform(0, 1, n)}  # says nothing about mistakes
    row = E.selective_row("ensemble", {"cifar10": clean})
    assert row["score"] == S.SELECTIVE_SCORE == "msp"
    assert row["acc_at_80"] == 1.0, "the 20% refused are exactly the 40 mistakes"
    assert E.headline_row("ensemble", {"cifar10": clean})["aurc"] == row["aurc"]


def test_all_jobs_reuse_softmax_seeds_for_the_ensemble():
    jobs = T.all_jobs(5)
    assert jobs.count(("mcdropout", 0)) == 1 and jobs.count(("edl", 0)) == 1
    assert [s for m, s in jobs if m == "softmax"] == [0, 1, 2, 3, 4]
    assert T.parse_jobs("softmax:2,edl:0") == [("softmax", 2), ("edl", 0)]
    with pytest.raises(ValueError):
        T.parse_jobs("ensemble:0")
