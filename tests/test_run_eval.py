"""Tests for the evaluation sweep: statistics helpers and an offline end-to-end run."""

import csv
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "evaluation"))
for name in ("sentence_transformers", "faiss"):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.SentenceTransformer = object
            sys.modules[name] = stub

import retrieve  # noqa: E402
import run_eval  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

# --- helpers -------------------------------------------------------------------


def test_auroc():
    assert run_eval.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert run_eval.auroc([0.1], [0.9]) == 0.0
    assert run_eval.auroc([0.5], [0.5]) == 0.5
    assert run_eval.auroc([0.9, 0.3], [0.5]) == 0.5
    assert run_eval.auroc([], [0.5]) is None


def test_best_threshold_perfect_separation():
    b = run_eval.best_threshold([0.9, 0.8, 0.1, 0.05], [True, True, False, False])
    assert b["abstention_accuracy"] == 1.0
    assert 0.1 < b["threshold"] <= 0.8


def test_paired_bootstrap_detects_consistent_gain():
    a = [0.0] * 20 + [1.0] * 20
    b = [1.0] * 40            # b wins on 20 queries, ties on the rest
    ci = run_eval.paired_bootstrap(a, b)
    assert ci["delta"] == pytest.approx(0.5)
    assert ci["significant"] and ci["ci_low"] > 0


def test_paired_bootstrap_no_difference():
    a = [0.0, 1.0] * 10
    ci = run_eval.paired_bootstrap(a, list(a))
    assert ci["delta"] == 0 and not ci["significant"]


def test_validate_labels_rejects_changed_text():
    chunks = {"c0": {"text": "hello"}}
    ok = [{"qid": "q1", "relevant": [{"chunk_id": "c0",
                                     "text_hash": run_eval.text_hash("hello")}]}]
    run_eval.validate_labels(ok, chunks)
    bad = [{"qid": "q1", "relevant": [{"chunk_id": "c0", "text_hash": "nope"}]}]
    with pytest.raises(SystemExit):
        run_eval.validate_labels(bad, chunks)


# --- end to end with a stub retriever ----------------------------------------------

TEXTS = [
    ("pA", "evidential deep learning places a dirichlet over class probabilities"),
    ("pA", "the evidence network outputs concentration parameters"),
    ("pB", "outlier exposure trains on auxiliary outliers such as tiny images"),
    ("pB", "false positive rate at 95 percent true positive rate on svhn"),
    ("pC", "deep ensembles average predictions from independently trained networks"),
    ("pC", "ensemble distillation compresses an ensemble into a single model"),
]


class StubRetriever(retrieve.HybridRetriever):
    """Real ranking/fusion/cap code; embedding and reranker replaced by BM25."""

    def __init__(self, data_dir, load_reranker=False, **_):
        self.meta = [json.loads(line) for line in
                     (data_dir / "chunks.jsonl").read_text().splitlines()]
        self.bm25 = BM25Okapi([retrieve.tokenize_for_bm25(m["text"]) for m in self.meta])

    def _dense_rank(self, q, n):
        return self._bm25_rank(q, n)

    def _load_reranker(self):
        class R:
            def predict(self, pairs):
                return [float(len(set(retrieve.tokenize_for_bm25(q)) &
                                  set(retrieve.tokenize_for_bm25(t))))
                        for q, t in pairs]
        return R()


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "eval").mkdir(parents=True)
    chunks = [{"chunk_id": f"c{i}", "arxiv_id": p, "title": p, "year": "2020",
               "page_start": 1, "page_end": 1, "is_reference": False, "text": t}
              for i, (p, t) in enumerate(TEXTS)]
    (data / "chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks))

    def q(qid, cat, text, ids):
        return {"qid": qid, "category": cat, "question": text, "source": "manual",
                "relevant": [{"chunk_id": i, "text_hash":
                              run_eval.text_hash(chunks[int(i[1:])]["text"])}
                             for i in ids]}
    queries = [
        q("q1", "semantic", "how does evidential learning use a dirichlet", ["c0", "c1"]),
        q("q2", "exact_term", "false positive rate svhn", ["c3"]),
        q("q3", "multi_paper", "ensembles versus ensemble distillation", ["c4", "c5"]),
        q("q4", "unanswerable", "what learning rate does ppo use", []),
    ]
    (data / "eval" / "queries.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in queries))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(retrieve, "HybridRetriever", StubRetriever)
    return tmp_path


def test_end_to_end_writes_all_outputs(workdir, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--suite", "no_llm",
                                      "--run-name", "t"])
    run_eval.main()
    out = workdir / "results" / "eval" / "t"
    for f in ("run_info.json", "per_query.jsonl", "summary.csv",
              "summary_by_category.csv", "significance.csv", "abstention.csv"):
        assert (out / f).exists(), f

    summary = {r["config"]: r for r in csv.DictReader((out / "summary.csv").open())}
    assert set(summary) == {"dense", "bm25", "hybrid", "hybrid_rerank"}
    assert all(r["n_queries"] == "3" for r in summary.values()), "unanswerable excluded"
    for r in summary.values():
        for k in ("recall@5", "mrr", "ndcg@5"):
            assert 0.0 <= float(r[k]) <= 1.0

    per_query = [json.loads(line) for line in (out / "per_query.jsonl").open()]
    assert len(per_query) == 4 * 4
    assert all(len(r["ranked_ids"]) <= 10 for r in per_query)

    report = capsys.readouterr().out
    assert "Recall@5 by category" in report and "AUROC" in report


def test_multi_query_config_uses_variants(workdir):
    r = StubRetriever(workdir / "data")
    queries = [json.loads(line) for line in
               (workdir / "data/eval/queries.jsonl").read_text().splitlines()]
    q2 = [q for q in queries if q["qid"] == "q2"]
    q2[0]["question"] = "fpr95"   # abbreviation: shares no token with c3...
    plain = run_eval.run_config(r, q2, run_eval.CONFIGS["bm25"])
    mq = run_eval.run_config(r, q2, run_eval.CONFIGS["bm25+mq3"],
                             {"q2": ["fpr95",
                                     "false positive rate at 95 percent"]})
    assert plain[0]["metrics"]["recall@5"] == 0.0
    assert mq[0]["metrics"]["recall@1"] == 1.0  # ...which the rephrasing recovers


# --- regressions from the first real run -------------------------------------------


def test_zero_answerable_queries_exits_cleanly(workdir, monkeypatch):
    (workdir / "data/eval/queries.jsonl").write_text("")
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--suite", "no_llm"])
    with pytest.raises(SystemExit) as e:
        run_eval.main()
    assert "nothing to score" in str(e.value)


def test_missing_queries_file_exits_cleanly(workdir, monkeypatch):
    (workdir / "data/eval/queries.jsonl").unlink()
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--suite", "no_llm"])
    with pytest.raises(SystemExit) as e:
        run_eval.main()
    assert "not found" in str(e.value)


def test_report_survives_missing_metrics(capsys):
    run_eval.print_report([{"config": "dense", "n_queries": 0},
                           {"config": "bm25", "n_queries": 0}], [], [], [])
    assert "dense" in capsys.readouterr().out


def test_export_refuses_when_nothing_kept(workdir, monkeypatch):
    import review_queries
    cand = workdir / "data/eval/candidates.jsonl"
    cand.write_text(json.dumps({"qid": "q1", "category": "semantic", "question": "x",
                                "relevant": [], "evidence": [], "source": "llm",
                                "status": "pending"}) + "\n")
    out = workdir / "data/eval/queries.jsonl"
    before = out.read_text()
    monkeypatch.setattr(sys, "argv", ["review_queries.py", "export"])
    with pytest.raises(SystemExit):
        review_queries.main()
    assert out.read_text() == before, "existing queries.jsonl must not be overwritten"


# --- a bootstrap CI needs enough queries to mean anything -------------------------


def test_bootstrap_withholds_the_interval_on_too_few_queries():
    """At n=1 every resample draws the same difference, so the CI collapses to
    a point and any non-zero delta reads as significant. One real run reported
    'CI [+1.000, +1.000], significant=True' off a single query."""
    r = run_eval.paired_bootstrap([0.0], [1.0])
    assert r["delta"] == 1.0
    assert r["ci_low"] is None and r["ci_high"] is None
    assert r["significant"] is None
    assert "confidence interval" in r["note"]


def test_bootstrap_still_reports_an_interval_with_enough_queries():
    a = [0.0] * 10
    b = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    r = run_eval.paired_bootstrap(a, b)
    assert r["ci_low"] is not None and r["ci_high"] is not None
    assert r["significant"] in (True, False)
