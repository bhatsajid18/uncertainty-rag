"""Tests for the Streamlit app: every page renders, with the API client patched.

AppTest runs the script in this process, so patching `ui.client` functions
replaces the HTTP calls with canned responses; no server is needed.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ui import client  # noqa: E402

APP = str(ROOT / "src" / "ui" / "app.py")

TURN = {"question": "What accuracy does it reach?", "standalone":
        "What accuracy does EDL reach on CIFAR5?", "answer": "EDL reaches 83% [S1].",
        "refused": False, "rephrasings": [],
        "sources": [{"n": 1, "chunk_id": "1806.01768__0011", "arxiv_id": "1806.01768",
                     "title": "Evidential Deep Learning", "year": "2018", "pages": "p.7",
                     "score": 0.91, "text": "EDL 99.3 83 Table 1: Test accuracies",
                     "url": "https://arxiv.org/abs/1806.01768",
                     "table_markdown": "| Method | CIFAR5 |\n| --- | --- |\n| EDL | 83 |",
                     "table_source": "geometry", "figures": [], "expanded_with": []}]}
UNC = {"fake": True, "headline": [
    {"method": m, "score": s, "accuracy": 0.9, "nll": 0.3, "ece": 0.02, "brier": 0.15,
     "svhn_auroc": 0.95, "svhn_fpr95": 0.2, "cifar100_auroc": 0.88,
     "cifar100_fpr95": 0.5, "aurc": 0.01}
    for m, s in (("softmax", "msp"), ("edl", "vacuity"))],
    "near_far": [{"method": "softmax", "far_auroc": 0.95, "near_auroc": 0.88, "gap": 0.07}],
    "selective": [{"method": "softmax", "aurc": 0.01, "acc_at_80": 0.98,
                   "acc_at_90": 0.96, "acc_at_100": 0.9}],
    "corruption_by_severity": [
        {"method": "softmax", "severity": s, "accuracy": 0.9 - 0.1 * s, "ece": 0.02 * s,
         "uncertainty": 0.1 * s, "shift_auroc": 0.5 + 0.08 * s} for s in range(6)]}
EVAL = {"run": "core_v3",
        "summary": [{"config": "dense", "recall@1": 0.16, "recall@3": 0.39,
                     "recall@5": 0.51, "recall@10": 0.66, "ndcg@5": 0.36, "mrr": 0.35},
                    {"config": "hybrid_rerank", "recall@1": 0.44, "recall@3": 0.6,
                     "recall@5": 0.68, "recall@10": 0.79, "ndcg@5": 0.58, "mrr": 0.6}],
        "significance": [{"config": "hybrid_rerank", "vs": "dense", "metric": "mrr",
                          "delta": 0.25, "significant": True}],
        "abstention": [{"config": "hybrid_rerank", "abstention_auroc": 0.997}],
        "summary_by_category": [
            {"config": "dense", "category": "semantic", "recall@5": 0.6},
            {"config": "dense", "category": "exact_term", "recall@5": 0.4}]}


@pytest.fixture(autouse=True)
def fresh_cache():
    """The app caches API reads for up to a minute; tests must not share them."""
    import streamlit as st

    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def api(monkeypatch):
    state = {"session": {"session_id": "s1", "scope": "corpus", "documents": [],
                         "with_corpus": True, "turns": []}, "asked": []}

    def ask(sid, q):
        state["asked"].append(q)
        return dict(TURN, question=q)

    monkeypatch.setattr(client, "health", lambda: {
        "status": "ok", "retriever_loaded": True, "papers": 12, "corpus_chunks": 408})
    monkeypatch.setattr(client, "new_session", lambda mode=None: dict(state["session"]))
    monkeypatch.setattr(client, "ask", ask)
    monkeypatch.setattr(client, "uncertainty", lambda fake=False: UNC)
    monkeypatch.setattr(client, "uncertainty_figure", lambda name, fake=False: (
        _ for _ in ()).throw(client.ApiError("no figure", 404)))
    monkeypatch.setattr(client, "eval_runs", lambda kind: ["core_v3"]
                        if kind == "retrieval" else [])
    monkeypatch.setattr(client, "eval_run", lambda kind, run: EVAL)
    return state


def _go(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    return at


def test_chat_page_asks_and_shows_sources(api):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Ask the papers"
    assert "12 papers" in at.sidebar.success[0].value
    at.chat_input[0].set_value("What accuracy does it reach?").run()
    assert not at.exception, at.exception
    assert api["asked"] == ["What accuracy does it reach?"]
    text = " ".join(m.value for m in at.markdown)
    assert "EDL reaches 83%" in text and "| EDL | 83 |" in text
    assert any("Searched for" in c.value for c in at.caption)


def test_uncertainty_dashboard_renders(api):
    at = _go(AppTest.from_file(APP, default_timeout=30).run(), "Uncertainty benchmark")
    assert at.title[0].value == "Uncertainty benchmark"
    assert any("smoke test" in w.value for w in at.warning), "fake results are flagged"
    assert len(at.dataframe) >= 2


def test_retrieval_dashboard_renders(api):
    at = _go(AppTest.from_file(APP, default_timeout=30).run(), "Retrieval evaluation")
    assert at.selectbox[0].value == "core_v3"
    assert len(at.dataframe) >= 3


def test_api_down_is_reported_not_crashed(monkeypatch):
    def down(*a, **k):
        raise client.ApiError("Cannot reach the API at http://localhost:8000")
    for name in ("health", "new_session", "eval_runs", "uncertainty"):
        monkeypatch.setattr(client, name, down)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, at.exception
    assert "Cannot reach the API" in at.sidebar.error[0].value
    assert any("Could not start" in e.value for e in at.error)
    for page in ("Uncertainty benchmark", "Retrieval evaluation", "About"):
        _go(at, page)


def test_missing_llm_key_is_flagged(monkeypatch, api):
    monkeypatch.setattr(client, "health", lambda: {
        "status": "ok", "retriever_loaded": False, "papers": 0, "corpus_chunks": 0,
        "provider": "groq", "llm_ready": False})
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, at.exception
    assert any("GROQ_API_KEY" in w.value for w in at.sidebar.warning)


def test_missing_results_explain_how_to_get_them(monkeypatch, api):
    monkeypatch.setattr(client, "uncertainty", lambda fake=False: (
        _ for _ in ()).throw(client.ApiError("No uncertainty results yet", 404)))
    at = _go(AppTest.from_file(APP, default_timeout=30).run(), "Uncertainty benchmark")
    assert any("kaggle_uncertainty.ipynb" in i.value for i in at.info)


def test_expired_conversation_restarts_instead_of_crashing(monkeypatch, api):
    def gone(sid, q):
        raise client.ApiError("Unknown or expired session. Start a new one.", 404)

    monkeypatch.setattr(client, "ask", gone)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("What accuracy does it reach?").run()
    assert not at.exception, at.exception
    assert any("new one was started" in w.value for w in at.warning)
    assert "session" not in at.session_state


def test_unreachable_run_is_an_error_message(monkeypatch, api):
    def down(kind, run):
        raise client.ApiError("Cannot reach the API")

    monkeypatch.setattr(client, "eval_run", down)
    at = _go(AppTest.from_file(APP, default_timeout=30).run(), "Retrieval evaluation")
    assert any("Cannot reach the API" in e.value for e in at.error)


def test_incomplete_generation_run_is_flagged(monkeypatch, api):
    gen = {"run": "gen_v1", "summary": [{"config": "dense", "faithfulness": 0.9}],
           "run_info": {"complete": False, "pairs_done": 100, "pairs_total": 174,
                        "n_compared": 41}}
    monkeypatch.setattr(client, "eval_runs", lambda kind: ["core_v3"] if kind ==
                        "retrieval" else ["gen_v1"])
    monkeypatch.setattr(client, "eval_run", lambda kind, run: EVAL if kind ==
                        "retrieval" else gen)
    at = _go(AppTest.from_file(APP, default_timeout=30).run(), "Retrieval evaluation")
    assert any("Incomplete run" in w.value and "41" in w.value for w in at.warning)
