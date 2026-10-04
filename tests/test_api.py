"""Tests for the web API: every endpoint, sessions, uploads and error mapping.

The retriever is an in-memory index with a hashed bag-of-words embedder, the
LLM is a scripted function and the tokenizer splits on whitespace, so this runs
offline in a few seconds. The upload test sends a real PDF made with PyMuPDF.
"""

import json
import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_chat  # noqa: E402,F401  (installs module stubs)
from test_chat import FakeEmbedder, chunk, fake_tokenizer  # noqa: E402

import llm  # noqa: E402
from api.main import create_app  # noqa: E402
from api.service import RagService, Settings  # noqa: E402
from chat import REWRITE_SYSTEM  # noqa: E402
from generate import REFUSAL  # noqa: E402
from retrieve import HybridRetriever  # noqa: E402

pymupdf = pytest.importorskip("pymupdf")

CORPUS = [
    chunk("1806.01768__0011", "1806.01768",
          "Evidential deep learning places a Dirichlet over class probabilities; "
          "EDL reaches 83 percent test accuracy on CIFAR5.",
          title="Evidential Deep Learning", year="2018"),
    chunk("1812.04606__0008", "1812.04606",
          "Outlier exposure trains against an auxiliary dataset of outliers.",
          title="Deep Anomaly Detection with Outlier Exposure", year="2018"),
]
CORPUS[0]["table_markdown"] = "| Method | CIFAR5 |\n| --- | --- |\n| EDL | 83 |"
CORPUS[0]["table_grid_source"] = "geometry"
CORPUS[0]["figure_images"] = "data/figures/1806.01768_p7_fig2.png"


def scripted_llm(calls):
    def chat_fn(messages):
        calls.append(messages)
        if messages[0]["content"] == REWRITE_SYSTEM:
            follow = messages[-1]["content"]
            return "What accuracy does EDL reach on CIFAR5?" if "it" in follow else follow
        user = messages[-1]["content"]
        if "capital of Peru" in user:
            return REFUSAL
        return "EDL reaches 83% accuracy on CIFAR5 [S1]."
    return chat_fn


def upload_factory(chunks, with_corpus):
    docs = chunks + (CORPUS if with_corpus else [])
    return HybridRetriever.from_chunks(docs, embedder=FakeEmbedder())


@pytest.fixture
def client(tmp_path):
    calls = []
    (tmp_path / "results").mkdir()
    (tmp_path / "data" / "figures").mkdir(parents=True)
    service = RagService(
        Settings(data_dir=tmp_path / "data", results_dir=tmp_path / "results",
                 mode="hybrid", max_sessions=3),
        retriever=HybridRetriever.from_chunks(CORPUS, embedder=FakeEmbedder()),
        chat_fn=scripted_llm(calls), tokenizer=fake_tokenizer,
        upload_retriever_factory=upload_factory)
    c = TestClient(create_app(service))
    c.calls, c.tmp = calls, tmp_path
    return c


def make_pdf(path: Path, text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path.read_bytes()


# --- corpus ---------------------------------------------------------------------------


def test_health_and_papers(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["corpus_chunks"] == 2 and h["papers"] == 2
    papers = client.get("/papers").json()["papers"]
    assert [p["arxiv_id"] for p in papers] == ["1806.01768", "1812.04606"]


def test_search_returns_citable_sources(client):
    r = client.post("/search", json={"query": "evidential dirichlet", "k": 2})
    assert r.status_code == 200
    top = r.json()["hits"][0]
    assert top["arxiv_id"] == "1806.01768" and top["pages"] == "p.1"
    assert top["url"] == "https://arxiv.org/abs/1806.01768"
    assert top["table_source"] == "geometry" and "EDL | 83" in top["table_markdown"]
    assert top["figures"] == [{"name": "1806.01768_p7_fig2.png",
                               "url": "/figures/1806.01768_p7_fig2.png"}]
    assert client.post("/search", json={"query": "x", "mode": "magic"}).status_code == 422
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_ask_once(client):
    r = client.post("/ask", json={"question": "What accuracy does EDL get on CIFAR5?"})
    body = r.json()
    assert body["answer"].startswith("EDL reaches 83%") and not body["refused"]
    assert body["sources"][0]["n"] == 1


def test_small_talk_uses_no_llm(client):
    body = client.post("/ask", json={"question": "hi"}).json()
    assert "Ask me anything" in body["answer"] and body["sources"] == []
    assert client.calls == []


# --- conversations ------------------------------------------------------------------------


def test_conversation_with_follow_up(client):
    sid = client.post("/sessions").json()["session_id"]
    first = client.post(f"/sessions/{sid}/messages",
                        json={"question": "Tell me about evidential deep learning"})
    assert first.status_code == 200
    follow = client.post(f"/sessions/{sid}/messages",
                         json={"question": "what accuracy does it reach?"}).json()
    assert follow["standalone"] == "What accuracy does EDL reach on CIFAR5?"
    history = client.get(f"/sessions/{sid}").json()
    assert len(history["turns"]) == 2 and history["scope"] == "corpus"
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_unknown_session_is_404(client):
    r = client.post("/sessions/nope/messages", json={"question": "x"})
    assert r.status_code == 404 and "session" in r.json()["detail"].lower()


def test_old_sessions_are_evicted(client):
    ids = [client.post("/sessions").json()["session_id"] for _ in range(4)]
    assert client.get(f"/sessions/{ids[0]}").status_code == 404, "oldest evicted"
    assert client.get(f"/sessions/{ids[-1]}").status_code == 200


# --- uploads --------------------------------------------------------------------------------


def test_upload_pdf_and_ask_about_it(client):
    sid = client.post("/sessions").json()["session_id"]
    pdf = make_pdf(client.tmp / "mine.pdf",
                   "My Paper on Conformal Prediction\n\nConformal prediction gives "
                   "coverage guarantees for any classifier without retraining.")
    r = client.post(f"/sessions/{sid}/documents",
                    files=[("files", ("mine.pdf", pdf, "application/pdf"))],
                    data={"with_corpus": "false"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scope"] == "uploads"
    assert body["documents"][0]["doc_id"] == "upload:mine.pdf"
    ans = client.post(f"/sessions/{sid}/messages",
                      json={"question": "What does conformal prediction guarantee?"}).json()
    assert ans["sources"][0]["arxiv_id"] == "upload:mine.pdf"
    assert ans["sources"][0]["url"] is None, "uploads have no arXiv page"

    merged = client.post(f"/sessions/{sid}/documents",
                         files=[("files", ("mine.pdf", pdf, "application/pdf"))],
                         data={"with_corpus": "true"}).json()
    assert merged["scope"] == "uploads+corpus"
    back = client.delete(f"/sessions/{sid}/documents").json()
    assert back["scope"] == "corpus" and back["documents"] == []


def test_bad_uploads_are_rejected_with_a_reason(client):
    sid = client.post("/sessions").json()["session_id"]
    r = client.post(f"/sessions/{sid}/documents",
                    files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 400 and "not a .pdf" in r.json()["detail"]
    broken = client.post(f"/sessions/{sid}/documents",
                         files=[("files", ("x.pdf", b"not a pdf", "application/pdf"))])
    assert broken.status_code == 400 and "could not open" in broken.json()["detail"]
    pdf = make_pdf(client.tmp / "a.pdf", "Some text about calibration.")
    many = [("files", (f"a{i}.pdf", pdf, "application/pdf")) for i in range(6)]
    assert client.post(f"/sessions/{sid}/documents", files=many).status_code == 400


# --- provider failures ---------------------------------------------------------------------


def test_provider_errors_map_to_http_codes(tmp_path):
    def down(messages):
        raise llm.ProviderUnavailable("busy")

    def limited(messages):
        raise llm.RateLimitTooLong("m", 1800, True)

    for fn, code in ((down, 503), (limited, 429)):
        service = RagService(Settings(data_dir=tmp_path, results_dir=tmp_path, mode="hybrid"),
                             retriever=HybridRetriever.from_chunks(
                                 CORPUS, embedder=FakeEmbedder()), chat_fn=fn)
        r = TestClient(create_app(service)).post("/ask", json={"question": "EDL accuracy?"})
        assert r.status_code == code, r.text


def test_missing_api_key_is_a_503_with_the_fix(tmp_path, monkeypatch):
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    service = RagService(Settings(data_dir=tmp_path, results_dir=tmp_path, mode="hybrid",
                                  provider="groq"),
                         retriever=HybridRetriever.from_chunks(CORPUS, embedder=FakeEmbedder()))
    c = TestClient(create_app(service))
    assert c.get("/health").json()["llm_ready"] is False
    r = c.post("/ask", json={"question": "EDL accuracy?"})
    assert r.status_code == 503 and "GROQ_API_KEY is not set" in r.json()["detail"]
    assert c.post("/search", json={"query": "evidential"}).status_code == 200, \
        "retrieval works without a key"


def test_client_that_exits_does_not_stop_the_server(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def exits(*a, **k):
        raise SystemExit("The Groq SDK is not installed")

    monkeypatch.setattr(llm, "make_chat_fn", exits)
    service = RagService(Settings(data_dir=tmp_path, results_dir=tmp_path, mode="hybrid",
                                  provider="groq"),
                         retriever=HybridRetriever.from_chunks(CORPUS, embedder=FakeEmbedder()))
    c = TestClient(create_app(service))
    assert c.get("/health").json()["llm_ready"] is True
    r = c.post("/ask", json={"question": "EDL accuracy?"})
    assert r.status_code == 503 and "SDK is not installed" in r.json()["detail"]
    assert c.get("/health").status_code == 200


def test_warm_start_runs_in_the_background(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_WARM_START", "1")
    service = RagService(Settings(data_dir=tmp_path / "empty", results_dir=tmp_path))
    threads = []
    real = service.warm_up_in_background
    monkeypatch.setattr(service, "warm_up_in_background",
                        lambda: threads.append(real()) or threads[-1])
    with TestClient(create_app(service)) as c:  # the context runs the lifespan
        assert c.get("/health").status_code == 200
        threads[0].join(timeout=10)
        assert c.get("/health").json()["loading"] is False
        r = c.post("/search", json={"query": "x"})
        assert r.status_code == 503, "the failed warm start is reported on use"


def test_missing_index_is_503_not_500(tmp_path):
    service = RagService(Settings(data_dir=tmp_path / "empty", results_dir=tmp_path))
    r = TestClient(create_app(service)).post("/search", json={"query": "x"})
    assert r.status_code == 503 and "build_index" in r.json()["detail"]


# --- results -------------------------------------------------------------------------------


def test_uncertainty_results(client):
    assert client.get("/results/uncertainty").status_code == 404
    out = client.tmp / "results" / "uncertainty"
    (out / "figures").mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({"headline": [{"method": "edl"}]}))
    (out / "figures" / "roc_svhn.png").write_bytes(b"\x89PNG fake")
    assert client.get("/results/uncertainty").json()["headline"][0]["method"] == "edl"
    assert client.get("/results/uncertainty/figures/roc_svhn.png").status_code == 200
    assert client.get("/results/uncertainty/figures/..%2Fsummary.json").status_code == 404


def test_retrieval_eval_results(client):
    run = client.tmp / "results" / "eval" / "core_v3"
    run.mkdir(parents=True)
    (run / "summary.csv").write_text("config,recall@5,mrr\ndense,0.5149,0.3519\n"
                                     "hybrid_rerank,0.6791,0.6024\n")
    (run / "significance.csv").write_text("config,vs,metric,delta,significant\n"
                                          "hybrid_rerank,dense,mrr,0.25,True\n")
    assert client.get("/results/retrieval").json()["runs"] == ["core_v3"]
    body = client.get("/results/retrieval/core_v3").json()
    assert body["summary"][1]["recall@5"] == 0.6791
    assert body["significance"][0]["significant"] is True
    assert client.get("/results/retrieval/nope").status_code == 404
    assert client.get("/results/other").status_code == 404


def test_figures_are_served_only_from_the_figures_folder(client):
    (client.tmp / "data" / "figures" / "a_p1_fig1.png").write_bytes(b"\x89PNG")
    assert client.get("/figures/a_p1_fig1.png").status_code == 200
    assert client.get("/figures/missing.png").status_code == 404
    assert client.get("/figures/..%2F..%2Fsecret.png").status_code == 404


# --- found by review: uploads, sessions, loading, provider errors --------------------------


def test_two_uploads_with_the_same_name_are_both_indexed(client):
    sid = client.post("/sessions").json()["session_id"]
    a = make_pdf(client.tmp / "a.pdf", "Temperature scaling calibrates a classifier.")
    b = make_pdf(client.tmp / "b.pdf", "Conformal prediction gives coverage guarantees.")
    r = client.post(f"/sessions/{sid}/documents", data={"with_corpus": "false"},
                    files=[("files", ("paper.pdf", a, "application/pdf")),
                           ("files", ("paper.pdf", b, "application/pdf"))])
    assert r.status_code == 200, r.text
    assert [d["doc_id"] for d in r.json()["documents"]] == ["upload:paper.pdf",
                                                            "upload:paper-2.pdf"]


def test_uploaded_pdfs_are_not_left_on_disk(client, tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    sid = client.post("/sessions").json()["session_id"]
    pdf = make_pdf(client.tmp / "c.pdf", "Deep ensembles average several networks.")
    client.post(f"/sessions/{sid}/documents", files=[("files", ("c.pdf", pdf,
                                                                "application/pdf"))])
    assert list((tmp_path / "tmp").iterdir()) == []


def test_session_keeps_its_mode_through_uploads(client):
    sid = client.post("/sessions", json={"mode": "bm25"}).json()["session_id"]
    service = client.app.state.service
    pdf = make_pdf(client.tmp / "d.pdf", "Label smoothing changes calibration.")
    client.post(f"/sessions/{sid}/documents", files=[("files", ("d.pdf", pdf,
                                                                "application/pdf"))])
    assert service.get_session(sid).chat.search_kwargs["mode"] == "bm25"
    client.delete(f"/sessions/{sid}/documents")
    assert service.get_session(sid).chat.search_kwargs["mode"] == "bm25"


def test_idle_session_expires_on_access(client):
    sid = client.post("/sessions").json()["session_id"]
    service = client.app.state.service
    service.get_session(sid).last_used -= service.settings.session_ttl + 1
    assert client.get(f"/sessions/{sid}").status_code == 404


def test_rejected_api_key_is_a_503_not_a_500(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "revoked")
    rejected = type("AuthenticationError", (Exception,),
                    {"__module__": "groq", "status_code": 401})

    def chat(messages):
        raise rejected("Error code: 401 - Invalid API Key")

    monkeypatch.setattr(llm, "make_chat_fn", lambda *a, **k: chat)
    service = RagService(Settings(data_dir=tmp_path, results_dir=tmp_path, mode="hybrid",
                                  provider="groq"),
                         retriever=HybridRetriever.from_chunks(CORPUS, embedder=FakeEmbedder()))
    r = TestClient(create_app(service)).post("/ask", json={"question": "EDL accuracy?"})
    assert r.status_code == 503 and "API key was rejected" in r.json()["detail"]


def test_conversation_starts_while_models_load(tmp_path):
    service = RagService(Settings(data_dir=tmp_path, results_dir=tmp_path, mode="hybrid"),
                         chat_fn=scripted_llm([]))
    service._loading = True  # as during a first start's download
    c = TestClient(create_app(service))
    sid = c.post("/sessions").json()["session_id"]
    assert c.post(f"/sessions/{sid}/messages", json={"question": "hi"}).status_code == 200
    r = c.post(f"/sessions/{sid}/messages", json={"question": "What is EDL?"})
    assert r.status_code == 503 and "still loading" in r.json()["detail"]
    assert r.headers["retry-after"] == "30"


def test_no_cross_origin_access_unless_configured(tmp_path, monkeypatch):
    def make():
        service = RagService(Settings(data_dir=tmp_path, results_dir=tmp_path))
        return TestClient(create_app(service))

    hdr = {"Origin": "https://evil.example"}
    assert "access-control-allow-origin" not in make().get("/health", headers=hdr).headers
    monkeypatch.setenv("RAG_CORS_ORIGINS", "http://localhost:3000")
    ok = make().get("/health", headers={"Origin": "http://localhost:3000"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"
