"""Tests for conversational chat, in-memory indexes and PDF uploads.

Runs offline: the LLM, embedding model and tokenizer are replaced with small
deterministic stand-ins. FAISS and PyMuPDF are used for real where installed.
"""

import json
import re
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

RAG = Path(__file__).resolve().parents[1] / "src" / "rag"
sys.path.insert(0, str(RAG))
for name in ("sentence_transformers", "transformers"):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.SentenceTransformer = object
            stub.AutoTokenizer = object
            sys.modules[name] = stub

faiss = pytest.importorskip("faiss")
pymupdf = pytest.importorskip("pymupdf")

import chat  # noqa: E402
import retrieve  # noqa: E402
import uploads  # noqa: E402
from generate import REFUSAL  # noqa: E402

# --- stand-ins -------------------------------------------------------------------


class FakeEmbedder:
    """Hashed bag of words -> normalised vector. Similar words, similar vectors."""

    def encode(self, texts, **_):
        out = np.zeros((len(texts), 64), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % 64] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def fake_tokenizer(text, **_):
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    return {"input_ids": list(range(len(spans))), "offset_mapping": spans}


def chunk(cid, doc, text, title="T", year="2020"):
    return {"chunk_id": cid, "arxiv_id": doc, "title": title, "year": year,
            "page_start": 1, "page_end": 1, "is_reference": False, "text": text}


class RecordingRetriever:
    def __init__(self, hits):
        self.hits, self.queries, self.variants = hits, [], []

    def search(self, query, k=5, query_variants=None, **_):
        self.queries.append(query)
        self.variants.append(query_variants)
        return self.hits[:k]


class ScriptedLLM:
    """Returns queued replies; records every message list it was sent."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


HITS = [chunk("oe__1", "1812.04606", "outlier exposure fpr95 results", "Outlier Exposure")]

# --- rewrite helpers -----------------------------------------------------------------


def test_clean_rewrite():
    assert chat.clean_rewrite('Standalone question: "What is X?"', "fb") == "What is X?"
    assert chat.clean_rewrite("\n\n  What is Y?\nextra", "fb") == "What is Y?"
    assert chat.clean_rewrite("", "fallback q") == "fallback q"
    assert chat.clean_rewrite("word " * 200, "short q") == "short q"  # answered instead


def test_strip_citations():
    assert chat.strip_citations("A [S1] B [S2][S3].") == "A B."
    assert chat.strip_citations("x 【S1†L5-L7】 y") == "x y"


# --- conversation ------------------------------------------------------------------------


def test_first_turn_skips_rewrite():
    llm = ScriptedLLM(["Outlier exposure trains on outliers [S1]."])
    r = RecordingRetriever(HITS)
    turn = chat.ChatSession(r, llm).ask("What is outlier exposure?")
    assert len(llm.calls) == 1, "no rewrite call on the first turn"
    assert r.queries == ["What is outlier exposure?"]
    assert turn.standalone == turn.question


def test_follow_up_is_rewritten_and_retrieved_standalone():
    llm = ScriptedLLM([
        "Outlier exposure trains on auxiliary outliers [S1].",
        "What is the FPR95 of outlier exposure?",          # the rewrite
        "It reaches low FPR95 [S1].",
    ])
    r = RecordingRetriever(HITS)
    s = chat.ChatSession(r, llm)
    s.ask("What is outlier exposure?")
    turn = s.ask("what about its FPR95?")

    assert r.queries[-1] == "What is the FPR95 of outlier exposure?"
    rewrite_msgs = llm.calls[1]
    assert "Follow-up: what about its FPR95?" in rewrite_msgs[-1]["content"]

    answer_msgs = llm.calls[2]
    assert chat.CHAT_RULES in answer_msgs[0]["content"]
    history = [m["content"] for m in answer_msgs[1:-1]]
    assert history == ["What is outlier exposure?",
                       "Outlier exposure trains on auxiliary outliers."], \
        "earlier answer passed as context, with stale [S1] citations removed"
    final = answer_msgs[-1]["content"]
    assert "what about its FPR95?" in final
    assert "In context, this means: What is the FPR95 of outlier exposure?" in final
    assert turn.answer == "It reaches low FPR95 [S1]."


def test_history_window_and_reset():
    llm = ScriptedLLM(["a1", "q2 standalone", "a2", "q3 standalone", "a3"])
    s = chat.ChatSession(RecordingRetriever(HITS), llm, history_turns=1)
    s.ask("q1")
    s.ask("q2")
    s.ask("q3")
    assert [m["content"] for m in llm.calls[-1][1:-1]] == ["q2", "a2"], \
        "only the most recent turn is sent"
    s.reset()
    assert s.turns == []


def test_expander_receives_standalone_question():
    class Exp:
        def expand(self, q):
            return [q, q + " rephrased"]
    llm = ScriptedLLM(["a1", "standalone two", "a2"])
    r = RecordingRetriever(HITS)
    s = chat.ChatSession(r, llm, expander=Exp())
    s.ask("one")
    s.ask("two?")
    assert r.variants[-1] == ["standalone two", "standalone two rephrased"]


def test_no_hits_refuses_without_calling_llm():
    llm = ScriptedLLM([])
    turn = chat.ChatSession(RecordingRetriever([]), llm).ask("anything")
    assert turn.refused and llm.calls == []


def test_refused_property():
    assert chat.Turn("q", "q", REFUSAL).refused
    assert not chat.Turn("q", "q", "An answer [S1].").refused


# --- in-memory indexes ---------------------------------------------------------------------


def make_base_corpus(d: Path, emb: FakeEmbedder):
    base = [chunk("a__0", "paperA", "evidential deep learning dirichlet evidence"),
            chunk("b__0", "paperB", "deep ensembles average several networks")]
    vecs = emb.encode([c["text"] for c in base])
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    d.mkdir()
    faiss.write_index(idx, str(d / "index.faiss"))
    (d / "chunk_meta.json").write_text(json.dumps(base))
    (d / "index_config.json").write_text(json.dumps(
        {"model": "fake-embedder", "query_prefix": ""}))
    return base, vecs


UPLOADED = [chunk("upload:my.pdf__0000", "upload:my.pdf",
                  "graph neural networks message passing", year="")]


def test_isolated_upload_index_contains_only_uploads():
    r = retrieve.HybridRetriever.from_chunks(UPLOADED, embedder=FakeEmbedder(),
                                             query_prefix="")
    assert r.index.ntotal == 1
    assert r.search_ids("message passing", mode="hybrid") == ["upload:my.pdf__0000"]


def test_merged_index_keeps_corpus_vectors_and_adds_uploads(tmp_path):
    emb = FakeEmbedder()
    base, base_vecs = make_base_corpus(tmp_path / "data", emb)
    r = retrieve.HybridRetriever.from_chunks(UPLOADED, base_dir=tmp_path / "data",
                                             embedder=emb)
    assert r.index.ntotal == len(base) + 1
    assert np.allclose(r.index.reconstruct_n(0, len(base)), base_vecs), \
        "corpus vectors copied, not re-embedded"
    assert [m["chunk_id"] for m in r.meta] == ["a__0", "b__0", "upload:my.pdf__0000"]
    assert r.search_ids("message passing", mode="dense", k=1) == ["upload:my.pdf__0000"]
    assert r.search_ids("dirichlet evidence", mode="dense", k=1) == ["a__0"]
    assert r.embed_model_name == "fake-embedder", "base corpus model wins when merging"


def test_from_chunks_rejects_empty():
    with pytest.raises(ValueError):
        retrieve.HybridRetriever.from_chunks([], embedder=FakeEmbedder())


# --- uploads from real PDFs -----------------------------------------------------------------


def make_pdf(path: Path, pages: list[str], title: str = ""):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=9)
    if title:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()


def test_chunk_pdf_format_pages_and_title(tmp_path):
    p = tmp_path / "study.pdf"
    make_pdf(p, ["alpha " * 150, "beta " * 150], title="A Study of Things")
    chunks = uploads.chunk_pdf(p, fake_tokenizer, chunk_size=100, overlap=10)
    assert chunks and all(set(retrieve.META_FIELDS) <= set(c) for c in chunks)
    assert chunks[0]["chunk_id"] == "upload:study.pdf__0000"
    assert chunks[0]["arxiv_id"] == "upload:study.pdf"
    assert chunks[0]["title"] == "A Study of Things"
    assert chunks[0]["page_start"] == 1 and chunks[-1]["page_end"] == 2
    assert all(c["year"] == "" for c in chunks)


def test_title_falls_back_to_first_line(tmp_path):
    p = tmp_path / "x.pdf"
    make_pdf(p, ["Graph Networks for Molecules\n" + "body text " * 50])
    assert uploads.pdf_title(p) == "Graph Networks for Molecules"


def test_scanned_pdf_rejected(tmp_path):
    p = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(p)
    with pytest.raises(uploads.UploadError, match="no extractable text"):
        uploads.chunk_pdf(p, fake_tokenizer)


def test_upload_checks(tmp_path):
    with pytest.raises(uploads.UploadError, match="not found"):
        uploads.check_pdf(tmp_path / "missing.pdf")
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    with pytest.raises(uploads.UploadError, match="not a .pdf"):
        uploads.check_pdf(txt)
    big = tmp_path / "big.pdf"
    make_pdf(big, ["page"] * 5)
    with pytest.raises(uploads.UploadError, match="page limit"):
        uploads.check_pdf(big, max_pages=3)


def test_same_file_name_twice_gets_distinct_ids(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for d in ("a", "b"):
        make_pdf(tmp_path / d / "paper.pdf", ["words " * 120])
    chunks = uploads.load_uploads([tmp_path / "a/paper.pdf", tmp_path / "b/paper.pdf"],
                                  fake_tokenizer)
    docs = {c["arxiv_id"] for c in chunks}
    assert docs == {"upload:paper.pdf", "upload:paper-2.pdf"}
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_upload_to_chat_end_to_end(tmp_path):
    """PDF -> chunks -> isolated index -> two-turn conversation."""
    p = tmp_path / "gnn.pdf"
    make_pdf(p, ["Graph neural networks pass messages between neighbouring nodes. " * 20,
                 "Oversmoothing blurs features as layers stack deeper. " * 20],
             title="Graph Neural Networks")
    r = retrieve.HybridRetriever.from_chunks(
        uploads.load_uploads([p], fake_tokenizer, chunk_size=60, overlap=5),
        embedder=FakeEmbedder(), query_prefix="")
    rewritten = "Why does oversmoothing blur features as layers stack deeper?"
    llm = ScriptedLLM(["They pass messages between nodes [S1].", rewritten,
                       "Features blur [S1]."])
    s = chat.ChatSession(r, llm, search_kwargs={"mode": "hybrid", "max_per_paper": 0})
    s.ask("How do graph neural networks work?")
    turn = s.ask("what goes wrong when they get deeper?")
    assert turn.standalone == rewritten
    assert "oversmoothing" in turn.hits[0]["text"].lower(), \
        "the rewritten follow-up retrieved the passage it is about"
    assert turn.hits[0]["page_end"] == 2 and turn.hits[0]["arxiv_id"] == "upload:gnn.pdf"


# --- regressions from the first real chat session -------------------------------------


def test_source_label_names_the_paper():
    from generate import build_user_prompt, source_tag
    h = chunk("x__1", "2205.09526", "EnD2 error 11.46 on CIFAR-10",
              title="Simple Regularisation for Uncertainty-Aware Knowledge Distillation",
              year="2022")
    assert '"Simple Regularisation for Uncertainty-Aware Knowledge Distillation"' \
        in source_tag(h, 1)
    assert "Simple Regularisation" in build_user_prompt("q", [h]), \
        "the model must see which paper a number comes from"


def test_attribution_rule_in_prompt():
    from generate import SYSTEM_PROMPT
    assert "Never present one paper's number as another paper's own result" in SYSTEM_PROMPT


def test_gpt_oss_citations_normalised_in_answers():
    from generate import normalize_citations
    assert normalize_citations("≈ 88.54 %【S1】 and x【S2†L5-L7】 [S3]") == \
        "≈ 88.54 %[S1] and x[S2] [S3]"
    llm = ScriptedLLM(["It is 5% error【S1†L2-L4】."])
    turn = chat.ChatSession(RecordingRetriever(HITS), llm).ask("q?")
    assert turn.answer == "It is 5% error[S1]."


def test_recapitalised_rewrite_keeps_users_wording():
    assert chat.clean_rewrite("What does ensemble distribution distillation do?",
                              "what does ensemble distribution distillation do") == \
        "what does ensemble distribution distillation do"
    assert chat.clean_rewrite("What is the FPR95 of outlier exposure?",
                              "what about its FPR95?") == \
        "What is the FPR95 of outlier exposure?"
