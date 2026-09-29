"""Tests for query expansion and multi-query fusion in HybridRetriever.

The embedding model and FAISS are replaced with stubs, so these run offline
without downloading any model.
"""

import sys
import types
from pathlib import Path

import pytest

RAG = Path(__file__).resolve().parents[1] / "src" / "rag"
sys.path.insert(0, str(RAG))

# retrieve.py imports sentence_transformers and faiss at module load; stub them
# if unavailable so the ranking logic can be tested in isolation.
for name in ("sentence_transformers", "faiss"):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.SentenceTransformer = object
            sys.modules[name] = stub

from query_expansion import QueryExpander, parse_variants  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from retrieve import HybridRetriever, tokenize_for_bm25  # noqa: E402

# --- parse_variants ---------------------------------------------------------


def test_parse_strips_numbering_bullets_and_quotes():
    raw = '1. "false positive rate at 95% TPR"\n- FPR@95TPR of ensembles\n* third one'
    out = parse_variants(raw, "what is the FPR95?", 3)
    assert out == [
        "false positive rate at 95% TPR",
        "FPR@95TPR of ensembles",
        "third one",
    ]


def test_parse_drops_original_duplicates_blanks_and_headers():
    raw = "Rephrasings:\n\nWhat is the FPR95?\nvariant a\nVARIANT A\nvariant b"
    assert parse_variants(raw, "what is the fpr95?", 5) == ["variant a", "variant b"]


def test_parse_caps_at_n():
    assert len(parse_variants("a\nb\nc\nd", "q", 2)) == 2


# --- QueryExpander cache ----------------------------------------------------


def test_expander_caches_and_keeps_original_first(tmp_path):
    calls = []

    def fake_llm(system, user):
        calls.append(user)
        return "alt one\nalt two"

    cache = tmp_path / "exp.json"
    e = QueryExpander(fake_llm, model_name="m", n=2, cache_path=cache)
    assert e.expand("orig q") == ["orig q", "alt one", "alt two"]
    assert e.expand("orig q") == ["orig q", "alt one", "alt two"]
    assert len(calls) == 1, "second call must hit the cache"

    # a fresh expander reads the on-disk cache: no LLM call at all
    e2 = QueryExpander(lambda s, u: pytest.fail("should not call LLM"),
                       model_name="m", n=2, cache_path=cache)
    assert e2.expand("orig q")[1:] == ["alt one", "alt two"]


def test_cache_key_includes_model_and_n(tmp_path):
    cache = tmp_path / "exp.json"
    QueryExpander(lambda s, u: "x", "m1", 2, cache).expand("q")
    calls = []
    QueryExpander(lambda s, u: calls.append(1) or "y", "m2", 2, cache).expand("q")
    assert calls, "different model must not reuse another model's expansions"


# --- multi-query fusion in HybridRetriever ----------------------------------

CHUNKS = [
    "evidential deep learning dirichlet evidence",          # 0
    "false positive rate at 95 percent true positive rate",  # 1  (spelled out)
    "deep ensembles cifar results table",                    # 2
    "prior networks distributional uncertainty",             # 3
]


def make_retriever():
    r = HybridRetriever.__new__(HybridRetriever)
    r.meta = [
        {"chunk_id": f"c{i}", "arxiv_id": f"p{i}", "is_reference": False, "text": t}
        for i, t in enumerate(CHUNKS)
    ]
    r.bm25 = BM25Okapi([tokenize_for_bm25(t) for t in CHUNKS])
    r._reranker = None
    # dense stub: pretend the embedder only "understands" exact wording
    r._dense_rank = lambda q, n: r._bm25_rank(q, n)
    return r


def test_single_query_misses_spelled_out_chunk():
    r = make_retriever()
    ids = r.search_ids("fpr95 of ensembles", k=2, mode="bm25", max_per_paper=0)
    assert "c1" not in ids


def test_multi_query_recovers_chunk_via_rephrasing():
    r = make_retriever()
    variants = ["fpr95 of ensembles",
                "false positive rate at 95 percent true positive rate"]
    ids = r.search_ids("fpr95 of ensembles", k=2, mode="bm25",
                       max_per_paper=0, query_variants=variants)
    assert "c1" in ids and "c2" in ids


def test_single_variant_is_same_as_no_expansion():
    r = make_retriever()
    base = r.search_ids("dirichlet evidence", k=3, mode="hybrid", max_per_paper=0)
    same = r.search_ids("dirichlet evidence", k=3, mode="hybrid", max_per_paper=0,
                        query_variants=["dirichlet evidence"])
    assert base == same
