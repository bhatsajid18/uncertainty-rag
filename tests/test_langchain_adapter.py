"""Tests for the LangChain adapter, with langchain_core stubbed.

LangChain is not a dependency of the pipeline, so the test supplies the two
classes the adapter needs (Document, BaseRetriever) and enough of the Runnable
protocol to compose the demo chain.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_chat  # noqa: E402,F401  (installs module stubs)
from test_chat import FakeEmbedder, chunk  # noqa: E402

import langchain_adapter as la  # noqa: E402
from generate import TABLE_GRID_LABEL_GEOMETRY  # noqa: E402
from retrieve import HybridRetriever  # noqa: E402


class _Document:
    def __init__(self, page_content, metadata=None):
        self.page_content, self.metadata = page_content, metadata or {}


class _BaseRetriever:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def invoke(self, query):
        return self._get_relevant_documents(query)


class _Runnable:
    def __init__(self, func):
        self.func = func

    def __ror__(self, other):  # {"a": x, "b": y} | RunnableLambda(f)
        def run(query):
            payload = {k: (v.invoke(query) if hasattr(v, "invoke") else query)
                       for k, v in other.items()}
            return self.func(payload)
        return _Runnable(run)

    def invoke(self, query):
        return self.func(query)


@pytest.fixture(autouse=True)
def fake_langchain(monkeypatch):
    core = types.ModuleType("langchain_core")
    docs = types.ModuleType("langchain_core.documents")
    docs.Document = _Document
    retrievers = types.ModuleType("langchain_core.retrievers")
    retrievers.BaseRetriever = _BaseRetriever
    runnables = types.ModuleType("langchain_core.runnables")
    runnables.RunnableLambda = _Runnable
    runnables.RunnablePassthrough = lambda: None
    for name, mod in [("langchain_core", core),
                      ("langchain_core.documents", docs),
                      ("langchain_core.retrievers", retrievers),
                      ("langchain_core.runnables", runnables)]:
        monkeypatch.setitem(sys.modules, name, mod)


def _inner():
    docs = [chunk("a__0000", "1806.01768", "evidential deep learning dirichlet "
                  "uncertainty", title="Evidential Deep Learning"),
            chunk("b__0000", "1812.04606", "outlier exposure auxiliary dataset",
                  title="Outlier Exposure")]
    docs[0]["table_markdown"] = "| Method | CIFAR5 |\n| --- | --- |\n| EDL | 83 |"
    docs[0]["table_grid_source"] = "geometry"
    return HybridRetriever.from_chunks(docs, embedder=FakeEmbedder())


def test_documents_carry_the_citation_metadata():
    retriever = la.as_langchain_retriever(_inner(), k=2, mode="dense")
    docs = retriever.invoke("evidential dirichlet")
    assert docs[0].page_content.startswith("evidential deep learning")
    m = docs[0].metadata
    assert m["chunk_id"] == "a__0000" and m["arxiv_id"] == "1806.01768"
    assert m["title"] == "Evidential Deep Learning" and "score" in m
    assert m["page_start"] == 1 and m["table_grid_source"] == "geometry"


def test_search_kwargs_reach_the_retriever():
    retriever = la.as_langchain_retriever(_inner(), k=1, mode="dense", min_score=2.0)
    assert retriever.invoke("evidential") == [], "the score floor was applied"


def test_chain_answers_from_the_retrieved_documents():
    seen = {}

    def complete(system, user):
        seen["system"], seen["user"] = system, user
        return "EDL reaches 83% on CIFAR-5 [S1]."

    retriever = la.as_langchain_retriever(_inner(), k=2, mode="dense")
    chain = la.build_chain(retriever, complete)
    assert chain.invoke("cifar5 accuracy") == "EDL reaches 83% on CIFAR-5 [S1]."
    assert "Question: cifar5 accuracy" in seen["user"]
    assert "1806.01768" in seen["user"], "sources are labelled for citation"
    assert TABLE_GRID_LABEL_GEOMETRY in seen["user"], "rebuilt table is passed on"
    assert "Cite every factual claim" in seen["system"]


def test_adapter_needs_no_langchain_import_at_module_level(monkeypatch):
    for name in list(sys.modules):
        if name.startswith("langchain_core"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    import importlib

    importlib.reload(la)  # must not raise: the pipeline runs without LangChain
