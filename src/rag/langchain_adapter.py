"""
Use this project's retriever from LangChain - without LangChain inside it.

The pipeline is deliberately framework-free: retrieval, fusion, reranking,
chunking and evaluation are all ours, which is what makes them measurable and
fixable (see §8 of the README for the bugs that found). But a LangChain
application should still be able to use it, so this adapter exposes
HybridRetriever as a `BaseRetriever`, which any LangChain chain or agent
accepts.

Nothing else in the project imports this file, and LangChain is not a
dependency of the pipeline: `pip install langchain-core` only if you want this.

    from retrieve import HybridRetriever
    from langchain_adapter import as_langchain_retriever

    retriever = as_langchain_retriever(HybridRetriever("data", load_reranker=True),
                                       k=5, mode="hybrid_rerank")
    docs = retriever.invoke("what error does EnD2 get on CIFAR-10?")
    # ... or drop `retriever` into any chain that expects a retriever

Each Document carries the chunk text plus metadata a citation needs
(chunk_id, arxiv_id, title, year, pages, score) and, when they exist, the
rebuilt table grid and figure notes.

Run it as a script for a demo, including a small retrieval-augmented chain
built only from langchain_core primitives and this project's Groq/Gemini
client:

  python src/rag/langchain_adapter.py "what error does EnD2 get on CIFAR-10?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from llm import add_provider_args, exit_on_rate_limit, make_complete_fn, resolve  # noqa: E402
from retrieve import DEFAULT_RERANKER, expand_split_tables  # noqa: E402

DOC_FIELDS = ("chunk_id", "arxiv_id", "title", "year", "page_start", "page_end",
              "score", "is_reference", "table_markdown", "table_grid_source",
              "figure_note", "figure_images")


def to_documents(hits: list[dict]):
    """Retrieved chunks as LangChain Documents (metadata keeps the citation)."""
    from langchain_core.documents import Document

    return [Document(page_content=h["text"],
                     metadata={f: h[f] for f in DOC_FIELDS if f in h})
            for h in hits]


def as_langchain_retriever(retriever, k: int = 5, expand_tables: bool = True,
                           **search_kwargs):
    """Wrap a HybridRetriever as a LangChain BaseRetriever.

    search_kwargs are passed straight through to HybridRetriever.search, so
    mode, fusion, max_per_paper, mmr_lambda, min_score and the rest all work.
    """
    from langchain_core.retrievers import BaseRetriever

    class HybridRetrieverAdapter(BaseRetriever):
        """LangChain view of this project's hybrid retriever."""

        model_config = {"arbitrary_types_allowed": True}
        inner: object
        k: int = 5
        expand_tables: bool = True
        search_kwargs: dict = {}

        def _get_relevant_documents(self, query: str, **_):
            hits = self.inner.search(query, k=self.k, **self.search_kwargs)
            lookup = getattr(self.inner, "chunk_by_id", None)
            if self.expand_tables and lookup is not None:
                hits = expand_split_tables(hits, lookup)
            return to_documents(hits)

    return HybridRetrieverAdapter(inner=retriever, k=k, expand_tables=expand_tables,
                                  search_kwargs=search_kwargs)


def build_chain(retriever, complete_fn):
    """A minimal RAG chain from langchain_core primitives: retrieve, then answer
    with this project's grounded prompt and its own LLM client."""
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    def answer(payload: dict) -> str:
        hits = [{**d.metadata, "text": d.page_content} for d in payload["docs"]]
        return complete_fn(SYSTEM_PROMPT, build_user_prompt(payload["question"], hits))

    return ({"docs": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(answer))


@exit_on_rate_limit
def main():
    ap = argparse.ArgumentParser(description="LangChain adapter demo.")
    ap.add_argument("question")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--reranker", default=DEFAULT_RERANKER)
    ap.add_argument("--docs-only", action="store_true",
                    help="Just show the Documents; no LLM call.")
    add_provider_args(ap)
    args = ap.parse_args()

    try:
        import langchain_core  # noqa: F401
    except ImportError:
        sys.exit("This demo needs LangChain: pip install langchain-core")

    from retrieve import HybridRetriever

    retriever = as_langchain_retriever(
        HybridRetriever(args.data_dir, reranker_model=args.reranker,
                        load_reranker=True),
        k=args.k, mode="hybrid_rerank")

    docs = retriever.invoke(args.question)
    print(f"\n{len(docs)} Document(s) from the LangChain retriever:")
    for i, d in enumerate(docs, 1):
        m = d.metadata
        print(f"  [{i}] {m['arxiv_id']} ({m.get('year')}) "
              f"pp.{m['page_start']}-{m['page_end']} score={m.get('score', 0):.3f}"
              + ("  [+ rebuilt table]" if m.get("table_markdown") else ""))
        print(f"      {' '.join(d.page_content.split())[:150]}...")
    if args.docs_only:
        return

    provider, model, _ = resolve(args)
    chain = build_chain(retriever, make_complete_fn(model, provider=provider))
    print("\nAnswer from the chain:")
    print(chain.invoke(args.question))


if __name__ == "__main__":
    main()
