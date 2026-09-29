"""
Grounded answer generation for the RAG pipeline.

Retrieves top-k chunks for a question using the multi-strategy HybridRetriever
(dense / BM25 / hybrid fusion / cross-encoder reranking), builds a grounded
prompt with labeled sources, and calls a Groq-hosted LLM to answer STRICTLY
from those sources with inline citations like [S1]. If the sources are
insufficient, the model is instructed to say so rather than invent an answer.

Requires:
  pip install groq python-dotenv
  GROQ_API_KEY in .env (or the environment)
  a prebuilt index (run build_index.py build first)

Usage:
  python generate.py "How does evidential deep learning quantify uncertainty?"
  python generate.py "FPR95 of deep ensembles on CIFAR-10" -k 6
  python generate.py "..." --mode dense           # compare retrieval strategies
  python generate.py "..." --mode hybrid_rerank   # default
  python generate.py "..." --show-sources         # print the retrieved chunks too
  python generate.py "..." --multi-query 3        # LLM query expansion + RRF

Because --mode is a flag, the same question can be answered from different
retrieval strategies, which separates two distinct questions: does better
retrieval rank the right chunks higher, and does it actually produce better
ANSWERS? Iteration 3 measures both.
"""

import argparse
import re
import sys
from pathlib import Path

# reuse the multi-strategy retriever from retrieve.py and the shared Groq
# helpers from llm.py - no retrieval or API logic is duplicated here
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import (  # noqa: E402
    DEFAULT_PROVIDER, add_provider_args, default_model, exit_on_rate_limit,
    make_complete_fn, resolve,
)
from query_expansion import QueryExpander  # noqa: E402
from retrieve import (  # noqa: E402
    DEFAULT_RERANKER, MODES, REF_PENALTY, HybridRetriever, expand_split_tables,
)

SYSTEM_PROMPT = (
    "You are a precise research assistant answering questions about machine "
    "learning papers. Follow these rules strictly:\n"
    "1. Answer ONLY using the numbered sources provided by the user. Do not use "
    "any outside knowledge.\n"
    "2. Cite every factual claim with its source tag, e.g. [S1] or [S2]. You may "
    "cite multiple sources for one claim, e.g. [S1][S3].\n"
    "3. If the sources do not contain enough information to answer the question, "
    "reply with exactly: 'The retrieved sources do not contain enough "
    "information to answer this question.' and nothing else.\n"
    "4. Be concise and factual. Do not speculate or add caveats beyond what the "
    "sources support.\n"
    "5. Each source is labelled with the paper it comes from. Attribute results "
    "to the paper that reports them: if a source reports a number for another "
    "method (for example a baseline in its own experiments), say so, e.g. "
    "'reported by <paper> for <method>'. Never present one paper's number as "
    "another paper's own result.\n"
    "6. Tables appear as flattened text: the column headers come first, then "
    "each row as a label followed by one value per column (values often carry "
    "a ± term). For every number you take from a table, name its row and "
    "column, e.g. 'EnD2, C10 error: 7.3 ± 0.2'. Results tables usually include "
    "baselines: when asked what a paper reports, give the paper's own method "
    "and label any baseline numbers as baselines. If you cannot tie a number "
    "to its row and column with certainty, say the table is ambiguous instead "
    "of guessing. Some sources are followed by the same table rebuilt as a "
    "Markdown grid: use it to find which value sits in which row and column, "
    "but cite the source it belongs to, and if the grid and the flattened text "
    "disagree, trust the text and say the table is ambiguous."
)

# The exact refusal the model is told to give; chat.py uses it to recognise one.
REFUSAL = ("The retrieved sources do not contain enough information to answer "
           "this question.")
assert REFUSAL in SYSTEM_PROMPT.replace("'", "")


def source_tag(hit: dict, n: int) -> str:
    """Human-readable citation label for a retrieved chunk."""
    if hit["page_start"] == hit["page_end"]:
        loc = f"p.{hit['page_start']}"
    else:
        loc = f"pp.{hit['page_start']}-{hit['page_end']}"
    # The title matters: without it the model cannot tell which paper a chunk
    # belongs to, and will attribute a baseline number quoted in paper A to the
    # paper that proposed the method. Uploaded PDFs have no year, so the
    # brackets are skipped rather than printing "()".
    year = f" ({hit['year']})" if hit.get("year") else ""
    title = f' "{hit["title"]}"' if hit.get("title") else ""
    return f"S{n}: {hit['arxiv_id']}{year}{title}, {loc}"


_GPT_OSS_CITATION = re.compile(r"【\s*(S\d+)[^】]*】")


def normalize_citations(text: str) -> str:
    """Rewrite gpt-oss style citations (【S1†L5-L7】) into this project's [S1].

    gpt-oss models sometimes ignore the requested citation format and emit
    their own. Normalising keeps every answer's citations consistent for
    display and for anything that parses them later.
    """
    return _GPT_OSS_CITATION.sub(r"[\1]", text)


TABLE_GRID_LABEL = "Table from this source, rebuilt as a grid:"
# Grids recovered from the PDF's column positions cannot contain a value that
# is not in the paper, so the model is told where a grid came from.
TABLE_GRID_LABEL_GEOMETRY = ("Table from this source, rebuilt from the PDF's own "
                             "column positions:")
FIGURE_LABEL = "Figures on these pages:"


def build_user_prompt(question: str, hits: list[dict]) -> str:
    lines = ["Sources:\n"]
    shown_grids = set()
    for n, h in enumerate(hits, 1):
        text = " ".join(h["text"].split())
        lines.append(f"[{source_tag(h, n)}]\n{text}\n")
        # Rebuilt tables from table_notes.py. Both halves of a split table can
        # carry the same grid, so each grid is shown once.
        label = (TABLE_GRID_LABEL_GEOMETRY if h.get("table_grid_source") == "geometry"
                 else TABLE_GRID_LABEL)
        for grid in h.get("table_markdown", "").split("\n\n"):
            if grid.strip() and grid not in shown_grids:
                shown_grids.add(grid)
                lines.append(f"{label}\n{grid}\n")
        if h.get("figure_note"):
            lines.append(f"{FIGURE_LABEL} {h['figure_note']}\n")
    lines.append(f"\nQuestion: {question}")
    return "\n".join(lines)


def answer(
    question: str,
    data_dir: Path,
    k: int = 5,
    model: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    fallback_models: tuple[str, ...] = (),
    include_refs: bool = False,
    mode: str = "hybrid_rerank",
    candidate_n: int = 20,
    fusion: str = "rrf",
    alpha: float = 0.5,
    max_per_paper: int = 2,
    min_score_frac: float = 0.25,
    reranker: str = DEFAULT_RERANKER,
    multi_query: int = 0,
    mmr_lambda: float | None = None,
    min_score: float | None = None,
    ref_penalty: float = REF_PENALTY,
):
    """Retrieve, then generate a grounded answer.

    Returns (answer_text, hits, query_variants). multi_query > 0 asks the LLM
    for that many rephrasings and fuses retrieval across all of them.
    """
    complete = make_complete_fn(model, provider=provider,
                                fallback_models=fallback_models)

    variants = None
    if multi_query > 0:
        # model is None when the caller wants the provider's default. The cache
        # is keyed by model name, so passing None through would file the
        # rephrasings under "None" and re-generate them - at a free-tier LLM
        # call each - as soon as the same question arrived with the model named
        # explicitly, as the CLI does.
        expander = QueryExpander(
            complete,
            model_name=model or default_model(provider),
            n=multi_query,
            cache_path=data_dir / "query_expansions.json",
        )
        variants = expander.expand(question)

    # the reranker is ~1.1 GB, so only load it for the mode that needs it
    retr = HybridRetriever(
        data_dir,
        reranker_model=reranker,
        load_reranker=(mode == "hybrid_rerank"),
    )
    hits = retr.search(
        question,
        k=k,
        mode=mode,
        candidate_n=candidate_n,
        include_refs=include_refs,
        fusion=fusion,
        alpha=alpha,
        max_per_paper=max_per_paper,
        min_score_frac=min_score_frac,
        query_variants=variants,
        mmr_lambda=mmr_lambda,
        min_score=min_score,
        ref_penalty=ref_penalty,
    )
    if not hits:
        if min_score is not None:
            return REFUSAL, [], variants  # nothing cleared the floor: no LLM call
        return "No chunks retrieved (is the index built?).", [], variants

    hits = expand_split_tables(hits, retr.chunk_by_id)
    user_prompt = build_user_prompt(question, hits)
    text = normalize_citations(complete(SYSTEM_PROMPT, user_prompt))
    return text, hits, variants


@exit_on_rate_limit
def main():
    ap = argparse.ArgumentParser(description="Ask a grounded, cited question.")
    ap.add_argument("question", type=str)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("-k", type=int, default=5,
                    help="Number of sources handed to the model.")
    add_provider_args(ap)
    ap.add_argument("--show-sources", action="store_true",
                    help="Also print the retrieved chunk text.")

    # --- retrieval options (passed through to HybridRetriever) ---
    ret = ap.add_argument_group("retrieval")
    ret.add_argument("--mode", choices=MODES, default="hybrid_rerank",
                     help="Retrieval strategy.")
    ret.add_argument("--candidate-n", type=int, default=20,
                     help="First-stage candidate pool before rerank/cut.")
    ret.add_argument("--fusion", choices=("rrf", "weighted"), default="rrf")
    ret.add_argument("--alpha", type=float, default=0.5,
                     help="Weight on dense when --fusion weighted.")
    ret.add_argument("--max-per-paper", type=int, default=2,
                     help="Max sources from any one paper (0 disables).")
    ret.add_argument("--min-score-frac", type=float, default=0.25,
                     help="Relevance floor for diversity picks (0 disables).")
    ret.add_argument("--reranker", default=DEFAULT_RERANKER)
    ret.add_argument("--include-refs", action="store_true",
                     help="Score bibliography chunks with no penalty at all.")
    ret.add_argument("--ref-penalty", type=float, default=REF_PENALTY,
                     help="Score multiplier for bibliography chunks "
                          f"(default {REF_PENALTY}; 0 excludes them).")
    ret.add_argument("--mmr", type=float, default=None, metavar="LAMBDA",
                     help="Maximal Marginal Relevance selection instead of the "
                          "per-paper cap (e.g. 0.7).")
    ret.add_argument("--min-score", type=float, default=None,
                     help="Refuse without an LLM call if no source's reranker "
                          "score reaches this (0-1).")
    ret.add_argument("--multi-query", type=int, default=0, metavar="N",
                     help="Generate N LLM rephrasings and fuse retrieval "
                          "across them (default 1; 0 = off, and saves one "
                          "LLM call per new question).")

    args = ap.parse_args()

    provider, model, _ = resolve(args)
    text, hits, variants = answer(
        args.question,
        args.data_dir,
        k=args.k,
        model=model,
        provider=provider,
        include_refs=args.include_refs,
        mode=args.mode,
        candidate_n=args.candidate_n,
        fusion=args.fusion,
        alpha=args.alpha,
        max_per_paper=args.max_per_paper,
        min_score_frac=args.min_score_frac,
        reranker=args.reranker,
        multi_query=args.multi_query,
        mmr_lambda=args.mmr,
        min_score=args.min_score,
        ref_penalty=args.ref_penalty,
    )

    print(f'\nQuestion: {args.question}')
    print(f"Retrieval: mode={args.mode}, k={args.k}, "
          f"max_per_paper={args.max_per_paper}, "
          f"min_score_frac={args.min_score_frac}, "
          f"multi_query={args.multi_query}")
    if variants and len(variants) > 1:
        print("Query variants:")
        for v in variants[1:]:
            print(f"  - {v}")
    print("=" * 60)
    print(text)
    print("=" * 60)
    print("\nSources provided to the model:")
    for n, h in enumerate(hits, 1):
        loc = (f"p.{h['page_start']}" if h["page_start"] == h["page_end"]
               else f"pp.{h['page_start']}-{h['page_end']}")
        # reranked hits carry both the rerank score and the fusion score that
        # got them into the candidate pool - showing both makes it clear which
        # stage is responsible for a given result
        score_str = f"score={h['score']:.3f}"
        if "first_stage_score" in h:
            score_str += f" (fusion={h['first_stage_score']:.4f})"
        ref_flag = " [ref]" if h.get("is_reference") else ""
        print(f"  [S{n}] {h['arxiv_id']} ({h['year']}) {loc} "
              f"{score_str}{ref_flag} - {h['title'][:50]}")
        if args.show_sources:
            print(f"        {' '.join(h['text'].split())[:200]}...")


if __name__ == "__main__":
    main()
