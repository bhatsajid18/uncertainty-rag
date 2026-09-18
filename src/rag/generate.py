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

Because --mode is a flag, the same question can be answered from different
retrieval strategies, which separates two distinct questions: does better
retrieval rank the right chunks higher, and does it actually produce better
ANSWERS? Iteration 3 measures both.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# reuse the multi-strategy retriever from retrieve.py - no retrieval logic
# is duplicated here
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import DEFAULT_RERANKER, MODES, HybridRetriever  # noqa: E402

# Groq deprecated llama-3.3-70b-versatile on the free tier (June 2026).
# gpt-oss-120b is the recommended free replacement. Override with --model.
DEFAULT_LLM = "openai/gpt-oss-120b"
MAX_RETRIES = 5

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
    "sources support."
)


def source_tag(hit: dict, n: int) -> str:
    """Human-readable citation label for a retrieved chunk."""
    if hit["page_start"] == hit["page_end"]:
        loc = f"p.{hit['page_start']}"
    else:
        loc = f"pp.{hit['page_start']}-{hit['page_end']}"
    # first author-ish label from title is noisy; use arxiv id + year for stability
    return f"S{n}: {hit['arxiv_id']} ({hit['year']}), {loc}"


def build_user_prompt(question: str, hits: list[dict]) -> str:
    lines = ["Sources:\n"]
    for n, h in enumerate(hits, 1):
        text = " ".join(h["text"].split())
        lines.append(f"[{source_tag(h, n)}]\n{text}\n")
    lines.append(f"\nQuestion: {question}")
    return "\n".join(lines)


def _retry_after_seconds(err, attempt: int) -> float:
    """Honor the server's retry-after header if present, else exponential backoff.

    Groq free tier can block all requests for ~60s after a 429, and only sets
    retry-after when the limit is actually hit, so we prefer the header and fall
    back to a backoff starting at a safe minimum.
    """
    headers = getattr(err, "response", None)
    retry_after = None
    if headers is not None:
        try:
            retry_after = err.response.headers.get("retry-after")
        except Exception:  # noqa: BLE001
            retry_after = None
    if retry_after is not None:
        try:
            return float(retry_after) + 1.0  # small cushion
        except ValueError:
            pass
    # exponential backoff, min 2s, capped
    return min(60.0, max(2.0, 2 ** attempt * 2))


def call_groq(client, model, system_prompt, user_prompt):
    """Call Groq, honoring rate-limit retry-after with exponential fallback."""
    from groq import APIConnectionError, RateLimitError

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,  # deterministic, factual
            )
            return resp.choices[0].message.content
        except RateLimitError as e:
            if attempt < MAX_RETRIES - 1:
                wait = _retry_after_seconds(e, attempt)
                print(f"  (rate limited, waiting {wait:.0f}s then retrying ...)",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except APIConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                wait = min(30.0, 2 ** attempt * 2)
                print(f"  (connection error, retrying in {wait:.0f}s ...)",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("exhausted retries calling Groq")


def answer(
    question: str,
    data_dir: Path,
    k: int = 5,
    model: str = DEFAULT_LLM,
    include_refs: bool = False,
    mode: str = "hybrid_rerank",
    candidate_n: int = 20,
    fusion: str = "rrf",
    alpha: float = 0.5,
    max_per_paper: int = 2,
    min_score_frac: float = 0.25,
    reranker: str = DEFAULT_RERANKER,
):
    """Retrieve, then generate a grounded answer. Returns (answer_text, hits)."""
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found. Put it in .env or export it.", file=sys.stderr)
        sys.exit(1)

    from groq import Groq
    client = Groq(api_key=api_key)

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
    )
    if not hits:
        return "No chunks retrieved (is the index built?).", []

    user_prompt = build_user_prompt(question, hits)
    text = call_groq(client, model, SYSTEM_PROMPT, user_prompt)
    return text, hits


def main():
    ap = argparse.ArgumentParser(description="Ask a grounded, cited question.")
    ap.add_argument("question", type=str)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("-k", type=int, default=5,
                    help="Number of sources handed to the model.")
    ap.add_argument("--model", default=DEFAULT_LLM,
                    help="Groq model id (catalogs change; see README).")
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
                     help="Allow bibliography chunks as sources.")

    args = ap.parse_args()

    text, hits = answer(
        args.question,
        args.data_dir,
        k=args.k,
        model=args.model,
        include_refs=args.include_refs,
        mode=args.mode,
        candidate_n=args.candidate_n,
        fusion=args.fusion,
        alpha=args.alpha,
        max_per_paper=args.max_per_paper,
        min_score_frac=args.min_score_frac,
        reranker=args.reranker,
    )

    print(f'\nQuestion: {args.question}')
    print(f"Retrieval: mode={args.mode}, k={args.k}, "
          f"max_per_paper={args.max_per_paper}, "
          f"min_score_frac={args.min_score_frac}")
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