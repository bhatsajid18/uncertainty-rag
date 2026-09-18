"""
Multi-strategy retrieval: dense, sparse (BM25), hybrid fusion, and reranking.

Provides one HybridRetriever that supports four selectable configurations, so
the same query set can be run through each and compared in evaluation:

  dense          FAISS dense vector search only (bge-base embeddings)
  bm25           BM25 sparse keyword search only
  hybrid         Reciprocal Rank Fusion (RRF) over dense + BM25 rankings
  hybrid_rerank  hybrid, then cross-encoder reranking of the top candidates

Why RRF for fusion: dense cosine scores (~0-1) and BM25 scores (unbounded)
live on different scales, so naively adding them is meaningless. RRF combines
by RANK POSITION instead of raw score, which sidesteps normalization entirely:

    RRF(d) = sum over rankers r of  1 / (k + rank_r(d))

A weighted normalized-score fusion is also provided for comparison.

Why two-stage (retrieve wide, then rerank): the bi-encoder compares
pre-computed vectors, which is fast but approximate. A cross-encoder scores
(query, chunk) pairs jointly and is far more accurate, but far too slow to run
over the whole corpus. So we retrieve a wide net cheaply, then rerank only
those candidates.

The BM25 index is built in memory at load time. At this corpus size (hundreds
to low thousands of chunks) tokenization takes well under a second; if the
corpus grew to tens of thousands of chunks it would be worth persisting.

Usage:
  python retrieve.py "your query" --mode dense
  python retrieve.py "your query" --mode bm25
  python retrieve.py "your query" --mode hybrid
  python retrieve.py "your query" --mode hybrid_rerank -k 5
  python retrieve.py "your query" --compare        # run all four side by side
"""

import argparse
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DEFAULT_RERANKER = "BAAI/bge-reranker-base"  # 278M, 512-token max: matches our chunks
MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
RRF_K = 60  # standard constant from the RRF paper; dampens top-rank dominance


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tokenize_for_bm25(text: str) -> list[str]:
    """Simple lowercase alphanumeric tokenizer.

    Kept deliberately simple and transparent: BM25's strength here is exact
    term matching (e.g. 'fpr95', 'cifar-100'), so we lowercase and split on
    non-alphanumerics while keeping digits attached to words.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion over several ranked lists of document indices."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return scores


def weighted_fuse(
    dense_scores: dict[int, float],
    sparse_scores: dict[int, float],
    alpha: float = 0.5,
) -> dict[int, float]:
    """Min-max normalize each score set, then blend: alpha*dense + (1-alpha)*sparse."""

    def norm(d: dict[int, float]) -> dict[int, float]:
        if not d:
            return {}
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return {i: 1.0 for i in d}
        return {i: (v - lo) / (hi - lo) for i, v in d.items()}

    dn, sn = norm(dense_scores), norm(sparse_scores)
    out: dict[int, float] = {}
    for i in set(dn) | set(sn):
        out[i] = alpha * dn.get(i, 0.0) + (1 - alpha) * sn.get(i, 0.0)
    return out


def apply_diversity_cap(
    ranked: list[tuple[int, float]],
    meta: list[dict],
    k: int,
    max_per_paper: int,
    min_score_frac: float = 0.25,
) -> list[tuple[int, float]]:
    """Take the top-k, allowing at most max_per_paper chunks from any one paper.

    Cross-encoder rerankers in particular tend to collapse onto a single
    strongly-matching document, returning several chunks from the same paper.
    For research Q&A we usually want the answer synthesized across the
    literature, so we cap each source's contribution. This diversifies by
    SOURCE rather than by embedding distance (the axis that matters here),
    making it a simple deterministic stand-in for MMR.

    Two guards keep the cap from hurting more than it helps:

    1. min_score_frac - a chunk promoted into the results purely by diversity
       must score at least this fraction of the TOP chunk's score. Without it,
       a query that only one paper genuinely answers gets padded with chunks
       the ranker considered nearly irrelevant. The threshold is RELATIVE, not
       absolute, because score scales differ wildly by mode: reranker ~0-1,
       cosine ~0-1, BM25 unbounded, RRF ~0.03.

    2. Results are re-sorted by score before returning, so enforcing the cap
       can never produce an out-of-order ranking.

    max_per_paper <= 0 disables the cap; min_score_frac <= 0 disables the
    threshold.
    """
    if max_per_paper <= 0:
        return ranked[:k]
    if not ranked:
        return []

    top_score = ranked[0][1]
    floor = (
        top_score * min_score_frac
        if (min_score_frac > 0 and top_score > 0)
        else None
    )

    selected: list[tuple[int, float]] = []
    per_paper: dict[str, int] = {}
    deferred: list[tuple[int, float]] = []  # hit the cap, may be needed to fill

    for idx, score in ranked:
        if len(selected) >= k:
            break
        paper = meta[idx]["arxiv_id"]

        if per_paper.get(paper, 0) >= max_per_paper:
            deferred.append((idx, score))   # over cap: hold in reserve
            continue

        # A chunk is admitted if it clears the relevance floor. Chunks below
        # the floor are held back so they only appear if nothing better exists.
        if floor is not None and score < floor:
            deferred.append((idx, score))
            continue

        selected.append((idx, score))
        per_paper[paper] = per_paper.get(paper, 0) + 1

    # Backfill from the reserve if the cap/threshold left us short. The reserve
    # is taken in score order so we always add the strongest available chunk.
    if len(selected) < k:
        chosen_ids = {i for i, _ in selected}
        for idx, score in sorted(deferred, key=lambda x: x[1], reverse=True):
            if len(selected) >= k:
                break
            if idx in chosen_ids:
                continue
            selected.append((idx, score))
            chosen_ids.add(idx)

    # Never return an out-of-order ranking.
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[:k]


class HybridRetriever:
    """Dense + sparse retrieval with optional fusion and cross-encoder reranking."""

    def __init__(self, data_dir: Path, reranker_model: str = DEFAULT_RERANKER,
                 load_reranker: bool = False):
        cfg = json.loads((data_dir / "index_config.json").read_text())
        self.embed_model_name = cfg["model"]
        self.query_prefix = cfg.get("query_prefix", "")
        self.device = pick_device()

        # dense side
        self.index = faiss.read_index(str(data_dir / "index.faiss"))
        self.meta = json.loads((data_dir / "chunk_meta.json").read_text())
        self.embedder = SentenceTransformer(self.embed_model_name, device=self.device)

        # sparse side: build BM25 over the same chunk order as the dense index
        corpus_tokens = [tokenize_for_bm25(m["text"]) for m in self.meta]
        self.bm25 = BM25Okapi(corpus_tokens)

        # reranker is lazy-loaded: only needed for hybrid_rerank
        self.reranker_model_name = reranker_model
        self._reranker = None
        if load_reranker:
            self._load_reranker()

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model_name, device=self.device)
        return self._reranker

    # --- individual strategies -------------------------------------------

    def _dense_rank(self, query: str, top_n: int) -> list[tuple[int, float]]:
        q = self.query_prefix + query
        qemb = self.embedder.encode(
            [q], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, idxs = self.index.search(qemb, min(top_n, len(self.meta)))
        return [(int(i), float(s)) for s, i in zip(scores[0], idxs[0]) if i >= 0]

    def _bm25_rank(self, query: str, top_n: int) -> list[tuple[int, float]]:
        q_tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    # --- main entry point -------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid_rerank",
        candidate_n: int = 20,
        include_refs: bool = False,
        fusion: str = "rrf",
        alpha: float = 0.5,
        max_per_paper: int = 2,
        min_score_frac: float = 0.25,
    ) -> list[dict]:
        """Retrieve top-k chunks using the chosen strategy.

        candidate_n controls how wide the first stage casts its net before
        fusion/reranking narrows it to k.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        # over-fetch so reference filtering can't starve the final top-k
        fetch = candidate_n if include_refs else candidate_n * 3

        if mode == "dense":
            ranked = self._dense_rank(query, fetch)
        elif mode == "bm25":
            ranked = self._bm25_rank(query, fetch)
        else:  # hybrid or hybrid_rerank
            dense = self._dense_rank(query, fetch)
            sparse = self._bm25_rank(query, fetch)
            if fusion == "rrf":
                fused = rrf_fuse([[i for i, _ in dense], [i for i, _ in sparse]])
            else:
                fused = weighted_fuse(dict(dense), dict(sparse), alpha=alpha)
            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # filter references if requested, keep a candidate pool
        candidates = []
        for idx, score in ranked:
            m = self.meta[idx]
            if not include_refs and m["is_reference"]:
                continue
            candidates.append((idx, score))
            if len(candidates) >= candidate_n:
                break

        if not candidates:
            return []

        # optional second stage: cross-encoder reranking
        if mode == "hybrid_rerank":
            reranker = self._load_reranker()
            pairs = [(query, self.meta[i]["text"]) for i, _ in candidates]
            rerank_scores = reranker.predict(pairs)
            order = np.argsort(rerank_scores)[::-1]
            # rank by reranker score, then enforce per-paper diversity
            reranked = [(candidates[j][0], float(rerank_scores[j])) for j in order]
            fusion_by_idx = {i: s for i, s in candidates}
            capped = apply_diversity_cap(
                reranked, self.meta, k, max_per_paper, min_score_frac
            )
            return [
                {
                    "score": rs,
                    "first_stage_score": fusion_by_idx.get(i, 0.0),
                    "mode": mode,
                    **self.meta[i],
                }
                for i, rs in capped
            ]

        capped = apply_diversity_cap(
            candidates, self.meta, k, max_per_paper, min_score_frac
        )
        return [{"score": s, "mode": mode, **self.meta[i]} for i, s in capped]


def _print_hits(label: str, hits: list[dict]):
    print(f"\n--- {label} ---")
    if not hits:
        print("  (no results)")
        return
    for rank, h in enumerate(hits, 1):
        loc = (f"p.{h['page_start']}" if h["page_start"] == h["page_end"]
               else f"pp.{h['page_start']}-{h['page_end']}")
        extra = ""
        if "first_stage_score" in h:
            extra = f" (fusion={h['first_stage_score']:.4f})"
        print(f"  [{rank}] {h['score']:.4f}{extra}  {h['arxiv_id']} ({h['year']}) {loc}")
        print(f"      {h['title'][:65]}")
        print(f"      {' '.join(h['text'].split())[:160]}...")


def main():
    ap = argparse.ArgumentParser(description="Multi-strategy retrieval.")
    ap.add_argument("query", type=str)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    ap.add_argument("--candidate-n", type=int, default=20,
                    help="First-stage candidate pool size before rerank/cut.")
    ap.add_argument("--fusion", choices=("rrf", "weighted"), default="rrf")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Weight on dense when --fusion weighted (0=BM25 only).")
    ap.add_argument("--reranker", default=DEFAULT_RERANKER)
    ap.add_argument("--include-refs", action="store_true")
    ap.add_argument("--max-per-paper", type=int, default=2,
                    help="Max chunks from any one paper in the final results "
                         "(0 disables the cap).")
    ap.add_argument("--min-score-frac", type=float, default=0.25,
                    help="A chunk admitted by the diversity cap must score at "
                         "least this fraction of the top chunk's score "
                         "(0 disables the threshold).")
    ap.add_argument("--compare", action="store_true",
                    help="Run all four modes on this query and print side by side.")
    args = ap.parse_args()

    needs_reranker = args.compare or args.mode == "hybrid_rerank"
    retr = HybridRetriever(args.data_dir, reranker_model=args.reranker,
                           load_reranker=needs_reranker)

    print(f'\nQuery: "{args.query}"')
    print("=" * 70)

    modes = MODES if args.compare else (args.mode,)
    for mode in modes:
        hits = retr.search(
            args.query, k=args.k, mode=mode, candidate_n=args.candidate_n,
            include_refs=args.include_refs, fusion=args.fusion, alpha=args.alpha,
            max_per_paper=args.max_per_paper,
            min_score_frac=args.min_score_frac,
        )
        _print_hits(mode, hits)


if __name__ == "__main__":
    main()