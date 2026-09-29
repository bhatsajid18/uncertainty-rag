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
import unicodedata
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from llm import add_provider_args, exit_on_rate_limit

DEFAULT_RERANKER = "BAAI/bge-reranker-base"  # 278M, 512-token max: matches our chunks
MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
RRF_K = 60  # standard constant from the RRF paper; dampens top-rank dominance
# Score multiplier for a chunk tagged is_reference. Bibliography chunks used to
# be dropped outright, which was a silent, unrecoverable error whenever the
# tagger was wrong: body prose that ranked #3 of 408 on dense, BM25 and hybrid
# alike could never be returned, and the model answered that the sources did
# not cover the question. Since no cheap feature separates a reference list
# from citation-dense prose (see chunk_papers.REFERENCE_DENSITY_THRESHOLD),
# the tag is treated as a penalty instead: a real bibliography chunk, which
# scores near zero for a content question anyway, stays buried, while a
# mis-tagged chunk that genuinely answers the question can still surface.
#
# Calibrated on the cross-encoder's scores over the top-30 hybrid candidates
# for five content questions: chunks tagged as bibliography score a median of
# 0.18 against 0.52 for content chunks, so genuine reference lists are already
# well separated and need only a nudge. The tagged chunks that DO score highly
# are overwhelmingly the mis-tagged ones - the very chunks this penalty must
# not bury. Hence a mild 0.75: it leaves a real bibliography entry at ~0.13,
# far below the content median, while a mis-tagged chunk scoring 0.9998 stays
# at 0.75 and can still be returned. 0.5 was tried first and was too harsh -
# it put that chunk below the median content chunk, which is the old hard
# filter in slow motion. run_eval.py sweeps 0.0/0.25/0.5 against this default.
REF_PENALTY = 0.75


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Names that papers write differently from how people ask for them. BM25 only
# matches exact tokens, so without this "CIFAR10" in a question never meets
# "C10" in a results table, and "EnDD" never meets the paper's "EnD2". Kept
# small and explicit on purpose: a hidden, sprawling synonym list is hard to
# reason about. Applied identically to documents and queries.
TOKEN_ALIASES = {
    "c10": "cifar10",
    "c100": "cifar100",
    "endd": "end2",       # Ensemble Distribution Distillation, written EnD²
}
_HYPHENATED_NAME = re.compile(r"\b([a-z]+)-(\d+)\b")


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, with a few normalisations.

    BM25's value here is exact term matching (e.g. 'fpr95'), so tokenization
    stays simple and transparent. Three fixes make "the same name" match:
      - NFKC folds superscripts and ligatures (EnD² -> end2)
      - hyphenated names join (CIFAR-10 -> cifar10, ResNet-18 -> resnet18),
        so they match the unhyphenated spelling people type
      - TOKEN_ALIASES maps paper-specific abbreviations to a canonical form
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _HYPHENATED_NAME.sub(r"\1\2", text)
    return [TOKEN_ALIASES.get(t, t) for t in re.findall(r"[a-z0-9]+", text)]


def index_text(meta: dict, contextual: bool = True) -> str:
    """The text a chunk is indexed and reranked by: its paper title, then its text.

    A chunk from the middle of a results table often never names its paper,
    so a question like "what error does Ensemble Distribution Distillation get
    on CIFAR-10?" has nothing to match. Prefixing the title (a lightweight form
    of "contextual retrieval") fixes that. Only the INDEXED text changes; the
    stored chunk text - and so every evaluation label, which is pinned to a
    hash of that text - stays the same.

    Title and table note come first, so if the result exceeds the models'
    512-token limit it is the tail of the chunk text that gets cut.
    """
    parts = []
    if contextual and meta.get("title"):
        parts.append(meta["title"])
    # An LLM description of the chunk's table(s), from table_notes.py. Present
    # only in indexes built after running it; see that module for why.
    if meta.get("table_note"):
        parts.append(meta["table_note"])
    # figure captions and descriptions (figures.py): a chunk's prose rarely
    # repeats what a figure shows
    if meta.get("figure_note"):
        parts.append(meta["figure_note"])
    parts.append(meta["text"])
    return "\n".join(parts)


_CHUNK_ID = re.compile(r"^(.*)__(\d{4})$")
_TABLE_TOKEN = re.compile(r"^(?:[±+\-]?\d+(?:\.\d+)?%?|±|NA|N/A|-|–)$")


def _table_like(tokens: list[str], threshold: float = 0.5) -> bool:
    """Mostly numbers, ± signs and NA - i.e. the inside of a results table."""
    return bool(tokens) and \
        sum(bool(_TABLE_TOKEN.match(t)) for t in tokens) / len(tokens) >= threshold


def _merge_overlapping(first: str, second: str, min_overlap: int = 20) -> str:
    """Join two consecutive chunks without repeating their overlap region.

    Consecutive chunks are slices of the same page text with a token overlap,
    so the end of `first` equals the start of `second`. Find the longest such
    suffix/prefix match; if there is none (e.g. whitespace was normalised
    differently), fall back to plain concatenation.
    """
    for n in range(min(len(first), len(second)), min_overlap - 1, -1):
        if first.endswith(second[:n]):
            return first + second[n:]
    return first + " " + second


def expand_split_tables(hits: list[dict], chunk_by_id, max_expand: int = 2,
                        window: int = 30) -> list[dict]:
    """Attach the neighbouring chunk when a hit starts or ends mid-table.

    Fixed-size chunking can cut a results table in two. Retrieval often finds
    the half with prose around it (a caption, "results in Table 3 ...") while
    the rows the question needs sit in the neighbouring chunk, which is mostly
    numbers and so ranks poorly for both BM25 and embeddings, and which a
    cross-encoder scores as irrelevant. Observed: for "EnD2's CIFAR-10 error",
    the chunk with the C10 rows ranked #36 (BM25) / #62 (dense) with reranker
    score 0.0012, while the table's other half ranked #1 at 0.99.

    Rather than hope the right half is retrieved, a hit whose first or last
    `window` tokens look like table cells gets its previous or next chunk
    merged in, so the model sees the whole table. Capped at `max_expand` hits
    to keep the prompt inside the free tier's tokens-per-minute budget.

    Retrieval metrics are unaffected: this runs after ranking, when assembling
    the context the model reads. chunk_by_id(chunk_id) -> meta dict or None.
    """
    have = {h["chunk_id"] for h in hits}
    out, expanded = [], 0
    for h in hits:
        m = _CHUNK_ID.match(h["chunk_id"])
        tokens = h["text"].split()
        if expanded >= max_expand or not m or len(tokens) < window:
            out.append(h)
            continue
        prefix, idx = m.group(1), int(m.group(2))
        prev_id, next_id = f"{prefix}__{idx - 1:04d}", f"{prefix}__{idx + 1:04d}"
        prev = chunk_by_id(prev_id) if idx > 0 and _table_like(tokens[:window]) \
            and prev_id not in have else None
        nxt = chunk_by_id(next_id) if _table_like(tokens[-window:]) \
            and next_id not in have else None
        if not prev and not nxt:
            out.append(h)
            continue
        merged = dict(h)
        text, added = h["text"], []
        if prev:
            text = _merge_overlapping(prev["text"], text)
            merged["page_start"] = min(h["page_start"], prev["page_start"])
            added.append(prev_id)
            have.add(prev_id)
        if nxt:
            text = _merge_overlapping(text, nxt["text"])
            merged["page_end"] = max(h["page_end"], nxt["page_end"])
            added.append(next_id)
            have.add(next_id)
        merged["text"] = text
        merged["expanded_with"] = added
        # keep the neighbours' rebuilt tables (table_notes.py) too
        grids = []
        for part in (prev, h, nxt):
            for g in (part or {}).get("table_markdown", "").split("\n\n"):
                if g.strip() and g not in grids:
                    grids.append(g)
        if grids:
            merged["table_markdown"] = "\n\n".join(grids)
        out.append(merged)
        expanded += 1
    return out


def demote_refs(ranked: list[tuple[int, float]], meta: list[dict],
                penalty: float) -> list[tuple[int, float]]:
    """Shrink bibliography chunks' scores toward the weakest candidate's.

    Not a plain multiplication, because the score scales are not comparable.
    Reranker scores spread over most of 0-1, so halving one is the moderate
    demotion intended. RRF scores sit in a narrow band near 1/RRF_K - the gap
    between rank 1 and rank 20 is under 25% - so halving one drops it below
    every other candidate, which is the hard filter again by accident.

    Interpolating toward the minimum makes the penalty proportional to the
    spread the mode actually produces: it stays a halving when scores reach
    down to zero (the reranker), and becomes proportionally gentler when they
    are bunched (RRF). Re-sorted, since demoting changes the order.
    """
    if penalty >= 1.0 or not ranked:
        return ranked
    lo = min(s for _, s in ranked)
    out = [(i, lo + (s - lo) * penalty if meta[i]["is_reference"] else s)
           for i, s in ranked]
    out.sort(key=lambda x: x[1], reverse=True)
    return out


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
    hard_cap: bool = False,
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

    hard_cap makes the cap binding: the backfill will return fewer than k
    results rather than exceed max_per_paper. Off by default, because the
    looser behaviour is what every result so far was measured with - turn it
    on through the evaluation's rerank_hardcap* configs to compare the two.
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
    # Two separate reserves, because they are held back for different reasons
    # and a hard cap must be able to draw on one without the other. Merging
    # them (the original behaviour) meant the backfill below silently
    # reinstated the very chunks the cap had just excluded: measured over an
    # 87-query evaluation, 46 of 68 answerable queries came back with MORE
    # than max_per_paper chunks from a single paper, so the cap was close to
    # inert and the max_per_paper ablations were comparing almost nothing.
    over_cap: list[tuple[int, float]] = []
    below_floor: list[tuple[int, float]] = []

    for idx, score in ranked:
        if len(selected) >= k:
            break
        paper = meta[idx]["arxiv_id"]

        if per_paper.get(paper, 0) >= max_per_paper:
            over_cap.append((idx, score))
            continue

        # A chunk is admitted if it clears the relevance floor. Chunks below
        # the floor are held back so they only appear if nothing better exists.
        if floor is not None and score < floor:
            below_floor.append((idx, score))
            continue

        selected.append((idx, score))
        per_paper[paper] = per_paper.get(paper, 0) + 1

    # Backfill if the cap/threshold left us short, strongest first. With
    # hard_cap the over-cap reserve is not touched, so the result genuinely
    # honours max_per_paper and simply returns fewer than k when the corpus
    # cannot offer k chunks from enough different papers.
    reserve = below_floor if hard_cap else below_floor + over_cap
    if len(selected) < k:
        chosen_ids = {i for i, _ in selected}
        for idx, score in sorted(reserve, key=lambda x: x[1], reverse=True):
            if len(selected) >= k:
                break
            if idx in chosen_ids:
                continue
            if hard_cap:
                paper = meta[idx]["arxiv_id"]
                if per_paper.get(paper, 0) >= max_per_paper:
                    continue
                per_paper[paper] = per_paper.get(paper, 0) + 1
            selected.append((idx, score))
            chosen_ids.add(idx)

    # Never return an out-of-order ranking.
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[:k]



def mmr_select(
    ranked: list[tuple[int, float]],
    vectors: np.ndarray,
    k: int,
    lam: float = 0.7,
) -> list[tuple[int, float]]:
    """Maximal Marginal Relevance: pick k results that are relevant AND unlike
    the ones already picked.

        next = argmax  lam * relevance(d) - (1 - lam) * max_sim(d, picked)

    relevance is the ranker's score min-max scaled to 0-1 over the candidates
    (so it works for reranker, cosine or RRF scores alike); similarity is the
    cosine between chunk embeddings. lam = 1 is plain top-k, lam = 0 is pure
    diversity. Unlike the per-paper cap, which diversifies by SOURCE, MMR
    diversifies by CONTENT: it also stops two near-identical chunks from the
    same paper (e.g. the overlap region of consecutive chunks) filling two of
    the k slots. `vectors[j]` must be the normalised embedding of ranked[j].

    Returns (index, original score) pairs in selection order.
    """
    if not ranked or k <= 0:
        return []
    scores = np.array([s for _, s in ranked], dtype="float64")
    lo, hi = scores.min(), scores.max()
    rel = (scores - lo) / (hi - lo) if hi - lo > 1e-12 else np.ones_like(scores)
    sims = vectors @ vectors.T
    chosen: list[int] = []
    remaining = list(range(len(ranked)))
    while remaining and len(chosen) < k:
        if chosen:
            penalty = sims[np.ix_(remaining, chosen)].max(axis=1)
        else:
            penalty = np.zeros(len(remaining))
        mmr = lam * rel[remaining] - (1 - lam) * penalty
        best = remaining[int(np.argmax(mmr))]
        chosen.append(best)
        remaining.remove(best)
    return [ranked[j] for j in chosen]

# Fields every chunk carries in the index metadata (see build_index.py).
META_FIELDS = ("chunk_id", "arxiv_id", "title", "year", "page_start", "page_end",
               "is_reference", "text")
# Present only on chunks that have table notes (table_notes.py).
OPTIONAL_META_FIELDS = ("table_note", "table_markdown", "table_grid_source",
                        "figure_note", "figure_images")


def meta_row(chunk: dict) -> dict:
    row = {f: chunk[f] for f in META_FIELDS}
    row.update({f: chunk[f] for f in OPTIONAL_META_FIELDS if chunk.get(f)})
    return row
DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class HybridRetriever:
    """Dense + sparse retrieval with optional fusion and cross-encoder reranking.

    Two ways to build one:
      HybridRetriever(data_dir)             the persisted corpus index
      HybridRetriever.from_chunks(chunks)   an in-memory index over given chunks
                                            (e.g. uploaded PDFs), optionally
                                            merged with the persisted corpus
    """

    def __init__(self, data_dir: Path, reranker_model: str = DEFAULT_RERANKER,
                 load_reranker: bool = False):
        cfg = json.loads((data_dir / "index_config.json").read_text())
        self._setup(
            meta=json.loads((data_dir / "chunk_meta.json").read_text()),
            index=faiss.read_index(str(data_dir / "index.faiss")),
            embedder=SentenceTransformer(cfg["model"], device=pick_device()),
            embed_model_name=cfg["model"],
            query_prefix=cfg.get("query_prefix", ""),
            reranker_model=reranker_model,
            load_reranker=load_reranker,
            # indexes built before titles were indexed must keep matching how
            # their vectors were made
            contextual=cfg.get("contextual_header", False),
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: list[dict],
        base_dir: Path | None = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        query_prefix: str = BGE_QUERY_PREFIX,
        embedder=None,
        reranker_model: str = DEFAULT_RERANKER,
        load_reranker: bool = False,
        batch_size: int = 32,
        contextual: bool = True,
    ) -> "HybridRetriever":
        """In-memory retriever over `chunks`, alone or merged with a base corpus.

        With base_dir, the persisted corpus vectors are copied out of its FAISS
        index (no re-embedding) and the new chunks are embedded with the SAME
        model and query prefix the corpus was built with - mixing embedding
        models in one index would make their scores incomparable. Nothing is
        written to disk, so an uploaded document never touches the shared
        corpus.
        """
        self = cls.__new__(cls)
        meta, blocks = [], []
        if base_dir is not None:
            cfg = json.loads((base_dir / "index_config.json").read_text())
            embed_model = cfg["model"]
            query_prefix = cfg.get("query_prefix", "")
            contextual = cfg.get("contextual_header", False)  # match the corpus
            base = faiss.read_index(str(base_dir / "index.faiss"))
            meta = json.loads((base_dir / "chunk_meta.json").read_text())
            blocks.append(base.reconstruct_n(0, base.ntotal))

        if embedder is None:
            embedder = SentenceTransformer(embed_model, device=pick_device())
        if chunks:
            vecs = embedder.encode(
                [index_text(c, contextual) for c in chunks], batch_size=batch_size,
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=len(chunks) > 64,
            )
            blocks.append(np.asarray(vecs, dtype="float32"))
            meta = meta + [meta_row(c) for c in chunks]
        if not blocks:
            raise ValueError("nothing to index: no chunks and no base corpus")

        matrix = np.ascontiguousarray(np.vstack(blocks), dtype="float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self._setup(meta, index, embedder, embed_model, query_prefix,
                    reranker_model, load_reranker, contextual)
        return self

    def _setup(self, meta, index, embedder, embed_model_name, query_prefix,
               reranker_model, load_reranker, contextual=False):
        if index.ntotal != len(meta):
            raise ValueError(f"index has {index.ntotal} vectors but {len(meta)} "
                             "metadata rows; they must align one-to-one")
        self.meta = meta
        self.index = index
        self.embedder = embedder
        self.embed_model_name = embed_model_name
        self.query_prefix = query_prefix
        self.contextual = contextual
        # sparse side: BM25 over the same chunk order as the dense index
        self.bm25 = BM25Okapi([tokenize_for_bm25(index_text(m, contextual))
                               for m in meta])
        # reranker is lazy-loaded: only needed for hybrid_rerank
        self.reranker_model_name = reranker_model
        self._reranker = None
        if load_reranker:
            self._load_reranker()

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model_name,
                                          device=pick_device())
        return self._reranker

    # --- individual strategies -------------------------------------------

    def _cache(self, name: str) -> dict:
        """Per-instance memo dicts. Evaluation re-runs the same query at several
        cutoffs and configurations; caching query embeddings and reranker
        scores makes those repeats nearly free without changing any result."""
        return self.__dict__.setdefault(name, {})

    def _dense_rank(self, query: str, top_n: int) -> list[tuple[int, float]]:
        cache = self._cache("_qemb_cache")
        if query not in cache:
            cache[query] = self.embedder.encode(
                [self.query_prefix + query],
                convert_to_numpy=True, normalize_embeddings=True,
            ).astype("float32")
        scores, idxs = self.index.search(cache[query], min(top_n, len(self.meta)))
        return [(int(i), float(s)) for s, i in zip(scores[0], idxs[0]) if i >= 0]

    def _bm25_rank(self, query: str, top_n: int) -> list[tuple[int, float]]:
        q_tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:top_n]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    def _first_stage(self, query: str, mode: str, fetch: int,
                     fusion: str, alpha: float) -> list[tuple[int, float]]:
        """Ranked (chunk_index, score) list for ONE query string."""
        if mode == "dense":
            return self._dense_rank(query, fetch)
        if mode == "bm25":
            return self._bm25_rank(query, fetch)
        # hybrid or hybrid_rerank
        dense = self._dense_rank(query, fetch)
        sparse = self._bm25_rank(query, fetch)
        if fusion == "rrf":
            fused = rrf_fuse([[i for i, _ in dense], [i for i, _ in sparse]])
        else:
            fused = weighted_fuse(dict(dense), dict(sparse), alpha=alpha)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

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
        query_variants: list[str] | None = None,
        mmr_lambda: float | None = None,
        min_score: float | None = None,
        ref_penalty: float = REF_PENALTY,
        hard_cap: bool = False,
    ) -> list[dict]:
        """Retrieve top-k chunks using the chosen strategy.

        candidate_n controls how wide the first stage casts its net before
        fusion/reranking narrows it to k.

        query_variants enables multi-query retrieval: the first stage runs once
        per variant and the ranked lists are fused with RRF. The reranker, if
        used, still scores against `query` (the original question) so that
        rephrasings widen the candidate pool without redefining relevance.

        mmr_lambda switches the final selection from the per-paper cap to
        Maximal Marginal Relevance (see mmr_select); max_per_paper and
        min_score_frac are then ignored.

        hard_cap makes max_per_paper binding, returning fewer than k results
        rather than exceeding it (see apply_diversity_cap).

        ref_penalty multiplies the score of chunks tagged is_reference, so a
        bibliography chunk has to be markedly better than the alternatives to
        be returned. include_refs=True switches the penalty off entirely;
        ref_penalty=0 restores the old behaviour of excluding them.

        min_score drops final results scoring below it. It is meant for the
        reranker's 0-1 scores: if nothing clears it, search returns [] and
        callers answer with the refusal without calling the LLM at all. Other
        modes' scores live on other scales (RRF ~0.03, BM25 unbounded), so
        pick the value per mode - the evaluation's abstention analysis reports
        the best threshold.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        # over-fetch so reference filtering can't starve the final top-k
        fetch = candidate_n if include_refs else candidate_n * 3

        variants = [v for v in (query_variants or []) if v.strip()]
        if len(variants) > 1:
            per_variant = [
                self._first_stage(v, mode, fetch, fusion, alpha) for v in variants
            ]
            fused = rrf_fuse([[i for i, _ in lst] for lst in per_variant])
            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked = self._first_stage(query, mode, fetch, fusion, alpha)

        # Bibliography chunks stay in the candidate pool and are demoted at
        # selection time (see demote_refs); ref_penalty=0 restores the old
        # behaviour of excluding them outright. The demotion deliberately
        # happens AFTER reranking rather than here, because in hybrid_rerank
        # the cross-encoder replaces this score, which would discard it.
        penalty = 1.0 if include_refs else ref_penalty
        candidates = []
        for idx, score in ranked:
            if penalty <= 0 and self.meta[idx]["is_reference"]:
                continue
            candidates.append((idx, score))
            if len(candidates) >= candidate_n:
                break

        if not candidates:
            return []

        # optional second stage: cross-encoder reranking
        if mode == "hybrid_rerank":
            cache = self._cache("_rerank_cache")
            todo = [i for i, _ in candidates if (query, i) not in cache]
            if todo:
                ctx = getattr(self, "contextual", False)
                scores = self._load_reranker().predict(
                    [(query, index_text(self.meta[i], ctx)) for i in todo])
                cache.update({(query, i): float(sc) for i, sc in zip(todo, scores)})
            rerank_scores = np.array([cache[(query, i)] for i, _ in candidates])
            order = np.argsort(rerank_scores)[::-1]
            # rank by reranker score, then enforce per-paper diversity
            reranked = [(candidates[j][0], float(rerank_scores[j])) for j in order]
            reranked = demote_refs(reranked, self.meta, penalty)
            fusion_by_idx = {i: s for i, s in candidates}
            capped = self._select(reranked, k, max_per_paper, min_score_frac,
                                  mmr_lambda, min_score, hard_cap)
            return [
                {
                    "score": rs,
                    "first_stage_score": fusion_by_idx.get(i, 0.0),
                    "mode": mode,
                    **self.meta[i],
                }
                for i, rs in capped
            ]

        capped = self._select(demote_refs(candidates, self.meta, penalty), k,
                              max_per_paper, min_score_frac, mmr_lambda, min_score,
                              hard_cap)
        return [{"score": s, "mode": mode, **self.meta[i]} for i, s in capped]

    def _select(self, ranked, k, max_per_paper, min_score_frac, mmr_lambda,
                min_score, hard_cap=False):
        """Final top-k: the per-paper cap or MMR, then the absolute floor."""
        if min_score is not None:
            ranked = [(i, s) for i, s in ranked if s >= min_score]
        if mmr_lambda is not None:
            vectors = np.vstack([self.index.reconstruct(int(i)) for i, _ in ranked]) \
                if ranked else np.zeros((0, 1), dtype="float32")
            return mmr_select(ranked, vectors, k, mmr_lambda)
        return apply_diversity_cap(ranked, self.meta, k, max_per_paper,
                                   min_score_frac, hard_cap)

    def explain(self, query: str, chunk_id: str, candidate_n: int = 20,
                fusion: str = "rrf", alpha: float = 0.5,
                query_variants: list[str] | None = None) -> dict:
        """Where does one chunk land at each retrieval stage for this query?

        Answers "why wasn't the chunk I know is relevant retrieved?" with data:
        its rank under dense, BM25 and fused search over the whole corpus,
        whether that got it into the reranker's candidate pool, and its reranker
        score if a reranker is loaded.
        """
        idx = next((i for i, m in enumerate(self.meta) if m["chunk_id"] == chunk_id), None)
        if idx is None:
            raise KeyError(f"{chunk_id} is not in the index")
        n = len(self.meta)

        def rank_in(ranked):
            order = [i for i, _ in ranked]
            return order.index(idx) + 1 if idx in order else None

        report = {"chunk_id": chunk_id, "n_chunks": n,
                  "is_reference": self.meta[idx]["is_reference"],
                  "dense": rank_in(self._dense_rank(query, n)),
                  "bm25": rank_in(self._bm25_rank(query, n))}
        fused = self._first_stage(query, "hybrid", n, fusion, alpha)
        variants = [v for v in (query_variants or []) if v.strip()]
        if len(variants) > 1:
            per = [self._first_stage(v, "hybrid", n, fusion, alpha) for v in variants]
            report["hybrid_single_query"] = rank_in(fused)
            fused = sorted(rrf_fuse([[i for i, _ in p] for p in per]).items(),
                           key=lambda x: x[1], reverse=True)
        report["hybrid"] = rank_in(fused)
        report["in_rerank_pool"] = (report["hybrid"] is not None
                                    and report["hybrid"] <= candidate_n)
        if self._reranker is not None:
            ctx = getattr(self, "contextual", False)
            report["rerank_score"] = float(self._reranker.predict(
                [(query, index_text(self.meta[idx], ctx))])[0])
        return report

    def chunk_by_id(self, chunk_id: str) -> dict | None:
        lookup = self.__dict__.get("_by_id")
        if lookup is None:
            lookup = self._by_id = {m["chunk_id"]: m for m in self.meta}
        return lookup.get(chunk_id)

    def search_ids(self, query: str, **kwargs) -> list[str]:
        """Like search(), but returns only chunk_ids in rank order.

        Used by the evaluation harness, which only needs the ranking.
        """
        return [h["chunk_id"] for h in self.search(query, **kwargs)]


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


@exit_on_rate_limit
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
    ap.add_argument("--include-refs", action="store_true",
                    help="Score bibliography chunks with no penalty at all.")
    ap.add_argument("--ref-penalty", type=float, default=REF_PENALTY,
                    help="Score multiplier for bibliography chunks "
                         f"(default {REF_PENALTY}; 0 excludes them entirely).")
    ap.add_argument("--max-per-paper", type=int, default=2,
                    help="Max chunks from any one paper in the final results "
                         "(0 disables the cap).")
    ap.add_argument("--min-score-frac", type=float, default=0.25,
                    help="A chunk admitted by the diversity cap must score at "
                         "least this fraction of the top chunk's score "
                         "(0 disables the threshold).")
    ap.add_argument("--hard-cap", action="store_true",
                    help="Make --max-per-paper binding: return fewer than k "
                         "results rather than exceed it.")
    ap.add_argument("--mmr", type=float, default=None, metavar="LAMBDA",
                    help="Select the final k with Maximal Marginal Relevance "
                         "(1 = relevance only, 0 = diversity only; e.g. 0.7). "
                         "Replaces the per-paper cap.")
    ap.add_argument("--min-score", type=float, default=None,
                    help="Drop results scoring below this (reranker scores are "
                         "0-1). Nothing left = no answer.")
    ap.add_argument("--compare", action="store_true",
                    help="Run all four modes on this query and print side by side.")
    ap.add_argument("--multi-query", type=int, default=0, metavar="N",
                    help="Generate N LLM rephrasings (needs GROQ_API_KEY) and "
                         "fuse retrieval across them (default 1; 0 = off, "
                         "which keeps retrieval fully offline).")
    add_provider_args(ap)
    ap.add_argument("--explain", metavar="CHUNK_ID",
                    help="Show where this chunk ranks at each stage for the query.")
    args = ap.parse_args()

    variants = None
    if args.multi_query > 0:
        from llm import make_complete_fn, resolve
        from query_expansion import QueryExpander

        provider, model, _ = resolve(args)
        expander = QueryExpander(
            make_complete_fn(model, provider=provider), model_name=model,
            n=args.multi_query,
            cache_path=args.data_dir / "query_expansions.json",
        )
        variants = expander.expand(args.query)

    needs_reranker = args.compare or args.mode == "hybrid_rerank"
    retr = HybridRetriever(args.data_dir, reranker_model=args.reranker,
                           load_reranker=needs_reranker)

    print(f'\nQuery: "{args.query}"')
    if variants and len(variants) > 1:
        print("Variants:")
        for v in variants[1:]:
            print(f"  - {v}")
    print("=" * 70)

    if args.explain:
        r = retr.explain(args.query, args.explain, candidate_n=args.candidate_n,
                         fusion=args.fusion, alpha=args.alpha, query_variants=variants)
        print(f"\nWhere {r['chunk_id']} ranks (out of {r['n_chunks']} chunks):")
        for stage in ("dense", "bm25", "hybrid_single_query", "hybrid"):
            if stage in r:
                val = f"#{r[stage]}" if r[stage] else "not matched"
                label = {"hybrid": "hybrid (fused)" + (" with rephrasings" if
                         "hybrid_single_query" in r else ""),
                         "hybrid_single_query": "hybrid, original query only"}.get(stage, stage)
                print(f"  {label:36s} {val}")
        print(f"  {'reaches the reranker (top ' + str(args.candidate_n) + ')':36s} "
              f"{'yes' if r['in_rerank_pool'] else 'NO - so it cannot be returned'}")
        if "rerank_score" in r:
            print(f"  {'reranker score':36s} {r['rerank_score']:.4f}")
        if r["is_reference"]:
            print(f"  note: tagged as bibliography, so its score is multiplied "
                  f"by {args.ref_penalty} (--include-refs to score it in full, "
                  "--ref-penalty 0 to exclude it)")
        print()

    modes = MODES if args.compare else (args.mode,)
    for mode in modes:
        hits = retr.search(
            args.query, k=args.k, mode=mode, candidate_n=args.candidate_n,
            include_refs=args.include_refs, fusion=args.fusion, alpha=args.alpha,
            max_per_paper=args.max_per_paper,
            min_score_frac=args.min_score_frac,
            query_variants=variants,
            mmr_lambda=args.mmr,
            min_score=args.min_score,
            ref_penalty=args.ref_penalty,
            hard_cap=args.hard_cap,
        )
        _print_hits(mode + ("+mq" if variants else "")
                    + (f"+mmr{args.mmr}" if args.mmr is not None else ""), hits)


if __name__ == "__main__":
    main()
