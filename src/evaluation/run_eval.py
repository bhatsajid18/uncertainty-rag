"""
Retrieval evaluation sweep.

Runs the labelled query set (data/eval/queries.jsonl) through a suite of
retrieval configurations and writes machine-readable results, so the numbers
behind every claim can be regenerated and later read by a dashboard.

What it measures, per configuration:
  retrieval    Recall, Precision, NDCG and Success at k = 1, 3, 5, 10, plus MRR,
               overall and per query category. Unanswerable queries are excluded
               from these (recall is undefined without relevant chunks).
  significance paired bootstrap 95% confidence interval on the difference in
               Recall@5 and MRR versus the dense baseline. With ~50 queries a
               few points of difference can be noise; the CI says whether it is.
  abstention   whether the top-1 retrieval score separates answerable from
               unanswerable queries, reported as AUROC. This treats "is this
               question outside the corpus?" as an out-of-distribution
               detection problem - the same framing, and the same metric, as
               the uncertainty module.

Each cutoff k is a separate search, because the per-paper diversity cap means
the top 5 of a k=10 search is not the same list the system would return at
k=5. The retriever caches embeddings and reranker scores, so this is cheap.

Outputs, in results/eval/<run-name>/ (small CSV/JSON, meant to be committed):
  run_info.json               configs, query-set hash, corpus size, timestamp
  per_query.jsonl             every query x config: metrics + ranked chunk ids
  summary.csv                 config x metric (answerable queries)
  summary_by_category.csv     config x category x metric
  significance.csv            deltas vs baseline with bootstrap CIs
  abstention.csv              AUROC and best-threshold accuracy per config

Usage (from the repo root):
  python src/evaluation/run_eval.py                         # core suite
  python src/evaluation/run_eval.py --suite ablations --run-name ablations_v1
  python src/evaluation/run_eval.py --configs dense hybrid_rerank --limit 10
  python src/evaluation/run_eval.py --list                  # show all configs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "evaluation"))
from llm import add_provider_args, exit_on_rate_limit, resolve  # noqa: E402
from metrics import abstention_metrics, aggregate, evaluate_query  # noqa: E402

K_VALUES = (1, 3, 5, 10)
# Below this many paired queries a bootstrap CI is not meaningful (see
# paired_bootstrap), so the interval is withheld rather than printed.
MIN_QUERIES_FOR_CI = 8
BASELINE = "dense"
HEADLINE = ("recall@5", "mrr", "ndcg@5", "success@5")

# Shared defaults, matching the CLI defaults of retrieve.py / generate.py.
BASE = dict(candidate_n=20, fusion="rrf", alpha=0.5, max_per_paper=2,
            min_score_frac=0.25, include_refs=False, ref_penalty=0.75)


def cfg(mode, multi_query=0, **overrides):
    return {"mode": mode, "multi_query": multi_query, **BASE, **overrides}


CONFIGS = {
    # core: the retrieval strategies, and query expansion on the strongest one
    "dense": cfg("dense"),
    "bm25": cfg("bm25"),
    "hybrid": cfg("hybrid"),
    "hybrid_rerank": cfg("hybrid_rerank"),
    "hybrid_rerank+mq3": cfg("hybrid_rerank", multi_query=3),
    # ablations: one knob changed at a time relative to a core config
    "dense+mq3": cfg("dense", multi_query=3),
    "bm25+mq3": cfg("bm25", multi_query=3),
    "hybrid+mq3": cfg("hybrid", multi_query=3),
    "hybrid_weighted_a0.3": cfg("hybrid", fusion="weighted", alpha=0.3),
    "hybrid_weighted_a0.7": cfg("hybrid", fusion="weighted", alpha=0.7),
    "rerank_nocap": cfg("hybrid_rerank", max_per_paper=0),
    "rerank_cap1": cfg("hybrid_rerank", max_per_paper=1),
    "rerank_nofloor": cfg("hybrid_rerank", min_score_frac=0.0),
    # candidate_n sweep. cand10 led the first ablation run by +0.067 recall@5
    # over the default 20, but the paired CI straddled zero, so the sweep is
    # here to settle whether a narrower first stage really helps the
    # cross-encoder or whether that was one query's worth of luck.
    "rerank_cand8": cfg("hybrid_rerank", candidate_n=8),
    "rerank_cand10": cfg("hybrid_rerank", candidate_n=10),
    "rerank_cand12": cfg("hybrid_rerank", candidate_n=12),
    "rerank_cand15": cfg("hybrid_rerank", candidate_n=15),
    "rerank_cand40": cfg("hybrid_rerank", candidate_n=40),
    # Is the per-paper cap worth enforcing? Without hard_cap the backfill
    # reinstates over-cap chunks, so max_per_paper is close to advisory.
    "rerank_hardcap2": cfg("hybrid_rerank", hard_cap=True),
    "rerank_hardcap1": cfg("hybrid_rerank", max_per_paper=1, hard_cap=True),
    "rerank_refs": cfg("hybrid_rerank", include_refs=True),
    # bibliography handling: excluded outright (the pre-fix behaviour), lightly
    # demoted, or demoted hard. The filter used to be unconditional, which cost
    # recall whenever the citation-density tagger was wrong.
    "rerank_refs_excluded": cfg("hybrid_rerank", ref_penalty=0.0),
    "rerank_refs_demote0.25": cfg("hybrid_rerank", ref_penalty=0.25),
    "rerank_refs_demote0.5": cfg("hybrid_rerank", ref_penalty=0.5),
    "rerank_mmr0.7": cfg("hybrid_rerank", mmr_lambda=0.7),
    "rerank_mmr0.5": cfg("hybrid_rerank", mmr_lambda=0.5),
}

SUITES = {
    "core": ["dense", "bm25", "hybrid", "hybrid_rerank", "hybrid_rerank+mq3"],
    "ablations": ["dense", "dense+mq3", "bm25+mq3", "hybrid+mq3",
                  "hybrid_weighted_a0.3", "hybrid_weighted_a0.7", "hybrid_rerank",
                  "rerank_nocap", "rerank_cap1", "rerank_nofloor",
                  "rerank_cand8", "rerank_cand10", "rerank_cand12",
                  "rerank_cand15", "rerank_cand40",
                  "rerank_hardcap2", "rerank_hardcap1", "rerank_refs",
                  "rerank_refs_excluded", "rerank_refs_demote0.25",
                  "rerank_refs_demote0.5",
                  "rerank_mmr0.7", "rerank_mmr0.5"],
    "no_llm": ["dense", "bm25", "hybrid", "hybrid_rerank"],
}


# --- pure helpers (unit tested) ---------------------------------------------


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def validate_labels(queries: list[dict], chunks_by_id: dict[str, dict]):
    """Refuse to evaluate against a corpus that changed since labelling."""
    problems = []
    for q in queries:
        for rel in q["relevant"]:
            c = chunks_by_id.get(rel["chunk_id"])
            if c is None:
                problems.append(f"{q['qid']}: {rel['chunk_id']} no longer exists")
            elif text_hash(c["text"]) != rel["text_hash"]:
                problems.append(f"{q['qid']}: {rel['chunk_id']} text changed")
    if problems:
        raise SystemExit(
            "Labels no longer match data/chunks.jsonl (was it re-chunked?):\n  "
            + "\n  ".join(problems[:10])
            + "\nRe-chunk with the settings used for labelling, or relabel."
        )


def auroc(pos_scores: list[float], neg_scores: list[float]) -> float | None:
    """P(random positive scores higher than random negative), ties count half.

    Here positives are answerable queries and negatives unanswerable ones, so
    AUROC is how well the top-1 retrieval score detects out-of-corpus questions.
    """
    if not pos_scores or not neg_scores:
        return None
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos_scores) * len(neg_scores))


def best_threshold(scores: list[float], answerable: list[bool]) -> dict:
    """Threshold on top-1 score that maximises abstention accuracy.

    Chosen on the same queries it is scored on, so the accuracy is optimistic;
    AUROC is the threshold-free number to quote.
    """
    best = None
    for t in sorted(set(scores)) + [float("inf")]:
        answered = [s >= t for s in scores]
        m = abstention_metrics(answered, answerable)
        if best is None or m["abstention_accuracy"] > best["abstention_accuracy"]:
            best = {"threshold": t, **m}
    return best or {}


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """95% CI for mean(b - a) by resampling queries with replacement.

    Paired because both configurations answer the SAME queries; resampling
    per-query differences removes between-query variance and gives a much
    tighter, fairer comparison than two independent intervals.
    """
    diffs = [y - x for x, y in zip(a, b)]
    if len(diffs) < MIN_QUERIES_FOR_CI:
        # Resampling n values with replacement cannot widen an interval that
        # has no width to begin with: at n=1 every resample draws the same
        # single difference, so the CI collapses onto the delta and ANY
        # non-zero difference is reported as significant. A run on one query
        # duly produced "CI [+1.000, +1.000], significant=True", which reads
        # as a strong result and means nothing. Report the delta, withhold the
        # interval, and say why.
        return {"delta": sum(diffs) / len(diffs) if diffs else 0.0,
                "ci_low": None, "ci_high": None, "significant": None,
                "note": f"fewer than {MIN_QUERIES_FOR_CI} queries; "
                        "no confidence interval"}
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n
                   for _ in range(n_boot))
    lo, hi = means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
    return {"delta": sum(diffs) / n, "ci_low": lo, "ci_high": hi,
            "significant": lo > 0 or hi < 0}


# --- the sweep ---------------------------------------------------------------


def run_config(retriever, queries, config, variants_by_qid=None):
    """Evaluate one configuration. Returns a list of per-query result dicts."""
    search_kwargs = {k: v for k, v in config.items() if k != "multi_query"}
    rows = []
    for q in queries:
        variants = (variants_by_qid or {}).get(q["qid"]) if config["multi_query"] else None
        relevant = {r["chunk_id"] for r in q["relevant"]}
        metrics, ranked_at = {}, {}
        top_score = float("-inf")
        for k in K_VALUES:
            hits = retriever.search(q["question"], k=k, query_variants=variants,
                                    **search_kwargs)
            ids = [h["chunk_id"] for h in hits]
            ranked_at[k] = ids
            m = evaluate_query(ids, relevant, k_values=(k,))
            metrics.update({key: val for key, val in m.items() if key != "mrr"})
            if k == max(K_VALUES):
                metrics["mrr"] = m["mrr"]
                top_score = hits[0]["score"] if hits else float("-inf")
        rows.append({
            "qid": q["qid"], "category": q["category"],
            "answerable": bool(relevant), "top_score": top_score,
            "metrics": metrics, "ranked_ids": ranked_at[max(K_VALUES)],
        })
    return rows


def expand_all(queries, n, model, data_dir, provider="groq"):
    from llm import make_complete_fn
    from query_expansion import QueryExpander

    exp = QueryExpander(make_complete_fn(model, provider=provider), model_name=model,
                        n=n, cache_path=data_dir / "query_expansions.json")
    out = {}
    for i, q in enumerate(queries, 1):
        print(f"  expanding [{i}/{len(queries)}] {q['qid']}", end="\r")
        out[q["qid"]] = exp.expand(q["question"])
    print()
    return out


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})


def summarise(results: dict[str, list[dict]], out_dir: Path):
    """Write all summary files and return rows for printing."""
    summary, by_cat, sig, abst = [], [], [], []
    base = results.get(BASELINE)

    for name, rows in results.items():
        ans = [r for r in rows if r["answerable"]]
        agg = aggregate([r["metrics"] for r in ans])
        summary.append({"config": name, "n_queries": len(ans), **agg})

        cats = sorted({r["category"] for r in ans})
        for cat in cats:
            sub = [r["metrics"] for r in ans if r["category"] == cat]
            by_cat.append({"config": name, "category": cat, "n": len(sub),
                           **aggregate(sub)})

        if base is not None and name != BASELINE:
            base_by_qid = {r["qid"]: r for r in base if r["answerable"]}
            paired = [(base_by_qid[r["qid"]], r) for r in ans if r["qid"] in base_by_qid]
            for metric in ("recall@5", "mrr"):
                ci = paired_bootstrap([b["metrics"][metric] for b, _ in paired],
                                      [c["metrics"][metric] for _, c in paired])
                sig.append({"config": name, "vs": BASELINE, "metric": metric, **ci})

        scores = [r["top_score"] for r in rows]
        answerable = [r["answerable"] for r in rows]
        a = auroc([s for s, ok in zip(scores, answerable) if ok],
                  [s for s, ok in zip(scores, answerable) if not ok])
        abst.append({"config": name, "abstention_auroc": a,
                     **{f"best_{k}": v for k, v in
                        best_threshold(scores, answerable).items()}})

    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "summary_by_category.csv", by_cat)
    write_csv(out_dir / "significance.csv", sig)
    write_csv(out_dir / "abstention.csv", abst)
    return summary, by_cat, sig, abst


def fmt(v, width=9):
    return f"{v:{width}.3f}" if isinstance(v, (int, float)) and v is not None else " " * (width - 1) + "-"


def print_report(summary, by_cat, sig, abst):
    base = next((s for s in summary if s["config"] == BASELINE), None)
    print("\n" + "=" * 92)
    print(f"{'config':24s}" + "".join(f"{m:>11s}" for m in HEADLINE)
          + f"{'R@5 vs dense':>16s}")
    print("-" * 92)
    for s in summary:
        rel = ""
        b5, s5 = (base or {}).get("recall@5"), s.get("recall@5")
        if s["config"] != BASELINE and b5 and s5 is not None:
            rel = f"{100 * (s5 / b5 - 1):+.1f}%"
        print(f"{s['config']:24s}" + "".join(fmt(s.get(m), 11) for m in HEADLINE)
              + f"{rel:>16s}")

    cats = sorted({c["category"] for c in by_cat})
    print("\nRecall@5 by category")
    print(f"{'config':24s}" + "".join(f"{c:>14s}" for c in cats))
    for name in dict.fromkeys(c["config"] for c in by_cat):
        vals = {c["category"]: c.get("recall@5") for c in by_cat if c["config"] == name}
        print(f"{name:24s}" + "".join(fmt(vals.get(c), 14) for c in cats))

    if sig:
        print(f"\nDifference vs {BASELINE} (paired bootstrap 95% CI)")
        for s in sig:
            if "delta" not in s:
                continue
            if s.get("ci_low") is None:
                print(f"  {s['config']:24s} {s['metric']:9s} {s['delta']:+.3f} "
                      f"  ({s.get('note', 'no interval')})")
                continue
            star = "  *" if s["significant"] else ""
            print(f"  {s['config']:24s} {s['metric']:9s} {s['delta']:+.3f} "
                  f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]{star}")
        print("  * = interval excludes zero")

    print("\nAbstention: does top-1 score separate answerable from unanswerable?")
    for a in abst:
        au = a["abstention_auroc"]
        acc = a.get("best_abstention_accuracy")
        print(f"  {a['config']:24s} AUROC {fmt(au, 6)}   best-threshold accuracy {fmt(acc, 6)}")


@exit_on_rate_limit
def main():
    ap = argparse.ArgumentParser(description="Retrieval evaluation sweep.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--queries", type=Path, default=Path("data/eval/queries.jsonl"))
    ap.add_argument("--suite", choices=sorted(SUITES), default="core")
    ap.add_argument("--configs", nargs="+", help="Explicit config names (overrides --suite).")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-root", type=Path, default=Path("results/eval"))
    ap.add_argument("--limit", type=int, default=0, help="Only the first N queries (smoke test).")
    add_provider_args(ap)
    ap.add_argument("--list", action="store_true", help="List configs and suites, then exit.")
    args = ap.parse_args()

    if args.list:
        for name, c in CONFIGS.items():
            print(f"{name:24s} {c}")
        for s, names in SUITES.items():
            print(f"\nsuite {s}: {', '.join(names)}")
        return

    names = args.configs or SUITES[args.suite]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        sys.exit(f"Unknown config(s): {unknown}. See --list.")

    if not args.queries.exists():
        sys.exit(f"{args.queries} not found. Build it first: build_queries.py, "
                 "then review_queries.py review and export.")
    queries = [json.loads(line) for line in args.queries.read_text().splitlines()
               if line.strip()]
    if args.limit:
        queries = queries[: args.limit]
    chunks = {c["chunk_id"]: c for c in
              (json.loads(line) for line in
               (args.data_dir / "chunks.jsonl").read_text().splitlines() if line.strip())}
    validate_labels(queries, chunks)

    n_ans = sum(bool(q["relevant"]) for q in queries)
    if n_ans == 0:
        sys.exit(
            f"{args.queries} has {len(queries)} queries and none are answerable, "
            "so there is nothing to score.\nReview candidates first "
            "(review_queries.py review), then export (review_queries.py export)."
        )
    print(f"{len(queries)} queries ({n_ans} answerable, {len(queries) - n_ans} "
          f"unanswerable), {len(names)} configs")

    variants = {}
    mq_sizes = {CONFIGS[n]["multi_query"] for n in names} - {0}
    if mq_sizes:
        provider, model, _ = resolve(args)
        for size in sorted(mq_sizes):
            print(f"Query expansion (n={size}, {model}); cached after the first run")
            variants[size] = expand_all(queries, size, model, args.data_dir, provider)

    from retrieve import HybridRetriever
    needs_rerank = any(CONFIGS[n]["mode"] == "hybrid_rerank" for n in names)
    retriever = HybridRetriever(args.data_dir, load_reranker=needs_rerank)

    results = {}
    for name in names:
        c = CONFIGS[name]
        t0 = time.time()
        results[name] = run_config(retriever, queries, c,
                                   variants.get(c["multi_query"]))
        print(f"  {name:24s} done in {time.time() - t0:5.1f}s")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_query.jsonl").open("w") as f:
        for name, rows in results.items():
            for r in rows:
                f.write(json.dumps({"config": name, **r}) + "\n")
    (out_dir / "run_info.json").write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configs": {n: CONFIGS[n] for n in names},
        "k_values": K_VALUES,
        "n_queries": len(queries),
        "n_answerable": n_ans,
        "queries_file": str(args.queries),
        "queries_sha1": hashlib.sha1(args.queries.read_bytes()).hexdigest(),
        "n_chunks": len(chunks),
        "query_expansion_model": (args.model or "default") if variants else None,
    }, indent=2))

    print_report(*summarise(results, out_dir))
    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()
