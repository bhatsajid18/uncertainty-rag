"""
Retrieval evaluation metrics.

Pure functions over (ranked_ids, relevant_ids). No dependency on how the
ground-truth set was built or how retrieval was run, so these can be unit
tested in isolation and reused for any ranking task.

Conventions:
  ranked_ids    list of chunk_ids in the order the retriever returned them
  relevant_ids  set of chunk_ids judged relevant for that query
  k             cutoff; metrics consider only the first k results

Unanswerable queries (relevant_ids empty) are a special case: recall, NDCG,
success and MRR are undefined there, so they return None and aggregate()
skips them. Those queries are scored separately by abstention accuracy.
"""

from __future__ import annotations

import math


def _top(ranked_ids: list[str], k: int) -> list[str]:
    return ranked_ids[:k] if k > 0 else list(ranked_ids)


def recall_at_k(ranked_ids, relevant_ids, k):
    """Fraction of all relevant chunks that appear in the top k."""
    if not relevant_ids:
        return None
    return len(set(_top(ranked_ids, k)) & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids, relevant_ids, k):
    """Fraction of the top k that are relevant.

    Denominator is k, not len(results), so a configuration that returns fewer
    results is not flattered.
    """
    if not relevant_ids:
        return None
    if k <= 0:
        return 0.0
    return len(set(_top(ranked_ids, k)) & relevant_ids) / k


def reciprocal_rank(ranked_ids, relevant_ids, k=None):
    """1 / rank of the first relevant chunk; averaged over queries this is MRR.

    Matches how RAG is used: the LLM reads the top few chunks, so a relevant
    chunk at rank 1 versus rank 5 matters a lot.
    """
    if not relevant_ids:
        return None
    ids = _top(ranked_ids, k) if k else ranked_ids
    for i, cid in enumerate(ids, start=1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def _dcg(ranked_ids, gains, k):
    return sum(
        gains.get(cid, 0.0) / math.log2(i + 1)
        for i, cid in enumerate(_top(ranked_ids, k), start=1)
    )


def ndcg_at_k(ranked_ids, relevant_ids, k, gains=None):
    """Normalized DCG: rank-sensitive, credits putting relevant chunks earlier.

    Binary relevance by default; pass `gains` for graded judgements.
    """
    if not relevant_ids:
        return None
    gains = gains or {cid: 1.0 for cid in relevant_ids}
    ideal = _dcg(sorted(gains, key=gains.get, reverse=True), gains, k)
    return _dcg(ranked_ids, gains, k) / ideal if ideal else 0.0


def success_at_k(ranked_ids, relevant_ids, k):
    """1.0 if at least one relevant chunk is in the top k (hit rate).

    Often the metric that matters most for RAG: the LLM usually needs one good
    chunk, not all of them.
    """
    if not relevant_ids:
        return None
    return 1.0 if set(_top(ranked_ids, k)) & relevant_ids else 0.0


def evaluate_query(ranked_ids, relevant_ids, k_values=(1, 3, 5, 10)):
    """All metrics for one query, keyed like 'recall@5'."""
    relevant_ids = set(relevant_ids)
    out = {}
    for k in k_values:
        out[f"recall@{k}"] = recall_at_k(ranked_ids, relevant_ids, k)
        out[f"precision@{k}"] = precision_at_k(ranked_ids, relevant_ids, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevant_ids, k)
        out[f"success@{k}"] = success_at_k(ranked_ids, relevant_ids, k)
    out["mrr"] = reciprocal_rank(ranked_ids, relevant_ids)
    return out


def aggregate(per_query):
    """Mean of each metric across queries, skipping None (unanswerable)."""
    if not per_query:
        return {}
    keys = []
    for q in per_query:
        for key in q:
            if key not in keys:
                keys.append(key)
    out = {}
    for key in keys:
        vals = [q[key] for q in per_query if q.get(key) is not None]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def abstention_metrics(answered, answerable):
    """How well the system decides when NOT to answer.

    answered[i]    True if the system gave an answer (did not refuse)
    answerable[i]  True if the corpus actually contains the answer

    false_answer_rate   answered something unanswerable (hallucination risk)
    false_abstain_rate  refused something it could have answered (over-caution)
    """
    if not answered or len(answered) != len(answerable):
        return {}
    pairs = list(zip(answered, answerable))
    unans = [a for a, ok in pairs if not ok]
    ans = [a for a, ok in pairs if ok]
    return {
        "abstention_accuracy": sum(a == ok for a, ok in pairs) / len(pairs),
        "false_answer_rate": sum(unans) / len(unans) if unans else None,
        "false_abstain_rate": sum(not a for a in ans) / len(ans) if ans else None,
    }
