"""Unit tests for retrieval metrics, with hand-computed expected values."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))
from metrics import (  # noqa: E402
    abstention_metrics,
    aggregate,
    evaluate_query,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    success_at_k,
)

RANKED = ["a", "b", "c", "d", "e"]
REL = {"b", "e", "z"}  # z is never retrieved


def test_recall():
    assert recall_at_k(RANKED, REL, 1) == 0
    assert recall_at_k(RANKED, REL, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, REL, 5) == pytest.approx(2 / 3)


def test_precision_uses_k_as_denominator():
    assert precision_at_k(RANKED, REL, 5) == pytest.approx(2 / 5)
    # only 2 results returned but k=5: still divided by 5
    assert precision_at_k(["b", "e"], REL, 5) == pytest.approx(2 / 5)


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, REL) == pytest.approx(1 / 2)
    assert reciprocal_rank(["x", "y"], REL) == 0.0
    assert reciprocal_rank(["a", "c", "b"], REL, k=2) == 0.0


def test_ndcg_hand_computed():
    # relevant at ranks 2 and 5; 3 relevant total, so ideal fills ranks 1-3
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, REL, 5) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_is_one():
    assert ndcg_at_k(["b", "e"], {"b", "e"}, 5) == pytest.approx(1.0)


def test_success():
    assert success_at_k(RANKED, REL, 1) == 0.0
    assert success_at_k(RANKED, REL, 2) == 1.0


def test_unanswerable_returns_none():
    for fn in (recall_at_k, precision_at_k, ndcg_at_k, success_at_k):
        assert fn(RANKED, set(), 5) is None
    assert reciprocal_rank(RANKED, set()) is None


def test_aggregate_skips_none():
    q1 = evaluate_query(["b"], {"b"}, k_values=(1,))
    q2 = evaluate_query(["x"], set(), k_values=(1,))  # unanswerable
    q3 = evaluate_query(["x"], {"b"}, k_values=(1,))
    agg = aggregate([q1, q2, q3])
    # unanswerable excluded: mean of 1.0 and 0.0
    assert agg["recall@1"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx(0.5)


def test_abstention_metrics():
    answered = [True, True, False, False]
    answerable = [True, False, True, False]
    m = abstention_metrics(answered, answerable)
    assert m["abstention_accuracy"] == pytest.approx(0.5)
    assert m["false_answer_rate"] == pytest.approx(0.5)
    assert m["false_abstain_rate"] == pytest.approx(0.5)
