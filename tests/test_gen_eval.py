"""Tests for generation evaluation: splitting, judge parsing, metrics, caching.

The generator and the judge are scripted functions, so this runs offline.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_chat  # noqa: E402,F401  (installs module stubs)
from test_chat import FakeEmbedder, chunk  # noqa: E402

import gen_eval as G  # noqa: E402
from generate import REFUSAL  # noqa: E402
from retrieve import HybridRetriever  # noqa: E402

ANSWER = ("EnD2 reaches 7.3% error on CIFAR-10 [S1]. The ensemble itself gets 6.2% "
          "[S1][S2].\n\n- EnD2 also keeps the ensemble's uncertainty [S3].\n"
          "| Method | C10 |\n|---|---|\n| EnD2 | 7.3 |")


def test_sentences_are_split_and_noise_dropped():
    sents = G.split_sentences(ANSWER)
    assert sents[0] == "EnD2 reaches 7.3% error on CIFAR-10 [S1]."
    assert sents[1].startswith("The ensemble itself")
    assert sents[2].startswith("EnD2 also keeps"), "bullet marker removed"
    assert not any(set(s) <= set("|-: ") for s in sents), "table separator dropped"
    assert len(G.split_sentences("ok.\n\n# Heading\n---")) == 0


def test_cited_sources():
    assert G.cited_sources("x [S1][S3] y") == {1, 3}
    assert G.cited_sources("no citations") == set()


def test_judgement_parsing_is_tolerant():
    raw = ("**SENTENCE 1:** 1\nSENTENCE 2: 1, 2\nSENTENCE 3: none\n"
           "RELEVANCE: 4\nCORRECT: partly\n")
    j = G.parse_judgement(raw, 4)
    assert j["support"] == [[1], [1, 2], [], []], "unmentioned sentence = unsupported"
    assert j["relevance"] == 4 and j["correct"] == 0.5


def test_answer_metrics():
    sents = ["A [S1].", "B [S2].", "C has no citation.", "D [S9]."]
    j = {"support": [[1], [3], [2], []], "relevance": 5, "correct": 1.0}
    m = G.answer_metrics(sents, j, n_sources=3)
    assert m["faithfulness"] == 0.75           # A, B, C supported; D not
    assert m["citation_coverage"] == 0.75      # C has no citation
    assert abs(m["citation_accuracy"] - 1 / 3) < 1e-9  # only A cites a supporter
    assert m["invalid_citations"] == 1          # [S9] with 3 sources
    assert m["relevance"] == 1.0 and m["correct"] == 1.0


def test_aggregate_separates_answerable_and_unanswerable():
    rows = [
        {"answerable": True, "refused": False,
         "metrics": {"faithfulness": 1.0, "relevance": 1.0, "correct": 1.0,
                     "citation_accuracy": 1.0, "citation_coverage": 1.0}},
        {"answerable": True, "refused": True, "metrics": {}},
        {"answerable": False, "refused": True, "metrics": {}},
        {"answerable": False, "refused": False, "metrics": {"faithfulness": 0.0}},
    ]
    a = G.aggregate(rows)
    assert a["faithfulness"] == 1.0, "refusals don't count as faithful answers"
    assert a["false_refusal_rate"] == 0.5
    assert a["abstention_rate"] == 0.5
    assert a["unanswerable_faithfulness"] == 0.0


def _retriever():
    docs = [chunk("a__0000", "1905.00076", "EnD2 reaches 7.3 error on CIFAR-10",
                  title="Ensemble Distribution Distillation"),
            chunk("b__0000", "1812.04606", "outlier exposure uses auxiliary data",
                  title="Outlier Exposure")]
    return HybridRetriever.from_chunks(docs, embedder=FakeEmbedder())


QUERIES = [
    {"qid": "q001", "category": "exact_term", "question": "EnD2 error on CIFAR-10?",
     "relevant": [{"chunk_id": "a__0000"}], "evidence": ["EnD2 reaches 7.3 error"]},
    {"qid": "q002", "category": "unanswerable", "question": "What is the capital of Peru?",
     "relevant": []},
]


def test_run_generates_judges_and_caches(tmp_path):
    gen_calls, judge_calls = [], []

    def generate(system, user):
        gen_calls.append(user)
        return ("EnD2 reaches 7.3% error on CIFAR-10 [S1]."
                if "Question: EnD2" in user else REFUSAL)

    def judge(system, user):
        judge_calls.append(user)
        return "SENTENCE 1: 1\nRELEVANCE: 5\nCORRECT: yes"

    cache = tmp_path / "cache.jsonl"
    configs = {"dense": {"mode": "dense"}}
    recs = G.run(QUERIES, configs, _retriever(), generate, judge, cache, "g", "j",
                 log=lambda *_: None)
    by_q = {r["qid"]: r for r in recs}
    assert by_q["q001"]["metrics"]["faithfulness"] == 1.0
    assert by_q["q001"]["metrics"]["correct"] == 1.0
    assert "Reference quote" in judge_calls[0], "evidence reaches the judge"
    assert by_q["q002"]["refused"] and len(judge_calls) == 1, "refusals aren't judged"

    again = G.run(QUERIES, configs, _retriever(), generate, judge, cache, "g", "j",
                  log=lambda *_: None)
    assert len(gen_calls) == 2 and len(again) == 2, "second run is served from cache"

    rows = G.summarise(again, tmp_path / "out")
    assert rows[0]["faithfulness"] == 1.0 and rows[0]["abstention_rate"] == 1.0
    assert json.loads((tmp_path / "out" / "summary.json").read_text())[0]["config"] == "dense"


def test_changing_a_model_invalidates_the_cache(tmp_path):
    cache = tmp_path / "cache.jsonl"
    calls = []

    def gen(s, u):
        calls.append(1)
        return REFUSAL

    G.run(QUERIES[:1], {"dense": {"mode": "dense"}}, _retriever(), gen,
          lambda s, u: "", cache, "g1", "j", log=lambda *_: None)
    G.run(QUERIES[:1], {"dense": {"mode": "dense"}}, _retriever(), gen,
          lambda s, u: "", cache, "g2", "j", log=lambda *_: None)
    assert len(calls) == 2


def test_unavailable_provider_skips_and_stops(tmp_path):
    import llm

    def down(s, u):
        raise llm.ProviderUnavailable("busy")

    many = [dict(QUERIES[0], qid=f"q{i:03d}") for i in range(6)]
    recs = G.run(many, {"dense": {"mode": "dense"}}, _retriever(), down,
                 lambda s, u: "", tmp_path / "c.jsonl", "g", "j", log=lambda *_: None)
    assert recs == []
    assert not (tmp_path / "c.jsonl").exists(), "nothing cached, so a rerun retries"


def test_evidence_is_joined_from_candidates(tmp_path):
    cands = tmp_path / "candidates.jsonl"
    cands.write_text("\n".join(json.dumps(c) for c in [
        {"qid": "q001", "evidence": ["EnD2 reaches 7.3 error"], "status": "kept"},
        {"qid": "q002", "evidence": [""], "status": "kept"},
    ]))
    queries = [{"qid": "q001", "question": "x"}, {"qid": "q002", "question": "y"},
               {"qid": "q003", "question": "z", "evidence": ["already set"]}]
    assert G.attach_evidence(queries, cands) == 1
    assert queries[0]["evidence"] == ["EnD2 reaches 7.3 error"]
    assert "evidence" not in queries[1], "empty quotes are not evidence"
    assert queries[2]["evidence"] == ["already set"]
    assert G.attach_evidence(queries, tmp_path / "missing.jsonl") == 0


def test_summary_compares_configs_on_the_same_questions(tmp_path):
    def rec(config, qid, refused):
        return {"config": config, "qid": qid, "answerable": True, "refused": refused,
                "metrics": {} if refused else {"faithfulness": 1.0}}

    # hybrid finished q1 and q2; dense was cut off after q1
    records = [rec("dense", "q1", False), rec("hybrid", "q1", False),
               rec("hybrid", "q2", True)]
    assert G.common_questions(records) == {"q1"}
    rows = {r["config"]: r for r in G.summarise(records, tmp_path)}
    assert rows["hybrid"]["n_answerable"] == 1, "q2 is left out: dense never did it"
    assert rows["hybrid"]["false_refusal_rate"] == 0.0


def test_a_config_with_no_results_is_not_silently_dropped(tmp_path):
    records = [{"config": "dense", "qid": "q1", "answerable": True, "refused": False,
                "metrics": {"faithfulness": 1.0}}]
    assert G.common_questions(records, ["dense", "hybrid_rerank"]) == set()
    rows = G.summarise(records, tmp_path, ["dense", "hybrid_rerank"])
    assert [r["config"] for r in rows] == ["dense", "hybrid_rerank"]
    assert all(r["n_answerable"] == 0 for r in rows), "nothing in common to compare"
