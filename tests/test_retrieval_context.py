"""Tests for name normalisation, title-contextual indexing and explain().

Regression source: asking the chat for EnD2's CIFAR-10 error never retrieved
the results table, because the table writes "C10" and "EnD2" and never names
its paper, while the question said "CIFAR10", "EnDD" and "the EnDD paper".
"""

import json
import sys
import types
from pathlib import Path

import pytest

RAG = Path(__file__).resolve().parents[1] / "src" / "rag"
sys.path.insert(0, str(RAG))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for name in ("sentence_transformers", "transformers"):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.SentenceTransformer = object
            stub.AutoTokenizer = object
            sys.modules[name] = stub
pytest.importorskip("faiss")

import retrieve  # noqa: E402
from test_chat import FakeEmbedder, chunk, make_base_corpus  # noqa: E402

# Verbatim excerpts from the corpus (1905.00076, chunks 12 and 13).
TABLE_TOP = ("Table 3: Mean Classification Error, % PRR , test-set negative "
             "log-likelihood (NLL) and expected calibration error (ECE) on "
             "C10/C100/TIM across three models ±2σ. DSET CRIT. IND ENSM EnD EnD2 "
             "EnD+AUX EnD2 +AUX PN+AUX C10 ERR 8.0 ±0.4 6.2 ± NA 6.7 ±0.3 7.3 ±0.2 "
             "6.7 ±0.2 6.9 ±0.2 7.5 ±0.3 C100 ERR 30.4 ±0.3 26.3 ± NA 28.0 ±0.4 "
             "27.9 ±0.3 28.2 ±0.3 28.0 ±0.5 28.0 ±0.7")
TABLE_BOTTOM = ("NLL 1.16 ±0.03 0.88 ± NA TIM ERR 41.8 ±0.6 36.6 ±NA 38.3 ±0.2 "
                "37.6 ±0.2 PRR 70.8 ±1.1 73.8 ± NA ECE 18.3 ±0.8 3.8 ± NA")
CORPUS = [
    chunk("endd__12", "1905.00076", TABLE_TOP, "Ensemble Distribution Distillation"),
    chunk("endd__13", "1905.00076", TABLE_BOTTOM, "Ensemble Distribution Distillation"),
    chunk("scale__7", "2105.06987",
          "we train ensemble members for 193,000 steps with Adam on 8 GPUs; the "
          "distillation objective uses a proxy Dirichlet target",
          "Scaling Ensemble Distribution Distillation to Many Classes with Proxy Targets"),
    chunk("kd__11", "2205.09526",
          "baselines include ensemble distillation; accuracy and calibration are "
          "reported for knowledge distillation students on image classification",
          "Simple Regularisation for Uncertainty-Aware Knowledge Distillation"),
    chunk("pn__8", "1802.10501",
          "prior networks model distributional uncertainty with a Dirichlet prior "
          "and are evaluated on MNIST and CIFAR-10 for out-of-distribution detection",
          "Predictive Uncertainty Estimation via Prior Networks"),
]
QUESTION = ("What is the error rate (ERR%) reported by the EnDD paper on the "
            "CIFAR100 and CIFAR10 datasets?")

# --- tokenizer ---------------------------------------------------------------------


def test_spellings_of_the_same_name_match():
    t = retrieve.tokenize_for_bm25
    assert t("CIFAR-10") == t("CIFAR10") == t("C10") == ["cifar10"]
    assert t("CIFAR-100") == t("c100") == ["cifar100"]
    assert t("EnD²") == t("EnDD") == t("EnD2") == ["end2"]
    assert t("ResNet-18") == ["resnet18"]
    assert t("FPR95 at 95%") == ["fpr95", "at", "95"], "unrelated tokens untouched"


def test_question_now_shares_the_key_terms_with_the_table():
    shared = set(retrieve.tokenize_for_bm25(QUESTION)) & \
        set(retrieve.tokenize_for_bm25(TABLE_TOP))
    assert {"cifar10", "cifar100", "end2", "err", "error"} <= shared


# --- title-contextual indexing ---------------------------------------------------------


def test_index_text():
    m = {"title": "Ensemble Distribution Distillation", "text": "C10 ERR 7.3"}
    assert retrieve.index_text(m) == "Ensemble Distribution Distillation\nC10 ERR 7.3"
    assert retrieve.index_text(m, contextual=False) == "C10 ERR 7.3"
    assert retrieve.index_text({"title": "", "text": "x"}) == "x"


def bm25_rank(r, query, chunk_id):
    ids = [r.meta[i]["chunk_id"] for i, _ in r._bm25_rank(query, len(r.meta))]
    return ids.index(chunk_id) + 1 if chunk_id in ids else None


def test_table_chunk_found_for_the_failing_question():
    r = retrieve.HybridRetriever.from_chunks(CORPUS, embedder=FakeEmbedder(),
                                             query_prefix="")
    assert bm25_rank(r, QUESTION, "endd__12") == 1


def test_title_lets_a_question_naming_the_paper_find_its_table():
    q = "classification error of ensemble distribution distillation"
    plain = retrieve.HybridRetriever.from_chunks(CORPUS, embedder=FakeEmbedder(),
                                                 query_prefix="", contextual=False)
    titled = retrieve.HybridRetriever.from_chunks(CORPUS, embedder=FakeEmbedder(),
                                                  query_prefix="")
    rank_plain, rank_titled = bm25_rank(plain, q, "endd__12"), bm25_rank(titled, q, "endd__12")
    assert rank_titled == 1 and (rank_plain is None or rank_plain > 1)


def test_titles_are_embedded_but_stored_text_is_unchanged():
    seen = []

    class Recorder(FakeEmbedder):
        def encode(self, texts, **kw):
            seen.extend(texts)
            return super().encode(texts, **kw)
    r = retrieve.HybridRetriever.from_chunks(CORPUS[:1], embedder=Recorder(),
                                             query_prefix="")
    assert seen == ["Ensemble Distribution Distillation\n" + TABLE_TOP]
    assert r.meta[0]["text"] == TABLE_TOP, "evaluation labels hash this text"


def test_reranker_sees_the_title():
    pairs = []

    class Rec:
        def predict(self, ps):
            pairs.extend(ps)
            return [1.0] * len(ps)
    r = retrieve.HybridRetriever.from_chunks(CORPUS, embedder=FakeEmbedder(),
                                             query_prefix="")
    r._reranker = Rec()
    r.search(QUESTION, mode="hybrid_rerank", k=2)
    assert any(t.startswith("Ensemble Distribution Distillation\n") for _, t in pairs)


def test_old_index_without_titles_still_works(tmp_path, monkeypatch):
    """An index built before this change has no contextual_header flag; it must
    keep matching how its vectors were made instead of mixing in titles."""
    make_base_corpus(tmp_path / "data", FakeEmbedder())
    monkeypatch.setattr(retrieve, "SentenceTransformer", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr(retrieve, "pick_device", lambda: "cpu")
    r = retrieve.HybridRetriever(tmp_path / "data")
    assert r.contextual is False
    merged = retrieve.HybridRetriever.from_chunks(
        CORPUS[:1], base_dir=tmp_path / "data", embedder=FakeEmbedder())
    assert merged.contextual is False, "uploads follow the corpus's convention"

    cfg = json.loads((tmp_path / "data/index_config.json").read_text())
    cfg["contextual_header"] = True
    (tmp_path / "data/index_config.json").write_text(json.dumps(cfg))
    assert retrieve.HybridRetriever(tmp_path / "data").contextual is True


# --- explain ------------------------------------------------------------------------------


def test_explain_reports_each_stage():
    r = retrieve.HybridRetriever.from_chunks(CORPUS, embedder=FakeEmbedder(),
                                             query_prefix="")
    rep = r.explain(QUESTION, "endd__12", candidate_n=2)
    assert rep["bm25"] == 1 and rep["hybrid"] is not None
    assert rep["in_rerank_pool"] == (rep["hybrid"] <= 2)
    assert "rerank_score" not in rep, "no reranker loaded"
    with pytest.raises(KeyError):
        r.explain(QUESTION, "nope")


def test_explain_with_rephrasings_reports_both():
    r = retrieve.HybridRetriever.from_chunks(CORPUS, embedder=FakeEmbedder(),
                                             query_prefix="")
    rep = r.explain("EnDD accuracy", "endd__12",
                    query_variants=["EnDD accuracy", "C10 classification error"])
    assert "hybrid_single_query" in rep and rep["hybrid"] is not None


# --- bibliography chunks are demoted, not dropped ----------------------------------
#
# Regression: is_reference chunks were filtered out of the candidate pool
# unconditionally. The citation-density tagger cannot reliably tell a reference
# list from a paragraph that cites heavily, so real body text was being deleted
# from retrieval - one chunk ranked #3 of 408 on dense, BM25 and hybrid alike
# and still could never be returned.

REF_QUESTION = "Can Prior Networks detect adaptive C&W L2 and EAD L1 attacks?"
# body prose that the density tagger scores as bibliography
MISTAGGED = ("It is of interest to assess how well Prior Networks are able to "
             "detect adaptive C&W L2 attacks and EAD L1 attacks")


def _ref_corpus():
    """The mis-tagged chunk, a sibling, and distractors.

    The distractors are not padding: BM25Okapi gives a term that appears in
    more than half the corpus a NEGATIVE idf, so on a two-document corpus
    every score goes below zero and _bm25_rank returns nothing at all.
    """
    tagged = chunk("pn__16", "1905.13472", MISTAGGED, "Reverse KL Prior Networks")
    tagged["is_reference"] = True
    others = [
        chunk("pn__02", "1905.13472",
              "we train prior networks on CIFAR-10 with a reverse KL objective",
              "Reverse KL Prior Networks"),
        chunk("edl__01", "1806.01768",
              "evidential deep learning places a Dirichlet distribution over "
              "the class probabilities to quantify evidence",
              "Evidential Deep Learning"),
        chunk("ens__01", "1612.01474",
              "deep ensembles average the predictions of several independently "
              "trained networks to calibrate confidence",
              "Simple and Scalable Predictive Uncertainty Estimation"),
        chunk("ood__01", "1812.04606",
              "outlier exposure fine-tunes on an auxiliary dataset of natural "
              "images to improve anomaly scores",
              "Deep Anomaly Detection with Outlier Exposure"),
    ]
    return [tagged, *others]


def _ref_retriever():
    return retrieve.HybridRetriever.from_chunks(
        _ref_corpus(), embedder=FakeEmbedder(), contextual=True)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_mistagged_reference_chunk_can_still_be_retrieved(mode):
    """It is demoted a place or two, not deleted - which is the whole point."""
    hits = _ref_retriever().search(REF_QUESTION, k=3, mode=mode)
    assert "pn__16" in [h["chunk_id"] for h in hits]


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_ref_penalty_zero_restores_the_old_exclusion(mode):
    hits = _ref_retriever().search(REF_QUESTION, k=3, mode=mode, ref_penalty=0.0)
    assert "pn__16" not in [h["chunk_id"] for h in hits]


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_penalty_lowers_a_reference_chunk_but_leaves_others_alone(mode):
    retr = _ref_retriever()
    full = {h["chunk_id"]: h["score"]
            for h in retr.search(REF_QUESTION, k=5, mode=mode, include_refs=True)}
    demoted = {h["chunk_id"]: h["score"]
               for h in retr.search(REF_QUESTION, k=5, mode=mode, ref_penalty=0.5)}
    assert demoted["pn__16"] < full["pn__16"]
    for cid in full:
        if cid != "pn__16":
            assert demoted[cid] == pytest.approx(full[cid])


def test_demotion_is_proportional_to_the_spread_of_the_scores():
    """A halving of the raw score would sink an RRF hit below every rival,
    because RRF scores are bunched near 1/RRF_K. Interpolating toward the
    weakest candidate keeps the demotion proportionate."""
    ranked = [(0, 0.0328), (1, 0.0323), (2, 0.0317)]
    meta = [{"is_reference": True}, {"is_reference": False}, {"is_reference": False}]
    [(i, demoted), *_] = sorted(retrieve.demote_refs(ranked, meta, 0.5),
                                key=lambda x: x[0])
    assert i == 0
    assert demoted > 0.0328 * 0.5   # far gentler than a plain multiplication
    assert demoted < 0.0328         # but still a demotion


def test_demotion_is_a_halving_when_scores_reach_down_to_zero():
    """The reranker's scores do span 0-1, so there the penalty is literal."""
    ranked = [(0, 0.98), (1, 0.40), (2, 0.0)]
    meta = [{"is_reference": True}, {"is_reference": False}, {"is_reference": False}]
    by_idx = dict(retrieve.demote_refs(ranked, meta, 0.5))
    assert by_idx[0] == pytest.approx(0.49)


def test_results_stay_sorted_after_the_penalty_is_applied():
    hits = _ref_retriever().search(REF_QUESTION, k=5, mode="hybrid")
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


# --- is the per-paper cap actually binding? ----------------------------------------
#
# Regression: over-cap and below-floor chunks shared one reserve, and the
# backfill drew from it indiscriminately, reinstating the chunks the cap had
# just excluded. Over an 87-query evaluation, 46 of 68 answerable queries came
# back with more than max_per_paper chunks from one paper, so the cap was close
# to advisory and the max_per_paper ablations compared almost nothing.

def _cap_input():
    """Four strong chunks from paper A, two weaker ones from B and C."""
    ranked = [(0, 0.99), (1, 0.98), (2, 0.97), (3, 0.96), (4, 0.95), (5, 0.94)]
    meta = [{"arxiv_id": "A"}] * 4 + [{"arxiv_id": "B"}, {"arxiv_id": "C"}]
    return ranked, meta


def test_soft_cap_exceeds_max_per_paper_when_backfilling():
    ranked, meta = _cap_input()
    got = retrieve.apply_diversity_cap(ranked, meta, k=5, max_per_paper=2,
                                       min_score_frac=0.0)
    papers = [meta[i]["arxiv_id"] for i, _ in got]
    assert len(got) == 5
    assert papers.count("A") > 2          # the documented, historical behaviour


def test_hard_cap_never_exceeds_max_per_paper():
    ranked, meta = _cap_input()
    got = retrieve.apply_diversity_cap(ranked, meta, k=5, max_per_paper=2,
                                       min_score_frac=0.0, hard_cap=True)
    papers = [meta[i]["arxiv_id"] for i, _ in got]
    assert papers.count("A") == 2
    assert set(papers) == {"A", "B", "C"}
    assert len(got) == 4                  # fewer than k, rather than break the cap


def test_hard_cap_still_backfills_below_floor_chunks():
    """Only the OVER-CAP reserve is off limits; a chunk held back purely for
    being below the relevance floor may still fill an empty slot."""
    ranked = [(0, 1.0), (1, 0.05)]
    meta = [{"arxiv_id": "A"}, {"arxiv_id": "B"}]
    got = retrieve.apply_diversity_cap(ranked, meta, k=2, max_per_paper=2,
                                       min_score_frac=0.25, hard_cap=True)
    assert [i for i, _ in got] == [0, 1]


def test_hard_cap_reaches_search():
    hits = _ref_retriever().search(REF_QUESTION, k=5, mode="hybrid",
                                   max_per_paper=1, hard_cap=True)
    papers = [h["arxiv_id"] for h in hits]
    assert len(papers) == len(set(papers))
