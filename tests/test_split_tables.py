"""Tests for split-table context expansion, using the real chunks that failed.

1905.00076__0012 holds Table 3's C10/C100 rows but ranked #36 (BM25) and #62
(dense) with reranker score 0.0012; __0013, the table's second half, ranked #1.
"""

import sys
from pathlib import Path

RAG = Path(__file__).resolve().parents[1] / "src" / "rag"
sys.path.insert(0, str(RAG))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_chat  # noqa: E402,F401  (installs module stubs)
import chat  # noqa: E402
from retrieve import _table_like, expand_split_tables  # noqa: E402

C12 = ("d) auxiliary data on which ensemble predictions can be obtained. Further note "
       "that in these experiments we explicitly chose to use the simpler VGG-16 "
       "architecture. Table 3: Mean Classification Error, % PRR , test-set negative "
       "log-likelihood (NLL) and expected calibration error (ECE) on C10/C100/TIM "
       "across three models ±2σ. DSET CRIT. IND ENSM EnD EnD2 EnD+AUX EnD2 +AUX "
       "PN+AUX C10 ERR 8.0 ±0.4 6.2 ± NA 6.7 ±0.3 7.3 ±0.2 6.7 ±0.2 6.9 ±0.2 7.5 "
       "±0.3 PRR 84.6 ±1.2 86.8 ± NA 84.8 ±0.8 85.3 ±1.1 85.1 ±0.1 85.7 ±0.3 82.0 "
       "±1.4 C100 ERR 30.4 ±0.3 26.3 ± NA 28.0 ±0.4 27.9 ±0.3 28.2 ±0.3 28.0 ±0.5 "
       "28.0 ±0.7 PRR 72.5 ±1.0 75.0 ± NA 73.1 ±0.5 73.7 ±0.7 74.0 ±0.3 74.0 ±0.2 "
       "63.7 ±0.8 ECE 9.3 ±0.8 1.2 ± NA 8.2 ±0.3 4.9 ±0.5 1.9 ±0.3 5.6 ±0.5 37.9 ±0.4")
C13 = ("ECE 9.3 ±0.8 1.2 ± NA 8.2 ±0.3 4.9 ±0.5 1.9 ±0.3 5.6 ±0.5 37.9 ±0.4 NLL 1.16 "
       "±0.03 0.88 ± NA 1.06 ±0.01 1.14 ±0.01 0.98 ±0.00 1.14 ±0.01 1.87 ±0.03 TIM ERR "
       "41.8 ±0.6 36.6 ±NA 38.3 ±0.2 37.6 ±0.2 38.5 ±0.3 37.3 ±0.5 40.0 ±0.6 Firstly, "
       "we investigate the ability of a single model to retain the ensemble's "
       "classification performance after either Ensemble Distillation (EnD) or "
       "Ensemble Distribution Distillation (EnD2), with results presented in table 3 "
       "in terms of error rate and prediction rejection ratio.")
PROSE = ("Ensemble Distribution Distillation trains a single network to capture the "
         "diversity of an ensemble by modelling a Dirichlet over its predictions, "
         "which preserves knowledge uncertainty in one cheap model. " * 3)


def meta(cid, text, page):
    return {"chunk_id": cid, "arxiv_id": "1905.00076", "title": "Ensemble Distribution "
            "Distillation", "year": "2019", "page_start": page, "page_end": page,
            "is_reference": False, "text": text}


CHUNKS = {m["chunk_id"]: m for m in [
    meta("1905.00076__0011", PROSE, 6), meta("1905.00076__0012", C12, 7),
    meta("1905.00076__0013", C13, 8), meta("1905.00076__0014", PROSE, 8)]}


def test_table_detection():
    assert _table_like(C13.split()[:30])
    assert _table_like(C12.split()[-30:])
    assert not _table_like(PROSE.split()[:30])


def test_second_half_hit_gets_the_rows_it_was_missing():
    hit = dict(CHUNKS["1905.00076__0013"], score=0.99)
    [out] = expand_split_tables([hit], CHUNKS.get)
    assert out["expanded_with"] == ["1905.00076__0012"]
    assert "C10 ERR 8.0 ±0.4 6.2 ± NA 6.7 ±0.3 7.3" in out["text"], "EnD2 7.3% now visible"
    assert "C100 ERR" in out["text"] and "TIM ERR" in out["text"]
    assert out["text"].count("37.9 ±0.4") == 1, "overlap region not duplicated"
    assert (out["page_start"], out["page_end"]) == (7, 8)
    assert out["chunk_id"] == "1905.00076__0013" and out["score"] == 0.99


def test_first_half_hit_gets_the_next_chunk():
    [out] = expand_split_tables([dict(CHUNKS["1905.00076__0012"])], CHUNKS.get)
    assert out["expanded_with"] == ["1905.00076__0013"]
    assert "TIM ERR" in out["text"]


def test_prose_hits_untouched():
    hit = CHUNKS["1905.00076__0011"]
    assert expand_split_tables([hit], CHUNKS.get) == [hit]


def test_no_expansion_when_neighbour_already_retrieved():
    hits = [CHUNKS["1905.00076__0013"], CHUNKS["1905.00076__0012"]]
    out = expand_split_tables(hits, CHUNKS.get)
    assert [h["chunk_id"] for h in out] == ["1905.00076__0013", "1905.00076__0012"]
    assert all("expanded_with" not in h for h in out)


def test_expansion_is_capped():
    hits = [dict(CHUNKS["1905.00076__0013"], chunk_id=f"p{i}__0013") for i in range(4)]
    lookup = {f"p{i}__0012": CHUNKS["1905.00076__0012"] for i in range(4)}
    out = expand_split_tables(hits, lookup.get, max_expand=2)
    assert sum("expanded_with" in h for h in out) == 2


def test_missing_neighbour_is_harmless():
    hit = CHUNKS["1905.00076__0013"]
    assert expand_split_tables([hit], lambda cid: None) == [hit]


def test_chat_passes_expanded_table_to_the_model():
    class R:
        def search(self, q, **_):
            return [CHUNKS["1905.00076__0013"]]

        def chunk_by_id(self, cid):
            return CHUNKS.get(cid)
    llm = test_chat.ScriptedLLM(["EnD2 has 7.3% error on CIFAR-10 [S1]."])
    turn = chat.ChatSession(R(), llm).ask("What error does EnDD get on CIFAR10?")
    assert "C10 ERR 8.0" in llm.calls[0][-1]["content"]
    assert turn.hits[0]["expanded_with"] == ["1905.00076__0012"]


# --- table reading rules and the table probe ---------------------------------------


def test_prompt_requires_row_and_column_for_table_numbers():
    from generate import SYSTEM_PROMPT
    assert "name its row and" in SYSTEM_PROMPT and "column" in SYSTEM_PROMPT
    assert "give the paper's own method" in SYSTEM_PROMPT
    assert "say the table is ambiguous instead" in SYSTEM_PROMPT
    assert chat.CHAT_RULES.lstrip().startswith("7."), "chat rule numbered after 6"


def test_probe_keyword_tolerates_hyphens_and_case():
    from debug_tables import keyword_pattern
    pat = keyword_pattern("CIFAR-5")
    for text in ("CIFAR5", "cifar-5", "CIFAR 5"):
        assert pat.search(text)
    assert keyword_pattern("C10 ERR").search("PN+AUX C10 ERR 8.0")


def test_quit_typo_gets_a_hint_not_a_query(monkeypatch, capsys):
    import builtins
    import llm
    monkeypatch.setattr(llm, "get_client", lambda: None)
    monkeypatch.setattr(llm, "call_groq_messages",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))
    monkeypatch.setattr(chat, "build_retriever", lambda args: None)
    inputs = iter(["qiut", "exti", "quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    monkeypatch.setattr(sys, "argv", ["chat.py"])
    chat.main()
    assert capsys.readouterr().out.count("Did you mean to leave?") == 2
