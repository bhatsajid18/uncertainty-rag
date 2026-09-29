"""Tests for the methods adopted from rag-for-beginners, in our own stack:
table notes (multimodal-RAG summaries), MMR and a score floor (retrieval
methods), sentence and semantic chunking, plus the fixes prompted by the same
session's terminal log: small talk without retrieval, and failing fast on
Groq's daily limit.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_chat  # noqa: E402,F401  (installs module stubs)
from test_chat import FakeEmbedder, chunk, fake_tokenizer  # noqa: E402
from test_split_tables import C12, C13, CHUNKS, PROSE  # noqa: E402

import chat  # noqa: E402
import chunk_papers as cp  # noqa: E402
import retrieve  # noqa: E402
import table_notes as tn  # noqa: E402
from generate import REFUSAL, TABLE_GRID_LABEL, build_user_prompt  # noqa: E402

GRID = ("| DSET CRIT | IND | ENSM | EnD | EnD2 |\n|---|---|---|---|---|\n"
        "| C10 ERR | 8.0 ±0.4 | 6.2 ± NA | 6.7 ±0.3 | 7.3 ±0.2 |\n"
        "| C100 ERR | 30.4 ±0.3 | 26.3 ± NA | 28.0 ±0.4 | 27.9 ±0.3 |")
SOURCE = C12 + " " + C13


# --- table notes -------------------------------------------------------------------


def test_table_chunks_are_found_and_prose_is_not():
    assert tn.has_table(C12) and tn.has_table(C13)
    assert not tn.has_table(PROSE)
    ref = dict(CHUNKS["1905.00076__0012"], is_reference=True)
    assert tn.table_chunks([ref, CHUNKS["1905.00076__0011"]]) == []


SENSOY = ("uncertainty threshold for EDL. Method MNIST CIFAR 5 L2 99.4 76 Dropout 99.5 84 "
          "Deep Ensemble 99.3 79 FFGU 99.1 78 FFLU 99.1 77 MNFG 99.3 84 EDL 99.3 83 "
          "Table 1: Test accuracies (%) for MNIST and CIFAR5 datasets. the Bayesian "
          "neural net used in [18] with the additive parametrization [26], (e) MNFG")
NUMERIC_PROSE = ("Let us note that the maximum entropy is log(10) and log(5) for the MNIST "
                 "and CIFAR5 datasets, respectively. We trained for 200 epochs with "
                 "learning rate 0.1, batch size 128 and weight decay 5e-4, decayed by 0.2 "
                 "at epochs 60, 120 and 160 on 4 GPUs with 16 GB.")


def test_small_real_table_found_numeric_prose_not():
    assert tn.has_table(SENSOY), "Sensoy Table 1 (real chunk text)"
    assert not tn.has_table(NUMERIC_PROSE)


def test_correct_grid_passes():
    assert tn.check_markdown(GRID, SOURCE) == (True, "")


def test_invented_number_is_rejected():
    ok, why = tn.check_markdown(GRID.replace("27.9", "26.9"), SOURCE)
    assert not ok and "26.9" in why


def test_value_copied_into_two_cells_is_rejected():
    ok, why = tn.check_markdown(GRID.replace("6.7 ±0.3", "7.3 ±0.2"), SOURCE)
    assert not ok, "7.3 appears once in the text but twice in the grid"


def test_ragged_row_is_rejected():
    bad = GRID.replace("| 7.3 ±0.2 |", "|")
    assert tn.check_markdown(bad, SOURCE)[0] is False


def test_row_labels_with_digits_are_not_counted_as_values():
    grid = ("| Split | Metric | EDL |\n|---|---|---|\n| C10 | ERR | 7.3 ±0.2 |\n"
            "| C10 | PRR | 85.3 ±1.1 |")
    assert tn.check_markdown(grid, SOURCE)[0], "'C10' twice is a label, not a value"


def test_integer_column_grid_passes():
    text = ("Method MNIST CIFAR 5 L2 99.4 76 Dropout 99.5 84 EDL 99.3 83 "
            "Table 1: Test accuracies (%) for MNIST and CIFAR5 datasets.")
    grid = ("| Method | MNIST | CIFAR5 |\n|---|---|---|\n| L2 | 99.4 | 76 |\n"
            "| Dropout | 99.5 | 84 |\n| EDL | 99.3 | 83 |")
    assert tn.check_markdown(grid, text) == (True, "")


DESCRIPTION = ("Compares ensemble distillation (EnD) and the paper's Ensemble "
               "Distribution Distillation (EnD2) by error on CIFAR-10, CIFAR-100 "
               "and TinyImageNet.")


def _reply(grid=GRID):
    """A well-formed reply in the line format the prompt asks for."""
    return f"TABLE\nCAPTION: Table 3\nDESCRIPTION: {DESCRIPTION}\nGRID:\n{grid}\nEND"


def test_json_reply_is_still_accepted():
    raw = "```json\n" + json.dumps({"tables": [{
        "caption": "Table 3", "description": DESCRIPTION, "markdown": GRID}]}) + "\n```"
    tables, error = tn.parse_tables(raw)
    assert not error and tables[0]["markdown"] == GRID


def test_line_format_survives_formatting_and_chatter():
    raw = ("Sure, here is the table.\n\n**TABLE**\n"
           "CAPTION: Table 3: Mean Classification Error\n"
           f"DESCRIPTION: {DESCRIPTION}\n  Values are means over three models.\n"
           f"GRID:\n{GRID}\n**END**\n")
    [t], error = tn.parse_tables(raw)
    assert not error
    assert t["caption"].startswith("Table 3") and t["markdown"] == GRID
    assert t["description"].endswith("three models.")


def test_grid_without_its_label_is_still_read():
    raw = f"TABLE\nCAPTION: Table 3\nDESCRIPTION: {DESCRIPTION}\n\n{GRID}\nEND"
    [t], error = tn.parse_tables(raw)
    assert not error and t["markdown"] == GRID and t["description"] == DESCRIPTION


def test_two_tables_in_one_reply():
    tables, error = tn.parse_tables(_reply() + "\n" + _reply())
    assert not error and len(tables) == 2


def test_grid_none_keeps_the_description():
    raw = f"TABLE\nCAPTION: Table 3\nDESCRIPTION: {DESCRIPTION}\nGRID: none\nEND"
    [t], error = tn.parse_tables(raw)
    assert not error and t["markdown"] == "" and t["description"] == DESCRIPTION


def test_no_tables_is_not_an_error():
    assert tn.parse_tables("NO TABLES") == ([], "")


def test_reply_in_the_wrong_format_is_an_error():
    tables, error = tn.parse_tables("The table shows error rates of 7.3 and 27.9.")
    assert tables == [] and "format" in error


def test_fields_without_the_table_marker_are_salvaged():
    """The largest real failure mode: the model gives CAPTION/DESCRIPTION/GRID
    but omits the bare "TABLE" line, and the strict scan dropped the lot."""
    raw = f"CAPTION: Table 3\nDESCRIPTION: {DESCRIPTION}\nGRID:\n{GRID}"
    [t], error = tn.parse_tables(raw)
    assert not error
    assert t["markdown"] == GRID and t["description"] == DESCRIPTION


def test_numbered_table_marker_opens_a_block():
    raw = f"TABLE 2\nCAPTION: Table 2\nDESCRIPTION: {DESCRIPTION}\nGRID:\n{GRID}\nEND"
    [t], error = tn.parse_tables(raw)
    assert not error and t["markdown"] == GRID


def test_code_fences_around_the_block_are_ignored():
    raw = f"```\nTABLE\nCAPTION: Table 3\nDESCRIPTION: {DESCRIPTION}\nGRID:\n{GRID}\nEND\n```"
    [t], error = tn.parse_tables(raw)
    assert not error and t["markdown"] == GRID


def test_salvage_does_not_invent_a_table_from_a_bare_grid_label():
    """A reply with a field but no content must still be reported as an error,
    so a genuinely bad run is not silently recorded as a table with no data."""
    tables, error = tn.parse_tables("GRID: none")
    assert tables == [] and "format" in error


def test_notes_are_generated_once_and_resumed(tmp_path):
    chunks = list(CHUNKS.values())
    calls = []

    def complete(system, user):
        calls.append(user)
        return _reply()

    path = tmp_path / "table_notes.json"
    notes = tn.generate_notes(chunks, complete, "m", path, log=lambda *_: None)
    assert set(notes) == {"1905.00076__0012", "1905.00076__0013"}
    assert "TIM ERR" in calls[0], "the split table's other half is in the context"
    assert json.loads(path.read_text()) == notes
    tn.generate_notes(chunks, complete, "m", path, log=lambda *_: None)
    assert len(calls) == 2, "second run makes no calls"


def test_attach_uses_current_notes_only(tmp_path):
    chunks = [dict(c) for c in CHUNKS.values()]
    notes = tn.generate_notes(chunks, lambda s, u: _reply(), "m",
                              tmp_path / "n.json", log=lambda *_: None)
    stale = dict(chunks[2], text=chunks[2]["text"] + " edited")
    stats = tn.attach_notes([chunks[1], stale], notes)
    assert stats == {"described": 1, "rebuilt": 1, "stale": 1}
    assert "EnD2" in chunks[1]["table_note"] and chunks[1]["table_markdown"] == GRID
    assert "table_note" not in stale


def test_rejected_grid_keeps_the_description(tmp_path):
    chunks = [dict(CHUNKS["1905.00076__0012"]), dict(CHUNKS["1905.00076__0013"])]
    notes = tn.generate_notes(chunks, lambda s, u: _reply(GRID.replace("8.0", "9.9")),
                              "m", tmp_path / "n.json", log=lambda *_: None)
    stats = tn.attach_notes(chunks, notes)
    assert stats["described"] == 2 and stats.get("rebuilt", 0) == 0
    assert stats["rejected"] == 2


def test_unparseable_reply_is_recorded_with_a_sample(tmp_path):
    notes = tn.generate_notes([CHUNKS["1905.00076__0012"]],
                              lambda s, u: "sorry, I can't read that table",
                              "m", tmp_path / "n.json", log=lambda *_: None)
    [t] = notes["1905.00076__0012"]["tables"]
    assert not t["markdown_ok"] and "format" in t["reject_reason"]
    assert t["reply_start"].startswith("sorry")
    assert tn.summarize(notes) == {"unparseable": 1}


def test_retry_failed_redoes_only_the_empty_ones(tmp_path):
    chunks = [CHUNKS["1905.00076__0012"], CHUNKS["1905.00076__0013"]]
    path = tmp_path / "n.json"
    replies = [_reply(), "no idea"]  # the second chunk's reply is unusable

    def complete(system, user):
        return replies.pop(0) if replies else _reply()

    tn.generate_notes(chunks, complete, "m", path, log=lambda *_: None)
    written = json.loads(path.read_text())
    assert tn.summarize(written) == {"with_grid": 1, "unparseable": 1}
    good_before = written["1905.00076__0012"]

    redone = []
    tn.generate_notes(chunks, lambda s, u: redone.append(u) or _reply(), "m", path,
                      log=lambda *_: None, retry_failed=True)
    assert len(redone) == 1, "only the failed chunk is redone"
    written = json.loads(path.read_text())
    assert tn.summarize(written) == {"with_grid": 2}
    assert written["1905.00076__0012"] == good_before, "the good notes are untouched"


def test_index_text_includes_note_before_text():
    m = dict(CHUNKS["1905.00076__0012"], table_note="Compares EnD and EnD2.")
    text = retrieve.index_text(m)
    assert text.index("Compares EnD") < text.index("auxiliary data")
    assert text.startswith("Ensemble Distribution Distillation\n")


def test_prompt_shows_each_grid_once():
    a = dict(CHUNKS["1905.00076__0012"], table_markdown=GRID)
    b = dict(CHUNKS["1905.00076__0013"], table_markdown=GRID)
    prompt = build_user_prompt("q", [a, b])
    assert prompt.count(TABLE_GRID_LABEL) == 1 and GRID in prompt


def test_split_table_merge_keeps_neighbour_grid():
    lookup = {"1905.00076__0012": dict(CHUNKS["1905.00076__0012"], table_markdown=GRID)}
    [out] = retrieve.expand_split_tables([CHUNKS["1905.00076__0013"]], lookup.get)
    assert out["table_markdown"] == GRID


def test_meta_row_keeps_notes_only_when_present():
    c = chunk("d__0000", "d", "text")
    assert "table_note" not in retrieve.meta_row(c)
    assert retrieve.meta_row(dict(c, table_note="x"))["table_note"] == "x"


# --- MMR and the score floor ----------------------------------------------------------


def test_mmr_skips_near_duplicates():
    v = np.array([[1, 0], [0.999, 0.045], [0, 1]], dtype="float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ranked = [(10, 0.9), (11, 0.89), (12, 0.5)]
    assert [i for i, _ in retrieve.mmr_select(ranked, v, 2, lam=0.5)] == [10, 12]
    assert [i for i, _ in retrieve.mmr_select(ranked, v, 2, lam=1.0)] == [10, 11]


def _retriever():
    docs = [chunk("a__0000", "a", "evidential deep learning dirichlet uncertainty"),
            chunk("a__0001", "a", "evidential deep learning dirichlet uncertainty"),
            chunk("b__0000", "b", "outlier exposure auxiliary dataset anomaly")]
    return retrieve.HybridRetriever.from_chunks(docs, embedder=FakeEmbedder())


def test_search_with_mmr_avoids_duplicate_chunk():
    r = _retriever()
    ids = r.search_ids("evidential dirichlet", k=2, mode="hybrid", mmr_lambda=0.3)
    assert ids[0].startswith("a__") and ids[1] == "b__0000"


def test_min_score_can_empty_the_results():
    r = _retriever()
    assert r.search("evidential", k=3, mode="dense", min_score=2.0) == []
    assert r.search("evidential", k=3, mode="dense", min_score=None)


def test_chat_refuses_without_llm_when_nothing_clears_the_floor():
    llm = test_chat.ScriptedLLM([])
    session = chat.ChatSession(_retriever(), llm,
                               search_kwargs={"mode": "dense", "min_score": 2.0})
    assert session.ask("evidential?").answer == REFUSAL and llm.calls == []


# --- small talk --------------------------------------------------------------------


@pytest.mark.parametrize("text", ["hi", "Hello!", "hey there", "thanks", "Thank you so much.",
                                  "ok", "good morning"])
def test_small_talk_gets_canned_reply(text):
    assert chat.small_talk_reply(text)


@pytest.mark.parametrize("text", ["how are you?", "who are you", "what can you do?",
                                  "What is this?"])
def test_questions_about_the_assistant_are_answered_directly(text):
    assert chat.small_talk_reply(text) == chat.ABOUT_REPLY


@pytest.mark.parametrize("text", ["hi, what is EDL?", "how does OE work",
                                  "ok so what is PRR", "how are ensembles trained?",
                                  "what can you do with a Dirichlet prior?"])
def test_questions_are_not_small_talk(text):
    assert chat.small_talk_reply(text) is None


def test_greeting_skips_retrieval_llm_and_history():
    class NoSearch:
        def search(self, *a, **k):
            raise AssertionError("retrieval should not run")
    llm = test_chat.ScriptedLLM([])
    session = chat.ChatSession(NoSearch(), llm)
    turn = session.ask("hi")
    assert turn.answer == chat.GREETING_REPLY and llm.calls == [] and session.turns == []


# --- Groq limits ---------------------------------------------------------------------


def _groq_client(fail_models, retry_after):
    groq = pytest.importorskip("groq")
    import httpx

    calls = []

    class Completions:
        def create(self, model, messages, temperature):
            calls.append(model)
            if model in fail_models:
                resp = httpx.Response(429, headers={"retry-after": str(retry_after)},
                                      request=httpx.Request("POST", "https://x"))
                raise groq.RateLimitError(
                    "Rate limit reached on tokens per day (TPD)", response=resp, body=None)
            msg = type("M", (), {"content": f"answer from {model}"})
            return type("R", (), {"choices": [type("C", (), {"message": msg})]})

    client = type("Client", (), {})()
    client.chat = type("Chat", (), {})()
    client.chat.completions = Completions()
    return client, calls


def test_daily_limit_fails_fast_instead_of_sleeping():
    import llm
    client, calls = _groq_client({"big"}, retry_after=1798)
    with pytest.raises(llm.RateLimitTooLong) as e:
        llm.call_groq_messages(client, "big", [], max_wait=90)
    assert e.value.daily and calls == ["big"]
    assert "30 min" in str(e.value)


def test_fallback_model_answers_when_daily_limit_hit():
    import llm
    client, calls = _groq_client({"big"}, retry_after=1798)
    out = llm.call_groq_messages(client, "big", [], max_wait=90, fallback_models=("small",))
    assert out == "answer from small" and calls == ["big", "small"]


def test_exit_on_rate_limit_turns_error_into_message():
    import llm

    @llm.exit_on_rate_limit
    def main():
        raise llm.RateLimitTooLong("big", 600, True)
    with pytest.raises(SystemExit) as e:
        main()
    assert "daily limit" in str(e.value.code)


# --- sentence and semantic chunking ---------------------------------------------------

TEXT = ("Evidential networks place a Dirichlet over class probabilities. They are "
        "trained with a single forward pass. Ensembles average several networks. "
        "They cost more at test time. Outlier exposure adds an auxiliary dataset. "
        "It teaches the model to flag anomalies.")


def test_sentence_spans():
    spans = cp.sentence_spans(TEXT)
    assert len(spans) == 6 and TEXT[slice(*spans[1])].startswith("They are trained")
    assert len(cp.sentence_spans("See Fig. 3 and Eq. (2) for details.")) == 1


def _chunks(strategy_breaks=None, size=20, overlap=0, text=TEXT):
    spans = [(0, len(text) + 1, 1)]
    return list(cp.chunk_text_by_sentences(text, fake_tokenizer, size, overlap, spans,
                                           breaks=strategy_breaks))


def test_sentence_chunks_end_on_sentence_boundaries():
    out = _chunks(size=20)
    assert len(out) > 1
    for c in out:
        assert c["text"].endswith("."), c["text"]
        assert c["n_tokens"] <= 20
    assert " ".join(c["text"] for c in out) == TEXT


def test_sentence_overlap_repeats_whole_sentences():
    out = _chunks(size=20, overlap=8)
    assert any(out[i + 1]["text"].split(". ")[0] in out[i]["text"]
               for i in range(len(out) - 1))


def test_sentence_longer_than_chunk_falls_back_to_windows():
    long = " ".join(["7.3 ±0.2"] * 40) + "."
    out = _chunks(size=16, text=long)
    assert len(out) >= 5 and all(c["n_tokens"] <= 16 for c in out)


def test_semantic_breaks_split_at_topic_shift():
    topic = {"Evidential": 0, "They are trained": 0, "Ensembles": 1, "They cost": 1,
             "Outlier": 2, "It teaches": 2}

    def embed(windows):
        out = []
        for w in windows:
            v = np.zeros(3)
            for key, t in topic.items():
                v[t] += w.count(key)
            out.append(v / np.linalg.norm(v))
        return np.array(out)
    breaks = cp.semantic_breaks(TEXT, embed, percentile=50, buffer=0)
    assert breaks == {1, 3}
    out = list(cp.chunk_text_by_sentences(TEXT, fake_tokenizer, 100, 0,
                                          [(0, len(TEXT) + 1, 1)], breaks=breaks))
    # min_tokens=64 keeps these short topics together; lower it to see the split
    assert len(out) == 1
    units = cp._token_units(TEXT, fake_tokenizer, 100)
    groups = list(cp.pack_units(units, 100, 0, breaks, min_tokens=1))
    assert [(units[a][4], units[b][4]) for a, b in groups] == [(0, 1), (2, 3), (4, 5)]


# --- batch runs survive an unavailable provider -------------------------------------


def test_unavailable_chunk_is_left_for_the_next_run(tmp_path):
    import llm

    chunks = [CHUNKS["1905.00076__0012"], CHUNKS["1905.00076__0013"]]
    replies = iter([llm.ProviderUnavailable("503"), _reply()])

    def complete(system, user):
        r = next(replies)
        if isinstance(r, Exception):
            raise r
        return r

    failures = []
    path = tmp_path / "n.json"
    notes = tn.generate_notes(chunks, complete, "m", path, log=lambda *_: None,
                              failures=failures)
    assert failures == ["1905.00076__0012"]
    assert set(notes) == {"1905.00076__0013"}, "the failed one has no notes yet"
    rerun = []
    tn.generate_notes(chunks, lambda s, u: rerun.append(u) or _reply(), "m", path,
                      log=lambda *_: None)
    assert len(rerun) == 1, "a plain rerun picks up only the skipped chunk"


def test_run_stops_after_three_unavailable_in_a_row(tmp_path):
    import llm

    many = [dict(CHUNKS["1905.00076__0012"], chunk_id=f"x__{i:04d}") for i in range(6)]
    calls = []

    def down(system, user):
        calls.append(1)
        raise llm.ProviderUnavailable("down")

    failures = []
    tn.generate_notes(many, down, "m", tmp_path / "n.json", log=lambda *_: None,
                      failures=failures)
    assert len(calls) == 3 and len(failures) == 3


def test_stats_survive_notes_whose_failing_replies_were_all_empty(capsys):
    """_print_stats looked for an example failing reply and assumed one
    existed. When the model answers with nothing at all, reply_start is "",
    so the search found none and StopIteration killed the command."""
    notes = {"c__0001": {"text_hash": "x", "prompt_version": tn.PROMPT_VERSION,
                         "model": "m", "tables": [
                             {"caption": "", "description": "", "markdown": "",
                              "markdown_ok": False,
                              "reject_reason": "reply did not follow the format",
                              "reply_start": ""}]}}
    tn._print_stats(notes)
    assert "EMPTY" in capsys.readouterr().out
