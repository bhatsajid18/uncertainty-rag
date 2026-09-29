"""Tests for the provider layer (Groq and Gemini) and figure extraction.

The Gemini SDK is stubbed, so these run offline and without the package
installed; what is checked is our side of the contract - message conversion,
pacing, rate-limit handling and fallback.
"""

import json
import sys
import types
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rag"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_chat  # noqa: E402,F401  (installs module stubs)

import figures as F  # noqa: E402
import llm  # noqa: E402
import providers  # noqa: E402


# --- a stand-in for google.genai --------------------------------------------------


class _Part:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    @classmethod
    def from_text(cls, text):
        return cls(text=text)

    @classmethod
    def from_bytes(cls, data, mime_type):
        return cls(data=data, mime_type=mime_type)


class _Content:
    def __init__(self, role, parts):
        self.role, self.parts = role, parts


class _Config:
    def __init__(self, temperature=0.0):
        self.temperature, self.system_instruction = temperature, None


@pytest.fixture(autouse=True)
def fake_genai(monkeypatch):
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai_types = types.ModuleType("google.genai.types")
    genai_types.Part, genai_types.Content = _Part, _Content
    genai_types.GenerateContentConfig = _Config
    genai.types = genai_types
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)
    return genai


class _Model:
    def __init__(self, name):
        self.name = f"models/{name}"
        self.supported_actions = ["generateContent"]


class FakeGeminiClient:
    """Fails for `failing` models with the given error, answers otherwise."""

    def __init__(self, failing=(), error="429 RESOURCE_EXHAUSTED {'retryDelay': '7s'}",
                 catalog=("gemini-3.6-flash", "gemini-3.6-flash-lite",
                          "gemini-2.0-flash", "gemini-3.6-pro-preview")):
        self.calls, self.failing, self.error = [], set(failing), error
        self.catalog = catalog
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        if model in self.failing:
            raise RuntimeError(self.error)
        return types.SimpleNamespace(text=f"answer from {model}")

    def list(self):
        return [_Model(n) for n in self.catalog]


def backend(**kw):
    client = FakeGeminiClient(**kw)
    return providers.GeminiBackend(client=client, min_interval=0.0), client


# --- message conversion ------------------------------------------------------------


def test_system_and_turns_are_converted():
    b, client = backend()
    out = b.complete("gemini-2.5-flash", [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ])
    assert out == "answer from gemini-2.5-flash"
    model, contents, config = client.calls[0]
    assert config.system_instruction == "rules"
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert [c.parts[0].text for c in contents] == ["q1", "a1", "q2"]


def test_complete_fn_works_through_the_backend():
    b, client = backend()
    complete = llm.make_complete_fn("gemini-2.5-flash", backend=b)
    assert complete("sys", "user") == "answer from gemini-2.5-flash"
    _, contents, config = client.calls[0]
    assert config.system_instruction == "sys" and contents[0].parts[0].text == "user"


# --- rate limits -------------------------------------------------------------------


def test_short_wait_is_slept_through(monkeypatch):
    slept = []
    monkeypatch.setattr(providers.time, "sleep", slept.append)
    client = FakeGeminiClient(failing={"gemini-2.5-flash"})

    def stop_failing(*a, **k):
        client.failing.clear()
        return 7.0
    monkeypatch.setattr(providers, "gemini_retry_after", stop_failing)
    b = providers.GeminiBackend(client=client, min_interval=0.0)
    assert b.complete("gemini-2.5-flash", [{"role": "user", "content": "q"}])
    assert slept == [7.0]


def test_daily_limit_raises_rather_than_waiting():
    b, _ = backend(failing={"gemini-2.5-flash"},
                   error="429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProject")
    with pytest.raises(llm.RateLimitTooLong) as e:
        b.complete("gemini-2.5-flash", [{"role": "user", "content": "q"}])
    assert e.value.daily and e.value.provider == "Gemini"
    assert "--provider groq" in str(e.value)


def test_fallback_model_is_used():
    b, client = backend(failing={"gemini-2.5-flash"},
                        error="429 RESOURCE_EXHAUSTED PerDay")
    out = b.complete("gemini-2.5-flash", [{"role": "user", "content": "q"}],
                     fallback_models=("gemini-2.5-flash-lite",))
    assert out == "answer from gemini-2.5-flash-lite"
    assert [c[0] for c in client.calls] == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]


def test_retry_delay_is_read_from_the_error():
    assert providers.gemini_retry_after(Exception("'retryDelay': '27s'")) == 27.0
    assert providers.gemini_retry_after(Exception("retry-delay = 5.5s")) == 5.5
    assert providers.gemini_retry_after(Exception("no delay here")) is None


def test_requests_are_paced(monkeypatch):
    slept = []
    monkeypatch.setattr(providers.time, "sleep", slept.append)
    monkeypatch.setattr(providers.time, "monotonic", lambda: 100.0)
    b = providers.GeminiBackend(client=FakeGeminiClient(), min_interval=6.5)
    for _ in range(3):
        b.complete("gemini-2.5-flash", [{"role": "user", "content": "q"}])
    assert slept == [pytest.approx(6.5), pytest.approx(6.5)], "first call not delayed"


# --- model catalogue changes -------------------------------------------------------


RETIRED = ("404 NOT_FOUND. {'error': {'code': 404, 'message': 'This model "
           "models/gemini-2.5-flash is no longer available to new users. Please "
           "update your code to use models/gemini-3.6-flash for the latest "
           "features and improvements.', 'status': 'NOT_FOUND'}}")


def test_retired_model_is_swapped_for_the_one_google_names(capsys):
    b, client = backend(failing={"gemini-2.5-flash"}, error=RETIRED)
    out = b.complete("gemini-2.5-flash", [{"role": "user", "content": "q"}])
    assert out == "answer from gemini-3.6-flash"
    assert [c[0] for c in client.calls] == ["gemini-2.5-flash", "gemini-3.6-flash"]
    assert "using gemini-3.6-flash" in capsys.readouterr().err


def test_unknown_model_falls_back_to_the_account_catalogue():
    b, client = backend(failing={"gemini-9-flash"},
                        error="404 NOT_FOUND. model not found")
    assert b.complete("gemini-9-flash", [{"role": "user", "content": "q"}]) == \
        "answer from gemini-3.6-flash"


def test_no_usable_model_says_what_is_available():
    b, _ = backend(failing={"gemini-3.6-flash", "gemini-3.6-flash-lite",
                            "gemini-2.0-flash", "gemini-3.6-pro-preview"},
                   error="404 NOT_FOUND. model not found")
    with pytest.raises(RuntimeError, match="gemini-3.6-flash"):
        b.complete("gemini-3.6-flash", [{"role": "user", "content": "q"}])


def test_pick_model_prefers_newest_non_preview_of_the_same_family():
    catalog = ["gemini-2.0-flash", "gemini-3.6-flash", "gemini-3.6-flash-lite",
               "gemini-3.6-pro", "gemini-4.0-flash-preview"]
    assert providers.pick_model(catalog, "gemini-2.5-flash") == "gemini-3.6-flash"
    assert providers.pick_model(catalog, "gemini-2.5-flash-lite") == \
        "gemini-3.6-flash-lite"
    assert providers.pick_model([], "gemini-2.5-flash") is None


def test_suggested_model_is_read_from_the_error():
    assert providers.suggested_model(Exception(RETIRED)) == "gemini-3.6-flash"
    assert providers.suggested_model(Exception("404 not found")) is None


# --- the provider having a bad moment ----------------------------------------------


BUSY = ("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is "
        "currently experiencing high demand. Spikes in demand are usually "
        "temporary. Please try again later.', 'status': 'UNAVAILABLE'}}")


def test_server_error_is_retried_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr(providers.time, "sleep", slept.append)
    client = FakeGeminiClient(failing={"gemini-flash-latest"}, error=BUSY)
    b = providers.GeminiBackend(client=client, min_interval=0.0)

    calls = {"n": 0}
    original = client.generate_content

    def flaky(model, contents, config):
        calls["n"] += 1
        if calls["n"] > 2:
            client.failing.clear()
        return original(model, contents, config)
    client.generate_content = flaky

    assert b.complete("gemini-flash-latest", [{"role": "user", "content": "q"}]) == \
        "answer from gemini-flash-latest"
    assert len(slept) == 2, "backed off twice, then got through"


def test_busy_model_hands_over_to_the_fallback(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)
    b, client = backend(failing={"gemini-flash-latest"}, error=BUSY)
    out = b.complete("gemini-flash-latest", [{"role": "user", "content": "q"}],
                     fallback_models=("gemini-flash-lite-latest",))
    assert out == "answer from gemini-flash-lite-latest"


def test_all_models_busy_says_so(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)
    b, _ = backend(failing={"gemini-flash-latest", "gemini-flash-lite-latest"},
                   error=BUSY)
    with pytest.raises(llm.ProviderUnavailable, match="server busy"):
        b.complete("gemini-flash-latest", [{"role": "user", "content": "q"}],
                   fallback_models=("gemini-flash-lite-latest",))


class _ConnectError(Exception):
    """Named like httpx's, which is how the backend recognises it."""


_ConnectError.__name__ = "ConnectError"


def test_dropped_connection_is_retried_like_a_busy_server(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)
    client = FakeGeminiClient()
    calls = {"n": 0}
    original = client.generate_content

    def drops_once(model, contents, config):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred "
                                "in violation of protocol")
        return original(model, contents, config)
    client.generate_content = drops_once
    b = providers.GeminiBackend(client=client, min_interval=0.0)
    assert b.complete("gemini-flash-latest", [{"role": "user", "content": "q"}])
    assert calls["n"] == 2


def test_network_error_wrapped_by_the_sdk_is_still_recognised():
    try:
        try:
            raise _ConnectError("EOF")
        except _ConnectError as inner:
            raise RuntimeError("request failed") from inner
    except RuntimeError as outer:
        assert providers._is_network_error(outer)
    assert not providers._is_network_error(ValueError("bad request"))


def test_giving_up_raises_provider_unavailable(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda *_: None)
    b, _ = backend(failing={"gemini-flash-latest"}, error=BUSY)
    with pytest.raises(llm.ProviderUnavailable):
        b.complete("gemini-flash-latest", [{"role": "user", "content": "q"}])


def test_quota_message_containing_500_is_not_a_server_error():
    quota = "429 RESOURCE_EXHAUSTED. Limit 500 requests per day. retryDelay: '30s'"
    assert providers._is_rate_limit(Exception(quota))
    assert not providers._is_server_error(Exception(quota))
    assert providers._is_server_error(Exception(BUSY))


def test_provider_defaults():
    assert llm.default_model("gemini") == providers.GEMINI_DEFAULT
    assert "latest" in providers.GEMINI_DEFAULT, "an alias, not a pinned version"
    assert llm.default_model("groq") == llm.DEFAULT_LLM
    with pytest.raises(ValueError):
        llm.get_backend("openai")


def test_groq_backend_has_no_vision():
    b = providers.GroqBackend(client=object())
    with pytest.raises(NotImplementedError, match="gemini"):
        b.describe_image("m", "x.png", "sys", "prompt")


# --- figures -----------------------------------------------------------------------


def make_figure_pdf(path, caption="Figure 2: The change of accuracy with respect to "
                                  "the uncertainty threshold for EDL."):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(60, 60, 540, 130),
                        "Results are shown in Figure 2 below, where accuracy rises "
                        "with the uncertainty threshold for EDL. " * 2, fontsize=9)
    page.draw_rect(pymupdf.Rect(100, 170, 400, 370), color=(0, 0, 0))
    for i in range(8):
        page.draw_line(pymupdf.Point(110 + i * 30, 360),
                       pymupdf.Point(140 + i * 30, 350 - i * 14), color=(1, 0, 0))
    page.insert_textbox(pymupdf.Rect(100, 380, 500, 410), caption, fontsize=9)
    doc.save(path)
    doc.close()


def test_figure_is_rendered_with_its_caption(tmp_path):
    pdf = tmp_path / "p.pdf"
    make_figure_pdf(pdf)
    [fig] = F.extract_figures(pdf, "1806.01768", tmp_path / "figures")
    assert fig["figure_no"] == "2" and fig["page"] == 1
    assert fig["caption"].startswith("Figure 2: The change of accuracy")
    image = Path(fig["image"])
    assert image.exists() and image.stat().st_size > 1000
    assert fig["height"] > 150, "the plot area, not just the caption"


def test_prose_mentioning_a_figure_is_not_a_caption():
    assert F.CAPTION.match("Figure 2: The change of accuracy with respect to")
    assert F.CAPTION.match("Fig. 3. Entropy of the predictive distribution")
    assert not F.CAPTION.match("Figure 2 below, where accuracy rises with the")
    assert not F.CAPTION.match("as shown in Figure 2: accuracy rises")


FIGURES = {"1806.01768": [{"page": 7, "figure_no": "2", "image": "data/figures/a.png",
                           "caption": "Figure 2: Accuracy against uncertainty threshold.",
                           "description": "A line plot with threshold on the x axis."}]}


def test_figure_note_goes_to_the_chunk_that_mentions_it():
    mentions = {"chunk_id": "a__0010", "arxiv_id": "1806.01768", "page_start": 8,
                "page_end": 8, "text": "As shown in Figure 2, accuracy rises."}
    same_page = {"chunk_id": "a__0011", "arxiv_id": "1806.01768", "page_start": 7,
                 "page_end": 7, "text": "Unrelated results discussion."}
    stats = F.attach_figures([mentions, same_page], FIGURES)
    assert "Accuracy against uncertainty threshold" in mentions["figure_note"]
    assert "line plot" in mentions["figure_note"], "description is indexed too"
    assert mentions["figure_images"] == "data/figures/a.png"
    assert "figure_note" not in same_page, "matched by reference, not by page"
    assert stats == {"figures": 1, "attached": 1, "described": 1}


def test_describe_figures_skips_the_done_ones(tmp_path):
    path = tmp_path / "figures.json"
    figs = {"p": [{"page": 1, "figure_no": "1", "caption": "Figure 1: x.",
                   "image": str(tmp_path / "a.png")},
                  {"page": 2, "figure_no": "2", "caption": "Figure 2: y.",
                   "image": str(tmp_path / "b.png"), "description": "already done"}]}
    calls = []

    class VisionBackend:
        def describe_image(self, model, image_path, system, prompt):
            calls.append((model, image_path, prompt))
            return "A bar chart comparing\n  three methods."

    F.describe_figures(figs, VisionBackend(), "gemini-2.5-flash", path,
                       log=lambda *_: None)
    assert len(calls) == 1 and calls[0][1].endswith("a.png")
    assert "Figure 1: x." in calls[0][2], "the caption is given to the vision model"
    assert figs["p"][0]["description"] == "A bar chart comparing three methods."
    assert json.loads(path.read_text())["p"][1]["description"] == "already done"


def test_describe_figures_skips_an_unavailable_figure(tmp_path):
    figs = {"p": [{"page": i, "figure_no": str(i), "caption": f"Figure {i}: x.",
                   "image": str(tmp_path / f"{i}.png")} for i in (1, 2, 3)]}
    replies = iter([llm.ProviderUnavailable("busy"), "A line plot.", "A bar chart."])

    class Flaky:
        def describe_image(self, *a, **k):
            r = next(replies)
            if isinstance(r, Exception):
                raise r
            return r

    failures = []
    F.describe_figures(figs, Flaky(), "m", tmp_path / "f.json", log=lambda *_: None,
                       failures=failures)
    assert failures == ["p figure 1"]
    assert "description" not in figs["p"][0], "left for the next run"
    assert figs["p"][2]["description"] == "A bar chart."


def test_describe_figures_stops_when_the_provider_is_down(tmp_path):
    figs = {"p": [{"page": i, "figure_no": str(i), "caption": f"Figure {i}: x.",
                   "image": str(tmp_path / f"{i}.png")} for i in range(1, 7)]}
    calls = []

    class Down:
        def describe_image(self, *a, **k):
            calls.append(1)
            raise llm.ProviderUnavailable("down")

    F.describe_figures(figs, Down(), "m", tmp_path / "f.json", log=lambda *_: None)
    assert len(calls) == 3, "stopped after three in a row, not six"


# --- an empty reply is not an answer ----------------------------------------------
#
# Regression: Groq's gpt-oss models can return message.content == "" with no
# API error. That "" was handed back as the model's answer, and a table-notes
# run cached it as "this chunk has no table" for 34 chunks - a rate limit (the
# big model's daily cap pushing every call onto the small one) turning into
# corrupt data rather than a visible failure.


class _Msg:
    def __init__(self, content):
        self.message = types.SimpleNamespace(content=content)


class FakeGroqClient:
    """Returns the queued content per model; None means an empty reply."""

    def __init__(self, by_model):
        self.by_model, self.calls = by_model, []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, model, messages, temperature):
        self.calls.append(model)
        return types.SimpleNamespace(choices=[_Msg(self.by_model.get(model))])


def test_empty_groq_reply_raises_instead_of_returning_it():
    client = FakeGroqClient({"big": None})
    with pytest.raises(llm.EmptyCompletion):
        llm.call_groq_messages(client, "big", [{"role": "user", "content": "hi"}])


def test_empty_reply_falls_back_to_the_next_model():
    client = FakeGroqClient({"big": None, "small": "a real answer"})
    out = llm.call_groq_messages(client, "big", [{"role": "user", "content": "hi"}],
                                 fallback_models=("small",))
    assert out == "a real answer"
    assert client.calls == ["big", "small"]


def test_whitespace_only_reply_counts_as_empty():
    client = FakeGroqClient({"big": "   \n  "})
    with pytest.raises(llm.EmptyCompletion):
        llm.call_groq_messages(client, "big", [{"role": "user", "content": "hi"}])


def test_empty_completion_is_skippable_by_batch_jobs():
    """Batch jobs catch ProviderUnavailable to skip and resume, so an empty
    reply must leave no note behind rather than caching the gap."""
    assert issubclass(llm.EmptyCompletion, llm.ProviderUnavailable)


def test_empty_gemini_reply_falls_back_to_the_next_model():
    client = FakeGeminiClient()
    original = client.generate_content

    def maybe_empty(model, contents, config):
        if model == "gemini-flash-latest":
            return types.SimpleNamespace(text="")
        return original(model, contents, config)

    client.generate_content = maybe_empty
    b = providers.GeminiBackend(client=client, min_interval=0.0)
    out = b.complete("gemini-flash-latest", [{"role": "user", "content": "hi"}],
                     fallback_models=("gemini-flash-lite-latest",))
    assert out == "answer from gemini-flash-lite-latest"


def test_daily_limit_message_does_not_say_zero_minutes():
    """A daily quota often reports a short retry-after; rounding it to minutes
    printed "it asks to wait 0 min", which reads as a bug."""
    msg = str(llm.RateLimitTooLong("m", 12.0, daily=True))
    assert "0 min" not in msg and "daily" in msg


def test_short_rate_limit_reports_seconds():
    assert "12s" in str(llm.RateLimitTooLong("m", 12.0, daily=False))


def test_long_rate_limit_still_reports_minutes():
    assert "6 min" in str(llm.RateLimitTooLong("m", 360.0, daily=False))
