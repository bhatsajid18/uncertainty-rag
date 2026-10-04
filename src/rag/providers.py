"""
LLM backends: Groq (default) and Gemini.

One interface, so every caller - generation, chat, query expansion, table
notes, figure descriptions, evaluation - works with either:

    backend = get_backend("gemini")
    backend.complete(model, messages, temperature) -> str
    backend.describe_image(model, image_path, system, prompt) -> str

Why two. Groq is fast and its limit is tokens per day (200k on
gpt-oss-120b, which one table-notes run uses up). Gemini's free tier limits
requests per day instead (a few hundred), which suits batch jobs over the
corpus, and it accepts images, which is what figure descriptions need. Groq
has no vision model here, so `describe_image` is Gemini-only.

Both backends raise llm.RateLimitTooLong when the service asks for a wait
longer than MAX_WAIT, rather than sleeping through a daily quota.

Keys live in .env: GROQ_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY).
Gemini needs `pip install google-genai`.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time

from dotenv import load_dotenv

# Free-tier models. Flash is the default; Flash-Lite has a larger daily request
# allowance and is the fallback. Google retires ids without warning - "gemini-2.5
# -flash is no longer available to new users. Please update your code to use
# models/gemini-3.6-flash" - so a 404 is not fatal here: the backend takes the
# replacement out of the error, or picks the closest model the account can
# actually list, and says which it used.
# The "-latest" aliases are used rather than a pinned version, because the
# pinned ones come and go (gemini-2.5-flash was retired mid-project, and this
# account lists gemini-3.6-flash but no gemini-3.6-flash-lite). Pass --model
# with a pinned id for a run that has to stay reproducible.
GEMINI_DEFAULT = "gemini-flash-latest"
GEMINI_FALLBACK = "gemini-flash-lite-latest"
# Free tier allows roughly 10 requests/minute, so calls are spaced out rather
# than fired and retried after a 429.
load_dotenv(".env")  # so GEMINI_MIN_INTERVAL can be set there too
GEMINI_MIN_INTERVAL = float(os.environ.get("GEMINI_MIN_INTERVAL", "6.5"))
_RETRY_DELAY = re.compile(r"retry[- ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.I)
_SUGGESTED = re.compile(r"use\s+models/([A-Za-z0-9.\-]+)")


class Backend:
    """Common behaviour: request pacing and a uniform message format.

    messages are OpenAI-style [{"role": "system"|"user"|"assistant",
    "content": str}], because that is what the rest of the project already
    passes around.
    """

    name = "backend"

    def __init__(self, min_interval: float = 0.0):
        self.min_interval = min_interval
        self._last_call = 0.0

    def _pace(self):
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def complete(self, model, messages, temperature=0.0, fallback_models=()) -> str:
        raise NotImplementedError

    def describe_image(self, model, image_path, system, prompt) -> str:
        raise NotImplementedError(
            f"{self.name} has no vision support here; use --provider gemini")


class GroqBackend(Backend):
    name = "groq"

    def __init__(self, client=None, min_interval: float = 0.0):
        super().__init__(min_interval)
        if client is None:
            from llm import get_client

            client = get_client()
        self.client = client

    def complete(self, model, messages, temperature=0.0, fallback_models=()) -> str:
        from llm import call_groq_messages

        self._pace()
        return call_groq_messages(self.client, model, messages, temperature,
                                  fallback_models=tuple(fallback_models))


def gemini_parts(messages: list[dict]) -> tuple[str, list]:
    """Split OpenAI-style messages into (system instruction, Gemini contents)."""
    from google.genai import types

    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user",
                      parts=[types.Part.from_text(text=m["content"])])
        for m in messages if m["role"] != "system"
    ]
    return system, contents


def gemini_retry_after(err) -> float | None:
    """Seconds Gemini asks us to wait, from the error body if it says."""
    m = _RETRY_DELAY.search(str(err))
    return float(m.group(1)) if m else None


def _is_rate_limit(err) -> bool:
    return "429" in str(err) or "RESOURCE_EXHAUSTED" in str(err)


def _is_unknown_model(err) -> bool:
    return "404" in str(err) or "NOT_FOUND" in str(err)


_SERVER_ERROR = re.compile(r"^\s*50[0234]\b|\b(UNAVAILABLE|INTERNAL)\b")


_NETWORK_ERRORS = ("ConnectError", "ConnectTimeout", "ReadTimeout", "ReadError",
                   "WriteError", "RemoteProtocolError", "SSLError", "TimeoutError",
                   "ConnectionError", "ConnectionResetError")


def _is_network_error(err) -> bool:
    """The connection failed rather than the request ("[SSL:
    UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol").
    Checked by class name, because httpx, httpcore and the ssl module each
    raise their own types and the SDK does not wrap them."""
    return any(type(e).__name__ in _NETWORK_ERRORS
               for e in (err, err.__cause__, err.__context__) if e is not None)


def _is_server_error(err) -> bool:
    """A fault on Google's side, not ours: worth retrying, then worth trying
    another model ("503 UNAVAILABLE. This model is currently experiencing high
    demand"). Matched at the start of the message or on the status word, so a
    quota message that happens to contain "500" is not mistaken for one."""
    return bool(_SERVER_ERROR.search(str(err)))


def suggested_model(err) -> str | None:
    """The replacement Google names in a 404, if it names one."""
    m = _SUGGESTED.search(str(err))
    return m.group(1) if m else None


def pick_model(available: list[str], wanted: str) -> str | None:
    """Closest listed model to one that came back 404.

    Same family (flash, or flash-lite when that is what was asked for), then
    the highest version number, with previews and experiments last.
    """
    lite = "lite" in wanted
    same = [m for m in available if "flash" in m and ("lite" in m) == lite]
    pool = same or [m for m in available if "flash" in m] or list(available)

    def key(name: str):
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", name)] or [0.0]
        return ("preview" in name or "exp" in name, [-n for n in nums])

    return sorted(pool, key=key)[0] if pool else None


class GeminiBackend(Backend):
    name = "gemini"

    def __init__(self, api_key: str | None = None,
                 min_interval: float = GEMINI_MIN_INTERVAL, client=None):
        super().__init__(min_interval)
        if client is not None:
            self.client = client
            return
        load_dotenv(".env")
        api_key = api_key or os.environ.get("GEMINI_API_KEY") \
            or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            sys.exit("GEMINI_API_KEY not found. Get one at "
                     "https://aistudio.google.com/apikey and put it in .env")
        try:
            from google import genai
        except ImportError:
            sys.exit("The Gemini provider needs the SDK: pip install google-genai")
        # the SDK logs an automatic-function-calling notice on every call; no
        # tools are passed here, so it is noise
        logging.getLogger("google_genai.models").setLevel(logging.ERROR)
        self.client = genai.Client(api_key=api_key)

    def list_models(self) -> list[str]:
        """Model ids this account can call for text generation."""
        out = []
        for m in self.client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None)
            if name and (actions is None or "generateContent" in actions):
                out.append(name)
        return out

    def _replacement_for(self, name: str, err) -> str | None:
        """What to call instead of a model that came back 404."""
        return suggested_model(err) or pick_model(self.list_models(), name)

    def _generate(self, model, contents, system, temperature, fallback_models=()):
        from google.genai import types

        from llm import (
            MAX_RETRIES, MAX_WAIT, EmptyCompletion, ProviderUnavailable,
            RateLimitTooLong,
        )

        config = types.GenerateContentConfig(temperature=temperature)
        if system:
            config.system_instruction = system
        models = [model, *[m for m in fallback_models if m and m != model]]
        tried: set[str] = set()
        m_i = 0
        while m_i < len(models):
            name = models[m_i]
            for attempt in range(MAX_RETRIES):
                self._pace()
                try:
                    text = self.client.models.generate_content(
                        model=name, contents=contents, config=config).text or ""
                    if not text.strip():
                        # same trap as Groq's gpt-oss models: a reply with no
                        # content is not an answer, and must not be cached
                        raise EmptyCompletion(f"{name} returned an empty reply")
                    return text
                except EmptyCompletion:
                    if m_i < len(models) - 1:
                        print(f"  ({name} returned an empty reply; trying "
                              f"{models[m_i + 1]})", file=sys.stderr)
                        break
                    raise
                except Exception as e:  # noqa: BLE001 - SDK error types vary
                    if _is_unknown_model(e):
                        tried.add(name)
                        swap = self._replacement_for(name, e)
                        if swap and swap not in tried:
                            print(f"  ({name} is not available on this account; "
                                  f"using {swap}. Pass --model to choose.)",
                                  file=sys.stderr)
                            models[m_i] = name = swap
                            continue
                        raise RuntimeError(
                            f"Gemini has no model '{name}' for this account. "
                            "Available: " + ", ".join(self.list_models()[:12])
                            + ". Pick one with --model.") from e
                    if _is_rate_limit(e):
                        wait = gemini_retry_after(e) or min(60.0, 2 ** attempt * 4)
                        daily = "PerDay" in str(e) or "per day" in str(e).lower()
                        if wait > MAX_WAIT or daily:
                            if m_i < len(models) - 1:
                                print(f"  ({name} limit reached; switching to "
                                      f"{models[m_i + 1]})", file=sys.stderr)
                                break
                            raise RateLimitTooLong(name, wait, daily,
                                                   provider="Gemini") from e
                        print(f"  (rate limited, waiting {wait:.0f}s then "
                              "retrying ...)", file=sys.stderr)
                        time.sleep(wait)
                        continue
                    network = _is_network_error(e)
                    if network or _is_server_error(e):
                        # Google's side is busy, or the connection dropped.
                        # Back off, then let the next model in the chain try.
                        why = "connection dropped" if network else "server busy"
                        if attempt < MAX_RETRIES - 1:
                            wait = min(30.0, 2 ** attempt * 3)
                            print(f"  ({name}: {why}; retrying in {wait:.0f}s ...)",
                                  file=sys.stderr)
                            time.sleep(wait)
                            continue
                        if m_i < len(models) - 1:
                            print(f"  ({name}: still {why}; trying "
                                  f"{models[m_i + 1]})", file=sys.stderr)
                            break
                        raise ProviderUnavailable(
                            f"Gemini could not serve {', '.join(models)} ({why})."
                        ) from e
                    raise
            m_i += 1
        raise AssertionError("unreachable")

    def complete(self, model, messages, temperature=0.0, fallback_models=()) -> str:
        system, contents = gemini_parts(messages)
        return self._generate(model, contents, system, temperature, fallback_models)

    def describe_image(self, model, image_path, system, prompt,
                       temperature=0.0, fallback_models=()) -> str:
        """One image plus a prompt. Used for figure descriptions."""
        from pathlib import Path

        from google.genai import types

        data = Path(image_path).read_bytes()
        contents = [types.Content(role="user", parts=[
            types.Part.from_bytes(data=data, mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ])]
        return self._generate(model, contents, system, temperature, fallback_models)


def main():
    """`python src/rag/providers.py --list` - what this account can actually call."""
    import argparse

    ap = argparse.ArgumentParser(description="Inspect an LLM provider.")
    ap.add_argument("--provider", default="gemini", choices=("groq", "gemini"))
    ap.add_argument("--list", action="store_true", help="List available models.")
    args = ap.parse_args()

    if args.provider == "gemini":
        models = GeminiBackend().list_models()
    else:
        from llm import get_client

        skip = ("whisper", "tts", "guard", "orpheus")
        models = sorted(m.id for m in get_client().models.list().data
                        if not any(s in m.id.lower() for s in skip))
    print(f"{len(models)} model(s) on {args.provider}:")
    for m in models:
        print(f"  {m}")
    print("\nUse one with --model <id>.")


if __name__ == "__main__":
    main()
