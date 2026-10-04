"""
LLM access shared by generation, chat, query expansion, table notes and
evaluation.

Two providers (see providers.py): `groq` by default, `gemini` for batch work
over the corpus and for anything needing vision. Everything downstream uses
make_complete_fn / make_chat_fn, so switching provider is a flag, not a code
change.

Kept separate from generate.py so that modules which need an LLM don't have to
import the whole generation pipeline, and so there is exactly one place that
handles rate limits.
"""

from __future__ import annotations

import functools
import os
import re
import sys
import time
from typing import Callable

from dotenv import load_dotenv

# Read .env now, not only when a client is created: RAG_PROVIDER, GROQ_MAX_WAIT
# and GEMINI_MIN_INTERVAL below are module-level defaults. Variables already set
# in the shell win (load_dotenv never overrides them).
load_dotenv(".env")

# Groq deprecated llama-3.3-70b-versatile on the free tier (June 2026).
# gpt-oss-120b is the recommended free replacement. Override with --model.
DEFAULT_LLM = "openai/gpt-oss-120b"
# Smaller model with its own, separate free-tier quota. Chat switches to it when
# the default model's daily token limit is used up.
FALLBACK_LLM = "openai/gpt-oss-20b"
MAX_RETRIES = 5
# Waits longer than this are not a per-minute limit but the per-day one (Groq
# asked for 386s and then 1798s after a day of query generation). Sleeping
# half an hour inside a chat is worse than failing with a clear message, so
# longer waits raise RateLimitTooLong instead. Override with GROQ_MAX_WAIT.
MAX_WAIT = float(os.environ.get("GROQ_MAX_WAIT", "90"))

PROVIDERS = ("groq", "gemini")
DEFAULT_PROVIDER = os.environ.get("RAG_PROVIDER", "groq")


def default_model(provider: str) -> str:
    from providers import GEMINI_DEFAULT

    return {"groq": DEFAULT_LLM, "gemini": GEMINI_DEFAULT}[provider]


def default_fallback(provider: str) -> str:
    from providers import GEMINI_FALLBACK

    return {"groq": FALLBACK_LLM, "gemini": GEMINI_FALLBACK}[provider]


def get_backend(provider: str = DEFAULT_PROVIDER, **kwargs):
    """A Backend for this provider (see providers.py)."""
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")
    import providers as _p

    return (_p.GroqBackend if provider == "groq" else _p.GeminiBackend)(**kwargs)


class ProviderUnavailable(RuntimeError):
    """The provider kept failing on its side - busy (5xx) or the connection
    dropping - through every retry and fallback. Transient: batch jobs record
    the item as failed and move on, and the same command later retries it."""


class EmptyCompletion(ProviderUnavailable):
    """The provider accepted the request and returned no content at all.

    Groq's gpt-oss models do this: the reply arrives with message.content
    empty (the model can spend its budget on the reasoning channel). No API
    error is raised, so an earlier version handed the "" straight back as the
    model's answer. That is how one table-notes run recorded 34 chunks as
    permanently having no table: the daily limit on the big model pushed every
    call onto the small fallback, which returned nothing, and the emptiness
    was cached as a result. A rate limit degraded into corrupt data.

    Subclasses ProviderUnavailable so batch jobs already skip the item and
    retry it on the next run instead of storing the gap.
    """


class RateLimitTooLong(RuntimeError):
    """The provider asked for a wait longer than MAX_WAIT (usually its daily
    quota, which is not worth sleeping through)."""

    def __init__(self, model: str, wait: float, daily: bool, provider: str = "Groq"):
        self.model, self.wait, self.daily, self.provider = model, wait, daily, provider
        kind = "daily limit" if daily else "rate limit"
        other = (f"--model {FALLBACK_LLM}" if provider.lower() == "groq"
                 else "--provider groq")
        # A daily quota often reports a short retry-after, and rounding that to
        # minutes printed "it asks to wait 0 min", which reads as a bug rather
        # than as "you are out of quota for today".
        if wait >= 60:
            how_long = f"it asks to wait {wait / 60:.0f} min"
        elif daily:
            # a daily quota often reports only a token retry-after
            how_long = "the quota resets on the provider's daily schedule"
        else:
            how_long = f"it asks to wait {wait:.0f}s"
        super().__init__(
            f"{provider} {kind} reached for {model}: {how_long}. Try again "
            f"later, or switch ({other}); each model and provider has its own "
            "quota.")


def _is_daily_limit(err) -> bool:
    msg = str(err).lower()
    return "per day" in msg or bool(re.search(r"\b(tpd|rpd)\b", msg))


def add_provider_args(ap, fallback: bool = False):
    """--provider / --model (and optionally --fallback-model) for a CLI."""
    ap.add_argument("--provider", choices=PROVIDERS, default=DEFAULT_PROVIDER,
                    help=f"LLM provider (default {DEFAULT_PROVIDER}; set "
                         "RAG_PROVIDER to change the default).")
    ap.add_argument("--model", default=None,
                    help="Model id; defaults to the provider's free-tier model.")
    if fallback:
        ap.add_argument("--fallback-model", default=None,
                        help="Model used when --model's limit is reached "
                             "('none' to disable).")


def resolve(args) -> tuple[str, str, tuple[str, ...]]:
    """(provider, model, fallback_models) from parsed --provider/--model args."""
    provider = getattr(args, "provider", DEFAULT_PROVIDER)
    model = args.model or default_model(provider)
    fb = getattr(args, "fallback_model", None)
    if fb is None:
        fb = default_fallback(provider)
    fallbacks = () if fb in ("", "none") or fb == model else (fb,)
    return provider, model, fallbacks


def exit_on_rate_limit(main_fn):
    """Decorator for command-line entry points: print the RateLimitTooLong
    message and exit, instead of a traceback."""
    @functools.wraps(main_fn)
    def wrapper(*args, **kwargs):
        try:
            return main_fn(*args, **kwargs)
        except RateLimitTooLong as e:
            sys.exit(f"\n{e}")
    return wrapper


def get_client():
    """Groq client using GROQ_API_KEY from .env or the environment."""
    load_dotenv(".env")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found. Put it in .env or export it.", file=sys.stderr)
        sys.exit(1)
    from groq import Groq

    return Groq(api_key=api_key)


def _retry_after_seconds(err, attempt: int) -> float:
    """Honor the server's retry-after header if present, else exponential backoff.

    Groq free tier can block all requests for ~60s after a 429, and only sets
    retry-after when the limit is actually hit, so we prefer the header and fall
    back to a backoff starting at a safe minimum.
    """
    retry_after = None
    try:
        retry_after = err.response.headers.get("retry-after")
    except Exception:  # noqa: BLE001
        retry_after = None
    if retry_after is not None:
        try:
            return float(retry_after) + 1.0  # small cushion
        except ValueError:
            pass
    return min(60.0, max(2.0, 2 ** attempt * 2))


def _call_one(client, model, messages, temperature, max_wait):
    from groq import APIConnectionError, APIStatusError, RateLimitError

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text
            # Retrying the same model is pointless at temperature 0 - it is
            # deterministic - so hand straight over to the next model.
            raise EmptyCompletion(f"{model} returned an empty reply")
        except RateLimitError as e:
            wait = _retry_after_seconds(e, attempt)
            if wait > max_wait:
                raise RateLimitTooLong(model, wait, _is_daily_limit(e)) from e
            if attempt < MAX_RETRIES - 1:
                print(f"  (rate limited, waiting {wait:.0f}s then retrying ...)",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except APIConnectionError:
            if attempt < MAX_RETRIES - 1:
                wait = min(30.0, 2 ** attempt * 2)
                print(f"  (connection error, retrying in {wait:.0f}s ...)",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            raise ProviderUnavailable(f"Groq unreachable for {model}")
        except APIStatusError as e:
            # 5xx is the provider having a bad moment ("experiencing high
            # demand"), not a problem with the request: back off and retry.
            if getattr(e, "status_code", 0) < 500:
                raise
            if attempt >= MAX_RETRIES - 1:
                raise ProviderUnavailable(
                    f"Groq kept failing for {model} ({e.status_code})") from e
            wait = min(30.0, 2 ** attempt * 3)
            print(f"  ({model} is busy ({e.status_code}); retrying in "
                  f"{wait:.0f}s ...)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("exhausted retries calling Groq")


def call_groq_messages(client, model, messages, temperature=0.0,
                       max_wait: float | None = None,
                       fallback_models: tuple[str, ...] = ()):
    """Chat completion over a full message list, honoring rate limits.

    Short 429 waits (the per-minute limits) are slept through using the
    server's retry-after header, with exponential fallback; connection errors
    are retried with backoff. A wait longer than max_wait raises
    RateLimitTooLong - unless fallback_models are given, in which case the
    next model is tried (each Groq model has its own quota).
    """
    max_wait = MAX_WAIT if max_wait is None else max_wait
    models = [model, *[m for m in fallback_models if m and m != model]]
    for i, m in enumerate(models):
        try:
            return _call_one(client, m, messages, temperature, max_wait)
        except RateLimitTooLong as e:
            if i == len(models) - 1:
                raise
            print(f"  ({e.model} {'daily limit' if e.daily else 'rate limit'} "
                  f"reached; answering with {models[i + 1]} instead)", file=sys.stderr)
        except EmptyCompletion:
            if i == len(models) - 1:
                raise
            print(f"  ({m} returned an empty reply; trying {models[i + 1]})",
                  file=sys.stderr)
    raise AssertionError("unreachable")


def call_groq(client, model, system_prompt, user_prompt, temperature=0.0,
              **kwargs):
    """Single-turn convenience wrapper: one system and one user message."""
    return call_groq_messages(
        client, model,
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        temperature, **kwargs,
    )


def make_chat_fn(model: str | None = None, temperature: float = 0.0,
                 fallback_models: tuple[str, ...] = (),
                 provider: str = DEFAULT_PROVIDER,
                 backend=None) -> Callable[[list[dict]], str]:
    """A (messages) -> text function bound to one model, for multi-turn chat."""
    backend = backend or get_backend(provider)
    model = model or default_model(backend.name)
    return lambda messages: backend.complete(model, messages, temperature,
                                             fallback_models=fallback_models)


def make_complete_fn(model: str | None = None, temperature: float = 0.0,
                     fallback_models: tuple[str, ...] = (),
                     provider: str = DEFAULT_PROVIDER,
                     backend=None) -> Callable[[str, str], str]:
    """A (system_prompt, user_prompt) -> text function bound to one model.

    No fallback by default: evaluation and query generation should use one
    model throughout, or their outputs are not comparable.
    """
    chat = make_chat_fn(model, temperature, fallback_models, provider, backend)
    return lambda system, user: chat([{"role": "system", "content": system},
                                      {"role": "user", "content": user}])
