"""
Multi-query expansion: ask an LLM for several rephrasings of a question, then
retrieve for each and fuse the ranked lists with Reciprocal Rank Fusion.

Why: dense and BM25 retrieval both depend on the user's wording. Research
papers often phrase the same idea differently ("FPR95" vs "false positive
rate at 95% true positive rate"), so a single phrasing can miss the chunk that
answers it. Rephrasings recover terminology the user didn't think to use.

Design notes:
- The retriever never talks to an LLM. This module produces a list of query
  strings, and HybridRetriever.search(..., query_variants=[...]) fuses them.
  That keeps retrieval testable without network access.
- The LLM call is injected as `complete_fn(system_prompt, user_prompt) -> str`,
  so this module has no Groq dependency and no circular import with
  generate.py.
- Expansions are cached on disk, keyed by (model, n, question). An evaluation
  sweep runs the same question through many configurations; without the cache
  each would re-call the LLM, burning free-tier quota and - worse - getting
  different rephrasings each time, which would make configurations
  incomparable.
- The ORIGINAL question is always kept as the first variant, and reranking
  (in hybrid_rerank mode) scores chunks against the original question only.
  Rephrasings widen the candidate net; they never redefine what is relevant.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable

EXPANSION_SYSTEM_PROMPT = (
    "You rewrite search queries for a retrieval system over machine learning "
    "research papers. Given a question, write alternative phrasings that a "
    "paper might use for the same idea. Rules:\n"
    "1. Keep the exact meaning of the question. Do not answer it.\n"
    "2. Vary the terminology: use synonyms, expand abbreviations (e.g. FPR95 -> "
    "false positive rate at 95% true positive rate), or use the abbreviation "
    "where the question spells a term out.\n"
    "3. If the question asks for a metric, also try the form papers report it "
    "in: accuracy is usually reported as classification error or error rate, "
    "and many results appear in tables with abbreviated dataset names (C10 for "
    "CIFAR-10, C100 for CIFAR-100).\n"
    "4. Output ONLY the rephrasings, one per line, with no numbering, bullets, "
    "quotes or commentary."
)

PROMPT_VERSION = hashlib.sha1(EXPANSION_SYSTEM_PROMPT.encode()).hexdigest()[:8]

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\(\d+\))\s*")


def parse_variants(raw: str, original: str, n: int) -> list[str]:
    """Clean an LLM response into at most n distinct rephrasings.

    Tolerates numbering, bullets and quotes even though the prompt forbids
    them, and drops blank lines, duplicates, and copies of the original.
    """
    seen = {original.strip().lower()}
    out: list[str] = []
    for line in raw.splitlines():
        text = _BULLET.sub("", line).strip().strip('"\'').strip()
        if not text or text.lower() in seen:
            continue
        if text.endswith(":"):  # a stray header like "Rephrasings:"
            continue
        seen.add(text.lower())
        out.append(text)
        if len(out) >= n:
            break
    return out


class QueryExpander:
    """Generates and caches query rephrasings."""

    def __init__(
        self,
        complete_fn: Callable[[str, str], str],
        model_name: str,
        n: int = 3,
        cache_path: Path | None = Path("data/query_expansions.json"),
    ):
        self.complete_fn = complete_fn
        self.model_name = model_name
        self.n = n
        self.cache_path = cache_path
        self._cache: dict[str, list[str]] = {}
        if cache_path and cache_path.exists():
            self._cache = json.loads(cache_path.read_text())

    def _key(self, question: str) -> str:
        # The prompt's hash is part of the key, so editing the prompt doesn't
        # silently keep serving rephrasings made with the old one.
        return f"{self.model_name}|{self.n}|{PROMPT_VERSION}|{question.strip()}"

    def expand(self, question: str) -> list[str]:
        """Return [original, variant1, variant2, ...]."""
        key = self._key(question)
        if key not in self._cache:
            raw = self.complete_fn(
                EXPANSION_SYSTEM_PROMPT,
                f"Question: {question}\n\nWrite {self.n} rephrasings.",
            )
            self._cache[key] = parse_variants(raw, question, self.n)
            self._save()
        return [question] + self._cache[key]

    def _save(self):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=2))
