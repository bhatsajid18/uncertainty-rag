"""
Conversational RAG: ask follow-up questions about the corpus or your own PDFs.

A single question is easy: retrieve, then answer. Follow-ups are not, because
"what about its FPR95?" or "explain that more simply" can't be retrieved on
their own - the words that matter ("Outlier Exposure", "evidential deep
learning") are in earlier turns. Each turn therefore runs:

  1. rewrite    the LLM turns the follow-up into a standalone question using
                the recent conversation ("what about its FPR95?" ->
                "What is the FPR95 of Outlier Exposure on CIFAR-10?").
                Skipped on the first turn - there is nothing to resolve.
  2. expand     optional (--multi-query N): rephrasings of the standalone
                question, fused with RRF in the first retrieval stage.
  3. retrieve   hybrid retrieval + reranking on the standalone question.
  4. answer     grounded, cited answer. Recent turns are passed as context so
                replies read like a conversation, but earlier answers are NOT
                sources: every claim must cite this turn's sources, and the
                model must still refuse if they don't support an answer.

Earlier answers are shown to the model with their [S1]-style citations
removed, because source numbers restart every turn; an old [S2] would otherwise
point at a different chunk than the current [S2].

Sources to chat over:
  (default)                the arXiv corpus in data/
  --pdf a.pdf [b.pdf ...]  only the uploaded PDFs (isolated, nothing saved)
  --pdf ... --with-corpus  uploaded PDFs merged with the corpus

Usage (from the repo root):
  python src/rag/chat.py
  python src/rag/chat.py --pdf ~/Downloads/paper.pdf
  python src/rag/chat.py --pdf ~/Downloads/paper.pdf --with-corpus
  python src/rag/chat.py --multi-query 3

In the chat: /sources shows the last answer's sources with text, /history the
conversation, /reset starts over, /exit quits.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import (  # noqa: E402
    REFUSAL, SYSTEM_PROMPT, build_user_prompt, normalize_citations,
)
from llm import add_provider_args  # noqa: E402
from retrieve import (  # noqa: E402
    DEFAULT_RERANKER, MODES, REF_PENALTY, expand_split_tables,
)

REWRITE_SYSTEM = (
    "You rewrite follow-up questions for a search system over machine learning "
    "papers. Given a conversation and a follow-up, output ONE standalone question "
    "that means the same as the follow-up but can be understood without the "
    "conversation: replace pronouns and references like 'it', 'that method' or "
    "'the second one' with what they refer to. If the follow-up is already "
    "standalone, output it unchanged. Output only the question - no answer, no "
    "explanation, no quotes."
)

CHAT_RULES = (
    "\n7. Earlier turns of the conversation are included only so your reply fits "
    "the conversation. They are NOT sources: every factual claim must cite the "
    "numbered sources given with the current question, and rule 3 still applies "
    "if those sources are insufficient."
)

_CITATION = re.compile(r"\s?\[S\d+\]|\s?【[^】]*】")

# Greetings and thanks get a fixed reply: no retrieval, no LLM call. Running
# "hi" through the pipeline cost two model calls to produce a refusal, on a
# free tier with a daily token limit.
_SMALL_TALK = re.compile(
    r"^(hi+|hello|hey+|hiya|yo|good (morning|afternoon|evening)|thanks?|thank you"
    r"|thx|ty|ok(ay)?|cool|great|nice|got it)( there| so much| a lot)?[\s!.,:)]*$",
    re.I)
# Questions about the assistant itself. The corpus cannot answer them, so they
# were retrieved, reranked and refused at the cost of two LLM calls each.
_ABOUT_ME = re.compile(
    r"^(how are you|how'?s it going|how are things|who are you|what are you"
    r"|what can you do|what do you do|what is this|what's this|help)"
    r"[\s?!.,]*$", re.I)

GREETING_REPLY = ("Hi! Ask me anything about the papers - methods, results, "
                  "comparisons - and I'll answer with citations. Follow-up "
                  "questions work too.")
THANKS_REPLY = "You're welcome. Ask another question whenever you like."
ABOUT_REPLY = ("I answer questions from a corpus of machine learning papers on "
               "uncertainty and out-of-distribution detection. Every claim comes "
               "with a citation to the paper and page it came from, and I say so "
               "when the retrieved sources don't cover your question. Ask about a "
               "method, a reported number, or a comparison across papers.")


def small_talk_reply(text: str) -> str | None:
    """A canned reply for a greeting, thanks, or a question about the assistant;
    None for a question the corpus should answer."""
    text = text.strip()
    if _ABOUT_ME.match(text):
        return ABOUT_REPLY
    m = _SMALL_TALK.match(text)
    if not m:
        return None
    word = m.group(1).lower()
    thanks = word.startswith(("thank", "thx", "ty", "ok", "cool", "great", "nice", "got"))
    return THANKS_REPLY if thanks else GREETING_REPLY


@dataclass
class Turn:
    question: str
    standalone: str
    answer: str
    hits: list[dict] = field(default_factory=list)
    variants: list[str] | None = None

    @property
    def refused(self) -> bool:
        return REFUSAL.rstrip(".").lower() in self.answer.lower()


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text)


def _norm(q: str) -> str:
    return re.sub(r"[\s?.!]+", " ", q.lower()).strip()


def clean_rewrite(raw: str, fallback: str, max_ratio: float = 4.0) -> str:
    """First usable line of the rewrite, or the original question.

    Falls back when the model returns nothing, answers instead of rewriting
    (a long reply), or wraps the question in a label.
    """
    for line in raw.strip().splitlines():
        line = re.sub(r"^(standalone question|question|rewritten)\s*:\s*", "",
                      line.strip(), flags=re.I).strip().strip('"\'').strip()
        if line:
            too_long = len(line) > max(200, max_ratio * len(fallback))
            if too_long or _norm(line) == _norm(fallback):
                return fallback  # answered instead, or only re-capitalised
            return line
    return fallback


class ChatSession:
    """Multi-turn grounded Q&A over one retriever.

    chat_fn(messages) -> text is injected, so the session can be tested with a
    stub and doesn't depend on Groq directly.
    """

    def __init__(
        self,
        retriever,
        chat_fn: Callable[[list[dict]], str],
        k: int = 5,
        search_kwargs: dict | None = None,
        expander=None,
        history_turns: int = 3,
        max_history_chars: int = 700,
    ):
        self.retriever = retriever
        self.chat_fn = chat_fn
        self.k = k
        self.search_kwargs = search_kwargs or {}
        self.expander = expander
        self.history_turns = history_turns
        self.max_history_chars = max_history_chars
        self.turns: list[Turn] = []

    def reset(self):
        self.turns = []

    def _recent(self) -> list[Turn]:
        return self.turns[-self.history_turns:] if self.history_turns > 0 else []

    def _history_text(self, turn: Turn) -> str:
        text = strip_citations(turn.answer).strip()
        if len(text) > self.max_history_chars:
            text = text[: self.max_history_chars].rsplit(" ", 1)[0] + " ..."
        return text

    def rewrite(self, question: str) -> str:
        recent = self._recent()
        if not recent:
            return question
        convo = "\n\n".join(f"User: {t.question}\nAssistant: {self._history_text(t)}"
                            for t in recent)
        raw = self.chat_fn([
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": f"Conversation:\n{convo}\n\n"
                                        f"Follow-up: {question}\n\nStandalone question:"},
        ])
        return clean_rewrite(raw, question)

    def ask(self, question: str) -> Turn:
        canned = small_talk_reply(question)
        if canned:
            # not added to the history: it carries nothing a follow-up needs
            return Turn(question, question, canned)
        standalone = self.rewrite(question)
        variants = self.expander.expand(standalone) if self.expander else None
        hits = self.retriever.search(standalone, k=self.k, query_variants=variants,
                                     **self.search_kwargs)

        if not hits:
            turn = Turn(question, standalone, REFUSAL, [], variants)
            self.turns.append(turn)
            return turn

        lookup = getattr(self.retriever, "chunk_by_id", None)
        if lookup is not None:
            hits = expand_split_tables(hits, lookup)

        messages = [{"role": "system", "content": SYSTEM_PROMPT + CHAT_RULES}]
        for t in self._recent():
            messages.append({"role": "user", "content": t.question})
            messages.append({"role": "assistant", "content": self._history_text(t)})
        prompt_q = question if standalone == question else \
            f"{question}\n(In context, this means: {standalone})"
        messages.append({"role": "user", "content": build_user_prompt(prompt_q, hits)})

        answer = normalize_citations(self.chat_fn(messages)).strip()
        turn = Turn(question, standalone, answer, hits, variants)
        self.turns.append(turn)
        return turn


# --- command-line chat ---------------------------------------------------------


def _loc(h: dict) -> str:
    return (f"p.{h['page_start']}" if h["page_start"] == h["page_end"]
            else f"pp.{h['page_start']}-{h['page_end']}")


def print_turn(turn: Turn, show_text: bool = False):
    if turn.standalone != turn.question:
        print(f"  (searched for: {turn.standalone})")
    if turn.variants and len(turn.variants) > 1:
        print(f"  (+ {len(turn.variants) - 1} rephrasings)")
    print()
    print(textwrap.indent(turn.answer, "  "))
    if turn.hits:
        print("\n  Sources:")
        for n, h in enumerate(turn.hits, 1):
            year = f" ({h['year']})" if h.get("year") else ""
            extra = " (+ neighbouring chunk: split table)" if h.get("expanded_with") else ""
            if h.get("table_markdown"):
                extra += (" (+ table rebuilt from the PDF)"
                          if h.get("table_grid_source") == "geometry"
                          else " (+ rebuilt table)")
            if h.get("figure_note"):
                extra += " (+ figure)"
            print(f"    [S{n}] {h['arxiv_id']}{year} {_loc(h)} - {h['title'][:55]}{extra}")
            if show_text:
                body = " ".join(h["text"].split())[:400]
                print(textwrap.indent(textwrap.fill(body, 90), "         "))
                for grid in h.get("table_markdown", "").split("\n\n"):
                    if grid.strip():
                        print(textwrap.indent(grid, "         "))
                if h.get("figure_images"):
                    print(textwrap.indent("figures: " + h["figure_images"], "         "))


def build_retriever(args):
    from retrieve import HybridRetriever

    rerank = args.mode == "hybrid_rerank"
    if not args.pdf:
        return HybridRetriever(args.data_dir, reranker_model=args.reranker,
                               load_reranker=rerank)

    from transformers import AutoTokenizer
    from uploads import UploadError, load_uploads

    cfg_path = args.data_dir / "index_config.json"
    embed_model = (json.loads(cfg_path.read_text())["model"]
                   if cfg_path.exists() else "BAAI/bge-base-en-v1.5")
    print(f"Reading {len(args.pdf)} PDF(s) ...")
    try:
        chunks = load_uploads(args.pdf, AutoTokenizer.from_pretrained(embed_model))
    except UploadError as e:
        sys.exit(f"Upload problem: {e}")
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c["arxiv_id"]] = by_doc.get(c["arxiv_id"], 0) + 1
    for doc, n in by_doc.items():
        title = next(c["title"] for c in chunks if c["arxiv_id"] == doc)
        print(f"  {doc}: {n} chunks - {title[:60]}")
    print("Embedding" + (" and merging with the corpus" if args.with_corpus else "")
          + " ...")
    return HybridRetriever.from_chunks(
        chunks, base_dir=args.data_dir if args.with_corpus else None,
        embed_model=embed_model, reranker_model=args.reranker, load_reranker=rerank,
    )


HELP = ("Commands: /sources  show last answer's sources with text | /history | "
        "/reset  forget the conversation | /exit")


def main():
    ap = argparse.ArgumentParser(description="Chat with follow-up questions.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--pdf", type=Path, nargs="+",
                    help="Chat over these PDFs instead of the corpus.")
    ap.add_argument("--with-corpus", action="store_true",
                    help="With --pdf: search the PDFs AND the corpus together.")
    ap.add_argument("-k", type=int, default=5, help="Sources per answer.")
    add_provider_args(ap, fallback=True)
    ap.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    ap.add_argument("--multi-query", type=int, default=0, metavar="N",
                    help="Also retrieve with N LLM rephrasings (default 1; "
                         "0 = off, and saves one LLM call per new question).")
    ap.add_argument("--max-per-paper", type=int, default=None,
                    help="Per-document source cap (default 2; 0 when chatting "
                         "over uploads alone, since there are few documents).")
    ap.add_argument("--history-turns", type=int, default=3,
                    help="How many earlier turns the model sees.")
    ap.add_argument("--reranker", default=DEFAULT_RERANKER)
    ap.add_argument("--ref-penalty", type=float, default=REF_PENALTY,
                    help="Score multiplier for bibliography chunks "
                         f"(default {REF_PENALTY}; 0 excludes them).")
    ap.add_argument("--mmr", type=float, default=None, metavar="LAMBDA",
                    help="Pick sources with Maximal Marginal Relevance instead "
                         "of the per-paper cap (e.g. 0.7).")
    ap.add_argument("--min-score", type=float, default=None,
                    help="Refuse without calling the LLM when no source's "
                         "reranker score reaches this (0-1; off by default - "
                         "calibrate it with run_eval.py's abstention report).")
    args = ap.parse_args()

    if args.with_corpus and not args.pdf:
        sys.exit("--with-corpus only makes sense together with --pdf")
    if args.max_per_paper is None:
        args.max_per_paper = 0 if (args.pdf and not args.with_corpus) else 2

    from llm import RateLimitTooLong, get_backend, make_chat_fn, make_complete_fn, resolve

    provider, model, fallback = resolve(args)
    backend = get_backend(provider)  # one client for the whole session
    chat_fn = make_chat_fn(model, fallback_models=fallback, backend=backend)

    expander = None
    if args.multi_query > 0:
        from query_expansion import QueryExpander
        expander = QueryExpander(
            make_complete_fn(model, fallback_models=fallback, backend=backend),
            model_name=model,
            n=args.multi_query, cache_path=args.data_dir / "query_expansions.json")

    search_kwargs = {"mode": args.mode, "max_per_paper": args.max_per_paper,
                     "ref_penalty": args.ref_penalty}
    if args.mmr is not None:
        search_kwargs["mmr_lambda"] = args.mmr
    if args.min_score is not None:
        search_kwargs["min_score"] = args.min_score
    session = ChatSession(
        build_retriever(args), chat_fn, k=args.k, search_kwargs=search_kwargs,
        expander=expander, history_turns=args.history_turns,
    )

    scope = ("your PDF(s) + the corpus" if args.with_corpus else
             "your PDF(s)" if args.pdf else "the paper corpus")
    print(f"\nChatting over {scope}. {HELP}")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower().rstrip(".!") in ("/exit", "/quit", "exit", "quit", "bye", "q"):
            break
        if " " not in q and difflib.get_close_matches(
                q.lower(), ["quit", "exit"], n=1, cutoff=0.7):
            print("  (Did you mean to leave? Type quit or /exit.)")
            continue
        if q == "/help":
            print(HELP)
        elif q == "/reset":
            session.reset()
            print("  Conversation cleared.")
        elif q == "/history":
            for i, t in enumerate(session.turns, 1):
                print(f"  {i}. {t.question}"
                      + (f"   -> {t.standalone}" if t.standalone != t.question else ""))
        elif q == "/sources":
            if session.turns:
                print_turn(session.turns[-1], show_text=True)
            else:
                print("  Nothing asked yet.")
        elif q.startswith("/"):
            print(f"  Unknown command. {HELP}")
        else:
            try:
                print_turn(session.ask(q))
            except RateLimitTooLong as e:
                print(f"  {e}")


if __name__ == "__main__":
    main()
