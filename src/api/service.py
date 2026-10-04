"""
The application layer behind the web API: models loaded once, chat sessions,
PDF uploads, and read access to the evaluation results.

Everything the RAG pipeline does is reused, not reimplemented: retrieval is
retrieve.HybridRetriever, a conversation is chat.ChatSession, uploads go
through uploads.load_uploads. This module only adds what a multi-user server
needs on top:

  sessions      one ChatSession per browser session, evicted when idle
                (session_ttl) or when there are too many (max_sessions), so
                memory stays bounded however many tabs are opened.
  uploads       PDFs are written to a temporary folder, chunked, indexed in
                memory (alone, or merged with the corpus) and deleted again. The
                embedder and the reranker are SHARED with the corpus retriever:
                a fresh 1.1 GB reranker per upload would exhaust a laptop.
  concurrency   retrieval runs one request at a time (the models are not
                guaranteed thread-safe), while the slow LLM calls of different
                sessions overlap. A session handles one message at a time.

Every heavy dependency is injectable (retriever, chat function, tokenizer), so
the API is tested with small in-memory stand-ins.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

RAG = Path(__file__).resolve().parents[1] / "rag"
sys.path.insert(0, str(RAG))

from chat import ChatSession, Turn  # noqa: E402
from uploads import MAX_MB, UploadError  # noqa: E402

ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
# The environment variables that hold each provider's API key (any one will do).
KEY_VARS = {"groq": ("GROQ_API_KEY",), "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    """Server configuration, from environment variables (see .env.example)."""

    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("RAG_DATA_DIR", "data")))
    results_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("RAG_RESULTS_DIR", "results")))
    provider: str = field(default_factory=lambda: os.environ.get("RAG_PROVIDER", "groq"))
    model: str | None = field(default_factory=lambda: os.environ.get("RAG_MODEL") or None)
    reranker: str = field(default_factory=lambda: os.environ.get(
        "RAG_RERANKER", "BAAI/bge-reranker-base"))
    mode: str = field(default_factory=lambda: os.environ.get("RAG_MODE", "hybrid_rerank"))
    k: int = field(default_factory=lambda: _env_int("RAG_K", 5))
    max_sessions: int = field(default_factory=lambda: _env_int("RAG_MAX_SESSIONS", 50))
    session_ttl: int = field(default_factory=lambda: _env_int("RAG_SESSION_TTL", 3600))
    max_upload_files: int = field(default_factory=lambda: _env_int("RAG_MAX_UPLOADS", 5))


class LockedRetriever:
    """Serialises search() across threads; everything else passes through."""

    def __init__(self, inner, lock: threading.Lock):
        self._inner, self._lock = inner, lock

    def search(self, *args, **kwargs):
        with self._lock:
            return self._inner.search(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _DeferredRetriever:
    """Looks the real retriever up on every use (see RagService._corpus)."""

    def __init__(self, get):
        self._get = get

    def search(self, *args, **kwargs):
        return self._get().search(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._get(), name)


@dataclass
class Session:
    id: str
    chat: ChatSession
    documents: list[dict] = field(default_factory=list)
    with_corpus: bool = True
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)
    mode: str | None = None

    @property
    def scope(self) -> str:
        if not self.documents:
            return "corpus"
        return "uploads+corpus" if self.with_corpus else "uploads"


class SessionNotFound(KeyError):
    pass


class ModelsLoading(RuntimeError):
    """The warm start is still loading the models (the first start downloads them)."""


# The provider SDKs' own exceptions (bad key, bad request, ...). They are not
# imported here, so they are recognised by the module they come from.
_PROVIDER_MODULES = ("groq", "google.genai", "google.api_core", "httpx")


def guard_provider_errors(chat_fn: Callable[[list[dict]], str], provider: str,
                          key_names: str) -> Callable[[list[dict]], str]:
    """Turn an SDK error (a rejected API key, say) into ProviderUnavailable.

    llm.py already retries what can be retried and raises ProviderUnavailable /
    RateLimitTooLong itself; what reaches here is a request the provider refused.
    Unhandled, it would be an HTTP 500 with no explanation.
    """
    from llm import ProviderUnavailable

    def call(messages):
        try:
            return chat_fn(messages)
        except Exception as e:
            if not type(e).__module__.startswith(_PROVIDER_MODULES):
                raise
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status in (401, 403):
                raise ProviderUnavailable(
                    f"the {provider} API key was rejected ({status}). Check {key_names} "
                    "in .env and restart the API.") from e
            raise ProviderUnavailable(f"{provider} refused the request: "
                                      f"{type(e).__name__}: {str(e)[:300]}") from e
    return call


class RagService:
    def __init__(self, settings: Settings | None = None, retriever=None,
                 chat_fn: Callable[[list[dict]], str] | None = None,
                 tokenizer=None, upload_retriever_factory=None):
        self.settings = settings or Settings()
        self._retriever = retriever
        self._chat_fn = chat_fn
        self._tokenizer = tokenizer
        self._factory = upload_retriever_factory
        self._model_lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._sessions_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._tokenizer_lock = threading.Lock()
        self._loading = False

    # --- heavy resources, loaded on first use ------------------------------------

    def _load_retriever(self):
        with self._load_lock:
            if self._retriever is not None:
                return
            from retrieve import HybridRetriever

            if not (self.settings.data_dir / "index.faiss").exists():
                raise RuntimeError(
                    f"No index in {self.settings.data_dir}/. Build it first: "
                    "python src/rag/build_index.py build")
            self._loading = True
            try:
                self._retriever = HybridRetriever(
                    self.settings.data_dir, reranker_model=self.settings.reranker,
                    load_reranker=self.settings.mode == "hybrid_rerank")
            finally:
                self._loading = False

    @property
    def retriever(self):
        if self._retriever is None:
            if self._loading:
                # Another thread is loading. Waiting could take minutes on a first
                # start (the download) and the client would time out with a
                # misleading "cannot reach the API"; say what is happening instead.
                raise ModelsLoading(
                    "The models are still loading (a first start downloads about "
                    "1.5 GB). Try again in a minute.")
            self._load_retriever()
        return LockedRetriever(self._retriever, self._model_lock)

    def _corpus(self):
        """The corpus retriever, resolved at search time: a conversation can
        start before the models are loaded."""
        return _DeferredRetriever(lambda: self.retriever)

    def _llm(self, messages: list[dict]) -> str:
        """The chat function, resolved at call time: a conversation can start
        (and small talk works) without an API key."""
        return self.chat_fn(messages)

    def llm_key_set(self) -> bool:
        if self._chat_fn is not None:
            return True
        return any(os.environ.get(v) for v in KEY_VARS.get(self.settings.provider, ()))

    @property
    def chat_fn(self):
        if self._chat_fn is None:
            from llm import (ProviderUnavailable, default_fallback, default_model,
                             make_chat_fn)

            provider = self.settings.provider
            names = " or ".join(KEY_VARS.get(provider, ("an API key",)))
            if not self.llm_key_set():
                raise ProviderUnavailable(
                    f"{names} is not set. Add it to .env and restart the API "
                    "(search and the dashboards work without it).")
            model = self.settings.model or default_model(provider)
            try:
                chat = make_chat_fn(model, provider=provider,
                                    fallback_models=(default_fallback(provider),))
            except SystemExit as e:
                # The CLI helpers exit on a missing key or SDK. In a server that
                # must be an error response, not a shutdown.
                raise ProviderUnavailable(
                    str(e.code) if isinstance(e.code, str) else
                    f"could not start the {provider} client") from None
            self._chat_fn = guard_provider_errors(chat, provider, names)
        return self._chat_fn

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            cfg = json.loads((self.settings.data_dir / "index_config.json").read_text())
            self._tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
        return self._tokenizer

    def warm_up(self):
        """Load the corpus retriever now rather than on the first request."""
        self._load_retriever()

    def warm_up_in_background(self) -> threading.Thread:
        """Load in a thread, so /health answers while the models download.

        A request that needs the retriever meanwhile gets ModelsLoading (a 503).
        A failure is printed here and raised again, as an HTTP error, by the
        first request that needs the retriever afterwards.
        """
        def run():
            try:
                self.warm_up()
            except Exception as e:
                print(f"Warm start failed: {e}", file=sys.stderr, flush=True)
            finally:
                self._loading = False

        self._loading = True  # already now, so no request starts a second load
        thread = threading.Thread(target=run, name="warm-up", daemon=True)
        thread.start()
        return thread

    # --- corpus ----------------------------------------------------------------------

    def status(self) -> dict:
        loaded = self._retriever is not None
        meta = self._retriever.meta if loaded else []
        return {"status": "ok", "retriever_loaded": loaded, "loading": self._loading,
                "corpus_chunks": len(meta),
                "papers": len({m["arxiv_id"] for m in meta}),
                "mode": self.settings.mode, "provider": self.settings.provider,
                "llm_ready": self.llm_key_set(),
                "sessions": len(self._sessions)}

    def papers(self) -> list[dict]:
        by_id: dict[str, dict] = {}
        for m in self.retriever.meta:
            p = by_id.setdefault(m["arxiv_id"], {
                "arxiv_id": m["arxiv_id"], "title": m.get("title", ""),
                "year": m.get("year", ""), "chunks": 0})
            p["chunks"] += 1
        return sorted(by_id.values(), key=lambda p: p["arxiv_id"])

    def search_kwargs(self, mode: str | None = None, uploads_only: bool = False) -> dict:
        kw = {"mode": mode or self.settings.mode}
        if uploads_only:
            kw["max_per_paper"] = 0  # a few documents: no per-paper cap
        return kw

    def search(self, query: str, k: int | None = None, mode: str | None = None) -> list[dict]:
        return self.retriever.search(query, k=k or self.settings.k,
                                     **self.search_kwargs(mode))

    def ask_once(self, question: str, mode: str | None = None) -> Turn:
        """A single question with no conversation."""
        session = ChatSession(self._corpus(), self._llm, k=self.settings.k,
                              search_kwargs=self.search_kwargs(mode))
        return session.ask(question)

    # --- sessions ----------------------------------------------------------------------

    def _evict(self):
        now = time.time()
        stale = [s for s in self._sessions.values()
                 if now - s.last_used > self.settings.session_ttl]
        over = len(self._sessions) - len(stale) - self.settings.max_sessions + 1
        if over > 0:
            alive = sorted((s for s in self._sessions.values() if s not in stale),
                           key=lambda s: s.last_used)
            stale += alive[:over]
        for s in stale:
            self._drop(s)

    def _drop(self, session: Session):
        self._sessions.pop(session.id, None)

    def create_session(self, mode: str | None = None) -> Session:
        chat = ChatSession(self._corpus(), self._llm, k=self.settings.k,
                           search_kwargs=self.search_kwargs(mode))
        with self._sessions_lock:
            self._evict()
            session = Session(id=uuid.uuid4().hex, chat=chat, mode=mode)
            self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(session_id)
            now = time.time()
            if now - session.last_used > self.settings.session_ttl:
                self._drop(session)  # idle too long: expired, even if not evicted yet
                raise SessionNotFound(session_id)
            session.last_used = now
            return session

    def delete_session(self, session_id: str):
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFound(session_id)
            self._drop(session)

    def ask(self, session_id: str, question: str) -> Turn:
        session = self.get_session(session_id)
        with session.lock:
            return session.chat.ask(question)

    # --- uploads -----------------------------------------------------------------------

    def _build_upload_retriever(self, chunks: list[dict], with_corpus: bool):
        if self._factory is not None:
            return self._factory(chunks, with_corpus)
        from retrieve import HybridRetriever

        _ = self.retriever  # make sure the corpus models are loaded
        base = self._retriever
        cfg = json.loads((self.settings.data_dir / "index_config.json").read_text())
        with self._model_lock:  # the embedder is shared with every search
            built = HybridRetriever.from_chunks(
                chunks, base_dir=self.settings.data_dir if with_corpus else None,
                embed_model=cfg["model"], query_prefix=cfg.get("query_prefix", ""),
                embedder=base.embedder, reranker_model=self.settings.reranker,
                reranker=getattr(base, "_reranker", None),
                load_reranker=self.settings.mode == "hybrid_rerank")
        return LockedRetriever(built, self._model_lock)

    def upload(self, session_id: str, files: list[tuple[str, bytes]],
               with_corpus: bool) -> Session:
        """Replace the session's documents with these PDFs and start a fresh
        conversation over them (alone, or merged with the corpus)."""
        from uploads import load_uploads

        if not files:
            raise UploadError("No files were uploaded.")
        if len(files) > self.settings.max_upload_files:
            raise UploadError(f"At most {self.settings.max_upload_files} PDFs at a time.")
        session = self.get_session(session_id)
        with session.lock:
            folder = Path(tempfile.mkdtemp(prefix=f"rag-{session_id[:8]}-"))
            try:
                paths = []
                for i, (name, data) in enumerate(files):
                    if len(data) > MAX_MB * 1e6:
                        raise UploadError(f"{name}: larger than the {MAX_MB} MB limit")
                    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name) or "upload.pdf"
                    if not safe.lower().endswith(".pdf"):
                        raise UploadError(f"{name}: not a .pdf file")
                    # a folder per file: two uploads can share a name
                    path = folder / str(i) / safe
                    path.parent.mkdir()
                    path.write_bytes(data)
                    paths.append(path)
                # The tokenizer is shared by all uploads (not by search), and a
                # fast tokenizer is not safe to use from two threads at once.
                with self._tokenizer_lock:
                    chunks = load_uploads(paths, self.tokenizer)
                retriever = self._build_upload_retriever(chunks, with_corpus)
            finally:
                # The chunks hold the text now; the PDFs are not needed again.
                shutil.rmtree(folder, ignore_errors=True)
            docs: dict[str, dict] = {}
            for c in chunks:
                d = docs.setdefault(c["arxiv_id"], {"doc_id": c["arxiv_id"],
                                                    "title": c["title"], "chunks": 0})
                d["chunks"] += 1
            session.documents = list(docs.values())
            session.with_corpus = with_corpus
            session.chat = ChatSession(
                retriever, self._llm, k=self.settings.k,
                search_kwargs=self.search_kwargs(session.mode,
                                                 uploads_only=not with_corpus))
        return session

    def clear_uploads(self, session_id: str) -> Session:
        session = self.get_session(session_id)
        with session.lock:
            session.documents, session.with_corpus = [], True
            session.chat = ChatSession(self._corpus(), self._llm, k=self.settings.k,
                                       search_kwargs=self.search_kwargs(session.mode))
        return session

    # --- results -----------------------------------------------------------------------

    def figure_path(self, name: str) -> Path | None:
        """A figure image, only from the figures folder (no path traversal)."""
        if not re.fullmatch(r"[A-Za-z0-9._:-]+\.png", name):
            return None
        path = self.settings.data_dir / "figures" / name
        return path if path.is_file() else None

    def uncertainty_results(self, fake: bool = False) -> dict | None:
        path = self.settings.results_dir / "uncertainty" / ("fake" if fake else "") \
            / "summary.json"
        return json.loads(path.read_text()) if path.exists() else None

    def uncertainty_figure(self, name: str, fake: bool = False) -> Path | None:
        if not re.fullmatch(r"[a-z0-9_]+\.png", name):
            return None
        path = (self.settings.results_dir / "uncertainty" / ("fake" if fake else "")
                / "figures" / name)
        return path if path.is_file() else None

    def eval_runs(self, kind: str) -> list[str]:
        root = self.settings.results_dir / ("eval" if kind == "retrieval" else "gen_eval")
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir()
                      if p.is_dir() and (p / ("summary.csv" if kind == "retrieval"
                                              else "summary.json")).exists())

    def eval_run(self, kind: str, run: str) -> dict | None:
        if run not in self.eval_runs(kind):
            return None
        root = self.settings.results_dir / ("eval" if kind == "retrieval" else "gen_eval") / run
        out: dict = {"run": run}
        for name in ("summary", "significance", "abstention", "summary_by_category"):
            path = root / f"{name}.csv"
            if path.exists():
                out[name] = read_csv(path)
        for name in ("run_info", "summary"):
            path = root / f"{name}.json"
            if path.exists() and name not in out:
                out[name] = json.loads(path.read_text())
        return out


def read_csv(path: Path) -> list[dict]:
    """CSV rows with numbers and booleans converted."""
    def conv(v: str):
        if v in ("True", "False"):
            return v == "True"
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    with path.open(newline="") as f:
        return [{k: conv(v) for k, v in row.items()} for row in csv.DictReader(f)]


def source_view(hit: dict, n: int) -> dict:
    """What the UI needs to show one source."""
    pages = (f"p.{hit['page_start']}" if hit["page_start"] == hit["page_end"]
             else f"pp.{hit['page_start']}-{hit['page_end']}")
    arxiv = hit.get("arxiv_id", "")
    figures = [Path(p).name for p in (hit.get("figure_images") or "").split(",") if p]
    return {
        "n": n, "chunk_id": hit["chunk_id"], "arxiv_id": arxiv,
        "title": hit.get("title", ""), "year": hit.get("year", ""), "pages": pages,
        "score": float(hit.get("score", 0.0)),
        "text": " ".join(hit.get("text", "").split())[:1200],
        "url": f"https://arxiv.org/abs/{arxiv}" if ARXIV_ID.match(arxiv) else None,
        "table_markdown": hit.get("table_markdown") or None,
        "table_source": hit.get("table_grid_source") or None,
        "figures": [{"name": f, "url": f"/figures/{f}"} for f in figures],
        "expanded_with": hit.get("expanded_with") or [],
    }


def turn_view(turn: Turn) -> dict:
    return {"question": turn.question, "standalone": turn.standalone,
            "answer": turn.answer, "refused": turn.refused,
            "rephrasings": (turn.variants or [])[1:],
            "sources": [source_view(h, n) for n, h in enumerate(turn.hits, 1)]}
