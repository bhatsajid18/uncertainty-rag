"""
REST API for the research assistant and the benchmark results.

Run (from the repo root):
  uvicorn api.main:app --app-dir src --port 8000
  # interactive docs at http://localhost:8000/docs

Endpoints
  GET    /health                          liveness + what is loaded
  GET    /papers                          the corpus: one row per paper
  POST   /search                          retrieval only, no LLM
  POST   /ask                             one question, no conversation
  POST   /sessions                        start a conversation
  GET    /sessions/{id}                   its history and documents
  DELETE /sessions/{id}
  POST   /sessions/{id}/messages          ask in the conversation (follow-ups work)
  POST   /sessions/{id}/documents         upload PDFs (multipart), alone or +corpus
  DELETE /sessions/{id}/documents         back to the corpus
  GET    /figures/{name}                  a figure image extracted from a paper
  GET    /results/uncertainty             the uncertainty benchmark (summary.json)
  GET    /results/uncertainty/figures/{name}
  GET    /results/{retrieval|generation}  evaluation runs
  GET    /results/{retrieval|generation}/{run}

The models load in a background thread from startup (RAG_WARM_START=0: on the
first request instead); /health answers meanwhile, and a question that needs
them gets a 503 "still loading". Errors map to HTTP codes: bad upload 400,
unknown session 404, provider daily limit 429, provider unavailable or no LLM
key 503, no index built 503.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()  # RAG_* settings and API keys from .env, before Settings() reads them

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.service import (  # noqa: E402
    ModelsLoading, RagService, SessionNotFound, Settings, source_view, turn_view,
)

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=20)
    mode: str | None = Field(None, description=f"One of {', '.join(MODES)}")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: str | None = None


class SessionRequest(BaseModel):
    mode: str | None = None


def _check_mode(mode: str | None):
    if mode is not None and mode not in MODES:
        raise HTTPException(422, f"mode must be one of {', '.join(MODES)}")


def create_app(service: RagService | None = None) -> FastAPI:
    service = service or RagService(Settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if os.environ.get("RAG_WARM_START", "1") == "1":
            service.warm_up_in_background()
        yield

    app = FastAPI(title="Uncertainty-Aware Research Intelligence API", version="1.0",
                  lifespan=lifespan)
    app.state.service = service
    # Browsers may call the API only from these origins. The Streamlit app calls
    # it from its server, so it needs none; the default allows no other site.
    origins = [o.strip() for o in os.environ.get("RAG_CORS_ORIGINS", "").split(",")
               if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"],
                           allow_headers=["*"])

    # --- errors from the pipeline, as HTTP responses ----------------------------------

    from llm import ProviderUnavailable, RateLimitTooLong
    from uploads import UploadError

    @app.exception_handler(SessionNotFound)
    async def _no_session(request: Request, exc: SessionNotFound):
        return JSONResponse(status_code=404, content={
            "detail": "Unknown or expired session. Start a new one."})

    @app.exception_handler(UploadError)
    async def _bad_upload(request: Request, exc: UploadError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RateLimitTooLong)
    async def _limited(request: Request, exc: RateLimitTooLong):
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.exception_handler(ProviderUnavailable)
    async def _unavailable(request: Request, exc: ProviderUnavailable):
        return JSONResponse(status_code=503, content={
            "detail": f"The language model is unavailable right now: {exc}"})

    @app.exception_handler(ModelsLoading)
    async def _loading(request: Request, exc: ModelsLoading):
        return JSONResponse(status_code=503, content={"detail": str(exc)},
                            headers={"Retry-After": "30"})

    @app.exception_handler(RuntimeError)
    async def _not_ready(request: Request, exc: RuntimeError):
        if "No index" in str(exc):
            return JSONResponse(status_code=503, content={"detail": str(exc)})
        raise exc

    # --- corpus --------------------------------------------------------------------------

    @app.get("/health")
    def health():
        return service.status()

    @app.get("/papers")
    def papers():
        return {"papers": service.papers()}

    @app.post("/search")
    def search(req: SearchRequest):
        _check_mode(req.mode)
        hits = service.search(req.query, k=req.k, mode=req.mode)
        return {"query": req.query, "hits": [source_view(h, n) for n, h in
                                              enumerate(hits, 1)]}

    @app.post("/ask")
    def ask(req: AskRequest):
        _check_mode(req.mode)
        return turn_view(service.ask_once(req.question, mode=req.mode))

    # --- conversations -----------------------------------------------------------------

    def _session_view(s) -> dict:
        return {"session_id": s.id, "scope": s.scope, "documents": s.documents,
                "with_corpus": s.with_corpus,
                "turns": [turn_view(t) for t in s.chat.turns]}

    @app.post("/sessions", status_code=201)
    def create_session(req: SessionRequest | None = None):
        mode = req.mode if req else None
        _check_mode(mode)
        return _session_view(service.create_session(mode=mode))

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        return _session_view(service.get_session(session_id))

    @app.delete("/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        service.delete_session(session_id)

    @app.post("/sessions/{session_id}/messages")
    def message(session_id: str, req: AskRequest):
        return turn_view(service.ask(session_id, req.question))

    @app.post("/sessions/{session_id}/documents")
    async def upload(session_id: str, files: list[UploadFile] = File(...),
                     with_corpus: bool = Form(True)):
        data = [(f.filename or "upload.pdf", await f.read()) for f in files]
        from starlette.concurrency import run_in_threadpool

        session = await run_in_threadpool(service.upload, session_id, data, with_corpus)
        return _session_view(session)

    @app.delete("/sessions/{session_id}/documents")
    def clear_documents(session_id: str):
        return _session_view(service.clear_uploads(session_id))

    # --- files and results -------------------------------------------------------------

    @app.get("/figures/{name}")
    def figure(name: str):
        path = service.figure_path(name)
        if path is None:
            raise HTTPException(404, "No such figure")
        return FileResponse(path, media_type="image/png")

    @app.get("/results/uncertainty")
    def uncertainty(fake: bool = False):
        res = service.uncertainty_results(fake=fake)
        if res is None:
            raise HTTPException(404, "No uncertainty results yet. Train and evaluate "
                                     "with notebooks/kaggle_uncertainty.ipynb, then unzip "
                                     "the results into the repo.")
        return res

    @app.get("/results/uncertainty/figures/{name}")
    def uncertainty_figure(name: str, fake: bool = False):
        path = service.uncertainty_figure(name, fake=fake)
        if path is None:
            raise HTTPException(404, "No such figure")
        return FileResponse(path, media_type="image/png")

    @app.get("/results/{kind}")
    def eval_runs(kind: str):
        if kind not in ("retrieval", "generation"):
            raise HTTPException(404, "kind must be retrieval or generation")
        return {"kind": kind, "runs": service.eval_runs(kind)}

    @app.get("/results/{kind}/{run}")
    def eval_run(kind: str, run: str):
        if kind not in ("retrieval", "generation"):
            raise HTTPException(404, "kind must be retrieval or generation")
        res = service.eval_run(kind, run)
        if res is None:
            raise HTTPException(404, f"No {kind} run named {run}")
        return res

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", app_dir=str(Path(__file__).resolve().parents[1]),
                host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))
