"""
Thin HTTP client for the API, used by the Streamlit app.

The UI never imports the RAG pipeline: everything goes through the API, so the
frontend container stays small (no torch, no models) and the two scale apart.
Errors come back as ApiError with the API's own message, ready to show.
"""

from __future__ import annotations

import os

import requests

try:  # API_URL / API_TIMEOUT may be in .env (Docker sets them in the environment)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.environ.get("API_TIMEOUT", "180"))  # an LLM answer can take a while


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _call(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as e:
        raise ApiError(f"Cannot reach the API at {API_URL} ({e.__class__.__name__}). "
                       "Is it running?") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(d.get("msg", str(d)) for d in detail)
        raise ApiError(str(detail), r.status_code)
    if r.status_code == 204:
        return None
    if r.headers.get("content-type", "").startswith("image/"):
        return r.content
    return r.json()


def health() -> dict:
    return _call("GET", "/health")


def papers() -> list[dict]:
    return _call("GET", "/papers")["papers"]


def new_session(mode: str | None = None) -> dict:
    return _call("POST", "/sessions", json={"mode": mode})


def get_session(session_id: str) -> dict:
    return _call("GET", f"/sessions/{session_id}")


def ask(session_id: str, question: str) -> dict:
    return _call("POST", f"/sessions/{session_id}/messages", json={"question": question})


def upload(session_id: str, files: list[tuple[str, bytes]], with_corpus: bool) -> dict:
    return _call("POST", f"/sessions/{session_id}/documents",
                 files=[("files", (name, data, "application/pdf")) for name, data in files],
                 data={"with_corpus": str(with_corpus).lower()})


def clear_documents(session_id: str) -> dict:
    return _call("DELETE", f"/sessions/{session_id}/documents")


def figure(name: str) -> bytes:
    return _call("GET", f"/figures/{name}")


def uncertainty(fake: bool = False) -> dict:
    return _call("GET", "/results/uncertainty", params={"fake": str(fake).lower()})


def uncertainty_figure(name: str, fake: bool = False) -> bytes:
    return _call("GET", f"/results/uncertainty/figures/{name}",
                 params={"fake": str(fake).lower()})


def eval_runs(kind: str) -> list[str]:
    return _call("GET", f"/results/{kind}")["runs"]


def eval_run(kind: str, run: str) -> dict:
    return _call("GET", f"/results/{kind}/{run}")
