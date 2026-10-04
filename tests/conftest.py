"""Keep the tests independent of the developer's .env and shell.

llm.py, providers.py and the API read .env when they are imported, and
load_dotenv never overrides a variable that is already set. Setting these here,
before any test module imports them, means a local RAG_PROVIDER=gemini or a real
API key cannot change what the tests see (and no test can reach a real API).
"""

import os

os.environ.update({
    "RAG_PROVIDER": "groq",
    "RAG_MODEL": "",
    "GROQ_API_KEY": "",
    "GEMINI_API_KEY": "",
    "GOOGLE_API_KEY": "",
    "GROQ_MAX_WAIT": "90",
    "GEMINI_MIN_INTERVAL": "6.5",
    "RAG_DATA_DIR": "data",
    "RAG_RESULTS_DIR": "results",
    "RAG_MODE": "hybrid_rerank",
    "RAG_RERANKER": "BAAI/bge-reranker-base",
    "RAG_K": "5",
    "RAG_MAX_SESSIONS": "50",
    "RAG_SESSION_TTL": "3600",
    "RAG_MAX_UPLOADS": "5",
    "RAG_WARM_START": "0",
    "RAG_CORS_ORIGINS": "",
    "API_URL": "http://localhost:8000",
})
