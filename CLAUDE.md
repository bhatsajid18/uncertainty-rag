# Notes for Claude Code (and other contributors)

Uncertainty-Aware Research Intelligence Platform: a framework-free RAG system over
ML papers on uncertainty/OOD detection, an evaluation harness for it, an
uncertainty benchmark (CIFAR-10 vs SVHN/CIFAR-100), and a FastAPI + Streamlit app.
README.md is the overview; docs/DEVELOPMENT.md has the design notes and history.

## Ground rules

- Keep the stack: PyMuPDF, bge-base + FAISS, rank_bm25, bge-reranker, Groq/Gemini
  via `src/rag/llm.py`. No LangChain/LlamaIndex rewrite; `langchain_adapter.py` is
  a thin optional wrapper and stays that way.
- Never print, log or commit secrets. `.env` is gitignored. To check which keys
  are set without showing them: `grep -E '_KEY=' .env | sed -E 's/=.+/=***SET***/'`
  (an empty key stays visibly empty).
- Target machine: an 8 GB Apple-Silicon Mac. Don't load extra large models by
  default; GPU training happens on Kaggle (`notebooks/kaggle_uncertainty.ipynb`).
- Run everything from the repo root, in the `rag` conda env.
- Before committing: `make check` (ruff + pytest). Tests must stay offline: use
  the stand-ins in `tests/test_chat.py` (FakeEmbedder, fake_tokenizer, scripted
  LLMs), never real models, keys or network.
- Commits: short imperative subject (`feat: ...`, `fix: ...`, `docs: ...`).

## Layout

```
src/rag/           ingestion (download, chunk, tables, figures), index, retrieval,
                   generation, chat, LLM providers, uploads
src/evaluation/    query set building/review, retrieval eval (run_eval.py),
                   answer-quality eval (gen_eval.py)
src/uncertainty/   ResNet-18 softmax / MC Dropout / Deep Ensemble / EDL:
                   train.py -> predict.py -> evaluate.py
src/api/           FastAPI app (main.py) over RagService (service.py)
src/ui/            Streamlit app; talks to the API only (client.py)
configs/           uncertainty.yaml
docker/            api + ui Dockerfiles (docker-compose.yml at the root)
data/              corpus artifacts; most are gitignored and rebuilt by `make corpus`
results/           eval/ and gen_eval/ runs (committed selectively), uncertainty/
```

Scripts import siblings by path (`sys.path.insert`), e.g. `from retrieve import
HybridRetriever` inside `src/rag`; `src/uncertainty` uses package imports
(`from uncertainty.metrics import ...`) because `metrics` exists in two packages.

## Things that are easy to break

- `data/eval/queries.jsonl` is the reviewed query set; each relevant chunk is
  pinned by a hash of its text. Re-chunking changes the texts, and run_eval.py
  then refuses the stale labels: re-review with `src/evaluation/review_queries.py`.
  Don't hand-edit the file.
- Evaluation runs are committed deliberately: `git add -f results/eval/<run>`
  (same for `results/gen_eval/<run>`). Compare runs only on the same query set.
- The answer prompt's fixed refusal string (`generate.REFUSAL`) is what the
  abstention metrics look for; change it and the eval breaks.
- LLM calls go through `llm.make_chat_fn` / `make_complete_fn`, which handle
  rate limits, fallbacks and provider errors; don't call the SDKs directly.

## Commands

`make help` lists everything. The common ones:

```
make test / make lint / make check
make corpus            # download -> chunk -> tables -> figures -> index
make chat              # terminal chat
make eval-core         # retrieval metrics
make uncertainty-smoke # the whole uncertainty pipeline on fake data, ~1 min
make api / make ui     # http://localhost:8000/docs, http://localhost:8501
make docker-up
```
