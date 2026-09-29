# Uncertainty-Aware Research Intelligence Platform

Retrieval-augmented Q&A over ML research papers, combined with an uncertainty
and OOD evaluation framework.

**Status:** Iterations 1–3 built; Iteration 3 results pending the labelled query set.
- **Iteration 1** — end-to-end RAG pipeline (download → chunk → embed/index →
  grounded generation) running locally on Apple Silicon.
- **Iteration 2** — hybrid retrieval (dense + BM25), reciprocal-rank fusion,
  cross-encoder reranking, source-diversity control, density-based
  bibliography classification, and retrieval-strategy selection wired through
  to answer generation.
- **Iteration 3** — evaluation harness: LLM-assisted, human-reviewed query set
  across four query categories; Recall/Precision/NDCG/Success@k and MRR;
  paired-bootstrap significance tests; abstention AUROC; LLM multi-query
  expansion as an additional retrieval strategy. Results saved as CSV/JSON.
- **Conversational chat** — multi-turn Q&A with follow-up questions, over the
  corpus, over your own uploaded PDFs, or both together.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Prerequisites](#2-prerequisites)
3. [One-time environment setup](#3-one-time-environment-setup)
4. [Getting your Groq API key](#4-getting-your-groq-api-key)
5. [The pipeline, module by module](#5-the-pipeline-module-by-module)
6. [Command reference](#6-command-reference)
7. [Design notes](#7-design-notes)
8. [Debugging log](#8-debugging-log)
9. [Ablation surface and evaluation](#9-ablation-surface-and-evaluation)
10. [Tech stack](#10-tech-stack)
11. [What each script does](#11-what-each-script-does)
12. [Roadmap](#12-roadmap)

---

## 1. Project overview

Two connected modules, built iteratively:

1. **Research RAG** — ask questions about a corpus of ML papers, get answers
   grounded strictly in retrieved chunks, with citations back to paper + page.
2. **Uncertainty / OOD evaluation** (upcoming) — compare softmax, MC Dropout,
   Deep Ensembles, and Evidential Deep Learning on CIFAR-10 vs OOD datasets.

This README covers everything through **Iteration 2**.

---

## 2. Prerequisites

- macOS on Apple Silicon (M1/M2/M3) — this setup targets that specifically.
- ~7 GB free disk space for models, corpus, and caches (the reranker alone is
  ~1.1 GB).
- A free [Groq](https://console.groq.com) account for LLM calls (no credit card).
- A GitHub account, for version control.

---

## 3. One-time environment setup

### 3.1 Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3.2 Install Miniconda

```bash
brew install --cask miniconda
conda init "$(basename "${SHELL}")"
```

Restart your terminal after this so `conda` is available.

### 3.3 Create the project directory and conda environment

```bash
mkdir -p ~/PROJECTS/uncertainty-rag      # or wherever you keep projects
cd ~/PROJECTS/uncertainty-rag

conda create -n rag python=3.11 -y
conda activate rag
```

> **Important:** always run `conda activate rag` before working in this
> project, and always install packages with `python -m pip install ...`
> (not bare `pip install ...`) to guarantee packages land in this env and
> not some other Python on your system.

### 3.4 Install PyTorch (with Apple Silicon GPU support)

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1
```

Verify GPU (MPS) support:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

Should print `MPS available: True`.

### 3.5 Install all project dependencies

```bash
python -m pip install \
  sentence-transformers==3.3.1 \
  faiss-cpu==1.9.0 \
  rank-bm25==0.2.2 \
  pymupdf==1.24.14 \
  arxiv \
  groq \
  python-dotenv \
  numpy pandas scikit-learn matplotlib seaborn \
  requests tqdm \
  wandb hydra-core omegaconf \
  pytest ruff \
  jupyterlab ipykernel
```

Register the environment as a Jupyter kernel (optional):

```bash
python -m ipykernel install --user --name rag --display-name "Python (rag)"
```

### 3.6 Repo structure

```
uncertainty-rag/
├── src/
│   ├── rag/
│   │   ├── download_papers.py   # fetch arXiv PDFs + metadata
│   │   ├── chunk_papers.py      # parse PDFs -> clean, tagged text chunks
│   │   ├── build_index.py       # embed chunks, build FAISS index, dense search
│   │   ├── retrieve.py          # dense / BM25 / hybrid / rerank retrieval
│   │   ├── generate.py          # grounded, cited answer generation via Groq
│   │   ├── llm.py               # shared Groq client + rate-limit handling
│   │   ├── query_expansion.py   # LLM multi-query rephrasing (cached)
│   │   ├── chat.py              # multi-turn chat with follow-up questions
│   │   ├── uploads.py           # chunk user PDFs for chat (in memory)
│   │   ├── providers.py         # LLM backends: Groq and Gemini
│   │   ├── table_extract.py     # rebuild tables from PDF word positions
│   │   ├── table_notes.py       # LLM table descriptions + rebuilt tables (cached)
│   │   ├── figures.py           # figure images, captions, vision descriptions
│   │   ├── langchain_adapter.py # expose the retriever to LangChain (optional)
│   │   ├── debug_tables.py      # diagnostic: how a table reaches the model
│   │   └── debug_refs.py        # diagnostic: inspect bibliography detection
│   ├── evaluation/
│   │   ├── metrics.py           # Recall/Precision/NDCG/Success@k, MRR, abstention
│   │   ├── build_queries.py     # LLM-generated candidate queries (4 categories)
│   │   ├── review_queries.py    # human review -> ground-truth query set
│   │   └── run_eval.py          # sweep configs, significance, CSV/JSON output
│   ├── uncertainty/              # (Iteration 4+)
│   └── api/                       # (later, if deploying)
├── data/
│   ├── papers.txt                 # your arXiv ID list (tracked in git)
│   ├── papers/                    # downloaded PDFs (gitignored)
│   ├── metadata.json              # paper metadata (gitignored)
│   ├── chunks.jsonl               # chunked text (gitignored)
│   ├── index.faiss                # FAISS vector index (gitignored)
│   ├── chunk_meta.json            # vector-position -> chunk metadata (gitignored)
│   ├── index_config.json          # records embedding model used (gitignored)
│   ├── query_expansions.json      # cached LLM rephrasings (tracked)
│   ├── table_notes.json           # cached LLM table notes (tracked)
│   ├── tables.json                # tables rebuilt from PDF geometry (tracked)
│   ├── figures.json               # figure captions + descriptions (tracked)
│   ├── figures/                   # rendered figure PNGs (gitignored)
│   ├── chunk_config.json          # chunking strategy used for chunks.jsonl
│   └── eval/
│       ├── candidates.jsonl       # generated candidates + review status (tracked)
│       └── queries.jsonl          # reviewed ground-truth query set (tracked)
├── results/eval/<run-name>/       # evaluation outputs, CSV/JSON (tracked)
├── configs/                       # Hydra configs (later)
├── experiments/                   # experiment scripts (later)
├── notebooks/                     # exploratory notebooks
├── tests/                         # pytest: metrics, multi-query, eval, chat, uploads
├── .env                           # secrets (GROQ_API_KEY) — gitignored
├── .gitignore
├── pyproject.toml
└── README.md
```

### 3.7 `.gitignore`

```
# Secrets
.env

# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/

# Environments
.venv/
venv/

# Data and models — never commit these (large, and PDFs are copyrighted)
data/papers/
data/*.faiss
data/*.jsonl
data/metadata.json
data/chunk_meta.json
data/index_config.json
data/figures/
.cache/

# Model checkpoints
*.pt
*.pth
*.ckpt
*.pkl

# Experiment tracking / outputs
wandb/
outputs/
logs/
multirun/

# OS / IDE
.DS_Store
.vscode/
.idea/
*.swp
```

Note: `data/papers.txt` is deliberately **not** ignored — it's small and worth
version-controlling as the source-of-truth ID list.

### 3.8 GitHub setup

```bash
git init
git add .
git commit -m "chore: initial project structure"
git branch -M main
```

Create an empty repo on github.com (don't initialize with a README), then:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Authenticate with a **Personal Access Token** (Settings → Developer settings →
Personal access tokens → Generate new token (classic) → scope: `repo`). When
git prompts for a password, paste the token. Remember it via Keychain:

```bash
git config --global credential.helper osxkeychain
```

---

## 4. Getting your Groq API key

1. Go to https://console.groq.com and sign up (free, no credit card).
2. Go to https://console.groq.com/keys → "Create API Key" → copy it
   immediately (shown only once).
3. Create a `.env` file in the project root:
   ```bash
   echo "GROQ_API_KEY=gsk_your_actual_key_here" > .env
   ```
4. Confirm it's set (without printing the actual key):
   ```bash
   grep GROQ_API_KEY .env | sed 's/=.*/=***SET***/'
   ```

**Available chat models on the free tier (verified Sept 2026):**
`openai/gpt-oss-120b` (default, strongest), `openai/gpt-oss-20b` (faster,
smaller), `qwen/qwen3.8-27b`. Groq's lineup changes — if a model 404s, list
what's actually available on your account:

```bash
python - << 'EOF'
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(".env")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
models = client.models.list().data
skip = ("whisper", "tts", "guard", "orpheus")
for m in sorted(x.id for x in models if not any(s in x.id.lower() for s in skip)):
    print(m)
EOF
```

**Free tier rate limits:** roughly 30 requests/minute and 6,000 tokens/minute,
and about 200k tokens/day per model. Short waits are slept through; a wait
longer than 90 seconds (the daily limit) stops with a message instead, and
chat falls back to `openai/gpt-oss-20b`, which has its own quota.

### 4.1 Gemini key (optional second provider)

Groq's limit is tokens per day, which one table-notes run uses up. Gemini's
free tier limits requests per day instead (hundreds), which suits batch jobs
over the corpus, and it accepts images, which figure descriptions need.

1. Go to https://aistudio.google.com/apikey → "Create API key" (free, no card).
2. Add it to `.env`:
   ```bash
   echo "GEMINI_API_KEY=your_key_here" >> .env
   ```
3. Install the SDK:
   ```bash
   python -m pip install google-genai
   ```
4. Check it works:
   ```bash
   python src/rag/generate.py "what is evidential deep learning?" --provider gemini
   ```

List what your account can actually call (Google retires model ids often):
```bash
python src/rag/providers.py --list
```
```bash
python src/rag/providers.py --provider groq --list
```

Every script that calls an LLM takes `--provider groq|gemini` and `--model`;
the default model is the provider's free-tier one (`openai/gpt-oss-120b`,
`gemini-flash-latest`). Gemini's defaults are the `-latest` aliases because
pinned ids come and go; pass `--model` with a pinned id when a run has to stay
reproducible. If a model id has been retired, the client reads the replacement
out of Google's 404 (or picks the closest model your account lists), says
which it used, and carries on; if a model is busy (503), it backs off, then
tries the next model in the chain. `export RAG_PROVIDER=gemini` changes the default for a
shell session. Requests to Gemini are spaced ~6.5s apart to stay inside its
requests-per-minute limit, so batch jobs are slower but finish.

Which to use where: Groq for chat (much faster to first token), Gemini for
table notes, figure descriptions and query generation (they are batch jobs
whose cost is tokens, not latency), and whichever you like for evaluation -
but keep one provider for a whole evaluation run, or the runs aren't
comparable.

---

## 5. The pipeline, module by module

### 5.1 `download_papers.py` — fetch the corpus

Downloads PDFs + metadata (title, authors, year, abstract, categories) for a
list of arXiv IDs, using the `arxiv` library for metadata and a direct
`requests` download for the PDF itself.

**Input:** `data/papers.txt` — one arXiv ID per line, `#` comments allowed.
**Output:** `data/papers/*.pdf` and `data/metadata.json`.

Re-running is safe — it skips PDFs already downloaded and merges metadata, so
you can grow the corpus by adding IDs and re-running.

### 5.2 `chunk_papers.py` — parse and chunk

Extracts text page-by-page with PyMuPDF, cleans it (page-number/header noise,
de-hyphenation, Unicode NFKC normalization for ligatures like `ﬁ`→`fi`), splits
into ~512-token overlapping chunks using the embedding model's own tokenizer,
and classifies each chunk as content or bibliography using **citation density**
(see §8 for why this replaced an earlier heading-based approach).

**Output:** `data/chunks.jsonl` — one JSON object per line:
```json
{
  "chunk_id": "1806.01768__0000",
  "arxiv_id": "1806.01768",
  "title": "Evidential Deep Learning to Quantify Classification Uncertainty",
  "year": "2018",
  "chunk_index": 0,
  "page_start": 1,
  "page_end": 1,
  "is_reference": false,
  "n_tokens": 512,
  "text": "..."
}
```

### 5.3 `build_index.py` — embed and index

Embeds every chunk with `BAAI/bge-base-en-v1.5` (normalized vectors) and builds
a FAISS inner-product index (equivalent to cosine similarity on normalized
vectors). bge models expect a query-side instruction prefix, applied
automatically to queries only, never to indexed documents.

**Output:** `data/index.faiss`, `data/chunk_meta.json`, `data/index_config.json`.

Each chunk is embedded with its paper title prepended (`--no-title` turns this
off), because results tables and mid-paper chunks rarely name their own paper.
`index_config.json` records whether titles were used; indexes built without
them keep working as they were.

### 5.4 `retrieve.py` — multi-strategy retrieval *(Iteration 2)*

The main retrieval interface. Four selectable strategies over the same corpus:

| Mode | What it does |
|---|---|
| `dense` | FAISS vector search only (semantic similarity) |
| `bm25` | BM25 sparse keyword search only (exact term matching) |
| `hybrid` | Reciprocal Rank Fusion over dense + BM25 rankings |
| `hybrid_rerank` | hybrid, then cross-encoder reranking (**default**) |

### 5.5 `generate.py` — grounded, cited answers

Retrieves top-k chunks via `HybridRetriever`, builds a prompt with labeled
sources (`[S1]`, `[S2]`, ...), and calls a Groq-hosted LLM with a system prompt
enforcing **strict grounding**: answer only from the provided sources, cite
every claim, and explicitly refuse if the sources are insufficient rather than
falling back on general knowledge.

All retrieval flags from §5.4 are available here too, so the same question can
be answered from different retrieval strategies — which separates two distinct
questions: does better retrieval *rank* better, and does it produce better
*answers*?

### 5.6 `debug_refs.py` — bibliography detection diagnostic *(Iteration 2)*

For any paper, prints where the classifier lands, every occurrence of
"references"/"bibliography" with its position as a percentage through the
document, and the tail of the extracted text. Written while debugging
reference classification; kept because it's the right tool whenever tagging
looks wrong.

---

### 5.7 `query_expansion.py` + `llm.py` — multi-query retrieval *(Iteration 3)*

`--multi-query N` asks the LLM for N rephrasings of the question (varying
terminology, expanding or introducing abbreviations), runs the first retrieval
stage once per phrasing, and fuses the ranked lists with the same RRF used for
dense + BM25. Reranking still scores against the **original** question, so
rephrasings widen the candidate pool without redefining what's relevant.
Rephrasings are cached in `data/query_expansions.json`, so an evaluation sweep
uses identical variants for every configuration and doesn't re-spend quota.

`llm.py` holds the Groq client and rate-limit handling, shared by generation,
query expansion and query-set construction.

### 5.8 Evaluation harness — `src/evaluation/` *(Iteration 3)*

A three-step workflow: **generate** candidate queries with an LLM, **review**
them by hand into ground truth, then **sweep** retrieval configurations
against that ground truth.

| Script | Does |
|---|---|
| `build_queries.py` | samples prose chunks round-robin across papers; asks the LLM for questions each chunk answers, in four categories (below); pins every labelled chunk by a hash of its text |
| `review_queries.py` | interactive terminal review: keep / edit / drop / mark extra relevant chunks; also lets you add your own questions and label them; resumable |
| `run_eval.py` | runs every query through each configuration at k = 1, 3, 5, 10; writes per-query and summary CSV/JSON; prints comparison tables |
| `metrics.py` | pure metric functions, unit-tested against hand-computed values |

Query categories (reported separately, because strategies differ by query type):

| Category | What it tests |
|---|---|
| `semantic` | paraphrased to avoid the passage's own wording — counters the bias of LLM-generated questions reusing chunk vocabulary |
| `exact_term` | contains a specific term, dataset, metric or number from the passage |
| `multi_paper` | needs passages from two different papers; both labelled relevant |
| `unanswerable` | the corpus doesn't answer it; scored by abstention, excluded from recall |

### 5.9 `chat.py` + `uploads.py` — conversational chat over the corpus or your PDFs

Follow-up questions can't be retrieved as they are: in "what about its FPR95?"
the word that matters ("Outlier Exposure") is two turns back. So each turn:

1. **Rewrites** the follow-up into a standalone question using the last few
   turns ("what about its FPR95?" → "What is the FPR95 of Outlier Exposure?").
   The first turn skips this, as there's nothing to resolve.
2. **Expands** it into rephrasings, if `--multi-query` is on.
3. **Retrieves** with the standalone question (hybrid + rerank by default).
4. **Answers** with the recent conversation included for context — but earlier
   answers are not sources. Every claim must cite this turn's sources, and the
   model must still refuse when they don't cover the question.

Sources to chat over:

| Command | Searches |
|---|---|
| `chat.py` | the arXiv corpus |
| `chat.py --pdf a.pdf` | only your PDF(s), in memory — nothing written to the corpus |
| `chat.py --pdf a.pdf --with-corpus` | your PDF(s) and the corpus together |

`uploads.py` runs a PDF through the same cleaning, tokenizer-aligned chunking
and bibliography classification as the corpus. Uploaded chunks cite as
`upload:<file name>`, and each file counts as its own paper for the diversity
cap. PDFs over 80 pages or 30 MB, and scanned PDFs with no text layer, are
rejected with a clear message.

## 6. Command reference

### 6.1 Every session starts here

```bash
cd ~/PROJECTS/uncertainty-rag        # adjust to your actual path
conda activate rag
```

### 6.2 Build the corpus (only when papers change)

Download papers listed in `data/papers.txt`:
```bash
python src/rag/download_papers.py --id-file data/papers.txt
```

Download specific papers directly:
```bash
python src/rag/download_papers.py --ids 1706.03762 1612.01474
```

Force re-download of existing PDFs:
```bash
python src/rag/download_papers.py --id-file data/papers.txt --force
```

Chunk the PDFs:
```bash
python src/rag/chunk_papers.py
```

Chunk with custom parameters:
```bash
python src/rag/chunk_papers.py --chunk-size 512 --overlap 50
python src/rag/chunk_papers.py --tokenizer BAAI/bge-base-en-v1.5
python src/rag/chunk_papers.py --ref-threshold 0.3
```

Chunking strategy (default `fixed` token windows; see §7 "Methods adopted"):
```bash
python src/rag/chunk_papers.py --strategy sentence
```
```bash
python src/rag/chunk_papers.py --strategy semantic --semantic-percentile 90
```

Rebuild tables from the PDF's geometry (no LLM, no API key):
```bash
python src/rag/table_extract.py
```
```bash
python src/rag/table_extract.py --show 1905.13472
```

Figures: render each figure, keep its caption, optionally describe it:
```bash
python src/rag/figures.py
```
```bash
python src/rag/figures.py --describe --provider gemini
```
```bash
python src/rag/figures.py --show 1806.01768
```

Table notes: an LLM description and a rebuilt grid for every chunk with a
results table. Optional; one Groq call per table chunk, cached, resumable.
Run after chunking, before building the index:
```bash
python src/rag/table_notes.py --dry-run
```
```bash
python src/rag/table_notes.py
```
```bash
python src/rag/table_notes.py --show 1905.00076__0012
```

How a run went, and which chunks to redo:
```bash
python src/rag/table_notes.py --stats
```

Redo only the chunks that came back empty (after switching model, say):
```bash
python src/rag/table_notes.py --retry-failed --model openai/gpt-oss-120b
```

On Gemini instead (one request per chunk, ~77 requests):
```bash
python src/rag/table_notes.py --provider gemini
```

77 of this corpus's 408 chunks contain tables, roughly 150k tokens for a full
run, which is about a day of the free tier. `--limit N` does the next N only;
everything is cached and resumable, so several short runs are fine. If Groq's
daily limit stops a run, the same command later continues where it stopped.
`build_index.py build` uses whatever notes exist (`--no-table-notes` to leave
them out).

Build the vector index:
```bash
python src/rag/build_index.py build
```

Build with a different embedding model or batch size:
```bash
python src/rag/build_index.py build --model BAAI/bge-small-en-v1.5
python src/rag/build_index.py build --batch-size 16
```

Inspect what was built:
```bash
wc -l data/chunks.jsonl
head -1 data/chunks.jsonl | python -m json.tool
grep -c '"is_reference": true' data/chunks.jsonl
```

### 6.3 Retrieval — comparing strategies

**Run one strategy:**
```bash
python src/rag/retrieve.py "how does evidential deep learning quantify uncertainty?" --mode dense
python src/rag/retrieve.py "how does evidential deep learning quantify uncertainty?" --mode bm25
python src/rag/retrieve.py "how does evidential deep learning quantify uncertainty?" --mode hybrid
python src/rag/retrieve.py "how does evidential deep learning quantify uncertainty?" --mode hybrid_rerank
```

**Run all four side by side** (the fastest way to see how strategies differ):
```bash
python src/rag/retrieve.py "your query here" --compare
```

**Control result count and candidate pool:**
```bash
python src/rag/retrieve.py "your query" --mode hybrid_rerank -k 10
python src/rag/retrieve.py "your query" --mode hybrid_rerank --candidate-n 40
```

**Switch fusion method** (RRF is rank-based; weighted blends normalized scores):
```bash
python src/rag/retrieve.py "your query" --mode hybrid --fusion rrf
python src/rag/retrieve.py "your query" --mode hybrid --fusion weighted --alpha 0.5
python src/rag/retrieve.py "your query" --mode hybrid --fusion weighted --alpha 0.8   # dense-leaning
python src/rag/retrieve.py "your query" --mode hybrid --fusion weighted --alpha 0.2   # BM25-leaning
```

**Source diversity control:**
```bash
python src/rag/retrieve.py "your query" --max-per-paper 2     # default
python src/rag/retrieve.py "your query" --max-per-paper 1     # strict: one chunk per paper
python src/rag/retrieve.py "your query" --max-per-paper 0     # disabled: pure top-k
```

**Relevance floor for diversity picks:**
```bash
python src/rag/retrieve.py "your query" --min-score-frac 0.25   # default
python src/rag/retrieve.py "your query" --min-score-frac 0.5    # stricter
python src/rag/retrieve.py "your query" --min-score-frac 0      # disabled
```

**Swap the reranker model:**
```bash
python src/rag/retrieve.py "your query" --reranker BAAI/bge-reranker-base       # default, 278M
python src/rag/retrieve.py "your query" --reranker BAAI/bge-reranker-v2-m3      # larger, 568M
python src/rag/retrieve.py "your query" --reranker cross-encoder/ms-marco-MiniLM-L-6-v2   # tiny, ~80MB
```

**Diagnose a miss** — where a chunk you know is relevant ranks at each stage
(dense, BM25, fused, whether it reaches the reranker, its reranker score):
```bash
python src/rag/retrieve.py "EnD2 error rate on CIFAR-10" --explain 1905.00076__0012
```

**MMR instead of the per-paper cap** (λ = 1 relevance only, 0 diversity only):
```bash
python src/rag/retrieve.py "your query" --mmr 0.7
```

**Absolute score floor** (reranker scores are 0–1; nothing left = no answer):
```bash
python src/rag/retrieve.py "hi" --min-score 0.05
```

**Include bibliography chunks in results:**
```bash
python src/rag/retrieve.py "your query" --include-refs
```

**Multi-query expansion** (needs `GROQ_API_KEY`; rephrasings are cached):
```bash
python src/rag/retrieve.py "what is the FPR95 of deep ensembles?" --multi-query 3
python src/rag/retrieve.py "your query" --multi-query 3 --compare
python src/rag/retrieve.py "your query" --multi-query 3 --model openai/gpt-oss-20b
```

**Dense-only search via `build_index.py`** (Iteration 1 interface, still works):
```bash
python src/rag/build_index.py search "your query"
python src/rag/build_index.py search "your query" -k 5
python src/rag/build_index.py search "your query" --include-refs
```

### 6.4 Generation — asking questions

**Basic question (uses hybrid_rerank by default):**
```bash
python src/rag/generate.py "How does evidential deep learning quantify uncertainty?"
```

**Compare how retrieval strategy affects the answer:**
```bash
python src/rag/generate.py "your question" --mode dense
python src/rag/generate.py "your question" --mode bm25
python src/rag/generate.py "your question" --mode hybrid
python src/rag/generate.py "your question" --mode hybrid_rerank
```

**Control how many sources the model sees:**
```bash
python src/rag/generate.py "your question" -k 3
python src/rag/generate.py "your question" -k 8
```

**Switch LLM:**
```bash
python src/rag/generate.py "your question" --model openai/gpt-oss-120b   # default
python src/rag/generate.py "your question" --model openai/gpt-oss-20b    # faster
python src/rag/generate.py "your question" --model qwen/qwen3.8-27b      # different family
```

**See the retrieved chunk text alongside the answer:**
```bash
python src/rag/generate.py "your question" --show-sources
```

**Apply any retrieval option from §6.3:**
```bash
python src/rag/generate.py "your question" --mode hybrid --fusion weighted --alpha 0.3
python src/rag/generate.py "your question" --max-per-paper 0 --min-score-frac 0
python src/rag/generate.py "your question" --candidate-n 40 -k 8
python src/rag/generate.py "your question" --include-refs
```

**Multi-query expansion before retrieval:**
```bash
python src/rag/generate.py "your question" --multi-query 3
```

**Verify strict grounding still holds** (should return the fixed refusal):
```bash
python src/rag/generate.py "What is the boiling point of water?"
```

### 6.5 Chat — follow-up questions

Chat over the corpus:
```bash
python src/rag/chat.py
```

Chat over your own PDF (only that PDF is searched):
```bash
python src/rag/chat.py --pdf ~/Downloads/paper.pdf
```

Your PDF and the corpus together:
```bash
python src/rag/chat.py --pdf ~/Downloads/paper.pdf --with-corpus
```

Several PDFs, and multi-query expansion on every turn:
```bash
python src/rag/chat.py --pdf a.pdf b.pdf --multi-query 3
```

Other options: `-k` sources per answer, `--mode`, `--model`,
`--history-turns` (earlier turns the model sees; default 3), `--max-per-paper`
(default 2, or 0 when chatting over uploads alone), `--mmr LAMBDA`,
`--min-score` (refuse without an LLM call when no source reaches it; off by
default), `--fallback-model` (answers when `--model`'s daily limit is used up;
default `openai/gpt-oss-20b`, `''` to disable).

Greetings and thanks ("hi", "thanks") get a fixed reply with no retrieval and
no LLM call. If both models hit their daily limits, the chat prints Groq's
wait time and keeps running.

Inside the chat: `/sources` (last answer's sources, with text), `/history`
(each question and what it was rewritten to), `/reset`, `/exit`.

### 6.6 Diagnostics

Inspect how a table reaches the model — the flattened text it reads, next to
what PyMuPDF's table finder recovers from the same page:
```bash
python src/rag/debug_tables.py 1806.01768 --find CIFAR5
```
```bash
python src/rag/debug_tables.py 1905.00076 --find "C10 ERR"
```

Use the retriever from LangChain (needs `pip install langchain-core`):
```bash
python src/rag/langchain_adapter.py "what error does EnD2 get on CIFAR-10?" --docs-only
```
```bash
python src/rag/langchain_adapter.py "what error does EnD2 get on CIFAR-10?"
```

When a paper yields no tables, ask why — captions found, figure areas, the
lines around each caption and the reason each side was rejected:
```bash
python src/rag/table_extract.py --debug 1806.01768 --page 7
```

Inspect bibliography classification for specific papers:
```bash
python src/rag/debug_refs.py 1812.04606 2110.03051
```

List available Groq models on your account:
```bash
python - << 'EOF'
import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv(".env")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
skip = ("whisper", "tts", "guard", "orpheus")
for m in sorted(x.id for x in client.models.list().data
                if not any(s in x.id.lower() for s in skip)):
    print(m)
EOF
```

Check which papers are in the corpus:
```bash
python -c "import json; d=json.load(open('data/metadata.json')); [print(r['arxiv_id'], '-', r['title'][:70]) for r in d]"
```

### 6.7 Evaluation (Iteration 3)

**Step 1 — generate candidate queries** (~45 LLM calls, roughly 5–10 minutes on
the free tier; refuses to overwrite an existing candidates file):
```bash
python src/evaluation/build_queries.py
```
```bash
python src/evaluation/build_queries.py --n-chunks 30 --n-multi 15 --n-unanswerable 12
```
```bash
python src/evaluation/build_queries.py --model openai/gpt-oss-20b
```

**Step 2 — review them into ground truth** (resumable; progress saved after
every decision):
```bash
python src/evaluation/review_queries.py review
```
Keys: `k` keep, `e` edit the wording, `d` drop, `a` mark other shown chunks as
also relevant, `s` skip for now, `q` save and quit.

Add questions of your own, and label which chunks answer them:
```bash
python src/evaluation/review_queries.py add
```

Check progress, then export the kept queries to `data/eval/queries.jsonl`:
```bash
python src/evaluation/review_queries.py stats
```
```bash
python src/evaluation/review_queries.py export
```

**Step 3 — run the sweep:**
```bash
python src/evaluation/run_eval.py --suite core --run-name core_v1
```
```bash
python src/evaluation/run_eval.py --suite ablations --run-name ablations_v1
```

Other options:
```bash
python src/evaluation/run_eval.py --list
```
```bash
python src/evaluation/run_eval.py --suite no_llm
```
```bash
python src/evaluation/run_eval.py --configs dense hybrid_rerank --limit 5
```

`--list` shows every configuration and suite; `no_llm` skips multi-query (no
API calls); `--limit` runs only the first N queries as a smoke test.

**Run the unit tests:**
```bash
python -m pytest tests -q
```

### 6.8 Git workflow

```bash
git status                          # always check before adding
git add src/rag/<specific-file>.py  # never `git add .` here — data/ risk
git status                          # confirm what's staged
git commit -m "your message"
git push
```

Verify ignore rules are working:
```bash
git check-ignore -v data/papers.txt data/metadata.json data/index.faiss
```
(`papers.txt` should print nothing — it's meant to be tracked.)

### 6.9 Full pipeline, start to finish

```bash
conda activate rag
cd ~/PROJECTS/uncertainty-rag

python src/rag/download_papers.py --id-file data/papers.txt
python src/rag/chunk_papers.py
python src/rag/table_extract.py        # tables from PDF geometry (no LLM)
python src/rag/figures.py              # figure images + captions (no LLM)
python src/rag/table_notes.py          # table descriptions (uses LLM quota)
python src/rag/figures.py --describe --provider gemini   # optional, vision
python src/rag/build_index.py build
python src/rag/retrieve.py "your query" --compare
python src/rag/generate.py "your question here"
```

Steps 1–4 only need re-running when papers or chunking parameters change
(table notes regenerate only for chunks whose text changed). Steps 5–6 are
per-query.

> **Re-chunking invalidates labels.** The evaluation labels point at chunk IDs.
> If you re-run `chunk_papers.py` with different settings, `run_eval.py` and
> `review_queries.py export` detect the changed text and refuse to run rather
> than silently scoring against the wrong chunks.

> **zsh note:** pasting a multi-line block where a line starts with `#` makes
> zsh try to execute it (`command not found: #`). Run commands one at a time,
> or drop the comment lines.

---

## 7. Design notes

### Iteration 1 decisions

- **No RAG framework** (no LangChain/LlamaIndex) — the pipeline is hand-built
  with `sentence-transformers`, `faiss-cpu`, `rank-bm25`, and the Groq API
  directly, for full control over retrieval and evaluation hooks.
- **Token-based chunking** using the embedding model's own tokenizer, so
  "512 tokens" matches what the embedder actually sees.
- **References are tagged, not dropped** — enables an ablation rather than
  baking in an untested assumption.
- **Strict grounding** in generation — verified empirically: asked "What is the
  boiling point of water?" against an ML-papers corpus, the system returns the
  fixed refusal string instead of answering from world knowledge.
- **Config over hardcoding** — the LLM model name is a CLI parameter. This paid
  off when Groq deprecated `llama-3.3-70b-versatile` on the free tier
  mid-project: the fix was a `--model` override, not a code change.

### Iteration 2 decisions

- **Why hybrid retrieval.** Iteration 1 exposed a concrete weakness: on
  exact-term queries ("FPR95 of deep ensembles on CIFAR-10"), dense retrieval
  returned the right *topical neighborhood* but not the most term-precise
  chunks, and scored notably lower (0.66–0.70) than on semantic queries (0.85).
  Numeric result tables carry weak semantic signal. BM25 rewards exact term
  matches — the missing capability.
- **Why RRF for fusion.** Dense cosine scores (~0–1) and BM25 scores
  (unbounded) live on different scales, so naively summing them is meaningless.
  Reciprocal Rank Fusion combines by *rank position*, sidestepping
  normalization: `RRF(d) = Σ_r 1 / (k + rank_r(d))` with k=60. A
  min-max-normalized weighted fusion is also implemented for comparison.
- **Why two-stage retrieval.** A bi-encoder compares pre-computed vectors —
  fast, approximate. A cross-encoder scores (query, chunk) pairs jointly — far
  more accurate, far too slow corpus-wide. So: retrieve ~20 cheaply, rerank
  those.
- **Why `bge-reranker-base` over `bge-reranker-v2-m3`.** v2-m3 is the current
  open-weight leader but is ~568M params with an 8192-token window. Our chunks
  are 512 tokens, so `bge-reranker-base` (~278M, 512-token max) is right-sized
  and lighter on an 8 GB machine. Swappable via `--reranker`.
- **Source diversity cap.** Inspecting real reranked output revealed the
  cross-encoder collapsing onto a single paper — 5 of 5 results from one
  document on multiple queries. `--max-per-paper` caps each source's
  contribution (default 2), diversifying by *source* rather than embedding
  distance as classic MMR does.
- **Relative relevance threshold.** The cap alone backfires when only one paper
  genuinely answers a query: it pads results with near-zero-scoring chunks.
  `--min-score-frac` requires a diversity-promoted chunk to clear a fraction of
  the top chunk's score. **Relative, not absolute**, because score scales
  differ wildly by mode (reranker ~0–1, cosine ~0–1, BM25 unbounded, RRF ~0.03).

### Observed effects (qualitative, pending Iteration 3 measurement)

- **Better retrieval produced a visibly better answer.** On "how does EDL
  quantify uncertainty?", `dense` gave a correct but shallow answer (Dirichlet
  parameters, vacuity/dissonance). `hybrid_rerank` additionally surfaced the
  mutual-information decomposition with its formula, the aleatoric/epistemic
  split, and the single-forward-pass property — same model, same question,
  better sources.
- **Retrieval quality and answerability are separate axes.** On "what is the
  FPR95 of deep ensembles on CIFAR-10?", `hybrid_rerank` retrieved
  substantially better chunks than `dense` (term-precise, all from the relevant
  paper) — yet **both modes correctly refused**, because the corpus simply
  doesn't contain that number. Improved retrieval does not imply an improved
  answer when the information isn't present. Iteration 3's evaluation must
  measure these independently.
- **The reranker is far better calibrated than dense similarity.** On an
  out-of-domain question ("boiling point of water"), dense scored retrieved
  chunks at 0.51–0.55 — not obviously different from weak in-domain hits —
  while the cross-encoder scored them **0.001 and 0.000**. This suggests
  reranker scores could drive an explicit abstention threshold, worth testing
  in Iteration 3.

---

### Iteration 3 decisions

- **Ground truth is LLM-drafted but human-reviewed.** Fully manual labelling
  takes hours; fully automatic labelling is biased (below). Drafting a pool and
  reviewing it down to ~50 is the middle ground, and every kept query has been
  read by a person.
- **Countering generation bias.** LLM-written questions tend to reuse the
  source passage's wording, which flatters any retriever that matches words.
  The `semantic` category is explicitly instructed to paraphrase, and the
  review tool lets you add hand-written questions.
- **Categories, not one average.** Iteration 2's manual testing showed BM25
  helping exact-term queries and diversity helping broad ones; a single mean
  would hide exactly the effects worth reporting.
- **Labels pinned by content hash.** Chunk IDs are `arxiv_id__index`, so
  re-chunking with a different size silently repoints them at different text.
  Every label stores a hash of the chunk text, and evaluation refuses to run if
  any has changed. A consequence: chunk size can't be evaluated as an ablation
  with these labels.
- **Per-cutoff searches.** The diversity cap means the top 5 of a k=10 search
  isn't what the system returns at k=5, so each cutoff is its own search. The
  retriever caches query embeddings and reranker scores, so this costs little.
- **Paired bootstrap for significance.** Both configurations answer the same
  queries, so the CI is computed on per-query differences (2,000 resamples).
  With ~50 queries, a few points of Recall@5 can be noise; the interval says
  whether a gain is real.
- **Abstention as OOD detection.** Whether a question falls outside the corpus
  is scored as AUROC of the top-1 retrieval score for answerable vs
  unanswerable queries — the same framing and metric as the uncertainty module
  in Iteration 4.
- **Multi-query reranks against the original question.** Rephrasings are used
  only to widen the first-stage candidate pool. If the reranker scored against
  a rephrasing, an LLM paraphrase could shift what counts as relevant.
- **Expansions cached per (model, n, question).** Without the cache each
  configuration would get different rephrasings, and the comparison between
  configurations would no longer be controlled.
- **Shared `llm.py`.** Query expansion, query generation and answer generation
  all call Groq; one module owns the client and the 429 handling, and nothing
  else has to import the generation pipeline to get at it.

### Retrieval fixes from the chat session

- **Normalise names, don't paraphrase them.** BM25's strength is exact
  matching, so rather than fuzzy matching, the tokenizer makes different
  spellings of the same name produce the same token (NFKC, joined hyphenated
  names, a small explicit alias table). The alias table is short and visible on
  purpose; a sprawling hidden synonym list is hard to reason about.
- **Index the title, store the text.** The title is prepended only to what's
  embedded, BM25-scored and reranked. Chunk text is untouched, so evaluation
  labels pinned to its hash stay valid and no re-chunking is needed.
- **Fix context, not just ranking.** When the right chunk is unreachable by
  any ranker (a table of numbers), attach it to a neighbour that *is*
  reachable. Split tables are detected from the hit's first/last 30 tokens and
  the neighbour is merged in, capped at two per answer.
- **Old indexes keep their behaviour.** `index_config.json` records whether
  titles were embedded, and the retriever (and any upload merged into that
  index) follows it, so vectors and query-time text never disagree.

### Conversational chat decisions

- **Rewrite, then retrieve.** Retrieving on the raw follow-up fails whenever
  it uses a pronoun; retrieving on the whole conversation drags in earlier
  topics. A rewrite into one standalone question fixes both, at the cost of one
  extra LLM call per follow-up.
- **History is context, not evidence.** Earlier answers are passed to the
  model so replies read as a conversation, with their `[S1]` citations removed
  first — source numbers restart every turn, so an old `[S2]` would point at a
  different chunk than the current one.
- **Bounded history.** Only the last 3 turns, each answer cut to ~700
  characters, so long chats stay inside the free tier's 6,000 tokens/minute.
- **The model sees both phrasings.** The final prompt carries the user's
  original follow-up (so "explain that more simply" is answered simply) plus the
  standalone rewrite (so it knows what "that" is).
- **Uploads stay in memory.** `from_chunks` copies the corpus vectors out of
  the FAISS index instead of re-embedding them, embeds uploads with the corpus's
  own model (mixed embedding models give incomparable scores), and writes
  nothing to disk.

### Observed so far (on synthetic test data; real numbers pending)

- **RRF scores carry no abstention signal.** A fused RRF score depends only on
  rank positions, so the top-1 score is ~2/61 for every query, relevant or not.
  In the offline test run, plain `hybrid` scored abstention AUROC 0.5 (chance)
  while the reranked configuration separated answerable from unanswerable
  cleanly. Expect the same pattern on real data: abstention needs an absolute
  relevance score (reranker or cosine), not a fused rank.

### Methods adopted from rag-for-beginners

[harishneel1/rag-for-beginners](https://github.com/harishneel1/rag-for-beginners)
is a tutorial series built on LangChain, Chroma, OpenAI and Cohere. It covers
most of what this project already has; what it adds was re-implemented here in
the existing stack (PyMuPDF, bge + FAISS, rank_bm25, bge-reranker, Groq), with
no new dependencies.

| Tutorial step | Their implementation | Here |
|---|---|---|
| Ingestion, retrieval, grounded answer (1–3) | LangChain loaders, Chroma, gpt-4o | Already built: PyMuPDF, FAISS, Groq, citations and refusal |
| History-aware chat (4) | Rewrite follow-up, then retrieve | Already built, plus bounded history and "history is not a source" |
| Recursive splitter (5) | Split on `\n\n`, `\n`, `. `, ` ` to fit a size | **Added** as `--strategy sentence`: whole sentences up to 512 tokens, token windows for anything longer |
| Semantic chunking (6) | `SemanticChunker`, percentile breakpoints | **Added** as `--strategy semantic`: same method with bge embeddings |
| Agentic chunking (7) | LLM inserts split markers | **Not added**: one LLM pass over the whole corpus (~200k tokens) exceeds a Groq free-tier day, for chunks that can't be evaluated against our labels |
| Multimodal RAG (8) | unstructured `hi_res` (OCR, layout model), GPT-4o summaries of text, tables, images | **Added for tables** as `table_notes.py`: LLM description indexed, rebuilt grid shown to the model, grid checked against the text. `find_tables` could not recover these papers' tables (§8), and unstructured's layout model needs Tesseract, Poppler and a detection model on an 8 GB Mac. Figures are left out: they are plots, and answering from them needs a vision model |
| Score threshold, MMR (9) | Chroma retriever options | **Added**: `--min-score` (absolute floor, refuses without an LLM call) and `--mmr` (content diversity, as an alternative to the per-paper cap). MMR is in the `ablations` suite; the floor's value comes from the abstention report |
| Multi-query + RRF (10–11) | Pydantic structured output, RRF over variants | Already built; rephrasings are cached and reranking scores the original question |
| Hybrid search (12) | `EnsembleRetriever`, weights 0.7 / 0.3 | Already built: RRF by default, weighted fusion with `--alpha` |
| Reranker (13) | Cohere `rerank-english-v3.0` (API) | Already built with a local cross-encoder (bge-reranker-base): no API key, no per-call cost |

What this project has that the tutorial doesn't: BM25 tokenisation fixes
(CIFAR-10 = C10, EnD² = EnDD), title-prefixed indexing, bibliography tagging,
per-paper diversity, split-table expansion, uploads with a corpus toggle, and
an evaluation harness with labelled queries, significance tests and an
abstention analysis.

### Tables and figures: three ways, in order of trust

A flattened table is a stream of numbers, and answering from one means
counting positions - which is where the wrong-column answers came from (§8).
Three mechanisms now run, and the strongest one available wins:

1. **Geometry** (`table_extract.py`). PyMuPDF gives every word's bounding box.
   Words are grouped into lines by vertical position, split into cells at gaps
   wider than a word space, anchored to a "Table N:" caption, and lined up into
   columns. Nothing is generated, so nothing can be invented or moved. It
   fails on layouts it doesn't fit, and says so by producing nothing.
2. **LLM notes** (`table_notes.py`). Used for the table description that goes
   into the index either way, and for the grid when geometry found none. Every
   number in a generated grid is checked against the source text.
3. **The flattened text**, always in the prompt, and named as the authority if
   it disagrees with a grid.

Figures (`figures.py`) get the same treatment in miniature: the image is
rendered from the PDF (deterministic), the caption is attached to whichever
chunk refers to that figure ("as shown in Figure 3"), and a vision model
optionally describes the plot for retrieval. The description is never a source
of numbers - the prompt tells the model not to read values off a plot.

Why the caption is attached by reference and not by page: the sentence
discussing Figure 3 is often a page away from the plot, and that sentence is
the chunk a question about the figure should retrieve.

### Why the pipeline is framework-free (and where LangChain fits)

The tutorial stack (LangChain + Chroma + an API reranker) covers the same
ground as Iterations 1-3, and would have made the first version faster to
write. It would not have made any of the bugs in §8 findable: the
bibliography classifier, the diversity cap ordering, the split-table
expansion, the BM25 alias tokeniser, the page-number filter that deleted table
columns - each was found by reading our own code and printing our own
intermediate ranks. `--explain` exists because the retriever is ours.

So the pipeline keeps no framework, and `langchain_adapter.py` exposes
`HybridRetriever` as a LangChain `BaseRetriever` instead, so a LangChain
application can use this retriever without the project depending on LangChain.
The demo builds a small chain out of `langchain_core` primitives and this
project's own prompt and LLM client.

## 8. Debugging log

Each entry is a concrete example of diagnosing rather than guessing.

### Bibliography detection: three attempts

| Approach | Papers detected | Ref chunks | Problem |
|---|---|---|---|
| Heading must start a page | 3 / 12 | 25 | Missed most papers entirely |
| Full-text heading scan | 12 / 12 | 160 (40% of corpus) | Swept in appendices |
| **Per-chunk citation density** | **12 / 12** | **70 (17%)** | — |

The second attempt looked like a regex problem, but a diagnostic script
(`debug_refs.py`) showed the heading was being found almost exactly right —
within 1 character on two papers, 250 characters on a third. The real fault was
**architectural**: tagging everything after the references heading also tagged
the **appendix**, because papers commonly run body → references → appendix.
Appendix content (proofs, extra results, experimental detail) is real content
that should stay retrievable.

The fix abandoned heading detection entirely in favour of classifying each
chunk independently by **citation density** — a weighted count of citation
markers (years, `[1]`, `et al.`, `Surname, A.`, venue names, page ranges,
full-name author lists) per word. Calibrated against real chunks from the
corpus: bibliography entries score 0.35–0.79, ordinary prose 0.00, and the
hardest negatives — prose that cites work ("Following Sensoy et al.
(2018)...") and results text full of years and numbers — score 0.23 and 0.06
against a 0.25 threshold. This needs no heading, no position heuristic, and no
assumption about document structure.

The transferable lesson: *stop patching a heuristic once the failure turns out
to be structural rather than parametric.*

### Diversity cap silently dropped the top results

The first implementation of `apply_diversity_cap` returned results out of score
order and dropped the two highest-scoring chunks entirely. Cause: rejected
chunks were appended after the selected list without re-sorting, so chunks
hitting the per-paper cap were relegated below weaker ones that had squeaked
through. Fixed by holding rejected chunks in a reserve, backfilling in score
order, and re-sorting before return.

Caught not by a failing test but by **comparing live output against an earlier
run of the same query and noticing the top result had changed** — an argument
for keeping human eyes on retrieval quality rather than trusting aggregate
metrics alone.

### Chat attributed another paper's number to the wrong paper

Asked "what accuracy on CIFAR-10 does the EnDD paper get?", the chat answered
"The EnDD paper reports an error rate of 11.46%" — citing a chunk from
*Simple Regularisation for Uncertainty-Aware Knowledge Distillation*
(2205.09526), which reports EnDD only as a baseline in its own experiments.
Root cause: source labels carried only the arXiv ID (`S1: 2205.09526 (2022),
pp.11-12`), so the model had no way to know which paper a chunk came from. Fix:
every source label now includes the paper title, and a prompt rule requires
numbers to be attributed to the paper that reports them ("reported by <paper>
for <method>").

The user's figure (~92%) was right: Table 3 of the EnDD paper gives EnD²
7.3% error on CIFAR-10 (92.7% accuracy). The chat refused, and the reason was
retrieval, not generation. The chunk holding those rows (`1905.00076__0012`)
was never retrieved in three differently-worded attempts; only the table's
second half (`__0013`, without the CIFAR-10 rows) was. The question and the
chunk shared almost no tokens:

| Question said | Table chunk says |
|---|---|
| CIFAR10 / CIFAR100 | C10 / C100 |
| EnDD | EnD2 (EnD² after NFKC) |
| accuracy | Classification Error, ERR |
| "the EnDD paper" | the paper is never named |

A table embeds weakly, and BM25 found no overlap, so the chunk never reached
the reranker's 20 candidates. Given what it was shown, the model's refusal was
correct.

Two fixes, both at index time so no chunk text (and no evaluation label)
changes:

1. **Name normalisation in BM25** — NFKC, hyphenated names joined
   (`CIFAR-10` → `cifar10`), and a small explicit alias table
   (`C10`→`cifar10`, `C100`→`cifar100`, `EnDD`→`end2`), applied to both
   queries and documents.
2. **Paper title indexed with every chunk** — prepended to the text that is
   embedded, BM25-scored and reranked (a light form of contextual retrieval).

An offline check on 55 chunks from the terminal logs suggested these would
move the table chunk into BM25's top few. **On the full 405-chunk index they
didn't**: `--explain` showed it at #36 (BM25) and #62 (dense), outside the
reranker's 20 candidates, and the reranker scored it 0.0012 anyway. At full
scale "cifar10" and "end2" occur in dozens of chunks, so they carry little
weight, and the table's many number tokens make the chunk long, which BM25
penalises. The small sample overstated the effect — a reminder to test on the
real index before claiming a fix.

The same `--explain` output showed the way out: the chunk ranked **#1 (0.99)
was `__0013`, the second half of the same table**, which starts mid-table. So
the fix moved from ranking to context assembly: when a retrieved chunk starts
or ends inside a table (mostly numbers, `±`, `NA` in its first or last 30
tokens), its neighbouring chunk is merged in before the model reads it, with
the overlap removed. The model then sees the whole table, including the
C10/C100 rows. At most two hits are expanded per answer, to stay inside the
free tier's tokens-per-minute budget. Retrieval metrics are unaffected, since
this happens after ranking.

The normalisation and title changes stay: they are principled, cost nothing,
and `build_index.py --no-title` makes the title's effect measurable in the
evaluation sweep. The multi-query prompt now also says that accuracy is
usually reported as error rate, and that tables abbreviate dataset names, and
its cache is keyed by the prompt's hash so edited prompts don't reuse stale
rephrasings.

Still open: table-aware extraction (PyMuPDF `find_tables`), which would keep
tables whole and label their cells, but re-chunks the corpus and so
invalidates the evaluation labels.

Smaller fixes from the same session: gpt-oss's own citation style
(`【S1†L5-L7】`) is normalised to `[S1]`; a rewrite that only re-capitalises
the question is ignored instead of shown as "searched for"; typing `quit`,
`exit` or `bye` ends the chat instead of becoming a question.

### Retrieval fixed, table reading not

With the split-table fix in place the right table reached the model, and the
answers were still wrong:

| Asked | Answered | That value is… | Correct (EnD²) |
|---|---|---|---|
| EnDD error, CIFAR-10 | 6.7 ± 0.3 | EnD (a baseline) | 7.3 ± 0.2 |
| EnDD error, CIFAR-100 | 28.0 ± 0.4 | EnD | 27.9 ± 0.3 |
| EnDD error, TinyImageNet | 38.3 ± 0.2 | EnD | 37.6 ± 0.2 |
| EDL accuracy, CIFAR-5 | 99.3 (same as MNIST) | — | 83 |

The three EnD² answers all read the column *next to* EnD², consistently, so the
model took "EnDD" to mean EnD. The CIFAR-5 answer repeated the MNIST value and
held to it when challenged. Both are failures to read a flattened table:
extraction turns a table into a stream of numbers, and the model has to count
positions to know which number belongs to which row and column.

Two responses:

1. **Auditable answers now.** A new prompt rule makes the model name the row
   and column of every number it takes from a table, give the paper's own
   method when asked what a paper reports (labelling baselines as such), and
   say the table is ambiguous rather than guess. That would have turned the
   silent EnD/EnD² swap into a visibly labelled one.
2. **Measure before building extraction.** `debug_tables.py` shows the exact
   flattened text the model received next to what PyMuPDF's `find_tables`
   recovers from the same page, under both detection strategies. On a
   synthetic booktabs-style table (horizontal rules only, as in LaTeX papers),
   `lines` found nothing and `text` recovered every row and column correctly.
   Whether that holds on the real PDFs decides whether table-aware extraction
   is worth building.

### The CIFAR-5 column was never in the index

`debug_tables.py 1806.01768 --find CIFAR5` settled both open questions.

**The model wasn't misreading the table.** The flattened text of chunk
`1806.01768__0011` was:

```
Method MNIST CIFAR 5 L2 99.4 Dropout 99.5 Deep Ensemble 99.3 ... EDL 99.3
```

Two headers, one value per row. The CIFAR-5 column (76, 84, …, 83) was gone
before chunking. The cause was a page-cleaning rule meant to remove page
numbers: it dropped **every** line that was a bare integer. PyMuPDF emits each
table cell on its own line, so every whole-number cell was deleted. Decimal
columns (`99.4`) survived, which is why the EnD² table looked complete while
this one didn't.

Fix: a bare 1–4 digit number is dropped only when it is the first or last line
of the page, where headers and footers sit. Unit tests and a PyMuPDF-generated
PDF test check that integer cells survive and the footer number doesn't. The
remaining edge case is a page with no page number that ends on an integer
table cell. Its last cell would still be dropped. That is rare and much less
harmful than the old rule.

**Prompt rule 6 didn't catch it.** With two headers and one value per row, the
model could have noticed the mismatch and said the table was ambiguous.
Instead it listed the MNIST values as CIFAR-5, and added BBH/GEN numbers from
another paper. A prompt rule can't check what the text doesn't contain. The
fix belongs in extraction.

**`find_tables` isn't reliable enough to build on.** On the real PDFs:

- `lines` squashed 1806.01768 p.7 into a 4×2 table and split 1905.00076 p.7
  into six fragments.
- `text` treated whole pages as tables (54×8, 79×6).

The synthetic booktabs test had been too clean. Table-aware extraction is
shelved. The flattened text is correct again once page cleaning stops
deleting cells.

This fix changes chunk text, so it needs a re-chunk, an index rebuild and new
eval candidates. The candidates' hash guard would reject the old labels anyway.

### After the page-number fix: one fixed, three new

The re-run confirmed the CIFAR-5 fix: the chunk now reads
`L2 99.4 76 Dropout 99.5 84 … EDL 99.3 83`, and the chat listed all seven
CIFAR-5 accuracies correctly, EDL at 83%. The same session showed three
other problems:

1. **"hi" ran the whole pipeline.** It was retrieved, reranked and sent to the
   model, which refused. Two LLM calls on a daily-capped free tier bought
   nothing. Greetings and thanks now get a fixed reply with no retrieval.
   For anything else off-topic, `--min-score` refuses before the LLM call. It
   is off by default until the evaluation's abstention report gives a
   threshold.
2. **Waiting half an hour for Groq.** After a day of query generation, Groq
   asked for 386 s, then 1,798 s: the model's daily token limit, not the
   per-minute one. The client slept through both. Now any wait over 90 s
   (`GROQ_MAX_WAIT`) stops with a message naming the limit. The chat instead
   switches to a fallback model (`openai/gpt-oss-20b`), which has its own
   quota, and says so. Evaluation and query generation never fall back, so
   one run never mixes models.
3. **Another column misread.** Asked for TinyImageNet OOD-detection AUROC,
   the model reported values from the neighbouring LSUN column, the same
   kind of error as EnD vs EnD². This is what the rebuilt grids from
   `table_notes.py` are for: row/column alignment is worked out once, per
   table, with the whole table in view, and checked against the text,
   instead of by counting positions while answering.

### The table notes came back empty 57 times out of 69

The first real run went to `openai/gpt-oss-20b` (the 120b model's daily limit
was gone), and produced 12 descriptions out of 69 chunks. Everything else was
recorded as "reply was not valid JSON".

The cause is in the output format, not the model. A Markdown grid is
multi-line, JSON strings are not: a newline inside a JSON string has to be
written `\n`, and models routinely emit a real newline instead, which makes the
whole reply unparseable. The bigger the table, the more likely it is to happen.

Two changes:

1. **The reply is now line-based**, `TABLE / CAPTION: / DESCRIPTION: / GRID: /
   END`, where a multi-line grid is the natural thing to write. JSON is still
   accepted when it arrives.
2. **Failures are visible.** A reply that fits no format is stored with its
   first 300 characters. `--stats` prints the split — usable grids, descriptions
   only, no table found, unparseable — and `--retry-failed` redoes just the
   empty ones, which matters when each attempt costs quota.

A related lesson for anything LLM-shaped: `--dry-run` said 77 chunks and the
run reported "12 described", and only `--show` made the failure visible. A
batch job needs its own summary, or a quiet failure reads as a small result.

### A retired model id, and a table printed beside a figure

The first real run of the new pieces turned up two things.

**Gemini retired `gemini-2.5-flash`** between the docs and the run: *"no longer
available to new users. Please update your code to use models/gemini-3.6-flash"*.
Hard-coding the new id would only postpone this, so the backend now treats a
404 as recoverable: it takes the replacement out of Google's own error
message, or, failing that, picks the closest model the account can list
(same family, highest version, previews last), prints which it used, and
retries. `providers.py --list` prints the catalogue for either provider.

**Geometry found 37 tables in 12 papers, and none at all in four of them** -
including Sensoy et al., whose CIFAR-5 table started this whole thread. That
page prints Figure 2 on the left and Table 1 on the right, so:

- the caption was never found: grouping words by vertical position puts the
  figure's caption and the table's header on the same line, and the line does
  not *start* with "Table 1:". Captions are now matched per cell, not per
  line.
- the figure's tick labels would have become extra columns. Cells that fall
  inside a figure's drawing area are now dropped first (thin rectangles are
  ignored, because a booktabs rule is a drawing too and lies across the
  table).
- the table is wider than its caption, so the row nearest the caption lost its
  outer columns. The block is grown twice: once to learn how wide the table
  is, then again with that width.

**Gemini answered 503** on the next run: *"This model is currently
experiencing high demand."* Only 429s and connection errors were being
retried, so one busy minute killed a 77-chunk batch job. Server errors are now
backed off and then handed to the fallback model, on both providers. The
detection is anchored (`^50[0234]` or the status word) so that a quota message
reading "Limit 500 requests per day" is not mistaken for a server error.

**Then the connection itself dropped** (`[SSL: UNEXPECTED_EOF_WHILE_READING]`),
which surfaces as an httpx/httpcore exception rather than an API error, so it
went straight through. Two changes:

- dropped connections are now retried exactly like a busy server, recognised
  by exception class along the cause chain (httpx, httpcore and ssl each raise
  their own types, and the SDK doesn't wrap them);
- when a provider gives up for good it raises `ProviderUnavailable`, and the
  batch jobs (`table_notes.py`, `figures.py --describe`) catch it per item:
  that chunk is skipped and left without notes, so a plain re-run picks it up.
  Three failures in a row stop the run, because at that point the provider or
  the network is down and each further chunk would only spend a minute in
  back-off.

`--debug ARXIV_ID --page N` prints what the page looks like to the extractor:
lines with their cell counts, which captions matched, and why each side of
each caption was rejected. That is the difference between tuning thresholds by
guesswork and reading what actually happened.

### Other issues worth noting

- **`pip install X` installed to the wrong Python:** the shell's `pip` resolved
  to a different interpreter than the active conda env. Always use
  `python -m pip install X` after `conda activate rag`.
- **Groq model 404:** `llama-3.3-70b-versatile` was deprecated on the free tier
  (June 2026). Fix: query the account for available models (§4) and pass the
  current one via `--model`.
- **`git status` shows an untracked `data/` directory** even with correct
  ignore rules — normal git behaviour (it collapses a directory into one line
  when nothing inside it is tracked). Verify with `git check-ignore -v <path>`.
- **Ligature artifacts** in extracted PDF text ("classiﬁcation") — fixed via
  Unicode NFKC normalization.
- **zsh and `#` comments:** pasting a multi-line block where a line starts with
  `#` makes zsh try to execute it. Run commands one at a time.

---

## 9. Ablation surface and evaluation

Every one of these is a runtime flag, not a code change — so the evaluation
harness sweeps them without touching pipeline code:

| Dimension | Flag | Values |
|---|---|---|
| Retrieval strategy | `--mode` | dense, bm25, hybrid, hybrid_rerank |
| Query expansion | `--multi-query` | 0 (off), N rephrasings |
| Fusion method | `--fusion` | rrf, weighted |
| Dense/sparse balance | `--alpha` | 0.0–1.0 (weighted fusion only) |
| First-stage pool size | `--candidate-n` | any integer |
| Source diversity | `--max-per-paper` | 0 (off), 1, 2, 3, ... |
| Relevance floor | `--min-score-frac` | 0.0 (off) – 1.0 |
| Bibliography inclusion | `--include-refs` | on / off |
| Bibliography threshold | `--ref-threshold` | citation density cutoff |
| Reranker model | `--reranker` | any HF cross-encoder |
| Chunk size / overlap | `--chunk-size`, `--overlap` | any integers |
| Chunking strategy | `chunk_papers.py --strategy` | fixed, sentence, semantic |
| Final selection | `--mmr` | off (per-paper cap), λ in 0–1 |
| Absolute score floor | `--min-score` | off, 0–1 (reranker scores) |
| Table notes in index + prompt | `build_index.py build --no-table-notes` | on (if generated) / off |
| Geometric table grids | `build_index.py build --no-tables` | on (if generated) / off |
| Figure captions + descriptions | `build_index.py build --no-figures` | on (if generated) / off |
| LLM provider | `--provider` | groq, gemini |
| Title in indexed text | `build_index.py build --no-title` | on (default) / off |
| Generation model | `--model` | any available Groq model |
| Sources per answer | `-k` | any integer |

**Evaluation suites** (`run_eval.py --list` prints the exact settings):

| Suite | Configurations | LLM calls |
|---|---|---|
| `core` | dense, bm25, hybrid, hybrid_rerank, hybrid_rerank+mq3 | one expansion per query, cached |
| `ablations` | multi-query on each mode; weighted fusion α = 0.3 / 0.7; diversity cap off / 1; relevance floor off; candidate pool 10 / 40; bibliography included; MMR λ = 0.7 / 0.5 | same cache |
| `no_llm` | dense, bm25, hybrid, hybrid_rerank | none |

Chunk size, `--ref-threshold` and the reranker model need a re-chunk, a
re-classification or a code-level change respectively, so they are outside the
sweep for now.

**Table notes with and without.** Notes change the index but not the chunk
text, so the same labels work for both. Build without, run, build with, run:
```bash
python src/rag/build_index.py build --no-table-notes
python src/evaluation/run_eval.py --suite no_llm --run-name notes_off
python src/rag/build_index.py build
python src/evaluation/run_eval.py --suite no_llm --run-name notes_on
```

**Chunking strategies can't be compared with the current labels**: labels are
chunk IDs pinned to chunk text, and a different strategy produces different
chunks. Comparing them needs a query set reviewed per strategy (or labels
defined as answer text rather than chunk IDs).

**Outputs** (`results/eval/<run-name>/`, small enough to commit):

| File | Contents |
|---|---|
| `summary.csv` | configuration × every metric (answerable queries) |
| `summary_by_category.csv` | the same, split by query category |
| `significance.csv` | Recall@5 and MRR difference vs dense, with 95% CI |
| `abstention.csv` | AUROC and best-threshold accuracy per configuration |
| `per_query.jsonl` | every query × configuration: metrics and ranked chunk IDs |
| `run_info.json` | configurations, query-set hash, corpus size, timestamp |

**Hypotheses to test:**
1. Source diversity helps broad questions ("what methods exist for uncertainty
   estimation?") and hurts narrow factual ones. Early evidence supports this:
   the broad query returned 5 strong chunks across 4 papers (all 0.98+), while
   the narrow query's diverse alternatives scored an order of magnitude below
   the top hit.
2. Reranker scores are well-calibrated enough to serve as an abstention signal
   (0.000–0.001 on out-of-domain queries vs 0.5–0.99 on in-domain).
3. Retrieval improvements and answer improvements are only partly correlated —
   a query can have better retrieval yet the same (correct) refusal, if the
   corpus lacks the information. (Needs generation-level evaluation; not yet
   built.)
4. Multi-query expansion helps most on `exact_term` queries whose wording
   differs from the paper's (abbreviation vs spelled out), and little on
   `semantic` ones.

---

## 10. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Matches library compatibility across the stack |
| Environment manager | conda (Miniconda) | Isolates dependencies; standard for ML work |
| PDF source | arXiv API via the `arxiv` library | Free, reliable, structured metadata alongside the PDF |
| PDF parsing | PyMuPDF (`pymupdf`) | Fast, page-accurate extraction; preserves page numbers for citations, and its word boxes are what tables and figures are rebuilt from |
| LLM providers | Groq (`groq`), Gemini (`google-genai`, optional) | Both free; Groq is faster, Gemini's free tier is per-request and takes images |
| LangChain | `langchain-core` (optional, adapter only) | The pipeline stays framework-free; the adapter lets LangChain apps use the retriever |
| Text normalization | Python `unicodedata` (NFKC) | Fixes typographic ligatures LaTeX PDFs embed |
| Tokenization (chunking) | HF `transformers` `AutoTokenizer` (bge's own) | Chunk sizes measured in the *actual* tokens the embedder sees |
| Embedding model | `BAAI/bge-base-en-v1.5` via `sentence-transformers` | Free, local, strong on MTEB for its size; runs on Apple Silicon MPS |
| Dense vector search | FAISS (`faiss-cpu`), `IndexFlatIP` | Exact inner-product over normalized embeddings (= cosine); no server needed |
| Sparse retrieval | `rank-bm25` (`BM25Okapi`) | Pure Python, transparent, built in memory at load (<1s at this scale) |
| Fusion | Reciprocal Rank Fusion (own implementation) | Rank-based, so no score normalization across incompatible scales |
| Reranking | `BAAI/bge-reranker-base` via `CrossEncoder` | Joint (query, chunk) scoring; 512-token window matches our chunks |
| LLM (generation) | Groq API, `openai/gpt-oss-120b` | Free tier, no credit card, very fast inference |
| LLM client | `groq` Python SDK | Typed exceptions (`RateLimitError`) for clean retry logic |
| Query expansion | LLM rephrasings + RRF (own implementation) | Recovers vocabulary the user didn't use; reuses the existing fusion code |
| Conversation | LLM rewrite of follow-ups + bounded history (own implementation) | Makes follow-ups retrievable; no agent framework needed |
| Evaluation | own metrics + paired bootstrap, `pytest` | Metrics small enough to write and unit-test directly; no evaluation framework to learn or trust blindly |
| Secrets management | `python-dotenv` + `.env` (gitignored) | Keeps the API key out of source control |
| Config / control flow | plain `argparse` per script | Each module independently runnable and scriptable |
| Version control | Git + GitHub | `.gitignore` keeps PDFs, embeddings, indexes out of the repo |

**Deliberately not used (yet, or at all):**
- **LangChain / LlamaIndex** — the pipeline is simple enough (parse → chunk →
  embed → search → fuse → rerank → prompt → generate) to hand-write, keeping
  every step inspectable and evaluable.
- **A hosted/managed vector DB (Qdrant, Pinecone)** — deferred to an optional
  deployment iteration; FAISS in-process is sufficient at this scale.
- **Paid LLM APIs (OpenAI, Anthropic)** — unnecessary; Groq's free tier with an
  open-weight 120B model handles this corpus well.
- **A persisted BM25 index** — at hundreds of chunks, building it in memory at
  load time costs under a second and can never drift out of sync with
  `chunks.jsonl`. Worth revisiting at tens of thousands of chunks.

---

## 11. What each script does

### `download_papers.py` — corpus acquisition

**Purpose:** turn a list of arXiv IDs into a local corpus of PDFs plus
structured metadata.

**How it works:**
1. Reads arXiv IDs from `data/papers.txt` (or `--ids`), normalizing whatever
   form they're given in (bare ID, versioned ID, `arxiv.org/abs/...` URL).
2. Sends one batched request to the arXiv API for metadata on all requested IDs
   at once — title, authors, year, abstract, categories, canonical URLs.
3. Downloads each PDF via `requests`, checking the response actually starts
   with the PDF magic bytes (`%PDF`) rather than an HTML error page.
4. Writes `data/metadata.json` — the *source of truth* for every citation the
   system later produces.
5. Idempotent: re-running skips PDFs already on disk and merges new metadata.

**Why it's built this way:** the API gives clean structured fields (vs
scraping); separating one batched metadata call from individually throttled PDF
downloads is a natural politeness/performance split; idempotency is a basic
data-pipeline habit one-shot scripts usually skip.

---

### `chunk_papers.py` — PDF parsing and chunking

**Purpose:** turn raw PDFs into clean, retrieval-ready chunks carrying enough
metadata to produce an accurate citation.

**How it works:**
1. Opens each PDF with PyMuPDF and extracts text **page by page**, tracking
   which page every character came from — this is what lets a citation say
   "page 4" rather than just naming the paper.
2. Cleans each page: strips page numbers, arXiv stamps and similar furniture;
   de-hyphenates words split across line breaks; applies Unicode NFKC
   normalization; collapses whitespace into paragraph text.
3. Concatenates pages into one string per paper while recording which character
   ranges belong to which page (a "page map").
4. Tokenizes with the **same tokenizer the embedding model uses**, then slides
   a 512-token window with 50-token overlap, using the tokenizer's
   character-offset mapping to know exactly which text and page(s) each chunk
   spans. `--strategy sentence` packs whole sentences up to the same limit
   instead; `--strategy semantic` also cuts where consecutive sentences'
   embeddings are unusually far apart. The strategy is saved to
   `data/chunk_config.json` and recorded in the index config.
5. Classifies each chunk independently as content or bibliography via
   **citation density**, compared against `--ref-threshold` (default 0.25).
   Bibliography chunks are tagged `is_reference: true`, never dropped.
6. Writes every chunk as one line of JSON to `data/chunks.jsonl`.

**Why it's built this way:** token-based chunking means "512 tokens" is
literally true for the embedder; page-level attribution is what makes citations
meaningful; per-chunk density classification replaced a structurally-flawed
heading-span approach (§8) and requires no assumption about where in a document
the bibliography sits.

---

### `table_extract.py` — tables from PDF geometry

**Purpose:** recover a results table as a real grid, without an LLM.

**How it works:** words are grouped into lines by vertical position (PyMuPDF's
own line grouping puts each column in its own line when a row is several text
objects); each line is split into cells wherever the gap reaches 2.5 word
spaces (the space width is measured from the page's small gaps, because a
median over a table-heavy page is a column gap); a "Table N:" caption anchors
the search, and lines are taken away from it while they stay multi-cell and
close together; cells are assigned to columns by x-overlap with the widest
row, and a second header row is folded into the first. A grid is kept only if
it has 3+ rows, 2+ columns, enough numeric cells and no empty column.

**Why it's built this way:** PyMuPDF's own `find_tables` failed on these
papers (§8) because it looks for tables anywhere on the page. Starting from
the caption is what makes it work, and everything downstream (`--show`) is
there because a table extractor that can't be checked by eye can't be trusted.

---

### `figures.py` — figure images, captions and descriptions

**Purpose:** make the plots retrievable and displayable.

**How it works:** finds "Figure N:" captions (the separator is required, so a
wrapped prose line starting "Figure 2 shows" isn't one), unions the vector
drawings and images next to the caption into a rectangle, renders it to a PNG
at 120 dpi, and records the caption. `--describe` sends each image to a vision
model (Gemini) for a description of axes, curves and trend. `build_index.py`
attaches caption and description to every chunk that refers to that figure, so
both are searchable; the PNG path travels with the chunk for the web app.

---

### `providers.py` — Groq and Gemini backends

**Purpose:** one interface (`complete`, `describe_image`) over two free-tier
APIs, so every script takes `--provider`.

**How it works:** OpenAI-style messages in; Gemini's system instruction and
`Content`/`Part` objects built on the way out. Both backends pace their
requests (Gemini's free tier allows ~10/minute), sleep through short 429s
using the server's retry delay, switch to the fallback model when a daily
limit is hit, and raise `RateLimitTooLong` rather than sleeping for half an
hour.

---

### `langchain_adapter.py` — optional LangChain interface

**Purpose:** let a LangChain application use this retriever, without the
pipeline depending on LangChain.

**How it works:** `as_langchain_retriever()` wraps `HybridRetriever` as a
`BaseRetriever`, turning hits into `Document`s whose metadata carries
everything a citation needs plus any rebuilt table grid. `build_chain()`
composes a small RAG chain from `langchain_core` primitives and this project's
own grounded prompt and LLM client. LangChain is imported inside the
functions, so the module imports fine without it installed.

---

### `table_notes.py` — table descriptions and rebuilt tables

**Purpose:** make results tables findable and readable. A flattened table is
mostly numbers, so it ranks badly, and the model misreads which number is in
which column.

**How it works:** finds chunks where some 20-token run is mostly numbers; for
each (plus the neighbouring chunk if the table crosses the boundary) asks the
LLM for a caption, a description without numbers, and the table as a Markdown
grid, in a line-based `TABLE … END` format (not JSON: a grid is multi-line, and
models put raw newlines inside JSON strings, which makes the JSON invalid — §8).
A JSON reply is still accepted if one arrives. The grid is kept only if every
row has the header's number of cells and every number in it occurs in the text
(at most as often). Notes are cached in `data/table_notes.json`, pinned to
hashes of the chunk text and the prompt, and saved after every chunk. A reply
that fits no format is stored with its first 300 characters, so a run that
failed the same way 57 times can be diagnosed rather than guessed at
(`--stats`, `--retry-failed`).

**Where they are used:** `build_index.py` copies them onto the chunks; the
description becomes part of the indexed text (dense, BM25 and reranker), the
grid is shown to the answering model after its source. The chunk text itself
is unchanged, so evaluation labels stay valid.

---

### `build_index.py` — embedding and dense retrieval

**Purpose:** make the chunked corpus searchable by meaning.

**How it works:**
1. **`build`:** loads all chunks, embeds each with `bge-base-en-v1.5` (on MPS
   if available), L2-normalizes every vector.
2. Builds a FAISS `IndexFlatIP` — inner product on unit-length vectors is
   mathematically equivalent to cosine similarity, giving proper similarity
   search without a more complex index type.
3. Saves three artifacts: the index, metadata aligned position-for-position
   with the vectors (so "vector #37" always maps to the right paper/page/text),
   and a config recording the embedding model used.
4. **`Retriever` class:** loads those artifacts and exposes `.search(query, k)`,
   prepending bge's query-side instruction before embedding. bge trains queries
   and documents asymmetrically, so only queries get this prefix.
5. Reference chunks are excluded by default, over-fetching a wider candidate
   pool internally so the final top-k is still full after filtering.

**Why it's built this way:** normalized embeddings + inner product is the
standard efficient route to cosine similarity in FAISS; persisting the build
config prevents a subtle bug class where a query is later embedded with an
incompatible model.

---

### `retrieve.py` — multi-strategy retrieval *(Iteration 2)*

**Purpose:** improve on dense-only retrieval, and make every retrieval choice a
measurable runtime parameter rather than a hardcoded decision.

**How it works:**
1. **Loads both indexes.** Reuses the persisted FAISS index for dense search,
   and builds a BM25 index in memory over the same chunks in the same order, so
   chunk indices are directly comparable between the two.
2. **Dense ranking:** embeds the query (with bge's prefix) and searches FAISS.
3. **Sparse ranking:** tokenizes the query with a deliberately simple
   lowercase-alphanumeric tokenizer and scores with BM25. Simplicity is the
   point — BM25's value here is exact term matching (`fpr95`, `cifar`, `10`),
   so aggressive normalization would destroy the signal we want.
4. **Fusion:** combines the two rankings by Reciprocal Rank Fusion (default) or
   min-max-normalized weighted blending.
5. **Reranking (optional second stage):** rescores each (query, chunk) pair
   jointly with a cross-encoder, far more accurate than comparing pre-computed
   vectors. Lazy-loaded so the lighter modes stay fast.
6. **Diversity + relevance control:** caps how many chunks any one paper can
   contribute, but only promotes a chunk for diversity if it clears a relative
   relevance floor. Results are re-sorted by score before returning, so
   enforcing the cap can never produce an out-of-order ranking, and the reserve
   is backfilled in score order if the cap would otherwise return fewer than k.
7. **`--compare` mode:** runs all four strategies on one query side by side —
   the qualitative precursor to Iteration 3's quantitative evaluation.

**Why it's built this way:** every knob is a flag because the point of
Iteration 2 is to *set up* Iteration 3. The diversity/relevance interaction was
found empirically (§7, §8), not designed up front.

---

### `generate.py` — grounded answer generation

**Purpose:** turn "a question + retrieved chunks" into a trustworthy, cited
answer — and resist hallucination.

**How it works:**
1. Loads the Groq API key from `.env` and retrieves top-k chunks using
   `HybridRetriever`, with the retrieval strategy and all its options
   selectable from the command line. The reranker is loaded only when the
   chosen mode needs it.
2. Builds a prompt listing each chunk as a **numbered, labeled source**
   (e.g. `S1: 1806.01768 (2018), p.1`) followed by its text, then the question.
3. Sends it with a **system prompt enforcing strict grounding**: answer only
   from the numbered sources, cite every claim with its tag, and if the sources
   don't contain the answer, output an exact fixed refusal sentence rather than
   falling back on general knowledge. Temperature 0 for deterministic output.
4. Retries on Groq's `RateLimitError` (429), reading the server's `retry-after`
   header when present and honouring it plus a cushion, falling back to
   exponential backoff otherwise. This matters because Groq's free tier can
   briefly block all requests after a rate-limit hit.
5. Prints the retrieval config used, the answer, then every source handed to
   the model with its rerank score, the fusion score that got it into the
   candidate pool, and a `[ref]` marker if it was a bibliography chunk — so you
   can always see *why* the model said what it said and which retrieval stage
   is responsible for a given result.

**Why it's built this way:** strict grounding is what makes citations mean
something; proper rate-limit handling is a real production concern that
surfaces immediately at free-tier limits; exposing `--mode` here means the same
question can be answered from different retrieval strategies, which is how you
measure whether better retrieval yields better *answers* and not just better
rankings.

---

### `debug_refs.py` — bibliography detection diagnostic *(Iteration 2)*

**Purpose:** answer "why did reference classification do *that*?" with data
rather than speculation.

For each requested paper it prints the total extracted length, where the
classifier currently lands (with surrounding context), every occurrence of
"references"/"bibliography" with its absolute position and percentage through
the document, and the last 600 characters of extracted text.

That last item is what revealed the real bug: papers whose text *ends* with
results tables and formulas, not a bibliography, because the bibliography sits
mid-document with an appendix after it.

---

### `query_expansion.py` and `llm.py` — multi-query *(Iteration 3)*

**Purpose:** retrieve with several phrasings of the question, not just the
user's.

**How it works:**
1. `QueryExpander.expand(question)` prompts the LLM for N rephrasings that keep
   the meaning but vary terminology, then cleans the reply (strips numbering,
   bullets, quotes, duplicates and copies of the original).
2. Results are cached on disk keyed by model, N and question, and the original
   question is always returned first.
3. `HybridRetriever.search(..., query_variants=[...])` runs the first stage
   (dense, BM25 or hybrid) once per variant and fuses the lists with RRF;
   reranking and the diversity cap then run as normal, scored against the
   original question.
4. The LLM is passed in as a function, so the expander can be tested with a
   stub and has no Groq import of its own; `llm.py` supplies the real one.

---

### `metrics.py` — retrieval metrics *(Iteration 3)*

**Purpose:** the numbers, in one small file that can be checked by hand.

Recall@k, Precision@k (denominator k, so returning fewer results isn't
rewarded), NDCG@k (log2 discount, optional graded relevance), Success@k (hit
rate) and reciprocal rank. Every function returns `None` for unanswerable
queries rather than 0, so averaging skips them instead of dragging recall down.
`abstention_metrics` reports accuracy, false-answer rate (answered something it
shouldn't) and false-abstain rate (refused something it could answer).

---

### `build_queries.py` — candidate query generation *(Iteration 3)*

**Purpose:** a first draft of the ground truth, fast.

**How it works:**
1. Samples prose chunks round-robin across papers (skipping bibliography,
   table and equation chunks) so no paper dominates.
2. For each chunk, one LLM call returns a `semantic` and an `exact_term`
   question plus a verbatim evidence quote for each. Questions that lean on
   unseen context ("this paper", "the proposed method") are rejected.
3. For `multi_paper`, pairs each anchor chunk with its most lexically similar
   chunk from a *different* paper and asks for a question needing both.
4. For `unanswerable`, asks for plausible ML questions outside the corpus's
   topics, plus three fixed off-domain controls.
5. Stores each relevant chunk with a hash of its text.

---

### `review_queries.py` — human review *(Iteration 3)*

**Purpose:** turn candidates into ground truth someone has actually checked.

Shows each candidate with its evidence quote and relevant chunk(s); for
unanswerable ones, shows the three closest chunks so you can check the corpus
really doesn't answer it. `a` shows other close chunks and lets you mark any
that also answer the question, since an unlabelled relevant chunk counts as a
miss. Saves after every decision (atomically), so quitting loses nothing.
`add` handles hand-written questions; `export` validates every label's hash and
writes `queries.jsonl`.

---

### `run_eval.py` — the sweep *(Iteration 3)*

**Purpose:** compare retrieval configurations on the same queries, with enough
rigour that the comparison can be quoted.

**How it works:**
1. Loads the query set and refuses to run if any labelled chunk's text changed.
2. Expands queries once per multi-query size (cached), loads the retriever
   once, then evaluates each configuration at k = 1, 3, 5, 10.
3. Aggregates over answerable queries, overall and per category.
4. Computes paired-bootstrap 95% CIs for Recall@5 and MRR against dense.
5. Computes abstention AUROC from each configuration's top-1 score.
6. Writes CSV/JSON (see §9) and prints the comparison, per-category,
   significance and abstention tables.

---

### `chat.py` and `uploads.py` — conversation over the corpus or your PDFs

**Purpose:** ask follow-up questions, about the corpus or a document you bring.

**How it works:**
1. `ChatSession.ask()` rewrites a follow-up into a standalone question from
   the last few turns, optionally expands it, retrieves, and answers with the
   recent turns included (their citations removed).
2. The LLM is injected as a function, so the whole conversation flow is tested
   with scripted replies.
3. `uploads.load_uploads()` validates each PDF (type, size, pages, text
   layer), then chunks it with the corpus pipeline and gives it a
   `upload:<file name>` identity.
4. `HybridRetriever.from_chunks()` builds an in-memory index over those
   chunks, alone or merged with the persisted corpus (`--with-corpus`).
5. The command-line loop adds `/sources`, `/history` and `/reset`. The
   Iteration 5 web app will wrap the same `ChatSession`.

---

## 12. Roadmap

- **Iteration 3 (built; results pending):** generate and review the query set,
  run the `core` and `ablations` suites, record the numbers here. Optional
  follow-up: generation-level evaluation (does the LLM actually refuse the
  unanswerable questions; are answers faithful to their sources), which costs
  one LLM call per query per configuration.
- **Iteration 4:** Uncertainty / OOD module — softmax, MC Dropout, Deep
  Ensembles, EDL on CIFAR-10 vs SVHN / CIFAR-100, reporting AUROC, AUPR,
  FPR@95TPR, ECE and Brier score. Training on Kaggle's free GPU tier;
  evaluation runs locally from saved checkpoints. Training and evaluation will
  be separate entry points with the backbone and OOD dataset list as config, so
  new architectures or OOD sets can be benchmarked later without retraining.
- **Iteration 5:** Deployment — FastAPI backend, Streamlit frontend,
  Docker, possibly Qdrant in place of FAISS. The chat UI wraps `ChatSession`
  (one per browser session) with PDF upload via `uploads.py` and a
  "search the corpus too" toggle; uncertainty results are served as a
  pre-computed benchmark dashboard. Answers can show the figure PNGs from
  `data/figures/`, which is why `figures.py` saves them with the chunk.
