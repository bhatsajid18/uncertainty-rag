# Uncertainty-Aware Research Intelligence Platform

Retrieval-augmented Q&A over ML research papers, combined with an uncertainty
and OOD evaluation framework.

**Status:** Iterations 1–2 complete.

- **Iteration 1** — end-to-end RAG pipeline (download → chunk → embed/index →
  grounded generation) running locally on Apple Silicon.
- **Iteration 2** — hybrid retrieval (dense + BM25), reciprocal-rank fusion,
  cross-encoder reranking, source-diversity control, density-based
  bibliography classification, and retrieval-strategy selection wired through
  to answer generation.

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
9. [Ablation surface](#9-ablation-surface)
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
│   │   └── debug_refs.py        # diagnostic: inspect bibliography detection
│   ├── uncertainty/              # (Iteration 4+)
│   ├── evaluation/                # (Iteration 3+)
│   └── api/                       # (later, if deploying)
├── data/
│   ├── papers.txt                 # your arXiv ID list (tracked in git)
│   ├── papers/                    # downloaded PDFs (gitignored)
│   ├── metadata.json              # paper metadata (gitignored)
│   ├── chunks.jsonl               # chunked text (gitignored)
│   ├── index.faiss                # FAISS vector index (gitignored)
│   ├── chunk_meta.json            # vector-position -> chunk metadata (gitignored)
│   └── index_config.json          # records embedding model used (gitignored)
├── configs/                       # Hydra configs (later)
├── experiments/                   # experiment scripts (later)
├── notebooks/                     # exploratory notebooks
├── tests/                         # pytest tests (later)
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
per organization (not per key). `generate.py` handles 429s with automatic
backoff that honours the server's `retry-after` header.

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

### 5.4 `retrieve.py` — multi-strategy retrieval _(Iteration 2)_

The main retrieval interface. Four selectable strategies over the same corpus:

| Mode            | What it does                                          |
| --------------- | ----------------------------------------------------- |
| `dense`         | FAISS vector search only (semantic similarity)        |
| `bm25`          | BM25 sparse keyword search only (exact term matching) |
| `hybrid`        | Reciprocal Rank Fusion over dense + BM25 rankings     |
| `hybrid_rerank` | hybrid, then cross-encoder reranking (**default**)    |

### 5.5 `generate.py` — grounded, cited answers

Retrieves top-k chunks via `HybridRetriever`, builds a prompt with labeled
sources (`[S1]`, `[S2]`, ...), and calls a Groq-hosted LLM with a system prompt
enforcing **strict grounding**: answer only from the provided sources, cite
every claim, and explicitly refuse if the sources are insufficient rather than
falling back on general knowledge.

All retrieval flags from §5.4 are available here too, so the same question can
be answered from different retrieval strategies — which separates two distinct
questions: does better retrieval _rank_ better, and does it produce better
_answers_?

### 5.6 `debug_refs.py` — bibliography detection diagnostic _(Iteration 2)_

For any paper, prints where the classifier lands, every occurrence of
"references"/"bibliography" with its position as a percentage through the
document, and the tail of the extracted text. Written while debugging
reference classification; kept because it's the right tool whenever tagging
looks wrong.

---

## 6. Command reference

### 6.1 Every session starts here

```bash
cd ~/Downloads/PROJECTS/uncertainty-rag        # adjust to your actual path
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

**Include bibliography chunks in results:**

```bash
python src/rag/retrieve.py "your query" --include-refs
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

**Verify strict grounding still holds** (should return the fixed refusal):

```bash
python src/rag/generate.py "What is the boiling point of water?"
```

### 6.5 Diagnostics

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

### 6.6 Git workflow

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

### 6.7 Full pipeline, start to finish

```bash
conda activate rag
cd ~/PROJECTS/uncertainty-rag

python src/rag/download_papers.py --id-file data/papers.txt
python src/rag/chunk_papers.py
python src/rag/build_index.py build
python src/rag/retrieve.py "your query" --compare
python src/rag/generate.py "your question here"
```

Steps 1–3 only need re-running when papers or chunking parameters change.
Steps 4–5 are per-query.

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
  returned the right _topical neighborhood_ but not the most term-precise
  chunks, and scored notably lower (0.66–0.70) than on semantic queries (0.85).
  Numeric result tables carry weak semantic signal. BM25 rewards exact term
  matches — the missing capability.
- **Why RRF for fusion.** Dense cosine scores (~0–1) and BM25 scores
  (unbounded) live on different scales, so naively summing them is meaningless.
  Reciprocal Rank Fusion combines by _rank position_, sidestepping
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
  contribution (default 2), diversifying by _source_ rather than embedding
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

## 8. Debugging log

Each entry is a concrete example of diagnosing rather than guessing.

### Bibliography detection: three attempts

| Approach                       | Papers detected | Ref chunks          | Problem                     |
| ------------------------------ | --------------- | ------------------- | --------------------------- |
| Heading must start a page      | 3 / 12          | 25                  | Missed most papers entirely |
| Full-text heading scan         | 12 / 12         | 160 (40% of corpus) | Swept in appendices         |
| **Per-chunk citation density** | **12 / 12**     | **70 (17%)**        | —                           |

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

The transferable lesson: _stop patching a heuristic once the failure turns out
to be structural rather than parametric._

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

## 9. Ablation surface

Every one of these is a runtime flag, not a code change — so the Iteration 3
evaluation harness can sweep them without touching pipeline code:

| Dimension              | Flag                        | Values                             |
| ---------------------- | --------------------------- | ---------------------------------- |
| Retrieval strategy     | `--mode`                    | dense, bm25, hybrid, hybrid_rerank |
| Fusion method          | `--fusion`                  | rrf, weighted                      |
| Dense/sparse balance   | `--alpha`                   | 0.0–1.0 (weighted fusion only)     |
| First-stage pool size  | `--candidate-n`             | any integer                        |
| Source diversity       | `--max-per-paper`           | 0 (off), 1, 2, 3, ...              |
| Relevance floor        | `--min-score-frac`          | 0.0 (off) – 1.0                    |
| Bibliography inclusion | `--include-refs`            | on / off                           |
| Bibliography threshold | `--ref-threshold`           | citation density cutoff            |
| Reranker model         | `--reranker`                | any HF cross-encoder               |
| Chunk size / overlap   | `--chunk-size`, `--overlap` | any integers                       |
| Generation model       | `--model`                   | any available Groq model           |
| Sources per answer     | `-k`                        | any integer                        |

**Hypotheses worth testing in Iteration 3:**

1. Source diversity helps broad questions ("what methods exist for uncertainty
   estimation?") and hurts narrow factual ones. Early evidence supports this:
   the broad query returned 5 strong chunks across 4 papers (all 0.98+), while
   the narrow query's diverse alternatives scored an order of magnitude below
   the top hit.
2. Reranker scores are well-calibrated enough to serve as an abstention signal
   (0.000–0.001 on out-of-domain queries vs 0.5–0.99 on in-domain).
3. Retrieval improvements and answer improvements are only partly correlated —
   a query can have better retrieval yet the same (correct) refusal, if the
   corpus lacks the information.

---

## 10. Tech stack

| Layer                   | Choice                                              | Why                                                                         |
| ----------------------- | --------------------------------------------------- | --------------------------------------------------------------------------- |
| Language                | Python 3.11                                         | Matches library compatibility across the stack                              |
| Environment manager     | conda (Miniconda)                                   | Isolates dependencies; standard for ML work                                 |
| PDF source              | arXiv API via the `arxiv` library                   | Free, reliable, structured metadata alongside the PDF                       |
| PDF parsing             | PyMuPDF (`pymupdf`)                                 | Fast, page-accurate extraction; preserves page numbers for citations        |
| Text normalization      | Python `unicodedata` (NFKC)                         | Fixes typographic ligatures LaTeX PDFs embed                                |
| Tokenization (chunking) | HF `transformers` `AutoTokenizer` (bge's own)       | Chunk sizes measured in the _actual_ tokens the embedder sees               |
| Embedding model         | `BAAI/bge-base-en-v1.5` via `sentence-transformers` | Free, local, strong on MTEB for its size; runs on Apple Silicon MPS         |
| Dense vector search     | FAISS (`faiss-cpu`), `IndexFlatIP`                  | Exact inner-product over normalized embeddings (= cosine); no server needed |
| Sparse retrieval        | `rank-bm25` (`BM25Okapi`)                           | Pure Python, transparent, built in memory at load (<1s at this scale)       |
| Fusion                  | Reciprocal Rank Fusion (own implementation)         | Rank-based, so no score normalization across incompatible scales            |
| Reranking               | `BAAI/bge-reranker-base` via `CrossEncoder`         | Joint (query, chunk) scoring; 512-token window matches our chunks           |
| LLM (generation)        | Groq API, `openai/gpt-oss-120b`                     | Free tier, no credit card, very fast inference                              |
| LLM client              | `groq` Python SDK                                   | Typed exceptions (`RateLimitError`) for clean retry logic                   |
| Secrets management      | `python-dotenv` + `.env` (gitignored)               | Keeps the API key out of source control                                     |
| Config / control flow   | plain `argparse` per script                         | Each module independently runnable and scriptable                           |
| Version control         | Git + GitHub                                        | `.gitignore` keeps PDFs, embeddings, indexes out of the repo                |

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
4. Writes `data/metadata.json` — the _source of truth_ for every citation the
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
   spans.
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

### `retrieve.py` — multi-strategy retrieval _(Iteration 2)_

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
Iteration 2 is to _set up_ Iteration 3. The diversity/relevance interaction was
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
   can always see _why_ the model said what it said and which retrieval stage
   is responsible for a given result.

**Why it's built this way:** strict grounding is what makes citations mean
something; proper rate-limit handling is a real production concern that
surfaces immediately at free-tier limits; exposing `--mode` here means the same
question can be answered from different retrieval strategies, which is how you
measure whether better retrieval yields better _answers_ and not just better
rankings.

---

### `debug_refs.py` — bibliography detection diagnostic _(Iteration 2)_

**Purpose:** answer "why did reference classification do _that_?" with data
rather than speculation.

For each requested paper it prints the total extracted length, where the
classifier currently lands (with surrounding context), every occurrence of
"references"/"bibliography" with its absolute position and percentage through
the document, and the last 600 characters of extracted text.

That last item is what revealed the real bug: papers whose text _ends_ with
results tables and formulas, not a bibliography, because the bibliography sits
mid-document with an appendix after it.

---

## 12. Roadmap

- **Iteration 3 (next):** Evaluation harness — a labelled query set, then
  Recall@K, MRR and NDCG for retrieval, and faithfulness / answer relevance for
  generation, swept across the ablation surface in §9. Must measure retrieval
  quality and answer quality independently (see §7).
- **Iteration 4:** Uncertainty / OOD module — softmax, MC Dropout, Deep
  Ensembles, EDL on CIFAR-10 vs SVHN / CIFAR-100, reporting AUROC, AUPR,
  FPR@95TPR, ECE and Brier score. Training on Kaggle's free GPU tier;
  evaluation runs locally from saved checkpoints. Training and evaluation will
  be separate entry points with the backbone and OOD dataset list as config, so
  new architectures or OOD sets can be benchmarked later without retraining.
- **Iteration 5 (optional):** Deployment — FastAPI backend, Streamlit frontend,
  Docker, possibly Qdrant in place of FAISS. Retrieval and generation run live;
  uncertainty results are served as a pre-computed benchmark dashboard.
