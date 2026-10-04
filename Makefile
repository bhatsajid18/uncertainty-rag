# Common tasks. Run from the repo root with the `rag` environment active.
#   make help
PY ?= python
.DEFAULT_GOAL := help
.PHONY: help install test lint check corpus index notes chat \
        eval-core eval-ablations eval-gen api ui dev docker-up docker-down \
        uncertainty-smoke uncertainty-train uncertainty-predict uncertainty-eval clean

help:  ## List the commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install everything, including test tools
	$(PY) -m pip install -r requirements-dev.txt

test:  ## Run the test suite
	$(PY) -m pytest -q

lint:  ## Lint with ruff
	ruff check src tests

check: lint test  ## Lint + tests (what CI runs)

# --- RAG corpus (Iterations 1-2) ---------------------------------------------------------

corpus:  ## Download papers, chunk, extract tables + figures, build the index
	$(PY) src/rag/download_papers.py --id-file data/papers.txt
	$(PY) src/rag/chunk_papers.py
	$(PY) src/rag/table_extract.py
	$(PY) src/rag/figures.py
	$(PY) src/rag/build_index.py build

index:  ## Rebuild only the search index (after table/figure notes)
	$(PY) src/rag/build_index.py build

notes:  ## LLM table descriptions (uses quota; resumable)
	$(PY) src/rag/table_notes.py

chat:  ## Chat with the corpus in the terminal
	$(PY) src/rag/chat.py

# --- Evaluation (Iteration 3) ------------------------------------------------------------

eval-core:  ## Retrieval metrics for the core configurations
	$(PY) src/evaluation/run_eval.py --suite core

eval-ablations:  ## Retrieval metrics for every ablation
	$(PY) src/evaluation/run_eval.py --suite ablations

eval-gen:  ## Answer quality: faithfulness, citations, relevance (uses quota)
	$(PY) src/evaluation/gen_eval.py --configs dense hybrid_rerank --judge-provider gemini

# --- Uncertainty benchmark (Iteration 4) ------------------------------------------------

uncertainty-smoke:  ## Whole pipeline on tiny fake data (1 min, CPU, no downloads)
	$(PY) src/uncertainty/train.py --method all --fake --epochs 2
	$(PY) src/uncertainty/predict.py --fake
	$(PY) src/uncertainty/evaluate.py --fake

uncertainty-train:  ## Train all 7 models (GPU; see notebooks/kaggle_uncertainty.ipynb)
	$(PY) src/uncertainty/train.py --method all

uncertainty-predict:  ## Run the trained models on every evaluation set
	$(PY) src/uncertainty/predict.py

uncertainty-eval:  ## Tables + figures from saved predictions (CPU)
	$(PY) src/uncertainty/evaluate.py

# --- Web app (Iteration 5) ----------------------------------------------------------------

api:  ## Run the API on :8000 (docs at /docs)
	uvicorn api.main:app --app-dir src --port 8000 --reload

ui:  ## Run the Streamlit UI on :8501 (needs the API)
	streamlit run src/ui/app.py

dev:  ## API and UI together (Ctrl-C stops both)
	@trap 'kill 0' INT TERM; \
	  uvicorn api.main:app --app-dir src --port 8000 & \
	  streamlit run src/ui/app.py & \
	  wait

docker-up:  ## Build and start API + UI in Docker
	docker compose up --build

docker-down:  ## Stop the containers
	docker compose down

clean:  ## Remove caches and the fake-data smoke outputs
	rm -rf .pytest_cache .ruff_cache checkpoints/fake results/uncertainty/fake
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
