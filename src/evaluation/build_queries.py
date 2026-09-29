"""
Generate a CANDIDATE evaluation query set from the chunked corpus.

Every candidate is a question plus the chunk_id(s) that answer it, so
retrieval metrics (Recall@K, MRR, NDCG) can be computed against it. The output
is a pool of candidates for a human to review with review_queries.py - it is
not ground truth until reviewed.

Four categories, because earlier manual testing showed retrieval strategies
differ BY QUERY TYPE (BM25 helped exact-term queries; diversity helped broad
ones), and a single averaged number would hide that:

  semantic     paraphrased; deliberately avoids the passage's distinctive
               wording. Guards against a known bias: LLM-generated questions
               tend to reuse the passage's vocabulary, which flatters any
               retriever that matches words.
  exact_term   contains a specific term, dataset, metric or number that
               appears verbatim in the passage.
  multi_paper  needs two passages from DIFFERENT papers; both are relevant.
  unanswerable plausible questions the corpus does not answer; no relevant
               chunks. Scored by abstention, not recall.

Each relevant chunk is stored with a hash of its text. The sweep refuses to
run if chunks.jsonl has changed since labelling (e.g. re-chunked with a
different size), because chunk_ids would then point at different text and
every metric would be silently wrong.

Usage (from the repo root):
  python src/evaluation/build_queries.py
  python src/evaluation/build_queries.py --n-chunks 30 --n-multi 15 --n-unanswerable 12
  python src/evaluation/build_queries.py --model openai/gpt-oss-20b   # cheaper
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))
from llm import (  # noqa: E402
    add_provider_args, exit_on_rate_limit, make_complete_fn, resolve,
)
from rank_bm25 import BM25Okapi  # noqa: E402

SINGLE_PROMPT = """You write evaluation questions for a retrieval system over machine learning papers.

Paper: {title}
Passage:
\"\"\"{text}\"\"\"

Write two questions that THIS passage answers well:
1. "semantic": ask about the idea or mechanism using DIFFERENT wording from the passage. Avoid copying its distinctive phrases or rare terms.
2. "exact_term": include at least one specific term, dataset name, metric name or number that appears verbatim in the passage.

Rules:
- Each question must stand alone for someone who has NOT seen the passage. Never write "this paper", "the passage", "the authors" or "the proposed method"; name the method or topic instead.
- Each must be specific enough that this passage is a correct answer to retrieve.
- For each, give "evidence": a verbatim quote from the passage (at most 30 words) that answers it.
- If the passage is only a fragment (a bare table, equations with no context, a bibliography), return {{"skip": true}}.

Reply with JSON only:
{{"semantic": "...", "semantic_evidence": "...", "exact_term": "...", "exact_term_evidence": "..."}}"""

MULTI_PROMPT = """You write evaluation questions for a retrieval system over machine learning papers.

Passage A (from "{title_a}"):
\"\"\"{text_a}\"\"\"

Passage B (from "{title_b}"):
\"\"\"{text_b}\"\"\"

Write ONE question whose complete answer needs information from BOTH passages, for example a comparison or a relationship between the two approaches.

Rules:
- It must stand alone for someone who has not seen the passages. Name methods or topics; never write "passage A", "this paper" or "the authors".
- If the passages share no meaningful connection, return {{"skip": true}}.

Reply with JSON only:
{{"question": "...", "evidence_a": "<verbatim quote from A, at most 30 words>", "evidence_b": "<verbatim quote from B, at most 30 words>"}}"""

UNANSWERABLE_PROMPT = """A retrieval system answers questions using ONLY these machine learning papers:
{titles}

Write {n} realistic questions a machine learning researcher might ask this system that these papers do NOT answer. Use other ML topics (for example reinforcement learning algorithms, transformer architecture details, diffusion models, optimisers, specific benchmark numbers these papers would not report). Avoid anything about uncertainty estimation, out-of-distribution detection, Dirichlet or evidential models, ensembles, or knowledge distillation, since the papers may cover those.

Reply with JSON only: {{"questions": ["...", "..."]}}"""

SYSTEM = "You create evaluation data for information retrieval. Reply with JSON only."

# Clearly off-domain controls, always included.
OFF_DOMAIN = [
    "What is the boiling point of water at sea level?",
    "Who won the FIFA World Cup in 2018?",
    "What is the capital city of Australia?",
]


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def extract_json(raw: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (tolerates code fences)."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def looks_standalone(q: str) -> bool:
    """Reject questions that lean on context the reader won't have."""
    bad = ("this paper", "the passage", "passage a", "passage b", "the authors",
           "the proposed", "this work", "this study", "the above")
    return len(q.split()) >= 5 and not any(b in q.lower() for b in bad)


def is_prose(text: str) -> bool:
    """Skip chunks that are mostly numbers (tables) or symbols (equations)."""
    tokens = text.split()
    if len(tokens) < 80:
        return False
    alpha = sum(1 for t in tokens if re.fullmatch(r"[A-Za-z][A-Za-z\-]{2,}", t))
    return alpha / len(tokens) > 0.55


def sample_chunks(chunks: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Round-robin across papers so no single paper dominates the query set."""
    by_paper = defaultdict(list)
    for c in chunks:
        if not c["is_reference"] and is_prose(c["text"]):
            by_paper[c["arxiv_id"]].append(c)
    for lst in by_paper.values():
        rng.shuffle(lst)
    papers = sorted(by_paper)
    rng.shuffle(papers)
    out = []
    while len(out) < n and any(by_paper.values()):
        for p in papers:
            if by_paper[p] and len(out) < n:
                out.append(by_paper[p].pop())
    return out


def rel(chunk: dict) -> dict:
    return {"chunk_id": chunk["chunk_id"], "text_hash": text_hash(chunk["text"])}


@exit_on_rate_limit
def main():
    ap = argparse.ArgumentParser(description="Generate candidate eval queries.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/candidates.jsonl"))
    add_provider_args(ap)
    ap.add_argument("--n-chunks", type=int, default=28,
                    help="Chunks sampled; each yields a semantic + exact_term question.")
    ap.add_argument("--n-multi", type=int, default=14)
    ap.add_argument("--n-unanswerable", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.out.exists():
        sys.exit(f"{args.out} already exists. Delete it first to regenerate "
                 f"(review decisions in it would be lost).")

    chunks = [json.loads(line) for line in
              (args.data_dir / "chunks.jsonl").read_text().splitlines() if line.strip()]
    by_id = {c["chunk_id"]: c for c in chunks}
    rng = random.Random(args.seed)
    provider, model, _ = resolve(args)
    complete = make_complete_fn(model, provider=provider)
    candidates: list[dict] = []

    def add(category, question, relevant, evidence, source):
        if not looks_standalone(question):
            print(f"    - rejected (not standalone): {question[:70]}")
            return
        candidates.append({
            "qid": f"q{len(candidates) + 1:03d}",
            "category": category,
            "question": question.strip(),
            "relevant": relevant,
            "evidence": evidence,
            "source": source,
            "status": "pending",
        })

    # 1. semantic + exact_term from single chunks
    picked = sample_chunks(chunks, args.n_chunks, rng)
    print(f"Generating semantic + exact_term questions from {len(picked)} chunks ...")
    for i, c in enumerate(picked, 1):
        print(f"  [{i}/{len(picked)}] {c['chunk_id']}")
        data = extract_json(complete(SYSTEM, SINGLE_PROMPT.format(
            title=c["title"], text=" ".join(c["text"].split()))))
        if not data or data.get("skip"):
            print("    - skipped")
            continue
        for cat in ("semantic", "exact_term"):
            if data.get(cat):
                add(cat, data[cat], [rel(c)], [data.get(f"{cat}_evidence", "")],
                    "llm")

    # 2. multi_paper from lexically related chunk pairs in different papers
    print(f"\nGenerating {args.n_multi} multi_paper questions ...")
    pool = [c for c in chunks if not c["is_reference"] and is_prose(c["text"])]
    bm25 = BM25Okapi([re.findall(r"[a-z0-9]+", c["text"].lower()) for c in pool])
    anchors = sample_chunks(chunks, args.n_multi * 2, rng)
    made = 0
    for a in anchors:
        if made >= args.n_multi:
            break
        scores = bm25.get_scores(re.findall(r"[a-z0-9]+", a["text"].lower()))
        best = next((pool[j] for j in scores.argsort()[::-1]
                     if pool[j]["arxiv_id"] != a["arxiv_id"]), None)
        if best is None:
            continue
        print(f"  pair {a['chunk_id']} + {best['chunk_id']}")
        data = extract_json(complete(SYSTEM, MULTI_PROMPT.format(
            title_a=a["title"], text_a=" ".join(a["text"].split()),
            title_b=best["title"], text_b=" ".join(best["text"].split()))))
        if not data or data.get("skip") or not data.get("question"):
            print("    - skipped")
            continue
        before = len(candidates)
        add("multi_paper", data["question"], [rel(a), rel(best)],
            [data.get("evidence_a", ""), data.get("evidence_b", "")], "llm")
        made += len(candidates) - before

    # 3. unanswerable: LLM-proposed in-domain-adjacent + fixed off-domain controls
    print(f"\nGenerating {args.n_unanswerable} unanswerable questions ...")
    titles = "\n".join(f"- {t}" for t in sorted({c["title"] for c in chunks}))
    data = extract_json(complete(SYSTEM, UNANSWERABLE_PROMPT.format(
        titles=titles, n=args.n_unanswerable))) or {}
    for q in data.get("questions", [])[: args.n_unanswerable]:
        add("unanswerable", q, [], [], "llm")
    for q in OFF_DOMAIN:
        add("unanswerable", q, [], [], "fixed")

    # sanity: every relevant chunk_id must exist in the corpus
    for cand in candidates:
        for r in cand["relevant"]:
            assert r["chunk_id"] in by_id, r

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for cand in candidates:
            f.write(json.dumps(cand, ensure_ascii=False) + "\n")

    counts = defaultdict(int)
    for cand in candidates:
        counts[cand["category"]] += 1
    print("\n" + "=" * 50)
    for cat in ("semantic", "exact_term", "multi_paper", "unanswerable"):
        print(f"  {cat:13s} {counts[cat]}")
    print(f"  {'total':13s} {len(candidates)}")
    print(f"Written to {args.out}")
    print("Next: python src/evaluation/review_queries.py review")


if __name__ == "__main__":
    main()
