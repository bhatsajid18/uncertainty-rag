"""
Generation evaluation: are the ANSWERS any good, not just the retrieved chunks?

run_eval.py measures retrieval (did the right chunk come back?). This measures
what a user actually sees, for the same labelled questions:

  faithfulness       share of the answer's sentences that some retrieved source
                     supports. A sentence the sources don't back is a
                     hallucination, whether or not it happens to be true.
  citation accuracy  of the sentences that cite a source, the share whose cited
                     source is one that supports it. Catches "right fact, wrong
                     [S2]", which faithfulness alone misses.
  citation coverage  share of sentences carrying at least one citation.
  relevance          1-5 judge rating of how well the answer addresses the
                     question, reported as 0-1.
  correctness        answer consistent with the reference quote the question
                     was written from (answerable questions that have one).
  false refusals     answerable questions the model refused.
  abstention         unanswerable questions the model correctly refused. This
                     is the generation-side counterpart of run_eval's
                     retrieval abstention AUROC.

How: for each question and retrieval configuration, retrieve, generate the
answer with the production prompt, then ask a judge model to check each
sentence of the answer against the sources. The answer is split into
sentences HERE, deterministically, and the judge only says which numbered
sources support each numbered sentence - it never decides what the claims are,
which keeps runs comparable.

Two different models for generating and judging (e.g. --provider groq,
--judge-provider gemini) avoid a model grading its own answers. Every
generation and judgement is cached in the run directory, keyed by the config,
the question, both models and both prompts, so a run stopped by a rate limit
resumes where it stopped and a re-run costs nothing.

Outputs, in results/gen_eval/<run-name>/:
  cache.jsonl        one line per (config, question): answer, judgement, metrics
  summary.csv/json   config x metric
  run_info.json      configs, models, prompt hashes, timestamp

Usage (from the repo root):
  python src/evaluation/gen_eval.py --limit 5                     # smoke test
  python src/evaluation/gen_eval.py --configs dense hybrid_rerank \\
      --provider groq --judge-provider gemini
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rag"))
sys.path.insert(0, str(ROOT / "evaluation"))
from generate import (  # noqa: E402
    REFUSAL, SYSTEM_PROMPT, build_user_prompt, normalize_citations,
)
from llm import (  # noqa: E402
    PROVIDERS, ProviderUnavailable, RateLimitTooLong, add_provider_args,
    default_model, exit_on_rate_limit, make_complete_fn, resolve,
)

DEFAULT_CONFIGS = ("dense", "hybrid_rerank")
MAX_SENTENCES = 15

JUDGE_SYSTEM = (
    "You check answers written by a research assistant against the numbered "
    "sources it was given. You are not asked whether the answer is true in "
    "general, only whether the sources support it.\n"
    "1. For each numbered sentence of the answer, list the numbers of the "
    "sources that support it. A source supports a sentence only if everything "
    "the sentence states - including every number, and which method or dataset "
    "a number belongs to - can be read from that source. Write 'none' if no "
    "source supports it.\n"
    "2. Rate how well the answer addresses the question: RELEVANCE 5 = fully "
    "answers it, 3 = partly, 1 = does not address it.\n"
    "3. If a reference quote is given, say whether the answer is consistent with "
    "it: CORRECT yes, partly or no. If no reference is given, write CORRECT: n/a.\n"
    "Output exactly this and nothing else:\n"
    "SENTENCE 1: 2, 3\n"
    "SENTENCE 2: none\n"
    "RELEVANCE: 4\n"
    "CORRECT: yes"
)
PROMPT_VERSION = hashlib.sha1((SYSTEM_PROMPT + JUDGE_SYSTEM).encode()).hexdigest()[:8]

_CITE = re.compile(r"\[S(\d+)\]")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[*])")


# --- splitting and parsing (pure, unit tested) ---------------------------------------


def is_refusal(answer: str) -> bool:
    return REFUSAL.rstrip(".").lower() in answer.lower()


def split_sentences(answer: str, limit: int = MAX_SENTENCES) -> list[str]:
    """The answer as a list of checkable sentences.

    Lines first (answers use bullets and small tables), then sentences within a
    line. Markdown table separators, headings and fragments without a word are
    dropped: there is nothing in them to verify.
    """
    out = []
    for line in answer.splitlines():
        line = _BULLET.sub("", line).strip()
        if not line or re.fullmatch(r"[|:\-\s]+", line) or line.startswith("#"):
            continue
        for sent in _SENT_END.split(line):
            sent = sent.strip().strip("*").strip()
            if len(re.findall(r"[A-Za-z]{2,}", sent)) >= 2:
                out.append(sent)
    return out[:limit]


def cited_sources(sentence: str) -> set[int]:
    return {int(n) for n in _CITE.findall(sentence)}


def parse_judgement(raw: str, n_sentences: int) -> dict:
    """Read the judge's reply. Missing sentences count as unsupported."""
    support: dict[int, set[int]] = {}
    relevance, correct = None, None
    for line in raw.splitlines():
        line = line.strip().strip("*").strip()
        m = re.match(r"SENTENCE\s+(\d+)\s*:\s*(.*)$", line, re.I)
        if m:
            idx = int(m.group(1))
            nums = {int(x) for x in re.findall(r"\d+", m.group(2))}
            if "none" in m.group(2).lower() and not nums:
                nums = set()
            support[idx] = nums
            continue
        m = re.match(r"RELEVANCE\s*:\s*(\d)", line, re.I)
        if m:
            relevance = min(5, max(1, int(m.group(1))))
            continue
        m = re.match(r"CORRECT\s*:\s*(yes|partly|partial|no|n/?a)", line, re.I)
        if m:
            word = m.group(1).lower()
            correct = {"yes": 1.0, "partly": 0.5, "partial": 0.5, "no": 0.0}.get(word)
    return {"support": [sorted(support.get(i, set())) for i in range(1, n_sentences + 1)],
            "parsed_sentences": len(support), "relevance": relevance,
            "correct": correct}


def answer_metrics(sentences: list[str], judgement: dict, n_sources: int) -> dict:
    """Per-answer scores from the sentences and the judge's verdicts."""
    n = len(sentences)
    if n == 0:
        return {"n_sentences": 0}
    support = judgement["support"]
    cited = [cited_sources(s) for s in sentences]
    supported = [bool(sup) for sup in support]
    with_cites = [i for i in range(n) if cited[i]]
    return {
        "n_sentences": n,
        "faithfulness": sum(supported) / n,
        "citation_coverage": len(with_cites) / n,
        "citation_accuracy": (sum(bool(cited[i] & set(support[i])) for i in with_cites)
                              / len(with_cites)) if with_cites else None,
        "invalid_citations": sum(1 for c in cited for s in c if s < 1 or s > n_sources),
        "relevance": (judgement["relevance"] - 1) / 4 if judgement["relevance"] else None,
        "correct": judgement["correct"],
    }


def aggregate(rows: list[dict]) -> dict:
    """Config-level summary. Answerable and unanswerable are scored differently."""
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]

    def avg(key, pool):
        vals = [r["metrics"].get(key) for r in pool if r["metrics"].get(key) is not None]
        return mean(vals) if vals else None

    return {
        "n_answerable": len(ans), "n_unanswerable": len(una),
        "faithfulness": avg("faithfulness", answered),
        "citation_accuracy": avg("citation_accuracy", answered),
        "citation_coverage": avg("citation_coverage", answered),
        "relevance": avg("relevance", answered),
        "correctness": avg("correct", answered),
        "false_refusal_rate": (sum(r["refused"] for r in ans) / len(ans)) if ans else None,
        "abstention_rate": (sum(r["refused"] for r in una) / len(una)) if una else None,
        # unanswerable questions the model answered anyway: how much of that
        # answer did the sources support? (low = invented)
        "unanswerable_faithfulness": avg(
            "faithfulness", [r for r in una if not r["refused"]]),
    }


# --- running ---------------------------------------------------------------------


def attach_evidence(queries: list[dict], candidates_path: Path) -> int:
    """Give each query the verbatim quote(s) it was written from.

    queries.jsonl is the reviewed, hash-pinned export and does not carry the
    quotes; candidates.jsonl (the generation step's output) does. Joining by qid
    keeps the reviewed file untouched. Returns how many queries got a quote.
    """
    if not candidates_path.exists():
        return 0
    quotes = {}
    for line in candidates_path.read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            ev = [e for e in (c.get("evidence") or []) if e and e.strip()]
            if ev:
                quotes[c["qid"]] = ev
    n = 0
    for q in queries:
        if not q.get("evidence") and q["qid"] in quotes:
            q["evidence"] = quotes[q["qid"]]
            n += 1
    return n


def cache_key(config: str, qid: str, gen_model: str, judge_model: str) -> str:
    return f"{config}|{qid}|{gen_model}|{judge_model}|{PROMPT_VERSION}"


def judge_prompt(question: str, hits: list[dict], sentences: list[str],
                 evidence: list[str] | None) -> str:
    sources = build_user_prompt(question, hits).rsplit("\nQuestion:", 1)[0]
    lines = [sources, f"\nQuestion: {question}\n", "Answer, split into sentences:"]
    lines += [f"SENTENCE {i}: {s}" for i, s in enumerate(sentences, 1)]
    quotes = [e for e in (evidence or []) if e and e.strip()]
    if quotes:
        lines.append("\nReference quote(s) the question was written from:")
        lines += [f"- {q}" for q in quotes]
    else:
        lines.append("\nNo reference quote is given.")
    return "\n".join(lines)


def evaluate_one(query: dict, config_name: str, search_kwargs: dict, retriever,
                 generate_fn: Callable[[str, str], str],
                 judge_fn: Callable[[str, str], str], k: int = 5,
                 expander=None) -> dict:
    """Retrieve, answer, judge. Returns the cache record for this question."""
    from retrieve import expand_split_tables

    answerable = bool(query.get("relevant"))
    variants = expander.expand(query["question"]) if expander else None
    hits = retriever.search(query["question"], k=k, query_variants=variants,
                            **search_kwargs)
    lookup = getattr(retriever, "chunk_by_id", None)
    if hits and lookup is not None:
        hits = expand_split_tables(hits, lookup)
    if hits:
        answer = normalize_citations(generate_fn(
            SYSTEM_PROMPT, build_user_prompt(query["question"], hits))).strip()
    else:
        answer = REFUSAL
    refused = is_refusal(answer)
    record = {"qid": query["qid"], "config": config_name,
              "category": query.get("category"), "answerable": answerable,
              "question": query["question"], "answer": answer, "refused": refused,
              "sources": [h["chunk_id"] for h in hits], "metrics": {}}
    if refused:
        return record
    sentences = split_sentences(answer)
    raw = judge_fn(JUDGE_SYSTEM, judge_prompt(query["question"], hits, sentences,
                                              query.get("evidence")))
    judgement = parse_judgement(raw, len(sentences))
    record.update(sentences=sentences, judgement=judgement,
                  metrics=answer_metrics(sentences, judgement, len(hits)))
    return record


def run(queries, configs: dict, retriever, generate_fn, judge_fn, cache_path: Path,
        gen_model: str, judge_model: str, log=print,
        expanders: dict | None = None) -> list[dict]:
    """Every (config, question) pair, reusing cached results; saves as it goes.

    A pair the provider cannot serve is skipped (and retried on the next run);
    three in a row stop the run, as in table_notes.py.
    """
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec
    out, in_a_row = [], 0
    todo = [(name, q) for name in configs for q in queries]
    for i, (name, q) in enumerate(todo, 1):
        key = cache_key(name, q["qid"], gen_model, judge_model)
        if key in cache:
            out.append(cache[key])
            continue
        log(f"  [{i}/{len(todo)}] {name:18s} {q['qid']}")
        try:
            rec = evaluate_one(q, name, configs[name], retriever, generate_fn, judge_fn,
                               expander=(expanders or {}).get(name))
        except ProviderUnavailable as e:
            in_a_row += 1
            log(f"    skipped: {e}")
            if in_a_row >= 3:
                log("  3 failures in a row; stopping. Run the same command later.")
                break
            continue
        in_a_row = 0
        rec["key"] = key
        cache[key] = rec
        out.append(rec)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def common_questions(records: list[dict], configs=None) -> set[str]:
    """The questions every configuration has a result for.

    A run cut short by the provider leaves some (config, question) pairs
    undone. Averaging each configuration over whatever it happened to finish
    would compare them on different questions, so the summary uses only the
    questions they all finished. A configuration in `configs` with no results
    at all leaves nothing to compare.
    """
    by_config: dict[str, set[str]] = {c: set() for c in (configs or [])}
    for r in records:
        by_config.setdefault(r["config"], set()).add(r["qid"])
    return set.intersection(*by_config.values()) if by_config else set()


def summarise(records: list[dict], out_dir: Path, configs=None) -> list[dict]:
    common = common_questions(records, configs)
    rows = []
    for name in dict.fromkeys([*(configs or []), *(r["config"] for r in records)]):
        rows.append({"config": name, **aggregate([r for r in records
                                                  if r["config"] == name
                                                  and r["qid"] in common])})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    fields = list(rows[0]) if rows else []
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    return rows


def print_summary(rows: list[dict]):
    cols = ["faithfulness", "citation_accuracy", "relevance", "correctness",
            "false_refusal_rate", "abstention_rate"]
    head = {"faithfulness": "faithful", "citation_accuracy": "cite-acc",
            "relevance": "relevant", "correctness": "correct",
            "false_refusal_rate": "false-ref", "abstention_rate": "abstain"}
    print(f"\n{'config':22s}" + "".join(f"{head[c]:>11s}" for c in cols))
    for r in rows:
        print(f"{r['config']:22s}" + "".join(
            f"{r[c]:11.3f}" if isinstance(r.get(c), float) else f"{'-':>11s}"
            for c in cols))
    print("\nfaithful/cite-acc/relevant/correct: answered answerable questions; "
          "false-ref: answerable refused (lower is better); abstain: unanswerable "
          "refused (higher is better).")


@exit_on_rate_limit
def main():
    from run_eval import CONFIGS

    ap = argparse.ArgumentParser(description="Evaluate generated answers.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--queries", type=Path, default=Path("data/eval/queries.jsonl"))
    ap.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                    help="Retrieval configs from run_eval.py --list.")
    ap.add_argument("--run-name", default="gen_v1")
    ap.add_argument("--out-root", type=Path, default=Path("results/gen_eval"))
    ap.add_argument("--limit", type=int, default=0, help="Only the first N queries.")
    add_provider_args(ap)
    ap.add_argument("--judge-provider", choices=PROVIDERS, default=None,
                    help="Provider for the judge (default: same as --provider). "
                         "A different one avoids a model grading itself.")
    ap.add_argument("--judge-model", default=None)
    args = ap.parse_args()

    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"Unknown config(s) {unknown}; see run_eval.py --list")
    if not args.queries.exists():
        sys.exit(f"{args.queries} not found. Review and export queries first "
                 "(review_queries.py review, then export).")
    queries = [json.loads(line) for line in args.queries.read_text().splitlines()
               if line.strip()]
    if args.limit:
        queries = queries[:args.limit]
    n_ev = attach_evidence(queries, args.queries.parent / "candidates.jsonl")
    print(f"{n_ev} of {len(queries)} queries have a reference quote for the "
          "correctness check")

    provider, gen_model, _ = resolve(args)
    judge_provider = args.judge_provider or provider
    judge_model = args.judge_model or (gen_model if judge_provider == provider
                                       else default_model(judge_provider))
    if judge_provider == provider and judge_model == gen_model:
        print("note: the same model generates and judges; --judge-provider "
              "gemini (or groq) gives an independent judge.")

    from retrieve import HybridRetriever

    retriever = HybridRetriever(args.data_dir, load_reranker=True)
    configs = {c: {k: v for k, v in CONFIGS[c].items() if k != "multi_query"}
               for c in args.configs}
    # configs with query expansion share the retrieval eval's rephrasing cache
    expanders = {}
    for c in args.configs:
        if CONFIGS[c]["multi_query"]:
            from query_expansion import QueryExpander
            expanders[c] = QueryExpander(
                make_complete_fn(gen_model, provider=provider), model_name=gen_model,
                n=CONFIGS[c]["multi_query"],
                cache_path=args.data_dir / "query_expansions.json")
    out_dir = args.out_root / args.run_name
    print(f"{len(queries)} queries x {len(configs)} configs; generator "
          f"{provider}/{gen_model}, judge {judge_provider}/{judge_model}")
    try:
        records = run(queries, configs, retriever,
                      make_complete_fn(gen_model, provider=provider),
                      make_complete_fn(judge_model, provider=judge_provider),
                      out_dir / "cache.jsonl", gen_model, judge_model,
                      expanders=expanders)
    except RateLimitTooLong as e:
        sys.exit(f"\n{e}\nEverything so far is cached in {out_dir}; run the same "
                 "command later to continue.")
    done, total = len(records), len(queries) * len(configs)
    compared = len(common_questions(records, list(configs)))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_info.json").write_text(json.dumps({
        "configs": configs, "n_queries": len(queries),
        "n_compared": compared, "complete": done == total,
        "pairs_done": done, "pairs_total": total,
        "generator": f"{provider}/{gen_model}", "judge": f"{judge_provider}/{judge_model}",
        "prompt_version": PROMPT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    rows = summarise(records, out_dir, list(configs))
    print_summary(rows)
    if done < total:
        print(f"\nINCOMPLETE: {done} of {total} (configuration, question) pairs are "
              f"done. The summary covers the {compared} questions every configuration "
              "finished; run the same command again to complete it.")
    print(f"\nWritten to {out_dir}/")


if __name__ == "__main__":
    main()
