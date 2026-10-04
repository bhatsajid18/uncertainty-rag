"""
Review candidate eval queries into a ground-truth set, and add your own.

Subcommands (run from the repo root):
  review   step through pending candidates: keep / edit / drop / add relevant
  add      write your own question and mark which retrieved chunks answer it
  stats    counts by category and review status
  export   write the kept queries to data/eval/queries.jsonl (the ground truth)

Progress is saved after every decision, so you can quit and resume anytime.

Review checklist for each candidate:
  - Does the question make sense to someone who hasn't read the passage?
  - Does the shown passage actually answer it?
  - unanswerable: do the top matching chunks shown really NOT answer it?
    (LLM-proposed "unanswerable" questions are sometimes answered by the
    corpus - drop those.)
  - Use [a] if another shown chunk also answers the question. Labelling every
    relevant chunk matters: an unlabelled relevant chunk counts as a miss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

from rank_bm25 import BM25Okapi

CATEGORIES = ("semantic", "exact_term", "multi_paper", "unanswerable")


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)  # atomic: a crash mid-write can't corrupt your labels


def toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Corpus:
    def __init__(self, data_dir: Path):
        self.chunks = load_jsonl(data_dir / "chunks.jsonl")
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.content = [c for c in self.chunks if not c["is_reference"]]
        self.bm25 = BM25Okapi([toks(c["text"]) for c in self.content])

    def top(self, query: str, n: int, exclude: set[str] = frozenset()) -> list[dict]:
        scores = self.bm25.get_scores(toks(query))
        out = []
        for j in scores.argsort()[::-1]:
            c = self.content[j]
            if c["chunk_id"] not in exclude:
                out.append(c)
            if len(out) >= n:
                break
        return out


def show_chunk(c: dict, label: str, width: int = 600):
    loc = (f"p.{c['page_start']}" if c["page_start"] == c["page_end"]
           else f"pp.{c['page_start']}-{c['page_end']}")
    print(f"  {label} {c['chunk_id']}  {c['title'][:55]} ({loc})")
    body = " ".join(c["text"].split())[:width]
    print(textwrap.indent(textwrap.fill(body, 96), "      "))


def ask(prompt: str, valid: str) -> str:
    while True:
        ans = input(prompt).strip().lower()
        if ans and ans[0] in valid:
            return ans[0]


def pick_numbers(prompt: str, n_max: int) -> list[int]:
    raw = input(prompt).strip()
    nums = [int(x) for x in re.findall(r"\d+", raw)]
    return [x for x in nums if 1 <= x <= n_max]


# ---------------------------------------------------------------------------


def cmd_review(args):
    corpus = Corpus(args.data_dir)
    rows = load_jsonl(args.candidates)
    wanted = set(args.qids or [])
    if wanted:
        # Revisit named candidates whatever their status. Without this, a
        # decided candidate could only be reopened by hand-editing the JSONL,
        # so a label found to be incomplete later - e.g. a multi_paper query
        # whose second relevant chunk was never attached - was effectively
        # frozen, and every metric computed against it stayed wrong.
        pending = [r for r in rows if r["qid"] in wanted]
        missing = wanted - {r["qid"] for r in pending}
        if missing:
            print(f"note: no such qid(s): {', '.join(sorted(missing))}")
        if not pending:
            sys.exit("None of those qids are in the candidate file.")
        print(f"Revisiting {len(pending)} named candidate(s), current status "
              "shown per item.")
    else:
        pending = [r for r in rows if r["status"] == "pending"]
    print(f"{len(pending)} to review of {len(rows)} candidates. "
          f"Keys: [k]eep [e]dit [d]rop [a]dd-relevant [s]kip [q]uit\n")

    for i, r in enumerate(pending, 1):
        print("=" * 100)
        print(f"[{i}/{len(pending)}] {r['qid']}  category={r['category']}"
              + (f"  status={r['status']}" if wanted else ""))
        print(f"  Q: {r['question']}")
        for ev in r.get("evidence") or []:
            if ev:
                print(f"  evidence: \"{ev}\"")
        print()
        for n, rel in enumerate(r["relevant"], 1):
            c = corpus.by_id.get(rel["chunk_id"])
            if c:
                show_chunk(c, f"relevant #{n}")
        if r["category"] == "unanswerable":
            print("  Closest chunks in the corpus (should NOT answer it):")
            for n, c in enumerate(corpus.top(r["question"], 3), 1):
                show_chunk(c, f"  near #{n}", width=300)

        while True:
            choice = ask("\n  [k/e/d/a/s/q] > ", "kedasq")
            if choice == "q":
                save_jsonl(args.candidates, rows)
                print("Saved. Resume later with the same command.")
                return
            if choice == "s":
                break
            if choice == "d":
                r["status"] = "dropped"
                break
            if choice == "k":
                r["status"] = "kept"
                break
            if choice == "e":
                new_q = input("  new question: ").strip()
                if new_q:
                    r["question"] = new_q
                    r["edited"] = True
                    print(f"  Q: {r['question']}")
                continue  # still decide keep/drop after editing
            if choice == "a":
                have = {x["chunk_id"] for x in r["relevant"]}
                pool = corpus.top(r["question"], 6, exclude=have)
                for n, c in enumerate(pool, 1):
                    show_chunk(c, f"#{n}", width=350)
                for n in pick_numbers("  also relevant (e.g. 1 3, blank for none): ",
                                      len(pool)):
                    c = pool[n - 1]
                    r["relevant"].append({"chunk_id": c["chunk_id"],
                                          "text_hash": text_hash(c["text"])})
                    print(f"  + {c['chunk_id']}")
                if r["category"] == "unanswerable" and r["relevant"]:
                    print("  note: an unanswerable question with relevant chunks "
                          "is really answerable; consider [d]rop.")
                continue
        save_jsonl(args.candidates, rows)

    print("\nNo pending candidates left. Next: review_queries.py export")


def cmd_add(args):
    corpus = Corpus(args.data_dir)
    rows = load_jsonl(args.candidates)
    while True:
        q = input("\nQuestion (blank to finish): ").strip()
        if not q:
            break
        cat = input(f"Category {CATEGORIES}: ").strip()
        if cat not in CATEGORIES:
            print("  unknown category, skipped")
            continue
        relevant = []
        if cat != "unanswerable":
            pool = corpus.top(q, 10)
            for n, c in enumerate(pool, 1):
                show_chunk(c, f"#{n}", width=300)
            for n in pick_numbers("Relevant chunk numbers (e.g. 1 4): ", len(pool)):
                c = pool[n - 1]
                relevant.append({"chunk_id": c["chunk_id"],
                                 "text_hash": text_hash(c["text"])})
            if not relevant:
                print("  no relevant chunks chosen; skipped (use 'unanswerable' "
                      "if the corpus really doesn't answer it)")
                continue
        rows.append({
            "qid": f"m{sum(r['source'] == 'manual' for r in rows) + 1:03d}",
            "category": cat, "question": q, "relevant": relevant,
            "evidence": [], "source": "manual", "status": "kept",
        })
        save_jsonl(args.candidates, rows)
        print(f"  added {rows[-1]['qid']}")


def cmd_stats(args):
    rows = load_jsonl(args.candidates)
    table = Counter((r["category"], r["status"]) for r in rows)
    print(f"{'category':14s} {'kept':>6s} {'pending':>8s} {'dropped':>8s}")
    for cat in CATEGORIES:
        print(f"{cat:14s} {table[(cat, 'kept')]:6d} {table[(cat, 'pending')]:8d} "
              f"{table[(cat, 'dropped')]:8d}")
    kept = sum(v for (c, s), v in table.items() if s == "kept")
    print(f"{'total kept':14s} {kept:6d}")


def cmd_export(args):
    corpus = Corpus(args.data_dir)
    rows = load_jsonl(args.candidates)
    kept = [r for r in rows if r["status"] == "kept"]
    pending = sum(r["status"] == "pending" for r in rows)
    if pending:
        print(f"note: {pending} candidates still pending (not exported)")
    if not kept:
        sys.exit("No kept queries yet, so nothing was exported (and any existing "
                 f"{args.out} was left untouched).\n"
                 "Review first: python src/evaluation/review_queries.py review")
    out = []
    for r in kept:
        for rel in r["relevant"]:
            c = corpus.by_id.get(rel["chunk_id"])
            if c is None or text_hash(c["text"]) != rel["text_hash"]:
                sys.exit(f"{r['qid']}: chunk {rel['chunk_id']} changed since "
                         f"labelling. Was chunks.jsonl regenerated?")
        out.append({k: r[k] for k in
                    ("qid", "category", "question", "relevant", "source", "evidence")
                    if k in r})
    save_jsonl(args.out, out)
    counts = Counter(r["category"] for r in out)
    print(f"Exported {len(out)} queries to {args.out}")
    for cat in CATEGORIES:
        print(f"  {cat:13s} {counts[cat]}")


def main():
    ap = argparse.ArgumentParser(description="Review and curate eval queries.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--candidates", type=Path,
                    default=Path("data/eval/candidates.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/eval/queries.jsonl"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    rv = sub.add_parser("review")
    rv.add_argument("--qids", nargs="+", metavar="QID",
                    help="Revisit these candidates whatever their status, "
                         "instead of working through the pending queue.")
    rv.set_defaults(func=cmd_review)
    sub.add_parser("add").set_defaults(func=cmd_add, qids=None)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    sub.add_parser("export").set_defaults(func=cmd_export)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
