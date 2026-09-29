"""
Diagnose how a results table reaches the model.

Chat answers have misread tables (a baseline column reported as the method's
result; one number repeated for two datasets). Before building table-aware
extraction, this shows, for one paper and one keyword:

  1. what the model sees - the flattened chunk text around the keyword
  2. what PyMuPDF's table finder recovers from the same PDF page, with both
     detection strategies:
       lines  uses drawn ruling lines (misses tables with only horizontal
              rules, which is most LaTeX/booktabs tables)
       text   infers columns from how the words line up

If `text` (or `lines`) recovers the right rows and columns, table-aware
extraction is worth building on it; if both scramble the table, it isn't.

Usage (from the repo root):
  python src/rag/debug_tables.py 1806.01768 --find CIFAR5
  python src/rag/debug_tables.py 1905.00076 --find "C10 ERR"
  python src/rag/debug_tables.py 1905.00076 --page 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import pymupdf


def keyword_pattern(keyword: str) -> re.Pattern:
    """Case-insensitive, tolerant of an optional hyphen or space between a
    word and a number, so CIFAR5, CIFAR-5 and CIFAR 5 all match."""
    parts = re.findall(r"[A-Za-z]+|\d+", keyword)
    return re.compile(r"[\s\-]?".join(re.escape(p) for p in parts), re.I)


def render(rows: list[list], width: int = 14) -> str:
    def cell(v):
        v = " ".join(str(v).split()) if v is not None else ""
        return (v[: width - 1] + "…") if len(v) > width else v
    rows = [r for r in rows if any(v not in (None, "") for v in r)]  # drop blank rows
    return "\n".join(" | ".join(f"{cell(v):{width}s}" for v in row) for row in rows)


def model_view(chunks: list[dict], pat: re.Pattern, window: int = 450) -> set[int]:
    pages = set()
    print("=" * 100)
    print("1. WHAT THE MODEL SEES (flattened chunk text)")
    print("=" * 100)
    hits = [c for c in chunks if pat.search(c["text"])]
    if not hits:
        print("  no chunk contains that keyword")
    for c in hits:
        m = pat.search(c["text"])
        start, end = max(0, m.start() - window), m.end() + window
        print(f"\n--- {c['chunk_id']}  pp.{c['page_start']}-{c['page_end']}"
              f"{'  [bibliography]' if c['is_reference'] else ''}")
        snippet = c["text"][start:end]
        print(textwrap.indent(textwrap.fill(snippet, 110), "  "))
        pages.update(range(c["page_start"], c["page_end"] + 1))
    return pages


def table_view(pdf_path: Path, pages: set[int], pat: re.Pattern | None):
    print("\n" + "=" * 100)
    print("2. WHAT PyMuPDF's TABLE FINDER RECOVERS")
    print("=" * 100)
    with pymupdf.open(pdf_path) as doc:
        for pno in sorted(pages):
            if not 1 <= pno <= doc.page_count:
                continue
            page = doc[pno - 1]
            for strategy in ("lines", "text"):
                try:
                    tabs = page.find_tables(strategy=strategy)
                except Exception as e:  # noqa: BLE001 - version differences
                    print(f"\n--- page {pno}, strategy={strategy}: failed ({e})")
                    continue
                found = list(tabs)
                print(f"\n--- page {pno}, strategy={strategy}: {len(found)} table(s)")
                for t_i, tab in enumerate(found, 1):
                    rows = tab.extract()
                    flat = " ".join(str(v) for r in rows for v in r if v)
                    if pat and not pat.search(flat):
                        print(f"  table {t_i}: {len(rows)} rows x {tab.col_count} cols "
                              "(doesn't contain the keyword, not shown)")
                        continue
                    print(f"  table {t_i}: {len(rows)} rows x {tab.col_count} cols")
                    print(textwrap.indent(render(rows), "    "))


def main():
    ap = argparse.ArgumentParser(description="Inspect how a table reaches the model.")
    ap.add_argument("arxiv_id")
    ap.add_argument("--find", help="Keyword to locate the table (e.g. CIFAR5).")
    ap.add_argument("--page", type=int, nargs="+", help="Inspect these pages instead.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    if not args.find and not args.page:
        sys.exit("Give --find KEYWORD or --page N")

    chunks = [json.loads(line) for line in
              (args.data_dir / "chunks.jsonl").read_text().splitlines() if line.strip()]
    chunks = [c for c in chunks if c["arxiv_id"] == args.arxiv_id]
    if not chunks:
        sys.exit(f"No chunks for {args.arxiv_id}")
    meta = {r["arxiv_id"]: r for r in json.loads((args.data_dir / "metadata.json").read_text())}
    pdf_path = Path(meta[args.arxiv_id]["pdf_path"])

    pat = keyword_pattern(args.find) if args.find else None
    pages = set(args.page or []) or (model_view(chunks, pat) if pat else set())
    if args.page and pat:
        model_view(chunks, pat)
    if not pages:
        return
    table_view(pdf_path, pages, pat)


if __name__ == "__main__":
    main()
