"""
Table notes: an LLM-written description and a rebuilt table for every chunk
that contains a results table.

Idea borrowed from multimodal RAG pipelines (e.g. unstructured.io + LLM
summaries), which index an LLM's description of a table or figure instead of
its raw content, and hand the raw content to the answering model. Here, with
our own stack (PyMuPDF text, Groq, FAISS/BM25), each table chunk gets two
notes:

  description   1-3 sentences on what the table compares: methods (which one
                is the paper's own), datasets, metrics. No numbers. It is added
                to the text the chunk is INDEXED by (see retrieve.index_text),
                because a flattened table is mostly numbers, so embeddings, BM25
                and the reranker all struggle to match it to a question.
                Observed: the chunk with EnD2's C10/C100 rows ranked #62 dense,
                #36 BM25, reranker score 0.0012.
  markdown      the table rebuilt as a Markdown grid. It is shown to the
                answering model next to the flattened text, because reading a
                flattened table means counting positions, and the model got
                that wrong (EnD's column reported as EnD2's; LSUN's column
                reported as TinyImageNet's).

A rebuilt table is only used if it passes check_markdown(): every row has as
many cells as the header, and every number in it appears in the source text.
That catches invented numbers and ragged rows, not a value placed in the
wrong cell, so the flattened text stays in the prompt and the prompt says it
is the authority. `--show CHUNK_ID` prints a chunk's notes for checking by
hand.

Notes are cached in data/table_notes.json, keyed by chunk id and pinned to a
hash of the chunk text and of the prompt, so a re-chunk or a prompt edit
regenerates only what changed. The file is saved after every chunk, so a run
stopped by Groq's daily limit resumes where it stopped.

Usage (from the repo root):
  python src/rag/table_notes.py --dry-run          # which chunks have tables
  python src/rag/table_notes.py                    # write notes (resumable)
  python src/rag/table_notes.py --show 1905.00076__0012
  python src/rag/build_index.py build              # picks the notes up
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import (  # noqa: E402
    ProviderUnavailable, RateLimitTooLong, add_provider_args, exit_on_rate_limit,
    make_complete_fn, resolve,
)
from retrieve import _table_like, expand_split_tables  # noqa: E402

TABLE_SYSTEM = (
    "You read results tables from machine learning papers. PDF extraction has "
    "flattened each table into running text: the column headers come first, then "
    "each row as a label followed by one value per column; values often carry a "
    "± term, and a header may have two levels (e.g. a dataset spanning several "
    "metric columns, or a dataset label at the start of a group of rows).\n\n"
    "For every table whose rows appear in the text, output one block in EXACTLY "
    "this format:\n\n"
    "TABLE\n"
    "CAPTION: the table's caption if the text has one, else a short title\n"
    "DESCRIPTION: 1-3 sentences for a search index: what the table compares - the "
    "methods (say which one is the paper's own if the text makes it clear), the "
    "datasets, the metrics and the setting. Spell out abbreviations the text "
    "defines. Do NOT put any numbers from the table in the description.\n"
    "GRID:\n"
    "| Row | First column | Second column |\n"
    "| --- | --- | --- |\n"
    "| first row label | value | value |\n"
    "END\n\n"
    "Rules for GRID:\n"
    "- one header row only; combine two-level headers into one label "
    "(e.g. 'C10 ERR')\n"
    "- every row must have exactly as many cells as the header\n"
    "- copy each value exactly as written, including its ± term; leave a cell "
    "empty if the table leaves it empty\n"
    "- if you cannot tell with certainty which value belongs to which column, "
    "write 'GRID: none' and give the caption and description only.\n\n"
    "If the text contains no table rows, output exactly: NO TABLES\n"
    "Output nothing else: no JSON, no code fences, no commentary."
)
PROMPT_VERSION = hashlib.sha1(TABLE_SYSTEM.encode()).hexdigest()[:8]
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_LETTERS = re.compile(r"[A-Za-z]")


def _is_value_cell(cell: str) -> bool:
    """A cell holding a result ('8.0 ±0.4', '76', '± NA'), not a label ('C10')."""
    return bool(_NUMBER.search(cell)) and not _LETTERS.search(
        re.sub(r"\bN/?A\b", "", cell))


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def has_table(text: str, window: int = 20, stride: int = 5) -> bool:
    """True if some run of `window` tokens is mostly table cells.

    20 tokens, not the 30 used at chunk edges, so that small tables count:
    Sensoy et al.'s 7-row accuracy table scores 0.65 over 20 tokens but only
    0.50 over 30 once caption words are included, while number-heavy prose
    (training settings, "log(10) and log(5)") stays near 0.3.
    """
    tokens = text.split()
    if len(tokens) < window:
        return _table_like(tokens)
    return any(_table_like(tokens[i:i + window])
               for i in range(0, len(tokens) - window + 1, stride))


def table_chunks(chunks: list[dict]) -> list[dict]:
    return [c for c in chunks if not c.get("is_reference") and has_table(c["text"])]


def extract_json(raw: str) -> dict:
    """The first {...} object in an LLM reply (tolerates text or fences around it)."""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    return json.loads(raw[start:end + 1])


# "TABLE", optionally numbered ("TABLE 2") and decorated with markdown
# emphasis or heading marks, alone on its line.
_BLOCK_START = re.compile(r"^[#*\s]*TABLE\s*\d*\s*[:*#]*\s*$", re.I)
_BLOCK_END = re.compile(r"^[#*\s]*END\s*[:*#]*\s*$", re.I)
_FIELD = re.compile(r"^[#*\s]*(CAPTION|DESCRIPTION|GRID)\s*:\s*(.*)$", re.I)
_NO_TABLES = re.compile(r"\bNO\s+TABLES\b", re.I)
_FENCE = re.compile(r"^\s*```.*$")


def parse_tables(raw: str) -> tuple[list[dict], str]:
    """Parse the model's reply into {caption, description, markdown} dicts.

    Returns (tables, error). An empty list with no error means the model found
    no table; an error means the reply didn't follow the format, and is kept in
    the notes so a bad run can be spotted and retried.

    The format is line-based rather than JSON because a Markdown grid is
    multi-line, and models routinely put raw newlines inside a JSON string,
    which makes the JSON invalid. openai/gpt-oss-20b failed that way on 57 of
    69 chunks. A JSON reply is still accepted if one arrives.

    Parsing is deliberately forgiving, because every rejected reply is a spent
    free-tier call that has to be made again: code fences are ignored, the
    block marker may be numbered ("TABLE 2") or emphasised, and a reply that
    gives the fields but omits the marker entirely is re-scanned as a single
    implicit block rather than thrown away.
    """
    if raw.lstrip().startswith("{") or '"tables"' in raw:
        try:
            tables = extract_json(raw).get("tables", [])
            return ([{"caption": str(t.get("caption") or "").strip(),
                      "description": str(t.get("description") or "").strip(),
                      "markdown": str(t.get("markdown") or "").strip()}
                     for t in tables if isinstance(t, dict)], "")
        except (ValueError, json.JSONDecodeError):
            pass  # fall through to the line format

    lines = [ln for ln in raw.splitlines() if not _FENCE.match(ln)]

    tables = _scan(lines, implicit_open=False)
    if not tables:
        # Salvage: the model gave the FIELDS but not the bare "TABLE" line that
        # opens a block, so the strict scan above skipped the whole reply and
        # the call was wasted. This was the single largest failure mode - 28 of
        # 77 chunks in one run - and the content was usually fine. Re-scan with
        # a block already open, and keep the result only if it actually carries
        # a description or a grid, so genuine chatter still reports an error.
        # A real grid, not the literal "none"/"N/A" the prompt allows, so an
        # empty shell is still reported as an error rather than stored.
        tables = [t for t in _scan(lines, implicit_open=True)
                  if t["description"] or "|" in t["markdown"]]

    for t in tables:
        if "|" not in t["markdown"]:
            t["markdown"] = ""  # "none", "N/A", or a stray line
    if not tables and not _NO_TABLES.search(raw):
        return [], "reply did not follow the TABLE/END format"
    return tables, ""


def _new_block() -> dict:
    return {"caption": "", "description": "", "markdown": ""}


def _scan(lines: list[str], implicit_open: bool) -> list[dict]:
    """One pass of the line format. With implicit_open, a block is already
    open at the first line, so fields that precede any "TABLE" marker count."""
    tables: list[dict] = []
    cur = _new_block() if implicit_open else None
    field = None
    for line in lines:
        if _BLOCK_START.match(line):
            if cur:
                tables.append(cur)
            cur, field = _new_block(), None
            continue
        if _BLOCK_END.match(line):
            if cur:
                tables.append(cur)
            cur, field = None, None
            continue
        if cur is None:
            continue
        m = _FIELD.match(line)
        if m:
            field = m.group(1).lower()
            key = "markdown" if field == "grid" else field
            cur[key] = m.group(2).strip()
            continue
        if line.strip().startswith("|"):  # a grid row, even if GRID: was omitted
            field = "grid"
            cur["markdown"] = (cur["markdown"] + "\n" + line.strip()).strip()
        elif field == "description" and line.strip():
            cur["description"] = (cur["description"] + " " + line.strip()).strip()
    if cur:
        tables.append(cur)
    return tables


def _cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def parse_markdown(md: str) -> list[list[str]]:
    """Rows of a Markdown table, header first, separator row dropped."""
    rows = []
    for line in md.strip().splitlines():
        if "|" not in line:
            continue
        cells = _cells(line)
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c) and any(cells):
            continue  # |---|:---:| separator
        rows.append(cells)
    return rows


def check_markdown(md: str, source: str) -> tuple[bool, str]:
    """Is a rebuilt table safe to show the answering model?

    Checks the two errors that can be detected mechanically: a row with the
    wrong number of cells, and a number that isn't in the source text.
    """
    rows = parse_markdown(md)
    if len(rows) < 2:
        return False, "fewer than one header and one data row"
    width = len(rows[0])
    if width < 2:
        return False, "only one column"
    ragged = [r[0] for r in rows[1:] if len(r) != width]
    if ragged:
        return False, f"{len(ragged)} row(s) with a different number of cells than the header"
    source_numbers = Counter(_NUMBER.findall(unicodedata.normalize("NFKC", source)))
    values = [c for r in rows[1:] for c in r if _is_value_cell(c)]
    table_numbers = Counter(_NUMBER.findall(unicodedata.normalize("NFKC", " ".join(values))))
    if not table_numbers:
        return False, "no numbers in the table body"
    invented = [n for n, cnt in table_numbers.items() if cnt > source_numbers.get(n, 0)]
    if invented:
        return False, "numbers not in the source text: " + ", ".join(invented[:5])
    return True, ""


def note_context(chunk: dict, chunk_by_id: Callable[[str], dict | None]) -> str:
    """The chunk text, plus the neighbouring chunk when a table runs across the
    boundary, so the model sees whole tables (same rule as answering uses)."""
    [merged] = expand_split_tables([chunk], chunk_by_id, max_expand=1)
    return merged["text"]


def describe_chunk(chunk: dict, context: str, complete_fn: Callable[[str, str], str]
                   ) -> list[dict]:
    """One LLM call: the tables in this chunk, each checked against the text."""
    raw = complete_fn(TABLE_SYSTEM, f"Paper: {chunk.get('title', '')}\n\nText:\n{context}")
    tables, error = parse_tables(raw)
    if error:
        # keep a slice of the reply: a whole run failing the same way is a
        # prompt or model problem, and guessing at it wastes a day's quota
        return [{"caption": "", "description": "", "markdown": "", "markdown_ok": False,
                 "reject_reason": error, "reply_start": " ".join(raw.split())[:300]}]
    out = []
    for t in tables:
        md = t["markdown"]
        ok, why = check_markdown(md, context) if md else (False, "model gave no grid")
        out.append({**t, "markdown_ok": ok, "reject_reason": why})
    return out


def is_current(entry: dict | None, chunk: dict) -> bool:
    return bool(entry) and entry.get("text_hash") == text_hash(chunk["text"]) \
        and entry.get("prompt_version") == PROMPT_VERSION


def failed(entry: dict) -> bool:
    """True if this chunk's notes came back with nothing usable at all."""
    return not any(t.get("description") or t.get("markdown_ok") for t in entry["tables"])


MAX_CONSECUTIVE_FAILURES = 3


def generate_notes(chunks: list[dict], complete_fn, model: str, notes_path: Path,
                   limit: int = 0, log=print, retry_failed: bool = False,
                   failures: list | None = None) -> dict:
    """Describe every table chunk not already described; save after each one.

    retry_failed also redoes chunks whose stored notes are empty - use it after
    switching model, since the notes record which model wrote them.

    A chunk the provider could not serve (busy, connection dropped - after the
    provider's own retries) is skipped and left without notes, so the same
    command later picks it up; its id goes into `failures`. Three such chunks
    in a row means the provider or the network is down, not unlucky, so the
    run stops there instead of spending minutes of back-off per chunk.
    """
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    by_id = {c["chunk_id"]: c for c in chunks}
    todo = [c for c in table_chunks(chunks)
            if not is_current(notes.get(c["chunk_id"]), c)
            or (retry_failed and failed(notes[c["chunk_id"]]))]
    if limit:
        todo = todo[:limit]
    failures = [] if failures is None else failures
    in_a_row = 0
    for n, c in enumerate(todo, 1):
        log(f"  [{n}/{len(todo)}] {c['chunk_id']}")
        try:
            tables = describe_chunk(c, note_context(c, by_id.get), complete_fn)
        except ProviderUnavailable as e:
            failures.append(c["chunk_id"])
            in_a_row += 1
            log(f"    skipped: {e}")
            if in_a_row >= MAX_CONSECUTIVE_FAILURES:
                log(f"  {in_a_row} chunks in a row failed; stopping. The provider "
                    "or the connection is down - run the same command later.")
                break
            continue
        in_a_row = 0
        notes[c["chunk_id"]] = {"text_hash": text_hash(c["text"]),
                                "prompt_version": PROMPT_VERSION, "model": model,
                                "tables": tables}
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(json.dumps(notes, indent=1, ensure_ascii=False))
    return notes


def attach_notes(chunks: list[dict], notes: dict) -> dict:
    """Copy current notes onto chunk dicts as table_note / table_markdown.

    Notes for chunks whose text changed since they were written are ignored.
    Returns counts for reporting.
    """
    stats = Counter()
    for c in chunks:
        entry = notes.get(c["chunk_id"])
        if not entry:
            continue
        if not is_current(entry, c):
            stats["stale"] += 1
            continue
        desc = " ".join(t["description"] for t in entry["tables"] if t["description"])
        mds = [t["markdown"] for t in entry["tables"] if t["markdown_ok"]]
        if desc:
            c["table_note"] = desc
            stats["described"] += 1
        if mds:
            c["table_markdown"] = "\n\n".join(mds)
            stats["rebuilt"] += 1
        rejected = sum(1 for t in entry["tables"] if t["markdown"] and not t["markdown_ok"])
        if rejected:
            stats["rejected"] += rejected
    return dict(stats)


def summarize(notes: dict) -> dict:
    """How a run went: usable grids, descriptions only, no table, unparseable."""
    stats = Counter()
    for entry in notes.values():
        tables = entry["tables"]
        if any(t.get("markdown_ok") for t in tables):
            stats["with_grid"] += 1
        elif any(t.get("description") for t in tables):
            stats["description_only"] += 1
        elif any(t.get("reject_reason", "").startswith("reply did not follow") for t in tables):
            stats["unparseable"] += 1
        else:
            stats["no_table_found"] += 1
    return dict(stats)


def load_notes(data_dir: Path) -> dict:
    path = data_dir / "table_notes.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _show(chunk_id: str, notes: dict, chunks_by_id: dict):
    entry = notes.get(chunk_id)
    if not entry:
        sys.exit(f"No notes for {chunk_id}.")
    c = chunks_by_id.get(chunk_id)
    state = "current" if c and is_current(entry, c) else "STALE (text or prompt changed)"
    print(f"{chunk_id}  ({state}, model {entry['model']})")
    for i, t in enumerate(entry["tables"], 1):
        print(f"\n--- table {i}: {t['caption']}")
        print(f"description: {t['description']}")
        verdict = "used" if t["markdown_ok"] else f"NOT used: {t['reject_reason']}"
        print(f"rebuilt table ({verdict}):\n{t['markdown'] or '(none)'}")


def _print_stats(notes: dict):
    if not notes:
        print("No notes written yet.")
        return
    stats = summarize(notes)
    labels = [("with_grid", "with a rebuilt table"),
              ("description_only", "described, no usable grid"),
              ("no_table_found", "model found no table (usually not a real table)"),
              ("unparseable", "reply did not follow the format - retry these")]
    print(f"{len(notes)} chunk(s) with notes:")
    for key, label in labels:
        if stats.get(key):
            print(f"  {stats[key]:4d}  {label}")
    if stats.get("unparseable"):
        # The stored reply may be empty, which is itself the finding: the model
        # answered with no content at all. Looking for a non-empty one used to
        # raise StopIteration and take the whole command down with it, right at
        # the point where it was trying to explain what had gone wrong.
        bad = next((cid for cid, e in notes.items()
                    if any(t.get("reply_start") for t in e["tables"])), None)
        if bad:
            print(f"  first failing reply ({bad}):")
            print("    " + next(t["reply_start"] for t in notes[bad]["tables"]
                                if t.get("reply_start"))[:200])
        else:
            print("  every failing reply was EMPTY - the model returned no "
                  "content, which usually means the small fallback model was "
                  "answering. Re-run on a stronger model or another provider.")
        print("  Retry with: python src/rag/table_notes.py --retry-failed"
              " [--model openai/gpt-oss-120b]")


@exit_on_rate_limit
def main():
    ap = argparse.ArgumentParser(description="LLM notes for table chunks.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    add_provider_args(ap)
    ap.add_argument("--limit", type=int, default=0, help="Only the next N chunks.")
    ap.add_argument("--retry-failed", action="store_true",
                    help="Also redo chunks whose notes came back empty (e.g. "
                         "after switching --model).")
    ap.add_argument("--stats", action="store_true",
                    help="Summarise the notes already written, then exit.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List the chunks that would be described; no LLM calls.")
    ap.add_argument("--show", metavar="CHUNK_ID", help="Print one chunk's notes.")
    args = ap.parse_args()

    provider, model, _ = resolve(args)
    chunks_path = args.data_dir / "chunks.jsonl"
    if not chunks_path.exists():
        sys.exit(f"No {chunks_path}. Run chunk_papers.py first.")
    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines()
              if line.strip()]
    notes_path = args.data_dir / "table_notes.json"

    if args.show:
        _show(args.show, load_notes(args.data_dir), {c["chunk_id"]: c for c in chunks})
        return

    found = table_chunks(chunks)
    notes = load_notes(args.data_dir)
    if args.stats:
        _print_stats(notes)
        return
    todo = [c for c in found
            if not is_current(notes.get(c["chunk_id"]), c)
            or (args.retry_failed and failed(notes[c["chunk_id"]]))]
    print(f"{len(found)} of {len(chunks)} chunks contain table rows; "
          f"{len(todo)} need notes.")
    if args.dry_run:
        for c in todo:
            print(f"  {c['chunk_id']}  pp.{c['page_start']}-{c['page_end']}")
        return
    if not todo:
        print("Nothing to do.")
        return

    failures: list[str] = []
    try:
        notes = generate_notes(chunks, make_complete_fn(model, provider=provider),
                               model, notes_path, args.limit,
                               retry_failed=args.retry_failed, failures=failures)
    except RateLimitTooLong as e:
        sys.exit(f"\n{e}\nProgress is saved in {notes_path}; run the same command "
                 "again later to continue.")
    print(f"\nSaved {notes_path}.")
    _print_stats(notes)
    if failures:
        print(f"\n{len(failures)} chunk(s) not reached because the provider was "
              "unavailable; they still need notes. Run the same command again "
              "(or switch with --provider).")
    print("\nNext: python src/rag/build_index.py build")


if __name__ == "__main__":
    main()
