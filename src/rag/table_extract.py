"""
Rebuild results tables from the PDF's own geometry - no LLM, nothing invented.

PyMuPDF gives every word with its bounding box. A LaTeX results table is a set
of words whose columns line up, sitting next to a "Table N:" caption, so the
grid can be recovered directly:

  1. group words into lines (PyMuPDF's own line grouping)
  2. split each line into cells wherever the horizontal gap is much wider than
     a word space
  3. find "Table N" captions, and grow a block of multi-cell lines away from
     each caption while the lines stay close together
  4. line the cells up into columns: the widest row is the template, and every
     other cell joins the column its x-range overlaps most; anything that fits
     nowhere starts a new column
  5. keep the result only if it looks like a results table (enough rows,
     columns and numbers)

Why not `find_tables`: PyMuPDF's own finder failed on these papers - `lines`
needs ruling lines that booktabs tables don't draw, and `text` swallowed whole
pages (54x8, 79x6). This works from the caption outwards instead of trying to
find tables anywhere on the page, which is what made those attempts fail.

Compared with the LLM notes (table_notes.py): this cannot invent or misplace a
value, but it fails on layouts it doesn't fit. Both run: a geometric grid is
used when it passes the checks, otherwise the LLM's, and the LLM description
is used either way for retrieval.

Output: data/tables.json
  {"1905.13472": [{"page": 7, "caption": "Table 2: ...", "markdown": "| ...",
                   "n_rows": 8, "n_cols": 5}], ...}

Usage (from the repo root):
  python src/rag/table_extract.py                     # all papers -> data/tables.json
  python src/rag/table_extract.py --show 1905.13472   # print what was found
  python src/rag/table_extract.py --show 1806.01768 --page 7
  python src/rag/table_extract.py --debug 1806.01768 --page 7   # why not?
  python src/rag/build_index.py build                 # picks tables.json up
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import pymupdf

# "Table 3:" or "Table 3." - the separator is required, so a wrapped prose
# line starting "Table 3 reports ..." is not mistaken for a caption.
CAPTION = re.compile(r"^\s*(?:TABLE|Table)\s+(\d+|[IVX]+)\s*[:.]\s")
_NUMERIC = re.compile(r"^[±+\-(]?\d+(?:[.,]\d+)?%?\)?$|^[±+\-]?\d*\.\d+")
MIN_GAP = 6.0          # points; a word space is ~2-4pt at 9-10pt type
MIN_ROWS = 3           # header plus two data rows
MIN_COLS = 2
MIN_NUMERIC = 0.3      # share of body cells that must look like values


def estimate_space(rows: list[list[tuple]]) -> float:
    """Typical width of a word space on this page.

    Taken from the small gaps only: the median of every gap on the page would
    be dominated by column gaps on a table-heavy page, and a threshold derived
    from it would never split a header row (measured: a median of 65pt on a
    five-column header, so nothing split).
    """
    gaps = [b[0] - a[2] for words in rows
            for a, b in zip(sorted(words, key=lambda w: w[0]),
                            sorted(words, key=lambda w: w[0])[1:])]
    small = [g for g in gaps if 0 < g <= 8.0]
    return statistics.median(small) if small else 3.0


def split_cells(words: list[tuple], threshold: float = MIN_GAP) -> list[tuple]:
    """Split one line's words into cells wherever the gap reaches `threshold`.

    words are PyMuPDF (x0, y0, x1, y1, text, ...) tuples. A line of prose stays
    one cell; a table row splits at its columns.
    """
    words = sorted(words, key=lambda w: w[0])
    if not words:
        return []
    gaps = [b[0] - a[2] for a, b in zip(words, words[1:])]
    cells, start = [], 0
    for i, gap in enumerate(gaps, start=1):
        if gap >= threshold:
            cells.append(words[start:i])
            start = i
    cells.append(words[start:])
    return [(c[0][0], c[-1][2], " ".join(w[4] for w in c)) for c in cells if c]


def page_lines(page, tol_factor: float = 0.6) -> list[dict]:
    """Lines of the page, each with its vertical extent and its cells.

    Words are grouped by vertical position rather than by PyMuPDF's own block
    and line numbers: a table row is often several blocks (each column can be
    its own text object), and grouping by block would put every cell on a line
    of its own. Two words share a line when their vertical centres are within
    `tol_factor` of a word height.
    """
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    if not words:
        return []
    tol = tol_factor * statistics.median(w[3] - w[1] for w in words)
    rows: list[list] = []
    for w in words:
        centre = (w[1] + w[3]) / 2
        if rows and abs(centre - rows[-1][0]) <= tol:
            rows[-1][1].append(w)
        else:
            rows.append([centre, [w]])
    threshold = max(MIN_GAP, 2.5 * estimate_space([r[1] for r in rows]))
    lines = []
    for _, row_words in rows:
        cells = split_cells(row_words, threshold)
        lines.append({
            "y0": min(w[1] for w in row_words), "y1": max(w[3] for w in row_words),
            "cells": cells, "text": " ".join(c[2] for c in cells),
        })
    return lines


def _numeric_share(cells: list[str]) -> float:
    body = [c for c in cells if c.strip()]
    if not body:
        return 0.0
    return sum(bool(_NUMERIC.match(c.strip())) for c in body) / len(body)


def figure_areas(page, min_size: float = 20.0) -> list:
    """Rectangles that belong to figures, not tables.

    Drawings and images big enough to be a plot. Thin rectangles are skipped:
    a booktabs rule is a drawing too, and it lies across the table itself.
    """
    out, page_area = [], abs(page.rect)
    boxes = [d["rect"] for d in page.get_drawings()]
    boxes += [i["bbox"] for i in page.get_image_info()]
    for b in boxes:
        r = pymupdf.Rect(b)
        if r.is_empty or abs(r) > 0.9 * page_area:
            continue
        if r.width >= min_size and r.height >= min_size:
            out.append(r)
    return out


def _in_figure(cell: tuple, y0: float, y1: float, areas: list) -> bool:
    x = (cell[0] + cell[1]) / 2
    y = (y0 + y1) / 2
    return any(r.x0 <= x <= r.x1 and r.y0 <= y <= r.y1 for r in areas)


def grow_block(lines: list[dict], caption_idx: int, step: int,
               band: tuple[float, float], figures: list | None = None,
               max_gap_factor: float = 3.0,
               caption_gap_factor: float = 5.0, max_band_jump: float = 160.0
               ) -> list[list[tuple]]:
    """Rows of the table next to a caption, running away from it.

    `band` is the horizontal span the table occupies, seeded from the caption
    and widened by each row taken (a table is usually wider than its caption).
    Only cells inside the band - or within max_band_jump of its edge, which is
    wider than a column gap and narrower than the gutter between two pieces of
    page furniture - are taken, so a figure printed BESIDE the table is not
    read as extra columns. That layout (Sensoy et al. p.7: Figure 2 on the
    left, Table 1 on the right) is why this is per-band and not per-line.

    Cells that fall inside a figure's drawing area are dropped first, which is
    what keeps a figure printed beside a table out of the grid.

    Growth stops at a line with fewer than two cells in the band (prose), or
    when the vertical gap to the next line is much larger than a line height.
    The caption is set off from the body by a wider gap than the rows are from
    each other, so the first step allows more.
    """
    lo, hi = band
    rows, i = [], caption_idx + step
    while 0 <= i < len(lines):
        line, prev = lines[i], lines[i - step]
        cells = [c for c in line["cells"]
                 if c[1] >= lo - max_band_jump and c[0] <= hi + max_band_jump
                 and not _in_figure(c, line["y0"], line["y1"], figures or [])]
        height = max(6.0, line["y1"] - line["y0"])
        gap = abs((line["y0"] + line["y1"]) / 2 - (prev["y0"] + prev["y1"]) / 2)
        limit = caption_gap_factor if not rows else max_gap_factor
        if len(cells) < 2 or gap > limit * height:
            break
        rows.append(cells)
        lo = min(lo, min(c[0] for c in cells))
        hi = max(hi, max(c[1] for c in cells))
        i += step
    return rows if step > 0 else rows[::-1]


def to_grid(rows: list[list[tuple]]) -> list[list[str]]:
    """Line cells up into columns using the widest row as the template."""
    template = max(rows, key=len)
    columns = [[x0, x1] for x0, x1, _ in template]
    grid = []
    for row in rows:
        placed: dict[int, str] = {}
        for x0, x1, text in row:
            overlaps = [(min(x1, c1) - max(x0, c0), j)
                        for j, (c0, c1) in enumerate(columns)]
            width, j = max(overlaps) if overlaps else (0, -1)
            if width <= 0:  # a cell in no known column: add one
                columns.append([x0, x1])
                j = len(columns) - 1
                placed[j] = text
                continue
            columns[j] = [min(columns[j][0], x0), max(columns[j][1], x1)]
            placed[j] = f"{placed[j]} {text}" if j in placed else text
        grid.append(placed)
    order = sorted(range(len(columns)), key=lambda j: columns[j][0])
    return [[placed.get(j, "") for j in order] for placed in grid]


def merge_header_rows(grid: list[list[str]]) -> list[list[str]]:
    """Fold a second header row (no numbers, e.g. metric names under dataset
    names) into the first, so the Markdown table has one header."""
    if len(grid) < 3 or _numeric_share(grid[1]) > 0:
        return grid
    header = [" ".join(p for p in pair if p).strip()
              for pair in zip(grid[0], grid[1])]
    return [header, *grid[2:]]


def grid_to_markdown(grid: list[list[str]]) -> str:
    width = len(grid[0])
    rows = [grid[0], ["---"] * width, *grid[1:]]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def usable_reason(grid: list[list[str]]) -> tuple[bool, str]:
    """Is this grid a results table, and if not, what failed?"""
    if len(grid) < MIN_ROWS:
        return False, f"{len(grid)} row(s), needs {MIN_ROWS}"
    if len(grid[0]) < MIN_COLS:
        return False, f"{len(grid[0])} column(s), needs {MIN_COLS}"
    body = [c for row in grid[1:] for c in row]
    share = _numeric_share(body)
    if share < MIN_NUMERIC:
        return False, f"only {share:.0%} of cells look like values, needs {MIN_NUMERIC:.0%}"
    empty = [j for j in range(len(grid[0]))
             if not any(row[j].strip() for row in grid[1:])]
    if empty:
        return False, f"column {empty[0] + 1} is empty in every row"
    return True, ""


def usable(grid: list[list[str]]) -> bool:
    return usable_reason(grid)[0]


def caption_cells(lines: list[dict]) -> list[tuple[int, tuple[float, float], str]]:
    """(line index, x-span, text) for every table caption, found per cell.

    Per cell, not per line: when a figure sits beside a table, the caption
    shares a line with the figure's own text, and a line-level match misses it.
    """
    found = []
    for i, line in enumerate(lines):
        for x0, x1, text in line["cells"]:
            if CAPTION.match(text):
                found.append((i, (x0, x1), " ".join(text.split())[:200]))
    return found


def analyse_page(page, page_no: int) -> tuple[list[dict], list[dict]]:
    """Both what was found and why: (lines, one report per caption)."""
    lines = page_lines(page)
    figures = figure_areas(page)
    reports = []
    for idx, band, caption in caption_cells(lines):
        report = {"line": idx, "caption": caption, "band": band, "sides": {}}
        for step in (-1, 1):
            rows = grow_block(lines, idx, step, band, figures)
            if rows:
                # second pass with the band the first pass learned: the row
                # nearest the caption would otherwise lose the columns that
                # reach past the caption's own width
                full = (min(band[0], min(c[0] for r in rows for c in r)),
                        max(band[1], max(c[1] for r in rows for c in r)))
                rows = grow_block(lines, idx, step, full, figures)
            if len(rows) < MIN_ROWS:
                report["sides"][step] = (None, f"{len(rows)} row(s) next to the caption")
                continue
            grid = merge_header_rows(to_grid(rows))
            ok, why = usable_reason(grid)
            report["sides"][step] = (grid if ok else None, why or "kept")
        reports.append(report)
    return lines, reports


def tables_on_page(page, page_no: int) -> list[dict]:
    """Every table this page's captions lead to."""
    _, reports = analyse_page(page, page_no)
    found = []
    for report in reports:
        # the body sits above the caption in most papers and below in some:
        # take whichever side yields a usable grid, preferring the larger
        grids = [g for g, _ in report["sides"].values() if g]
        if not grids:
            continue
        best = max(grids, key=len)
        found.append({
            "page": page_no, "caption": report["caption"],
            "markdown": grid_to_markdown(best),
            "n_rows": len(best) - 1, "n_cols": len(best[0]),
        })
    return found


def extract_tables(pdf_path: Path) -> list[dict]:
    out = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            out.extend(tables_on_page(page, page_no))
    return out


# --- attaching tables to chunks ------------------------------------------------


def _values(markdown: str) -> list[str]:
    body = markdown.splitlines()[2:]
    return [c.strip() for line in body for c in line.strip("|").split("|")
            if _NUMERIC.match(c.strip())]


def attach_tables(chunks: list[dict], tables: dict, min_overlap: float = 0.4) -> dict:
    """Put each rebuilt table on the chunk(s) whose text contains it.

    A table is attached when the chunk covers its page and holds either the
    caption or at least `min_overlap` of the table's values - a chunk on the
    same page that doesn't contain the table's text is not given it. Overrides
    any grid from table_notes.py, and records which produced it.
    """
    stats = {"attached": 0, "tables": 0, "unplaced": 0}
    for arxiv_id, paper_tables in tables.items():
        for t in paper_tables:
            stats["tables"] += 1
            values = _values(t["markdown"])
            caption_key = " ".join(t["caption"].split()[:2])  # "Table 3:"
            placed = False
            for c in chunks:
                if c["arxiv_id"] != arxiv_id or not (
                        c["page_start"] <= t["page"] <= c["page_end"]):
                    continue
                text = c["text"]
                hits = sum(v in text for v in values)
                if not (caption_key in text
                        or (values and hits / len(values) >= min_overlap)):
                    continue
                grids = [g for g in c.get("table_markdown", "").split("\n\n")
                         if g.strip() and g != t["markdown"]]
                c["table_markdown"] = "\n\n".join([t["markdown"], *grids])
                c["table_grid_source"] = "geometry"
                stats["attached"] += 1
                placed = True
            if not placed:
                stats["unplaced"] += 1
    return stats


def debug_page(page, page_no: int, width: int = 90, only_interesting: bool = True):
    """Why this page's captions did or didn't yield a table.

    Pages with neither a caption nor a mention of one are skipped by default,
    so a 47-page survey prints a handful of pages instead of all of them.
    """
    lines, reports = analyse_page(page, page_no)
    mentions = [ln for ln in lines if re.search(r"\bTable\s+\d", ln["text"])]
    if only_interesting and not reports and not mentions:
        return
    print(f"\n=== page {page_no}: {len(lines)} line(s), "
          f"{len(figure_areas(page))} figure area(s), "
          f"{len(reports)} caption(s) matched, "
          f"{len(mentions)} line(s) mentioning a table")
    if not reports and mentions:
        print("  no caption matched; lines mentioning 'Table N' were:")
        for ln in mentions[:4]:
            print(f"    [{len(ln['cells'])} cell(s)] {ln['text'][:width]}")
    for r in reports:
        print(f"\n  caption (line {r['line']}, x {r['band'][0]:.0f}-{r['band'][1]:.0f}): "
              f"{r['caption'][:width]}")
        for step, (grid, why) in sorted(r["sides"].items()):
            where = "above" if step == -1 else "below"
            print(f"    {where}: {'KEPT ' if grid else 'rejected - '}{why}"
                  + (f" ({len(grid) - 1} rows x {len(grid[0])} cols)" if grid else ""))
        lo, hi = max(0, r["line"] - 6), min(len(lines), r["line"] + 7)
        print("    lines around it (cells | text):")
        for i in range(lo, hi):
            mark = ">" if i == r["line"] else " "
            print(f"    {mark} {i:3d} [{len(lines[i]['cells'])}] "
                  f"{lines[i]['text'][:width]}")


def load_tables(data_dir: Path) -> dict:
    path = data_dir / "tables.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    ap = argparse.ArgumentParser(description="Rebuild tables from PDF geometry.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--show", metavar="ARXIV_ID",
                    help="Print the tables found for one paper instead of writing.")
    ap.add_argument("--page", type=int, nargs="+",
                    help="With --show or --debug: only these pages.")
    ap.add_argument("--debug", metavar="ARXIV_ID",
                    help="Explain, page by page, why captions did or did not "
                         "produce a table.")
    args = ap.parse_args()

    meta_path = args.data_dir / "metadata.json"
    if not meta_path.exists():
        sys.exit(f"No {meta_path}. Run download_papers.py first.")
    records = json.loads(meta_path.read_text())
    wanted = args.show or args.debug
    if wanted:
        records = [r for r in records if r["arxiv_id"] == wanted]
        if not records:
            sys.exit(f"{wanted} is not in the corpus")

    if args.debug:
        pdf_path = Path(records[0]["pdf_path"])
        with pymupdf.open(pdf_path) as doc:
            pages = args.page or range(1, doc.page_count + 1)
            for page_no in pages:
                if 1 <= page_no <= doc.page_count:
                    debug_page(doc[page_no - 1], page_no,
                               only_interesting=not args.page)
        return

    tables, total = {}, 0
    for rec in records:
        pdf_path = Path(rec.get("pdf_path", ""))
        if not pdf_path.exists():
            print(f"  ! missing PDF for {rec['arxiv_id']}", file=sys.stderr)
            continue
        found = extract_tables(pdf_path)
        if args.page:
            found = [t for t in found if t["page"] in args.page]
        tables[rec["arxiv_id"]] = found
        total += len(found)
        if args.show:
            for t in found:
                print(f"\n--- p.{t['page']}  {t['caption']}")
                print(f"    ({t['n_rows']} rows x {t['n_cols']} cols)")
                print(t["markdown"])
            if not found:
                print("No tables rebuilt for this paper "
                      "(the LLM notes still apply; see table_notes.py).")
        else:
            print(f"  = {rec['arxiv_id']}: {len(found)} table(s)")

    if args.show:
        return
    out = args.data_dir / "tables.json"
    out.write_text(json.dumps(tables, indent=1, ensure_ascii=False))
    print(f"\n{total} table(s) rebuilt from {len(tables)} paper(s) -> {out}")
    print("Check a few before trusting them:")
    print("  python src/rag/table_extract.py --show <arxiv_id>")
    print("Then: python src/rag/build_index.py build")


if __name__ == "__main__":
    main()
