"""
Figures: pull each figure out of the PDF, keep its caption, and optionally have
a vision model describe it.

Why: the papers put real results in plots (accuracy against uncertainty
threshold, entropy histograms, error against adversarial perturbation).
Nothing about them is in the text a text-only pipeline indexes, except the
caption, which is usually one line. So three things happen here:

  1. **Extract.** Find "Figure N" captions, take the drawing area next to each
     one, and render it to a PNG in data/figures/. These are what the app will
     show next to an answer.
  2. **Caption.** The caption is attached to the chunk that mentions that
     figure and becomes part of the text that chunk is indexed by, so
     "the accuracy-versus-uncertainty plot" can retrieve it.
  3. **Describe** (optional, --describe). A vision model says what the plot
     shows - axes, curves, trend - and that goes into the index too. Groq has
     no vision model here, so this needs --provider gemini.

A description is a description, not a measurement: the prompt tells the model
not to read values off the plot, and answers still cite the paper text. What
this buys is retrieval ("which paper plots entropy against perturbation?") and
something to show the user.

Output:
  data/figures/<arxiv_id>_p<page>_fig<n>.png
  data/figures.json   {"1806.01768": [{"page": 7, "figure_no": "2",
                       "caption": "...", "image": "data/figures/...png",
                       "description": "..."}]}

Usage (from the repo root):
  python src/rag/figures.py                              # extract + captions
  python src/rag/figures.py --describe --provider gemini # add descriptions
  python src/rag/figures.py --show 1806.01768
  python src/rag/build_index.py build                    # picks figures.json up
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import (  # noqa: E402
    RateLimitTooLong, add_provider_args, exit_on_rate_limit, get_backend, resolve,
)
from table_extract import page_lines  # noqa: E402

# The separator is required: a caption reads "Figure 2: ..." or "Figure 2. ...",
# while a wrapped prose line can easily start "Figure 2 shows that ...".
CAPTION = re.compile(r"^\s*(?:Figure|Fig\.)\s+(\d+[a-z]?)\s*[:.]\s", re.I)
MAX_FIGURE_HEIGHT = 420.0   # points: a figure rarely fills more than half a page
MIN_FIGURE_HEIGHT = 40.0
CAPTION_LINES = 3
RENDER_DPI = 120

VISION_SYSTEM = (
    "You describe figures from machine learning papers for a search index. "
    "Say what kind of plot it is, what each axis shows, which methods, curves "
    "or panels appear, and the overall trend or comparison. Two to four "
    "sentences. Do not read numeric values off the plot and do not guess at "
    "anything you cannot see clearly; say what is shown, not what it proves."
)


def caption_blocks(lines: list[dict]) -> list[tuple[int, str]]:
    """(line index, caption text) for every figure caption on the page."""
    out = []
    for i, line in enumerate(lines):
        if not CAPTION.match(line["text"]):
            continue
        text = [line["text"]]
        for nxt in lines[i + 1:i + CAPTION_LINES]:
            if CAPTION.match(nxt["text"]) or not nxt["text"].strip():
                break
            gap = nxt["y0"] - line["y1"]
            if gap > 1.6 * max(6.0, line["y1"] - line["y0"]):
                break
            text.append(nxt["text"])
        out.append((i, " ".join(" ".join(text).split())))
    return out


def graphic_boxes(page) -> list[pymupdf.Rect]:
    """Bounding boxes of the drawings and images on the page.

    Plots in these papers are vector drawings, not embedded images, so both
    are collected. Boxes as large as the page are background rectangles.
    """
    boxes = []
    page_area = abs(page.rect)
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if r.is_empty or abs(r) > 0.9 * page_area:
            continue
        boxes.append(r)
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if not r.is_empty and abs(r) <= 0.9 * page_area:
            boxes.append(r)
    return boxes


def figure_rect(page, caption_line: dict, boxes: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    """The area the figure occupies, next to its caption.

    Graphics above the caption are preferred (the usual layout); if there are
    none, graphics below it are used, for papers that caption above the figure.
    """
    for side in ("above", "below"):
        if side == "above":
            near = [b for b in boxes
                    if b.y1 <= caption_line["y0"] + 2
                    and b.y1 > caption_line["y0"] - MAX_FIGURE_HEIGHT]
        else:
            near = [b for b in boxes
                    if b.y0 >= caption_line["y1"] - 2
                    and b.y0 < caption_line["y1"] + MAX_FIGURE_HEIGHT]
        if not near:
            continue
        rect = near[0]
        for b in near[1:]:
            rect |= b
        rect = rect & page.rect
        if rect.height >= MIN_FIGURE_HEIGHT and rect.width >= MIN_FIGURE_HEIGHT:
            return rect + (-4, -4, 4, 4)  # a little air around it
    return None


def extract_figures(pdf_path: Path, arxiv_id: str, out_dir: Path,
                    dpi: int = RENDER_DPI) -> list[dict]:
    """Render every figure of one paper; returns one record per figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            lines = page_lines(page)
            boxes = graphic_boxes(page)
            for idx, caption in caption_blocks(lines):
                rect = figure_rect(page, lines[idx], boxes)
                if rect is None:
                    continue
                number = CAPTION.match(caption).group(1)
                if any(f["page"] == page_no and f["figure_no"] == number
                       for f in figures):
                    continue  # the caption line wrapped, or repeats
                image = out_dir / f"{arxiv_id}_p{page_no}_fig{number}.png"
                page.get_pixmap(clip=rect, dpi=dpi).save(image)
                figures.append({
                    "page": page_no, "figure_no": number, "caption": caption[:400],
                    "image": str(image),
                    "width": round(rect.width), "height": round(rect.height),
                })
    return figures


# --- descriptions ----------------------------------------------------------------


def describe_figures(figures: dict, backend, model: str, figures_path: Path,
                     limit: int = 0, log=print, failures: list | None = None) -> dict:
    """Add a vision description to every figure that has none. Saves as it goes.

    Like table_notes.generate_notes: a figure the provider could not serve is
    skipped (still undescribed, so a later run retries it), and three failures
    in a row stop the run.
    """
    from llm import ProviderUnavailable

    failures = [] if failures is None else failures
    in_a_row = 0
    todo = [(paper, f) for paper, items in figures.items() for f in items
            if not f.get("description")]
    if limit:
        todo = todo[:limit]
    for n, (paper, fig) in enumerate(todo, 1):
        log(f"  [{n}/{len(todo)}] {paper} p.{fig['page']} figure {fig['figure_no']}")
        prompt = (f"Caption: {fig['caption']}\n\nDescribe this figure for a search "
                  "index.")
        try:
            text = backend.describe_image(model, fig["image"], VISION_SYSTEM, prompt)
        except ProviderUnavailable as e:
            failures.append(f"{paper} figure {fig['figure_no']}")
            in_a_row += 1
            log(f"    skipped: {e}")
            if in_a_row >= 3:
                log("  3 figures in a row failed; stopping. Run the same command "
                    "later.")
                break
            continue
        in_a_row = 0
        fig["description"] = " ".join(text.split())
        fig["described_by"] = model
        figures_path.write_text(json.dumps(figures, indent=1, ensure_ascii=False))
    return figures


# --- attaching to chunks -----------------------------------------------------------


def figure_key(number: str) -> re.Pattern:
    """Matches how a chunk's text refers to this figure ("Figure 3", "Fig. 3")."""
    return re.compile(rf"\b(?:Figure|Fig\.?)\s*{re.escape(number)}\b", re.I)


def attach_figures(chunks: list[dict], figures: dict) -> dict:
    """Give each chunk the captions (and descriptions) of the figures it mentions.

    Matching is by reference in the text, not by page: the sentence discussing
    Figure 3 is often a page away from the plot, and that sentence is the chunk
    a question about the figure should retrieve.
    """
    stats = {"figures": 0, "attached": 0, "described": 0}
    for arxiv_id, items in figures.items():
        for fig in items:
            stats["figures"] += 1
            stats["described"] += bool(fig.get("description"))
            pattern = figure_key(fig["figure_no"])
            note = f"Figure {fig['figure_no']}: {fig['caption']}"
            if fig.get("description"):
                note += " " + fig["description"]
            placed = False
            for c in chunks:
                if c["arxiv_id"] != arxiv_id or not pattern.search(c["text"]):
                    continue
                notes = [n for n in c.get("figure_note", "").split("\n") if n.strip()]
                if note not in notes:
                    notes.append(note)
                c["figure_note"] = "\n".join(notes)
                images = [i for i in c.get("figure_images", "").split(",") if i]
                if fig["image"] not in images:
                    images.append(fig["image"])
                c["figure_images"] = ",".join(images)
                placed = True
            stats["attached"] += bool(placed)
    return stats


def load_figures(data_dir: Path) -> dict:
    path = data_dir / "figures.json"
    return json.loads(path.read_text()) if path.exists() else {}


@exit_on_rate_limit
def main():
    ap = argparse.ArgumentParser(description="Extract and describe paper figures.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--describe", action="store_true",
                    help="Also describe each figure with a vision model.")
    ap.add_argument("--limit", type=int, default=0,
                    help="With --describe: only the next N figures.")
    ap.add_argument("--dpi", type=int, default=RENDER_DPI)
    ap.add_argument("--show", metavar="ARXIV_ID", help="List one paper's figures.")
    add_provider_args(ap)
    args = ap.parse_args()

    figures_path = args.data_dir / "figures.json"
    figures = load_figures(args.data_dir)

    if args.show:
        for fig in figures.get(args.show, []):
            print(f"\n--- p.{fig['page']} figure {fig['figure_no']}  "
                  f"({fig['width']}x{fig['height']}pt)  {fig['image']}")
            print(f"caption: {fig['caption']}")
            print(f"description: {fig.get('description', '(none yet)')}")
        if not figures.get(args.show):
            print(f"No figures recorded for {args.show}.")
        return

    if not figures:
        meta_path = args.data_dir / "metadata.json"
        if not meta_path.exists():
            sys.exit(f"No {meta_path}. Run download_papers.py first.")
        for rec in json.loads(meta_path.read_text()):
            pdf_path = Path(rec.get("pdf_path", ""))
            if not pdf_path.exists():
                print(f"  ! missing PDF for {rec['arxiv_id']}", file=sys.stderr)
                continue
            found = extract_figures(pdf_path, rec["arxiv_id"],
                                    args.data_dir / "figures", args.dpi)
            figures[rec["arxiv_id"]] = found
            print(f"  = {rec['arxiv_id']}: {len(found)} figure(s)")
        figures_path.write_text(json.dumps(figures, indent=1, ensure_ascii=False))
        total = sum(len(v) for v in figures.values())
        print(f"\n{total} figure(s) -> {figures_path} and {args.data_dir}/figures/")
        print("Look at a few of the PNGs before describing them.")

    if args.describe:
        provider, model, _ = resolve(args)
        if provider != "gemini":
            sys.exit("Descriptions need a vision model: add --provider gemini")
        try:
            describe_figures(figures, get_backend(provider), model, figures_path,
                             args.limit)
        except RateLimitTooLong as e:
            sys.exit(f"\n{e}\nProgress is saved in {figures_path}; run again later.")
        done = sum(1 for v in figures.values() for f in v if f.get("description"))
        print(f"\n{done} figure(s) described -> {figures_path}")

    print("Next: python src/rag/build_index.py build")


if __name__ == "__main__":
    main()
