"""
PDF parsing + chunking for the RAG corpus.

Reads the PDFs listed in data/metadata.json, extracts text page by page
with PyMuPDF, cleans it, splits into token-based overlapping chunks, and
tags each chunk with its source paper and page number(s).

References/bibliography are NOT dropped: chunks after the detected
"References" heading are tagged is_reference=True so they can be filtered
downstream (e.g. as an ablation) without re-running extraction.

Output:
  data/chunks.jsonl    one JSON object per line, each a chunk:
    {
      "chunk_id": "1806.01768__0007",
      "arxiv_id": "1806.01768",
      "title": "...",
      "year": "2018",
      "page_start": 3,
      "page_end": 4,
      "chunk_index": 7,
      "is_reference": false,
      "n_tokens": 498,
      "text": "..."
    }

Chunking is token-based using the embedding model's tokenizer, so
"512 tokens" matches what the embedder will actually see.

Usage:
  python chunk_papers.py                       # defaults
  python chunk_papers.py --chunk-size 512 --overlap 50
  python chunk_papers.py --tokenizer BAAI/bge-base-en-v1.5
"""

import argparse
import unicodedata
import json
import re
import sys
from pathlib import Path

import pymupdf  # modern import name for PyMuPDF
from transformers import AutoTokenizer

# Lines that are almost certainly page furniture, not content.
_JUNK_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                      # a lone page number
    re.compile(r"^\s*arXiv:\d+\.\d+", re.IGNORECASE),  # arXiv stamp
    re.compile(r"^\s*Preprint\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*Under review", re.IGNORECASE),
]

# Headings that mark the start of the bibliography.
_REF_HEADING = re.compile(
    r"^\s*(references|bibliography)\s*$", re.IGNORECASE
)


def clean_page_text(text: str) -> str:
    """Clean a single page's raw text."""
    # NFKC normalization decomposes typographic ligatures (fi, fl, ...) and
    # other compatibility characters into plain ASCII equivalents.
    text = unicodedata.normalize("NFKC", text)
    lines = text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(p.match(stripped) for p in _JUNK_PATTERNS):
            continue
        kept.append(stripped)
    joined = "\n".join(kept)
    # de-hyphenate words split across line breaks: "neural-\nnetwork" -> "neural-network"
    # (keep the hyphen; safer than guessing whether it was a real hyphen)
    joined = re.sub(r"(\w)-\n(\w)", r"\1\2", joined)
    # turn remaining single newlines into spaces (reflow paragraphs)
    joined = re.sub(r"\n+", " ", joined)
    # collapse runs of whitespace
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def extract_pages(pdf_path: Path) -> list[str]:
    """Return a list of cleaned page texts, index i = page i+1."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            raw = page.get_text("text")
            pages.append(clean_page_text(raw))
    return pages


# Citation markers, weighted by how strongly each indicates a bibliography.
_CITATION_PATTERNS = [
    (re.compile(r"\b(?:19|20)\d{2}\b"), 2),                        # a year
    (re.compile(r"\[\d+\]"), 3),                                    # [1]
    (re.compile(r"\[[A-Z][a-z]+[^\]]{0,40}\d{4}\]"), 3),           # [Bishop 2006]
    (re.compile(r"\bet\s+al\."), 3),
    (re.compile(r"\b[A-Z]\.\s*[A-Z]?\.?\s*[;,]"), 2),             # "C. M.," / "B.;"
    (re.compile(
        r"\b(?:In Proceedings|In Advances|Conference on|Journal of"
        r"|arXiv preprint|Transactions on|NeurIPS|ICML|ICLR|CVPR|Nature|PMLR)\b",
        re.IGNORECASE,
    ), 3),
    (re.compile(r"\d+\(\d+\):\d+"), 3),                            # 33(8):831
    (re.compile(r"\bpp?\.\s*\d+[-\u2013]\d+"), 3),                # pp. 100-109
    (re.compile(r"\d+[-\u2013]\d+\s*,\s*(?:19|20)\d{2}"), 3),     # 831-838, 2015
    # full-name author lists: "Firstname Lastname, Firstname Lastname, and X"
    (re.compile(r"[A-Z][a-z]+\s+[A-Z][a-z]+,\s+[A-Z][a-z]+\s+[A-Z][a-z]+,\s+and\s+[A-Z]"), 4),
    (re.compile(r",\s+and\s+[A-Z][a-z]+\s+[A-Z]\.?\s*[A-Z]?[a-z]*\."), 3),
]

# Chunks at or above this citation density are treated as bibliography.
# Calibrated on real chunks from this corpus: bibliography entries score
# 0.35-0.72, ordinary prose 0.00, and the hardest negatives - prose that
# cites work ("Following Sensoy et al. (2018)...") and results text full of
# years and numbers - score 0.15 and 0.06 respectively.
REFERENCE_DENSITY_THRESHOLD = 0.25


def citation_density(text: str) -> float:
    """Weighted count of citation markers per word.

    Used to classify a chunk as bibliography vs content. This replaces an
    earlier approach that found the "References" heading and tagged everything
    after it, which had two failure modes: it mis-fired on the word
    "references" appearing in acknowledgements, and - more damagingly - it
    tagged APPENDIX material as bibliography, since papers commonly run
    body -> references -> appendix. Appendix content (proofs, extra results,
    experimental detail) is real content that should stay retrievable.

    Density degrades gracefully: it needs no heading, no position heuristic,
    and no assumption about document structure.
    """
    markers = sum(len(pat.findall(text)) * weight for pat, weight in _CITATION_PATTERNS)
    words = max(1, len(text.split()))
    return markers / words


def is_reference_chunk(text: str, threshold: float = REFERENCE_DENSITY_THRESHOLD) -> bool:
    """True if this chunk looks like bibliography rather than content."""
    return citation_density(text) >= threshold


def build_page_char_map(pages: list[str]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate pages into one string, recording (start, end, page_no) spans.

    page_no is 1-based (citation-style).
    """
    full = []
    spans = []
    cursor = 0
    for idx, ptext in enumerate(pages):
        if not ptext:
            continue
        piece = ptext + " "
        start = cursor
        full.append(piece)
        cursor += len(piece)
        spans.append((start, cursor, idx + 1))
    return "".join(full), spans


def page_for_char(char_pos: int, spans: list[tuple[int, int, int]]) -> int:
    """Which 1-based page does this character position fall in?"""
    for start, end, page_no in spans:
        if start <= char_pos < end:
            return page_no
    return spans[-1][2] if spans else 1


def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int,
    overlap: int,
    spans: list[tuple[int, int, int]],
    ref_threshold: float = REFERENCE_DENSITY_THRESHOLD,
):
    """Yield chunk dicts with token-based windows and page attribution."""
    # Encode with offset mapping so we can map tokens back to char positions.
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    n = len(input_ids)
    if n == 0:
        return

    step = max(1, chunk_size - overlap)
    idx = 0
    chunk_index = 0
    while idx < n:
        window_ids = input_ids[idx : idx + chunk_size]
        window_offsets = offsets[idx : idx + chunk_size]
        if not window_offsets:
            break
        char_start = window_offsets[0][0]
        char_end = window_offsets[-1][1]
        chunk_str = text[char_start:char_end].strip()
        if chunk_str:
            page_start = page_for_char(char_start, spans)
            page_end = page_for_char(max(char_start, char_end - 1), spans)
            is_ref = is_reference_chunk(chunk_str, ref_threshold)
            yield {
                "chunk_index": chunk_index,
                "page_start": page_start,
                "page_end": page_end,
                "is_reference": is_ref,
                "n_tokens": len(window_ids),
                "text": chunk_str,
            }
            chunk_index += 1
        if idx + chunk_size >= n:
            break
        idx += step


def main():
    ap = argparse.ArgumentParser(description="Parse + chunk PDFs into chunks.jsonl")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--tokenizer", default="BAAI/bge-base-en-v1.5",
                    help="HF tokenizer id (should match the embedding model).")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--ref-threshold", type=float,
                    default=REFERENCE_DENSITY_THRESHOLD,
                    help="Citation-density threshold above which a chunk is "
                         "tagged as bibliography.")
    args = ap.parse_args()

    metadata_path = args.data_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"No metadata at {metadata_path}. Run download_papers.py first.",
              file=sys.stderr)
        sys.exit(1)

    records = json.loads(metadata_path.read_text())
    print(f"Loaded {len(records)} paper record(s).")
    print(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:  # noqa: BLE001
        print(
            f"\nCould not load tokenizer '{args.tokenizer}'.\n"
            f"  Reason: {e}\n"
            f"  This usually means no internet on first run (it needs to\n"
            f"  download once from HuggingFace) or a typo in the name.\n"
            f"  Fix: run once with internet so it caches, or pass a\n"
            f"  locally-cached tokenizer via --tokenizer.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = args.data_dir / "chunks.jsonl"
    total_chunks = 0
    total_ref_chunks = 0
    papers_done = 0

    with out_path.open("w") as out_f:
        for rec in records:
            pdf_path = Path(rec.get("pdf_path", ""))
            if not pdf_path.exists():
                print(f"  ! missing PDF for {rec['arxiv_id']}, skipping", file=sys.stderr)
                continue

            try:
                pages = extract_pages(pdf_path)
            except Exception as e:  # noqa: BLE001
                print(f"  ! failed to parse {rec['arxiv_id']}: {e}", file=sys.stderr)
                continue

            full_text, spans = build_page_char_map(pages)
            if not full_text.strip():
                print(f"  ! no extractable text in {rec['arxiv_id']} "
                      f"(scanned PDF?), skipping", file=sys.stderr)
                continue

            n_here = 0
            n_ref_here = 0
            for chunk in chunk_text(
                full_text, tokenizer, args.chunk_size, args.overlap,
                spans, args.ref_threshold,
            ):
                chunk_id = f"{rec['arxiv_id']}__{chunk['chunk_index']:04d}"
                row = {
                    "chunk_id": chunk_id,
                    "arxiv_id": rec["arxiv_id"],
                    "title": rec.get("title", ""),
                    "year": rec.get("year", ""),
                    **chunk,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_here += 1
                if chunk["is_reference"]:
                    n_ref_here += 1

            total_chunks += n_here
            total_ref_chunks += n_ref_here
            papers_done += 1
            ref_note = f", {n_ref_here} tagged reference" if n_ref_here else ""
            print(f"  = {rec['arxiv_id']}: {len(pages)} pages -> {n_here} chunks{ref_note}")

    print("\n" + "=" * 50)
    print(f"Papers processed : {papers_done}/{len(records)}")
    print(f"Total chunks     : {total_chunks}")
    print(f"  content chunks : {total_chunks - total_ref_chunks}")
    print(f"  reference chunks: {total_ref_chunks} (tagged is_reference=true)")
    print(f"Output           : {out_path}")


if __name__ == "__main__":
    main()