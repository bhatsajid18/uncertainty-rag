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
  python chunk_papers.py --strategy sentence    # chunks end on sentence boundaries
  python chunk_papers.py --strategy semantic    # ... and at topic shifts
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
# A bare number is only a page number when it is the first or last line of
# the page. Anywhere else it is content - most often a table cell, since
# PyMuPDF emits each cell on its own line. Dropping every bare-number line
# (the original rule) silently deleted whole-number table columns: Sensoy et
# al.'s CIFAR-5 accuracies (76, 84, ..., 83) vanished while the MNIST column
# (99.4, 99.5, ...) survived only because it has decimals.
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")

_JUNK_PATTERNS = [
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
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    last = len(lines) - 1
    kept = []
    for i, stripped in enumerate(lines):
        if i in (0, last) and _PAGE_NUMBER.match(stripped):
            continue  # page number in the header or footer
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
#
# This flag is a HINT, not a verdict: retrieval demotes a flagged chunk rather
# than dropping it (see retrieve.py's ref_penalty). That matters, because no
# cheap feature separates a bibliography entry from a paragraph that cites
# heavily. Measured over this corpus, three candidates were tried and all
# three overlap:
#
#   citation density   genuine references score 0.30-0.72, but related-work
#                      and setup prose scores 0.28-0.44 - interleaved, so no
#                      threshold splits them. Raising 0.25 to 0.36 recovered
#                      3 content chunks and released 18 real reference chunks.
#   function words     references 0.048, the worst mis-tagged prose 0.055.
#   position in paper  references span 0.49-0.96 of the way through a paper;
#                      the mis-tagged prose sits at 0.50, 0.74 and 0.76.
#
# So 0.25 is kept deliberately INCLUSIVE. Over-tagging is now cheap - a
# demoted chunk can still be retrieved when it is genuinely the best match -
# whereas under-tagging lets reference lists compete at full strength.
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



# --- sentence-aware and semantic chunking -------------------------------------
#
# chunk_text() above cuts every `chunk_size` tokens, wherever that falls - often
# mid-sentence. Two alternatives, selected with --strategy:
#
#   sentence  packs whole sentences into chunks of at most chunk_size tokens,
#             with an overlap of whole sentences. A sentence longer than a chunk
#             (in practice: a flattened table) falls back to token windows. This
#             is the idea behind LangChain's RecursiveCharacterTextSplitter
#             (split on the largest natural boundary that fits), in tokens.
#   semantic  also starts a new chunk where the topic shifts: consecutive
#             sentences whose embeddings are unusually far apart (distance above
#             the given percentile of the paper's distances) mark a boundary.
#             Each sentence is embedded with its neighbours, as LangChain's
#             SemanticChunker does, so a single short sentence doesn't
#             register as a topic change. No overlap: boundaries are meant to
#             fall between topics.
#
# All three produce the same chunk records, so everything downstream is
# unchanged. Evaluation labels are chunk ids pinned to chunk text, so comparing
# strategies needs a query set labelled per strategy.

STRATEGIES = ("fixed", "sentence", "semantic")
# A full stop, then whitespace, then a capital letter. Deliberately simple:
# "Fig. 3" and "Eq. (2)" don't split, and an occasional wrong split ("et al.
# Smith") only moves a boundary within the size limit.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) character spans of the sentences in text."""
    spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def _token_units(text, tokenizer, chunk_size):
    """Sentences as units (char_start, char_end, tok_lo, tok_hi, sentence_no);
    a sentence longer than chunk_size becomes several token-window units."""
    import bisect

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False,
                    return_attention_mask=False)
    offsets = enc["offset_mapping"]
    starts = [a for a, _ in offsets]
    units = []
    for s_no, (a, b) in enumerate(sentence_spans(text)):
        lo, hi = bisect.bisect_left(starts, a), bisect.bisect_left(starts, b)
        if hi <= lo:
            continue
        for w in range(lo, hi, chunk_size):
            w_hi = min(w + chunk_size, hi)
            units.append((offsets[w][0], offsets[w_hi - 1][1], w, w_hi, s_no))
    return units


def pack_units(units, chunk_size: int, overlap: int, breaks: set[int] | None = None,
               min_tokens: int = 64):
    """Group consecutive units into chunks of at most chunk_size tokens.

    breaks: sentence numbers after which a new chunk should start (semantic
    chunking), honoured once the current chunk has min_tokens. overlap: carry
    trailing whole units totalling at most this many tokens into the next chunk.
    Yields (first_unit, last_unit) index pairs.
    """
    breaks = breaks or set()
    n, i = len(units), 0
    while i < n:
        j, size = i, 0
        while j < n:
            u_tokens = units[j][3] - units[j][2]
            if size and size + u_tokens > chunk_size:
                break
            size += u_tokens
            j += 1
            at_break = units[j - 1][4] in breaks and (j == n or units[j][4] != units[j - 1][4])
            if at_break and size >= min_tokens:
                break
        yield i, j - 1
        if j >= n:
            return
        # step back over whole units for the overlap, but always move forward
        back, carried = j, 0
        while back - 1 > i and carried + (units[back - 1][3] - units[back - 1][2]) <= overlap:
            back -= 1
            carried += units[back][3] - units[back][2]
        ended_on_break = units[j - 1][4] in breaks
        i = j if ended_on_break else back


def semantic_breaks(text: str, embed_fn, percentile: float = 90.0,
                    buffer: int = 1) -> set[int]:
    """Sentence numbers after which the topic shifts.

    embed_fn(list[str]) -> normalised vectors. Sentence i is embedded together
    with `buffer` sentences either side.
    """
    import numpy as np

    spans = sentence_spans(text)
    if len(spans) < 3:
        return set()
    sents = [text[a:b] for a, b in spans]
    windows = [" ".join(sents[max(0, i - buffer): i + buffer + 1])
               for i in range(len(sents))]
    vecs = np.asarray(embed_fn(windows), dtype="float64")
    dist = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)
    threshold = np.percentile(dist, percentile)
    return {i for i, d in enumerate(dist) if d > threshold}


def chunk_text_by_sentences(
    text: str,
    tokenizer,
    chunk_size: int,
    overlap: int,
    spans: list[tuple[int, int, int]],
    ref_threshold: float = REFERENCE_DENSITY_THRESHOLD,
    breaks: set[int] | None = None,
):
    """Like chunk_text(), but chunks end on sentence boundaries (and, with
    `breaks`, on topic shifts). Yields the same chunk dicts."""
    units = _token_units(text, tokenizer, chunk_size)
    for chunk_index, (first, last) in enumerate(pack_units(units, chunk_size, overlap,
                                                           breaks)):
        char_start, char_end = units[first][0], units[last][1]
        chunk_str = text[char_start:char_end].strip()
        yield {
            "chunk_index": chunk_index,
            "page_start": page_for_char(char_start, spans),
            "page_end": page_for_char(max(char_start, char_end - 1), spans),
            "is_reference": is_reference_chunk(chunk_str, ref_threshold),
            "n_tokens": units[last][3] - units[first][2],
            "text": chunk_str,
        }

def main():
    ap = argparse.ArgumentParser(description="Parse + chunk PDFs into chunks.jsonl")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--tokenizer", default="BAAI/bge-base-en-v1.5",
                    help="HF tokenizer id (should match the embedding model).")
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--strategy", choices=STRATEGIES, default="fixed",
                    help="fixed token windows (default), whole sentences, or "
                         "sentences split at topic shifts.")
    ap.add_argument("--semantic-percentile", type=float, default=90.0,
                    help="semantic: a sentence-to-sentence distance above this "
                         "percentile starts a new chunk.")
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

    embed_fn = None
    if args.strategy == "semantic":
        from sentence_transformers import SentenceTransformer
        print(f"Loading {args.tokenizer} to find topic shifts ...")
        embedder = SentenceTransformer(args.tokenizer)

        def embed_fn(texts):
            return embedder.encode(texts, batch_size=64, convert_to_numpy=True,
                                   normalize_embeddings=True)

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
            if args.strategy == "fixed":
                pieces = chunk_text(full_text, tokenizer, args.chunk_size,
                                    args.overlap, spans, args.ref_threshold)
            else:
                breaks = (semantic_breaks(full_text, embed_fn, args.semantic_percentile)
                          if args.strategy == "semantic" else None)
                pieces = chunk_text_by_sentences(
                    full_text, tokenizer, args.chunk_size,
                    0 if args.strategy == "semantic" else args.overlap,
                    spans, args.ref_threshold, breaks)
            for chunk in pieces:
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
    print(f"Strategy         : {args.strategy}")
    print(f"Output           : {out_path}")
    (args.data_dir / "chunk_config.json").write_text(json.dumps({
        "strategy": args.strategy, "chunk_size": args.chunk_size,
        "overlap": 0 if args.strategy == "semantic" else args.overlap,
        "tokenizer": args.tokenizer,
        **({"semantic_percentile": args.semantic_percentile}
           if args.strategy == "semantic" else {}),
    }, indent=2))


if __name__ == "__main__":
    main()
