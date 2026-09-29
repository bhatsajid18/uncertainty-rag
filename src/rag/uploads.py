"""
Turn user-supplied PDFs into chunks, using the same pipeline as the corpus.

The arXiv corpus goes download -> chunk -> index on disk. An uploaded PDF has
no arXiv metadata and must not be written into the shared corpus, so this
module chunks it in memory with the SAME cleaning, tokenizer-aligned windows
and bibliography classification (reusing chunk_papers.py), and hands the
chunks to HybridRetriever.from_chunks().

Uploaded chunks look like corpus chunks except:
  arxiv_id  "upload:<file name>"   so citations and the per-paper diversity cap
                                   treat each uploaded file as its own paper
  title     from the PDF's metadata, else its first text line, else file name
  year      "" (unknown)

Size limits exist because this will sit behind a web upload in Iteration 5:
embedding runs on CPU there, and a 300-page book would stall a request.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from chunk_papers import (
    REFERENCE_DENSITY_THRESHOLD,
    build_page_char_map,
    chunk_text,
    extract_pages,
)

MAX_PAGES = 80
MAX_MB = 30


class UploadError(ValueError):
    """A PDF that can't be used, with a message fit to show the user."""


def pdf_title(pdf_path: Path) -> str:
    """Best-effort human title for citations."""
    with pymupdf.open(pdf_path) as doc:
        meta_title = (doc.metadata or {}).get("title", "") or ""
        first_page = doc[0].get_text("text") if doc.page_count else ""
    meta_title = meta_title.strip()
    junk = ("untitled", "microsoft word", ".dvi", ".tex", "arxiv")
    if len(meta_title) > 3 and not any(j in meta_title.lower() for j in junk):
        return meta_title[:150]
    for line in first_page.splitlines():
        line = line.strip()
        if len(line) > 8 and not re.fullmatch(r"[\d\W]+", line) \
                and "arxiv" not in line.lower():
            return line[:150]
    return pdf_path.stem


def check_pdf(pdf_path: Path, max_pages: int = MAX_PAGES, max_mb: float = MAX_MB):
    if not pdf_path.exists():
        raise UploadError(f"{pdf_path}: file not found")
    if pdf_path.suffix.lower() != ".pdf":
        raise UploadError(f"{pdf_path.name}: not a .pdf file")
    size_mb = pdf_path.stat().st_size / 1e6
    if size_mb > max_mb:
        raise UploadError(f"{pdf_path.name}: {size_mb:.0f} MB exceeds the {max_mb} MB limit")
    try:
        with pymupdf.open(pdf_path) as doc:
            pages = doc.page_count
    except Exception as e:  # noqa: BLE001 - corrupt or encrypted files
        raise UploadError(f"{pdf_path.name}: could not open ({e})") from e
    if pages > max_pages:
        raise UploadError(f"{pdf_path.name}: {pages} pages exceeds the {max_pages}-page limit")


def chunk_pdf(pdf_path: Path, tokenizer, chunk_size: int = 512, overlap: int = 50,
              ref_threshold: float = REFERENCE_DENSITY_THRESHOLD,
              doc_id: str | None = None) -> list[dict]:
    """Chunks for one PDF, in the corpus chunk format."""
    pages = extract_pages(pdf_path)
    full_text, spans = build_page_char_map(pages)
    if not full_text.strip():
        raise UploadError(f"{pdf_path.name}: no extractable text "
                          "(probably a scanned PDF; it needs OCR first)")
    doc_id = doc_id or f"upload:{pdf_path.name}"
    title = pdf_title(pdf_path)
    return [
        {"chunk_id": f"{doc_id}__{c['chunk_index']:04d}", "arxiv_id": doc_id,
         "title": title, "year": "", **c}
        for c in chunk_text(full_text, tokenizer, chunk_size, overlap, spans,
                            ref_threshold)
    ]


def load_uploads(pdf_paths: list[Path], tokenizer, chunk_size: int = 512,
                 overlap: int = 50) -> list[dict]:
    """Chunk several PDFs. Validates all of them before doing any work."""
    for p in pdf_paths:
        check_pdf(p)
    chunks, seen = [], set()
    for p in pdf_paths:
        doc_id = f"upload:{p.name}"
        n = 2
        while doc_id in seen:           # two files with the same name
            doc_id = f"upload:{p.stem}-{n}{p.suffix}"
            n += 1
        seen.add(doc_id)
        chunks.extend(chunk_pdf(p, tokenizer, chunk_size, overlap, doc_id=doc_id))
    return chunks
