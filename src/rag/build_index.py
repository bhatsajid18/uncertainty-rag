"""
Embed chunks and build a FAISS index for retrieval.

Reads data/chunks.jsonl, embeds each chunk's text with a sentence-transformers
model (default BAAI/bge-base-en-v1.5), and builds a FAISS inner-product index
over L2-normalized embeddings (inner product on normalized vectors == cosine).

Artifacts written:
  data/index.faiss        the FAISS index (vectors only)
  data/chunk_meta.json     ordered list of chunk metadata; row i corresponds to
                           vector i in the index, so a search hit maps back to
                           its paper / page / text.
  data/index_config.json   records the model + settings used to build the index,
                           so search can rebuild the same embedding at query time.

Usage:
  # build the index
  python build_index.py build

  # query it from the CLI
  python build_index.py search "how does evidential deep learning quantify uncertainty?"
  python build_index.py search "AUROC of deep ensembles on CIFAR-10" -k 5
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
# bge retrieval models expect this instruction on the QUERY side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_chunks(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build(args):
    chunks_path = args.data_dir / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"No chunks at {chunks_path}. Run chunk_papers.py first.", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks.")

    # LLM table notes (table_notes.py), if they have been generated
    from table_notes import attach_notes, load_notes
    notes = {} if args.no_table_notes else load_notes(args.data_dir)
    if notes:
        stats = attach_notes(chunks, notes)
        print(f"Table notes: {stats.get('described', 0)} chunks get a table "
              f"description, {stats.get('rebuilt', 0)} a rebuilt table"
              + (f"; {stats['stale']} stale note(s) ignored (re-run "
                 "table_notes.py)" if stats.get("stale") else "") + ".")

    # Tables rebuilt from the PDF's geometry (table_extract.py) take precedence
    # over the LLM's version of the same table: they cannot invent a value.
    from table_extract import attach_tables, load_tables
    tables = {} if args.no_tables else load_tables(args.data_dir)
    if tables:
        stats = attach_tables(chunks, tables)
        print(f"Rebuilt tables: {stats['tables']} from the PDFs, attached to "
              f"{stats['attached']} chunk(s)"
              + (f"; {stats['unplaced']} could not be matched to a chunk"
                 if stats["unplaced"] else "") + ".")

    # Figure captions and descriptions (figures.py)
    from figures import attach_figures, load_figures
    figures = {} if args.no_figures else load_figures(args.data_dir)
    if figures:
        stats = attach_figures(chunks, figures)
        print(f"Figures: {stats['figures']} found, {stats['attached']} attached "
              f"to chunks, {stats['described']} with a description.")

    device = pick_device()
    print(f"Loading model {args.model} on {device} ...")
    model = SentenceTransformer(args.model, device=device)

    # Embed each chunk with its paper title prepended (see retrieve.index_text):
    # table and results chunks rarely name their own paper otherwise. The stored
    # chunk text is unchanged, so evaluation labels stay valid.
    from retrieve import index_text
    contextual = not args.no_title
    texts = [index_text(c, contextual) for c in chunks]
    print("Embedding chunks (documents, no prefix"
          + (", paper title prepended" if contextual else "") + ") ...")
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product == cosine similarity
    ).astype("float32")

    dim = embeddings.shape[1]
    print(f"Embeddings: {embeddings.shape} (dim={dim})")

    index = faiss.IndexFlatIP(dim)  # exact inner-product search
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # persist everything
    faiss.write_index(index, str(args.data_dir / "index.faiss"))

    # metadata rows aligned to vector positions (drop the big text? no - keep it,
    # it's small enough and makes search self-contained)
    from retrieve import meta_row
    meta = [meta_row(c) for c in chunks]
    (args.data_dir / "chunk_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False)
    )
    (args.data_dir / "index_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "dim": dim,
                "normalized": True,
                "metric": "inner_product",
                "query_prefix": BGE_QUERY_PREFIX,
                "contextual_header": contextual,
                "table_notes": sum(1 for m in meta if m.get("table_note")),
                "table_grids": sum(1 for m in meta if m.get("table_markdown")),
                "figure_notes": sum(1 for m in meta if m.get("figure_note")),
                "chunking": (json.loads((args.data_dir / "chunk_config.json").read_text())
                             if (args.data_dir / "chunk_config.json").exists()
                             else None),
                "n_vectors": index.ntotal,
            },
            indent=2,
        )
    )
    print("Saved: index.faiss, chunk_meta.json, index_config.json")


class Retriever:
    """Loads a prebuilt index and answers search queries."""

    def __init__(self, data_dir: Path):
        cfg = json.loads((data_dir / "index_config.json").read_text())
        self.model_name = cfg["model"]
        self.query_prefix = cfg.get("query_prefix", "")
        self.index = faiss.read_index(str(data_dir / "index.faiss"))
        self.meta = json.loads((data_dir / "chunk_meta.json").read_text())
        self.model = SentenceTransformer(self.model_name, device=pick_device())

    def search(self, query: str, k: int = 5, include_refs: bool = False):
        q = self.query_prefix + query
        qemb = self.model.encode(
            [q], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        # over-fetch so we can drop reference chunks if requested and still fill k
        fetch = k if include_refs else min(len(self.meta), k * 4)
        scores, idxs = self.index.search(qemb, fetch)
        hits = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            m = self.meta[i]
            if not include_refs and m["is_reference"]:
                continue
            hits.append({"score": float(score), **m})
            if len(hits) >= k:
                break
        return hits


def search_cli(args):
    retr = Retriever(args.data_dir)
    hits = retr.search(args.query, k=args.k, include_refs=args.include_refs)
    print(f'\nQuery: "{args.query}"\n' + "=" * 60)
    for rank, h in enumerate(hits, 1):
        loc = (
            f"p.{h['page_start']}"
            if h["page_start"] == h["page_end"]
            else f"pp.{h['page_start']}-{h['page_end']}"
        )
        print(f"\n[{rank}] score={h['score']:.3f}  {h['arxiv_id']} ({h['year']}) {loc}")
        print(f"    {h['title'][:70]}")
        snippet = " ".join(h["text"].split())[:300]
        print(f"    {snippet}...")


def main():
    ap = argparse.ArgumentParser(description="Build/query a FAISS index over chunks.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Embed chunks and build the index.")
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--no-title", action="store_true",
                   help="Embed chunk text alone, without the paper title "
                        "(the pre-Iteration-3 behaviour; useful as an ablation).")
    b.add_argument("--batch-size", type=int, default=32)
    b.add_argument("--no-table-notes", action="store_true",
                   help="Ignore data/table_notes.json (for the with/without "
                        "comparison).")
    b.add_argument("--no-tables", action="store_true",
                   help="Ignore data/tables.json (geometric table rebuilds).")
    b.add_argument("--no-figures", action="store_true",
                   help="Ignore data/figures.json (figure captions).")
    b.set_defaults(func=build)

    s = sub.add_parser("search", help="Query the prebuilt index.")
    s.add_argument("query", type=str)
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--include-refs", action="store_true",
                   help="Include reference/bibliography chunks in results.")
    s.set_defaults(func=search_cli)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
