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
import numpy as np
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

    device = pick_device()
    print(f"Loading model {args.model} on {device} ...")
    model = SentenceTransformer(args.model, device=device)

    texts = [c["text"] for c in chunks]
    print("Embedding chunks (documents, no prefix) ...")
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
    meta = [
        {
            "chunk_id": c["chunk_id"],
            "arxiv_id": c["arxiv_id"],
            "title": c["title"],
            "year": c["year"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "is_reference": c["is_reference"],
            "text": c["text"],
        }
        for c in chunks
    ]
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
    b.add_argument("--batch-size", type=int, default=32)
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