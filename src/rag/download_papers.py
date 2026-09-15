"""arXiv paper downloader (metadata via arxiv lib, PDF via requests)."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import arxiv
import requests

HEADERS = {"User-Agent": "uncertainty-rag/0.1 (academic project)"}


def normalize_arxiv_id(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    s = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", s)
    s = re.sub(r"\.pdf$", "", s)
    return s


def load_ids_from_file(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(normalize_arxiv_id(line))
    return ids


def id_to_filename(arxiv_id: str) -> str:
    base = re.sub(r"v\d+$", "", arxiv_id)
    return base.replace("/", "_")


def result_to_record(result):
    short_id = result.get_short_id()
    bare_id = re.sub(r"v\d+$", "", short_id)
    published = result.published.isoformat() if result.published else ""
    return {
        "arxiv_id": bare_id,
        "version_id": short_id,
        "title": " ".join(result.title.split()),
        "authors": [a.name for a in result.authors],
        "year": str(result.published.year) if result.published else "",
        "published": published,
        "abstract": " ".join(result.summary.split()),
        "primary_category": result.primary_category or "",
        "categories": list(result.categories),
        "abs_url": result.entry_id,
        "pdf_url": result.pdf_url,
    }


def download_pdf(pdf_url: str, dest: Path) -> bool:
    try:
        resp = requests.get(pdf_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    request error: {e}", file=sys.stderr)
        return False
    content = resp.content
    if not content.startswith(b"%PDF"):
        print(f"    not a PDF (got {content[:20]!r})", file=sys.stderr)
        return False
    dest.write_bytes(content)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download arXiv papers + metadata.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--id-file", type=Path)
    src.add_argument("--ids", nargs="+")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.id_file:
        ids = load_ids_from_file(args.id_file)
    else:
        ids = [normalize_arxiv_id(x) for x in args.ids]
    ids = [i for i in ids if i]
    seen = set()
    ids = [i for i in ids if not (i in seen or seen.add(i))]

    if not ids:
        print("No valid arXiv IDs given.", file=sys.stderr)
        sys.exit(1)

    print(f"Requested {len(ids)} paper(s): {', '.join(ids)}")

    papers_dir = args.out_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_dir / "metadata.json"

    existing = {}
    if metadata_path.exists():
        for rec in json.loads(metadata_path.read_text()):
            existing[rec["arxiv_id"]] = rec

    client = arxiv.Client(page_size=100, delay_seconds=args.delay, num_retries=5)

    print("Fetching metadata from arXiv API...")
    search = arxiv.Search(id_list=ids)
    try:
        results = list(client.results(search))
    except Exception as e:
        print(f"arXiv API request failed: {e}", file=sys.stderr)
        sys.exit(1)

    by_id = {}
    for r in results:
        by_id[re.sub(r"v\d+$", "", r.get_short_id())] = r

    requested_bare = [re.sub(r"v\d+$", "", i) for i in ids]
    missing = [i for i in requested_bare if i not in by_id]
    if missing:
        print(f"  ! no metadata returned for: {', '.join(missing)}", file=sys.stderr)

    ok, skipped, failed = [], [], []
    for bare_id, result in by_id.items():
        pdf_path = papers_dir / f"{id_to_filename(bare_id)}.pdf"
        if pdf_path.exists() and not args.force:
            print(f"  = {bare_id}: PDF already exists, skipping")
            skipped.append(bare_id)
        else:
            print(f"  > downloading {bare_id} ...")
            if download_pdf(result.pdf_url, pdf_path):
                ok.append(bare_id)
                time.sleep(args.delay)
            else:
                print(f"  ! download failed for {bare_id}", file=sys.stderr)
                failed.append(bare_id)
                continue
        rec = result_to_record(result)
        rec["pdf_path"] = str(pdf_path)
        existing[bare_id] = rec

    records = sorted(existing.values(),
                     key=lambda r: (r.get("year", ""), r.get("arxiv_id", "")))
    metadata_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    print("\n" + "=" * 50)
    print(f"Downloaded : {len(ok)}   {ok}")
    print(f"Skipped    : {len(skipped)} (already present)")
    print(f"Failed     : {len(failed)} {failed}")
    print(f"Metadata   : {len(records)} records -> {metadata_path}")
    print(f"PDFs       : {papers_dir}")


if __name__ == "__main__":
    main()
