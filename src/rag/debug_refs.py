"""Diagnose why reference detection fails on specific papers.

Usage:
  python debug_refs.py 1812.04606 2110.03051
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_papers import (
    build_page_char_map,
    extract_pages,
    find_reference_char_start,
)


def main():
    ids = sys.argv[1:]
    if not ids:
        print("usage: python debug_refs.py <arxiv_id> [<arxiv_id> ...]")
        sys.exit(1)

    records = {r["arxiv_id"]: r for r in json.loads(Path("data/metadata.json").read_text())}

    for aid in ids:
        rec = records.get(aid)
        if not rec:
            print(f"\n{aid}: not in metadata.json")
            continue
        print("\n" + "=" * 70)
        print(f"{aid} - {rec['title'][:60]}")
        print("=" * 70)

        pages = extract_pages(Path(rec["pdf_path"]))
        full, spans = build_page_char_map(pages)
        print(f"pages: {len(pages)}, full text: {len(full)} chars")

        detected = find_reference_char_start(full)
        print(f"current detector says: {detected}")
        if detected is not None:
            print(f"  context: ...{full[max(0,detected-100):detected+200]}...")

        # Where does the word actually appear?
        print("\nAll occurrences of references/bibliography (case-insensitive):")
        for m in re.finditer(r"references|bibliography|literature cited", full, re.I):
            pos = m.start()
            frac = pos / len(full)
            snippet = " ".join(full[pos:pos + 150].split())
            print(f"  @ {pos} ({frac:.1%} through doc): {snippet[:120]}")

        # What does the end of the document look like? (bibliography usually lives there)
        print("\nLast 600 chars of document:")
        print("  " + " ".join(full[-600:].split()))


if __name__ == "__main__":
    main()