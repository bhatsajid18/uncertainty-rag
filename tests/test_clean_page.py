"""Tests for page cleaning: page numbers go, integer table cells stay.

The old cleaner dropped every line that was a bare integer, which deleted
whole-number table columns. Sensoy et al. (1806.01768) Table 1 lost its CIFAR5
column (76, 84, 83, ...) and the model then reported the MNIST column as CIFAR5.
"""

import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rag"))
from chunk_papers import clean_page_text, extract_pages  # noqa: E402

TABLE_PAGE = "\n".join([
    "Method", "MNIST", "CIFAR5",
    "L2", "99.4", "76",
    "Dropout", "99.5", "84",
    "EDL", "99.3", "83",
    "Table 1: Test accuracies (%) for MNIST and CIFAR5 datasets.",
])


def test_integer_table_cells_are_kept():
    out = clean_page_text(TABLE_PAGE)
    assert "L2 99.4 76 Dropout 99.5 84 EDL 99.3 83" in out


def test_page_number_in_footer_is_dropped():
    out = clean_page_text(TABLE_PAGE + "\n7\n")
    assert out.endswith("CIFAR5 datasets.")
    assert "83" in out  # the last table cell survives, only the footer went


def test_page_number_in_header_is_dropped():
    out = clean_page_text("  12  \nSome body text here.")
    assert out == "Some body text here."


def test_arxiv_stamp_still_removed():
    out = clean_page_text("arXiv:1806.01768v3 [cs.LG] 31 Oct 2018\nBody text.")
    assert out == "Body text."


def test_real_pdf_keeps_table_column(tmp_path):
    """End to end through PyMuPDF: table cells on their own lines plus a footer."""
    pdf = tmp_path / "table.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    rows = [("Method", "MNIST", "CIFAR5"), ("L2", "99.4", "76"),
            ("Dropout", "99.5", "84"), ("EDL", "99.3", "83")]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((72 + 120 * c, 100 + 20 * r), cell)
    page.insert_text((300, 800), "7")  # page number in the footer
    doc.save(pdf)
    doc.close()

    text = extract_pages(pdf)[0]
    for value in ("76", "84", "83"):
        assert value in text.split()
    assert not text.endswith(" 7")
