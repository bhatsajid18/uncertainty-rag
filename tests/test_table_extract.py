"""Tests for rebuilding tables from PDF geometry.

The PDFs are generated here (the corpus PDFs aren't in git), laid out the way
the papers lay theirs out: no ruling lines, a caption under the body, numbers
centred under their headers, and prose around them.
"""

import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rag"))
import table_extract as te  # noqa: E402

PROSE = ("We now compare the proposed method against several baselines on three "
         "datasets, reporting classification error and calibration for each. ") * 2
AFTER = ("Firstly, we investigate the ability of a single model to retain the "
         "ensemble's classification performance after distillation. ") * 2
FS = 9


def _centre(page, x_centre, y, text, fontsize=FS):
    width = pymupdf.get_text_length(text, fontsize=fontsize)
    page.insert_text((x_centre - width / 2, y), text, fontsize=fontsize)


def make_pdf(path, rows, caption, columns=(90, 220, 310, 400, 480),
             caption_below=True, header2=None):
    """A page laid out like a paper's: prose, a table with no ruling lines and
    values centred under their headers, its caption, then more prose."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(60, 60, 540, 120), PROSE, fontsize=FS)
    y = 150
    if not caption_below:
        page.insert_textbox(pymupdf.Rect(70, y, 540, y + 30), caption, fontsize=FS)
        y += 30
    body = [rows[0]] + ([header2] if header2 else []) + rows[1:]
    for row in body:
        for x, cell in zip(columns, row):
            if cell:
                _centre(page, x + 20, y, cell)
        y += 16
    if caption_below:
        page.insert_textbox(pymupdf.Rect(70, y + 6, 540, y + 40), caption, fontsize=FS)
        y += 40
    page.insert_textbox(pymupdf.Rect(60, y + 10, 540, y + 70), AFTER, fontsize=FS)
    doc.save(path)
    doc.close()


ROWS = [["Method", "MNIST", "CIFAR5", "SVHN", "TIM"],
        ["L2", "99.4", "76", "94.2", "41.8"],
        ["Dropout", "99.5 ±0.1", "84", "93.8", "40.3"],
        ["Deep Ensemble", "99.3", "79", "95.1", "36.9"],
        ["EDL", "99.3 ±0.2", "83", "94.7", "37.6"]]
CAPTION = "Table 1: Test accuracies (%) for MNIST, CIFAR5, SVHN and TinyImageNet."


@pytest.fixture
def simple_pdf(tmp_path):
    path = tmp_path / "simple.pdf"
    make_pdf(path, ROWS, CAPTION)
    return path


def test_grid_matches_the_table(simple_pdf):
    [t] = te.extract_tables(simple_pdf)
    assert (t["n_rows"], t["n_cols"]) == (4, 5)
    assert t["caption"].startswith("Table 1:")
    assert t["markdown"].splitlines()[0] == "| Method | MNIST | CIFAR5 | SVHN | TIM |"
    assert "| EDL | 99.3 ±0.2 | 83 | 94.7 | 37.6 |" in t["markdown"]
    assert "| L2 | 99.4 | 76 | 94.2 | 41.8 |" in t["markdown"]


def test_every_value_stays_in_its_own_column(simple_pdf):
    [t] = te.extract_tables(simple_pdf)
    rows = [line.strip("|").split("|") for line in t["markdown"].splitlines()]
    cifar5 = [r[2].strip() for r in rows[2:]]
    assert cifar5 == ["76", "84", "79", "83"], "the CIFAR5 column, unshifted"


def test_caption_above_the_body_also_works(tmp_path):
    path = tmp_path / "above.pdf"
    make_pdf(path, ROWS, CAPTION, caption_below=False)
    [t] = te.extract_tables(path)
    assert "| EDL | 99.3 ±0.2 | 83 | 94.7 | 37.6 |" in t["markdown"]


def test_two_level_header_is_merged(tmp_path):
    path = tmp_path / "twolevel.pdf"
    rows = [["", "C10", "C10", "C100", "C100"],
            ["IND", "8.0 ±0.4", "84.6", "30.4 ±0.3", "72.5"],
            ["ENSM", "6.2 ±0.2", "86.8", "26.3 ±0.2", "75.0"],
            ["EnD2", "7.3 ±0.2", "85.3", "27.9 ±0.3", "73.7"]]
    make_pdf(path, rows, "Table 3: Error and PRR on C10 and C100.",
             header2=["Method", "ERR", "PRR", "ERR", "PRR"])
    [t] = te.extract_tables(path)
    header = t["markdown"].splitlines()[0]
    assert "C10 ERR" in header and "C100 PRR" in header
    assert "| EnD2 | 7.3 ±0.2 | 85.3 | 27.9 ±0.3 | 73.7 |" in t["markdown"]


def test_missing_cells_stay_empty(tmp_path):
    path = tmp_path / "gappy.pdf"
    rows = [["Method", "C10", "C100", "TIM"],
            ["PN-KL", "14.7", "", ""],
            ["PN-RKL", "7.5", "28.1", "40.3"],
            ["ENSM", "6.6", "26.9", "36.9"]]
    make_pdf(path, rows, "Table 1: Mean classification error.",
             columns=(90, 240, 350, 460))
    [t] = te.extract_tables(path)
    row = next(line for line in t["markdown"].splitlines() if "PN-KL" in line)
    assert [c.strip() for c in row.strip("|").split("|")] == ["PN-KL", "14.7", "", ""]


def test_table_beside_a_figure(tmp_path):
    """Sensoy et al. p.7: Figure 2 on the left, Table 1 on the right, both
    captions and the table's rows sharing horizontal bands."""
    path = tmp_path / "beside.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(60, 60, 540, 110), PROSE, fontsize=FS)
    # the figure: a drawing with tick labels, and its caption
    page.draw_rect(pymupdf.Rect(60, 140, 260, 260), color=(0, 0, 0))
    for i, label in enumerate(("0.0", "0.2", "0.4", "0.6")):
        page.insert_text((70 + i * 45, 275), label, fontsize=7)
    page.insert_textbox(pymupdf.Rect(60, 285, 270, 330),
                        "Figure 2: The change of accuracy with respect to the "
                        "uncertainty threshold for EDL.", fontsize=FS)
    # the table, to the right of it
    rows = [["Method", "MNIST", "CIFAR5"], ["L2", "99.4", "76"],
            ["Dropout", "99.5", "84"], ["EDL", "99.3", "83"]]
    y = 150
    for row in rows:
        for x, cell in zip((300, 400, 470), row):
            _centre(page, x + 20, y, cell)
        y += 18
    page.insert_textbox(pymupdf.Rect(300, y + 8, 540, y + 50),
                        "Table 1: Test accuracies (%) for MNIST and CIFAR5 "
                        "datasets.", fontsize=FS)
    doc.save(path)
    doc.close()

    [t] = te.extract_tables(path)
    assert t["n_cols"] == 3, "the figure's tick labels are not extra columns"
    assert t["markdown"].splitlines()[0] == "| Method | MNIST | CIFAR5 |"
    assert "| EDL | 99.3 | 83 |" in t["markdown"]
    assert "0.2" not in t["markdown"]


def test_prose_pages_yield_nothing(tmp_path):
    path = tmp_path / "prose.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 60, 290, 700), PROSE * 6, fontsize=FS)
    page.insert_textbox(pymupdf.Rect(310, 60, 550, 700), AFTER * 6, fontsize=FS)
    page.insert_text((60, 720), "Table 1: this caption has no table under it.",
                     fontsize=FS)
    doc.save(path)
    doc.close()
    assert te.extract_tables(path) == []


# --- attaching to chunks ---------------------------------------------------------

GRID = ("| Method | CIFAR5 |\n| --- | --- |\n| L2 | 76 |\n| Dropout | 84 |\n"
        "| EDL | 83 |")
TABLES = {"1806.01768": [{"page": 7, "caption": "Table 1: Test accuracies",
                          "markdown": GRID, "n_rows": 3, "n_cols": 2}]}


def chunk(cid, page_start, page_end, text, arxiv="1806.01768"):
    return {"chunk_id": cid, "arxiv_id": arxiv, "page_start": page_start,
            "page_end": page_end, "text": text}


def test_table_goes_to_the_chunk_that_contains_it():
    on_page = chunk("a__0011", 6, 7, "Method MNIST CIFAR 5 L2 99.4 76 Dropout 99.5 "
                    "84 EDL 99.3 83 Table 1: Test accuracies (%)")
    elsewhere = chunk("a__0009", 7, 7, "We tested these approaches on MNIST.")
    stats = te.attach_tables([on_page, elsewhere], TABLES)
    assert on_page["table_markdown"] == GRID
    assert on_page["table_grid_source"] == "geometry"
    assert "table_markdown" not in elsewhere
    assert stats == {"attached": 1, "tables": 1, "unplaced": 0}


def test_other_papers_and_pages_are_left_alone():
    other_paper = chunk("b__0011", 7, 7, "L2 99.4 76 Dropout 99.5 84 EDL 99.3 83",
                        arxiv="1905.00076")
    stats = te.attach_tables([other_paper], TABLES)
    assert "table_markdown" not in other_paper and stats["unplaced"] == 1


def test_geometric_grid_comes_before_an_llm_grid():
    c = chunk("a__0011", 7, 7, "Table 1: Test accuracies L2 99.4 76 Dropout 99.5 84")
    c["table_markdown"] = "| Method | CIFAR5 |\n| --- | --- |\n| L2 | 99.4 |"
    te.attach_tables([c], TABLES)
    grids = c["table_markdown"].split("\n\n")
    assert grids[0] == GRID and len(grids) == 2, "LLM grid kept, but second"
