"""PDF -> pages of blocks (paragraphs, headings, tables).

What we learned by looking at the reports (see README, "Parsing"):
- every page has a text layer, so no OCR is needed;
- PyMuPDF's reading order handles the two-column pages;
- plain text destroys tables (one cell per line), but `find_tables()` recovers
  rows and columns, so tables are extracted separately and their area is
  removed from the text;
- `find_tables` runs with `use_layout=False`. PyMuPDF 2's layout model (an
  ONNX network) segfaulted on "Origin Bancorp" p.17, was ~7x slower, kept
  several cores busy, and garbled numbers in borderless tables ("-3.3 %" came
  out as "33 - ."). Without it such tables stay as text, where numbers are intact;
- the page number printed on a page differs from its position in the PDF, so
  we only ever use the PDF position (1-based) for citations.

The PyMuPDF-specific code is in `parse_pdf`. The small pure helpers below it
take plain Python values and are unit-tested without a PDF.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

# a cell holding only a currency sign or unit belongs to the number next to it
FILLER_CELL = re.compile(r"^[$€£¥%]?$")


@dataclass
class Block:
    kind: str  # "text" | "heading" | "table"
    text: str
    section: str | None = None  # nearest heading above, carried across pages


@dataclass
class Page:
    number: int  # 1-based position in the PDF (what citations use)
    blocks: list[Block] = field(default_factory=list)


# ------------------------------------------------------------------ PDF reading


def parse_pdf(source: str | bytes) -> list[Page]:
    doc = pymupdf.open(stream=source) if isinstance(source, bytes) else pymupdf.open(source)
    raw_pages = [_read_page(p) for p in doc]  # list of (lines, tables) per page
    body_size = body_font_size(line for lines, _ in raw_pages for line in lines)
    boilerplate = repeated_lines([[l["text"] for l in lines] for lines, _ in raw_pages])

    pages, section = [], None
    for number, (lines, tables) in enumerate(raw_pages, start=1):
        page = Page(number)
        paragraph: list[str] = []

        def flush() -> None:
            if paragraph:
                page.blocks.append(Block("text", " ".join(paragraph), section))
                paragraph.clear()

        last_block = None
        for line in lines:
            if line_shape(line["text"]) in boilerplate:
                continue
            if is_heading(line, body_size):
                flush()
                section = line["text"]
                page.blocks.append(Block("heading", line["text"], section))
            else:
                if line["block"] != last_block:  # PyMuPDF blocks ~ paragraphs
                    flush()
                paragraph.append(line["text"])
            last_block = line["block"]
        flush()
        page.blocks.extend(Block("table", t, section) for t in tables)
        pages.append(page)
    return pages


def _read_page(page: pymupdf.Page) -> tuple[list[dict], list[str]]:
    """Lines of text outside tables (with font info), and the tables as markdown."""
    try:
        found = page.find_tables(use_layout=False).tables
    except Exception:  # a broken table never stops the page from being read
        found = []
    tables, boxes = [], []
    for t in found:
        md = table_to_markdown(t.extract())
        if md:
            tables.append(md)
            boxes.append(pymupdf.Rect(t.bbox))

    lines = []
    for b_no, block in enumerate(page.get_text("dict")["blocks"]):
        for line in block.get("lines", []):
            rect = pymupdf.Rect(line["bbox"])
            if any(rect.intersects(box) for box in boxes):
                continue  # already in a table
            spans = [s for s in line["spans"] if s["text"].strip()]
            text = " ".join(" ".join(s["text"] for s in spans).split())
            if not text:
                continue
            lines.append({
                "text": text,
                "block": b_no,
                "size": max(s["size"] for s in spans),
                "bold": all(s["flags"] & 16 for s in spans),
                "chars": len(text),
            })
    return lines, tables


# ------------------------------------------------------------------ pure helpers


def body_font_size(lines) -> float:
    """The font size that most characters are written in."""
    sizes = Counter()
    for line in lines:
        sizes[round(line["size"], 1)] += line["chars"]
    return sizes.most_common(1)[0][0] if sizes else 10.0


def is_heading(line: dict, body_size: float) -> bool:
    """A short line, bigger than body text or bold, that contains words."""
    text = line["text"]
    if not (3 <= len(text) <= 100) or not re.search(r"[A-Za-z]{3}", text):
        return False
    if text.endswith((".", ",", ";")):  # a sentence, not a title
        return False
    return line["size"] >= body_size * 1.15 or (line["bold"] and line["size"] >= body_size)


def line_shape(text: str) -> str:
    """'Page | 5' and 'Page | 6' are the same footer: compare lines with digits masked."""
    return re.sub(r"\d+", "#", text)


def repeated_lines(pages: list[list[str]], share: float = 0.3, min_pages: int = 10) -> set[str]:
    """Running headers, footers and side menus: shapes (see `line_shape`) of
    lines with words that appear on at least `share` of the pages. Numbers
    alone are kept (they may be data)."""
    if len(pages) < min_pages:
        return set()
    counts = Counter(
        s for lines in pages for s in {line_shape(t) for t in lines} if re.search(r"[A-Za-z]", s)
    )
    return {s for s, n in counts.items() if n >= share * len(pages)}


def merge_columns(rows: list[list[str]]) -> list[list[str]]:
    """Merge neighbouring columns that never disagree.

    `find_tables` often splits one column in two: one with "$" and one with the
    number, or the same number twice. Two columns are merged when, on every row,
    one cell is empty/a currency sign or both cells are equal."""
    if not rows:
        return rows
    columns = [list(c) for c in zip(*rows)]
    merged = [columns[0]]
    for col in columns[1:]:
        prev = merged[-1]
        if all(FILLER_CELL.match(a) or FILLER_CELL.match(b) or a == b for a, b in zip(prev, col)):
            merged[-1] = [b if FILLER_CELL.match(a) else a for a, b in zip(prev, col)]
        else:
            merged.append(col)
    return [list(r) for r in zip(*merged)]


def table_to_markdown(raw: list[list[str | None]]) -> str:
    """Clean a table from `find_tables().extract()` into pipe-separated rows."""
    rows = [[" ".join((c or "").split()) for c in row] for row in raw]
    rows = [r for r in rows if any(r)]
    if not rows or max(len(r) for r in rows) < 2:
        return ""
    rows = merge_columns(rows)
    rows = [r for r in rows if any(r)]
    if len(rows) < 2 or len(rows[0]) < 2:  # one row or one column is not a table
        return ""
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)
