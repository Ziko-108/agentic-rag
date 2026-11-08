"""Pages of blocks -> chunks ready to embed.

Rules (sizes are starting values, tuned later with eval.py):
- a chunk never crosses a page, so every chunk has exactly one page to cite;
- text is packed paragraph by paragraph up to `max_tokens`, and the next chunk
  repeats the last ~`overlap` tokens so a fact on the border is not cut in two;
- a heading starts a new chunk (unless the current one is still tiny);
- each table is its own chunk; a big table is split by rows and every part
  repeats the header row, so numbers never lose their column names;
- every chunk starts with "[Company | p.N | Section]" so that both the
  embedding and the LLM know where a bare number like "406.1" comes from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from app.ingest.parser import Block, Page

MAX_TOKENS = 500
OVERLAP = 60
MIN_TOKENS = 80  # below this a heading does not close the chunk

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    page: int
    kind: str  # "text" | "table"
    section: str | None
    text: str  # without the header
    content: str  # header + text: what is embedded, searched and shown to the LLM


def default_counter() -> Callable[[str], int]:
    import tiktoken  # imported lazily: tests use a word counter instead

    enc = tiktoken.get_encoding("cl100k_base")  # the encoding of text-embedding-3-*
    return lambda s: len(enc.encode(s))


def chunk_pages(
    pages: list[Page],
    company: str,
    count: Callable[[str], int] | None = None,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP,
) -> list[Chunk]:
    count = count or default_counter()
    chunks: list[Chunk] = []
    for page in pages:
        for kind, section, text in _page_pieces(page, count, max_tokens, overlap):
            header = f"[{company} | p.{page.number}" + (f" | {section}]" if section else "]")
            chunks.append(Chunk(page.number, kind, section, text, f"{header}\n{text}"))
    return chunks


def _page_pieces(page: Page, count, max_tokens: int, overlap: int):
    """Yield (kind, section, text) for one page."""
    current: list[str] = []  # paragraphs of the chunk being built
    section = None  # section of the first paragraph in `current`
    tokens = 0
    fresh = False  # does `current` hold anything beyond the overlap tail?

    def emit():
        nonlocal current, tokens, fresh
        text = "\n\n".join(current)
        current = overlap_tail(current, count, overlap)
        tokens, fresh = sum(count(p) for p in current), False
        return ("text", section, text)

    for block in page.blocks:
        if block.kind == "table":
            for part in split_table(block.text, count, max_tokens):
                yield ("table", block.section, part)
            continue
        if block.kind == "heading" and fresh and tokens >= MIN_TOKENS:
            yield emit()
            current, tokens = [], 0  # a new section does not need the old tail
        for piece in split_long(block.text, count, max_tokens):
            n = count(piece)
            if fresh and tokens + n > max_tokens:
                yield emit()
            if tokens + n > max_tokens:  # the tail plus this piece is still too big
                current, tokens = [], 0
            if not current:
                section = block.section
            current.append(piece)
            tokens += n
            fresh = True

    if fresh:  # a tail alone was already part of the previous chunk
        yield ("text", section, "\n\n".join(current))


def overlap_tail(paragraphs: list[str], count, overlap: int) -> list[str]:
    """The last paragraphs whose total size stays within `overlap` tokens."""
    tail, total = [], 0
    for p in reversed(paragraphs):
        n = count(p)
        if total + n > overlap:
            break
        tail.insert(0, p)
        total += n
    return tail


def split_long(text: str, count, max_tokens: int) -> list[str]:
    """Split a paragraph bigger than `max_tokens` at sentence ends (or words)."""
    if count(text) <= max_tokens:
        return [text]
    units = SENTENCE_END.split(text)
    if len(units) == 1:  # one giant "sentence": fall back to words
        units = text.split()
    parts, current = [], []
    for unit in units:
        if current and count(" ".join(current + [unit])) > max_tokens:
            parts.append(" ".join(current))
            current = []
        current.append(unit)
    if current:
        parts.append(" ".join(current))
    return parts


def split_table(table: str, count, max_tokens: int) -> list[str]:
    """Split a markdown table by rows; every part keeps the header row."""
    if count(table) <= max_tokens:
        return [table]
    header, *rows = table.split("\n")
    parts, current = [], [header]
    for row in rows:
        if len(current) > 1 and count("\n".join(current + [row])) > max_tokens:
            parts.append("\n".join(current))
            current = [header]
        current.append(row)
    if len(current) > 1:
        parts.append("\n".join(current))
    return parts
