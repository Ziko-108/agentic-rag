"""Which reports a question is about, from the company names in it.

The reports are named after their company ("Origin Bancorp, Inc..pdf"), and
questions name the company ("For Origin Bancorp, Inc., what was ..."). Legal
suffixes are dropped so "Origin Bancorp" alone still matches.
"""

import re

SUFFIXES = {"inc", "corp", "corporation", "plc", "ag", "group", "co", "ltd", "llc", "sa", "spa", "nv", "se"}


def _norm(s: str) -> str:
    s = re.sub(r"\.pdf$", "", s.strip(), flags=re.IGNORECASE).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def core_name(filename: str) -> str:
    """'Origin Bancorp, Inc..pdf' -> 'origin bancorp'; '1-800-FLOWERS.COM, INC..pdf' -> '1 800 flowers com'."""
    words = _norm(filename).split()
    while len(words) > 1 and words[-1] in SUFFIXES:
        words.pop()
    return " ".join(words)


def detect_documents(question: str, filenames: list[str]) -> list[str]:
    """File names whose company is named in the question, in file-name order."""
    q = f" {_norm(question)} "
    return [f for f in sorted(filenames) if f" {core_name(f)} " in q]
