"""Ask the golden questions, score the answers, write a scorecard.

    python eval.py --label agent
    python eval.py --label pipeline --api http://localhost:8000

Change `ask()` so it calls your app. Leave the scoring below it alone: it is the
same for everyone, so scorecards can be compared. It uses no LLM and no network,
so the same answers always get the same score. Python standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
GOLDEN = HERE / "data" / "golden_set.json"
RUNS = HERE / "runs"


def ask(question: str, api: str, user: str) -> dict:
    """One question to your app. Return:

        {"answer": "...",
         "citations": [{"document": "RWE AG.pdf", "page": 47}],   # file name, 1-based page
         "usd": 0.0021}                                           # optional

    Each question should start a new conversation. The default below posts to
    `/chat` the way the reference app expects; change it to fit your API.
    """
    req = urllib.request.Request(
        f"{api}/chat",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json", "X-User-Id": user},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.load(r)
    return {
        "answer": body["answer"],
        "citations": [{"document": c.get("document") or c.get("doc"), "page": c["page"]}
                      for c in body.get("citations", [])],
        "usd": (body.get("usage") or {}).get("usd", 0.0),
    }


# ---------------------------------------------------------------- scoring: do not edit

REFUSAL = re.compile(
    r"\b(n/?a|not available|no (information|data|mention)|cannot (find|answer)"
    r"|not (mentioned|disclosed|found|stated|provided|reported|specified)"
    r"|(does|do) not (mention|disclose|contain|say|state|provide|include|report))\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion|bn|m|k)?\b", re.IGNORECASE)
SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9}
BOOLEAN = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
PAGE_TOLERANCE = 1  # a citation one page off still counts


def norm(s: str) -> str:
    """Lowercase, drop '.pdf' and punctuation: 'RWE AG.pdf' -> 'rwe ag'."""
    s = re.sub(r"\.pdf$", "", s.strip(), flags=re.IGNORECASE).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def numbers(text: str) -> list[float]:
    """Every number in the text, with 'million' and friends applied: '$406.1 million' -> 406100000."""
    return [float(m[1].replace(",", "")) * SCALE.get((m[2] or "").lower(), 1)
            for m in NUMBER.finditer(text)]


def is_correct(q: dict, answer: str) -> bool:
    if not q["answerable"]:
        return bool(REFUSAL.search(answer))
    gold, kind = q["answer"], q["kind"]
    if kind == "number":
        target = float(gold)
        return any(abs(n - target) <= max(0.01, abs(target) * 0.01) for n in numbers(answer))
    if kind == "boolean":
        m = BOOLEAN.search(answer)
        said = m[1].lower() in ("yes", "true") if m else (False if REFUSAL.search(answer) else None)
        return said == (gold == "True")
    if kind == "name":
        return norm(gold) in norm(answer)
    if kind == "names":  # positions: naming at least half of them counts
        found = sum(norm(g) in norm(answer) for g in gold)
        return found >= len(gold) / 2
    raise ValueError(f"unknown kind {kind!r}")


def evidence_cited(q: dict, citations: list[dict]) -> bool:
    """Every evidence group has at least one cited page in it. A comparison across
    three reports has three groups, so it needs a citation in each report."""
    cited = {(norm(c["document"] or ""), int(c["page"])) for c in citations}

    def hit(e: dict) -> bool:
        return any((norm(e["document"]), e["page"] + d) in cited
                   for d in range(-PAGE_TOLERANCE, PAGE_TOLERANCE + 1))

    return all(any(hit(e) for e in group) for group in q["evidence"])


def scorecard(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    with_evidence = [r for r in answerable if r["has_evidence"]]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    return {
        "answer accuracy": mean([r["correct"] for r in answerable]),
        "correct refusals on unanswerable": mean([r["correct"] for r in unanswerable]),
        "evidence cited": mean([r["evidence_cited"] for r in with_evidence]),
        "seconds per question": mean([r["seconds"] for r in rows]),
        "usd for the whole set": round(sum(r["usd"] for r in rows), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="names the run, e.g. agent or pipeline")
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()

    rows = []
    for q in json.loads(GOLDEN.read_text()):
        start = time.perf_counter()
        try:
            got = ask(q["question"], args.api.rstrip("/"), f"eval-{args.label}-{q['id']}")
        except Exception as e:  # a failed question scores zero, the run goes on
            got = {"answer": f"ERROR: {e}", "citations": []}
        seconds = time.perf_counter() - start
        row = {
            "id": q["id"], "kind": q["kind"], "answerable": q["answerable"],
            "expected": q["answer"], "answer": got["answer"], "citations": got["citations"],
            "correct": is_correct(q, got["answer"]),
            "has_evidence": bool(q["evidence"]),
            "evidence_cited": evidence_cited(q, got["citations"]),
            "seconds": round(seconds, 2), "usd": got.get("usd") or 0.0,
        }
        rows.append(row)
        print(f"{q['id']}  {'ok ' if row['correct'] else 'BAD'}  {seconds:5.1f}s  {got['answer'][:70]!r}")

    card = scorecard(rows)
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "scorecard": card, "rows": rows}, indent=2))
    print()
    for k, v in card.items():
        print(f"{k:<36}{v}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
