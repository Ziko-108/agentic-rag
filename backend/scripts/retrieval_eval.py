"""Measure retrieval alone (no LLM): does the top-k contain the golden evidence?

    python scripts/retrieval_eval.py            (from backend/, DB running)

A question counts as found when every evidence group has a retrieved chunk on
one of its pages (±1, like eval.py). A question about several companies gets
one search per company, as the agent does. Only the question text is used to
pick documents. Results go to ../starter/runs/retrieval.json.
"""

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import CrossEncoder
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Document, engine
from app.ingest.pipeline import embed
from app.retrieval.companies import detect_documents
from app.retrieval.search import dense_ids, lexical_ids, load_hits, rrf

ROOT = Path(__file__).resolve().parents[2] / "starter"
KS = (5, 8, 12)
SHORTLIST = 30
RERANKERS = {"bge-base": "BAAI/bge-reranker-base", "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2"}


def found(evidence, hits) -> bool:
    got = {(h.document, h.page) for h in hits}
    return all(any((e["document"], e["page"] + d) in got for e in g for d in (-1, 0, 1)) for g in evidence)


@lru_cache
def model(name):
    return CrossEncoder(RERANKERS[name])


def retrieve(session, question, vector, docs, method, k, reranker=None):
    """One list of hits; one search per document when several are named."""
    hits = []
    for scope in ([[d] for d in docs] if len(docs) > 1 else [docs or None]):
        dense = dense_ids(session, vector, scope) if method in ("dense", "hybrid") else []
        lex = lexical_ids(session, question, scope) if method in ("lexical", "hybrid") else []
        ids = [cid for cid, _ in rrf([r for r in (dense, lex) if r])]
        if reranker:
            by_id = load_hits(session, ids[:SHORTLIST])
            cand = [by_id[i] for i in ids[:SHORTLIST]]
            scores = model(reranker).predict([(question, h.content) for h in cand], batch_size=16)
            for h, s in zip(cand, scores):
                h.score = float(s)
            part = sorted(cand, key=lambda h: h.score, reverse=True)[:k]
        else:
            by_id = load_hits(session, ids[:k])
            part = [by_id[i] for i in ids[:k]]
        hits.extend(part)
    return hits


def main():
    golden = json.loads((ROOT / "data" / "golden_set.json").read_text(encoding="utf-8"))
    configs = [
        ("dense, all docs", "dense", False, None),
        ("lexical, all docs", "lexical", False, None),
        ("hybrid, all docs", "hybrid", False, None),
        ("hybrid + company filter", "hybrid", True, None),
        ("hybrid + filter + bge-base rerank", "hybrid", True, "bge-base"),
        ("hybrid + filter + minilm rerank", "hybrid", True, "minilm"),
    ]
    with Session(engine) as session:
        files = list(session.scalars(select(Document.filename).where(Document.status == "ready")))
        vectors = embed([q["question"] for q in golden])
        results, top_scores = [], {}
        for label, method, use_filter, reranker in configs:
            t = time.perf_counter()
            row = {"config": label}
            for k in KS:
                ok = []
                for q, v in zip(golden, vectors):
                    docs = detect_documents(q["question"], files) if use_filter else []
                    hits = retrieve(session, q["question"], v, docs, method, k, reranker)
                    if q["answerable"] and q["evidence"]:
                        ok.append(found(q["evidence"], hits))
                    if reranker and k == KS[0]:
                        top_scores.setdefault(reranker, []).append(
                            (q["id"], q["answerable"], round(max((h.score for h in hits), default=0), 3)))
                row[f"recall@{k}"] = round(sum(ok) / len(ok), 3)
            row["seconds"] = round(time.perf_counter() - t, 1)
            results.append(row)
            print(f"{label:<36}" + "  ".join(f"@{k}={row[f'recall@{k}']:.2f}" for k in KS) + f"  ({row['seconds']}s)")
            session.rollback()
    print("\ntop reranker score per question (id, answerable, score):")
    for name, rows in top_scores.items():
        print(f"  {name}: {rows}")
    out = ROOT / "runs" / "retrieval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"questions_with_evidence": sum(1 for q in golden if q["answerable"] and q["evidence"]),
                               "results": results, "top_reranker_scores": top_scores}, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
