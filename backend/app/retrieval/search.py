"""Hybrid search: dense (pgvector) + lexical (Postgres full-text), fused with
RRF, then reranked by a local cross-encoder and cut by a relevance threshold.

Why each step:
- dense finds paraphrases ("staff" ~ "employees"), lexical finds exact terms
  and numbers ("FLWS", "R&D"); each misses things the other finds;
- RRF fuses by rank, so the two very different score scales never need tuning;
- the cross-encoder reads question and chunk together, which ranks far better
  than comparing two separate vectors, but it is too slow for all 5,768 chunks,
  so it only sees the fused shortlist;
- the threshold lets a search return nothing. It is on the embedding cosine
  similarity, not on the reranker score: measured on this corpus, on-topic
  queries (even 2-word ones) reach >= 0.42 and off-topic ones <= 0.25, while
  reranker scores are not calibrated (0.006 for a correct rank-1 hit on a short
  query, 0.35 for "weather in Paris"). See EVAL.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ingest.pipeline import embed
from app.tracing import NOOP, Span

CANDIDATES = 40  # per search method, before fusion
RRF_K = 60  # the usual constant from the RRF paper


@dataclass
class Hit:
    chunk_id: int
    document: str  # PDF file name, as citations need it
    page: int
    section: str | None
    content: str
    similarity: float = 0.0  # cosine similarity to the query embedding
    score: float = 0.0  # RRF score, replaced by the reranker score when reranked


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: ids ordered by sum of 1 / (k + rank) over the lists."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _doc_filter(documents: list[str] | None) -> tuple[str, dict]:
    if not documents:
        return "", {}
    return "AND d.filename = ANY(:docs)", {"docs": documents}


def dense_ids(session: Session, vector: list[float], documents=None, n: int = CANDIDATES) -> list[int]:
    where, params = _doc_filter(documents)
    session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))  # keep n rows under a filter
    rows = session.execute(text(f"""
        SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE d.status = 'ready' {where}
        ORDER BY c.embedding <=> CAST(:v AS vector) LIMIT :n
    """), {"v": str(vector), "n": n, **params})
    return [r[0] for r in rows]


def lexical_ids(session: Session, query: str, documents=None, n: int = CANDIDATES) -> list[int]:
    where, params = _doc_filter(documents)
    # plainto_tsquery joins words with AND; turning & into | ranks chunks that
    # have *some* of the words instead of only those that have all of them
    rows = session.execute(text(f"""
        WITH q AS (SELECT NULLIF(replace(plainto_tsquery('english', :q)::text, '&', '|'), '')::tsquery AS q)
        SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id, q
        WHERE d.status = 'ready' AND c.tsv @@ q.q {where}
        ORDER BY ts_rank_cd(c.tsv, q.q) DESC LIMIT :n
    """), {"q": query, "n": n, **params})
    return [r[0] for r in rows]


def load_hits(session: Session, ids: list[int], vector: list[float] | None = None) -> dict[int, Hit]:
    sim = "1 - (c.embedding <=> CAST(:v AS vector))" if vector else "0.0"
    rows = session.execute(text(f"""
        SELECT c.id, d.filename, c.page, c.section, c.content, {sim}
        FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ANY(:ids)
    """), {"ids": ids, **({"v": str(vector)} if vector else {})})
    return {r[0]: Hit(*r) for r in rows}


@lru_cache
def _reranker():
    from sentence_transformers import CrossEncoder  # heavy import: only when first needed

    return CrossEncoder(get_settings().reranker_model)  # uses the GPU when torch sees one


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    if not hits:
        return hits
    scores = _reranker().predict([(query, h.content) for h in hits], batch_size=16)
    for h, s in zip(hits, scores):
        h.score = float(s)
    return sorted(hits, key=lambda h: h.score, reverse=True)


def search(
    session: Session,
    query: str,
    k: int | None = None,
    documents: list[str] | None = None,
    use_reranker: bool | None = None,
    threshold: float | None = None,
    trace: Span = NOOP,
) -> list[Hit]:
    s = get_settings()
    k = k or s.top_k
    use_reranker = s.use_reranker if use_reranker is None else use_reranker
    threshold = s.relevance_threshold if threshold is None else threshold

    span = trace.child("retrieval", as_type="retriever", input={"query": query, "documents": documents},
                       metadata={"top_k": k, "reranker": use_reranker, "threshold": threshold})
    vector = embed([query])[0]
    dense, lexical = dense_ids(session, vector, documents), lexical_ids(session, query, documents)
    fused = rrf([dense, lexical])
    by_id = load_hits(session, [cid for cid, _ in fused], vector)
    hits = []
    for cid, score in fused:
        if by_id[cid].similarity >= threshold:  # too far from the question: not evidence
            by_id[cid].score = score
            hits.append(by_id[cid])
    passed = len(hits)
    hits = hits[: s.rerank_candidates if use_reranker else k]
    if use_reranker:
        hits = rerank(query, hits)
    hits = hits[:k]
    span.end(output=[{"document": h.document, "page": h.page, "similarity": round(h.similarity, 3),
                      "score": round(h.score, 4)} for h in hits],
             metadata={"dense": len(dense), "lexical": len(lexical), "fused": len(fused),
                       "above_threshold": passed, "returned": len(hits)})
    return hits
