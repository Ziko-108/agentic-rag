"""Rebuild the search indexes from the database alone, without parsing a PDF.

Postgres is the source of truth: every chunk's text is stored, so the vector
index can always be rebuilt from it.

    python scripts/reindex.py           # rebuild the HNSW and full-text indexes (free)
    python scripts/reindex.py --embed   # also recompute every embedding from the stored
                                        # text, e.g. after changing EMBEDDING_MODEL (~$0.03)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, text, update  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db import Chunk, engine  # noqa: E402
from app.ingest.pipeline import embed  # noqa: E402

BATCH = 500


def reembed() -> None:
    with Session(engine) as session:
        ids = list(session.scalars(select(Chunk.id).order_by(Chunk.id)))
        for i in range(0, len(ids), BATCH):
            rows = session.execute(select(Chunk.id, Chunk.content).where(Chunk.id.in_(ids[i : i + BATCH]))).all()
            vectors = embed([content for _, content in rows])
            for (chunk_id, _), vector in zip(rows, vectors):
                session.execute(update(Chunk).where(Chunk.id == chunk_id).values(embedding=vector))
            session.commit()
            print(f"embedded {min(i + BATCH, len(ids))}/{len(ids)} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embed", action="store_true", help="recompute embeddings from the stored chunk text")
    args = parser.parse_args()

    if args.embed:
        reembed()
    start = time.time()
    with engine.begin() as conn:
        conn.execute(text("REINDEX INDEX chunks_embedding_idx"))
        conn.execute(text("REINDEX INDEX chunks_tsv_idx"))
        n = conn.scalar(text("SELECT count(*) FROM chunks"))
    print(f"rebuilt the vector and full-text indexes over {n} chunks in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
