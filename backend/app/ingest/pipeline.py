"""One document: PDF bytes -> parsed pages -> chunks -> embeddings -> Postgres.

Parsing runs in a separate process (one worker, low priority, fresh per
document) because PyMuPDF is CPU-heavy native code: a crash there only fails
that document instead of killing the API, and the machine stays usable.
A lock lets only one document parse at a time, however many are uploaded.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from openai import OpenAI
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Chunk, Document, engine
from app.ingest.chunker import chunk_pages, default_counter
from app.ingest.parser import Page, parse_pdf

log = logging.getLogger(__name__)

EMBED_BATCH = 100  # chunks per embeddings request
_parse_lock = threading.Lock()


def company_from_filename(filename: str) -> str:
    """'Atreca, Inc..pdf' -> 'Atreca, Inc.' (the reports are named after the company)."""
    return filename[:-4] if filename.lower().endswith(".pdf") else filename


def _low_priority() -> None:
    if hasattr(os, "nice"):  # Linux (the container); Windows has no nice()
        os.nice(10)


def parse_isolated(data: bytes) -> list[Page]:
    with _parse_lock, ProcessPoolExecutor(max_workers=1, initializer=_low_priority, max_tasks_per_child=1) as pool:
        return pool.submit(parse_pdf, data).result()


def embed(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = client.embeddings.create(input=texts[i : i + EMBED_BATCH], model=settings.embedding_model)
        vectors.extend(d.embedding for d in resp.data)
    return vectors


def ingest_document(document_id: int, data: bytes) -> None:
    """Run as a background task; the outcome is recorded in documents.status."""
    with Session(engine) as session:
        doc = session.get(Document, document_id)
        doc.status, doc.error = "processing", None
        session.commit()
        try:
            pages = parse_isolated(data)
            count = default_counter()
            chunks = chunk_pages(pages, doc.company, count)
            vectors = embed([c.content for c in chunks])
            session.add_all(
                Chunk(
                    document_id=doc.id, page=c.page, kind=c.kind, section=c.section,
                    content=c.content, tokens=count(c.content), embedding=v,
                )
                for c, v in zip(chunks, vectors)
            )
            doc.pages, doc.status = len(pages), "ready"
            session.commit()
            log.info("ingested %s: %d pages, %d chunks", doc.filename, len(pages), len(chunks))
        except Exception as e:  # includes a crashed parser process (BrokenProcessPool)
            session.rollback()
            doc = session.get(Document, document_id)
            doc.status, doc.error = "failed", f"{type(e).__name__}: {e}"[:1000]
            session.commit()
            log.exception("ingest failed for document %s", document_id)


def mark_interrupted() -> None:
    """At startup: a document still pending/processing lost its job in a restart."""
    with Session(engine) as session:
        for doc in session.scalars(select(Document).where(Document.status.in_(["pending", "processing"]))):
            doc.status, doc.error = "failed", "interrupted by a restart; upload it again"
        session.commit()


def seed_folder(folder: str) -> None:
    """Ingest, one by one, every PDF in `folder` that is not ready in the database."""
    paths = sorted(Path(folder).glob("*.pdf"))
    jobs = []
    with Session(engine) as session:
        known = {d.filename: d for d in session.scalars(select(Document))}
        for path in paths:
            doc = known.get(path.name)
            if doc is None:
                doc = Document(filename=path.name, company=company_from_filename(path.name))
                session.add(doc)
            elif doc.status == "ready":
                continue
            doc.status = "pending"
            jobs.append((doc, path))
        session.commit()
        jobs = [(doc.id, path) for doc, path in jobs]
    log.info("seed: %d of %d PDFs in %s to ingest", len(jobs), len(paths), folder)
    for document_id, path in jobs:
        ingest_document(document_id, path.read_bytes())
