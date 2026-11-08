"""The baseline: retrieve once, answer once. The agent is compared against it.

`run` yields the same events as the agent (see app/chat/agent.py), so the API
and the UI treat both alike.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chat.llm import Usage, stream
from app.chat.prompts import REFUSAL_TEXT, SINGLE_PASS_SYSTEM, citations, format_sources
from app.db import Document
from app.retrieval.companies import detect_documents
from app.retrieval.search import Hit, search
from app.tracing import NOOP, Span


def retrieve_for(session: Session, question: str, trace: Span = NOOP) -> list[Hit]:
    """One search, or one per company when the question names several."""
    files = list(session.scalars(select(Document.filename).where(Document.status == "ready")))
    docs = detect_documents(question, files)
    if len(docs) <= 1:
        return search(session, question, documents=docs or None, trace=trace)
    hits = []
    for d in docs:
        hits.extend(search(session, question, k=4, documents=[d], trace=trace))
    return hits


def run(session: Session, question: str, history: list[dict] | None = None, trace: Span = NOOP):
    usage = Usage()
    hits = retrieve_for(session, question, trace)
    step = {"type": "step", "step": 1, "tool": "search_reports", "args": {"query": question},
            "summary": f"{len(hits)} results"}
    yield step
    answer = REFUSAL_TEXT  # nothing passed the relevance threshold: refuse without an LLM call
    if hits:
        messages = [
            {"role": "system", "content": SINGLE_PASS_SYSTEM},
            *(history or []),
            {"role": "user", "content": f"{format_sources(hits)}\n\nQuestion: {question}"},
        ]
        for kind, value in stream(messages, usage, trace=trace):
            if kind == "text":
                yield {"type": "token", "text": value}
            else:
                answer = value["content"]
    else:
        yield {"type": "token", "text": answer}
    yield {"type": "done", "answer": answer, "citations": citations(answer, hits),
           "usage": usage.as_dict(), "steps": [step]}
