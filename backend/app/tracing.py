"""Langfuse tracing. On only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

One trace per chat turn:
    chat-turn (agent)               user, conversation (session), question → answer
      llm (generation)              each model call: messages, output, tokens, cost
      search_reports (tool)         the agent's arguments → result summary
        retrieval (retriever)       query, filter, candidate counts, hits with scores
      read_page / calculator (tool)

Each span is created from its parent object, not from the "current span"
context: the agent is a generator, and under a streaming response each of its
steps can run in a different thread, where that context is not carried over.

Tracing must never break a request, so every call into Langfuse is guarded.
Without keys, `start_trace` returns NOOP and nothing is sent anywhere.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings

log = logging.getLogger(__name__)


class Span:
    """A no-op span: what the code uses when tracing is off."""

    trace_id: str | None = None

    def child(self, name: str, as_type: str = "span", **fields) -> Span:
        return self

    def end(self, **fields) -> None:
        pass

    def url(self) -> str | None:
        return None


NOOP = Span()


class _LangfuseSpan(Span):
    def __init__(self, client, span, attributes: dict):
        self.client, self.span, self.attributes = client, span, attributes
        self.trace_id = span.trace_id

    def child(self, name: str, as_type: str = "span", **fields) -> Span:
        from langfuse import propagate_attributes

        try:
            with propagate_attributes(**self.attributes):  # user and session on every span
                return _LangfuseSpan(self.client, self.span.start_observation(name=name, as_type=as_type, **fields),
                                     self.attributes)
        except Exception:
            log.warning("tracing: could not start span %s", name, exc_info=True)
            return NOOP

    def end(self, **fields) -> None:
        try:
            if fields:
                self.span.update(**fields)
            self.span.end()
        except Exception:
            log.warning("tracing: could not end span", exc_info=True)

    def url(self) -> str | None:
        try:
            return self.client.get_trace_url(trace_id=self.trace_id)
        except Exception:
            return None


@lru_cache
def _client():
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    from langfuse import Langfuse

    return Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, base_url=s.langfuse_base_url)


def start_trace(name: str, *, user_id: str, session_id: str, input, metadata: dict | None = None) -> Span:
    client = _client()
    if client is None:
        return NOOP
    from langfuse import propagate_attributes

    attributes = {"user_id": user_id, "session_id": session_id, "trace_name": name,
                  "tags": [str(metadata.get("mode"))] if metadata and metadata.get("mode") else None}
    attributes = {k: v for k, v in attributes.items() if v is not None}
    try:
        with propagate_attributes(**attributes):
            span = client.start_observation(name=name, as_type="agent", input=input, metadata=metadata)
        return _LangfuseSpan(client, span, attributes)
    except Exception:
        log.warning("tracing: could not start trace", exc_info=True)
        return NOOP


def flush() -> None:
    client = _client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            log.warning("tracing: flush failed", exc_info=True)
