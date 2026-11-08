"""The trace of one agent turn has the right shape, also when every step of the
generator runs in another thread (as under a streaming response).

Spans go to an in-memory exporter: nothing is sent to Langfuse.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import tracing
from app.chat import agent, llm
from app.retrieval import search as search_module
from app.retrieval.search import Hit


class FakeSession:
    def scalars(self, _):
        return ["RWE AG.pdf"]


def fake_openai(replies):
    """An OpenAI client whose streamed replies are scripted."""
    replies = iter(replies)

    def create(**_):
        reply = next(replies)
        if "tool" in reply:
            fn = NS(name=reply["tool"], arguments=json.dumps(reply["args"]))
            yield NS(usage=None, choices=[NS(delta=NS(content=None, tool_calls=[NS(index=0, id="c1", function=fn)]))])
        else:
            yield NS(usage=None, choices=[NS(delta=NS(content=reply["text"], tool_calls=None))])
        yield NS(usage=NS(prompt_tokens=1000, completion_tokens=50), choices=[])

    return NS(chat=NS(completions=NS(create=create)))


@pytest.fixture
def exporter(monkeypatch):
    exp = InMemorySpanExporter()
    client = Langfuse(public_key="pk-test", secret_key="sk-test", base_url="http://127.0.0.1:9", span_exporter=exp)
    monkeypatch.setattr(tracing, "_client", lambda: client)
    yield exp, client
    client.shutdown()


def test_agent_turn_trace(monkeypatch, exporter):
    exp, client = exporter
    script = [{"tool": "search_reports", "args": {"query": "installed capacity"}}, {"text": "39.3 GW [S1]."}]
    client_ = fake_openai(script)  # one client for the whole turn, so the script advances
    monkeypatch.setattr(llm, "_clients", lambda: iter([("gpt-4.1-mini", client_)]))
    monkeypatch.setattr(search_module, "embed", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(search_module, "dense_ids", lambda *a, **k: [1])
    monkeypatch.setattr(search_module, "lexical_ids", lambda *a, **k: [1])
    monkeypatch.setattr(search_module, "load_hits", lambda s, ids, vector=None: {
        1: Hit(1, "RWE AG.pdf", 47, "Capacity", "39.3 GW", similarity=0.6)})
    monkeypatch.setattr(search_module, "rerank", lambda q, hits: hits)

    trace = tracing.start_trace("chat-turn", user_id="u1", session_id="42", input="q", metadata={"mode": "agent"})
    steps = agent.run(FakeSession(), "What is RWE's capacity?", trace=trace)
    with ThreadPoolExecutor(4) as pool:  # every next() in a thread of its own
        while True:
            try:
                pool.submit(next, steps).result()
            except StopIteration:
                break
    trace.end(output="done")
    client.flush()

    spans = exp.get_finished_spans()
    name = {s.context.span_id: s.name for s in spans}
    tree = sorted((s.name, name.get(s.parent.span_id) if s.parent else None) for s in spans)
    assert tree == [("chat-turn", None), ("llm", "chat-turn"), ("llm", "chat-turn"),
                    ("retrieval", "search_reports"), ("search_reports", "chat-turn")]
    assert len({s.context.trace_id for s in spans}) == 1
    assert all(s.attributes["user.id"] == "u1" and s.attributes["session.id"] == "42" for s in spans)

    gen = next(s for s in spans if s.name == "llm")
    assert json.loads(gen.attributes["langfuse.observation.usage_details"]) == {"input": 1000, "output": 50}
    assert json.loads(gen.attributes["langfuse.observation.cost_details"])["total"] == pytest.approx(0.00048)


def test_tracing_is_off_without_keys(monkeypatch):
    monkeypatch.setattr(tracing, "_client", lambda: None)
    trace = tracing.start_trace("chat-turn", user_id="u", session_id="1", input="q")
    assert trace is tracing.NOOP and trace.child("x").url() is None
