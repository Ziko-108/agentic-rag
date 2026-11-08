"""The agent loop with a scripted fake model: no network, no database."""

import json

import pytest

from app.chat import agent
from app.chat.prompts import citations
from app.retrieval.search import Hit


class FakeSession:
    def scalars(self, _):
        return ["RWE AG.pdf", "Datalogic.pdf"]


def call(name, **args):
    return {"id": f"c-{name}-{json.dumps(args, sort_keys=True)}", "name": name, "arguments": json.dumps(args)}


def scripted(replies):
    """A fake `stream`: each model call takes the next reply (tool calls or text)."""
    replies = iter(replies)
    seen_tools = []

    def stream(messages, usage, tools=None):
        seen_tools.append(tools is not None)
        reply = next(replies)
        if isinstance(reply, str):
            yield "text", reply
            yield "message", {"content": reply, "tool_calls": []}
        else:
            yield "message", {"content": "", "tool_calls": reply}

    return stream, seen_tools


@pytest.fixture
def fake_search(monkeypatch):
    def search(session, query, documents=None, **_):
        return [Hit(1, (documents or ["RWE AG.pdf"])[0], 47, "Capacity", f"[RWE AG | p.47]\n{query}: 39.3 GW")]
    monkeypatch.setattr(agent, "search", search)


def run(monkeypatch, replies):
    stream, seen_tools = scripted(replies)
    monkeypatch.setattr(agent, "stream", stream)
    events = list(agent.run(FakeSession(), "What is RWE's capacity?"))
    return events, seen_tools


def test_search_then_answer_with_citation(monkeypatch, fake_search):
    events, _ = run(monkeypatch, [[call("search_reports", query="capacity", companies=["RWE AG"])],
                                  "RWE had 39.3 GW [S1]."])
    done = events[-1]
    assert done["type"] == "done"
    assert (done["citations"][0]["document"], done["citations"][0]["page"]) == ("RWE AG.pdf", 47)
    assert [e["tool"] for e in events if e["type"] == "step"] == ["search_reports"]


def test_stops_at_the_step_cap(monkeypatch, fake_search):
    endless = [[call("search_reports", query=f"q{i}")] for i in range(agent.MAX_STEPS)]
    events, seen_tools = run(monkeypatch, endless + ["Final answer [S1]."])
    assert sum(e["type"] == "step" for e in events) == agent.MAX_STEPS
    assert seen_tools[-1] is False  # the last call is made without tools
    assert events[-1]["answer"] == "Final answer [S1]."


def test_repeated_call_is_not_run_twice(monkeypatch, fake_search):
    same = call("search_reports", query="capacity")
    events, _ = run(monkeypatch, [[same], [same], "Done."])
    steps = [e for e in events if e["type"] == "step"]
    assert steps[1]["summary"] == "repeated call skipped"


def test_tool_error_goes_back_to_the_model(monkeypatch, fake_search):
    events, _ = run(monkeypatch, [[call("calculator", expression="__import__('os')")], "Sorry."])
    assert [e["summary"] for e in events if e["type"] == "step"] == ["error"]
    assert events[-1]["answer"] == "Sorry."


def test_unknown_company_is_reported_not_raised(monkeypatch, fake_search):
    events, _ = run(monkeypatch, [[call("search_reports", query="x", companies=["Nonexistent Ltd"])], "N/A"])
    assert [e["summary"] for e in events if e["type"] == "step"] == ["no such company"]


def test_calculator():
    assert agent.calculate("(406.1 - 380.2) / 380.2 * 100") == pytest.approx(6.8122, rel=1e-4)
    assert agent.calculate("1,000 * 2") == 2000
    with pytest.raises(ValueError):
        agent.calculate("open('x')")


def test_citations_ignore_unknown_tags_and_repeat_pages():
    hits = [Hit(1, "A.pdf", 3, None, ""), Hit(2, "A.pdf", 3, None, ""), Hit(3, "B.pdf", 9, None, "")]
    got = citations("x [S2] y [S1] z [S3] w [S9]", hits)
    assert [(c["document"], c["page"]) for c in got] == [("A.pdf", 3), ("B.pdf", 9)]
    assert [c["tags"] for c in got] == [["S2", "S1"], ["S3"]]
