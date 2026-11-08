"""Integration test through the running API, with a fake LLM.

Starts the real API (uvicorn, separate process) against the real Postgres, with
the OpenAI and Ollama URLs pointing at tests/integration/fake_openai.py. Then it
uploads a small generated PDF, waits for ingestion, chats in both modes,
streams, checks conversations and the fallback, and deletes everything it made.

Needs Postgres (docker compose up db); skipped when it is not reachable.
Run: pytest tests/integration -v
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pymupdf
import pytest
import uvicorn

from app.config import get_settings
from tests.integration import fake_openai

BACKEND = Path(__file__).resolve().parents[2]
USER = {"X-User-Id": "integration-test"}
FILENAME = "Testco.pdf"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def postgres_up() -> bool:
    s = get_settings()
    try:
        socket.create_connection((s.postgres_host, s.postgres_port), timeout=2).close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not postgres_up(), reason="Postgres is not running")


def make_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Testco plc Annual Report 2023", fontsize=20)
    page = doc.new_page()
    page.insert_text((72, 72), "Financial review", fontsize=14)
    page.insert_text((72, 100), "Testco revenue was 42 million euros in 2023, up from 37 million.", fontsize=11)
    return doc.tobytes()


@pytest.fixture(scope="module")
def api():
    fake_port, api_port = free_port(), free_port()
    fake = uvicorn.Server(uvicorn.Config(fake_openai.app, port=fake_port, log_level="warning"))
    threading.Thread(target=fake.run, daemon=True).start()

    env = {
        **os.environ,
        "OPENAI_API_KEY": "fake-key",  # the real key is never sent anywhere by this test
        "OPENAI_BASE_URL": f"http://127.0.0.1:{fake_port}/v1",
        "OLLAMA_URL": f"http://127.0.0.1:{fake_port}/v1",
        "FALLBACK_MODEL": "fake-local",
        "USE_RERANKER": "false",  # the reranker is measured elsewhere; here it would only slow startup
        "SEED_DIR": "",
        "LANGFUSE_PUBLIC_KEY": "",  # no traces from tests
        "LANGFUSE_SECRET_KEY": "",
    }
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(api_port)],
                            cwd=BACKEND, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client = httpx.Client(base_url=f"http://127.0.0.1:{api_port}", timeout=60)
    try:
        for _ in range(120):
            try:
                if client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.5)
        else:
            pytest.fail("the API did not start")
        _delete_leftovers(client)
        yield client
    finally:
        _delete_leftovers(client)
        proc.terminate()
        proc.wait(timeout=20)
        fake.should_exit = True


def _delete_leftovers(client: httpx.Client):
    for d in client.get("/documents").json():
        if d["filename"] == FILENAME:
            client.delete(f"/documents/{d['id']}")
    for c in client.get("/conversations", headers=USER).json():
        client.delete(f"/conversations/{c['id']}", headers=USER)


@pytest.fixture(scope="module")
def document(api):
    r = api.post("/documents", files={"file": (FILENAME, make_pdf(), "application/pdf")})
    assert r.status_code == 202, r.text
    doc = r.json()
    assert doc["status"] == "pending"
    for _ in range(120):
        doc = api.get(f"/documents/{doc['id']}").json()
        if doc["status"] in ("ready", "failed"):
            break
        time.sleep(0.5)
    assert doc["status"] == "ready", doc
    return doc


def test_upload_and_status(api, document):
    assert document["pages"] == 2 and document["chunks"] >= 1
    assert api.post("/documents", files={"file": (FILENAME, make_pdf(), "application/pdf")}).status_code == 409
    assert api.post("/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}).status_code == 415


def test_agent_answer_cites_the_page(api, document):
    r = api.post("/chat", headers=USER, json={"question": "What was Testco revenue in 2023?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [s["tool"] for s in body["steps"]] == ["search_reports"]
    assert {(c["document"], c["page"]) for c in body["citations"]} == {(FILENAME, 2)}

    # the passage behind the citation can be opened
    passage = api.get(f"/chunks/{body['citations'][0]['chunk_id']}").json()
    assert "42 million" in passage["content"]

    # a follow-up goes into the same conversation, which other users cannot see
    conv = body["conversation_id"]
    assert api.post("/chat", headers=USER, json={"question": "And in 2022?", "conversation_id": conv}).status_code == 200
    saved = api.get(f"/conversations/{conv}", headers=USER).json()
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant", "user", "assistant"]
    assert saved["messages"][1]["usage"]["usd"] > 0
    assert api.get(f"/conversations/{conv}", headers={"X-User-Id": "someone-else"}).status_code == 404


def test_pipeline_refuses_without_evidence(api, document):
    before = len(fake_openai.CALLS)
    r = api.post("/pipeline/chat", headers=USER, json={"question": "zebra quantum giraffe"})
    assert r.json()["answer"].startswith("N/A") and r.json()["citations"] == []
    assert len(fake_openai.CALLS) == before  # no relevant source: no LLM call at all


def test_stream_events(api, document):
    with api.stream("POST", "/chat/stream", headers=USER, json={"question": "Testco revenue 2023"}) as r:
        assert r.status_code == 200 and r.headers["x-conversation-id"]
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "step" and "token" in kinds and kinds[-1] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "token") == events[-1]["answer"]


def test_fallback_model_when_main_fails(api, document):
    fake_openai.FAIL_MODELS.add(get_settings().chat_model)
    try:
        r = api.post("/chat", headers=USER, json={"question": "What was Testco revenue in 2023?"})
        assert r.status_code == 200
        assert r.json()["usage"]["models"] == ["fake-local"]

        fake_openai.FAIL_MODELS.add("fake-local")  # every model down: a clean 503
        assert api.post("/chat", headers=USER, json={"question": "Testco revenue?"}).status_code == 503
    finally:
        fake_openai.FAIL_MODELS.clear()


def test_delete_removes_everything(api, document):
    assert api.delete(f"/documents/{document['id']}").status_code == 204
    assert api.get(f"/documents/{document['id']}").status_code == 404
    r = api.post("/pipeline/chat", headers=USER, json={"question": "What was Testco revenue in 2023?"})
    assert r.json()["answer"].startswith("N/A")
