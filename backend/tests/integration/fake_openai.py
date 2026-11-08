"""A fake OpenAI-compatible server for the integration test.

The real API is started with OPENAI_BASE_URL (read by the openai SDK) and
OLLAMA_URL pointing here, so no request leaves the machine and the answers are
predictable:
- embeddings: a bag-of-words hash, so texts that share words are similar;
- chat: with tools and no tool result yet, it calls search_reports with the
  question; once it sees sources it answers citing the first one, or refuses.
Setting `FAIL_MODELS` makes chosen models return HTTP 500 (to test the fallback).
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import struct

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

DIM = 1536
REFUSAL = "N/A — this is not available in the uploaded reports."
FAIL_MODELS: set[str] = set()
CALLS: list[str] = []  # model of every chat request, in order

app = FastAPI()


def embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
    data = []
    for i, text in enumerate(texts):
        vec = embed(text)
        if body.get("encoding_format") == "base64":  # what the openai SDK asks for by default
            vec = base64.b64encode(struct.pack(f"<{DIM}f", *vec)).decode()
        data.append({"object": "embedding", "index": i, "embedding": vec})
    return {"object": "list", "data": data, "model": body["model"], "usage": {"prompt_tokens": 1, "total_tokens": 1}}


def reply(body: dict) -> dict:
    """{"content": str} or {"tool_call": {...}} for this conversation."""
    messages = body["messages"]
    has_tool_result = any(m["role"] == "tool" for m in messages)
    if body.get("tools") and not has_tool_result:
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        return {"tool_call": {"name": "search_reports", "arguments": json.dumps({"query": question})}}
    text = "\n".join(str(m.get("content") or "") for m in messages)
    blocks = re.findall(r"<sources>(.*?)</sources>", text, re.S)
    tags = re.findall(r"\[S(\d+)\]", "\n".join(blocks))
    if not tags:
        return {"content": REFUSAL}
    return {"content": f"The report says so [S{tags[0]}]."}


def chunk(model: str, delta: dict, usage: dict | None = None) -> str:
    payload = {"id": "fake", "object": "chat.completion.chunk", "created": 0, "model": model,
               "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": None}]}
    if usage:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body["model"]
    CALLS.append(model)
    if model in FAIL_MODELS:
        return JSONResponse({"error": {"message": f"{model} is down (fake)"}}, status_code=500)
    r = reply(body)
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}

    if not body.get("stream"):
        message = {"role": "assistant", "content": r.get("content")}
        if "tool_call" in r:
            message["tool_calls"] = [{"id": "call_1", "type": "function", "function": r["tool_call"]}]
        return {"id": "fake", "object": "chat.completion", "created": 0, "model": model, "usage": usage,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}

    def events():
        if "tool_call" in r:
            yield chunk(model, {"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": r["tool_call"]}]})
        else:
            for word in re.findall(r"\S+\s*", r["content"]):
                yield chunk(model, {"content": word})
        yield chunk(model, {}, usage=usage)
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
