"""The chat model, with a local fallback and per-call cost.

OpenAI is the main model. If a call fails (outage, rate limit, no key), the
same request goes to Ollama, which speaks the OpenAI API, so one client type
serves both. Cost comes from the token usage the API reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from openai import OpenAI

from app.config import get_settings

log = logging.getLogger(__name__)

# USD per 1M tokens (input, output); a model missing here costs 0 (local models)
PRICES = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5-mini": (0.25, 2.00),
}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    models: list[str] = field(default_factory=list)

    def add(self, model: str, prompt: int, completion: int) -> None:
        inp, out = PRICES.get(model, (0.0, 0.0))
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.usd += (prompt * inp + completion * out) / 1e6
        if model not in self.models:
            self.models.append(model)

    def as_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "usd": round(self.usd, 6), "models": self.models}


def _clients():
    s = get_settings()
    yield s.chat_model, OpenAI(api_key=s.openai_api_key or "missing", timeout=60, max_retries=1)
    if s.fallback_model:
        yield s.fallback_model, OpenAI(base_url=s.ollama_url, api_key="ollama", timeout=300, max_retries=0)


def complete(messages: list[dict], usage: Usage, tools: list[dict] | None = None):
    """One chat completion (the message), trying the fallback model on failure."""
    error = None
    for model, client in _clients():
        try:
            kwargs = {"tools": tools} if tools else {}
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0, **kwargs)
            if resp.usage:
                usage.add(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
            return resp.choices[0].message
        except Exception as e:  # try the next model
            log.warning("LLM %s failed: %s", model, e)
            error = e
    raise RuntimeError(f"all language models failed: {error}")


def stream(messages: list[dict], usage: Usage, tools: list[dict] | None = None):
    """One completion, streamed. Yields ("text", piece) while the model writes,
    then ("message", {"content": ..., "tool_calls": [...]}) once at the end."""
    error = None
    for model, client in _clients():
        started = False
        try:
            kwargs = {"tools": tools} if tools else {}
            chunks = client.chat.completions.create(
                model=model, messages=messages, temperature=0, stream=True,
                stream_options={"include_usage": True}, **kwargs,
            )
            content, calls = [], {}  # tool calls arrive in pieces, keyed by index
            for chunk in chunks:
                if chunk.usage:
                    usage.add(model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    started = True
                    content.append(delta.content)
                    yield "text", delta.content
                for tc in delta.tool_calls or []:
                    call = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    call["id"] = tc.id or call["id"]
                    if tc.function:
                        call["name"] += tc.function.name or ""
                        call["arguments"] += tc.function.arguments or ""
            yield "message", {"content": "".join(content), "tool_calls": [calls[i] for i in sorted(calls)]}
            return
        except Exception as e:
            if started:  # half an answer already went out: don't mix in another model
                raise
            log.warning("LLM %s failed: %s", model, e)
            error = e
    raise RuntimeError(f"all language models failed: {error}")
