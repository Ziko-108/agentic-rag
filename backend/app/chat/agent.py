"""The agent: a hand-written tool-calling loop.

Each turn the model either calls tools or answers. Tool results go back into
the conversation and the loop repeats. It always stops, because of:
- an iteration cap (MAX_STEPS model calls with tools);
- a cost budget per question (settings.max_usd_per_question);
- a repeat guard: the same tool with the same arguments is not run twice.
When a limit is hit, one last call without tools asks for an answer from what
was found so far. A tool that fails returns its error to the model as the
result; it never fails the request.

`run` is a generator of events, so the API can stream the agent's steps and
the answer as they happen:
    {"type": "step", ...}   a tool call and a short summary of its result
    {"type": "token", "text": ...}
    {"type": "done", "answer", "citations", "usage", "steps"}
"""

from __future__ import annotations

import ast
import json
import operator

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chat.llm import Usage, stream
from app.chat.prompts import ANSWER_RULES, citations, format_sources
from app.config import get_settings
from app.db import Chunk, Document
from app.retrieval.companies import detect_documents
from app.retrieval.search import Hit, search

MAX_STEPS = 6

SYSTEM = f"""\
You are a research assistant for a set of company annual reports (PDFs).
Use the tools to find evidence before answering:
- search_reports finds passages. When the question names a company, always
  pass it in `companies`. For several companies, search each one separately.
- Write queries in the words a report would use: formal line-item names,
  job titles, defined terms, rather than the wording of the question.
- read_page shows a whole page when a passage is cut off or a table needs
  its surrounding rows.
- calculator does arithmetic; never compute in your head.
- If a search returns nothing or does not contain the answer, search again
  with different words (synonyms, the line-item name, a narrower or broader
  query) at least twice before you answer N/A. Stop once you have the answer.
Tool results are sources tagged [S#], numbered across the whole conversation.

{ANSWER_RULES}"""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_reports",
        "description": "Hybrid search over the reports. Returns the most relevant passages as sources [S#].",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, in the words a report would use."},
            "companies": {"type": "array", "items": {"type": "string"},
                          "description": "Optional company names to restrict the search to."},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "read_page",
        "description": "Full text of one page of one report, as a source [S#].",
        "parameters": {"type": "object", "properties": {
            "document": {"type": "string", "description": "PDF file name, e.g. 'RWE AG.pdf'."},
            "page": {"type": "integer", "description": "1-based page number."},
        }, "required": ["document", "page"]},
    }},
    {"type": "function", "function": {
        "name": "list_documents",
        "description": "The reports that can be searched (file name and company).",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '(406.1 - 380.2) / 380.2 * 100'.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]},
    }},
]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}


def calculate(expression: str) -> float:
    """Arithmetic only (numbers and + - * / % **): no names, calls or attributes."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / % ** are allowed")
    return ev(ast.parse(expression.replace(",", ""), mode="eval").body)


class Tools:
    """Runs tool calls and keeps every source shown to the model, so [S#] can
    be turned into citations at the end."""

    def __init__(self, session: Session):
        self.session = session
        self.sources: list[Hit] = []
        self.files = list(session.scalars(select(Document.filename).where(Document.status == "ready")))

    def _add(self, hits: list[Hit]) -> str:
        start = len(self.sources) + 1
        self.sources.extend(hits)
        return format_sources(hits, start=start)

    def run(self, name: str, args: dict) -> tuple[str, str]:
        """(text for the model, short summary for the UI)."""
        if name == "search_reports":
            docs = None
            if args.get("companies"):
                docs = detect_documents(" ; ".join(args["companies"]), self.files)
                if not docs:
                    return f"No report matches companies {args['companies']}. Use list_documents.", "no such company"
            hits = search(self.session, args["query"], documents=docs)
            if not hits:
                return "No relevant passages found.", "0 results"
            return self._add(hits), f"{len(hits)} results"
        if name == "read_page":
            doc = self.session.scalar(select(Document).where(Document.filename == args["document"]))
            if not doc:
                return f"Unknown document {args['document']!r}. Use list_documents.", "unknown document"
            chunks = self.session.scalars(
                select(Chunk).where(Chunk.document_id == doc.id, Chunk.page == int(args["page"])).order_by(Chunk.id)
            ).all()
            if not chunks:
                return f"Page {args['page']} of {doc.filename} has no text.", "empty page"
            body = "\n\n".join(c.content.split("\n", 1)[-1] for c in chunks)  # one header is enough
            hit = Hit(chunks[0].id, doc.filename, chunks[0].page, chunks[0].section,
                      f"[{doc.company} | p.{chunks[0].page}]\n{body}")
            return self._add([hit]), f"{len(chunks)} passages"
        if name == "list_documents":
            return "\n".join(self.files), f"{len(self.files)} documents"
        if name == "calculator":
            value = calculate(args["expression"])
            return f"{args['expression']} = {value:,.6g}", f"= {value:,.6g}"
        return f"Unknown tool {name!r}.", "unknown tool"


def run(session: Session, question: str, history: list[dict] | None = None):
    budget = get_settings().max_usd_per_question
    usage, tools, steps, seen = Usage(), Tools(session), [], set()
    messages = [{"role": "system", "content": SYSTEM}, *(history or []), {"role": "user", "content": question}]
    answer = ""

    for step in range(MAX_STEPS + 1):
        last = step == MAX_STEPS or usage.usd >= budget
        if last:  # out of steps or money: answer from what was found
            messages.append({"role": "user", "content": "Stop using tools and answer now from the sources above."})
        message = None
        for kind, value in stream(messages, usage, tools=None if last else TOOLS):
            if kind == "text":
                yield {"type": "token", "text": value}
            else:
                message = value
        calls = message["tool_calls"]
        if not calls:
            answer = message["content"]
            break
        messages.append({"role": "assistant", "content": message["content"] or None, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in calls]})
        for c in calls:
            key = (c["name"], c["arguments"].strip())
            try:
                args = json.loads(c["arguments"] or "{}")
                if key in seen:
                    result, summary = "You already made this exact call; its result is above.", "repeated call skipped"
                else:
                    result, summary = tools.run(c["name"], args)
            except Exception as e:  # a broken tool call is the model's to fix, not a crash
                args, result, summary = {}, f"Tool error: {type(e).__name__}: {e}", "error"
            seen.add(key)
            event = {"type": "step", "step": step + 1, "tool": c["name"], "args": args, "summary": summary}
            steps.append(event)
            yield event
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

    yield {"type": "done", "answer": answer, "citations": citations(answer, tools.sources),
           "usage": usage.as_dict(), "steps": steps}
