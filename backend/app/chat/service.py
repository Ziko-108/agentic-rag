"""One chat turn: check the user's budget, load the history, run the agent or
the pipeline, save both messages. Used by the JSON and the streaming endpoint."""

from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.chat import agent, single_pass
from app.config import get_settings
from app.db import Conversation, Message, engine
from app.ingest.chunker import default_counter


class BudgetExceeded(Exception):
    pass


class NotFound(Exception):
    pass


def spent_today(session: Session, user_id: str) -> float:
    return session.scalar(
        select(func.coalesce(func.sum(Message.usd), 0.0))
        .join(Conversation).where(Conversation.user_id == user_id, func.date(Message.created_at) == date.today())
    )


def history(conv: Conversation, budget_tokens: int) -> list[dict]:
    """The newest user/assistant messages that fit in `budget_tokens` (oldest first)."""
    count, out, used = default_counter(), [], 0
    for m in reversed(conv.messages):
        n = count(m.content)
        if used + n > budget_tokens:
            break
        out.insert(0, {"role": m.role, "content": m.content})
        used += n
    return out


def start(session: Session, user_id: str, question: str, conversation_id: int | None, mode: str) -> Conversation:
    """Checks that must fail before any streaming starts (they become HTTP errors)."""
    limit = get_settings().daily_usd_per_user
    if spent_today(session, user_id) >= limit:
        raise BudgetExceeded(f"daily budget of ${limit:.2f} for user '{user_id}' is spent")
    if conversation_id is None:
        conv = Conversation(user_id=user_id, title=question[:200], mode=mode)
        session.add(conv)
        session.commit()
        return conv
    conv = session.get(Conversation, conversation_id)
    if not conv or conv.user_id != user_id:
        raise NotFound("conversation not found")
    return conv


def turn(conversation_id: int, question: str, mode: str):
    """Run one turn in its own session (it outlives the request handler when
    streaming) and yield the agent's events; the last one is "done"."""
    with Session(engine) as session:
        conv = session.get(Conversation, conversation_id)
        past = history(conv, get_settings().history_tokens)
        runner = agent.run if mode == "agent" else single_pass.run
        done = None
        for event in runner(session, question, past):
            if event["type"] == "done":
                done = event
            else:
                yield event
        session.rollback()  # end the read-only search transaction before writing
        u = done["usage"]
        session.add_all([
            Message(conversation_id=conv.id, role="user", content=question),
            Message(conversation_id=conv.id, role="assistant", content=done["answer"],
                    citations=done["citations"], steps=done["steps"], prompt_tokens=u["prompt_tokens"],
                    completion_tokens=u["completion_tokens"], usd=u["usd"]),
        ])
        session.commit()
        yield {**done, "conversation_id": conv.id}
