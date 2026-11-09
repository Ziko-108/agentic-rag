"""HTTP API. Run with: uvicorn app.main:app --reload (from backend/)."""

import json
import logging
import threading
from typing import Literal
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app import tracing
from app.chat import service
from app.config import get_settings
from app.db import Chunk, Conversation, Document, get_session, init_db
from app.retrieval.search import _reranker
from app.ingest.pipeline import company_from_filename, ingest_document, mark_interrupted, seed_folder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    mark_interrupted()
    if get_settings().use_reranker:  # load the model now, not on the first question
        threading.Thread(target=_reranker, daemon=True).start()
    if seed_dir := get_settings().seed_dir:  # in the background: the API is usable meanwhile
        threading.Thread(target=seed_folder, args=(seed_dir,), daemon=True).start()
    yield
    tracing.flush()  # send the traces still in the buffer


app = FastAPI(title="Agentic RAG", lifespan=lifespan)


class DocumentOut(BaseModel):
    id: int
    filename: str
    company: str
    status: str
    error: str | None
    pages: int | None
    chunks: int


def _out(doc: Document, chunks: int) -> DocumentOut:
    return DocumentOut(
        id=doc.id, filename=doc.filename, company=doc.company, status=doc.status,
        error=doc.error, pages=doc.pages, chunks=chunks,
    )


def _chunk_count(session: Session, document_id: int) -> int:
    return session.scalar(select(func.count()).where(Chunk.document_id == document_id))


@app.get("/health")
def health(session: Session = Depends(get_session)):
    try:
        session.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


@app.post("/documents", status_code=202, response_model=DocumentOut)
async def upload_document(
    file: UploadFile, background: BackgroundTasks, session: Session = Depends(get_session)
):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file is larger than 50 MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "only PDF files are accepted")
    filename = file.filename or "upload.pdf"
    if session.scalar(select(Document).where(Document.filename == filename)):
        raise HTTPException(409, f"'{filename}' is already uploaded; delete it first")

    doc = Document(filename=filename, company=company_from_filename(filename))
    session.add(doc)
    session.commit()
    background.add_task(ingest_document, doc.id, data)
    return _out(doc, 0)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(get_session)):
    counts = dict(session.execute(select(Chunk.document_id, func.count()).group_by(Chunk.document_id)).all())
    docs = session.scalars(select(Document).order_by(Document.filename))
    return [_out(d, counts.get(d.id, 0)) for d in docs]


@app.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, session: Session = Depends(get_session)):
    doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "document not found")
    return _out(doc, _chunk_count(session, doc.id))


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: int, session: Session = Depends(get_session)):
    doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "document not found")
    session.delete(doc)  # chunks go with it (ON DELETE CASCADE)
    session.commit()


# ------------------------------------------------------------------ chat


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    mode: Literal["agent", "pipeline"] = "agent"


def _start(body: ChatIn, user_id: str, session: Session):
    try:
        return service.start(session, user_id, body.question, body.conversation_id, body.mode)
    except service.BudgetExceeded as e:
        raise HTTPException(429, str(e))
    except service.NotFound as e:
        raise HTTPException(404, str(e))


def _chat_json(body: ChatIn, user_id: str, session: Session) -> dict:
    conv = _start(body, user_id, session)
    try:
        *_, done = service.turn(conv.id, body.question, body.mode)
    except RuntimeError as e:  # every language model failed
        raise HTTPException(503, str(e))
    return {k: v for k, v in done.items() if k != "type"}


@app.post("/chat")
def chat(body: ChatIn, x_user_id: str = Header("anonymous"), session: Session = Depends(get_session)):
    return _chat_json(body, x_user_id, session)


@app.post("/pipeline/chat")
def chat_pipeline(body: ChatIn, x_user_id: str = Header("anonymous"), session: Session = Depends(get_session)):
    """The single-pass baseline: `python eval.py --label pipeline --api http://localhost:8000/pipeline`."""
    return _chat_json(body.model_copy(update={"mode": "pipeline"}), x_user_id, session)


@app.post("/chat/stream")
def chat_stream(body: ChatIn, x_user_id: str = Header("anonymous"), session: Session = Depends(get_session)):
    """Server-sent events: `step`, `token` (and `reset` when the agent retries), then `done` (or `error`)."""
    conv = _start(body, x_user_id, session)

    def events():
        try:
            for event in service.turn(conv.id, body.question, body.mode):
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        except Exception as e:  # headers are sent: report the failure as an event
            log.exception("chat stream failed")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Conversation-Id": str(conv.id)})


@app.get("/conversations")
def list_conversations(x_user_id: str = Header("anonymous"), session: Session = Depends(get_session)):
    convs = session.scalars(
        select(Conversation).where(Conversation.user_id == x_user_id).order_by(Conversation.id.desc())
    )
    return [{"id": c.id, "title": c.title, "mode": c.mode, "created_at": c.created_at} for c in convs]


def _own_conversation(conversation_id: int, user_id: str, session: Session) -> Conversation:
    conv = session.get(Conversation, conversation_id)
    if not conv or conv.user_id != user_id:
        raise HTTPException(404, "conversation not found")
    return conv


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: int, x_user_id: str = Header("anonymous"),
                     session: Session = Depends(get_session)):
    conv = _own_conversation(conversation_id, x_user_id, session)
    return {"id": conv.id, "title": conv.title, "mode": conv.mode, "messages": [
        {"role": m.role, "content": m.content, "citations": m.citations, "steps": m.steps,
         "usage": {"prompt_tokens": m.prompt_tokens, "completion_tokens": m.completion_tokens, "usd": m.usd}}
        for m in conv.messages]}


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: int, x_user_id: str = Header("anonymous"),
                        session: Session = Depends(get_session)):
    session.delete(_own_conversation(conversation_id, x_user_id, session))
    session.commit()


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: int, session: Session = Depends(get_session)):
    """The passage behind a citation."""
    chunk = session.get(Chunk, chunk_id)
    if not chunk:
        raise HTTPException(404, "passage not found")
    return {"id": chunk.id, "document": chunk.document.filename, "page": chunk.page,
            "section": chunk.section, "content": chunk.content}
