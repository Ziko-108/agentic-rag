"""Postgres is the single source of truth: documents, chunks and their vectors.

Tables are created at startup (no migrations: a known limitation, see README).
`chunks.tsv` is computed by Postgres from `content`, so full-text search can
never get out of sync with the text.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, Index, String, Text, create_engine, func, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from app.config import get_settings

settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), unique=True)
    company: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|processing|ready|failed
    error: Mapped[str | None] = mapped_column(Text)
    pages: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page: Mapped[int]  # 1-based position in the PDF
    kind: Mapped[str] = mapped_column(String(10))  # text|table
    section: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)  # "[Company | p.N | Section]\n..."
    tokens: Mapped[int]
    embedding = mapped_column(Vector(settings.embedding_dim))
    tsv = mapped_column(TSVECTOR, Computed("to_tsvector('english', content)", persisted=True))

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("chunks_tsv_idx", "tsv", postgresql_using="gin"),
        Index(
            "chunks_embedding_idx", "embedding",
            postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True)
    title: Mapped[str] = mapped_column(String(200))
    mode: Mapped[str] = mapped_column(String(10), default="agent")  # agent|pipeline
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True,
        order_by="Message.id",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(10))  # user|assistant
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSONB)
    steps: Mapped[list | None] = mapped_column(JSONB)
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    usd: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
