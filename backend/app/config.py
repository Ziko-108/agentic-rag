"""All settings come from environment variables (.env in development)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    chat_model: str = "gpt-4.1-mini"
    fallback_model: str = ""  # e.g. "qwen2.5:7b" served by Ollama; empty = no fallback
    ollama_url: str = "http://localhost:11434/v1"
    max_usd_per_question: float = 0.05  # agent stops calling tools beyond this
    daily_usd_per_user: float = 1.00  # per X-User-Id; requests beyond it get 429
    history_tokens: int = 2000  # how much of a conversation goes back to the model

    postgres_user: str = "rag"
    postgres_password: str = ""
    postgres_db: str = "rag"
    postgres_host: str = "localhost"  # "db" inside docker compose
    postgres_port: int = 5432

    # retrieval (defaults chosen with scripts/retrieval_eval.py, see EVAL.md)
    top_k: int = 12
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_candidates: int = 30
    relevance_threshold: float = 0.30  # cosine similarity below which a chunk is dropped

    seed_dir: str = ""  # PDFs ingested at startup if not in the database yet

    # tracing: off unless both keys are set (see app/tracing.py)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
