from typing import Optional
from urllib.parse import parse_qs, urlsplit, urlunsplit

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Database Config
    # A single DATABASE_URL wins over the discrete fields when set. Every managed
    # Postgres (Neon, Supabase, Render) hands out one URL, so this is what
    # production sets; the discrete fields remain the local-compose path.
    database_url: Optional[str] = Field(default=None)
    db_user: str = Field(default="postgres")
    db_password: str = Field(default="password")
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=6432)
    db_direct_port: int = Field(default=5433)
    db_name: str = Field(default="tutorial_db")

    # Broker/result backend. Celery reads this too, so it lives here rather than
    # being pulled straight from os.environ in worker.py.
    redis_url: str = Field(default="redis://localhost:6379/0")

    # File handling
    upload_dir: str = Field(default="data/uploads")
    ingestion_log_path: str = Field(default="data/ingestion.log")
    # Uploads ride through Postgres as bytea, so an unbounded PDF would become an
    # unbounded row. 25 MB keeps a single upload well inside a 512 MB free tier.
    max_upload_bytes: int = Field(default=25 * 1024 * 1024)

    # Schema bootstrap is a deploy-time step, not a boot-time one: two web
    # replicas racing on create_all is a real failure mode. Local dev flips this
    # on for convenience.
    bootstrap_db_on_startup: bool = Field(default=False)

    # Embedding config
    # "local" runs sentence-transformers in-process (dev, eval, GPU box).
    # "gemini" calls a hosted API, which is what makes the worker fit in a
    # free-tier container without torch installed at all.
    embedding_provider: str = Field(default="local")
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5")
    embedding_dimension: int = Field(default=768)
    embedding_batch_size: int = Field(default=64)
    # Hosted-provider settings. The model is asked for embedding_dimension
    # explicitly (via outputDimensionality), so it matches the existing
    # Vector(768) column and switching providers needs no migration.
    embedding_api_key: str = Field(default="")
    embedding_api_base: str = Field(default="https://generativelanguage.googleapis.com/v1beta")
    # text-embedding-004 was retired; the API now 404s on it. gemini-embedding-001
    # replaces it but defaults to 3072 dimensions, so embedding_dimension is sent
    # as outputDimensionality on every request to keep the Vector(768) column valid.
    embedding_api_model: str = Field(default="gemini-embedding-001")
    # Only OpenAI's text-embedding-3-* family accepts a dimensions override.
    # Sending it to a model that does not (Jina, most others) is a 400, so this
    # stays opt-in. Natively-768 models like jina-embeddings-v2-base-en need it off.
    embedding_api_send_dimensions: bool = Field(default=False)
    # Hosted embedding APIs reject over-long inputs outright (Jina: 8192 tokens),
    # where a local sentence-transformer silently truncates at 512. One giant
    # input therefore fails a whole document on the hosted path but not locally.
    #
    # 8000 chars, not a token count, because we cannot count in the provider's
    # tokenizer: a 21k-char table from iso27001.pdf measures 5,039 tokens under
    # tiktoken and still blew Jina's 8192 limit — their BERT-style tokenizer
    # splits tables and symbols far more finely. Characters are the only unit
    # every provider agrees on, and 8000 is safe even at ~1 char/token.
    #
    # Nothing is lost relative to local behaviour: bge-base-en-v1.5 has a
    # 512-token (~2000 char) window, so it never saw beyond this anyway.
    embedding_max_input_chars: int = Field(default=8000)
    # Batch size and HTTP timeout for hosted embedding calls.
    #
    # 32, not the API's maximum of 100: a 100-item batch of ~1.5k-char chunks
    # regularly ran past a 60s read timeout on a free tier, and a timeout costs
    # the whole retry budget. Smaller batches mean more requests but each one
    # completes well inside the timeout, which is the better trade when a single
    # failure fails an entire document ingestion.
    embedding_api_batch_size: int = Field(default=32)
    embedding_api_timeout: float = Field(default=120.0)
    
    # Chunker config
    window_size: int = Field(default=3)
    threshold_factor: float = Field(default=0.8)
    min_sentences: int = Field(default=2)
    min_words: int = Field(default=50)
    max_tokens: int = Field(default=400)
    overlap_tokens: int = Field(default=100)
    chunk_embedding_strategy: str = Field(default="hybrid")
    auto_unload_embeddings: bool = Field(default=True)
    auto_unload_delay: float = Field(default=10.0)



    # LLM Config
    llm_provider: str = Field(default="gemini")
    llm_api_key: str = Field(default="")
    llm_api_base: str = Field(default="https://generativelanguage.googleapis.com/v1beta")
    # gemini-1.5-flash and 2.0-flash are both retired and now 404.
    llm_model: str = Field(default="gemini-2.5-flash")
    llm_temperature: float = Field(default=0.0)
    agent_max_iterations: int = Field(default=5)


    @property
    def async_database_url(self) -> str:
        """
        The SQLAlchemy URL the app connects with.

        When DATABASE_URL is set it is rewritten for asyncpg: the driver is
        forced to postgresql+asyncpg, and the query string is dropped entirely.
        That last part is not cosmetic — managed providers append libpq-only
        parameters (Neon sends `?sslmode=require&channel_binding=require`) and
        asyncpg raises on them instead of ignoring them. TLS is re-applied
        separately through connect_args, which is the form asyncpg understands.
        """
        if self.database_url:
            parts = urlsplit(self.database_url)
            return urlunsplit(("postgresql+asyncpg", parts.netloc, parts.path, "", ""))
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def db_connect_args(self) -> dict:
        """
        asyncpg-native connect arguments, carrying the TLS intent that was
        stripped out of the URL above.

        libpq's `sslmode` and asyncpg's `ssl` are different parameters with
        different vocabularies; this maps between them. Anything other than an
        explicit `disable` on a remote URL gets TLS, because every managed
        provider requires it.
        """
        if not self.database_url:
            return {}

        sslmode = parse_qs(urlsplit(self.database_url).query).get("sslmode", ["require"])[0]
        if sslmode in ("disable", "allow"):
            return {}
        # asyncpg accepts "prefer"/"require"; the libpq verify-* modes need a
        # real SSLContext, and "require" is the correct floor for a demo.
        return {"ssl": "prefer" if sslmode == "prefer" else "require"}

    @property
    def async_postgres_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_direct_port}/postgres"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
