from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Database Config
    db_user: str = Field(default="postgres")
    db_password: str = Field(default="password")
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5433)
    db_name: str = Field(default="tutorial_db")
    
    # Embedding config
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5")
    embedding_dimension: int = Field(default=768)
    
    # Chunker config
    window_size: int = Field(default=3)
    threshold_factor: float = Field(default=0.8)
    min_sentences: int = Field(default=2)
    min_words: int = Field(default=50)
    max_tokens: int = Field(default=400)
    overlap_tokens: int = Field(default=100)

    # LLM Config
    llm_provider: str = Field(default="gemini")
    llm_api_key: str = Field(default="")
    llm_api_base: str = Field(default="https://generativelanguage.googleapis.com/v1beta")
    llm_model: str = Field(default="gemini-1.5-flash")
    llm_temperature: float = Field(default=0.0)
    agent_max_iterations: int = Field(default=5)


    @property
    def async_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def async_postgres_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/postgres"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
