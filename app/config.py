from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic RAG"
    debug: bool = False

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "agentic_rag"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"

    # Embeddings
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dimensions: int = 1024

    # Retrieval
    retrieval_top_k: int = 10
    rerank_top_k: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
