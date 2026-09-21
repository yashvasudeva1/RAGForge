"""
RAGForge application configuration.

All settings are read from environment variables (or a .env file).
Use the `get_settings()` function to access the singleton instance.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for RAGForge.

    Every field maps directly to an environment variable of the same name
    (case-insensitive). Sensitive fields are marked with `repr=False` to
    prevent accidental logging.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    app_name: str = Field(default="RAGForge", description="Application display name.")
    app_version: str = Field(default="0.1.0", description="Semantic version string.")
    debug: bool = Field(default=False, description="Enable debug mode and verbose logging.")
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment.",
    )

    # ------------------------------------------------------------------ #
    # API Server
    # ------------------------------------------------------------------ #
    host: str = Field(default="0.0.0.0", description="Bind host for the FastAPI server.")
    port: int = Field(default=8000, description="Bind port for the FastAPI server.")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins for the frontend.",
    )
    api_prefix: str = Field(default="/api", description="Global API route prefix.")

    # ------------------------------------------------------------------ #
    # LLM Provider Keys
    # ------------------------------------------------------------------ #
    openai_api_key: str | None = Field(default=None, repr=False, description="OpenAI API key.")
    openai_org_id: str | None = Field(default=None, repr=False, description="OpenAI organization ID.")
    openai_base_url: str | None = Field(
        default=None,
        description="Custom OpenAI-compatible base URL (e.g. for Azure or local proxies).",
    )

    anthropic_api_key: str | None = Field(default=None, repr=False, description="Anthropic API key.")

    google_api_key: str | None = Field(default=None, repr=False, description="Google Generative AI API key.")
    google_project_id: str | None = Field(default=None, description="Google Cloud project ID.")

    groq_api_key: str | None = Field(default=None, repr=False, description="Groq API key.")
    mistral_api_key: str | None = Field(default=None, repr=False, description="Mistral AI API key.")
    cohere_api_key: str | None = Field(default=None, repr=False, description="Cohere API key.")
    voyage_api_key: str | None = Field(default=None, repr=False, description="VoyageAI API key.")
    huggingface_api_key: str | None = Field(default=None, repr=False, description="HuggingFace Hub token.")

    # ------------------------------------------------------------------ #
    # Embedding Defaults
    # ------------------------------------------------------------------ #
    default_embedding_provider: str = Field(
        default="openai",
        description="Default embedding provider: openai | google | huggingface | cohere | voyage | local.",
    )
    default_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Default embedding model name.",
    )
    default_embedding_dimensions: int = Field(
        default=1536,
        description="Dimensionality of the default embedding model.",
    )

    # ------------------------------------------------------------------ #
    # Chunking Defaults
    # ------------------------------------------------------------------ #
    default_chunk_size: int = Field(default=1000, description="Default chunk size in characters.")
    default_chunk_overlap: int = Field(default=200, description="Default chunk overlap in characters.")
    default_chunker: str = Field(
        default="recursive",
        description="Default chunking strategy: character | recursive | token | sentence | markdown | semantic.",
    )

    # ------------------------------------------------------------------ #
    # Vector Store
    # ------------------------------------------------------------------ #
    default_vectorstore: str = Field(
        default="chroma",
        description="Default vector store backend: chroma | qdrant | faiss | pinecone | weaviate | milvus | lancedb.",
    )

    # Chroma
    chroma_host: str = Field(default="localhost", description="ChromaDB host.")
    chroma_port: int = Field(default=8001, description="ChromaDB HTTP port.")
    chroma_persist_dir: str = Field(default="./data/chroma", description="ChromaDB local persistence directory.")

    # Qdrant
    qdrant_url: str | None = Field(default=None, description="Qdrant server URL.")
    qdrant_api_key: str | None = Field(default=None, repr=False, description="Qdrant Cloud API key.")
    qdrant_collection: str = Field(default="ragforge", description="Default Qdrant collection name.")

    # Pinecone
    pinecone_api_key: str | None = Field(default=None, repr=False, description="Pinecone API key.")
    pinecone_index: str | None = Field(default=None, description="Default Pinecone index name.")
    pinecone_environment: str | None = Field(default=None, description="Pinecone environment/region.")

    # Weaviate
    weaviate_url: str | None = Field(default=None, description="Weaviate server URL.")
    weaviate_api_key: str | None = Field(default=None, repr=False, description="Weaviate API key.")

    # Milvus / Zilliz
    milvus_uri: str | None = Field(default=None, description="Milvus or Zilliz URI.")
    milvus_token: str | None = Field(default=None, repr=False, description="Milvus or Zilliz API token.")

    # LanceDB
    lancedb_uri: str = Field(default="./data/lancedb", description="LanceDB local or cloud URI.")

    # ------------------------------------------------------------------ #
    # Retrieval Defaults
    # ------------------------------------------------------------------ #
    default_retrieval_k: int = Field(default=5, description="Default number of chunks to retrieve.")
    default_retrieval_strategy: str = Field(
        default="dense",
        description="Default retrieval strategy: dense | sparse | hybrid | mmr.",
    )

    # ------------------------------------------------------------------ #
    # Reranking Defaults
    # ------------------------------------------------------------------ #
    default_reranker: str | None = Field(
        default=None,
        description="Default reranker: cross_encoder | cohere | voyage | llm | None (disabled).",
    )
    default_rerank_top_n: int = Field(default=3, description="Default number of results after reranking.")

    # ------------------------------------------------------------------ #
    # Generation Defaults
    # ------------------------------------------------------------------ #
    default_llm_provider: str = Field(
        default="openai",
        description="Default LLM provider: openai | anthropic | google | groq | ollama | mistral.",
    )
    default_llm_model: str = Field(default="gpt-4o-mini", description="Default LLM model name.")
    default_temperature: float = Field(default=0.0, description="Default LLM sampling temperature.")
    default_max_tokens: int = Field(default=1024, description="Default maximum tokens in LLM response.")

    # ------------------------------------------------------------------ #
    # Ollama (local LLMs)
    # ------------------------------------------------------------------ #
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server base URL.")

    # ------------------------------------------------------------------ #
    # Pipeline Execution
    # ------------------------------------------------------------------ #
    max_concurrent_pipelines: int = Field(
        default=5,
        description="Maximum number of pipeline executions allowed concurrently.",
    )
    pipeline_timeout_seconds: int = Field(
        default=300,
        description="Maximum allowed wall-clock time per pipeline execution.",
    )
    pipeline_storage_dir: str = Field(
        default="./data/pipelines",
        description="Directory for persisting saved pipeline JSON configs.",
    )

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    enable_tracing: bool = Field(default=False, description="Enable OpenTelemetry tracing.")
    otel_exporter_endpoint: str | None = Field(
        default=None,
        description="OpenTelemetry exporter endpoint (e.g. OTLP gRPC URL).",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Root log level.",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("default_temperature")
    @classmethod
    def temperature_range(cls, v: float) -> float:
        """Ensure temperature is within the valid [0.0, 2.0] range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("default_temperature must be between 0.0 and 2.0")
        return v

    @field_validator("default_embedding_dimensions")
    @classmethod
    def dimensions_positive(cls, v: int) -> int:
        """Ensure embedding dimensions are a positive integer."""
        if v <= 0:
            raise ValueError("default_embedding_dimensions must be a positive integer")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached singleton Settings instance.

    Using lru_cache ensures the .env file is read only once per process.
    Call ``get_settings.cache_clear()`` in tests to reset between cases.
    """
    return Settings()
