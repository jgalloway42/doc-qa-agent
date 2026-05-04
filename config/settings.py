from enum import StrEnum

from pydantic_settings import BaseSettings


class EmbeddingProvider(StrEnum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class Settings(BaseSettings):
    # Embedding
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    sentence_transformers_model: str = "all-MiniLM-L6-v2"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    # LLM (agent reasoning)
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    openai_chat_model: str = "gpt-4o"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "doc_qa"

    # Ingestion
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_chars: int = 100
    pdf_ocr_threshold: int = 50

    # MLflow
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "doc-qa-agent"

    # Retrieval
    default_top_k: int = 5
    retrieval_score_threshold: float = 0.0
    max_tool_retries: int = 2

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
