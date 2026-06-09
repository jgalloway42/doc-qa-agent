"""LangSmith observability helpers. Agent turns are traced automatically by LangGraph."""

import logging
import os

from langsmith import traceable

logger = logging.getLogger(__name__)


def setup_langsmith() -> None:
    """Set LangSmith env vars so LangGraph auto-traces every graph.invoke() call.

    Tracing is only enabled when both langchain_tracing_enabled is True and
    a non-empty langsmith_api_key is configured — prevents 401 errors when
    the key is absent (e.g., in tests or local dev without a LangSmith account).
    """
    from config.settings import settings

    if not settings.langchain_tracing_enabled or not settings.langsmith_api_key:
        return
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)


@traceable(run_type="chain", name="ingest_document")
def log_ingestion_run(
    filename: str,
    chunks_created: int,
    embedding_time_s: float,
    total_time_s: float,
    skipped: bool,
    file_hash: str,
    embedding_provider: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    """Create a LangSmith trace for a single ingestion. All args captured as inputs."""
    return {
        "chunks_created": chunks_created,
        "embedding_time_s": round(embedding_time_s, 3),
        "total_time_s": round(total_time_s, 3),
        "skipped": skipped,
    }
