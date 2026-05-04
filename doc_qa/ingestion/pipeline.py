import logging
import time
from dataclasses import dataclass
from pathlib import Path

from doc_qa.embeddings import EmbeddingProvider
from doc_qa.ingestion.chunker import chunk_pages
from doc_qa.ingestion.dedup import HASH_STORE_PATH, is_duplicate, register_file
from doc_qa.ingestion.parsers import SUPPORTED_EXTENSIONS, parse_file
from doc_qa.observability import log_ingestion_run
from doc_qa.store.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    filename: str
    skipped: bool
    chunks_created: int
    embedding_time_s: float
    total_time_s: float
    mlflow_run_id: str


def ingest_file(
    path: Path,
    store: VectorStore,
    embedder: EmbeddingProvider,
    force: bool = False,
    hash_store_path: Path = HASH_STORE_PATH,
) -> IngestionResult:
    """
    Full ingestion pipeline for a single file.
    Raises FileNotFoundError if path does not exist.
    Raises ValueError for unsupported file extensions.
    """
    from config.settings import settings

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    t_start = time.monotonic()

    if not force and is_duplicate(path, hash_store_path):
        run_id = log_ingestion_run(
            filename=path.name,
            chunks_created=0,
            embedding_time_s=0.0,
            total_time_s=0.0,
            skipped=True,
            file_hash="",
            embedding_provider=str(settings.embedding_provider),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        return IngestionResult(
            filename=path.name,
            skipped=True,
            chunks_created=0,
            embedding_time_s=0.0,
            total_time_s=time.monotonic() - t_start,
            mlflow_run_id=run_id,
        )

    pages = parse_file(path)

    chunks = chunk_pages(
        pages,
        filename=path.name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_chars=settings.min_chunk_chars,
    )

    t_embed = time.monotonic()
    try:
        embeddings = embedder.embed_documents([c.text for c in chunks]) if chunks else []
    except Exception as exc:
        logger.exception("Embedding failed for %s", path.name)
        log_ingestion_run(
            filename=path.name,
            chunks_created=0,
            embedding_time_s=0.0,
            total_time_s=time.monotonic() - t_start,
            skipped=False,
            file_hash="",
            embedding_provider=str(settings.embedding_provider),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        raise exc
    embedding_time_s = time.monotonic() - t_embed

    if chunks:
        store.add_chunks(chunks, embeddings)

    file_hash = register_file(path, hash_store_path)
    total_time_s = time.monotonic() - t_start

    run_id = log_ingestion_run(
        filename=path.name,
        chunks_created=len(chunks),
        embedding_time_s=embedding_time_s,
        total_time_s=total_time_s,
        skipped=False,
        file_hash=file_hash,
        embedding_provider=str(settings.embedding_provider),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return IngestionResult(
        filename=path.name,
        skipped=False,
        chunks_created=len(chunks),
        embedding_time_s=embedding_time_s,
        total_time_s=total_time_s,
        mlflow_run_id=run_id,
    )


def ingest_directory(
    dir_path: Path,
    store: VectorStore,
    embedder: EmbeddingProvider,
    force: bool = False,
    hash_store_path: Path = HASH_STORE_PATH,
) -> list[IngestionResult]:
    """Ingest all supported files in dir_path. Non-recursive."""
    results = []
    for path in sorted(dir_path.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            result = ingest_file(path, store, embedder, force=force, hash_store_path=hash_store_path)
            results.append(result)
    return results
