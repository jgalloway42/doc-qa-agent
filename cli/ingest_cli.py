"""Typer CLI for ingesting documents into the vector store."""

from pathlib import Path

import typer

app = typer.Typer(help="doc-qa ingestion CLI")


def _make_store():
    from config.settings import settings
    from doc_qa.store.chroma import ChromaVectorStore

    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )


def _make_embedder():
    from doc_qa.embeddings import get_embedding_provider

    return get_embedding_provider()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest"),  # noqa: B008
    force: bool = typer.Option(False, "--force", help="Re-ingest even if already seen"),  # noqa: B008
) -> None:
    """Ingest a single file or all supported files in a directory."""
    from doc_qa.ingestion.pipeline import ingest_directory, ingest_file
    from doc_qa.observability import setup_mlflow

    setup_mlflow()
    store = _make_store()
    embedder = _make_embedder()

    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)

    if path.is_dir():
        results = ingest_directory(path, store, embedder, force=force)
    else:
        results = [ingest_file(path, store, embedder, force=force)]

    for r in results:
        status = "skipped" if r.skipped else f"{r.chunks_created} chunks"
        typer.echo(f"  {r.filename}: {status}")

    total_new = sum(r.chunks_created for r in results)
    typer.echo(f"\nDone. {len(results)} file(s) processed, {total_new} chunk(s) added.")


@app.command(name="list")
def list_docs() -> None:
    """Print all ingested document filenames and their chunk counts."""
    store = _make_store()
    filenames = store.list_documents()

    if not filenames:
        typer.echo("No documents ingested yet.")
        return

    typer.echo(f"{'Filename':<50} {'Chunks':>6}")
    typer.echo("-" * 58)
    for filename in filenames:
        count = len(store.get_chunks_for_document(filename))
        typer.echo(f"{filename:<50} {count:>6}")
    typer.echo(f"\nTotal: {len(filenames)} document(s)")


@app.command()
def status() -> None:
    """Print store stats: total chunks, embedding provider, collection name."""
    from config.settings import settings

    store = _make_store()
    total = store.count()
    typer.echo(f"Embedding provider : {settings.embedding_provider}")
    typer.echo(f"Chroma collection  : {settings.chroma_collection_name}")
    typer.echo(f"Chroma persist dir : {settings.chroma_persist_dir}")
    typer.echo(f"Total chunks       : {total}")


if __name__ == "__main__":
    app()
