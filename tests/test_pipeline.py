import pytest

from doc_qa.ingestion.pipeline import IngestionResult, ingest_directory, ingest_file


def _make_txt(tmp_path, name="sample.txt", content=None):
    if content is None:
        content = "Banking document content.\n" * 20
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# ingest_file happy path
# ---------------------------------------------------------------------------


def test_ingest_file_returns_correct_chunk_count(tmp_path, chroma_store, fake_embedder):
    path = _make_txt(tmp_path)
    result = ingest_file(
        path, chroma_store, fake_embedder, hash_store_path=tmp_path / "hashes.json"
    )

    assert isinstance(result, IngestionResult)
    assert result.filename == path.name
    assert not result.skipped
    assert result.chunks_created >= 1
    assert result.embedding_time_s >= 0.0
    assert result.total_time_s >= 0.0


def test_ingest_file_skips_duplicate(tmp_path, chroma_store, fake_embedder):
    path = _make_txt(tmp_path)
    hash_store = tmp_path / "hashes.json"

    first = ingest_file(path, chroma_store, fake_embedder, hash_store_path=hash_store)
    second = ingest_file(path, chroma_store, fake_embedder, hash_store_path=hash_store)

    assert not first.skipped
    assert second.skipped
    assert second.chunks_created == 0


def test_ingest_file_force_flag_overrides_dedup(tmp_path, chroma_store, fake_embedder):
    path = _make_txt(tmp_path)
    hash_store = tmp_path / "hashes.json"

    ingest_file(path, chroma_store, fake_embedder, hash_store_path=hash_store)
    second = ingest_file(path, chroma_store, fake_embedder, force=True, hash_store_path=hash_store)

    assert not second.skipped
    assert second.chunks_created >= 1


def test_ingest_file_empty_file_creates_zero_chunks(tmp_path, chroma_store, fake_embedder):
    path = _make_txt(tmp_path, content="")
    result = ingest_file(
        path, chroma_store, fake_embedder, hash_store_path=tmp_path / "hashes.json"
    )

    assert result.chunks_created == 0
    assert not result.skipped


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_ingest_file_raises_for_missing_file(tmp_path, chroma_store, fake_embedder):
    missing = tmp_path / "no_such_file.txt"
    with pytest.raises(FileNotFoundError):
        ingest_file(missing, chroma_store, fake_embedder, hash_store_path=tmp_path / "hashes.json")


# ---------------------------------------------------------------------------
# MLflow tracking
# ---------------------------------------------------------------------------


def test_ingest_file_creates_langsmith_trace(tmp_path, chroma_store, fake_embedder, mocker):
    mock_log = mocker.patch("doc_qa.ingestion.pipeline.log_ingestion_run")
    path = _make_txt(tmp_path)
    ingest_file(path, chroma_store, fake_embedder, hash_store_path=tmp_path / "hashes.json")

    mock_log.assert_called_once()
    kwargs = mock_log.call_args.kwargs
    assert kwargs["filename"] == path.name
    assert kwargs["skipped"] is False


def test_ingest_file_skipped_creates_langsmith_trace(tmp_path, chroma_store, fake_embedder, mocker):
    mock_log = mocker.patch("doc_qa.ingestion.pipeline.log_ingestion_run")
    path = _make_txt(tmp_path)
    hash_store = tmp_path / "hashes.json"

    ingest_file(path, chroma_store, fake_embedder, hash_store_path=hash_store)
    ingest_file(path, chroma_store, fake_embedder, hash_store_path=hash_store)

    assert mock_log.call_count == 2
    second_kwargs = mock_log.call_args.kwargs
    assert second_kwargs["skipped"] is True


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------


def test_ingest_directory_processes_all_supported_files(tmp_path, chroma_store, fake_embedder):
    (tmp_path / "doc1.txt").write_text("First document with enough text to chunk.\n" * 20)
    (tmp_path / "doc2.txt").write_text("Second document with enough text to chunk.\n" * 20)
    (tmp_path / "ignored.xyz").write_text("Should be ignored.")

    hash_store = tmp_path / "hashes.json"
    results = ingest_directory(tmp_path, chroma_store, fake_embedder, hash_store_path=hash_store)

    filenames = {r.filename for r in results}
    assert "doc1.txt" in filenames
    assert "doc2.txt" in filenames
    assert "ignored.xyz" not in filenames
    assert len(results) == 2


def test_ingest_directory_empty_dir(tmp_path, chroma_store, fake_embedder):
    results = ingest_directory(
        tmp_path, chroma_store, fake_embedder, hash_store_path=tmp_path / "hashes.json"
    )
    assert results == []
