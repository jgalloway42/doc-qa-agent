import numpy as np
import pytest

from doc_qa.store.base import Chunk
from doc_qa.store.chroma import ChromaVectorStore


def _make_embedding(seed: int, dim: int = 384) -> list[float]:
    return np.random.default_rng(seed).random(dim).tolist()


def _populate(store: ChromaVectorStore, chunks: list[Chunk]) -> list[list[float]]:
    embeddings = [_make_embedding(i) for i in range(len(chunks))]
    store.add_chunks(chunks, embeddings)
    return embeddings


class TestAddAndSearch:
    def test_add_and_search_returns_results(self, chroma_store, sample_chunks):
        embeddings = _populate(chroma_store, sample_chunks[:3])
        results = chroma_store.search(embeddings[0], top_k=3)
        assert len(results) >= 1
        assert all(hasattr(r, "chunk") and hasattr(r, "score") for r in results)

    def test_search_respects_top_k(self, chroma_store, sample_chunks):
        _populate(chroma_store, sample_chunks)
        results = chroma_store.search(_make_embedding(99), top_k=2)
        assert len(results) <= 2

    def test_search_returns_empty_on_empty_store(self, chroma_store):
        results = chroma_store.search(_make_embedding(0), top_k=5)
        assert results == []

    def test_search_scores_are_floats(self, chroma_store, sample_chunks):
        embeddings = _populate(chroma_store, sample_chunks)
        results = chroma_store.search(embeddings[0], top_k=3)
        assert all(isinstance(r.score, float) for r in results)


class TestListDocuments:
    def test_list_documents_returns_unique_filenames(self, chroma_store, sample_chunks):
        _populate(chroma_store, sample_chunks)
        docs = chroma_store.list_documents()
        assert docs == sorted({"doc_a.pdf", "doc_b.txt"})

    def test_list_documents_returns_sorted(self, chroma_store, sample_chunks):
        _populate(chroma_store, sample_chunks)
        docs = chroma_store.list_documents()
        assert docs == sorted(docs)

    def test_list_documents_empty_store(self, chroma_store):
        assert chroma_store.list_documents() == []


class TestGetChunksForDocument:
    def test_get_chunks_for_document_sorted_by_index(self, chroma_store, sample_chunks):
        _populate(chroma_store, sample_chunks)
        chunks = chroma_store.get_chunks_for_document("doc_a.pdf")
        assert len(chunks) == 3
        assert [c.chunk_index for c in chunks] == [0, 1, 2]

    def test_get_chunks_for_document_correct_filename(self, chroma_store, sample_chunks):
        _populate(chroma_store, sample_chunks)
        chunks = chroma_store.get_chunks_for_document("doc_b.txt")
        assert all(c.filename == "doc_b.txt" for c in chunks)

    def test_get_chunks_for_missing_document_returns_empty(self, chroma_store, sample_chunks):
        _populate(chroma_store, sample_chunks)
        assert chroma_store.get_chunks_for_document("nonexistent.pdf") == []


class TestDocumentExists:
    def test_document_exists_true(self, chroma_store, sample_chunks):
        _populate(chroma_store, sample_chunks)
        assert chroma_store.document_exists("doc_a.pdf") is True

    def test_document_exists_false(self, chroma_store):
        assert chroma_store.document_exists("ghost.pdf") is False


class TestGetChunk:
    def test_get_chunk_by_id(self, chroma_store, sample_chunk):
        emb = [_make_embedding(0)]
        chroma_store.add_chunks([sample_chunk], emb)
        result = chroma_store.get_chunk("doc_a.pdf::0000")
        assert result is not None
        assert result.chunk_id == "doc_a.pdf::0000"
        assert result.filename == "doc_a.pdf"

    def test_get_chunk_returns_none_for_missing_id(self, chroma_store):
        assert chroma_store.get_chunk("does_not_exist::9999") is None


class TestCount:
    def test_count_reflects_added_chunks(self, chroma_store, sample_chunks):
        assert chroma_store.count() == 0
        _populate(chroma_store, sample_chunks[:3])
        assert chroma_store.count() == 3
        _populate(chroma_store, sample_chunks[3:])
        assert chroma_store.count() == 5

    def test_count_empty_store(self, chroma_store):
        assert chroma_store.count() == 0
