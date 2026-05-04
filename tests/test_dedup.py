from pathlib import Path

import pytest

from doc_qa.ingestion.dedup import (
    compute_file_hash,
    is_duplicate,
    load_hash_store,
    register_file,
    save_hash_store,
)


@pytest.fixture
def store_path(tmp_path) -> Path:
    return tmp_path / ".ingested_hashes.json"


@pytest.fixture
def sample_file(tmp_path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text("This is a sample document for dedup testing.")
    return f


class TestComputeFileHash:
    def test_deterministic(self, sample_file):
        h1 = compute_file_hash(sample_file)
        h2 = compute_file_hash(sample_file)
        assert h1 == h2

    def test_returns_hex_string(self, sample_file):
        h = compute_file_hash(sample_file)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 = 32 bytes = 64 hex chars
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        h = compute_file_hash(f)
        assert len(h) == 64


class TestHashStore:
    def test_load_returns_empty_when_missing(self, store_path):
        assert load_hash_store(store_path) == {}

    def test_save_and_load_roundtrip(self, store_path):
        data = {"doc.pdf": "abc123", "policy.txt": "def456"}
        save_hash_store(data, store_path)
        loaded = load_hash_store(store_path)
        assert loaded == data

    def test_corrupted_store_treated_as_empty(self, store_path):
        store_path.write_text("{not valid json", encoding="utf-8")
        result = load_hash_store(store_path)
        assert result == {}

    def test_non_dict_root_treated_as_empty(self, store_path):
        store_path.write_text("[1, 2, 3]", encoding="utf-8")
        result = load_hash_store(store_path)
        assert result == {}


class TestIsDuplicate:
    def test_false_for_new_file(self, sample_file, store_path):
        assert is_duplicate(sample_file, store_path) is False

    def test_true_after_register(self, sample_file, store_path):
        register_file(sample_file, store_path)
        assert is_duplicate(sample_file, store_path) is True

    def test_detects_by_hash_not_name(self, tmp_path, store_path):
        original = tmp_path / "original.txt"
        original.write_text("Identical content in both files.")
        register_file(original, store_path)

        renamed = tmp_path / "renamed_copy.txt"
        renamed.write_text("Identical content in both files.")
        # Different filename, same content → still a duplicate
        assert is_duplicate(renamed, store_path) is True

    def test_different_content_not_duplicate(self, tmp_path, store_path):
        f1 = tmp_path / "doc1.txt"
        f2 = tmp_path / "doc2.txt"
        f1.write_text("First document content.")
        f2.write_text("Second document content — completely different.")
        register_file(f1, store_path)
        assert is_duplicate(f2, store_path) is False


class TestRegisterFile:
    def test_register_returns_hash(self, sample_file, store_path):
        h = register_file(sample_file, store_path)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_register_persists_to_store(self, sample_file, store_path):
        h = register_file(sample_file, store_path)
        store = load_hash_store(store_path)
        assert h in store.values()

    def test_register_multiple_files(self, tmp_path, store_path):
        files = []
        for i in range(3):
            f = tmp_path / f"doc{i}.txt"
            f.write_text(f"Content of document {i}.")
            files.append(f)
            register_file(f, store_path)

        store = load_hash_store(store_path)
        assert len(store) == 3
