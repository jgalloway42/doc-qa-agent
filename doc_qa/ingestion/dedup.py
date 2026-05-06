import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HASH_STORE_PATH = Path(".ingested_hashes.json")


def compute_file_hash(path: Path) -> str:
    """Return hex SHA-256 digest of file contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_hash_store(store_path: Path = HASH_STORE_PATH) -> dict[str, str]:
    """Load {filename: sha256_hex} mapping. Return {} if not found or corrupted."""
    if not store_path.exists():
        return {}
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Hash store root must be a JSON object")
        return data
    except (json.JSONDecodeError, ValueError, OSError):
        logger.warning("Hash store at %s is corrupted; treating as empty.", store_path)
        return {}


def save_hash_store(store: dict[str, str], store_path: Path = HASH_STORE_PATH) -> None:
    """Persist hash store to store_path."""
    store_path.write_text(json.dumps(store, indent=2), encoding="utf-8")


def is_duplicate(path: Path, store_path: Path = HASH_STORE_PATH) -> bool:
    """Return True if path's hash already exists in the hash store.

    Detects duplicates by content hash, not filename — renaming a file
    and re-ingesting it is still considered a duplicate.
    """
    file_hash = compute_file_hash(path)
    return file_hash in load_hash_store(store_path).values()


def register_file(path: Path, store_path: Path = HASH_STORE_PATH, filename: str | None = None) -> str:
    """Add path to hash store and persist. Return the hash."""
    file_hash = compute_file_hash(path)
    store = load_hash_store(store_path)
    store[filename or path.name] = file_hash
    save_hash_store(store, store_path)
    return file_hash
