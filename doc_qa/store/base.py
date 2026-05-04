from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    text: str
    filename: str
    page_or_line: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Persist chunks with their embeddings."""
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Return top_k results by cosine similarity."""
        ...

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Fetch a single chunk by ID. Return None if not found."""
        ...

    @abstractmethod
    def list_documents(self) -> list[str]:
        """Return sorted list of unique filenames in the store."""
        ...

    @abstractmethod
    def get_chunks_for_document(self, filename: str) -> list[Chunk]:
        """Return all chunks for a given filename, sorted by chunk_index."""
        ...

    @abstractmethod
    def document_exists(self, filename: str) -> bool:
        """Return True if any chunk for this filename is already stored."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Total number of chunks stored."""
        ...
