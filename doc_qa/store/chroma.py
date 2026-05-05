from typing import TYPE_CHECKING, Optional

import chromadb

from doc_qa.store.base import Chunk, SearchResult, VectorStore

if TYPE_CHECKING:
    from chromadb import ClientAPI, Collection


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        client: Optional["ClientAPI"] = None,
    ) -> None:
        self._client: ClientAPI = (
            client if client is not None else chromadb.PersistentClient(path=persist_dir)
        )
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._chunk_to_meta(c) for c in chunks]
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        n = min(top_k, self._collection.count())
        if n == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        output: list[SearchResult] = []
        ids = results["ids"][0]
        documents = results["documents"][0]  # type: ignore[index]
        metadatas = results["metadatas"][0]  # type: ignore[index]
        distances = results["distances"][0]  # type: ignore[index]
        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances, strict=True):
            chunk = self._meta_to_chunk(chunk_id, text, meta)
            score = 1.0 - float(dist)
            output.append(SearchResult(chunk=chunk, score=score))
        return output

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        result = self._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        return self._meta_to_chunk(
            result["ids"][0],
            result["documents"][0],  # type: ignore[index]
            result["metadatas"][0],  # type: ignore[index, arg-type]
        )

    def list_documents(self) -> list[str]:
        result = self._collection.get(include=["metadatas"])
        filenames: set[str] = {
            str(m["filename"])
            for m in result["metadatas"]  # type: ignore[union-attr]
        }
        return sorted(filenames)

    def get_chunks_for_document(self, filename: str) -> list[Chunk]:
        result = self._collection.get(
            where={"filename": filename},
            include=["documents", "metadatas"],
        )
        chunks = [
            self._meta_to_chunk(cid, text, dict(meta))  # type: ignore[arg-type]
            for cid, text, meta in zip(
                result["ids"],
                result["documents"],  # type: ignore[arg-type]
                result["metadatas"],  # type: ignore[arg-type, arg-type]
                strict=True,
            )
        ]
        return sorted(chunks, key=lambda c: c.chunk_index)

    def document_exists(self, filename: str) -> bool:
        result = self._collection.get(
            where={"filename": filename},
            limit=1,
            include=["metadatas"],
        )
        return len(result["ids"]) > 0

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_to_meta(chunk: Chunk) -> dict:
        meta: dict = {
            "filename": chunk.filename,
            "page_or_line": chunk.page_or_line,
            "chunk_index": chunk.chunk_index,
        }
        for k, v in chunk.metadata.items():
            meta[f"meta_{k}"] = v
        return meta

    @staticmethod
    def _meta_to_chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
        extra = {k[5:]: v for k, v in meta.items() if k.startswith("meta_")}
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            filename=str(meta["filename"]),
            page_or_line=int(meta["page_or_line"]),
            chunk_index=int(meta["chunk_index"]),
            metadata=extra,
        )
