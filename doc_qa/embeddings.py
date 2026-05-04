from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents. Returns list of float vectors."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension this provider produces."""
        ...


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: _ST | None = None

    def _get_model(self) -> "_ST":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._get_model().encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._get_model().encode(text, convert_to_numpy=True).tolist()

    @property
    def dimension(self) -> int:
        dim = self._get_model().get_embedding_dimension()
        return int(dim) if dim is not None else 384


class OpenAIEmbeddingProvider(EmbeddingProvider):
    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai. "
                "Set it in your .env file or environment."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    @property
    def dimension(self) -> int:
        return self._DIMENSIONS.get(self._model, 1536)


def get_embedding_provider() -> EmbeddingProvider:
    """Factory: reads settings.embedding_provider and returns the right instance."""
    from config.settings import settings

    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformerProvider(settings.sentence_transformers_model)
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings.openai_api_key, settings.openai_embedding_model)
    raise ValueError(f"Unsupported embedding_provider: {settings.embedding_provider}")
