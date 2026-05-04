import pytest

from doc_qa.embeddings import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
    get_embedding_provider,
)


class TestSentenceTransformerProvider:
    def test_embed_documents_shape(self):
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        texts = ["hello world", "banking document", "loan application"]
        result = provider.embed_documents(texts)
        assert len(result) == len(texts)
        assert all(isinstance(v, list) for v in result)
        assert all(isinstance(x, float) for x in result[0])

    def test_embed_query_is_list_of_floats(self):
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        result = provider.embed_query("what is the interest rate?")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    def test_embed_documents_empty_list_returns_empty(self):
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        result = provider.embed_documents([])
        assert result == []

    def test_dimension_is_positive_int(self):
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        assert isinstance(provider.dimension, int)
        assert provider.dimension > 0

    def test_model_loaded_lazily(self):
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        assert provider._model is None
        provider.embed_query("trigger load")
        assert provider._model is not None


class TestOpenAIEmbeddingProvider:
    def test_raises_on_empty_api_key(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIEmbeddingProvider(api_key="")

    def test_raises_on_missing_api_key(self):
        with pytest.raises(ValueError):
            OpenAIEmbeddingProvider(api_key="")

    def test_dimension_known_model(self):
        # Construct without calling the API — just check the dimension property
        provider = OpenAIEmbeddingProvider.__new__(OpenAIEmbeddingProvider)
        provider._model = "text-embedding-3-small"
        assert provider.dimension == 1536

    def test_dimension_large_model(self):
        provider = OpenAIEmbeddingProvider.__new__(OpenAIEmbeddingProvider)
        provider._model = "text-embedding-3-large"
        assert provider.dimension == 3072

    def test_embed_documents_empty_list_returns_empty(self, mocker):
        provider = OpenAIEmbeddingProvider.__new__(OpenAIEmbeddingProvider)
        provider._model = "text-embedding-3-small"
        provider._client = mocker.MagicMock()
        result = provider.embed_documents([])
        assert result == []
        provider._client.embeddings.create.assert_not_called()


class TestGetEmbeddingProvider:
    def test_returns_sentence_transformer_by_default(self, monkeypatch):
        from config.settings import settings

        monkeypatch.setattr(settings, "embedding_provider", "sentence_transformers")
        provider = get_embedding_provider()
        assert isinstance(provider, SentenceTransformerProvider)

    def test_returns_openai_provider(self, monkeypatch):
        from config.settings import settings

        monkeypatch.setattr(settings, "embedding_provider", "openai")
        monkeypatch.setattr(settings, "openai_api_key", "sk-test-fake-key-for-type-check")
        provider = get_embedding_provider()
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_raises_on_unknown_provider(self, monkeypatch):
        from config.settings import settings

        monkeypatch.setattr(settings, "embedding_provider", "unknown_provider")
        with pytest.raises(ValueError, match="Unsupported"):
            get_embedding_provider()

    def test_is_embedding_provider_subclass(self, monkeypatch):
        from config.settings import settings

        monkeypatch.setattr(settings, "embedding_provider", "sentence_transformers")
        provider = get_embedding_provider()
        assert isinstance(provider, EmbeddingProvider)
