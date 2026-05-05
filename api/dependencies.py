from config.settings import settings
from doc_qa.agent.runner import AgentRunner, AgentSession
from doc_qa.embeddings import get_embedding_provider
from doc_qa.observability import setup_mlflow
from doc_qa.store.chroma import ChromaVectorStore

setup_mlflow()

_store = ChromaVectorStore(
    persist_dir=settings.chroma_persist_dir,
    collection_name=settings.chroma_collection_name,
)
_embedder = get_embedding_provider()
_runner = AgentRunner(store=_store, embedder=_embedder)
_sessions: dict[str, AgentSession] = {}


def get_store() -> ChromaVectorStore:
    return _store


def get_embedder():
    return _embedder


def get_runner() -> AgentRunner:
    return _runner


def get_sessions() -> dict[str, AgentSession]:
    return _sessions
