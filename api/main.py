import logging
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage

from api.dependencies import get_embedder, get_runner, get_sessions, get_store
from api.models import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    IngestResponse,
    MessageRecord,
    SessionHistoryResponse,
    SessionResponse,
    StatusResponse,
)
from config.settings import settings
from doc_qa.agent.runner import AgentRunner, AgentSession
from doc_qa.embeddings import EmbeddingProvider
from doc_qa.ingestion.pipeline import ingest_file
from doc_qa.store.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)

app = FastAPI(title="doc-qa-agent API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status(store: ChromaVectorStore = Depends(get_store)):  # noqa: B008
    docs = store.list_documents()
    return StatusResponse(
        total_chunks=store.count(),
        document_count=len(docs),
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
        collection_name=settings.chroma_collection_name,
    )


@app.post("/sessions", response_model=SessionResponse)
def create_session(
    runner: AgentRunner = Depends(get_runner),  # noqa: B008
    sessions: dict[str, AgentSession] = Depends(get_sessions),  # noqa: B008
):
    session = runner.new_session()
    sessions[session.session_id] = session
    return SessionResponse(session_id=session.session_id, mlflow_run_id=session.mlflow_run_id)


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(
    session_id: str,
    body: ChatRequest,
    runner: AgentRunner = Depends(get_runner),  # noqa: B008
    sessions: dict[str, AgentSession] = Depends(get_sessions),  # noqa: B008
):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        response = runner.run_turn(session, body.message)
    except Exception as exc:
        logger.exception("run_turn failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(response=response, session_id=session_id)


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
def get_session(
    session_id: str,
    sessions: dict[str, AgentSession] = Depends(get_sessions),  # noqa: B008
):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = []
    for msg in session.history:
        if isinstance(msg, HumanMessage):
            messages.append(MessageRecord(role="user", content=str(msg.content)))
        elif isinstance(msg, AIMessage):
            messages.append(MessageRecord(role="assistant", content=str(msg.content)))
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents(store: ChromaVectorStore = Depends(get_store)):  # noqa: B008
    docs = store.list_documents()
    return [
        DocumentInfo(filename=f, chunk_count=len(store.get_chunks_for_document(f))) for f in docs
    ]


@app.post("/documents/ingest", response_model=IngestResponse)
def ingest_document(
    file: UploadFile,
    store: ChromaVectorStore = Depends(get_store),  # noqa: B008
    embedder: EmbeddingProvider = Depends(get_embedder),  # noqa: B008
):
    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = Path(tmp.name)

    try:
        result = ingest_file(path=tmp_path, store=store, embedder=embedder, filename=file.filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        filename=file.filename or tmp_path.name,
        skipped=result.skipped,
        chunks_created=result.chunks_created,
        embedding_time_s=result.embedding_time_s,
        total_time_s=result.total_time_s,
    )
