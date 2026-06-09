from pydantic import BaseModel


class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    session_id: str


class MessageRecord(BaseModel):
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageRecord]


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class IngestResponse(BaseModel):
    filename: str
    skipped: bool
    chunks_created: int
    embedding_time_s: float
    total_time_s: float


class StatusResponse(BaseModel):
    total_chunks: int
    document_count: int
    embedding_provider: str
    llm_provider: str
    collection_name: str
