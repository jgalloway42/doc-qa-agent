import mlflow
import numpy as np
import pytest

from doc_qa.embeddings import EmbeddingProvider
from doc_qa.store.base import Chunk
from doc_qa.store.chroma import ChromaVectorStore

# ---------------------------------------------------------------------------
# MLflow — redirect to a temp dir so tests never pollute ./mlruns
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mlflow_tmp(tmp_path):
    uri = f"sqlite:///{tmp_path}/test_mlruns.db"
    mlflow.set_tracking_uri(uri)
    yield
    mlflow.set_tracking_uri("sqlite:///mlruns.db")


# ---------------------------------------------------------------------------
# Fake embedding provider — fixed-length random vectors (dim=384)
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider(EmbeddingProvider):
    DIM = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        rng = np.random.default_rng(42)
        return [rng.random(self.DIM).tolist() for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        rng = np.random.default_rng(hash(text) % (2**31))
        return rng.random(self.DIM).tolist()

    @property
    def dimension(self) -> int:
        return self.DIM


@pytest.fixture
def fake_embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


# ---------------------------------------------------------------------------
# ChromaDB store backed by an in-memory EphemeralClient
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_store() -> ChromaVectorStore:
    import uuid

    import chromadb

    client = chromadb.EphemeralClient()
    # Unique collection name per test so EphemeralClient singletons don't bleed state
    collection_name = f"test_{uuid.uuid4().hex}"
    return ChromaVectorStore("", collection_name, client=client)


# ---------------------------------------------------------------------------
# Sample chunks
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="doc_a.pdf::0000",
        text="The interest rate on this loan is 6.75% per annum.",
        filename="doc_a.pdf",
        page_or_line=1,
        chunk_index=0,
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="doc_a.pdf::0000",
            text="Loan amount is $150,000.",
            filename="doc_a.pdf",
            page_or_line=1,
            chunk_index=0,
        ),
        Chunk(
            chunk_id="doc_a.pdf::0001",
            text="Interest rate is 6.75% per annum.",
            filename="doc_a.pdf",
            page_or_line=2,
            chunk_index=1,
        ),
        Chunk(
            chunk_id="doc_a.pdf::0002",
            text="Monthly payment is $972.90.",
            filename="doc_a.pdf",
            page_or_line=3,
            chunk_index=2,
        ),
        Chunk(
            chunk_id="doc_b.txt::0000",
            text="Borrower must maintain adequate insurance coverage.",
            filename="doc_b.txt",
            page_or_line=1,
            chunk_index=0,
        ),
        Chunk(
            chunk_id="doc_b.txt::0001",
            text="Late payments incur a 5% penalty after 15 days.",
            filename="doc_b.txt",
            page_or_line=2,
            chunk_index=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm(mocker):
    from langchain_core.messages import AIMessage

    llm = mocker.MagicMock()
    llm.invoke.return_value = AIMessage(content="test response")
    llm.bind_tools.return_value = llm
    return llm


# ---------------------------------------------------------------------------
# Temp docs directory with sample files
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_docs_dir(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Line one\nLine two\nLine three\n")

    csv_file = tmp_path / "test.csv"
    csv_file.write_text("product,rate\n30yr_fixed,6.75\n15yr_fixed,6.25\n")

    # Minimal valid PDF (no OCR needed — just text-based)
    pdf_file = tmp_path / "test.pdf"
    try:
        import io

        from reportlab.pdfgen import canvas as rl_canvas

        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf)
        c.drawString(72, 720, "Test PDF content for unit tests.")
        c.save()
        pdf_file.write_bytes(buf.getvalue())
    except ImportError:
        # Fallback: write a 1-byte placeholder so the path exists
        pdf_file.write_bytes(b"%PDF-1.4\n")

    return tmp_path
