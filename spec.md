# Project Specification: `doc-qa-agent`

---

## 0. Purpose of This Document

This document is the **as-built specification** for `doc-qa-agent`. It reflects the
system as implemented, including the FastAPI backend added in phase 2 and the MLflow
SQLite migration. Use it as the authoritative reference for architecture decisions,
module boundaries, and configuration defaults.

All configuration values belong in `config/settings.py` — never hardcode them in
module bodies.

---

## 1. Project Overview

`doc-qa-agent` is a production-quality agentic document Q&A system built for a
banking context. It ingests a corpus of loan and banking documents (PDFs with OCR
fallback, Markdown, CSV, JSON, plain text, DOCX), stores chunked embeddings in
ChromaDB, and exposes a LangGraph
agent through a Streamlit UI. The agent reasons over the corpus using five tools —
search, list, summarize, classify, and calculate — and supports multi-turn
conversation within a session. MLflow tracks every ingestion run and every agent
interaction. The architecture uses a `VectorStore` abstract base class so ChromaDB
can be swapped for Snowflake Cortex without touching agent or ingestion logic. The
embedding provider is configurable (default: local `sentence-transformers`; optional:
OpenAI). CI runs on GitHub Actions: lint → type-check → test → coverage gate.

---

## 2. Bugs / Issues in Existing Code

_Not applicable — this is a greenfield project._

---

## 3. Architecture

### 3.1 Separation of Concerns

| Layer | Module(s) | Owns | Must NOT import |
|---|---|---|---|
| Configuration | `config/settings.py` | All env vars, constants, provider selection | Any app module |
| Storage abstraction | `doc_qa/store/base.py` | `VectorStore` ABC | Any concrete store |
| Storage implementation | `doc_qa/store/chroma.py` | ChromaDB adapter | `agent`, `ingestion`, `ui`, `api` |
| Embedding | `doc_qa/embeddings.py` | `EmbeddingProvider` ABC + concrete providers | `store`, `agent`, `ui`, `api` |
| Ingestion | `doc_qa/ingestion/` | File parsing, chunking, dedup, embedding, MLflow run | `agent`, `ui`, `api` |
| Agent tools | `doc_qa/agent/tools.py` | All tool functions with `@tool` decorators | `ui`, `ingestion`, `api` |
| Agent graph | `doc_qa/agent/graph.py` | LangGraph `StateGraph` definition, `validate_tool_results` node, compilation | `ui`, `ingestion`, `api` |
| Agent runner | `doc_qa/agent/runner.py` | Session memory, `run_turn()` entry point | `ui`, `api` |
| Observability | `doc_qa/observability.py` | MLflow helpers, decorators, retrieval quality logging | Nothing app-specific |
| API layer | `api/main.py` | REST endpoint handlers, HTTP request/response serialization | `graph.py`, `store` directly |
| API dependencies | `api/dependencies.py` | Singleton lifecycle (store, embedder, runner), in-memory session store | `ui` |
| UI | `ui/app.py` | Streamlit session state, layout, HTTP calls to API | Everything except `config/settings.py` and `requests` |
| CLI | `cli/ingest_cli.py` | `typer` CLI entry point for ingestion | `ui`, `api` |

**Hard cross-boundary rules:**
- No module outside `doc_qa/store/` imports `chromadb` directly.
- No module outside `doc_qa/embeddings.py` imports `sentence_transformers` or `openai` embedding classes directly.
- `ui/app.py` calls the API over HTTP only — never imports `AgentRunner`, `ChromaVectorStore`, or `ingest_file` directly.
- `doc_qa/observability.py` must not import from `agent` or `ingestion` — it is imported by them, not the reverse.
- `api/dependencies.py` initializes singletons at module level (not inside request handlers) — one store, one embedder, one runner per process.

### 3.2 Package / Directory Structure

```
doc-qa-agent/
├── .github/
│   └── workflows/
│       └── ci.yml
├── api/
│   ├── __init__.py
│   ├── dependencies.py        # singleton lifecycle + in-memory session store
│   ├── main.py                # FastAPI app + all route handlers
│   └── models.py              # Pydantic request/response schemas
├── cli/
│   └── ingest_cli.py          # typer CLI: ingest one or more files
├── config/
│   └── settings.py            # pydantic-settings BaseSettings
├── docs/                      # sample document corpus (committed to repo)
│   ├── fannie_mae_1003_loan_application.pdf
│   ├── cfpb_closing_disclosure.pdf
│   ├── cfpb_loan_estimate.pdf
│   ├── hud1_settlement_statement.pdf
│   ├── underwriting_guidelines.docx
│   ├── mortgage_products_faq.md
│   ├── mortgage_rate_sheet.csv
│   ├── tila_disclosure_statement.txt
│   ├── loan_products_catalog.json
│   └── README.md
├── doc_qa/
│   ├── __init__.py
│   ├── embeddings.py          # EmbeddingProvider ABC + SentenceTransformerProvider + OpenAIProvider
│   ├── observability.py       # MLflow helpers + decorators
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parsers.py         # file-type parsers → raw text + metadata
│   │   ├── chunker.py         # text → List[Chunk]
│   │   ├── dedup.py           # SHA-256 file hash, skip-if-seen logic
│   │   └── pipeline.py        # orchestrates parse → chunk → embed → store + MLflow
│   ├── store/
│   │   ├── __init__.py
│   │   ├── base.py            # VectorStore ABC
│   │   └── chroma.py          # ChromaDB implementation
│   └── agent/
│       ├── __init__.py
│       ├── tools.py           # all 5 LangGraph tools
│       ├── graph.py           # StateGraph definition
│       └── runner.py          # session memory + run_turn()
├── ui/
│   └── app.py                 # Streamlit frontend (HTTP client only)
├── tests/
│   ├── conftest.py            # shared fixtures
│   ├── test_parsers.py
│   ├── test_chunker.py
│   ├── test_dedup.py
│   ├── test_embeddings.py
│   ├── test_store_chroma.py
│   ├── test_pipeline.py
│   ├── test_tools.py
│   ├── test_graph.py
│   └── test_runner.py
├── mlruns.db                  # gitignored; MLflow SQLite backend
├── chroma_db/                 # gitignored; ChromaDB persistence directory
├── pyproject.toml
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## 4. Module Specifications

### 4.1 `config/settings.py`

**Purpose:** Single source of all configuration. Loaded once at startup via
`pydantic-settings`; all other modules import `settings` from here.

**Class:**
```python
from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum

class EmbeddingProvider(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"

class Settings(BaseSettings):
    # Embedding
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    sentence_transformers_model: str = "all-MiniLM-L6-v2"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    # LLM (agent reasoning)
    llm_provider: str = "anthropic"          # "anthropic" | "openai"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    openai_chat_model: str = "gpt-4o"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "doc_qa"

    # Ingestion
    chunk_size: int = 512          # tokens (approximate, character-based with ratio)
    chunk_overlap: int = 64
    min_chunk_chars: int = 100     # discard chunks shorter than this
    pdf_ocr_threshold: int = 50    # mean chars/page below this triggers OCR fallback

    # MLflow — SQLite required for MLflow 3.x GenAI UI (traces, overview charts)
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    mlflow_experiment_name: str = "doc-qa-agent"

    # Retrieval
    default_top_k: int = 5
    retrieval_score_threshold: float = 0.0  # minimum cosine similarity to return
    max_tool_retries: int = 2               # max search_documents retries per agent turn

    # API
    api_base_url: str = "http://localhost:8000"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
```

**Must NOT import:** any `doc_qa` module.

---

### 4.2 `doc_qa/embeddings.py`

**Purpose:** Abstract embedding interface with two concrete providers. All embedding
calls in the system go through this module.

**Public API:**
```python
from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents. Returns list of float vectors."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension this provider produces."""
        ...

class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"): ...

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"): ...

def get_embedding_provider() -> EmbeddingProvider:
    """Factory: reads settings.embedding_provider and returns the right instance."""
    ...
```

**Edge cases:**
- `embed_documents([])` must return `[]` without calling the backend.
- `SentenceTransformerProvider` loads the model lazily on first call (not at import time) to keep CLI startup fast.
- If `OPENAI` provider selected but `openai_api_key` is empty, raise `ValueError` at construction time with a clear message.

**Must NOT import:** `store`, `agent`, `ingestion`, `ui`.

---

### 4.3 `doc_qa/store/base.py`

**Purpose:** Abstract base class for vector store implementations. This is the seam
that allows swapping ChromaDB for Snowflake Cortex.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Chunk:
    chunk_id: str           # "{filename}::{chunk_index}" e.g. "loan_app.pdf::003"
    text: str
    filename: str
    page_or_line: int       # page number for PDFs, line number for text files; 0 if N/A
    chunk_index: int        # 0-based index within the document
    metadata: dict = field(default_factory=dict)  # arbitrary extra fields

@dataclass
class SearchResult:
    chunk: Chunk
    score: float            # cosine similarity [0, 1]

class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Persist chunks with their embeddings."""
        ...

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Return top_k results by cosine similarity."""
        ...

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Fetch a single chunk by ID. Return None if not found."""
        ...

    @abstractmethod
    def list_documents(self) -> List[str]:
        """Return sorted list of unique filenames in the store."""
        ...

    @abstractmethod
    def get_chunks_for_document(self, filename: str) -> List[Chunk]:
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
```

**Must NOT import:** `chromadb` or any concrete implementation.

---

### 4.4 `doc_qa/store/chroma.py`

**Purpose:** ChromaDB implementation of `VectorStore`.

**Class:**
```python
class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str, collection_name: str): ...
```

**Implementation notes:**
- Use `chromadb.PersistentClient(path=persist_dir)`.
- Store chunk fields as ChromaDB metadata: `filename`, `page_or_line`, `chunk_index`, plus any `metadata` dict keys flattened with prefix `meta_`.
- `chunk_id` is the ChromaDB document ID.
- `search()` uses `collection.query(query_embeddings=[query_embedding], n_results=top_k)` and maps distances to similarity scores: `score = 1 - distance` (ChromaDB returns L2 distance by default when embeddings are not normalized; use cosine distance space: set `metadata={"hnsw:space": "cosine"}` on collection creation).
- `get_chunks_for_document()` uses `collection.get(where={"filename": filename})`.
- `document_exists()` uses `collection.get(where={"filename": filename}, limit=1)` and checks if results are non-empty.

**Must NOT import:** `agent`, `ingestion`, `ui`.

---

### 4.5 `doc_qa/ingestion/parsers.py`

**Purpose:** Convert a file path to a list of `(text, page_or_line)` tuples. One
parser function per file type.

**Public API:**
```python
from pathlib import Path
from typing import List, Tuple

ParsedPage = Tuple[str, int]   # (text_content, page_or_line_number)

def parse_pdf(path: Path) -> List[ParsedPage]: ...
def parse_txt(path: Path) -> List[ParsedPage]: ...
def parse_markdown(path: Path) -> List[ParsedPage]: ...
def parse_csv(path: Path) -> List[ParsedPage]: ...
def parse_json(path: Path) -> List[ParsedPage]: ...
def parse_docx(path: Path) -> List[ParsedPage]: ...

def parse_file(path: Path) -> List[ParsedPage]:
    """Dispatch to the correct parser based on file suffix. Raises ValueError for
    unsupported extensions."""
    ...

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".docx"}
```

**Implementation notes:**

- `parse_pdf` — **two-pass with OCR fallback**:
  1. First pass: use `pypdf.PdfReader`. Extract text from each page. Each page is one `ParsedPage` with 1-based page number.
  2. After first pass, compute mean characters per page across all pages that returned any text.
  3. If mean chars per page < `OCR_FALLBACK_THRESHOLD` (default: 50, defined in `settings.py` as `pdf_ocr_threshold: int = 50`), trigger OCR fallback for the entire document:
     - Use `pdf2image.convert_from_path(path)` to rasterize each page to a PIL Image.
     - Run `pytesseract.image_to_string(image)` on each page.
     - Replace the `pypdf` results entirely with the OCR results.
  4. Log which pass was used as a warning (`logger.warning("PDF {filename}: OCR fallback triggered")`) when OCR is used.
  5. Always strip and skip empty strings. Page numbers remain 1-based.

- `parse_txt`: split on `\n`. Each line is one `ParsedPage` with its 1-based line number.

- `parse_markdown`: same as `parse_txt` — line-by-line. Preserve heading markers (`#`, `##`, etc.) in the text content — do not strip them.

- `parse_csv`: use `csv.DictReader`. Each row becomes a JSON-serialized string via `json.dumps(dict(row))`. Row index is 1-based.

- `parse_json`:
  - Load with `json.load()`.
  - If the root value is a **list**: each element becomes one `ParsedPage` serialized via `json.dumps(element)`. Index is 1-based.
  - If the root value is a **dict**: each top-level key-value pair becomes one `ParsedPage` as `f"{key}: {json.dumps(value)}"`. Index is 1-based.
  - If the root value is a scalar (string, number): wrap as a single `ParsedPage` with index 1.
  - Invalid JSON → raise `ValueError("Invalid JSON in file: {path}")`.

- `parse_docx`:
  - Use `python-docx` (`docx.Document`).
  - Each paragraph is one `ParsedPage`. Index is the 1-based paragraph number.
  - Preserve paragraph text as-is (including any heading text — `python-docx` exposes `paragraph.style.name` but we do not need to use it; just take `paragraph.text`).
  - Skip paragraphs where `paragraph.text.strip()` is empty.
  - Tables: for each table, serialize each row as a tab-separated string and treat each row as one `ParsedPage`. Table row indices continue from where paragraph indices left off.

- All parsers must strip leading/trailing whitespace from each `ParsedPage` text and skip entries where the stripped text is empty.

**Edge cases:**
- Empty file → return `[]`.
- PDF where both `pypdf` and OCR return nothing → return `[]` and log a warning (do not raise).
- CSV with no rows → return `[]`.
- JSON file that is an empty list or empty dict → return `[]`.
- DOCX with only empty paragraphs → return `[]`.
- `pytesseract` not installed / Tesseract binary not found → catch `ImportError` / `pytesseract.TesseractNotFoundError`, log a clear error message, and return whatever `pypdf` extracted (even if sparse) rather than raising.

**OCR system dependency note (for README and Dockerfile):**
OCR fallback requires the `tesseract-ocr` system package. In Docker, add:
`RUN apt-get update && apt-get install -y tesseract-ocr poppler-utils`.
`poppler-utils` is required by `pdf2image`. If the system packages are absent, the
parser degrades gracefully to `pypdf`-only extraction (see edge case above).

**Must NOT import:** `store`, `agent`, `embeddings`, `ui`.

---

### 4.6 `doc_qa/ingestion/chunker.py`

**Purpose:** Split a list of `ParsedPage` tuples into `Chunk` objects.

**Public API:**
```python
from doc_qa.store.base import Chunk
from typing import List, Tuple

ParsedPage = Tuple[str, int]

def chunk_pages(
    pages: List[ParsedPage],
    filename: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_chars: int = 100,
) -> List[Chunk]:
    """
    Concatenate all page text with page-break markers, then apply a sliding
    window with overlap to produce Chunk objects.

    chunk_size and chunk_overlap are in characters (not tokens).
    chunk_index is 0-based and sequential across the full document.
    page_or_line is set to the page/line of the first character in each chunk.
    chunk_id is f"{filename}::{chunk_index:04d}".
    """
    ...
```

**Implementation notes:**
- Concatenate pages as: `"\n\n".join(text for text, _ in pages)`. Track a character-to-page mapping so you can assign the correct `page_or_line` to each chunk.
- Sliding window: step = `chunk_size - chunk_overlap`. Start positions: `0, step, 2*step, ...` until end of text.
- Drop chunks with `len(chunk.text.strip()) < min_chunk_chars`.
- `chunk_id` format: `f"{Path(filename).name}::{chunk_index:04d}"` — zero-padded to 4 digits.

**Must NOT import:** `store.chroma`, `agent`, `embeddings`, `ui`.

---

### 4.7 `doc_qa/ingestion/dedup.py`

**Purpose:** SHA-256 hash-based duplicate detection. Prevents re-ingesting a file
that is already stored.

**Public API:**
```python
from pathlib import Path

HASH_STORE_PATH = Path(".ingested_hashes.json")  # configurable via settings

def compute_file_hash(path: Path) -> str:
    """Return hex SHA-256 digest of file contents."""
    ...

def load_hash_store() -> dict[str, str]:
    """Load {filename: sha256_hex} from HASH_STORE_PATH. Return {} if not found."""
    ...

def save_hash_store(store: dict[str, str]) -> None:
    """Persist hash store to HASH_STORE_PATH."""
    ...

def is_duplicate(path: Path) -> bool:
    """Return True if path's hash already exists in the hash store."""
    ...

def register_file(path: Path) -> str:
    """Add path to hash store and persist. Return the hash."""
    ...
```

**Edge cases:**
- Hash store file is corrupted JSON → log warning, treat as empty dict, overwrite on next `save_hash_store`.
- `is_duplicate` checks by hash value, not filename — renaming a file and re-ingesting it IS a duplicate.

**Must NOT import:** `store`, `agent`, `embeddings`, `ui`.

---

### 4.8 `doc_qa/ingestion/pipeline.py`

**Purpose:** Orchestrate parse → chunk → embed → store for a single file. Log an
MLflow run for every ingestion.

**Public API:**
```python
from pathlib import Path
from dataclasses import dataclass

@dataclass
class IngestionResult:
    filename: str
    skipped: bool           # True if duplicate
    chunks_created: int
    embedding_time_s: float
    total_time_s: float
    mlflow_run_id: str

def ingest_file(
    path: Path,
    store: VectorStore,
    embedder: EmbeddingProvider,
    force: bool = False,    # if True, skip dedup check
) -> IngestionResult:
    """
    Full ingestion pipeline for a single file.
    1. Check dedup (unless force=True). Return skipped IngestionResult if duplicate.
    2. Parse file.
    3. Chunk pages.
    4. Embed chunks (time this).
    5. Store chunks + embeddings.
    6. Register hash.
    7. Log MLflow run (see Section 4.11 for what to log).
    Return IngestionResult.
    """
    ...

def ingest_directory(
    dir_path: Path,
    store: VectorStore,
    embedder: EmbeddingProvider,
    force: bool = False,
) -> list[IngestionResult]:
    """Ingest all supported files in dir_path. Non-recursive. Returns list of results."""
    ...
```

**Edge cases:**
- File not found → raise `FileNotFoundError` (don't catch it; let caller handle).
- Unsupported extension → raise `ValueError` from `parse_file`.
- Empty file (parse returns `[]`) → still register hash, return `IngestionResult` with `chunks_created=0`.
- Embedding failure (API error) → re-raise after logging the exception to MLflow as a failed run tag.

**Must NOT import:** `agent`, `ui`.

---

### 4.9 `doc_qa/agent/tools.py`

**Purpose:** Define all five LangGraph agent tools. Each tool is a Python function
decorated with `@tool`. Tools are stateless functions; they receive `store` and
`embedder` via closure (injected at graph build time via `partial` or a tool factory).

**Tool signatures and behavior:**

```python
from langchain_core.tools import tool

# Tool 1
def make_search_documents(store: VectorStore, embedder: EmbeddingProvider):
    @tool
    def search_documents(query: str, top_k: int = 5) -> str:
        """Search the document corpus for chunks relevant to a query.
        Returns formatted list of matching passages with source citations."""
        ...
    return search_documents

# Tool 2
def make_list_documents(store: VectorStore):
    @tool
    def list_documents() -> str:
        """List all documents currently ingested in the knowledge base.
        Returns filenames and chunk counts."""
        ...
    return list_documents

# Tool 3
def make_summarize_document(store: VectorStore, llm):
    @tool
    def summarize_document(filename: str) -> str:
        """Summarize the full content of a specific document.
        Uses all chunks from the document to produce a concise summary."""
        ...
    return summarize_document

# Tool 4
def make_classify_document(store: VectorStore, llm):
    @tool
    def classify_document(filename: str) -> str:
        """Classify a banking document into its standard form type.
        Identifies document type (e.g., Promissory Note, Deed of Trust,
        HUD-1 Settlement Statement, TILA Disclosure, Uniform Residential
        Loan Application (1003), Appraisal Report, Closing Disclosure).
        Returns the classification with a confidence rationale."""
        ...
    return classify_document

# Tool 5
@tool
def calculate(expression: str) -> str:
    """Evaluate a financial math expression safely.
    Supports: basic arithmetic, loan payment formula, APR calculation,
    amortization schedule (first N payments).
    Input: a natural-language or formula string.
    Returns: computed result with units."""
    ...
```

**Implementation notes for each tool:**

- **`search_documents`**: embed query, call `store.search(query_embedding, top_k)`, format results as:
  ```
  [1] Source: loan_application.pdf (page 3, chunk 012)
  Score: 0.87
  Text: ...
  ```
  If `store.search()` returns an empty list, return exactly: `"No results found for query."` — this exact string is detected by `validate_tool_results` to trigger the retry hint.
- **`list_documents`**: call `store.list_documents()`, for each filename call `len(store.get_chunks_for_document(filename))`. Format as a table.
- **`summarize_document`**: fetch all chunks via `store.get_chunks_for_document(filename)`, concatenate texts (truncate to 8000 chars if needed), call LLM with a summarization prompt. Use `llm.invoke()`.
- **`classify_document`**: fetch first 3 chunks of the document (enough for header/form info), call LLM with a classification prompt listing the known banking document types. Return type + rationale.
- **`calculate`**: parse the expression using a safe evaluator. Support these specific operations:
  - Monthly payment: `PMT(rate, nper, pv)` → standard annuity formula
  - APR: given fees, interest rate, and term
  - Simple arithmetic via Python's `ast.literal_eval`-safe approach (use `simpleeval` library, not `eval`)
  - If expression is unrecognizable → return a clear "Cannot compute: ..." message.

**Must NOT import:** `ui`, `ingestion`.

---

### 4.10 `doc_qa/agent/graph.py`

**Purpose:** Define and compile the LangGraph `StateGraph` that powers the agent.

**Public API:**
```python
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    retry_count: int           # incremented by validate_tool_results; reset to 0 at turn start
    last_tool_name: str        # name of the most recently executed tool
    last_tool_empty: bool      # True if the most recent search_documents returned 0 results

def build_graph(
    store: VectorStore,
    embedder: EmbeddingProvider,
    llm,
) -> "CompiledGraph":
    """
    Build and compile a LangGraph ReAct-style agent graph with tool validation.

    Nodes:
      - "agent":               LLM node — decides whether to call a tool or respond
      - "tools":               ToolNode — executes whichever tool the LLM requested
      - "validate_tool_results": lightweight function node — inspects tool output,
                               injects a retry hint message if results are poor,
                               increments retry_count

    Edges:
      - START → "agent"
      - "agent" → END (if no tool call) or → "tools" (if tool call)
      - "tools" → "validate_tool_results" (always)
      - "validate_tool_results" → "agent" (always)

    Bind all 5 tools to the LLM via llm.bind_tools(tools).
    Use langgraph.prebuilt.ToolNode for the tools node.
    Use a conditional edge on "agent" via tools_condition from langgraph.prebuilt.
    """
    ...
```

**Implementation notes:**

- Import `tools_condition` from `langgraph.prebuilt`.
- The compiled graph is stateless — session memory lives in `runner.py`.
- Construct the LLM based on `settings.llm_provider`:
  ```python
  if settings.llm_provider == "anthropic":
      from langchain_anthropic import ChatAnthropic
      llm = ChatAnthropic(
          model=settings.anthropic_model,
          api_key=settings.anthropic_api_key,
      )
  elif settings.llm_provider == "openai":
      from langchain_openai import ChatOpenAI
      llm = ChatOpenAI(
          model=settings.openai_chat_model,
          api_key=settings.openai_api_key,
      )
  elif settings.llm_provider == "ollama":
      from langchain_ollama import ChatOllama
      llm = ChatOllama(
          model=settings.ollama_model,
          base_url=settings.ollama_base_url,
      )
  else:
      raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")
  ```

**`validate_tool_results` node — full specification:**

This is a pure Python function (no LLM call). It runs after every tool execution and before the agent sees the result. Its job is to detect low-quality retrieval and inject a corrective hint into the message stream.

```python
def validate_tool_results(state: AgentState) -> AgentState:
    """
    Inspect the most recent ToolMessage in state.messages.
    If the tool was search_documents AND the result indicates 0 matches
    (detected by checking for the string "No results found" in the content),
    AND retry_count < settings.max_tool_retries (default: 2):
      - Append a SystemMessage: "The previous search returned no results.
        Try broadening the query — use fewer keywords or more general terms."
      - Set last_tool_empty = True
      - Increment retry_count
    Otherwise:
      - Set last_tool_empty = False
      - Do not modify messages
    Always return the updated state.
    """
```

Key design decisions:
- Only triggers on `search_documents` results — other tools (`calculate`, `classify_document`, etc.) are not subject to retry logic because their failure modes are different (a bad expression or missing file should surface to the user, not be silently retried).
- The retry hint is a `SystemMessage`, not a `HumanMessage` — it is injected as internal agent guidance, not visible in the UI chat history.
- `max_tool_retries: int = 2` is a new field in `config/settings.py`. After 2 retries the agent proceeds with whatever it has, preventing infinite loops.
- "No results found" detection: `search_documents` must emit this exact string when `store.search()` returns an empty list. Specify this in `tools.py` as: `if not results: return "No results found for query."`.
- The `retry_count` field in `AgentState` is reset to `0` at the start of each turn in `runner.py` (by passing `{"messages": [...], "retry_count": 0, "last_tool_name": "", "last_tool_empty": False}` to `graph.invoke()`).

**`settings.py` addition:**
```python
max_tool_retries: int = 2   # max search_documents retries per turn
```

**System prompt** (injected as a `SystemMessage` prepended to messages in the runner):
```
You are a precise document analysis assistant for a banking institution.
You have access to a corpus of loan and banking documents.
Always cite your sources (document name and page number) when answering.
Use the calculate tool for any numerical computations.
Use classify_document when asked to identify document types.
If a search returns no results, try a different, broader query before concluding
the information is not available.
Reason step by step before producing a final answer.

GROUNDING RULES — these are mandatory, not suggestions:
1. You MUST only answer from information retrieved via the search_documents tool.
2. You MUST NOT answer from general knowledge or training data.
3. If after retrying your search you still have no retrieved evidence, you MUST
   respond with exactly this format:
   "UNGROUNDED: I was unable to find information about [topic] in the available documents."
4. You MUST NOT fabricate citations, page numbers, or document names.
5. Every factual claim in your response must be traceable to a retrieved chunk.
```

**Must NOT import:** `ui`, `ingestion`.

---

### 4.11 `doc_qa/agent/runner.py`

**Purpose:** Manage per-session conversation history and expose a single `run_turn()`
entry point. This is the only agent module the UI imports.

**Public API:**
```python
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.messages import BaseMessage

UNGROUNDED_PREFIX = "UNGROUNDED:"

@dataclass
class AgentSession:
    session_id: str
    history: list[BaseMessage] = field(default_factory=list)
    mlflow_run_id: Optional[str] = None

class AgentRunner:
    def __init__(self, store: VectorStore, embedder: EmbeddingProvider): ...

    def new_session(self) -> AgentSession:
        """Create a new session with a UUID session_id. Start an MLflow run."""
        ...

    def run_turn(self, session: AgentSession, user_message: str) -> str:
        """
        Invoke the graph, run post-response grounding check, append to history,
        log to MLflow. Return the final response string.
        """
        ...

    def _check_grounded(self, response: str) -> bool:
        """
        Return True if the response cites at least one real document filename
        from store.list_documents(), OR if it already starts with UNGROUNDED_PREFIX.
        Return False if non-empty but contains zero real filenames and no prefix.
        """
        ...

    def _format_grounding_failure(self) -> str:
        """
        Build the ⚠️ warning message shown to the user when grounding check fails.
        Includes bulleted list of currently ingested document filenames.
        """
        ...
```

**Implementation notes:**
- Build the graph once per `AgentRunner` instance (not per turn).
- Prepend the `SystemMessage` on every graph invocation (do not store it in `session.history` — add it fresh each call so the system prompt is always first).
- `run_turn()` passes `{"messages": [system_message] + session.history + [HumanMessage(user_message)], "retry_count": 0, "last_tool_name": "", "last_tool_empty": False}` to `graph.invoke()`.
- After invocation, extract the last `AIMessage` from the returned state. Then apply the grounding check:
  1. Call `_check_grounded(response_text)`.
  2. **False** (non-empty response, no real filenames, no prefix) → final response = `_format_grounding_failure()`. Log `grounded=False`.
  3. **True + starts with `UNGROUNDED_PREFIX`** → strip prefix, prepend `"⚠️ "`, append available docs list. Log `grounded=False`.
  4. **True + no prefix** → use response as-is. Log `grounded=True`.
- Append both the `HumanMessage` and the final post-check response as an `AIMessage` to `session.history`.
- Log to MLflow: `user_message` (truncated to 500 chars), `response_preview` (first 500 chars), `tool_calls` (comma-separated names), `latency_s`, `turn_index`, `grounded` (bool as int: 1/0).

**`_check_grounded` implementation:**
- Get `known_filenames = set(store.list_documents())`.
- If `response.startswith(UNGROUNDED_PREFIX)`: return `True` (handled by case 3 above).
- If `len(response.strip()) == 0`: return `False`.
- Return `any(fname in response for fname in known_filenames)`.
- This is a heuristic. It catches the most common failure (agent answers from training data with no retrieval) but does not verify cited content is accurate. Document this limitation in the README under "Known Limitations."

**`_format_grounding_failure` output format:**
```
⚠️ I was unable to find relevant information about this question
in the current document corpus.

Documents available:
• loan_application.pdf
• lending_policy.pdf
• mortgage_faq.md
• loan_rates.csv
• tila_disclosure.txt

If this topic is covered in another document, you can ingest it
using the sidebar uploader.
```

**Must NOT import:** `ui`.

---

### 4.12 `doc_qa/observability.py`

**Purpose:** MLflow helpers used by both ingestion and agent modules.

**Public API:**
```python
import mlflow
from typing import Callable, Any
from functools import wraps

def setup_mlflow() -> None:
    """Set tracking URI and create experiment if not exists. Call once at startup."""
    ...

def log_ingestion_run(
    filename: str,
    chunks_created: int,
    embedding_time_s: float,
    total_time_s: float,
    skipped: bool,
    file_hash: str,
    embedding_provider: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Start and end an MLflow run for a single ingestion. Return run_id."""
    ...

def log_agent_turn(
    run_id: str,
    turn_index: int,
    user_message: str,
    response_preview: str,
    tool_calls: list[str],
    latency_s: float,
    grounded: bool,
) -> None:
    """Log metrics and params for one agent turn to an existing MLflow run.
    Logs grounded as a metric (1.0 = grounded, 0.0 = ungrounded) so grounding
    rate can be tracked as a time-series metric across sessions."""
    ...

def log_retrieval_quality(
    run_id: str,
    query: str,
    top_k_scores: list[float],
) -> None:
    """Log retrieval hit quality: mean score, max score, min score of top-k results."""
    ...
```

**Must NOT import:** `agent`, `ingestion`, `store`, `ui`.

---

### 4.13 `ui/app.py`

**Purpose:** Streamlit chat frontend. Pure HTTP client — all business logic lives
in the FastAPI backend. Does not import `AgentRunner`, `ChromaVectorStore`, or any
ingestion internals.

**Layout and behavior:**
- Sidebar:
  - Section: **Document Corpus** — calls `GET /documents` and displays filenames + chunk counts.
  - Section: **Ingest New File** — file uploader (PDF, TXT, MD, CSV, JSON, DOCX), "Ingest" button. On click: `POST /documents/ingest` (multipart), display result summary. Refresh document list after ingestion.
  - Section: **Settings** — read-only display of `settings.embedding_provider`, `settings.llm_provider`, `settings.chroma_collection_name`, `API_BASE_URL`.
  - Section: **MLflow** — link to `http://localhost:5000`. Display current session's MLflow run ID if active.
- Main area:
  - Chat message history using `st.chat_message`.
  - `st.chat_input` at the bottom.
  - On user submit: `POST /sessions/{id}/chat`, display response.
  - Show a spinner ("Agent is thinking…") during the request.
- Session state keys: `st.session_state.session_id`, `st.session_state.mlflow_run_id`, `st.session_state.messages` (display list).
- On first load: `POST /sessions` → stores `session_id` and `mlflow_run_id` in session state.

**Must NOT import:** `AgentRunner`, `ChromaVectorStore`, `ingest_file`, `graph.py`, or any `doc_qa` internals.

---

### 4.14 `cli/ingest_cli.py`

**Purpose:** Typer CLI for ingesting files from the command line.

---

### 4.15 `api/` package

**Purpose:** FastAPI backend that owns all agent and retrieval logic. The Streamlit
UI (and any other client) calls this API over HTTP.

#### `api/models.py` — Pydantic request/response schemas

```python
class SessionResponse(BaseModel):
    session_id: str
    mlflow_run_id: str | None = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

class MessageRecord(BaseModel):
    role: str       # "user" | "assistant"
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
```

#### `api/dependencies.py` — Singleton lifecycle + session store

Singletons are created at **module load time** (not inside request handlers) so
every request shares the same store, embedder, and runner:

```python
setup_mlflow()

_store   = ChromaVectorStore(persist_dir, collection_name)
_embedder = get_embedding_provider()
_runner  = AgentRunner(store=_store, embedder=_embedder)
_sessions: dict[str, AgentSession] = {}   # in-memory; lost on restart

def get_store() -> ChromaVectorStore: ...
def get_embedder() -> EmbeddingProvider: ...
def get_runner() -> AgentRunner: ...
def get_sessions() -> dict[str, AgentSession]: ...
```

#### `api/main.py` — Route handlers

| Method | Path | Request body | Response |
|--------|------|-------------|----------|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `GET` | `/status` | — | `StatusResponse` |
| `POST` | `/sessions` | — | `SessionResponse` |
| `POST` | `/sessions/{session_id}/chat` | `ChatRequest` | `ChatResponse` |
| `GET` | `/sessions/{session_id}` | — | `SessionHistoryResponse` |
| `GET` | `/documents` | — | `list[DocumentInfo]` |
| `POST` | `/documents/ingest` | multipart `file` | `IngestResponse` |

All `Depends(...)` calls use `# noqa: B008` to suppress the ruff B008 false positive
(same pattern as Typer's `Argument`/`Option` defaults).

**Session lifecycle:** `POST /sessions` calls `runner.new_session()`, stores the
`AgentSession` in `_sessions[session_id]`, and returns the ID + MLflow run ID.
`POST /sessions/{id}/chat` looks up the session by ID, calls `runner.run_turn()`,
and returns the response string.

**Interactive docs:** `http://localhost:8000/docs` (Swagger UI, built into FastAPI).

```bash
# Usage examples:
python -m cli.ingest_cli ingest docs/loan_application.pdf
python -m cli.ingest_cli ingest docs/ --force
python -m cli.ingest_cli list
python -m cli.ingest_cli status
```

**Commands:**
- `ingest <path> [--force]`: ingest a single file or directory.
- `list`: print all ingested document filenames and chunk counts.
- `status`: print total chunk count, embedding provider, chroma collection.

**Must NOT import:** `ui`.

---

## 5. Build System

### `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "doc-qa-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "chromadb>=0.5.0",
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-anthropic>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-ollama>=0.2.0",
    "langgraph>=0.2.0",
    "mlflow>=2.16.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "pypdf>=4.0.0",
    "pdf2image>=1.17.0",
    "pytesseract>=0.3.13",
    "python-docx>=1.1.0",
    "sentence-transformers>=3.0.0",
    "simpleeval>=1.0.0",
    "fastapi>=0.115.0",
    "httpx>=0.27.0",
    "requests>=2.32.0",
    "streamlit>=1.38.0",
    "typer>=0.12.0",
    "uvicorn[standard]>=0.30.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "pytest-mock>=3.14.0",
    "ruff>=0.6.0",
    "mypy>=1.11.0",
    "reportlab>=4.0.0",
]

[project.scripts]
doc-qa-ingest = "cli.ingest_cli:app"

[tool.hatch.build.targets.wheel]
packages = ["doc_qa", "config", "cli", "ui", "api"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=doc_qa --cov-report=term-missing --cov-fail-under=80"
filterwarnings = ["ignore::DeprecationWarning"]

[tool.coverage.run]
omit = ["ui/*", "cli/*", "tests/*", "config/*"]

[tool.mypy]
python_version = "3.11"
strict = false
ignore_missing_imports = true
```

---

## 6. Testing

### Coverage target
80% line coverage on `doc_qa/` package. `ui/`, `cli/`, `config/` are excluded.

### Mocking strategy
- **ChromaDB**: use an in-memory `chromadb.EphemeralClient()` in fixtures — never mock ChromaDB's API, test against the real in-memory implementation.
- **LLM calls** (`ChatAnthropic`, `ChatOpenAI`): mock with `pytest-mock` at the `langchain` interface level. Return deterministic `AIMessage` objects.
- **Embedding calls**: use a `FakeEmbeddingProvider` fixture that returns fixed-length random vectors (dim=384 to match `all-MiniLM-L6-v2`).
- **MLflow**: use `mlflow.set_tracking_uri("./test_mlruns")` in `conftest.py`; clean up in teardown.
- **File I/O**: use `tmp_path` pytest fixture for all temp files.

### `tests/conftest.py`
Required fixtures:
- `fake_embedder` → `FakeEmbeddingProvider` (returns `np.random.rand(384).tolist()`)
- `chroma_store` → `ChromaVectorStore` backed by `chromadb.EphemeralClient()`
- `sample_chunk` → one `Chunk` with known values
- `sample_chunks` → list of 5 `Chunk` objects across 2 filenames
- `mock_llm` → `MagicMock` with `.invoke()` returning `AIMessage(content="test response")`
- `tmp_docs_dir` → `tmp_path` with 3 sample files: `test.pdf` (mocked), `test.txt`, `test.csv`

### `tests/test_parsers.py`
- `test_parse_txt_returns_lines` — parse a known 3-line string, assert 3 `ParsedPage` tuples with correct line numbers
- `test_parse_csv_serializes_rows` — parse a 2-row CSV, assert JSON-serialized output
- `test_parse_markdown_preserves_headings` — heading markers (`#`, `##`) remain in text
- `test_parse_file_raises_on_unsupported` — `.xlsx` extension raises `ValueError`
- `test_parse_txt_empty_file` — empty file returns `[]`
- `test_parse_pdf_pypdf_path` — mock `pypdf.PdfReader` returning 2 pages with 200+ chars each; assert 2 `ParsedPage` tuples and `pytesseract` is NOT called
- `test_parse_pdf_ocr_fallback_triggered` — mock `pypdf.PdfReader` returning pages with < 50 mean chars; assert `pdf2image.convert_from_path` and `pytesseract.image_to_string` are called
- `test_parse_pdf_tesseract_missing_degrades_gracefully` — mock `pytesseract` raising `TesseractNotFoundError`; assert function returns `pypdf` results without raising
- `test_parse_json_list_root` — JSON file with root list of 3 dicts → 3 `ParsedPage` tuples
- `test_parse_json_dict_root` — JSON file with root dict of 2 keys → 2 `ParsedPage` tuples
- `test_parse_json_empty_list` — `[]` → returns `[]`
- `test_parse_json_invalid_raises_value_error`
- `test_parse_docx_paragraphs` — mock `docx.Document` with 3 non-empty paragraphs, assert 3 `ParsedPage` tuples
- `test_parse_docx_skips_empty_paragraphs` — paragraph with empty text is excluded from output
- `test_parse_docx_includes_table_rows` — document with 1 table of 2 rows → both rows included as `ParsedPage` entries

### `tests/test_chunker.py`
- `test_chunk_produces_correct_count` — known text of 1000 chars, chunk_size=200, overlap=20 → assert expected chunk count
- `test_chunk_ids_are_unique` — all `chunk_id` values in output are unique
- `test_chunk_overlap_content` — consecutive chunks share overlapping text
- `test_chunk_drops_short_chunks` — chunk shorter than `min_chunk_chars` is excluded
- `test_chunk_page_assignment` — chunk spanning page 1 and 2 gets `page_or_line=1`
- `test_chunk_id_format` — IDs match `f"{filename}::{i:04d}"` pattern

### `tests/test_dedup.py`
- `test_compute_file_hash_deterministic` — same file hashed twice returns same value
- `test_is_duplicate_false_for_new_file`
- `test_is_duplicate_true_after_register`
- `test_duplicate_detects_by_hash_not_name` — rename file, re-register → still duplicate
- `test_corrupted_hash_store_treated_as_empty` — write invalid JSON to store path, assert `is_duplicate` returns False

### `tests/test_embeddings.py`
- `test_sentence_transformer_embed_documents_shape` — output length matches input length
- `test_sentence_transformer_embed_query_is_list_of_floats`
- `test_embed_documents_empty_list_returns_empty`
- `test_openai_provider_raises_on_empty_api_key`
- `test_get_embedding_provider_returns_correct_type` — test both enum values

### `tests/test_store_chroma.py`
- `test_add_and_search_returns_results` — add 3 chunks, search with matching embedding, assert ≥1 result
- `test_list_documents_returns_unique_filenames`
- `test_get_chunks_for_document_sorted_by_index`
- `test_document_exists_true_and_false`
- `test_get_chunk_by_id` — fetch by known chunk_id
- `test_get_chunk_returns_none_for_missing_id`
- `test_count_reflects_added_chunks`
- `test_search_respects_top_k`

### `tests/test_pipeline.py`
- `test_ingest_file_returns_correct_chunk_count` — use `fake_embedder`, `chroma_store`, real `.txt` file
- `test_ingest_file_skips_duplicate` — ingest same file twice, assert second result has `skipped=True`
- `test_ingest_file_force_flag_overrides_dedup`
- `test_ingest_file_empty_file_creates_zero_chunks`
- `test_ingest_file_logs_mlflow_run` — assert MLflow run was created with expected params
- `test_ingest_directory_processes_all_supported_files`

### `tests/test_tools.py`
- `test_search_documents_returns_formatted_results` — mock `store.search`, assert citation format
- `test_list_documents_returns_table` — mock `store.list_documents` and `store.get_chunks_for_document`
- `test_summarize_document_calls_llm` — mock LLM, assert it was called with chunk text
- `test_classify_document_returns_known_type` — mock LLM returning "Promissory Note"
- `test_calculate_pmt` — `calculate("PMT(0.005, 360, 200000)")` → assert result is ~$1199
- `test_calculate_simple_arithmetic` — `calculate("1200 * 12")` → `"14400"`
- `test_calculate_unknown_expression_returns_error_string`

### `tests/test_graph.py`
- `test_graph_builds_without_error` — `build_graph(store, embedder, mock_llm)` completes without raising
- `test_graph_returns_ai_message` — invoke graph with a simple message; mock LLM returns direct response (no tool call); assert final state contains an `AIMessage`
- `test_validate_tool_results_no_op_on_good_results` — call `validate_tool_results` with a state whose last `ToolMessage` contains real text; assert `retry_count` unchanged, `last_tool_empty` is False, no `SystemMessage` injected
- `test_validate_tool_results_injects_hint_on_empty_search` — call `validate_tool_results` with a state whose last `ToolMessage` content is `"No results found for query."` and `last_tool_name == "search_documents"`; assert a `SystemMessage` hint is appended and `retry_count` incremented to 1
- `test_validate_tool_results_does_not_retry_beyond_max` — set `retry_count = settings.max_tool_retries` before calling; assert no hint is injected even on empty results
- `test_validate_tool_results_ignores_non_search_tools` — last tool is `calculate` with empty-looking output; assert no hint injected and `retry_count` unchanged

### `tests/test_runner.py`
- `test_new_session_returns_session_with_id`
- `test_run_turn_returns_string` — mock LLM returning a response that contains a known filename; assert return type is `str`
- `test_run_turn_appends_to_history` — after 2 turns, `session.history` has 4 messages
- `test_run_turn_logs_mlflow` — assert `mlflow.log_metric` or equivalent was called with `grounded`
- `test_check_grounded_true_when_filename_present` — response contains `"loan_application.pdf"`; mock `store.list_documents()` returns that filename; assert `True`
- `test_check_grounded_false_when_no_filename` — response is "The interest rate is 6.5% per current market rates." with no filename; assert `False`
- `test_check_grounded_true_on_ungrounded_prefix` — response starts with `"UNGROUNDED:"`; assert `_check_grounded` returns `True` (LLM self-reported)
- `test_check_grounded_false_on_empty_response` — empty string response; assert `False`
- `test_run_turn_formats_grounding_failure_when_ungrounded` — mock graph returning response with no filename; assert returned string starts with `"⚠️"`
- `test_run_turn_formats_grounding_failure_on_ungrounded_prefix` — mock graph returning `"UNGROUNDED: I could not find..."` response; assert returned string starts with `"⚠️"` and contains document filenames
- `test_run_turn_grounded_response_returned_as_is` — mock graph returning response with filename present; assert returned string does NOT start with `"⚠️"`

---

## 7. CI Pipeline

### `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dev dependencies
        run: pip install -e ".[dev]"
      - name: Ruff lint
        run: ruff check .
      - name: Ruff format check
        run: ruff format --check .
      - name: Mypy
        run: mypy doc_qa/ --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    needs: lint-and-type-check
    env:
      EMBEDDING_PROVIDER: sentence_transformers
      LLM_PROVIDER: anthropic
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      MLFLOW_TRACKING_URI: sqlite:///test_mlruns.db
      CHROMA_PERSIST_DIR: ./test_chroma_db
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install system dependencies
        run: sudo apt-get update && sudo apt-get install -y tesseract-ocr poppler-utils
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run tests with coverage
        run: pytest --cov=doc_qa --cov-report=xml --cov-fail-under=80
```

**Notes:**
- Tests that make real LLM calls must be guarded with `pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key")` or fully mocked. All tests in CI must pass without a real API key — mock LLM calls in tests.
- `ANTHROPIC_API_KEY` secret is optional in CI — tests use mocks. Add the secret to enable integration tests later.

---

## 8. Makefile

```makefile
.PHONY: install lint format type-check test test-cov ingest api ui mlflow clean docker-build docker-up

install:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

type-check:
	mypy doc_qa/ --ignore-missing-imports

test:
	pytest

test-cov:
	pytest --cov=doc_qa --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

ingest:
	python -m cli.ingest_cli ingest docs/

api:
	uvicorn api.main:app --reload --port 8000

ui:
	streamlit run ui/app.py

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

clean:
	rm -rf chroma_db/ mlruns/ mlruns.db mlflow_data/ test_mlruns/ test_mlruns.db test_chroma_db/ .ingested_hashes.json __pycache__/ .pytest_cache/ htmlcov/ coverage.xml

docker-build:
	docker build -t doc-qa-agent .

docker-up:
	docker-compose up
```

---

## 9. Configuration

### `.env.example`

```bash
# Embedding provider: "sentence_transformers" (default, local) or "openai"
EMBEDDING_PROVIDER=sentence_transformers

# Only needed if EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=

# LLM provider for agent reasoning: "anthropic" (default) or "openai"
LLM_PROVIDER=anthropic

# Only needed if LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=

# Only needed if LLM_PROVIDER=openai
# OPENAI_API_KEY= (same key as above, reused)

# ChromaDB persistence directory
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=doc_qa

# Chunking parameters
CHUNK_SIZE=512
CHUNK_OVERLAP=64
MIN_CHUNK_CHARS=100
# Mean chars/page below this triggers OCR fallback for PDFs (0 to disable OCR fallback)
PDF_OCR_THRESHOLD=50

# MLflow — SQLite required for MLflow 3.x GenAI UI (traces, overview charts)
# Swap to s3://your-bucket/mlruns.db for AWS production deployment
MLFLOW_TRACKING_URI=sqlite:///mlruns.db
MLFLOW_EXPERIMENT_NAME=doc-qa-agent

# Retrieval
DEFAULT_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.0
MAX_TOOL_RETRIES=2             # max search_documents retries per agent turn before proceeding

# API backend URL (used by Streamlit UI to reach FastAPI)
API_BASE_URL=http://localhost:8000
```

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY doc_qa/ doc_qa/
COPY cli/ cli/
COPY ui/ ui/
COPY config/ config/
COPY docs/ docs/

RUN pip install -e .

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### `docker-compose.yml`

```yaml
version: "3.9"
services:
  api:
    build: .
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./mlflow_data:/app/mlflow_data
      - ./.ingested_hashes.json:/app/.ingested_hashes.json
    env_file:
      - .env
    environment:
      - MLFLOW_TRACKING_URI=sqlite:////app/mlflow_data/mlruns.db

  app:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    environment:
      - API_BASE_URL=http://api:8000
    depends_on:
      - api

  mlflow:
    image: python:3.11-slim
    command: >
      sh -c "pip install mlflow -q &&
             mlflow ui --backend-store-uri sqlite:////mlflow_data/mlruns.db --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow_data:/mlflow_data
```

---

## 10. Sample Documents (`docs/`)

Claude Code must create or source these files and commit them to `docs/`. Each must
be rich enough to require multi-chunk retrieval.

| Filename | Type | Content description | Why it's interesting |
|---|---|---|---|
| `loan_application.pdf` | PDF | Provided in the assignment — a real loan application form | Primary document; tests PDF parsing |
| `lending_policy.pdf` | PDF | Fictional "First National Bank Lending Policy" — 3-4 pages covering LTV limits, DTI thresholds, credit score minimums, appraisal requirements, and exception approval process | Tests multi-page PDF retrieval; classify_document returns "Internal Lending Policy" |
| `mortgage_faq.md` | Markdown | 20+ Q&A pairs about fixed vs ARM mortgages, escrow, PMI, refinancing, prepayment penalties | Tests Markdown parsing; rich enough for multi-hop retrieval |
| `loan_rates.csv` | CSV | Rate table: columns = `product` (30yr fixed, 15yr fixed, 5/1 ARM, etc.), `term_months`, `rate_pct`, `apr_pct`, `points`, `effective_date` — 15+ rows | Tests CSV parsing; enables calculate tool use for payment estimates |
| `tila_disclosure.txt` | Plain text | Fictional TILA Disclosure Statement for a $250,000 mortgage — includes APR, finance charge, amount financed, total payments, payment schedule | Tests TXT parsing; classify_document returns "TILA Disclosure" |

All fictional documents must use plausible but clearly fake institution names
(e.g., "First National Bank of Plainview") and fake borrower names to avoid any
risk of containing real PII.

---

## 11. Future-Proofing Notes

These are NOT being built in v1, but the architecture must not foreclose them:

1. **Snowflake Cortex swap**: The `VectorStore` ABC exists precisely for this.
   Implementing `SnowflakeCortexVectorStore(VectorStore)` requires only implementing
   the 7 abstract methods. Document this in the `README.md` in a "Production Migration"
   section.

2. **AWS S3 artifact store**: `MLFLOW_TRACKING_URI` is a config value. Setting it to
   `s3://bucket/mlruns` with appropriate IAM credentials requires zero code changes.

3. **OpenAI / other LLM providers**: `settings.llm_provider` gates the LLM construction
   in `graph.py`. Adding a new provider is one `elif` branch.

5. **Semantic grounding verification**: The current `_check_grounded` heuristic checks for filename presence in the response — it does not verify that cited content actually matches retrieved chunks. A stronger v2 implementation would use an LLM-as-judge pass to verify factual grounding, or compare response embeddings against the top-k retrieved chunk embeddings for semantic similarity.

6. **`get_chunk` tool**: The `VectorStore.get_chunk()` method is already implemented.
   Adding the tool is 10 lines in `tools.py`. Noted as stretch goal.

5. **Multi-user sessions**: `AgentRunner.new_session()` already returns `AgentSession`
   objects with UUIDs. The `_sessions` dict in `api/dependencies.py` is the only thing
   that needs to be replaced with a Redis or database-backed store.

6. **Async ingestion**: `ingest_file` is currently synchronous. The FastAPI backend
   is in place — wrapping ingestion in `asyncio.run_in_executor` or a task queue
   (Celery, SQS) requires no changes to `ingest_file` itself.

---

## 12. Implementation Order

1. `config/settings.py` — foundation; everything else imports it
2. `doc_qa/embeddings.py` — needed by store and ingestion
3. `doc_qa/store/base.py` — `Chunk`, `SearchResult`, `VectorStore` ABC
4. `doc_qa/store/chroma.py` — concrete store; test with `test_store_chroma.py`
5. `doc_qa/ingestion/parsers.py` — file → text; test with `test_parsers.py`
6. `doc_qa/ingestion/chunker.py` — text → chunks; test with `test_chunker.py`
7. `doc_qa/ingestion/dedup.py` — hash store; test with `test_dedup.py`
8. `doc_qa/observability.py` — MLflow helpers (needed by pipeline and runner)
9. `doc_qa/ingestion/pipeline.py` — full ingestion; test with `test_pipeline.py`
10. `cli/ingest_cli.py` — CLI wrapper; run `make ingest` to ingest `docs/`
11. `docs/` — nine sample banking documents
12. `doc_qa/agent/tools.py` — all 5 tools; test with `test_tools.py`
13. `doc_qa/agent/graph.py` — LangGraph graph; test with `test_graph.py`
14. `doc_qa/agent/runner.py` — session runner; test with `test_runner.py`
15. `ui/app.py` — Streamlit UI (no unit tests; verify manually)
16. `Dockerfile` + `docker-compose.yml` — verify `docker-compose up` starts all services
17. `.github/workflows/ci.yml` — verify CI passes on a push
18. `README.md` — initial documentation
19. `api/` package — `models.py`, `dependencies.py`, `main.py`; refactor `ui/app.py` to HTTP client; update `pyproject.toml`, `Makefile`, `docker-compose.yml`, `.env.example`, `.gitignore`, `ci.yml`; switch MLflow to SQLite
20. `README.md` + `spec.md` — as-built documentation updates

---

## 13. Definition of Done

- [ ] `make install` completes without errors on a clean Python 3.11 environment
- [ ] `make lint` exits 0 (no ruff errors)
- [ ] `make format` reports no files would be reformatted
- [ ] `make type-check` exits 0 (no mypy errors in `doc_qa/`)
- [ ] `make test` exits 0 with ≥80% coverage on `doc_qa/`
- [ ] `make ingest` successfully ingests all documents in `docs/` and prints results
- [ ] `make api` starts FastAPI at `localhost:8000` without errors
- [ ] `curl localhost:8000/health` returns `{"status":"ok"}`
- [ ] `make ui` starts Streamlit at `localhost:8501` and the UI successfully calls the API
- [ ] `make mlflow` starts MLflow UI at `localhost:5000` with all tabs (Overview, Experiments, Traces) functional
- [ ] The following Q&A scenarios work correctly in the UI (manual verification):
  1. **Multi-doc retrieval**: "Would a borrower with a 660 credit score and 85% LTV qualify for a conventional loan, and what rate would they get?" → agent searches `underwriting_guidelines.docx` and `mortgage_rate_sheet.csv`
  2. **Document classification**: "What type of document is tila_disclosure_statement.txt?" → agent calls `classify_document`, returns "TILA Disclosure"
  3. **Summarization**: "Summarize the mortgage FAQ document" → agent calls `summarize_document("mortgage_products_faq.md")`
  4. **Financial calculation**: "What would the monthly payment be on a $250,000 loan at 6.5% for 30 years?" → agent calls `calculate`
  5. **Follow-up memory**: After Q4, ask "What about for 15 years instead?" → agent uses conversation history, recalculates without re-asking for loan amount
- [ ] MLflow UI shows: ingestion runs with chunk counts and embedding time; agent sessions with turn-level latency, tool calls, and grounding metrics
- [ ] `docker-compose up` starts all three services (api :8000, app :8501, mlflow :5000)
- [ ] GitHub Actions CI passes on `main` branch (lint → type-check → test)
- [ ] `README.md` and `spec.md` reflect as-built architecture including FastAPI layer and SQLite MLflow

---

## 14. Suggested Repo Name

**`doc-qa-agent`**

Matches the Python package name, is searchable, and is descriptive without being
verbose. Use this as both the GitHub repo name and the `pyproject.toml` project name.
