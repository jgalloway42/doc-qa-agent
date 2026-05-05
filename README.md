# doc-qa-agent

An agentic document Q&A system for banking document corpora. Ingest loan documents,
policies, disclosures, and rate sheets — then ask questions across them using a
multi-step reasoning agent that cites its sources, retries failed searches, and
refuses to answer from general knowledge.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![CI](https://github.com/jgalloway42/doc-qa-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/jgalloway42/doc-qa-agent/actions/workflows/ci.yml)

---

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Capabilities](#agent-capabilities)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running with Docker](#running-with-docker)
- [Development](#development)
- [Document Corpus](#document-corpus)
- [Sample Q&A](#sample-qa)
- [Observability](#observability)
- [Known Limitations](#known-limitations)
- [Production Migration](#production-migration)
- [License](#license)

---

## Overview

`doc-qa-agent` combines a document ingestion pipeline with a LangGraph-powered
reasoning agent. The agent does not answer from general knowledge — every response
is grounded in retrieved document passages, with citations to source files and page
numbers. When it cannot find an answer, it says so and lists what it does have.

**Key design principles:**

- **Grounded by default** — two-layer enforcement: system prompt rules plus a
  post-response programmatic check. Ungrounded responses are replaced with a
  structured warning listing available documents.
- **Modular by design** — every major component sits behind an abstract interface.
  The vector store, embedding provider, and LLM are all swappable via environment
  variables or a one-class implementation.
- **Observable** — MLflow tracks every ingestion run and every agent turn,
  including per-turn grounding status and retrieval quality scores.
- **Enterprise-ready dependencies** — every package is MIT, Apache 2.0, or
  BSD 3-Clause licensed. No GPL. No SaaS-only dependencies.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Streamlit UI                      │
│                        ui/app.py                         │
└────────────────────────┬────────────────────────────────┘
                         │ run_turn()
┌────────────────────────▼────────────────────────────────┐
│                     AgentRunner                          │
│              doc_qa/agent/runner.py                      │
│   • session memory   • grounding check   • MLflow log    │
└────────────────────────┬────────────────────────────────┘
                         │ graph.invoke()
┌────────────────────────▼────────────────────────────────┐
│                   LangGraph StateGraph                   │
│               doc_qa/agent/graph.py                      │
│                                                          │
│   START → [agent] ──tool call──→ [tools]                │
│               ▲                      │                   │
│               │                      ▼                   │
│               └──── [validate_tool_results] ◄────────── │
│                      (retry hint if empty)               │
└──────────┬──────────────────────────┬───────────────────┘
           │ LLM calls                │ tool calls
┌──────────▼──────┐        ┌──────────▼──────────────────┐
│  Anthropic /    │        │  5 Tools                     │
│  OpenAI LLM     │        │  • search_documents          │
│                 │        │  • list_documents            │
└─────────────────┘        │  • summarize_document        │
                           │  • classify_document         │
                           │  • calculate                 │
                           └──────────┬───────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────┐
│                    VectorStore ABC                       │
│               doc_qa/store/base.py                       │
└─────────────────────────────────────┬───────────────────┘
                                      │
              ┌───────────────────────▼────────────────┐
              │         ChromaVectorStore               │
              │         doc_qa/store/chroma.py          │
              │   (swap → SnowflakeCortexVectorStore)   │
              └────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│               Ingestion Pipeline                        │
│                                                        │
│  File  →  parsers.py  →  chunker.py  →  pipeline.py   │
│              │               │               │         │
│           parse_pdf       sliding         embed +      │
│           parse_docx      window          store +      │
│           parse_csv       overlap         MLflow run   │
│           parse_json      dedup.py                     │
│           parse_md        (SHA-256)                    │
│           parse_txt                                    │
└────────────────────────────────────────────────────────┘
```

### Module boundaries

| Layer | Module | Must NOT import |
|---|---|---|
| Configuration | `config/settings.py` | Any `doc_qa` module |
| Storage abstraction | `doc_qa/store/base.py` | Any concrete store |
| Storage implementation | `doc_qa/store/chroma.py` | `agent`, `ingestion`, `ui` |
| Embedding | `doc_qa/embeddings.py` | `store`, `agent`, `ui` |
| Ingestion | `doc_qa/ingestion/` | `agent`, `ui` |
| Agent tools | `doc_qa/agent/tools.py` | `ui`, `ingestion` |
| Agent graph | `doc_qa/agent/graph.py` | `ui`, `ingestion` |
| Agent runner | `doc_qa/agent/runner.py` | `ui` |
| Observability | `doc_qa/observability.py` | `agent`, `ingestion`, `store`, `ui` |
| UI | `ui/app.py` | `graph.py`, `store` directly |

---

## Agent Capabilities

The agent uses a **ReAct loop with validated tool execution**. It can call tools
multiple times before producing a final answer, and it automatically retries
failed searches with broader queries (up to `MAX_TOOL_RETRIES`, default: 2).

### Tools

| Tool | What it does |
|---|---|
| `search_documents(query, top_k)` | Semantic search across all ingested chunks. Returns passages with source citations and similarity scores. |
| `list_documents()` | Lists all ingested files with chunk counts. Useful for corpus navigation. |
| `summarize_document(filename)` | Summarizes the full content of a specific document using all its chunks. |
| `classify_document(filename)` | Identifies the banking document type — Promissory Note, Deed of Trust, HUD-1, TILA Disclosure, Form 1003, Appraisal Report, Closing Disclosure, etc. |
| `calculate(expression)` | Evaluates financial expressions: monthly payment (PMT), APR, amortization, and basic arithmetic. Uses a safe evaluator — not `eval`. |

### Grounding enforcement

The agent is constrained by two layers:

1. **System prompt rules** — five numbered mandatory grounding constraints
   including a required `UNGROUNDED:` prefix format when no evidence is found.
2. **Post-response check** in `runner.py` — verifies that at least one real
   document filename appears in the response before returning it to the UI.

When the grounding check fails, the user receives:

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

---

## Quick Start

### Prerequisites

- Python 3.11+
- An LLM API key: `ANTHROPIC_API_KEY` (default) or `OPENAI_API_KEY`
- For PDF OCR support (scanned documents): `tesseract-ocr` and `poppler-utils`

```bash
# macOS
brew install tesseract poppler

# Ubuntu / Debian
sudo apt-get install tesseract-ocr poppler-utils
```

OCR is optional — if Tesseract is not installed, the PDF parser falls back to
`pypdf` text extraction and logs a warning. Digitally-created PDFs (the majority
of banking documents) do not require OCR.

### Install

```bash
git clone https://github.com/jgalloway42/doc-qa-agent.git
cd doc-qa-agent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the package with dev dependencies
make install
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
# Choose your LLM provider
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Embedding runs locally by default — no key needed
EMBEDDING_PROVIDER=sentence_transformers
```

See [Configuration](#configuration) for all options.

### Ingest the sample corpus

```bash
make ingest
```

This ingests all documents in `docs/` — five banking documents covering loan
applications, lending policy, mortgage FAQ, rate tables, and a TILA disclosure.
Duplicate files are skipped automatically (SHA-256 hash check).

Expected output:

```
✓ loan_application.pdf     — 42 chunks  (1.3s embed)
✓ lending_policy.pdf       — 67 chunks  (2.1s embed)
✓ mortgage_faq.md          — 38 chunks  (1.2s embed)
✓ loan_rates.csv           — 18 chunks  (0.6s embed)
✓ tila_disclosure.txt      — 24 chunks  (0.8s embed)

Total: 189 chunks ingested across 5 documents.
```

### Start the UI

```bash
make ui
```

Open [http://localhost:8501](http://localhost:8501).

### Start MLflow (optional, recommended)

In a second terminal:

```bash
make mlflow
```

Open [http://localhost:5000](http://localhost:5000) to view ingestion runs and
agent session traces.

---

## Configuration

All configuration is managed via environment variables loaded from `.env`.
The full reference is in `.env.example`. Key settings:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `ollama` |
| `ANTHROPIC_API_KEY` | — | Required if `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | — | Required if `LLM_PROVIDER=openai` or `EMBEDDING_PROVIDER=openai` |
| `OLLAMA_MODEL` | `llama3.2` | Model name when `LLM_PROVIDER=ollama` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | `sentence_transformers` (local) or `openai` |
| `SENTENCE_TRANSFORMERS_MODEL` | `all-MiniLM-L6-v2` | Model name for local embeddings |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage directory |
| `CHUNK_SIZE` | `512` | Characters per chunk (approximate) |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `PDF_OCR_THRESHOLD` | `50` | Mean chars/page below which OCR is triggered. Set to `0` to disable OCR fallback. |
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow backend. Swap to `s3://bucket/mlruns` for AWS. |
| `DEFAULT_TOP_K` | `5` | Number of chunks returned per search |
| `MAX_TOOL_RETRIES` | `2` | Max automatic search retries per agent turn |

> **Important:** Once documents are ingested with a given `EMBEDDING_PROVIDER`,
> the corpus cannot be queried with a different provider — the vector spaces are
> incompatible. To switch providers, run `make clean` and re-ingest.

---

## Running with Docker

Docker Compose runs the Streamlit app and a local MLflow tracking server:

```bash
cp .env.example .env
# Edit .env with your API key

docker-compose up
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

Data is persisted in `./chroma_db/` and `./mlruns/` via volume mounts — it
survives container restarts.

To rebuild after code changes:

```bash
docker-compose up --build
```

---

## Development

### Make targets

```bash
make install      # install package + dev dependencies
make lint         # ruff check
make format       # ruff format (in-place)
make type-check   # mypy on doc_qa/
make test         # pytest with 80% coverage gate
make test-cov     # pytest + HTML coverage report (htmlcov/)
make ingest       # ingest docs/ directory
make ui           # start Streamlit
make mlflow       # start MLflow UI on :5000
make clean        # remove chroma_db/, mlruns/, caches
make docker-build # build Docker image
make docker-up    # docker-compose up
```

### Running tests

```bash
make test
```

Tests run against a real in-memory ChromaDB instance (`EphemeralClient`) — not
a mock. LLM calls are mocked with deterministic `AIMessage` responses. The 80%
coverage gate is enforced; `ui/` and `cli/` are excluded from measurement.

```bash
# View HTML coverage report
make test-cov
open htmlcov/index.html
```

### Adding a new document

```bash
# Single file
python -m cli.ingest_cli ingest path/to/document.pdf

# Directory
python -m cli.ingest_cli ingest path/to/docs/

# Force re-ingest (ignore duplicate check)
python -m cli.ingest_cli ingest path/to/document.pdf --force

# List ingested documents
python -m cli.ingest_cli list

# System status
python -m cli.ingest_cli status
```

### Supported file types

| Extension | Parser | Notes |
|---|---|---|
| `.pdf` | `pypdf` + `pytesseract` fallback | Two-pass: digital extraction first, OCR if sparse |
| `.txt` | stdlib | Line-by-line with line number provenance |
| `.md` | stdlib | Line-by-line, heading markers preserved |
| `.csv` | `csv.DictReader` | Each row serialized as JSON |
| `.json` | `json` stdlib | Root list → one entry per element; root dict → one entry per key |
| `.docx` | `python-docx` | Paragraphs + table rows; empty paragraphs skipped |

### Project structure

```
doc-qa-agent/
├── .github/workflows/ci.yml   # GitHub Actions: lint → type-check → test
├── cli/ingest_cli.py           # Typer CLI for ingestion
├── config/settings.py          # Pydantic-settings BaseSettings
├── doc_qa/
│   ├── embeddings.py           # EmbeddingProvider ABC + providers
│   ├── observability.py        # MLflow helpers
│   ├── ingestion/
│   │   ├── parsers.py          # File-type parsers
│   │   ├── chunker.py          # Sliding window chunker
│   │   ├── dedup.py            # SHA-256 duplicate detection
│   │   └── pipeline.py         # Ingestion orchestration
│   ├── store/
│   │   ├── base.py             # VectorStore ABC
│   │   └── chroma.py           # ChromaDB implementation
│   └── agent/
│       ├── tools.py            # 5 LangGraph tools
│       ├── graph.py            # StateGraph + validate_tool_results node
│       └── runner.py           # Session memory + grounding check
├── ui/app.py                   # Streamlit chat interface
├── tests/                      # pytest suite (80% coverage gate)
├── docs/                       # Sample banking document corpus
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Document Corpus

The `docs/` directory contains eight banking documents across six file types. They are
designed to require cross-document retrieval — most interesting questions cannot be
answered from a single document. All fictional documents use **Meridian Bank of
Springfield, N.A.** as the issuing institution. No real PII is present in any document.

### Documents

#### Downloaded from public government sources

| Filename | Type | Source | Document class |
|---|---|---|---|
| `fannie_mae_1003_loan_application.pdf` | PDF | Fannie Mae / FHFA | Uniform Residential Loan Application (Form 1003) — filled sample showing borrower info, employment, assets, liabilities, and loan details across 8 pages |
| `cfpb_closing_disclosure.pdf` | PDF | CFPB (consumerfinance.gov) | Closing Disclosure (TRID) — 5-page completed form showing final loan terms, projected payments, itemized closing costs, and cash to close |
| `cfpb_loan_estimate.pdf` | PDF | CFPB (consumerfinance.gov) | Loan Estimate (TRID) — 3-page completed form showing estimated rate, payment, and closing costs; pairs with the Closing Disclosure for comparison questions |

To download these files:
```bash
# Fannie Mae Form 1003 (filled sample)
curl -L "https://singlefamily.fanniemae.com/media/7991/display" -o docs/fannie_mae_1003_loan_application.pdf

# CFPB Closing Disclosure (filled sample)
curl -L "https://files.consumerfinance.gov/f/201311_cfpb_kbyo_closing-disclosure.pdf" -o docs/cfpb_closing_disclosure.pdf

# CFPB Loan Estimate (filled sample)
curl -L "https://files.consumerfinance.gov/f/201311_cfpb_kbyo_loan-estimate.pdf" -o docs/cfpb_loan_estimate.pdf
```

#### Created for this project (fictional, no real PII)

| Filename | Type | Contents |
|---|---|---|
| `underwriting_guidelines.docx` | DOCX | 9-section internal bank policy covering borrower eligibility, credit score minimums by product (see table below), DTI limits, LTV maximums, PMI rules, reserve requirements, the exception approval hierarchy, and appraisal requirements |
| `mortgage_products_faq.md` | Markdown | 30 Q&A pairs across 7 topic areas: mortgage types, loan types (conforming/jumbo/FHA/VA/USDA), qualification criteria, PMI, escrow, closing costs, and refinancing |
| `mortgage_rate_sheet.csv` | CSV | 18 rows of current product rates with columns: `product`, `loan_type`, `term_months`, `rate_pct`, `apr_pct`, `points`, `min_credit_score`, `max_ltv_pct`, `max_dti_pct`, `requires_pmi_above_80_ltv`, `effective_date`, `notes` |
| `tila_disclosure_statement.txt` | TXT | Complete Truth in Lending Act disclosure for a fictional $285,000 30-year fixed mortgage — APR, finance charge, amount financed, total of payments, payment schedule, late charge terms, prepayment clause, and assumption clause |
| `loan_products_catalog.json` | JSON | Array of 10 loan product objects with full qualification data per product: credit score minimums, max LTV/DTI, PMI rules, MIP/funding fee details, eligible property types, eligible occupancy types, and reserve requirements |

### Cross-document query map

The corpus is deliberately structured so that the most useful questions require
retrieving from multiple documents. The table below shows which documents each
query type touches — this is what makes the demo compelling.

| Question type | Documents required | Tools called |
|---|---|---|
| "Would a borrower with a 660 credit score and 85% LTV qualify for a conventional loan, and what rate would they get?" | `underwriting_guidelines.docx` + `mortgage_rate_sheet.csv` | `search_documents` (×2) |
| "How did the closing costs change between the loan estimate and the closing disclosure?" | `cfpb_loan_estimate.pdf` + `cfpb_closing_disclosure.pdf` | `search_documents` (×2) |
| "What is the monthly payment on the loan in the closing disclosure, and how much total interest will be paid?" | `cfpb_closing_disclosure.pdf` | `search_documents` + `calculate` |
| "What type of document is the Form 1003 and what sections does it contain?" | `fannie_mae_1003_loan_application.pdf` | `classify_document` + `summarize_document` |
| "Does an FHA loan require PMI, and what are the DTI limits if I have strong compensating factors?" | `loan_products_catalog.json` + `underwriting_guidelines.docx` + `mortgage_products_faq.md` | `search_documents` (×3) |
| "What is the APR on the TILA disclosure, and what would the monthly payment be on a $300,000 loan at the same rate?" | `tila_disclosure_statement.txt` | `search_documents` + `calculate` |

---

## Sample Q&A

The following examples demonstrate multi-step agent reasoning across the sample
corpus. Each shows which tools the agent calls.

---

**Q: What credit score is required for a 30-year fixed mortgage, and what rate would I get today?**

> Tools called: `search_documents` (lending_policy.pdf) → `search_documents` (loan_rates.csv)

> A: According to the First National Bank of Plainview Lending Policy
> (lending_policy.pdf, page 2), the minimum credit score for a conforming
> 30-year fixed mortgage is 620. Borrowers with scores below 680 are subject
> to additional compensating factor requirements. Per the current rate sheet
> (loan_rates.csv, effective 2024-01-15), the 30-year fixed rate is 7.125%
> with an APR of 7.284% and 0.5 points.

---

**Q: What type of document is tila_disclosure.txt?**

> Tools called: `classify_document` (tila_disclosure.txt)

> A: `tila_disclosure.txt` is a **Truth in Lending Act (TILA) Disclosure
> Statement**, also known as a Reg Z disclosure. This is a federally mandated
> form under 15 U.S.C. § 1601 that discloses the APR, finance charge, amount
> financed, total of payments, and payment schedule to the borrower prior to
> loan consummation. The document's structure — with labeled fields for APR,
> Finance Charge, Amount Financed, and Total of Payments — is consistent with
> the standard TILA disclosure format.

---

**Q: What would the monthly payment be on a $250,000 loan at 6.5% for 30 years?**

> Tools called: `calculate` (PMT formula)

> A: Using the standard annuity formula PMT(r, n, PV):
> - Monthly rate: 6.5% / 12 = 0.5417%
> - Term: 360 payments
> - Principal: $250,000
>
> **Monthly payment: $1,580.17**
>
> Total of payments over 30 years: $568,861.20
> Total interest paid: $318,861.20

---

**Q: What about for 15 years instead?** *(follow-up — no loan amount repeated)*

> Tools called: `calculate` (PMT formula, reusing context from previous turn)

> A: For the same $250,000 at 6.5% over 15 years (180 payments):
>
> **Monthly payment: $2,178.25**
>
> Total of payments: $392,085.00
> Total interest paid: $142,085.00 — a saving of $176,776.20 versus the
> 30-year term, at a cost of $598.08 more per month.

---

**Q: Summarize the mortgage FAQ document.**

> Tools called: `list_documents` → `summarize_document` (mortgage_faq.md)

> A: The mortgage FAQ (mortgage_faq.md) covers five main areas:
>
> **Loan types** — fixed vs. adjustable-rate mortgages, including how ARM
> adjustment caps work and when each product is appropriate.
>
> **Costs** — breakdown of closing costs, points, PMI requirements (triggered
> below 80% LTV), and how to calculate when PMI cancellation applies.
>
> **Process** — pre-approval vs. pre-qualification, what documents are required,
> typical timeline from application to close.
>
> **Refinancing** — break-even calculation for refinancing decisions, cash-out
> vs. rate-and-term refinancing, and prepayment penalty considerations.
>
> **Escrow** — how escrow accounts work, what they cover (taxes and insurance),
> and how escrow analysis adjustments are calculated annually.

---

## Observability

MLflow tracks two categories of runs under the `doc-qa-agent` experiment:

### Ingestion runs

Each file ingestion creates one MLflow run with:

| Signal | Type | Example |
|---|---|---|
| `chunks_created` | Metric | `42` |
| `embedding_time_s` | Metric | `1.34` |
| `total_time_s` | Metric | `1.61` |
| `skipped` | Param | `False` (not duplicate) |
| `filename` | Param | `loan_application.pdf` |
| `file_hash` | Param | `a3f2...` |
| `embedding_provider` | Param | `sentence_transformers` |
| `chunk_size` | Param | `512` |
| `chunk_overlap` | Param | `64` |

### Agent session runs

Each conversation session creates one MLflow run. Per-turn metrics are logged
with `step=turn_index`:

| Signal | Type | What it tells you |
|---|---|---|
| `latency_s` | Metric (per step) | Agent response time per turn |
| `grounded` | Metric (per step) | `1.0` = grounded, `0.0` = ungrounded response |
| `retrieval_score_mean` | Metric (per step) | Mean cosine similarity of top-k results |
| `retrieval_score_max` | Metric (per step) | Best matching chunk score |
| `retrieval_score_min` | Metric (per step) | Weakest matching chunk score |
| `tool_calls` | Param (per step) | e.g., `search_documents,calculate` |
| `user_message` | Param (per step) | Truncated to 500 chars |

The `grounded` metric is queryable as a time-series across sessions — use it
to track grounding rate over time and detect regressions when new documents
are added to the corpus.

---

## Known Limitations

**Grounding check is heuristic-based.** The post-response grounding check
verifies that at least one real document filename appears in the response.
It does not verify that the cited content is accurate or that specific
page numbers are correct. A response that includes a filename in a fabricated
citation would pass the check. A semantic grounding verifier (LLM-as-judge or
embedding similarity) is the correct v2 solution.

**Embedding provider is fixed at ingestion time.** The vector space produced
by `sentence-transformers` and OpenAI embeddings is incompatible. Switching
`EMBEDDING_PROVIDER` after ingesting documents requires clearing the vector
store (`make clean`) and re-ingesting the full corpus.

**Synchronous ingestion blocks the UI.** Ingesting a large document through
the Streamlit uploader blocks the UI thread during processing. For documents
larger than ~50 pages, use the CLI (`python -m cli.ingest_cli ingest`) instead.

**No authentication.** The Streamlit UI has no login or session isolation.
All users share the same document corpus. Authentication is required before
any production deployment.

**Single-process ChromaDB.** ChromaDB's `PersistentClient` is not safe for
concurrent writes from multiple processes. Multi-worker deployments require
ChromaDB's HTTP server mode or a Snowflake Cortex / pgvector migration.

---

## Production Migration

### Migrating to Snowflake Cortex

The `VectorStore` abstract base class (`doc_qa/store/base.py`) is the migration
seam. Implementing Snowflake Cortex support requires one new file:

```python
# doc_qa/store/snowflake_cortex.py
from doc_qa.store.base import VectorStore, Chunk, SearchResult

class SnowflakeCortexVectorStore(VectorStore):
    def __init__(self, connection, table_name: str): ...

    def add_chunks(self, chunks, embeddings): ...
    # → INSERT INTO {table_name} (id, text, filename, ..., embedding)
    #   VALUES (..., ARRAY_CONSTRUCT(...))

    def search(self, query_embedding, top_k=5):  ...
    # → SELECT *, VECTOR_COSINE_SIMILARITY(embedding, ARRAY_CONSTRUCT(...))
    #   AS score FROM {table_name} ORDER BY score DESC LIMIT {top_k}

    def get_chunk(self, chunk_id): ...
    def list_documents(self): ...
    def get_chunks_for_document(self, filename): ...
    def document_exists(self, filename): ...
    def count(self): ...
```

Then set the store in `pipeline.py` and `runner.py`:

```python
store = SnowflakeCortexVectorStore(snowflake_connection, table_name="doc_qa_chunks")
```

No changes to the agent, ingestion pipeline, tools, or tests are required.

### Migrating MLflow to AWS S3

Zero code changes required. Update one environment variable:

```bash
MLFLOW_TRACKING_URI=s3://your-bucket/mlruns
```

Ensure the application's IAM role has `s3:GetObject`, `s3:PutObject`, and
`s3:ListBucket` on the target bucket. The MLflow tracking server (or sidecar)
needs the same permissions.

### Recommended production additions

| Capability | Approach |
|---|---|
| Authentication | OAuth2/SAML SSO in front of Streamlit, or replace Streamlit with FastAPI + React |
| Multi-user sessions | Replace in-memory session dict with Redis or PostgreSQL session store |
| Async ingestion | Wrap `ingest_file` in Celery or AWS SQS for background processing |
| Re-ranking | Add `cross-encoder/ms-marco-MiniLM-L-6-v2` as a re-ranking pass after retrieval |
| Semantic grounding | LLM-as-judge grounding verification (opt-in via `STRICT_GROUNDING=true`) |
| SBOM generation | Add `pip-licenses` to CI, output `THIRD_PARTY_LICENSES.md` as artifact |
| Concurrent writes | Migrate to ChromaDB HTTP server mode or Snowflake Cortex |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

All runtime dependencies are MIT, Apache 2.0, or BSD 3-Clause licensed.
No GPL or LGPL dependencies. Full license inventory:

| Package | License |
|---|---|
| langchain / langgraph | MIT |
| chromadb | Apache 2.0 |
| mlflow | Apache 2.0 |
| streamlit | Apache 2.0 |
| sentence-transformers | Apache 2.0 |
| pypdf | BSD 3-Clause |
| pdf2image / pytesseract | MIT / Apache 2.0 |
| Tesseract OCR | Apache 2.0 |
| python-docx | MIT |
| pydantic / pydantic-settings | MIT |
| simpleeval | MIT |
| typer / ruff / pytest | MIT |
